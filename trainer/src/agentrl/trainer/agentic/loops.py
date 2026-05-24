import asyncio
import json
import traceback
from typing import Awaitable, Callable, Any

import torch

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser
try:
    from sglang.srt.entrypoints.openai.protocol import Tool
except ImportError:
    from sglang.srt.openai_api.protocol import Tool
from transformers import PreTrainedTokenizerBase

from ..utils import to_plasma

SessionIdType = int
StarFnType = Callable[[int], Awaitable[dict]]
GenFnType = Callable[..., Awaitable]
ObsFnType = Callable[[Any, SessionIdType], Awaitable[dict]]
EndFnType = Callable[[int, bool], Awaitable]


def collect_metrics(src, tgt):
    for k, v in src.items():
        if k == "score":
            continue
        if k not in tgt:
            tgt[k] = v
        else:
            tgt[k] += v


async def openai_chat_agent_loop(
    start_args: dict,
    start_fn: StarFnType,
    gen_fn: GenFnType,
    obs_fn: ObsFnType,
    end_fn: EndFnType,
    max_turns: int,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    tool_call_parser: str,
    incomplete_punishment: float,
    content_as_parts: bool = False,
    **_
) -> dict:
    done = False
    reward = 0
    status = ""
    obs_metrics = {}

    def _wrap_content(text_value: str):
        # VL processors (e.g. Qwen3VLProcessor) require message["content"] to be a
        # list of typed parts; text-only tokenizers accept a plain string. The
        # caller flips `content_as_parts` for VL-backed runs.
        if content_as_parts:
            return [{"type": "text", "text": text_value}]
        return text_value

    def _maybe_load_json(raw: str):
        if not isinstance(raw, str):
            return raw
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw

    def _flat_ids(result):
        # VL processors return a BatchFeature/dict; text tokenizers return a
        # flat list[int]. Normalize to list[int].
        if isinstance(result, list):
            if result and isinstance(result[0], list):
                return result[0]
            return result
        if hasattr(result, "input_ids"):
            ids_obj = result.input_ids
        elif isinstance(result, dict) and "input_ids" in result:
            ids_obj = result["input_ids"]
        else:
            ids_obj = result
        if hasattr(ids_obj, "tolist"):
            ids_obj = ids_obj.tolist()
        if ids_obj and isinstance(ids_obj[0], list):
            return ids_obj[0]
        return ids_obj

    # start
    start = await start_fn(**start_args)
    history = start.pop("messages")
    tools = start.pop("tools")
    sid = start.pop("sid")
    collect_metrics(start.get("metrics", {}), obs_metrics)

    ids = await asyncio.to_thread(
        tokenizer.apply_chat_template,
        history,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
    )
    ids = _flat_ids(ids)
    loss_mask = [0] * len(ids)
    log_probs = [0] * len(ids)

    # interact
    for turn in range(max_turns):
        text, received_log_probs = await gen_fn(input_ids=ids)
        new_ids = [t[1] for t in received_log_probs]
        new_log_probs = [t[0] for t in received_log_probs]
        ids += new_ids
        loss_mask += [1] * len(new_ids)
        log_probs += new_log_probs

        message: dict[str, str | list] = {
            "role": "assistant",
        }
        if tools:
            parser = FunctionCallParser(
                tools=[Tool.model_validate(tool) for tool in tools],
                tool_call_parser=tool_call_parser,
            )
            try:
                normal_text, info_list = parser.parse_non_stream(text)
            except:
                normal_text = text
                info_list = []
            message["content"] = _wrap_content(normal_text)
            message["tool_calls"] = [{
                "id": str(info.tool_index),
                "function": {
                    "name": info.name,
                    # FunctionCallParser returns parameters as a JSON string; some
                    # chat templates (Qwen3.5) iterate arguments as a mapping, so
                    # decode here and fall back to the raw string on parse error.
                    "arguments": _maybe_load_json(info.parameters),
                }
            } for info in info_list]
        else:
            message["content"] = _wrap_content(text)

        history.append(message)

        obs = await obs_fn(message, sid)
        # possible injection here
        messages = obs.pop("messages")
        # use diff as new ids
        last = await asyncio.to_thread(
            tokenizer.apply_chat_template,
            history,
            tools=tools,
            tokenize=True,
            )
        last = _flat_ids(last)
        history.extend(messages)
        now = await asyncio.to_thread(
            tokenizer.apply_chat_template,
            history,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
        )
        now = _flat_ids(now)
        diff = now[len(last):]
        ids += diff
        loss_mask += [0] * len(diff)
        log_probs += [0] * len(diff)

        done = obs.pop("finish")
        reward = obs.pop("reward")
        status = obs.pop("status")
        collect_metrics(obs.get("metrics", {}), obs_metrics)

        if done or len(ids) >= max_length:
            break

    obs_metrics["pass_rate"] = int(reward == 1)

    if status != "completed":
        reward = incomplete_punishment

    await end_fn(sid, done)

    assert len(ids) == len(loss_mask) == len(log_probs), f"{len(ids)=}, {len(loss_mask)=}, {len(log_probs)=}"

    return to_plasma({
        "seq_len": len(ids[:max_length]),
        "loss_tokens": sum(loss_mask[:max_length]),
        "input_ids": torch.tensor([ids[:max_length]]),
        "loss_mask": torch.tensor([loss_mask[:max_length]]),
        "position_ids": torch.arange(0, min(max_length, len(ids))).unsqueeze(0),
        "rollout_log_prob": torch.tensor([log_probs[:max_length]]),
        "reward": reward,
        "token_level_rewards": torch.tensor([[reward]], dtype=torch.float32),
        "metrics": obs_metrics,
        "history": history,
    })


async def retry_openai_chat_agent_loop(
    start_args: dict,
    start_fn: StarFnType,
    gen_fn: GenFnType,
    obs_fn: ObsFnType,
    end_fn: EndFnType,
    max_turns: int,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    tool_call_parser: str,
    incomplete_punishment: float = 0,
    max_retries: int = 10,
    content_as_parts: bool = False,
    **_
) -> dict | None:
    for i in range(max_retries):
        try:
            return await openai_chat_agent_loop(
                start_args,
                start_fn,
                gen_fn,
                obs_fn,
                end_fn,
                max_turns,
                max_length,
                tokenizer,
                tool_call_parser,
                incomplete_punishment,
                content_as_parts=content_as_parts,
            )
        except RuntimeError:
            return None
        except Exception:
            traceback.print_exc()
            print(f"nodedup Retrying openai_chat_agent_loop... {start_args=}")
            await asyncio.sleep(1)
    print(f"nodedup Failed to run openai_chat_agent_loop after retries! {start_args=}")
    return None

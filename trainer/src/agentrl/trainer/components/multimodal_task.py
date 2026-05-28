"""
multimodal_task.py  -> trainer/src/agentrl/trainer/components/

Approach A: the rollout uses the VALIDATED Qwen35VLAgent message construction
(via RLMessageBuilder), so the prompt distribution is identical to inference
(same folding image_max/fold_size, same <tool_response> wrapping, same system/
instruction prompts, same smart_resize). RL only:
  - tokenizes those messages ONCE with the processor (apply_chat_template,
    enable_thinking matching inference) -> prompt_input_ids
  - generates via AsyncSglangWorkerMM.generate_with_ids -> output_ids preserved
  - records the response back into the agent so history evolves exactly as in
    inference
  - emits one per-turn training item, with prompt_input_ids + output_ids stored
    verbatim (tokenize-once -> no retokenization drift)

position_ids are filled on the trainer side (get_rope_index); see async_trainer.
"""

from __future__ import annotations

import uuid

import torch

from agentrl.trainer.components.rl_agent_adapter import (
    RLQwen35VLAgent, add_agent_to_path,
)


def build_env_from_item(item, task_config):
    """Open an OSWorld session via the controller API (async)."""
    from agentrl.trainer.components.osworld_env_adapter import OSWorldSession
    return OSWorldSession(
        url=task_config["base_url"],
        name=item.get("name", task_config.get("train_tasks", ["osworld"])[0]),
        index=item.get("index", 0),
    )


def _make_agent(task_config):
    """Construct the validated agent with the SAME knobs as inference.

    image_max / fold_size / collapse_text / enable_thinking come from config so
    rollout == inference. Defaults mirror the pasted agent (20 / 10 / off).
    """
    agent_kwargs = dict(
        model=task_config.get("model", "qwen35-vl"),
        max_tokens=task_config.get("max_tokens", 32768),
        top_p=task_config.get("top_p", 0.9),
        temperature=task_config.get("temperature", 0.0),
        history_n=task_config.get("history_n", 100),
        coordinate_type=task_config.get("coordinate_type", "relative"),
        image_max=task_config.get("image_max", 20),
        fold_size=task_config.get("fold_size", 10),
        collapse_text=task_config.get("collapse_text"),
        enable_thinking=task_config.get("enable_thinking", False),
    )
    repo_dir = task_config.get("agent_repo_dir")
    if repo_dir:
        add_agent_to_path(repo_dir)  # safety for Ray workers w/o PYTHONPATH
    agent = RLQwen35VLAgent(**agent_kwargs)
    agent.reset()
    return agent


def _encode(processor, messages, live_images, enable_thinking):
    """Tokenize messages ONCE. enable_thinking matches inference's chat template
    kwarg so the tokenization is identical."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    kwargs = dict(text=[text], return_tensors="pt")
    if live_images:
        kwargs["images"] = live_images
    enc = processor(**kwargs)
    return enc


async def _rollout_trajectory(item, processor, task_config, gen_fn):
    instruction = item["instruction"]
    enable_thinking = task_config.get("enable_thinking", False)

    agent = _make_agent(task_config)
    env = build_env_from_item(item, task_config)

    records = []
    obs = await env.reset()
    done, info, turn = False, {}, 0
    max_turns = task_config.get("max_turns", 30)

    while not done and turn < max_turns:
        # 1) build messages via the adapter (screenshot append + folding +
        #    message construction, identical to inference predict()'s first half)
        messages, live_images, _pw, _ph = agent.build_messages_only(instruction, obs)

        # 2) tokenize ONCE (processor); enable_thinking matches inference
        enc = _encode(processor, messages, live_images, enable_thinking)
        prompt_ids = enc["input_ids"][0].tolist()

        # 3) generate via sglang, preserving output ids
        import base64, io
        image_data = []
        for im in live_images:
            buf = io.BytesIO(); im.save(buf, format="PNG")
            image_data.append(base64.b64encode(buf.getvalue()).decode())
        _text, output_ids, rollout_lp = await gen_fn(
            input_ids=prompt_ids,
            image_data=image_data if image_data else None,
        )

        # 4) decode response text and feed it back so agent history evolves
        response_text = processor.tokenizer.decode(output_ids, skip_special_tokens=False)
        agent.record_response(response_text, obs)

        records.append({
            "prompt_ids": prompt_ids,
            "output_ids": output_ids,
            "rollout_lp": rollout_lp,
            "pixel_values": enc.get("pixel_values"),
            "image_grid_thw": enc.get("image_grid_thw"),
            "turn_index": turn,
        })

        # 5) build the assistant message the OSWorld worker expects (it parses
        #    tool_calls to drive pyautogui), mirroring openai_chat_agent_loop.
        assistant_message = _build_assistant_message(
            response_text, task_config, content_as_parts=task_config.get("content_as_parts", True))

        obs, reward, done, info = await env.step(assistant_message)
        records[-1]["reward"] = float(reward)
        turn += 1

    await env.end(done)
    success = env.is_success(info) if turn > 0 else False
    return records, success


def _build_assistant_message(text, task_config, content_as_parts=True):
    """Construct the OpenAI-style assistant message with tool_calls, the same way
    inference's openai_chat_agent_loop does (FunctionCallParser). The OSWorld
    task worker parses these tool_calls to execute pyautogui."""
    def _wrap(t):
        return [{"type": "text", "text": t}] if content_as_parts else t
    try:
        try:
            from sglang.srt.function_call.function_call_parser import FunctionCallParser
        except ImportError:
            from sglang.srt.function_call_parser import FunctionCallParser
        try:
            from sglang.srt.entrypoints.openai.protocol import Tool
        except ImportError:
            from sglang.srt.openai_api.protocol import Tool
        import json as _json
        tools = task_config.get("_tools")  # optionally cache tools; else None
        parser = FunctionCallParser(
            tools=[Tool.model_validate(t) for t in tools] if tools else [],
            tool_call_parser=task_config.get("tool_call_parser", "qwen3_coder"),
        )
        normal_text, info_list = parser.parse_non_stream(text)
        msg = {"role": "assistant", "content": _wrap(normal_text)}
        if info_list:
            def _ld(p):
                try:
                    return _json.loads(p)
                except Exception:
                    return p
            msg["tool_calls"] = [{
                "id": str(i.tool_index),
                "function": {"name": i.name, "arguments": _ld(i.parameters)},
            } for i in info_list]
        return msg
    except Exception:
        # fallback: send raw text; worker may still parse <tool_call> from text
        return {"role": "assistant", "content": _wrap(text)}


def _to_item(rec, *, trajectory_id, group_id, success, data_source,
             num_turns, traj_reward):
    prompt_ids = rec["prompt_ids"]
    output_ids = rec["output_ids"]
    full = prompt_ids + output_ids
    seq_len = len(full)

    input_ids = torch.tensor(full, dtype=torch.long).unsqueeze(0)
    loss_mask = torch.tensor(
        [0] * len(prompt_ids) + [1] * len(output_ids), dtype=torch.long).unsqueeze(0)
    tlr = torch.zeros((1, seq_len), dtype=torch.float32)
    tlr[0, -1] = float(traj_reward)
    rlp = rec["rollout_lp"] or [float("nan")] * len(output_ids)
    rollout_lp = torch.tensor(
        [float("nan")] * len(prompt_ids) + list(rlp), dtype=torch.float32).unsqueeze(0)

    mm = {}
    if rec.get("pixel_values") is not None:
        mm["pixel_values"] = rec["pixel_values"]
        mm["image_grid_thw"] = rec["image_grid_thw"]

    return {
        "uid": uuid.uuid4().hex,
        "group_id": group_id,
        "trajectory_id": trajectory_id,
        "turn_index": rec["turn_index"],
        "num_turns": num_turns,
        "data_source": data_source,
        "input_ids": input_ids,
        "seq_len": seq_len,
        "loss_mask": loss_mask,
        "loss_tokens": int(loss_mask.sum().item()),
        "token_level_rewards": tlr,
        "multi_modal_inputs": mm,
        "rollout_log_prob": rollout_lp,
        "reward": float(rec["reward"]),
        "success": success,
    }


async def multimodal_chat_task(item, *, config, tokenizer, gen_fn):
    """task_fn returning a LIST of per-turn items (manager handles list)."""
    processor = tokenizer
    task_config = config

    records, success = await _rollout_trajectory(item, processor, task_config, gen_fn)
    if not records:
        return None

    trajectory_id = uuid.uuid4().hex[:12]
    group_id = str(item.get("group_id", item.get("instruction_id", trajectory_id)))
    data_source = item.get("data_source", task_config.get("name", "mm"))
    success_reward = task_config.get("success_reward", 1.0)
    fail_reward = task_config.get("fail_reward", 0.0)
    traj_reward = success_reward if success else fail_reward

    return [
        _to_item(rec, trajectory_id=trajectory_id, group_id=group_id,
                 success=success, data_source=data_source,
                 num_turns=len(records), traj_reward=traj_reward)
        for rec in records
    ]
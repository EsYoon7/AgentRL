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


def _load_instruction(item, task_config):
    """Load instruction from the task config JSON, exactly like inference does:
       {test_config_base_dir}/examples/{domain}/{example_id}.json -> ["instruction"]
    item provides index (==example_id) and name (==domain, unless overridden by
    `domain_field`). Falls back to item["instruction"] or env-provided one.
    """
    import json, os
    base = task_config.get("test_config_base_dir")
    if base is None:
        return item.get("instruction", "")
    domain = item.get("domain") or item.get("name")
    example_id = item.get("index")
    path = os.path.join(base, "examples", str(domain), f"{example_id}.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            example = json.load(f)
        return example["instruction"]
    except Exception as e:
        print(f"[multimodal_task] could not load instruction from {path}: {e}")
        return item.get("instruction", "")


def build_env_from_item(item, task_config):
    """Open an OSWorld session via the controller API (async)."""
    from agentrl.trainer.components.osworld_env_adapter import OSWorldSession
    return OSWorldSession(
        url=task_config["base_url"],
        name=item.get("name", task_config.get("train_tasks", ["osworld"])[0]),
        index=item.get("index", 0),
        start_timeout_s=task_config.get("start_timeout_s", 600),
        start_poll_s=task_config.get("start_poll_s", 10),
        save_screenshots_dir=task_config.get("save_screenshots_dir"),
        traj_id=str(item.get("index", 0)),
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


def _encode_text_and_images(processor, messages, live_images, enable_thinking):
    """Build the TEXT string for SGLang (SGLang tokenizes + expands image
    placeholders, avoiding the double-process IndexError), AND build the
    training-side tensors from the SAME processor call: input_ids (our prompt
    ids) + pixel_values + image_grid_thw.

    We use the processor's input_ids as prompt_ids instead of asking SGLang for
    them (which requires logprob_start_len=0 -> computes full-prompt logits ->
    OOM on long OSWorld prompts). SGLang and this processor share the same
    transformers processor, so the tokenizations match; the Layer-6 logprob
    check is the safety net for any drift."""
    # Qwen3.5 chat template reads `enable_thinking` directly: when False it emits
    # an empty `<think>\n\n</think>` block (thinking off). apply_chat_template
    # forwards **kwargs into the template, so pass enable_thinking as a direct
    # keyword. Wrapping it in chat_template_kwargs={...} does NOT work here (that
    # dict is only unwrapped by servers like vLLM, not by a direct
    # apply_chat_template call) -- it gets silently ignored and thinking stays on.
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # very old transformers that only accept chat_template_kwargs
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )
    mm = {}
    enc_kwargs = dict(text=[text], return_tensors="pt")
    if live_images:
        enc_kwargs["images"] = live_images
    enc = processor(**enc_kwargs)
    prompt_ids = enc["input_ids"][0].tolist()
    if live_images and "pixel_values" in enc:
        mm["pixel_values"] = enc["pixel_values"]
        mm["image_grid_thw"] = enc["image_grid_thw"]
    return text, prompt_ids, mm


async def _rollout_trajectory(item, processor, task_config, gen_fn):
    enable_thinking = task_config.get("enable_thinking", False)

    agent = _make_agent(task_config)
    env = build_env_from_item(item, task_config)

    # Instruction source = SAME as inference: load the task JSON directly and
    # read example["instruction"]. dataset item has {index, name}; inference uses
    # {test_config_base_dir}/examples/{domain}/{example_id}.json with
    # example_id == index. We mirror that exactly so RL == inference.
    instruction = _load_instruction(item, task_config)

    # Optional debug trajectory dump (a): human-readable per-turn log of what the
    # model saw (messages w/ screenshots stripped), what it generated, the parsed
    # action, and the reward. Enabled via task.save_trajectory_dir.
    traj_log = []
    save_traj_dir = task_config.get("save_trajectory_dir")

    import time as _time
    t_traj_start = _time.time()
    records = []
    t_reset_start = _time.time()
    obs = await env.reset()
    t_reset = _time.time() - t_reset_start
    print(f"[timing] reset took {t_reset:.1f}s (name={item.get('name')} "
          f"index={item.get('index')})", flush=True)
    done, info, turn = False, {}, 0
    max_turns = task_config.get("max_turns", 30)
    t_gen_total = 0.0
    t_step_total = 0.0

    while not done and turn < max_turns:
        # 1) build messages via the adapter (screenshot append + folding +
        #    message construction, identical to inference predict()'s first half)
        messages, live_images, _pw, _ph = agent.build_messages_only(instruction, obs)

        # 2) build TEXT (for SGLang) + our prompt_ids + training image tensors
        text, prompt_ids, mm = _encode_text_and_images(
            processor, messages, live_images, enable_thinking)

        # 3) generate via sglang: send TEXT + image_data (NOT input_ids), so
        #    SGLang tokenizes + expands placeholders without the double-process
        #    IndexError. We do NOT request full prompt logprobs (that OOMs), so
        #    we use our processor's prompt_ids above; sglang's returned prompt
        #    ids (if any) are only for optional drift checks.
        import base64, io
        image_data = []
        for im in live_images:
            buf = io.BytesIO(); im.save(buf, format="PNG")
            image_data.append(base64.b64encode(buf.getvalue()).decode())
        _t_gen = _time.time()
        _text, output_ids, rollout_lp, _sglang_prompt_ids = await gen_fn(
            prompt=text,
            image_data=image_data if image_data else None,
        )
        t_gen_total += _time.time() - _t_gen

        # 4) decode response text and feed it back so agent history evolves
        response_text = processor.tokenizer.decode(output_ids, skip_special_tokens=False)
        low_level, pyautogui_code = agent.record_response(response_text, obs)

        records.append({
            "prompt_ids": prompt_ids,           # SGLang-tokenized prompt
            "output_ids": output_ids,
            "rollout_lp": rollout_lp,
            "pixel_values": mm.get("pixel_values"),
            "image_grid_thw": mm.get("image_grid_thw"),
            "turn_index": turn,
        })

        # 5) build the assistant message the OSWorld worker expects (it parses
        #    tool_calls to drive pyautogui), mirroring openai_chat_agent_loop.
        assistant_message = _build_assistant_message(
            response_text, task_config, content_as_parts=task_config.get("content_as_parts", True))

        _t_step = _time.time()
        obs, reward, done, info = await env.step(assistant_message)
        t_step_total += _time.time() - _t_step
        records[-1]["reward"] = float(reward)

        if save_traj_dir is not None:
            traj_log.append({
                "turn": turn,
                "messages": _strip_images_for_log(messages),
                "num_live_images": len(live_images),
                "prompt_len": len(prompt_ids),
                "response_text": response_text,
                "low_level_instruction": low_level,
                "pyautogui_code": pyautogui_code,
                "reward": float(reward),
                "done": bool(done),
                "status": info.get("status"),
                "gen_s": round(t_gen_total, 2),
                "env_step_s": round(t_step_total, 2),
            })
        turn += 1

    await env.end(done)
    success = env.is_success(info) if turn > 0 else False

    t_traj_total = _time.time() - t_traj_start
    print(f"[timing] trajectory done: total={t_traj_total:.1f}s "
          f"reset={t_reset:.1f}s gen={t_gen_total:.1f}s "
          f"env_step={t_step_total:.1f}s turns={turn} "
          f"success={success} (name={item.get('name')} index={item.get('index')})",
          flush=True)

    if save_traj_dir is not None:
        _write_trajectory_dump(
            save_traj_dir, item, instruction, traj_log, success, info)

    return records, success


def _strip_images_for_log(messages):
    """Replace base64 image payloads with a short marker so the dump stays
    readable and small. Handles both part formats this agent may emit:
      {"type": "image_url", "image_url": {"url": "data:image/..."}}
      {"type": "image",     "url": "data:image/..."}
    and, defensively, any string value that looks like a base64 image."""
    def _shorten(url):
        if isinstance(url, str) and url:
            return url[:32] + "...<omitted>"
        return url

    def _clean_part(p):
        if not isinstance(p, dict):
            return p
        t = p.get("type")
        if t == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            return {"type": "image_url", "image_url": {"url": _shorten(url)}}
        if t == "image":
            # url may be under "url" or "image"
            if "url" in p:
                return {"type": "image", "url": _shorten(p.get("url"))}
            if "image" in p:
                return {"type": "image", "image": _shorten(p.get("image"))}
            return {"type": "image", "_omitted": True}
        # any other part: scrub long data:image strings in its values
        cleaned = {}
        for k, v in p.items():
            if isinstance(v, str) and v.startswith("data:image"):
                cleaned[k] = _shorten(v)
            else:
                cleaned[k] = v
        return cleaned

    out = []
    for m in messages:
        content = m.get("content", [])
        if isinstance(content, str):
            out.append({"role": m.get("role"), "content": content})
            continue
        parts = [_clean_part(p) for p in content]
        out.append({"role": m.get("role"), "content": parts})
    return out


def _write_trajectory_dump(save_dir, item, instruction, traj_log, success, info):
    import json, os
    name = str(item.get("name", "task"))
    traj_id = str(item.get("index", 0))
    out_dir = os.path.join(save_dir, name, traj_id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "trajectory.jsonl")
    try:
        with open(path, "w", encoding="utf-8") as f:
            # header line: task-level metadata
            f.write(json.dumps({
                "event": "meta", "name": name, "index": traj_id,
                "instruction": instruction, "success": bool(success),
                "num_turns": len(traj_log),
                "final_status": info.get("status"),
                "final_reward": info.get("reward"),
            }, ensure_ascii=False) + "\n")
            for entry in traj_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[trajectory] saved {path} ({len(traj_log)} turns, success={success})")
    except Exception as e:
        print(f"[trajectory] failed to save {path}: {e}")


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
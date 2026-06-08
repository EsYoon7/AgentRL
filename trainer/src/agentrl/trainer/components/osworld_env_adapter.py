"""
osworld_env_adapter.py  -> trainer/src/agentrl/trainer/components/

Bridges the AgentRL controller/session API (the same one openai_chat_start/obs/
end use) into the simple env interface our multimodal_task expects:

    env.reset()              -> obs   with obs["screenshot"] = raw PNG bytes
    env.step(pyautogui_code) -> (obs, reward, done, info)
    env.is_success(info)     -> bool

How OSWorld speaks (from agentic/tasks.py):
  * POST {url}/start_sample {index, name}  -> {messages, tools, sid, metrics}
  * POST {url}/interact     {messages:[msg]} hdr{session_id} -> {messages, finish,
                                                  reward, status, metrics}
  * POST {url}/cancel       hdr{session_id}

The screenshot arrives INSIDE the returned `messages` as an image_url data URL
(base64 PNG). We extract the most recent image and hand raw bytes to the agent.

The action we send back must be a proper assistant tool-call message, because
the OSWorld task worker parses tool calls to drive pyautogui. We reuse the
agent's RAW response text (which already contains <tool_call>...), not the
parsed pyautogui_code -- the worker does its own parsing. So step() takes the
assistant's raw response text.
"""

from __future__ import annotations

import asyncio
import base64
import random
import traceback

import aiohttp


def _extract_latest_screenshot(messages) -> bytes | None:
    """Find the last image_url (base64 PNG) in OpenAI-style messages -> bytes."""
    latest = None
    for m in messages:
        content = m.get("content", [])
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if url.startswith("data:image"):
                        b64 = url.split(",", 1)[1]
                        latest = base64.b64decode(b64)
    return latest


def _extract_text(messages) -> str:
    """Concatenate text parts (instruction / tool_response text) for the agent's
    obs text, if the agent needs it. Our agent mainly needs the screenshot; the
    instruction comes from the dataset item, so this is auxiliary."""
    chunks = []
    for m in messages:
        content = m.get("content", [])
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(part.get("text", ""))
    return "\n".join(chunks)



def _extract_instruction(messages) -> str:
    """Pull the task instruction from the controller's first messages.

    OSWorld/AgentBench typically place the instruction in the first user message
    text. We take the text of the first user-role message; adjust if your
    controller formats it differently (inspect raw_messages once to confirm).
    """
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", [])
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text"]
                if texts:
                    return "\n".join(texts)
    # fallback: system message or empty
    return ""


class OSWorldSession:
    """Synchronous-looking wrapper over the async controller API. We run the
    async HTTP calls on the running event loop via asyncio.

    NOTE: multimodal_task currently calls env.reset()/step() synchronously inside
    an async task. To keep it simple we expose async methods and have
    multimodal_task await them (see the small change in multimodal_task).
    """

    def __init__(self, url, name, index, start_timeout_s=600, start_poll_s=10,
                 save_screenshots_dir=None, traj_id=None):
        self.url = url
        self.name = name
        self.index = index
        self.sid = None
        self._last_info = {}
        # VM boot can take >80s; wait up to start_timeout_s, polling every
        # start_poll_s, before giving up on /start_sample.
        self.start_timeout_s = start_timeout_s
        self.start_poll_s = start_poll_s
        # Optional: save each turn's screenshot to disk for debugging (mirrors
        # inference). save_screenshots_dir=None disables it.
        self.save_screenshots_dir = save_screenshots_dir
        self.traj_id = traj_id or str(index)
        self._save_idx = 0
        if self.save_screenshots_dir:
            import os
            self._shot_dir = os.path.join(
                self.save_screenshots_dir, str(self.name), str(self.traj_id))
            os.makedirs(self._shot_dir, exist_ok=True)

    def _save_screenshot(self, screenshot_bytes, tag):
        """Write a screenshot to disk if saving is enabled. No-op otherwise."""
        if not self.save_screenshots_dir or screenshot_bytes is None:
            return
        import os
        path = os.path.join(self._shot_dir, f"{self._save_idx:03d}_{tag}.png")
        try:
            with open(path, "wb") as f:
                f.write(screenshot_bytes)
        except Exception as e:
            print(f"[OSWorldSession] failed to save screenshot {path}: {e}")
        self._save_idx += 1

    async def reset(self):
        # Normalize index exactly like inference openai_chat_start does.
        import torch as _torch
        if isinstance(self.index, _torch.Tensor):
            self.index = self.index.item()
        await asyncio.sleep(random.randint(0, 3))  # avoid peak (mirrors start_fn)
        ret = None
        # VM boot/reset can take well over a minute (observed ~84s). We must
        # outlast that: retry for up to start_timeout_s, polling every
        # start_poll_s. A 400 here usually means "VM not ready yet", so we keep
        # retrying until the deadline rather than giving up after a few tries.
        import time as _time
        start_timeout_s = self.start_timeout_s
        start_poll_s = self.start_poll_s
        deadline = _time.time() + start_timeout_s
        attempt = 0
        while True:
            attempt += 1
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=600), trust_env=True) as s:
                    payload = {"index": self.index, "name": self.name}
                    async with s.post(self.url + "/start_sample", json=payload) as r:
                        body = await r.text()
                        if r.status != 200:
                            print(f"[OSWorldSession] /start_sample {r.status} "
                                  f"attempt={attempt} payload={payload!r} "
                                  f"body={body[:300]}")
                        r.raise_for_status()
                        import json as _json
                        ret = _json.loads(body)
                        self.sid = r.headers["session_id"]
                        break
            except Exception:
                if _time.time() >= deadline:
                    print(f"[OSWorldSession] /start_sample giving up after "
                          f"{attempt} attempts / {start_timeout_s}s")
                    traceback.print_exc()
                    raise
                await asyncio.sleep(start_poll_s)
                continue
        messages = ret["messages"]
        screenshot = _extract_latest_screenshot(messages)
        self._save_screenshot(screenshot, "reset")
        # OSWorld puts the task instruction in the controller's first messages
        # (not in the dataset item). Extract it so the agent can build prompts.
        instruction = _extract_instruction(messages)
        self.instruction = instruction
        return {"screenshot": screenshot, "text": _extract_text(messages),
                "instruction": instruction, "raw_messages": messages}

    async def step(self, assistant_message: dict):
        """assistant_message: the OpenAI-style assistant message (with tool_calls)
        that the OSWorld worker will parse to drive pyautogui."""
        payload = {"messages": [assistant_message]}
        header = {"session_id": str(self.sid)}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600), trust_env=True) as s:
            async with s.post(self.url + "/interact", json=payload, headers=header) as r:
                assert r.status == 200, f"bad status {r.status}: {await r.text()}"
                ret = await r.json()
        messages = ret.get("messages", [])
        screenshot = _extract_latest_screenshot(messages)
        self._save_screenshot(screenshot, "step")
        done = ret.get("finish", False)
        reward = ret.get("reward", 0.0)
        status = ret.get("status", "")
        info = {"status": status, "reward": reward, "metrics": ret.get("metrics", {})}
        self._last_info = info
        obs = {"screenshot": screenshot, "text": _extract_text(messages),
               "raw_messages": messages}
        return obs, float(reward), bool(done), info

    async def end(self, done):
        if done:
            return
        header = {"session_id": self.sid}
        try:
            async with aiohttp.ClientSession(trust_env=True) as s:
                async with s.post(self.url + "/cancel", headers=header, json={}):
                    pass
        except Exception as e:
            print(f"end/cancel failed: {e}")

    def is_success(self, info):
        return float(info.get("reward", 0.0)) == 1.0
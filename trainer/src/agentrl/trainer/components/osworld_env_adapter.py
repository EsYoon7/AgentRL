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


class OSWorldSession:
    """Synchronous-looking wrapper over the async controller API. We run the
    async HTTP calls on the running event loop via asyncio.

    NOTE: multimodal_task currently calls env.reset()/step() synchronously inside
    an async task. To keep it simple we expose async methods and have
    multimodal_task await them (see the small change in multimodal_task).
    """

    def __init__(self, url, name, index):
        self.url = url
        self.name = name
        self.index = index
        self.sid = None
        self._last_info = {}

    async def reset(self):
        if isinstance(self.index, object) and hasattr(self.index, "item"):
            self.index = self.index.item()
        await asyncio.sleep(random.randint(0, 3))  # avoid peak (mirrors start_fn)
        ret = None
        for attempt in range(5):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=600), trust_env=True) as s:
                    async with s.post(self.url + "/start_sample",
                                      json={"index": self.index, "name": self.name}) as r:
                        ret = await r.json()
                        r.raise_for_status()
                        self.sid = r.headers["session_id"]
                        break
            except Exception:
                traceback.print_exc()
                if attempt < 4:
                    await asyncio.sleep(random.randint(1, 10)); continue
                raise
        messages = ret["messages"]
        screenshot = _extract_latest_screenshot(messages)
        return {"screenshot": screenshot, "text": _extract_text(messages),
                "raw_messages": messages}

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
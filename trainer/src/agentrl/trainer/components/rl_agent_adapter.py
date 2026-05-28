"""
rl_agent_adapter.py  -> trainer/src/agentrl/trainer/components/

RL adapter for the OSWorld Qwen35VLAgent. The original inference agent is left
UNTOUCHED; we import it (add its repo to sys.path at init) and SUBCLASS it,
adding only `build_messages_only` for RL rollout.

`build_messages_only` is a COPY of predict()'s message-construction half (it
duplicates that logic on purpose -- the user accepted the duplication to avoid
editing the original). It performs the same state updates predict() does
(append screenshot, update folding, build messages) but STOPS before the network
call / parsing, returning (messages, live_image_pils, processed_w, processed_h).

>>> SYNC NOTE <<<
This duplicates message construction from Qwen35VLAgent.predict(). If the
original's folding / <tool_response> wrapping / prompts / process_image change,
update build_messages_only to match. The Layer-6 logprob check is the safety net.

PATH SETUP: call add_agent_to_path(<repo_dir>) once at process init, or set
PYTHONPATH, before importing this module's Qwen35VLAgent.
"""

from __future__ import annotations

import base64
import json
import sys
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

from PIL import Image


def add_agent_to_path(repo_dir: str):
    """Add the inference-agent submodule dir to sys.path. Call once at init."""
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


# Lazy base-class resolution: the original agent is imported the first time
# RLQwen35VLAgent is instantiated, AFTER add_agent_to_path() has run. This avoids
# an import error at module-load time when the submodule isn't on the path yet.
_BASE = None


def _load_base():
    global _BASE
    if _BASE is None:
        from mm_agents.qwen35_agent import Qwen35VLAgent
        _BASE = Qwen35VLAgent
    return _BASE


class RLQwen35VLAgent:
    """Subclass-by-composition wrapper that adds build_messages_only.

    We can't statically subclass Qwen35VLAgent (it may not be importable at
    module load). Instead we dynamically create the subclass on first use.
    Use RLQwen35VLAgent(**kwargs) -> returns an instance of the real subclass.
    """

    def __new__(cls, *args, **kwargs):
        base = _load_base()
        # build the real subclass dynamically, attaching our extra methods
        subclass = type("RLQwen35VLAgentImpl", (base,), {
            "build_messages_only": _build_messages_only,
            "record_response": _record_response,
        })
        return subclass(*args, **kwargs)


def _build_messages_only(
    self, instruction: str, obs: Dict
) -> Tuple[List[Dict], List[Image.Image], int, int]:
    """Replicates predict() up to (not including) the network call.

    Performs the SAME state mutations as predict():
      - process current screenshot, append to self.screenshots
      - update folding state
      - build messages (system + history with folding + current obs)
    Returns (messages, live_images, processed_w, processed_h).

    After RL generates and you have the response text, call
    record_response(response_text, obs) to update history exactly like
    predict()'s tail does.
    """
    # ---- copied from predict() head ----
    # process_image is a module-level function in the original agent module;
    # reuse it directly so smart_resize params stay identical to inference.
    from mm_agents.qwen35_agent import process_image

    screenshot_bytes = obs["screenshot"]
    processed_b64 = process_image(screenshot_bytes)
    processed_img = Image.open(BytesIO(base64.b64decode(processed_b64)))
    processed_width, processed_height = processed_img.size

    self.screenshots.append(processed_b64)
    total_steps = len(self.screenshots)
    self._update_folding_state(total_steps)

    start_step = max(1, total_steps - self.history_n)

    previous_actions = [
        f"Step {i + 1}: {self.actions[i]}"
        for i in range(0, min(start_step - 1, len(self.actions)))
    ]
    previous_actions_str = "\n".join(previous_actions) if previous_actions else "None"

    description_prompt_lines = [
        "Use a mouse and keyboard to interact with a computer, and take screenshots.",
        "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.",
        "* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.",
        (f"* The screen's resolution is {processed_width}x{processed_height}."
         if self.coordinate_type == "absolute"
         else "* The screen's resolution is 1000x1000."),
        "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.",
        "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.",
        "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    ]
    description_prompt = "\n".join(description_prompt_lines)

    action_description_prompt = """
* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.
* `type`: Type a string of text on the keyboard.
* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.
* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys (e.g., "ctrl", "shift", "ctrl+shift") that will be held during the click.
* `left_click_drag`: Click and drag the cursor to a specified (x, y) coordinate.
* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen. Optional `text` parameter can specify modifier keys that will be held during the click.
* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen (simulated as double-click since it's the closest action). Optional `text` parameter can specify modifier keys that will be held during the click.
* `scroll`: Performs a scroll of the mouse scroll wheel. Optional `text` parameter can specify a modifier key (e.g., "shift", "ctrl") that will be held during scrolling.
* `hscroll`: Performs a horizontal scroll (mapped to regular scroll). Optional `text` parameter can specify a modifier key that will be held during scrolling.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
* `answer`: Answer a question."""

    tools_def = {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": description_prompt,
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {"type": "string", "description": action_description_prompt,
                               "enum": ["key", "type", "mouse_move", "left_click",
                                        "left_click_drag", "right_click", "middle_click",
                                        "double_click", "triple_click", "scroll",
                                        "hscroll", "wait", "terminate", "answer"]},
                    "keys": {"type": "array", "description": "Required only by `action=key`."},
                    "text": {"type": "string", "description": "Required by `action=type` and `action=answer`. Optional for click actions (left_click, right_click, middle_click, double_click, triple_click) to specify modifier keys (e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions (scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') to hold during scrolling."},
                    "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                    "pixels": {"type": "number", "description": "Scroll amount."},
                    "time": {"type": "number", "description": "Seconds to wait."},
                    "status": {"type": "string", "description": "Task status for terminate.",
                               "enum": ["success", "failure"]},
                },
            },
        },
    }

    system_prompt = (
        "You are a multi-purpose intelligent assistant. Based on my requests, you can use tools to help me complete various tasks.\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        "<tools>\n" + json.dumps(tools_def) + "\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
        "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n</function>\n</tool_call>\n\n"
        "<IMPORTANT>\nReminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
        f"- The current date is {datetime.today().strftime('%A, %B %d, %Y')}.\n"
        f"- Collapsed screenshots appear as text: {self.collapse_text}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "Response format for every step:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "Rules:\n"
        "- Output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything else outside those parts.\n"
        "- If finishing, use action=terminate in the tool call."
    )

    instruction_prompt = (
        f"\nPlease generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Previous actions:\n{previous_actions_str}"
    )

    messages: List[Dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]
    live_images: List[Image.Image] = []

    for step_num in range(start_step, total_steps + 1):
        is_first_turn = step_num == start_step
        is_collapsed = self._should_collapse_step(step_num)

        if is_collapsed:
            parts = [{"type": "text", "text": self.collapse_text}]
            if is_first_turn:
                user_content = [{"type": "text", "text": instruction_prompt}]
            else:
                user_content = self._wrap_tool_response(parts)
            messages.append({"role": "user", "content": user_content})
        else:
            b64 = self.screenshots[step_num - 1]
            img_url = f"data:image/png;base64,{b64}"
            if is_first_turn:
                user_content = [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": instruction_prompt},
                ]
            else:
                user_content = self._wrap_tool_response(
                    [{"type": "image_url", "image_url": {"url": img_url}}])
            messages.append({"role": "user", "content": user_content})
            live_images.append(
                Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

        if step_num <= total_steps - 1 and (step_num - 1) < len(self.responses):
            prior_response = self.responses[step_num - 1] or ""
            import re
            cur_idx = step_num - 1
            last_prior_idx = total_steps - 2
            mode = self.thinking_history_mode
            if mode == "keep_all":
                keep_thinking = True
            elif mode == "keep_last_only":
                keep_thinking = (cur_idx == last_prior_idx)
            elif mode == "keep_recent_n":
                keep_thinking = (cur_idx >= last_prior_idx - (self.thinking_history_keep_recent - 1))
            else:
                keep_thinking = False
            if not keep_thinking:
                prior_response = re.sub(
                    r"<think>.*?</think>\s*", "", prior_response, flags=re.DOTALL).strip()
            messages.append({"role": "assistant",
                             "content": [{"type": "text", "text": prior_response}]})

    return messages, live_images, processed_width, processed_height


def _record_response(self, response_text: str, obs: Dict):
    """Reproduce predict()'s tail: append response, parse, append action."""
    self.responses.append(response_text or "")
    original_img = Image.open(BytesIO(obs["screenshot"]))
    ow, oh = original_img.size
    proc_img = Image.open(BytesIO(base64.b64decode(self.screenshots[-1])))
    pw, ph = proc_img.size
    low_level, pyautogui_code = self.parse_response(
        response_text or "", original_width=ow, original_height=oh,
        processed_width=pw, processed_height=ph)
    self.actions.append(low_level)
    return low_level, pyautogui_code
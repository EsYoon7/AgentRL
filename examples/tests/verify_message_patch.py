"""
Verify our duplicated build_messages_only produces the SAME messages as the
original inference agent's predict() message construction. CPU-only, no model.

Strategy: feed the SAME obs sequence to BOTH:
  (a) the original Qwen35VLAgent.predict() -- but intercept call_llm so it
      returns a fixed dummy response and we capture the messages it built.
  (b) our RLQwen35VLAgent.build_messages_only().
Then diff the captured messages. They must be identical (the image base64 is
the same because both call the same process_image).

Run with the agent repo on PYTHONPATH:
  PYTHONPATH=/path/to/agent/repo python verify_messages_match.py
"""

import base64
import io
import json
import sys

from PIL import Image


def _fake_obs(color):
    img = Image.new("RGB", (1280, 720), color)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return {"screenshot": buf.getvalue()}


def _sanitize(messages):
    """Replace base64 image urls with a short hash so diffs are readable but
    still detect image-content changes."""
    import hashlib
    out = []
    for m in messages:
        c = {"role": m.get("role"), "content": []}
        content = m.get("content", [])
        if isinstance(content, str):
            c["content"] = content
        else:
            for p in content:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    h = hashlib.sha1(url.encode()).hexdigest()[:12]
                    c["content"].append({"type": "image_url", "sha": h})
                else:
                    c["content"].append(p)
        out.append(c)
    return out


def main():
    from mm_agents.qwen35_agent import Qwen35VLAgent
    from agentrl.trainer.components.rl_agent_adapter import RLQwen35VLAgent

    instruction = "Open the file manager and create a new folder."
    obs_seq = [_fake_obs((100, 100, 100)),
               _fake_obs((120, 120, 120)),
               _fake_obs((140, 140, 140))]

    kw = dict(image_max=20, fold_size=10, enable_thinking=False)

    # (a) original via predict() with intercepted call_llm
    orig = Qwen35VLAgent(**kw); orig.reset()
    orig_msgs_per_turn = []
    def fake_call(payload, model):
        orig_msgs_per_turn.append(payload["messages"])
        return ("Action: click\n<tool_call>\n<function=computer_use>\n"
                "<parameter=action>\nleft_click\n</parameter>\n"
                "<parameter=coordinate>\n[100, 100]\n</parameter>\n"
                "</function>\n</tool_call>")
    orig.call_llm = fake_call
    for obs in obs_seq:
        orig.predict(instruction, obs)

    # (b) ours via build_messages_only + record_response (same dummy response)
    ours = RLQwen35VLAgent(**kw); ours.reset()
    ours_msgs_per_turn = []
    for obs in obs_seq:
        msgs, live, pw, ph = ours.build_messages_only(instruction, obs)
        ours_msgs_per_turn.append(msgs)
        ours.record_response(
            ("Action: click\n<tool_call>\n<function=computer_use>\n"
             "<parameter=action>\nleft_click\n</parameter>\n"
             "<parameter=coordinate>\n[100, 100]\n</parameter>\n"
             "</function>\n</tool_call>"), obs)

    # compare
    all_match = True
    for i, (a, b) in enumerate(zip(orig_msgs_per_turn, ours_msgs_per_turn)):
        sa, sb = _sanitize(a), _sanitize(b)
        if sa != sb:
            all_match = False
            print(f"=== TURN {i}: MISMATCH ===")
            print("ORIG:", json.dumps(sa, ensure_ascii=False)[:500])
            print("OURS:", json.dumps(sb, ensure_ascii=False)[:500])
        else:
            print(f"turn {i}: match ({len(a)} messages)")
    print("\nRESULT:", "ALL MATCH" if all_match else "MISMATCH FOUND")


if __name__ == "__main__":
    main()
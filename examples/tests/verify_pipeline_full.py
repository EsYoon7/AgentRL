"""
Full multimodal pipeline verification, layered. Run in order; stop at first fail.

  stageA  build_messages_only matches inference          (CPU)        [DONE for you]
  stageB  one full turn: messages -> tokenize -> (mock) gen -> record (CPU, mock sglang)
  stageC  per-turn item shape matches FSDP collate/ppo_loss expectations (CPU)
  stageD  trajectory -> list[item], tracking + reward correctness        (CPU)
  stageE  [GPU] real sglang generate + get_rope_index positions + logprob match

Stages B-D mock sglang generation (no GPU) and the env, so we can validate the
ITEM CONSTRUCTION and TRACKING logic cheaply. Stage E is the GPU money-check.

Run:
  PYTHONPATH=/path/to/agent/repo python verify_pipeline_full.py B
  ... C, D ... then E on a GPU node.
"""

import asyncio
import sys

import torch


# ---- a fake env + fake gen_fn so stages B-D need no GPU / no OSWorld ----
class FakeEnv:
    def __init__(self, n_turns=3):
        self.n = n_turns; self.i = 0
        import io
        from PIL import Image
        def shot(c):
            b = io.BytesIO(); Image.new("RGB", (1280, 720), (c, c, c)).save(b, "PNG")
            return b.getvalue()
        self._shots = [shot(100 + 10 * k) for k in range(n_turns + 1)]

    def reset(self):
        self.i = 0
        return {"screenshot": self._shots[0]}

    def step(self, pyautogui_code):
        self.i += 1
        done = self.i >= self.n
        reward = 1.0 if done else 0.0   # success at the end
        return {"screenshot": self._shots[self.i]}, reward, done, {"ok": True}

    def is_success(self, info):
        return True


_FAKE_RESPONSE_TEXT = (
    "Action: click the menu\n<tool_call>\n<function=computer_use>\n"
    "<parameter=action>\nleft_click\n</parameter>\n"
    "<parameter=coordinate>\n[100, 100]\n</parameter>\n"
    "</function>\n</tool_call>"
)


def make_fake_gen(processor):
    """gen_fn that returns a REAL tokenized tool_call response, so
    record_response/parse_response behave like production."""
    async def fake_gen_fn(input_ids=None, image_data=None, **kw):
        out = processor.tokenizer(
            _FAKE_RESPONSE_TEXT, add_special_tokens=False)["input_ids"]
        lp = [-0.1] * len(out)
        return _FAKE_RESPONSE_TEXT, out, lp
    return fake_gen_fn


async def fake_gen_fn(input_ids=None, image_data=None, **kw):
    """Fallback (no processor): tiny fixed ids."""
    out = [1, 2, 3, 4]
    lp = [-0.1, -0.2, -0.3, -0.4]
    return "dummy", out, lp


def _patch_task_for_mock(task_mod, env):
    """Inject FakeEnv into multimodal_task.build_env_from_item."""
    task_mod.build_env_from_item = lambda item, cfg: env


def stageB(processor, task_config):
    """One turn end-to-end with mock gen, real processor + real agent."""
    import agentrl.trainer.components.multimodal_task as T
    env = FakeEnv(n_turns=1)
    _patch_task_for_mock(T, env)
    item = {"instruction": "Open the file manager.", "name": "t", "index": 0}
    recs, success = asyncio.run(
        T._rollout_trajectory(item, processor, task_config, make_fake_gen(processor)))
    print("=== stage B ===")
    print("num turn-records:", len(recs))
    r = recs[0]
    print("prompt_ids len:", len(r["prompt_ids"]),
          "output_ids:", r["output_ids"],
          "has pixel_values:", r.get("pixel_values") is not None)
    assert len(r["prompt_ids"]) > 0 and r["output_ids"], "empty rollout record"
    print("stage B OK")


def stageC(processor, task_config):
    """Item shape vs FSDP collate / ppo_loss expectations."""
    import agentrl.trainer.components.multimodal_task as T
    env = FakeEnv(n_turns=1)
    _patch_task_for_mock(T, env)
    item = {"instruction": "x", "name": "t", "index": 0}
    items = asyncio.run(T.multimodal_chat_task(
        item, config=task_config, tokenizer=processor, gen_fn=make_fake_gen(processor)))
    it = items[0]
    print("=== stage C ===")
    for k in ["uid", "group_id", "trajectory_id", "turn_index",
              "input_ids", "seq_len", "loss_mask", "loss_tokens",
              "token_level_rewards", "multi_modal_inputs", "rollout_log_prob"]:
        assert k in it, f"missing key {k}"
    assert it["input_ids"].shape[1] == it["seq_len"], "seq_len mismatch"
    assert it["loss_mask"].shape == it["input_ids"].shape
    assert it["loss_mask"][0, 0] == 0, "first token must have no loss (ppo asserts)"
    assert it["token_level_rewards"].shape == it["input_ids"].shape
    print("keys/shapes OK; seq_len =", it["seq_len"],
          "loss_tokens =", it["loss_tokens"])
    print("stage C OK")


def stageD(processor, task_config):
    """Multi-turn -> per-turn items; tracking + shared success reward."""
    import agentrl.trainer.components.multimodal_task as T
    env = FakeEnv(n_turns=3)
    _patch_task_for_mock(T, env)
    item = {"instruction": "x", "name": "t", "index": 0, "group_id": "g1"}
    items = asyncio.run(T.multimodal_chat_task(
        item, config=task_config, tokenizer=processor, gen_fn=make_fake_gen(processor)))
    print("=== stage D ===")
    print("num items (turns):", len(items))
    tids = {it["trajectory_id"] for it in items}
    gids = {it["group_id"] for it in items}
    turns = [it["turn_index"] for it in items]
    rewards = [float(it["token_level_rewards"].sum()) for it in items]
    print("trajectory_ids:", tids, "(must be 1)")
    print("group_ids:", gids)
    print("turn_indices:", turns)
    print("per-turn summed reward:", rewards, "(all == success reward)")
    assert len(tids) == 1, "turns must share one trajectory_id"
    assert turns == list(range(len(items))), "turn_index must be 0..n-1"
    assert len(set(rewards)) == 1, "all turns must carry the SAME success reward"
    print("stage D OK")


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "B"
    from transformers import AutoProcessor
    MODEL_PATH = "/mnt/home/justiwag/esyoon/models/Qwen3.5-9B"
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    task_config = {
        "image_max": 20, "fold_size": 10, "enable_thinking": False,
        "max_turns": 15, "success_reward": 1.0, "fail_reward": 0.0,
        "name": "osworld_smoke",
        # agent_repo_dir not needed if PYTHONPATH already set
    }

    if stage == "B": stageB(processor, task_config)
    elif stage == "C": stageC(processor, task_config)
    elif stage == "D": stageD(processor, task_config)
    else: print("stages: B | C | D  (E is the GPU stage in verify_stage1_v2.py)")


if __name__ == "__main__":
    main()
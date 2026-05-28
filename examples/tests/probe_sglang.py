"""
STEP 0 (run on a GPU node with the model loaded): probe the SGLang result dict.

Goal: find out exactly where output token ids live in YOUR sglang version, so
we keep only the correct branch in extract_generation().

Run this as a tiny standalone, OR paste the probe block into your existing
worker.generate temporarily. The standalone version mirrors how AsyncSglangWorker
builds the engine.
"""

import asyncio
import json


def _summarize(obj, depth=0, max_list=3):
    """Print structure (keys/types/lengths) WITHOUT dumping huge tensors/ids."""
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__}", end="")
                if isinstance(v, list):
                    print(f" (len={len(v)})")
                else:
                    print()
                _summarize(v, depth + 1, max_list)
            else:
                s = repr(v)
                if len(s) > 80:
                    s = s[:80] + "..."
                print(f"{pad}{k}: {type(v).__name__} = {s}")
    elif isinstance(obj, list):
        print(f"{pad}[list len={len(obj)}], showing up to {max_list}:")
        for i, v in enumerate(obj[:max_list]):
            print(f"{pad}  [{i}]: {type(v).__name__} = "
                  f"{repr(v)[:80]}")


async def probe(engine, prompt_ids=None, prompt_text="Hello, world."):
    """Call async_generate once and dump the structure."""
    kwargs = {}
    if prompt_ids is not None:
        kwargs["input_ids"] = prompt_ids        # preferred: id-in path
    else:
        kwargs["text"] = prompt_text
    kwargs["sampling_params"] = {"max_new_tokens": 8, "temperature": 0.0}

    ret = await engine.async_generate(return_logprob=True, **kwargs)
    if isinstance(ret, list):
        print(f"[ret is a list of len {len(ret)}; using ret[0]]")
        ret = ret[0]

    print("\n===== TOP-LEVEL KEYS =====")
    print(list(ret.keys()))

    print("\n===== meta_info KEYS =====")
    print(list(ret["meta_info"].keys()))

    print("\n===== STRUCTURE =====")
    _summarize(ret)

    # the specific checks we care about
    print("\n===== DECISIONS =====")
    print("has top-level 'output_ids':", "output_ids" in ret)
    otlp = ret["meta_info"].get("output_token_logprobs")
    print("has meta_info['output_token_logprobs']:", otlp is not None)
    if otlp:
        print("  first entry:", otlp[0], "(expect (logprob, token_id, ...))")
        print("  -> token ids extractable from index [1]:",
              all(isinstance(x[1], int) for x in otlp[:3]))
    # input ids echoed back?
    print("has 'input_ids' echoed:", "input_ids" in ret)
    print("meta has 'prompt_tokens'/'completion_tokens':",
          [k for k in ret["meta_info"] if "token" in k.lower()])

    return ret


# ---------------------------------------------------------------------------
# Standalone runner (adapt model_path / server_args to your config)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sglang as sgl

    MODEL_PATH = "/mnt/home/justiwag/esyoon/models/Qwen3.5-9B"  # <-- your model

    engine = sgl.Engine(model_path=MODEL_PATH, dtype="bfloat16", tp_size=1)
    engine.tokenizer_manager.auto_create_handle_loop()

    # text probe first (simplest), then ids probe
    asyncio.run(probe(engine, prompt_text="Hello, world."))
    # asyncio.run(probe(engine, prompt_ids=[9707, 11, 1879, 13]))
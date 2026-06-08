"""
Probe: does SGLang return prompt input_ids (so we can use method B)?

Method B = send {text, image_data} to SGLang, let it tokenize + expand image
placeholders, then read BOTH prompt_ids and output_ids back. This avoids the
double-processing IndexError entirely.

For this to work, async_generate(return_logprob=True, ...) must expose prompt
token ids somewhere in ret['meta_info']. This probe dumps the structure so we
know which field to read.

Run on a GPU node:
  python probe_sglang_prompt_ids.py
"""

import asyncio
import io
import base64

from PIL import Image

MODEL_PATH = "/mnt/home/justiwag/esyoon/models/Qwen3.5-9B"


async def main():
    import sglang as sgl
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # build a tiny multimodal prompt as TEXT (not pre-expanded ids)
    img = Image.new("RGB", (256, 256), (120, 120, 120))
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe in one word."},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    engine = sgl.Engine(model_path=MODEL_PATH, dtype="bfloat16", tp_size=1)

    # METHOD B call: text + image_data (let SGLang do everything)
    ret = await engine.async_generate(
        prompt=text,                      # TEXT, not input_ids
        image_data=[img_b64],
        sampling_params={"max_new_tokens": 4, "temperature": 0.0},
        return_logprob=True,
        logprob_start_len=0,              # <-- request FULL prompt logprobs
    )
    if isinstance(ret, list):
        ret = ret[0]

    print("=== ret top-level keys ===")
    print(list(ret.keys()))
    print("\n=== meta_info keys ===")
    meta = ret["meta_info"]
    print(list(meta.keys()))

    # look for prompt ids under common names
    print("\n=== searching for prompt token ids ===")
    for k in meta.keys():
        if "prompt" in k.lower() or "input" in k.lower():
            v = meta[k]
            print(f"  meta['{k}'] type={type(v)} "
                  f"sample={str(v)[:120]}")
    # input_token_logprobs often carries prompt ids as (logprob, token_id, ...)
    itl = meta.get("input_token_logprobs")
    pt = meta.get("prompt_tokens")
    if itl is not None:
        print(f"\n  prompt_tokens={pt}  input_token_logprobs len={len(itl)}")
        print(f"  MATCH={len(itl)==pt} (need len == prompt_tokens for method B)")
        print(f"  first 3={itl[:3]}")
        print("  -> prompt ids = [t[1] for t in input_token_logprobs]")
    otl = meta.get("output_token_logprobs")
    if otl is not None:
        print(f"  output_token_logprobs len={len(otl)} first={otl[:3]}")

    print("\n=== text out ===", repr(ret.get("text"))[:100])


if __name__ == "__main__":
    asyncio.run(main())
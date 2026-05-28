"""
Diagnose the image-count mismatch. Run on the GPU node.

The error `image_grid_thw[index] index 1 out of bounds for size 1` means the
processor counted MORE image markers in input_ids than the number of images in
image_data. We inspect:
  - how many <|vision_start|> appear in our prompt_ids
  - how many <|image_pad|> appear
  - what SGLang/Qwen3-VL expects: usually ONE <|image_pad|> per image in the
    input_ids (the processor EXPANDS it to grid size), NOT n_ph copies.

This tells us whether we should put 1 placeholder (let processor expand) or
n_ph placeholders (pre-expanded) when passing input_ids to sglang.
"""

import base64, io
from PIL import Image
from transformers import AutoProcessor

MODEL_PATH = "/mnt/home/justiwag/esyoon/models/Qwen3.5-9B"
VISION_START = 248053
VISION_END = 248054
IMAGE_PAD = 248056

proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Build the SAME conversation via the official chat template + processor, and
# see how many image_pad tokens IT produces. That is ground truth for what
# input_ids should look like.
img = Image.new("RGB", (224, 224), (120, 120, 120))

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe the image in one word."},
    ],
}]
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== chat template text (first 300 chars) ===")
print(text[:300])

out = proc(text=[text], images=[img], return_tensors="pt")
ids = out["input_ids"][0].tolist()

n_vstart = ids.count(VISION_START)
n_vend = ids.count(VISION_END)
n_pad = ids.count(IMAGE_PAD)
print("\n=== official processor output ===")
print(f"input_ids len   : {len(ids)}")
print(f"<|vision_start|> : {n_vstart}")
print(f"<|vision_end|>   : {n_vend}")
print(f"<|image_pad|>    : {n_pad}")
print(f"image_grid_thw   : {out['image_grid_thw'].tolist()}")
gt = out["image_grid_thw"][0]
merge = proc.image_processor.merge_size
print(f"expected placeholders = prod(grid)/merge^2 = "
      f"{int(gt.prod()) // (merge*merge)}")
print()
print(">>> CONCLUSION:")
print(f"    input_ids should contain {n_pad} copies of <|image_pad|> for ONE image.")
print(f"    If that equals prod(grid)/merge^2, the processor PRE-EXPANDS and we")
print(f"    must match it. If it's 1, sglang expands it for us -> send only 1.")
"""
extract_model_constants.py

Pull the vision/assistant special-token ids straight from the processor so you
never hardcode integers. Run this once against your model and paste the printed
dict into your task config.

Why this works: Qwen-VL family registers named special tokens, e.g.
  <|vision_start|>, <|vision_end|>, <|image_pad|>, <|im_start|>, <|im_end|>
convert_tokens_to_ids() maps the name -> id deterministically for that
checkpoint. We also derive the assistant header/eot id blocks by tokenizing the
chat template's role markers, so they match exactly what the model trained with.
"""

from __future__ import annotations

import json


# Candidate names per model family. Add yours if different.
VISION_TOKEN_CANDIDATES = {
    "vision_start": ["<|vision_start|>", "<|vision_bos|>"],
    "vision_end":   ["<|vision_end|>", "<|vision_eos|>"],
    "image_pad":    ["<|image_pad|>", "<|image|>", "<image>"],
}
IM_START_CANDIDATES = ["<|im_start|>"]
IM_END_CANDIDATES = ["<|im_end|>"]


def _first_valid_id(tokenizer, names):
    """Return (name, id) for the first candidate that maps to a real id."""
    unk = tokenizer.unk_token_id
    for name in names:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and tid != unk and tid >= 0:
            return name, tid
    return None, None


def extract_constants(model_path: str) -> dict:
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tok = proc.tokenizer

    # ---- 1) vision tokens -------------------------------------------------
    vs_name, vs_id = _first_valid_id(tok, VISION_TOKEN_CANDIDATES["vision_start"])
    ve_name, ve_id = _first_valid_id(tok, VISION_TOKEN_CANDIDATES["vision_end"])
    pad_name, pad_id = _first_valid_id(tok, VISION_TOKEN_CANDIDATES["image_pad"])

    # Some processors expose these directly; prefer the processor attribute if present.
    pad_id = getattr(proc, "image_token_id", None) or pad_id
    # Qwen processors sometimes store it on the model config instead; if None,
    # fall back to the name-based lookup above.

    # ---- 2) assistant header / eot blocks --------------------------------
    # Derive from the chat template so they match training exactly. We render a
    # tiny 2-message conversation and diff to isolate the assistant header.
    ims_name, ims_id = _first_valid_id(tok, IM_START_CANDIDATES)
    ime_name, ime_id = _first_valid_id(tok, IM_END_CANDIDATES)

    # assistant header: the ids the template inserts right before the assistant
    # content. Robust way: tokenize the literal "<|im_start|>assistant\n".
    assistant_header_ids = tok(
        f"{ims_name}assistant\n", add_special_tokens=False
    )["input_ids"] if ims_name else []
    # eot block: usually the im_end id (+ maybe a trailing newline)
    assistant_eot_ids = [ime_id] if ime_id is not None else []

    # ---- 3) image pad open/close blocks ----------------------------------
    # In Qwen-VL, an image is wrapped as: <|vision_start|> <image_pad>*N <|vision_end|>
    image_pad_open_ids = [vs_id] if vs_id is not None else []
    image_pad_close_ids = [ve_id] if ve_id is not None else []

    # ---- 4) patch / merge (for placeholder-count formula) ----------------
    ip = proc.image_processor
    patch_size = getattr(ip, "patch_size", None)
    merge_size = getattr(ip, "merge_size", None)

    result = {
        "image_placeholder_id": pad_id,        # the <image_pad> id repeated N times
        "image_pad_open_ids": image_pad_open_ids,   # [<|vision_start|>]
        "image_pad_close_ids": image_pad_close_ids,  # [<|vision_end|>]
        "assistant_header_ids": assistant_header_ids,
        "assistant_eot_ids": assistant_eot_ids,
        "patch_size": patch_size,
        "merge_size": merge_size,
        # NOTE: collapse_marker_ids is NOT auto-derivable -- it must be the EXACT
        # id block your model was TRAINED with for a folded/collapsed image.
        # Set it to whatever marker your training data used, e.g. the ids of a
        # literal "<|vision_start|><|vision_end|>" (empty image) or a custom
        # "[image omitted]" string. See the note printed below.
        "collapse_marker_ids": None,
    }

    print("=" * 60)
    print("Resolved names -> ids:")
    print(f"  vision_start: {vs_name} -> {vs_id}")
    print(f"  vision_end  : {ve_name} -> {ve_id}")
    print(f"  image_pad   : {pad_name} -> {result['image_placeholder_id']}")
    print(f"  im_start    : {ims_name} -> {ims_id}")
    print(f"  im_end      : {ime_name} -> {ime_id}")
    print(f"  patch/merge : {patch_size}/{merge_size}")
    print("=" * 60)
    print("Paste into task config (fill collapse_marker_ids yourself):")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("patch_size", "merge_size")},
                     indent=2, ensure_ascii=False))
    print("=" * 60)
    print("collapse_marker_ids: set this to the EXACT id block your model was")
    print("trained with for a collapsed image. To get the ids for a literal")
    print("marker string, run:")
    print("  proc.tokenizer('<your marker>', add_special_tokens=False)['input_ids']")
    print("If your model folds an old image to an EMPTY vision block, that is")
    print(f"  [{vs_id}, {ve_id}]  (vision_start immediately followed by vision_end)")

    return result


if __name__ == "__main__" and "--collapse" not in __import__("sys").argv:
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-VL-..."
    extract_constants(model_path)


# ===========================================================================
# collapse marker -> fixed id block (text marker case)
# ===========================================================================
def resolve_collapse_marker_ids(
    model_path: str,
    marker_text: str = "[image omitted]",
    wrap_in_vision_tokens: bool = True,
):
    """Turn a TEXT collapse marker into a FIXED id block.

    The returned ids are pinned ONCE and reused verbatim everywhere a folded
    image appears -- we never re-tokenize the marker string in context (that
    could shift boundaries). Set wrap_in_vision_tokens=True to emit
        <|vision_start|> <marker tokens> <|vision_end|>
    so the model still sees a structural "an image used to be here" signal,
    mirroring how an empty/placeholder image slot looks.
    """
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tok = proc.tokenizer

    marker_ids = tok(marker_text, add_special_tokens=False)["input_ids"]

    if wrap_in_vision_tokens:
        vs = tok.convert_tokens_to_ids("<|vision_start|>")
        ve = tok.convert_tokens_to_ids("<|vision_end|>")
        block = [vs] + marker_ids + [ve]
    else:
        block = marker_ids

    print("=" * 60)
    print(f"collapse marker text : {marker_text!r}")
    print(f"marker token ids     : {marker_ids}")
    print(f"wrap_in_vision_tokens: {wrap_in_vision_tokens}")
    print(f"collapse_marker_ids  : {block}")
    print("=" * 60)
    print("Paste this list as task.collapse_marker_ids. It is a FIXED block;")
    print("the context manager inserts these ids directly (no re-tokenization).")
    print("Round-trip check (decode):")
    print("  ", repr(tok.decode(block)))
    return block


if __name__ == "__main__" and "--collapse" in __import__("sys").argv:
    # usage:
    #   python extract_model_constants.py <model> --collapse "This screenshot has been collapsed."
    #   add --no-wrap to keep it as plain text (no <|vision_start|>/<|vision_end|>)
    import sys
    argv = sys.argv
    model = argv[1]
    i = argv.index("--collapse")
    # join everything after --collapse (except flags) so spaces are preserved
    # even if the user forgot to quote the string.
    rest = [a for a in argv[i + 1:] if a != "--no-wrap"]
    marker = " ".join(rest) if rest else "[image omitted]"
    wrap = "--no-wrap" not in argv
    resolve_collapse_marker_ids(model, marker_text=marker, wrap_in_vision_tokens=wrap)
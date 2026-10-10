"""Score memes with Hugging Face transformers (optionally with a LoRA adapter). Slower than vLLM; used for the
validation split during training, for the zero-shot baseline, and as a fallback. UNTESTED here (no GPU stack).

  python -m memes.predict_hf --model HuggingFaceTB/SmolVLM-Instruct --jsonl data/hm/dev_seen.jsonl --out out/zs_dev.jsonl
  python -m memes.predict_hf --model ... --adapter out/lora/adapter --jsonl ... --out out/lora_dev.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from memes.data import filter_ids, load_memes, write_jsonl
from memes.hf_common import DEFAULT_MODEL, load_processor, open_image, prompt_text, yes_no_token_ids
from memes.scoring import score_from_logits


def score_rows(model, processor, rows: list[dict], batch_size: int = 16) -> np.ndarray:
    import torch
    tok = processor.tokenizer
    tok.padding_side = "left"                       # next-token logits must be read at the LAST position
    yes_ids, no_ids = yes_no_token_ids(tok)
    model.eval()
    out = []
    for s in range(0, len(rows), batch_size):
        chunk = rows[s:s + batch_size]
        inputs = processor(text=[prompt_text(processor, r["text"]) for r in chunk],
                           images=[[open_image(r["path"])] for r in chunk], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :].float().cpu().numpy()
        out.append(score_from_logits(logits, yes_ids, no_ids))
    return np.concatenate(out)


def load_model(model_id: str, adapter: str | None = None):
    import torch
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-ids", default=None, help="score only the ids in this JSONL (e.g. out/lora/val_ids.jsonl)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=384)
    a = ap.parse_args()
    rows = load_memes(a.jsonl)
    if a.only_ids:
        rows = filter_ids(rows, a.only_ids)
    processor = load_processor(a.model, a.image_size)
    model = load_model(a.model, a.adapter)
    t0 = time.time()
    scores = score_rows(model, processor, rows, a.batch_size)
    write_jsonl([{"id": r["id"], "label": r["label"], "score": float(s)} for r, s in zip(rows, scores)], a.out)
    print(json.dumps({"n": len(rows), "seconds": round(time.time() - t0, 1)}))

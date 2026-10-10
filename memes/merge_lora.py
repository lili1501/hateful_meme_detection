"""Merge a LoRA adapter into the base weights so vLLM can load one plain checkpoint. UNTESTED here (no GPU stack).

  python -m memes.merge_lora --model HuggingFaceTB/SmolVLM-Instruct --adapter out/lora/adapter --out out/merged
"""
import argparse

from memes.hf_common import DEFAULT_MODEL

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out)
    AutoProcessor.from_pretrained(a.adapter).save_pretrained(a.out)    # processor saved with the training image settings
    print("merged model written to", a.out)

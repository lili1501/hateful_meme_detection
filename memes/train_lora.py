"""LoRA fine-tune a vision-language model to answer "Is this meme hateful? Yes/No".

Only the language model's attention/MLP projections get LoRA adapters; the vision tower stays frozen.
Loss is computed on the answer tokens only. UNTESTED here: it needs a GPU stack (torch, transformers, peft) that
could not be installed in the build sandbox. Expect to fix small API issues on the first run.

  python -m memes.train_lora --train data/hm/train.jsonl --out out/lora --epochs 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np

from memes.data import load_memes, stratified_split, write_jsonl
from memes.hf_common import DEFAULT_MODEL, load_processor, open_image, prompt_text
from memes.labels import mask_prompt_labels
from memes.metrics import auroc
from memes.prompt import answer_word

# Matches q/k/v/o and MLP projections inside the TEXT model only (vision tower also has q_proj/k_proj/v_proj).
DEFAULT_TARGET_REGEX = r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def make_collate(processor, end_token: str):
    def collate(batch: list[dict]):
        import torch
        tok = processor.tokenizer
        tok.padding_side = "right"
        prompts = [prompt_text(processor, r["text"]) for r in batch]
        images = [[open_image(r["path"])] for r in batch]
        full = [p + answer_word(r["label"]) + end_token for p, r in zip(prompts, batch)]
        enc = processor(text=full, images=images, return_tensors="pt", padding=True)
        # prompt length (with image tokens expanded) = tokens of the prompt-only text, processed with the same image
        plens = [int(processor(text=[p], images=[im], return_tensors="pt")["attention_mask"].sum())
                 for p, im in zip(prompts, images)]
        labels = mask_prompt_labels(enc["input_ids"].numpy(), enc["attention_mask"].numpy(), plens)
        enc["labels"] = torch.from_numpy(labels)
        return enc
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train.jsonl of the Hateful Memes data")
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--target-regex", default=DEFAULT_TARGET_REGEX)
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--end-token", default="<end_of_utterance>", help="end-of-turn marker in the chat template")
    ap.add_argument("--max-train-rows", type=int, default=None, help="debug: use only N rows")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows = load_memes(a.train)
    train_rows, val_rows = stratified_split(rows, a.val_frac, a.seed)
    if a.max_train_rows:
        train_rows = train_rows[:a.max_train_rows]
    os.makedirs(a.out, exist_ok=True)
    write_jsonl([{"id": r["id"]} for r in val_rows], os.path.join(a.out, "val_ids.jsonl"))

    processor = load_processor(a.model, a.image_size)
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout,
                                             target_modules=a.target_regex, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    loader = DataLoader(train_rows, batch_size=a.batch_size, shuffle=True, collate_fn=make_collate(processor, a.end_token),
                        num_workers=4, drop_last=True)
    steps = math.ceil(len(loader) / a.grad_accum) * a.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    from memes.predict_hf import score_rows
    log, t0, step = [], time.time(), 0
    for epoch in range(a.epochs):
        model.train()
        for i, batch in enumerate(loader):
            batch = batch.to("cuda")
            loss = model(**batch).loss / a.grad_accum
            loss.backward()
            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 10 == 0:
                    log.append({"step": step, "epoch": epoch, "loss": loss.item() * a.grad_accum,
                                "elapsed_s": round(time.time() - t0, 1)})
                    print(log[-1], flush=True)
        scores = score_rows(model, processor, val_rows)
        val_auc = auroc(np.array([r["label"] for r in val_rows]), scores)
        log.append({"epoch_end": epoch, "val_auroc": val_auc})
        print(f"epoch {epoch}: val AUROC {val_auc:.4f}", flush=True)

    write_jsonl([{"id": r["id"], "label": r["label"], "score": float(x)} for r, x in zip(val_rows, scores)],
                os.path.join(a.out, "val_pred.jsonl"))
    model.save_pretrained(os.path.join(a.out, "adapter"))
    processor.save_pretrained(os.path.join(a.out, "adapter"))
    json.dump({"args": vars(a), "log": log, "train_seconds": round(time.time() - t0, 1),
               "n_train": len(train_rows), "n_val": len(val_rows)}, open(os.path.join(a.out, "train_log.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

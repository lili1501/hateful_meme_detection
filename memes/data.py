"""Hateful Memes data: JSONL reader, validation, and a seeded stratified train/val split.

Official format, one JSON object per line:  {"id": 42953, "img": "img/42953.png", "label": 0, "text": "..."}
Files used: train.jsonl (training), dev_seen.jsonl (final evaluation). The `label` field is 1 = hateful.
The data is NOT in this repo: request it from https://hatefulmemes.com (licence terms apply).
"""
from __future__ import annotations

import json
import os

import numpy as np


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_memes(jsonl_path: str, image_root: str | None = None, check_images: bool = True) -> list[dict]:
    """Rows with absolute `path`. image_root defaults to the folder holding the jsonl (img/ lives next to it)."""
    root = image_root or os.path.dirname(os.path.abspath(jsonl_path))
    rows = read_jsonl(jsonl_path)
    out, seen = [], set()
    for r in rows:
        for key in ("id", "img", "text", "label"):
            if key not in r:
                raise ValueError(f"{jsonl_path}: row missing '{key}': {r}")
        if r["label"] not in (0, 1):
            raise ValueError(f"{jsonl_path}: label must be 0/1, got {r['label']!r} (id {r['id']}). "
                             "Unlabelled test files cannot be scored.")
        if r["id"] in seen:
            raise ValueError(f"{jsonl_path}: duplicate id {r['id']}")
        seen.add(r["id"])
        path = os.path.join(root, r["img"])
        if check_images and not os.path.exists(path):
            raise FileNotFoundError(path)
        out.append({"id": r["id"], "path": path, "text": r["text"].strip(), "label": int(r["label"])})
    return out


def filter_ids(rows: list[dict], ids_jsonl: str) -> list[dict]:
    """Keep only rows whose id appears in a JSONL of {"id": ...} (e.g. the val_ids.jsonl written by training),
    so the zero-shot baseline is scored on exactly the same validation memes as the fine-tuned model."""
    keep = {r["id"] for r in read_jsonl(ids_jsonl)}
    out = [r for r in rows if r["id"] in keep]
    if len(out) != len(keep):
        raise ValueError(f"{len(keep) - len(out)} ids from {ids_jsonl} are not in the data file")
    return out


def stratified_split(rows: list[dict], val_frac: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Seeded split that keeps the hateful/non-hateful ratio in both parts. Returns (train, val)."""
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be in (0, 1)")
    rng = np.random.default_rng(seed)
    val_idx = set()
    for lab in (0, 1):
        idx = np.array([i for i, r in enumerate(rows) if r["label"] == lab])
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        val_idx.update(idx[:max(1, int(round(len(idx) * val_frac)))].tolist())
    train = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [r for i, r in enumerate(rows) if i in val_idx]
    return train, val

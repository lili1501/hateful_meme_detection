"""Tiny fake dataset in the Hateful Memes file format, used ONLY to test the plumbing (loader, split, reports).
The images are plain coloured cards with text; the labels are arbitrary and mean nothing."""
from __future__ import annotations

import os
import random

from PIL import Image, ImageDraw

from memes.data import write_jsonl

_WORDS = ["cat", "dog", "tree", "river", "coffee", "monday", "friend", "storm", "garden", "pizza"]


def make_fake_split(out_dir: str, name: str, n: int, seed: int = 0, pos_rate: float = 0.36, start_id: int = 1000) -> str:
    rng = random.Random(seed)
    os.makedirs(os.path.join(out_dir, "img"), exist_ok=True)
    rows = []
    for k in range(n):
        i = start_id + k
        im = Image.new("RGB", (160, 120), tuple(rng.randint(60, 220) for _ in range(3)))
        text = " ".join(rng.choices(_WORDS, k=rng.randint(3, 7)))
        ImageDraw.Draw(im).text((8, 50), text, fill=(255, 255, 255))
        im.save(os.path.join(out_dir, "img", f"{i}.png"))
        rows.append({"id": i, "img": f"img/{i}.png", "label": int(rng.random() < pos_rate), "text": text})
    path = os.path.join(out_dir, f"{name}.jsonl")
    write_jsonl(rows, path)
    return path

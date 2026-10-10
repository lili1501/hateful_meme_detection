"""Turn prediction files into the results table (no GPU/torch needed).

A prediction file is JSONL with {"id", "label", "score"} per meme (score = P(hateful)); this is what
predict_hf.py and predict_vllm.py write. The decision threshold is tuned on the VALIDATION predictions of the
same run (a split of train.jsonl) and then applied once to the dev predictions, so dev is never used to tune.

  python -m memes.report --dev zero_shot=out/zs_dev.jsonl --dev lora=out/lora_dev.jsonl \
                         --val zero_shot=out/zs_val.jsonl --val lora=out/lora_val.jsonl --out reports/results
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from memes.data import read_jsonl
from memes.metrics import auroc, best_threshold, summarize


def load_preds(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = read_jsonl(path)
    ids = np.array([r["id"] for r in rows]); y = np.array([r["label"] for r in rows])
    s = np.array([r["score"] for r in rows], dtype=np.float64)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    if not np.isfinite(s).all():
        raise ValueError(f"{path}: non-finite scores")
    return ids, y, s


def paired_auroc_diff(y, s_a, s_b, n: int = 1000, seed: int = 0) -> dict:
    """AUROC(b) - AUROC(a) on the SAME memes with a paired bootstrap 95% CI."""
    y, s_a, s_b = map(np.asarray, (y, s_a, s_b))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(auroc(y[i], s_b[i]) - auroc(y[i], s_a[i]))
    return {"diff": auroc(y, s_b) - auroc(y, s_a),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def build_report(dev: dict[str, str], val: dict[str, str]) -> dict:
    out = {"runs": {}}
    aligned = {}
    for name, path in dev.items():
        ids, y, s = load_preds(path)
        thr = 0.5
        if name in val:
            _, yv, sv = load_preds(val[name])
            thr = best_threshold(yv, sv)
        r = summarize(y, s, 0.5)
        r["at_val_threshold"] = {"threshold": thr, **{k: v for k, v in summarize(y, s, thr).items()
                                                      if k in ("accuracy", "confusion")}}
        out["runs"][name] = r
        aligned[name] = (ids, y, s)
    names = list(dev)
    if len(names) >= 2:
        a, b = names[0], names[-1]
        (ia, ya, sa), (ib, yb, sb) = aligned[a], aligned[b]
        if sorted(ia.tolist()) != sorted(ib.tolist()):
            raise ValueError(f"'{a}' and '{b}' were not scored on the same memes")
        order = np.argsort(ib); inv = np.searchsorted(ib[order], ia)
        sb = sb[order][inv]
        out["paired_auroc_diff"] = {"from": a, "to": b, **paired_auroc_diff(ya, sa, sb)}
    return out


def to_markdown(rep: dict) -> str:
    lines = ["| Run | n | AUROC (95% CI) | Acc @0.5 | Acc @val-tuned thr |", "|---|---|---|---|---|"]
    for name, r in rep["runs"].items():
        lo, hi = r["auroc_ci95"]; v = r["at_val_threshold"]
        lines.append(f"| {name} | {r['n']} | {r['auroc']:.4f} ({lo:.4f}-{hi:.4f}) | {r['accuracy']:.4f} | "
                     f"{v['accuracy']:.4f} (thr {v['threshold']:.3f}) |")
    if "paired_auroc_diff" in rep:
        d = rep["paired_auroc_diff"]
        lines += ["", f"AUROC {d['to']} - {d['from']}: {d['diff']:+.4f} (paired bootstrap 95% CI {d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f})"]
    return "\n".join(lines) + "\n"


def _kv(items):
    return dict(i.split("=", 1) for i in items or [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="append", required=True, help="name=dev_predictions.jsonl (repeatable)")
    ap.add_argument("--val", action="append", help="name=val_predictions.jsonl (repeatable)")
    ap.add_argument("--out", default="reports/results")
    a = ap.parse_args()
    rep = build_report(_kv(a.dev), _kv(a.val))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out + ".json", "w"), indent=2)
    open(a.out + ".md", "w").write(to_markdown(rep))
    print(to_markdown(rep))

"""Fast batch scoring with vLLM. Reads the first generated token's top log-probs and converts Yes/No into P(hateful).
The vLLM calls are UNTESTED here (vLLM needs a GPU); `first_token_logprobs` is tested against stand-in objects that
mimic vLLM's RequestOutput shape.

  python -m memes.predict_vllm --model out/merged --jsonl data/hm/dev_seen.jsonl --out out/lora_dev.jsonl
  python -m memes.predict_vllm --model HuggingFaceTB/SmolVLM-Instruct --jsonl ... --out out/zs_dev.jsonl   # zero-shot
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from memes.data import filter_ids, load_memes, write_jsonl
from memes.hf_common import DEFAULT_MODEL, load_processor, open_image, prompt_text
from memes.scoring import score_from_logprobs


def first_token_logprobs(request_output) -> dict[str, float]:
    """decoded token -> log-prob for the first generated position of a vLLM RequestOutput."""
    out = {}
    for lp in request_output.outputs[0].logprobs[0].values():
        tok = lp.decoded_token if lp.decoded_token is not None else ""
        out[tok] = max(lp.logprob, out.get(tok, -1e30))
    return out


def scores_from_outputs(outputs) -> tuple[np.ndarray, int]:
    scores, unparsed = [], 0
    for o in outputs:
        s, ok = score_from_logprobs(first_token_logprobs(o))
        scores.append(s); unparsed += (not ok)
    return np.array(scores), unparsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="merged checkpoint dir or a Hugging Face id")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-ids", default=None, help="score only the ids in this JSONL (e.g. out/lora/val_ids.jsonl)")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--latency-n", type=int, default=0, help="also time N single-meme requests (batch size 1)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    rows = load_memes(a.jsonl)
    if a.only_ids:
        rows = filter_ids(rows, a.only_ids)
    processor = load_processor(a.model, a.image_size)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"do_image_splitting": False, "size": {"longest_edge": a.image_size}})
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=a.top_logprobs)

    def request(r):
        return {"prompt": prompt_text(processor, r["text"]), "multi_modal_data": {"image": open_image(r["path"])}}

    t0 = time.time()
    outputs = llm.generate([request(r) for r in rows], sp)
    batch_s = time.time() - t0
    scores, unparsed = scores_from_outputs(outputs)
    write_jsonl([{"id": r["id"], "label": r["label"], "score": float(s)} for r, s in zip(rows, scores)], a.out)

    stats = {"n": len(rows), "unparsed_yes_no": unparsed, "batch_seconds": round(batch_s, 1),
             "images_per_second": round(len(rows) / batch_s, 2)}
    if a.latency_n:
        lat = []
        for r in rows[:a.latency_n]:
            t = time.perf_counter(); llm.generate([request(r)], sp, use_tqdm=False); lat.append((time.perf_counter() - t) * 1000)
        stats.update({"single_request_ms_p50": float(np.percentile(lat, 50)), "single_request_ms_p95": float(np.percentile(lat, 95))})
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

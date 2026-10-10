"""Score memes through a running vLLM OpenAI-compatible server and measure request latency.

Start the server (needs a GPU):
  vllm serve out/merged --dtype bfloat16 --max-model-len 4096 --limit-mm-per-prompt '{"image":1}' --port 8001
Then:
  python -m memes.serve_client --url http://localhost:8001 --model out/merged --jsonl data/hm/dev_seen.jsonl --n 200

The request/response handling is tested against a mock HTTP transport; the real server was never started here.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time

import httpx
import numpy as np

from memes.data import load_memes
from memes.hf_common import open_image
from memes.prompt import QUESTION
from memes.scoring import score_from_logprobs


def build_payload(model: str, image, text: str, top_logprobs: int = 20) -> dict:
    buf = io.BytesIO(); image.save(buf, "PNG")
    uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return {"model": model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": top_logprobs,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": QUESTION.format(text=text)}]}]}


def parse_response(resp: dict) -> tuple[float, bool]:
    top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return score_from_logprobs({t["token"]: t["logprob"] for t in top})


def score_one(client: httpx.Client, url: str, model: str, row: dict) -> tuple[float, bool, float]:
    t = time.perf_counter()
    r = client.post(f"{url}/v1/chat/completions", json=build_payload(model, open_image(row["path"]), row["text"]), timeout=120)
    r.raise_for_status()
    ms = (time.perf_counter() - t) * 1000
    s, ok = parse_response(r.json())
    return s, ok, ms


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--model", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    rows = load_memes(a.jsonl)[:a.n]
    with httpx.Client() as c:
        res = [score_one(c, a.url, a.model, r) for r in rows]
    ms = np.array([x[2] for x in res])
    print(json.dumps({"n": len(rows), "unparsed": int(sum(not x[1] for x in res)),
                      "latency_ms_p50": float(np.percentile(ms, 50)), "latency_ms_p95": float(np.percentile(ms, 95)),
                      "note": "includes client-side image encoding and HTTP; sequential requests"}, indent=2))

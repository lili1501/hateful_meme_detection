"""Turn the model's Yes/No next-token distribution into a score P(hateful) in [0, 1].

AUROC needs a continuous score, so we never use the generated word itself: we read the probabilities of the
"Yes" and "No" tokens at the first answer position. Tokenizers produce variants ("Yes", " Yes", "▁Yes", "yes"),
so all variants are pooled with logsumexp.
"""
from __future__ import annotations

import math

import numpy as np

MISSING_LOGPROB = -30.0      # stand-in log-prob when only one of Yes/No shows up in the returned top-k


def norm_token(tok: str) -> str:
    return tok.replace("▁", "").replace("Ġ", "").strip().lower()


def _logsumexp(vals: list[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def score_from_logprobs(token_logprobs: dict[str, float]) -> tuple[float, bool]:
    """token_logprobs: decoded token string -> log-prob (e.g. top-20 from vLLM). Returns (P(hateful), parsed).
    parsed=False (score 0.5) when neither Yes nor No is among the tokens, which callers must count and report."""
    yes = [lp for t, lp in token_logprobs.items() if norm_token(t) == "yes"]
    no = [lp for t, lp in token_logprobs.items() if norm_token(t) == "no"]
    if not yes and not no:
        return 0.5, False
    ly = _logsumexp(yes) if yes else MISSING_LOGPROB
    ln = _logsumexp(no) if no else MISSING_LOGPROB
    return _sigmoid(ly - ln), True


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def score_from_logits(logits: np.ndarray, yes_ids: list[int], no_ids: list[int]) -> np.ndarray:
    """logits: [N, vocab] at the first answer position. Returns P(hateful) [N] using only Yes vs No mass."""
    logits = np.asarray(logits, dtype=np.float64)
    def lse(ids):
        sub = logits[:, ids]
        m = sub.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(sub - m).sum(axis=1, keepdims=True))).ravel()
    z = lse(yes_ids) - lse(no_ids)
    return 1.0 / (1.0 + np.exp(-z))

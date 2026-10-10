"""Loss masking for supervised fine-tuning: train only on the answer ("Yes"/"No" + end-of-turn), not the prompt."""
from __future__ import annotations

import numpy as np

IGNORE = -100


def mask_prompt_labels(input_ids: np.ndarray, attention_mask: np.ndarray, prompt_lens: list[int]) -> np.ndarray:
    """labels = input_ids, except prompt tokens and padding are IGNORE (-100).

    Assumes RIGHT padding (real tokens first), which is how training batches are built.
    prompt_lens[i] = number of leading tokens of row i that belong to the prompt (image placeholders included).
    Raises if a row has no answer tokens left (e.g. the sequence was truncated inside the prompt).
    """
    input_ids, attention_mask = np.asarray(input_ids), np.asarray(attention_mask)
    labels = input_ids.copy()
    for i, (row_mask, p) in enumerate(zip(attention_mask, prompt_lens)):
        n = int(row_mask.sum())
        if not (row_mask[:n] == 1).all():
            raise ValueError(f"row {i}: attention mask is not right-padded")
        if p >= n:
            raise ValueError(f"row {i}: prompt ({p} tokens) leaves no answer tokens in a {n}-token sequence")
        labels[i, :p] = IGNORE
        labels[i, n:] = IGNORE
    return labels

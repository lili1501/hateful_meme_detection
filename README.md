# Multimodal Hateful Meme Detection (VLM + LoRA + vLLM)

Fine-tunes an open vision-language model with **LoRA** to answer *"Is this meme hateful? Yes/No"* on the
[Facebook Hateful Memes](https://hatefulmemes.com) data, and scores memes with **vLLM**. The score is
P(hateful) read from the model's Yes/No next-token probabilities, so AUROC is computed from a real continuous score,
not from generated text.

Default model: **`HuggingFaceTB/SmolVLM-Instruct`** (Idefics3 architecture, Apache-2.0, Hugging Face). It is a US/French
open model, chosen because Qwen is not usable on the target cluster. Any other VLM works if its chat template and
LoRA target modules are given (`--model`, `--target-regex`, `--end-token`); Llama 3.2 Vision, PaliGemma and Pixtral are
alternatives but are gated or larger. Only the default has been considered in detail, and none were run.

> **Status.** Run end to end once on a B200 (one training run, one seed): zero-shot scoring, LoRA fine-tuning (2 epochs),
> merge, and vLLM scoring all worked. Results and caveats are in the Results section. The sbatch scripts were not used (the
> run used `scripts/run_eval_local.sh` in an interactive session), so they remain untested.

## Pipeline

```
train.jsonl ─ stratified 90/10 ─▶ train ─▶ LoRA SFT (loss on the "Yes"/"No" tokens only) ─▶ adapter ─▶ merge ─▶ vLLM
                                  └ val  ─▶ tune decision threshold                                              │
dev_seen.jsonl ───────────────────────────────────────────────────────────────▶ score = P(Yes)/(P(Yes)+P(No)) ◀─┘
                                                      zero-shot baseline uses the same scoring with no adapter
```

| File | Role | Tested here |
|---|---|---|
| `memes/data.py` | JSONL loader with validation, seeded stratified split, id filter | yes |
| `memes/prompt.py` | Yes/No question and chat messages | yes |
| `memes/labels.py` | masks prompt/padding so loss is on answer tokens only | yes |
| `memes/scoring.py` | Yes/No log-probs or logits → P(hateful) (pools token variants, stable) | yes |
| `memes/metrics.py` | AUROC (ties handled), accuracy, threshold search, bootstrap CI | yes, vs scikit-learn |
| `memes/report.py` | prediction files → results table, paired bootstrap of zero-shot vs LoRA | yes |
| `memes/predict_vllm.py`, `serve_client.py` | vLLM batch scoring; OpenAI-API client with latency | parsing and HTTP payloads only |
| `memes/train_lora.py`, `merge_lora.py`, `predict_hf.py`, `hf_common.py` | training, merge, HF scoring | **no** (import and `--help` only) |
| `scripts/hipergator_*.sbatch` | job scripts | **no** |

## How to run (HiPerGator or any GPU box)

Data is not included: request the Hateful Memes dataset from the authors (their licence applies; the images carry
their own licensing) and place `train.jsonl`, `dev_seen.jsonl` and `img/` under `data/hm/`. The public `test` split has
no labels here and cannot be scored; `dev_seen` is the evaluation set.

```bash
python -m venv .venv-train && source .venv-train/bin/activate && pip install -r requirements-train.txt && deactivate
python -m venv .venv-vllm  && source .venv-vllm/bin/activate  && pip install -r requirements-vllm.txt  && deactivate
# on a node WITH internet: pre-download the weights into $HF_HOME (compute nodes may be offline)
HF_HOME=$PWD/.hf_cache huggingface-cli download HuggingFaceTB/SmolVLM-Instruct
sbatch scripts/hipergator_train.sbatch     # edit partition/modules first
sbatch scripts/hipergator_eval.sbatch      # after training finishes; writes reports/results.md and results.json
```

`results.md` reports, for the zero-shot model and the LoRA model on `dev_seen`: AUROC with a 95% bootstrap CI, accuracy at
0.5, accuracy at a threshold tuned on the held-out validation split (never on dev), and the paired AUROC difference.
`out/lora_dev.jsonl.stats.json` has vLLM throughput (images/s), single-request latency, and the count of responses where
neither Yes nor No was in the top-20 tokens (should be ~0). Optional serving check:
`vllm serve out/merged ...` then `python -m memes.serve_client` (see its docstring).

Run the torch-free tests anywhere: `pip install -r requirements.txt && python -m pytest tests -q`.

## Results (one run, SmolVLM-Instruct + LoRA, B200)
Data: a third-party Hugging Face mirror of the Hateful Memes data (see Data below), **not** the official download.
That mirror lacks about 20% of the images, so the evaluation set is **398 of the 500 `dev_seen` memes** (102 dropped) and
training used 6,744 of the 8,500 `train` memes, split 90/10 into training and validation. The dropped rows have almost the
same hateful rate as the kept ones (train 0.356 vs 0.355, dev 0.480 vs 0.497), but only the label was checked.
These numbers are **not comparable to published `dev_seen` results**.

| Run | n | AUROC (95% bootstrap CI) | Accuracy @ 0.5 |
|---|---|---|---|
| zero-shot SmolVLM | 398 | 0.596 (0.545 - 0.651) | 0.540 |
| + LoRA (2 epochs) | 398 | 0.781 (0.730 - 0.824) | 0.706 |

Paired AUROC difference, LoRA minus zero-shot: **+0.185** (95% CI +0.130 to +0.234).

- AUROC is the headline: it needs no threshold. Accuracy at a validation-tuned threshold was *lower* (0.663 at threshold
  0.622) than at 0.5, because the validation split comes from `train` (35.5% hateful) and `dev_seen` is 49.7% hateful, so
  a threshold tuned on one balance does not carry over. It is reported in `reports/results.md` but not used as a result.
- One training run with one seed and untuned hyperparameters (rank 16, lr 2e-4, 2 epochs, 384-px images). No variance
  across seeds, so the size of the gain is an estimate. The CI covers resampling of the 398 dev memes only.
- The text of each meme is given to the model, so it is not a pure image task.
- I believe SmolVLM's pre-training data (The Cauldron) includes Hateful Memes training images; this is unverified. The
  zero-shot score on `dev_seen` (0.596) shows no sign of memorisation there, but the fine-tuned gain should still be read as
  "relative to this zero-shot baseline", not as an absolute capability claim.
- Early stopping on validation AUROC with best-epoch saving was added after the first run; this run used 2 epochs, where
  it cannot trigger.

## Data
Not included. The licence (`LICENSE.txt` in the dataset) restricts redistribution and use to a stated research purpose and
requires accepting it to access the data; the copy used here came from a third-party mirror
(`neuralcatcher/hateful_memes`), not from Facebook's own download, so check the licence terms for your own use.
`data/` and any prediction file containing meme text must stay out of git. Cite Kiela et al., "The Hateful Memes Challenge",
NeurIPS 2020. If some images are missing, `HM_DROP_MISSING=1` drops those rows and prints how many (and the label rates).

## Verified vs not verified

**Verified by running it (30 tests, plus one real GPU run: loading, LoRA training, merging, vLLM scoring of base and merged models, report):**
- AUROC equals scikit-learn on random data including heavy ties; known values; threshold search; bootstrap CI.
- Yes/No scoring: variants pooled, symmetric under swapping classes, no overflow at extreme logits, missing-token cases.
- Loss masking on right-padded batches, including error cases (prompt swallowing the answer, left padding).
- Loader rejects bad labels, missing fields, duplicate ids, missing images, unlabeled files; split is stratified and seeded.
- The report pipeline end to end on synthetic predictions (reordered rows, mismatched runs rejected, paired CI).
- vLLM output parsing against stand-in objects; the serve client's request body and response parsing via a mock HTTP transport.
- Deliberately breaking the masking, scoring sign and AUROC tie handling each made tests fail.

**Not verified:**
- Multiple seeds, hyperparameter sensitivity, and group-level error rates (see Notes).
- The sbatch scripts (partition names, modules, whether compute nodes are offline).
- The early-stopping branch on a GPU run longer than 2 epochs (unit-tested only).

## Notes and limits
- Hyperparameters (rank 16, lr 2e-4, 2 epochs, 384-px images) are untuned defaults.
- Latency figures from `predict_vllm` time the first token only (prefill), one request at a time.
- The text of each meme is given to the model in the prompt (the dataset provides it); the model must still use the image
  to catch the "benign confounder" pairs the dataset is built around.
- Hateful-content classifiers can encode bias against identity groups; evaluate error rates by group before any real use.

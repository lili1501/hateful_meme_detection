# Multimodal Hateful Meme Detection (VLM + LoRA + vLLM)

Fine-tunes an open vision-language model with **LoRA** to answer *"Is this meme hateful? Yes/No"* on the
[Facebook Hateful Memes](https://hatefulmemes.com) data, and scores memes with **vLLM**. The score is
P(hateful) read from the model's Yes/No next-token probabilities, so AUROC is computed from a real continuous score,
not from generated text.

Default model: **`HuggingFaceTB/SmolVLM-Instruct`** (Idefics3 architecture, Apache-2.0, Hugging Face). It is a US/French
open model, chosen because Qwen is not usable on the target cluster. Any other VLM works if its chat template and
LoRA target modules are given (`--model`, `--target-regex`, `--end-token`); Llama 3.2 Vision, PaliGemma and Pixtral are
alternatives but are gated or larger. Only the default has been considered in detail, and none were run.

> **Status.** The model-free parts are tested here (27 tests). **Nothing that loads a model has been run**: the build
> sandbox has no GPU, no PyTorch CUDA libraries and no access to Hugging Face. So this repo contains **no accuracy or AUROC
> results**, and `train_lora.py`, `predict_hf.py`, `predict_vllm.py`, `merge_lora.py` and the sbatch scripts are
> untested against the real libraries. Expect small API fixes on the first GPU run.

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

## Verified vs not verified

**Verified by running it:**
- AUROC equals scikit-learn on random data including heavy ties; known values; threshold search; bootstrap CI.
- Yes/No scoring: variants pooled, symmetric under swapping classes, no overflow at extreme logits, missing-token cases.
- Loss masking on right-padded batches, including error cases (prompt swallowing the answer, left padding).
- Loader rejects bad labels, missing fields, duplicate ids, missing images, unlabeled files; split is stratified and seeded.
- The report pipeline end to end on synthetic predictions (reordered rows, mismatched runs rejected, paired CI).
- vLLM output parsing against stand-in objects; the serve client's request body and response parsing via a mock HTTP transport.
- Deliberately breaking the masking, scoring sign and AUROC tie handling each made tests fail.

**Not verified:**
- Everything that needs the model stack: model/processor loading, LoRA target regex matching the right layers, the
  `size`/`do_image_splitting` processor settings, prompt-length computation with expanded image tokens, bf16 memory, the
  `<end_of_utterance>` end-of-turn marker, merging, and vLLM loading of the merged checkpoint with those processor settings.
- Any accuracy, AUROC, speed, or latency figure. None exist yet.
- The sbatch scripts (partition names, modules, whether compute nodes are offline).
- Real Hateful Memes data (only fake memes in the same file format were used).

## Notes and limits
- Hyperparameters (rank 16, lr 2e-4, 2 epochs, 384-px images) are untuned defaults.
- The text of each meme is given to the model in the prompt (the dataset provides it); the model must still use the image
  to catch the "benign confounder" pairs the dataset is built around.
- Hateful-content classifiers can encode bias against identity groups; evaluate error rates by group before any real use.

#!/bin/bash
# Merge a trained LoRA adapter, score dev_seen + the validation memes with vLLM, and build the results table.
# Interactive-terminal version of hipergator_eval.sbatch. UNTESTED on a GPU here.
#   LORA=out/lora_e2 bash scripts/run_eval_local.sh
# Needs: $EXT set (source external/_env/env.sh) and envs $EXT/_envs/train (merge) and $EXT/_envs/vllm (scoring).
set -eo pipefail
: "${EXT:?source external/_env/env.sh first}"
LORA=${LORA:-out/lora}; MODEL=${MODEL:-HuggingFaceTB/SmolVLM-Instruct}
export HM_DROP_MISSING=1
export VLLM_CACHE_ROOT=$EXT/_vllm/cache VLLM_CONFIG_ROOT=$EXT/_vllm/config TRITON_CACHE_DIR=$EXT/_vllm/triton
export TORCHINDUCTOR_CACHE_DIR=$EXT/_vllm/inductor FLASHINFER_WORKSPACE_BASE=$EXT/_vllm CUDA_CACHE_PATH=$EXT/_vllm/cuda
mkdir -p out reports
[ -f "$LORA/adapter/adapter_config.json" ] || { echo "no adapter in $LORA/adapter - training not finished?"; exit 1; }

echo "== 1/3 merge adapter"
( source "$EXT/_envs/train/bin/activate"; python -m memes.merge_lora --model "$MODEL" --adapter "$LORA/adapter" --out out/merged )

echo "== 2/3 score with vLLM"
( source "$EXT/_envs/vllm/bin/activate"
  V="--only-ids $LORA/val_ids.jsonl"
  [ -f out/zs_dev.jsonl ] || python -m memes.predict_vllm --model "$MODEL" --jsonl data/hm/dev_seen.jsonl --out out/zs_dev.jsonl
  python -m memes.predict_vllm --model "$MODEL"   --jsonl data/hm/train.jsonl $V --out out/zs_val.jsonl
  python -m memes.predict_vllm --model out/merged --jsonl data/hm/dev_seen.jsonl --out out/lora_dev.jsonl --latency-n 100
  python -m memes.predict_vllm --model out/merged --jsonl data/hm/train.jsonl $V --out out/lora_val.jsonl )

echo "== 3/3 report"
( source "$EXT/_envs/train/bin/activate"
  python -m memes.report --dev zero_shot=out/zs_dev.jsonl --dev lora=out/lora_dev.jsonl \
                         --val zero_shot=out/zs_val.jsonl --val lora=out/lora_val.jsonl --out reports/results )
cat out/lora_dev.jsonl.stats.json; echo; cat reports/results.md

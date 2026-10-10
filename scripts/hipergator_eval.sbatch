#!/bin/bash
# Zero-shot vs LoRA scoring with vLLM, then the results table. UNTESTED. Run after hipergator_train.sbatch.
#SBATCH --job-name=hm-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
##SBATCH --account=YOUR_GROUP
##SBATCH --qos=YOUR_GROUP
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p logs out reports
command -v module >/dev/null 2>&1 && module load python cuda || echo "WARNING: 'module' not found"
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}; export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
MODEL=${MODEL:-HuggingFaceTB/SmolVLM-Instruct}

# 1) merge the adapter into a standalone checkpoint (needs the training env's peft)
source .venv-train/bin/activate
python -m memes.merge_lora --model "$MODEL" --adapter out/lora/adapter --out out/merged
deactivate

# 2) score with vLLM (separate env): dev_seen for both models, and the SAME validation memes for threshold tuning
source .venv-vllm/bin/activate
V="--only-ids out/lora/val_ids.jsonl"
python -m memes.predict_vllm --model "$MODEL"     --jsonl data/hm/dev_seen.jsonl --out out/zs_dev.jsonl
python -m memes.predict_vllm --model "$MODEL"     --jsonl data/hm/train.jsonl $V --out out/zs_val.jsonl
python -m memes.predict_vllm --model out/merged   --jsonl data/hm/dev_seen.jsonl --out out/lora_dev.jsonl --latency-n 100
python -m memes.predict_vllm --model out/merged   --jsonl data/hm/train.jsonl $V --out out/lora_val.jsonl

# 3) table: AUROC with bootstrap CI, accuracy, val-tuned threshold, paired zero-shot -> LoRA difference
python -m memes.report --dev zero_shot=out/zs_dev.jsonl --dev lora=out/lora_dev.jsonl \
                       --val zero_shot=out/zs_val.jsonl --val lora=out/lora_val.jsonl --out reports/results
cat out/lora_dev.jsonl.stats.json

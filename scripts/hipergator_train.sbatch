#!/bin/bash
# LoRA fine-tune on one GPU. UNTESTED. Edit partition/gres/modules for your HiPerGator allocation first
# (check UF Research Computing docs for current GPU partition names). Submit: sbatch scripts/hipergator_train.sbatch
#SBATCH --job-name=hm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=gpu
#SBATCH --gpus=1
##SBATCH --account=YOUR_GROUP
##SBATCH --qos=YOUR_GROUP
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"; mkdir -p logs out
command -v module >/dev/null 2>&1 && module load python cuda || echo "WARNING: 'module' not found"
source .venv-train/bin/activate
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}           # keep model weights on /blue, not in $HOME
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}          # compute nodes may have no internet: pre-download on a login node
MODEL=${MODEL:-HuggingFaceTB/SmolVLM-Instruct}
nvidia-smi | head -15; date
python -m memes.train_lora --train data/hm/train.jsonl --out out/lora --model "$MODEL" --epochs "${EPOCHS:-2}"
date

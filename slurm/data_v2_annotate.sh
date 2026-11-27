#!/bin/bash
# Data v2, step 2: label SA-1B images (objects, facing, depth) on GPUs. One task per shard:
#   sbatch --array=0-3 slurm/data_v2_annotate.sh          # 4 shards of the JPEGs in data/train_v2/images/sa1b
# Resumable: re-submitting skips images already in a shard's labels file.
#SBATCH --job-name=v2-annotate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
DATA="${DATA:-/home/bhsshah/vlm-project/data/train_v2}"
NUM="${SLURM_ARRAY_TASK_COUNT:-1}"
nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONPATH=src:training/data_v2 "${PY:-/home/bhsshah/vlm-project/.venv/bin/python}" training/data_v2/annotate.py \
  --images-dir "$DATA/images/sa1b" --out "$DATA/annotations" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NUM" --wait-for "$DATA/images/sa1b/fetch_report.json" "$@"

#!/bin/bash
# Data v2, step 5: round-trip check of the object descriptions with Qwen3.5-9B (thinking off) served by vLLM.
# Defaults are for the Quadro RTX 8000 nodes (CUDA 12.9 vLLM build, float16). Extra args go to the script:
#   sbatch slurm/data_v2_verify.sh --limit 40 --scale 1000      # calibration run
#   sbatch slurm/data_v2_verify.sh                              # all descriptions
#SBATCH --job-name=v2-verify
#SBATCH --exclude=tesla1,tesla2,h100,orion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
DATA="${DATA:-/home/bhsshah/vlm-project/data/train_v2}"
VLLM_ENV="${VLLM_ENV:-$HOME/venvs/vllm-cu129}"
PORT="${PORT:-18452}"
OUT="${OUT:-$DATA/verified.jsonl}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$VLLM_ENV/bin/vllm" serve Qwen/Qwen3.5-9B --port "$PORT" --host 127.0.0.1 --max-model-len 16384 \
  --dtype "${DTYPE:-float16}" --limit-mm-per-prompt '{"image": 1}' --gpu-memory-utilization 0.88 \
  > "logs/vllm_v2_verify_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "vLLM exited; see logs/vllm_v2_verify_${SLURM_JOB_ID:-local}.log" >&2; exit 1; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "vLLM did not come up in 30 min" >&2; exit 1; }
PYTHONPATH=training/data_v2:src "${PY:-/home/bhsshah/vlm-project/.venv/bin/python}" training/data_v2/verify_descriptions.py \
  --descriptions "$DATA/descriptions.jsonl" --labels "$DATA/annotations/labels_*.jsonl" --out "$OUT" \
  --base-url "http://127.0.0.1:$PORT/v1" "$@"

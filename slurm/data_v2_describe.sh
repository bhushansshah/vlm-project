#!/bin/bash
# Data v2, step 4: referring descriptions for the objects the candidate questions name, with Qwen3.5-4B
# (thinking off) served by vLLM. Defaults are for the Quadro RTX 8000 nodes (CUDA 12.9 vLLM build, float16).
#   sbatch slurm/data_v2_describe.sh
#SBATCH --job-name=v2-describe
#SBATCH --exclude=tesla1,tesla2,h100,orion
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=22G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
DATA="${DATA:-/home/bhsshah/vlm-project/data/train_v2}"
VLLM_ENV="${VLLM_ENV:-$HOME/venvs/vllm-cu129}"
PORT="${PORT:-18451}"
nvidia-smi --query-gpu=name --format=csv,noheader
"$VLLM_ENV/bin/vllm" serve Qwen/Qwen3.5-4B --port "$PORT" --host 127.0.0.1 --max-model-len 8192 \
  --dtype "${DTYPE:-float16}" --limit-mm-per-prompt '{"image": 1}' --gpu-memory-utilization 0.85 \
  > "logs/vllm_v2_describe_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "vLLM exited; see logs/vllm_v2_describe_${SLURM_JOB_ID:-local}.log" >&2; exit 1; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "vLLM did not come up in 30 min" >&2; exit 1; }
"${PY:-/home/bhsshah/vlm-project/.venv/bin/python}" training/data_v2/describe.py \
  --objects "$DATA/candidates.objects.jsonl" --out "$DATA/descriptions.jsonl" \
  --base-url "http://127.0.0.1:$PORT/v1" "$@"

#!/bin/bash
# TreeBench (no tools) with Qwen3.5-4B, thinking on, served by vLLM on one GPU.
#   sbatch slurm/treebench_qwen35_4b.sh                 # all 405 questions (resumes if re-run)
#   sbatch slurm/treebench_qwen35_4b.sh --limit 5       # smoke test; extra args go to `vlmbench run`
# Quadro RTX 8000 (driver 570, no bf16): CUDA 12.9 vLLM build, float16, its own run tag:
#   VLLM_ENV=$HOME/venvs/vllm-cu129 DTYPE=float16 CONFIG=configs/treebench_local/qwen3.5-4b-think-fp16.yaml \
#     sbatch --nodelist=quadro2 slurm/treebench_qwen35_4b.sh --limit 5
#SBATCH --job-name=tb-qwen35-4b
#SBATCH --nodelist=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

VLLM_ENV="${VLLM_ENV:-$HOME/venvs/vllm-qwen35}"   # separate env: vLLM needs a newer torch than .venv (CUDA 13 build: H100 driver 580)
DTYPE="${DTYPE:-auto}"                            # auto = bf16; float16 on Turing GPUs
PORT="${PORT:-18431}"                             # a second job on the same node needs another PORT
CONFIG="${CONFIG:-configs/treebench_local/qwen3.5-4b-think.yaml}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo "port $PORT is already in use on $(hostname); stop that server or change PORT and the config" >&2
  exit 1
fi

# 32768 context = up to ~6.7K image tokens + prompt + 25K output
"$VLLM_ENV/bin/vllm" serve Qwen/Qwen3.5-4B \
  --port "$PORT" --host 127.0.0.1 \
  --max-model-len 32768 \
  --dtype "$DTYPE" \
  --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 1}' \
  --gpu-memory-utilization 0.85 \
  > "logs/vllm_qwen35_4b_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "vLLM exited; see logs/vllm_qwen35_4b_${SLURM_JOB_ID:-local}.log" >&2; exit 1; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "vLLM did not come up in 15 min" >&2; exit 1; }
echo "vLLM ready after ~$((i * 5)) s"

.venv/bin/vlmbench run --config "$CONFIG" -o base_url="http://localhost:$PORT/v1" "$@"

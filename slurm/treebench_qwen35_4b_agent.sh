#!/bin/bash
# TreeBench tool agent with Qwen3.5-4B (thinking on), served by vLLM on one GPU; the agent's tools
# (Grounding DINO, Depth Anything, ViTPose) run in the vlmbench process on the same GPU.
#   sbatch slurm/treebench_qwen35_4b_agent.sh                           # all 405 questions (resumes if re-run)
#   sbatch slurm/treebench_qwen35_4b_agent.sh --limit 5                 # smoke test; extra args go to `vlmbench run`
#   sbatch slurm/treebench_qwen35_4b_agent.sh --question-ids analysis/treebench_tuning_ids.txt
# Differences from slurm/treebench_qwen35_4b.sh (the no-tools baseline):
#   --enable-auto-tool-choice --tool-call-parser qwen3_coder   native tool calls in Qwen3.5's format
#   --limit-mm-per-prompt image 48                            the original image + one image per tool result
#   --max-model-len 131072                                    room for many steps (Qwen3.5 allows 262144)
#   --gpu-memory-utilization 0.80                             leaves ~9 GB for the tools' networks
# --mem 22G: the baseline jobs peaked at 9-11 GB host RAM; the tools add ~5 GB. The Quadro nodes have
# little RAM left unallocated, so a bigger request can wait in the queue.
# Defaults are for the Quadro RTX 8000 nodes (driver 570, no bf16): the CUDA 12.9 vLLM build in float16.
#SBATCH --job-name=tb-qwen35-4b-agent
#SBATCH --nodelist=quadro2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=22G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

VLLM_ENV="${VLLM_ENV:-$HOME/venvs/vllm-cu129}"
DTYPE="${DTYPE:-float16}"
PORT="${PORT:-18431}"                              # a second job on the same node needs another PORT (and -o base_url=...)
CONFIG="${CONFIG:-configs/treebench_local/qwen3.5-4b-agent-tb3.yaml}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
  echo "port $PORT is already in use on $(hostname); stop that server or change PORT" >&2
  exit 1
fi

"$VLLM_ENV/bin/vllm" serve Qwen/Qwen3.5-4B \
  --port "$PORT" --host 127.0.0.1 \
  --max-model-len 131072 \
  --dtype "$DTYPE" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image": 48}' \
  --gpu-memory-utilization 0.80 \
  > "logs/vllm_qwen35_4b_agent_${SLURM_JOB_ID:-local}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
  if ! kill -0 $SERVER 2>/dev/null; then echo "vLLM exited; see logs/vllm_qwen35_4b_agent_${SLURM_JOB_ID:-local}.log" >&2; exit 1; fi
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "vLLM did not come up in 15 min" >&2; exit 1; }
echo "vLLM ready after ~$((i * 5)) s"

.venv/bin/vlmbench run --config "$CONFIG" -o base_url="http://localhost:$PORT/v1" "$@"

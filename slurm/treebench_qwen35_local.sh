#!/bin/bash
# TreeBench with Qwen3.5 on a local H100 machine (no Slurm): picks the config and the server script for
#   <size>  4b | 27b
#   <tools> notools | tools     (tools = the tb3 agent: zoom, detect, depth, distance, orient, viewpoint)
#   <think> think | nothink     (enable_thinking on/off, each with Qwen's recommended sampling)
# and runs it in bf16 on the GPUs in CUDA_VISIBLE_DEVICES (one GPU = TP 1, two = TP 2, ...).
# Configs: configs/treebench_h100/qwen3.5-<size>[-agent-tb3]-<think>.yaml; outputs: runs/<run name>/.
#
#   source env.local.sh
#   CUDA_VISIBLE_DEVICES=7 bash slurm/treebench_qwen35_local.sh 4b notools think --limit 5   # smoke test
#   CUDA_VISIBLE_DEVICES=7 bash slurm/treebench_qwen35_local.sh 4b notools think             # all 405 (resumes)
#   CUDA_VISIBLE_DEVICES=6,7 bash slurm/treebench_qwen35_local.sh 27b tools think            # TP 2
# One shared server for several runs (it has the agent's settings, which also serve no-tools runs):
#   CUDA_VISIBLE_DEVICES=6,7 bash slurm/treebench_qwen35_local.sh 27b serve                  # server only
#   ATTACH=1 CUDA_VISIBLE_DEVICES=6 bash slurm/treebench_qwen35_local.sh 27b tools think     # run on it
#   ATTACH=1 CUDA_VISIBLE_DEVICES=6 bash slurm/treebench_qwen35_local.sh 27b notools think   # (GPU: tools)
# Extra args go to `vlmbench run`. Each runs in a detached tmux session tb-<size>-<tools>-<think>
# (tb-<size>-serve; `tmux attach -t <name>`; NO_TMUX=1 runs in the foreground). A second server at the
# same time needs its own PORT=... . Override GPU_UTIL=... if the GPU is shared.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() { sed -n '2,20p' "$0" >&2; exit 2; }
[ $# -ge 2 ] || usage
SIZE=$1 TOOLS=$2; shift 2
case $SIZE in 4b) MODEL=Qwen/Qwen3.5-4B ;; 27b) MODEL=Qwen/Qwen3.5-27B ;; *) usage ;; esac
if [ "$TOOLS" = serve ]; then
  THINK=""; SCRIPT=slurm/treebench_qwen35_4b_agent.sh; CONFIG=configs/treebench_h100/qwen3.5-$SIZE-agent-tb3-think.yaml
  SESSION="tb-$SIZE-serve"
else
  [ $# -ge 1 ] || usage
  THINK=$1; shift
  case $THINK in think|nothink) ;; *) usage ;; esac
  case $TOOLS in
    notools) SCRIPT=slurm/treebench_qwen35_4b.sh;       CONFIG=configs/treebench_h100/qwen3.5-$SIZE-$THINK.yaml ;;
    tools)   SCRIPT=slurm/treebench_qwen35_4b_agent.sh; CONFIG=configs/treebench_h100/qwen3.5-$SIZE-agent-tb3-$THINK.yaml ;;
    *) usage ;;
  esac
  SESSION="tb-$SIZE-$TOOLS-$THINK"
fi
[ -f "$CONFIG" ] || { echo "missing $CONFIG" >&2; exit 1; }

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPU(s) to use}"
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
  case $g in 2|4) [ -n "${ALLOW_GPU_2_4:-}" ] || { echo "GPU $g is off-limits on this machine (ALLOW_GPU_2_4=1 when cleared)" >&2; exit 1; } ;; esac
done
TP=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")

# GPU memory share for vLLM: the agent leaves room for the tools' networks (~5 GB) on the same GPU.
if [ "$TOOLS" = notools ]; then DEFAULT_UTIL=0.85; else DEFAULT_UTIL=0.80; fi
[ "$SIZE" = 27b ] && [ "$TP" = 1 ] && DEFAULT_UTIL=$( [ "$TOOLS" = notools ] && echo 0.90 || echo 0.85 )

if [ -z "${TMUX:-}" ] && [ -z "${NO_TMUX:-}" ]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux session $SESSION already exists" >&2; exit 1; fi
  mkdir -p logs
  LOG="logs/${SESSION}_$(date -u +%Y%m%dT%H%M%SZ).out"
  # the tmux server has its own environment: pass on the caller's settings explicitly
  FWD="NO_TMUX=1 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  for v in PORT GPU_UTIL DTYPE VLLM_ENV ALLOW_GPU_2_4 ATTACH VLLM_EXTRA_ARGS; do [ -n "${!v:-}" ] && FWD+=" $v=$(printf '%q' "${!v}")"; done
  ARGS=("$SIZE" "$TOOLS"); [ -n "$THINK" ] && ARGS+=("$THINK")
  tmux new-session -d -s "$SESSION" \
    "env $FWD bash $(printf '%q ' "$0" "${ARGS[@]}" "$@") 2>&1 | tee $LOG"
  echo "started tmux session $SESSION (log: $LOG)"
  exit 0
fi

[ -f env.local.sh ] && source env.local.sh
export MODEL CONFIG TP
export PORT="${PORT:-18431}"
export DTYPE="${DTYPE:-auto}"                       # bf16 on H100
export GPU_UTIL="${GPU_UTIL:-$DEFAULT_UTIL}"
export VLLM_ENV="${VLLM_ENV:-$HOME/venvs/vllm-qwen35}"
# NVLink SHARP multicast fails here ("Failed to bind NVLS Multicast memory ... CUDA error 401"); TP > 1
# works with it off (plain NVLink is still used).
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
echo "$(date -u) model=$MODEL tp=$TP gpus=$CUDA_VISIBLE_DEVICES util=$GPU_UTIL config=$CONFIG attach=${ATTACH:-}"

if [ -n "${ATTACH:-}" ]; then
  # run only the client, against the server already on PORT (wait up to 30 min for it to come up)
  for i in $(seq 1 360); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:$PORT/v1/models" | grep -q "\"$MODEL\"" \
    || { echo "no server for $MODEL on port $PORT (start one with: ... $SIZE serve)" >&2; exit 1; }
  exec .venv/bin/vlmbench run --config "$CONFIG" -o base_url="http://localhost:$PORT/v1" "$@"
fi
[ "$TOOLS" = serve ] && export SERVE_ONLY=1
exec bash "$SCRIPT" "$@"

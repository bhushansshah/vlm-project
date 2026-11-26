#!/bin/bash
# RL stage, run 1 (docs/plans/treebench_sft_rl_plan.md, section 5): GRPO vs GSPO from the base Qwen3.5-4B,
# trained with thinking off on the RL pool, then each evaluated like the SFT models (dev + TreeBench, tested with
# thinking off = same mode, and with thinking on = cross-mode).
#   GPU=4 bash training/rl_v1.sh grpo gspo      # arms run one after another on one GPU
set -uo pipefail
cd "$(dirname "$0")/.."
source env.local.sh
TRAIN_PY=/home/marafat/venvs/train-qwen35/bin/python
GPU=${GPU:-4}
PORT=${PORT:-18436}
STEPS=${STEPS:-100}
LOG=logs/rl-v1.log
log() { echo "$(date -u +%FT%TZ) [rl-v1 gpu$GPU] $*" | tee -a "$LOG"; }
die() { log "FAILED: $*"; exit 1; }

for algo in "$@"; do
  out=checkpoints/rl_${algo}_base
  log "training $algo from the base model ($STEPS steps)"
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $TRAIN_PY training/rl_loop.py --algo "$algo" \
    --out "$out" --steps "$STEPS" --prompts-per-step 8 --steps-per-round 2 --max-new 256 > logs/rl-v1-train-$algo.log 2>&1 \
    || die "training $algo (see logs/rl-v1-train-$algo.log)"
  log "trained $algo; serving $out/merged"
  sess=rl-v1-serve-$algo; rm -f logs/$sess.out
  tmux new-session -d -s "$sess" "cd $PWD; source env.local.sh; SERVE_ONLY=1 MODEL=$PWD/$out/merged PORT=$PORT GPU_UTIL=0.85 DTYPE=auto \
    VLLM_EXTRA_ARGS='--max-num-seqs 128' NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=$GPU bash slurm/treebench_qwen35_4b_agent.sh 2>&1 | tee logs/$sess.out"
  sleep 10; until grep -qE "serving|vLLM exited|did not come up" logs/$sess.out 2>/dev/null; do sleep 10; done
  grep -q serving logs/$sess.out || die "server for $algo did not start"
  for mode in nothink think; do
    vlmbench run --config configs/train_v1/base-4b-$mode-dev.yaml --model "vllm:$PWD/$out/merged" -o base_url=http://localhost:$PORT/v1 \
      --run-name "train_v1__rl-${algo}-base-${mode}__dev" >> "$LOG" 2>&1 || die "dev eval $algo $mode"
    vlmbench run --config configs/treebench_h100/qwen3.5-4b-$mode.yaml --model "vllm:$PWD/$out/merged" -o base_url=http://localhost:$PORT/v1 \
      --run-name "treebench__rl-${algo}-base__${mode}-h100" >> "$LOG" 2>&1 || die "treebench eval $algo $mode"
    log "evaluated $algo (tested $mode): dev $(python -c "import json;print(json.load(open('runs/train_v1__rl-${algo}-base-${mode}__dev/summary.json'))['accuracy']['overall']['acc'])"), TreeBench $(python -c "import json;print(json.load(open('runs/treebench__rl-${algo}-base__${mode}-h100/summary.json'))['accuracy']['overall']['acc'])")"
  done
  tmux send-keys -t "$sess" C-c; sleep 20
done
log "done"

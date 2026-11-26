#!/bin/bash
# Tool-call SFT, run 1 (arm B1), all on one GPU, queued after the RL run (tmux session rl-v1):
#   1. Qwen3.5-27B tb3 agent (thinking off) on the SFT pool, dumping every conversation with its tool images
#   2. one training example per assistant turn from the correct conversations (training/make_agent_sft.py)
#   3. LoRA SFT of Qwen3.5-4B on them (thinking off; tool results in the prompt, not in the loss)
#   4. evaluate as the tb3 agent: dev + TreeBench, tested with thinking off (same mode) and on (cross-mode);
#      plus the base 4B agent on dev (both modes) and on TreeBench with thinking on (H100), for comparison.
#   GPU=4 bash training/tool_sft_v1.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source env.local.sh
TRAIN_PY=/home/marafat/venvs/train-qwen35/bin/python
GPU=${GPU:-4}
LOG=logs/tool-sft-v1.log
log() { echo "$(date -u +%FT%TZ) [tool-sft-v1 gpu$GPU] $*" | tee -a "$LOG"; }
die() { log "FAILED: $*"; exit 1; }
acc() { python -c "import json;print(json.load(open('runs/$1/summary.json'))['accuracy']['overall']['acc'])"; }

serve() { # model gpu_util port session [max-num-seqs]
  rm -f logs/$4.out
  tmux new-session -d -s "$4" "cd $PWD; source env.local.sh; SERVE_ONLY=1 MODEL=$1 PORT=$3 GPU_UTIL=$2 DTYPE=auto \
    VLLM_EXTRA_ARGS='--max-num-seqs ${5:-128}' NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=$GPU bash slurm/treebench_qwen35_4b_agent.sh 2>&1 | tee logs/$4.out"
  sleep 10; until grep -qE "serving|vLLM exited|did not come up" logs/$4.out 2>/dev/null; do sleep 10; done
  grep -q serving logs/$4.out || die "server $4 did not start"
}
stop() { tmux send-keys -t "$1" C-c 2>/dev/null; for i in $(seq 1 60); do tmux has-session -t "$1" 2>/dev/null || return 0; sleep 2; done; }

log "waiting for the RL run (rl-v1) to finish"
while tmux has-session -t rl-v1 2>/dev/null; do sleep 60; done

# 1. teacher trajectories
serve Qwen/Qwen3.5-27B 0.90 18431 tsv1-serve-27b 64
log "27B teacher agent on the SFT pool"
vlmbench run --config configs/train_v1/teacher-27b-agent-nothink-sft.yaml >> "$LOG" 2>&1 || die "teacher agent run"
log "teacher agent: $(acc train_v1__teacher-27b-agent-nothink__sft)% correct on the SFT pool"
stop tsv1-serve-27b

# 2. data
python training/make_agent_sft.py --dumps data/agent_dumps/teacher-27b-nothink \
  --run runs/train_v1__teacher-27b-agent-nothink__sft --out data/sft_arms/b1_teacher_agent.jsonl >> "$LOG" 2>&1 || die "make b1"
log "b1 data: $(python -c "import json;r=json.load(open('data/sft_arms/b1_teacher_agent.report.json'));print(r['conversations'],'conversations,',r['examples'],'examples')")"

# 3. train
CUDA_VISIBLE_DEVICES=$GPU $TRAIN_PY training/sft.py --data data/sft_arms/b1_teacher_agent.jsonl --out checkpoints/b1_teacher_agent \
  --max-len 16384 > logs/tool-sft-v1-train.log 2>&1 || die "training b1 (see logs/tool-sft-v1-train.log)"
log "trained b1"

# 4. evaluate
M=$PWD/checkpoints/b1_teacher_agent/merged
serve "$M" 0.85 18437 tsv1-serve-b1
for mode in nothink think; do
  vlmbench run --config configs/train_v1/agent-4b-$mode-dev.yaml --model "vllm:$M" -o base_url=http://localhost:18437/v1 \
    --run-name "train_v1__sft-b1-agent-${mode}__dev" >> "$LOG" 2>&1 || die "b1 dev $mode"
  vlmbench run --config configs/treebench_h100/qwen3.5-4b-agent-tb3-$mode.yaml --model "vllm:$M" -o base_url=http://localhost:18437/v1 \
    --run-name "treebench__sft-b1-agent__${mode}-h100" >> "$LOG" 2>&1 || die "b1 treebench $mode"
  log "evaluated b1 agent (trained thinking off, tested $mode): dev $(acc train_v1__sft-b1-agent-${mode}__dev), TreeBench $(acc treebench__sft-b1-agent__${mode}-h100)"
done
stop tsv1-serve-b1

serve Qwen/Qwen3.5-4B 0.85 18437 tsv1-serve-base
for mode in nothink think; do
  vlmbench run --config configs/train_v1/agent-4b-$mode-dev.yaml -o base_url=http://localhost:18437/v1 >> "$LOG" 2>&1 || die "base agent dev $mode"
  log "base 4B agent, dev, tested $mode: $(acc train_v1__base-4b-agent-${mode}__dev)"
done
ATTACH=1 NO_TMUX=1 ALLOW_GPU_2_4=1 PORT=18437 CUDA_VISIBLE_DEVICES=$GPU bash slurm/treebench_qwen35_local.sh 4b tools think >> "$LOG" 2>&1 \
  || die "base agent treebench think"
log "base 4B agent, TreeBench, tested think (H100): $(acc treebench__vllm__Qwen_Qwen3.5-4B__default__agent-tools-tb3-native-sys__think-h100)"
stop tsv1-serve-base
log "done"

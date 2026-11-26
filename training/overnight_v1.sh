#!/bin/bash
# Unattended SFT pipeline for data v1 (docs/plans/treebench_sft_rl_plan.md, section 4), one lane per GPU:
#   bash training/overnight_v1.sh lane4    # GPU 4: A3 (self-distillation), then A1 (bare answers)
#   bash training/overnight_v1.sh lane2    # GPU 2: A2 (27B teacher traces)
# Each lane waits for its inputs (trace runs finishing), trains a LoRA, merges it, serves the merged model with
# vLLM and evaluates it in both thinking modes on the dev set and on all of TreeBench, with the same configs
# as the baselines. Every step logs to logs/tv1-<lane>.log; a lane stops at the first failure.
set -uo pipefail
cd "$(dirname "$0")/.."
source env.local.sh
TRAIN_PY=/home/marafat/venvs/train-qwen35/bin/python
LANE=$1
LOG=logs/tv1-$LANE.log
log() { echo "$(date -u +%FT%TZ) [$LANE] $*" | tee -a "$LOG"; }
die() { log "FAILED: $*"; exit 1; }

wait_session_gone() { while tmux has-session -t "$1" 2>/dev/null; do sleep 60; done; }
stop_session() { tmux has-session -t "$1" 2>/dev/null || return 0; tmux send-keys -t "$1" C-c
                 for i in $(seq 1 60); do tmux has-session -t "$1" 2>/dev/null || return 0; sleep 2; done
                 die "session $1 did not stop"; }

serve() { # model_path gpu port session
  local out=logs/$4.out; rm -f "$out"
  tmux new-session -d -s "$4" "cd $PWD; source env.local.sh; SERVE_ONLY=1 MODEL=$1 PORT=$3 GPU_UTIL=0.85 DTYPE=auto \
    VLLM_EXTRA_ARGS='--max-num-seqs 128' NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES=$2 bash slurm/treebench_qwen35_4b_agent.sh 2>&1 | tee $out"
  sleep 10
  until grep -qE "serving|vLLM exited|did not come up" "$out" 2>/dev/null; do sleep 10; done
  grep -q serving "$out" || die "server for $1 did not start (see $out)"
}

evaluate() { # model_path port name
  local m=$1 port=$2 name=$3
  for mode in think nothink; do
    vlmbench run --config configs/train_v1/base-4b-$mode-dev.yaml --model "vllm:$m" -o base_url=http://localhost:$port/v1 \
      --run-name "train_v1__${name}-${mode}__dev" >> "$LOG" 2>&1 || die "dev eval $name $mode"
    vlmbench run --config configs/treebench_h100/qwen3.5-4b-$mode.yaml --model "vllm:$m" -o base_url=http://localhost:$port/v1 \
      --run-name "treebench__${name}__${mode}-h100" >> "$LOG" 2>&1 || die "treebench eval $name $mode"
    log "evaluated $name ($mode): dev $(python -c "import json;print(json.load(open('runs/train_v1__${name}-${mode}__dev/summary.json'))['accuracy']['overall']['acc'])"), TreeBench $(python -c "import json;print(json.load(open('runs/treebench__${name}__${mode}-h100/summary.json'))['accuracy']['overall']['acc'])")"
  done
}

train_eval() { # arm gpu port
  local arm=$1 gpu=$2 port=$3
  [ -s data/sft_arms/$arm.jsonl ] || die "missing data/sft_arms/$arm.jsonl"
  log "training $arm on GPU $gpu ($(wc -l < data/sft_arms/$arm.jsonl) examples)"
  CUDA_VISIBLE_DEVICES=$gpu $TRAIN_PY training/sft.py --data data/sft_arms/$arm.jsonl --out checkpoints/$arm \
    > logs/tv1-train-$arm.log 2>&1 || die "training $arm (see logs/tv1-train-$arm.log)"
  log "trained $arm; serving checkpoints/$arm/merged"
  serve "$PWD/checkpoints/$arm/merged" "$gpu" "$port" "tv1-serve-$arm"
  evaluate "$PWD/checkpoints/$arm/merged" "$port" "sft-$arm"
  stop_session "tv1-serve-$arm"
}

case $LANE in
  lane4)
    log "waiting for the 4B trace runs (tv1-4b) and the 4B TreeBench baseline (tb-4b-notools-think)"
    wait_session_gone tv1-4b; wait_session_gone tb-4b-notools-think
    stop_session tb-4b-serve
    python training/make_sft_arms.py --arms a3_self >> "$LOG" 2>&1 || die "make a3"
    train_eval a3_self 4 18434
    log "waiting for a1_bare (built by lane2 when the teacher run ends)"
    until [ -s data/sft_arms/a1_bare.jsonl ] && [ -e data/sft_arms/.teacher_arms_done ]; do sleep 60; done
    train_eval a1_bare 4 18434
    log "lane done" ;;
  lane2)
    log "waiting for the teacher run (tv1-teacher)"
    wait_session_gone tv1-teacher
    stop_session tb-27b-serve
    python training/make_sft_arms.py --arms a1_bare a2_teacher >> "$LOG" 2>&1 || die "make a1/a2"
    touch data/sft_arms/.teacher_arms_done
    train_eval a2_teacher 2 18435
    log "lane done" ;;
  *) echo "usage: $0 lane2|lane4" >&2; exit 2 ;;
esac

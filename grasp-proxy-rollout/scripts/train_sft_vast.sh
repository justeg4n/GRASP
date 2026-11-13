#!/usr/bin/env bash
set -euo pipefail

# Override these paths when your Vast.ai template mounts a different workspace.
PROJECT_DIR="${PROJECT_DIR:-/workspace/grasp-proxy-rollout}"
SFT_DIR="${SFT_DIR:-/workspace/data/grasp-sft}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints/grasp-qwen2.5-3b-sft}"
OUTPUT_ARCHIVE="${OUTPUT_ARCHIVE:-${OUTPUT_DIR}-handoff.zip}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR/logs"
exec > >(tee -a "$OUTPUT_DIR/logs/console.log") 2>&1

GPU_LOG="$OUTPUT_DIR/logs/gpu_metrics.csv"
if [[ ! -f "$GPU_LOG" ]]; then
  echo "timestamp,index,name,gpu_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_w" \
    > "$GPU_LOG"
fi
(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits >> "$GPU_LOG" || true
    sleep 30
  done
) &
GPU_MONITOR_PID=$!

finalize() {
  status=$?
  trap - EXIT
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
  python "$PROJECT_DIR/scripts/package_sft_output.py" \
    --checkpoint-dir "$OUTPUT_DIR" \
    --output "$OUTPUT_ARCHIVE" \
    --allow-partial || true
  echo "Run exit code: $status"
  echo "Download handoff ZIP from: $OUTPUT_ARCHIVE"
  exit "$status"
}
trap finalize EXIT

cd "$PROJECT_DIR"
python -m pip install --upgrade pip
python -m pip install -e ".[train]"

grasp-validate-sft-data \
  "$SFT_DIR/sft/train.jsonl" \
  "$SFT_DIR/sft/val.jsonl" \
  --output "$OUTPUT_DIR/data_validation.json"

resume_args=()
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  resume_args=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

grasp-train-sft \
  --base-model "$BASE_MODEL" \
  --train-file "$SFT_DIR/sft/train.jsonl" \
  --validation-file "$SFT_DIR/sft/val.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --max-length 3072 \
  --epochs 1 \
  --batch-size 1 \
  --eval-batch-size 1 \
  --max-eval-samples 1024 \
  --gradient-accumulation-steps 16 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --lora-r 64 \
  --lora-alpha 128 \
  --lora-dropout 0.05 \
  --compute-dtype bf16 \
  --use-4bit \
  --gradient-checkpointing \
  --eval-steps 250 \
  --save-steps 250 \
  --logging-steps 1 \
  --early-stopping-patience 3 \
  --early-stopping-threshold 0.001 \
  --save-total-limit 2 \
  "${resume_args[@]}"

grasp-validate-checkpoint \
  --checkpoint "$OUTPUT_DIR" \
  --data-file "$SFT_DIR/sft/val.jsonl" \
  --samples-per-role 32 \
  --batch-size 4 \
  --max-sequence-length 3072 \
  --minimum-format-rate 0.95 \
  --load-in-4bit \
  --output "$OUTPUT_DIR/checkpoint_validation.json"

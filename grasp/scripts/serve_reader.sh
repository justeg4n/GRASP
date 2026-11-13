#!/usr/bin/env bash
# Serve reader đóng băng (Qwen2.5-7B-Instruct) bằng vLLM trong Docker, OpenAI-compatible tại :8000.
#   bash scripts/serve_reader.sh fp8     # mặc định — RTX 50xx (Blackwell) hỗ trợ FP8 phần cứng
#   bash scripts/serve_reader.sh awq     # fallback nếu FP8 lỗi
#   DETACH=1 bash scripts/serve_reader.sh fp8   # chạy nền (container tên grasp-reader), xem log: docker logs -f grasp-reader
# Cần Docker Desktop (WSL2 backend) + NVIDIA driver mới. Lần đầu sẽ tải image (~10GB) và model (~15GB).
set -euo pipefail
MODE="${1:-fp8}"
UTIL="${GPU_UTIL:-0.72}"          # 16GB card: ~2.5GB desktop + ~0.7GB E5 query encoder chạy cùng lúc
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# Git Bash (MSYS) tự đổi "/root/..." thành đường dẫn Windows -> mount hỏng âm thầm, model bị tải lại mỗi lần.
if command -v cygpath >/dev/null 2>&1; then
  export MSYS_NO_PATHCONV=1
  HF_CACHE="$(cygpath -w "$HF_CACHE")"
fi

case "$MODE" in
  fp8) MODEL=Qwen/Qwen2.5-7B-Instruct;     QUANT=(--quantization fp8 --dtype bfloat16) ;;
  awq) MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ; QUANT=(--quantization awq_marlin --dtype float16) ;;
  *) echo "mode phải là fp8 hoặc awq" >&2; exit 1 ;;
esac

RUN_FLAGS=(--rm)
[[ "${DETACH:-0}" == "1" ]] && RUN_FLAGS=(-d)   # không --rm để còn đọc log khi container chết

docker rm -f grasp-reader >/dev/null 2>&1 || true
docker run "${RUN_FLAGS[@]}" --gpus all --ipc=host -p 8000:8000 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  --name grasp-reader "$IMAGE" \
  --model "$MODEL" "${QUANT[@]}" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --port 8000

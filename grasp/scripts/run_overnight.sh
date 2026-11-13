#!/usr/bin/env bash
# Chạy trọn pipeline sinh dữ liệu SFT, để treo máy qua đêm. Chạy lại đúng lệnh này để tiếp tục.
#   bash scripts/run_overnight.sh                          # 2wiki -> hotpotqa -> nq, teacher Bedrock
#   DATASETS="2wiki" bash scripts/run_overnight.sh         # chỉ 1 dataset
#   TEACHER=deepseek_v4_flash bash scripts/run_overnight.sh
#
# Mỗi dataset: tắt reader -> dựng index (GPU cho E5) -> bật reader (FP8, lỗi thì AWQ) -> check_api
#   -> smoke test 10 câu (dừng nếu lỗi format nhiều) -> collect full. Cuối cùng export.
# Mọi bước đều resume được: index theo shard, collect theo câu. Log: $GRASP_DATA_DIR/logs/.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8   # console Windows cp1252 không in được tiếng Việt

DATASETS="${DATASETS:-2wiki hotpotqa nq}"
TEACHER="${TEACHER:-qwen3_235b_bedrock}"
SMOKE_N="${SMOKE_N:-10}"
SMOKE_MAX_BAD="${SMOKE_MAX_BAD:-1}"     # tối đa bao nhiêu câu parse_error/bad_request trong smoke test
READER_WAIT="${READER_WAIT:-1800}"     # giây chờ reader lên (lần đầu tải model ~15GB)

DATA_DIR="$(grep -E '^GRASP_DATA_DIR=' .env 2>/dev/null | cut -d= -f2-)"
DATA_DIR="${GRASP_DATA_DIR:-${DATA_DIR:-./data}}"
mkdir -p "$DATA_DIR/logs"
LOG="$DATA_DIR/logs/overnight-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

READER_CFG=""
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "DỪNG: $*"; log "log đầy đủ: $LOG"; stop_reader; exit "${2:-1}"; }
stop_reader() { docker rm -f grasp-reader >/dev/null 2>&1 || true; }

wait_reader() {  # $1 = model id mong đợi
  local t=0
  while (( t < READER_WAIT )); do
    if [[ "$(docker inspect -f '{{.State.Running}}' grasp-reader 2>/dev/null)" != "true" ]]; then
      log "reader container đã dừng:"; docker logs --tail 30 grasp-reader 2>&1 | sed 's/^/    /'
      return 1
    fi
    if curl -fsS http://localhost:8000/v1/models 2>/dev/null | grep -q "\"$1\""; then return 0; fi
    sleep 15; t=$((t + 15))
  done
  log "reader không lên sau ${READER_WAIT}s"; return 1
}

start_reader() {
  local mode model
  for mode in ${READER_MODES:-fp8 awq}; do
    [[ $mode == fp8 ]] && model=Qwen/Qwen2.5-7B-Instruct || model=Qwen/Qwen2.5-7B-Instruct-AWQ
    log "bật reader ($mode)..."
    DETACH=1 bash scripts/serve_reader.sh "$mode" >/dev/null || { log "docker run lỗi ($mode)"; continue; }
    if wait_reader "$model"; then
      READER_CFG="reader=reader_local_7b_$mode"; READER_MODES="$mode"   # dataset sau dùng luôn mode này
      log "reader sẵn sàng ($mode)"; return 0
    fi
    stop_reader
  done
  return 1
}

collect() {  # $1 = dataset, phần còn lại = override thêm
  local ds=$1; shift
  uv run python scripts/collect_sft.py "data=$ds" "teacher=$TEACHER" "$READER_CFG" "$@"
}

smoke_ok() {  # đọc summary sau smoke test
  uv run python - "$DATA_DIR/trajectories/summary/$1.json" "$SMOKE_MAX_BAD" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
c = s["status_counts"]; bad = c.get("parse_error", 0) + c.get("bad_request", 0)
n = sum(c.values()); usd = s["spent_total_usd"]
print(f"    smoke: {c} | đúng {s['correct']}/{n} | ${usd:.4f} (~${usd / max(n, 1):.4f}/câu)")
sys.exit(0 if s["exit_code"] == 0 and n > 0 and bad <= int(sys.argv[2]) else 1)
EOF
}

log "datasets=[$DATASETS] teacher=$TEACHER log=$LOG"
docker info >/dev/null 2>&1 || die "Docker Desktop chưa chạy"

for ds in $DATASETS; do
  log "===== $ds ====="
  stop_reader   # nhường GPU cho E5 khi encode
  log "dựng index $ds (đã có thì bỏ qua)"
  uv run python scripts/build_index.py "data=$ds" 2>&1 | { grep --line-buffered -v "Batches:" || true; }
  [[ ${PIPESTATUS[0]} -eq 0 ]] || die "build_index $ds lỗi"

  start_reader || die "không bật được reader (fp8 lẫn awq)" 3
  uv run python scripts/check_api.py "teacher=$TEACHER" "$READER_CFG" +check_reader=true \
    || die "check_api lỗi (key / model / reader)" 2

  marker="$DATA_DIR/trajectories/state/$ds.smoke_ok"
  if [[ ! -f "$marker" ]]; then
    log "smoke test $SMOKE_N câu"
    collect "$ds" "collect.limit=$SMOKE_N"; code=$?
    [[ $code -eq 0 ]] || die "smoke test $ds exit=$code" "$code"
    smoke_ok "$ds" || die "smoke test $ds không đạt (xem trajectories/$ds.jsonl, trường error)"
    touch "$marker"
  fi

  log "collect full $ds"
  for attempt in 1 2 3; do
    collect "$ds"; code=$?
    [[ $code -eq 3 ]] || break           # 3 = API/reader sập: nghỉ rồi thử lại, các mã khác dừng
    log "API/reader sập (lần $attempt), nghỉ 10 phút rồi chạy tiếp"; sleep 600
    [[ "$(docker inspect -f '{{.State.Running}}' grasp-reader 2>/dev/null)" == "true" ]] || start_reader \
      || die "reader chết và không bật lại được" 3
  done
  [[ $code -eq 0 ]] || die "collect $ds exit=$code (2 credit/key, 4 budget — xem docs/sft-data.md)" "$code"
  log "$ds xong"
done

stop_reader
log "export"
uv run python scripts/export_sft.py || die "export lỗi"
log "XONG. Kết quả: $DATA_DIR/sft/, tóm tắt: $DATA_DIR/trajectories/summary/"

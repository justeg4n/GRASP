# Sinh dữ liệu SFT — hướng dẫn chạy

Teacher **DeepSeek-V4-Flash** (FPT AI Factory) đóng vai 3 agent RD/QF/KI, reader **Qwen2.5-7B-Instruct**
chạy vLLM local, retriever **e5-large-v2**. Công thức theo SPARKLE: 3K câu × (HotpotQA, 2Wiki, NQ) →
7,5K train / 1,5K val, chỉ giữ trajectory trả lời đúng. Thiết kế: `docs/superpowers/specs/2026-10-15-sft-data-collection-design.md`.

## Chạy 1 lệnh, treo máy qua đêm

Sau khi xong mục 0 (`.env` có key, Docker Desktop đang mở):

```bash
bash scripts/run_overnight.sh          # 2wiki -> hotpotqa -> nq, teacher Bedrock; log ở $GRASP_DATA_DIR/logs/
```

Mỗi dataset: tắt reader → dựng index → bật reader (FP8, lỗi thì AWQ) → `check_api` → smoke test 10 câu (dừng nếu
>1 câu lỗi format) → collect full → cuối cùng export. Bị dừng giữa chừng: **chạy lại đúng lệnh**, mọi bước đều
tiếp tục từ chỗ dở. Tùy chọn: `DATASETS="2wiki"`, `TEACHER=deepseek_v4_flash`, `READER_MODES=awq`.
Thời gian ước tính: index HotpotQA ~4 giờ, wiki-18 (NQ) ~15 giờ.

Các mục dưới là từng bước thủ công (để gỡ rối).

## 0. Chuẩn bị (1 lần)

| Cần | Ghi chú |
|---|---|
| `uv` | `uv sync --extra dev` — kéo PyTorch CUDA 12.8 (bắt buộc cho RTX 50xx) |
| Docker Desktop (WSL2 backend) + driver NVIDIA mới | chạy vLLM reader; vLLM không chạy native trên Windows |
| ~100GB trống | corpus + embedding + index; đặt `GRASP_DATA_DIR` (vd. `D:/grasp-data`) |
| API key FPT AI Factory | <https://marketplace.fptcloud.com> |

```bash
uv sync --extra dev
cp .env.example .env        # điền FPT_API_KEY, GRASP_DATA_DIR
uv run python scripts/check_api.py      # key + model + giá hiện hành so với config
```

**Teacher qua Amazon Bedrock (dùng credit AWS)** — thêm `teacher=qwen3_235b_bedrock` vào *mọi* lệnh
`check_api.py` / `collect_sft.py`, điền `BEDROCK_API_KEY` trong `.env`.

- AWS **không** báo lỗi khi hết credit mà charge thẻ → `collect.budget_usd` là lớp chặn duy nhất. Giữ tổng budget
  3 dataset < credit còn lại (mặc định 3 × $40 = $120) và đặt AWS Budgets alert.
- Kiểm tra Billing → Credits xem credit có áp dụng cho model này không.
- Không trộn teacher giữa các dataset nếu không cần (mỗi trajectory vẫn ghi `teacher_model`).

Docker Desktop → Settings → Resources: cho WSL tối đa ~10GB RAM (máy 16GB) để còn RAM cho index.

## 1. Reader (terminal riêng, để chạy suốt)

```bash
bash scripts/serve_reader.sh fp8        # lỗi FP8 trên card -> bash scripts/serve_reader.sh awq
uv run python scripts/check_api.py +check_reader=true
```

Nếu dùng AWQ thì mọi lệnh bên dưới thêm `reader=reader_local_7b_awq`.

## 2. Câu hỏi + split

`splits/*.json` đã commit sẵn (seed 42) — **không tạo lại**. Lệnh dưới chỉ tải file câu hỏi về:

```bash
uv run python scripts/prepare_data.py
```

## 3. Index (chạy lâu, 1 lần, tự chạy tiếp nếu bị ngắt)

```bash
uv run python scripts/build_index.py data=2wiki      # ~0.4M passage (sau dedup) — chạy trước để thử
uv run python scripts/build_index.py data=hotpotqa   # ~5.2M (tải 1GB parquet BeIR/hotpotqa từ HF Hub)
uv run python scripts/build_index.py data=nq         # wiki-18 ~21M (tải 5GB), lâu nhất: chạy qua đêm
```

Tắt reader trong lúc encode để GPU dành hết cho E5 (nhanh hơn nhiều).

## 4. Smoke test → chạy thật

```bash
uv run python scripts/collect_sft.py data=2wiki collect.limit=10
cat $GRASP_DATA_DIR/trajectories/summary/2wiki.json
```

Chỉ chạy full khi `parse_error` < 5% và chi phí/câu hợp lý. Sau đó:

```bash
uv run python scripts/collect_sft.py data=2wiki
uv run python scripts/collect_sft.py data=hotpotqa
uv run python scripts/collect_sft.py data=nq
```

**Bị dừng giữa chừng (hết credit, mất mạng, Ctrl+C, tắt máy): chạy lại đúng lệnh cũ.** Câu đã xong được giữ,
câu đang chạy dở được làm lại từ đầu, triple đã trích không phải trả tiền lại.

| Exit code | Nghĩa | Làm gì |
|---|---|---|
| 0 | xong | — |
| 1 | lỗi code lặp lại | xem log, báo lại |
| 2 | hết credit / sai key / sai model | nạp credit hoặc sửa `.env`, chạy lại |
| 3 | API FPT hoặc reader sập kéo dài | kiểm tra reader / mạng, chạy lại |
| 4 | chạm budget (`collect.budget_usd`, mặc định $40/dataset) | tăng budget nếu muốn: `collect.budget_usd=60` |
| 130 | Ctrl+C | chạy lại |

Mỗi lần chạy ghi `trajectories/summary/{dataset}.json`: số câu theo trạng thái, tiền đã chi, lý do dừng.

## 5. Export → HF Hub

```bash
uv run python scripts/export_sft.py                                 # sft/train.jsonl, sft/val.jsonl, stats.json
uv run python scripts/export_sft.py +push_to_hub=<org>/grasp-sft    # dataset private
```

Mỗi dòng: `{qid, dataset, split, role, step, messages: [user=prompt, assistant=completion]}` — đúng format chat để
SFT proxy. Đổi ngưỡng lọc không tốn tiền: `collect.filter.f1_min=0.7`.

Ghi kết quả (số trajectory đạt, chi phí, commit) vào `research-notes/experiments/results.md`.

## Gỡ rối

- **`parse_error` cao**: xem `error` trong trajectory. Thường do model có reasoning token làm cạn `max_tokens` →
  tăng `teacher.client.max_tokens`.
- **Chạy lại các câu lỗi format sau khi sửa prompt**: `collect.retry_statuses=[parse_error]`.
- **Hết RAM khi chạy NQ**: index wiki-18 PQ64 ~1,4GB; giảm `collect.concurrency` và RAM của Docker.
- **Giá FPT đổi**: `check_api.py` sẽ cảnh báo; sửa `configs/teacher/deepseek_v4_flash.yaml` để budget tính đúng.

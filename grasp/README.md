# GRASP

Proxy model 3 agent (Retrieval Decision / Query Formulation / Knowledge Integration) điều khiển
adaptive RAG, huấn luyện bằng **GRPO** (không cần critic) với **group-structured rollout**.
LLM và retriever được đóng băng. Proxy: Qwen2.5-3B-Instruct full finetune (ADR-002).

## Cài đặt

```bash
uv sync --extra dev          # PyTorch CUDA 12.8 (cần cho RTX 50xx)
cp .env.example .env         # FPT_API_KEY, GRASP_DATA_DIR, …
pre-commit install
```

## Pipeline

```
scripts/prepare_data.py   # tải câu hỏi; split cố định đã commit ở splits/
scripts/build_index.py    # corpus -> passage store -> E5-large -> FAISS (data=hotpotqa|2wiki|nq)
scripts/serve_reader.sh   # reader Qwen2.5-7B qua vLLM (Docker)
scripts/run_overnight.sh  # chạy trọn index -> reader -> smoke -> collect -> export cho 3 dataset
scripts/check_api.py      # kiểm tra key / model / giá FPT AI Factory (+check_reader=true)
scripts/collect_sft.py    # teacher (FPT API) sinh trajectory, resume được, chặn hết credit
scripts/export_sft.py     # lọc trajectory đúng -> sft/{train,val}.jsonl (+push HF Hub)
scripts/train.py          # train=sft | grpo_g2 | grpo_g4
scripts/evaluate.py       # EM / F1 / Recall + profiler hiệu năng
```

Hướng dẫn sinh dữ liệu SFT từng bước: `docs/sft-data.md`.
Mọi thí nghiệm cấu hình bằng Hydra: `uv run python scripts/collect_sft.py data=2wiki collect.limit=10`

## Quy ước

- Nhánh: `feat/…`, `fix/…`, `exp/…`; PR cần 1 review; squash merge.
- Commit: Conventional Commits.
- Mỗi run phải log W&B kèm commit hash + tên config. Không có run "chạy tay không log".
- Dữ liệu / index / checkpoint **không** vào git (xem .gitignore) — dùng Hugging Face Hub.
  Ngoại lệ: `splits/*.json` (danh sách qid) được commit để mọi người dùng cùng tập câu.

## Tài liệu

- `docs/architecture.md` — kiến trúc 3 agent và vòng lặp suy luận
- `docs/hardware.md` — local sinh SFT, GPU thuê cho SFT/GRPO
- `docs/sft-data.md` — chạy pipeline sinh dữ liệu SFT
- `docs/reproducibility.md` — cách chạy lại từng bảng kết quả

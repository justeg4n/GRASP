# Phần cứng (ADR-002)

| Giai đoạn | Chạy ở đâu | Thành phần |
|---|---|---|
| Sinh dữ liệu SFT | máy local RTX 5060 Ti 16GB, RAM 16GB | reader Qwen2.5-7B FP8/AWQ (vLLM, Docker) + E5-large + FAISS IVF-PQ; teacher và trích triple qua API FPT |
| SFT + GRPO | GPU thuê, `hardware=1gpu` (mặc định) hoặc `2gpu` | proxy Qwen2.5-3B **full finetune**, reference model, reader bf16 |

## Local (sinh SFT)

| Tài nguyên | Dùng |
|---|---|
| VRAM | reader FP8 ~8GB + KV cache (vLLM `--gpu-memory-utilization 0.80`) + E5 query encoder ~0,7GB |
| RAM | 1 index IVF-PQ (wiki-18 ~1,4GB) + Python ~2GB + VM Docker cho vLLM; chạy **lần lượt từng dataset** |
| Đĩa | ~100GB: embedding fp16 ~54GB, passage SQLite ~15GB, index ~3GB, model ~25GB |

- vLLM không chạy native trên Windows → dùng Docker Desktop (WSL2 backend), xem `scripts/serve_reader.sh`.
- PyTorch phải là build CUDA ≥ 12.8 (đã cấu hình trong `pyproject.toml`).
- FP8 trên consumer Blackwell chưa được xác minh trên máy này: lỗi thì dùng AWQ (`reader=reader_local_7b_awq`).
- Encode corpus (~26M passage) mất khoảng 10–15 giờ; nên tắt reader lúc encode.

## GPU thuê (SFT + GRPO)

| Thành phần | VRAM xấp xỉ |
|---|---|
| Proxy 3B full FT (bf16 + grad + AdamW fp32) | ~48GB + activations |
| Reference model (đóng băng) | ~6GB |
| Reader 7B bf16 qua vLLM | ~16GB + KV cache |

- `1gpu` (80GB): tách pha rollout (vLLM) ↔ train trên cùng card, 8-bit Adam hoặc offload optimizer.
- `2gpu`: card 0 serve reader liên tục, card 1 train proxy + reference.

Nút nghẽn khi GRPO là số lần gọi reader: `n_questions * G * L`.

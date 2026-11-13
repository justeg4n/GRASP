"""Smoke test tuần 1: môi trường GPU có chạy được không (issue M0)."""

import torch

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
assert torch.cuda.is_available(), "Không thấy CUDA — kiểm tra driver / build CUDA 12.8+"
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"gpu: {name} | VRAM: {total:.1f} GB")
if total < 15:
    print("CẢNH BÁO: dưới 16GB — phải dùng proxy 0.5B và reader qua API")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).dtype)

# Chạy SFT một lần trên Vast.ai

## 1. Máy nên thuê

Cho QLoRA SFT `Qwen/Qwen2.5-3B-Instruct`, chọn **1 x RTX 4090 24 GB**:

- host verified, reliability từ 99%;
- RAM 64 GB (tối thiểu 32 GB), từ 8 vCPU;
- disk 50 GB nếu chỉ chạy SFT;
- download từ 300 Mbps, disk bandwidth từ 500 MB/s;
- image PyTorch, CUDA từ 12.1, Python 3.11;
- on-demand cho lần chạy đầu.

Không cần upload corpus retrieval, FAISS index, trajectories teacher, PDF hoặc base model.
Base model Qwen2.5-3B được tải trực tiếp từ Hugging Face khi chạy.

## 2. Tạo đúng một ZIP đầu vào trên Windows

Từ thư mục repo:

```powershell
Set-Location D:\FPTU\FPTU-SP26-CaptonesProject\grasp-proxy-rollout
powershell -ExecutionPolicy Bypass -File .\scripts\package_sft_vast_input.ps1 -Force
```

Script tạo:

```text
D:\FPTU\FPTU-SP26-CaptonesProject\vast-upload\grasp-sft-vast-input.zip
```

ZIP này chỉ chứa những gì cần cho SFT:

```text
grasp-sft-vast-input/
├── grasp-proxy-rollout/       # code, cấu hình, tests và scripts
└── data/grasp-sft/sft/
    ├── train.jsonl
    ├── val.jsonl
    └── stats.json
```

Chỉ upload file ZIP này lên `/workspace` của Vast.ai.

## 3. Giải nén và chạy

```bash
cd /workspace
python -m zipfile -e grasp-sft-vast-input.zip .

cd /workspace/grasp-sft-vast-input/grasp-proxy-rollout
chmod +x scripts/train_sft_vast.sh

PROJECT_DIR=/workspace/grasp-sft-vast-input/grasp-proxy-rollout \
SFT_DIR=/workspace/grasp-sft-vast-input/data/grasp-sft \
OUTPUT_DIR=/workspace/checkpoints/grasp-qwen2.5-3b-sft \
OUTPUT_ARCHIVE=/workspace/checkpoints/grasp-qwen2.5-3b-sft-handoff.zip \
  ./scripts/train_sft_vast.sh
```

Nên chạy bên trong `tmux` để terminal trình duyệt mất kết nối không làm mất tiến trình:

```bash
tmux new -s grasp-sft
# chạy command ở trên, sau đó Ctrl+B rồi D để detach
tmux attach -t grasp-sft
```

## 4. Theo dõi log

Script ghi ba nguồn theo dõi:

```text
/workspace/checkpoints/grasp-qwen2.5-3b-sft/logs/
├── console.log       # toàn bộ stdout/stderr
├── metrics.jsonl     # loss, grad_norm, learning_rate, epoch theo optimizer step
└── gpu_metrics.csv   # GPU utilization, VRAM, nhiệt độ, power mỗi 30 giây
```

Theo dõi trực tiếp:

```bash
tail -f /workspace/checkpoints/grasp-qwen2.5-3b-sft/logs/console.log
tail -f /workspace/checkpoints/grasp-qwen2.5-3b-sft/logs/metrics.jsonl
watch -n 2 nvidia-smi
```

`logging_steps=1` nghĩa là một log cho mỗi **effective batch/optimizer step**. Mỗi
effective batch gồm 16 micro-batch vì cấu hình `batch_size=1` và
`gradient_accumulation_steps=16`. Không log từng micro-batch riêng vì sẽ tạo 25.896 dòng
và làm chậm training mà không cung cấp thêm tín hiệu tối ưu hữu ích.

## 5. Early stopping và regularization

Cấu hình một-lần-chạy đã bật:

- tối đa 1 epoch;
- evaluate/save mỗi 250 optimizer steps trên 1.024 validation examples cố định;
- early stopping patience 3, threshold 0.001;
- tự load checkpoint có `eval_loss` thấp nhất ở cuối;
- LoRA dropout 0.05;
- weight decay 0.01;
- gradient clipping 1.0;
- warmup 3% và cosine learning-rate schedule;
- NF4 double quantization, BF16 và paged AdamW 8-bit.

Không dùng label smoothing vì output RD/QF/KI cần format và token chính xác. Không bật
`torch.compile` hoặc tự build FlashAttention trong lần chạy duy nhất vì rủi ro lỗi môi
trường cao hơn phần tốc độ có thể tiết kiệm.

## 6. Resume khi instance bị ngắt

Nếu `/workspace/checkpoints` vẫn còn:

```bash
RESUME_FROM_CHECKPOINT=latest \
PROJECT_DIR=/workspace/grasp-sft-vast-input/grasp-proxy-rollout \
SFT_DIR=/workspace/grasp-sft-vast-input/data/grasp-sft \
OUTPUT_DIR=/workspace/checkpoints/grasp-qwen2.5-3b-sft \
OUTPUT_ARCHIVE=/workspace/checkpoints/grasp-qwen2.5-3b-sft-handoff.zip \
  /workspace/grasp-sft-vast-input/grasp-proxy-rollout/scripts/train_sft_vast.sh
```

## 7. Output cần tải xuống

Trong mọi trường hợp, script cố tạo:

```text
/workspace/checkpoints/grasp-qwen2.5-3b-sft-handoff.zip
```

Nếu training thành công, ZIP chứa:

- `adapter_model.safetensors` và `adapter_config.json`;
- tokenizer;
- `training_manifest.json`;
- `data_validation.json` và `checkpoint_validation.json`;
- `train_results.json`, `eval_results.json`, `trainer_state.json`;
- toàn bộ `logs/`;
- `handoff_manifest.json` xác nhận adapter có đầy đủ hay không.

Smoke-test chạy theo batch 4 và giới hạn tổng sequence ở 3.072 token để không tạo đỉnh
VRAM bất ngờ trên RTX 4090 khi kiểm tra 32 mẫu cho mỗi vai trò.

Các thư mục `checkpoint-*` không được đưa vào ZIP handoff để giảm dung lượng. Chúng vẫn
nằm trong `OUTPUT_DIR` để resume khi instance còn tồn tại. Nếu run lỗi giữa chừng, ZIP
partial vẫn được tạo để tải log về điều tra; kiểm tra
`handoff_ready` trong `handoff_manifest.json` phải là `true` trước khi bàn giao model.
Giá trị này chỉ đúng khi adapter đầy đủ, có `training_manifest.json` và checkpoint đã vượt
kiểm tra format trên validation set. ZIP partial hoặc checkpoint không đạt vẫn giữ log nhưng
`handoff_ready` sẽ là `false`.

Chỉ xóa Vast instance sau khi đã tải ZIP về và kiểm tra trong ZIP có cả
`adapter_model.safetensors` lẫn `adapter_config.json`.

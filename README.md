# GRASP

GRASP (Group-Relative Agentic policy for Structured retrieval and knowledge integration in RAG Proxies) là dự án nghiên cứu proxy điều khiển truy xuất và tích hợp tri thức cho hệ thống Retrieval-Augmented Generation (RAG).

Proxy sử dụng ba vai trò: **RD** (Retrieval Decision), **QF** (Query Formulation) và **KI** (Knowledge Integration). Pipeline gồm thu thập dữ liệu từ teacher, huấn luyện SFT, huấn luyện GRPO và đánh giá với reader đóng băng.

Repository này tập hợp năm thành phần trong cùng một cây mã nguồn; mỗi thành phần có cấu hình và môi trường Python riêng.

## Cấu trúc

| Thư mục | Vai trò | Tài liệu |
| --- | --- | --- |
| `grasp/` | Pipeline RAG, thu thập dữ liệu SFT, cấu hình thí nghiệm và đánh giá | [README](grasp/README.md) |
| `grasp-proxy-rollout/` | Huấn luyện proxy SFT/QLoRA và xuất trajectory cho PPO/GRPO | [README](grasp-proxy-rollout/README.md) |
| `grasp-gold-evidence-builder/` | Ánh xạ gold evidence của benchmark sang passage ID và kiểm chứng dữ liệu | [README](grasp-gold-evidence-builder/README.md) |
| `grasp-grpo-trainer/` | Huấn luyện adapter SFT bằng GRPO với rollout dạng cây | [README](grasp-grpo-trainer/README.md) |
| `grasp-demo/` | Giao diện demo GRASP cùng các baseline RAG và IRCoT | [README](grasp-demo/README.md) |

## Luồng sử dụng

1. Dùng `grasp` để chuẩn bị dữ liệu, retrieval index và teacher trajectories.
2. Dùng `grasp-proxy-rollout` để huấn luyện adapter SFT và xuất rollout.
3. Dùng `grasp-gold-evidence-builder` để chuẩn bị nhãn evidence đã đối chiếu với retrieval corpus.
4. Dùng `grasp-grpo-trainer` để huấn luyện GRPO và đánh giá.
5. Dùng `grasp-demo` để trình diễn mô hình và so sánh các phương pháp.

## Cài đặt

Khuyến nghị Python **3.11 hoặc 3.12** để phù hợp với các package. Các bước huấn luyện cần GPU và bộ PyTorch/CUDA phù hợp; xem tài liệu từng thành phần trước khi cài.

```powershell
git clone https://github.com/justeg4n/GRASP.git
cd GRASP
```

Tạo môi trường riêng trong thành phần bạn cần, ví dụ với công cụ gold evidence:

```powershell
cd grasp-gold-evidence-builder
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[download,dev]"
.\.venv\Scripts\python.exe -m pytest
```

Không có một lệnh cài đặt chung cho toàn bộ repository. Các thành phần có thể yêu cầu phiên bản thư viện khác nhau; `grasp-proxy-rollout` và `grasp-grpo-trainer` còn cùng cung cấp lệnh `grasp-build-triple-cache`, vì vậy nên cài ở môi trường riêng.

## Chạy demo

Từ thư mục gốc repository:

```powershell
cd grasp-demo
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e "../grasp-grpo-trainer[rollout,quantization]"
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item models.yaml models.local.yaml
```

Thiết lập đường dẫn checkpoint, corpus và reader endpoint trong `models.local.yaml`. Tải các artifact theo [hướng dẫn demo](grasp-demo/README.md), rồi chạy:

```powershell
.\.venv\Scripts\grasp-demo.exe --config models.local.yaml
```

Mở <http://127.0.0.1:7860>. Chạy lệnh bằng `python`/`grasp-demo` sau khi kích hoạt môi trường tương ứng trên Linux.

## Cấu hình và dữ liệu

- Với `grasp`, sao chép `grasp/.env.example` thành `grasp/.env` và điền cấu hình cần dùng.
- Với GRPO trên Vast.ai, sao chép `grasp-grpo-trainer/config/vast.env.example` thành `vast.env` trong cùng thư mục. Điều chỉnh `PROJECT_DIR` theo vị trí clone thực tế.
- Với demo, giữ cấu hình riêng trong `grasp-demo/models.local.yaml`; API key có thể đặt trong `.env` của demo hoặc biến môi trường theo tài liệu thành phần.
- API key, token, môi trường ảo, model weights, retrieval index và dữ liệu tải về được loại khỏi Git. Một số artifact trên Hugging Face có thể cần tài khoản được cấp quyền và `hf auth login`.
- Giữ lại `grasp/splits/`, các file schema, `grasp-proxy-rollout/examples/` và `sample_output/` để tái lập và kiểm tra contract.
- Bộ handoff đã được chọn trong `grasp-gold-evidence-builder/outputs/grasp-gold-evidence/` vẫn được đưa vào Git theo whitelist của thành phần đó. `.gitattributes` giữ nguyên byte của bộ này để bảo toàn SHA256 trong manifest.

## Kiểm tra

Chạy `python -m pytest` trong từng thành phần sau khi cài extra `dev` của thành phần đó. Các lệnh huấn luyện, gọi API và chạy model cần thêm artifact hoặc dịch vụ bên ngoài; unit test không thay thế việc kiểm tra toàn bộ pipeline.

## Đưa thư mục Final lên GitHub

Xem [PUSH_TO_GITHUB.md](PUSH_TO_GITHUB.md) để commit và push lên `justeg4n/GRASP`, cùng cách xử lý khi remote đã có commit.

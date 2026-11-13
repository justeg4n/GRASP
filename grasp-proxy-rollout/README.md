# GRASP Proxy: JSON SFT and PPO/GRPO Rollout

Repository này thực hiện hai việc liên tiếp cho proxy `Qwen/Qwen2.5-3B-Instruct`:

1. train QLoRA từ các bản ghi chat JSONL do teacher tạo;
2. load checkpoint SFT vừa tạo và xuất trajectory JSON đầy đủ cho bước PPO/GRPO.

Repository **không cập nhật actor bằng PPO/GRPO**. Nó tạo checkpoint ban đầu và dữ
liệu rollout on-policy để bàn giao cho phần RL.

## JSON không phải checkpoint

Dataset private `doancuavahk/grasp-sft` chứa:

```text
grasp-sft/
├── sft/train.jsonl       # 25,896 teaching examples
├── sft/val.jsonl         # 5,275 held-out examples
└── trajectories/*.jsonl  # teacher traces và candidate triples
```

Mỗi dòng SFT là một object độc lập:

```json
{
  "qid": "...",
  "dataset": "2wiki",
  "split": "sft_train",
  "role": "rd",
  "step": 0,
  "messages": [
    {"role": "user", "content": "...exact task prompt..."},
    {"role": "assistant", "content": "No (retrieval is not required)"}
  ]
}
```

Các JSONL chỉ chứa input/target text, không có tensor trọng số. Training biến chúng
thành `adapter_model.safetensors` + `adapter_config.json`; cặp file đó mới là PEFT
checkpoint có thể load cùng base model.

Validator đã được đối chiếu với toàn bộ 31.171 mẫu phát hành. Ba output contract là:

- RD: `Yes (retrieval is required)` hoặc `No (retrieval is not required)`;
- QF: đúng ba dòng `Reasoning Chain:`, `Knowledge Gap:`, `Query:`;
- KI: copy đúng một triple dạng `<head; relation; tail>` từ danh sách candidate.

Format cũ RD=`Yes/No`, QF=JSON object và KI=index số **không khớp dữ liệu SFT này**.

Metadata của 9.000 teacher trajectories ghi `teacher_model` là
`qwen.qwen3-235b-a22b-2507` và `reader_model` là
`Qwen/Qwen2.5-7B-Instruct`; artifact hiện tại không ghi model 72B.

## Cài đặt

Python 3.11 hoặc 3.12:

```bash
python -m venv .venv
source .venv/bin/activate            # Linux
python -m pip install -U pip
pip install -e ".[train,rollout,dev]"
```

Trên PowerShell, activate bằng `.venv\Scripts\Activate.ps1`.

Tải dataset private sau khi `hf auth login`:

```bash
hf download doancuavahk/grasp-sft \
  --repo-type dataset \
  --local-dir data/grasp-sft

hf download doancuavahk/grasp-retrieval \
  --repo-type dataset \
  --local-dir data/grasp-retrieval
```

## 1. Validate và train SFT từ JSONL

Luôn validate trước khi thuê GPU lâu:

```bash
grasp-validate-sft-data \
  data/grasp-sft/sft/train.jsonl \
  data/grasp-sft/sft/val.jsonl \
  --output outputs/sft_data_validation.json
```

Train QLoRA một epoch:

```bash
grasp-train-sft \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --train-file data/grasp-sft/sft/train.jsonl \
  --validation-file data/grasp-sft/sft/val.jsonl \
  --output-dir checkpoints/grasp-qwen2.5-3b-sft \
  --max-length 3072 \
  --batch-size 1 \
  --gradient-accumulation-steps 16 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --weight-decay 0.01 \
  --lora-dropout 0.05 \
  --max-eval-samples 1024 \
  --eval-steps 250 \
  --save-steps 250 \
  --logging-steps 1 \
  --early-stopping-patience 3 \
  --early-stopping-threshold 0.001 \
  --use-4bit \
  --compute-dtype bf16
```

Pipeline dùng NF4 4-bit + double quantization, LoRA trên toàn bộ projection linear,
gradient checkpointing và chỉ tính loss trên token target của assistant. Prompt được
render bằng chat template chính thức của tokenizer Qwen. Nếu prompt dài quá 3072 token,
phần đầu và cuối prompt được giữ lại còn toàn bộ target vẫn được bảo toàn.
Mặc định trên Vast còn dùng weight decay 0.01, LoRA dropout 0.05, gradient clipping,
warmup + cosine schedule và early stopping. `logs/metrics.jsonl` lưu loss theo từng
optimizer step; `logs/gpu_metrics.csv` lưu GPU telemetry mỗi 30 giây.

Resume an toàn từ checkpoint mới nhất:

```bash
grasp-train-sft <các tham số như trên> --resume-from-checkpoint latest
```

Smoke-test checkpoint trên held-out data:

```bash
grasp-validate-checkpoint \
  --checkpoint checkpoints/grasp-qwen2.5-3b-sft \
  --data-file data/grasp-sft/sft/val.jsonl \
  --samples-per-role 32 \
  --batch-size 4 \
  --max-sequence-length 3072 \
  --minimum-format-rate 0.95 \
  --load-in-4bit \
  --output outputs/checkpoint_validation.json
```

Hướng dẫn tạo đúng một ZIP input, chạy, theo dõi log và tải ZIP output trên Vast.ai nằm
ở [docs/VAST_AI.md](docs/VAST_AI.md).

## 2. Chuẩn bị retrieval đúng artifact phát hành

Mỗi corpus phát hành có layout:

```text
2wiki_ircot/
├── passages.sqlite
└── e5-large-v2/index.faiss
```

Searcher tự nhận diện layout này; truyền thư mục corpus, không cần di chuyển file.
Các triple trong teacher trajectories có thể gom thành cache SQLite:

```bash
grasp-build-triple-cache \
  --trajectories data/grasp-sft/trajectories/2wiki.jsonl \
  --output data/2wiki-triples.sqlite

grasp-build-triple-cache \
  --trajectories data/grasp-sft/trajectories/hotpotqa.jsonl \
  --output data/hotpotqa-triples.sqlite

grasp-build-triple-cache \
  --trajectories data/grasp-sft/trajectories/nq.jsonl \
  --output data/wiki18-triples.sqlite
```

Cache này chỉ chứa các candidate triple đã xuất hiện trong teacher traces. Với rollout
nghiêm ngặt, dùng `--require-triples` để dừng nếu toàn bộ retrieval result không có
candidate triple trong cache, thay vì âm thầm đổi sang fallback passage text.
Giữ cache riêng cho từng corpus để các `doc_id` trùng tên giữa corpus không trộn triple.

## 3. Xuất rollout JSON cho PPO/GRPO

Reader teacher trong artifact là `Qwen/Qwen2.5-7B-Instruct`; endpoint phải tương thích
OpenAI API. Một smoke run:

```bash
grasp-export-rollouts \
  --checkpoint checkpoints/grasp-qwen2.5-3b-sft \
  --qa data/grasp-sft/trajectories/2wiki.jsonl \
  --index-dir data/grasp-retrieval/2wiki_ircot \
  --triple-cache data/2wiki-triples.sqlite \
  --reader-url http://localhost:8000/v1 \
  --reader-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir outputs/sft_initial_rollouts \
  --group-size 4 \
  --limit 1 \
  --load-in-4bit
```

Trước thought đầu tiên, environment retrieve top-10 bằng câu hỏi gốc để seed context.
Đây là cách teacher data được tạo: initial retrieval không phải action của proxy và không
tính vào retrieval cost. Có thể tắt bằng `--no-initial-retrieval` cho ablation, nhưng khi
đó runtime distribution không còn khớp SFT.

Output contract v2:

```text
outputs/sft_initial_rollouts/
├── manifest.json
├── summary.json
├── groups.json
├── trajectory_schema.json
└── trajectories_00000.json
```

Mỗi proxy action lưu exact prompt/response, parsed action, token IDs và old-policy
token log-probabilities. Mỗi trajectory lưu initial context, các lần context chuyển đổi,
reward components và standardized group advantage. [sample_output](sample_output) là ví
dụ synthetic đúng schema, không phải kết quả model thật.

## Quy tắc on-policy bắt buộc

`old_token_logprobs` chỉ hợp lệ cho actor được ghi trong
`behavior_policy_checkpoint`. Sau mỗi lần PPO/GRPO cập nhật actor, phải rollout lại bằng
actor mới. Không được dùng bộ JSON ban đầu cho mọi epoch RL.

PPO value/return/GAE không được xuất vì chúng phải do critic hiện tại tính. GRPO trainer
có thể tự tính lại group advantage từ raw rewards để kiểm tra consistency.

## Kiểm tra code

```bash
ruff check .
pytest
```

Phân chia trách nhiệm:

- repo này: validate JSONL, QLoRA SFT, load checkpoint, RD/QF/KI prompts/parsers,
  retriever, reward, grouped rollout và JSON schema;
- repo PPO/GRPO: clone SFT actor thành frozen reference, tạo critic nếu PPO, cập nhật
  actor và thu rollout mới sau mỗi policy update.

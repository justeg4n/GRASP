# Thiết kế: Pipeline sinh dữ liệu SFT qua FPT AI Factory

- **Ngày:** 2026-10-15
- **Phạm vi:** chỉ sinh dữ liệu SFT (trajectory của teacher). Train SFT/GRPO, baseline, cấu hình GPU thuê nằm ngoài spec này.
- **Thay thế giả định của:** `docs/hardware.md`, `research-notes/decisions/ADR-001` (xem ADR-002).

## 1. Bối cảnh và quyết định đã chốt

| Hạng mục | Trước | Nay |
|---|---|---|
| Proxy | 1.5B LoRA (thu nhỏ cho 5060 Ti) | **Qwen2.5-3B-Instruct full finetune** trên GPU thuê |
| Thuật toán RL | GRPO + PPO baseline | **Chỉ GRPO**. Bỏ PPO, RQ2/RQ6 viết lại |
| Baseline | PPO proxy + RAG thường | RAG không RL (teammate quyết, ngoài phạm vi) |
| GPU train | 5060 Ti 16GB | Thuê; config hỗ trợ `hardware=1gpu` (mặc định) và `2gpu` |
| Teacher SFT | Qwen2.5-72B qua API chưa xác định | **DeepSeek-V4-Flash** qua FPT AI Factory |
| Trích triple | LLM local, offline toàn corpus | **DeepSeek-V4-Flash** qua API, lazy theo doc được retrieve |
| Reader | 7B-AWQ | **Qwen2.5-7B-Instruct**, vLLM local: FP8, fallback AWQ |
| Retriever | E5-base, corpus từ dataset | **intfloat/e5-large-v2**, top-10, corpus theo SPARKLE |
| Quy mô | 1k/500 câu | Theo SPARKLE (mục 3) |

Công thức dữ liệu bám SPARKLE (Fang et al., ACL 2026, Appendix D.3): 3.000 câu/dataset từ tập train
HotpotQA, 2Wiki, NQ → 7.500 train / 1.500 val; teacher sinh 1 trajectory/câu; chỉ giữ trajectory
có đáp án cuối đúng. Khác biệt duy nhất có chủ ý: teacher (DeepSeek-V4-Flash thay Qwen2.5-72B) và
model trích triple (DeepSeek-V4-Flash thay Llama-3-8B).

### FPT AI Factory

- Endpoint OpenAI-compatible: `https://mkp-api.fptcloud.com/v1` (danh sách model public tại `/v1/models`).
- `DeepSeek-V4-Flash`: $0.14/M input, $0.28/M output, context 500K, hỗ trợ `response_format`.
- **Không model nào hỗ trợ `logprobs`** → mọi thứ cần logprob (GRPO) chạy trên proxy local / GPU thuê.
- Không có Qwen2.5-7B-Instruct bản text → reader bắt buộc tự serve.

## 2. Kiến trúc

```
          ┌─────────── máy local (WSL2) ────────────┐        ┌── FPT API ──┐
questions │ Searcher (E5 GPU + FAISS IVF-PQ CPU)     │        │ DeepSeek-   │
 (3×3K) ─►│ Reader  Qwen2.5-7B FP8|AWQ (vLLM :8000)  │◄──────►│ V4-Flash    │
          │ TripleCache (SQLite, theo doc_id)        │        │ (teacher +  │
          │ run_episode(proxy=ApiProxyBackend) ──────┼───────►│  extractor) │
          └──────────────┬───────────────────────────┘        └─────────────┘
                         ▼
     trajectories/{dataset}.jsonl   (mọi trajectory, kể cả sai / lỗi)
                         ▼  lọc
     sft/{train,val}.jsonl          (1 dòng = 1 cặp prompt/completion theo role)
                         ▼
     HF Hub (dataset private) → GPU thuê kéo về khi SFT
```

Nguyên tắc chính: **vòng lặp sinh SFT, eval và GRPO dùng chung một code** (`pipeline/loop.py` + 3 agent +
prompt). Proxy là một backend thay được; nhờ đó format mà SFT dạy chính là format GRPO parse.

### Thành phần

| Unit | File | Trách nhiệm | Phụ thuộc |
|---|---|---|---|
| `LLMClient` | `llm/client.py` | AsyncOpenAI + semaphore, retry 429/5xx/timeout, đếm token & chi phí | openai |
| `ProxyBackend` (Protocol) | `llm/proxy.py` | `generate(prompt, n, role) -> list[Generation]` | — |
| `ApiProxyBackend` | `llm/proxy.py` | Teacher qua `LLMClient`, `logprob=None` | `LLMClient` |
| `ProxyModel` | `llm/proxy.py` | Bản local có logprob — **giữ stub**, làm ở giai đoạn GRPO | — |
| Agents RD/QF/KI | `agents/*.py`, `agents/prompts/*.txt` | Build prompt, parse output; prompt chuyển thể từ SPARKLE Appendix A | `ProxyBackend` |
| `PassageStore` | `retrieval/store.py` | `doc_id -> Passage` từ SQLite trên đĩa, không nạp RAM | sqlite3 |
| `Searcher` | `retrieval/searcher.py` | Encode query (prefix `query: `), FAISS IVF-PQ top-100, re-score bằng embedding gốc (mmap), trả top-10 | faiss, sentence-transformers |
| Index builder | `retrieval/index.py` | Encode corpus theo shard, lưu embedding fp16 (`.npy` mmap), train + build IVF-PQ | |
| `TripleCache` | `kg/extract.py` | Khoá `(doc_id, extractor_model)`; miss → gọi API, ghi SQLite | `LLMClient` |
| Triple ranking | `kg/extract.py` | Rank triple của các doc retrieve bằng E5, giữ top-20 cho KI | `Searcher` encoder |
| Reasoning chain | `kg/chain.py` | Thought → triple chain (do QF agent tạo, theo SPARKLE) | |
| Loop | `pipeline/loop.py` | `run_episode` / `step_once`, tối đa `max_steps` | tất cả trên |
| Metrics | `eval/metrics.py` | EM / F1 (chuẩn hoá SQuAD), context recall | |
| Collector | `scripts/collect_sft.py` | Chạy loop cho 1 dataset, resume, budget, ghi jsonl | |
| Filter/export | `scripts/export_sft.py` | Lọc trajectory → cặp prompt/completion theo role, push HF Hub | datasets |

Một bước suy luận (giữ đúng `docs/architecture.md`):

```
reader sinh thought
  -> RD (teacher): retrieve? Yes/No
       Yes -> QF (teacher): KG chain + query -> Searcher top-10
            -> TripleCache: triple của 10 doc -> rank E5 -> top-20
            -> KI (teacher): chọn 1 triple -> nối doc nguồn vào context
reader: thought tiếp / đáp án cuối
```

## 3. Dữ liệu

### Split (`scripts/prepare_data.py`, seed=42)

| Split | Mỗi dataset | Tổng | Dùng cho |
|---|---|---|---|
| `sft_train` | 2.500 | 7.500 | Sinh trajectory → SFT |
| `sft_val` | 500 | 1.500 | Sinh trajectory → val SFT, chọn hyperparameter |
| `rl_pool` | 3.000 | 9.000 | Dành cho GRPO, **chưa gọi API**. Sau SFT lọc giữ câu có độ khó vừa (tránh group toàn đúng / toàn sai) |

Ba split không giao nhau. Danh sách `qid` lưu ở `splits/{dataset}.json` và **commit vào git**.

### Corpus và index

| Dataset | Corpus | Quy mô xấp xỉ |
|---|---|---|
| HotpotQA | Wikipedia abstracts 2017 do tác giả HotpotQA phát hành | ~5,2M |
| 2Wiki | Dựng theo quy trình IRCoT (Trivedi et al., 2023) | ~0,4M |
| NQ | wiki-18 DPR (Karpukhin et al., 2020) | ~21M |

- Index: FAISS `IVF{nlist},PQ64`, nprobe cấu hình được. Truy vấn lấy top-100 rồi re-score bằng tích vô hướng trên
  embedding fp16 gốc (mmap từ đĩa) → top-10. Bù phần recall mất do PQ.
- Mỗi lần chạy chỉ nạp **một** index (chạy lần lượt từng dataset).
- Embedding, index, passage store đẩy lên HF Hub (private) để GPU thuê dùng lại đúng index này cho GRPO.

### Format đầu ra

`trajectories/{dataset}.jsonl` — 1 dòng/câu:
`qid, dataset, split, question, gold_answers, steps[], context_doc_ids, answer, em, f1, status
(ok | parse_error | bad_request | abandoned), teacher_model, reader_model, usage{in,out,usd}, git_commit`.

`sft/{train,val}.jsonl` — 1 dòng/lượt gọi agent trong trajectory đạt:
`qid, dataset, role (rd|qf|ki), prompt, completion`. Prompt được render bởi **chính** `build_prompt` của agent.

Tiêu chí giữ: `status == ok` và (`em == 1` hoặc `f1 >= 0.8`). Ngưỡng nằm trong config; lọc lại không cần gọi API.

## 4. Cấu hình

Config Hydra mới / sửa:

- `configs/data/{hotpotqa,2wiki,nq}.yaml` — nguồn câu hỏi, corpus, đường dẫn index, file split.
- `configs/retriever/e5_large.yaml` — `intfloat/e5-large-v2`, `top_k: 10`, `index_type: ivfpq`, `candidates: 100`, `rescore: true`.
- `configs/reader/reader_local_7b_fp8.yaml` (mặc định) và sửa `reader_local_7b_awq.yaml` (fallback).
- `configs/teacher/deepseek_v4_flash.yaml` — `base_url: https://mkp-api.fptcloud.com/v1`, `model: DeepSeek-V4-Flash`,
  `api_key: ${oc.env:FPT_API_KEY}`, `concurrency`, giá input/output để tính chi phí.
- `configs/collect/sft.yaml` — `max_steps: 5`, `triples_top_k: 20`, `filter: {em_min: 1, f1_min: 0.8}`, `budget_usd`
  (budget cứng **cho mỗi dataset**, mặc định $40, cộng dồn qua các lần chạy).
- `.env.example`: thêm `FPT_API_KEY`, bỏ `TEACHER_*` cũ.
- `.gitignore`: bỏ các dòng `config.yaml`, `experiment/`, `model/`, `reader/`, `retriever/`, `train/`
  (đang khiến config mới bị ignore âm thầm); chỉ ignore thư mục dữ liệu ở gốc repo (`/data/`, …).
- `scripts/serve_reader.sh`: nhận chế độ `fp8|awq`, `--gpu-memory-utilization` mặc định 0.75 (chừa ~0,7GB cho E5 query encoder).
- `docs/hardware.md`: viết lại cho bố cục mới (local sinh SFT, GPU thuê train).

## 5. Môi trường local

- Máy: RTX 5060 Ti 16GB (Blackwell sm_120), RAM 15,8GB, ổ D trống ~250GB.
- Pipeline Python chạy native trên Windows; **reader vLLM chạy trong Docker Desktop** (WSL2 backend) vì vLLM
  không hỗ trợ Windows native. Giới hạn RAM của VM Docker ~10GB.
- Cần PyTorch/vLLM build CUDA ≥ 12.8. **Việc đầu tiên trong kế hoạch là xác minh vLLM serve được Qwen2.5-7B FP8,
  rồi AWQ, trên máy này.** Nếu cả hai đều thất bại: thuê GPU vài giờ cho bước sinh SFT (FPT không có reader thay thế).
- Ngân sách tài nguyên: VRAM ~8GB reader FP8 + KV cache + 0,7GB E5; RAM ~3–4GB vLLM + ≤1,5GB index + ~2GB Python;
  đĩa ~100GB (embedding fp16 ~54GB, passage ~15GB, index ~3GB, model ~25GB).
- Build index: encode ~26M passage, ước tính 10–15 giờ, theo shard, chạy tiếp được nếu bị ngắt.

## 6. Xử lý lỗi

### Phân loại lỗi

| Loại | Dấu hiệu | Xử lý |
|---|---|---|
| Tạm thời (theo request) | 429, 500/502/503/504, timeout, mất kết nối | Exponential backoff + jitter, tôn trọng `Retry-After`, tối đa 6 lần |
| Lỗi của câu hỏi | 400 (prompt quá dài, request sai) | Không retry; câu đó `status=bad_request` |
| Toàn cục: hết credit / sai key | 401, 402, 403, hoặc message chứa `insufficient`/`balance`/`quota`/`credit` | **Dừng toàn bộ ngay** (circuit breaker), không đánh dấu câu nào lỗi |
| Toàn cục: API sập kéo dài | ≥ 10 câu liên tiếp hết lượt retry | Dừng toàn bộ như trên |
| Lưới an toàn | Lỗi 4xx lạ lặp ở 5 câu khác nhau liên tiếp | Coi là toàn cục, dừng |
| Teacher sai format | Parser không đọc được | Retry 1 lần temperature thấp hơn; vẫn sai → `status=parse_error` |
| Reader vLLM | Lỗi kết nối kéo dài | Như API sập: dừng toàn cục |

### Resume

- Trạng thái kết thúc (ghi vào `trajectories/{dataset}.jsonl`, resume bỏ qua): `ok` (đúng hay sai đều là `ok`),
  `parse_error`, `bad_request`.
- Câu bị gián đoạn vì lỗi toàn cục: **không ghi gì**; trajectory dở bị bỏ, lần chạy sau làm lại từ đầu câu đó.
- `state/{dataset}.json` đếm `attempts` mỗi `qid`; gián đoạn quá 3 lần → `status=abandoned` (ghi kết thúc).
- Không checkpoint từng bước: mất tối đa ~concurrency × $0.005 mỗi lần dừng; triple đã trích nằm vĩnh viễn trong `TripleCache`.

### Chặn hết credit

- Preflight lúc khởi động: 1 request tối thiểu để kiểm tra key/credit; reader health check.
- Ước tính trước khi chạy: chi phí/câu (từ các trajectory đã có, hoặc giá trị mặc định trong config) × số câu còn lại;
  vượt `budget_usd` thì từ chối chạy trừ khi `force=true`.
- Budget cứng: cộng dồn từ `usage` × giá trong config; chạm `budget_usd` → dừng gọn như lỗi toàn cục.
- FPT chưa có endpoint xem số dư → budget local là lớp bảo vệ chính; nên nạp dư ~30%.

### Khi dừng

Huỷ task đang chạy, flush output, ghi `summary/{dataset}.json` (số câu theo trạng thái, chi phí, lý do dừng, lệnh chạy tiếp).
Exit code: `0` xong, `2` hết credit / sai key, `3` API hoặc reader sập, `4` hết budget.

## 7. Chi phí ước tính

- Teacher: ~9.000 trajectory × (~30K input + ~2K output) ≈ 270M in / 18M out ≈ **$43**.
- Trích triple: ~270K doc × (~250 in + ~150 out) ≈ **$20**.
- Tổng dự kiến dưới **$70**; `budget_usd` mặc định $40 mỗi dataset (tối đa $120 cho cả 3).

## 8. Kiểm thử

- **Unit test (CI, CPU, không mạng):** EM/F1 theo ca SQuAD chuẩn; parser 3 role (output hợp lệ, hỏng, thiếu trường);
  `TripleCache` hit/miss; resume theo `qid`; fake server giả lập 429→ok, 402 giữa chừng (dừng, không ghi câu dở, resume đúng câu), 500 kéo dài (dừng sau 10 câu), vượt budget; bộ lọc + export (fake backend thay LLM); split không giao nhau và
  tái lập theo seed; `Searcher` trên index tí hon (vài trăm vector ngẫu nhiên).
- **Smoke test thủ công:** 10 câu/dataset với API + reader thật. Ghi tỉ lệ `parse_error`, tỉ lệ trajectory đạt,
  chi phí/câu, thời gian/câu. Chỉ chạy full khi `parse_error` < 5%.
- Kết quả smoke và full run ghi vào `research-notes/experiments/results.md`.

## 9. Ngoài phạm vi

Train SFT / GRPO, `ProxyModel` local, rollout có branching, baseline, eval trên test set, thesis, cấu hình GPU thuê.

## 10. Ghi chú triển khai

- Prompt RD/QF/KI/reader/trích triple được **tự viết** theo mô tả công khai ở Appendix A của paper SPARKLE, giữ
  cùng format output (`Yes/No (retrieval is …)`, `<head; relation; tail>`, `<think>/<answer>`). Repo
  `jyfang6/sparkle` không có license nên không sao chép prompt hay code từ đó.
- Hành vi vòng lặp theo code SPARKLE: retrieve ban đầu bằng câu hỏi; điểm doc = max điểm triple, doc chứa triple
  được KI chọn +0.5; context = 10 doc điểm cao nhất; hết bước mà chưa có `<answer>` thì reader sinh lại trọn vẹn.
- Nguồn dữ liệu: câu hỏi từ `RUC-NLPIR/FlashRAG_datasets` (2Wiki ở đó là tập con 15K câu train, đủ cho 6K câu
  cần lấy); corpus 2Wiki dựng từ file gốc `voidful/2WikiMultihopQA`; wiki-18 từ bản jsonl của FlashRAG.
- Corpus 2Wiki **lệch nhẹ so với IRCoT**: `train.json` và dev/test chứa cùng paragraph nhưng khác khoảng trắng
  (`War( Wojna` vs `War (Wojna`) → 31.713 nhóm trùng (~7%) làm top-k toàn bản sao. Gộp theo khóa bỏ khoảng trắng,
  giữ bản ít khoảng trắng lạ nhất: 430.225 → **398.350** passage.
- Teacher thay thế qua Amazon Bedrock (`teacher=qwen3_235b_bedrock`, Qwen3 235B A22B 2507, dùng credit AWS). Bedrock
  trả sau → `collect.budget_usd` là lớp chặn chi tiêu duy nhất.
- Corpus HotpotQA lấy bản parquet `BeIR/hotpotqa` trên HF Hub (cùng 5.2M abstract enwiki-20171001, cùng id) thay
  cho tar.bz2 gốc ở nlp.stanford.edu (~100KB/s). Bỏ thẻ `<a href>` sót lại và 94 đoạn rỗng.
- Completion SFT là output đã parse rồi render lại về format chuẩn (`AgentCall.completion`); output gốc của teacher
  giữ ở `AgentCall.raw`.


# Chạy lại kết quả

| Bảng / Hình | Lệnh |
|---|---|
| Baseline | `(trong repo grasp-baselines)` |
| RQ1 kết quả chính | `make grpo && make eval` |
| RQ2 hiệu năng | `python scripts/train.py +experiment=rq2_efficiency train=grpo_g4` rồi `train=ppo` |
| RQ3 group size | `train=grpo_g2 / grpo_g4 / grpo_g8` |
| RQ4 generalisation | `python scripts/evaluate.py +experiment=rq4_generalisation` |

Mỗi bảng ghi kèm: commit hash, link W&B run, ngày chạy — vào `research-notes/experiments/`.

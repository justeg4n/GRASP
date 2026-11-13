# Kiến trúc

## Multi-agent MDP

| Agent | State | Action |
|---|---|---|
| RD | (q, context, thoughts) | Yes / No |
| QF | (q, thoughts) | KG reasoning chain + query |
| KI | (q, chain, candidate triples) | 1 triple -> nối doc nguồn vào context |

Ba agent **dùng chung một proxy model**, phân biệt bằng prompt theo role (`agents/prompts/*.txt`).
Reader (LLM) và retriever đóng băng hoàn toàn.

## Luồng một bước

```
reader sinh thought
   -> RD: có cần retrieve?
        No  -> bước tiếp
        Yes -> QF: chain + query -> retriever top-k
                 -> KI: chọn triple -> nối doc nguồn vào context
```

## Training vs Inference

- Training: group rollout (G nhánh ở L bước đầu) -> reward -> MC decomposition -> group advantage -> GRPO.
- Inference: đi 1 đường, không branching, không grouping.

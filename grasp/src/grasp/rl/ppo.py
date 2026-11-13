"""PPO baseline cùng kiến trúc — chỉ để so sánh ở RQ2/RQ6. Có critic + GAE."""

from __future__ import annotations


class PPOTrainer:
    def __init__(self, proxy, value_head, cfg): ...

    def step(self, batch) -> dict:
        raise NotImplementedError("TODO: GAE, loss policy + value, đo thêm VRAM của critic")

"""GRPO: advantage = (R - mean(group)) / (std(group) + eps). KHÔNG có value network."""

from __future__ import annotations


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    raise NotImplementedError("TODO: nếu std == 0 thì advantage = 0 và log cảnh báo")


class GRPOTrainer:
    """Clipped surrogate + KL tới policy tham chiếu (bản SFT, đóng băng)."""

    def __init__(self, proxy, ref_proxy, cfg): ...

    def step(self, groups) -> dict:
        raise NotImplementedError("TODO: ratio pi/pi_old, clip, trừ beta*KL, trả metrics cho W&B")

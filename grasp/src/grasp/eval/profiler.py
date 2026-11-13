"""Đo phần hiệu năng cho RQ2: cái phân biệt GRASP với PPO baseline."""

from __future__ import annotations


def count_trainable_params(model) -> int: ...


class RunProfiler:
    """Theo dõi peak VRAM, thời gian/step, số lần gọi reader & retriever."""

    def __init__(self): ...
    def __enter__(self): ...
    def __exit__(self, *exc): ...
    def report(self) -> dict:
        raise NotImplementedError("TODO: torch.cuda.max_memory_allocated, time/step, call counts")

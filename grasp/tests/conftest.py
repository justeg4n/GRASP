"""Fake dùng chung: không gọi mạng, không cần torch/GPU."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pytest

from grasp.llm.client import Generation


class HashEncoder:
    """Bag-of-words băm vào 64 chiều, chuẩn hoá L2 — đủ để 'query giống passage' cho điểm cao."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_queries(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_passages(self, texts, show_progress=False):
        return np.stack([self._vec(t) for t in texts])


class ScriptedProxy:
    """Trả lời theo role từ danh sách kịch bản; ghi lại prompt để kiểm tra."""

    def __init__(self, scripts: dict[str, list[str]]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.prompts: list[tuple[str, str]] = []

    async def generate(self, prompt, n=1, role="rd", temperature=None, usage=None):
        self.prompts.append((role, prompt))
        return [Generation(self.scripts[role].pop(0))]


@pytest.fixture
def encoder():
    return HashEncoder()

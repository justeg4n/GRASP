"""Proxy backend: thứ đóng vai 3 agent RD/QF/KI.

- ApiProxyBackend: teacher qua API (sinh dữ liệu SFT). Không có logprob.
- ProxyModel: model local được train (GRPO) — cần logprob theo token. Làm ở giai đoạn train.
"""

from __future__ import annotations

from typing import Protocol

from grasp.llm.budget import Usage
from grasp.llm.client import Generation, LLMClient


class ProxyBackend(Protocol):
    async def generate(
        self,
        prompt: str,
        n: int = 1,
        role: str = "rd",
        temperature: float | None = None,
        usage: Usage | None = None,
    ) -> list[Generation]: ...


class ApiProxyBackend:
    """Teacher đóng vai cả 3 agent. Giới hạn max_tokens theo role để chặn output lan man."""

    def __init__(self, client: LLMClient, max_tokens: dict[str, int] | None = None):
        self.client = client
        self.max_tokens = dict(max_tokens or {})

    async def generate(
        self, prompt, n=1, role="rd", temperature=None, usage=None
    ) -> list[Generation]:
        return await self.client.chat(
            prompt, n=n, temperature=temperature, max_tokens=self.max_tokens.get(role), usage=usage
        )


class ProxyModel:
    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def load(cls, cfg):
        raise NotImplementedError("TODO: transformers full finetune 3B, bf16, device_map cuda")

    async def generate(self, prompt, n=1, role="rd", temperature=None, usage=None):
        raise NotImplementedError("TODO: sample n output, trả text + sum logprob")

    def logprobs_of(self, prompt: str, action_text: str) -> float:
        """Dùng cho tỉ số pi_theta / pi_old trong GRPO."""
        raise NotImplementedError("TODO")

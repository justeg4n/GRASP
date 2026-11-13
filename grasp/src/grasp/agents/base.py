"""Ba agent dùng CHUNG một proxy model, phân biệt bằng prompt theo role.

Mỗi agent: build_prompt(state) -> proxy sinh text -> parse() -> canonical() (format chuẩn dùng làm completion SFT).
Prompt render ở đây là prompt DUY NHẤT: teacher sinh SFT, proxy khi train GRPO và khi eval đều dùng nó.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cache
from pathlib import Path
from typing import Any

from grasp.types import AgentCall, Role

PROMPT_DIR = Path(__file__).parent / "prompts"


class ParseError(ValueError):
    pass


@cache
def load_template(name: str) -> str:
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8").strip() + "\n"


class Agent(ABC):
    role: Role

    def __init__(self, proxy, parse_retries: int = 1, retry_temperature: float = 0.0):
        self.proxy = (
            proxy  # ProxyBackend: generate(prompt, n, role, temperature) -> list[Generation]
        )
        self.parse_retries = parse_retries
        self.retry_temperature = retry_temperature

    @property
    def template(self) -> str:
        return load_template(self.role)

    @abstractmethod
    def build_prompt(self, state) -> str: ...

    @abstractmethod
    def parse(self, output: str, state) -> Any:
        """Raise ParseError nếu output không đúng format."""

    @abstractmethod
    def canonical(self, parsed, state) -> str:
        """Render lại action đã parse về đúng format chuẩn (completion SFT)."""

    async def act(self, state, n: int = 1):
        """Trả về n action được sample (n > 1 khi đang branching trong group rollout)."""
        prompt = self.build_prompt(state)
        outs = await self.proxy.generate(prompt, n=n, role=self.role)
        return [self.parse(o.text, state) for o in outs], outs

    def retry_hint(self) -> str:
        """Nhắc thêm khi hỏi lại sau output sai format (agent con override)."""
        return "Follow the required output format exactly."

    async def decide(self, state, step: int, usage=None) -> tuple[Any, AgentCall]:
        """1 action, có retry khi parse lỗi. Hết lượt -> ParseError.

        Lần hỏi lại kèm output bị từ chối + lý do (hỏi lại y nguyên ở temperature 0 thì model lặp lại đúng
        câu sai). AgentCall vẫn lưu prompt GỐC: dữ liệu SFT không chứa phần phản hồi này.
        """
        prompt = self.build_prompt(state)
        last_err, raw = None, ""
        for attempt in range(self.parse_retries + 1):
            temp = None if attempt == 0 else self.retry_temperature
            ask = prompt
            if attempt > 0:
                ask = (
                    f"{prompt}{raw.strip()}\n\n"
                    f"The answer above was rejected. {self.retry_hint()}\n"
                    "Corrected answer:\n"
                )
            gen = (
                await self.proxy.generate(ask, n=1, role=self.role, temperature=temp, usage=usage)
            )[0]
            raw = gen.text
            try:
                parsed = self.parse(raw, state)
            except ParseError as e:
                last_err = e
                continue
            call = AgentCall(
                self.role, step, prompt, self.canonical(parsed, state), raw, attempt > 0
            )
            return parsed, call
        raise ParseError(f"{self.role}: {last_err} | raw={raw[:300]!r}")

"""Client OpenAI-compatible: dùng chung cho vLLM local (reader) và API ngoài (teacher, extractor).

Retry/phân loại lỗi do client tự làm (SDK để max_retries=0) để collector biết chính xác lỗi thuộc loại nào.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from grasp.llm.budget import CostMeter, Price, Usage
from grasp.llm.errors import ErrorKind, classify, to_exception


@dataclass
class Generation:
    text: str
    logprob: float | None = None
    finish_reason: str | None = None


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        concurrency: int = 8,
        temperature: float = 0.7,
        max_tokens: int = 512,
        price: Price | None = None,
        meter: CostMeter | None = None,
        source: str = "",
        max_retries: int = 6,
        backoff_base: float = 1.0,
        backoff_cap: float = 30.0,
        timeout: float = 180.0,
        extra_body: dict | None = None,
        http_client: Any = None,
        **_ignored,
    ):
        from openai import AsyncOpenAI

        self.base_url, self.model, self.concurrency = base_url, model, concurrency
        self.api_key = api_key or os.environ.get("READER_API_KEY", "EMPTY")
        self.temperature, self.max_tokens = temperature, max_tokens
        self.price, self.meter, self.source = price, meter, source
        self.max_retries, self.backoff_base, self.backoff_cap = (
            max_retries,
            backoff_base,
            backoff_cap,
        )
        self.extra_body = dict(extra_body or {})
        self._sem = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=self.api_key,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )

    @classmethod
    def from_cfg(
        cls, cfg, meter: CostMeter | None = None, source: str = "", **overrides
    ) -> LLMClient:
        from omegaconf import DictConfig, OmegaConf

        d = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        price = d.pop("price", None)
        if price:
            price = Price(float(price.get("input_per_m", 0)), float(price.get("output_per_m", 0)))
        d.update(overrides)
        return cls(price=price, meter=meter, source=source, **d)

    # ---------------------------------------------------------------- public API

    async def chat(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        n: int = 1,
        usage: Usage | None = None,
    ) -> list[Generation]:
        """1 user message -> n generation. Dùng cho teacher và extractor."""

        async def call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
                n=n,
                extra_body=self.extra_body or None,
            )

        resp = await self._with_retry(call)
        self._account(resp, usage)
        return [Generation(c.message.content or "", None, c.finish_reason) for c in resp.choices]

    async def complete(
        self,
        prompt: str,
        *,
        stop: list[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        usage: Usage | None = None,
    ) -> Generation:
        """Text completion trên prompt đã áp chat template (reader cần tiếp nối output dở)."""

        async def call():
            return await self._client.completions.create(
                model=self.model,
                prompt=prompt,
                stop=stop,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=max_tokens or self.max_tokens,
                extra_body=self.extra_body or None,
            )

        resp = await self._with_retry(call)
        self._account(resp, usage)
        c = resp.choices[0]
        return Generation(c.text or "", None, c.finish_reason)

    async def ping(self) -> None:
        """Preflight: 1 request tối thiểu để lộ ra sớm lỗi key / credit / model."""
        await self.chat("ping", max_tokens=16, temperature=0.0)

    async def health(self) -> list[str]:
        """Liệt kê model của server (vLLM). Raise nếu server không phản hồi."""
        resp = await self._with_retry(lambda: self._client.models.list())
        return [m.id for m in resp.data]

    async def aclose(self) -> None:
        await self._client.close()

    # ---------------------------------------------------------------- internals

    def _account(self, resp, usage: Usage | None) -> None:
        u = getattr(resp, "usage", None)
        tin = getattr(u, "prompt_tokens", 0) or 0
        tout = getattr(u, "completion_tokens", 0) or 0
        if self.meter is not None and self.price is not None:
            rec = self.meter.record(self.price, tin, tout)
        else:
            rec = Usage(tin, tout, 0.0, 1)
        if usage is not None:
            usage.add(rec)

    async def _with_retry(self, fn: Callable[[], Awaitable[Any]]):
        import openai

        attempt = 0
        while True:
            if self.meter is not None and self.price is not None:
                self.meter.check()
            try:
                async with self._sem:
                    return await fn()
            except openai.APIStatusError as e:
                status, message = e.status_code, _message_of(e)
                retry_after = _retry_after(e)
            except (TimeoutError, openai.APIConnectionError) as e:
                status, message, retry_after = None, str(e) or type(e).__name__, None
            kind = classify(status, message)
            if kind is not ErrorKind.TRANSIENT or attempt >= self.max_retries:
                raise to_exception(kind, message, self.source, status)
            await asyncio.sleep(self._delay(attempt, retry_after))
            attempt += 1

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return min(retry_after, self.backoff_cap * 4)
        d = min(self.backoff_cap, self.backoff_base * 2**attempt)
        return d * (0.5 + random.random() / 2)


def _message_of(e) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        return str(body)
    return getattr(e, "message", None) or str(e)


def _retry_after(e) -> float | None:
    try:
        v = e.response.headers.get("retry-after")
        return float(v) if v is not None else None
    except (AttributeError, ValueError, TypeError):
        return None

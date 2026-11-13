"""LLMClient với server giả (httpx.MockTransport): retry, phân loại lỗi, đếm chi phí, budget."""

import asyncio
import json

import httpx
import pytest

from grasp.llm.budget import CostMeter, Price, Usage
from grasp.llm.client import LLMClient
from grasp.llm.errors import (
    BadRequest,
    BudgetExceeded,
    ErrorKind,
    FatalAPIError,
    TransientExhausted,
)


def ok_body(text="hi", tin=100, tout=10):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout},
    }


PRICE = Price(1.0, 2.0)


def make_client(responses, meter=None, price=PRICE, max_retries=3):
    calls = []

    def handler(request: httpx.Request):
        calls.append(json.loads(request.content))
        status, body = responses.pop(0) if len(responses) > 1 else responses[0]
        return httpx.Response(status, json=body)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    c = LLMClient(
        "http://fake/v1",
        "k",
        "m",
        price=price,
        meter=meter or CostMeter(),
        source="teacher",
        max_retries=max_retries,
        backoff_base=0.001,
        backoff_cap=0.002,
        http_client=http,
    )
    return c, calls


def run(coro):
    return asyncio.run(coro)


def test_retry_429_then_ok_and_cost_accounting():
    meter = CostMeter()
    c, calls = make_client(
        [(429, {"error": {"message": "rate limit"}}), (200, ok_body("pong", 1_000_000, 500_000))],
        meter=meter,
    )
    usage = Usage()
    out = run(c.chat("ping", usage=usage))
    assert out[0].text == "pong"
    assert len(calls) == 2
    assert usage.tokens_in == 1_000_000 and usage.usd == pytest.approx(1.0 + 1.0)
    assert meter.total.usd == pytest.approx(2.0)


def test_credit_error_is_fatal_without_retry():
    c, calls = make_client([(402, {"error": {"message": "Insufficient balance"}})])
    with pytest.raises(FatalAPIError) as e:
        run(c.chat("x"))
    assert e.value.kind is ErrorKind.CREDIT and len(calls) == 1


def test_persistent_500_exhausts_retries():
    c, calls = make_client([(500, {"error": {"message": "boom"}})], max_retries=3)
    with pytest.raises(TransientExhausted):
        run(c.chat("x"))
    assert len(calls) == 4


def test_400_is_bad_request():
    c, _ = make_client([(400, {"error": {"message": "context length exceeded"}})])
    with pytest.raises(BadRequest):
        run(c.chat("x"))


def test_budget_checked_before_each_call():
    meter = CostMeter(budget_usd=1.5)
    c, calls = make_client([(200, ok_body("a", 1_000_000, 0))], meter=meter)

    async def go():
        await c.chat("x")  # chi $1.0
        await c.chat("x")  # $1.0 < 1.5 nên vẫn gọi -> $2.0
        await c.chat("x")

    with pytest.raises(BudgetExceeded):
        run(go())
    assert len(calls) == 2

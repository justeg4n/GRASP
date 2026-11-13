"""Collector: resume, hết credit giữa chừng, API sập, budget, abandoned — không gọi mạng."""

import asyncio
import json

import pytest

from grasp.agents.base import ParseError
from grasp.llm.budget import CostMeter, Price
from grasp.llm.errors import (
    BadRequest,
    BudgetExceeded,
    ErrorKind,
    FatalAPIError,
    TransientExhausted,
)
from grasp.sft.collector import (
    EXIT_BUDGET,
    EXIT_FATAL,
    EXIT_OK,
    EXIT_OUTAGE,
    CollectConfig,
    Collector,
    read_records,
)
from grasp.types import Trajectory

QS = [
    {
        "qid": f"q{i}",
        "question": f"question {i}",
        "golden_answers": ["a"],
        "dataset": "ds",
        "split": "sft_train",
    }
    for i in range(1, 6)
]


def ok_traj(q):
    return Trajectory(
        question=q["question"], gold_answers=q["golden_answers"], answer="a", em=1.0, f1=1.0
    )


class Script:
    """episode_fn giả: hành vi theo qid; ghi lại các qid đã được chạy."""

    def __init__(self, behaviour=None, cost=0.0, meter=None):
        self.behaviour, self.cost, self.meter, self.ran = dict(behaviour or {}), cost, meter, []

    async def __call__(self, q, usage):
        self.ran.append(q["qid"])
        await asyncio.sleep(0)
        if self.meter is not None:
            self.meter.check()
            usage.add(self.meter.record(Price(1e6 * self.cost, 0), 1, 0))
        b = self.behaviour.get(q["qid"])
        if b is not None:
            raise b() if callable(b) else b
        return ok_traj(q)


def collector(tmp_path, fn, meter=None, preflight=None, **cfg):
    cfg.setdefault("concurrency", 1)
    return Collector(
        fn,
        tmp_path / "out.jsonl",
        tmp_path / "state.json",
        tmp_path / "summary.json",
        meter or CostMeter(),
        CollectConfig(**cfg),
        preflight=preflight,
        log=lambda *_: None,
    )


def run(c, qs=QS):
    return asyncio.run(c.run(qs))


def credit():
    return FatalAPIError(ErrorKind.CREDIT, "Insufficient balance", "teacher", 402)


def outage():
    return TransientExhausted(ErrorKind.TRANSIENT, "503", "teacher", 503)


def test_all_ok_then_resume_skips_everything(tmp_path):
    fn = Script()
    assert run(collector(tmp_path, fn)).exit_code == EXIT_OK
    assert sorted(read_records(tmp_path / "out.jsonl")) == ["q1", "q2", "q3", "q4", "q5"]
    fn2 = Script()
    assert run(collector(tmp_path, fn2)).exit_code == EXIT_OK and fn2.ran == []
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"] == {"ok": 5} and summary["correct"] == 5


def test_credit_exhausted_mid_run_stops_and_resume_continues(tmp_path):
    fn = Script({"q3": credit})
    res = run(collector(tmp_path, fn))
    assert res.exit_code == EXIT_FATAL and "credit" in res.stop_reason
    assert fn.ran == ["q1", "q2", "q3"]  # dừng ngay, không chạy q4, q5
    recs = read_records(tmp_path / "out.jsonl")
    assert sorted(recs) == ["q1", "q2"]  # câu dở không bị ghi
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["attempts"] == {}  # hết credit không tính là lỗi của câu

    fn2 = Script()  # nạp credit, chạy lại đúng lệnh
    assert run(collector(tmp_path, fn2)).exit_code == EXIT_OK
    assert fn2.ran == ["q3", "q4", "q5"]
    assert sorted(read_records(tmp_path / "out.jsonl")) == ["q1", "q2", "q3", "q4", "q5"]


def test_credit_stop_with_concurrency_cancels_inflight(tmp_path):
    class Slow(Script):
        async def __call__(self, q, usage):
            self.ran.append(q["qid"])
            if q["qid"] == "q1":
                raise credit()
            await asyncio.sleep(5)
            return ok_traj(q)

    res = run(collector(tmp_path, Slow(), concurrency=3))
    assert res.exit_code == EXIT_FATAL
    assert read_records(tmp_path / "out.jsonl") == {}


def test_outage_streak_stops_run(tmp_path):
    fn = Script({q["qid"]: outage for q in QS})
    res = run(collector(tmp_path, fn, outage_streak=3))
    assert res.exit_code == EXIT_OUTAGE and fn.ran == ["q1", "q2", "q3"]
    assert read_records(tmp_path / "out.jsonl") == {}
    assert json.loads((tmp_path / "state.json").read_text())["attempts"] == {
        "q1": 1,
        "q2": 1,
        "q3": 1,
    }


def test_isolated_transient_failure_is_retried_next_run_then_abandoned(tmp_path):
    for _ in range(2):
        res = run(collector(tmp_path, Script({"q2": outage}), max_attempts=2))
        assert res.exit_code == EXIT_OUTAGE and res.stop_reason == "pending_after_transient_errors"
    fn = Script()
    assert run(collector(tmp_path, fn, max_attempts=2)).exit_code == EXIT_OK
    assert fn.ran == []
    assert read_records(tmp_path / "out.jsonl")["q2"]["status"] == "abandoned"


def test_parse_error_and_bad_request_are_terminal(tmp_path):
    fn = Script(
        {
            "q1": ParseError("rd: garbage"),
            "q2": BadRequest(ErrorKind.BAD_REQUEST, "too long", "teacher", 400),
        }
    )
    assert run(collector(tmp_path, fn)).exit_code == EXIT_OK
    recs = read_records(tmp_path / "out.jsonl")
    assert recs["q1"]["status"] == "parse_error" and recs["q2"]["status"] == "bad_request"
    fn2 = Script()
    run(collector(tmp_path, fn2))
    assert fn2.ran == []
    fn3 = Script()  # chủ động chạy lại câu lỗi format
    run(collector(tmp_path, fn3, retry_statuses=("parse_error",)))
    assert fn3.ran == ["q1"] and read_records(tmp_path / "out.jsonl")["q1"]["status"] == "ok"


def test_consecutive_client_errors_trip_breaker(tmp_path):
    bad = BadRequest(ErrorKind.BAD_REQUEST, "model renamed?", "teacher", 400)
    fn = Script({q["qid"]: bad for q in QS})
    res = run(collector(tmp_path, fn, client_error_streak=2))
    assert res.exit_code == EXIT_FATAL and fn.ran == ["q1", "q2"]


def test_hard_budget_stops_run(tmp_path):
    meter = CostMeter(budget_usd=0.25)
    fn = Script(cost=0.1, meter=meter)
    res = run(collector(tmp_path, fn, meter=meter, force=True))
    assert res.exit_code == EXIT_BUDGET
    assert len(read_records(tmp_path / "out.jsonl")) == 3  # 0.1, 0.2, 0.3 rồi dừng
    # lần chạy sau tính cả tiền đã chi trước đó
    meter2 = CostMeter(budget_usd=0.25)
    fn2 = Script(cost=0.1, meter=meter2)
    assert run(
        collector(tmp_path, fn2, meter=meter2, force=True)
    ).exit_code == EXIT_BUDGET and fn2.ran == ["q4"]


def test_estimate_over_budget_refuses_to_start(tmp_path):
    fn = Script()
    res = run(collector(tmp_path, fn, meter=CostMeter(budget_usd=0.01), est_usd_per_question=0.007))
    assert res.exit_code == EXIT_BUDGET and fn.ran == []
    fn2 = Script()
    assert (
        run(collector(tmp_path, fn2, meter=CostMeter(budget_usd=0.01), force=True)).exit_code
        == EXIT_OK
    )


@pytest.mark.parametrize("err,code", [(credit, EXIT_FATAL), (outage, EXIT_OUTAGE)])
def test_preflight_failure_runs_nothing(tmp_path, err, code):
    async def preflight():
        raise err()

    fn = Script()
    assert run(collector(tmp_path, fn, preflight=preflight)).exit_code == code and fn.ran == []


def test_truncated_last_line_is_ignored(tmp_path):
    run(collector(tmp_path, Script()))
    with open(tmp_path / "out.jsonl", "a", encoding="utf-8") as f:
        f.write('{"qid": "q9", "status": "o')  # tắt máy giữa lúc ghi
    assert len(read_records(tmp_path / "out.jsonl")) == 5


def test_budget_exceeded_from_client_is_handled(tmp_path):
    fn = Script({"q2": BudgetExceeded(1.0, 1.0)})
    assert run(collector(tmp_path, fn)).exit_code == EXIT_BUDGET
    assert sorted(read_records(tmp_path / "out.jsonl")) == ["q1"]

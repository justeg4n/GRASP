"""Chạy episode cho từng câu hỏi, ghi trajectory, xử lý lỗi API / hết credit / resume (spec mục 6).

Bất biến quan trọng:
- Chỉ trạng thái KẾT THÚC mới được ghi vào output: ok, parse_error, bad_request, abandoned.
- Câu bị gián đoạn vì lỗi toàn cục / lỗi tạm thời KHÔNG được ghi -> lần chạy sau tự làm lại.
- Mỗi dòng được flush + fsync ngay, nên dừng đột ngột chỉ mất các trajectory đang chạy dở.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from grasp.agents.base import ParseError
from grasp.llm.budget import CostMeter, Usage
from grasp.llm.errors import (
    BadRequest,
    BudgetExceeded,
    FatalAPIError,
    TransientExhausted,
    UnknownClientError,
)
from grasp.types import Trajectory

EXIT_OK, EXIT_CRASH, EXIT_FATAL, EXIT_OUTAGE, EXIT_BUDGET, EXIT_INTERRUPTED = 0, 1, 2, 3, 4, 130
TERMINAL = ("ok", "parse_error", "bad_request", "abandoned")

EpisodeFn = Callable[[dict, Usage], Awaitable[Trajectory]]


@dataclass
class CollectConfig:
    concurrency: int = 16
    max_attempts: int = 3  # số lần bị gián đoạn tối đa trước khi bỏ câu (abandoned)
    outage_streak: int = 10  # số câu liên tiếp hết lượt retry -> coi là API sập
    client_error_streak: int = 5  # số câu liên tiếp lỗi 4xx / crash -> coi là lỗi toàn cục
    retry_statuses: tuple[str, ...] = ()
    force: bool = False  # bỏ qua kiểm tra ước tính chi phí
    est_usd_per_question: float = 0.007
    limit: int | None = None
    progress_every: int = 25

    @classmethod
    def from_cfg(cls, cfg) -> CollectConfig:
        d = {k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg}
        if "retry_statuses" in d:
            d["retry_statuses"] = tuple(d["retry_statuses"] or ())
        return cls(**d)


@dataclass
class RunResult:
    exit_code: int
    stop_reason: str | None
    written: Counter = field(default_factory=Counter)
    pending: int = 0
    spent_run: float = 0.0


def read_records(path: Path) -> dict[str, dict]:
    """qid -> record mới nhất (một qid có thể có nhiều dòng nếu được chạy lại qua retry_statuses)."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # dòng cuối bị cắt khi tắt máy đột ngột
            out[rec["qid"]] = rec
    return out


class _State:
    """Đếm số lần bị gián đoạn theo qid — lưu `state/{dataset}.json`."""

    def __init__(self, path: Path):
        self.path = path
        self.attempts: dict[str, int] = {}
        if path.exists():
            self.attempts = json.loads(path.read_text(encoding="utf-8")).get("attempts", {})

    def bump(self, qid: str) -> None:
        self.attempts[qid] = self.attempts.get(qid, 0) + 1

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"attempts": self.attempts}, indent=1), encoding="utf-8")
        tmp.replace(self.path)


class Collector:
    def __init__(
        self,
        episode_fn: EpisodeFn,
        out_path: str | Path,
        state_path: str | Path,
        summary_path: str | Path,
        meter: CostMeter,
        cfg: CollectConfig,
        preflight: Callable[[], Awaitable[None]] | None = None,
        record_extra: dict | None = None,
        on_progress: Callable[[dict], None] | None = None,
        resume_cmd: str = "",
        log: Callable[[str], None] = print,
    ):
        self.episode_fn, self.meter, self.cfg = episode_fn, meter, cfg
        self.out_path, self.summary_path = Path(out_path), Path(summary_path)
        self.state = _State(Path(state_path))
        self.preflight, self.record_extra = preflight, dict(record_extra or {})
        self.on_progress, self.resume_cmd, self.log = on_progress, resume_cmd, log

    # ------------------------------------------------------------------ public

    async def run(self, questions: list[dict]) -> RunResult:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_records(self.out_path)
        self.meter.spent_before = sum(r.get("usage", {}).get("usd", 0.0) for r in existing.values())
        self.result = RunResult(EXIT_OK, None)
        self._streak_outage = self._streak_client = 0
        self._stop = asyncio.Event()
        self._started = time.time()
        self._fh = open(self.out_path, "a", encoding="utf-8")  # noqa: SIM115 — đóng ở finally
        try:
            todo = self._plan(questions, existing)
            self.result.pending = len(todo)
            if todo and not self._estimate_ok(todo, existing):
                return self._finish(EXIT_BUDGET, "estimate_over_budget")
            if todo and self.preflight is not None:
                try:
                    await self.preflight()
                except FatalAPIError as e:
                    return self._finish(EXIT_FATAL, f"preflight: {e}")
                except TransientExhausted as e:
                    return self._finish(EXIT_OUTAGE, f"preflight: {e}")
                except BudgetExceeded as e:
                    return self._finish(EXIT_BUDGET, f"preflight: {e}")
            await self._run_pool(todo)
            if self.result.stop_reason:
                return self._finish(self.result.exit_code, self.result.stop_reason)
            if self.result.pending:
                return self._finish(EXIT_OUTAGE, "pending_after_transient_errors")
            return self._finish(EXIT_OK, None)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self._finish(EXIT_INTERRUPTED, "interrupted")
            raise
        finally:
            self._fh.close()
            self.state.save()

    # ------------------------------------------------------------------ planning

    def _plan(self, questions: list[dict], existing: dict[str, dict]) -> list[dict]:
        todo = []
        for q in questions:
            rec = existing.get(q["qid"])
            if (
                rec is not None
                and rec["status"] in TERMINAL
                and rec["status"] not in self.cfg.retry_statuses
            ):
                continue
            if self.state.attempts.get(q["qid"], 0) >= self.cfg.max_attempts:
                self._write(self._stub(q, "abandoned", f"gián đoạn {self.cfg.max_attempts} lần"))
                continue
            todo.append(q)
        if self.cfg.limit is not None:
            todo = todo[: self.cfg.limit]
        self.log(f"[collect] đã xong {len(existing)} câu, còn {len(todo)} câu cần chạy")
        return todo

    def _estimate_ok(self, todo: list[dict], existing: dict[str, dict]) -> bool:
        costs = [
            r["usage"]["usd"]
            for r in existing.values()
            if r.get("status") == "ok" and r.get("usage")
        ]
        per_q = sum(costs) / len(costs) if len(costs) >= 10 else self.cfg.est_usd_per_question
        est = per_q * len(todo)
        budget = self.meter.budget_usd
        self.log(
            f"[collect] ước tính ${est:.2f} cho {len(todo)} câu (${per_q:.4f}/câu); đã chi ${self.meter.spent:.2f}"
            + (f"; budget ${budget:.2f}" if budget is not None else "")
        )
        if budget is None or self.cfg.force or self.meter.spent + est <= budget:
            return True
        self.log(
            "[collect] ước tính vượt budget -> không chạy (tăng collect.budget_usd hoặc đặt collect.force=true)"
        )
        return False

    # ------------------------------------------------------------------ worker pool

    async def _run_pool(self, todo: list[dict]) -> None:
        queue: asyncio.Queue[dict] = asyncio.Queue()
        for q in todo:
            queue.put_nowait(q)
        workers = [
            asyncio.create_task(self._worker(queue))
            for _ in range(min(self.cfg.concurrency, len(todo)))
        ]
        stopper = asyncio.create_task(self._stop.wait())
        try:
            done_all = asyncio.gather(*workers)
            await asyncio.wait({done_all, stopper}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in workers:
                w.cancel()
            stopper.cancel()
            await asyncio.gather(*workers, stopper, return_exceptions=True)

    async def _worker(self, queue: asyncio.Queue) -> None:
        while not self._stop.is_set():
            try:
                q = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await self._one(q)

    async def _one(self, q: dict) -> None:
        usage = Usage()
        try:
            traj = await self.episode_fn(q, usage)
        except ParseError as e:
            self._done(self._stub(q, "parse_error", str(e), usage))
        except BadRequest as e:
            self._done(self._stub(q, "bad_request", str(e), usage), client_error=True)
        except UnknownClientError as e:
            self._interrupted(q, f"4xx lạ: {e}", client_error=True)
        except TransientExhausted as e:
            self._interrupted(q, f"hết lượt retry: {e}", outage=True)
        except (
            FatalAPIError
        ) as e:  # hết credit / sai key: không phải lỗi của câu này, không đếm attempts
            self._request_stop(EXIT_FATAL, f"{e.kind.value}: {e}")
        except BudgetExceeded as e:
            self._request_stop(EXIT_BUDGET, f"budget: {e}")
        except asyncio.CancelledError:
            raise
        except (
            Exception
        ) as e:  # bug trong code: không ghi, đếm như lỗi client để không đốt tiền vô ích
            self.log(f"[collect] lỗi không lường trước ở {q['qid']}:\n{traceback.format_exc()}")
            self._interrupted(q, f"crash: {type(e).__name__}: {e}", client_error=True, crash=True)
        else:
            traj.qid, traj.dataset, traj.split = q["qid"], q.get("dataset", ""), q.get("split", "")
            traj.usage = usage.as_dict()
            self._done(self._record(traj))

    # ------------------------------------------------------------------ bookkeeping

    def _done(self, rec: dict, client_error: bool = False) -> None:
        self._write(rec)
        self.result.pending -= 1
        if client_error:
            self._streak_client += 1
            self._check_streaks()
        else:
            self._streak_client = 0
        self._streak_outage = 0
        n = sum(self.result.written.values())
        if n % self.cfg.progress_every == 0:
            self._progress()

    def _interrupted(
        self,
        q: dict,
        why: str,
        outage: bool = False,
        client_error: bool = False,
        crash: bool = False,
    ):
        self.state.bump(q["qid"])
        self.log(f"[collect] {q['qid']} bị gián đoạn ({why}) — sẽ chạy lại ở lần sau")
        if outage:
            self._streak_outage += 1
        if client_error:
            self._streak_client += 1
        self._check_streaks(crash=crash)

    def _check_streaks(self, crash: bool = False) -> None:
        if self._streak_outage >= self.cfg.outage_streak:
            self._request_stop(
                EXIT_OUTAGE, f"{self._streak_outage} câu liên tiếp hết lượt retry (API/reader sập?)"
            )
        elif self._streak_client >= self.cfg.client_error_streak:
            code = EXIT_CRASH if crash else EXIT_FATAL
            self._request_stop(
                code,
                f"{self._streak_client} câu liên tiếp lỗi 4xx/crash — kiểm tra key, model, code",
            )

    def _request_stop(self, code: int, reason: str) -> None:
        if self.result.stop_reason is None:
            self.result.exit_code, self.result.stop_reason = code, reason
            self.log(f"[collect] DỪNG: {reason}")
        self._stop.set()

    def _record(self, traj: Trajectory) -> dict:
        return {**asdict(traj), **self.record_extra, "ts": time.time()}

    def _stub(self, q: dict, status: str, error: str, usage: Usage | None = None) -> dict:
        t = Trajectory(
            question=q["question"],
            gold_answers=q["golden_answers"],
            qid=q["qid"],
            dataset=q.get("dataset", ""),
            split=q.get("split", ""),
            status=status,
            error=error,
        )
        t.usage = (usage or Usage()).as_dict()
        return self._record(t)

    def _write(self, rec: dict) -> None:
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.result.written[rec["status"]] += 1

    def _progress(self) -> dict:
        n = sum(self.result.written.values())
        elapsed = time.time() - self._started
        info = {
            "written": n,
            "ok": self.result.written["ok"],
            "parse_error": self.result.written["parse_error"],
            "pending": self.result.pending,
            "spent_run_usd": round(self.meter.total.usd, 4),
            "spent_total_usd": round(self.meter.spent, 4),
            "sec_per_question": round(elapsed / max(n, 1), 2),
        }
        self.log("[collect] " + " ".join(f"{k}={v}" for k, v in info.items()))
        if self.on_progress:
            self.on_progress(info)
        return info

    def _finish(self, code: int, reason: str | None) -> RunResult:
        self.result.exit_code, self.result.stop_reason = code, reason
        self.result.spent_run = self.meter.total.usd
        all_recs = read_records(self.out_path)
        summary = {
            "exit_code": code,
            "stop_reason": reason,
            "status_counts": dict(Counter(r["status"] for r in all_recs.values())),
            "correct": sum(
                1 for r in all_recs.values() if r["status"] == "ok" and (r.get("em") or 0) >= 1
            ),
            "written_this_run": dict(self.result.written),
            "pending": self.result.pending,
            "spent_run_usd": round(self.meter.total.usd, 4),
            "spent_total_usd": round(self.meter.spent, 4),
            "tokens_this_run": self.meter.total.as_dict(),
            "resume": self.resume_cmd,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        self.log(f"[collect] kết thúc: exit={code} reason={reason} -> {self.summary_path}")
        return self.result

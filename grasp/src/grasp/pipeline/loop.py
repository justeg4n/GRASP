"""Vòng lặp suy luận: reader đóng băng sinh thought, proxy quyết định retrieve/query/integrate.

Bám Algorithm 1 của SPARKLE:
  0. (tuỳ chọn) retrieve bằng chính câu hỏi để có context ban đầu.
  1. reader sinh 1 thought (dừng ở </think>); có <answer> thì kết thúc.
  2. RD: thought này có cần retrieve không?
  3. QF: KG chain + knowledge gap + query (query None -> không retrieve).
  4. retriever top-k -> trích triple (cache) -> rank E5 -> top-20 candidate.
  5. KI chọn 1 triple. Điểm doc = max điểm triple của nó; doc chứa triple được chọn cộng thêm `selected_bonus`.
     Context = `max_context` doc điểm cao nhất (giữ các doc từ các bước trước).
  Hết `max_steps` mà chưa có đáp án -> reader sinh lại trọn vẹn với context cuối cùng.

Dùng chung cho sinh SFT (proxy = teacher API), eval và GRPO (proxy = model local).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from grasp.agents.ki import KIState, KnowledgeIntegrationAgent
from grasp.agents.qf import QFState, QueryFormulationAgent
from grasp.agents.rd import RDState, RetrievalDecisionAgent
from grasp.eval.metrics import exact_match, f1, normalize_answer
from grasp.kg.rank import rank_triples
from grasp.llm.budget import Usage
from grasp.llm.reader import Reader, extract_answer, split_thoughts
from grasp.types import Passage, Step, Trajectory, Triple


@dataclass
class LoopConfig:
    max_steps: int = 5
    max_retrievals: int = 5
    initial_retrieval: bool = True
    top_k_docs: int = 10
    triples_top_k: int = 20
    max_context: int = 10
    selected_bonus: float = 0.5

    @classmethod
    def from_cfg(cls, cfg) -> LoopConfig:
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})


@dataclass
class EpisodeDeps:
    reader: Reader
    searcher: object  # .search(query, top_k) -> list[Passage]   (đồng bộ)
    extractor: object  # await .triples_for(passages, usage) -> list[Triple]
    encoder: object  # dùng để rank triple
    rd: RetrievalDecisionAgent
    qf: QueryFormulationAgent
    ki: KnowledgeIntegrationAgent


@dataclass
class _Context:
    max_docs: int
    scores: dict[str, float] = field(default_factory=dict)
    docs: dict[str, Passage] = field(default_factory=dict)

    def add(
        self,
        ranked: list[tuple[Triple, float]],
        passages: list[Passage],
        selected: Triple | None,
        bonus: float,
    ):
        by_id = {p.doc_id: p for p in passages}
        for t, s in ranked:
            if t.doc_id not in by_id:
                continue
            if selected is not None and t is selected:
                s += bonus
            self.docs.setdefault(t.doc_id, by_id[t.doc_id])
            self.scores[t.doc_id] = max(self.scores.get(t.doc_id, float("-inf")), s)

    def top(self) -> list[Passage]:
        ids = sorted(self.scores, key=lambda d: -self.scores[d])[: self.max_docs]
        return [self.docs[d] for d in ids]

    def text(self) -> str:
        return "\n\n".join(f"Title: {p.title or ''}\nText: {p.text}" for p in self.top())


async def run_episode(
    question: str, gold: list[str], deps: EpisodeDeps, cfg: LoopConfig, usage: Usage | None = None
) -> Trajectory:
    """Chạy tuần tự (dùng cho sinh SFT, eval, inference — KHÔNG branching).

    ParseError / lỗi API được ném ra ngoài cho caller (collector) phân loại.
    """
    traj = Trajectory(question=question, gold_answers=list(gold))
    ctx = _Context(cfg.max_context)

    async def retrieve(
        query: str, step: Step | None
    ) -> tuple[list[Passage], list[tuple[Triple, float]]]:
        passages = await asyncio.to_thread(deps.searcher.search, query, cfg.top_k_docs)
        triples = await deps.extractor.triples_for(passages, usage)
        ranked = await asyncio.to_thread(
            rank_triples,
            deps.encoder,
            question,
            "" if step is None else query,
            triples,
            cfg.triples_top_k,
        )
        if step is not None:
            step.retrieved, step.candidates = passages, [t for t, _ in ranked]
        return passages, ranked

    if cfg.initial_retrieval:
        passages, ranked = await retrieve(question, None)
        ctx.add(ranked, passages, None, 0.0)

    output, n_retrievals, seen_queries = "", 0, set()
    for t in range(cfg.max_steps):
        piece = await deps.reader.next_thought(question, ctx.text(), output, usage)
        if not piece:
            break
        output += piece
        if "<answer>" in piece:
            break
        thoughts = split_thoughts(output)
        current = thoughts[-1] if thoughts else piece.strip()
        step = Step(t=t, thought=current)
        traj.steps.append(step)
        if n_retrievals >= cfg.max_retrievals:
            step.skipped = "max_retrievals"
            continue

        retrieve_needed, call = await deps.rd.decide(
            RDState(question, ctx.text(), thoughts[:-1], current), t, usage
        )
        traj.calls.append(call)
        step.retrieve = retrieve_needed
        if not retrieve_needed:
            continue

        act, call = await deps.qf.decide(QFState(question, thoughts), t, usage)
        traj.calls.append(call)
        step.chain, step.gap, step.query = act.chain, act.gap, act.query
        if act.query is None:
            step.skipped = "no_query"
            continue
        qkey = normalize_answer(act.query)
        if qkey in seen_queries:
            step.skipped = "duplicate_query"
            continue
        seen_queries.add(qkey)

        passages, ranked = await retrieve(act.query, step)
        if not ranked:
            step.skipped = "no_triples"
            continue
        selected, call = await deps.ki.decide(
            KIState(question, act.chain, act.query, step.candidates), t, usage
        )
        traj.calls.append(call)
        step.selected = selected
        ctx.add(ranked, passages, selected, cfg.selected_bonus)
        n_retrievals += 1

    answer = extract_answer(output)
    if answer is None:
        final = await deps.reader.final_answer(question, ctx.text(), usage)
        traj.meta["final_regenerated"] = True
        output += "\n[final]\n" + final
        answer = extract_answer(final)
    traj.reader_output = output
    traj.context = ctx.top()
    traj.answer = answer or ""
    traj.em = exact_match(traj.answer, traj.gold_answers)
    traj.f1 = f1(traj.answer, traj.gold_answers)
    traj.meta["n_retrievals"] = n_retrievals
    return traj


def step_once(traj: Trajectory, agents, reader, searcher, cfg) -> Step:
    """Một bước — dùng lại được trong rollout có branching (giai đoạn GRPO)."""
    raise NotImplementedError("TODO: tách thân vòng lặp run_episode khi làm group rollout")

"""Lắp các thành phần từ config Hydra: đường dẫn corpus, client, agent, episode function."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from grasp.agents.ki import KnowledgeIntegrationAgent
from grasp.agents.qf import QueryFormulationAgent
from grasp.agents.rd import RetrievalDecisionAgent
from grasp.kg.extract import TripleCache, TripleExtractor
from grasp.llm.budget import CostMeter, Usage
from grasp.llm.client import LLMClient
from grasp.llm.errors import ErrorKind, FatalAPIError
from grasp.llm.proxy import ApiProxyBackend
from grasp.llm.reader import Reader, hf_chat_template
from grasp.pipeline.loop import EpisodeDeps, LoopConfig, run_episode


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.split("/")[-1])


def corpus_dir(cfg) -> Path:
    """{paths.corpus}/{tên corpus}/{tên encoder}: passages.sqlite, emb/, index.faiss."""
    return Path(cfg.paths.corpus) / cfg.data.corpus.name / slug(cfg.retriever.name)


def passages_path(cfg) -> Path:
    """Passage store không phụ thuộc encoder nên đặt ở cấp corpus."""
    return Path(cfg.paths.corpus) / cfg.data.corpus.name / "passages.sqlite"


@dataclass
class Components:
    deps: EpisodeDeps
    teacher: LLMClient
    extractor_client: LLMClient
    reader_client: LLMClient
    triple_cache: TripleCache
    meter: CostMeter
    loop_cfg: LoopConfig

    async def preflight(self) -> None:
        if not self.teacher.api_key or self.teacher.api_key == "EMPTY":
            raise FatalAPIError(
                ErrorKind.AUTH, "chưa đặt FPT_API_KEY (xem .env.example)", "teacher"
            )
        await self.teacher.ping()
        models = await self.reader_client.health()
        if self.reader_client.model not in models:
            raise FatalAPIError(
                ErrorKind.CONFIG,
                f"reader server không có model {self.reader_client.model}, chỉ có {models}",
                "reader",
            )

    async def episode(self, q: dict, usage: Usage):
        return await run_episode(
            q["question"], q["golden_answers"], self.deps, self.loop_cfg, usage
        )

    async def aclose(self) -> None:
        for c in (self.teacher, self.extractor_client, self.reader_client):
            await c.aclose()
        self.triple_cache.close()


def build_components(cfg, searcher, encoder) -> Components:
    meter = CostMeter(budget_usd=cfg.collect.get("budget_usd"))
    teacher = LLMClient.from_cfg(cfg.teacher.client, meter=meter, source="teacher")
    extractor_client = LLMClient.from_cfg(
        cfg.teacher.client,
        meter=meter,
        source="extractor",
        **dict(cfg.teacher.extractor.client_overrides),
    )
    reader_client = LLMClient.from_cfg(cfg.reader.client, source="reader")
    reader = Reader(
        reader_client, hf_chat_template(cfg.reader.tokenizer), cfg.reader.max_tokens_answer
    )

    proxy = ApiProxyBackend(teacher, max_tokens=dict(cfg.teacher.get("role_max_tokens") or {}))
    retry = dict(
        parse_retries=cfg.teacher.parse_retries, retry_temperature=cfg.teacher.retry_temperature
    )
    cache = TripleCache(Path(cfg.paths.cache) / "triples.sqlite")
    extractor = TripleExtractor(
        extractor_client,
        cache,
        max_tokens=cfg.teacher.extractor.max_tokens,
        temperature=cfg.teacher.extractor.temperature,
    )
    deps = EpisodeDeps(
        reader=reader,
        searcher=searcher,
        extractor=extractor,
        encoder=encoder,
        rd=RetrievalDecisionAgent(proxy, **retry),
        qf=QueryFormulationAgent(proxy, **retry),
        ki=KnowledgeIntegrationAgent(proxy, **retry),
    )
    return Components(
        deps,
        teacher,
        extractor_client,
        reader_client,
        cache,
        meter,
        LoopConfig.from_cfg(cfg.collect),
    )

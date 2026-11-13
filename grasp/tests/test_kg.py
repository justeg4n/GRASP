import asyncio

from grasp.kg.extract import TripleCache, TripleExtractor
from grasp.kg.rank import rank_triples
from grasp.kg.triples import parse_triples
from grasp.llm.client import Generation
from grasp.llm.errors import BadRequest, ErrorKind
from grasp.types import Passage, Triple


class FakeExtractorClient:
    model = "fake-extractor"

    def __init__(self, fail_doc=None):
        self.calls, self.fail_doc = [], fail_doc

    async def chat(self, prompt, **kw):
        self.calls.append(prompt)
        await asyncio.sleep(0.01)
        if self.fail_doc and self.fail_doc in prompt:
            raise BadRequest(ErrorKind.BAD_REQUEST, "too long", "extractor", 400)
        title = prompt.split("Title: ")[1].split("\n")[0]
        return [Generation(f"<{title}; is; thing>\n<{title}; located in; Paris>\nnoise line")]


def test_parse_triples():
    ts = parse_triples("<A; b; C>, < D ;e;F >\n<bad>", doc_id="x")
    assert [(t.head, t.relation, t.tail, t.doc_id) for t in ts] == [
        ("A", "b", "C", "x"),
        ("D", "e", "F", "x"),
    ]


def test_cache_miss_then_hit_and_inflight_dedup(tmp_path):
    client = FakeExtractorClient()
    ex = TripleExtractor(client, TripleCache(tmp_path / "t.sqlite"))
    docs = [Passage("d1", "text one", "Alpha"), Passage("d2", "text two", "Beta")]

    async def go():
        # 2 episode cùng cần d1 cùng lúc -> chỉ 1 request cho d1
        a, b = await asyncio.gather(ex.triples_for(docs), ex.triples_for(docs[:1]))
        return a, b

    a, b = asyncio.run(go())
    assert len(client.calls) == 2
    assert {t.doc_id for t in a} == {"d1", "d2"} and len(a) == 4 and len(b) == 2
    asyncio.run(ex.triples_for(docs))
    assert len(client.calls) == 2  # lần sau đọc cache

    ex2 = TripleExtractor(client, TripleCache(tmp_path / "t.sqlite"))  # cache bền qua lần chạy mới
    asyncio.run(ex2.triples_for(docs))
    assert len(client.calls) == 2


def test_bad_request_doc_is_cached_empty(tmp_path):
    client = FakeExtractorClient(fail_doc="Gamma")
    ex = TripleExtractor(client, TripleCache(tmp_path / "t.sqlite"))
    out = asyncio.run(ex.triples_for([Passage("d3", "x", "Gamma")]))
    assert out == [] and ex.cache.get("d3", client.model) == []


def test_rank_triples_dedup_and_order(encoder):
    ts = [
        Triple("Paris", "capital of", "France", "d1"),
        Triple("Bob", "likes", "cheese", "d2"),
        Triple("paris", "capital of", "France.", "d3"),
    ]
    ranked = rank_triples(encoder, "What is the capital of France?", "capital France", ts, top_k=5)
    assert len(ranked) == 2 and ranked[0][0].head == "Paris"

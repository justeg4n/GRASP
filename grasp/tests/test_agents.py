import asyncio

import pytest
from conftest import ScriptedProxy

from grasp.agents.base import ParseError
from grasp.agents.ki import KIState, KnowledgeIntegrationAgent
from grasp.agents.qf import QueryFormulationAgent
from grasp.agents.rd import RDState, RetrievalDecisionAgent
from grasp.types import Triple


@pytest.mark.parametrize(
    "out,expected",
    [
        ("Yes (retrieval is required)", True),
        ("no (retrieval is not required)", False),
        ("**Yes**", True),
        ("No.", False),
    ],
)
def test_rd_parse(out, expected):
    assert RetrievalDecisionAgent(None).parse(out) is expected


def test_rd_parse_rejects_garbage():
    with pytest.raises(ParseError):
        RetrievalDecisionAgent(None).parse("I think we should look it up")


def test_rd_prompt_contains_inputs():
    p = RetrievalDecisionAgent(None).build_prompt(RDState("Q?", "", ["t1"], "t2"))
    assert "Q?" in p and "t1" in p and "t2" in p and "Retrieved Context:\nNone" in p


QF_OUT = """Step 1: Reasoning Chain: <Polish-Russian War; directed by; Xawery Żuławski>
Step 2: Knowledge Gap: To continue answering the question, we need to know: who is the mother of Xawery Żuławski
Step 3: Query: Xawery Żuławski mother"""


def test_qf_parse_and_canonical():
    a = QueryFormulationAgent(None).parse(QF_OUT)
    assert a.query == "Xawery Żuławski mother"
    assert a.chain == [Triple("Polish-Russian War", "directed by", "Xawery Żuławski")]
    assert a.gap == "who is the mother of Xawery Żuławski"
    canon = QueryFormulationAgent(None).canonical(a)
    assert QueryFormulationAgent(None).parse(canon) == a  # canonical đọc lại được


def test_qf_query_none_and_multiline():
    a = QueryFormulationAgent(None).parse(
        "Reasoning Chain: None\nKnowledge Gap: nothing\nQuery: None"
    )
    assert a.query is None and a.chain == []
    b = QueryFormulationAgent(None).parse(
        "**Reasoning Chain:**\n<a; b; c>,\n<d; e; f>\n**Query:** find g"
    )
    assert len(b.chain) == 2 and b.query == "find g"


def test_qf_missing_query_is_parse_error():
    with pytest.raises(ParseError):
        QueryFormulationAgent(None).parse("Reasoning Chain: <a; b; c>")


CANDS = [Triple("A", "born in", "Paris", "d1"), Triple("B", "wife of", "C", "d2")]


@pytest.mark.parametrize(
    "out", ["<B; wife of; C>", "[2] <B; wife of; C>", "Selected: <b;  wife of ; C.>", "[2]"]
)
def test_ki_parse_matches_candidate(out):
    st = KIState("q", [], "query", CANDS)
    assert KnowledgeIntegrationAgent(None).parse(out, st) is CANDS[1]


def test_ki_rejects_triple_not_in_candidates():
    with pytest.raises(ParseError):
        KnowledgeIntegrationAgent(None).parse("<X; y; Z>", KIState("q", [], "q", CANDS))


def test_decide_retries_once_then_records_canonical():
    proxy = ScriptedProxy({"rd": ["hmm let me think", "Yes"]})
    agent = RetrievalDecisionAgent(proxy, parse_retries=1)
    parsed, call = asyncio.run(agent.decide(RDState("q", "", [], "t"), step=0))
    assert parsed is True and call.retried and call.completion == "Yes (retrieval is required)"
    assert call.raw == "Yes"
    first, second = (pr for _, pr in proxy.prompts)
    assert second.startswith(first) and "hmm let me think" in second and "rejected" in second
    assert call.prompt == first  # dữ liệu SFT giữ prompt gốc, không có phần phản hồi


def test_ki_accepts_paraphrase_only_when_unambiguous():
    cands = [
        Triple("Neil Jordan", "occupation", "Irish film director", "d1"),
        Triple("Neil Jordan", "born in", "Sligo", "d2"),
        Triple("Bob", "nationality", "Irish", "d3"),
    ]
    st = KIState("q", [], "q", cands)
    ki = KnowledgeIntegrationAgent(None)
    assert ki.parse("<Neil Jordan; nationality; Irish>", st) is cands[0]
    tie = [Triple("A", "born in", "Paris"), Triple("A", "died in", "Paris")]
    with pytest.raises(ParseError):
        ki.parse("<A; lived in; Paris>", KIState("q", [], "q", tie))


def test_decide_gives_up_after_retries():
    proxy = ScriptedProxy({"rd": ["??", "??"]})
    with pytest.raises(ParseError):
        asyncio.run(
            RetrievalDecisionAgent(proxy, parse_retries=1).decide(RDState("q", "", [], "t"), step=0)
        )

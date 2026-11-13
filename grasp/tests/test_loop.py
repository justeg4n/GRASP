import asyncio

from conftest import ScriptedProxy

from grasp.agents.ki import KnowledgeIntegrationAgent
from grasp.agents.qf import QueryFormulationAgent
from grasp.agents.rd import RetrievalDecisionAgent
from grasp.llm.client import Generation
from grasp.llm.reader import Reader, extract_answer, split_thoughts
from grasp.pipeline.loop import EpisodeDeps, LoopConfig, run_episode
from grasp.types import Passage, Triple

DOCS = {
    "film x": [
        Passage("d1", "Film X is a 2001 film directed by Bob Stone.", "Film X"),
        Passage("d9", "Unrelated text about apples.", "Apple"),
    ],
    "bob stone mother": [Passage("d2", "Bob Stone is the son of Mary Stone.", "Bob Stone")],
}
TRIPLES = {
    "d1": [Triple("Film X", "directed by", "Bob Stone", "d1")],
    "d9": [Triple("Apple", "is", "fruit", "d9")],
    "d2": [Triple("Bob Stone", "mother", "Mary Stone", "d2")],
}


class FakeSearcher:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k=None):
        self.queries.append(query)
        for k, docs in DOCS.items():
            if k in query.lower():
                return docs
        return DOCS["film x"]


class FakeExtractor:
    async def triples_for(self, passages, usage=None):
        return [t for p in passages for t in TRIPLES[p.doc_id]]


class FakeReaderClient:
    def __init__(self, pieces):
        self.pieces, self.prompts = list(pieces), []

    async def complete(self, prompt, stop=None, temperature=None, max_tokens=None, usage=None):
        self.prompts.append(prompt)
        return Generation(self.pieces.pop(0))


def make_deps(reader_pieces, scripts, encoder):
    proxy = ScriptedProxy(scripts)
    reader_client = FakeReaderClient(reader_pieces)
    deps = EpisodeDeps(
        reader=Reader(reader_client, chat_template=lambda u: f"<user>{u}</user><assistant>"),
        searcher=FakeSearcher(),
        extractor=FakeExtractor(),
        encoder=encoder,
        rd=RetrievalDecisionAgent(proxy),
        qf=QueryFormulationAgent(proxy),
        ki=KnowledgeIntegrationAgent(proxy),
    )
    return deps, proxy, reader_client


QF = (
    "Reasoning Chain: <Film X; directed by; Bob Stone>\n"
    "Knowledge Gap: To continue answering the question, we need to know: the mother of Bob Stone\n"
    "Query: Bob Stone mother"
)


def test_reader_helpers():
    out = "<think> a  b </think>\n<think>c</think>\n<answer> Mary Stone </answer>"
    assert split_thoughts(out) == ["a b", "c"]
    assert extract_answer(out) == "Mary Stone"
    assert extract_answer("<answer> X") == "X" and extract_answer("no answer") is None


def test_full_episode(encoder):
    deps, proxy, rc = make_deps(
        [
            "<think> Film X was directed by Bob Stone.",
            "<think> I need Bob Stone's mother.",
            "<think> Bob Stone's mother is Mary Stone.",
            "<answer> Mary Stone </answer>",
        ],
        {
            "rd": ["No (retrieval is not required)", "Yes (retrieval is required)", "No"],
            "qf": [QF],
            "ki": ["<Bob Stone; mother; Mary Stone>"],
        },
        encoder,
    )
    traj = asyncio.run(
        run_episode(
            "Who is the mother of the director of Film X?", ["Mary Stone"], deps, LoopConfig()
        )
    )
    assert traj.answer == "Mary Stone" and traj.em == 1.0
    assert [c.role for c in traj.calls] == ["rd", "rd", "qf", "ki", "rd"]
    assert deps.searcher.queries == [
        "Who is the mother of the director of Film X?",
        "Bob Stone mother",
    ]
    assert traj.steps[1].selected == Triple("Bob Stone", "mother", "Mary Stone", "d2")
    assert traj.context[0].doc_id == "d2"  # doc của triple được chọn có bonus -> đứng đầu
    # thought trước được giữ, context mới được đưa vào prompt reader ở bước sau
    assert "Bob Stone is the son of Mary Stone." in rc.prompts[2]
    assert rc.prompts[2].endswith("<think> I need Bob Stone's mother. </think>\n")
    assert traj.calls[3].completion == "<Bob Stone; mother; Mary Stone>"


def test_no_answer_after_max_steps_regenerates(encoder):
    deps, _, rc = make_deps(
        [
            "<think> step one",
            "<think> step two",
            "<think> Answer is Mary Stone. </think> <answer> Mary Stone </answer>",
        ],
        {"rd": ["No", "No"], "qf": [], "ki": []},
        encoder,
    )
    traj = asyncio.run(
        run_episode("q?", ["Mary Stone"], deps, LoopConfig(max_steps=2, initial_retrieval=False))
    )
    assert traj.meta["final_regenerated"] and traj.answer == "Mary Stone"
    assert not rc.prompts[-1].endswith("</think>\n")  # sinh lại từ đầu, không nối output dở


def test_query_none_and_duplicate_query_skip_retrieval(encoder):
    qf_none = "Reasoning Chain: None\nKnowledge Gap: nothing\nQuery: None"
    deps, _, _ = make_deps(
        ["<think> a", "<think> b", "<think> c", "<answer> x </answer>"],
        {"rd": ["Yes", "Yes", "Yes"], "qf": [qf_none, QF, QF], "ki": ["[1]"]},
        encoder,
    )
    traj = asyncio.run(run_episode("q?", ["y"], deps, LoopConfig(initial_retrieval=False)))
    assert [s.skipped for s in traj.steps] == ["no_query", None, "duplicate_query"]
    assert deps.searcher.queries == ["Bob Stone mother"]
    assert traj.em == 0.0

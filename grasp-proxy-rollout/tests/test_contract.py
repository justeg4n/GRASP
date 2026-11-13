from grasp_proxy_rollout.parsers import match_candidate, parse_output
from grasp_proxy_rollout.prompts import PROMPT_SCHEMA, build_prompt


def test_rd_prompt_and_parser_match_sft_contract():
    prompt = build_prompt(
        "rd",
        {
            "question": "Question?",
            "context": [{"title": "Document", "text": "Evidence."}],
            "thoughts": ["Earlier thought.", "Need evidence."],
        },
    )
    assert PROMPT_SCHEMA == "grasp-sft-text-v1"
    assert "Title: Document\nText: Evidence." in prompt
    assert "Previous Reasoning:\nEarlier thought." in prompt
    assert "Current Reasoning Step:\nNeed evidence." in prompt
    assert parse_output("rd", "Yes (retrieval is required)") == (True, True, None)
    assert parse_output("rd", "No (retrieval is not required)") == (False, True, None)
    assert parse_output("rd", "Yes, because I am unsure")[1] is False


def test_qf_three_line_parser():
    output = """Reasoning Chain: <A; located in; B>, <B; part of; C>
Knowledge Gap: To continue answering the question, we need to know: the capital of C
Query: capital of C"""
    parsed, valid, error = parse_output("qf", output)
    assert valid is True
    assert error is None
    assert parsed == {
        "chain": [
            {"head": "A", "relation": "located in", "tail": "B"},
            {"head": "B", "relation": "part of", "tail": "C"},
        ],
        "knowledge_gap": "the capital of C",
        "query": "capital of C",
    }


def test_qf_none_values_are_supported():
    output = """Reasoning Chain: None
Knowledge Gap: missing fact
Query: None"""
    parsed, valid, _ = parse_output("qf", output)
    assert valid is True
    assert parsed["chain"] == []
    assert parsed["query"] is None


def test_ki_triple_parser_and_candidate_match():
    candidates = [{"head": "a", "relation": "r", "tail": "b", "doc_id": "doc-1"}]
    prompt = build_prompt(
        "ki",
        {
            "question": "Question?",
            "chain": [],
            "query": "query",
            "candidates": candidates,
        },
    )
    assert "[1] <a; r; b>" in prompt
    parsed, valid, error = parse_output("ki", "<A; r; b>")
    assert valid is True
    assert error is None
    assert match_candidate(parsed, candidates) == 0

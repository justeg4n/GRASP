import json

import pytest
from hydra import compose, initialize_config_dir

from grasp.data.questions import load_questions, make_splits, read_splits, select, write_splits
from grasp.llm.client import LLMClient
from grasp.sft.export import export, passes
from grasp.utils import REPO_ROOT


def rec(qid, split, status="ok", em=1.0, f1=1.0):
    return {
        "qid": qid,
        "dataset": "ds",
        "split": split,
        "status": status,
        "em": em,
        "f1": f1,
        "calls": [
            {
                "role": "rd",
                "step": 0,
                "prompt": "P-rd",
                "completion": "Yes (retrieval is required)",
            },
            {"role": "qf", "step": 0, "prompt": "P-qf", "completion": "Reasoning Chain: None"},
        ],
    }


def test_passes_filter():
    assert passes(rec("a", "sft_train"))
    assert passes(rec("a", "sft_train", em=0.0, f1=0.85))
    assert not passes(rec("a", "sft_train", em=0.0, f1=0.5))
    assert not passes(rec("a", "sft_train", status="parse_error"))


def test_export_writes_role_pairs_and_uses_latest_record(tmp_path):
    lines = [
        rec("q1", "sft_train"),
        rec("q2", "sft_val"),
        rec("q3", "sft_train", em=0.0, f1=0.1),
        rec("q4", "sft_train", status="parse_error"),
        rec("q4", "sft_train"),
    ]  # q4 chạy lại thành công
    p = tmp_path / "ds.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    stats = export([p], tmp_path / "sft")
    train = [
        json.loads(x)
        for x in (tmp_path / "sft" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {r["qid"] for r in train} == {"q1", "q4"} and len(train) == 4
    assert train[0]["messages"] == [
        {"role": "user", "content": "P-rd"},
        {"role": "assistant", "content": "Yes (retrieval is required)"},
    ]
    assert stats["rows"] == {"train": 4, "val": 2} and stats["roles"] == {"rd": 3, "qf": 3}


def test_splits_disjoint_and_reproducible(tmp_path):
    qids = [f"train_{i}" for i in range(100)]
    sizes = {"sft_train": 20, "sft_val": 5, "rl_pool": 30}
    a, b = make_splits(qids, sizes, 42), make_splits(list(reversed(qids)), sizes, 42)
    assert a == b
    assert {k: len(v) for k, v in a.items()} == sizes
    allq = a["sft_train"] + a["sft_val"] + a["rl_pool"]
    assert len(set(allq)) == len(allq)
    with pytest.raises(ValueError):
        make_splits(qids[:10], sizes, 42)
    write_splits(tmp_path / "s.json", "ds", "src", 42, a)
    assert read_splits(tmp_path / "s.json") == a


def test_load_and_select(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(
        '{"id": "train_0", "question": "Q0?", "golden_answers": ["A"]}\n'
        '{"id": "train_1", "question": "Q1?", "golden_answers": ["B", "b"]}\n',
        encoding="utf-8",
    )
    qs = load_questions(p)
    picked = select(qs, {"sft_val": ["train_1"]}, ["sft_val"], "nq")
    assert picked == [
        {
            "qid": "train_1",
            "question": "Q1?",
            "golden_answers": ["B", "b"],
            "dataset": "nq",
            "split": "sft_val",
        }
    ]


@pytest.mark.parametrize("data", ["hotpotqa", "2wiki", "nq"])
def test_configs_compose_and_build_clients(data, monkeypatch):
    monkeypatch.delenv("FPT_API_KEY", raising=False)
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"data={data}"])
    assert cfg.data.name == data and cfg.retriever.top_k == 10
    assert cfg.teacher.client.api_key == ""
    teacher = LLMClient.from_cfg(cfg.teacher.client, source="teacher")
    assert teacher.model == "DeepSeek-V4-Flash" and teacher.price.input_per_m == 0.14
    ext = LLMClient.from_cfg(
        cfg.teacher.client, source="extractor", **dict(cfg.teacher.extractor.client_overrides)
    )
    assert ext.temperature == 0.0
    reader = LLMClient.from_cfg(cfg.reader.client, source="reader")
    assert reader.model == "Qwen/Qwen2.5-7B-Instruct"
    assert sum(cfg.splits.sizes.values()) == 6000


def test_committed_splits_match_config():
    """splits/*.json (nếu đã tạo) phải đúng kích thước trong config và không giao nhau."""
    for p in sorted((REPO_ROOT / "splits").glob("*.json")):
        s = read_splits(p)
        assert {k: len(v) for k, v in s.items()} == {
            "sft_train": 2500,
            "sft_val": 500,
            "rl_pool": 3000,
        }
        allq = [q for v in s.values() for q in v]
        assert len(set(allq)) == len(allq), p.name

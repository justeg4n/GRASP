"""End-to-end: scripts/collect_sft.py + config Hydra thật, server OpenAI giả đóng vai teacher FPT và vLLM reader.

Thay encoder/searcher/chat template bằng bản giả để không cần torch, index hay mạng.
Kiểm tra: lắp ráp đúng, trajectory ghi ra đủ, hết credit giữa chừng thì dừng exit 2 và chạy lại thì tiếp tục.
"""

import importlib.util
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import HashEncoder
from hydra import compose, initialize_config_dir

from grasp.types import Passage
from grasp.utils import REPO_ROOT

QS = [
    {
        "id": f"train_{i}",
        "question": f"Who is the mother of the director of Film {i}?",
        "golden_answers": ["Mary Stone"],
    }
    for i in range(6)
]


class FakeOpenAI(BaseHTTPRequestHandler):
    credit_left = 10_000  # số request chat còn "credit"

    def log_message(self, *a):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(
            200, {"object": "list", "data": [{"id": "Qwen/Qwen2.5-7B-Instruct", "object": "model"}]}
        )

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["content-length"])))
        usage = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
        if self.path.endswith("/completions") and "messages" not in req:  # reader
            out = req["prompt"].split("<assistant>", 1)[1]
            text = (
                "<think> Film was directed by Bob Stone."
                if "</think>" not in out
                else "<answer> Mary Stone </answer>"
            )
            self._send(
                200,
                {
                    "id": "r",
                    "object": "text_completion",
                    "created": 0,
                    "model": req["model"],
                    "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                    "usage": usage,
                },
            )
            return
        if FakeOpenAI.credit_left <= 0:
            self._send(402, {"error": {"message": "Insufficient balance"}})
            return
        FakeOpenAI.credit_left -= 1
        p = req["messages"][0]["content"]
        if "retrieval decision assistant" in p:
            text = "Yes (retrieval is required)"
        elif "query formulation assistant" in p:
            text = (
                "Reasoning Chain: <Film; directed by; Bob Stone>\nKnowledge Gap: To continue answering the "
                "question, we need to know: Bob Stone's mother\nQuery: Bob Stone mother"
            )
        elif "knowledge integration assistant" in p:
            text = re.search(r"\[1\] (<[^>]+>)", p).group(1)
        elif "Extract knowledge triples" in p:
            title = p.split("Title: ")[1].split("\n")[0]
            text = f"<{title}; mother; Mary Stone>\n<{title}; occupation; director>"
        else:
            text = "pong"
        self._send(
            200,
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": req["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )


class FakeSearcher:
    def search(self, query, top_k=None):
        return [
            Passage("d1", "Bob Stone is the son of Mary Stone.", "Bob Stone"),
            Passage("d2", "Film is directed by Bob Stone.", "Film"),
        ]


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def load_script():
    spec = importlib.util.spec_from_file_location(
        "collect_sft", REPO_ROOT / "scripts" / "collect_sft.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_end_to_end_with_credit_exhaustion(tmp_path, server, monkeypatch):
    import asyncio

    import grasp.retrieval.encoder as enc_mod
    import grasp.retrieval.searcher as s_mod
    import grasp.sft.build as build_mod

    qfile = tmp_path / "train.jsonl"
    qfile.write_text("".join(json.dumps(q) + "\n" for q in QS), encoding="utf-8")
    mod = load_script()
    monkeypatch.setattr(mod, "fetch", lambda src, raw: qfile)
    monkeypatch.setattr(
        mod,
        "read_splits",
        lambda p: {"sft_train": [q["id"] for q in QS[:4]], "sft_val": [q["id"] for q in QS[4:]]},
    )
    monkeypatch.setattr(enc_mod, "E5Encoder", lambda **kw: HashEncoder())
    monkeypatch.setattr(s_mod.Searcher, "load", classmethod(lambda cls, *a, **k: FakeSearcher()))
    monkeypatch.setattr(
        build_mod, "hf_chat_template", lambda name: lambda u: f"<user>{u}</user><assistant>"
    )
    monkeypatch.setenv("FPT_API_KEY", "test-key")

    overrides = [
        "data=2wiki",
        f"paths.data={tmp_path.as_posix()}",
        f"teacher.client.base_url={server}",
        f"reader.client.base_url={server}",
        "wandb.mode=disabled",
        "collect.concurrency=1",
    ]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    build_mod.corpus_dir(cfg).mkdir(parents=True)  # Searcher đã được thay, chỉ cần file tồn tại
    (build_mod.corpus_dir(cfg) / "index.faiss").touch()
    build_mod.passages_path(cfg).touch()

    # mỗi câu: 1 rd + 1 qf + 1 ki + 2 doc trích triple (lần đầu) = 5 request; +1 ping
    FakeOpenAI.credit_left = (
        1 + 5 + 3 + 3 + 1
    )  # đủ cho ping + 3 câu, câu thứ 4 hết credit giữa chừng
    assert asyncio.run(mod.run(cfg)) == 2
    traj = tmp_path / "trajectories" / "2wiki.jsonl"
    recs = [json.loads(x) for x in traj.read_text(encoding="utf-8").splitlines()]
    assert [r["qid"] for r in recs] == ["train_0", "train_1", "train_2"]
    r0 = recs[0]
    assert r0["status"] == "ok" and r0["em"] == 1.0 and r0["answer"] == "Mary Stone"
    assert [c["role"] for c in r0["calls"]] == ["rd", "qf", "ki"]
    assert r0["teacher_model"] == "DeepSeek-V4-Flash" and r0["usage"]["usd"] > 0
    summary = json.loads(
        (tmp_path / "trajectories" / "summary" / "2wiki.json").read_text(encoding="utf-8")
    )
    assert summary["exit_code"] == 2 and "credit" in summary["stop_reason"]

    FakeOpenAI.credit_left = 10_000  # "nạp thêm credit" rồi chạy lại đúng lệnh
    assert asyncio.run(mod.run(cfg)) == 0
    recs = [json.loads(x) for x in traj.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["qid"] for r in recs) == [q["id"] for q in QS]
    assert {r["split"] for r in recs} == {"sft_train", "sft_val"}

    from grasp.sft.export import export

    stats = export([traj], tmp_path / "sft")
    assert stats["rows"] == {"train": 12, "val": 6}

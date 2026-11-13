import json
import sqlite3

from grasp_proxy_rollout.build_triple_cache import build_triple_cache
from grasp_proxy_rollout.retriever import (
    _SQLitePassageStore,
    _SQLiteTripleStore,
    resolve_retrieval_files,
)


def test_resolve_released_retrieval_layout(tmp_path):
    corpus = tmp_path / "2wiki_ircot"
    encoder = corpus / "e5-large-v2"
    encoder.mkdir(parents=True)
    (corpus / "passages.sqlite").touch()
    (encoder / "index.faiss").touch()

    index, passages = resolve_retrieval_files(corpus, "intfloat/e5-large-v2")
    assert index == encoder / "index.faiss"
    assert passages == corpus / "passages.sqlite"


def test_sqlite_passage_store_preserves_requested_faiss_order(tmp_path):
    path = tmp_path / "passages.sqlite"
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE passages ("row" INTEGER PRIMARY KEY, doc_id TEXT, title TEXT, text TEXT)'
    )
    connection.executemany(
        "INSERT INTO passages VALUES (?, ?, ?, ?)",
        [(0, "d0", "zero", "text zero"), (1, "d1", "one", "text one")],
    )
    connection.commit()
    connection.close()

    store = _SQLitePassageStore(path)
    try:
        records = store.fetch([1, 0])
    finally:
        store.close()
    assert [record["doc_id"] for record in records] == ["d1", "d0"]


def test_build_triple_cache_from_teacher_trajectory(tmp_path):
    source = tmp_path / "trajectory.jsonl"
    source.write_text(
        json.dumps(
            {
                "status": "ok",
                "steps": [
                    {
                        "candidates": [
                            {
                                "doc_id": "d1",
                                "head": "Lake Grove",
                                "relation": "population",
                                "tail": "11,163",
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "triples.sqlite"
    report = build_triple_cache([str(source)], str(output))
    assert report["unique_triples"] == 1

    store = _SQLiteTripleStore(output)
    try:
        triples = store.fetch(["d1"])
    finally:
        store.close()
    assert triples["d1"][0]["tail"] == "11,163"

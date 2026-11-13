import bz2
import io
import json
import tarfile
import zipfile

import numpy as np
import pytest

from grasp.retrieval.corpus import (
    iter_2wiki_contexts,
    iter_hotpotqa_abstracts,
    iter_hotpotqa_beir,
    iter_wiki18,
)
from grasp.retrieval.index import EmbeddingStore, auto_nlist, build_faiss, encode_corpus
from grasp.retrieval.searcher import Searcher
from grasp.retrieval.store import PassageStore

WORDS = [
    "apple",
    "banana",
    "cherry",
    "delta",
    "echo",
    "falcon",
    "garden",
    "harbor",
    "island",
    "jungle",
    "kettle",
    "lemon",
]


def make_store(tmp_path, n=300):
    rng = np.random.default_rng(0)
    rows = [(f"doc{i}", f"T{i}", " ".join(rng.choice(WORDS, 6))) for i in range(n)]
    rows[7] = ("doc7", "Eiffel", "eiffel tower stands in paris france")
    path = tmp_path / "passages.sqlite"
    assert PassageStore.build(path, iter(rows), commit_every=50) == n
    return PassageStore(path)


def test_store_get_keeps_order(tmp_path):
    s = make_store(tmp_path)
    got = s.get([5, 2, 999, 7])
    assert [p.doc_id for p in got] == ["doc5", "doc2", "doc7"]
    assert len(list(s.iter_range(0, 300, batch=64))) == 300


def test_encode_refuses_stale_embeddings_after_store_rebuild(tmp_path, encoder):
    store = make_store(tmp_path)
    encode_corpus(store, encoder, tmp_path / "emb", shard_size=128)
    rows = [(f"new{i}", "T", "apple banana") for i in range(200)]
    PassageStore.build(tmp_path / "other.sqlite", iter(rows))
    with pytest.raises(RuntimeError, match="passage store khác"):
        encode_corpus(PassageStore(tmp_path / "other.sqlite"), encoder, tmp_path / "emb", 128)


@pytest.mark.parametrize("index_type", ["flat", "ivfpq"])
def test_encode_index_search_with_rescore(tmp_path, encoder, index_type):
    store = make_store(tmp_path)
    emb_dir = tmp_path / "emb"
    assert encode_corpus(store, encoder, emb_dir, shard_size=128) == 3
    encode_corpus(store, encoder, emb_dir, shard_size=128)  # resume: không encode lại
    emb = EmbeddingStore(emb_dir, 128)
    assert len(emb) == 300 and emb.get([7]).dtype == np.float32
    build_faiss(emb, tmp_path / "index.faiss", index_type, nlist=8, pq_m=8, train_size=300)
    import faiss

    s = Searcher(
        faiss.read_index(str(tmp_path / "index.faiss")),
        store,
        encoder,
        emb,
        top_k=3,
        candidates=50,
        nprobe=8,
    )
    res = s.search("eiffel tower paris")
    assert res[0].doc_id == "doc7" and len(res) == 3
    assert res[0].score >= res[1].score


def test_auto_nlist():
    assert auto_nlist(21_000_000) == 16384 and auto_nlist(5_200_000) == 8192


def test_hotpotqa_builder(tmp_path):
    lines = [
        json.dumps(
            {
                "id": "12",
                "title": "Ed Wood",
                "text": ["Ed Wood is a film.", " It was made in 1994."],
            }
        ),
        json.dumps({"id": "13", "title": "Empty", "text": []}),
    ]
    inner = bz2.compress(("\n".join(lines) + "\n").encode())
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        ti = tarfile.TarInfo("abstracts/AA/wiki_00.bz2")
        ti.size = len(inner)
        tar.addfile(ti, io.BytesIO(inner))
    p = tmp_path / "a.tar.bz2"
    p.write_bytes(buf.getvalue())
    assert list(iter_hotpotqa_abstracts(p)) == [
        ("12", "Ed Wood", "Ed Wood is a film. It was made in 1994.")
    ]


def test_hotpotqa_beir_builder(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = tmp_path / "corpus.parquet"
    table = pa.table(
        {
            "_id": ["12", "13", "14"],
            "title": ["Ed Wood", "Empty", "Linked"],
            "text": ["Ed Wood is a film.", "", 'See <a href="Ed%20Wood">Ed Wood</a> here.'],
        }
    )
    pq.write_table(table, p)
    assert list(iter_hotpotqa_beir(p)) == [
        ("12", "Ed Wood", "Ed Wood is a film."),
        ("14", "Linked", "See Ed Wood here."),
    ]


def test_2wiki_builder_dedups_across_files(tmp_path):
    item = {"context": [["A", ["x is y.", "z."]], ["B", ["b."]]]}
    for name in ("train.json", "dev.json"):
        (tmp_path / name).write_text(json.dumps([item, item]))
    rows = list(iter_2wiki_contexts([tmp_path / "train.json", tmp_path / "dev.json"]))
    assert [(t, x) for _, t, x in rows] == [("A", "x is y. z."), ("B", "b.")]
    assert rows[0][0].startswith("2wiki-")


def test_2wiki_dedup_ignores_spacing_and_keeps_clean_variant(tmp_path):
    ugly = {"context": [["War (film)", ["Polish- Russian War( Wojna) is a film ."]]]}
    clean = {"context": [["War (film)", ["Polish-Russian War (Wojna) is a film."]]]}
    (tmp_path / "train.json").write_text(json.dumps([ugly]))
    (tmp_path / "dev.json").write_text(json.dumps([clean]))
    rows = list(iter_2wiki_contexts([tmp_path / "train.json", tmp_path / "dev.json"]))
    assert [x for _, _, x in rows] == ["Polish-Russian War (Wojna) is a film."]
    rows2 = list(iter_2wiki_contexts([tmp_path / "dev.json", tmp_path / "train.json"]))
    assert rows2 == rows  # doc_id + bản giữ lại không phụ thuộc thứ tự file


def test_wiki18_builder(tmp_path):
    p = tmp_path / "w.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr(
            "wiki18_100w.jsonl",
            json.dumps({"id": "0", "contents": '"Evan Morris"\nEvan Morris was a lobbyist.'})
            + "\n",
        )
    assert list(iter_wiki18(p)) == [("0", "Evan Morris", "Evan Morris was a lobbyist.")]

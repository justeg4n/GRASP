"""Đọc corpus thô theo đúng setting SPARKLE (Appendix D.1) -> luồng (doc_id, title, text).

| builder              | nguồn                                                    | ~số passage |
|----------------------|----------------------------------------------------------|-------------|
| hotpotqa_abstracts   | enwiki-20171001 abstracts do tác giả HotpotQA phát hành  | 5.2M        |
| hotpotqa_beir        | cùng bộ abstract đó, bản parquet của BEIR trên HF Hub    | 5.2M        |
| 2wiki_contexts       | gộp mọi paragraph trong context train/dev/test (IRCoT)   | ~0.4M       |
| wiki18               | DPR wiki-18 (bản jsonl của FlashRAG)                     | 21M         |

Mọi builder đều stream, không nạp cả file vào RAM (trừ bảng dedup của 2wiki, ~200MB).
"""

from __future__ import annotations

import bz2
import hashlib
import json
import re
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path

Row = tuple[str, str, str]


def iter_hotpotqa_abstracts(tar_path: str | Path) -> Iterator[Row]:
    """tar.bz2 chứa nhiều file wiki_xx.bz2; mỗi dòng: {id, title, text: [câu...]}."""
    with tarfile.open(tar_path, mode="r|bz2") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".bz2"):
                continue
            data = tar.extractfile(member).read()
            for line in bz2.decompress(data).decode("utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                text = "".join(d.get("text") or []).strip()
                if text:
                    yield str(d["id"]), d["title"], text


_LINK = re.compile(r"</?a\b[^>]*>")


def iter_hotpotqa_beir(parquet_path: str | Path) -> Iterator[Row]:
    """BeIR/hotpotqa corpus (_id, title, text): cùng abstract enwiki-20171001, cùng id với bản gốc.

    Server Stanford chỉ ~100KB/s nên dùng bản HF này. Vài trăm đoạn còn sót thẻ <a href>: bỏ thẻ, giữ chữ.
    """
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(parquet_path).iter_batches(
        batch_size=50_000, columns=["_id", "title", "text"]
    ):
        for doc_id, title, text in zip(
            *(batch.column(i).to_pylist() for i in range(3)), strict=True
        ):
            text = _LINK.sub("", text or "").strip()
            if text:
                yield str(doc_id), title, text


_ODD_SPACING = re.compile(r"\( | [,.;:)]|\w- \w")


def _dedup_key(title: str, text: str) -> str:
    return re.sub(r"\s+", "", f"{title}\t{text}").lower()


def iter_2wiki_contexts(json_paths: list[str | Path]) -> Iterator[Row]:
    """File gốc 2WikiMultihopQA (mảng JSON lớn) -> paragraph duy nhất.

    Cùng paragraph xuất hiện ở train và dev/test với khoảng trắng khác nhau ("War( Wojna" vs "War (Wojna"):
    gộp theo khóa bỏ khoảng trắng, giữ bản ít khoảng trắng lạ nhất. Giữ ~0.4M paragraph trong RAM (~200MB).
    """
    import ijson

    best: dict[str, tuple[int, str, str]] = {}
    for p in json_paths:
        with open(p, "rb") as f:
            for item in ijson.items(f, "item"):
                for title, sents in item.get("context") or []:
                    text = " ".join(s.strip() for s in sents).strip()
                    if not text:
                        continue
                    key = _dedup_key(title, text)
                    score = len(_ODD_SPACING.findall(text))
                    if key not in best or score < best[key][0]:
                        best[key] = (score, title, text)
    for key, (_, title, text) in best.items():
        yield "2wiki-" + hashlib.sha1(key.encode()).hexdigest()[:16], title, text


def iter_wiki18(zip_path: str | Path) -> Iterator[Row]:
    """FlashRAG wiki18_100w.zip: jsonl {id, contents: '"Title"\\ntext'}."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".jsonl"))
        with z.open(name) as f:
            for line in f:
                d = json.loads(line)
                title, _, text = d["contents"].partition("\n")
                yield str(d["id"]), title.strip().strip('"'), text.strip()


BUILDERS = {
    "hotpotqa_abstracts": lambda files: iter_hotpotqa_abstracts(files[0]),
    "hotpotqa_beir": lambda files: iter_hotpotqa_beir(files[0]),
    "2wiki_contexts": iter_2wiki_contexts,
    "wiki18": lambda files: iter_wiki18(files[0]),
}

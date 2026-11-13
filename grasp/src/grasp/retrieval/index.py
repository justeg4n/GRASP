"""Encode corpus bằng E5 + build FAISS index (flat | ivfpq).

Embedding fp16 lưu theo shard `.npy` (mmap được) để:
  1. chạy tiếp được khi encode bị ngắt (shard đã có thì bỏ qua);
  2. re-score chính xác top-candidates sau khi tìm bằng IVF-PQ (bù recall mất do nén).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from grasp.retrieval.encoder import passage_text
from grasp.retrieval.store import PassageStore


class EmbeddingStore:
    """Các shard 0000.npy, 0001.npy, … (fp16, mmap). Shard i chứa row [i*S, (i+1)*S)."""

    def __init__(self, emb_dir: str | Path, shard_size: int):
        self.dir, self.shard_size = Path(emb_dir), shard_size
        self.shards = [
            np.load(p, mmap_mode="r") for p in sorted(self.dir.glob("[0-9][0-9][0-9][0-9].npy"))
        ]
        self.n = sum(len(s) for s in self.shards)
        self.dim = self.shards[0].shape[1] if self.shards else 0

    def __len__(self) -> int:
        return self.n

    def get(self, rows) -> np.ndarray:
        out = np.empty((len(rows), self.dim), dtype=np.float32)
        for i, r in enumerate(rows):
            out[i] = self.shards[r // self.shard_size][r % self.shard_size]
        return out

    def iter_batches(self, batch: int):
        for s in self.shards:
            for lo in range(0, len(s), batch):
                yield np.asarray(s[lo : lo + batch], dtype=np.float32)

    def sample(self, k: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(self.n, size=min(k, self.n), replace=False))
        return self.get(rows)


def encode_corpus(
    store: PassageStore, encoder, emb_dir: str | Path, shard_size: int = 250_000
) -> int:
    """Encode mọi passage thành shard fp16. Trả về số shard.

    Shard đã có được dùng lại (chạy tiếp sau khi bị ngắt), nên emb_dir ghi dấu vân tay của store:
    store bị dựng lại mà embedding cũ còn đó -> báo lỗi thay vì trộn embedding lệch dòng.
    """
    emb_dir = Path(emb_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    fp = store_fingerprint(store)
    manifest = emb_dir / "store.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old != fp:
            raise RuntimeError(
                f"{emb_dir} được encode từ passage store khác ({old} != {fp}): xóa thư mục này "
                "và index.faiss rồi chạy lại"
            )
    elif any(emb_dir.glob("*.npy")):
        raise RuntimeError(
            f"{emb_dir} có shard nhưng không có store.json: xóa thư mục rồi chạy lại"
        )
    else:
        manifest.write_text(json.dumps(fp), encoding="utf-8")
    n_shards = math.ceil(len(store) / shard_size)
    for i in range(n_shards):
        out = emb_dir / f"{i:04d}.npy"
        if out.exists():
            continue
        lo, hi = i * shard_size, min(len(store), (i + 1) * shard_size)
        texts = [passage_text(t, x) for _, t, x in store.iter_range(lo, hi)]
        print(f"[encode] shard {i + 1}/{n_shards}: rows {lo}..{hi}")
        emb = encoder.encode_passages(texts, show_progress=True).astype(np.float16)
        tmp = emb_dir / f"{i:04d}.tmp.npy"
        np.save(tmp, emb)
        tmp.replace(out)
    return n_shards


def store_fingerprint(store: PassageStore) -> dict:
    n = len(store)
    rows = sorted({0, n // 2, n - 1}) if n else []
    return {"rows": n, "doc_ids": [p.doc_id for p in store.get(rows)]}


def auto_nlist(n: int) -> int:
    """~4*sqrt(N), làm tròn lũy thừa 2: 21M -> 16384, 5.2M -> 8192."""
    return max(64, 2 ** int(round(math.log2(4 * math.sqrt(n)))))


def build_faiss(
    emb: EmbeddingStore,
    index_path: str | Path,
    index_type: str = "ivfpq",
    nlist: int | None = None,
    pq_m: int = 64,
    train_size: int = 500_000,
    add_batch: int = 200_000,
) -> str:
    import faiss

    d = emb.dim
    if index_type == "flat":
        index = faiss.IndexFlatIP(d)
    elif index_type == "ivfpq":
        nlist = nlist or auto_nlist(len(emb))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
        sample = emb.sample(max(train_size, 40 * nlist))
        print(f"[faiss] train IVF{nlist},PQ{pq_m} trên {len(sample)} vector")
        index.train(sample)
        del sample
    else:
        raise ValueError(f"index_type không hợp lệ: {index_type}")
    added = 0
    for batch in emb.iter_batches(add_batch):
        index.add(batch)
        added += len(batch)
        print(f"[faiss] add {added}/{len(emb)}")
    index_path = Path(index_path)
    tmp = index_path.with_suffix(".tmp")
    faiss.write_index(index, str(tmp))
    tmp.replace(index_path)
    return str(index_path)

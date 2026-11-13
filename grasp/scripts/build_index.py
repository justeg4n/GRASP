"""Corpus -> passage store -> embedding E5 (theo shard) -> FAISS index. Mỗi bước chạy tiếp được khi bị ngắt.

    uv run python scripts/build_index.py data=2wiki      # nhỏ nhất, nên chạy trước để thử
    uv run python scripts/build_index.py data=hotpotqa
    uv run python scripts/build_index.py data=nq         # wiki-18, lâu nhất (~21M passage)

Bước đã xong thì bỏ qua: store có rồi không build lại, shard embedding có rồi không encode lại.
Chỉ chạy một bước: +stage=store | encode | index
"""

import time

import hydra
from omegaconf import DictConfig, OmegaConf

from grasp.data.download import fetch
from grasp.retrieval.corpus import BUILDERS
from grasp.retrieval.index import EmbeddingStore, build_faiss, encode_corpus
from grasp.retrieval.store import PassageStore
from grasp.sft.build import corpus_dir, passages_path
from grasp.utils import script_init


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    stage = cfg.get("stage", "all")
    corpus = cfg.data.corpus
    store_path, out_dir = passages_path(cfg), corpus_dir(cfg)
    t0 = time.time()

    if stage in ("all", "store") and not store_path.exists():
        files = [fetch(OmegaConf.to_container(s), cfg.paths.raw) for s in corpus.sources]
        print(f"[store] build {corpus.name} từ {[str(f) for f in files]}")
        n = PassageStore.build(store_path, BUILDERS[corpus.builder](files))
        print(f"[store] {n} passage -> {store_path} ({time.time() - t0:.0f}s)")

    if stage in ("all", "encode"):
        from grasp.retrieval.encoder import E5Encoder

        store = PassageStore(store_path)
        encoder = E5Encoder(**OmegaConf.to_container(cfg.retriever))
        n = encode_corpus(store, encoder, out_dir / "emb", cfg.retriever.shard_size)
        print(f"[encode] {n} shard -> {out_dir / 'emb'} ({time.time() - t0:.0f}s)")

    if stage in ("all", "index") and not (out_dir / "index.faiss").exists():
        emb = EmbeddingStore(out_dir / "emb", cfg.retriever.shard_size)
        if len(emb) != len(PassageStore(store_path)):
            raise RuntimeError(
                f"embedding ({len(emb)}) chưa khớp số passage — chạy lại bước encode"
            )
        ix = cfg.retriever.index
        path = build_faiss(
            emb, out_dir / "index.faiss", corpus.index_type, ix.nlist, ix.pq_m, ix.train_size
        )
        print(f"[index] {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    script_init()
    main()

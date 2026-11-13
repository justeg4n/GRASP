"""CLI for building the retriever artifacts required by full rollout export."""
from __future__ import annotations

import argparse

from grasp_proxy_rollout.retriever import build_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", default="intfloat/e5-large-v2")
    parser.add_argument("--passage-prefix", default="passage: ")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    output = build_index(
        args.corpus,
        args.output_dir,
        args.encoder,
        args.passage_prefix,
        args.batch_size,
    )
    print(f"Retriever index written to: {output}")


if __name__ == "__main__":
    main()

"""Eval in-domain / OOD / ablation. Chạy: uv run python scripts/evaluate.py data=musique"""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError(
        "TODO: run_episode trên n_eval câu, xuất metrics + per-question jsonl"
    )


if __name__ == "__main__":
    main()

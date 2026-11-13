"""Lọc trajectory đạt -> sft/{train,val}.jsonl (+ stats.json). Không gọi API, chạy lại thoải mái.

uv run python scripts/export_sft.py
uv run python scripts/export_sft.py collect.filter.f1_min=0.7 +push_to_hub=<org>/grasp-sft
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from grasp.sft.export import export, push_to_hub
from grasp.utils import script_init


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    traj_dir = Path(cfg.paths.trajectories)
    paths = sorted(traj_dir.glob("*.jsonl"))
    if not paths:
        raise SystemExit(f"không có trajectory nào trong {traj_dir}")
    stats = export(paths, cfg.paths.sft, cfg.collect.filter.em_min, cfg.collect.filter.f1_min)
    for k, v in stats.items():
        print(f"{k}: {v}")
    if cfg.get("push_to_hub"):
        push_to_hub(cfg.paths.sft, cfg.push_to_hub, private=True)
        print(f"đã push lên https://huggingface.co/datasets/{cfg.push_to_hub}")


if __name__ == "__main__":
    script_init()
    main()

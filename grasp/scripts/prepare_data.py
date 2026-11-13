"""Tải câu hỏi train + tạo split cố định (splits/{dataset}.json, commit vào git).

    uv run python scripts/prepare_data.py                 # cả 3 dataset
    uv run python scripts/prepare_data.py +only=[nq]

Split đã tồn tại thì không ghi đè (thêm +overwrite=true nếu thực sự muốn tạo lại).
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from grasp.data.download import fetch
from grasp.data.questions import load_questions, make_splits, write_splits
from grasp.utils import REPO_ROOT, script_init

DATASETS = ["hotpotqa", "2wiki", "nq"]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    only = cfg.get("only") or DATASETS
    sizes = OmegaConf.to_container(cfg.splits.sizes)
    for name in only:
        data_cfg = OmegaConf.load(REPO_ROOT / "configs" / "data" / f"{name}.yaml")
        out = REPO_ROOT / "splits" / f"{name}.json"
        path = fetch(OmegaConf.to_container(data_cfg.questions), cfg.paths.raw)
        questions = load_questions(path)
        print(f"[prepare] {name}: {len(questions)} câu từ {path}")
        if out.exists() and not cfg.get("overwrite", False):
            print(f"[prepare] {out} đã có -> giữ nguyên")
            continue
        splits = make_splits(list(questions), sizes, cfg.splits.seed)
        src = f"{data_cfg.questions.repo}/{data_cfg.questions.file}"
        write_splits(out, name, src, cfg.splits.seed, splits)
        print(f"[prepare] ghi {out}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    script_init()
    main()

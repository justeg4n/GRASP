"""Teacher (API) sinh trajectory -> ghi jsonl theo câu. Chạy lại đúng lệnh cũ để tiếp tục.

    uv run python scripts/collect_sft.py data=2wiki collect.limit=10     # smoke test
    uv run python scripts/collect_sft.py data=hotpotqa

Exit code: 0 xong | 1 lỗi code | 2 hết credit / sai key / sai model | 3 API hoặc reader sập | 4 hết budget | 130 Ctrl+C
"""

import asyncio
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from grasp.data.download import fetch
from grasp.data.questions import load_questions, read_splits, select
from grasp.sft.collector import CollectConfig, Collector
from grasp.utils import REPO_ROOT, git_commit, script_init


async def run(cfg: DictConfig) -> int:
    from grasp.retrieval.encoder import E5Encoder
    from grasp.retrieval.searcher import Searcher
    from grasp.sft.build import build_components, corpus_dir, passages_path

    name = cfg.data.name
    questions = load_questions(fetch(OmegaConf.to_container(cfg.data.questions), cfg.paths.raw))
    todo = select(
        questions,
        read_splits(REPO_ROOT / "splits" / f"{name}.json"),
        list(cfg.collect.splits),
        name,
    )

    if not (corpus_dir(cfg) / "index.faiss").exists() or not passages_path(cfg).exists():
        print(f"[collect] chưa có index cho {name}: chạy scripts/build_index.py data={name} trước")
        return 1
    encoder = E5Encoder(**OmegaConf.to_container(cfg.retriever))
    searcher = Searcher.load(corpus_dir(cfg), passages_path(cfg), cfg.retriever, encoder)
    comp = build_components(cfg, searcher, encoder)

    wandb_run = _wandb(cfg)
    traj_dir = Path(cfg.paths.trajectories)
    overrides = " ".join(a for a in sys.argv[1:])
    collector = Collector(
        episode_fn=comp.episode,
        out_path=traj_dir / f"{name}.jsonl",
        state_path=traj_dir / "state" / f"{name}.json",
        summary_path=traj_dir / "summary" / f"{name}.json",
        meter=comp.meter,
        cfg=CollectConfig.from_cfg(cfg.collect),
        preflight=comp.preflight,
        record_extra={
            "teacher_model": cfg.teacher.client.model,
            "reader_model": cfg.reader.client.model,
            "retriever": cfg.retriever.name,
            "corpus": cfg.data.corpus.name,
            "git_commit": git_commit(),
        },
        on_progress=(lambda info: wandb_run.log(info)) if wandb_run else None,
        resume_cmd=f"uv run python scripts/collect_sft.py {overrides}".strip(),
    )
    try:
        result = await collector.run(todo)
        print(
            f"[collect] trích triple: {comp.deps.extractor.misses} doc mới, "
            f"cache {len(comp.triple_cache)} doc"
        )
    finally:
        await comp.aclose()
        if wandb_run:
            wandb_run.finish()
    return result.exit_code


def _wandb(cfg):
    if cfg.wandb.mode == "disabled":
        return None
    try:
        import wandb

        return wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            job_type="collect_sft",
            name=f"collect-{cfg.data.name}",
            config=OmegaConf.to_container(cfg, resolve=False),
        )
    except Exception as e:  # W&B lỗi không được làm hỏng lần chạy tốn tiền
        print(f"[wandb] bỏ qua: {e}")
        return None


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    try:
        code = asyncio.run(run(cfg))
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    script_init()
    main()

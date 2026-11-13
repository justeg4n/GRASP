"""Entry point duy nhất cho SFT / GRPO / PPO. Chọn bằng: train=sft|grpo_g4|ppo"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    algo = cfg.train.algo
    if algo == "sft":
        raise NotImplementedError("TODO: SFT warm-start")
    if algo == "grpo":
        raise NotImplementedError("TODO: group_rollout -> reward -> advantage -> GRPOTrainer.step")
    if algo == "ppo":
        raise NotImplementedError("TODO: PPO baseline (RQ2/RQ6)")
    raise ValueError(f"algo không hợp lệ: {algo}")


if __name__ == "__main__":
    main()

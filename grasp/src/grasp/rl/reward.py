"""Reward hệ thống: alpha * Recall(context) + (1 - alpha) * F1(answer)."""

from __future__ import annotations

from grasp.types import Trajectory


def trajectory_reward(traj: Trajectory, alpha: float = 0.3) -> float:
    raise NotImplementedError("TODO: dùng grasp.eval.metrics, chuẩn hoá đáp án trước khi so")


def monte_carlo_decompose(trajs: list[Trajectory]) -> dict[tuple[str, str], float]:
    """Gán reward cho từng (state, action) của từng agent: trung bình reward các lá đi qua nó."""
    raise NotImplementedError("TODO: duyệt cây node_id/parent_id, trung bình reward lá")

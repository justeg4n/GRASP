"""Group-structured rollout — trái tim của GRASP.

Tại L bước đầu: sample G joint action (RD/QF/KI) thay vì đi 1 đường.
Prune nhánh có context trùng với nhánh em/anh (vì sẽ dẫn tới cùng bước tiếp theo).
Sau bước L: chạy đơn luồng on-policy để chặn chi phí.
"""

from __future__ import annotations

from grasp.types import Trajectory


def group_rollout(
    question: str, gold: list[str], agents, reader, searcher, cfg
) -> list[Trajectory]:
    """Trả về G trajectory cùng prefix -> chính là 'group' cho GRPO."""
    raise NotImplementedError("TODO: BFS theo bước, prune context trùng, gắn node_id/parent_id")


def context_signature(traj: Trajectory) -> str:
    """Khoá để phát hiện nhánh trùng context (dùng khi prune)."""
    raise NotImplementedError("TODO: hash tập doc_id trong context")

import pytest

from grasp.rl.grpo import group_advantages


@pytest.mark.xfail(reason="chưa implement", strict=False)
def test_zero_variance_group_gives_zero_advantage():
    assert group_advantages([0.5, 0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0, 0.0]

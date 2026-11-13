import pytest


@pytest.mark.xfail(reason="chưa implement", strict=False)
def test_duplicate_context_branch_is_pruned():
    """Hai nhánh cùng tập doc_id phải bị gộp còn 1."""
    raise NotImplementedError

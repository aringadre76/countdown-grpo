import pytest

from countdown_grpo.data import canonical_key_text
from countdown_grpo.prepare_confirmation import observed_keys, select_source


def test_confirmation_excludes_canonical_permutations_and_nested_old_rollouts():
    key = canonical_key_text(10, [2, 3, 5])
    old = {"rollouts": [{"target": 10, "nums": [5, 2, 3]}]}
    assert set(observed_keys(old)) == {key}
    tasks = [{"task_id": "a", "target": 10, "nums": [2, 3, 5], "split": "test"},
             {"task_id": "b", "target": 11, "nums": [2, 3, 6], "split": "test"}]
    selected = select_source(tasks, {key}, 1, 42)
    assert selected[0]["task_id"] == "b"
    assert selected[0]["split"] == "source_confirmation"
    assert selected[0]["oracle_solvable"]
    assert "witness" not in selected[0]
    with pytest.raises(ValueError, match="not enough"):
        select_source(tasks, {key}, 2, 42)
    tasks[1]["split"] = "dev"
    with pytest.raises(ValueError, match="held-out"):
        select_source(tasks, {key}, 1, 42)

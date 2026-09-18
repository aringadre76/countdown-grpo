import pytest

from countdown_grpo.comparison import paired_bootstrap, task_outcomes


def test_bootstrap_pairs_tasks_and_is_reproducible():
    baseline = {"a": 0, "b": 0}
    trained = {"a": 1, "b": 1}
    result = paired_bootstrap(baseline, trained)
    assert result["paired_gain_percentile_95"] == [1, 1]
    assert result == paired_bootstrap(baseline, trained)
    with pytest.raises(ValueError, match="identical"):
        paired_bootstrap(baseline, {"a": 1})


def test_pass_four_is_one_task_outcome_not_four_observations():
    rows = [{"task_id": "a", "target": 10, "nums": [2, 3, 5],
             "generation_mode": "sample", "reward": float(index == 0)}
            for index in range(4)]
    assert task_outcomes(rows, "sample") == {"a": 1}
    with pytest.raises(ValueError, match="expected 4"):
        task_outcomes(rows[:3], "sample")

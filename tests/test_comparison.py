import pytest

from countdown_grpo.comparison import (
    audit_pairs,
    normalized_record,
    paired_bootstrap,
    task_outcomes,
)


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


def test_normalization_is_diagnostic_and_preserves_integer_and_clipping_contract():
    row = {"task_id": "a", "target": 10, "nums": [2, 3, 5],
           "generation_mode": "greedy", "reward": 0.0, "split": "source_confirmation",
           "raw_completion": "2 + 3 + 5 = 10", "truncation": False}
    assert normalized_record(row)["reward"] == 1.0
    assert row["reward"] == 0.0
    trained = {**row, "raw_completion": "<answer>2+3+5</answer>", "reward": 1.0}
    assert audit_pairs([row], [trained])[0]["rubric_category"] == "format_only_repair_sufficient"
    assert normalized_record({**row, "truncation": True})["reward"] == 0.0
    fractional = {**row, "target": 8, "nums": [8, 3, 3], "raw_completion": "8/3*3 = 8"}
    assert normalized_record(fractional)["normalized_failure_category"] == "non_integer_intermediate"

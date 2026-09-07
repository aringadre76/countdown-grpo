import json

import pytest

from countdown_grpo.data import (
    assert_no_canonical_leakage,
    canonicalise_source_rows,
    make_fresh_tasks,
    split_canonical_tasks,
    write_jsonl,
)


def test_canonicalisation_deduplicates_permuted_tasks_and_hides_witnesses():
    tasks, audit = canonicalise_source_rows(
        [
            {"target": 7, "nums": [8, 2, 3]},
            {"target": 7, "nums": [3, 8, 2]},
            {"target": 15, "nums": [10, 4, 1, 3]},
        ]
    )
    assert audit["raw_row_count"] == 3
    assert audit["canonical_task_count"] == 2
    assert audit["canonical_duplicate_row_count"] == 1
    assert all("witness" not in task and "7" not in task["prompt"].split("Target:")[0] for task in tasks)


def test_seeded_splits_have_no_canonical_leakage():
    tasks, _ = canonicalise_source_rows(
        {"target": target, "nums": [target, 1, 1]} for target in range(3, 23)
    )
    splits = split_canonical_tasks(tasks, seed=7, dev_fraction=0.2, test_fraction=0.2)
    assert {name: len(records) for name, records in splits.items()} == {"train": 12, "dev": 4, "test": 4}
    assert_no_canonical_leakage(splits)


def test_leakage_check_rejects_cross_split_duplicate():
    task = {"canonical_key": "[7,[2,3,8]]"}
    with pytest.raises(ValueError, match="canonical leakage"):
        assert_no_canonical_leakage({"train": [task], "test": [task]})


def test_fresh_suite_is_solvable_and_disjoint_from_forbidden_keys(tmp_path):
    fresh = make_fresh_tasks(count=8, seed=11, forbidden_keys=["[7,[2,3,8]]"])
    assert len(fresh) == 8
    assert all(task["oracle_solvable"] and task["canonical_key"] != "[7,[2,3,8]]" for task in fresh)
    assert all("witness" not in task for task in fresh)
    output = tmp_path / "fresh.jsonl"
    write_jsonl(output, fresh)
    assert all("witness" not in json.loads(line) for line in output.read_text().splitlines())

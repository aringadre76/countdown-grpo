import json

import pytest

from countdown_grpo.data import file_sha256, write_jsonl
from countdown_grpo.search_report import (
    _select_candidate,
    _task_outcomes,
    _validated_task_groups,
    generate_report,
)


def _record(mode, index, reward):
    return {
        "task_id": "task-a",
        "split": "source_confirmation",
        "target": 9,
        "nums": [2, 3, 4],
        "oracle_solvable": True,
        "generation_mode": mode,
        "sample_index": index,
        "reward": reward,
        "raw_completion": f"proposal-{mode}-{index}",
        "extracted_expression": "(2+3)+4" if reward else None,
        "verifier_result": {"valid": bool(reward)},
    }


def _group(sample_rewards, greedy_reward=0):
    rows = [_record("greedy", 0, greedy_reward)]
    rows.extend(_record("sample", index, reward) for index, reward in enumerate(sample_rewards))
    return _validated_task_groups(rows, samples_per_task=8)


def test_search_outcomes_include_greedy_then_ordered_sampled_candidates():
    groups = _group([0, 0, 1, 0, 0, 0, 0, 0])
    outcomes = _task_outcomes(groups)["source_confirmation"]
    assert outcomes["greedy_pass_at_1"]["task-a"] == 0.0
    assert outcomes["sample_pass_at_4"]["task-a"] == 1.0
    assert outcomes["sample_pass_at_8"]["task-a"] == 1.0
    assert outcomes["search_at_1"]["task-a"] == 0.0
    assert outcomes["search_at_5"]["task-a"] == 1.0
    assert outcomes["search_at_9"]["task-a"] == 1.0


def test_selector_returns_first_exact_candidate_and_never_scores_prose():
    group = _group([0, 0, 1, 1, 0, 0, 0, 0])["task-a"]
    selected = _select_candidate(group)
    assert selected["selected_candidate_mode"] == "sample"
    assert selected["selected_sample_index"] == 2
    assert selected["candidates_checked"] == 4
    assert selected["reward"] == 1.0


def test_selector_returns_no_candidate_when_all_verifier_rewards_are_zero():
    group = _group([0] * 8)["task-a"]
    selected = _select_candidate(group)
    assert selected["selected_candidate_mode"] is None
    assert selected["candidates_checked"] == 9
    assert selected["failure_category"] == "no_exact_candidate_in_budget"
    assert selected["reward"] == 0.0


def test_groups_reject_missing_or_reordered_sample_indices():
    rows = [_record("greedy", 0, 0)]
    rows.extend(_record("sample", index, 0) for index in range(8))
    rows[-1]["sample_index"] = 9
    with pytest.raises(ValueError, match="indices"):
        _validated_task_groups(rows, samples_per_task=8)


def test_groups_reject_inconsistent_task_identity():
    rows = [_record("greedy", 0, 0)]
    rows.extend(_record("sample", index, 0) for index in range(8))
    rows[1]["target"] = 10
    with pytest.raises(ValueError, match="task fields differ"):
        _validated_task_groups(rows, samples_per_task=8)


def test_generate_report_writes_paired_metrics_and_selected_candidates(tmp_path):
    freeze_path = tmp_path / "frozen-design.json"
    freeze_path.write_text(json.dumps({
        "status": "frozen",
        "experiment_id": "search-test",
        "confirmation": {"paired_bootstrap": {"resamples": 100, "seed": 23}},
        "budget": {"hard_cap_gpu_process_hours": 6.0},
    }))
    task_rows = []
    for split, task_id in (("source_confirmation", "source-a"), ("fresh_confirmation", "fresh-a")):
        task_rows.append({"task_id": task_id, "split": split, "target": 9, "nums": [2, 3, 4]})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "freeze_sha256": file_sha256(freeze_path),
        "source_task_ids": ["source-a"],
        "fresh_task_ids": ["fresh-a"],
    }))
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(json.dumps({"gpu": "test-device"}))

    eval_paths = []
    for policy in ("base", "sft-42", "sft-43", "sft-44"):
        rows = []
        for task in task_rows:
            rows.append({**task, "oracle_solvable": True, "generation_mode": "greedy", "sample_index": 0, "reward": 0.0, "truncation": False, "completion_length": 8, "failure_category": "wrong_target", "raw_completion": "bad", "extracted_expression": None, "verifier_result": {"valid": False}})
            for index in range(8):
                reward = float(policy != "base" and index == 2)
                rows.append({**task, "oracle_solvable": True, "generation_mode": "sample", "sample_index": index, "reward": reward, "truncation": False, "completion_length": 9, "failure_category": "ok" if reward else "wrong_target", "raw_completion": f"sample-{index}", "extracted_expression": "2+3+4" if reward else None, "verifier_result": {"valid": bool(reward)}})
        for row in rows:
            row["device"] = "cuda"
        path = tmp_path / f"{policy}.jsonl"
        write_jsonl(path, rows)
        path.with_suffix(".summary.json").write_text(json.dumps({
            "record_count": len(rows),
            "wall_time_seconds": 1.0,
            "truncation_rate": 0.0,
        }))
        eval_paths.append(path)

    report_dir = tmp_path / "report"
    report = generate_report(
        baseline_path=eval_paths[0],
        sft_paths=eval_paths[1:],
        freeze_path=freeze_path,
        manifest_path=manifest_path,
        environment_path=environment_path,
        output_dir=report_dir,
    )
    assert report["decision"]["positive_criterion_met"] is True
    assert report["compute"]["completion_count"] == 72
    assert report["source_solvable_only"]["task_count"] == 1
    selected = [json.loads(line) for line in (report_dir / "selected-candidates.jsonl").read_text().splitlines()]
    assert len(selected) == 8
    assert all(row["selected_candidate_mode"] == "sample" for row in selected if row["policy"] != "base")
    assert all((report_dir / name).is_file() for name in ("comparison.json", "report.md", "comparison.svg"))

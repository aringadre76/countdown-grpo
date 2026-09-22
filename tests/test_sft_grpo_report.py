import pytest

from countdown_grpo.sft_grpo_report import _compare, _validate_paired


def _records(split="source_confirmation"):
    output = []
    for task_index in range(2):
        for mode, count in (("greedy", 1), ("sample", 4)):
            for sample_index in range(count):
                output.append(
                    {
                        "task_id": f"{split}-{task_index}",
                        "split": split,
                        "target": 9,
                        "nums": [2, 3, 4],
                        "prompt": "Reach 9 using 2, 3, 4.",
                        "rendered_prompt": "Reach 9 using 2, 3, 4.",
                        "generation_config": {"temperature": 1.0, "top_p": 0.95},
                        "generation_mode": mode,
                        "sample_index": sample_index,
                        "seed": 42,
                        "model_id": "Qwen/Qwen3.5-0.8B-Base",
                        "revision": "revision",
                        "oracle_solvable": True,
                        "reward": float(task_index == 0 and sample_index == 0),
                        "raw_completion": "<answer>2 + 3 + 4</answer>",
                        "truncation": False,
                    }
                )
    return output


def test_paired_comparison_includes_task_level_bootstrap_and_solvable_subset():
    rows = _records()
    result = _compare(rows, rows, bootstrap_seed=20260922)
    assert result["source_confirmation"]["greedy"]["tasks"] == 2
    assert result["source_confirmation"]["sample"]["tasks"] == 2
    assert result["source_confirmation"]["greedy"]["paired_gain"] == 0
    assert result["source_confirmation"]["solvable_only"]["sample"]["tasks"] == 2


def test_paired_comparison_rejects_generation_or_task_mismatch():
    left = _records()
    right = _records()
    right.pop()
    with pytest.raises(ValueError, match="identical unique task/generation"):
        _validate_paired(left, right)

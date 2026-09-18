import pytest

from countdown_grpo.confirmation_report import compare_records


def test_confirmation_requires_matched_generation_and_puzzle_identity():
    base = [{"task_id": "a", "target": 10, "nums": [2, 3, 5], "prompt": "solve",
             "generation_mode": mode, "sample_index": index, "seed": index + 42,
             "generation_config": {"max_new_tokens": 128, "do_sample": mode == "sample"},
             "oracle_solvable": True, "split": "fresh_confirmation", "reward": 0.0,
             "raw_completion": "2+3-5", "truncation": False}
            for mode, count in [("greedy", 1), ("sample", 4)] for index in range(count)]
    trained = [{**row, "reward": 1.0, "raw_completion": "2+3+5"} for row in base]
    result = compare_records(base, trained)
    assert result["fresh_confirmation"]["greedy"]["paired_gain"] == 1.0
    trained[0]["seed"] = 999
    with pytest.raises(ValueError, match="identities"):
        compare_records(base, trained)

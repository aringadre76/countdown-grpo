import pytest

from countdown_grpo.confirmation_report import compare_records, write_confirmation_svg


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


def test_confirmation_plot_uses_saved_seed_and_aggregate_rates(tmp_path):
    values = {"baseline_rate": 0.01, "trained_rate": 0.1, "paired_gain": 0.09,
              "paired_gain_percentile_95": [0.03, 0.15]}
    comparison = {
        "seeds": {
            f"seed-{seed}": {"measurements": {
                split: {mode: {**values, "trained_rate": 0.1 + seed / 1000}
                        for mode in ("greedy", "sample")}
                for split in ("source_confirmation", "fresh_confirmation")
            }}
            for seed in (42, 43, 44)
        },
        "aggregate": {
            split: {mode: values for mode in ("greedy", "sample")}
            for split in ("source_confirmation", "fresh_confirmation")
        },
    }
    output = tmp_path / "comparison.svg"
    write_confirmation_svg(output, comparison)
    svg = output.read_text()
    assert "Frozen Countdown confirmation accuracy" in svg
    assert "Source confirmation" in svg
    assert "Fresh confirmation" in svg
    assert "Seed 44" in svg
    assert "Mean" in svg

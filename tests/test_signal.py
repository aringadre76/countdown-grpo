from countdown_grpo.signal import exploration_signal


def test_greedy_success_does_not_count_as_a_mixed_sampled_group():
    records = [
        {"task_id": "a", "generation_mode": "greedy", "reward": 1, "truncation": False},
        {"task_id": "a", "generation_mode": "sample", "reward": 0, "truncation": True},
        {"task_id": "a", "generation_mode": "sample", "reward": 0, "truncation": False},
    ]
    signal = exploration_signal(records)
    assert signal["sampled_positive_tasks"] == 0
    assert signal["sampled_mixed_tasks"] == 0
    assert signal["truncation_rate"] == 1 / 3
    assert not signal["diagnostic_gate_passed"]

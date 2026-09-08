from countdown_grpo.evaluate import summarise_records


def test_summary_reports_exact_legal_and_pass_at_k_metrics():
    records = [
        {
            "task_id": "a",
            "generation_mode": "greedy",
            "reward": 1.0,
            "failure_category": "ok",
            "truncation": False,
            "completion_length": 4,
            "num_count": 3,
            "split": "test",
        },
        {
            "task_id": "b",
            "generation_mode": "sample",
            "reward": 0.0,
            "failure_category": "wrong_target",
            "truncation": False,
            "completion_length": 7,
            "num_count": 4,
            "split": "fresh_test",
        },
        {
            "task_id": "c",
            "generation_mode": "sample",
            "reward": 0.0,
            "failure_category": "non_integer_intermediate",
            "truncation": False,
            "completion_length": 9,
            "num_count": 3,
            "split": "test",
            "target": 8,
            "extracted_expression": "8 / 3 * 3",
        },
        {
            "task_id": "b",
            "generation_mode": "sample",
            "reward": 1.0,
            "failure_category": "ok",
            "truncation": False,
            "completion_length": 5,
            "num_count": 4,
            "split": "fresh_test",
        },
    ]
    summary = summarise_records(records)
    assert summary["exact_solve_rate"] == 0.5
    assert summary["legal_expression_rate"] == 0.75
    assert summary["target_hit_but_illegal_rate"] == 0.25
    assert summary["pass_at_k_by_generation_mode"] == {"greedy": 1.0, "sample": 0.5}

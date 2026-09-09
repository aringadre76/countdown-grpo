from countdown_grpo.evaluate import _render_prompt, summarise_records


class _ChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template_kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT[{chat_template_kwargs['enable_thinking']}]:{messages[0]['content']}"


def test_control_chat_prompt_can_disable_thinking_without_changing_raw_default():
    tokenizer = _ChatTokenizer()
    assert _render_prompt(tokenizer, "solve", False, False) == "solve"
    assert _render_prompt(tokenizer, "solve", True, True) == "CHAT[False]:solve"


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

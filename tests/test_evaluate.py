from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from countdown_grpo.evaluate import _generate, _render_prompt, summarise_records


@pytest.mark.parametrize("last_token,truncated", [(99, False), (98, False), (7, True)])
def test_token_limit_termination_matches_trl_eos_and_pad_rule(last_token, truncated):
    class Input:
        shape = (1, 2)

        def to(self, device):
            return self

    class Generated:
        shape = (2,)

        def __getitem__(self, index):
            return [5, last_token][index]

    class Output:
        def __getitem__(self, index):
            assert index == (0, slice(2, None))
            return Generated()

    class Tokenizer:
        eos_token_id = 99
        pad_token_id = 98

        def __call__(self, prompt, return_tensors):
            return {"input_ids": Input()}

        def decode(self, tokens, skip_special_tokens):
            return "5+3+2"

    def generate(**options):
        assert options["eos_token_id"] == 99
        assert options["pad_token_id"] == 98
        return Output()

    result = _generate(model=SimpleNamespace(generate=generate), tokenizer=Tokenizer(),
                       torch=SimpleNamespace(inference_mode=nullcontext), device="cpu",
                       prompt="solve", generation_mode="greedy", max_new_tokens=2,
                       temperature=1, top_p=0.95)
    assert result == ("5+3+2", 2, truncated)


class _ChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT[{enable_thinking}]:{messages[0]['content']}"


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

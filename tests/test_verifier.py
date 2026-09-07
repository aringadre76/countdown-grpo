from fractions import Fraction

import pytest

from countdown_grpo.oracle import reachable_values, solve_countdown
from countdown_grpo.rewards import countdown_reward
from countdown_grpo.verifier import extract_expression, verify_completion, verify_expression


@pytest.mark.parametrize(
    ("expression", "target", "nums"),
    [
        ("(44 - 19) * 2", 50, [44, 19, 2]),
        ("8 / 2 + 3", 7, [8, 2, 3]),
        ("(10 - (4 + 1)) * 3", 15, [10, 4, 1, 3]),
    ],
)
def test_valid_expressions(expression, target, nums):
    result = verify_expression(expression, target=target, nums=nums)
    assert result.valid
    assert result.reason == "ok"
    assert result.value == Fraction(target)


@pytest.mark.parametrize(
    ("expression", "target", "nums", "reason"),
    [
        ("8 / 3 * 3", 8, [8, 3, 3], "non_integer_intermediate"),
        ("1 + 1 + 3", 5, [1, 2, 3], "wrong_number_multiset"),
        ("2 + 3", 5, [2, 2, 3], "wrong_number_multiset"),
        ("12 + 3", 15, [1, 2, 3], "wrong_number_multiset"),
        ("2 ** 3", 8, [2, 3], "unsupported_syntax"),
        ("2 // 3", 0, [2, 3], "unsupported_syntax"),
        ("2 % 3", 2, [2, 3], "unsupported_syntax"),
        ("-2 + 3", 1, [2, 3], "unsupported_syntax"),
        ("abs(2) + 3", 5, [2, 3], "unsupported_syntax"),
        ("2 / (3 - 3)", 1, [2, 3, 3], "division_by_zero"),
        ("(2 + 3", 5, [2, 3], "syntax_error"),
        ("2 +", 2, [2, 3], "syntax_error"),
        ("2 + 3", 6, [2, 3], "wrong_target"),
    ],
)
def test_rejected_expressions_have_stable_categories(expression, target, nums, reason):
    result = verify_expression(expression, target=target, nums=nums)
    assert not result.valid
    assert result.reason == reason


def test_answer_tag_is_preferred_and_last_well_formed_span_wins():
    completion = "<answer>1 + 2</answer> prose <ANSWER> (44 - 19) * 2 </ANSWER>"
    assert extract_expression(completion) == "(44 - 19) * 2"
    assert verify_completion(completion, 50, [44, 19, 2]).valid


@pytest.mark.parametrize(
    ("completion", "reason"),
    [
        ("", "no_answer"),
        ("working: 44 - 19 * 2", "unsupported_syntax"),
        ("<answer>44 - 19", "malformed_answer"),
        ("Target: 50", "unsupported_syntax"),
        ("<answer> (44 - 19) * 2 </answer> </s>", "ok"),
    ],
)
def test_completion_extraction_does_not_reward_prose(completion, reason):
    result = verify_completion(completion, 50, [44, 19, 2])
    assert result.reason == reason


def test_truncated_completion_is_not_rewarded():
    result = verify_completion("<answer>(44 - 19)", 50, [44, 19, 2], truncated=True)
    assert result.reason == "truncated"
    assert not result.valid


def test_reward_accepts_text_and_chat_style_completions():
    completions = ["<answer>2 + 3</answer>", [{"role": "assistant", "content": "<answer>4 / 2</answer>"}]]
    rewards = countdown_reward(completions, target=[5, 2], nums=[[2, 3], [4, 2]])
    assert rewards == [1.0, 1.0]


def test_oracle_witnesses_are_accepted_by_independent_verifier():
    nums = [10, 4, 1, 3]
    result = solve_countdown(nums, 15)
    assert result.solvable
    assert result.witness is not None
    verdict = verify_expression(result.witness, 15, nums)
    assert verdict.valid
    assert result.reachable_count == len(reachable_values(nums))
    audit_result = solve_countdown(nums, 15, include_witness=False)
    assert audit_result.solvable
    assert audit_result.witness is None
    assert audit_result.reachable_count == result.reachable_count


def test_oracle_handles_duplicate_inputs_and_unsolvable_targets():
    result = solve_countdown([2, 2, 3], 7)
    assert result.solvable
    assert result.witness is not None
    assert verify_expression(result.witness, 7, [2, 2, 3]).valid
    assert not solve_countdown([1, 1, 1], 7).solvable

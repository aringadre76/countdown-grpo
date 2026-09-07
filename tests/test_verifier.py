from fractions import Fraction

from countdown_grpo.verifier import extract_expression, verify_expression


def test_valid_expression_uses_each_number_once():
    result = verify_expression("(44 - 19) * 2", target=50, nums=[44, 19, 2])
    assert result.valid
    assert result.value == Fraction(50)


def test_rejects_reused_number():
    result = verify_expression("44 + 44", target=88, nums=[44, 19, 2])
    assert not result.valid


def test_rejects_unused_number():
    result = verify_expression("44 + 19", target=63, nums=[44, 19, 2])
    assert not result.valid


def test_rejects_division_by_zero():
    result = verify_expression("44 / (19 - 19) + 2", target=1, nums=[44, 19, 19, 2])
    assert not result.valid


def test_answer_tag_is_preferred():
    assert extract_expression("working...\n<answer> (44 - 19) * 2 </answer>") == "(44 - 19) * 2"

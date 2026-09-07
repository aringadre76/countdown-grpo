"""Independent exhaustive Countdown solvability oracle.

This module deliberately does not import the verifier. It independently
enumerates legal integer-only binary-expression trees and is used only for
dataset quality checks and evaluation analysis; callers must never place its
witness expressions in prompts or training examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class OracleResult:
    solvable: bool
    witness: str | None = None
    reachable_count: int = 0


def reachable_values(nums: list[int]) -> dict[int, str]:
    """Return every reachable exact integer and one legal witness for it.

    Every recursive state is already an integer, and division is included only
    when exact, so returned witnesses satisfy the core arithmetic contract.
    """

    if not nums:
        return {}

    def search(values: tuple[tuple[int, str], ...]) -> dict[int, str]:
        if len(values) == 1:
            value, expression = values[0]
            return {value: expression}

        found: dict[int, str] = {}
        for left_index, right_index in combinations(range(len(values)), 2):
            left_value, left_expression = values[left_index]
            right_value, right_expression = values[right_index]
            remaining = tuple(
                item for index, item in enumerate(values) if index not in {left_index, right_index}
            )
            operations: list[tuple[int, str]] = [
                (left_value + right_value, f"({left_expression} + {right_expression})"),
                (left_value * right_value, f"({left_expression} * {right_expression})"),
                (left_value - right_value, f"({left_expression} - {right_expression})"),
                (right_value - left_value, f"({right_expression} - {left_expression})"),
            ]
            if right_value != 0 and left_value % right_value == 0:
                operations.append((left_value // right_value, f"({left_expression} / {right_expression})"))
            if left_value != 0 and right_value % left_value == 0:
                operations.append((right_value // left_value, f"({right_expression} / {left_expression})"))

            for result, expression in operations:
                found.update(search(remaining + ((result, expression),)))
        return found

    return search(tuple((int(number), str(int(number))) for number in nums))


def solve_countdown(nums: list[int], target: int, *, include_witness: bool = True) -> OracleResult:
    """Determine solvability without supplying a solution to any model prompt."""

    values = reachable_values(nums)
    return OracleResult(
        solvable=int(target) in values,
        witness=values.get(int(target)) if include_witness else None,
        reachable_count=len(values),
    )

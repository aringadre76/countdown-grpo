"""Independent exhaustive Countdown solvability oracle.

This module deliberately does not import the verifier. It independently
enumerates legal integer-only binary-expression trees and is used only for
dataset quality checks and evaluation analysis; callers must never place its
witness expressions in prompts or training examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache, lru_cache
from itertools import combinations


@dataclass(frozen=True)
class OracleResult:
    solvable: bool
    witness: str | None = None
    reachable_count: int = 0


@lru_cache(maxsize=16_384)
def _reachable_integer_values(numbers: tuple[int, ...]) -> frozenset[int]:
    """Fast value-only dynamic program used by large dataset audits.

    The state contains only valid integer partial results. It is independent
    from the parser and has no witness strings to accidentally expose during
    dataset preparation.
    """

    @cache
    def search(values: tuple[int, ...]) -> frozenset[int]:
        if len(values) == 1:
            return frozenset(values)
        found: set[int] = set()
        for left_index, right_index in combinations(range(len(values)), 2):
            left_value = values[left_index]
            right_value = values[right_index]
            remaining = tuple(
                value for index, value in enumerate(values) if index not in {left_index, right_index}
            )
            operations = {
                left_value + right_value,
                left_value * right_value,
                left_value - right_value,
                right_value - left_value,
            }
            if right_value != 0 and left_value % right_value == 0:
                operations.add(left_value // right_value)
            if left_value != 0 and right_value % left_value == 0:
                operations.add(right_value // left_value)
            for result in operations:
                found.update(search(tuple(sorted((*remaining, result)))))
        return frozenset(found)

    return search(tuple(sorted(numbers)))


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

    if not include_witness:
        values = _reachable_integer_values(tuple(sorted(int(number) for number in nums)))
        return OracleResult(
            solvable=int(target) in values,
            witness=None,
            reachable_count=len(values),
        )

    values = reachable_values(nums)
    return OracleResult(
        solvable=int(target) in values,
        witness=values.get(int(target)) if include_witness else None,
        reachable_count=len(values),
    )

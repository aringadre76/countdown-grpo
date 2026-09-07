"""TRL-compatible reward functions."""

from __future__ import annotations

from collections.abc import Sequence

from .verifier import verify_completion


def _completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion)


def countdown_reward(
    completions: list[object],
    target: list[int],
    nums: list[list[int]],
    **_: object,
) -> list[float]:
    """Give 1.0 for a legal exact solution and 0.0 otherwise."""

    rewards: list[float] = []
    for completion, task_target, task_nums in zip(completions, target, nums, strict=True):
        result = verify_completion(_completion_text(completion), int(task_target), list(task_nums))
        rewards.append(1.0 if result.valid else 0.0)
    return rewards

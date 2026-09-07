"""Dependency-light GRPO configuration and binary-reward diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, pvariance
from typing import Any

from .rewards import _completion_text
from .verifier import extract_expression, verify_completion

LORA_TARGET_SUFFIXES = (
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)


def effective_generation_batch_size(
    *, per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> int:
    """TRL 1.12's default generation batch from its documented config fields."""

    return per_device_train_batch_size * gradient_accumulation_steps * world_size


def validate_grpo_batch(
    *, per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    world_size: int = 1,
) -> int:
    """Reject a partial GRPO prompt group before a costly trainer construction."""

    if num_generations < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    generation_batch = effective_generation_batch_size(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=world_size,
    )
    if generation_batch % num_generations:
        raise ValueError(
            f"generation batch {generation_batch} is not divisible by num_generations {num_generations}"
        )
    return generation_batch


def select_lora_target_suffixes(module_names: Sequence[str]) -> list[str]:
    """Return only target suffixes observed in a live Qwen3.5 module tree."""

    selected = [
        suffix for suffix in LORA_TARGET_SUFFIXES if any(name.endswith(suffix) for name in module_names)
    ]
    required_linear = {"in_proj_qkv", "in_proj_z", "out_proj"}
    required_full = {"q_proj", "k_proj", "v_proj", "o_proj"}
    if not required_linear.issubset(selected) or not required_full.issubset(selected):
        raise ValueError("live model is missing required Qwen3.5 linear or full attention LoRA targets")
    return selected


@dataclass
class RewardTelemetry:
    """Binary reward callable plus exact group-level diagnostics for GRPO logs."""

    num_generations: int

    def __post_init__(self) -> None:
        self.__name__ = "countdown_exact_binary_reward"
        self.events: list[dict[str, Any]] = []

    def __call__(
        self,
        completions: list[object],
        target: list[int],
        nums: list[list[int]],
        **kwargs: object,
    ) -> list[float]:
        rewards: list[float] = []
        details: list[dict[str, Any]] = []
        truncated_values = kwargs.get("truncated")
        for index, (completion, task_target, task_nums) in enumerate(
            zip(completions, target, nums, strict=True)
        ):
            truncated = False
            if isinstance(truncated_values, Sequence) and not isinstance(truncated_values, str):
                truncated = bool(truncated_values[index])
            text = _completion_text(completion)
            verdict = verify_completion(text, int(task_target), list(task_nums), truncated=truncated)
            reward = 1.0 if verdict.valid else 0.0
            rewards.append(reward)
            details.append(
                {
                    "target": int(task_target),
                    "nums": [int(number) for number in task_nums],
                    "raw_completion": text,
                    "extracted_expression": extract_expression(text),
                    "reward": reward,
                    "failure_category": verdict.reason,
                    "completion_length": len(text),
                    "completion_length_unit": "characters",
                    "truncated": truncated,
                }
            )
        self.events.append(self._summarise_event(rewards, details))
        return rewards

    def _summarise_event(self, rewards: list[float], details: list[dict[str, Any]]) -> dict[str, Any]:
        groups = [rewards[index : index + self.num_generations] for index in range(0, len(rewards), self.num_generations)]
        full_groups = [group for group in groups if len(group) == self.num_generations]
        all_zero = sum(all(reward == 0.0 for reward in group) for group in full_groups)
        all_one = sum(all(reward == 1.0 for reward in group) for group in full_groups)
        mixed = sum(0.0 < sum(group) < len(group) for group in full_groups)
        failures = Counter(str(detail["failure_category"]) for detail in details)
        lengths = [int(detail["completion_length"]) for detail in details]
        return {
            "mean_reward": fmean(rewards) if rewards else None,
            "reward_variance": pvariance(rewards) if len(rewards) > 1 else 0.0,
            "positive_completion_rate": fmean(rewards) if rewards else None,
            "mixed_reward_group_rate": mixed / len(full_groups) if full_groups else None,
            "all_zero_group_rate": all_zero / len(full_groups) if full_groups else None,
            "all_one_group_rate": all_one / len(full_groups) if full_groups else None,
            "reward_group_count": len(full_groups),
            "failure_categories": dict(sorted(failures.items())),
            "completion_length_mean": fmean(lengths) if lengths else None,
            "completion_length_unit": "characters",
            "truncation_rate": fmean([float(detail["truncated"]) for detail in details]) if details else None,
            "rollouts": details,
        }

    def pop_events(self) -> list[dict[str, Any]]:
        events = self.events
        self.events = []
        return events

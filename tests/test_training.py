import pytest

from countdown_grpo.training import (
    RewardTelemetry,
    select_lora_target_suffixes,
    validate_grpo_batch,
)


def test_default_effective_generation_batch_is_legal_for_four_generations():
    assert validate_grpo_batch(
        per_device_train_batch_size=1, gradient_accumulation_steps=8, num_generations=4
    ) == 8


def test_partial_grpo_group_is_rejected():
    with pytest.raises(ValueError, match="not divisible"):
        validate_grpo_batch(per_device_train_batch_size=1, gradient_accumulation_steps=3, num_generations=4)


def test_live_target_selection_requires_linear_and_full_attention_families():
    module_names = [
        "layers.0.linear_attn.in_proj_qkv",
        "layers.0.linear_attn.in_proj_z",
        "layers.0.linear_attn.out_proj",
        "layers.3.self_attn.q_proj",
        "layers.3.self_attn.k_proj",
        "layers.3.self_attn.v_proj",
        "layers.3.self_attn.o_proj",
    ]
    assert select_lora_target_suffixes(module_names) == [
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def test_binary_reward_telemetry_reports_group_signal_and_failures():
    telemetry = RewardTelemetry(num_generations=2)
    rewards = telemetry(
        ["<answer>2 + 3</answer>", "no expression", "<answer>4 / 2</answer>", "<answer>4 + 2</answer>"],
        target=[5, 5, 2, 2],
        nums=[[2, 3], [2, 3], [4, 2], [4, 2]],
    )
    assert rewards == [1.0, 0.0, 1.0, 0.0]
    event = telemetry.pop_events()[0]
    assert event["mixed_reward_group_rate"] == 1.0
    assert event["all_zero_group_rate"] == 0.0
    assert event["positive_completion_rate"] == 0.5
    assert event["failure_categories"]["unsupported_syntax"] == 1
    assert event["failure_categories"]["wrong_target"] == 1

"""Minimal LoRA + GRPO training entry point."""

from __future__ import annotations

import argparse

from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .data import load_countdown
from .rewards import countdown_reward


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--output-dir", default="outputs/qwen35-08b-countdown-grpo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dataset, eval_dataset = load_countdown(seed=args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=countdown_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        ),
        args=GRPOConfig(
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            num_generations=4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_prompt_length=128,
            max_completion_length=256,
            logging_steps=1,
            save_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            report_to="none",
            seed=args.seed,
            remove_unused_columns=False,
        ),
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

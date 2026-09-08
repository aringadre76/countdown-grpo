"""LoRA + exact-binary-reward GRPO entry point for Qwen3.5-0.8B-Base."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import load_prepared_dataset
from .training import RewardTelemetry, select_lora_target_suffixes, validate_grpo_batch

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _load_model_and_tokenizer(model_id: str, revision: str | None, device: str):
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForImageTextToText, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("Install the project train extra before running GRPO") from error

    resolved_revision = revision or HfApi().model_info(model_id).sha
    selected_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch_device = torch.device(selected_device)
    dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=resolved_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=resolved_revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(torch_device)
    model.config.use_cache = False
    return model, tokenizer, torch, torch_device, resolved_revision


def make_diagnostics_callback(telemetry: RewardTelemetry, output_path: Path):
    """Build the optional Transformers callback that persists GRPO rollouts.

    Keeping this import local lets the package's CPU-only verifier tests run
    without Transformers installed.
    """

    from transformers import TrainerCallback

    class DiagnosticsCallback(TrainerCallback):
        def __init__(self) -> None:
            self.telemetry = telemetry
            self.output_path = output_path
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **_: Any,
        ):
            self._flush(state.global_step, logs or {})
            return control

        def on_train_end(self, args: Any, state: Any, control: Any, **_: Any):
            self._flush(state.global_step, {})
            return control

        def _flush(self, step: int, trainer_logs: dict[str, Any]) -> None:
            events = self.telemetry.pop_events()
            if not events:
                return
            with self.output_path.open("a", encoding="utf-8", newline="\n") as handle:
                for event in events:
                    event["optimizer_step"] = step
                    event["timestamp"] = datetime.now(UTC).isoformat()
                    event["trainer_logs"] = trainer_logs
                    handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")

    return DiagnosticsCallback()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--train-file", default="smoke_train.jsonl")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen35-08b-countdown-grpo"))
    parser.add_argument("--experiment-id", default="grpo-smoke")
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    evidence_dir = args.evidence_dir or Path("artifacts/experiments") / args.experiment_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = evidence_dir / "attempt.json"
    attempt = {
        "status": "started",
        "started_at": datetime.now(UTC).isoformat(),
        "command": [sys.executable, "-m", "countdown_grpo.train_grpo", *sys.argv[1:]],
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")

    generation_batch = validate_grpo_batch(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
    )
    model, tokenizer, _torch, device, revision = _load_model_and_tokenizer(
        args.model, args.revision, args.device
    )
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU fallback requires --allow-cpu so it cannot be mistaken for the GPU experiment")

    module_names = [name for name, _ in model.named_modules()]
    target_modules = select_lora_target_suffixes(module_names)
    try:
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("Install the project train extra before running GRPO") from error

    telemetry = RewardTelemetry(num_generations=args.num_generations)
    diagnostics_callback = make_diagnostics_callback(telemetry, evidence_dir / "grpo-diagnostics.jsonl")
    train_dataset = load_prepared_dataset(args.data_dir / args.train_file)
    maximum_prompt_tokens = max(
        len(tokenizer.encode(str(task["prompt"]), add_special_tokens=False)) for task in train_dataset
    )
    if maximum_prompt_tokens > args.max_prompt_length:
        raise ValueError(
            f"prepared prompt length {maximum_prompt_tokens} exceeds --max-prompt-length {args.max_prompt_length}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "model_id": args.model,
        "revision": revision,
        "device": str(device),
        "train_file": str(args.data_dir / args.train_file),
        "dataset_rows": len(train_dataset),
        "maximum_observed_prompt_tokens": maximum_prompt_tokens,
        "target_modules": target_modules,
        "effective_generation_batch_size": generation_batch,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "reward": "binary exact Countdown verifier only",
        "seed": args.seed,
    }
    (evidence_dir / "run-config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=telemetry,
        train_dataset=train_dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        ),
        callbacks=[diagnostics_callback],
        args=GRPOConfig(
            output_dir=str(args.output_dir),
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_completion_length=args.max_completion_length,
            logging_steps=1,
            logging_first_step=True,
            save_strategy="no",
            eval_strategy="no",
            report_to="none",
            seed=args.seed,
            remove_unused_columns=False,
            use_cpu=device.type == "cpu",
            bf16=device.type == "cuda",
            gradient_checkpointing=False,
            dataloader_pin_memory=False,
            generation_batch_size=generation_batch,
        ),
    )
    try:
        train_result = trainer.train()
        trainer.save_model(str(args.output_dir / "final-adapter"))
    except Exception as error:
        attempt.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
        raise
    attempt.update(
        {
            "status": "completed",
            "finished_at": datetime.now(UTC).isoformat(),
            "global_step": trainer.state.global_step,
            "training_loss": train_result.training_loss,
        }
    )
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

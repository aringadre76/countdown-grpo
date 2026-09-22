"""LoRA + exact-binary-reward GRPO entry point for Qwen3.5-0.8B-Base."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .data import file_sha256, load_prepared_dataset
from .evaluate import _render_prompt
from .training import (
    RewardTelemetry,
    select_lora_target_suffixes,
    validate_adapter_target_suffixes,
    validate_grpo_batch,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _load_model_and_tokenizer(
    model_id: str,
    revision: str | None,
    device: str,
    attn_implementation: str | None = None,
):
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
    model_kwargs: dict[str, Any] = {
        "revision": resolved_revision,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_kwargs,
    ).to(torch_device)
    model.config.use_cache = False
    return model, tokenizer, torch, torch_device, resolved_revision


def _install_attempt_failure_recorder(attempt: dict[str, Any], attempt_path: Path) -> None:
    """Persist setup failures too, not only exceptions inside trainer.train()."""
    previous_hook = sys.excepthook

    def record_failure(error_type: type[BaseException], error: BaseException, tb: Any) -> None:
        attempt.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error_type": error_type.__name__,
                "error": str(error),
                "traceback": "".join(traceback.format_exception(error_type, error, tb)),
            }
        )
        attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
        previous_hook(error_type, error, tb)

    sys.excepthook = record_failure


def make_diagnostics_callback(
    telemetry: RewardTelemetry,
    output_path: Path,
    *,
    model: Any = None,
    torch_module: Any = None,
    gradient_output_path: Path | None = None,
):
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

        def on_pre_optimizer_step(
            self, args: Any, state: Any, control: Any, **_: Any
        ):
            if model is None or torch_module is None or gradient_output_path is None:
                return control
            gradients = [
                parameter.grad.detach()
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            finite = bool(
                torch_module.stack(
                    [torch_module.isfinite(gradient).all() for gradient in gradients]
                ).all().item()
            ) if gradients else False
            nonzero_parameters = int(
                torch_module.stack(
                    [torch_module.count_nonzero(gradient) > 0 for gradient in gradients]
                ).sum().item()
            ) if gradients else 0
            event = {
                "optimizer_step": state.global_step + 1,
                "timestamp": datetime.now(UTC).isoformat(),
                "trainable_gradient_tensors": len(gradients),
                "parameters_with_nonzero_gradient": nonzero_parameters,
                "gradients_finite": finite,
            }
            gradient_output_path.parent.mkdir(parents=True, exist_ok=True)
            with gradient_output_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            if not finite:
                raise FloatingPointError("GRPO produced missing or non-finite trainable gradients")
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
    parser.add_argument("--adapter-path", type=Path, default=None,
                        help="Optional trainable PEFT adapter to use as the policy initialization")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--train-file", default="smoke_train.jsonl")
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen35-08b-countdown-grpo"))
    parser.add_argument("--experiment-id", default="grpo-smoke")
    parser.add_argument("--evidence-dir", type=Path, default=None)
    parser.add_argument("--frozen-design", type=Path, default=None,
                        help="Optional frozen design file whose hash is recorded in run-config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa"),
        default=None,
        help="Optional Transformers attention backend; record compatibility overrides explicitly.",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=128)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()
    if args.disable_thinking and not args.use_chat_template:
        parser.error("--disable-thinking requires --use-chat-template")
    evidence_dir = args.evidence_dir or Path("artifacts/experiments") / args.experiment_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = evidence_dir / "attempt.json"
    recorded_python = os.path.relpath(sys.executable, Path.cwd())
    attempt = {
        "status": "started",
        "started_at": datetime.now(UTC).isoformat(),
        "command": [recorded_python, "-m", "countdown_grpo.train_grpo", *sys.argv[1:]],
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    _install_attempt_failure_recorder(attempt, attempt_path)

    generation_batch = validate_grpo_batch(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
    )
    model, tokenizer, _torch, device, revision = _load_model_and_tokenizer(
        args.model, args.revision, args.device, args.attn_implementation
    )
    if device.type == "cpu" and not args.allow_cpu:
        raise RuntimeError("CPU fallback requires --allow-cpu so it cannot be mistaken for the GPU experiment")
    if device.type == "cuda":
        _torch.cuda.reset_peak_memory_stats(device)

    try:
        from peft import LoraConfig, PeftConfig, PeftModel
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("Install the project train extra before running GRPO") from error

    module_names = [name for name, _ in model.named_modules()]
    starting_adapter = None
    if args.adapter_path is not None:
        if not args.adapter_path.is_dir():
            raise FileNotFoundError(f"starting adapter directory does not exist: {args.adapter_path}")
        starting_config = PeftConfig.from_pretrained(args.adapter_path)
        adapter_base = starting_config.base_model_name_or_path
        if adapter_base and adapter_base != args.model:
            raise ValueError(
                f"starting adapter is for {adapter_base!r}, but --model is {args.model!r}"
            )
        peft_type = getattr(starting_config.peft_type, "value", starting_config.peft_type)
        if peft_type != "LORA":
            raise ValueError(f"only a LoRA starting adapter is supported, got {peft_type!r}")
        task_type = getattr(starting_config.task_type, "value", starting_config.task_type)
        if task_type != "CAUSAL_LM":
            raise ValueError(f"starting adapter must have task_type='CAUSAL_LM', got {task_type!r}")
        adapter_targets = starting_config.target_modules
        if isinstance(adapter_targets, str):
            raise ValueError("regex/string adapter target modules are not supported for warm-start validation")
        target_modules = validate_adapter_target_suffixes(adapter_targets, module_names)
        model = PeftModel.from_pretrained(
            model,
            str(args.adapter_path),
            is_trainable=True,
            autocast_adapter_dtype=False,
        )
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("the loaded starting adapter has no trainable parameters")
        adapter_weights_path = args.adapter_path / "adapter_model.safetensors"
        if not adapter_weights_path.is_file():
            raise FileNotFoundError(f"starting adapter weights do not exist: {adapter_weights_path}")
        starting_adapter = {
            "path": str(args.adapter_path),
            "sha256": file_sha256(adapter_weights_path),
            "base_model_name_or_path": adapter_base,
            "peft_type": str(peft_type),
            "target_modules": target_modules,
            "rank": starting_config.r,
            "alpha": starting_config.lora_alpha,
            "dropout": starting_config.lora_dropout,
            "trainable": True,
            "autocast_adapter_dtype": False,
        }
        trainer_peft_config = None
    else:
        target_modules = select_lora_target_suffixes(module_names)
        trainer_peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

    telemetry = RewardTelemetry(
        num_generations=args.num_generations,
        termination_token_ids=tuple({tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}),
    )
    diagnostics_callback = make_diagnostics_callback(
        telemetry,
        evidence_dir / "grpo-diagnostics.jsonl",
        model=model,
        torch_module=_torch,
        gradient_output_path=evidence_dir / "gradient-diagnostics.jsonl",
    )
    train_dataset = load_prepared_dataset(args.data_dir / args.train_file)
    if args.use_chat_template:
        train_dataset = train_dataset.map(lambda task: {
            "prompt": _render_prompt(tokenizer, str(task["prompt"]), True, args.disable_thinking)
        })
    maximum_prompt_tokens = max(
        len(tokenizer.encode(str(task["prompt"]), add_special_tokens=False)) for task in train_dataset
    )
    if maximum_prompt_tokens > args.max_prompt_length:
        raise ValueError(
            f"prepared prompt length {maximum_prompt_tokens} exceeds --max-prompt-length {args.max_prompt_length}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        package_versions = {
            name: version(name)
            for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets")
        }
    except PackageNotFoundError as error:  # pragma: no cover - runtime environment diagnostic.
        raise RuntimeError(f"required training package is missing: {error}") from error
    run_config = {
        "experiment_id": args.experiment_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "model_id": args.model,
        "revision": revision,
        "starting_adapter": starting_adapter,
        "frozen_design_sha256": (
            file_sha256(args.frozen_design) if args.frozen_design is not None else None
        ),
        "device": str(device),
        "device_name": _torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device_architecture": (
            getattr(_torch.cuda.get_device_properties(device), "gcnArchName", None)
            if device.type == "cuda" else None
        ),
        "free_device_memory_bytes_before_training": (
            _torch.cuda.mem_get_info(device)[0] if device.type == "cuda" else None
        ),
        "total_device_memory_bytes": (
            _torch.cuda.mem_get_info(device)[1] if device.type == "cuda" else None
        ),
        "torch_hip_version": _torch.version.hip,
        "package_versions": package_versions,
        "requested_attention_implementation": args.attn_implementation,
        "train_file": str(args.data_dir / args.train_file),
        "train_file_sha256": file_sha256(args.data_dir / args.train_file),
        "dataset_rows": len(train_dataset),
        "maximum_observed_prompt_tokens": maximum_prompt_tokens,
        "target_modules": target_modules,
        "effective_generation_batch_size": generation_batch,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "reward": "binary exact Countdown verifier only",
        "prompt_format": "chat_template" if args.use_chat_template else "raw",
        "thinking_disabled": args.disable_thinking,
        "seed": args.seed,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_config": {
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_generation_batch_size": generation_batch,
            "num_generations": args.num_generations,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": 1.0,
            "top_k": 0,
            "beta": 0.0,
            "loss_type": "dapo",
            "bf16": device.type == "cuda",
            "gradient_checkpointing": False,
            "save_strategy": "no",
        },
    }
    (evidence_dir / "run-config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=telemetry,
        train_dataset=train_dataset,
        peft_config=trainer_peft_config,
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
            temperature=args.temperature,
            top_p=1.0,
            top_k=0,
            beta=0.0,
            loss_type="dapo",
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
            "trainer_metrics": train_result.metrics,
        }
    )
    if device.type == "cuda":
        _torch.cuda.synchronize(device)
        attempt["cuda_memory_bytes"] = {
            "allocated_after_training": _torch.cuda.memory_allocated(device),
            "reserved_after_training": _torch.cuda.memory_reserved(device),
            "peak_allocated": _torch.cuda.max_memory_allocated(device),
            "peak_reserved": _torch.cuda.max_memory_reserved(device),
        }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

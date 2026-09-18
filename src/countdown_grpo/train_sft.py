"""Explicit supervised LoRA branch; all targets are verified training witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from .data import read_jsonl
from .train_grpo import _git_commit, _load_model_and_tokenizer
from .training import select_lora_target_suffixes
from .verifier import verify_completion


def _trainable_digest(model) -> str:
    """Hash actual trainable tensors to verify optimizer updates."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "branch": "supervised LoRA", "git_commit": _git_commit(),
        "model_id": args.model, "revision": args.revision, "seed": args.seed,
        "command": [os.path.relpath(sys.executable, Path.cwd()), "-m", "countdown_grpo.train_sft", *sys.argv[1:]],
        "started_at": datetime.now(UTC).isoformat(), "status": "started",
        "max_steps": args.max_steps, "learning_rate": 2e-4,
        "per_device_batch_size": 1, "gradient_accumulation": 8,
        "max_length": 256, "completion_only_loss": True,
        "train_sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
    }
    attempt_path = args.evidence_dir / "attempt.json"
    attempt_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import TrainerCallback, set_seed
        from trl import SFTConfig, SFTTrainer

        set_seed(args.seed)
        model, tokenizer, torch, device, _ = _load_model_and_tokenizer(
            args.model, args.revision, "cuda", "eager"
        )
        torch.cuda.reset_peak_memory_stats(device)
        tasks = read_jsonl(args.train_file)
        if not tasks or any(task["split"] != "train" for task in tasks):
            raise ValueError("SFT requires nonempty training-only examples")
        for task in tasks:
            if not verify_completion(task["completion"], task["target"], task["nums"]).valid:
                raise ValueError(f"invalid supervised target: {task['task_id']}")
            ids = tokenizer.encode(task["prompt"]) + tokenizer.encode(
                task["completion"] + tokenizer.eos_token, add_special_tokens=False
            )
            if len(ids) > 256:
                raise ValueError("supervised sequence would be truncated")
        targets = select_lora_target_suffixes([name for name, _ in model.named_modules()])
        targets += [suffix for suffix in ("gate_proj", "up_proj", "down_proj")
                    if any(name.endswith(suffix) for name, _ in model.named_modules())]
        record.update({"target_modules": targets, "dataset_rows": len(tasks)})

        class EvidenceCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                row = {"step": state.global_step, "timestamp": datetime.now(UTC).isoformat(), **(logs or {})}
                if any(not math.isfinite(value) for value in row.values() if isinstance(value, float)):
                    raise FloatingPointError("non-finite supervised training metric")
                with (args_evidence / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")

        args_evidence = args.evidence_dir
        examples = []
        for task in tasks:
            prompt_ids = tokenizer.encode(task["prompt"])
            answer_ids = tokenizer.encode(task["completion"] + tokenizer.eos_token, add_special_tokens=False)
            examples.append({
                "input_ids": prompt_ids + answer_ids,
                "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
                "labels": [-100] * len(prompt_ids) + answer_ids,
            })
        dataset = Dataset.from_list(examples)
        record["tokenization"] = "separate prompt and completion IDs; explicit completion-only labels"
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=dataset,
            peft_config=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                  target_modules=targets, task_type="CAUSAL_LM"),
            callbacks=[EvidenceCallback()],
            args=SFTConfig(
                output_dir=str(args.output_dir), max_steps=args.max_steps,
                per_device_train_batch_size=1, gradient_accumulation_steps=8,
                learning_rate=2e-4, max_length=256, completion_only_loss=True,
                dataset_kwargs={"skip_prepare_dataset": True},
                loss_type="nll", bf16=True, gradient_checkpointing=False,
                dataloader_pin_memory=False, logging_steps=10, logging_first_step=True,
                save_strategy="no", eval_strategy="no", report_to="none", seed=args.seed,
            ),
        )
        record["trainable_parameters"] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        probe = trainer.data_collator([trainer.train_dataset[0]])
        labels = probe["labels"][0].tolist()
        expected_labels = examples[0]["labels"]
        if labels[:len(expected_labels)] != expected_labels or any(
            token != -100 for token in labels[len(expected_labels):]
        ):
            raise ValueError("collator changed the explicit completion-only label mask")
        supervised_ids = [token for token in labels if token != -100]
        if not supervised_ids or labels[0] != -100:
            raise ValueError("completion-only label mask is missing")
        supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=True)
        if "<answer>" not in supervised_text or "Numbers:" in supervised_text:
            raise ValueError("supervised label mask includes prompt or omits the answer")
        record["label_mask_probe"] = {
            "sequence_tokens": len(labels), "supervised_tokens": len(supervised_ids),
            "masked_tokens": labels.count(-100), "prompt_masked": True,
        }
        attempt_path.write_text(json.dumps(record, indent=2) + "\n")
        record["initial_trainable_sha256"] = _trainable_digest(trainer.model)
        result = trainer.train()
        record["final_trainable_sha256"] = _trainable_digest(trainer.model)
        record["adapter_weights_changed"] = record["initial_trainable_sha256"] != record["final_trainable_sha256"]
        if not record["adapter_weights_changed"]:
            raise RuntimeError("optimizer did not change trainable weights")
        trainer.save_model(str(args.output_dir / "final-adapter"))
        record.update({"status": "completed", "global_step": trainer.state.global_step,
                       "trainer_metrics": result.metrics,
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                       "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
    except Exception as error:
        record.update({"status": "failed", "error": str(error), "error_type": type(error).__name__})
        raise
    finally:
        record["finished_at"] = datetime.now(UTC).isoformat()
        attempt_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Generate and score saved Countdown evaluation tasks with a frozen checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import read_jsonl
from .verifier import extract_expression, reaches_target_without_contract, verify_completion

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _wilson_interval(successes: int, trials: int) -> list[float] | None:
    if trials == 0:
        return None
    z = 1.959963984540054  # 95% confidence.
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    radius = z * ((proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) ** 0.5)
    return [centre - radius / denominator, centre + radius / denominator]


def summarise_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise saved records without changing any reward or verifier result."""

    total = len(records)
    exact = sum(record["reward"] == 1.0 for record in records)
    legal = sum(record["failure_category"] in {"ok", "wrong_target"} for record in records)
    target_hit_but_illegal = sum(
        record["reward"] == 0.0
        and isinstance(record.get("extracted_expression"), str)
        and "target" in record
        and reaches_target_without_contract(str(record["extracted_expression"]), int(record["target"]))
        for record in records
    )
    truncated = sum(bool(record["truncation"]) for record in records)
    failures = Counter(record["failure_category"] for record in records)
    by_numbers: dict[str, dict[str, Any]] = {}
    by_split: dict[str, dict[str, Any]] = {}
    for destination, field in ((by_numbers, "num_count"), (by_split, "split")):
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record[field])].append(record)
        for group_name, group_records in grouped.items():
            group_exact = sum(item["reward"] == 1.0 for item in group_records)
            destination[group_name] = {
                "records": len(group_records),
                "exact_solve_rate": group_exact / len(group_records),
                "exact_solve_wilson_95": _wilson_interval(group_exact, len(group_records)),
            }
    task_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_groups[(str(record["task_id"]), str(record["generation_mode"]))].append(record)
    modes = sorted({str(record["generation_mode"]) for record in records})
    pass_at_k: dict[str, float | None] = {}
    for mode in modes:
        groups = [items for (_, group_mode), items in task_groups.items() if group_mode == mode]
        pass_at_k[mode] = sum(any(item["reward"] == 1.0 for item in items) for items in groups) / len(groups)
    return {
        "record_count": total,
        "exact_solve_rate": exact / total if total else None,
        "exact_solve_wilson_95": _wilson_interval(exact, total),
        "legal_expression_rate": legal / total if total else None,
        "target_hit_but_illegal_rate": target_hit_but_illegal / total if total else None,
        "invalid_expression_rate": (total - legal) / total if total else None,
        "no_answer_rate": failures["no_answer"] / total if total else None,
        "truncation_rate": truncated / total if total else None,
        "completion_length": {
            "min": min((record["completion_length"] for record in records), default=None),
            "max": max((record["completion_length"] for record in records), default=None),
            "mean": sum(record["completion_length"] for record in records) / total if total else None,
        },
        "failure_categories": dict(sorted(failures.items())),
        "by_number_count": by_numbers,
        "by_split": by_split,
        "pass_at_k_by_generation_mode": pass_at_k,
        "target_hit_but_illegal_definition": (
            "Selected expression reaches the target under the exact parser while failing the locked "
            "Countdown number-use or integer-intermediate rules; diagnostic only, never reward."
        ),
    }


def _load_model(model_id: str, revision: str | None, device: str, adapter_path: str | None):
    try:
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoModelForImageTextToText, AutoTokenizer
    except ImportError as error:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("Install the project train extra before evaluating a model") from error

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
    )
    if adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as error:  # pragma: no cover - optional runtime dependency.
            raise RuntimeError("Install the project train extra before loading an adapter") from error
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(torch_device)
    model.eval()
    return model, tokenizer, torch, torch_device, resolved_revision


def _generate(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: Any,
    prompt: str,
    generation_mode: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[str, int, bool]:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    options: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": generation_mode == "sample",
    }
    if generation_mode == "sample":
        options.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    generated = output[0, inputs["input_ids"].shape[1] :]
    completion_length = int(generated.shape[0])
    return tokenizer.decode(generated, skip_special_tokens=False), completion_length, completion_length >= max_new_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter-path", default=None, help="Optional local LoRA adapter to compare with base")
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--splits", nargs="+", default=["source_test", "fresh_test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--generation-modes", nargs="+", choices=["greedy", "sample"], default=["greedy", "sample"])
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.samples_per_task < 1:
        raise ValueError("samples_per_task must be positive")

    model, tokenizer, torch, device, resolved_revision = _load_model(
        args.model, args.revision, args.device, args.adapter_path
    )
    from transformers import set_seed

    records: list[dict[str, Any]] = []
    git_commit = _git_commit()
    task_index = 0
    for requested_split in args.splits:
        tasks = read_jsonl(args.data_dir / f"{requested_split}.jsonl")
        if args.limit is not None:
            tasks = tasks[: args.limit]
        for task in tasks:
            for generation_mode in args.generation_modes:
                generation_count = 1 if generation_mode == "greedy" else args.samples_per_task
                for sample_index in range(generation_count):
                    generation_seed = args.seed + task_index
                    set_seed(generation_seed)
                    completion, completion_length, truncated = _generate(
                        model=model,
                        tokenizer=tokenizer,
                        torch=torch,
                        device=device,
                        prompt=str(task["prompt"]),
                        generation_mode=generation_mode,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                    verdict = verify_completion(
                        completion,
                        int(task["target"]),
                        [int(number) for number in task["nums"]],
                        truncated=truncated,
                    )
                    records.append(
                        {
                            "task_id": task["task_id"],
                            "split": task["split"],
                            "nums": task["nums"],
                            "num_count": len(task["nums"]),
                            "target": task["target"],
                            "oracle_solvable": task["oracle_solvable"],
                            "prompt": task["prompt"],
                            "raw_completion": completion,
                            "extracted_expression": extract_expression(completion),
                            "verifier_result": {
                                "valid": verdict.valid,
                                "value": str(verdict.value) if verdict.value is not None else None,
                                "reason": verdict.reason,
                                "detail": verdict.detail,
                            },
                            "reward": 1.0 if verdict.valid else 0.0,
                            "failure_category": verdict.reason,
                            "prompt_length": len(tokenizer.encode(str(task["prompt"]), add_special_tokens=False)),
                            "completion_length": completion_length,
                            "truncation": truncated,
                            "model_id": args.model,
                            "adapter_path": args.adapter_path,
                            "revision": resolved_revision,
                            "generation_mode": generation_mode,
                            "generation_config": {
                                "max_new_tokens": args.max_new_tokens,
                                "temperature": args.temperature if generation_mode == "sample" else None,
                                "top_p": args.top_p if generation_mode == "sample" else None,
                                "do_sample": generation_mode == "sample",
                            },
                            "sample_index": sample_index,
                            "seed": generation_seed,
                            "timestamp": datetime.now(UTC).isoformat(),
                            "experiment_id": args.experiment_id,
                            "git_commit": git_commit,
                            "device": str(device),
                        }
                    )
                    task_index += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    summary = summarise_records(records)
    summary.update(
        {
            "experiment_id": args.experiment_id,
            "model_id": args.model,
            "revision": resolved_revision,
            "adapter_path": args.adapter_path,
            "output_jsonl": str(args.output),
            "git_commit": git_commit,
        }
    )
    summary_path = args.summary_output or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

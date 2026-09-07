"""Countdown task provenance, prompt construction, and leakage-safe splits.

The core helpers in this module have no Hugging Face dependency. Dataset I/O
is imported lazily so CPU verifier/oracle tests remain lightweight.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .oracle import reachable_values, solve_countdown

DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
SPLIT_NAMES = ("train", "dev", "test")


def canonical_key(target: int, nums: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    """Canonical logical-task key used for deduplication and leakage checks."""

    return int(target), tuple(sorted(int(number) for number in nums))


def canonical_key_text(target: int, nums: Sequence[int]) -> str:
    """Stable JSON representation of the canonical logical task key."""

    task_target, sorted_nums = canonical_key(target, nums)
    return json.dumps([task_target, list(sorted_nums)], separators=(",", ":"))


def task_id(target: int, nums: Sequence[int], *, namespace: str) -> str:
    """Create a stable public id that does not encode an oracle witness."""

    digest = hashlib.sha256(canonical_key_text(target, nums).encode("utf-8")).hexdigest()[:16]
    return f"{namespace}-{digest}"


def format_prompt(example: Mapping[str, Any]) -> str:
    """Return the frozen core prompt without any solver-derived information."""

    nums = ", ".join(str(int(number)) for number in example["nums"])
    return (
        "Use each supplied number exactly once with the binary operators +, -, *, or /. "
        "Parentheses are allowed. Every intermediate result and the final result must be an integer. "
        "Return only one expression inside <answer>...</answer>.\n"
        f"Numbers: {nums}\n"
        f"Target: {int(example['target'])}\n"
        "Answer:"
    )


def _normalise_task(row: Mapping[str, Any], *, namespace: str, source_index: int | None) -> dict[str, Any]:
    target = int(row["target"])
    nums = [int(number) for number in row["nums"]]
    if len(nums) not in (3, 4):
        raise ValueError(f"Countdown task must have three or four numbers, received {len(nums)}")
    oracle = solve_countdown(nums, target, include_witness=False)
    task: dict[str, Any] = {
        "task_id": task_id(target, nums, namespace=namespace),
        "target": target,
        "nums": nums,
        "canonical_key": canonical_key_text(target, nums),
        "oracle_solvable": oracle.solvable,
        "oracle_reachable_count": oracle.reachable_count,
        "prompt": format_prompt({"target": target, "nums": nums}),
    }
    if source_index is not None:
        task["source_index"] = source_index
    return task


def canonicalise_source_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate source rows by canonical key and audit the original rows.

    Oracle witnesses are intentionally discarded before the task records leave
    this function. The resulting records can safely be used for prompts.
    """

    raw_count = 0
    length_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    number_counts: Counter[str] = Counter()
    ordered_row_counts: Counter[str] = Counter()
    by_key: dict[str, dict[str, Any]] = {}

    for source_index, row in enumerate(rows):
        target = int(row["target"])
        nums = [int(number) for number in row["nums"]]
        if len(nums) not in (3, 4):
            continue
        raw_count += 1
        length_counts[str(len(nums))] += 1
        target_counts[str(target)] += 1
        number_counts.update(str(number) for number in nums)
        ordered_row_counts[json.dumps([target, nums], separators=(",", ":"))] += 1

        key = canonical_key_text(target, nums)
        if key not in by_key:
            by_key[key] = _normalise_task(row, namespace="source", source_index=source_index)

    tasks = [by_key[key] for key in sorted(by_key)]
    solvable_count = sum(bool(task["oracle_solvable"]) for task in tasks)
    audit = {
        "dataset_id": DATASET_ID,
        "raw_row_count": raw_count,
        "three_number_count": length_counts["3"],
        "four_number_count": length_counts["4"],
        "target_distribution": dict(sorted(target_counts.items(), key=lambda item: int(item[0]))),
        "number_distribution": dict(sorted(number_counts.items(), key=lambda item: int(item[0]))),
        "exact_duplicate_row_count": sum(count - 1 for count in ordered_row_counts.values() if count > 1),
        "canonical_task_count": len(tasks),
        "canonical_duplicate_row_count": raw_count - len(tasks),
        "oracle_solvable_task_count": solvable_count,
        "oracle_solvability_rate": solvable_count / len(tasks) if tasks else None,
        "oracle_contract": "integer-only + - * / with every supplied number used exactly once",
    }
    return tasks, audit


def split_canonical_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    dev_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic train/dev/test splits with canonical-key isolation."""

    if not 0 < dev_fraction < 1 or not 0 < test_fraction < 1 or dev_fraction + test_fraction >= 1:
        raise ValueError("dev_fraction and test_fraction must be positive and sum to less than one")
    unique: dict[str, dict[str, Any]] = {}
    for task in tasks:
        key = str(task["canonical_key"])
        if key in unique:
            raise ValueError(f"input contains duplicate canonical task: {key}")
        unique[key] = dict(task)

    shuffled = [unique[key] for key in sorted(unique)]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    dev_count = round(total * dev_fraction)
    test_count = round(total * test_fraction)
    train_count = total - dev_count - test_count
    splits = {
        "train": shuffled[:train_count],
        "dev": shuffled[train_count : train_count + dev_count],
        "test": shuffled[train_count + dev_count :],
    }
    for split_name, records in splits.items():
        for task in records:
            task["split"] = split_name
    assert_no_canonical_leakage(splits)
    return {name: sorted(records, key=lambda task: str(task["task_id"])) for name, records in splits.items()}


def assert_no_canonical_leakage(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Raise if a canonical logical task appears in more than one split."""

    seen: dict[str, str] = {}
    for split_name, records in splits.items():
        for task in records:
            key = str(task["canonical_key"])
            previous = seen.setdefault(key, split_name)
            if previous != split_name:
                raise ValueError(f"canonical leakage: {key} appears in both {previous} and {split_name}")


def make_fresh_tasks(
    *,
    count: int,
    seed: int,
    forbidden_keys: Iterable[str],
    min_numbers: int = 3,
    max_numbers: int = 4,
    number_min: int = 1,
    number_max: int = 25,
    target_min: int = 10,
    target_max: int = 999,
) -> list[dict[str, Any]]:
    """Build an independent, oracle-confirmed fresh evaluation suite.

    The generator selects only targets that the independent oracle can reach,
    then drops all witnesses. It neither reads source rows nor exposes a
    constructed expression in prompts or artifacts.
    """

    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    forbidden = set(forbidden_keys)
    tasks: list[dict[str, Any]] = []
    attempts = 0
    while len(tasks) < count:
        attempts += 1
        if attempts > count * 1_000:
            raise RuntimeError("fresh task generation exhausted its duplicate-avoidance budget")
        nums = [rng.randint(number_min, number_max) for _ in range(rng.randint(min_numbers, max_numbers))]
        candidates = [
            value
            for value in reachable_values(nums)
            if target_min <= value <= target_max
        ]
        if not candidates:
            continue
        target = rng.choice(candidates)
        key = canonical_key_text(target, nums)
        if key in forbidden:
            continue
        task = _normalise_task({"target": target, "nums": nums}, namespace="fresh", source_index=None)
        if not task["oracle_solvable"]:
            raise AssertionError("fresh task failed its independent oracle check")
        task["split"] = "fresh_test"
        task["fresh_generator_seed"] = seed
        tasks.append(task)
        forbidden.add(key)
    return sorted(tasks, key=lambda task: str(task["task_id"]))


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write deterministic UTF-8 JSONL with no hidden solver data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            if "witness" in record:
                raise ValueError("oracle witnesses must never be written to task artifacts")
            handle.write(json.dumps(dict(record), sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a prepared JSONL task artifact."""

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split_manifest(
    splits: Mapping[str, Sequence[Mapping[str, Any]]], *, seed: int
) -> dict[str, Any]:
    """Return persisted split metadata and canonical-key hashes."""

    assert_no_canonical_leakage(splits)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "canonical_key_definition": "(target, sorted multiset of nums)",
        "seed": seed,
        "splits": {
            name: {
                "task_count": len(records),
                "canonical_key_sha256": hashlib.sha256(
                    "\n".join(sorted(str(record["canonical_key"]) for record in records)).encode("utf-8")
                ).hexdigest(),
            }
            for name, records in splits.items()
        },
    }


def load_prepared_dataset(path: Path):
    """Load prepared records as a Hugging Face Dataset only when training is installed."""

    try:
        from datasets import Dataset
    except ImportError as error:  # pragma: no cover - depends on optional extra.
        raise RuntimeError("Install the project train extra to load Hugging Face datasets") from error
    return Dataset.from_list(read_jsonl(path))


def load_countdown(data_dir: str | Path = "artifacts/data"):
    """Load the persisted source train/dev splits for GRPO without reshuffling."""

    root = Path(data_dir)
    return load_prepared_dataset(root / "source_train.jsonl"), load_prepared_dataset(root / "source_dev.jsonl")

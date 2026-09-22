"""Prepare unseen confirmation tasks only after a method freeze is recorded."""

import argparse
import json
import random
from pathlib import Path

from .data import canonical_key_text, file_sha256, make_fresh_tasks, write_jsonl
from .oracle import solve_countdown


def records(path):
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def observed_keys(value):
    """Include task identities nested inside historical rollout diagnostics."""
    if isinstance(value, dict):
        if "nums" in value and "target" in value:
            yield canonical_key_text(value["target"], value["nums"])
        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from observed_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from observed_keys(child)


def select_source(tasks, forbidden, count, seed):
    candidates = sorted((dict(task) for task in tasks
                         if canonical_key_text(task["target"], task["nums"]) not in forbidden),
                        key=lambda task: task["task_id"])
    random.Random(seed).shuffle(candidates)
    selected = candidates[:count]
    if len(selected) != count:
        raise ValueError("not enough unseen source tasks")
    keys = [canonical_key_text(task["target"], task["nums"]) for task in selected]
    if len(set(keys)) != count:
        raise ValueError("duplicate canonical source confirmation tasks")
    for task in selected:
        if task["split"] != "test":
            raise ValueError("source confirmation requires held-out test tasks")
        task["oracle_solvable"] = solve_countdown(task["nums"], task["target"]).solvable
        task["split"] = "source_confirmation"
    return sorted(selected, key=lambda task: task["task_id"])


def frozen_confirmation_size(freeze):
    """Return the suite size from either supported frozen-design schema."""
    if freeze.get("status") != "frozen":
        raise ValueError("a frozen confirmation design is required")
    if "confirmation_tasks_per_suite" in freeze:
        return freeze["confirmation_tasks_per_suite"]
    confirmation = freeze.get("confirmation", freeze)
    source_count = confirmation.get("source_tasks")
    fresh_count = confirmation.get("fresh_tasks")
    if source_count != fresh_count or source_count is None:
        raise ValueError("frozen source and fresh confirmation sizes must match")
    return source_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--history-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    if args.size != frozen_confirmation_size(freeze):
        raise ValueError("a matching frozen confirmation design is required")
    # Permit callers to create the destination directory up front, but never
    # overwrite an existing artifact or manifest.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("confirmation output must be new")
    if args.manifest.exists():
        raise ValueError("confirmation output must be new")
    forbidden = set()
    all_source = set()
    provenance = {}
    for name in ("source_train.jsonl", "source_dev.jsonl", "source_test.jsonl", "fresh_test.jsonl"):
        path = args.data_dir / name
        keys = {canonical_key_text(row["target"], row["nums"]) for row in records(path)}
        if name.startswith("source_"):
            all_source.update(keys)
        if name != "source_test.jsonl":
            forbidden.update(keys)
        provenance[str(path)] = file_sha256(path)
    for path in sorted(args.history_dir.rglob("*.jsonl")):
        # Ignored data includes the already-excluded supervised train set.
        if "data" in path.relative_to(args.history_dir).parts:
            continue
        for row in records(path):
            forbidden.update(observed_keys(row))
        provenance[str(path)] = file_sha256(path)
    source = select_source(records(args.data_dir / "source_test.jsonl"), forbidden, args.size, args.seed)
    fresh = make_fresh_tasks(count=args.size, seed=args.seed + 1,
                             forbidden_keys=forbidden | all_source)
    for task in fresh:
        task["split"] = "fresh_confirmation"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, tasks in [("source_confirmation", source), ("fresh_confirmation", fresh)]:
        write_jsonl(args.output_dir / f"{name}.jsonl", tasks)
    manifest = {
        "freeze_sha256": file_sha256(args.freeze), "seed": args.seed,
        "fresh_seed": args.seed + 1, "tasks_per_suite": args.size,
        "fresh_generator": {"number_count_range": [3, 4], "number_range": [1, 25],
                            "target_range": [10, 999], "oracle_witnesses_written": False},
        "excluded_key_count": len(forbidden), "inputs_sha256": provenance,
        "source_task_ids": [task["task_id"] for task in source],
        "fresh_task_ids": [task["task_id"] for task in fresh],
        "source_solvable_count": sum(task["oracle_solvable"] for task in source),
        "files_sha256": {path.name: file_sha256(path) for path in args.output_dir.glob("*.jsonl")},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

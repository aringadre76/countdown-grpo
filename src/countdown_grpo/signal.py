"""Check predeclared binary-GRPO exploration gates from saved dev completions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data import read_jsonl


def exploration_signal(records: list[dict]) -> dict:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["generation_mode"] == "sample":
            groups[record["task_id"]].append(float(record["reward"]))
    positive = sum(any(reward > 0 for reward in group) for group in groups.values())
    mixed = sum(min(group) != max(group) for group in groups.values())
    rate = mixed / len(groups) if groups else 0.0
    return {
        "records": len(records),
        "sampled_tasks": len(groups),
        "sampled_positive_tasks": positive,
        "sampled_mixed_tasks": mixed,
        "sampled_mixed_rate": rate,
        "sample_counts": sorted({len(group) for group in groups.values()}),
        "truncation_rate": sum(bool(record["truncation"]) for record in records) / len(records) if records else None,
        "diagnostic_gate_passed": positive >= 2 and rate >= 0.1,
        "gate": "at least 2 sampled positive tasks and mixed sampled group rate >= 0.10",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = exploration_signal(read_jsonl(args.input))
    result["input"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

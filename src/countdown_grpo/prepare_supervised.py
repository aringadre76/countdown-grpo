"""Build verified train-only demonstrations for an explicitly supervised branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .data import canonical_key, read_jsonl
from .oracle import solve_countdown
from .verifier import verify_completion


def supervised_pairs(tasks: list[dict], size: int, seed: int) -> list[dict]:
    if size < 1:
        raise ValueError("size must be positive")
    if any(task["split"] != "train" for task in tasks):
        raise ValueError("supervised witnesses require training-only inputs")
    shuffled = list(tasks)
    random.Random(seed).shuffle(shuffled)
    pairs = []
    seen = set()
    for task in shuffled:
        key = canonical_key(task["target"], task["nums"])
        if key in seen:
            continue
        seen.add(key)
        oracle = solve_countdown(task["nums"], task["target"], include_witness=True)
        if not oracle.solvable:
            continue
        completion = f"<answer>{oracle.witness}</answer>"
        if not verify_completion(completion, task["target"], task["nums"]).valid:
            raise ValueError(f"invalid training witness: {task['task_id']}")
        pairs.append({
            "task_id": task["task_id"], "split": "train",
            "nums": task["nums"], "target": task["target"],
            "prompt": task["prompt"], "completion": completion,
        })
        if len(pairs) == size:
            return pairs
    raise ValueError(f"only {len(pairs)} distinct solvable training tasks available")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    pairs = supervised_pairs(read_jsonl(args.input), args.size, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(pair, sort_keys=True) + "\n" for pair in pairs)
    args.output.write_text(payload)
    manifest = {
        "branch": "supervised initialization; not the historical no-SFT experiment",
        "input": str(args.input), "output": str(args.output),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "seed": args.seed, "rows": len(pairs),
        "verified_witnesses": len(pairs), "source_split": "train",
        "task_ids": [pair["task_id"] for pair in pairs],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(pairs)} training-only demonstrations")


if __name__ == "__main__":
    main()

"""Download, audit, deduplicate, split, and persist Countdown task artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .data import (
    DATASET_ID,
    assert_no_canonical_leakage,
    build_split_manifest,
    canonicalise_source_rows,
    file_sha256,
    make_fresh_tasks,
    split_canonical_tasks,
    write_jsonl,
)


def _source_rows(revision: str | None) -> tuple[list[dict[str, Any]], str]:
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - depends on optional extra.
        raise RuntimeError("Install the project train extra before preparing source data") from error

    info = HfApi().dataset_info(DATASET_ID, revision=revision)
    resolved_revision = info.sha
    dataset = load_dataset(DATASET_ID, split="train", revision=resolved_revision)
    return [dict(row) for row in dataset], resolved_revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=None, help="Optional source dataset revision; default resolves current SHA")
    parser.add_argument("--fresh-size", type=int, default=256)
    parser.add_argument("--smoke-size", type=int, default=8)
    args = parser.parse_args()

    rows, revision = _source_rows(args.revision)
    source_tasks, audit = canonicalise_source_rows(rows)
    splits = split_canonical_tasks(source_tasks, seed=args.seed)
    assert_no_canonical_leakage(splits)
    forbidden = [task["canonical_key"] for records in splits.values() for task in records]
    fresh = make_fresh_tasks(count=args.fresh_size, seed=args.seed + 1, forbidden_keys=forbidden)
    smoke = splits["train"][: args.smoke_size]

    args.data_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        write_jsonl(args.data_dir / f"source_{split_name}.jsonl", records)
    write_jsonl(args.data_dir / "fresh_test.jsonl", fresh)
    write_jsonl(args.data_dir / "smoke_train.jsonl", smoke)

    audit.update(
        {
            "dataset_revision": revision,
            "resolved_at": datetime.now(UTC).isoformat(),
            "source_split": "train",
            "seed": args.seed,
        }
    )
    manifest = build_split_manifest(splits, seed=args.seed)
    manifest.update(
        {
            "dataset_id": DATASET_ID,
            "dataset_revision": revision,
            "fresh_test_count": len(fresh),
            "fresh_generator": {
                "seed": args.seed + 1,
                "number_count_range": [3, 4],
                "number_range": [1, 25],
                "target_range": [10, 999],
                "oracle_witnesses_written": False,
            },
            "files": {},
        }
    )
    for path in sorted(args.data_dir.glob("*.jsonl")):
        manifest["files"][path.name] = {"sha256": file_sha256(path), "record_count": len(path.read_text().splitlines())}
    (args.data_dir / "source_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (args.data_dir / "source_split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"audit": audit, "manifest": manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Paired confirmation measurements and fixed-order traces for every seed."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .comparison import audit_pairs, normalized_record, paired_bootstrap, task_outcomes
from .data import file_sha256, read_jsonl, write_jsonl


def compare_records(baseline, trained):
    def identities(rows):
        return {(row["task_id"], row["generation_mode"], row["sample_index"]):
                (row["target"], tuple(sorted(row["nums"])), row["prompt"],
                 row.get("rendered_prompt", row["prompt"]),
                 json.dumps(row["generation_config"], sort_keys=True), row["seed"],
                 row.get("model_id"), row.get("revision"))
                for row in rows}
    if identities(baseline) != identities(trained) or len(identities(baseline)) != len(baseline):
        raise ValueError("confirmation task/prompt/generation identities must match uniquely")
    output = {}
    for split in sorted({row["split"] for row in baseline}):
        left = [row for row in baseline if row["split"] == split]
        right = [row for row in trained if row["split"] == split]
        output[split] = {}
        for mode in ("greedy", "sample"):
            output[split][mode] = paired_bootstrap(task_outcomes(left, mode), task_outcomes(right, mode))
            output[split][mode]["normalized"] = paired_bootstrap(
                task_outcomes([normalized_record(row) for row in left], mode),
                task_outcomes([normalized_record(row) for row in right], mode))
        solvable_ids = {row["task_id"] for row in left if row["oracle_solvable"]}
        if solvable_ids:
            output[split]["solvable_only"] = {
                mode: paired_bootstrap(task_outcomes([row for row in left if row["task_id"] in solvable_ids], mode),
                                       task_outcomes([row for row in right if row["task_id"] in solvable_ids], mode))
                for mode in ("greedy", "sample")}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, nargs=3, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    if freeze.get("status") != "frozen" or len(set(args.trained)) != 3:
        raise ValueError("three distinct trained evaluations and a frozen design are required")
    checkpoints = freeze["checkpoints"]
    if set(checkpoints) != {"42", "43", "44"} or len(set(checkpoints.values())) != 3:
        raise ValueError("freeze must identify three independent seed checkpoints")
    if args.output_dir.exists():
        raise ValueError("report output must be new")
    baseline = read_jsonl(args.baseline)
    args.output_dir.mkdir(parents=True)
    result = {"baseline_sha256": file_sha256(args.baseline),
              "freeze_sha256": file_sha256(args.freeze), "seeds": {}}
    aggregate = defaultdict(list)
    rows = []
    for seed, path in zip(sorted(checkpoints), args.trained, strict=True):
        trained = read_jsonl(path)
        if not trained or {row["adapter_path"] for row in trained} != {checkpoints[seed]}:
            raise ValueError(f"seed {seed} evaluation does not match its frozen checkpoint")
        # Generation seeds are paired; the training seed comes from the checkpoint label.
        label = f"seed-{seed}"
        measurements = compare_records(baseline, trained)
        result["seeds"][label] = {"records_sha256": file_sha256(path), "measurements": measurements}
        write_jsonl(args.output_dir / f"{label}-audit.jsonl", audit_pairs(baseline, trained))
        for split, modes in measurements.items():
            for mode in ("greedy", "sample"):
                values = modes[mode]
                interval = values["paired_gain_percentile_95"]
                rows.append(f"| {label} | {split} | {mode} | {values['baseline_rate']:.2%} | "
                            f"{values['trained_rate']:.2%} | {values['paired_gain']:.2%} | "
                            f"[{interval[0]:.2%}, {interval[1]:.2%}] |")
                aggregate[(split, mode)].append(task_outcomes([row for row in trained if row["split"] == split], mode))
    result["aggregate"] = {}
    for (split, mode), seeds in aggregate.items():
        averaged = {key: sum(seed[key] for seed in seeds) / len(seeds) for key in seeds[0]}
        base_outcomes = task_outcomes([row for row in baseline if row["split"] == split], mode)
        result["aggregate"].setdefault(split, {})[mode] = paired_bootstrap(base_outcomes, averaged)
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "report.md").write_text("\n".join([
        "# Frozen confirmation comparison", "",
        "Greedy and sampled pass@4 are task-level measurements. Paired percentile",
        "bootstrap intervals use 10,000 task resamples with seed 20260918.",
        "Aggregate intervals condition on these three observed training runs; they",
        "do not measure uncertainty over the population of possible training seeds.", "",
        "| Checkpoint | Suite | Mode | Initialization | Trained | Paired gain | 95% interval |",
        "|---|---|---|---:|---:|---:|---|", *rows, "",
        "Normalized comparisons and solvable-only results are in comparison.json.",
        "Audit JSONL includes both gains and losses selected in fixed task-ID order.",
        "Ambiguous illegal-to-valid changes do not by themselves prove arithmetic search.", "",
    ]))


if __name__ == "__main__":
    main()

"""Paired confirmation measurements and fixed-order traces for every seed."""

import argparse
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .comparison import audit_pairs, normalized_record, paired_bootstrap, task_outcomes
from .data import file_sha256, read_jsonl, write_jsonl


def write_confirmation_svg(path: Path, comparison: dict) -> None:
    """Draw primary pass@1/pass@4 rates directly from the saved comparison."""

    panels = [
        ("source_confirmation", "greedy", "Source confirmation · greedy pass@1"),
        ("source_confirmation", "sample", "Source confirmation · sampled pass@4"),
        ("fresh_confirmation", "greedy", "Fresh confirmation · greedy pass@1"),
        ("fresh_confirmation", "sample", "Fresh confirmation · sampled pass@4"),
    ]
    colors = ["#687386", "#2d6cdf", "#24a0a8", "#8a63d2", "#23835b"]
    series_names = ["Base", "Seed 42", "Seed 43", "Seed 44", "Mean"]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="650" viewBox="0 0 1000 650">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="36" y="32" font-family="sans-serif" font-size="21" font-weight="bold">Frozen Countdown confirmation accuracy</text>',
        '<text x="36" y="54" font-family="sans-serif" font-size="13" fill="#46505e">Task-level pass@1 and pass@4 · base vs three final-step supervised adapters</text>',
    ]
    for name_index, name in enumerate(series_names):
        x = 552 + name_index * 89
        parts.extend(
            [
                f'<rect x="{x}" y="17" width="12" height="12" rx="2" fill="{colors[name_index]}"/>',
                f'<text x="{x + 17}" y="28" font-family="sans-serif" font-size="12">{html.escape(name)}</text>',
            ]
        )
    for panel_index, (split, mode, title) in enumerate(panels):
        col, row = panel_index % 2, panel_index // 2
        x0, y0 = 42 + col * 478, 78 + row * 278
        values = [comparison["seeds"][f"seed-{seed}"]["measurements"][split][mode]["trained_rate"] for seed in (42, 43, 44)]
        values = [comparison["seeds"]["seed-42"]["measurements"][split][mode]["baseline_rate"], *values,
                  comparison["aggregate"][split][mode]["trained_rate"]]
        ceiling = min(1.0, max(0.05, math.ceil(max(values) * 20) / 20))
        plot_x, plot_y, plot_w, plot_h = x0 + 46, y0 + 40, 370, 170
        parts.append(f'<text x="{x0}" y="{y0 + 18}" font-family="sans-serif" font-size="15" font-weight="bold">{html.escape(title)}</text>')
        for tick in range(5):
            fraction = tick / 4
            y = plot_y + plot_h * (1 - fraction)
            label = f"{ceiling * fraction:.0%}"
            parts.extend(
                [
                    f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#d7dce3" stroke-width="1"/>',
                    f'<text x="{plot_x - 7}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#596273">{label}</text>',
                ]
            )
        slot = plot_w / len(values)
        bar_w = 43
        for bar_index, value in enumerate(values):
            height = plot_h * value / ceiling
            bx = plot_x + slot * bar_index + (slot - bar_w) / 2
            by = plot_y + plot_h - height
            parts.extend(
                [
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w}" height="{height:.1f}" rx="3" fill="{colors[bar_index]}"/>',
                    f'<text x="{bx + bar_w / 2:.1f}" y="{max(plot_y - 2, by - 5):.1f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#273140">{value:.1%}</text>',
                    f'<text x="{bx + bar_w / 2:.1f}" y="{plot_y + plot_h + 18}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#46505e">{html.escape(series_names[bar_index])}</text>',
                ]
            )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


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
              "freeze_sha256": file_sha256(args.freeze), "seeds": {}, "audit_counts": {}}
    aggregate = defaultdict(list)
    normalized_aggregate = defaultdict(list)
    rows = []
    for seed, path in zip(sorted(checkpoints), args.trained, strict=True):
        trained = read_jsonl(path)
        if not trained or {row["adapter_path"] for row in trained} != {checkpoints[seed]}:
            raise ValueError(f"seed {seed} evaluation does not match its frozen checkpoint")
        # Generation seeds are paired; the training seed comes from the checkpoint label.
        label = f"seed-{seed}"
        measurements = compare_records(baseline, trained)
        result["seeds"][label] = {"records_sha256": file_sha256(path), "measurements": measurements}
        audit = audit_pairs(baseline, trained)
        write_jsonl(args.output_dir / f"{label}-audit.jsonl", audit)
        result["audit_counts"][label] = dict(sorted(Counter(row["rubric_category"] for row in audit).items()))
        for split, modes in measurements.items():
            for mode in ("greedy", "sample"):
                values = modes[mode]
                interval = values["paired_gain_percentile_95"]
                rows.append(f"| {label} | {split} | {mode} | {values['baseline_rate']:.2%} | "
                            f"{values['trained_rate']:.2%} | {values['paired_gain']:.2%} | "
                            f"[{interval[0]:.2%}, {interval[1]:.2%}] |")
                split_rows = [row for row in trained if row["split"] == split]
                aggregate[(split, mode)].append(task_outcomes(split_rows, mode))
                normalized_aggregate[(split, mode)].append(
                    task_outcomes([normalized_record(row) for row in split_rows], mode)
                )
    result["aggregate"] = {}
    for (split, mode), seeds in aggregate.items():
        averaged = {key: sum(seed[key] for seed in seeds) / len(seeds) for key in seeds[0]}
        base_outcomes = task_outcomes([row for row in baseline if row["split"] == split], mode)
        normalized_base = task_outcomes(
            [normalized_record(row) for row in baseline if row["split"] == split], mode
        )
        normalized_seeds = normalized_aggregate[(split, mode)]
        normalized_average = {
            key: sum(seed[key] for seed in normalized_seeds) / len(normalized_seeds)
            for key in normalized_seeds[0]
        }
        result["aggregate"].setdefault(split, {})[mode] = paired_bootstrap(base_outcomes, averaged)
        result["aggregate"][split][mode]["normalized"] = paired_bootstrap(
            normalized_base, normalized_average
        )
    aggregate_rows = []
    normalized_rows = []
    for split in sorted(result["aggregate"]):
        for mode in ("greedy", "sample"):
            values = result["aggregate"][split][mode]
            interval = values["paired_gain_percentile_95"]
            aggregate_rows.append(
                f"| three-seed mean | {split} | {mode} | {values['baseline_rate']:.2%} | "
                f"{values['trained_rate']:.2%} | {values['paired_gain']:.2%} | "
                f"[{interval[0]:.2%}, {interval[1]:.2%}] |"
            )
            if mode == "greedy":
                normalized = values["normalized"]
                interval = normalized["paired_gain_percentile_95"]
                normalized_rows.append(
                    f"| three-seed mean | {split} | {normalized['baseline_rate']:.2%} | "
                    f"{normalized['trained_rate']:.2%} | {normalized['paired_gain']:.2%} | "
                    f"[{interval[0]:.2%}, {interval[1]:.2%}] |"
                )
    audit_rows = []
    for seed, counts in result["audit_counts"].items():
        audit_rows.append(
            f"| {seed} | {counts.get('changed_legal_arithmetic_reaches_target', 0)} | "
            f"{counts.get('format_only_repair_sufficient', 0)} | "
            f"{counts.get('constraint_or_search_ambiguous', 0)} | "
            f"{counts.get('lost_primary_solution', 0)} |"
        )
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "report.md").write_text("\n".join([
        "# Frozen confirmation comparison", "",
        "Greedy and sampled pass@4 are task-level measurements. Paired percentile",
        "bootstrap intervals use 10,000 task resamples with seed 20260918.",
        "Aggregate intervals condition on these three observed training runs; they",
        "do not measure uncertainty over the population of possible training seeds.", "",
        "| Checkpoint | Suite | Mode | Initialization | Trained | Paired gain | 95% interval |",
        "|---|---|---|---:|---:|---:|---|", *rows, "",
        "## Three-seed aggregate",
        "",
        "This is the mean of three seed-specific task outcomes. The interval resamples tasks; it is not uncertainty over the training-seed population.",
        "",
        "| Models | Suite | Mode | Initialization | Mean trained | Paired gain | 95% interval |",
        "|---|---|---|---:|---:|---:|---|", *aggregate_rows, "",
        "## Greedy comparison after equality-suffix normalization",
        "",
        "The diagnostic removes only a terminal `= integer` from extracted expressions, then applies the unchanged verifier.",
        "It checks whether that simple formatting repair is sufficient to explain the gain.",
        "",
        "| Models | Suite | Initialization | Mean trained | Paired gain | 95% interval |",
        "|---|---|---:|---:|---:|---|", *normalized_rows, "",
        "## Fixed-order trace audit",
        "",
        "Each seed includes up to 20 greedy primary-solve disagreements selected by task ID across both suites, including losses.",
        "Repeated tasks across seeds are repeated seed-task comparisons, not independent puzzles.",
        "",
        "| Seed | Changed legal arithmetic reaches target | Formatting repair sufficient | Ambiguous | Lost solution |",
        "|---|---:|---:|---:|---:|", *audit_rows, "",
        "The audit supports a narrow arithmetic-search interpretation where a legal wrong-target expression changes to a legal exact expression.",
        "Ambiguous illegal-to-valid changes do not establish arithmetic search by themselves.",
        "",
        "Normalized comparisons and solvable-only results are in comparison.json.",
        "Audit JSONL includes both gains and losses selected in fixed task-ID order.",
        "",
    ]))
    write_confirmation_svg(args.output_dir / "comparison.svg", result)


if __name__ == "__main__":
    main()

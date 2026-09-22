"""Generate paired comparisons for the frozen SFT-initialized GRPO follow-up."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .comparison import audit_pairs, normalized_record, paired_bootstrap, task_outcomes
from .data import file_sha256, read_jsonl, write_jsonl
from .report import _diagnostic_summary


def _validate_paired(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> None:
    def identities(rows):
        return {
            (row["task_id"], row["generation_mode"], row["sample_index"]): (
                row["target"], tuple(sorted(row["nums"])), row["prompt"],
                row.get("rendered_prompt", row["prompt"]),
                json.dumps(row["generation_config"], sort_keys=True), row["seed"],
                row.get("model_id"), row.get("revision"), row["oracle_solvable"],
            )
            for row in rows
        }

    if identities(left) != identities(right) or len(left) != len(identities(left)):
        raise ValueError("paired evaluations must contain identical unique task/generation records")


def _compare(left: list[dict[str, Any]], right: list[dict[str, Any]], *, bootstrap_seed: int):
    _validate_paired(left, right)
    result: dict[str, Any] = {}
    for split in sorted({row["split"] for row in left}):
        left_split = [row for row in left if row["split"] == split]
        right_split = [row for row in right if row["split"] == split]
        result[split] = {}
        for mode in ("greedy", "sample"):
            result[split][mode] = paired_bootstrap(
                task_outcomes(left_split, mode), task_outcomes(right_split, mode),
                resamples=10_000, seed=bootstrap_seed,
            )
            result[split][mode]["normalized"] = paired_bootstrap(
                task_outcomes([normalized_record(row) for row in left_split], mode),
                task_outcomes([normalized_record(row) for row in right_split], mode),
                resamples=10_000, seed=bootstrap_seed,
            )
        solvable_ids = {
            row["task_id"] for row in left_split
            if row.get("oracle_solvable") and row["generation_mode"] == "greedy"
        }
        if solvable_ids:
            result[split]["solvable_only"] = {
                mode: paired_bootstrap(
                    task_outcomes([row for row in left_split if row["task_id"] in solvable_ids], mode),
                    task_outcomes([row for row in right_split if row["task_id"] in solvable_ids], mode),
                    resamples=10_000, seed=bootstrap_seed,
                )
                for mode in ("greedy", "sample")
            }
    return result


def _mean_outcomes(per_seed: list[list[dict[str, Any]]], split: str, mode: str):
    maps = [task_outcomes([row for row in rows if row["split"] == split], mode) for rows in per_seed]
    return {key: sum(values[key] for values in maps) / len(maps) for key in maps[0]}


def _aggregate(left: list[list[dict[str, Any]]], right: list[list[dict[str, Any]]], *, seed: int):
    result = {}
    splits = sorted({row["split"] for rows in left for row in rows})
    for split in splits:
        result[split] = {}
        for mode in ("greedy", "sample"):
            result[split][mode] = paired_bootstrap(
                _mean_outcomes(left, split, mode), _mean_outcomes(right, split, mode),
                resamples=10_000, seed=seed,
            )
            norm_left = [[normalized_record(row) for row in rows] for rows in left]
            norm_right = [[normalized_record(row) for row in rows] for rows in right]
            result[split][mode]["normalized"] = paired_bootstrap(
                _mean_outcomes(norm_left, split, mode), _mean_outcomes(norm_right, split, mode),
                resamples=10_000, seed=seed,
            )
    return result


def _write_svg(path: Path, results: dict[str, Any]) -> None:
    suites = ("source_confirmation", "fresh_confirmation")
    methods = (("base", "Base"), ("sft", "SFT init"), ("grpo", "GRPO"))
    colors = {"base": "#667085", "sft": "#2d6cdf", "grpo": "#20835b"}
    panels = [(suite, mode) for suite in suites for mode in ("greedy", "sample")]
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="590" viewBox="0 0 980 590">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="32" y="30" font-family="sans-serif" font-size="20" font-weight="bold">SFT initialization and binary GRPO confirmation</text>',
        '<text x="32" y="52" font-family="sans-serif" font-size="13" fill="#475467">Three-seed mean on frozen source and fresh tasks; exact task-level pass@1 / pass@4</text>',
    ]
    for i, (key, label) in enumerate(methods):
        x = 660 + i * 98
        parts.extend([f'<rect x="{x}" y="21" width="12" height="12" fill="{colors[key]}"/>',
                      f'<text x="{x + 17}" y="32" font-family="sans-serif" font-size="12">{label}</text>'])
    for panel_index, (suite, mode) in enumerate(panels):
        col, row = panel_index % 2, panel_index // 2
        x0, y0 = 35 + col * 480, 75 + row * 250
        metric = "pass@1" if mode == "greedy" else "pass@4"
        parts.append(f'<text x="{x0}" y="{y0 + 18}" font-family="sans-serif" font-size="15" font-weight="bold">{html.escape(suite.replace("_", " ").title())} {metric}</text>')
        left, top, width, height = x0 + 42, y0 + 40, 390, 155
        vals = [results["aggregate"]["base_vs_sft"][suite][mode]["trained_rate"],
                results["aggregate"]["base_vs_grpo"][suite][mode]["trained_rate"]]
        base = results["aggregate"]["base_vs_sft"][suite][mode]["baseline_rate"]
        vals = [base, vals[0], vals[1]]
        ceiling = min(1.0, max(0.05, ((max(vals) * 20 + 0.999999) // 1) / 20))
        for tick in range(5):
            fraction = tick / 4
            y = top + height * (1 - fraction)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" stroke="#e4e7ec"/>')
            parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#667085">{ceiling * fraction:.0%}</text>')
        slot, bar_width = width / 3, 54
        for index, ((key, label), value) in enumerate(zip(methods, vals, strict=True)):
            bar_height = height * value / ceiling
            bx = left + index * slot + (slot - bar_width) / 2
            by = top + height - bar_height
            parts.extend([f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width}" height="{bar_height:.1f}" rx="3" fill="{colors[key]}"/>',
                          f'<text x="{bx + bar_width / 2:.1f}" y="{max(top - 3, by - 5):.1f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.1%}</text>',
                          f'<text x="{bx + bar_width / 2:.1f}" y="{top + height + 17}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>'])
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def generate_report(*, baseline_path: Path, sft_paths: list[Path], grpo_paths: list[Path], freeze_path: Path, evidence_dir: Path, output_dir: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen" or len(sft_paths) != 3 or len(grpo_paths) != 3:
        raise ValueError("a frozen design and three SFT/GRPO evaluation pairs are required")
    if output_dir.exists():
        raise ValueError("report output directory must be new")
    baseline = read_jsonl(baseline_path)
    sft = [read_jsonl(path) for path in sft_paths]
    grpo = [read_jsonl(path) for path in grpo_paths]
    for index, seed in enumerate((42, 43, 44)):
        expected_start = freeze["starting_checkpoints"][str(seed)]["adapter_path"]
        expected_grpo = f"outputs/sft-init-grpo-s{seed}/final-adapter"
        if {row.get("adapter_path") for row in sft[index]} != {expected_start}:
            raise ValueError(f"seed {seed} SFT evaluation does not match its frozen initialization")
        if {row.get("adapter_path") for row in grpo[index]} != {expected_grpo}:
            raise ValueError(f"seed {seed} GRPO evaluation does not match its final checkpoint")
        _validate_paired(baseline, sft[index])
        _validate_paired(baseline, grpo[index])
        _validate_paired(sft[index], grpo[index])

    output_dir.mkdir(parents=True)
    result: dict[str, Any] = {
        "freeze_sha256": file_sha256(freeze_path),
        "baseline_sha256": file_sha256(baseline_path),
        "bootstrap": {"resamples": 10_000, "seed": freeze["confirmation"]["bootstrap"]["seed"], "unit": "task"},
        "seeds": {},
        "aggregate": {},
        "training": {},
        "audit_counts": {},
    }
    bootstrap_seed = result["bootstrap"]["seed"]
    for seed, sft_rows, grpo_rows, sft_path, grpo_path in zip((42, 43, 44), sft, grpo, sft_paths, grpo_paths, strict=True):
        label = f"seed-{seed}"
        result["seeds"][label] = {
            "sft_sha256": file_sha256(sft_path),
            "grpo_sha256": file_sha256(grpo_path),
            "base_vs_sft": _compare(baseline, sft_rows, bootstrap_seed=bootstrap_seed),
            "base_vs_grpo": _compare(baseline, grpo_rows, bootstrap_seed=bootstrap_seed),
            "sft_vs_grpo": _compare(sft_rows, grpo_rows, bootstrap_seed=bootstrap_seed),
        }
        seed_dir = evidence_dir / f"grpo-s{seed}"
        diagnostics = _diagnostic_summary(seed_dir / "grpo-diagnostics.jsonl")
        result["training"][label] = {
            **diagnostics,
            "attempt": json.loads((seed_dir / "attempt.json").read_text(encoding="utf-8")),
            "run_config": json.loads((seed_dir / "run-config.json").read_text(encoding="utf-8")),
            "initial_adapter_sha256": json.loads((seed_dir / "run-config.json").read_text(encoding="utf-8"))["starting_adapter"]["sha256"],
            "final_adapter_sha256": file_sha256(Path(f"outputs/sft-init-grpo-s{seed}/final-adapter/adapter_model.safetensors")),
        }
        audit = []
        for split in ("source_confirmation", "fresh_confirmation"):
            left = [row for row in sft_rows if row["split"] == split]
            right = [row for row in grpo_rows if row["split"] == split]
            audit.extend(audit_pairs(left, right, limit=10))
        write_jsonl(output_dir / f"{label}-audit.jsonl", audit)
        result["audit_counts"][label] = dict(sorted(Counter(row["rubric_category"] for row in audit).items()))

    contrasts = {
        "base_vs_sft": ([baseline] * 3, sft),
        "base_vs_grpo": ([baseline] * 3, grpo),
        "sft_vs_grpo": (sft, grpo),
    }
    for name, (left, right) in contrasts.items():
        result["aggregate"][name] = _aggregate(left, right, seed=bootstrap_seed)
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = ["| Contrast | Suite | Metric | Initialization | Trained mean | Paired gain | 95% task-bootstrap interval |", "|---|---|---|---:|---:|---:|---|"]
    for contrast in ("base_vs_sft", "base_vs_grpo", "sft_vs_grpo"):
        for split, metrics in result["aggregate"][contrast].items():
            for mode in ("greedy", "sample"):
                values = metrics[mode]
                interval = values["paired_gain_percentile_95"]
                rows.append(f"| {contrast} | {split} | {'pass@1' if mode == 'greedy' else 'pass@4'} | {values['baseline_rate']:.2%} | {values['trained_rate']:.2%} | {values['paired_gain']:.2%} | [{interval[0]:.2%}, {interval[1]:.2%}] |")
    lines = [
        "# SFT-initialized binary-GRPO confirmation",
        "",
        "Generated only from the frozen design, saved evaluations, final checkpoints, and per-step GRPO diagnostics.",
        "The outcome reward and exact Countdown verifier are unchanged. Reward changes alone are not treated as learning evidence.",
        "",
        "## Three-seed paired comparisons",
        "",
        *rows,
        "",
        "Each per-seed result and task bootstrap is in `comparison.json`. Three-seed intervals resample paired tasks while averaging these three observed training seeds; they are not uncertainty over the full population of possible seeds.",
        "The fixed search audit compares matching SFT and GRPO greedy completions, includes gains and losses, and is stratified to at most 10 source and 10 fresh tasks per seed.",
        "Equality-suffix normalization and source-solvable-only comparisons are reported in the JSON.",
        "",
        "## GRPO training diagnostics",
        "",
        "| Seed | Steps | Rollouts | Mean reward | Positive rate | Mixed groups | All-zero groups | All-one groups | Truncation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (42, 43, 44):
        diag = result["training"][f"seed-{seed}"]
        lines.append(f"| {seed} | {diag['steps']} | {diag['rollouts']} | {diag['mean_reward']:.3f} | {diag['positive_completion_rate']:.3f} | {diag['mixed_reward_group_rate']:.3f} | {diag['all_zero_group_rate']:.3f} | {diag['all_one_group_rate']:.3f} | {diag['truncation_rate']:.3f} |")
    lines.extend(["", "Positive/negative conclusion language must follow the paired exact-task comparisons, not training reward alone.", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_svg(output_dir / "comparison.svg", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sft", type=Path, nargs=3, required=True)
    parser.add_argument("--grpo", type=Path, nargs=3, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate_report(
        baseline_path=args.baseline, sft_paths=args.sft, grpo_paths=args.grpo,
        freeze_path=args.freeze, evidence_dir=args.evidence_dir, output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "seeds": sorted(result["seeds"]), "aggregate_contrasts": sorted(result["aggregate"])}, indent=2))


if __name__ == "__main__":
    main()

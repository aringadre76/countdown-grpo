"""Report a frozen, exact-verifier-filtered test-time sampling experiment."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from random import Random
from typing import Any

from .data import file_sha256, read_jsonl, write_jsonl

SPLITS = ("source_confirmation", "fresh_confirmation")
METRICS = (
    "greedy_pass_at_1",
    "sample_pass_at_4",
    "sample_pass_at_8",
    "search_at_1",
    "search_at_5",
    "search_at_9",
)


def _paired_bootstrap(
    baseline: dict[str, float], trained: dict[str, float], *, resamples: int, seed: int
) -> dict[str, Any]:
    if not baseline or set(baseline) != set(trained):
        raise ValueError("paired comparison requires identical nonempty task identities")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    keys = sorted(baseline)
    differences = [trained[key] - baseline[key] for key in keys]
    rng = Random(seed)
    draws = sorted(sum(rng.choices(differences, k=len(keys))) / len(keys) for _ in range(resamples))
    return {
        "tasks": len(keys),
        "baseline_rate": sum(baseline.values()) / len(keys),
        "trained_rate": sum(trained.values()) / len(keys),
        "paired_gain": sum(differences) / len(keys),
        "paired_gain_percentile_95": [
            draws[int(0.025 * (resamples - 1))],
            draws[int(0.975 * (resamples - 1))],
        ],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def _validated_task_groups(
    records: list[dict[str, Any]], *, samples_per_task: int
) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"greedy": [], "sample": []})
    for row in records:
        if row["generation_mode"] not in {"greedy", "sample"}:
            raise ValueError(f"unexpected generation mode {row['generation_mode']!r}")
        groups[str(row["task_id"])][row["generation_mode"]].append(row)

    validated: dict[str, dict[str, Any]] = {}
    for task_id, group in groups.items():
        greedy = group["greedy"]
        sampled = sorted(group["sample"], key=lambda row: int(row["sample_index"]))
        if len(greedy) != 1 or len(sampled) != samples_per_task:
            raise ValueError(
                f"{task_id}: expected one greedy and {samples_per_task} sampled completions"
            )
        if [int(row["sample_index"]) for row in sampled] != list(range(samples_per_task)):
            raise ValueError(f"{task_id}: sampled completion indices are not contiguous from zero")
        rows = greedy + sampled
        identities = {
            (row["split"], int(row["target"]), tuple(sorted(int(n) for n in row["nums"])))
            for row in rows
        }
        if len(identities) != 1:
            raise ValueError(f"{task_id}: task fields differ across completions")
        split, target, numbers = next(iter(identities))
        if split not in SPLITS:
            raise ValueError(f"{task_id}: unexpected split {split!r}")
        validated[task_id] = {
            "greedy": greedy[0],
            "sampled": sampled,
            "split": split,
            "target": target,
            "nums": list(numbers),
            "oracle_solvable": all(bool(row["oracle_solvable"]) for row in rows),
        }
    if not validated:
        raise ValueError("evaluation has no records")
    return validated


def _task_outcomes(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    outcomes: dict[str, dict[str, dict[str, float]]] = {
        split: {metric: {} for metric in METRICS} for split in SPLITS
    }
    for task_id, group in groups.items():
        greedy = float(group["greedy"]["reward"] == 1.0)
        sampled = [float(row["reward"] == 1.0) for row in group["sampled"]]
        candidates = [greedy, *sampled]
        task_values = {
            "greedy_pass_at_1": greedy,
            "sample_pass_at_4": float(any(sampled[:4])),
            "sample_pass_at_8": float(any(sampled[:8])),
            "search_at_1": float(any(candidates[:1])),
            "search_at_5": float(any(candidates[:5])),
            "search_at_9": float(any(candidates[:9])),
        }
        for metric, value in task_values.items():
            outcomes[group["split"]][metric][task_id] = value
    return outcomes


def _select_candidate(group: dict[str, Any]) -> dict[str, Any]:
    """Return the earliest exact solution, using only saved verifier rewards."""
    candidates = [group["greedy"], *group["sampled"]]
    for checked, candidate in enumerate(candidates, start=1):
        if candidate["reward"] == 1.0:
            return {
                "selected_candidate_mode": candidate["generation_mode"],
                "selected_sample_index": candidate["sample_index"],
                "candidates_checked": checked,
                "raw_completion": candidate["raw_completion"],
                "extracted_expression": candidate["extracted_expression"],
                "verifier_result": candidate["verifier_result"],
                "reward": 1.0,
                "failure_category": "ok",
            }
    return {
        "selected_candidate_mode": None,
        "selected_sample_index": None,
        "candidates_checked": len(candidates),
        "raw_completion": None,
        "extracted_expression": None,
        "verifier_result": None,
        "reward": 0.0,
        "failure_category": "no_exact_candidate_in_budget",
    }


def _mean_maps(maps: list[dict[str, float]]) -> dict[str, float]:
    if not maps or any(set(item) != set(maps[0]) for item in maps[1:]):
        raise ValueError("cannot average unpaired task outcome maps")
    return {key: sum(item[key] for item in maps) / len(maps) for key in maps[0]}


def _svg(path: Path, rates: dict[str, dict[str, float]]) -> None:
    width, row_height, left, max_width = 980, 34, 240, 620
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{120 + row_height * len(rates)}" viewBox="0 0 {width} {120 + row_height * len(rates)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="sans-serif" font-size="22" font-weight="bold">Exact-verifier search solve rates</text>',
        '<text x="28" y="61" font-family="sans-serif" font-size="14" fill="#475467">Greedy@1 versus ordered verifier selection across 9 candidates</text>',
    ]
    y = 92
    palette = {"greedy_pass_at_1": "#7a869a", "search_at_9": "#147d64"}
    for label, metrics in rates.items():
        for metric in ("greedy_pass_at_1", "search_at_9"):
            value = metrics[metric]
            bar_y = y if metric == "greedy_pass_at_1" else y + 12
            lines.extend([
                f'<text x="20" y="{bar_y + 10}" font-family="sans-serif" font-size="12">{html.escape(label)} {"greedy" if metric == "greedy_pass_at_1" else "search@9"}</text>',
                f'<rect x="{left}" y="{bar_y}" width="{max_width * max(0.0, min(1.0, value)):.2f}" height="10" fill="{palette[metric]}"/>',
                f'<text x="{left + max_width + 12}" y="{bar_y + 10}" font-family="sans-serif" font-size="12">{value:.2%}</text>',
            ])
        y += row_height
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(
    *,
    baseline_path: Path,
    sft_paths: list[Path],
    freeze_path: Path,
    manifest_path: Path,
    environment_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen":
        raise ValueError("frozen design is required")
    if manifest.get("freeze_sha256") != file_sha256(freeze_path):
        raise ValueError("confirmation manifest was not generated from this frozen design")
    if len(sft_paths) != 3:
        raise ValueError("exactly three supervised seed evaluations are required")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("report output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_paths = [baseline_path, *sft_paths]
    eval_names = ["base", "sft_42", "sft_43", "sft_44"]
    task_groups: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    raw_records_by_policy: dict[str, list[dict[str, Any]]] = {}
    for policy, path in zip(eval_names, eval_paths, strict=True):
        records = read_jsonl(path)
        raw_records_by_policy[policy] = records
        groups = _validated_task_groups(records, samples_per_task=8)
        task_groups[policy] = groups
        summary_path = path.with_suffix(".summary.json")
        summaries[policy] = json.loads(summary_path.read_text(encoding="utf-8"))
        if summaries[policy].get("record_count") != len(records):
            raise ValueError(f"{policy}: summary record count disagrees with JSONL")

    expected_tasks = set(manifest["source_task_ids"]) | set(manifest["fresh_task_ids"])
    for policy, groups in task_groups.items():
        if set(groups) != expected_tasks:
            raise ValueError(f"{policy}: evaluated task identities do not match frozen manifest")
    base_groups = task_groups["base"]
    for policy, groups in task_groups.items():
        for task_id in expected_tasks:
            left = base_groups[task_id]
            right = groups[task_id]
            if (left["split"], left["target"], sorted(left["nums"])) != (
                right["split"], right["target"], sorted(right["nums"])
            ):
                raise ValueError(f"{policy}/{task_id}: task contents differ from base evaluation")

    policy_outcomes = {name: _task_outcomes(groups) for name, groups in task_groups.items()}
    solvable_source_ids = {
        task_id
        for task_id, group in base_groups.items()
        if group["split"] == "source_confirmation" and group["oracle_solvable"]
    }
    source_solvable_outcomes = {
        policy: _task_outcomes({task_id: groups[task_id] for task_id in solvable_source_ids})[
            "source_confirmation"
        ]
        for policy, groups in task_groups.items()
    }
    bootstrap = freeze["confirmation"]["paired_bootstrap"]
    n_resamples, bootstrap_seed = bootstrap["resamples"], bootstrap["seed"]
    report: dict[str, Any] = {
        "experiment_id": freeze["experiment_id"],
        "freeze_sha256": file_sha256(freeze_path),
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_inputs": {
            name: {"jsonl": str(path), "sha256": file_sha256(path), "summary": summaries[name]}
            for name, path in zip(eval_names, eval_paths, strict=True)
        },
        "environment": environment,
        "compute": {
            "evaluation_count": len(summaries),
            "evaluation_wall_seconds": sum(float(s["wall_time_seconds"]) for s in summaries.values()),
            "evaluation_process_hours": sum(float(s["wall_time_seconds"]) for s in summaries.values()) / 3600,
            "gpu_cap_hours": freeze["budget"]["hard_cap_gpu_process_hours"],
            "completion_count": sum(int(s["record_count"]) for s in summaries.values()),
            "truncation_count": sum(
                int(row["truncation"])
                for records in raw_records_by_policy.values()
                for row in records
            ),
            "devices": {name: sorted({str(row["device"]) for row in read_jsonl(path)}) for name, path in zip(eval_names, eval_paths, strict=True)},
        },
        "per_seed": {},
        "three_seed_mean": {},
        "source_solvable_only": {"task_count": len(solvable_source_ids), "policies": {}, "per_seed_vs_base": {}},
        "search_selection": {},
    }
    for policy in eval_names:
        report["source_solvable_only"]["policies"][policy] = {
            metric: {
                "tasks": len(outcomes),
                "rate": sum(outcomes.values()) / len(outcomes) if outcomes else None,
            }
            for metric, outcomes in source_solvable_outcomes[policy].items()
        }

    for seed, policy in zip((42, 43, 44), eval_names[1:], strict=True):
        report["per_seed"][str(seed)] = {"policy": policy, "outcomes": policy_outcomes[policy], "vs_base": {}}
        for split in SPLITS:
            for metric in METRICS:
                report["per_seed"][str(seed)]["vs_base"].setdefault(split, {})[metric] = _paired_bootstrap(
                    policy_outcomes["base"][split][metric],
                    policy_outcomes[policy][split][metric],
                    resamples=n_resamples,
                    seed=bootstrap_seed,
                )
            greedy = policy_outcomes[policy][split]["greedy_pass_at_1"]
            search = policy_outcomes[policy][split]["search_at_9"]
            report["per_seed"][str(seed)]["vs_greedy"] = report["per_seed"][str(seed)].get("vs_greedy", {})
            report["per_seed"][str(seed)]["vs_greedy"][split] = _paired_bootstrap(
                greedy, search, resamples=n_resamples, seed=bootstrap_seed
            )
        report["source_solvable_only"]["per_seed_vs_base"][str(seed)] = {
            metric: _paired_bootstrap(
                source_solvable_outcomes["base"][metric],
                source_solvable_outcomes[policy][metric],
                resamples=n_resamples,
                seed=bootstrap_seed,
            )
            for metric in METRICS
        }

    mean_sft = {
        split: {
            metric: _mean_maps([policy_outcomes[f"sft_{seed}"][split][metric] for seed in (42, 43, 44)])
            for metric in METRICS
        }
        for split in SPLITS
    }
    for split in SPLITS:
        report["three_seed_mean"][split] = {
            "vs_base": {
                metric: _paired_bootstrap(
                    policy_outcomes["base"][split][metric], mean_sft[split][metric],
                    resamples=n_resamples, seed=bootstrap_seed,
                ) for metric in METRICS
            },
            "search_at_9_vs_greedy": _paired_bootstrap(
                mean_sft[split]["greedy_pass_at_1"], mean_sft[split]["search_at_9"],
                resamples=n_resamples, seed=bootstrap_seed,
            ),
        }

    selection_rows = []
    for policy in eval_names:
        for task_id, group in sorted(task_groups[policy].items()):
            selection = _select_candidate(group)
            selection_rows.append({
                "experiment_id": freeze["experiment_id"],
                "policy": policy,
                "task_id": task_id,
                "split": group["split"],
                "target": group["target"],
                "nums": group["nums"],
                "oracle_solvable": group["oracle_solvable"],
                **selection,
            })
        for split in SPLITS:
            groups = {key: group for key, group in task_groups[policy].items() if group["split"] == split}
            failure_counts = Counter(
                row["failure_category"]
                for group in groups.values()
                for row in [group["greedy"], *group["sampled"]]
            )
            report["search_selection"].setdefault(policy, {})[split] = {
                "tasks": len(groups),
                "oracle_solvable_tasks": sum(group["oracle_solvable"] for group in groups.values()),
                "selected_exact_solutions": sum(_select_candidate(group)["reward"] == 1.0 for group in groups.values()),
                "candidate_failure_categories": dict(sorted(failure_counts.items())),
                "completion_truncations": sum(
                    int(row["truncation"])
                    for group in groups.values()
                    for row in [group["greedy"], *group["sampled"]]
                ),
                "mean_completion_length": sum(
                    int(row["completion_length"])
                    for group in groups.values()
                    for row in [group["greedy"], *group["sampled"]]
                ) / (len(groups) * 9),
            }
    write_jsonl(output_dir / "selected-candidates.jsonl", selection_rows)

    per_seed_positive = all(
        report["per_seed"][str(seed)]["vs_greedy"][split]["paired_gain"] > 0
        for seed in (42, 43, 44)
        for split in SPLITS
    )
    aggregate_ci_positive = all(
        report["three_seed_mean"][split]["search_at_9_vs_greedy"]["paired_gain_percentile_95"][0] > 0
        for split in SPLITS
    )
    report["decision"] = {
        "all_three_sft_seeds_improve_search_at_9_over_greedy_on_both_suites": per_seed_positive,
        "three_seed_paired_intervals_positive_on_both_suites": aggregate_ci_positive,
        "positive_criterion_met": per_seed_positive and aggregate_ci_positive,
    }
    (output_dir / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Exact-verifier-filtered test-time search",
        "",
        "Generated from the frozen design, confirmation manifest, and saved raw evaluator records. This is an inference-time search result, not a training or learning claim.",
        "",
        f"Recorded GPU evaluator time: {report['compute']['evaluation_process_hours']:.3f} hours of the {report['compute']['gpu_cap_hours']:.1f}-hour cap across {report['compute']['evaluation_count']} models.",
        "",
        "## Exact solve rates",
        "",
        "Rates are task-level. `sample pass@k` uses the first k sampled records only. `search@k` considers greedy first, then samples in sample-index order, returning the first candidate accepted by the exact verifier.",
        "",
        "| Suite | Policy | Greedy pass@1 | Sample pass@4 | Sample pass@8 | Search@5 | Search@9 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        for policy in eval_names:
            groups = [g for g in policy_outcomes[policy][split]["greedy_pass_at_1"]]
            rows = [
                f"| {split} | {policy} | " + " | ".join(
                    f"{sum(policy_outcomes[policy][split][metric].values()) / len(groups):.2%}"
                    for metric in ("greedy_pass_at_1", "sample_pass_at_4", "sample_pass_at_8", "search_at_5", "search_at_9")
                ) + " |"
            ]
            lines.extend(rows)
    lines.extend([
        "",
        f"Source-solvable-only secondary analysis: {report['source_solvable_only']['task_count']} of 256 source tasks were oracle-solvable.",
        "",
        "| Policy | Greedy pass@1 | Sample pass@4 | Sample pass@8 | Search@5 | Search@9 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for policy in eval_names:
        lines.append(
            f"| {policy} | " + " | ".join(
                f"{report['source_solvable_only']['policies'][policy][metric]['rate']:.2%}"
                for metric in METRICS
            ) + " |"
        )
    lines.extend(["", "## Paired search@9 gains for each SFT seed", "", "| Seed | Suite | SFT greedy | SFT search@9 | Gain (95% paired task-bootstrap interval) | SFT search@9 vs base |", "|---:|---|---:|---:|---|---|"])
    for seed in (42, 43, 44):
        for split in SPLITS:
            gain = report["per_seed"][str(seed)]["vs_greedy"][split]
            vs_base = report["per_seed"][str(seed)]["vs_base"][split]["search_at_9"]
            lines.append(
                f"| {seed} | {split} | {gain['baseline_rate']:.2%} | {gain['trained_rate']:.2%} | "
                f"{gain['paired_gain']:+.2%} [{gain['paired_gain_percentile_95'][0]:+.2%}, {gain['paired_gain_percentile_95'][1]:+.2%}] | "
                f"{vs_base['paired_gain']:+.2%} [{vs_base['paired_gain_percentile_95'][0]:+.2%}, {vs_base['paired_gain_percentile_95'][1]:+.2%}] |"
            )
    lines.extend(["", "## Three-seed decision", ""])
    if report["decision"]["positive_criterion_met"]:
        conclusion = "The predeclared bounded inference-time criterion is met: all three SFT policies improved exact solve rate from greedy to nine ordered candidates on both suites, and the paired three-seed task-bootstrap intervals are positive on both. This does not show that GRPO learned or that general reasoning emerged."
    else:
        conclusion = "The predeclared positive criterion is not met. Report the per-seed and paired intervals above as negative, mixed, or inconclusive; do not treat a single-seed gain or the mathematical monotonicity of a larger candidate set as evidence of learned reasoning."
    lines.extend([conclusion, "", "## Selected candidates and failures", "", "`selected-candidates.jsonl` records the first exact verifier-approved candidate or an explicit no-candidate result for every policy/task pair. Full raw proposals and per-completion failure categories remain in the four evaluator JSONLs.", ""])
    lines.append("| Policy | Suite | Tasks | Oracle-solvable | Selected exact solutions | Truncations | Mean completion tokens |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for policy in eval_names:
        for split in SPLITS:
            entry = report["search_selection"][policy][split]
            lines.append(f"| {policy} | {split} | {entry['tasks']} | {entry['oracle_solvable_tasks']} | {entry['selected_exact_solutions']} | {entry['completion_truncations']} | {entry['mean_completion_length']:.2f} |")
    lines.extend(["", "![Greedy and verifier-search solve rates](comparison.svg)", "", "## Reproduction", "", "Exact commands, dependency versions, device observations, raw completions, and runtime are saved in each adjacent `*.summary.json` and `environment.json`. The frozen settings are in `frozen-design.json` and the leakage/solvability audit is in `confirmation-manifest.json`."])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_rates: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        for policy in eval_names:
            outcomes = policy_outcomes[policy][split]
            # One SVG shows both suites by adding an independent row per suite/policy.
            plot_rates[f"{split}/{policy}"] = {
                "greedy_pass_at_1": sum(outcomes["greedy_pass_at_1"].values()) / len(outcomes["greedy_pass_at_1"]),
                "search_at_9": sum(outcomes["search_at_9"].values()) / len(outcomes["search_at_9"]),
            }
    _svg(output_dir / "comparison.svg", plot_rates)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sft", type=Path, nargs=3, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate_report(
        baseline_path=args.baseline,
        sft_paths=args.sft,
        freeze_path=args.freeze,
        manifest_path=args.manifest,
        environment_path=args.environment,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover - exercised as a CLI in the experiment
    main()

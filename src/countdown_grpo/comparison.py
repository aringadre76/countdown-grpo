"""Paired task-level comparisons; sampled completions are not independent tasks."""

from collections import defaultdict
from random import Random


def task_outcomes(records, mode):
    groups = defaultdict(list)
    for record in records:
        if record["generation_mode"] == mode:
            groups[record["task_id"]].append(record)
    outcomes = {}
    for key, rows in groups.items():
        expected = 1 if mode == "greedy" else 4
        if len(rows) != expected:
            raise ValueError(f"{key}: expected {expected} {mode} completions")
        if len({(row["target"], tuple(sorted(row["nums"]))) for row in rows}) != 1:
            raise ValueError(f"inconsistent task identity: {key}")
        outcomes[key] = int(any(row["reward"] == 1.0 for row in rows))
    if not outcomes:
        raise ValueError(f"no {mode} tasks")
    return outcomes


def paired_bootstrap(baseline, trained, *, resamples=10000, seed=20260918):
    if not baseline or set(baseline) != set(trained):
        raise ValueError("paired comparison requires identical nonempty task IDs")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    differences = [trained[key] - baseline[key] for key in sorted(baseline)]
    rng = Random(seed)
    size = len(differences)
    draws = sorted(sum(rng.choices(differences, k=size)) / size for _ in range(resamples))
    return {
        "tasks": size,
        "baseline_rate": sum(baseline.values()) / size,
        "trained_rate": sum(trained.values()) / size,
        "paired_gain": sum(differences) / size,
        "paired_gain_percentile_95": [draws[int(0.025 * (resamples - 1))],
                                      draws[int(0.975 * (resamples - 1))]],
        "bootstrap_resamples": resamples, "bootstrap_seed": seed,
    }

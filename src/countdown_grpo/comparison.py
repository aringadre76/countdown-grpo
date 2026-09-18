"""Paired task-level comparisons; sampled completions are not independent tasks."""

import re
from collections import defaultdict
from random import Random

from .verifier import extract_expression, verify_completion


def normalized_record(record):
    """Diagnostic equality-suffix removal only; primary scoring is untouched."""
    expression = extract_expression(record["raw_completion"])
    normalized = re.sub(r"\s*=\s*-?\d+\s*$", "", expression) if expression is not None else None
    result = verify_completion(normalized or "", record["target"], record["nums"],
                               truncated=bool(record["truncation"]))
    return {**record, "reward": float(result.valid), "normalized_expression": normalized,
            "normalized_failure_category": result.reason}


def audit_pairs(baseline, trained, *, limit=20):
    """Fixed-ID greedy disagreements, including losses, without cherry-picking."""
    base = {row["task_id"]: row for row in baseline if row["generation_mode"] == "greedy"}
    adapted = {row["task_id"]: row for row in trained if row["generation_mode"] == "greedy"}
    if set(base) != set(adapted):
        raise ValueError("audit requires identical task IDs")
    audit = []
    for key in sorted(base):
        left, right = base[key], adapted[key]
        if left["reward"] == right["reward"]:
            continue
        if (left["target"], sorted(left["nums"])) != (right["target"], sorted(right["nums"])):
            raise ValueError("audit task identity mismatch")
        normalized_left, normalized_right = normalized_record(left), normalized_record(right)
        if right["reward"] < left["reward"]:
            category = "lost_primary_solution"
        elif normalized_left["reward"] == 1:
            category = "format_only_repair_sufficient"
        elif normalized_left["normalized_failure_category"] == "wrong_target":
            category = "changed_legal_arithmetic_reaches_target"
        else:
            category = "constraint_or_search_ambiguous"
        audit.append({"task_id": key, "target": left["target"], "nums": left["nums"],
                      "split": left["split"], "rubric_category": category,
                      "baseline": normalized_left, "trained": normalized_right})
        if len(audit) == limit:
            break
    return audit


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

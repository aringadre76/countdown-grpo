"""Generate evidence-only Markdown and SVG summaries from saved experiment files."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentage(value: float | None) -> str:
    return "not measured" if value is None else f"{value * 100:.2f}%"


def _summary_rows(summary: dict[str, Any]) -> list[tuple[str, float | None]]:
    rows = [("overall", summary.get("exact_solve_rate"))]
    rows.extend((f"split: {name}", values.get("exact_solve_rate")) for name, values in summary.get("by_split", {}).items())
    return rows


def write_svg(path: Path, series: list[tuple[str, float | None]]) -> None:
    """Write a compact dependency-free bar chart from observed proportions."""

    width, height, left, bar_height, gap = 760, 70 + 42 * len(series), 220, 24, 18
    bars: list[str] = []
    for index, (label, value) in enumerate(series):
        y = 42 + index * (bar_height + gap)
        numeric = 0.0 if value is None else max(0.0, min(1.0, value))
        bars.extend(
            [
                f'<text x="10" y="{y + 17}" font-size="14">{html.escape(label)}</text>',
                f'<rect x="{left}" y="{y}" width="480" height="{bar_height}" fill="#e5e7eb"/>',
                f'<rect x="{left}" y="{y}" width="{480 * numeric:.2f}" height="{bar_height}" fill="#2563eb"/>',
                f'<text x="710" y="{y + 17}" text-anchor="end" font-size="14">{_percentage(value)}</text>',
            ]
        )
    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<text x="10" y="22" font-size="18" font-weight="bold">Saved-evidence exact solve rates</text>',
            *bars,
            "</svg>",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg + "\n", encoding="utf-8")


def generate_report(
    *,
    audit_path: Path,
    manifest_path: Path,
    environment_path: Path,
    baseline_path: Path,
    smoke_attempt_path: Path,
    smoke_diagnostics_path: Path,
    markdown_output: Path,
    svg_output: Path,
    adapter_path: Path | None = None,
) -> None:
    audit = _load(audit_path)
    manifest = _load(manifest_path)
    environment = _load(environment_path)
    baseline = _load(baseline_path)
    smoke_attempt = _load(smoke_attempt_path)
    smoke_diagnostic = json.loads(smoke_diagnostics_path.read_text(encoding="utf-8").splitlines()[-1])
    adapter = _load(adapter_path) if adapter_path is not None else None

    lines = [
        "# Countdown GRPO evidence report",
        "",
        "This report is generated from the saved JSON artifacts named below. It contains no inferred metrics.",
        "",
        "## Dataset and split audit",
        "",
        f"- Source revision: `{audit['dataset_revision']}`",
        f"- Raw rows: {audit['raw_row_count']:,}; canonical tasks: {audit['canonical_task_count']:,}",
        f"- Canonical duplicate rows removed: {audit['canonical_duplicate_row_count']:,}",
        f"- Oracle solvability under the locked contract: {_percentage(audit['oracle_solvability_rate'])}",
        f"- Frozen split sizes: train {manifest['splits']['train']['task_count']:,}, dev {manifest['splits']['dev']['task_count']:,}, test {manifest['splits']['test']['task_count']:,}; fresh test {manifest['fresh_test_count']:,}.",
        "",
        "## Observed environment",
        "",
        f"- Python {environment['python']}; Torch {environment['torch']['version']}; CUDA available: {environment['torch']['cuda_available']}.",
        f"- ROCm probe: {environment['gpu_probes'][0]['stderr'] or environment['gpu_probes'][0]['stdout'] or 'no output'}",
        "",
        "## Untouched-base baseline",
        "",
        f"- Records: {baseline['record_count']}; exact solve rate: {_percentage(baseline['exact_solve_rate'])}; legal expression rate: {_percentage(baseline['legal_expression_rate'])}.",
        f"- Source-held-out: {_percentage(baseline['by_split'].get('test', {}).get('exact_solve_rate'))}; fresh-task: {_percentage(baseline['by_split'].get('fresh_test', {}).get('exact_solve_rate'))}.",
        f"- Failure categories: `{json.dumps(baseline['failure_categories'], sort_keys=True)}`",
        "",
        "## GRPO smoke",
        "",
        f"- Status: {smoke_attempt['status']}; optimizer steps: {smoke_attempt.get('global_step', 'not completed')}; training loss: {smoke_attempt.get('training_loss', 'not completed')}.",
        f"- Mean reward: {smoke_diagnostic['mean_reward']}; reward variance: {smoke_diagnostic['reward_variance']}; mixed-group rate: {smoke_diagnostic['mixed_reward_group_rate']}; all-zero-group rate: {smoke_diagnostic['all_zero_group_rate']}.",
        f"- Failure categories: `{json.dumps(smoke_diagnostic['failure_categories'], sort_keys=True)}`",
    ]
    if adapter is not None:
        lines.extend(
            [
                "",
                "## Smoke-adapter comparison",
                "",
                f"- Records: {adapter['record_count']}; exact solve rate: {_percentage(adapter['exact_solve_rate'])}; legal expression rate: {_percentage(adapter['legal_expression_rate'])}.",
                "- This one-step all-zero-reward smoke is an integration comparison, not a learning result.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The saved smoke diagnostics show no mixed reward groups. Therefore the CPU smoke verifies the end-to-end GRPO plumbing but does not establish a useful group-relative learning signal or improved Countdown solving.",
            "",
            "## Inputs",
            "",
            f"- `{audit_path}`",
            f"- `{manifest_path}`",
            f"- `{environment_path}`",
            f"- `{baseline_path}`",
            f"- `{smoke_attempt_path}`",
            f"- `{smoke_diagnostics_path}`",
        ]
    )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    series = [(f"base {name}", value) for name, value in _summary_rows(baseline)]
    if adapter is not None:
        series.extend((f"adapter {name}", value) for name, value in _summary_rows(adapter))
    write_svg(svg_output, series)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--smoke-attempt", type=Path, required=True)
    parser.add_argument("--smoke-diagnostics", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    args = parser.parse_args()
    generate_report(
        audit_path=args.audit,
        manifest_path=args.manifest,
        environment_path=args.environment,
        baseline_path=args.baseline,
        smoke_attempt_path=args.smoke_attempt,
        smoke_diagnostics_path=args.smoke_diagnostics,
        markdown_output=args.markdown_output,
        svg_output=args.svg_output,
        adapter_path=args.adapter,
    )


if __name__ == "__main__":
    main()

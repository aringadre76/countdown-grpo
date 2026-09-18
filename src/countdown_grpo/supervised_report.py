"""Generate a supervised dev-gate report from saved measurements."""

import argparse
import json
import math
from pathlib import Path

from .data import read_jsonl


def diagnostic_gate(metrics, summary):
    losses = [row["loss"] for row in metrics if "loss" in row]
    finite_gradients = all(math.isfinite(row["grad_norm"])
                           for row in metrics if "grad_norm" in row)
    return {
        "loss_decreased": len(losses) >= 2 and losses[-1] < losses[0],
        "finite_logged_gradients": finite_gradients and any("grad_norm" in row for row in metrics),
        "dev_legality_at_least_20_percent": summary["legal_expression_rate"] >= 0.2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.baseline.read_text())
    adapter = json.loads(args.adapter.read_text())
    metrics = read_jsonl(args.metrics)
    gate = diagnostic_gate(metrics, adapter)
    gate["passed"] = all(gate.values())
    args.output_prefix.with_suffix(".gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    rows = []
    for label, summary in [("Untouched base", base), ("Supervised diagnostic", adapter)]:
        modes = summary["pass_at_k_by_generation_mode"]
        rows.append(f"| {label} | {summary['legal_expression_rate']:.2%} | "
                    f"{summary['exact_solve_rate']:.2%} | {modes['greedy']:.2%} | {modes['sample']:.2%} |")
    text = "\n".join([
        "# Supervised diagnostic — dev only", "",
        "This named supervised branch is not the original no-SFT GRPO experiment.",
        "The same 16 source-dev tasks have one greedy and four sampled completions each.", "",
        "| Model | Legal completions | Exact completions | Greedy task accuracy | Sampled pass@4 |",
        "|---|---:|---:|---:|---:|", *rows, "",
        f"Predeclared diagnostic gate passed: {gate['passed']}.",
        "These are method-selection measurements, not confirmation estimates.",
        "Fresh-task generalization, seed replication, and search gains remain untested.", "",
        "## Evidence", "", *(f"- `{path}`" for path in [args.baseline, args.adapter, args.metrics]), "",
    ])
    args.output_prefix.with_suffix(".md").write_text(text)
    losses = [(row["step"], row["loss"]) for row in metrics if "loss" in row]
    maximum_step = max(step for step, _ in losses)
    maximum_loss = max(loss for _, loss in losses)
    points = " ".join(f"{50 + 500 * step / maximum_step:.1f},{250 - 200 * loss / maximum_loss:.1f}"
                      for step, loss in losses)
    args.output_prefix.with_suffix(".svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="300" viewBox="0 0 600 300">'
        '<rect width="600" height="300" fill="white"/>'
        '<text x="50" y="24" font-family="sans-serif">Supervised diagnostic: logged training loss</text>'
        '<path d="M50 45 V250 H560" fill="none" stroke="#555"/>'
        f'<polyline points="{points}" fill="none" stroke="#2764b5" stroke-width="3"/>'
        f'<text x="50" y="280" font-family="sans-serif">Step 0 to {maximum_step}; loss 0 to {maximum_loss:.3f}</text>'
        '</svg>\n'
    )


if __name__ == "__main__":
    main()

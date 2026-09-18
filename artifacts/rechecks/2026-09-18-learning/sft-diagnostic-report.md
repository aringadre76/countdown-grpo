# Supervised diagnostic — dev only

This named supervised branch is not the original no-SFT GRPO experiment.
The same 16 source-dev tasks have one greedy and four sampled completions each.

| Model | Legal completions | Exact completions | Greedy task accuracy | Sampled pass@4 |
|---|---:|---:|---:|---:|
| Untouched base | 3.75% | 0.00% | 0.00% | 0.00% |
| Supervised diagnostic | 63.75% | 8.75% | 12.50% | 25.00% |

Predeclared diagnostic gate passed: True.
These are method-selection measurements, not confirmation estimates.
Fresh-task generalization, seed replication, and search gains remain untested.

## Evidence

- `artifacts/rechecks/2026-09-18-learning/base-sft-comparison-dev.summary.json`
- `artifacts/rechecks/2026-09-18-learning/sft-diagnostic-dev.summary.json`
- `artifacts/rechecks/2026-09-18-learning/sft-diagnostic-s42/metrics.jsonl`

# SFT-initialized binary-GRPO confirmation

Generated only from the frozen design, saved evaluations, final checkpoints, and per-step GRPO diagnostics.
The outcome reward and exact Countdown verifier are unchanged. Reward changes alone are not treated as learning evidence.

## Three-seed paired comparisons

| Contrast | Suite | Metric | Initialization | Trained mean | Paired gain | 95% task-bootstrap interval |
|---|---|---|---:|---:|---:|---|
| base_vs_sft | fresh_confirmation | pass@1 | 0.39% | 7.81% | 7.42% | [4.56%, 10.55%] |
| base_vs_sft | fresh_confirmation | pass@4 | 0.00% | 16.80% | 16.80% | [12.89%, 20.70%] |
| base_vs_sft | source_confirmation | pass@1 | 0.00% | 21.48% | 21.48% | [16.93%, 26.17%] |
| base_vs_sft | source_confirmation | pass@4 | 0.39% | 31.64% | 31.25% | [26.17%, 36.33%] |
| base_vs_grpo | fresh_confirmation | pass@1 | 0.39% | 7.55% | 7.16% | [4.30%, 10.29%] |
| base_vs_grpo | fresh_confirmation | pass@4 | 0.00% | 17.06% | 17.06% | [13.15%, 21.09%] |
| base_vs_grpo | source_confirmation | pass@1 | 0.00% | 21.88% | 21.88% | [17.19%, 26.56%] |
| base_vs_grpo | source_confirmation | pass@4 | 0.39% | 32.94% | 32.55% | [27.47%, 37.63%] |
| sft_vs_grpo | fresh_confirmation | pass@1 | 7.81% | 7.55% | -0.26% | [-0.78%, 0.26%] |
| sft_vs_grpo | fresh_confirmation | pass@4 | 16.80% | 17.06% | 0.26% | [-1.04%, 1.69%] |
| sft_vs_grpo | source_confirmation | pass@1 | 21.48% | 21.88% | 0.39% | [-0.65%, 1.56%] |
| sft_vs_grpo | source_confirmation | pass@4 | 31.64% | 32.94% | 1.30% | [-0.39%, 2.99%] |

Each per-seed result and task bootstrap is in `comparison.json`. Three-seed intervals resample paired tasks while averaging these three observed training seeds; they are not uncertainty over the full population of possible seeds.
The fixed search audit compares matching SFT and GRPO greedy completions, includes gains and losses, and is stratified to at most 10 source and 10 fresh tasks per seed.
Equality-suffix normalization and source-solvable-only comparisons are reported in the JSON.

### Per-seed greedy comparison (SFT → GRPO)

| Seed | Source SFT | Source GRPO | Source paired gain (95% CI) | Fresh SFT | Fresh GRPO | Fresh paired gain (95% CI) |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 21.88% | 23.44% | 1.56% [0.39%, 3.12%] | 7.81% | 7.81% | 0.00% [0.00%, 0.00%] |
| 43 | 20.70% | 21.09% | 0.39% [-1.17%, 1.95%] | 7.03% | 6.64% | -0.39% [-1.95%, 0.78%] |
| 44 | 21.88% | 21.09% | -0.78% [-3.12%, 1.56%] | 8.59% | 8.20% | -0.39% [-1.17%, 0.00%] |

### Trace-audit counts

| Seed | Audit categories |
|---:|---|
| 42 | `changed_legal_arithmetic_reaches_target`: 4 |
| 43 | `changed_legal_arithmetic_reaches_target`: 4, `lost_primary_solution`: 4 |
| 44 | `changed_legal_arithmetic_reaches_target`: 4, `lost_primary_solution`: 7 |

## Conclusion

The predeclared positive-result criterion is not met. Across seeds, source greedy pass@1 changed by +0.39% (95% task-bootstrap interval -0.65% to +1.56%), and fresh greedy pass@1 changed by -0.26% (95% interval -0.78% to +0.26%). Neither interval establishes a positive gain on both suites, and the per-seed changes are not consistently positive. This is no demonstrated improvement from this 50-step GRPO follow-up over its SFT initialization; it does not show that GRPO cannot help under other designs.

## Compute and observed environment

Recorded follow-up GPU process time: 5.064 hours of the 6.0-hour cap (0.936 hours unused). This sums 7 confirmation evaluators, 3 signal probes, the integration attempt, and three GRPO process lifetimes from saved summaries/attempt timestamps.
All recorded evaluations used `cuda` on AMD Radeon RX 7900 XTX (`gfx1100`); observed stack: Torch 2.13.0+rocm7.14.0, HIP 7.14.60850, Transformers 5.16.1, TRL 1.12.0, PEFT 0.20.0. See `environment.json` for the full WSL manifest and `docs/hardware.md` for the `rocm-smi` caveat.

## GRPO training diagnostics

| Seed | Steps | Rollouts | Mean reward | Positive rate | Mixed groups | All-zero groups | All-one groups | Truncation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 50 | 400 | 0.133 | 0.133 | 0.300 | 0.690 | 0.010 | 0.000 |
| 43 | 50 | 400 | 0.083 | 0.083 | 0.210 | 0.780 | 0.010 | 0.000 |
| 44 | 50 | 400 | 0.135 | 0.135 | 0.360 | 0.640 | 0.000 | 0.000 |

Positive/negative conclusion language must follow the paired exact-task comparisons, not training reward alone.

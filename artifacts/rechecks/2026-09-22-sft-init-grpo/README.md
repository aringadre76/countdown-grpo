# SFT-initialized binary-GRPO follow-up

Status: completed on 2026-09-23. This is a separately labeled continuation of
the three-seed supervised study. It does not change the historical no-SFT,
base-initialized binary-GRPO result.

## Frozen question and contract

Does 50 steps of TRL GRPO add held-out Countdown solving ability beyond each
matching SFT checkpoint? The frozen model is
`Qwen/Qwen3.5-0.8B-Base`, revision
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. See
[`frozen-design.json`](frozen-design.json) and
[`../../../../docs/amendment-2026-09-22-sft-init-grpo.md`](../../../../docs/amendment-2026-09-22-sft-init-grpo.md)
for the immutable method, seed eligibility gate, task sampling, and budget.

Reward and evaluation use the same exact contract: binary `+`, `-`, `*`, `/`
with parentheses; every supplied number exactly once; exact rational checks;
integer intermediate and final values; reward `1.0` only for a legal exact
target solution, otherwise `0.0`. Oracle witnesses never enter prompts, and
generated text is never executed.

## Conclusion

All three SFT adapters passed their train-only signal gate. Each 50-step GRPO
run recorded 400 completions, zero truncations, and mixed reward groups:

| Seed | Mean reward / positive rate | Mixed groups | All-zero groups |
|---:|---:|---:|---:|
| 42 | 0.133 | 30% | 69% |
| 43 | 0.083 | 21% | 78% |
| 44 | 0.135 | 36% | 64% |

The confirmation suite contains 256 oracle-solvable source tasks and 256
independently generated oracle-solvable fresh tasks. Base, all three matching
SFT adapters, and all three final GRPO adapters were evaluated on those same
tasks with one greedy and four sampled completions per task. Each evaluator
saved 2,560 records and reported zero truncation for the six adapter runs.

| Seed | Source SFT → GRPO pass@1 | Fresh SFT → GRPO pass@1 |
|---:|---:|---:|
| 42 | 21.88% → 23.44% (+1.56 pp) | 7.81% → 7.81% (0.00 pp) |
| 43 | 20.70% → 21.09% (+0.39 pp) | 7.03% → 6.64% (−0.39 pp) |
| 44 | 21.88% → 21.09% (−0.78 pp) | 8.59% → 8.20% (−0.39 pp) |
| Mean | 21.48% → 21.88% (+0.39 pp) | 7.81% → 7.55% (−0.26 pp) |

Three-seed paired task-bootstrap intervals (10,000 resamples, seed 20260922)
were −0.65 to +1.56 percentage points for source greedy gain and −0.78 to
+0.26 points for fresh greedy gain. Sampled pass@4 gains were +1.30 points on
source (95% interval −0.39 to +2.99) and +0.26 on fresh (−1.04 to +1.69).
The positive-result criterion was not met: the intervals do not establish
improvement on both suites, and the seed directions are not consistent. The
appropriate conclusion is no demonstrated benefit from this 50-step GRPO
follow-up over SFT—not that GRPO cannot help in general.

The fixed search audit found four changed legal target-reaching completions
per seed; it also found four lost primary solutions for seed 43 and seven for
seed 44. The audit traces are illustrative evidence only and do not override
the paired task-level result.

## Compute and hardware

Recorded follow-up process time was 18,229.856 seconds (5.064 hours) out of
the frozen six-hour cap. It sums seven GPU confirmation evaluators, three
train-only signal probes, the seed-42 integration attempt, and three 50-step
GRPO process lifetimes. The saved report has this accounting in `comparison.json`.

Observed device: AMD Radeon RX 7900 XTX (`gfx1100`) via WSL2/ROCm. Saved
training configs record Torch `2.13.0+rocm7.14.0`, HIP `7.14.60850`,
Transformers `5.16.1`, TRL `1.12.0`, and PEFT `0.20.0`. Evaluator records use
`device: cuda`. `environment.json` is the observed WSL environment snapshot;
`rocm-smi` reports its known WSL driver-initialization caveat while Torch and
`rocminfo` identify the device.

## Evidence map and reproduction

- `signal-s*-t1.jsonl`, `.summary.json`, and `.gate.json`: full train-only gate.
- `integration-s42/`: separate one-step warm-start integration evidence.
- `grpo-s*/`: exact commands, run configs, step-level raw completions and
  reward-group/gradient diagnostics, and final adapter hashes.
- `base-confirmation.jsonl` and summary: untouched base on the frozen suite.
- `sft-s*-confirmation.jsonl` and summaries: matching SFT checkpoints.
- `grpo-s*-confirmation.jsonl` and summaries: final 50-step GRPO checkpoints.
- `confirmation-manifest.json` and `confirmation/`: frozen source/fresh tasks,
  task hashes, solvability, and exclusion checks.
- [`report-final/`](report-final/): generated paired
  comparisons, bootstrap intervals, fixed-order audit traces, compute
  accounting, and SVG.
- `attempts.jsonl`: exact failed preparation attempts and their outputs; these
  were repaired before training and do not replace or alter the frozen design.

Every summary JSON records the exact command argv, model revision, start/end
timestamps, runtime, record/token counts, failure categories, and device
memory. Re-run the analysis with a fresh output directory using the command in
[`docs/reproduction.md`](../../../../docs/reproduction.md#11-frozen-sft-initialized-binary-grpo-follow-up).
Do not overwrite these saved raw records or extend the cap; a longer training
study needs a new dated amendment and new frozen confirmation tasks.

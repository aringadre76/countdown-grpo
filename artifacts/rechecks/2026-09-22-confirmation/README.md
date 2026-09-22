# Frozen three-seed confirmation — 2026-09-22

This artifact set completes the supervised-initialization extension authorized
by [the dated amendment](../../../../docs/amendment-2026-09-18.md). It is separate
from the historical no-SFT/binary-reward GRPO experiment. The frozen design and
base results remain under `artifacts/rechecks/2026-09-18-learning/`.

## Frozen design

- Model: `Qwen/Qwen3.5-0.8B-Base`, revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- Initialization: each seed starts from the same untouched base checkpoint.
- Supervised data: 4,096 verified solutions from source-train only; solutions
  are targets, not prompt hints. See the data file and manifest in
  `../2026-09-18-learning/data/supervised-train.jsonl` and
  `../2026-09-18-learning/supervised-data-manifest.json`.
- Training: completion-only LoRA, 512 steps, seeds 42, 43, and 44; rank 16,
  alpha 32, dropout 0.05, batch 1, accumulation 8, learning rate `2e-4`,
  linear schedule, maximum sequence length 256, bf16, eager attention.
- Checkpoint choice: the predeclared final-step adapter for each seed.
- Evaluation: 256 unseen source tasks and 256 independently generated fresh
  solvable tasks; raw prompts; one greedy plus four sampled completions; 128
  new tokens; temperature 1.0; top-p 0.95; generation seed 42 for paired
  task/sample randomization; eager attention on `cuda:0`.
- Arithmetic: binary `+`, `-`, `*`, `/`, parentheses, every number exactly
  once, exact rational checks, integer intermediate and final values.

The confirmation manifest records canonical task identities and hashes. It
excludes prior train/dev/test and historical evaluation tasks, plus overlap
between source and fresh suites. The source confirmation suite is 256/256
oracle-solvable; every fresh task is also oracle-solvable. No oracle witness is
present in evaluation prompts.

## Results

Each evaluation contains 2,560 scored records: 1,280 per suite. All three runs
had zero truncation. These are task-level pass rates; repeated sampled
completions are grouped within their task.

| Checkpoint | Source greedy | Source sampled pass@4 | Fresh greedy | Fresh sampled pass@4 | Legal-expression rate |
|---|---:|---:|---:|---:|---:|
| Untouched base | 0.00% | 1.17% | 0.39% | 0.39% | 3.71% |
| Seed 42 | 17.58% | 34.77% | 9.38% | 14.84% | 79.49% |
| Seed 43 | 21.09% | 35.94% | 7.42% | 15.23% | 82.77% |
| Seed 44 | 16.80% | 35.94% | 8.59% | 16.80% | 82.97% |
| Mean of seeds | 18.49% | 35.55% | 8.46% | 15.62% | 81.74% |

The paired task bootstrap uses 10,000 resamples and seed 20260918. For the
three-seed mean, greedy paired gain was +18.49 percentage points on source
(95% interval +14.32 to +22.92) and +8.07 points on fresh (+5.08 to +11.20).
Sampled pass@4 paired gain was +34.37 points on source (+28.91 to +39.97) and
+15.23 on fresh (+11.33 to +19.40). Intervals describe task uncertainty
conditional on these three runs, not uncertainty over the training-seed
population.

After removing only terminal `= integer` suffixes from both sides and applying
the same verifier, greedy gain remained +16.93 points on source (95% interval
+12.37 to +21.48) and +6.51 on fresh (+3.26 to +9.77). This rules out that
specific suffix repair as a sufficient explanation; it does not remove every
possible formatting effect.

The frozen fixed-order trace audit sampled up to 20 greedy disagreements per
seed across the two suites. Its 60 seed-task comparisons were classified as 15
changed legal wrong-target expressions becoming legal exact solutions, 3 cases
where a simple formatting repair was sufficient, 39 ambiguous constraint/search
changes, and 3 lost solutions. Task IDs repeat across seeds. The lexical audit
order selected only fresh-task disagreements, so these traces do not directly
sample source-suite disagreements. The normalized paired results cover both
suites. See `report-final/seed-42-audit.jsonl` through `seed-44-audit.jsonl`.

## Compute and environment

The three 512-step training runtimes sum to 5,543.927 seconds. The untouched
base and three adapter confirmation evaluator runtimes sum to 9,244.008
seconds. Together, those primary training and full-confirmation jobs used
14,787.935 seconds (4.11 hours) of recorded runtime. This subtotal excludes
the earlier integration, diagnostic, dev, and control probes.

`environment.json` was captured after evaluation. It records WSL2 kernel
`6.18.33.2-microsoft-standard-WSL2`, Python 3.11.15, Torch
`2.13.0+rocm7.14.0`, HIP `7.14.60850`, AMD Radeon RX 7900 XTX (`gfx1100`),
and 25,708,240,896 bytes total device memory. The manifest ran
`sum(arange(1024) ** 2)` on `cuda:0` and recorded the result. `rocminfo` listed
the GPU; `rocm-smi` reported `Driver not initialized (amdgpu not found in
modules)` under WSL. Torch and the completed evaluations are the direct
training/inference evidence. No `llama.*server` process was present at capture.

The evaluator recorded peak allocated memory of 1,892,670,976 bytes and peak
reserved memory of 1,981,808,640 bytes for each adapter run. The base peak
allocated memory was 1,851,776,512 bytes. The pinned device and software
versions are in the environment manifest.

## Evidence files

- `sft-confirmation-s42.jsonl`, `sft-confirmation-s43.jsonl`,
  `sft-confirmation-s44.jsonl`: all raw scored generations.
- Matching `*.summary.json` files: commands, timestamps, metrics, hashes,
  token totals, peak memory, and evaluator wall times.
- `environment.json`: observed WSL, Python, packages, HSA/Torch devices, and
  device-memory snapshot.
- `report-final/comparison.json`: per-seed and three-seed paired bootstrap
  results, normalized comparisons, source-solvable breakdowns, and input hashes.
- `report-final/report.md`: generated tables and conservative interpretation.
- `report-final/comparison.svg`: generated plot of source/fresh pass@1 and
  pass@4 from the saved comparison JSON.
- `report-final/seed-*-audit.jsonl`: fixed-order raw trace comparisons.

Regenerate the report from the immutable baseline and these three JSONL files
with the command in [`docs/reproduction.md`](../../../../docs/reproduction.md).

# Verifier-filtered test-time search — 2026-09-23

Status: frozen before task preparation and confirmation evaluation. This is a
separately named inference-time experiment; it adds no training and does not
change the historical base-GRPO or SFT-initialized GRPO conclusions.

The machine-readable design is
[`artifacts/rechecks/2026-09-23-verifier-search/frozen-design.json`](../artifacts/rechecks/2026-09-23-verifier-search/frozen-design.json).

## Why this experiment

The latest SFT-initialized GRPO comparison found no demonstrated gain over
matching SFT after 50 steps: source greedy pass@1 changed by +0.39 percentage
points (95% paired task-bootstrap interval −0.65 to +1.56), and fresh greedy
pass@1 by −0.26 points (−0.78 to +0.26). In the same report, the SFT policies'
sampled pass@4 was 31.64% on source and 16.80% on fresh tasks, compared with
greedy pass@1 of 21.48% and 7.81%. That saved evidence suggests useful correct
outputs already exist in the sampling distribution even when greedy decoding
misses them.

The next bounded test is therefore exact-verifier-filtered sampling rather
than another optimizer run. Self-consistency work motivates sampling multiple
reasoning paths, but this task has a stronger selection signal than majority
vote: the unchanged Countdown verifier can accept only a legal exact solution.
The hypothesis is that a fixed candidate budget of eight independent samples,
with greedy as the first candidate, raises exact solve rate over greedy for
each of the three SFT policies on both new held-out suites. This is an
inference-system result, not a learning or emergent-reasoning claim.

Alternative optimizer changes were considered. DAPO's dynamic sampling and
clip-higher recipe was demonstrated at much larger scale; the previous SFT
policies already had 21%–36% mixed reward groups, so a no-signal-group remedy
is not the clearest next intervention here. Dr. GRPO targets response-length
normalization bias; the prior Countdown completions averaged about 36 tokens
with no truncations, so that mechanism is not currently supported by project
evidence. The exact-verifier search directly tests the observed gap between
greedy and sampled solve rates while staying within the existing AMD GPU path.

Primary references:

- Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in
  Language Models](https://arxiv.org/abs/2203.11171), which samples multiple
  reasoning paths and aggregates them. Here we test exact verifier selection,
  not majority voting.
- Yu et al., [DAPO: An Open-Source LLM Reinforcement Learning System at
  Scale](https://arxiv.org/abs/2503.14476).
- Liu et al., [Understanding R1-Zero-Like Training: A Critical
  Perspective](https://arxiv.org/abs/2503.20783), which analyzes GRPO's
  response-length bias and proposes Dr. GRPO.
- [TRL 1.12.0 GRPO configuration and sampling semantics](https://huggingface.co/docs/trl/v0.22.1/grpo_trainer)
  are not used for this no-training experiment; the installed evaluator's
  pinned behavior and generation command are recorded in its summaries.

## Frozen method

- Task and exact verifier stay unchanged: integer-only binary `+`, `-`, `*`,
  `/`, parentheses, each supplied number exactly once, exact rational checking,
  integer intermediates/final value, and binary reward only for exact solutions.
- Compare the pinned untouched `Qwen/Qwen3.5-0.8B-Base` checkpoint with all
  three frozen final SFT adapters (seeds 42, 43, 44). Adapter hashes are frozen
  in the design. Do not include GRPO adapters in this inference-only contrast;
  their result remains in the previous report.
- Before evaluation, prepare 256 source-test tasks and 256 fresh tasks with
  the existing witness-free confirmation utility. Exclude canonical task keys
  observed in prior artifact JSONL, all train/dev/fresh-source splits, and
  duplicates between suites. Report source oracle solvability; all fresh tasks
  must be oracle-solvable. Oracle witnesses never enter prompts or candidate
  selection.
- Use the existing raw prompt, greedy plus eight separately seeded sampled
  completions per task, 128 new-token cap, temperature 1.0, top-p 0.95, seed
  base 20260923, eager attention, and the observed ROCm `cuda` device. Use the
  same tasks and generation seed schedule for base and every SFT adapter.
- Candidate order is greedy first, then sampled completions by ascending
  `sample_index`. Select the first completion accepted by the exact verifier;
  if none is valid, return no solution. The fixed bank is fully generated for
  comparable pass@k measurement; result records simulate the ordered selector.
  No outputs are fed back to the model.

## Frozen analysis and success rule

Report greedy pass@1; sample-only pass@4 and pass@8; and verifier-search solve
rates after 1, 5, and 9 candidates (greedy plus zero, four, or eight samples).
Report exact counts and failure categories, truncation, output lengths,
selected candidate index, and all selected verifier outcomes. Preserve full
raw completions. Compute paired task-level percentile intervals from 10,000
resamples with seed 20260923, per seed and for the three-seed mean; report
source-solvable-only source results alongside all-source results.

The predeclared positive result requires all three SFT seeds to improve
verifier-search solve rate from one greedy candidate to nine total candidates
on both source and fresh suites, and positive three-seed paired task-bootstrap
intervals for that gain on both suites. Also compare every SFT seed with the
untouched base under the same candidate budget. If this condition is not met,
report negative, mixed, or inconclusive results as warranted. A higher sampled
solve rate is evidence for this bounded inference-time solver only—not that
GRPO learned or that general reasoning emerged.

## Compute and integrity limits

The four frozen policies require 4 × 512 × 9 = 18,432 generated completions.
The prior seven-model confirmation required 17,920 completions and recorded
17,002.174 seconds of evaluator wall time, so this run is estimated at about
4.86 GPU process-hours. This separately named confirmation has a hard cap of
6.0 GPU process-hours, including model startup and all four evaluators. Check
GPU occupancy before each evaluator. Stop before the cap; if it cannot fit,
record the exact command/output and leave the study explicitly incomplete.
There is no training phase in this design.

Do not tune prompts, sampling settings, model choice, selection rules, or
checkpoint choice on these tasks. Do not regenerate tasks or rerun an
unfavorable model result. A technical retry, if required, must keep the frozen
seed/config, write to a new output path, and retain the failed attempt evidence.
Write all new evidence under
`artifacts/rechecks/2026-09-23-verifier-search/`; never modify prior artifacts.

# Next steps after the frozen confirmation

The 2026-09-18 amendment authorized separately labeled approaches while
preserving the original no-SFT/base/binary-GRPO result. The supervised
initialization branch completed three 512-step seeds and the frozen source and
fresh evaluations on 2026-09-22. Its full measurements and generated plot are
in [`artifacts/rechecks/2026-09-22-confirmation/report-final`](../artifacts/rechecks/2026-09-22-confirmation/report-final/).

## Five success criteria

| Criterion | Evidence | Assessment |
|---|---|---|
| Improve exact solving over untouched initialization | Paired source and fresh pass@1 and pass@4 comparisons against the same base records | Met on the three observed runs |
| Improve on unseen source-distribution tasks | All three source greedy gains are positive; each paired task-bootstrap 95% interval is above zero | Met on this 256-task confirmation suite |
| Improve on independently generated fresh tasks | All three fresh greedy gains are positive; each paired interval is above zero | Met on this 256-task solvable suite |
| Repeat across at least three seeds | Seeds 42, 43, and 44 all improve on both suites | Met; three training seeds do not characterize the full seed population |
| Support arithmetic-search improvement beyond formatting | Gains remain after identical terminal `= integer` normalization; 15 sampled disagreement cases change legal wrong-target arithmetic to legal exact arithmetic | Supported narrowly; most sampled disagreements remain ambiguous |

For the three-seed mean, greedy pass@1 was 18.49% on source tasks versus 0.00%
for the base, and 8.46% on fresh tasks versus 0.39%. Paired task-bootstrap
95% intervals for gains were +14.32 to +22.92 and +5.08 to +11.20 percentage
points, respectively. The normalized greedy gains were +16.93 points on source
and +6.51 on fresh, with positive intervals for both.

The fixed audit selected at most 20 greedy disagreements per seed in lexical
task-ID order across the suites. Because fresh IDs sort before source IDs, all
60 sampled comparisons came from the fresh suite. The audit recorded 15 direct
legal-arithmetic improvements, 3 formatting-only cases, 39 ambiguous changes,
and 3 losses; task IDs recur across seeds. Future work may predeclare a
stratified audit, but it must leave this saved audit unchanged.

## What the original GRPO branch established

The no-SFT base branch remains a negative result for its tested configuration.
The 25-step diagnostic produced only one positive step and 24 all-zero groups.
Three train/dev-only sampling probes did not recover positive or mixed groups,
so the predeclared binary-GRPO gate stopped before 100 or 300 steps. The frozen
untouched base scored 5 exact completions among 2,560 records. The supervised
extension does not retroactively turn that into a GRPO result.

The local instruct checkpoint remains a packaging diagnostic because its
tokenizer metadata did not match the official instruct revision.

## SFT-initialized GRPO follow-up (completed)

The frozen method in
[`amendment-2026-09-22-sft-init-grpo.md`](amendment-2026-09-22-sft-init-grpo.md)
completed all three train-only signal gates, 50-step GRPO runs, and frozen
base/SFT/GRPO evaluations. Its paired report is
[`report-final`](../artifacts/rechecks/2026-09-22-sft-init-grpo/report-final/).

| Criterion for additional GRPO benefit | Result |
|---|---|
| Paired greedy gain over matching SFT on source tasks | +0.39 percentage points; 95% task-bootstrap interval −0.65 to +1.56 (not established) |
| Paired greedy gain over matching SFT on fresh tasks | −0.26 points; interval −0.78 to +0.26 (not established) |
| Positive direction for all three seeds on both suites | Not met: seed 42 was flat on fresh; seeds 43 and 44 declined on fresh; seed 44 also declined on source |
| Sampled pass@4 improvement | Small aggregate gains (+1.30 source, +0.26 fresh), with both intervals spanning zero |
| GPU cap | 5.064 of 6.0 follow-up process-hours recorded; no further run is authorized under this frozen design |

Conclusion: this 50-step binary-GRPO follow-up did not demonstrate improvement
over its supervised initialization. Mixed reward groups and positive training
reward were present, but did not translate into a repeatable held-out gain.
This result is limited to the frozen model, reward, tasks, and 50-step budget;
it does not show that GRPO cannot help more generally. Keep this branch
separate from both the successful SFT result and the historical no-SFT GRPO
result.

## Recommended next experiment

The most useful next test is whether 50 GRPO steps were simply too few. If the
project continues, predeclare a separately named three-seed follow-up with a
longer fixed budget (for example, 150–300 total steps from each frozen SFT
adapter), a train/dev-only checkpoint or stopping rule, and new untouched
source/fresh confirmation tasks. Keep the binary exact reward and arithmetic
contract for a direct comparison; any curriculum, reward, model, or task change
must be a separate explicitly justified method. Estimate the full three-seed
training and evaluation cost first and authorize a new budget—the completed
six-hour amendment must not be extended after seeing its confirmation results.

Require the final comparison to report paired pass@1 and pass@4 against the
matching SFT checkpoints and untouched base, seed directions, task-bootstrap
intervals, reward-group diagnostics, the same equality-suffix normalization,
and audited gains and losses. Use new task identities and freeze the analysis
before the first optimizer step. Do not infer success from reward curves,
legal-expression rate, or one seed.

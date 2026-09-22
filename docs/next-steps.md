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

## Recommended next experiment

Test binary GRPO initialized from the supervised checkpoint. First score
train-only rollouts from the SFT policy using the frozen binary verifier and
measure positive completions, mixed reward groups, and truncation. Do not use
source confirmation or fresh confirmation tasks for this gate. If the amendment's
signal threshold passes, freeze a short GRPO diagnostic against the same SFT
adapter, then evaluate the post-GRPO checkpoint against both the pre-GRPO SFT
checkpoint and untouched base under a newly predeclared confirmation design.
Retain the same task contract and report whether any improvement survives the
same formatting normalization.

The final-step SFT checkpoints are the fixed starting point for this next
comparison. Do not retune them from confirmation outcomes. Complete and publish
the budget and design amendment before launching this follow-up.

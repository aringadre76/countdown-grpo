# Learning study amendment — 2026-09-18

The user authorized alternative models, rewards, curricula, algorithms, and
supervised initialization to pursue five criteria: improvement over initialization,
unseen source gains, fresh gains, replication across three seeds, and arithmetic
search improvement beyond formatting. Historical results remain immutable.
The original no-SFT/binary/base claim continues to refer only to its original runs.

## Evidence and initial hypotheses

The old TRL callback defaults truncation to false although installed TRL passes
completion IDs and computes clipping from their final token. Correct this first,
record token IDs, and treat unobserved termination as unknown. Historical
callback truncation fields must not be used as measurements of clipping.

The earlier control has weight-hash provenance but reuses base tokenizer/config
metadata. Retain that caveat. Its chat eval used 32 tokens and training used a
raw prompt; this does not establish that a consistently formatted control cannot
learn. First test 128-token native chat/no-thinking completions on 16 dev tasks,
one greedy and four samples per task, seed 42, temperature 1.0, top-p 0.95.
If truncation exceeds 20%, allow one 256-token repetition on the same dev tasks.
Do not revisit historical final tasks to choose a setting.

Primary sources consulted:

- https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/README.md
  documents native chat use, default non-thinking, and small-model thinking loops.
- https://huggingface.co/docs/trl/grpo_trainer documents completion IDs in rewards.
- https://github.com/Jiayi-Pan/TinyZero motivates competence/sparse-reward checks.

## Budget and decision gates

Budget: at most 12 GPU-hours for this study, including evaluation. Record actual
wall time and GPU process lifetimes rather than inferred utilization.
Initially allow the two control probes above. At least two sampled positive
tasks and 10% mixed sampled groups permit a 50-step binary-GRPO diagnostic on
training-only tasks. Use batch 1, accumulation 8, four generations, bf16 LoRA,
learning rate 1e-5, eager attention, and the selected chat/completion settings.
Require positive completions in at least five of the last 20 steps, at least
10% mixed groups over that window, and finite gradients to advance to 100 steps.
Apply the same gate at 100 before three 300-step runs (seeds 42, 43, 44),
starting each from identical initialization. Select final-step checkpoints.

If the control fails its gate, predeclare the next bounded intervention and its
appropriate signal test before running it. Supervised training is authorized
only in a named branch with train-only solutions, separate datasets, and
pre-RL supervised evaluation. Do not extend budgets based on confirmation results.

## Confirmation design

Before confirmation, freeze settings and sample 256 unseen canonical source
tasks plus 256 new oracle-solvable fresh tasks. Exclude training, dev, all old
evaluation tasks, and cross-suite duplicates. Use one greedy and four sampled
completions per task. Compare initialization and all three final checkpoints.
Report solvable-only source results as well as all-source results.

Compute paired task bootstrap intervals with 10,000 resamples and seed 20260918,
separately for greedy accuracy and sampled pass@4. Require positive paired greedy
gain intervals on both suites and positive point gains for all three seeds;
also publish failures and intervals per seed. A method may satisfy fewer than
five criteria; report each verdict honestly.

For the search audit, apply identical diagnostic removal of an equality suffix
to initialization and trained outputs, with the resulting expression rechecked
under the unchanged arithmetic contract. Never alter primary reward or accuracy.
Require gains after this normalization and audit 20 task pairs selected by fixed
task-ID order among disagreements, publishing traces and a rubric separating
format-only repairs from changed legal target-reaching arithmetic. This supports
a narrow search claim; it cannot establish general reasoning.

## Candidate B: supervised initialization (conditional on candidate A failure)

The metadata audit also verified that the local control tokenizer/configuration
does not match the official instruct tokenizer. Candidate A is therefore a
local packaging diagnostic, not a valid official-model competence control.
If matching metadata cannot be obtained through the authorized browser path,
proceed to candidate B with the correctly cached pinned base snapshot, recording
the control limitation. Do not interpret packaging failure as a capacity result.

The 128-token instruct probe failed with 0/80 exact completions and 47.5%
clipping. The predeclared 256-token repetition must finish before selecting
this candidate. If neither passes the binary-GRPO signal gate, test explicit
supervised LoRA on Qwen3.5-0.8B-Base at its historical pinned revision.
Rationale: prompt-only exploration has not supplied useful exact trajectories;
verified demonstrations provide a direct learning signal. This changes the
question to supervised learning of Countdown and possible subsequent RL gains.
TRL's SFT documentation explains prompt/completion datasets and completion-only
loss: https://huggingface.co/docs/trl/sft_trainer.

Generate 4,096 canonical solvable examples from source_train, shuffled with
seed 20260918. Verify all witnesses independently with the parser. Store them
in an ignored separate supervised data directory; record IDs and file hashes.
Witnesses are completion targets, never prompt hints or evaluation inputs.

Use the original raw prompt, bf16/eager LoRA rank 16, alpha 32, dropout 0.05,
observed hybrid attention targets plus observed gate/up/down MLP projections,
batch 1, accumulation 8, learning rate 2e-4, linear schedule, completion-only
standard NLL, and max sequence length 256 with a hard check rejecting truncation.
First run one integration step; then start a fresh 64-step diagnostic from the
same untouched initialization (seed 42). Compare baseline and diagnostic on
the same 16 dev tasks at 128 tokens, greedy plus four samples, temperature 1.0,
top-p 0.95. Continue to fresh 512-step training if logged loss decreases,
gradients are finite, and dev legal-expression rate reaches 20%. This is a
supervised-learning gate, not a claim that arithmetic search already improved.

After 512 steps, require higher dev exact accuracy or sampled pass@4 than
initialization to justify three fresh 512-step seed runs (42, 43, 44). These
are substantive supervised runs, independent of the original GRPO study.
If no exact dev gain appears, retain the result and predeclare a next candidate
within the remaining budget. If supervised replication passes, freeze final-step
checkpoints and conduct the confirmation analysis above. Any subsequent GRPO
must compare against the supervised checkpoint and pass its own signal gate.

Integration correction: TRL's default concatenated-text tokenization produced
prompt-prefix mismatches and masked the start of the answer. The label audit
rejected the run before training. Use separately tokenized prompt/answer IDs
and explicit -100 prompt labels, matching the generation boundary. Preserve
the failed integration attempt; require the label probe to pass before training.

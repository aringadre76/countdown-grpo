# SFT-initialized binary GRPO follow-up — 2026-09-22

Status: frozen before the train-only signal probe, new confirmation-suite
preparation, and any warm-start GRPO training. The machine-readable design and
input adapter hashes are in
[`artifacts/rechecks/2026-09-22-sft-init-grpo/frozen-design.json`](../artifacts/rechecks/2026-09-22-sft-init-grpo/frozen-design.json).
This is an addendum to the 2026-09-18 learning-study amendment, not a change to
the historical no-SFT/base/binary-GRPO result or the completed SFT confirmation.

## Question and provenance

Does TRL `GRPOTrainer` with the unchanged binary exact verifier improve the
frozen supervised Countdown policy further? The three final-step SFT adapters
(seeds 42, 43, 44) are the starting checkpoints, not new SFT runs. The policy
must continue the existing LoRA weights; it must not attach a fresh random
LoRA layer or reload an instruct model. All GRPO training prompts come from
`source_train`; source-test and fresh confirmation data remain evaluation-only.

The verifier contract is unchanged: binary `+`, `-`, `*`, `/`, parentheses,
every input number exactly once, exact rational checking, integer intermediate
and final values, and a reward of one only for a legal exact target solution.
Oracle witnesses are not supplied to GRPO prompts. Generated text is never
executed.

## Train-only signal gate

Before training, evaluate each frozen SFT adapter on the same first 32 rows of
the hashed `artifacts/data/source_train.jsonl`, in saved row order. Use one
greedy completion and four sampled completions per task, 128 new tokens,
temperature 1.0, top-p 1.0, raw prompts, and the binary verifier. Each adapter
passes only if it has at least two sampled-positive task groups, at least a
10% mixed sampled-group rate, and no more than 20% truncation. This evaluates
training tasks only and does not tune on dev or confirmation results.

For any adapter that fails that gate, allow one predeclared train-only repeat
on those exact same 32 tasks at temperature 1.3 and top-p 1.0. That adapter
may proceed only if the same gate passes; its GRPO sampling temperature then
matches the passing probe. No further temperature search is authorized by
this amendment. A failed gate means no GRPO run for that seed; record the
observed blocker and continue only with seeds that pass.

## GRPO configuration and staged stop

After a passing signal probe, run a separate one-step integration check from
the frozen seed-42 SFT adapter, saving to its own output path. Confirm that the
loaded adapter has the expected live target modules, trainable parameters, and
finite gradients. This is only an integration check; it is not a learning
result. The 50-step runs, when authorized by the gate, each start again from
their untouched frozen SFT adapter.

The bounded diagnostic uses up to 50 steps per passing seed, learning rate
`1e-5`, per-device batch 1, gradient accumulation 8, four generations, and
effective generation batch 8. It continues the SFT adapter weights using LoRA
rank 16 / alpha 32 / dropout 0.05 as saved in each adapter, with bf16, eager
attention, raw prompts, a 128-token prompt/completion cap, top-p 1.0, and the
temperature that passed the train-only signal gate. TRL 1.12.0's installed
`GRPOConfig` confirms the effective-batch rule: the default steps per
generation equals gradient accumulation, so the generation batch is 8 and is
divisible by four generations. `beta=0.0` and `loss_type="dapo"` are explicit.
No vLLM or QLoRA is used.

An adapter must load as a trainable PEFT model and be passed to
`GRPOTrainer` without a second `peft_config`; the pinned TRL implementation
rejects passing both. The runner records the initial adapter hash, target
modules, package versions, model/device, GPU memory snapshot, training config,
and frozen-design hash. Reward logs retain task IDs, raw completions, verifier
categories, group statistics, and truncation. A callback records finite
gradient status immediately before optimizer steps. Final-step checkpoints
are fixed in advance; confirmation performance cannot select or retune a
checkpoint.

The experiment remains a 50-step comparison even if the original GRPO
amendment's later 100/300-step gates could be satisfied. Any longer training
requires a new predeclared amendment and budget check.

## Frozen confirmation

Before the first GRPO optimizer step, prepare 256 unseen source-test tasks and
256 independently generated fresh tasks. Exclude task identities in all prior
artifact JSONL, the source train/dev sets, and previous fresh/source
evaluations. Verify source solvability and fresh-suite solvability with the
independent oracle; witnesses stay out of prompts.

After eligible final checkpoints are frozen, evaluate the untouched base, each
matching SFT initialization, and each corresponding GRPO adapter on these same
tasks using raw prompts, one greedy plus four sampled completions, 128 new
tokens, temperature 1.0, top-p 0.95, and eager attention. Report exact
pass@1 and sampled pass@4 separately. Pair GRPO both with its matching SFT
adapter and with the untouched base; use 10,000 task-level bootstrap
resamples with seed 20260922. Also report the predeclared equality-suffix
normalization and fixed-order trace audit, stratified to at most ten source
and ten fresh disagreements per seed (including losses). A positive result
must be described only as a measured Countdown improvement; it is not evidence
of general reasoning.

## Budget

The parent 2026-09-18 amendment caps the full study at 12 GPU-hours including
evaluation. Saved evidence through 2026-09-22 records 10,111.732 seconds of
evaluation wall time and 5,809 seconds of SFT command/process lifetimes,
4.422 hours combined. The first instruct L=128 probe predates evaluator
runtime logging; its first-to-last completion timestamps span about 144
seconds, but startup/runtime was not measured. Reserve 0.5 hours for that
unmetered probe and measurement overhead. This follow-up has a hard cap of
6.0 additional command/process hours, leaving about 1.08 hours below the
12-hour amendment limit after the reserve. Do not infer utilization from idle
time. Record actual start/end and runtime for every new GPU process and stop
before the follow-up cap.

The first artifact in the evidence directory is the frozen design. Subsequent
manifests, train-only probes, attempts, raw completions, confirmations, and
reports must be saved alongside it without overwriting previous evidence.

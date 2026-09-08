# Next steps after the CPU lower-bound run

The next objective is not a bigger model or a longer CPU run. It is to find
out whether the intended Qwen3.5-0.8B base experiment has usable reward signal
on the RX 7900 XTX while preserving the original task and reward.

## What the current result says

The saved CPU smoke completed one optimizer step but all eight rollouts had
reward 0. The paired bounded evaluation also had 0 exact solves in 40 source
and 40 fresh completions for both the base model and the smoke adapter. This
shows that the pipeline can execute on CPU. It does not estimate the final
held-out rate, diagnose GPU training, or show that RL cannot learn Countdown.
The full 2026-09-08 CPU recheck reproduced those stable scored outputs.

## First: recover a usable GPU environment

1. Start a fresh WSL session and record `rocm-smi`, `torch.cuda.is_available()`,
   `torch.version.hip`, and the device name in a new environment manifest.
2. Use a ROCm Torch build matched to the machine. A successful llama.cpp HIP
   inference stack is not proof that PyTorch training kernels will work.
3. Check whether the named llama.cpp server is occupying VRAM. If it is the
   blocker, stop only that process and record the action; do not alter its
   installation.
4. Run the model/LoRA preflight unchanged. It must load, generate, backprop,
   attach LoRA to observed modules, and change adapter weights after a real
   optimizer step.

Stop here if any item fails and save the actual command output. Do not switch
to an instruct model, SFT, a shaped reward, or a different Countdown contract
to make the primary run easier.

## Then: establish whether GRPO has a signal

Use the frozen base prompt and train/dev data only to explore a documented
sampling range. The purpose is not to maximize a held-out score; it is to
measure whether exact solutions occur at all before spending GPU time.

| Gate | Run | Continue only when |
| --- | --- | --- |
| Baseline | Fixed base checkpoint on the frozen evaluation suite | JSONL and failure breakdown are saved. |
| Smoke | 1–2 optimizer steps | Reward pipeline is verified against oracle witnesses. |
| Diagnostic | 25–50 steps | Some groups have both 0 and 1 rewards. |
| Intermediate | about 100 steps | Diagnostics remain stable and signal persists. |
| Demonstration | about 300 steps, plus another seed if feasible | Intermediate results justify more compute. |

At every gate, save mean reward, variance, positive-completion rate,
mixed-reward-group rate, all-zero/all-one groups, failure categories,
completion length, truncation, trainer loss/KL when exposed, checkpoint, and
actual compute. The mixed-group rate is the decision metric; a higher mean
reward alone is not enough.

## Read the failure mode before changing anything

- Mostly truncation: check the completion limit and save examples. Any length
  change is a documented train/dev setting, not a test-tuned change.
- Mostly malformed or unsupported syntax: report a formatting issue. Do not
  call it arithmetic improvement.
- Legal expressions that miss the target: report constraint following without
  arithmetic-search success.
- A few positives but no mixed groups: increase exploration only within a
  documented train/dev range, then rerun the signal check.
- Source gains with no fresh-task gains: discuss source dependence or
  memorization rather than generalization.
- All-zero groups after a verified pipeline: report insufficient exploration
  signal for this setup. That is a useful lower-bound result.

## Evaluation discipline

Keep the source test and fresh suite frozen until a checkpoint is selected by
the prior gates. Evaluate the untouched base and every reported checkpoint with
identical generation settings. Report pass@1 separately from pass@k, source
and fresh performance separately, and uncertainty intervals where the sample
size makes them useful. The exact verifier should continue to reject parser
tricks, reused inputs, missing inputs, fractions, division by zero, malformed
tags, prose, and target-only answers.

## Optional controls, only after the base path is understood

If the core base run works technically but never receives a useful signal, a
cheap Qwen3.5-0.8B instruct control can test whether initial policy competence
is the limiting factor. It is a separately labeled control, never a replacement
for the base/no-SFT result. Do not start a 3B run first. A shaped-reward run is
also a named ablation, not evidence for the primary binary-reward question.

## Best next experiment

Restore ROCm PyTorch access on the RX 7900 XTX, rerun the recorded baseline,
and run a 25–50-step LoRA GRPO diagnostic only if train/dev sampling shows
mixed exact-reward groups. This answers the immediate uncertainty—whether the
base policy supplies any group-relative signal—without weakening the research
question.

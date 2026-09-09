# Next steps after the GPU diagnostic

The AMD/ROCm path is now functional, so the next question is no longer “can
the machine train?” It is whether the base policy can produce enough legal
exact solutions for binary GRPO to learn from.

## What is established

- The untouched base scored 0/40 source-held-out and 0/40 fresh exact solves in
  the bounded GPU evaluation.
- The 25-step adapter scored 1/40 source and 0/40 fresh. Its one source hit is
  consistent with formatting improvement because the base generated the same
  arithmetic with an illegal `=` suffix.
- The diagnostic completed on the RX 7900 XTX, but 24/25 steps were all-zero
  groups. The protocol therefore stops before the 100- and 300-step gates.
- A train/dev-only follow-up tried 32 tokens at temperature 1.0 and 64 tokens
  at temperature 1.3, then 128 tokens at temperature 1.8. All three settings
  produced 0/32 exact rewards, 0/8 positive tasks, and 0/8 mixed tasks. This
  did not justify another GRPO diagnostic.
- The separately labeled Qwen3.5-0.8B instruct control is currently blocked
  before model loading because its config is not in the local cache. No shell
  download was attempted; the exact blocker and Brave model-page link are in
  `artifacts/rechecks/2026-09-08-gpu-followup/instruct-control-blocker.json`.

## Recommended sequence

1. Keep the source test and fresh suite frozen. Do not tune against them.
2. Do not launch another binary-reward diagnostic from the current all-zero
   dev signal. The saved probes are the stopping evidence for this branch.
3. Download `Qwen/Qwen3.5-0.8B` through the approved Brave path, then run the
   separately labelled instruct control with the same verifier, prompt, and
   frozen evaluation settings. Keep it a control, never a replacement for the
   base result.
4. Evaluate any preselected checkpoint once on the frozen source and fresh
   suites with unchanged generation settings. Report pass@1 and pass@k
   separately.
5. Only after the base result is understood, consider a separately labelled
   0.8B instruct positive control. It cannot replace the base/no-SFT result.

## Failure-guided choices

- Mostly truncation: increase the completion limit only in a documented
  train/dev experiment and retain the old baseline for comparison.
- Mostly prose or unsupported syntax: call it format learning, not arithmetic
  learning; inspect raw completions before changing prompts.
- Legal expressions that miss the target: report constraint following without
  claiming search improvement.
- Source gain with no fresh gain: discuss source dependence or memorization.
- All-zero groups after verified setup: report insufficient exploration signal
  for this lower-bound configuration. Do not silently add shaped reward or
  switch to an instruct model.

The current strongest conclusion is a reproducible GPU negative result with a
small formatting-sensitive source hit and no fresh-task improvement.

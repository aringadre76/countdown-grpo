# Changelog

## Unreleased

- Completed the frozen three-seed SFT-initialized binary-GRPO follow-up on the
  RX 7900 XTX, including seven confirmation evaluations and 5.064 recorded GPU
  process-hours of the six-hour cap. The paired report found no demonstrated
  improvement over SFT on both frozen suites. Saved all raw completions,
  summaries, audit traces, report, and plot; updated the README, hardware,
  reproduction, lessons, next-step, and agent-routing documentation.
- Enhanced the follow-up report generator to state the predeclared result
  criterion, per-seed source/fresh pass@1 changes, audit categories, observed
  environment, and compute budget from saved evidence.
- Froze a separately labeled SFT-initialized binary-GRPO follow-up, including
  adapter hashes, train-only signal gates, a new source/fresh confirmation
  design, pinned TRL batch semantics, and a six-hour hard follow-up budget.
  No result is claimed before the corresponding GPU artifacts are recorded.
- Added warm-start support to the GRPO runner so a saved PEFT LoRA adapter is
  loaded trainably rather than replaced by a new random adapter. The runner
  now records adapter/data hashes, exact training settings and package/device
  versions, task IDs in reward telemetry, and finite-gradient diagnostics.

- Completed the frozen source/fresh confirmation for supervised seeds 42, 43,
  and 44 on the RX 7900 XTX. All three saved 2,560 records with zero
  truncation; paired task bootstrap, normalization checks, trace audits, and an
  evidence-generated SVG are in `artifacts/rechecks/2026-09-22-confirmation/`.
- Recorded an observed 2026-09-22 WSL/ROCm manifest, including Torch/HIP
  versions, `gfx1100`, free/total device memory, an on-device tensor probe, and
  the `rocm-smi` diagnostic.
- Updated the README, reproduction steps, hardware notes, lessons, next-step
  recommendation, and agent routing with the completed confirmation result.
- Added aggregate normalized comparisons, fixed-order audit counts, and a
  four-panel SVG to the confirmation report generator. CPU validation reached
  52 passing tests and Ruff clean.

- Completed three 512-step supervised LoRA seeds and their fixed dev checks;
  prepared a leakage-safe 256-source/256-fresh confirmation suite and recorded
  the untouched-base confirmation (5/2,560 exact). The full adapter
  confirmation was subsequently completed on all three seeds; see the dated
  evidence above.

- Aligned evaluation termination with TRL's EOS-or-PAD rule and added CPU-only
  token-limit tests for EOS, PAD, and genuinely clipped completions. Base-model
  dev comparisons are unchanged because its EOS and PAD IDs are identical.

- Verified 48 tests and lint in a fresh core/dev venv without Torch, TRL, or
  Transformers. Added strict paired confirmation comparisons and fixed-order
  trace audits; normalized diagnostics remain outside primary scoring.

- Verified user-supplied official instruct metadata, assembled a separate
  ignored control folder, and identified its different EOS token. Corrected
  LFS content-vs-pointer hashing without altering historical evidence.
- Added confirmation preparation that requires a frozen design, excludes
  train/dev and historical canonical tasks, and never writes oracle witnesses.

- Added explicitly supervised, train-only witness preparation and completion-only
  LoRA training, with a label-mask audit and saved failed/successful integration
  evidence. Added paired task-level bootstrap helpers for confirmation analysis.

- Opened the authorized learning-study extension with a dated protocol,
  12 GPU-hour budget, dev signal gates, three-seed replication design, and
  unseen source/fresh confirmation rules.
- Corrected unobserved GRPO truncation being logged as false; new telemetry
  records completion IDs and detects termination consistently with TRL.
- Applied native chat formatting consistently in training and evaluation,
  corrected the tokenizer thinking keyword, and distinguished EOS at the token
  limit from clipped output. Evaluations now save rendered prompts and compute.

- Recovered and documented the RX 7900 XTX ROCm/Torch training path in WSL2,
  including the local Python-header workaround and eager-attention compatibility
  setting.
- Recorded a real one-step GPU smoke, 25-step GPU diagnostic, untouched-base
  baseline, paired adapter evaluation, GPU memory, and raw completions. The
  diagnostic stopped at 25 steps because 24/25 reward groups were all zero.
- Reworked the public README and generated report language around the measured
  GPU result: one source formatting-sensitive hit, no fresh-task improvement,
  and no claim of learned arithmetic search.
- Replaced the placeholder target-hit-but-illegal summary field with an
  exact-parser diagnostic that never enters reward.
- Simplified GRPO diagnostics callback construction and retained command
  metadata in future attempt records.
- Added reproduction, hardware, lessons, and next-step documentation.
- Removed the generic Torch pin from the `train` extra so installing project
  extras cannot silently replace a device-matched ROCm build.
- Ran the gated source-dev follow-up at 32 tokens/temperature 1.0 and 64
  tokens/temperature 1.3. Both settings produced zero exact rewards and zero
  mixed groups, so no second GRPO diagnostic was started.
- Added a third 128-token/temperature 1.8 source-dev probe with the same
  all-zero result, plus an explicit cache blocker for the separately labeled
  instruct control.
- Completed the separately labeled Qwen3.5-0.8B instruct control after the
  user supplied its safetensors weight: recorded the SHA256 and metadata
  provenance, added chat-template/thinking-disabled evaluation flags, and ran
  an 80-record frozen control plus a one-step GPU GRPO smoke. Both exact eval
  splits scored 0/40; the smoke produced two all-zero reward groups.
- Reran the pinned data, preflight, base evaluation, GRPO smoke, paired adapter
  evaluation, and report on 2026-09-08. Stable scored fields match the original
  80-record base and adapter JSONL outputs.

## 0.1.0 — 2026-09-07

- Added the exact integer-only Countdown verifier, independent oracle,
  canonical splits, fresh evaluation suite, and CPU-only CI.
- Recorded the untouched-base evaluation, model/LoRA preflight, one-step CPU
  GRPO smoke, environment manifest, report, and plot.
- Result: no positive rollout group in the bounded CPU fallback. No claim of
  improved Countdown solving or general reasoning is supported.

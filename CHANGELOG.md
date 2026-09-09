# Changelog

## Unreleased

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

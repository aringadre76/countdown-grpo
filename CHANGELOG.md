# Changelog

## Unreleased

- Reworked the public README and generated report language around the actual
  CPU lower-bound result.
- Replaced the placeholder target-hit-but-illegal summary field with an
  exact-parser diagnostic that never enters reward.
- Simplified GRPO diagnostics callback construction and retained command
  metadata in future attempt records.
- Added reproduction, hardware, lessons, and next-step documentation.
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

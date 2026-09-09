# Countdown GRPO evidence report

Generated from the saved artifacts listed below. It reports recorded values only; unrun stages are not filled in.

## Dataset and split audit

- Source revision: `408f70d177020686d34a56bba5952feb45aaaee4`
- Raw rows: 490,364; canonical tasks: 449,570
- Canonical duplicate rows removed: 40,794
- Oracle solvability under the locked contract: 100.00%
- Frozen split sizes: train 359,656, dev 44,957, test 44,957; fresh test 256.

## Observed environment

- Python 3.11.15; Torch 2.13.0+rocm7.14.0; CUDA available: True; device: AMD Radeon RX 7900 XTX; HIP: 7.14.60850.
- HSA probe: `rocminfo` returned 0 and reported the RX 7900 XTX; `rocm-smi` remains non-diagnostic under this WSL session.

## Untouched-base baseline

- Records: 80; exact solve rate: 0.00%; legal expression rate: 0.00%.
- Source-held-out: 0.00%; fresh-task: 0.00%.
- Failure categories: `{"no_answer": 6, "non_integer_intermediate": 4, "truncated": 57, "unsupported_syntax": 9, "wrong_number_multiset": 4}`

## GRPO diagnostic

- Status: completed; optimizer steps: 25; training loss: 0.0033638131618499755.
- Aggregated over 25 steps and 200 rollouts: mean step reward 0.005000; mean step reward variance 0.004375; positive-completion rate 0.005000.
- Mixed-group rate: 0.020000; all-zero-group rate: 0.980000; all-one-group rate: 0.000000; truncation rate: 0.000000.
- Positive steps: `[1]`; mean completion length: 26.14 characters.
- Failure categories: `{"malformed_answer": 17, "no_answer": 18, "non_integer_intermediate": 9, "ok": 1, "syntax_error": 4, "unsupported_syntax": 120, "wrong_number_multiset": 26, "wrong_target": 5}`

## Smoke-adapter comparison

- Records: 80; exact solve rate: 1.25%; legal expression rate: 1.25%.
- This 25-step sparse-signal adapter comparison is not evidence of learning improvement; source and fresh breakdowns remain the deciding evidence.

## Interpretation

The GPU smoke confirms that Qwen3.5, the observed LoRA targets, Triton, the binary reward, and TRL GRPO run together. The 25-step diagnostic had one positive step and 24 all-zero steps, so there was no sustained usable group-relative signal. The run does not support a claim of improved Countdown solving.

## Inputs

- `artifacts/data/source_audit.json`
- `artifacts/data/source_split_manifest.json`
- `artifacts/rechecks/2026-09-08-gpu-recovery/environment-rocm-torch.json`
- `artifacts/rechecks/2026-09-08-gpu-recovery/base-eager.summary.json`
- `artifacts/rechecks/2026-09-08-gpu-recovery/grpo-diagnostic-eager-s42/attempt.json`
- `artifacts/rechecks/2026-09-08-gpu-recovery/grpo-diagnostic-eager-s42/grpo-diagnostics.jsonl`

# Countdown GRPO evidence report

Generated from the saved artifacts listed below. It reports recorded values only; unrun stages are not filled in.

## Dataset and split audit

- Source revision: `408f70d177020686d34a56bba5952feb45aaaee4`
- Raw rows: 490,364; canonical tasks: 449,570
- Canonical duplicate rows removed: 40,794
- Oracle solvability under the locked contract: 100.00%
- Frozen split sizes: train 359,656, dev 44,957, test 44,957; fresh test 256.

## Observed environment

- Python 3.11.15; Torch 2.14.0+cpu; CUDA available: False.
- ROCm probe: ERROR:root:Driver not initialized (amdgpu not found in modules)

## Untouched-base baseline

- Records: 80; exact solve rate: 0.00%; legal expression rate: 0.00%.
- Source-held-out: 0.00%; fresh-task: 0.00%.
- Failure categories: `{"no_answer": 7, "non_integer_intermediate": 7, "syntax_error": 1, "truncated": 47, "unsupported_syntax": 9, "wrong_number_multiset": 9}`

## GRPO smoke

- Status: completed; optimizer steps: 1; training loss: 0.0.
- Mean reward: 0.0; reward variance: 0.0; mixed-group rate: 0.0; all-zero-group rate: 1.0.
- Failure categories: `{"malformed_answer": 2, "no_answer": 1, "syntax_error": 1, "unsupported_syntax": 4}`

## Smoke-adapter comparison

- Records: 80; exact solve rate: 0.00%; legal expression rate: 0.00%.
- This one-step all-zero-reward smoke is an integration comparison, not a learning result.

## Interpretation

The smoke had no mixed reward groups, so it provided no group-relative learning signal. It confirms only that the CPU components run together; it cannot support a claim of improved Countdown solving.

## Inputs

- `artifacts/data/source_audit.json`
- `artifacts/data/source_split_manifest.json`
- `artifacts/environment/cpu-train-manifest.json`
- `artifacts/results/qwen35-08b-base-cpu-baseline-s42.summary.json`
- `artifacts/experiments/qwen35-08b-base-cpu-smoke-s42-retry/attempt.json`
- `artifacts/experiments/qwen35-08b-base-cpu-smoke-s42-retry/grpo-diagnostics.jsonl`

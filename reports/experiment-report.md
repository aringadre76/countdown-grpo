# Historical CPU-only GRPO integration report

Scope: this is the early CPU scaffold check, not the RX 7900 XTX training or frozen confirmation result. The current three-seed GPU confirmation is in `artifacts/rechecks/2026-09-22-confirmation/README.md`.

Generated from the saved artifacts listed below. It reports recorded values only; unrun stages are not filled in.

## Dataset and split audit

- Source revision: `408f70d177020686d34a56bba5952feb45aaaee4`
- Raw rows: 490,364; canonical tasks: 449,570
- Canonical duplicate rows removed: 40,794
- Oracle solvability under the locked contract: 100.00%
- Frozen split sizes: train 359,656, dev 44,957, test 44,957; fresh test 256.

## Observed environment

- Python 3.11.15; Torch 2.14.0+cpu; CUDA available: False; device: not reported; HIP: None.
- HSA probe: `rocminfo` returned 0 and reported the RX 7900 XTX; `rocm-smi` remains non-diagnostic under this WSL session.

## Untouched-base baseline

- Records: 80; exact solve rate: 0.00%; legal expression rate: 0.00%.
- Source-held-out: 0.00%; fresh-task: 0.00%.
- Failure categories: `{"no_answer": 7, "non_integer_intermediate": 7, "syntax_error": 1, "truncated": 47, "unsupported_syntax": 9, "wrong_number_multiset": 9}`

## GRPO diagnostic

- Status: completed; optimizer steps: 1; training loss: 0.0.
- Aggregated over 1 steps and 8 rollouts: mean step reward 0.000000; mean step reward variance 0.000000; positive-completion rate 0.000000.
- Mixed-group rate: 0.000000; all-zero-group rate: 1.000000; all-one-group rate: 0.000000; truncation rate: 0.000000.
- Positive steps: `[]`; mean completion length: 36.50 characters.
- Failure categories: `{"malformed_answer": 2, "no_answer": 1, "syntax_error": 1, "unsupported_syntax": 4}`

## Smoke-adapter comparison

- Records: 80; exact solve rate: 0.00%; legal expression rate: 0.00%.
- This 25-step sparse-signal adapter comparison is not evidence of learning improvement; source and fresh breakdowns remain the deciding evidence.

## Interpretation

The GPU smoke confirms that Qwen3.5, the observed LoRA targets, Triton, the binary reward, and TRL GRPO run together. The 25-step diagnostic had one positive step and 24 all-zero steps, so there was no sustained usable group-relative signal. The run does not support a claim of improved Countdown solving.

## Inputs

- `artifacts/data/source_audit.json`
- `artifacts/data/source_split_manifest.json`
- `artifacts/environment/cpu-train-manifest.json`
- `artifacts/results/qwen35-08b-base-cpu-baseline-s42.summary.json`
- `artifacts/experiments/qwen35-08b-base-cpu-smoke-s42-retry/attempt.json`
- `artifacts/experiments/qwen35-08b-base-cpu-smoke-s42-retry/grpo-diagnostics.jsonl`

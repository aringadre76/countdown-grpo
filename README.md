# Testing GRPO at the Countdown Lower Bound

This is a small, reproducible RLVR experiment: can
[`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
improve at Countdown arithmetic with TRL GRPO and a binary exact reward, but
without supervised Countdown solutions? It is motivated by
[TinyZero](https://github.com/Jiayi-Pan/TinyZero), which found that a smaller
Qwen base model did not learn the task and used a stronger 3B setup.

## Current result

The completed run is a CPU fallback integration test, not a GPU training
result. It produced no rewarded rollout groups, so I stopped after one GRPO
step rather than turning a zero-signal run into a longer story.

| Run | Source held-out | Fresh tasks | What it shows |
| --- | ---: | ---: | --- |
| Untouched base | 0 / 40 exact | 0 / 40 exact | No legal solution in this bounded evaluation. |
| One-step LoRA GRPO adapter | 0 / 40 exact | 0 / 40 exact | Identical scored completions to the base evaluation. |

The evaluation used eight source-held-out and eight fresh tasks, one greedy
completion and four samples per task, a 16-token cap, and seed 42. That is a
bounded CPU measurement, not a result on all 44,957 held-out source tasks.

The smoke had mean reward 0.0, reward variance 0.0, a mixed-reward-group rate
of 0.0, an all-zero-group rate of 1.0, and gradient norm 0.0. One completed
optimizer step confirms that the model, adapter, reward callback, and trainer
can run together; it does not show learning. The full generated
[evidence report](reports/experiment-report.md) and
[plot](plots/experiment-summary.svg) are derived from saved artifacts only.

On 2026-09-08, the complete bounded CPU path was rerun after the evaluator and
logging cleanup. The new base and adapter evaluations each had 80 records with
stable scored fields identical to the original JSONL, including every raw
completion, verifier verdict, reward, and failure category. The fresh
[recheck report](artifacts/rechecks/2026-09-08/report.md) and
[environment manifest](artifacts/rechecks/2026-09-08/environment.json) are
kept alongside the original evidence.

## Experiment design

The core run is intentionally narrow:

- Base checkpoint only: `Qwen/Qwen3.5-0.8B-Base` at revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- No Countdown SFT data, hidden solution demonstrations, instruct checkpoint,
  shaped reward, or test-set tuning.
- Reward is exactly `1.0` for a legal solution and `0.0` otherwise.
- The verifier accepts only binary `+`, `-`, `*`, `/`, and parentheses. It
  requires every supplied number exactly once and integer intermediate/final
  values, using exact rational arithmetic. It never evaluates model text as
  Python.
- An independent exhaustive oracle checks solvability but never writes a
  witness into a prompt or task record.

The source is
[`Jiayi-Pan/Countdown-Tasks-3to4`](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
at revision `408f70d177020686d34a56bba5952feb45aaaee4`. Canonicalizing by
`(target, sorted nums)` left 449,570 tasks from 490,364 rows; 40,794
cross-order duplicates were removed. The oracle found every canonical source
task solvable under the locked contract. Seed 42 produced 359,656 train,
44,957 dev, and 44,957 final test tasks. A separate 256-task fresh suite is
oracle-checked and deduplicated against the source splits. The split hashes
are in [artifacts/data/source_split_manifest.json](artifacts/data/source_split_manifest.json).

## What actually ran

The recorded environment used Python 3.11.15, Torch `2.14.0+cpu`, Accelerate
1.14.0, Datasets 5.0.1, Transformers 5.16.1, TRL 1.12.0, and PEFT 0.20.0.
[`rocm-smi`](artifacts/environment/cpu-train-manifest.json) reported that the
AMD driver was not initialized; CUDA was unavailable. The CPU wheel was an
explicit fallback, not a substitute for the intended RX 7900 XTX run.

Before GRPO, the live base model loaded as `Qwen3_5ForConditionalGeneration`
with 852,985,920 parameters. Text forward pass, generation, LoRA attachment,
backpropagation, and a synthetic optimizer update all worked. LoRA targets
were selected from the live module tree and include the hybrid model's
`in_proj_qkv`, `in_proj_z`, and `out_proj` linear-attention projections as
well as `q_proj`, `k_proj`, `v_proj`, and `o_proj`. The observed details are
in [artifacts/experiments/qwen35-08b-base-cpu-preflight.json](artifacts/experiments/qwen35-08b-base-cpu-preflight.json).

TRL 1.12's generation batch is 8 (`1 × 8`) and is divisible by the configured
four generations per prompt. The runner checks this before constructing the
trainer.

## Reproduce the saved CPU fallback

Use a project virtual environment. On a working AMD system, install a matching
ROCm Torch wheel instead of the CPU wheel below.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
python -m pip install -e '.[dev,train]'

pytest
ruff check .

python -m countdown_grpo.prepare_data --data-dir artifacts/data --seed 42 --fresh-size 256 --smoke-size 8
python -m countdown_grpo.environment --output artifacts/environment/cpu-train-manifest.json
```

Run the base evaluation and one-step smoke explicitly:

```bash
python -m countdown_grpo.evaluate \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir artifacts/data --splits source_test fresh_test --limit 8 \
  --output artifacts/results/qwen35-08b-base-cpu-baseline-s42.jsonl \
  --experiment-id qwen35-08b-base-cpu-baseline-s42 --seed 42 \
  --samples-per-task 4 --max-new-tokens 16 --device cpu

python -m countdown_grpo.train_grpo \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir artifacts/data --train-file smoke_train.jsonl --max-steps 1 \
  --max-completion-length 16 --seed 42 --allow-cpu \
  --experiment-id qwen35-08b-base-cpu-smoke-s42-retry
```

Model weights, caches, and adapters stay ignored. Commit-ready configs,
diagnostics, and completions are written under `artifacts/`.

## Where to look

```text
src/countdown_grpo/verifier.py   exact parser and scorer
src/countdown_grpo/oracle.py     independent solvability check
src/countdown_grpo/data.py       canonical splits and fresh-task generation
src/countdown_grpo/evaluate.py   JSONL evaluator and summary metrics
src/countdown_grpo/train_grpo.py LoRA + binary-reward GRPO runner
src/countdown_grpo/report.py     report and SVG generator
docs/experiment-protocol.md      full methods and stopping rules
```

## Limits and next step

This repository does not claim that RL cannot learn Countdown. The observed
base policy simply produced no positive rollout group in a short CPU run.
There is no functional ROCm training result, 25–100-step diagnostic, extra
seed, or instruct-model control.

The next useful experiment is to restore a compatible ROCm PyTorch setup on
the RX 7900 XTX, rerun the fixed base baseline, and use only train/dev data to
check sampling diversity. A 25–50-step diagnostic is justified only if mixed
reward groups appear; the final source test remains frozen.

## Sources

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [Qwen3.5-0.8B-Base model card](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Countdown-Tasks-3to4 dataset card](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
- [TRL GRPO documentation](https://huggingface.co/docs/trl/main/grpo_trainer)

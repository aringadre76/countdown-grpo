# Testing the Lower Bound of RLVR: GRPO on Countdown with Qwen3.5-0.8B

This repository tests whether [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), with no supervised Countdown solutions, can improve held-out Countdown solving through TRL GRPO and one binary exact reward. It is a small-model lower-bound experiment motivated by [TinyZero](https://github.com/Jiayi-Pan/TinyZero), which reported a failure at Qwen2.5-0.5B and used a stronger 3B setup.

Current evidence is a valid negative/incomplete result: the end-to-end CPU fallback works, but no usable GRPO signal appeared. This does not support a claim of improved arithmetic search or general reasoning.

## Result at a glance

| Evidence | Source held-out | Fresh solvable tasks | Interpretation |
| --- | ---: | ---: | --- |
| Untouched base, greedy plus sampled pass@4 | 0 / 40 exact | 0 / 40 exact | No legal expression in the bounded baseline. |
| One-step LoRA GRPO smoke adapter, paired settings | 0 / 40 exact | 0 / 40 exact | Exactly identical scored completions to the base run. |

The frozen evaluation used eight tasks from each suite, one greedy and four sampled completions per task, a 16-token cap, seed 42, and model revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. It is deliberately a bounded CPU evaluation, not a full 44,957-task held-out run.

The one-step smoke used the same base checkpoint, LoRA, and primary reward. Its eight rollouts had mean reward `0.0`, reward variance `0.0`, mixed-reward-group rate `0.0`, all-zero-group rate `1.0`, and gradient norm `0.0`. The protocol therefore forbids treating it as learning evidence and we did not start the 25–100-step diagnostic.

Full generated report: [reports/experiment-report.md](reports/experiment-report.md). The plot is generated only from saved artifacts: [plots/experiment-summary.svg](plots/experiment-summary.svg).

## Locked task and reward

The core task accepts only binary `+`, `-`, `*`, `/`, and parentheses. Every supplied input number must be used exactly once as a multiset; all intermediate and final values must be integers; evaluation is exact rational arithmetic. Unary minus, `**`, `%`, `//`, concatenation, names, calls, prose, division by zero, and malformed tags are rejected. The verifier never evaluates generated text as Python.

The sole primary reward is `1.0` for a legal expression that reaches the target and `0.0` otherwise. Formatting, legality, and parser diagnostics are logged but never shape this reward. The scorer uses the last well-formed `<answer>...</answer>` span; without an answer tag, the whole completion must itself be a legal expression.

## Data integrity

The source is [`Jiayi-Pan/Countdown-Tasks-3to4`](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) at revision `408f70d177020686d34a56bba5952feb45aaaee4`.

- 490,364 source rows became 449,570 canonical `(target, sorted nums)` tasks.
- 40,794 cross-order canonical duplicates were removed; exact ordered duplicate rows were zero.
- The independent integer-only oracle found all 449,570 canonical tasks solvable.
- Seed 42 created 359,656 train, 44,957 dev, and 44,957 final source-held-out tasks. Their canonical-key hashes are saved in [artifacts/data/source_split_manifest.json](artifacts/data/source_split_manifest.json).
- A separate 256-task fresh suite is generated independently, oracle-checked, and deduplicated against every source split. Oracle witnesses are never written to prompts or task artifacts.

The large transformed train/dev/test JSONL cache is intentionally ignored instead of committed. Its source revision, construction code, counts, and SHA-256 hashes are committed, so the split can be rebuilt without committing redundant source data.

## What ran

The actual CPU fallback environment was Python 3.11.15, Torch `2.14.0+cpu`, Accelerate 1.14.0, Datasets 5.0.1, Transformers 5.16.1, TRL 1.12.0, and PEFT 0.20.0. The full observed manifest is [artifacts/environment/cpu-train-manifest.json](artifacts/environment/cpu-train-manifest.json).

`rocm-smi` reported `Driver not initialized (amdgpu not found in modules)`; CUDA was unavailable and no named llama server was running. A generic PyPI Torch install began resolving CUDA 13 components, so it was stopped before completion. The actual fallback used the official CPU wheel, explicitly labeled as CPU rather than pretending it was the intended AMD GPU run.

Before GRPO, the live base model loaded as `Qwen3_5ForConditionalGeneration` with 852,985,920 base parameters. Text forward and generation worked; a synthetic LoRA backward/optimizer check changed adapter weights. The inspected targets include the hybrid model’s `in_proj_qkv`, `in_proj_z`, and `out_proj` linear-attention projections plus `q_proj`, `k_proj`, `v_proj`, and `o_proj` full-attention projections. See [artifacts/experiments/qwen35-08b-base-cpu-preflight.json](artifacts/experiments/qwen35-08b-base-cpu-preflight.json).

TRL 1.12 computes the default generation batch as `per_device_train_batch_size × gradient_accumulation_steps = 1 × 8 = 8`; it is divisible by `num_generations = 4`. The trainer checks that condition before construction. TRL 1.12 no longer accepts the scaffold’s `max_prompt_length` argument, so the code enforces the 128-token limit against tokenized prepared prompts before trainer construction; the observed maximum was 71 tokens.

## Reproduce

Use a project venv. The CPU-only commands below reproduce the fallback used for these artifacts; on a functioning AMD system, install a matching ROCm Torch wheel first instead of the CPU wheel.

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

Run the untouched base evaluation and CPU smoke explicitly:

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

The training command saves adapter weights only under ignored `outputs/`. It writes safe, commit-ready configs, status, rollout diagnostics, and raw completions to `artifacts/experiments/`.

## Repository map

```text
src/countdown_grpo/
  verifier.py        Safe exact integer-only parser and scorer
  oracle.py          Independent exhaustive solvability oracle
  data.py            Canonical tasks, splits, fresh-suite generation
  prepare_data.py    Source audit and persisted artifact builder
  evaluate.py        Saved JSONL baseline/checkpoint evaluator
  training.py        GRPO batch legality and reward-group diagnostics
  train_grpo.py      LoRA + binary-reward GRPO runner
  preflight.py       Model, LoRA, backward, and optimizer validation
  report.py          Evidence-only Markdown/SVG generator
artifacts/           Saved audit, manifests, completions, and diagnostics
reports/             Generated experiment report
plots/               Generated evidence-only SVG
```

## Limitations and next experiment

This did not test a functional ROCm GPU path, a 25–100-step diagnostic, a 300-step run, an extra seed, or an instruct positive control. It also does not establish that RL cannot learn Countdown: the observed base policy had no rewarded trajectories under this bounded configuration.

The single best next experiment is to restore a working ROCm PyTorch environment on the intended RX 7900 XTX, rerun the untouched base baseline with a documented train/dev-only sampling-diversity check, and proceed to a 25–50-step diagnostic only if mixed reward groups occur. Keep the final source test frozen and retain this all-zero CPU result.

## Sources

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [Qwen3.5-0.8B-Base model card](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Countdown-Tasks-3to4 dataset card](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
- [TRL GRPO documentation](https://huggingface.co/docs/trl/main/grpo_trainer)

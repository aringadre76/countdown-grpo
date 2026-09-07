# Countdown GRPO

Teaching a small pretrained language model to solve arithmetic search problems with reinforcement learning and a verifiable reward.

## Project question

Can `Qwen/Qwen3.5-0.8B-Base` learn useful, generalizable solving behavior when it receives no supervised task solutions and is rewarded only for producing a legal expression that reaches the target?

This project adapts the idea from [The Global Minima](https://x.com/TheGlobalMinima/status/2096532361844609320) into a reproducible RLVR experiment. RLVR means reinforcement learning with verifiable rewards: the evaluator can independently check whether an answer is correct.

## Current status

The repository is an experiment scaffold. The training run has not been performed yet, so it contains no invented results or fake charts.

## Ingredients

- **Base model:** [`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base), not an instruct checkpoint.
- **Dataset:** [`Jiayi-Pan/Countdown-Tasks-3to4`](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4), whose rows contain a target and a sequence of three or four numbers.
- **Algorithm:** Hugging Face TRL [`GRPOTrainer`](https://huggingface.co/docs/trl/main/grpo_trainer).
- **Adaptation:** LoRA, so the experiment is practical on a single consumer GPU.
- **Primary reward:** `1.0` for an expression that uses every supplied number exactly once and evaluates to the target; `0.0` otherwise.

No SFT task examples are used in the core experiment.

## Why this is more than a toy demo

A reward curve alone does not establish reasoning. The project therefore treats evaluation as a first-class artifact:

1. Measure the untouched base model before training.
2. Train with GRPO on a reproducible split.
3. Evaluate on held-out examples and newly generated Countdown tasks.
4. Audit invalid expressions and possible reward hacking.
5. Compare answer accuracy, legality, response length, and visible solution traces.

The intended claim is narrow: whether this small base model improves at a verifiable arithmetic task under outcome-only RL. It is not a claim of general reasoning or consciousness.

## Quick start

Create an environment with Python 3.11 or newer, then install the package and development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run the verifier tests:

```bash
pytest
```

Run a small baseline evaluation:

```bash
python -m countdown_grpo.evaluate --limit 32 --seed 42
```

Run the initial GRPO smoke experiment after confirming the local PyTorch/ROCm setup:

```bash
python -m countdown_grpo.train_grpo \
  --max-steps 300 \
  --output-dir outputs/qwen35-08b-countdown-grpo
```

The command is intentionally conservative. It uses a small generation group, a short completion limit, and LoRA. Training settings should be recorded with every run rather than silently optimized until a favorable result appears.

## Reward contract

The verifier accepts an arithmetic expression using the provided numbers and the operators `+`, `-`, `*`, and `/`. Every input number must be used exactly once. Parentheses are allowed. Division must be exact under the verifier's rational arithmetic, and division by zero is invalid.

The parser is deliberately independent of Python `eval`. This keeps the reward function auditable and prevents arbitrary generated text from becoming executable code.

## Repository layout

```text
src/countdown_grpo/
  data.py          Dataset loading and prompt formatting
  evaluate.py      Baseline and checkpoint evaluation
  rewards.py       TRL-compatible reward function
  train_grpo.py    LoRA + GRPO training entry point
  verifier.py      Safe expression parsing and exact validation
tests/
  test_verifier.py
```

## Planned evidence

The first publishable run should include:

- zero-shot base-model solve rate;
- train and held-out solve rates;
- performance split by three-number and four-number tasks;
- reward mean and standard deviation over time;
- invalid-expression rate and truncation rate;
- representative before/after completions;
- a fresh generated-task evaluation to test distribution shift;
- exact model, dataset, dependency, seed, and hardware metadata.

## Roadmap

- [x] Establish a safe exact verifier.
- [x] Add dataset formatting and baseline evaluation entry points.
- [x] Add a minimal GRPO + LoRA training entry point.
- [ ] Validate the PyTorch/ROCm training environment on an RX 7900 XTX.
- [ ] Run a short smoke test and inspect logs.
- [ ] Run controlled training with fixed seeds.
- [ ] Add plots and checkpoint comparisons from real runs.
- [ ] Publish an adapter and an evidence-backed model card if the result is meaningful.

## Sources

- [Qwen3.5-0.8B-Base model card](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Countdown-Tasks-3to4 dataset card](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
- [TRL GRPO documentation](https://huggingface.co/docs/trl/main/grpo_trainer)
- [TRL PEFT documentation](https://huggingface.co/docs/trl/main/en/peft_integration)

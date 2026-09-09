# Testing GRPO at the Countdown Lower Bound

This repository tests a narrow RLVR question: can the pretrained
[`Qwen/Qwen3.5-0.8B-Base`](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
learn held-out Countdown solutions with TRL GRPO and a binary exact reward,
without supervised Countdown solutions? The design is motivated by
[TinyZero](https://github.com/Jiayi-Pan/TinyZero) and treats 0.8B as a lower
bound. A careful negative result is useful.

## Current result

The intended AMD GPU path now works in WSL2: Torch sees an RX 7900 XTX and a
real Qwen3.5 GRPO run completed. The measured result is still negative for
solving improvement. This was a bounded diagnostic, not a claim about the
full 44,957-task source test split.

| Run | Source held-out | Fresh tasks | Interpretation |
| --- | ---: | ---: | --- |
| Untouched base, 40 records each | 0 / 40 exact | 0 / 40 exact | No legal solution in this bounded sample. |
| 25-step LoRA adapter, 40 records each | 1 / 40 exact | 0 / 40 exact | One source success; no fresh gain, and it is consistent with output-format improvement rather than demonstrated arithmetic learning. |

The paired evaluations used the same frozen prompts, eight source tasks and
eight fresh tasks, one greedy completion plus four samples per task, a
16-token cap, and seed 42. The adapter's only success was
`49 + 49 - 65` for target 33; the untouched base produced the arithmetic
equivalent with an unsupported `=` suffix. That is evidence of a possible
formatting change, not evidence of improved search.

The one-step GPU smoke had mean reward 0.125 and a 0.5 mixed-reward-group
rate. The documented 25-step diagnostic produced 200 rollouts: one positive
completion, one mixed-reward step, and 24 all-zero steps. Per the protocol,
the sparse signal was not strong enough to justify a 100- or 300-step run.
The exact counts, raw completions, failure categories, trainer metrics, GPU
memory, and plot are in the
[GPU evidence directory](artifacts/rechecks/2026-09-08-gpu-recovery/).

As the gated follow-up, three source-dev-only sampling probes used
(32 tokens, temperature 1.0), (64, 1.3), and (128, 1.8). All produced 0/32
exact rewards, 0/8 positive tasks, and 0/8 mixed tasks. Because the
development probes supplied no usable exploration signal, the protocol
correctly stopped without another diagnostic or a longer run. Their raw
records, the decision, and the blocked instruct-control cache check are in
[the follow-up evidence directory](artifacts/rechecks/2026-09-08-gpu-followup/).

## Locked experiment

- Model: `Qwen/Qwen3.5-0.8B-Base`, revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.
- No Countdown SFT data, hidden solutions, instruct checkpoint, shaped reward,
  or final-test tuning.
- Reward: `1.0` only for a legal exact solution; otherwise `0.0`.
- Contract: binary `+`, `-`, `*`, `/`, parentheses; every supplied number
  exactly once; exact rational checking; integer intermediate and final values.
  Generated text is never executed as Python.
- An independent exhaustive oracle audits solvability but never places a
  witness in a model prompt.

The source dataset is
[`Jiayi-Pan/Countdown-Tasks-3to4`](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4),
revision `408f70d177020686d34a56bba5952feb45aaaee4`. Canonicalizing by
`(target, sorted nums)` produced 449,570 tasks from 490,364 rows. Seed 42
created 359,656 train, 44,957 dev, and 44,957 source-test tasks, plus a
256-task independently generated fresh suite. The split manifest and hashes
are in [artifacts/data/source_split_manifest.json](artifacts/data/source_split_manifest.json).

## Reproduce it

The exact command-level runbook, including the ROCm wheel setup, Python header
workaround, eager-attention compatibility flag, baseline, smoke, diagnostic,
and paired evaluation is [docs/reproduction.md](docs/reproduction.md).
Hardware observations and WSL caveats are in [docs/hardware.md](docs/hardware.md).

The project keeps a CPU-only `.venv` for tests and a separate ignored
`.venv-rocm` for the GPU run. Installing `.[dev,train]` no longer installs a
generic Torch wheel: choose a device-matched Torch build first.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
```

## Repository map

```text
src/countdown_grpo/verifier.py   exact parser and scorer
src/countdown_grpo/oracle.py     independent solvability check
src/countdown_grpo/data.py       canonical splits and fresh-task generation
src/countdown_grpo/evaluate.py   JSONL evaluator and summary metrics
src/countdown_grpo/train_grpo.py LoRA + binary-reward GRPO runner
src/countdown_grpo/report.py     evidence-only report and SVG generator
docs/experiment-protocol.md      locked methods and gates
docs/reproduction.md             exact rerun commands
docs/hardware.md                 observed hardware and software
```

## Interpretation and next step

The run demonstrates a working AMD/ROCm training path, not emergent reasoning.
The dominant failure mode was unsupported or incomplete output, and the
diagnostic lacked sustained mixed reward groups. The next defensible step is
train/dev-only exploration of completion length and sampling diversity, then a
new seeded diagnostic only if mixed groups persist. The source test and fresh
suite remain frozen. See [docs/next-steps.md](docs/next-steps.md) and
[docs/lessons.md](docs/lessons.md).

## Sources

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero)
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- [Qwen3.5-0.8B-Base model card](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base)
- [Countdown-Tasks-3to4 dataset card](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
- [TRL GRPO documentation](https://huggingface.co/docs/trl/main/grpo_trainer)

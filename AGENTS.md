# Agent instructions

Status: Current
Last updated: 2026-09-08

This file is the operational contract for Codex, Cursor, and other coding
agents working in this repository. After context compaction, a long GPU step,
or a resume, re-read this file and
[`docs/experiment-protocol.md`](docs/experiment-protocol.md).

The paste-ready Codex `/goal` text is in [`docs/codex-goal.md`](docs/codex-goal.md).
Do not paste the full protocol into `/goal`. The goal string must stay under
4,000 characters.

## Repository

- GitHub: https://github.com/aringadre76/countdown-grpo
- Local path: `/home/robot/countdown-grpo`
- Work only in this repository. Do not create another repository.
- Preserve unrelated files and Git history. Do not force-push.

## Documentation routing

Read [`docs/README.md`](docs/README.md) before changing project prose. It maps
each document to one purpose.

- `README.md` is the concise public summary. It must match saved evidence.
- `docs/experiment-protocol.md` locks the research design and gate rules.
- `docs/reproduction.md` is the command-level rerun guide.
- `docs/hardware.md` distinguishes the intended RX 7900 XTX from what the
  recorded WSL environment actually observed.
- `docs/lessons.md` contains durable implementation and experiment lessons.
- `docs/next-steps.md` contains the gated follow-up plan.
- `CHANGELOG.md` records intentional project changes.

Artifacts, not Markdown prose, are the source of truth for counts, commands,
versions, hardware probes, completions, and metrics. Keep original evidence
immutable; write any rerun to a new directory under `artifacts/rechecks/`.

## Research question

Can `Qwen/Qwen3.5-0.8B-Base`, with no SFT Countdown solutions, improve held-out
Countdown solving via TRL GRPO and a binary exact reward?

TinyZero reports that Qwen2.5-0.5B failed to learn Countdown reasoning and
presents a stronger 3B configuration. Treat 0.8B as a lower-bound test. A
measured negative result is a successful project.

## Integrity

- Do not invent metrics, plots, timings, environment details, or successful runs.
- Do not use supervised Countdown solutions in the core experiment.
- Do not silently switch to an instruct checkpoint.
- Do not silently replace binary outcome reward with shaped reward.
- Do not tune hyperparameters on the final held-out test set.
- Do not cherry-pick the best test checkpoint for the README.
- Do not call rising reward "emergent reasoning."
- Do not execute generated model text with Python `eval` or `exec`.
- Do not commit secrets, tokens, caches, downloaded weights, or private paths.
- If a result is uncertain, label it uncertain. If it is negative, keep it.

## Arithmetic contract

Locked for the core experiment:

- Operators: binary `+`, `-`, `*`, `/`, and parentheses.
- Every supplied number is used exactly once as a multiset.
- Exact rational arithmetic internally.
- Every intermediate value and the final value must be an integer.
- Reject unary-minus tricks, `**`, `%`, `//`, concatenation, function calls,
  identifiers, and prose.
- Never use Python `eval`.
- Extract the last well-formed `<answer>...</answer>` span. Score a tag-free
  completion only if the remaining text is itself a legal expression.

Primary reward: `1.0` for a legal exact integer solution, otherwise `0.0`.

## Commands

Use a project venv. Do not install into the system Python or the llama.cpp
environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
```

After training extras exist:

```bash
# Install a device-matched Torch build first; the project extra does not pin one.
pip install -e '.[dev,train]'
python -m countdown_grpo.evaluate --help
python -m countdown_grpo.train_grpo --help
```

CI must run lightweight CPU tests only. It must not `pip install` Torch, TRL,
or the model.

## Gates

Complete Must before Should. Start Stretch only if diagnostics show mixed
reward groups.

**Must ship**

1. Integer-only verifier with adversarial tests. `pytest` passes without a GPU.
2. Independent solvability oracle that never feeds solutions to the model.
3. Canonical `(target, sorted nums)` train/dev/test splits with no leakage.
4. Real untouched-base baseline JSONL. The current `evaluate.py` is a placeholder.
5. Environment manifest written from observed commands.
6. One real GRPO smoke on `Qwen/Qwen3.5-0.8B-Base`, or the exact blocker.
7. README matches evidence, including negative-result framing if needed.
8. CPU CI without Torch/TRL. `local main` equals `origin/main`. `git status --short`
   is empty.

**Should ship if smoke is healthy**

- About 50-100 diagnostic GRPO steps.
- Logs for mean reward, reward variance, positive-completion rate,
  mixed-reward-group rate, and all-zero groups.
- Frozen source held-out eval and fresh-task eval.

**Stretch only if mixed-reward groups exist**

- About 300 steps, an extra seed, and a cheap 0.8B instruct positive control.
- Do not start a 3B run first.

## Landmines

These are already known. Fix or document them; do not rediscover them slowly.

- `evaluate.py` prints a placeholder and does not generate.
- `train_grpo.py` uses `per_device_train_batch_size=1`,
  `gradient_accumulation_steps=8`, and `num_generations=4`. Current TRL
  documents divisibility using the effective batch size, so this is 8 and is
  divisible by 4. Verify the pinned TRL version and effective-batch semantics
  before changing it; do not label the scaffold illegal without that check.
- `pyproject.toml` and GitHub Actions keep CPU CI separate from training
  dependencies. The `train` extra intentionally does not pin Torch, because a
  generic wheel could silently replace a device-matched ROCm build.
- Qwen3.5-0.8B-Base is a vision-language hybrid with Gated DeltaNet layers.
  Inspect `named_modules()` before choosing LoRA targets. Default
  `q_proj,k_proj,v_proj,o_proj` is not sufficient.
- The local everyday stack may run llama.cpp HIP inference of Qwen3.8-27B and
  occupy much of the RX 7900 XTX. Inspect whether it is running and holding
  VRAM; if it blocks training, stop only that named server and record the
  observed process. Do not modify the llama.cpp/ROCm inference install.
- Use a ROCm PyTorch wheel that matches this machine. Inference compatibility
  is not training compatibility.
- Prefer bf16 LoRA. Treat bitsandbytes QLoRA as last resort on AMD.
- Do not make vLLM a requirement.

## Coordination

The parent agent implements, trains, evaluates, and commits. At most two
read-only research subagents. Verify every subagent claim against local files
or command output. Do not accept "the command passed" without the output.

## Logging

Baseline and eval JSONL records must include at least: task id, split, nums,
target, oracle solvability, prompt, raw completion, extracted expression,
verifier result, reward, failure category, lengths, truncation, model id,
revision, generation config, seed, timestamp, experiment id.

GRPO logs must include at least: mean reward, reward variance, positive rate,
mixed-reward-group rate, all-zero groups, all-one groups, failure categories,
completion length, truncation rate, and whatever KL/loss fields TRL exposes.

The mixed-reward-group rate is the GRPO signal diagnostic. All-zero groups
are not evidence that RL cannot work.

## Git

Use focused Conventional Commits. Push intentional commits as gates pass.
Do not commit weights, caches, `.env`, or Hugging Face tokens. At completion,
`git status --short` is empty and `main` matches `origin/main`.

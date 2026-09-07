# Countdown GRPO experiment protocol

Status: Current
Last updated: 2026-09-07

This is the scientific protocol. [`../AGENTS.md`](../AGENTS.md) is the short
operational contract. [`codex-goal.md`](codex-goal.md) is the Codex `/goal`
string. Re-read the current gate from this file before acting on it.

This project is a rigorous small-model RLVR replication and extension. It is
not a novel algorithm. Prior art: TinyZero (Jiayi Pan et al.) and DeepSeek
R1-Zero. Distinguishing choices: Qwen3.5 at 0.8B, pretrained base rather than
instruct for the core run, no SFT task solutions, TRL `GRPOTrainer`, a single
consumer AMD GPU when feasible, an independently tested exact verifier, an
independent solvability oracle, leakage-resistant splits, held-out and fresh
tasks, sparse-reward diagnostics, and honest negative results.

## Research question

Can `Qwen/Qwen3.5-0.8B-Base`, without supervised Countdown solutions, improve
its ability to solve held-out Countdown arithmetic tasks through GRPO using
only an exact, independently verifiable outcome reward?

TinyZero reported that Qwen2.5-0.5B failed to learn Countdown reasoning and
presented a stronger 3B configuration. This experiment tests a newer hybrid
0.8B base model as a lower-bound experiment on an RX 7900 XTX.

A positive result is not required. A carefully measured negative result is
valid. If no meaningful learning occurs, the README should be rewritten around
that outcome, for example:

> Testing the Lower Bound of RLVR: GRPO on Countdown with Qwen3.5-0.8B

## What the writeup must distinguish

- Better output formatting
- More legal arithmetic expressions
- More target-reaching expressions
- Improved arithmetic search
- Memorization or dataset leakage
- Source-distribution generalization
- Fresh-task generalization
- Insufficient exploration signal
- Model capacity limitations
- Broader reasoning claims, which must remain conservative

Do not describe improved Countdown performance as proof of general reasoning.
Do not call a rising reward curve an "aha moment" unless traces and fresh-task
gains actually support a narrow, qualified claim. The preferred conclusion
language is "learned to satisfy the verifier," "formatting improved,"
"arithmetic search improved," or "no usable GRPO signal."

## Integrity rules

- Do not invent results, metrics, plots, timings, environment details, or
  successful commands.
- Do not use supervised Countdown solutions in the core experiment.
- Do not silently replace Qwen3.5-0.8B-Base with an instruct checkpoint.
- Do not silently replace binary outcome reward with shaped reward.
- Do not tune hyperparameters against the final held-out test set.
- Do not select a checkpoint only because it performed best on the final test
  set.
- Do not execute generated model text as Python.
- Do not expose secrets, tokens, credentials, or unnecessary private paths.
- Do not commit Hugging Face tokens, API keys, model caches, downloaded
  weights, or credentials.
- Preserve unrelated changes. Do not rewrite Git history.
- If the experiment fails, diagnose and document the failure.
- If a result is uncertain, label it uncertain.

## Arithmetic contract

Locked unless a dated protocol amendment says otherwise.

The core contract is classic Countdown with exact checking:

- Parse only integer literals from the supplied multiset.
- Binary operators `+`, `-`, `*`, `/` and parentheses only.
- Ordinary operator precedence.
- Every supplied number used exactly once. Duplicates in the input remain
  duplicates in the multiset.
- Reject omitted numbers, reused numbers, and concatenation that creates a
  number not in the multiset.
- Reject unsupported operators, including `**`, `%`, `//`, bitwise operators,
  function calls, identifiers, attributes, lists, and unary-operator tricks.
- Reject division by zero.
- Use exact rational arithmetic internally. Do not use floating-point
  equality.
- Require every intermediate and the final value to be an integer. Fractional
  intermediates are invalid in the core experiment even if they later multiply
  back to an integer.
- Never call Python `eval` or `exec`.

Answer extraction:

- Prefer the last well-formed `<answer>...</answer>` span, case-insensitive.
- Handle trailing whitespace and EOS artifacts.
- If no tag exists, score the completion only when the remaining text is
  itself a legal expression. Do not score a bare last-line number as a
  solution when the rest of the text is prose.

Failure categories (required):

- `ok`
- `no_answer`
- `malformed_answer`
- `syntax_error`
- `unsupported_syntax`
- `wrong_number_multiset`
- `non_integer_intermediate`
- `division_by_zero`
- `wrong_target`
- `truncated`
- `other_invalid`

Primary reward:

- `1.0` when the expression is legal under this contract and equals the target
- `0.0` otherwise

Format checks and parser diagnostics may be logged. They must not enter the
primary reward. A format, legality, or dense-shaping run is allowed only as a
named ablation.

## Provenance

Record starting Git commit before edits. Record exact model and dataset
revisions or commit hashes, not only names. Prefer stable released dependency
versions. If a development Transformers build is required for Qwen3.5, document
exactly why and freeze the working set.

Suggested artifact layout:

```text
configs/
results/
reports/
plots/
artifacts/
```

Each important experiment must record: experiment id, Git commit, timestamp,
model id and revision, dataset id and revision, split manifest hash,
dependency versions, hardware, seed, generation config, training config,
checkpoint path, and exact command.

Do not commit model caches, downloaded base weights, secrets, giant logs,
redundant raw datasets, or unsanitized private paths.

## Gate A. Package, verifier, and tests

Audit the working tree, history, remote, README, source, tests, CI, and
`pyproject.toml`. Starting scaffold notes:

- HEAD was a public `main` scaffold at commit `c672c36`.
- `evaluate.py` is a placeholder.
- `train_grpo.py` needs its effective-batch and `num_generations` settings
  checked against the pinned TRL API before training.
- CI installs the full training stack.

Improve:

- `src/countdown_grpo/verifier.py`
- `src/countdown_grpo/rewards.py`
- `tests/test_verifier.py`
- `pyproject.toml`
- `.github/workflows/ci.yml`

Split dependencies so lightweight verifier tests do not download Torch/TRL:

- core: verifier, oracle, data helpers that can run without a GPU
- dev: pytest, ruff
- train: torch, transformers, trl, peft, accelerate, datasets as required

CI on `ubuntu-latest` should `pip install -e '.[dev]'` only, then `pytest`
and `ruff check .`.

Verifier tests must include: valid 3-number and 4-number expressions,
precedence, nested parentheses, integer-only intermediates, rejected
fractional intermediates, duplicate inputs such as `[2, 2, 3]`, reused
numbers, omitted numbers, concatenation, unsupported operators, malformed
text, prose, division by zero, trailing whitespace, EOS artifacts, tagged
answers, multiple answer tags, target text without a legal expression, and
parser-abuse strings. The verifier must be testable without loading a model.

The reward function must match the current TRL callback shape. Add tests for
string completions and chat-style completion objects if TRL supplies them.

## Gate B. Independent solvability oracle

Add an exact solver for three- and four-number puzzles under the same
integer-only contract. The oracle is for dataset QA and evaluation analysis.
It must not provide solutions to the training model.

Required:

- Exhaustive reachability for the supplied multiset
- Support for duplicate input numbers
- Optional witness expression for diagnostics
- Tests that oracle-generated valid expressions score `ok` in the verifier
- Tests that illegal expressions remain illegal

Never include oracle solutions in prompts, SFT, hidden demonstrations, or
rollout hints.

Optional stretch: count of reachable integer values, or a simple difficulty
proxy. Solvable vs not is enough for the must-ship bar.

## Gate C. Dataset audit and leakage-safe splits

Source dataset: [`Jiayi-Pan/Countdown-Tasks-3to4`](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4).
Preserve raw `target` and `nums`. Keep the primary experiment on this source
even if a Unique variant is used for comparison.

Audit and persist:

- Total row count
- Three-number and four-number counts
- Target and number distributions
- Exact duplicate rows
- Canonical duplicate tasks
- Oracle solvability rate
- Dataset revision metadata

Canonical logical key: `(target, sorted multiset of nums)`.
Deduplicate or group by this key before splitting. No canonical task may
appear in more than one split. Do not use a naive row-level shuffle if
duplicates can cross the boundary.

Create seeded, persisted splits:

- train
- train-side validation/dev
- final source-distribution held-out test
- tiny smoke subsets

Do not use the final test set for prompt or hyperparameter tuning. Report
metrics on all evaluated puzzles and on oracle-confirmed solvable puzzles.
If the source contains impossible tasks, say so. Do not treat those failures
as model failures.

## Gate D. Fresh solvable evaluation suite

Generate a second evaluation suite independently of the Hugging Face rows.
Every fresh task must be solvable under this contract.

Allowed construction:

- Sample numbers, construct a legal integer expression, use its value as the
  target
- Sample number-target pairs and keep only oracle-validated solvable items

Deduplicate fresh tasks against train, dev, held-out, and other fresh tasks.
Store generator seed, config, task ids, composition, and oracle metadata. Do
not expose witness expressions to the model.

## Gate E. Prompt

Core model: `Qwen/Qwen3.5-0.8B-Base`.

The prompt must state the numbers, the target, the exactly-once rule, the
permitted operators, that parentheses are allowed, and the extraction format.
Prefer a concise final expression. Do not make `<think>` traces central. The
question is verifiable arithmetic search, not whether the model emits
convincing explanations.

Because this is a base checkpoint, verify current Qwen tokenizer, control
tokens, and chat-template guidance. Do not assume instruct behavior. Choose
one core prompt before frozen evaluation. Any prompt-format search uses train
or dev only and must be reported as such.

## Gate F. Machine and model

Inspect whether the local Qwen3.8-27B llama.cpp server is running and holding
VRAM. If it blocks training, stop only that named server and record the
observed process. Use a project venv and do not change the llama.cpp/ROCm
inference stack.

Record from observed commands: OS, Python, CPU, GPU model, architecture, VRAM,
ROCm, PyTorch, Transformers, TRL, PEFT, Accelerate, bitsandbytes if used, and
relevant environment variables.

Verify in order:

1. PyTorch sees the RX 7900 XTX.
2. A tensor operation runs on the GPU.
3. Qwen3.5-0.8B-Base loads. Prefer text-only weights if the API allows it.
4. A text forward pass works.
5. Generation works.
6. Backpropagation works.
7. PEFT/LoRA attaches.
8. Selected LoRA modules exist in `named_modules()`.
9. Trainable parameter count is non-zero and plausible.
10. A tiny optimizer step changes adapter weights.
11. TRL can construct `GRPOTrainer` with a legal batch and `num_generations`.
12. One tiny GRPO step completes.

Qwen3.5 interleaves Gated DeltaNet linear attention with softmax attention
and has a large vocabulary. Inspect live module names. Include linear-attention
projections when they exist, for example `linear_attn.in_proj_qkv`,
`linear_attn.in_proj_z`, and `linear_attn.out_proj`, plus gated-attention
`q_proj`/`k_proj`/`v_proj`/`o_proj`. Start with ordinary LoRA in a supported
dtype. Do not introduce QLoRA for its own sake. Use quantization only if
necessary and proven on this AMD stack. Native Transformers/TRL generation is
acceptable. vLLM is optional and only if it works.

If GPU training is unavailable: finish verifier, oracle, dataset, and CPU
smoke work; document the exact failing layer; do not claim full training
completed; do not silently change the research question.

Declared fallback ladder, all labeled in the README:

1. Preferred: Qwen3.5-0.8B-Base, LoRA, GRPO, GPU.
2. Same model, CPU or reduced-memory smoke, if GPU kernels fail.
3. Only if hybrid backward/LoRA is unsupported: a predeclared dense fallback
   such as `Qwen/Qwen3-0.6B-Base` or `Qwen/Qwen2.5-0.5B`, labeled as a
   compatibility experiment.

Never silently switch models. Never use an instruct model for the core no-SFT
claim.

## Gate G. Untouched-model baseline

Replace the placeholder evaluator. Evaluate the exact untouched base
checkpoint before RL. Use a frozen evaluation suite and record the model
revision.

Run at least:

- Greedy / deterministic generation
- A sampling configuration comparable to the intended GRPO rollout

Sampling matters because a base model may have zero greedy accuracy while
occasionally discovering rewarded trajectories.

Save every completion as JSONL. Required fields are listed in `AGENTS.md`.
Report exact solve rate, legal-expression rate, target-hit-but-illegal rate,
invalid-expression rate, no-answer rate, truncation rate, length distribution,
3-number vs 4-number, solvable-only, source held-out, fresh-task, pass@1, and
pass@k when multiple samples are generated on purpose. Do not substitute
pass@k for pass@1. Include Wilson or bootstrap intervals where practical.

## Gate H. GRPO signal and staged runs

Use the current supported TRL `GRPOTrainer` API. Verify the pinned version's
effective-batch divisibility rule before changing the scaffold's
`per_device_train_batch_size=1`, `gradient_accumulation_steps=8`, and
`num_generations=4` settings. Current TRL documents an effective batch of 8,
which is divisible by 4. Change the settings only if the installed API or a
measured training constraint requires it, and record the reason.

Core run: Qwen3.5-0.8B-Base, no supervised solutions, binary exact reward,
fixed split, fixed prompt, fixed seed, LoRA unless evidence supports another
setup.

Before a long run, sample enough baseline rollouts to estimate whether
rewarded trajectories occur at all.

Log during GRPO: mean reward, reward variance, positive-completion fraction,
mixed-reward-group rate, all-zero groups, all-one groups, verifier failure
categories, completion length, truncation rate, and TRL KL/loss fields.

Stages:

- A. Integration smoke: one or two optimizer steps.
- B. Short diagnostic: about 25-50 steps. Inspect memory, completions, reward
  signal, and stability.
- C. Intermediate: about 100 steps, only if B is healthy.
- D. Demonstration: about 300 steps, only if C shows usable mixed groups.

Do not blindly run 300 steps if the experiment is broken. Save predetermined
checkpoints such as base, 50, 100, and 300 when they exist. Do not keep only
the most favorable checkpoint. Record actual optimizer steps, examples,
generated completions, tokens, wall-clock, peak VRAM, and config.

If almost every group receives identical rewards, document that there is no
useful group-relative signal. Continued training under zero reward variance
is not evidence against RL in general.

## Gate I. Sparse-reward contingency and controls

If binary reward yields almost no positive rollouts:

1. Verify the reward pipeline.
2. Verify that oracle-generated known-valid expressions score `1.0`.
3. Inspect completions manually.
4. Check completion length and truncation.
5. Increase sampling diversity only inside a documented range.
6. Separate formatting, syntax, arithmetic, and exploration failures.
7. Keep the binary-reward run as the primary result.

Do not silently introduce shaping. Secondary labeled ablations may include
small format reward, legal-expression reward, temperature, group size, or
completion length.

If hardware and time permit after the 0.8B-Base smoke path works, add one
cheap positive control, preferably `Qwen/Qwen3.5-0.8B` instruct, to distinguish
broken implementation from insufficient initial competence. Do not start a 3B
Countdown run before that. Never substitute a control for the original
question.

## Gate J. Learning, hacking, and plots

Evaluate the untouched base and every reported checkpoint on the same frozen
tasks. Compare short, intermediate, and final checkpoints, plus an extra seed
if feasible. Evaluate source held-out, fresh tasks, 3 vs 4 numbers, solvable
tasks, and target buckets. Use identical inference settings for paired
comparisons. Report bootstrap intervals for solve rate and paired improvement
when practical. Do not claim that a few extra successes prove learning if
uncertainty is large.

Audit reward hacking automatically: reused inputs, omitted inputs, unsupported
operators, decimals, malformed tags, injected prose, parser corners, division
by zero, target written without a valid expression, huge completions,
truncation, and memorized source tasks. Test adversarial strings against the
verifier. If source-held-out improves and fresh tasks collapse, discuss
memorization or source dependence. If legality improves and exact accuracy
does not, report formatting or constraint-following, not reasoning.

Generate plots only from saved experiment data. Useful figures: reward vs
step, positive rate vs step, mixed-group rate vs step, exact solve rate,
legal-expression rate, 3 vs 4 numbers, source vs fresh, failure categories,
completion length, solvable vs impossible. Every chart must be reproducible
from a committed script. Do not hand-edit values.

## Gate K. README, review, and publish

The README must tell a concise, honest research story. Include question, why
this experiment, design, verifier, hardware, baseline, training, signal
diagnostics, results, what changed, failure analysis, limitations,
reproduction commands, and related work (TinyZero, DeepSeek-R1 / DeepSeekMath
where relevant, TRL, Qwen, the Countdown dataset). Replace success-oriented
language if the result is negative.

Run unit tests, adversarial verifier tests, oracle tests, ruff, install test,
smoke tests that exist, `git diff --check`, and a secrets scan. GitHub Actions
must stay CPU-only. Review the public GitHub page: public repo, accurate
description, README render, license, CI, links, no fake results, no
credentials, no private paths.

Use focused Conventional Commits and push intentional commits as gates pass.
Completion requires `local main == origin/main` and a clean working tree.

## Decision rules

- Environment fails: document the exact failing layer and the strongest
  working path.
- Model loads but LoRA fails: diagnose architecture and PEFT before changing
  the model.
- GRPO runs with all-zero rewards: treat sparse exploration as an observation.
  Verify the pipeline. Do not rewrite the core reward.
- Formatting improves, exact solve rate does not: report formatting.
- Source-held-out improves, fresh tasks do not: report limited or
  source-specific generalization.
- Both improve: report improved Countdown arithmetic-search performance with
  uncertainty.
- Shaped reward succeeds: label it an ablation.
- Instruct/larger control succeeds while 0.8B-Base fails: evidence that the
  implementation can learn when the policy has enough initial competence, and
  that this 0.8B base setup may be below the tested threshold.

## Final report

When stopping, report: repo URL, final SHA, commits created, important files,
exact environment, commands actually run, test results, verifier and oracle
results, dataset audit, split and leakage checks, baseline, GRPO runs,
held-out eval, fresh-task eval, sparse-reward and mixed-group diagnostics,
reward-hacking findings, controls/ablations, limitations, blockers,
justified conclusions, unjustified conclusions, and the single best next
experiment.

Do not claim completion of any step without authoritative evidence. The
objective is a technically sound, reproducible experiment that explains what
happened.

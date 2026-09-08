# Reproducing the experiment

This runbook recreates the saved bounded CPU fallback. It does not turn that
fallback into a GPU experiment or claim that it is a full held-out evaluation.
Use a new directory for each rerun so the committed 2026-09-07 evidence stays
unchanged.

The full sequence below was rerun on 2026-09-08. It reproduced the source
audit and split counts, completed preflight and one CPU GRPO step, and produced
80 base plus 80 adapter evaluation records. After excluding run-specific
timestamps, IDs, and paths, every stable scored field matched the 2026-09-07
JSONL. See [the recheck report](../artifacts/rechecks/2026-09-08/report.md).

## 1. Create the environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
python -m pip install -e '.[dev,train]'

pytest
ruff check .
```

For the intended GPU path, install a ROCm Torch build compatible with the
machine instead of the CPU wheel. Do not install into the system Python or the
llama.cpp environment.

## 2. Choose a fresh evidence directory

```bash
RUN_DIR=artifacts/rechecks/2026-09-08
mkdir -p "$RUN_DIR"
```

The recheck data JSONL is ignored because it is a rebuildable cache. The small
audit, manifest, configurations, summaries, diagnostics, and completions are
safe to review and may be committed when they reflect an intentional rerun.

## 3. Rebuild the fixed tasks and record the machine

```bash
python -m countdown_grpo.prepare_data \
  --data-dir "$RUN_DIR/data" \
  --revision 408f70d177020686d34a56bba5952feb45aaaee4 \
  --seed 42 --fresh-size 256 --smoke-size 8

python -m countdown_grpo.environment \
  --output "$RUN_DIR/environment.json"
```

The data step checks source-task solvability with the independent oracle,
canonicalizes `(target, sorted nums)`, and rejects cross-split leakage. Oracle
witnesses never reach prompts or persisted task rows.

## 4. Check the base model and LoRA targets

```bash
python -m countdown_grpo.preflight \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --device cpu \
  --output "$RUN_DIR/preflight.json"
```

For GPU use `--device cuda` only after the hardware checks in
[hardware.md](hardware.md) pass. The preflight inspects the live module tree;
do not replace its selected hybrid attention LoRA targets with a generic list.

## 5. Run the untouched-base baseline

```bash
python -m countdown_grpo.evaluate \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir "$RUN_DIR/data" --splits source_test fresh_test --limit 8 \
  --output "$RUN_DIR/base.jsonl" --summary-output "$RUN_DIR/base.summary.json" \
  --experiment-id qwen35-08b-base-cpu-recheck-s42 --seed 42 \
  --samples-per-task 4 --max-new-tokens 16 --device cpu
```

This produces 80 records: greedy plus four samples for each of eight source
and eight fresh tasks. It is intentionally too small to estimate final-test
performance. It does, however, check the prompt, extraction, JSONL schema,
and exact binary scorer on a fixed model revision.

## 6. Run the one-step GRPO smoke and paired adapter evaluation

```bash
python -m countdown_grpo.train_grpo \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir "$RUN_DIR/data" --train-file smoke_train.jsonl --max-steps 1 \
  --max-completion-length 16 --seed 42 --allow-cpu \
  --output-dir outputs/qwen35-08b-grpo-cpu-recheck \
  --evidence-dir "$RUN_DIR/grpo" \
  --experiment-id qwen35-08b-base-cpu-smoke-recheck-s42

python -m countdown_grpo.evaluate \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --adapter-path outputs/qwen35-08b-grpo-cpu-recheck/final-adapter \
  --data-dir "$RUN_DIR/data" --splits source_test fresh_test --limit 8 \
  --output "$RUN_DIR/adapter.jsonl" --summary-output "$RUN_DIR/adapter.summary.json" \
  --experiment-id qwen35-08b-base-cpu-smoke-adapter-recheck-s42 --seed 42 \
  --samples-per-task 4 --max-new-tokens 16 --device cpu
```

The smoke must retain the base checkpoint, no supervised solutions, the
integer-only verifier, and the 0/1 exact reward. It is an integration check
only. Do not start a longer run unless the saved diagnostics include mixed
reward groups.

## 7. Generate the report from saved files

```bash
python -m countdown_grpo.report \
  --audit "$RUN_DIR/data/source_audit.json" \
  --manifest "$RUN_DIR/data/source_split_manifest.json" \
  --environment "$RUN_DIR/environment.json" \
  --baseline "$RUN_DIR/base.summary.json" \
  --smoke-attempt "$RUN_DIR/grpo/attempt.json" \
  --smoke-diagnostics "$RUN_DIR/grpo/grpo-diagnostics.jsonl" \
  --adapter "$RUN_DIR/adapter.summary.json" \
  --markdown-output "$RUN_DIR/report.md" \
  --svg-output "$RUN_DIR/summary.svg"
```

The report generator reads saved artifacts; it does not reconstruct metrics
from prose or accept hand-entered values.

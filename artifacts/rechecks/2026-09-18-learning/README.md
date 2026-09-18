# Learning study — 2026-09-18

Starting commit: `de1fe13a758c40316f169ba91e4ba944a6cae18c`.
The design is `docs/amendment-2026-09-18.md`.
All probes use source-dev until confirmation settings are frozen.
`environment.json` records hardware, versions, and occupancy.
The local model retains the weight-hash and metadata caveat in
`../2026-09-08-gpu-followup/instruct-control-manifest.json`.

First probe commands, run from the project root:

```bash
export C_INCLUDE_PATH="$PWD/.rocm-python-headers/usr/include/python3.11:$PWD/.rocm-python-headers/usr/include"
export HF_HUB_OFFLINE=1
.venv-rocm/bin/python -m countdown_grpo.environment \
  --output artifacts/rechecks/2026-09-18-learning/environment.json
.venv-rocm/bin/python -m countdown_grpo.evaluate \
  --model .control-model --revision instruct-local-weight-sha256-04b1c301231dd422 \
  --data-dir artifacts/data --splits source_dev --limit 16 \
  --generation-modes greedy sample --samples-per-task 4 --max-new-tokens 128 \
  --temperature 1.0 --top-p 0.95 --device cuda --attn-implementation eager \
  --use-chat-template --disable-thinking --experiment-id instruct-dev-chat-l128-s42 \
  --output artifacts/rechecks/2026-09-18-learning/instruct-dev-l128.jsonl
```

This probe began before evaluator compute logging was added. Its records have
timestamps, but no exact process start/end measurement is claimed. Subsequent
probes record command, wall time, completion tokens, and GPU memory in summaries.

## Supervised branch

The official instruct tokenizer download was blocked by the browser security
policy. `control-metadata-audit.json` establishes that local instruct weights
match the official weights but its tokenizer does not. Neither control probe
is a valid official-instruct competence result.

The named supervised branch uses the correctly cached base revision. It is
not a continuation of the original no-SFT experiment. Prepare train-only
targets, then run fresh diagnostics:

```bash
.venv-rocm/bin/python -m countdown_grpo.prepare_supervised \
  --input artifacts/data/source_train.jsonl \
  --output artifacts/rechecks/2026-09-18-learning/data/supervised-train.jsonl \
  --manifest artifacts/rechecks/2026-09-18-learning/supervised-data-manifest.json \
  --size 4096 --seed 20260918
.venv-rocm/bin/python -m countdown_grpo.train_sft \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --train-file artifacts/rechecks/2026-09-18-learning/data/supervised-train.jsonl \
  --output-dir outputs/learning-sft-diagnostic-s42 \
  --evidence-dir artifacts/rechecks/2026-09-18-learning/sft-diagnostic-s42 \
  --max-steps 64 --seed 42
```

For a rerun, replace evidence/output directories rather than overwriting them.
The failed `sft-integration-s42` attempt stopped before an optimizer step because
the label audit detected a tokenization boundary mismatch. The corrected
`sft-integration-tokenized-s42` attempt completed one optimizer step. Explicit
labels mask prompt tokens and supervise only the answer and termination token.
The 64-step diagnostic starts again from untouched base weights, not that
integration adapter. Neither an integration pass nor decreasing training loss
establishes unseen-task solving gains.

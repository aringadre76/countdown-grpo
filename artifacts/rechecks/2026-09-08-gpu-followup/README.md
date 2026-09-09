# Train/dev signal follow-up

This directory records the predeclared diversity checks after the 25-step base
diagnostic. All runs used `Qwen/Qwen3.5-0.8B-Base`, seed 42, the locked prompt
and binary reward, `source_dev` only, and eager attention on the RX 7900 XTX.

- `dev-l32-t1.jsonl`: 32-token completions at temperature 1.0.
- `dev-l64-t13.jsonl`: 64-token completions at temperature 1.3.
- `dev-l128-t18.jsonl`: 128-token completions at temperature 1.8.
- `dev-exploration-summary.json`: exact counts and the stopping decision.
- `instruct-control-blocker.json`: historical cache/download blocker, retained
  as immutable evidence.
- `instruct-control-manifest.json`: observed weight hash, metadata provenance,
  preflight, baseline, and one-step control-smoke references.
- `instruct-*.summary.json` and `instruct-*.jsonl`: separately labeled
  instruct-control evaluations. The 32-token chat-template/no-thinking run
  produced 0/80 exact solves, with 39 truncated and 34 unsupported outputs.
- `instruct-grpo-smoke/`: one GPU GRPO step using the same binary verifier;
  both reward groups were all zero.

Each probe produced 32 records, 0 exact rewards, 0 positive tasks, and 0 mixed
tasks. The base branch therefore stops before another GRPO diagnostic. The
instruct control is now available locally, but it did not produce an exact
solve in the recorded frozen evaluation. It remains a control and must not
replace the base/no-SFT result.

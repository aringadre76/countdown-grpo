# Codex `/goal` text

Status: Current
Last updated: 2026-09-07

Paste the block below after `/goal` in Codex. Start Codex in
`/home/robot/countdown-grpo`. The repository protocol is deliberately longer
than the goal and must be re-read at each major gate.

```text
/goal Complete the countdown-grpo project end-to-end in /home/robot/countdown-grpo (https://github.com/aringadre76/countdown-grpo). Do not create another repo.

This is an execution goal, not a planning request. First read AGENTS.md and docs/experiment-protocol.md, then work through every applicable gate in order. Continue across turns and after compaction; re-read those files after compaction, resuming, or a long GPU step. Do not stop after an audit, plan, documentation change, unit tests, baseline, or smoke test.

Research question: can Qwen/Qwen3.5-0.8B-Base, with no supervised Countdown solutions, improve held-out Countdown solving through TRL GRPO with a binary exact reward? Treat this as a lower-bound test motivated by TinyZero. A carefully measured negative result is valid.

Preserve unrelated changes and never invent results. Keep the base checkpoint, task, and binary reward. Do not silently use SFT, an instruct model, shaped reward, a different model, or a different task. Do not tune on the final test split. Never execute generated text as Python or call eval/exec. Do not call reward increases emergent reasoning.

Implement and verify the complete path: integer-only exact verifier with adversarial tests; independent solvability oracle whose witnesses never reach model prompts; leakage-safe canonical (target, sorted nums) train/dev/test splits; real untouched-base baseline JSONL; observed environment manifest; legal pinned-TRL configuration; real GRPO smoke; bounded diagnostic; frozen held-out and fresh-task evaluation whenever the environment permits; README, plots, and reports generated only from saved evidence; CPU CI without Torch/TRL.

The contract is binary +, -, *, / with parentheses, every supplied number exactly once, exact rational checking, and integer intermediate/final values. Keep it identical across verifier, oracle, reward, splits, evaluation, and README. Report source solvability under it.

If a gate fails, diagnose it, try safe in-scope alternatives, and record the exact command and output. If GPU training is genuinely blocked, finish the strongest runnable partial project and clearly mark the experiment incomplete; do not silently substitute a model or claim a smoke test proves learning. If training works, continue beyond smoke through the documented diagnostic and final evaluation. Stop only when the protocol’s completion evidence is present or an external blocker has been exhausted and documented.

Known risks: evaluate.py is a placeholder; inspect live Qwen3.5 named_modules() before choosing LoRA targets; verify the pinned TRL effective-batch rule; use a project venv; inspect GPU occupancy before training and do not modify the llama.cpp/ROCm inference stack; bitsandbytes and vLLM are optional.

Record commands, versions, seeds, configs, completions, reward-group statistics, failure categories, checkpoints, and actual compute. Use focused Conventional Commits and push intentional changes. Finish with local main equal to origin/main and an empty status, while preserving and reporting any authorized unrelated change.
```

# Codex `/goal` continuation

Status: Current
Last updated: 2026-09-23

The original base/no-SFT/GRPO experiment remains historical evidence. The
separately authorized three-seed SFT confirmation improved over base. The
predeclared 50-step SFT-initialized binary-GRPO follow-up is also complete, but
did not demonstrate additional gains over matching SFT checkpoints on both
source and fresh tasks. Read both saved reports and `docs/next-steps.md` before
starting another experiment.

If continuing the project, paste this block after `/goal` in Codex and start in
`/home/robot/countdown-grpo`:

```text
Continue the Countdown study in /home/robot/countdown-grpo. Do not create another repository. First read AGENTS.md, docs/experiment-protocol.md, the applicable dated amendments, docs/README.md, and the latest reports. After compaction, resume, or any long GPU step, reread AGENTS.md and docs/experiment-protocol.md. Preserve all saved evidence.

The objective is to test whether a model can improve exact Countdown solving over its untouched initialization, generalize to unseen source and independently generated fresh tasks, replicate across at least three training seeds, and show arithmetic-search improvement beyond formatting. The original Qwen/Qwen3.5-0.8B-Base no-SFT binary-reward GRPO branch remains a separately reported negative result. Alternative methods are authorized only as named studies that follow the amendment and do not get credited to the original claim.

The supervised-initialization study and its binary-GRPO follow-up are complete. The latter ran 50 steps for seeds 42, 43, and 44 from the frozen SFT adapters; all train-only gates passed. Its full base/SFT/GRPO confirmation and paired report are under artifacts/rechecks/2026-09-22-sft-init-grpo/. The report found no demonstrated repeatable GRPO improvement over SFT on both suites. Do not extend that frozen design or tune on its confirmation tasks. Any next experiment requires a new dated amendment, method, budget, dev-only checkpoint rule, and new frozen confirmation tasks.

Keep the task contract identical: binary +, -, *, / and parentheses; every supplied number exactly once; exact rational checking; integer intermediate and final values. Keep the independent verifier and solvability oracle. Oracle solutions may be train-side targets only in an explicitly supervised branch and must never appear in prompts. Never execute generated text as Python or call eval/exec. Do not claim general or emergent reasoning.

Use the project ROCm venv, inspect GPU occupancy, preserve the llama.cpp stack, and record commands, versions, hardware, seeds, configs, raw completions, failure categories, checkpoints, memory, and actual compute. CPU CI stays free of Torch/TRL. Generate reports and plots only from saved evidence. Update the routed documentation, commit intentionally, push, and verify local main equals origin/main with an empty status.
```

# Codex `/goal` continuation

Status: Current
Last updated: 2026-09-23

The original base/no-SFT/GRPO experiment remains historical evidence. The
separately authorized three-seed SFT confirmation improved over base. The
predeclared 50-step SFT-initialized binary-GRPO follow-up is also complete, but
did not demonstrate additional gains over matching SFT checkpoints on both
source and fresh tasks. A separately named exact-verifier-filtered inference
study is frozen, its new source/fresh suites are prepared, and its four GPU
evaluations are pending. Read the dated amendment, frozen design, and
`docs/next-steps.md` before continuing.

If continuing the project, paste this block after `/goal` in Codex and start in
`/home/robot/countdown-grpo`:

```text
Continue the Countdown study in /home/robot/countdown-grpo. Do not create another repository. First read AGENTS.md, docs/experiment-protocol.md, the applicable dated amendments, docs/README.md, and the latest reports. After compaction, resume, or any long GPU step, reread AGENTS.md and docs/experiment-protocol.md. Preserve all saved evidence.

The objective is to test whether a model can improve exact Countdown solving over its untouched initialization, generalize to unseen source and independently generated fresh tasks, replicate across at least three training seeds, and show arithmetic-search improvement beyond formatting. The original Qwen/Qwen3.5-0.8B-Base no-SFT binary-reward GRPO branch remains a separately reported negative result. Alternative methods are authorized only as named studies that follow the amendment and do not get credited to the original claim.

The supervised-initialization study and its binary-GRPO follow-up are complete. The latter ran 50 steps for seeds 42, 43, and 44 from the frozen SFT adapters; all train-only gates passed. Its full base/SFT/GRPO confirmation and paired report are under artifacts/rechecks/2026-09-22-sft-init-grpo/. The report found no demonstrated repeatable GRPO improvement over SFT on both suites. Do not extend that frozen design or tune on its confirmation tasks. Any next experiment requires a new dated amendment, method, budget, dev-only checkpoint rule, and new frozen confirmation tasks.

The current frozen follow-up is governed by docs/amendment-2026-09-23-verifier-search.md and artifacts/rechecks/2026-09-23-verifier-search/frozen-design.json. It tests ordered exact-verifier filtering across one greedy plus eight sampled outputs on 256 new source and 256 new fresh tasks, comparing the untouched base with all three fixed SFT adapters. The task manifest and environment preflight are saved; run all four pinned ROCm evaluations under the six-hour cap, then generate the paired pass@1/pass@4/pass@8 and search@1/@5/@9 report. This is an inference-time solver result, not training or a GRPO claim. Preserve the frozen task IDs, candidate order, generation settings, and bootstrap plan; do not tune on these tasks.

Keep the task contract identical: binary +, -, *, / and parentheses; every supplied number exactly once; exact rational checking; integer intermediate and final values. Keep the independent verifier and solvability oracle. Oracle solutions may be train-side targets only in an explicitly supervised branch and must never appear in prompts. Never execute generated text as Python or call eval/exec. Do not claim general or emergent reasoning.

Use the project ROCm venv, inspect GPU occupancy, preserve the llama.cpp stack, and record commands, versions, hardware, seeds, configs, raw completions, failure categories, checkpoints, memory, and actual compute. CPU CI stays free of Torch/TRL. Generate reports and plots only from saved evidence. Update the routed documentation, commit intentionally, push, and verify local main equals origin/main with an empty status.
```

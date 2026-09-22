# Codex `/goal` continuation

Status: Current
Last updated: 2026-09-22

The original base/no-SFT/GRPO experiment remains historical evidence. The
separately authorized supervised-initialization extension has completed its
three-seed frozen confirmation and met the five study criteria on the recorded
runs. See `artifacts/rechecks/2026-09-22-confirmation/report-final/` and
`docs/next-steps.md` before starting more work.

If continuing the project, paste this block after `/goal` in Codex and start in
`/home/robot/countdown-grpo`:

```text
Continue the Countdown study in /home/robot/countdown-grpo. Do not create another repository. First read AGENTS.md, docs/experiment-protocol.md, docs/amendment-2026-09-18.md, and the latest confirmation report. After compaction, resume, or any long GPU step, reread AGENTS.md and docs/experiment-protocol.md. Preserve all saved evidence.

The objective is to test whether a model can improve exact Countdown solving over its untouched initialization, generalize to unseen source and independently generated fresh tasks, replicate across at least three training seeds, and show arithmetic-search improvement beyond formatting. The original Qwen/Qwen3.5-0.8B-Base no-SFT binary-reward GRPO branch remains a separately reported negative result. Alternative methods are authorized only as named studies that follow the amendment and do not get credited to the original claim.

The supervised-initialization study is complete: three 512-step LoRA seeds and paired frozen confirmation are saved under artifacts/rechecks/2026-09-22-confirmation/. Do not tune or select checkpoints on those confirmation suites. The next proposed study is binary GRPO from the supervised checkpoint, beginning with a train-only positive/mixed-reward signal gate and a matched comparison against the SFT checkpoint. Before running it, freeze the method, budget, gate, checkpoint rule, confirmation sample, and analysis in a dated amendment. Continue only if the gate passes and the authorized budget allows it.

Keep the task contract identical: binary +, -, *, / and parentheses; every supplied number exactly once; exact rational checking; integer intermediate and final values. Keep the independent verifier and solvability oracle. Oracle solutions may be train-side targets only in an explicitly supervised branch and must never appear in prompts. Never execute generated text as Python or call eval/exec. Do not claim general or emergent reasoning.

Use the project ROCm venv, inspect GPU occupancy, preserve the llama.cpp stack, and record commands, versions, hardware, seeds, configs, raw completions, failure categories, checkpoints, memory, and actual compute. CPU CI stays free of Torch/TRL. Generate reports and plots only from saved evidence. Update the routed documentation, commit intentionally, push, and verify local main equals origin/main with an empty status.
```

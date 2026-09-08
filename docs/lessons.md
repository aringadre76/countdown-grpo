# Lessons from the first CPU run

Status: current as of 2026-09-08.

- Treat the 2026-09-07 result as a CPU integration result. WSL exposed no
  usable GPU to PyTorch, so it cannot answer the intended RX 7900 XTX training
  question.
- An all-zero GRPO group has no within-group reward comparison. The one-step
  smoke therefore checks integration, not learning and not RL's general
  usefulness.
- Preserve the base checkpoint, task, prompt, and exact binary reward when
  investigating sparse reward. Changing more than one of them loses the
  lower-bound comparison.
- The exact verifier is a security and validity boundary. Do not use `eval`,
  accept partial expressions, or permit fractional intermediate values in the
  primary reward.
- The target-hit-but-illegal metric is now computed by the exact parser in
  diagnostic mode. It may inspect number-use and integer-intermediate failures
  but can never affect reward.
- Inspect Qwen3.5's live module names before attaching LoRA. The model includes
  Gated DeltaNet projections in addition to the usual attention projections.
- Keep raw evaluation JSONL and run metadata. Generate summaries, reports, and
  plots from those files rather than manually transcribing results.
- Put reruns in a new evidence directory. This keeps the original baseline and
  smoke artifacts available for comparison.
- The 2026-09-08 recheck reproduced every stable scored field in the two
  80-record evaluations. JSONL hashes changed only because timestamps,
  experiment IDs, and output paths are intentionally recorded per run.
- When issuing WSL commands through a Windows shell wrapper, escape shell
  variables before handing them to WSL. An unescaped temporary-venv variable
  created untracked root `bin/`, `lib/`, `lib64`, and `pyvenv.cfg` files during
  the 2026-09-08 validation; those exact files were verified and removed.

For the proposed GPU recovery and experiment sequence, read
[next-steps.md](next-steps.md). For the exact commands, read
[reproduction.md](reproduction.md).

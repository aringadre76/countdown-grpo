# Lessons and durable constraints

Status: current as of 2026-09-08.

## GPU recovery

- A successful llama.cpp HIP stack does not prove that PyTorch training works.
  The project needed a separate ROCm Torch/Triton environment and an observed
  `torch.cuda.is_available()` check.
- In WSL2, `rocminfo` was the useful HSA probe. `rocm-smi` still reported that
  `amdgpu` was not initialized, while Torch saw and used the RX 7900 XTX.
- Triton needed `Python.h`; extracting the supplied `libpython3.11-dev`
  package into the ignored project directory fixed the build without changing
  the system or llama.cpp environment.
- Transformers SDPA failed with `CUDA error: invalid argument` on this ROCm
  build. Explicit eager attention is the recorded compatibility workaround.

## Experiment signal

- The one-step GPU smoke was healthy enough to produce a mixed group: mean
  reward 0.125 and mixed-group rate 0.5.
- The 25-step diagnostic had 200 rollouts, one positive completion, one mixed
  step, and 24 all-zero steps. That is sparse reward with no sustained usable
  group-relative signal, not evidence that RL is impossible.
- Three follow-up source-dev probes (32 tokens at temperature 1.0; 64 at 1.3;
  128 at 1.8) produced 96 more rollouts with no exact rewards and no mixed
  tasks. Increasing length and temperature in this predeclared range did not
  recover a usable binary-reward signal.
- A positive-control model must be downloaded separately. If it is not cached,
  record the blocker instead of silently falling back to the base checkpoint.
- The adapter produced one source-held-out exact completion and zero fresh
  completions. The corresponding base completion used an unsupported `=`
  suffix, so the cautious interpretation is possible output-format change;
  arithmetic-search improvement is unestablished.
- Do not call a lower loss or a single exact completion emergent reasoning.
  Keep source, fresh, pass@1, pass@k, failure categories, and raw completions
  separate.

## Engineering

- Keep the exact verifier as a validity boundary. Never execute generated text;
  reject parser tricks, omitted/reused numbers, fractional intermediates, and
  unsupported syntax.
- Inspect Qwen3.5's live module tree. Its Gated DeltaNet projections require
  `in_proj_qkv`, `in_proj_z`, and `out_proj` targets in addition to the usual
  attention projections.
- Record every diagnostic line, not only the last line. Reports now aggregate
  all saved steps and generate plots from those records.
- Keep CPU tests independent of Torch/TRL. Training extras no longer install a
  generic Torch wheel that could overwrite a device-matched ROCm build.
- Put every rerun in a new evidence directory; keep original evidence
  immutable and do not commit weights, caches, or credentials.

For exact commands and the gated follow-up, read
[reproduction.md](reproduction.md) and [next-steps.md](next-steps.md).

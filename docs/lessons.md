# Lessons and durable constraints

Status: current as of 2026-09-18.

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
- A positive-control model must be downloaded separately. The first cache check
  was blocked and is retained as `instruct-control-blocker.json`; after the
  user supplied the weight, the control ran from an ignored local directory.
  Record the weight hash and any metadata reuse rather than implying a complete
  model snapshot.
- Qwen3.5 instruct emits chat-style thinking/tool-call patterns unless its chat
  template is used with thinking disabled. In the recorded 32-token control,
  this still yielded 0/80 exact solves and mostly truncation or unsupported
  syntax. The one-step control GRPO smoke had two all-zero reward groups.
- The adapter produced one source-held-out exact completion and zero fresh
  completions. The corresponding base completion used an unsupported `=`
  suffix, so the cautious interpretation is possible output-format change;
  arithmetic-search improvement is unestablished.
- Do not call a lower loss or a single exact completion emergent reasoning.
  Keep source, fresh, pass@1, pass@k, failure categories, and raw completions
  separate.

## Engineering

- Check supervised label masks before training. Tokenizing `prompt + answer`
  can merge tokens across the boundary and cause TRL to mask part of the answer.
  The integration audit caught this before an optimizer step. Separately
  tokenize the generation prompt and answer and supply explicit prompt masks.

- Matching model configs does not establish matching tokenizer metadata. The
  2026-09-18 Git-blob comparison verified that base and instruct tokenizer files
  differ despite identical model configs and shared vocab/merges. Compare exact
  metadata against the pinned official revision before interpreting a control.
  For files stored in Git LFS, compare content SHA256 to the official LFS OID,
  not local bytes to the Git pointer hash. The recovered instruct tokenizer
  uses EOS 248046 while the historical reused base tokenizer uses 248044.
  Incorrect termination metadata can invalidate completion/clipping diagnostics.

- Historical GRPO callback truncation defaults were false even when TRL's
  clipped ratio was one. Those callback fields are unreliable: use the saved
  trainer clipping statistics for old runs. New callbacks record token IDs and
  derive termination from EOS/pad tokens, matching installed TRL; missing
  termination data is unknown. Clipped completions receive zero primary reward.
  Historical training could reward a clipped expression while evaluation
  rejected clipped output. Preserve those records, but do not describe their
  positive reward as evidence of consistent end-to-end exact solving.
- Pass `enable_thinking` directly to Transformers `apply_chat_template`.
  The previous nested keyword is not the documented tokenizer API. The cached
  0.8B template produced the same prefix in the observed comparison, so this
  API correction alone is not evidence of a changed model response.

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
  Verify this in a fresh core/dev environment: an older CPU fallback venv may
  contain training libraries even though the current dependency extra does not.
- Put every rerun in a new evidence directory; keep original evidence
  immutable and do not commit weights, caches, or credentials.

For exact commands and the gated follow-up, read
[reproduction.md](reproduction.md) and [next-steps.md](next-steps.md).

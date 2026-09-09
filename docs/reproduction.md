# Reproducing the experiment

This runbook describes the recorded GPU run and the lightweight CPU checks.
Use a new evidence directory for each rerun. Keep model weights, Hugging Face
caches, adapters, and virtual environments outside Git.

## 1. CPU test environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
ruff check .
```

CI runs this same dependency class. It does not install Torch, TRL, or model
weights.

## 2. GPU environment used for the evidence

Create a separate environment and install the device-matched wheels supplied
for this machine. Replace `<DOWNLOADS>` with the directory containing the
files; do not commit them.

```bash
python3.11 -m venv .venv-rocm
source .venv-rocm/bin/activate
python -m pip install --upgrade pip
python -m pip install <DOWNLOADS>/rocm_bootstrap-0.1.0-py3-none-any.whl
python -m pip install <DOWNLOADS>/rocm_sdk_core-7.14.0-py3-none-linux_x86_64.whl \
  <DOWNLOADS>/rocm_sdk_libraries-7.14.0-py3-none-linux_x86_64.whl \
  <DOWNLOADS>/rocm_sdk_device_gfx1100-7.14.0-py3-none-linux_x86_64.whl
python -m pip install <DOWNLOADS>/torch-2.13.0+rocm7.14.0-cp311-cp311-linux_x86_64.whl \
  <DOWNLOADS>/amd_torch_device_gfx1100-2.13.0+rocm7.14.0-cp311-cp311-linux_x86_64.whl \
  <DOWNLOADS>/triton-3.8.0+git4cff872c.rocm7.14.0-cp311-cp311-linux_x86_64.whl
python -m pip install -e '.[train]'
python -m pip check
```

The recorded Torch/ROCm environment reported Torch `2.13.0+rocm7.14.0`, HIP
`7.14.60850`, and an AMD Radeon RX 7900 XTX. Verify before running:

```bash
rocminfo
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.version.hip)'
```

On this machine Triton also needed Python headers. Extract the supplied
`libpython3.11-dev_3.11.15-1+jammy1_amd64.deb` into an ignored directory (do
not install it system-wide), then export the include path:

```bash
mkdir -p .rocm-python-headers
dpkg-deb --extract <DOWNLOADS>/libpython3.11-dev_3.11.15-1+jammy1_amd64.deb .rocm-python-headers
export C_INCLUDE_PATH="$PWD/.rocm-python-headers/usr/include/python3.11:$PWD/.rocm-python-headers/usr/include"
```

The Debian package SHA256 was
`1adc394918add62fb6e497382046d67b66d4d73cc887cb8be597d9e623db98ad`.
`rocm-smi` may still report `amdgpu` not initialized under WSL; use the
Torch device check and `rocminfo` as the training observations. See
[hardware.md](hardware.md) for the full manifest and caveats.

## 3. Prepare data and record the environment

```bash
source .venv-rocm/bin/activate
python -m countdown_grpo.prepare_data \
  --data-dir artifacts/data --revision 408f70d177020686d34a56bba5952feb45aaaee4 \
  --seed 42 --fresh-size 256 --smoke-size 8
python -m countdown_grpo.environment \
  --output artifacts/rechecks/<DATE>/environment-rocm-torch.json
```

The data step canonicalizes `(target, sorted nums)`, checks the independent
oracle, and never puts oracle witnesses in prompts. Use the existing frozen
`artifacts/data` for an exact replay rather than regenerating test tasks.

## 4. Preflight and the attention compatibility flag

The default SDPA implementation failed with `CUDA error: invalid argument` on
this ROCm build. The successful runs explicitly selected eager attention:

```bash
export HF_HUB_OFFLINE=1
export C_INCLUDE_PATH="$PWD/.rocm-python-headers/usr/include/python3.11:$PWD/.rocm-python-headers/usr/include"
python -m countdown_grpo.preflight \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --device cuda --attn-implementation eager \
  --output artifacts/rechecks/<DATE>/qwen-preflight-eager.json
```

The preflight must show a Qwen3.5 conditional-generation model, the observed
hybrid LoRA targets, a successful forward/generation/backward pass, and changed
adapter weights after an optimizer step.

## 5. Untouched-base baseline

```bash
python -m countdown_grpo.evaluate \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir artifacts/data --splits source_test fresh_test --limit 8 \
  --generation-modes greedy sample --samples-per-task 4 --max-new-tokens 16 \
  --device cuda --attn-implementation eager --seed 42 \
  --experiment-id qwen35-08b-base-rocm-baseline-eager-s42 \
  --output artifacts/rechecks/<DATE>/base-eager.jsonl \
  --summary-output artifacts/rechecks/<DATE>/base-eager.summary.json
```

This bounded command produces 80 records: 40 source-held-out and 40 fresh.
It is not a full final-test estimate.

## 6. GRPO smoke and diagnostic

```bash
python -m countdown_grpo.train_grpo \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir artifacts/data --train-file smoke_train.jsonl --max-steps 1 \
  --max-completion-length 16 --seed 42 --device cuda \
  --attn-implementation eager \
  --output-dir outputs/qwen35-08b-grpo-rocm-eager-smoke-s42 \
  --evidence-dir artifacts/rechecks/<DATE>/grpo-smoke-eager-s42 \
  --experiment-id qwen35-08b-base-rocm-eager-smoke-s42

python -m countdown_grpo.train_grpo \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir artifacts/data --train-file smoke_train.jsonl --max-steps 25 \
  --max-completion-length 16 --seed 42 --device cuda \
  --attn-implementation eager \
  --output-dir outputs/qwen35-08b-grpo-rocm-eager-diagnostic-s42 \
  --evidence-dir artifacts/rechecks/<DATE>/grpo-diagnostic-eager-s42 \
  --experiment-id qwen35-08b-base-rocm-eager-diagnostic-s42
```

The effective generation batch is `1 × 8 = 8`, divisible by four generations
per prompt. Continue beyond the diagnostic only when mixed reward groups remain
usable; all-zero groups are not evidence that RL cannot work.

## 7. Paired adapter evaluation and report

```bash
python -m countdown_grpo.evaluate \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --adapter-path outputs/qwen35-08b-grpo-rocm-eager-diagnostic-s42/final-adapter \
  --data-dir artifacts/data --splits source_test fresh_test --limit 8 \
  --generation-modes greedy sample --samples-per-task 4 --max-new-tokens 16 \
  --device cuda --attn-implementation eager --seed 42 \
  --experiment-id qwen35-08b-base-rocm-eager-diagnostic-adapter-s42 \
  --output artifacts/rechecks/<DATE>/adapter-eager-diagnostic.jsonl \
  --summary-output artifacts/rechecks/<DATE>/adapter-eager-diagnostic.summary.json

python -m countdown_grpo.report \
  --audit artifacts/data/source_audit.json \
  --manifest artifacts/data/source_split_manifest.json \
  --environment artifacts/rechecks/<DATE>/environment-rocm-torch.json \
  --baseline artifacts/rechecks/<DATE>/base-eager.summary.json \
  --smoke-attempt artifacts/rechecks/<DATE>/grpo-diagnostic-eager-s42/attempt.json \
  --smoke-diagnostics artifacts/rechecks/<DATE>/grpo-diagnostic-eager-s42/grpo-diagnostics.jsonl \
  --adapter artifacts/rechecks/<DATE>/adapter-eager-diagnostic.summary.json \
  --markdown-output artifacts/rechecks/<DATE>/report.md \
  --svg-output artifacts/rechecks/<DATE>/summary.svg
```

Reports and plots are generated only from JSONL, manifests, and trainer
artifacts. Never hand-edit a metric into Markdown.

## 8. Train/dev-only signal follow-up

After the recorded 25-step diagnostic, the next-step gate was checked without
touching source-test or fresh-test tasks:

```bash
python -m countdown_grpo.evaluate \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir artifacts/data --splits source_dev --limit 8 \
  --generation-modes sample --samples-per-task 4 --max-new-tokens 32 \
  --temperature 1.0 --top-p 0.95 --device cuda \
  --attn-implementation eager \
  --experiment-id qwen35-08b-base-rocm-dev-explore-l32-t1-s42 \
  --output artifacts/rechecks/<DATE>/dev-l32-t1.jsonl \
  --summary-output artifacts/rechecks/<DATE>/dev-l32-t1.summary.json

python -m countdown_grpo.evaluate \
  --model Qwen/Qwen3.5-0.8B-Base \
  --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --data-dir artifacts/data --splits source_dev --limit 8 \
  --generation-modes sample --samples-per-task 4 --max-new-tokens 64 \
  --temperature 1.3 --top-p 0.95 --device cuda \
  --attn-implementation eager \
  --experiment-id qwen35-08b-base-rocm-dev-explore-l64-t13-s42 \
  --output artifacts/rechecks/<DATE>/dev-l64-t13.jsonl \
  --summary-output artifacts/rechecks/<DATE>/dev-l64-t13.summary.json
```

Both probes produced 0/32 exact rewards, 0/8 positive tasks, and 0/8 mixed
tasks. The saved decision is
`artifacts/rechecks/2026-09-08-gpu-followup/dev-exploration-summary.json`;
under the protocol this stops the binary-reward branch before another
diagnostic.

The follow-up also ran a third source-dev probe with `--max-new-tokens 128`
and `--temperature 1.8`; it likewise produced 0/32 exact rewards and no mixed
tasks. This stops the base binary-reward branch under the protocol.

## 9. Separately labeled instruct control

The optional control was run only after the base branch was frozen. The user
downloaded the single 1.75 GB instruct safetensors file through Brave. Its
observed SHA256 is recorded in
`artifacts/rechecks/2026-09-08-gpu-followup/instruct-control-manifest.json`.
The local control directory reuses the cached Qwen3.5 config/tokenizer metadata
after the official instruct config page was checked; this is explicitly not
claimed as a complete reproducible model snapshot. The directory is ignored by
Git and the private download path is not committed.

The control uses the same verifier and frozen source/fresh tasks, but the
Qwen3.5 chat template is enabled with thinking disabled:

```bash
python -m countdown_grpo.evaluate \
  --model .control-model \
  --revision instruct-local-weight-sha256-04b1c301231dd422 \
  --data-dir artifacts/data --splits source_test fresh_test --limit 8 \
  --generation-modes greedy sample --samples-per-task 4 --max-new-tokens 32 \
  --temperature 1.0 --top-p 0.95 --device cuda --attn-implementation eager \
  --use-chat-template --disable-thinking \
  --experiment-id qwen35-08b-instruct-control-rocm-chat-nothink-l32-s42 \
  --output artifacts/rechecks/<DATE>/instruct-chat-nothink-l32.jsonl \
  --summary-output artifacts/rechecks/<DATE>/instruct-chat-nothink-l32.summary.json
```

Observed control result: 0/80 exact solves (0/40 source and 0/40 fresh),
39/80 truncated records, and 34/80 unsupported-syntax records. A one-step
control GRPO smoke also completed on the RX 7900 XTX; it produced two all-zero
reward groups and zero positive completions. These are control observations,
not a replacement for the locked base-model result.

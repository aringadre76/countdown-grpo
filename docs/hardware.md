# Hardware and observed environment

The frozen three-seed confirmation ran on the RX 7900 XTX on 2026-09-22. The
fresh post-run observations are in
[environment.json](../artifacts/rechecks/2026-09-22-confirmation/environment.json).
The earlier 2026-09-18 environment snapshot remains in
[the learning evidence](../artifacts/rechecks/2026-09-18-learning/environment.json).

Status: GPU path verified again during the 2026-09-22 full confirmation runs in
WSL2. Torch executed a tensor kernel and loaded the pinned Qwen base plus each
adapter on the intended AMD Radeon RX 7900 XTX.

| Item | Observed value | Evidence |
| --- | --- | --- |
| CPU | 12th Gen Intel Core i7-12700K; 6 vCPUs visible to WSL | `environment-rocm-torch.json` |
| WSL kernel | 6.18.33.2-microsoft-standard-WSL2 | `environment-rocm-torch.json` |
| Python | 3.11.15 | `environment-rocm-torch.json` |
| Torch | 2.13.0+rocm7.14.0 | `environment-rocm-torch.json` |
| ROCm/HIP | 7.14.60850 | `environment-rocm-torch.json` |
| GPU | AMD Radeon RX 7900 XTX, `gfx1100`, 96 CUs | `rocminfo` and Torch |
| Transformers / TRL / PEFT | 5.16.1 / 1.12.0 / 0.20.0 | manifest |
| Torch CUDA view | available, one device | manifest |
| llama.cpp server | no matching process observed | manifest |

The current manifest captured `gfx1100`, 25,708,240,896 bytes total device
memory, and 23,255,171,072 bytes free at capture (21.66 GiB free of 23.94 GiB).
Its on-device `sum(arange(1024) ** 2)` probe returned `357389824.0` on
`cuda:0`. The three adapter confirmations each recorded a peak of
1,892,670,976 allocated bytes and 1,981,808,640 reserved bytes. Those runs
each completed 2,560 generations without truncation.

The detailed GPU-recovery manifest is
[environment-rocm-torch.json](../artifacts/rechecks/2026-09-08-gpu-recovery/environment-rocm-torch.json);
the current post-confirmation snapshot is
[environment.json](../artifacts/rechecks/2026-09-22-confirmation/environment.json).
`rocminfo` returned successfully and reported the GPU. `rocm-smi` printed
`Driver not initialized (amdgpu not found in modules)`, which is a WSL
userspace caveat here rather than proof that the GPU was unavailable: Torch
reported one available AMD device, ran a tensor operation, and completed the
Qwen evaluations on `cuda:0`. The manifest preserves both the diagnostic and
the successful Torch/HSA evidence.

## Two environments

- `.venv` is the ignored CPU/test environment. It keeps CI and verifier tests
  independent of Torch and TRL.
- `.venv-rocm` is the ignored GPU environment. It contains the AMD-provided
  Torch, Triton, ROCm SDK, Transformers, TRL, PEFT, and Accelerate packages.
  The project does not commit wheels, model weights, adapters, or caches.

Triton initially required Python development headers. The supplied
`libpython3.11-dev` Debian package was extracted into an ignored project-local
directory; no system package manager or llama.cpp installation was changed.
The exact package hash and setup are recorded in the GPU reproduction notes.

## Compatibility notes

The default Transformers SDPA path failed on this ROCm build with
`CUDA error: invalid argument`; a minimal bf16 SDPA reproduction failed the
same way. The successful runs explicitly use `--attn-implementation eager`.
This changes the attention implementation, not the checkpoint, task, reward,
or LoRA target selection, and is recorded in every GPU run configuration.

Before another run, record a fresh manifest, inspect the named llama.cpp
server and GPU occupancy, run the Qwen preflight, and retain the exact output.
Do not infer training compatibility from an unrelated HIP inference stack.

## SFT-initialized GRPO follow-up

The frozen follow-up completed on the same WSL2 ROCm environment. Its saved
training `run-config.json` files identify the device as AMD Radeon RX 7900 XTX,
`gfx1100`, with Torch `2.13.0+rocm7.14.0` and HIP `7.14.60850`. The seven
confirmation evaluator summaries record `device: cuda`; each adapter run had
1,892,670,976 bytes peak allocated and 1,981,808,640 bytes peak reserved, and
all six adapter evaluations completed 2,560 records without truncation. The
base confirmation used 1,851,776,512 peak allocated bytes.

Observed follow-up GPU process time was 18,229.856 seconds (5.064 hours) out
of the predeclared six-hour cap. That total includes seven evaluators, three
train-only signal probes, one integration attempt, and three 50-step GRPO
processes. The generated accounting is in
[`comparison.json`](../artifacts/rechecks/2026-09-22-sft-init-grpo/report-final/comparison.json);
the per-command versions, exact arguments, start/finish times, and memory are
in the adjacent summary and attempt JSON files. No llama.cpp process was
observed before the runs. As in the earlier snapshot, `rocm-smi` can still
print `Driver not initialized (amdgpu not found in modules)` in WSL; the
Torch `cuda` device records and successful model executions are the evidence
of this path, not the `rocm-smi` output alone.

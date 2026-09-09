# Hardware and observed environment

Status: GPU path verified on 2026-09-08 in WSL2. The intended device and the
device used for the recorded run are both an AMD Radeon RX 7900 XTX.

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

The canonical GPU manifest is
[environment-rocm-torch.json](../artifacts/rechecks/2026-09-08-gpu-recovery/environment-rocm-torch.json).
`rocminfo` returned successfully and reported the GPU. `rocm-smi` printed
`Driver not initialized (amdgpu not found in modules)`, which is a WSL
userspace caveat here rather than proof that the GPU was unavailable: Torch
successfully ran a bf16 autograd test and the Qwen preflight, baseline, and
GRPO jobs on `cuda:0`. The recorded run therefore uses Torch and `rocminfo`
as the authoritative training observations, while retaining the `rocm-smi`
output for completeness.

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

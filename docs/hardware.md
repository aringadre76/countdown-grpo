# Hardware and observed environment

Status: observed CPU fallback on 2026-09-07 and rechecked on 2026-09-08.

The intended training machine is an AMD Radeon RX 7900 XTX system. That is the
target hardware for the core experiment, not evidence that a GPU run occurred.
The saved run was executed inside WSL2, where the GPU was unavailable to
PyTorch.

| Item | Observed value | Evidence |
| --- | --- | --- |
| CPU | 12th Gen Intel Core i7-12700K | `lscpu` in the manifest |
| CPUs visible to WSL | 6 logical CPUs | `lscpu` in the manifest |
| OS | Linux 5.15.167.4-microsoft-standard-WSL2 | manifest |
| Python | 3.11.15 | manifest |
| Torch | 2.14.0+cpu | manifest |
| Accelerate / Datasets | 1.14.0 / 5.0.1 | manifest |
| Transformers / TRL / PEFT | 5.16.1 / 1.12.0 / 0.20.0 | manifest |
| CUDA visible to Torch | no; device count 0 | manifest |
| ROCm probe | `Driver not initialized (amdgpu not found in modules)` | `rocm-smi` in manifest |
| llama.cpp server | none found | `pgrep -af 'llama.*server'` in manifest |

The canonical record is
[artifacts/environment/cpu-train-manifest.json](../artifacts/environment/cpu-train-manifest.json).
It stores command output, package versions, and only the names of relevant
environment variables. It does not infer a device model from the host plan.
The 2026-09-08 recheck produced the same device availability and ROCm probe:
[artifacts/rechecks/2026-09-08/environment.json](../artifacts/rechecks/2026-09-08/environment.json).

## Before attempting GPU training

1. Use the project virtual environment, not the llama.cpp or system Python
   environment.
2. Record a fresh manifest with `python -m countdown_grpo.environment`.
3. Confirm `rocm-smi` can see the device and that `torch.cuda.is_available()`
   is true before loading the model.
4. Inspect the named llama server and VRAM use. Stop only that named server if
   it blocks the experiment; do not change its ROCm installation.
5. Run `countdown_grpo.preflight` on the base model before a GRPO step.

Until those observations exist, describe every result as a CPU fallback, not
as RX 7900 XTX training.

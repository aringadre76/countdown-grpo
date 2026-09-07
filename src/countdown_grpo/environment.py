"""Capture a redacted, observed machine and dependency manifest for an experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_manifest() -> dict[str, Any]:
    """Observe, rather than infer, the supported training environment."""

    gpu_commands = [_run(["rocm-smi"]), _run(["nvidia-smi"])]
    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "hip_version": torch.version.hip,
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available():
            torch_info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        torch_info = {"version": None, "cuda_available": False, "reason": "torch not installed"}

    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": _run(["lscpu"]),
        "gpu_probes": gpu_commands,
        "torch": torch_info,
        "dependencies": {
            package: _version(package)
            for package in ("accelerate", "datasets", "peft", "torch", "transformers", "trl", "bitsandbytes")
        },
        "relevant_environment_variable_names": sorted(
            key for key in os.environ if key.startswith(("ROCR_", "HSA_", "HIP_", "CUDA_"))
        ),
        "named_llama_processes": _run(["pgrep", "-af", "llama.*server"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = capture_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

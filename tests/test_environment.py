from countdown_grpo.environment import capture_manifest


def test_manifest_includes_hsa_runtime_probe(monkeypatch):
    def fake_run(command):
        return {"command": command, "returncode": 0, "stdout": "observed", "stderr": ""}

    monkeypatch.setattr("countdown_grpo.environment._run", fake_run)
    manifest = capture_manifest()

    assert [probe["command"] for probe in manifest["gpu_probes"]] == [
        ["rocminfo"],
        ["rocm-smi"],
        ["nvidia-smi"],
    ]

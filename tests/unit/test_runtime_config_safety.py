"""Runtime config safety regressions that do not require model weights."""

from __future__ import annotations

from pathlib import Path

from hermes_katana.scabbard.config import ScabbardConfig
from hermes_katana.scabbard.embedder import _move_module_to_device, _resolve_default_device


def test_device_override_accepts_known_values(monkeypatch):
    monkeypatch.setenv("HERMES_KATANA_DEVICE", "cpu")
    assert _resolve_default_device() == "cpu"

    monkeypatch.setenv("HERMES_KATANA_DEVICE", "CUDA:0")
    assert _resolve_default_device() == "cuda:0"


def test_device_override_invalid_falls_back_to_cpu(monkeypatch):
    monkeypatch.setenv("HERMES_KATANA_DEVICE", "not-a-device")
    assert _resolve_default_device() == "cpu"


def test_accelerator_move_failure_falls_back_to_cpu():
    class FlakyModule:
        def __init__(self) -> None:
            self.devices: list[str] = []

        def to(self, device: str):
            self.devices.append(device)
            if device == "cuda":
                raise RuntimeError("cuda oom")
            return self

    module = FlakyModule()
    moved, device = _move_module_to_device(module, "cuda", module_name="test")

    assert moved is module
    assert device == "cpu"
    assert module.devices == ["cuda", "cpu"]


def test_model_version_resolves_from_onnx_sibling():
    cfg = ScabbardConfig.katana_v14(
        model_path="training/checkpoints/katana_v14/onnx_int8",
        backend="onnx_int8",
    )
    assert cfg.model_version == "katana_v14"


def test_model_version_falls_back_for_unknown_path(tmp_path: Path):
    custom = tmp_path / "custom_model" / "best"
    cfg = ScabbardConfig.production(model_path=str(custom))
    assert cfg.model_version == "custom_model"

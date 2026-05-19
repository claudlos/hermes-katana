"""Shared CLI helper functions extracted from the main command module."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

from hermes_katana.runtime_artifacts import verify_runtime_artifact_manifest
from rich import box
from rich.panel import Panel

VERSION = "2.0.0"
HERMETIC_ML_READY_ENV = "HERMES_KATANA_REQUIRE_ML_READY"
_TRUTHY = {"1", "true", "yes", "on"}


def version_string() -> str:
    """Build the version banner string."""
    return (
        f"HermesKatana v{VERSION}  |  Python {platform.python_version()}  |  {platform.system()} {platform.machine()}"
    )


def build_proxy_url(host: str, port: int) -> str:
    """Build the proxy URL string for display and environment export."""
    return f"http://{host}:{port}"


def _package_probe(module_name: str, distribution: str | None = None) -> dict[str, Any]:
    distribution_name = distribution or module_name
    installed = importlib.util.find_spec(module_name) is not None
    version: str | None = None
    if installed:
        try:
            version = metadata.version(distribution_name)
        except metadata.PackageNotFoundError:
            version = None
    return {
        "installed": installed,
        "version": version,
    }


def _training_models_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "training" / "models"


def _training_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "training" / "data"


def _path_has_entries(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _model_metric(path: Path, key: str) -> float:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return 0.0
    try:
        import json

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _is_deberta_model_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "best").exists() or (path / "final").exists() or any(path.glob("*.onnx")))


def _resolve_deberta_artifact() -> dict[str, Any]:
    models_dir = _training_models_dir()
    override = os.environ.get("HERMES_KATANA_DEBERTA_MODEL_DIR")

    if override:
        candidate = Path(override).expanduser()
        if _is_deberta_model_dir(candidate):
            return {
                "models_dir": str(models_dir),
                "override": override,
                "artifact_dir": str(candidate),
                "checkpoint_dir": str(candidate / "best")
                if (candidate / "best").exists()
                else str(candidate / "final")
                if (candidate / "final").exists()
                else None,
                "onnx_path": str(candidate / "deberta_v3_small_cpu.onnx")
                if (candidate / "deberta_v3_small_cpu.onnx").exists()
                else None,
                "ready": True,
                "error": None,
            }
        return {
            "models_dir": str(models_dir),
            "override": override,
            "artifact_dir": None,
            "checkpoint_dir": None,
            "onnx_path": None,
            "ready": False,
            "error": f"HERMES_KATANA_DEBERTA_MODEL_DIR points to invalid artifact: {candidate}",
        }

    candidates: list[Path] = []
    for path in [models_dir / "deberta_v3_small_katana", *sorted(models_dir.glob("deberta_v3_small_katana*"))]:
        if not path.is_dir():
            continue
        candidates.append(path)
        candidates.extend(child for child in path.iterdir() if child.is_dir())

    valid = [path for path in dict.fromkeys(candidates) if _is_deberta_model_dir(path)]
    valid.sort(
        key=lambda path: (
            path.name == "deberta_v3_small_katana",
            _model_metric(path, "final_test_f1"),
            (path / "deberta_v3_small_cpu.onnx").exists(),
            (path / "best").exists(),
            path.stat().st_mtime,
        ),
        reverse=True,
    )

    if not valid:
        return {
            "models_dir": str(models_dir),
            "override": override,
            "artifact_dir": None,
            "checkpoint_dir": None,
            "onnx_path": None,
            "ready": False,
            "error": f"No DeBERTa-v3-small artifact found under {models_dir}",
        }

    selected = valid[0]
    checkpoint_dir = selected / "best" if (selected / "best").exists() else selected / "final"
    onnx_path = selected / "deberta_v3_small_cpu.onnx"
    return {
        "models_dir": str(models_dir),
        "override": override,
        "artifact_dir": str(selected),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir.exists() else None,
        "onnx_path": str(onnx_path) if onnx_path.exists() else None,
        "ready": True,
        "error": None,
    }


def _collect_semantic_status() -> dict[str, Any]:
    try:
        from hermes_katana.scanner.semantic_recall import semantic_backend_status

        semantic = dict(semantic_backend_status())
    except Exception as exc:
        return {
            "backend": "unavailable",
            "reason": str(exc) or exc.__class__.__name__,
            "full_backend_ready": False,
            "missing": ["semantic backend import failed"],
        }

    semantic["full_backend_ready"] = semantic.get("backend") in {"contrastive", "zvec_quantized"}
    missing: list[str] = []
    semantic_index_dir = Path(str(semantic.get("semantic_index_dir", "")))
    contrastive_model_dir = Path(str(semantic.get("contrastive_model_dir", "")))
    quantized_model_dir = Path(str(semantic.get("quantized_model_dir", "")))

    if not _path_has_entries(semantic_index_dir):
        missing.append(f"semantic index missing at {semantic_index_dir}")
    if not _path_has_entries(contrastive_model_dir) and not _path_has_entries(quantized_model_dir):
        missing.append("no semantic model artifact available (contrastive or quantized)")

    semantic["missing"] = missing
    return semantic


def _collect_scabbard_status(packages: dict[str, Any]) -> dict[str, Any]:
    from hermes_katana.scabbard.config import ScabbardConfig

    models_dir = _training_models_dir()
    tfidf_path = models_dir / "tfidf_vectorizer.pkl"
    fusion_path = models_dir / "fusion_xgb.json"
    centroids_128 = models_dir / "attack_centroids_128d.npz"
    centroids_768 = models_dir / "attack_centroids.npz"
    centroid_path = centroids_128 if centroids_128.exists() else centroids_768 if centroids_768.exists() else None

    zvec_dir = models_dir / "zvec_quantized-20260408T061203Z-3-001" / "zvec_quantized"
    zvec_backbone = zvec_dir / "backbone_fp32"
    zvec_projector = zvec_dir / "projector_fp32.pt"
    zvec_tokenizer = zvec_dir / "tokenizer"

    missing: list[str] = []
    if not tfidf_path.is_file():
        missing.append(f"missing TF-IDF vectorizer at {tfidf_path}")
    if not fusion_path.is_file():
        missing.append(f"missing fusion model at {fusion_path}")
    if centroid_path is None:
        missing.append("missing attack centroids under training/models")
    if not _path_has_entries(zvec_backbone):
        missing.append(f"missing zvec backbone at {zvec_backbone}")
    if not zvec_projector.is_file():
        missing.append(f"missing zvec projector at {zvec_projector}")
    if not _path_has_entries(zvec_tokenizer):
        missing.append(f"missing zvec tokenizer at {zvec_tokenizer}")

    missing_dependencies = [
        name
        for name in ("numpy", "joblib", "xgboost", "torch", "transformers", "sentence_transformers")
        if not packages[name]["installed"]
    ]
    minimal_profile_ready = ScabbardConfig.minimal_runtime_ready()
    standard_profile_ready = not missing_dependencies and not missing

    return {
        "models_dir": str(models_dir),
        "tfidf_path": str(tfidf_path),
        "fusion_path": str(fusion_path),
        "centroid_path": str(centroid_path) if centroid_path is not None else None,
        "zvec_dir": str(zvec_dir),
        "minimal_profile_ready": minimal_profile_ready,
        "standard_profile_ready": standard_profile_ready,
        "recommended_profile": ScabbardConfig.default_runtime_profile(),
        "missing": missing,
        "missing_dependencies": missing_dependencies,
    }


def _collect_protectai_status(packages: dict[str, Any]) -> dict[str, Any]:
    try:
        from hermes_katana.scanner.protectai_gate import ProtectAIGate

        protectai_model = ProtectAIGate.MODEL_ID
    except Exception:
        protectai_model = "ProtectAI/deberta-v3-base-prompt-injection-v2"

    return {
        "model_id": protectai_model,
        "dependencies_ready": packages["transformers"]["installed"],
        "note": "ProtectAI is lazy-loaded at runtime; dependency readiness does not guarantee model warm-load success.",
    }


def _collect_eval_status(
    deberta: dict[str, Any],
    scabbard: dict[str, Any],
    semantic: dict[str, Any],
    artifact_manifest: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    if not deberta["ready"]:
        blockers.append(str(deberta["error"] or "DeBERTa artifact missing"))
    if not deberta["dependencies_ready"]:
        blockers.append("missing DeBERTa runtime dependencies (torch/transformers)")
    if not deberta["cpu_inference_ready"]:
        warnings.append("ONNX CPU inference not ready; runtime will fall back to PyTorch")
    if not scabbard["standard_profile_ready"]:
        blockers.extend(scabbard["missing"])
        blockers.extend(
            f"missing Scabbard runtime dependency: {name}" for name in scabbard.get("missing_dependencies", [])
        )
    if not semantic.get("full_backend_ready", False):
        warnings.append(f"semantic backend degraded: {semantic.get('reason', 'unknown reason')}")
        warnings.extend(semantic.get("missing", []))
    if not artifact_manifest["ready"]:
        blockers.extend(artifact_manifest["missing"])
        blockers.extend(artifact_manifest["mismatched"])
        blockers.extend(artifact_manifest["empty"])
        blockers.extend(artifact_manifest["errors"])

    return {
        "ready": not blockers and semantic.get("full_backend_ready", False),
        "blockers": blockers,
        "warnings": warnings,
    }


def collect_ml_runtime_status() -> dict[str, Any]:
    """Return a lightweight view of ML dependency and artifact readiness."""
    packages = {
        "numpy": _package_probe("numpy"),
        "torch": _package_probe("torch"),
        "transformers": _package_probe("transformers"),
        "onnxruntime": _package_probe("onnxruntime"),
        "sentence_transformers": _package_probe("sentence_transformers", "sentence-transformers"),
        "scikit_learn": _package_probe("sklearn", "scikit-learn"),
        "xgboost": _package_probe("xgboost"),
        "joblib": _package_probe("joblib"),
    }

    deberta = _resolve_deberta_artifact()
    deberta["dependencies_ready"] = all(packages[name]["installed"] for name in ("torch", "transformers"))
    deberta["cpu_inference_ready"] = deberta["ready"] and bool(
        packages["onnxruntime"]["installed"] and deberta["onnx_path"]
    )
    scabbard = _collect_scabbard_status(packages)
    semantic = _collect_semantic_status()
    protectai = _collect_protectai_status(packages)
    artifact_manifest = verify_runtime_artifact_manifest()
    eval_status = _collect_eval_status(deberta, scabbard, semantic, artifact_manifest)

    return {
        "packages": packages,
        "deberta": deberta,
        "scabbard": scabbard,
        "semantic": semantic,
        "protectai": protectai,
        "artifact_manifest": artifact_manifest,
        "eval": eval_status,
    }


def hermetic_ml_ready_required(config: dict[str, Any] | None = None) -> bool:
    """Return whether hermetic fail-closed ML readiness is required."""
    if config is not None and "require_ml_ready" in config:
        return bool(config["require_ml_ready"])
    return os.getenv(HERMETIC_ML_READY_ENV, "").strip().lower() in _TRUTHY


def enforce_hermetic_ml_readiness(config: dict[str, Any] | None = None) -> None:
    """Raise when hermetic mode requires a fully ready ML/runtime stack."""
    if not hermetic_ml_ready_required(config):
        return

    status = collect_ml_runtime_status()
    eval_status = status["eval"]
    if eval_status["ready"]:
        return

    reasons = [*eval_status["blockers"], *eval_status["warnings"]]
    if not reasons:
        reasons = ["ML runtime readiness requirements not satisfied"]
    detail = "; ".join(reasons[:4])
    raise RuntimeError(f"Hermetic ML readiness required ({HERMETIC_ML_READY_ENV}=1), but startup is degraded: {detail}")


def format_katana_status() -> Optional[Panel]:
    """Format a Rich panel showing Katana protection status."""
    try:
        from hermes_katana.config import load_config
        from hermes_katana.proxy import KatanaProxy

        lines = []
        lines.append("[bold green]HermesKatana Protection Active[/bold green]")
        lines.append(f"   Version: {VERSION}")

        checkout_root = os.environ.get("KATANA_CHECKOUT_ROOT")
        if checkout_root:
            lines.append(f"   Checkout: {checkout_root}")

        proxy_status = KatanaProxy().status()
        if proxy_status.get("running"):
            lines.append(
                "   Proxy: "
                + build_proxy_url(
                    str(proxy_status.get("host", proxy_status["config"]["host"])),
                    int(proxy_status.get("port", proxy_status["config"]["port"])),
                )
            )
        else:
            proxy_url = os.environ.get("KATANA_PROXY_URL")
            if proxy_url:
                lines.append(f"   Proxy: {proxy_url}")
            else:
                lines.append("   Proxy: [dim]not running[/dim]")

        config = load_config()
        runtime_policy_source = os.environ.get("KATANA_POLICY_SOURCE")
        policy_path = config.effective_policy_path()
        if runtime_policy_source:
            lines.append(f"   Policy: {runtime_policy_source}")
        elif policy_path is not None:
            lines.append(f"   Policy: custom file {policy_path}")
        else:
            preset = os.environ.get("KATANA_POLICY_PRESET", config.policy_preset)
            lines.append(f"   Policy: {preset}")

        ml_status = collect_ml_runtime_status()
        deberta = ml_status["deberta"]
        if deberta["ready"]:
            lines.append(f"   DeBERTa: ready ({Path(str(deberta['artifact_dir'])).name})")
        else:
            lines.append("   DeBERTa: [yellow]artifact/deps incomplete[/yellow]")

        scabbard = ml_status["scabbard"]
        lines.append("   Scabbard: " + ("standard-ready" if scabbard["standard_profile_ready"] else "degraded"))

        semantic = ml_status["semantic"]
        lines.append(f"   Semantic: {semantic.get('backend', 'unavailable')}")

        eval_status = ml_status["eval"]
        lines.append("   Eval readiness: " + ("ready" if eval_status["ready"] else "[yellow]partial[/yellow]"))
        if hermetic_ml_ready_required():
            lines.append("   Hermetic mode: fail-closed")

        return Panel(
            "\n".join(lines),
            title="[bold]Katana Security[/bold]",
            border_style="green",
            box=box.ROUNDED,
        )
    except Exception:
        return None


def check_command(name: str) -> tuple[bool, str]:
    """Check if a command is available on PATH."""
    path = shutil.which(name)
    if path is None:
        return False, "not found"
    try:
        result = subprocess.run(
            [name, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        version = result.stdout.strip().split("\n")[0] or result.stderr.strip().split("\n")[0]
        return True, version or "installed"
    except (subprocess.TimeoutExpired, OSError):
        return True, "installed (version check failed)"


def resolve_target(target: str | None) -> Path:
    """Resolve the target path, defaulting to current directory."""
    if target:
        return Path(target).resolve()
    return Path.cwd()


def load_policy_engine() -> tuple[Any, str]:
    """Load the active policy engine from persisted config or environment."""
    from hermes_katana.config import load_config
    from hermes_katana.policy import PolicyEngine

    config = load_config()
    policy_path = config.effective_policy_path()
    if policy_path is not None:
        return PolicyEngine.from_file(policy_path), f"custom file {policy_path}"

    preset = os.environ.get("KATANA_POLICY_PRESET", config.policy_preset)
    return PolicyEngine.with_defaults(preset), f"preset {preset}"


def open_vault(*, auto_create: bool) -> Any:
    """Open the vault backend."""
    from hermes_katana.vault import Vault

    return Vault(auto_create=auto_create)


def open_audit_trail() -> Any:
    """Open the default audit trail."""
    from hermes_katana.audit import AuditTrail

    return AuditTrail()

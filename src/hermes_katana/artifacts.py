"""Hugging Face artifact resolution and download helpers.

Large model/data artifacts live outside GitHub. This module keeps runtime
resolution explicit and offline-safe: nothing downloads unless the caller asks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MINILM_ONNX_REPO = "claudlos/hermes-katana-v15-distill-minilm-onnx"
DEFAULT_REVISION = "main"

MINILM_ONNX_REQUIRED_FILES = (
    "model.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
)

MINILM_ONNX_ALLOW_PATTERNS = (*MINILM_ONNX_REQUIRED_FILES, "README.md", "artifact_manifest.json")


class ArtifactError(RuntimeError):
    """Base artifact error."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact is missing and download was not requested."""


class ArtifactDownloadError(ArtifactError):
    """Raised when an explicit artifact download fails."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    repo_id: str
    repo_type: str
    revision: str
    required_files: tuple[str, ...]
    allow_patterns: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactStatus:
    spec: ArtifactSpec
    path: Path
    present: bool
    missing_files: tuple[str, ...]
    source: str


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def default_artifact_cache_dir() -> Path:
    """Return the default artifact cache directory."""
    override = os.environ.get("KATANA_ARTIFACT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return (base / "hermes-katana" / "artifacts").resolve()


def minilm_onnx_spec(repo_id: str | None = None, revision: str | None = None) -> ArtifactSpec:
    """Build the default MiniLM ONNX artifact spec with env overrides."""
    return ArtifactSpec(
        name="katana_v15_distill_minilm_onnx",
        repo_id=repo_id or os.environ.get("KATANA_HF_REPO_ID") or DEFAULT_MINILM_ONNX_REPO,
        repo_type="model",
        revision=revision or os.environ.get("KATANA_HF_REVISION") or DEFAULT_REVISION,
        required_files=MINILM_ONNX_REQUIRED_FILES,
        allow_patterns=MINILM_ONNX_ALLOW_PATTERNS,
    )


def artifact_path(spec: ArtifactSpec, target_dir: str | Path | None = None) -> Path:
    """Return the local directory where an artifact should live."""
    explicit = os.environ.get("KATANA_MINILM_ONNX_DIR") if spec.name == "katana_v15_distill_minilm_onnx" else None
    if explicit:
        return Path(explicit).expanduser().resolve()
    if target_dir:
        return Path(target_dir).expanduser().resolve()
    safe_repo = spec.repo_id.replace("/", "__")
    safe_rev = spec.revision.replace("/", "__")
    return default_artifact_cache_dir() / spec.name / safe_repo / safe_rev


def artifact_status(spec: ArtifactSpec | None = None, target_dir: str | Path | None = None) -> ArtifactStatus:
    """Inspect local artifact readiness without network access."""
    spec = spec or minilm_onnx_spec()
    path = artifact_path(spec, target_dir)
    missing = tuple(rel for rel in spec.required_files if not (path / rel).is_file())
    return ArtifactStatus(spec=spec, path=path, present=not missing, missing_files=missing, source="local")


def _download_with_huggingface_hub(spec: ArtifactSpec, target: Path, force: bool) -> Path | None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        return None

    token = os.environ.get("KATANA_HF_TOKEN") or os.environ.get("HF_TOKEN")
    snapshot_download(
        repo_id=spec.repo_id,
        repo_type=spec.repo_type,
        revision=spec.revision,
        local_dir=str(target),
        allow_patterns=list(spec.allow_patterns),
        token=token,
        force_download=force,
    )
    return target


def _download_with_hf_cli(spec: ArtifactSpec, target: Path) -> Path:
    hf = shutil.which("hf")
    if not hf:
        raise ArtifactDownloadError(
            "Install `huggingface_hub` (`pip install hermes-katana[hf]`) or the modern `hf` CLI."
        )

    cmd = [
        hf,
        "download",
        spec.repo_id,
        "--repo-type",
        spec.repo_type,
        "--revision",
        spec.revision,
        "--local-dir",
        str(target),
    ]
    for pattern in spec.allow_patterns:
        cmd.extend(["--include", pattern])
    env = os.environ.copy()
    if os.environ.get("KATANA_HF_TOKEN") and not env.get("HF_TOKEN"):
        env["HF_TOKEN"] = os.environ["KATANA_HF_TOKEN"]
    subprocess.run(cmd, check=True, env=env)
    return target


def download_artifact(
    spec: ArtifactSpec | None = None,
    target_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> ArtifactStatus:
    """Explicitly download an artifact from Hugging Face and validate it."""
    spec = spec or minilm_onnx_spec()
    target = artifact_path(spec, target_dir)
    target.mkdir(parents=True, exist_ok=True)
    try:
        if _download_with_huggingface_hub(spec, target, force) is None:
            _download_with_hf_cli(spec, target)
    except subprocess.CalledProcessError as exc:
        raise ArtifactDownloadError(f"hf download failed with exit code {exc.returncode}") from exc
    except Exception as exc:
        if isinstance(exc, ArtifactDownloadError):
            raise
        raise ArtifactDownloadError(str(exc)) from exc

    status = artifact_status(spec, target)
    if not status.present:
        raise ArtifactDownloadError(f"Downloaded artifact is incomplete; missing: {', '.join(status.missing_files)}")
    return status


def resolve_minilm_onnx(
    *,
    download: bool | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    target_dir: str | Path | None = None,
) -> Path:
    """Return a ready MiniLM ONNX artifact directory.

    No network access occurs unless ``download=True`` or
    ``KATANA_ARTIFACT_AUTO_DOWNLOAD=1``.
    """
    spec = minilm_onnx_spec(repo_id=repo_id, revision=revision)
    status = artifact_status(spec, target_dir)
    if status.present:
        return status.path
    should_download = _truthy(os.environ.get("KATANA_ARTIFACT_AUTO_DOWNLOAD")) if download is None else download
    if should_download:
        return download_artifact(spec, target_dir).path
    raise ArtifactNotFoundError(
        "MiniLM ONNX artifact is missing. Run `katana artifacts download`, set "
        "KATANA_MINILM_ONNX_DIR, or set KATANA_ARTIFACT_AUTO_DOWNLOAD=1. "
        f"Missing files: {', '.join(status.missing_files)}"
    )

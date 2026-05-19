"""Hugging Face artifact resolution and download helpers.

Large model/data artifacts live outside GitHub. This module keeps runtime
resolution explicit and offline-safe: nothing downloads unless the caller asks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_ARTIFACT_MODEL = "minilm"
DEFAULT_MINILM_ONNX_REPO = "claudlos/hermes-katana-v15-distill-minilm-onnx"
DEFAULT_V15_LARGE_REPO = "claudlos/hermes-katana-v15-large"
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

V15_LARGE_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
)

V15_LARGE_ALLOW_PATTERNS = (*V15_LARGE_REQUIRED_FILES, "README.md", "artifact_manifest.json")


class ArtifactError(RuntimeError):
    """Base artifact error."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact is missing and download was not requested."""


class ArtifactDownloadError(ArtifactError):
    """Raised when an explicit artifact download fails."""


class UnknownArtifactError(ArtifactError):
    """Raised when a requested artifact model is not registered."""


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    repo_id: str
    repo_type: str
    revision: str
    required_files: tuple[str, ...]
    allow_patterns: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    display_name: str = ""
    size_label: str = ""
    role: str = ""
    profile: str = ""
    path_env_var: str | None = None
    repo_env_var: str | None = None
    revision_env_var: str | None = None
    interactive_default: bool = False
    requires_confirmation: bool = False


@dataclass(frozen=True)
class ArtifactStatus:
    spec: ArtifactSpec
    path: Path
    present: bool
    missing_files: tuple[str, ...]
    source: str


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_BASE_ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        name="katana_v15_distill_minilm_onnx",
        aliases=("minilm", "small", "fast_cpu", "katana_v15_minilm", "katana_v15_distill_minilm_onnx"),
        display_name="Katana v15 MiniLM ONNX",
        repo_id=DEFAULT_MINILM_ONNX_REPO,
        repo_type="model",
        revision=DEFAULT_REVISION,
        required_files=MINILM_ONNX_REQUIRED_FILES,
        allow_patterns=MINILM_ONNX_ALLOW_PATTERNS,
        size_label="~88 MB",
        role="default fast CPU Scabbard classifier",
        profile="fast_cpu",
        path_env_var="KATANA_MINILM_ONNX_DIR",
        repo_env_var="KATANA_HF_REPO_ID",
        revision_env_var="KATANA_HF_REVISION",
        interactive_default=True,
        requires_confirmation=False,
    ),
    ArtifactSpec(
        name="katana_v15_large",
        aliases=("large", "v15_large", "katana_v15_large", "deberta", "teacher"),
        display_name="Katana v15 Large",
        repo_id=DEFAULT_V15_LARGE_REPO,
        repo_type="model",
        revision=DEFAULT_REVISION,
        required_files=V15_LARGE_REQUIRED_FILES,
        allow_patterns=V15_LARGE_ALLOW_PATTERNS,
        size_label="large",
        role="optional high-accuracy local Scabbard model",
        profile="paranoid_local",
        path_env_var="KATANA_V15_LARGE_DIR",
        repo_env_var="KATANA_V15_LARGE_HF_REPO_ID",
        revision_env_var="KATANA_V15_LARGE_HF_REVISION",
        interactive_default=False,
        requires_confirmation=True,
    ),
)


def _normalize_model_name(model: str | None) -> str:
    return (model or DEFAULT_ARTIFACT_MODEL).strip().lower().replace("-", "_")


def _base_artifact_spec(model: str | None = None) -> ArtifactSpec:
    normalized = _normalize_model_name(model)
    for spec in _BASE_ARTIFACT_SPECS:
        names = {spec.name.lower(), *(alias.lower().replace("-", "_") for alias in spec.aliases)}
        if normalized in names:
            return spec
    choices = ", ".join(spec.aliases[0] for spec in _BASE_ARTIFACT_SPECS)
    raise UnknownArtifactError(f"Unknown artifact model {model!r}. Choose one of: {choices}")


def artifact_spec(
    model: str | None = None,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
) -> ArtifactSpec:
    """Return a registered artifact spec with env/argument overrides applied."""
    spec = _base_artifact_spec(model)
    resolved_repo = repo_id or (os.environ.get(spec.repo_env_var) if spec.repo_env_var else None) or spec.repo_id
    resolved_revision = (
        revision or (os.environ.get(spec.revision_env_var) if spec.revision_env_var else None) or spec.revision
    )
    return replace(spec, repo_id=resolved_repo, revision=resolved_revision)


def artifact_specs() -> tuple[ArtifactSpec, ...]:
    """Return all registered artifact specs with env overrides applied."""
    return tuple(artifact_spec(spec.name) for spec in _BASE_ARTIFACT_SPECS)


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
    return artifact_spec("minilm", repo_id=repo_id, revision=revision)


def v15_large_spec(repo_id: str | None = None, revision: str | None = None) -> ArtifactSpec:
    """Build the optional large v15 artifact spec with env overrides."""
    return artifact_spec("large", repo_id=repo_id, revision=revision)


def artifact_path(spec: ArtifactSpec, target_dir: str | Path | None = None) -> Path:
    """Return the local directory where an artifact should live."""
    explicit = os.environ.get(spec.path_env_var) if spec.path_env_var else None
    if explicit:
        return Path(explicit).expanduser().resolve()
    if target_dir:
        return Path(target_dir).expanduser().resolve()
    safe_repo = spec.repo_id.replace("/", "__")
    safe_rev = spec.revision.replace("/", "__")
    return default_artifact_cache_dir() / spec.name / safe_repo / safe_rev


def artifact_status(
    spec: ArtifactSpec | str | None = None,
    target_dir: str | Path | None = None,
) -> ArtifactStatus:
    """Inspect local artifact readiness without network access."""
    spec = _coerce_spec(spec)
    path = artifact_path(spec, target_dir)
    missing = tuple(rel for rel in spec.required_files if not (path / rel).is_file())
    if spec.path_env_var and os.environ.get(spec.path_env_var):
        source = spec.path_env_var
    elif target_dir:
        source = "target-dir"
    else:
        source = "cache"
    return ArtifactStatus(spec=spec, path=path, present=not missing, missing_files=missing, source=source)


def _coerce_spec(spec: ArtifactSpec | str | None) -> ArtifactSpec:
    if spec is None:
        return artifact_spec(DEFAULT_ARTIFACT_MODEL)
    if isinstance(spec, str):
        return artifact_spec(spec)
    return spec


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
    spec: ArtifactSpec | str | None = None,
    target_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> ArtifactStatus:
    """Explicitly download an artifact from Hugging Face and validate it."""
    spec = _coerce_spec(spec)
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
    return resolve_artifact(spec, download=download, target_dir=target_dir)


def resolve_v15_large(
    *,
    download: bool | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    target_dir: str | Path | None = None,
) -> Path:
    """Return a ready optional large v15 artifact directory."""
    spec = v15_large_spec(repo_id=repo_id, revision=revision)
    return resolve_artifact(spec, download=download, target_dir=target_dir)


def resolve_artifact(
    spec: ArtifactSpec | str | None = None,
    *,
    download: bool | None = None,
    target_dir: str | Path | None = None,
) -> Path:
    """Return a ready artifact directory for a registered model.

    No network access occurs unless ``download=True`` or
    ``KATANA_ARTIFACT_AUTO_DOWNLOAD=1``.
    """
    spec = _coerce_spec(spec)
    status = artifact_status(spec, target_dir)
    if status.present:
        return status.path
    should_download = _truthy(os.environ.get("KATANA_ARTIFACT_AUTO_DOWNLOAD")) if download is None else download
    if should_download:
        return download_artifact(spec, target_dir).path
    raise ArtifactNotFoundError(
        f"{spec.display_name or spec.name} artifact is missing. Run `katana artifacts setup`, "
        f"run `katana artifacts download {spec.aliases[0] if spec.aliases else spec.name}`, "
        f"set {spec.path_env_var or 'KATANA_ARTIFACT_DIR'}, or set KATANA_ARTIFACT_AUTO_DOWNLOAD=1. "
        f"Missing files: {', '.join(status.missing_files)}"
    )

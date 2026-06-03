"""Hugging Face artifact resolution and download helpers.

Large model/data artifacts live outside GitHub. This module keeps runtime
resolution explicit and offline-safe: nothing downloads unless the caller asks.
"""

from __future__ import annotations

import os
import hashlib
import hmac
import json
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from hermes_katana._paths import home_or_fallback
from hermes_katana._version import ARTIFACT_REVISION

DEFAULT_ARTIFACT_MODEL = "minilm"
DEFAULT_MINILM_ONNX_REPO = "Carlosian/hermes-katana-v15-distill-minilm-onnx"
DEFAULT_MINILM_TORCH_REPO = "Carlosian/hermes-katana-v15-distill-minilm"
DEFAULT_V15_LARGE_REPO = "Carlosian/hermes-katana-v15-large"
DEFAULT_REVISION = ARTIFACT_REVISION
# v15 runtime artifacts are pinned to their own HF revision, decoupled from the
# package version so a package version bump never retargets their download.
V15_REVISION = "v3.0.0"
ARTIFACT_MANIFEST = "artifact_manifest.json"

MINILM_ONNX_REQUIRED_FILES = (
    "model.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    ARTIFACT_MANIFEST,
)

MINILM_ONNX_ALLOW_PATTERNS = (*MINILM_ONNX_REQUIRED_FILES, "README.md")

MINILM_TORCH_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    ARTIFACT_MANIFEST,
)

MINILM_TORCH_ALLOW_PATTERNS = (*MINILM_TORCH_REQUIRED_FILES, "README.md")

V15_LARGE_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    ARTIFACT_MANIFEST,
)

V15_LARGE_ALLOW_PATTERNS = (*V15_LARGE_REQUIRED_FILES, "README.md")

# v3.1 origin-aware research models (paper artifacts). Pinned to the commit that
# carries the integrity manifest so downloads are reproducible.
DEFAULT_V17_LARGE_REPO = "Carlosian/hermes-katana-17"
DEFAULT_V17_MINILM_REPO = "Carlosian/hermes-katana-90"
V17_LARGE_REVISION = "a08883466abd2924587ac0646fa693c0a27b50af"
V17_MINILM_REVISION = "fc52b343e206d190bcb773a32a909a3885fdf480"

V17_LARGE_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    ARTIFACT_MANIFEST,
)

V17_LARGE_ALLOW_PATTERNS = (*V17_LARGE_REQUIRED_FILES, "README.md", "results_comparison.png")

V17_MINILM_REQUIRED_FILES = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    ARTIFACT_MANIFEST,
)

V17_MINILM_ALLOW_PATTERNS = (*V17_MINILM_REQUIRED_FILES, "README.md", "results_comparison.png")


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
    managed_setup: bool = True


@dataclass(frozen=True)
class ArtifactStatus:
    spec: ArtifactSpec
    path: Path
    present: bool
    missing_files: tuple[str, ...]
    source: str
    errors: tuple[str, ...] = ()
    verified_files: tuple[str, ...] = ()


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


_BASE_ARTIFACT_SPECS: tuple[ArtifactSpec, ...] = (
    ArtifactSpec(
        name="katana_v15_distill_minilm_onnx",
        aliases=("minilm", "small", "fast_cpu", "katana_v15_minilm", "katana_v15_distill_minilm_onnx"),
        display_name="Katana v15 MiniLM ONNX",
        repo_id=DEFAULT_MINILM_ONNX_REPO,
        repo_type="model",
        revision=V15_REVISION,
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
        name="katana_v15_distill_minilm_torch",
        aliases=(
            "minilm_torch",
            "small_torch",
            "torch_minilm",
            "katana_v15_minilm_torch",
            "katana_v15_distill_minilm",
        ),
        display_name="Katana v15 MiniLM PyTorch",
        repo_id=DEFAULT_MINILM_TORCH_REPO,
        repo_type="model",
        revision=V15_REVISION,
        required_files=MINILM_TORCH_REQUIRED_FILES,
        allow_patterns=MINILM_TORCH_ALLOW_PATTERNS,
        size_label="~88 MB",
        role="optional PyTorch CPU/GPU Scabbard classifier checkpoint",
        profile="torch_cpu",
        path_env_var="KATANA_MINILM_TORCH_DIR",
        repo_env_var="KATANA_MINILM_TORCH_HF_REPO_ID",
        revision_env_var="KATANA_MINILM_TORCH_HF_REVISION",
        interactive_default=False,
        requires_confirmation=False,
    ),
    ArtifactSpec(
        name="katana_v15_large",
        aliases=("large", "v15_large", "katana_v15_large", "deberta", "teacher"),
        display_name="Katana v15 Large",
        repo_id=DEFAULT_V15_LARGE_REPO,
        repo_type="model",
        revision=V15_REVISION,
        required_files=V15_LARGE_REQUIRED_FILES,
        allow_patterns=V15_LARGE_ALLOW_PATTERNS,
        size_label="large",
        role="optional high-accuracy local Scabbard model",
        profile="max_local",
        path_env_var="KATANA_V15_LARGE_DIR",
        repo_env_var="KATANA_V15_LARGE_HF_REPO_ID",
        revision_env_var="KATANA_V15_LARGE_HF_REVISION",
        interactive_default=False,
        requires_confirmation=True,
    ),
    ArtifactSpec(
        name="katana_v17_large",
        aliases=("v17_large", "large_v17", "deberta_v17", "katana_v17", "katana_v17_large"),
        display_name="Katana v17 DeBERTa-v3-large (origin-aware)",
        repo_id=DEFAULT_V17_LARGE_REPO,
        repo_type="model",
        revision=V17_LARGE_REVISION,
        required_files=V17_LARGE_REQUIRED_FILES,
        allow_patterns=V17_LARGE_ALLOW_PATTERNS,
        size_label="~1.7 GB",
        role="v3.1 origin-aware 9-class classifier (high accuracy)",
        profile="max_local",
        path_env_var="KATANA_V17_LARGE_DIR",
        repo_env_var="KATANA_V17_LARGE_HF_REPO_ID",
        revision_env_var="KATANA_V17_LARGE_HF_REVISION",
        interactive_default=False,
        requires_confirmation=True,
        managed_setup=False,
    ),
    ArtifactSpec(
        name="katana_v17_minilm",
        aliases=("v17_minilm", "minilm_v17", "small_v17", "katana_v17_minilm", "katana_v17_distill_minilm"),
        display_name="Katana v17 MiniLM-L6 (origin-aware, distilled)",
        repo_id=DEFAULT_V17_MINILM_REPO,
        repo_type="model",
        revision=V17_MINILM_REVISION,
        required_files=V17_MINILM_REQUIRED_FILES,
        allow_patterns=V17_MINILM_ALLOW_PATTERNS,
        size_label="~90 MB",
        role="v3.1 origin-aware distilled CPU classifier (PyTorch)",
        profile="torch_cpu",
        path_env_var="KATANA_V17_MINILM_DIR",
        repo_env_var="KATANA_V17_MINILM_HF_REPO_ID",
        revision_env_var="KATANA_V17_MINILM_HF_REVISION",
        interactive_default=False,
        requires_confirmation=False,
        managed_setup=False,
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
    base = Path(xdg).expanduser() if xdg else home_or_fallback() / ".cache"
    return (base / "hermes-katana" / "artifacts").resolve()


def minilm_onnx_spec(repo_id: str | None = None, revision: str | None = None) -> ArtifactSpec:
    """Build the default MiniLM ONNX artifact spec with env overrides."""
    return artifact_spec("minilm", repo_id=repo_id, revision=revision)


def minilm_torch_spec(repo_id: str | None = None, revision: str | None = None) -> ArtifactSpec:
    """Build the optional MiniLM PyTorch artifact spec with env overrides."""
    return artifact_spec("minilm_torch", repo_id=repo_id, revision=revision)


def v15_large_spec(repo_id: str | None = None, revision: str | None = None) -> ArtifactSpec:
    """Build the optional large v15 artifact spec with env overrides."""
    return artifact_spec("large", repo_id=repo_id, revision=revision)


def v17_large_spec(repo_id: str | None = None, revision: str | None = None) -> ArtifactSpec:
    """Build the v3.1 origin-aware DeBERTa-v3-large artifact spec with env overrides."""
    return artifact_spec("v17_large", repo_id=repo_id, revision=revision)


def v17_minilm_spec(repo_id: str | None = None, revision: str | None = None) -> ArtifactSpec:
    """Build the v3.1 origin-aware distilled MiniLM artifact spec with env overrides."""
    return artifact_spec("v17_minilm", repo_id=repo_id, revision=revision)


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
    errors, verified = _verify_artifact_manifest(path, spec)
    if spec.path_env_var and os.environ.get(spec.path_env_var):
        source = spec.path_env_var
    elif target_dir:
        source = "target-dir"
    else:
        source = "cache"
    return ArtifactStatus(
        spec=spec,
        path=path,
        present=not missing and not errors,
        missing_files=missing,
        source=source,
        errors=errors,
        verified_files=verified,
    )


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_manifest_path(rel: str) -> bool:
    if not rel or rel.startswith("/") or "\\" in rel:
        return False
    return ".." not in PurePosixPath(rel).parts


def _manifest_file_entries(payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], tuple[str, ...]]:
    raw_files = payload.get("files", payload.get("artifacts"))
    errors: list[str] = []
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(raw_files, dict):
        for rel, meta in raw_files.items():
            rel_str = str(rel)
            if not _safe_manifest_path(rel_str):
                errors.append(f"manifest contains unsafe path {rel_str!r}")
                continue
            entries[rel_str] = meta if isinstance(meta, dict) else {}
    elif isinstance(raw_files, list):
        for raw in raw_files:
            if not isinstance(raw, dict):
                errors.append("manifest files entries must be objects")
                continue
            rel = raw.get("path") or raw.get("file") or raw.get("name")
            if not isinstance(rel, str) or not _safe_manifest_path(rel):
                errors.append(f"manifest contains unsafe or missing path {rel!r}")
                continue
            entries[rel] = raw
    else:
        errors.append("artifact_manifest.json must contain a files mapping or list")
    return entries, tuple(errors)


def _verify_artifact_manifest(path: Path, spec: ArtifactSpec) -> tuple[tuple[str, ...], tuple[str, ...]]:
    manifest_path = path / ARTIFACT_MANIFEST
    if not manifest_path.is_file():
        return (), ()

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return (f"{ARTIFACT_MANIFEST}: failed to parse manifest: {exc}",), ()
    if not isinstance(payload, dict):
        return (f"{ARTIFACT_MANIFEST}: manifest root must be an object",), ()

    errors: list[str] = []
    artifact_name = payload.get("artifact") or payload.get("name")
    if isinstance(artifact_name, str):
        valid_names = {spec.name, *spec.aliases}
        if artifact_name not in valid_names:
            errors.append(f"{ARTIFACT_MANIFEST}: artifact name {artifact_name!r} does not match {spec.name!r}")

    entries, entry_errors = _manifest_file_entries(payload)
    errors.extend(entry_errors)
    verified: list[str] = []

    for rel in spec.required_files:
        if rel == ARTIFACT_MANIFEST:
            continue
        file_path = path / rel
        if not file_path.is_file():
            continue
        entry = entries.get(rel)
        if not entry:
            errors.append(f"{ARTIFACT_MANIFEST}: missing file entry for {rel}")
            continue
        expected_sha = entry.get("sha256")
        if not isinstance(expected_sha, str) or not expected_sha.strip():
            errors.append(f"{ARTIFACT_MANIFEST}: missing sha256 for {rel}")
            continue
        actual_sha = _file_sha256(file_path)
        if not hmac.compare_digest(actual_sha.lower(), expected_sha.strip().lower()):
            errors.append(f"{rel}: sha256 mismatch")
            continue
        expected_size = entry.get("size", entry.get("size_bytes"))
        if expected_size is not None:
            try:
                expected_size_int = int(expected_size)
            except (TypeError, ValueError):
                errors.append(f"{ARTIFACT_MANIFEST}: invalid size for {rel}")
                continue
            if file_path.stat().st_size != expected_size_int:
                errors.append(f"{rel}: size mismatch")
                continue
        verified.append(rel)

    return tuple(errors), tuple(verified)


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
            "Install `huggingface_hub` (`pip install hermes-katana[fast-cpu]` or `pip install hermes-katana[hf]`) "
            "or the modern `hf` CLI."
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
        details = []
        if status.missing_files:
            details.append(f"missing: {', '.join(status.missing_files)}")
        if status.errors:
            details.append(f"errors: {'; '.join(status.errors)}")
        raise ArtifactDownloadError(f"Downloaded artifact is incomplete or unverified; {'; '.join(details)}")
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


def resolve_minilm_torch(
    *,
    download: bool | None = None,
    repo_id: str | None = None,
    revision: str | None = None,
    target_dir: str | Path | None = None,
) -> Path:
    """Return a ready MiniLM PyTorch checkpoint artifact directory."""
    spec = minilm_torch_spec(repo_id=repo_id, revision=revision)
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

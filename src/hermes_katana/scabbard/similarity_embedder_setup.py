"""Install and verify the ONNX sentence embedder used by the similarity softener."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

_REPO = "Xenova/all-MiniLM-L6-v2"
_ENV_TARGET = "KATANA_SIM_EMBEDDER_DIR"
_ENV_SKIP_SETUP = "HERMES_KATANA_SKIP_SETUP"
_DEFAULT_EMBEDDER_SUBDIR = "onnx_embedder_allMiniLM"
_FILES = (
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


@dataclass(frozen=True)
class SimilarityEmbedderSetupResult:
    """Outcome from a similarity embedder setup attempt."""

    target_dir: Path
    downloaded_files: tuple[str, ...] = ()
    skipped: bool = False
    ready: bool = False
    message: str = ""


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_similarity_embedder_dir(target_dir: str | Path | None = None) -> Path:
    """Return the concrete directory that should contain the ONNX embedder files."""
    if target_dir is not None:
        return Path(target_dir).expanduser().resolve()
    override = os.environ.get(_ENV_TARGET)
    if override:
        return Path(override).expanduser().resolve()

    from hermes_katana.artifacts import default_artifact_cache_dir

    return default_artifact_cache_dir() / _DEFAULT_EMBEDDER_SUBDIR


def missing_similarity_embedder_files(target_dir: str | Path | None = None) -> tuple[str, ...]:
    """Return required embedder files that are absent from ``target_dir``."""
    target = resolve_similarity_embedder_dir(target_dir)
    return tuple(rel for rel in _FILES if not (target / rel).is_file())


def similarity_embedder_files_present(target_dir: str | Path | None = None) -> bool:
    """Return True when all files needed to load the similarity embedder exist."""
    return not missing_similarity_embedder_files(target_dir)


def install_similarity_embedder(
    *,
    target_dir: str | Path | None = None,
    force: bool = False,
) -> SimilarityEmbedderSetupResult:
    """Download the torch-free all-MiniLM ONNX embedder into ``target_dir``.

    The installer is intentionally explicit and fail-closed: it only performs a
    network download when called, it honors ``HERMES_KATANA_SKIP_SETUP=1``, and
    it reports an incomplete artifact instead of treating a non-empty directory
    as ready.
    """
    target = resolve_similarity_embedder_dir(target_dir)
    if _truthy(os.environ.get(_ENV_SKIP_SETUP)):
        return SimilarityEmbedderSetupResult(
            target_dir=target,
            skipped=True,
            ready=similarity_embedder_files_present(target),
            message=f"{_ENV_SKIP_SETUP} is set",
        )

    if not force and similarity_embedder_files_present(target):
        return SimilarityEmbedderSetupResult(
            target_dir=target,
            ready=True,
            message="similarity embedder already present",
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required: pip install hermes-katana[fast-cpu]") from exc

    target.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("KATANA_HF_TOKEN") or os.environ.get("HF_TOKEN")
    downloaded: list[str] = []
    for filename in _FILES:
        try:
            hf_hub_download(
                repo_id=_REPO,
                filename=filename,
                local_dir=str(target),
                token=token,
                force_download=force,
            )
        except Exception as exc:
            raise RuntimeError(f"failed to download {filename} from {_REPO}: {exc}") from exc
        downloaded.append(filename)

    missing = missing_similarity_embedder_files(target)
    if missing:
        raise RuntimeError(f"incomplete similarity embedder at {target}; missing: {', '.join(missing)}")

    return SimilarityEmbedderSetupResult(
        target_dir=target,
        downloaded_files=tuple(downloaded),
        ready=True,
        message="similarity embedder downloaded",
    )


def verify_similarity_embedder_runtime(target_dir: str | Path | None = None) -> bool:
    """Return True if the similarity allowlist can load and use the embedder."""
    target = resolve_similarity_embedder_dir(target_dir)
    previous = os.environ.get(_ENV_TARGET)
    os.environ[_ENV_TARGET] = str(target)
    try:
        from hermes_katana.scabbard.similarity_allowlist import _allowlist

        # Clear before probing so a temporary KATANA_SIM_EMBEDDER_DIR override
        # is honored even if the allowlist singleton was already cached.
        _allowlist.cache_clear()
        return _allowlist().is_ready()
    finally:
        from hermes_katana.scabbard.similarity_allowlist import _allowlist

        # Clear again after restoring the environment so later callers do not
        # keep using the temporary verification target.
        _allowlist.cache_clear()
        if previous is None:
            os.environ.pop(_ENV_TARGET, None)
        else:
            os.environ[_ENV_TARGET] = previous


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default=None, help="Directory where the ONNX embedder files should be stored.")
    parser.add_argument("--force", action="store_true", help="Re-download files even when they already exist.")
    parser.add_argument("--no-verify", action="store_true", help="Skip runtime verification after download.")
    args = parser.parse_args(argv)

    try:
        result = install_similarity_embedder(target_dir=args.target_dir, force=args.force)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    if result.skipped:
        return 0

    if result.downloaded_files:
        print(f"Downloaded {_REPO} (ONNX) -> {result.target_dir}")
        for filename in result.downloaded_files:
            path = result.target_dir / filename
            print(f"  ok  {filename} ({path.stat().st_size // 1024} KB)")
    else:
        print(f"Present similarity embedder: {result.target_dir}")

    if not args.no_verify and not verify_similarity_embedder_runtime(result.target_dir):
        print("WARNING: files are present, but the similarity allowlist could not load the embedder.")
    else:
        print(f"Done. Similarity softener will use: {result.target_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

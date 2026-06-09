#!/usr/bin/env python3
"""Run Hermes drift check and optionally open a snapshot-refresh PR.

This script is intentionally GitHub-Actions friendly but not Actions-specific:
it applies Katana patch templates to a supplied Hermes checkout, refreshes the
pinned hermes-current fixture when running on the default branch, validates the
refreshed snapshot, and opens or updates a maintenance PR if the fixture changed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATHS = [
    "tests/fixtures/hermes_compat/hermes-current-snapshot",
    "tests/unit/test_compat_snapshots.py",
]
VALIDATION_COMMANDS = [
    [
        sys.executable,
        "-m",
        "ruff",
        "check",
        "scripts/update_hermes_current_snapshot.py",
        "tests/unit/test_update_hermes_current_snapshot.py",
        "tests/unit/test_hermes_drift_autorepair.py",
        "tests/unit/test_compat_snapshots.py",
        "tests/unit/test_patches.py",
    ],
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/unit/test_update_hermes_current_snapshot.py",
        "tests/unit/test_hermes_drift_autorepair.py",
        "tests/unit/test_compat_snapshots.py",
        "tests/unit/test_patches.py",
    ],
]


def _run(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, check=check, text=True, capture_output=True)


def _run_visible(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = _run(command)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result


def _run_visible_checked(command: list[str]) -> None:
    result = _run_visible(command)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)


def _git(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], check=check)


def _hermes_commit(checkout: Path) -> tuple[str, str]:
    full = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    short = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "--short=12", "HEAD"], text=True).strip()
    return full, short


def _check_drift(checkout: Path) -> tuple[int, str]:
    result = _run([sys.executable, "scripts/check_hermes_drift.py", str(checkout)])
    output = result.stdout + result.stderr
    print(output, end="")
    return result.returncode, output




def _restore_hermes_checkout(checkout: Path) -> None:
    """Undo check_hermes_drift mutations before copying snapshot fixtures."""
    subprocess.run(["git", "-C", str(checkout), "reset", "--hard", "HEAD"], check=True, text=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "clean", "-fd"], check=True, text=True, capture_output=True)

def _refresh_snapshot(checkout: Path) -> bool:
    result = _run_visible([sys.executable, "scripts/update_hermes_current_snapshot.py", "--source", str(checkout)])
    if result.returncode != 0:
        raise RuntimeError("snapshot refresh failed")
    diff = _git("diff", "--quiet", "--", *SNAPSHOT_PATHS)
    return diff.returncode != 0


def _validate_snapshot() -> tuple[int, str]:
    combined: list[str] = []
    status = 0
    for command in VALIDATION_COMMANDS:
        result = _run(command)
        output = result.stdout + result.stderr
        combined.append("$ " + " ".join(command) + "\n" + output)
        print(output, end="")
        if result.returncode != 0:
            status = result.returncode
    return status, "\n".join(combined)


def _write_pr_body(
    path: Path,
    *,
    hermes_commit: str,
    drift_status: int,
    validation_status: int | None,
    drift_output: str,
    validation_output: str,
) -> None:
    lines = [
        f"Refreshes the pinned hermes-current compatibility snapshot from NousResearch/hermes-agent@{hermes_commit}.",
        "",
        "Automation status:",
        f"- Drift check exit: {drift_status}",
        f"- Snapshot validation exit: {validation_status if validation_status is not None else 'not-run'}",
        "",
        "If the drift check or validation failed, use this PR as the reviewable fixture update and repair patch anchors in src/hermes_katana/installer/patches.py before merging.",
        "",
        "Drift output:",
        "```",
        drift_output.rstrip(),
        "```",
    ]
    if validation_output:
        lines.extend(["", "Snapshot validation output:", "```", validation_output.rstrip(), "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _open_or_update_pr(branch: str, title: str, body_path: Path, default_branch: str) -> None:
    _git("config", "user.name", "github-actions[bot]", check=True)
    _git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com", check=True)
    _git("checkout", "-B", branch, check=True)
    _git("add", *SNAPSHOT_PATHS, check=True)
    _git("commit", "-m", "chore: refresh Hermes compatibility snapshot", check=True)
    _git("push", "--force-with-lease", "origin", branch, check=True)

    existing = _run(["gh", "pr", "list", "--head", branch, "--json", "number", "--jq", ".[0].number // empty"])
    number = existing.stdout.strip()
    if number:
        _run_visible_checked(["gh", "pr", "edit", number, "--title", title, "--body-file", str(body_path)])
    else:
        _run_visible_checked(
            [
                "gh",
                "pr",
                "create",
                "--base",
                default_branch,
                "--head",
                branch,
                "--title",
                title,
                "--body-file",
                str(body_path),
            ]
        )


def _can_open_pr(default_branch: str) -> bool:
    if os.environ.get("GITHUB_REF_NAME") != default_branch:
        return False
    return bool(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True, type=Path, help="Path to the latest Hermes Agent checkout")
    parser.add_argument("--default-branch", required=True, help="Repository default branch for PR targets")
    parser.add_argument("--pr", action="store_true", help="Open/update a snapshot refresh PR when running on default branch")
    args = parser.parse_args(argv)

    checkout = args.checkout.resolve()
    if not checkout.is_dir():
        print(f"error: Hermes checkout does not exist: {checkout}", file=sys.stderr)
        return 2

    hermes_commit, hermes_short = _hermes_commit(checkout)
    drift_status, drift_output = _check_drift(checkout)

    validation_status: int | None = None
    validation_output = ""
    changed = False
    if args.pr and _can_open_pr(args.default_branch):
        _restore_hermes_checkout(checkout)
        changed = _refresh_snapshot(checkout)
        if changed:
            validation_status, validation_output = _validate_snapshot()
            body_path = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "hermes-snapshot-pr-body.md"
            _write_pr_body(
                body_path,
                hermes_commit=hermes_commit,
                drift_status=drift_status,
                validation_status=validation_status,
                drift_output=drift_output,
                validation_output=validation_output,
            )
            branch = f"automation/hermes-current-snapshot-{hermes_short}"
            title = f"chore: refresh Hermes compatibility snapshot ({hermes_short})"
            _open_or_update_pr(branch, title, body_path, args.default_branch)
        else:
            print("hermes-current snapshot already matches latest Hermes; no PR needed.")
    elif args.pr:
        print("Skipping snapshot PR: not on the default branch or no GitHub token is available.")

    if drift_status != 0:
        return drift_status
    if changed and validation_status not in (None, 0):
        return validation_status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

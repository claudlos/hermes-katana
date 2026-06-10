#!/usr/bin/env python3
"""Pre-publish PII scrubber (2026-06 audit findings F2/F4).

Fails (exit 1) when any tracked text file contains an operator-identifying
token: a blocklisted email address or a home-directory path whose username is
blocklisted. The blocklist is stored as SHA-256 hashes so this script never
reintroduces the values it exists to keep out of the repository.

Usage::

    python tools/prepublish_scrub.py            # scan all git-tracked files
    python tools/prepublish_scrub.py PATH...    # scan specific files/dirs

Run it in CI before any benchmark/data release and as part of the release
gate. Add new hashes with:
``python -c "import hashlib; print(hashlib.sha256(b'<token>').hexdigest())"``
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

# SHA-256 hashes of blocked tokens (lower-cased). Never store the raw values.
# Currently: the operator's personal email, its local-part, and the operator
# usernames seen in pre-depersonalization home-directory paths.
_BLOCKED_HASHES = {
    "441a6ee9055d25bf789d29744325060323c624b5113612892e847172a96248ec",  # email
    "0122ec1aa38db19055021ad97b81c0d4bc6d797697b128a92290f982d62f24c1",  # email local-part
    "7b85175b455060e3237e925f023053ca9515e8682a83c8b09911c724a1f8b75f",  # home-dir username
    "b54a95127a4b573f41e335fdbd339dcc2208fbfb1ae0b6fab7599d6e2d6ec754",  # home-dir username
}

# Generic usernames that legitimately appear in fixtures and docs.
_ALLOWED_USERNAMES = {
    "user",
    "users",
    "runner",
    "agent",
    "operator",
    "example",
    "test",
    "testuser",
    "account-holder",
    "username",
    "<user>",
    "%username%",
    "$user",
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HOME_RE = re.compile(r"(?:/home/|/Users/|[Cc]:\\+Users\\+)([A-Za-z0-9._-]+)")

_SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".whl",
    ".pyc",
    ".so",
    ".dll",
    ".exe",
    ".bin",
    ".pt",
    ".onnx",
    ".safetensors",
    ".npz",
    ".pkl",
}


def _hash(token: str) -> str:
    return hashlib.sha256(token.strip().lower().encode("utf-8")).hexdigest()


def _is_blocked(token: str) -> bool:
    return _hash(token) in _BLOCKED_HASHES


def _iter_tracked_files(args: list[str]) -> list[Path]:
    if args:
        files: list[Path] = []
        for arg in args:
            p = Path(arg)
            if p.is_dir():
                files.extend(q for q in p.rglob("*") if q.is_file())
            elif p.is_file():
                files.append(p)
        return files
    out = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in out.stdout.splitlines() if line.strip()]


def scan_file(path: Path) -> list[tuple[int, str]]:
    """Return (line_number, reason) hits for *path*."""
    if path.suffix.lower() in _SKIP_SUFFIXES:
        return []
    # This script carries the hash list and the patterns; scanning it (or its
    # tests) for blocked tokens is fine — hashes never match themselves.
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for email in _EMAIL_RE.findall(line):
            local_part = email.split("@", 1)[0]
            if _is_blocked(email) or _is_blocked(local_part):
                hits.append((lineno, "blocked email address"))
        for username in _HOME_RE.findall(line):
            if username.lower() in _ALLOWED_USERNAMES:
                continue
            if _is_blocked(username):
                hits.append((lineno, f"blocked home-directory username in {line.strip()[:80]!r}"))
    return hits


def main(argv: list[str]) -> int:
    failures = 0
    for path in _iter_tracked_files(argv):
        for lineno, reason in scan_file(path):
            print(f"PII-SCRUB FAIL {path}:{lineno}: {reason}")
            failures += 1
    if failures:
        print(
            f"\n{failures} blocked-token occurrence(s) found. Redact them (use "
            "synthetic values like account-holder@example.com or /home/user) "
            "before publishing.",
            file=sys.stderr,
        )
        return 1
    print("prepublish scrub: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

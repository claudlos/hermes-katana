#!/usr/bin/env python3
"""Apply Katana source patches to a Hermes checkout and verify they still land.

The source patches are exact-anchor patches: they match verbatim text in Hermes
internals. When upstream Hermes refactors those internals, the anchors stop
matching and a real install silently loses protection on the affected patch.

This script is the early-warning system. The ``hermes-drift`` CI job clones the
latest Hermes Agent and runs this against it on a schedule, so anchor drift shows
up as a red build with an actionable message instead of a field incident.

Exit status:
    0 — every *critical* patch applied and all patched Python files compile.
    1 — a critical patch failed to apply, or a patched file failed to compile.

Non-critical patches (banner, Docker, gateway) that fail to apply are reported
but do not fail the run; they degrade gracefully at install time.
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path

from hermes_katana.installer.patches import (
    CURRENT_CORE_PATCHES,
    PatchStatus,
    apply_patches,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path, help="Path to a Hermes Agent checkout")
    args = parser.parse_args()

    checkout: Path = args.checkout
    if not checkout.is_dir():
        print(f"error: {checkout} is not a directory", file=sys.stderr)
        return 2

    critical_by_name = {p.name: p.critical for p in CURRENT_CORE_PATCHES}
    results = apply_patches(checkout, patches=CURRENT_CORE_PATCHES)

    failures: list[str] = []
    print("Patch results against latest Hermes:")
    for result in results:
        status = result.status.value if hasattr(result.status, "value") else str(result.status)
        critical = critical_by_name.get(result.name, False)
        marker = "CRITICAL" if critical else "optional"
        print(f"  {result.name:30} {status:10} [{marker}] {result.message or ''}".rstrip())
        # Intentionally conservative: for a drift canary, anything other than a
        # clean APPLIED on a critical patch is worth failing on. In practice
        # apply_patches only ever returns APPLIED or ERROR for a critical patch
        # (search-not-found and ambiguous-anchor both map to ERROR; SKIPPED is
        # non-critical only), so this does not produce false positives.
        if result.status != PatchStatus.APPLIED and critical:
            failures.append(f"patch '{result.name}' did not apply ({status})")

    print("\nCompile check of patched files:")
    py_targets = sorted({p.target_file for p in CURRENT_CORE_PATCHES if str(p.target_file).endswith(".py")})
    for rel in py_targets:
        target = checkout / rel
        if not target.exists():
            print(f"  {rel:40} MISSING")
            continue
        try:
            py_compile.compile(str(target), doraise=True)
            print(f"  {rel:40} OK")
        except Exception as exc:  # noqa: BLE001 - report any compile error verbatim
            print(f"  {rel:40} FAIL: {exc}")
            failures.append(f"patched file '{rel}' did not compile")

    if failures:
        print("\nDRIFT DETECTED — Katana patches no longer fit latest Hermes:")
        for item in failures:
            print(f"  - {item}")
        print(
            "\nNext steps: refresh tests/fixtures/hermes_compat snapshots and update the\n"
            "affected patch templates in src/hermes_katana/installer/patches.py."
        )
        return 1

    print("\nOK: all critical patches applied and all patched files compiled against latest Hermes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

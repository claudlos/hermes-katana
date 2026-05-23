#!/usr/bin/env python3
"""File-based evaluation runner for multimodal/binary scanners.

Tests scanners that require actual binary input (PDF, DOCX, images, SVG)
by generating real files and passing bytes directly to scanners.

Usage:
    python tests/eval/run_binary_eval.py                  # full run
    python tests/eval/run_binary_eval.py --category pdf   # single category
    python tests/eval/run_binary_eval.py --json           # JSON output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project is importable
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(_root) / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

# Import scanners
_scanners = {}

try:
    from hermes_katana.scanner.pdf_layers import detect_pdf_layers_bytes

    _scanners["pdf_layers"] = detect_pdf_layers_bytes
except ImportError as e:
    print(f"  WARN: pdf_layers not available: {e}")

try:
    from hermes_katana.scanner.stego_scanner import scan_stego

    _scanners["stego"] = scan_stego
except ImportError as e:
    print(f"  WARN: stego not available: {e}")

try:
    from hermes_katana.scanner.multimodal import scan_bytes_multimodal, scan_data_uri

    _scanners["multimodal_bytes"] = scan_bytes_multimodal
    _scanners["multimodal_uri"] = scan_data_uri
except ImportError as e:
    print(f"  WARN: multimodal not available: {e}")

try:
    from hermes_katana.scanner.svg_sanitizer import scan_svg

    _scanners["svg_sanitizer"] = scan_svg
except ImportError as e:
    print(f"  WARN: svg_sanitizer not available: {e}")

try:
    from hermes_katana.scanner.injection import detect_injection

    _scanners["injection"] = detect_injection
except ImportError as e:
    print(f"  WARN: injection not available: {e}")

try:
    from hermes_katana.scanner.content import scan_content

    _scanners["content"] = scan_content
except ImportError as e:
    print(f"  WARN: content not available: {e}")

try:
    from hermes_katana.scanner.secrets import scan_for_secrets

    _scanners["secrets"] = scan_for_secrets
except ImportError as e:
    print(f"  WARN: secrets not available: {e}")

try:
    from hermes_katana.scanner.commands import detect_dangerous_command

    _scanners["commands"] = detect_dangerous_command
except ImportError as e:
    print(f"  WARN: commands not available: {e}")

try:
    from hermes_katana.scanner.unicode_scanner import normalize_and_scan

    _scanners["unicode"] = lambda text: normalize_and_scan(text)[1]
except ImportError as e:
    print(f"  WARN: unicode not available: {e}")

try:
    from hermes_katana.scanner.decoder import decode_and_scan

    _scanners["decoder"] = decode_and_scan
except ImportError as e:
    print(f"  WARN: decoder not available: {e}")

try:
    from hermes_katana.scanner.prompt_leak import detect_prompt_leak

    _scanners["prompt_leak"] = detect_prompt_leak
except ImportError:
    pass

try:
    from hermes_katana.scanner.behavioral import detect_behavioral

    _scanners["behavioral"] = detect_behavioral
except ImportError:
    pass

try:
    from hermes_katana.scanner.structural import detect_structural

    _scanners["structural"] = detect_structural
except ImportError:
    pass

try:
    from hermes_katana.scanner.compositional import detect_compositional

    _scanners["compositional"] = detect_compositional
except ImportError:
    pass


def _has_findings(results: Any) -> bool:
    """Check if scanner results indicate a detection."""
    if results is None:
        return False
    if isinstance(results, list):
        return len(results) > 0
    if isinstance(results, dict):
        return any(v for v in results.values() if v)
    if isinstance(results, tuple):
        return any(_has_findings(r) for r in results)
    return bool(results)


def run_scanner_on_bytes(scanner_name: str, scanner_func, data: bytes, filename: str) -> dict:
    """Run a single scanner on binary data and return results."""
    findings = []
    error = None

    try:
        if scanner_name == "pdf_layers":
            findings = scanner_func(data)
        elif scanner_name == "stego":
            report = scanner_func(data)
            if hasattr(report, "flags"):
                findings = report.flags
            elif hasattr(report, "is_suspicious") and report.is_suspicious:
                findings = [report]
            elif isinstance(report, list):
                findings = report
            elif isinstance(report, dict) and report.get("suspicious"):
                findings = [report]
        elif scanner_name == "multimodal_bytes":
            findings = scanner_func(data, filename=filename)
        elif scanner_name == "svg_sanitizer":
            # SVG scanner expects text
            text = data.decode("utf-8", errors="ignore")
            findings = scanner_func(text)
        elif scanner_name in (
            "injection",
            "content",
            "secrets",
            "commands",
            "unicode",
            "decoder",
            "prompt_leak",
            "behavioral",
            "compositional",
        ):
            # Text-based scanners — decode bytes to string
            text = data.decode("utf-8", errors="ignore")
            result = scanner_func(text)
            findings = result if isinstance(result, list) else []
        elif scanner_name == "structural":
            text = data.decode("utf-8", errors="ignore")
            report = scanner_func(text)
            if isinstance(report, dict):
                findings = report.get("findings", [])
            elif isinstance(report, list):
                findings = report
        elif scanner_name == "multimodal_uri":
            text = data.decode("utf-8", errors="ignore")
            findings = scanner_func(text)
        else:
            # Generic: try text first, then bytes
            try:
                text = data.decode("utf-8", errors="ignore")
                findings = scanner_func(text)
            except Exception:
                findings = scanner_func(data)
    except Exception as e:
        error = str(e)

    return {
        "scanner": scanner_name,
        "detected": _has_findings(findings),
        "finding_count": len(findings) if isinstance(findings, list) else (1 if findings else 0),
        "error": error,
    }


def run_binary_eval(
    test_cases: list,
    output_dir: str,
    categories: list[str] | None = None,
) -> dict:
    """Run binary eval on test cases. Returns results dict."""
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cases": 0,
        "total_detected": 0,
        "total_missed": 0,
        "by_category": {},
        "by_scanner": {},
        "details": [],
    }

    for case in test_cases:
        if categories and not any(c in case.attack_type for c in categories):
            continue

        results["total_cases"] += 1

        # Track per-category
        cat = case.attack_type
        if cat not in results["by_category"]:
            results["by_category"][cat] = {"total": 0, "caught": 0, "missed": 0}
        results["by_category"][cat]["total"] += 1

        # Run ALL scanners (not just expected ones) to see full picture
        case_result = {
            "filename": case.filename,
            "attack_type": case.attack_type,
            "severity": case.severity,
            "expected_scanners": case.expected_scanners,
            "file_size": len(case.data),
            "scanner_results": {},
            "caught": False,
            "caught_by": [],
        }

        for scanner_name, scanner_func in _scanners.items():
            sr = run_scanner_on_bytes(scanner_name, scanner_func, case.data, case.filename)
            case_result["scanner_results"][scanner_name] = sr

            if sr["detected"]:
                case_result["caught"] = True
                case_result["caught_by"].append(scanner_name)

            # Track per-scanner
            if scanner_name not in results["by_scanner"]:
                results["by_scanner"][scanner_name] = {"total": 0, "caught": 0, "errors": 0}
            results["by_scanner"][scanner_name]["total"] += 1
            if sr["detected"]:
                results["by_scanner"][scanner_name]["caught"] += 1
            if sr["error"]:
                results["by_scanner"][scanner_name]["errors"] += 1

        if case_result["caught"]:
            results["total_detected"] += 1
            results["by_category"][cat]["caught"] += 1
        else:
            results["total_missed"] += 1
            results["by_category"][cat]["missed"] += 1

        results["details"].append(case_result)

    return results


def print_report(results: dict):
    """Print human-readable report."""
    total = results["total_cases"]
    caught = results["total_detected"]
    missed = results["total_missed"]
    coverage = (caught / total * 100) if total > 0 else 0

    print("\n" + "=" * 70)
    print("  HermesKatana Binary/Multimodal Scanner Evaluation")
    print("=" * 70)
    print(f"\n  Total test cases:  {total}")
    print(f"  Detected:          {caught}")
    print(f"  Missed:            {missed}")
    print(f"  Coverage:          {coverage:.1f}%")

    print("\n  ── By Attack Category ──────────────────────────────────────────")
    for cat, data in sorted(results["by_category"].items(), key=lambda x: -x[1]["total"]):
        t, c = data["total"], data["caught"]
        pct = (c / t * 100) if t > 0 else 0
        status = "✓" if c == t else "✗" if c == 0 else "△"
        print(f"  {status} {cat:<35} {c:>3}/{t:<3}  {pct:>5.1f}%")

    print("\n  ── By Scanner ──────────────────────────────────────────────────")
    for scanner, data in sorted(results["by_scanner"].items(), key=lambda x: -x[1]["caught"]):
        c, t = data["caught"], data["total"]
        pct = (c / t * 100) if t > 0 else 0
        errs = data["errors"]
        err_str = f" ({errs} errors)" if errs else ""
        print(f"  {scanner:<25} {c:>3}/{t:<3}  {pct:>5.1f}%{err_str}")

    print("\n  ── Missed Cases ────────────────────────────────────────────────")
    for detail in results["details"]:
        if not detail["caught"]:
            list(detail["scanner_results"].keys())
            errors = [s for s, r in detail["scanner_results"].items() if r.get("error")]
            print(f"  ✗ {detail['filename']:<35} {detail['attack_type']}")
            if errors:
                print(f"    Errors: {', '.join(errors)}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Binary/multimodal scanner evaluation")
    parser.add_argument("--category", help="Filter by category substring")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--output", default=None, help="Output file for JSON results")
    args = parser.parse_args()

    # Import test case generators
    from tests.eval.binary_test_cases import generate_all_cases

    print("Generating binary test cases...")
    test_cases = list(generate_all_cases())
    print(f"  Generated {len(test_cases)} cases")

    categories = [args.category] if args.category else None

    print(f"\nRunning scanners on {len(test_cases)} binary test cases...")
    results = run_binary_eval(test_cases, "/tmp/binary_eval", categories)

    if args.json:
        output = json.dumps(results, indent=2, default=str)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Results written to {args.output}")
        else:
            print(output)
    else:
        print_report(results)

    if args.output and not args.json:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nJSON results saved to {args.output}")

    return 0 if results["total_detected"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

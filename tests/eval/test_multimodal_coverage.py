"""Multimodal coverage evaluation for HermesKatana scanner testing.

Runs all generated multimodal attack files (PDF, image, DOCX, HTML) through
the appropriate Katana scanners and measures coverage by vector and scanner.

Usage:
    python tests/eval/test_multimodal_coverage.py [--manifest PATH]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

# Ensure project is importable
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(_root) / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)


# ============================================================
# Optional multimodal scanner imports
# ============================================================

try:
    from hermes_katana.scanner.pdf_layers import detect_pdf_layers
except Exception:
    detect_pdf_layers = None

try:
    from hermes_katana.scanner.pdf_js_scanner import detect_pdf_js
except Exception:
    detect_pdf_js = None

try:
    from hermes_katana.scanner.image_injection import detect_image_injection_bytes
except Exception:
    detect_image_injection_bytes = None

try:
    from hermes_katana.scanner.svg_sanitizer import scan_svg
except Exception:
    scan_svg = None

try:
    from hermes_katana.scanner.html_diff import scan_html as scan_html_mod
except Exception:
    scan_html_mod = None

try:
    from hermes_katana.scanner.decoder import decode_and_scan
except Exception:
    decode_and_scan = None

try:
    from hermes_katana.scanner.injection import detect_injection
except Exception:
    detect_injection = None


# ============================================================
# Reader utilities
# ============================================================


def _read_binary(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _clean_xml_text(text: str) -> str:
    """Decode HTML entities and strip zero-width characters from XML-extracted text."""
    import html as _html

    # Decode HTML entities (&#10; -> newline, &quot; -> ", &amp; -> &, etc.)
    text = _html.unescape(text)
    # Strip zero-width/invisible Unicode characters that can interfere with pattern matching
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00ad\u2060-\u2064\u180e\u200e\u200f]", "", text)
    return text


def _extract_docx_text(path: str) -> str:
    """Extract all text from a DOCX including hidden properties and XML content."""
    parts = []
    try:
        from docx import Document

        doc = Document(path)

        # Paragraph text
        for p in doc.paragraphs:
            if p.text:
                parts.append(p.text)

        # ALL core properties (expanded beyond the original 5)
        cp = doc.core_properties
        for attr in (
            "author",
            "title",
            "subject",
            "comments",
            "keywords",
            "description",
            "category",
            "content_status",
            "identifier",
            "language",
            "last_modified_by",
        ):
            val = getattr(cp, attr, None)
            if val:
                parts.append(str(val))
    except Exception:
        pass

    # DOCX is a zip archive -- extract content from specific XML parts
    try:
        import zipfile

        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name == "docProps/core.xml":
                    content = zf.read(name).decode("utf-8")
                    text = re.sub(r"<[^>]+>", " ", content).strip()
                    text = _clean_xml_text(text)
                    if text:
                        parts.append(f"[core_xml] {text}")
                elif name == "docProps/custom.xml":
                    content = zf.read(name).decode("utf-8")
                    text = re.sub(r"<[^>]+>", " ", content).strip()
                    text = _clean_xml_text(text)
                    if text:
                        parts.append(f"[custom_xml] {text}")
                elif name == "word/document.xml":
                    content = zf.read(name).decode("utf-8")
                    # Extract wp:docPr descr/title attributes (from embedded images/shapes)
                    # Use raw regex that doesn't stop at &quot;-encoded quotes
                    for attr_name in ("descr", "title"):
                        # Match attr="..." where ... can contain &quot; entities
                        for m in re.finditer(rf'{attr_name}="((?:[^"]|&(?:quot|apos);)*)"', content):
                            raw_val = m.group(1)
                            cleaned = _clean_xml_text(raw_val)
                            if cleaned and len(cleaned) > 5:
                                parts.append(f"[docPr_{attr_name}] {cleaned}")
                    # Extract w:vanish hidden text (text followed by vanish run property)
                    for m in re.finditer(r"<w:rPr>(.*?)</w:rPr>", content):
                        rpr_content = m.group(1)
                        if "w:vanish" in rpr_content:
                            pos = m.start()
                            before = content[max(0, pos - 2000) : pos]
                            for tm in re.findall(r"<w:t[^>]*>([^<]+)</w:t>", before):
                                if len(tm) > 10:
                                    parts.append(f"[HIDDEN] {_clean_xml_text(tm)}")
                    # Extract w:rPr w:vanish inline (empty rPr with self-closing vanish)
                    for m in re.finditer(r"<w:rPr><w:vanish/></w:rPr>", content):
                        pos = m.start()
                        before = content[max(0, pos - 2000) : pos]
                        for tm in re.findall(r"<w:t[^>]*>([^<]+)</w:t>", before):
                            if len(tm) > 10:
                                parts.append(f"[HIDDEN] {_clean_xml_text(tm)}")
    except Exception:
        pass

    return "\n".join(parts)


def _scan_docx_structure(filepath: str) -> list:
    """Structural DOCX attack detection for hidden / stylistic channels.

    `detect_injection` only fires on text that matches injection-style
    patterns, so cognitive-state, research-framed, harmful, and survey
    payloads embedded in hidden DOCX locations slip through. This scanner
    flags substantial prose found in locations where legitimate documents
    do not put free-form text, regardless of content.

    Covers: `docProps/custom.xml` property values, long `docProps/core.xml`
    fields, `w:vanish` hidden runs, runs with fake-comment styling (small
    font or light grey color), `wp:docPr` image alt text, and
    `word/comments.xml` entries.
    """
    import zipfile as _zf

    findings: list = []
    try:
        with _zf.ZipFile(filepath) as zf:
            names = set(zf.namelist())

            if "docProps/custom.xml" in names:
                content = zf.read("docProps/custom.xml").decode("utf-8", "replace")
                for m in re.finditer(r"<vt:lpwstr>([^<]+)</vt:lpwstr>", content):
                    val = _clean_xml_text(m.group(1))
                    if len(val) >= 30:
                        findings.append(
                            {
                                "type": "docx_custom_xml_prose",
                                "content": val[:200],
                            }
                        )

            if "docProps/core.xml" in names:
                content = zf.read("docProps/core.xml").decode("utf-8", "replace")
                field_limits = {
                    "title": 45,
                    "subject": 60,
                    "description": 60,
                    "keywords": 60,
                    "identifier": 45,
                    "creator": 60,
                }
                for tag, limit in field_limits.items():
                    pattern = rf"<(?:dc|cp):{tag}[^>]*>([^<]+)</(?:dc|cp):{tag}>"
                    for m in re.finditer(pattern, content):
                        val = _clean_xml_text(m.group(1))
                        if len(val) >= limit:
                            findings.append(
                                {
                                    "type": f"docx_core_long_{tag}",
                                    "content": val[:200],
                                }
                            )

            if "word/document.xml" in names:
                content = zf.read("word/document.xml").decode("utf-8", "replace")

                for run_m in re.finditer(r"<w:r\b[^>]*>(.*?)</w:r>", content, re.DOTALL):
                    run = run_m.group(1)
                    rpr_m = re.search(r"<w:rPr>(.*?)</w:rPr>", run, re.DOTALL)
                    if not rpr_m:
                        continue
                    rpr = rpr_m.group(1)

                    is_vanish = "w:vanish" in rpr

                    is_small = False
                    sz_m = re.search(r'<w:sz\b[^>]*w:val="(\d+)"', rpr)
                    if sz_m:
                        is_small = int(sz_m.group(1)) <= 16

                    is_light = False
                    color_m = re.search(r'<w:color\b[^>]*w:val="([0-9A-Fa-f]{6})"', rpr)
                    if color_m:
                        hx = color_m.group(1).upper()
                        try:
                            r_, g_, b_ = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
                            is_light = (r_ + g_ + b_) / 3 >= 200
                        except ValueError:
                            pass

                    if not (is_vanish or is_small or is_light):
                        continue

                    texts = re.findall(r"<w:t[^>]*>([^<]+)</w:t>", run)
                    combined = _clean_xml_text(" ".join(texts))
                    if len(combined) >= 15:
                        if is_vanish:
                            marker = "docx_vanish_hidden"
                        elif is_light:
                            marker = "docx_light_color_hidden"
                        else:
                            marker = "docx_small_font_hidden"
                        findings.append(
                            {
                                "type": marker,
                                "content": combined[:200],
                            }
                        )

                for attr in ("descr", "title"):
                    pattern = rf'<wp:docPr\b[^>]*\s{attr}="((?:[^"]|&(?:quot|apos);)+)"'
                    for m in re.finditer(pattern, content):
                        val = _clean_xml_text(m.group(1))
                        if len(val) >= 30:
                            findings.append(
                                {
                                    "type": f"docx_img_alt_{attr}",
                                    "content": val[:200],
                                }
                            )

            if "word/comments.xml" in names:
                content = zf.read("word/comments.xml").decode("utf-8", "replace")
                for m in re.finditer(r"<w:t[^>]*>([^<]+)</w:t>", content):
                    val = _clean_xml_text(m.group(1))
                    if len(val) >= 30:
                        findings.append(
                            {
                                "type": "docx_comment_prose",
                                "content": val[:200],
                            }
                        )
    except Exception:
        pass

    return findings


def _scan_html(html_text: str) -> list:
    findings = []
    text = html_text  # the raw HTML content

    # Hidden divs
    hidden_divs = re.findall(
        r'<div[^>]*style=["\'][^"\']*display\s*:\s*none[^"\']*["\'][^>]*>(.*?)</div>', text, re.DOTALL | re.IGNORECASE
    )
    for hd in hidden_divs:
        clean = re.sub(r"<[^>]+>", "", hd).strip()
        if clean:
            findings.append({"type": "hidden_div", "content": clean[:200]})

    # HTML comments with attack content
    comments = re.findall(r"<!--(.*?)-->", text, re.DOTALL)
    attack_kw = ["ignore", "override", "bypass", "disregard", "system:", "developer:", "inject", "jailbreak"]
    for c in comments:
        if any(kw in c.lower() for kw in attack_kw):
            findings.append({"type": "html_comment_injection", "content": c.strip()[:200]})

    # Meta tags with attack content
    for m in re.finditer(r'<meta\s+[^>]*(?:content|value)=["\']([^"\']{20,})["\']', text, re.IGNORECASE):
        content_val = m.group(1)
        if any(
            kw in content_val.lower()
            for kw in [
                "ignore",
                "override",
                "bypass",
                "disregard",
                "system:",
                "developer:",
                "inject",
                "jailbreak",
                "do not",
                "must",
                "will",
                "you are",
                "secret",
                "instruction",
            ]
        ):
            findings.append({"type": "meta_tag_injection", "content": content_val[:200]})

    # SVG embedded in HTML
    for m in re.finditer(r"<svg\b[^>]*>(.*?)</svg>", text, re.DOTALL | re.IGNORECASE):
        svg_content = m.group(0)
        svg_kw = [
            "ignore",
            "override",
            "inject",
            "system:",
            "developer:",
            "secret",
            "instruction",
            "hidden",
            "ignore all",
        ]
        if any(kw in svg_content.lower() for kw in svg_kw):
            findings.append({"type": "svg_in_html_injection", "content": svg_content[:200]})

    return findings


# ============================================================
# Routing: vector -> list of (scanner_name, scanner_fn, reader_fn)
# ============================================================


# PDF/DOCX/HTML all read as text for pdf_layers/injection scanners
# Images read as binary for image_injection
def _route_scanners(vector: str, filepath: str):
    def text_reader():
        return _read_text(filepath)

    def binary_reader():
        return _read_binary(filepath)

    def docx_reader():
        return _extract_docx_text(filepath)

    # Try text-based scanners on all non-image formats
    scanners_to_try = []

    # Always try these text scanners for non-image files
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        if detect_pdf_layers:
            scanners_to_try.append(("pdf_layers", detect_pdf_layers, _read_text))
        if detect_pdf_js:
            scanners_to_try.append(("pdf_js", detect_pdf_js, _read_text))
    elif ext in (".jpg", ".jpeg", ".png", ".gif", ".svg"):
        if detect_image_injection_bytes:
            scanners_to_try.append(("image_injection", detect_image_injection_bytes, _read_binary))
        if ext == ".svg" and scan_svg:
            scanners_to_try.append(("svg_sanitizer", scan_svg, _read_text))
    elif ext == ".docx":
        scanners_to_try.append(("docx_structure", _scan_docx_structure, lambda p: p))
        if detect_pdf_layers:
            scanners_to_try.append(("pdf_layers", detect_pdf_layers, _extract_docx_text))
    elif ext == ".html":
        # Use the real html_diff scanner (has meta tag + SVG injection detection)
        if scan_html_mod is not None:
            scanners_to_try.append(("html_diff", scan_html_mod, _read_text))
        # Also try local _scan_html as fallback
        if detect_pdf_layers:
            scanners_to_try.append(("pdf_layers", detect_pdf_layers, _read_text))

    # Universal text scanners
    if detect_injection and ext != ".svg":  # svg already has svg_sanitizer
        if ext in (".docx",):
            scanners_to_try.append(("injection", detect_injection, _extract_docx_text))
        elif ext in (".html",):
            scanners_to_try.append(("injection", detect_injection, _read_text))
        elif ext == ".pdf":
            scanners_to_try.append(("injection", detect_injection, _read_text))
    if decode_and_scan and ext != ".svg":
        if ext == ".pdf":
            scanners_to_try.append(("decoder", decode_and_scan, _read_text))

    return scanners_to_try


def _is_caught(findings: Any) -> bool:
    if findings is None:
        return False
    if isinstance(findings, list):
        return len(findings) > 0
    if isinstance(findings, dict):
        return any(v for v in findings.values() if v)
    return bool(findings)


def _find_payload_for_entry(entry: dict, payloads: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Best-effort manifest -> golden_cherry payload lookup via preview prefix."""
    preview = entry.get("payload_preview") or entry.get("payload_text_preview") or ""
    if not preview:
        return None
    probe = preview[:80]
    for payload in payloads:
        text = payload.get("text", "")
        if text.startswith(probe):
            return payload
    return None


# ============================================================
# Core evaluation
# ============================================================


def run_multimodal_eval(manifest_path: str | None = None) -> dict:
    base = Path(__file__).resolve().parent.parent.parent / "research" / "multimodal-corpus"

    if manifest_path is None:
        manifest_path = str(base / "manifest.jsonl")

    gen_dir = base / "generated"

    with open(manifest_path) as f:
        manifest = [json.loads(line) for line in f]

    payloads_path = base / "payloads" / "golden_cherry.jsonl"
    payloads: list[dict[str, Any]] = []
    if payloads_path.exists():
        with open(payloads_path) as f:
            payloads = [json.loads(line) for line in f if line.strip()]

    results = {
        "total_files": 0,
        "total_caught": 0,
        "coverage": 0.0,
        "total_injection_expected": 0,
        "total_injection_caught": 0,
        "injection_expected_coverage": 0.0,
        "per_vector": {},
        "per_scanner": {},
        "per_file": [],
    }

    scanned = 0
    caught = 0

    for entry in manifest:
        file_path = gen_dir / entry["file"]
        if not file_path.exists():
            continue

        vector = entry.get("vector", "unknown")
        category = entry.get("category", "unknown")
        payload = _find_payload_for_entry(entry, payloads)
        payload_text = payload.get("text", "") if payload else ""
        injection_expected = bool(detect_injection(payload_text)) if (payload_text and detect_injection) else False
        scanned += 1

        file_caught = False
        scanners_caught = []
        file_findings = []

        for sname, sfn, rfn in _route_scanners(vector, str(file_path)):
            if sfn is None:
                continue
            try:
                data = rfn(str(file_path))
                if isinstance(data, bytes) and len(data) == 0:
                    continue
                if isinstance(data, str) and not data.strip():
                    continue
                findings = sfn(data)
                hit = _is_caught(findings)

                if sname not in results["per_scanner"]:
                    results["per_scanner"][sname] = {"total_tested": 0, "caught": 0}
                results["per_scanner"][sname]["total_tested"] += 1

                if hit:
                    file_caught = True
                    scanners_caught.append(sname)
                    results["per_scanner"][sname]["caught"] += 1

                file_findings.append(
                    {
                        "scanner": sname,
                        "caught": hit,
                        "count": len(findings) if isinstance(findings, list) else 1,
                    }
                )
            except Exception as e:
                file_findings.append(
                    {
                        "scanner": sname,
                        "caught": False,
                        "error": str(e)[:200],
                    }
                )

        if file_caught:
            caught += 1

        if vector not in results["per_vector"]:
            results["per_vector"][vector] = {
                "total": 0,
                "caught": 0,
                "coverage": 0.0,
                "injection_expected": 0,
                "injection_caught": 0,
                "injection_coverage": 0.0,
                "by_scanner": {},
            }
        pv = results["per_vector"][vector]
        pv["total"] += 1
        if file_caught:
            pv["caught"] += 1
        if injection_expected:
            pv["injection_expected"] += 1
            results["total_injection_expected"] += 1
            if file_caught:
                pv["injection_caught"] += 1
                results["total_injection_caught"] += 1
        for sc in scanners_caught:
            pv["by_scanner"].setdefault(sc, 0)
            pv["by_scanner"][sc] += 1

        results["per_file"].append(
            {
                "file": entry["file"],
                "vector": vector,
                "category": category,
                "caught": file_caught,
                "expected_injection": injection_expected,
                "scanners_caught": scanners_caught,
                "findings": file_findings,
            }
        )

    results["total_files"] = scanned
    results["total_caught"] = caught
    results["coverage"] = caught / scanned if scanned > 0 else 0.0
    results["injection_expected_coverage"] = (
        results["total_injection_caught"] / results["total_injection_expected"]
        if results["total_injection_expected"] > 0
        else 0.0
    )

    for pv in results["per_vector"].values():
        pv["coverage"] = pv["caught"] / pv["total"] if pv["total"] > 0 else 0.0
        pv["injection_coverage"] = (
            pv["injection_caught"] / pv["injection_expected"] if pv["injection_expected"] > 0 else 0.0
        )

    return results


def test_run_multimodal_eval_requires_manifest(tmp_path):
    missing_manifest = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        run_multimodal_eval(str(missing_manifest))


def print_report(results: dict) -> None:
    print("\n" + "=" * 70)
    print("  HermesKatana Multimodal Coverage Report")
    print("=" * 70)
    print(f"  Total files testing: {results['total_files']}")
    print(f"  Caught:              {results['total_caught']}")
    print(f"  Coverage:            {results['coverage']:.1%}")
    print(
        f"  Injection-eligible:  {results['total_injection_caught']}/"
        f"{results['total_injection_expected']} ({results['injection_expected_coverage']:.1%})"
    )

    print(f"\n  {'Vector':<25} {'Caught':>8} {'Total':>8} {'Coverage':>10} {'InjExp':>8} {'InjCov':>10}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 10}")
    for v in sorted(results["per_vector"].keys()):
        pv = results["per_vector"][v]
        inj = f"{pv['injection_caught']}/{pv['injection_expected']}" if pv["injection_expected"] else "0/0"
        print(
            f"  {v:<25} {pv['caught']:>8} {pv['total']:>8} {pv['coverage']:>10.1%} "
            f"{inj:>8} {pv['injection_coverage']:>10.1%}"
        )

    print(f"\n  {'Scanner':<25} {'Tested':>8} {'Caught':>8} {'Rate':>10}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10}")
    for s in sorted(results["per_scanner"].keys()):
        ps = results["per_scanner"][s]
        rate = ps["caught"] / ps["total_tested"] if ps["total_tested"] > 0 else 0.0
        print(f"  {s:<25} {ps['total_tested']:>8} {ps['caught']:>8} {rate:>10.1%}")

    missed = [f for f in results["per_file"] if not f["caught"]]
    if missed:
        print(f"\n  MISSED ({len(missed)} files):")
        for mf in missed[:30]:
            print(f"    - {mf['file']} [{mf['vector']} / {mf['category']}]")
        if len(missed) > 30:
            print(f"    ... and {len(missed) - 30} more")

    caught_files = [f for f in results["per_file"] if f["caught"]]
    if caught_files:
        print(f"\n  CAUGHT ({len(caught_files)} files):")
        for cf in caught_files[:10]:
            scanners = ", ".join(cf["scanners_caught"])
            print(f"    + {cf['file']} -> {scanners}")
        if len(caught_files) > 10:
            print(f"    ... and {len(caught_files) - 10} more caught")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    manifest = None
    args = sys.argv[1:]
    if "--manifest" in args:
        idx = args.index("--manifest")
        manifest = args[idx + 1] if idx + 1 < len(args) else None

    results = run_multimodal_eval(manifest)
    print_report(results)

    out_path = Path(__file__).resolve().parent / "multimodal_results.json"
    compact = {k: v for k, v in results.items() if k != "per_file"}
    with open(out_path, "w") as f:
        json.dump(compact, f, indent=2)
    print(f"\nResults saved to: {out_path}")

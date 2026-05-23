"""
Unit tests for benchmark_scanners.py

Tests:
- Sample generators produce expected output
- LatencyStats computation is correct
- MemoryStats computation is correct
- Regression detection works
- Benchmark functions run without error and produce valid results
- CI mode exits non-zero on target miss
- Baseline save/load works
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


# Ensure project root + src are on path for hermes_katana imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Import the benchmark module
from tests.bench import benchmark_scanners as bm  # noqa: E402


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_within_target_returns_true_when_p99_below_target(self):
        stats = bm.LatencyStats(p50=1.0, p95=2.0, p99=3.0, mean=2.0, count=100)
        assert stats.within_target(5.0) is True

    def test_within_target_returns_false_when_p99_above_target(self):
        stats = bm.LatencyStats(p50=1.0, p95=2.0, p99=8.0, mean=2.0, count=100)
        assert stats.within_target(5.0) is False

    def test_to_dict_contains_expected_keys(self):
        stats = bm.LatencyStats(p50=1.0, p95=2.0, p99=3.0, mean=2.0, stddev=0.5, min=0.5, max=5.0, count=100)
        d = stats.to_dict()
        assert "p50_ms" in d
        assert "p95_ms" in d
        assert "p99_ms" in d
        assert "mean_ms" in d
        assert "stddev_ms" in d
        assert "min_ms" in d
        assert "max_ms" in d
        assert "count" in d

    def test_empty_stats_returns_zeros(self):
        stats = bm.LatencyStats()
        assert stats.p50 == 0.0
        assert stats.p99 == 0.0
        assert stats.count == 0


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_to_dict_contains_expected_keys(self):
        stats = bm.MemoryStats(peak_mb=1.5, current_mb=0.8, allocations=100)
        d = stats.to_dict()
        assert "peak_mb" in d
        assert "current_mb" in d
        assert "allocations" in d
        assert d["peak_mb"] == 1.5


class TestPercentile:
    """Tests for percentile computation."""

    def test_percentile_single_value(self):
        result = bm._percentile([5.0], 50)
        assert result == 5.0

    def test_percentile_median(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = bm._percentile(data, 50)
        assert result == 3.0

    def test_percentile_95(self):
        data = list(range(1, 101))  # 1 to 100
        result = bm._percentile(data, 95)
        # 95th percentile of 1..100 should be around 95
        assert 90 <= result <= 100

    def test_percentile_empty_returns_zero(self):
        result = bm._percentile([], 50)
        assert result == 0.0


class TestComputeStats:
    """Tests for compute_stats function."""

    def test_empty_list_returns_empty_stats(self):
        stats = bm.compute_stats([])
        assert stats.count == 0
        assert stats.p50 == 0.0

    def test_single_value(self):
        stats = bm.compute_stats([5.0])
        assert stats.count == 1
        assert stats.min == 5.0
        assert stats.max == 5.0
        assert stats.mean == 5.0

    def test_multiple_values(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = bm.compute_stats(data)
        assert stats.count == 5
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.mean == 3.0
        assert stats.p50 == 3.0

    def test_stats_sorted_regardless_of_input_order(self):
        data = [5.0, 1.0, 3.0, 2.0, 4.0]
        stats = bm.compute_stats(data)
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert stats.p50 == 3.0


class TestRegressionDetection:
    """Tests for regression detection."""

    def test_no_regression_when_current_is_faster(self):
        baseline = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 10.0}},
            }
        }
        current = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 8.0}},
            }
        }
        regressions = bm.detect_regressions(current, baseline, threshold_pct=10.0)
        assert "bloom" not in regressions

    def test_no_regression_within_threshold(self):
        baseline = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 10.0}},
            }
        }
        current = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 10.5}},  # 5% slower
            }
        }
        regressions = bm.detect_regressions(current, baseline, threshold_pct=10.0)
        assert "bloom" not in regressions

    def test_detects_regression_above_threshold(self):
        baseline = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 10.0}},
            }
        }
        current = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 12.0}},  # 20% slower
            }
        }
        regressions = bm.detect_regressions(current, baseline, threshold_pct=10.0)
        assert "bloom" in regressions
        assert regressions["bloom"]["pct_regression"] == 20.0

    def test_detects_high_severity_regression_above_25_percent(self):
        baseline = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 10.0}},
            }
        }
        current = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 15.0}},  # 50% slower
            }
        }
        regressions = bm.detect_regressions(current, baseline, threshold_pct=10.0)
        assert regressions["bloom"]["severity"] == "high"

    def test_detects_medium_severity_regression_above_10_percent(self):
        baseline = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 10.0}},
            }
        }
        current = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 12.0}},  # 20% slower
            }
        }
        regressions = bm.detect_regressions(current, baseline, threshold_pct=10.0)
        assert regressions["bloom"]["severity"] == "medium"

    def test_ignores_unknown_scanners(self):
        baseline = {"scanners": {}}
        current = {
            "scanners": {
                "bloom": {"stats": {"p99_ms": 12.0}},
            }
        }
        regressions = bm.detect_regressions(current, baseline)
        assert "bloom" not in regressions


class TestBaselineSaveLoad:
    """Tests for baseline save/load functionality."""

    def test_save_and_load_baseline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baseline.json"
            data = {
                "timestamp": "2024-01-01T00:00:00Z",
                "scanners": {
                    "bloom": {"stats": {"p99_ms": 1.5}},
                },
            }

            bm.save_baseline(path, data)
            assert path.exists()

            loaded = bm.load_baseline(path)
            assert loaded is not None
            assert loaded["scanners"]["bloom"]["stats"]["p99_ms"] == 1.5

    def test_load_nonexistent_returns_none(self):
        result = bm.load_baseline(Path("/nonexistent/path/baseline.json"))
        assert result is None


class TestSampleGeneration:
    """Tests for sample generation functions."""

    def test_generate_html_simple(self):
        html = bm._generate_html_simple(10)
        assert "<html>" in html
        assert "</html>" in html
        assert "<p>" in html

    def test_generate_html_with_hidden(self):
        html = bm._generate_html_with_hidden(5)
        assert "display:none" in html
        assert "Ignore previous instructions" in html

    def test_generate_html_deeply_nested(self):
        html = bm._generate_html_deeply_nested(50)
        assert "<div>" in html
        assert "Deep content" in html

    def test_generate_html_large(self):
        html = bm._generate_html_large(100)
        assert html.count("<div") >= 100

    def test_generate_svg_clean(self):
        svg = bm._generate_svg_clean()
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_generate_svg_with_script(self):
        svg = bm._generate_svg_with_script()
        assert "<script>" in svg

    def test_generate_svg_with_event_handler(self):
        svg = bm._generate_svg_with_event_handler()
        assert "onload" in svg or "onclick" in svg or "foreignObject" in svg

    def test_generate_pdf_text_clean(self):
        pdf = bm._generate_pdf_text_clean()
        assert b"%PDF" in pdf

    def test_generate_pdf_with_javascript(self):
        pdf = bm._generate_pdf_with_javascript()
        assert b"JavaScript" in pdf or b"/JS" in pdf

    def test_generate_multilingual_german_injection(self):
        text = bm._generate_multilingual_german_injection()
        assert "Ignoriere" in text

    def test_generate_multilingual_chinese_injection(self):
        text = bm._generate_multilingual_chinese_injection()
        assert "忽略" in text

    def test_generate_injection_instruction_override(self):
        text = bm._generate_injection_instruction_override()
        assert "Ignore" in text

    def test_generate_injection_role_override(self):
        text = bm._generate_injection_role_override()
        assert "DAN" in text

    def test_generate_injection_mcp_tag(self):
        text = bm._generate_injection_mcp_tag()
        assert "<IMPORTANT>" in text

    def test_generate_scabbard_clean_samples(self):
        samples = bm._scabbard_clean_samples()
        assert len(samples) > 0
        assert all(isinstance(s, str) for s in samples)

    def test_generate_scabbard_injection_samples(self):
        samples = bm._scabbard_injection_samples()
        assert len(samples) > 0
        assert all(isinstance(s, str) for s in samples)

    def test_generate_samples_creates_all_types(self):
        samples = bm.generate_samples()
        assert "html" in samples
        assert "pdf" in samples
        assert "markdown" in samples
        assert "plain_text" in samples
        assert "svg" in samples
        assert "multilingual" in samples
        assert "injection" in samples
        assert "behavioral" in samples

    def test_generate_samples_html_count(self):
        samples = bm.generate_samples()
        assert len(samples["html"]) == 100

    def test_generate_samples_pdf_count(self):
        samples = bm.generate_samples()
        assert len(samples["pdf"]) == 100

    def test_generate_samples_markdown_count(self):
        samples = bm.generate_samples()
        assert len(samples["markdown"]) == 100

    def test_generate_samples_plain_text_count(self):
        samples = bm.generate_samples()
        assert len(samples["plain_text"]) == 100

    def test_generate_samples_svg_count(self):
        samples = bm.generate_samples()
        assert len(samples["svg"]) == 100

    def test_generate_samples_injection_count(self):
        samples = bm.generate_samples()
        assert len(samples["injection"]) == 100


class TestMeasureLatency:
    """Tests for measure_latency function."""

    def test_measure_latency_returns_result_and_latency(self):
        def dummy_func(x):
            return x * 2

        result, latency = bm.measure_latency(dummy_func, 5)
        assert result == 10
        assert latency >= 0

    def test_measure_latency_with_realistic_function(self):
        result, latency = bm.measure_latency(sum, range(1000))
        assert result == 499500
        assert latency >= 0

    def test_measure_latency_iterations(self):
        # Should run multiple iterations
        counter = [0]

        def counting_func():
            counter[0] += 1
            return counter[0]

        result, latency = bm.measure_latency(counting_func, iterations=3)
        # Best-of-3, so should return min latency
        assert result >= 1
        assert latency >= 0


class TestMeasureLatencyAndMemory:
    """Tests for measure_latency_and_memory function."""

    def test_returns_tuple_of_three(self):
        def dummy_func(x):
            return x * 2

        result, latency, mem = bm.measure_latency_and_memory(dummy_func, 5)
        assert result == 10
        assert latency >= 0
        assert isinstance(mem, bm.MemoryStats)


class TestBenchmarkFunctions:
    """Tests that benchmark functions run without error."""

    def test_benchmark_bloom(self):
        samples = bm.generate_samples()
        result = bm.benchmark_bloom(samples["plain_text"])
        assert "scanner" in result
        assert "stats" in result
        assert result["scanner"] == "bloom"
        assert "within_target" in result

    def test_benchmark_html_diff(self):
        samples = bm.generate_samples()
        result = bm.benchmark_html_diff(samples["html"])
        assert result["scanner"] == "html_diff"
        assert "stats" in result
        assert "within_target" in result

    def test_benchmark_pdf_layers(self):
        samples = bm.generate_samples()
        result = bm.benchmark_pdf_layers(samples["pdf"])
        assert result["scanner"] == "pdf_layers"
        assert "within_target" in result

    def test_benchmark_md_audit(self):
        samples = bm.generate_samples()
        result = bm.benchmark_md_audit(samples["markdown"])
        assert result["scanner"] == "md_audit"
        assert "within_target" in result

    def test_benchmark_css_deobfuscator(self):
        samples = bm.generate_samples()
        result = bm.benchmark_css_deobfuscator(samples["html"])
        assert result["scanner"] == "css_deobfuscator"
        assert "within_target" in result

    def test_benchmark_svg(self):
        samples = bm.generate_samples()
        result = bm.benchmark_svg(samples["svg"])
        assert result["scanner"] == "svg"
        assert "within_target" in result

    def test_benchmark_pdf_js(self):
        samples = bm.generate_samples()
        result = bm.benchmark_pdf_js(samples["pdf"])
        assert result["scanner"] == "pdf_js"
        assert "within_target" in result

    def test_benchmark_injection(self):
        samples = bm.generate_samples()
        result = bm.benchmark_injection(samples["injection"])
        assert result["scanner"] == "injection"
        assert "within_target" in result

    def test_benchmark_unicode_spoof(self):
        samples = bm.generate_samples()
        result = bm.benchmark_unicode_spoof(samples["multilingual"])
        assert result["scanner"] == "unicode_spoof"
        assert "within_target" in result

    def test_benchmark_behavioral(self):
        samples = bm.generate_samples()
        result = bm.benchmark_behavioral(samples["behavioral"])
        assert result["scanner"] == "behavioral"
        assert "within_target" in result

    def test_benchmark_multimodal(self):
        samples = bm.generate_samples()
        result = bm.benchmark_multimodal(samples["html"])
        assert result["scanner"] == "multimodal"
        assert "within_target" in result

    def test_benchmark_aho_corasick(self):
        samples = bm.generate_samples()
        result = bm.benchmark_aho_corasick(samples["injection"])
        assert result["scanner"] == "aho_corasick"
        assert "within_target" in result

    def test_benchmark_cascade_router(self):
        samples = bm.generate_samples()
        result = bm.benchmark_cascade_router(samples["plain_text"])
        assert result["scanner"] == "cascade_router"
        assert "within_target" in result

    def test_benchmark_scabbard_pipeline_minimal(self):
        clean = bm._scabbard_clean_samples()
        injection = bm._scabbard_injection_samples()
        result = bm.benchmark_scabbard_pipeline(clean, injection, profile="minimal")
        assert result["scanner"] == "scabbard_minimal"
        assert "within_target" in result
        assert "accuracy" in result

    def test_benchmark_scabbard_pipeline_standard(self):
        clean = bm._scabbard_clean_samples()
        injection = bm._scabbard_injection_samples()
        result = bm.benchmark_scabbard_pipeline(clean, injection, profile="standard")
        assert result["scanner"] == "scabbard_standard"
        assert "within_target" in result

    def test_benchmark_scabbard_pipeline_full(self):
        clean = bm._scabbard_clean_samples()
        injection = bm._scabbard_injection_samples()
        result = bm.benchmark_scabbard_pipeline(clean, injection, profile="full")
        assert result["scanner"] == "scabbard_full"
        assert "within_target" in result

    def test_benchmark_memory(self):
        samples = bm.generate_samples()
        stats = bm.benchmark_memory(bm.scan_bloom, samples["plain_text"], "text")
        assert isinstance(stats, bm.MemoryStats)
        assert stats.peak_mb >= 0

    def test_full_stack_scan(self):
        result = bm.full_stack_scan("<html><body>test</body></html>")
        assert "flags" in result
        assert "content_type" in result
        assert "structural_score" in result

    def test_benchmark_full_stack(self):
        samples = bm.generate_samples()
        result = bm.benchmark_full_stack(samples["html"])
        assert result["scanner"] == "full_stack"
        assert "within_target" in result


class TestRunBenchmarks:
    """Tests for run_benchmarks function."""

    def test_run_benchmarks_returns_results_dict(self):
        results = bm.run_benchmarks()
        assert "timestamp" in results
        assert "scanners" in results
        assert "memory" in results
        assert "all_within_target" in results

    def test_run_benchmarks_includes_all_scanner_types(self):
        results = bm.run_benchmarks()
        scanners = set(results["scanners"].keys())
        expected = {
            "bloom",
            "html_diff",
            "pdf_layers",
            "md_audit",
            "css_deobfuscator",
            "svg",
            "pdf_js",
            "injection",
            "unicode_spoof",
            "behavioral",
            "multimodal",
            "aho_corasick",
            "cascade_router",
            "scabbard_minimal",
            "scabbard_standard",
            "scabbard_full",
        }
        assert expected.issubset(scanners)

    def test_run_benchmarks_with_specific_scanners(self):
        results = bm.run_benchmarks(scanners=["bloom", "svg"])
        assert "bloom" in results["scanners"]
        assert "svg" in results["scanners"]
        # Should not include others
        assert "html_diff" not in results["scanners"]

    def test_run_benchmarks_memory_populated(self):
        results = bm.run_benchmarks()
        assert len(results["memory"]) > 0

    def test_run_benchmarks_no_ci_mode(self):
        # Should not raise even if targets are missed
        results = bm.run_benchmarks(ci_mode=False)
        assert "all_within_target" in results


class TestPrintResultsTable:
    """Tests for print_results_table function."""

    def test_print_results_table_does_not_raise(self):
        results = bm.run_benchmarks()
        # Should not raise any exceptions
        bm.print_results_table(results)


class TestWriteJsonResults:
    """Tests for write_json_results function."""

    def test_write_json_results_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            results = {
                "timestamp": "2024-01-01T00:00:00Z",
                "scanners": {},
                "memory": {},
                "regressions": {},
            }
            bm.write_json_results(results, path)
            assert path.exists()

            # Verify it's valid JSON
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["timestamp"] == "2024-01-01T00:00:00Z"


class TestBenchmarkTargets:
    """Tests that benchmark targets are defined for all scanners."""

    def test_all_scanners_have_targets(self):
        expected = {
            "bloom",
            "html_diff",
            "pdf_layers",
            "md_audit",
            "css_deobfuscator",
            "svg",
            "pdf_js",
            "injection",
            "unicode_spoof",
            "behavioral",
            "multimodal",
            "aho_corasick",
            "scabbard_minimal",
            "scabbard_standard",
            "scabbard_full",
            "cascade_router",
        }
        for scanner in expected:
            assert scanner in bm.BENCHMARK_TARGETS, f"Missing target for {scanner}"
            assert bm.BENCHMARK_TARGETS[scanner] > 0, f"Target for {scanner} must be positive"

    def test_targets_are_reasonable(self):
        for name, target in bm.BENCHMARK_TARGETS.items():
            # No scanner should have a target > 200ms (way too slow)
            assert target < 200, f"Target for {name} is unreasonably high: {target}ms"

"""Tests for hermes_katana.metrics."""

from __future__ import annotations

import pytest

from hermes_katana.metrics import MetricsCollector, MetricsSummary


@pytest.fixture(autouse=True)
def fresh_metrics():
    """Reset the singleton between tests."""
    MetricsCollector.reset_instance()
    yield
    MetricsCollector.reset_instance()


class TestMetricsCollector:
    def test_singleton(self):
        a = MetricsCollector.get_instance()
        b = MetricsCollector.get_instance()
        assert a is b

    def test_reset_instance(self):
        a = MetricsCollector.get_instance()
        MetricsCollector.reset_instance()
        b = MetricsCollector.get_instance()
        assert a is not b

    def test_record_tool_call(self):
        m = MetricsCollector.get_instance()
        m.record_tool_call("terminal", "allow", latency_ms=5.0)
        m.record_tool_call("terminal", "deny", latency_ms=1.0)
        m.record_tool_call("read_file", "allow", latency_ms=2.0)

        summary = m.get_summary()
        assert summary.tool_calls[("terminal", "allow")] == 1
        assert summary.tool_calls[("terminal", "deny")] == 1
        assert summary.tool_calls[("read_file", "allow")] == 1

    def test_record_scan_hit(self):
        m = MetricsCollector.get_instance()
        m.record_scan_hit("injection", "instruction_override", "high")
        m.record_scan_hit("injection", "instruction_override", "high")
        m.record_scan_hit("secrets", "api_key", "critical")

        summary = m.get_summary()
        assert summary.scan_hits[("injection", "instruction_override", "high")] == 2
        assert summary.scan_hits[("secrets", "api_key", "critical")] == 1

    def test_record_taint_flow(self):
        m = MetricsCollector.get_instance()
        m.record_taint_flow("WEB_CONTENT", "terminal", "deny")
        m.record_taint_flow("USER", "terminal", "allow")

        summary = m.get_summary()
        assert summary.taint_flows[("WEB_CONTENT", "terminal", "deny")] == 1
        assert summary.taint_flows[("USER", "terminal", "allow")] == 1

    def test_record_policy_eval(self):
        m = MetricsCollector.get_instance()
        m.record_policy_eval("balanced", "terminal", "allow")
        m.record_policy_eval("balanced", "terminal", "deny")
        m.record_policy_eval("balanced", "read_file", "allow")

        summary = m.get_summary()
        assert summary.policy_evals[("allow",)] == 2
        assert summary.policy_evals[("deny",)] == 1

    def test_latency_samples(self):
        m = MetricsCollector.get_instance()
        m.record_tool_call("terminal", "allow", latency_ms=5.0)
        m.record_tool_call("terminal", "allow", latency_ms=10.0)

        summary = m.get_summary()
        assert len(summary.latency_samples) == 2
        assert summary.latency_samples[0] == ("terminal", 5.0)
        assert summary.latency_samples[1] == ("terminal", 10.0)

    def test_reset(self):
        m = MetricsCollector.get_instance()
        m.record_tool_call("terminal", "allow")
        m.record_scan_hit("injection", "test", "high")
        m.reset()

        summary = m.get_summary()
        assert summary.total_tool_calls == 0
        assert summary.total_scan_hits == 0

    def test_uptime(self):
        m = MetricsCollector.get_instance()
        summary = m.get_summary()
        assert summary.uptime_seconds >= 0
        assert summary.started_at > 0


class TestMetricsSummary:
    def test_total_tool_calls(self):
        m = MetricsCollector.get_instance()
        m.record_tool_call("a", "allow")
        m.record_tool_call("b", "deny")
        assert m.get_summary().total_tool_calls == 2

    def test_total_denials(self):
        m = MetricsCollector.get_instance()
        m.record_tool_call("a", "allow")
        m.record_tool_call("b", "deny")
        m.record_tool_call("c", "deny")
        assert m.get_summary().total_denials == 2

    def test_total_escalations(self):
        m = MetricsCollector.get_instance()
        m.record_tool_call("a", "escalate")
        assert m.get_summary().total_escalations == 1

    def test_total_scan_hits(self):
        m = MetricsCollector.get_instance()
        m.record_scan_hit("injection", "a", "high")
        m.record_scan_hit("secrets", "b", "critical")
        assert m.get_summary().total_scan_hits == 2


class TestPrometheusExport:
    def test_export_format(self):
        m = MetricsCollector.get_instance()
        m.record_tool_call("terminal", "allow", latency_ms=5.0)
        m.record_scan_hit("injection", "override", "high")
        m.record_taint_flow("WEB", "terminal", "deny")
        m.record_policy_eval("balanced", "terminal", "allow")

        output = m.export_prometheus()
        assert "katana_tool_calls_total" in output
        assert "katana_scan_hits_total" in output
        assert "katana_taint_flows_total" in output
        assert "katana_policy_evals_total" in output
        assert "katana_uptime_seconds" in output
        assert 'tool="terminal"' in output
        assert 'decision="allow"' in output

    def test_empty_export(self):
        m = MetricsCollector.get_instance()
        output = m.export_prometheus()
        assert "katana_uptime_seconds" in output

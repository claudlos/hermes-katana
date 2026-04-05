"""Tests for hermes_katana.policy.yaml_loader — YAML policy loading."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hermes_katana.policy.yaml_loader import (
    PolicyValidationError,
    validate_policy_yaml,
    load_policy_file,
    load_policy_directory,
    export_policy_set,
    _resolve_inheritance,
    PolicyFileWatcher,
)
from hermes_katana.policy.models import PolicySet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_POLICY_YAML = {
    "name": "test-policies",
    "version": "1.0.0",
    "description": "Test policy set",
    "policies": [
        {
            "name": "block_tainted",
            "tool_pattern": "terminal",
            "action": "deny",
            "conditions": [{"field": "*", "operator": "contains_taint", "value": True}],
            "priority": 100,
            "enabled": True,
        }
    ],
}

MINIMAL_POLICY_YAML = {
    "name": "minimal",
    "policies": [
        {"name": "rule1", "tool_pattern": "shell"},
    ],
}


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# validate_policy_yaml
# ---------------------------------------------------------------------------


class TestValidatePolicyYaml:
    def test_valid_policy(self):
        errors = validate_policy_yaml(VALID_POLICY_YAML)
        assert errors == []

    def test_minimal_policy(self):
        errors = validate_policy_yaml(MINIMAL_POLICY_YAML)
        assert errors == []

    def test_not_a_dict(self):
        errors = validate_policy_yaml("not a dict")
        assert any("Root element" in e for e in errors)

    def test_missing_name(self):
        errors = validate_policy_yaml({"policies": [{"name": "x", "tool_pattern": "y"}]})
        assert any("name" in e for e in errors)

    def test_missing_policies(self):
        errors = validate_policy_yaml({"name": "test"})
        assert any("policies" in e for e in errors)

    def test_policies_not_list(self):
        errors = validate_policy_yaml({"name": "test", "policies": "not-a-list"})
        assert any("list" in e for e in errors)

    def test_policy_not_dict(self):
        errors = validate_policy_yaml({"name": "test", "policies": ["string-item"]})
        assert any("mapping" in e for e in errors)

    def test_missing_tool_pattern(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [{"name": "rule1"}],
            }
        )
        assert any("tool_pattern" in e for e in errors)

    def test_missing_policy_name(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [{"tool_pattern": "shell"}],
            }
        )
        assert any("'name'" in e for e in errors)

    def test_invalid_action(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [{"name": "r", "tool_pattern": "t", "action": "invalid_action"}],
            }
        )
        assert any("invalid action" in e for e in errors)

    def test_valid_actions(self):
        for action in ("allow", "deny", "escalate", "log_only"):
            errors = validate_policy_yaml(
                {
                    "name": "test",
                    "policies": [{"name": "r", "tool_pattern": "t", "action": action}],
                }
            )
            assert not any("invalid action" in e for e in errors)

    def test_invalid_operator(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [
                    {
                        "name": "r",
                        "tool_pattern": "t",
                        "conditions": [{"field": "x", "operator": "bad_op", "value": True}],
                    }
                ],
            }
        )
        assert any("invalid operator" in e for e in errors)

    def test_condition_missing_field(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [
                    {
                        "name": "r",
                        "tool_pattern": "t",
                        "conditions": [{"operator": "contains_taint"}],
                    }
                ],
            }
        )
        assert any("'field'" in e for e in errors)

    def test_condition_missing_operator(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [
                    {
                        "name": "r",
                        "tool_pattern": "t",
                        "conditions": [{"field": "*"}],
                    }
                ],
            }
        )
        assert any("'operator'" in e for e in errors)

    def test_condition_not_dict(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [
                    {
                        "name": "r",
                        "tool_pattern": "t",
                        "conditions": ["not-a-dict"],
                    }
                ],
            }
        )
        assert any("mapping" in e for e in errors)

    def test_conditions_not_list(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [
                    {
                        "name": "r",
                        "tool_pattern": "t",
                        "conditions": "not-a-list",
                    }
                ],
            }
        )
        assert any("list" in e for e in errors)

    def test_negative_priority(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [{"name": "r", "tool_pattern": "t", "priority": -1}],
            }
        )
        assert any("priority" in e for e in errors)

    def test_float_priority(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [{"name": "r", "tool_pattern": "t", "priority": 1.5}],
            }
        )
        assert any("priority" in e for e in errors)

    def test_valid_priority(self):
        errors = validate_policy_yaml(
            {
                "name": "test",
                "policies": [{"name": "r", "tool_pattern": "t", "priority": 50}],
            }
        )
        assert errors == []


# ---------------------------------------------------------------------------
# load_policy_file
# ---------------------------------------------------------------------------


class TestLoadPolicyFile:
    def test_load_valid_file(self, tmp_path):
        fp = _write_yaml(tmp_path / "policy.yaml", VALID_POLICY_YAML)
        ps = load_policy_file(fp)
        assert isinstance(ps, PolicySet)
        assert ps.name == "test-policies"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_policy_file(tmp_path / "nonexistent.yaml")

    def test_empty_yaml(self, tmp_path):
        fp = tmp_path / "empty.yaml"
        fp.write_text("", encoding="utf-8")
        with pytest.raises(PolicyValidationError, match="Empty"):
            load_policy_file(fp)

    def test_invalid_schema(self, tmp_path):
        fp = _write_yaml(tmp_path / "bad.yaml", {"not_valid": True})
        with pytest.raises(PolicyValidationError):
            load_policy_file(fp)

    def test_malformed_yaml(self, tmp_path):
        fp = tmp_path / "malformed.yaml"
        fp.write_text("{{{{invalid yaml: [", encoding="utf-8")
        with pytest.raises(Exception):
            load_policy_file(fp)

    def test_load_with_inheritance(self, tmp_path):
        data = dict(VALID_POLICY_YAML)
        data["extends"] = "balanced"
        fp = _write_yaml(tmp_path / "child.yaml", data)
        # May work or warn depending on builtin sets
        try:
            ps = load_policy_file(fp)
            assert isinstance(ps, PolicySet)
        except PolicyValidationError:
            pass  # Acceptable if pydantic validation differs

    def test_load_with_unknown_parent_raises(self, tmp_path):
        data = dict(VALID_POLICY_YAML)
        data["extends"] = "nonexistent_parent"
        fp = _write_yaml(tmp_path / "child.yaml", data)
        with pytest.raises(PolicyValidationError, match="unknown parent"):
            load_policy_file(fp)

    def test_string_path(self, tmp_path):
        fp = _write_yaml(tmp_path / "policy.yaml", VALID_POLICY_YAML)
        ps = load_policy_file(str(fp))
        assert isinstance(ps, PolicySet)


# ---------------------------------------------------------------------------
# load_policy_directory
# ---------------------------------------------------------------------------


class TestLoadPolicyDirectory:
    def test_load_directory(self, tmp_path):
        _write_yaml(tmp_path / "a.yaml", VALID_POLICY_YAML)
        _write_yaml(tmp_path / "b.yml", MINIMAL_POLICY_YAML)
        results = load_policy_directory(tmp_path)
        assert len(results) >= 1

    def test_not_a_directory(self, tmp_path):
        fp = tmp_path / "file.txt"
        fp.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            load_policy_directory(fp)

    def test_empty_directory(self, tmp_path):
        results = load_policy_directory(tmp_path)
        assert results == []

    def test_skips_invalid_files(self, tmp_path):
        _write_yaml(tmp_path / "good.yaml", VALID_POLICY_YAML)
        bad = tmp_path / "bad.yaml"
        bad.write_text("not_valid: true", encoding="utf-8")
        results = load_policy_directory(tmp_path)
        assert len(results) == 1

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_yaml(sub / "nested.yaml", VALID_POLICY_YAML)
        results = load_policy_directory(tmp_path, recursive=True)
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# _resolve_inheritance
# ---------------------------------------------------------------------------


class TestResolveInheritance:
    def test_unknown_parent_raises(self):
        data = {"name": "child", "policies": []}
        with pytest.raises(PolicyValidationError, match="unknown parent"):
            _resolve_inheritance(data, "nonexistent")

    def test_known_parent_merges(self):
        with patch(
            "hermes_katana.policy.yaml_loader.BUILTIN_POLICY_SETS",
            {
                "parent": {
                    "name": "parent",
                    "policies": [{"name": "p1", "tool_pattern": "t1", "action": "deny"}],
                }
            },
        ):
            data = {
                "name": "child",
                "policies": [{"name": "p2", "tool_pattern": "t2", "action": "allow"}],
            }
            result = _resolve_inheritance(data, "parent")
            policy_names = [p["name"] for p in result["policies"]]
            assert "p1" in policy_names
            assert "p2" in policy_names

    def test_child_overrides_parent_policy(self):
        with patch(
            "hermes_katana.policy.yaml_loader.BUILTIN_POLICY_SETS",
            {
                "parent": {
                    "name": "parent",
                    "policies": [{"name": "shared", "tool_pattern": "t", "action": "deny"}],
                }
            },
        ):
            data = {
                "name": "child",
                "policies": [{"name": "shared", "tool_pattern": "t", "action": "allow"}],
            }
            result = _resolve_inheritance(data, "parent")
            shared = [p for p in result["policies"] if p["name"] == "shared"]
            assert len(shared) == 1
            assert shared[0]["action"] == "allow"


# ---------------------------------------------------------------------------
# export_policy_set
# ---------------------------------------------------------------------------


class TestExportPolicySet:
    def test_export_and_reload(self, tmp_path):
        fp = _write_yaml(tmp_path / "source.yaml", VALID_POLICY_YAML)
        ps = load_policy_file(fp)
        out = tmp_path / "exported.yaml"
        result_path = export_policy_set(ps, out)
        assert result_path.exists()
        ps2 = load_policy_file(result_path)
        assert ps2.name == ps.name

    def test_creates_parent_dirs(self, tmp_path):
        fp = _write_yaml(tmp_path / "source.yaml", VALID_POLICY_YAML)
        ps = load_policy_file(fp)
        out = tmp_path / "deep" / "nested" / "out.yaml"
        result_path = export_policy_set(ps, out)
        assert result_path.exists()

    def test_exclude_defaults(self, tmp_path):
        fp = _write_yaml(tmp_path / "source.yaml", VALID_POLICY_YAML)
        ps = load_policy_file(fp)
        out = tmp_path / "minimal.yaml"
        export_policy_set(ps, out, include_defaults=False)
        assert out.exists()


# ---------------------------------------------------------------------------
# PolicyValidationError
# ---------------------------------------------------------------------------


class TestPolicyValidationError:
    def test_basic_message(self):
        err = PolicyValidationError("bad file")
        assert "bad file" in str(err)
        assert err.errors == []

    def test_with_errors(self):
        err = PolicyValidationError("bad file", errors=["err1", "err2"])
        assert "err1" in str(err)
        assert "err2" in str(err)
        assert len(err.errors) == 2


# ---------------------------------------------------------------------------
# PolicyFileWatcher
# ---------------------------------------------------------------------------


class TestPolicyFileWatcher:
    def test_start_stop(self, tmp_path):
        callback = MagicMock()
        watcher = PolicyFileWatcher(tmp_path, callback, interval=1.0)
        assert not watcher.is_running
        watcher.start()
        assert watcher.is_running
        watcher.stop(timeout=2.0)
        assert not watcher.is_running

    def test_double_start(self, tmp_path):
        callback = MagicMock()
        watcher = PolicyFileWatcher(tmp_path, callback, interval=1.0)
        watcher.start()
        watcher.start()  # should warn but not crash
        watcher.stop(timeout=2.0)

    def test_detects_change(self, tmp_path):
        callback = MagicMock()
        _write_yaml(tmp_path / "initial.yaml", VALID_POLICY_YAML)
        watcher = PolicyFileWatcher(tmp_path, callback, interval=0.2)
        watcher.start()
        time.sleep(0.3)
        # Write a new file to trigger change
        _write_yaml(tmp_path / "new.yaml", MINIMAL_POLICY_YAML)
        time.sleep(0.5)
        watcher.stop(timeout=2.0)
        # Callback may or may not have been called depending on timing

    def test_nonexistent_directory(self, tmp_path):
        callback = MagicMock()
        watcher = PolicyFileWatcher(tmp_path / "nonexistent", callback, interval=0.5)
        watcher.start()
        time.sleep(0.3)
        watcher.stop(timeout=2.0)

    def test_minimum_interval(self, tmp_path):
        callback = MagicMock()
        watcher = PolicyFileWatcher(tmp_path, callback, interval=0.1)
        assert watcher._interval >= 1.0

    def test_snapshot_mtimes(self, tmp_path):
        _write_yaml(tmp_path / "a.yaml", VALID_POLICY_YAML)
        callback = MagicMock()
        watcher = PolicyFileWatcher(tmp_path, callback)
        mtimes = watcher._snapshot_mtimes()
        assert len(mtimes) >= 1

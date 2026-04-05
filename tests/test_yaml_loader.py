"""Comprehensive tests for hermes_katana.policy.yaml_loader (GAP — 16% coverage)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path: Path, filename: str, data) -> Path:
    """Write a Python object as YAML to a temp file."""
    fp = tmp_path / filename
    with open(fp, "w") as f:
        yaml.dump(data, f)
    return fp


def _valid_policy_data() -> dict:
    return {
        "name": "test-policies",
        "version": "1.0.0",
        "description": "Test policy set",
        "policies": [
            {
                "name": "test_allow_all",
                "tool_pattern": "*",
                "conditions": [],
                "action": "allow",
                "priority": 1,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestValidatePolicyYaml:
    """Tests for validate_policy_yaml()."""

    def test_valid_data_no_errors(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        errors = validate_policy_yaml(_valid_policy_data())
        assert errors == []

    def test_non_dict_root(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        errors = validate_policy_yaml("not a dict")
        assert any("mapping" in e.lower() or "dict" in e.lower() for e in errors)

    def test_missing_name(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = _valid_policy_data()
        del data["name"]
        errors = validate_policy_yaml(data)
        assert any("name" in e for e in errors)

    def test_missing_policies(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {"name": "test"}
        errors = validate_policy_yaml(data)
        assert any("policies" in e for e in errors)

    def test_policies_not_a_list(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {"name": "test", "policies": "not a list"}
        errors = validate_policy_yaml(data)
        assert any("list" in e for e in errors)

    def test_policy_not_a_dict(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {"name": "test", "policies": ["not a dict"]}
        errors = validate_policy_yaml(data)
        assert any("mapping" in e.lower() for e in errors)

    def test_policy_missing_required_fields(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {"name": "test", "policies": [{"action": "allow"}]}
        errors = validate_policy_yaml(data)
        assert any("name" in e for e in errors)
        assert any("tool_pattern" in e for e in errors)

    def test_invalid_action(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {
            "name": "test",
            "policies": [
                {
                    "name": "bad_action",
                    "tool_pattern": "*",
                    "action": "explode",
                }
            ],
        }
        errors = validate_policy_yaml(data)
        assert any("action" in e and "explode" in e for e in errors)

    def test_invalid_operator(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {
            "name": "test",
            "policies": [
                {
                    "name": "bad_op",
                    "tool_pattern": "*",
                    "conditions": [
                        {"field": "*", "operator": "nonexistent_op", "value": True}
                    ],
                }
            ],
        }
        errors = validate_policy_yaml(data)
        assert any("operator" in e and "nonexistent_op" in e for e in errors)

    def test_condition_missing_field(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {
            "name": "test",
            "policies": [
                {
                    "name": "missing_field",
                    "tool_pattern": "*",
                    "conditions": [{"operator": "contains_taint", "value": True}],
                }
            ],
        }
        errors = validate_policy_yaml(data)
        assert any("field" in e for e in errors)

    def test_condition_missing_operator(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {
            "name": "test",
            "policies": [
                {
                    "name": "missing_op",
                    "tool_pattern": "*",
                    "conditions": [{"field": "*", "value": True}],
                }
            ],
        }
        errors = validate_policy_yaml(data)
        assert any("operator" in e for e in errors)

    def test_negative_priority(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {
            "name": "test",
            "policies": [
                {
                    "name": "neg_pri",
                    "tool_pattern": "*",
                    "priority": -5,
                }
            ],
        }
        errors = validate_policy_yaml(data)
        assert any("priority" in e for e in errors)

    def test_valid_actions_accepted(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        for action in ("allow", "deny", "escalate", "log_only"):
            data = {
                "name": "test",
                "policies": [
                    {"name": f"test_{action}", "tool_pattern": "*", "action": action}
                ],
            }
            errors = validate_policy_yaml(data)
            assert not any("action" in e for e in errors), f"Action {action} should be valid"

    def test_conditions_not_a_list(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {
            "name": "test",
            "policies": [
                {"name": "bad_conds", "tool_pattern": "*", "conditions": "not a list"}
            ],
        }
        errors = validate_policy_yaml(data)
        assert any("conditions" in e and "list" in e for e in errors)

    def test_condition_not_a_dict(self):
        from hermes_katana.policy.yaml_loader import validate_policy_yaml

        data = {
            "name": "test",
            "policies": [
                {"name": "bad_cond", "tool_pattern": "*", "conditions": ["string"]}
            ],
        }
        errors = validate_policy_yaml(data)
        assert any("mapping" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# File loading tests
# ---------------------------------------------------------------------------


class TestLoadPolicyFile:
    """Tests for load_policy_file()."""

    def test_load_valid_file(self, tmp_path):
        from hermes_katana.policy.yaml_loader import load_policy_file

        fp = _write_yaml(tmp_path, "valid.yaml", _valid_policy_data())
        ps = load_policy_file(fp)
        assert ps.name == "test-policies"
        assert len(ps.policies) == 1

    def test_file_not_found(self):
        from hermes_katana.policy.yaml_loader import load_policy_file

        with pytest.raises(FileNotFoundError):
            load_policy_file("/nonexistent/path/policy.yaml")

    def test_empty_yaml_file(self, tmp_path):
        from hermes_katana.policy.yaml_loader import (
            load_policy_file,
            PolicyValidationError,
        )

        fp = tmp_path / "empty.yaml"
        fp.write_text("")
        with pytest.raises(PolicyValidationError, match="Empty"):
            load_policy_file(fp)

    def test_malformed_yaml(self, tmp_path):
        from hermes_katana.policy.yaml_loader import (
            load_policy_file,
            PolicyValidationError,
        )

        fp = tmp_path / "bad.yaml"
        fp.write_text("name: test\npolicies:\n  - name: [invalid\n")
        with pytest.raises((PolicyValidationError, yaml.YAMLError)):
            load_policy_file(fp)

    def test_missing_required_fields_raises(self, tmp_path):
        from hermes_katana.policy.yaml_loader import (
            load_policy_file,
            PolicyValidationError,
        )

        fp = _write_yaml(tmp_path, "no_policies.yaml", {"name": "test"})
        with pytest.raises(PolicyValidationError):
            load_policy_file(fp)

    def test_invalid_action_raises(self, tmp_path):
        from hermes_katana.policy.yaml_loader import (
            load_policy_file,
            PolicyValidationError,
        )

        data = _valid_policy_data()
        data["policies"][0]["action"] = "kaboom"
        fp = _write_yaml(tmp_path, "bad_action.yaml", data)
        with pytest.raises(PolicyValidationError):
            load_policy_file(fp)


# ---------------------------------------------------------------------------
# Directory loading tests
# ---------------------------------------------------------------------------


class TestLoadPolicyDirectory:
    """Tests for load_policy_directory()."""

    def test_load_directory(self, tmp_path):
        from hermes_katana.policy.yaml_loader import load_policy_directory

        _write_yaml(tmp_path, "a.yaml", _valid_policy_data())
        data2 = _valid_policy_data()
        data2["name"] = "second-set"
        _write_yaml(tmp_path, "b.yml", data2)

        results = load_policy_directory(tmp_path)
        assert len(results) == 2

    def test_not_a_directory(self):
        from hermes_katana.policy.yaml_loader import load_policy_directory

        with pytest.raises(NotADirectoryError):
            load_policy_directory("/nonexistent/dir")

    def test_skips_invalid_files(self, tmp_path):
        from hermes_katana.policy.yaml_loader import load_policy_directory

        _write_yaml(tmp_path, "good.yaml", _valid_policy_data())
        (tmp_path / "bad.yaml").write_text("name: broken")  # missing policies
        results = load_policy_directory(tmp_path)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Inheritance tests
# ---------------------------------------------------------------------------


class TestInheritance:
    """Tests for 'extends' field resolving to built-in policy sets."""

    def test_extends_balanced(self, tmp_path):
        from hermes_katana.policy.yaml_loader import load_policy_file

        data = {
            "name": "custom-extended",
            "extends": "balanced",
            "policies": [
                {"name": "custom_rule", "tool_pattern": "my_tool", "action": "deny", "priority": 200}
            ],
        }
        fp = _write_yaml(tmp_path, "extended.yaml", data)
        ps = load_policy_file(fp)
        assert ps.name == "custom-extended"
        # Should have balanced policies + custom one
        names = [p.name for p in ps.policies]
        assert "custom_rule" in names

    def test_extends_unknown_parent_raises(self, tmp_path):
        from hermes_katana.policy.yaml_loader import load_policy_file, PolicyValidationError

        data = {
            "name": "custom-unknown-parent",
            "extends": "nonexistent_parent",
            "policies": [
                {"name": "solo", "tool_pattern": "*", "action": "allow", "priority": 1}
            ],
        }
        fp = _write_yaml(tmp_path, "unknown_parent.yaml", data)
        with pytest.raises(PolicyValidationError, match="unknown parent"):
            load_policy_file(fp)


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------


class TestExportPolicySet:
    """Tests for export_policy_set()."""

    def test_roundtrip(self, tmp_path):
        from hermes_katana.policy.yaml_loader import load_policy_file, export_policy_set
        from hermes_katana.policy.models import PolicySet

        data = _valid_policy_data()
        ps = PolicySet.model_validate(data)
        out = tmp_path / "exported.yaml"
        export_policy_set(ps, out)
        assert out.exists()

        # Re-load and verify
        ps2 = load_policy_file(out)
        assert ps2.name == ps.name
        assert len(ps2.policies) == len(ps.policies)


# ---------------------------------------------------------------------------
# PolicyValidationError tests
# ---------------------------------------------------------------------------


class TestPolicyValidationError:
    """Tests for the PolicyValidationError exception."""

    def test_error_with_details(self):
        from hermes_katana.policy.yaml_loader import PolicyValidationError

        err = PolicyValidationError("bad file", errors=["missing name", "bad action"])
        assert "missing name" in str(err)
        assert "bad action" in str(err)
        assert err.errors == ["missing name", "bad action"]

    def test_error_without_details(self):
        from hermes_katana.policy.yaml_loader import PolicyValidationError

        err = PolicyValidationError("simple error")
        assert "simple error" in str(err)
        assert err.errors == []

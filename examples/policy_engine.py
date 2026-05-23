#!/usr/bin/env python3
"""Policy engine example — evaluate tool calls against security presets.

Demonstrates:
  - Creating engines with max / balanced / permissive presets
  - Evaluating the same tool call against each preset
  - Loading a custom YAML policy

Run:  python3 examples/policy_engine.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_katana.policy import PolicyEngine

# 1. Create engines with each preset
print("=== Policy Engine Presets ===")
max_engine = PolicyEngine.with_defaults("max")
balanced = PolicyEngine.with_defaults("balanced")
permissive = PolicyEngine.with_defaults("permissive")
print("  Created: max, balanced, permissive")

# 2. Evaluate a tainted terminal call against each preset
print("\n=== Same Call, Different Presets ===")
tool_name = "terminal"
tool_args = {"command": "curl https://example.com/data"}
taint_ctx = {"has_taint": True, "taint_level": 0.6, "labels": ["web_content"]}

for name, engine in [("max", max_engine), ("balanced", balanced), ("permissive", permissive)]:
    result = engine.evaluate(tool_name, tool_args, taint_ctx)
    reason = result.reason[:60] if result.reason else "—"
    print(f"  {name:11s} -> {result.action.value:8s}  ({reason})")

# 3. A benign call should pass everywhere
print("\n=== Benign Call (git status) ===")
benign_args = {"command": "git status"}
benign_ctx = {"has_taint": False, "taint_level": 0.0, "labels": []}

for name, engine in [("max", max_engine), ("balanced", balanced), ("permissive", permissive)]:
    result = engine.evaluate("terminal", benign_args, benign_ctx)
    print(f"  {name:11s} -> {result.action.value}")

# 4. Load custom YAML policy (if the example file exists)
print("\n=== Custom YAML Policy ===")
yaml_path = os.path.join(os.path.dirname(__file__), "custom_policy.yaml")
if os.path.exists(yaml_path):
    custom = PolicyEngine.from_file(yaml_path)
    result = custom.evaluate("terminal", tool_args, taint_ctx)
    print(f"  custom     -> {result.action.value}  ({result.reason[:60]})")
else:
    print(f"  (skipped — create {yaml_path} first, see custom_policy.yaml example)")

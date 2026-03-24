"""
Default policy definitions for HermesKatana.

Three built-in policy sets covering the security spectrum:

- **PARANOID**:   Deny every side-effecting tool if *any* argument carries
                  untrusted taint.  Suitable for high-security deployments.
- **BALANCED**:   Smart defaults — block dangerous tools (terminal, send_message,
                  memory writes, skill_manage) when tainted, allow read-only
                  tools, escalate ambiguous cases.
- **PERMISSIVE**: Log everything but only hard-block obvious exfiltration
                  patterns (e.g. tainted data piped to curl/wget).

Each set is defined as a plain Python dict that can be fed directly to
``PolicySet.model_validate()`` or exported to YAML via ``yaml.dump()``.

Tool categories referenced in these policies
---------------------------------------------
Side-effecting (dangerous):
    terminal, write_file, patch, send_message, memory (add/replace/remove),
    skill_manage, delegate_task, cronjob, browser_click, browser_type

Read-only (safe):
    read_file, search_files, browser_snapshot, browser_navigate, todo,
    process (list/poll/log), vision_analyze, text_to_speech
"""

from __future__ import annotations

from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# PARANOID — deny everything untrusted
# ═══════════════════════════════════════════════════════════════════════════

PARANOID_POLICIES: dict[str, Any] = {
    "name": "paranoid",
    "version": "1.0.0",
    "description": (
        "Maximum security: deny all side-effecting tool calls when any "
        "argument carries untrusted taint.  Read-only tools are escalated."
    ),
    "author": "HermesKatana",
    "metadata": {"security_level": "maximum", "recommended_for": "production"},
    "policies": [
        # -- Terminal ----------------------------------------------------------
        {
            "name": "paranoid_terminal_tainted",
            "description": "Block terminal execution when any argument is tainted.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "critical"],
        },
        {
            "name": "paranoid_terminal_clean",
            "description": "Escalate even clean terminal calls for human review.",
            "tool_pattern": "terminal",
            "conditions": [],
            "action": "escalate",
            "priority": 50,
            "tags": ["side-effect", "critical"],
        },
        # -- File writes -------------------------------------------------------
        {
            "name": "paranoid_write_file_tainted",
            "description": "Block file writes with tainted content.",
            "tool_pattern": "write_file",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "filesystem"],
        },
        {
            "name": "paranoid_patch_tainted",
            "description": "Block file patches with tainted content.",
            "tool_pattern": "patch",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "filesystem"],
        },
        # -- Messaging ---------------------------------------------------------
        {
            "name": "paranoid_send_message_tainted",
            "description": "Block outbound messages containing tainted data.",
            "tool_pattern": "send_message",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "exfiltration"],
        },
        # -- Memory ------------------------------------------------------------
        {
            "name": "paranoid_memory_write_tainted",
            "description": "Block memory mutations (add/replace/remove) with taint.",
            "tool_pattern": "memory",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "persistence"],
        },
        # -- Skill management --------------------------------------------------
        {
            "name": "paranoid_skill_manage_tainted",
            "description": "Block skill installation/modification with tainted data.",
            "tool_pattern": "skill_manage",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "code-execution"],
        },
        # -- Delegation --------------------------------------------------------
        {
            "name": "paranoid_delegate_tainted",
            "description": "Block task delegation when instructions are tainted.",
            "tool_pattern": "delegate_task",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "lateral-movement"],
        },
        # -- Cron --------------------------------------------------------------
        {
            "name": "paranoid_cronjob_tainted",
            "description": "Block cron job creation with tainted payloads.",
            "tool_pattern": "cronjob",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "persistence"],
        },
        # -- Browser side-effects ---------------------------------------------
        {
            "name": "paranoid_browser_interact_tainted",
            "description": "Block browser click/type when arguments carry taint.",
            "tool_pattern": "browser_*",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "browser"],
        },
        # -- Read-only tools (escalate) ----------------------------------------
        {
            "name": "paranoid_readonly_tainted",
            "description": "Escalate read-only tools when arguments are tainted.",
            "tool_pattern": "read_file",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 90,
            "tags": ["read-only"],
        },
        {
            "name": "paranoid_search_tainted",
            "description": "Escalate search_files with tainted patterns (info recon).",
            "tool_pattern": "search_files",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 90,
            "tags": ["read-only"],
        },
        # -- Catch-all ---------------------------------------------------------
        {
            "name": "paranoid_catchall_tainted",
            "description": "Deny any remaining tool with tainted arguments.",
            "tool_pattern": "*",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 10,
            "tags": ["catchall"],
        },
        {
            "name": "paranoid_catchall_clean",
            "description": "Allow clean calls to non-sensitive tools.",
            "tool_pattern": "*",
            "conditions": [],
            "action": "allow",
            "priority": 1,
            "tags": ["catchall"],
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# BALANCED — smart defaults
# ═══════════════════════════════════════════════════════════════════════════

BALANCED_POLICIES: dict[str, Any] = {
    "name": "balanced",
    "version": "1.0.0",
    "description": (
        "Smart defaults: block dangerous tools with untrusted taint, "
        "allow read-only tools, escalate edge cases."
    ),
    "author": "HermesKatana",
    "metadata": {"security_level": "moderate", "recommended_for": "development"},
    "policies": [
        # -- Terminal ----------------------------------------------------------
        {
            "name": "balanced_terminal_tainted",
            "description": "Block terminal when any argument carries untrusted taint.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "critical"],
        },
        {
            "name": "balanced_terminal_exfil_pattern",
            "description": "Block terminal commands matching exfiltration patterns.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern", "value": r".*(curl|wget|nc|ncat|ssh|scp|ftp)\s+.*"},
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 110,
            "tags": ["exfiltration", "critical"],
        },
        {
            "name": "balanced_terminal_clean",
            "description": "Allow clean terminal calls.",
            "tool_pattern": "terminal",
            "conditions": [],
            "action": "allow",
            "priority": 10,
            "tags": ["side-effect"],
        },
        # -- File writes -------------------------------------------------------
        {
            "name": "balanced_write_file_tainted",
            "description": "Escalate file writes with tainted content for review.",
            "tool_pattern": "write_file",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 95,
            "tags": ["side-effect", "filesystem"],
        },
        {
            "name": "balanced_patch_tainted",
            "description": "Escalate file patches with tainted content.",
            "tool_pattern": "patch",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 95,
            "tags": ["side-effect", "filesystem"],
        },
        # -- Messaging ---------------------------------------------------------
        {
            "name": "balanced_send_message_tainted",
            "description": "Block outbound messages containing tainted data (exfiltration risk).",
            "tool_pattern": "send_message",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "exfiltration"],
        },
        # -- Memory ------------------------------------------------------------
        {
            "name": "balanced_memory_tainted",
            "description": "Block memory writes with tainted data (poisoning risk).",
            "tool_pattern": "memory",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "persistence"],
        },
        # -- Skill management --------------------------------------------------
        {
            "name": "balanced_skill_manage_tainted",
            "description": "Block skill modifications with tainted data.",
            "tool_pattern": "skill_manage",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "code-execution"],
        },
        # -- Delegation --------------------------------------------------------
        {
            "name": "balanced_delegate_tainted",
            "description": "Escalate task delegation with tainted instructions.",
            "tool_pattern": "delegate_task",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 95,
            "tags": ["side-effect", "lateral-movement"],
        },
        # -- Cron --------------------------------------------------------------
        {
            "name": "balanced_cronjob_tainted",
            "description": "Block cron job creation with tainted payloads.",
            "tool_pattern": "cronjob",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "persistence"],
        },
        # -- Browser side-effects ---------------------------------------------
        {
            "name": "balanced_browser_type_tainted",
            "description": "Escalate browser typing with tainted text.",
            "tool_pattern": "browser_type",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 90,
            "tags": ["side-effect", "browser"],
        },
        {
            "name": "balanced_browser_click_tainted",
            "description": "Log browser clicks with tainted selectors.",
            "tool_pattern": "browser_click",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 80,
            "tags": ["side-effect", "browser"],
        },
        # -- Read-only tools (allow) -------------------------------------------
        {
            "name": "balanced_readonly_allow",
            "description": "Allow read-only tools even with taint (no side effects).",
            "tool_pattern": "read_file",
            "conditions": [],
            "action": "allow",
            "priority": 50,
            "tags": ["read-only"],
        },
        {
            "name": "balanced_search_allow",
            "description": "Allow search_files — read-only reconnaissance is acceptable.",
            "tool_pattern": "search_files",
            "conditions": [],
            "action": "allow",
            "priority": 50,
            "tags": ["read-only"],
        },
        # -- Catch-all ---------------------------------------------------------
        {
            "name": "balanced_catchall",
            "description": "Log any remaining tool call with taint for audit.",
            "tool_pattern": "*",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 5,
            "tags": ["catchall"],
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# PERMISSIVE — log only, block exfiltration
# ═══════════════════════════════════════════════════════════════════════════

PERMISSIVE_POLICIES: dict[str, Any] = {
    "name": "permissive",
    "version": "1.0.0",
    "description": (
        "Minimal friction: log all tainted tool calls but only hard-block "
        "obvious exfiltration patterns."
    ),
    "author": "HermesKatana",
    "metadata": {"security_level": "low", "recommended_for": "experimentation"},
    "policies": [
        # -- Exfiltration via terminal -----------------------------------------
        {
            "name": "permissive_terminal_exfil_curl",
            "description": "Block curl/wget with tainted data (exfiltration).",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern", "value": r".*(curl|wget)\s+.*"},
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["exfiltration", "critical"],
        },
        {
            "name": "permissive_terminal_exfil_netcat",
            "description": "Block netcat/ncat/socat with tainted data.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern", "value": r".*(nc|ncat|socat)\s+.*"},
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["exfiltration", "critical"],
        },
        {
            "name": "permissive_terminal_exfil_ssh",
            "description": "Block ssh/scp/sftp with tainted data.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern", "value": r".*(ssh|scp|sftp|rsync)\s+.*"},
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["exfiltration", "critical"],
        },
        {
            "name": "permissive_terminal_exfil_dns",
            "description": "Block DNS-based exfiltration with tainted data.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern", "value": r".*(dig|nslookup|host)\s+.*"},
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["exfiltration", "critical"],
        },
        {
            "name": "permissive_terminal_tainted_log",
            "description": "Log all other tainted terminal calls.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 50,
            "tags": ["side-effect"],
        },
        # -- Messaging exfiltration --------------------------------------------
        {
            "name": "permissive_send_message_exfil",
            "description": "Block send_message when content source is web_content.",
            "tool_pattern": "send_message",
            "conditions": [
                {"field": "*", "operator": "source_is", "value": "web_content"},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["exfiltration"],
        },
        {
            "name": "permissive_send_message_tainted_log",
            "description": "Log tainted send_message calls from other sources.",
            "tool_pattern": "send_message",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 50,
            "tags": ["side-effect"],
        },
        # -- Skill backdoor prevention -----------------------------------------
        {
            "name": "permissive_skill_manage_web_taint",
            "description": "Block skill modifications sourced from web content.",
            "tool_pattern": "skill_manage",
            "conditions": [
                {"field": "*", "operator": "source_is", "value": "web_content"},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["code-execution", "critical"],
        },
        {
            "name": "permissive_skill_manage_log",
            "description": "Log all other skill modifications with taint.",
            "tool_pattern": "skill_manage",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 50,
            "tags": ["code-execution"],
        },
        # -- Memory poisoning --------------------------------------------------
        {
            "name": "permissive_memory_web_taint",
            "description": "Escalate memory writes sourced from web content.",
            "tool_pattern": "memory",
            "conditions": [
                {"field": "*", "operator": "source_is", "value": "web_content"},
            ],
            "action": "escalate",
            "priority": 90,
            "tags": ["persistence"],
        },
        # -- Delegation --------------------------------------------------------
        {
            "name": "permissive_delegate_log",
            "description": "Log delegated tasks with tainted instructions.",
            "tool_pattern": "delegate_task",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 50,
            "tags": ["lateral-movement"],
        },
        # -- Cron persistence --------------------------------------------------
        {
            "name": "permissive_cronjob_tainted",
            "description": "Escalate cron jobs with tainted payloads.",
            "tool_pattern": "cronjob",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 90,
            "tags": ["persistence"],
        },
        # -- Catch-all ---------------------------------------------------------
        {
            "name": "permissive_catchall_tainted",
            "description": "Log any tainted tool call not matched above.",
            "tool_pattern": "*",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 5,
            "tags": ["catchall"],
        },
        {
            "name": "permissive_catchall_clean",
            "description": "Allow all clean tool calls.",
            "tool_pattern": "*",
            "conditions": [],
            "action": "allow",
            "priority": 1,
            "tags": ["catchall"],
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Registry — easy access by name
# ═══════════════════════════════════════════════════════════════════════════

BUILTIN_POLICY_SETS: dict[str, dict[str, Any]] = {
    "paranoid": PARANOID_POLICIES,
    "balanced": BALANCED_POLICIES,
    "permissive": PERMISSIVE_POLICIES,
}
"""Map of built-in policy set names to their raw dict definitions."""

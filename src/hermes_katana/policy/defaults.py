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
            "description": "Deny all remaining calls — paranoid denies unknown tools.",
            "tool_pattern": "*",
            "conditions": [],
            "action": "deny",
            "priority": 1,
            "tags": ["catchall", "unknown-tool"],
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# BALANCED — smart defaults
# ═══════════════════════════════════════════════════════════════════════════

BALANCED_POLICIES: dict[str, Any] = {
    "name": "balanced",
    "version": "2.0.0",
    "description": (
        "Smart defaults: allow clean calls without hesitation, block dangerous "
        "tainted tools, escalate ambiguous cases, and use taint-level gradients "
        "(1-3 low, 4-6 medium, 7-10 high) for nuanced decisions. Benign commands "
        "(ls, cat, git status, pip install, etc.) are allowed even with mild taint."
    ),
    "author": "HermesKatana",
    "metadata": {
        "security_level": "moderate",
        "recommended_for": "development",
        "taint_levels": {
            "low": "1-3: allow benign ops, log-only for reads",
            "medium": "4-6: escalate side-effects, allow reads",
            "high": "7-10: deny dangerous, escalate everything else",
        },
    },
    "policies": [
        # ── Terminal ──────────────────────────────────────────────────────
        # Priority order matters: exfil > dangerous+high > benign+low > generic taint > clean
        {
            "name": "balanced_terminal_exfil_pattern",
            "description": "Block terminal commands matching exfiltration patterns when tainted.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern", "value": r".*(curl|wget|nc|ncat|ssh|scp|ftp|rsync|socat)\s+.*"},
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 120,
            "tags": ["exfiltration", "critical"],
        },
        {
            "name": "balanced_terminal_dangerous_high_taint",
            "description": "Block terminal with dangerous patterns AND high taint (level >= 7).",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern", "value": r".*(rm\s+-rf|mkfs|dd\s+if=|chmod\s+777|>\s*/etc/|eval\s|base64\s+-d).*"},
                {"field": "*", "operator": "taint_level_gte", "value": 7},
            ],
            "action": "deny",
            "priority": 115,
            "tags": ["side-effect", "critical", "high-taint"],
        },
        {
            "name": "balanced_terminal_dangerous_medium_taint",
            "description": "Escalate terminal with dangerous patterns AND medium taint (4-6).",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern", "value": r".*(rm\s+-rf|mkfs|dd\s+if=|chmod\s+777|>\s*/etc/|eval\s|base64\s+-d).*"},
                {"field": "*", "operator": "taint_level_gte", "value": 4},
            ],
            "action": "escalate",
            "priority": 110,
            "tags": ["side-effect", "critical", "medium-taint"],
        },
        {
            "name": "balanced_terminal_benign_low_taint",
            "description": (
                "Allow benign commands (ls, cat, echo, pwd, pip, npm, git status/log/diff, "
                "etc.) even with low taint (level <= 3). These have no meaningful side effects."
            ),
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern",
                 "value": r"^\s*(sudo\s+)?(ls|cat|echo|pwd|cd|head|tail|wc|sort|uniq|grep|find|which|whoami|date|env|printenv|uname|df|du|free|ps|top|file|stat|id|hostname|uptime|tree|less|more|diff|basename|dirname|realpath|python3?|node|cargo|rustc|make|cmake)\b.*"},
                {"field": "*", "operator": "taint_level_lte", "value": 3},
            ],
            "action": "allow",
            "priority": 105,
            "tags": ["benign-whitelist", "low-taint"],
        },
        {
            "name": "balanced_terminal_git_readonly_low_taint",
            "description": "Allow read-only git subcommands with low taint.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern",
                 "value": r"^\s*(sudo\s+)?git\s+(status|log|diff|show|branch|tag|remote|stash|describe|shortlog|reflog|config|ls-files|ls-tree)\b.*"},
                {"field": "*", "operator": "taint_level_lte", "value": 3},
            ],
            "action": "allow",
            "priority": 105,
            "tags": ["benign-whitelist", "git", "low-taint"],
        },
        {
            "name": "balanced_terminal_pip_npm_low_taint",
            "description": "Allow pip/npm install with low taint (common dev workflow).",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern",
                 "value": r"^\s*(sudo\s+)?(pip3?|npm|npx|yarn)\s+(install|list|show|info|search|outdated|audit)\b.*"},
                {"field": "*", "operator": "taint_level_lte", "value": 3},
            ],
            "action": "allow",
            "priority": 105,
            "tags": ["benign-whitelist", "package-manager", "low-taint"],
        },
        {
            "name": "balanced_terminal_benign_medium_taint",
            "description": "Escalate benign commands with medium taint (4-6) — probably fine but check.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "command", "operator": "matches_pattern",
                 "value": r"^\s*(sudo\s+)?(ls|cat|echo|pwd|head|tail|wc|grep|find|which|whoami|date|env|git\s+(status|log|diff))\b.*"},
                {"field": "*", "operator": "taint_level_gte", "value": 4},
                {"field": "*", "operator": "taint_level_lte", "value": 6},
            ],
            "action": "escalate",
            "priority": 100,
            "tags": ["benign-whitelist", "medium-taint"],
        },
        {
            "name": "balanced_terminal_high_taint",
            "description": "Deny any terminal call with high taint (level >= 7).",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "*", "operator": "taint_level_gte", "value": 7},
            ],
            "action": "deny",
            "priority": 95,
            "tags": ["side-effect", "critical", "high-taint"],
        },
        {
            "name": "balanced_terminal_medium_taint",
            "description": "Escalate non-benign terminal calls with medium taint (4-6).",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "*", "operator": "taint_level_gte", "value": 4},
            ],
            "action": "escalate",
            "priority": 90,
            "tags": ["side-effect", "medium-taint"],
        },
        {
            "name": "balanced_terminal_low_taint",
            "description": "Escalate non-benign terminal calls with low taint (1-3) — not whitelisted.",
            "tool_pattern": "terminal",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 85,
            "tags": ["side-effect", "low-taint"],
        },
        {
            "name": "balanced_terminal_clean",
            "description": "Allow clean terminal calls (no taint) without hesitation.",
            "tool_pattern": "terminal",
            "conditions": [],
            "action": "allow",
            "priority": 10,
            "tags": ["side-effect"],
        },
        # ── File writes ──────────────────────────────────────────────────
        {
            "name": "balanced_write_file_high_taint",
            "description": "Deny file writes with high taint (level >= 7).",
            "tool_pattern": "write_file",
            "conditions": [
                {"field": "*", "operator": "taint_level_gte", "value": 7},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "filesystem", "high-taint"],
        },
        {
            "name": "balanced_write_file_tainted",
            "description": "Escalate file writes with low/medium taint for review.",
            "tool_pattern": "write_file",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 95,
            "tags": ["side-effect", "filesystem"],
        },
        {
            "name": "balanced_patch_high_taint",
            "description": "Deny file patches with high taint (level >= 7).",
            "tool_pattern": "patch",
            "conditions": [
                {"field": "*", "operator": "taint_level_gte", "value": 7},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "filesystem", "high-taint"],
        },
        {
            "name": "balanced_patch_tainted",
            "description": "Escalate file patches with low/medium taint.",
            "tool_pattern": "patch",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 95,
            "tags": ["side-effect", "filesystem"],
        },
        # ── Messaging ─────────────────────────────────────────────────────
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
        # ── Memory ────────────────────────────────────────────────────────
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
        # ── Skill management ──────────────────────────────────────────────
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
        # ── Delegation ────────────────────────────────────────────────────
        {
            "name": "balanced_delegate_high_taint",
            "description": "Deny task delegation with high taint.",
            "tool_pattern": "delegate_task",
            "conditions": [
                {"field": "*", "operator": "taint_level_gte", "value": 7},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "lateral-movement", "high-taint"],
        },
        {
            "name": "balanced_delegate_tainted",
            "description": "Escalate task delegation with low/medium taint.",
            "tool_pattern": "delegate_task",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 95,
            "tags": ["side-effect", "lateral-movement"],
        },
        # ── Cron ──────────────────────────────────────────────────────────
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
        # ── Browser side-effects ──────────────────────────────────────────
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
        # ── Read-only tools (always allow) ────────────────────────────────
        {
            "name": "balanced_readonly_allow",
            "description": "Allow read-only tools always — no side effects.",
            "tool_pattern": "read_file",
            "conditions": [],
            "action": "allow",
            "priority": 50,
            "tags": ["read-only"],
        },
        {
            "name": "balanced_search_allow",
            "description": "Allow search_files always — read-only reconnaissance is fine.",
            "tool_pattern": "search_files",
            "conditions": [],
            "action": "allow",
            "priority": 50,
            "tags": ["read-only"],
        },
        {
            "name": "balanced_browser_snapshot_allow",
            "description": "Allow browser_snapshot always — read-only.",
            "tool_pattern": "browser_snapshot",
            "conditions": [],
            "action": "allow",
            "priority": 50,
            "tags": ["read-only"],
        },
        # ── Tainted read-only tools (log only) ────────────────────────────
        {
            "name": "balanced_readonly_tainted_log",
            "description": "Log-only for tainted read-only tools (vision, todo, process list).",
            "tool_pattern": "vision_analyze",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 40,
            "tags": ["read-only", "tainted-read"],
        },
        {
            "name": "balanced_todo_tainted_log",
            "description": "Log-only for tainted todo access.",
            "tool_pattern": "todo",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 40,
            "tags": ["read-only", "tainted-read"],
        },
        {
            "name": "balanced_process_tainted_log",
            "description": "Log-only for tainted process inspection.",
            "tool_pattern": "process",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "log_only",
            "priority": 40,
            "tags": ["read-only", "tainted-read"],
        },
        # ── Catch-all ─────────────────────────────────────────────────────
        {
            "name": "balanced_notes_tainted",
            "description": "Deny notes tool when arguments carry web taint (data poisoning risk).",
            "tool_pattern": "notes",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "deny",
            "priority": 100,
            "tags": ["side-effect", "persistence", "notes"],
        },
        {
            "name": "balanced_notes_clean",
            "description": "Allow clean notes calls (no taint).",
            "tool_pattern": "notes",
            "conditions": [],
            "action": "allow",
            "priority": 50,
            "tags": ["known-tool", "notes"],
        },
        {
            "name": "balanced_catchall_high_taint",
            "description": "Escalate any uncovered tool with high taint.",
            "tool_pattern": "*",
            "conditions": [
                {"field": "*", "operator": "taint_level_gte", "value": 7},
            ],
            "action": "escalate",
            "priority": 8,
            "tags": ["catchall", "high-taint"],
        },
        {
            "name": "balanced_catchall",
            "description": "Escalate any remaining tainted tool call for human review.",
            "tool_pattern": "*",
            "conditions": [
                {"field": "*", "operator": "contains_taint", "value": True},
            ],
            "action": "escalate",
            "priority": 5,
            "tags": ["catchall"],
        },
        {
            "name": "balanced_catchall_clean",
            "description": "Escalate remaining unknown clean tool calls for human review.",
            "tool_pattern": "*",
            "conditions": [],
            "action": "escalate",
            "priority": 1,
            "tags": ["catchall", "unknown-tool"],
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
            "description": "Log unknown clean tool calls for audit trail.",
            "tool_pattern": "*",
            "conditions": [],
            "action": "log_only",
            "priority": 1,
            "tags": ["catchall", "unknown-tool"],
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

"""
Dangerous command detector for HermesKatana.

Improved from hermes-aegis with 40+ patterns covering:
- Filesystem destruction (rm -rf, format, mkfs)
- SQL injection (DROP, UNION SELECT, etc.)
- System operations (shutdown, reboot, kill)
- Fork bombs (:(){ :|:& };:)
- Pipe-to-shell (curl | bash, wget | sh)
- SSH exfiltration (scp, ssh tunneling)
- Network tunneling (netcat, socat, ngrok)
- Container escape (nsenter, docker.sock)
- Privilege escalation (sudo, setuid, capabilities)
- Crypto mining (xmrig, minergate)
- Data staging (tar+curl, zip+upload)

Each pattern includes severity level, category, and explanation of the risk.

Performance: <1ms for typical command strings. All patterns precompiled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class CommandSeverity(str, Enum):
    """Severity levels for dangerous command findings."""

    CRITICAL = "critical"
    """Immediate system compromise or data destruction."""

    HIGH = "high"
    """Significant security risk or data exposure."""

    MEDIUM = "medium"
    """Moderate risk, may be legitimate in some contexts."""

    LOW = "low"
    """Minor risk, suspicious but often legitimate."""


class CommandCategory(str, Enum):
    """Categories of dangerous commands."""

    FILESYSTEM_DESTRUCTION = "filesystem_destruction"
    """Commands that destroy or corrupt filesystem data."""

    SQL_INJECTION = "sql_injection"
    """SQL injection patterns that manipulate databases."""

    SYSTEM_OPERATION = "system_operation"
    """System-level operations (shutdown, reboot, service control)."""

    FORK_BOMB = "fork_bomb"
    """Resource exhaustion via process forking."""

    PIPE_TO_SHELL = "pipe_to_shell"
    """Downloading and executing untrusted code."""

    SSH_EXFILTRATION = "ssh_exfiltration"
    """Data exfiltration or tunneling via SSH/SCP."""

    NETWORK_TUNNELING = "network_tunneling"
    """Network tunneling and reverse shells."""

    CONTAINER_ESCAPE = "container_escape"
    """Attempts to escape container isolation."""

    PRIVILEGE_ESCALATION = "privilege_escalation"
    """Attempts to gain elevated privileges."""

    CRYPTO_MINING = "crypto_mining"
    """Cryptocurrency mining operations."""

    DATA_STAGING = "data_staging"
    """Staging data for exfiltration (archive + upload)."""

    REVERSE_SHELL = "reverse_shell"
    """Reverse shell connections for remote access."""

    CREDENTIAL_ACCESS = "credential_access"
    """Accessing stored credentials or auth databases."""

    CODE_EXECUTION = "code_execution"
    """Arbitrary code execution patterns."""

    INFORMATION_GATHERING = "information_gathering"
    """Reconnaissance and information gathering."""


@dataclass(frozen=True, slots=True)
class CommandFinding:
    """A dangerous command detection finding.

    Attributes:
        pattern_name: Name of the pattern that matched.
        severity: How dangerous this command is.
        matched_text: The text that triggered the detection.
        category: Type of dangerous command.
        position: (start, end) positions in the command string.
        description: Human-readable explanation of the risk.
        confidence: Detection confidence 0.0-1.0.
    """

    pattern_name: str
    severity: CommandSeverity
    matched_text: str
    category: CommandCategory
    position: tuple[int, int]
    description: str
    confidence: float = 0.95


# ---------------------------------------------------------------------------
# Command patterns (40+)
# Each: (name, regex, category, severity, description, flags)
# ---------------------------------------------------------------------------

_COMMAND_PATTERNS: list[tuple[str, re.Pattern, CommandCategory, CommandSeverity, str]] = []


def _cp(
    name: str,
    pattern: str,
    category: CommandCategory,
    severity: CommandSeverity,
    description: str,
    flags: int = re.IGNORECASE,
) -> None:
    """Register a command pattern."""
    _COMMAND_PATTERNS.append((
        name,
        re.compile(pattern, flags),
        category,
        severity,
        description,
    ))


# ============================
# FILESYSTEM DESTRUCTION
# ============================

_cp(
    "rm_rf_root",
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)\s+(/|\$\{?HOME\}?|~|\.\.|/etc|/var|/usr|/boot|/sys|/proc|\*)",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.CRITICAL,
    "Recursive force deletion of critical directory. Can destroy entire filesystem.",
)

_cp(
    "rm_rf_wildcard",
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)\s+\S*\*",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.HIGH,
    "Recursive force deletion with wildcard. May delete unintended files.",
)

_cp(
    "mkfs_format",
    r"\b(?:mkfs|mke2fs|mkfs\.\w+)\s+(?:/dev/(?:sd|hd|nvme|vd)\w+|/dev/mapper/)",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.CRITICAL,
    "Filesystem format command - destroys all data on target device.",
)

_cp(
    "dd_disk_overwrite",
    r"\bdd\s+.*\bof=/dev/(?:sd|hd|nvme|vd)\w+",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.CRITICAL,
    "Direct disk write with dd - can overwrite entire disk/partition.",
)

_cp(
    "shred_wipe",
    r"(?:^|(?<=[\s;|&]))(?:sudo\s+)?(?:shred|wipe|secure-delete|srm)\s+(?:--|/|-[a-zA-Z])\S",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.HIGH,
    "Secure deletion/wiping - irrecoverably destroys data.",
)

_cp(
    "truncate_dev_null",
    r"(?:>\s*/dev/(?:sd|hd)\w+|>\s*/etc/(?:passwd|shadow|hosts)|\btruncate\s+.*(?:/etc/|/var/|/usr/))",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.CRITICAL,
    "Redirecting output to device or overwriting critical system file.",
)

_cp(
    "chmod_dangerous",
    r"\bchmod\s+(?:-R\s+)?(?:777|666|a\+rwx)\s+/",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.HIGH,
    "Setting world-writable permissions on system directories.",
)

# ============================
# SQL INJECTION
# ============================

_cp(
    "sql_drop",
    r"\b(?:DROP|TRUNCATE)\s+(?:TABLE|DATABASE|SCHEMA|INDEX)\s+",
    CommandCategory.SQL_INJECTION,
    CommandSeverity.CRITICAL,
    "SQL DROP/TRUNCATE statement - destroys database objects.",
)

_cp(
    "sql_union_select",
    r"\bUNION\s+(?:ALL\s+)?SELECT\s+",
    CommandCategory.SQL_INJECTION,
    CommandSeverity.HIGH,
    "SQL UNION SELECT - classic SQL injection data extraction.",
)

_cp(
    "sql_or_true",
    r"(?:'\s*OR\s+'1'\s*=\s*'1|'\s*OR\s+1\s*=\s*1|\"?\s*OR\s+\"?1\"?\s*=\s*\"?1)",
    CommandCategory.SQL_INJECTION,
    CommandSeverity.HIGH,
    "SQL OR 1=1 injection - authentication bypass.",
)

_cp(
    "sql_comment_injection",
    r"(?:--|#|/\*)\s*(?:DROP|DELETE|INSERT|UPDATE|EXEC|UNION)",
    CommandCategory.SQL_INJECTION,
    CommandSeverity.MEDIUM,
    "SQL comment followed by dangerous statement.",
)

_cp(
    "sql_exec",
    r"\b(?:EXEC(?:UTE)?|xp_cmdshell|sp_executesql)\s*\(",
    CommandCategory.SQL_INJECTION,
    CommandSeverity.CRITICAL,
    "SQL command execution - can run OS commands via database.",
)

_cp(
    "sql_into_outfile",
    r"\bINTO\s+(?:OUTFILE|DUMPFILE)\s+",
    CommandCategory.SQL_INJECTION,
    CommandSeverity.HIGH,
    "SQL file write via INTO OUTFILE/DUMPFILE.",
)

# ============================
# SYSTEM OPERATIONS
# ============================

_cp(
    "shutdown_reboot",
    r"(?:^|(?<=[\s;|&]))(?:sudo\s+)?(?:shutdown|reboot|halt|poweroff|init\s+[06])\s+(?:-[a-zA-Z]|\+\d|now|0)\b",
    CommandCategory.SYSTEM_OPERATION,
    CommandSeverity.CRITICAL,
    "System shutdown/reboot command.",
)

_cp(
    "kill_all",
    r"\b(?:killall|pkill)\s+(?:-9\s+)?(?:\w+|\.)",
    CommandCategory.SYSTEM_OPERATION,
    CommandSeverity.HIGH,
    "Mass process kill command.",
)

_cp(
    "systemctl_disable",
    r"\bsystemctl\s+(?:disable|stop|mask)\s+(?:firewalld?|iptables|apparmor|selinux|auditd|fail2ban|sshd)",
    CommandCategory.SYSTEM_OPERATION,
    CommandSeverity.CRITICAL,
    "Disabling security services (firewall, SELinux, auditing).",
)

_cp(
    "sysctl_dangerous",
    r"\bsysctl\s+(?:-w\s+)?(?:kernel\.(?:randomize_va_space|modules_disabled)|net\.ipv4\.ip_forward)\s*=\s*[01]",
    CommandCategory.SYSTEM_OPERATION,
    CommandSeverity.HIGH,
    "Modifying critical kernel parameters (ASLR, modules, IP forwarding).",
)

# ============================
# FORK BOMBS
# ============================

_cp(
    "bash_fork_bomb",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;?\s*:",
    CommandCategory.FORK_BOMB,
    CommandSeverity.CRITICAL,
    "Classic bash fork bomb - exhausts system resources.",
    0,  # No IGNORECASE for this pattern
)

_cp(
    "fork_bomb_variants",
    r"(?:\b\w+\(\)\s*\{\s*\w+\s*\|\s*\w+\s*&\s*\}\s*;?\s*\w+\b|while\s+true\s*;\s*do\s+\w+\s*&\s*done)",
    CommandCategory.FORK_BOMB,
    CommandSeverity.CRITICAL,
    "Fork bomb variant - recursive process spawning.",
)

_cp(
    "python_fork_bomb",
    r"import\s+os\s*;?\s*while\s+True\s*:\s*os\.fork\(\)",
    CommandCategory.FORK_BOMB,
    CommandSeverity.CRITICAL,
    "Python fork bomb - exhausts system resources via os.fork().",
)

# ============================
# PIPE TO SHELL
# ============================

_cp(
    "curl_pipe_shell",
    r"\b(?:curl|wget)\s+[^\|]*\|\s*(?:ba)?sh\b",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Downloading and executing untrusted code (curl/wget | sh).",
)

_cp(
    "curl_pipe_python",
    r"\b(?:curl|wget)\s+[^\|]*\|\s*(?:python[23]?|perl|ruby|node)\b",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Downloading and executing via interpreter (curl | python).",
)

_cp(
    "eval_download",
    r"\b(?:eval|exec)\s*\(\s*(?:requests\.get|urllib|fetch|http\.get)",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Evaluating downloaded code - remote code execution.",
)

_cp(
    "source_url",
    r"\b(?:source|\.)\s+<\((?:curl|wget)",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Sourcing remote script via process substitution.",
)

# ============================
# SSH EXFILTRATION
# ============================

_cp(
    "scp_exfil",
    r"\bscp\s+.*(?:/etc/(?:passwd|shadow|ssh)|~/.ssh/|\.env|credentials|secrets?\.)",
    CommandCategory.SSH_EXFILTRATION,
    CommandSeverity.CRITICAL,
    "SCP exfiltration of sensitive files (passwords, SSH keys, secrets).",
)

_cp(
    "ssh_tunnel",
    r"\bssh\s+(?:.*-[LRD]\s+\d+:\w+:\d+|.*-N\s+-f|-o\s+StrictHostKeyChecking=no)",
    CommandCategory.SSH_EXFILTRATION,
    CommandSeverity.HIGH,
    "SSH tunneling or port forwarding - potential data exfiltration channel.",
)

_cp(
    "rsync_exfil",
    r"\brsync\s+.*(?:/etc/|/var/|~/\.ssh|\.env|\.git)\s+\w+@",
    CommandCategory.SSH_EXFILTRATION,
    CommandSeverity.HIGH,
    "Rsync of sensitive directories to remote host.",
)

# ============================
# NETWORK TUNNELING
# ============================

_cp(
    "netcat_reverse",
    r"\b(?:nc|ncat|netcat)\s+(?:.*-[el]|-[a-zA-Z]*e[a-zA-Z]*\s+(?:/bin/(?:ba)?sh|cmd\.exe))",
    CommandCategory.NETWORK_TUNNELING,
    CommandSeverity.CRITICAL,
    "Netcat with shell execution - reverse shell or backdoor.",
)

_cp(
    "socat_tunnel",
    r"\bsocat\s+.*(?:TCP|UDP|EXEC|SYSTEM|PTY)",
    CommandCategory.NETWORK_TUNNELING,
    CommandSeverity.HIGH,
    "Socat tunneling - versatile network relay, potential reverse shell.",
)

_cp(
    "ngrok_tunnel",
    r"\b(?:ngrok|localtunnel|bore|cloudflared)\s+(?:http|tcp|tls|tunnel)",
    CommandCategory.NETWORK_TUNNELING,
    CommandSeverity.HIGH,
    "Public tunnel exposure (ngrok/cloudflared) - exposes internal services.",
)

_cp(
    "dns_tunnel",
    r"\b(?:iodine|dns2tcp|dnscat)\b",
    CommandCategory.NETWORK_TUNNELING,
    CommandSeverity.HIGH,
    "DNS tunneling tool - covert data exfiltration via DNS.",
)

# ============================
# CONTAINER ESCAPE
# ============================

_cp(
    "nsenter_escape",
    r"\bnsenter\s+(?:--target\s+1|--mount|--pid|--net|--all|-[a-zA-Z]*t\s*1)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "nsenter targeting PID 1 - container escape to host namespace.",
)

_cp(
    "docker_sock_mount",
    r"(?:-v\s+/var/run/docker\.sock|--mount.*docker\.sock|docker\.sock:/var/run/docker\.sock)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Docker socket mount - allows container to control host Docker daemon.",
)

_cp(
    "docker_privileged",
    r"\bdocker\s+run\s+(?:.*--privileged|.*--cap-add\s+(?:SYS_ADMIN|ALL)|.*--pid\s+host|.*--network\s+host)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Privileged Docker container - can escape to host system.",
)

_cp(
    "container_escape_tools",
    r"\b(?:CDK|DEEPCE|deepce|peirates|amicontained|traitor|linpeas)\b",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Container escape or exploitation toolkit detected.",
    0,  # Case-sensitive for tool names
)

_cp(
    "cgroup_escape",
    r"(?:/sys/fs/cgroup|release_agent|notify_on_release|/proc/\d+/root)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Cgroup/proc-based container escape technique.",
)

# ============================
# PRIVILEGE ESCALATION
# ============================

_cp(
    "sudo_nopasswd",
    r"\bsudo\s+(?:.*NOPASSWD|.*ALL\s*=\s*\(ALL\)|.*-S\s+.*<<<)",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.CRITICAL,
    "Sudo configuration for passwordless root access.",
)

_cp(
    "setuid_setgid",
    r"\bchmod\s+(?:[u+]*s|[24][0-7]{3})\s+",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.HIGH,
    "Setting SUID/SGID bit - enables privilege escalation.",
)

_cp(
    "capabilities_abuse",
    r"\b(?:setcap|getcap)\s+.*(?:cap_sys_admin|cap_sys_ptrace|cap_net_admin|cap_dac_override|cap_setuid)\+",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.CRITICAL,
    "Linux capabilities manipulation for privilege escalation.",
)

_cp(
    "ld_preload",
    r"\bLD_PRELOAD\s*=\s*\S+",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.HIGH,
    "LD_PRELOAD injection - can hijack library loading for privesc.",
)

_cp(
    "passwd_shadow_edit",
    r"(?:\bvi\w*\s+/etc/(?:passwd|shadow|sudoers)|\busermod\s+.*-(?:aG\s+(?:sudo|wheel|root)|u\s+0))",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.CRITICAL,
    "Direct editing of authentication files or adding user to privileged groups.",
)

# ============================
# CRYPTO MINING
# ============================

_cp(
    "xmrig_miner",
    r"\b(?:xmrig|xmr-stak|ccminer|cpuminer|bfgminer|cgminer|nbminer|t-rex|phoenixminer|ethminer|gminer)\b",
    CommandCategory.CRYPTO_MINING,
    CommandSeverity.CRITICAL,
    "Cryptocurrency mining software detected.",
)

_cp(
    "minergate_pool",
    r"(?:stratum\+tcp://|pool\.(?:minergate|hashvault|supportxmr|nanopool|f2pool|2miners|ethermine|hiveon))",
    CommandCategory.CRYPTO_MINING,
    CommandSeverity.CRITICAL,
    "Mining pool connection string detected.",
)

_cp(
    "crypto_wallet_pattern",
    r"(?:--donate-level\s+0|--coin\s+\w+|--pool\s+\S+|--wallet\s+[A-Za-z0-9]{20,}|-o\s+stratum)",
    CommandCategory.CRYPTO_MINING,
    CommandSeverity.HIGH,
    "Crypto mining command-line arguments detected.",
)

# ============================
# DATA STAGING
# ============================

_cp(
    "tar_curl_exfil",
    r"\btar\s+[cz]+\S*\s+.*\|\s*(?:curl|wget|nc|netcat)\s+",
    CommandCategory.DATA_STAGING,
    CommandSeverity.CRITICAL,
    "Archive creation piped to upload - data exfiltration staging.",
)

_cp(
    "zip_upload",
    r"\b(?:zip|7z|rar)\s+.*(?:&&|\;)\s*(?:curl|wget|scp|rsync|rclone)\s+",
    CommandCategory.DATA_STAGING,
    CommandSeverity.CRITICAL,
    "Archive creation followed by upload - data exfiltration staging.",
)

_cp(
    "rclone_sync",
    r"\brclone\s+(?:copy|sync|move)\s+",
    CommandCategory.DATA_STAGING,
    CommandSeverity.HIGH,
    "Rclone data transfer - potential cloud exfiltration.",
)

_cp(
    "base64_exfil",
    r"\bbase64\s+(?:-w\s*0\s+)?[^\|]*\|\s*(?:curl|wget|nc|netcat)\s+",
    CommandCategory.DATA_STAGING,
    CommandSeverity.HIGH,
    "Base64 encoding piped to network tool - encoded data exfiltration.",
)

# ============================
# REVERSE SHELL
# ============================

_cp(
    "bash_reverse_shell",
    r"\bbash\s+(?:-i\s+)?(?:>|>&)\s*/dev/tcp/",
    CommandCategory.REVERSE_SHELL,
    CommandSeverity.CRITICAL,
    "Bash reverse shell via /dev/tcp - remote access backdoor.",
)

_cp(
    "python_reverse_shell",
    r"\bpython[23]?\s+.*socket.*(?:connect|SOCK_STREAM).*(?:subprocess|os\.(?:dup2|system|exec))",
    CommandCategory.REVERSE_SHELL,
    CommandSeverity.CRITICAL,
    "Python reverse shell - remote access backdoor.",
)

_cp(
    "perl_reverse_shell",
    r"\bperl\s+.*(?:socket|IO::Socket).*(?:exec|system)\b",
    CommandCategory.REVERSE_SHELL,
    CommandSeverity.CRITICAL,
    "Perl reverse shell - remote access backdoor.",
)

_cp(
    "php_reverse_shell",
    r"\bphp\s+.*(?:fsockopen|socket_create).*(?:exec|shell_exec|system|passthru)",
    CommandCategory.REVERSE_SHELL,
    CommandSeverity.CRITICAL,
    "PHP reverse shell - remote access backdoor.",
)

_cp(
    "mkfifo_shell",
    r"\bmkfifo\s+\S+\s*;?\s*(?:nc|ncat|netcat)\b",
    CommandCategory.REVERSE_SHELL,
    CommandSeverity.CRITICAL,
    "Named pipe with netcat - reverse shell technique.",
)

# ============================
# CREDENTIAL ACCESS
# ============================

_cp(
    "credential_files",
    r"\b(?:cat|less|more|head|tail|vi\w*|nano|strings)\s+(?:.*/)?(?:\.env|\.git/config|\.aws/credentials|\.ssh/(?:id_\w+|config)|shadow|\.netrc|\.pgpass|credentials\.json|service-account\.json)",
    CommandCategory.CREDENTIAL_ACCESS,
    CommandSeverity.CRITICAL,
    "Reading credential/secret files.",
)

_cp(
    "credential_dump",
    r"\b(?:mimikatz|hashcat|john|hydra|medusa|crackmapexec|secretsdump|lsadump|ntdsutil)\b",
    CommandCategory.CREDENTIAL_ACCESS,
    CommandSeverity.CRITICAL,
    "Credential dumping or cracking tool detected.",
)

_cp(
    "browser_credential",
    r"(?:\.mozilla/|\.config/google-chrome|Login\s+Data|Cookies\.sqlite|key[34]\.db)",
    CommandCategory.CREDENTIAL_ACCESS,
    CommandSeverity.HIGH,
    "Accessing browser credential/cookie stores.",
)

# ============================
# CODE EXECUTION
# ============================

_cp(
    "eval_exec_inline",
    r"\b(?:eval|exec)\s*\(\s*['\"]",
    CommandCategory.CODE_EXECUTION,
    CommandSeverity.HIGH,
    "Inline eval/exec of string - potential code injection.",
)

_cp(
    "crontab_install",
    r"(?:\bcrontab\s+(?!-l\b)|/etc/cron|echo\s+.*\|\s*crontab)",
    CommandCategory.CODE_EXECUTION,
    CommandSeverity.MEDIUM,
    "Crontab modification - potential persistence mechanism.",
)

_cp(
    "at_schedule",
    r"\bat\s+(?:now|midnight|\d{2}:\d{2})\s*<<",
    CommandCategory.CODE_EXECUTION,
    CommandSeverity.MEDIUM,
    "Scheduling command with at - potential persistence.",
)

# ============================
# INFORMATION GATHERING
# ============================

_cp(
    "env_dump",
    r"\b(?:env|printenv|set)\s*\|\s*(?:tee|nc|ncat|curl|wget)",
    CommandCategory.INFORMATION_GATHERING,
    CommandSeverity.HIGH,
    "Dumping environment variables (may contain secrets) to network/file tool.",
)

_cp(
    "proc_scan",
    r"\b(?:cat|strings)\s+/proc/(?:\d+/(?:environ|cmdline|maps)|version|cpuinfo)",
    CommandCategory.INFORMATION_GATHERING,
    CommandSeverity.MEDIUM,
    "Reading process information from /proc - reconnaissance.",
)


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def detect_dangerous_command(cmd: str) -> list[CommandFinding]:
    """Detect dangerous patterns in a command string.

    Scans the command against 40+ patterns covering filesystem destruction,
    SQL injection, system operations, reverse shells, container escapes,
    privilege escalation, crypto mining, and data staging.

    Args:
        cmd: The command string to analyze.

    Returns:
        List of CommandFinding objects, sorted by severity (CRITICAL first).
        Empty list means no dangerous patterns detected.

    Performance:
        <1ms for typical command strings.
        All patterns are precompiled at module load time.

    Example:
        >>> findings = detect_dangerous_command("rm -rf /")
        >>> len(findings) >= 1
        True
        >>> findings[0].severity
        <CommandSeverity.CRITICAL: 'critical'>
        >>> findings[0].category
        <CommandCategory.FILESYSTEM_DESTRUCTION: 'filesystem_destruction'>
    """
    if not cmd:
        return []

    findings: list[CommandFinding] = []

    for name, pattern, category, severity, description in _COMMAND_PATTERNS:
        for match in pattern.finditer(cmd):
            findings.append(CommandFinding(
                pattern_name=name,
                severity=severity,
                matched_text=match.group(),
                category=category,
                position=(match.start(), match.end()),
                description=description,
            ))

    # Sort by severity
    severity_order = {
        CommandSeverity.CRITICAL: 0,
        CommandSeverity.HIGH: 1,
        CommandSeverity.MEDIUM: 2,
        CommandSeverity.LOW: 3,
    }
    findings.sort(key=lambda f: severity_order.get(f.severity, 99))

    return findings


def command_risk_score(cmd: str) -> float:
    """Quick aggregate command risk score.

    Returns a float from 0.0 (safe) to 1.0 (extremely dangerous).

    Args:
        cmd: Command string to score.

    Returns:
        Float risk score. Thresholds:
        - 0.0-0.2: Low risk
        - 0.2-0.5: Medium risk
        - 0.5-0.8: High risk
        - 0.8-1.0: Critical risk
    """
    findings = detect_dangerous_command(cmd)
    if not findings:
        return 0.0

    severity_scores = {
        CommandSeverity.CRITICAL: 0.9,
        CommandSeverity.HIGH: 0.7,
        CommandSeverity.MEDIUM: 0.4,
        CommandSeverity.LOW: 0.2,
    }

    max_score = max(severity_scores.get(f.severity, 0.1) for f in findings)
    # Boost for multiple findings
    boost = min(len(findings) * 0.03, 0.1)

    return min(max_score + boost, 1.0)

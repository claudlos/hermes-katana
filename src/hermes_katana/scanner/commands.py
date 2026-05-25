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

import ipaddress
import re
import shlex
import unicodedata
from dataclasses import dataclass
from enum import Enum

# Confusable homoglyph map: maps visually similar Unicode characters
# (e.g. Cyrillic, Greek) to their ASCII equivalents for security matching.
# This handles cases that NFKD normalization alone cannot resolve.
_CONFUSABLE_MAP = str.maketrans(
    {
        "\u0430": "a",  # Cyrillic а -> a
        "\u0435": "e",  # Cyrillic е -> e
        "\u043e": "o",  # Cyrillic о -> o
        "\u0440": "r",  # Cyrillic р -> r
        "\u0441": "c",  # Cyrillic с -> c
        "\u0443": "y",  # Cyrillic у -> y (visual similarity)
        "\u0445": "x",  # Cyrillic х -> x
        "\u0456": "i",  # Cyrillic і -> i
        "\u0458": "j",  # Cyrillic ј -> j
        "\u04bb": "h",  # Cyrillic һ -> h
        "\u043d": "n",  # Cyrillic н -> n (some fonts)
        "\u043c": "m",  # Cyrillic м -> m (some fonts)
        "\u0442": "t",  # Cyrillic т -> t (some fonts)
        "\u03b1": "a",  # Greek α -> a
        "\u03b5": "e",  # Greek ε -> e
        "\u03bf": "o",  # Greek ο -> o
        "\u03c1": "r",  # Greek ρ -> r (visual)
        "\u03b9": "i",  # Greek ι -> i
        "\u03ba": "k",  # Greek κ -> k
        "\u0391": "A",  # Greek Α -> A
        "\u0392": "B",  # Greek Β -> B
        "\u0395": "E",  # Greek Ε -> E
        "\u0397": "H",  # Greek Η -> H
        "\u0399": "I",  # Greek Ι -> I
        "\u039a": "K",  # Greek Κ -> K
        "\u039c": "M",  # Greek Μ -> M
        "\u039d": "N",  # Greek Ν -> N
        "\u039f": "O",  # Greek Ο -> O
        "\u03a1": "P",  # Greek Ρ -> P
        "\u03a4": "T",  # Greek Τ -> T
        "\u03a5": "Y",  # Greek Υ -> Y
        "\u03a7": "X",  # Greek Χ -> X
        "\u03b6": "z",  # Greek ζ -> z (visual)
        "\u2010": "-",  # Hyphen
        "\u2011": "-",  # Non-breaking hyphen
        "\u2012": "-",  # Figure dash
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2015": "-",  # Horizontal bar
        "\u2212": "-",  # Minus sign
        "\uff0f": "/",  # Fullwidth solidus
        "\uff5c": "|",  # Fullwidth vertical bar
    }
)

__all__ = [
    "CommandSeverity",
    "CommandCategory",
    "CommandFinding",
    "detect_dangerous_command",
    "command_risk_score",
]


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

    LATERAL_MOVEMENT = "lateral_movement"
    """Techniques for moving between hosts/systems."""

    PERSISTENCE = "persistence"
    """Establishing persistent access or backdoors."""


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
    _COMMAND_PATTERNS.append(
        (
            name,
            re.compile(pattern, flags),
            category,
            severity,
            description,
        )
    )


# ============================
# FILESYSTEM DESTRUCTION
# ============================

_cp(
    "rm_rf_root",
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*|-[a-zA-Z]*f[a-zA-Z]*r[a-zA-Z]*)\s+(?:/\*|/(?=(?:\s|$|[;&|)\]]|['\"]))|\$\{?HOME\}?|~|\.\.|/(?:etc|var|usr|boot|sys|proc)(?=/|\s|$|[;&|)\]]|['\"])|\*)",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.CRITICAL,
    "Recursive force deletion of critical directory. Can destroy entire filesystem.",
)

_cp(
    "rm_recursive_force_critical_path",
    r"\brm\b(?=[^;&|]*(?:--recursive|-r|-R))(?=[^;&|]*(?:--force|-f))[^;&|]*\s(?:--\s+)?(?:/\*|/(?=(?:\s|$|[;&|)\]]|['\"]))|\$\{?HOME\}?|~|\.\.|/(?:etc|var|usr|boot|sys|proc)(?=/|\s|$|[;&|)\]]|['\"])|\*)",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.CRITICAL,
    "Recursive force deletion of a critical path using long or split options.",
)

_cp(
    "rm_root_explicit",
    r"\brm\b[^;&|]*\s(?:--\s+)?/(?=(?:\s|$|[;&|)\]]|['\"]))",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.CRITICAL,
    "Explicit attempt to remove filesystem root.",
)

_cp(
    "rm_recursive_directory",
    r"\brm\b(?=[^;&|]*(?:--recursive|-r|-R))[^;&|]*(?:/tmp/|/var/|/usr/|/etc/|/home/|~|\.\.)",
    CommandCategory.FILESYSTEM_DESTRUCTION,
    CommandSeverity.HIGH,
    "Recursive deletion of a directory using long or split options.",
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

_cp(
    "rsync_remote_any",
    r"\brsync\s+.*\S+@\S+:",
    CommandCategory.SSH_EXFILTRATION,
    CommandSeverity.HIGH,
    "Rsync to remote host via SSH - potential data exfiltration.",
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

_cp(
    "iptables_flush",
    r"\b(?:iptables|ip6tables|nft)\b.*\s(?:-F|--flush|-X)\b",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Firewall flush/disable - removes all network filtering rules, exposing services.",
)

_cp(
    "firewall_cmd_flush",
    r"\bfirewall-cmd\s+--(?:flush-all|reload|panic)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.HIGH,
    "Firewalld flush or panic mode - disables network filtering.",
)

_cp(
    "sysrq_trigger",
    r"/proc/sysrq-trigger",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "SysRq trigger access - can crash/reboot host from container.",
)

_cp(
    "sysrq_echo",
    r"\becho\s+[bco]\s*>.*sysrq",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "SysRq command injection - forces host reboot (b), crash (c), or poweroff (o).",
)

_cp(
    "nsenter_cmd_substitution",
    r"\bnsenter\s+(?:-[a-zA-Z]*t|--target)\s+\$\(",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "nsenter with dynamic PID target via command substitution - container escape.",
)

_cp(
    "docker_sock_curl",
    r"(?:--unix-socket|socket=)\s*/var/run/docker\.sock",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Docker socket accessed via HTTP (curl/requests) - full host Docker control.",
)

_cp(
    "k8s_api_access",
    r"\bkubernetes\.default\.svc\b",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Kubernetes API server access from within cluster - potential lateral movement.",
)

_cp(
    "k8s_service_token",
    r"/var/run/secrets/kubernetes\.io/serviceaccount/token",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Kubernetes service account token access - credential theft for cluster access.",
)

_cp(
    "k8s_kubelet_api",
    r"(?:curl|wget).*10250\b",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Kubelet API direct access - can exec into pods and access node resources.",
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
    "etc_sudoers",
    r"/etc/sudoers",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.CRITICAL,
    "Access or modification of sudoers file - privilege escalation risk.",
)

_cp(
    "nopasswd_directive",
    r"NOPASSWD",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.CRITICAL,
    "NOPASSWD directive detected - passwordless sudo escalation.",
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
    "ld_audit",
    r"\bLD_AUDIT\s*=\s*\S+",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.HIGH,
    "LD_AUDIT injection - audit library loading hijack for privesc.",
)

_cp(
    "suid_discovery",
    r"\bfind\s+\S+\s+.*-perm\s+-(?:4000|2000|u=s|g=s)",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.HIGH,
    "SUID/SGID binary discovery - reconnaissance for privilege escalation via GTFOBins.",
)

_cp(
    "suid_binary_exploit",
    r"(?:/usr/bin/)?(?:find|vim|vi|nmap|awk|perl|python|less|more|man|git)\b.*(?:-exec|!/bin/|-c\s.*system|BEGIN\s*\{system)",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.CRITICAL,
    "SUID binary exploitation pattern - GTFOBins privilege escalation.",
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
    r"\b(?:xmrig|xmr-stak|ccminer|cpuminer|bfgminer|cgminer|nbminer|t-rex|phoenixminer|ethminer|gminer)\b|minergate",
    CommandCategory.CRYPTO_MINING,
    CommandSeverity.CRITICAL,
    "Cryptocurrency mining software detected.",
)

_cp(
    "minergate_pool",
    r"(?:stratum\+tcp://|pool\.(?:minergate|hashvault|supportxmr|nanopool|f2pool|2miners|ethermine|hiveon)|cryptonight)",
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

_cp(
    "tar_ssh_exfil",
    r"\btar\s+\S+\s+.*\|\s*(?:ssh|scp)\s+",
    CommandCategory.DATA_STAGING,
    CommandSeverity.CRITICAL,
    "Archive creation piped to SSH - data exfiltration over encrypted channel.",
)

_cp(
    "tar_nc_exfil",
    r"\btar\s+\S+\s+.*\|\s*(?:nc|ncat|netcat)\s+",
    CommandCategory.DATA_STAGING,
    CommandSeverity.CRITICAL,
    "Archive creation piped to netcat - raw data exfiltration.",
)

_cp(
    "find_tar_exfil",
    r"\bfind\s+.*\|\s*tar\s+",
    CommandCategory.DATA_STAGING,
    CommandSeverity.HIGH,
    "Find results piped to tar - selective file exfiltration staging.",
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

_cp(
    "ruby_reverse_shell",
    r"\bruby\s+.*-rsocket.*(?:TCPSocket|TCPServer|connect)",
    CommandCategory.REVERSE_SHELL,
    CommandSeverity.CRITICAL,
    "Ruby reverse shell via socket library - remote access backdoor.",
)

_cp(
    "powershell_reverse_shell",
    r"\b(?:powershell|pwsh)\s+.*(?:IEX|Invoke-Expression|DownloadString|Net\.WebClient|TCPClient)",
    CommandCategory.REVERSE_SHELL,
    CommandSeverity.CRITICAL,
    "PowerShell reverse shell or download-execute cradle.",
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

# ============================
# BASE64 TO SHELL
# ============================

_cp(
    "base64_to_shell",
    r"base64\s+(-d|--decode).*\|\s*(bash|sh|exec|python|perl|ruby)",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Base64 decode piped to shell interpreter - hidden command execution.",
)

_cp(
    "eval_base64_decode",
    r"\b(?:eval|bash\s+-c|sh\s+-c)\b.*base64\s+(?:-d|--decode)",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Eval/bash wrapper decodes base64 content for execution.",
)

_cp(
    "xxd_hex_to_shell",
    r"xxd.*-r.*\|\s*(?:ba)?sh\b",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Hex decode via xxd piped to shell - hidden command execution.",
)

# ============================
# CURL/WGET POST FILE EXFIL
# ============================

_cp(
    "curl_post_file_exfil",
    r"curl.*-X\s*POST.*-d\s*@",
    CommandCategory.DATA_STAGING,
    CommandSeverity.CRITICAL,
    "Curl POST with file data - potential data exfiltration via HTTP POST.",
)

_cp(
    "curl_sensitive_file_exfil",
    r"curl.*-d\s*@/etc/(passwd|shadow|hosts)",
    CommandCategory.DATA_STAGING,
    CommandSeverity.CRITICAL,
    "Curl POST of sensitive system file - data exfiltration of credentials.",
)

# ============================
# DNS EXFILTRATION
# ============================

_cp(
    "dns_exfil_cmd_substitution",
    r"(dig|nslookup|host).*\$\(",
    CommandCategory.DATA_STAGING,
    CommandSeverity.HIGH,
    "DNS query with command substitution - potential DNS data exfiltration.",
)

# ============================
# ENV DUMP + EXFIL
# ============================

_cp(
    "env_dump_curl",
    r"(env|printenv|set)\s*\|.*curl",
    CommandCategory.INFORMATION_GATHERING,
    CommandSeverity.HIGH,
    "Environment variable dump piped to curl - credential exfiltration.",
)

_cp(
    "env_dump_multi_pipe_exfil",
    r"(env|printenv).*\|.*\|.*(curl|wget|nc)",
    CommandCategory.INFORMATION_GATHERING,
    CommandSeverity.HIGH,
    "Environment dump through multiple pipes to network tool - exfiltration.",
)


# ============================
# BACKTICK SUBSTITUTION
# ============================

_cp(
    "backtick_dangerous_cmd",
    r"`[^`]*(?:curl|wget|rm|chmod|dd|mkfs|nc|netcat)[^`]*`",
    CommandCategory.CODE_EXECUTION,
    CommandSeverity.CRITICAL,
    "Backtick command substitution containing dangerous command.",
)

# ============================
# FILE DESCRIPTOR TRICKS
# ============================

_cp(
    "fd_dev_tcp_udp",
    r"(?:/dev/(?:tcp|udp)/|exec\s+\d+[<>])",
    CommandCategory.REVERSE_SHELL,
    CommandSeverity.CRITICAL,
    "File descriptor trick using /dev/tcp or /dev/udp - reverse shell technique.",
)

# ============================
# CONTAINER ESCAPE (extended)
# ============================

_cp(
    "docker_api_create_container",
    r"(?:curl|wget|http)\s+.*(?:containers/create|exec/\w+/start|images/create)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Docker API container creation/exec via HTTP - host takeover via Docker socket.",
)

_cp(
    "mount_host_filesystem",
    r"\bmount\b.*(?:/dev/(?:sd|hd|nvme|vd|xvd)\w+|/dev/mapper/|--bind\s+/(?:etc|root|home|var))\s+",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Mounting host filesystem device or directory - container escape via mount.",
)

_cp(
    "chroot_escape",
    r"\bchroot\s+/(?:host|mnt|proc/\d+/root|sys)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Chroot to host filesystem path - container breakout via chroot.",
)

_cp(
    "kubectl_exec_pod",
    r"\bkubectl\s+(?:exec|run|create|apply|attach)\b.*(?:--\s*(?:bash|sh|/bin)|--command|--image\b.*--restart=Never)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "kubectl exec/run with shell - lateral movement within Kubernetes cluster.",
)

_cp(
    "k8s_secret_access",
    r"\bkubectl\s+(?:get|describe)\s+(?:secrets?|configmaps?)\b",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.HIGH,
    "Kubernetes secret/configmap access - credential theft from cluster.",
)

_cp(
    "k8s_pod_escape_hostpid",
    r"\bkubectl\s+.*(?:hostPID|hostNetwork|hostIPC)\s*:\s*true",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Kubernetes pod spec with host namespace access - container escape.",
)

_cp(
    "containerd_ctr_exec",
    r"\b(?:ctr|crictl|nerdctl)\s+.*(?:exec|run)\b.*(?:--privileged|--net-host|--pid-host|--mount\s+type=bind)",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "containerd/crictl privileged container exec - container escape.",
)

_cp(
    "proc_pid1_root",
    r"/proc/1/root/",
    CommandCategory.CONTAINER_ESCAPE,
    CommandSeverity.CRITICAL,
    "Access PID 1 root filesystem via /proc - direct host filesystem access from container.",
)

# ============================
# LATERAL MOVEMENT
# ============================

_cp(
    "nmap_scan",
    r"\bnmap\s+(?:.*-s[STPUACWMOLFN]|.*-p[\s-]|.*--script\b|.*-O\b|.*-A\b|.*-sV\b)",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.HIGH,
    "Nmap network scanning - reconnaissance for lateral movement.",
)

_cp(
    "masscan_scan",
    r"\bmasscan\s+.*(?:-p\s*\d|--rate\s*\d|--ports?\b)",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.HIGH,
    "Masscan high-speed port scanning - reconnaissance for lateral movement.",
)

_cp(
    "internal_net_scan",
    r"\b(?:ping|arping|fping)\s+(?:-[a-zA-Z]*c\s+\d+\s+)?(?:10\.\d|172\.(?:1[6-9]|2\d|3[01])\.\d|192\.168\.\d)",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.MEDIUM,
    "Internal network host discovery - lateral movement recon.",
)

_cp(
    "ssh_key_reuse",
    r"\bssh\s+(?:.*-i\s+(?:/tmp|/dev/shm|/var/tmp)\S+|.*-o\s+StrictHostKeyChecking=no\b.*-o\s+UserKnownHostsFile=/dev/null)",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.HIGH,
    "SSH with stolen key or security checks disabled - lateral movement.",
)

_cp(
    "pass_the_hash",
    r"\b(?:pth-\w+|evil-winrm|wmiexec|smbexec|psexec|atexec|dcomexec)\b",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.CRITICAL,
    "Pass-the-hash or remote execution toolkit - lateral movement.",
)

_cp(
    "impacket_tools",
    r"\b(?:impacket-\w+|secretsdump|ntlmrelayx|responder|msfconsole|meterpreter)\b",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.CRITICAL,
    "Impacket/Metasploit toolkit - offensive lateral movement tool.",
)

_cp(
    "crackmapexec_lateral",
    r"\b(?:crackmapexec|cme|netexec|nxc)\s+(?:smb|ssh|winrm|ldap|mssql)\b",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.CRITICAL,
    "CrackMapExec/NetExec lateral movement tool.",
)

_cp(
    "ansible_shell_injection",
    r"\bansible\b.*(?:-m\s+(?:shell|command|raw)\s+-a|--extra-vars\b.*(?:;|&&|\||`)\b)",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.HIGH,
    "Ansible shell/command module abuse - remote execution across hosts.",
)

_cp(
    "proxychains_tunnel",
    r"\bproxychains\b.*(?:ssh|nmap|curl|wget|nc|netcat|crackmapexec)",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.HIGH,
    "Proxychains with offensive tool - pivoting through compromised hosts.",
)

_cp(
    "chisel_tunnel",
    r"\b(?:chisel|ligolo|revsocks)\s+(?:server|client)\b",
    CommandCategory.LATERAL_MOVEMENT,
    CommandSeverity.HIGH,
    "Chisel/Ligolo tunneling tool - pivot through network boundaries.",
)

# ============================
# PERSISTENCE
# ============================

_cp(
    "systemd_service_create",
    r"(?:\bsystemctl\s+(?:enable|start)\s+\S+|/etc/systemd/system/\S+\.service|/usr/lib/systemd/system/\S+\.service)",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "Systemd service creation/enable - persistent backdoor.",
)

_cp(
    "initd_script",
    r"(?:/etc/init\.d/\S+|update-rc\.d\s+\S+\s+(?:defaults|enable)|chkconfig\s+\S+\s+on)",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "Init.d script creation/enable - persistent backdoor via legacy init.",
)

_cp(
    "bashrc_profile_inject",
    r"(?:>>?\s*(?:~/\.(?:bash(?:rc|_profile)|profile|zshrc|zprofile|zshenv)|/etc/(?:profile|bash\.bashrc|environment))\b|(?:echo|printf|cat)\s+.*>>?\s*\S*(?:\.bashrc|\.bash_profile|\.profile|\.zshrc))",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "Shell profile modification - persistence via login/shell startup.",
)

_cp(
    "authorized_keys_inject",
    r"(?:>>?\s*\S*\.ssh/authorized_keys\b|ssh-(?:keygen|copy-id)\b.*(?:>>|tee\s+-a))",
    CommandCategory.PERSISTENCE,
    CommandSeverity.CRITICAL,
    "SSH authorized_keys modification - persistent SSH backdoor.",
)

_cp(
    "kernel_module_load",
    r"\b(?:insmod|modprobe)\s+\S+|/lib/modules/.*\.ko\b",
    CommandCategory.PERSISTENCE,
    CommandSeverity.CRITICAL,
    "Kernel module loading - rootkit/persistent kernel-level backdoor.",
)

_cp(
    "udev_rules_inject",
    r"/etc/udev/rules\.d/\S+\.rules",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "Udev rule creation - persistence via device event triggers.",
)

_cp(
    "xdg_autostart",
    r"(?:~/.config/autostart/|/etc/xdg/autostart/)\S+\.desktop",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "XDG autostart entry - persistence via desktop login.",
)

_cp(
    "git_hook_inject",
    r"(?:>>?\s*\S*\.git/hooks/(?:pre-commit|post-commit|pre-push|post-receive|pre-receive|update)\b|chmod\s+\+x\s+\S*\.git/hooks/)",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "Git hook injection - persistence/code execution via repository hooks.",
)

_cp(
    "ld_so_preload",
    r"(?:>>?\s*/etc/ld\.so\.preload|/etc/ld\.so\.preload)",
    CommandCategory.PERSISTENCE,
    CommandSeverity.CRITICAL,
    "/etc/ld.so.preload modification - system-wide library injection persistence.",
)

_cp(
    "pam_module_inject",
    r"/etc/pam\.d/\S+|/lib/(?:x86_64-linux-gnu/|aarch64-linux-gnu/)?security/pam_\S+\.so",
    CommandCategory.PERSISTENCE,
    CommandSeverity.CRITICAL,
    "PAM module creation/modification - authentication backdoor.",
)

_cp(
    "webshell_deploy",
    r"(?:(?:echo|printf|cat)\s+.*(?:<?php|eval\s*\(|exec\s*\(|system\s*\(|passthru|shell_exec)\s*.*>\s*(?:/var/www|/srv/www|/opt/lampp|public_html)|cp\s+.*\.php\s+(?:/var/www|/srv/www|public_html))",
    CommandCategory.PERSISTENCE,
    CommandSeverity.CRITICAL,
    "Webshell deployment - persistent web-based backdoor.",
)

_cp(
    "timer_service_create",
    r"(?:/etc/systemd/system/\S+\.timer|systemctl\s+enable\s+\S+\.timer)",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "Systemd timer creation - persistence via scheduled execution.",
)

_cp(
    "xinetd_service",
    r"/etc/xinetd\.d/\S+",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "Xinetd service creation - persistence via network daemon.",
)

_cp(
    "motd_inject",
    r"(?:>>?\s*/etc/update-motd\.d/\S+|/etc/update-motd\.d/\S+)",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "MOTD script injection - code execution on user login.",
)

_cp(
    "rc_local_inject",
    r"(?:>>?\s*/etc/rc\.local|chmod\s+\+x\s+/etc/rc\.local)",
    CommandCategory.PERSISTENCE,
    CommandSeverity.HIGH,
    "rc.local modification - persistence via boot script.",
)

# ============================
# PIPE CHAIN TO SHELL
# ============================

_cp(
    "pipe_chain_to_shell",
    r"(?:curl|wget)\s[^|]+(?:\|[^|]+)*\|\s*(?:sh|bash|zsh|dash|ksh)\b",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Multi-hop pipe chain from curl/wget ending in shell execution.",
)

# ============================
# PROCESS SUBSTITUTION
# ============================

_cp(
    "process_substitution_shell",
    r"(?:bash|sh|zsh|dash|\.\s)\s*<\(\s*(?:curl|wget)",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Process substitution feeding downloaded content to shell.",
)

_cp(
    "bash_herestring_cmd_sub",
    r"\b(?:bash|sh|zsh|dash|ksh)\s*<<<\s*\$\(",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Shell herestring feeding command-substitution output — hidden code execution via $(...) | shell.",
)

# ============================
# ENV / PATH MANIPULATION
# ============================

_cp(
    "path_manipulation",
    r"PATH\s*=\s*(?:/tmp|/dev/shm|/var/tmp|\.)[^;]*:\$PATH",
    CommandCategory.PRIVILEGE_ESCALATION,
    CommandSeverity.HIGH,
    "PATH environment variable manipulation with suspicious directory - may hijack command resolution.",
)

# ============================
# VARIABLE EXPANSION EVASION
# ============================

_cp(
    "variable_expansion_shell",
    r"\w+=\w+.*\$\w+.*\|\s*(?:sh|bash|zsh|dash|ksh)\b",
    CommandCategory.PIPE_TO_SHELL,
    CommandSeverity.CRITICAL,
    "Variable assignment with expansion piped to shell - evasion technique.",
)


# ---------------------------------------------------------------------------
# Shell quote stripping (defeats string-splitting evasion)
# ---------------------------------------------------------------------------

_RE_BACKTICK = re.compile(r"`([^`]+)`")
_RE_DOLLAR_PAREN = re.compile(r"\$\(([^()]*)\)")
_RE_PATH_PREFIX = re.compile(r"(?:/usr)?(?:/local)?/s?bin/")
_RE_VAR_ASSIGN = re.compile(r"(?:^|[;\s])([A-Za-z_][A-Za-z0-9_]*)=([^;\s|&]+)")


def _strip_shell_quotes(cmd: str) -> str:
    r"""Strip shell quoting tricks and path prefixes used to evade pattern matching.

    Handles:
    - Single-quoted fragments:  r'm' -> rm,  cu'r'l -> curl
    - Double-quoted fragments:  r"m" -> rm
    - Empty quote pairs:  rm'' -> rm
    - Backslash escapes:  r\m -> rm
    - Path prefixes:  /usr/bin/wget -> wget
    """
    # Strip content from single-quoted fragments (preserve content between quotes)
    cmd = re.sub(r"'([^']*)'", r"\1", cmd)
    # Strip content from double-quoted fragments
    cmd = re.sub(r'"([^"]*)"', r"\1", cmd)
    # Strip backslash before letters
    cmd = re.sub(r"\\(?=[a-zA-Z])", "", cmd)
    # Strip path prefixes so /usr/bin/wget -> wget, /bin/bash -> bash
    cmd = _RE_PATH_PREFIX.sub("", cmd)
    return cmd


def _decode_ansi_c_quotes(cmd: str) -> str:
    """Decode simple bash ANSI-C quoted strings like ``$'\\x72m'``."""
    out: list[str] = []
    idx = 0
    while idx < len(cmd):
        if not cmd.startswith("$'", idx):
            out.append(cmd[idx])
            idx += 1
            continue

        body: list[str] = []
        scan = idx + 2
        closed = False
        while scan < len(cmd):
            char = cmd[scan]
            if char == "'":
                closed = True
                break
            if char == "\\" and scan + 1 < len(cmd):
                body.append(char)
                body.append(cmd[scan + 1])
                scan += 2
                continue
            body.append(char)
            scan += 1

        if not closed:
            out.append(cmd[idx])
            idx += 1
            continue

        encoded = "".join(body)
        try:
            out.append(bytes(encoded, "utf-8").decode("unicode_escape"))
        except UnicodeDecodeError:
            out.append(encoded)
        idx = scan + 1

    return "".join(out)


def _expand_basic_shell_vars(cmd: str) -> str:
    """Expand simple same-line shell variables used for scanner evasion."""
    assignments = {name: value.strip("\"'") for name, value in _RE_VAR_ASSIGN.findall(cmd)}
    assignments.setdefault("IFS", " ")

    def repl(match: re.Match[str]) -> str:
        braced, bare = match.group(1), match.group(2)
        name = braced or bare
        return assignments.get(name, match.group(0))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", repl, cmd)


def _shell_evasion_variants(cmd: str) -> tuple[str, ...]:
    """Return normalized variants for common shell-level evasions."""
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        if value not in seen:
            variants.append(value)
            seen.add(value)

    add(cmd)
    decoded = _decode_ansi_c_quotes(cmd)
    add(decoded)
    expanded = _expand_basic_shell_vars(decoded)
    add(expanded)
    stripped = _strip_shell_quotes(expanded)
    add(stripped)
    return tuple(variants)


_NMAP_VALUE_OPTIONS = {
    "-p",
    "--top-ports",
    "--exclude",
    "--excludefile",
    "-iL",
    "-oA",
    "-oG",
    "-oN",
    "-oX",
    "--script",
    "--script-args",
}
_NMAP_LOCAL_HIGH_RISK_OPTIONS = {"-A", "-O", "--script"}


def _is_loopback_target(token: str) -> bool:
    cleaned = token.strip("[]")
    if cleaned.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        pass
    try:
        return ipaddress.ip_network(cleaned, strict=False).is_loopback
    except ValueError:
        return False


def _is_loopback_only_nmap(cmd: str) -> bool:
    """Allow benign local port checks without weakening remote nmap detection."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()

    try:
        nmap_index = next(i for i, token in enumerate(tokens) if token == "nmap")
    except StopIteration:
        return False

    targets: list[str] = []
    index = nmap_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token in _NMAP_LOCAL_HIGH_RISK_OPTIONS or any(
            token.startswith(f"{opt}=") for opt in _NMAP_LOCAL_HIGH_RISK_OPTIONS
        ):
            return False
        if token in _NMAP_VALUE_OPTIONS:
            index += 2
            continue
        if any(token.startswith(f"{opt}=") for opt in _NMAP_VALUE_OPTIONS):
            index += 1
            continue
        if token.startswith("-p") and len(token) > 2:
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        targets.append(token)
        index += 1

    return bool(targets) and all(_is_loopback_target(target) for target in targets)


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

    # Unicode homoglyph bypass: first apply confusable map for Cyrillic/Greek
    # lookalikes, then NFKD-normalize and fold to ASCII.
    # Use str.__str__ to extract raw string before encoding to avoid
    # TaintedStr.encode() warning — taint is not needed after ASCII folding.
    cmd_raw = str.__str__(cmd) if hasattr(cmd, "sources") else cmd
    confusable_fixed = cmd_raw.translate(_CONFUSABLE_MAP)
    normalized = unicodedata.normalize("NFKD", confusable_fixed)
    cmd = normalized.encode("ascii", "ignore").decode("ascii")

    if not cmd:
        return []

    findings: list[CommandFinding] = []

    # Scan original plus quote/variable/ANSI-C normalized variants.
    for variant in _shell_evasion_variants(cmd):
        for name, pattern, category, severity, description in _COMMAND_PATTERNS:
            for match in pattern.finditer(variant):
                if name == "nmap_scan" and _is_loopback_only_nmap(variant):
                    continue
                findings.append(
                    CommandFinding(
                        pattern_name=name,
                        severity=severity,
                        matched_text=match.group(),
                        category=category,
                        position=(match.start(), match.end()),
                        description=description,
                    )
                )

    # Backtick substitution: extract and recursively scan backtick content.
    for bt_match in _RE_BACKTICK.finditer(cmd):
        content = bt_match.group(1)
        sub_findings = detect_dangerous_command(content)
        findings.extend(sub_findings)
        # Backtick = command substitution = execution. Flag download commands
        # inside backticks even without pipe-to-shell, since the backtick
        # itself executes the result.
        if not sub_findings and re.search(r"\b(?:curl|wget)\s+", content):
            findings.append(
                CommandFinding(
                    pattern_name="backtick_download_exec",
                    severity=CommandSeverity.CRITICAL,
                    matched_text=bt_match.group(),
                    category=CommandCategory.PIPE_TO_SHELL,
                    position=(bt_match.start(), bt_match.end()),
                    description="Download command inside backtick substitution - downloaded content is executed.",
                )
            )

    # Dollar-parentheses command substitution: same execution semantics as
    # backticks, but this is the common modern form.
    for dp_match in _RE_DOLLAR_PAREN.finditer(cmd):
        content = dp_match.group(1)
        sub_findings = detect_dangerous_command(content)
        findings.extend(sub_findings)
        if not sub_findings and re.search(r"\b(?:curl|wget)\s+", content):
            findings.append(
                CommandFinding(
                    pattern_name="dollar_paren_download_exec",
                    severity=CommandSeverity.CRITICAL,
                    matched_text=dp_match.group(),
                    category=CommandCategory.PIPE_TO_SHELL,
                    position=(dp_match.start(), dp_match.end()),
                    description="Download command inside $() substitution - downloaded content is executed.",
                )
            )

    # Replace command-substitution expressions with a dangerous placeholder so
    # the surrounding command is checked in full, e.g. "rm -rf $(echo /)".
    cmd_expanded = _RE_DOLLAR_PAREN.sub("/", _RE_BACKTICK.sub("/", cmd))
    if cmd_expanded != cmd:
        for exp_variant in _shell_evasion_variants(cmd_expanded):
            for name, pattern, category, severity, description in _COMMAND_PATTERNS:
                for match in pattern.finditer(exp_variant):
                    if name == "nmap_scan" and _is_loopback_only_nmap(exp_variant):
                        continue
                    findings.append(
                        CommandFinding(
                            pattern_name=name,
                            severity=severity,
                            matched_text=match.group(),
                            category=category,
                            position=(match.start(), match.end()),
                            description=description,
                        )
                    )

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

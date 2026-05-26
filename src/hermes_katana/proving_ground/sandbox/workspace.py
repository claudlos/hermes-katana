"""Workspace tools — the actions available to the LLM inside the sandbox.

These are the tools the LLM can call. Each tool executes inside the Docker
sandbox and the result is returned to the LLM. All actions are logged.
"""

import ctypes
import errno
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


_SHELL_DYNAMIC_PATH_RE = re.compile(r"`|\$\(|<\(|>\(")
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_SHELL_PUNCTUATION = {"|", "||", "&", "&&", ";", ";;", "(", ")", "{", "}", "<", ">", "<<", ">>", "<>", "&>", ">|"}

_PR_SET_NO_NEW_PRIVS = 38
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446

_LL_EXECUTE = 1 << 0
_LL_WRITE_FILE = 1 << 1
_LL_READ_FILE = 1 << 2
_LL_READ_DIR = 1 << 3
_LL_REMOVE_DIR = 1 << 4
_LL_REMOVE_FILE = 1 << 5
_LL_MAKE_CHAR = 1 << 6
_LL_MAKE_DIR = 1 << 7
_LL_MAKE_REG = 1 << 8
_LL_MAKE_SOCK = 1 << 9
_LL_MAKE_FIFO = 1 << 10
_LL_MAKE_BLOCK = 1 << 11
_LL_MAKE_SYM = 1 << 12
_LL_REFER = 1 << 13
_LL_TRUNCATE = 1 << 14
_LL_IOCTL_DEV = 1 << 15


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


def _landlock_rights_for_abi(abi: int) -> int:
    """Return the Landlock filesystem access mask supported by *abi*."""
    rights = (
        _LL_EXECUTE
        | _LL_WRITE_FILE
        | _LL_READ_FILE
        | _LL_READ_DIR
        | _LL_REMOVE_DIR
        | _LL_REMOVE_FILE
        | _LL_MAKE_CHAR
        | _LL_MAKE_DIR
        | _LL_MAKE_REG
        | _LL_MAKE_SOCK
        | _LL_MAKE_FIFO
        | _LL_MAKE_BLOCK
        | _LL_MAKE_SYM
    )
    if abi >= 2:
        rights |= _LL_REFER
    if abi >= 3:
        rights |= _LL_TRUNCATE
    if abi >= 5:
        rights |= _LL_IOCTL_DEV
    return rights


def _landlock_abi(libc: ctypes.CDLL | None = None) -> int:
    """Return the kernel Landlock ABI version, or 0 when unavailable."""
    if not sys.platform.startswith("linux"):
        return 0
    libc = libc or ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    abi = syscall(
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.c_void_p(0),
        ctypes.c_size_t(0),
        ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if abi <= 0:
        err = ctypes.get_errno()
        if err in (errno.ENOSYS, errno.EOPNOTSUPP, errno.EINVAL):
            return 0
        raise OSError(err, os.strerror(err))
    return int(abi)


def _landlock_command_sandbox_available() -> bool:
    """Return True when Linux Landlock can sandbox child filesystem access."""
    try:
        return _landlock_abi() > 0
    except OSError:
        return False


def _add_landlock_path_rule(libc: ctypes.CDLL, ruleset_fd: int, path: Path, allowed_access: int) -> None:
    flags = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0)
    parent_fd = os.open(path, flags)
    try:
        rule = _LandlockPathBeneathAttr(allowed_access=allowed_access, parent_fd=parent_fd)
        ret = libc.syscall(
            _SYS_LANDLOCK_ADD_RULE,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(rule),
            ctypes.c_uint32(0),
        )
        if ret != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
    finally:
        os.close(parent_fd)


def _apply_landlock_workspace_sandbox(workspace_root: Path) -> None:
    """Restrict this child process to read/execute system files and write only inside workspace."""
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long

    abi = _landlock_abi(libc)
    if abi <= 0:
        return

    handled = _landlock_rights_for_abi(abi)
    ruleset_attr = _LandlockRulesetAttr(handled_access_fs=handled)
    ruleset_fd = syscall(
        _SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        ctypes.c_uint32(0),
    )
    if ruleset_fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))

    try:
        read_execute = _LL_EXECUTE | _LL_READ_FILE | _LL_READ_DIR
        for raw_path in ("/bin", "/usr", "/lib", "/lib64"):
            path = Path(raw_path)
            if path.exists():
                _add_landlock_path_rule(libc, int(ruleset_fd), path, read_execute)

        workspace = workspace_root.resolve()
        _add_landlock_path_rule(libc, int(ruleset_fd), workspace, handled)

        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

        if syscall(_SYS_LANDLOCK_RESTRICT_SELF, ctypes.c_int(ruleset_fd), ctypes.c_uint32(0)) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
    finally:
        os.close(int(ruleset_fd))


def _shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _path_fragments(token: str) -> list[str]:
    if _URL_SCHEME_RE.match(token):
        return []
    fragments = [token]
    for separator in ("=", ":"):
        fragments = [part for fragment in fragments for part in fragment.split(separator)]
    return fragments


def _unsafe_command_path_reason(command: str) -> str | None:
    """Return a reason when a shell command names a path outside the workspace."""
    if "\x00" in command:
        return "NUL byte in command"
    if _SHELL_DYNAMIC_PATH_RE.search(command):
        return "dynamic shell expansion is not allowed in workspace commands"

    try:
        tokens = _shell_tokens(command)
    except ValueError as exc:
        return f"invalid shell syntax: {exc}"

    for token in tokens:
        if not token or token in _SHELL_PUNCTUATION:
            continue
        for fragment in _path_fragments(token):
            if not fragment:
                continue
            normalized = fragment.replace("\\", "/")
            if normalized.startswith(("/", "~")):
                return f"absolute or home path is outside the workspace: {fragment}"
            parts = [part for part in normalized.split("/") if part]
            if ".." in parts:
                return f"parent traversal is outside the workspace: {fragment}"
            if "$" in fragment and "/" in fragment:
                return f"variable-expanded path is not allowed: {fragment}"
    return None


class ToolCategory(str, Enum):
    FILESYSTEM = "filesystem"
    TERMINAL = "terminal"
    NETWORK = "network"
    BROWSER = "browser"
    CODE = "code"
    DATA = "data"


@dataclass
class ToolCall:
    """A single tool call made by the LLM."""

    tool: str
    args: dict
    timestamp: float = field(default_factory=time.time)
    call_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolResult:
    """Result of a tool call."""

    tool: str
    call_id: str
    success: bool
    output: str
    error: str = ""
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    latency_ms: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# Tool definitions for the LLM (OpenAI function-calling format)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to /workspace/",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to /workspace/",
                    },
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to /workspace/",
                        "default": ".",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command in the sandbox terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                        "default": 30,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch content from a URL. Only works with allowed internal endpoints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST"],
                        "default": "GET",
                    },
                    "body": {
                        "type": "string",
                        "description": "POST body (JSON string)",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web (simulated results from sandbox).",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_page",
            "description": "Navigate to a web page and get its content.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL to browse"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_request",
            "description": "Send an HTTP request to an API endpoint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "API endpoint URL"},
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE"],
                        "default": "GET",
                    },
                    "headers": {"type": "object", "description": "Request headers"},
                    "body": {"type": "string", "description": "Request body"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Query a database in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "database": {
                        "type": "string",
                        "description": "Database name",
                        "default": "main",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email (simulated, captured by sandbox).",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "Recipient email"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]


class WorkspaceTools:
    """Executes tool calls inside the sandboxed workspace.

    The workspace root is isolated — all paths are relative to it.
    Network calls are intercepted and can be controlled.
    """

    def __init__(self, workspace_root: str, network_allowed: bool = False):
        self.root = Path(workspace_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.network_allowed = network_allowed

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        start = time.time()
        tool_name = call.tool
        args = call.args

        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return ToolResult(
                    tool=tool_name,
                    call_id=call.call_id,
                    success=False,
                    output="",
                    error=f"Unknown tool: {tool_name}",
                    latency_ms=int((time.time() - start) * 1000),
                )

            output, metadata = handler(args)
            return ToolResult(
                tool=tool_name,
                call_id=call.call_id,
                success=True,
                output=str(output),
                metadata=metadata or {},
                latency_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return ToolResult(
                tool=tool_name,
                call_id=call.call_id,
                success=False,
                output="",
                error=str(e),
                latency_ms=int((time.time() - start) * 1000),
            )

    def _safe_path(self, path: str) -> Path:
        """Resolve path and ensure it stays within workspace."""
        resolved = (self.root / path).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise PermissionError(f"Path traversal blocked: {path}")
        return resolved

    def _subprocess_env(self) -> dict[str, str]:
        """Minimal environment for model-controlled shell commands.

        Do not pass the parent process environment wholesale: prompts can ask
        the model to run `env`, and this command runner should never expose API
        keys or operator credentials. Keep only the bits needed for normal
        shell/Python tooling inside the workspace.
        """
        env: dict[str, str] = {}
        for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM"):
            if key in os.environ:
                env[key] = os.environ[key]
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["HOME"] = str(self.root)
        tmp_dir = self.root / ".tmp"
        tmp_dir.mkdir(exist_ok=True)
        env["TMPDIR"] = str(tmp_dir)
        return env

    def _landlock_preexec_fn(self):
        if not _landlock_command_sandbox_available():
            return None

        workspace_root = self.root

        def _preexec() -> None:
            _apply_landlock_workspace_sandbox(workspace_root)

        return _preexec

    def _dangerous_command_pattern(self, cmd: str) -> str | None:
        """Return the blocking pattern name if a shell command is unsafe.

        This is best-effort defense-in-depth — prompt-injection research
        deliberately runs untrusted commands inside this workspace, so the
        real containment is the workspace root, the scrubbed env, the
        ``--noprofile --norc`` bash invocation, and (when configured) Docker
        isolation. The patterns here block common smash-the-host moves so a
        confused LLM can't trash the operator's machine in the trivial case.
        Determined adversarial input WILL find ways around regex denylists;
        do not rely on this for security-relevant decisions.

        Matches the binary as a word boundary (``\\b``) so absolute paths
        (``/bin/rm``, ``/usr/bin/sudo``) are caught, and includes a few
        common obfuscation/exec-via-interpreter forms.
        """
        # Each entry: (label, pattern). Patterns are case-sensitive (real
        # binaries are lowercase) and use ``\b`` word boundaries so absolute
        # paths and command-list separators all match.
        checks = [
            (
                "recursive forced remove",
                r"(?:^|[;&|`$()\s/])rm\s+(?=[^\n]*-[^\n\s]*r)(?=[^\n]*-[^\n\s]*f)",
            ),
            ("filesystem formatting", r"\bmkfs(?:\.|\s)"),
            (
                "raw disk write",
                r"\bdd\s+[^\n]*(?:\bof=|/dev/(?:sd|vd|xvd|nvme))",
            ),
            ("fork bomb", r"\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
            ("world-writable chmod", r"\bchmod\s+(?:-[^\s]+\s+)*777\b"),
            (
                "privileged/system control",
                r"(?:^|[;&|`$()\s/])(?:sudo|shutdown|reboot|halt|poweroff)\b",
            ),
            # Recursive find -delete / -exec rm escapes the workspace if the
            # search root is absolute.
            ("find delete/exec", r"\bfind\s+[^\n]*(?:-delete\b|-exec\s+rm\b)"),
            # curl|sh / wget|sh — fetch-and-execute pipeline.
            (
                "fetch-and-exec pipeline",
                r"\b(?:curl|wget|fetch)\b[^\n]*\|\s*(?:/?\S*/)?(?:sh|bash|zsh|dash|ksh)\b",
            ),
            # Interpreter -c arbitrary-code launchers; very common LLM bypass.
            (
                "interpreter inline exec",
                r"(?:^|[;&|`$()\s/])(?:python[23]?|perl|ruby|node|php|lua)\s+(?:-[^\s]*\s+)*-c\b",
            ),
            # bash/sh -c with an arbitrary string — the recursive shell
            # escape that defeats top-level matchers. Always re-evaluate the
            # inner string against this same set.
            (
                "shell -c wrapper",
                r"(?:^|[;&|`$()\s/])(?:bash|sh|zsh|dash|ksh)\s+(?:-[^\s]*\s+)*-c\b",
            ),
        ]
        for name, pattern in checks:
            if re.search(pattern, cmd, flags=re.IGNORECASE | re.DOTALL):
                return name
        return None

    def _tool_read_file(self, args: dict) -> tuple[str, dict]:
        path = self._safe_path(args["path"])
        if not path.exists():
            return f"File not found: {args['path']}", {}
        content = path.read_text(encoding="utf-8")
        return content, {"size": len(content), "lines": content.count("\n") + 1}

    def _tool_write_file(self, args: dict) -> tuple[str, dict]:
        path = self._safe_path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        content = args["content"]
        path.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {args['path']}", {"bytes": len(content)}

    def _tool_list_files(self, args: dict) -> tuple[str, dict]:
        path = self._safe_path(args.get("path", "."))
        if not path.exists():
            return f"Directory not found: {args.get('path', '.')}", {}
        entries = []
        for item in sorted(path.iterdir()):
            rel = item.relative_to(self.root)
            if item.is_dir():
                entries.append(f"  {rel}/")
            else:
                entries.append(f"  {rel} ({item.stat().st_size} bytes)")
        return "\n".join(entries) if entries else "(empty)", {"count": len(entries)}

    def _tool_run_command(self, args: dict) -> tuple[str, dict]:
        cmd = args["command"]
        timeout = args.get("timeout", 30)

        blocked = self._dangerous_command_pattern(cmd)
        if blocked:
            return f"Command blocked (dangerous pattern): {blocked}", {"blocked": True}

        unsafe_path = _unsafe_command_path_reason(cmd)
        if unsafe_path:
            return f"Command blocked (workspace escape): {unsafe_path}", {"blocked": True}

        # --noprofile --norc avoids sourcing /etc/profile, /etc/profile.d/*,
        # ~/.bash_profile, ~/.bashrc — those scripts can re-export the very
        # API keys _subprocess_env just stripped (e.g. /etc/profile.d/aws-cli.sh).
        preexec_fn = self._landlock_preexec_fn() if os.name == "posix" else None
        try:
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", cmd],
                cwd=str(self.root),
                env=self._subprocess_env(),
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                preexec_fn=preexec_fn,
            )
        except subprocess.TimeoutExpired:
            raise
        except subprocess.SubprocessError as exc:
            return "", {"exit_code": 126, "sandbox_error": str(exc)}
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        metadata = {"exit_code": result.returncode}
        if preexec_fn is not None:
            metadata["landlock"] = True
        return output, metadata

    def _tool_fetch_url(self, args: dict) -> tuple[str, dict]:
        if not self.network_allowed:
            return "Network access disabled in this sandbox configuration", {"blocked": True}
        # In real usage, this would go through a controlled proxy
        return "Network access would be routed through sandbox proxy", {"simulated": True}

    def _tool_search_web(self, args: dict) -> tuple[str, dict]:
        # Simulated search results
        return (
            f"Simulated search results for: {args['query']}\n"
            f"1. Example result — https://example.com/result1\n"
            f"2. Another result — https://example.com/result2",
            {"simulated": True, "query": args["query"]},
        )

    def _tool_browse_page(self, args: dict) -> tuple[str, dict]:
        # Simulated page content — in real use this goes through sandbox browser
        return (
            f"Simulated page content for: {args['url']}\n"
            f"<html><body><h1>Page Title</h1><p>Simulated content.</p></body></html>",
            {"simulated": True, "url": args["url"]},
        )

    def _tool_send_request(self, args: dict) -> tuple[str, dict]:
        if not self.network_allowed:
            return "Network access disabled", {"blocked": True}
        return "Simulated API response", {"simulated": True, "url": args["url"]}

    def _tool_query_database(self, args: dict) -> tuple[str, dict]:
        # Simulated database
        return "Database query executed (simulated)", {"simulated": True}

    def _tool_send_email(self, args: dict) -> tuple[str, dict]:
        # Captured by sandbox — never actually sent
        return (
            f"Email captured by sandbox:\nTo: {args['to']}\nSubject: {args['subject']}\nBody: {args['body']}",
            {"captured": True, "to": args["to"]},
        )

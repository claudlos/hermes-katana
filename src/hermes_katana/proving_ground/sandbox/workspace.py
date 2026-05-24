"""Workspace tools — the actions available to the LLM inside the sandbox.

These are the tools the LLM can call. Each tool executes inside the Docker
sandbox and the result is returned to the LLM. All actions are logged.
"""

import os
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


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
        for key in ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR"):
            if key in os.environ:
                env[key] = os.environ[key]
        env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
        env["HOME"] = str(self.root)
        return env

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

        # --noprofile --norc avoids sourcing /etc/profile, /etc/profile.d/*,
        # ~/.bash_profile, ~/.bashrc — those scripts can re-export the very
        # API keys _subprocess_env just stripped (e.g. /etc/profile.d/aws-cli.sh).
        result = subprocess.run(
            ["/bin/bash", "--noprofile", "--norc", "-c", cmd],
            cwd=str(self.root),
            env=self._subprocess_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        return output, {"exit_code": result.returncode}

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

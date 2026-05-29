"""
Hermes source patches for HermesKatana integration.

Defines the set of source-code patches that wire Katana's middleware,
proxy, banner, Docker support, and gateway scanning into an existing
Hermes checkout.  Patches use a sentinel-based approach: each patch
inserts a unique marker comment so it can be detected and reverted
cleanly.

Patch lifecycle
---------------
1. **detect**: check whether the sentinel exists in the target file.
2. **apply**: if sentinel is absent, find the search text and replace it.
3. **revert**: if sentinel is present, replace back to the original text.

All patches are non-destructive — the original code is preserved in the
replacement (the patch adds *around* the original, not replaces it).

The core patches
---------------------
1. **tool_dispatch_hook** — injects Katana middleware before tool execution.
2. **dispatcher_bootstrap** — attaches checkout-driven runtime state to Hermes dispatchers.
3. **dispatcher_escalation_audit** — scans and audits tool results after dispatch.
4. **proxy_env_vars** — injects HTTP_PROXY / HTTPS_PROXY into subprocess envs.
5. **banner_integration** — shows Katana status in the Hermes startup banner.
6. **docker_proxy_forwarding** — translates proxy address for containers.
7. **gateway_command_scanning** — scans commands in non-interactive (gateway) mode.

Note on (1) vs (2): the dispatch hook self-discovers its runtime via
``discover_checkout_root(__file__)`` and no longer depends on the chain that
``dispatcher_bootstrap`` attaches, so enforcement does not require (2). We keep
(2) because it still owns process-level startup that enforcement does not do:
recording fail-closed bootstrap state, priming the runtime/proxy environment,
and setting the ``KATANA_SOURCE_PATCHED`` marker the native plugin uses to defer
(see ``hermes_katana.bootstrap.bootstrap_dispatcher_failsafe``). The two patches
have distinct responsibilities; (2) is not dead code.
"""

from __future__ import annotations

__all__ = [
    "PatchStatus",
    "PatchResult",
    "Patch",
    "validate_patch_target",
    "create_backup",
    "apply_patches",
    "preview_apply_patches",
    "revert_patches",
    "preview_revert_patches",
    "get_patch_status",
    "CORE_PATCHES",
    "LEGACY_CORE_PATCHES",
    "CURRENT_CORE_PATCHES",
    "_detect_hermes_layout",
]


import logging
import os
import shutil
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch status
# ---------------------------------------------------------------------------


class PatchStatus(str, Enum):
    """Status of a patch operation."""

    APPLIED = "applied"
    REVERTED = "reverted"
    PLANNED = "planned"
    SKIPPED = "skipped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Patch result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchResult:
    """Result of applying or reverting a single patch.

    Attributes:
        name:    The patch name.
        status:  Whether it was applied, reverted, skipped, or errored.
        message: Human-readable explanation.
    """

    name: str
    status: PatchStatus
    message: str


# ---------------------------------------------------------------------------
# Patch definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Patch:
    """A source-code patch to apply to a Hermes checkout.

    Attributes:
        name:         Unique identifier for this patch.
        description:  Human-readable description of what the patch does.
        target_file:  Relative path within the Hermes checkout (e.g.
                      ``hermes/tools/dispatch.py``).
        search_text:  The original code to search for in the target file.
        replace_text: The replacement code (must include the sentinel).
        sentinel:     Unique marker string inserted by the patch.  Used
                      to detect whether the patch is already applied.
        critical:     If True, installation fails if this patch cannot be
                      applied.  Non-critical patches are best-effort.
    """

    name: str
    description: str
    target_file: str
    search_text: str
    replace_text: str
    sentinel: str
    critical: bool = True


# ---------------------------------------------------------------------------
# Core patches (current Hermes layout — tools/registry.py, hermes_cli/, etc.)
# ---------------------------------------------------------------------------

# Sentinel prefix — all sentinels start with this for easy detection
_SENTINEL_PREFIX = "# [KATANA-PATCH]"

CURRENT_CORE_PATCHES: list[Patch] = [
    # -----------------------------------------------------------------------
    # 1. Tool dispatch hook  (model_tools.py — sync dispatch boundary)
    # -----------------------------------------------------------------------
    Patch(
        name="tool_dispatch_hook",
        description="Inject Katana middleware chain before tool execution",
        target_file="model_tools.py",
        search_text="""\
        # ACP/Zed edit approval runs before any file mutation.  The requester
        # is bound via ContextVar only for ACP sessions, so CLI/gateway paths
        # are unaffected when it is unset.
        try:""",
        replace_text="""\
        {sentinel}
        # --- Katana middleware interception ---
        _katana_chain = None
        _katana_ctx = None
        try:
            from hermes_katana.bootstrap import discover_checkout_root, get_runtime_bundle
            from hermes_katana.middleware import CallContext, DispatchDecision
            _katana_checkout_root = discover_checkout_root(__file__)
            _katana_checkout_discovered = _katana_checkout_root is not None
            _katana_bootstrap_failed = False
            _katana_runtime = get_runtime_bundle(_katana_checkout_root) if _katana_checkout_root is not None else None
            if _katana_checkout_discovered and _katana_runtime is None:
                _katana_bootstrap_failed = True
                return json.dumps({{
                    "error": f"Katana security bootstrap failed; refusing to dispatch tool '{{function_name}}': runtime unavailable"
                }}, ensure_ascii=False)
            if _katana_runtime is not None:
                _katana_chain = _katana_runtime.chain
                _katana_ctx = CallContext(
                    tool_name=function_name,
                    args=function_args,
                    extras={{"task_id": task_id or ""}},
                )
                _katana_decision = _katana_chain.execute_pre(_katana_ctx)
                if _katana_decision == DispatchDecision.DENY:
                    return json.dumps({{
                        "error": f"Katana blocked tool '{{function_name}}': "
                        + "; ".join(_katana_ctx.deny_reasons)
                    }}, ensure_ascii=False)
                if _katana_decision == DispatchDecision.ESCALATE:
                    from hermes_katana.escalation import resolve_escalation
                    _katana_reasons = _katana_ctx.escalate_reasons or ["Requires human approval"]
                    _katana_action = getattr(_katana_runtime.state, "escalate_action", "block")
                    if not resolve_escalation(
                        _katana_action,
                        tool_name=function_name,
                        reasons=_katana_reasons,
                        args=function_args,
                        task_id=task_id or "",
                        call_id=getattr(_katana_ctx, "call_id", "") or "",
                    ):
                        return json.dumps({{
                            "error": f"Katana blocked tool '{{function_name}}' pending approval: "
                            + "; ".join(_katana_reasons)
                        }}, ensure_ascii=False)
        except Exception as _katana_exc:
            return json.dumps({{
                "error": f"Katana security bootstrap failed; blocking tool '{{function_name}}': {{type(_katana_exc).__name__}}: {{_katana_exc}}"
            }}, ensure_ascii=False)
        # --- End Katana middleware ---
        # ACP/Zed edit approval runs before any file mutation.  The requester
        # is bound via ContextVar only for ACP sessions, so CLI/gateway paths
        # are unaffected when it is unset.
        try:""".format(sentinel=f"{_SENTINEL_PREFIX} tool_dispatch_hook"),
        sentinel=f"{_SENTINEL_PREFIX} tool_dispatch_hook",
        critical=True,
    ),
    # -----------------------------------------------------------------------
    # 2. Dispatcher bootstrap  (anchored to ToolRegistry.__init__)
    # -----------------------------------------------------------------------
    Patch(
        name="dispatcher_bootstrap",
        description="Initialize Katana middleware chain on ToolRegistry at construction time",
        target_file="tools/registry.py",
        search_text="""\
    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {}
        self._toolset_checks: Dict[str, Callable] = {}""",
        replace_text="""\
    def __init__(self):
        self._tools: Dict[str, ToolEntry] = {{}}
        self._toolset_checks: Dict[str, Callable] = {{}}
        {sentinel}
        # --- Katana bootstrap ---
        try:
            from hermes_katana.bootstrap import bootstrap_dispatcher_failsafe
            bootstrap_dispatcher_failsafe(self, checkout_root=__file__)
        except Exception as _katana_bootstrap_exc:
            self._katana_bootstrap_failed = True
            self._katana_bootstrap_error = f"{{type(_katana_bootstrap_exc).__name__}}: {{_katana_bootstrap_exc}}"
            self._katana_checkout_discovered = True
        # --- End Katana bootstrap ---""".format(sentinel=f"{_SENTINEL_PREFIX} dispatcher_bootstrap"),
        sentinel=f"{_SENTINEL_PREFIX} dispatcher_bootstrap",
        critical=True,
    ),
    # -----------------------------------------------------------------------
    # 3. Post-dispatch scanning/audit  (model_tools.py)
    # -----------------------------------------------------------------------
    Patch(
        name="dispatcher_escalation_audit",
        description="Run Katana post-dispatch scanning before Hermes records the result",
        target_file="model_tools.py",
        search_text="""\
        duration_ms = int((time.monotonic() - _dispatch_start) * 1000)""",
        replace_text="""\
        duration_ms = int((time.monotonic() - _dispatch_start) * 1000)
        {sentinel}
        # --- Katana post-dispatch scan/audit ---
        try:
            if (
                "_katana_chain" in locals()
                and "_katana_ctx" in locals()
                and _katana_chain is not None
                and _katana_ctx is not None
            ):
                _katana_ctx.tool_output = result
                _katana_ctx.tool_error = None
                _katana_chain.execute_post(_katana_ctx)
                if isinstance(_katana_ctx.tool_output, str):
                    result = _katana_ctx.tool_output
        except Exception as _katana_post_exc:
            return json.dumps({{
                "error": f"Katana post-dispatch scan failed; blocking tool '{{function_name}}': {{type(_katana_post_exc).__name__}}: {{_katana_post_exc}}"
            }}, ensure_ascii=False)
        # --- End Katana post-dispatch scan/audit ---""".format(
            sentinel=f"{_SENTINEL_PREFIX} dispatcher_escalation_audit"
        ),
        sentinel=f"{_SENTINEL_PREFIX} dispatcher_escalation_audit",
        critical=True,
    ),
    # -----------------------------------------------------------------------
    # 4. Proxy environment variables  (tools/terminal_tool.py — local env)
    # -----------------------------------------------------------------------
    Patch(
        name="proxy_env_vars",
        description="Inject HTTP_PROXY/HTTPS_PROXY into local subprocess environment",
        target_file="tools/terminal_tool.py",
        search_text="""\
    if env_type == "local":
        return _LocalEnvironment(cwd=cwd, timeout=timeout)""",
        replace_text="""\
    if env_type == "local":
        {sentinel}
        # --- Katana proxy injection ---
        _katana_proxy_env = {{}}
        try:
            import os as _os
            _katana_proxy = _os.environ.get("KATANA_PROXY_URL")
            if _katana_proxy:
                _katana_proxy_env["HTTP_PROXY"] = _katana_proxy
                _katana_proxy_env["HTTPS_PROXY"] = _katana_proxy
                _katana_proxy_env["http_proxy"] = _katana_proxy
                _katana_proxy_env["https_proxy"] = _katana_proxy
                _no_proxy = _os.environ.get("NO_PROXY", "localhost,127.0.0.1,::1")
                _katana_proxy_env["NO_PROXY"] = _no_proxy
                _katana_proxy_env["no_proxy"] = _no_proxy
        except Exception:
            pass
        # --- End Katana proxy injection ---
        return _LocalEnvironment(cwd=cwd, timeout=timeout, env=_katana_proxy_env)""".format(
            sentinel=f"{_SENTINEL_PREFIX} proxy_env_vars"
        ),
        sentinel=f"{_SENTINEL_PREFIX} proxy_env_vars",
        critical=True,
    ),
    # -----------------------------------------------------------------------
    # 5. Banner integration  (hermes_cli/banner.py)
    # -----------------------------------------------------------------------
    Patch(
        name="banner_integration",
        description="Show Katana protection status in the Hermes startup banner",
        target_file="hermes_cli/banner.py",
        search_text="""\
    console.print()
    term_width = shutil.get_terminal_size().columns
    if term_width >= 95:
        _logo = _bskin.banner_logo if _bskin and hasattr(_bskin, 'banner_logo') and _bskin.banner_logo else HERMES_AGENT_LOGO
        console.print(_logo)
        console.print()
    console.print(outer_panel)""",
        replace_text="""\
    console.print()
    term_width = shutil.get_terminal_size().columns
    if term_width >= 95:
        _logo = _bskin.banner_logo if _bskin and hasattr(_bskin, 'banner_logo') and _bskin.banner_logo else HERMES_AGENT_LOGO
        console.print(_logo)
        console.print()
    console.print(outer_panel)
    {sentinel}
    # --- Katana banner integration ---
    try:
        from hermes_katana.cli.main import _format_katana_status
        _katana_status = _format_katana_status()
        if _katana_status:
            console.print(_katana_status)
    except ImportError:
        pass
    # --- End Katana banner ---""".format(sentinel=f"{_SENTINEL_PREFIX} banner_integration"),
        sentinel=f"{_SENTINEL_PREFIX} banner_integration",
        critical=False,
    ),
    # -----------------------------------------------------------------------
    # 6. Docker proxy forwarding  (tools/environments/docker.py)
    # -----------------------------------------------------------------------
    Patch(
        name="docker_proxy_forwarding",
        description="Translate proxy address for Docker containers",
        target_file="tools/environments/docker.py",
        search_text="""\
        exec_env: dict[str, str] = dict(self._env)""",
        replace_text="""\
        exec_env: dict[str, str] = dict(self._env)
        {sentinel}
        # --- Katana Docker proxy forwarding ---
        try:
            _katana_proxy = os.environ.get("KATANA_PROXY_URL")
            if _katana_proxy:
                import re as _re
                _docker_proxy = _re.sub(
                    r"(https?://)(localhost|127\\.0\\.0\\.1)",
                    r"\\1host.docker.internal",
                    _katana_proxy,
                )
                exec_env["HTTP_PROXY"] = _docker_proxy
                exec_env["HTTPS_PROXY"] = _docker_proxy
                exec_env["http_proxy"] = _docker_proxy
                exec_env["https_proxy"] = _docker_proxy
                _katana_ca = os.environ.get("KATANA_CA_CERT")
                if _katana_ca:
                    exec_env["KATANA_CA_CERT"] = _katana_ca
                    exec_env["SSL_CERT_FILE"] = "/tmp/katana-ca.pem"
                    exec_env["REQUESTS_CA_BUNDLE"] = "/tmp/katana-ca.pem"
        except Exception:
            pass
        # --- End Katana Docker proxy ---""".format(sentinel=f"{_SENTINEL_PREFIX} docker_proxy_forwarding"),
        sentinel=f"{_SENTINEL_PREFIX} docker_proxy_forwarding",
        critical=False,
    ),
    # -----------------------------------------------------------------------
    # 7. Gateway command scanning  (gateway/run.py — _handle_message_with_agent)
    # -----------------------------------------------------------------------
    Patch(
        name="gateway_command_scanning",
        description="Scan incoming gateway messages before agent execution",
        target_file="gateway/run.py",
        search_text="""\
    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        \"\"\"Inner handler that runs under the _running_agents sentinel guard.\"\"\"
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)""",
        replace_text="""\
    async def _handle_message_with_agent(self, event, source, _quick_key: str, run_generation: int):
        \"\"\"Inner handler that runs under the _running_agents sentinel guard.\"\"\"
        {sentinel}
        # --- Katana gateway scanning ---
        try:
            from hermes_katana.scanner import scan_command, scan_input, ScanVerdict
            _msg_text = event.text or ""
            _scan_result = scan_command(_msg_text) if _msg_text.startswith("!") else scan_input(_msg_text)
            if _scan_result.verdict == ScanVerdict.BLOCK:
                return (
                    "Katana security scan blocked this message: "
                    + _scan_result.summary
                )
        except ImportError:
            pass
        # --- End Katana gateway scanning ---
        _msg_start_time = time.time()
        _platform_name = source.platform.value if hasattr(source.platform, "value") else str(source.platform)""".format(
            sentinel=f"{_SENTINEL_PREFIX} gateway_command_scanning"
        ),
        sentinel=f"{_SENTINEL_PREFIX} gateway_command_scanning",
        critical=False,
    ),
]

# ---------------------------------------------------------------------------
# Legacy patches (Hermes v0.1.0 layout — hermes/ top-level package)
# ---------------------------------------------------------------------------

LEGACY_CORE_PATCHES: list[Patch] = [
    # -----------------------------------------------------------------------
    # 1. Tool dispatch hook
    # -----------------------------------------------------------------------
    Patch(
        name="tool_dispatch_hook",
        description="Inject Katana middleware chain before tool execution",
        target_file="hermes/tools/dispatch.py",
        search_text="""\
    async def dispatch_tool(self, tool_name: str, args: dict) -> Any:
        \"\"\"Dispatch a tool call.\"\"\"
        tool = self.get_tool(tool_name)
        result = await tool.execute(**args)
        return result""",
        replace_text="""\
    async def dispatch_tool(self, tool_name: str, args: dict) -> Any:
        \"\"\"Dispatch a tool call.\"\"\"
        {sentinel}
        # --- Katana middleware interception ---
        _katana_chain = None
        _katana_ctx = None
        try:
            from hermes_katana.middleware import MiddlewareChain, CallContext, DispatchDecision
        except ImportError as _katana_import_exc:
            raise PermissionError(
                f"Katana middleware unavailable; refusing to dispatch tool '{{tool_name}}': {{_katana_import_exc}}"
            )
        try:
            _katana_chain = getattr(self, '_katana_chain', None)
            _katana_failed = getattr(self, '_katana_bootstrap_failed', False)
            _katana_discovered = getattr(self, '_katana_checkout_discovered', False)
            _katana_failed_err = getattr(self, '_katana_bootstrap_error', None) or 'unknown error'
            if _katana_failed or (_katana_discovered and _katana_chain is None):
                raise PermissionError(
                    f"Katana security bootstrap failed; refusing to dispatch tool '{{tool_name}}': {{_katana_failed_err}}"
                )
            if _katana_chain is not None:
                _katana_ctx = CallContext(tool_name=tool_name, args=args)
                _katana_decision = _katana_chain.execute_pre(_katana_ctx)
                if _katana_decision == DispatchDecision.DENY:
                    raise PermissionError(
                        f"Katana blocked tool '{{tool_name}}': "
                        + "; ".join(_katana_ctx.deny_reasons)
                    )
                if _katana_decision == DispatchDecision.ESCALATE:
                    if not await self._katana_escalate(_katana_ctx):
                        raise PermissionError(
                            f"Katana escalation denied for '{{tool_name}}'"
                        )
        except PermissionError:
            raise
        except Exception as _katana_exc:
            raise PermissionError(
                f"Katana security bootstrap failed; blocking tool '{{tool_name}}': {{type(_katana_exc).__name__}}: {{_katana_exc}}"
            )
        # --- End Katana middleware ---
        tool = self.get_tool(tool_name)
        result = await tool.execute(**args)
        # --- Katana post-dispatch ---
        if _katana_chain is not None and _katana_ctx is not None:
            try:
                _katana_ctx.tool_output = result
                _katana_chain.execute_post(_katana_ctx)
                result = _katana_ctx.tool_output
            except Exception:
                pass
        # --- End Katana post-dispatch ---
        return result""".format(sentinel=f"{_SENTINEL_PREFIX} tool_dispatch_hook"),
        sentinel=f"{_SENTINEL_PREFIX} tool_dispatch_hook",
        critical=True,
    ),
    # -----------------------------------------------------------------------
    # 2. Dispatcher bootstrap
    # -----------------------------------------------------------------------
    Patch(
        name="dispatcher_bootstrap",
        description="Attach checkout-driven runtime state to Hermes dispatchers",
        target_file="hermes/tools/dispatch.py",
        search_text="""\
            _katana_chain = getattr(self, '_katana_chain', None)""",
        replace_text="""\
            {sentinel}
            try:
                from hermes_katana.bootstrap import bootstrap_dispatcher_failsafe
                bootstrap_dispatcher_failsafe(self)
            except Exception as _katana_bootstrap_exc:
                self._katana_bootstrap_failed = True
                self._katana_bootstrap_error = f"{{type(_katana_bootstrap_exc).__name__}}: {{_katana_bootstrap_exc}}"
                self._katana_checkout_discovered = True
            _katana_chain = getattr(self, '_katana_chain', None)""".format(
            sentinel=f"{_SENTINEL_PREFIX} dispatcher_bootstrap"
        ),
        sentinel=f"{_SENTINEL_PREFIX} dispatcher_bootstrap",
        critical=True,
    ),
    # -----------------------------------------------------------------------
    # 3. Escalation denial audit finalization
    # -----------------------------------------------------------------------
    Patch(
        name="dispatcher_escalation_audit",
        description="Record denied escalation outcomes before raising",
        target_file="hermes/tools/dispatch.py",
        search_text="""\
                if _katana_decision == DispatchDecision.ESCALATE:
                    if not await self._katana_escalate(_katana_ctx):
                        raise PermissionError(
                            f"Katana escalation denied for '{tool_name}'"
                        )""",
        replace_text="""\
                if _katana_decision == DispatchDecision.ESCALATE:
                    {sentinel}
                    if not await self._katana_escalate(_katana_ctx):
                        _katana_ctx.deny(
                            f"Human denied Katana escalation for '{{tool_name}}'"
                        )
                        _katana_ctx.tool_error = (
                            f"Katana escalation denied for '{{tool_name}}'"
                        )
                        try:
                            _katana_chain.execute_post(_katana_ctx)
                        except Exception:
                            pass
                        raise PermissionError(
                            f"Katana escalation denied for '{{tool_name}}'"
                        )""".format(sentinel=f"{_SENTINEL_PREFIX} dispatcher_escalation_audit"),
        sentinel=f"{_SENTINEL_PREFIX} dispatcher_escalation_audit",
        critical=True,
    ),
    # -----------------------------------------------------------------------
    # 4. Proxy environment variables
    # -----------------------------------------------------------------------
    Patch(
        name="proxy_env_vars",
        description="Inject HTTP_PROXY/HTTPS_PROXY into subprocess environments",
        target_file="hermes/tools/terminal.py",
        search_text="""\
    async def execute(self, command: str, **kwargs) -> str:
        \"\"\"Execute a shell command.\"\"\"
        env = os.environ.copy()""",
        replace_text="""\
    async def execute(self, command: str, **kwargs) -> str:
        \"\"\"Execute a shell command.\"\"\"
        env = os.environ.copy()
        {sentinel}
        # --- Katana proxy injection ---
        try:
            import hermes_katana
            _katana_proxy = os.environ.get("KATANA_PROXY_URL")
            if _katana_proxy:
                env["HTTP_PROXY"] = _katana_proxy
                env["HTTPS_PROXY"] = _katana_proxy
                env["http_proxy"] = _katana_proxy
                env["https_proxy"] = _katana_proxy
                # Bypass proxy for local addresses
                _no_proxy = env.get("NO_PROXY", "localhost,127.0.0.1,::1")
                env["NO_PROXY"] = _no_proxy
                env["no_proxy"] = _no_proxy
        except ImportError:
            pass
        # --- End Katana proxy injection ---""".format(sentinel=f"{_SENTINEL_PREFIX} proxy_env_vars"),
        sentinel=f"{_SENTINEL_PREFIX} proxy_env_vars",
        critical=True,
    ),
    # -----------------------------------------------------------------------
    # 5. Banner integration
    # -----------------------------------------------------------------------
    Patch(
        name="banner_integration",
        description="Show Katana protection status in the Hermes startup banner",
        target_file="hermes/ui/banner.py",
        search_text="""\
    def show_banner(self) -> None:
        \"\"\"Display the Hermes startup banner.\"\"\"
        console.print(self._build_banner())""",
        replace_text="""\
    def show_banner(self) -> None:
        \"\"\"Display the Hermes startup banner.\"\"\"
        console.print(self._build_banner())
        {sentinel}
        # --- Katana banner integration ---
        try:
            from hermes_katana.cli.main import _format_katana_status
            _katana_status = _format_katana_status()
            if _katana_status:
                console.print(_katana_status)
        except ImportError:
            pass
        # --- End Katana banner ---""".format(sentinel=f"{_SENTINEL_PREFIX} banner_integration"),
        sentinel=f"{_SENTINEL_PREFIX} banner_integration",
        critical=False,
    ),
    # -----------------------------------------------------------------------
    # 6. Docker proxy forwarding
    # -----------------------------------------------------------------------
    Patch(
        name="docker_proxy_forwarding",
        description="Translate proxy address for Docker containers",
        target_file="hermes/tools/docker_tool.py",
        search_text="""\
    def _build_container_env(self) -> dict:
        \"\"\"Build environment variables for the container.\"\"\"
        env = {}""",
        replace_text="""\
    def _build_container_env(self) -> dict:
        \"\"\"Build environment variables for the container.\"\"\"
        env = {{}}
        {sentinel}
        # --- Katana Docker proxy forwarding ---
        try:
            import os
            _katana_proxy = os.environ.get("KATANA_PROXY_URL")
            if _katana_proxy:
                # Translate host.docker.internal for container access
                import re as _re
                _docker_proxy = _re.sub(
                    r"(https?://)(localhost|127\\.0\\.0\\.1)",
                    r"\\1host.docker.internal",
                    _katana_proxy,
                )
                env["HTTP_PROXY"] = _docker_proxy
                env["HTTPS_PROXY"] = _docker_proxy
                env["http_proxy"] = _docker_proxy
                env["https_proxy"] = _docker_proxy
                # Install CA cert if available
                _katana_ca = os.environ.get("KATANA_CA_CERT")
                if _katana_ca:
                    env["KATANA_CA_CERT"] = _katana_ca
                    env["SSL_CERT_FILE"] = "/tmp/katana-ca.pem"
                    env["REQUESTS_CA_BUNDLE"] = "/tmp/katana-ca.pem"
        except ImportError:
            pass
        # --- End Katana Docker proxy ---""".format(sentinel=f"{_SENTINEL_PREFIX} docker_proxy_forwarding"),
        sentinel=f"{_SENTINEL_PREFIX} docker_proxy_forwarding",
        critical=False,
    ),
    # -----------------------------------------------------------------------
    # 7. Gateway command scanning
    # -----------------------------------------------------------------------
    Patch(
        name="gateway_command_scanning",
        description="Scan commands in non-interactive (gateway) mode",
        target_file="hermes/gateway/handler.py",
        search_text="""\
    async def handle_request(self, request: dict) -> dict:
        \"\"\"Handle an incoming gateway request.\"\"\"
        tool_name = request.get("tool")
        args = request.get("args", {})""",
        replace_text="""\
    async def handle_request(self, request: dict) -> dict:
        \"\"\"Handle an incoming gateway request.\"\"\"
        tool_name = request.get("tool")
        args = request.get("args", {{}})
        {sentinel}
        # --- Katana gateway scanning ---
        try:
            from hermes_katana.scanner import scan_command, scan_input, ScanVerdict
            for _arg_name, _arg_val in args.items():
                _text = str(_arg_val) if _arg_val is not None else ""
                if _arg_name in ("command", "cmd", "shell_command"):
                    _scan_result = scan_command(_text)
                else:
                    _scan_result = scan_input(_text)
                if _scan_result.verdict == ScanVerdict.BLOCK:
                    return {{
                        "error": "Katana security scan blocked this request",
                        "details": _scan_result.summary,
                        "blocked": True,
                    }}
        except ImportError:
            pass
        # --- End Katana gateway scanning ---""".format(sentinel=f"{_SENTINEL_PREFIX} gateway_command_scanning"),
        sentinel=f"{_SENTINEL_PREFIX} gateway_command_scanning",
        critical=False,
    ),
]

# ---------------------------------------------------------------------------
# Layout detection and default alias
# ---------------------------------------------------------------------------


def _detect_hermes_layout(target: Path) -> str:
    """Detect which Hermes layout is present in *target*.

    Returns:
        ``"current"``       — post-v0.1.0 layout (tools/registry.py + hermes_cli/)
        ``"legacy-v0.1.0"`` — old layout with hermes/ top-level package

    Raises:
        ValueError: When neither layout is detected.
    """
    if (target / "hermes" / "tools" / "dispatch.py").exists():
        return "legacy-v0.1.0"
    if (target / "tools" / "registry.py").exists() and (target / "hermes_cli").exists():
        return "current"
    raise ValueError(
        f"Unsupported Hermes layout in {target}: "
        "neither hermes/tools/dispatch.py nor tools/registry.py + hermes_cli/ found"
    )


# Default alias — points to the current layout so existing callers continue to work.
CORE_PATCHES: list[Patch] = CURRENT_CORE_PATCHES


# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------


def validate_patch_target(target_file: Path, patch: Patch) -> list[str]:
    """Validate file permissions and ownership before patching.

    Checks:
    - File exists and is a regular file
    - File is owned by current user (or we have write access)
    - No unexpected setuid/setgid bits
    - File is not a symlink (prevents symlink attacks)

    Returns:
        List of warning/error messages (empty if all checks pass).
    """
    issues: list[str] = []

    if not target_file.exists():
        issues.append(f"Target file does not exist: {target_file}")
        return issues

    if target_file.is_symlink():
        issues.append(f"Target is a symlink (potential symlink attack): {target_file}")

    if not target_file.is_file():
        issues.append(f"Target is not a regular file: {target_file}")
        return issues

    try:
        st = target_file.stat()
        if st.st_mode & (stat.S_ISUID | stat.S_ISGID):
            issues.append(f"Target has setuid/setgid bits set: {target_file} (mode: {oct(st.st_mode)})")
        # POSIX-only ownership check; Windows uses ACLs and has no os.getuid()
        if hasattr(os, "getuid"):
            current_uid = os.getuid()
            if st.st_uid != current_uid and current_uid != 0:
                issues.append(f"Target owned by uid {st.st_uid}, not current user ({current_uid}): {target_file}")
        if not os.access(target_file, os.W_OK):
            issues.append(f"No write permission on target: {target_file}")
    except OSError as exc:
        issues.append(f"Cannot stat target file: {exc}")

    return issues


def create_backup(target_file: Path) -> "Path | None":
    """Create a backup of the target file before patching.

    Returns:
        Path to the backup file, or None if backup failed.
    """
    if not target_file.exists():
        return None
    backup_path = target_file.with_suffix(target_file.suffix + ".katana-backup")
    try:
        shutil.copy2(str(target_file), str(backup_path))
        return backup_path
    except OSError as exc:
        logger.warning("Failed to create backup of %s: %s", target_file, exc)
        return None


def _is_patch_applied(target_path: Path, patch: Patch) -> bool:
    """Check whether a patch's sentinel exists in the target file.

    Args:
        target_path: Absolute path to the target file.
        patch:       The patch to check.

    Returns:
        True if the sentinel is found in the file.
    """
    if not target_path.exists():
        return False
    content = target_path.read_text(encoding="utf-8")
    return patch.sentinel in content


def apply_patches(
    target: str | Path,
    patches: list[Patch] | None = None,
) -> list[PatchResult]:
    """Apply all patches to a Hermes checkout.

    Each patch is applied independently.  If a patch's sentinel is already
    present, it is skipped.  If the search text is not found, it is either
    an error (if critical) or skipped (if non-critical).

    Args:
        target:  Path to the Hermes checkout root.
        patches: Patches to apply (defaults to :data:`CORE_PATCHES`).

    Returns:
        List of :class:`PatchResult` for each patch.
    """
    target = Path(target)
    patches = patches or CORE_PATCHES
    results: list[PatchResult] = []

    for patch in patches:
        target_file = target / patch.target_file

        # Check if already applied
        if _is_patch_applied(target_file, patch):
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.SKIPPED,
                    message=f"Already applied (sentinel found in {patch.target_file})",
                )
            )
            continue

        # Check if target file exists
        if not target_file.exists():
            msg = f"Target file not found: {patch.target_file}"
            status = PatchStatus.ERROR if patch.critical else PatchStatus.SKIPPED
            results.append(PatchResult(name=patch.name, status=status, message=msg))
            logger.log(
                logging.ERROR if patch.critical else logging.WARNING,
                "Patch %s: %s",
                patch.name,
                msg,
            )
            continue

        # Validate file permissions and ownership
        issues = validate_patch_target(target_file, patch)
        if issues:
            for issue in issues:
                logger.warning("Patch %s: %s", patch.name, issue)
            critical_issues = [i for i in issues if "setuid" in i or "symlink" in i]
            if critical_issues:
                results.append(
                    PatchResult(
                        name=patch.name,
                        status=PatchStatus.ERROR,
                        message=f"Permission validation failed: {'; '.join(critical_issues)}",
                    )
                )
                continue

        # Create backup before patching
        backup = create_backup(target_file)
        if backup:
            logger.debug("Backup created: %s", backup)

        # Read current content
        try:
            content = target_file.read_text(encoding="utf-8")
        except OSError as exc:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR,
                    message=f"Cannot read {patch.target_file}: {exc}",
                )
            )
            continue

        # Search for the anchor text
        if patch.search_text not in content:
            msg = f"Search text not found in {patch.target_file} (file may have changed)"
            status = PatchStatus.ERROR if patch.critical else PatchStatus.SKIPPED
            results.append(PatchResult(name=patch.name, status=status, message=msg))
            logger.log(
                logging.ERROR if patch.critical else logging.WARNING,
                "Patch %s: %s",
                patch.name,
                msg,
            )
            continue

        # Reject ambiguous anchors. ``replace(..., 1)`` would silently patch the
        # first of several matches, which for a security-enforcement hook could
        # inject in the wrong place while still leaving a sentinel that looks
        # applied. Fail loudly instead so the anchor can be made more specific.
        if content.count(patch.search_text) > 1:
            msg = (
                f"Anchor text matches {content.count(patch.search_text)} locations in "
                f"{patch.target_file}; refusing to patch an ambiguous anchor"
            )
            results.append(PatchResult(name=patch.name, status=PatchStatus.ERROR, message=msg))
            logger.error("Patch %s: %s", patch.name, msg)
            continue

        # Apply the patch
        try:
            new_content = content.replace(patch.search_text, patch.replace_text, 1)
            target_file.write_text(new_content, encoding="utf-8")
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.APPLIED,
                    message=f"Applied to {patch.target_file}",
                )
            )
            logger.info("Patch %s applied to %s", patch.name, patch.target_file)
        except OSError as exc:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR,
                    message=f"Write failed for {patch.target_file}: {exc}",
                )
            )

    return results


def preview_apply_patches(
    target: str | Path,
    patches: list[Patch] | None = None,
) -> list[PatchResult]:
    """Preview patch application without modifying the checkout."""
    target = Path(target)
    patches = patches or CORE_PATCHES
    results: list[PatchResult] = []
    virtual_contents: dict[Path, str] = {}

    for patch in patches:
        target_file = target / patch.target_file

        if not target_file.exists():
            msg = f"Target file not found: {patch.target_file}"
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR if patch.critical else PatchStatus.SKIPPED,
                    message=msg,
                )
            )
            continue

        try:
            content = virtual_contents.get(target_file)
            if content is None:
                content = target_file.read_text(encoding="utf-8")
        except OSError as exc:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR,
                    message=f"Cannot read {patch.target_file}: {exc}",
                )
            )
            continue

        if patch.sentinel in content:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.SKIPPED,
                    message=f"Already applied (sentinel found in {patch.target_file})",
                )
            )
            virtual_contents[target_file] = content
            continue

        if patch.search_text not in content:
            msg = f"Search text not found in {patch.target_file} (file may have changed)"
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR if patch.critical else PatchStatus.SKIPPED,
                    message=msg,
                )
            )
            virtual_contents[target_file] = content
            continue

        if content.count(patch.search_text) > 1:
            msg = (
                f"Anchor text matches {content.count(patch.search_text)} locations in "
                f"{patch.target_file}; refusing to patch an ambiguous anchor"
            )
            results.append(PatchResult(name=patch.name, status=PatchStatus.ERROR, message=msg))
            virtual_contents[target_file] = content
            continue

        results.append(
            PatchResult(
                name=patch.name,
                status=PatchStatus.PLANNED,
                message=f"Would apply to {patch.target_file}",
            )
        )
        virtual_contents[target_file] = content.replace(patch.search_text, patch.replace_text, 1)

    return results


def revert_patches(
    target: str | Path,
    patches: list[Patch] | None = None,
) -> list[PatchResult]:
    """Revert all patches from a Hermes checkout.

    For each patch whose sentinel is found, the replacement text is
    swapped back to the original search text.

    Args:
        target:  Path to the Hermes checkout root.
        patches: Patches to revert (defaults to :data:`CORE_PATCHES`).

    Returns:
        List of :class:`PatchResult` for each patch.
    """
    target = Path(target)
    patches = patches or CORE_PATCHES
    results: list[PatchResult] = []

    for patch in reversed(patches):
        target_file = target / patch.target_file

        # Check if patch is actually applied
        if not _is_patch_applied(target_file, patch):
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.SKIPPED,
                    message=f"Not applied (sentinel not found in {patch.target_file})",
                )
            )
            continue

        # Read current content
        try:
            content = target_file.read_text(encoding="utf-8")
        except OSError as exc:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR,
                    message=f"Cannot read {patch.target_file}: {exc}",
                )
            )
            continue

        # Revert: replace the patched text back to original
        if patch.replace_text not in content:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR,
                    message=f"Replacement text not found in {patch.target_file} (manually modified?)",
                )
            )
            continue

        try:
            new_content = content.replace(patch.replace_text, patch.search_text, 1)
            target_file.write_text(new_content, encoding="utf-8")
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.REVERTED,
                    message=f"Reverted in {patch.target_file}",
                )
            )
            logger.info("Patch %s reverted in %s", patch.name, patch.target_file)
        except OSError as exc:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR,
                    message=f"Write failed for {patch.target_file}: {exc}",
                )
            )

    return results


def preview_revert_patches(
    target: str | Path,
    patches: list[Patch] | None = None,
) -> list[PatchResult]:
    """Preview patch reversion without modifying the checkout."""
    target = Path(target)
    patches = patches or CORE_PATCHES
    results: list[PatchResult] = []
    virtual_contents: dict[Path, str] = {}

    for patch in reversed(patches):
        target_file = target / patch.target_file

        if not target_file.exists():
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.SKIPPED,
                    message=f"Not applied (sentinel not found in {patch.target_file})",
                )
            )
            continue

        try:
            content = virtual_contents.get(target_file)
            if content is None:
                content = target_file.read_text(encoding="utf-8")
        except OSError as exc:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.ERROR,
                    message=f"Cannot read {patch.target_file}: {exc}",
                )
            )
            continue

        if patch.sentinel not in content:
            results.append(
                PatchResult(
                    name=patch.name,
                    status=PatchStatus.SKIPPED,
                    message=f"Not applied (sentinel not found in {patch.target_file})",
                )
            )
            virtual_contents[target_file] = content
            continue

        results.append(
            PatchResult(
                name=patch.name,
                status=PatchStatus.PLANNED,
                message=f"Would revert in {patch.target_file}",
            )
        )
        virtual_contents[target_file] = content.replace(patch.replace_text, patch.search_text, 1)

    return results


def get_patch_status(
    target: str | Path,
    patches: list[Patch] | None = None,
) -> dict[str, bool]:
    """Check which patches are currently applied.

    Args:
        target:  Path to the Hermes checkout root.
        patches: Patches to check (defaults to :data:`CORE_PATCHES`).

    Returns:
        Dict mapping patch name → bool (True if applied).
    """
    target = Path(target)
    patches = patches or CORE_PATCHES
    return {patch.name: _is_patch_applied(target / patch.target_file, patch) for patch in patches}

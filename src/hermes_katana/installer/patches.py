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

The five core patches
---------------------
1. **tool_dispatch_hook** — injects Katana middleware before tool execution.
2. **proxy_env_vars** — injects HTTP_PROXY / HTTPS_PROXY into subprocess envs.
3. **banner_integration** — shows Katana status in the Hermes startup banner.
4. **docker_proxy_forwarding** — translates proxy address for containers.
5. **gateway_command_scanning** — scans commands in non-interactive (gateway) mode.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch status
# ---------------------------------------------------------------------------


class PatchStatus(str, Enum):
    """Status of a patch operation."""

    APPLIED = "applied"
    REVERTED = "reverted"
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
# Core patches
# ---------------------------------------------------------------------------

# Sentinel prefix — all sentinels start with this for easy detection
_SENTINEL_PREFIX = "# [KATANA-PATCH]"

CORE_PATCHES: list[Patch] = [
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
        try:
            from hermes_katana.middleware import MiddlewareChain, CallContext, DispatchDecision
            _katana_chain = getattr(self, '_katana_chain', None)
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
        except ImportError:
            pass
        # --- End Katana middleware ---
        tool = self.get_tool(tool_name)
        result = await tool.execute(**args)
        # --- Katana post-dispatch ---
        try:
            if _katana_chain is not None:
                _katana_ctx.tool_output = result
                _katana_chain.execute_post(_katana_ctx)
        except (ImportError, NameError):
            pass
        # --- End Katana post-dispatch ---
        return result""".format(sentinel=f"{_SENTINEL_PREFIX} tool_dispatch_hook"),
        sentinel=f"{_SENTINEL_PREFIX} tool_dispatch_hook",
        critical=True,
    ),

    # -----------------------------------------------------------------------
    # 2. Proxy environment variables
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
        # --- End Katana proxy injection ---""".format(
            sentinel=f"{_SENTINEL_PREFIX} proxy_env_vars"
        ),
        sentinel=f"{_SENTINEL_PREFIX} proxy_env_vars",
        critical=True,
    ),

    # -----------------------------------------------------------------------
    # 3. Banner integration
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
        # --- End Katana banner ---""".format(
            sentinel=f"{_SENTINEL_PREFIX} banner_integration"
        ),
        sentinel=f"{_SENTINEL_PREFIX} banner_integration",
        critical=False,
    ),

    # -----------------------------------------------------------------------
    # 4. Docker proxy forwarding
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
        # --- End Katana Docker proxy ---""".format(
            sentinel=f"{_SENTINEL_PREFIX} docker_proxy_forwarding"
        ),
        sentinel=f"{_SENTINEL_PREFIX} docker_proxy_forwarding",
        critical=False,
    ),

    # -----------------------------------------------------------------------
    # 5. Gateway command scanning
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
        # --- End Katana gateway scanning ---""".format(
            sentinel=f"{_SENTINEL_PREFIX} gateway_command_scanning"
        ),
        sentinel=f"{_SENTINEL_PREFIX} gateway_command_scanning",
        critical=False,
    ),
]


# ---------------------------------------------------------------------------
# Patch operations
# ---------------------------------------------------------------------------


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
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.SKIPPED,
                message=f"Already applied (sentinel found in {patch.target_file})",
            ))
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

        # Read current content
        try:
            content = target_file.read_text(encoding="utf-8")
        except OSError as exc:
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.ERROR,
                message=f"Cannot read {patch.target_file}: {exc}",
            ))
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

        # Apply the patch
        try:
            new_content = content.replace(patch.search_text, patch.replace_text, 1)
            target_file.write_text(new_content, encoding="utf-8")
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.APPLIED,
                message=f"Applied to {patch.target_file}",
            ))
            logger.info("Patch %s applied to %s", patch.name, patch.target_file)
        except OSError as exc:
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.ERROR,
                message=f"Write failed for {patch.target_file}: {exc}",
            ))

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

    for patch in patches:
        target_file = target / patch.target_file

        # Check if patch is actually applied
        if not _is_patch_applied(target_file, patch):
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.SKIPPED,
                message=f"Not applied (sentinel not found in {patch.target_file})",
            ))
            continue

        # Read current content
        try:
            content = target_file.read_text(encoding="utf-8")
        except OSError as exc:
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.ERROR,
                message=f"Cannot read {patch.target_file}: {exc}",
            ))
            continue

        # Revert: replace the patched text back to original
        if patch.replace_text not in content:
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.ERROR,
                message=f"Replacement text not found in {patch.target_file} (manually modified?)",
            ))
            continue

        try:
            new_content = content.replace(patch.replace_text, patch.search_text, 1)
            target_file.write_text(new_content, encoding="utf-8")
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.REVERTED,
                message=f"Reverted in {patch.target_file}",
            ))
            logger.info("Patch %s reverted in %s", patch.name, patch.target_file)
        except OSError as exc:
            results.append(PatchResult(
                name=patch.name,
                status=PatchStatus.ERROR,
                message=f"Write failed for {patch.target_file}: {exc}",
            ))

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
    return {
        patch.name: _is_patch_applied(target / patch.target_file, patch)
        for patch in patches
    }

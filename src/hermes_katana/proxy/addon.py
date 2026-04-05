"""
KatanaAddon - mitmproxy addon for HermesKatana.

Integrates the scanner module with the MITM proxy to provide:
- Request scanning: secret leakage, injection detection, rate limiting
- Response scanning: content attacks, indirect prompt injection
- Credential injection from vault for 12+ LLM providers
- Domain allowlist enforcement
- X-Katana-Scanned header injection for downstream awareness
- Thread-safe rate tracking with anomaly escalation
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_katana.audit.trail import AuditTrail
    from hermes_katana.vault.store import Vault

from hermes_katana.proxy.config import ProxyConfig
from hermes_katana.proxy.injector import inject_credentials

logger = logging.getLogger(__name__)


@dataclass
class RateTracker:
    """Thread-safe per-client rate tracker with escalation.

    Tracks request timestamps per client IP and detects anomalous
    bursts that exceed the configured rate limit.

    Attributes:
        max_requests: Maximum requests allowed per window.
        window_seconds: Duration of the sliding window.
        escalation_factor: Multiplier for successive violations.
    """

    max_requests: int = 50
    window_seconds: float = 1.0
    escalation_factor: float = 2.0
    _MAX_CLIENTS: int = 10_000

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _windows: dict[str, deque[float]] = field(
        default_factory=lambda: defaultdict(deque), repr=False
    )
    _violations: dict[str, int] = field(
        default_factory=lambda: defaultdict(int), repr=False
    )

    def check(self, client_id: str) -> tuple[bool, int]:
        """Check if a request from client_id is within rate limits.

        Args:
            client_id: Client identifier (typically IP address).

        Returns:
            Tuple of (allowed: bool, current_count: int).
            If not allowed, the request should be blocked.
        """
        now = time.monotonic()
        with self._lock:
            window = self._windows[client_id]
            cutoff = now - self.window_seconds

            # Purge expired entries
            while window and window[0] < cutoff:
                window.popleft()

            # Apply escalation: reduce limit for repeat violators
            violations = self._violations[client_id]
            effective_limit = max(
                1,
                int(
                    self.max_requests
                    / (self.escalation_factor ** min(violations, 5))
                ),
            )

            if len(window) >= effective_limit:
                self._violations[client_id] = violations + 1
                return False, len(window)

            window.append(now)
            # Decay violations over time
            if violations > 0 and len(window) < effective_limit // 2:
                self._violations[client_id] = max(0, violations - 1)

            # Evict stale clients when tracking too many
            if len(self._windows) > self._MAX_CLIENTS:
                # Remove clients with empty windows (already pruned)
                stale = [k for k, v in self._windows.items() if not v]
                for k in stale:
                    del self._windows[k]
                    self._violations.pop(k, None)

            return True, len(window)

    def get_stats(self) -> dict[str, Any]:
        """Return rate tracking statistics."""
        with self._lock:
            return {
                "tracked_clients": len(self._windows),
                "total_violations": dict(self._violations),
            }

    def reset(self) -> None:
        """Reset all rate tracking state."""
        with self._lock:
            self._windows.clear()
            self._violations.clear()


class KatanaAddon:
    """mitmproxy addon for HermesKatana proxy.

    Intercepts HTTP/HTTPS traffic and applies multi-layer security scanning
    on both requests and responses. Integrates with the vault for credential
    injection and the audit trail for logging.

    Args:
        config: Proxy configuration.
        vault: Optional vault for credential injection.
        audit: Optional audit trail for logging scan results.
    """

    def __init__(
        self,
        config: ProxyConfig,
        vault: Optional["Vault"] = None,
        audit: Optional["AuditTrail"] = None,
    ) -> None:
        self.config = config
        self.vault = vault
        self.audit = audit
        self._rate_tracker = RateTracker(
            max_requests=config.rate_limit_requests,
            window_seconds=config.rate_limit_window,
        )
        self._vault_values: Optional[set[str]] = None
        self._vault_values_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = defaultdict(int)

        # Pre-cache vault values for secret scanning
        self._refresh_vault_values()

    def _refresh_vault_values(self) -> None:
        """Refresh the cached set of vault values for secret scanning."""
        if self.vault is None:
            return
        try:
            with self._vault_values_lock:
                all_vals = self.vault._get_all_values()
                self._vault_values = set(all_vals.values()) if all_vals else None
        except Exception:
            logger.debug("Could not refresh vault values for scanning")

    def _increment_stat(self, key: str) -> None:
        """Thread-safe stat increment."""
        with self._stats_lock:
            self._stats[key] += 1

    def _is_ignored_host(self, host: str) -> bool:
        """Check if a host should be passed through without scanning."""
        host_lower = host.lower()
        return host_lower in self.config.ignore_hosts

    def _is_allowed_domain(self, host: str) -> bool:
        """Check if a domain is in the allowlist (if configured)."""
        if not self.config.allowed_domains:
            return True  # No allowlist = all domains allowed
        host_lower = host.lower()
        return any(
            host_lower == d or host_lower.endswith(f".{d}")
            for d in self.config.allowed_domains
        )

    def _body_too_large(self, body: Optional[bytes]) -> bool:
        """Check if a body exceeds the scan size limit."""
        if body is None:
            return False
        if self.config.max_body_scan_size == 0:
            return False  # Unlimited
        return len(body) > self.config.max_body_scan_size

    def _scan_text(
        self,
        text: str,
        direction: str = "request",
    ) -> dict[str, Any]:
        """Run the scanner on text content.

        Args:
            text: The text to scan.
            direction: Either 'request' or 'response'.

        Returns:
            Dict with scan results and verdict.
        """
        # Lazy import to avoid circular dependency at module load
        from hermes_katana.scanner import scan_input, scan_output

        modes = self.config.scan_modes
        vault_vals = None
        with self._vault_values_lock:
            vault_vals = self._vault_values

        if direction == "request":
            result = scan_input(
                text,
                vault_values=vault_vals,
                check_injection=modes.injection,
                check_secrets=modes.secrets,
                check_unicode=modes.unicode,
                check_content=modes.content,
            )
        else:
            # Response scanning includes indirect injection detection
            result = scan_output(
                text,
                vault_values=vault_vals,
                check_content=modes.content,
                check_secrets=modes.secrets,
                check_unicode=modes.unicode,
                check_injection=modes.injection,  # Indirect injection defense
            )

        return {
            "verdict": result.verdict.value,
            "risk_score": result.risk_score,
            "is_blocked": result.is_blocked,
            "finding_count": result.finding_count,
            "summary": result.summary,
        }

    def _args_hash(self, *args: str) -> str:
        """Compute a deterministic hash of arguments for audit logging."""
        combined = "|".join(args)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # mitmproxy hooks
    # ------------------------------------------------------------------

    def request(self, flow: Any) -> None:
        """mitmproxy request hook.

        Called for every HTTP(S) request passing through the proxy.
        Performs: domain allowlist check, rate limiting, credential
        injection, and request body scanning.
        """
        try:
            host = flow.request.host
        except AttributeError:
            return

        self._increment_stat("requests_total")

        # Skip ignored hosts
        if self._is_ignored_host(host):
            self._increment_stat("requests_ignored")
            return

        # Domain allowlist check
        if not self._is_allowed_domain(host):
            self._increment_stat("requests_blocked_domain")
            logger.warning("Blocked request to non-allowed domain: %s", host)
            flow.response = _make_block_response(
                403, f"Domain not in allowlist: {host}"
            )
            self._log_audit(
                "POLICY_DECISION",
                f"domain_block:{host}",
                "deny",
                f"Domain {host} not in allowlist",
            )
            return

        # Rate limiting
        client_id = _get_client_id(flow)
        allowed, count = self._rate_tracker.check(client_id)
        if not allowed:
            self._increment_stat("requests_rate_limited")
            logger.warning(
                "Rate limit exceeded for %s (%d requests in window)",
                client_id,
                count,
            )
            flow.response = _make_block_response(
                429, "Rate limit exceeded"
            )
            self._log_audit(
                "RATE_ANOMALY",
                f"rate_limit:{client_id}",
                "deny",
                f"Rate limit exceeded: {count} requests",
            )
            return

        # Credential injection
        if self.config.inject_credentials and self.vault is not None:
            provider_name = inject_credentials(flow, self.vault)
            if provider_name:
                self._increment_stat("credentials_injected")
                logger.debug("Injected credentials for %s", provider_name)

        # --- Scan request URL, headers, query params, cookies (GAP 3.1) ---
        # Collect injected header keys so we can skip them during scanning
        _injected_headers: set[str] = set()
        if self.config.inject_credentials and self.vault is not None:
            # The injector may have added Authorization or api-key headers
            for hdr in ("authorization", "api-key", "x-api-key"):
                if hdr in {k.lower() for k in flow.request.headers}:
                    _injected_headers.add(hdr)

        # Scan URL path segments
        try:
            url_text = flow.request.url
            if url_text:
                url_result = self._scan_text(url_text, direction="request")
                if url_result["is_blocked"]:
                    self._increment_stat("requests_blocked_scan")
                    flow.response = _make_block_response(400, f"URL blocked: {url_result['summary']}")
                    self._log_audit("SCAN_RESULT", f"url_scan:{host}", "deny", url_result["summary"])
                    return
        except Exception as exc:
            logger.debug("URL scan error: %s", exc)

        # Scan request headers (skip injected ones — GAP 3.8)
        try:
            for hdr_name, hdr_value in flow.request.headers.items():
                if hdr_name.lower() in _injected_headers:
                    continue
                hdr_result = self._scan_text(hdr_value, direction="request")
                if hdr_result["is_blocked"]:
                    self._increment_stat("requests_blocked_scan")
                    flow.response = _make_block_response(400, f"Header blocked: {hdr_result['summary']}")
                    self._log_audit("SCAN_RESULT", f"header_scan:{host}:{hdr_name}", "deny", hdr_result["summary"])
                    return
        except Exception as exc:
            logger.debug("Header scan error: %s", exc)

        # Scan query parameters
        try:
            if hasattr(flow.request, 'query') and flow.request.query:
                for qname, qvalue in flow.request.query.items():
                    q_result = self._scan_text(qvalue, direction="request")
                    if q_result["is_blocked"]:
                        self._increment_stat("requests_blocked_scan")
                        flow.response = _make_block_response(400, f"Query param blocked: {q_result['summary']}")
                        self._log_audit("SCAN_RESULT", f"query_scan:{host}:{qname}", "deny", q_result["summary"])
                        return
        except Exception as exc:
            logger.debug("Query param scan error: %s", exc)

        # Scan cookie values
        try:
            cookie_header = flow.request.headers.get("cookie", "")
            if cookie_header:
                for part in cookie_header.split(";"):
                    if "=" in part:
                        _, cval = part.split("=", 1)
                        c_result = self._scan_text(cval.strip(), direction="request")
                        if c_result["is_blocked"]:
                            self._increment_stat("requests_blocked_scan")
                            flow.response = _make_block_response(400, f"Cookie blocked: {c_result['summary']}")
                            self._log_audit("SCAN_RESULT", f"cookie_scan:{host}", "deny", c_result["summary"])
                            return
        except Exception as exc:
            logger.debug("Cookie scan error: %s", exc)

        # Request body scanning (GAP 3.5 — scan prefix of oversized bodies)
        body = flow.request.get_content()
        if body:
            scan_body = body
            if self._body_too_large(body):
                logger.warning(
                    "Oversized request body (%d bytes) from %s to %s — scanning first %d bytes",
                    len(body), client_id, host, self.config.max_body_scan_size,
                )
                scan_body = body[:self.config.max_body_scan_size]
                self._increment_stat("requests_oversized")
            try:
                text = scan_body.decode("utf-8", errors="replace")
                scan_result = self._scan_text(text, direction="request")

                if scan_result["is_blocked"]:
                    self._increment_stat("requests_blocked_scan")
                    logger.warning(
                        "Request blocked by scan: %s -> %s",
                        host,
                        scan_result["summary"],
                    )
                    flow.response = _make_block_response(
                        400,
                        f"Request blocked: {scan_result['summary']}",
                    )
                    self._log_audit(
                        "SCAN_RESULT",
                        f"request_scan:{host}",
                        "deny",
                        scan_result["summary"],
                    )
                    return
                elif scan_result["finding_count"] > 0:
                    self._increment_stat("requests_warned")
                    self._log_audit(
                        "SCAN_RESULT",
                        f"request_scan:{host}",
                        "warn",
                        scan_result["summary"],
                    )
            except Exception as exc:
                logger.debug("Request body scan error: %s", exc)

        self._increment_stat("requests_passed")

    def response(self, flow: Any) -> None:
        """mitmproxy response hook.

        Called for every HTTP(S) response passing through the proxy.
        Performs: response body scanning (content attacks, indirect
        injection), and injects the X-Katana-Scanned header.
        """
        try:
            host = flow.request.host
        except AttributeError:
            return

        self._increment_stat("responses_total")

        # Skip ignored hosts
        if self._is_ignored_host(host):
            return

        # Scan response body
        scanned = False
        try:
            body = flow.response.get_content()
        except AttributeError:
            body = None

        # Scan response headers (GAP 3.2)
        try:
            for hdr_name, hdr_value in flow.response.headers.items():
                rh_result = self._scan_text(hdr_value, direction="response")
                if rh_result["is_blocked"]:
                    self._increment_stat("responses_blocked_scan")
                    logger.warning("Response header blocked: %s -> %s", hdr_name, rh_result["summary"])
                    flow.response.set_content(
                        f"[HermesKatana] Response header blocked: {rh_result['summary']}".encode()
                    )
                    flow.response.status_code = 502
                    self._log_audit("SCAN_RESULT", f"response_header_scan:{host}:{hdr_name}", "deny", rh_result["summary"])
                    return
        except Exception as exc:
            logger.debug("Response header scan error: %s", exc)

        if body:
            scan_body = body
            if self._body_too_large(body):
                logger.warning(
                    "Oversized response body (%d bytes) from %s — scanning first %d bytes",
                    len(body), host, self.config.max_body_scan_size,
                )
                scan_body = body[:self.config.max_body_scan_size]
                self._increment_stat("responses_oversized")
            try:
                text = scan_body.decode("utf-8", errors="replace")
                scan_result = self._scan_text(text, direction="response")
                scanned = True

                if scan_result["is_blocked"]:
                    self._increment_stat("responses_blocked_scan")
                    logger.warning(
                        "Response blocked by scan: %s -> %s",
                        host,
                        scan_result["summary"],
                    )
                    # Replace response body with warning
                    flow.response.set_content(
                        f"[HermesKatana] Response blocked: {scan_result['summary']}".encode()
                    )
                    flow.response.status_code = 502
                    self._log_audit(
                        "SCAN_RESULT",
                        f"response_scan:{host}",
                        "deny",
                        scan_result["summary"],
                    )
                elif scan_result["finding_count"] > 0:
                    self._increment_stat("responses_warned")
                    self._log_audit(
                        "SCAN_RESULT",
                        f"response_scan:{host}",
                        "warn",
                        scan_result["summary"],
                    )
            except Exception as exc:
                logger.debug("Response body scan error: %s", exc)

        # Inject X-Katana-Scanned header (opt-in — GAP 3.7)
        if self.config.add_scanned_header:
            try:
                flow.response.headers["X-Katana-Scanned"] = (
                    "true" if scanned else "passthrough"
                )
            except AttributeError:
                pass

        self._increment_stat("responses_passed")

    def _log_audit(
        self,
        event_type: str,
        tool_name: str,
        decision: str,
        details: str,
    ) -> None:
        """Log an event to the audit trail if available."""
        if self.audit is None:
            return
        try:
            from hermes_katana.audit.trail import AuditEntry, AuditEventType

            entry = AuditEntry(
                event_type=AuditEventType(event_type.lower())
                if hasattr(AuditEventType, event_type)
                else AuditEventType.SCAN_RESULT,
                tool_name=tool_name,
                args_hash=self._args_hash(tool_name, details),
                decision=decision,
                details=details,
            )
            self.audit.log(entry)
        except Exception as exc:
            logger.debug("Audit logging failed: %s", exc)

    def websocket_message(self, flow: Any) -> None:
        """mitmproxy WebSocket message hook (GAP 3.3).

        Scans WebSocket message content through the same pipeline.
        """
        try:
            msg = flow.websocket.messages[-1]
            content = msg.content
        except (AttributeError, IndexError):
            return

        self._increment_stat("ws_messages_total")

        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)

        try:
            direction = "request" if getattr(msg, "from_client", True) else "response"
            scan_result = self._scan_text(text, direction=direction)

            if scan_result["is_blocked"]:
                self._increment_stat("ws_messages_blocked")
                logger.warning("WebSocket message blocked: %s", scan_result["summary"])
                msg.content = f"[HermesKatana] Blocked: {scan_result['summary']}".encode()
                self._log_audit(
                    "SCAN_RESULT", "websocket_scan", "deny", scan_result["summary"],
                )
            elif scan_result["finding_count"] > 0:
                self._increment_stat("ws_messages_warned")
                self._log_audit(
                    "SCAN_RESULT", "websocket_scan", "warn", scan_result["summary"],
                )
        except Exception as exc:
            logger.debug("WebSocket scan error: %s", exc)

    def get_stats(self) -> dict[str, Any]:
        """Return addon statistics."""
        with self._stats_lock:
            stats = dict(self._stats)
        stats["rate_tracker"] = self._rate_tracker.get_stats()
        return stats


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def _get_client_id(flow: Any) -> str:
    """Extract a client identifier from a flow."""
    try:
        return flow.client_conn.peername[0]
    except (AttributeError, IndexError, TypeError):
        return "unknown"


def _make_block_response(status_code: int, reason: str) -> Any:
    """Create a mitmproxy Response to block a request.

    Uses lazy import to avoid hard dependency on mitmproxy at module load.
    Falls back to a simple mock if mitmproxy is not installed.
    """
    try:
        from mitmproxy import http

        return http.Response.make(
            status_code,
            reason.encode("utf-8"),
            {"Content-Type": "text/plain", "X-Katana-Blocked": "true"},
        )
    except ImportError:
        # Fallback for environments without mitmproxy
        class _MockResponse:
            def __init__(self, code: int, body: str) -> None:
                self.status_code = code
                self.content = body.encode("utf-8")
                self.headers = {
                    "Content-Type": "text/plain",
                    "X-Katana-Blocked": "true",
                }

            def get_content(self) -> bytes:
                return self.content

            def set_content(self, data: bytes) -> None:
                self.content = data

        return _MockResponse(status_code, reason)

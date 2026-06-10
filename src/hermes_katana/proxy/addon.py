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

__all__ = [
    "KatanaAddon",
    "RateTracker",
]

import hashlib
import logging
import threading
import time
import zlib
from collections.abc import Iterable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from email import policy
from email.parser import BytesParser
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_katana.audit.trail import AuditTrail
    from hermes_katana.vault.store import Vault

from hermes_katana.proxy.config import ProxyConfig
from hermes_katana.proxy.injector import inject_credentials_with_metadata
from hermes_katana.security_logging import redact_text_for_log

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
    _windows: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque), repr=False)
    _violations: dict[str, int] = field(default_factory=lambda: defaultdict(int), repr=False)

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
                int(self.max_requests / (self.escalation_factor ** min(violations, 5))),
            )

            if len(window) >= effective_limit:
                self._violations[client_id] = violations + 1
                return False, len(window)

            window.append(now)
            # Decay violations over time
            if violations > 0 and len(window) < effective_limit // 2:
                self._violations[client_id] = max(0, violations - 1)

            # Evict stale clients when tracking too many.
            # Codex audit finding #6 (MED, 2026-05-07): the previous
            # implementation only removed clients whose window was already
            # empty. But ``check()`` only popleft's the *current* client's
            # window, so clients that hit the limit and never came back kept
            # stale-but-non-empty windows forever and slipped past eviction.
            # The fix: prune *every* tracked window's stale entries against
            # the time cutoff before deciding which to delete.
            if len(self._windows) > self._MAX_CLIENTS:
                for k in list(self._windows.keys()):
                    w = self._windows[k]
                    while w and w[0] < cutoff:
                        w.popleft()
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
        self._vault_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats: dict[str, int] = defaultdict(int)

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
        return any(host_lower == d or host_lower.endswith(f".{d}") for d in self.config.allowed_domains)

    def _is_explicitly_allowed(self, host: str) -> bool:
        """Whether *host* appears in the configured allowlist (operator intent)."""
        host_lower = host.lower()
        return any(host_lower == d or host_lower.endswith(f".{d}") for d in getattr(self.config, "allowed_domains", []))

    @staticmethod
    def _is_private_destination(host: str) -> bool:
        """Best-effort check for SSRF-style destinations (audit finding C4).

        Catches IP-literal loopback/RFC1918/link-local/reserved targets and
        well-known metadata/localhost hostnames. Hostnames that *resolve* to
        private addresses (DNS rebinding) are out of scope here — that needs
        connect-time enforcement — but the common localhost/metadata SSRF
        shapes are blocked by default.
        """
        candidate = (host or "").strip().lower().rstrip(".")
        if not candidate:
            return False
        if candidate.startswith("[") and candidate.endswith("]"):
            candidate = candidate[1:-1]
        try:
            import ipaddress

            ip = ipaddress.ip_address(candidate)
            return bool(ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_unspecified)
        except ValueError:
            pass
        if candidate in {
            "localhost",
            "metadata",
            "metadata.google.internal",
            "instance-data",
            "instance-data.ec2.internal",
        }:
            return True
        return candidate.endswith((".localhost", ".internal"))

    @staticmethod
    def _get_scannable_content(message: Any) -> tuple[Optional[bytes], bool]:
        """Fetch a message body for scanning.

        Returns ``(content, unavailable)``. ``unavailable=True`` means the
        body exists but is being streamed past the buffering cap
        (``stream_large_bodies``) and therefore cannot be scanned — chunked
        transfers carry no Content-Length, so the pre-buffer header gates
        never fire for them (audit finding C1). Callers must fail closed on
        it in strict/max instead of treating it like an absent body.
        """
        if getattr(message, "stream", False):
            return None, True
        try:
            return message.get_content(), False
        except AttributeError:
            return None, False
        except ValueError:
            # mitmproxy raises ValueError when content is unavailable for a
            # streamed message.
            return None, True

    def _body_too_large(self, body: Optional[bytes]) -> bool:
        """Check if a body exceeds the scan size limit."""
        if body is None:
            return False
        if self.config.max_body_scan_size == 0:
            return False  # Unlimited
        return len(body) > self.config.max_body_scan_size

    @staticmethod
    def _needs_binary_scan(data: bytes, content_type: str = "") -> bool:
        """Return true for file-like bodies that should not be treated as UTF-8 text."""
        lower_type = content_type.lower()
        if any(
            marker in lower_type
            for marker in (
                "application/pdf",
                "application/zip",
                "application/x-zip-compressed",
                "application/octet-stream",
                "openxmlformats-officedocument",
                "macroenabled",
                "vnd.ms-word",
                "vnd.ms-excel",
                "vnd.ms-powerpoint",
                "image/",
            )
        ):
            return True
        stripped = data.lstrip()
        if stripped.startswith((b"%PDF-", b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            return True
        if stripped.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"RIFF")):
            return True
        sample = data[:4096]
        if b"\x00" in sample:
            return True
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False

    @staticmethod
    def _decompress_with_limit(data: bytes, wbits: int, max_output_size: int) -> tuple[bytes, bool]:
        """Decompress concatenated streams without producing more than max_output_size bytes."""
        decoded = bytearray()
        remaining_input = data

        while remaining_input:
            decompressor = zlib.decompressobj(wbits)
            if max_output_size <= 0:
                decoded.extend(decompressor.decompress(remaining_input))
                decoded.extend(decompressor.flush())
            else:
                allowance = max_output_size + 1 - len(decoded)
                if allowance <= 0:
                    return bytes(decoded[:max_output_size]), True
                decoded.extend(decompressor.decompress(remaining_input, allowance))
                if decompressor.unconsumed_tail:
                    return bytes(decoded[:max_output_size]), True
                allowance = max_output_size + 1 - len(decoded)
                if allowance > 0:
                    decoded.extend(decompressor.flush(allowance))
                if len(decoded) > max_output_size:
                    return bytes(decoded[:max_output_size]), True

            if not decompressor.eof:
                raise zlib.error("incomplete compressed stream")
            if not decompressor.unused_data:
                break
            remaining_input = decompressor.unused_data

        if max_output_size > 0 and len(decoded) > max_output_size:
            return bytes(decoded[:max_output_size]), True
        return bytes(decoded), False

    @classmethod
    def _decode_content_encoding(
        cls,
        data: bytes,
        content_encoding: str,
        *,
        max_output_size: int = 0,
    ) -> tuple[bytes, bool]:
        """Decode supported HTTP content encodings before scanning.

        Returns the decoded bytes plus a flag indicating whether decoding was
        truncated at max_output_size. This prevents gzip/deflate bodies from
        expanding unboundedly before the proxy scan cap is applied.
        """
        decoded = data
        oversized = False
        encodings = [entry.strip().lower() for entry in content_encoding.split(",") if entry.strip()]
        for encoding in reversed(encodings):
            if encoding == "identity":
                continue
            if encoding == "gzip":
                decoded, oversized = cls._decompress_with_limit(
                    decoded,
                    16 + zlib.MAX_WBITS,
                    max_output_size,
                )
                if oversized:
                    break
                continue
            if encoding == "deflate":
                try:
                    decoded, oversized = cls._decompress_with_limit(decoded, zlib.MAX_WBITS, max_output_size)
                except zlib.error:
                    decoded, oversized = cls._decompress_with_limit(decoded, -zlib.MAX_WBITS, max_output_size)
                if oversized:
                    break
                continue
            raise ValueError(f"Unsupported content encoding: {encoding}")
        return decoded, oversized

    def _collect_scan_vault_values(self, *, extra_values: Iterable[str] = ()) -> Optional[set[str]]:
        """Collect vault secrets for the current scan without keeping a long-lived plaintext cache."""
        values = {value for value in extra_values if value}
        if self.vault is None:
            return values or None
        try:
            with self._vault_lock:
                all_vals = self.vault._get_all_values()
        except Exception:
            logger.debug("Could not collect vault values for scanning")
            return values or None
        if all_vals:
            values.update(value for value in all_vals.values() if value)
        return values or None

    def _decode_body_for_scan(self, body: bytes, headers: Any) -> tuple[bytes, str, bool]:
        """Decode supported encodings and apply the scan-size cap to the decoded payload."""
        try:
            content_type = str(headers.get("content-type", ""))
            content_encoding = str(headers.get("content-encoding", ""))
        except AttributeError:
            content_type = ""
            content_encoding = ""

        decoded_body, decoded_oversized = self._decode_content_encoding(
            body,
            content_encoding,
            max_output_size=self.config.max_body_scan_size,
        )
        if self.config.max_body_scan_size and len(decoded_body) > self.config.max_body_scan_size:
            decoded_body = decoded_body[: self.config.max_body_scan_size]
            decoded_oversized = True
        return decoded_body, content_type, decoded_oversized

    def _scan_multipart_body(
        self,
        body: bytes,
        *,
        content_type: str,
        direction: str,
        vault_values: Optional[set[str]] = None,
    ) -> Optional[dict[str, Any]]:
        """Scan each non-container part of a multipart payload."""
        if "multipart/" not in content_type.lower():
            return None

        parser_input = b"MIME-Version: 1.0\nContent-Type: " + content_type.encode("utf-8") + b"\n\n" + body
        message = BytesParser(policy=policy.default).parsebytes(parser_input)
        part_summaries: list[str] = []
        total_findings = 0
        max_risk_score = 0.0
        saw_part = False

        for part in message.walk():
            if part.is_multipart():
                continue

            raw_payload = part.get_payload(decode=True)
            if isinstance(raw_payload, bytearray):
                payload = bytes(raw_payload)
            elif isinstance(raw_payload, bytes):
                payload = raw_payload
            elif raw_payload is None:
                payload = b""
            else:
                payload = str(raw_payload).encode("utf-8", errors="replace")

            if not payload:
                continue
            saw_part = True
            part_content_type = part.get_content_type() or "application/octet-stream"
            if self._needs_binary_scan(payload, part_content_type):
                part_result = self._scan_bytes(
                    payload,
                    direction=direction,
                    content_type=part_content_type,
                    vault_values=vault_values,
                )
            else:
                part_text = payload.decode("utf-8", errors="replace")
                part_result = self._scan_text(part_text, direction=direction, vault_values=vault_values)

            max_risk_score = max(max_risk_score, float(part_result["risk_score"]))
            total_findings += int(part_result["finding_count"])
            if part_result["summary"]:
                part_summaries.append(str(part_result["summary"]))
            if part_result["is_blocked"]:
                return part_result

        if not saw_part:
            return None

        summary = "; ".join(part_summaries[:3])
        return {
            "verdict": "warn" if total_findings > 0 else "pass",
            "risk_score": max_risk_score,
            "is_blocked": False,
            "finding_count": total_findings,
            "summary": summary,
        }

    def _scan_text(
        self,
        text: str,
        direction: str = "request",
        *,
        vault_values: Optional[set[str]] = None,
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

        if direction == "request":
            result = scan_input(
                text,
                vault_values=vault_values,
                check_injection=modes.injection,
                check_secrets=modes.secrets,
                check_unicode=modes.unicode,
                check_content=modes.content,
                security_level=self.config.scanner_security_level,
            )
        else:
            # Response scanning includes indirect injection detection
            result = scan_output(
                text,
                vault_values=vault_values,
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

    def _scan_bytes(
        self,
        data: bytes,
        direction: str = "request",
        content_type: str = "",
        *,
        vault_values: Optional[set[str]] = None,
    ) -> dict[str, Any]:
        """Run the scanner on raw request/response body bytes."""
        from hermes_katana.scanner import scan_bytes

        modes = self.config.scan_modes

        result = scan_bytes(
            data,
            content_type=content_type,
            direction="output" if direction == "response" else "input",
            vault_values=vault_values,
            check_injection=modes.injection,
            check_secrets=modes.secrets,
            check_unicode=modes.unicode,
            check_content=modes.content,
            check_content_harm=True,
            check_prompt_leak=True,
            security_level=self.config.scanner_security_level,
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

    @staticmethod
    def _safe_log_text(text: object, redaction_values: Optional[Iterable[str]] = None) -> str:
        """Return text safe for logs, audit details, and proxy block messages."""
        return redact_text_for_log(str(text), extra_values=redaction_values or ())

    # ------------------------------------------------------------------
    # mitmproxy hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _content_length(headers: Any) -> Optional[int]:
        """Best-effort parse of a Content-Length header. Returns None on miss."""
        if headers is None:
            return None
        try:
            raw = headers.get("Content-Length") or headers.get("content-length")
        except AttributeError:
            return None
        if raw is None:
            return None
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            return None

    def requestheaders(self, flow: Any) -> None:
        """Pre-buffer size gate for requests.

        Closes Codex audit finding #1 (HIGH): without this hook, mitmproxy
        fully buffers the body before ``request()`` runs, so a hostile peer
        could allocate ``max_request_body_size`` worth of memory before
        Katana ever gets to reject. By rejecting on ``Content-Length`` here,
        we never buffer oversized bodies in the first place.
        """
        try:
            content_length = self._content_length(flow.request.headers)
        except AttributeError:
            return
        limit = getattr(self.config, "max_request_body_size", 0) or 0
        if limit > 0 and content_length is not None and content_length > limit:
            try:
                host = flow.request.host
            except AttributeError:
                host = "unknown"
            self._increment_stat("requests_blocked_pre_buffer")
            logger.warning(
                "Request body Content-Length=%d exceeds max_request_body_size=%d for %s; rejecting before buffer",
                content_length,
                limit,
                host,
            )
            flow.response = _make_block_response(
                413,
                f"Request body exceeds proxy limit ({content_length} > {limit} bytes).",
            )
            self._log_audit(
                "SCAN_RESULT",
                f"requestheaders_size:{host}",
                "deny",
                f"Content-Length {content_length} exceeds max_request_body_size {limit} (pre-buffer)",
            )

    def responseheaders(self, flow: Any) -> None:
        """Pre-buffer size gate for responses (Codex audit #1, HIGH).

        Same rationale as requestheaders, but for responses returning to the
        agent. Replaces an oversized response with a 502 before the body is
        buffered.
        """
        try:
            content_length = self._content_length(flow.response.headers)
            host = flow.request.host
        except AttributeError:
            return
        limit = getattr(self.config, "max_response_body_size", 0) or 0
        if limit > 0 and content_length is not None and content_length > limit:
            self._increment_stat("responses_blocked_pre_buffer")
            logger.warning(
                "Response Content-Length=%d exceeds max_response_body_size=%d from %s; rejecting before buffer",
                content_length,
                limit,
                host,
            )
            try:
                flow.response.set_content(
                    f"[HermesKatana] Response exceeds proxy limit "
                    f"({content_length} > {limit} bytes); rejected before buffer.".encode()
                )
                flow.response.status_code = 502
                if self.config.add_scanned_header:
                    flow.response.headers["X-Katana-Scanned"] = "blocked"
            except AttributeError:
                flow.response = _make_block_response(
                    502,
                    f"Response exceeds proxy limit ({content_length} > {limit} bytes).",
                )
            self._log_audit(
                "SCAN_RESULT",
                f"responseheaders_size:{host}",
                "deny",
                f"Content-Length {content_length} exceeds max_response_body_size {limit} (pre-buffer)",
            )

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

        # Private/SSRF destination gate (audit finding C4): with the default
        # empty allowlist, requests to loopback/RFC1918/link-local/metadata
        # would otherwise be forwarded — with credentials injected. Explicit
        # allowed_domains entries and the allow_private_destinations opt-out
        # express operator intent.
        if (
            not getattr(self.config, "allow_private_destinations", False)
            and self._is_private_destination(host)
            and not self._is_explicitly_allowed(host)
        ):
            self._increment_stat("requests_blocked_private_destination")
            logger.warning("Blocked request to private/SSRF destination: %s", host)
            flow.response = _make_block_response(
                403,
                f"Private/internal destination blocked: {host}. "
                "Set allow_private_destinations=true or add the host to allowed_domains to permit.",
            )
            self._log_audit(
                "POLICY_DECISION",
                f"private_destination_block:{host}",
                "deny",
                f"Destination {host} is loopback/private/link-local/metadata",
            )
            return

        # Domain allowlist check
        if not self._is_allowed_domain(host):
            self._increment_stat("requests_blocked_domain")
            logger.warning("Blocked request to non-allowed domain: %s", host)
            flow.response = _make_block_response(403, f"Domain not in allowlist: {host}")
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
            flow.response = _make_block_response(429, "Rate limit exceeded")
            self._log_audit(
                "RATE_ANOMALY",
                f"rate_limit:{client_id}",
                "deny",
                f"Rate limit exceeded: {count} requests",
            )
            return

        # Credential injection
        pre_injection_headers = _normalise_header_values(flow.request.headers)
        injected_secret_values: set[str] = set()
        if self.config.inject_credentials and self.vault is not None:
            if not self.config.tls_verify:
                self._increment_stat("requests_blocked_insecure_tls")
                flow.response = _make_block_response(
                    502,
                    "Proxy credential injection requires tls_verify=true.",
                )
                self._log_audit(
                    "SCAN_RESULT",
                    f"request_tls:{host}",
                    "deny",
                    "Credential injection refused because tls_verify is disabled",
                )
                return
            with self._vault_lock:
                injection = inject_credentials_with_metadata(flow, self.vault)
            if injection:
                self._increment_stat("credentials_injected")
                injected_secret_values.add(injection.secret_value)
                logger.debug("Injected provider auth header")

        # --- Scan request URL, headers, query params, cookies (GAP 3.1) ---
        # Collect only headers actually added/changed by the injector so
        # user-supplied Authorization values remain in scope for scanning.
        _injected_headers: set[str] = set()
        if self.config.inject_credentials and self.vault is not None:
            post_injection_headers = _normalise_header_values(flow.request.headers)
            credential_header_names = {"authorization", "api-key", "x-api-key", "x-goog-api-key"}
            _injected_headers = {
                name
                for name in credential_header_names
                if name in post_injection_headers
                and post_injection_headers.get(name) != pre_injection_headers.get(name)
            }
        self._last_injected_headers = set(_injected_headers)
        scan_vault_values = self._collect_scan_vault_values(extra_values=injected_secret_values)

        # Scan URL path segments
        try:
            url_text = flow.request.url
            if url_text:
                url_result = self._scan_text(url_text, direction="request", vault_values=scan_vault_values)
                if url_result["is_blocked"]:
                    self._increment_stat("requests_blocked_scan")
                    safe_summary = self._safe_log_text(url_result["summary"], scan_vault_values)
                    flow.response = _make_block_response(400, f"URL blocked: {safe_summary}")
                    self._log_audit(
                        "SCAN_RESULT",
                        f"url_scan:{host}",
                        "deny",
                        url_result["summary"],
                        redaction_values=scan_vault_values,
                    )
                    return
        except Exception as exc:
            self._fail_request_scan(flow, host, "url", exc)
            return

        # Scan request headers (skip injected ones — GAP 3.8)
        try:
            for hdr_name, hdr_value in flow.request.headers.items():
                if hdr_name.lower() in _injected_headers:
                    continue
                hdr_result = self._scan_text(hdr_value, direction="request", vault_values=scan_vault_values)
                if hdr_result["is_blocked"]:
                    self._increment_stat("requests_blocked_scan")
                    safe_summary = self._safe_log_text(hdr_result["summary"], scan_vault_values)
                    flow.response = _make_block_response(400, f"Header blocked: {safe_summary}")
                    self._log_audit(
                        "SCAN_RESULT",
                        f"header_scan:{host}:{hdr_name}",
                        "deny",
                        hdr_result["summary"],
                        redaction_values=scan_vault_values,
                    )
                    return
        except Exception as exc:
            self._fail_request_scan(flow, host, "header", exc)
            return

        # Scan query parameters
        try:
            if hasattr(flow.request, "query") and flow.request.query:
                for qname, qvalue in flow.request.query.items():
                    q_result = self._scan_text(qvalue, direction="request", vault_values=scan_vault_values)
                    if q_result["is_blocked"]:
                        self._increment_stat("requests_blocked_scan")
                        safe_summary = self._safe_log_text(q_result["summary"], scan_vault_values)
                        flow.response = _make_block_response(400, f"Query param blocked: {safe_summary}")
                        self._log_audit(
                            "SCAN_RESULT",
                            f"query_scan:{host}:{qname}",
                            "deny",
                            q_result["summary"],
                            redaction_values=scan_vault_values,
                        )
                        return
        except Exception as exc:
            self._fail_request_scan(flow, host, "query", exc)
            return

        # Scan cookie values
        try:
            cookie_header = flow.request.headers.get("cookie", "")
            if cookie_header:
                for part in cookie_header.split(";"):
                    if "=" in part:
                        _, cval = part.split("=", 1)
                        c_result = self._scan_text(cval.strip(), direction="request", vault_values=scan_vault_values)
                        if c_result["is_blocked"]:
                            self._increment_stat("requests_blocked_scan")
                            safe_summary = self._safe_log_text(c_result["summary"], scan_vault_values)
                            flow.response = _make_block_response(400, f"Cookie blocked: {safe_summary}")
                            self._log_audit(
                                "SCAN_RESULT",
                                f"cookie_scan:{host}",
                                "deny",
                                c_result["summary"],
                                redaction_values=scan_vault_values,
                            )
                            return
        except Exception as exc:
            self._fail_request_scan(flow, host, "cookie", exc)
            return

        # Request body scanning (GAP 3.5 — scan prefix of oversized bodies)
        body, body_unavailable = self._get_scannable_content(flow.request)
        if body_unavailable:
            # Chunked/oversized body streamed past the buffering cap: there
            # is a body, we just cannot see it. Same trust decision as an
            # oversized declared body (audit finding C1).
            if self._fail_closed_active():
                self._increment_stat("requests_blocked_streamed")
                flow.response = _make_block_response(
                    413,
                    "Request body is streamed (chunked/oversized) and cannot be scanned.",
                )
                self._log_audit(
                    "SCAN_RESULT",
                    f"request_scan:{host}",
                    "deny",
                    "Streamed request body (no Content-Length within cap); unscannable, blocked",
                )
                return
            self._increment_stat("requests_allowed_streamed")
            self._log_audit(
                "SCAN_RESULT",
                f"request_scan:{host}",
                "allow",
                "Streamed request body forwarded unscanned (permissive mode)",
            )
        if body:
            scan_body = body
            oversized = self._body_too_large(body)
            if oversized:
                oversize_action = (
                    "scanning first %d bytes then blocking"
                    if self._fail_closed_active()
                    else "scanning first %d bytes; unscanned tail may pass in permissive mode"
                )
                logger.warning(
                    "Oversized request body (%d bytes) from %s to %s — " + oversize_action,
                    len(body),
                    client_id,
                    host,
                    self.config.max_body_scan_size,
                )
                scan_body = body[: self.config.max_body_scan_size]
                self._increment_stat("requests_oversized")
            try:
                content_type = flow.request.headers.get("content-type", "")
                if not oversized:
                    scan_body, content_type, decoded_oversized = self._decode_body_for_scan(
                        scan_body, flow.request.headers
                    )
                    if decoded_oversized:
                        oversized = True
                        self._increment_stat("requests_oversized")
                multipart_result = self._scan_multipart_body(
                    scan_body,
                    content_type=content_type,
                    direction="request",
                    vault_values=scan_vault_values,
                )
                if multipart_result is not None:
                    scan_result = multipart_result
                elif self._needs_binary_scan(scan_body, content_type):
                    scan_result = self._scan_bytes(
                        scan_body,
                        direction="request",
                        content_type=content_type,
                        vault_values=scan_vault_values,
                    )
                else:
                    text = scan_body.decode("utf-8", errors="replace")
                    scan_result = self._scan_text(text, direction="request", vault_values=scan_vault_values)

                if scan_result["is_blocked"]:
                    self._increment_stat("requests_blocked_scan")
                    safe_summary = self._safe_log_text(scan_result["summary"], scan_vault_values)
                    logger.warning(
                        "Request blocked by scan: %s -> %s",
                        host,
                        safe_summary,
                    )
                    flow.response = _make_block_response(
                        400,
                        "Request blocked by security policy.",
                    )
                    self._log_audit(
                        "SCAN_RESULT",
                        f"request_scan:{host}",
                        "deny",
                        scan_result["summary"],
                        redaction_values=scan_vault_values,
                    )
                    return
                elif scan_result["finding_count"] > 0:
                    self._increment_stat("requests_warned")
                    self._log_audit(
                        "SCAN_RESULT",
                        f"request_scan:{host}",
                        "warn",
                        scan_result["summary"],
                        redaction_values=scan_vault_values,
                    )

                if oversized:
                    if not self._fail_closed_active():
                        self._increment_stat("requests_allowed_oversized")
                        self._log_audit(
                            "SCAN_RESULT",
                            f"request_scan:{host}",
                            "allow",
                            "Request body exceeds scan limit; unscanned tail allowed (permissive mode)",
                        )
                        self._increment_stat("requests_passed")
                    else:
                        self._increment_stat("requests_blocked_oversized")
                        flow.response = _make_block_response(
                            413,
                            "Request body exceeds scan limit and cannot be safely forwarded unscanned.",
                        )
                        self._log_audit(
                            "SCAN_RESULT",
                            f"request_scan:{host}",
                            "deny",
                            "Request body exceeds scan limit; unscanned tail blocked",
                        )
                    return
            except Exception as exc:
                self._fail_request_scan(flow, host, "body", exc)
                return

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

        scan_vault_values = self._collect_scan_vault_values()

        # Scan response body
        scanned = False
        oversized_response = False  # tracked for Codex audit #2: differentiate
        # X-Katana-Scanned: true vs partial.
        body, body_unavailable = self._get_scannable_content(flow.response)
        if body_unavailable:
            # Response is streaming past the buffering cap (no Content-Length
            # within limits) and cannot be scanned (audit finding C1).
            if self._fail_closed_active():
                self._increment_stat("responses_blocked_streamed")
                try:
                    flow.response.stream = False
                except Exception:  # noqa: BLE001
                    pass
                try:
                    flow.response.set_content(
                        b"[HermesKatana] Response is streamed (chunked/oversized) and cannot be scanned; blocked."
                    )
                    flow.response.status_code = 502
                except AttributeError:
                    flow.response = _make_block_response(502, "Streamed response cannot be scanned.")
                self._log_audit(
                    "SCAN_RESULT",
                    f"response_scan:{host}",
                    "deny",
                    "Streamed response body (no Content-Length within cap); unscannable, blocked",
                )
                return
            self._increment_stat("responses_allowed_streamed")
            self._log_audit(
                "SCAN_RESULT",
                f"response_scan:{host}",
                "allow",
                "Streamed response body forwarded unscanned (permissive mode)",
            )

        # Scan response headers (GAP 3.2)
        try:
            for hdr_name, hdr_value in flow.response.headers.items():
                rh_result = self._scan_text(hdr_value, direction="response", vault_values=scan_vault_values)
                if rh_result["is_blocked"]:
                    self._increment_stat("responses_blocked_scan")
                    safe_summary = self._safe_log_text(rh_result["summary"], scan_vault_values)
                    logger.warning("Response header blocked: %s -> %s", hdr_name, safe_summary)
                    flow.response.set_content(f"[HermesKatana] Response header blocked: {safe_summary}".encode())
                    flow.response.status_code = 502
                    self._log_audit(
                        "SCAN_RESULT",
                        f"response_header_scan:{host}:{hdr_name}",
                        "deny",
                        rh_result["summary"],
                        redaction_values=scan_vault_values,
                    )
                    return
        except Exception as exc:
            self._fail_response_scan(flow, host, "header", exc)
            return

        if body:
            scan_body = body
            oversized = self._body_too_large(body)
            if oversized:
                oversized_response = True
                oversize_action = (
                    "scanning first %d bytes then blocking"
                    if self._fail_closed_active()
                    else "scanning first %d bytes; unscanned tail may pass in permissive mode"
                )
                logger.warning(
                    "Oversized response body (%d bytes) from %s — " + oversize_action,
                    len(body),
                    host,
                    self.config.max_body_scan_size,
                )
                scan_body = body[: self.config.max_body_scan_size]
                self._increment_stat("responses_oversized")
            try:
                content_type = flow.response.headers.get("content-type", "")
                if not oversized:
                    scan_body, content_type, decoded_oversized = self._decode_body_for_scan(
                        scan_body, flow.response.headers
                    )
                    if decoded_oversized:
                        oversized = True
                        oversized_response = True
                        self._increment_stat("responses_oversized")
                multipart_result = self._scan_multipart_body(
                    scan_body,
                    content_type=content_type,
                    direction="response",
                    vault_values=scan_vault_values,
                )
                if multipart_result is not None:
                    scan_result = multipart_result
                elif self._needs_binary_scan(scan_body, content_type):
                    scan_result = self._scan_bytes(
                        scan_body,
                        direction="response",
                        content_type=content_type,
                        vault_values=scan_vault_values,
                    )
                else:
                    text = scan_body.decode("utf-8", errors="replace")
                    scan_result = self._scan_text(text, direction="response", vault_values=scan_vault_values)
                scanned = True

                if scan_result["is_blocked"]:
                    self._increment_stat("responses_blocked_scan")
                    safe_summary = self._safe_log_text(scan_result["summary"], scan_vault_values)
                    logger.warning(
                        "Response blocked by scan: %s -> %s",
                        host,
                        safe_summary,
                    )
                    # Replace response body with warning
                    flow.response.set_content(b"[HermesKatana] Response blocked by security policy.")
                    flow.response.status_code = 502
                    self._log_audit(
                        "SCAN_RESULT",
                        f"response_scan:{host}",
                        "deny",
                        scan_result["summary"],
                        redaction_values=scan_vault_values,
                    )
                    # Inject header and return early — do not count as passed
                    if self.config.add_scanned_header:
                        try:
                            flow.response.headers["X-Katana-Scanned"] = "true"
                        except AttributeError:
                            pass
                    return
                elif scan_result["finding_count"] > 0:
                    self._increment_stat("responses_warned")
                    self._log_audit(
                        "SCAN_RESULT",
                        f"response_scan:{host}",
                        "warn",
                        scan_result["summary"],
                        redaction_values=scan_vault_values,
                    )

                if oversized:
                    if not self._fail_closed_active():
                        self._increment_stat("responses_allowed_oversized")
                        self._log_audit(
                            "SCAN_RESULT",
                            f"response_scan:{host}",
                            "allow",
                            "Response body exceeds scan limit; unscanned tail allowed (permissive mode)",
                        )
                    else:
                        self._increment_stat("responses_blocked_oversized")
                        flow.response.set_content(
                            b"[HermesKatana] Response body exceeds scan limit; unscanned tail blocked."
                        )
                        flow.response.status_code = 502
                        self._log_audit(
                            "SCAN_RESULT",
                            f"response_scan:{host}",
                            "deny",
                            "Response body exceeds scan limit; unscanned tail blocked",
                        )
                        if self.config.add_scanned_header:
                            try:
                                flow.response.headers["X-Katana-Scanned"] = "blocked"
                            except AttributeError:
                                pass
                        return
            except Exception as exc:
                self._fail_response_scan(flow, host, "body", exc)
                return

        # Inject X-Katana-Scanned header (opt-in — GAP 3.7).
        # Codex audit finding #2 (HIGH): when scanning a partial body (oversized
        # in permissive mode), we used to mark the response as fully scanned.
        # Set "partial" instead so downstream consumers know the tail wasn't
        # examined.
        if self.config.add_scanned_header:
            if not scanned:
                value = "passthrough"
            elif oversized_response:
                value = "partial"
            else:
                value = "true"
            try:
                flow.response.headers["X-Katana-Scanned"] = value
            except AttributeError:
                pass

        self._increment_stat("responses_passed")

    def _fail_closed_active(self) -> bool:
        """Whether the proxy is configured to fail closed on scan errors."""
        return getattr(self.config, "mode", "strict") in ("strict", "max")

    def _fail_request_scan(self, flow: Any, host: str, scope: str, exc: Exception) -> None:
        """Block the request fail-closed in strict/max; log-and-allow in permissive."""
        self._increment_stat("requests_scan_errors")
        if not self._fail_closed_active():
            logger.warning(
                "Request %s scan failed for %s; allowing in permissive mode: %s",
                scope,
                host,
                self._safe_log_text(exc),
            )
            self._log_audit(
                "SCAN_RESULT",
                f"request_{scope}_scan:{host}",
                "allow",
                f"Scanner failure ({type(exc).__name__}); request allowed (permissive mode)",
            )
            return
        logger.warning(
            "Request %s scan failed for %s; blocking fail-closed: %s",
            scope,
            host,
            self._safe_log_text(exc),
        )
        flow.response = _make_block_response(
            502,
            "HermesKatana scanner failed; request blocked fail-closed.",
        )
        self._log_audit(
            "SCAN_RESULT",
            f"request_{scope}_scan:{host}",
            "deny",
            f"Scanner failure ({type(exc).__name__}); request blocked fail-closed",
        )

    def _fail_response_scan(self, flow: Any, host: str, scope: str, exc: Exception) -> None:
        """Block the response fail-closed in strict/max; log-and-allow in permissive."""
        self._increment_stat("responses_scan_errors")
        if not self._fail_closed_active():
            logger.warning(
                "Response %s scan failed for %s; allowing in permissive mode: %s",
                scope,
                host,
                self._safe_log_text(exc),
            )
            self._log_audit(
                "SCAN_RESULT",
                f"response_{scope}_scan:{host}",
                "allow",
                f"Scanner failure ({type(exc).__name__}); response allowed (permissive mode)",
            )
            return
        logger.warning(
            "Response %s scan failed for %s; blocking fail-closed: %s",
            scope,
            host,
            self._safe_log_text(exc),
        )
        try:
            flow.response.set_content(b"[HermesKatana] Scanner failed; response blocked fail-closed.")
            flow.response.status_code = 502
            if self.config.add_scanned_header:
                flow.response.headers["X-Katana-Scanned"] = "blocked"
        except AttributeError:
            flow.response = _make_block_response(
                502,
                "HermesKatana scanner failed; response blocked fail-closed.",
            )
        self._log_audit(
            "SCAN_RESULT",
            f"response_{scope}_scan:{host}",
            "deny",
            f"Scanner failure ({type(exc).__name__}); response blocked fail-closed",
        )

    def _log_audit(
        self,
        event_type: str,
        tool_name: str,
        decision: str,
        details: str,
        *,
        redaction_values: Optional[Iterable[str]] = None,
    ) -> None:
        """Log an event to the audit trail if available."""
        if self.audit is None:
            return
        try:
            from hermes_katana.audit.trail import AuditEntry, AuditEventType

            safe_details = self._safe_log_text(details, redaction_values)
            entry = AuditEntry(
                event_type=AuditEventType(event_type.lower())
                if hasattr(AuditEventType, event_type)
                else AuditEventType.SCAN_RESULT,
                tool_name=tool_name,
                args_hash=self._args_hash(tool_name, safe_details),
                decision=decision,
                details=safe_details,
            )
            self.audit.log(entry)
        except Exception as exc:
            logger.debug("Audit logging failed: %s", self._safe_log_text(exc))

    def websocket_message(self, flow: Any) -> None:
        """mitmproxy WebSocket message hook (GAP 3.3).

        Scans WebSocket message content through the same pipeline.

        Codex audit finding #7 (LOW, 2026-05-07): WebSocket scanning previously
        had no size cap, so a peer could send arbitrarily-large frames and
        force expensive decode + scanner work. We now enforce
        ``max_body_scan_size`` on raw bytes before decode; oversized frames are
        replaced with a block marker (or, in permissive mode, scanned in
        prefix-only mode).
        """
        try:
            msg = flow.websocket.messages[-1]
            content = msg.content
        except (AttributeError, IndexError):
            return

        self._increment_stat("ws_messages_total")
        scan_vault_values = self._collect_scan_vault_values()

        # Codex #7: pre-decode size gate.
        cap = getattr(self.config, "max_body_scan_size", 0) or 0
        if cap > 0 and isinstance(content, (bytes, bytearray)) and len(content) > cap:
            self._increment_stat("ws_messages_oversized")
            if self._fail_closed_active():
                logger.warning(
                    "WebSocket message %d bytes exceeds max_body_scan_size %d; blocking fail-closed",
                    len(content),
                    cap,
                )
                msg.content = b"[HermesKatana] WebSocket message exceeds size limit; blocked fail-closed."
                self._log_audit(
                    "SCAN_RESULT",
                    "websocket_size",
                    "deny",
                    f"WebSocket message {len(content)} bytes exceeds max_body_scan_size {cap} (fail-closed)",
                )
                return
            # Permissive mode: scan only the prefix.
            content = bytes(content[:cap])

        try:
            direction = "request" if getattr(msg, "from_client", True) else "response"
            if isinstance(content, bytes):
                if self._needs_binary_scan(content):
                    scan_result = self._scan_bytes(content, direction=direction, vault_values=scan_vault_values)
                else:
                    text = content.decode("utf-8", errors="replace")
                    scan_result = self._scan_text(text, direction=direction, vault_values=scan_vault_values)
            else:
                scan_result = self._scan_text(str(content), direction=direction, vault_values=scan_vault_values)

            if scan_result["is_blocked"]:
                self._increment_stat("ws_messages_blocked")
                safe_summary = self._safe_log_text(scan_result["summary"], scan_vault_values)
                logger.warning("WebSocket message blocked: %s", safe_summary)
                msg.content = b"[HermesKatana] Message blocked by security policy."
                self._log_audit(
                    "SCAN_RESULT",
                    "websocket_scan",
                    "deny",
                    scan_result["summary"],
                    redaction_values=scan_vault_values,
                )
            elif scan_result["finding_count"] > 0:
                self._increment_stat("ws_messages_warned")
                self._log_audit(
                    "SCAN_RESULT",
                    "websocket_scan",
                    "warn",
                    scan_result["summary"],
                    redaction_values=scan_vault_values,
                )
        except Exception as exc:
            self._increment_stat("ws_messages_scan_errors")
            if not self._fail_closed_active():
                logger.warning("WebSocket scan failed; allowing in permissive mode: %s", self._safe_log_text(exc))
                self._log_audit(
                    "SCAN_RESULT",
                    "websocket_scan",
                    "allow",
                    f"Scanner failure ({type(exc).__name__}); websocket message allowed (permissive mode)",
                )
                return
            logger.warning("WebSocket scan failed; blocking fail-closed: %s", self._safe_log_text(exc))
            self._increment_stat("ws_messages_blocked")
            msg.content = b"[HermesKatana] Scanner failed; WebSocket message blocked fail-closed."
            self._log_audit(
                "SCAN_RESULT",
                "websocket_scan",
                "deny",
                f"Scanner failure ({type(exc).__name__}); websocket message blocked fail-closed",
            )

    def get_stats(self) -> dict[str, Any]:
        """Return addon statistics."""
        with self._stats_lock:
            stats: dict[str, Any] = dict(self._stats)
        stats["rate_tracker"] = self._rate_tracker.get_stats()
        return stats


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def _normalise_header_values(headers: Any) -> dict[str, str]:
    """Return lowercase header names mapped to string values."""
    try:
        return {str(name).lower(): str(value) for name, value in headers.items()}
    except AttributeError:
        return {}


def _get_client_id(flow: Any) -> str:
    """Extract a client identifier from a flow."""
    try:
        return str(flow.client_conn.peername[0])
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
                """Return the response content."""
                return self.content

            def set_content(self, data: bytes) -> None:
                """Set the response content."""
                self.content = data

        return _MockResponse(status_code, reason)

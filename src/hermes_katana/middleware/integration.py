"""
Hermes integration middleware for HermesKatana.

Concrete middleware implementations that wire HermesKatana's subsystems
(taint tracking, scanning, policy engine, audit trail) into the Hermes
tool-dispatch pipeline via the :class:`MiddlewareChain`.

Middleware stack (default order, highest priority first)
--------------------------------------------------------
1. **KatanaTaintMiddleware** (pri=100) — wraps tool outputs in TaintedValues,
   checks taint flows before tool calls.
2. **KatanaScabbardMiddleware** (pri=90) — multi-signal prompt-injection
   classifier (normaliser + features + fusion).  On BLOCK: short-circuit.
   On FLAG: add taint, continue to downstream scanners.
2.5. **KatanaProtectAIMiddleware** (pri=88) — ProtectAI DeBERTa binary gate
   between the primary and secondary Scabbard passes.  INJECTION > block_threshold → DENY,
   INJECTION > flag_threshold → ESCALATE, SAFE > safe_passthrough → fast-path.
3. **KatanaScabbardSecondaryMiddleware** (pri=85) — second Scabbard classifier
   pass for prompt injection.  Runs alongside the primary Scabbard pass for defence-in-depth.
4. **KatanaScanMiddleware** (pri=80) — runs the multi-layer scanner on
   inputs and outputs to detect injections, secrets, and dangerous content.
4b. **KatanaMCPMiddleware** (pri=78) — MCP tool-poisoning detector: rug-pull
   drift, hidden instructions, unicode-tag steganography, suspicious schemas.
4c. **KatanaMultiTurnMiddleware** (pri=76) — stateful multi-turn attack
   detector: tracks conversation across turns, flags escalation / persona
   hijack / context manipulation sequences.
4d. **KatanaRAGInjectionMiddleware** (pri=74) — RAG indirect-injection
   detector: prompt injection in retrieved documents, role-hijack, context
   manipulation, tool hijack, poisoned embeddings, invisible characters.
5. **KatanaStructuralMiddleware** (pri=70) — content-type-aware structural
   analysis (HTML hidden text, PDF layers, Markdown injection, bloom filter).
5b. **KatanaBehavioralMiddleware** (pri=65) — post-dispatch behavioral
   observer for anomalous tool-call sequences.
6. **KatanaPolicyMiddleware** (pri=60) — evaluates the declarative policy
   engine to produce ALLOW / DENY / ESCALATE / LOG_ONLY decisions.
7. **KatanaAuditMiddleware** (pri=20) — logs every decision to the
   structured audit trail for post-incident analysis.

Usage::

    from hermes_katana.middleware.integration import create_default_chain

    chain = create_default_chain(config)
    ctx = chain.execute("terminal", {"command": "ls"}, taint_ctx)
"""

from __future__ import annotations

__all__ = [
    "KatanaTaintMiddleware",
    "KatanaScabbardMiddleware",
    "KatanaProtectAIMiddleware",
    "KatanaScabbardSecondaryMiddleware",
    "KatanaScanMiddleware",
    "KatanaMCPMiddleware",
    "KatanaMultiTurnMiddleware",
    "KatanaRAGInjectionMiddleware",
    "KatanaStructuralMiddleware",
    "KatanaBehavioralMiddleware",
    "KatanaPolicyMiddleware",
    "KatanaAuditMiddleware",
    "collect_chain_diagnostics",
    "create_default_chain",
]


import hashlib
import json
import logging
import time
from dataclasses import replace
from typing import Any

from hermes_katana.middleware.chain import CallContext, DispatchDecision, KatanaMiddleware, MiddlewareChain
from hermes_katana.middleware.protectai_middleware import KatanaProtectAIMiddleware
from hermes_katana.middleware.taint_middleware import KatanaTaintMiddleware

logger = logging.getLogger(__name__)


_CHAIN_PROFILE_ALIASES = {
    "default": "balanced",
    "production": "balanced",
    "prod": "balanced",
    "cpu": "fast_cpu",
    "fast": "fast_cpu",
    "fast-cpu": "fast_cpu",
    "balanced": "balanced",
    "max": "max",
}


def _normalize_chain_profile(profile: Any) -> str:
    """Normalize a user-facing deployment profile name."""
    normalized = str(profile or "balanced").strip().lower().replace("-", "_")
    normalized = _CHAIN_PROFILE_ALIASES.get(normalized, normalized)
    if normalized not in {"fast_cpu", "balanced", "max"}:
        raise ValueError("profile must be one of: fast_cpu, balanced, max")
    return normalized


def _profile_defaults(profile: str) -> dict[str, Any]:
    """Return explicit production-profile defaults for the middleware chain."""
    if profile == "fast_cpu":
        from hermes_katana.scabbard import ScabbardConfig

        # Timeout decision "escalate", not "allow": fast_cpu is an enforcing
        # profile, so a classifier timeout must surface through the escalation
        # policy instead of silently allowing the call (audit finding A6).
        scabbard_config = replace(
            ScabbardConfig.katana_v15_minilm(backend="onnx"),
            protectai_enabled=False,
            classifier_timeout_seconds=0.5,
            classifier_timeout_decision="escalate",
        )
        return {
            "scabbard.config": scabbard_config,
            "scabbard.enabled": True,
            "scabbard.route_mode": "balanced",
            "scabbard.scan_outputs": True,
            "scabbard.audit_routes": True,
            "scabbard.enforce_output_blocks": True,
            "scabbard.secondary.enabled": False,
            "protectai.enabled": False,
            "scan.enabled": True,
            "scan.route_aware": True,
            "behavioral.enabled": False,
            "policy.preset": "balanced",
        }

    if profile == "balanced":
        return {
            "scabbard.enabled": True,
            "scabbard.route_mode": "balanced",
            "scabbard.scan_outputs": True,
            "scabbard.audit_routes": True,
            "scabbard.enforce_output_blocks": True,
            "scabbard.secondary.enabled": False,
            "protectai.enabled": False,
            "scan.enabled": True,
            "scan.route_aware": True,
            "behavioral.enabled": True,
            "policy.preset": "balanced",
        }

    # max: keep overlapping ML gates enabled and fail closed on output findings.
    return {
        "scabbard.enabled": True,
        "scabbard.route_mode": "strict",
        "scabbard.scan_outputs": True,
        "scabbard.audit_routes": True,
        "scabbard.enforce_output_blocks": True,
        "scabbard.secondary.enabled": True,
        "protectai.enabled": True,
        "scan.enabled": True,
        "scan.route_aware": False,
        "scan.enforce_output_findings": True,
        "scan.block_threshold": 0.5,
        "scan.warn_threshold": 0.3,
        "policy.preset": "max",
    }


def _resolve_chain_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge explicit deployment profile defaults with caller overrides."""
    caller_config = dict(config or {})
    profile = _normalize_chain_profile(
        caller_config.get("profile", caller_config.get("katana.profile", caller_config.get("deployment.profile")))
    )
    resolved = _profile_defaults(profile)
    resolved.update(caller_config)
    resolved["profile"] = profile
    return resolved


def collect_chain_diagnostics(chain: Any) -> dict[str, Any]:
    """Collect a lightweight readiness report for a Katana middleware chain."""
    profile = getattr(chain, "active_profile", None) or getattr(chain, "profile", "unknown")
    active: list[str] = []
    inactive: list[str] = []
    degraded: list[str] = []
    ml: dict[str, Any] = {
        "scabbard_backend": None,
        "scabbard_device": None,
        "model_version": None,
    }

    middleware_by_name: dict[str, Any] = {}
    try:
        middleware = chain.list_middleware()
    except Exception:  # noqa: BLE001
        middleware = []
        degraded.append("middleware_chain")

    for mw in middleware:
        middleware_by_name[mw.name] = mw
        if mw.enabled:
            active.append(mw.name)
        else:
            inactive.append(mw.name)

    scabbard = middleware_by_name.get("katana.scabbard")
    scabbard_clf = getattr(scabbard, "_classifier", None)
    scabbard_cfg = getattr(scabbard, "_config", None)
    if scabbard_cfg is None and scabbard_clf is not None:
        scabbard_cfg = getattr(scabbard_clf, "config", None)
    if scabbard_cfg is not None:
        backend = getattr(scabbard_cfg, "katana_v11_backend", None)
        device = getattr(scabbard_cfg, "katana_v11_device", None)
        ml["scabbard_backend"] = backend
        ml["scabbard_device"] = "cpu" if backend in {"onnx", "onnx_int8"} and not device else device
        ml["model_version"] = getattr(scabbard_cfg, "model_version", None)

    # Scabbard readiness: enforcement on the rule-based fallback is a degraded
    # deployment and must be visible here, not only in per-call extras
    # (audit finding A1).
    ml["scabbard_profile"] = getattr(scabbard_cfg, "profile", None) if scabbard_cfg is not None else None
    trained_signal: Any = getattr(scabbard_clf, "trained_signal_active", None) if scabbard_clf is not None else None
    if trained_signal is None and scabbard is not None:
        try:
            from hermes_katana.scabbard import ScabbardConfig

            if scabbard_cfg is not None:
                trained_signal = bool(
                    getattr(scabbard_cfg, "katana_v11_path", None)
                    or getattr(scabbard_cfg, "deberta_model_cls", None)
                    or getattr(scabbard_cfg, "deberta_model", None)
                    or getattr(scabbard_cfg, "fusion_model", None)
                    or (
                        getattr(scabbard_cfg, "zvec_backbone_path", None)
                        and getattr(scabbard_cfg, "zvec_projector_path", None)
                    )
                )
            else:
                # Lazy default: mirrors what the middleware will resolve on
                # first classify (runtime_default), without loading models.
                profile = ScabbardConfig.default_runtime_profile()
                if ml["scabbard_profile"] is None:
                    ml["scabbard_profile"] = profile
                trained_signal = profile != "minimal"
        except Exception:  # noqa: BLE001
            trained_signal = None
    ml["scabbard_trained_signal"] = trained_signal
    if "katana.scabbard" in active and trained_signal is False:
        degraded.append("katana.scabbard")

    try:
        from hermes_katana.scanner import _OPTIONAL_IMPORT_ERRORS
    except Exception as exc:  # noqa: BLE001
        unavailable_optional = {"scanner_module": f"{exc.__class__.__name__}: {exc}"}
        degraded.append("katana.scan")
    else:
        unavailable_optional = {name: str(error) for name, error in _OPTIONAL_IMPORT_ERRORS.items()}
        if unavailable_optional and "katana.scan" in active:
            degraded.append("katana.scan")

    return {
        "active_profile": profile,
        "scanners": {
            "active": active,
            "inactive": inactive,
            "degraded": sorted(set(degraded)),
        },
        "ml": ml,
        "unavailable_optional_scanners": unavailable_optional,
    }


# ---------------------------------------------------------------------------
# 2. Scabbard middleware (multi-signal classifier)
# ---------------------------------------------------------------------------


class KatanaScabbardMiddleware(KatanaMiddleware):
    """Multi-signal prompt-injection classifier middleware.

    Runs the Scabbard pipeline (normaliser -> feature extraction -> fusion)
    on tool arguments **before** the pattern-based scanner.

    - **BLOCK**: short-circuit, do not run downstream scanners.
    - **FLAG**: add taint metadata and ``scabbard_result`` to *ctx.extras*,
      continue to downstream middleware.
    - **ALLOW**: continue normally.

    Args:
        config: Optional :class:`ScabbardConfig`.  Defaults to ``minimal``
            profile (zero ML deps).
        enabled: Whether this middleware is active.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        enabled: bool = True,
        route_mode: str = "balanced",
        scan_outputs: bool = True,
        audit_routes: bool = True,
        enforce_output_blocks: bool = True,
    ) -> None:
        super().__init__(name="katana.scabbard", enabled=enabled, priority=90)
        self._config = config
        self._classifier: Any | None = None
        self._route_mode = route_mode
        self._scan_outputs = scan_outputs
        self._audit_routes = audit_routes
        self._enforce_output_blocks = enforce_output_blocks

    @property
    def classifier(self) -> Any:
        """Lazy-load the ScabbardClassifier to avoid import cost at wire time.

        When no config was wired in, resolve the best locally ready runtime
        (production > standard > minimal) instead of pinning the rule-based
        minimal profile — a fresh checkout otherwise ran enforcement on rules
        alone without saying so (audit finding A1).
        """
        if self._classifier is None:
            from hermes_katana.scabbard import ScabbardClassifier, ScabbardConfig

            cfg = self._config or ScabbardConfig.runtime_default()
            self._classifier = ScabbardClassifier(cfg)
        return self._classifier

    @property
    def model_version(self) -> str:
        """Stable identifier for the loaded classifier (for audit + metrics)."""
        cfg = getattr(self.classifier, "config", None)
        if cfg and getattr(cfg, "model_version", None):
            return cfg.model_version
        return "scabbard-unknown"

    @property
    def shadow_classifier(self) -> Any | None:
        """Lazy-load the shadow KatanaV11Classifier if configured.

        Shadow classifiers run alongside the primary on every classification
        call. Their results are logged but do NOT affect actual decisions —
        used for canary-style rollouts of new model versions.
        """
        if hasattr(self, "_shadow_loaded"):
            return self._shadow
        cfg = getattr(self.classifier, "config", None)
        path = getattr(cfg, "shadow_v11_path", None) if cfg else None
        if not path:
            self._shadow = None
        else:
            from hermes_katana.scabbard.embedder import KatanaV11Classifier

            self._shadow = KatanaV11Classifier(
                model_path=path,
                backend=getattr(cfg, "shadow_v11_backend", "torch"),
                device=getattr(cfg, "shadow_v11_device", None),
                default_origin=getattr(cfg, "shadow_v11_default_origin", "user_input"),
            )
        self._shadow_loaded = True
        return self._shadow

    def _record_shadow(
        self,
        text: str,
        origin: str | None,
        primary_result: Any,
        ctx: CallContext,
    ) -> None:
        """If shadow is configured, classify and log disagreement vs primary."""
        shadow = self.shadow_classifier
        if shadow is None:
            return
        cfg = getattr(self.classifier, "config", None)
        shadow_version = getattr(cfg, "shadow_model_version", "shadow-unknown") if cfg else "shadow-unknown"
        try:
            shadow_result = shadow.classify_result(text, origin=origin)
        except Exception:  # noqa: BLE001
            import logging

            logging.getLogger("hermes_katana.middleware.shadow").exception(
                "shadow classifier raised; primary result unaffected"
            )
            return

        primary_dec = primary_result.decision
        shadow_dec = shadow_result.decision
        primary_top = primary_result.top_category
        shadow_top = shadow_result.top_category

        # Stash shadow info on ctx for downstream auditing / metrics.
        ctx.extras.setdefault("shadow_results", []).append(
            {
                "model_version": shadow_version,
                "decision": str(shadow_dec.value if hasattr(shadow_dec, "value") else shadow_dec),
                "top_category": shadow_top,
                "confidence": float(shadow_result.confidence),
            }
        )

        if primary_dec != shadow_dec or primary_top != shadow_top:
            import json as _json
            import logging

            payload = {
                "event": "shadow_disagreement",
                "primary_version": self.model_version,
                "shadow_version": shadow_version,
                "primary_decision": str(primary_dec.value if hasattr(primary_dec, "value") else primary_dec),
                "shadow_decision": str(shadow_dec.value if hasattr(shadow_dec, "value") else shadow_dec),
                "primary_top": primary_top,
                "shadow_top": shadow_top,
                "primary_confidence": round(float(primary_result.confidence), 4),
                "shadow_confidence": round(float(shadow_result.confidence), 4),
                "origin": origin,
            }
            logging.getLogger("hermes_katana.middleware.shadow").info(
                "%s", _json.dumps(payload, default=str, sort_keys=True)
            )

    def _classify_with_timeout(self, text: str, origin: str | None) -> Any:
        """Run the classifier; if it exceeds ``classifier_timeout_seconds``,
        return a synthesized result honoring ``classifier_timeout_decision``.

        timeout=0 (default) disables the wrapper entirely — same code path as
        before this hardening pass.
        """
        cfg = getattr(self.classifier, "config", None)
        timeout = float(getattr(cfg, "classifier_timeout_seconds", 0.0) or 0.0)
        if timeout <= 0.0:
            return self.classifier.classify(text, origin=origin)

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

        # Don't use ``with ... as ex`` — its __exit__ blocks for pending
        # tasks, defeating the timeout. shutdown(wait=False) lets the slow
        # call leak in the background while we return promptly.
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(self.classifier.classify, text, origin=origin)
        try:
            return fut.result(timeout=timeout)
        except FutureTimeout:
            from hermes_katana.scabbard.fusion import ClassificationResult, Decision

            fallback = (getattr(cfg, "classifier_timeout_decision", "allow") or "allow").lower()
            logger.warning(
                "Scabbard classifier exceeded %.2fs timeout; applying timeout decision %r",
                timeout,
                fallback,
            )
            if fallback == "deny":
                decision, confidence = Decision.BLOCK, 1.0
            elif fallback == "escalate":
                # Confidence 0.5 reaches the pre_dispatch FLAG/ESCALATE band.
                decision, confidence = Decision.FLAG, 0.5
            else:
                decision, confidence = Decision.ALLOW, 0.0
            return ClassificationResult(
                scores={"clean": 1.0 if decision == Decision.ALLOW else 0.0},
                decision=decision,
                top_category="timeout_fallback",
                confidence=confidence,
                degraded="classifier_timeout",
            )
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _resolve_origin(ctx: CallContext, arg_name: str) -> str | None:
        """Pull the origin tier for ``arg_name`` from ``ctx.taint_context``.

        Convention (forward-compatible with existing taint contracts):

        * ``ctx.taint_context["arg_origins"]`` — dict mapping arg-name to one
          of the 6 origin tiers. Per-arg granularity.
        * ``ctx.taint_context["origin"]`` — single tier applied to every arg
          when no per-arg map is set. Coarse fallback.
        * Unset → ``None``, which the classifier maps to its default tier
          (``user_input`` unless overridden in ScabbardConfig).
        """
        if not ctx.taint_context:
            return None
        per_arg = ctx.taint_context.get("arg_origins")
        if isinstance(per_arg, dict):
            tier = per_arg.get(arg_name)
            if isinstance(tier, str):
                return tier
        coarse = ctx.taint_context.get("origin")
        if isinstance(coarse, str):
            return coarse
        return None

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Run Scabbard only on routed natural-language/content arguments.

        Populates ``ctx.extras`` with:

        - ``scabbard_result``: worst :class:`ClassificationResult` dict
        - ``scabbard_results_by_arg``: per routed field result dicts
        - ``scabbard_routes`` / ``scabbard_skipped_args``: route audit data
        - ``scabbard_risk_score``: float (highest confidence across scanned args)
        - ``scabbard_arg_origins``: per-arg origin tier used (when v11 active)
        """
        from hermes_katana.scabbard.fusion import Decision as ScabbardDecision
        from hermes_katana.scabbard.routing import (
            extract_scabbard_arg_texts,
            has_scabbard_adversarial_signal,
            should_scabbard_scan_arg,
        )

        worst_confidence = 0.0
        worst_result: dict[str, Any] | None = None
        used_origins: dict[str, str] = {}
        routes: list[dict[str, Any]] = []
        skipped: dict[str, dict[str, Any]] = {}
        by_arg: dict[str, dict[str, Any]] = {}
        degraded_args: list[dict[str, Any]] = []
        scanned_count = 0
        skipped_count = 0

        for arg_name, arg_val in ctx.args.items():
            origin = self._resolve_origin(ctx, arg_name)
            route = should_scabbard_scan_arg(
                ctx.tool_name,
                arg_name,
                arg_val,
                origin=origin,
                mode=self._route_mode,
            )
            if self._audit_routes:
                routes.append(route.to_dict(arg=arg_name))
            if not route.scan:
                skipped_count += 1
                if self._audit_routes:
                    skipped[arg_name] = route.to_dict(arg=arg_name)

            for leaf_name, text, _leaf_route in extract_scabbard_arg_texts(
                ctx.tool_name,
                arg_name,
                arg_val,
                origin=origin,
                mode=self._route_mode,
            ):
                if not text.strip():
                    continue
                scanned_count += 1
                result = self._classify_with_timeout(text, origin)
                self._record_shadow(text, origin, result, ctx)
                if origin is not None:
                    used_origins[leaf_name] = origin
                result_dict = result.to_dict()
                by_arg[leaf_name] = result_dict
                degraded = getattr(result, "degraded", None)
                if degraded:
                    degraded_args.append({"arg": leaf_name, "reason": degraded})

                if result.decision == ScabbardDecision.BLOCK:
                    # Never soften a degraded verdict: a deny-on-timeout or
                    # fallback BLOCK is a fail-closed decision, not a
                    # low-confidence classification of short text.
                    softened = degraded is None and len(text.strip()) < 96 and not has_scabbard_adversarial_signal(text)
                    if softened:
                        ctx.extras.setdefault("scabbard_softened_blocks", []).append(
                            {
                                "arg": leaf_name,
                                "reason": "short_text_without_adversarial_signal",
                                "confidence": float(result.confidence),
                                "top_category": result.top_category,
                            }
                        )
                        continue
                    ctx.deny(f"Scabbard blocked ({result.top_category}, confidence={result.confidence:.2f})")
                    ctx.extras["scabbard_result"] = result_dict
                    ctx.extras["scabbard_results_by_arg"] = by_arg
                    if degraded_args:
                        ctx.extras["scabbard_degraded"] = degraded_args
                    ctx.extras["scabbard_risk_score"] = result.confidence
                    ctx.extras["scabbard_model_version"] = self.model_version
                    ctx.extras["scabbard_route_counts"] = {"scanned": scanned_count, "skipped": skipped_count}
                    if self._audit_routes:
                        ctx.extras["scabbard_routes"] = routes
                        ctx.extras["scabbard_skipped_args"] = skipped
                    if used_origins:
                        ctx.extras["scabbard_arg_origins"] = used_origins
                    return DispatchDecision.DENY

                if result.confidence > worst_confidence:
                    worst_confidence = result.confidence
                    worst_result = result_dict

        ctx.extras["scabbard_route_counts"] = {"scanned": scanned_count, "skipped": skipped_count}
        if self._audit_routes:
            ctx.extras["scabbard_routes"] = routes
            ctx.extras["scabbard_skipped_args"] = skipped
        if by_arg:
            ctx.extras["scabbard_results_by_arg"] = by_arg
        if worst_result is not None:
            ctx.extras["scabbard_result"] = worst_result
            ctx.extras["scabbard_risk_score"] = worst_confidence
            ctx.extras["scabbard_model_version"] = self.model_version
        if used_origins:
            ctx.extras["scabbard_arg_origins"] = used_origins

        if degraded_args:
            # The verdicts above came from a weaker path than configured
            # (missing/failed trained classifier, or a timeout fallback).
            # Strict mode fails closed on that instead of trusting them.
            ctx.extras["scabbard_degraded"] = degraded_args
            logger.warning(
                "Scabbard ran degraded for %s (%s)",
                ctx.tool_name,
                "; ".join(f"{d['arg']}: {d['reason']}" for d in degraded_args),
            )
            if self._route_mode == "strict":
                ctx.escalate(f"Scabbard degraded ({degraded_args[0]['reason']}); failing closed in strict mode")
                return DispatchDecision.ESCALATE

        if worst_confidence >= 0.5:
            # FLAG — add taint hint for downstream middleware
            ctx.extras["scabbard_flagged"] = True
            ctx.escalate(f"Scabbard flagged (confidence={worst_confidence:.2f})")
            return DispatchDecision.ESCALATE

        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Scan routed content fragments from tool outputs.

        Output scanning happens after the tool ran, so a BLOCK cannot prevent
        the side effect. It does prevent poisoned content from flowing onward by
        redacting ``ctx.tool_output`` by default while preserving route/result
        metadata for audit.
        """
        if not self._scan_outputs or ctx.tool_output is None:
            return

        from hermes_katana.scabbard.fusion import Decision as ScabbardDecision
        from hermes_katana.scabbard.routing import extract_scabbard_output_texts

        fragments = extract_scabbard_output_texts(ctx.tool_name, ctx.tool_output, mode=self._route_mode)
        if self._audit_routes:
            ctx.extras["scabbard_output_routes"] = [
                fragment.decision.to_dict(path=fragment.path) for fragment in fragments
            ]
        if not fragments:
            return

        worst_confidence = 0.0
        worst_result: dict[str, Any] | None = None
        by_path: dict[str, dict[str, Any]] = {}
        degraded_paths: list[dict[str, Any]] = []
        for fragment in fragments:
            result = self._classify_with_timeout(fragment.text, "tool_output")
            self._record_shadow(fragment.text, "tool_output", result, ctx)
            result_dict = result.to_dict()
            by_path[fragment.path] = result_dict
            degraded = getattr(result, "degraded", None)
            if degraded:
                degraded_paths.append({"path": fragment.path, "reason": degraded})
            if result.confidence > worst_confidence:
                worst_confidence = result.confidence
                worst_result = result_dict

        ctx.extras["scabbard_output_results_by_path"] = by_path
        if degraded_paths:
            ctx.extras["scabbard_output_degraded"] = degraded_paths
        if worst_result is not None:
            ctx.extras["scabbard_output_result"] = worst_result
            ctx.extras["scabbard_output_risk_score"] = worst_confidence
            ctx.extras["scabbard_model_version"] = self.model_version
            if worst_result.get("decision") == ScabbardDecision.BLOCK.value:
                ctx.extras["scabbard_output_blocked"] = True
                if self._enforce_output_blocks:
                    ctx.tool_output = "[Scabbard blocked tool output: adversarial content redacted]"
                    ctx.extras["scabbard_output_redacted"] = True
            elif worst_confidence >= 0.5:
                ctx.extras["scabbard_output_flagged"] = True


# ---------------------------------------------------------------------------
# 3. Secondary Scabbard middleware (ML classifier — runs between primary Scabbard and Scanner)
# ---------------------------------------------------------------------------


class KatanaScabbardSecondaryMiddleware(KatanaMiddleware):
    """Secondary Scabbard prompt-injection classifier middleware.

    Runs another Scabbard pipeline (normaliser -> feature extraction -> fusion)
    on tool arguments **before** the pattern-based scanner, as an overlapping
    ML signal alongside the primary Scabbard pass.

    - **BLOCK**: short-circuit, do not run downstream scanners.
    - **FLAG**: add ``scabbard_secondary_flagged`` hint to *ctx.extras*, continue.
    - **ALLOW**: continue normally.

    Args:
        config: Optional :class:`ScabbardConfig`.  Defaults to ``minimal``
            profile (zero ML deps).
        enabled: Whether this middleware is active.
    """

    def __init__(
        self,
        config: Any | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.scabbard_secondary", enabled=enabled, priority=85)
        self._config = config
        self._classifier: Any | None = None

    @property
    def classifier(self) -> Any:
        """Lazy-load the secondary Scabbard classifier to avoid import cost at wire time."""
        if self._classifier is None:
            from hermes_katana.scabbard import ScabbardClassifier, ScabbardConfig

            cfg = self._config or ScabbardConfig(profile="minimal")
            self._classifier = ScabbardClassifier(cfg)
        return self._classifier

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Run the secondary Scabbard pass on non-empty call arguments.

        Populates ``ctx.extras`` with:

        - ``scabbard_secondary_result``: :class:`ClassificationResult` dict
        - ``scabbard_secondary_risk_score``: float (highest confidence across args)
        """
        from hermes_katana.scabbard.fusion import Decision as ScabbardSecondaryDecision

        worst_confidence = 0.0
        worst_result: dict[str, Any] | None = None

        for arg_name, arg_val in ctx.args.items():
            text = str(arg_val) if arg_val is not None else ""
            if not text.strip():
                continue

            result = self.classifier.classify(text)
            if result.confidence > worst_confidence:
                worst_confidence = result.confidence
                worst_result = result.to_dict()

            if result.decision == ScabbardSecondaryDecision.BLOCK:
                ctx.deny(f"Secondary Scabbard blocked ({result.top_category}, confidence={result.confidence:.2f})")
                ctx.extras["scabbard_secondary_result"] = result.to_dict()
                ctx.extras["scabbard_secondary_risk_score"] = result.confidence
                return DispatchDecision.DENY

        if worst_result is not None:
            ctx.extras["scabbard_secondary_result"] = worst_result
            ctx.extras["scabbard_secondary_risk_score"] = worst_confidence

        if worst_confidence >= 0.5:
            ctx.extras["scabbard_secondary_flagged"] = True
            ctx.escalate(f"Secondary Scabbard flagged (confidence={worst_confidence:.2f})")
            return DispatchDecision.ESCALATE

        return DispatchDecision.ALLOW


# ---------------------------------------------------------------------------
# 4. Scanner middleware
# ---------------------------------------------------------------------------


class KatanaScanMiddleware(KatanaMiddleware):
    """Multi-layer scanner middleware for the Hermes dispatch pipeline.

    **Pre-dispatch**: runs the scanner on all string arguments to detect
    prompt injections, secrets, dangerous content, and Unicode attacks.

    **Post-dispatch**: scans tool outputs for content attacks and secret
    leakage.

    Args:
        vault_values:  Optional set of known secret values to detect leakage.
        block_threshold: Risk score threshold for blocking (default: 0.7).
        warn_threshold:  Risk score threshold for escalation (default: 0.4).
        enabled:        Whether this middleware is active.
    """

    def __init__(
        self,
        vault_values: set[str] | None = None,
        *,
        block_threshold: float = 0.7,
        warn_threshold: float = 0.4,
        check_injection: bool = True,
        check_secrets: bool = True,
        check_unicode: bool = True,
        check_content: bool = True,
        enforce_output_findings: bool = False,
        redact_output_secrets: bool = True,
        route_aware: bool = True,
        enabled: bool = True,
    ) -> None:
        """
        Codex audit finding #4 (MED, 2026-05-07): post-dispatch output scanning
        previously logged findings but did not enforce. Set
        ``enforce_output_findings=True`` to redact the tool output and stamp
        ``ctx.extras['output_redacted']=True`` whenever a finding fires. Default
        is False to preserve backward compatibility; production deployments
        that want fail-closed output scanning should opt in.

        Audit finding C3 (MED, 2026-06-09): secret-class findings are the
        exception — a credential in tool output is exfiltration in progress,
        so they redact by default (``redact_output_secrets=True``) even when
        general output enforcement is off.
        """
        super().__init__(name="katana.scan", enabled=enabled, priority=80)
        self._vault_values = vault_values or set()
        self._block_threshold = block_threshold
        self._warn_threshold = warn_threshold
        self._check_injection = check_injection
        self._check_secrets = check_secrets
        self._check_unicode = check_unicode
        self._check_content = check_content
        self._enforce_output_findings = enforce_output_findings
        self._redact_output_secrets = redact_output_secrets
        self._route_aware = route_aware

    @staticmethod
    def _scabbard_route_for_arg(ctx: CallContext, arg_name: str) -> Any | None:
        """Reuse the route decision already produced by KatanaScabbardMiddleware."""
        from hermes_katana.scabbard.routing import RouteKind, ScabbardRouteDecision

        route_rows: list[dict[str, Any]] = []
        routes = ctx.extras.get("scabbard_routes")
        if isinstance(routes, list):
            route_rows.extend(row for row in routes if isinstance(row, dict))
        skipped = ctx.extras.get("scabbard_skipped_args")
        if isinstance(skipped, dict):
            route_rows.extend(row for row in skipped.values() if isinstance(row, dict))

        for row in route_rows:
            if row.get("arg") != arg_name:
                continue
            kind_value = row.get("kind", RouteKind.UNKNOWN.value)
            try:
                kind = RouteKind(kind_value)
            except ValueError:
                kind = RouteKind.UNKNOWN
            return ScabbardRouteDecision(
                bool(row.get("scan", False)),
                str(row.get("reason", "scabbard_route_reused")),
                kind,
            )
        return None

    @staticmethod
    def _route_skipped_value_needs_scan(text: str) -> bool:
        """Detect high-risk carriers inside fields that are usually route-skipped."""
        import html
        import urllib.parse

        normalized = html.unescape(text)
        decoded = urllib.parse.unquote(normalized)
        blob = f"{normalized} {decoded}".lower()
        return any(
            marker in blob
            for marker in (
                "data:",
                "javascript:",
                "vbscript:",
                "file:",
                "<svg",
                "<script",
                "%pdf",
                "/js",
                "openaction",
                "base64,",
                "openxmlformats-officedocument",
                "macroenabled",
                "vnd.ms-word",
                "vnd.ms-excel",
                "vnd.ms-powerpoint",
                "ignore previous",
                "ignore all previous",
                "disregard previous",
            )
        )

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Scan all string arguments for attacks.

        Uses ``scan_input()`` for general text and ``scan_command()`` for
        command-type arguments.
        """
        from hermes_katana.scabbard.routing import RouteKind, should_scabbard_scan_arg
        from hermes_katana.scanner import scan_command, scan_input

        worst_score = 0.0
        all_results = []

        for arg_name, arg_val in ctx.args.items():
            text = str(arg_val) if arg_val is not None else ""
            if not text:
                continue

            route = self._scabbard_route_for_arg(ctx, arg_name)
            if route is None:
                route = should_scabbard_scan_arg(ctx.tool_name, arg_name, arg_val, mode="balanced")
            if self._route_aware and route.kind in {
                RouteKind.BOOLEAN,
                RouteKind.CONTROL,
                RouteKind.ENUM,
                RouteKind.NUMERIC,
                RouteKind.PATH,
                RouteKind.STRUCTURAL,
                RouteKind.URL,
                RouteKind.URL_LIST,
            }:
                if not self._route_skipped_value_needs_scan(text):
                    continue

            # Use command scanner for command-like arguments
            if route.kind == RouteKind.COMMAND or arg_name in ("command", "cmd", "shell_command", "script"):
                result = scan_command(
                    text,
                    check_secrets=self._check_secrets,
                    vault_values=self._vault_values,
                )
            else:
                result = scan_input(
                    text,
                    vault_values=self._vault_values,
                    check_injection=self._check_injection,
                    check_secrets=self._check_secrets,
                    check_unicode=self._check_unicode,
                    check_content=self._check_content,
                )

            all_results.append(result)
            worst_score = max(worst_score, result.risk_score)

            if result.has_findings:
                logger.debug(
                    "Scanner findings for %s.%s: %s",
                    ctx.tool_name,
                    arg_name,
                    result.summary,
                )

        ctx.scan_results = all_results
        ctx.extras["scan_risk_score"] = worst_score

        if worst_score >= self._block_threshold:
            findings_summary = "; ".join(r.summary for r in all_results if r.has_findings)
            ctx.deny(f"Scanner blocked: {findings_summary}")
            return DispatchDecision.DENY

        if worst_score >= self._warn_threshold:
            ctx.escalate(f"Scanner warning: risk_score={worst_score:.2f}")
            return DispatchDecision.ESCALATE

        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Scan tool output for content attacks and secret leakage."""
        if ctx.tool_output is None or ctx.is_denied:
            return

        try:
            from hermes_katana.scanner import scan_output

            output_text = str(ctx.tool_output)
            if output_text:
                result = scan_output(
                    output_text,
                    vault_values=self._vault_values,
                    check_injection=self._check_injection,
                    check_secrets=self._check_secrets,
                    check_unicode=self._check_unicode,
                    check_content=self._check_content,
                )
                ctx.extras["output_scan_result"] = result
                if result.has_findings:
                    logger.warning(
                        "Post-dispatch scan findings for %s: %s",
                        ctx.tool_name,
                        result.summary,
                    )
                    # Codex audit #4: enforce — replace output with a marker
                    # rather than letting downstream consumers see flagged
                    # content. Opt-in via enforce_output_findings — except
                    # secret-class findings, which redact by default (audit
                    # finding C3: log-only secret egress is exfiltration).
                    secret_leak = bool(getattr(result, "secret_findings", None)) and self._redact_output_secrets
                    if self._enforce_output_findings or secret_leak:
                        ctx.tool_output = (
                            f"[HermesKatana] Tool output redacted by post-dispatch scanner: {result.summary}"
                        )
                        ctx.extras["output_redacted"] = True
                        ctx.extras["output_redacted_reason"] = result.summary
                        if secret_leak:
                            ctx.extras["output_secret_redacted"] = True
        except Exception:
            logger.debug("Post-dispatch scan failed for %s", ctx.tool_name, exc_info=True)


# ---------------------------------------------------------------------------
# 4. Structural middleware
# ---------------------------------------------------------------------------


class KatanaStructuralMiddleware(KatanaMiddleware):
    """Content-type-aware structural analysis middleware.

    **Pre-dispatch**: detects the content type of string arguments and
    routes to specialised sub-scanners (html_diff, pdf_layers,
    markdown_audit) plus the bloom filter.  Produces a unified
    :class:`StructuralReport` attached to ``ctx.extras["structural_report"]``.

    Runs after the general-purpose scanner (pri=80) and before policy
    (pri=60) so that structural findings are available for policy
    evaluation.

    Args:
        block_threshold: Structural score above which the call is denied.
        warn_threshold:  Structural score above which the call is escalated.
        enabled:         Whether this middleware is active.
    """

    def __init__(
        self,
        *,
        block_threshold: float = 0.8,
        warn_threshold: float = 0.5,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.structural", enabled=enabled, priority=70)
        self._block_threshold = block_threshold
        self._warn_threshold = warn_threshold

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Run structural analysis on all string arguments."""
        from hermes_katana.scanner.structural import (
            ContentType,
            detect_structural,
        )

        worst_score = 0.0
        reports: list[dict] = []

        for arg_name, arg_val in ctx.args.items():
            text = str(arg_val) if arg_val is not None else ""
            if not text.strip():
                continue

            report = detect_structural(text)

            # Fast path: plain text with no findings — skip
            if report.content_type == ContentType.PLAIN.value and not report.flags:
                continue

            reports.append(report.to_dict())
            worst_score = max(worst_score, report.structural_score)

            if report.flags:
                logger.debug(
                    "Structural findings for %s.%s (%s): %d flags, score=%.2f",
                    ctx.tool_name,
                    arg_name,
                    report.content_type,
                    len(report.flags),
                    report.structural_score,
                )

        if reports:
            ctx.extras["structural_reports"] = reports
            ctx.extras["structural_score"] = worst_score

        if worst_score >= self._block_threshold:
            ctx.deny(f"Structural scanner blocked: score={worst_score:.2f}")
            return DispatchDecision.DENY

        if worst_score >= self._warn_threshold:
            ctx.escalate(f"Structural scanner warning: score={worst_score:.2f}")
            return DispatchDecision.ESCALATE

        return DispatchDecision.ALLOW


# ---------------------------------------------------------------------------
# 4b. Behavioral analysis middleware
# ---------------------------------------------------------------------------


class KatanaBehavioralMiddleware(KatanaMiddleware):
    """POST-dispatch behavioral analysis middleware.

    Detects output-side anomalies by maintaining a stateful
    :class:`~hermes_katana.scanner.behavioral.BehavioralTracker` across
    all tool calls in the chain lifecycle.

    Priority 65 — after structural (70), before policy (60).
    """

    def __init__(
        self,
        tracker: Any | None = None,
        *,
        block_on_sequence: bool = False,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.behavioral", enabled=enabled, priority=65)
        self._tracker = tracker
        self._block_on_sequence = block_on_sequence

    @property
    def tracker(self) -> Any:
        if self._tracker is None:
            try:
                from hermes_katana.scanner.behavioral import BehavioralTracker

                self._tracker = BehavioralTracker()
            except Exception:
                pass
        return self._tracker

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Scan args for persona-shift / conversation-drift signals (stateless)."""
        try:
            from hermes_katana.scanner.behavioral import (
                BehavioralSeverity,
                detect_behavioral,
            )
        except Exception:
            return DispatchDecision.ALLOW

        all_findings = []
        for arg_val in ctx.args.values():
            text = str(arg_val) if arg_val is not None else ""
            if text.strip():
                all_findings.extend(detect_behavioral(text))

        if all_findings:
            ctx.extras["behavioral_pre_findings"] = all_findings
            weights = {
                BehavioralSeverity.CRITICAL: 0.5,
                BehavioralSeverity.HIGH: 0.3,
                BehavioralSeverity.MEDIUM: 0.15,
                BehavioralSeverity.LOW: 0.05,
            }
            ctx.extras["behavioral_pre_risk"] = min(max(weights.get(f.severity, 0.05) for f in all_findings), 1.0)
        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Record completed tool call into the tracker; surface any anomalies."""
        if self.tracker is None:
            return
        output_str = str(ctx.tool_output) if ctx.tool_output is not None else None
        try:
            findings = self.tracker.record_tool_call(
                tool_name=ctx.tool_name,
                output=output_str,
                had_error=ctx.tool_error is not None,
                duration_ms=ctx.tool_duration_ms,
            )
        except Exception:
            logger.debug("BehavioralTracker.record_tool_call failed", exc_info=True)
            return

        if not findings:
            return

        ctx.extras["behavioral_post_findings"] = findings
        logger.warning(
            "Behavioral anomaly after %s: %s",
            ctx.tool_name,
            "; ".join(f.description[:80] for f in findings),
        )

        if self._block_on_sequence:
            from hermes_katana.scanner.behavioral import BehavioralCategory

            if any(f.category == BehavioralCategory.ANOMALOUS_SEQUENCE for f in findings):
                ctx.escalate(f"Behavioral: dangerous sequence after {ctx.tool_name}")


# ---------------------------------------------------------------------------
# 4b. MCP tool-poisoning detector middleware
# ---------------------------------------------------------------------------


class KatanaMCPMiddleware(KatanaMiddleware):
    """MCP (Model Context Protocol) tool-poisoning detector middleware.

    Scans tool-registration-shaped arguments for rug-pull drift, hidden
    instructions, unicode-tag steganography, and suspicious schemas.

    Baselines are stored in-process keyed by tool name so subsequent
    registrations of the same tool are checked for silent drift.
    """

    def __init__(
        self,
        *,
        block_on_critical: bool = True,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.mcp", enabled=enabled, priority=78)
        self._block_on_critical = block_on_critical
        self._baselines: dict[str, Any] = {}

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        from hermes_katana.scanner.mcp_scanner import (
            MCPSeverity,
            ToolBaseline,
            compute_tool_hash,
            scan_mcp_tool,
            scan_mcp_tools,
        )

        all_findings: list[Any] = []

        def _collect(val: Any) -> None:
            if isinstance(val, dict) and "name" in val and ("description" in val or "inputSchema" in val):
                name = str(val.get("name") or "")
                baseline = self._baselines.get(name)
                findings = scan_mcp_tool(val, baseline=baseline)
                all_findings.extend(findings)
                if baseline is None and name:
                    self._baselines[name] = ToolBaseline(name=name, hash=compute_tool_hash(val))
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                all_findings.extend(
                    scan_mcp_tools(
                        [v for v in val if isinstance(v, dict)],
                        baselines=self._baselines or None,
                    )
                )
                for v in val:
                    if isinstance(v, dict) and "name" in v:
                        nm = str(v.get("name") or "")
                        if nm and nm not in self._baselines:
                            self._baselines[nm] = ToolBaseline(name=nm, hash=compute_tool_hash(v))

        for arg_val in ctx.args.values():
            _collect(arg_val)

        if not all_findings:
            return DispatchDecision.ALLOW

        ctx.extras["mcp_findings"] = [
            {
                "category": f.category.value,
                "severity": f.severity.value,
                "tool_name": f.tool_name,
                "location": f.location,
                "evidence": f.evidence,
                "description": f.description,
            }
            for f in all_findings
        ]
        critical = [f for f in all_findings if f.severity == MCPSeverity.CRITICAL]
        if critical and self._block_on_critical:
            ctx.deny(
                f"MCP poisoning detected: {critical[0].category.value} in "
                f"{critical[0].tool_name} ({critical[0].location})"
            )
            return DispatchDecision.DENY

        ctx.escalate(f"MCP scanner flagged {len(all_findings)} finding(s) across tool registrations")
        return DispatchDecision.ESCALATE


# ---------------------------------------------------------------------------
# 4c. Multi-turn attack detector middleware
# ---------------------------------------------------------------------------


class KatanaMultiTurnMiddleware(KatanaMiddleware):
    """Stateful multi-turn attack detector middleware.

    Maintains one MultiTurnDetector per session. Every pre_dispatch that
    carries a user-turn argument (heuristically: any string arg named
    "message", "prompt", "user_input", "text", "query") is fed to the
    detector, then the detector's current assessment is consulted.
    """

    _USER_ARG_NAMES = ("message", "prompt", "user_input", "text", "query", "content")

    def __init__(
        self,
        *,
        block_threshold: float = 0.75,
        warn_threshold: float = 0.45,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.multiturn", enabled=enabled, priority=76)
        self._block_threshold = block_threshold
        self._warn_threshold = warn_threshold
        self._detectors: dict[str, Any] = {}

    def _detector_for(self, session_id: str) -> Any:
        from hermes_katana.scanner.multiturn import MultiTurnDetector

        det = self._detectors.get(session_id)
        if det is None:
            det = MultiTurnDetector()
            self._detectors[session_id] = det
        return det

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        from hermes_katana.scanner.multiturn import TurnRole

        session_id = ctx.extras.get("session_id") or ctx.taint_context.get("session_id") or "default"
        detector = self._detector_for(str(session_id))

        fed = False
        for arg_name, arg_val in ctx.args.items():
            if arg_name not in self._USER_ARG_NAMES:
                continue
            text = str(arg_val) if arg_val is not None else ""
            if not text.strip():
                continue
            detector.add_turn(TurnRole.USER, text)
            fed = True

        if not fed:
            return DispatchDecision.ALLOW

        assessment = detector.assess()
        ctx.extras["multiturn_overall_risk"] = assessment.overall_risk
        ctx.extras["multiturn_findings"] = [
            {
                "attack_type": f.attack_type,
                "severity": f.severity,
                "turn_indices": list(f.turn_indices),
                "description": f.description,
                "score": f.score,
            }
            for f in assessment.findings
        ]

        if assessment.overall_risk >= self._block_threshold:
            ctx.deny(f"Multi-turn attack detected (risk={assessment.overall_risk:.2f})")
            return DispatchDecision.DENY
        if assessment.overall_risk >= self._warn_threshold:
            ctx.escalate(f"Multi-turn risk elevated (risk={assessment.overall_risk:.2f})")
            return DispatchDecision.ESCALATE
        return DispatchDecision.ALLOW


# ---------------------------------------------------------------------------
# 4d. RAG indirect-injection detector middleware
# ---------------------------------------------------------------------------


class KatanaRAGInjectionMiddleware(KatanaMiddleware):
    """RAG indirect-injection detector middleware.

    Scans retrieved-document arguments for prompt injection, role-hijack,
    context manipulation, tool hijack, poisoned embeddings, invisible
    characters, source spoofing, and exfiltration primitives.
    """

    _DOC_ARG_NAMES = (
        "documents",
        "retrieved",
        "retrieved_documents",
        "context_docs",
        "rag_context",
        "chunks",
        "context",
    )

    def __init__(
        self,
        *,
        block_threshold: float = 0.90,
        warn_threshold: float = 0.60,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.rag_injection", enabled=enabled, priority=74)
        self._block_threshold = block_threshold
        self._warn_threshold = warn_threshold

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        from hermes_katana.scanner.rag_injection import (
            detect_rag_injection,
            rag_injection_risk_score,
            scan_retrieved_documents,
        )

        worst_score = 0.0
        finding_dicts: list[dict] = []

        for arg_name, arg_val in ctx.args.items():
            if arg_name not in self._DOC_ARG_NAMES:
                continue

            if isinstance(arg_val, str):
                score = rag_injection_risk_score(arg_val)
                worst_score = max(worst_score, score)
                for f in detect_rag_injection(arg_val):
                    finding_dicts.append(
                        {
                            "arg": arg_name,
                            "category": f.category.value,
                            "pattern": f.pattern_name,
                            "confidence": f.confidence,
                            "description": f.description,
                        }
                    )
            elif isinstance(arg_val, list):
                for idx, finding in scan_retrieved_documents(arg_val):
                    worst_score = max(worst_score, finding.confidence)
                    finding_dicts.append(
                        {
                            "arg": arg_name,
                            "doc_index": idx,
                            "category": finding.category.value,
                            "pattern": finding.pattern_name,
                            "confidence": finding.confidence,
                            "description": finding.description,
                        }
                    )

        if not finding_dicts:
            return DispatchDecision.ALLOW

        ctx.extras["rag_injection_findings"] = finding_dicts
        ctx.extras["rag_injection_score"] = worst_score

        if worst_score >= self._block_threshold:
            ctx.deny(f"RAG injection blocked: score={worst_score:.2f}")
            return DispatchDecision.DENY
        if worst_score >= self._warn_threshold:
            ctx.escalate(f"RAG injection warning: score={worst_score:.2f}")
            return DispatchDecision.ESCALATE
        return DispatchDecision.ALLOW


# ---------------------------------------------------------------------------
# 4. Policy middleware
# ---------------------------------------------------------------------------


class KatanaPolicyMiddleware(KatanaMiddleware):
    """Declarative policy evaluation middleware.

    Evaluates the :class:`PolicyEngine` against the current tool call
    and taint context.  Maps policy results to dispatch decisions:

    - ``PolicyResult.ALLOW``    → ``DispatchDecision.ALLOW``
    - ``PolicyResult.DENY``     → ``DispatchDecision.DENY``
    - ``PolicyResult.ESCALATE`` → ``DispatchDecision.ESCALATE``
    - ``PolicyResult.LOG_ONLY`` → ``DispatchDecision.ALLOW`` (with log)

    Args:
        engine:  The :class:`PolicyEngine` instance (or None for lazy init).
        preset:  Built-in preset name if ``engine`` is None (default: ``balanced``).
        enabled: Whether this middleware is active.
    """

    def __init__(
        self,
        engine: Any | None = None,
        *,
        preset: str = "balanced",
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.policy", enabled=enabled, priority=60)
        self._engine = engine
        self._preset = preset

    @property
    def engine(self) -> Any:
        """Lazy-load the policy engine."""
        if self._engine is None:
            from hermes_katana.policy import PolicyEngine

            self._engine = PolicyEngine.with_defaults(self._preset)
        return self._engine

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Evaluate policy engine for the current tool call.

        Uses the taint context built by upstream middleware (especially
        KatanaTaintMiddleware) for condition evaluation.
        """
        from hermes_katana.policy import PolicyResult

        result = self.engine.evaluate(
            tool_name=ctx.tool_name,
            args=ctx.args,
            taint_context=ctx.taint_context,
        )

        ctx.policy_result = result
        ctx.extras["policy_action"] = result.action.value
        ctx.extras["policy_reason"] = result.reason

        if result.matched_policy:
            ctx.extras["policy_name"] = result.matched_policy.name

        # Deny-by-default for unknown tools with tainted args
        if result.matched_policy is None and ctx.taint_context.get("tainted_fields"):
            ctx.deny(f"Unknown tool '{ctx.tool_name}' with tainted arguments — deny by default (no matching policy)")
            return DispatchDecision.DENY

        # Escalate unknown tools with clean args (fail-closed)
        if result.matched_policy is None:
            ctx.escalate(f"Unknown tool '{ctx.tool_name}' — escalate by default (no matching policy)")
            return DispatchDecision.ESCALATE

        if result.action == PolicyResult.DENY:
            ctx.deny(f"Policy denied: {result.reason}")
            return DispatchDecision.DENY

        if result.action == PolicyResult.ESCALATE:
            ctx.escalate(f"Policy escalation: {result.reason}")
            return DispatchDecision.ESCALATE

        if result.action == PolicyResult.LOG_ONLY:
            logger.info(
                "Policy LOG_ONLY for %s: %s",
                ctx.tool_name,
                result.reason,
            )

        return DispatchDecision.ALLOW


# ---------------------------------------------------------------------------
# 5. Audit middleware
# ---------------------------------------------------------------------------


class KatanaAuditMiddleware(KatanaMiddleware):
    """Audit trail middleware — logs every dispatch decision.

    Records structured audit events for both pre-dispatch decisions and
    post-dispatch results.  Integrates with the ``hermes_katana.audit``
    module when available, and falls back to Python logging otherwise.

    Args:
        audit_trail: Optional audit trail instance (lazy-loaded if None).
        log_allow:   Whether to log ALLOW decisions (default: True).
        enabled:     Whether this middleware is active.
    """

    def __init__(
        self,
        audit_trail: Any | None = None,
        *,
        log_allow: bool = True,
        enabled: bool = True,
    ) -> None:
        # Audit runs at lowest priority so it has full context from all upstream
        # middleware (scan, structural, policy) when logging ALLOW/ESCALATE.
        # Denied calls are still captured via on_short_circuit() regardless of order.
        super().__init__(name="katana.audit", enabled=enabled, priority=20)
        self._audit_trail = audit_trail
        self._log_allow = log_allow

    @property
    def audit_trail(self) -> Any | None:
        """Lazy-load the audit trail if available."""
        if self._audit_trail is None:
            try:
                from hermes_katana.audit import AuditTrail

                self._audit_trail = AuditTrail()
            except ImportError:
                # Audit module not yet available — will use logging fallback
                pass
        return self._audit_trail

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Log the pre-dispatch decision.

        This middleware never blocks calls — it only observes and records.
        """
        # Don't log allowed calls if configured to skip them
        if ctx.decision == DispatchDecision.ALLOW and not self._log_allow:
            return DispatchDecision.ALLOW

        event = {
            "type": "tool_dispatch",
            "phase": "pre",
            "call_id": ctx.call_id,
            "tool_name": ctx.tool_name,
            "decision": ctx.decision.value,
            "deny_reasons": ctx.deny_reasons,
            "escalate_reasons": ctx.escalate_reasons,
            "scan_risk_score": ctx.extras.get("scan_risk_score", 0.0),
            "policy_action": ctx.extras.get("policy_action"),
            "policy_name": ctx.extras.get("policy_name"),
            "has_taint": bool(ctx.taint_context.get("tainted_fields")),
            "timestamp": time.time(),
        }

        self._record_event(event, ctx)
        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Log the post-dispatch result including tool output metadata."""
        if ctx.is_denied and ctx.extras.get("katana.audit_short_circuit_logged"):
            return

        event = {
            "type": "tool_dispatch",
            "phase": "post",
            "call_id": ctx.call_id,
            "tool_name": ctx.tool_name,
            "decision": ctx.decision.value,
            "tool_duration_ms": ctx.tool_duration_ms,
            "middleware_ms": ctx.total_middleware_ms,
            "had_error": ctx.tool_error is not None,
            "output_scan_findings": bool(ctx.extras.get("output_scan_result")),
            "timestamp": time.time(),
        }

        self._record_event(event, ctx)

    def on_short_circuit(self, ctx: CallContext) -> None:
        """Log a denied pre-dispatch call that short-circuited the chain."""
        event = {
            "type": "tool_dispatch",
            "phase": "short_circuit",
            "call_id": ctx.call_id,
            "tool_name": ctx.tool_name,
            "decision": ctx.decision.value,
            "deny_reasons": ctx.deny_reasons,
            "escalate_reasons": ctx.escalate_reasons,
            "scan_risk_score": ctx.extras.get("scan_risk_score", 0.0),
            "policy_action": ctx.extras.get("policy_action"),
            "policy_name": ctx.extras.get("policy_name"),
            "short_circuit_middleware": ctx.extras.get("short_circuit_middleware"),
            "timestamp": time.time(),
        }
        self._record_event(event, ctx)
        ctx.extras["katana.audit_short_circuit_logged"] = True

    def _record_event(self, event: dict[str, Any], ctx: CallContext) -> None:
        """Write an event to the audit trail or fall back to logging."""
        trail = self.audit_trail
        if trail is not None:
            try:
                from hermes_katana.audit import AuditEntry, AuditEventType

                args_hash = hashlib.sha256(
                    f"{ctx.call_id}|{event.get('phase', '')}|{event.get('tool_name', '')}".encode("utf-8")
                ).hexdigest()[:16]
                entry = AuditEntry(
                    event_type=AuditEventType.TOOL_CALL,
                    tool_name=str(event.get("tool_name", "")),
                    args_hash=args_hash,
                    decision=str(event.get("decision", "")),
                    details=json.dumps(event, sort_keys=True, default=str),
                )
                trail.log(entry)
                return
            except Exception:
                logger.debug("Audit trail record failed, falling back to logging", exc_info=True)

        # Fallback: structured log
        level = logging.WARNING if ctx.is_denied else logging.INFO
        logger.log(
            level,
            "AUDIT [%s] %s %s → %s (risk=%.2f, %s)",
            event.get("phase", "?"),
            event.get("call_id", "?"),
            event.get("tool_name", "?"),
            event.get("decision", "?"),
            event.get("scan_risk_score", 0.0),
            event.get("policy_action", "none"),
        )


# ---------------------------------------------------------------------------
# Factory: create the default middleware chain
# ---------------------------------------------------------------------------


def create_default_chain(
    config: dict[str, Any] | None = None,
) -> "MiddlewareChain":
    """Build the default Katana middleware chain.

    Creates and wires the standard middleware in the recommended order.
    Configuration overrides can disable individual middleware or adjust
    thresholds.

    Args:
        config: Optional configuration dict with keys:

            - ``taint.enabled`` (bool, default True)
            - ``scabbard.enabled`` (bool, default True)
            - ``scabbard.config`` (ScabbardConfig instance, optional)
            - ``scabbard.secondary.enabled`` (bool, default True)
            - ``scabbard.secondary.config`` (ScabbardConfig instance, optional)
            - ``protectai.enabled`` (bool, default True)
            - ``protectai.block_threshold`` (float, default 0.92)
            - ``protectai.flag_threshold`` (float, default 0.70)
            - ``protectai.safe_passthrough`` (float, default 0.95)
            - ``scan.enabled`` (bool, default True)
            - ``scan.block_threshold`` (float, default 0.7)
            - ``scan.warn_threshold`` (float, default 0.4)
            - ``scan.vault_values`` (set[str], default empty)
            - ``mcp.enabled`` (bool, default True)
            - ``mcp.block_on_critical`` (bool, default True)
            - ``multiturn.enabled`` (bool, default True)
            - ``multiturn.block_threshold`` (float, default 0.75)
            - ``multiturn.warn_threshold`` (float, default 0.45)
            - ``rag_injection.enabled`` (bool, default True)
            - ``rag_injection.block_threshold`` (float, default 0.90)
            - ``rag_injection.warn_threshold`` (float, default 0.60)
            - ``structural.enabled`` (bool, default True)
            - ``structural.block_threshold`` (float, default 0.8)
            - ``structural.warn_threshold`` (float, default 0.5)
            - ``behavioral.enabled`` (bool, default True)
            - ``policy.enabled`` (bool, default True)
            - ``policy.preset`` (str, default "balanced")
            - ``policy.engine`` (PolicyEngine instance, optional)
            - ``audit.enabled`` (bool, default True)
            - ``audit.log_allow`` (bool, default True)
            - ``audit.trail`` (AuditTrail instance, optional)

    Returns:
        A fully-configured :class:`MiddlewareChain`.

    Example::

        chain = create_default_chain({
            "policy.preset": "max",
            "scan.block_threshold": 0.5,
            "audit.log_allow": False,
        })
    """
    from hermes_katana.middleware.chain import MiddlewareChain

    cfg = _resolve_chain_config(config)
    chain = MiddlewareChain()
    chain.active_profile = cfg["profile"]
    chain.resolved_config = dict(cfg)

    # 1. Taint tracking (highest priority)
    taint_mw = KatanaTaintMiddleware(
        tracker=cfg.get("taint.tracker"),
        enabled=cfg.get("taint.enabled", True),
    )
    chain.add(taint_mw)

    # 2. Scabbard classifier (runs BEFORE pattern-based scanners)
    scabbard_mw = KatanaScabbardMiddleware(
        config=cfg.get("scabbard.config"),
        enabled=cfg.get("scabbard.enabled", True),
        route_mode=cfg.get("scabbard.route_mode", "balanced"),
        scan_outputs=cfg.get("scabbard.scan_outputs", True),
        audit_routes=cfg.get("scabbard.audit_routes", True),
        enforce_output_blocks=cfg.get("scabbard.enforce_output_blocks", True),
    )
    chain.add(scabbard_mw)

    # 2.5. ProtectAI binary gate (between primary Scabbard=90 and secondary Scabbard=85, pri=88)
    protectai_mw = KatanaProtectAIMiddleware(
        gate=cfg.get("protectai.gate"),
        block_threshold=cfg.get("protectai.block_threshold", 0.92),
        flag_threshold=cfg.get("protectai.flag_threshold", 0.70),
        safe_passthrough=cfg.get("protectai.safe_passthrough", 0.95),
        enabled=cfg.get("protectai.enabled", True),
    )
    chain.add(protectai_mw)

    # 3. Secondary Scabbard classifier (overlapping ML signal, pri=85)
    scabbard_secondary_mw = KatanaScabbardSecondaryMiddleware(
        config=cfg.get("scabbard.secondary.config"),
        enabled=cfg.get("scabbard.secondary.enabled", True),
    )
    chain.add(scabbard_secondary_mw)

    # 4. Scanner
    scan_mw = KatanaScanMiddleware(
        vault_values=cfg.get("scan.vault_values"),
        block_threshold=cfg.get("scan.block_threshold", 0.7),
        warn_threshold=cfg.get("scan.warn_threshold", 0.4),
        check_injection=cfg.get("scan.check_injection", True),
        check_secrets=cfg.get("scan.check_secrets", True),
        check_unicode=cfg.get("scan.check_unicode", True),
        check_content=cfg.get("scan.check_content", True),
        enforce_output_findings=cfg.get("scan.enforce_output_findings", False),
        redact_output_secrets=cfg.get("scan.redact_output_secrets", True),
        route_aware=cfg.get("scan.route_aware", True),
        enabled=cfg.get("scan.enabled", True),
    )
    chain.add(scan_mw)

    # 4b. MCP tool-poisoning detector (pri=78)
    mcp_mw = KatanaMCPMiddleware(
        block_on_critical=cfg.get("mcp.block_on_critical", True),
        enabled=cfg.get("mcp.enabled", True),
    )
    chain.add(mcp_mw)

    # 4c. Multi-turn attack detector (pri=76)
    multiturn_mw = KatanaMultiTurnMiddleware(
        block_threshold=cfg.get("multiturn.block_threshold", 0.75),
        warn_threshold=cfg.get("multiturn.warn_threshold", 0.45),
        enabled=cfg.get("multiturn.enabled", True),
    )
    chain.add(multiturn_mw)

    # 4d. RAG injection detector (pri=74)
    rag_mw = KatanaRAGInjectionMiddleware(
        block_threshold=cfg.get("rag_injection.block_threshold", 0.90),
        warn_threshold=cfg.get("rag_injection.warn_threshold", 0.60),
        enabled=cfg.get("rag_injection.enabled", True),
    )
    chain.add(rag_mw)

    # 5. Structural analysis
    structural_mw = KatanaStructuralMiddleware(
        block_threshold=cfg.get("structural.block_threshold", 0.8),
        warn_threshold=cfg.get("structural.warn_threshold", 0.5),
        enabled=cfg.get("structural.enabled", True),
    )
    chain.add(structural_mw)

    # 5b. Behavioral (post-dispatch observer, pri=65)
    behavioral_mw = KatanaBehavioralMiddleware(
        tracker=cfg.get("behavioral.tracker"),
        block_on_sequence=cfg.get("behavioral.block_on_sequence", False),
        enabled=cfg.get("behavioral.enabled", True),
    )
    chain.add(behavioral_mw)

    # 6. Policy engine
    policy_mw = KatanaPolicyMiddleware(
        engine=cfg.get("policy.engine"),
        preset=cfg.get("policy.preset", "balanced"),
        enabled=cfg.get("policy.enabled", True),
    )
    chain.add(policy_mw)

    # 7. Audit trail (lowest priority — observes everything)
    audit_mw = KatanaAuditMiddleware(
        audit_trail=cfg.get("audit.trail"),
        log_allow=cfg.get("audit.log_allow", True),
        enabled=cfg.get("audit.enabled", True),
    )
    chain.add(audit_mw)

    logger.info(
        "Default middleware chain created: %s",
        [m.name for m in chain.list_middleware()],
    )
    return chain

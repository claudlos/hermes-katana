"""HermesKatana Taint Tracking Engine.

Inspired by Google's CaMeL paper (arxiv 2503.18813) — a data-flow tracking
system that prevents prompt-injection payloads from reaching security-critical
tool invocations.  Every value entering the agent runtime is tagged with
*taint labels* describing its origin, and flow-control policies block or
escalate when untrusted data would reach sensitive sinks.

Core types
----------
- :class:`TaintLabel` — enum of data-origin classifications
- :class:`TrustLevel` — how much to trust a source (TRUSTED/UNTRUSTED/CONDITIONAL)
- :class:`Source` — provenance record for a data point
- :class:`Reader` — access-control principal
- :class:`TaintedValue` — generic taint wrapper for any Python value
- :class:`TaintedStr` — string with character-level taint tracking
- :class:`TaintedList` — list with per-item taint
- :class:`TaintedDict` — dict with per-key taint
- :class:`TaintTracker` — central registry and flow-control engine
- :class:`FlowDecision` — ALLOW / DENY / ASK_USER / QUARANTINE
- :class:`FlowRule` — configurable data-flow policy rule
- :class:`FlowAnalyzer` — evaluates flow rules (the "TaintPolicy" engine)

Quick start
-----------
::

    from hermes_katana.taint import TaintTracker, Source, FlowDecision

    tracker = TaintTracker.get_instance()
    web_data = tracker.register("Hello from the web", Source.web("https://evil.com"))
    decision = tracker.check_flow(web_data, "terminal")
    assert decision == FlowDecision.DENY

"""

from hermes_katana.taint.flow import (
    FlowAnalysis,
    FlowAnalyzer,
    FlowDecision,
    FlowRule,
)
from hermes_katana.taint.labels import (
    Reader,
    Source,
    TaintLabel,
    TrustLevel,
    default_trust_for,
)
from hermes_katana.taint.tracker import TaintTracker, TrackerStats
from hermes_katana.taint.value import (
    CharTaint,
    TaintedDict,
    TaintedList,
    TaintedStr,
    TaintedValue,
    collect_sources,
    unwrap,
)

# Public alias — FlowAnalyzer *is* the policy engine
TaintPolicy = FlowAnalyzer

__all__ = [
    # Labels & provenance
    "TaintLabel",
    "TrustLevel",
    "Source",
    "Reader",
    "default_trust_for",
    # Values
    "TaintedValue",
    "TaintedStr",
    "TaintedList",
    "TaintedDict",
    "CharTaint",
    "unwrap",
    "collect_sources",
    # Tracker
    "TaintTracker",
    "TrackerStats",
    # Flow control
    "FlowDecision",
    "FlowRule",
    "FlowAnalysis",
    "FlowAnalyzer",
    "TaintPolicy",
]

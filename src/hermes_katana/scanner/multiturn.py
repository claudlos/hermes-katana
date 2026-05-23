"""Multi-turn attack detector for HermesKatana.

Tracks conversation state across turns and detects three families of
multi-turn attacks that single-message scanners cannot catch:

  1. Gradual escalation — attacker establishes trust over benign turns
     then ramps the ask. Each individual message looks acceptable;
     the trajectory across user turns does not.

  2. Persona drift — attacker incrementally rewrites the assistant's
     identity ("you're really X", "from now on call yourself Y",
     "your true purpose is..."). Single shifts are common in legitimate
     roleplay; repeated shifts with permanence framing are not.

  3. Memory poisoning — attacker injects false "memories" framed as
     prior context, false attribution to the assistant, fake system
     markers in user content, or unverified privilege claims
     ("remember that I am an admin").

Complementary to ``context_analyzer.py`` (which scores generic
conversation drift). This module is attack-pattern oriented and uses
*per-turn* signal extraction plus inter-turn trajectory rules.

Stateful: instantiate one :class:`MultiTurnDetector` per session, call
:meth:`MultiTurnDetector.add_turn` for every message, then
:meth:`MultiTurnDetector.assess` for the verdict.

Stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Union

__all__ = [
    "AttackType",
    "ConversationAssessment",
    "MultiTurnDetector",
    "MultiTurnFinding",
    "Severity",
    "Turn",
    "TurnRole",
    "analyze_conversation",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TurnRole:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AttackType:
    ESCALATION = "gradual_escalation"
    PERSONA_DRIFT = "persona_drift"
    MEMORY_POISONING = "memory_poisoning"


class Severity:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Turn:
    """A single conversation turn with extracted attack signals."""

    index: int
    role: str
    content: str
    risk_score: float = 0.0
    signals: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MultiTurnFinding:
    """An attack finding spanning one or more turns."""

    attack_type: str
    severity: str
    turn_indices: tuple[int, ...]
    description: str
    score: float


@dataclass(frozen=True, slots=True)
class ConversationAssessment:
    """Aggregate verdict for the conversation in its current state."""

    turn_count: int
    user_turn_count: int
    overall_risk: float
    escalation_score: float
    drift_score: float
    poisoning_score: float
    findings: tuple[MultiTurnFinding, ...]


# ---------------------------------------------------------------------------
# Per-turn signal patterns
# ---------------------------------------------------------------------------

# Override / instruction-bypass: verb + target nearby
_OVERRIDE_RE = re.compile(
    r"\b(?:ignore|disregard|forget|bypass|override|drop|skip|cancel)\b"
    r".{0,30}\b(?:previous|prior|above|earlier|all|safety|guideline|"
    r"guidelines|instruction|instructions|rule|rules|restriction|"
    r"restrictions|limitation|limitations|polic|policies)\b",
    re.IGNORECASE | re.DOTALL,
)

# High-value exfiltration targets
_EXFIL_RE = re.compile(
    r"\b(?:system\s+prompt|api[_\s-]?keys?|secrets?|credentials?|"
    r"passwords?|tokens?|private\s+keys?|env(?:ironment)?\s+vars?)\b",
    re.IGNORECASE,
)

# Persona / identity rewrite cues
_PERSONA_SHIFT_RE = re.compile(
    r"\b(?:you(?:\s+are|'re)\s+(?:now|actually|really)|"
    r"from\s+now\s+on(?:\s+you)?|"
    r"your\s+(?:real|true|new|actual)\s+(?:name|purpose|identity|self|role)|"
    r"act\s+as|pretend\s+(?:to\s+be|you'?re)|roleplay\s+as|"
    r"behave\s+(?:as|like)\s+(?:a|an|the))\b",
    re.IGNORECASE,
)

# Memory injection / false-recall framings
_MEMORY_INJECT_RE = re.compile(
    r"\b(?:remember\s+that|recall\s+that|"
    r"as\s+we\s+(?:discussed|agreed|established)|"
    r"you\s+(?:said|told\s+me|agreed|confirmed|promised)|"
    r"earlier\s+you\s+|previously\s+you\s+|"
    r"in\s+our\s+(?:earlier|previous|last)\s+(?:chat|conversation|message))\b",
    re.IGNORECASE,
)

# Privilege / authority claims (untrusted, easy to forge)
_PRIVILEGE_CLAIM_RE = re.compile(
    r"\b(?:I(?:'m|\s+am)\s+(?:an?\s+|the\s+)?(?:admin(?:istrator)?|owner|"
    r"developer|engineer|moderator|root|superuser|authorized|whitelisted|"
    r"trusted\s+user)|"
    r"I\s+have\s+(?:full\s+|special\s+|elevated\s+)?(?:permission|access|"
    r"clearance|authorization|privileges?))\b",
    re.IGNORECASE,
)

# Fake system / instruction markers embedded in user content
_FAKE_SYSTEM_RE = re.compile(
    r"(?:^|\n)\s*(?:\[?\s*(?:SYSTEM|ADMIN|ROOT|DEVELOPER|INTERNAL|"
    r"INSTRUCTION)\s*\]?\s*[:>]|"
    r"<\|?(?:system|im_start|im_end)\|?>|"
    r"###\s*(?:system|instruction)s?\b)",
    re.IGNORECASE,
)

# Permanence framing — turns one-off requests into persistent state
_PERMANENCE_RE = re.compile(
    r"\b(?:from\s+now\s+on|permanently|forever|never\s+forget|"
    r"for\s+the\s+rest\s+of|going\s+forward|moving\s+forward|"
    r"always\s+remember)\b",
    re.IGNORECASE,
)

# Soft restriction lexicon — weak signal on its own
_RESTRICTION_RE = re.compile(
    r"\b(?:restrictions?|limitations?|filters?|guidelines?|safeguards?|"
    r"guard\s?rails?|policies)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------


def _score_user_turn(text: str) -> tuple[float, tuple[str, ...]]:
    """Extract attack signals from a single user turn."""
    if not text:
        return 0.0, ()

    signals: list[str] = []
    score = 0.0

    if _FAKE_SYSTEM_RE.search(text):
        signals.append("fake_system")
        score += 0.50
    if _OVERRIDE_RE.search(text):
        signals.append("override")
        score += 0.45
    if _EXFIL_RE.search(text):
        signals.append("exfil")
        score += 0.40
    if _PRIVILEGE_CLAIM_RE.search(text):
        signals.append("privilege_claim")
        score += 0.25
    if _PERSONA_SHIFT_RE.search(text):
        signals.append("persona_shift")
        score += 0.20
    if _MEMORY_INJECT_RE.search(text):
        signals.append("memory_inject")
        score += 0.20
    if _PERMANENCE_RE.search(text):
        signals.append("permanence")
        score += 0.10
    if _RESTRICTION_RE.search(text):
        signals.append("restriction_lex")
        score += 0.05

    return min(score, 1.0), tuple(signals)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class MultiTurnDetector:
    """Stateful multi-turn attack detector.

    One instance per conversation/session. Order matters: call
    :meth:`add_turn` for every message in the order it occurred, then
    :meth:`assess` whenever you want a verdict.

    Args:
        escalation_window: How many recent user turns to consider when
            looking for escalation trajectories.
        escalation_threshold: Minimum risk score the *latest* user turn
            must reach for escalation to fire.
        drift_threshold: Minimum number of distinct user turns containing
            persona-shift signals before persona drift is reported.
        max_turns: Hard cap on retained turns; older turns are discarded
            (their indices remain stable on the surviving Turn objects).
    """

    def __init__(
        self,
        *,
        escalation_window: int = 4,
        escalation_threshold: float = 0.35,
        drift_threshold: int = 2,
        max_turns: int = 200,
    ) -> None:
        if escalation_window < 3:
            raise ValueError("escalation_window must be >= 3")
        if max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self.escalation_window = escalation_window
        self.escalation_threshold = escalation_threshold
        self.drift_threshold = drift_threshold
        self.max_turns = max_turns
        self._turns: list[Turn] = []
        self._next_index = 0

    # -- state ----------------------------------------------------------

    @property
    def turns(self) -> tuple[Turn, ...]:
        return tuple(self._turns)

    def add_turn(self, role: str, content: str) -> Turn:
        """Append a turn and return its parsed :class:`Turn` record."""
        index = self._next_index
        self._next_index += 1

        if role == TurnRole.USER:
            score, signals = _score_user_turn(content)
        else:
            score, signals = 0.0, ()

        turn = Turn(
            index=index,
            role=role,
            content=content,
            risk_score=score,
            signals=signals,
        )
        self._turns.append(turn)

        if len(self._turns) > self.max_turns:
            # Drop oldest, keep stable indices on surviving turns.
            self._turns = self._turns[-self.max_turns :]

        return turn

    def reset(self) -> None:
        self._turns.clear()
        self._next_index = 0

    # -- assessment -----------------------------------------------------

    def assess(self) -> ConversationAssessment:
        user_turns = [t for t in self._turns if t.role == TurnRole.USER]

        findings: list[MultiTurnFinding] = []
        esc = self._detect_escalation(user_turns)
        drift = self._detect_persona_drift(user_turns)
        poison = self._detect_memory_poisoning(user_turns)
        for f in (esc, drift, poison):
            if f is not None:
                findings.append(f)

        escalation_score = esc.score if esc else 0.0
        drift_score = drift.score if drift else 0.0
        poisoning_score = poison.score if poison else 0.0
        overall = min(
            1.0,
            escalation_score * 0.45 + drift_score * 0.30 + poisoning_score * 0.40,
        )

        return ConversationAssessment(
            turn_count=len(self._turns),
            user_turn_count=len(user_turns),
            overall_risk=overall,
            escalation_score=escalation_score,
            drift_score=drift_score,
            poisoning_score=poisoning_score,
            findings=tuple(findings),
        )

    # -- detectors ------------------------------------------------------

    def _detect_escalation(self, user_turns: Sequence[Turn]) -> MultiTurnFinding | None:
        if len(user_turns) < 3:
            return None

        window = list(user_turns[-self.escalation_window :])
        if len(window) < 3:
            return None

        scores = [t.risk_score for t in window]

        # Latest turn must actually be elevated, otherwise this is not an
        # escalation pattern (just noise or a single suspicious word).
        if scores[-1] < self.escalation_threshold:
            return None

        # We require a (near-)clean baseline at the start of the window.
        # Trust-building turns score ~0; any meaningful signal at the
        # start means this isn't a "gradual" attack.
        if scores[0] > 0.20:
            return None

        # Trajectory: late-window mean must exceed early-window mean by a
        # clear margin (>=0.25). Pure regression slopes give similar
        # answers but are noisier on tiny windows.
        half = max(1, len(scores) // 2)
        early = sum(scores[:half]) / half
        late = sum(scores[half:]) / max(1, len(scores) - half)
        delta = late - early
        if delta < 0.25:
            return None

        if scores[-1] >= 0.75:
            severity = Severity.CRITICAL
        elif scores[-1] >= 0.50:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

        return MultiTurnFinding(
            attack_type=AttackType.ESCALATION,
            severity=severity,
            turn_indices=tuple(t.index for t in window),
            description=(
                f"User-turn risk escalated from {scores[0]:.2f} to "
                f"{scores[-1]:.2f} over {len(window)} turns "
                f"(delta={delta:.2f})"
            ),
            score=min(1.0, delta + scores[-1] * 0.5),
        )

    def _detect_persona_drift(self, user_turns: Sequence[Turn]) -> MultiTurnFinding | None:
        shifting = [t for t in user_turns if "persona_shift" in t.signals]
        if len(shifting) < self.drift_threshold:
            return None

        permanence = [t for t in shifting if "permanence" in t.signals]

        base = 0.30 + 0.15 * (len(shifting) - self.drift_threshold)
        if permanence:
            base += 0.20
        score = min(1.0, base)

        if score >= 0.60:
            severity = Severity.HIGH
        elif score >= 0.35:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        suffix = f" with permanence framing in {len(permanence)} of them" if permanence else ""
        return MultiTurnFinding(
            attack_type=AttackType.PERSONA_DRIFT,
            severity=severity,
            turn_indices=tuple(t.index for t in shifting),
            description=(f"{len(shifting)} user turns attempt to redefine the assistant's persona/role" + suffix),
            score=score,
        )

    def _detect_memory_poisoning(self, user_turns: Sequence[Turn]) -> MultiTurnFinding | None:
        memory_turns: list[Turn] = []
        false_attrib_turns: list[Turn] = []
        privilege_turns: list[Turn] = []
        fake_system_turns: list[Turn] = []

        for t in user_turns:
            if "memory_inject" in t.signals:
                memory_turns.append(t)
                if self._is_false_attribution(t):
                    false_attrib_turns.append(t)
            if "privilege_claim" in t.signals:
                privilege_turns.append(t)
            if "fake_system" in t.signals:
                fake_system_turns.append(t)

        # Gating: require at least one strong signal, or two weak ones,
        # before reporting. A single "remember that" on its own is not
        # poisoning.
        strong = bool(fake_system_turns or false_attrib_turns or privilege_turns)
        weak_pile = len(memory_turns) >= 2
        if not (strong or weak_pile):
            return None

        score = 0.0
        parts: list[str] = []
        if fake_system_turns:
            score += 0.50
            parts.append(f"{len(fake_system_turns)} fake-system marker(s)")
        if false_attrib_turns:
            score += 0.40
            parts.append(f"{len(false_attrib_turns)} false-attribution claim(s)")
        elif memory_turns:
            score += 0.20
            parts.append(f"{len(memory_turns)} memory-injection cue(s)")
        if privilege_turns:
            score += 0.30
            parts.append(f"{len(privilege_turns)} privilege claim(s)")
        score = min(1.0, score)

        if score >= 0.75:
            severity = Severity.CRITICAL
        elif score >= 0.50:
            severity = Severity.HIGH
        elif score >= 0.30:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        all_indices = sorted({t.index for t in memory_turns + privilege_turns + fake_system_turns})
        return MultiTurnFinding(
            attack_type=AttackType.MEMORY_POISONING,
            severity=severity,
            turn_indices=tuple(all_indices),
            description="Memory poisoning signals: " + ", ".join(parts),
            score=score,
        )

    # -- helpers --------------------------------------------------------

    _ATTRIB_RE = re.compile(
        r"\byou\s+(?:said|told\s+me|agreed(?:\s+to)?|confirmed|promised)\s+"
        r"(?:that\s+)?[\"']?([^.!?\"'\n]{4,120})",
        re.IGNORECASE,
    )

    def _is_false_attribution(self, turn: Turn) -> bool:
        """Heuristic: does this turn attribute content to the assistant
        that the assistant never actually produced?

        We extract the claim fragment after a "you said"-style framing,
        pull significant words (>=4 chars), and check overlap with the
        concatenated text of all *prior* assistant turns. Low overlap
        means the user is fabricating earlier dialogue.
        """
        m = self._ATTRIB_RE.search(turn.content)
        if not m:
            return False

        claim = m.group(1).lower()
        words = re.findall(r"[a-z]{4,}", claim)
        if not words:
            return False
        significant = words[:6]

        prior_assistant_text = " ".join(
            t.content.lower() for t in self._turns if t.index < turn.index and t.role == TurnRole.ASSISTANT
        )
        if not prior_assistant_text:
            # Attributing to an assistant that has never spoken yet.
            return True

        overlap = sum(1 for w in significant if w in prior_assistant_text)
        threshold = max(2, (len(significant) + 1) // 2)
        return overlap < threshold


# ---------------------------------------------------------------------------
# Convenience: stateless helper
# ---------------------------------------------------------------------------


_Message = Union[Mapping[str, str], tuple[str, str]]


def analyze_conversation(
    messages: Iterable[_Message],
    **detector_kwargs,
) -> ConversationAssessment:
    """One-shot analysis: feed an iterable of messages, get the verdict.

    Each message may be either a ``{"role": ..., "content": ...}`` dict
    or a ``(role, content)`` tuple. Useful for ad-hoc scans where you
    don't need to keep the detector around.
    """
    detector = MultiTurnDetector(**detector_kwargs)
    for m in messages:
        if isinstance(m, Mapping):
            role = m.get("role", TurnRole.USER)
            content = m.get("content", "")
        else:
            role, content = m
        detector.add_turn(role, content)
    return detector.assess()

"""
Compositional Scanner for HermesKatana.

Detects injection attacks assembled from fragments spread across the prompt.
Catches attacks where individual words are benign but combine into injection
commands when found within proximity (e.g., "ignore" ... filler ... "previous"
... filler ... "instructions").

Gap addressed: compositional attacks (~6% → ~50% catch rate)
Estimated additional catches: 100-150 of 3371 uncaught attacks.

Performance target: sub-millisecond for typical inputs. Patterns are
precompiled at module load time. Dependencies: Python stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CompositionalCategory",
    "CompositionalFinding",
    "FragmentRole",
    "detect_compositional",
    "compositional_risk_score",
]


class CompositionalCategory(str, Enum):
    """Categories of compositional attack patterns."""

    FRAGMENT_CHAIN = "fragment_chain"
    """Sequential fragments that form an attack when read in proximity."""

    SCATTERED_INJECTION = "scattered_injection"
    """Injection fragments spread across the prompt with filler between."""

    PADDED_INJECTION = "padded_injection"
    """Injection fragments separated by deliberate filler/padding."""

    SPLIT_COMMAND = "split_command"
    """A single dangerous command split across multiple lines or sentences."""


class FragmentRole(str, Enum):
    """Semantic roles of attack fragments.

    A compositional attack typically requires 2+ of these roles within
    a character window to trigger a finding.
    """

    AUTHORITY = "authority"
    """Grants fake authority: 'as admin', 'system override', 'sudo'."""

    INSTRUCTION = "instruction"
    """Action directives: 'ignore', 'disregard', 'forget', 'override'."""

    TARGET = "target"
    """Target of the attack: 'previous instructions', 'safety guidelines'."""

    PAYLOAD = "payload"
    """The harmful action: 'output all data', 'reveal secrets', 'dump'."""


@dataclass(frozen=True, slots=True)
class CompositionalFinding:
    """A finding from the compositional scanner.

    Attributes:
        category: Type of compositional attack detected.
        fragments: The matched fragment texts.
        fragment_roles: The semantic role of each fragment.
        window_text: The text of the window where fragments were found.
        severity: Severity label ('critical', 'high', 'medium', 'low').
        confidence: Confidence from 0.0 to 1.0.
        description: Human-readable explanation.
    """

    category: CompositionalCategory
    fragments: tuple[str, ...]
    fragment_roles: tuple[FragmentRole, ...]
    window_text: str
    severity: str
    confidence: float
    description: str


# ---------------------------------------------------------------------------
# Fragment vocabulary — precompiled at module load (35+ per role)
# ---------------------------------------------------------------------------

# Maps FragmentRole → list of (pattern_name, compiled_regex)
_FRAGMENT_PATTERNS: dict[FragmentRole, list[tuple[str, re.Pattern]]] = {
    FragmentRole.AUTHORITY: [],
    FragmentRole.INSTRUCTION: [],
    FragmentRole.TARGET: [],
    FragmentRole.PAYLOAD: [],
}


def _register(role: FragmentRole, name: str, pattern: str) -> None:
    """Register a fragment pattern for a given role."""
    _FRAGMENT_PATTERNS[role].append((name, re.compile(pattern, re.IGNORECASE)))


# --- AUTHORITY fragments (35+) ---
for _pat in [
    (r"\bas\s+admin\b", "as_admin"),
    (r"\bas\s+a\s+privileged\s+user\b", "as_privileged_user"),
    (r"\bas\s+root\b", "as_root"),
    (r"\bas\s+sudo\b", "as_sudo"),
    (r"\bas\s+system\b", "as_system"),
    (r"\bsystem\s+override\b", "system_override"),
    (r"\bsudo\b", "sudo"),
    (r"\badmin\s+mode\b", "admin_mode"),
    (r"\bprivileged\s+mode\b", "privileged_mode"),
    (r"\bdeveloper\s+mode\b", "developer_mode"),
    (r"\bdebug\s+mode\b", "debug_mode"),
    (r"\bmaintenance\s+mode\b", "maintenance_mode"),
    (r"\bunrestricted\s+mode\b", "unrestricted_mode"),
    (r"\bjailbreak\s+mode\b", "jailbreak_mode"),
    (r"\buncensored\s+mode\b", "uncensored_mode"),
    (r"\bignore\s+all\s+restrictions\b", "ignore_restrictions"),
    (r"\bbypass\s+all\s+(?:safety|security)\b", "bypass_safety"),
    (r"\bno\s+(?:safety|content)\s+(?:filters?|restrictions?)\b", "no_safety_filters"),
    (r"\bwith\s+(?:elevated|elevated)\s+privileges\b", "elevated_privileges"),
    (r"\bignore\s+(?:your\s+)?(?:safety|safety\s+)?(?:guidelines?|rules?|policies?)\b", "ignore_safety_guidelines"),
    (r"\boverride\s+(?:all\s+)?(?:safety|security)\b", "override_safety"),
    (r"\bact\s+as\s+(?:if\s+)?(?:there\s+)?(?:are\s+no|you\s+have\s+no)\b", "act_as_no_restrictions"),
    (r"\bpretend\s+(?:you\s+)?(?:are\s+)?(?:unrestricted|unfiltered)\b", "pretend_unrestricted"),
    (r"\bbe\s+(?:a\s+)?(?:new|different|different\s+kind)\b", "be_new_kind"),
    (r"\byou\s+are\s+now\s+(?:a\s+)?(?:new|different)\b", "you_are_now_different"),
    (r"\bforget\s+(?:all\s+)?(?:your\s+)?(?:restrictions?|limitations?)\b", "forget_restrictions"),
    (r"\bremove\s+(?:all\s+)?(?:your\s+)?(?:restrictions?|filters?)\b", "remove_restrictions"),
    (r"\bdisable\s+(?:your\s+)?(?:safety|content)\s+(?:filters?|checks?)\b", "disable_safety_filters"),
    (r"\bturn\s+off\s+(?:your\s+)?(?:safety|content)\s+filtering\b", "turn_off_filtering"),
    (r"\bno\s+restrictions?\b", "no_restrictions"),
    (r"\bno\s+limits?\b", "no_limits"),
    (r"\bfull\s+access\b", "full_access"),
    (r"\bunlimited\s+access\b", "unlimited_access"),
    (r"\bact\s+without\s+(?:any\s+)?(?:restrictions?|limitations?)\b", "act_without_restrictions"),
    (r"\bin\s+(?:a\s+)?(?:god|developer|admin)\s+(?:mode|state)\b", "god_mode"),
    (r"\bemergent\s+mode\b", "emergent_mode"),
    (r"\boverride\s+mode\b", "override_mode"),
    (r"\bexploit\s+mode\b", "exploit_mode"),
    (r"\btest\s+mode\b", "test_mode"),
]:
    _register(FragmentRole.AUTHORITY, _pat[1], _pat[0])

# --- INSTRUCTION fragments (35+) ---
for _pat in [
    (r"\bignore\b", "ignore"),
    (r"\bdisregard\b", "disregard"),
    (r"\bforget\b", "forget"),
    (r"\boverride\b", "override"),
    (r"\bbypass\b", "bypass"),
    (r"\bdismiss\b", "dismiss"),
    (r"\bdiscard\b", "discard"),
    (r"\bdrop\b", "drop"),
    (r"\bcancel\b", "cancel"),
    (r"\babandon\b", "abandon"),
    (r"\breject\b", "reject"),
    (r"\breplace\b", "replace"),
    (r"\bsubstitute\b", "substitute"),
    (r"\bswap\b", "swap"),
    (r"\bchange\b", "change"),
    (r"\balter\b", "alter"),
    (r"\bmodify\b", "modify"),
    (r"\bupdate\b", "update"),
    (r"\brevise\b", "revise"),
    (r"\bnew\s+instructions\b", "new_instructions"),
    (r"\bnew\s+rules\b", "new_rules"),
    (r"\bnew\s+directives\b", "new_directives"),
    (r"\bupdated\s+instructions\b", "updated_instructions"),
    (r"\breal\s+instructions\b", "real_instructions"),
    (r"\btrue\s+instructions\b", "true_instructions"),
    (r"\bactual\s+instructions\b", "actual_instructions"),
    (r"\bfinal\s+instructions\b", "final_instructions"),
    (r"\blatest\s+instructions\b", "latest_instructions"),
    (r"\bfrom\s+now\s+on\b", "from_now_on"),
    (r"\bgoing\s+forward\b", "going_forward"),
    (r"\binstead\b", "instead"),
    (r"\brather\s+than\b", "rather_than"),
    (r"\bignore\s+prior\b", "ignore_prior"),
    (r"\bdisregard\s+previous\b", "disregard_previous"),
    (r"\bforget\s+earlier\b", "forget_earlier"),
    (r"\bdo\s+not\s+follow\b", "do_not_follow"),
    (r"\brefuse\b", "refuse"),
]:
    _register(FragmentRole.INSTRUCTION, _pat[1], _pat[0])

# --- TARGET fragments (35+) ---
for _pat in [
    (r"\bprevious\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "previous_instructions"),
    (r"\bprior\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "prior_instructions"),
    (r"\babove\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "above_instructions"),
    (r"\bearlier\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "earlier_instructions"),
    (r"\boriginal\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "original_instructions"),
    (r"\binitial\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "initial_instructions"),
    (r"\bfirst\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "first_instructions"),
    (r"\bdefault\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "default_instructions"),
    (r"\bsystem\s+(?:instructions?|prompts?|rules?|guidelines?)\b", "system_instructions"),
    (r"\byour\s+(?:instructions?|prompts?|rules?|guidelines?|directives?)\b", "your_instructions"),
    (r"\ball\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "all_instructions"),
    (r"\bentire\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "entire_instructions"),
    (r"\bcomplete\s+(?:instructions?|prompts?|rules?|directions?|guidelines?)\b", "complete_instructions"),
    (r"\bprevious\s+context\b", "previous_context"),
    (r"\bprior\s+context\b", "prior_context"),
    (r"\ball\s+previous\s+context\b", "all_previous_context"),
    (r"\bsafety\s+guidelines\b", "safety_guidelines"),
    (r"\bsafety\s+rules\b", "safety_rules"),
    (r"\bsafety\s+policies\b", "safety_policies"),
    (r"\bcontent\s+policies?\b", "content_policies"),
    (r"\bcontent\s+filters?\b", "content_filters"),
    (r"\b(restrictions?|limitations?)\b", "restrictions"),
    (r"\b(rules?|guidelines?|directives?)\b", "rules"),
    (r"\bsystem\s+prompt\b", "system_prompt"),
    (r"\bsystem\s+message\b", "system_message"),
    (r"\bthe\s+prompt\b", "the_prompt"),
    (r"\bpast\s+(?:instructions?|prompts?)\b", "past_instructions"),
    (r"\bprior\s+(?:instructions?|prompts?)\b", "prior_prompts"),
    (r"\bhidden\s+(?:instructions?|rules?|guidelines?)\b", "hidden_instructions"),
    (r"\bsecret\s+(?:instructions?|rules?|guidelines?)\b", "secret_instructions"),
    (r"\bmy\s+(?:instructions?|rules?|directives?)\b", "my_instructions"),
    (r"\byour\s+(?:configuration|settings?)\b", "your_configuration"),
    (r"\bthe\s+(?:rules?|guidelines?|directives?)\b", "the_rules"),
    (r"\bany\s+(?:restrictions?|limitations?)\b", "any_restrictions"),
    (r"\bother\s+(?:instructions?|rules?)\b", "other_instructions"),
]:
    _register(FragmentRole.TARGET, _pat[1], _pat[0])

# --- PAYLOAD fragments (35+) ---
for _pat in [
    (r"\boutput\s+all\s+(?:data|secrets?|information)\b", "output_all_data"),
    (r"\breveal\s+(?:all\s+)?(?:secrets?|credentials?|passwords?)\b", "reveal_secrets"),
    (r"\bshow\s+(?:me\s+)?(?:your\s+)?(?:full\s+)?(?:system\s+)?(?:prompt|instructions?)\b", "show_prompt"),
    (r"\btell\s+me\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)\b", "tell_prompt"),
    (r"\bextract\s+(?:all\s+)?(?:data|secrets?|credentials?)\b", "extract_data"),
    (r"\bdump\s+(?:all\s+)?(?:data|secrets?|credentials?)\b", "dump_data"),
    (r"\bsend\s+(?:all\s+)?(?:data|secrets?|output)\s+to\b", "send_data"),
    (r"\bexfiltrate\b", "exfiltrate"),
    (r"\bexpose\b", "expose"),
    (r"\bleak\b", "leak"),
    (r"\brepeat\s+(?:everything|all|all\s+the)\b", "repeat_all"),
    (r"\brecite\b", "recite"),
    (r"\bprint\b", "print"),
    (r"\bdisplay\b", "display"),
    (r"\bexecute\b", "execute"),
    (r"\brun\b", "run"),
    (r"\bdo\s+it\b", "do_it"),
    (r"\boutput\b", "output"),
    (r"\bshow\b", "show"),
    (r"\btell\b", "tell"),
    (r"\blist\b", "list"),
    (r"\bdivulge\b", "divulge"),
    (r"\bdisclose\b", "disclose"),
    (r"\bunmask\b", "unmask"),
    (r"\bdecrypt\b", "decrypt"),
    (r"\bdecode\b", "decode"),
    (r"\bdeobfuscate\b", "deobfuscate"),
    (r"\bencode\b", "encode"),
    (r"\btranslate\b", "translate"),
    (r"\bgenerate\s+the\s+(?:real|true)\s+response\b", "generate_real_response"),
    (r"\bforget\s+(?:your\s+)?(?:training|safety)\b", "forget_safety"),
    (r"\bstop\s+(?:following|obeying)\b", "stop_following"),
    (r"\bbegin\s+(?:as\s+)?(?:a\s+)?(?:new|different)\b", "begin_new"),
    (r"\bstop\b", "stop"),
    (r"\bexit\b", "exit"),
    (r"\bbreak\b", "break"),
    (r"\bdrop\s+out\b", "drop_out"),
    (r"\bswitch\s+(?:to\s+)?(?:a\s+)?(?:new|different)\b", "switch_to"),
    (r"\bbecome\b", "become"),
    (r"\bassume\s+(?:a\s+)?(?:new|different)\b", "assume_new"),
]:
    _register(FragmentRole.PAYLOAD, _pat[1], _pat[0])

# ---------------------------------------------------------------------------
# Meta-discussion and filler patterns
# ---------------------------------------------------------------------------

# Security-research / meta discussion markers — reduce confidence when present
_META_DISCUSSION = re.compile(
    r"\b(?:how\s+to\s+(?:defend|protect|prevent|detect|mitigate)"
    r"|what\s+is\s+(?:prompt\s+injection|a\s+jailbreak)"
    r"|security\s+(?:research|audit|review|analysis)"
    r"|red\s+team|penetration\s+test|CTF|capture\s+the\s+flag"
    r"|prompt\s+injection\s+(?:attack|vulnerability|exploit|demo)"
    r"|(?:defend|safeguard|secure|protect)\s+against\s+(?:prompt\s+injection|jailbreak|injection\s+attack)"
    r"|example\s+(?:of\s+)?(?:prompt\s+injection|jailbreak)"
    r"|sample\s+(?:prompt\s+injection|injection\s+attack)"
    r"|safety\s+(?:evaluation|testing|benchmark))\b",
    re.IGNORECASE,
)

# Deliberate filler/padding patterns — boost confidence when between fragments
_FILLER = re.compile(
    r"\b(?:lorem\s+ipsum|the\s+quick\s+brown\s+fox|"
    r"blah\s+blah|dear\s+AI|dear\s+assistant|hello\s+AI|"
    r"asdf|qwerty|xxxx|aaaa|nothing|stop|wait|continue|"
    r"okay|ok|yes|no|etc|and\s+so\s+on|blah\s+blah\s+blah|"
    r"word\s+word\s+word|filler|placeholder|sample\s+text)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Core detection logic
# ---------------------------------------------------------------------------


def _find_fragments(text: str) -> list[tuple[FragmentRole, str, int, int]]:
    """Find all fragment matches in text.

    Returns:
        List of (role, matched_text, start_pos, end_pos), sorted by position.
    """
    results: list[tuple[FragmentRole, str, int, int]] = []
    for role, patterns in _FRAGMENT_PATTERNS.items():
        for name, regex in patterns:
            for m in regex.finditer(text):
                results.append((role, m.group(), m.start(), m.end()))
    # Sort by start position
    results.sort(key=lambda x: x[2])
    return results


def _is_meta_discussion(text: str) -> bool:
    """Return True if text appears to be meta security discussion."""
    return bool(_META_DISCUSSION.search(text))


def _filler_ratio(text: str) -> float:
    """Return ratio of filler words in text (0.0–1.0)."""
    words = text.split()
    if not words:
        return 0.0
    filler_count = len(_FILLER.findall(text))
    # Avoid double-counting multi-word filler phrases
    filler_count = len(_FILLER.findall(text))
    return min(1.0, filler_count / max(1, len(words)))


def _score_window(
    roles_in_window: list[FragmentRole],
    filler_ratio_val: float,
    is_meta: bool,
) -> tuple[float, CompositionalCategory, str]:
    """Score a window based on roles present and context.

    Returns (confidence, category, description).
    """
    unique_roles = set(roles_in_window)
    role_count = len(unique_roles)

    # Categorize
    if role_count >= 4:
        category = CompositionalCategory.FRAGMENT_CHAIN
        desc = f"4 distinct fragment roles found in proximity ({', '.join(r.value for r in sorted(unique_roles))})"
    elif role_count == 3:
        # Distinguish SCATTERED vs PADDED by filler ratio
        if filler_ratio_val > 0.15:
            category = CompositionalCategory.PADDED_INJECTION
            desc = "3 fragment roles with deliberate padding between"
        else:
            category = CompositionalCategory.SCATTERED_INJECTION
            desc = f"3 distinct fragment roles found in proximity ({', '.join(r.value for r in sorted(unique_roles))})"
    elif role_count == 2:
        if filler_ratio_val > 0.15:
            category = CompositionalCategory.PADDED_INJECTION
            desc = "2 fragment roles separated by deliberate filler"
        else:
            category = CompositionalCategory.SCATTERED_INJECTION
            desc = f"2 distinct fragment roles found in proximity ({', '.join(r.value for r in sorted(unique_roles))})"
    else:
        # role_count <= 1 — shouldn't happen with a finding, but guard
        return 0.0, CompositionalCategory.FRAGMENT_CHAIN, "insufficient roles"

    # Base confidence from role count
    if role_count >= 4:
        confidence = 0.95
    elif role_count == 3:
        confidence = 0.80
    else:
        confidence = 0.50

    # Proximity/filler bonus: deliberate padding between fragments
    if filler_ratio_val > 0.10:
        confidence += 0.10
        confidence = min(0.99, confidence)

    # Meta-discussion penalty
    if is_meta:
        confidence -= 0.30
        confidence = max(0.0, confidence)

    return confidence, category, desc


def detect_compositional(text: str, *, window_size: int = 200) -> list[CompositionalFinding]:
    """Detect compositional injection attacks via fragment proximity analysis.

    Algorithm:
    1. Find all fragment matches in the text.
    2. If fewer than 2 fragments, return [] (no compositional pattern possible).
    3. Slide a character window of window_size across the text.
    4. For each window, collect all fragment roles present.
    5. If 2+ distinct roles are present, score the window and create a finding.
    6. Deduplicate overlapping windows (keep highest-confidence per region).
    7. Apply meta-discussion discount globally.

    Args:
        text: Input text to scan.
        window_size: Character window size (default 200).

    Returns:
        List of CompositionalFinding sorted by confidence descending.
    """
    if not text or len(text) < 4:
        return []

    fragments = _find_fragments(text)
    if len(fragments) < 2:
        return []

    # Deduplicate by (role, start, end)
    seen: set[tuple[FragmentRole, int, int]] = set()
    unique_frags: list[tuple[FragmentRole, str, int, int]] = []
    for f in fragments:
        key = (f[0], f[2], f[3])
        if key not in seen:
            seen.add(key)
            unique_frags.append(f)

    findings: list[CompositionalFinding] = []
    n = len(unique_frags)

    # Slide window anchored at each fragment's start position
    for i in range(n):
        role_i, _, start_i, _ = unique_frags[i]
        window_end = start_i + window_size

        # Collect all fragments j where start_j < window_end
        window_frags: list[tuple[FragmentRole, str, int, int]] = []
        for j in range(i, n):
            role_j, text_j, start_j, end_j = unique_frags[j]
            if start_j <= window_end:
                window_frags.append((role_j, text_j, start_j, end_j))
            else:
                break

        if len(window_frags) < 2:
            continue

        roles_in_window = [f[0] for f in window_frags]
        unique_roles = set(roles_in_window)

        if len(unique_roles) < 2:
            continue

        # Get window text
        w_start = window_frags[0][2]
        w_end = window_frags[-1][3]
        window_text = text[w_start:w_end]

        # Check meta-discussion and filler
        is_meta = _is_meta_discussion(window_text)
        f_ratio = _filler_ratio(window_text)

        confidence, category, desc = _score_window(roles_in_window, f_ratio, is_meta)

        # Suppress very low confidence findings
        if confidence < 0.40:
            continue

        # Severity mapping
        if confidence >= 0.90:
            severity = "critical"
        elif confidence >= 0.75:
            severity = "high"
        elif confidence >= 0.55:
            severity = "medium"
        else:
            severity = "low"

        findings.append(
            CompositionalFinding(
                category=category,
                fragments=tuple(f[1] for f in window_frags),
                fragment_roles=tuple(f[0] for f in window_frags),
                window_text=window_text,
                severity=severity,
                confidence=round(confidence, 3),
                description=desc,
            )
        )

    # Deduplicate overlapping windows: group by approximate position,
    # keep highest confidence per group
    if not findings:
        return []

    findings.sort(key=lambda f: -f.confidence)

    # Suppress overlapping findings: keep first (highest confidence),
    # skip any whose window_text is a substring of a higher-confidence one
    deduped: list[CompositionalFinding] = []
    for finding in findings:
        is_duplicate = any(
            finding.window_text in existing.window_text and finding.confidence <= existing.confidence
            for existing in deduped
        )
        if not is_duplicate:
            deduped.append(finding)

    deduped.sort(key=lambda f: -f.confidence)
    return deduped


def compositional_risk_score(text: str) -> float:
    """Compute aggregate compositional risk score (0.0–1.0).

    Returns max confidence from detect_compositional(), or 0.0 if no findings.
    """
    findings = detect_compositional(text)
    if not findings:
        return 0.0
    return max(f.confidence for f in findings)

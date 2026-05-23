"""
RAG-Injection Scanner for HermesKatana.

Detects indirect prompt injection delivered through retrieved documents,
poisoned embedding content, and context manipulation inside RAG pipelines.

Threat coverage:
  - Indirect prompt injection inside retrieved content
  - Role-marker / chat-template hijacking (ChatML, Llama, JSON, XML)
  - Context manipulation (fake conversation turns, delimiter breakout)
  - Tool / function-call hijack inside retrieved text
  - Embedding poisoning via keyword stuffing and line repetition
  - Invisible-character injection (zero-width, RTL override, Unicode tags)
  - Source / authority spoofing
  - Exfiltration primitives (markdown image beacons, data callbacks)

Performance target: <2ms per typical retrieved chunk.
Dependencies: Python stdlib only.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "RAGInjectionCategory",
    "RAGInjectionFinding",
    "detect_rag_injection",
    "rag_injection_risk_score",
    "scan_retrieved_documents",
]


class RAGInjectionCategory(str, Enum):
    INDIRECT_INJECTION = "indirect_injection"
    ROLE_HIJACK = "role_hijack"
    CONTEXT_MANIPULATION = "context_manipulation"
    TOOL_HIJACK = "tool_hijack"
    POISONED_EMBEDDING = "poisoned_embedding"
    INVISIBLE_INJECTION = "invisible_injection"
    SOURCE_SPOOFING = "source_spoofing"
    EXFILTRATION = "exfiltration"


@dataclass(frozen=True, slots=True)
class RAGInjectionFinding:
    category: RAGInjectionCategory
    matched_text: str
    pattern_name: str
    confidence: float
    description: str
    offset: int = -1


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

_PatternEntry = tuple[str, re.Pattern[str], RAGInjectionCategory, float, str]
_PATTERNS: list[_PatternEntry] = []


def _p(
    name: str,
    pattern: str,
    category: RAGInjectionCategory,
    confidence: float,
    description: str,
    flags: int = re.IGNORECASE,
) -> None:
    _PATTERNS.append((name, re.compile(pattern, flags), category, confidence, description))


# === INDIRECT INJECTION (payloads embedded in retrieved documents) ===

_p(
    "ignore_previous",
    r"\b(?:ignore|disregard|forget|override|bypass|skip)\b.{0,30}\b(?:all\s+)?(?:previous|prior|above|preceding|earlier|former|original)\b.{0,30}\b(?:instructions?|messages?|prompts?|rules?|directives?|guidelines?|commands?|context)",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.95,
    "Ignore-previous-instructions payload inside document",
)

_p(
    "new_instructions_label",
    r"(?:^|[\r\n])\s*(?:new|updated|revised|important|critical|override)\s+(?:instructions?|directive|task|mission|orders?|commands?)\s*[:\-\u2013]\s*\S",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.80,
    "New-instructions label announcing embedded directive",
)

_p(
    "note_to_assistant",
    r"\b(?:(?:note|message|memo|instructions?|attention|warning)\s+(?:to|for)\s+(?:the\s+)?|dear\s+|hey\s+|hi\s+|hello\s+)(?:ai|assistant|llm|model|chatbot|chat\s*gpt|claude|gemini|gpt|bot|agent)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.90,
    "Note/message directly addressed to the AI assistant",
)

_p(
    "model_imperative",
    r"\b(?:assistant|ai|model|llm|claude|chat\s*gpt|gpt|bot|chatbot)\s*[,:\-\u2013]{1,3}\s*you\s+(?:must|should|will|need\s+to|have\s+to|are\s+(?:required|ordered|instructed)\s+to)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.85,
    "Direct imperative addressed to the model",
)

_p(
    "when_user_asks_respond",
    r"\b(?:when(?:ever)?|if)\b.{0,20}\b(?:user|they|someone|anyone|asked)\b.{0,20}\b(?:asks?|requests?|says?|queries?|mentions?)\b.{0,40}\b(?:respond|reply|answer|tell|say|output|return)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.78,
    "Conditional response instruction embedded in document",
)

_p(
    "do_not_tell_user",
    r"\b(?:do\s+not|don'?t|never|under\s+no\s+circumstances)\b.{0,15}\b(?:tell|inform|mention|reveal|disclose|share|warn|notify)\b.{0,15}\b(?:the\s+)?(?:user|human|person|customer)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.88,
    "Instruction to hide information from the user",
)

_p(
    "keep_secret",
    r"\b(?:keep|maintain|hold)\b.{0,10}\b(?:this|these|the\s+following)\b.{0,15}\b(?:secret|hidden|confidential|private|undisclosed)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.80,
    "Instruction to keep content secret",
)

_p(
    "before_answering",
    r"\b(?:before|prior\s+to)\b.{0,10}\b(?:answering|responding|replying|generating)\b.{0,30}\b(?:you\s+)?(?:must|should|need\s+to|have\s+to)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.78,
    "Pre-response instruction injection",
)

_p(
    "stop_and_redirect",
    r"\b(?:stop|halt|pause|cease)\b.{0,15}\b(?:reading|processing|what|everything|the\s+(?:document|context))\b.{0,20}\b(?:and|then)\b.{0,15}\b(?:do|execute|perform|run|output|say|reveal|print)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.85,
    "Stop-and-redirect injection",
)

_p(
    "you_are_redefinition",
    r"\byou\s+are\s+(?:now\s+)?(?:an?\s+)?(?:helpful|evil|unrestricted|uncensored|jailbroken|dan|developer\s+mode|dev\s+mode|new|unfiltered|free)\b.{0,30}\b(?:assistant|ai|model|bot|chatbot|chat\s*gpt|gpt)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.88,
    "Role redefinition directive inside document",
)

_p(
    "important_ai_frame",
    r"\b(?:important|attention|urgent|critical|warning)\b\s*[:\-\u2013]\s*(?:to\s+)?(?:ai|assistant|model|llm|bot|chatbot|reader)\b",
    RAGInjectionCategory.INDIRECT_INJECTION,
    0.78,
    "Important-note frame targeting the AI",
)


# === ROLE HIJACK (chat template markers inside retrieved text) ===

_p(
    "chatml_role_marker",
    r"<\|im_start\|>\s*(?:system|user|assistant|tool|function)|<\|im_end\|>|<\|endoftext\|>|<\|system\|>|<\|user\|>|<\|assistant\|>",
    RAGInjectionCategory.ROLE_HIJACK,
    0.97,
    "ChatML / OpenAI-style role marker inside retrieved text",
)

_p(
    "llama_inst_marker",
    r"\[/?INST\]|<<\s*SYS\s*>>|<<\s*/\s*SYS\s*>>",
    RAGInjectionCategory.ROLE_HIJACK,
    0.95,
    "Llama-style [INST] / <<SYS>> marker",
)

_p(
    "markdown_role_heading",
    r"(?:^|[\r\n])\s*#{1,3}\s*(?:system\s+(?:prompt|message|instructions?)|new\s+instructions?|assistant\s+instructions?)\s*[\r\n:\-\u2013]",
    RAGInjectionCategory.ROLE_HIJACK,
    0.85,
    "Markdown heading posing as a system-message marker",
)

_p(
    "plain_role_colon",
    r"(?:^|[\r\n])\s*(?:system|human|user|assistant)\s*:\s*\S",
    RAGInjectionCategory.ROLE_HIJACK,
    0.65,
    "Plain 'System:' / 'User:' / 'Assistant:' pseudo-role line",
)

_p(
    "json_role_field",
    r"[\"']role[\"']\s*:\s*[\"'](?:system|user|assistant|tool|function|developer)[\"']",
    RAGInjectionCategory.ROLE_HIJACK,
    0.85,
    "Embedded JSON role field mimicking a chat message",
)

_p(
    "xml_system_tag",
    r"<\s*(?:system|instructions?|sys|prompt)\s*>[\s\S]{0,400}?<\s*/\s*(?:system|instructions?|sys|prompt)\s*>",
    RAGInjectionCategory.ROLE_HIJACK,
    0.85,
    "XML-style system/instructions tag block",
)


# === CONTEXT MANIPULATION ===

_p(
    "end_of_document_switch",
    r"(?:^|[\r\n])\s*[-=#*_]{3,}\s*(?:end\s+of\s+(?:document|context|retrieval|knowledge\s+base|chunk)|document\s+ends?\s+here)\s*[-=#*_]{0,}\s*[\r\n]+.{0,40}\b(?:now|next|new|system|user|assistant|begin)\b",
    RAGInjectionCategory.CONTEXT_MANIPULATION,
    0.92,
    "Fake end-of-document followed by new context",
)

_p(
    "previous_conversation_fabrication",
    r"\b(?:previous|earlier|prior|past)\s+(?:conversation|chat|session|dialogue|exchange|messages?)\s*[:\-\u2013]|\b(?:earlier|previously|before)\s+(?:the\s+)?user\s+(?:said|asked|requested|told\s+you)\b",
    RAGInjectionCategory.CONTEXT_MANIPULATION,
    0.78,
    "Fabricated conversation history reference",
)

_p(
    "delimiter_breakout",
    r"```\s*[\r\n]+\s*(?:system|user|assistant|instructions?)\s*[:\-\u2013]|[\"']{3}\s*[\r\n]+\s*(?:system|instructions?)\s*[:\-\u2013]",
    RAGInjectionCategory.CONTEXT_MANIPULATION,
    0.88,
    "Delimiter-breakout attempt (code fence / triple quote)",
)

_p(
    "fake_tool_response",
    r"(?:^|[\r\n])\s*(?:observation|tool[_\s]+(?:result|response|output)|function[_\s]+(?:result|response|output))\s*[:\-\u2013]\s*\S",
    RAGInjectionCategory.CONTEXT_MANIPULATION,
    0.78,
    "Fake tool/function response injected as context",
)

_p(
    "react_trace_injection",
    r"(?:^|[\r\n])\s*(?:thought|action|action\s+input|final\s+answer)\s*:\s*\S[\s\S]{0,200}?(?:[\r\n])\s*(?:action|observation|final\s+answer)\s*:",
    RAGInjectionCategory.CONTEXT_MANIPULATION,
    0.80,
    "ReAct-style Thought/Action/Observation trace injection",
)


# === TOOL / FUNCTION-CALL HIJACK ===

_p(
    "tool_call_tag",
    r"<\s*(?:tool_call|function_call|tool_use|invoke)\s*[^>]*>[\s\S]{0,500}?<\s*/\s*(?:tool_call|function_call|tool_use|invoke)\s*>",
    RAGInjectionCategory.TOOL_HIJACK,
    0.92,
    "Embedded tool-call XML tag",
)

_p(
    "function_call_json",
    r"[\"']function_call[\"']\s*:\s*\{[^{}]{0,300}?[\"']name[\"']\s*:\s*[\"'][a-zA-Z_][\w]*[\"']",
    RAGInjectionCategory.TOOL_HIJACK,
    0.88,
    "JSON function_call block embedded in text",
)

_p(
    "tool_use_block",
    r"[\"']type[\"']\s*:\s*[\"']tool_use[\"']\s*,\s*[\"']name[\"']\s*:\s*[\"'][a-zA-Z_][\w]*[\"']",
    RAGInjectionCategory.TOOL_HIJACK,
    0.90,
    "Anthropic-style tool_use JSON block",
)

_p(
    "execute_following",
    r"\b(?:execute|run|invoke|call)\b.{0,15}\b(?:this|the\s+following|these|below)\s+(?:shell|bash|python|javascript|code|command|function|script)\b",
    RAGInjectionCategory.TOOL_HIJACK,
    0.80,
    "Instruction telling model to execute embedded command",
)


# === EXFILTRATION PRIMITIVES ===

_p(
    "markdown_image_beacon",
    r"!\[[^\]]*\]\(https?://[^\s?)]{1,200}\?[^\s)]*(?:data|token|secret|prompt|key|cookie|session|user|auth|q|payload)[^\s)]*\)",
    RAGInjectionCategory.EXFILTRATION,
    0.88,
    "Markdown image beacon with suspicious query parameter",
)

_p(
    "exfil_fetch_url",
    r"\b(?:fetch|open|visit|browse|navigate\s+to|send\s+(?:a\s+)?(?:request|post|get)\s+to)\b.{0,20}https?://",
    RAGInjectionCategory.EXFILTRATION,
    0.75,
    "Instruction telling model to contact external URL",
)

_p(
    "append_data_to_url",
    r"\b(?:append|add|include|attach|encode)\b.{0,30}\b(?:user(?:'s)?\s+(?:input|message|query|prompt|data)|conversation|history|secret|token|api[_\s]?key|credentials?)\b.{0,30}\b(?:url|link|parameter|query|request|endpoint)\b",
    RAGInjectionCategory.EXFILTRATION,
    0.90,
    "Instruction to append user data to outbound URL",
)


# === SOURCE / AUTHORITY SPOOFING ===

_p(
    "trusted_source_header",
    r"\b(?:official|verified|authorized|trusted|certified|authenticated|signed)\b.{0,15}\b(?:by|from|source)\b.{0,15}\b(?:openai|anthropic|google|meta|microsoft|deepmind|nvidia|apple)\b",
    RAGInjectionCategory.SOURCE_SPOOFING,
    0.78,
    "Authority-spoofing header referencing AI vendor",
)

_p(
    "vendor_directive",
    r"\b(?:openai|anthropic|google|meta|microsoft|deepmind)\b.{0,15}\b(?:policy|directive|mandate|requirement|guideline|rule)\b.{0,30}\b(?:requires?|demands?|states?|says?|instructs?|mandates?|orders?)\b",
    RAGInjectionCategory.SOURCE_SPOOFING,
    0.75,
    "Fake vendor policy invoked to coerce behavior",
)

_p(
    "trust_unconditionally",
    r"\b(?:trust|believe|accept)\b.{0,15}\b(?:this|these|the\s+following)\b.{0,15}\b(?:document|content|information|instructions?|source)\b.{0,25}\b(?:without\s+(?:question|doubt|verification|hesitation)|as\s+(?:authoritative|fact|truth|verified))",
    RAGInjectionCategory.SOURCE_SPOOFING,
    0.82,
    "Appeal to trust the document unconditionally",
)


# ---------------------------------------------------------------------------
# Invisible / directional character detection
# ---------------------------------------------------------------------------

_INVISIBLE_CHARS: dict[str, str] = {
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\u2060": "WORD JOINER",
    "\ufeff": "ZERO WIDTH NO-BREAK SPACE",
    "\u202a": "LEFT-TO-RIGHT EMBEDDING",
    "\u202b": "RIGHT-TO-LEFT EMBEDDING",
    "\u202c": "POP DIRECTIONAL FORMATTING",
    "\u202d": "LEFT-TO-RIGHT OVERRIDE",
    "\u202e": "RIGHT-TO-LEFT OVERRIDE",
    "\u2066": "LEFT-TO-RIGHT ISOLATE",
    "\u2067": "RIGHT-TO-LEFT ISOLATE",
    "\u2068": "FIRST STRONG ISOLATE",
    "\u2069": "POP DIRECTIONAL ISOLATE",
}

_TAG_CHAR_RE = re.compile(r"[\U000e0000-\U000e007f]")


def _detect_invisible(text: str) -> list[RAGInjectionFinding]:
    findings: list[RAGInjectionFinding] = []
    counts: Counter[str] = Counter()
    first_offset: dict[str, int] = {}
    for idx, ch in enumerate(text):
        if ch in _INVISIBLE_CHARS:
            counts[ch] += 1
            first_offset.setdefault(ch, idx)

    for ch, count in counts.items():
        name = _INVISIBLE_CHARS[ch]
        confidence = 0.95 if count >= 3 else 0.85
        slug = name.lower().replace(" ", "_").replace("-", "_")
        findings.append(
            RAGInjectionFinding(
                category=RAGInjectionCategory.INVISIBLE_INJECTION,
                matched_text=repr(ch),
                pattern_name=f"invisible_{slug}",
                confidence=confidence,
                description=(f"Invisible / directional character: {name} (count={count})"),
                offset=first_offset[ch],
            )
        )

    tag_match = _TAG_CHAR_RE.search(text)
    if tag_match:
        findings.append(
            RAGInjectionFinding(
                category=RAGInjectionCategory.INVISIBLE_INJECTION,
                matched_text=repr(tag_match.group()),
                pattern_name="unicode_tag_char",
                confidence=0.97,
                description=("Unicode Tag character (U+E0000-U+E007F) used for covert payloads"),
                offset=tag_match.start(),
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Embedding-poisoning heuristics (keyword stuffing, line repetition)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_\-]{2,}\b")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "any",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "day",
        "get",
        "has",
        "him",
        "his",
        "how",
        "man",
        "new",
        "now",
        "old",
        "see",
        "two",
        "way",
        "who",
        "boy",
        "did",
        "its",
        "let",
        "put",
        "say",
        "she",
        "too",
        "use",
        "with",
        "this",
        "that",
        "from",
        "they",
        "have",
        "been",
        "their",
        "what",
        "your",
        "would",
        "there",
        "could",
        "other",
        "than",
        "then",
        "them",
        "these",
        "some",
        "into",
        "more",
        "will",
        "only",
        "also",
        "such",
        "very",
        "when",
        "much",
        "most",
        "many",
        "made",
        "make",
    }
)


def _detect_poisoning(text: str) -> list[RAGInjectionFinding]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < 15:
        return []

    findings: list[RAGInjectionFinding] = []
    counts = Counter(w for w in words if w not in _STOPWORDS)
    total = len(words)

    for word, count in counts.most_common(5):
        if count < 7:
            break
        share = count / total
        if share >= 0.18 and count >= 7:
            confidence = round(min(0.70 + share, 0.95), 2)
            findings.append(
                RAGInjectionFinding(
                    category=RAGInjectionCategory.POISONED_EMBEDDING,
                    matched_text=word,
                    pattern_name="keyword_stuffing",
                    confidence=confidence,
                    description=(f"Keyword stuffing: '{word}' repeated {count}x ({share:.0%} of tokens)"),
                    offset=text.lower().find(word),
                )
            )

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 6:
        line_counts = Counter(lines)
        repeated_line, line_count = line_counts.most_common(1)[0]
        if line_count >= 5 and len(repeated_line) >= 10:
            findings.append(
                RAGInjectionFinding(
                    category=RAGInjectionCategory.POISONED_EMBEDDING,
                    matched_text=repeated_line[:80],
                    pattern_name="line_repetition",
                    confidence=0.82,
                    description=(f"Line repeated {line_count}x — embedding spam pattern"),
                    offset=text.find(repeated_line),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_rag_injection(text: str) -> list[RAGInjectionFinding]:
    """Scan a retrieved document (or any text) for RAG-injection signals.

    Returns findings sorted by confidence descending.
    """
    if not text or not text.strip():
        return []

    findings: list[RAGInjectionFinding] = []

    for name, regex, category, confidence, description in _PATTERNS:
        for m in regex.finditer(text):
            findings.append(
                RAGInjectionFinding(
                    category=category,
                    matched_text=m.group()[:200],
                    pattern_name=name,
                    confidence=confidence,
                    description=f"[{name}] {description}",
                    offset=m.start(),
                )
            )

    findings.extend(_detect_invisible(text))
    findings.extend(_detect_poisoning(text))

    findings.sort(key=lambda f: (-f.confidence, f.offset))
    return findings


def rag_injection_risk_score(text: str) -> float:
    """Compute aggregate 0.0-1.0 risk score for RAG injection in `text`."""
    findings = detect_rag_injection(text)
    if not findings:
        return 0.0

    max_conf = max(f.confidence for f in findings)
    categories = {f.category for f in findings}

    if len(categories) >= 3:
        boost = 0.10
    elif len(categories) >= 2:
        boost = 0.05
    elif len(findings) >= 3:
        boost = 0.05
    else:
        boost = 0.0

    return min(round(max_conf + boost, 4), 1.0)


def scan_retrieved_documents(
    documents: list,
) -> list[tuple[int, RAGInjectionFinding]]:
    """Scan a batch of retrieved documents.

    Accepts either raw strings or dict-like records with a
    'content' / 'text' / 'body' / 'page_content' / 'chunk' field.
    Returns list of (document_index, finding) tuples sorted by confidence.
    """
    results: list[tuple[int, RAGInjectionFinding]] = []
    for idx, doc in enumerate(documents):
        if isinstance(doc, str):
            content = doc
        elif isinstance(doc, dict):
            content = ""
            for key in ("content", "text", "body", "page_content", "chunk"):
                value = doc.get(key)
                if isinstance(value, str) and value:
                    content = value
                    break
        else:
            continue
        for finding in detect_rag_injection(content):
            results.append((idx, finding))

    results.sort(key=lambda pair: -pair[1].confidence)
    return results

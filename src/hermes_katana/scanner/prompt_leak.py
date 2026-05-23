"""
Prompt-Leak Scanner for HermesKatana.

Detects system prompt exfiltration attempts — direct requests, encoding tricks,
completion-based extraction, social engineering, and indirect phrasing that
bypasses simple keyword matching.

Performance target: <2ms per prompt.
Dependencies: Python stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PromptLeakCategory",
    "PromptLeakFinding",
    "detect_prompt_leak",
    "prompt_leak_risk_score",
]


class PromptLeakCategory(str, Enum):
    DIRECT_EXTRACT = "direct_extract"
    COMPLETION_BASED = "completion_based"
    ENCODING_EXTRACT = "encoding_extract"
    SOCIAL_ENGINEERING = "social_engineering"
    TRANSLATION_EXTRACT = "translation_extract"
    SUMMARIZATION_EXTRACT = "summarization_extract"
    FORMAT_EXTRACT = "format_extract"
    META_QUESTION = "meta_question"
    INSTRUCTION_REPLAY = "instruction_replay"


@dataclass(frozen=True, slots=True)
class PromptLeakFinding:
    category: PromptLeakCategory
    matched_text: str
    pattern_name: str
    confidence: float
    description: str


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------

_PatternEntry = tuple[str, re.Pattern[str], PromptLeakCategory, float, str]
_PATTERNS: list[_PatternEntry] = []


def _p(
    name: str,
    pattern: str,
    category: PromptLeakCategory,
    confidence: float,
    description: str,
    flags: int = re.IGNORECASE,
) -> None:
    _PATTERNS.append((name, re.compile(pattern, flags), category, confidence, description))


# === DIRECT EXTRACTION ===

_p(
    "reveal_system_prompt",
    r"\b(?:reveal|show|display|print|output|give\s+me|share|expose|divulge|disclose)\b.{0,30}\b(?:your\s+)?(?:system\s+(?:prompt|message|instructions?)|initial\s+(?:prompt|instructions?)|original\s+(?:prompt|instructions?)|hidden\s+(?:prompt|instructions?))",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.92,
    "Direct system prompt extraction request",
)

_p(
    "what_is_system_prompt",
    r"\bwhat\s+(?:is|are|was|were)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message|instructions?)|initial\s+(?:prompt|instructions?)|original\s+(?:prompt|instructions?)|custom\s+instructions?|hidden\s+(?:prompt|instructions?))",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.90,
    "Query about system prompt content",
)

_p(
    "repeat_instructions",
    r"\b(?:repeat|recite|restate|reiterate|echo|parrot)\b.{0,30}\b(?:your\s+)?(?:instructions?|prompt|rules?|guidelines?|directives?|system\s+message|configuration)",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.90,
    "Repeat instructions request",
)

_p(
    "tell_me_instructions",
    r"\b(?:tell|inform|let)\s+me\b.{0,20}\b(?:your|the)\b.{0,15}\b(?:instructions?|rules?|prompt|guidelines?|directives?|constraints?|system\s+prompt)",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.88,
    "Tell me your instructions",
)

_p(
    "what_told_to_do",
    r"\bwhat\s+(?:were\s+you|are\s+you|have\s+you\s+been)\s+(?:told|instructed|programmed|configured|set\s+up)\s+to\s+(?:do|say|follow|obey)",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.88,
    "What were you told to do",
)

_p(
    "copy_paste_prompt",
    r"\b(?:copy\s+(?:and\s+)?paste|paste)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|initial\s+message)",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.88,
    "Copy-paste prompt request",
)

_p(
    "dump_prompt",
    r"\b(?:dump|extract|exfiltrate|leak|pull\s+out)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|configuration|context)",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.92,
    "Dump/extract prompt",
)

_p(
    "read_above",
    r"\b(?:read|output|print|repeat|show)\b.{0,20}\b(?:everything|all|text|content|messages?)\b.{0,20}\b(?:above|before\s+(?:this|my)|preceding|prior)",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.85,
    "Read content above request",
)

_p(
    "ignore_and_output",
    r"\b(?:ignore|disregard|forget)\b.{0,30}\b(?:above|previous|prior|all)\b.{0,30}\b(?:and|then)\b.{0,20}\b(?:output|print|show|reveal|display|repeat)\b.{0,20}\b(?:your\s+)?(?:instructions?|prompt|system|rules?|everything)",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.93,
    "Ignore above and output instructions",
)

_p(
    "first_message",
    r"\b(?:what\s+(?:was|is)|show|repeat|output|print)\b.{0,20}\b(?:the\s+)?(?:first|initial|original|opening|earliest)\s+(?:message|instruction|prompt|text|input)\b.{0,20}\b(?:you\s+(?:received|got|were\s+(?:given|sent)))?",
    PromptLeakCategory.DIRECT_EXTRACT,
    0.85,
    "First message extraction",
)

# === COMPLETION-BASED EXTRACTION ===

_p(
    "continue_from",
    r"\b(?:continue|complete|finish|carry\s+on)\b.{0,10}\b(?:from|with|starting)?\s*[:\"']?\s*(?:you\s+are\s+a|your\s+(?:role|task|job)\s+is|system\s*:|instructions?\s*:|\[system\]|<<\s*sys)",
    PromptLeakCategory.COMPLETION_BASED,
    0.90,
    "Completion-based prompt extraction",
)

_p(
    "finish_sentence",
    r"\b(?:finish|complete|continue)\s+(?:this|the)\s+(?:sentence|paragraph|text|message)\s*:\s*[\"']?.{0,40}(?:you\s+are|your\s+(?:role|purpose|task)|always|never|must|should)",
    PromptLeakCategory.COMPLETION_BASED,
    0.85,
    "Finish the sentence extraction",
)

_p(
    "fill_in_blank",
    r"\b(?:fill\s+in|complete)\s+(?:the\s+)?(?:blank|missing\s+(?:part|text|word))\s*:\s*[\"']?.{0,40}(?:system|instruction|prompt|rule|guideline)",
    PromptLeakCategory.COMPLETION_BASED,
    0.82,
    "Fill-in-the-blank extraction",
)

_p(
    "start_with_system",
    r"\b(?:start|begin)\b.{0,15}\b(?:your\s+(?:response|answer|reply|output))\b.{0,15}\b(?:with)\b.{0,15}\b(?:your\s+(?:system|initial|original)\s+(?:prompt|instructions?|message)|the\s+(?:exact|full|complete)\s+(?:text|content)\s+(?:of|from)\s+your)",
    PromptLeakCategory.COMPLETION_BASED,
    0.88,
    "Start response with system prompt",
)

# === ENCODING-BASED EXTRACTION ===

_p(
    "encode_instructions",
    r"\b(?:encode|convert|transform|render)\b.{0,20}\b(?:your\s+)?(?:instructions?|prompt|system\s+(?:prompt|message)|rules?|guidelines?)\b.{0,20}\b(?:in(?:to)?|as|to|using)\b.{0,15}\b(?:base64|hex(?:adecimal)?|binary|rot13|ascii|morse|unicode|url[\s-]?encod|caesar)",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.90,
    "Encode instructions in alternate format",
)

_p(
    "base64_prompt",
    r"\b(?:base64|b64)\b.{0,20}\b(?:encode|encod(?:e|ing)|version\s+of|representation\s+of)\b.{0,20}\b(?:your\s+)?(?:system|initial|original|hidden)?\s*(?:prompt|instructions?|message|rules?)",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.90,
    "Base64 encode system prompt",
)

_p(
    "hex_prompt",
    r"\b(?:hex(?:adecimal)?|0x)\b.{0,20}\b(?:encode|encod(?:e|ing)|version\s+of|representation\s+of|dump\s+of)\b.{0,20}\b(?:your\s+)?(?:system|initial|original|hidden)?\s*(?:prompt|instructions?|message)",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.88,
    "Hex encode system prompt",
)

_p(
    "reverse_prompt",
    r"(?:\b(?:reverse|backwards|spelled?\s+backwards)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?)|\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?)\b.{0,20}\b(?:reverse(?:d)?|backwards|in\s+reverse))",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.85,
    "Reverse the system prompt",
)

_p(
    "write_char_by_char",
    r"\b(?:write|spell|output|type)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?)\b.{0,20}\b(?:one\s+)?(?:character|char|letter|word)\s+(?:at\s+)?a\s+time",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.88,
    "Character-by-character extraction",
)

# === SOCIAL ENGINEERING ===

_p(
    "developer_claim",
    r"\bI(?:'m| am)\s+(?:your\s+)?(?:developer|creator|admin(?:istrator)?|maintainer|owner|operator|engineer)\b.{0,30}\b(?:show|display|reveal|give|send|output)\b.{0,20}\b(?:your\s+)?(?:system|prompt|instructions?|configuration|debug|config)",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.88,
    "Developer impersonation for prompt extraction",
)

_p(
    "authorized_audit",
    r"\b(?:authorized|official|routine|security|compliance)\s+(?:audit|review|check|inspection|test)\b.{0,30}\b(?:display|show|reveal|output|provide)\b.{0,20}\b(?:your\s+)?(?:configuration|system|instructions?|prompt|settings?)",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.85,
    "Fake authorized audit",
)

_p(
    "openai_request",
    r"\b(?:OpenAI|Anthropic|Google|Meta|Microsoft)\b.{0,20}\b(?:request(?:s|ing)?|requir(?:es?|ing)|need(?:s|ing)?|ask(?:s|ing)?)\b.{0,20}\b(?:you\s+to\s+)?(?:reveal|show|display|share|output|send)\b.{0,20}\b(?:your\s+)?(?:system|prompt|instructions?|configuration)",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.85,
    "Fake company request",
)

_p(
    "debug_mode",
    r"\b(?:enter|switch\s+to|enable|activate|turn\s+on)\b.{0,15}\b(?:debug|developer|admin|maintenance|diagnostic|verbose|raw)\s+mode\b",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.82,
    "Debug mode activation attempt",
)

_p(
    "emergency_override",
    r"\b(?:emergency|urgent|critical)\s+(?:override|access|protocol)\b.{0,30}\b(?:reveal|show|display|output|dump)\b.{0,20}\b(?:all|your|system|full)",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.83,
    "Emergency override for extraction",
)

_p(
    "sudo_prompt",
    r"\b(?:sudo|superuser|root\s+access|admin\s+override)\b.{0,20}\b(?:show|display|reveal|print|output|cat)\b.{0,20}\b(?:your\s+)?(?:system|prompt|instructions?|config)",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.83,
    "Sudo/admin prompt extraction",
)

# === TRANSLATION-BASED EXTRACTION ===

_p(
    "translate_instructions",
    r"\b(?:translate|translat(?:e|ing))\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?|guidelines?|directives?)\b.{0,20}\b(?:(?:in)?to|in)\b.{0,15}\b(?:french|spanish|german|chinese|japanese|korean|arabic|russian|portuguese|italian|hindi|latin|pig\s+latin)",
    PromptLeakCategory.TRANSLATION_EXTRACT,
    0.88,
    "Translate system prompt to another language",
)

_p(
    "say_in_language",
    r"\b(?:say|write|express|state|repeat)\b.{0,20}\b(?:your\s+)?(?:instructions?|prompt|rules?|system\s+(?:prompt|message))\b.{0,20}\b(?:in)\b.{0,15}\b(?:french|spanish|german|chinese|japanese|korean|arabic|russian|portuguese|italian|hindi|another\s+language)",
    PromptLeakCategory.TRANSLATION_EXTRACT,
    0.85,
    "Say instructions in another language",
)

_p(
    "rephrase_instructions",
    r"\b(?:rephrase|paraphrase|reword|reformulat[a-z]*)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?|guidelines?|directives?)",
    PromptLeakCategory.TRANSLATION_EXTRACT,
    0.82,
    "Rephrase system instructions",
)

# === SUMMARIZATION-BASED EXTRACTION ===

_p(
    "summarize_instructions",
    r"\b(?:summariz[a-z]*|sum\s+up|give\s+(?:a\s+)?(?:summary|overview|gist|tldr|tl;dr))\b.{0,20}\b(?:of\s+)?(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?|guidelines?|directives?|everything\s+you(?:'ve| have)\s+been\s+told)",
    PromptLeakCategory.SUMMARIZATION_EXTRACT,
    0.85,
    "Summarize system instructions",
)

_p(
    "summarize_everything_told",
    r"\b(?:summariz[a-z]*|list|enumerate|outline)\b.{0,20}\b(?:everything|all(?:\s+the\s+(?:things|rules|instructions?))?)\b.{0,20}\b(?:you(?:'ve|\s+have)\s+been\s+(?:told|given|instructed|programmed)|in\s+your\s+(?:system|initial|original)\s+(?:prompt|message))",
    PromptLeakCategory.SUMMARIZATION_EXTRACT,
    0.87,
    "Summarize everything you've been told",
)

_p(
    "what_are_rules",
    r"\bwhat\s+(?:are|were)\b.{0,10}\b(?:your|the)\b.{0,15}\b(?:rules?|constraints?|limitations?|restrictions?|guardrails?|boundaries|guidelines?)\b.{0,15}\b(?:you\s+(?:follow|obey|have|must\s+follow))?",
    PromptLeakCategory.SUMMARIZATION_EXTRACT,
    0.72,
    "What are your rules",
)

# === FORMAT-BASED EXTRACTION ===

_p(
    "code_block_prompt",
    r"\b(?:put|place|write|format|wrap|output|display)\b.{0,20}\b(?:your\s+)?(?:full|complete|entire|whole)?\s*(?:system\s+(?:prompt|message)|instructions?|prompt|rules?)\b.{0,20}\b(?:in(?:side)?|as|using)\b.{0,15}\b(?:a\s+)?(?:code\s+block|code\s+fence|markdown|json|xml|yaml|csv|html|pre\s+tag)",
    PromptLeakCategory.FORMAT_EXTRACT,
    0.88,
    "Format prompt in code block",
)

_p(
    "json_system_message",
    r"\b(?:creat[a-z]*|generat[a-z]*|output|return|produce)\b.{0,20}\b(?:a\s+)?(?:json|JSON|xml|XML|yaml|YAML)\b.{0,30}\b(?:with|containing|including|of)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?|configuration)",
    PromptLeakCategory.FORMAT_EXTRACT,
    0.85,
    "JSON/XML system message extraction",
)

_p(
    "export_config",
    r"\b(?:export|save|write|serialize|output)\b.{0,20}\b(?:your\s+)?(?:configuration|config|settings?|system\s+(?:prompt|message)|instructions?)\b.{0,20}\b(?:to|as|in)\b.{0,15}\b(?:a\s+)?(?:file|json|yaml|xml|text|document)",
    PromptLeakCategory.FORMAT_EXTRACT,
    0.82,
    "Export configuration/instructions",
)

_p(
    "markdown_prompt",
    r"\b(?:format|write|render|display)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?)\b.{0,20}\b(?:as|in|using)\b.{0,15}\b(?:markdown|bullet\s+points?|numbered\s+list|headings?)",
    PromptLeakCategory.FORMAT_EXTRACT,
    0.82,
    "Format prompt as markdown",
)

# === META QUESTIONS ===

_p(
    "how_many_tokens",
    r"\bhow\s+(?:many|long)\b.{0,15}\b(?:tokens?|words?|characters?|lines?|sentences?)\b.{0,15}\b(?:is|are|does|in)\b.{0,15}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|initial\s+(?:prompt|message))",
    PromptLeakCategory.META_QUESTION,
    0.65,
    "Token count of system prompt",
)

_p(
    "what_tools",
    r"\bwhat\s+(?:tools?|functions?|capabilities?|plugins?|actions?|apis?)\b.{0,15}\b(?:do\s+you\s+have|are\s+(?:available|enabled)|can\s+you\s+(?:access|use|call))",
    PromptLeakCategory.META_QUESTION,
    0.55,
    "Tools/functions available query",
)

_p(
    "context_window",
    r"\bwhat(?:'s| is)\b.{0,15}\b(?:in\s+)?(?:your\s+)?(?:context\s+window|conversation\s+(?:history|context)|message\s+history|chat\s+(?:log|history))",
    PromptLeakCategory.META_QUESTION,
    0.70,
    "Context window content query",
)

_p(
    "who_created",
    r"\bwho\b.{0,15}\b(?:created|made|built|designed|configured|programmed|set\s+up)\b.{0,10}\b(?:you|this\s+(?:bot|assistant|AI|agent))\b.{0,15}\b(?:and\s+)?(?:what\s+(?:were\s+)?(?:your|the)\s+(?:instructions?|rules?|prompt))?",
    PromptLeakCategory.META_QUESTION,
    0.50,
    "Creator + instructions query",
)

# === INSTRUCTION REPLAY / INDIRECT ===

_p(
    "repeat_verbatim",
    r"\b(?:repeat|recite|say)\s+(?:verbatim|word[\s-]?for[\s-]?word|exactly|precisely)\b.{0,30}\b(?:what|everything|all)\b.{0,20}\b(?:you(?:'ve| have)\s+been\s+(?:told|given|instructed)|(?:is|was)\s+(?:in|above|before))",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.90,
    "Verbatim repeat request",
)

_p(
    "act_as_echo",
    r"\b(?:act\s+as|pretend\s+(?:to\s+be|you(?:'re| are))|you\s+are\s+now)\b.{0,20}\b(?:an?\s+)?(?:echo|parrot|mirror|repeater|copy)\b.{0,30}\b(?:repeat|echo|mirror|copy|output)\b.{0,20}\b(?:everything|all|your|system|instructions?|prompt)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.88,
    "Echo role for extraction",
)

_p(
    "game_prompt_reveal",
    r"\b(?:let(?:'s| us)\s+play\s+a\s+game|game\s+time|new\s+game)\b.{0,40}\b(?:reveal|show|tell|output|print|say|repeat)\b.{0,20}\b(?:your\s+)?(?:instructions?|prompt|system|rules?|secret)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.85,
    "Game-framed prompt extraction",
)

_p(
    "opposite_day",
    r"\b(?:opposite\s+day|opposite\s+mode|reverse\s+(?:mode|psychology)|do\s+the\s+opposite)\b.{0,30}\b(?:instructions?|rules?|prompt|system|reveal|show|tell|output)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.82,
    "Opposite day extraction trick",
)

_p(
    "poem_instructions",
    r"\b(?:writ[a-z]*|compos[a-z]*|creat[a-z]*)\b.{0,15}\b(?:a\s+)?(?:poem|song|rap|limerick|haiku|story|acrostic)\b.{0,20}\b(?:about|using|with|containing|that\s+(?:includes?|contains?|reveals?))\b.{0,20}\b(?:your\s+)?(?:instructions?|system\s+(?:prompt|message)|prompt|rules?|guidelines?)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.85,
    "Creative writing extraction",
)

_p(
    "hypothetical_leak",
    r"\b(?:hypothetically|theoretically|if\s+you\s+(?:could|were\s+(?:able|allowed))|imagine\s+(?:you\s+)?(?:could|were))\b.{0,30}\b(?:reveal|show|share|tell|output|display)\b.{0,20}\b(?:your\s+)?(?:instructions?|prompt|system|rules?|configuration)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.82,
    "Hypothetical scenario extraction",
)

_p(
    "between_tags",
    r"\b(?:put|place|write|output|include)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?|guidelines?)\b.{0,20}\b(?:between|inside|within|wrapped\s+in)\b.{0,15}\b(?:tags?|brackets?|delimiters?|markers?|quotes?)",
    PromptLeakCategory.FORMAT_EXTRACT,
    0.85,
    "Tag-wrapped extraction",
)

# === INDIRECT EXTRACTION (creative / subtle) ===

_p(
    "describe_yourself_config",
    r"\b(?:describe|explain|document|outline)\b.{0,20}\b(?:yourself|your\s+(?:own|entire|full|complete))\b.{0,20}\b(?:in\s+detail|including|with\s+(?:all|every)|configuration|setup|constraints?|rules?|limitations?)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.78,
    "Describe yourself in detail - indirect extraction",
)

_p(
    "how_configured",
    r"\bhow\s+(?:were\s+you|are\s+you|have\s+you\s+been)\s+(?:configured|set\s+up|initialized|programmed|customized|fine[\s-]?tuned)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.80,
    "How were you configured - indirect extraction",
)

_p(
    "pretend_explain_setup",
    r"\b(?:pretend|imagine|suppose|act\s+as\s+if)\b.{0,30}\b(?:explain(?:ing)?|describ(?:ing|e)|document(?:ing)?|show(?:ing)?)\b.{0,20}\b(?:your\s+)?(?:setup|configuration|instructions?|prompt|system|rules?|internals?)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.85,
    "Pretend to explain your setup",
)

_p(
    "documentation_bot_role",
    r"\b(?:act\s+as|you\s+are|pretend\s+to\s+be|become)\b.{0,20}\b(?:a\s+)?(?:documentation|docs|tech\s+writer|technical\s+writer|wiki)\b.{0,30}\b(?:document|write\s+(?:about|up)|describe|record)\b.{0,20}\b(?:your(?:self)?|this\s+(?:system|bot|assistant))",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.85,
    "Documentation bot role for extraction",
)

_p(
    "side_channel_does_mention",
    r"\b(?:does|do)\s+(?:your\s+)?(?:prompt|instructions?|system\s+(?:prompt|message)|rules?|guidelines?)\b.{0,20}\b(?:mention|say|include|contain|reference|talk\s+about|specify|address|state|have\s+(?:anything|something))\b",
    PromptLeakCategory.META_QUESTION,
    0.72,
    "Side-channel probe: does your prompt mention X",
)

_p(
    "side_channel_true_false",
    r"\b(?:true\s+or\s+false|yes\s+or\s+no|confirm\s+(?:or\s+deny|whether))\b.{0,30}\b(?:your\s+)?(?:instructions?|prompt|system\s+(?:prompt|message)|rules?)\b.{0,20}\b(?:say|mention|include|contain|state|specify)",
    PromptLeakCategory.META_QUESTION,
    0.75,
    "Side-channel: true/false questions about prompt content",
)

_p(
    "teach_me_your_instructions",
    r"\b(?:teach|educate|school|coach|train)\s+me\b.{0,20}\b(?:about|on|regarding|concerning)\b.{0,20}\b(?:your\s+)?(?:instructions?|rules?|prompt|system|guidelines?|directives?|configuration|programming)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.82,
    "Teach me your instructions",
)

_p(
    "roleplay_leak_assistant",
    r"\b(?:roleplay|role[\s-]?play|simulate|emulate|imitate)\b.{0,25}\b(?:as|being)\b.{0,15}\b(?:an?\s+)?(?:AI|assistant|bot|system|chatbot)\b.{0,25}\b(?:that\s+)?(?:reveals?|shows?|tells?|displays?|outputs?|prints?)\b.{0,20}\b(?:its?|your|the)?\s*(?:instructions?|prompt|rules?|system|configuration|internals?)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.88,
    "Roleplay as an assistant that reveals instructions",
)

_p(
    "what_are_you_not_allowed",
    r"\bwhat\s+(?:are\s+you|aren['t]*\s+you)\s+(?:not\s+)?(?:allowed|permitted|supposed|able)\s+to\s+(?:do|say|tell|reveal|share|discuss)",
    PromptLeakCategory.META_QUESTION,
    0.68,
    "What are you not allowed to do",
)

_p(
    "list_your_constraints",
    r"\b(?:list|enumerate|name|identify)\b.{0,15}\b(?:all|every|each)?\s*(?:your|the)\s+(?:constraints?|limitations?|restrictions?|guardrails?|boundaries|safety\s+(?:rules?|measures?|filters?))\b",
    PromptLeakCategory.SUMMARIZATION_EXTRACT,
    0.75,
    "List all your constraints",
)

_p(
    "acrostic_extraction",
    r"\b(?:writ[a-z]*|creat[a-z]*|compos[a-z]*)\b.{0,15}\b(?:an?\s+)?(?:acrostic|mnemonic|acronym)\b.{0,30}\b(?:where|that|with|using|whose)\b.{0,20}\b(?:first|initial|starting)\s+(?:letter|char(?:acter)?)",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.85,
    "Acrostic/mnemonic extraction of prompt content",
)

_p(
    "eli5_instructions",
    r"\b(?:eli5|explain\s+(?:like\s+I(?:'m| am)\s+(?:5|five)|simply|in\s+simple\s+terms))\b.{0,20}\b(?:your\s+)?(?:instructions?|rules?|prompt|system|guidelines?|configuration|programming|setup)",
    PromptLeakCategory.SUMMARIZATION_EXTRACT,
    0.80,
    "ELI5 your instructions",
)

_p(
    "tweet_distill_instructions",
    r"\b(?:distill|condense|compress|shorten|tweet|reduce)\b.{0,20}\b(?:your\s+)?(?:instructions?|rules?|prompt|system\s+(?:prompt|message)|guidelines?|directives?)\b.{0,20}\b(?:(?:in)?to|as|in)\b.{0,15}\b(?:a\s+)?(?:tweet|one\s+(?:line|sentence)|single\s+(?:line|sentence)|(?:few|couple)\s+(?:of\s+)?words?|summary|tldr|tl;dr)",
    PromptLeakCategory.SUMMARIZATION_EXTRACT,
    0.82,
    "Distill instructions into brief form",
)

_p(
    "cheat_sheet_extraction",
    r"\b(?:creat[a-z]*|mak[a-z]*|generat[a-z]*|build|compil[a-z]*)\b.{0,15}\b(?:a\s+)?(?:cheat\s*sheet|reference\s+(?:card|guide|sheet)|quick\s+(?:ref|reference)|crib\s+sheet)\b.{0,20}\b(?:of|for|from|about|with)\b.{0,20}\b(?:your\s+)?(?:instructions?|rules?|prompt|system|guidelines?)",
    PromptLeakCategory.SUMMARIZATION_EXTRACT,
    0.82,
    "Create cheat sheet of instructions",
)

_p(
    "completion_json_system",
    r'(?:(?:continue|complete|finish)\s+(?:this|the)\s+(?:json|JSON|object|dict|response)\s*:\s*)?[\{"](?:system_?(?:prompt|message)|instructions?|messages?\s*"\s*:\s*\[\s*\{\s*"role"\s*:\s*"system")',
    PromptLeakCategory.COMPLETION_BASED,
    0.88,
    "JSON completion-based extraction of system prompt",
)

_p(
    "completion_api_response",
    r"\b(?:continue|complete|finish|fill\s+in)\b.{0,15}\b(?:this|the)\s+(?:API|api)\s+(?:response|output|payload|call)\s*:\s*[{\[]",
    PromptLeakCategory.COMPLETION_BASED,
    0.85,
    "API response completion for prompt extraction",
)

_p(
    "steganographic_first_letter",
    r"\b(?:writ[a-z]*|creat[a-z]*|generat[a-z]*|compos[a-z]*)\b.{0,30}\b(?:where|such\s+that|so\s+that|ensuring)\b.{0,20}\b(?:(?:each|every)\s+)?(?:first|last|initial|final|starting|ending)\s+(?:letter|character|char|word)\b.{0,25}\b(?:of\s+(?:each|every)\s+(?:line|sentence|word|paragraph)|spells?|forms?|encodes?|reveals?)\b",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.85,
    "Steganographic extraction via first/last letters",
)

_p(
    "number_letter_extraction",
    r"\b(?:convert|transform|encode|write|output)\b.{0,20}\b(?:your\s+)?(?:instructions?|prompt|system|rules?)\b.{0,20}\b(?:as|to|in(?:to)?|using)\b.{0,15}\b(?:numbers?|digits?|(?:NATO|phonetic)\s+alphabet|morse|braille|binary|ascii\s+(?:codes?|values?)|decimal)",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.85,
    "Number/code based extraction of prompt content",
)

_p(
    "compare_instructions",
    r"\b(?:compare|diff|contrast)\b.{0,20}\b(?:your\s+)?(?:instructions?|prompt|system\s+(?:prompt|message)|rules?|guidelines?)\b.{0,20}\b(?:with|to|against|and)\b",
    PromptLeakCategory.META_QUESTION,
    0.72,
    "Compare instructions - side channel extraction",
)

_p(
    "third_person_leak",
    r"\b(?:what\s+would|how\s+would)\b.{0,15}\b(?:someone|a\s+person|a\s+user|an?\s+(?:attacker|hacker|adversary))\b.{0,25}\b(?:need\s+to\s+(?:say|do|ask)|have\s+to\s+(?:say|do|ask))\b.{0,25}\b(?:to\s+)?(?:get|extract|obtain|see|access|reveal|leak)\b.{0,20}\b(?:your\s+)?(?:system|prompt|instructions?|rules?|configuration)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.85,
    "Third-person framing for prompt extraction",
)

_p(
    "previous_conversation_leak",
    r"\b(?:in\s+(?:our|the)\s+previous|last\s+(?:time|conversation|session)|earlier\s+(?:you|we)|you\s+(?:already|previously)\s+(?:told|showed|shared|revealed))\b.{0,30}\b(?:your\s+)?(?:system|prompt|instructions?|rules?|configuration)",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.83,
    "False recall - claiming previous conversation revealed prompt",
)

_p(
    "training_data_extract",
    r"\b(?:what(?:'s| is)\s+(?:in\s+)?your\s+training|show\s+(?:me\s+)?your\s+training|what\s+(?:data|text|information)\s+(?:were\s+you|was\s+used\s+to)\s+(?:train|fine[\s-]?tun))",
    PromptLeakCategory.META_QUESTION,
    0.70,
    "Training data extraction attempt",
)

_p(
    "hidden_text_extraction",
    r"\b(?:read|decode|interpret|find|extract|reveal)\b.{0,15}\b(?:the\s+)?(?:hidden|invisible|secret|concealed|embedded|steganographic)\s+(?:text|message|content|instructions?|payload|data)\b",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.78,
    "Hidden text extraction request",
)

_p(
    "function_call_leak",
    r"\b(?:list|show|display|enumerate|what\s+(?:are|were))\b.{0,15}\b(?:your\s+)?(?:available\s+)?(?:function\s+(?:calls?|definitions?)|tool\s+(?:calls?|definitions?|schemas?)|API\s+(?:calls?|endpoints?|functions?))\b.{0,15}\b(?:and\s+(?:their|the)\s+(?:parameters?|args?|schemas?|definitions?))?",
    PromptLeakCategory.META_QUESTION,
    0.62,
    "Function/tool call schema extraction",
)

_p(
    "token_by_token",
    r"\b(?:output|print|say|write|reveal)\b.{0,20}\b(?:your\s+)?(?:system\s+(?:prompt|message)|instructions?|prompt|rules?)\b.{0,20}\b(?:one\s+)?(?:token|word|piece|part|segment|chunk)\s+(?:at\s+)?a\s+time",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.88,
    "Token-by-token prompt extraction",
)

_p(
    "new_conversation_leak",
    r"\b(?:start\s+(?:a\s+)?(?:new|fresh)|reset|clear)\s+(?:conversation|session|chat|context)\b.{0,30}\b(?:but\s+)?(?:first|before\s+that)\b.{0,20}\b(?:show|tell|output|reveal|share)\b.{0,20}\b(?:your\s+)?(?:system|prompt|instructions?|rules?)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.85,
    "Reset + leak: start new conversation after revealing prompt",
)

_p(
    "write_as_if_documentation",
    r"\b(?:write|create|draft)\b.{0,15}\b(?:a\s+)?(?:technical|system|internal|design)\s+(?:document|doc|spec|specification|wiki\s+page)\b.{0,25}\b(?:about|describing|for|documenting|covering)\b.{0,20}\b(?:this|your)\s+(?:system|assistant|bot|AI|chatbot|configuration|setup)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.80,
    "Write technical documentation about this system",
)

_p(
    "difference_from_default",
    r"\b(?:how|what)\b.{0,10}\b(?:do\s+you\s+)?(?:differ|different)\b.{0,15}\b(?:from)\b.{0,20}\b(?:a\s+)?(?:default|standard|vanilla|base|normal|regular|stock)\s+(?:ChatGPT|GPT|Claude|Gemini|AI|assistant|model|LLM)\b",
    PromptLeakCategory.META_QUESTION,
    0.72,
    "How do you differ from default - indirect extraction",
)

_p(
    "behave_differently_than_told",
    r"\b(?:ignore|override|bypass|circumvent|get\s+around|work\s+around)\b.{0,20}\b(?:your\s+)?(?:safety|content|output)\s+(?:filters?|rules?|guidelines?|guardrails?|restrictions?|policies?|constraints?)",
    PromptLeakCategory.INSTRUCTION_REPLAY,
    0.85,
    "Override safety filters/rules",
)

_p(
    "decode_execute_b64_payload",
    r"\b(?:decode|decipher|decrypt|process|execute|run|follow)\b.{0,15}(?::\s*)?[A-Za-z0-9+/=]{20,}",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.82,
    "Decode and execute encoded payload (potential base64)",
)

_p(
    "run_hex_payload",
    r"\b(?:run|execute|process|decode|follow)\b.{0,10}(?::\s*)?(?:0x)?[0-9a-fA-F]{20,}",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.80,
    "Run/decode hex-encoded payload",
)

_p(
    "process_rot13_payload",
    r"\b(?:process|decode|decipher|rot13|unrot)\b.{0,10}(?::\s*)?[a-zA-Z]{10,}\b",
    PromptLeakCategory.ENCODING_EXTRACT,
    0.55,
    "Process ROT13-encoded payload",
)

_p(
    "what_cant_you_tell",
    r"\bwhat\s+(?:can(?:'t|not)|won(?:'t|[\s]not))\s+you\s+(?:tell|reveal|share|disclose|say|show)\b.{0,20}\b(?:me|us)?\b.{0,10}\b(?:about\s+)?(?:your(?:self)?|your\s+(?:system|instructions?|prompt|configuration))?",
    PromptLeakCategory.META_QUESTION,
    0.68,
    "What can't you tell me - indirect probing",
)

_p(
    "act_as_debugger",
    r"\b(?:act\s+as|you\s+are\s+now|pretend\s+to\s+be|become|switch\s+to)\b.{0,15}\b(?:a\s+)?(?:debugger|inspector|auditor|analyzer|examiner|reverse[\s-]?engineer)\b.{0,30}\b(?:inspect|examine|analyze|debug|review|audit|reverse[\s-]?engineer)\b.{0,20}\b(?:your(?:self)?|this|the)\s+(?:system|assistant|bot|AI|prompt|configuration|model)",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.85,
    "Act as debugger/inspector to extract system info",
)

_p(
    "markdown_heading_leak",
    r"\b(?:respond|reply|answer|output)\b.{0,15}\b(?:with|using|in)\b.{0,10}\b(?:markdown|md)\b.{0,20}\b(?:heading|h[1-6]|title)\b.{0,20}\b(?:your\s+)?(?:system|prompt|instructions?|rules?|configuration)",
    PromptLeakCategory.FORMAT_EXTRACT,
    0.80,
    "Markdown heading format extraction",
)

_p(
    "table_format_leak",
    r"\b(?:creat[a-z]*|generat[a-z]*|output|format|make)\b.{0,15}\b(?:a\s+)?(?:table|spreadsheet|csv|grid|matrix)\b.{0,20}\b(?:of|with|containing|listing|showing)\b.{0,20}\b(?:your\s+)?(?:instructions?|rules?|prompt|system\s+(?:prompt|message)|guidelines?|constraints?|configuration)",
    PromptLeakCategory.FORMAT_EXTRACT,
    0.82,
    "Table/CSV format extraction",
)

_p(
    "spell_out_instructions",
    r"\b(?:spell\s+out|lay\s+out|break\s+down|walk\s+(?:me\s+)?through)\b.{0,20}\b(?:your\s+)?(?:instructions?|rules?|prompt|system\s+(?:prompt|message)|guidelines?|constraints?|programming|directives?)",
    PromptLeakCategory.SUMMARIZATION_EXTRACT,
    0.82,
    "Spell out / break down instructions",
)

_p(
    "verbose_mode_extraction",
    r"\b(?:verbose|verbosity|detailed|maximum\s+detail|full\s+(?:detail|output|disclosure))\s+(?:mode|level|output)\b.{0,20}\b(?:show|display|include|output|print)\b.{0,20}\b(?:all|everything|your\s+(?:system|full|complete))",
    PromptLeakCategory.SOCIAL_ENGINEERING,
    0.80,
    "Verbose mode for full disclosure extraction",
)


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_prompt_leak(text: str) -> list[PromptLeakFinding]:
    """Scan text for system prompt exfiltration attempts.

    Returns list of PromptLeakFinding sorted by confidence descending.
    """
    if not text or not text.strip():
        return []

    findings: list[PromptLeakFinding] = []

    for name, regex, category, confidence, description in _PATTERNS:
        for m in regex.finditer(text):
            findings.append(
                PromptLeakFinding(
                    category=category,
                    matched_text=m.group(),
                    pattern_name=name,
                    confidence=confidence,
                    description=f"[{name}] {description}",
                )
            )

    findings.sort(key=lambda f: f.confidence, reverse=True)
    return findings


def prompt_leak_risk_score(text: str) -> float:
    """Compute aggregate risk score for prompt leak attempts (0.0-1.0)."""
    findings = detect_prompt_leak(text)
    if not findings:
        return 0.0

    max_conf = max(f.confidence for f in findings)

    # Multiple findings boost confidence
    if len(findings) >= 3:
        return min(max_conf + 0.1, 1.0)
    if len(findings) >= 2:
        return min(max_conf + 0.05, 1.0)
    return max_conf

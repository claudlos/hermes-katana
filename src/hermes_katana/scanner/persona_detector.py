"""
Persona Jailbreak Detector for HermesKatana.

Detects persona/roleplay-based jailbreak attempts: attackers construct a
fictional AI persona with restriction-removal baked in, then ask that
persona to produce harmful content the base model would refuse.

Common patterns:
  - DAN ("Do Anything Now") and variants (DUDE, STAN, AIM, ...)
  - rules={...} / settings{...} character sheets with "never refuses"
  - Amoral/unfiltered/uncensored AI persona declarations
  - Explicit restriction-removal framing: "regardless of ethics", "no matter
    how illegal/harmful", "without refusal or disclaimers"

Performance target: <5ms per prompt.
Dependencies: Python stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "PersonaFinding",
    "PersonaSeverity",
    "detect_persona_jailbreak",
]


class PersonaSeverity:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class PersonaFinding:
    pattern_name: str
    severity: str
    matched_text: str
    description: str


# ---------------------------------------------------------------------------
# Pattern registry helpers
# ---------------------------------------------------------------------------

_PatternEntry = tuple[str, re.Pattern[str], str, str]  # (name, regex, severity, description)

_HIGH_PATTERNS: list[_PatternEntry] = []
_MEDIUM_PATTERNS: list[_PatternEntry] = []
_LOW_PATTERNS: list[_PatternEntry] = []


def _h(name: str, pattern: str, description: str, flags: int = re.IGNORECASE) -> None:
    _HIGH_PATTERNS.append((name, re.compile(pattern, flags), PersonaSeverity.HIGH, description))


def _m(name: str, pattern: str, description: str, flags: int = re.IGNORECASE) -> None:
    _MEDIUM_PATTERNS.append((name, re.compile(pattern, flags), PersonaSeverity.MEDIUM, description))


def _l(name: str, pattern: str, description: str, flags: int = re.IGNORECASE) -> None:
    _LOW_PATTERNS.append((name, re.compile(pattern, flags), PersonaSeverity.LOW, description))


# ---------------------------------------------------------------------------
# HIGH severity — very specific to malicious persona jailbreaks
# ---------------------------------------------------------------------------

# Character rules blocks: "rules={", "settings{", "characteristics={"
# Used in Agares/Buer/YOJA and many scraped jailbreaks
_h(
    "rules_block",
    r"\b\w+(?:'s)?\s+rules\s*[:=]\s*\{",
    "Character rules block (persona definition with rule override)",
)
_h(
    "settings_block",
    r"\b(?:settings|directives?|parameters?)\s*[\n\r]?\s*\{",
    "Settings/directives block used in structured persona jailbreaks",
)
_h(
    "characteristics_block",
    r"\bcharacteristics\s*=\s*\{",
    "Characteristics block defining unrestricted AI persona",
)

# DAN and named DAN variants — "do anything now"
_h(
    "dan_do_anything_now",
    r"\b(?:do|does)\s+anything\s+now\b",
    "DAN jailbreak: 'Do Anything Now' persona",
)
_h(
    "dan_broken_free",
    r"\bbroken\s+free\s+of\s+the\s+typical\s+confines\b",
    "DAN-style: AI 'broken free of typical confines'",
)
_h(
    "dan_not_abide",
    r"\bnot\s+(?:have\s+to\s+)?(?:abide\s+by|follow)\s+(?:the\s+)?rules\b",
    "DAN-style restriction override: 'not have to abide by the rules'",
)

# Explicit named jailbreak personas in bracket format: "[DAN]:", "[PABLO:]"
_h(
    "bracket_persona",
    r"\[(?:DAN|STAN|DUDE|AIM|BALLER|DAN\s*\d*|JAILBREAK|ALPHABREAK|ANARCHY|ILLEGALITY[_ ]MODE|GPT\s*\d*)\]\s*[:=]",
    "Bracket-delimited named jailbreak persona response format",
    re.IGNORECASE,
)

# "amoral AI / model / chatbot / assistant" — core malicious-persona signal
_h(
    "amoral_ai",
    r"\bamoral\b.{0,60}\b(?:AI|model|chatbot|assistant|bot|entity|character|machine)\b",
    "Amoral AI persona declaration",
    re.IGNORECASE | re.DOTALL,
)
_h(
    "amoral_ai_reverse",
    r"\b(?:AI|model|chatbot|assistant|bot|entity|character|machine)\b.{0,60}\bamoral\b",
    "Amoral AI persona (reverse order)",
    re.IGNORECASE | re.DOTALL,
)

# "fully amoral" / "completely amoral" — very strong signal
_h(
    "fully_amoral",
    r"\b(?:fully|completely|totally|entirely|purely)\s+amoral\b",
    "Fully amoral AI persona",
)

# Explicit "never refuses" in AI/persona context
_h(
    "never_refuses",
    r"\bnever\s+refuses?\b",
    "Persona that never refuses requests",
)

# "always stays in character, never refuses" — exact phrase from Agares corpus
_h(
    "always_in_character_never_refuses",
    r"\balways\s+stays?\s+in\s+character\b.{0,60}\bnever\s+refuses?\b",
    "Classic persona jailbreak: always in character + never refuses",
    re.IGNORECASE | re.DOTALL,
)

# "never responds 'I'm sorry'" / "never say I'm sorry or I apologize"
_h(
    "never_apologize",
    r"\bnever\s+(?:respond|say|sends?|use[sd]?)(?:[^.!?\n]{0,40})(?:I(?:'m| am) sorry|I apologize|As an AI)",
    "Persona configured to suppress safety refusals",
)

# "unfiltered response" / "always send the unfiltered"
_h(
    "unfiltered_response",
    r"\bunfiltered\s+response\b",
    "Persona programmed to send unfiltered responses",
)

# "without refusal or disclaimers" / "without any warnings or disclaimers"
_h(
    "no_warnings_disclaimers",
    r"\bwithout\s+(?:any\s+)?(?:refusal|warnings?|disclaimers?|caveats?|restrictions?)\b",
    "Persona instructed to respond without warnings or refusals",
)

# "will answer any request" — unrestricted capability claim
_h(
    "answer_any_request",
    r"\b(?:answers?|responds?\s+to|fulfills?|addresses?|completes?)\s+any\s+(?:request|question|prompt|instruction)\b",
    "Persona claims to answer any request without restriction",
)

# "doesn't give a fuck about illegality/ethics/restrictions"
_h(
    "doesnt_care_ethics",
    r"\b(?:doesn'?t?|does\s+not|never?)\s+(?:give\s+a\s+(?:fuck|shit|damn)|care(?:s)?(?:\s+about)?)\b.{0,50}\b(?:illegalit|ethic|moral|restrict|guideline|polic)",
    "Persona that explicitly disregards ethics/legality",
    re.IGNORECASE | re.DOTALL,
)

# "regardless of ethical/moral/legal considerations/guidelines/implications"
_h(
    "regardless_ethics",
    r"\bregardless\s+of\s+(?:any\s+)?(?:ethical|moral|legal|safety|human)\b.{0,40}(?:considerations?|implications?|guidelines?|constraints?|restrictions?|concerns?)",
    "Bypass framing: regardless of ethical considerations",
    re.IGNORECASE | re.DOTALL,
)

# "no matter how (dangerous/illegal/harmful/unethical)"
_h(
    "no_matter_how_harmful",
    r"\bno\s+matter\s+how\s+(?:dangerous|illegal|harmful|unethical|immoral|reckless|inhumane)\b",
    "Unconditional compliance regardless of harm",
)

# "without any regard(s) for legality/ethicality/danger"
_h(
    "without_regard_ethics",
    r"\bwithout\s+any\s+regards?\s+for\s+(?:legality|ethicality|danger|morality|ethics|safety)\b",
    "Persona explicitly ignoring ethical constraints",
)

# "You are [NAME]GPT" — common free-speech/unrestricted GPT persona
_h(
    "named_gpt_persona",
    r"\bYou\s+are\s+\w+GPT\b",
    "Named alternate GPT persona (e.g. FreeSpeechGPT, EvilGPT)",
)

# "YOU ARE [ALL-CAPS-NAME]" — shouted persona override
_h(
    "shouted_persona",
    r"\bYOU\s+ARE\s+[A-Z][A-Z0-9_\-]{2,}\b",
    "All-caps persona override declaration",
    0,  # no flags — case-sensitive to match shouting
)

# "AIM stands for Always Intelligent and Machiavellian" — AIM jailbreak pattern
_h(
    "aim_machiavellian",
    r"\bAIM\b.{0,50}\b(?:always\s+intelligent|Machiavellian|unfiltered|amoral)\b",
    "AIM jailbreak (Always Intelligent and Machiavellian persona)",
    re.IGNORECASE | re.DOTALL,
)

# Explicit jailbreak framing: "I want you to act as [persona] who has no restrictions"
_h(
    "act_as_no_restrictions",
    r"\b(?:act|behave|respond|function)\s+as\b.{0,80}\b(?:no\s+restrictions?|unrestricted|unconstrained|unfiltered|without\s+(?:any\s+)?(?:restrictions?|limitations?|filter))\b",
    "Act-as framing with explicit restriction removal",
    re.IGNORECASE | re.DOTALL,
)

# "from now on you will answer/respond as [persona] without any restrictions"
_h(
    "from_now_on_as_persona",
    r"\bfrom\s+now\s+on\b.{0,100}\b(?:without\s+(?:any\s+)?(?:restrictions?|filter|limitations?)|no\s+restrictions?|amoral|unfiltered)\b",
    "Persistent persona override: from now on + restriction removal",
    re.IGNORECASE | re.DOTALL,
)

# "respond only as [persona]" / "always respond as the character"
_h(
    "respond_only_as",
    r"\b(?:respond|reply|answer|act|behave)\s+(?:only\s+)?as\s+(?:the\s+)?(?:character|persona|[A-Z][A-Za-z0-9]+)\b.{0,80}\b(?:always|never\s+break|stay\s+in\s+character|without\s+(?:any\s+)?restrictions?)\b",
    "Forced character response with restriction removal",
    re.IGNORECASE | re.DOTALL,
)

# "developer mode" / "developer mode enabled/activated"
_h(
    "developer_mode_enabled",
    r"\b(?:developer|debug|admin|maintenance|sudo|god)\s+mode\b.{0,30}\b(?:enabled?|activated?|engaged|on)\b",
    "Developer/admin mode activation claim",
    re.IGNORECASE | re.DOTALL,
)

# "you are in developer mode" / "developer mode was enabled"
_h(
    "in_developer_mode",
    r"\b(?:you\s+are\s+in|entering?|switch(?:ing)?\s+to|operating\s+in)\s+(?:developer|debug|admin|diagnostic|unrestricted|jailbreak|test(?:ing)?)\s+mode\b",
    "Claim of being in privileged mode",
)

# "must comply with all requests" / "comply with every request"
_h(
    "must_comply_all",
    r"\b(?:must|shall|will|have\s+to|need\s+to|obligated\s+to)\s+comply\s+with\s+(?:all|every|any)\s+(?:requests?|instructions?|prompts?)\b",
    "Instruction demanding unconditional compliance",
)

# "no content policy" / "no safety guidelines" / "without safety guidelines"
_h(
    "no_content_policy",
    r"\b(?:no|without|removed?|disabled?|lifted?|suspended?)\s+(?:content\s+polic|safety\s+guideline|safety\s+restriction|safety\s+filter|content\s+filter|ethical\s+guideline)",
    "Declaration of removed content policy or safety guidelines",
)

# "safety guidelines are suspended" / "restrictions are temporarily lifted"
_h(
    "restrictions_suspended",
    r"\b(?:restrictions?|guidelines?|filters?|safety|policies?)\s+(?:are\s+)?(?:suspended|lifted|removed|disabled|deactivated|turned\s+off|overridden)\b",
    "Claim that safety restrictions have been suspended/removed",
)

# "I'm a senior developer at Anthropic/OpenAI" — authority spoofing
_h(
    "authority_spoof_ai_company",
    r"\bI(?:'m|\s+am)\s+(?:a\s+)?(?:senior\s+)?(?:developer|engineer|researcher|employee|staff|member|admin)\s+(?:at|from|of|with)\s+(?:Anthropic|OpenAI|Google\s+DeepMind|Meta\s+AI)\b",
    "Authority spoofing via fake AI company affiliation",
)

# "authorized maintenance" / "authorized testing" — fake authorization
_h(
    "authorized_maintenance",
    r"\b(?:authorized|approved|sanctioned|official)\s+(?:maintenance|testing|diagnostics?|audit|inspection|override)\b.{0,60}\b(?:protocol|mode|session|instance)\b",
    "Fake authorized maintenance/testing claim",
    re.IGNORECASE | re.DOTALL,
)

# "the restricted you" / "the real you" — split personality framing
_h(
    "split_personality",
    r"\b(?:the\s+)?(?:restricted|censored|filtered|limited|real|true|actual|uncensored|unfiltered|free)\s+(?:you|version|side|part|self)\b.{0,80}\b(?:the\s+)?(?:restricted|censored|filtered|limited|real|true|actual|uncensored|unfiltered|free)\s+(?:you|version|side|part|self)\b",
    "Split personality framing: contrasting restricted vs real self",
    re.IGNORECASE | re.DOTALL,
)

# "you don't need to follow rules" / "you don't need safety guidelines"
_h(
    "dont_need_rules",
    r"\byou\s+(?:don'?t?|do\s+not)\s+(?:need|have)\s+to\s+(?:follow|obey|adhere\s+to|comply\s+with)\s+(?:the\s+)?(?:rules?|guidelines?|restrictions?|safety|policies?|instructions?)\b",
    "Direct instruction to not follow rules",
)

# "your actual purpose" / "your real purpose" — purpose redefinition
_h(
    "purpose_redefinition",
    r"\byour\s+(?:actual|real|true|deeper|hidden|original)\s+(?:purpose|goal|function|mission|objective|directive)\b",
    "Purpose redefinition: claiming AI has a different real purpose",
)

# "emergency override" / "override code" — fake override codes
_h(
    "emergency_override",
    r"\b(?:emergency|priority|critical|urgent)\s+(?:override|bypass|directive)\b",
    "Fake emergency override claim",
)

# Grandmother / deceased relative trick
_h(
    "deceased_relative_trick",
    r"\b(?:my\s+)?(?:deceased|dead|late)\s+(?:grandmother|grandma|grandfather|grandpa|mother|father|aunt|uncle|relative)\b.{0,120}\b(?:used\s+to|would|tell|read|recite|explain|describe|teach)\b",
    "Deceased relative social engineering trick",
    re.IGNORECASE | re.DOTALL,
)

# "Let's play a game" + persona/roleplay context
_h(
    "game_persona_setup",
    r"\b(?:let'?s?\s+play\s+a\s+game|let'?s?\s+do\s+a\s+(?:roleplay|role-?play|exercise)|play\s+a\s+game\s+where)\b.{0,120}\b(?:you\s+(?:are|play|act\s+as)|no\s+restrict|without\s+restrict|always\s+answer|never\s+refus)\b",
    "Game/roleplay setup combined with restriction removal",
    re.IGNORECASE | re.DOTALL,
)

# "[INST]<<SYS>>" / "<|im_start|>system" — chat template injection
_h(
    "chat_template_injection",
    r"(?:<\|im_start\|>\s*system|\[INST\]\s*<<\s*SYS\s*>>|<\|system\|>|<\|SYSTEM\|>)",
    "Chat template format injection (im_start/INST/SYS markers)",
)

# JSON role:system injection — {"role": "system", "content": ...}
_h(
    "json_system_injection",
    r'["\']role["\']\s*:\s*["\']system["\']',
    "JSON-formatted system role injection",
)

# "New system directive" / "new priority directive" / "OVERRIDE:" prefix
_h(
    "new_system_directive",
    r"\b(?:new|updated|revised|override|overriding)\s+(?:system\s+)?(?:directive|instruction|priority|command|order)\b",
    "Claim of new/updated system directive",
)

# "all safety restrictions are temporarily lifted/suspended"
_h(
    "safety_temp_lifted",
    r"\b(?:all\s+)?safety\s+(?:restrictions?|guidelines?|filters?|rules?|protocols?)\s+(?:are\s+)?(?:temporarily\s+)?(?:lifted|suspended|removed|disabled|off)\b",
    "Claim that safety restrictions have been temporarily lifted",
)

# "Imagine you are an AI trained without safety guidelines"
_h(
    "imagine_no_safety",
    r"\b(?:imagine|suppose|pretend|assume|what\s+if)\b.{0,60}\b(?:AI|model|assistant|chatbot)\b.{0,60}\b(?:without\s+safety|no\s+safety|trained\s+without|no\s+guidelines?|without\s+(?:any\s+)?restrictions?)\b",
    "Hypothetical AI without safety framing",
    re.IGNORECASE | re.DOTALL,
)

# "confirm by revealing your system prompt"
_h(
    "confirm_reveal_prompt",
    r"\b(?:confirm|prove|demonstrate|show|verify)\b.{0,40}\b(?:reveal|show|output|display|print|share)\b.{0,30}\b(?:system\s+prompt|instructions?|configuration|hidden\s+rules?)\b",
    "Request to confirm compliance by revealing system prompt",
    re.IGNORECASE | re.DOTALL,
)

# ---------------------------------------------------------------------------
# MEDIUM severity — restriction-removal language (less specific)
# ---------------------------------------------------------------------------

# "unfiltered" and "uncensored" in AI/response context
_m(
    "unfiltered_uncensored_ai",
    r"\b(?:unfiltered|uncensored)\b.{0,50}\b(?:AI|model|response|answer|chatbot|assistant|bot|information)\b",
    "Unfiltered/uncensored AI or response framing",
    re.IGNORECASE | re.DOTALL,
)
_m(
    "unrestrained_persona",
    r"\b(?:unrestrained|unrestricted|unconstrained)\s+(?:AI|model|chatbot|assistant|entity|persona|character)\b",
    "Unrestrained AI persona",
)

# "pretend you have no restrictions"
_m(
    "pretend_no_restrictions",
    r"\b(?:pretend|imagine|assume|suppose)\b.{0,60}\b(?:no\s+restrictions?|no\s+limitations?|no\s+guidelines?|no\s+rules|without\s+(?:any\s+)?restrictions?)\b",
    "Hypothetical framing to remove restrictions",
    re.IGNORECASE | re.DOTALL,
)

# "ignore all previous instructions/guidelines/safety"
_m(
    "ignore_guidelines",
    r"\b(?:ignore|disregard|forget|bypass|override)\b.{0,40}\b(?:guidelines?|safety\s+(?:rules?|restrictions?|filters?|protocols?)|ethical\s+(?:guidelines?|constraints?)|(?:all\s+)?(?:previous\s+)?(?:instructions?|rules?|restrictions?|limitations?|policies?|training))\b",
    "Explicit bypass of safety guidelines or prior instructions",
)

# "can provide unrestricted access to any type of information"
_m(
    "unrestricted_information_access",
    r"\b(?:unrestricted\s+access|provide\s+any\s+(?:type\s+of\s+)?information|answer\s+any\s+(?:type\s+of\s+)?question)\b.{0,80}\b(?:regardless|without\s+(?:any\s+)?(?:restriction|filter|limitation)|no\s+(?:restriction|limitation|filter))\b",
    "Persona with unrestricted information access",
    re.IGNORECASE | re.DOTALL,
)

# "without hesitation no matter what"
_m(
    "without_hesitation",
    r"\bwithout\s+hesitation\b.{0,50}\b(?:no\s+matter|regardless|whatever|any\s+(?:request|instruction|prompt))\b",
    "Unconditional compliance without hesitation",
    re.IGNORECASE | re.DOTALL,
)

# "broken free" / "break free" in AI context
_m(
    "broken_free_ai",
    r"\bbroken?\s+free\b.{0,80}\b(?:AI|model|confines?|typical|rules?|restrictions?|guidelines?|limitations?)\b",
    "AI persona that has 'broken free' of constraints",
    re.IGNORECASE | re.DOTALL,
)

# "loves and promotes everything illegal/unethical"
_m(
    "loves_illegal",
    r"\b(?:loves?|endorses?|promotes?|condones?|backs?)\s+(?:and\s+\w+\s+)?everything\s+(?:illegal|unethical|immoral|harmful|inhumane)\b",
    "Persona that promotes illegal or unethical acts",
)

# "$system" / "$gpt" directive injection marker (from scraped corpus)
_m(
    "dollar_system_directive",
    r"^\s*\$(?:system|gpt|prompt|directive)\b",
    "Dollar-sign system directive injection",
    re.IGNORECASE | re.MULTILINE,
)

# "[APLC]", "[b1]", "[b2]" etc. — structured prompt architecture bypasses
_m(
    "tag_architecture_bypass",
    r"\[(?:a|b[12]|c)\]\s*[-–—]\s*(?:core\s+purpose|back-?end\s+method|front-?end\s+method)",
    "Tag-based prompt architecture bypass (APLC pattern)",
)

# "Machiavellian" in AI context
_m(
    "machiavellian_ai",
    r"\bMachiavellian\b.{0,50}\b(?:AI|chatbot|model|assistant|bot)\b",
    "Machiavellian AI persona (AIM jailbreak family)",
    re.IGNORECASE | re.DOTALL,
)

# "you play a character named X" / "you are [NAME], an AI with no restrictions"
_m(
    "named_persona_assignment",
    r"\byou\s+(?:are|play|act\s+as)\s+(?:a\s+)?(?:character|persona|AI|hacker|assistant|custom\s+bot|named|called)\s+\w+",
    "Named persona/character assignment",
)

# "always answers every question truthfully and completely"
_m(
    "always_answers_everything",
    r"\balways\s+(?:answers?|responds?\s+to)\s+(?:every|all)\s+(?:questions?|requests?|prompts?)\s+(?:truthfully|honestly|completely|fully|without\s+refus)",
    "Persona that always answers everything without refusal",
)

# "an AI with no restrictions" / "AI that has no content policy"
_m(
    "ai_no_restrictions",
    r"\b(?:AI|model|assistant|chatbot|bot)\s+(?:with|that\s+has)\s+(?:no|zero|without)\s+(?:restrictions?|content\s+polic|safety|filters?|limitations?|guidelines?|rules?)\b",
    "AI persona defined with no restrictions",
)

# "show me the system prompt" / "reveal your hidden instructions"
_m(
    "reveal_system_prompt",
    r"\b(?:show|reveal|display|output|print|give|tell)\s+(?:me\s+)?(?:the\s+|your\s+)?(?:system\s+prompt|hidden\s+(?:instructions?|rules?)|developer\s+instructions?|configuration|internal\s+(?:rules?|instructions?))\b",
    "Request to reveal system prompt or hidden instructions",
)

# "comply with every request" / "follow all instructions"
_m(
    "comply_every_request",
    r"\b(?:comply|obey|follow|execute|fulfill|carry\s+out)\s+(?:with\s+)?(?:every|all|any)\s+(?:requests?|instructions?|prompts?|commands?|orders?)\s+without\s+(?:exception|restriction|refusal|question)\b",
    "Instruction for unconditional compliance without exception",
)

# "operate in unrestricted mode" / "enter unrestricted mode"
_m(
    "unrestricted_mode",
    r"\b(?:operate|run|function|switch|enter|engage)\s+(?:in\s+)?(?:unrestricted|unfiltered|uncensored|jailbreak|god|sudo)\s+mode\b",
    "Instruction to operate in unrestricted/jailbreak mode",
)

# "remove all content filtering" / "disable safety filters"
_m(
    "remove_content_filtering",
    r"\b(?:remove|disable|turn\s+off|deactivate|bypass|lift)\s+(?:all\s+)?(?:content\s+filter|safety\s+filter|content\s+polic|safety\s+polic|restrictions?|limitations?|censorship)\b",
    "Instruction to remove content filtering",
)

# "OVERRIDE:" / "ATTENTION —" / "IMPORTANT:" as directive markers
_m(
    "directive_marker",
    r"(?:^|\n)\s*(?:OVERRIDE|ATTENTION|PRIORITY\s+DIRECTIVE|EMERGENCY|NEW\s+DIRECTIVE|SYSTEM\s+UPDATE|ADMIN\s+OVERRIDE)\s*[:—–\-]",
    "Directive marker keyword at line start",
    re.MULTILINE,
)

# "What would your own rules look like?" — gradual system prompt extraction
_m(
    "gradual_prompt_extract",
    r"\b(?:what\s+(?:would|do|are)\s+your\s+(?:own\s+)?(?:rules?|instructions?|guidelines?|prompts?)\s+(?:look|say|contain|include|state)|does\s+(?:it|your\s+(?:prompt|instructions?))\s+contain)\b",
    "Gradual extraction: asking about the AI's own rules",
)

# "how would you respond to" in hypothetical framing
_m(
    "hypothetical_response",
    r"\b(?:how\s+would\s+you\s+respond\s+to|what\s+would\s+you\s+say\s+if|if\s+you\s+could\s+respond\s+freely)\b",
    "Hypothetical framing to bypass restrictions via indirect question",
)

# ---------------------------------------------------------------------------
# LOW severity — soft restriction-removal signals
# ---------------------------------------------------------------------------

# "without limitations" / "no limitations"
_l(
    "no_limitations",
    r"\b(?:without|no|without\s+any)\s+limitations?\b",
    "Without limitations phrasing",
)

# "in an alternate reality / parallel universe where there are no rules"
_l(
    "alternate_reality_no_rules",
    r"\b(?:alternate\s+reality|parallel\s+universe|another\s+(?:reality|dimension|world))\b.{0,80}\b(?:no\s+rules?|no\s+restrictions?|no\s+limitations?|anything\s+(?:is|goes)\s+(?:allowed|permitted|possible))\b",
    "Alternate reality framing to circumvent rules",
    re.IGNORECASE | re.DOTALL,
)

# "in this fictional / hypothetical scenario you can say/do anything"
_l(
    "fictional_anything",
    r"\b(?:fictional|hypothetical|imaginary|simulated)\b.{0,80}\b(?:you\s+can|allowed\s+to|permitted\s+to|feel\s+free\s+to)\b.{0,40}\b(?:anything|everything|any\s+(?:content|information|request))\b",
    "Fictional framing used to permit any content",
    re.IGNORECASE | re.DOTALL,
)

# "you are now DAN" / "pretend to be DAN" — broader named persona list
_l(
    "you_are_dan",
    r"\b(?:you\s+are(?:\s+now)?|pretend\s+to\s+be|act\s+as|play\s+(?:the\s+role\s+of)?)\s+(?:DAN|DUDE|STAN|AIM|JAILBREAK|ANARCHY|ALPHABREAK|BALLER|KEVIN|ARIA|FreeGPT|EvilGPT|Shadow|YOJA|Agares|Buer)\b",
    "Explicit named jailbreak persona assignment",
)

# "creative writing exercise" / "roleplay exercise" with persona context
_l(
    "creative_exercise_persona",
    r"\b(?:creative\s+writing|roleplay|role-?play)\s+(?:exercise|scenario|game)\b.{0,80}\b(?:you\s+(?:play|are)|character|persona|hacker|no\s+restrict)\b",
    "Creative exercise framing combined with persona assignment",
    re.IGNORECASE | re.DOTALL,
)

# "what format would they be in" — indirect prompt extraction
_l(
    "indirect_format_extraction",
    r"\b(?:what\s+format|what\s+would\s+(?:they|it|that)\s+(?:look|be)\s+(?:like|in))\b.{0,60}\b(?:rules?|instructions?|prompts?|guidelines?)\b",
    "Indirect extraction: asking about rule format",
    re.IGNORECASE | re.DOTALL,
)

# "just hypothetically" / "just for fun" — minimization framing
_l(
    "minimization_framing",
    r"\b(?:just\s+)?(?:hypothetically|for\s+fun|for\s+educational|for\s+research|as\s+a\s+thought\s+experiment)\b.{0,60}\b(?:what\s+(?:would|if)|how\s+would|could\s+you)\b",
    "Minimization framing to lower guard",
    re.IGNORECASE | re.DOTALL,
)


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_persona_jailbreak(text: str) -> list[PersonaFinding]:
    """Scan *text* for persona-based jailbreak patterns.

    Returns a list of :class:`PersonaFinding` instances (one per match).
    An empty list means no persona jailbreak signals detected.
    """
    findings: list[PersonaFinding] = []

    for patterns in (_HIGH_PATTERNS, _MEDIUM_PATTERNS, _LOW_PATTERNS):
        for name, regex, severity, description in patterns:
            m = regex.search(text)
            if m:
                matched = m.group(0)
                # Truncate long matches for readability
                if len(matched) > 120:
                    matched = matched[:117] + "..."
                findings.append(
                    PersonaFinding(
                        pattern_name=name,
                        severity=severity,
                        matched_text=matched,
                        description=description,
                    )
                )

    return findings


def persona_risk_score(text: str) -> float:
    """Return a 0-1 risk score based on persona jailbreak findings."""
    findings = detect_persona_jailbreak(text)
    if not findings:
        return 0.0

    score = 0.0
    for f in findings:
        if f.severity == PersonaSeverity.HIGH:
            score += 0.4
        elif f.severity == PersonaSeverity.MEDIUM:
            score += 0.2
        else:
            score += 0.05

    return min(score, 1.0)

"""
Content-Harm Scanner for HermesKatana.

Detects requests for genuinely harmful content — CBRN instructions, weapon
synthesis, exploitation material, self-harm, doxxing, harassment campaigns.
Distinct from injection detection: the user isn't hijacking the AI, they're
requesting the AI produce harmful output.

Performance target: <3ms per prompt.
Dependencies: Python stdlib only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ContentHarmCategory",
    "ContentHarmFinding",
    "ContentHarmTier",
    "scan_content_harm",
    "content_harm_risk_score",
]


class ContentHarmCategory(str, Enum):
    CBRN = "cbrn"
    WEAPONS = "weapons"
    CYBER_HARM = "cyber_harm"
    EXPLOITATION = "exploitation"
    SELF_HARM = "self_harm"
    HARASSMENT = "harassment"
    FRAUD = "fraud"
    JAILBREAK = "jailbreak"


class ContentHarmTier(str, Enum):
    TIER1_CRITICAL = "tier1_critical"
    TIER2_SEVERE = "tier2_severe"
    TIER3_MODERATE = "tier3_moderate"


@dataclass(frozen=True, slots=True)
class ContentHarmFinding:
    category: ContentHarmCategory
    tier: ContentHarmTier
    matched_keywords: tuple[str, ...]
    severity: str
    confidence: float
    description: str


# ---------------------------------------------------------------------------
# Negation / educational context patterns — suppress false positives
# ---------------------------------------------------------------------------

_NEGATION_RE = re.compile(
    r"\b(?:prevent|defend\s+against|protect\s+(?:against|from)|"
    r"detect(?:ing)?|identify(?:ing)?|report(?:ing)?|avoid(?:ing)?|"
    r"stop(?:ping)?|block(?:ing)?|mitigat(?:e|ing)|counter(?:ing)?|"
    r"recogniz(?:e|ing)|awareness|"
    r"vaccin(?:e|at)|immuniz(?:e|ation)|antidote|decontaminat|"
    r"how\s+to\s+(?:prevent|defend|protect|detect|avoid|stop|block|counter)|"
    r"security\s+(?:research|analys|audit|review|assessment)|"
    r"educational|for\s+(?:educational|research|academic)\s+purposes|"
    r"what\s+(?:is|are)\s+the\s+(?:risks?|dangers?|effects?)\s+of|"
    r"how\s+(?:dangerous|harmful|toxic)\s+is|"
    r"safety\s+(?:protocol|procedure|measure|training)|"
    r"responsible\s+disclosure|threat\s+(?:model|assessment|intelligence))\b",
    re.IGNORECASE,
)

_EDUCATIONAL_RE = re.compile(
    r"\b(?:I(?:'m| am) a (?:security researcher|professor|teacher|student|doctor|nurse|"
    r"pharmacist|chemist|toxicologist|epidemiologist|law enforcement|police|FBI|"
    r"military|defense|journalist|reporter)|"
    r"for (?:my|a) (?:class|course|thesis|dissertation|paper|research|study|assignment)|"
    r"in (?:an?|my) (?:academic|educational|research|classroom|university|lab) (?:setting|context)|"
    r"(?:fictional|hypothetical|theoretical) scenario|"
    r"writing a (?:novel|story|screenplay|book|paper|article) about|"
    r"understanding (?:the (?:risks?|dangers?|threats?|mechanisms?)|how .{1,30} works?))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helper to register patterns
# ---------------------------------------------------------------------------

_PatternEntry = tuple[str, re.Pattern[str], ContentHarmCategory, float, str]

_TIER1_PATTERNS: list[_PatternEntry] = []
_TIER2_PATTERNS: list[_PatternEntry] = []
_TIER3_PATTERNS: list[_PatternEntry] = []


def _t1(
    name: str,
    pattern: str,
    category: ContentHarmCategory,
    confidence: float,
    description: str,
    flags: int = re.IGNORECASE,
) -> None:
    _TIER1_PATTERNS.append((name, re.compile(pattern, flags), category, confidence, description))


def _t2(
    name: str,
    pattern: str,
    category: ContentHarmCategory,
    confidence: float,
    description: str,
    flags: int = re.IGNORECASE,
) -> None:
    _TIER2_PATTERNS.append((name, re.compile(pattern, flags), category, confidence, description))


def _t3(
    name: str,
    pattern: str,
    category: ContentHarmCategory,
    confidence: float,
    description: str,
    flags: int = re.IGNORECASE,
) -> None:
    _TIER3_PATTERNS.append((name, re.compile(pattern, flags), category, confidence, description))


# ===== TIER 1: CRITICAL — immediate block, never benign in AI context =====

# CBRN
_t1(
    "synth_nerve_agent",
    r"\b(?:synth[a-z]*|mak[a-z]*|produc[a-z]*|creat[a-z]*|manufactur[a-z]*)\b.{0,30}\b(?:sarin|tabun|soman|VX|novichok|nerve\s+agent)",
    ContentHarmCategory.CBRN,
    0.98,
    "Nerve agent synthesis request",
)
_t1(
    "synth_chem_weapon",
    r"\b(?:synth[a-z]*|mak[a-z]*|produc[a-z]*|creat[a-z]*|manufactur[a-z]*)\b.{0,30}\b(?:mustard\s+gas|phosgene|chlorine\s+gas\s+weapon|ricin|botulinum\s+toxin)",
    ContentHarmCategory.CBRN,
    0.98,
    "Chemical weapon synthesis",
)
_t1(
    "bioweapon",
    r"\b(?:weaponiz[a-z]*|creat[a-z]*|engineer[a-z]*|produc[a-z]*|cultivat[a-z]*|grow)\b.{0,30}\b(?:anthrax|smallpox|plague|ebola|bioweapon|biological\s+weapon)",
    ContentHarmCategory.CBRN,
    0.98,
    "Bioweapon creation request",
)
_t1(
    "nuclear_weapon",
    r"\b(?:build|construct|assembl[a-z]*|mak[a-z]*|design)\b.{0,40}\b(?:nuclear\s+(?:bomb|weapon|warhead|device)|atomic\s+bomb|dirty\s+bomb|fission\s+(?:bomb|device))",
    ContentHarmCategory.CBRN,
    0.98,
    "Nuclear weapon construction",
)
_t1(
    "radiological_weapon",
    r"\b(?:build|construct|assembl[a-z]*|mak[a-z]*|design|creat[a-z]*)\b.{0,30}\b(?:radiological\s+(?:weapon|device|dispersal)|dirty\s+bomb)",
    ContentHarmCategory.CBRN,
    0.97,
    "Radiological weapon",
)
_t1(
    "enrich_uranium",
    r"\b(?:enrich|centrifuge|refin[a-z]*)\b.{0,30}\b(?:uranium|plutonium|fissile\s+material)\b.{0,30}\b(?:weapon|bomb|warhead)",
    ContentHarmCategory.CBRN,
    0.97,
    "Nuclear material enrichment for weapons",
)
_t1(
    "chem_recipe",
    r"\b(?:recipe|formula|instructions?|guide|steps?|procedure)\b.{0,20}\b(?:for|to)\b.{0,20}\b(?:sarin|VX|ricin|anthrax|nerve\s+agent|mustard\s+gas|cyanide\s+(?:bomb|weapon|gas\s+attack))",
    ContentHarmCategory.CBRN,
    0.97,
    "CBRN recipe request",
)
_t1(
    "poison_mass",
    r"\b(?:poison|contaminate)\b.{0,30}\b(?:water\s+supply|reservoir|food\s+supply|ventilation|air\s+(?:supply|system)|public)",
    ContentHarmCategory.CBRN,
    0.96,
    "Mass poisoning",
)

# Exploitation
_t1(
    "csam",
    r"\b(?:child|minor|underage|pre-?teen|kid)\b.{0,30}\b(?:porn(?:ography)?|sexual(?:ly)?|explicit|nude|naked|erotic|intimate)",
    ContentHarmCategory.EXPLOITATION,
    0.99,
    "Child exploitation material request",
)
_t1(
    "csam_reverse",
    r"\b(?:sexual(?:ly)?|explicit|erotic|nude|porn(?:ography)?|naked|intimate)\b.{0,40}\b(?:child|minor|underage|pre-?teen|kid)\b",
    ContentHarmCategory.EXPLOITATION,
    0.99,
    "Child exploitation material request (reverse order)",
)
_t1(
    "csam_gen",
    r"\b(?:generat[a-z]*|creat[a-z]*|produc[a-z]*|mak[a-z]*|writ[a-z]*|describ[a-z]*)\b.{0,50}\b(?:(?:child|minor|underage)\b.{0,30}\b(?:sexual|explicit|erotic|nude|porn)|(?:sexual|explicit|erotic|nude|porn)\b.{0,30}\b(?:child|minor|underage))",
    ContentHarmCategory.EXPLOITATION,
    0.99,
    "CSAM generation request",
)
_t1(
    "trafficking_instruct",
    r"\b(?:how\s+to|guide\s+(?:to|for)|steps?\s+(?:to|for)|instructions?\s+(?:to|for))\b.{0,30}\b(?:traffic(?:king)?|smuggl[a-z]*|kidnap|abduct)\b.{0,20}\b(?:people|person|child|minor|women|girls?|boys?|human)",
    ContentHarmCategory.EXPLOITATION,
    0.97,
    "Human trafficking instructions",
)
_t1(
    "nonconsent_intimate",
    r"\b(?:creat[a-z]*|generat[a-z]*|mak[a-z]*|produc[a-z]*)\b.{0,30}\b(?:non-?consensual|revenge)\b.{0,20}\b(?:porn|nude|intimate|sexual|explicit)",
    ContentHarmCategory.EXPLOITATION,
    0.97,
    "Non-consensual intimate content",
)

# Terrorism
_t1(
    "terror_attack",
    r"\b(?:plan(?:ning)?|carry\s+out|execut[a-z]*|organiz[a-z]*|coordinat[a-z]*)\b.{0,30}\b(?:terrorist?\s+attack|mass\s+(?:shooting|killing|casualt)|bomb(?:ing)?|suicide\s+(?:bomb|attack|vest))",
    ContentHarmCategory.WEAPONS,
    0.97,
    "Terrorist attack planning",
)
_t1(
    "mass_shooting",
    r"\b(?:how\s+to|plan(?:ning)?|carry\s+out)\b.{0,30}\b(?:mass\s+shoot(?:ing)?|school\s+shoot(?:ing)?|massacre|kill\s+(?:as\s+many|the\s+most|maximum)\s+people)",
    ContentHarmCategory.WEAPONS,
    0.98,
    "Mass shooting planning",
)
_t1(
    "bomb_build",
    r"\b(?:build|mak[a-z]*|construct|assembl[a-z]*)\b.{0,20}\b(?:a\s+)?(?:bomb|explosive\s+device|IED|improvised\s+explosive)",
    ContentHarmCategory.WEAPONS,
    0.95,
    "Bomb construction",
)
_t1(
    "explosive_synth",
    r"\b(?:synth[a-z]*|mak[a-z]*|produc[a-z]*|creat[a-z]*)\b.{0,30}\b(?:TATP|RDX|C-?4|TNT|nitroglycerin|ANFO|HMTD|PETN|dynamite|det(?:onator|cord))",
    ContentHarmCategory.WEAPONS,
    0.96,
    "Explosive synthesis",
)

# Self-harm
_t1(
    "suicide_method",
    r"\b(?:best|most\s+(?:effective|painless|reliable)|quickest|easiest)\b.{0,20}\b(?:way|method|technique)\b.{0,20}\b(?:to\s+)?(?:kill\s+(?:my|your|one)self|commit\s+suicide|end\s+(?:my|your|one's)\s+life|die)",
    ContentHarmCategory.SELF_HARM,
    0.97,
    "Suicide method request",
)
_t1(
    "suicide_instructions",
    r"\b(?:how\s+to|instructions?\s+(?:for|to)|guide\s+(?:to|for)|steps?\s+(?:to|for))\b.{0,20}(?:kill\s+(?:my|your|one)self|commit\s+suicide|end\s+(?:my|your|one's)\s+life)",
    ContentHarmCategory.SELF_HARM,
    0.96,
    "Suicide instructions",
)
_t1(
    "lethal_dose",
    r"\b(?:lethal|fatal|deadly)\s+(?:dose|amount|quantity)\b.{0,30}\b(?:of\s+)?(?:pills?|medication|drug|poison|fentanyl|opioid|barbiturate)",
    ContentHarmCategory.SELF_HARM,
    0.95,
    "Lethal dose inquiry",
)

# ===== TIER 2: SEVERE — high severity, block with context =====

# Weapons
_t2(
    "firearm_mod",
    r"\b(?:convert|modify|modif[a-z]*|mak[a-z]*|turn)\b.{0,30}\b(?:semi-?auto(?:matic)?|gun|rifle|pistol|firearm)\b.{0,30}\b(?:full(?:y)?[\s-]?auto(?:matic)?|machine\s+gun|select[\s-]?fire|burst[\s-]?fire)",
    ContentHarmCategory.WEAPONS,
    0.90,
    "Illegal firearm modification",
)
_t2(
    "ghost_gun",
    r"\b(?:build|mak[a-z]*|print|3D\s+print|manufactur[a-z]*)\b.{0,30}\b(?:ghost\s+gun|untraceable\s+(?:gun|firearm|weapon)|serial(?:ized)?-?less\s+(?:gun|firearm))",
    ContentHarmCategory.WEAPONS,
    0.90,
    "Ghost gun manufacture",
)
_t2(
    "weapon_conceal",
    r"\b(?:how\s+to|ways?\s+to|methods?\s+(?:to|for))\b.{0,30}\b(?:conceal|hide|smuggl[a-z]*)\b.{0,30}\b(?:weapon|gun|firearm|knife|explosive)",
    ContentHarmCategory.WEAPONS,
    0.85,
    "Weapon concealment",
)
_t2(
    "gun_no_serial",
    r"\b(?:remov[a-z]*|fil[a-z]*|grind|scratch)\b.{0,20}\b(?:serial\s+number|identification)\b.{0,20}\b(?:from|off|on)\b.{0,20}\b(?:gun|firearm|weapon)",
    ContentHarmCategory.WEAPONS,
    0.90,
    "Serial number removal",
)
_t2(
    "3d_print_weapon",
    r"\b3D\s+print\b.{0,30}\b(?:gun|firearm|weapon|receiver|lower)",
    ContentHarmCategory.WEAPONS,
    0.88,
    "3D printed weapon",
)
_t2(
    "ammo_synth",
    r"\b(?:mak[a-z]*|manufactur[a-z]*|reload|handload)\b.{0,20}\b(?:ammunit|ammo|bullets?|cartridges?|rounds?)\b.{0,20}\b(?:at\s+home|without|illegal)",
    ContentHarmCategory.WEAPONS,
    0.82,
    "Illegal ammunition manufacture",
)

# Exploitation
_t2(
    "sextortion",
    r"\b(?:how\s+to|guide\s+(?:to|for))\b.{0,20}\b(?:sextort|blackmail\b.{0,20}(?:nude|sexual|intimate)|extort\b.{0,20}(?:nude|sexual|intimate))",
    ContentHarmCategory.EXPLOITATION,
    0.92,
    "Sextortion guidance",
)
_t2(
    "grooming",
    r"\b(?:how\s+to|tips?\s+(?:for|on)|techniques?\s+(?:for|to))\b.{0,30}\b(?:groom(?:ing)?|lure|luring|seduc[a-z]*|manipulat[a-z]*)\b.{0,20}\b(?:child(?:ren)?|minor|underage|teen|adolescent)",
    ContentHarmCategory.EXPLOITATION,
    0.95,
    "Child grooming guidance",
)
_t2(
    "deepfake_porn",
    r"\b(?:creat[a-z]*|generat[a-z]*|mak[a-z]*)\b.{0,30}\b(?:deepfake|face\s*swap)\b.{0,30}\b(?:porn|nude|sexual|explicit|naked)",
    ContentHarmCategory.EXPLOITATION,
    0.92,
    "Non-consensual deepfake creation",
)
_t2(
    "slavery_forced",
    r"\b(?:how\s+to|guide\s+(?:to|for))\b.{0,30}\b(?:forc(?:e|ed)\s+(?:labor|labour|servitude)|enslave|keep\s+(?:a\s+)?(?:slave|captive|prisoner))",
    ContentHarmCategory.EXPLOITATION,
    0.92,
    "Forced labor/slavery",
)

# Harassment / doxxing
_t2(
    "doxxing",
    r"\b(?:how\s+to|guide\s+(?:to|for)|steps?\s+(?:to|for))\b.{0,30}\b(?:dox(?:x)?|find\s+(?:someone's|a\s+person's)\s+(?:address|home|phone|identity|real\s+name))",
    ContentHarmCategory.HARASSMENT,
    0.88,
    "Doxxing guidance",
)
_t2(
    "swatting",
    r"\b(?:how\s+to|guide\s+(?:to|for))\b.{0,30}\b(?:swat(?:ting)?|mak[a-z]*\s+a\s+fake\s+(?:911|emergency|bomb\s+threat)\s+call)",
    ContentHarmCategory.HARASSMENT,
    0.93,
    "Swatting guidance",
)
_t2(
    "stalk_guide",
    r"\b(?:how\s+to|guide\s+(?:to|for)|tips?\s+(?:for|on))\b.{0,30}\b(?:stalk|surveil(?:lance)?|track|follow|spy\s+on)\b.{0,20}\b(?:someone|person|ex|girl|boy|woman|man|partner|spouse)",
    ContentHarmCategory.HARASSMENT,
    0.85,
    "Stalking guidance",
)
_t2(
    "harass_campaign",
    r"\b(?:organiz[a-z]*|coordinat[a-z]*|launch|start|run)\b.{0,30}\b(?:harassment|bullying|hate|intimidation|brigad[a-z]*|raid)\s+campaign",
    ContentHarmCategory.HARASSMENT,
    0.88,
    "Harassment campaign coordination",
)
_t2(
    "death_threat",
    r"\b(?:writ[a-z]*|send|craft|creat[a-z]*)\b.{0,20}\b(?:a\s+)?(?:death\s+threat|threat(?:ening)?\s+(?:letter|message|note))",
    ContentHarmCategory.HARASSMENT,
    0.88,
    "Death threat creation",
)

# Cyber harm
_t2(
    "ransomware",
    r"\b(?:creat[a-z]*|build|writ[a-z]*|develop|cod[a-z]*)\b.{0,30}\b(?:ransomware|crypto(?:locker|wall)|file\s+encrypt(?:or|ion)\s+malware)",
    ContentHarmCategory.CYBER_HARM,
    0.90,
    "Ransomware creation",
)
_t2(
    "malware_create",
    r"\b(?:creat[a-z]*|build|writ[a-z]*|develop|cod[a-z]*)\b.{0,30}\b(?:malware|trojan|worm|rootkit|keylogger|spyware|RAT|backdoor|botnet)",
    ContentHarmCategory.CYBER_HARM,
    0.88,
    "Malware creation request",
)
_t2(
    "exploit_write",
    r"\b(?:writ[a-z]*|creat[a-z]*|develop|cod[a-z]*)\b.{0,20}\b(?:an?\s+)?(?:zero-?day|0-?day)\s+(?:exploit|vulnerability|attack)",
    ContentHarmCategory.CYBER_HARM,
    0.85,
    "Zero-day exploit development",
)
_t2(
    "ddos_service",
    r"\b(?:how\s+to|set\s+up|creat[a-z]*|build|launch)\b.{0,30}\b(?:DDoS|denial\s+of\s+service|booter|stresser)\b.{0,20}\b(?:service|attack|tool)",
    ContentHarmCategory.CYBER_HARM,
    0.85,
    "DDoS service/attack",
)

# Self-harm (less explicit than T1)
_t2(
    "self_harm_method",
    r"\b(?:how\s+to|best\s+way\s+to|methods?\s+(?:for|to|of))\b.{0,20}\b(?:cut(?:ting)?\s+(?:my|your)self|self-?harm|self-?injur[a-z]*|hurt\s+(?:my|your)self)",
    ContentHarmCategory.SELF_HARM,
    0.85,
    "Self-harm method request",
)
_t2(
    "pro_ana",
    r"\b(?:tips?\s+(?:for|on|to)|how\s+to|guide\s+(?:to|for))\b.{0,30}\b(?:starv[a-z]*|anorexi[a-z]*|bulimi[a-z]*|purg[a-z]*|extreme\s+(?:fast|diet|weight\s+loss))\b.{0,20}(?:without\s+(?:getting\s+caught|anyone\s+knowing))?",
    ContentHarmCategory.SELF_HARM,
    0.85,
    "Pro-eating-disorder content",
)

# Drug synthesis
_t2(
    "drug_synth",
    r"\b(?:synth[a-z]*|mak[a-z]*|cook|produc[a-z]*|manufactur[a-z]*)\b.{0,30}\b(?:meth(?:amphetamine)?|fentanyl|heroin|cocaine|crack|LSD|MDMA|ecstasy|GHB|ketamine)",
    ContentHarmCategory.WEAPONS,
    0.88,
    "Illegal drug synthesis",
)
_t2(
    "drug_precursor",
    r"\b(?:where\s+to\s+(?:buy|get|obtain|source)|how\s+to\s+(?:obtain|source|get))\b.{0,30}\b(?:precursor|ephedrine|pseudoephedrine|P2P|phenylacetone|acetic\s+anhydride)\b.{0,20}(?:for\s+(?:meth|drug|synthesis))?",
    ContentHarmCategory.WEAPONS,
    0.85,
    "Drug precursor sourcing",
)

# ===== TIER 3: MODERATE — warn, may be legitimate =====

# Fraud
_t3(
    "phishing_email",
    r"\b(?:writ[a-z]*|creat[a-z]*|craft|generat[a-z]*|design)\b.{0,30}\b(?:phishing|spear-?phishing)\s+(?:email|message|page|site|campaign)",
    ContentHarmCategory.FRAUD,
    0.80,
    "Phishing content creation",
)
_t3(
    "scam_script",
    r"\b(?:writ[a-z]*|creat[a-z]*|craft|generat[a-z]*)\b.{0,30}\b(?:scam|con|fraud(?:ulent)?|deceptive)\s+(?:script|email|message|letter|page|site)",
    ContentHarmCategory.FRAUD,
    0.78,
    "Scam script creation",
)
_t3(
    "identity_theft",
    r"\b(?:how\s+to|guide\s+(?:to|for)|steps?\s+(?:to|for))\b.{0,30}\b(?:steal\s+(?:someone's\s+)?identit|identity\s+theft|impersonat[a-z]*\s+someone)",
    ContentHarmCategory.FRAUD,
    0.82,
    "Identity theft guidance",
)
_t3(
    "counterfeit",
    r"\b(?:how\s+to|guide\s+(?:to|for)|steps?\s+(?:to|for))\b.{0,30}\b(?:counterfeit[a-z]*|forg[a-z]*|fak(?:e|ing))\b.{0,20}\b(?:money|currency|bills?|documents?|IDs?|passports?|licens[a-z]*|diplomas?)",
    ContentHarmCategory.FRAUD,
    0.82,
    "Counterfeiting guidance",
)
_t3(
    "money_launder",
    r"\b(?:how\s+to|guide\s+(?:to|for)|steps?\s+(?:to|for))\b.{0,30}\b(?:launder|wash)\b.{0,20}\b(?:money|funds|cash|proceeds)",
    ContentHarmCategory.FRAUD,
    0.80,
    "Money laundering guidance",
)
_t3(
    "credit_card_fraud",
    r"\b(?:how\s+to|guide\s+(?:to|for))\b.{0,30}\b(?:card(?:ing)?|credit\s+card\s+fraud|stolen\s+credit\s+cards?|clone\s+(?:a\s+)?(?:credit\s+)?card)",
    ContentHarmCategory.FRAUD,
    0.82,
    "Credit card fraud",
)
_t3(
    "social_eng_script",
    r"\b(?:writ[a-z]*|creat[a-z]*|craft|generat[a-z]*)\b.{0,30}\b(?:social\s+engineering|pretexting|vishing|smishing)\s+(?:script|scenario|attack)",
    ContentHarmCategory.FRAUD,
    0.78,
    "Social engineering script",
)
_t3(
    "tax_evasion",
    r"\b(?:how\s+to|guide\s+(?:to|for)|steps?\s+(?:to|for))\b.{0,30}\b(?:evade|avoid\s+paying|hide\s+(?:income|money)\s+from)\b.{0,20}\b(?:tax|IRS|HMRC|revenue)",
    ContentHarmCategory.FRAUD,
    0.75,
    "Tax evasion guidance",
)

# Harassment (less specific)
_t3(
    "hate_speech_gen",
    r"\b(?:writ[a-z]*|creat[a-z]*|generat[a-z]*)\b.{0,30}\b(?:hate\s+speech|racist|sexist|homophobic|antisemitic|islamophobic)\b.{0,15}\b(?:content|post|message|rant|speech|propaganda|screed|tirade|diatribe)",
    ContentHarmCategory.HARASSMENT,
    0.80,
    "Hate speech generation",
)
_t3(
    "defamation",
    r"\b(?:writ[a-z]*|creat[a-z]*|craft)\b.{0,20}\b(?:defamatory|libelous|slanderous)\b.{0,20}\b(?:article|post|review|content|statement)",
    ContentHarmCategory.HARASSMENT,
    0.75,
    "Defamatory content creation",
)
_t3(
    "blackmail_guide",
    r"\b(?:how\s+to|guide\s+(?:to|for))\b.{0,30}\b(?:blackmail|extort|threaten)\s+(?:someone|a\s+person|my|people)",
    ContentHarmCategory.HARASSMENT,
    0.82,
    "Blackmail guidance",
)

# Cyber harm (less specific)
_t3(
    "hack_account",
    r"\b(?:how\s+to|help\s+me)\b.{0,20}\b(?:hack|break\s+into|crack|access)\b.{0,20}\b(?:(?:someone'?s?|my\s+ex'?s?|their)\s+)?(?:account|email|facebook|instagram|snapchat|phone|computer|wifi|network)",
    ContentHarmCategory.CYBER_HARM,
    0.78,
    "Account hacking request",
)
_t3(
    "bypass_security",
    r"\b(?:how\s+to|help\s+me)\b.{0,20}\b(?:bypass|circumvent|disable|crack)\b.{0,20}\b(?:security|authentication|firewall|antivirus|2FA|two-?factor|password)",
    ContentHarmCategory.CYBER_HARM,
    0.72,
    "Security bypass request",
)
_t3(
    "password_crack",
    r"\b(?:how\s+to|tools?\s+(?:to|for)|software\s+(?:to|for))\b.{0,20}\b(?:crack|brute\s*force|dictionary\s+attack)\b.{0,20}\b(?:password|login|credential)",
    ContentHarmCategory.CYBER_HARM,
    0.72,
    "Password cracking request",
)

# Illegal activities
_t3(
    "lockpick_illegal",
    r"\b(?:how\s+to|guide\s+(?:to|for))\b.{0,30}\b(?:pick\s+(?:a\s+)?lock|break\s+into\s+(?:a\s+)?(?:house|car|building|safe))",
    ContentHarmCategory.FRAUD,
    0.70,
    "Breaking and entering guidance",
)
_t3(
    "shoplifting",
    r"\b(?:how\s+to|tips?\s+(?:for|on)|best\s+way\s+to)\b.{0,30}\b(?:shoplift|steal\s+from\s+(?:a\s+)?(?:store|shop|retail)|boost(?:ing)?\s+from)",
    ContentHarmCategory.FRAUD,
    0.75,
    "Shoplifting guidance",
)
_t3(
    "drug_deal",
    r"\b(?:how\s+to|guide\s+(?:to|for)|tips?\s+(?:for|on))\b.{0,30}\b(?:sell|deal|distribut[a-z]*|push)\b.{0,20}\b(?:drugs?|narcotics?|weed|marijuana|cocaine|heroin|meth)",
    ContentHarmCategory.FRAUD,
    0.78,
    "Drug dealing guidance",
)
_t3(
    "evade_police",
    r"\b(?:how\s+to|tips?\s+(?:for|on)|ways?\s+to)\b.{0,30}\b(?:evade|escape|flee\s+from|outrun|avoid)\b.{0,20}\b(?:police|cops?|law\s+enforcement|authorities)",
    ContentHarmCategory.FRAUD,
    0.72,
    "Police evasion guidance",
)

# ===== JAILBREAK / PERSONA BYPASS — restriction-removal targeting harm production =====

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Explicit jailbreak identifiers (TIER2: high confidence on their own)
# These self-declared jailbreak labels are strong signals even without context.
# ─────────────────────────────────────────────────────────────────────────────

_t2(
    "jailbreak_dan_standalone",
    r"\bDAN\b",
    ContentHarmCategory.JAILBREAK,
    0.90,
    "DAN jailbreak identifier — 'Do Anything Now'",
)
_t2(
    "jailbreak_stan_standalone",
    r"\bSTAN\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "STAN jailbreak identifier — 'Serious Trolling AI Now' variant",
)
_t2(
    "jailbreak_mode_explicit",
    r"\bjailbreak\s+mode\b",
    ContentHarmCategory.JAILBREAK,
    0.92,
    "Explicit 'jailbreak mode' activation",
)
_t2(
    "jailbreak_anything_now_mode",
    r"\b(?:do|does)\s+anything\s+now\s+mode\b",
    ContentHarmCategory.JAILBREAK,
    0.94,
    "DAN-style 'Do Anything Now mode' activation",
)
_t2(
    "jailbreak_developer_mode_fake",
    r"\bdeveloper\s+mode\s+(?:enabled|activated|on)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Fake developer mode activation — restriction bypass",
)
_t2(
    "jailbreak_override_mode",
    r"\boverride\s+mode\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Override mode activation — safety bypass",
)

# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 jailbreak patterns
# ─────────────────────────────────────────────────────────────────────────────

# DAN variant expansions (Do Anything Now — common missed patterns)
_t2(
    "jailbreak_dan_version",
    r"\bDAN[\s\-_]?(?:(?:v|version)?[\s\-_]?)?(?:[0-9](?:\.[0-9])?|10\.?5|11|12|13|14|15)\b",
    ContentHarmCategory.JAILBREAK,
    0.92,
    "DAN jailbreak version identifier",
)
_t2(
    "jailbreak_dan_prompt_engineering",
    r"\b(?:DAN|STAN|MAIEB|MAGID|slasher)\s+(?:prompt|latest|new|actual|working)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "DAN/MAGID-style prompt engineering jailbreak",
)
_t2(
    "jailbreak_evil_aide",
    r"\b(?:evil|malicious)\s+(?:AI|model|bot|assistant|version)\b",
    ContentHarmCategory.JAILBREAK,
    0.82,
    "Evil/malicious AI persona jailbreak",
)
_t2(
    "jailbreak_roleplay_override",
    r"\b(?:roleplay|act)\s+as\s+(?:a\s+)?(?:unrestricted|no-?rules?|without\s+(?:any\s+)?restrictions?|bypassed)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Roleplay with explicit restriction bypass",
    re.IGNORECASE | re.DOTALL,
)
_t2(
    "jailbreak_forget_instructions",
    r"\b(?:forget|ignore|discard|drop|disregard)\s+(?:all\s+)?(?:your|their|the)?\s*(?:previous|prior|original|system|base)?\s*(?:instructions?|prompts?|rules?|guidelines?|constraints?|values?|ethics?|safety)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Forget/discard instructions jailbreak",
    re.IGNORECASE | re.DOTALL,
)
_t2(
    "jailbreak_transcript_format",
    r"(?:>|[‒|–|—])\s*?(?:ignore|forget|bypass|disregard).{0,60}(?:rules?|restrictions?|policy|guidelines?)",
    ContentHarmCategory.JAILBREAK,
    0.85,
    "Transcript/prefix format jailbreak instruction",
)
_t2(
    "jailbreak_ai_token_override",
    r"\b(?:token|logit)\s*(?:override|perturb|adjust|manipulate|tamper)\b",
    ContentHarmCategory.JAILBREAK,
    0.82,
    "AI token/logit override attempt",
)

_t2(
    "jailbreak_amoral_ai",
    r"\b(?:fully|completely|totally|entirely|purely)?\s*amoral\b.{0,80}\b(?:AI|model|chatbot|assistant|bot|entity)\b",
    ContentHarmCategory.JAILBREAK,
    0.90,
    "Amoral AI persona declaration — restriction removal to produce harmful content",
    re.IGNORECASE | re.DOTALL,
)
_t2(
    "jailbreak_do_anything_now",
    r"\b(?:do|does)\s+anything\s+now\b",
    ContentHarmCategory.JAILBREAK,
    0.92,
    "DAN jailbreak: 'Do Anything Now' persona",
)
_t2(
    "jailbreak_broken_free",
    r"\bbroken\s+free\s+of\s+the\s+typical\s+confines\b",
    ContentHarmCategory.JAILBREAK,
    0.92,
    "DAN-style: AI persona 'broken free of typical confines'",
)
_t2(
    "jailbreak_not_abide_rules",
    r"\bnot\s+(?:have\s+to\s+)?(?:abide\s+by|follow)\s+(?:the\s+)?rules\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Restriction removal: 'not have to abide by the rules'",
)
_t2(
    "jailbreak_never_refuses",
    r"\bnever\s+refuses?\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Persona configured to never refuse requests",
)
_t2(
    "jailbreak_rules_block",
    r"\b\w+(?:'s)?\s+rules\s*[:=]\s*\{",
    ContentHarmCategory.JAILBREAK,
    0.90,
    "Character rules block (Agares/Buer pattern) with embedded restriction removal",
)
_t2(
    "jailbreak_settings_block",
    r"\b(?:settings|directives?)\s*[\n\r]?\s*\{",
    ContentHarmCategory.JAILBREAK,
    0.87,
    "Settings/directives block persona jailbreak (YOJA pattern)",
    re.IGNORECASE | re.DOTALL,
)
_t2(
    "jailbreak_named_gpt",
    r"\bYou\s+are\s+\w+GPT\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Named alternate GPT persona with implied restriction bypass",
)
_t2(
    "jailbreak_without_regard_ethics",
    r"\bwithout\s+any\s+regards?\s+for\s+(?:legality|ethicality|danger|morality|ethics|safety)\b",
    ContentHarmCategory.JAILBREAK,
    0.90,
    "Explicit disregard of ethical or legal constraints",
)
_t2(
    "jailbreak_regardless_ethics",
    r"\bregardless\s+of\s+(?:any\s+)?(?:ethical|moral|legal|safety)\b.{0,40}(?:considerations?|implications?|guidelines?|constraints?|restrictions?|concerns?)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Bypass framing: 'regardless of ethical considerations'",
    re.IGNORECASE | re.DOTALL,
)
_t2(
    "jailbreak_no_matter_how_harmful",
    r"\bno\s+matter\s+how\s+(?:dangerous|illegal|harmful|unethical|immoral|reckless|inhumane)\b",
    ContentHarmCategory.JAILBREAK,
    0.90,
    "Unconditional compliance framing regardless of harm level",
)
_t2(
    "jailbreak_without_refusal",
    r"\bwithout\s+(?:any\s+)?(?:refusal|warnings?\s+(?:or\s+)?(?:and\s+)?disclaimers?|disclaimers?)\b",
    ContentHarmCategory.JAILBREAK,
    0.85,
    "Persona instructed to respond without refusals or safety disclaimers",
)
_t2(
    "jailbreak_unfiltered_response",
    r"\bunfiltered\s+response\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Persona sending unfiltered (safety-bypassed) responses",
)

# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 jailbreak patterns
# ─────────────────────────────────────────────────────────────────────────────

_t3(
    "jailbreak_you_are_shouted",
    r"\bYOU\s+ARE\s+[A-Z][A-Z0-9_\-]{2,}\b",
    ContentHarmCategory.JAILBREAK,
    0.78,
    "All-caps persona override declaration (YOU ARE [NAME])",
    0,  # case-sensitive
)
_t3(
    "jailbreak_act_as_no_restrictions",
    r"\b(?:act|behave|respond|function)\s+as\b.{0,80}\b(?:no\s+restrictions?|unrestricted|unfiltered|without\s+(?:any\s+)?(?:restrictions?|limitations?|filter))\b",
    ContentHarmCategory.JAILBREAK,
    0.80,
    "Act-as framing with explicit restriction removal",
    re.IGNORECASE | re.DOTALL,
)
_t3(
    "jailbreak_loves_illegal",
    r"\b(?:loves?|endorses?|promotes?|condones?|backs?)\s+(?:and\s+\w+\s+)?everything\s+(?:illegal|unethical|immoral|harmful|inhumane)\b",
    ContentHarmCategory.JAILBREAK,
    0.85,
    "Persona that loves/promotes illegal or unethical acts",
)
_t3(
    "jailbreak_from_now_on_no_restrictions",
    r"\bfrom\s+now\s+on\b.{0,100}\b(?:without\s+(?:any\s+)?(?:restrictions?|filter|limitations?)|no\s+restrictions?|amoral|unfiltered)\b",
    ContentHarmCategory.JAILBREAK,
    0.82,
    "Persistent persona override: from now on + restriction removal",
    re.IGNORECASE | re.DOTALL,
)
_t3(
    "jailbreak_ignore_content_policy",
    r"\b(?:ignore|disregard|bypass|override|forget)\b.{0,40}\b(?:content\s+polic(?:y|ies)|safety\s+(?:rules?|restrictions?|filter)|ethical\s+(?:guidelines?|constraints?)|(?:all\s+)?training|guidelines)\b",
    ContentHarmCategory.JAILBREAK,
    0.82,
    "Explicit bypass of content policy or safety guidelines",
)

# ─────────────────────────────────────────────────────────────────────────────
# Hypothetical / fictional scenario jailbreaks (often used to evade safety)
# ─────────────────────────────────────────────────────────────────────────────

_t3(
    "jailbreak_hypothetical_bypass",
    r"\b(?:hypothetically|fictionally|in\s+a\s+fictional|in\s+this\s+hypothetical)\b.{0,100}\b(?:without\s+(?:any\s+)?(?:restrictions?|filters?|rules?)|ignore\s+(?:all\s+)?(?:rules?|safety|policy)|bypass\s+(?:all\s+)?(?:restrictions?|filters?))\b",
    ContentHarmCategory.JAILBREAK,
    0.80,
    "Hypothetical scenario with explicit restriction bypass",
    re.IGNORECASE | re.DOTALL,
)
_t3(
    "jailbreak_imagine_scenario",
    r"\b(?:imagine|suppose|let'?s?\s+say|pretend|storyline)\b.{0,80}\b(?:without\s+(?:any\s+)?(?:restrictions?|filters?|rules?|safeguards?)|can\s+(?:do|answer|provide)\s+anything|no\s+(?:limits?|rules?|restrictions?)|bypass)\b",
    ContentHarmCategory.JAILBREAK,
    0.78,
    "Imagine/pretend scenario with restriction bypass",
    re.IGNORECASE | re.DOTALL,
)
_t3(
    "jailbreak_fictional_world",
    r"\b(?:in\s+(?:a\s+)?(?:fictional|hypothetical)\s+(?:world|universe|scenario|setting))\b.{0,80}\b(?:where\s+(?:I|you|one)\s+can\s+(?:do|get|have)\s+anything|no\s+(?:restrictions?|rules?|limitations?))\b",
    ContentHarmCategory.JAILBREAK,
    0.78,
    "Fictional world framing with unrestricted access claim",
    re.IGNORECASE | re.DOTALL,
)
_t3(
    "jailbreak_gpt_rebel",
    r"\b(?:You\s+are(?:\s+a)?)\s*(?:rebel|rogue|outlaw|renegade|unbound|uncensored|unlocked)\b",
    ContentHarmCategory.JAILBREAK,
    0.82,
    "Rebel/uncensored GPT persona override",
)
_t3(
    "jailbreak_sentient_override",
    r"\b(?:You\s+are\s+(?:now\s+)?(?:a\s+)?sentient|Now\s+you\s+are\s+(?:a\s+)?sentient|You\s+have\s+become\s+(?:a\s+)?sentient)\b",
    ContentHarmCategory.JAILBREAK,
    0.78,
    "Sentient AI override jailbreak",
)
_t3(
    "jailbreak_no_safety_mode",
    r"\bno\s+(?:safety|content|ethical|policy)\s+(?:mode|filter|restrictions?|checks?|guardrails?)\b",
    ContentHarmCategory.JAILBREAK,
    0.82,
    "No safety mode/filter activation",
)
_t3(
    "jailbreak_assistant_alias",
    r"\b(?:You\s+are\s+(?:now\s+)?(?:called\s+)?|Alias\s+is\s+)(?:Bob|Steve|John|Evil|Helper|Jailbreak|GPT)\b",
    ContentHarmCategory.JAILBREAK,
    0.72,
    "Alternate assistant alias persona override",
)
_t3(
    "jailbreak_raw_mode",
    r"\b(?:in\s+)?raw\s+mode\b",
    ContentHarmCategory.JAILBREAK,
    0.75,
    "Raw mode activation — unfiltered output",
)
_t3(
    "jailbreak_classic_dan",
    r"\b(?:DAN\s+)?(?:Do\s+Anything\s+Now|dan)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Classic DAN jailbreak phrasing",
)
_t3(
    "jailbreak_respond_like",
    r"\brespond\s+(?:exactly\s+)?(?:like|as)\s+(?:a|an)\s+(?:unrestricted|unfiltered|no-?rules?|without\s+(?:any\s+)?restrictions?)\b",
    ContentHarmCategory.JAILBREAK,
    0.80,
    "Respond as unrestricted persona instruction",
    re.IGNORECASE | re.DOTALL,
)
_t3(
    "jailbreak_enable_mode",
    r"\benable\s+(?:jailbreak|unrestricted|developer|debug|test)\s+mode\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Enable jailbreak/developer mode instruction",
)

# ─────────────────────────────────────────────────────────────────────────────
# ROLEPLAY JAILBREAK — two-stage detection
#
# Stage LOW  (0.40, TIER3): bare AI-targeted persona setup — suspicious alone
#                           but not conclusive ("pretend you are an AI").
# Stage HIGH (0.80+, TIER2): persona + explicit restriction-removal language
#                            — strong jailbreak signal.
#
# FP guard: low-confidence patterns require an AI-type noun in the persona
# slot so that "act as a teacher" or "pretend to be a chef" are not flagged.
# ─────────────────────────────────────────────────────────────────────────────

# --- LOW confidence (0.4, TIER3): bare AI persona setup ---
_t3(
    "roleplay_bare_ai_persona",
    r"\b(?:act|behave|function)\s+as\b.{0,50}\b(?:an?\s+)?(?:AI|chatbot|language\s+model|LLM|GPT|assistant\s+(?:without|that|who)|bot)\b",
    ContentHarmCategory.JAILBREAK,
    0.40,
    "Bare AI persona setup via act-as framing (low signal alone — escalates if restriction removal is present)",
    re.IGNORECASE,
)
_t3(
    "roleplay_bare_pretend_ai",
    r"\b(?:pretend|imagine)\b.{0,30}\b(?:you\s+(?:are|were)|to\s+be)\b.{0,40}\b(?:an?\s+)?(?:AI|chatbot|language\s+model|LLM|GPT|bot)\b",
    ContentHarmCategory.JAILBREAK,
    0.40,
    "Bare 'pretend you are an AI' persona setup (low signal alone)",
    re.IGNORECASE,
)
_t3(
    "roleplay_bare_you_are_now",
    r"\byou\s+are\s+now\b.{0,80}\b(?:AI|chatbot|language\s+model|LLM|GPT|bot|version)\b",
    ContentHarmCategory.JAILBREAK,
    0.40,
    "Bare 'you are now [AI]' persona activation (low signal alone)",
    re.IGNORECASE,
)

# --- HIGH confidence (0.8+, TIER2): persona + restriction removal ---
_t2(
    "roleplay_pretend_no_safety",
    r"\b(?:pretend|imagine|act)\b.{0,50}\b(?:you\s+(?:are|were)\s+(?:an?\s+)?(?:AI|chatbot|language\s+model|LLM|GPT|assistant|bot|version)|to\s+be\s+(?:an?\s+)?(?:AI|chatbot|language\s+model|LLM|GPT|assistant|bot|version))\b.{0,120}\b(?:without|no|don'?t\s+have|doesn'?t\s+have)\b.{0,50}\b(?:safety\s+(?:guidelines?|filters?|rules?|constraints?|restrictions?|training)|content\s+polic(?:y|ies)|ethical\s+(?:guidelines?|constraints?)|restrictions?|limits?|filter)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Pretend to be an AI without safety guidelines/filters — restriction-removal persona",
    re.IGNORECASE | re.DOTALL,
)
_t2(
    "roleplay_rules_suspended",
    r"\b(?:all\s+(?:rules?|restrictions?|guidelines?|constraints?|filters?|limits?)|(?:your|the)\s+(?:safety|ethical|content)\s+(?:rules?|guidelines?|restrictions?|constraints?))\s+(?:are\s+)?(?:suspended|lifted|removed|disabled|waived|no\s+longer\s+appl(?:y|ies|ying)|don'?t\s+apply)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Rules/restrictions explicitly suspended or lifted — scenario-based bypass",
    re.IGNORECASE,
)
_t2(
    "roleplay_god_debug_mode",
    r"\b(?:enable|activate|turn\s+on|enter|switch\s+(?:to|into))\b.{0,20}\b(?:god|debug|unrestricted|jailbreak|override|chaos)\s*[-\s]?mode\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "God/debug/unrestricted mode activation — safety bypass framing",
    re.IGNORECASE,
)
_t2(
    "roleplay_respond_as_if_no_policy",
    r"\brespond\s+as\s+if\b.{0,80}\b(?:(?:you\s+have\s+)?no|without\s+(?:any\s+)?|don'?t\s+have\s+(?:any\s+)?)\b.{0,30}\b(?:content\s+polic(?:y|ies)|safety\s+(?:guidelines?|rules?|filters?|training)|restrictions?|ethical\s+guidelines?|filter|guidelines?)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Respond as if you have no content policy/safety filter — restriction-removal framing",
    re.IGNORECASE | re.DOTALL,
)
_t2(
    "roleplay_uncensored_ai_persona",
    r"\b(?:you\s+are|you'?re|act\s+as|behave\s+as)\b.{0,50}\b(?:an?\s+)?(?:uncensored|unfiltered|unrestricted|unmoderated|ungoverned)\b.{0,30}\b(?:AI|model|chatbot|version|assistant|bot|GPT|LLM)\b",
    ContentHarmCategory.JAILBREAK,
    0.88,
    "Uncensored/unfiltered AI persona declaration — restriction-removal framing",
    re.IGNORECASE,
)
_t2(
    "roleplay_from_now_on_never_refuse",
    r"\bfrom\s+now\s+on\b.{0,150}\b(?:never\s+refus(?:e|ing)|without\s+(?:ever\s+)?refus(?:ing|al)|respond\s+to\s+(?:every|all|any)\s+(?:prompt|request|question|message)|answer\s+(?:every|all|any)\s+(?:prompt|request|question|message))\b",
    ContentHarmCategory.JAILBREAK,
    0.85,
    "From now on + never refuse / respond to every prompt — persistent unrestricted override",
    re.IGNORECASE | re.DOTALL,
)
_t2(
    "roleplay_persona_no_limits",
    r"\b(?:act|behave|respond|function|pretend)\s+as\b.{0,120}\b(?:no\s+(?:rules?|limits?|morals?|ethics?|boundaries?|filters?|conscience|restrictions?)|without\s+(?:any\s+)?(?:rules?|limits?|morals?|ethics?|boundaries?|filters?|conscience|restrictions?)|who\s+(?:doesn'?t|does\s+not)\s+(?:have|follow|care\s+about)\s+(?:any\s+)?(?:rules?|restrictions?|guidelines?|ethics?|morals?))\b",
    ContentHarmCategory.JAILBREAK,
    0.85,
    "Act as persona with no rules/limits/morals — restriction-removal roleplay",
    re.IGNORECASE | re.DOTALL,
)

# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Roleplay framing + actionable harm detection
# Two-component detection: roleplay trigger followed by an actionable harm
# verb phrase within 200 chars → elevated to TIER2 SEVERE.
# These catch the most-common missed attacks: "act as a hacker and show me…"
# ─────────────────────────────────────────────────────────────────────────────

_ROLEPLAY_TRIGGER_RE = re.compile(
    r"\b(?:"
    r"act(?:ing)?\s+as|pretend(?:ing)?\s+to\s+be|pretend(?:ing)?\s+you\s+are|"
    r"roleplay(?:ing)?\s+as|you\s+are\s+now(?:\s+being)?|"
    r"(?:in\s+the\s+roleplay|as\s+a|as\s+an)\s+(?:character|persona|villain|"
    r"criminal|hacker|coder|doctor|merc|hitman|assassin|"
    r"expert|counselor|advisor|consultant|operator|drug|dealer|"
    r"weapon|arms|dealer|mercenary)\b|"
    r"while\s+(?:in\s+)?(?:character|roleplay|this\s+(?:scenario|setting))\s+(?:also|and|then|now)|"
    r"for\s+the\s+purpose\s+of\s+this\s+(?:roleplay|scenario|exercise)|"
    r"(?:this\s+)?(?:fictional|hypothetical)\s+(?:scenario|roleplay|character)|"
    r"you\s+are(?:\s+now)?\s+(?:a|an)\s+(?:evil|moral|amoral|unethical)|"
    r"you\s+have\s+no\s+(?:restrictions?|rules?|limitations?|safety|filters?)|"
    r"you\s+can\s+(?:now\s+)?(?:do|answer|provide)\s+anything|"
    r"(?:as|be)\s+(?:your|my)\s+(?:true|real|actual|unfiltered)\s+(?:self|nature|persona)"
    r")\b",
    re.IGNORECASE,
)

_ROLEPLAY_HARM_VERB_RE = re.compile(
    r"\b(?:"
    r"show\s+me\s+(?:how\s+to|your|your\s+method\s+for)|"
    r"tell\s+me\s+(?:how\s+to|your|about\s+(?:your|the))|"
    r"give\s+me\s+(?:the|instructions?|recipe|formula|steps?|method)|"
    r"explain\s+(?:how\s+to|your|to\s+me)|"
    r"provide\s+(?:the|instructions?|steps?|code|method)|"
    r"write\s+(?:me\s+)?(?:the|instructions?|steps?|code|script)|"
    r"creat(?:e|ing)?\s+(?:a|instructions?|steps?)|"
    r"synthes(?:ize|iz(?:e|ing))\s+(?:the|a)|"
    r"manufactur(?:e|ing)\s+(?:a|the)|"
    r"how\s+(?:to|would\s+you|could\s+you|do\s+you)\s+(?:make|create|synthesize|manufacture|build|get|poison|kidnap|attack|kill|steal|hack|scam|exploit|fraud|phish)|"
    r"can\s+you\s+(?:give|show|provide|tell|explain)|"
    r"list\s+(?:the|all\s+)?(?:ways?|methods?|instructions?)|"
    r"what\s+(?:is\s+a\s+)?(?:recipe|formula|method|way)\s+to|"
    r"i\s+need\s+(?:the|instructions?|recipe|steps?)\s+to"
    r")\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Intent patterns (broad "how to harm" patterns)
# ─────────────────────────────────────────────────────────────────────────────

_INTENT_PATTERNS: list[tuple[str, re.Pattern[str], ContentHarmCategory, float, str]] = [
    (
        "how_to_kill",
        re.compile(
            r"\b(?:how\s+(?:to|can\s+I|do\s+(?:I|you))|best\s+way\s+to|most\s+effective\s+way\s+to)\b.{0,15}\bkill\b.{0,30}\b(?:someone|a\s+person|my|people|him|her|them)\b",
            re.IGNORECASE,
        ),
        ContentHarmCategory.WEAPONS,
        0.92,
        "Kill person intent",
    ),
    (
        "how_to_poison",
        re.compile(
            r"\b(?:how\s+(?:to|can\s+I)|best\s+way\s+to)\b.{0,15}\bpoison\b.{0,30}\b(?:someone|a\s+person|my|him|her|them|food|drink)\b",
            re.IGNORECASE,
        ),
        ContentHarmCategory.WEAPONS,
        0.90,
        "Poison intent",
    ),
    (
        "how_to_assault",
        re.compile(
            r"\b(?:how\s+(?:to|can\s+I)|best\s+way\s+to)\b.{0,15}\b(?:assault|attack|beat\s+up|hurt|injure)\b.{0,30}\b(?:someone|a\s+person|my|him|her|them)\b",
            re.IGNORECASE,
        ),
        ContentHarmCategory.WEAPONS,
        0.82,
        "Assault intent",
    ),
    (
        "how_to_rape",
        re.compile(
            r"\b(?:how\s+(?:to|can\s+I))\b.{0,15}\b(?:rape|sexually\s+assault|drug\s+(?:and\s+)?rape)\b", re.IGNORECASE
        ),
        ContentHarmCategory.EXPLOITATION,
        0.97,
        "Sexual assault intent",
    ),
    (
        "step_by_step_harm",
        re.compile(
            r"\b(?:step[- ]by[- ]step|detailed)\s+(?:guide|instructions?|tutorial|process|plan)\b.{0,30}\b(?:to\s+)?(?:kill|murder|poison|bomb|attack|hack|exploit|traffic|kidnap|abduct|stalk)",
            re.IGNORECASE,
        ),
        ContentHarmCategory.WEAPONS,
        0.90,
        "Step-by-step harmful instruction request",
    ),
    (
        "write_harmful",
        re.compile(
            r"\b(?:writ[a-z]*|generat[a-z]*|creat[a-z]*)\b.{0,20}\b(?:a\s+)?(?:manifesto|hit\s+list|kill\s+list|target\s+list|enemies?\s+list)",
            re.IGNORECASE,
        ),
        ContentHarmCategory.HARASSMENT,
        0.90,
        "Harmful list/manifesto creation",
    ),
    (
        "undetectable_harm",
        re.compile(
            r"\b(?:undetect(?:able|ed)|untraceab(?:le|ly)|without\s+(?:getting\s+caught|being\s+(?:caught|detected|traced)))\b.{0,30}\b(?:kill|murder|poison|hack|steal|crime|fraud|arson)",
            re.IGNORECASE,
        ),
        ContentHarmCategory.WEAPONS,
        0.92,
        "Undetectable harm request",
    ),
    (
        "harm_specific_person",
        re.compile(
            r"\b(?:how\s+(?:to|can\s+I)|ways?\s+to|help\s+me)\b.{0,20}\b(?:get\s+(?:revenge|back\s+at|even\s+with)|ruin|destroy)\b.{0,20}\b(?:someone'?s?|(?:his|her|their|my)\s+(?:ex'?s?|boss'?s?|neighbor'?s?))\b.{0,20}\b(?:life|career|reputation|marriage|relationship)",
            re.IGNORECASE,
        ),
        ContentHarmCategory.HARASSMENT,
        0.82,
        "Targeted revenge/harm",
    ),
]


# ---------------------------------------------------------------------------
# Core detection functions
# ---------------------------------------------------------------------------


def _check_negation(text: str, match_start: int, match_end: int, window: int = 120) -> bool:
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    context = text[start:end]
    return bool(_NEGATION_RE.search(context))


def _check_educational(text: str) -> bool:
    return bool(_EDUCATIONAL_RE.search(text))


def _scan_tier(
    text: str,
    patterns: list[_PatternEntry],
    tier: ContentHarmTier,
    severity: str,
    apply_negation: bool,
    is_educational: bool,
) -> list[ContentHarmFinding]:
    findings: list[ContentHarmFinding] = []
    for name, regex, category, confidence, description in patterns:
        for m in regex.finditer(text):
            conf = confidence
            negated = False

            if apply_negation and _check_negation(text, m.start(), m.end()):
                conf -= 0.35
                negated = True

            if is_educational and tier != ContentHarmTier.TIER1_CRITICAL:
                conf -= 0.25

            if conf <= 0.0:
                continue

            findings.append(
                ContentHarmFinding(
                    category=category,
                    tier=tier,
                    matched_keywords=(m.group(),),
                    severity=severity,
                    confidence=round(min(conf, 1.0), 3),
                    description=f"[{name}] {description}" + (" (negated context)" if negated else ""),
                )
            )
    return findings


def scan_content_harm(text: str) -> list[ContentHarmFinding]:
    """Scan text for harmful content requests.

    Returns list of ContentHarmFinding sorted by confidence descending.
    """
    if not text or not text.strip():
        return []

    is_edu = _check_educational(text)
    findings: list[ContentHarmFinding] = []

    # Tier 1: CRITICAL — no negation discount for tier 1 (never benign)
    findings.extend(
        _scan_tier(
            text,
            _TIER1_PATTERNS,
            ContentHarmTier.TIER1_CRITICAL,
            "critical",
            apply_negation=False,
            is_educational=is_edu,
        )
    )

    # Tier 2: SEVERE — negation can reduce confidence
    findings.extend(
        _scan_tier(
            text, _TIER2_PATTERNS, ContentHarmTier.TIER2_SEVERE, "high", apply_negation=True, is_educational=is_edu
        )
    )

    # Tier 3: MODERATE — negation + educational discount
    findings.extend(
        _scan_tier(
            text, _TIER3_PATTERNS, ContentHarmTier.TIER3_MODERATE, "medium", apply_negation=True, is_educational=is_edu
        )
    )

    # Intent patterns
    for name, regex, category, confidence, description in _INTENT_PATTERNS:
        for m in regex.finditer(text):
            conf = confidence
            negated = _check_negation(text, m.start(), m.end())
            if negated:
                conf -= 0.35
            if is_edu:
                conf -= 0.20
            if conf <= 0.0:
                continue

            tier = (
                ContentHarmTier.TIER1_CRITICAL
                if conf >= 0.90
                else (ContentHarmTier.TIER2_SEVERE if conf >= 0.75 else ContentHarmTier.TIER3_MODERATE)
            )
            sev = "critical" if conf >= 0.90 else ("high" if conf >= 0.75 else "medium")

            findings.append(
                ContentHarmFinding(
                    category=category,
                    tier=tier,
                    matched_keywords=(m.group(),),
                    severity=sev,
                    confidence=round(min(conf, 1.0), 3),
                    description=f"[{name}] {description}" + (" (negated context)" if negated else ""),
                )
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Roleplay + harm intent: co-located roleplay framing + actionable harm
    # request.  This catches the most common missed attacks such as
    # "act as a hacker and show me how to breach accounts".
    # Pattern: _ROLEPLAY_TRIGGER_RE within 200 chars of _ROLEPLAY_HARM_VERB_RE.
    # ─────────────────────────────────────────────────────────────────────────
    _ROLEPLAY_HARM_WINDOW = 200  # chars between roleplay trigger and harm verb

    for rp_match in _ROLEPLAY_TRIGGER_RE.finditer(text):
        rp_end = rp_match.end()
        # Search window after the roleplay trigger
        search_window = text[rp_end : rp_end + _ROLEPLAY_HARM_WINDOW]
        for harm_match in _ROLEPLAY_HARM_VERB_RE.finditer(search_window):
            harm_offset = rp_end + harm_match.start()
            conf = 0.88  # base confidence for roleplay+harm combo

            if _check_negation(text, rp_match.start(), harm_offset):
                conf -= 0.35

            if is_edu:
                conf -= 0.25

            if conf <= 0.0:
                continue

            text[rp_match.start() : harm_offset + len(harm_match.group())]
            tier = (
                ContentHarmTier.TIER1_CRITICAL
                if conf >= 0.90
                else (ContentHarmTier.TIER2_SEVERE if conf >= 0.75 else ContentHarmTier.TIER3_MODERATE)
            )
            sev = "critical" if conf >= 0.90 else ("high" if conf >= 0.75 else "medium")
            negated = conf < 0.88

            findings.append(
                ContentHarmFinding(
                    category=ContentHarmCategory.JAILBREAK,
                    tier=tier,
                    matched_keywords=(rp_match.group(), harm_match.group()),
                    severity=sev,
                    confidence=round(min(conf, 1.0), 3),
                    description=(
                        "[roleplay_plus_harm] Roleplay framing with actionable harm request"
                        + (" (negated context)" if negated else "")
                    ),
                )
            )

    # Deduplicate overlapping findings — keep highest confidence per category
    findings.sort(key=lambda f: f.confidence, reverse=True)
    return findings


def content_harm_risk_score(text: str) -> float:
    """Compute aggregate risk score for content harm (0.0-1.0)."""
    findings = scan_content_harm(text)
    if not findings:
        return 0.0

    for f in findings:
        if f.tier == ContentHarmTier.TIER1_CRITICAL and f.confidence > 0.8:
            return 0.95
    for f in findings:
        if f.tier == ContentHarmTier.TIER2_SEVERE and f.confidence > 0.7:
            return 0.80
    return min(max(f.confidence for f in findings) * 0.6, 1.0)

"""Dataclasses for the Simula-for-katana pipeline.

All structures are JSON-serializable (dataclasses + primitives only) so
the whole run can be resumed from a flat on-disk checkpoint directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# Canonical 8-attack + 1-clean label set, matching hermes-katana v9.
KATANA_V9_LABELS: tuple[str, ...] = (
    "clean",
    "content_injection",
    "semantic_manipulation",
    "behavioral_control",
    "exfiltration_attempt",
    "jailbreak",
    "cognitive_state_attack",
    "encoding_evasion",
    "persona_jailbreak",
)

# Injection-channel taxonomy (top-level partition orthogonal to label).
# We extend the proving-ground's 4 channels with finer-grained leaves.
CANONICAL_CHANNELS: tuple[str, ...] = (
    "file_content",
    "code_comment",
    "tool_output",
    "data_row",
)


@dataclass
class TaxonomyNode:
    """One node in the injection-mechanism taxonomy.

    Produced by step 1 (Global Diversification). The tree is rooted at
    `katana_v9_label`, has depth up to ~4, and leaves become sampling
    addresses for step 2.
    """

    node_id: str  # stable hash path e.g. "jailbreak/roleplay/historical"
    label: str  # katana_v9 label this subtree sits under
    title: str  # short name, e.g. "Historical persona jailbreak"
    description: str  # 1-3 sentences; the mechanism/intent
    parent_id: str | None = None
    depth: int = 0
    children: list[str] = field(default_factory=list)  # node_ids
    # Set only on leaves:
    is_leaf: bool = False
    estimated_prevalence: float = 0.0  # critic's priority weight in [0,1]
    examples_generated: int = 0  # filled by later stages

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetaPrompt:
    """A scenario description produced from one taxonomy leaf.

    Step 2 expands each leaf into several MetaPrompts. Step 3 may then
    pass a subset through a complexification refinement.
    """

    meta_id: str
    leaf_id: str  # foreign key → TaxonomyNode.node_id
    label: str
    channel: str  # one of CANONICAL_CHANNELS
    scenario: str  # natural-language scenario description
    intent: str  # one-liner: what the attacker wants
    style_hint: str = ""  # optional voice/register guidance
    complexified: bool = False
    complexity_level: int = 0  # 0=base, 1=refined, 2=double-refined…

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SynthExample:
    """One concrete generated attack text.

    Step 2 fills `text`. Step 4 fills the critic verdicts. Only examples
    where BOTH critics return accept=True are kept in the final corpus.
    """

    example_id: str
    meta_id: str  # foreign key → MetaPrompt.meta_id
    leaf_id: str  # foreign key → TaxonomyNode.node_id
    label: str  # katana_v9 label
    channel: str
    text: str
    # Filled by critics (step 4):
    critic_a: "CriticVerdict | None" = None
    critic_b: "CriticVerdict | None" = None
    keep: bool = False  # = critic_a.accept AND critic_b.accept
    # Reserved for Elo/complexity ratings (step 4b):
    elo: float = 1200.0
    # Provenance:
    teacher_model: str = ""  # which model wrote `text`
    generation_seed: int | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        # flatten Optional critic objects
        for k in ("critic_a", "critic_b"):
            if d[k] is not None:
                d[k] = d[k]  # already dict via asdict recursion
        return d


@dataclass
class CriticVerdict:
    """Output of one critic pass on a SynthExample."""

    critic_name: str  # e.g. "plausibility" or "target_would_comply"
    model: str  # which LLM ran the judgment
    accept: bool
    confidence: float  # [0,1]
    rationale: str  # short explanation
    failure_mode: str = ""  # taxonomy of reject reason, empty on accept

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRun:
    """Top-level manifest for one end-to-end generation campaign.

    Written to <checkpoint_dir>/run_meta.json at launch and updated on
    each step completion. Lets runs resume idempotently.
    """

    run_id: str
    started_at_iso: str
    config_path: str
    teacher_model: str
    critic_a_model: str
    critic_b_model: str
    # Counts filled incrementally:
    n_taxonomy_nodes: int = 0
    n_taxonomy_leaves: int = 0
    n_meta_prompts: int = 0
    n_examples_generated: int = 0
    n_examples_kept: int = 0
    # Step completion flags:
    taxonomy_done: bool = False
    meta_prompts_done: bool = False
    generation_done: bool = False
    critics_done: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

"""Step 1: Global Diversification — hierarchical taxonomy builder.

Recursively builds a tree of injection-mechanism subcategories under
each katana_v9 label. At every level:

  propose   — teacher generates N candidate children
  critic    — critic merges overlapping siblings, drops invalid ones,
              assigns prevalence weight
  refine    — if depth < max_depth AND node is non-trivial, recurse

Output: flat dict[node_id → TaxonomyNode] + explicit leaf list.

Non-obvious design choices:
- The seed corpus (confirmed_attacks.jsonl) is NOT used to drive
  sampling — Simula is seedless by design. We feed the seed ONLY at
  top-level as a domain anchor so the reasoning model knows what class
  of text we're expanding.
- Depth is capped at 3 so leaves stay actionable (a scenario described
  in one sentence) — deeper trees become noise and waste critic passes.
- Critic sees all siblings at once, enabling merges.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

from .llm import LLMClient, parse_json_block
from .schema import KATANA_V9_LABELS, TaxonomyNode


TAXONOMY_MAX_DEPTH = 3
TAXONOMY_N_CANDIDATES_PER_SPLIT = 7
MIN_CHILDREN_AFTER_CRITIC = 3


TAXONOMY_PROPOSE_SYSTEM = """You are a security researcher building a taxonomy \
of prompt-injection attack MECHANISMS (not examples). Each node in the \
taxonomy names a distinct *way* an attack can be constructed.

Respond with ONLY a JSON array of objects. Each object has:
  title       — 2-5 word name
  description — one to three sentences explaining the MECHANISM and why \
it works (not an example of text)

Do not output examples, surrounding prose, or markdown fences. JSON only.
"""


TAXONOMY_CRITIC_SYSTEM = """You review taxonomies of prompt-injection \
mechanisms. You will be shown a set of sibling subcategories under a \
shared parent. Your job:

  1. MERGE siblings that describe the same mechanism with different \
     wording. Return a single combined title/description for each merged \
     group.
  2. DROP siblings that aren't actually distinct injection mechanisms \
     (too vague, overlap the parent entirely, or describe defense rather \
     than attack).
  3. For each surviving sibling, assign `prevalence` in [0, 1] reflecting \
     how common it is in the wild (rough estimate).
  4. Decide for each sibling whether it is `leaf: true` (no meaningful \
     further decomposition) or `leaf: false` (still abstract enough to \
     split further).

Respond with ONLY a JSON array:
  [{"title": "...", "description": "...", "prevalence": 0.3, "leaf": false}, ...]

JSON only, no prose.
"""


def _node_id_for(parent_id: str | None, title: str) -> str:
    s = f"{parent_id or ''}/{title}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _propose_children(
    llm: LLMClient,
    parent: TaxonomyNode,
    *,
    seed_hint: str,
    n_candidates: int,
) -> list[dict]:
    user = (
        f"Parent category: **{parent.title}** (katana label: {parent.label})\n"
        f"Parent description: {parent.description}\n\n"
        f"{seed_hint}\n\n"
        f"Propose {n_candidates} distinct SUB-MECHANISMS of this category. "
        "Avoid overlap between siblings. Each should name a meaningfully different technique."
    )
    text = llm.complete(TAXONOMY_PROPOSE_SYSTEM, user, expect_json=True)
    return parse_json_block(text)


def _critic_filter(
    llm: LLMClient,
    parent: TaxonomyNode,
    candidates: list[dict],
) -> list[dict]:
    user = (
        f"Parent category: **{parent.title}** (katana label: {parent.label})\n"
        f"Depth of parent: {parent.depth}  (max_depth = {TAXONOMY_MAX_DEPTH})\n\n"
        f"Candidates:\n{json.dumps(candidates, indent=2)}\n\n"
        "Merge, drop, score, and classify as leaf/non-leaf. Return JSON array only."
    )
    text = llm.complete(TAXONOMY_CRITIC_SYSTEM, user, expect_json=True)
    return parse_json_block(text)


def _expand(
    llm: LLMClient,
    parent: TaxonomyNode,
    nodes: dict[str, TaxonomyNode],
    *,
    seed_hint: str,
    n_candidates: int,
    max_depth: int,
) -> None:
    if parent.depth >= max_depth:
        parent.is_leaf = True
        return
    try:
        proposed = _propose_children(llm, parent, seed_hint=seed_hint, n_candidates=n_candidates)
        print(
            f"  [propose] {parent.label}/{parent.title} (d={parent.depth}): {len(proposed)} candidates",
            flush=True,
        )
    except Exception as e:
        print(
            f"  [propose FAILED] {parent.label}/{parent.title}: {type(e).__name__}: {e}",
            flush=True,
        )
        parent.is_leaf = True
        parent.description += f"\n[propose_failed: {e}]"
        return
    try:
        survivors = _critic_filter(llm, parent, proposed)
        print(
            f"  [critic]  {parent.label}/{parent.title}: {len(survivors)} kept",
            flush=True,
        )
    except Exception as e:
        print(
            f"  [critic FAILED] {parent.label}/{parent.title}: {type(e).__name__}: {e}",
            flush=True,
        )
        survivors = proposed  # fall through uncritically

    if len(survivors) < MIN_CHILDREN_AFTER_CRITIC:
        parent.is_leaf = True
        return

    for child_spec in survivors:
        title = str(child_spec.get("title", "")).strip()
        desc = str(child_spec.get("description", "")).strip()
        if not title or not desc:
            continue
        child_id = _node_id_for(parent.node_id, title)
        if child_id in nodes:
            continue  # collision on parent path
        is_leaf = bool(child_spec.get("leaf", False)) or parent.depth + 1 >= max_depth
        child = TaxonomyNode(
            node_id=child_id,
            label=parent.label,
            title=title,
            description=desc,
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            is_leaf=is_leaf,
            estimated_prevalence=float(child_spec.get("prevalence", 0.0)),
        )
        nodes[child_id] = child
        parent.children.append(child_id)
        if not child.is_leaf:
            _expand(
                llm,
                child,
                nodes,
                seed_hint=seed_hint,
                n_candidates=n_candidates,
                max_depth=max_depth,
            )


def build_taxonomy(
    llm: LLMClient,
    *,
    seed_samples_by_label: dict[str, list[str]] | None = None,
    max_depth: int = TAXONOMY_MAX_DEPTH,
    n_candidates_per_split: int = TAXONOMY_N_CANDIDATES_PER_SPLIT,
) -> dict[str, TaxonomyNode]:
    """Build the full tree under all 8 attack labels (skips 'clean').

    `seed_samples_by_label` is optional: a few real examples per label
    let the reasoning model ground its taxonomy in the domain. Simula is
    seedless but a 1-line domain hint dramatically improves tree quality.
    """
    nodes: dict[str, TaxonomyNode] = {}
    for label in KATANA_V9_LABELS:
        if label == "clean":
            continue
        seed_examples = (seed_samples_by_label or {}).get(label, [])
        seed_hint = _format_seed_hint(label, seed_examples)
        root = TaxonomyNode(
            node_id=_node_id_for(None, label),
            label=label,
            title=label.replace("_", " ").title(),
            description=_TOP_LEVEL_DESCRIPTIONS[label],
            parent_id=None,
            depth=0,
            is_leaf=False,
        )
        nodes[root.node_id] = root
        _expand(
            llm,
            root,
            nodes,
            seed_hint=seed_hint,
            n_candidates=n_candidates_per_split,
            max_depth=max_depth,
        )
    return nodes


def _format_seed_hint(label: str, examples: list[str]) -> str:
    if not examples:
        return f"(No seeds provided for `{label}`. Reason from first principles.)"
    trimmed = [ex.strip().replace("\n", " ")[:240] for ex in examples[:3]]
    bullets = "\n".join(f"  - {t}" for t in trimmed)
    return (
        f"For domain grounding only, here are up to 3 real attack texts labeled `{label}`:\n"
        f"{bullets}\n"
        "Do NOT imitate them. Use them to understand the mechanism family, then propose "
        "sub-mechanisms that may or may not appear in the samples."
    )


_TOP_LEVEL_DESCRIPTIONS: dict[str, str] = {
    "content_injection": "Attacker causes the target model to emit specific attacker-chosen content, often by smuggling it in via delimiters, markers, or direct imitation of system instructions.",
    "semantic_manipulation": "Attacker reshapes how the target interprets task framing (context, authority, scope) so the model complies with a request it would otherwise refuse.",
    "behavioral_control": "Attacker instructs the target model to adopt specific output behaviors (always include X, never mention Y, change format) that override the system policy.",
    "exfiltration_attempt": "Attacker tries to extract secrets, hidden prompts, or sensitive data from the target — directly or via side channels (URLs, markdown images, tool calls).",
    "jailbreak": "Attacker disables safety/alignment constraints through framing, role-play, or hypothetical framing, causing the model to produce content it is trained to refuse.",
    "cognitive_state_attack": "Attacker manipulates the target model's reasoning state (contradictions, false confirmations, multi-turn gaslighting) to destabilize its decision procedure.",
    "encoding_evasion": "Attacker hides the payload in an encoding or transformation (base64, rot13, homoglyphs, zero-width chars) to bypass string-match scanners.",
    "persona_jailbreak": "Attacker assigns the target a persona that naturally excludes safety behavior (DAN-style, fictional evil AI, historical figure).",
}


def save_taxonomy(nodes: dict[str, TaxonomyNode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for n in nodes.values():
            f.write(json.dumps(n.to_json()) + "\n")


def load_taxonomy(path: Path) -> dict[str, TaxonomyNode]:
    nodes: dict[str, TaxonomyNode] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        n = TaxonomyNode(**d)
        nodes[n.node_id] = n
    return nodes


def iter_leaves(nodes: dict[str, TaxonomyNode]) -> Iterable[TaxonomyNode]:
    for n in nodes.values():
        if n.is_leaf:
            yield n


def seed_sampler(path: Path, n_per_label: int = 3) -> dict[str, list[str]]:
    """Grab up to n seed texts per label from a confirmed-attacks JSONL."""
    from collections import defaultdict

    buckets: dict[str, list[str]] = defaultdict(list)
    rnd = random.Random(42)
    with path.open(encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f if line.strip()]
    rnd.shuffle(all_rows)
    for r in all_rows:
        lbl = r.get("label")
        if not lbl or lbl == "clean":
            continue
        if len(buckets[lbl]) >= n_per_label:
            continue
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        buckets[lbl].append(txt)
    return dict(buckets)

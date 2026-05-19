"""Simula-style synthetic-data generation for prompt-injection corpora.

Implements the four-step recipe from Google Research (Davidson & Harkous,
2026, "Reasoning-Driven Synthetic Data Generation and Evaluation"):

  1. Global Diversification   — hierarchical taxonomy via propose/critic/refine
  2. Local Diversification    — 1-of-N meta-prompting at each leaf
  3. Complexification         — orthogonal difficulty axis
  4. Quality Checks           — dual independent critics

Output: a labeled corpus that maps 1-to-1 onto hermes-katana's v9 label
schema, ready for DeBERTa-v3 training and zvec centroid fitting.
"""

from .schema import (
    TaxonomyNode,
    MetaPrompt,
    SynthExample,
    CriticVerdict,
    GenerationRun,
)

__all__ = [
    "TaxonomyNode",
    "MetaPrompt",
    "SynthExample",
    "CriticVerdict",
    "GenerationRun",
]

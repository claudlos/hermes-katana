"""Research engine for proving-grounds.

Turns the fleet + corpus infrastructure (Phase 0-1) into a scientific-rigor
research agent (Phase 2) that proposes, tests, and validates hypotheses
about LLM and agent-harness security.

Modules:
  statistics — Wilson CIs, McNemar, paired bootstrap, Cohen's h, AUC
  registry   — hypothesis preregistration (YAML lock files)
  rigor      — statistical-rigor contract enforced at the type level
  events     — typed event stream (Action / Observation / Hypothesis / ...)
  dag        — hypothesis DAG that survives context compaction
  budget     — ledger for claude_max / openai_usd / gpu_thermal
  doom       — loop-fingerprint detector
  verifier   — separate critic that audits claims against raw evidence
  tools      — proving-grounds-specific tool router for the agent
  kernel     — ResearchKernel: composes all of the above
"""

__all__: list[str] = []

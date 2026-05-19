# Tiny Scabbard Research Track

Tiny Scabbard is the V3.1 research path for a classifier smaller and faster than the current MiniLM ONNX artifact.
It should not block the V3 release.

## Goal

Find a CPU-friendly classifier that can replace or complement `katana_v15_distill_minilm_onnx` in the `fast_cpu`
profile while preserving Katana's scanner gates.

## Candidate families

- Tiny transformer encoder exported to ONNX or ONNX int8.
- Static embedding plus linear or GBDT classifier.
- Character/ngram triage model paired with existing rule scanners.

BusyBee-style CPU mini-agents and Hirundo/Gemma local judges are separate later tracks. They are useful for risky
web/search/tool-policy work, but they should not be treated as the first MiniLM replacement.

## Promotion gate

A candidate can be promoted only if it clears all of these:

- Artifact size is smaller than the current MiniLM ONNX artifact.
- p95 CPU latency is lower than MiniLM on `scripts/benchmark_hermes_katana_tool_sandbox.py`.
- Attack detection does not materially regress on confirmed, evasion, and adversarial integration suites.
- False-positive rate does not regress on benign tool args, docs, code, URLs, paths, and command-like strings.
- Disagreements against MiniLM are exported for review before promotion.

## Baseline command

```bash
PYTHONPATH=src python scripts/benchmark_hermes_katana_tool_sandbox.py \
  --variants base,minilm \
  --stack scabbard-only \
  --suites benign_tool_args,content_payloads,tool_outputs \
  --route-mode balanced \
  --warmups 1 \
  --iterations 2
```

Store candidate artifacts outside GitHub and register them through the artifact registry only after the promotion gate
has a repeatable result.

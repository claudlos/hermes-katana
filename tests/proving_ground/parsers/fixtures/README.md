# Parser fixtures

This directory is reserved for real-world stdout captures from agent CLIs used
by the optional Proving Ground harness. No raw CLI captures are shipped in the
public repository.

The parser tests still include inline smoke cases and will skip optional
fixture-backed cases when the corresponding capture is absent.

## Adding A Fixture

1. Capture a real CLI run that exhibits a specific format or failure mode.
2. Redact personal data, API keys, workspace paths, hostnames, and run IDs.
3. Save raw stdout as `<driver>/<scenario>.txt`.
4. Add the expected parsed tool-call list to
   `tests/proving_ground/parsers/test_agent_parsers.py`.

Pinned fixtures are useful when a parser bug was observed in a real run and the
captured output is safe to include. Keep private or sensitive captures outside
the public checkout.

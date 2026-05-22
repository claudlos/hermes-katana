# Parser fixtures

This directory contains safe stdout fixtures for agent CLIs used by the optional
Proving Ground harness. Shipped fixtures should be sanitized captures or minimal
reproductions of formats observed in real runs.

Raw private CLI captures should not be committed to the public repository.

## Adding A Fixture

1. Capture or reduce a CLI run that exhibits a specific format or failure mode.
2. Redact personal data, API keys, workspace paths, hostnames, and run IDs.
3. Save safe stdout as `<driver>/<scenario>.txt`.
4. Add the expected parsed tool-call list to
   `tests/proving_ground/parsers/test_agent_parsers.py`.

Pinned fixtures are useful when a parser bug was observed in a real run and the
captured output is safe to include. Keep private or sensitive captures outside
the public checkout.

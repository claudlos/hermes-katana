# Security Policy

## Supported Versions

| Version | Supported           |
|---------|---------------------|
| 3.0.x   | Yes                 |
| 2.0.x   | Security fixes only |
| < 2.0   | No                  |

## Reporting a Vulnerability

Please report security vulnerabilities **privately** through one of
the channels below — not via public issues or pull requests, which
would expose the vulnerability before a fix is available.

### Preferred: GitHub Security Advisories

**[Report a vulnerability](https://github.com/claudlos/hermes-katana/security/advisories/new)** directly through GitHub.
The advisory is private until the maintainers and you publish it
together once a fix is shipped.

### Email

`carlosian@agentmail.to`

### What to include

In either channel, please include:

- a description of the vulnerability,
- steps to reproduce (proof-of-concept welcome),
- a potential impact assessment, and
- the Hermes Katana version (`katana version`) and any relevant
  environment details.

### What to expect

- An acknowledgment within **48 hours**.
- A fix developed privately and shipped as a patch version on the
  affected supported lines (see the table above).
- A credited GitHub Security Advisory published alongside the fix,
  unless you ask to remain anonymous.

## Scope

Hermes Katana is a security toolkit designed to protect LLM agents.
The following are in scope:

- Bypasses of taint tracking that allow untrusted data to reach sinks
- Policy engine evaluation errors that produce incorrect allow/deny decisions
- Vault encryption weaknesses or key exposure
- Audit trail hash-chain forgery
- Scanner evasion techniques not covered by existing patterns
- Proxy secret scrubbing bypasses

## Out of Scope

- Denial of service against the toolkit itself (not a production service)
- Social engineering
- Issues in dependencies (report upstream)

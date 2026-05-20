# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 3.0.x   | Yes                |
| 2.0.x   | Security fixes only |
| < 1.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in HermesKatana, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities.
2. Email **claudlos@users.noreply.github.com** with:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
3. You will receive an acknowledgment within 48 hours.
4. A fix will be developed and released as a patch version.

## Scope

HermesKatana is a security toolkit designed to protect LLM agents. The following are in scope:

- Bypasses of taint tracking that allow untrusted data to reach sinks
- Policy engine evaluation errors that produce incorrect allow/deny decisions
- Vault encryption weaknesses or key exposure
- Audit trail hash chain forgery
- Scanner evasion techniques not covered by existing patterns
- Proxy secret scrubbing bypasses

## Out of Scope

- Denial of service against the toolkit itself (not a production service)
- Social engineering
- Issues in dependencies (report upstream)

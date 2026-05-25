# Proxy and Vault Threat Model

This threat model covers the high-risk runtime path where HermesKatana reads vault material, injects provider credentials into proxied LLM traffic, scans request and response surfaces, and records audit evidence.

## Assets

- Vault master key and encrypted vault file.
- Provider credentials returned by `Vault.get()`.
- Injected request headers such as `Authorization`, `x-api-key`, and `x-goog-api-key`.
- Request and response bodies, headers, URLs, query values, cookies, and WebSocket frames.
- Audit trail entries and local proxy logs.
- Child process environment built by the proxy runner.
- Release artifacts, SBOMs, and attestations used to ship the runtime.

## Trust Boundaries

| Boundary | Trusted side | Untrusted side | Required control |
|---|---|---|---|
| Local CLI to vault | Local operator process | Filesystem, stale locks, corrupted vault JSON | Authenticated encryption, integrity checks, file locking, lock sentinel |
| Vault to proxy addon | In-process vault API | HTTP flow data and scanner findings | Per-request value collection only; no long-lived plaintext cache |
| Proxy to upstream provider | Local proxy | Network and remote provider | `tls_verify=true` before credential injection |
| Scanner boundary | Katana scanner APIs | Encoded, compressed, malformed, multipart, binary, or oversized payloads | Decode known encodings, scan multipart parts, fail closed in strict/max |
| Proxy runner to child process | Minimal allowlisted env | Parent `os.environ` | Build child env from allowlist and strip vault/secret material |
| Audit/log boundary | Operator-visible evidence | Secret-bearing details and scanner summaries | Redact credential-like text and current vault values before logging |
| Release boundary | Reviewed source and workflows | Published artifacts | Release gate, SBOM, provenance attestation, trusted publishing/OIDC |

## Primary Attacker Paths

1. Indirect prompt injection in upstream content tries to exfiltrate a vault value through a tool call or provider request.
2. Malformed multipart, invalid compression, unsupported encoding, binary wrapper, or oversized body tries to bypass scanner coverage.
3. A hostile local environment variable tries to leak into the proxy child process.
4. `tls_verify=false` downgrades upstream transport while credential injection is enabled.
5. Scanner summaries or exception messages include credential material and leak through logs, block responses, or audit details.
6. A compromised release path publishes an artifact without matching SBOM/provenance evidence.

## Required Mitigations

- Credential injection must only happen for exact registered provider domains, empty target auth headers, present vault values, and verified TLS.
- Injected credentials must be excluded from later header scanning, while user-supplied auth headers remain scan targets.
- Vault values used for scanning must be collected per request or response and then released by normal stack unwinding.
- Request and response scanning must cover URL, headers, query parameters, cookies, body text, binary-like bodies, multipart parts, supported compression, and WebSocket frames.
- Unsupported encodings, decompression failures, scanner exceptions, and oversized strict/max traffic must fail closed.
- Permissive mode may allow some failed or oversized traffic, but audit details must record that the tail or failed scope was not fully scanned.
- Logs, audit entries, and block messages must not include provider tokens, authorization headers, passwords, secret parameters, or current vault values.
- Release artifacts must be built by the release gate, uploaded with an SBOM, and attested on non-PR events.
- Publishing must use PyPI trusted publishing/OIDC when enabled; long-lived PyPI API tokens are not part of the intended release path.

## Residual Risks

- The proxy runs in the local operator environment. A compromised local account can still observe process memory or tamper with local files outside this project boundary.
- Permissive mode intentionally trades safety for continuity. Use strict or max for real secret-bearing operation.
- Unknown compression formats are blocked rather than decoded; operators may need an explicit compatibility change for legitimate providers using new encodings.
- Logs redact known high-risk credential shapes and current vault values, but arbitrary natural-language secrets can still require source-specific redaction rules.
- Artifact attestations prove workflow origin, not semantic correctness of the release.

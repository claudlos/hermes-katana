# HermesKatana — Project Handoff

Location: C:\Users\example\HermesKatana\hermes-katana
Workspace root: C:\Users\example\HermesKatana (contains all three source repos)

---

## What This Is

HermesKatana is a defense-in-depth security toolkit for the Hermes AI agent. It
blocks prompt injection attacks at 7 layers: detection, taint tracking, policy
enforcement, MITM proxy, vault encryption, audit logging, and middleware dispatch
hooks. Built by synthesizing Google's CaMeL research paper (arXiv:2503.18813),
hermes-aegis, and camelup into something more complete than any of them alone.

---

## Project State

Git commits:
  a3089f2  HermesKatana v0.1.0 — initial build
  6042905  sweep: research docs + apply 5 targeted improvements, 225 tests green

Test suite: 225 passing, 0 failing, 1.70s
All 36 source files import cleanly.

---

## Repository Layout

C:\Users\example\HermesKatana\
  camel-prompt-injection\      Google Research CaMeL reference repo (54 Python files)
  hermes-aegis\                Tranquil-Flow MITM proxy security layer (113 Python files)
  camelup\                     nativ3ai installer script (Python CLI)
  hermes-katana\               The unified toolkit (this project)

hermes-katana\
  src\hermes_katana\
    taint\          CaMeL-inspired data flow tracking
    policy\         Declarative YAML policy engine
    scanner\        Multi-strategy injection/secret/command/content/unicode detection
    proxy\          mitmproxy-based MITM proxy with API key injection
    vault\          AES-256-GCM encrypted secret store
    audit\          SHA-256 hash-chained append-only audit trail
    middleware\     Pre/post dispatch chain (wires taint+scan+policy into tool calls)
    installer\      Patch-based Hermes integration
    cli\            Rich CLI (20+ commands)
    config.py       Central config with env var overrides
  tests\
    unit\           218 original tests covering all core modules
    integration\    15 end-to-end flow tests
  docs\research\    10 research files (9260 lines) from the sweep session
  policies\         5 YAML policy files (paranoid, balanced, permissive + 2 examples)
  pyproject.toml    Package config, entry points (katana / hermes-katana)

---

## What Was Built (v0.1.0)

Seven defense layers, all wired together:

  Layer 1 — scanner/
    injection.py    47 heuristic patterns across 7 InjectionCategory types
                    Structural analysis (instruction density, topic shift)
                    Encoding detection (base64, hex, URL, Unicode, ROT13)
    secrets.py      20 API key regex patterns + Shannon entropy (4.5 bits)
                    Multi-encoding: base64/hex/URL/reversed/ROT13 variants
                    Vault exact-value matching
    commands.py     61 dangerous command patterns across 15 CommandCategory types
                    CRITICAL/HIGH/MEDIUM/LOW severity
                    Container escape, crypto mining, data staging, SSH exfil
    content.py      Homograph URL detection (30+ confusable chars)
                    ANSI/terminal escape detection
                    Code injection (eval, exec, os.system, pickle)
                    Markdown injection, HTML/SVG injection
    unicode.py      13 bidi override chars (CVE-2021-42574)
                    11 zero-width chars + ZWJ binary encoding detection
                    Unicode Tags block U+E0000-U+E007F (arXiv:2603.00164)
                    70+ homoglyphs, 5 mixed-script combinations
                    normalize_text() and normalize_and_scan()

  Layer 2 — taint/
    labels.py       TaintLabel enum: USER/SYSTEM/TOOL_OUTPUT/WEB_CONTENT/FILE_CONTENT/
                      MEMORY/MCP/MCP_TOOL_DESCRIPTION/MCP_TOOL_RESULT/MCP_RESOURCE/
                      MCP_PROMPT/AGENT/AGENT_DELEGATED/CROSS_SESSION/UNKNOWN (15 labels)
                    TrustLevel: TRUSTED/UNTRUSTED/CONDITIONAL
                    Source frozen dataclass with 12 factory methods
    value.py        TaintedValue[T] generic wrapper with dependency graph
                    TaintedStr — character-level taint tracking via CharTaint map
                      Supports: __add__, __radd__, __getitem__, upper/lower/strip/split/
                                replace/join/format — all propagate taint correctly
                    TaintedList (MutableSequence), TaintedDict (MutableMapping)
                    unwrap() and collect_sources() utilities
    flow.py         FlowDecision: ALLOW/DENY/ASK_USER/QUARANTINE
                    FlowRule frozen dataclass with priority ordering
                    FlowAnalyzer with 7 default CaMeL-inspired rules:
                      - Untrusted labels → critical sinks: DENY
                      - Conditional labels → critical sinks: ASK_USER
                      - Trusted (USER/SYSTEM) → anything: ALLOW
                      - AGENT → critical sinks: QUARANTINE
                      - MCP → skill_manage: DENY (skill mutation attack)
                      - Untrusted → delegate_task: ASK_USER (sub-agent amplification)
                      - MEMORY/CROSS_SESSION → send_message: DENY (stored injection chain)
                    UNTRUSTED_LABELS and CONDITIONAL_LABELS sets auto-include new labels
    tracker.py      TaintTracker singleton
                    register(value, source) → TaintedValue (auto-type-wraps)
                    propagate(result, *inputs) → TaintedValue
                    check_flow(value, tool_name) → FlowDecision
                    get_taint_chain(value) → list[Source] (full provenance DFS)
                    check_args_flow(kwargs, tool_name) → dict[str, FlowDecision]
                    TaintTracker.scoped() context manager for isolated sessions
                    Thread-safe with mutex

  Layer 3 — policy/
    models.py       Policy (pydantic), PolicyResult (ALLOW/DENY/ESCALATE/LOG_ONLY)
                    ConditionOperator: contains_taint/source_is/reader_lacks/
                      matches_pattern/argument_matches/taint_level_gte/has_label
                    PolicySet with version, inheritance, merge()
    defaults.py     PARANOID (14 rules), BALANCED (15 rules), PERMISSIVE (14 rules)
    engine.py       PolicyEngine.with_defaults(preset) factory
                    evaluate(tool_name, args, taint_context) → EvaluationResult
                    Thread-safe RLock, glob matching, priority ordering
                    evaluate_batch(), add/remove/list/clear policies
    yaml_loader.py  load_policy_file(), validate_policy_yaml(), export_policy_set()
                    Inheritance: "extends: balanced" merges parent policies
                    PolicyFileWatcher for hot-reload
    policies/       balanced.yaml, paranoid.yaml, permissive.yaml
                    examples/banking.yaml, examples/code-review.yaml

  Layer 4 — proxy/
    config.py       ProxyConfig pydantic: port, host, scan modes, rate limits
    injector.py     12 LLM providers: OpenAI/Anthropic/Google/Groq/Together/
                      OpenRouter/Vercel/DeepSeek/Mistral/Cohere/Replicate/HuggingFace
                    O(1) domain lookup index, inject_credentials(flow, vault)
    addon.py        KatanaAddon mitmproxy addon
                    request(): domain allowlist, rate limit, secret scan, injection
                      detect, credential injection
                    response(): content scan, indirect injection defense
                    X-Katana-Scanned header for downstream awareness
    runner.py       KatanaProxy lifecycle: start/stop/status/is_running
                    Cross-platform file locking (fcntl/msvcrt)
                    Atomic PID file writes (tmp + rename)
                    Watchdog thread with auto-restart on same port
                    Health check HTTP endpoint

  Layer 5 — vault/
    store.py        AES-256-GCM per-value encryption (not Fernet AES-128-CBC)
                    Random 96-bit nonces per value
                    Master key in OS keyring (Keychain / Secret Service / Credential Locker)
                    HMAC-SHA256 integrity over all entries
                    Circuit breaker: vault.lock sentinel blocks all reads
                    Atomic writes via tmp+rename
                    Key rotation: re-encrypts all values with new master key
                    verify_integrity() public method
    migrate.py      Secret discovery from env vars, hermes config.yaml, .env files
                    Secure delete (overwrite with zeros before unlink)
                    27 secret key patterns, priority: env > hermes_config > dotenv

  Layer 6 — audit/
    trail.py        SHA-256 hash-chained append-only JSONL log
                    O(1) last-hash caching (fixed from hermes-aegis O(n) reads)
                    Cross-platform file locking (fcntl/msvcrt)
                    Auto-rotation at 10MB threshold
                    11 AuditEventType values
                    verify_chain() tamper detection
                    query(filters) and stats()

  Layer 7 — middleware/
    chain.py        KatanaMiddleware ABC, MiddlewareChain
                    DispatchDecision: ALLOW/DENY/ESCALATE
                    CallContext pydantic with deny()/escalate() helpers
                    DENY short-circuits chain immediately
                    Post-dispatch runs in reverse order (onion model)
    integration.py  KatanaTaintMiddleware (priority 100)
                    KatanaScanMiddleware (priority 80)
                    KatanaPolicyMiddleware (priority 60)
                    KatanaAuditMiddleware (priority 20)
                    create_default_chain(config) factory

  installer/
    patches.py      5 core patches for Hermes source files (idempotent, sentinel-based)
    installer.py    KatanaInstaller: detect_hermes/install/uninstall/verify/status

  cli/
    main.py         Click-based CLI: doctor/install/uninstall/run/scan/scan-file/
                      scan-command/policy/vault/audit/proxy/status/benchmark/version
                    Rich colored output, exit codes: 0 success, 1 error, 2 security issue

  config.py         KatanaConfig pydantic: policy_preset, scan toggles, proxy settings,
                    vault settings, audit settings, taint_tracking, strict_mode, log_level
                    Loads from ~/.hermes-katana/config.yaml with KATANA_* env var overrides

---

## Research Corpus

docs/research/ — 10 files, 9260 lines. Written from real sources.

  01-prompt-injection.md        CaMeL paper analysis, AgentDojo results, Crescendo,
                                Many-Shot, Skeleton Key, 25 specific improvements
  02-taint-tracking-capabilities.md  DTA theory, Biba IFC model, CHERI, CaMeL
                                implementation analysis, 25 improvements
  03-mcp-and-multiagent-security.md  Real Invariant Labs attacks, WhatsApp exfil demo
                                (1224 lines), ETDI paper (arXiv:2506.01333),
                                MCPDescriptionPinner design, 20 improvements
  04-cryptography-secret-management.md  AES-256-GCM nonce theory, Argon2id KDF,
                                key hierarchy, BLAKE3, 22 improvements
  05-unicode-attacks.md         Trojan Source (CVE-2021-42574), arXiv:2603.00164
                                (Unicode Tags attack, Feb 2026), ZWJ binary encoding,
                                17 improvements
  06-dangerous-commands-container-security.md  Container escape techniques,
                                848 lines, 25 improvements including new patterns
  07-behavioral-anomaly-reactive-agents.md  SentinelAgent (arXiv:2505.24201),
                                1168 lines, KatanaSessionBudgetMiddleware design
  08-proxy-architecture.md      mitmproxy official docs deep dive, 1075 lines,
                                WebSocket gap (OpenAI Realtime API unmonitored),
                                25 improvements
  09-policy-engines.md          OPA + Rego, Cedar (ETDI paper), 1248 lines,
                                24 improvements including OPA integration path
  10-benchmarking-redteam.md    Real AgentDojo results table (all 27 model/defense
                                combos from live site), red-team payload library
                                design, 16 improvements

---

## Sweep Changes (commit 6042905)

Applied directly from research findings:

  taint/labels.py
    + 6 new TaintLabel values: MCP_TOOL_DESCRIPTION, MCP_TOOL_RESULT, MCP_RESOURCE,
        MCP_PROMPT, AGENT_DELEGATED, CROSS_SESSION
    + Trust mappings for all new labels
    + 4 new Source factory methods

  taint/flow.py
    + New labels added to UNTRUSTED_LABELS and CONDITIONAL_LABELS sets
    + Rule 5: MCP → skill_manage DENY
    + Rule 6: Untrusted → delegate_task ASK_USER
    + Rule 7: MEMORY/CROSS_SESSION → send_message DENY

  scanner/unicode.py (critical gap fixed)
    + Unicode Tags block detection (U+E0000-U+E007F) — CRITICAL severity
    + Payload decoding for audit report (shows what was hidden)
    + ZWJ binary-encoding detection (8+ consecutive ZWSP/ZWNJ)
    + normalize_text() now strips Tags block chars
    + normalize_and_scan() runs Tags + ZWJ checks first

  scanner/injection.py
    + 10 new patterns: mcp_important_tag, mcp_important_prefix,
        concealment_instruction, file_exfil_in_description, cross_agent_redirect,
        recipient_hijack, authorized_override_framing, goal_hijacking,
        memory_instruction_injection, whitespace_padding_exfil

  tests
    + 7 new tests for new labels and unicode detection
    + Stray test_taint.py (project root) deleted
    + Duplicate 03-mcp-multiagent-security.md deleted

---

## What Doesn't Exist Yet (highest-value next work)

From the research docs, in priority order:

  P0  MCPDescriptionPinner (docs/research/03, Section 9.3)
      New module: src/hermes_katana/mcp/pinner.py
      Pins SHA-256 of tool descriptions at install time, detects rug-pull attacks
      when description changes between approval and execution. Single biggest
      missing piece for MCP security.

  P0  AgentDojo integration benchmark (docs/research/10, Section 5.3)
      tests/benchmark/agentdojo_integration.py
      Runs HermesKatana middleware against real AgentDojo scenarios.
      The only way to know if the defenses actually work end-to-end.
      From real results table: claude-3-7-sonnet baseline is 7.31% ASR;
      a good defense with BALANCED preset should target <= 5%.

  P1  Argon2id for headless vault key derivation (research/04, Section 2.1)
      vault/store.py: when OS keyring unavailable, derive master key from
      passphrase using Argon2id (not PBKDF2). pip install argon2-cffi.

  P1  WebSocket scanning hook (research/08, Section 2)
      proxy/addon.py: add websocket_message() event hook.
      OpenAI Realtime API uses WebSocket — currently completely unmonitored.

  P1  HKDF-based DEK key hierarchy (research/04, Section 2.2)
      vault/store.py: separate Data Encryption Keys per category
      (LLM keys, git tokens, MCP credentials) derived via HKDF-SHA256.

  P2  KatanaSessionBudgetMiddleware (research/07, Section 7)
      middleware/integration.py: trip circuit after N anomalies per session.
      Catches Crescendo-style multi-turn escalation attacks.

  P2  stdio MCP tool dispatch hook (research/03, Section 9.5)
      installer/patches.py: patch Hermes tool dispatch at the Python level
      to scan MCP tool arguments/results before they reach the LLM.
      stdio MCP is invisible to the MITM proxy.

  P3  Scanner precision/recall benchmark (research/10, Section 5.1)
      tests/benchmark/scanner_benchmark.py: 100 injection payloads, 100 clean
      samples, measure F1 score. Current scanner has 47 injection patterns but
      no measurement of false-positive rate.

  P3  Policy false-positive rate benchmark (research/10, Section 5.2)
      tests/benchmark/policy_benchmark.py: measure FP rate per preset.
      Target: BALANCED <= 5%, PERMISSIVE <= 1%.

---

## Known Bugs in Source Repos (not in HermesKatana)

These were found in the CaMeL and hermes-aegis codebases during the review.
Not present in HermesKatana. Documented here for reference:

  camel-prompt-injection/
    workspace.py:129   delete_email_policy logic is INVERTED (deny when trusted=True)
    travel.py:72-75    reserve_car_rental checks "restaurant" field, reserve_restaurant
                       checks "company" field — swapped
    conditional_cache.py  Creates new @lru_cache inside wrapper on every call —
                          cache never hits
    interpreter.py:1448   raise e before unreachable return — dead code
    chat_turn.py:55       stray "logout" text at end of file

  hermes-aegis/
    dangerous_blocker.py:105  Raises SecurityError instead of returning DENY —
                               breaks middleware chain contract
    trail.py:162-179          O(1) last-hash reads (fixed in HermesKatana)
    trail.py:226-227          No file locking on audit trail writes (fixed)
    vault/store.py             AES-128-CBC (Fernet) — fixed to AES-256-GCM

---

## Differences vs Source Repos

  vs hermes-aegis
    Encryption:      AES-256-GCM vs AES-128-CBC (Fernet)
    Audit trail:     O(1) last-hash, file locking vs O(n) reads, no locking
    Taint tracking:  Full CaMeL-inspired system (hermes-aegis has none)
    Policy engine:   Declarative YAML (hermes-aegis has none)
    Secret patterns: 20 patterns + entropy vs 7 patterns
    Commands:        61 patterns vs ~30 patterns
    Unicode:         Tags block, ZWJ binary encoding, 70+ homoglyphs vs 16 chars
    LLM providers:   12 (adds DeepSeek, Mistral, Cohere, Replicate, HuggingFace)

  vs camel-prompt-injection
    Deployment:      Drop-in middleware, no custom interpreter needed
    Taint labels:    15 labels including MCP granularity vs 5 sources
    Policies:        YAML hot-reload vs hardcoded Python
    Scanner:         Multi-layer detection (hermes-aegis patterns + CaMeL policy)
    Bugs fixed:      Inverted delete policy, swapped travel fields, broken cache

  vs camelup
    Purpose:         Full toolkit vs installer-only
    Integration:     Patch-based middleware hooks vs git branch wiring

---

## Running the Project

```bash
cd /mnt/c/Users/example/HermesKatana/hermes-katana
source ~/hermes-venv/bin/activate

# Tests
PYTHONPATH=src pytest tests/ -v

# Quick import check
python3 -c "import sys; sys.path.insert(0, 'src'); from hermes_katana.taint import TaintTracker; from hermes_katana.policy import PolicyEngine; from hermes_katana.scanner import scan_input; print('OK')"

# Install for use
pip install -e .
katana --help
```

---

## Key Design Decisions

1. No custom interpreter. CaMeL required its restricted Python interpreter to
   enforce taint at runtime. HermesKatana instead enforces at tool dispatch via
   middleware — simpler, drop-in, no code changes to the agent itself.

2. YAML policies over hardcoded logic. Every security rule is a declarative YAML
   file that can be versioned, shared, and hot-reloaded without restarting.

3. Proxy-level secret injection. API keys exist only in the encrypted vault and
   the proxy process. The agent process never sees real keys.

4. Research-first. The 10 research files (9260 lines) in docs/research/ document
   the decisions behind every module. Before extending anything, read the relevant
   research file — it contains the attack context, the tradeoffs, and a prioritized
   improvement list.

5. Fail-safe over fail-open. Middleware errors produce DENY, not ALLOW. The
   circuit breaker stops all operations rather than silently degrading.

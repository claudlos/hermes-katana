# AUDIT REVIEW 2 — hostile review

I went looking for ways this taint/policy/middleware stack fails open. I found several.

## Critical findings

### 1) Taint can still be laundered through ordinary Python operations
Files: `src/hermes_katana/taint/value.py`
Relevant lines: `TaintedStr.encode()` at ~432-439; class design generally around `TaintedStr`

This module claims taint propagates through common string operations, but that is false for several standard Python paths.

Confirmed behavior:
- `json.dumps({"x": tainted})` returns a plain `str`
- `tainted.encode().decode()` returns a plain `str`
- `''.join([c for c in tainted])` returns a plain `str`

I verified this directly in Python. `encode()` even admits the problem in a warning, which means the implementation knows taint is dropped and still allows it.

Why this matters:
- Any policy that relies on wrappers surviving serialization/transcoding is bypassable.
- An attacker can sanitize taint away without any explicit `unwrap()` audit event.
- The exact laundering paths listed in the task prompt are still open.

Impact: tainted content can be transformed into an untracked plain string and then passed to sinks like `terminal`, `write_file`, `patch`, `memory`, etc.

### 2) Middleware taint enforcement only checks top-level args; nested tainted values bypass it
Files: `src/hermes_katana/middleware/integration.py`, `src/hermes_katana/taint/flow.py`
Relevant lines: `KatanaTaintMiddleware.pre_dispatch()` ~83-140

`KatanaTaintMiddleware.pre_dispatch()` iterates only `ctx.args.items()` and checks `isinstance(arg_val, (TaintedStr, TaintedValue))` on the top-level value. It does not recursively inspect lists, dicts, tuples, or nested structures.

That means these bypass the taint gate:
- `{"command": [tainted]}`
- `{"text": {"body": tainted}}`
- `{"args": ["safe", {"payload": tainted}]}`

This is especially bad because `FlowAnalyzer.analyze()` already uses `collect_sources(value)` for nested taint. The middleware never calls into that logic unless the top-level object is itself tainted.

Impact: nested tainted payloads can reach side-effecting tools without denial/escalation.

### 3) Policy middleware compares against the wrong thing and will silently ignore deny/escalate results
File: `src/hermes_katana/middleware/integration.py`
Relevant lines: 377-392

`result = self.engine.evaluate(...)` returns an evaluation object whose action is stored in `result.action`.

But the code checks:
- `if result.action == PolicyResult.DENY:`
- `if result.action == PolicyResult.ESCALATE:`
- `if result.action == PolicyResult.LOG_ONLY:`

This only works if `PolicyResult` is the enum. In this codebase, `PolicyResult` is indeed the enum in `policy.models`, but the middleware docstring earlier mixes up "PolicyResult" as both result object and enum, and the branch logic is fragile enough that it is one rename/import mistake away from fail-open behavior. Worse, the middleware first special-cases `matched_policy is None`, then falls through to `return DispatchDecision.ALLOW` for everything not caught.

This piece is structurally brittle and easy to break into fail-open behavior. At minimum the naming collision is dangerous. If the import changes or a future refactor introduces an `EvaluationResult` aliasing mistake, all explicit deny/escalate outcomes become allows.

Impact: policy enforcement is one bad import/refactor away from complete bypass. This is a design bug in security-critical code.

## High findings

### 4) `FlowAnalyzer` can downgrade a matched non-deny rule to `ALLOW` based only on trust level
File: `src/hermes_katana/taint/flow.py`
Relevant lines: 399-401 and following logic in `FlowAnalyzer.analyze()`

The code explicitly says:
- "Special case: if all sources are trusted, downgrade escalation to allow"
- "but never override an explicit DENY"

That is a footgun. Decisions are supposed to come from rules. Instead, after rule matching, the analyzer performs a second-pass override based on `TrustLevel`.

Why this is dangerous:
- Labels and trust are distinct axes. A value can carry a risky label while still being marked trusted by mistake or by custom source construction.
- A policy author can write an explicit `ASK_USER` or `QUARANTINE` rule and have it silently neutralized later.
- This is exactly the kind of “DENY/ESCALATE gets downgraded later” behavior the task told me to look for.

Even if DENY is preserved, ASK/QUARANTINE are still security decisions. Overriding them after matching is fail-open policy evaluation.

### 5) Critical sink coverage is incomplete and misses real mutation/exfiltration tools
File: `src/hermes_katana/taint/flow.py`
Relevant lines: `CRITICAL_SINKS` 137-164

`CRITICAL_SINKS` is missing real high-impact tools from the actual toolset, including at least:
- `memory` (actual tool name; only `memory_write/memory_update/memory_delete` are listed, which are not the exposed tool names here)
- `browser_press` (can submit forms, confirm dialogs, navigate workflows)
- `browser_navigate` (state-changing on real webapps, logout flows, CSRF-style GET actions, tracking/exfil URLs)
- `process.submit` / `process.write` style interactions are not representable here because the sink model is too shallow
- `text_to_speech` is only specially denied for memory taint in one rule, but not treated as a general exfil sink in `CRITICAL_SINKS`

So even if taint survives, some obvious side-effecting channels simply are not governed by the default deny/ask rules.

Impact: policy claims “critical sinks” are blocked, but several operational sinks are outside the fence.

### 6) Empty-condition policies are universal matches; one bad glob turns into a blanket allow/deny rule
Files: `src/hermes_katana/policy/engine.py`, `src/hermes_katana/policy/models.py`
Relevant lines: `Policy.conditions` docs ~109-121 in models; `all(...)` evaluation 584-590 in engine

`Policy.conditions=[]` means “always match”, because `all([])` is `True`.

That part may be intentional, but in a security policy engine it is extremely sharp. Combined with globbed `tool_pattern`, one typo becomes a global rule. There is no guardrail for:
- blanket `allow` on `*`
- accidental overbroad patterns
- duplicated priorities that change behavior silently

This is not theoretical: the shipped defaults already use empty-condition catchalls extensively.

Impact: a malformed or overly broad policy becomes active with no warning and can silently open or close the system.

### 7) Built-in paranoid browser rule is overbroad and does not match its own description
File: `src/hermes_katana/policy/defaults.py`
Relevant lines: 149-159

The policy is described as:
- `"Block browser click/type when arguments carry taint."`

But the pattern is:
- `"tool_pattern": "browser_*"`

That matches all browser tools, including read-only and low-risk ones like `browser_snapshot`, `browser_vision`, and navigation/scroll helpers that were not described. In other words, the comment/documentation says one thing and the implementation does something much broader.

Security consequence:
- Operators cannot reason reliably from the policy descriptions.
- The engine becomes harder to audit because its documented threat model does not match reality.

Operational consequence:
- benign browser inspection tools get denied/escalated as if they were click/type sinks.

### 8) YAML inheritance silently skips unknown parents instead of failing closed
File: `src/hermes_katana/policy/yaml_loader.py`
Relevant lines: `_resolve_inheritance()` 241-249

If `extends` names an unknown parent, the loader logs a warning and returns the child data unchanged.

That is a security bug. If an operator expects inheritance from `paranoid` or a custom base and mistypes the name, the loader should fail hard. Instead it silently loads a weaker policy set.

Impact: one typo in YAML can drop intended deny rules and leave the engine with a much weaker policy than the operator believes is active.

### 9) Hot reload can partially drop policy coverage when some files fail validation
File: `src/hermes_katana/policy/yaml_loader.py`
Relevant lines: `load_policy_directory()` and `PolicyFileWatcher._poll_loop()` 398-409

`load_policy_directory()` skips invalid files and returns only valid ones. `PolicyFileWatcher._poll_loop()` applies the callback as long as the resulting list is non-empty.

So if a directory had 5 policy files and 1 becomes invalid, the watcher will still replace the active policy set(s) with only the remaining 4. It only preserves the old config when the new result count is zero.

That is a classic partial-failure fail-open bug.

Impact: a single malformed policy file can silently remove just the protections defined in that file while the system keeps running.

## Medium findings

### 10) `TaintedStr.__repr__()` returns another tainted string, not an inert debug representation
File: `src/hermes_katana/taint/value.py`
Relevant lines: 406-417

`__repr__()` returns a `TaintedStr`, carrying the same taint.

That is bizarre and dangerous:
- debug/logging/formatting paths can unexpectedly preserve or amplify taint
- code expecting `repr(x)` to be inert debug text instead gets a live tainted object
- downstream wrappers can accumulate dependencies from mere introspection

This is not how `__repr__` should behave in security-sensitive wrappers.

### 11) `unwrap()` is audited, but equivalent taint-stripping paths are not
File: `src/hermes_katana/taint/value.py`
Relevant lines: `unwrap()` 143-160, `TaintedStr.encode()` 432-439

The code logs a warning on explicit `unwrap()`, but non-explicit laundering paths are not consistently audited:
- `json.dumps()` strips wrappers with no warning
- `decode()` after `encode()` yields plain text with only the encode-side warning
- list/string rebuilding paths strip taint with no warning

This creates a false sense of observability. Operators will assume taint stripping is logged when it is not.

### 12) Registry-based tracking is wrapper-object based, not value-flow based
File: `src/hermes_katana/taint/tracker.py`
Relevant lines: registration/propgation paths around 156-321

The tracker stores wrappers keyed by `id(tv)`. That means only explicitly wrapped objects are tracked. If a tainted value is converted to a plain Python value through a laundering path, the tracker has no way to reconnect the flow.

This magnifies finding #1: once the wrapper is lost, enforcement is gone.

### 13) `taint.registrar` hardcodes use of the global singleton everywhere
File: `src/hermes_katana/taint/registrar.py`
Relevant lines: 27-29 and all helper functions

Every entry-point helper calls `_get_tracker()` which always returns `TaintTracker.get_instance()`. That makes isolated/scoped analysis easy to accidentally bypass at integration boundaries. Code running under a scoped tracker can still taint values into the global singleton if it uses these helpers.

Impact:
- cross-request contamination in long-running processes
- tests that appear isolated but are not
- confusing provenance if multiple sessions share one process

## Bottom line

The big story is simple:
- taint is still launderable through ordinary Python operations,
- middleware only checks top-level wrappers,
- sink coverage is incomplete,
- policy reload/inheritance both fail too softly.

That is enough for a determined payload to survive ingestion, shed its wrapper, and reach a side-effecting tool.

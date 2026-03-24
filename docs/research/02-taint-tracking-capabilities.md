# Taint Tracking & Capability-Based Security Research

Research compiled for the HermesKatana project.
Primary sources: CaMeL paper (arXiv 2503.18813), Norm Hardy "Confused Deputy",
Fuzzing Book information flow chapter, DeepWiki CaMeL breakdown, and multiple
academic surveys retrieved via web search.

---

## Table of Contents

1. Classical Taint Tracking
2. Information Flow Control Theory
3. Capability-Based Security
4. CaMeL Implementation Analysis
5. Weaknesses of Taint for LLMs
6. LLM Output Trust Boundaries
7. Taint Through Serialization
8. Python Implementation Strategies
9. HermesKatana Improvements (20+ items)

---

## 1. Classical Taint Tracking

### 1.1 Origins — Perl Taint Mode (1987)

Perl introduced "taint mode" (`-T` flag) as one of the earliest production
implementations of dynamic taint analysis. Every value read from user input,
environment variables, or external files is automatically marked "tainted".
A tainted value cannot be used in certain unsafe operations (shell exec,
file open with user-specified path) without first passing through a "taint
sink" — a regex match that explicitly validates and untaints the data.

Perl's model is binary (tainted / not-tainted) and propagates transitively:
any string derived by concatenation, substitution, or expression involving
a tainted value is itself tainted. Untainting is explicit and deliberate —
you must write a regex pattern that captures the sanitized portion.

Key lessons from Perl taint mode:
- Binary tainting is easy to implement but causes over-tainting.
- Explicit untainting forces developers to think about sanitization.
- The model collapses when C extensions are used (taint does not cross
  the FFI boundary), an early example of the taint-loss-at-boundary problem.

### 1.2 Dynamic Taint Analysis (DTA)

Dynamic Taint Analysis (DTA) is the general family of techniques that
propagate metadata labels ("taints") through variables, memory locations, or
objects during execution. Unlike static analysis, DTA operates at runtime and
can track flows through dynamically computed paths.

Core components of any DTA system:
- Sources: where taint is introduced (user input, network reads, file reads).
- Propagation rules: how taint moves through operations (arithmetic,
  string manipulation, data structure access).
- Sinks: where tainted data is checked before unsafe use (eval, exec, SQL
  query construction, network send).

Key academic/industrial DTA systems:

**libdft** (Liu & Keromytis, 2012): A practical DTA framework for x86 Linux
binaries using Intel's Pin dynamic binary instrumentation framework. libdft
provides byte-level taint tracking at the machine-instruction level without
source code. It defines taint sources via `__libdft_set_taint` and sinks via
`__libdft_get_taint`. The system was subsequently extended to 64-bit (libdft64)
and used in Angora fuzzer for taint-guided fuzzing.

libdft overhead is typically 2-20x slowdown due to shadow memory maintenance
and instrumentation of every memory read/write.

**TaintDroid** (Enck et al., 2010): Extended Android's Dalvik VM to track
privacy-sensitive data flows at four levels: variable, method, message, and
file. TaintDroid monitors when sensitive data (location, IMEI, contacts) flows
to network sinks, detecting privacy leaks in Android apps. It uses a coarser
taint granularity (variable-level rather than byte-level) to reduce overhead
to approximately 14%.

**HardTaint** (2024): Addresses the core DTA performance problem by using
selective hardware-assisted tracking. Rather than instrumenting every
instruction, HardTaint uses hardware performance counters and selective
sampling to reduce overhead to near-zero for uncontentious paths.

**podft** (NDSS Bar 2023): Accelerates DTA with precise path filtering —
only tracking taints along paths that actually reach a sink, using
static pre-analysis to identify which paths can propagate to sinks.

### 1.3 The Over-Tainting / Under-Tainting Problem

This is the fundamental tension in any taint system:

**Over-tainting** (false positives): Too many values are marked tainted.
Classic cause: implicit flows. If tainted data is used as a branch condition,
the value not-taken by that branch may still leak information implicitly.
Example:
```
secret = tainted_value
if secret > 100:
    result = "big"   # result indirectly depends on secret
else:
    result = "small"
```
If you do not taint `result`, you miss the implicit information flow.
If you do taint everything conditioned on tainted values, nearly all program
data ends up tainted (overtainting explosion).

**Under-tainting** (false negatives): Taint is lost at some operation.
Common causes:
- C extensions / FFI calls where the runtime has no instrumentation.
- Hash/encode operations where derived values do not retain source labels.
- Type coercions (tainted int converted to plain str).
- Sanitization functions that do not explicitly untaint output.

The Fuzzing Book (fuzzingbook.org, "Tracking Information Flow") notes
explicitly: "At some point, however, this will still break down, because as
soon as an internal C function in the Python library is reached, the taint
will not propagate into and across the C function."

### 1.4 Taint Granularity Levels

From coarsest to finest:

1. **Value-level** (Perl, most practical systems): entire variable is
   tainted or not. Fast, low memory overhead, but loses position information.

2. **Byte-level** (libdft): each byte of memory has its own taint bits.
   Enables precise sub-string taint tracking. High overhead.

3. **Character-level** (CaMeL CaMeLStr, HermesKatana TaintedStr): each
   character in a string carries its own source set. Enables surgically
   precise taint — a string partially assembled from user input and web
   content carries per-character provenance.

4. **Bit-level** (academic): theoretically most precise, practically
   infeasible for most applications.

HermesKatana uses character-level taint for strings via the `CharTaint` map
(a dict from index to `frozenset[Source]`, with a uniform default for
identical-source runs).

---

## 2. Information Flow Control Theory

### 2.1 Bell-LaPadula Model (1973) — Confidentiality

Developed by David Elliott Bell and Leonard J. LaPadula for the US DoD,
the Bell-LaPadula (BLP) model enforces *confidentiality* through two rules:

- **Simple Security Property (No Read Up)**: A subject at security level L
  cannot read an object at a higher security level. ("read down")
- **Star Property (No Write Down)**: A subject at level L cannot write to an
  object at a *lower* security level. ("write up")

The intuition: a secret document cannot leak to unclassified storage by
having a SECRET-cleared user write content they read into an UNCLASSIFIED
file. The star property prevents downward information flow.

In the context of injection defense: BLP maps cleanly to *confidentiality*
of user data — preventing private information from leaking to untrusted
external services. This is one half of the CaMeL threat model (data
exfiltration via prompt injection).

### 2.2 Biba Model (1977) — Integrity

The Biba model is the integrity dual of Bell-LaPadula. It enforces:

- **Simple Integrity Property (No Read Down)**: A subject cannot read from
  objects at a *lower* integrity level. ("read up")
- **Star Integrity Property (No Write Up)**: A subject cannot write to
  objects at a *higher* integrity level. ("write down")

The intuition: untrusted data (low integrity) should not contaminate trusted
data stores (high integrity). A web scraper's output (low integrity) should
not directly modify a trusted database (high integrity) without sanitization.

**Why Biba maps directly to injection defense**: Prompt injection is an
integrity violation. An attacker-controlled web page (low integrity) attempts
to write instructions into the trusted control flow of the agent (high
integrity). Biba's no-write-up rule says: data from low-integrity sources
cannot influence high-integrity control flow — exactly what taint tracking
for injection defense enforces.

HermesKatana's `CRITICAL_SINKS` set (terminal, bash, send_message, etc.)
represents high-integrity targets. The default rules deny flows from
`WEB_CONTENT`, `MCP`, `UNKNOWN` (low-integrity sources) to those sinks,
implementing Biba's star integrity property.

### 2.3 Non-Interference (Goguen & Meseguer, 1982)

Non-interference is a stronger, more formal security property than BLP/Biba.
It states: the behavior of a system at a low security level should not be
affected by the inputs at a high security level, and vice versa.

Formally, for a system S with two security domains H (high) and L (low):
if we change any input at level H, the observable outputs at level L should
be unchanged.

Non-interference is strictly stronger than BLP. A system can comply with BLP
(via access controls on objects) but still have covert channels that leak H
information to L observations. Non-interference would prohibit these.

The limitation for practical systems: non-interference rules out almost all
useful programs. A login system that returns "wrong password" leaks
(non-interferes with) the existence of the user account to the low domain.
Most real systems relax non-interference via *declassification*.

### 2.4 Declassification and Endorsement

**Declassification**: Converting high-security data to low-security data
in a controlled, policy-approved manner. Example: a password checker reads
a high-security password but outputs only a boolean — the boolean is
declassified because it leaks minimal information.

**Endorsement**: Raising the trust level of data. A validation function
might read untrusted user input, apply sanitization, and produce an
"endorsed" result that is treated as trusted. Endorsement is the inverse
of taint marking — it requires explicit, auditable decisions.

Both concepts are crucial for AI systems:
- The LLM acts as a declassifier (reads a raw document, produces a summary)
  but this is dangerous because LLMs can be manipulated to leak more.
- The user/human-in-the-loop is the correct endorsement point — a human
  review of LLM output before it flows to critical sinks.

### 2.5 Information Flow Control (IFC) in Type Systems

Modern programming language research embeds IFC into type systems. Examples:

- **Jif** (Java with Information Flow): extends Java's type system with
  security labels on every type. `int{Alice:}` means an integer readable only
  by Alice. The type checker statically verifies no information flows from
  high to low labels without declassification.

- **FlowCaml**: an ML language with a security lattice in the type system.
  Each value has both a type and a label; the type checker rejects programs
  with improper information flows.

- **Haskell SecLib/LIO (Labeled IO)**: uses monads to isolate labeled
  computations. `LIO` actions carry a floating label that tracks the
  sensitivity of all data read so far; it can only increase, never decrease.

The advantage of type-system IFC: violations are caught at compile time.
The disadvantage: most useful programs require declassification annotations
that are verbose and error-prone. Python's dynamic typing makes compile-time
IFC infeasible; HermesKatana uses runtime tracking instead.

---

## 3. Capability-Based Security

### 3.1 The Confused Deputy Problem (Norm Hardy, 1988)

Norm Hardy's "Confused Deputy" paper is the foundational document explaining
why capability security matters. The original story:

A Fortran compiler at Tymshare was instrumented to collect usage statistics.
The statistics file was called `(SYSX)STAT` in the compiler's home directory,
and the compiler was granted "home files license" — permission to write any
file in SYSX. Users could also tell the compiler to write debugging output
to a file of their choosing.

A user supplied `(SYSX)BILL` (the billing database) as their debugging output
file. The compiler, having home files license for SYSX, happily wrote
debugging garbage over the billing file, destroying it.

The compiler was a "confused deputy" — it served two principals
simultaneously (the user and the system) without any way to keep their
authorities separate. When exercising the user's request, it accidentally
used the system authority it was granted for a different purpose.

Hardy's key insight: "The fundamental problem is that the compiler runs with
authority stemming from two sources." The ACL-based solution (adding more
rules to file access) was tried and failed repeatedly. Every new rule
introduced new security holes in previously-secure programs.

The capability solution: endow the compiler with a direct capability to
`(SYSX)STAT` rather than a categorical "home files license". When writing
debugging output, the compiler presents only the *invoker's* capability.
There is no way to accidentally use the wrong authority because capabilities
are unforgeable and precisely scoped.

### 3.2 Object Capabilities (ocaps)

An object capability system unifies the concepts of designation (naming
an object) and authorization (having permission to use it). In an ocap
system:
- To access an object, you must hold a reference (capability) to it.
- You cannot forge capabilities — they can only be obtained from their
  creator or from someone who already holds them.
- Capabilities can be attenuated: you can give someone a restricted version
  of your capability (e.g., read-only instead of read-write).

The three ocap properties (per the E language / erights.org specification):
1. **No ambient authority**: authority comes only from held capabilities,
   not from global state like environment variables or file paths.
2. **Unforgeable references**: you cannot guess or construct a capability
   to something you don't already have access to.
3. **Attenuation**: holders can create restricted capabilities to delegate
   limited authority.

### 3.3 The E Language and Mark Miller's Thesis

The E programming language (erights.org) is the canonical demonstration of
ocap principles in a practical programming language. E's design:
- Every value is a capability.
- Interprocess communication happens via message passing (`<-` sends),
  never via shared mutable state.
- Promise pipelining allows chaining of capability invocations without
  blocking.

Mark Miller's thesis ("Robust Composition: Towards a Unified Approach to
Access Control and Concurrency Control") formalizes the E approach and
demonstrates that capability security and concurrent programming solve
related problems via the same mechanisms.

The E language directly inspired:
- JavaScript's SES (Secure ECMAScript) and Compartments proposal
- The `safe` subset of Rust
- seL4's capability model

### 3.4 CHERI — Hardware Capability Architecture

CHERI (Capability Hardware Enhanced RISC Instructions) is a CPU architecture
extension developed at Cambridge/SRI that implements hardware-enforced
capabilities. CHERI capabilities are 128-bit tagged values that encode:
- A base address and bounds (restricts pointer arithmetic to a valid range)
- Permission bits (read/write/execute/compartment flags)
- An unforgeable tag bit maintained by hardware

CHERI enforces capability security at the machine level. A CHERI pointer
is a capability: it carries its own authority. You cannot fabricate a
pointer to memory you do not own; bounds checking is hardware-enforced.

CHERI-seL4 combines CHERI hardware capabilities with seL4's formally
verified microkernel to provide:
- Between-address-space isolation (seL4 guarantee)
- Within-address-space memory safety (CHERI guarantee)
- A unified capability-based security model from hardware to kernel to user

### 3.5 seL4 — Formally Verified Microkernel

seL4 is a microkernel with a machine-checked formal correctness proof
(General Dynamics / NICTA, 2009). Its security model is entirely
capability-based:
- Every kernel object is accessed via a capability slot.
- Capabilities can be derived (minted), delegated, and revoked.
- Revocation is O(n) in the number of derived capabilities.

seL4 proves: if you hold no capability to an object, you cannot access it,
regardless of what other processes do. This is a hard formal guarantee,
not a policy.

### 3.6 Attenuation Patterns

Capability attenuation means creating a new capability that is strictly
less powerful than the original. Key patterns:

**Read-only facet**: A capability to an object with write methods removed.
```
# Conceptual
full_cap = get_database_cap()
readonly_cap = full_cap.read_only()  # can only SELECT, not INSERT/UPDATE
delegate_to_untrusted_code(readonly_cap)
```

**Temporal attenuation**: A capability that expires after a time limit.
**Quota attenuation**: A capability limited to N uses or N bytes.
**Revocable forwarder**: A proxy that can be revoked without revoking
the original capability. Useful for third-party service integration.

In AI agent systems, tool capabilities are natural attenuation targets:
a web search tool should not carry file system write authority, and an
email send tool should not carry memory read authority.

---

## 4. CaMeL Implementation Analysis

### 4.1 Overview

CaMeL (CApabilities for MachinE Learning) is a system from Google DeepMind
(Debenedetti et al., arXiv 2503.18813, submitted March 2025, revised June
2025) that provides a principled defense against prompt injection attacks
in LLM agents. It achieves 77% task completion with provable security on
the AgentDojo benchmark (vs 84% for an undefended system).

The key insight: instead of trying to make LLMs resistant to prompt injection
through training or detection, CaMeL creates a *system layer* around the LLM
that enforces security properties regardless of what the LLM does internally.

### 4.2 Dual-LLM Architecture

CaMeL improves on Simon Willison's 2023 "Dual LLM pattern":

- **P-LLM (Privileged LLM)**: sees only the user's query. Its job is to
  convert the user's request into a structured program in a restricted Python
  dialect. It plans the execution but never sees tool outputs.

- **Q-LLM (Quarantined LLM)**: sees potentially untrusted data. Its job is
  to extract specific data fields from tool outputs (e.g., "find Bob's email
  address in this email"). It has no tool-calling capabilities.

The flaw in the original Dual LLM pattern that CaMeL fixes: the Q-LLM could
still be injected to return a malicious email address as its extracted output.
The Q-LLM's output would then be passed to a tool (send_email) by the P-LLM.

CaMeL's fix: data returned by the Q-LLM (which processed untrusted content)
is marked as tainted. Even if the Q-LLM is injected to return "attacker@evil.com"
as the "extracted email", the taint system prevents this from flowing to
`send_email` if the policy requires a trusted recipient address.

### 4.3 The CaMeL Capabilities Model

CaMeL's capability representation is a frozen dataclass with three fields:

```python
@dataclass(frozen=True)
class Capabilities:
    sources_set: frozenset[SourceEnum]   # where did this data come from?
    readers_set: Readers                  # who is allowed to read it?
    other_metadata: dict[str, Any]        # contextual information
```

**Sources axis (SourceEnum)**:
- `User`: data provided directly by the end user (trusted)
- `CaMeL`: data generated by the CaMeL system itself (trusted)
- `Assistant`: data generated by LLM assistants (conditional)
- `TrustedToolSource`: data from verified/whitelisted external tools
- `Tool(tool_name, inner_sources)`: data from a specific tool with provenance

**Readers axis**:
- `Public`: no access restriction, any component may use this data
- `frozenset[_T]`: explicit set of authorized readers

The dual-axis design is deliberate: `sources` tracks provenance (data flow),
while `readers` controls visibility (access control). These are orthogonal
concerns:
- High-confidentiality user data might be Public-readable (it's not secret)
  but Assistant-sourced (it came from the LLM's analysis).
- Tool output might have restricted readers (only the sending component)
  while being Tool-sourced (provenance).

### 4.4 CaMeLStr — Character-Level Taint

CaMeL's `CaMeLStr` is a string subclass that carries a `Capabilities` object
alongside the string value. Every string operation that produces a new string
also propagates the capabilities.

Crucially, the capability propagation uses **union of sources** and
**intersection of readers** (most-restrictive wins on the access control axis).
If you concatenate a User-sourced string with a WebContent-sourced string,
the result carries both source labels. If a reader that could read both
individually cannot read the combined result unless it is in both reader sets.

### 4.5 Dependency Graph and Control/Data Flow Separation

The core of CaMeL's security guarantee: by parsing the P-LLM's generated
code into an AST and interpreting it with a custom Python interpreter, CaMeL
can:

1. **Trace the dependency graph**: which variables depend on which other
   variables. If `address` is derived from `email_content` which came from
   a tool call, `address` carries the taint of `email_content`.

2. **Separate control flow from data flow**: the P-LLM determines the
   *structure* of the program (which tools to call, in what order, with
   what parameters). Tool outputs influence only *data values*, never
   the program structure itself.

3. **Enforce at call time**: before each tool invocation, the interpreter
   checks each argument's taint labels against the tool's security policy.
   A tainted `recipient` being passed to `send_email` is blocked unless
   the policy allows it.

### 4.6 What CaMeL Deliberately Disables

The custom Python interpreter is a *restricted* subset of Python. CaMeL
deliberately disables:
- Dynamic `eval()` / `exec()` — these would allow injected code to escape
  the interpreter's control.
- Arbitrary imports — only whitelisted standard library operations are
  available.
- File system access except through taint-aware tool wrappers.
- Any operation that could modify the interpreter's own policy rules.
- Loops with unbounded iteration (prevents resource exhaustion).

This "secure by restriction" approach is the interpreter equivalent of
the principle of least privilege.

### 4.7 Policy Enforcement at Tool Invocation

Each tool in CaMeL has an associated security policy: a function from
`(argument_name, capabilities)` to `Allow | Deny | AskUser`. The default
policy for `send_email`:

```
if recipient.sources contains {User} or {CaMeL}:
    ALLOW
elif recipient.readers is Public and recipient.sources contains {TrustedToolSource}:
    ASK_USER  # user must confirm the address
else:
    DENY  # data from untrusted sources cannot be an email recipient
```

CaMeL's benchmark results: 77% of tasks completed with *provable* security
(no injection attack succeeded), vs 84% for the undefended baseline. The 7%
gap represents tasks requiring capabilities CaMeL's policy cannot approve
automatically.

### 4.8 Limitations CaMeL Acknowledges

From section 8.3 of the paper:
1. Security policies must be codified and maintained by users — policy
   authoring burden is high.
2. Excessive user approval requests cause "approval fatigue" (users start
   saying yes to everything).
3. CaMeL requires significant changes to the agent framework — it is not
   a drop-in retrofit for existing agents.
4. The P-LLM must reliably generate well-structured Python code — if the
   P-LLM itself is compromised, the guarantees weaken.

---

## 5. Weaknesses of Taint for LLMs

### 5.1 The LLM-as-Taint-Boundary Problem

This is the core fundamental challenge when applying taint tracking to LLM
agent systems, and it is what CaMeL specifically addresses.

In a classical system, a tainted byte of memory either propagates or doesn't.
The system is deterministic. In an LLM system, the "transformation" of data
is a language model that may:
1. Extract data from untrusted context faithfully (good).
2. Extract data but be subtly manipulated by injected instructions (bad).
3. Generate entirely new content based on untrusted context (ambiguous).

If we pass a tainted email body to an LLM to extract the sender's name,
the LLM's *output* is derived from tainted input. But the LLM might have
been manipulated by the email body to output a different name. The taint
label on the output correctly records it as derived from untrusted input —
but the *semantics* of what was extracted cannot be verified.

This is why CaMeL's Q-LLM model is necessary: the Q-LLM's output is always
marked tainted regardless of whether injection occurred. The taint system
cannot distinguish "legitimate extraction" from "injected payload" — but it
doesn't need to. The *policy* enforces that tainted data cannot reach
sensitive sinks without human review.

### 5.2 Semantic vs. Syntactic Taint

Classical DTA is *syntactic*: it tracks which bytes/characters are derived
from which sources at the level of data operations. LLM systems require
*semantic* taint because:

- An LLM can infer information from tainted context even if it never directly
  copies tainted bytes into the output (implicit information flow at the
  semantic level).
- A malicious prompt can instruct the LLM to mix trusted and untrusted content
  in ways that appear trusted (laundering).

Semantic taint would require a model that understands what each LLM output
"means" — which requires another LLM, creating a regress. CaMeL's solution:
accept this limitation and mark all Q-LLM output as tainted by default,
regardless of the actual content.

### 5.3 Overtainting in LLM Contexts

As Simon Willison notes: "user fatigue" is real. If every tool output is
tainted and every tainted value requires user approval before reaching any
sink, users will develop approval fatigue and start approving everything.

The tension: making taint too strict (every Q-LLM output tainted) causes
user fatigue. Making taint too loose (only definitely-untrusted content
tainted) allows injection attacks through the gaps.

CaMeL's approach: policy-based escalation. Some sinks (memory write, email
send) require human approval for tainted data. Others (read operations,
display to user) allow tainted data freely. Policies can be pre-approved
for common patterns (e.g., "always allow sending email to contacts already
in my address book").

### 5.4 Performance Overhead

Character-level taint tracking carries meaningful overhead:
- Every string concatenation must merge two `CharTaint` maps.
- Every string slice must create a new `CharTaint` covering the slice range.
- The overhead is proportional to string length and number of string
  operations performed.

For typical LLM outputs (hundreds to thousands of tokens = characters),
the overhead is manageable. For very large documents (thousands of characters)
with frequent manipulation, the CharTaint maps can become significant
in memory.

Optimization strategies:
- Run-length encoding for uniform taints (all characters from same source).
- Lazy evaluation: only compute CharTaint when a policy check is imminent.
- Uniform-taint fast paths: if all characters have identical sources, skip
  per-character tracking entirely.

---

## 6. LLM Output Trust Boundaries

### 6.1 When to Trust LLM Output

LLM output trustworthiness depends on what was in the LLM's context:

**Level 0 — Full Trust**: LLM saw only the user's query (clean system prompt,
no tool outputs, no external data). Output represents the model's trained
knowledge. Label: SYSTEM or AGENT with high trust.

**Level 1 — Conditional Trust**: LLM saw tool outputs from trusted internal
tools (database queries with known schemas, user's own files). Label: AGENT
with conditional trust. Policy: allow to non-critical sinks, ask for critical.

**Level 2 — Low Trust**: LLM saw external content (web search results, emails,
documents from untrusted parties). Label: AGENT inheriting WEB_CONTENT or
MCP labels. Policy: deny to critical sinks, allow to display/read sinks.

**Level 3 — No Trust**: LLM explicitly told to process adversarial input
(e.g., "analyze this suspicious email"). Label: AGENT with UNTRUSTED.
Policy: treat output as untrusted regardless of content.

HermesKatana's `TaintLabel.AGENT` with `TrustLevel.CONDITIONAL` maps to
Level 1 by default. For Level 2/3 scenarios, the AGENT taint should inherit
the highest-sensitivity source label from its context.

### 6.2 Endorsement Patterns

**Pattern 1 — Human-in-the-Loop Endorsement**: Before an AGENT-tainted value
reaches a critical sink, present it to the human user for approval. If
approved, create an endorsed copy with USER label. This is CaMeL's "ASK_USER"
decision.

```python
# Conceptual
def human_endorse(tainted_val: TaintedValue[str], user_approved: bool) -> TaintedValue[str]:
    if user_approved:
        endorsed_source = Source.user(origin="human_endorsement")
        return TaintedStr(
            value=tainted_val.value,
            sources=frozenset({endorsed_source}),
            dependencies=(tainted_val,),  # preserve provenance chain
        )
    raise TaintPolicyViolation("User rejected endorsement")
```

**Pattern 2 — Validated Endorsement**: A validation function checks the
value against an allowlist or schema. If it matches, produce an endorsed
copy. If not, reject.

```python
# Conceptual
def validate_email_address(tainted: TaintedValue[str]) -> TaintedValue[str]:
    import re
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', tainted.value):
        return tainted.derive(tainted.value)  # structure validated
    raise ValueError("Invalid email address format")
```

**Pattern 3 — Allowlist Endorsement**: Check whether the value appears in
a user-maintained allowlist (known email addresses, approved domains).
Values on the allowlist are trusted.

**Pattern 4 — Cryptographic Endorsement**: Use a signature or HMAC to verify
that output was produced by a trusted system. Only applicable when the
producing system is a known, authenticated service.

### 6.3 Conditional Trust and Policy Gradients

Binary trust (trusted/untrusted) is too coarse for most agent scenarios.
A graduated policy model:

```
TrustLevel.TRUSTED    → allow to all sinks
TrustLevel.CONDITIONAL → allow to read sinks, ask for write sinks, deny for exec sinks
TrustLevel.UNTRUSTED  → deny to write/exec sinks, allow to display sinks
```

The "display to user" sink is special: it should accept tainted content
(users need to see what the web returned) but the system should visually
indicate trust level to the user (e.g., showing a warning banner for
untrusted content).

---

## 7. Taint Through Serialization

### 7.1 The Boundary Loss Problem

Every serialization/deserialization boundary is a potential taint sink:

```python
# Taint is present
tainted_str = TaintedStr("hello", sources=frozenset({Source.web("http://evil.com")}))

# JSON serialization — taint is LOST
import json
raw = json.dumps(tainted_str.value)  # raw is a plain str with no taint

# HTTP response parsing — taint is LOST
import httpx
response = httpx.get("https://api.example.com")
data = response.json()  # data is plain dict/list/str — no taint
```

This is the fundamental problem: Python's standard library knows nothing
about taint. Every `json.loads()`, `requests.get().json()`, `yaml.safe_load()`,
and `open().read()` discards taint information.

### 7.2 Boundary Injection Points

The solution is to inject taint at every boundary where external data enters:

**HTTP responses**: Wrap the response content immediately after receipt.
```python
def tainted_http_get(url: str) -> TaintedDict:
    response = httpx.get(url)
    raw = response.json()
    return tracker.register(raw, Source.web(url))
```

**File reads**: Wrap at the read call.
```python
def tainted_file_read(path: str) -> TaintedStr:
    content = pathlib.Path(path).read_text()
    return tracker.register(content, Source.file(path))
```

**MCP tool calls**: Wrap the tool's return value.
```python
def tainted_mcp_call(server: str, tool: str, args: dict) -> TaintedValue:
    result = mcp_client.call(server, tool, args)
    return tracker.register(result, Source.mcp(server))
```

### 7.3 Pydantic Integration Challenge

Pydantic models are the standard way to parse structured data in modern
Python. The problem: Pydantic creates plain Python objects from raw data,
discarding any taint wrappers in the process.

```python
from pydantic import BaseModel

class EmailData(BaseModel):
    sender: str
    subject: str
    body: str

# Taint is lost through Pydantic parsing
tainted_json_str = TaintedStr('{"sender": "...", ...}', sources=...)
parsed = EmailData.model_validate_json(tainted_json_str.value)
# parsed.sender is a plain str — taint is gone
```

Approaches to taint-aware Pydantic:

**Option A — Post-parse re-tainting**: After Pydantic parsing, re-wrap
all string fields with the container taint label.
```python
def parse_with_taint(raw: TaintedStr, model_class: type[BaseModel]) -> dict[str, TaintedValue]:
    parsed = model_class.model_validate_json(raw.value)
    container_sources = raw.sources
    result = {}
    for field_name, field_val in parsed.model_dump().items():
        if isinstance(field_val, str):
            result[field_name] = TaintedStr(field_val, sources=container_sources)
        else:
            result[field_name] = TaintedValue(field_val, sources=container_sources)
    return result
```

**Option B — Custom Pydantic type**: Define a `TaintedStr` type that is
recognized by Pydantic and carries taint through validation.
This requires custom `__get_validators__` / `__get_pydantic_core_schema__`
implementations.

**Option C — Annotated types**: Use `Annotated[TaintedStr, ...]` with a
custom validator that preserves source metadata.

### 7.4 JSON Taint Propagation

When JSON is parsed, the taint should propagate to all extracted values:

```
{
  "name": "Bob",          # should inherit container taint
  "email": "bob@...",     # should inherit container taint
  "nested": {             # should inherit container taint
    "value": "secret"     # should inherit container taint (deepest)
  }
}
```

The `TaintedDict` class in HermesKatana handles this at the container level
(the dict itself is tainted) but individual string values extracted from
the dict lose their taint once accessed as raw strings:
```python
tainted_dict = tracker.register({"email": "bob@..."}, Source.web("..."))
email = tainted_dict["email"]  # plain str — taint lost!
```

A proper fix requires `TaintedDict.__getitem__` to return `TaintedValue`
wrapped items when the underlying values are not already tainted.

### 7.5 The String-to-Command Injection Path

The most dangerous taint path in an agent system:

```
Web content (WEB_CONTENT taint)
  → LLM processes it (AGENT taint inherits WEB_CONTENT)
  → LLM proposes a shell command (string contains untrusted substring)
  → Agent calls terminal(command=proposed_cmd)
  → DENIED: WEB_CONTENT → terminal is a critical sink
```

This path is exactly what HermesKatana's default rules protect against.
The key requirement: the command string passed to `terminal` must carry the
taint from any web content that influenced its construction, even if the
command looks "clean" syntactically.

---

## 8. Python Implementation Strategies

### 8.1 Wrapper Class Pattern (HermesKatana's Approach)

The wrapper class approach subclasses or wraps Python types to carry taint
metadata alongside values. `TaintedStr`, `TaintedList`, `TaintedDict`
override relevant methods to propagate taint.

Advantages:
- No interpreter modification required.
- Works with existing Python code with minimal changes at ingestion points.
- Taint metadata travels with the value through function calls.

Disadvantages:
- Taint is silently lost when code calls `str(tainted_val)` or accesses
  `.value` — the wrapper is stripped.
- Cannot track taint through standard library functions written in C.
- Requires every string operation to be overloaded.

### 8.2 ContextVar Implicit Propagation

Python's `contextvars.ContextVar` provides per-task local storage that
survives async context switches. This can be used for implicit taint
propagation:

```python
from contextvars import ContextVar

_current_taint: ContextVar[frozenset[Source]] = ContextVar(
    "_current_taint", default=frozenset()
)

def with_taint_context(sources: frozenset[Source]):
    """Context manager that sets ambient taint for a code block."""
    token = _current_taint.set(sources)
    try:
        yield
    finally:
        _current_taint.reset(token)

def get_ambient_taint() -> frozenset[Source]:
    """Return the taint labels for the current execution context."""
    return _current_taint.get()
```

This approach propagates taint implicitly through async tasks — any `async`
function called within a tainted context inherits the ambient taint. It's
complementary to explicit wrapper propagation.

### 8.3 AST Instrumentation

For statically analyzable code, AST instrumentation can automatically insert
taint propagation calls:

```python
# Original code
result = fetch_web(url) + " processed"

# After AST instrumentation
_t1 = fetch_web(url)
_taint_1 = get_taint(_t1)
_t2 = " processed"
result = _t1 + _t2
set_taint(result, merge_taint(_taint_1, get_taint(_t2)))
```

This approach is used by some Python taint tools (bandit, CodeQL's Python
support). It's more comprehensive than wrapper classes because it can
track taint through C extensions by wrapping their call sites.

### 8.4 MyPy Integration

HermesKatana's generic types (`TaintedValue[T]`) are fully compatible with
mypy. The type system can express:

```python
def process_user_input(data: TaintedStr) -> TaintedStr: ...
def send_to_terminal(cmd: str) -> None: ...  # expects plain str

tainted = tracker.register("data", Source.web("..."))
send_to_terminal(tainted)  # mypy should warn: TaintedStr is not str
send_to_terminal(tainted.value)  # mypy allows: .value is str (but taint lost!)
```

A stricter mypy plugin could:
1. Mark `TaintedValue.value` and `.unwrap()` as requiring a `# type: ignore`
   comment to force explicit acknowledgment of taint loss.
2. Provide a `Trusted[T]` type that is only assignable from `TRUSTED`-sourced
   values, enforced at type-check time.

### 8.5 Async Taint Considerations

When using `asyncio`, taint must propagate through `await` boundaries:

```python
async def handler(request: aiohttp.Request) -> None:
    # request.text() returns plain str — taint must be re-injected
    body = await request.text()
    tainted_body = tracker.register(body, Source.web(str(request.url)))
    # tainted_body correctly carries WEB_CONTENT label
    ...
```

The `ContextVar` approach handles async correctly — each `asyncio.Task` has
its own context, and `ContextVar.set()` in a parent task does NOT propagate
to child tasks unless explicitly inherited. This is the correct behavior for
taint: a subtask processing untrusted data should inherit the ambient taint,
but should not contaminate its parent's context.

---

## 9. HermesKatana Improvements

Based on the research above and analysis of the existing codebase, the
following improvements are identified. Each item includes rationale from
the research.

### 9.1 New TaintLabels

**Item 1: DATABASE label**
Current code has no distinction between "file content" and "database
query results". Database results have different integrity properties —
they are structured, validated by the DB schema, but may still contain
attacker-controlled data if the database was corrupted.
```python
DATABASE = auto()
"""Data returned from a database query (structured but potentially attacker-influenced)."""
```

**Item 2: ENVIRONMENT label**
Environment variables are a classic attack vector (LD_PRELOAD, PATH
manipulation). Taint from `os.environ` should be distinct from file content.
```python
ENVIRONMENT = auto()
"""Data read from environment variables (os.environ)."""
```

**Item 3: CODE_OUTPUT label**
When the agent executes code and captures stdout/stderr, the result is
conceptually different from WEB_CONTENT or TOOL_OUTPUT. Code output may
contain injected content from the code's data sources.
```python
CODE_OUTPUT = auto()
"""Captured stdout/stderr from executed code or subprocesses."""
```

**Item 4: EXTERNAL_API label**
A more specific label for authenticated external API calls (vs raw WEB_CONTENT
which implies unauthenticated crawl). Allows policy differentiation: a trusted
API partner might warrant CONDITIONAL trust rather than UNTRUSTED.
```python
EXTERNAL_API = auto()
"""Data from an authenticated external API call."""
```

**Item 5: PEER_AGENT label**
Multi-agent systems need to track data from other agents separately from
the primary agent's own outputs. A peer agent is an agent with its own
trust level that may or may not be verified.
```python
PEER_AGENT = auto()
"""Data received from another AI agent in a multi-agent system."""
```

### 9.2 Missing TaintedStr Operations

**Item 6: TaintedStr.__mod__ (% formatting)**
Python's `%` string formatting is widely used:
```python
result = "Hello %s" % tainted_name  # taint not propagated!
```
Need to override `__mod__` and `__rmod__`.

**Item 7: TaintedStr.encode() / TaintedBytes**
When a tainted string is encoded to bytes (for HTTP, crypto, etc.), the taint
should carry through. Currently `encode()` returns plain `bytes`.
```python
def encode(self, encoding: str = "utf-8", errors: str = "strict") -> TaintedBytes:
    return TaintedBytes(
        value=self.value.encode(encoding, errors),
        sources=self.sources,
        readers=self.readers,
        dependencies=(self,),
    )
```

**Item 8: TaintedStr.strip() — bug in current implementation**
The current `strip()` uses `self.value.index(raw)` which raises ValueError
if `raw` is empty. Needs a guard:
```python
if not raw:
    start, stop = 0, 0
else:
    start = self.value.index(raw)
    stop = start + len(raw)
```

**Item 9: TaintedStr.lstrip() / rstrip()**
Only `strip()` is implemented. `lstrip()` and `rstrip()` are missing and
will return plain `str`.

**Item 10: TaintedStr.find() / index() / rfind()**
These return integer positions, not strings, so taint does not propagate —
that is correct. But if the returned index is then used in a slice, the
slice correctly propagates taint. Document this behavior explicitly.

**Item 11: TaintedStr.partition() / rpartition()**
Returns a 3-tuple of strings; currently returns plain `str` tuples.

**Item 12: f-string support**
Python f-strings (`f"Hello {name}"`) call `__format__` on each embedded
expression. The current `format()` method handles `str.format()` but not
the `__format__` protocol used by f-strings. Need:
```python
def __format__(self, format_spec: str) -> str:
    # Note: this returns plain str; the f-string itself is not a TaintedStr.
    # Consider wrapping the caller to detect f-string usage.
    return format(self.value, format_spec)
```

### 9.3 Endorsement API

**Item 13: Endorsement function in tracker**
There is no explicit endorsement pathway in the current tracker. Add:
```python
def endorse(
    self,
    value: TaintedValue[T],
    endorser: str = "human",
    reason: str = "",
) -> TaintedValue[T]:
    """Upgrade a value's trust level, recording the endorsement.
    
    Creates a new value with USER-level trust, preserving the full
    provenance chain so the endorsement decision is auditable.
    """
    endorsed_source = Source(
        label=TaintLabel.USER,
        origin=f"endorsed_by:{endorser}",
        trust_level=TrustLevel.TRUSTED,
        metadata={"reason": reason, "original_labels": str(value.labels)},
    )
    return value.merge_metadata(TaintedValue(value.value, frozenset({endorsed_source})))
```

**Item 14: Endorsement registry**
Track all endorsements for audit purposes:
```python
@dataclass
class EndorsementRecord:
    original_value_id: int
    endorsed_value_id: int
    endorser: str
    reason: str
    timestamp: float
    original_labels: frozenset[TaintLabel]
```

### 9.4 FlowRule Additions

**Item 15: FlowRule for AGENT → terminal with mixed sources**
The current QUARANTINE rule for AGENT → critical sinks is too permissive.
An AGENT value that inherits WEB_CONTENT taint should be DENIED, not
QUARANTINED. Need a composite rule:
```python
FlowRule(
    source_labels=frozenset({TaintLabel.AGENT}),
    target_tools=CRITICAL_SINKS,
    decision=FlowDecision.QUARANTINE,
    reason="Agent output to critical sinks is logged for review.",
    priority=25,
    condition=lambda labels: TaintLabel.WEB_CONTENT not in labels and TaintLabel.MCP not in labels,
)
```
This requires adding a `condition` field to `FlowRule`.

**Item 16: Tool-specific policies**
The current FlowRule model uses a flat `frozenset[str]` for target tools.
A richer model should allow tool-argument-level policies:
```python
FlowRule(
    source_labels=frozenset({TaintLabel.WEB_CONTENT}),
    target_tools=frozenset({"send_email"}),
    target_arguments=frozenset({"recipient"}),  # NEW: per-argument
    decision=FlowDecision.DENY,
    reason="Web content cannot set email recipient.",
    priority=110,
)
```

**Item 17: Allow-listing by pattern**
Add support for trust-based allow-listing in FlowRule:
```python
FlowRule(
    source_labels=frozenset({TaintLabel.WEB_CONTENT}),
    target_tools=frozenset({"send_email"}),
    decision=FlowDecision.ALLOW,
    condition=lambda val, args: args.get("recipient", "").endswith("@company.com"),
    reason="Sending to internal company addresses is always allowed.",
    priority=90,
)
```

### 9.5 Async Taint

**Item 18: ContextVar-based ambient taint**
Add ambient taint propagation via `ContextVar` for async code:
```python
# In tracker.py or a new async_context.py
from contextvars import ContextVar

_ambient_taint: ContextVar[frozenset[Source]] = ContextVar(
    "_ambient_taint", default=frozenset()
)

@contextmanager
def taint_context(sources: frozenset[Source]) -> Iterator[None]:
    """Set ambient taint for all code in this context."""
    token = _ambient_taint.set(sources)
    try:
        yield
    finally:
        _ambient_taint.reset(token)
```

**Item 19: AsyncTaintTracker**
The current singleton `TaintTracker` uses a threading lock. For async code,
this may cause unnecessary blocking. An async-native version should use
`asyncio.Lock` and track per-task taint context via `ContextVar`.

### 9.6 Pydantic Integration

**Item 20: TaintedModel base class**
Provide a Pydantic base model that preserves taint:
```python
from pydantic import BaseModel
from typing import Annotated, Any

class TaintedModel(BaseModel):
    """Base model that preserves taint on all string fields during parsing.
    
    Usage:
        class EmailData(TaintedModel):
            sender: str
            subject: str
            body: str
        
        data = EmailData.model_validate_with_taint(raw_json_str, source=Source.web(url))
        # data.sender is now a TaintedStr
    """
    
    @classmethod
    def model_validate_with_taint(
        cls,
        obj: Any,
        source: Source,
        **kwargs: Any,
    ) -> "TaintedModel":
        """Parse the model and re-taint all string fields with source."""
        instance = cls.model_validate(obj, **kwargs)
        # Post-processing: wrap string fields with taint
        for field_name in cls.model_fields:
            val = getattr(instance, field_name)
            if isinstance(val, str):
                setattr(instance, field_name, TaintedStr(val, frozenset({source})))
        return instance
```

### 9.7 Additional Items

**Item 21: TaintedBytes type**
Web content is often received as bytes before decoding. A `TaintedBytes`
class parallel to `TaintedStr` with `decode() -> TaintedStr` would prevent
taint loss during the bytes-to-str conversion step.

**Item 22: Provenance visualization**
Add a method to produce a Mermaid or ASCII graph of the dependency chain:
```python
def render_provenance(value: TaintedValue, tracker: TaintTracker) -> str:
    """Return a Mermaid diagram of the value's provenance chain."""
    ...
```

**Item 23: Policy audit log**
The `FlowAnalyzer._history` stores `FlowAnalysis` records but they are not
persisted. Add structured logging (JSON) to enable security audit trails.

**Item 24: Reader-based output filtering**
The `Reader` class defines who may read a value, but there is no enforcement
in the current code — `TaintedValue.readers` is set but never checked at
output time. Add enforcement: before displaying or logging a value, check
whether the current principal (user, logger, tool) is in the readers set.

**Item 25: Taint-aware json.loads wrapper**
```python
def tainted_json_loads(raw: str | TaintedStr, fallback_source: Source) -> TaintedValue:
    """Parse JSON and apply taint from the raw string or fallback source."""
    source = raw.sources if isinstance(raw, TaintedStr) else frozenset({fallback_source})
    raw_str = raw.value if isinstance(raw, TaintedStr) else raw
    data = json.loads(raw_str)
    return TaintedValue(data, sources=source)
```

---

## Appendix A: Key References

1. Debenedetti et al., "Defeating Prompt Injections by Design" (CaMeL paper),
   arXiv 2503.18813, Google DeepMind / ETH Zurich, March 2025.
   https://arxiv.org/abs/2503.18813

2. Norm Hardy, "The Confused Deputy (or why capabilities might have been
   invented)", 1988. Canonical text on confused deputy and capability security.
   https://crypto.stanford.edu/cs155old/cs155-spring09/papers/ConfusedDeputy.html

3. Simon Willison, "CaMeL offers a promising new direction for mitigating
   prompt injection attacks", April 2025.
   https://simonwillison.net/2025/Apr/11/camel/

4. DeepWiki, "google-research/camel-prompt-injection", automated wiki of
   CaMeL implementation details. https://deepwiki.com/google-research/camel-prompt-injection

5. Fuzzingbook, "Tracking Information Flow", chapter on taint tracking in
   Python with TaintedStr and character-level tracking.
   https://www.fuzzingbook.org/html/InformationFlow.html

6. Bell and LaPadula, "Secure Computer System: Unified Exposition and
   Multics Interpretation", 1976. (BLP model for confidentiality.)

7. Biba, "Integrity Considerations for Secure Computer Systems", 1977.
   (Biba model for integrity — the direct theoretical basis for injection defense.)

8. Goguen and Meseguer, "Security Policies and Security Models", IEEE S&P 1982.
   (Non-interference definition.)

9. Liu and Keromytis, "libdft: Practical Dynamic Data Flow Tracking for
   Commodity Systems", VEE 2012. (libdft implementation.)

10. Enck et al., "TaintDroid: An Information-Flow Tracking System for
    Realtime Privacy Monitoring on Smartphones", OSDI 2010.

11. Watson et al., "CHERI: A Hybrid Capability-System Architecture for
    Scalable Software Compartmentalization", IEEE S&P 2015.

12. Klein et al., "seL4: Formal Verification of an OS Kernel", SOSP 2009.

13. Mark Miller, "Robust Composition: Towards a Unified Approach to Access
    Control and Concurrency Control", PhD thesis, Johns Hopkins 2006.
    http://erights.org/talks/thesis/

14. Russo and Sabelfeld, "A Taint Mode for Python via a Library", OWASP
    AppSec Research 2010. (Python-specific taint tracking design.)

15. Semgrep, "Security Like It's 1977: Capabilities for the Modern Agentic
    Web", 2026. (confused deputy problem applied to AI agents)
    https://semgrep.dev/blog/2026/security-like-its-1977-capabilities-for-the-modern-agentic-web/

---

## Appendix B: HermesKatana Current State Summary

**labels.py**:
- `TrustLevel`: TRUSTED / UNTRUSTED / CONDITIONAL
- `TaintLabel`: USER, SYSTEM, TOOL_OUTPUT, WEB_CONTENT, FILE_CONTENT, MEMORY, MCP, AGENT, UNKNOWN
- `Source`: frozen dataclass with label, origin, timestamp, trust_level, metadata
  - Factory methods: `.user()`, `.system()`, `.tool()`, `.web()`, `.file()`, `.mcp()`, `.memory()`, `.agent()`, `.unknown()`
- `Reader`: frozen dataclass with identity, can_read (frozenset of TaintLabels)
  - Factory methods: `.unrestricted()`, `.trusted_only()`

**value.py**:
- `TaintedValue[T]`: generic wrapper with sources, readers, dependencies, created_at
  - Methods: labels (property), is_trusted(), is_untrusted(), is_public(), has_label(), derive(), merge_metadata(), unwrap()
- `CharTaint`: character-index-to-sources map with `uniform()`, `get()`, `set()`, `slice()`, `concat()`, `all_sources()`
- `TaintedStr(TaintedValue[str])`: with char_taint, implements `__add__`, `__radd__`, `__getitem__`, `__iter__`, `upper()`, `lower()`, `strip()`, `split()`, `replace()`, `join()`, `format()`, `startswith()`, `endswith()`
- `TaintedList(TaintedValue[list])`: MutableSequence protocol, per-item taint
- `TaintedDict(TaintedValue[dict])`: MutableMapping protocol, per-key taint
- `unwrap()`: recursive taint stripping
- `collect_sources()`: recursive source collection

**flow.py**:
- `FlowDecision`: ALLOW, DENY, ASK_USER, QUARANTINE
- `FlowRule`: frozen dataclass with source_labels, target_tools, decision, reason, priority
  - `matches_labels()`, `matches_tool()` (supports trailing `*` glob)
- `FlowAnalysis`: decision, matched_rules, labels_present, tool_name, reasoning, timestamp
- Default rules: untrusted→critical=DENY(100), conditional→critical=ASK_USER(50), trusted→*=ALLOW(10), agent→critical=QUARANTINE(25)
- `FlowAnalyzer`: add_rule(), remove_rule(), analyze(), check(), clear_history()

**tracker.py**:
- `TrackerStats`: counters for registrations, propagations, flow decisions
- `TaintTracker`: singleton with `get_instance()`, `reset_instance()`, `scoped()` context manager
  - `register()`, `register_multi()`: wrap raw values with taint
  - `propagate()`: derive tainted value from multiple inputs
  - `get_taint_chain()`: reconstruct full provenance (depth-first walk of dependency graph)
  - `get_labels()`: all labels in the full chain
  - `check_flow()`, `analyze_flow()`: wrapper around FlowAnalyzer
  - `check_args_flow()`: most-restrictive decision across all kwargs
  - Threading: uses `threading.Lock` for all mutations

**Gaps identified**:
- No DATABASE, ENVIRONMENT, CODE_OUTPUT, EXTERNAL_API, PEER_AGENT labels
- TaintedStr missing: `__mod__`, `encode()`, `lstrip()`, `rstrip()`, `partition()`, `__format__`
- Bug in `strip()` when result is empty
- No endorsement API
- FlowRule lacks per-argument targeting and conditional logic
- No ContextVar-based ambient taint for async
- No Pydantic integration
- No TaintedBytes type
- Reader access control defined but not enforced at output time
- No taint-aware json.loads / HTTP client wrappers

#!/usr/bin/env python3
"""Middleware chain example — the full security pipeline for tool calls.

Demonstrates:
  - Building a middleware chain with scanner + policy middleware
  - Processing a benign tool call (ALLOW)
  - Processing a malicious tool call (DENY)
  - Inspecting the audit trail

Run:  python3 examples/middleware_chain.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_katana.middleware import MiddlewareChain, CallContext, DispatchDecision, KatanaMiddleware


# --- Custom middleware: scan for injection in tool args ---
class ScannerMiddleware(KatanaMiddleware):
    """Scans tool arguments for prompt injection and dangerous commands."""

    def __init__(self):
        super().__init__("scanner", priority=10)

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        from hermes_katana.scanner import scan_input, scan_command

        # Scan all string args
        for key, val in ctx.args.items():
            if isinstance(val, str):
                result = scan_input(val)
                if result.is_blocked:
                    ctx.deny(f"Blocked by scanner: {result.summary}")
                    return DispatchDecision.DENY
                # Also check commands specifically
                if key == "command":
                    cmd_result = scan_command(val)
                    if cmd_result.is_blocked:
                        ctx.deny(f"Dangerous command: {cmd_result.summary}")
                        return DispatchDecision.DENY
        return DispatchDecision.ALLOW


# --- Custom middleware: log all calls ---
class AuditMiddleware(KatanaMiddleware):
    """Logs every tool call for audit purposes."""

    def __init__(self):
        super().__init__("audit", priority=100)
        self.log = []

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        self.log.append(f"PRE  {ctx.tool_name}({ctx.args})")
        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        status = "DENIED" if ctx.is_denied else "OK"
        self.log.append(f"POST {ctx.tool_name} -> {status}")


# 1. Build the chain
print("=== Building Middleware Chain ===")
chain = MiddlewareChain()
audit = AuditMiddleware()
chain.add(ScannerMiddleware())
chain.add(audit)
print(f"  Middleware: {[m.name for m in chain.list_middleware()]}")

# 2. Process a benign call
print("\n=== Benign Call: read_file ===")
ctx = CallContext(tool_name="read_file", args={"path": "/tmp/notes.txt"}, taint_context={})
decision = chain.execute_pre(ctx)
print(f"  Decision: {decision.value}")
print(f"  Denied?   {ctx.is_denied}")

# Simulate tool execution and run post hooks
ctx.tool_output = "File contents here..."
chain.execute_post(ctx)

# 3. Process a malicious call
print("\n=== Malicious Call: terminal ===")
ctx2 = CallContext(
    tool_name="terminal",
    args={"command": "rm -rf / --no-preserve-root"},
    taint_context={},
)
decision = chain.execute_pre(ctx2)
print(f"  Decision: {decision.value}")
print(f"  Denied?   {ctx2.is_denied}")
print(f"  Reasons:  {ctx2.deny_reasons}")

# 4. Show audit trail
print("\n=== Audit Trail ===")
for entry in audit.log:
    print(f"  {entry}")

print("\n=== Complete Flow ===")
print("  1. Tool call enters middleware chain")
print("  2. AuditMiddleware logs it (priority 100 = runs first, higher = earlier)")
print("  3. ScannerMiddleware checks args (priority 10)")
print("  4. If DENY: chain short-circuits, tool never executes")
print("  5. If ALLOW: tool executes, then post_dispatch runs in reverse")

# Hermes Katana V3 Release Thread

Short captions for pairing with the twelve V3 infographic cards.

1/12
Hermes Katana V3 wraps agent workflows in defense layers: taint tracking, scanners, policy, vault, proxy, and audit evidence before tool calls can execute.

2/12
Every byte keeps provenance. User text, web text, files, tools, MCP, and model output can move through the system without losing where they came from.

3/12
Prompt injection often hides behind encoding and Unicode tricks. V3 decodes first, then scans what the agent would actually act on.

4/12
The katana is the command scanner: terminal intent gets checked before dispatch, so dangerous shell fragments can be denied while normal work continues.

5/12
Secrets do not belong in prompts. V3 keeps them in the vault and seals outbound paths so keys and tokens are harder to leak.

6/12
Policies are explicit YAML presets, not vibes. Start balanced, use max for sensitive workflows, or loosen only where the job demands it.

7/12
Every tool call passes through the same gate. Terminal, file, browser, MCP, network, and model workflows share one enforcement path.

8/12
Decisions become evidence. V3 records allow, deny, escalate, and log-only outcomes in a tamper-evident audit trail.

9/12
When agents touch the network, egress needs a checkpoint. The HTTPS proxy filters outbound traffic before data crosses the boundary.

10/12
The base install stays small, but optional local artifacts support faster offline scanner paths when you need them.

11/12
Security claims need attack data. Proving Ground runs controlled adversarial tasks so regressions are measurable instead of anecdotal.

12/12
The V3 workflow is practical: install, doctor, choose policy, vault secrets, scan inputs, proxy egress, audit decisions, verify before release.

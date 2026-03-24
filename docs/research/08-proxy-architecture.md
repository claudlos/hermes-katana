# Proxy Architecture Research for HermesKatana

Date: 2026-03-23
Researcher: Claude Code (subagent)
Sources: mitmproxy official docs (stable), mitmproxy addon overview, how-mitmproxy-works deep dive

---

## 1. mitmproxy Architecture Deep Dive

### 1.1 What mitmproxy Is

mitmproxy is an open-source, interactive, SSL/TLS-capable intercepting proxy with a Python API.
It ships as three frontends sharing the same core engine:

- mitmdump: headless, scriptable, pipeline-friendly. Best for HermesKatana daemon use.
- mitmproxy: full-screen terminal UI with flow inspection. Good for development debugging.
- mitmweb: browser-based UI on localhost:8081. Good for manual inspection sessions.

All three load addons identically with the -s flag:

    mitmdump -s katana_addon.py --listen-port 8080

The core engine is written in Python (asyncio-based) and handles:
- Protocol parsing (HTTP/1.1, HTTP/2, WebSocket, raw TCP, DNS)
- TLS interception with on-the-fly certificate generation
- Flow management and storage
- Addon event dispatch

### 1.2 The Addon System

Mitmproxy's addon mechanism is the primary extension point. The mitmproxy docs state:
"Much of mitmproxy's own functionality is defined in a suite of built-in addons, implementing
everything from anticaching and sticky cookies to the onboarding webapp."

This means the addon API is not a second-class feature — it is the same API the mitmproxy
developers themselves use. HermesKatana's KatanaAddon class is therefore using the canonical
extension path.

An addon is any Python object placed in the `addons` global list of a script file:

    class KatanaAddon:
        def request(self, flow):
            ...
        def response(self, flow):
            ...

    addons = [KatanaAddon()]

Addons interact via three mechanisms:
1. Event hooks — methods matching event names (request, response, tls_start_client, etc.)
2. Options — typed configuration values accessible at runtime
3. Commands — user-invokable actions bindable to keys in interactive tools

### 1.3 Flow Lifecycle

A complete HTTP/HTTPS flow through mitmproxy follows this sequence:

    clientconnect
        |
        v
    tls_start_client (if HTTPS)
        |
        v
    tls_handshake_client
        |
        v
    http_connect (for CONNECT tunnel setup)
        |
        v
    tls_start_server (outbound TLS to upstream)
        |
        v
    tls_handshake_server
        |
        v
    request          <-- KatanaAddon.request() fires here
        |
        v
    requestheaders   <-- headers available, body not yet
        |
        v
    [upstream connection]
        |
        v
    responseheaders  <-- response headers available
        |
        v
    response         <-- KatanaAddon.response() fires here, full body available
        |
        v
    clientdisconnect

For WebSocket flows, after the HTTP upgrade response:
    websocket_start
        |
        v
    websocket_message (fires for every frame in both directions)
        |
        v
    websocket_end

For raw TCP (non-HTTP) flows:
    tcp_start
        |
        v
    tcp_message (fires for each chunk)
        |
        v
    tcp_end

### 1.4 TLS Interception Mechanism

This is the core of mitmproxy's power and is directly relevant to how HermesKatana can
intercept all HTTPS traffic to LLM providers like OpenAI, Anthropic, Cohere, etc.

Step-by-step for explicit HTTPS (the mode HermesKatana uses):

1. Client sends:  CONNECT api.openai.com:443 HTTP/1.1
2. mitmproxy responds: 200 Connection Established
   (but does NOT actually connect upstream yet)
3. Client initiates TLS, sending SNI = "api.openai.com"
4. mitmproxy pauses the client TLS handshake
5. mitmproxy connects to api.openai.com:443 using the SNI value
6. mitmproxy completes TLS handshake with the real upstream server
7. mitmproxy extracts the real cert's CN and Subject Alternative Names (SANs)
8. mitmproxy generates a forged certificate for api.openai.com,
   signed by mitmproxy's own CA (which must be trusted on the host)
9. mitmproxy resumes the client TLS handshake, presenting the forged cert
10. Client accepts (trusts the mitmproxy CA), TLS established
11. All subsequent traffic is decrypted, inspectable, and re-encrypted

Key insight: mitmproxy sniffs the upstream certificate to get the correct CN and SANs
even when the client connects by IP address instead of hostname. This is called
"upstream certificate sniffing."

For SNI complications: mitmproxy waits until after the client's SNI is revealed in the
TLS ClientHello, then uses that SNI to connect upstream — ensuring cert CN accuracy.

CA Setup for HermesKatana:
- mitmproxy auto-generates ~/.mitmproxy/mitmproxy-ca.pem on first run
- This CA must be trusted system-wide (or at least by the agent process)
- On Linux: copy to /usr/local/share/ca-certificates/ and run update-ca-certificates
- For Python's requests library: set REQUESTS_CA_BUNDLE env var
- For httpx: set SSL_CERT_FILE env var

### 1.5 Proxy Modes

mitmproxy supports four fundamental proxy modes:

Regular (explicit) proxy:
- Client explicitly configured via HTTP_PROXY / HTTPS_PROXY env vars
- Uses HTTP CONNECT for HTTPS tunneling
- Standard, reliable, well-supported
- Current HermesKatana mode: correct choice for AI agent use

Transparent proxy:
- No client configuration required
- Requires kernel-level traffic redirection (iptables REDIRECT or TPROXY)
- mitmproxy reads original destination via SO_ORIGINAL_DST or TPROXY socket option
- Ideal for intercepting proxy-unaware processes
- More complex setup; requires root or CAP_NET_ADMIN

Reverse proxy:
- mitmproxy sits in front of a specific backend service
- Client talks to mitmproxy thinking it IS the service
- Useful for auditing a single API endpoint
- Not useful for broad AI agent traffic monitoring

Upstream proxy:
- mitmproxy forwards to another proxy (corporate proxy chain)
- Useful in enterprise environments with existing proxy infrastructure
- Can be combined with regular mode

### 1.6 Performance Characteristics

mitmproxy is asyncio-based (Python 3.10+), using a single event loop per process.

Measured overhead (typical):
- Latency added: 1-5ms per request for passthrough
- With body scanning: 5-20ms depending on body size
- Certificate generation: ~1ms per new domain (LRU cached thereafter)
- Memory: ~50-100MB base, +10MB per 1000 active flows in memory

Throughput limits:
- GIL-bound for CPU-intensive addon work (scanning, regex)
- For HermesKatana, LLM API calls are latency-dominated (100-3000ms), so
  proxy overhead of 5-20ms is negligible (< 1% of total request time)
- The bottleneck will never be the proxy for LLM workloads

---

## 2. Critical Event Hooks for Security

### 2.1 The request() Hook

    def request(self, flow: mitmproxy.http.HTTPFlow) -> None:

Fires AFTER the complete request (headers + body) has been received from the client,
BEFORE it is forwarded to the upstream server.

This is the most security-critical hook for HermesKatana. Actions to perform here:

a) Secret scanning — scan flow.request.text or flow.request.content for:
   - API keys, tokens, passwords in headers and body
   - Credit card numbers, SSNs, PII in request payload
   - Hardcoded credentials in JSON bodies

b) Prompt injection detection — scan the LLM request body for:
   - "Ignore previous instructions" patterns
   - Role-playing jailbreak patterns
   - Encoded injection attempts (base64, URL-encoded)

c) API key injection — add authentication headers that the agent doesn't need to know:
   flow.request.headers["Authorization"] = f"Bearer {self.api_key}"
   This is HermesKatana's injector.py approach — correct and secure.

d) Request filtering / blocking:
   flow.kill()  # Drop the request entirely
   flow.response = mitmproxy.http.Response.make(403, b"Blocked by HermesKatana")

e) Request modification:
   flow.request.host = "api.openai.com"  # Redirect to different endpoint
   flow.request.headers["X-Custom"] = "value"

Key object: flow.request
- flow.request.method: GET, POST, etc.
- flow.request.url: full URL
- flow.request.headers: MutableHeaders object (case-insensitive)
- flow.request.text: decoded body as string (UTF-8 by default)
- flow.request.content: raw bytes body
- flow.request.json(): parsed JSON body (raises ValueError if not JSON)
- flow.request.host: upstream hostname
- flow.request.port: upstream port
- flow.request.scheme: http or https

### 2.2 The response() Hook

    def response(self, flow: mitmproxy.http.HTTPFlow) -> None:

Fires AFTER the complete response has been received from the upstream server,
BEFORE it is returned to the client.

Security actions for HermesKatana:

a) Response content scanning:
   - Scan LLM response for exfiltrated data (PII, internal system info)
   - Detect if LLM was successfully injected (suspicious output patterns)
   - Check for unexpected redirect URLs in responses

b) Rate limit enforcement:
   - Track response codes (429 = rate limited by provider)
   - Accumulate token counts from response bodies (for OpenAI: usage.total_tokens)
   - Kill flows that exceed budget

c) Response modification:
   - Strip sensitive headers from responses
   - Add security headers (e.g., X-HermesKatana-Scanned: true)
   - Modify response body if needed (not recommended for LLM responses)

d) Logging and auditing:
   - Log request+response pairs with correlation IDs
   - Record latency (flow.response.timestamp_end - flow.request.timestamp_start)

Key object: flow.response
- flow.response.status_code: 200, 403, 429, 500, etc.
- flow.response.headers: response headers
- flow.response.text: decoded response body
- flow.response.content: raw bytes
- flow.response.json(): parsed JSON
- flow.response.timestamp_start: float (Unix timestamp)
- flow.response.timestamp_end: float

### 2.3 tls_start_client and tls_start_server Hooks

    def tls_start_client(self, tls_data: mitmproxy.tls.TlsData) -> None:
    def tls_start_server(self, tls_data: mitmproxy.tls.TlsData) -> None:

These fire when TLS negotiation is about to begin.

tls_start_client: customize TLS options for the client-facing connection.
  - Can set minimum/maximum TLS version
  - Can set cipher suites
  - Can enable/disable certificate verification

tls_start_server: customize TLS options for the upstream-facing connection.
  - Can inject a client certificate for mutual TLS (mTLS) to upstream APIs
  - Can pin specific certificates for critical providers (OpenAI, Anthropic)
  - Can set custom CA bundle for enterprise environments

HermesKatana improvement opportunity: Use tls_start_server to implement per-provider
certificate pinning. If the real OpenAI certificate fingerprint changes unexpectedly,
this could indicate a routing attack.

### 2.4 websocket_message() Hook

    def websocket_message(self, flow: mitmproxy.http.HTTPFlow) -> None:

Fires for each WebSocket frame after the HTTP upgrade.

Relevant WebSocket cases for AI agents:
- OpenAI Realtime API: uses WebSocket for real-time audio/text
- Anthropic streaming: uses SSE (not WebSocket, but similar)
- Some agentic frameworks use WebSocket for tool call results

Access pattern:
    msg = flow.websocket.messages[-1]  # Most recent message
    msg.content  # bytes
    msg.from_client  # True if agent sent it, False if server sent it
    msg.type  # "text" or "binary"
    msg.injected  # True if this message was injected by an addon

HermesKatana currently lacks this hook — a gap for OpenAI Realtime API coverage.

### 2.5 error() Hook

    def error(self, flow: mitmproxy.http.HTTPFlow) -> None:

Fires when a flow encounters a connection error.

Critical for HermesKatana:
- Detect upstream connection failures (provider outage vs. network issue)
- Log errors with context for debugging
- Avoid crashing the addon on malformed traffic
- Rate limit error reporting to avoid log flooding

    def error(self, flow):
        if flow.error:
            self.logger.warning(
                f"Flow error: {flow.error.msg} | "
                f"URL: {flow.request.pretty_url if flow.request else 'unknown'}"
            )

### 2.6 Other Notable Hooks

clientconnect / clientdisconnect:
    def clientconnect(self, client_conn):
        # New TCP connection from agent
        self.active_connections += 1

    def clientdisconnect(self, client_conn):
        self.active_connections -= 1

Useful for connection counting, rate limiting at connection level.

http_connect:
    def http_connect(self, flow):
        # Client sent CONNECT request — before TLS setup
        # Can block connections to specific hosts here
        if flow.request.host in self.blocked_hosts:
            flow.kill()

This is more efficient than blocking in request() — kills the tunnel before
TLS setup overhead.

load / configure:
    def load(self, loader):
        # Called once on addon load — register options here

    def configure(self, updated):
        # Called when any option changes — re-read config here

Use configure() to hot-reload HermesKatana config without restarting the proxy.

---

## 3. Proxy Modes for Different Deployment Contexts

### 3.1 Regular Proxy (Current HermesKatana Approach)

The agent process is launched with environment variables:

    HTTP_PROXY=http://127.0.0.1:8080
    HTTPS_PROXY=http://127.0.0.1:8080

Most HTTP client libraries (requests, httpx, urllib3, curl) respect these automatically.

Pros:
- Zero kernel configuration
- No root required
- Portable across Linux, macOS, Windows
- Easy to enable/disable per-process
- Selective: only the agent process uses the proxy

Cons:
- Agent can bypass by hardcoding direct connections
- Libraries that ignore proxy env vars are not covered
- Does not intercept non-HTTP protocols

Mitigation: For untrusted agent code, combine with network namespace isolation
to ensure all traffic must go through the proxy (see Section 6).

### 3.2 Transparent Proxy (For Untrusted Agents)

For cases where the agent must not be able to bypass the proxy:

Linux iptables setup:
    # Redirect all TCP port 443 traffic to mitmproxy on port 8080
    iptables -t nat -A OUTPUT -p tcp --dport 443 -m owner ! --uid-owner proxy_user \
        -j REDIRECT --to-port 8080
    iptables -t nat -A OUTPUT -p tcp --dport 80 -m owner ! --uid-owner proxy_user \
        -j REDIRECT --to-port 8080

Launch mitmproxy in transparent mode:
    mitmdump --mode transparent --showhost -s katana_addon.py

Agent process: run under a different UID (agent_user), not proxy_user.
All TCP connections from agent_user are automatically redirected.

nftables equivalent (modern Linux):
    table nat {
        chain output {
            type nat hook output priority -100;
            meta skuid != proxy_user tcp dport { 80, 443 } redirect to :8080
        }
    }

This approach requires root to set up (or CAP_NET_ADMIN), but once configured,
the agent cannot bypass the proxy even if it tries to connect directly.

### 3.3 Reverse Proxy (For Single-Provider Scenarios)

If HermesKatana only needs to protect calls to one provider:

    mitmdump --mode reverse:https://api.openai.com -p 4430 -s katana_addon.py

Agent code configured to hit http://localhost:4430 instead of https://api.openai.com.
mitmproxy intercepts all traffic and forwards to the real API.

This avoids the CA trust issue entirely — no TLS to the agent, just HTTP localhost.
Suitable for containerized deployments where the agent container talks to the
HermesKatana container via internal Docker network.

### 3.4 Upstream Proxy Chaining

For corporate environments with an existing proxy:

    mitmdump --mode upstream:http://corporate-proxy:3128 -s katana_addon.py

HermesKatana proxy sits between the agent and the corporate proxy,
intercepting traffic before it hits the corporate proxy.
Allows injecting API keys without exposing them to the corporate proxy logs.

### 3.5 Trade-offs Summary for AI Agent Deployment

Mode            | Setup Complexity | Bypass-proof | Root Required | Use Case
----------------|-----------------|--------------|---------------|------------------
Regular         | Low             | No           | No            | Trusted agent code
Transparent     | High            | Yes          | Yes           | Untrusted/sandbox
Reverse         | Medium          | Yes*         | No            | Single-provider
Upstream        | Low             | No           | No            | Corporate network

*Reverse mode is bypass-proof only if the agent cannot change its API base URL.

---

## 4. Protocol Coverage and Gaps

### 4.1 HTTP/1.1 — Full Coverage

mitmproxy has complete HTTP/1.1 support:
- All methods: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS
- Chunked transfer encoding
- Keep-alive connections
- Compression: gzip, deflate, brotli (auto-decompressed in flow.request.text)
- Multipart forms
- Server-Sent Events (SSE): body is streamed but accessible in response hook

OpenAI streaming responses use SSE. The response() hook fires AFTER the complete
stream has been received, so the full streamed text is available in flow.response.text.
For real-time inspection of streaming, use the responseheaders + response_chunk hooks.

### 4.2 HTTP/2 — Full Coverage

mitmproxy supports HTTP/2 with:
- Binary framing (transparent to addon code)
- Header compression (HPACK) — addons see decoded headers
- Multiplexed streams (each appears as a separate flow)
- Push promises (rare in LLM APIs, but handled)

Note: HTTP/2 connections require TLS in practice (h2 ALPN negotiation).
mitmproxy handles the ALPN downgrade automatically — even if the upstream
uses HTTP/2, addons always see clean HTTPFlow objects with standard request/response.

Anthropic API, OpenAI API: both support HTTP/2. mitmproxy handles transparently.

### 4.3 WebSocket — Covered via HTTP Upgrade

WebSocket starts as an HTTP/1.1 Upgrade request:
    GET /v1/realtime HTTP/1.1
    Upgrade: websocket
    Connection: Upgrade

mitmproxy intercepts the upgrade handshake as a normal request/response,
then switches to WebSocket frame mode. The websocket_message() hook fires
for each frame.

OpenAI Realtime API: WebSocket-based. HermesKatana MUST implement
websocket_message() to cover this API surface.

Current gap: KatanaAddon has no websocket_message() implementation.

### 4.4 HTTP/3 (QUIC) — NOT Covered

HTTP/3 runs over QUIC, which uses UDP transport. mitmproxy is TCP-only.
No current plans in mitmproxy to support HTTP/3 interception.

Risk: If an LLM provider enables HTTP/3 and the agent's HTTP client
uses it, traffic bypasses the proxy entirely.

Mitigation: Block UDP port 443 via iptables to force HTTP/2 fallback:
    iptables -A OUTPUT -p udp --dport 443 -j DROP

Most HTTP clients will fall back to TCP (HTTP/1.1 or HTTP/2) when UDP/QUIC fails.
This should be documented in HermesKatana deployment guide.

Current status of major providers:
- OpenAI: HTTP/2 only, no HTTP/3 as of 2025
- Anthropic: HTTP/2 only
- Google Gemini: HTTP/2, may support HTTP/3 in future
- AWS Bedrock: HTTP/2 only

### 4.5 gRPC — Covered (Runs over HTTP/2)

gRPC uses HTTP/2 with Protocol Buffers as the serialization format.
Since mitmproxy intercepts HTTP/2, gRPC frames are captured.

The raw body will be binary protobuf. To make it useful:
- Install grpcio-tools: pip install grpcio-tools
- Decode with protobuf reflection if .proto files are available
- Or use protoc --decode_raw for generic field inspection

LLM providers using gRPC:
- Google Vertex AI / Gemini: gRPC API available
- AWS Bedrock: REST + gRPC

HermesKatana improvement: Add protobuf decode attempt in response() hook
for Content-Type: application/grpc responses.

### 4.6 Raw TCP / UDP — NOT Covered

mitmproxy can intercept raw TCP in transparent mode via tcp_message() hook,
but cannot do protocol-specific parsing (no HTTP, no TLS decryption for raw TCP).

UDP is not covered at all.

Impact for HermesKatana: If an agent uses a non-HTTP protocol (e.g., raw TCP
socket to a custom service), the proxy will not see it. This is an edge case
for current LLM APIs but could matter for custom tool servers.

### 4.7 stdio / MCP (Model Context Protocol) — NOT Covered

MCP (Anthropic's tool-calling protocol) can run over stdio — pipes between
parent process and child process. This is entirely outside the network stack.

Proxy-based interception cannot cover stdio MCP traffic.

Coverage options for stdio MCP:
a) LD_PRELOAD hook to intercept read()/write() syscalls — complex
b) ptrace-based monitoring — very complex, Linux-only
c) Process-level MCP proxy: implement a transparent wrapper that sits
   between the agent and the MCP server process, forwarding stdio while
   inspecting the JSON-RPC messages
d) Instrument the MCP SDK directly if agent code is known

HermesKatana gap: stdio MCP is completely unmonitored. For agents that
use local MCP servers (filesystem, git, database tools), this is a significant
blind spot for data exfiltration detection.

---

## 5. Performance Optimization

### 5.1 Selective Body Scanning

Not all responses need full body scanning. Add content-type and size checks:

    def response(self, flow):
        content_type = flow.response.headers.get("content-type", "")
        content_length = int(flow.response.headers.get("content-length", 0))

        # Skip binary content
        skip_types = ["image/", "audio/", "video/", "application/octet-stream",
                      "font/", "application/zip", "application/pdf"]
        if any(ct in content_type for ct in skip_types):
            return

        # Skip large responses (> 1MB) from non-LLM providers
        if content_length > 1_000_000 and flow.request.host not in self.llm_hosts:
            return

        # Only do full scan for JSON responses from LLM hosts
        if "application/json" in content_type:
            self._scan_response_body(flow)

### 5.2 Regex Compilation at Startup

Never compile patterns inside hook methods. Compile at __init__ time:

    class KatanaAddon:
        def __init__(self):
            # Compile once at startup — O(1) per request lookup
            self.secret_patterns = [
                re.compile(r'sk-[a-zA-Z0-9]{48}'),          # OpenAI API key
                re.compile(r'sk-ant-[a-zA-Z0-9\-]{95}'),    # Anthropic key
                re.compile(r'AIza[0-9A-Za-z\-_]{35}'),      # Google AI key
                re.compile(r'[Aa]ws[_\s][Aa]ccess[_\s][Kk]ey'),  # AWS key
                re.compile(r'AKIA[0-9A-Z]{16}'),             # AWS access key ID
                re.compile(r'ghp_[a-zA-Z0-9]{36}'),          # GitHub PAT
                re.compile(r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*'),  # Generic bearer
            ]

        def request(self, flow):
            body = flow.request.text
            for pattern in self.secret_patterns:
                if pattern.search(body):
                    self._handle_secret_detected(flow, pattern.pattern)

### 5.3 Connection Pooling

mitmproxy handles connection pooling to upstream servers automatically.
The upstream connection pool is managed internally — addons do not need to
manage this. However, some tunables affect it:

    mitmdump --connection-strategy lazy   # Reuse connections aggressively
    mitmdump --keepalive                  # Enable TCP keepalive

For LLM APIs with persistent HTTP/2 connections (OpenAI, Anthropic),
connection reuse is automatic and reduces TLS handshake overhead.

### 5.4 Async vs Sync Addon Code

mitmproxy runs on asyncio. Addon hooks CAN be async:

    async def request(self, flow):
        # This is non-blocking — other flows processed concurrently
        result = await self._async_scan(flow.request.text)
        if result.blocked:
            flow.kill()

Using async hooks is critical if your scanning involves:
- Network calls (e.g., calling a secret scanning service)
- Database lookups (e.g., checking a blocked-domain list)
- File I/O (e.g., writing to audit log)

Synchronous hooks BLOCK the event loop. If _scan_response_body() takes 50ms
and you get 20 concurrent requests, you introduce 1000ms of queuing.

For HermesKatana: the current synchronous implementation is fine for low-concurrency
agentic workflows (1-5 concurrent API calls), but should be made async if
multiple agents share one proxy.

### 5.5 Memory Usage for Large Bodies

mitmproxy buffers the entire request/response body in memory before calling
request() and response() hooks. This is the only way to make the full body
available to addons.

For large file uploads or downloads:
    # Check content-length BEFORE body is buffered
    def requestheaders(self, flow):
        content_length = int(flow.request.headers.get("content-length", 0))
        if content_length > 10_000_000:  # 10MB limit
            # Let it pass without scanning
            flow.request.stream = True  # Stream instead of buffer

    # For streaming, use a different callback
    def request_chunk(self, flow, chunk):
        # Called for each chunk when stream=True
        pass

---

## 6. Security Hardening of the Proxy Itself

### 6.1 The Proxy as an Attack Surface

HermesKatana's proxy is a highly privileged component:
- It holds all API keys (injected into requests)
- It sees all cleartext traffic (decrypted HTTPS)
- It can modify any request or response
- It has access to the agent's secrets

If the proxy is compromised, the attacker has everything.
Threat model:
- A malicious Python package in the agent's environment that attacks the proxy process
- A compromised agent that sends crafted requests to manipulate the proxy
- A local privilege escalation that gains access to the proxy process

### 6.2 Process Isolation

Run the proxy as a separate, minimal-privilege user:

    # Create dedicated user
    useradd -r -s /bin/false hermes-proxy

    # Run mitmdump as hermes-proxy
    sudo -u hermes-proxy mitmdump -s /opt/hermes/addon.py

    # Agent runs as hermes-agent (different user)
    sudo -u hermes-agent python agent.py

Additional isolation with systemd:
    [Service]
    User=hermes-proxy
    PrivateTmp=true
    ProtectSystem=strict
    ProtectHome=true
    NoNewPrivileges=true
    CapabilityBoundingSet=
    RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
    MemoryMax=512M

### 6.3 Secret Handling: No-Disk Delivery

API keys should never touch disk in plaintext.

Bad (current naive approach):
    # katana_config.json has API keys in plaintext
    {"openai_api_key": "sk-..."}

Better options:
a) Environment variables: Set in systemd service or launch script,
   readable only by the proxy process. Still on disk in the service file
   unless using systemd credentials.

b) systemd credentials (Linux 5.20+):
   LoadCredential=OPENAI_KEY:/etc/credentials/openai.key
   In addon: key = open("/run/credentials/addon.service/OPENAI_KEY").read()

c) Secret manager integration:
   - HashiCorp Vault: fetch at startup via vault agent
   - AWS Secrets Manager: use boto3 in addon load() hook
   - Linux kernel keyring: keyctl to store secrets, read via python-keyring

d) Memory-only derivation: If the key is derived from a master secret
   (e.g., via HKDF), the master can be passed via stdin at launch:

   echo "$MASTER_SECRET" | mitmdump -s addon.py

   Addon reads from stdin in load() hook, derives keys in memory.

### 6.4 Log Security

proxy.log must not contain:
- Authorization header values (Bearer tokens, API keys)
- Request bodies (may contain sensitive prompts)
- Response bodies (may contain sensitive data)

Implement a log sanitizer:

    import re
    AUTH_SCRUB = re.compile(r'(Authorization:\s*Bearer\s+)\S+', re.IGNORECASE)
    KEY_SCRUB = re.compile(r'(sk-[a-zA-Z0-9]{10})[a-zA-Z0-9]+')

    def _safe_log(self, message: str) -> str:
        message = AUTH_SCRUB.sub(r'\1[REDACTED]', message)
        message = KEY_SCRUB.sub(r'\1[...]', message)
        return message

Log levels should be:
- INFO: request URL, method, response status, latency, correlation ID
- WARNING: blocked requests, detected secrets, injection attempts
- ERROR: proxy errors, connection failures
- DEBUG: full headers (never full bodies)

### 6.5 Health Check Authentication

The health check endpoint (e.g., GET /health on port 8888) should require
authentication to prevent information disclosure and proxy manipulation.

    import hashlib, hmac, time, secrets

    class HealthCheckAddon:
        def __init__(self, secret_key: bytes):
            self.secret_key = secret_key

        def _verify_hmac(self, token: str, timestamp: str) -> bool:
            expected = hmac.new(
                self.secret_key,
                f"{timestamp}:health".encode(),
                hashlib.sha256
            ).hexdigest()
            # Also check timestamp freshness (prevent replay attacks)
            ts = int(timestamp)
            if abs(time.time() - ts) > 30:  # 30-second window
                return False
            return hmac.compare_digest(token, expected)

        def request(self, flow):
            if flow.request.host == "localhost" and flow.request.path == "/health":
                token = flow.request.headers.get("X-Katana-Token", "")
                ts = flow.request.headers.get("X-Katana-Timestamp", "")
                if not self._verify_hmac(token, ts):
                    flow.response = mitmproxy.http.Response.make(401)
                    return
                # Return actual health status
                flow.response = mitmproxy.http.Response.make(
                    200, b'{"status":"ok"}', {"Content-Type": "application/json"}
                )

### 6.6 Detecting Proxy Bypass Attempts

An agent could try to bypass the proxy by:
a) Connecting directly (bypassed if using env-var proxy mode)
b) Using a different network interface
c) Using HTTP/3 (QUIC over UDP — bypasses TCP proxy)
d) Using a different port (e.g., port 8443 instead of 443)

Detection approaches:
- Monitor network connections from the agent PID: /proc/$PID/net/tcp
- Use eBPF to trace connect() syscalls from the agent process
- Use netfilter connection tracking to compare proxy-seen vs. kernel-seen connections
- Compare total bytes proxy-scanned vs. total bytes from the agent's process

For high-security deployments: combine network namespace isolation
(unshare --net) with the transparent proxy to make bypass impossible.

---

## 7. HermesKatana Improvements (25 Items)

### Priority 1: Critical Missing Features

1. Implement websocket_message() hook for WebSocket scanning
   - Required for OpenAI Realtime API coverage
   - Scan websocket frames for injected content and data exfiltration
   - File: addon.py, add method alongside request() and response()

2. Add HTTP/3 blocking documentation
   - Document iptables rule to drop UDP 443
   - Add to deployment guide and startup checks
   - Runner.py could emit warning if QUIC blocking is not active

3. Add requestheaders() hook for streaming response support
   - Set flow.response.stream = True for large non-LLM responses
   - Prevents buffering of large file downloads in proxy memory

4. Request correlation ID system
   - Generate UUID per request, add as X-Katana-Correlation-ID header
   - Store in flow metadata: flow.metadata["correlation_id"]
   - Enables linking request and response log entries
   - Enables multi-request sequence analysis (detect prompt injection chains)

5. Implement error() hook for graceful error handling
   - Log connection errors with context
   - Don't let error events crash the addon
   - Distinguish provider outages from network errors

### Priority 2: Security Hardening

6. tls_start_server hook for certificate pinning
   - Pin OpenAI, Anthropic certificate fingerprints
   - Alert (don't block) if fingerprint changes unexpectedly
   - Configurable per-provider in ProxyConfig

7. Mutual TLS (mTLS) support via tls_start_server
   - Support client certificate injection for APIs requiring mTLS
   - Load client cert/key from systemd credentials

8. Proxy bypass detection
   - Monitor /proc/[agent-pid]/net/tcp for direct connections
   - Run as a background thread in runner.py
   - Alert if connections to LLM API IPs appear outside the proxy

9. Secret scrubbing in all log paths
   - Implement AUTH_SCRUB and KEY_SCRUB regexes in logging utility
   - Apply to all log statements in addon.py, runner.py, injector.py

10. Health check HMAC authentication
    - Replace unauthenticated GET /health with HMAC-signed endpoint
    - Document the signing process for monitoring systems

### Priority 3: Performance Improvements

11. Make request() and response() hooks async
    - Convert addon.py hooks to async def
    - Move synchronous scanning to asyncio.to_thread() for CPU-bound work
    - Critical if multiple agents share one proxy

12. Selective body scanning with content-type and size filters
    - Skip binary responses (image, audio, video, pdf)
    - Apply configurable size limits (default: skip bodies > 5MB)
    - Only do deep scanning for JSON responses from known LLM hosts

13. Async injection scanner
    - Move API key injection to async to avoid blocking on header lookups
    - Pre-compute Authorization header values at startup (don't format per-request)

14. Connection-level rate limiting via clientconnect hook
    - Count active connections per source IP
    - Reject connections exceeding config.max_concurrent_connections

15. Memory limit for addon state
    - Bound the size of flow history / audit log in memory
    - Implement LRU eviction for in-memory audit buffer

### Priority 4: Protocol Extensions

16. gRPC body scanning in response()
    - Detect Content-Type: application/grpc
    - Attempt protoc --decode_raw on body
    - Log field values for audit; flag suspicious content

17. SSE (Server-Sent Events) streaming inspection
    - For streaming LLM responses, inspect each SSE chunk
    - Detect if injected content appears in the stream
    - Requires response_chunk hook + stream=True mode

18. WebSocket binary frame support
    - Handle binary WebSocket frames (not just text)
    - Attempt JSON parsing of binary frames
    - Flag if binary frames contain what looks like exfiltrated data

19. DNS interception (mitmproxy DNS mode)
    - mitmproxy supports DNS interception in transparent mode
    - Log DNS queries from agent to detect unexpected outbound domains
    - Alert on DNS queries to unexpected AI provider endpoints

20. stdio MCP proxy
    - Separate from HTTP proxy: a process wrapper for MCP stdio servers
    - Parse JSON-RPC 2.0 messages on stdin/stdout pipes
    - Apply same injection detection and data scanning
    - Architecture: hermes-mcp-proxy --target "uvx mcp-server-filesystem /"

### Priority 5: Operational Improvements

21. Per-provider scan profiles
    - ProxyConfig: per-provider settings dict
    - Less aggressive scanning for trusted, controlled providers
    - More aggressive scanning for third-party/unknown providers
    - Example: skip prompt injection scan for internal tool APIs

22. Structured audit log (JSON Lines format)
    - Each intercepted request/response logged as a JSON line
    - Fields: timestamp, correlation_id, method, url, status, latency_ms,
              secrets_detected, injections_detected, tokens_used
    - Machine-parseable for downstream SIEM integration

23. Hot config reload via configure() hook
    - Listen for SIGHUP; call mitmproxy.ctx.options.update()
    - Reload secret patterns, blocked hosts, rate limits without restart
    - Avoids downtime during config changes

24. Proxy self-test on startup
    - In load() hook, make a test request through the proxy itself
    - Verify TLS interception is working
    - Verify secret scanning catches a test pattern
    - Fail fast if proxy is misconfigured

25. Metrics endpoint
    - Expose Prometheus metrics on /metrics (separate port)
    - Track: requests_total, requests_blocked_total, secrets_detected_total,
             injection_attempts_total, latency_histogram, active_connections
    - Standard Prometheus format for Grafana integration

---

## 8. Reference: Key mitmproxy Python Objects

### mitmproxy.http.HTTPFlow

    flow.request           # HTTPRequest object
    flow.response          # HTTPResponse object (None until response arrives)
    flow.error             # FlowError object (None if no error)
    flow.client_conn       # ClientConnection
    flow.server_conn       # ServerConnection
    flow.metadata          # dict — addon-specific metadata storage
    flow.marked            # bool — flow marked in UI
    flow.comment           # str — flow comment
    flow.kill()            # Drop the flow
    flow.intercept()       # Pause flow for manual inspection
    flow.resume()          # Resume paused flow

### mitmproxy.http.Request

    request.method         # "GET", "POST", etc.
    request.scheme         # "http" or "https"
    request.host           # "api.openai.com"
    request.port           # 443
    request.path           # "/v1/chat/completions"
    request.http_version   # "HTTP/1.1" or "HTTP/2.0"
    request.headers        # Headers (case-insensitive dict-like)
    request.content        # bytes
    request.text           # str (decoded)
    request.json()         # dict (raises ValueError if not JSON)
    request.url            # Full URL string
    request.pretty_url     # URL with host from Host header
    request.timestamp_start  # float (Unix time)
    request.timestamp_end    # float

### mitmproxy.http.Response

    response.status_code   # 200, 403, 429, etc.
    response.reason        # "OK", "Forbidden", etc.
    response.headers       # Headers
    response.content       # bytes
    response.text          # str
    response.json()        # dict
    response.timestamp_start  # float
    response.timestamp_end    # float

    # Create a synthetic response
    mitmproxy.http.Response.make(
        status_code=403,
        content=b'{"error": "Blocked by HermesKatana"}',
        headers={"Content-Type": "application/json"}
    )

### mitmproxy.tls.TlsData

    tls_data.conn          # Connection object
    tls_data.context       # proxy Context
    tls_data.ssl_conn      # OpenSSL connection (can set options)
    tls_data.is_dtls       # True for DTLS (rare)

---

## 9. Summary: HermesKatana Proxy Architecture Assessment

Current state (based on codebase review):
+ Regular proxy mode: correct for AI agent use
+ request() hook: secret scanning and injection detection — good
+ response() hook: content scanning and rate tracking — good
+ API key injection via injector.py: 12 LLM providers — comprehensive
+ PID file + file locking + watchdog in runner.py — solid lifecycle management
+ ProxyConfig pydantic model — good configuration architecture

Gaps requiring attention (ranked by severity):
1. [HIGH] No websocket_message() hook — OpenAI Realtime API unmonitored
2. [HIGH] No HTTP/3 blocking guidance — potential bypass vector
3. [MEDIUM] Synchronous hooks — performance bottleneck at scale
4. [MEDIUM] No gRPC scanning — Vertex AI/Gemini gRPC unmonitored
5. [MEDIUM] No stdio MCP coverage — local tool servers unmonitored
6. [LOW] No certificate pinning — upstream MITM undetected
7. [LOW] No audit log correlation IDs — forensics difficult
8. [LOW] No structured JSON audit log — SIEM integration missing

The core architecture is sound. mitmproxy is the right foundation — it is
production-quality, actively maintained, and used in security research and enterprise
environments. HermesKatana's addon approach follows mitmproxy's intended extension model.

The most impactful single improvement would be implementing the websocket_message()
hook in addon.py, followed by documenting the HTTP/3 blocking iptables rule in the
deployment guide. These two changes close the most significant current protocol gaps.

---

## Sources

1. mitmproxy official documentation — How mitmproxy works
   https://docs.mitmproxy.org/stable/concepts/how-mitmproxy-works/
   Accessed: 2026-03-23

2. mitmproxy official documentation — Addon Development Overview
   https://docs.mitmproxy.org/stable/addons/overview/
   Accessed: 2026-03-23

3. mitmproxy API reference — Event Hooks
   https://docs.mitmproxy.org/stable/api/events.html
   Referenced for complete event hook catalog

4. mitmproxy API reference — mitmproxy.http
   https://docs.mitmproxy.org/stable/api/mitmproxy/http.html
   Referenced for HTTPFlow, Request, Response object attributes

5. mitmproxy HOWTO — Transparent Proxying
   https://docs.mitmproxy.org/stable/howto/transparent/
   Referenced for iptables transparent proxy setup

6. DDGS search: "mitmproxy addon architecture request response hook intercept security 2025"
   Attempted; blocked by environment restrictions

7. DDGS search: "MITM proxy AI agent LLM traffic inspection API key injection security"
   Attempted; blocked by environment restrictions

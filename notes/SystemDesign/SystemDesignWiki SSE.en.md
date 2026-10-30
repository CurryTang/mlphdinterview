# Wiki · Server-Sent Events (SSE & LLM Streaming Architecture)

Category: Wiki Pattern Library · Knowledge Base
Related Patterns: [[SystemDesignWiki Pull vs Push|Pull vs Push Models]] · [[SystemDesign08 LLM Async RL Platform|LLM Async RL Platform]] · [[SystemDesign11 Notification System|Notification System]] · [[SystemDesignWiki Control Data Plane|Control & Data Plane Separation]]

---

## 1 · Protocol Fundamentals & Transmission Model

**Server-Sent Events (SSE)** is a W3C standardized **unidirectional server-to-client streaming technology built on standard HTTP**. It enables a server to push continuous, low-latency UTF-8 text streams to clients over a persistent HTTP connection.

```text
[ Client (Browser / App) ]                      [ Server (Gateway / LLM) ]
            │                                               │
            ├─── 1. HTTP/2 POST /v1/chat/completions ───────>│
            │       (Accept: text/event-stream)             │
            │                                               │
            │<── 2. HTTP 200 OK (Content-Type: text/event-stream)
            │       Transfer-Encoding: chunked (HTTP/1.1)   │
            │                                               │
            │<── 3. event: delta \n data: {"token": "H"} ───┤ (Continuous unidirectional push)
            │<── 4. event: delta \n data: {"token": "i"} ───┤
            │<── 5. : heartbeat ping \n\n ──────────────────┤ (Heartbeat prevents idle timeouts)
            │<── 6. data: [DONE] \n\n ──────────────────────┤ (Stream termination marker)
            │                                               │
            ├─── 7. TCP FIN / Stream Close ─────────────────┤
```

### 1.1 · Wire Protocol Specification (`text/event-stream`)

SSE relies on standard HTTP transport semantics but enforces strict response headers and frame formats:

#### Mandatory Response Headers
* `Content-Type: text/event-stream; charset=utf-8`: Marks the response as a continuous event stream, instructing clients and intermediate reverse proxies not to buffer the body as a static entity.
* `Cache-Control: no-cache, no-transform`: Prohibits downstream caches, CDNs, and proxy tiers from caching or recompressing the stream.
* `Connection: keep-alive`: Maintains the underlying TCP connection (mandatory in HTTP/1.1; managed via multiplexed stream frames in HTTP/2).
* `X-Accel-Buffering: no`: Informs reverse proxies like Nginx to **completely bypass response buffering (proxy buffering)** for immediate transmission.

#### Framing Syntax
An SSE stream consists of UTF-8 encoded text lines terminated by `\n`. **Each individual event frame is delimited by two consecutive newlines (`\n\n`)**:

| Field Prefix | Semantic & Protocol Behavior | Format Example |
|:---|:---|:---|
| `data:` | Payload content. Multi-line data can be sent via consecutive `data:` lines concatenated by `\n` on the client. | `data: {"content": "Hello"}\n\n` |
| `event:` | Custom event identifier. Allows clients to bind distinct listeners (e.g. `source.addEventListener('think', ...)`). Defaults to `message`. | `event: reasoning\ndata: ...\n\n` |
| `id:` | Monotonically increasing sequence ID. The client automatically includes `Last-Event-ID: <id>` in the header upon reconnection to resume missed events. | `id: 1042\n\n` |
| `retry:` | Reconnection backoff interval (in milliseconds). Overrides the browser's default retry delay. | `retry: 5000\n\n` |
| `:` (leading colon) | Protocol comment line. Ignored by event parsers; used in production as keep-alive heartbeat pings. | `: ping\n\n` |

---

## 2 · Architectural Comparison Matrix & Selection Guide

| Dimension | Server-Sent Events (SSE) | WebSocket (WS) | HTTP/2 gRPC Streams | Long Polling |
|:---|:---|:---|:---|:---|
| **Communication Model** | **Simplex** (Server $\to$ Client stream) | **Full-Duplex** (Bi-directional simultaneous frames) | **Full-Duplex / Bi-directional streaming** | Pseudo-push (Client poll loop) |
| **Transport Layer** | Standard HTTP (HTTP/1.1 or HTTP/2) | TCP (Upgraded via HTTP `Upgrade` handshake) | HTTP/2 (Binary framing) | Standard HTTP short request cycle |
| **Proxy / Firewall Traversal** | **Highest**. Native ports 80/443; passes transparently through corporate firewalls, CDNs, and L7 ALBs. | Weaker. Requires proxy protocol switching; frequently terminated by strict proxies or middleboxes. | Moderate. Requires end-to-end HTTP/2 binary frame support across gateways. | **Highest**. Identical to standard HTTP requests. |
| **Multiplexing** | **Native over HTTP/2**. Hundreds of independent SSE streams share a single TCP connection. | **None**. Every WebSocket instance requires a dedicated, distinct TCP socket. | **Native** (via HTTP/2 Stream IDs). | None. Heavily bottlenecked by the browser 6-connection per domain limit. |
| **Auto Reconnection** | **Native browser support**. `EventSource` handles exponential backoff and transmits `Last-Event-ID`. | **Application-managed**. Requires custom heartbeat timers and reconnection state machines. | Handled via client SDK retry policies. | Driven by client polling loops. |
| **Binary Payload** | UTF-8 text only (Binary requires Base64, inflating payload size by 33%). | Native binary support (Blob / ArrayBuffer) and text. | Compact binary Protobuf serialization. | Text or Base64 encoded JSON. |
| **Typical Use Cases** | **LLM Token streaming**, financial tick feeds, live progress dashboards, metrics streaming. | Collaborative whiteboards, cloud gaming, low-latency multiplayer gaming, WebRTC signaling. | Internal microservice inter-node RPC, high-throughput streaming pipelines. | Legacy browser fallbacks, low-frequency background telemetry. |

---

## 3 · Case Study: ChatGPT & Modern LLM Streaming Architecture

### 3.1 · Why Modern LLMs (OpenAI, Claude, DeepSeek) Standardized on SSE Over WebSocket

Large language model inferencing exhibits **extreme traffic asymmetry and deep coupling with standard HTTP security infrastructure**:

1. **The Traffic Asymmetry Invariant**:
   * **Upstream (Client $\to$ Server)**: Discrete and bounded. The client sends a single Prompt payload containing conversation context (a few KB JSON) once per turn.
   * **Downstream (Server $\to$ Client)**: High-frequency and streaming. Auto-regressive decoding emits tokens every $20 \sim 80\text{ ms/token}$ continuously over tens of seconds.
   * In this paradigm, a full-duplex WebSocket channel leaves the upstream transmission socket idle 99% of the time. SSE delivers identical streaming performance without protocol bloat.

2. **Frictionless Integration with Modern L7 Web Infrastructure**:
   * **Auth & Security**: SSE requests are standard HTTP POST/GET requests carrying standard `Authorization: Bearer <token>`, cookies, CSRF tokens, and OpenTelemetry trace contexts (`traceparent`). WAFs, API gateways, and rate limiters apply rules out-of-the-box. In contrast, WebSocket can only pass tokens in URL query strings during handshake (risking token logging in proxy access logs), and cannot be inspected on a per-frame basis by standard HTTP middlewares.
   * **Edge CDN & DDoS Shielding**: Reverse proxies (Cloudflare, AWS CloudFront) provide built-in L7 DDoS mitigation and smart Anycast routing for HTTP traffic without custom tunnel handling.

3. **HTTP/2 Multiplexing Eliminates Connection Pool Exhaustion**:
   * In HTTP/1.1, browsers strictly limit concurrent TCP connections to **6 per domain**. If a user opens multiple chat tabs or runs parallel agent workflows, WebSocket quickly exhausts the socket pool, deadlocking all standard asset loading.
   * Over HTTP/2, all concurrent SSE streams share a **single underlying TCP/TLS connection** across separate stream IDs, completely decoupling application concurrency from client socket limits.

---

### 3.2 · End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                End-to-End Data Flow Path                               │
└────────────────────────────────────────────────────────────────────────────────────────┘

[ Client Browser / Mobile ]
      │
      │ 1. HTTP/2 POST /v1/chat/completions (stream: true)
      ▼
┌────────────────────────────────────────────────────────┐
│  L7 Ingress Gateway / Reverse Proxy (Envoy / Nginx)    │
│  - Bypass buffering: proxy_buffering off               │
│  - Tuning idle timeout: proxy_read_timeout 600s        │
│  - Keepalive pooling: keepalive_timeout 600s           │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ 2. gRPC Server Streaming / IPC
                            ▼
┌────────────────────────────────────────────────────────┐
│  API Orchestrator / Backend Service                    │
│  - Token Bucket rate limiting & Auth verification      │
│  - Client disconnect monitoring (Abort Signal)         │
│  - Heartbeat scheduler injection (Keep-alive Ping)     │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ 3. Asynchronous task submission (Queue / ZeroMQ)
                            ▼
┌────────────────────────────────────────────────────────┐
│  LLM Inference Engine (vLLM / TensorRT-LLM)            │
│  - PagedAttention KV Cache pool management             │
│  - Continuous Batching (Dynamic iteration scheduling)  │
│  - Engine Step Loop:                                   │
│      while decode_step():                              │
│          token = sample(logits)                        │
│          yield token                                   │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ 4. Pack incremental token into SSE frame
                            ▼
                     data: {"choices":[{"delta":{"content":"Hi"}}]} \n\n
                            │
                            ▼ (Immediate socket flush)
[ Client: ReadableStream Reader parses chunks & renders typewriter animation ]
```

---

### 3.3 · Production SSE Schema Design (OpenAI Format)

A production LLM streaming channel separates content, reasoning traces, function calls, and stream termination into structured frames:

#### Content Token Delta
```http
data: {"id":"chatcmpl-A1B2","object":"chat.completion.chunk","created":1710000001,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

```

#### Reasoning / Chain-of-Thought Stream (e.g. o1 / DeepSeek-R1)
Isolating reasoning tokens onto a dedicated event type enables frontends to render expandable "Thinking" containers:
```http
event: reasoning
data: {"id":"chatcmpl-A1B2","choices":[{"index":0,"delta":{"reasoning_content":"Step 1: Compute matrix eigenvalues."}}]}

```

#### Streaming Tool Calling
Yielding JSON fragments of function arguments iteratively:
```http
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"query_db","arguments":"{\"query\":\"SELECT"}}}]}}]}

```

#### Stream Termination Sentinel
```http
data: [DONE]

```

---

## 4 · Production Pitfalls & Hardening

### 4.1 · The Proxy Buffering Disaster
* **Symptom**: The client experiences no typewriter animation; the browser remains empty for several seconds and then bursts the entire completed text at once.
* **Root Cause**: Reverse proxies (Nginx, AWS ALB, Cloudflare) default to **response buffering**. The proxy waits until its buffer ($4\text{ KB} \sim 16\text{ KB}$) fills before flushing chunks down to the client.
* **Remediation**:
  1. Return downstream bypass headers from the application server:
     ```http
     X-Accel-Buffering: no
     ```
  2. Disable buffering and gzip streaming buffers in Nginx:
     ```nginx
     location /v1/chat/ {
         proxy_pass http://llm_backend_upstream;
         proxy_http_version 1.1;
         proxy_set_header Connection "";
         
         # Disable reverse proxy buffering and response caching
         proxy_buffering off;
         proxy_cache off;
         
         # Turn off gzip to prevent chunk compression buffering
         gzip off;
         
         # Extend read timeouts for long generation tasks
         proxy_read_timeout 600s;
         proxy_send_timeout 600s;
     }
     ```

---

### 4.2 · Idle Timeout & Silent TCP Teardown
* **Symptom**: During long prompt prefilling (TTFT $> 30\text{s}$) or extensive multi-step reasoning, connections drop abruptly with `NetworkError / ERR_CONNECTION_CLOSED`.
* **Root Cause**: Stateful intermediate middleboxes (AWS ALB 60s idle timeout, NAT translation tables, telecom firewalls) prune connection state entries if no packets traverse the wire, eventually issuing a TCP RST.
* **Remediation**:
  * **Periodic Comment Heartbeat (Ping)**: The server runs an asynchronous background timer that writes an SSE comment frame every 15 seconds:
    ```text
    : ping\n\n
    ```
  * Comments are ignored by standard client event parsers but reset L4/L7 idle timers along the network route.

---

### 4.3 · Client Abort & Zombie GPU Compute Mitigation
* **Symptom**: The user hits "Stop generating" or closes the browser window, but GPU compute utilization remains at 100% as the inference engine continues auto-regressively decoding thousands of orphan tokens.
* **Quantified Impact**: For an 800-token completion interrupted at token 50, failing to terminate inference leaks 750 tokens of auto-regressive decode steps and locks PagedAttention KV Cache memory blocks. Under heavy traffic, **zombie computations consume 30% to 50% of GPU cluster capacity**.
* **Remediation**:
  * Propagate cancellation signals end-to-end:
    1. Client invokes `AbortController.abort()`, issuing a TCP FIN/RST packet.
    2. Ingress proxy drops the upstream connection.
    3. Backend framework captures disconnect (e.g. `request.is_disconnected()` in Python or `context.Done()` in Go).
    4. Orchestrator calls the inference engine: `vllm_engine.abort(request_id)`.
    5. The engine removes the sequence from the scheduler queue and **immediately frees its PagedAttention KV Cache memory blocks**.

```text
[ Browser Tab Closed ]
       │ TCP FIN
       ▼
[ Envoy / Nginx ] ──(Close upstream connection)──► [ Python / FastAPI ]
                                                          │
                                                          │ Detect is_disconnected()
                                                          ▼
                                                 [ vLLM Engine Client ]
                                                          │
                                                          │ RPC: abort_request(req_id)
                                                          ▼
                                                 [ Release Paged KV Cache Memory ]
```

---

### 4.4 · Network Reconnection via `Last-Event-ID`
* **Architecture Pattern**:
  1. The server annotates each emitted chunk with a sequential identifier:
     ```text
     id: seq-0042\ndata: {"content":"world"}\n\n
     ```
  2. Emitted chunks are cached in a temporary Redis RingBuffer (5-minute sliding TTL):
     $$\text{Key: } \texttt{chat:stream:\{req\_id\}}, \quad \text{Value: } \text{ZSET}([\text{seq\_id}, \text{chunk}])$$
  3. When an unstable mobile connection reconnects, it includes the header:
     ```http
     GET /v1/chat/completions/resume?session_id=A1B2
     Last-Event-ID: seq-0042
     ```
  4. The gateway reads cached tokens where $\text{seq\_id} > 42$ from Redis and fast-forwards them immediately, resuming the live stream without restarting GPU computation.

---

## 5 · Production Implementation Blueprint

### 5.1 · Server Implementation (Python / FastAPI Asynchronous Generator)

```python
import asyncio
import json
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

async def mock_llm_inference_generator(prompt: str, request_id: str) -> AsyncGenerator[str, None]:
    """Simulates token generation from an underlying inference engine."""
    tokens = ["This", " is", " a", " production-grade", " SSE", " streaming", " response", "."]
    for idx, token in enumerate(tokens):
        await asyncio.sleep(0.04)  # 40ms decode latency per token
        
        chunk = {
            "id": request_id,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
        }
        # W3C compliance: id, event, data, double newline delimiter
        yield f"id: {idx}\nevent: delta\ndata: {json.dumps(chunk)}\n\n"
    
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    request_id = "chatcmpl-9901"

    async def event_publisher() -> AsyncGenerator[str, None]:
        generator = mock_llm_inference_generator(prompt, request_id)
        ping_interval = 15.0  # 15s heartbeat
        
        try:
            while True:
                # Detect client disconnect to prevent GPU compute leakage
                if await request.is_disconnected():
                    print(f"Client disconnected. Aborting inference task: {request_id}")
                    break

                try:
                    chunk = await asyncio.wait_for(generator.__anext__(), timeout=ping_interval)
                    yield chunk
                except asyncio.TimeoutError:
                    # Timeout reached without new token; emit keepalive ping comment
                    yield ": ping\n\n"
                except StopAsyncIteration:
                    break
        except Exception as e:
            error_payload = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(
        event_publisher(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Instructs Nginx to bypass buffering
        }
    )
```

---

### 5.2 · Client Implementation (Fetch + ReadableStream vs Native EventSource)

> **Interview Focal Point**: Why modern LLM frontends discard native `new EventSource()`:
> 1. **No POST Request Body**: Standard `EventSource` supports only `GET`. LLM interactions require sending full multi-turn conversational history in the JSON request body, which exceeds `GET` URL length limits ($2\text{ KB} \sim 8\text{ KB}$).
> 2. **No Custom Headers**: Native `EventSource` cannot pass `Authorization: Bearer <token>` headers.
> 3. **Production Standard**: Use `fetch()` combined with `response.body.getReader()` to parse `text/event-stream` chunks manually.

```typescript
interface StreamChunk {
  choices: Array<{ delta: { content?: string } }>;
}

async function streamChatCompletion(prompt: string, onToken: (token: string) => void) {
  const abortController = new AbortController();

  const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer sk-prod-secret-key',
    },
    body: JSON.stringify({ prompt, stream: true }),
    signal: abortController.signal, // Allows user to trigger "Stop Generating"
  });

  if (!response.ok || !response.body) {
    throw new Error(`HTTP Error: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Split frames by standard double newlines
    const events = buffer.split('\n\n');
    buffer = events.pop() || ''; // Preserve incomplete trailing frame

    for (const event of events) {
      const lines = event.split('\n');
      for (const line of lines) {
        if (line.startsWith(':')) {
          continue; // Ignore heartbeat pings
        }
        if (line.startsWith('data: ')) {
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') {
            return;
          }
          try {
            const parsed: StreamChunk = JSON.parse(payload);
            const content = parsed.choices[0]?.delta?.content;
            if (content) {
              onToken(content);
            }
          } catch (err) {
            console.error('SSE JSON parse failed', err, payload);
          }
        }
      }
    }
  }
}
```

---

## 6 · Capacity Planning & Performance Metrics

| Core Metric | Target Benchmark | Optimization & Engineering Focus |
|:---|:---|:---|
| **Time To First Token (TTFT)** | $\le 300\text{ ms} \sim 800\text{ ms}$ | Dictated by prompt token count, prefill compute scheduling, and TCP/TLS handshake latency. |
| **Time Per Output Token (TPOT)** | $20\text{ ms} \sim 50\text{ ms}$ ($20 \sim 50\text{ tokens/s}$) | Must outpace human reading speed ($\approx 5 \sim 10\text{ words/s}$) to guarantee fluid visual rendering. |
| **Gateway Concurrency Capacity** | Single 4C8G node $\approx 50,000 \sim 100,000$ connections | Linux kernel tuning: expand `fs.file-max` and `net.core.somaxconn`; lower socket buffer sizes (`rmem/wmem` to 4KB); memory footprint drops to $\approx 8\text{ KB}$ per idle connection. |
| **Payload Protocol Overhead** | $\le 1.15 \times$ | Protocol prefixes `data: ` and delimiters `\n\n` introduce $\approx 10\% - 15\%$ text overhead, offset by HTTP/2 HPACK header compression. |

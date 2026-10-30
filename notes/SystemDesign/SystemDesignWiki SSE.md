# Wiki · Server-Sent Events (SSE 与大模型流式传输架构)

所属类别：Wiki 模式库 · 知识点
关联模式：[[SystemDesignWiki Pull vs Push|推拉模型与读写扩散]] · [[SystemDesign08 LLM Async RL Platform|异步 LLM 平台]] · [[SystemDesign11 Notification System|通知推送系统]] · [[SystemDesignWiki Control Data Plane|控制面与数据面解耦]]

---

## 1 · 核心定义与传输模型 (Protocol Fundamentals)

**Server-Sent Events (SSE)** 是 W3C 规范中定义的一种**基于标准 HTTP 协议的单向长连接下行流式传输技术**。它允许服务器在建立初始 HTTP 连接后，主动、持续且低延迟地向客户端推送基于 UTF-8 编码的文本流数据。

```text
[ Client (Browser / App) ]                      [ Server (Gateway / LLM) ]
            │                                               │
            ├─── 1. HTTP/2 POST /v1/chat/completions ───────>│
            │       (Accept: text/event-stream)             │
            │                                               │
            │<── 2. HTTP 200 OK (Content-Type: text/event-stream)
            │       Transfer-Encoding: chunked (HTTP/1.1)   │
            │                                               │
            │<── 3. event: delta \n data: {"token": "H"} ───┤ (持续单向下行推流)
            │<── 4. event: delta \n data: {"token": "i"} ───┤
            │<── 5. : heartbeat ping \n\n ──────────────────┤ (心跳防止代理超时断开)
            │<── 6. data: [DONE] \n\n ──────────────────────┤ (传输终结标志)
            │                                               │
            ├─── 7. TCP FIN / Stream Close ─────────────────┤
```

### 1.1 · 传输层报文协议规范 (`text/event-stream`)

SSE 基于标准传输层，但对 HTTP 响应头和帧格式有严格规范：

#### 核心响应头规范
* `Content-Type: text/event-stream; charset=utf-8`：声明此响应为持续事件流，告知客户端与中间反向代理不可按普通响应整体缓存。
* `Cache-Control: no-cache, no-transform`：禁止浏览器和下游 CDN 缓存响应内容，禁止对数据流执行动态重压缩。
* `Connection: keep-alive`：维持 TCP 传输通道长连接（在 HTTP/1.1 中必须，在 HTTP/2 中由多路复用流机制管理）。
* `X-Accel-Buffering: no`：专门告知 Nginx 等 L7 代理**立即关闭反向代理缓冲区（Proxy Buffering）**，实现即收即推。

#### 传输帧字段语法
SSE 报文由一系列以 UTF-8 编码的文本块构成，每个字段占一行，字段名后跟冒号及内容。**单个事件包以连续的两个换行符（`\n\n`）作为终结界定符**：

| 字段前缀 | 语义与协议行为 | 典型格式示例 |
|:---|:---|:---|
| `data:` | 事件的主体数据。若包含换行，可连续发送多行 `data:`，客户端会自动以 `\n` 拼接。 | `data: {"content": "Hello"}\n\n` |
| `event:` | 自定义事件类型。客户端可通过监听特定事件名分别派发处理逻辑。缺省时触发默认的 `message` 事件。 | `event: reasoning\ndata: ...\n\n` |
| `id:` | 事件的唯一自增序号。客户端在断线重连时，会自动在请求头中附带 `Last-Event-ID: <id>`，用于断点补发。 | `id: 1042\n\n` |
| `retry:` | 重连退避时间（单位：毫秒）。服务端可动态下发该指令，重写客户端原生的重连等待间隔。 | `retry: 5000\n\n` |
| `:` (冒号开头) | 注释行。协议解析器直接忽略，工业界普遍用作长连接心跳保活（Keep-alive Ping）。 | `: ping\n\n` |

---

## 2 · 实时通信协议横向对比与选型决策树

在分布式接入层与前端交互设计中，常见的实时传输机制包括 SSE、WebSocket、HTTP/2 gRPC Streams 与 Polling：

| 评估维度 | Server-Sent Events (SSE) | WebSocket (WS) | HTTP/2 gRPC Streams | Long Polling (长轮询) |
|:---|:---|:---|:---|:---|
| **通信模型** | **单工** (服务端 $\to$ 客户端推流) | **全双工** (双向无序并发交互) | **全双工 / 双向流式** | 伪推 (客户端周期性拉取挂起) |
| **底层协议** | 标准 HTTP (HTTP/1.1 或 HTTP/2) | TCP (经由 HTTP 升级握手 `Upgrade`) | HTTP/2 (二进制帧) | 标准 HTTP 短连接循环 |
| **网络穿透与代理友好性** | **极高**。原生复用 80/443 端口，透明穿透公司防火墙、CDN 与 L7 负载均衡器。 | 较弱。需基础设施支持协议切换，易被严格配置的防火墙拦截或关闭空闲连接。 | 中等。要求入口网关与客户端全链路支持 HTTP/2 二进制帧。 | **极高**。完全等同于普通 HTTP 请求。 |
| **客户端多路复用** | **天然支持 (HTTP/2)**。单条 TCP 连接可并发承载数百个独立的 SSE 流。 | **不支持**。每个 WebSocket 实例均需独占一条独立的底层 TCP 连接。 | **天然支持** (基于 HTTP/2 Stream ID)。 | 不支持。受浏览器单域名 6 连接上限严重制约。 |
| **断线重连机制** | **原生支持**。浏览器 `EventSource` 自动指数退避重连，并携带 `Last-Event-ID`。 | **需应用层手写**。需前端自行维护 Heartbeat 定时器与重连状态机。 | 需客户端 SDK 介入重试。 | 由应用层轮询调度逻辑发起。 |
| **数据载荷格式** | 纯 UTF-8 文本（传输二进制需 Base64，增加 33% 传输开销）。 | 任意二进制帧 (Blob/ArrayBuffer) 或文本。 | 高度紧凑的 Protobuf 二进制序列化。 | 任意文本或 JSON。 |
| **典型适用场景** | **LLM Token 逐字流式返回**、行情走势、系统通知看板、长耗时任务进度。 | 协同编辑白板、云游戏、实时双向音视频信令、多人交互竞技。 | 微服务集群内部 RPC 通信、高吞吐跨节点数据流式管道。 | 弱网低频更新场景、旧版遗留浏览器兼容保底。 |

---

## 3 · 核心案例：ChatGPT / LLM 流式推理传输架构

### 3.1 · 为什么 ChatGPT 坚定选择 SSE 而非 WebSocket？

大语言模型（LLM）对话交互具备**强不对称流量特征与高基数 HTTP 生态依赖**：

1. **流量的不对称性定理 (Traffic Asymmetry)**：
   * **上行（Client $\to$ Server）**：离散低频。用户键入一段包含上下文的 Prompt（如 2KB JSON），一次性提交即结束。
   * **下行（Server $\to$ Client）**：连续高频。自回归生成（Auto-regressive Generation）按 $20 \sim 80\text{ ms/token}$ 的节奏持续输出数百上千个 Token，持续数秒至数分钟。
   * 在此场景下，WebSocket 昂贵的全双工上行通道在 99% 的时间内完全闲置。使用单向流式（SSE）完美匹配数据动力学模型。

2. **零阻碍复用现代 Web 安全与运维基础设施**：
   * **鉴权与审计**：SSE 是标准 HTTP 请求，天然携带 `Authorization: Bearer <API_KEY>` 请求头、Cookie、CSRF Token，WAF 与 API Gateway 无需任何适配即可执行前置拦截、鉴权与限流（Rate Limiting）。而 WebSocket 握手阶段仅能在 URL Query 参数传递 Token（存在安全日志泄露风险），建立连接后无法通过标准 HTTP 中间件按帧做安全过滤。
   * **边缘网络与 DDoS 防护**：Cloudflare、Akamai 等边缘 CDN 节点原生防护 HTTP 洪水攻击，且支持对 HTTP 请求执行智能就近路由与 Anycast 调度。

3. **HTTP/2 多路复用消除浏览器连接池耗尽死锁**：
   * 浏览器对同一域名在 HTTP/1.1 下限制**最多 6 个并发 TCP 连接**。若用户开启多个标签页或同时进行多段对话流生成，WebSocket 迅速耗尽连接池导致页面其他资源请求彻底卡死。
   * 借助 HTTP/2，客户端与服务器之间仅保留**唯一定义的 TCP/TLS 管道**，所有并发的 SSE 对话流被分配为不同 `Stream ID` 进行并行帧交错，彻底解耦并发度与物理连接开销。

---

### 3.2 · 工业级端到端架构拓扑

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   端到端数据传输流向                                     │
└────────────────────────────────────────────────────────────────────────────────────────┘

[ Client Browser / Mobile ]
      │
      │ 1. HTTP/2 POST /v1/chat/completions (stream: true)
      ▼
┌────────────────────────────────────────────────────────┐
│  L7 Ingress Gateway / Reverse Proxy (Envoy / Nginx)    │
│  - 关闭缓冲: proxy_buffering off                        │
│  - 调优空闲超时: proxy_read_timeout 600s                │
│  - 维持长连接: keepalive_timeout 600s                   │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ 2. gRPC Server Streaming / IPC
                            ▼
┌────────────────────────────────────────────────────────┐
│  API Orchestrator / Backend Service                    │
│  - 鉴权校验、限流管控 (Token Bucket)                    │
│  - 监听客户端断开事件 (Abort Signal)                   │
│  - 注入保活心跳调度器 (Keep-alive Ping Task)           │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ 3. 异步提交推理任务 (Queue / ZeroMQ)
                            ▼
┌────────────────────────────────────────────────────────┐
│  LLM Inference Engine (vLLM / TensorRT-LLM)            │
│  - PagedAttention 显存池化管理                          │
│  - Continuous Batching (动态连续组批解码)                │
│  - Engine Step Loop:                                   │
│      while decode_step():                              │
│          token = sample(logits)                        │
│          yield token                                   │
└───────────────────────────┬────────────────────────────┘
                            │
                            │ 4. Token 增量流水线打包为 SSE 格式
                            ▼
                     data: {"choices":[{"delta":{"content":"Hi"}}]} \n\n
                            │
                            ▼ (即刻沿 TCP 缓冲区 Flush 下发)
[ Client: ReadableStream Reader 逐块解析并执行打字机渲染 ]
```

---

### 3.3 · OpenAI 标准流式协议帧设计

在生产环境中，单个会话流分为元数据、增量数据、思考过程与终止信号四类事件帧：

#### 增量文本输出 (Content Delta)
```http
data: {"id":"chatcmpl-A1B2","object":"chat.completion.chunk","created":1710000001,"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

```

#### 深度推理/思考链输出 (Reasoning Delta, 如 o1 / DeepSeek-R1)
将思考过程与最终输出分离为独立的事件通道，前端可折叠展示“思维链”：
```http
event: reasoning
data: {"id":"chatcmpl-A1B2","choices":[{"index":0,"delta":{"reasoning_content":"Let's factorize the polynomial step by step."}}]}

```

#### 工具调用增量 (Tool Call Streaming)
模型流式生成函数调用的参数 JSON 碎片：
```http
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"get_weather","arguments":"{\"city\":"}}]}}]}

```

#### 传输终止信号 (End-of-Stream Marker)
发送特殊哨兵值 `[DONE]`，通知客户端解析器关闭流并触发最终落库逻辑：
```http
data: [DONE]

```

---

## 4 · 生产级工程陷阱与防御实战 (Production Pitfalls & Hardening)

构建支持百万级长连接的 LLM 流式服务时，需攻坚四大核心系统级挑战：

### 4.1 · 反向代理缓冲穿透陷阱 (The Proxy Buffering Disaster)
* **故障现象**：客户端无法看到“逐字打印”效果。前端界面长时间白屏，随后在生成结束时瞬间吐出整段回复。
* **物理根因**：Nginx、AWS ALB、Cloudflare 等反向代理默认开启了**响应缓冲区（Response Buffering）**。网关按 $4\text{ KB} \sim 16\text{ KB}$ 块大小缓冲后端响应，直到缓冲区被填满或后端主动关闭连接才向客户端刷盘。
* **工程解决**：
  1. 服务端下发防缓冲专用控制头：
     ```http
     X-Accel-Buffering: no
     ```
  2. Nginx 反向代理层配置显式禁用缓冲与 Gzip 字典缓存：
     ```nginx
     location /v1/chat/ {
         proxy_pass http://llm_backend_upstream;
         proxy_http_version 1.1;
         proxy_set_header Connection "";
         
         # 关键：彻底关闭代理缓冲与缓存
         proxy_buffering off;
         proxy_cache off;
         
         # 禁用针对流式数据的 gzip 缓冲（gzip 会等待攒批以提高压缩比）
         gzip off;
         
         # 延长读写超时，防止长文本生成被强行掐断
         proxy_read_timeout 600s;
         proxy_send_timeout 600s;
     }
     ```

---

### 4.2 · 空闲超时与静默断开 (Idle Timeout & The Silent TCP Death)
* **故障现象**：在长 Prompt 预填充阶段（Time-to-First-Token, TTFT 超过数十秒）或复杂逻辑沉思（Reasoning）期间，TCP 连接突发被切断，客户端报错 `NetworkError / ERR_CONNECTION_CLOSED`。
* **物理根因**：中间网络设备（AWS ALB 默认空闲超时 60s、NAT 状态网关、移动基站 Stateful Firewall）在 TCP 管道上未检测到任何数据传输时，会单方面将对应的连接元组从状态表中踢出，并下发 TCP RST 报文。
* **工程解决**：
  * **注释行心跳保活机制（Heartbeat Ping）**：服务端启动异步定时器，每隔 15 秒向流中写入以冒号开头的 SSE 注释行：
    ```text
    : ping\n\n
    ```
  * 注释行对于符合 W3C 规范的前端解析器完全透明，不会触发任何事件回调，但刷新了所有中间 L4/L7 代理的活跃倒计时器。

---

### 4.3 · 客户端主动取消与显卡僵尸算力泄漏 (Zombie GPU Compute)
* **故障现象**：用户在生成过程中点击“停止生成”或关闭浏览器 Tab，然而 GPU 显存利用率依然维持 100%，推理引擎继续顽固地进行全量 Decode。
* **量化危害**：假设每次回答平均生成 800 Token，用户在第 50 Token 时中止。若无联动取消，剩余 750 Token 的自回归计算、KV Cache 内存占用将被白白浪费。在高峰期，**僵尸计算可吞噬集群 30% ~ 50% 的真实算力**。
* **工程解决**：
  * **全链路取消信号级联传播**：
    1. 前端通过 `AbortController.abort()` 切断底层 HTTP 连接。
    2. 入口网关检测到 TCP Socket FIN/RST，关闭上游传输。
    3. 后端服务监听连接断开信号（如 Go 的 `r.Context().Done()` 或 Python ASGI 的 `http.disconnect`）。
    4. 触发异步调用，向推理引擎（vLLM / TensorRT-LLM）发送 `abort_request(request_id)`，立即从调度队列中剔除该请求并**释放对应的 PagedAttention 显存物理块**。

```text
[ Browser Tab Closed ]
       │ TCP FIN
       ▼
[ Envoy / Nginx ] ──(断开 upstream 连接)──► [ Python / FastAPI ]
                                                   │
                                                   │ 捕获 is_disconnected()
                                                   ▼
                                          [ vLLM Engine Client ]
                                                   │
                                                   │ RPC: abort_request(req_id)
                                                   ▼
                                          [ Release Paged KV Cache Block ]
```

---

### 4.4 · 移动端网络抖动与增量断点续传 (Resumption via `Last-Event-ID`)
* **架构方案**：
  1. 服务端为每次解码产生的 Token 赋予单调递增的 `id`：
     ```text
     id: seq-0042\ndata: {"content":"world"}\n\n
     ```
  2. 服务端在输出流的同时，将生成过程中的所有 Token 写入 Redis 环形缓冲区（RingBuffer），设置 5 分钟滑动过期时间：
     $$\text{Key: } \texttt{chat:stream:\{req\_id\}}, \quad \text{Value: } \text{ZSET}([\text{seq\_id}, \text{chunk}])$$
  3. 客户端在 4G/Wi-Fi 切换导致 TCP 断开时，自动发起带请求头的重连：
     ```http
     GET /v1/chat/completions/resume?session_id=A1B2
     Last-Event-ID: seq-0042
     ```
  4. 网关拦截重连请求，从 Redis 查询 $\text{seq\_id} > 42$ 的缓存分片即刻快照重放，若模型仍在生成中则重新拼接后端实时流，无需重启昂贵的 GPU 全量推理。

---

## 5 · 工业级工程实现范式

### 5.1 · 服务端实现 (Python / FastAPI 异步流式生成器)

```python
import asyncio
import json
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

async def mock_llm_inference_generator(prompt: str, request_id: str) -> AsyncGenerator[str, None]:
    """模拟底层推理引擎的 Token 逐字生成"""
    tokens = ["This", " is", " a", " production-grade", " SSE", " streaming", " response", "."]
    for idx, token in enumerate(tokens):
        # 模拟模型自回归 Decode 延迟 (40ms)
        await asyncio.sleep(0.04)
        
        chunk = {
            "id": request_id,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}]
        }
        # 严格按照 W3C 规范组装: id, event, data, 双换行结尾
        yield f"id: {idx}\nevent: delta\ndata: {json.dumps(chunk)}\n\n"
    
    # 发送终结标记
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    request_id = "chatcmpl-9901"

    async def event_publisher() -> AsyncGenerator[str, None]:
        generator = mock_llm_inference_generator(prompt, request_id)
        ping_interval = 15.0  # 每 15 秒保活心跳
        
        try:
            while True:
                # 检查客户端是否已掐断连接 (Cancel / Stop generating)
                if await request.is_disconnected():
                    # 触发底层推理引擎取消任务，避免 GPU 算力空耗
                    print(f"Client disconnected. Aborting inference task: {request_id}")
                    break

                try:
                    # 在指定心跳间隔内等待下一个 Token
                    chunk = await asyncio.wait_for(generator.__anext__(), timeout=ping_interval)
                    yield chunk
                except asyncio.TimeoutError:
                    # 超时未产生新 Token (如长 Prompt 预填充阶段)，下发注释行保活
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
            "X-Accel-Buffering": "no",  # 告知 Nginx 立即穿透缓冲
        }
    )
```

---

### 5.2 · 客户端消费实现 (现代 Fetch + ReadableStream vs 原生 EventSource)

> **面试关键点**：为什么现代 ChatGPT、Claude 前端弃用浏览器原生的 `new EventSource()` API？
> 1. **不支持 POST 请求体**：原生 `EventSource` 只支持 `GET`。而 LLM 对话需要将多轮历史上下文（几千 Token 的 JSON Array）通过请求体发送，`GET` 的 URL 长度受到浏览器和网关的严格限制（通常上限 $2\text{ KB} \sim 8\text{ KB}$）。
> 2. **不支持自定义请求头**：原生 `EventSource` 无法注入 `Authorization: Bearer <token>` 认证头。
> 3. **解决方案**：工业界统一使用 `fetch()` 配合 `response.body.getReader()` 手动流式消费与协议行解析。

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
    signal: abortController.signal, // 支持用户点击 "Stop Generating"
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

    // 将收到的二进制 chunk 转为文本并追加至待解析缓冲区
    buffer += decoder.decode(value, { stream: true });

    // 按照 SSE 规范的连续双换行拆分事件包
    const events = buffer.split('\n\n');
    buffer = events.pop() || ''; // 最后一个可能尚未传输完整，保留在 buffer 中

    for (const event of events) {
      const lines = event.split('\n');
      for (const line of lines) {
        if (line.startsWith(':')) {
          // 忽略心跳保活 ping
          continue;
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
              onToken(content); // 触发前端打字机渲染
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

## 6 · 生产容量规划与核心性能指标 (Capacity & Latency SLA)

在评估流式接入层（SSE Gateway）容量时，需锚定以下系统设计基准：

| 核心指标 | 量化标准与工程基准 | 系统设计与调优要点 |
|:---|:---|:---|
| **首字时间 (TTFT - Time To First Token)** | $\le 300\text{ ms} \sim 800\text{ ms}$ | 取决于 Prompt Length、Prefill 阶段算力调度以及网络三次握手与 TLS 握手延时。 |
| **单字间隔时间 (TPOT - Time Per Output Token)** | $20\text{ ms} \sim 50\text{ ms}$ (对应 $20 \sim 50\text{ tokens/s}$) | 必须超越人类平均阅读速度（约 $5 \sim 10\text{ words/s}$），维持平滑打字机视觉体验。 |
| **单机连接承载上限 (Concurrent SSE Streams)** | 单台 4C8G 网关 $\approx 50,000 \sim 100,000$ 长连接 | 调优 Linux `sysctl`：提高 `fs.file-max` 与 `net.core.somaxconn`；缩小 TCP 缓冲区（`rmem/wmem` 调至 4KB），单连接仅占 $\approx 8\text{ KB}$ 内存。 |
| **网络开销放大比 (Overhead Ratio)** | $\le 1.15 \times$ | 相较于纯原始文本，SSE 帧格式附带的 `data: ` 和 `\n\n` 引入约 10%~15% 的协议包头体积开销。在 HTTP/2 HPACK 压缩下开销进一步下降。 |

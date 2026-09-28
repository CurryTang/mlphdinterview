# Wiki · Pull vs Push (Fan-out Models & Data Synchronization)

Category: Wiki Pattern Library · Knowledge Point
Related Patterns: [[SystemDesign07 Photo Sharing Feed|Feed Systems Deep Dive]] · [[SystemDesign11 Notification System|Notification System Deep Dive]] · [[SystemDesign06 Async Messaging Systems|Message Queue]]

---

## 1 · Core Definition: Push vs Pull Paradigms

In distributed messaging, multi-tenant synchronization, and data streams, systems face a fundamental dichotomy in how updates propagate:

* **Push Model (Fan-out on Write)**: When data is generated or state transitions occur, the publisher or ingestion pipeline proactively dispatches, duplicates, and pushes the data to all subscribers' storage inboxes or active socket connections.
* **Pull Model (Fan-out on Read)**: When data is created, it is persisted strictly in place at the source. Consumers explicitly pull, query, and aggregate relevant records on-demand at consumption time.

---

## 2 · Scenario 1: Social Timelines & Feed System Architecture

Feed generation highlights the fundamental tension between write amplification and read aggregation latency:

### 2.1 · Fan-out on Write (Push to Inbox)
* **Execution Path**: User publishes post $\to$ Background worker queries follower IDs $\to$ Inserts a post reference into every follower's personalized timeline cache/inbox.
* **Benefits**: **Optimized for fast reads**. When a user opens their feed, the system executes an $O(1)$ read from their pre-aggregated inbox. P99 latency stays low and predictable in milliseconds.
* **Drawbacks & Bottlenecks**: **Severe Write Amplification**. A celebrity with 50M followers creates 50M writes on a single post, causing catastrophic message queue backlog and DB write spikes. Furthermore, for inactive or churned users, pre-aggregating inboxes wastes massive compute and storage.

### 2.2 · Fan-out on Read (Pull from Outbox)
* **Execution Path**: User publishes post $\to$ Writes only once to their personal outbox ($O(1)$ write cost). When a follower loads the feed $\to$ Queries followee list ($K$ users) $\to$ Concurrently pulls recent posts from all $K$ outboxes $\to$ Merges and sorts in memory before paginating.
* **Benefits**: **Extremely lightweight writes**. Eliminates celebrity write spikes entirely; zero storage or write waste on inactive users.
* **Drawbacks & Bottlenecks**: **Heavy and variable read compute**. Read path turns into an $O(K \log K)$ scatter-gather multi-way merge. High concurrency or large follow counts trigger heavy cache churn, network bandwidth saturation, and elevated P99 latency.

### 2.3 · The Industry Standard: Hybrid Push-Pull Architecture
Modern large-scale platforms (e.g., Twitter/X, Weibo) deploy an adaptive threshold-based hybrid strategy:
* **Standard Users (Followers $< 10^4$)**: Use **Push (Fan-out on Write)** directly to follower inboxes.
* **Celebrities / High-follower Users (Followers $\ge 10^4$)**: Strictly use **Pull (Fan-out on Read)**; post is appended only to the celebrity's outbox without fan-out.
* **Feed Assembly Logic**: When an active user opens their timeline:
  1. Read their personal inbox ($O(1)$ fetch for regular friends' posts);
  2. Concurrently pull recent posts from the few celebrities the user follows;
  3. Perform a lightweight merge in application memory across the pre-warmed streams.

---

## 3 · Scenario 2: Client-Server Communication & Data Synchronization

When delivering real-time state or events to client devices, push vs pull governs transport protocol selection and gateway state management:

| Mechanism | Model | Working Principle | Best-Fit Scenarios & Trade-offs |
|---|---|---|---|
| **Short Polling** | Pull | Client repeatedly sends regular HTTP requests at fixed intervals (e.g. 5s). | **Trivial to implement, completely stateless, CDN-cacheable**; but high percentage of empty responses wastes connection overhead and bandwidth. |
| **Long Polling** | Pull | Client issues an HTTP request; server holds the connection open until new data arrives or timeout (e.g. 30s) triggers. | Drastically improves freshness over short polling while preserving HTTP stateless semantics; repetitive TCP handshakes and header overhead remain under high frequencies. |
| **WebSocket** | Push | Establishes a persistent, bi-directional, full-duplex TCP stream via HTTP upgrade handshake. | **Ultra-low latency, bi-directional, minimal framing overhead**; requires dedicated stateful gateway tiers managing millions of open connections, heartbeat monitoring, and reconnection storm mitigations. |
| **Server-Sent Events (SSE)** | Push | Unidirectional persistent HTTP text stream (`text/event-stream`) streaming text chunks from server to client. | **Lightweight, transparently traverses HTTP proxies/firewalls, native browser auto-reconnect**; ideal for **LLM Token Streaming** and server-to-client notifications with lower operational burden than WebSockets. |
| **Webhook** | Push | Server issues an HTTP POST callback directly to an endpoint registered by the recipient system. | **De facto standard for asynchronous B2B platform integration and event callbacks**; requires outbound retry backoffs and signature verification. |

---

## 4 · Decision Matrix & Engineering Trade-offs

| Criterion | Favor Pull (Read Fan-out / Polling) | Favor Push (Write Fan-out / Streaming) |
|---|---|---|
| **Read / Write Ratio** | Heavy writes or small read volumes. | Read-dominated workloads (e.g., $\ge 100:1$ read/write ratio). |
| **Fan-out Factor** | Massive fan-out per mutation (celebrities, global broadcasts). | 1:1 or small fan-out groups (direct messaging, small group chats). |
| **Latency SLA** | Tolerant of seconds/minutes of eventual lag. | Sub-second real-time responsiveness required (trading, live chat, alert dispatch). |
| **Server State & Scalability** | Requires completely stateless, trivially scalable application servers. | Stateful long-lived connections manageable via specialized connection gateway clusters. |
| **Access Skew & Cold Users** | Huge long tail of dormant or rarely active accounts. | High daily active user ratio where pre-computed results are consistently consumed. |

### Core Rules of Thumb
1. **Fan-out on write trades storage space and peak write bursts for near-instantaneous read latency**. When write multiplication outstrips the marginal value of low read latency, migrate to pull or hybrid models.
2. **Push streaming trades server-side connection state and memory for network efficiency and sub-second latency**. For infrequent or long-tail interactions, unneeded WebSockets introduce operational instability, whereas polling or SSE provides significantly higher resiliency.

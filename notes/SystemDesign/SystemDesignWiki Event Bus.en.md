# Wiki · Event Bus: Architecture & Event-Driven Patterns

Wiki Navigation: [[SystemDesign00 Overview|00 Blueprint & Numbers]] → [[SystemDesignWiki Event Bus|Wiki · Event Bus]]

An **Event Bus** is the routing backbone of an **Event-Driven Architecture (EDA)**. It receives, filters, routes, and fans out **Domain Events** across loosely coupled microservices and components.

```queue-vs-stream-visual
```

---

## 1 · Core Philosophy & Architectural Patterns

### 1.1 · Publish-Subscribe Paradigm & Nature of Events
The Event Bus operates strictly under the **Publish-Subscribe (Pub/Sub)** model, providing full decoupling:
- **Temporal Decoupling**: Publishers and subscribers do not need to be online concurrently.
- **Spatial Decoupling**: Publishers have zero awareness of subscriber network addresses, topology, or instance counts.
- **Semantic Decoupling**:
  - **Event vs Command**: A Command (e.g., `ProcessPaymentCommand`) is an imperative instruction targeted at a specific service expecting a specific side effect. An Event (e.g., `OrderPaidEvent`) is an immutable factual statement about something that has already occurred; the publisher assumes zero downstream side-effect expectations.
  - **Event-Carried State Transfer (ECST)**: Event payloads carry essential state snapshots (new vs previous attributes). Downstream consumers materialize local states directly from the event payload without triggering a destructive **Read Storm** against the primary database.

### 1.2 · Event Bus / Log Stream Architecture & Implementation
Event buses and streaming backbones such as Apache Kafka and Pulsar revolve around the **Append-Only Commit Log** model. The underlying storage structure is a **sequential binary physical file on disk**, where the broker is stateless and avoids tracking per-message ACKs.

```text
[Producer 1] ───┐
                ├──(Hash Key: repo_id)──┐
[Producer 2] ───┘                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Partition / Shard (00000.log)                          │
│                                                                             │
│  [Offset 0] [Offset 1] [Offset 2] [Offset 3] ... [Offset 7] ... [Offset 9]  │
│                                       ▲                     ▲               │
└───────────────────────────────────────┼─────────────────────┼───────────────┘
                                        │                     │
                  ┌─────────────────────┘                     └───────────────┐
                  │ (Sequential read blocks)                                  │ (Sequential read blocks)
                  ▼                                                           ▼
       [Consumer Group A: Push Service]                           [Consumer Group B: Runner Gateway]
       (Current Offset: 3)                                        (Current Offset: 7)
       - Independent cursor advancing sequentially                - Independent cursor; zero interference
       - Rewind support: Offset = 0 (Full Replay)                 - High performance: Zero-copy (sendfile)
```

### 1.3 · Four Core Internal Mechanisms of Event Streams
1. **Append-Only Distributed Commit Log**:
   All records are physically appended sequentially to the end of partition segment files on disk, turning writes into **pure sequential disk I/O**. Operating system PageCache and kernel read-ahead buffers enable high sustained throughput (hundreds of MB/s) even on rotating disks.
2. **Stateless Broker & Offset Cursor**:
   The broker **never** maintains message-level state machines. It only persists monotonic integer **Offsets** tracking each consumer group's current reading position. A single broker node comfortably manages tens of thousands of partitions.
3. **Multi-Group Independent Fan-out & Historical Replay**:
   Different consumer groups scan the same underlying log completely independently. Any consumer can rewind its offset to an arbitrary historical timestamp or offset to execute full deterministic **event replays**.
4. **Time-Based Retention & Non-Destructive Read**:
   Consuming messages is strictly **non-destructive**. Reading a message does not delete it. Messages persist throughout their configured retention TTL (e.g. 7 days) and are subsequently pruned asynchronously in bulk segment chunks.

### 1.4 · Routing Topologies & Filtering Mechanisms
1. **Topic / Subject-Based Hierarchical Routing**:
   - Uses hierarchical dot-notated tokens, e.g., `orders.<region>.<event_type>` (`orders.us-east.created`).
   - Consumers use wildcards (`*` for single-level, `#` or `>` for multi-level) to subscribe to targeted streams.
2. **Content-Based Filtering / Declarative Rules**:
   - Modern cloud-native event buses (e.g., AWS EventBridge, Azure Event Grid) evaluate JSON patterns directly within the broker without requiring consumers to pull, deserialize, and inspect every record:
     ```json
     {
       "source": ["ecommerce.order"],
       "detail-type": ["OrderPaid"],
       "detail": {
         "amount": [{ "numeric": [ ">", 500 ] }],
         "currency": ["USD"]
       }
     }
     ```

### 1.5 · In-Memory Ephemeral vs Durable Event Buses

| Feature Dimension | Ephemeral / In-Memory | Durable / Persistent |
|---|---|---|
| **Representative Tech** | Spring ApplicationEventMulticaster, Redis Pub/Sub, NATS Core | AWS EventBridge, NATS JetStream, Google Cloud Pub/Sub |
| **Storage Medium** | Process heap memory / Broker in-memory ring buffers | Distributed append-only WAL / Replicated storage |
| **Offline Durability** | Dropped immediately if consumers are offline (Fire-and-forget) | Retained (e.g. 24h to 7d), supporting catch-up pulls & DLQs |
| **Contract Governance** | In-code POJOs / Plain JSON | Schema Registry (Protobuf, JSON Schema, Avro) |

---

## 2 · Dispatcher & Fan-out Architectural Paradigm

### 2.1 · Engineering Nature & Twin Challenges of Fan-out
In event-driven architectures, **Fan-out** describes the mechanism where a single upstream event triggers processing across multiple downstream services or entities. In production-grade systems, naive broker broadcasting introduces two acute failure modes:
1. **Inter-Service Heterogeneity & Slow Consumer Head-of-Line Blocking**:
   If publishers broadcast synchronously or rely on a shared single-buffer broker, an outage or latency spike in a slow downstream consumer (e.g., audit logging or OLAP sync) blocks or backpressures the entire core transaction pipeline (e.g., billing or checkout).
2. **1-to-N Entity Explosion & Egress Amplification**:
   In user-facing workloads (e.g., global broadcast notifications, celebrity timeline fan-out, mass cache invalidation), a single root event instantaneously explodes into $10^5 \sim 10^7$ physical sub-tasks. If a single worker iterates over millions of targets in a monolithic loop, it triggers CPU saturation, OOM crashes, third-party provider rate limits (e.g., APNs/SMS HTTP 429), and severe head-of-line blocking.

### 2.2 · Two-Tier Fan-out Architecture

Production-grade event-driven platforms decouple fan-out into two distinct physical tiers:

```text
[Upstream Event: OrderPaid / BroadcastAlert]
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier-1: Service-Level Topology Fan-out (Topic-to-Queue)                     │
│                                                                             │
│                               [Topic / Exchange]                            │
│                                       │                                     │
│            ┌──────────────────────────┼──────────────────────────┐          │
│            ▼                          ▼                          ▼          │
│     [Queue: Billing]           [Queue: Fraud]             [Queue: Dispatcher]│
│            │                          │                          │          │
│            ▼                          ▼                          │          │
│     [Billing Service]          [Fraud Service]                   │          │
└──────────────────────────────────────────────────────────────────┼──────────┘
                                                                   │
                                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier-2: Workload-Level Dispatcher Fan-out (Sharded Slicing & Chunking)      │
│                                                                             │
│                        [Notification Dispatcher]                            │
│                 (1. Idempotency -> 2. Target Resolution & Micro-Batching)   │
│                                       │                                     │
│       ┌───────────────────────────────┼───────────────────────────────┐     │
│       ▼ (Chunk 1: 1~1,000)            ▼ (Chunk 2: 1,001~2,000)        ▼ ... │
│ [Priority Worker Queue 1]       [Priority Worker Queue 2]       [...]       │
│       │                               │                               │     │
│       ▼                               ▼                               ▼     │
│ [Delivery Worker Pool]         [Delivery Worker Pool]          [...]        │
│       │                               │                               │     │
│       ▼ (Token Bucket Limiter)        ▼ (Token Bucket Limiter)        ▼     │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │           Downstream Channels (APNs HTTP/2, FCM, SMS, Webhooks)           │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 · Tier-1: Service-Level Topology Fan-out (Topic-to-Dedicated-Queue)
1. **SNS-to-SQS Paradigm / Fan-out Exchange**:
   - The publisher emits a single domain event into a centralized Topic.
   - The messaging infrastructure (e.g., AWS SNS, RabbitMQ Fan-out Exchange) replicates the message payload across dedicated, persistent buffers (SQS / Queue) owned by each subscriber.
2. **Four Critical Architectural Guarantees**:
   - **Fault Isolation**: An outage in the analytics service accumulates lag exclusively in its private queue, leaving billing and fraud pipelines entirely unaffected.
   - **Elastic Peak Leveling**: Each subscriber consumes at its own prefetch concurrency, preventing slow consumers from starving the broker.
   - **Independent Retry & Dead Letter Queues (DLQ)**: Exponential backoff, retry caps, and DLQ routing are configured per business domain (e.g., billing retries 10 times; telemetry retries once).
   - **IAM & Security Decoupling**: Subscriber services only require read permissions for their specific queue.

### 2.4 · Tier-2: Workload-Level Dispatcher Fan-out (Dispatcher & Batch Slicing)
When an event must reach massive recipient sets (e.g., [[SystemDesign11 Notification System|Case 11 · Mobile Push Platform]], [[SystemDesign07 Photo Sharing Feed|Case 07 · Feed Fan-out on Write]]), systems employ a dedicated **Dispatcher** executing a four-stage pipeline:

1. **Stage 1: Global Idempotency Admission**:
   - The Dispatcher performs atomic CAS validation on the event or campaign key (e.g., `SETNX campaign:{id}:status IN_PROGRESS EX 86400` in Redis) to prevent duplicate fan-out loops from upstream re-deliveries.
2. **Stage 2: Target Resolution & Micro-Batch Slicing**:
   - **Cursor Streaming**: Target user IDs are streamed via keyset pagination or partition scanning from relational databases, social graphs (followers), or user profile segments.
   - **Uniform Micro-Batching (Chunking)**: Targets are partitioned into deterministic slices of 500 to 1,000 recipients:
     $$\text{Chunk}_k = \{ \text{user\_id}_i \}_{i=(k-1)B + 1}^{\min(kB, N)}, \quad B \in [500, 1000]$$
   - **Deterministic Chunk Key**: Assigns `chunk_id = hash(event_id, k)`, guaranteeing chunk-level retry idempotency.
   - **Early Preference Filtering**: Evaluates cached Redis bitsets or tables (Do Not Disturb schedules, opt-outs, unsubscribes) to prune invalid recipients before emitting network jobs.
3. **Stage 3: Queuing & Physical Priority Isolation**:
   - Micro-batch chunks are dispatched into separated downstream priority queues (`high-priority-queue` for transactional 2FA/OTPs; `bulk-queue` for marketing blasts).
4. **Stage 4: Parallel Worker Execution & Egress Rate Limiting**:
   - Delivery worker pools consume chunks concurrently.
   - **Egress Rate Limiting**: Distributed Redis Token Bucket limiters throttle traffic outbound to external vendor APIs (e.g., APNs HTTP/2 connection concurrency, FCM quotas, SMS carrier TPS limits).
   - **Partial Failure Containment**: If specific device tokens fail within a chunk, only the failed IDs are isolated into retry queues or unregister routines, avoiding full chunk replay.

### 2.5 · Core Engineering Trade-offs & Decision Matrix

| Dimension | In-Process Monolithic Loop | Topic-Only Multi-Subscriber | Dispatcher + Worker Slicing |
|---|---|---|---|
| **Max Fan-out Throughput** | Low (Single node, thread & socket limits) | Medium (Constrained by partition bandwidth) | **Virtually Infinite** (Horizontally scalable slicing) |
| **Fault Isolation** | None (Single crash drops remaining targets) | Moderate (Consumer lag risks memory pressure) | **Highest** (Chunk-level isolation, private DLQs) |
| **Egress Throttling** | Difficult across distributed nodes | Broker-level backpressure only | **Granular** (Token bucket aligned to downstream quotas) |
| **Retry & Idempotency** | Coarse; full re-run causes massive duplicates | Partition-level coarse replay | **Micro-batch / Individual-level idempotency** |

---

## 3 · Architectural Comparison Matrix (Stream vs Queue)

| Dimension | Event Bus / Commit Log Stream | Traditional Message Queue |
| :--- | :--- | :--- |
| **Storage Structure** | Append-only sequential binary log segments (`.log`) | Dynamic FIFO queue / Doubly-linked list / Ring buffer |
| **State Ownership** | **Consumer-side Stateless** (Broker only persists integer Offset cursors) | **Broker-side Stateful** (Per-message Ready / Unacked / Acked) |
| **Read Semantics** | **Non-Destructive Read** (Replayable from any historical offset) | **Destructive Read** (Permanently purged upon Ack) |
| **Consumption Model** | **Multi-Group Independent Fan-out** (Strict partition order) | **Competing Consumers** (Point-to-point single lease) |
| **Backlog Endurance** | **Extremely High** (Cold/hot log paging; TB-scale backlog has zero write impact) | **Poor** (Millions of backlog messages trigger RAM paging collapse) |

---

## 4 · Throughput & Latency Quantitative Ceilings

```text
┌────────────────────────────────────────────────────────────────────────┐
│                     Event Bus Throughput & Latency                     │
├─────────────────────────┬──────────────────────┬───────────────────────┤
│ Deployment Model        │ Typical QPS Baseline │ End-to-End Latency    │
├─────────────────────────┼──────────────────────┼───────────────────────┤
│ In-Process Bus (Disrupt)│ 1,000,000 - 10,000,000│ < 1 microsecond (μs) │
│ In-Memory Net (Redis PS)│ 50,000 - 100,000/node │ 0.5 - 2 milliseconds  │
│ High-Perf Cluster (NATS)│ 500,000 - 2,000,000  │ 1 - 5 milliseconds    │
│ Cloud Managed (EvBridge)│ 5,000 - 20,000       │ 20 - 100 milliseconds │
└─────────────────────────┴──────────────────────┴───────────────────────┘
```

- **In-Process Bus**: Uses lock-free ring buffers (e.g. LMAX Disruptor) achieving tens of millions of QPS with sub-microsecond latency, bounded by single-host memory and core count.
- **Redis Pub/Sub**: Single-threaded event loop broadcasting to connected subscriber sockets; QPS bounded by network packet processing overhead ($50\text{k} - 100\text{k}\text{ QPS}$); no message replay.
- **Cloud-Native Managed Bus (EventBridge)**: Evaluates complex JSON pattern matching, account permissions, and DLQs, operating comfortably at thousands of QPS with double-digit millisecond latency.

---

## 5 · Production Use Cases & Decision Matrix

### 4.1 · Ideal Scenarios
1. **Domain Event Fan-Out (Case 07 / Case 10 / Case 11)**:
   - When payment succeeds, order service publishes `OrderPaid`.
   - Inventory, Notification, Billing, Reward Points, and Fraud Detection independently receive and process the event in parallel without touching the core payment pipeline.
2. **Cluster-Wide Cache Invalidation**:
   - Admin updates global system configuration; the bus broadcasts `ConfigUpdated` to all running app nodes to flush local in-memory caches (Caffeine / Guava Cache).
3. **Webhook Ingestion & Microservice Routing**:
   - Consolidates incoming webhooks from external vendors (Stripe, GitHub) and routes them to appropriate microservices via declarative rules.

### 4.2 · Anti-Patterns & When NOT to Use
- **Competing Task Queue**: If multiple workers need to compete for individual compute-heavy tasks (e.g. video transcoding, PDF generation), Pub/Sub causes every worker to execute duplicate jobs. Use a **Message Queue**.
- **Historical Event Replay & Time Travel**: If you need multi-tenant historical log retention and arbitrary offset resets for ML training or analytics, use **Kafka / Partitioned Log**.

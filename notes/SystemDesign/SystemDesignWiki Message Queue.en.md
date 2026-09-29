# Wiki · Message Queue: Competing Consumers & Task Scheduling

Wiki Navigation: [[SystemDesign00 Overview|00 Blueprint & Numbers]] → [[SystemDesignWiki Message Queue|Wiki · Message Queue]]

A **Message Queue (MQ)** is the quintessential point-to-point asynchronous coordination primitive in distributed systems, designed for **load leveling, peak shaving, and preemptive task execution**.

```queue-vs-stream-visual
```

```async-messaging-architecture-visual
```

---

## 1 · Core Philosophy & Architectural Patterns

### 1.1 · Traditional Message Queue Architecture & Execution Model
Traditional message queues (e.g., RabbitMQ, ActiveMQ) adhere to the **Post Office / Work-Dispatch Model**. The underlying storage structure is fundamentally a **dynamic FIFO queue or doubly linked list in memory**, where the broker actively tracks the lifecycle state of each individual message.

```text
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      Message Queue Broker                   │
                    │                                                             │
                    │   [Exchange / Router]                                       │
[Producer] ──(Push)─┼─>   (Rule-based Routing)                                    │
                    │           │                                                 │
                    │           ▼                                                 │
                    │   ┌─────────────────────────────────────────────────────┐   │
                    │   │ In-Memory Queue (Linked List / Double-ended Buffer)  │   │
                    │   │                                                     │   │
                    │   │  [Msg 4]  ──>  [Msg 3]  ──>  [Msg 2]  ──>  [Msg 1]  │   │
                    │   │ (Ready)       (Ready)      (Unacked)     (Unacked)  │   │
                    │   └─────────────────────────────────────────────────────┘   │
                    │                     │                     │                 │
                    └─────────────────────┼─────────────────────┼─────────────────┘
                                          │ (Push/Prefetch)     │ (Push/Prefetch)
                                          ▼                     ▼
                                     [Worker 1]            [Worker 2]
                                         │                     │
                                         └───── ACK(Msg 1) ────┴──── ACK(Msg 2)
                                                 │
                                                 ▼
                                     (Broker purges confirmed message)
```

### 1.2 · Four Core Internal Mechanisms of Message Queues
1. **Fine-Grained Server-Side State Machine**:
   The broker actively maintains granular lifecycle states for **every single message** in memory:
   - `Ready`: Waiting in line to be dispatched;
   - `Unacked`: Leased to a worker, awaiting confirmation ACK;
   - `Acked`: Processing confirmed by worker;
   - `Dead-letter`: Retry threshold exceeded, relegated to DLQ.
2. **Competing Consumers (Point-to-Point Execution)**:
   Multiple workers attached to the same queue compete with each other (via work-stealing or round-robin prefetching). Each message is claimed and processed by **exactly one worker**.
3. **Destructive Read**:
   Once the broker receives an `Ack` from the worker, the message index in memory/disk is **immediately purged / marked for deletion**. The active backlog depth is expected to remain near zero under steady state.
4. **Lock Contention & Memory Paging Bottlenecks**:
   Every state transition requires internal locking and metadata updates. When workers fall behind and backlog climbs into millions, RAM exhaustion forces operating system Page Swapping to disk, precipitating severe throughput collapse.

### 1.3 · Message Leases, Visibility Timeout & Two-Phase Ack
To prevent job loss when a worker abruptly crashes, production message queues utilize visibility timeouts:

```text
[Producer] ──(1. Push Task)──> [Queue Storage]
                                      │
                               (2. Lease Message)
                                      ▼
                                [Worker Node]
                             (Task In-flight...)
                             ┌─────────────────┐
                             │ Success: Ack()  │ ──> (3a. Delete from Queue)
                             │ Crash / Timeout │ ──> (3b. Re-visible to other Workers)
                             └─────────────────┘
```

1. **Lease (In-Flight)**: When a worker fetches a message, it is not physically deleted. Instead, it enters an invisible `In-flight` state with a countdown timer (e.g. 30 seconds).
2. **Ack (Explicit Confirmation)**: Upon successfully completing all business side-effects, the worker calls `Ack(msg_id)`. The broker permanently purges the message.
3. **Nack / Timeout (Failover Recovery)**: If the worker dies or the timeout expires before an `Ack` is received, the message automatically returns to the visible state for another worker to claim.

### 1.4 · Dead Letter Queues (DLQ) & Poison Pill Isolation
When a corrupted message or edge-case bug causes unhandled exceptions across every worker, the item is known as a **Poison Pill**:
- Without isolation, the poison pill loops infinitely through workers, triggering continuous crashes and **Head-of-Line Blocking**.
- **DLQ Circuit Breaker**: The queue tracks a `ReceiveCount` metadata field. When `ReceiveCount > maxReceiveCount` (e.g. 5 retries), the broker moves the poison message into a **Dead Letter Queue (DLQ)** and sounds alarms for manual triage.

### 1.5 · Delay Queues & Priority Queues
- **Delayed / Scheduled Queues**: Using hierarchical timing wheels or min-heaps, messages remain hidden until a relative delay (e.g., 15 minutes) elapses. Widely used for automatic order cancellation.
- **Priority Queues**: Assigns strict urgency bands (e.g., Tier 1 to 10), allowing VIP high-priority tasks to jump ahead of massive background batch jobs.

---

## 2 · Architectural Comparison Matrix (Queue vs Stream)

| Dimension | Traditional Message Queue | Event Bus / Commit Log Stream |
| :--- | :--- | :--- |
| **Storage Structure** | Dynamic FIFO queue / Doubly-linked list / Ring buffer | Append-only sequential binary log segments (`.log`) |
| **State Ownership** | **Broker-side Stateful** (Per-message Ready / Unacked / Acked) | **Consumer-side Stateless** (Broker only persists integer Offset cursors) |
| **Read Semantics** | **Destructive Read** (Permanently purged upon Ack) | **Non-Destructive Read** (Replayable from any historical offset) |
| **Consumption Model** | **Competing Consumers** (Point-to-point single lease) | **Multi-Group Independent Fan-out** (Strict partition order) |
| **Backlog Endurance** | **Poor** (Millions of backlog messages trigger RAM paging collapse) | **Extremely High** (Cold/hot log paging; TB-scale backlog has zero write impact) |

---

## 3 · Quantitative Baselines: QPS & Latency

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   Message Queue Throughput & Latency                   │
├─────────────────────────┬──────────────────────┬───────────────────────┤
│ Component Type          │ Typical QPS Capacity │ End-to-End Latency    │
├─────────────────────────┼──────────────────────┼───────────────────────┤
│ RabbitMQ (AMQP)         │ 20,000 - 50,000 /node│ 1 - 5 ms (Low Jitter) │
│ AWS SQS Standard        │ Virtually Unlimited  │ 10 - 50 ms            │
│ AWS SQS FIFO            │ 300 - 30,000 (Batch) │ 20 - 60 ms (Ordered)  │
│ Redis List / BullMQ     │ 50,000 - 100,000/node│ < 1 ms (RAM Bound)    │
└─────────────────────────┴──────────────────────┴───────────────────────┘
```

- **RabbitMQ**: Built on Erlang's lightweight actor model; provides sub-5ms low latency and fine-grained routing; however, severe queue accumulation overflowing into disk paging degrades throughput rapidly.
- **AWS SQS**: Fully managed, horizontally distributed partition architecture. Scales automatically to hundreds of thousands of QPS; delivers at-least-once semantics with potential message reordering.
- **Redis List (`LPUSH` / `BRPOPLPUSH`)**: Single-threaded in-memory queues delivering sub-millisecond latencies; strictly bounded by single-instance RAM size.

---

## 4 · Architectural Decision: When to Choose Message Queue Over Event Bus?

Choosing a **Message Queue (e.g., RabbitMQ, AWS SQS, Celery)** instead of an **Event Bus (e.g., Kafka, Pulsar)** fundamentally boils down to **whether you are processing "Commands" or "Facts"**, and **whether you require fine-grained per-message lifecycle control**.

### 4.1 · Six Killer Scenarios Mandating a Message Queue

#### 1. Task Worker Pipelines & Work-Stealing
- **Scenarios**: Video transcoding, PDF generation, email/SMS dispatch, offline LLM batch inference.
- **Core Rationale**: These payloads represent imperative **"Commands"**. The primary invariant is **"let any idle worker claim the job immediately, and guarantee exactly one worker runs it"**.
- **Why NOT Event Bus**:
  - Event buses (like Kafka) process data sequentially within a partition. If a video transcoding task takes 30 minutes, that entire partition is **completely blocked**, starving all subsequent lightweight tasks behind it (**Head-of-Line Blocking**).
  - Message queues lease items independently; slow tasks never starve adjacent items.

#### 2. Native Message Priority Queuing
- **Scenarios**: Premium VIP enterprise customers jumping ahead of free-tier users in CI build queues.
- **Core Rationale**: Message queues (e.g., RabbitMQ Priority Queue) natively support per-message `priority` integer weights (1-255). The broker's internal heap dispatches higher-priority tasks first.
- **Why NOT Event Bus**:
  - Kafka append-only logs cannot alter their physical write order.
  - Prioritizing in Kafka requires creating multiple distinct topics (`topic_vip`, `topic_normal`) and writing complex custom polling logic in consumer application code.

#### 3. Delayed Messages & Scheduled Triggers
- **Scenarios**: 30-minute unpaid e-commerce order cancellation, coupon expiration reminders, and **exponential backoff retries** (10s -> 1m -> 5m).
- **Core Rationale**: Traditional MQs natively support hiding individual messages until a future relative/absolute timestamp elapses (e.g., AWS SQS `DelaySeconds`, RabbitMQ TTL).
- **Why NOT Event Bus**:
  - Kafka cannot natively "hide" a message mid-log.
  - Emulating delay in Kafka requires external timing wheels or dozens of bucketed intermediate topics, introducing severe architectural overhead.

#### 4. Fine-Grained Per-Message Retry & Dead-Letter Queues (DLQ)
- **Scenarios**: Payment webhook callbacks and flaky third-party API isolation.
- **Core Rationale**: If message #100 fails due to an external network timeout:
  - **In a Message Queue**: The worker issues a `NACK`; the item is re-queued or routed to a DLQ, and the worker immediately processes message #101.
  - **In an Event Bus**: Kafka consumers track progress via monotonic `Offset`s. If message #100 fails and data loss is disallowed, the consumer's offset **cannot advance**, stalling the entire partition.

#### 5. Highly Dynamic Consumer Worker Autoscaling
- **Scenarios**: Microservice pods scaling from 10 to 500 instances in 60 seconds during traffic flash sales.
- **Core Rationale**:
  - **Kafka's Hard Ceiling**: Maximum active consumer concurrency is strictly **bounded by the partition count**. If a topic has 32 partitions, deploying 100 pods leaves 68 pods completely idle and starving.
  - **Message Queue Elasticity**: Hundreds or thousands of workers can bind to a single queue concurrently, decoupling worker auto-scaling from storage topology.

#### 6. Complex Content-Based Routing & Exchange Topologies
- **Scenarios**: Enterprise system integration requiring dynamic routing based on message headers and multi-tenant rules.
- **Core Rationale**: RabbitMQ provides rich **Exchange Topologies** (Direct, Topic, Headers, Fanout). Producers simply publish to an exchange; pattern bindings (e.g., `order.us.*`) route messages internally. Kafka requires producers to explicitly compute and hardcode target topics.

---

### 4.2 · Architectural Decision Tree

Use this decision path to quickly evaluate architectural choices:

```text
                  Require durable retention with multi-consumer replay?
                                    │
                    ┌───────────────┴───────────────┐
                   YES                              NO
                    │                               │
             【Use Event Bus】                 Individual message duration variable/long?
            (Kafka, Pulsar)                         │
                                            ┌───────┴───────┐
                                           YES              NO
                                            │               │
                                     【Use Message Queue】 Require > 100k QPS extreme throughput?
                                    (RabbitMQ, SQS)         │
                                                    ┌───────┴───────┐
                                                   YES              NO
                                                    │               │
                                             【Use Event Bus】 【Use Message Queue】
```

---

### 4.3 · Division of Labor in Modern CI/CD Platforms

In production CI/CD architectures (e.g., GitHub Actions, GitLab CI), the two systems operate in tandem:

1. **Where to Use Event Bus (Kafka)**:
   - **`Controller -> Push Service -> UI Dashboard`**:
   - Status transitions (`JobStarted`, `JobFinished`) are immutable **Facts**.
   - These events fan out concurrently to WebSocket gateways, audit logs, and Prometheus metrics collectors without latency crosstalk.
2. **Where to Use Message Queue (or DB + Pull + Lease Pool)**:
   - **`Controller / Gateway -> Runner Platform`**:
   - Dispatching compilation, testing, and container build commands. Each item is an imperative **Command**.
   - Job durations vary drastically (5 seconds for a linter vs 45 minutes for integration tests).
   - Requires machine-affinity matching (GPU vs macOS), priority preemption, and visibility timeout failovers. Using a **Competing Consumer Message Queue** prevents long jobs from blocking partitions.

---

### 4.4 · Anti-Patterns & When NOT to Use
- **Multi-Service Data Sharing**: Once a message is acknowledged, it is destroyed. Separate downstream systems cannot receive independent copies. Use an **Event Bus** or **Kafka**.
- **Historical Event Replay**: Message queues cannot reset offsets to replay old tasks.

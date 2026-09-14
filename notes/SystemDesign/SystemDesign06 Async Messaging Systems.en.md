# System Design 06 · Message Queue

Course location: [[SystemDesign04 Storage Systems|04 Storage]] → this note → [[SystemDesign09 Consistent Hashing|09 Consistent Hashing]]

Asynchronous architecture rearranges the timing of responsibility transfer. It does not guarantee that the function itself runs faster. The core question is: if the current process crashes immediately, at what moment can the system be certain the work is durably accepted?

```async-messaging-architecture-visual
```

## 1 · High-Level API and Two Acknowledgments

The high-level API of a messaging system typically revolves around delivery and consumption:
- **Produce**: Write a message into the system.
- **Consume**: Retrieve a message from the system.
- **Ack (Acknowledge)**: Confirm the message was processed successfully.
- **Nack / Requeue**: Indicate processing failure and request a retry.
- **Seek / Commit Offset**: Locate or commit the current consumption progress.

There are two distinct acknowledgments in the system:

```text
Client -> API: Initiate request
API -> Queue: Enqueue task
Queue -> API: Broker confirmation (Durable Ack)
API -> Client: 202 Accepted and return job_id

Queue -> Worker: Dispatch task
Worker -> DB: Execute business write
Worker -> Queue: Consumer confirmation (Worker Ack)
```

1. **Broker Ack**: The broker confirms to the API after receiving and persisting the message. This marks the system officially taking over the task. If the API returns success to the client before this, it's Fire-and-Hope, not Fire-and-Forget.
2. **Consumer Ack**: The consumer confirms to the broker after completing the business side effects (e.g., writing to a DB), indicating the message can be deleted or the offset advanced.

## 2 · Topology Patterns and Semantics

When analyzing asynchronous scenarios, it is necessary to distinguish between connection topology and delivery semantics:

```text
1 to 1 (Point to Point):   Sender -> Queue -> One logical receiver
1 to N (Fan-out):          Publisher -> Topic -> Subscriber A / B / C
N to 1 (Fan-in):           Producers -> Collector -> Stream processor
N to N (Event Backbone):   Producers -> Event Bus/Log -> Consumer groups
```

Even if the topology is 1-to-1, the underlying semantics can be completely different: it might specify a unique receiver, let any worker compete for execution, or be a two-way Request/Reply pattern.

### Queue, Pub/Sub, and Log Semantics Comparison

| Model | Characteristic | Consumption Mechanism |
|---|---|---|
| **Queue** | Competing Consumers. A message is successfully processed by only one consumer. | Multiple workers compete for messages in the same queue. |
| **Pub/Sub** | Fan-out. Each logical subscriber gets a full copy of the message. | Subscribers fetch their own copies, and workers within a subscriber compete. |
| **Partitioned Log** | Append-only. Retains history independently of consumption progress, supporting replay. | Systems like Kafka use Consumer Groups to determine semantics: same group acts as a Queue, different groups act as Pub/Sub. |

A Topic is merely a name or logical grouping; it does not directly dictate the underlying distribution semantics.

## 3 · Kafka as a Partitioned Log

Kafka did not adopt the traditional Queue model. Instead, its core abstraction is an append-only log that is retainable, locatable, and replayable (Partitioned Log).

- **Topic, Partition, and Offset**: A producer routes a message to a specific Partition based on its Key. Order is guaranteed within a Partition. Consumers track their own Offset. Message retention is independent of the consumption progress.
- **Parallelism Limit**: In a traditional Consumer Group, a single Partition can only be exclusively consumed by one consumer instance at a time. Thus, the maximum effective parallelism ≤ the number of Partitions.
- **Replication and Fault Tolerance**: Each Partition has a Leader and multiple Replicas. The `acks` parameter combined with the `min.insync.replicas` configuration determines the quorum needed for a safe write.
- **Strengths**: Highly suitable for cross-system data pipelines (CDC), history replay, multi-tenant stream analytics, and scenarios requiring local ordering by Key.
- **Weaknesses and Traps**: Fine-grained per-task acknowledgments and complex header-based routing rules are not its strengths. A Poison Record (e.g., one that fails deserialization perpetually) can completely stall a consumer strictly adhering to offset advancement.

## 4 · Delivery, Idempotency, and Ordering

### Delivery Semantics
- **At-most-once**: Advance the offset first, then process. If a crash occurs during processing, the message is lost.
- **At-least-once**: Ack after successful processing. If the Ack fails due to network partition, the system resends, causing duplication.
- **Exactly-once effect**: True exactly-once requires encompassing the boundaries of external systems. Kafka's internal transactions cannot cover external API calls. This relies on the consumer's idempotency regarding external operations.

### Idempotency
An idempotency key should be a unique business identifier (e.g., a combination of `order_id` and `status`), never just a hash of the payload body. Otherwise, if a user initiates two independent orders for the same item and amount, deduplication by hash would erroneously discard the second order. Table shape and in-flight duplicates: [[SystemDesign01 Stateless Service|01]].

### Local Ordering
Global ordering forces all traffic through a single processing node, sacrificing concurrency. In practice, systems settle for **local ordering**. By using the same `partition_key` (e.g., `user_id`), all operations for the same user land in the same Partition and are processed sequentially.

### Dead-Letter Queue (DLQ)
Infinite retries lead to a Retry Storm. Messages encountering permanent failures (e.g., data validation errors) should be routed to a Dead-Letter Queue (DLQ) preserving the original `event_id`, failure reason, and stack trace for subsequent manual inspection and replay.

## 5 · Outbox Pattern and Event Bus Paradigms

### 5.1 What is the Transactional Outbox Pattern?

#### The Dual-Write Problem
In microservices and distributed architectures, atomically persisting business domain state (writing to a database) and publishing an asynchronous event (sending to a message broker) cannot be achieved without heavyweight distributed transactions (such as 2PC/XA):

```text
Approach A (Write DB first, then publish to MQ):
1. Local DB transaction commits: UPDATE orders SET status = 'PAID'
2. Invoke Broker API: publish(OrderPaidEvent)
--> Failure Window: If step 2 fails due to network partition, broker downtime, or application process crash, the DB change is permanently committed while downstream systems never receive the event, causing irrecoverable state drift.

Approach B (Publish to MQ first, then write DB):
1. Invoke Broker API: publish(OrderPaidEvent)
2. Local DB transaction commits: UPDATE orders SET status = 'PAID'
--> Failure Window: If step 2 fails due to unique constraint violations, deadlock aborts, or timeouts, downstream consumers process a "phantom event" that was never actually committed in the source of truth.
```

Distributed two-phase commit (2PC) protocols introduce high blocking latencies, single-point coordinator bottlenecks, and are rarely supported by modern cloud message brokers.

#### Core Mechanism of Transactional Outbox
The **Transactional Outbox Pattern** eliminates dual writes by demoting the cross-network event emission into a single local ACID transaction. Domain state mutations and the outbound event payload are committed together in one atomic database operation:

```sql
BEGIN;

-- 1. Mutate domain state
UPDATE orders 
SET status = 'PAID', updated_at = now() 
WHERE order_id = :order_id AND status = 'PENDING';

-- 2. Insert the corresponding event into the Outbox table within the same transaction
INSERT INTO outbox_events (
    event_id, 
    aggregate_type, 
    aggregate_id, 
    event_type, 
    payload, 
    created_at
) VALUES (
    :event_id, 
    'order', 
    :order_id, 
    'order.paid', 
    :payload_json, 
    now()
);

COMMIT;
```

Relying on the database's local Atomicity and Durability, the domain mutation and event record either both persist or both roll back, completely eliminating the distributed inconsistency window.

#### Outbox Relay Mechanisms
Once persisted in the Outbox table, an external relay component transfers the event to downstream consumers or the message broker. Two primary relay mechanisms are used in production:

1. **Polling Publisher**:
   A background worker periodically scans the Outbox table, claims unhandled rows, publishes them to the broker, and marks them completed or deletes them upon acknowledgment.
   - *Pros*: Simple SQL implementation, easy to inspect and debug, no external middleware dependencies.
   - *Cons*: High-frequency polling places extra read load on the database; poll intervals add latency; large batches cause index contention.
2. **Log-based Change Data Capture (CDC)**:
   A dedicated CDC engine (such as Debezium's Outbox Event Router) directly tails the database transaction write-ahead log (e.g., MySQL Binlog or PostgreSQL WAL), extracts rows appended to `outbox_events`, and streams them into Kafka.
   - *Pros*: Zero polling SQL overhead, sub-second latency, completely decoupled from application runtime, zero read locks on the primary database.
   - *Cons*: Requires provisioning and operating Kafka Connect / Debezium infrastructure; schema migrations require operational care.

**Delivery Guarantees**: Regardless of whether Polling or CDC is used, network timeouts and worker restarts inevitably cause duplicate transmissions. The Outbox pattern provides **At-Least-Once** delivery. **Downstream consumers must implement strict idempotency using `event_id` or unique business keys.**

---

### 5.2 Event Bus Paradigms: DB as Bus vs. Queue / Log as Bus

Once events are reliably generated via the Outbox pattern, two classic paradigms govern the event distribution bus:

| Paradigm | Design & Use Case | Trade-offs & Operational Costs |
|---|---|---|
| **DB as bus** | Insert Outbox records within business transactions; consumers maintain durable cursors or claim tasks in short transactions with leases, executing external I/O only after commit | Requires vacuum/cleanup of history, expired lease indexing, and cursor tracking; multiple logical subscribers need independent cursors / task rows and cannot share a single status flag |
| **Queue / log as bus** | Outbox relay / CDC publishes event_id to Kafka; Router consumes, matches subscriptions, and generates Deliveries; raw log retained for disaster recovery and replay | Introduces broker clusters, replication, and consumer lag monitoring; relay may duplicate emissions, requiring idempotent Routers and consumers |

---

### 5.3 Deep Dive: DB as Bus (Database as the Event Backbone)

For monolithic architectures, low-to-medium throughput services, or teams wishing to avoid the operational overhead of running a dedicated Kafka cluster, using the relational database directly as the event bus and dispatch engine is an effective choice.

#### Core Implementation Mechanisms

1. **Job Lease Pattern via SKIP LOCKED**:
   Multiple competing consumers claim ready tasks using short transactions and explicit row-level locking:
   ```sql
   BEGIN;

   -- 1. Claim a batch of ready tasks, using SKIP LOCKED to avoid lock contention between consumers
   SELECT event_id, payload 
   FROM outbox_events 
   WHERE status = 'READY' AND available_at <= now()
   ORDER BY available_at, event_id
   FOR UPDATE SKIP LOCKED 
   LIMIT 10;

   -- 2. Mark claimed rows as running and set a lease expiration
   UPDATE outbox_events 
   SET status = 'RUNNING', 
       lease_expires_at = now() + INTERVAL '30 seconds', 
       attempts = attempts + 1 
   WHERE event_id = ANY(:claimed_ids);

   COMMIT;
   ```
   > **Critical Engineering Rule: Execute external I/O only AFTER the transaction commits.**
   > Never make external HTTP requests, RPC calls, or lengthy computations while holding an open database transaction. A slow downstream response will hold row locks indefinitely, exhaust the connection pool, and trigger cascading failure across the database. The correct workflow is: commit the lease transaction immediately, execute external I/O outside the transaction, and open a separate short transaction to mark `status = 'COMPLETED'`. If a worker crashes, the lease expires and other workers safely reclaim the task.

2. **Durable Cursor Stream**:
   If events are append-only without in-place updates or deletions, consumers maintain an independent cursor table tracking their highest processed `event_id` (auto-incrementing integer):
   ```sql
   SELECT event_id, event_type, payload 
   FROM outbox_events 
   WHERE event_id > :last_read_id 
   ORDER BY event_id ASC 
   LIMIT 100;
   ```

#### Inherent Costs and Bottlenecks

- **Table Bloat and Garbage Collection**: Relational database storage engines are not optimized as queue queues. Frequent `UPDATE` and `DELETE` operations generate dead tuples (e.g., PostgreSQL MVCC churn), inducing table bloat, index fragmentation, and heavy autovacuum overhead. Systems must implement partition pruning (dropping daily partition tables) or batch archival jobs.
- **Multi-Subscriber Fan-out Complexity**: In traditional message brokers, a single message is naturally fanned out across multiple consumer groups. In a DB-as-bus design, a single row cannot track the independent progress of 5 distinct subscribers via a single `status` column. The architecture must either maintain separate cursor records per subscriber (for append-only streams) or duplicate task rows per subscriber upon write (write amplification).
- **Throughput Ceiling**: Constrained by ACID transaction coordination and connection pools, DB as bus is typically suited for workloads with QPS < 1,000–2,000. When throughput exceeds this scale, the bus should transition to a dedicated broker.

---

### 5.4 Deep Dive: Queue / Log as Bus (Partitioned Log as the Event Backbone)

When microservices proliferate, throughput reaches tens of thousands of QPS, or multi-subscriber fan-out and long-term history replay are required, a partitioned append-only log (such as Kafka) serves as the event backbone.

#### Core Implementation Mechanisms

```text
[Business Service] 
    │  (Local ACID Transaction)
    ├──► Domain Tables (orders)
    └──► Outbox Table (outbox_events)
            │
            ▼ (CDC tails Binlog / WAL)
     [Debezium / Relay]
            │
            ▼ (At-Least-Once Delivery)
    [Kafka Partitioned Log]  (Topic: events.orders, Key: order_id)
            │
            ▼ (Batched Stream Consumption)
     [Event Router] 
            │
            ├──► Evaluates subscription filters
            ├──► Generates durable Delivery records
            └──► Dispatches to downstream services / Webhook Workers
```

1. **CDC-Driven Pipelined Decoupling**:
   The Outbox table serves solely as an ephemeral staging buffer. The CDC engine extracts events in sub-second intervals and streams them into Kafka partitioned by `aggregate_id`. Application database connections are completely insulated from downstream consumer latency.
2. **Decoupled Router and Delivery State Machines**:
   A dedicated Event Router service consumes from Kafka, evaluates tenant subscription rules (e.g., `event_type == 'order.paid' && amount > 1000`), creates isolated Delivery items per subscriber, and forwards them to execution queues or webhook dispatchers.
3. **Log Replayability**:
   Kafka retains raw event logs independently of consumer progress (e.g., 7 days retention). When a new microservice is deployed and needs historical backfills, or when a downstream bug requires reprocessing historical data, consumers can simply `Seek(Offset)` back to an earlier timestamp without putting any query load on the upstream primary database.

#### Inherent Costs and Operational Trade-offs

- **Broker Operational Overhead**: Operating Kafka or RocketMQ requires maintaining broker clusters, controller consensus (KRaft / ZooKeeper), partition sizing, rebalances, disk watermarks, and multi-replica durability configurations (`acks=all`, `min.insync.replicas=2`).
- **Consumer Lag Monitoring**: Systems must monitor `Consumer Lag` and `Oldest unacked age` to detect processing bottlenecks and trigger auto-scaling before queue buffers saturate.
- **End-to-End Idempotency Requirement**: Network retries across CDC relays and routers produce duplicate messages. Downstream consumers must implement idempotency (e.g., database unique constraints `ON CONFLICT DO NOTHING`, distributed deduplication locks, or state machine condition checks `WHERE status = 'PREV'`).

## 6 · How to Choose

| System | Use Case | Core Mechanism |
|---|---|---|
| **RabbitMQ** | Fine-grained task processing, complex routing | Exchange routing, consume-and-delete |
| **Kafka** | High-volume data pipelines, replay, stream processing | Partitioned Log, Offset tracking, sequential I/O |
| **Managed Queue** | Cloud-native default choice | API driven, zero-ops scaling (SQS / PubSub) |
| **DB Table** | Monoliths or small-scale services | Local transaction guarantee, `SKIP LOCKED` |

Decision matrix (6-line text block):

```text
1. Do you need strict history replay or stream analytics? -> Kafka
2. Do you need complex routing and per-message ack for tasks? -> RabbitMQ
3. Can you just use a cloud managed API? -> SQS / PubSub
4. Is your scale small? -> Database table queue (SKIP LOCKED)
```

## 7 · Event Bus and Webhook

An **Event Bus** is a general term for a many-to-many router. It encompasses ingestion, match filtering, and durable delivery, with separated control and data planes. It is usually built on top of a DB, RabbitMQ, or Kafka.

A **Webhook** is a common HTTP target endpoint supported by this router. For security and isolation, it requires HMAC signature verification and separate rate-limiting and retry backoff strategies per tenant.

## 8 · Observability and Primary Sources

**Key Metrics**:
- **Oldest unacked age**: Reflects system lag much better than simply looking at CPU utilization.
- **Consumer lag** and **DLQ rate**.

Primary Sources:
- [RabbitMQ: Consumer Acknowledgements and Publisher Confirms](https://www.rabbitmq.com/docs/confirms)
- [RabbitMQ: Quorum Queues](https://www.rabbitmq.com/docs/quorum-queues)
- [RabbitMQ: Native AMQP 1.0 and AMQP history](https://www.rabbitmq.com/blog/2024/08/05/native-amqp)
- [OASIS AMQP 1.0 Standard](https://www.oasis-open.org/standard/amqp/)
- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Kafka: a Distributed Messaging System for Log Processing, NetDB 2011](https://www.odbms.org/2011/01/kafka-a-distributed-messaging-system-for-log-processing/)
- [Debezium Outbox Event Router](https://debezium.io/documentation/reference/stable/transformations/outbox-event-router.html)
- [PostgreSQL SELECT: SKIP LOCKED](https://www.postgresql.org/docs/current/sql-select.html)
- [Amazon EventBridge: Event buses](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-event-bus.html)
- [GitHub webhook best practices](https://docs.github.com/en/webhooks/using-webhooks/best-practices-for-using-webhooks)
- [Stripe webhook best practices](https://docs.stripe.com/webhooks)

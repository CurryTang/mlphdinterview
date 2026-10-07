# Wiki · Pub/Sub + Transactional Outbox Pattern

Category: Wiki Pattern Library · Knowledge Point  
Related Patterns: [[SystemDesignWiki Event Bus|Event Bus]] · [[SystemDesignWiki Kafka|Kafka Deep Dive]] · [[SystemDesignWiki Message Queue|Message Queue]] · [[SystemDesignWiki Idempotency|Idempotency Guarantees]] · [[SystemDesign11 Notification System|Notification System Deep Dive]]

---

## 1 · Core Problem: Dual-Write Hazard & Distributed Coordination Failure

In microservices and event-driven architectures (EDA), domain state mutations frequently trigger cross-system event distribution requirements (e.g., after an Order Service confirms a payment, it must notify inventory, shipping, invoicing, and notification services).

Because application services cannot establish a single global atomic transaction between a "local database" and an "external event bus (Pub/Sub Broker)", naive dual-write approaches inevitably encounter timing failure windows:

```text
[Failure Scenario A: Write DB First, Then Publish to Pub/Sub]
  1. BEGIN TX -> UPDATE orders SET status = 'PAID' -> COMMIT TX (Success)
  2. Network jitter / Pub/Sub Broker timeout / Process OOM Crash occurs
  3. Pub/Sub event is lost permanently (Silent Event Loss)
  => Database reflects payment, but downstream systems never receive notifications.
     The system state suffers irreversible divergence.

[Failure Scenario B: Publish to Pub/Sub First, Then Write DB]
  1. Publish Event('OrderPaid') to Pub/Sub Broker (Success, queued in Topic)
  2. Downstream consumers immediately receive the event and initiate fulfillment
  3. Local DB transaction commit fails (Unique key collision / Deadlock / Crash)
  4. Local DB rolls back
  => Downstream consumers processed a "Phantom Event" that never occurred.
```

### 1.1 · Why 2PC / XA Distributed Transactions Fail in Modern Pub/Sub Architectures
1. **Lack of Protocol Support**: Modern high-throughput message brokers (Kafka, AWS SNS, Pulsar, RabbitMQ) do not natively act as XA-compliant Resource Managers (RMs) and cannot coordinate with database transaction managers.
2. **Throughput Collapse**: Two-Phase Commit is a blocking protocol. Between Phase 1 (Prepare) and Phase 2 (Commit), the database must retain exclusive row locks (X-Locks). Any network round-trip time (RTT) jitter magnifies lock contention duration, causing system concurrency TPS to plunge by over 90% and precipitating cascading connection pool exhaustion.

---

## 2 · Architectural Topology: End-to-End Pub/Sub + Outbox Pipeline

The core philosophy of the **Pub/Sub + Transactional Outbox Pattern** is: **reduce cross-system distributed coordination to an internal ACID local transaction within a single database, and subsequently drive Pub/Sub broadcast fan-out and downstream decoupling via an asynchronous pipeline.**

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Producer Service (e.g., Order Service)                                 │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ Local ACID Transaction Boundary                                    │ │
│ │   1. INSERT INTO orders (order_id, user_id, amount, status, ...)   │ │
│ │   2. INSERT INTO outbox_events (id, aggregate_id, topic, payload)  │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Commit (Atomic Persistence)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Database (MySQL / PostgreSQL / DynamoDB)                               │
│ ┌───────────────────────────────┐    ┌───────────────────────────────┐ │
│ │ Business Table (orders)       │    │ Outbox Table (outbox_events)  │ │
│ └───────────────────────────────┘    └───────────────────────────────┘ │
│                     ▲                                                  │
│                     └────── Binlog / WAL Sequential Append Write       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Tailing (CDC / Transaction Log Stream)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Message Relay / CDC Engine (Debezium / Kafka Connect)                  │
│ - Reads committed changes from outbox_events via commit logs           │
│ - Enforces At-Least-Once Delivery                                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Publish to Topic
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Pub/Sub Broker / Event Bus (Kafka / AWS SNS / Pulsar)                  │
│ Topic: order.events (Partition Key: order_id)                          │
└───────────────┬───────────────────────────────────────┬────────────────┘
                │ 1-to-N Broadcast Fan-out              │
                ▼                                       ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│ Subscription Queue A          │       │ Subscription Queue B          │
│ (SQS / Kafka Group: Billing)  │       │ (SQS / Kafka Group: Inventory)│
└───────────────┬───────────────┘       └───────────────┬───────────────┘
                │ Pull / Push                           │ Pull / Push
                ▼                                       ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│ Consumer Service A (Billing)  │       │ Consumer Service B (Inventory)│
│ ┌───────────────────────────┐ │       │ ┌───────────────────────────┐ │
│ │ Local ACID (Inbox Dedup): │ │       │ │ Local ACID (Inbox Dedup): │ │
│ │ 1. INSERT INTO inbox(...) │ │       │ │ 1. INSERT INTO inbox(...) │ │
│ │ 2. UPDATE billing_records │ │       │ │ 2. DEDUCT inventory_stock │ │
│ └───────────────────────────┘ │       │ └───────────────────────────┘ │
└───────────────────────────────┘       └───────────────────────────────┘
```

### 2.1 · Why Combine Pub/Sub with Outbox?
* **Outbox Solves Reliable Egress from a Single Producer**: Outbox alone lacks distribution topology; without Pub/Sub, the producer must write point-to-point RPC or point-to-point enqueue logic for each consumer. This creates tight coupling where adding new consumers requires modifying upstream relay code.
* **Pub/Sub Solves 1-to-N Fan-Out and Consumer Isolation**: Publishing Outbox events to a Pub/Sub topic/exchange allows the broker to manage dynamic routing and fan-out. Each downstream consumer manages an independent subscription queue with decoupled consumption rates, providing complete spatial and temporal decoupling.

---

## 3 · Message Relay Mechanisms & Trade-offs (Polling vs CDC)

The component transferring records from `outbox_events` to the Pub/Sub Broker is the **Message Relay**. Production systems rely on two distinct patterns:

| Dimension | Pattern A: Polling Publisher | Pattern B: Transaction Log Tailing (CDC) |
|---|---|---|
| **Mechanism** | Scheduled worker executes SQL:<br>`SELECT * FROM outbox_events WHERE status = 'PENDING' ORDER BY id LIMIT 100 FOR UPDATE SKIP LOCKED`, updates status or purges rows upon ack. | CDC engine (Debezium, Canal, DynamoDB Streams) tails storage commit logs (MySQL Binlog, PostgreSQL WAL) and streams parsed rows to Pub/Sub. |
| **Primary DB Overhead** | **High**. Continuous table/index scans, row-level locks, and undo log generation compete with production queries for connection pools and CPU. | **Minimal**. Streams sequential disk logs or replication feeds without submitting SQL queries or taking application locks. |
| **End-to-End Latency** | **Seconds (1s - 5s)**. Constrained by polling sleep intervals required to prevent database thrashing. | **Sub-second (10ms - 100ms)**. WAL generation is synchronous with transaction commit, streaming events near real-time. |
| **Infra Dependencies** | **Zero external dependencies**. Requires only internal application scheduler threads; database-agnostic. | **Moderate to High**. Requires a dedicated CDC cluster (Kafka Connect / Debezium) and database-level replication privileges. |
| **Production Guideline** | Small-scale systems, low-frequency MVPs, or background batch processors. | **The industrial standard for high-throughput, microservice architectures**. |

---

## 4 · Fan-out Topology: The Topic-to-Queue Pattern (SNS-to-SQS Paradigm)

In distributed messaging architectures, a strict distinction must be drawn between **Topics (broadcast routing logic)** and **Queues (buffering consumption physical carriers)**:

```text
                ┌───────────────────────────────────┐
                │ Pub/Sub Topic (AWS SNS / Rabbit Ex)│
                └─────────────────┬─────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │ Fan-out                │ Fan-out                │ Fan-out
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Queue A (Bill)   │     │ Queue B (Ship)   │     │ Queue C (Audit)  │
│ - Buffer: 100k   │     │ - Buffer: 500    │     │ - Buffer: 10k    │
│ - Rate: 200/s    │     │ - Rate: 5,000/s  │     │ - Rate: 50/s     │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Billing Worker   │     │ Shipping Worker  │     │ Audit Worker     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

1. **Buffering & Backpressure**:
   - Direct push broadcasts to consumer HTTP/RPC endpoints fail under load spikes, leading to dropped messages or cascaded outages.
   - Attaching dedicated durable queues between the Topic and each Consumer (e.g., SQS, RabbitMQ bound queues, Kafka Consumer Groups) buffers transient spikes. Downstream workers pull messages at their own sustainable rate (Pull-based backpressure).
2. **Failure Blast Radius Isolation**:
   - A crash in the billing service causes its dedicated queue to buffer messages without stalling the shipping service; a slow audit consumer will never degrade real-time logistics.
3. **Independent Retries & DLQs**:
   - Each queue configures independent visibility timeouts, maximum retry thresholds, and dead letter queues.

---

## 5 · Symmetric Design: Consumer-Side Transactional Inbox Pattern

The Outbox pattern guarantees only **At-Least-Once Delivery**. Network retries, relay failovers, and dropped acknowledgments guarantee that consumers **will receive duplicate events**.

To achieve **Effectively-Once Processing**, downstream consumers must implement the symmetric **Transactional Inbox** pattern:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Consumer Service Worker                                                │
│                                                                        │
│   Event: { message_id: "evt_1001", order_id: "ord_99", amount: 100 }  │
│                                                                        │
│   BEGIN TRANSACTION;                                                   │
│     -- 1. Deduplicate via Unique Constraint                            │
│     INSERT INTO inbox_records (message_id, handler_name, processed_at) │
│     VALUES ('evt_1001', 'billing_handler', NOW());                     │
│                                                                        │
│     -- If Unique Constraint Violation occurs:                          │
│     -- Immediately ROLLBACK business work and send ACK to Broker!       │
│                                                                        │
│     -- 2. Execute Actual Business State Mutation                       │
│     UPDATE user_balance SET balance = balance - 100                    │
│     WHERE user_id = 'usr_42';                                          │
│   COMMIT;                                                              │
│                                                                        │
│   ACK to Queue (Advance consumer cursor)                               │
└────────────────────────────────────────────────────────────────────────┘
```

* **Atomic Deduplication**: By colocating the inbox record insertion and business mutation within the same local ACID transaction, re-delivered duplicate messages trigger a unique key constraint violation, rolling back the business mutation and preventing duplicate charges.
* **Idempotent Acknowledgment**: When a transaction rolls back due to an existing inbox key, the consumer **must still acknowledge (ACK) the message to the queue**. Omitting the ACK causes the message to be continually redelivered, creating head-of-line blocking.

---

## 6 · Event Design & Partition Ordering Guarantees

### 6.1 · Thin Notification vs Event-Carried State Transfer (ECST)

| Pattern | Payload Contents | Advantages | Vulnerabilities & Trade-offs |
|---|---|---|---|
| **Thin Notification** | Minimal metadata:<br>`{ "event_id": "...", "order_id": "123", "action": "UPDATED" }` | 1. Lightweight message size.<br>2. Clear architectural boundaries without leaking domain models. | **Read Storm & Race Conditions**: Multiple consumers concurrently query the upstream primary database via RPC ($N\times$ query amplification). Queries may read "future dirty state" written after the event. |
| **Event-Carried State Transfer (ECST)** | Complete entity snapshot:<br>`{ "order_id": "123", "status": "PAID", "items": [...], "version": 4 }` | **Zero Downstream RPC Callbacks**: Consumers update internal projections solely using the event body, severing runtime network coupling with the producer. | Increased payload size. Frequent schema evolution requires strict backward/forward compatibility governance (Avro / Protobuf + Schema Registry). |

> **Production Recommendation**: High-throughput and high-fanout distributed systems strongly favor **ECST**, trading message size for runtime service isolation and protecting the primary database from read storms.

### 6.2 · Causal Ordering via Partition Keys
When an entity undergoes sequential state transitions (`OrderCreated` $\to$ `OrderPaid` $\to$ `OrderCancelled`), consumers must process these events in strict causal sequence.
* **Partition Key Binding**: The relay must assign the `aggregate_id` (e.g. `order_id`) as the Kafka / Pulsar Partition Key.
* **Mathematical Invariant**:
  $$\text{Partition} = \text{MurmurHash2}(\text{aggregate\_id}) \pmod{\text{NumPartitions}}$$
  All lifecycle events for a specific order route deterministically to the same physical partition, consumed sequentially by a single worker thread and eliminating out-of-order race conditions.

---

## 7 · Resilience Engineering: Poison Pills, Backoff & Dead Letter Queues (DLQ)

```text
[Incoming Event] ──► [Try Consume] ──(Success)──► [ACK to Queue]
                           │
                           ▼ (Exception / Transient Timeout)
                     [Retry Count < 3 ?]
                      ├── Yes ──► [Delayed Queue: Backoff + Jitter]
                      └── No  ──► [Dead Letter Queue (DLQ)]
                                          │
                                          ▼
                                   [P1 Alert / Investigation]
                                          │
                                          ▼ (Post-fix)
                                   [Batch Replay Tooling]
```

1. **Poison Pill Messages**:
   - When a payload contains malformed bytes or triggers a fatal runtime bug, uncapped retries cause workers to crash loop indefinitely, stalling queue processing.
2. **Three-Tier Retry Architecture**:
   - **Immediate Retries**: 2 in-memory retries for transient network drops.
   - **Exponential Backoff with Full Jitter Queue**:
     $$T_{\text{sleep}} = \min(T_{\text{max}}, T_{\text{base}} \times 2^{\text{attempt}}) \times \text{Uniform}(0, 1)$$
     Failed messages route to delay queues (e.g., 10s, 1m, 5m buckets), releasing worker threads for other traffic.
3. **Dead Letter Queue (DLQ)**:
   - Upon exhausting maximum retry limits (e.g., 3-5 attempts), the message, stack trace, and metadata are diverted to a dedicated DLQ.
   - Alerts notify on-call engineers, and an administrative tool allows replays once the bug is resolved.

---

## 8 · Production Schema & Implementation Blueprint

### 8.1 · Database Schemas (PostgreSQL Production Blueprint)

```sql
-- 1. Producer-side Outbox Table
CREATE TABLE outbox_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type VARCHAR(64) NOT NULL,       -- e.g. 'Order', 'Payment'
    aggregate_id VARCHAR(64) NOT NULL,         -- Entity key for Kafka partition hashing
    event_type VARCHAR(64) NOT NULL,           -- e.g. 'OrderPaid', 'OrderCancelled'
    payload JSONB NOT NULL,                    -- Domain event body (ECST payload)
    trace_id VARCHAR(64) NOT NULL,             -- Distributed trace identifier
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- Partition daily to allow dropping historical partitions cleanly without DELETE bloat
CREATE TABLE outbox_events_2026_10_07 PARTITION OF outbox_events
    FOR VALUES FROM ('2026-10-07 00:00:00+00') TO ('2026-10-08 00:00:00+00');

-- 2. Consumer-side Inbox Table
CREATE TABLE inbox_records (
    message_id VARCHAR(64) NOT NULL,           -- Global event unique identifier
    consumer_group VARCHAR(64) NOT NULL,       -- Consumer group name
    handler_name VARCHAR(64) NOT NULL,         -- Processing function identifier
    status VARCHAR(32) NOT NULL DEFAULT 'DONE',
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (message_id, consumer_group)
);
```

### 8.2 · Producer: Atomic Local Commit (Go Blueprint)

```go
func (s *OrderService) CompleteOrderPayment(ctx context.Context, orderID string) error {
    tx, err := s.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback()

    // 1. Update business table
    const updateOrderSQL = `UPDATE orders SET status = 'PAID', updated_at = NOW() WHERE id = $1`
    if _, err := tx.ExecContext(ctx, updateOrderSQL, orderID); err != nil {
        return fmt.Errorf("failed to update order: %w", err)
    }

    // 2. Build domain event (ECST payload)
    eventPayload := OrderPaidEvent{
        OrderID:     orderID,
        Status:      "PAID",
        PaidAt:      time.Now().UTC(),
        TraceID:     tracer.TraceIDFromContext(ctx),
    }
    payloadBytes, _ := json.Marshal(eventPayload)

    // 3. Colocate insert into outbox within the same local transaction
    const insertOutboxSQL = `
        INSERT INTO outbox_events (aggregate_type, aggregate_id, event_type, payload, trace_id)
        VALUES ($1, $2, $3, $4, $5)`
    if _, err := tx.ExecContext(ctx, insertOutboxSQL, "Order", orderID, "OrderPaid", payloadBytes, eventPayload.TraceID); err != nil {
        return fmt.Errorf("failed to insert outbox: %w", err)
    }

    // 4. Commit local ACID transaction atomically
    return tx.Commit()
}
```

### 8.3 · Consumer: Inbox Idempotent Worker (Python Blueprint)

```python
def process_order_event(db_pool, message: KafkaMessage):
    event = json.loads(message.value)
    event_id = event["event_id"]
    order_id = event["aggregate_id"]

    with db_pool.get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # 1. Attempt atomic Inbox insertion
                cur.execute(
                    """
                    INSERT INTO inbox_records (message_id, consumer_group, handler_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (message_id, consumer_group) DO NOTHING
                    RETURNING message_id;
                    """,
                    (event_id, "billing_service_group", "handle_order_paid")
                )
                row = cur.fetchone()
                if not row:
                    # Duplicate detected: bypass execution and safely acknowledge
                    logger.info("Duplicate event %s skipped successfully.", event_id)
                    conn.commit()
                    return

                # 2. Mutate domain state
                cur.execute(
                    "UPDATE account_balance SET balance = balance - %s WHERE order_id = %s",
                    (event["amount"], order_id)
                )

                conn.commit()  # Inbox record and domain mutation commit atomically
            except Exception as e:
                conn.rollback()
                logger.error("Processing failed, will be retried: %s", str(e))
                raise e
```

---

## 9 · Architectural Comparison Matrix

| Architecture Pattern | Consistency Guarantee | Primary DB Overhead | End-to-End Latency | System Complexity | Applicable Scope |
|---|---|---|---|---|---|
| **Naive Dual-Write** | None (guaranteed silent event loss or phantom events) | Low | Milliseconds | Minimal | **Prohibited in production core paths**. Only viable for best-effort metrics telemetry. |
| **Distributed 2PC / XA** | Strong Consistency (CP) | **Catastrophic** (lock contention, throughput drops >90%) | Seconds (multi-phase coordination RTT) | Very High (unsupported by modern brokers) | Confined to legacy homogeneous relational database systems within a single datacenter. |
| **Outbox + Polling Publisher** | Eventual Consistency (At-Least-Once) | High (polling queries, lock contention, table bloat) | Seconds (1s - 5s) | Low (internal application threads only) | Small-scale systems, internal administrative tools, or low-frequency batch processes. |
| **Outbox + CDC + Pub/Sub (Industry Standard)** | **Eventual Consistency (At-Least-Once + Effectively-Once)** | **Minimal** (reads append WAL directly; zero query load) | **Milliseconds (10ms - 50ms)** | Moderate (requires CDC pipeline and broker topology) | **The industrial standard for scalable microservices and mission-critical event-driven architectures**. |
| **Direct Table CDC (without Outbox)** | Weak Eventual Consistency | Minimal | Milliseconds | Moderate | **Suffers from severe limitations**: loses domain event semantics (exposes low-level CRUD delta only); exposes uncommitted intermediate states; tightly couples consumers to primary table schema. |

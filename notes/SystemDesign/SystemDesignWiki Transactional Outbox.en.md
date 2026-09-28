# Wiki · Transactional Outbox Pattern

Category: Wiki Pattern Library · Knowledge Point
Related Patterns: [[SystemDesign06 Async Messaging Systems|Message Queue]] · [[SystemDesign01 Stateless Service|Idempotency Guarantees]] · [[SystemDesign11 Notification System|Notification System Deep Dive]]

---

## 1 · Core Problem: The Dual-Write Hazard

In distributed and event-driven architectures, services frequently need to update database state and publish events to external systems (Kafka, RabbitMQ, Webhooks). Because cross-system calls cannot naturally share a single local transaction, naive dual-writes inevitably fail at boundaries:

* **Write DB first, then publish to MQ**: If the DB commit succeeds, but network timeout or process crash happens during MQ publishing, the event is permanently lost, causing irreparable state drift.
* **Publish to MQ first, then write DB**: If MQ publish succeeds, but the subsequent DB transaction rolls back due to constraint violations, deadlocks, or node crashes, downstream consumers process phantom events that never truly existed.
* **Limitations of 2PC / XA**: Two-phase commit imposes massive coordination overhead, long lock retention windows resulting in severe throughput collapse, and most modern distributed log brokers (e.g. Kafka) do not natively support cross-database XA transactions.

---

## 2 · Core Idea: Exploiting Local ACID Transactions

The Transactional Outbox pattern avoids coordinating two disparate distributed systems at the application layer by reducing the problem to an internal **single database transaction**:

```text
[Business Service]
        │
        ▼ (Local ACID Transaction)
┌───────────────────────────────────────┐
│  Database                             │
│  ┌─────────────────┐ ┌──────────────┐ │
│  │ Business Table  │ │ Outbox Table │ │
│  │ (e.g. Order/Job)│ │ (Event Queue)│ │
│  └─────────────────┘ └──────────────┘ │
└───────────────────────────────────────┘
                     │
             (Asynchronous Read)
                     ▼
           [Message Relay / CDC]
                     │
                     ▼
          [Event Bus (Kafka/Pulsar)]
```

1. **Colocated Transactional Write**: Besides business tables, maintain a dedicated `outbox` table in the same database. When a business mutation occurs, the event payload is inserted into `outbox` inside the exact same **Local ACID Transaction**.
2. **Atomic Persistence**: When the transaction commits, business records and event payloads are committed atomically. If anything aborts, both roll back cleanly with zero intermediate state.
3. **Asynchronous Relay**: A separate, asynchronous Message Relay reads records from the `outbox` table and forwards them to the external event bus.

---

## 3 · Message Relay Mechanisms & Trade-offs

Moving events from the `outbox` table to the broker is handled via two primary architectural patterns:

| Dimension | Polling Publisher | Change Data Capture (CDC / Log Tailing) |
|---|---|---|
| **Mechanism** | Scheduled worker threads periodically query:<br>`SELECT * FROM outbox WHERE status = 'PENDING' ORDER BY created_at LIMIT 100 FOR UPDATE`, update status or delete upon ack. | Uses CDC engines (e.g., Debezium, Canal) tailing the database commit log (MySQL Binlog, PostgreSQL WAL) directly and streaming rows to Kafka. |
| **Database Overhead** | High. Constant table/index scans and row locks compete for connection pools and CPU on the primary database. | Minimal. Reads sequential append logs from disk or replication stream; incurs zero query engine load or lock contention. |
| **Relay Latency** | Seconds (bounded by polling interval and batch processing size). | Milliseconds (nearly real-time streaming alongside commit logs). |
| **Infra Complexity** | Low. Plain application scheduler code; database-agnostic. | Moderate to High. Requires dedicated CDC pipeline infrastructure (e.g., Kafka Connect cluster) tied to storage-specific log formats. |
| **Production Guideline** | Low-frequency systems, simple MVPs, or background batch processors. | **The industrial standard for high-throughput, microservice production architectures**. |

---

## 4 · Guarantees & System Boundaries

1. **Delivery Semantics: At-least-once Delivery**
   * Network retries, timeout reconnections, or relay failovers may result in duplicate event deliveries to the broker.
   * **Hard Requirement**: Downstream consumers must implement strict **Consumer Idempotency**. Consumers deduplicate messages via `event_id` or `biz_id + version` keys stored in an idempotency cache or unique constraint index.
2. **Ordering Guarantees**
   * To preserve causality for a single entity, the relay must assign the entity ID (e.g. `order_id`) as the Kafka **Partition Key**. This confines events of the same entity to a single partition, consumed in strict sequential order.
3. **Outbox Table Governance & Table Bloat**
   * Under polling models, unpruned outbox tables grow unbounded, degrading index lookups. Scheduled purge jobs or range partitioning drops must be configured.
   * Under CDC models, rows in `outbox` can be safely truncated via lightweight background retention workers once captured.

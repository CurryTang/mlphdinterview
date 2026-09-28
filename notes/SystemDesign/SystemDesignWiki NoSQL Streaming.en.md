# Wiki · NoSQL + Streaming: CDC & Event-Driven Pipelines

Wiki Navigation: [[SystemDesign00 Overview|00 Blueprint & Numbers]] → [[SystemDesignWiki NoSQL Streaming|Wiki · NoSQL + Streaming]]

**NoSQL + Streaming** is a foundational cloud-native data architecture pattern. By coupling NoSQL persistence with native Change Data Capture (CDC) and stream processing engines, it delivers **sub-second event-driven reactions backed by the single source of truth of a high-throughput NoSQL database**.

---

## 1 · Core Philosophy & Architectural Patterns

### 1.1 · Eliminating the Dual-Write Hazard via Built-in CDC
In distributed systems, writing simultaneously to a primary database and publishing an event to a message broker introduces race conditions, network failures, and state inconsistency.
The NoSQL + Streaming pattern eliminates application-level dual-writes:
- **Single Write Path**: Applications execute a single write mutation against the NoSQL store (e.g., DynamoDB `PutItem` or MongoDB `updateOne`).
- **Engine-Level Native CDC**: The NoSQL engine exposes its internal physical transaction commit log (DynamoDB Storage Nodes Log, MongoDB Oplog, Cassandra CommitLog, Redis Streams WAL) as an ordered, durable **Change Stream**.

```text
[Business Service]
        │
        ▼ (Single Direct Write)
┌──────────────────────────────────────┐
│  NoSQL Primary Store                 │
│  (e.g., DynamoDB / MongoDB / Scylla) │
│  ┌────────────────────────────────┐  │
│  │ Primary Key Indexed Documents  │  │
│  └────────────────────────────────┘  │
└──────────────────┬───────────────────┘
                   │
                   ▼ (Built-in Sharded Log Stream)
           [Change Stream] (DynamoDB Streams / Oplog)
                   │
                   ├───────────────────────┬───────────────────────┐
                   ▼                       ▼                       ▼
           [Search Index Sync]     [Cache Invalidation]    [Real-time Analytics]
             (Elasticsearch)             (Redis)              (Flink / OLAP)
```

### 1.2 · Stream-Table Duality (Kappa Architecture)
- **A Table is a static snapshot of state at a discrete point in time**;
- **A Stream is the complete ordered sequence of historical mutations that produced that state**:
  $$\text{Table} = \int \text{Stream} \, dt, \quad \text{Stream} = \frac{d(\text{Table})}{dt}$$
- **OldImage vs NewImage Snapshots**: Change stream records natively embed both pre-mutation (`OldImage`) and post-mutation (`NewImage`) states. Downstream consumers can calculate deltas without reading the primary database.
- **Per-Partition FIFO Invariant**: All mutations targeting the same Partition Key / Document ID maintain strict FIFO ordering in the stream.

### 1.3 · Materialized Views & Polyglot Persistence via CQRS
NoSQL + Streaming powers Command Query Responsibility Segregation (CQRS) without distributed 2PC transactions:
- **Write-Optimized Store**: NoSQL handles high-throughput key-based mutations and point queries with partition isolation.
- **Async Read View Materialization**: Stream workers (AWS Lambda, Flink, Kafka Connect) consume the change stream and asynchronously hydrate read-optimized replicas:
  - Synchronize to **Elasticsearch** for multi-dimensional full-text indexing;
  - Synchronize to **Redis** for hot-key in-memory caches;
  - Hydrate **ClickHouse / Snowflake** for real-time OLAP aggregations.

---

## 2 · Quantitative Baselines: QPS & Latency

Because NoSQL change streams are partitioned 1:1 with the underlying storage shards, throughput scales horizontally with table capacity:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   NoSQL + Streaming Performance Baselines              │
├─────────────────────────┬──────────────────────┬───────────────────────┤
│ Component & Stream Tech │ Stream QPS Baseline  │ End-to-End Latency    │
├─────────────────────────┼──────────────────────┼───────────────────────┤
│ AWS DynamoDB Streams    │ 100,000 - 1,000,000+ │ 50 - 200 ms (P99<1s)  │
│ MongoDB Change Streams  │ 20,000 - 50,000 /RS  │ 10 - 50 ms            │
│ Redis Streams           │ 100,000 - 500,000/node│ < 1 ms (Microseconds) │
│ ScyllaDB / Cassandra CDC│ 500,000 - 2,000,000+ │ 20 - 100 ms           │
└─────────────────────────┴──────────────────────┴───────────────────────┘
```

- **DynamoDB Streams**: Stream shards scale automatically alongside DynamoDB table partition splits, handling millions of write mutations per second with zero read-capacity penalty on the primary table; messages are retained for 24 hours.
- **MongoDB Change Streams**: Reads directly from the WiredTiger `local.oplog.rs` capped collection without disk head thrashing, supporting tens of thousands of ops per replica set.
- **Redis Streams (`XADD` / `XREADGROUP`)**: Single-threaded in-memory append logs with persistent consumer groups and microsecond-level consumption, bounded by host RAM limits.

---

## 3 · Production Use Cases & Decision Matrix

### 3.1 · Ideal Scenarios
1. **Accurate Cache Eviction (Zero Dirty Reads)**:
   - Eliminates race conditions between writing to DB and invalidating Redis cache.
   - The application writes strictly to NoSQL; the stream consumer receives mutation events and purges the corresponding Redis key.
2. **Polyglot Search Index Synchronization**:
   - E-commerce catalog data is written to DynamoDB/MongoDB.
   - The change stream drives continuous updates to Elasticsearch inverted indexes without coupling the write API to search infrastructure.
3. **Cross-Region Active-Active Replication (Global Tables Engine)**:
   - DynamoDB Global Tables uses DynamoDB Streams under the hood to replicate mutations across AWS regions with Last-Writer-Wins conflict resolution.
4. **Real-Time Fraud Detection & Sliding Window Analytics**:
   - Transaction records written to NoSQL immediately enter Apache Flink to evaluate fraud rules (e.g. transaction frequency over a 5-minute sliding window).

### 3.2 · Anti-Patterns & When NOT to Use
- **Pure Transient Task Dispatching**: If tasks require no durable database entity (e.g. firing crawler jobs), use a **Message Queue** directly rather than persisting dummy records into NoSQL.
- **Cross-Entity Total Global Ordering**: NoSQL change streams guarantee FIFO ordering only within the same partition key. If an application requires global wall-clock ordering across unrelated entities, use a single-partition **Kafka Log** or centralized consensus coordinator.

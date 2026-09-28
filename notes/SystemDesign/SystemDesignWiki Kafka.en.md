# Wiki · Kafka Internals & System Design Practice

Category: Wiki Pattern Library · Knowledge Point
Related Patterns: [[SystemDesignWiki Transactional Outbox|Transactional Outbox]] · [[SystemDesignWiki Pull vs Push|Pull vs Push]] · [[SystemDesign06 Async Messaging Systems|Message Queue]] · [[SystemDesign11 Notification System|Notification System]] · [[SystemDesign10 Flash Sale|Flash Sale]]

---

## 1 · Core Identity: From Queue to Distributed Append-Only Commit Log

Traditional message brokers (RabbitMQ / ActiveMQ) are designed around **in-memory message queues with destructive reads (delete-on-ack)**, making them ill-suited for hyperscale ingestion, multi-tenant stream processing, and historical replays.

Kafka centers its paradigm on the **Distributed Append-Only Commit Log**:
* **Topology Abstraction**: A Topic represents a logical stream of events, partitioned into multiple physical shards called **Partitions**.
* **Immutable Sequential Log**: Each Partition only supports strict sequential appends. Messages receive a monotonically increasing 64-bit integer **Offset** and become immutable once committed.
* **Non-destructive Consumption**: Consumers only report their reading position (Committed Offset). Retention is fully decoupled from consumption progress, natively unlocking **historical replays, multi-consumer group independence, and time-travel debugging**.

---

## 2 · The Four Mechanical Sympathies of Kafka

A standard Kafka broker comfortably handles hundreds of thousands of writes per second while saturating 10GbE network interfaces, powered by four hardware-aligned architectural decisions:

### 2.1 · Sequential Disk I/O
* While conventional disk access is slow due to random seek times, modern SAS/SSD sequential disk appends achieve **$300 - 600\text{ MB/s}$**, outperforming random memory access across system buses.
* Kafka maps Partitions directly to disk file segments. Every append writes strictly to the end of active segment files, eliminating disk head repositioning and B-tree split overhead.

### 2.2 · Deep OS Page Cache Integration
* Kafka avoids custom memory caching within the JVM heap, delegating memory management entirely to the **Linux OS Page Cache**.
* **Key Advantages**:
  1. **Zero JVM GC Overhead**: Millions of messages remain cached in memory with minimal JVM heap footprint, eliminating GC Stop-The-World (STW) pauses.
  2. **Zero Cold-Start on Restarts**: When a broker restarts or crashes, the OS Page Cache survives, preserving hot cached data.
  3. **Read/Write Co-location**: Newly produced messages written to the Page Cache are immediately fetched by tailing consumers directly from memory without touching physical disk platters.

### 2.3 · Zero-Copy via `sendfile()`
Standard server I/O involves **4 user/kernel context switches and 4 data copies** (Disk $\to$ Kernel Page Cache $\to$ JVM buffer $\to$ Socket buffer $\to$ NIC buffer).

Kafka utilizes the Linux `sendfile()` system call to bypass user space:
```text
[Disk File] ──(DMA)──► [OS Page Cache] ──(Descriptor Copy / DMA)──► [NIC Buffer (Network Interface)]
```
* Context switches drop from 4 to 2, and data copies drop to 2 hardware DMA transfers with **0 CPU copies**. CPU utilization remains negligible even at wire speeds.

### 2.4 · End-to-End Batching & Compression
* **Batch Aggregation**: Producers pool records using `batch.size` (e.g. 64KB) and `linger.ms` (e.g. 20ms), amortizing RPC protocol header overhead across thousands of messages per request.
* **End-to-End Compression**: Producers compress batches (`lz4` or `zstd`). Brokers write the raw compressed byte stream directly to disk without decompressing. Consumers stream and decompress in flight, saving up to 70% of network bandwidth and disk storage.

---

## 3 · Replication & Distributed Consensus

```text
               ┌────────────────────────────────────────────────────────┐
               │                      Topic Partition                   │
               │                                                        │
               │   Leader Replica (Broker 1)                            │
               │   [ 0 ][ 1 ][ 2 ][ 3 ][ 4 ][ 5 ][ 6 ] (LEO = 7)        │
               │                 ▲               ▲                      │
               │                 │               │                      │
               │            High Watermark      Log End Offset          │
               │               (HW = 3)            (LEO)                │
               │                 │                                      │
               │  Follower 1 (ISR)               Follower 2 (Lagging)   │
               │  [ 0 ][ 1 ][ 2 ][ 3 ] (LEO=4)   [ 0 ][ 1 ] (LEO=2)     │
               └────────────────────────────────────────────────────────┘
```

### 3.1 · Storage Internals: Segments & Sparse Indexes
Each Partition is split physically into immutable **Segments** (default 1GB):
* `.log`: Binary message payload log.
* `.index`: Sparse offset index (Offset $\to$ Physical File Position).
* `.timeindex`: Timestamp index.
* **Lookup Algorithm**: An index entry is written every 4KB. To look up Offset 1005:
  1. Binary search the segment list in memory to locate the active segment file;
  2. Binary search `.index` to find the largest indexed offset $\le 1005$ (e.g., Offset 1000 $\to$ Position 16384);
  3. Sequentially scan the `.log` from byte offset 16384 to find record 1005 within sub-milliseconds.

### 3.2 · ISR (In-Sync Replicas) & High Watermark (HW)
* **ISR Set**: Followers that heartbeat and keep up with the Leader within `replica.lag.time.max.ms`.
* **LEO (Log End Offset)**: The next offset to be written in a specific replica.
* **HW (High Watermark)**: The minimum LEO across all replicas in the ISR. **Consumers can only read messages below the HW**, preventing reads of uncommitted records if the leader crashes.
* **Zero-Loss Quorum Configuration**:
  $$\text{acks} = \text{all (-1)} \quad \land \quad \text{min.insync.replicas} = 2 \quad \land \quad \text{replication.factor} \ge 3$$

### 3.3 · Leader Epoch
* Legacy Kafka relied on HW truncation during crash recovery, causing silent log truncation and replica divergence bugs.
* **Leader Epoch**: Assigns an epoch number and starting offset to each leadership tenure. When synchronizing, replicas reconcile state against the Leader Epoch instead of ambiguous HW offsets, guaranteeing deterministic recovery.

### 3.4 · KRaft Consensus
* **ZooKeeper Limitations**: Synchronizing metadata for tens of thousands of partitions overwhelmed ZK; controller failover took minutes.
* **KRaft (Kafka Raft Metadata Mode)**: Treats metadata as an internal replicated commit log managed by a Raft quorum. Controller failover happens in milliseconds, easily scaling clusters to millions of partitions.

---

## 4 · Delivery Semantics & Exactly-Once Semantics (EOS)

### 4.1 · Idempotent Producer
* Configured via `enable.idempotence=true`.
* The Broker assigns each producer a 64-bit **PID (Producer ID)**.
* Producers tag messages sent to each partition with a monotonically increasing **Sequence Number**.
* The Broker checks incoming sequence numbers:
  $$\text{NextSeq} = \text{CurrentSeq} + 1 \implies \text{Commit}$$
  $$\text{NextSeq} \le \text{CurrentSeq} \implies \text{Acknowledge duplicate without appending}$$
* Transparently neutralizes network retry duplicates within a single partition.

### 4.2 · Transactions & Read-Committed Isolation
* Enables atomic multi-partition writes, crucial for stream transformations ("consume-transform-produce").
* Managed by a **Transaction Coordinator** backed by the `__transaction_state` internal topic.
* Writes a two-phase commit marker (Commit/Abort Control Batch) into target partitions.
* Consumers configured with `isolation.level=read_committed` pause reading at uncommitted boundaries, exposing only committed messages to downstream processors.

### 4.3 · Consumer Rebalancing: Eager vs Cooperative Sticky
* **Eager Protocol (Legacy)**: Drops all assigned partitions on any group member change (Stop-The-World), triggering full cluster consumption stalls during rolling deployments.
* **Cooperative Sticky Protocol (Modern Standard)**: Incrementally reassigns only partitions that need migration, allowing unaffected consumers to proceed uninterrupted.

---

## 5 · Kafka in System Design Interview Problems

In production architectures and system design interviews, Kafka serves as the central circulatory system. Here is how it is applied across major interview archetypes:

### 5.1 · Extreme Concurrency Flash Sale (Flash Sale / Ticket Booking)
* **Core Problem**: Database write throughput caps at $1,000 - 2,000\text{ TPS}$. Hundreds of thousands of concurrent checkouts will cause lock deadlock and database failure.
* **Kafka Architecture**:
  1. Redis pre-deducts inventory, then emits an `OrderCreationCommand` to Kafka and immediately returns an order queued token to the client.
  2. **Partition Key Design**: Set `sku_id` (Product ID) as the Partition Key.
  3. **Serialization Advantage**: All checkout requests for the same SKU land in the same partition, handled sequentially by a single worker thread. This **converts distributed row-lock contention into single-threaded in-memory FIFO processing**.
  4. **Batch Insertion**: Consumers pull records in micro-batches (e.g. 500 records) to execute single multi-row `INSERT INTO orders VALUES (...), (...)` statements, increasing DB write throughput tenfold.

### 5.2 · Massive Scale Mobile Notification System (Notification System)
* **Core Problem**: External vendors (APNs, FCM, SMS gateways) have strict concurrency quotas. Burst marketing campaigns (billions of pushes) must not starve critical high-priority alerts (OTP login codes).
* **Kafka Architecture**:
  1. **Physical Topic Priority Isolation**:
     * `notif-priority-otp`: Dedicated to 2FA / transactional alerts, configured with `linger.ms=0` for immediate low-latency dispatch.
     * `notif-bulk-marketing`: Bulk marketing campaigns, tuned for large batch throughput.
  2. **Dynamic Backpressure & Throttling**: Consumers integrate a token bucket rate limiter. When external vendor gateways return `429 Too Many Requests`, consumers call `pause()` on specific partitions, leveling peak bursts without overwhelming upstream systems.

### 5.3 · Social Timelines & Feed System (Photo Sharing & Feed)
* **Core Problem**: Celebrity posts trigger massive write amplification during follower fan-out.
* **Kafka Architecture**:
  1. User posts a photo $\to$ Primary DB writes record + Transactional Outbox $\to$ Emits `PostCreatedEvent` to Kafka.
  2. **Partition Key Design**: Use `author_id` as the Partition Key to guarantee causal ordering (post, edit, delete) per author.
  3. **Tiered Fan-out Pipeline**: Fan-out worker pools ingest the event stream and route dynamically based on follower count: standard accounts fan out to follower inboxes; celebrity posts are routed only to the author's outbox.

### 5.4 · Distributed Financial Transactions & Saga Orchestration (Payment & Order Saga)
* **Core Problem**: Multi-service distributed consistency without distributed 2PC locking.
* **Kafka Architecture**:
  1. Services update local datastores and emit domain events to Kafka via **Transactional Outbox + Debezium CDC**.
  2. **Partition Key Design**: Mandate `order_id` as the Partition Key. This ensures all events concerning a specific transaction (`OrderCreated`, `PaymentAuthorized`, `InventoryReserved`) land on a single partition in chronological causal order.
  3. **Idempotent Ingestion**: Downstream services insert an idempotency token (`order_id + event_status`) into a local unique constraint table before mutating state, guaranteeing safe Saga compensations.

### 5.5 · Distributed Logging & Real-time Metrics (Logging & Telemetry Highway)
* **Core Problem**: Millions of QPS log events from distributed microservices must be ingested without blocking application threads or dropping logs.
* **Kafka Architecture**:
  1. **High-Throughput Configuration**:
     * Producer: `acks=1`, `compression.type=lz4`, `linger.ms=50`, `batch.size=131072` (128KB).
     * Partition Strategy: Round-robin or uniform hashing across all available partitions to distribute disk and network load uniformly across the cluster.
  2. Downstream Flink clusters read the log streams for real-time alerting, while ClickHouse micro-batches ingest data in the background for analytical ad-hoc querying.

### 5.6 · Asynchronous LLM RL & Inference Serving (Async LLM RL Platform)
* **Core Problem**: Massive throughput imbalance between CPU-bound Actor rollout generation and GPU-bound Learner gradient backpropagation.
* **Kafka Architecture**:
  1. Rollout workers serialize prompt token streams, logprobs, and verified reward scores into binary payloads pushed to a dedicated trajectory topic.
  2. Kafka acts as a shock absorber during heavy GPU checkpointing or tensor parallel synchronization, allowing thousands of Rollout pods to continue generating at maximum capacity without GPU blocking.

---

## 6 · Essential Configuration & Interview Cheat Sheet

| Parameter | Recommended Value | Architectural Rationale |
|---|---|---|
| `acks` | `-1` / `all` | Strict durability guarantee; requires acknowledgment from all ISR members. Mandatory for financial/order pipelines. |
| `min.insync.replicas` | `2` | Guarantees at least two copies are committed before acknowledging, preventing silent data loss upon node failover. |
| `enable.idempotence` | `true` | Enforces broker-side PID + sequence number deduplication, eliminating network retry duplicates. |
| `compression.type` | `lz4` or `zstd` | Reduces network transmission volume by over 60% and relieves disk I/O bottlenecks. |
| `linger.ms` | `10 - 50` | Introduces small artificial producer delay to assemble large batches, boosting throughput by 3–5x. |
| `max.poll.interval.ms` | Match workload SLA | Prevents coordinator from misinterpreting lengthy consumer business logic as a node crash. |
| `partition.assignment.strategy` | `CooperativeStickyAssignor` | Incremental cooperative rebalancing; eliminates full-cluster STW stalls during rolling deployments. |

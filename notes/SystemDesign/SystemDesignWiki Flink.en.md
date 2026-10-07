# Wiki · Apache Flink Architecture & Geographic Locality Stream Processing

Category: Wiki Pattern Library · Core Processing Engine
Related Patterns: [[SystemDesignWiki Kafka|Kafka]] · [[SystemDesignWiki NoSQL Streaming|NoSQL Change Streams]] · [[SystemDesignWiki Stateless Architecture|Stateless Architecture & Externalization]] · [[SystemDesign07 Photo Sharing Feed|Social Feeds & Trending]]

---

## 1 · Core Definition & Design Philosophy: What Is Flink?

The core essence of Apache Flink can be summarized in one equation:

$$\boxed{\text{Flink = distributed + stateful + event-time stream processing}}$$

Flink represents a fundamental paradigm shift away from traditional stateless workers (e.g., AWS Lambda, generic consumer microservices):

### 1.1 · Stateless Worker
```text
Event ──► [ Function / Microservice ] ──► Output
                   │
                   ▼ (Every state mutation requires a network RPC)
          [ Remote Redis / MySQL ]
```
- Receives events and executes pure function transformations. Maintaining aggregated state (counters, sliding windows, deduplication sets) demands network RPCs to external storage;
- Under massive event throughput, external databases suffer from severe network round-trip time (RTT), row-level lock contention, and distributed transaction bottlenecks.

### 1.2 · Flink Stateful Stream Processing
```text
Event ──► [ Local TaskSlot ] ──► In-Place State Update (Heap / Embedded RocksDB) ──► Maybe Emit Aggregate
```
- Flink colocates computation and state storage. State lives directly inside the worker process's local memory or embedded RocksDB;
- Event-driven state transitions execute in-place in local memory with sub-microsecond latency, bypassing external databases completely.

Consider the classic Trending Hashtag event stream:
```text
post1: #AI, US
post2: #NBA, US
post3: #AI, US
...
```
Flink partitions the stream:
```java
stream.keyBy(event -> Tuple2.of(event.country, event.hashtag))
```
Deterministic hash routing assigns each composite key to a dedicated worker:
- Worker A owns `(#AI, US)`
- Worker B owns `(#NBA, US)`
- Worker C owns `(#AI, UK)`

Worker A maintains in-place rolling state locally:
```text
(#AI, US)
├── 1 min count  = 105
├── 5 min count  = 423
└── 1 hour count = 4,921
```

---

## 2 · The Four Architectural Pillars of Flink

### 2.1 · Keyed State & Operator Local State
- **Deterministic Partition Binding**:
  $$\text{hash}(country, hashtag) \pmod N \longrightarrow \text{TaskSlot } i$$
- **Eliminating Distributed Transactions**: Incrementing counters (`count += 1`) executes entirely in-process, bypassing distributed locks and multi-node coordination.
- **State Backend Options**:
  - `HashMapStateBackend`: State resides in JVM heap memory with nanosecond-level access, bounded by heap capacity and GC overhead;
  - `EmbeddedRocksDBStateBackend`: State lives in an out-of-core embedded RocksDB (off-heap memory + local NVMe SSD), handling terabyte-scale state per node exceeding RAM boundaries.

### 2.2 · Event Time Processing & Watermarks
- **Time Semantics Delineation**:
  - **Processing Time**: Wall-clock time of the executing physical node. Highly susceptible to network congestion, GC pauses, and consumer lag; historically non-reproducible;
  - **Event Time**: Timestamp attached when the event actually occurred on the client device (e.g., user posted at `12:01:00`, but packet arrived at Flink at `12:04:30` due to mobile network congestion).
- **Watermark Out-of-Order Toleration**:
  Watermarks act as monotonic progress markers $W(t)$ embedded in data streams. When an operator receives $W(t)$, it assumes all events with timestamp $\le t$ have arrived, triggering window materialization:
  $$W(t) = \max(\text{EventTime}) - \Delta_{\text{allowed\_delay}}$$
- **Native Multi-Window Models**: Out-of-the-box support for Tumbling, Sliding, and Session windows, complemented by Allowed Lateness and Side Outputs for lagging data, fitting requirements like "counts over trailing 1-min / 5-min / 1-hour windows".

### 2.3 · Distributed Checkpointing & End-to-End Exactly-Once Recovery
- **Chandy-Lamport Asynchronous Snapshots**:
  Flink JobManager periodically injects Checkpoint Barriers into source streams. As barriers flow through the DAG topology and align across inputs, operators asynchronously flush local keyed state to durable distributed storage (S3 / HDFS / Ceph).
- **Crash Recovery Pipeline**:
  $$\text{Restore Checkpoint State from S3} \;+\; \text{Replay Kafka from Barrier Offset} \Longrightarrow \text{Exactly-Once State Guarantee}$$
  When worker nodes crash, replacement nodes restore state from S3 within seconds and catch up from Kafka offsets, achieving zero data loss and zero state drift without application-level compensation logic.

### 2.4 · Incremental In-Place Computation vs Full-Scan Batch Queries
- **Traditional Relational / OLAP Anti-Pattern**:
  Periodic cron queries scanning raw historical records:
  ```sql
  SELECT hashtag, COUNT(*)
  FROM posts
  WHERE created_at > NOW() - INTERVAL 5 MINUTE
  GROUP BY hashtag;
  ```
  Repeatedly scanning tens of millions of raw rows forces CPU and disk I/O to explode linearly with traffic.
- **Flink Incremental Sliding Windows**:
  - Incoming event: Increments local accumulator (`+1`);
  - Expired bucket sliding out: Subtracts evicted bucket count (`-old_bucket_count`);
  - Amortized per-event update complexity is strictly $O(1)$, delivering millisecond end-to-end latency at multi-million QPS.

---

## 3 · Deep Architecture Deep-Dive: “Flink Write Locally on Geography Data Center”

In high-concurrency global system design interviews (e.g., Twitter Trending Hashtags, global real-time fraud detection, international user analytics), top-level architecture diagrams often highlight:

> **“Flink write locally on geography data center”**

This phrase **does not mean Flink merely writes to a single local disk**. Rather, it conveys a fundamental distributed systems invariant: **Raw events must be absorbed, aggregated, and stored locally within the originating geographic region / data center, strictly avoiding shipping unaggregated raw event streams across continents to a global centralized hub.**

### 3.1 · Why Enforce Geographic Local Processing?

#### 1. Minimizing Inter-Continental WAN Bandwidth (Push Computation Toward Data)
- Assume a global raw event rate of $500,000 \text{ events/sec}$ with an average event payload of $200 \text{ Bytes}$. Shipping raw streams generates $100 \text{ MB/s}$ of sustained cross-continental WAN traffic, incurring exorbitant bandwidth fees and risking trans-oceanic cable congestion;
- **Pushing Computation to Data**: Running independent regional Flink clusters in US, EU, and APAC data centers locally aggregates raw streams:
  $$1,000,000 \text{ Raw Events} \xrightarrow[\text{Flink}]{\text{Local Regional}} \text{Thousands of } (\text{hashtag}, \text{count}) \text{ aggregate pairs}$$
  Inter-continental data transfer volume drops by over $99.9\%$, preserving costly WAN links.

#### 2. Sub-Second End-to-End Latency
- Regional event processing remains contained within local DC networks:
  $$\text{Client} \longrightarrow \text{Regional Kafka} \longrightarrow \text{Regional Flink} \longrightarrow \text{Regional Cache (Redis)}$$
  Intra-datacenter network RTT is typically $< 2\text{ ms}$;
- If every raw event travelled to a single global data center across the Atlantic or Pacific, physical fiber RTTs of $150 \sim 250\text{ ms}$ plus internet routing jitter would make sub-second real-time trend discovery impossible.

#### 3. Failure Blast Radius Isolation
- When trans-oceanic cables sever or global cross-region network partitions occur:
  - **Centralized Architecture**: Remote regions fail to push events, causing global system outage;
  - **Geographic Locality Architecture**: US Flink continues serving real-time US trends; EU Flink continues serving EU trends. Only the global merge layer experiences temporary staleness, ensuring graceful degradation.

#### 4. Natural Geographic Affinity in Business Queries
- Real-time trending queries are inherently localized (e.g., "Top 50 trending in US", "Top 50 trending in UK");
- Partitioning by geography as the first layer:
  $$\text{US Traffic} \longrightarrow \text{US Flink Compute} \longrightarrow \text{US Regional Storage}$$
  Satisfies over $90\%$ of production read/write demands without any cross-region distributed locking.

---

## 4 · Global Aggregation Architecture: Two-Tier Hierarchical Aggregation

When the business requires a consolidated "Global Trending" view, the system implements a **Two-Tier Hierarchical Aggregation (MapReduce: Local Combine $\to$ Global Reduce)** topology:

```text
[US Users]            [EU Users]            [Asia Users]
    │                     │                      │
    ▼                     ▼                      ▼
[US Regional DC]     [EU Regional DC]       [Asia Regional DC]
 • US Kafka           • EU Kafka             • Asia Kafka
 • US Flink           • EU Flink             • Asia Flink
 • (#AI: 100k)        • (#AI: 70k)           • (#AI: 150k)
 • (#NBA: 80k)        • (#football: 120k)    • (#Kpop: 200k)
    │                     │                      │
    └─────────────────────┼──────────────────────┘
                          │ (Periodic aggregate deltas, e.g., every 10s)
                          ▼
             [Global Aggregation Layer]
               • Global Merge & Reduce:
                 Global(#AI) = 100k + 70k + 150k = 320k
               • Global Top-K Priority Queue
                          │
                          ▼
             [Global Regional Read Cache / CDN]
```

### Quantitative Comparison: Centralized vs Hierarchical Two-Tier Architecture

| Dimension | Centralized Raw Ingestion | Hierarchical Regional Processing |
| :--- | :--- | :--- |
| **Traffic Flow** | All global raw events stream directly to central hub | Raw events processed in-region; only summary deltas sent centrally |
| **WAN Bandwidth** | Extreme ($500\text{k QPS} \times 200\text{ B} = 100\text{ MB/s}$ continuous WAN) | Minimal ($99.9\%$ compression; sparse key counters sent every 10s) |
| **End-to-End Latency** | High & jitter-prone ($150 \sim 300\text{ ms}$ trans-oceanic fiber RTT) | Sub-second regional updates (Intra-DC RTT $< 2\text{ ms}$); global view adds fixed window delay |
| **Fault Isolation** | Poor (Central outage or fiber partition crashes entire global system) | Resilient (Regions operate autonomously; global merge tolerates network partition) |
| **Horizontal Scalability**| Bounded by central Kafka/Flink network NIC interface capacity | Horizontally scalable without limits (new regions plug into aggregation tree) |

---

## 5 · Architectural Mental Model: Unifying the Two Layers of Locality

In system design, candidates must distinguish and integrate **two distinct dimensions of locality**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Key Locality within Cluster:                                        │
│    • Mechanism: keyBy(hashtag) binds identical keys to dedicated slots │
│    • Target: In-place memory/RocksDB updates; zero RPCs/transactions   │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Geographic Locality across Data Centers:                            │
│    • Mechanism: Absorb raw traffic within regional DCs for pre-aggreg. │
│    • Target: Save WAN bandwidth, minimize latency, isolate blast radii │
└────────────────────────────────────────────────────────────────────────┘
```

$$\boxed{
\text{Key Locality within Cluster} \quad+\quad \text{Geographic Locality across Clusters}
}$$

This paradigm delivers both single-node memory efficiency and globally distributed fault tolerance.

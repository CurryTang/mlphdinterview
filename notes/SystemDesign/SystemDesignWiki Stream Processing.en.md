# Wiki · Stream Processing Evolution & Architecture Trade-offs

Category: Wiki Pattern Library · Knowledge Point  
Related Patterns: [[SystemDesignWiki Flink|Flink Stateful Streaming]] · [[SystemDesignWiki Kafka|Kafka Partitioned Log]] · [[SystemDesignWiki NoSQL Streaming|NoSQL Change Streams]] · [[SystemDesignWiki Event Bus|Event Bus]] · [[SystemDesign06 Async Messaging Systems|Messaging Systems]]

---

## 1 · Paradigm Evolution: The Three Generations of Distributed Computing

Distributed big data architectures have undergone three major generational leaps: from **disk-bound batch processing**, to **in-memory micro-batching**, and ultimately to **native event-driven continuous stream processing**.

```text
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Gen 1: Disk-Bound Batch │      │ Gen 2: In-Memory Micro  │      │ Gen 3: Native Continuous│
│ MapReduce / Hadoop      │ ───► │ Spark / Spark Streaming │ ───► │ Flink / Stream Engines  │
│ (2004 - 2010)           │      │ (2010 - 2015)           │      │ (2015 - Present)        │
│ Disk spills, high latency│     │ In-memory DAG, sub-second│     │ Per-event, sub-10ms,    │
│ static bounded datasets │      │ micro-batch partitions  │      │ stateful & event-time   │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

---

## 2 · MapReduce Physical Bottlenecks & The Spark In-Memory DAG Revolution

### 2.1 · MapReduce Physical I/O and Spill Bottlenecks
Google's MapReduce framework and its open-source implementation Apache Hadoop resolved petabyte-scale offline distributed compute. However, execution was hardcoded into a rigid two-phase contract: $\text{Map} \to \text{Shuffle} \to \text{Reduce}$.

```text
[Job 1: Map] ──► (Local Disk Spill) ──► (Network Shuffle) ──► [Job 1: Reduce] ──► (3x HDFS Replica Write)
                                                                                         │
┌────────────────────────────────────────────────────────────────────────────────────────┘
▼
[Job 2: Map] ──► (Local Disk Spill) ──► (Network Shuffle) ──► [Job 2: Reduce] ──► (3x HDFS Replica Write)
```

1. **Forced Inter-Stage Materialization Barriers**:
   - Every Map task sorted intermediate output and **spilled it to local disk**.
   - Every Reduce task wrote its output over the network to **HDFS with 3x replication**, incurring immense CPU serialization, disk write overhead, and cross-rack networking latency.
2. **Catastrophic Penalty on Iterative Computations**:
   - In graph computing (e.g. PageRank), machine learning (e.g. Gradient Descent, K-Means), or multi-way relational joins, algorithms loop for $K$ iterations.
   - MapReduce forced these into $K$ completely decoupled jobs. Data had to be repeatedly serialized to HDFS, then read back and deserialized in the next iteration.
   - **I/O Dominance**: In iterative pipelines, raw CPU calculation accounted for $<10\%$ of runtime, while disk and network I/O accounted for $>90\%$.
3. **Coarse-Grained Task Scheduling & Process Cold Starts**:
   - Early MR1 JobTracker/TaskTracker architectures provisioned dedicated JVM processes per task, preventing memory or executor reuse across pipeline stages.

---

### 2.2 · Spark's Breakthrough: RDD Abstraction and In-Memory DAG Engine
Apache Spark redesigned distributed computation abstractions to bypass disk materialization barriers:

```text
[Data Source]
      │
      ▼ (Narrow Dependency: In-Memory Pipeline, Zero Shuffle)
┌────────────────────────────────────────────────────────┐
│ RDD A (map) ──► RDD B (filter) ──► RDD C (flatMap)     │ Stage 1 (Pipelined)
└────────────────────────────────────────────────────────┘
      │
      ▼ (Wide Dependency: Shuffle Boundary, Partition Re-hash)
┌────────────────────────────────────────────────────────┐
│ RDD D (reduceByKey) ──► RDD E (mapValues)              │ Stage 2
└────────────────────────────────────────────────────────┘
      │
      ▼ (In-Memory Persistence Cache)
   [Memory Pool (RAM)] <--- Reused across iterations directly, eliminating HDFS writes
```

1. **Resilient Distributed Datasets (RDD)**:
   - Read-only, logically partitioned collections of records distributed across a cluster.
2. **Lineage Graphs and Coarse-Grained Fault Tolerance**:
   - Instead of checkpointing data partitions to disk between operators, Spark records a directed acyclic graph (**Lineage DAG**) of operations.
   - If a partition is lost due to an executor crash, the scheduler reconstructs **only the lost partition** by replaying upstream transformations, removing the need for physical replication at every step.
3. **Narrow vs Wide Dependencies**:
   - **Narrow Dependencies**: Each partition of the parent RDD is consumed by at most one child partition (e.g., `map`, `filter`). Executors fuse consecutive narrow operations into a single pipelined stage executed directly in CPU cache/registers.
   - **Wide Dependencies**: Multiple child partitions depend on parent partitions (e.g., `groupByKey`, `reduceByKey`, `join`). These form Stage boundaries where network shuffle is unavoidable.
4. **Iterative In-Memory Reuse**:
   - Algorithms reuse intermediate state via `rdd.persist(StorageLevel.MEMORY_ONLY)`, retaining state in JVM heap or off-heap memory, achieving $10\times - 100\times$ speedups over MapReduce.

---

## 3 · Spark Micro-Batch Architecture & Its Inherent Limitations

To leverage Spark's mature batch DAG scheduler and fault tolerance, Spark Streaming (DStreams) and Spark Structured Streaming adopted a **Micro-Batching** architecture.

```text
Continuous Input Stream:  ... ● ● ● ● ● ● ● ● ● ● ● ● ● ● ...
                               │           │           │
           [Time Window: 500ms]▼           ▼           ▼
                           ┌───────┐   ┌───────┐   ┌───────┐
                           │Batch 1│   │Batch 2│   │Batch 3│
                           └───────┘   └───────┘   └───────┘
                               │           │           │
                               ▼           ▼           ▼
                      [Spark DAG Job] [Spark DAG Job] [Spark DAG Job]
```

### 3.1 · Core Philosophy of Micro-Batching
- **"Streaming as Discretized Batches"**: Continuous unbounded streams are artificially sliced into small discrete time intervals ($100\text{ms} - 500\text{ms}$).
- Each micro-batch is converted into a standard RDD / Dataset and submitted to the Spark core DAG engine as an independent batch job.

---

### 3.2 · Four Inherent Architectural Flaws of Micro-Batching

#### 1. The Hard Latency Floor ($100\text{ms} - 500\text{ms}$)
End-to-end latency in a micro-batch engine is bounded by three unavoidable factors:
$$T_{\text{latency}} = T_{\text{batch\_window}} + T_{\text{driver\_scheduling}} + T_{\text{execution\_barrier}}$$
- **Scheduling Overhead**: For every micro-batch, the Driver must construct DAG stages, serialize task closures, and dispatch them to worker thread pools.
- **Barrier Synchronization**: A micro-batch cannot emit results until the slowest straggler task across all partitions finishes.
- Sub-millisecond latency is physically unattainable under this paradigm.

#### 2. Semantic Mismatch with Event Time and Late Events
- Micro-batch windows partition records strictly by **Processing Time** (the arrival time at the processing engine).
- In the real world, mobile clients and IoT devices experience intermittent connectivity, generating unpredictable network skew:
  $$t_{\text{event}} \ll t_{\text{processing}}$$
- When an event delayed by hours arrives in the current batch, it belongs to an earlier window. A micro-batch engine must retrieve, unpack, and mutate state across dozens of previously committed batches, causing state amplification and heavy rewrites.

#### 3. Sliding Window Misalignment
- If a pipeline requires a $30\text{-second}$ sliding window with an arbitrary non-divisible micro-batch size (e.g., $7\text{ seconds}$), batch boundaries do not align with window steps.
- The engine must maintain complex inter-batch buffers and union operations across overlapping intervals, inflating memory overhead.

#### 4. Driver Bottlenecks and Garbage Collection Thrashing
- Squeezing batch intervals down to $50\text{ms}$ generates:
  $$\frac{86,400\text{ s}}{0.05\text{ s}} \approx 1,728,000\text{ independent batch jobs/day}$$
- The Driver experiences relentless RPC dispatch overhead, while rapid creation of ephemeral RDD task objects fragments the JVM old generation, triggering severe Garbage Collection (GC) pauses that destabilize latency SLAs.

---

## 4 · Foundations of Native Stream Processing: The Dataflow Model

Google's seminal paper *The Dataflow Model* (2015) codified native stream computing around four orthogonal questions:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        The Dataflow Matrix                             │
├───────────────────┬────────────────────────────────────────────────────┤
│ 1. What?          │ Transformations (map, filter, sum, count)          │
│ 2. Where?         │ Windowing in Event Time (Tumbling, Sliding, Session)│
│ 3. When?          │ Watermarks & Triggers in Processing Time           │
│ 4. How?           │ Accumulating, Retracting, Discarding               │
└───────────────────┴────────────────────────────────────────────────────┘
```

### 4.1 · Time Semantics
1. **Event Time**: The timestamp embedded within the record at the physical moment of creation (e.g. sensor reading, user click). **Event Time is the only guarantee of deterministic, reproducible results under replayed logs**.
2. **Ingestion Time**: The timestamp assigned when the record is appended to the message broker's distributed commit log (Kafka / Pulsar).
3. **Processing Time**: The local system clock of the compute node executing the transformation. Non-deterministic and subject to clock skew, but carries zero synchronization overhead.

---

### 4.2 · Watermark Mechanisms
A Watermark is a monotonically increasing timestamp marker $W(t)$ flowing inline with events.

```text
Data Flow Stream:
... [e: 10:02] [e: 10:01] ───► [Watermark(10:00)] ───► [e: 09:59] [e: 09:58] ...
                                      │
                                      ▼
             Assertion: "The system asserts with high confidence that no future
                         events with Event Time <= 10:00 will arrive."
             Trigger: Fire evaluation for the [09:00 - 10:00] window.
```

- **Mathematical Invariant**: Receiving watermark $W(t)$ asserts that all subsequent records will have event time $t_e > W(t)$.
- **Bounded Out-of-Orderness**:
  $$W(t) = \max_{e \in \text{received}}(t_e) - \Delta t_{\text{delay}}$$
  where $\Delta t_{\text{delay}}$ represents the allowable network jitter budget.
- **Handling Late Events ($t_e \le W(t)$)**:
  1. *Discard*: Drop the record if past the allowed lateness budget.
  2. *Accumulate & Retract*: Emit a revised downstream emission retracting previous window totals.
  3. *Side Output*: Divert late records to a dedicated dead-letter stream for asynchronous reconciliation.

---

### 4.3 · Window Topologies

| Window Model | Boundary Logic | Overlap Behavior | Production Use Case |
|---|---|---|---|
| **Tumbling Window** | Fixed duration, uniform intervals | Non-overlapping: $[0, 10), [10, 20)$ | Aggregating hourly sales volume or per-minute API error rates |
| **Sliding Window** | Fixed duration, periodic slide step | Overlapping: $[0, 10), [5, 15)$ | 5-minute rolling averages updated every 10 seconds |
| **Session Window** | Inactivity timeout gap ($T_{\text{gap}}$) | Variable duration, dynamic | Tracking web user session inactivity and bounce rates |

---

## 5 · Architectural Comparison: Spark Streaming vs Apache Flink

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Distributed Stream Engine Matrix                                       │
├─────────────────────┬──────────────────┬───────────────────────────────┤
│ Dimension           │ Spark Streaming  │ Apache Flink                  │
│                     │ (Structured)     │ (Stateful Stream Engine)      │
├─────────────────────┼──────────────────┼───────────────────────────────┤
│ **Processing Model**│ Micro-batching   │ Native continuous streaming   │
│ **Per-Record Latency**│ $100\text{ms} - 500\text{ms}$  │ **$1\text{ms} - 10\text{ms}$ (Sub-millisecond)**│
│ **Throughput**      │ Maximum (vectorized batch)│ High (pipelined memory state) │
│ **Core Philosophy** │ Streaming as small batches│ **Batch as bounded streaming**│
│ **State Backend**   │ HDFS / StateStore│ Embedded RocksDB / Heap state │
│ **Fault Tolerance** │ Micro-batch recomputation│ Asynchronous Barrier Snapshots│
│ **Flow Control**    │ Dynamic PID rate estimation│ Credit-based backpressure   │
│ **Unified API**     │ Mature Catalyst engine │ Blink Planner / Streaming SQL │
└─────────────────────┴──────────────────┴───────────────────────────────┘
```

### 5.1 · Flink's Core Paradigm: Batch as a Special Case of Streaming
Flink inverted Spark's model:
- All real-world datasets are inherently **unbounded continuous streams**.
- Batch processing is merely a degenerate sub-case: an **unbounded stream that happens to have a known beginning and end (bounded stream)**.
- The execution runtime should be built around per-event pipelined dataflows, with batch optimizations layered on top when boundaries are known.

### 5.2 · State Management and End-to-End Exactly-Once
1. **Embedded Local Keyed State**:
   - Operators store keyed counters, lists, and hash maps in local heap memory or embedded RocksDB on TaskManager local disks.
   - Updates are in-place local mutations ($O(1)$ latency), bypassing remote database round-trips.
2. **Asynchronous Barrier Snapshotting (ABS / Chandy-Lamport)**:
   - Checkpoint barriers are injected into source operators and propagate downstream with data records without stalling pipeline execution.
   - Operators snapshot local state to persistent storage (S3 / HDFS) asynchronously when barriers align.
3. **End-to-End Exactly-Once (EOS)**:
   - True end-to-end exactly-once processing requires coordinating three subsystems:
     $$\text{EOS} = \text{Replayable Source (Kafka)} + \text{State Snapshotting (ABS)} + \text{Two-Phase Commit Sink (2PC)}$$

---

## 6 · Industrial Architecture Evolution: Lambda, Kappa, and Lakehouse

```text
[Lambda Architecture]
                        ┌──► Batch Layer (Hadoop / Spark Batch) ──► Batch View (Hive/HDFS) ──┐
                        │                                                                    │
Raw Events (Kafka) ─────┤                                                                    ├──► Serving Layer (Merge)
                        │                                                                    │
                        └──► Speed Layer (Storm / Spark Streaming) ──► Real-Time View (HBase)─┘

[Kappa Architecture]
Raw Events (Kafka Log) ──► Unified Stream Engine (Flink / Spark) ──► Real-Time Store (ClickHouse)
        ▲
        └────── Logic changes: Replay Kafka topic from Offset 0 to build new view

[Lakehouse Streaming Architecture (Modern Standard)]
Kafka Realtime Stream ──► Flink Stream Sink ──┐
                                              ├──► Open Table Format (Apache Iceberg / Paimon)
Batch Historical Data ──► Spark Batch Write ──┘
```

1. **Lambda Architecture Deficiencies**:
   - Requires dual engineering pipelines: one batch codebase (Hive/Spark) and one streaming codebase (Storm/Flink).
   - Inconsistencies between batch and speed layers cause logic divergence and complex reconciliation overhead.
2. **Kappa Architecture Revolution**:
   - Retires the batch layer entirely; all business logic runs on a single streaming engine (Flink).
   - Historical backfills replay the append-only commit log (Kafka / Pulsar / S3) through a parallel streaming job, writing to a new projection before cutting over traffic.
3. **Modern Lakehouse Convergence**:
   - Modern open table formats (Apache Iceberg, Apache Paimon, Delta Lake) support ACID transactions, fast metadata snapshotting, and concurrent read/writes.
   - Stream engines append continuous micro-commits, while batch engines perform deep columnar queries over the exact same underlying storage layer.

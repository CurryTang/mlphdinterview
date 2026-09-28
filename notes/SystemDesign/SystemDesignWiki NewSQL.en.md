# Wiki · NewSQL & Distributed SQL Architecture & Selection

Wiki Navigation: [[SystemDesign00 Overview|00 Blueprint & Numbers]] → [[SystemDesignWiki NewSQL|Wiki · NewSQL]]

In large-scale data infrastructure evolution, engineering teams face a foundational architectural choice when scaling relational data stores: **Sharded RDBMS** (traditional middleware partitioning) versus **NewSQL / Distributed SQL** (native compute-storage decoupling).

```newsql-architecture-visual
```

---

## 1 · Core Architectural Philosophies

### 1.1 · Sharded RDBMS (Partitioned Relational Databases)
- **Definition**: Horizontally partitioning relational databases (e.g., MySQL, PostgreSQL) across independent physical instances using application-tier routing or proxy middleware (e.g., Vitess, ShardingSphere, Citus) based on a chosen Sharding Key.
- **Shared-Nothing Invariant**:
  - Each physical database instance runs independently with its own local storage engine (e.g., InnoDB).
  - The middleware inspects inbound SQL, extracts the sharding key (e.g., `repo_id % 4`), and routes queries to designated target shards.
  - **Advantages**: Raw single-shard queries execute locally on mature storage engines, achieving sub-millisecond latencies with battle-tested operational tooling.
  - **Limitations**: Cross-shard distributed transactions (XA) suffer catastrophic latency penalties; large-tenant data skew cannot be rebalanced dynamically; secondary re-sharding requires painful multi-week migrations.

### 1.2 · NewSQL (Native Distributed SQL)
- **Definition**: Next-generation relational database engines built from scratch to provide elastic horizontal scale-out while preserving **native ACID guarantees and standard SQL dialect compliance** (e.g., TiDB, CockroachDB, Google Spanner, YugabyteDB).
- **Four Foundational Pillars**:
  1. **Compute-Storage Decoupling**:
     - Stateless SQL compute nodes (e.g., TiDB Server) parse SQL, run cost-based query optimization (CBO), and generate parallel execution plans.
     - Distributed storage engines (e.g., TiKV, RocksDB LSM-Tree) guarantee high-availability durability.
  2. **Range-Based Automatic Region Splitting**:
     - Data is mapped into continuous sorted key ranges divided into thousands of small, manageable units called **Regions / Ranges** (e.g., 96MB).
     - When updates push a Region past its size ceiling, it automatically splits into two halves without downtime.
  3. **Multi-Raft Consensus Replication**:
     - Each Region maintains $2F+1$ replicas distributed across distinct physical storage nodes.
     - Independent **Raft** consensus groups elect Raft Leaders for reads/writes, while Followers stream replicated state machines, achieving zero data loss (RPO=0) on node failure.
  4. **Distributed Transactions & Clock Coordination (Percolator 2PC + HLC/TrueTime)**:
     - Implements decentralized two-phase commit (2PC) based on the Google Percolator model to eliminate centralized lock bottlenecks.
     - Utilizes Hybrid Logical Clocks (HLC) or synchronized atomic/GPS clocks (Google TrueTime) to allocate monotonically increasing timestamps for Snapshot Isolation (SI) and linearizable reads.

---

## 2 · Architectural Blueprint Comparison

```text
======================= Path A: Sharded RDBMS (Shared-Nothing) =======================

       [Application Client]
                 │ (SQL Query)
                 ▼
       [Sharding Proxy / Middleware (Vitess / ShardingSphere)]
                 │
                 ├── Routing Hash (repo_id % 3)
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
┌──────────┐┌──────────┐┌──────────┐
│ Shard 0  ││ Shard 1  ││ Shard 2  │
│ (MySQL)  ││ (MySQL)  ││ (MySQL)  │
│ Engine   ││ Engine   ││ Engine   │
└──────────┘└──────────┘└──────────┘
* Bottleneck: Cross-shard aggregations require proxy scatter-gather, pulling raw rows into proxy RAM; re-sharding requires manual dual-write migrations.

======================= Path B: NewSQL (Compute-Storage Decoupled) =======================

       [Application Client]
                 │ (Standard SQL)
                 ▼
┌────────────────────────────────────────────────────────┐
│  Stateless SQL Compute Layer (TiDB / Cockroach Nodes)  │ ──> Stateless, instant horizontal scaling
│  · Distributed Optimizer · Coprocessor Pushdown        │
└────────────────────────────────────────────────────────┘
                 │ (Internal gRPC RPC)
                 ▼
┌────────────────────────────────────────────────────────┐
│  Metadata & Clock Coordinator (PD / HLC / TrueTime)    │ ──> Global timestamp allocation & scheduling
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Distributed Storage Engine (Multi-Raft Groups / TiKV) │
│                                                        │
│  [Storage Node 1]      [Storage Node 2]      [Storage Node 3]
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │ Region 1 (Leader)│  │ Region 1 (Follow)│  │ Region 1 (Follow)│  <- Raft Group 1
│  │ Region 2 (Follow)│  │ Region 2 (Leader)│  │ Region 2 (Follow)│  <- Raft Group 2
│  │ Region 3 (Follow)│  │ Region 3 (Follow)│  │ Region 3 (Leader)│  <- Raft Group 3
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
└────────────────────────────────────────────────────────┘
* Advantages: Filter/Aggregation operators pushed down directly to TiKV Coprocessors; adding storage nodes triggers automatic background Raft rebalancing.
```

---

## 3 · Quantitative Architectural Trade-Off Matrix

| Dimension | Sharded RDBMS (e.g., Vitess + MySQL) | NewSQL (e.g., TiDB / CockroachDB) |
| :--- | :--- | :--- |
| **Underlying Architecture** | Shared-Nothing physical instances behind middleware | Native Compute-Storage Decoupling + Multi-Raft Consensus |
| **Single-Shard Latency Floor**| **Ultra-Low & Predictable (1 - 2 ms)**<br>Direct local B+Tree execution; zero consensus network RTT | **Moderately Higher (3 - 10 ms)**<br>Every commit traverses Raft majority quorum over physical network |
| **Elastic Resharding** | **Painful & Operationally Heavy**<br>Requires dual-writes, binlog catch-up, and table lock cutovers | **Native & Transparent (Zero-Downtime)**<br>Adding nodes triggers automatic Region splitting and Raft replica moves |
| **Cross-Shard Transactions** | **Severe Bottleneck**<br>2PC/XA severely degrades throughput; cross-shard JOINs risk proxy OOM | **Native Pushdown Support**<br>Query optimizer pushes down predicates and aggregations to parallel storage nodes |
| **Data Skew / Hotspots** | **Requires Manual Intervention**<br>Giant tenants saturate single shard disk & CPU; difficult to isolate | **Adaptive Slicing & Balance**<br>Auto-splits large tables into tens of thousands of Regions scattered cluster-wide |
| **Operational Complexity** | **Low / Mature**<br>Decades of production battle testing, proven backup/restore tooling | **High**<br>Demands deep expertise in distributed tracing, Raft consensus, and clock drift |
| **Hardware / Cost Overhead** | **Cost-Efficient**<br>Predictable resource utilization; runs comfortably on modest nodes | **Higher**<br>Multi-replica quorum and LSM-Tree write amplification require NVMe SSDs & 10GbE |

---

## 4 · CI/CD Production Selection Strategy

For large-scale CI/CD platforms handling 50,000 peak QPS and billions of `workflow_run` and `job` records:

### 4.1 · Case for Sharded RDBMS (Partitioned on `repo_id`)
1. **Natural Data Locality**: Over 99% of mission-critical queries are repository-scoped (e.g., query workflow status by commit, update job dependencies within a run).
2. **Zero Cross-Shard Overhead**: Selecting `repo_id` (or `tenant_id`) as the Sharding Key ensures all related tables (`workflow_run`, `job`, `dependency`) reside on the same physical shard. DAG updates execute as **local ACID transactions** without distributed 2PC overhead, delivering peak performance with minimal hardware cost.

### 4.2 · Case for NewSQL
1. **Eliminating the Re-Sharding Apocalypse**: When job archives exceed billions of rows, partitioned MySQL faces disk exhaustion, requiring painful resharding. NewSQL scales dynamically simply by adding commodity storage nodes.
2. **Mitigating Giant Mono-Repo Hotspots**: In enterprise deployments with massive mono-repositories (tens of thousands of engineers pushing commits concurrently), write hotspots overpower single-instance MySQL. NewSQL transparently divides the mono-repo's key ranges into distinct Regions across multiple physical nodes.
3. **Cross-Tenant Analytics & Governance**: SRE dashboards requiring real-time cross-tenant metrics (runner pool utilization, failure trends) execute efficiently via NewSQL parallel coprocessor pushdown.

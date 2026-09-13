# System Design 02 · Database

Course Location: [[SystemDesign01D Redis|01D Redis]] → This Section → [[SystemDesign04 Storage Systems|04 Storage Systems]]

When choosing a database, consider two things first: which business invariants must be atomically satisfied, and what are the system's most critical access paths. The product name comes later.

```text
transaction boundary -> correctness
access pattern       -> data layout and indexes
```

"SQL cannot scale" or "NoSQL has no transactions" are too simplistic. Modern products have overlapping capabilities; the differences lie in default models, transaction boundaries, and scaling costs.

| API | Unit | Example |
|---|---|---|
| SQL | row / txn | Postgres, MySQL |
| KV get/put | key | Dynamo, Redis-as-DB (usually wrong) |
| Document | doc by id | Mongo |
| Wide-column | partition + clustering | Cassandra |
| Graph | vertex/edge walk | Neo4j |

---


## 1 · RDBMS vs NoSQL


## RDBMS: Expressing Relationships and Constraints First

The relational model organizes data into rows and tables, expressing invariants through primary keys, foreign keys, unique constraints, and transactions.

```sql
BEGIN;

UPDATE accounts
SET balance = balance - 100
WHERE account_id = 1 AND balance >= 100;

UPDATE accounts
SET balance = balance + 100
WHERE account_id = 2;

COMMIT;
```

If any step fails, the entire transfer must not leave behind a partial state. 
The costs are direct: cross-node transactions, joins, and global constraints are difficult to scale.


## NoSQL: Organizing Data Around Access Patterns First

NoSQL emphasizes partition-local access.

```text
GetUser(user_id)
ListOrders(user_id, created_at range)
GetFeed(viewer_id, cursor)
```

To ensure a single request hits one partition, data may be denormalized:

```json
{
  "user_id": "u42",
  "profile": {"name": "Kai"},
  "shipping_city": "Seattle"
}
```

Reads become simpler, but writes and consistency costs increase.
If the partition key is chosen incorrectly, hot keys, scans, and cross-partition transactions will emerge.


## Scenarios

| Scenario | Recommendation | Reason |
|---|---|---|
| Orders / Ledger | RDBMS | Dense relationships, strong invariants |
| Profile by `user_id` | KV / Document | Fetched by ID, flexible schema |
| Feed Timeline | Sorted KV | Partitioned by viewer, time-sorted |
| Logs and Events | Log system | High-throughput append, not OLTP |

---


## 2 · Transactions, Locks, 2PC

ACID can be remembered as follows:

| Property | Practical Question |
|---|---|
| Atomicity | Could it be only half-finished? |
| Consistency | Does the constraint still hold after commit? |
| Isolation | What intermediate states will concurrent operations see? |
| Durability | Will the result disappear after a failure once success is returned? |

Define the invariants first, then determine the transaction scope:

```text
order.total == sum(order_items)
payment may be captured at most once
username must be unique
inventory cannot fall below zero
```

Enforcing invariants requires concurrency control:
- **Pessimistic lock**: Blocks concurrent reads/writes; introduces queuing and deadlocks.
- **Optimistic lock**: Uses versioning, validates on commit. Efficient when conflicts are rare.
- **Unique constraint**: Prevents duplication via primary keys, acting as a highly practical idempotency tool. Key choice and in-flight duplicates: [[SystemDesign01 Stateless Service|01]].

When spanning independent nodes:
- **2PC (Two-Phase Commit)**: Uses coordinator and participants. If the coordinator dies after prepare, participants block. Never span untrusted networks or HTTP-to-Stripe calls across 2PC.
- **Saga / Compensation**: Reverts applied steps by executing compensating operations when databases cannot be shared.


## Local vs Workflow
A single-database transaction usually finishes within milliseconds.

```text
local transaction
  -> write order + outbox
  -> async payment command
  -> state transition
  -> compensation when needed
```

See [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]].

---


## 3 · Consistency per API

Consistency must be bound to specific operations. Idempotency in [[SystemDesign01 Stateless Service|01]] keeps invariants true under retry. This axis is: after a successful write, which read sees it.

```text
User updates profile to v2
User immediately reads profile
```

Possible contracts:
- **Linearizable read**: Acts as if there is only one latest copy.
- **Read-your-writes**: The user can read their own writes.
- **Monotonic read**: Once v2 is seen, it will not revert to v1.
- **Eventual consistency**: Replicas eventually converge.

Mix these in the same system: read primary for order confirmation, read replica for public product pages.

---

## 4 · Scalability Cube: The 3D Database Scaling Model (AKF Scale Cube)

A single-node database architecture inevitably hits rigid physical hardware limits (CPU cores, memory bandwidth, disk IOPS, and network throughput). The AKF Scale Cube abstracts horizontal system scalability into an orthogonal 3D geometric space $(X, Y, Z)$:

```text
                  Y-Axis: Functional Partitioning / Decomposition
                  ▲  [Vertical splitting by domain/microservices; isolates teams & blast radius]
                  │
                  │  / Z-Axis: Data Partitioning / Horizontal Sharding
                  │ /   [Horizontal slicing by Shard Key; breaks through write & storage limits]
                  │/
  ────────────────┼────────────────────────► X-Axis: Horizontal Duplication / Replication
                 /│                             [Full cloning/read-write split; linearly scales read QPS & HA]
                / │
```

Scaling along any single dimension eventually encounters physical or operational boundaries. Production-grade, high-throughput systems rely on the orthogonal synergy of all three axes.

```database-scaling-visual
```

---

### 4.1 · X-Axis Scaling: Horizontal Duplication, Primary-Replica, and Multi-Primary

X-axis scaling deploys identical system clones across multiple nodes, where each database instance maintains a 100% complete mirror of the entire dataset. In database architectures, X-axis scaling is divided into two distinct engineering paradigms: **Primary-Replica (Master-Slave)** and **Multi-Primary / Active-Active**.

#### 4.1.1 · Primary-Replica Replication: Mechanisms, Modes, and Tradeoffs

Primary-Replica is the most battle-tested and widely deployed high-availability and read-scaling pattern in industry. Writes converge onto a single Primary, while reads scale out across multiple Replicas.

##### 1. Core Mechanism and the Three Synchronization Protocols
All state mutations (`INSERT`, `UPDATE`, `DELETE`) are strictly routed to a single Primary node. Upon local commit, the primary writes changes to a write-ahead log (WAL / Binlog). Replica nodes pull the log stream across the network asynchronously or semi-synchronously and replay it locally (Relay Log Replay) to converge with the primary's state.

```text
Client Write ──► Primary (Execute & Append Binlog)
                    │
                    ├─► [Network Stream] ──► Replica 1 (Relay Log -> Replay) ──► Client Read
                    └─► [Network Stream] ──► Replica 2 (Relay Log -> Replay) ──► Client Read
```

Depending on tradeoffs between durability and latency, primary-replica employs three distinct synchronization modes:
- **Asynchronous Replication**:
  - *Mechanism*: The primary commits locally to its WAL/binlog and returns success immediately to the client without waiting for replica acknowledgments. Replicas pull and replay logs asynchronously.
  - *Applicable Scenarios*: Read-heavy workloads, latency-critical writes ($< 5\text{ms}$), and systems tolerating brief data loss on sudden crashes.
  - *Tradeoffs*: Maximum throughput and lowest write latency; but unexpected primary crashes cause unpropagated logs to be lost permanently ($\text{RPO} > 0$), and replication lag causes stale reads (violating Read-your-writes).
- **Semi-Synchronous Replication (Lossless Semi-Sync)**:
  - *Mechanism*: The primary suspends client acknowledgment until at least one replica confirms that the log has been received and persisted into its local relay log (Relay Log ACK).
  - *Applicable Scenarios*: E-commerce orders, transaction ledgers, and financial OLTP systems where data loss is unacceptable but a small network RTT penalty is tolerable.
  - *Tradeoffs*: Guarantees zero data loss failover ($\text{RPO} \approx 0$) if at least one replica is healthy; increases write latency by one network round-trip time (RTT); automatically degrades to asynchronous replication if all replicas timeout.
- **Synchronous Consensus Quorum (Raft / Multi-Paxos)**:
  - *Mechanism*: Mutations require durable quorum acknowledgment ($W > N/2$) across the cluster before transaction commit (e.g., TiKV / Spanner Paxos Groups).
  - *Applicable Scenarios*: Financial ledgers, metadata consensus services (etcd, ZooKeeper), distributed NewSQL.
  - *Tradeoffs*: Strict linearizability and automated leader failover ($\text{RPO} = 0, \text{RTO} < 5\text{s}$); write tail latency is bound by the slowest quorum node (straggler), requires an odd number of nodes (3 or 5), and has slightly lower write throughput than asynchronous modes.

##### 2. Benefits and Applicable Scenarios
- **Read-Heavy Workloads (Read/Write ratio $\ge 10:1$ to $1000:1$)**: Product catalogs, social feeds, and dictionary tables benefit from adding read replicas to absorb read traffic;
- **High Availability & Automated Failover**: On primary failure, the freshest replica is promoted to the new primary (guarded by Fencing mechanisms to prevent split-brain; see Section 5);
- **Analytics & Backup Workload Isolation**: Long-running analytical queries (OLAP), ETL pipelines, and logical backups are routed to dedicated replicas to avoid blocking transactional buffer pools.

##### 3. Tradeoffs & Architectural Costs
- **Zero Write Scalability**: All mutations hit the single primary. Adding replicas increases the primary's CPU and network egress overhead due to fan-out log distribution;
- **Linear Storage Cost Multiplication**: Every replica stores a 100% complete dataset copy ($O(N \times \text{Storage})$), failing to alleviate storage capacity ceilings;
- **Replication Lag & Stale Reads**: Asynchronous replication causes stale reads; reading immediately after writing can display outdated state;
- **Connection Pool & Buffer Pool Dilution**: Connection pools multiply per Little's Law ($\text{Concurrency} \approx \text{QPS} \times \text{Latency}$); query fan-out dilutes the memory locality and cache hit ratio of each replica's buffer pool.

---

#### 4.1.2 · Multi-Primary / Active-Active Replication: Conflict Resolution and Tradeoffs

Multi-Primary (Active-Active) allows multiple nodes across different datacenters to accept concurrent write requests, replicating state bidirectionally.

##### 1. The Core Challenge: Concurrent Write-Write Conflicts
In a single-primary architecture, row locks serialize mutations deterministically. In multi-primary architectures, Client A mutates record $X$ on Node 1 while Client B concurrently mutates record $X$ on Node 2. Due to the speed of light and network latency, both nodes accept writes concurrently, creating divergent conflicting states.

##### 2. Conflict Resolution Strategies
- **Pattern 1: Conflict Avoidance via Single-Writer Ownership & Cell Architecture [Production Recommended]**:
  - *Mechanism*: The routing layer statically binds write ownership of each partition/record to a specific primary based on attributes (e.g. user location or `user_id` hash). For example, APAC users write strictly to the Singapore primary, while EU users write strictly to Frankfurt. Primaries replicate cross-region strictly as read-only replicas.
  - *Benefits*: **Eliminates write-write conflicts entirely**, removing the need for distributed cross-datacenter locking;
  - *Costs*: Roaming users require cross-region RPC forwarding to their home primary; ownership handover protocols during failover are complex.
- **Pattern 2: Last-Write-Wins (LWW via NTP or Hybrid Logical Clocks)**:
  - *Mechanism*: Each mutation carries a physical timestamp or HLC. Conflicts are resolved by keeping the write with the largest timestamp, silently overwriting older writes.
  - *Fatal Pitfall*: Cross-node clock skew causes **silent valid data loss**; strictly prohibited in transactional financial applications.
- **Pattern 3: Conflict-Free Replicated Data Types (CRDTs)**:
  - *Mechanism*: Leverages semi-lattice algebraic properties (commutative, associative, idempotent) to guarantee convergence regardless of message arrival ordering. Used for counters (PN-Counters), append-only sets (OR-Sets), and collaborative text editing.
  - *Benefits*: Zero coordination; nodes write locally even during network partitions and converge automatically upon reconnection;
  - *Costs*: Constrained to specialized algebraic types; cannot express relational foreign keys, pessimistic locks, or complex multi-table ACID workflows.
- **Pattern 4: Vector Clocks & Application-Level Merge**:
  - *Mechanism*: Stores capture causal history via vector clocks. Upon detecting concurrent divergent branches, both sibling versions are preserved and surfaced to the application to resolve during the next read (e.g., merging shopping cart items in early Dynamo).
  - *Costs*: Greatly inflates application logic complexity; unmerged siblings consume excess disk and network bandwidth.

##### 3. Engineering Tradeoff Matrix: Primary-Replica vs Multi-Primary

| Dimension | Primary-Replica (Master-Slave) | Multi-Primary (Active-Active) |
|---|---|---|
| **Write Endpoints** | Strictly single writer (Single-Writer) | Multiple concurrent writers (Multi-Writer) |
| **Write Conflicts** | None (serialized via primary row locks) | **Inherent conflicts**; requires ownership or CRDT arbitration |
| **ACID Guarantees** | Full native single-node ACID | **Breaks cross-node ACID**; risks dirty reads & divergence |
| **Write Latency** | Remote writes incur cross-region WAN RTT | **Ultra-low local latency** (writes commit to nearest datacenter) |
| **Auto-Increment IDs** | Simple monotonic increment | Requires distinct offsets/strides (Node 1: 1,3,5; Node 2: 2,4,6) or UUIDs |
| **Replication Loops** | Unidirectional stream | **Bidirectional loop**; requires `server-id` filtering to avoid infinite loops |
| **Adoption Viability** | Recommended for 95%+ of transactional OLTP workloads | Limited to geo-distributed active-active or collaborative editing |

---

### 4.2 · Y-Axis Scaling: Functional Partitioning & Decomposition

Y-axis scaling divides a monolithic database along functional responsibilities, domain boundaries, and bounded contexts into multiple dedicated, physically isolated databases.

#### 1. Core Mechanism
Driven by Domain-Driven Design (DDD), a shared monolithic database is vertically partitioned into isolated, service-owned databases:
$$\text{Monolithic DB} \longrightarrow \text{User DB} \oplus \text{Order DB} \oplus \text{Inventory DB} \oplus \text{Payment DB}$$
Under the Database-per-Service pattern, direct cross-database SQL queries and joins are prohibited. All cross-domain interactions occur strictly via well-defined RPC/REST APIs or asynchronous message event streams (Event-Driven Architecture).

#### 2. Applicable Scenarios
- **Divergent Workloads & SLA Requirements**: The catalog database is read-intensive and benefits from heavy secondary caching/search indexes, whereas the payment ledger requires strict row-level pessimistic locking and ACID guarantees. Vertical decomposition allows tailored database engines and parameters per domain;
- **Organizational Scaling & Conway's Law**: When engineering teams grow to hundreds of developers, shared monolithic tables lead to schema migration lockouts, deployment conflicts, and ambiguous ownership. Domain-partitioned databases enable independent deployment, evolution, and scaling;
- **Blast Radius Containment**: A runaway slow query, deadlock, or connection pool exhaustion in non-critical modules (e.g., user reviews, comments, gamification) is strictly confined to its own database and cannot bring down core ordering, payment, or auth pipelines.

#### 3. Tradeoffs & Engineering Costs
- **Loss of Single-Node ACID Transactions**: Workflows spanning multiple domains (e.g., "Place Order $\to$ Deduct Inventory $\to$ Debit Account") cannot rely on a single atomic `BEGIN ... COMMIT`. Teams must adopt distributed consistency paradigms:
  - Two-Phase Commit (2PC / XA): Strong consistency at the cost of high latency, distributed locking, coordinator single points of failure, and severe throughput degradation;
  - Eventual Consistency (Saga Pattern / Transactional Outbox / Local Message Tables): Forward steps executed via state machines, with compensating transactions on failure. Substantially escalates design, testing, and financial reconciliation complexity;
- **Loss of Relational SQL JOINs & Foreign Keys**: Cross-entity joins cannot execute within the database engine.
  - *Mitigations*: Application-side joining, denormalized data redundancy, or streaming changes via CDC (e.g., Debezium) into dedicated read models (CQRS / Elasticsearch / ClickHouse);
- **Infrastructure Proliferation & Operational Overhead**: Multiplies database instances, connection pools, backup/restore pipelines (RPO/RTO tracking), monitoring alerts, and security patch cycles;
- **Single-Module Physical Bottlenecks Persist**: If a single domain table (e.g., Orders or Messaging) exceeds billions of rows or tens of thousands of write QPS, Y-axis decomposition reaches its limit and must be paired with Z-axis sharding.

---

### 4.3 · Z-Axis Scaling: Data Partitioning & Horizontal Sharding

Z-axis scaling preserves the exact schema while splitting a homogeneous dataset horizontally into $K$ discrete physical partitions (Shard $0 \dots K-1$) based on a specific attribute (the Shard Key / Partition Key).

#### 1. Core Mechanism
Each shard holds only $1/K$ of the total dataset and independently handles approximately $1/K$ of read/write throughput. An intelligent routing tier (e.g., Vitess, ShardingSphere, or distributed storage engine proxies) parses SQL statements, extracts the shard key, and routes requests to the target shard:

```text
                               ┌──► Shard 0 (Hold keys: hash(key) % 4 == 0)
Client Query ──► Query Router ──┼──► Shard 1 (Hold keys: hash(key) % 4 == 1)
   (with Shard Key)            ├──► Shard 2 (Hold keys: hash(key) % 4 == 2)
                               └──► Shard 3 (Hold keys: hash(key) % 4 == 3)
```

- **Partitioning Strategies**:
  - **Hash-Based**: Uses `hash(shard_key) % N` or consistent hashing rings. Guarantees uniform distribution but sacrifices range-scan efficiency;
  - **Range-Based**: Partitions by time (`created_at`) or ID ranges (`[1, 10000000]`). Facilitates range queries, but latest sequential writes cluster onto the newest shard, creating write hot-spots;
  - **Directory / Lookup Table**: Centrally maps entity IDs to shard locations. Highly flexible for custom routing, but introduces an extra network hop and availability dependency.

#### 2. Applicable Scenarios
- **Write QPS Exceeds Single-Node Hardware Limits**: When write throughput reaches tens or hundreds of thousands of QPS, saturating single-primary CPU, row locks, and SSD IOPS, Z-axis sharding is the sole mechanism to scale write capacity linearly;
- **Dataset Exceeds Single Disk Capacity & B+ Tree Depth**: Tables exceeding tens or hundreds of millions of rows push B+ tree depths to 4–5 levels, amplifying random disk I/O per query. Total storage exceeds affordable SSD sizes and causes backup/restore windows (RTO) to exceed SLA tolerances;
- **Multi-Tenant SaaS Isolation**: Partitioning by `tenant_id` isolates large enterprise tenants on dedicated shards ("noisy neighbor" prevention) while pooling smaller tenants;
- **Data Sovereignty & Geo-Distributed Latency**: Partitioning by user region ensures EU citizen data resides strictly on EU-based infrastructure (GDPR compliance) while routing users to nearby datacenters to minimize optical round-trip times (RTT).

#### 3. Tradeoffs & Engineering Costs
- **Shard Key Lock-In & Query Dimension Collapse**:
  - **Point Queries**: Queries containing the shard key (`WHERE user_id = 1024`) execute with $O(1)$ direct routing;
  - **Non-Key Queries Degrade to Scatter-Gather**: Queries lacking the shard key (e.g., lookup by phone number or cross-tenant merchant queries) must broadcast to all $K$ shards concurrently. Results are merged, sorted, and paginated in the router's memory. Overall latency is determined by the slowest straggler shard (tail latency amplification);
- **Cross-Shard Distributed Transactions (Cross-Shard 2PC)**: Atomically updating multiple shards requires 2PC / XA protocols. Network round-trips and prolonged lock durations severely degrade throughput by 1–2 orders of magnitude;
- **Data Skew & Hot Shards**: Real-world data is rarely uniform. High-profile accounts (influencers) or viral products concentrate massive traffic onto individual shards, creating system-wide performance bottlenecks;
- **Resharding & Data Migration Complexity**: Scaling from $N$ to $2N$ shards requires rebalancing historical data across the network, running dual-write reconciliation, and executing zero-downtime routing cutovers.

---

### 4.4 · The 3D Convergence ($X \times Y \times Z$ Synergy) & Architectural Decision Matrix

#### 1. Real-World Industrial Topology
In hyperscale distributed systems, scaling is never an isolated choice; architectures converge along all three dimensions:

```text
                       [ Ingress Traffic / API Gateway ]
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            │  Y-Axis: Functional Decomposition (Microservices)   │
            ▼                                                     ▼
     [ User Service ]                                     [ Order Service ]
            │                                                     │
            │ (Moderate volume)                                   ▼ (Hyperscale writes/data)
            │                                     ┌───────────────────────────────┐
            │                                     │ Z-Axis: Sharding (user_id % 2)│
            │                                     └───────┬───────────────┬───────┘
            │                                             │               │
            │                                   Shard 0   ▼               ▼   Shard 1
            ▼                                  ┌────────────────┐ ┌────────────────┐
  ┌──────────────────┐                         │ X-Axis: Replicas│ │ X-Axis: Replicas│
  │ X-Axis: Replicas │                         │ Primary (Write)│ │ Primary (Write)│
  │ Primary (Write)  │                         │ ├─ Replica (R) │ │ ├─ Replica (R) │
  │ └─ Replica (R)   │                         │ └─ Replica (R) │ │ └─ Replica (R) │
  └──────────────────┘                         └────────────────┘ └────────────────┘
```

1. **Step 1 (Y-Axis)**: Vertically decompose the monolith into domain-driven microservice databases, isolating team lifecycles and blast radiuses;
2. **Step 2 (Z-Axis)**: For high-volume, write-intensive domains (e.g., Orders, Messaging), horizontally shard data by an optimal Shard Key to eliminate write IOPS and disk limits;
3. **Step 3 (X-Axis)**: Within each shard, deploy a Primary with multiple Replicas for read-write splitting and automated failover.

#### 2. Architectural Tradeoff Matrix

| Dimension | X-Axis: Duplication (Replicas) | Y-Axis: Decomposition (Services) | Z-Axis: Partitioning (Sharding) |
|---|---|---|---|
| **Core Mechanism** | Full cloning (100% data per node) | Vertical split by domain boundaries | Horizontal slicing by Shard Key |
| **Primary Bottleneck Solved** | Read QPS, Single Point of Failure (HA) | Team merge friction, divergent workloads, blast radius | Single-node write IOPS, disk volume, B+ tree depth |
| **Write Scaling** | **None** (single primary bottleneck) | **Indirect** (spread across service DBs) | **Linear** (writes distributed across $K$ shards) |
| **Storage Scaling** | **None** (each node stores 100% data) | **Domain-bound** (limited per domain) | **Linear** (each shard stores $1/K$ data) |
| **Transaction Impact** | Retains single-node ACID; asynchronous replication risks stale reads | **Breaks local ACID**; requires Saga / Outbox eventual consistency | **Cross-shard 2PC bottleneck**; intra-shard ACID preserved |
| **Query Impact** | None; full SQL, JOINs, and secondary indexes supported | **Breaks cross-service SQL JOINs**; requires CQRS or app joins | **Point queries fast**; non-key queries degrade to scatter-gather |
| **Primary Tradeoffs** | Replication lag, stale reads, buffer pool dilution | Distributed transaction complexity, schema/infra proliferation | Shard key lock-in, scatter-gather tail latency, resharding overhead |
| **Adoption Milestone** | High read/write ratio ($\ge 10:1$), need for automated failover | Organizational scaling, divergent domain workloads | Write IOPS saturation, tables exceeding tens of millions of rows |

---

### 4.5 · Data Dependency Driven Architecture Selection

In distributed database architecture, **internal data dependencies and invariant boundaries dictate the physical feasibility of scaling paths**:

#### 1. Strong Transactional Invariants
- **Characteristics**: Indivisible atomic invariants spanning multiple rows (e.g., $\Delta A + \Delta B = 0$ in banking ledgers, inventory deductions, foreign key cascades). Demands **strict Atomicity and Isolation (Snapshot Isolation / Serializable)**.
- **Architectural Solution**:
  - Keep within a single high-performance RDBMS or natively distributed SQL engine (Google Spanner, TiDB);
  - When vertically partitioning across microservices (Y-Axis), downgrade to **eventual consistency**: use local transactions to update business records and record a transactional outbox table, then propagate asynchronously via Kafka using **Saga state machines with compensating actions**.
- **Fatal Anti-Pattern**: Blindly sharding horizontally (Z-Axis) by `user_id` without decoupling invariants. Inter-user transfers trigger cross-shard Two-Phase Commit (2PC) on every request, holding distributed locks across WANs and causing catastrophic throughput collapse.

#### 2. Entity Colocation & Aggregate Affinity
- **Characteristics**: Sub-entities bound entirely to a parent aggregate root, with 95%+ of queries fetching or modifying parent and child together (e.g., `orders` and `order_items`, `users` and `user_preferences`).
- **Architectural Solution**:
  - **Colocated Sharding**: Enforce identical sharding keys across parent and child tables (e.g., child table redundantly stores `order_id` as its sharding key). The routing proxy ensures all sub-entities land on the exact same physical shard;
  - **Document Embedding**: In document stores (MongoDB), embed child arrays directly inside the parent document.
- **Core Benefit**: Intra-shard queries retain **native local ACID transactions and joins with zero network hops**, completely eliminating 2PC.

#### 3. Weak / Derived / Analytics Streams
- **Characteristics**: High-throughput updates where business workflows do not depend on real-time statistical perfection (e.g., short URL click counts, social post likes, telemetry counters, trending feeds).
- **Architectural Solution**:
  - **CQRS Physical Decoupling**: Core mutation commits or enqueues an asynchronous event to Kafka and acknowledges the client immediately;
  - Downstream stream processors consume events using **micro-batching** to update read models (Redis caches or ClickHouse OLAP stores).
- **Fatal Anti-Pattern**: Synchronously executing `UPDATE counters SET count = count + 1` directly on the transactional primary database path, causing severe row-lock contention and crashing OLTP throughput.

#### 4. Stateless / Static Configuration
- **Characteristics**: Read-intensive, near-zero writes, globally referenced across all system domains (e.g., postal codes, currency exchange benchmarks, platform fee schedules).
- **Architectural Solution**:
  - **Broadcast Replication**: Replicate the static dictionary table completely across every shard database;
  - **In-Memory L1 Cache**: Prewarm dictionary tables into application memory (e.g., Caffeine / Guava Cache); invalidate via broadcast pub/sub on administrative updates.

---

### 4.6 · Sharding Key Design: 6 Critical Factors & Production Pitfalls

The Sharding Key is the cornerstone of horizontal data partitioning. Choosing the wrong key forces destructive full-scale data re-sharding migrations later:

#### 1. High Cardinality
- **Rule**: Sharding keys must have a massive value domain (cardinality $\gg$ number of shards, e.g., millions of unique `user_id`s or `UUID`s).
- **Pitfall**: Never shard by low-cardinality status columns (`status: [0, 1, 2]`) or enums (`gender`, `country_code`). Doing so clusters tens of millions of records onto 2–3 shards, causing severe data skew while other shards sit idle.

#### 2. Uniform Distribution & Hotspot Avoidance
- **Rule**: Writes must distribute across all shards with pseudo-random uniformity.
- **Pitfall (Sequential ID & Timestamp Trap)**:
  - Sharding by sequential auto-increment IDs or timestamps (`created_at`) routes 100% of current incoming writes to the single latest shard, creating an extreme **Write Hotspot** while historical shards remain dormant;
  - Production Standard: Apply pseudo-random hashing (`MurmurHash3(key) % N`). For viral celebrities or mega-merchants, append a salted suffix (`key + "_" + random(0, M)`) to scatter traffic across shadow shards.

#### 3. Query Filter Alignment
- **Rule**: **Over 85%–90% of core business queries must include the sharding key in their `WHERE` clause**.
- **Pitfall (Scatter-Gather Broadcast Amplification)**:
  - Queries with the sharding key execute as $O(1)$ point queries routed to a single node;
  - Queries lacking the sharding key force the proxy to broadcast to all $K$ shards concurrently (**Scatter-Gather**), sorting and paginating in proxy memory. Tail latency degrades to the slowest straggler node, exhausting connection pools and risking router out-of-memory crashes.

#### 4. Colocation & Transaction Affinity
- **Rule**: Closely coupled parent-child entities must share the same sharding key.
- **Practice**: Both `orders` and `order_items` use `order_id` as their sharding key, co-locating records onto the same physical node to preserve local ACID transactions and joins without 2PC.

#### 5. Key Immutability
- **Rule**: **Once created, a sharding key must be permanently immutable**.
- **Pitfall**: Never use mutable business attributes (`phone_number`, `email`, `department_id`) as sharding keys. Mutating a sharding key triggers an expensive distributed operation: "delete on old shard + insert on new shard", which easily deadlocks concurrent transactions and risks data loss.

#### 6. Multi-Dimensional Query Strategies
When applications require queries across two competing dimensions (e.g., buyers querying by `buyer_id` and merchants querying by `seller_id`), use these industry-standard patterns:

```text
                              [ Client Write Request ]
                                         │
                                         ▼
                     ┌────────────────────────────────────────┐
                     │ Buyer Primary Cluster (by buyer_id)    │
                     │ (order_id embeds buyer_id hash suffix) │
                     └───────────────────┬────────────────────┘
                                         │ (Binlog / WAL)
                                         ▼
                              [ CDC Streaming (Debezium) ]
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
 ┌──────────────────────────────────────┐ ┌──────────────────────────────────────┐
 │ Merchant Shadow Cluster (by seller_id)│ │ Elasticsearch / ClickHouse Search DB │
 │ (Handles merchant dashboard queries) │ │ (Handles multi-filter admin reports) │
 └──────────────────────────────────────┘ └──────────────────────────────────────┘
```

- **Pattern A: Gene Sharding**:
  - When generating `order_id`, embed the lower 4–6 bits of `hash(buyer_id)` as the suffix of `order_id`;
  - Queries by `buyer_id` route directly to the target shard; queries by `order_id` extract the embedded suffix to locate the identical shard without cluster broadcast.
- **Pattern B: CDC-Driven CQRS Shadow Tables**:
  - Transactional writes commit solely to the buyer-sharded primary cluster;
  - Change Data Capture (CDC) streams Binlog events to construct a merchant-sharded read replica or Elasticsearch index, fully decoupling complex analytical queries from the transactional write path.

---

### 4.7 · Post-Sharding Query Routing: The Reality of Scatter-Gather and 4 Alternative Architectural Patterns

#### 1. Core Question: Is Scatter-Gather Mandatory After Sharding?
**The answer is an emphatic NO: Scatter-Gather is NOT mandatory after sharding!**

In a properly architected sharded system, **the vast majority of high-frequency production queries completely bypass Scatter-Gather**:
- **Single-Shard Point Queries (The Ideal State)**: Whenever a query includes the Sharding Key in its filter (e.g., `WHERE user_id = 1024`), the routing tier hashes the key and forwards the request strictly to a single physical shard. The time complexity is $O(1)$, and latency matches a single-node database.
- **When Does Scatter-Gather Occur?**: Scatter-gather is triggered **only when** a query **lacks the sharding key** (e.g., looking up a user by phone `WHERE phone_number = ?` or a merchant looking up store orders `WHERE seller_id = ?`), or when performing unindexed **global aggregations** (`COUNT(*)`) or **deep pagination** (`ORDER BY created_at LIMIT 10000, 20`).

```text
【Single-Shard Point Query (Ideal)】         【Cluster Scatter-Gather Broadcast】
Client (with Shard Key: user_id=42)         Client (Missing Shard Key: phone='138...')
          │                                           │
          ▼                                           ▼
    Query Router                                Query Router (Fan-out to ALL Shards)
          │                                      ┌────┼────┬────┐
          │ (Direct O(1) Route)                  ▼    ▼    ▼    ▼
          ▼                                    [S0] [S1] [S2] [S3]
       Shard 2                                   └────┴────┼────┘
          │                                                ▼
          ▼                                         Merge Sort & Limit
     Fast Result                                   (Tail Latency Risk!)
```

#### 2. Why is Scatter-Gather a System Bottleneck? (Tradeoffs & Pitfalls)
1. **Tail Latency Amplification**:
   - The query fans out to all $K$ shards concurrently; total response latency is bound by the **slowest straggler node**:
     $$T_{\text{total}} = \max(T_0, T_1, \dots, T_{K-1})$$
   - If a single shard has just a $p = 1\%$ probability of an ephemeral latency spike (GC pause, disk queue), with $K = 64$ shards, the probability that the client experiences a tail spike is:
     $$P(\text{Tail Spike}) = 1 - (1 - p)^K = 1 - (1 - 0.01)^{64} \approx 47.44\%$$
     Nearly half of all broadcast requests suffer tail latency degradation!
2. **Proxy Memory Exhaustion & Network Fan-In (Deep Pagination OOM)**:
   - For `SELECT * FROM orders ORDER BY create_time LIMIT 10000, 20`, the router cannot fetch only 20 rows per shard. Every shard must locally scan and return its top 10,020 rows;
   - The proxy must ingest and merge-sort $K \times 10,020$ records in memory before discarding the first 10,000, causing severe memory spikes and router OOM crashes under high concurrency.
3. **Connection Pool Starvation**:
   - A single scatter-gather query simultaneously occupies a database connection on every shard, slashing cluster-wide concurrency by a factor of $K$.

---

#### 3. The 4 Production Architectural Patterns to Eliminate Scatter-Gather

To eliminate or mitigate scatter-gather, distributed architectures employ four primary design patterns:

```text
                       [ Non-Shard-Key Query Request ]
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            ▼                         ▼                         ▼
   [ Pattern 1: Global Index ]   [ Pattern 2: Gene Sharding ] [ Pattern 3: CQRS Read Model ]
   Query phone_number GSI to     order_id embeds user bits;   CDC streams to Elasticsearch
   fetch user_id, then point     extract suffix to route to   or seller shadow tables for
   query main shard (O(1))       identical shard directly     complex multi-field filtering
```

##### Pattern 1: Global Secondary Index (GSI) & Reverse Lookup Table
- **Mechanism**:
  - Build a lightweight secondary index table, partitioned by the secondary lookup attribute (e.g., `phone_number` or `email`), where the record value simply points to the primary shard key (`user_id`);
  - **Two-Step Point Query**:
    1. Step 1: Query GSI: `SELECT user_id FROM phone_index WHERE phone = '...'` $\to$ routes to the specific GSI shard ($< 2\text{ms}$);
    2. Step 2: Query Primary: `SELECT * FROM users WHERE user_id = 42` $\to$ routes directly to Shard 42 ($< 2\text{ms}$).
- **Applicable Scenarios**: High-frequency equality lookups by unique non-sharding attributes (login by phone/email, transaction reference numbers).
- **Tradeoffs**:
  - Adds one network RTT (mitigated via router-side caching of hot mappings);
  - Introduces dual writes on creation (primary table + index table); typically reconciled via asynchronous eventual consistency with millisecond-level propagation delay.

##### Pattern 2: Gene Sharding (Embedded Hash Routing)
- **Mechanism**:
  - Embed the hash characteristics of a secondary query key into the lower bits of the entity's generated primary key;
  - Example: In e-commerce, extract the lower 6 bits of `hash(buyer_id)` and embed them as the suffix of `order_id`;
  - **Zero-Broadcast Dual-Key Routing**:
    1. Query by `buyer_id`: `hash(buyer_id) % 64` routes directly to Shard 44;
    2. Query by `order_id`: Extract the embedded 6-bit suffix (`101100` = 44) to locate Shard 44 **without cluster broadcast or secondary lookups**!
- **Applicable Scenarios**: Two strongly correlated high-frequency access keys (e.g. Orders accessed via `order_id` and `buyer_id`).
- **Tradeoffs**:
  - Requires custom ID generation schemes;
  - Solves exactly two correlated keys; cannot generalize to arbitrary multi-attribute query filters.

##### Pattern 3: CQRS Heterogeneous Read Models & CDC Streaming (Elasticsearch / Shadow Shards)
- **Mechanism**:
  - **Physical Decoupling of Writes and Reads**: Transactional writes commit strictly to the buyer-sharded primary cluster;
  - Change Data Capture (CDC, e.g. Debezium / Flink) captures Binlog streams asynchronously to populate specialized read stores:
    - *Seller Shadow Shards*: A secondary relational cluster sharded by `seller_id` for merchant dashboards;
    - *Elasticsearch*: Inverted indexes supporting arbitrary multi-condition faceted searches;
    - *ClickHouse*: Columnar store for cross-cluster analytical aggregations and BI reporting.
- **Applicable Scenarios**: Merchant management portals, multi-filter administrative dashboards, and heavy analytical reporting.
- **Tradeoffs**:
  - Eventual consistency lag (milliseconds to seconds between write commit and read visibility);
  - Increases operational complexity with CDC streaming pipelines and heterogeneous database clusters.

##### Pattern 4: Pushdown Aggregation & Stream Precomputation
- **Mechanism**:
  - **Pushdown Aggregation**: When global aggregations (`COUNT(*)`) are unavoidable, the proxy pushes computation down to each shard to compute local counts; the router aggregates only $K$ scalar integers (Tree-based Hierarchical Merge);
  - **Stream Precomputation**: Real-time aggregations (metrics, counters) are continuously computed via Flink streaming and maintained in Redis counters or HyperLogLog structures for $O(1)$ reads;
  - **Cursor-Based Pagination (Keyset Pagination)**: Replace offset-based pagination (`LIMIT offset, size`) with keyset pagination (`WHERE (create_time, id) < (?, ?) ORDER BY create_time DESC, id DESC LIMIT 20`), allowing shards to perform index-bounded scans without deep offsets.
- **Applicable Scenarios**: Platform dashboards, real-time leaderboards, and infinite-scrolling feeds.
- **Tradeoffs**:
  - Relaxes real-time transactional consistency;
  - Cursor pagination disallows random page jumping (e.g., jumping directly to page 500).

---

#### 4. Post-Sharding Query Solutions: Architectural Decision Matrix

| Solution | Mechanism | Query Complexity | Consistency Model | Best Suited For | Key Tradeoffs & Limitations |
|---|---|---|---|---|---|
| **Global Secondary Index (GSI)** | Sharded index table mapping attribute to Shard Key | Two $O(1)$ Point Queries | Eventual (or 2PC for strong) | High-frequency single-key lookups (phone/email login) | 1 extra network RTT; write amplification; index sync lag |
| **Gene Sharding** | ID embeds suffix hash bits of secondary key | Single $O(1)$ Point Query | Strict Local Consistency | Core parent-child keys (e.g. `order_id` & `buyer_id`) | Rigid ID generation scheme; restricted to two correlated keys |
| **CQRS Heterogeneous Models (CDC+ES)** | CDC streams Binlogs to inverted/columnar stores | $O(\log N)$ faceted search | Eventual Consistency | Merchant dashboards, multi-field filters, admin search | Read replication lag; multi-cluster operational overhead |
| **Pushdown & Stream Precomputation** | Push aggregations to shards + Keyset Pagination | $O(K)$ Merge / $O(1)$ Cache | Eventual / Approximate | Global KPI dashboards, leaderboards, infinite scroll | Cannot jump to arbitrary pages; approximate counts |

---

## 5 · Survive a failure

Redundancy is more than "launching a few more machines." A high-availability database must preserve core invariants across network partitions, hardware faults, and split-brain scenarios.

### 5.1 · Failure Domains
```text
process
  < machine
  < rack / power domain
  < availability zone (AZ)
  < region
```
Determine the SLA target upfront: **Which failure level must the system survive without human intervention?**
- Node failure: Same-rack standby takeover;
- Datacenter outage: Multi-AZ synchronous/semi-synchronous replication;
- Regional disaster: Multi-Region asynchronous disaster recovery.

### 5.2 · Failover and Fencing
Promoting a replica to become the new primary safely is one of the most perilous operations in distributed systems:
```text
1. Failure detector suspects primary is unavailable (Timeout ≠ Death)
2. Lease expiry threshold reached
3. Consensus mechanism selects the freshest replica (highest GTID / LSN)
4. Fence the old primary (Fencing / STONITH)
5. Promote new primary, increment global Epoch / Term
6. Router configuration cutover
7. Restore write traffic
```

> [!IMPORTANT]
> **Fencing is the inviolable baseline for data integrity**: An old primary may merely be experiencing a prolonged GC pause or unilateral network partition. Without definitive fencing (e.g., etcd/ZooKeeper lease revocation, storage-level write barrier, or IPMI power cutoff STONITH), the old primary might resume processing writes upon reconnection, triggering catastrophic **split-brain** and silent data corruption.

### 5.3 · Deployment Modes
- **Active-Passive**:

| Standby Mode | Steady State | Failover Speed (RTO) | Infrastructure Cost |
|---|---|---|---|
| **Cold Standby** | Offline; periodic snapshots & provisioning scripts only | Hours to days | Minimal |
| **Warm Standby** | Instance running and replaying logs, downscaled or cold cache | Tens of seconds to minutes | Moderate |
| **Hot Standby** | Full capacity online, real-time log sync, warm buffer pool | Seconds (automated) | High (100% redundancy) |

- **Active-Active**:
  - Both clusters accept concurrent writes. Resolving same-row write conflicts across high-latency WANs is notoriously intractable (Last-Write-Wins loses updates; CRDTs are constrained to commutating mutations). In practice, systems enforce **Single-Writer Ownership (Geo-partitioning by user ID)** to avoid cross-region distributed locking;
- **Quorum Consensus (N/W/R)**:
  - Configuring write quorum $W$, read quorum $R$, and replica count $N$ such that $W + R > N$ guarantees read-write set intersection. Note that Quorum alone does not guarantee linearizability without an explicit state machine consensus protocol (Raft/Paxos).

### 5.4 · Replicas Are Not Backups
- **Core Distinction**: Replication provides high availability (Availability) to survive hardware crashes within seconds; however, replication propagates malicious drops (`DROP TABLE`), application bugs, and data corruptions to all replicas within milliseconds.
- **Backups Provide Durability and Recoverability**: Regular cold snapshots and immutable WAL log archives enable Point-in-Time Recovery (PITR) to roll state back to a known healthy timestamp.
- **Key Disaster Recovery Metrics**:
  - **RPO (Recovery Point Objective)**: The maximum acceptable data loss window during a disaster (e.g., $\text{RPO} \le 1\text{ min}$);
  - **RTO (Recovery Time Objective)**: The maximum acceptable downtime before service restoration (e.g., $\text{RTO} \le 5\text{ min}$).

---

## 6 · Architectural Decision Matrix

| Business Requirement | Recommended Starting Architecture | Critical Tradeoffs & Engineering Costs |
|---|---|---|
| **Read-Heavy Workload** | Primary + Read Replicas (X-Axis) | Must tolerate replication lag; pin immediate post-write reads to primary |
| **Write-Heavy Workload** | Horizontal Sharding (Z-Axis) | Incurs scatter-gather queries, cross-shard 2PC, and resharding overhead |
| **Single-Row Invariants** | Distributed KV / Document Store | Denormalized data model; application manages referential integrity |
| **Complex Cross-Row Invariants** | Relational Database (RDBMS) / Spanner | Harder to scale; must accept distributed lock contention or single-node limits |
| **Multi-Team Scale & Heterogeneous Workloads** | Functional Partitioning (Y-Axis) | Breaks single-node ACID; necessitates Saga/Outbox reconciliation |
| **Global Low-Latency Writes** | Geo-Partition Ownership | Strict data ownership boundaries required; cross-region writes prohibited |

---

## 7 · Primary Sources & References

- [PostgreSQL Documentation: High Availability, Load Balancing, and Replication](https://www.postgresql.org/docs/current/high-availability.html)
- [MySQL 8.0 Reference Manual: Replication](https://dev.mysql.com/doc/refman/8.0/en/replication.html)
- [MongoDB Manual: Sharding Architecture](https://www.mongodb.com/docs/manual/sharding/)
- [Dynamo: Amazon's Highly Available Key-value Store (SOSP 2007)](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf)
- [Cassandra - A Decentralized Structured Storage System (LADIS 2009)](https://www.cs.cornell.edu/projects/ladis2009/papers/lakshman-ladis2009.pdf)
- [Spanner: Google's Globally-Distributed Database (OSDI 2012)](https://static.googleusercontent.com/media/research.google.com/en//archive/spanner-osdi2012.pdf)
- [Vitess: Scalable Database Clustering System for MySQL](https://vitess.io/)

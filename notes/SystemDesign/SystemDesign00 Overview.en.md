# System Design 00 · Architectural Blueprint & Quantitative Estimation Numbers

Course Location: This Section (Master Blueprint & Numbers) → [[SystemDesign01 Stateless Service|01 Stateless Service]]

System design is not a disjoint collection of components; it is a discipline of **making principled architectural tradeoffs under explicit business invariants, SLA availability targets, and hard physical hardware boundaries as bottlenecks shift**.

In production-grade systems architecture, theoretical designs divorced from **quantitative estimation (Numbers & Capacity Sizing)** are ungrounded fantasies. This overview establishes the **four fundamental physical scaling constants** and maps the complete evolutionary storyline of distributed systems.

---

## 1 · Critical Quantitative Constants Every Architect Must Master

### 1.1 · Application Server: When is QPS Considered High?

The QPS capacity of a single stateless application server (standardized on a 4-core / 8GB to 8-core / 16GB computing Pod) follows clear industrial tiers:

| QPS per Node | System State | Typical Scenarios & Engineering Characteristics |
|---|---|---|
| **10 QPS / node** | Minimal Load | Completely acceptable; common in internal admin dashboards, low-frequency microservices, or batch computing endpoints. |
| **100 QPS / node** | Casual / Easy | The steady-state operating zone for most microservices; CPU utilization typically remains $< 15\%$. |
| **500 QPS / node** | **Normal Working Zone** | The healthy sweet spot for microservices (JSON deserialization, auth token validation, 1–2 DB roundtrips, and network IO). |
| **2,000 QPS / node** | **Optimization Threshold** | Active profiling required; monitor GC pauses, thread pool queue depths, connection pool contention, and CPU context switching. |
| **> 2,000 QPS / node** | **High Load** | Approaching the limit for standard business logic; requires L2 in-memory caching, async batching, or zero-reflection serializers. |
| **> 10,000 QPS / node** | **Extreme High Throughput** | **Rare in general application services**. Requires: ultra-simple logic, pure memory caching, non-blocking IO (epoll / netpoll), persistent connection pooling, and zero heavy ORM/reflection overhead. |

> [!IMPORTANT]
> **The Universal Capacity Saturation Rule (The +30% Knee-of-the-Curve Rule)**:
> In load testing or live monitoring, when **incoming QPS increases by just 30%**, if you observe:
> 1. A sharp, non-linear explosion in p95 / p99 latency (the latency knee);
> 2. Sudden CPU spikes;
> 3. Substantial queue buildup in worker thread pools;
> 
> The system has crossed its saturation knee point. Horizontally adding application nodes will fail; the bottleneck has shifted down into the persistence and storage layers!

---

### 1.2 · Cache: QPS & Capacity Numbers

#### 1. Remote Cache QPS
For simple `GET` / `SET` operations on remote distributed caches (Redis / Memcached):
- **Round-Trip Time (RTT)**: Local datacenter network RTT is $\approx 0.2 - 1.0\text{ ms}$;
- **Single-Node Capacity**: A single instance comfortably handles **tens of thousands of QPS** (CPU core bound and network interface packet-per-second bound).

**Estimation Formula via CPU Processing Time**:
$$\text{QPS} \approx N_{\text{cores}} \times \frac{1000}{t_{\text{cpu}}} \times u$$
Where:
- $N_{\text{cores}}$ is the number of dedicated compute cores;
- $t_{\text{cpu}}$ is the pure CPU compute time per request in milliseconds;
- $u$ is the target safe CPU utilization ($0.7 - 0.8$, reserving $20\% - 30\%$ headroom for traffic bursts).

> **Calculation**: A 4-core instance with average CPU execution time $t_{\text{cpu}} = 0.1\text{ ms}$ at $u = 0.8$:
> $$\text{QPS} \approx 4 \times \frac{1000}{0.1} \times 0.8 = 32,000\text{ QPS}$$
> A 4-core Redis instance comfortably sustains **~32,000 QPS** in steady state.

#### 2. Cache Capacity: How Many Keys Fit in 1 GB RAM?
- **Theoretical Limit**:
  For an average Key-Value pair plus metadata size of $\approx 200\text{ B}$:
  $$\text{Keys} = \frac{10^9\text{ Bytes}}{200\text{ Bytes}} \approx 5,000,000\text{ (~5 million keys)}$$
- **Production Conservative Baseline**:
  Accounting for allocator fragmentation (jemalloc), pointer overhead (`dictEntry` and `redisObject`), and reserving $30\%$ memory headroom (preventing Copy-on-Write OOM during BGSAVE or AOF rewrites):
  $$\mathbf{1\text{ GB RAM} \approx 2,000,000 - 3,000,000\text{ stable keys}}$$

> [!TIP]
> **The Golden Rule of Caching**:
> When a system exhibits **read-heavy traffic (Read/Write $\ge 10:1$)**, **endpoint QPS in the hundreds**, and **the database begins struggling**, introducing a cache is the single most cost-effective architectural intervention.

---

### 1.3 · Database & Storage: Physical Hardware Ceilings

| Component / Operation | Typical Safe QPS per Node | Physical Bottleneck Source | Architectural Scaling Threshold |
|---|---|---|---|
| **Relational DB Reads (MySQL/PG)** | $1,000 - 5,000\text{ QPS}$ | Buffer pool hit ratio, random disk read IOPS | Exceeding 5,000 QPS $\to$ Add read replicas (X-Axis) or front with Redis |
| **Relational DB Writes (MySQL/PG)** | $500 - 2,000\text{ QPS}$ | WAL sequential disk flush (fsync), row lock contention | Exceeding 2,000 QPS $\to$ Add MQ async buffering or horizontal sharding (Z-Axis) |
| **Single Table Row Ceiling** | $\approx 20,000,000\text{ rows}$ | A 3-level B+ tree holds ~20M rows; deeper trees require 4 levels, adding extra random disk IO per lookup | Exceeding 20M rows or 50GB $\to$ Enforce hot/cold tiering or sharding |
| **Object Storage (S3/MinIO)** | 3,500 PUT / 5,500 GET QPS per prefix | HTTP throughput & partitioned metadata management | Large blobs must use client direct uploads (Pre-signed URLs), bypassing API memory |

---

## 2 · The End-to-End System Design Storyline

```system-design-overview-visual
```

This 12-chapter curriculum advances systematically from **single-instance baselines to hyperscale distributed platforms**:

```text
[ Stage 0: Blueprint & Numbers ] ──► 00 Blueprint & Estimation Numbers (This Note)
                                            │
  ┌─────────────────────────────────────────┴─────────────────────────────────────────┐
  ▼                                                                                   ▼
[ Stage 1: Compute & Orchestration ]                                [ Stage 2: Cache & Persistent Storage ]
01  Stateless Service (Stateless API & URL Shortener)               01D Redis (In-Memory Cache, Anti-Stampede, Locks)
01B Virtualization & Containers (Namespaces/Cgroups)                02  Database Paradigms (Scalability Cube/Sharding/Replication)
01C Kubernetes (Pod Lifecycle, Scheduling & HPA)                    04  Storage Systems (Block/File/Object S3 & LSM-Trees)
  │                                                                                   │
  └─────────────────────────────────────────┬─────────────────────────────────────────┘
                                            ▼
[ Stage 3: Asynchronous Decoupling & Distribution ]
06  Async Messaging Systems (Kafka/MQ Buffering, Durable Logs & Ordering)
09  Consistent Hashing (Elastic Routing & Virtual Node Rebalancing)
                                            │
                                            ▼
[ Stage 4: End-to-End Industrial Case Studies ]
07  Photo Sharing & Feed System (Direct S3 Uploads, Fan-out, Timeline Aggregation)
08  Async LLM RL Training Platform (Modern AI Infrastructure, Actor-Learner Decoupling)
10  Flash Sale Architecture (Redis Admission Filtering, MQ Leveling, Zero Overselling)
11  Mobile Push & Notification System (Transactional Outbox, Priority Queues, APNs/FCM Adapters)
                                            │
                                            ▼
[ Stage 5: Reference ] ──► 99 Glossary & Architectural Comparison Matrix
```

---

## 3 · Chapter Alignment & Core Bottlenecks Solved

1. **[[SystemDesign01 Stateless Service|01 Stateless Service]]**: Decouples compute from state to enable horizontal scaling (Scale Out); end-to-end URL Shortener system design.
2. **[[SystemDesign01B Virtualization Containers|01B Virtualization & Containers]]**: Lightweight Linux Namespaces & Cgroups isolation for multi-tenant container packing.
3. **[[SystemDesign01C Kubernetes|01C Kubernetes]]**: Automated scheduling, declarative controllers, service routing, and horizontal pod autoscaling (HPA).
4. **[[SystemDesign01D Redis|01D Redis]]**: High-throughput in-memory caching, Cache-Aside, multi-layer stampede defenses, distributed leases, and admission control.
5. **[[SystemDesign02 Database Paradigms|02 Database Paradigms]]**: ACID invariants, Scalability Cube (X-axis replicas, Y-axis microservices, Z-axis sharding), 6 sharding key rules, and 4 patterns to eliminate scatter-gather.
6. **[[SystemDesign04 Storage Systems|04 Storage Systems]]**: Object storage (S3/MinIO), blob handling, LSM-Tree sequential storage engines, and metadata decoupling.
7. **[[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]]**: Kafka/MQ log streaming, peak leveling, asynchronous decoupling, and consumer delivery semantics.
8. **[[SystemDesign09 Consistent Hashing|09 Consistent Hashing]]**: Elastic node ring routing, virtual nodes for skew prevention, and minimal rebalancing overhead.
9. **[[SystemDesign07 Photo Sharing Feed|07 Photo Sharing & Feed System]]**: 100:1 read/write ratio, CDN direct uploads, Push vs Pull fan-out, and timeline hydration.
10. **[[SystemDesign08 LLM Async RL Platform|08 Async LLM RL Platform]]**: Asynchronous Actor-Learner decoupling, parameter synchronization, and training cluster pipelines.
11. **[[SystemDesign10 Flash Sale|10 Flash Sale Architecture]]**: Extreme microsecond spikes, Redis fast reject admission, MQ buffering, and atomic DB inventory reservation.
12. **[[SystemDesign11 Notification System|11 Mobile Push & Notification System]]**: 1B notifs/day scale, Transactional Outbox zero drop, Dispatcher batch slicing, durable priority queues, APNs/FCM HTTP/2 connection pooling, and At-least-once idempotent feedback loops.
13. **[[SystemDesign99 Glossary|99 Glossary & Key Concepts]]**: Distributed systems axioms, theorems (CAP/BASE/Little's Law), and architectural decision matrices.

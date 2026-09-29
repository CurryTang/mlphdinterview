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
> **Caching Principles**:
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

## 2 · The System Design Landscape: Two Core Pillars

```system-design-overview-visual
```

The system design curriculum is structured into two focused pillars: **Practical Cases (End-to-End Deep Dives)** and **Wiki Pattern Library (Atomic Design Patterns & Component Baselines)**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      System Design Knowledge Map                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
[ Pillar 1: Practical Cases (End-to-End) ]             [ Pillar 2: Wiki Pattern Library ]
07 Photo Sharing & Feed (Fan-out, Timelines)           · System Design Blueprint (FR+Non-FR+QPS+Diagram+Deep Dive)
08 Async LLM RL Platform (Actor-Learner)               · Blueprint & Quantitative Numbers (Physical Sizing Limits)
10 Flash Sale (Redis Admission, Inventory Lock)        · Stateless Architecture (Compute-State Decoupling & Drain)
11 Mobile Push & Notifications (Outbox, Queues)        · Virtualization & Containers (Namespaces/Cgroups Boundaries)
12 Distributed Crossword Solver (Bit-CSP & CAS)        · Kubernetes (Pod Lifecycle, Controllers & Scheduling)
                                                       · Redis (Cache-Aside, Anti-Stampede & Distributed Locks)
                                                       · Database Paradigms & Sharding (Cube & Sharding Keys)
                                                       · Distributed Storage Systems (Object S3 & Append Logs)
                                                       · Consistent Hashing (Virtual Nodes & Minimal Migration)
                                                       · Glossary & Key Concepts (CAP/BASE/Little's Law Matrix)
                                                       · Event Bus (Domain Events, ECST & Declarative Rules)
                                                       · Message Queue (Work Queues, Leases & Poison Pills)
                                                       · NoSQL + Streaming (Built-in CDC & Materialized Views)
                                                       · Kafka (Partitioned Log, High-Throughput & Consistency)
                                                       · Transactional Outbox (DB + MQ Dual-Write Fix)
                                                       · Control vs Data Plane (K8s/Envoy Survival Invariant)
                                                       · Pull vs Push (Fan-out on Read vs Write Trade-offs)
                                                       · Idempotency (Intent Keys & Atomic Implementation)
                                                       · NewSQL (Storage-Compute Decoupling & Distributed SQL)
```

---

## 3 · Chapter Alignment & Core Bottlenecks Solved

### 3.1 · Practical Cases (End-to-End System Deep Dives)

1. **[[SystemDesign07 Photo Sharing Feed|Case 07 · Photo Sharing & Feed System]]**: 100:1 read/write ratio, CDN direct uploads, Push vs Pull fan-out, and timeline hydration.
2. **[[SystemDesign08 LLM Async RL Platform|Case 08 · Async LLM RL Platform]]**: Asynchronous Actor-Learner decoupling, parameter synchronization, and training cluster pipelines.
3. **[[SystemDesign10 Flash Sale|Case 10 · Flash Sale Architecture]]**: Extreme microsecond spikes, Redis fast reject admission, MQ buffering, and atomic DB inventory reservation.
4. **[[SystemDesign11 Notification System|Case 11 · Mobile Push & Notification System]]**: 1B notifs/day scale, Transactional Outbox zero drop, Dispatcher batch slicing, durable priority queues, APNs/FCM HTTP/2 connection pooling, and At-least-once idempotent feedback loops.
5. **[[SystemDesign12 Crossword Solver|Case 12 · Distributed Crossword Solver]]**: NP-complete CSP solving and Distributed DFS engine. Dual-track ingestion (static complexity gate + dynamic watchdog) enables single-worker bit-parallel indexing to resolve >90% of traffic; heavy tails escalate to a Distributed DFS pool with shallow-choice work stealing, credit conservation barrier, and atomic CAS consensus.

### 3.2 · Wiki Pattern Library & Knowledge Points

1. **[[SystemDesign01 Stateless Service|Wiki · System Design Blueprint (FR + Non-FR + QPS + Diagram + Deep Dive)]]**: End-to-end 5-stage interview methodology, capacity sizing, layered topology, 3 core deep-dive trade-offs, and request tracing via URL Shortener.
2. **[[SystemDesignWiki Stateless Architecture|Wiki · Stateless Architecture & State Externalization]]**: Compute-state decoupling, 3 externalization pillars (Session/Blob/Workflow), idempotency intent keys, graceful draining, and dual-probe lifecycles.
3. **[[SystemDesign00 Overview|Wiki · Blueprint & Quantitative Numbers]]**: Physical hardware performance boundaries across Application Servers, In-Memory Caches, Databases, and Object Storage.
4. **[[SystemDesign01B Virtualization Containers|Wiki · Virtualization & Containers]]**: Lightweight Linux Namespaces & Cgroups isolation for multi-tenant container packing.
5. **[[SystemDesign01C Kubernetes|Wiki · Kubernetes (Orchestration & Control Plane)]]**: Automated scheduling, declarative controllers, service routing, and horizontal pod autoscaling (HPA).
6. **[[SystemDesign01D Redis|Wiki · Redis (In-Memory Cache & Coordination)]]**: High-throughput in-memory caching, Cache-Aside, multi-layer stampede defenses, distributed leases, and admission control.
7. **[[SystemDesign02 Database Paradigms|Wiki · Database Paradigms & Sharding]]**: ACID invariants, Scalability Cube (X-axis replicas, Y-axis microservices, Z-axis sharding), 6 sharding key rules, and patterns to eliminate scatter-gather.
8. **[[SystemDesign04 Storage Systems|Wiki · Distributed Storage Systems]]**: Object storage (S3/MinIO), blob handling, LSM-Tree sequential storage engines, and metadata decoupling.
9. **[[SystemDesign09 Consistent Hashing|Wiki · Consistent Hashing]]**: Elastic node ring routing, virtual nodes for skew prevention, and minimal rebalancing overhead.
10. **[[SystemDesign99 Glossary|Wiki · Glossary & Key Concepts]]**: Distributed systems axioms, theorems (CAP/BASE/Little's Law), and architectural decision matrices.
11. **[[SystemDesignWiki Event Bus|Wiki · Event Bus & Event-Driven Architecture]]**: Pub/Sub broadcasting, Event-Carried State Transfer (ECST), content-based declarative filtering; in-process, Redis P/S, NATS, and EventBridge throughput baselines ($1	ext{k} - 10	ext{M QPS}$); domain event fan-out & cache invalidation.
12. **[[SystemDesignWiki Message Queue|Wiki · Message Queue: Competing Consumers & Task Scheduling]]**: Competing consumers (1-to-1 preemptive execution), visibility timeouts & two-phase acks, DLQ poison pill isolation, delay/priority queues; RabbitMQ, AWS SQS, and Redis List performance ($20	ext{k} - 100	ext{k QPS}$).
13. **[[SystemDesignWiki NoSQL Streaming|Wiki · NoSQL + Streaming: CDC & Event-Driven Pipelines]]**: Engine-level native CDC, stream-table duality ($S \iff T$), `OldImage / NewImage` delta tracking; CQRS materialized views (Elasticsearch / Redis / ClickHouse); DynamoDB Streams, MongoDB Change Streams, and Redis Streams ($100	ext{k} - 1	ext{M+ QPS}$).
14. **[[SystemDesignWiki Kafka|Wiki · Kafka: Partitioned Log & Architecture Deep Dive]]**: High-throughput distributed commit log, sequential disk I/O, OS Page Cache reuse and `sendfile` zero-copy; ISR, High Watermark, Leader Epoch recovery, and KRaft consensus; Idempotent producers & Exactly-Once Semantics (EOS); 6 production system design interview use cases.
15. **[[SystemDesignWiki Transactional Outbox|Wiki · Transactional Outbox Pattern]]**: Resolving the dual-write hazard between database mutations and message queues via local ACID persistence and Polling / CDC transaction log tailing.
16. **[[SystemDesignWiki Control Data Plane|Wiki · Control Plane vs Data Plane Separation]]**: Decoupling the "Brain" from the "Muscles"; the survival invariant (control plane failure never disrupts running data plane traffic); K8s Master vs Node Pods, Istio vs Envoy xDS, Kafka KRaft Controller vs Broker; quantitative baselines ($1	ext{k}$ vs $10	ext{M QPS}$).
17. **[[SystemDesignWiki Pull vs Push|Wiki · Pull vs Push Models]]**: Trade-offs between Fan-out on Write and Fan-out on Read; celebrity hotspot mitigation in feeds; Long Polling vs WebSocket vs SSE client-server streaming.
18. **[[SystemDesignWiki Idempotency|Wiki · Idempotency Design Patterns & Implementations]]**: Mathematical & engineering definitions; deep dive into CI/CD `workflow_run` (Run/Tenant/Repo/Commit/Event); 6 core paradigms (DB unique constraints, Idempotency-Key tokens, state machine CAS transitions, deduplication tables, deterministic hash IDs, and natural idempotence); 4 production pitfalls.
19. **[[SystemDesignWiki NewSQL|Wiki · NewSQL & Distributed SQL Architecture]]**: Compute-storage decoupling, range-based Region splitting, Multi-Raft consensus, Percolator 2PC + HLC/TrueTime distributed transactions, and parallel coprocessor pushdown vs Sharded RDBMS trade-offs.

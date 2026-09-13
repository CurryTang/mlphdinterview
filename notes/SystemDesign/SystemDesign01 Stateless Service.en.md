# System Design 01 · Stateless Services & End-to-End System Design

Course Location: this note → [[SystemDesign01B Virtualization Containers|01B Virtualization and Containers]]

```system-design-overview-visual
```

---

## Core Positioning: Statelessness as the Overarching Paradigm

In modern distributed system architecture, **"Stateless Service" does not mean that the system as a whole contains no state; rather, it means that the computing nodes (service processes) executing business logic do not exclusively own any non-recoverable, persistent business state**.

Every replaceable instance is fundamentally a pure computational container:
$$\text{Replaceable Compute Instance} = \text{Business Code} + \text{Static/Dynamic Config} + \text{Ephemeral Request Context} + \text{Disposable/Rebuildable Local Cache}$$

As long as service processes do not persist authoritative business facts, any individual instance can experience network partition, hardware failure, or be evicted and terminated by a container orchestrator (such as Kubernetes) without any permanent loss of business data. Reverse proxies and load balancers can transparently route subsequent traffic to any healthy replica in the cluster, unlocking the core engineering capabilities of **Horizontal Scalability (Scale-Out)**, **Self-Healing**, and **Zero-Downtime Rolling Deployments**.

This document is structured in two complementary parts:
- **Part I: Architectural Paradigm & Design Patterns**: A systematic exposition of why modern distributed systems prioritize statelessness, the taxonomy of state, the three externalization pillars, idempotency invariants, and operational lifecycle resilience;
- **Part II: Production-Grade System Design Case Study (High-Performance Distributed URL Shortener & Analytics System)**: Applying standard system design methodology (functional/non-functional requirements, rigorous capacity/QPS estimation, architecture topology and data modeling, three deep-dive architectural tradeoffs, and end-to-end request tracing) to concretely execute a complete stateless system design.

---

## Part I: Stateless Architectural Paradigm & Design Patterns

### 1 · Why Distributed Systems Prioritize Statelessness

In legacy stateful architectures, user authentication sessions, temporary upload files, or in-flight job progress are held directly inside the local memory or attached disk of a single server. This creates severe structural bottlenecks:
- **Sticky Routing Lock-In**: Load balancers must pin a client's subsequent requests to the exact same physical machine (e.g., via session-affinity hashing). If that instance crashes, the user's session and progress are permanently destroyed even if other cluster nodes have ample capacity;
- **Impaired Elastic Scaling**: During sudden traffic spikes, newly spawned instances cannot help drain existing sticky sessions because they lack the local state; during scale-down, complex state migration protocols are mandatory;
- **Constrained Scheduling**: Workload orchestrators (Kubernetes, Nomad) cannot freely bin-pack pods onto underutilized physical nodes because pods are chained to local state.

Stateless architecture eliminates this coupling entirely:
```text
Client
  -> Load Balancer (Round Robin / Least Connections)
      -> API Pod 1 (Stateless Compute)
      -> API Pod 2 (Stateless Compute)  ===>  Shared External Systems (DB / Redis / S3 / MQ)
      -> API Pod 3 (Stateless Compute)
```
- **Instant Faulty Instance Replacement**: If any pod crashes, the orchestrator terminates it and schedules a fresh replica on any available node; the load balancer detects probe changes and drops traffic instantly;
- **Turnkey Elasticity**: Scaling out simply entails booting identical container images with injected configuration, immediately accepting parallel production traffic;
- **Seamless Rolling Updates**: Following graceful draining protocols, old pods are retired while new pods spin up without client interruption.

---

### 2 · The Taxonomy of State in Distributed Systems

State does not vanish in a stateless architecture; it is externalized into purpose-built infrastructure engineered for consensus, durability, and high availability. Before architecting a system, every piece of data must be categorized:

| State Category | Concrete Examples | Invariants & Characteristics | Canonical Location | Failure Tolerance |
|---|---|---|---|---|
| **Authoritative State** | User accounts, transactions, financial ledgers, inventory balances | Core Source of Truth; must never be lost; requires strict ACID transactional semantics | Relational DB (MySQL/PostgreSQL) / Distributed SQL / Durable write-ahead logs | Absolutely zero data loss tolerated |
| **Durable Blob** | User media, raw videos, ML weights, compiled analytical reports | High throughput, append-only or immutable binary streams | Distributed Object Store (AWS S3 / GCS / Ceph) | Zero loss; metadata stored in DB, payload in S3 |
| **Coordination State** | Distributed locks, service registry, Leader Epoch, Worker leases | Metadata ensuring mutual exclusion and consensus across nodes | ZooKeeper / etcd / Consul / Redis (with TTL) | Must have TTL/lease timeouts to prevent deadlocks |
| **Rebuildable State** | Cache lookups, materialized views, inverted indices, feature vectors | Derived data reducing pressure on authoritative stores; rebuildable on demand | Redis / Memcached / Local in-memory LRU cache | Fully disposable; reconstructed from source of truth |
| **Request-Local State** | Trace ID, HTTP headers, loop iterators, intermediate tensor activations | Ephemeral to a single request; destroyed upon response completion | Process call stack & volatile heap | Destroyed on crash; client safely retries |

> 💡 **The Stateless Litmus Test**:
> To verify whether a service is genuinely stateless, ask:
> **"If this running service process is killed abruptly with `kill -9` this exact millisecond, what business facts are permanently lost?"**
> If the answer is "no business facts are lost; only the single in-flight unconfirmed request needs to be retried by the client or queue", the service conforms to true stateless standards.

---

### 3 · The Three Externalization Pillars

#### Pillar 1: Session & Authentication Externalization
- **Stateful Anti-Pattern**: Holding `Map<session_id, UserContext>` inside process memory, requiring sticky cookies at the gateway.
- **Modern Stateless Practice**:
  - **Approach A (Shared Session Store)**: Clients hold an opaque random token. The stateless API queries and caches user state in a distributed Redis cluster with short TTLs. Immediate revocation or ban takes effect instantly by evicting the token from Redis.
  - **Approach B (Cryptographically Signed JWT)**: Clients hold self-contained JSON Web Tokens signed by the auth authority. Stateless API instances locally verify signatures using a shared public key (zero network I/O).
  - *Edge Defense*: Pure JWTs cannot be instantly revoked. Production architectures combine short-lived JWTs (e.g., 15-minute expiry) with a centralized revocation blocklist or token generation version counter in Redis.

#### Pillar 2: Blob & Media Externalization
- **Anti-Pattern**: Clients upload gigabytes of media directly to `/tmp` on an API pod, and the API proxies the bytes downstream. Large payloads exhaust pod memory and disk, while crashes destroy upload progress.
- **Production Standard — Presigned URL Direct Upload**:
  1. The client sends metadata (filename, byte size, checksum) to the stateless API;
  2. The API creates a metadata row (`file_id, status=pending`) in the DB and generates a time-limited Presigned Upload URL from the object store SDK;
  3. The client uploads the binary payload directly to S3/GCS, completely bypassing API compute nodes;
  4. Upon durable write completion, the object store fires an EventBridge/Webhook event to update the DB record to `status=ready`.

#### Pillar 3: Long-Running Workflow Externalization
- **Anti-Pattern**: A client requests a 10-minute analytics report aggregation, keeping the HTTP socket open. A network blip or pod restart aborts the entire computation.
- **Production Standard — Asynchronous Ingestion & Worker Leasing**:
  1. The client sends `POST /jobs`. The stateless API validates inputs, writes `job_id, status=queued` to the DB, and enqueues the job ID into a message queue (Kafka / RabbitMQ / SQS);
  2. The API immediately returns `HTTP 202 Accepted` with a `Location: /jobs/{job_id}` polling header;
  3. Independent background worker pods claim jobs using **time-bounded leases**;
  4. Workers write artifacts to object storage and update DB status; clients query status via polling or WebSocket notifications.

---

### 4 · Strict Idempotency & Consistency Invariants

Because stateless instances can fail at any time, **network timeouts do not imply execution failure**. Clients must automatically retry upon transient errors, which requires stateless backends to guarantee **strict idempotency**.

#### Production-Grade Idempotency Protocol
1. **Client-Generated Idempotency Key**:
   The client assigns a unique UUID to every distinct business intent (e.g., passed in `X-Idempotency-Key: 9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d`).
2. **Authoritative Storage Unique Constraint**:
   Maintain a dedicated idempotency ledger in the transactional store:
   ```sql
   CREATE TABLE idempotency_keys (
       idempotency_key VARCHAR(64) PRIMARY KEY,
       user_id BIGINT NOT NULL,
       status VARCHAR(16) NOT NULL, -- 'PENDING', 'SUCCESS', 'FAILED'
       response_body TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```
3. **Atomic Transactional Insertion**:
   - When a request hits any stateless API instance, within an active database transaction execute:
     `INSERT INTO idempotency_keys (idempotency_key, user_id, status) VALUES (?, ?, 'PENDING');`
   - If a Primary Key conflict occurs:
     - If status is `SUCCESS`: Return the cached `response_body` directly; do not re-execute business logic;
     - If status is `PENDING`: An identical request is in-flight; return `HTTP 409 Conflict` or backoff;
   - If insertion succeeds: Execute the mutation within the **same local transaction**, update status to `SUCCESS`, save the response body, and commit.

> ⚠️ **Critical Trap: Never Use Payload Hash as an Idempotency Key**
> If you hash the request body as the key, a user rapidly double-clicking an "Order" button will have their second legitimate duplicate purchase eaten as a retry; conversely, if the payload includes a fluctuating timestamp, legitimate network retries will bypass idempotency protection. The key must represent the client's single intent.

---

### 5 · Process Lifecycle & Operational Resilience

To allow stateless instances to be scheduled and scaled elastically by Kubernetes, service processes must honor cloud-native lifecycle semantics:

```text
       [Pod Startup]
             │
     Load Immutable Config & Secrets
             │
             ▼
   [Readiness Probe Passes] ─── Begin Accepting Traffic (LB registers Pod IP)
             │
      Process Inbound Requests (Stateless Execution)
             │
   [SIGTERM Signal Received] (Triggered by scale-down or rolling deployment)
             │
             ├─ 1. Mark Readiness = Unready ─── LB removes Pod from active endpoints
             ├─ 2. Stop pulling new messages from queues
             ├─ 3. Drain in-flight work (bounded by grace period, e.g., 30s)
             └─ 4. Close database connection pools and message channels
             │
       [Process Exits Gracefully (Exit 0)]
```

- **Dual Probe Decoupling (Liveness vs Readiness)**:
  - **Liveness Probe**: Detects deadlocks or unrecoverable fatal loops. Failure triggers an immediate container restart;
  - **Readiness Probe**: Detects whether the pod is primed to serve live traffic. Marked unready during cold startup or during graceful drain, directing the load balancer to withhold new requests.
- **Worker Lease Heartbeats**:
  Workers processing asynchronous tasks do not own them permanently. A worker claims a task by updating `leased_by=worker_1, lease_until=now() + 60s`. The worker periodically refreshes the lease; if it crashes, the lease expires naturally, and healthy workers reclaim the task safely via idempotency.

---

### 6 · Architectural Tradeoffs: Where Does the Complexity Go?

Externalizing state makes the compute tier delightfully simple, but it **transfers and amplifies pressure onto shared external systems**:
- **Connection Storms**: If 1,000 stateless API pods each keep 20 DB connections, the primary database faces 20,000 concurrent sockets, exhausting connection pools. Connection poolers (PgBouncer, RDS Proxy) and caching tiers are indispensable;
- **Network Latency Overhead**: Every operation traverses the network to fetch state. Connection reuse (HTTP Keep-Alive), batching, and tiered caching are mandatory;
- **Circuit Breakers & Bulkheads**: Latency spikes in Redis or DB can backlog worker threads across stateless instances. Every external dependency must have tight timeouts, bounded thread pools, and fallback behaviors.

#### The Legitimate Boundary of Sticky Routing
Stateless services do not ban affinity. In **WebSocket gateways, collaborative editing, online game rooms, and LLM KV cache scheduling**, routing the same session to the same host yields massive cache-hit benefits.
> 🔑 **Unifying Principle**:
> **Affinity is an optimization; External Durable State is the correctness guarantee**.
> When affinity hits, latency is optimal; if that node crashes, any other node can reconstruct context from durable storage, maintaining correctness.

---

### 1.4 · Application Server QPS Tiers and the Saturation Knee Rule

In stateless compute tier sizing, architects must maintain an intuitive grasp of per-node QPS thresholds:

| QPS per Node | System State | Engineering Characteristics & Actions |
|---|---|---|
| **10 QPS / node** | Minimal Load | Completely acceptable; internal admin tools or heavy compute batch tasks. |
| **100 QPS / node** | Casual / Easy | Steady-state zone for most microservices; CPU utilization typically $< 15\%$. |
| **500 QPS / node** | **Normal Working Zone** | The healthy sweet spot for microservices (JSON parsing, auth, 1–2 DB queries). |
| **2,000 QPS / node** | **Optimization Alert** | Profiling required; monitor GC pauses, thread queue lengths, and DB connection limits. |
| **> 2,000 QPS / node** | **High Load** | Approaching physical limits of standard business logic; introduce caching or async batching. |
| **> 10,000 QPS / node** | **Extreme High Throughput** | **Rare in general apps**. Requires: ultra-simple logic, pure in-memory cache, non-blocking IO (epoll), connection reuse, and zero heavy ORM. |

> [!IMPORTANT]
> **The Universal Capacity Saturation Rule (The +30% Knee-of-the-Curve Rule)**:
> In load testing, when **incoming QPS increases by only 30%**, if you observe **a sharp non-linear spike in p95/p99 latency, steep CPU rise, or queue buildup**, the system has crossed its saturation knee point.
> **Read-heavy + API QPS in the hundreds + DB struggling $\implies$ Must immediately introduce an in-memory cache (Redis)!**

## Part II: Production-Grade System Design Case Study (URL Shortener & Analytics System)

To ground the stateless architectural paradigm in a complete, end-to-end design, this section details the complete architecture of an industry-standard **High-Performance Distributed URL Shortener & Analytics System**.

---

### 1. Requirements & Scope

#### Functional Requirements (FR)
1. **URL Shortening**: Given a long URL, generate a globally unique 7-character Base62 short alias (e.g., `https://short.link/a8K9zQ1`) with optional expiration;
2. **Fast Redirection**: When a client accesses the short link, resolve the original long URL with sub-15ms latency and return an HTTP 302 redirect;
3. **Click Analytics Ingestion**: Collect telemetry for every redirection (timestamp, client IP, geographic location, User-Agent, Referer) for aggregated reporting.

*Out of Scope*:
- Custom vanity alias reservation and conflict dispute arbitration;
- Real-time deep learning malware/phishing sandbox filtering;
- Advanced paid enterprise analytics reporting dashboards.

#### Non-Functional Requirements (NFR)
1. **High Availability**: $99.99\%$ uptime for the redirection read path ($< 52.6$ minutes downtime/year). The read path must remain operational via distributed caches even during primary database outages;
2. **Ultra-Low Latency**:
   - Read Redirection Path: $\text{P99} < 15\text{ ms}$;
   - Write Shortening Path: $\text{P99} < 100\text{ ms}$;
3. **High Read-to-Write Ratio & Elastic Scale-Out**: $100:1$ read-heavy workload; stateless compute instances must scale horizontally in seconds during marketing bursts;
4. **Data Durability & Non-Collision Invariants**: Generated short links must never be lost; two different long URLs must never resolve to the same short code.

---

### 2. Capacity & Scale Estimation

#### (1) Traffic Estimation
- **Write QPS**:
  - Assume 10 million ($10^7$) new short URLs generated per day;
  - 1 day $\approx 10^5\text{ seconds}$ ($86{,}400\text{ s}$);
  $$\text{Average Write QPS} = \frac{10^7}{10^5} = 100\text{ writes/s}$$
  - Peak Write QPS (factor of $2\times$):
  $$\text{Peak Write QPS} = 100 \times 2 = 200\text{ writes/s}$$

- **Read QPS (Redirection)**:
  - Read-to-write ratio of $100:1$;
  $$\text{Average Read QPS} = 100 \times 100 = 10{,}000\text{ reads/s}$$
  - Peak Read QPS (factor of $3\times$):
  $$\text{Peak Read QPS} = 10{,}000 \times 3 = 30{,}000\text{ reads/s}$$

#### (2) Storage Capacity Estimation (5-Year Horizon)
- **Record Size Breakdown**:
  - `id`: 8 bytes (BIGINT)
  - `short_code`: 7 bytes (VARCHAR(8))
  - `original_url`: 512 bytes average (VARCHAR(2048))
  - `user_id`: 8 bytes (BIGINT)
  - `created_at`: 8 bytes (TIMESTAMP)
  - `expires_at`: 8 bytes (TIMESTAMP)
  - B+ Tree index overhead: $\approx 60\text{ bytes}$
  - **Total per record**: $\approx 611\text{ bytes} \approx 0.6\text{ KB}$
- **Cumulative 5-Year Storage**:
  - Total records in 5 years: $10^7\text{/day} \times 365 \times 5 = 1.825 \times 10^{10}\text{ records}$ ($18.25\text{ billion}$);
  - Total persistent storage volume:
  $$\text{Storage Volume} = 1.825 \times 10^{10} \times 0.6\text{ KB} \approx 10.95\text{ TB}$$
  *Implication*: A single relational database table cannot comfortably house 18 billion rows. The database tier must be partitioned across a sharded cluster (e.g., hash partitioned on `short_code`) or deployed on Distributed SQL (TiDB / CockroachDB).

#### (3) Cache Memory Estimation (Redis Cluster)
- Using the **80/20 Pareto Principle**: $20\%$ of popular short URLs drive $80\%$ of redirection traffic;
- Daily unique hot short URLs $\approx 10^7 \times 20\% = 2\text{ million}$ links;
- Per-item cache size (Key: `short_code` 7B, Value: `original_url` 512B, plus Redis metadata $\approx 600\text{ bytes}$);
- **1-Day Hot Working Set**:
  $$\text{Cache Memory} = 2 \times 10^6 \times 600\text{ bytes} \approx 1.2\text{ GB}$$
- Caching a 7-day rolling window with an LRU eviction policy requires:
  $$1.2\text{ GB} \times 7 \approx 8.4\text{ GB}$$
  *Implication*: A single 16GB Redis replica easily holds the entire working set. In production, an active-standby Sentinel or Redis Cluster deployment is used to distribute 30,000 QPS read throughput and ensure high availability.

#### (4) Network Bandwidth
- **Peak Read Bandwidth**: $30{,}000\text{ reads/s} \times 512\text{ bytes} \approx 15.36\text{ MB/s} \approx 123\text{ Mbps}$;
- **Peak Write Bandwidth**: $200\text{ writes/s} \times 600\text{ bytes} \approx 120\text{ KB/s} \approx 0.96\text{ Mbps}$.

---

### 3. Architecture Topology & Data Schema

#### Layered System Topology
```text
[ Clients / Browsers ]
       │
       ▼
[ Anycast DNS / CDN Edge Caches ]
       │
       ▼
[ API Gateway / Nginx Proxy Cluster ]
  (TLS termination, Rate Limiting, WAF filtering)
       │
       ├────────────────────────────────────────┐
       │ (Write Path: POST /urls)               │ (Read Path: GET /{short_code})
       ▼                                        ▼
[ Stateless URL-Writer Pods ]            [ Stateless Redirect Pods ]
  - Pure compute, completely stateless       - Pure compute, sub-15ms response
  - In-memory pre-allocated Ticket Range     - Pre-filter via Bloom Filter
       │                                        │
       ├─ (1. Transactional Commit)              ├─ (1. Cache Hit: Immediate 302)
       │                                        ▼
       ▼                                 [ Redis Cache Cluster ]
[ Sharded Relational Database ]                 │ (Miss: Fetch from Replica)
  (MySQL / TiDB Partitioned Cluster)            ▼
  - Primaries handle writes              [ DB Read Replicas ]
  - Sharded on short_code hash                  │
       │                                        │ (2. Asynchronously emit click events)
       │                                        ▼
       │                                 [ Kafka / Event Stream ]
       │                                        │
       ▼                                        ▼
[ Elastic Worker Cluster ]               [ Stream Analytics Workers ]
  (Async cleanup of expired links)         (Micro-batch aggregates into ClickHouse)
```

#### Database Schema Design

##### 1. Master Short Link Table (`urls`)
```sql
CREATE TABLE urls (
    id BIGINT NOT NULL PRIMARY KEY,            -- Globally unique 64-bit auto-incrementing ID
    short_code VARCHAR(8) NOT NULL,            -- Base62 encoded 7-char alias
    original_url VARCHAR(2048) NOT NULL,       -- Destination long URL
    user_id BIGINT DEFAULT NULL,               -- Owning user ID (for quotas)
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT NULL,         -- Expiration timestamp
    UNIQUE KEY uk_short_code (short_code),     -- Guaranteed non-collision
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

##### 2. Write Idempotency Table (`idempotency_keys`)
```sql
CREATE TABLE idempotency_keys (
    request_id VARCHAR(64) NOT NULL PRIMARY KEY, -- Client-provided intent UUID
    short_code VARCHAR(8) NOT NULL,              -- Resulting short code
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### Protocol Decision: HTTP 301 vs HTTP 302 Redirection
- **HTTP 301 Moved Permanently**:
  The browser permanently caches the redirect mapping locally. Future visits never reach the shortener service; the browser redirects directly.
  - *Pro*: Slashes bandwidth and compute load on the shortener backend.
  - *Fatal Con*: **Completely blinds the system to click analytics** (cannot capture click count, IP, device, or timestamp); furthermore, destination URLs cannot be updated or expired.
- **HTTP 302 Found (Recommended)**:
  The browser treats the redirect as temporary and hits the short URL service on every single visit.
  - *Pro*: Enables $100\%$ accurate click analytics ingestion and dynamic revocation.
  - *Decision*: **HTTP 302 Found is mandatory**. The 30,000 read QPS is easily absorbed by the stateless API and Redis tiers.

---

### 4. Deep-Dive Architectural Tradeoffs (Deep Dives × 3)

#### Deep Dive 1: Global Stateless ID Generation
A 7-character Base62 string ($[0-9a-zA-Z]$) provides $62^7 \approx 3.52\text{ trillion}$ unique combinations. How do stateless API pods generate non-colliding IDs without becoming a bottleneck?

- **Option A: Hash Truncation + DB Collision Checks**
  - *Mechanism*: Compute MD5/MurmurHash on the long URL, encode the first 43 bits into 7 Base62 characters. If the DB unique constraint fails, append a salt and re-hash.
  - *Tradeoff*: At 18 billion rows, **the Birthday Paradox drastically spikes collision rates**. Repeated DB round-trips to resolve collisions create massive P99 latency spikes on write endpoints.
- **Option B (Preferred): Distributed Ticket Range Server + Base62 Mapping**
  - *Mechanism*: Treat the short code as a bijective Base62 representation of a 64-bit integer (`ID = 100,000,000` $\iff$ `a8K9zQ1`). Collision probability is mathematically zero.
  - *Stateless Execution*: A lightweight central coordinator (e.g., MySQL ticket table or etcd) dispenses integer **Ticket Ranges (Segments)**. When an API pod boots, it requests a range (e.g., $[1{,}000{,}001, 1{,}020{,}000]$). The pod increments an internal `AtomicLong` locally in memory (millions of ops/sec, zero network I/O). When $80\%$ consumed, it asynchronously pre-fetches the next range.
  - *Evaluation*: If an API pod crashes, unused IDs in its local buffer are abandoned, causing benign skips. Because $3.52\text{ trillion}$ space is immense, skips have zero business cost; in exchange, **stateless pods require zero inter-node coordination during request execution**.

#### Deep Dive 2: High-Concurrency Cache Resilience (Penetration, Breakdown, Avalanche)
How does the stateless read tier survive 30,000 QPS without crumbling under edge conditions?

- **Option A: Plain Cache-Aside Pattern (Redis LRU Only)**
  - *Failure Modes*:
    1. **Cache Penetration**: Malicious bots query millions of non-existent random short codes. Every lookup misses Redis and hits the database, exhausting DB thread pools;
    2. **Cache Breakdown**: A viral tweet link expires from Redis, causing thousands of concurrent requests to hammer the DB simultaneously to rebuild the cache item.
- **Option B (Preferred): Layered Cache Defense Strategy**
  - *Mechanism*:
    1. **Ingress Bloom Filter**: Keep a compact Bloom Filter in API pod memory or Redis. When short codes are created, their bits are set. When resolving redirects, if the Bloom Filter reports negative, **it is guaranteed not to exist; return 404 immediately at zero DB cost**;
    2. **Null Object Caching**: If a request slips past the Bloom Filter and the DB confirms non-existence, write a sentinel null value to Redis with a 30-second TTL;
    3. **Mutex Lock on Rebuild**: When a hot key expires, only the single request that acquires a distributed Redis lock (`SET key token NX EX 5`) queries the DB replica; other requests spin-wait 20ms and read from cache;
    4. **Jittered Expiry**: Add a $\pm 10\%$ random jitter to the standard 7-day Redis TTL to prevent mass simultaneous expiration (avalanche prevention).

#### Deep Dive 3: Decoupled Click Analytics Ingestion
Redirection requires capturing timestamp, IP, User-Agent, and Referer. How do we log telemetry without degrading redirection latency?

- **Option A: Synchronous In-Line DB Updates**
  - *Mechanism*: On every 302 redirect, execute `UPDATE urls SET clicks = clicks + 1 WHERE short_code = ?`.
  - *Fatal Con*: Transforms 30,000 read QPS into 30,000 lock-contended write transactions. DB connection pools exhaust in seconds, breaking the core principle of stateless compute.
- **Option B (Preferred): Asynchronous Event Streaming + Micro-Batch Aggregation**
  - *Mechanism*:
    1. The stateless Redirect Pod returns the `HTTP 302` response to the client immediately (or non-blockingly emits via a background thread pool);
    2. It pushes a compact event JSON to a Kafka topic (`click_events`):
       `{"code": "a8K9zQ1", "ts": 1789258800, "ip": "1.2.3.4", "ua": "Mobile Safari", "ref": "twitter"}`;
    3. Dedicated Stream Analytics Workers consume from Kafka, tumbling over 10-second windows in memory;
    4. Aggregated click sums are bulk-inserted into an analytical columnar database (ClickHouse) and flushed to coarse Redis counters for dashboard views.
  - *Evaluation*: Analytics latency is eventually consistent (visible within seconds), while the core redirection path remains blistering fast ($< 15\text{ ms}$ P99). Kafka buffers bursts gracefully.

---

### 5. End-to-End Life of a Request

#### (1) Write Path (Create Short URL)
```text
1. Client                  -> Sends HTTP POST /api/v1/urls
                              Header: X-Idempotency-Key: "uuid-123"
                              Body: {"url": "https://example.com/very/long/path"}
2. API Gateway             -> Validates token, enforces token-bucket rate limits;
                              Routes request to healthy Stateless URL-Writer Pod A.
3. URL-Writer Pod A        -> Checks idempotency_keys table;
                              If found, returns existing short_code immediately (idempotency hit).
4. URL-Writer Pod A        -> Atomically fetches next ID from local Ticket Range (e.g., 1000042);
                              Encodes integer to 7-character Base62 string "a8K9zQ1" in memory.
5. URL-Writer Pod A        -> Executes atomic local DB transaction on Sharded DB:
                              ├─ INSERT INTO urls (id, short_code, original_url, ...) VALUES (...);
                              └─ INSERT INTO idempotency_keys (request_id, short_code) VALUES (...);
                              Commits transaction durably.
6. URL-Writer Pod A        -> Asynchronously pre-warms Redis: SET "a8K9zQ1" -> "https://example.com/...";
                              Sets bit pattern in the Ingress Bloom Filter.
7. URL-Writer Pod A        -> Returns HTTP 201 Created with {"short_url": "https://short.link/a8K9zQ1"}.
```

#### (2) Read Path (Redirection & Analytics Emission)
```text
1. User Mobile Browser     -> Hits HTTP GET /a8K9zQ1
2. DNS / CDN               -> Anycast routes to the nearest edge PoP.
3. API Gateway             -> Load balances to arbitrary Stateless Redirect Pod B.
4. Redirect Pod B          -> Checks Bloom Filter: If negative, return 404 immediately.
5. Redirect Pod B          -> Queries Redis Cluster: GET "a8K9zQ1"
                              ├─ [Cache Hit]: Retrieves long URL (1-2 ms latency);
                              └─ [Cache Miss]: Acquires mutex, queries DB replica, populates Redis.
6. Redirect Pod B          -> Returns HTTP 302 Found immediately,
                              Header: Location: "https://example.com/very/long/path"
                              Browser redirects user to target destination.
7. Redirect Pod B          -> Emits telemetry payload asynchronously to Kafka "click_events" topic.
8. Analytics Workers       -> Stream workers aggregate clicks over 10s windows, flushing to ClickHouse.
```

---

## Summary: The Invariant Rules of Stateless Architecture

1. **Absolute Separation of Compute and State**: Compute instances are disposable workers; data durability and consensus are strictly delegated to dedicated storage tiers;
2. **Instances are Cattle, Not Pets**: Any pod can crash or be terminated without notice; recovery is trivial and automatic;
3. **Retries are Inevitable; Idempotency is Mandatory**: Because network failures are indistinguishable from slow execution, endpoints must use database-backed unique constraints to safeguard business invariants;
4. **Performance via Rebuildable Caches; Correctness via Source of Truth**: Local and distributed caches accelerate read performance, while the external authoritative store remains the sole anchor of truth.

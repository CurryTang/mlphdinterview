# System Design Blueprint: FR + Non-FR + QPS + Diagram + Deep Dive

```system-design-overview-visual
```

---

## Core Overview: The Universal 5-Stage System Design Blueprint

In industrial system design and interviews, the most frequent failure mode is not a lack of familiarity with a specific middleware, but rather **tackling the problem without a systematic derivation framework—drifting away from business scopes and physical numbers, prematurely diving into micro-optimizations, or arbitrarily assembling components**.

Distributed system design is never an arbitrary assembly of software blocks; it is **a disciplined sequence of trade-offs made along the progression of system bottlenecks under explicit functional boundaries, SLA availability targets, and hard physical resource constraints**.

A standard, production-grade architectural process follows five disciplined, tightly closed stages:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   Universal 5-Stage System Design Blueprint            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
  1. Scope Boundaries ──► 2. Capacity Sizing ──► 3. Topology & Schema ──► 4. Deep Dives ──► 5. Request Lifecycle
     (FR / Non-FR)           (QPS / Storage)        (Diagram / DDL)       (Trade-offs)         (End-to-End)
```

### Standard 45-Minute Interview Time Allocation

| Stage | Objective & Time Budget | Deliverables & Validation Criteria | Common Anti-Patterns to Avoid |
|---|---|---|---|
| **Step 1: FR & Scope** | **00 ~ 05 min**: Align core use cases & boundaries | Explicit feature list (2~3 primary APIs), Out-of-Scope boundaries | Never draw diagrams immediately; avoid absorbing peripheral nice-to-haves |
| **Step 2: Non-FR & SLAs** | **05 ~ 10 min**: Availability, latency, read/write profile, consistency | Availability target ($99.99\%$), P99 latency SLA, read/write ratio ($100:1$), strong vs. eventual consistency | Vaguely claiming "high availability & low latency" without measurable numerical targets |
| **Step 3: QPS & Capacity Sizing** | **10 ~ 15 min**: Throughput, storage growth, memory cache, network | Average/peak QPS, 5-year storage volume (TB), hot-key cache sizing (80/20 rule), network egress | Sizing only QPS while ignoring storage and RAM, leaving data tier decisions ungrounded |
| **Step 4: Architecture Diagram & Schema** | **15 ~ 25 min**: Layered topology, DB tables, protocols | End-to-end layered topology (Gateway, stateless compute, cache, storage, queue, worker), SQL DDL | Mixing compute and storage tiers; maintaining mutable state inside application memory |
| **Step 5: Deep Dives (Trade-offs)** | **25 ~ 40 min**: Tackle core bottlenecks with 2+ alternatives | Rigorous comparative trade-offs for 2~3 critical bottlenecks (Option A vs. Option B) | Claiming a solution is a "silver bullet"; omitting failure modes and operational costs |
| **Step 6: End-to-End Life of a Request** | **40 ~ 45 min**: Trace complete write/read execution flows | Sequential write & read lifecycles, single-point-of-failure recovery, degradation fallbacks | Disconnected flowcharts where components are linked without explaining protocol interactions |

To illustrate this blueprint in action, the following walkthrough executes the full framework on a canonical high-concurrency stateless microservice system: **Distributed URL Shortener & Analytics System**.

---

## Step 1: Functional Requirements & Scope

The first step requires establishing explicit boundaries with stakeholders to prevent the architecture from sprawling out of control.

### 1. Functional Requirements (In Scope)
1. **URL Shortening**: The system takes an original long URL and returns a globally unique 7-character short token (e.g., `https://short.link/a8K9zQ1`), supporting an optional expiration timestamp (TTL);
2. **Fast Redirection**: When a client accesses a short link, the system resolves the original URL in milliseconds and returns an HTTP redirect response to the destination;
3. **Click Analytics Ingestion**: Accurately log each redirection event (timestamp, client IP, geographic country/city, User-Agent, Referer) for aggregated analytical reporting.

### 2. Out of Scope
- Custom user vanity aliases and alias front-running dispute arbitration;
- Real-time ML malware sandbox analysis and phishing link crawling;
- Enterprise multi-tenant billing and visual reporting dashboards.

---

## Step 2: Non-Functional Requirements & SLAs

Non-functional requirements dictate the foundational choice of infrastructure (e.g., strong vs. eventual consistency, monolithic vs. distributed sharded storage):

1. **High Availability**:
   - Redirection read service SLA is **$99.99\%$ (four nines)**, corresponding to $< 52.6\text{ minutes}$ of unscheduled annual downtime;
   - The read path must remain resilient: even if the primary database degrades or becomes temporarily unreachable, redirection lookups must be sustained entirely by the distributed cache.
2. **Ultra-Low Latency**:
   - Read redirection path: $\text{P99 Latency} < 15\text{ ms}$ (redirections directly stall user browsing experience);
   - Write creation path: $\text{P99 Latency} < 100\text{ ms}$.
3. **Read-Heavy Traffic & Stateless Elasticity**:
   - Traffic follows a canonical **$100:1$ read-to-write ratio**;
   - The compute tier must be strictly **stateless**, scaling out horizontally within seconds to absorb marketing surges.
4. **Data Durability & Invariants**:
   - Generated mappings must never be lost, and the same short token must never collide or overwrite distinct long URLs.

---

## Step 3: QPS Estimation & Capacity Sizing

Back-of-the-envelope estimation anchors architecture in physical realities. Architectural decisions made without numbers cannot be validated.

### 1. Traffic QPS Estimation
- **Write QPS (URL Creation)**:
  - Assume the system generates $10\text{ million} (10^7)$ new URLs per day;
  - 1 day is approximated as $10^5\text{ seconds} (86{,}400\text{ s})$:
  $$\text{Average Write QPS} = \frac{10^7\text{ requests}}{10^5\text{ seconds}} = 100\text{ writes/s}$$
  - Applying a $2\times$ peak factor for surges:
  $$\text{Peak Write QPS} = 100 \times 2 = 200\text{ writes/s}$$

- **Read QPS (Redirection)**:
  - Applying the $100:1$ read-to-write ratio:
  $$\text{Average Read QPS} = 100\text{ writes/s} \times 100 = 10{,}000\text{ reads/s}$$
  - Applying a $3\times$ peak factor for viral traffic surges:
  $$\text{Peak Read QPS} = 10{,}000 \times 3 = 30{,}000\text{ reads/s}$$

### 2. Storage Capacity Sizing (5-Year Durability)
- **Single Record Byte Breakdown**:
  - `id`: $8\text{ bytes}$ (BIGINT)
  - `short_code`: $7\text{ bytes}$ (VARCHAR(8))
  - `original_url`: Average $512\text{ bytes}$ (VARCHAR(2048))
  - `user_id`: $8\text{ bytes}$ (BIGINT)
  - `created_at`: $8\text{ bytes}$ (TIMESTAMP)
  - `expires_at`: $8\text{ bytes}$ (TIMESTAMP)
  - B+ tree index & page metadata overhead reserve: $\approx 60\text{ bytes}$
  - **Total per record**: $\approx 611\text{ bytes} \approx 0.6\text{ KB}$

- **5-Year Cumulative Storage Volume**:
  - Total records created over 5 years:
    $$10^7\text{ records/day} \times 365 \times 5 = 1.825 \times 10^{10}\text{ records} (18.25\text{ billion})$$
  - Cumulative database physical storage space:
    $$\text{Total Storage} = 1.825 \times 10^{10} \times 0.6\text{ KB} \approx 1.095 \times 10^{10}\text{ KB} \approx 10.95\text{ TB}$$

> **Architectural Conclusion**:
> A single relational database table typically maintains B+ tree depth stability (3-level index) up to $\approx 20\text{ million rows}$. $18.25\text{ billion records}$ vastly exceeds single-node capacity.
> **The storage tier must employ horizontal database sharding based on `short_code` hash modulos, or adopt a native distributed SQL database (TiDB / CockroachDB)**.

### 3. In-Memory Cache Sizing (Redis Memory)
- Applying the **Pareto 80/20 Rule**: $20\%$ of hot URLs drive $80\%$ of redirection traffic;
- Hot working set active per day $\approx 20\%$ of daily volume plus historical active keys $\approx 2\text{ million} (2 \times 10^6)$ hot links;
- Memory per entry (Key: `short_code` 7B, Value: `original_url` 512B, plus Redis `dictEntry` and `redisObject` overhead $\approx 600\text{ bytes}$);
- **Daily Working Set RAM**:
  $$\text{Daily Hot Cache} = 2 \times 10^6 \times 600\text{ bytes} \approx 1.2\text{ GB}$$
- Caching the past 7 days of high-frequency links under an LRU eviction policy requires:
  $$1.2\text{ GB} \times 7 \approx 8.4\text{ GB}$$

> **Architectural Conclusion**:
> $8.4\text{ GB}$ is a compact memory footprint easily housed in a standard 16GB Redis instance. A primary-replica deployment with Sentinel or Redis Cluster provides high availability and absorbs the 30,000 read QPS effortlessly.

### 4. Network Throughput & Bandwidth Sizing
- **Peak Read Bandwidth**: $30{,}000\text{ reads/s} \times 512\text{ bytes} \approx 15.36\text{ MB/s} \approx 123\text{ Mbps}$;
- **Peak Write Bandwidth**: $200\text{ writes/s} \times 600\text{ bytes} \approx 120\text{ KB/s} \approx 0.96\text{ Mbps}$.
Network egress is well within standard datacenter gigabit NIC thresholds and does not constitute a bottleneck.

---

## Step 4: High-Level Architecture & Schema

### 1. Layered System Architecture Topology
The architecture enforces strict decoupling between stateless compute nodes and external durable state tiers:

```text
[ Clients / Browsers ]
       │
       ▼
[ Anycast DNS / CDN (Edge Network Acceleration) ]
       │
       ▼
[ API Gateway / Reverse Proxy Cluster ]
  (TLS termination, Rate Limiting, WAF inspection)
       │
       ├────────────────────────────────────────┐
       │ (Write path: POST /urls)               │ (Read path: GET /{short_code})
       ▼                                        ▼
[ Stateless URL-Writer Pods ]            [ Stateless Redirect Pods ]
  - Pure stateless compute                   - Pure stateless compute, millisecond response
  - Ticket segment pre-allocation            - In-memory Bloom Filter ingress check
       │                                        │
       ├─ (1. Transaction commit)                ├─ (1. Cache Hit: immediate 302)
       │                                        ▼
       ▼                                 [ Redis Cache Cluster ]
[ Sharded Relational Database ]                 │ (Miss: fallback to replica)
  (MySQL / TiDB Shards)                          ▼
  - Primary handles durable writes       [ DB Read Replicas ]
  - Sharded by hash(short_code)                 │
       │                                        │ (2. Async event emission)
       │                                        ▼
       │                                 [ Kafka / Event Stream ]
       │                                        │
       ▼                                        ▼
[ Elastic Worker Cluster ]               [ Stream Analytics Workers ]
  (Async expired URL cleanup)              (Windowed aggregation, batch writes to ClickHouse)
```

### 2. Database Schema Design (DDL)

#### (1) URL Mapping Primary Table (`urls`)
```sql
CREATE TABLE urls (
    id BIGINT NOT NULL PRIMARY KEY,            -- Globally unique 64-bit ID from ticket generator
    short_code VARCHAR(8) NOT NULL,            -- Base62 encoded 7-char short token
    original_url VARCHAR(2048) NOT NULL,       -- Destination long URL
    user_id BIGINT DEFAULT NULL,               -- Creator ID for tenancy and quotas
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT NULL,         -- Optional expiry timestamp (NULL = permanent)
    UNIQUE KEY uk_short_code (short_code),     -- Unique constraint guarantees zero collision
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### (2) Write Idempotency Ledger (`idempotency_keys`)
```sql
CREATE TABLE idempotency_keys (
    request_id VARCHAR(64) NOT NULL PRIMARY KEY, -- Client UUID intent key
    short_code VARCHAR(8) NOT NULL,              -- Corresponding generated short code
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 3. Protocol Decision: HTTP 301 Moved Permanently vs HTTP 302 Found
This represents the primary protocol-level trade-off in URL shortener design:

| Dimension | HTTP 301 Moved Permanently | HTTP 302 Found / 307 Temporary Redirect |
|---|---|---|
| **Browser Caching** | Browser permanently caches destination locally; subsequent visits **bypass the shortener server entirely** | Browser treats redirect as ephemeral; **every click hits the shortener backend** |
| **Server Load** | Minimal (repeat clicks are drained by client caches) | Must handle all 30,000 QPS (absorbed cleanly by Redis) |
| **Analytics Ingestion** | **Permanently lost** (no visibility into repeat PV/UV, IP, timestamps) | **$100\%$ accurate** (every redirection fires an analytics event) |
| **Configuration Agility**| Extremely rigid (cannot alter target URL or deactivate link prematurely) | Fully agile (server updates destination or revokes link in real time) |
| **Architectural Verdict** | ❌ Only viable for static redirects without metrics | **✅ Mandatory Choice**: Analytics is core business value; use 302 |

---

## Step 5: Deep Dives (Core Trade-offs × 3)

### Deep Dive 1: Global Stateless ID Generation Strategy

A 7-character Base62 token (drawn from `[0-9a-zA-Z]`, yielding $62^7 \approx 3.52\text{ trillion}$ unique combinations) vastly exceeds our 5-year requirement of 18 billion. How do stateless nodes generate these tokens with zero collisions and high throughput?

```text
Option A: Hash Truncation ──► MurmurHash64 ──► Take 43 bits Base62 ──► Check collision in DB ──► (Add salt on retry)
                                                                            │
                                                     [Birthday Paradox triggers explosion of retries at scale]

Option B: Ticket Range Server ──► Central allocator assigns Segment [10001, 20000] ──► AtomicLong increment ──► Base62
                                                                            │
                                                     [Crash skips numbers safely; pure RAM generation @ millions QPS]
```

- **Option A: Long URL Hashing with Collision Resolution**
  - *Mechanism*: Hash the long URL using MD5 or MurmurHash64, extract the first 43 bits, and convert to 7 Base62 characters. If the DB unique constraint detects a collision, append a salt and re-hash.
  - *Fatal Flaw*: At billions of rows, **the Birthday Paradox drastically increases collision frequency**. Each collision incurs an expensive DB lookup and re-hash retry, severely destabilizing write P99 latency.
- **Option B (Recommended): Distributed Ticket Range Server + Deterministic Base62 Encoding**
  - *Mechanism*: The token is a deterministic mathematical representation of a globally unique 64-bit integer ID (e.g., $\text{ID} = 100{,}000{,}000 \iff \text{"a8K9zQ1"}$, **zero possibility of collision**).
  - *Stateless Implementation*:
    1. A lightweight centralized sequencer (MySQL auto-increment ticket table or etcd) dispenses ID segments;
    2. Each stateless API pod requests a **ticket range** upon startup (e.g., Pod A receives $[1000001, 1020000]$, 20,000 IDs), atomically incrementing the global coordinator;
    3. Pod A issues IDs entirely from in-memory `AtomicLong` counters at millions of QPS with **zero network I/O and zero lock contention**;
    4. When consumption reaches $80\%$, an asynchronous thread pre-fetches the next range from the sequencer.
  - *Trade-off Evaluation*: If a pod crashes, unallocated IDs in its buffer are abandoned, causing numerical gaps. In a space of $3.52\text{ trillion}$, gaps are entirely harmless, while decoupling the compute fleet from database locks is an immense operational win.

---

### Deep Dive 2: High-Concurrency Cache Resilience (Penetration, Breakdown, Avalanche)

Handling 30,000 QPS without tiered defenses leaves the persistence layer vulnerable to collapse under edge-case traffic:

```text
[ Incoming Request ] ──► 1. In-memory Bloom Filter Check ──► Absent ──► [ Direct 404, Zero Penetration ]
                                  │ Present
                                  ▼
                         2. Query Redis Cluster
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
             [ Cache Hit ]                  [ Cache Miss ]
          (Immediate 302)                         │
                                            3. Acquire Mutex Lock (NX EX 5)
                                                  │
                                  ┌───────────────┴───────────────┐
                                  ▼                               ▼
                           [ Lock Acquired ]              [ Lock Denied ]
                         Query DB & Backfill Redis         Spin-wait 20ms & Retry
                                  │
                           Apply Jitter TTL
```

1. **Cache Penetration Defense**:
   - Attackers scan non-existent random short codes (e.g., `short.link/invalidXXX`), missing Redis and overwhelming DB read replicas;
   - **Remediation**: Position a **Bloom Filter** at the gateway or inside stateless pod memory. When a URL is created, its hash bits are set. If the Bloom filter reports a code absent, it is **guaranteed absent**, returning HTTP 404 immediately. For rare false positives that miss the DB, cache a sentinel null object with a 30-second TTL.
2. **Cache Breakdown Defense**:
   - A viral short link expires, and thousands of concurrent requests miss Redis simultaneously, storming the database;
   - **Remediation**: Use **distributed mutex locks** (`SET key token NX EX 5`). Only the single request that acquires the lock queries the replica and populates Redis; competing requests spin-wait for 20ms and read from cache.
3. **Cache Avalanche Defense**:
   - Batches of URLs created simultaneously share identical TTLs (e.g., 7 days), expiring together in a massive wave;
   - **Remediation**: Inject $\pm 10\%$ **pseudo-random jitter** into base expiration intervals, scattering cache evictions uniformly across time.

---

### Deep Dive 3: Asynchronous Analytics Decoupling & Micro-batching

Recording access timestamps, client IP addresses, and referrers must not degrade redirection performance:

- **Option A: Synchronous Counter Updates on the Read Path**
  - *Fatal Flaw*: Executing `UPDATE urls SET clicks = clicks + 1 WHERE short_code = ?` during every 302 redirect transforms read traffic into **row-level exclusive write-lock contention**. 30,000 QPS of concurrent row locks exhausts database connection pools in seconds, crashing the cluster.
- **Option B (Recommended): Event Streaming + Windowed Micro-batching**
  - *Mechanism*:
    1. The stateless Redirect Pod issues the `HTTP 302` response, then fires a lightweight JSON event to Kafka topic `click_events` via an asynchronous non-blocking thread:
       `{"code": "a8K9zQ1", "ts": 1789258800, "ip": "1.2.3.4", "ua": "Mobile Safari", "ref": "twitter"}`;
    2. Main redirection latency overhead is $< 1\text{ ms}$;
    3. Independent analytical worker pods (Flink or Go workers) consume from Kafka;
    4. Workers maintain a 10-second tumbling window in memory to aggregate raw hits per token;
    5. Aggregated batches flush periodically into a columnar OLAP database (ClickHouse), while Redis atomically increments coarse aggregate counters for dashboard rendering.
  - *Trade-off Evaluation*: Analytics exhibit eventual consistency (visible after a few seconds), but the critical redirection path gains absolute resilience against downstream analytics outages.

---

## Step 6: End-to-End Life of a Request

Tracing request lifecycles verifies that all components coordinate cleanly in execution:

### 1. Write Timeline (Create Short URL)
```text
1. Client                  -> Sends HTTP POST /api/v1/urls
                              Header: X-Idempotency-Key: "uuid-123"
                              Body: {"url": "https://example.com/very/long/path"}
2. API Gateway             -> Validates auth token, applies rate limiting;
                              Routes request to healthy URL-Writer Pod A.
3. URL-Writer Pod A        -> Queries idempotency_keys table;
                              If key exists, returns existing short_code (Idempotent hit).
4. URL-Writer Pod A        -> Fetches next integer ID from local pre-allocated segment (e.g., 1000042);
                              Performs in-memory Base62 encoding to produce "a8K9zQ1".
5. URL-Writer Pod A        -> Commits local database transaction:
                              ├─ INSERT INTO urls (id, short_code, original_url, ...) VALUES (...);
                              └─ INSERT INTO idempotency_keys (request_id, short_code) VALUES (...);
6. URL-Writer Pod A        -> Asynchronously warms Redis: SET "a8K9zQ1" -> "https://example.com/..."
                              Marks "a8K9zQ1" in the Bloom Filter.
7. URL-Writer Pod A        -> Responds to client with HTTP 201 Created: {"short_url": "https://short.link/a8K9zQ1"}.
```

### 2. Read Timeline (Redirection & Analytics)
```text
1. User Mobile Browser     -> Issues HTTP GET /a8K9zQ1
2. DNS / CDN               -> Anycast routes to the geographically closest edge PoP.
3. API Gateway             -> Load-balances request across healthy Redirect Pod B instances.
4. Redirect Pod B          -> Checks in-memory Bloom filter: if absent, returns HTTP 404 instantly.
5. Redirect Pod B          -> Look up Redis: GET "a8K9zQ1"
                              ├─ [Cache Hit]: Retrieves target URL (Latency ~1-2 ms);
                              └─ [Cache Miss]: Acquires mutex, queries DB replica, populates Redis.
6. Redirect Pod B          -> Returns HTTP 302 Found,
                              Header: Location: "https://example.com/very/long/path"
                              Browser immediately redirects to target page.
7. Redirect Pod B          -> Emits click event payload asynchronously to Kafka "click_events".
8. Analytics Worker        -> Consumes Kafka stream, aggregates counts in memory windows, batch-inserts into ClickHouse.
```

---

## Step 7: Universal Golden Rules & Anti-Patterns

### Three Golden Architectural Principles
1. **Numbers-Driven Architecture (Back-of-the-Envelope First)**:
   Never discuss architecture in a vacuum. Always let QPS dictate caching tiers, 5-year storage volume dictate sharding, and read/write ratios dictate replication schemes.
2. **Explicitly Articulate Trade-offs**:
   Mature engineering is defined by balancing competing forces. Clearly state: "Pre-allocated ID segments eliminate database locking, with the trade-off of discarded IDs upon pod crash; HTTP 302 ensures complete analytics visibility, with the trade-off that the server fleet must absorb all redirection traffic."
3. **Strict Stateless Compute with Externalized Durability**:
   Compute pods must tolerate abrupt `kill -9` terminations without losing authoritative business data.

### Four Critical Anti-Patterns
- ❌ **Anti-Pattern 1: Premature Microservice Sprawl** (Drawing gateways, Kafka, Redis, and ES before defining scale or query access patterns);
- ❌ **Anti-Pattern 2: Relying on Single-Node Auto-Increment IDs** (Collapses under tens of billions of rows and introduces an unscalable single point of failure);
- ❌ **Anti-Pattern 3: Omitting Idempotency Keys** (Treating networks as reliable; missing keys cause duplicate records upon network retry);
- ❌ **Anti-Pattern 4: Selecting HTTP 301 for Dynamic Redirects** (Browsers cache 301s permanently, obliterating down-stream analytics and real-time link control).

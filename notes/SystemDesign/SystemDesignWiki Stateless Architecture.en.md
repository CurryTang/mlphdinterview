# Wiki · Stateless Architecture & State Externalization

In modern distributed system architecture, **"Stateless Service" does not mean that the system as a whole contains no state; rather, it means that the computing nodes (service processes) executing business logic do not exclusively own any non-recoverable, persistent business state**.

Every replaceable instance is fundamentally a pure computational container:
$$\text{Replaceable Compute Instance} = \text{Business Code} + \text{Static/Dynamic Config} + \text{Ephemeral Request Context} + \text{Disposable/Rebuildable Local Cache}$$

As long as service processes do not persist authoritative business facts, any individual instance can experience network partition, hardware failure, or be evicted and terminated by a container orchestrator (such as Kubernetes) without any permanent loss of business data. Reverse proxies and load balancers can transparently route subsequent traffic to any healthy replica in the cluster, unlocking the core engineering capabilities of **Horizontal Scalability (Scale-Out)**, **Self-Healing**, and **Zero-Downtime Rolling Deployments**.

---

## 1 · Why Distributed Systems Prioritize Statelessness

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

## 2 · The Taxonomy of State in Distributed Systems

State does not vanish in a stateless architecture; it is externalized into purpose-built infrastructure engineered for consensus, durability, and high availability. Before architecting a system, every piece of data must be categorized:

| State Category | Concrete Examples | Invariants & Characteristics | Canonical Location | Failure Tolerance |
|---|---|---|---|---|
| **Authoritative State** | User accounts, transactions, financial ledgers, inventory balances | Core Source of Truth; must never be lost; requires strict ACID transactional semantics | Relational DB (MySQL/PostgreSQL) / Distributed SQL / Durable write-ahead logs | Absolutely zero data loss tolerated |
| **Durable Blob** | User media, raw videos, ML weights, compiled analytical reports | High throughput, append-only or immutable binary streams | Distributed Object Store (AWS S3 / GCS / Ceph) | Zero loss; metadata stored in DB, payload in S3 |
| **Coordination State** | Distributed locks, service registry, Leader Epoch, Worker leases | Metadata ensuring mutual exclusion and consensus across nodes | ZooKeeper / etcd / Consul / Redis (with TTL) | Must have TTL/lease timeouts to prevent deadlocks |
| **Rebuildable State** | Cache lookups, materialized views, inverted indices, feature vectors | Derived data reducing pressure on authoritative stores; rebuildable on demand | Redis / Memcached / Local in-memory LRU cache | Fully disposable; reconstructed from source of truth |
| **Request-Local State** | Trace ID, HTTP headers, loop iterators, intermediate tensor activations | Ephemeral to a single request; destroyed upon response completion | Process call stack & volatile heap | Destroyed on crash; client safely retries |

> **The Stateless Litmus Test**:
> To verify whether a service is genuinely stateless, ask:
> **"If this running service process is killed abruptly with `kill -9` this exact millisecond, what business facts are permanently lost?"**
> If the answer is "no business facts are lost; only the single in-flight unconfirmed request needs to be retried by the client or queue", the service conforms to true stateless standards.

---

## 3 · The Three Externalization Pillars

### Pillar 1: Session & Authentication Externalization
- **Stateful Anti-Pattern**: Holding `Map<session_id, UserContext>` inside process memory, requiring sticky cookies at the gateway.
- **Modern Stateless Practice**:
  - **Approach A (Shared Session Store)**: Clients hold an opaque random token. The stateless API queries and caches user state in a distributed Redis cluster with short TTLs. Immediate revocation or ban takes effect instantly by evicting the token from Redis.
  - **Approach B (Cryptographically Signed JWT)**: Clients hold self-contained JSON Web Tokens signed by the auth authority. Stateless API instances locally verify signatures using a shared public key (zero network I/O).
  - *Edge Defense*: Pure JWTs cannot be instantly revoked. Production architectures combine short-lived JWTs (e.g., 15-minute expiry) with a centralized revocation blocklist or token generation version counter in Redis.

### Pillar 2: Blob & Media Externalization
- **Anti-Pattern**: Clients upload gigabytes of media directly to `/tmp` on an API pod, and the API proxies the bytes downstream. Large payloads exhaust pod memory and disk, while crashes destroy upload progress.
- **Production Standard — Presigned URL Direct Upload**:
  1. The client sends metadata (filename, byte size, checksum) to the stateless API;
  2. The API creates a metadata row (`file_id, status=pending`) in the DB and generates a time-limited Presigned Upload URL from the object store SDK;
  3. The client uploads the binary payload directly to S3/GCS, completely bypassing API compute nodes;
  4. Upon durable write completion, the object store fires an EventBridge/Webhook event to update the DB record to `status=ready`.

### Pillar 3: Long-Running Workflow Externalization
- **Anti-Pattern**: A client requests a 10-minute analytics report aggregation, keeping the HTTP socket open. A network blip or pod restart aborts the entire computation.
- **Production Standard — Asynchronous Ingestion & Worker Leasing**:
  1. The client sends `POST /jobs`. The stateless API validates inputs, writes `job_id, status=queued` to the DB, and enqueues the job ID into a message queue (Kafka / RabbitMQ / SQS);
  2. The API immediately returns `HTTP 202 Accepted` with a `Location: /jobs/{job_id}` polling header;
  3. Independent background worker pods claim jobs using **time-bounded leases**;
  4. Workers write artifacts to object storage and update DB status; clients query status via polling or WebSocket notifications.

---

## 4 · Strict Idempotency & Consistency Invariants

Because stateless instances can fail at any time, **network timeouts do not imply execution failure**. Clients must automatically retry upon transient errors, which requires stateless backends to guarantee **strict idempotency**.

### Production-Grade Idempotency Protocol
1. **Client-Generated Idempotency Key**:
   The client assigns a unique UUID to every distinct business intent (e.g., passed in `X-Idempotency-Key: 9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d`).
2. **Authoritative Storage Layer Unique Constraint**:
   Establish a dedicated idempotency ledger in the database:
   ```sql
   CREATE TABLE idempotency_keys (
       idempotency_key VARCHAR(64) PRIMARY KEY,
       user_id BIGINT NOT NULL,
       status VARCHAR(16) NOT NULL, -- 'PENDING', 'SUCCESS', 'FAILED'
       response_body TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```
3. **Atomic Commit & Concurrency De-duplication**:
   - When a request hits any stateless API instance, open a database transaction and execute:
     `INSERT INTO idempotency_keys (idempotency_key, user_id, status) VALUES (?, ?, 'PENDING');`
   - If primary key conflict occurs:
     - If the existing record is `SUCCESS`: return the cached `response_body` immediately without re-executing downstream operations;
     - If the existing record is `PENDING`: another identical request is currently executing; return `HTTP 409 Conflict` or back off and poll briefly;
   - If insertion succeeds: execute the business write within the **same local transaction**, update status to `SUCCESS`, save the response body, and commit.

> **Engineering Rule: Never use Request Body Hash as an idempotency key**.
> A hash conflates repeated user submissions with retries. If a user rapidly taps "Submit" twice intending two separate orders, identical payloads would incorrectly drop the second order. Conversely, minute timestamp variances break hash detection for real retries. The key must bind to the client's single business intent.

---

## 5 · Process Lifecycle & Operational Resilience

To allow stateless instances to be scheduled dynamically and safely in orchestrators like Kubernetes, processes must adhere to strict cloud-native lifecycle boundaries:

```text
       [Pod Startup]
           │
     Load immutable configs & credentials
           │
           ▼
   [Readiness Probe Passes] ─── Traffic ingress commences (LB registers Pod IP)
           │
      Serve incoming requests (Strictly stateless computation)
           │
   [SIGTERM Signal Received] (Triggered by scale-in or rolling deployment)
           │
           ├─ 1. Instantly transition to Unready ─── LB stops routing new requests
           ├─ 2. Stop consuming new tasks from queues
           ├─ 3. Drain in-flight requests (Bounded grace period, e.g., 30s)
           └─ 4. Close DB connection pools and message consumers
           │
       [Process Exits Gracefully (Exit 0)]
```

- **Dual-Probe Isolation (Liveness vs. Readiness)**:
  - **Liveness Probe**: Detects deadlocks or unrecoverable fatal loops. Failure triggers an immediate container kill and restart;
  - **Readiness Probe**: Dictates whether the container can accept live traffic. During warm-up or shutdown draining, setting readiness to false ensures the load balancer immediately pulls the instance from routing endpoints.
- **Worker Lease-Based Coordination**:
  Workers executing asynchronous background jobs never permanently own tasks. A worker obtains a lease in Redis or the DB (e.g., `leased_by=worker_1, lease_until=now() + 60s`) and sends periodic heartbeats. If the worker crashes, the lease expires naturally, and another healthy worker reclaims the task idempotently.

---

## 6 · Architectural Costs & Pressure Redistribution

Stateless compute layers achieve massive operational agility by **redistributing and concentrating pressure onto external shared systems**:
- **Connection Storms**: If a stateless API layer scales out to 1,000 pods, each holding 20 DB connections, the database faces 20,000 concurrent sockets, risking resource collapse. Connection multiplexers (PgBouncer, AWS RDS Proxy) and in-memory caching tiers are essential;
- **Network Latency Overhead**: Every operation traverses the network to retrieve state. Connection pooling, keep-alive reuse, batch queries, and tiered caching are mandatory;
- **Circuit Breaking & Bulkheading**: If downstream databases slow down, stateless compute threads quickly back up and exhaust thread pools. Every external downstream dependency requires isolated timeouts, bulkhead thread caps, and fallback degradation circuits.

### Rational Boundaries for Sticky Routing
Stateless design does not forbid affinity. In **WebSocket gateways, real-time collaborative editing, game loops, and LLM KV cache routing**, keeping a session on a specific node provides substantial memory locality benefits.
> **The Unified Principle**:
> **Affinity is strictly a performance optimization; external durable state guarantees correctness**.
> Cache hits deliver optimal latency; node crashes fail over seamlessly to alternative nodes that rebuild state from the external source of truth.

---

## 7 · Application Server QPS Benchmarks & Capacity Inflection Points

When sizing stateless compute fleets, maintain calibrated reference numbers for application servers (standard 4-core / 8GB to 8-core / 16GB pods):

| Single-Node QPS Level | System State | Engineering Characteristics |
|---|---|---|
| **10 QPS / node** | Minimal Load | Standard for internal admin panels or heavy analytical endpoints. |
| **100 QPS / node** | Idling / Casual | Routine operational zone for general microservices; CPU typically $< 15\%$. |
| **500 QPS / node** | **Healthy Operating Baseline** | Expected production benchmark (including JSON serialization, auth tokens, and 1-2 DB calls). |
| **2,000 QPS / node** | **Optimization Threshold** | Mandatory profiling; inspect GC pauses, thread queue depth, and connection pool saturation. |
| **> 2,000 QPS / node** | **Heavy Load** | Physical ceiling for general business logic; requires L2 caching, batching, or custom serializers. |
| **> 10,000 QPS / node** | **Extreme Throughput** | Rare in general business tiers. Requires minimal logic, pure memory caching, asynchronous event loops (epoll), and zero heavy ORM overhead. |

> **The +30% Knee-of-the-Curve Rule**:
> In load testing or production, if an **incremental 30% increase in request QPS** produces **a non-linear spike in p95/p99 latency, explosive CPU usage, or runaway queue depth**, the system has passed its knee point.
> **Read-heavy + Hundreds of QPS + Database contention $\implies$ Add an in-memory cache (Redis) immediately!**

# Wiki · Idempotency Design Patterns & Implementations

Wiki Navigation: [[SystemDesign00 Overview|00 Blueprint & Numbers]] → [[SystemDesignWiki Idempotency|Wiki · Idempotency]]

In distributed systems and microservices architectures, **Idempotency** is one of the most critical mechanisms to counteract network unreliability, guarantee eventual consistency, and ensure system resilience.

---

## 1 · Mathematical Formulation & Core Motivation

### 1.1 · Mathematical & Engineering Definition
- **Mathematical Definition**: An operation $f$ is idempotent if applying it multiple times yields the same result as applying it once:
  $$f(f(x)) = f(x) \implies f^n(x) = f(x) \quad (\forall n \ge 1)$$
- **Distributed Systems Engineering Definition**: Whether an operation is invoked **once** or **repeatedly with identical arguments**, the resulting **system side effects** remain strictly identical, and the returned response preserves semantic parity across invocations.

### 1.2 · Root Cause: The Distributed Three-State Hazard
In local memory, an execution is binary: Success or Failure. In distributed communications across networks, every RPC, HTTP request, or message exchange carries a third state: **Timeout / Unknown**:

```text
[Client / Caller]                     [Server / Callee]
       │                                     │
       ├─── 1. Invoke Request (e.g., Trigger Run) ─>│ (Success: State written to DB)
       │                                     │
       │< - - - 2. Packet Drop / Network Timeout - - │
       │
(Client assumes failure; triggers retry policy)
       │
       ├─── 3. Retries Identical Request ─────>│ (Without Idempotency: Duplicate creation,
                                              │  double billing, Runner cluster exhaustion)
```

1. **Client-Side Retries**: Packet loss or reverse-proxy timeouts prevent the client from knowing whether the server processed the write; retrying is mandatory.
2. **At-least-once MQ Delivery**: Kafka, RabbitMQ, and SQS re-deliver unacknowledged messages during consumer partition rebalances or node restarts.
3. **Third-Party Webhooks**: GitHub, Stripe, and Shopify retry delivery until an explicit `200 OK` is acknowledged.

---

## 2 · Concrete Case Study: Pipeline Execution Schema (`workflow_run`)

In modern CI/CD orchestrators, a central execution entity is modeled as follows:

```text
┌────────────────────────────────────────────────────────┐
│                   workflow_run Entity                   │
├──────────────┬──────────────┬──────────────────────────┤
│ Field Name   │ Type         │ Semantics                │
├──────────────┼──────────────┼──────────────────────────┤
│ run_id       │ VARCHAR(64)  │ Primary Key              │
│ tenant_id    │ VARCHAR(64)  │ Multi-tenant Identifier  │
│ repo_id      │ VARCHAR(64)  │ Repository ID            │
│ commit_SHA   │ CHAR(40)     │ Git Commit Hash          │
│ event_type   │ VARCHAR(32)  │ Event (push / pr / etc.) │
│ status       │ VARCHAR(32)  │ PENDING / RUNNING / ...  │
│ created_at   │ TIMESTAMP    │ Creation Timestamp       │
└──────────────┴──────────────┴──────────────────────────┘
```

### The Hazard of Missing Idempotency:
- When an engineer runs `git push`, GitHub emits a webhook. If the receiver successfully persists the job but takes too long to respond with `200 OK`, GitHub retries multiple times.
- **Without idempotency**: Multiple independent `run_id`s are spawned for the identical commit. Each instance provisions expensive virtual machines (Runners), exhausting compute quotas, congesting database connection pools, and duplicating billing.

---

## 3 · Six Core Architectural Implementation Paradigms

### 3.1 · Paradigm 1: Relational Database Unique Key Constraints
Using the storage engine's underlying B+Tree unique index as the ultimate physical mutex.

- **Composite Unique Index Design**:
  ```sql
  ALTER TABLE workflow_run 
  ADD UNIQUE KEY uk_tenant_repo_commit_event (tenant_id, repo_id, commit_SHA, event_type);
  ```
- **Execution Pattern**:
  1. `INSERT INTO ... ON DUPLICATE KEY UPDATE run_id = run_id;` (noop on duplicate).
  2. Alternatively, catch the database duplicate key error (`1062` in MySQL), retrieve the existing record, and return it cleanly as a success response.
- **Trade-offs**: Robust and simple, but database lock contention and rollbacks under high-concurrency bursts cause write amplification on primary instances.

### 3.2 · Paradigm 2: Distributed Idempotency Key (Request Token)
The industry standard implemented by major platforms (e.g., Stripe, GitHub REST APIs).

```text
[Client]                         [API Gateway / Redis]                [Business DB]
   │                                       │                                │
   ├── 1. POST /runs                       │                                │
   │   Header: Idempotency-Key: <token> ──>│                                │
   │                                       ├── 2. SET token "LOCK" NX EX 30 │
   │                                       │   (Atomic lock acquisition)    │
   │                                       │                                │
   │                                       ├── 3. Lock Acquired ───────────>│ (Persist business state)
   │                                       │                                │
   │                                       ├── 4. Cache Response:           │
   │                                       │   SET token <response> EX 86400│
   │<── 5. Return HTTP 201 + Payload ──────┴────────────────────────────────┘
   │
   │   (Client retries identical request upon timeout)
   ├── 6. POST /runs (Same Token) ─────────>
   │                                       ├── 7. Token exists with cached payload
   │<── 8. Return cached Payload (HTTP 200) ┘ (Bypasses execution and database write)
```

1. **Client-Supplied Token**: Clients provide an `Idempotency-Key` HTTP header.
2. **Atomic In-Flight Mutex**:
   - Before execution: `SET idemp:{token} "PROCESSING" NX EX 60`.
   - If returns `0`: A concurrent request is already in-flight; return `409 Conflict` or queue.
3. **Response Materialization**: Upon completion, store response body: `SET idemp:{token} '{"run_id":"run_123","status":"PENDING"}' EX 86400`.
4. **Idempotent Replay**: Subsequent duplicates return the cached response immediately.

### 3.3 · Paradigm 3: Distributed Locks & Monotonic State Transitions (CAS)
Ideal for long-running asynchronous background pipelines.

- **Monotonic Progression Invariant**: Lifecycle transitions must be directed acyclic graphs (DAGs):
  $$\text{PENDING} \longrightarrow \text{RUNNING} \longrightarrow \text{COMPLETED / FAILED}$$
- **Optimistic Concurrency Control (CAS)**:
  ```sql
  UPDATE workflow_run 
  SET status = 'RUNNING', started_at = NOW() 
  WHERE run_id = :run_id AND status = 'PENDING';
  ```
- **Evaluation**:
  - `rows_affected == 1`: Runner successfully claimed execution.
  - `rows_affected == 0`: Task already claimed or finalized by another instance; safe to ignore.

### 3.4 · Paradigm 4: Dedicated Deduplication Table
Standard practice for message consumer endpoints handling at-least-once delivery.

- **Schema**:
  ```sql
  CREATE TABLE processed_events (
      event_id VARCHAR(128) PRIMARY KEY,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```
- **Atomic Local Transaction**:
  ```sql
  START TRANSACTION;
  -- 1. Attempt deduplication insertion
  INSERT INTO processed_events (event_id) VALUES (:msg_id);
  -- 2. Execute business insertion
  INSERT INTO workflow_run (...) VALUES (...);
  COMMIT;
  ```
- If the same event arrives twice, step 1 fails immediately with a primary key collision, automatically aborting the transaction.

### 3.5 · Paradigm 5: Deterministic ID Derivation (Deterministic Hash / UUIDv5)
Generating primary keys deterministically from input parameters instead of auto-incrementing integers or random UUIDv4.

- **Hashing Algorithm**:
  $$\text{run\_id} = \text{SHA256}(\text{tenant\_id} + \text{":"} + \text{repo\_id} + \text{":"} + \text{commit\_SHA} + \text{":"} + \text{event\_type})$$
- **Stateless Invariance**: Independent worker nodes calculate the identical 64-character hash given identical inputs.
- **Physical Collision Guard**: Re-insertions naturally collide on the primary key, eliminating duplicate records from inception.

### 3.6 · Paradigm 6: Natural Idempotent Operations
Designing APIs and data modifications to be intrinsically idempotent by mathematical nature:

| Operation Type | Non-Idempotent (Delta / Relative) | Naturally Idempotent Alternative |
| :--- | :--- | :--- |
| **Numerical Updates** | `UPDATE t SET count = count + 1` | `UPDATE t SET count = 5 WHERE version = 1` |
| **State Modifications** | `UPDATE t SET retry = retry + 1` | `UPDATE t SET status = 'CANCELED'` |
| **Collection Storage** | `list.append(item)` (duplicates on retry) | `set.add(item)` / `INSERT IGNORE` |
| **HTTP Semantics** | `POST /workflows/runs` | `PUT /workflows/runs/{deterministic_id}` |
| **File Writing** | Appending binary buffers to file tail | Atomic overwrite by target filename |

---

## 4 · Production Deployment: Dual-Layer Defense for `workflow_run`

A robust production system couples **Redis in-memory gating** with **relational database uniqueness constraints**:

```text
[Webhook / User Action]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 1: Fast Redis Mutex Ingestion Gate               │
│  · Key: lock:run:{tenant}:{repo}:{commit}:{event}      │
│  · Execution: SET key "PROCESSING" NX EX 60            │
│  · Collision: Return 409 Conflict / poll cached result │
└────────────────────────────────────────────────────────┘
           │ (First-time request penetrates)
           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 2: Relational Database Transaction Fortress      │
│  · Unique Constraint: uk_tenant_repo_commit_event      │
│  · State Machine CAS: WHERE status = 'PENDING'         │
│  · Cache Fill: SET idemp:result:{key} payload EX 24h   │
└────────────────────────────────────────────────────────┘
           │
           ▼
[Dispatch Job to Runner Node]
```

---

## 5 · Production Pitfalls & Engineering Traps

1. **TOCTOU Race Hazards (Time-of-Check to Time-of-Use)**:
   - *Flawed Pattern*: `if (!db.exists(run_id)) { db.insert(...); }`. Under concurrent bursts, two requests simultaneously pass the check and clash on insert.
   - *Remedy*: Check and state commitment must be atomic (`INSERT ON DUPLICATE`, Redis `SET NX`).
2. **In-Flight Crashes & Permanent Deadlocks**:
   - If an application node crashes mid-execution after acquiring a lock without a TTL, the lock remains orphaned forever.
   - *Remedy*: Every distributed lock must possess an expiration TTL accompanied by an asynchronous watchdog heartbeat renewal mechanism.
3. **Response Invariance Violations**:
   - Idempotency requires identical client-perceived outcomes. Returning `500 Internal Error` or throwing unhandled DB collision exceptions on duplicates will cause client-side failure loops.
   - *Remedy*: Intercept duplicate requests and return the original successful HTTP status (e.g. `200 OK`) and historical payload snapshot.
4. **Metadata Memory Leaks & Retention TTL**:
   - Idempotency stores must not grow unbounded.
   - *Remedy*: Enforce sliding retention windows (e.g. 24 hours to 7 days) via Redis key TTL or periodic database partitioning drop policies.

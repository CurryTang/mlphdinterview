# System Design 12 · Distributed Crossword Puzzle Solver

Course location: [[SystemDesign11 Notification System|11 Notification System]] → this note → [[SystemDesign99 Glossary|99 Glossary]]

Ingest grid topology and dictionary constraints, prioritize ultra-fast local bit-parallel CSP solving on a single worker, adaptively split heavy-tail combinatorial search trees into coarse disjoint branches, execute speculative multi-worker searches, and commit results via atomic CAS consensus.

![[assets/crossword-solver-whiteboard.png|Crossword Solver System Design Architecture Whiteboard]]

---

## 1. Functional Requirements

1. **Submit Puzzle Solving Job:**
   - Clients submit puzzle grid topology ($M \times N$ dimensions, blank slot definitions with starting coordinates, directions, and lengths), optional pre-filled fixed letters, and tenant identification.
   - The service validates input, returns a globally unique `job_id`, and pins an immutable `dictionary_version` snapshot and `solver_version`.
2. **Status & Result Query / Callback:**
   - Clients poll `GET /v1/jobs/{job_id}` to track real-time execution states (queued, running, completed, timed out).
   - Upon completion, the system returns the full valid grid letter assignment and performance telemetry; supports asynchronous Webhook callbacks.
3. **Cancel In-Flight Job:**
   - Clients invoke `POST /v1/jobs/{job_id}/cancel` to immediately terminate pending or active jobs, releasing compute leases across all worker nodes.
4. **Deterministic Terminal State Semantics:**
   - Job outcomes are strictly partitioned into five mutually exclusive states:
     - `SUCCEEDED`: A full valid fill was discovered and verified by an independent checker.
     - `UNSAT` (Unsatisfiable): The search space was completely and mathematically exhausted with zero valid fills under the specified dictionary.
     - `TIMED_OUT`: Hard wall-clock or CPU budget expired prior to complete search space coverage.
     - `CANCELLED`: Explicitly aborted by client request.
     - `FAILED`: Unrecoverable internal system errors (persistent crashes, corrupted data).

**Scope & Technical Assumptions:**
- **Objective:** Find **any single valid constraint-satisfying solution**; optimization over natural-language clues or probabilistic ranking is out of scope.
- **Global Uniqueness:** Repeated words within the same puzzle are strictly forbidden (enforcing a global `AllDifferent` constraint).
- **Core Abstraction:** The system is an intersection of a **Constraint Satisfaction Problem (CSP)** engine and a **distributed tree-search scheduler**. Over 90% of requests are resolved locally on a single worker via bit-parallel inference; only heavy-tail outliers trigger distributed tree partitioning.

---

## 2. Non-Functional Requirements & Architectural Mappings

| Non-Functional Requirement | Target SLA | Architectural & Algorithmic Strategy |
| :--- | :--- | :--- |
| **Latency** | Common puzzles P95 < 10s; pathological/hard puzzles allow async processing with a 10-minute hard timeout | **Single-worker bit-parallel CSP engine first**: Workers preload bitmap-indexed dictionaries with MRV and arc consistency (AC-3/FC); puzzles exceeding 5s are dynamically escalated to distributed subtree splitting. |
| **Throughput & Scale** | Peak submission rate of 100 new jobs/s; maintain 1,000+ active concurrent in-flight jobs | Stateless ingestion API; default 1 vCPU allocation per job; baseline compute pool sized at 400 vCPU with autoscaling; zero remote dictionary IO overhead. |
| **Correctness** | **Zero false positives, zero false UNSAT**; `TIMED_OUT` must never be misreported as `UNSAT` | All worker candidate fills must pass an independent verifier; `UNSAT` is emitted **only** after barrier synchronization confirms 100% of disjoint search branches are completely exhausted. |
| **Reliability & Fault Tolerance**| Worker process crashes, OOMs, or socket drops must not drop tasks | Task leasing mechanism (`lease_until` + heartbeat); orphaned tasks automatically re-queued on lease expiry; atomic CAS consensus prevents split-brain completion. |
| **Multi-Tenant Isolation & Cost**| Prevent adversarial or pathological puzzles from causing head-of-line blocking across tenants | Deficit Round Robin (DRR) / Weighted Fair Queuing (WFQ) across per-tenant queues; strict limits on per-job CPU-seconds, memory footprint, and max fanout parallelism. |
| **Reproducibility** | Repeated submissions of the same puzzle must produce deterministic, identical verdicts | Pin immutable `dictionary_version` and `solver_version`; index past results in a result cache by `(puzzle_hash, dictionary_version)` for instant replay. |

---

## 3. Capacity Estimation & Compute Modeling

### 3.1 Core Throughput & Concurrency Metrics

- **Peak Submission Rate ($\lambda$):** 100 jobs/s.
- **Average Compute Consumption ($\bar{t}$):** Typical puzzles consume 2 CPU-seconds.
- **Sustained Compute Demand:**
  $$\text{Sustained vCPU} = 100\text{ jobs/s} \times 2\text{ CPU-s} = 200\text{ vCPU}$$
- **Baseline Worker Pool (2x Headroom):**
  $$\text{Baseline Worker Pool} = 200\text{ vCPU} \times 2 = 400\text{ vCPU}$$
- **Active In-Flight Concurrent Jobs:**
  Assuming an end-to-end target latency of 10 seconds for common puzzles:
  $$\text{In-Flight Active Jobs} = 100\text{ jobs/s} \times 10\text{ s} = 1,000\text{ active concurrent jobs}$$
  This matches the system scale requirement of 1,000+ simultaneous in-flight puzzles.

### 3.2 Dictionary Storage & Memory Footprint

- **Raw Text Footprint:** 1,000,000 words $\times$ average 10 chars $\approx 10\text{ MB}$ raw text.
- **Dense Positional Bitmap Index:**
  - Segment words into length buckets (lengths 3 to 21).
  - For each length $L$, construct $L \times 26$ bitmaps, where each bitmap has length equal to the number of words $W_L$.
  - Total bitmap memory: $1,000,000 \times 10\text{ positions} \times 26\text{ bits} \approx 260\text{M bits} \approx 32.5\text{ MB}$.
- **Worker Memory Working Set:**
  Including word array tables, character indexes, and backtracking trail stacks, total resident memory per worker is **50 MB – 200 MB**.
- **Key Architectural Takeaway:**
  **Dictionary storage and network bandwidth are non-issues**. Dictionaries must reside permanently in worker memory or be shared via `mmap`. Network queries to external databases during the inner solving loop are strictly prohibited.
- **Primary Operational Hazard:** **Combinatorial CPU heavy tail** and thread starvation caused by pathological puzzles.

---

## 4. High-Level Architecture

### 4.1 Vanilla Direct Architecture & Fatal Flaws

```text
[Clients] ──▶ [API Service] ──▶ [Single Global FIFO Queue] 
                                          │
                                          ▼
                               [Naive DFS Workers]
                               (Linear scan Dict DB)
```

**Fatal Production Flaws:**
1. **Head-of-Line Blocking & System Collapse:** Crossword solving is NP-complete. A single pathological $25 \times 25$ puzzle can trigger a search space of $10^{18}$ states. A shared FIFO queue causes heavy-tail puzzles to saturate all worker threads, starving fast 50ms jobs.
2. **Network Latency Amplification:** Naive engines issuing database queries per slot assignment (`SELECT word WHERE len=5 AND char[2]='a'`) incur millisecond network round-trips multiplied over millions of backtracking steps.
3. **False UNSAT & Data Corruption:** In-flight tasks killed on timeout are erroneously marked as unsatisfiable, misleading clients with mathematically incorrect results.

---

### 4.2 Production Layered Architecture

```text
[Clients / Batch Callers]
           │
           │ 1. POST /v1/jobs (Puzzle Grid + Fixed Chars)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Ingestion & Admission Layer                              │
│   [API Gateway] ──(Auth & Per-Tenant Rate Limit)            │
│          │                                                  │
│          ▼                                                  │
│   [Job API Service]                                         │
│          │                                                  │
│          ├── Query [Result Cache] (puzzle_hash + dict_ver)  │
│          ▼                                                  │
│   [Job Store (PostgreSQL / DynamoDB)]                       │
│   (job_id, tenant_id, status, budget, pinned_versions)      │
└──────────┬──────────────────────────────────────────────────┘
           │ 2. Submit to Scheduler
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Fair Scheduling & Admission Control Layer                │
│   [Scheduler / Admission Controller]                        │
│          ├── Estimate CPU Backlog                           │
│          ├── Weighted Fair Queuing (DRR / WFQ)              │
│          ├───────────────────────┬──────────────────────────┤
│          ▼                       ▼                          │
│   [Normal Job Queue]      [Hard / Branch Queue]             │
│   (Default: 1 worker)     (Split coarse subtrees)           │
└──────────┬───────────────────────┬──────────────────────────┘
           │                       │
           ▼                       ▼
┌───────────────────────────┐ ┌───────────────────────────────┐
│ 3. Normal Solver Workers  │ │ 4. Hard-Job Distributed Pool  │
│   [Solver Worker Pool]    │ │   [Hard-Job Workers]          │
│   • Local Bitmap Dict     │ │   • Evaluate disjoint branch  │
│   • MRV + Degree + AC-3   │ │   • Bounded depth/backtracks  │
│   • Watchdog: > 5s slow?  │ │   • Heartbeat lease claim     │
└──────────┬────────────────┘ └───────────┬───────────────────┘
           │ Escalation Trigger            │ Candidate Fill / Branch Done
           ▼                               ▼
┌───────────────────────────┐ ┌───────────────────────────────┐
│ [Adaptive Splitter]       │ │ 5. Verifier & CAS Consensus   │
│ • Hard-Job Coordinator    │ │   [Independent Verifier]      │
│ • Shallow branching cut   │ │   • Strict all-diff check     │
│ • Disjoint subtrees to MQ │ │   • Verify cross intersections│
└───────────────────────────┘ │          │                    │
                              │          ▼                    │
                              │   [CAS Result Commit]         │
                              │   • Winner writes SUCCEEDED   │
                              │   • Broadcast Cancel Siblings │
                              │   • Barrier: all done → UNSAT │
                              └───────────────────────────────┘
```

**Subsystem Responsibilities:**
1. **Ingestion & Admission Layer:** Validates grid coordinates and generates a canonical `puzzle_hash`. Checks the Result Cache for instant deduplication hits. Writes task metadata into the durable `puzzle_jobs` store with pinned dictionary and solver versions.
2. **Scheduling & Queue Isolation Layer:** Uses Deficit Round Robin (DRR) to maintain strict tenant fairness. Segregates traffic between the **Normal Queue** and the **Hard / Branch Queue** to prevent head-of-line blocking.
3. **Single-Worker CSP Engine (Normal Pool):** Resolves >90% of puzzles on a single node. Employs local in-memory bitmap indexes, Minimum Remaining Values (MRV), and arc consistency pruning. A local watchdog monitors progress: if unsolved within 5 seconds, the task escalates to distributed tree splitting.
4. **Adaptive Subtree Splitter & Coordinator:** For escalated hard jobs, identifies shallow high-branching slots and partitions their domains into $K$ disjoint subsets, publishing coarse subtree tasks to the Hard Queue.
5. **Independent Verifier & CAS Consensus:** Validates every candidate fill across all intersections and global uniqueness constraints. The winning worker executes a CAS update to mark the job `SUCCEEDED`, followed by a cancellation broadcast to terminate sibling workers immediately.

---

### 4.3 Core Data Model

#### 1. `puzzle_jobs` (Primary Job Table)
```sql
CREATE TABLE puzzle_jobs (
    job_id               VARCHAR(64) PRIMARY KEY,      -- Global UUID
    tenant_id            VARCHAR(64) NOT NULL,         -- Tenant identifier
    puzzle_hash          VARCHAR(64) NOT NULL,         -- Canonical hash for caching
    grid_width           INT NOT NULL,                 -- Grid width M
    grid_height          INT NOT NULL,                 -- Grid height N
    raw_puzzle_json      JSONB NOT NULL,               -- Slot definitions & fixed chars
    dictionary_version   VARCHAR(32) NOT NULL,         -- Pinned immutable dictionary version
    solver_version       VARCHAR(32) NOT NULL,         -- Pinned solver engine version
    status               VARCHAR(16) NOT NULL,         -- 'ACCEPTED', 'RUNNING', 'SUCCEEDED', 'UNSAT', 'TIMED_OUT', 'CANCELLED', 'FAILED'
    solution_json        JSONB,                        -- Completed grid assignment matrix
    cpu_ms_spent         INT NOT NULL DEFAULT 0,       -- Cumulative CPU runtime in milliseconds
    max_budget_cpu_ms    INT NOT NULL,                 -- Maximum permitted CPU budget (e.g. 600,000 ms)
    hard_timeout_at      TIMESTAMP WITH TIME ZONE NOT NULL, -- Wall-clock hard expiration
    created_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    finished_at          TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_jobs_tenant_status ON puzzle_jobs(tenant_id, status);
CREATE INDEX idx_jobs_cache_lookup ON puzzle_jobs(puzzle_hash, dictionary_version, status);
```

#### 2. `search_subtrees` (Disjoint Subtree Partition Table)
```sql
CREATE TABLE search_subtrees (
    subtree_id           VARCHAR(64) PRIMARY KEY,      -- Subtree UUID
    job_id               VARCHAR(64) NOT NULL REFERENCES puzzle_jobs(job_id) ON DELETE CASCADE,
    branch_index         INT NOT NULL,                 -- Branch ordinal (0 .. K-1)
    prefix_assignment    JSONB NOT NULL,               -- Partial assignment path defining this subtree
    status               VARCHAR(16) NOT NULL,         -- 'PENDING', 'CLAIMED', 'EXHAUSTED', 'SOLVED', 'CANCELLED'
    worker_id            VARCHAR(64),                  -- Worker node holding the active lease
    lease_until          TIMESTAMP WITH TIME ZONE,     -- Heartbeat lease expiration timestamp
    attempt_count        INT NOT NULL DEFAULT 0,       -- Number of claim attempts
    updated_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_subtrees_job_status ON search_subtrees(job_id, status);
CREATE INDEX idx_subtrees_lease ON search_subtrees(status, lease_until) 
WHERE status = 'CLAIMED';
```

#### 3. `dictionary_snapshots` (Immutable Dictionary Catalog)
```sql
CREATE TABLE dictionary_snapshots (
    version_id           VARCHAR(32) PRIMARY KEY,      -- E.g. 'en_us_v20260901'
    word_count           INT NOT NULL,                 -- Word entry count
    storage_uri          TEXT NOT NULL,                -- Object store URI for binary index artifacts
    checksum_sha256      VARCHAR(64) NOT NULL,         -- Verification checksum
    created_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

---

### 4.4 State Machine & Deterministic Invariants

```text
       [ACCEPTED]
           │
           ▼
       [RUNNING] ──▶ (Trigger split) ──▶ [SPLIT_RUNNING]
      /    │    \                          /    │         /     │     \                        /     │         ▼      ▼      ▼                      ▼      ▼      ▼
[SUCCEEDED] [TIMED_OUT] [CANCELLED]   [SUCCEEDED] [TIMED_OUT] [CANCELLED]
                         \                    /
                          ▼                  ▼
                                  [UNSAT] 
                 (Guaranteed: Every disjoint branch is EXHAUSTED)
```

**State Invariants:**
1. **`TIMED_OUT != UNSAT`**: A job halted due to budget or wall-clock expiration prior to complete search tree exploration must always be assigned `TIMED_OUT`. Emitting `UNSAT` without complete verification is strictly prohibited.
2. **`UNSAT` Barrier Synchronization**: `UNSAT` is written if and only if all partitioned subtree records transition to `EXHAUSTED`:
   ```sql
   UPDATE puzzle_jobs 
   SET status = 'UNSAT', finished_at = NOW() 
   WHERE job_id = :job_id AND status IN ('RUNNING', 'SPLIT_RUNNING')
     AND NOT EXISTS (
       SELECT 1 FROM search_subtrees 
       WHERE job_id = :job_id AND status != 'EXHAUSTED'
     );
   ```
3. **Winner CAS Race**: When any worker verifies a solution, it executes an atomic CAS update:
   ```sql
   UPDATE puzzle_jobs 
   SET status = 'SUCCEEDED', solution_json = :solution, finished_at = NOW()
   WHERE job_id = :job_id AND status IN ('RUNNING', 'SPLIT_RUNNING');
   ```
   If 1 row is affected, the worker wins the race and triggers a broadcast cancellation across sibling tasks.

---

## 5. Architectural Deep Dives

### 5.1 Solver Latency & Search Pruning

Formulating the puzzle as a CSP: each blank slot is a variable $X_i$ with domain $D(X_i)$ consisting of dictionary words matching length $L_i$. Crossing slots $(X_i, X_j)$ must agree on intersecting characters. A global `AllDifferent` constraint enforces word uniqueness.

```text
Option 1: Naive DFS with Linear Dictionary Scans
[Slot 1: Len 5] ──Linear scan 10,000 words──▶ Assign "APPLE"
    └── [Slot 2: Cross pos 3] ──Linear scan 8,000 words for 'P'──▶ ... (Backtrack on conflict)

Option 2: Bit-Parallel CSP Engine with MRV & Arc Consistency (Recommended)
[Slot 1] ──Pick slot with Minimum Remaining Values (MRV = 4) ──▶ Fast bitwise AND
    └── Forward Checking cascades across intersections ──▶ Prunes dead branch at depth 2
```

| Dimension | Option 1: Naive DFS + Dictionary Scans | Option 2: Bit-Parallel CSP Engine (Recommended) |
| :--- | :--- | :--- |
| **Search Space Complexity** | Exponential explosion ($O(D^N)$). Inner loop lacks lookahead, leading to thrashing. | Minimum Remaining Values (MRV) + Degree heuristics reduce branching factor by >99%. |
| **Intersection Matching** | Evaluates string characters sequentially, causing branch mispredictions and memory stalls. | **Bit-Parallelism**: 64 words evaluated concurrently per CPU cycle using bitwise AND on positional bitmasks. |
| **Memory Footprint** | Negligible (few MBs). | 50 MB – 200 MB per worker for precomputed bitmasks, Trie structures, and trail stacks. |
| **Tail Latency** | Uncontrolled; complex grids frequently exceed minutes. | Highly predictable; common 15x15 puzzles solve in 50ms – 500ms. |
| **Recommendation** | Unusable for production services. | **Mandatory foundation** for industrial-scale CSP solvers. |

---

### 5.2 Completion Semantics & Fault-Tolerant Consensus

When distributed workers explore disjoint subtrees, the platform must coordinate speculative execution, handle worker crashes, and maintain result correctness without distributed deadlocks.

```text
Subtree Lease Claim & CAS Winner Sequence:
[Worker A (Subtree 1)] ──Acquire lease (lease_until = NOW()+30s)──▶ [DB]
[Worker B (Subtree 2)] ──Acquire lease (lease_until = NOW()+30s)──▶ [DB]
       │
Worker A finds a valid fill:
  1. [Worker A] ──▶ [Independent Verifier] (recheck crossing + all-diff) ──▶ PASS
  2. [Worker A] ──▶ CAS (UPDATE jobs SET status='SUCCEEDED' WHERE status='RUNNING')
  3. CAS succeeds! ──▶ Broadcast Cancel signal via Redis PubSub ──▶ Worker B stops
```

| Dimension | Option 1: Distributed Exactly-Once Coordination | Option 2: At-Least-Once + Leases & CAS Consensus (Recommended) |
| :--- | :--- | :--- |
| **Fault Recovery** | Requires 2PC or distributed locks; worker crashes risk leaving locks permanently orphaned. | **Robust self-healing**: Crashed workers drop heartbeats; uncompleted tasks are reclaimed on lease expiry. |
| **Consistency Guarantee** | Fragile attempt to prevent duplicate execution across distributed workers. | **End-to-end idempotency**: Idempotent `branch_id` and database CAS ensure exactly one winner commits. |
| **Speculative Compute** | Difficult to cleanly interrupt speculative sibling execution. | Winning worker immediately emits a cancellation token to terminate sibling tasks and reclaim CPU. |
| **Coordination Overhead**| Heavy synchronization latency and distributed state management. | Lightweight local execution; coordination occurs only on branch completion or solution discovery. |
| **Recommendation** | Not viable for high-throughput search workloads. | **Industry standard** for resilient distributed search platforms. |

---

### 5.3 Resource Isolation & Heavy-Tail Governance

Puzzle workloads exhibit compute variance spanning five orders of magnitude (10ms to 10 minutes).

```text
Adaptive Coarse Splitting Flow:
[Normal Worker] Detects local search duration exceeding 5s / backtrack quota
       │
       ▼ Select shallow slot with high branching factor (e.g. Slot 3 has 50 valid candidates)
[Adaptive Splitter]
       ├── Splits domain into 5 coarse partitions (10 candidates per prefix branch)
       └── Enqueues into [Hard / Branch Queue]
               ├── [Hard Worker 1] Solves Branch 0 (words 0..9)
               ├── [Hard Worker 2] Solves Branch 1 (words 10..19)
               └── ...
```

| Dimension | Option 1: Single Global Shared FIFO Queue | Option 2: Fair Queuing + Pool Separation + Coarse Splitting (Recommended) |
| :--- | :--- | :--- |
| **Multi-Tenant Protection**| **None**. A single tenant submitting 50 pathological puzzles starves the entire worker cluster. | **Strong isolation**. Deficit Round Robin (DRR) enforces fair throughput per tenant with strict CPU quotas. |
| **Queue Head-of-Line** | Fast 50ms jobs are trapped behind multi-minute pathological jobs. | **Traffic separation**. Fast jobs stay in the Normal Queue; escalated tasks move to the Hard Queue. |
| **Splitting Granularity** | Fine-grained DFS node distribution floods message queues with millions of tiny tasks. | **Coarse-grained cuts**: Subtrees are partitioned only at shallow depths (depth 1–2), bounding queue overhead. |
| **Autoscaling Metric** | Relying on raw queue message count causes severe mis-scaling. | **Scaled on estimated queued CPU-seconds** and worker utilization. |
| **Recommendation** | Unacceptable for shared multi-tenant infrastructures. | **Standard production pattern** for resilient, cost-governed SaaS platforms. |

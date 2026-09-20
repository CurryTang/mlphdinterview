# System Design 12 · Distributed Crossword Puzzle Solver

Course location: [[SystemDesign11 Notification System|11 Notification System]] → this note → [[SystemDesign99 Glossary|99 Glossary]]

Ingest grid topology and dictionary constraints, adaptively route jobs based on combinatorial complexity; prioritize local bit-parallel CSP solving for standard puzzles; escalate pathological heavy-tail puzzles to a Distributed Depth-First Search (Distributed DFS) cluster leveraging root frontier expansion, compact state replay, coarse shallow work stealing, credit conservation, and atomic CAS consensus for industrial-scale throughput.

![[assets/crossword-solver-whiteboard.png|Crossword Solver System Design Architecture Whiteboard]]

---

## 1. Functional Requirements

1. **Submit Puzzle Solving Job:**
   - Clients submit puzzle grid topology ($M 	imes N$ dimensions, blank slot definitions with starting coordinates, directions, and lengths), optional pre-filled fixed letters, and tenant credentials.
   - The ingestion tier performs static topology analysis to assess graph complexity; returns a globally unique `job_id`, pinning an immutable `dictionary_version` snapshot and `solver_version`.
2. **Status & Result Query / Callback:**
   - Clients poll `GET /v1/jobs/{job_id}` to track real-time execution states (queued, local CSP solving, distributed DFS expanding, completed, timed out).
   - Upon completion, the system returns the full valid grid letter assignment and performance telemetry; supports asynchronous Webhook callbacks.
3. **Cancel In-Flight Job:**
   - Clients invoke `POST /v1/jobs/{job_id}/cancel` to immediately terminate pending or active jobs, cascading distributed cancellation tokens to release compute leases across all worker nodes.
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
- **Core Abstraction:** The system unites a **Constraint Satisfaction Problem (CSP)** engine with a **distributed tree-search (Distributed DFS)** scheduler. Not all puzzles run distributed: over 90% of requests are resolved locally on a single worker within milliseconds; only pathological graph topologies or dynamic timeouts escalate to Distributed DFS.

---

## 2. Non-Functional Requirements & Architectural Mappings

| Non-Functional Requirement | Target SLA | Architectural & Algorithmic Strategy |
| :--- | :--- | :--- |
| **Latency** | Common puzzles P95 < 10s (majority 50ms~500ms); pathological/hard puzzles allow async processing with a 10-minute hard timeout | **Dual-path routing + single-worker first**: Fast local bit-parallel CSP engine (Bitmap Index + AC-3 + MRV) resolves common puzzles in milliseconds; complex puzzles route via static classifier or 5s watchdog into Distributed DFS. |
| **Throughput & Scale** | Peak submission rate of 100 new jobs/s; maintain 1,000+ active concurrent in-flight jobs | Stateless ingestion API; default 1 vCPU allocation per job; baseline compute pool sized at 400 vCPU with autoscaling; zero remote dictionary IO overhead. |
| **Correctness** | **Zero false positives, zero false UNSAT**; `TIMED_OUT` must never be misreported as `UNSAT` | All worker candidate fills must pass an independent verifier; `UNSAT` is emitted **only** after credit conservation verifies 100% of disjoint search branches are completely exhausted. |
| **Reliability & Fault Tolerance**| Worker process crashes, OOMs, or socket drops must not drop tasks | Task leasing mechanism (`lease_until` + heartbeat); orphaned tasks automatically re-queued on lease expiry; atomic CAS consensus prevents split-brain completion. |
| **Multi-Tenant Isolation & Cost**| Prevent adversarial or pathological puzzles from causing head-of-line blocking across tenants | Deficit Round Robin (DRR) / Weighted Fair Queuing (WFQ) across per-tenant queues; strict limits on per-job CPU-seconds, memory footprint, and max fanout parallelism. |
| **Reproducibility** | Repeated submissions of the same puzzle must produce deterministic, identical verdicts | Pin immutable `dictionary_version` and `solver_version`; index past results in a result cache by `(puzzle_hash, dictionary_version)` for instant replay. |

---

## 3. Capacity Estimation & Compute Modeling

### 3.1 Core Workload Metrics

- **Peak Submission Rate ($\lambda$):** 100 jobs/s.
- **Average Compute Cost ($ar{t}$):** Standard puzzles consume 2 CPU-seconds on average (local bit-parallel solve takes 50ms~500ms, plus queuing margin).
- **Sustained Compute Demand:**
  $$	ext{Sustained vCPU} = 100	ext{ jobs/s} 	imes 2	ext{ CPU-s} = 200	ext{ vCPU}$$
- **Headroom Pool (2x Peak Safety Factor):**
  $$	ext{Baseline Worker Pool} = 200	ext{ vCPU} 	imes 2 = 400	ext{ vCPU}$$
- **Active Concurrent Jobs:**
  Assuming an average 10-second lifecycle window (queuing + solving):
  $$	ext{In-Flight Active Jobs} = 100	ext{ jobs/s} 	imes 10	ext{ s} = 1,000	ext{ active concurrent jobs}$$

### 3.2 Dictionary Footprint & Memory Analysis

The solver relies on an extensive candidate vocabulary (e.g., a 1-million-word English dictionary):
- **Raw Text Size:** 1,000,000 words $	imes$ 10 characters average $pprox 10	ext{ MB}$ plain text.
- **Dense Positional Bitmap Index:**
  - Partitioned into length buckets (e.g., word lengths from 3 to 21 characters).
  - For each length bucket $L$, maintain an $L 	imes 26$ bitmap matrix where each bitmap has length equal to the number of candidate words $W_L$.
  - Total bitmap memory for 1M words: $1,000,000 	imes 10	ext{ positions} 	imes 26	ext{ bits} pprox 260	ext{M bits} pprox 32.5	ext{ MB}$.
- **Per-Worker Memory Overhead:**
  Including auxiliary indexing, Trie structures, and local backtrack trail stacks, each worker allocates only **50 MB ~ 200 MB** of RAM.
- **Key Architectural Deduction:**
  **Dictionary storage and network transit are NOT the system bottleneck.** The vocabulary must reside in local immutable memory or shared via `mmap`. Search loops must NEVER query remote databases or Redis. The true system bottleneck is **combinatorial search space explosion** and multi-tenant resource contention.

### 3.3 Distributed DFS Scheduling & Network Overhead Analysis

In Distributed DFS, inefficient state serialization will degrade cluster networking:
- **Naive Full-State Serialization (Anti-pattern):** Serializing the full CSP solver heap and variable domains yields $1	ext{ MB} \sim 5	ext{ MB}$ per branch. Fanout across 1,000 branches generates gigabytes of traffic and tens of milliseconds in serialization overhead.
- **Compact Prefix Encoding (Production Pattern):**
  A subtree is uniquely identified by its partial assignment path (e.g., `[(slot_0, "PLANET"), (slot_3, "LASER")]`).
  $$	ext{Payload Size} = K_{	ext{prefix\_slots}} 	imes (4	ext{ bytes slot\_id} + 16	ext{ bytes word}) pprox 64 \sim 256	ext{ bytes}$$
- **Near-Zero Transit Latency:** Sub-kilobyte descriptors transit a 10Gbps cluster link in $< 0.1	ext{ ms}$. Receiving workers replay forward checking (AC-3) against the local immutable bitmap index in $< 0.2	ext{ ms}$, reconstructing the full constraint state with zero network friction.

---

## 4. Architecture Design · High-Level Architecture

### 4.1 Naive Design & Fatal Bottlenecks

```text
[Clients] ──▶ [API Service] ──▶ [Single Global FIFO Queue] 
                                          │
                                          ▼
                                [Naive DFS Workers]
                                (Linear scan Dict DB)
```

**Why does this fail?**
1. **Head-of-Line Blocking & System Freezes:** Crossword solving is NP-complete. A single pathological $25 	imes 25$ puzzle can trigger $10^{18}$ search nodes, exhausting all workers in a shared FIFO queue and starving sub-second tasks.
2. **Remote Dictionary IO Storms:** Querying remote datastores for matching words per branch multiplies network RTT by millions of iterations, turning 50ms searches into multi-hour outages.
3. **Indiscriminate Distributed Overhead:** Naively distributing trivial 10ms puzzles forces them through 20ms~50ms queuing, leasing, and consensus overheads, degrading aggregate QPS by 10x.
4. **False UNSAT Verifications:** Timing out tasks and reporting them as "no solution" violates correctness SLAs.

---

### 4.2 Production Layered Architecture with Distributed DFS Engine

```text
[Clients / Batch Callers]
           │
           │ 1. POST /v1/jobs (Grid Topology + Fixed Letters)
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Ingestion & Admission Layer                                              │
│   [API Gateway] ──(Auth & Rate Limit)                                       │
│          │                                                                  │
│          ▼                                                                  │
│   [Job API Service]                                                         │
│          ├── Query [Result Cache] (puzzle_hash + dict_version)              │
│          ├── [Static Complexity Classifier] ── Extract (M×N, density, lens) │
│          ▼                                                                  │
│   [Job Store (PostgreSQL / DynamoDB)]                                       │
│   (job_id, tenant_id, status, budget, pinned_versions, path_mode)           │
└──────────┬──────────────────────────────────────────────────────────────────┘
           │ 2. Route Job via Classifier
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Fair Scheduling & Dual-Queue Layer                                       │
│   [Fair Scheduler / DRR Engine]                                             │
│          ├── Calculate Estimated CPU Backlog (seconds)                      │
│          ├── Tenant Fair Queuing (Weighted Fair Queuing / DRR)              │
│          ├───────────────────────────┬──────────────────────────────────────┤
│          │ (Standard Puzzles: >90%)  │ (Static Pathological or Escalated)   │
│          ▼                           ▼                                      │
│   [Normal Job Queue]          [Hard / Distributed DFS Queue]                │
│   (Default: 1 worker)         (Coarse Disjoint Subtree Partitions)          │
└──────────┬───────────────────────────┬──────────────────────────────────────┘
           │                           │
           ▼                           ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────────┐
│ 3. Normal Solver Workers      │ │ 4. Distributed DFS Worker Pool            │
│   [Solver Worker Pool]        │ │   [Distributed DFS Workers]               │
│   • Local Bitmap Dict (mmap)  │ │   • Pull Subtree (prefix assignment)      │
│   • MRV + Degree + AC-3       │ │   • Replay AC-3 Forward Checking (<0.2ms) │
│   • Watchdog: > 5s or budget? │ │   • Deep DFS with Local Trail Stack       │
│   • >90% exit here (10~500ms) │ │   • Work-Stealing: Shallowest Choice Cut  │
└──────────┬────────────────────┘ └───────────┬───────────────────────────────┘
           │ Dynamic Watchdog Escalation      │ Solution Discovered / Exhaust │
           ▼                                   ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────────┐
│ [Adaptive Splitter]           │ │ 5. Verifier & CAS Consensus Engine        │
│ • Root BFS Frontier Expansion │ │   [Independent Verifier]                  │
│ • Disjoint Subtrees Cut (K)   │ │   • Strict all-diff + cross letter check  │
│ • Assign Credit Weights (W/K) │ │          │                                │
│ • Push to Distributed Queue   │ │          ▼                                │
└───────────────────────────────┘ │   [CAS Winner Commit]                     │
                                  │   • Atomic UPDATE status='SUCCEEDED'      │
                                  │   • Broadcast Cancel to all siblings      │
                                  │   • Credit Accumulation Barrier → UNSAT   │
                                  └───────────────────────────────────────────┘
```

**Layer Responsibilities:**
1. **Ingestion, Caching & Static Gate:**
   - Normalizes input to compute `puzzle_hash`; checks cache for instant hits.
   - **Static Complexity Classifier**: Evaluates grid geometry and intersection density. Pathological large/sparse grids **bypass single-worker execution straight to Distributed DFS**.
2. **Fair Scheduling & Dual Queues:**
   - Enforces Deficit Round Robin (DRR) to eliminate cross-tenant starvation.
   - Segregates the **Normal Queue** from the **Distributed DFS Queue**.
3. **Single-Worker Local CSP Pool:**
   - Resolves >90% of traffic using local bitmap indexing, MRV variable ordering, and AC-3 forward checking (50ms~500ms).
   - Monitored by a watchdog: exceeding 5s wall-clock or $10^6$ backtracks triggers dynamic escalation to Distributed DFS.
4. **Distributed DFS Worker Pool:**
   - Processes escalated and statically heavy puzzles. Reconstructs search frontiers instantly and executes native deep DFS.
   - Implements **Coarse Work Stealing**: idle workers split and claim unvisited branches from the shallowest choice points of busy workers to balance irregular tree searches.
5. **Independent Verification & CAS Consensus:**
   - Candidate solutions must pass an independent verifier.
   - Workers compete via atomic database CAS for `SUCCEEDED` status, broadcasting cancellation across Redis Pub/Sub to halt sibling DFS loops.
   - **Credit Conservation Barrier**: `UNSAT` is confirmed if and only if all disjoint subtrees finish with 100% recovered credit weight.

---

### 4.3 Data Model

#### 1. `puzzle_jobs`
```sql
CREATE TABLE puzzle_jobs (
    job_id               VARCHAR(64) PRIMARY KEY,
    tenant_id            VARCHAR(64) NOT NULL,
    puzzle_hash          VARCHAR(64) NOT NULL,
    grid_width           INT NOT NULL,
    grid_height          INT NOT NULL,
    raw_puzzle_json      JSONB NOT NULL,
    dictionary_version   VARCHAR(32) NOT NULL,
    solver_version       VARCHAR(32) NOT NULL,
    execution_mode       VARCHAR(16) NOT NULL,         -- 'LOCAL_CSP', 'DISTRIBUTED_DFS'
    status               VARCHAR(16) NOT NULL,         -- 'ACCEPTED', 'RUNNING', 'SPLIT_RUNNING', 'SUCCEEDED', 'UNSAT', 'TIMED_OUT', 'CANCELLED', 'FAILED'
    solution_json        JSONB,
    cpu_ms_spent         INT NOT NULL DEFAULT 0,
    total_credit_weight  BIGINT NOT NULL DEFAULT 4294967296, -- 2^32 credit for UNSAT barrier
    recovered_credit     BIGINT NOT NULL DEFAULT 0,
    hard_timeout_at      TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    finished_at          TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_jobs_tenant_status ON puzzle_jobs(tenant_id, status);
CREATE INDEX idx_jobs_cache_lookup ON puzzle_jobs(puzzle_hash, dictionary_version, status);
```

#### 2. `search_subtrees`
```sql
CREATE TABLE search_subtrees (
    subtree_id           VARCHAR(64) PRIMARY KEY,
    job_id               VARCHAR(64) NOT NULL REFERENCES puzzle_jobs(job_id) ON DELETE CASCADE,
    branch_index         INT NOT NULL,
    tree_depth           INT NOT NULL,
    prefix_assignment    JSONB NOT NULL,               -- e.g. [{"slot_id": 1, "word": "APPLE"}]
    credit_weight        BIGINT NOT NULL,              -- Credit share for UNSAT validation
    status               VARCHAR(16) NOT NULL,         -- 'PENDING', 'CLAIMED', 'EXHAUSTED', 'SOLVED', 'CANCELLED'
    worker_id            VARCHAR(64),
    lease_until          TIMESTAMP WITH TIME ZONE,
    attempt_count        INT NOT NULL DEFAULT 0,
    updated_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_subtrees_job_status ON search_subtrees(job_id, status);
CREATE INDEX idx_subtrees_lease ON search_subtrees(status, lease_until) 
WHERE status = 'CLAIMED';
```

#### 3. `dictionary_snapshots`
```sql
CREATE TABLE dictionary_snapshots (
    version_id           VARCHAR(32) PRIMARY KEY,
    word_count           INT NOT NULL,
    storage_uri          TEXT NOT NULL,
    checksum_sha256      VARCHAR(64) NOT NULL,
    created_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

---

### 4.4 State Machine & Termination Invariants

```text
       [ACCEPTED]
           │
           ├── (Static Classifier) ──▶ [SPLIT_RUNNING] (Direct Distributed DFS)
           ▼                                   │
       [RUNNING] (Local CSP)                   │
      /    │    \                              │
     /     │     └── (5s Watchdog) ────────────┘
    ▼      ▼      ▼                            │
[SUCCEED][TIMED] [CANCEL]                      │
                                               ▼
                                         [SPLIT_RUNNING]
                                        /    │    \                                             ▼     ▼     ▼      ▼
                                   [SUCCEED][TIMED][CANCEL][UNSAT]
                                   (First)  (Budget)(Client)(Credit=2^32)
```

**State Invariants:**
1. **`TIMED_OUT != UNSAT`**: Budget expiration prior to complete space coverage must **always emit `TIMED_OUT`**.
2. **Credit Conservation UNSAT Barrier**: `UNSAT` is committed if and only if recovered credit matches total initial credit:
   ```sql
   UPDATE puzzle_jobs 
   SET status = 'UNSAT', finished_at = NOW() 
   WHERE job_id = :job_id 
     AND status = 'SPLIT_RUNNING'
     AND recovered_credit = total_credit_weight;
   ```
3. **Winner Claim & Cancel Broadcast**:
   ```sql
   UPDATE puzzle_jobs 
   SET status = 'SUCCEEDED', solution_json = :solution, finished_at = NOW()
   WHERE job_id = :job_id AND status IN ('RUNNING', 'SPLIT_RUNNING');
   ```
   If 1 row is affected, publish `JOB_CANCEL` via Redis Pub/Sub to stop sibling DFS workers.

---

## 5. Core Deep Dives

### 5.1 Dual-Path Ingestion: Static Gate vs Dynamic Watchdog

#### Why not run Distributed DFS on all puzzles? Is it only for "Hard" puzzles?
A common misconception is: *"Since we built a distributed DFS cluster, why not partition every puzzle into 10 subtrees and solve them concurrently?"*

**The answer is strictly no due to the Distributed Coordination Tax:**
1. **High Single-Node Efficiency:** Over 90% of standard crossword puzzles ($15 	imes 15$ NYT style) complete in **10ms ~ 300ms** on a single core using local bitmap indexing and AC-3/MRV pruning.
2. **Coordination Overhead Exceeds Solve Time:** Network serialization ($5	ext{ms}$), message queue dispatch ($10	ext{ms}$), lease acquisition ($10	ext{ms}$), and barrier consensus ($10	ext{ms}$) introduce a $25	ext{ms} \sim 50	ext{ms}$ fixed distributed tax.
3. **Amdahl's Law Inversion:** Distributing a 10ms job increases its end-to-end latency by 3x~5x while consuming multiple worker nodes, collapsing cluster throughput.

#### Dual-Path Architecture Design
Distributed DFS is **not merely a fallback after 5 seconds**. It incorporates both a **Static Gate** and **Dynamic Escalation**:

```text
[Incoming Puzzle Submission]
             │
             ▼ Extract Graph Metrics (M×N, Density ρ, Long Slot Ratio)
[Static Complexity Classifier]
    ├── Pathological? (M×N ≥ 21×21 AND ρ < 0.15 AND Fixed Chars ≤ 2%)
    │         │
    │         └── [YES] ──▶ Direct to Distributed DFS (Zero Wait, Parallel Splitting)
    │
    └── [NO] (Standard Grid Topology)
              │
              ▼ Route to Normal Worker Local CSP
         [Local CSP Engine]
              ├── Solved in 50ms~500ms ──▶ [Direct Return] (>90% Traffic)
              │
              └── Exceeds 5s / 10^6 Backtracks ──▶ [Dynamic Watchdog Escalation]
                                                          │
                                                          ▼ Trigger Adaptive Splitter
                                                  [Distributed DFS Pool]
```

1. **Static Complexity Gate (Zero Wait):**
   - Evaluates **Constraint Graph Density ($ho$)**:
     $$ho = rac{	ext{Cross Intersections}}{N(N - 1) / 2}$$
   - If dimensions reach $21 	imes 21$ or $25 	imes 25$, with sparse intersections ($ho < 0.15$) and pre-filled letter ratio $< 2\%$, early pruning is mathematically ineffective.
   - The job **immediately bypasses single-worker execution** and routes directly to the Distributed DFS engine.
2. **Dynamic Watchdog Escalation:**
   - Standard-looking grids with adversarial edge patterns run with a 5-second wall-clock and $10^6$ backtrack budget.
   - Upon threshold breach, the worker checkpoints its search frontier and offloads disjoint subtrees to the distributed cluster.

---

### 5.2 Distributed Depth-First Search (Distributed DFS) Engine

DFS backtracking inherently relies on call-stack pointers. Distributing this across nodes requires a shared-nothing systems architecture:

#### (1) Search Tree Decoupling: Root BFS & Compact State Replay
- **Phase 1: Root BFS Frontier Generation**
  - The splitter selects the top 2~3 highest-degree or MRV slots near the root and executes a shallow breadth-first search.
  - Pauses once $K$ ($8 \sim 32$) disjoint frontier nodes are generated:
    $$T_{	ext{root}} = T_1 \cup T_2 \cup \dots \cup T_K, \quad orall i 
e j: T_i \cap T_j = \emptyset$$
- **Phase 2: Compact Prefix Serialization**
  - Sends only the partial variable assignment path (e.g., `[{"slot":0, "w":"TIGER"}, {"slot":3, "w":"EAGLE"}]`), weighing $< 256	ext{ bytes}$.
  - The receiving worker replays AC-3 forward checking against its local `mmap` bitmap dictionary in $< 0.2	ext{ ms}$, reconstructing the full constraint domain locally before launching native DFS.

```text
               [Root: Empty Grid]
               /       |                 (Slot 0="CAT")|  (Slot 0="DOG")  ... Shallow BFS generates K frontiers
             /         |                [Subtree 1] [Subtree 2]  [Subtree K]
          │            │            │
          ▼            ▼            ▼ (Wire payload: prefix < 256 bytes)
     [Worker 1]   [Worker 2]   [Worker K]
     Local AC-3   Local AC-3   Local AC-3
     Deep DFS     Deep DFS     Deep DFS
```

#### (2) Irregular Tree Search & Coarse Work Stealing
Crossword search trees are highly **irregular**: branch 1 may hit a conflict at depth 4 and terminate in 5ms, while branch 2 harbors millions of combinations taking minutes. Static allocation produces severe stragglers.

**Work Stealing Protocol:**
- Idle Worker A sends a `STEAL` request to busy Worker B.
- **Critical Invariant: Steal from the Shallowest Unexplored Choice Point!**
  - Subtree size at depth $k$ scales as $O(b^{D-k})$.
  - **Never steal from the top of the stack (deepest point)**: work is $O(1)$, causing network thrashing.
  - **Steal from the bottom of the stack (shallowest point)**: yields coarse-grained subtrees with massive remaining volume, amortizing network round-trips over minutes of compute.

#### (3) Distributed Termination & Credit Conservation Invariant
In SAT problems, first-hit terminates the job. However, strictly proving **UNSAT** requires verifying that 100% of branches were exhausted without loss:

**Credit Conservation Protocol:**
- Initial root task is issued credit $W_{	ext{total}} = 2^{32}$.
- When splitting into $K$ branches, weight divides evenly: $W_{	ext{child}} = W_{	ext{parent}} / K$.
- When a worker exhausts a branch without finding a solution, it returns $W_{	ext{child}}$ to the coordinator for atomic accumulation:
  $$	ext{recovered\_credit} \leftarrow 	ext{recovered\_credit} + W_{	ext{child}}$$
- **Termination Verdict**:
  - **UNSAT Proof**: Committed if and only if $	ext{recovered\_credit} == W_{	ext{total}}$ with zero solutions found.
  - **Zero False UNSAT**: Dropped workers lose their unreturned credit until lease recovery reassigns them. If retries fail and the budget expires, $	ext{recovered\_credit} < W_{	ext{total}}$, enforcing a `TIMED_OUT` verdict rather than false `UNSAT`.

#### (4) Speculative Solution Broadcast & Nogood Sharing
1. **Solution Cancellation Broadcast**:
   - Winning worker passes the independent verifier and commits via CAS.
   - Emits `JOB_CANCEL` across Redis Pub/Sub. Active workers check an atomic boolean every 1,000 backtracks and immediately abort DFS loops.
2. **Cross-Worker Nogood Learning**:
   - When a worker proves that a short combination ($	ext{Slot}_2=	ext{"CAT"} \land 	ext{Slot}_5=	ext{"DOG"}$) produces an empty domain anywhere, this minimal conflict is published as a **Nogood clause** to an in-memory cluster cache.
   - Sibling workers filter candidate branches against active Nogoods, pruning vast subtrees collaboratively across physical machines.

---

### 5.3 Bit-Parallel Local CSP Engine

Within individual workers, algorithmic constants dictate cluster throughput:

```text
Approach A: Naive DFS + Linear Dictionary Scans
[Slot 1: Length 5] ──Scan 10,000 words──▶ Fill "APPLE"
    └── [Slot 2: Cross at pos 3] ──Scan 8,000 words filtering 'P'──▶ ... (Backtrack on conflict)

Approach B: Bitmap Index + MRV + Arc Consistency (Bit-Parallel CSP, Recommended)
[Slot 1] ──Pick slot with Minimum Remaining Values (MRV = 4) ──▶ Bitwise AND in 1 CPU cycle
    └── Forward Checking (AC-3) ──▶ Instant domain pruning to zero at depth 2
```

| Dimension | Approach A: Naive DFS + Linear Scan | Approach B: Bitmap Index + MRV + AC-3 (Recommended) |
| :--- | :--- | :--- |
| **Search Complexity** | Combinatorial branch explosion; degrades to $O(D^N)$ on complex grids. | MRV dynamic variable ordering and degree heuristics reduce search nodes by >99%. |
| **Matching Efficiency**| Iterates strings or regular expressions; high memory bandwidth and cache misses. | **Bit-Parallelism**: 64-bit integer vectors filter 64 candidate words per CPU instruction. |
| **Memory Footprint** | Minimal (a few MB). | Structured resident index and trail stack require 50MB~200MB per worker. |
| **Tail Latency** | **Severe**. P99 routinely stretches into minutes. | **Sub-second**. Standard 15x15 puzzles finish in 50ms~500ms. |
| **Recommendation** | Toy implementations only. | **Foundational standard for production solving engines**. |

---

### 5.4 Task Completion Semantics & Fault Tolerance

```text
Lease Claim & CAS Winner Flow:
[Worker A (Subtree 1)] ──Claim Lease (lease_until = NOW()+30s)──▶ [DB]
[Worker B (Subtree 2)] ──Claim Lease (lease_until = NOW()+30s)──▶ [DB]
       │
Worker A discovers valid candidate:
  1. [Worker A] ──▶ [Independent Verifier] (verify crossings + all-diff) ──▶ PASS
  2. [Worker A] ──▶ CAS (UPDATE jobs SET status='SUCCEEDED' WHERE status='RUNNING')
  3. CAS Succeeds! ──▶ Redis PubSub broadcast Cancel ──▶ Worker B interrupts DFS loop
```

| Dimension | Approach A: Distributed Exactly-Once Transactions | Approach B: At-Least-Once + Leases + CAS Winner (Recommended) |
| :--- | :--- | :--- |
| **Crash Recovery** | Relies on 2-Phase Commit (2PC) or global locks; crashes cause deadlocks. | **Self-healing**: Expired leases return uncompleted subtrees to the queue automatically. |
| **Consistency** | Fragile coordination layer prone to cascading failures. | **End-to-end idempotency**: `branch_id` deduplication and atomic CAS ensure exactly one winner. |
| **Speculative Waste** | Inflexible cancellation handling. | Global broadcast terminates sibling workers immediately upon winning CAS. |
| **Throughput Cost** | Massive synchronous coordination tax. | Lock-free high-throughput design with minimal coordination overhead. |
| **Recommendation** | Not viable for distributed tree search. | **Industry standard for distributed search engines**. |

---

### 5.5 Multi-Tenant Isolation & Resource Governance

Puzzles exhibit computational variation spanning $10	ext{ ms}$ to $10	ext{ min}$.

```text
Adaptive Coarse Splitting:
[Normal Worker] Reaches 5s timeout or backtrack threshold
       │
       ▼ Select shallow high-branching slot (e.g., Slot 3 has 50 valid words)
[Adaptive Splitter]
       ├── Partition into 5 coarse subtrees (10 prefix words each)
       └── Enqueue into [Hard / Branch Queue]
               ├── [Hard Worker 1] handles Branch 0 (words 0..9)
               ├── [Hard Worker 2] handles Branch 1 (words 10..19)
               └── ...
```

| Dimension | Approach A: Single Global FIFO Pool | Approach B: DRR Fair Queues + Dual-Track + Coarse Splitting (Recommended) |
| :--- | :--- | :--- |
| **Tenant Isolation** | **None**. Pathological puzzles monopolize all workers, stalling the platform. | **Strict Isolation**: Deficit Round Robin (DRR) allocates sliding CPU credits per tenant. |
| **Queuing Efficiency**| Heavy jobs delay lightweight queries across the entire fleet. | **Dual-track routing**: Normal Queue guarantees rapid exit for sub-second tasks. |
| **Splitting Granularity**| Pushing fine-grained DFS nodes floods message brokers with millions of RPCs. | **Coarse-grained cut**: Splits only at depth 1~2, keeping network traffic minimal. |
| **Autoscaling Metrics**| Scaling on raw queue depth produces inaccurate signals. | **Autoscaling based on Estimated CPU Backlog and worker utilization**. |
| **Recommendation** | Single-tenant offline batch systems only. | **Standard for multi-tenant cloud-native SaaS platforms**. |

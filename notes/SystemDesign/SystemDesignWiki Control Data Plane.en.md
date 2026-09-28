# Wiki · Control Plane vs Data Plane Separation

Wiki Navigation: [[SystemDesign00 Overview|00 Blueprint & Numbers]] → [[SystemDesignWiki Control Data Plane|Wiki · Control Plane vs Data Plane]]

**Control Plane vs Data Plane Separation** is a foundational architectural pattern across modern distributed systems, cloud-native infrastructures (such as Kubernetes, Service Meshes), and Software-Defined Networking (SDN). It provides high availability, fault isolation, and horizontal scalability.

---

## 1 · Definitions & Structural Boundary

The Control Plane and Data Plane occupy distinct stages along the system processing path:

```text
       [ Operator / API / GitOps ]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ Control Plane                                          │
│ Responsibility: Policy calculation, consensus, state   │
│                 reconciliation, and topology scheduling │
│ Characteristics: High consistency (Raft/etcd), complex │
│                  logic, low-to-medium QPS              │
└──────────────────────────┬─────────────────────────────┘
                           │ (Push updates / xDS / local rules)
                           ▼
┌────────────────────────────────────────────────────────┐
│ Data Plane                                             │
│ Responsibility: Packet forwarding, protocol proxying,  │
│                 I/O streaming, high-frequency compute  │
│ Characteristics: Extreme throughput (millions of QPS), │
│                  microsecond latency, lock-free/zero-cp│
└────────────────────────────────────────────────────────┘
        ▲                                ▲
        │ (Client Traffic)               │ (Client Traffic)
  [ Inbound I/O ]                 [ Outbound I/O ]
```

### 1.1 · The Control Plane: The System Brain
- **Primary Function**: Defines the desired system state (`Desired State`). It makes routing decisions, governs authentication/authorization policies, orchestrates service discovery topologies, schedules resources, and resolves distributed consensus.
- **Operating Model**: Computationally complex and heavily reliant on strong-consistency storage engines (Raft, etcd, Paxos). It operates via asynchronous reconciliation loops (`reconcile()`).

### 1.2 · The Data Plane: The System Muscles
- **Primary Function**: Executes the policies received from the control plane against incoming client traffic, data streams, and storage I/O.
- **Operating Model**: Highly optimized, flat execution paths. It never contacts centralized databases synchronously. It uses local in-memory routing tables, zero-copy kernel transfers, eBPF, DPDK, and lock-free concurrency to guarantee deterministic microsecond latency.

---

## 2 · Core Architectural Philosophy: The Survival Invariant

The primary engineering goal of separation is **blast radius containment and uninterrupted autonomous forwarding**:

> [!IMPORTANT]
> **The First Survival Invariant of Distributed Systems**:
> **Even if the entire control plane crashes or suffers a network partition, the data plane must remain 100% operational, forwarding traffic continuously based on the Last-Known Good State cached in local memory!**

1. **Zero Synchronous Control Plane RPCs in the Critical Path**:
   - The data plane must NEVER issue a blocking synchronous network call to the control plane to process an inbound user request.
   - All routing rules, cryptographic certificates, and rate-limiting quotas must be proactively pushed into local memory (e.g. Envoy xDS protocol, Linux kernel iptables/eBPF maps).
2. **Graceful Degradation**:
   - When the control plane is offline, the system temporarily loses *mutation capability* (cannot deploy new pods, alter routing policies, or trigger auto-scaling), but its *serving capability* continues without disruption.

---

## 3 · Industrial Implementation Mappings

### 3.1 · Kubernetes Architecture Mapping
Kubernetes is the standard reference implementation of this pattern:

| Plane | Components | Core Responsibilities & Characteristics |
|---|---|---|
| **K8s Control Plane** | `kube-apiserver`<br>`etcd`<br>`kube-scheduler`<br>`kube-controller-manager` | · Maintains declarative resource specifications (CRDs, Deployments, Services).<br>· etcd runs Raft consensus for single-source-of-truth consistency.<br>· Bin-packing scheduler scores and places workloads.<br>· Throughput benchmark: $100 - 1,000\text{ QPS}$. |
| **K8s Data Plane** | `kubelet`<br>`kube-proxy` / Cilium eBPF<br>Container Runtime (containerd)<br>Application Pods | · `kube-proxy` / Cilium maps Services to Linux kernel routing tables.<br>· Application pods process incoming user traffic ($10^5 - 10^7\text{ QPS}$).<br>· If Master nodes crash completely, worker node pods continue processing existing client requests without packet loss. |

### 3.2 · Service Mesh (Istio + Envoy)
- **Control Plane (`istiod`)**:
  - Watches the K8s API server for configuration changes, compiles VirtualServices into standardized **xDS configurations** (LDS, RDS, CDS, EDS), and pushes them via gRPC streams to proxies.
- **Data Plane (Envoy Sidecars)**:
  - High-performance C++ proxies utilizing non-blocking event loops (`epoll`).
  - Intercepts all pod ingress/egress traffic (mTLS encryption, retries, circuit breaking, telemetry). Processes tens of thousands of requests per second per core with sub-millisecond overhead.

### 3.3 · Distributed Storage & Streaming
- **Kafka**:
  - **Control Plane**: KRaft Controller Quorum (Raft consensus governing partition leadership, topic definitions, broker memberships).
  - **Data Plane**: Broker storage engines (`sendfile` zero-copy, Linux Page Cache append-only commit logs), processing millions of message bytes per second.
- **TiDB / CockroachDB**:
  - **Control Plane**: Placement Driver (PD) orchestrates region splits, rebalancing, and TSO distributed timestamps.
  - **Data Plane**: TiKV instances execute multi-raft key-value storage, SSTable compaction, and MVCC transaction concurrency.

### 3.4 · CI/CD Pipelines & Distributed Orchestration (Workflow vs Job Decoupling)
In task scheduling and pipeline engines (e.g., GitHub Actions, GitLab CI, Argo Workflows), **if an entire pipeline is modeled as a monolithic entity (only a Workflow without Job entities) or flattened into a script runner, the system quickly collapses under scheduling, scalability, and fault-recovery pressures**.

Separating **Workflow (Graph / Orchestration Control)** from **Job (Node / Execution Data)** is a direct embodiment of Control Plane vs Data Plane decoupling:

```text
[ Workflow Orchestrator (Control Plane) ]
  │
  ├─ DAG Resolution: Job A (Build) ──┬──> Job B (Unit Test) ──┬──> Job D (Push Image)
  │                                  └──> Job C (Package)   ──┘
  ├─ In-degree Listener: In-degree == 0 triggers dispatch
  └─ Scheduling Router (Distribute to Heterogeneous Queues)
         │                         │                         │
         ▼                         ▼                         ▼
   [ Queue: Linux ]          [ Queue: macOS ]          [ Queue: GPU ]
         │                         │                         │
[ Worker: 1-Core Linux ]   [ Worker: Xcode Host ]   [ Worker: A100 Cluster ]
 (Data Plane: Lint scan)   (Data Plane: iOS Build)   (Data Plane: Model Eval)
```

1. **DAG Topology Orchestration & State Modeling (Graph vs Nodes)**:
   - **Entity Abstraction**: A Workflow is a Directed Acyclic Graph (DAG); a Job is a node within that graph.
   - **Dependency Isolation**: Edges are explicitly represented in a relational schema (e.g. `job_dependency` storing `from_job_id` and `to_job_id`). Modeling Job as a discrete entity enables the orchestrator to model topological fan-out and fan-in (e.g., "Trigger Test B and Package C concurrently once Build A succeeds") via zero-in-degree queues.
2. **Heterogeneous Runners & Scheduling Isolation**:
   - In production, steps within a single commit pipeline require vastly different compute environments:
     - **Static Analysis / Lint**: Requires only a cheap, lightweight 1-core Linux container.
     - **iOS / macOS Build**: Must be routed to dedicated bare-metal macOS hardware with Xcode installed.
     - **Model Validation**: Requires instances with dedicated NVIDIA GPUs (e.g., A100).
   - Without the Job abstraction, systems cannot partition heterogeneous compute attributes across physical fleets. Treating Job as the atomic scheduling unit allows the runner gateway to bind resource requests and affinities (`nodeSelector`, labels) to matched compute nodes.
3. **Failure Blast Radius & Partial Retries**:
   - If an entire pipeline is a monolithic job, a transient network timeout during the final artifact push forces the user to rerun the preceding 40-minute build from scratch.
   - With discrete Job entities, each job maintains independent state machines (`PENDING`, `RUNNING`, `SUCCESS`, `FAILED`) and S3 artifact references. Retries execute only for the failed node, reusing cached outputs from upstream jobs.

---

## 4 · Quantitative Comparison Baselines

```text
┌────────────────────────────────────────────────────────────────────────┐
│              Control Plane vs Data Plane Quantitative Matrix           │
├───────────────────────┬────────────────────────┬───────────────────────┤
│ Metric Dimension      │ Control Plane          │ Data Plane            │
├───────────────────────┼────────────────────────┼───────────────────────┤
│ Typical QPS Baseline  │ 10 - 2,000 QPS         │ 100,000 - 10,000,000+ │
│ Processing Latency    │ 10 ms - seconds        │ < 100 μs - 5 ms       │
│ Persistence Layer     │ Strong-consistency WAL │ Pure memory / DMA / IO│
│ Core Data Structures  │ AST, DAG, Raft Log     │ Hash Tables, Radix, RBs│
│ Concurrency Model     │ Locking / Consensus    │ Lock-free / eBPF / C++│
│ Blast Radius          │ Halts admin changes    │ Direct business outage│
└───────────────────────┴────────────────────────┴───────────────────────┘
```

---

## 5 · System Design Interview Application Guidelines

### When to explicitly propose Plane Separation:
1. **API Gateway / Edge Proxy Designs**:
   - Decouple administrative rule management (Control Plane) from traffic forwarding engines (OpenResty/Envoy Data Plane). The gateway worker must never query the MySQL configuration database on each incoming HTTP request.
2. **Distributed Cache Routing / Sharding Proxies**:
   - Consistent hashing ring topology updates (Control Plane) are propagated to client SDK local memory (Data Plane), allowing clients to route queries directly to the target shard without an intermediate hop.
3. **Distributed Scheduler & Compute Platforms (Case 08 / Case 12)**:
   - Central schedulers assign workloads asynchronously; worker pools execute local compute pipelines autonomously.

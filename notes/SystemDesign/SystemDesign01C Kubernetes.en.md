# System Design 01C · Kubernetes and the LLM Training Control Plane

Course location: [[SystemDesign01B Virtualization Containers|01B Virtualization and Containers]] → this note → [[SystemDesign01D Redis|01D Redis]]

The previous note introduces containers as a mechanism for process isolation and packaging.
This note explains how those isolated processes are assigned to nodes.

Kubernetes operates as a distributed declarative control plane.
It maintains the desired state in the central apiserver and backend etcd storage.
Distributed controllers and the central scheduler drive the cluster toward that state.

LLM training requires an additional layer of worker group semantics.
Native Kubernetes does not provide this layer by default:
- Heterogeneous GPU device allocation.
- Explicit NIC topology mapping.
- Multi-worker simultaneous start (gang scheduling).
- Protection against partial starts.
- Multi-team queues, resource quota management, and dynamic preemption.
- Whole-group halt and recovery from external durable checkpoints after an isolated failure.
- Network topology awareness and underlying communication routing.

The companion lab implements a lightweight local training control plane.
The source code is located in `project/LLMTrainLab/` or [GitHub: CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab).

```k8s-hierarchy-visual
```

---

## 1 · Cluster Components and Object Lifecycle

A cluster consists of a centralized control plane and worker nodes that carry the computational load.

| Component | Role | Failure Mode |
|---|---|---|
| kube-apiserver | External HTTP API; single entry point for state reads/writes. | Control plane halts. New Jobs fail to submit. Existing Pods keep running if they do not require external API interactions. |
| etcd | Source of truth for cluster state. | Apiserver becomes read-only or unresponsive. Control plane operations freeze. |
| kube-scheduler | Binds pending Pods to compute nodes. | New Pods remain Pending. Existing running Pods are unaffected. |
| kube-controller-manager | Executes continuous reconcile loops for objects (e.g., ReplicaSet, Job). | Crashed Pods are not automatically rebuilt by their parent managing resources. |
| kubelet | Node agent: runs containers via CRI and reports status heartbeats. | Node heartbeats drop. Pod status on the affected node becomes Unknown. |
| container runtime | containerd or dockerd; creates the Linux containers. | Cannot start new processes on the node. Currently running containers might survive. |
| kube-proxy / CNI | Handles Service forwarding and cross-node Pod networking. | ClusterIP load balancing fails. Cross-node traffic routing breaks down. |
| Device Plugin | Registers GPUs and extended resources with the local kubelet. | Scheduler cannot discover GPU capacity. GPU mounts fail upon Pod startup. |

The control plane stores desired schemas and observed states.
The data plane carries application processes, NIC packet routing, GPU computation, and disk I/O.
Model token generation and all-reduce sync throughput do not route through the apiserver.
Transient control plane outages do not interrupt running computational processes on the data plane.

A 4-worker deep learning training Job timeline:
1. User submits a custom resource defined as LLMJob or PyTorchJob to the cluster.
2. Kueue intercepts and performs admission.
3. It validates team quota and priority.
4. It places the task in a waiting queue.
5. Gang scheduling triggers.
6. It admits the task when 4 GPU slots become available simultaneously.
7. Downstream controllers create 4 individual Pods.
8. The scheduler binds them to physical nodes.
9. The kubelet on the target nodes begins pulling container images.
10. It mounts required persistent PVC volumes.
11. The kubelet executes the Allocate() call via plugins.
12. This obtains specific physical GPU device nodes.
13. Container processes start.
14. The environment injects RANK, WORLD_SIZE, and MASTER_ADDR variables.
15. Processes execute the NCCL init barrier.
16. This performs network connectivity testing across nodes.
17. After confirming all workers are ready.
18. Step execution begins.
19. During training, workers write periodic checkpoints.
20. These are saved to a shared RWX storage volume backend.
21. The task eventually exits as Succeeded.
22. Alternatively, if hardware failure occurs.
23. The group halts and recovers from the latest valid ckpt.

Pod phase progression:
```text
Pending
  Unscheduled, or dependencies like container images/GPUs are not ready.
Running
  The primary container process is actively executing code.
Succeeded / Failed
  The primary process terminated with a 0 or non-0 exit code.
```

CrashLoopBackOff indicates that kubelet is restarting a failing container.
For stateless web services, single-Pod restarts are effective.
In tightly coupled training tasks, single-Pod restarts are ineffective.
Especially after the internal NCCL communicator crashes.
The internal logical network topology is broken.
The global Job controller must intervene.
The controller destroys all current Pods, releases resources, and rebuilds the worker group.

```k8s-lifecycle-visual
```

---

## 2 · Object Hierarchy

In diagrams, objects look nested as a tree structure.
In actual implementation, association relies on dynamic label filtering and ownerReferences.
If a parent object is deleted via API without explicit cascading flags, child objects may become orphans and continue consuming compute resources.

### Container
A container is the atomic execution unit, binding a specific image, an execution command, and hardware resource limits.
Multiple containers inside the same Pod share the network namespace and attached volume space.
A typical training worker contains three parts:
- A main container executing the `torchrun` compute process.
- A sidecar container collecting exporter metrics or providing network debugging tools.
- An init container for downloading datasets or pre-loading configurations before the main process boots.

### Pod
The Pod is the smallest schedulable and deployable unit in Kubernetes.
Same-Pod containers share the physical node or VM.
Pod IPs are dynamically allocated by network plugins.
They change upon Pod deletion.
Do not rely on a Pod IP as a durable identity that needs to cross restart lifecycles.

### ReplicaSet
A ReplicaSet monitors the global cluster state.
It ensures that the exact count of running Pods equals the declared `spec.replicas` integer.
It deletes excess Pods and creates missing ones.

### Deployment
A Deployment adds state management logic on top of ReplicaSets.
It provides zero-downtime rolling updates and version rollbacks.
Deployments are the standard choice for stateless request-handling microservices.
They are not suited for multi-node, tightly coupled parallel training workloads.
Deployments restart failed Pods individually, which breaks NCCL sync groups.
Their scaling operations lack gang synchronization semantics, leading to partial starts.
Training ranks require unique independent identities (e.g., Rank 0 as master), whereas Deployment Pods are designed to be interchangeable clones.

### Service
A Service provides a virtual entry IP (ClusterIP) and an internal DNS record abstracting the Pods below it.
Standard ClusterIP provides Layer 4 network load balancing, ideal for scattering traffic to redundant Web APIs.
Headless Services configure `clusterIP: None` in their manifest, bypassing centralized load balancing.
It resolves DNS queries directly to the backing Pod IPs.
Rank 0 rendezvous logic during distributed startup uses direct DNS resolution.
NCCL gradient traffic must not hit a Service load balancer mechanism.

### Namespace
A Namespace divides the global scope of resource names to prevent naming collisions.
It provides administrative boundaries for resource quotas, RBAC access permissions, and NetworkPolicy traffic isolation.
Namespaces offer zero physical process isolation at the underlying OS level.
They are API boundaries, not hardware sandboxes.

---

## 3 · Scheduling

kube-scheduler processes newly created Pods currently in the Pending state.

```text
watch unscheduled Pods sitting in the queue
  -> Filter phase (evaluate constraints to filter out ineligible nodes)
  -> Score phase (apply algorithms to rank eligible nodes based on optimization preferences)
  -> Bind phase (atomically write the highest-scoring physical node name back into the Pod object)
```

The default Kubernetes scheduler processes Pods individually.
This creates a risk of partial allocation deadlocks.
For instance, a task needing 4 GPUs might get 3 Pods Running, while the 1 remaining Pod gets stuck Pending due to fragmentation.
This causes 3 GPUs to hang indefinitely, computing zero steps while starving other waiting tasks.

### Requests, Limits, QoS
In Kubernetes manifests, request provides a theoretical ledger value for the central scheduler.
It tracks remaining allocatable node capacity, not the actual live CPU usage.
Limit represents the physical cap enforced by the local kubelet and the underlying Linux cgroups subsystem.
If a process exceeds its memory limit, it gets OOM killed.

Kubernetes classifies Pods into three Quality of Service (QoS) tiers:
- Guaranteed: Every container's CPU and memory request equals its limit.
- Burstable: Requests exist and are smaller than their defined limits.
- BestEffort: No requests or limits are defined. These Pods are evicted first during memory pressure.

Hardware devices like GPUs are categorized as Extended Resources.
Requests for any Extended Resources must equal their limits.
Consequently, training workers occupying GPUs operate in the Guaranteed tier by default.

### Affinity and Topology Spread
The scheduler controls where workloads land:
- nodeSelector provides exact matching of node labels.
- nodeAffinity and podAffinity allow soft preferences or hard placement rules.
- taint and toleration establish a repulsion relationship. Administrators use taints to reject default scheduling on premium nodes. Only Pods defining matching tolerations can enter.
- topologySpreadConstraints require replicas to scatter evenly across different physical zones or racks.

GPU nodes are often tainted in production to block CPU microservices.
Training topology requirements demand that workers pack densely within the same rack or NVLink switch domain to maximize interconnect bandwidth.
Inference services utilize topologySpreadConstraints to scatter instances across multiple AZs.

### Device Plugin
The kubelet lacks built-in logic regarding GPU models or VRAM management.
It registers and allocates GPUs via external Device Plugins.

```text
plugin executes ListAndWatch locally
  -> reporting physical capacity to the kubelet
kubelet aggregates and reports nvidia.com/gpu capacity
  -> up to the central API server
Pod explicitly declares its nvidia.com/gpu requirements in its YAML spec
  -> scheduler filters out nodes lacking capacity during the Filter phase
kubelet receives the final scheduling decision
  -> calls Allocate() on the target plugin
runtime securely mounts physical device nodes (e.g., /dev/nvidia0)
  -> directly into the isolated container environment
```

The NVIDIA GPU Operator bundles kernel drivers, discovery plugins, and monitoring exporters.
Early Device Plugins exposed GPUs as discrete integer slots without awareness of PCIe or NVLink topology.
The newer DRA (Dynamic Resource Allocation) framework provides richer 3D topology expressions and device slicing mechanics.
Many production clusters still rely on the classic Device Plugin model.
`gpuType` (e.g., requesting H100 over A100) acts as a hard Filter.
A matrix multiplication task compiled for H100 architecture cannot degrade onto A100 nodes without triggering runtime crashes.

### Gang and Preemption
Gang scheduling mandates that all parallel workers receive resource allocations simultaneously.
Volcano provides gang semantics directly at the scheduling layer via its PodGroup concept.
Kueue offers synchronized admission guarantees at a higher webhook layer.
The global task controller requires fallback mechanisms.
If partial allocation occurs, the controller identifies and deletes incomplete Pod sets to release GPU hardware.

PriorityClass defines the relative priority of tasks.
Native Kubernetes preemption targets individual Pods, evicting lower-priority ones one by one.
Training tasks require Job-level preemption to evict entire gangs simultaneously and clear contiguous blocks of compute resources.

### Queues
Namespace resource quotas act as static ceilings for total cluster consumption.
They do not constitute a fair-scheduling queue system.
Kueue introduces ClusterQueue, LocalQueue, and ResourceFlavor routing objects.
It supports FIFO ordering, priority-based queues, cross-team fair share balancing, and cross-queue resource preemption.
High-priority online inference scale-ups can preempt low-priority offline training batches to seize GPUs.

### Topology Labels
Common topology labels include `topology.kubernetes.io/zone`, `rack`, and organizational flags like `network=rdma`.
Hard constraints dictate requirements like specific GPU architectures or reliance on RDMA networking.
Soft constraints express preferences, such as placing all Pods of a Job within the same physical rack.
If a scheduler violates a cross-rack soft constraint due to resource scarcity, the Pod will still Bind and enter the Running state.
However, this induces slow cross-rack all-reduce communications, bottlenecking the iteration step time.

```k8s-gang-visual
```

---

## 4 · Application Management

### Job
The native Kubernetes Job controller ensures that a specified number of Pods (N) run to completion (exit 0).
Its `parallelism` field governs maximum allowed concurrency.
It does not confer synchronized gang startup semantics.
Indexed Jobs assign deterministic, zero-indexed identifiers to the launched Pods.
Upon failure, the native Job controller retries only the isolated Pod that crashed.

### LLMJob / PyTorchJob
Modern training tasks utilize Custom Resource Definitions (CRDs) for job submission.
The internal failure policy is configured to RestartAll.
The controller waits until all required Pods achieve the Running status before injecting the MASTER_ADDR.
Processes wait until all workers assemble and pass the initial network barrier.
These CRDs inject RANK and WORLD_SIZE environment variables into the containers.

### StatefulSet and DaemonSet
StatefulSets offer stable network identities (e.g., pod-0, pod-1) and persistent PVC volume bindings across restarts.
This design suits stateful consensus clusters like ZooKeeper or Kafka.
For LLM training, practice destroys and rebuilds entire worker groups upon node failure.
These tasks rarely demand sticky identities or local state disks that must survive restarts.

DaemonSets guarantee that exactly one Pod runs on every matching node.
This is used for installing infrastructure agents.
Examples include Fluent Bit for log collection, DCGM exporters for monitoring GPU temperature and power, and Device Plugins for hardware discovery.

### Operator
The Operator pattern encodes operational knowledge into executable logic.
It listens for changes to custom CRs in the API server.
It parses configurations, creates compute Pods, and provisions networking Services.
It takes over lifecycle management and failure recovery sequences.
It aggregates the execution progress of Pods and updates the top-level status phase field of the user's CR.

### Probes
Kubernetes offers three health-checking probes:

| Probe | Failure Action | Use Case |
|---|---|---|
| startupProbe | No restart, no traffic routing. | Protects slow-starting apps, shielding them from other probes until ready. |
| readinessProbe | Removes Pod from Service Endpoints. | Indicates the process isn't ready to accept external HTTP connections. |
| livenessProbe | Triggers kubelet to SIGKILL the container. | Breaks deadlocks by rescuing unresponsive processes. |

During heavy model compilation or NCCL network initializations, the Python thread may block for minutes.
If a liveness probe is misconfigured, it will false-kill a healthy rank.
Monitoring distributed training heartbeats and detecting deadlocks is the responsibility of the Job controller, not kubelet probes.

---

## 5 · Persistence

Storage backend selection dictates I/O throughput and data visibility:

| Volume | Scope | Practical Scenario |
|---|---|---|
| emptyDir | Tied to the host Pod's lifecycle. Data vanishes upon Pod deletion. | Ephemeral scratch data directories, or shared memory (shm). |
| hostPath | Maps to the node's local physical disk, surviving as long as the physical node does. | Node-level caches for datasets. |
| configMap | Injects small text objects stored in etcd as flat files inside the container. | Distributing hyperparameter configuration files. |
| PVC | A persistent storage claim independent of Pod lifecycles. | Saving checkpoint weights across epochs, or mounting external training datasets. |

When an emptyDir's medium property is set to Memory, the kernel mounts it as a RAM-backed tmpfs virtual disk.
If the tmpfs size limit is configured too low, NCCL will crash with an OOM error when allocating shared memory pages.

### PV and PVC

Container storage mounting operates as a supply chain mechanism:
```text
Pod (End Consumer) -> PVC (Request Claim) -> PV (Provisioned Resource) -> Storage Backend (Lustre/Ceph Physical Array)
```

A PVC (PersistentVolumeClaim) declares a formal request for storage.
A PV (PersistentVolume) is the mapped storage volume, pre-allocated or dynamically provisioned.
The StorageClass defines the backend provisioner and controls automated dynamic provisioning.

Backend concurrency capabilities are defined by access modes:

| Access Mode | Concurrency Property |
|---|---|
| RWO (ReadWriteOnce) | Mountable as read-write by a single specific node. |
| ROX (ReadOnlyMany) | Mountable concurrently by multiple nodes for read-only fetch operations. |
| RWX (ReadWriteMany) | Mountable concurrently by multiple nodes for full read-write operations. |

For LLM training, workers across multiple separate nodes must concurrently write checkpoint fragments to a unified destination.
This mandates RWX storage.
CSI (Container Storage Interface) plugins translate Kubernetes-level requirements into standardized API calls.
They connect to Lustre clusters, Ceph filesystems, or networked disks.

### Checkpoint Recovery Path

Comprehensive failure handling follows a defined pipeline:

```text
Detect a worker hardware or network failure
  → Controller issues a halt command to stop remaining workers in the group
  → Clean up network resources, request new compute capacity, and rebuild the group
  → New workers boot, loading the last verified complete checkpoint
  → Synchronize step counts globally and resume training
```

This sequence demands tracking of key metrics:
- Failure detection time: Latency between a process deadlock and the controller becoming aware of it.
- Rebuild duration: Time spent queueing, waiting for scheduler assignment, and pulling images.
- Lost steps: Computation iterations executed after the last successful save and before the crash occurred.
- Retry counts: Guardrails preventing infinite loops if the network is permanently broken.
- Corrupted files: Monitoring incomplete writes caused by sudden power loss.

To prevent data corruption during a mid-write crash, processes must write data to a temporary file.
They must fully flush it to disk, and then perform an atomic OS-level rename operation.
Systems typically rely on a pointer file (e.g., `latest.txt`) to track the location of the newest verified checkpoint.
RPO (Recovery Point Objective) correlates with the configured interval between checkpoint writes.
RTO (Recovery Time Objective) encompasses failure detection, queueing, scheduling, container startup, and weight loading times.
Kubernetes provisions compute Pods upon request, but it cannot recover computation steps that were not saved to a checkpoint.

---

## 6 · Networking

In the Kubernetes network model, every Pod receives a routable IP address.
CNI (Container Network Interface) plugins execute IP assignment, virtual NIC wiring, and host route configuration.
CoreDNS provides internal domain name resolution.
kube-proxy or eBPF alternatives maintain iptables mappings for ClusterIP load balancing.
NetworkPolicy acts as a software-defined firewall, using label selectors to enforce communication boundaries.

Gradient synchronization and NCCL communication use raw Pod IPs or dedicated RDMA endpoints for point-to-point connections.
If these data streams route through a standard ClusterIP, load balancers will scatter the connections, breaking NCCL communication rings.

Ingress controllers and the Gateway API manage north-south HTTP/HTTPS application traffic crossing the cluster boundary.
They handle Layer 7 logic and are never positioned on the all-reduce data path.

For internal east-west data flows:
- Intra-node communication between GPUs relies on NVLink interconnects.
- Cross-node heavy compute communication depends on RDMA networks (e.g., RoCEv2 or InfiniBand) bypassing the kernel network stack.
- Kubernetes control plane API interactions and log aggregation utilize standard Gigabit Ethernet.

Schedulers employ Filter rules to place RDMA-dependent training tasks on nodes equipped with RDMA NICs.
Allocating high-speed NICs directly to Pods as PCIe devices provides better isolation than enabling `hostNetwork: true`.
Physical network latency crossing availability zones (AZs) or distinct racks will bottleneck the per-iteration step time.

---

## 7 · Observability

Infrastructure observation is layered:

| Observation Level | Target Metrics | Monitoring Objective |
|---|---|---|
| Infrastructure & Cluster | Backlog of Pending Pods, ratio of healthy ready nodes, baseline kubelet/runtime error rates. | Ensure the cluster maintains physical capacity and base health. |
| Scheduling & Control Plane | Queue delays in Kueue, wait times for gang resource assembly, full-group recovery durations. | Evaluate resource fluidity, scheduling efficiency, and fairness. |
| Business & Training Task | Iteration step time, loss convergence curves, time spent in NCCL syncs, checkpoint write latency. | Align infrastructure performance with model output quality and compute utilization. |

Prometheus acts as the engine for collecting global time-series metric data.
NVIDIA's DCGM exporter collects low-level hardware metrics like GPU core temperatures, power draw, and SM utilization.
Within log aggregation pipelines, ingested log lines must be tagged with explicit job, rank, and step fields to debug distributed deadlocks.
The KWOK tool provisions simulated fake nodes, enabling low-cost scale testing of the scheduler.

---

## 8 · Layered Architecture

The object model and core scheduling logic solve a micro-level problem: deciding where to place one specific Pod.
Large-scale systems require macro-level slicing based on responsibility boundaries and topological physics.
These two axes are orthogonal; they must be mapped independently.

```k8s-layered-arch-visual
```

### Responsibility Layers
This architecture slices the system based on what operations a component performs.
The Edge layer manages authentication, rate limiting, TLS offloading, and routing.
The Compute logic layer orchestrates business flows and issues write commands.
The Data persistence layer saves factual records and maintains replica consistency.
The Async processing layer handles background retries and saves checkpoints, remaining out-of-band of synchronous user requests.

Kubernetes architecture adheres to responsibility layering:
- apiserver and etcd act as the data layer, storing schemas and observed states.
- Kueue and scheduler act as the async management layer, handling admission and placement.
- kubelet and container runtimes handle execution on the physical nodes.
- Pods, GPUs, and NCCL libraries constitute the data plane.
Token-generation pipelines do not route through the apiserver.

Horizontal scaling occurs within a single responsibility layer.
Communication across layers uses defined contracts such as RESTful HTTP, PVC mounts, or Kubernetes CRDs.

### Topology Layers
This architecture slices the system based on physical datacenter geography and failure domains.
Hierarchy: Region -> Availability Zone (AZ) -> physical rack -> host node -> container Pod -> underlying GPU/NIC.
The Kubernetes control plane relies on metadata labels applied to Nodes to perceive these boundaries.

| Typical Workload | Topological Placement Strategy |
|---|---|
| Stateless public API | Deploy across multiple AZs using topologySpreadConstraints. |
| High-performance training gang | Pack within the same physical rack. Require identical GPU architectures and RDMA. |
| Kubernetes control plane | Deploy on dedicated management nodes utilizing standard Ethernet. |
| Checkpoint persistent storage | Utilize a separate RWX storage cluster. It must constitute an independent failure domain. |

### Common Mixes and Mistakes
A common architectural error is treating a physical geographic zone as a responsibility layer.
In deep learning clusters, scattering a training gang across different AZs degrades all-reduce communication due to fiber latency.

A standard training task's path:
On the responsibility axis: Kueue handles admission → scheduler calculates placement → kubelet boots the process → NCCL manages communication.
On the topology axis: The 4 ranks reside within the same rack; they write checkpoints to RWX storage; the apiserver runs on a remote control plane node.

---

## 9 · Companion Lab

Kubernetes focuses on managing the lifecycle of individual Pods.
The lab control plane manages synchronized Worker groups instead.
Code lives in `project/LLMTrainLab/`. The README there is the hands-on path: landscape recipe → frontier cluster → LoRA / 70B full / vLLM → gang queue → kill a rank and recover.

```bash
cd project/LLMTrainLab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
llmctl tutorial print
llmctl tutorial run
```

After `tutorial run`: LoRA / 70B full / vLLM should be running, MoE RLHF and PCIe TP>1 stay Queued, and the LoRA job has `retries>=1`.

The older 8×H100 queue / preemption story is still:

```bash
llmctl demo canonical --preempt --virtual-nodes 40
```

---

## 10 · Primary Sources

- [Kubernetes components architecture overview](https://kubernetes.io/docs/concepts/overview/components/)
- [Pod lifecycle and detailed phase progression](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Scheduling Framework internals and mechanics](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Device Plugin design patterns and implementation](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [Jobs controller limitations and core design](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Persistent Volumes and storage claim mechanics](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Kueue advanced job queueing overview](https://kueue.sigs.k8s.io/docs/overview/)
- [KWOK zero-overhead mass fake node simulator](https://kwok.sigs.k8s.io/)
- [ByteDance: Robust LLM Training Infrastructure](https://www.alphaxiv.org/abs/2509.16293)
- [LLMTrainLab training control plane simulator](https://github.com/CurryTang/LLMTrainLab) (located in this repository: `project/LLMTrainLab/`)

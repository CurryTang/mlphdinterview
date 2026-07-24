# System Design 01 · Stateless Services

Course Location: [[SystemDesign00 Overview|00 Overview]] → This Section → [[SystemDesign01B Virtualization Containers|01B Virtualization and Containers]]

"Stateless" does not mean "the system has no state." It means that a service process does not exclusively own any non-recoverable state. An instance receives input, reads external state, completes computation, and then writes the result back to a shared system or returns it to the client.

```text
replaceable instance
= code + config + request context + rebuildable cache
```

If killing any single instance results in the loss of sessions, files, or business progress, that layer is not yet stateless.

---

## 1 · Why Learn Statelessness First

Load balancers can only perform true horizontal scaling when requests can be routed to any healthy instance:

```text
Client
  -> Load Balancer
      -> API-1
      -> API-2
      -> API-3
```

If user A's session exists only in API-1's memory, the load balancer must perform sticky routing. If API-1 crashes, the session is lost; furthermore, scaling and rolling deployments become complicated.

Direct benefits of stateless instances:

- Faulty instances can be replaced directly;
- Scaling only requires starting copies of the same version;
- Rolling deployments do not require migrating business state;
- The scheduler can place workloads on any node with available capacity.

The state has not disappeared; it has simply been moved to a system better suited for storing it.

---

## 2 · Classifying State

| State | Example | Location |
|---|---|---|
| Authoritative | Orders, users, final task status | Database / durable log |
| Durable blob | Images, models, checkpoints | Object storage |
| Coordination | Leases, locks, leader epochs | DB / coordination service |
| Rebuildable | Caches, indexes, materialized views | Redis / local cache / derived store |
| Request-local | Trace context, intermediate variables | Current process memory |

The key is not "can it be stored in memory," but rather "can we accept its disappearance after the process exits?" Request-local state and rebuildable caches are perfectly fine to keep locally.

### A Diagnostic Question

```text
If this process disappears now,
what information is permanently lost?
```

The answer should be "no business facts are lost." At most, work that is currently being calculated but not yet confirmed is lost, which the client or queue can safely retry.

---

## 3 · Common Externalization Patterns

### Externalizing Sessions

Stateful approach:

```text
API-1 memory: session_42 -> user_42
```

More scalable approach:

```text
opaque session_id
  -> shared session store
  -> user and permission context
```

You can also use short-lived signed tokens to allow instances to verify identity locally. However, immediate permission changes, forced logouts, and bans may still require checking a version or revoking state. JWT is not a universal, automatically stateless solution.

### Externalizing Files

Do not leave uploaded files only in a pod's `/tmp`. The API saves metadata, and the bytes go into object storage:

```text
DB: file_id, owner_id, object_key, checksum, status
Object store: actual bytes
```

Large files are typically uploaded directly using signed URLs to prevent the API server from becoming a data-transfer bottleneck. See [[SystemDesign04 Storage Systems|04 Storage Systems]] for storage selection.

### Externalizing Long-Running Tasks

A 20-minute report should not hold an HTTP request open:

```text
POST /jobs
  -> persist job(status=queued)
  -> enqueue(job_id)
  -> 202 Accepted

Worker
  -> claim job with lease
  -> execute
  -> persist result
```

The API does not track the task, and the worker does not own the task permanently. For queue acknowledgments, retries, and idempotency, see [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]].

---

## 4 · Keeping APIs and Workers Replaceable

### APIs Use Idempotency Keys

Clients will retry after timeouts. For creation-style APIs, the idempotency boundary should be placed in shared storage:

```sql
UNIQUE(user_id, idempotency_key)
```

When two instances receive the same request simultaneously, a database unique constraint is more reliable than "checking Redis first." Duplicate requests return the result or status of the original operation.

### Workers Use Leases

When a worker claims a task, it writes:

```text
worker_id
leased_until
attempt_count
```

The lease is renewed during execution; if the process crashes, the lease expires, and the task can be claimed by another worker. Business writes must still be idempotent, as the old worker might have already submitted the result but failed to acknowledge it.

### Local Caches Must Be Disposable

Items suitable for keeping in-process:

```text
parsed config
tokenizer
public keys
hot metadata
compiled template
```

These can be reloaded from the source of truth after an instance restarts. Update propagation can use short TTLs, version checks, or active invalidation; never let a local cache be the sole source of truth.

---

## 5 · Four Deployment Details

### Separate Readiness and Liveness

- Liveness: Is the process deadlocked and in need of a restart?
- Readiness: Is it currently suitable to receive new traffic?

Upon startup, load configurations and necessary resources before becoming ready. When shutting down, revoke readiness first to stop receiving new requests, then wait for in-flight work to finish.

### Graceful Shutdown

```text
receive SIGTERM
-> mark unready
-> stop accepting new work
-> finish / cancel in-flight work
-> flush safe telemetry
-> exit before grace period ends
```

Workers should stop claiming new tasks and release or shorten the leases of unfinished tasks. Simply using `kill -9` turns a normal deployment into a fault injection.

### Autoscaling Based on Saturation Metrics

APIs often monitor CPU, concurrency, p99 latency, and request queue depth; workers are better monitored by queue depth, oldest message age, and consumption rate.

Looking only at average CPU can miss single-shard hotspots or I/O wait. Scaling also has latency, so it must be paired with headroom, rate limits, and backpressure.

### Configurations Should Not Be Manually Modified on Hosts

Images and configurations should be versioned. Secrets should be injected via a secret manager or controlled mounts, avoiding hardcoding them into the image. See the next section, [[SystemDesign01B Virtualization Containers|01B Virtualization and Containers]], for delivery boundaries for containers and VMs.

---

## 6 · Where Does "Statelessness" Shift the Pressure?

Externalizing state simplifies the compute layer but increases shared dependencies:

- Session store failures affect all API replicas;
- Checking remote state on every request increases latency;
- Queues, DBs, and object storage require their own replication and backups;
- Cache misses can create synchronization storms.

Therefore, external state systems must have connection pooling, timeouts, bulkheads, caching, and capacity planning. You cannot simply push complexity to Redis or a DB and pretend the problem has disappeared.

### When Can Sticky Routing Be Used?

Sticky sessions are not forbidden. WebSockets, game rooms, and LLM KV caches can all benefit from affinity. However, durable state must still be externalized, with a recovery path prepared for instance failure.

```text
affinity = optimization
durable external state = correctness
```

By separating the two, the system remains fast when affinity is hit and remains correct when an instance is lost.

---

## 7 · State Layering in LLM Serving

LLM serving involves expensive GPU-local state compared to standard APIs, but the principles remain the same:

| State | Handling Method |
|---|---|
| Conversation / request | Durable store or carried by the client |
| KV cache | GPU-local, best effort, can be re-prefilled |
| Shared prefix cache | Rebuildable distributed cache |
| Model weights | Object store / model registry, reloadable |
| Adapter / LoRA | Versioned artifact, loadable on demand |

Losing the KV cache slows things down, but it should not result in the loss of conversation semantics. A router can prioritize sending the same session back to the original worker; if a miss occurs, it can re-prefill from the durable conversation.

Paged KV cache, prefix cache, disaggregated prefill/decode, and offloading are all state-placement optimizations that do not change the source of truth. Details are covered in the serving chapter of MLSYS and are not repeated here.

---

## 8 · Interview Checklist

```text
Process loss
- What is permanently lost if an instance is killed?

Routing
- Can requests be routed to any healthy instance?
- Is affinity an optimization or a correctness requirement?

State
- Where is the source of truth?
- Can local caches be rebuilt?
- Are files and long-running jobs offloaded from the local machine?

Lifecycle
- Are readiness and liveness separated?
- Does shutdown wait for in-flight work?
- Are retries protected by idempotency?

Scaling
- What metrics are used to scale APIs and workers?
- Will the external state system become a bottleneck?
```

In summary: Service instances can have in-memory state, but they must not exclusively own non-recoverable business state. Only then are they truly replaceable.

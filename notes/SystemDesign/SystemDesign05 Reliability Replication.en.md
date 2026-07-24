# System Design 05 · Reliability, Replication, and Failover

Course Location: [[SystemDesign04 Storage Systems|04 Storage Systems]] → This Section → [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]]

Redundancy is not as simple as "spinning up more machines." Where replicas are placed, who is authorized to write, how much replication is required before confirming success, and who is qualified to take over after a failure—these questions collectively determine whether a system can truly withstand a failure.

This section focuses solely on reliability and replication. For database selection, see [[SystemDesign02 Database Paradigms|02 Database Paradigms]]; for sharding, see [[SystemDesign03 Database Scaling|03 Database Scaling]]; for message retries, see [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]].

---

## 1 · Start with Failure Domains

Saying "we use three replicas" is insufficient. If all three copies reside on the same machine, in the same rack, or within the same availability zone, they provide no protection against larger-scale failures.

```text
process
  < machine
  < rack / power domain
  < availability zone
  < region
```

Before designing, answer this: Which level of failure must the system survive?

| Goal | Common Deployment Strategy |
|---|---|
| Service continuity during process crash | Multi-process on same or different machines, auto-restarted by supervisor |
| Service continuity during single-machine failure | Multi-machine replicas + load balancer health checks |
| Service continuity during single-AZ failure | Cross-AZ replicas, ingress capable of draining an entire AZ |
| Recovery from region failure | Cross-region replicas, traffic switching, and independent control planes |

Crossing larger failure domains typically increases latency and cost. Synchronous cross-region writes are particularly expensive, so do not default to global multi-active setups without specific RPO/RTO requirements.

---

## 2 · When is Redundancy Needed?

If any of the following conditions are met, you should seriously design for replication and failover:

- A single-instance failure causes core functionality to stop.
- The time to recover from backups exceeds the RTO.
- Single-machine maintenance causes unacceptable downtime.
- The data loss window exceeds the RPO.
- Read throughput requires multiple replicas to share the load.

Low traffic does not mean you can ignore reliability. A financial system with only a few dozen requests per day has low QPS, but its data may be more critical than a high-traffic recommendation cache. Reliability is determined by business impact, not by QPS alone.

### N+1 is about remaining capacity, not instance count

Assume peak traffic is 12K QPS, and a single instance safely handles 1K QPS at the target p99. Deploying 12 instances is just enough to run, but any single instance going offline will cause an overload.

```text
12K / 1K = 12 instances for traffic
12 × 1.3 headroom ≈ 16 instances
```

If an AZ has 6 instances, and the entire zone fails, only 10 instances remain, which still cannot handle 12K. In this case, you must recalculate based on the "largest failure domain going offline" rather than just checking single-machine N+1.

```text
surviving capacity after failure >= peak load
```

---

## 3 · Primary-Replica: One Write Ingress, Multiple Replicas

Primary-Replica is also known as leader-follower or master-slave. One node handles writes, while others replicate logs or data changes from it.

```text
Client write
    -> Primary
        -> Replica A
        -> Replica B

Client read
    -> Primary            strong / fresh path
    -> Replica A or B     scalable but may be stale
```

This separates two problems: who decides the write order and who stores the replicas. A single write ingress makes conflict resolution much simpler.

### Synchronous vs. Asynchronous Replication

Synchronous replication requires confirmation from one or more replicas before the primary returns success to the client:

```text
write latency ≈ primary work + replica network RTT + replica durable write
```

The advantage is that confirmed data is less likely to be lost during failover; the cost is higher write latency, and slow replicas can stall writes.

Asynchronous replication returns success from the primary first, with replicas catching up later. Writes are fast, but if the primary fails permanently before replication, the most recent data may be lost.

| Choice | Best Suited For |
|---|---|
| Synchronous to quorum | Ledgers, orders, metadata requiring very low RPO |
| Synchronous to same-AZ replica, async cross-region | Balancing local durability with cross-region latency |
| Fully asynchronous replicas | Reconstructible data, caches, systems allowing defined data windows |

"Synchronous" is not a boolean value. You must clarify how many replicas are synchronized, whether it crosses AZs, and whether confirmation means writing to memory or durable storage.

### Read replicas can cause stale reads

Distributing read traffic to replicas can scale throughput, but replication lag can break read-your-writes consistency:

```text
User writes profile = v2 to primary
User immediately reads from lagging replica
Replica still returns v1
```

Common handling methods include:

- Routing a user's read requests to the primary for a short period after a write.
- Returning a version/commit position and only reading from replicas that have caught up to that position.
- Pinning reads to the primary for the few APIs that require strong consistency.
- Having the UI use the locally submitted result immediately.

Do not generalize by saying "the database is strongly consistent." Within the same system, different read paths can have different consistency levels.

---

## 4 · Failover: Existence of Replicas Does Not Mean They Can Take Over

The truly difficult part is safely promoting a replica to a new primary.

```text
1. Failure detector suspects primary is unavailable
2. Threshold reached, avoiding triggers from brief jitters
3. Select a replica with sufficiently fresh data
4. Fence the old primary
5. Promote new primary, update routing / service discovery
6. Restore traffic, check lag and data discrepancies
```

### Why Fencing is Mandatory

The old primary might just be network-isolated, not truly dead. If both the new and old primaries accept writes, a split-brain scenario occurs.

The goal of fencing is to ensure the old node cannot continue writing even if the network recovers:

- Use a higher epoch/term, where storage only accepts writes from the latest term.
- Revoke the old node's lease.
- Cut off the old node's permissions at the network or storage layer.
- Use forced isolation mechanisms like STONITH for physical hosts.

"Promoting a replica when health checks fail" misses the most critical half: ensuring the old primary cannot write.

### Failure detectors cannot be perfectly accurate

A timeout only proves "no response received within the specified time," not that the node is permanently dead. Thresholds that are too short cause false positives; those that are too long increase RTO.

When designing, specify:

```text
probe interval
failure threshold
who makes the decision
whether quorum is required
how the old primary is fenced
```

---

## 5 · Active-Passive: One Serves, One Takes Over on Failure

Active-Passive is often called master-standby or hot-standby. The Active node handles traffic, while the Passive node remains in a state ready to take over.

| Standby | What it does normally | Switchover Speed | Cost |
|---|---|---|---|
| Cold | Only backups and deployment templates | Minutes to hours | Low |
| Warm | Instance running, data syncing, capacity may be smaller | Tens of seconds to minutes | Medium |
| Hot | Full capacity online, data near real-time sync | Seconds | High |

"Hot standby" must protect at least four things:

1. Data on the standby side is sufficiently fresh.
2. The standby side has sufficient capacity, not just an empty shell.
3. Ingress can be switched; DNS, LB, or routing will not point back to the old end.
4. The switchover process is practiced regularly.

A failover that hasn't been practiced is just a wish in the documentation.

### What is Active-Passive suitable for?

- Single-write systems where you want to keep the conflict model simple.
- A second region primarily used for disaster recovery.
- Traffic that doesn't justify the long-term complexity of active-active.
- Businesses that can accept a defined RTO.

Switching back to the original region must also be designed. Failback is not as simple as changing DNS back; it requires syncing new data, confirming the state of the old region, and avoiding double-writes again.

---

## 6 · Active-Active: Both Sides Serve

Active-Active (multi-master, multi-active) allows multiple sites to handle traffic simultaneously. The read path is usually not difficult; the challenge is that multiple locations can modify the same logical data.

```text
Region A writes user_42 = v2
Region B writes user_42 = v3
network partition delays replication
```

The system must define how v2 and v3 are merged. Common methods are not free:

| Method | Cost |
|---|---|
| Fixed home region per key | Cross-region write latency, ownership migration needed if home region fails |
| Last-write-wins | Simple, but clock skew and overwrites may lose business intent |
| Version vector / causal metadata | Increased metadata and implementation complexity |
| CRDT | Only suitable for data types with definable merge rules |
| Business conflict handling | Semantically correct, but each entity must be designed individually |

Many "global multi-active" setups are actually active-active serving with single-writer ownership: every region takes traffic, but there is still only one write master for a specific user or partition. This is usually more controllable than true multi-master concurrent writes.

### When is Active-Active worth it?

- Users are distributed across continents, and cross-region RTT already breaks latency targets.
- Single-region failure must be nearly imperceptible.
- Business data can be clearly partitioned by geography or tenant.
- The team is willing to bear the costs of conflicts, observability, drills, and data repair.

If these conditions are not met, Active-Passive is often more honest.

---

## 7 · Quorum: How Replication Count Affects Reads and Writes

In a system with N replicas, W is commonly used to represent how many replicas must confirm a write, and R represents how many replicas are queried for a read.

```text
N = 3
W = 2
R = 2
```

When `W + R > N`, the read and write sets theoretically intersect. However, this inequality does not automatically provide linearizability; it still requires version selection, conflict resolution, failure recovery, and correct membership.

Choosing a larger W:

- Confirmed writes are more fault-tolerant.
- Write latency depends on more replicas.
- Writability may decrease during network partitions.

Choosing a larger R:

- Easier to read newer versions.
- Read amplification and tail latency increase.
- Slow replicas are more likely to enter p99.

Do not treat quorum as a universal answer. In an interview, at least explain what N, R, and W are, and why they meet the latency and failure targets of the problem.

---

## 8 · Replicas are not Backups

Replicas quickly replicate normal writes, but they also quickly replicate accidental deletions, corrupted data, and the results of ransomware encryption. Backups save a historical moment and can be used to go back in time.

| Mechanism | Primarily Protects Against |
|---|---|
| Replica | Node failure, read scaling, fast failover |
| Snapshot | Data state at a specific point in time |
| WAL / incremental backup | Point-in-time recovery |
| Cross-account immutable backup | Operational errors, permission leaks, ransomware attacks |

Backups must address retention, encryption, access permissions, and restore tests. Saying "we back up daily" is not enough:

```text
Can we restore?
How long does restore take?
What data window is lost?
Does restored data include schema and encryption keys?
```

RPO describes how much data loss is acceptable, and RTO describes how quickly service is restored:

```text
RPO = 5 min   -> Worst case: lose the last 5 minutes of writes
RTO = 30 min  -> Restore core functionality within 30 minutes of failure
```

These two numbers determine replication frequency, standby capacity, and the degree of automation.

---

## 9 · How to Choose Between Multi-AZ and Multi-Region

### Multi-AZ

Latency across AZs within the same region is low, making it suitable for synchronous replication and automatic failover. It is usually the first step for high-availability systems.

Check:

- Are replicas truly across AZs?
- What is the remaining capacity after an AZ goes offline?
- Do LB, queue, cache, and database share the same implicit failure domain?
- Do dependencies like NAT, DNS, and KMS also have redundancy?

### Multi-Region

Cross-region can handle larger disasters and reduce latency for global users, but it introduces replication latency, data sovereignty issues, costs, and operational complexity.

A common progressive path is:

```text
single instance
-> multi-instance in one AZ
-> multi-AZ active-passive / replicated
-> cross-region warm standby
-> active-active serving with controlled write ownership
```

Each step should be driven by new failure or latency targets, not because the architecture diagram looks more complete.

---

## 10 · Capacity and Availability Estimation

### Availability budget

```text
99.9%   ≈ 43.8 min downtime / month
99.99%  ≈ 4.38 min / month
99.999% ≈ 26 sec / month
```

The availability of serial systems is multiplicative. If a request must pass through services A, B, and C, each with 99.9% availability:

```text
0.999³ ≈ 99.7%
```

Real systems have shared dependencies and correlated failures; you cannot simply multiply independent probabilities optimistically. The value of this calculation is to remind us: the longer the synchronous critical path, the harder it is to maintain overall availability.

### Replication bandwidth

If the primary write peak is 200 MB/s, and it is asynchronously replicated to two remote replicas:

```text
outbound replication ≈ 200 MB/s × 2 = 400 MB/s
```

Leave headroom for retransmissions, compression ratio changes, and replica reconstruction. New replica bootstrapping is often heavier than daily incremental replication and should be rate-limited or started from a snapshot.

### Replication lag

```text
lag growth rate = primary write rate - replica apply rate
```

If the primary writes at 200 MB/s and the replica can only apply at 150 MB/s, the backlog grows by 50 MB per second. Ten minutes equals about 30 GB. In this state, replication "still working" does not mean the replica can handle a failover.

---

## 11 · A Selection Table

| Requirement | Reasonable Starting Point | Costs to Clarify |
|---|---|---|
| No downtime on single-machine failure | Multi-instance + health check | State must be externalized, see 01 |
| Database read-heavy, write-light | Primary + read replicas | Lag, read-your-writes |
| No downtime on single-AZ failure | Cross-AZ sync or quorum replication | Write latency, N-1 capacity |
| 30-min recovery after region disaster | Warm standby | RPO, switchover, and failback |
| Global low-latency reads | Multi-region read replicas / cache | Stale reads, invalidation propagation |
| Global low-latency writes | Partition ownership or Active-Active | Conflicts, fencing, data sovereignty |
| Protection against accidental deletion | Versioned immutable backup | Recovery time, storage cost |

---

## 12 · Interview Checklist

```text
Failure domain
- Must it survive process, machine, AZ, or region failure?
- Are there shared dependencies causing all replicas to fail together?

Capacity
- Is remaining capacity sufficient after the largest failure domain goes offline?
- Will new replica reconstruction overwhelm online traffic?

Data
- Who can write? Is replication synchronous or asynchronous?
- Will confirmed writes be lost during failover?
- Which APIs are affected by read replica lag?

Failover
- Who detects failure, and what is the threshold?
- How is the old primary fenced?
- How is routing switched, and how long does it take?
- How is failback performed?

Recovery
- Are replicas and backups designed separately?
- What are the RPO and RTO?
- When was the last restore / failover drill?
```

Reliability design must ultimately boil down to a specific failure: assume an AZ disappears right now—which requests fail, what data might be lost, how long until recovery, and who executes the switchover. Being able to explain this timeline is far more useful than listing a string of high-availability buzzwords.

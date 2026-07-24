# System Design 00 · From Problem to Architecture

System design interviews are not about memorizing a "big tech architecture diagram." The problems posed by interviewers are usually broad; the real work lies in narrowing them down into a few calculable problems and ensuring every component has a clear justification.

This note focuses solely on methodology. Databases, replication, messaging systems, and case studies are covered in subsequent chapters to avoid repeating the same knowledge across different notes.

---

## How to Read These Notes

| Order | Chapter | Problem Solved |
|---|---|---|
| 00 | This Note: Design Methodology | What to say first, how to calculate, how to drive the interview |
| 01 | [[SystemDesign01 Stateless Service|Stateless Services]] | How to make service instances replaceable and horizontally scalable |
| 01B | [[SystemDesign01B Virtualization Containers|Virtualization and Containers]] | How VMs, containers, and Linux isolation mechanisms work |
| 02 | [[SystemDesign02 Database Paradigms|Database Paradigms]] | How to choose between RDBMS, NoSQL, consistency, and transactions |
| 03 | [[SystemDesign03 Database Scaling|Database Scaling]] | How to implement replication, sharding, and partitioning |
| 04 | [[SystemDesign04 Storage Systems|Storage Systems]] | What to store in block, file, and object storage |
| 05 | [[SystemDesign05 Reliability Replication|Reliability and Replication]] | How to design primary-replica, active-active, hot standby, and RPO/RTO |
| 06 | [[SystemDesign06 Async Messaging Systems|Async Messaging Systems]] | How to distinguish between Queues, Pub/Sub, Kafka, and Event Buses |
| 07 | [[SystemDesign07 Photo Sharing Feed|Photo Sharing and Feeds]] | Applying general methods to classic internet systems |
| 08 | [[SystemDesign08 LLM Async RL Platform|Async LLM RL Platform]] | Applying general methods to ML infrastructure |
| 09 | [[SystemDesign09 Consistent Hashing|Consistent Hashing]] | How to route keys stably in dynamic clusters and reduce data migration |
| 99 | [[SystemDesign99 Glossary|Glossary]] | Quick lookup before interviews; not a comprehensive tutorial |

You can follow the numbering for your first read. When preparing for an interview, you don't need to read from start to finish: start with this note, then pick one case study to walk through completely, and return to the foundational chapters only when you get stuck.

---

## The Core Thread: Five Steps to Complete a Design

```text
1. Problem navigation
   What exactly are we building, and which features are out of scope?

2. Back-of-the-envelope estimation
   What are the peak QPS, bandwidth, storage, concurrency, and internal amplification?

3. API + data model
   What is the contract exposed to the outside, and what does the source-of-truth data look like?

4. High-level architecture
   Get a minimal read/write path running first.

5. Deep dive
   Identify bottlenecks based on NFRs and gradually add components like cache, replicas, queues, and shards.
```

The order is not rigid. When designing a chat system, the connection model might need to be discussed early; when designing a ledger, transactions and data models might appear before APIs. However, none of the five steps should be skipped.

---

## Step 1 · Problem navigation

### Functional requirements: What are we actually building?

Identify the verbs in the prompt. When designing a photo-sharing system, you might determine:

```text
Must have
- Users upload photos and publish posts
- Users view home feed

Out of scope
- Comments, search, ads
- Video editing
- Recommendation model training
```

It is best to limit functional requirements to two or three core actions. The broader the scope, the more likely the QPS, data models, and architecture will conflict later.

This step also requires clarifying product semantics:

- Is the feed sorted by time or by recommendation?
- Should deletions be immediately invisible, or is a short delay acceptable?
- Does "upload successful" mean "original image received" or "all thumbnails are readable"?
- Does the user require read-your-writes consistency?

These questions directly change the design. For example, different definitions of "upload successful" lead to different API response timings and state machines.

### Non-functional requirements: To what extent must the system perform?

Don't just list a string of nouns. Each NFR should ideally map to a number or an observable behavior:

| Dimension | Discussable Goal |
|---|---|
| Latency | Feed read p99 < 200 ms |
| Availability | 99.99%, service continues during a single AZ failure |
| Durability | Confirmed uploaded original images must not be lost |
| Consistency | Author read-your-writes; seconds-level delay allowed for general feeds |
| Freshness | New posts appear in most followers' feeds within 5 seconds |
| Recovery | RPO < 5 min, RTO < 30 min |
| Growth | Horizontally scalable to 10x current load |

Goals often conflict. Synchronous cross-region replication improves RPO but increases write latency; caching reduces read latency but introduces stale data; automatic retries improve success rates but may amplify traffic during failures. The point of a design question is not to make every metric perfect, but to clarify priorities.

### Deliverables for this step

```text
Scope: upload photo + home feed
Traffic shape: read-heavy, bursty
Latency: feed p99 < 200 ms
Availability: 99.99%
Consistency: author read-your-writes; 5s eventual consistency for others
Durability: confirmed media must survive an AZ loss
```

With these lines, you have a standard for judging architectural choices later.

---

## Step 2 · BOE: Calculate numbers that change the design

Back-of-the-envelope estimation is not a math test. Only calculate quantities that might change the system's shape.

### Common conversions

```text
1 day ≈ 100,000 seconds
average QPS ≈ daily requests / 100,000
peak QPS ≈ average QPS × peak factor
bandwidth ≈ QPS × average payload size
storage growth ≈ writes/day × bytes/write × retention
concurrency ≈ QPS × average latency in seconds
```

There is no universal answer for the peak factor. Internal systems with steady traffic might use 2; consumer products often use 3 to 5 for a first-pass estimate; live streaming, flash sales, and training scheduling require separate analysis for bursts.

### A small example

Assumptions:

```text
10M DAU
20 feed reads per person per day
0.1 posts per person per day
Peak factor = 4
Average raw image size = 3 MB
```

Ingress traffic:

```text
feed average QPS = 10M × 20 / 100K ≈ 2K
feed peak QPS ≈ 8K

upload average QPS = 10M × 0.1 / 100K ≈ 10
upload peak QPS ≈ 40

raw media growth/day = 10M × 0.1 × 3 MB = 3 TB/day
```

The most useful conclusion here isn't that "8K is large." What actually changes the design is: reads far exceed writes; image bandwidth is far greater than metadata; long-term storage growth is more concerning than API QPS. Therefore, images should be uploaded directly to object storage, reads should go through a CDN, and business services should not handle raw bytes.

### Don't miss internal amplification

A user ingress of 40 upload QPS does not mean the system internal load is only 40:

```text
1 upload
  -> 4 image variants
  -> content scan
  -> metadata update
  -> one PostReady event
  -> N follower timeline writes
```

Fan-out, retries, replication, and index maintenance all amplify internal traffic. In an interview, you should state both the external QPS and the heaviest internal QPS.

### What does high or low QPS mean?

There is no universal "high/low" line for QPS across systems. For 5K QPS, an in-memory KV store might be light, while complex SQL, GPU inference, or third-party APIs might be heavy. The correct question is: how much can a single instance or shard handle safely under the target p99?

```text
required instances
= peak QPS / safe QPS per instance
× headroom
```

For example, if a service can stably handle 600 QPS at the target p99 after stress testing, with a peak of 8K and 30% headroom:

```text
8,000 / 600 × 1.3 ≈ 18 instances
```

At low loads, a monolithic service plus a relational database is often sufficient. Ensure backups, monitoring, and failure recovery first. Handle bottlenecks as traffic increases:

| Phenomenon | Check First | Common Actions |
|---|---|---|
| Repeated reads slow down DB | Hotspots, query plans, connection pools | Cache, read replica, indexing |
| Service CPU saturated | Single-instance throughput, p99 | Statelessness, horizontal scaling |
| Write bursts | Peak vs. consumption rate | Queue, batching, backpressure |
| Single DB write limit | Hot partitions, transaction boundaries | Sharding, split tables by access pattern |
| Large files saturate API bandwidth | Payload size | Signed URL, object storage, CDN |

"QPS is high, so add Kafka" is not a valid deduction. Kafka solves event delivery, buffering, and replay; it does not make a slow consumer faster.

### How to estimate Cache and TTL

Calculate whether the cache is worth it:

```text
DB QPS after cache
= total read QPS × (1 - hit rate)
```

If the database can safely handle 2K read QPS, and peak read traffic is 8K:

```text
required hit rate >= 1 - 2K / 8K = 75%
```

Capacity estimation starts from the working set:

```text
cache bytes
≈ active keys × average value bytes × metadata overhead × replica factor
```

TTL is not "the longer the better." It is constrained by four factors: how stale the business allows data to be, how often data changes, whether invalidation notifications are reliable, and whether the backend can handle cache penetration. A practical starting point is:

```text
TTL <= allowed staleness
Add 10%~20% jitter to TTL to avoid mass key expiration
Use request coalescing or single-flight for hot keys
```

Data like permissions, balances, and ban status usually cannot rely solely on long TTLs. You can shorten the TTL and combine it with active invalidation; the true source of truth remains in the database.

---

## Step 3 · Define APIs and source-of-truth data first

APIs are for exposing business semantics, not for mapping database tables directly to HTTP.

```http
POST /v1/posts
-> { post_id, upload_url, expires_at }

POST /v1/posts/{post_id}/publish
-> { operation_id, status }

GET /v1/feed?cursor=...
-> { items, next_cursor }
```

When writing APIs, specify at least:

- Whether the request is idempotent, and where the idempotency key is placed;
- Whether the result is returned synchronously or via an operation handle;
- Whether pagination uses offsets or cursors;
- Whether the client can safely retry after failure.

For data models, find the source of truth first, then list derived data:

```text
Post                  authoritative
- post_id
- author_id
- media_object_key
- status
- created_at

TimelineEntry         derived / rebuildable
- viewer_id
- rank_key
- post_id
```

If data is lost, it can be rebuilt from the Post and follow graph; therefore, TimelineEntry should not be the sole source of truth. This distinction affects TTL, backups, and consistency strategies.

See [[SystemDesign02 Database Paradigms|02 Database Paradigms]] for database selection; [[SystemDesign03 Database Scaling|03 Database Scaling]] for sharding and replicas; and [[SystemDesign04 Storage Systems|04 Storage Systems]] for images, files, and model weights.

---

## Step 4 · Draw the minimal closed loop first

Don't start by plastering Kafka, Redis, and a dozen microservices on the board. Get the core request to flow through completely first:

```system-design-overview-visual
```

The first line in the diagram is the synchronous closed loop:

```text
Client -> Gateway -> Stateless Service -> Primary Store
```

Branch components must answer a specific question:

| Component | Reason for inclusion |
|---|---|
| Cache | Repeated reads are stressing the DB or missing latency targets |
| Queue / Event Log | Work doesn't need to be synchronous; need load leveling, retries, or broadcasting |
| Worker | Slow tasks need independent scaling |
| Replica | Need failover or read scaling |
| Shard | Single-node capacity or write throughput is a boundary |
| Object storage | Data objects are large, access is simple, need high durability |

After drawing, explain the four paths clearly: normal read, normal write, asynchronous work, and failure recovery. Drawing component names without explaining the data flow makes even a complex diagram useless.

---

## Step 5 · Use NFRs to select a deep dive

A deep dive is not about piling more boxes onto the diagram, but picking one or two problems most likely to determine success or failure.

| Pressure exposed by problem | What to deep dive into | Corresponding note |
|---|---|---|
| Instance failure, cross-AZ, RPO/RTO | Failover, replication, fencing | [[SystemDesign05 Reliability Replication|05 Reliability and Replication]] |
| Service cannot scale horizontally | How to externalize sessions, files, tasks | [[SystemDesign01 Stateless Service|01 Stateless Service]] |
| Data model and consistency hard to choose | Transaction boundaries, access patterns | [[SystemDesign02 Database Paradigms|02 Database Paradigms]] |
| Single DB capacity or throughput limit | Replicas, partition keys, resharding | [[SystemDesign03 Database Scaling|03 Database Scaling]] |
| Tasks slow, bursty, many downstream | Queues, consumer groups, outbox | [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]] |
| Images or large files dominate traffic | Direct upload, CDN, object storage | [[SystemDesign04 Storage Systems|04 Storage Systems]] |

### Remember only one entry point for redundancy

When a component failure stops core functionality and the recovery time exceeds the target RTO, consider redundancy. Determine the failure domain first, then discuss the number of replicas: process, machine, rack, AZ, or region.

```text
N-1 capacity check:
Can the remaining replicas handle peak traffic after one failure domain goes offline?
```

The differences between Primary-Replica, Active-Passive, and Active-Active, as well as failover details, are covered in [[SystemDesign05 Reliability Replication|05 Reliability and Replication]] and will not be repeated here.

### Remember only one entry point for asynchrony

When work does not need to be completed before the request returns, and you need persistent handoffs, buffering, or independent retries, consider a Queue / Event Log. Answer three questions first:

```text
When is a task considered "accepted" by the broker?
Is the consumer idempotent when messages are duplicated?
How does the system apply backpressure when production rate consistently exceeds consumption rate?
```

Queues, Pub/Sub, Kafka, RabbitMQ, Event Bus, Webhooks, and outbox patterns are covered in [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]].

---

## How to allocate time in an interview

Using 45 minutes as an example:

```text
0 - 6 min    Narrow down functional and NFR requirements
6 - 11 min   BOE, identify the real big numbers
11 - 17 min  API and data model
17 - 27 min  Minimal architecture and main read/write paths
27 - 40 min  Deep dive into one bottleneck
40 - 45 min  Failure, trade-offs, and scaling questions
```

Do not force an interruption of a valuable discussion just to stick to the schedule. The schedule is meant to prevent you from spending twenty minutes on requirements or finishing the diagram just as time runs out.

---

## Final Checklist

### Requirements

- Are there only two or three core functions?
- Do NFRs have numbers or clear semantics?
- Which features are explicitly out of scope?

### Numbers

- Are averages and peaks separated?
- Did you calculate payload, retention, and internal fan-out?
- Are assumptions written for every capacity conclusion?

### Architecture

- Where is the source of truth?
- Can derived data like caches, timelines, and indexes be rebuilt?
- Does every new component correspond to a bottleneck or NFR?
- Have both normal and failure paths been discussed?

### Reliability

- What is the failure domain?
- Is there enough capacity after a single replica failure?
- Do retries have timeouts, limits, backoff, and idempotency protection?
- Have backups actually been tested with recovery drills?

Conclude with two sentences: what the current design prioritizes, and what trade-offs were accepted for it. Being able to articulate these two points usually indicates that the entire design was derived by you.

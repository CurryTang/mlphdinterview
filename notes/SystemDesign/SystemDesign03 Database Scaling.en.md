zsh:1: command not found: ...
# System Design 03 · The Database Scaling Trio

Course Location: [[SystemDesign02 Database Paradigms|02 Database Paradigms]] → This Article → [[SystemDesign04 Storage Systems|04 Storage Systems]]

This article discusses three core aspects of database scaling: how to distribute read pressure to replicas, how to shard writes and capacity, and how to combine these approaches. Fault detection, fencing, hot standby, and cross-region RPO/RTO are covered in [[SystemDesign05 Reliability Replication|05 Reliability and Replication]] and will not be repeated here.

## 0. Basic Concepts: QPS, IOPS, Throughput, and Latency

Before tackling database scaling problems, clarify these metrics. Many interview answers fail not because the "solution" is wrong, but because the candidate didn't first estimate whether the system is bottlenecked by CPU, network, disk, database connections, or single-node capacity.

### 0.1 QPS / RPS / TPS

QPS stands for queries per second. RPS stands for requests per second. TPS stands for transactions per second, often used for database transactions or payment processing.

They are often similar but not identical:

| Metric | Common Meaning | Example |
| --- | --- | --- |
| RPS | Service entry requests | API Gateway receives 10k HTTP requests per second |
| QPS | Query requests | Search service processes 20k queries per second |
| TPS | Successful transactions | Payment service completes 500 transactions per second |
| DB QPS | Database query count | If one API request hits the DB 5 times, DB QPS may be 5x API RPS |

A common pitfall:

~~~text
User QPS != Database QPS

1 API request
  -> Read user profile
  -> Read feature flags
  -> Query order list
  -> Write audit log

Entry RPS = 1
DB operations = 4
DB QPS is approximately 4
~~~

### 0.2 Throughput, Latency, and Concurrency

Throughput is the amount of work completed per unit of time; latency is the time taken for a single request; concurrency is the number of requests in the system at a given moment.

These can be estimated using Little's Law:

$$
\text{concurrency} \approx \text{QPS} \times \text{latency}
$$

Note: Latency must be converted to seconds.

Example:

~~~text
QPS = 10,000 requests/s
Average latency = 100 ms = 0.1 s

Average concurrent requests in the system:
10,000 * 0.1 = 1,000
~~~

This means even with 10,000 requests per second, if each request stays in the system for 100ms, the system must handle approximately 1,000 in-flight requests simultaneously.

Memory Diagram:

~~~mermaid
flowchart LR
  A["QPS: How many enter per second"] --> D["Concurrency: How many are in the system simultaneously"]
  B["Latency: How long each request stays"] --> D
  D --> C["Thread / Connection / Queue / Memory pressure"]
~~~

### 0.3 Average QPS and Peak QPS

DAU, MAU, and total request volume usually only provide average QPS. You must estimate peak QPS for system design.

There are:

$$
24\times 60\times 60 = 86400 \approx 10^5 \text{ seconds in a day}
$$

Therefore:

$$
\text{avg QPS} \approx \frac{\text{daily requests}}{10^5}
$$

Peak QPS can be roughly estimated by multiplying by a factor:

~~~text
peak QPS = avg QPS * peak factor

General business: peak factor 3~5
Noticeable tidal traffic: peak factor 5~10
Flash sales/Hot events: potentially 10~100+
~~~

Example:

~~~text
100 million requests per day
avg QPS ≈ 100,000,000 / 100,000 = 1,000

If peak factor = 5
peak QPS ≈ 5,000
~~~

In an interview, it is more important to state your assumptions than to memorize a specific multiplier.

### 0.4 IOPS and Disk Bandwidth

IOPS (input/output operations per second) represents how many I/O operations a storage system can handle per second. It is primarily used to estimate random read/write pressure.

Bandwidth/throughput represents how much data can be transferred per second, often used for large sequential reads/writes.

| Metric | Focus | Typical Bottleneck |
| --- | --- | --- |
| IOPS | Number of read/write operations per second | Small random I/O, index lookups, KV gets |
| Bandwidth | MB/GB per second | Large file scanning, backups, log transmission |
| Latency | Time for a single I/O | Tail latency, synchronous write paths |

A rough estimate:

$$
\text{required IOPS}
\approx
\text{QPS} \times \text{I/O ops per request}
$$

If each request requires 3 random reads and 1 random write:

~~~text
API peak QPS = 5,000
I/O per request = 4

required IOPS ≈ 20,000
~~~

If each request also reads 20KB of data, the network or disk bandwidth is approximately:

$$
\text{bandwidth} \approx \text{QPS} \times \text{bytes per request}
$$

~~~text
5,000 QPS * 20 KB ≈ 100 MB/s
~~~

These two estimates answer different questions:

~~~text
Many small random reads:
  Look at IOPS

Many large sequential reads:
  Look at bandwidth
~~~

### 0.5 Common Capacity Estimation Templates

#### Storage Capacity

~~~text
daily data = daily writes * average record size
retention storage = daily data * retention days * replication factor
~~~

Example:

~~~text
100 million events per day
500 bytes per event
30 days retention
3 replicas

raw daily data = 100,000,000 * 500B = 50GB/day
total storage ≈ 50GB * 30 * 3 = 4.5TB
~~~

#### Database Read/Write Splitting

~~~text
read QPS = total QPS * read ratio
write QPS = total QPS * write ratio
replica count ≈ read QPS / safe read QPS per replica
~~~

Example:

~~~text
peak QPS = 20,000
Read/Write ratio = 90% read, 10% write

read QPS = 18,000
write QPS = 2,000

If a single replica safely handles 4,000 read QPS
At least 5 read replicas are needed
~~~

#### Cache Hit Backend Pressure

~~~text
backend QPS = total QPS * (1 - cache hit rate)
~~~

Example:

~~~text
total QPS = 100,000
cache hit rate = 95%

backend QPS = 100,000 * 5% = 5,000
~~~

This is why in high-QPS systems, a drop in cache hit rate from 95% to 90% is critical: backend pressure doubles.

#### Queues and Worker Count

If the average task processing time is $T$ seconds and each worker processes one task at a time, the throughput per worker is approximately:

$$
\text{worker throughput} \approx \frac{1}{T}
$$

Required number of workers:

$$
\text{workers} \approx \text{arrival QPS} \times T
$$

Example:

~~~text
200 tasks arrive per second
Average processing time per task = 0.5 seconds

Concurrent workers needed ≈ 200 * 0.5 = 100
~~~

### 0.6 How to Use These Numbers in an Interview

In system design, estimation is not for precision, but for deciding the architectural direction.

~~~mermaid
flowchart TD
  A["Estimate QPS / storage / bandwidth / IOPS"] --> B{"Can a single node handle it?"}
  B -->|High read pressure| C["replica / cache / read pool"]
  B -->|High write pressure| D["partition / queue / batch"]
  B -->|High capacity| E["sharding / cold storage / retention"]
  B -->|High latency| F["index / cache / async / locality"]
  B -->|High peak| G["autoscale / rate limit / backpressure"]
~~~

A solid sequence for your answer:

~~~text
1. Estimate entry QPS and peak QPS.
2. Estimate how many DB / cache / storage hits each request generates.
3. Convert entry QPS into backend QPS, IOPS, and bandwidth.
4. Identify whether the bottleneck is read, write, capacity, or latency.
5. Choose replication, sharding, caching, queuing, or asynchronous processing.
~~~

---

## 0.7 First, Identify the Source of Pressure

In database scaling problems, don't immediately suggest "adding a cache" or "sharding." First, identify the system bottleneck:

~~~mermaid
flowchart TD
  A["Database pressure"] --> B{"What is the main pressure?"}
  B -->|Too many read requests| C["Primary-replica replication + Read/Write splitting"]
  B -->|Risk of primary unavailability| D["Multi-primary / Primary-standby / Auto-failover"]
  B -->|Too many write requests| E["Data partitioning / Sharding"]
  B -->|Data volume too large| E
  C --> F["Cost: replication lag / stale read"]
  D --> G["Cost: conflict resolution / complex failover"]
  E --> H["Cost: cross-shard queries / rebalancing / hot shards"]
~~~

Keep this summary in mind:

| Pattern | Solves | Does Not Solve |
| --- | --- | --- |
| Primary-replica replication | Read scaling, keeps replicas for failover | Does not scale primary write capacity, does not replace backups |
| Multi-primary replication | Multiple write entry points or more flexible takeover | Does not linearly double write capacity, introduces conflicts |
| Data partitioning | Capacity scaling, write scaling, smaller indexes | Increases query routing and cross-shard complexity |

---

## 1. Primary-Replica Replication: Solving Read Scaling First

The basic structure of primary-replica replication is:

~~~text
Primary / Master  ->  Replica / Slave
~~~

The primary receives writes, and replicas replicate the primary's data. All operations that modify data go to the primary:

~~~text
INSERT
UPDATE
DELETE
CREATE TABLE
ALTER TABLE
~~~

Replicas typically do not receive business writes directly; they follow the primary's change logs to update local data.

### 1.1 How the Replication Link Works

Taking MySQL as an example, the core of primary-replica replication is the binlog. After the primary executes a data modification, it writes the change to the binary log; the replica continuously pulls the incremental binlog from the primary, writes it to the relay log, and then replays these operations locally.

~~~mermaid
sequenceDiagram
  participant C as Client
  participant P as Primary
  participant B as Binlog
  participant R as Replica
  participant L as Relay Log

  C->>P: write request
  P->>P: execute mutation
  P->>B: append change event
  R->>B: pull changes after known position
  R->>L: write relay log
  R->>R: replay relay log
~~~

Essentially, it is three steps:

~~~text
Primary records changes;
Replica pulls changes;
Replica replays changes locally.
~~~

### 1.2 Uses of Primary-Replica Replication

There are four common uses for primary-replica replication.

First, read/write splitting. Write requests go to the primary, and read requests are distributed among multiple replicas:

~~~text
Write  -> Primary
Read   -> Replica 1 / Replica 2 / Replica 3
~~~

For example:

| Request | Routing |
| --- | --- |
| User browses products | Replica |
| User views order list | Replica |
| User updates address | Primary |
| User places order/pays | Primary |

This is suitable for read-heavy, write-light systems. It must be emphasized: primary-replica replication scales read capacity, not write capacity.

Second, query isolation. Different replicas can handle different types of read tasks:

~~~text
Replica 1: Online general queries
Replica 2: Reporting queries
Replica 3: Backup tasks
~~~

This prevents slow queries, reports, and backups from dragging down the primary.

Third, zero-downtime backups. The primary continues to serve online requests while the replica performs backup tasks.

Fourth, replica failover. If a replica goes down, it can be removed from the read traffic pool, and requests are redirected to other replicas.

### 1.3 Core Cost: Replication Lag

Primary-replica replication is usually asynchronous. The primary can return success after completing a write without waiting for all replicas to finish replaying.

This leads to stale reads:

~~~text
User changes nickname from Alice to Bob
  -> Write to Primary succeeds
  -> User immediately refreshes page
  -> Read request routed to Replica
  -> Replica has not synced yet
  -> Page still shows Alice
~~~

Therefore, the core trade-off of primary-replica replication is:

~~~text
Increases read capacity and availability;
But read replicas may be briefly behind.
~~~

Common handling strategies:

| Scenario | Strategy |
| --- | --- |
| Read own writes immediately | Read-your-writes: Force read from primary for a short time |
| Brief stale data acceptable | Read from replica |
| Critical path requiring consistency | Read from primary after write, or use synchronous replication/quorum protocols |
| Replica lag is severe | Remove lagging replica from read pool |

---

## 2. Multi-Primary Replication: Understanding Write Topology, Not as Write Doubling

In primary-replica replication, the primary is the only write entry point. If the primary goes down, you need to elect a new primary, switch traffic, and handle the state of the old primary after recovery. Multi-primary replication makes both nodes writable:

~~~text
Primary A  <->  Primary B
~~~

Writes to A are replicated to B, and writes to B are replicated to A. This is usually used for active-active high availability.

~~~mermaid
flowchart LR
  C["Client"] --> A["Primary A"]
  C --> B["Primary B"]
  A -->|replicate changes| B
  B -->|replicate changes| A
~~~

### 2.1 How to Avoid Replication Loops

If A's writes are replicated to B, and B replicates them back to A, an infinite loop occurs. In MySQL replication, each server has a server-id, and the change log records the event source.

Therefore:

~~~text
A generates event e
  -> B receives e
  -> B sees e came from A
  -> B does not replicate e back to A as its own new event
~~~

### 2.2 Why It Does Not Equal Write Scaling

Multi-primary replication appears to have two primaries, but it does not mean write capacity is doubled. The reason is that both nodes must eventually store the complete dataset and execute writes replicated from the other.

Write to A:

~~~text
Client -> A
A executes write
A writes binlog
B pulls and replays
~~~

Write to B:

~~~text
Client -> B
B executes write
B writes binlog
A pulls and replays
~~~

Ultimately, both A and B must bear the full data, full indexes, full storage, and replication writes. Therefore, multi-primary replication is better understood as a high-availability solution, not a horizontal write-scaling solution.

### 2.3 Costs of Multi-Primary Replication

Multi-primary replication introduces:

~~~text
Both sides store complete data;
Both sides execute all writes;
Replication increases disk and network I/O;
Conflicts may occur if both sides write to the same row simultaneously;
Failover and old primary recovery are more complex.
~~~

If the business truly needs write scaling, it usually requires data partitioning, not just two masters.

---

## 3. Data Partitioning: Using Sharding to Scale Capacity and Writes

Replication saves multiple copies of the same data; partitioning saves different data.

~~~text
Replication: Every machine has a complete copy
Sharding: Every machine only saves a portion of the data
~~~

If the data in a single database is too large, or the write pressure on a single primary is too high, you need sharding.

### 3.1 Basic Idea

Assume the user table is divided into 4 shards based on user_id:

~~~text
user_id % 4 = 0  ->  Shard 0
user_id % 4 = 1  ->  Shard 1
user_id % 4 = 2  ->  Shard 2
user_id % 4 = 3  ->  Shard 3
~~~

Access user_id = 123:

~~~text
123 % 4 = 3
Access Shard 3
~~~

This way, each machine is only responsible for a portion of users, and capacity and write pressure are distributed.

### 3.2 Sharding Key is the Core of Design

A good sharding key needs to satisfy three things.

First, it is frequently included in queries.

If most queries are:

~~~sql
SELECT * FROM orders WHERE user_id = ?
~~~

Then user_id is natural. If the sharding key rarely appears in queries, you must broadcast to all shards.

Second, distribution should be uniform.

Sharding by country might cause the US shard to overheat; hash(user_id) is usually more uniform.

Third, minimize cross-shard queries.

Ideal query:

~~~text
Locate shard -> Query -> Return
~~~

Cross-shard query:

~~~mermaid
flowchart TD
  Q["Query"] --> S0["Shard 0"]
  Q --> S1["Shard 1"]
  Q --> S2["Shard 2"]
  Q --> S3["Shard 3"]
  S0 --> M["Merge / Sort / Aggregate"]
  S1 --> M
  S2 --> M
  S3 --> M
  M --> R["Response"]
~~~

Cross-shard joins, cross-shard transactions, and global sorting become significantly more complex.

### 3.3 Benefits and Costs of Sharding

Benefits:

~~~text
Data volume distributed across multiple machines;
Write pressure distributed across multiple shards;
Each node maintains smaller indexes;
Single-node storage and memory pressure decrease;
Can scale by adding shards.
~~~

Costs:

~~~text
Cross-shard queries are complex;
Cross-shard transactions are complex;
Global unique IDs need design;
Scaling and data migration are difficult;
Application layer must handle routing, retries, and partial failures.
~~~

In a nutshell:

~~~text
Replication gives the same data more copies;
Sharding lets different machines handle different data.
~~~

---

## 4. Replication and Sharding are Usually Used Together

In real systems, primary-replica replication is often performed within each shard:

~~~mermaid
flowchart TD
  Router["Query Router"] --> S0P["Shard 0 Primary"]
  Router --> S1P["Shard 1 Primary"]
  Router --> S2P["Shard 2 Primary"]

  S0P --> S0R1["Shard 0 Replica"]
  S0P --> S0R2["Shard 0 Replica"]
  S1P --> S1R1["Shard 1 Replica"]
  S1P --> S1R2["Shard 1 Replica"]
  S2P --> S2R1["Shard 2 Replica"]
  S2P --> S2R2["Shard 2 Replica"]
~~~

This simultaneously achieves:

~~~text
Capacity and write scaling from sharding;
Read scaling, backups, and high availability from replication.
~~~

Summary:

~~~text
Replication solves read scaling, multiple copies, and high availability;
Sharding solves large data volume, high writes, and single-node capacity bottlenecks.
~~~

---

## 5. Corresponding Design in Feature Stores

A Feature Store can be understood as a distributed state system that provides online features for model serving.

During online prediction, models need not only the current request fields but also historical context, for example:

| Scenario | Required Features |
| --- | --- |
| Fraud Detection | User's transaction count in the last 5 minutes, number of users associated with a device, merchant chargeback rate |
| Recommendation | User's recent clicks, item exposure/click statistics, user-item interaction history |
| Advertising | User interests, advertiser budget status, real-time click-through rate |

These features cannot be calculated by scanning logs on the fly during a request; they are usually materialized into an Online Feature Store in advance.

### 5.1 Replication in Feature Stores

Analogous to databases:

| Database | Feature Store |
| --- | --- |
| Primary receives writes | Feature primary receives feature updates |
| Replica replicates data | Feature replica replicates feature state |
| Application reads replicas | Model serving reads replicas |

Write path:

~~~text
feature computation / materialization -> feature primary
~~~

Read path:

~~~text
model serving / feature service -> feature replicas
~~~

This is read/write splitting in feature systems: feature computation handles writes, and model serving handles reads.

### 5.2 Stale Features in Feature Stores

Databases have stale reads; Feature Stores have stale features.

~~~text
User just failed 5 consecutive payments;
Risk feature should increase;
Feature primary has updated;
Replica has not synced;
Model reads old feature from replica;
Risk is underestimated.
~~~

Therefore, freshness monitoring is required:

~~~text
Last feature update time;
How far behind the feature is;
Whether it exceeds the model's acceptable latency;
Which replicas are lagging.
~~~

Different features have different freshness requirements:

| Feature Type | Typical Freshness |
| --- | --- |
| Fraud short-window features | Seconds to tens of seconds |
| Recommendation behavior features | Minutes |
| Merchant long-term statistics | Hours |
| User profiles | Days |

### 5.3 Data Partitioning in Feature Stores

The sharding key for a Feature Store is usually an entity key:

~~~text
user_id
item_id
merchant_id
device_id
tenant_id
session_id
~~~

For example:

| Feature group | Sharding key |
| --- | --- |
| user_features | user_id |
| item_features | item_id |
| merchant_features | merchant_id |
| device_features | device_id |
| user_item_features | hash(user_id, item_id) |

A fraud detection request might need:

~~~text
user_features:user_id=123
merchant_features:merchant_id=888
device_features:device_id=abc
user_merchant_features:user_id=123,merchant_id=888
~~~

The Feature Service must be able to locate the shard directly based on the key in the request; it cannot broadcast to all nodes every time.

### 5.4 Costs of Feature Store Sharding

First, cross-entity features are not suitable for temporary online aggregation.

For example:

~~~text
Average transaction amount of all users in a city in the last hour;
Overall click-through rate of a category in the last 30 minutes;
Payment failure rate of the entire site in the last 10 minutes.
~~~

These are usually calculated in advance via batch or streaming jobs and then written back to the Online Feature Store.

Second, hot keys lead to load imbalance:

~~~text
Super popular items;
Large merchants;
Huge enterprise clients;
Highly active users.
~~~

The essence of hot keys is: the sharding rule might be uniform, but access traffic is not. A popular item, a large merchant, or a super-active user might overwhelm a specific shard or replica.

Handling methods can start by separating the read path and the update path.

On the read path, common methods are:

~~~text
Increase replicas;
Add caching;
Special splitting for hot keys;
Pre-load popular item features into model serving local cache;
Batch and merge multiple reads.
~~~

On the update path, you can discuss push and pull modes in an interview.

| Mode | How it works | Suitable Scenario | Cost |
| --- | --- | --- | --- |
| Push / active update | Upstream feature computation pushes new values to cache / serving replica / local cache | Few hotspots, controllable update frequency, high freshness requirements | Write amplification; requires fanout, versioning, and failure retries |
| Pull / lazy update | Serving side pulls and refreshes local cache when a feature is missing or expired | Rapidly changing hotspots, many long-tail keys, allows brief staleness | First miss is slow; requires TTL, singleflight, prevents cache stampede |

You can understand it this way:

~~~mermaid
flowchart LR
  A["Feature Update"] --> B{"Update Mode"}
  B -->|Push active| C["Proactively refresh hot cache / replica"]
  B -->|Pull lazy| D["Refresh when request misses / TTL expires"]
  C --> E["Low read latency + higher write amplification"]
  D --> F["Low write amplification + higher latency on miss"]
~~~

In recommendation systems, popular item features are often suitable for pushing to serving local cache; user long-tail features are more suitable for pull + TTL, because proactively pushing all user features would cause massive invalid writes.

Third, multiple types of features come from different shards, and update times may differ. The model usually receives an approximately consistent feature snapshot, not a strictly global state at the same moment.

### 5.5 Update Logs and Checkpoints

MySQL primary-replica replication relies on binlog position. Feature Stores have similar incremental sync positions:

~~~text
Streaming job processed to Kafka offset X;
Batch materialization processed to a certain time partition;
Feature group v7 synced to checkpoint Y.
~~~

The update link can be recorded horizontally:

~~~mermaid
flowchart LR
  A["Raw Events"] --> B["Feature Computation"]
  B --> C["Feature Update Log / Checkpoint"]
  C --> D["Online Feature Store Primary"]
  D --> E["Online Feature Store Replicas"]
  E --> F["Model Serving"]
~~~

This update log / checkpoint is conceptually similar to a binlog position in a database: it is not a full sync every time, but records processing progress for continuous incremental updates.

---

## 6. Interview Answer Template

If asked "How to scale a database," answer in this order:

~~~text
1. Identify the bottleneck: read-heavy, write-heavy, large data, or availability issues.
2. Read-heavy: Primary-replica replication + read/write splitting, but handle replication lag.
3. Primary failure recovery: Primary-standby or multi-primary, focus on failover, not write scaling.
4. Write-heavy or large data: Choose a sharding key based on business access patterns.
5. After sharding, discuss cross-shard queries, transactions, global IDs, migration, and hotspots.
6. Real systems usually use replication within shards, combining replication and sharding.
~~~

When expanding, cover these levels:

| Level | What to clarify | Common follow-up |
| --- | --- | --- |
| Load estimation | Entry QPS, read/write ratio, peak factor, DB/Cache hits per request | What is the difference between avg QPS and peak QPS? |
| Replication | Primary-replica scales reads, primary-standby/multi-primary solves failover | How to handle replication lag? |
| Sharding | Sharding key close to main query path, avoid broadcast queries | How to handle hot shards, cross-shard joins, global IDs? |
| Consistency | Which reads can be stale, which must be read-your-writes | Can order, payment, inventory paths read from replicas? |
| Operations | Rebalancing, backup, schema migration, observability | How to migrate data when adding shards? |

If the topic is Feature Store / Online KV / Embedding Store:

~~~text
1. Treat it as a distributed state system serving models.
2. Entity key determines sharding.
3. Replicas handle low-latency reads and high availability.
4. Freshness is equivalent to replication lag in ML systems.
5. Hot keys require splitting read and update paths: cache/replica/push/pull.
6. Update log / checkpoint determines incremental sync and fault recovery capability.
~~~

In an interview, don't just say "add a cache." A better phrasing is:

~~~text
I would first estimate read/write pressure and data scale.
If it's primarily read pressure, I'd use replicas and caching;
If it's primarily write or capacity pressure, I'd use sharding;
If it's an availability issue, I'd use failover and replication;
If it's a Feature Store, I'd additionally discuss freshness, hot keys, update logs, and checkpoints.
~~~

### 6.1 Multiple Choice: What to think of first for read scaling?

~~~quiz
title: Database Scaling Check 1
question: For a service with many read requests and relatively few write requests, what is the most direct database scaling method?
answer: B
A. Multi-primary replication, because two primaries can double all write throughput
B. Primary-replica replication with read/write splitting, letting multiple replicas handle read requests
C. Immediately shard by a random key without considering query paths
D. Change all requests to an asynchronous queue
explanation: When reads are heavy and writes are light, primary-replica replication and read/write splitting are usually the first step; it scales read capacity but not single-primary write capacity.
~~~

### 6.2 Multiple Choice: How to analyze Feature Store hot keys?

~~~quiz
title: Feature Store Check 1
question: A popular item's feature is read by a large number of requests, while the feature update is not frequent. Which statement is more reasonable?
answer: C
A. Must re-hash the item to another shard by item_id
B. Must read from the source of truth for every request to avoid staleness
C. Can push popular item features to serving local cache or more replicas to reduce read path pressure
D. Hot keys only affect writes, not reads
explanation: The problem with popular items is mainly concentrated read traffic. If freshness requirements are high and the hotspot set is small, push/active updates to local cache or replicas are very suitable.
~~~

### 6.3 Multiple Choice: What do update logs / checkpoints solve?

~~~quiz
title: Feature Store Check 2
question: What is the core function of the update log / checkpoint in a Feature Store?
answer: B
A. Let every sync re-scan all historical data
B. Record processing progress, support incremental updates, fault recovery, and determine replica lag
C. Replace the sharding key so queries don't need routing
D. Ensure all features are updated at strictly the same moment
explanation: The update log / checkpoint is similar to a binlog position or Kafka offset; its core is to record how much has been processed, thereby enabling continuous incremental sync and recovery.
~~~

## 7. Short Memory Version

~~~text
Primary-Replica Replication:
  Read scaling + backup + disaster recovery
  Cost is stale reads

Multi-Primary Replication:
  High availability + fast switching
  Not write doubling

Sharding:
  Capacity scaling + write scaling
  Cost is cross-shard complexity

Feature Store:
  Reuse of traditional database scaling ideas for ML online state
~~~

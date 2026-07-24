# System Design 99 · Glossary of High-Frequency Terms

Course location: [[SystemDesign09 Consistent Hashing|09 Consistent Hashing]] → This article; for the course entry point, see [[SystemDesign00 Overview|00 Overview]].

This page is placed at the end of the System Design course as a quick-reference guide for looking up terms and writing design documents. The goal is not to memorize these terms, but to understand what problem each term solves, when it is worth introducing, and what the trade-offs are.

## I. Scalability and Capacity

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| scale up | Vertical Scaling | Upgrading to stronger hardware (e.g., more CPU, RAM, disk) | Early stage we can scale up the DB, but eventually we need to scale out. |
| scale out | Horizontal Scaling | Adding more machines to distribute traffic | The API servers should be stateless so we can scale out behind a load balancer. |
| horizontal scaling | Horizontal Scaling | Same as scale out; more common for service, cache, and worker layers | Workers can be horizontally scaled based on queue depth. |
| vertical scaling | Vertical Scaling | Same as scale up; more common for early-stage DB expansion | Vertical scaling buys time, but it has a ceiling. |
| stateless service | Stateless Service | Service instances that do not store non-recoverable state | Make API servers stateless; keep sessions and business state outside the process. |
| stateful service | Stateful Service | Services that hold critical state, such as DB, Kafka, or Redis clusters | Stateful components need replication, backup, and recovery plans. |
| load balancing | Load Balancing | Distributing requests across multiple instances | Put a load balancer in front of the API servers. |
| autoscaling | Autoscaling | Adding/removing instances based on CPU, QPS, latency, or queue depth | Worker pools can autoscale based on queue depth. |
| capacity planning | Capacity Planning | Estimating QPS, storage, bandwidth, peaks, and growth | Plan for peak QPS, not just average QPS. |
| bottleneck | Bottleneck | The part of the system that fails first under load | The bottleneck is likely the database write path. |

## II. Performance Metrics

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| QPS / RPS | Queries/Requests Per Second | Online service throughput | We need to support 100k QPS at peak. |
| TPS | Transactions Per Second | Throughput for DBs, payments, and transaction systems | Payment TPS is much lower than read QPS. |
| throughput | Throughput | Number of requests or data processed per unit of time | Batching improves throughput but may hurt latency. |
| latency | Latency | Time taken for a single request from entry to return | The p99 latency target is under 200 ms. |
| p95 / p99 latency | Percentile Latency | The upper bound of latency for 95% / 99% of requests | Average latency hides tail behavior. |
| tail latency | Tail Latency | Latency of the slowest requests that degrade user experience | Fan-out queries amplify tail latency. |
| availability | Availability | The percentage of time a service responds correctly | The target is 99.9% availability. |

A common estimation formula:

```text
concurrency ~= QPS * average latency in seconds
```

If the QPS is 10,000 and the average latency is 100 ms, there are approximately 1,000 in-flight requests in the system on average.

## III. Data Architecture

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| hot/cold data separation | Hot/Cold Data Separation | Storing high-frequency recent data and low-frequency historical data separately | Recent messages stay in hot storage; old messages move to cold storage. |
| data tiering | Data Tiering | Multi-level storage (hot / warm / cold) | Use data tiering to control storage cost. |
| read-write separation | Read-Write Separation | Writing to the primary DB and reading from replicas | Serve read-heavy traffic from replicas, but account for replication lag. |
| compute-storage separation | Compute-Storage Separation | Scaling compute and storage layers independently | Compute-storage separation lets query workers scale independently. |
| sharding | Sharding | Splitting data across nodes by user_id, region, time, etc. | Shard by user_id to distribute writes. |
| partitioning | Partitioning | Splitting large tables by range, hash, or list | Partition event tables by time. |
| hot partition / hot shard | Hot Partition / Hot Shard | A partition receiving significantly more traffic than others | Celebrity users can create hot partitions. |
| consistent hashing | [[SystemDesign09 Consistent Hashing|Consistent Hashing]] | Minimizing data migration when adding/removing nodes | Consistent hashing is useful for cache routing. |
| resharding | Resharding | Data migration after changing sharding rules | We need an online resharding plan. |
| data locality | Data Locality | Keeping related data physically close | Co-locate related data to reduce cross-node reads. |
| denormalization | Denormalization | Redundant storage to improve read performance | Denormalize user name into posts to avoid a join on the hot path. |
| materialized view | Materialized View | Pre-computed query results | Use materialized views for expensive aggregations. |
| secondary index | Secondary Index | Indexes for queries other than the primary key | Searching by email needs a secondary index. |
| inverted index | Inverted Index | Index mapping terms to documents | Full-text search relies on an inverted index. |
| LSM tree | Log-Structured Merge-Tree | Write-optimized storage structure (e.g., RocksDB, LevelDB, Cassandra) | LSM trees trade background compaction for high write throughput. |
| B-tree / B+ tree | B-tree / B+ tree | Traditional DB index structures with balanced read/write performance | InnoDB indexes are based on B+ trees. |
| WAL | Write-Ahead Log | Writing logs before modifying data to ensure crash recovery | WAL provides durability before pages are flushed. |
| compaction | Compaction | Cleaning up old versions and merging files (common in LSM/Kafka) | Compaction trades background IO for read efficiency. |

## IV. Consistency and Replication

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| strong consistency | Strong Consistency | Reads always return the latest write | Payments and inventory usually need strong consistency. |
| eventual consistency | Eventual Consistency | Inconsistent for a short time, but converges eventually | Feeds and counters often tolerate eventual consistency. |
| read-your-writes consistency | Read-Your-Writes | Users can immediately read their own writes | Users should see their own post immediately. |
| monotonic reads | Monotonic Reads | A user will not see an older value after seeing a newer one | Sticky reads can help preserve monotonic reads. |
| quorum | Quorum | Operations succeed only after a majority of replicas acknowledge | Quorum writes balance consistency and availability. |
| leader-follower replication | Leader-Follower Replication | Leader handles writes; followers replicate and can serve reads | Followers can serve reads when stale reads are acceptable. |
| primary-replica | Primary-Replica | Same as leader-follower | Design docs often use primary / replica terminology. |
| sync replication | Synchronous Replication | Writes wait for replica acknowledgment | Sync replication improves durability but increases write latency. |
| async replication | Asynchronous Replication | Primary returns immediately; replicas sync later | Read replicas are usually asynchronously replicated. |
| replication lag | Replication Lag | Time delay between primary and replica updates | Read-after-write can fail because of replication lag. |
| consensus | Consensus | Multiple nodes agreeing on a single value | Consensus is needed for leader election and metadata consistency. |
| Raft / Paxos | Consensus Algorithms | Used for metadata, leader election, and configuration changes | Use Raft-backed metadata when split brain is unacceptable. |
| CAP theorem | CAP Theorem | Trade-off between consistency and availability during network partitions | CAP is a framing tool, not a full design answer. |
| ACID | ACID Properties | Atomicity, Consistency, Isolation, Durability | Orders, payments, and ledgers usually need ACID semantics. |
| BASE | BASE | Basically Available, Soft state, Eventual consistency (common in NoSQL) | BASE systems push more correctness logic into the application. |

## V. Write Semantics and Message Processing

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| idempotency | Idempotency | Retrying multiple times has the same effect as once | Payment APIs need idempotency keys. |
| deduplication | Deduplication | Avoiding duplicate writes from duplicate messages | Consumers deduplicate events by event_id. |
| at-most-once | At-Most-Once | May be lost, but never duplicated | At-most-once is acceptable for low-value telemetry. |
| at-least-once | At-Least-Once | Never lost, but may be duplicated | Most queues provide at-least-once delivery. |
| exactly-once | Exactly-Once | Usually achieved via transactions, idempotency, and deduplication | Say effectively exactly once unless the system truly guarantees it. |
| DLQ / Dead Letter Queue | Dead Letter Queue | Queue for messages that failed processing multiple times | Failed messages go to a DLQ for inspection and replay. |
| replay | Replay | Reprocessing historical messages or events | Kafka allows replay from an offset. |
| offset | Offset | The position of a consumer in a message stream | Store offsets only after processing succeeds. |
| checkpoint | Checkpoint | Saving progress or state for recovery | Stream jobs checkpoint state and offsets. |

## VI. Distributed Transactions

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| 2PC / Two-Phase Commit | Two-Phase Commit | Strong consistency protocol across services or databases | 2PC gives strong semantics but can block and increase latency. |
| Saga pattern | Saga Pattern | Breaking large transactions into local transactions with compensation | Use Saga for order-payment-inventory workflows. |
| compensating transaction | Compensating Transaction | Corrective action taken after a step fails | If inventory reservation fails after payment, issue a refund. |
| outbox pattern | Outbox Pattern | Writing to a local DB and an outbox table, then sending messages asynchronously | Outbox avoids the DB-write-succeeded-but-message-lost problem. |
| CDC / Change Data Capture | Change Data Capture | Capturing binlogs or change logs to push to downstream | CDC keeps search indexes and analytics stores in sync. |

## VII. Caching

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| cache-aside | Cache-Aside | App checks cache; on miss, fetches from DB and backfills | Cache-aside is the default pattern for read-heavy endpoints. |
| lazy loading | Lazy Loading | Loading data into cache only when needed | Lazy loading avoids caching data nobody reads. |
| write-through cache | Write-Through Cache | Writing to cache and DB synchronously | Write-through improves consistency at the cost of write latency. |
| write-back cache | Write-Back Cache | Writing to cache first, then flushing to DB asynchronously | Write-back is fast but riskier during failures. |
| TTL | Time To Live | Expiration time for cache entries | Add TTL jitter to avoid synchronized expiration. |
| cache invalidation | Cache Invalidation | Deleting or updating cache after data changes | Invalidation is where most cache bugs happen. |
| cache penetration | Cache Penetration | Querying non-existent data, hitting the DB every time | Use negative caching or a Bloom filter. |
| negative caching | Negative Caching | Caching "not found" results for a short time | Cache not-found results briefly. |
| cache breakdown | Cache Breakdown | Hot key expires, causing a surge of requests to the DB | Use singleflight or early refresh for hot keys. |
| cache avalanche | Cache Avalanche | Many keys expire simultaneously or the cache cluster fails | Randomize TTLs and use multi-level fallback. |
| thundering herd / cache stampede | Thundering Herd / Cache Stampede | Many requests trying to rebuild the same cache simultaneously | Coalesce requests for the same key. |
| singleflight / request coalescing | Request Coalescing | Only one request hits the DB for a key; others wait | Singleflight protects the backend on cache miss. |
| stale-while-revalidate | Stale-While-Revalidate | Returning stale data while refreshing in the background | CDN and feed systems often use stale-while-revalidate. |
| CDN | Content Delivery Network | Caching static assets at edge nodes | Use CDN for images, video, JS, and CSS. |
| edge cache | Edge Cache | Caching closer to the user | Edge cache reduces global latency. |

## VIII. Traffic Control

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| buffering | Buffering / Load Leveling | Using a queue to absorb spikes and process slowly | Use a queue as a buffer to smooth traffic spikes. |
| load leveling | Load Leveling | Architectural expression of smoothing traffic | Queue-based load leveling protects downstream systems. |
| traffic shaping | Traffic Shaping | Controlling the speed and shape of incoming traffic | Gateways can shape traffic before it reaches the service. |
| rate limiting | Rate Limiting | Limiting requests per unit of time per user/service | Rate limit by user_id, tenant_id, and IP. |
| token bucket | Token Bucket | Rate limiting algorithm allowing bursts | Token bucket allows short bursts while enforcing long-term rate. |
| leaky bucket | Leaky Bucket | Processing requests at a fixed rate | Leaky bucket smooths traffic more aggressively. |
| backpressure | Backpressure | Notifying upstream to slow down when downstream is overloaded | Apply backpressure when queue length grows. |
| load shedding | Load Shedding | Proactively dropping low-priority requests under load | Shed non-critical traffic to protect core flows. |
| priority queue | Priority Queue | Processing high-priority tasks first | Payment jobs should outrank notification jobs. |
| admission control | Admission Control | Deciding whether to accept a request before it enters the system | Admission control prevents overload before work is admitted. |

## IX. Resilient Design

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| timeout | Timeout | Not waiting indefinitely for external services | Every RPC should have a timeout. |
| retry | Retry | Requesting again after a temporary failure | Retries must be paired with idempotency. |
| exponential backoff | Exponential Backoff | Increasing retry intervals gradually | Backoff prevents retry storms. |
| jitter | Jitter | Adding randomness to retries or TTLs | Add jitter so clients do not retry at the same time. |
| circuit breaker | Circuit Breaker | Stopping calls to a failing downstream service temporarily | Open the circuit to avoid cascading failures. |
| graceful degradation | Graceful Degradation | Maintaining core functionality when non-core features fail | If recommendations fail, return a default ranking. |
| fallback | Fallback | Alternative result when the main path fails | Fallback to cached data. |
| bulkhead isolation | Bulkhead Isolation | Isolating resource pools to prevent cross-contamination | Separate thread pools for payment and analytics. |
| cascading failure | Cascading Failure | One service failure causing a chain reaction | Timeouts, circuit breakers, and isolation all reduce cascading failures. |

## X. Asynchronous Architecture

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| message queue | Message Queue | Decoupling producers and consumers | Queue writes to decouple the API from slow downstream processing. |
| pub/sub | Pub/Sub | One event subscribed to by multiple consumers | UserCreated can feed email, analytics, and recommendation systems. |
| event-driven architecture | Event-Driven Architecture | Services communicating via events | Event-driven design reduces direct service coupling. |
| stream processing | Stream Processing | Real-time processing of continuous event streams | Use Flink or Kafka Streams for real-time aggregates. |
| batch processing | Batch Processing | Scheduled processing of large historical datasets | Reports and offline features are usually batch processed. |
| event sourcing | Event Sourcing | Storing all state-changing events | Event sourcing helps with auditability. |
| fan-out | Fan-out | Distributing one request/event to multiple downstream targets | Feed and notification systems often fan out writes. |
| fan-in | Fan-in | Aggregating multiple results into one | Search services fan in results from multiple shards. |

## XI. Availability and Disaster Recovery

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| SPOF / single point of failure | Single Point of Failure | A component whose failure brings down the whole system | The primary DB is a potential SPOF. |
| HA / high availability | High Availability | System remains functional after a single-point failure | HA needs replicas, health checks, and failover. |
| DR / disaster recovery | Disaster Recovery | Recovery capability after a data center or region failure | DR planning covers region-level outages. |
| failover | Failover | Switching to a standby node when the primary fails | Automatic failover reduces downtime. |
| active-active | Active-Active | Multiple regions serving traffic simultaneously | Active-active is powerful but makes conflict handling harder. |
| active-passive | Active-Passive | Primary region serves traffic; standby waits to take over | Active-passive is simpler but uses resources less efficiently. |
| multi-AZ | Multi-AZ | Deployment across multiple Availability Zones in one region | Multi-AZ protects against single-AZ failure. |
| multi-region | Multi-Region | Deployment across different geographic regions | Multi-region protects against region-level outages. |
| RPO | Recovery Point Objective | Maximum tolerable data loss | RPO is 5 minutes. |
| RTO | Recovery Time Objective | Maximum tolerable time to restore service | RTO is under 30 minutes. |
| backup / restore | Backup / Restore | Periodic backups and verified restoration | A backup is not useful until restore has been tested. |
| health check | Health Check | Determining if a service can accept traffic | Load balancers rely on health checks. |
| liveness probe | Liveness Probe | Determining if a process needs a restart | Kubernetes uses liveness probes to restart stuck containers. |
| readiness probe | Readiness Probe | Determining if an instance is ready to accept traffic | Readiness prevents traffic from reaching a cold-starting pod. |

## XII. Release and Deployment

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| canary release | Canary Release | Releasing to a small subset of traffic first | Canary 1% traffic before full rollout. |
| blue-green deployment | Blue-Green Deployment | Switching traffic between two identical environments | Blue-green makes rollback fast but costs more capacity. |
| rollback | Rollback | Reverting to the old version when the new one fails | Keep rollback fast for risky releases. |
| roll-forward | Roll-Forward | Fixing and deploying a new version instead of rolling back | Roll-forward is common when schema changes are hard to undo. |
| feature flag | Feature Flag | Toggling features without redeployment | Feature flags make risky features easier to disable. |
| dark launch | Dark Launch | Running new logic in the backend without exposing it to users | Dark launch validates performance before user exposure. |
| shadow traffic | Shadow Traffic | Replicating production traffic to a new system for testing | Shadow traffic helps compare a new service with production behavior. |
| A/B testing | A/B Testing | Comparing metrics by showing different versions to different users | A/B testing measures product impact. |
| schema migration | Schema Migration | Changing the database schema | Schema migration must be compatible with old and new code. |
| backfill | Backfill | Populating new fields or indexes for historical data | Run backfill asynchronously. |

## XIII. Observability

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| observability | Observability | Understanding system state via metrics, logs, and traces | Observability starts with metrics, logs, and traces. |
| metrics | Metrics | Numerical data like QPS, latency, error rate, CPU | Metrics power dashboards and alerts. |
| logs | Logs | Recording event and error context | Logs should include request_id and useful context. |
| tracing | Tracing | Tracking request paths and latency across services | Distributed tracing helps debug slow microservice requests. |
| SLI | Service Level Indicator | Metric measuring service quality | Latency and availability are common SLIs. |
| SLO | Service Level Objective | Internal target for service quality | 99.9% of requests under 200 ms is an SLO. |
| SLA | Service Level Agreement | External contractual commitment | SLA is external and contractual. |
| error budget | Error Budget | Allowed failure threshold based on SLO | Error budget guides whether to ship features or fix reliability. |
| alerting | Alerting | Notifying on-call engineers of anomalies | Alerts should be actionable. |
| runbook | Runbook | Instructions for handling incidents | Critical alerts should link to a runbook. |

## XIV. API and Security

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| API gateway | API Gateway | Unified entry point for auth, rate limiting, and routing | Put authentication and rate limiting at the gateway. |
| pagination | Pagination | Returning large result sets in batches | Always paginate large list APIs. |
| cursor-based pagination | Cursor-based Pagination | Using a cursor to fetch the next page | Cursor pagination is better for feeds than offset pagination. |
| versioning | API Versioning | Managing v1/v2 compatibility | API versioning prevents breaking old clients. |
| backward compatibility | Backward Compatibility | Ensuring new versions don't break old clients | Keep response fields backward compatible. |
| contract | API Contract | Agreed-upon input/output between services | Define a clear API contract between services. |
| authentication / AuthN | Authentication | Verifying who you are | AuthN verifies identity. |
| authorization / AuthZ | Authorization | Verifying what you can do | AuthZ checks permissions. |
| RBAC | Role-Based Access Control | Permissions based on roles (admin/user/viewer) | RBAC works well for coarse-grained enterprise permissions. |
| ABAC | Attribute-Based Access Control | Permissions based on user, resource, and environment attributes | ABAC supports more fine-grained policies. |
| JWT | JSON Web Token | Common stateless token format | JWT is stateless, but revocation and key rotation need design. |
| OAuth | OAuth | Protocol for third-party authorization | OAuth is used for "login with Google" style flows. |
| mTLS | Mutual TLS | Mutual authentication between services | mTLS is common in service mesh setups. |
| encryption in transit | Encryption in Transit | Encrypting data over the network using TLS | Use TLS for data in transit. |
| encryption at rest | Encryption at Rest | Encrypting data on disk or in the database | Encrypt sensitive data at rest. |
| PII | Personally Identifiable Information | Email, phone, SSN, address, etc. | PII needs access control, masking, and audit logs. |
| data masking | Data Masking | Hiding sensitive fields | Mask PII in logs and admin tools. |
| audit log | Audit Log | Recording who did what and when | Audit logs are required for sensitive operations. |

## XV. Trade-offs

| Term | Chinese | Used to describe | Natural Usage |
| --- | --- | --- | --- |
| trade-off | Trade-off | Inability to optimize all metrics simultaneously | This is a latency-consistency trade-off. |
| cost-performance trade-off | Cost-Performance Trade-off | Faster usually means more expensive | Caching improves latency but adds consistency complexity. |
| complexity budget | Complexity Budget | Avoiding over-engineering for small-scale problems | At this scale, sharding is premature. |
| over-engineering | Over-engineering | Introducing complex systems for non-existent problems | Avoid over-engineering before measuring the bottleneck. |
| premature optimization | Premature Optimization | Optimizing before a bottleneck is identified | Start simple and optimize after measuring. |

## XVI. Common Expressions

### Load Leveling

```text
Use a queue as a buffer to smooth traffic spikes and decouple writes from downstream processing.
```

Can also be written as:

```text
Use queue-based buffering / load leveling to absorb peak traffic.
```

### Hot/Cold Data Separation

```text
Keep recent, frequently accessed data in hot storage, and move old data to cold storage to reduce cost.
```

### Read-Write Separation

```text
Route writes to the primary database and serve read-heavy traffic from read replicas, while accounting for replication lag.
```

### Compute-Storage Separation

```text
Separate compute from storage so each layer can scale independently.
```

### Consumer Design for At-Least-Once Delivery

```text
Since the queue provides at-least-once delivery, consumers must be idempotent and deduplicate events by event_id.
```

### Overload Protection

```text
Apply rate limiting at the edge, backpressure between services, circuit breakers for unhealthy dependencies, and graceful degradation for non-critical features.
```

## XVII. Questions to Ask When Using These Terms

```text
Does this mechanism solve a read bottleneck, write bottleneck, capacity bottleneck, latency issue, or availability issue?
Where does it shift the complexity?
What new problems will it introduce?
Is it truly necessary at the current scale?
```

Terminology itself adds no value; it is only useful if you can connect the mechanism, the bottleneck, and the trade-offs.

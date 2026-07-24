# System Design 02 · Database Fundamentals: Paradigms

Course Location: [[SystemDesign01B Virtualization Containers|01B Virtualization and Containers]] → This Section → [[SystemDesign03 Database Scaling|03 Database Scaling]]

When choosing a database, consider two things first: which business invariants must be atomically satisfied, and what are the system's most critical access paths? The product name comes later.

```text
transaction boundary -> correctness
access pattern        -> data layout and indexes
```

Claims like "SQL cannot scale" or "NoSQL has no transactions" are too simplistic. Modern products have overlapping capabilities; the differences lie in their default data models, transaction boundaries, and scaling costs.

---

## 1 · RDBMS: Expressing Relationships and Constraints First

The relational model organizes data into rows and tables, expressing invariants through primary keys, foreign keys, unique constraints, and transactions.

```sql
BEGIN;

UPDATE accounts
SET balance = balance - 100
WHERE account_id = 1 AND balance >= 100;

UPDATE accounts
SET balance = balance + 100
WHERE account_id = 2;

COMMIT;
```

The point of this code is not the SQL syntax, but that the two balance changes belong to the same commit boundary. If any step fails, the entire transfer must not leave behind a partial state.

RDBMS is suitable when:

- Relationships between entities are dense;
- Invariants frequently span across rows or tables;
- Query patterns are diverse and subject to future change;
- Mature secondary indexes, joins, and ad-hoc queries are required.

The costs are also direct: cross-node transactions, joins, and global constraints are difficult to scale as the number of shards increases.

---

## 2 · NoSQL: Organizing Data Around Access Patterns First

NoSQL is not a single type of database. KV, document, wide-column, and graph models differ, but many systems share an emphasis on partition-local access.

```text
GetUser(user_id)
ListOrders(user_id, created_at range)
GetFeed(viewer_id, cursor)
```

When modeling, first select the partition key and sort key for these reads. To ensure a single request hits one partition, data may be denormalized:

```json
{
  "user_id": "u42",
  "profile": {"name": "Kai"},
  "shipping_city": "Seattle"
}
```

When a city is renamed, multiple documents may need to be updated. Reads become simpler, but the costs of writes and consistency increase.

Common advantages of NoSQL:

- Simple single-key / single-partition paths;
- Data can be horizontally distributed by partition;
- Schema is more flexible for sparse or evolving fields;
- Latency and throughput are easier to plan around fixed access patterns.

It is not a tool to avoid modeling. If the partition key is chosen incorrectly, hot keys, scans, and cross-partition transactions will emerge.

---

## 3 · Transactions are the Boundaries of Business Invariants

ACID can be remembered as follows:

| Property | Practical Question |
|---|---|
| Atomicity | Could it be only half-finished? |
| Consistency | Does the constraint still hold after commit? |
| Isolation | What intermediate states will concurrent operations see? |
| Durability | Will the result disappear after a failure once success is returned? |

Define the invariants first, then determine the transaction scope:

```text
order.total == sum(order_items)
payment may be captured at most once
username must be unique
inventory cannot fall below zero
```

If these conditions must be satisfied atomically across multiple entities, a relational database or a distributed SQL system supporting such transactions is often less of a headache. If the process can be broken down into a state machine and accept eventual consistency, an event-driven workflow may also be appropriate.

### Do not mix database transactions with business workflows

A single-database transaction usually finishes within milliseconds. An order process spanning a payment provider, inventory, and shipping may last for minutes; you cannot hold a database connection open that long.

```text
local transaction
  -> write order + outbox
  -> async payment command
  -> state transition
  -> compensation when needed
```

Such processes maintain business consistency through idempotency, state machines, outboxes, and compensation. For message details, see [[SystemDesign06 Async Messaging Systems|06 Async Messaging Systems]].

---

## 4 · What Strong and Eventual Consistency Mean

Consistency must be bound to specific operations.

```text
User updates profile to v2
User immediately reads profile
```

Possible contracts:

- Linearizable read: Acts as if there is only one latest copy;
- Read-your-writes: The user can at least read their own v2;
- Monotonic read: Once v2 is seen, it will not revert to v1;
- Eventual consistency: Replicas eventually converge when there are no new writes.

A single system can mix these. Read the primary for order confirmation pages, read replicas for public product pages, and allow stale data for recommendation features. Rather than declaring a system "strongly consistent," it is more useful to clearly define the contract for each API.

For replication lag, failover, and RPO/RTO, see [[SystemDesign05 Reliability Replication|05 Reliability and Replication]].

---

## 5 · Six Questions to Ask During Selection

### 1. What is the basic unit of read/write?

```text
single key?
document?
partition + range?
graph traversal?
multi-row relation?
```

### 2. Can the most important queries be satisfied directly by primary keys or indexes?

If every feed requires a full table scan followed by filtering, changing the database name won't save it. Write out the queries and indexes first.

### 3. How wide is the transaction scope?

Costs increase progressively from single-row, single-document, and single-partition to cross-partition. Keep strong invariants within the same transaction boundary whenever possible.

### 4. How is data partitioned?

The partition key determines locality, parallelism, and hotspots. A low-cardinality country code is often not uniform enough; hash(user_id) is more balanced but makes regional scanning more difficult.

### 5. Which reads allow stale data?

Stale reads can utilize replicas, caches, and materialized views. Permissions, balances, and inventory deductions usually require more caution.

### 6. Where is the operational complexity placed?

The relational model places complexity in the database engine and query planner; the access-pattern-first model places more complexity in the application write path, denormalization, and data repair. There are no free options.

---

## 6 · Design Centers of Common Products

| System Type | Design Center | Common Use Case |
|---|---|---|
| PostgreSQL / MySQL | relation, constraint, transaction | order, account, metadata |
| Dynamo-style KV / document | partition key, predictable access path | profile, session, serving KV |
| Cassandra-style wide column | partition + clustering order, high write throughput | time series, event, timeline |
| MongoDB-style document | aggregated document, flexible fields | content, catalog, profile |
| Spanner-style distributed SQL | distributed transaction + SQL | global metadata, strongly consistent business data |
| Graph database | vertex / edge traversal | fraud graph, relationship exploration |

This table is only a starting point. Ultimately, you must check specific product versions, transaction scopes, indexes, region topologies, backups, and team operational capabilities.

---

## 7 · Typical Judgments

### Orders and Payments

Start with an RDBMS. Order status, amounts, idempotency keys, and accounting constraints require reliable transactions. Once the scale grows, offload search, analytics, and event streams.

### User Profile

If reads are primarily by user_id, a document/KV store is natural; if the profile is densely related to permissions, organizations, and billing, an RDBMS might be simpler.

### Feed Timeline

Partitioning by viewer_id and sorting by rank_key or time is suitable for wide-column / sorted KV stores. The timeline is a derived index; post metadata can still reside in a relational or document store.

### Logs and Events

High-throughput appends, time-based retention, and replays are more like a log system and should not be forced into an OLTP table. Move to a search or analytical store when querying is required.

### Financial Ledgers

Prioritize immutable entries, double-entry balancing, unique transaction IDs, and auditing. Do not sacrifice transaction semantics just because write volume is high.

---

## 8 · Data Can Be Split, But Don't Split Too Early

A mature system is often polyglot persistence:

```text
RDBMS          authoritative order metadata
Redis          hot cache
Object store   blobs
Event log      change propagation
Search index   text retrieval
Warehouse      analytics
```

This does not mean you need six systems on day one. Every additional store adds a new set of schemas, backups, permissions, monitoring, and data repair processes. Start with the simplest source of truth, and split only when access patterns or scale truly diverge.

---

## 9 · Interview Checklist

```text
Correctness
- Which invariants must be atomically satisfied?
- Is the transaction single-key, single-partition, or cross-entity?

Access
- What is the top read/write path?
- What are the indexes and partition keys?
- Are there any scans, hot keys, or cross-partition queries?

Consistency
- Which APIs require read-your-writes?
- Which derived data allows staleness, and for how long?

Operations
- How is data backed up and restored?
- How are schema / index changes deployed?
- Can the team operate the new store being introduced?

Growth
- Should we add indexes, caches, or replicas first, or is sharding already required?
- For sharding details, see 03 Database Scaling.
```

A one-sentence summary: Use transaction boundaries to protect correctness, then use access patterns to determine data layout. The database brand is a choice made after these two questions.

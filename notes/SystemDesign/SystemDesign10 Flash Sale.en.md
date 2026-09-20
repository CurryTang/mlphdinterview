# System Design 10 · Flash Sale

Course location: [[SystemDesign08 LLM Async RL Platform|08 Async LLM RL]] → this note → [[SystemDesign11 Notification System|11 Notification System]]

One sale, one SKU. Reads hit cache. Buys hit a queue. Stock moves only inside a DB transaction.

![[assets/flash-sale-whiteboard.png|Flash sale whiteboard]]

```flash-sale-architecture-visual
```

---

## 1. Functional requirements

1. View the sale: price and remaining stock.
2. Buy 1 item. At most one order per user per sale.
3. View own order. Pay within 5 minutes.

```text
Out of scope
multi-SKU, cart, payment reconciliation, fulfillment, refunds
```

---

## 2. Non-functional requirements

| | Target |
|---|---|
| Consistency | no oversell, no duplicate orders |
| Availability | 99.95% |
| Latency | admission p99 < 200 ms |

Size the edge for 200k request/s. Cache the reads. Bound the queue.

---

## 3. Workflow, schema, QPS

Starting path:

```text
Client → Gateway → Purchase Service → DB
```

After the three deep dives:

```text
view  Client → Gateway → Sale Service → DB / cache

buy   Client → Gateway (auth + rate limit)
        → Purchase Service
             Redis admission (fast reject / fast allow, see [[SystemDesign01D Redis|01D]])
             reject → immediate 4xx
             allow  → durable MQ ack → 202 + request_id
      Worker → DB txn → Redis (read model) → notify

202 does not reserve stock. Redis tokens are not stock.
```

| | Fields |
|---|---|
| sale | sale_id PK, item_id, price, currency, available_stock, start_at, end_at |
| order | order_id PK, request_id unique, sale_id, user_id, amount, status, expires_at, payment_id |

```text
One sale = one SKU
UNIQUE(sale_id, user_id)
stock-1 + create order + unique user: one transaction
request_id unique: retry of the same click
UNIQUE(sale_id, user_id): one order per user per sale
see [[SystemDesign01 Stateless Service|01]]
```

| | Number |
|---|---|
| Edge | 1M users / 10s × 2 attempts = 200k/s |
| Worker | ~2k attempts/s, from load test |
| Full queue | reject. MQ does not add DB write capacity |

---

## 4. Deep dives

### 1. Burst → availability / scale

A. Rate limit + synchronous DB  
+ Simple. Immediate result.  
- Overload is rejected. No burst buffer.

B. Rate limit + Kafka + Worker (chosen)  
+ Smooths bursts. Transient errors can retry.  
- Queue delay. Extra infrastructure.  
Change in the picture: MQ + Worker between Gateway and DB.

### 2. Data race → consistency

A. Pessimistic lock (chosen)  
+ Lock the sale row, check stock, create the order.  
- A hot sale waits on the lock.

B. Optimistic lock (version / CAS)  
+ No lock up front. Detect conflicts.  
- A hot row retries until the CPU melts.  

Both still need one transaction and UNIQUE(sale_id, user_id).

### 3. Result delivery → latency

A. Long polling (chosen)  
+ Ordinary HTTP. Cache the final result.  
- Held requests. Reconnect cost.

B. Pub/Sub → SSE / WebSocket  
+ Faster push. Fewer repeated reads.  
- Connection cost. Missed notifications.  

Both: on cache miss or reconnect, query the order through the API. Redis is a read model, not stock truth.

---

## 5. End-to-end

```text
1. View sale: Gateway → Sale Service → cache/DB
2. Buy: Gateway rate-limits, Purchase writes MQ, returns 202 + request_id
3. Same user, second device, same request_id or same (sale_id, user_id)
4. Worker transaction: lock sale row, stock-1, insert order
5. Duplicate request_id: return the existing order
6. Second user: UNIQUE(sale_id, user_id) rejects; stock=0 fails
7. Client long-polls Redis; miss goes to the API then DB
8. No pay in 5 minutes: expire the order, stock+1
```

# System Design 10 · 秒杀

课程位置：[[SystemDesign08 LLM Async RL Platform|08 异步 LLM RL]] → 本篇 → [[SystemDesign11 Notification System|11 移动推送与通知系统]]

一场活动、一个 SKU。读走 cache；买走队列；库存只在 DB 事务里扣。

![[assets/flash-sale-whiteboard.png|Flash sale whiteboard]]

```flash-sale-architecture-visual
```

---

## 1. Functional requirements

1. 查看活动：价格、剩余库存。
2. 购买 1 件；同一用户同一场最多 1 单。
3. 查看自己的订单；5 分钟内支付。

```text
Out of scope
多 SKU、购物车、支付渠道对账、履约、退款
```

---

## 2. Non-functional requirements

| | 目标 |
|---|---|
| Consistency | 不超卖、不重复下单 |
| Availability | 99.95% |
| Latency | 准入 p99 < 200 ms |

规模按 200k request/s 设计。读走 cache，队列有界。

---

## 3. Workflow, schema, QPS

起点是同步链路：

```text
Client → Gateway → Purchase Service → DB
```

三个 deep dive 之后变成：

```text
view  Client → Gateway → Sale Service → DB / cache

buy   Client → Gateway (auth + rate limit)
        → Purchase Service
             Redis 准入（fast reject / fast allow，见 [[SystemDesign01D Redis|01D]]）
             reject → 立刻 4xx
             allow  → durable MQ ack → 202 + request_id
      Worker → DB txn → Redis (read model) → notify

202 不预占库存。Redis 名额不是库存。
```

| | 字段 |
|---|---|
| sale | sale_id PK, item_id, price, currency, available_stock, start_at, end_at |
| order | order_id PK, request_id unique, sale_id, user_id, amount, status, expires_at, payment_id |

```text
One sale = one SKU
UNIQUE(sale_id, user_id)
stock-1 + create order + unique user：同一事务
request_id unique：同一点击重试
UNIQUE(sale_id, user_id)：每用户每场一单
见 [[SystemDesign01 Stateless Service|01]]
```

| | 数 |
|---|---|
| 边缘 | 1M users / 10s × 2 attempts = 200k/s |
| Worker | 约 2k attempts/s，压测定 |
| 队列满 | 拒绝。MQ 不增加 DB 写容量 |

---

## 4. Deep dives

### 1. Burst → availability / scale

A. Rate limit + 同步写 DB  
+ 简单，结果立刻返回。  
- 超载直接失败，没有缓冲。

B. Rate limit + Kafka + Worker（选）  
+ 削峰，瞬时错误可重试。  
- 多一段排队延迟，多一套基础设施。  
图上的变化：Gateway 和 DB 之间插入 MQ + Worker。

### 2. Data race → consistency

A. 悲观锁（选）  
+ 锁 sale 行，检查库存，建单。  
- 热门活动排队等锁。

B. 乐观锁（version / CAS）  
+ 不先锁，冲突再重试。  
- 热点行重试会打满。  

两种都要：一笔事务 + UNIQUE(sale_id, user_id)。

### 3. Result delivery → latency

A. 长轮询（选）  
+ 普通 HTTP；终态放 Redis。  
- 挂起连接；重连有开销。

B. Pub/Sub → SSE / WebSocket  
+ 推送快，少反复读。  
- 连接成本高；通知会丢。  

两种都要：cache miss / 重连时走 API 查 order。Redis 是读模型，不是库存真相。

---

## 5. End-to-end

```text
1. 查看活动：Gateway → Sale Service → cache/DB
2. 购买：Gateway 限流通过，Purchase 写入 MQ，返回 202 + request_id
3. 同一用户第二台设备再买，同一 request_id 或同一 (sale_id, user_id)
4. Worker 开事务：锁 sale 行，stock-1，插入 order
5. 重复 request_id：幂等返回已有订单
6. 第二个用户：UNIQUE(sale_id, user_id) 已有则拒绝；stock=0 则失败
7. 客户端长轮询 Redis；miss 则 API 查 DB
8. 5 分钟未支付：订单过期，stock+1
```

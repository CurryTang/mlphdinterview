# Wiki · Pub/Sub + Transactional Outbox (发布/订阅与事务发件箱模式)

所属类别：Wiki 模式库 · 知识点  
关联模式：[[SystemDesignWiki Event Bus|事件总线]] · [[SystemDesignWiki Kafka|Kafka 实战]] · [[SystemDesignWiki Message Queue|消息队列]] · [[SystemDesignWiki Idempotency|幂等性保障]] · [[SystemDesign11 Notification System|通知系统实战]]

---

## 1 · 核心痛点：双写困境与分布式协同失效 (Dual-Write Hazard)

在微服务与事件驱动架构（EDA）中，业务服务在处理领域状态变更时，往往伴随着跨系统的事件派发诉求（例如：订单服务完成扣款后，需驱动库存、物流、账单及通知等多个外部微服务协同）。

服务由于无法在**应用层**为“本地数据库”与“外部消息总线（Pub/Sub Broker）”建立全局原子事务，朴素的双写逻辑必然遭遇不可避免的时序故障窗口：

```text
[场景 A：先写 DB，后发 Pub/Sub]
  1. BEGIN TX -> UPDATE orders SET status = 'PAID' -> COMMIT TX (成功)
  2. 网络抖动 / MQ Broker 超时 / 进程 OOM Crash
  3. Pub/Sub 事件丢失 (Silent Event Loss)
  => 数据库记录已支付，但下游系统永久未收到通知，业务状态发生不可逆漂移。

[场景 B：先发 Pub/Sub，后写 DB]
  1. Publish Event('OrderPaid') 到 Pub/Sub Broker (成功，进入 Topic)
  2. 下游消费者立即收到事件并执行发货或扣库存
  3. 本地 DB 事务提交失败（唯一约束冲突 / 行锁死锁 / 数据库宕机）
  4. DB 回滚 (Rollback)
  => 下游处理了从未真实存在的“幽灵事件 (Phantom Event)”，产生严重脏数据。
```

### 1.1 · 分布式事务 (2PC / XA) 为何在现代 Pub/Sub 架构中失效？
1. **协议不支持**：业界主流高吞吐消息引擎（Kafka、AWS SNS、Pulsar、RabbitMQ）并不原生充当符合 XA 规范的 Resource Manager（RM），无法与 MySQL / PostgreSQL 的事务管理器协同。
2. **吞吐断崖式崩塌**：两阶段提交是强阻塞协议（Blocking Protocol）。在阶段一（Prepare）与阶段二（Commit）之间，数据库必须持有相关数据行的排他行锁（X-Lock）。网络往返延时（RTT）的任何波动均会导致锁持有时间被放大数倍，系统并发 TPS 骤降 90% 以上，极易诱发级联连接池耗尽。

---

## 2 · 核心拓扑：Pub/Sub + Outbox 全链路编排

**Pub/Sub + Transactional Outbox 模式**的核心哲学是：**将跨系统的分布式协同退化为单机数据库内部的 ACID 本地事务，再通过异步流水线驱动 Pub/Sub 广播扇出与下游解耦。**

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Producer Service (订单服务)                                             │
│ ┌────────────────────────────────────────────────────────────────────┐ │
│ │ Local ACID Transaction Boundary                                    │ │
│ │   1. INSERT INTO orders (order_id, user_id, amount, status, ...)   │ │
│ │   2. INSERT INTO outbox_events (id, aggregate_id, topic, payload)  │ │
│ └────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Commit (原子持久化)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Database (MySQL / PostgreSQL / DynamoDB)                               │
│ ┌───────────────────────────────┐    ┌───────────────────────────────┐ │
│ │ Business Table (orders)       │    │ Outbox Table (outbox_events)  │ │
│ └───────────────────────────────┘    └───────────────────────────────┘ │
│                     ▲                                                  │
│                     └────── Binlog / WAL 物理追加写                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Tailing (CDC / 日志流挖掘)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Message Relay / CDC Engine (Debezium / Kafka Connect)                  │
│ - 读取 commit 日志中的 outbox_events 变更                               │
│ - 确保至少一次投递 (At-Least-Once Delivery)                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Publish to Topic
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Pub/Sub Broker / Event Bus (Kafka / AWS SNS / Pulsar)                  │
│ Topic: order.events (Partition Key: order_id)                          │
└───────────────┬───────────────────────────────────────┬────────────────┘
                │ 1-to-N 广播扇出 (Fan-out)              │
                ▼                                       ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│ Subscription Queue A          │       │ Subscription Queue B          │
│ (SQS / Kafka Group: Billing)  │       │ (SQS / Kafka Group: Inventory)│
└───────────────┬───────────────┘       └───────────────┬───────────────┘
                │ Pull / Push                           │ Pull / Push
                ▼                                       ▼
┌───────────────────────────────┐       ┌───────────────────────────────┐
│ Consumer Service A (计费服务) │       │ Consumer Service B (库存服务) │
│ ┌───────────────────────────┐ │       │ ┌───────────────────────────┐ │
│ │ Local ACID (Inbox 去重):  │ │       │ │ Local ACID (Inbox 去重):  │ │
│ │ 1. INSERT INTO inbox(...) │ │       │ │ 1. INSERT INTO inbox(...) │ │
│ │ 2. UPDATE billing_records │ │       │ │ 2. DEDUCT inventory_stock │ │
│ └───────────────────────────┘ │       │ └───────────────────────────┘ │
└───────────────────────────────┘       └───────────────────────────────┘
```

### 2.1 · 为什么必须是 Pub/Sub 与 Outbox 结合？
* **Outbox 仅解决“单生产者的可靠发出”**：如果只有 Outbox 而没有 Pub/Sub，生产者必须在 Outbox 派发层为每个消费者点对点编写 RPC 调用或点对点入队逻辑。系统强耦合，新增消费者必须侵入生产端修改调度代码。
* **Pub/Sub 解决“1 对 N 的广播拓扑与消费者隔离”**：将 Outbox 的事件发布至 Pub/Sub 主题（Topic/Exchange），由总线负责路由分发。每个下游消费者维护自己的独立订阅队列，消费速率互不干扰，完全实现空间与时序解耦。

---

## 3 · 两种中继挖掘机制对比与权衡 (Relay Mechanisms)

从 `outbox_events` 搬运数据至 Pub/Sub Broker 的组件称为 **Message Relay**。工程实践中有两大演进方向：

| 评估维度 | 模式 A：轮询发布器 (Polling Publisher) | 模式 B：事务日志挖掘 (Transaction Log Tailing / CDC) |
|---|---|---|
| **核心机制** | 后台任务定期执行 SQL：<br>`SELECT * FROM outbox_events WHERE status = 'PENDING' ORDER BY id LIMIT 100 FOR UPDATE SKIP LOCKED`，投递成功后更新状态或删除。 | 依托 CDC 引擎（Debezium、Canal、DynamoDB Streams）监听存储引擎底层追加日志（MySQL Binlog、PostgreSQL WAL）。解析日志即派发。 |
| **主库资源消耗** | **极高**。持续高频全表/索引扫描、加行锁、生成大量 Undo 日志，严重争抢生产业务的连接池与 CPU 资源。 | **极低**。直接流式读取物理磁盘或主从复制流，不向数据库主查询引擎下发任何 SQL 解析与锁。 |
| **端到端延迟** | **秒级 (1s - 5s)**。受限于轮询周期间隔（避免压垮 DB 必须加 Sleep），天然存在拉取空窗期。 | **亚秒级 (10ms - 100ms)**。事务提交刷盘与 WAL 生成同步，日志解析后近乎实时推入 Pub/Sub。 |
| **基础设施依赖** | **零外部依赖**。单机几行调度代码即可运行，数据库中立，开发极快。 | **较高**。需维护 CDC 运维集群（Kafka Connect / Debezium），依赖底层数据库特定的复制特权与日志格式。 |
| **工业界适用场景** | 小型内部系统、原型验证、事件频率极低的批处理系统。 | **高并发、高吞吐、微服务核心骨干通信的绝对工业标准**。 |

---

## 4 · 扇出拓扑：Topic-to-Queue 模式 (SNS-to-SQS 范式)

在分布式消息架构中，必须严格区分 **Topic（广播分发逻辑单元）** 与 **Queue（缓冲消费物理载体）**：

```text
                ┌───────────────────────────────────┐
                │ Pub/Sub Topic (AWS SNS / Rabbit Ex)│
                └─────────────────┬─────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │ Fan-out                │ Fan-out                │ Fan-out
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Queue A (Bill)   │     │ Queue B (Ship)   │     │ Queue C (Audit)  │
│ - Buffer: 100k   │     │ - Buffer: 500    │     │ - Buffer: 10k    │
│ - Rate: 200/s    │     │ - Rate: 5,000/s  │     │ - Rate: 50/s     │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Billing Worker   │     │ Shipping Worker  │     │ Audit Worker     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

1. **缓冲与背压（Buffering & Backpressure）**：
   - 若直接向消费者的 HTTP/RPC 端点做纯推送广播，当下游突发流量冲垮某一个消费者时，消息将直接丢失。
   - 在 Topic 与 Consumer 之间挂接独立持久化队列（如 SQS、RabbitMQ 绑定队列、Kafka Consumer Group），由队列承载瞬态积压，下游按自身处理能力匀速拉取（Pull-based Backpressure）。
2. **故障爆炸半径隔离（Failure Blast Radius Isolation）**：
   - 计费服务宕机不会导致发货服务队列堆积；审计服务处理变慢不会拖慢实时物流履约。
3. **独立重试与死信隔离（Independent Retry & DLQ）**：
   - 每个队列可配置独立的可见性超时（Visibility Timeout）、最大重试次数与死信队列（DLQ）。

---

## 5 · 对称设计：消费端 Transactional Inbox 模式

Outbox 模式仅能提供**“至少一次投递 (At-Least-Once Delivery)”**。在网络超时、重传、Relay 重启或下游消费确认（ACK）丢失时，下游微服务**必然会收到重复事件**。

为了在业务层达成**有效仅一次（Effectively-Once Processing）**，消费端必须采用镜像对称的 **Transactional Inbox（事务收件箱）** 模式：

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Consumer Service Worker                                                │
│                                                                        │
│   Event: { message_id: "evt_1001", order_id: "ord_99", amount: 100 }  │
│                                                                        │
│   BEGIN TRANSACTION;                                                   │
│     -- 1. 利用唯一主键约束防重拦截                                     │
│     INSERT INTO inbox_records (message_id, handler_name, processed_at) │
│     VALUES ('evt_1001', 'billing_handler', NOW());                     │
│                                                                        │
│     -- 若主键冲突 (Duplicate Key)，立即 ROLLBACK 并直接向 MQ 发送 ACK   │
│                                                                        │
│     -- 2. 执行真正的业务领域变更                                       │
│     UPDATE user_balance SET balance = balance - 100                    │
│     WHERE user_id = 'usr_42';                                          │
│   COMMIT;                                                              │
│                                                                        │
│   ACK to Queue (确认消费位移)                                          │
└────────────────────────────────────────────────────────────────────────┘
```

* **原子去重**：由于收件箱记录（Inbox Record）与业务变更处在同一个本地 ACID 事务中，一旦由于重试再次消费相同事件，数据库唯一键冲突将促使整个事务回滚，杜绝二次扣款。
* **幂等确认**：遇到重复消息回滚后，消费者**必须正常向消息队列回复 ACK**。若不回复 ACK，该重复消息将永远在队列中循环投递，阻塞队列进度。

---

## 6 · 领域状态转移与分区时序保障

### 6.1 · 瘦事件通知 (Thin Notification) vs 状态转移 (ECST)

| 模式 | Payload 内容 | 核心收益 | 核心缺陷与工程风险 |
|---|---|---|---|
| **瘦事件 (Thin Notification)** | 仅包含最小元数据：<br>`{ "event_id": "...", "order_id": "123", "action": "UPDATED" }` | 1. 消息体极小，节省带宽与 MQ 存储。<br>2. 架构边界清晰，不泄露完整模型。 | **读风暴 (Read Storm) 与时序竞态**：10 个下游消费者收到事件后并发 RPC 回查订单主库，产生 $10\times$ 放大查询；且回查可能查到“更未来的脏数据”。 |
| **携带状态转移 (ECST)** | 包含变更后实体完整快照：<br>`{ "order_id": "123", "status": "PAID", "items": [...], "version": 4 }` | **彻底消除下游回查依赖**：下游凭事件自身数据即可完成本地物化视图更新，彻底切断下游对上游主库的网络调用。 | 消息体膨胀；若 schema 演化频繁，需维护严格兼容性协议（Avro / Protobuf + Schema Registry）。 |

> **生产选型标准**：高并发与高扇出系统强烈推荐 **ECST（Event-Carried State Transfer）**，通过消息大小换取系统间的运行时解耦与保护主库免于查询风暴。

### 6.2 · 实体因果时序 (Causal Ordering)
对于同一个实体的多次状态跃迁（例如：`OrderCreated` $\to$ `OrderPaid` $\to$ `OrderCancelled`），下游必须严格按顺序处理。
* **分区键绑定（Partition Key Binding）**：Relay 在将 Outbox 事件投递至 Kafka / Pulsar 时，必须指定 `aggregate_id`（如 `order_id`）作为 Partition Key。
* **数学保证**：
  $$\text{Partition} = \text{MurmurHash2}(\text{aggregate\_id}) \pmod{\text{NumPartitions}}$$
  同一订单的所有生命周期事件将严格落入同一物理分区，由单线程或单一消费者按 Offset 顺序串行消费，彻底杜绝“后发事件先到达”的因果颠倒。

---

## 7 · 生产级容灾体系：毒丸、重试退避与死信队列 (DLQ)

```text
[Incoming Event] ──► [Try Consume] ──(业务处理成功)──► [ACK to Queue]
                           │
                           ▼ (捕获异常 / 网络超时)
                     [Retry Count < 3 ?]
                      ├── Yes ──► [Delayed Queue: 指数退避 + Jitter]
                      └── No  ──► [Dead Letter Queue (DLQ)]
                                          │
                                          ▼
                                   [告警通知 / 人工排查]
                                          │
                                          ▼ (修复后)
                                   [一键重放工具 Replay]
```

1. **毒丸消息 (Poison Pill)**：
   - 当事件 Payload 存在格式畸变、反序列化异常或无法修复的业务逻辑 Bug 时，若无限制重试，消费者将陷入无限 Panic / CrashLoop，阻塞整条队列通道。
2. **三级重试与退避策略**：
   - **立即重试 (Immediate Retry)**：本地原地重试 2 次，应对瞬态网络抖动。
   - **指数退避重试队列 (Exponential Backoff with Full Jitter)**：
     $$T_{\text{sleep}} = \min(T_{\text{max}}, T_{\text{base}} \times 2^{\text{attempt}}) \times \text{Uniform}(0, 1)$$
     将消息移入延迟队列（如 10s、1m、5m 重试桶），释放当前工作线程。
3. **死信队列 (Dead Letter Queue, DLQ)**：
   - 超过最大重试阈值（如 3-5 次）后，将原始消息、异常堆栈与上下文元数据封装入 DLQ。
   - 触发 P1 监控报警，并提供 Webhook / CLI 工具在修复 Bug 后批量 Replay 回主队列。

---

## 8 · 工业级 Schema 定义与代码实现蓝图

### 8.1 · 数据库 Schema 定义 (PostgreSQL 生产范式)

```sql
-- 1. 生产者端 Outbox 表
CREATE TABLE outbox_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregate_type VARCHAR(64) NOT NULL,       -- 如 'Order', 'Payment'
    aggregate_id VARCHAR(64) NOT NULL,         -- 实体唯一键，用于 Kafka 分区 Hash
    event_type VARCHAR(64) NOT NULL,           -- 如 'OrderPaid', 'OrderCancelled'
    payload JSONB NOT NULL,                    -- 领域事件数据 (ECST 载荷)
    trace_id VARCHAR(64) NOT NULL,             -- 分布式链路追踪 ID
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- 建立单日分区，便于历史数据直接 DROP PARTITION，杜绝 DELETE 引起的表膨胀
CREATE TABLE outbox_events_2026_10_07 PARTITION OF outbox_events
    FOR VALUES FROM ('2026-10-07 00:00:00+00') TO ('2026-10-08 00:00:00+00');

-- 2. 消费者端 Inbox 表
CREATE TABLE inbox_records (
    message_id VARCHAR(64) NOT NULL,           -- 事件全局唯一 ID
    consumer_group VARCHAR(64) NOT NULL,       -- 消费者组名称
    handler_name VARCHAR(64) NOT NULL,         -- 具体处理函数
    status VARCHAR(32) NOT NULL DEFAULT 'DONE',
    processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (message_id, consumer_group)
);
```

### 8.2 · 生产端：单本地事务原子落盘 (Go 核心伪代码)

```go
func (s *OrderService) CompleteOrderPayment(ctx context.Context, orderID string) error {
    tx, err := s.db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback()

    // 1. 更新主业务订单状态
    const updateOrderSQL = `UPDATE orders SET status = 'PAID', updated_at = NOW() WHERE id = $1`
    if _, err := tx.ExecContext(ctx, updateOrderSQL, orderID); err != nil {
        return fmt.Errorf("failed to update order: %w", err)
    }

    // 2. 构造领域事件 (ECST 快照)
    eventPayload := OrderPaidEvent{
        OrderID:     orderID,
        Status:      "PAID",
        PaidAt:      time.Now().UTC(),
        TraceID:     tracer.TraceIDFromContext(ctx),
    }
    payloadBytes, _ := json.Marshal(eventPayload)

    // 3. 在同一事务内插入发件箱表
    const insertOutboxSQL = `
        INSERT INTO outbox_events (aggregate_type, aggregate_id, event_type, payload, trace_id)
        VALUES ($1, $2, $3, $4, $5)`
    if _, err := tx.ExecContext(ctx, insertOutboxSQL, "Order", orderID, "OrderPaid", payloadBytes, eventPayload.TraceID); err != nil {
        return fmt.Errorf("failed to insert outbox: %w", err)
    }

    // 4. 提交本地 ACID 事务：业务数据与事件记录原子持久化
    return tx.Commit()
}
```

### 8.3 · 消费端：Inbox 幂等消费模板 (Python 核心逻辑)

```python
def process_order_event(db_pool, message: KafkaMessage):
    event = json.loads(message.value)
    event_id = event["event_id"]
    order_id = event["aggregate_id"]

    with db_pool.get_connection() as conn:
        with conn.cursor() as cur:
            try:
                # 1. 尝试原子占位 Inbox
                cur.execute(
                    """
                    INSERT INTO inbox_records (message_id, consumer_group, handler_name)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (message_id, consumer_group) DO NOTHING
                    RETURNING message_id;
                    """,
                    (event_id, "billing_service_group", "handle_order_paid")
                )
                row = cur.fetchone()
                if not row:
                    # 冲突触发：说明该事件历史已处理完毕，直接 ACK 跳过
                    logger.info("Duplicate event %s skipped successfully.", event_id)
                    conn.commit()
                    return

                # 2. 执行真正的业务领域变更
                cur.execute(
                    "UPDATE account_balance SET balance = balance - %s WHERE order_id = %s",
                    (event["amount"], order_id)
                )

                conn.commit()  # Inbox 插入与业务更新原子提交
            except Exception as e:
                conn.rollback()
                logger.error("Processing failed, message will be retried: %s", str(e))
                raise e
```

---

## 9 · 全景架构方案选型对比矩阵

| 架构方案 | 数据一致性保证 | 对主库性能冲击 | 端到端发布延迟 | 系统复杂度与运维成本 | 适用场景与边界评价 |
|---|---|---|---|---|---|
| **朴素双写 (Naive Dual-Write)** | 无一致性保证（必然出现数据永久丢失或幽灵脏读） | 低 | 毫秒级 | 极低（零额外组件） | **严禁在生产核心链路使用**。仅适用于失步无任何影响的统计日志打点。 |
| **分布式 2PC / XA 事务** | 强一致性 (CP) | **灾难性**（长事务行锁争抢，并发吞吐崩溃） | 秒级（多阶段网络往返与协调超时） | 极高（多数现代 MQ 原生不支持） | 仅适用于单机房内支持 XA 的传统金融同构关系型数据库集群之间。 |
| **Outbox + 轮询发布器 (Polling)** | 最终一致性 (At-Least-Once) | 较高（定时轮询扫描、锁表与连接争用） | 秒级 (1s - 5s) | 低（应用内部起 Cron 调度线程即可） | 适用于低吞吐原型验证、小微型企业管理系统或非关键批处理。 |
| **Outbox + CDC + Pub/Sub (黄金组合)** | **最终一致性 (At-Least-Once + 端到端 Effectively-Once)** | **极低**（流式读取磁盘 WAL，零主库查询与行锁） | **毫秒级 (10ms - 50ms)** | 中等（需维护 Debezium / Kafka Connect 与 MQ 拓扑） | **现代分布式与微服务高并发核心生产系统的绝对首选标准架构**。 |
| **直接监听业务表 CDC (Direct Table CDC without Outbox)** | 弱最终一致性 | 极低 | 毫秒级 | 中等 | **存在严重缺陷**：丢失业务事件意图（只能捕获行级 CRUD 物理差异，无法推导复合动作）；中间态物理更新全部暴露；强行耦合下游与主表 Schema。无法替代 Outbox。 |

# Wiki · Event Bus (事件总线与事件驱动架构)

Wiki 词条归属：[[SystemDesign00 Overview|00 系统设计全局蓝图]] → [[SystemDesignWiki Event Bus|Wiki · Event Bus]]

事件总线（Event Bus）是**事件驱动架构（Event-Driven Architecture, EDA）**的核心路由骨干，负责在松耦合的微服务或组件之间接收、过滤、路由并分发**领域事件（Domain Events）**。

```queue-vs-stream-visual
```

---

## 1 · 核心设计理念与架构模式

### 1.1 · 发布/订阅范式与事件通知本质
事件总线严格遵循 **发布/订阅（Publish-Subscribe / Pub/Sub）** 模式，实现发布方与订阅方的全方位解耦：
- **时序解耦**：发布者与订阅者无需同时在线。
- **空间解耦**：发布者不知道也不会依赖任何订阅者的网络地址或服务标识。
- **语义解耦**：
  - **Event（事件）vs Command（命令）**：命令（如 `CreateOrderCommand`）具有强指向性，期望特定接收者执行特定副作用；事件（如 `OrderPaidEvent`）是对已发生事实的客观声明，发布者对“谁会响应、产生何种副作用”保持零感知（Zero Expectation）。
  - **事件携带状态转移（Event-Carried State Transfer, ECST）**：事件 Payload 包含实体发生变更后的关键全量/增量快照。下游消费者仅凭事件自身内容即可完成本地状态维护，彻底避免所有下游并发回查主业务数据库引发的**查询读风暴（Read Storm）**。

### 1.2 · 事件总线 / 日志流架构与实现原理
以 Kafka、Pulsar 为代表的事件总线与流式系统，核心设计哲学是**“分布式只追加日志（Append-only Commit Log）”**。它的底层数据结构是**磁盘上按顺序追加写入的二进制物理文件**，服务端不维护逐条消息的 ACK 状态。

```text
[Producer 1] ───┐
                ├──(Hash Key: repo_id)──┐
[Producer 2] ───┘                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Partition / Shard (00000.log)                          │
│                                                                             │
│  [Offset 0] [Offset 1] [Offset 2] [Offset 3] ... [Offset 7] ... [Offset 9]  │
│                                       ▲                     ▲               │
└───────────────────────────────────────┼─────────────────────┼───────────────┘
                                        │                     │
                  ┌─────────────────────┘                     └───────────────┐
                  │ (读取顺序数据块)                                          │ (读取顺序数据块)
                  ▼                                                           ▼
       [Consumer Group A: Push Service]                           [Consumer Group B: Runner Gateway]
       (Current Offset: 3)                                        (Current Offset: 7)
       - 独立游标，顺序向前移动                                      - 独立游标，互相完全不干扰
       - 支持回滚：Offset = 0 (全量重放)                             - 高性能：零拷贝 (sendfile)
```

### 1.3 · 事件总线 / 日志流底层四大核心运转机制
1. **分布式只追加物理日志（Append-only Commit Log）：**
   数据一律按顺序物理写入磁盘文件末尾，写操作退化为**纯顺序写（Sequential Write）**。充分利用 OS PageCache 与磁盘预读（Read-ahead），单机磁盘可跑满千兆网卡与数百 MB/s 顺序吞吐。
2. **服务端完全无状态（Stateless Broker & Offset Cursor）：**
   Broker **不记录**任何单条消息的处理状态，只记录每个消费者组当前的**游标位置（Offset）**。因此单机可以轻松挂载数万个分区，支撑数十万并发写入。
3. **多订阅组独立全量/扇出消费（Multi-Group Independent Fan-out）：**
   不同消费者组（Consumer Group）独立扫描同一份 Log，消费进度各不影响。支持将 Offset 随意重置到任何历史位置进行**全量重放（Replay）**。
4. **时间窗口生命周期（Time-based Retention & Non-destructive Read）：**
   消费操作是**非破坏性读取（Non-destructive Read）**。消息被读后绝不会被删除，而是保留配置的时间（如 7 天），过期后由后台异步清理整个物理段（Segment）。

### 1.4 · 路由拓扑与过滤机制
1. **基于主题与通配符路由（Topic / Subject-Based Routing）**：
   - 采用分层层级命名，如 `orders.<region>.<event_type>`（例如 `orders.us-east.created`）。
   - 消费者通过单层（`*`）或多层（`#` / `>`）通配符订阅感兴趣的数据流。
2. **基于内容过滤（Content-Based Filtering / Rule Engine）**：
   - 现代云原生总线（如 AWS EventBridge、Azure Event Grid）允许订阅者通过声明式 JSON Pattern 过滤事件，无需消费者拉取全量事件到应用层反序列化再过滤：
     ```json
     {
       "source": ["ecommerce.order"],
       "detail-type": ["OrderPaid"],
       "detail": {
         "amount": [{ "numeric": [ ">", 500 ] }],
         "currency": ["USD"]
       }
     }
     ```

### 1.5 · 瞬态内存型 vs 持久可靠型总线

| 机制维度 | 瞬态内存型 (In-Memory / Ephemeral) | 持久可靠型 (Durable / Persistent) |
|---|---|---|
| **代表技术** | Spring ApplicationEventMulticaster, Redis Pub/Sub, NATS Core | AWS EventBridge, NATS JetStream, Google Cloud Pub/Sub |
| **存储介质** | 纯进程堆内存 / Broker 内存 Buffer，不落地磁盘 | 分布式只追加 WAL / 分布式复制存储 |
| **离线容错** | 消费者断连期间产生的消息直接丢失（Fire-and-forget） | 保留消息（如 24 小时至 7 天），支持断点拉取与 DLQ |
| **元数据契约** | 代码级 POJO / 宽松 JSON | Schema Registry（Protobuf, JSON Schema, Avro）校验 |

---

## 2 · 核心架构机制对照（事件总线/流 vs 消息队列）

| 维度 | 事件总线 / 日志流 (Event Bus / Stream) | 传统消息队列 (Message Queue) |
| :--- | :--- | :--- |
| **底层存储结构** | 磁盘顺序只追加二进制日志段 (`.log` Commit Log) | 动态内存 FIFO 队列 / 双向链表 / 环形缓冲 |
| **消息状态所有权** | **Consumer 端无状态** (Broker 只存整型数字 Offset 游标) | **Broker 端强状态** (逐条维护 Ready / Unacked / Acked) |
| **读取操作性质** | **非破坏性读取 (Non-destructive Read)**，日志支持任意回滚与重放 | **破坏性读取 (Destructive Read)**，Ack 后立即物理删除 |
| **消费模型与扇出** | **多组独立广播 (Multi-group Fan-out)**，单分区内严格保序 | **竞争消费者 (Competing Consumers)**，点对点单次交付 |
| **海量积压承受力** | **极高** (冷热数据自动落盘，积压 TB 级数据对写入吞吐几乎零影响) | **极差** (数百万堆积耗尽 RAM 触发磁盘 Page Swapping 导致吞吐暴跌) |

---

## 3 · 吞吐量级与性能基准（QPS & Latency）

事件总线的性能边界严格受限于其持久化机制与网络 I/O 拓扑：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                      Event Bus 性能与延迟阶梯                           │
├─────────────────────────┬──────────────────────┬───────────────────────┤
│ 部署形态                │ 典型吞吐量 (QPS)      │ 端到端分发延迟        │
├─────────────────────────┼──────────────────────┼───────────────────────┤
│ 进程内总线 (In-Process) │ 1,000,000 - 10,000,000│ < 1 微秒 (0.1 - 1 μs) │
│ 内存网络型 (Redis P/S)  │ 50,000 - 100,000 /node│ 0.5 - 2 毫秒          │
│ 高性能集群 (NATS Core)  │ 500,000 - 2,000,000  │ 1 - 5 毫秒            │
│ 云原生托管 (EventBridge)│ 5,000 - 20,000       │ 20 - 100 毫秒         │
└─────────────────────────┴──────────────────────┴───────────────────────┘
```

- **In-Process 总线**：纯函数调用或内存 RingBuffer（如 LMAX Disruptor），吞吐达千万级，单核纳秒/微秒级响应，受限于单节点垂直扩展极限。
- **Redis Pub/Sub**：依赖单线程事件循环向所有订阅 Socket 写入数据，不持久化，QPS 上限受网卡中断与包吞吐限制（通常 $50\text{k} - 100\text{k}\text{ QPS}$）。
- **云原生 EventBridge**：具备复杂的 JSON 规则匹配、跨账号权限校验与 DLQ 保证，单规则评估开销较高，适合控制面或中等业务量（数千至万级 QPS），延迟为数十毫秒。

---

## 4 · 适用场景与面试系统设计选型

### 4.1 · 最佳适用场景
1. **领域微服务广播与业务解耦（Case 07 / Case 10 / Case 11）**：
   - 用户完成支付后，核心交易服务发布 `OrderPaid` 事件。
   - 物流履约、积分返还、营销发券、风控审计、用户推送等多个下游微服务独立接收并并行触发后续逻辑，交易系统无需硬编码下游依赖。
2. **分布式集群缓存失效与配置广播（Cache Invalidation Fan-out）**：
   - 当元数据在后台管理系统被修改时，向 Event Bus 发布 `ConfigChanged` 事件，集群内成百上千个应用节点收到广播后即刻清除本地堆内存（Caffeine / Guava Cache）。
3. **第三方 SaaS 与 Webhook 统一网关分发**：
   - 统一接收 GitHub、Stripe、Shopify 的外发 Webhook，通过事件总线的内容路由分流到各自内部对应的微服务。

### 4.2 · 架构禁忌与反模式（When NOT to Use）
- **禁忌 1：抢占式耗时重任务排队**：如果下游需要多个 Worker 竞争抢占执行单个耗时任务（如视频转码、PDF 生成），发布到 Pub/Sub 会导致每个 Worker 都执行一次该任务（全量重复）。应使用 **Message Queue（竞争消费者模型）**。
- **禁忌 2：海量时序回溯与重放**：若需要百万级 QPS 日志接入并支持随意回退游标重新消费（如离线数据训练、OLAP 导入），应使用 **Kafka / Partitioned Log**。

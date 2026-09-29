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

## 2 · Dispatcher 与 Fan-out 扇出架构范式

### 2.1 · 扇出（Fan-out）的工程本质与两大技术挑战
在事件驱动体系中，**扇出（Fan-out）** 指单一上游事件触发多个下游实体或服务消费的架构机制。然而，简单的“Broker 广播”在工业级场景下会瞬间引发两大技术矛盾：
1. **服务间异构消费与慢消费者阻塞（Inter-Service Heterogeneity & Slow Consumer Problem）**：
   若发布者直接循环调用或由单一消息代理同步分发，当下游某一个异构系统（如审计或数据分析）发生延迟或宕机时，会反向阻塞甚至击垮核心业务服务（如支付或交易）。
2. **爆炸级实体放大与出网带宽/系统击穿（1-to-N Entity Explosion & Egress Amplification）**：
   在面向用户的业务场景（如全员广播推送、明星发帖 Timeline 更新、大规模缓存批量失效），单个根事件在下游会瞬间放大为 $10^5 \sim 10^7$ 个物理任务。若由单个节点或进程同步循环遍历，会导致该节点 CPU 占满、内存暴涨（OOM）、第三方通道限流（如 APNs / SMS 429）以及严重的长尾延迟（Head-of-Line Blocking）。

### 2.2 · 双层扇出拓扑架构（Two-Tier Fan-out Architecture）

工业界高性能事件总线系统通常将扇出切分为两个清晰的物理层级：

```text
[Upstream Event: OrderPaid / BroadcastAlert]
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier-1: 服务级拓扑扇出 (Service-Level Fan-out / Topic-to-Queue)               │
│                                                                             │
│                               [Topic / Exchange]                            │
│                                       │                                     │
│            ┌──────────────────────────┼──────────────────────────┐          │
│            ▼                          ▼                          ▼          │
│     [Queue: Billing]           [Queue: Fraud]             [Queue: Dispatcher]│
│            │                          │                          │          │
│            ▼                          ▼                          │          │
│     [Billing Service]          [Fraud Service]                   │          │
└──────────────────────────────────────────────────────────────────┼──────────┘
                                                                   │
                                                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Tier-2: 工作负载级分发器扇出 (Workload-Level Dispatcher & Sharded Slicing)   │
│                                                                             │
│                        [Notification Dispatcher]                            │
│                 (1. 幂等去重 -> 2. 目标解析与分片切块)                       │
│                                       │                                     │
│       ┌───────────────────────────────┼───────────────────────────────┐     │
│       ▼ (Chunk 1: 1~1,000)            ▼ (Chunk 2: 1,001~2,000)        ▼ ... │
│ [Priority Worker Queue 1]       [Priority Worker Queue 2]       [...]       │
│       │                               │                               │     │
│       ▼                               ▼                               ▼     │
│ [Delivery Worker Pool]         [Delivery Worker Pool]          [...]        │
│       │                               │                               │     │
│       ▼ (Token Bucket 流控)           ▼ (Token Bucket 流控)           ▼     │
│ ┌─────────────────────────────────────────────────────────────────────────┐ │
│ │           第三方通道 / 下游目标 (APNs HTTP/2, FCM, SMS, Webhooks)         │ │
│ └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 · Tier-1：服务级拓扑扇出（Topic-to-Dedicated-Queue）
1. **SNS-to-SQS 范式 / Fan-out Exchange**：
   - 发布者仅向单个主题（Topic）发布事件。
   - 消息中间件（如 AWS SNS、RabbitMQ Fanout Exchange）负责将事件完整拷贝投递至各个订阅者独立的持久化队列（Dedicated SQS / Queue）。
2. **四大核心收益**：
   - **完全故障隔离（Fault Isolation）**：下游分析服务宕机，仅导致其私有队列产生积压，核心账单和通知服务毫发无损。
   - **弹性削峰缓冲（Independent Buffering）**：各消费者按自身消费能力（Prefetch / Concurrency）拉取，杜绝慢消费者反噬。
   - **独立重试与死信队列（Independent DLQ）**：重试策略、最大重试次数和 DLQ 针对业务特性差异化配置（例如账单重试 10 次，打点仅重试 1 次）。
   - **安全与权限解耦（IAM Least Privilege）**：各消费者仅对其绑定的私有队列拥有读写权限。

### 2.4 · Tier-2：工作负载级分发器扇出（Dispatcher & Batch Slicing）
当单个事件需要扇出至大规模实体（如 [[SystemDesign11 Notification System|Case 11 · 海量移动推送平台]]、[[SystemDesign07 Photo Sharing Feed|Case 07 · 社交 Feed 流写扩散]]）时，必须引入专门的 **Dispatcher（分发器）** 模式，执行经典的三阶段流水线：

1. **阶段 1：全局幂等准入（Admission & Global Idempotency）**：
   - Dispatcher 首先基于事件的唯一业务键（如 `event_id` 或 `campaign_id`）在 Redis / 数据库中执行原子 CAS（如 `SETNX campaign:{id}:status IN_PROGRESS EX 86400`）。
   - 若命中已存在，直接丢弃或跳过，杜绝上游重投导致千万级任务二次膨胀。
2. **阶段 2：目标解析与微批切片（Target Resolution & Micro-Batch Slicing）**：
   - **游标分页与分块生成**：Dispatcher 从关系数据库、图数据库（如关注列表）或画像系统（如标签群组）按游标（Cursor）或主键区间流式拉取目标 ID。
   - **规整微批（Chunking）**：将大规模目标列表按固定粒度（如 500 ~ 1,000 个目标）切分为独立的微批切片任务：
     $$\text{Chunk}_k = \{ \text{user\_id}_i \}_{i=(k-1)B + 1}^{\min(kB, N)}, \quad B \in [500, 1000]$$
   - **确定性切片 ID**：生成 `chunk_id = hash(event_id, k)`，赋予每个切片独立的幂等粒度。
   - **偏好与静音初筛**：结合 Redis 缓存中的全局免打扰设置（Do Not Disturb）或用户黑名单，在切片阶段提前剔除无效目标，减少下游无谓的网络 RPC 开销。
3. **阶段 3：分发入队与物理优先级隔离（Queuing & Priority Isolation）**：
   - Dispatcher 将切片任务异步写入下游的工作队列。
   - **优先级隔离**：验证码、安全告警等高优先级切片走 `high-priority-queue`，营销大促等大广播切片走 `bulk-queue`，避免海量广播切片阻塞实时高危任务。
4. **阶段 4：并行 Worker 抢占与第三方流控（Worker Execution & Egress Throttling）**：
   - 下游 Delivery Worker 集群并行抢占消费切片。
   - **出网流量整流（Egress Rate Limiting）**：针对第三方 Provider（如 APNs 单长连接并发限制、FCM 速率上限、短信网关 QPS 限制），挂载基于分布式 Redis 令牌桶（Token Bucket）的速率限制器，平滑下发，防止触发 429 Too Many Requests。
   - **细粒度失败归档与部分失败隔离**：单个切片内若仅部分目标因设备 Token 失效或网络超时失败，仅将失败的目标列表单独投递至重试队列或失效设备清理队列，绝不进行切片全量重投。

### 2.5 · 核心工程权衡与避坑矩阵

| 架构维度 | 简单循环广播 (Naive In-Process Loop) | 仅依赖 Topic 广播 (Topic-Only Fan-out) | Dispatcher + 队列微批分发 (Dispatcher Slicing) |
|---|---|---|---|
| **单任务吞吐能力** | 极低（单机阻塞，受限于线程与连接） | 中等（受 Broker 单分区读写与网络出带宽限制） | **极高**（水平扩展无上限，分片并行入队与消费） |
| **故障隔离性** | 极差（单点故障导致后续目标全部丢失） | 较好（消费者组解耦，但单一慢消费者易挤占 Broker 内存） | **最高**（切片级失败隔离，独立 DLQ 与重试） |
| **流控与防击穿** | 难以实现跨节点全局流控 | 依赖 Broker 端背压，无法按第三方出口渠道细粒度整流 | **最佳**（Worker 池结合分布式令牌桶，精准平滑削峰） |
| **重试与幂等成本** | 失败需全量重跑，产生大量重复投递 | 分区级别重试，重试粒度较粗 | **切片级/个体级幂等**，支持增量补发 |

---

## 3 · 核心架构机制对照（事件总线/流 vs 消息队列）

| 维度 | 事件总线 / 日志流 (Event Bus / Stream) | 传统消息队列 (Message Queue) |
| :--- | :--- | :--- |
| **底层存储结构** | 磁盘顺序只追加二进制日志段 (`.log` Commit Log) | 动态内存 FIFO 队列 / 双向链表 / 环形缓冲 |
| **消息状态所有权** | **Consumer 端无状态** (Broker 只存整型数字 Offset 游标) | **Broker 端强状态** (逐条维护 Ready / Unacked / Acked) |
| **读取操作性质** | **非破坏性读取 (Non-destructive Read)**，日志支持任意回滚与重放 | **破坏性读取 (Destructive Read)**，Ack 后立即物理删除 |
| **消费模型与扇出** | **多组独立广播 (Multi-group Fan-out)**，单分区内严格保序 | **竞争消费者 (Competing Consumers)**，点对点单次交付 |
| **海量积压承受力** | **极高** (冷热数据自动落盘，积压 TB 级数据对写入吞吐几乎零影响) | **极差** (数百万堆积耗尽 RAM 触发磁盘 Page Swapping 导致吞吐暴跌) |

---

## 4 · 吞吐量级与性能基准（QPS & Latency）

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

## 5 · 适用场景与面试系统设计选型

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

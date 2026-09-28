# Wiki · Message Queue (消息队列与点对点工作队列)

Wiki 词条归属：[[SystemDesign00 Overview|00 系统设计全局蓝图]] → [[SystemDesignWiki Message Queue|Wiki · Message Queue]]

消息队列（Message Queue, MQ）是分布式计算中最经典的点对点异步通信基础设施，其核心使命是实现**生产者与消费者的吞吐削峰、负载均衡与任务抢占式协同**。

```queue-vs-stream-visual
```

---

## 1 · 核心设计理念与架构模式

### 1.1 · 传统消息队列架构与实现原理
传统消息队列（如 RabbitMQ、ActiveMQ）的核心设计哲学是**“邮局 / 任务派发模型”**。它的底层数据结构本质是一个**动态的 FIFO 队列或双向链表**，服务端必须实时维护每条消息的生命周期状态。

```text
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      Message Queue Broker                   │
                    │                                                             │
                    │   [Exchange / Router]                                       │
[Producer] ──(Push)─┼─>   (根据规则路由)                                            │
                    │           │                                                 │
                    │           ▼                                                 │
                    │   ┌─────────────────────────────────────────────────────┐   │
                    │   │ In-Memory Queue (内存链表/双向缓冲)                  │   │
                    │   │                                                     │   │
                    │   │  [Msg 4]  ──>  [Msg 3]  ──>  [Msg 2]  ──>  [Msg 1]  │   │
                    │   │ (Ready)       (Ready)      (Unacked)     (Unacked)  │   │
                    │   └─────────────────────────────────────────────────────┘   │
                    │                     │                     │                 │
                    └─────────────────────┼─────────────────────┼─────────────────┘
                                          │ (Push/Prefetch)     │ (Push/Prefetch)
                                          ▼                     ▼
                                     [Worker 1]            [Worker 2]
                                         │                     │
                                         └───── ACK(Msg 1) ────┴──── ACK(Msg 2)
                                                 │
                                                 ▼
                                        (Broker 物理删除已确认消息)
```

### 1.2 · 传统消息队列底层四大核心运转机制
1. **细粒度状态跟踪（Server-side State Machine）：**
   Broker 为内存中的**每一条消息**维护状态机流转：
   - `Ready`：等待被消费；
   - `Unacked`：已派发给 Worker，等待确认回执；
   - `Acked`：Worker 确认处理完毕；
   - `Dead-letter`：重试超限，移入死信队列。
2. **点对点竞争消费（Competing Consumers）：**
   同一个队列上的多个 Worker 是**竞争关系**（Work-stealing 或 Round-robin），一条消息在同一时刻只会被其中一个 Worker 认领并消费。
3. **阅后即焚（Destructive Read）：**
   一旦 Broker 收到 Worker 的 ACK 回执，该消息在物理内存或磁盘上的索引会**立即被清理/标记删除**。队列长度通常保持在很低的水平。
4. **性能瓶颈（锁与内存换页）：**
   状态变更需要加锁和更新内存结构；一旦消费者处理变慢导致百万级消息积压，内存耗尽触发操作系统 Page Swapping，吞吐量将剧烈下滑。

### 1.3 · 消息租约、可见性超时与两阶段确认
为了防止 Worker 挂掉导致任务静默丢失，生产级消息队列均采用租约机制（如 AWS SQS Visibility Timeout、RabbitMQ Unacked Prefetch）：

```text
[Producer] ──(1. Push Task)──> [Queue Storage]
                                      │
                               (2. Lease Message)
                                      ▼
                                [Worker Node]
                             (Task In-flight...)
                             ┌─────────────────┐
                             │ Success: Ack()  │ ──> (3a. Delete from Queue)
                             │ Crash / Timeout │ ──> (3b. Re-visible to other Workers)
                             └─────────────────┘
```

1. **Lease (租约拉取)**：Worker 拉取消息后，消息在队列中并未物理删除，而是进入 `In-flight` 不可见状态，并启动计时器（如 30 秒）。
2. **Ack (显式确认)**：Worker 成功执行完业务副作用后，显式调用 `Ack(msg_id)`，Broker 随即物理删除该消息。
3. **Nack / Timeout (故障恢复)**：若 Worker 在可见性超时时间内宕机崩溃或显式返回 `Nack()`，消息立即重置为可见状态，供其他健康 Worker 重新拉取抢占。

### 1.4 · 死信队列与毒丸隔离（Dead Letter Queue & Poison Pill Isolation）
当某条消息本身由于数据损坏或代码 Bug 导致任何 Worker 处理都会抛出非预期崩溃时，该消息被称为**毒丸（Poison Pill）**：
- 如果没有隔离机制，毒丸会被无限次拉取、崩溃、重回队列，造成整个集群 CPU 占满与**队头阻塞（Head-of-Line Blocking）**。
- **DLQ 熔断策略**：队列为每条消息维护递增计数器 `ReceiveCount`。当 `ReceiveCount > maxReceiveCount`（如 5 次重试仍未 Ack），Broker 强制将消息转移至**死信队列（Dead Letter Queue, DLQ）**，并发出告警人工介入，确保主队列畅通无阻。

### 1.5 · 延迟队列与优先级调度（Delay Queues & Priority Queues）
- **延迟队列（Scheduled / Delay Queues）**：基于时间轮（Timing Wheel）或最小堆，实现消息入队后在指定的延迟时间（如 15 分钟）之后才对 Worker 暴露。广泛用于分布式超时关单（如未支付订单取消）。
- **优先级队列（Priority Queues）**：队列维护多层内部缓冲区（如 Priority 1-10），确保高优先级 VIP 任务打断常规批处理排队。

---

## 2 · 核心架构机制对照（消息队列 vs 事件总线/流）

| 维度 | 传统消息队列 (Message Queue) | 事件总线 / 日志流 (Event Bus / Stream) |
| :--- | :--- | :--- |
| **底层存储结构** | 动态内存 FIFO 队列 / 双向链表 / 环形缓冲 | 磁盘顺序只追加二进制日志段 (`.log` Commit Log) |
| **消息状态所有权** | **Broker 端强状态** (逐条维护 Ready / Unacked / Acked) | **Consumer 端无状态** (Broker 只存整型数字 Offset 游标) |
| **读取操作性质** | **破坏性读取 (Destructive Read)**，Ack 后立即物理删除 | **非破坏性读取 (Non-destructive Read)**，日志支持任意回滚与重放 |
| **消费模型与扇出** | **竞争消费者 (Competing Consumers)**，点对点单次交付 | **多组独立广播 (Multi-group Fan-out)**，单分区内严格保序 |
| **海量积压承受力** | **极差** (数百万堆积耗尽 RAM 触发磁盘 Page Swapping 导致吞吐暴跌) | **极高** (冷热数据自动落盘，积压 TB 级数据对写入吞吐几乎零影响) |

---

## 3 · 吞吐量级与性能基准（QPS & Latency）

不同实现技术在吞吐、单机容量与投递延迟上权衡各异：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                     Message Queue 典型性能阶梯                         │
├─────────────────────────┬──────────────────────┬───────────────────────┤
│ 组件类型                │ 典型 QPS 承载能力     │ 端到端延迟 / 特点     │
├─────────────────────────┼──────────────────────┼───────────────────────┤
│ RabbitMQ (AMQP)         │ 20,000 - 50,000 /node│ 1 - 5 毫秒 (极低延迟) │
│ AWS SQS Standard        │ 近乎无限 (自动分片)  │ 10 - 50 毫秒          │
│ AWS SQS FIFO            │ 300 - 30,000 (批处理)│ 20 - 60 毫秒 (保序)   │
│ Redis List / BullMQ     │ 50,000 - 100,000/node│ < 1 毫秒 (受内存容量限)│
└─────────────────────────┴──────────────────────┴───────────────────────┘
```

- **RabbitMQ**：基于 Erlang 轻量级 Actor 进程模型构建，内存状态机极度紧凑，路由能力强大，单机维持 $2\text{w} - 5\text{w}\text{ QPS}$，毫秒级投递；但当队列严重堆积超过 RAM 刷盘时，性能会出现断崖式下跌。
- **AWS SQS**：完全托管分布式哈希分片架构，标准队列自动水平弹性伸缩，吞吐可达数十万 QPS；但牺牲了严格保序（At-least-once，可能乱序和轻微重复）。
- **Redis List (`LPUSH` / `BRPOPLPUSH`)**：纯单线程内存操作，纳秒级入队、微秒级出队，QPS 达十万级；但所有任务受限于单机 RAM 空间，无分布式磁盘溢出能力。

---

## 4 · 架构决策：何时坚决选择 Message Queue 而非 Event Bus？

选择 **Message Queue（如 RabbitMQ、AWS SQS、Celery）** 而不是 **Event Bus（如 Kafka、Pulsar）**，核心取决于**你处理的是“命令（Command）”还是“事实（Fact）”**，以及**是否需要对单条消息进行精细化生命周期控制**。

### 4.1 · 坚决选用 Message Queue 的六大杀手级场景

#### 1. 异步任务分发与工作窃取（Task Worker / Work-Stealing）
- **典型场景**：视频转码渲染、PDF 账单生成、邮件短信外发、大语言模型离线批量评测。
- **核心逻辑**：此类载荷属于典型的“命令（Command）”，核心诉求是**“让空闲的 Worker 尽快把任务领走，且只能有一个 Worker 执行”**。
- **为什么不用 Event Bus**：
  - Event Bus（如 Kafka）按分区（Partition）顺序扫描。若某个视频转码耗时 30 分钟，该 Partition 就会被此任务**完全阻塞**，排在后面的数千个轻量任务全部停滞（**队头阻塞，Head-of-Line Blocking**）。
  - Message Queue 里的每条任务都是独立被 Worker 争抢的，长耗时任务绝不阻塞其他任务的并发获取。

#### 2. 原生消息优先级调度（Message Priority Queues）
- **典型场景**：企业付费 VIP 租户的编译构建任务需要实时插队，普通免费租户的任务按常规排队。
- **核心逻辑**：Message Queue（如 RabbitMQ Priority Queue）原生支持单条消息设置 `priority` 权重（1-255），Broker 内部维护堆结构保证权重高的消息优先出队派发。
- **为什么不用 Event Bus**：
  - Kafka 底层是只追加日志（Append-only Log），物理写入顺序固定。
  - 在 Kafka 中做优先级必须分裂为 `topic_vip`、`topic_normal` 多个 Topic，消费者在应用层手写权重轮询拉取代码，架构冗余且极难控制精确抢占。

#### 3. 延时消息与定时触发（Delayed & Scheduled Messages）
- **典型场景**：电商订单 30 分钟未支付自动关闭、优惠券到期前 2 小时提醒、任务失败后的**指数退避重试**（10s -> 1m -> 5m）。
- **核心逻辑**：传统 MQ（如 AWS SQS DelaySeconds、RabbitMQ 延迟插件 / 死信 TTL）原生支持设定消息在未来某个绝对/相对时间点才转为 `Ready` 对 Worker 暴露。
- **为什么不用 Event Bus**：
  - Kafka 无法原生“隐藏”某条消息到指定时间再读。
  - 若在 Kafka 实现延时，必须额外维护时间轮中间件，或者按延时阶梯创建数十个专用 Topic，运维与架构成本极其高昂。

#### 4. 细粒度单条失败重试与死信隔离（Per-Message Retry & DLQ）
- **典型场景**：支付回调处理、第三方 API 网关调用的故障隔离。
- **核心逻辑**：假设处理第 100 条消息时下游接口超时报错。
  - **在 Message Queue 中**：Worker 可以仅对该消息返回 `NACK`，该消息重新入队或移入死信队列（DLQ），Worker 可以立刻无缝处理第 101 条。
  - **在 Event Bus 中**：Kafka 的消费进度由单调递增的整型 `Offset` 控制。如果第 100 条处理失败且不允许丢数据，消费者的 Offset 就**无法向前推进**，整个 Partition 的后续消费全部停摆；否则必须在应用层搭建旁路重试流。

#### 5. 消费者高度动态弹性扩缩容（Dynamic Worker Scaling）
- **典型场景**：流量脉冲时，后台 Worker 容器需要在 1 分钟内从 10 个弹性扩展至 500 个。
- **核心逻辑**：
  - **Kafka 的硬上限**：消费并行度被**分区数（Partition Count）严格锁死**。若 Topic 仅有 32 个分区，即便启动 100 个 Worker Pod，也永远只有 32 个 Pod 能分到分区干活，其余 68 个处于饥饿等待状态。
  - **Message Queue 的弹性**：单队列可挂载成百上千个 Worker 实例进行竞争抢占，扩缩容与存储分片彻底解耦。

#### 6. 复杂的内容路由与交换拓扑（Complex Content-Based Routing）
- **典型场景**：多租户企业级系统集成、动态按 Header 路由到不同微服务集群。
- **核心逻辑**：RabbitMQ 具备成熟的 **Exchange 交换机模型**（Direct, Topic, Headers, Fanout）。生产者只管将消息丢给交换机，复杂的通配符绑定（如 `order.us.*`）由 Broker 内核就地完成。Kafka 缺乏交换机路由层，必须由生产者在代码中硬编码目标 Topic。

---

### 4.2 · 架构选型决策树

在系统设计面试或真实架构评审中，可按以下标准路径快速决断：

```text
                      需要持久化保留数据，并且需要多方独立消费/回溯？
                                    │
                    ┌───────────────┴───────────────┐
                   YES                              NO
                    │                               │
             【使用 Event Bus】                单条消息耗时较长 / 差异极大？
            (Kafka, Pulsar)                         │
                                            ┌───────┴───────┐
                                           YES              NO
                                            │               │
                                     【使用 Message Queue】 需要 > 100k QPS 极限吞吐？
                                    (RabbitMQ, SQS)         │
                                                    ┌───────┴───────┐
                                                   YES              NO
                                                    │               │
                                             【使用 Event Bus】 【使用 Message Queue】
```

---

### 4.3 · 在大规模 CI/CD 系统中的工程分工

在现代企业级 CI/CD 平台（如 GitHub Actions、GitLab CI）中，两者协同是工业界的标准实践：

1. **适合使用 Event Bus（Kafka）的路径：**
   - **`Controller -> Push Service -> UI Dashboard`**：
   - 流水线与 Job 的状态变更（如 `JobStarted`、`JobCompleted`）是不可逆的**“客观事实（Fact）”**。
   - 该事件不仅要通过 WebSocket 推送到前端 UI，还要被审计日志服务持久化、被 Prometheus 监控大盘聚合。使用 Kafka 可以天然支持**多组独立广播（Fan-out）**且互不干扰。
2. **坚决使用 Message Queue（或 DB+Pull+Lease）的路径：**
   - **`Controller / Gateway -> Runner Platform`**：
   - 派发具体的编译、打包、测试任务。每个任务是一个明确的**“操作命令（Command）”**。
   - 任务耗时差异极大（Lint 检查 5 秒，完整集成测试 45 分钟）。
   - 必须支持**按机器规格插队抢占（GPU / macOS Runner）、超时保护与失败重试**。若在此处使用 Kafka 会导致长任务卡死整条分区的队头阻塞，必须采用 **Message Queue 竞争消费者模型**。

---

### 4.4 · 架构禁忌与反模式（When NOT to Use）
- **禁忌 1：多业务系统独立消费数据副本**：MQ 消息被一个 Worker 消费确认后即被物理销毁，其他独立微服务无法二次消费。需要多团队共享数据时应使用 **Event Bus** 或 **Kafka**。
- **禁忌 2：需要时间回溯与全量历史重放**：MQ 无法像日志一样重置 Offset 重新消费上个月的历史数据。

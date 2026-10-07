# Wiki · 大数据流处理演进与架构权衡 (Stream Processing Evolution)

所属类别：Wiki 模式库 · 知识点  
关联模式：[[SystemDesignWiki Flink|Flink 实时流计算]] · [[SystemDesignWiki Kafka|Kafka 分区日志]] · [[SystemDesignWiki NoSQL Streaming|NoSQL 变更流]] · [[SystemDesignWiki Event Bus|事件总线]] · [[SystemDesign06 Async Messaging Systems|消息系统]]

---

## 1 · 分布式计算范式演进全景 (The Paradigm Shift)

分布式大数据计算架构经历了三次代际飞跃：从**静态磁盘批处理**，到**内存微批处理**，最终演进为**真正的原生事件驱动流处理**。

```text
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ 第一代：磁盘批处理       │      │ 第二代：内存微批处理     │      │ 第三代：原生事件驱动流   │
│ MapReduce / Hadoop      │ ───► │ Spark / Spark Streaming │ ───► │ Flink / Continuous Stream│
│ (2004 - 2010)           │      │ (2010 - 2015)           │      │ (2015 - 至今)           │
│ 磁盘落盘、高延迟、静态批 │      │ 内存 DAG、秒级微批       │      │ 逐条处理、毫秒延迟、状态化│
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

---

## 2 · MapReduce 的物理瓶颈与 Spark 的内存 DAG 革命

### 2.1 · MapReduce 的物理 I/O 与写屏障瓶颈 (I/O & Write Bottlenecks)
Google 提出的 MapReduce 及开源实现 Apache Hadoop 解决了海量数据的离线分布式计算，但其设计被硬编码为严格的两阶段计算：$\text{Map} \to \text{Shuffle} \to \text{Reduce}$。

```text
[Job 1: Map] ──► (溢写写本地磁盘) ──► (网络 Shuffle) ──► [Job 1: Reduce] ──► (3副本写 HDFS 磁盘)
                                                                                  │
┌─────────────────────────────────────────────────────────────────────────────────┘
▼
[Job 2: Map] ──► (溢写写本地磁盘) ──► (网络 Shuffle) ──► [Job 2: Reduce] ──► (3副本写 HDFS 磁盘)
```

1. **阶段间强制落盘阻断 (Materialization Barrier)**：
   - 每个 Map 任务的中间输出必须排序并**溢写到本地磁盘 (Spill to Disk)**。
   - 每个 Reduce 任务的输出必须持久化并经过网络写入 **HDFS（默认 3 副本）**，伴随昂贵的数据序列化、磁盘写入与跨节点网络传输。
2. **多轮迭代计算的灾难性惩罚 (Iterative Algorithms Penalty)**：
   - 在图计算（如 PageRank）、机器学习（如梯度下降、K-Means）或多表关联中，算法包含 $K$ 次循环迭代。
   - MapReduce 必须启动 $K$ 个完全独立的作业。数据在每轮迭代间被无意义地写回 HDFS，再重新从 HDFS 读取、反序列化。
   - **I/O 占比瓶颈**：在迭代作业中，纯 CPU 计算时间占比往往 $<10\%$，而磁盘与网络 I/O 耗时占比 $>90\%$。
3. **粗粒度进程调度与 JVM 启动开销**：
   - 早期 MR1 的 JobTracker/TaskTracker 模型为每个任务独立申请进程（JVM 频繁冷启动），资源无法跨 Stage 复用。

---

### 2.2 · Spark 的革新：RDD 抽象与内存 DAG 引擎
Apache Spark 针对 MapReduce 的物理 I/O 瓶颈，重构了分布式计算抽象：

```text
[Data Source]
      │
      ▼ (窄依赖 Narrow: 内存流水线，零 Shuffle)
┌────────────────────────────────────────────────────────┐
│ RDD A (map) ──► RDD B (filter) ──► RDD C (flatMap)     │ Stage 1 (Pipeline)
└────────────────────────────────────────────────────────┘
      │
      ▼ (宽依赖 Wide: Shuffle 边界，数据分区重排)
┌────────────────────────────────────────────────────────┐
│ RDD D (reduceByKey) ──► RDD E (mapValues)              │ Stage 2
└────────────────────────────────────────────────────────┘
      │
      ▼ (内存常驻 Cache / Persist)
   [Memory Pool (RAM)] <--- 供下一轮迭代直接复用，彻底消灭 HDFS 写入
```

1. **RDD (Resilient Distributed Datasets) 核心抽象**：
   - 分布式、不可变、逻辑分区的记录集合。
2. **血缘依赖与粗粒度容错 (Lineage / Provenance)**：
   - Spark 不靠跨网络数据复制容错，而是记录每个 RDD 是通过何种算子从上游派生而来的**血缘 DAG 图**。
   - 一旦某个分区数据在节点宕机时丢失，调度器仅需回溯血缘图，**仅重算该丢失分区的上游切片**，消除了高昂的阶段间磁盘快照开销。
3. **依赖划分与内存流水线 (Narrow vs Wide Dependencies)**：
   - **窄依赖 (Narrow Dependency)**：父 RDD 的每个分区至多被子 RDD 的一个分区使用（如 `map`, `filter`）。多个连续窄依赖会被执行器（Executor）熔合成同一个 Stage，以**流水线（Pipelining）**方式全在 CPU 缓存/内存中顺序推进，零网络与磁盘 I/O。
   - **宽依赖 (Wide Dependency)**：父 RDD 的分区会被子 RDD 的多个分区依赖（如 `groupByKey`, `reduceByKey`, `join`），构成 Stage 的切分边界，仅在必须发生数据重新分布时才触发 Shuffle。
4. **内存复用**：
   - 迭代算法通过 `rdd.persist(StorageLevel.MEMORY_ONLY)` 将中间状态常驻于 Executor 进程的堆内/堆外内存，将迭代任务效率提升 $10\times - 100\times$。

---

## 3 · Spark Micro-Batch（微批处理）架构设计及其根本局限

为了复用 Spark 成熟的批处理 DAG 调度器与容错机制，Spark Streaming（DStreams）与 Spark Structured Streaming 采用了**微批处理（Micro-Batching）**设计。

```text
Continuous Input Stream:  ... ● ● ● ● ● ● ● ● ● ● ● ● ● ● ...
                               │           │           │
           [时间切片: 500ms]    ▼           ▼           ▼
                           ┌───────┐   ┌───────┐   ┌───────┐
                           │Batch 1│   │Batch 2│   │Batch 3│
                           └───────┘   └───────┘   └───────┘
                               │           │           │
                               ▼           ▼           ▼
                      [Spark DAG Job] [Spark DAG Job] [Spark DAG Job]
```

### 3.1 · 微批处理的核心哲学
- **“流是极小批次的特例（Streaming as Discretized Batches）”**：连续无界流被人为划分为离散的时间小片段（如 $100\text{ms} - 500\text{ms}$）。
- 每一个小批次被转化为一个标准 RDD / Dataset，触发一次常规的 Spark Batch DAG 作业。

---

### 3.2 · 微批架构的四大根本局限 (Inherent Flaws)

#### 1. 物理延迟下限瓶颈 (Latency Floor: 无法突破的数十毫秒墙)
微批计算的端到端延迟由三个部分严格相加：
$$T_{\text{latency}} = T_{\text{batch\_window}} + T_{\text{driver\_scheduling}} + T_{\text{execution\_barrier}}$$
- **调度开销**：每个微批次到来时，Driver 必须为每个 Partition 实例化 Task、序列化并分发至 Worker 线程池。
- **任务栅栏（Barrier Synchronisation）**：一个批次必须等待该批次中执行最慢的长尾 Task（Straggler）完成后才能进入下一个 Stage。
- 物理现实决定了其延迟极限普遍在 $100\text{ms} - 500\text{ms}$，在金融高频量化、工业实时控制等亚毫秒（Sub-millisecond）场景下在架构层面直接失效。

#### 2. 事件时间 (Event Time) 与迟到数据 (Late Events) 的语义割裂
- 微批次划分依据的是**系统处理时间 (Processing Time)**，即事件到达流处理框架的时间点。
- 但现实世界中数据产生于终端，受弱网、断网重连影响，包含不可预测的网络延迟：
  $$t_{\text{event}} \ll t_{\text{processing}}$$
- 当一条 1 小时前的迟到事件（Late Event）进入当前批次时，它属于历史时间窗口。微批架构必须拉取历史数十个批次持久化的状态进行跨批次修改，极易导致状态膨胀与低效的全状态覆写。

#### 3. 窗口语义与批次切片错位 (Window Misalignment)
- 假设定义一个滑动步长为 $30\text{秒}$ 的滑动窗口（Sliding Window），而微批配置为 $7\text{秒}$。
- 由于 $30$ 不能被 $7$ 整除，批次边界无法天然对齐窗口边界，框架不得不维护复杂的跨微批状态缓冲合并逻辑，造成内存开销翻倍。

#### 4. 高频任务调度对 Driver 与 GC 的冲击
- 假设微批窗口压缩至 $50\text{ms}$，系统每天将生成：
  $$\frac{86,400\text{ s}}{0.05\text{ s}} \approx 1,728,000\text{ 个独立 Batch 作业}$$
- Driver 节点面临百万级 DAG 构建与 RPC 调度开销，频繁的对象创建引发 JVM 老年代内存碎片化，诱发严重的垃圾回收（GC Pause），从而导致吞吐与延迟剧烈抖动。

---

## 4 · 现代流处理核心理论体系：Dataflow 模型

Google 在 2015 年发表的论文 *The Dataflow Model* 奠定了现代原生流处理的基石，将计算解构为回答四个独立正交的问题：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        The Dataflow Matrix                             │
├───────────────────┬────────────────────────────────────────────────────┤
│ 1. What? (计算什么)│ Transformations (map, filter, sum, count)          │
│ 2. Where? (何时发生)│ Windowing in Event Time (Tumbling, Sliding, Session)│
│ 3. When? (何时触发)│ Watermarks & Triggers in Processing Time           │
│ 4. How? (如何修正) │ Accumulating, Retracting, Discarding               │
└───────────────────┴────────────────────────────────────────────────────┘
```

### 4.1 · 三种时间概念 (Time Semantics)
1. **事件时间 (Event Time)**：事件在源端真实物理世界产生时自带的时间戳（如传感器传感器读数时间、手机客户端打点时间）。**这是唯一保证跨节点重放计算结果具备确定性（Determinism）的时间语义**。
2. **摄入时间 (Ingestion Time)**：事件进入流系统消息总线（Kafka / Pulsar）被持久化追加至日志时的时间戳。
3. **处理时间 (Processing Time)**：流处理算子所在服务器的本地系统时钟。无序、无确定性，但开销最小。

---

### 4.2 · 水位线机制 (Watermarks)
水位线是衡量事件时间推进的单调递增时钟标记 $W(t)$。

```text
Data Flow Stream:
... [e: 10:02] [e: 10:01] ───► [Watermark(10:00)] ───► [e: 09:59] [e: 09:58] ...
                                      │
                                      ▼
             断言: "系统有较高置信度认为, 之后不会再有 Event Time <= 10:00 的数据到达"
             动作: 触发 [09:00 - 10:00] 窗口执行聚合计算并向下游输出
```

- **数学含义**：若算子接收到水位线 $W(t)$，则意味着该算子假定未来到达的所有事件其事件时间 $t_e > W(t)$。
- **乱序延迟容忍度 (Bounded Out-of-Ordnerness)**：
  $$W(t) = \max_{e \in \text{received}}(t_e) - \Delta t_{\text{delay}}$$
  其中 $\Delta t_{\text{delay}}$ 为工程允许的最大无序网络抖动容忍预算。
- **迟到事件 (Late Events) 处理**：
  若事件在其所属窗口关闭后才到达（$t_e \le W(t)$），系统提供三级容灾：
  1. *直接丢弃 (Discard)*。
  2. *更新触发 (Accumulating & Retracting)*：重新打开窗口重新发送修正结果。
  3. *侧输出流 (Side Output)*：将迟到数据路由至独立队列（如 Kafka 死信流），由离线对账作业单独修正。

---

### 4.3 · 窗口模型形态 (Window Models)

| 窗口类型 | 划分逻辑 | 切片特征 | 典型业务场景 |
|---|---|---|---|
| **滚动窗口 (Tumbling)** | 固定长度，无重叠 | $[0, 10), [10, 20), [20, 30)$ | 统计每分钟服务器请求数、每小时全网交易额 |
| **滑动窗口 (Sliding)** | 固定长度，固定滑动步长 | $[0, 10), [5, 15), [10, 20)$ | 最近 10 分钟交易量（每 10 秒刷新一次） |
| **会话窗口 (Session)** | 动态长度，依据非活跃间隔 (Gap) 驱动切分 | 两次事件间隔超过 $T_{\text{gap}}$ 即关闭前一窗口 | 用户移动端 APP 使用行为追踪、页面交互会话分析 |

---

## 5 · 流处理架构范式与主流引擎选型对比

```text
┌────────────────────────────────────────────────────────────────────────┐
│ 架构选型深度对比                                                        │
├─────────────────────┬──────────────────┬───────────────────────────────┤
│ 维度                │ Spark Streaming  │ Apache Flink                  │
│                     │ (Structured)     │ (Stateful Stream Engine)      │
├─────────────────────┼──────────────────┼───────────────────────────────┤
│ **处理范式**        │ 微批处理 (Micro-batch) │ 原生事件驱动连续流 (Continuous)│
│ **单条处理延迟**    │ $100\text{ms} - 500\text{ms}$  │ **$1\text{ms} - 10\text{ms}$ (亚毫秒级)**   │
│ **吞吐能力**        │ 极高 (批量调度与高效压缩) │ 极高 (基于分区的本地内存状态)  │
│ **哲学认知**        │ 流是批的特例     │ **批是流的特例 (有界流)**      │
│ **状态后端**        │ HDFS / StateStore│ RocksDB / 堆内存 (增量快照)    │
│ **容错机制**        │ 批次重算 + Checkpoint │ 异步屏障快照 (Chandy-Lamport)  │
│ **背压控制**        │ PID 速率估算调节 │ 基于信用的流控 (Credit-based)  │
│ **SQL / 批流一体**  │ 成熟度极高 (Catalyst) │ 持续完善 (Blink Planner)       │
└─────────────────────┴──────────────────┴───────────────────────────────┘
```

### 5.1 · Flink 的核心破局点：批是流的特例 (Batch as a Special Case of Streaming)
Flink 彻底颠覆了 Spark 的认知逻辑：
- 现实世界的客观世界数据本身就是持续产生的**无界数据流（Unbounded Stream）**。
- 所谓的批处理，不过是人为限制了起始与终止边界的**有界数据流（Bounded Stream）**。
- 因此，流计算引擎应当作为底层核心基石，批处理只是不需要维护持久状态与水位线的特殊流。

### 5.2 · 状态管理与端到端 Exactly-Once 保证
1. **本地状态持久化 (Keyed State)**：
   - Flink 将每个 Key 的状态（Counter、ListState、MapState）保存在 TaskManager 节点的本地内存或内嵌式 RocksDB 中。
   - 算子读取/更新状态为纯本地内存访问，耗时微秒级，避免每次更新跨网络打向远程 Redis / 数据库。
2. **异步屏障快照 (Asynchronous Barrier Snapshotting, ABS / Chandy-Lamport)**：
   - JobManager 周期性向数据源注入带有递增 ID 的 **Checkpoint Barrier（检查点屏障）**。
   - Barrier 随同普通事件在数据流中流动，不阻断正常事件的并发处理。
   - 当算子对齐 Barrier 后，异步将本地状态增量拷贝至远程持久存储（S3/HDFS）。
3. **端到端 Exactly-Once (End-to-End EOS)**：
   - 仅自身状态支持 Exactly-Once 不够，必须达成全链路三位一体：
     $$\text{EOS} = \text{可回放 Source (Kafka Offset)} + \text{引擎状态快照 (ABS)} + \text{两阶段提交 Sink (2PC)} $$

---

## 6 · 工业界架构演进：从 Lambda 到 Kappa 再到流批一体 Lakehouse

```text
[Lambda 架构]
                        ┌──► 批处理层 (Hadoop/Spark Batch) ──► 批视图 (HDFS/Hive) ──┐
                        │                                                           │
Raw Events (Kafka) ─────┤                                                           ├──► 服务查询层 (Merge)
                        │                                                           │
                        └──► 流处理层 (Storm/Spark Streaming) ──► 实时视图 (HBase) ──┘

[Kappa 架构]
Raw Events (Kafka / Log) ──► 统一流处理层 (Flink / Spark) ──► 实时服务存储 (ClickHouse/Redis)
        ▲
        └────── 需要重算历史业务逻辑时，直接回放 Kafka Offset 重跑流作业

[流批一体 Lakehouse (现代架构)]
Kafka Realtime Stream ──► Flink 流写入 ──┐
                                         ├──► 统一开放数据湖格式 (Apache Iceberg / Paimon)
Batch Historical Data ──► Spark 批量写入 ──┘
```

1. **Lambda 架构痛点**：
   - 维护两套完全隔离的代码库（批处理 MapReduce/Hive + 实时流 Storm/Flink），同一套业务逻辑写两遍。
   - 离线计算与实时计算指标口径极易发生隐蔽不一致，运维与故障对账成本高昂。
2. **Kappa 架构的革命**：
   - 彻底废除批处理层，全链路仅保留流处理引擎（Flink）。
   - 历史数据直接依靠可回放日志系统（Kafka / Pulsar）或者分布式对象存储中的追加日志。当逻辑变更时，启动新流任务从 Offset 0 回放全量历史数据，产出新物化视图后切换流量，彻底消灭双代码库。
3. **现代 Lakehouse 流批一体演进**：
   - 结合基于 Apache Iceberg / Apache Paimon 的新一代存储底座，流引擎以毫秒级写入并生成快照元数据，批引擎以列式格式高效扫描，最终在存储与计算两端实现真正的大一统。

# System Design 00 · 全局架构体系与量化估算基准

课程位置：本篇（全局总纲与量化基准） → [[SystemDesign01 Stateless Service|01 无状态服务]]

系统设计不是零散组件的随意堆砌，而是**在明确的业务不变量、SLA 可用性目标与物理资源硬边界约束下，沿着系统瓶颈演进做出的权衡取舍**。

面对任何工业级架构，脱离**量化估算（Numbers & Sizing）**的高谈阔论都是空中楼阁。本篇作为全专栏的统领总纲，确立分布式系统的**四大核心物理量化基准**，并绘制完整的架构演进路线图。

---

## 1 · 架构师必须掌握的核心量化基准 (The Critical Numbers)

### 1.1 · Application Server：什么时候算高 QPS？

单台无状态应用服务器（以 4 核 8GB / 8 核 16GB 的通用计算 Pod 为基准）的 QPS 承载能力存在明确的工业级分水岭：

| 单机 QPS 级别 | 系统运行状态 | 典型业务场景与工程特征 |
|---|---|---|
| **10 QPS / 机** | 极低负载 | 完全可以接受；多见于内部管理后台、低频微服务或执行大计算量的批处理接口。 |
| **100 QPS / 机** | 完全休闲 | 绝大多数日常中小型业务 API 的稳态运行区间，CPU 利用率通常 $< 15\%$。 |
| **500 QPS / 机** | **正常健康工作区间** | 典型微服务常规承载点（包含轻量 JSON 反序列化、权限校验、1~2 次简单 DB 查询与网络 IO）。 |
| **2,000 QPS / 机** | **优化警戒线** | 开始必须进行性能 Profiling；需关注垃圾回收（GC 停顿）、线程池排队、数据库连接池争用与 CPU 上下文切换。 |
| **> 2,000 QPS / 机** | **偏高负载** | 普通业务逻辑已达物理极限；必须引入二级缓存、异步批处理或优化序列化库。 |
| **> 10,000 QPS / 机** | **极限高吞吐** | **极少普通业务能达到**。必须满足：业务逻辑极其简单、纯内存强缓存、全异步非阻塞 IO（epoll / netpoll）、长连接复用、零重型 ORM 与反射开销。 |

> [!IMPORTANT]
> **系统容量饱和的通用工程判定准则（The +30% Knee-of-the-Curve Rule）**：
> 在压力测试或线上监控中，当**请求 QPS 仅提升 30%** 时，如果观察到：
> 1. p95 / p99 延迟出现**非线性剧烈抬升**（拐点爆发）；
> 2. CPU 消耗急剧飙升；
> 3. 线程池或请求排队队列（Request Queue）显著变长；
> 
> 则表明系统已越过膝点（Knee Point），内部计算或 IO 资源已处于排队饱和边缘。此时单靠横向扩容应用节点已无法解决问题，瓶颈已下沉至持久化层！

---

### 1.2 · Cache：QPS 与容量基准 (Numbers of Cache)

#### 1. 远程缓存 QPS (Cache QPS)
远程分布式缓存（Redis / Memcached）执行简单 `GET` / `SET` 操作：
- **网络往返时间（RTT）**：同机房内网络往返约 $\approx 0.2 - 1.0\text{ ms}$；
- **单机物理吞吐极限**：单实例通常可承载 **数万至十万 QPS** 级别（主要受 CPU 单核处理能力与网卡包转发速率 PPS 限制）。

**基于 CPU 处理时间的精确定量估算公式**：
$$\text{QPS} \approx N_{\text{cores}} \times \frac{1000}{t_{\text{cpu}}} \times u$$
其中：
- $N_{\text{cores}}$ 为分配给缓存的有效计算核心数；
- $t_{\text{cpu}}$ 为单次请求在 CPU 上消耗的纯计算时间（ms）；
- $u$ 为目标安全 CPU 利用率（通常取 $0.7 - 0.8$，预留 $20\% - 30\%$ 突发缓冲）。

> **算例验证**：4 核节点，平均每个请求占用 CPU 时间 $t_{\text{cpu}} = 0.1\text{ ms}$，安全利用率 $u = 0.8$：
> $$\text{QPS} \approx 4 \times \frac{1000}{0.1} \times 0.8 = 32,000\text{ QPS}$$
> 单台 4 核 Redis 实例在稳态下的安全吞吐线约为 **3.2 万 QPS**。

#### 2. 缓存容量：1 GB 内存到底能存多少 Key？
- **理论上限推导**：
  若单条 Key-Value 记录及其元数据体积约为 $200\text{ B}$：
  $$\text{Keys} = \frac{10^9\text{ Bytes}}{200\text{ Bytes}} \approx 5,000,000\text{ (约 500 万 Key)}$$
- **生产级保守基准（考虑碎片与元数据）**：
  在实际生产环境中，由于内存分配器（jemalloc）的内存碎片、`dictEntry` 与 `redisObject` 的指针开销，以及必须预留的 $30\%$ 内存安全水位（防止 BGSAVE 或 AOF 重写时写时复制 CoW 导致 OOM）：
  $$\mathbf{1\text{ GB 内存} \approx 200\text{ 万} - 300\text{ 万稳定 Key}}$$

> [!TIP]
> **缓存引入原则**：
> 当系统呈现 **读多写少（Read/Write $\ge 10:1$）** 且 **接口 QPS 达到数百以上**，同时 **数据库 CPU/IOPS 出现吃力** 时，引入缓存是性价比最高、改造成本最低的第一动作。

---

### 1.3 · Database & Storage：关系型数据库与存储的物理极限

| 组件 / 操作 | 典型单机安全 QPS | 物理瓶颈根源 | 架构演进与突破阈值 |
|---|---|---|---|
| **关系型 DB 读 (MySQL/PG)** | $1,000 - 5,000\text{ QPS}$ | 内存 Buffer Pool 局部性、磁盘随机读 IOPS | 超过 5,000 读 QPS $\to$ 必须增加只读从库（X 轴副本）或前置 Redis |
| **关系型 DB 写 (MySQL/PG)** | $500 - 2,000\text{ QPS}$ | WAL 顺序落盘（fsync）、行锁争用与事务提交延迟 | 超过 2,000 写 QPS $\to$ 必须引入 MQ 异步削峰，或水平分片（Z 轴 Sharding） |
| **单表数据行数上限** | $\approx 2,000\text{ 万行}$ | 3 层 B+ 树索引能够支撑约 2000 万行；超过后树高增至 4 层，每次查询增加 1 次物理磁盘 IO | 超过 2000 万行或单表体积破 50GB $\to$ 必须执行历史冷热分离或分库分表 |
| **对象存储 (S3/MinIO)** | 单前缀 3,500 写 / 5,500 读 QPS | HTTP 吞吐与分块元数据管理 | 大文件必须客户端直传（Pre-signed URL），禁止穿透 API 服务器内存 |

---

## 2 · 分布式系统设计体系全景图 (The Two Pillars)

```system-design-overview-visual
```

全专栏结构重塑为两大板块：**实战 Case（端到端工业级系统实战）** 与 **Wiki 模式库（原子化设计模式与组件基准）**：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        System Design 架构知识体系                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
【板块一：实战 Case (End-to-End Cases)】               【板块二：Wiki 模式库 (Patterns & Wiki)】
07 图片分享与 Feed 流 (推拉混合、Timeline 聚合)        · 系统设计解题框架 (FR+Non-FR+QPS+Diagram+Deep Dive)
08 异步 LLM 强化学习平台 (Actor-Learner 解耦)         · 全局量化估算基准 (Application/Cache/DB 物理极限)
10 极端瞬态高并发秒杀系统 (Redis 拦截、防超卖)          · 无状态架构与状态外置 (计算状态分离与优雅下线)
11 海量移动推送与通知平台 (Outbox 零丢失、队列削峰)      · 虚拟化与容器隔离 (Namespaces/Cgroups 运行时边界)
12 分布式填字游戏求解器 (位并行 CSP、CAS 竞态仲裁)      · Kubernetes (Pod 生命周期、控制器模型与调度)
                                                       · Redis (Cache-Aside、防穿透/击穿/雪崩与分布式锁)
                                                       · 数据库存储范式与分片 (Scalability Cube、分片键设计)
                                                       · 分布式存储系统 (对象存储 S3 与 LSM-Tree 追加写)
                                                       · 一致性哈希 (虚拟节点均衡、最小化数据迁移)
                                                       · 高频术语与核心定理 (CAP/BASE/Little's Law 对比矩阵)
                                                       · Event Bus (事件总线、ECST 与声明式过滤)
                                                       · Message Queue (工作队列、租约与毒丸隔离)
                                                       · NoSQL + Streaming (内建 CDC、流表二象性)
                                                       · Kafka (分布式分区日志、高吞吐与一致性)
                                                       · Transactional Outbox (解决 DB 与 MQ 双写困境)
                                                       · 控制面与数据面解耦 (K8s/Envoy 生存不变性)
                                                       · Pull vs Push (推拉模型、写扩散与读扩散权衡)
                                                       · Idempotency (幂等性意图键与实现模式)
                                                       · NewSQL (计算存储分离与分布式事务选型)
```

---

## 3 · 模块索引与解决的核心工程瓶颈

### 3.1 · 实战 Case (End-to-End Deep Dives)

1. **[[SystemDesign07 Photo Sharing Feed|Case 07 · 图片分享与 Feed 流]]**
   - *架构全链路*：读写比 100:1 的海量社交平台。大文件客户端直传对象存储，写扩散（Push）与读扩散（Pull）混合架构权衡，Timeline 预热缓存聚合。
2. **[[SystemDesign08 LLM Async RL Platform|Case 08 · 异步 LLM 强化学习平台]]**
   - *架构全链路*：前沿现代 AI 基础设施。Rollout 生成与 Trainer 梯度反向传播异步解耦，高吞吐参数更新与大规模分布式数据管道。
3. **[[SystemDesign10 Flash Sale|Case 10 · 极端并发秒杀系统]]**
   - *架构全链路*：微秒级突发瞬态洪峰。Redis 准入拦截、MQ 队列削峰、数据库行级悲观锁事务防超卖。
4. **[[SystemDesign11 Notification System|Case 11 · 海量移动推送与通知平台]]**
   - *架构全链路*：单日十亿级推送。Transaction Outbox 零丢失投递、Dispatcher 批量切片与偏好过滤、Durable Queues 优先级隔离削峰、APNs/FCM HTTP/2 连接池与 At-least-once 幂等去重闭环。
5. **[[SystemDesign12 Crossword Solver|Case 12 · 分布式填字游戏求解器]]**
   - *架构全链路*：NP-Complete 约束满足与分布式 DFS 搜索调度。静态复杂度初筛与动态看门狗双轨准入，单机位并行索引极速消化 90%+ 流量；长尾任务接入分布式 DFS 算力池，基于最浅选择点粗粒度工作窃取与 CAS 竞态胜出。

### 3.2 · Wiki 模式与知识点 (Design Patterns & Atomic Wiki)

1. **[[SystemDesign01 Stateless Service|Wiki · 系统设计解题框架 (FR + Non-FR + QPS + Diagram + Deep Dive)]]**
   - *核心瓶颈*：系统设计面试与方案推演的结构化全流程。
   - *主线逻辑*：5 阶段黄金推进链条，以高并发分布式短链系统为例，落地需求划界、容量精算、分层架构、三大核心 Deep Dive 权衡与端到端时序追踪。
2. **[[SystemDesignWiki Stateless Architecture|Wiki · 无状态架构与状态外置 (Stateless Architecture)]]**
   - *核心瓶颈*：计算层无状态扩展与分布式状态外置。
   - *主线逻辑*：状态分类矩阵、三大外置支柱（Session/Blob/Workflow）、幂等性意图键、进程优雅停机与双探针。
3. **[[SystemDesign00 Overview|Wiki · 全局量化估算基准 (System Architecture Numbers)]]**
   - *核心瓶颈*：系统容量规划与瓶颈膝点量化。
   - *主线逻辑*：Application Server、Cache、Database、Storage 四大物理量化基准与 Knee-of-the-curve 判定。
4. **[[SystemDesign01B Virtualization Containers|Wiki · 虚拟化与容器隔离 (Virtualization & Containers)]]**
   - *核心瓶颈*：应用如何在物理服务器上轻量、安全、隔离地密集部署。
   - *主线逻辑*：Hypervisor vs Linux Namespaces + Cgroups，运行时资源与安全边界。
5. **[[SystemDesign01C Kubernetes|Wiki · Kubernetes (容器编排与集群调度)]]**
   - *核心瓶颈*：超大规模容器集群自动化部署、服务发现与弹性自愈。
   - *主线逻辑*：Pod 控制器模型、声明式 API、Service 虚拟 IP 网络与 HPA 水平伸缩。
6. **[[SystemDesign01D Redis|Wiki · Redis (内存缓存与协调)]]**
   - *核心瓶颈*：磁盘 IOPS 瓶颈导致的高并发读延迟。
   - *主线逻辑*：Cache-Aside 模式、缓存穿透/击穿/雪崩三大防御机制、分布式租约锁。
7. **[[SystemDesign02 Database Paradigms|Wiki · 数据库存储范式与分片 (Database Paradigms & Sharding)]]**
   - *核心瓶颈*：单机数据库写入吞吐与磁盘存储容量枯竭。
   - *主线逻辑*：ACID 与隔离级别、Scalability Cube 三维扩展、分片键设计法则与跨分片关联消除。
8. **[[SystemDesign04 Storage Systems|Wiki · 分布式存储系统 (Storage Systems)]]**
   - *核心瓶颈*：海量非结构化二进制大文件存储与索引追加吞吐。
   - *主线逻辑*：块/文件/对象存储 S3 选型、LSM-Tree 追加写与元数据解耦。
9. **[[SystemDesign09 Consistent Hashing|Wiki · 一致性哈希 (Consistent Hashing)]]**
   - *核心瓶颈*：缓存与存储集群节点动态伸缩时全网哈希漂移。
   - *主线逻辑*：哈希环设计、虚拟节点（Virtual Nodes）消除倾斜、最小化数据迁移。
10. **[[SystemDesign99 Glossary|Wiki · 高频术语与核心定理 (Glossary & Laws)]]**
    - *全局检索*：CAP、BASE、Little's Law 与分布式核心概念对比矩阵速查。
11. **[[SystemDesignWiki Event Bus|Wiki · Event Bus (事件总线与事件驱动架构)]]**
    - *核心瓶颈*：跨微服务领域事件分发、复杂订阅规则过滤与读风暴防范。
    - *主线逻辑*：Pub/Sub 广播范式、Event-Carried State Transfer (ECST)、声明式 JSON 模式过滤；双层扇出范式（SNS-to-SQS 拓扑隔离与 Dispatcher 微批分发）。
12. **[[SystemDesignWiki Message Queue|Wiki · Message Queue (消息队列与点对点工作队列)]]**
    - *核心瓶颈*：高负载异步耗时任务编排、突发瞬态流量缓冲与慢系统保护。
    - *主线逻辑*：竞争消费者（1-to-1 抢占）、租约可见性超时（Visibility Timeout）、两阶段确认。
13. **[[SystemDesignWiki NoSQL Streaming|Wiki · NoSQL + Streaming (NoSQL 变更流与事件驱动)]]**
    - *核心瓶颈*：应用层双写数据不一致、多异构存储视图实时同步与海量吞吐持久化。
    - *主线逻辑*：存储引擎内建 CDC 捕获、流表二象性（$S \iff T$）、CQRS 物化视图同步。
14. **[[SystemDesignWiki Kafka|Wiki · Kafka 核心机制与实战 (Distributed Commit Log)]]**
    - *核心瓶颈*：高吞吐低延迟分布式日志存储、顺序 I/O 与可靠投递一致性。
    - *主线逻辑*：分区追加写、OS Page Cache 复用与 `sendfile` 零拷贝；ISR 水位线、EOS 幂等事务。
15. **[[SystemDesignWiki Transactional Outbox|Wiki · Transactional Outbox (事务发件箱)]]**
    - *核心瓶颈*：本地数据库与外部消息发布（DB + MQ）双写不一致难题。
    - *主线逻辑*：单机 ACID 事务保证业务表与 Outbox 表原子落盘；异步轮询与 CDC 挖掘。
16. **[[SystemDesignWiki Control Data Plane|Wiki · 控制面与数据面解耦 (Control & Data Plane)]]**
    - *核心瓶颈*：控制管理逻辑高开销拖垮高频数据路径，控制面崩溃引发业务全局断流。
    - *主线逻辑*：大脑与躯干切分；生存第一法则（控制面宕机，数据面基于本地缓存 100% 存活运行）。
17. **[[SystemDesignWiki Pull vs Push|Wiki · Pull vs Push (推拉模型与读写扩散)]]**
    - *核心瓶颈*：多端数据同步与社交 Feed 的写放大与读延迟冲突。
    - *主线逻辑*：写扩散（Push）vs 读扩散（Pull）；大 V / 明星问题与混合动静分离。
18. **[[SystemDesignWiki Idempotency|Wiki · Idempotency (幂等性设计与实现模式)]]**
    - *核心瓶颈*：网络超时三态重试引发的重复创建、重复扣费与计算资源雪崩。
    - *主线逻辑*：数学与工程定义；联合唯一约束、分布式 Token、状态机 CAS、确定性 Hash 派生。
19. **[[SystemDesignWiki NewSQL|Wiki · NewSQL (分布式数据库架构与选型)]]**
    - *核心瓶颈*：传统分库分表跨分片分布式事务性能崩塌、重分片运维沉重与数据倾斜。
    - *主线逻辑*：计算存储分离、Range-based 动态 Region 切片、Multi-Raft 多数派强一致、Percolator 2PC。

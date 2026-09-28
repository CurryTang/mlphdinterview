# Wiki · NewSQL 分布式数据库架构与选型 (NewSQL & Distributed SQL)

Wiki 词条归属：[[SystemDesign00 Overview|00 系统设计全局蓝图]] → [[SystemDesignWiki NewSQL|Wiki · NewSQL]]

在分布式数据基础设施的演进中，面对海量数据与高并发写入，系统架构通常面临两大经典路径的选择：**传统分库分表（Sharded RDBMS）** 与 **新一代分布式关系型数据库（NewSQL / Distributed SQL）**。

```newsql-architecture-visual
```

---

## 1 · 核心概念与架构设计哲学

### 1.1 · Sharded RDBMS（分库分表传统数据库）
- **定义**：在传统单体关系型数据库（如 MySQL、PostgreSQL）之上，通过**应用层路由**或**分布式中间件/代理（如 Vitess、ShardingSphere、Citus）**，将数据按照特定的分片键（Sharding Key）水平切分并分散存储在多个独立的数据库实例中。
- **架构特征（Shared-Nothing）**：
  - 各物理数据库实例彼此完全隔离独立，各自运行单机存储引擎（如 InnoDB）。
  - 中间件解析 SQL 并根据分片键（如 `repo_id % 4`）将请求路由到具体物理节点。
  - **优势**：单分片访问直接走单机成熟引擎，毫秒甚至亚毫秒级超低延迟，生态与工具链极度成熟。
  - **致命伤**：跨分片事务性能断崖式下跌，大租户数据倾斜无法调和，二次扩容重分片（Re-sharding）运维成本极高。

### 1.2 · NewSQL（原生分布式关系型数据库）
- **定义**：从存储引擎底层重构、原生支持水平弹性扩展，同时**天然具备完整 ACID 强一致性事务能力与标准 SQL 接口**的新一代分布式数据库（如 TiDB、CockroachDB、Google Spanner、YugabyteDB）。
- **架构四大技术支柱**：
  1. **计算与存储分离（Compute-Storage Decoupling）**：
     - 上层无状态 SQL 计算节点（如 TiDB Server）负责 SQL 解析、基于代价的查询优化（CBO）以及执行计划生成。
     - 下层分布式存储引擎（如 TiKV、RocksDB LSM-Tree）负责数据高可用持久化。
  2. **基于有序范围的自动切片（Range-based Region Partitioning）**：
     - 全局数据不再按物理库表切割，而是按 Key 的连续范围切分为数万个固定大小的 **Region / Range**（如默认 96MB）。
     - 当数据写入使 Region 超过阈值时，自动触发分裂（Split），突破单机容量限制。
  3. **Multi-Raft 共识复制与高可用**：
     - 每个 Region 在不同物理存储节点上维护 $2F+1$ 个副本（Replica）。
     - 副本之间独立运行 **Raft** 共识协议，动态选举 Raft Leader 承载读写，Follower 维持强一致同步，确保故障切换时 RPO=0。
  4. **全局分布式事务与时钟协调（Percolator 2PC + HLC/TrueTime）**：
     - 采用基于 Google Percolator 模型的两阶段提交（2PC），解耦全局锁。
     - 借助混合逻辑时钟（Hybrid Logical Clock, HLC）或 GPS/原子钟（Google TrueTime）提供单调递增的全局时间戳，实现快照隔离级别（Snapshot Isolation）与全局一致性读。

---

## 2 · 核心架构对比图解

```text
======================= 方案 A: Sharded RDBMS (Shared-Nothing) =======================

       [Application Client]
                 │ (SQL Query)
                 ▼
       [Sharding Proxy / Middleware (Vitess / ShardingSphere)]
                 │
                 ├── 路由规则 (repo_id % 3)
                 │
      ┌──────────┼──────────┐
      ▼          ▼          ▼
┌──────────┐┌──────────┐┌──────────┐
│ Shard 0  ││ Shard 1  ││ Shard 2  │
│ (MySQL)  ││ (MySQL)  ││ (MySQL)  │
│ 独立引擎 ││ 独立引擎 ││ 独立引擎 │
└──────────┘└──────────┘└──────────┘
* 瓶颈: 跨分片聚合需 Proxy 广播拉取全量行在内存拼装; 扩容需人工做数据双写迁移。

======================= 方案 B: NewSQL (原生分布式计算存储分离) =======================

       [Application Client]
                 │ (Standard SQL)
                 ▼
┌────────────────────────────────────────────────────────┐
│  Stateless SQL Compute Layer (TiDB / Cockroach Nodes)  │ ──> 无状态，秒级水平扩缩容
│  · 分布式查询优化器 · 算子下推 · 执行计划并行化          │
└────────────────────────────────────────────────────────┘
                 │ (gRPC RPC 交互)
                 ▼
┌────────────────────────────────────────────────────────┐
│  Metadata & Clock Coordinator (PD / HLC / TrueTime)    │ ──> 全局时间戳分配、拓扑调度
└────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Distributed Storage Engine (Multi-Raft Groups / TiKV) │
│                                                        │
│  [Storage Node 1]      [Storage Node 2]      [Storage Node 3]
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  │ Region 1 (Leader)│  │ Region 1 (Follow)│  │ Region 1 (Follow)│  <- Raft Group 1
│  │ Region 2 (Follow)│  │ Region 2 (Leader)│  │ Region 2 (Follow)│  <- Raft Group 2
│  │ Region 3 (Follow)│  │ Region 3 (Follow)│  │ Region 3 (Leader)│  <- Raft Group 3
│  └──────────────────┘  └──────────────────┘  └──────────────────┘
└────────────────────────────────────────────────────────┘
* 优势: 算子直接下推 (Coprocessor) 存储节点并行过滤; 节点增加自动触发 Region 迁移。
```

---

## 3 · 核心机制对比矩阵

| 评估维度 | Sharded RDBMS（如 Vitess + MySQL） | NewSQL（如 TiDB / CockroachDB） |
| :--- | :--- | :--- |
| **底层核心架构** | 传统单机引擎组合，Shared-Nothing 物理隔离 | 原生计算与存储分离，Multi-Raft 分布式一致性 |
| **单分片极值延迟** | **极低且稳定（1 - 2 毫秒）**<br>纯单机成熟 B+Tree 引擎，无跨网网络共识 RTT | **稍高（3 - 10 毫秒）**<br>提交需跨节点走 Raft/Paxos 多数派仲裁与网络往返 |
| **弹性扩容与重分片** | **极度沉重（人工双写、灰度切片、停机锁表）**<br>增加节点需全量校验与 Binlog 追平，周期漫长 | **原生自动化、零感知（秒级在线弹性伸缩）**<br>新增物理机，底层调度器自动通过 Raft 迁移分片副本 |
| **跨分片事务与复杂 JOIN** | **能力极差**<br>XA 分布式事务性能骤降；跨分片 JOIN 极易撑爆 Proxy 内存 (OOM) | **原生强支持**<br>分布式查询优化器将 Filter/Agg 下推到存储节点并行计算 |
| **数据倾斜（Hotspot Skew）** | **依赖预先容量规划与人工干预**<br>超级大租户单点打爆该分片磁盘与 CPU，无法自愈 | **自适应切片与动态打散**<br>自动将大表拆分为数万个 Region，支持分散在不同物理机 |
| **生态成熟度与排障心智** | **极高**<br>工具链成熟，单机慢查询、锁冲突排查体系高度标准化 | **系统复杂度高**<br>需深谙分布式追踪、Raft 状态机、时钟漂移与分布式死锁检测 |
| **硬件与基础设施成本** | **较低**<br>资源利用率精准，可用中低配置实例堆叠 | **较高**<br>多副本共识、LSM-Tree 写放大需要高配 CPU/NVMe/10GbE 网络 |

---

## 4 · 结合 CI/CD 流水线系统的生产选型决策

针对大规模 CI/CD 流水线（峰值 50,000 QPS，单月产生数十亿条 `workflow_run` 与 `job` 记录）：

### 4.1 · 选用 Sharded RDBMS 的决策依据（以 `repo_id` 为分片键）
1. **数据访问模式天然具备仓库局部性（Data Locality）**：
   - 99% 的核心查询是单仓库闭环的：查询特定 Commit 的流水线状态、更新当前流水线内所有 Job 的依赖拓扑。
2. **零跨分片事务代价**：
   - 将 `repo_id`（或 `tenant_id`）选为 Sharding Key，同一代码仓库的所有 `workflow_run`、`job`、`dependency` 数据全部落在**同一个物理分片**内。
   - 所有的 DAG 依赖状态跃迁与事务均为**分片内本地 ACID 事务**，完全绕开昂贵的分布式 2PC，以最低的服务器成本换取极致的低延迟。

### 4.2 · 选用 NewSQL 的决策依据
1. **彻底消除海量数据爆发导致的重分片噩梦**：
   - 传统分库分表在 1~2 年后随着构建记录突破百亿，面临分片容量耗尽二次扩容的巨大风险。NewSQL 随业务增长只需采购追加存储服务器，底层自动平衡。
2. **抵御单一巨型仓库（Mono-repo）的数据倾斜**：
   - 如果系统接入了超大型仓库（如数万人维护的主干 Mono-repo，每天触发几十万次 CI），其并发请求会彻底击垮单分片 MySQL。NewSQL 能够自动将该仓库的数据在底层按 Key 范围拆开并分布在几十台机器上，天然规避单机热点崩溃。
3. **全局控制面跨租户聚合报表**：
   - 管理后台需要统计所有企业租户的 Runner 配额使用率与构建成功率指标，NewSQL 原生支持下推聚合，无需额外抽取数据到复杂的数据仓库即可在线出具报表。

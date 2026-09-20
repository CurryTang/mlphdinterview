# System Design 12 · 分布式填字游戏求解器 (Crossword Puzzle Solver)

课程位置：[[SystemDesign11 Notification System|11 移动推送与通知系统]] → 本篇 → [[SystemDesign99 Glossary|99 术语]]

接收网格定义与词典约束，依据拓扑复杂度自适应分流；普通题目优先单机位并行 CSP 本地极速求解；病态长尾题目接入分布式深度优先搜索（Distributed DFS）集群，通过顶层前沿展开、紧凑状态重放、粗粒度最浅工作窃取、信用权重守恒与原子 CAS 竞态仲裁，实现兼顾微秒级单机吞吐与海量规模组合求解的工业级引擎。

![[assets/crossword-solver-whiteboard.png|Crossword Solver System Design 架构白板]]

---

## 1. 功能需求 · Functional Requirements

1. **提交题目求解任务 (Submit Puzzle Job)：**
   - 客户端输入网格拓扑定义（$M 	imes N$ 尺寸、空白槽位 Slots 起始坐标、方向与长度）、预填固定字母（Fixed Letters）以及租户信息；
   - 准入层执行静态拓扑分析，识别网格复杂度；返回全局唯一 `job_id`，并固化不可变词典快照（`dictionary_version`）与算法引擎版本（`solver_version`）。
2. **状态与结果轮询/回调 (Status & Result Retrieval)：**
   - 支持客户端通过 `GET /v1/jobs/{job_id}` 查询任务实时运行阶段（排队中、单机快速求解中、分布式 DFS 展开中、完成、超时）；
   - 求解成功时返回网格完整合法填充解（Filled Grid Assignment）及耗时指标；支持注册 Webhook 异步回调。
3. **取消运行中任务 (Cancel Running Job)：**
   - 客户端可通过 `POST /v1/jobs/{job_id}/cancel` 终止已提交或运行中的任务，系统通过分布式信令级联释放下属所有 Worker 的算力租约。
4. **确定性终态语义 (Deterministic Terminal States)：**
   - 任务终态严格划分为以下五种，禁止歧义混淆：
     - `SUCCEEDED`：找到并经验证器确认完全合法的填字解；
     - `UNSAT` (Unsatisfiable)：数学上完全遍历并证伪所有分支，证明该题在当前词典下绝对无解；
     - `TIMED_OUT`：超过租户或系统设定的硬超时时间（Hard Timeout），搜索树未完全覆盖；
     - `CANCELLED`：调用方显式取消；
     - `FAILED`：系统内部非预期异常（如节点连续崩溃、数据损坏）。

**范围边界与技术假设 (Scope & Assumptions)：**
- **解的目标**：求**任意一个合法满足约束的解**（Satisfiability），不进行全局穷举，亦不需要根据自然语言线索（Natural-Language Clues）进行语义概率打分。
- **全局词汇唯一性**：同一个填字游戏中不允许出现重复单词（即满足全局 `AllDifferent` 约束）。
- **计算本质**：系统是**约束满足问题 (Constraint Satisfaction Problem, CSP)** 与**分布式树搜索 (Distributed DFS)** 的工程结合体。绝非所有题目盲目跑分布式：普通题目利用单机位并行剪枝在毫秒级消化，仅对静态特征命中病态模式或动态超时长尾的任务启用 Distributed DFS。

---

## 2. 非功能需求 → 对应设计

| 非功能需求 | 目标指标 | 对应架构 / 策略设计 |
| :--- | :--- | :--- |
| **延迟 (Latency)** | 普通/常见复杂度题目 P95 < 10s（实际多在 50ms~500ms）；极端长尾复杂题允许异步长任务，硬超时上限 10 min | **双轨分流 + 单机优先**：普通题单机位并行位图引擎（Bitmap Index + AC-3 + MRV）秒出；复杂题通过静态分类器或 5s 动态看门狗转入分布式 DFS 算力池。 |
| **吞吐与规模 (Scale)** | 峰值接入 100 new jobs/s，系统维持 1,000+ 在途并发任务 (Concurrent In-flight Jobs) | 接入层纯无状态水平扩展；单任务默认占用 1 vCPU，基准算力池配置 400 vCPU 并配合自动化弹性伸缩；词典完全本地内存化，消除远端存储 IO 瓶颈。 |
| **正确性 (Correctness)** | **100% 绝对无假阳性、无假无解**；`TIMED_OUT` 绝对不能判定为 `UNSAT` | 任何 Worker 产出的候选解必须经过独立校验器（Independent Verifier）；只有当分布式 DFS 的所有互斥子分支依据信用权重守恒（Credit Conservation）全量收敛 Exhausted，才能宣告 `UNSAT`。 |
| **高可用与可靠性 (Reliability)** | 单 Worker 崩溃、超时、OOM 不丢题，系统具备自动容错自愈能力 | 任务派发基于分布式租约（Lease & Heartbeat）；Worker 崩溃后租约过期自动重新入队重试；分布式子树状态变更采用 CAS（Compare-And-Set）原子竞态提交。 |
| **多租户隔离与成本控制 (Fairness & Budget)** | 防止恶意或极端长尾题目霸占算力池，导致其他租户请求队头阻塞（Head-of-Line Blocking） | 按租户物理/逻辑多队列隔离，实行赤字加权轮询（Deficit Round Robin / WFQ）；设定单任务 CPU-Seconds、内存及最大分布式并发度配额（Parallelism Cap）。 |
| **可复现性 (Reproducibility)** | 相同题目输入在相同版本下必须保证严格幂等与复现 | 任务绑定不可变 `dictionary_version` 快照与 `solver_version`；解题结果基于 `(puzzle_hash, dict_version)` 写入结果缓存，瞬时秒级命中。 |

---

## 3. 容量估算与算力模型 · Capacity Estimation

### 3.1 核心业务体量指标

- **峰值提交速率 (Peak Submission Rate, $\lambda$)：** 100 jobs/s。
- **任务平均计算耗时 (Average Compute Cost, $ar{t}$)：** 普通题目平均消耗 2 CPU-seconds（单机位并行求解约 50ms~500ms，含排队损耗）。
- **基准稳态计算算力需求：**
  $$	ext{Sustained vCPU} = 100	ext{ jobs/s} 	imes 2	ext{ CPU-s} = 200	ext{ vCPU}$$
- **峰值安全裕量 (2x Headroom Pool)：**
  $$	ext{Baseline Worker Pool} = 200	ext{ vCPU} 	imes 2 = 400	ext{ vCPU}$$
- **在途活跃任务规模 (Active Concurrent Jobs)：**
  假设普通题目平均响应周期目标为 10 秒（含排队与求解）：
  $$	ext{In-Flight Active Jobs} = 100	ext{ jobs/s} 	imes 10	ext{ s} = 1,000	ext{ active concurrent jobs}$$

### 3.2 词典存储与内存占用分析 (Dictionary Footprint)

填字求解器依赖大规模候选词库（如百万级英语词典）：
- **原始文本规模：** 1,000,000 单词 $	imes$ 平均 10 字符 $pprox 10	ext{ MB}$ 纯文本。
- **位图倒排索引规模 (Dense Positional Bitmap Index)：**
  - 按单词长度分桶（Length Buckets，如 3 到 21 字符）；
  - 针对每个长度 $L$ 的词库，建立“位置-字母”位图矩阵：$L 	imes 26$ 个 Bitmaps，每个 Bitmap 长度为该长度下的单词总数 $W_L$；
  - 1M 单词总位图占用：$1,000,000 	imes 10	ext{ positions} 	imes 26	ext{ bits} pprox 260	ext{M bits} pprox 32.5	ext{ MB}$。
- **单机内存装载评估：**
  加上 Trie 树、前向索引与交点映射结构，每个 Worker 节点仅需占用 **50 MB ~ 200 MB** 内存。
- **核心系统设计推论：**
  **词典存储与网络传输根本不是该系统的架构瓶颈**。所有词典必须以不可变版本快照形式常驻 Worker 节点本地内存或通过 `mmap` 共享映射，严禁在搜索循环中向远程数据库或 Redis 发起读词请求。真正的系统瓶颈是**计算长尾的组合爆炸（Combinatorial Explosion）**与多租户资源竞争。

### 3.3 分布式 DFS 调度与网络通信开销评估

在分布式 DFS 场景下，若状态序列化设计不当，网络通信将迅速摧毁系统：
- **Naive 状态序列化开销（反面教材）：** 若将整个 CSP 求解器堆栈、已分配域镜像序列化，单个子任务大小将达 $1	ext{ MB} \sim 5	ext{ MB}$。切分 1,000 个分支需产生数 GB 网络吞吐，序列化耗时达数十毫秒，远超计算本身。
- **紧凑前缀编码开销（推荐生产方案）：**
  每个子树由前缀分配路径唯一定义（例如：`[(slot_0, "PLANET"), (slot_3, "LASER")]`）。
  $$	ext{Payload Size} = K_{	ext{prefix\_slots}} 	imes (4	ext{ bytes slot\_id} + 16	ext{ bytes word}) pprox 64 \sim 256	ext{ bytes}$$
- **网络开销微秒化：** 任务派发报文仅数百字节，在 10Gbps 内网中传输延迟 $< 0.1	ext{ ms}$。Worker 接收后利用本地固化词典位图执行一次弧相容回放（$< 0.2	ext{ ms}$）即可完全重建求解状态，通信开销近乎为零。

---

## 4. 架构设计 · High-Level Architecture

### 4.1 初版设计 (Naive Global Queue) 及其致命缺陷

```text
[Clients] ──▶ [API Service] ──▶ [Single Global FIFO Queue] 
                                          │
                                          ▼
                                [Naive DFS Workers]
                                (Linear scan Dict DB)
```

**为什么走不通？**
1. **队头阻塞与全盘休克：** 填字游戏属于 NP-Complete 问题。单个病态（Pathological）或无解的 $25 	imes 25$ 网格在无高效剪枝下可能展开 $10^{18}$ 种状态树。单一 FIFO 队列会导致长尾任务占满全部 Worker，所有正常 50ms 任务全部堆死超时。
2. **远程词典查询引发网络风暴：** Naive 求解器每尝试填一个词就去 Redis/MySQL 做 `SELECT word WHERE w[2]='a' AND len=5`，网络 RTT（0.5ms）放大数百万倍后导致单题耗时数小时。
3. **盲目全量分布式化引发调度反噬：** 若不加区分地对所有题目执行分布式拆解，大量 10ms 即可解出的普通题目被迫承受 20ms~50ms 的网络入队、租约拉取与分布式仲裁延迟，整体 QPS 下跌一个数量级。
4. **假无解（False UNSAT）与误判：** 任务超时退出时被直接标记为无解；或者 Worker 崩溃重启导致子分支丢失，调用方得到错误的业务结论。

---

### 4.2 生产级分层架构与分布式 DFS 全景

```text
[Clients / Batch Callers]
           │
           │ 1. POST /v1/jobs (Grid Topology + Fixed Letters)
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Ingestion & Admission Layer                                              │
│   [API Gateway] ──(Auth & Rate Limit)                                       │
│          │                                                                  │
│          ▼                                                                  │
│   [Job API Service]                                                         │
│          ├── 查 [Result Cache] (puzzle_hash + dict_version)                 │
│          ├── [Static Complexity Classifier] ── 提取网格特征 (M×N, 密度, 词长)│
│          ▼                                                                  │
│   [Job Store (PostgreSQL / DynamoDB)]                                       │
│   (job_id, tenant_id, status, budget, pinned_versions, path_mode)           │
└──────────┬──────────────────────────────────────────────────────────────────┘
           │ 2. Route Job via Classifier
           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Fair Scheduling & Dual-Queue Layer                                       │
│   [Fair Scheduler / DRR Engine]                                             │
│          ├── 计算排队估算 CPU 秒数 (Estimated Backlog)                      │
│          ├── 租户公平调度 (Weighted Fair Queuing / DRR)                     │
│          ├───────────────────────────┬──────────────────────────────────────┤
│          │ (普通题目: 90%+ 流量)       │ (静态病态题 或 动态升级题)            │
│          ▼                           ▼                                      │
│   [Normal Job Queue]          [Hard / Distributed DFS Queue]                │
│   (Default: 1 worker)         (Coarse Disjoint Subtree Partitions)          │
└──────────┬───────────────────────────┬──────────────────────────────────────┘
           │                           │
           ▼                           ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────────┐
│ 3. Normal Solver Workers      │ │ 4. Distributed DFS Worker Pool            │
│   [Solver Worker Pool]        │ │   [Distributed DFS Workers]               │
│   • Local Bitmap Dict (mmap)  │ │   • Pull Subtree (prefix assignment)      │
│   • MRV + Degree + AC-3       │ │   • Replay AC-3 Forward Checking (<0.2ms) │
│   • Watchdog: > 5s or budget? │ │   • Deep DFS with Local Trail Stack       │
│   • >90% exit here (10~500ms) │ │   • Work-Stealing: Shallowest Choice Cut  │
└──────────┬────────────────────┘ └───────────┬───────────────────────────────┘
           │ 动态超时升级                      │ 产出解 / 分支 Exhausted
           ▼                                   ▼
┌───────────────────────────────┐ ┌───────────────────────────────────────────┐
│ [Adaptive Splitter]           │ │ 5. Verifier & CAS Consensus Engine        │
│ • Root BFS Frontier Expansion │ │   [Independent Verifier]                  │
│ • Disjoint Subtrees Cut (K)   │ │   • Strict all-diff + cross letter check  │
│ • Assign Credit Weights (W/K) │ │          │                                │
│ • Push to Distributed Queue   │ │          ▼                                │
└───────────────────────────────┘ │   [CAS Winner Commit]                     │
                                  │   • Atomic UPDATE status='SUCCEEDED'      │
                                  │   • Broadcast Cancel to all siblings      │
                                  │   • Credit Accumulation Barrier → UNSAT   │
                                  └───────────────────────────────────────────┘
```

**分层职责剖析：**
1. **接入、缓存与静态分类层 (Ingestion, Cache & Static Gate)：**
   - 提取网格拓扑特征计算标准化指纹 `puzzle_hash`；优先检索结果缓存（秒级命中）。
   - **静态复杂度分类器 (Static Complexity Classifier)**：通过网格尺寸、交叉密度、长词比例评估搜索难度。命中病态特征的大型/稀疏题目**零等待直通分布式 DFS 队列**；其余标准题目进入普通队列。
2. **公平调度与双轨队列 (Fair Scheduling & Dual Queues)：**
   - 实施赤字加权轮询（Deficit Round Robin, DRR），杜绝多租户请求间的队头阻塞。
   - 物理拆分**普通单机队列 (Normal Queue)** 与**分布式 DFS 子树队列 (Distributed DFS Queue)**。
3. **单机位并行求解池 (Normal Solver Workers)：**
   - 消化 90%+ 的常规题目。采用不可变本地位图索引、MRV 变量排序与 AC-3 前向弧相容，大部分在 50ms~500ms 内求解完成。
   - 挂载看门狗（Watchdog）：一旦计算耗时超过 5 秒或回溯次数突破 $10^6$，立即中断单机执行，触发切片器无缝转入分布式 DFS 模式。
4. **分布式 DFS 算力池 (Distributed DFS Worker Pool)：**
   - 专职处理长尾困难与静态重型任务。接收紧凑前缀描述符，秒级恢复状态并开启深入深度优先回溯。
   - 支持**粗粒度工作窃取（Coarse Work Stealing）**：空闲节点可从高负载节点的最浅层未探索选择点窃取子分支，实现不规则树搜索（Irregular Tree Search）的动态负载均衡。
5. **独立验证与竞态仲裁层 (Verifier & CAS Consensus)：**
   - 任何 Worker 产生候选解必须经过轻量独立验证器复核；
   - 通过数据库 CAS 机制原子争抢唯一 Winner 资格；胜出后通过集群广播协议（Redis Pub/Sub / gRPC）通知所有协同兄弟 Worker 立即终止 DFS；
   - 基于**信用权重守恒（Credit Conservation）**统计，当且仅当整棵搜索树所有互斥分支均 Exhausted 且总信用回收完毕时，原子宣告 `UNSAT`。

---

### 4.3 核心数据模型 · Data Model

#### 1. `puzzle_jobs` (任务总表)
```sql
CREATE TABLE puzzle_jobs (
    job_id               VARCHAR(64) PRIMARY KEY,      -- 全局唯一 UUID
    tenant_id            VARCHAR(64) NOT NULL,         -- 租户标识
    puzzle_hash          VARCHAR(64) NOT NULL,         -- 网格与预填字符的标准哈希 (用于命中缓存)
    grid_width           INT NOT NULL,                 -- 网格宽度 M
    grid_height          INT NOT NULL,                 -- 网格高度 N
    raw_puzzle_json      JSONB NOT NULL,               -- 槽位拓扑与固定字母定义
    dictionary_version   VARCHAR(32) NOT NULL,         -- 固化绑定的词典版本快照
    solver_version       VARCHAR(32) NOT NULL,         -- 固化的求解算法引擎版本
    execution_mode       VARCHAR(16) NOT NULL,         -- 'LOCAL_CSP' (单机快速), 'DISTRIBUTED_DFS' (分布式搜索)
    status               VARCHAR(16) NOT NULL,         -- 'ACCEPTED', 'RUNNING', 'SPLIT_RUNNING', 'SUCCEEDED', 'UNSAT', 'TIMED_OUT', 'CANCELLED', 'FAILED'
    solution_json        JSONB,                        -- 求解成功时的全网格字符分配矩阵
    cpu_ms_spent         INT NOT NULL DEFAULT 0,       -- 累计消耗的 CPU 毫秒数
    total_credit_weight  BIGINT NOT NULL DEFAULT 4294967296, -- 初始信用权重 (2^32, 用于 UNSAT 屏障证明)
    recovered_credit     BIGINT NOT NULL DEFAULT 0,    -- 累加已确认 Exhausted 的信用总和
    hard_timeout_at      TIMESTAMP WITH TIME ZONE NOT NULL, -- 绝对壁钟超时时间 (如 NOW() + 10 min)
    created_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    finished_at          TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_jobs_tenant_status ON puzzle_jobs(tenant_id, status);
CREATE INDEX idx_jobs_cache_lookup ON puzzle_jobs(puzzle_hash, dictionary_version, status);
```

#### 2. `search_subtrees` (分布式 DFS 子树分支表)
```sql
CREATE TABLE search_subtrees (
    subtree_id           VARCHAR(64) PRIMARY KEY,      -- 子树分支唯一标识
    job_id               VARCHAR(64) NOT NULL REFERENCES puzzle_jobs(job_id) ON DELETE CASCADE,
    branch_index         INT NOT NULL,                 -- 分支序号 (0 .. K-1)
    tree_depth           INT NOT NULL,                 -- 分支前缀所在的树深度
    prefix_assignment    JSONB NOT NULL,               -- 极简分配前缀 (e.g. [{"slot_id": 1, "word": "APPLE"}])
    credit_weight        BIGINT NOT NULL,              -- 本分支分配的信用权重 (用于 UNSAT 证明)
    status               VARCHAR(16) NOT NULL,         -- 'PENDING', 'CLAIMED', 'EXHAUSTED', 'SOLVED', 'CANCELLED'
    worker_id            VARCHAR(64),                  -- 认领的 Worker 节点 ID
    lease_until          TIMESTAMP WITH TIME ZONE,     -- 租约截止时间 (Lease Expiration)
    attempt_count        INT NOT NULL DEFAULT 0,       -- 重试认领次数
    updated_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_subtrees_job_status ON search_subtrees(job_id, status);
CREATE INDEX idx_subtrees_lease ON search_subtrees(status, lease_until) 
WHERE status = 'CLAIMED';
```

#### 3. `dictionary_snapshots` (词典不可变版本表)
```sql
CREATE TABLE dictionary_snapshots (
    version_id           VARCHAR(32) PRIMARY KEY,      -- 如 'en_us_v20260901'
    word_count           INT NOT NULL,                 -- 词汇总条目数
    storage_uri          TEXT NOT NULL,                -- S3 / 本地共享存储位图与前缀文件地址
    checksum_sha256      VARCHAR(64) NOT NULL,         -- 校验和，确保所有 Worker 加载一致
    created_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

---

### 4.4 状态机与终态边界定义 (State Machine & Invariants)

```text
       [ACCEPTED]
           │
           ├── (静态复杂度判定) ──▶ [SPLIT_RUNNING] (Direct Distributed DFS)
           ▼                                │
       [RUNNING] (Local CSP)                │
      /    │    \                           │
     /     │     └── (5s Watchdog) ─────────┘
    ▼      ▼      ▼                         │
[SUCCEED][TIMED] [CANCEL]                   ▼
                                      [SPLIT_RUNNING]
                                     /    │    \                                          ▼     ▼     ▼      ▼
                                [SUCCEED][TIMED][CANCEL][UNSAT]
                                (First)  (Budget)(Client)(Credit=2^32)
```

**绝对状态不变式 (Invariant Rules)：**
1. **`TIMED_OUT != UNSAT`**：任何因达到 CPU 预算或壁钟超时退出、且搜索空间未被 100% 覆盖的任务，**一律标记为 `TIMED_OUT`**，严禁向客户端返回 `UNSAT`。
2. **信用守恒与 UNSAT 屏障判定 (Credit Conservation UNSAT Barrier)**：
   仅当分布式切片产生的所有互斥子树记录状态均变为 `EXHAUSTED`，且累计回收的信用总和等于初始总权重（$2^{32}$）时，协调器才执行原子更新宣告无解：
   ```sql
   UPDATE puzzle_jobs 
   SET status = 'UNSAT', finished_at = NOW() 
   WHERE job_id = :job_id 
     AND status = 'SPLIT_RUNNING'
     AND recovered_credit = total_credit_weight;
   ```
3. **Winner 竞争抢占与取消广播**：只要任意 Worker 发现第一个满足条件的解，立即执行 CAS 提交：
   ```sql
   UPDATE puzzle_jobs 
   SET status = 'SUCCEEDED', solution_json = :solution, finished_at = NOW()
   WHERE job_id = :job_id AND status IN ('RUNNING', 'SPLIT_RUNNING');
   ```
   若更新行数（Rows Affected）为 1，则胜出并异步向 Redis Pub/Sub 发送取消广播，通知其他兄弟节点立即可靠中断 DFS。

---

## 5. 核心深度权衡 · Deep Dive

### 5.1 求解路径分流机制：静态预判 (Static Gate) 与动态看门狗 (Dynamic Watchdog)

#### 为什么不全量跑分布式 DFS？只有 Hard 题目才有吗？
很多面试者容易陷入一个误区：“既然构建了强大的分布式 DFS 算力集群，为什么不把所有题目都切分成 10 个子任务并发求解？”

**答案是否定的，原因在于分布式系统的固有调度开销（Distributed Coordination Tax）：**
1. **单机极速消化率极高**：实际业务中，90% 以上的标准填字题目（如经典 $15 	imes 15$ 纽约时报风格）在单机位并行位图索引与 MRV 启发式下，**仅需 10ms ~ 300ms 即可出解**。
2. **分布式调度的反噬效应**：消息网络入队/出队开销（$5	ext{ms} \sim 15	ext{ms}$）、Worker 租约拉取与原子注册（$10	ext{ms}$）、结果跨网络聚合与广播中断（$5	ext{ms}$），构成了 $20	ext{ms} \sim 50	ext{ms}$ 的固定分布式开销。
3. **阿姆达尔定律反向惩罚**：让一个 10ms 即可解出的普通题目走分布式切片，**端到端延迟反而飙升 3~5 倍**，并且集群内多个 Worker 被迫唤醒、占用连接池，导致整体吞吐量（Throughput）暴跌 10 倍以上。

因此，系统绝不能“一刀切”盲目分布式化。

#### 双轨准入分流架构设计
分布式 DFS 绝非“只有等单机跑了 5 秒超时后才被动触发”，而是采用**静态预判直通**与**动态看门狗升级**双轨制：

```text
[Incoming Puzzle Submission]
             │
             ▼ 提取拓扑特征 (尺寸 M×N, 约束密度 ρ, 自由长槽比)
[Static Complexity Classifier]
    ├── 命中病态特征? (M×N ≥ 21×21 且 ρ < 0.15 且 固定字符 ≤ 2%)
    │         │
    │         └── [YES] ──▶ 直通分布式 DFS 队列 (Zero Wait, 直接切片并发求解)
    │
    └── [NO] (常规结构网格)
              │
              ▼ 路由至 Normal Worker 本地位图求解
         [Local CSP Engine]
              ├── 50ms~500ms 内求解完成 ──▶ [Direct Return] (消化 90%+ 流量)
              │
              └── 超出 5s 阈值 / 回溯 > 10^6 次 ──▶ [Dynamic Watchdog Escalation]
                                                          │
                                                          ▼ 触发自适应切片器
                                                  [Distributed DFS Pool]
```

1. **静态预判直通路径 (Static Gate, 零等待)：**
   - 在 API 准入层执行网格图分析，计算**约束图密度 (Constraint Graph Density, $ho$)**：
     $$ho = rac{	ext{相交单元格数 (Cross Intersections)}}{	ext{槽位数 } N 	imes (N - 1) / 2}$$
   - 若网格尺寸达到 $21 	imes 21$ 或 $25 	imes 25$，且 $ho < 0.15$（极度稀疏相交，意味着早期剪枝效果差、极易形成超深无效搜索），同时预填固定字符比例 $< 2\%$；
   - 系统判定该题目具备典型的“组合爆炸特征”，**直接绕过单机求解环节，直接在准入层执行顶层切片并推入 Distributed DFS 队列**。
2. **动态看门狗升级路径 (Dynamic Watchdog Escalation)：**
   - 表面看似规整但内部存在隐式对抗死胡同的题目，单机 Worker 设置壁钟超时（5 秒）与回溯步数计数器（$10^6$ 次）；
   - 一旦触发看门狗阈值，当前 Worker 立即中止本地搜索，将当前搜索前沿转交切片器，平滑升级到分布式 DFS 集群。

---

### 5.2 分布式深度优先搜索 (Distributed DFS) 架构与核心算法

分布式树搜索的核心挑战在于：**DFS 的深度回溯状态本质依赖单机内存堆栈，无法跨网络直接共享指针**。必须通过系统级抽象将其转化为无共享内存（Shared-Nothing）的分布式架构。

#### (1) 搜索树拓扑解耦：顶层 BFS 展平与紧凑状态重放
分布式切片器将单棵庞大搜索树在逻辑上拆分为两个阶段：
- **阶段 1：顶层广度优先前沿展开 (Root BFS Frontier Generation)**
  - 切片器在根节点选择约束度最高、或候选域受限最强的顶层 2~3 个槽位，执行浅层 BFS 展开。
  - 展开至预设分支数（如 $K = 8 \sim 32$）时暂停，生成 $K$ 个互斥的前沿节点（Frontier Nodes）。
  - 这些前沿节点构成了互斥且完备的子搜索空间：
    $$T_{	ext{root}} = T_1 \cup T_2 \cup \dots \cup T_K, \quad orall i 
e j: T_i \cap T_j = \emptyset$$
- **阶段 2：紧凑状态序列化与本地重放 (Compact Prefix Transmission)**
  - 严禁向网络发送整个求解器内存镜像；
  - 切片器仅将前沿节点的固定赋值路径序列化为短数组（如 `[{"slot":0, "w":"TIGER"}, {"slot":3, "w":"EAGLE"}]`），体积极小（$< 256	ext{ bytes}$）；
  - 接收到该子任务的 Worker，依赖本地已 `mmap` 的不可变词典位图，**在 $< 0.2	ext{ ms}$ 内对这组前缀执行一次 AC-3 弧相容前向检查**，瞬间在本地堆栈中完整重构出该子树的全部约束位图，随即进入原生的高性能深度优先回溯循环。

```text
               [Root: Empty Grid]
               /       |                 (Slot 0="CAT")|  (Slot 0="DOG")  ... 顶层 BFS 展平为 K 个前沿分支
             /         |                [Subtree 1] [Subtree 2]  [Subtree K]
          │            │            │
          ▼            ▼            ▼ (网络派发仅需前缀: < 256 字节)
     [Worker 1]   [Worker 2]   [Worker K]
     本地快速重构   本地快速重构  本地快速重构
     执行深层 DFS  执行深层 DFS  执行深层 DFS
```

#### (2) 不规则树搜索与粗粒度工作窃取 (Irregular Tree Search & Shallow Work Stealing)
填字游戏搜索树具有极端的**不规则性 (Irregularity)**：看似对称的子树，分支 1 可能在深度 4 就因为字母冲突被全量剪枝退出（耗时 5ms 宣告该分支 UNSAT），而分支 2 可能在极深层次潜藏着海量状态（耗时数分钟）。
如果只做静态划分，集群将迅速出现长尾掉队者（Straggler）：90% 的 Worker 快速闲置，剩下 1 个 Worker 孤军奋战。

**工作窃取机制（Work Stealing Protocol）：**
- 当 Worker A 率先耗尽其子树变为空闲态时，向处于高负载状态的 Worker B（或协调器）发起 `STEAL` 请求。
- **关键切分原则：从最浅层未探索选择点切分 (Shallowest-Unexplored Split)**！
  - 假设搜索树平均分支因子为 $b$，最大深度为 $D$，在深度 $k$ 切出的子树规模为 $O(b^{D-k})$；
  - **绝不能从栈顶（最深处）窃取**：栈深处剩余工作量趋近 $O(1)$，窃取后瞬间结束，会引发灾难性的网络抖动与空转（Thrashing）；
  - **必须从栈底附近（最浅未探索分支）窃取**：浅层分支蕴含着指数级的工作量（粗粒度），一次网络窃取即可让 Worker A 独立计算数十秒乃至数分钟，通信开销被彻底摊薄。

#### (3) 分布式终止判定与信用权重守恒屏障 (Termination & Credit Invariant)
在满足性问题（SAT）中，找到解即可退出；但在填字游戏中，**严格证明题目无解 (UNSAT)** 具有同等重要的商业与系统价值。
在异步无锁分布式网络中，必须防止由于网络丢包、Worker 崩溃重试导致“误判全部已完成”。

**信用权重守恒法 (Credit Conservation Protocol)：**
- 初始时，根任务被赋予总信用值 $W_{	ext{total}} = 2^{32}$；
- 当切片器将搜索树分裂为 $K$ 个互斥子树时，将权重严格均分：$W_{	ext{child}} = W_{	ext{parent}} / K$；
- 若发生二级工作窃取切分，继续均分信用权重；
- 当 Worker 彻底穷尽某个子树且证明无解时，将该分支的 $W_{	ext{child}}$ 提交给 Coordinator 进行原子累加：
  $$	ext{recovered\_credit} \leftarrow 	ext{recovered\_credit} + W_{	ext{child}}$$
- **终态判定原则**：
  - **证明 UNSAT**：当且仅当 $	ext{recovered\_credit} == W_{	ext{total}}$ 且未发现任何可行解时，才准许原子翻转任务状态为 `UNSAT`。
  - **防止假无解**：若某个 Worker 挂掉，其未回收的信用权重随租约超时重新派发给其他节点重算；若重试耗尽导致任务超时，由于 $	ext{recovered\_credit} < W_{	ext{total}}$，系统只能标记为 `TIMED_OUT`，从数学机制上杜绝假无解（False UNSAT）。

#### (4) 候选解极速广播剪枝与跨节点 Nogood 冲突共享
1. **首解广播与投机中断 (First Solution Broadcast & Cancellation)**：
   - 任一 Worker 搜出完整合法填充并经验证器通过后，通过 CAS 争取 Winner；
   - 胜出后立即向分布式消息总线（Redis Pub/Sub / Kafka）发布 `JOB_CANCEL` 广播；
   - 所有 Worker 在 DFS 循环内部定期（例如每 1,000 次回溯）通过非阻塞原子标志位（Atomic Boolean）检查取消信号，收到信号后立即清空私有堆栈并归还 Worker 算力池。
2. **跨节点冲突子句学习 (Cross-Worker Nogood Learning)**：
   - 当某个 Worker 在深入分支后，证明某种局部的短赋值组合（如：$	ext{Slot}_2=	ext{"CAT"} \land 	ext{Slot}_5=	ext{"DOG"}$）在当前网格拓扑下必然引发交点不可行（Empty Domain），该局部子集被称为 **Nogood 冲突子句**。
   - Worker 将长度较短（$\le 3$ 个槽位）的致命 Nogood 推送至内存缓存（In-Memory Nogood Store）；
   - 其他正在遍历完全不同子分支的 Worker 定期同步新增的高频 Nogood，在本地 DFS 回溯前利用位图进行快速子集过滤。若当前假设包含已知 Nogood，直接就地剪枝，从而实现跨机器的协同剪枝加速。

---

### 5.3 单机位并行求解与弧相容剪枝 (Bit-Parallel CSP Engine)

在单机 Worker 内部，算法的常数因子决定了整个系统的吞吐量上限：

```text
方案 A: Naive DFS + 遍历扫描词典
[Slot 1: Length 5] ──遍历 10,000 词──▶ 填入 "APPLE"
    └── [Slot 2: Cross at pos 3] ──遍历 8,000 词筛选匹配 'P'──▶ ... (遇到冲突回溯)

方案 B: 位图索引 + MRV 启发式 + 弧相容剪枝 (Bit-Parallel CSP Engine, 推荐)
[Slot 1] ──选择候选域最小的槽位 (MRV = 4) ──▶ 位图按位与 (Bitwise AND in 1 CPU cycle)
    └── 级联前向检查 (Forward Checking) ──▶ 迅速使无效分支域收缩为 0 (提前在深度 2 剪枝)
```

| 维度 | 方案 A: Naive DFS + 遍历扫描 | 方案 B: 位图索引 + MRV + 弧相容 (推荐) |
| :--- | :--- | :--- |
| **搜索复杂度** | 分支因子爆炸，最差情况退化至 $O(D^N)$，对复杂题型搜索深度受阻。 | 动态计算度数与最小剩余值（MRV），分支因子大幅降低，搜索节点减少 99% 以上。 |
| **匹配效率** | 每次校验需遍历字符串或执行正则比较，涉及大量内存拷贝与 CPU 缓存失效。 | **位图并行化 (Bit-Parallelism)**：利用 64 位整型向量执行位运算，一次 CPU 指令完成 64 个单词的交点字母匹配。 |
| **内存开销** | 极小（几 MB）。 | 每个节点常驻 50MB~200MB 结构化内存索引与回溯 Trail 栈。 |
| **尾部延迟表现** | **极差**。长尾题目 P99 经常冲破几分钟。 | **极优**。绝大多数标准 15x15 填字游戏在 50ms~500ms 内完成。 |
| **架构选型建议** | 仅适用于极小型玩具演示。 | **生产级高并发求解器的绝对基石**。 |

---

### 5.4 任务完成语义与容错一致性 (Completion Semantics & Correctness)

分布式搜索切片后，多个 Worker 节点并发运行在不同的搜索子树上，面临网络抖动、Worker OOM 及投机竞态。

```text
子树租约认领与胜出竞态时序 (Lease Claim & CAS Winner Flow):
[Worker A (Subtree 1)] ──认领租约 (lease_until = NOW()+30s)──▶ [DB]
[Worker B (Subtree 2)] ──认领租约 (lease_until = NOW()+30s)──▶ [DB]
       │
Worker A 率先找到候选解:
  1. [Worker A] ──▶ [Independent Verifier] (复核 crossing + all-diff) ──▶ PASS
  2. [Worker A] ──▶ CAS (UPDATE jobs SET status='SUCCEEDED' WHERE status='RUNNING')
  3. CAS 成功! ──▶ 向 Redis PubSub 发送 Cancel 信号 ──▶ Worker B 收到信号中断当前计算
```

| 维度 | 方案 A: 严格分布式事务 (Distributed Exactly-Once) | 方案 B: 至少一次执行 + 租约认领 + CAS 竞态胜出 (推荐) |
| :--- | :--- | :--- |
| **可靠性与崩溃恢复** | 依赖两阶段提交（2PC）或分布式锁；Worker 崩溃时锁释放困难，易引发集群死锁。 | **自愈能力极强**。Worker 挂掉后只需等待租约超时，未完成子树自动重新由健康 Worker 抢占。 |
| **一致性保障** | 试图在调度层实现绝对精准单次执行，系统极为脆弱。 | **端到端幂等保证**。利用 `branch_id` 保证子树幂等；利用终态 CAS 确保全局仅有一个 Winner 胜出。 |
| **投机成本控制** | 无法灵活控制兄弟节点的算力浪费。 | 胜出者一旦 CAS 成功，全局广播取消令牌；处于运行态的兄弟 Worker 立即退出并释放 CPU。 |
| **系统吞吐代价** | 协调开销极大，网络交互繁重。 | 仅付出极少量偶尔重复计算的代价，换取无锁、高吞吐的健壮分布式架构。 |
| **架构选型建议** | 分布式搜索场景不适用。 | **唯一推荐的工业级高容错架构**。 |

---

### 5.5 资源隔离与长尾任务治理 (Resource Isolation & Hard-Job Scaling)

面对海量并发请求，不同题目计算量可能跨越 $10	ext{ ms}$ 到 $10	ext{ min}$（数万倍差异）。

```text
自适应切片机制 (Adaptive Coarse Splitting):
[Normal Worker] 发现单机搜索达到 5s 阈值 / 回溯步数超限
       │
       ▼ 选择浅层高分支因子槽位 (e.g. Slot 3 有 50 个合法词)
[Adaptive Splitter]
       ├── 切片为 5 个粗粒度子集 (每个包含 10 个词的前缀约束)
       └── 投递至 [Hard / Branch Queue]
               ├── [Hard Worker 1] 负责 Branch 0 (words 0..9)
               ├── [Hard Worker 2] 负责 Branch 1 (words 10..19)
               └── ...
```

| 维度 | 方案 A: 全局单一 FIFO 共享池 | 方案 B: 租户公平队列 + 难易分流 + 自适应粗粒度切片 (推荐) |
| :--- | :--- | :--- |
| **多租户隔离性** | **无隔离**。恶意租户提交 50 个病态死循环题目，直接打爆全部 Worker，引发平台瘫痪。 | **强隔离**。采用赤字轮询（DRR / WFQ），每个租户拥有独立滑动算力配额，超额请求在各自队列排队。 |
| **队列调度效率** | 长尾任务与秒出任务混杂，严重拉高整体平均排队延迟。 | **难易分流**。普通任务走 Normal Queue 极速直出；长尾任务转入 Hard Queue，避免阻碍快道。 |
| **切片粒度控制** | 若盲目拆解微型 DFS 节点进队列，**消息队列将被网络 IO 打穿**。 | **粗粒度自适应切片 (Coarse-Grained Cut)**。仅在树顶层（深度 1~2）拆解大子树，网络只调度宏观分支。 |
| **弹性伸缩指标** | 单纯依据队列长度（Queue Depth）扩缩容，导致预估完全失真。 | **基于估算排队 CPU 时间（Estimated CPU Backlog）与 Worker 负载加权扩容**。 |
| **架构选型建议** | 仅适用于单一用户离线作业。 | **云原生多租户 SaaS 平台的行业标杆方案**。 |

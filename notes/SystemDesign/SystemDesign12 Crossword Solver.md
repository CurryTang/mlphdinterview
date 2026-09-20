# System Design 12 · 分布式填字游戏求解器 (Crossword Puzzle Solver)

课程位置：[[SystemDesign11 Notification System|11 移动推送与通知系统]] → 本篇 → [[SystemDesign99 Glossary|99 术语]]

接收网格定义与词典约束，优先单机位并行 CSP 本地极速求解；对长尾复杂网格执行粗粒度搜索树自适应切片、多 Worker 投机并行、独立验题与 CAS 竞态提交。

![[assets/crossword-solver-whiteboard.png|Crossword Solver System Design 架构白板]]

---

## 1. 功能需求 · Functional Requirements

1. **提交题目求解任务 (Submit Puzzle Job)：** 
   - 客户端输入网格拓扑定义（$M \times N$ 尺寸、空白槽位 Slots 起始坐标、方向与长度）、预填固定字母（Fixed Letters）以及租户信息；
   - 系统完成校验后返回全局唯一 `job_id`，并固定不可变的词典版本（Pinned `dictionary_version`）与求解器版本（`solver_version`）。
2. **状态与结果轮询/回调 (Status & Result Retrieval)：**
   - 支持客户端通过 `GET /v1/jobs/{job_id}` 查询任务实时运行阶段（排队中、求解中、完成、超时）；
   - 求解成功时返回网格完整合法填充解（Filled Grid Assignment）及耗时指标；支持注册 Webhook 异步回调。
3. **取消运行中任务 (Cancel Running Job)：**
   - 客户端可通过 `POST /v1/jobs/{job_id}/cancel` 终止已提交或运行中的任务，系统级联释放下属所有 Worker 的算力租约。
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
- **计算本质**：系统本质上是**约束满足问题 (Constraint Satisfaction Problem, CSP)** 与**分布式搜索调度**的结合体。优先通过单机位优化与启发式剪枝消化 90%+ 普通任务；仅对长尾困难任务执行分布式树切片。

---

## 2. 非功能需求 → 对应设计

| 非功能需求 | 目标指标 | 对应架构 / 策略设计 |
| :--- | :--- | :--- |
| **延迟 (Latency)** | 普通/常见复杂度题目 P95 < 10s；极端复杂长尾题目允许异步长任务，硬超时上限 10 min | **单机位并行极速求解优先**：Worker 预加载位图索引词典（Bitmap Index），运行 MRV + 弧相容（AC-3/FC）剪枝；超过 5s 判定为困难任务，触发协调器进行子树切片。 |
| **吞吐与规模 (Scale)** | 峰值接入 100 new jobs/s，系统维持 1,000+ 在途并发任务 (Concurrent In-flight Jobs) | 接入层纯无状态水平扩展；单任务默认占用 1 vCPU，基准算力池配置 400 vCPU 并配合自动化弹性伸缩；词典完全本地内存化，消除远端存储 IO 瓶颈。 |
| **正确性 (Correctness)** | **100% 绝对无假阳性、无假无解**；`TIMED_OUT` 绝对不能判定为 `UNSAT` | 任何 Worker 产出的候选解必须经过独立校验器（Independent Verifier）；只有当搜索树切片出的所有互斥子分支（Disjoint Subtrees）全量 Exhausted，才能宣告 `UNSAT`。 |
| **高可用与可靠性 (Reliability)** | 单 Worker 崩溃、超时、OOM 不丢题，系统具备自动容错自愈能力 | 任务派发基于分布式租约（Lease & Heartbeat）；Worker 崩溃后租约过期自动重新入队重试；状态变更采用 CAS（Compare-And-Set）原子竞态提交。 |
| **多租户隔离与成本控制 (Fairness & Budget)** | 防止恶意或极端长尾题目霸占算力池，导致其他租户请求队头阻塞（Head-of-Line Blocking） | 按租户物理/逻辑多队列隔离，实行赤字加权轮询（Deficit Round Robin / WFQ）；设定单任务 CPU-Seconds、内存及最大分布式并发度配额（Parallelism Cap）。 |
| **可复现性 (Reproducibility)** | 相同题目输入在相同版本下必须保证严格幂等与复现 | 任务绑定不可变 `dictionary_version` 快照与 `solver_version`；解题结果基于 `(puzzle_hash, dict_version)` 写入结果缓存，瞬时秒级命中。 |

---

## 3. 容量估算与算力模型 · Capacity Estimation

### 3.1 核心业务体量指标

- **峰值提交速率 (Peak Submission Rate, $\lambda$)：** 100 jobs/s。
- **任务平均计算耗时 (Average Compute Cost, $\bar{t}$)：** 普通题目平均消耗 2 CPU-seconds。
- **基准稳态计算算力需求：**
  $$\text{Sustained vCPU} = 100\text{ jobs/s} \times 2\text{ CPU-s} = 200\text{ vCPU}$$
- **峰值安全裕量 (2x Headroom Pool)：**
  $$\text{Baseline Worker Pool} = 200\text{ vCPU} \times 2 = 400\text{ vCPU}$$
- **在途活跃任务规模 (Active Concurrent Jobs)：**
  假设普通题目平均响应周期目标为 10 秒（含排队与求解）：
  $$\text{In-Flight Active Jobs} = 100\text{ jobs/s} \times 10\text{ s} = 1,000\text{ active concurrent jobs}$$
  此规模与题目设定的 1,000+ 并发完全吻合。

### 3.2 词典存储与内存占用分析 (Dictionary Footprint)

填字求解器依赖大规模候选词库（如百万级英语词典）：
- **原始文本规模：** 1,000,000 单词 $\times$ 平均 10 字符 $\approx 10\text{ MB}$ 纯文本。
- **位图倒排索引规模 (Dense Positional Bitmap Index)：**
  - 按单词长度分桶（Length Buckets，如 3 到 21 字符）；
  - 针对每个长度 $L$ 的词库，建立“位置-字母”位图矩阵：$L \times 26$ 个 Bitmaps，每个 Bitmap 长度为该长度下的单词总数 $W_L$；
  - 1M 单词总位图占用：$1,000,000 \times 10\text{ positions} \times 26\text{ bits} \approx 260\text{M bits} \approx 32.5\text{ MB}$。
- **单机内存装载评估：**
  加上 Trie 树、前向索引与交点映射结构，每个 Worker 节点仅需占用 **50 MB ~ 200 MB** 内存。
- **核心系统设计推论：**
  **词典存储与网络传输根本不是该系统的架构瓶颈**。所有词典必须以不可变版本快照形式常驻 Worker 节点本地内存或通过 `mmap` 共享映射，严禁在搜索循环中向远程数据库或 Redis 发起读词请求。
- **真正致命的系统风险：** **计算长尾（Heavy-Tail CPU Combinatorial Explosion）** 与恶劣题目引发的死锁/饥饿。

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
1. **队头阻塞与全盘休克：** 填字游戏属于 NP-Complete 问题。单个病态（Pathological）或无解的 $25 \times 25$ 网格在无高效剪枝下可能展开 $10^{18}$ 种状态树。单一 FIFO 队列会导致长尾任务占满全部 Worker，所有正常 50ms 任务全部堆死超时。
2. **远程词典查询引发网络风暴：** Naive 求解器每尝试填一个词就去 Redis/MySQL 做 `SELECT word WHERE w[2]='a' AND len=5`，网络 RTT（0.5ms）放大数百万倍后导致单题耗时数小时。
3. **假无解（False UNSAT）与误判：** 任务超时退出时被直接标记为无解；或者 Worker 崩溃重启导致子分支丢失，调用方得到错误的业务结论。

---

### 4.2 最终生产级分层架构

```text
[Clients / Batch Callers]
           │
           │ 1. POST /v1/jobs (Puzzle Grid + Fixed Chars)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Ingestion & Admission Layer                              │
│   [API Gateway] ──(Auth & Per-Tenant Rate Limit)            │
│          │                                                  │
│          ▼                                                  │
│   [Job API Service]                                         │
│          │                                                  │
│          ├── 查 [Result Cache] (puzzle_hash + dict_version) │
│          ▼                                                  │
│   [Job Store (PostgreSQL / DynamoDB)]                       │
│   (job_id, tenant_id, status, budget, pinned_versions)      │
└──────────┬──────────────────────────────────────────────────┘
           │ 2. Submit to Scheduler
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Fair Scheduling & Admission Control Layer                │
│   [Scheduler / Admission Controller]                        │
│          ├── 计算排队估算 CPU 秒数 (Estimated Backlog)      │
│          ├── 租户公平调度 (Weighted Fair Queuing / DRR)     │
│          ├───────────────────────┬──────────────────────────┤
│          ▼                       ▼                          │
│   [Normal Job Queue]      [Hard / Branch Queue]             │
│   (Default: 1 worker)     (Split coarse subtrees)           │
└──────────┬───────────────────────┬──────────────────────────┘
           │                       │
           ▼                       ▼
┌───────────────────────────┐ ┌───────────────────────────────┐
│ 3. Normal Solver Workers  │ │ 4. Hard-Job Distributed Pool  │
│   [Solver Worker Pool]    │ │   [Hard-Job Workers]          │
│   • Local Bitmap Dict     │ │   • Evaluate disjoint branch  │
│   • MRV + Degree + AC-3   │ │   • Bounded depth/backtracks  │
│   • Watchdog: > 5s slow?  │ │   • Heartbeat lease claim     │
└──────────┬────────────────┘ └───────────┬───────────────────┘
           │ 超时切片触发                  │ 发现解 / 完成分支
           ▼                               ▼
┌───────────────────────────┐ ┌───────────────────────────────┐
│ [Adaptive Splitter]       │ │ 5. Verifier & CAS Consensus   │
│ • Hard-Job Coordinator    │ │   [Independent Verifier]      │
│ • Shallow branching cut   │ │   • Strict all-diff check     │
│ • Disjoint subtrees to MQ │ │   • Verify cross intersections│
└───────────────────────────┘ │          │                    │
                              │          ▼                    │
                              │   [CAS Result Commit]         │
                              │   • Winner writes SUCCEEDED   │
                              │   • Broadcast Cancel Siblings │
                              │   • Barrier: all done → UNSAT │
                              └───────────────────────────────┘
```

**分层职责剖析：**
1. **接入与准入层 (Ingestion & Admission)：** 校验网格合法性，生成规范化指纹 `puzzle_hash`；优先检索结果缓存（秒级命中）；写入持久化 `job_store`，固化该任务使用的词典与求解引擎版本快照。
2. **公平调度与队列隔离层 (Scheduling & Queues)：** 实施加权公平排队（WFQ/DRR），彻底阻断跨租户队头阻塞；物理隔离**普通队列 (Normal Queue)** 与**困难子树队列 (Hard/Branch Queue)**。
3. **单机 CSP 核心引擎层 (Normal Workers)：** 90% 以上任务由单节点独立消化。预装载不可变词典位图快照，利用 MRV 与前向交点约束弧相容检测，将单题求解缩减在毫秒至数秒内。内置看门狗（Watchdog）：若单机计算达 5s 仍未出解，主动退出单机模式并移交切片协调器。
4. **自适应切片与困难任务协调 (Adaptive Splitter & Coordinator)：** 针对长尾困难题目，选择浅层具有高区分度的槽位，将其候选词域拆分为 $K$ 个互斥子集（Disjoint Subtrees），生成子任务推入 Hard Queue。
5. **独立验证与竞态仲裁层 (Verifier & CAS Consensus)：** 任何 Worker 产出解必须提交给独立验证器进行全量复核；复核无误后利用数据库 CAS 机制原子竞争成为 Winner；成功后向协调器与兄弟 Worker 广播取消信号，立刻停止投机算力消耗。

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
    status               VARCHAR(16) NOT NULL,         -- 'ACCEPTED', 'RUNNING', 'SUCCEEDED', 'UNSAT', 'TIMED_OUT', 'CANCELLED', 'FAILED'
    solution_json        JSONB,                        -- 求解成功时的全网格字符分配矩阵
    cpu_ms_spent         INT NOT NULL DEFAULT 0,       -- 累计消耗的 CPU 毫秒数
    max_budget_cpu_ms    INT NOT NULL,                 -- 单任务允许消耗的最大 CPU 预算 (如 600,000 ms)
    hard_timeout_at      TIMESTAMP WITH TIME ZONE NOT NULL, -- 绝对壁钟超时时间 (如 NOW() + 10 min)
    created_at           TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    finished_at          TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_jobs_tenant_status ON puzzle_jobs(tenant_id, status);
CREATE INDEX idx_jobs_cache_lookup ON puzzle_jobs(puzzle_hash, dictionary_version, status);
```

#### 2. `search_subtrees` (困难任务子树切片表)
```sql
CREATE TABLE search_subtrees (
    subtree_id           VARCHAR(64) PRIMARY KEY,      -- 子树分支唯一标识
    job_id               VARCHAR(64) NOT NULL REFERENCES puzzle_jobs(job_id) ON DELETE CASCADE,
    branch_index         INT NOT NULL,                 -- 分支序号 (0 .. K-1)
    prefix_assignment    JSONB NOT NULL,               -- 根节点预先固定的部分解剪枝路径
    status               VARCHAR(16) NOT NULL,         -- 'PENDING', 'CLAIMED', 'EXHAUSTED', 'SOLVED', 'CANCELLED'
    worker_id            VARCHAR(64),                  -- 领占的 Worker 节点 ID
    lease_until          TIMESTAMP WITH TIME ZONE,     -- 租约有效截止时间 (Lease Expiration)
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
           ▼
       [RUNNING] ──▶ (触发切片) ──▶ [SPLIT_RUNNING]
      /    │    \                    /    │         /     │     \                  /     │         ▼      ▼      ▼                ▼      ▼      ▼
[SUCCEEDED] [TIMED_OUT] [CANCELLED]   [SUCCEEDED] [TIMED_OUT] [CANCELLED]
                         \                /
                          ▼              ▼
                              [UNSAT] 
                 (必须满足：全部分支均为 EXHAUSTED)
```

**绝对状态不变式 (Invariant Rules)：**
1. **`TIMED_OUT != UNSAT`**：任何因达到 CPU 预算或壁钟超时退出、且搜索空间未被 100% 覆盖的任务，**一律标记为 `TIMED_OUT`**，严禁向客户端返回 `UNSAT`。
2. **`UNSAT` 屏障判定 (Barrier Synchronization)**：只有当切片产生的所有互斥子树记录状态均变为 `EXHAUSTED`，协调器才执行原子更新：
   ```sql
   UPDATE puzzle_jobs 
   SET status = 'UNSAT', finished_at = NOW() 
   WHERE job_id = :job_id AND status IN ('RUNNING', 'SPLIT_RUNNING')
     AND NOT EXISTS (
       SELECT 1 FROM search_subtrees 
       WHERE job_id = :job_id AND status != 'EXHAUSTED'
     );
   ```
3. **Winner 竞争抢占**：只要任意 Worker 发现第一个满足条件的解，立即执行 CAS 提交：
   ```sql
   UPDATE puzzle_jobs 
   SET status = 'SUCCEEDED', solution_json = :solution, finished_at = NOW()
   WHERE job_id = :job_id AND status IN ('RUNNING', 'SPLIT_RUNNING');
   ```
   若更新行数（Rows Affected）为 1，则胜出并异步将同属该 `job_id` 的所有子任务标记为 `CANCELLED`。

---

## 5. 核心深度权衡 · Deep Dive

### 5.1 单机求解延迟与搜索空间剪枝 (Solver Latency & Search Pruning)

填字游戏的形式化建模：网格内每个未填充槽位为变量 $X_i$，其定义域 $D(X_i)$ 为词典中与该槽位长度一致的所有候选单词集合；每对交叉槽位 $(X_i, X_j)$ 在交点单元格处必须满足字母匹配相等性约束；同时全局满足无重复单词约束。

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

### 5.2 任务完成语义与容错一致性 (Completion Semantics & Correctness)

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

### 5.3 资源隔离与长尾任务治理 (Resource Isolation & Hard-Job Scaling)

面对海量并发请求，不同题目计算量可能跨越 $10\text{ ms}$ 到 $10\text{ min}$（数万倍差异）。

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

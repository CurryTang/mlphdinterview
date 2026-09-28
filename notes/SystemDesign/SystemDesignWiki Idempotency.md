# Wiki · 幂等性设计与实现模式 (Idempotency Patterns)

Wiki 词条归属：[[SystemDesign00 Overview|00 系统设计全局蓝图]] → [[SystemDesignWiki Idempotency|Wiki · Idempotency]]

在分布式系统与微服务架构中，**幂等性（Idempotence）**是应对网络不可靠、保障数据最终一致性与系统健壮性的最核心机制之一。

---

## 1 · 幂等性形式化定义与产生根源

### 1.1 · 数学与工程定义
- **数学定义**：若一个函数或映射满足 $f(f(x)) = f(x)$，则称该函数具有幂等性。推广到 $n$ 次连续调用：
  $$f^n(x) = f(x) \quad (\forall n \ge 1)$$
- **分布式系统工程定义**：对同一个系统发起**一次**调用或连续发起**多次相同参数的调用**，系统产生的**业务副作用（Side Effects）完全相同**，且多次调用返回给客户端的业务结果与单次执行保持一致。

### 1.2 · 根本诱因：分布式通信的三态问题（The Three-State Hazard）
在单机内存中，函数调用只有“成功”与“失败”两种确定状态；而在分布式网络中，任何 RPC、HTTP 或 MQ 交互均存在**第三态——超时/未知（Timeout / Unknown）**：

```text
[Client / Caller]                     [Server / Callee]
       │                                     │
       ├─── 1. 发起业务请求 (e.g. 触发 Run) ───>│ (执行成功，状态已落库)
       │                                     │
       │< - - - 2. 网络丢包 / 连接超时 (ACK 未达) - - 
       │
(客户端判定失败，启动自动重试)
       │
       ├─── 3. 重试发起相同请求 ───────────────>│ (若无幂等: 重复创建、重复扣费、资源雪崩)
```

1. **上游客户端自动重试**：网络抖动导致响应丢包，客户端无法判断服务端是否已执行，必须执行重试（Retry）。
2. **消息中间件至少一次投递（At-least-once Delivery）**：Kafka、RabbitMQ 等在 Broker 故障漂移或消费者心跳超时时，会重新投递未确认的消息。
3. **第三方供应商 Webhook 机制**：如 GitHub Webhook、Stripe 支付回调，在未收到 200 OK 之前会按退避策略重推多次。

---

## 2 · 核心案例：流水线运行表（`workflow_run`）场景剖析

以典型的 CI/CD 流水线调度系统为例，核心实体为 `workflow_run`：

```text
┌────────────────────────────────────────────────────────┐
│                   workflow_run 实体模型                 │
├──────────────┬──────────────┬──────────────────────────┤
│ 字段名        │ 类型         │ 业务含义                 │
├──────────────┼──────────────┼──────────────────────────┤
│ run_id       │ VARCHAR(64)  │ 运行实例唯一标识 (主键)   │
│ tenant_id    │ VARCHAR(64)  │ 租户 ID                  │
│ repo_id      │ VARCHAR(64)  │ 代码仓库 ID              │
│ commit_SHA   │ CHAR(40)     │ 触发的代码提交 Commit     │
│ event_type   │ VARCHAR(32)  │ 事件类型 (push/pr/manual)│
│ status       │ VARCHAR(32)  │ 状态 (PENDING/RUNNING...)│
│ created_at   │ TIMESTAMP    │ 创建时间戳               │
└──────────────┴──────────────┴──────────────────────────┘
```

### 痛点与无幂等危害：
- 当开发人员执行 `git push` 时，GitHub 向系统推送 Webhook。若网关在创建任务后网络超时未及时回执 HTTP 200，GitHub 将重复推送。
- **若无幂等控制**：同一 Commit 会生成多个 `run_id`，并发拉起多台昂贵的高性能 Runner 实例编译测试同一个代码版本，造成**计算资源被耗尽、数据库连接池耗尽、计费账单重复翻倍**。

---

## 3 · 幂等性的 6 大主流实现范式

### 3.1 · 范式 1：数据库联合唯一约束（Unique Key Constraint）
利用关系型数据库底层的 B+Tree 唯一索引保证物理级绝对互斥。

- **约束设计**：针对业务维度建立天然唯一键：
  ```sql
  ALTER TABLE workflow_run 
  ADD UNIQUE KEY uk_tenant_repo_commit_event (tenant_id, repo_id, commit_SHA, event_type);
  ```
- **原子插入方案**：
  1. **方案 A (`INSERT ... ON DUPLICATE KEY UPDATE`)**：
     ```sql
     INSERT INTO workflow_run (run_id, tenant_id, repo_id, commit_SHA, event_type, status, created_at)
     VALUES (:run_id, :tenant_id, :repo_id, :commit_SHA, :event_type, 'PENDING', NOW())
     ON DUPLICATE KEY UPDATE run_id = run_id; -- 保持原值，不产生实际写入
     ```
  2. **方案 B (唯一键冲突捕获)**：
     应用层直接执行 `INSERT`；若捕获 MySQL 1062（`Duplicate entry`），则执行 `SELECT` 捞取已有记录并作为成功结果返回。
- **优缺点**：最坚固的最终防线；但在高并发下，频繁触发唯一键死锁检测与回滚会给主库带来写放大压力。

### 3.2 · 范式 2：分布式幂等令牌（Idempotency Key / Request Token）
业界主流开放平台（如 Stripe、GitHub API）的标准实现范式。

```text
[Client]                         [API Gateway / Redis]                [Business DB]
   │                                       │                                │
   ├── 1. POST /runs                       │                                │
   │   Header: Idempotency-Key: <token> ──>│                                │
   │                                       ├── 2. SET token "LOCK" NX EX 30 │
   │                                       │   (原子争抢执行锁)              │
   │                                       │                                │
   │                                       ├── 3. 争抢成功 ────────────────>│ (执行业务入库)
   │                                       │                                │
   │                                       ├── 4. 写入缓存:                  │
   │                                       │   SET token <response> EX 86400│
   │<── 5. 返回 HTTP 201 + Payload ────────┴────────────────────────────────┘
   │
   │   (客户端若超时重发相同请求)
   ├── 6. POST /runs (相同 token) ─────────>
   │                                       ├── 7. 发现 token 已存在且有数据
   │<── 8. 直接返回缓存的历史 Payload (HTTP 200) ───┘ (跳过业务执行与数据库写入)
```

1. **客户端生成**：由客户端发起时生成确定性 Token（或根据请求参数 Hash 派生）。
2. **防并发穿透（SETNX 占位）**：
   - 执行前：`SET idemp:{token} "PROCESSING" NX EX 60`。
   - 若返回 `0`：说明相同请求正在并发执行中，直接阻断并返回 `409 Conflict` 或排队重试。
3. **完成回填**：业务执行完毕后，将响应体存入该 Key：`SET idemp:{token} '{"run_id":"run_123","status":"PENDING"}' EX 86400`。
4. **回放历史**：后续到达的重复请求直接从 Redis 获取缓存的响应体返回，实现透明幂等。

### 3.3 · 范式 3：分布式锁 + 状态机单向跃迁（State Machine CAS）
适合跨多个阶段的长耗时异步任务流转。

- **核心原则**：状态流转必须是**有向无环（DAG）单向不可逆**的：
  $$\text{PENDING} \longrightarrow \text{RUNNING} \longrightarrow \text{COMPLETED / FAILED}$$
- **乐观锁 CAS 更新**：
  ```sql
  -- 状态机单向流转驱动
  UPDATE workflow_run 
  SET status = 'RUNNING', started_at = NOW() 
  WHERE run_id = :run_id AND status = 'PENDING';
  ```
- **执行判定**：
  - 若 `rows_affected == 1`：当前节点成功抢占调度权，继续派发 Runner。
  - 若 `rows_affected == 0`：说明该任务已由其他 Worker 认领并在运行中，或者已经执行完毕进入终态，当前 Worker 直接安全忽略该事件。

### 3.4 · 范式 4：独立去重表（Deduplication Table）
常用于处理消息队列（MQ / CDC）消费端的 At-least-once 投递。

- **表结构**：维护一张结构极简的 `processed_events` 表：
  ```sql
  CREATE TABLE processed_events (
      event_id VARCHAR(128) PRIMARY KEY,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```
- **同事务原子落盘**：
  ```sql
  START TRANSACTION;
  -- 1. 尝试插入事件主键
  INSERT INTO processed_events (event_id) VALUES (:msg_id);
  -- 2. 插入业务表
  INSERT INTO workflow_run (...) VALUES (...);
  COMMIT;
  ```
- **机制保障**：若同一个 `event_id` 重复被消费，第一步直接发生主键冲突引发异常，整个事务回滚，业务数据绝对不会重复写入。

### 3.5 · 范式 5：确定性哈希派生 ID（Deterministic ID Derivation）
利用数学单向哈希函数取代随机自增 ID 或随机 UUIDv4。

- **派生算法**：
  $$\text{run\_id} = \text{SHA256}(\text{tenant\_id} + \text{":"} + \text{repo\_id} + \text{":"} + \text{commit\_SHA} + \text{":"} + \text{event\_type})$$
- **无状态一致性**：无论任何集群节点、任何微服务实例何时计算，相同的上下文必定产生完全一致的 64 位 Hash 字符串。
- **天然防重**：直接将该 Hash 用作主键 `run_id`。重试请求会直接命中主键唯一性约束，杜绝了“生成不同主键导致重复插入多条记录”的结构性漏洞。

### 3.6 · 范式 6：天然幂等操作设计（Natural Idempotence）
在接口与数据建模初期，优先采用具备天然数学幂等性的操作指令：

| 操作性质 | 非幂等操作（增量/相对变化） | 天然幂等替代方案（状态重置/绝对赋值） |
| :--- | :--- | :--- |
| **数值更新** | `UPDATE t SET count = count + 1` | `UPDATE t SET count = 5 WHERE version = 1` |
| **状态流转** | `UPDATE t SET retry = retry + 1` | `UPDATE t SET status = 'CANCELED'` |
| **集合操作** | `list.append(item)`（重试产生重复项） | `set.add(item)` / `INSERT ... ON DUPLICATE IGNORE` |
| **HTTP 语义** | `POST /workflows/runs`（通常非幂等） | `PUT /workflows/runs/{deterministic_id}`（天然幂等） |
| **文件落盘** | 追加写文件末尾（重复追加） | 按固定文件名覆写全量内容 |

---

## 4 · 生产级落地架构：`workflow_run` 端到端双层防御方案

针对用户给出的 `workflow_run` 业务模型，生产级系统必须采用**“前置 Redis 快速拦截 + 后置 DB 唯一索引兜底”**的纵深防御架构：

```text
[Webhook / User Action]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 1: 内存与分布式锁快速拦截 (Redis / 网关层)       │
│  · Key: lock:run:{tenant}:{repo}:{commit}:{event}      │
│  · 执行: SET key "PROCESSING" NX EX 60                 │
│  · 拦截命中: 直接返回 409 Conflict 或等待已有结果      │
└────────────────────────────────────────────────────────┘
           │ (首次请求穿透)
           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 2: 核心业务事务落盘 (Database 最终防线)           │
│  · 主键/唯一键: uk_tenant_repo_commit_event           │
│  · 状态机单向锁: WHERE status = 'PENDING'              │
│  · 结果缓存回填: SET idemp:result:{key} payload EX 24h │
└────────────────────────────────────────────────────────┘
           │
           ▼
[派发计算任务给 Runner]
```

---

## 5 · 工程落地的 4 大深水区陷阱与防范

1. **TOCTOU 竞态漏洞（Time-of-Check to Time-of-Use）**：
   - *严重错误*：在代码层写 `if (!db.exists(run_id)) { db.insert(...); }`。在高并发瞬态请求下，两个并发请求同时通过 `exists` 校验，随后同时执行 `insert` 导致主键冲突崩溃。
   - *修正准则*：防重检查与状态标记必须是**原子操作**（如数据库唯一索引、Redis `SET NX` 或原子 SQL `INSERT`）。
2. **死锁与业务中途崩溃挂起（In-flight Crash）**：
   - 获取 Redis 锁后业务节点突然 OOM 宕机，若未设置锁过期时间（TTL），该任务将永久被死锁锁定。
   - *修正准则*：所有分布式锁必须设置合理的 TTL 超时保底，并由看门狗（Watchdog）在长任务执行期间异步自动续期。
3. **响应一致性破坏（Response Invariance）**：
   - 幂等性不仅要求服务端内部状态一致，还要求客户端拿到的响应一致。如果重试请求直接返回 `500 Server Error` 或抛出 `DuplicateKeyException`，客户端重试逻辑会认为流程失败并不断告警。
   - *修正准则*：捕获重复请求后，应向客户端返回与首次完全相同的 `200 OK` 及历史快照 Payload。
4. **幂等元数据生命周期管理（TTL 与内存溢出）**：
   - 去重表与 Redis 缓存若只增不减，随着业务运行会急剧吞噬存储。
   - *修正准则*：根据业务重试窗口设定合理的保留周期（如 24 小时或 7 天），过期数据由 Redis TTL 自动逐出或由定时数据库归档任务滚动清理。

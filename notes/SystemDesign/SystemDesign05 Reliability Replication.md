# System Design 05 · 可靠性、复制与故障切换

课程位置：[[SystemDesign04 Storage Systems|04 存储系统]] → 本篇 → [[SystemDesign06 Async Messaging Systems|06 异步消息系统]]

冗余不是“多开几台机器”这么简单。副本放在哪里、谁能写、复制到什么程度才确认成功、故障后谁有资格接管，这些问题共同决定系统能不能真的扛住故障。

本篇只讲可靠性和复制。数据库选型见 [[SystemDesign02 Database Paradigms|02 数据库范式]]，分片见 [[SystemDesign03 Database Scaling|03 数据库扩展]]，消息重投见 [[SystemDesign06 Async Messaging Systems|06 异步消息系统]]。

---

## 1 · 先从 failure domain 开始

一句“做三副本”信息不够。三份数据如果都在同一台机器、同一个机架或同一个可用区，对更大的故障没有帮助。

```text
process
  < machine
  < rack / power domain
  < availability zone
  < region
```

设计前先回答：题目要求系统活过哪一级故障？

| 目标 | 常见部署方式 |
|---|---|
| 进程崩溃继续服务 | 同机或跨机多进程，supervisor 自动拉起 |
| 单机故障继续服务 | 多机副本 + load balancer health check |
| 单 AZ 故障继续服务 | 副本跨 AZ，入口能摘除整个 AZ |
| region 故障恢复 | 跨 region 副本、流量切换和独立控制面 |

跨越越大的 failure domain，延迟和成本通常越高。跨区同步写尤其贵，所以不要在没有 RPO/RTO 要求时默认上全球多活。

---

## 2 · 什么时候需要冗余

满足下面任一条件，就应认真设计副本和切换：

- 单实例故障会让核心功能停止；
- 从备份恢复的时间超过 RTO；
- 单机维护会造成不可接受的停机；
- 数据丢失窗口超过 RPO；
- 读吞吐需要多个副本共同承担。

低流量不等于可以没有可靠性。一个每天只有几十个请求的财务系统，QPS 很低，但数据可能比高流量推荐缓存更重要。可靠性由业务损失决定，不由 QPS 单独决定。

### N+1 不是实例数量，而是剩余容量

假设峰值流量为 12K QPS，单实例在目标 p99 下安全处理 1K QPS。部署 12 台刚好够跑，但任何一台下线都会过载。

```text
12K / 1K = 12 instances for traffic
12 × 1.3 headroom ≈ 16 instances
```

如果一个 AZ 有 6 台，整区失效后只剩 10 台，仍然扛不住 12K。此时要按“最大 failure domain 下线”重算，而不是只检查单机 N+1。

```text
surviving capacity after failure >= peak load
```

---

## 3 · Primary-Replica：一个写入口，多份副本

Primary-Replica 也常被叫作 leader-follower 或主从。一个节点处理写入，其他节点从它复制日志或数据变化。

```text
Client write
    -> Primary
        -> Replica A
        -> Replica B

Client read
    -> Primary            strong / fresh path
    -> Replica A or B     scalable but may be stale
```

它把两个问题分开：谁决定写入顺序，谁保存副本。单写入口让冲突处理简单很多。

### 同步复制和异步复制

同步复制要求一个或多个副本确认后，primary 才向客户端返回成功：

```text
write latency ≈ primary work + replica network RTT + replica durable write
```

优点是已确认的数据更难在 failover 时丢失；代价是写延迟更高，而且慢副本可能拖住写入。

异步复制先由 primary 返回成功，replica 稍后追上。写入快，但 primary 在复制前永久故障时，最近一段数据可能丢失。

| 选择 | 更适合什么 |
|---|---|
| 同步到 quorum | 账本、订单、需要很小 RPO 的 metadata |
| 同步到同 AZ 副本，异步跨区 | 兼顾本地 durability 与跨区延迟 |
| 全异步副本 | 可重建数据、缓存、允许明确数据窗口的系统 |

“同步”也不是一个布尔值。要说清同步到几个副本、跨不跨 AZ、确认是写到内存还是 durable storage。

### Read replica 会产生旧读

把读流量分给 replica 可以扩展吞吐，但 replication lag 会破坏 read-your-writes：

```text
User writes profile = v2 to primary
User immediately reads from lagging replica
Replica still returns v1
```

常见处理方式包括：

- 写后的一小段时间把该用户读请求路由到 primary；
- 返回 version / commit position，只读已经追到该位置的 replica；
- 对少数需要强一致的 API 固定读 primary；
- UI 先使用刚刚提交的本地结果。

不要笼统地说“数据库是强一致的”。同一个系统里，不同读路径可以有不同一致性。

---

## 4 · Failover：副本存在不代表能接管

真正困难的是把一个 replica 安全地提升为新 primary。

```text
1. Failure detector 怀疑 primary 不可用
2. 达到判定阈值，避免短暂抖动触发切换
3. 选出数据足够新的 replica
4. 对旧 primary 做 fencing
5. 提升新 primary，更新 routing / service discovery
6. 恢复流量，检查 lag 和数据差异
```

### 为什么必须 fencing

旧 primary 可能只是网络隔离，并没有真的死。若新旧 primary 同时接受写入，就出现 split brain。

Fencing 的目标是让旧节点即使恢复网络也无法继续写：

- 使用更大的 epoch / term，存储只接受最新任期的写；
- 撤销旧节点的 lease；
- 在网络或存储层切断旧节点权限；
- 对物理主机使用 STONITH 一类强制隔离机制。

“健康检查失败就提升副本”少了最关键的一半：如何确保旧主不能写。

### Failure detector 不可能完全准确

Timeout 只能说明“在规定时间内没收到响应”，不能证明节点永久死亡。阈值太短会误切，太长会拉高 RTO。

设计时要说明：

```text
probe interval
failure threshold
who makes the decision
whether quorum is required
how the old primary is fenced
```

---

## 5 · Active-Passive：平时一边服务，故障时接管

Active-Passive 常被叫作主备或双机热备。Active 承担流量，Passive 保持可接管状态。

| Standby | 平时做什么 | 切换速度 | 成本 |
|---|---|---|---|
| Cold | 只有备份和部署模板 | 分钟到小时 | 低 |
| Warm | 实例运行，数据持续同步，容量可能较小 | 数十秒到分钟 | 中 |
| Hot | 完整容量在线，数据接近实时同步 | 秒级 | 高 |

“热备”至少要保护四件事：

1. 数据在备用侧足够新；
2. 备用侧有足够容量，而不是只启动了一个空壳；
3. 入口能切换，DNS、LB 或 routing 不会指回旧端；
4. 切换流程定期演练。

没有演练过的 failover 只是文档里的愿望。

### Active-Passive 适合什么

- 单写系统，希望保持冲突模型简单；
- 第二个 region 主要用于灾备；
- 流量不值得长期支付双活复杂度；
- 业务能接受明确的 RTO。

切回原 region 也要设计。Failback 不是把 DNS 改回去那么简单，需要先同步新数据、确认旧区域状态，并再次避免双写。

---

## 6 · Active-Active：两边都服务

Active-Active（主主、多活）让多个站点同时承担流量。读路径通常不难，难的是多个位置都能修改同一份逻辑数据。

```text
Region A writes user_42 = v2
Region B writes user_42 = v3
network partition delays replication
```

系统必须定义 v2 和 v3 如何合并。常见办法没有免费的：

| 方法 | 代价 |
|---|---|
| 按 key 固定 home region | 跨区写延迟，home region 故障要迁移 ownership |
| Last-write-wins | 简单，但时钟和覆盖写可能丢业务意图 |
| Version vector / causal metadata | metadata 和实现复杂度上升 |
| CRDT | 只适合可定义合并规则的数据类型 |
| 业务冲突处理 | 语义正确，但每种 entity 都要单独设计 |

很多“全球多活”实际上是 active-active serving 加 single-writer ownership：每个 region 都接流量，但同一个用户或 partition 仍只有一个写主。这通常比真正的多主并发写更可控。

### 什么时候值得上 Active-Active

- 用户跨洲分布，跨区 RTT 已经破坏延迟目标；
- 单 region 故障必须近乎无感；
- 业务数据能按 geography 或 tenant 清楚分区；
- 团队愿意承担冲突、观测、演练和数据修复成本。

如果这些条件不成立，Active-Passive 往往更诚实。

---

## 7 · Quorum：复制数量怎样影响读写

在 N 个副本的系统中，常用 W 表示一次写需要多少副本确认，R 表示一次读查询多少副本。

```text
N = 3
W = 2
R = 2
```

当 `W + R > N` 时，读写集合理论上至少相交。但这条不等式本身不自动提供线性一致性，还需要版本选择、冲突处理、失败恢复和正确的 membership。

选择更大的 W：

- 已确认写入更耐故障；
- 写延迟取决于更多副本；
- 可写性可能在网络分区时下降。

选择更大的 R：

- 更容易读到较新版本；
- 读放大和尾延迟增加；
- 慢副本更容易进入 p99。

不要把 quorum 当成一句万能答案。面试里至少说明 N、R、W 各是多少，以及为什么符合本题的延迟和故障目标。

---

## 8 · Replica 不是 backup

副本会迅速复制正常写入，也会迅速复制误删、坏数据和勒索软件加密后的结果。Backup 保存的是某个历史时刻，可以用来回到过去。

| 机制 | 主要保护什么 |
|---|---|
| Replica | 节点故障、读扩展、快速 failover |
| Snapshot | 某个时间点的数据状态 |
| WAL / incremental backup | point-in-time recovery |
| Cross-account immutable backup | 操作失误、权限泄漏、勒索攻击 |

备份要谈 retention、加密、访问权限和 restore test。只说“每天备份”还不够：

```text
Can we restore?
How long does restore take?
What data window is lost?
Does restored data include schema and encryption keys?
```

RPO 描述能接受丢多少数据，RTO 描述多快恢复服务：

```text
RPO = 5 min   -> 最坏可丢最近 5 分钟写入
RTO = 30 min  -> 故障后 30 分钟内恢复核心功能
```

这两个数字会决定复制频率、备用容量和自动化程度。

---

## 9 · Multi-AZ 和 Multi-Region 怎么选

### Multi-AZ

同 region 内跨 AZ 延迟较低，适合做同步复制和自动 failover。它通常是高可用系统的第一步。

需要检查：

- 副本是否真的跨 AZ；
- 每个 AZ 下线后的剩余容量；
- LB、queue、cache 和 database 是否共享同一个隐含 failure domain；
- NAT、DNS、KMS 等依赖是否也有冗余。

### Multi-Region

跨 region 能处理更大的灾难，也能降低全球用户延迟，但会带来复制延迟、数据主权、成本和运维复杂度。

一个常见的渐进路径是：

```text
single instance
-> multi-instance in one AZ
-> multi-AZ active-passive / replicated
-> cross-region warm standby
-> active-active serving with controlled write ownership
```

每一步都应由新的故障目标或延迟目标推动，而不是因为架构图看起来更完整。

---

## 10 · 容量和可用性估算

### Availability budget

```text
99.9%   ≈ 43.8 min downtime / month
99.99%  ≈ 4.38 min / month
99.999% ≈ 26 sec / month
```

串联系统的可用性会相乘。请求必须经过 A、B、C 三个服务，每个都是 99.9%：

```text
0.999³ ≈ 99.7%
```

真实系统还有 shared dependency 和 correlated failure，不能只用独立概率乐观相乘。这一计算的价值是提醒我们：同步关键路径越长，整体可用性越难守。

### Replication bandwidth

如果 primary 写入峰值为 200 MB/s，异步复制到两个远端副本：

```text
outbound replication ≈ 200 MB/s × 2 = 400 MB/s
```

还要为重传、压缩率变化和副本重建留余量。新副本 bootstrap 往往比日常增量复制更重，应限速或从 snapshot 开始。

### Replication lag

```text
lag growth rate = primary write rate - replica apply rate
```

Primary 写 200 MB/s，replica 只能 apply 150 MB/s，积压每秒增长 50 MB。十分钟就是约 30 GB。此时复制“仍在工作”并不代表副本还能承担 failover。

---

## 11 · 一个选择表

| 需求 | 合理起点 | 要明确的代价 |
|---|---|---|
| 单机故障不停服 | 多实例 + health check | 状态必须外置，见 01 |
| 数据库读多写少 | Primary + read replicas | lag、read-your-writes |
| 单 AZ 故障不停服 | 跨 AZ 同步或 quorum 复制 | 写延迟、N-1 容量 |
| region 灾难后半小时恢复 | Warm standby | RPO、切换和 failback |
| 全球低延迟读 | 多 region read replicas / cache | stale read、失效传播 |
| 全球低延迟写 | 分区 ownership 或 Active-Active | 冲突、fencing、数据主权 |
| 防误删 | Versioned immutable backup | 恢复时间、存储成本 |

---

## 12 · 面试检查清单

```text
Failure domain
- 要扛进程、机器、AZ 还是 region 故障？
- 是否存在共享依赖导致所有副本一起失效？

Capacity
- 最大 failure domain 下线后，剩余容量够吗？
- 新副本重建会不会压垮在线流量？

Data
- 谁能写？复制是同步还是异步？
- 已确认写入在 failover 时会不会丢？
- read replica 的 lag 对哪些 API 可见？

Failover
- 谁判断故障，阈值是多少？
- 旧 primary 怎样 fencing？
- routing 怎样切换，多久完成？
- failback 怎样做？

Recovery
- Replica 和 backup 是否分开设计？
- RPO、RTO 是多少？
- 最近一次 restore / failover 演练是什么时候？
```

可靠性设计最后要落到一次具体故障：假设一个 AZ 现在消失，哪些请求失败、哪些数据可能丢、多久恢复、谁执行切换。能把这条时间线讲清楚，比列出一串高可用名词有用得多。

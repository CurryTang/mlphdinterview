# System Design 00 · 从题目走到架构

System design 面试不是背一张“大厂架构图”。面试官给出的题目通常很宽，真正的工作是把它收窄成几个可计算的问题，再让每个组件都有明确的理由。

这篇只讲方法。数据库、复制、消息系统和案例都放在后续章节，避免同一套知识在不同笔记里反复出现。

---

## 这套笔记怎么读

| 顺序 | 章节 | 解决的问题 |
|---|---|---|
| 00 | 本篇：设计方法 | 面试时先说什么、怎么算、怎么推进 |
| 01 | [[SystemDesign01 Stateless Service|无状态服务]] | 服务实例怎样做到可替换、可横向扩展 |
| 01B | [[SystemDesign01B Virtualization Containers|虚拟化与容器]] | VM、container 和 Linux 隔离机制怎样工作 |
| 02 | [[SystemDesign02 Database Paradigms|数据库基本范式]] | RDBMS、NoSQL、一致性和事务怎么选 |
| 03 | [[SystemDesign03 Database Scaling|数据库扩展]] | replication、sharding、partitioning 怎么落地 |
| 04 | [[SystemDesign04 Storage Systems|存储系统]] | block、file、object storage 各自存什么 |
| 05 | [[SystemDesign05 Reliability Replication|可靠性与复制]] | 主从、主主、热备、RPO/RTO 怎么设计 |
| 06 | [[SystemDesign06 Async Messaging Systems|异步消息系统]] | Queue、Pub/Sub、Kafka、Event Bus 怎么区分 |
| 07 | [[SystemDesign07 Photo Sharing Feed|图片分享与 Feed]] | 把通用方法用在经典互联网系统 |
| 08 | [[SystemDesign08 LLM Async RL Platform|异步 LLM RL 平台]] | 把通用方法用在 ML infrastructure |
| 99 | [[SystemDesign99 Glossary|术语表]] | 面试前快速查词，不承担系统教学 |

第一次读可以按编号走。准备面试时不用从头翻：先看本篇，然后挑一个案例完整讲一遍，卡住的地方再回基础章节。

---

## 一条主线：五步完成设计

```text
1. Problem navigation
   题目到底要做什么，哪些功能这次不做？

2. Back-of-the-envelope estimation
   峰值 QPS、带宽、存储、并发和内部放大是多少？

3. API + data model
   系统对外暴露什么 contract，事实数据长什么样？

4. High-level architecture
   先跑通一条最小读写链路。

5. Deep dive
   根据 NFR 找瓶颈，逐步加入 cache、replica、queue、shard 等组件。
```

顺序不是死规定。设计聊天系统时，连接模型可能要提前讲；设计账本时，事务和数据模型会比 API 更早出现。但五个问题一个都不能漏。

---

## Step 1 · Problem navigation

### Functional requirements：这次到底做什么

先把题目里的动词找出来。设计图片分享系统时，可以先确定：

```text
Must have
- 用户上传图片并发布 post
- 用户查看 home feed

Out of scope
- comments、search、ads
- video editing
- 推荐模型训练
```

功能需求最好控制在两三个核心动作。范围越大，后面的 QPS、数据模型和架构越容易互相打架。

这一步还要补几句产品语义：

- Feed 是时间排序还是推荐排序？
- 删除后要立即不可见，还是允许短暂延迟？
- 上传成功指“原图收到”，还是“所有缩略图已经可读”？
- 用户是否需要 read-your-writes？

这些问题会直接改变设计。比如“上传成功”的定义不同，API 的返回时机和状态机就不同。

### Non-functional requirements：系统要做到什么程度

不要只报一串名词。每个 NFR 最好落到数字或可观察行为：

| 维度 | 能讨论的目标 |
|---|---|
| Latency | feed read p99 < 200 ms |
| Availability | 99.99%，单 AZ 故障继续服务 |
| Durability | 已确认上传的原图不能丢 |
| Consistency | 发布者 read-your-writes；普通 feed 允许秒级延迟 |
| Freshness | 新 post 在 5 秒内进入大多数 follower 的 feed |
| Recovery | RPO < 5 min，RTO < 30 min |
| Growth | 当前负载的 10 倍仍可水平扩展 |

目标之间会冲突。同步跨区复制提高 RPO，却拉高写延迟；缓存降低读延迟，却引入旧数据；自动重试提高成功率，也可能在故障时放大流量。设计题的重点不是把每项指标都写成最好，而是说清优先级。

### 这一步的交付物

```text
Scope: upload photo + home feed
Traffic shape: read-heavy，允许突发
Latency: feed p99 < 200 ms
Availability: 99.99%
Consistency: author read-your-writes；其他用户允许 5 秒最终一致
Durability: confirmed media must survive an AZ loss
```

有了这几行，后面的架构选择才有判断标准。

---

## Step 2 · BOE：先算会改变设计的数字

Back-of-the-envelope estimation 不是算术考试。只算那些可能改变系统形态的量。

### 常用换算

```text
1 day ≈ 100,000 seconds
average QPS ≈ daily requests / 100,000
peak QPS ≈ average QPS × peak factor
bandwidth ≈ QPS × average payload size
storage growth ≈ writes/day × bytes/write × retention
concurrency ≈ QPS × average latency in seconds
```

峰值系数没有统一答案。流量平稳的内部系统可能取 2；消费产品常用 3 到 5 做第一版估算；直播、抢购和训练调度要单独看 burst。

### 一个小例子

假设：

```text
10M DAU
每人每天读 feed 20 次
每人每天发布 0.1 个 post
峰值系数 = 4
每张原图平均 3 MB
```

入口流量：

```text
feed average QPS = 10M × 20 / 100K ≈ 2K
feed peak QPS ≈ 8K

upload average QPS = 10M × 0.1 / 100K ≈ 10
upload peak QPS ≈ 40

raw media growth/day = 10M × 0.1 × 3 MB = 3 TB/day
```

这里最有用的结论不是“8K 很大”。真正改变设计的是：读远多于写；图片带宽远大于 metadata；长期存储增长比 API QPS 更值得担心。因此图片应该直传 object storage，读取走 CDN，业务服务不搬运原始 bytes。

### 别漏掉内部放大

用户入口只有 40 upload QPS，不代表系统内部也只有 40：

```text
1 upload
  -> 4 image variants
  -> content scan
  -> metadata update
  -> one PostReady event
  -> N follower timeline writes
```

Fan-out、重试、replication、索引维护都会放大内部流量。面试里应同时写 external QPS 和最重的 internal QPS。

### QPS 高低意味着什么

QPS 没有跨系统通用的“高低线”。同样是 5K QPS，内存 KV 可能很轻，复杂 SQL、GPU inference 或第三方 API 已经很重。正确问法是：单实例或单分片在目标 p99 下能安全处理多少？

```text
required instances
= peak QPS / safe QPS per instance
× headroom
```

例如服务压测后在目标 p99 下能稳定处理 600 QPS，峰值为 8K，预留 30%：

```text
8,000 / 600 × 1.3 ≈ 18 instances
```

低负载时，单体服务加一个关系数据库往往已经够用。先保证备份、监控和故障恢复。流量上升后再按瓶颈处理：

| 现象 | 先检查什么 | 常见动作 |
|---|---|---|
| 重复读拖慢数据库 | 热点、查询计划、连接池 | cache、read replica、索引 |
| 单服务 CPU 饱和 | 单实例吞吐、p99 | 无状态化、水平扩容 |
| 写入突发 | 峰值与消费速率 | queue、batch、backpressure |
| 单库写入到顶 | 热 partition、事务边界 | shard、按访问模式拆表 |
| 大文件占满 API 带宽 | payload size | signed URL、object storage、CDN |

“QPS 高所以加 Kafka”不是推导。Kafka 解决事件传递、缓冲和重放；它不会让慢 consumer 凭空变快。

### Cache 和 TTL 怎么估

先算 cache 是否值得：

```text
DB QPS after cache
= total read QPS × (1 - hit rate)
```

如果数据库只能安全承担 2K read QPS，而峰值读流量为 8K：

```text
required hit rate >= 1 - 2K / 8K = 75%
```

容量估算从工作集出发：

```text
cache bytes
≈ active keys × average value bytes × metadata overhead × replica factor
```

TTL 不是越长越好。它同时受四个量约束：业务允许多旧、数据多久变化一次、失效通知是否可靠、缓存击穿时后端能否扛住。一个实用起点是：

```text
TTL <= allowed staleness
TTL 加 10%~20% jitter，避免大量 key 同时过期
热点 key 使用 request coalescing 或 single-flight
```

权限、余额、封禁状态等数据通常不能只靠长 TTL。可以缩短 TTL，再配合主动失效；真正的 source of truth 仍在数据库。

---

## Step 3 · 先定 API 和事实数据

API 用来暴露业务语义，不是把数据库表直接搬到 HTTP 上。

```http
POST /v1/posts
-> { post_id, upload_url, expires_at }

POST /v1/posts/{post_id}/publish
-> { operation_id, status }

GET /v1/feed?cursor=...
-> { items, next_cursor }
```

写 API 时至少说明：

- 请求是否幂等，idempotency key 放在哪里；
- 同步返回结果还是 operation handle；
- pagination 用 offset 还是 cursor；
- 失败后客户端能否安全重试。

数据模型先找 source of truth，再列派生数据：

```text
Post                  authoritative
- post_id
- author_id
- media_object_key
- status
- created_at

TimelineEntry         derived / rebuildable
- viewer_id
- rank_key
- post_id
```

如果数据丢了可以从 Post 和 follow graph 重建，TimelineEntry 就不应该承担唯一真相。这个区分会影响 TTL、备份和一致性策略。

数据库选择见 [[SystemDesign02 Database Paradigms|02 数据库基本范式]]；分片和副本见 [[SystemDesign03 Database Scaling|03 数据库扩展]]；图片、文件和模型权重见 [[SystemDesign04 Storage Systems|04 存储系统]]。

---

## Step 4 · 先画最小闭环

不要一上来铺满 Kafka、Redis 和十几个微服务。先让核心请求完整地走一遍：

```system-design-overview-visual
```

图中第一行是同步闭环：

```text
Client -> Gateway -> Stateless Service -> Primary Store
```

支线组件必须回答一个具体问题：

| 组件 | 加入它的理由 |
|---|---|
| Cache | 重复读已经压到数据库或延迟目标 |
| Queue / Event Log | 工作不必同步完成，需要削峰、重试或广播 |
| Worker | 慢任务要独立扩缩容 |
| Replica | 需要故障切换或读扩展 |
| Shard | 单机容量或写吞吐已经成为边界 |
| Object storage | 数据对象大、访问方式简单、需要高 durability |

画完后把四条路径讲清楚：正常读取、正常写入、异步工作和失败恢复。只画组件名而不讲数据怎么走，图再复杂也没有用。

---

## Step 5 · 用 NFR 选择 deep dive

Deep dive 不是继续往图上堆方框，而是挑最可能决定成败的一两个问题。

| 题目暴露的压力 | 应该深入什么 | 对应笔记 |
|---|---|---|
| 实例故障、跨 AZ、RPO/RTO | failover、replication、fencing | [[SystemDesign05 Reliability Replication|05 可靠性与复制]] |
| 服务不能水平扩展 | session、文件、任务如何外置 | [[SystemDesign01 Stateless Service|01 无状态服务]] |
| 数据模型和一致性难选 | transaction boundary、access pattern | [[SystemDesign02 Database Paradigms|02 数据库基本范式]] |
| 单库容量或吞吐到顶 | replica、partition key、resharding | [[SystemDesign03 Database Scaling|03 数据库扩展]] |
| 任务慢、突发大、下游很多 | queue、consumer group、outbox | [[SystemDesign06 Async Messaging Systems|06 异步消息系统]] |
| 图片或大文件占主流量 | direct upload、CDN、object storage | [[SystemDesign04 Storage Systems|04 存储系统]] |

### 冗余只记住一个入口

当某个组件失效会让核心功能停止，而且恢复时间超过目标 RTO，就要考虑冗余。先确定 failure domain，再谈副本数：进程、机器、机架、AZ 还是 region。

```text
N-1 capacity check:
剩余副本在一个 failure domain 下线后，仍能承受峰值流量吗？
```

Primary-Replica、Active-Passive 和 Active-Active 的区别以及 failover 细节统一放在 [[SystemDesign05 Reliability Replication|05 可靠性与复制]]，本篇不再重复展开。

### 异步也只记住一个入口

当工作不需要在请求返回前完成，并且需要持久交接、缓冲或独立重试，可以考虑 Queue / Event Log。先回答三个问题：

```text
什么时候算 broker 已经接住任务？
消息重复时 consumer 是否幂等？
生产速率长期高于消费速率时，系统怎样 backpressure？
```

Queue、Pub/Sub、Kafka、RabbitMQ、Event Bus、Webhook 和 outbox 统一放在 [[SystemDesign06 Async Messaging Systems|06 异步消息系统]]。

---

## 面试时怎样分配时间

以 45 分钟为例：

```text
0 - 6 min    收窄功能和 NFR
6 - 11 min   BOE，找出真正的大数字
11 - 17 min  API 与数据模型
17 - 27 min  最小架构与主要读写路径
27 - 40 min  针对一个瓶颈 deep dive
40 - 45 min  failure、trade-off 和扩展问题
```

不要为了守时间表强行打断有价值的讨论。它的用途是防止自己在需求阶段聊二十分钟，或者刚画完方框就到点。

---

## 最终检查

### Requirements

- 核心功能是否只有两三个？
- NFR 是否带数字或清晰语义？
- 哪些功能明确不做？

### Numbers

- 平均值和峰值是否分开？
- 是否算了 payload、retention 和内部 fan-out？
- 每个容量结论是否写了假设？

### Architecture

- source of truth 在哪里？
- cache、timeline、index 等派生数据能否重建？
- 每个新增组件对应哪个瓶颈或 NFR？
- 正常路径和失败路径是否都讲过？

### Reliability

- failure domain 是什么？
- 单副本故障后是否还有足够容量？
- retry 是否有超时、上限、backoff 和幂等保护？
- 备份是否真的做过恢复演练？

最后用两句话收尾：当前设计优先保护什么；为了它接受了什么代价。能说清这两句，通常说明整套设计是自己推出来的。

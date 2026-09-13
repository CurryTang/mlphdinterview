# System Design 07 · 设计图片分享与 Home Feed 流系统 (Photo Sharing & Home Feed)

课程位置：[[SystemDesign09 Consistent Hashing|09 一致性哈希]] → 本篇 → [[SystemDesign08 LLM Async RL Platform|08 异步 LLM RL 平台]]

设计类似 Instagram / Twitter 的海量图片分享与社交信息流（Feed 流）系统，核心挑战在于：**读写比极端倾斜（100:1）、海量非结构化图片二进制流传输、以及长尾粉丝分布（大 V 场景）引发的写扩散与读扩散冲突**。系统必须在物理架构上实现**控制面与数据面彻底分离**，并在数据层采用**推拉结合的混合分发模型**。

---

## 第一部分：需求定义与系统边界 (Requirements & Scope)

### 1.1 · 功能需求 (Functional Requirements)
1. **图片上传与发布**：用户可上传一张高清图片，附带文本说明（Caption）与地理位置标签；
2. **社交关系图谱**：用户可以关注（Follow）或取消关注（Unfollow）其他用户；
3. **个性化 Home Feed 生成**：用户可以分页下拉查看其关注的所有创作者发布的逆序时间线（Chronological Timeline）。

```text
【明确系统边界 (Out of Scope)】：
- 短视频 / Reels / 24小时快照 (Stories) / 直播音视频；
- 私信即时通讯 (Direct Messaging / IM)；
- 探索与发现页推荐算法 (Explore / Ranking ML Engine)；
- 复杂的客户端滤镜渲染引擎；
- 点赞与评论系统作为后续独立读写链路展开。
```

### 1.2 · 非功能需求与 SLA 约束 (Non-Functional Requirements)
- **高可用性 (High Availability)**：系统整体可用性达 $99.99\%$（全年停机 $< 52$ 分钟）。在发生网络分区或节点故障时，**读链路（刷 Feed）优先保障可用性（遵循 CAP 定理中的 AP 模型）**，允许暂时展示数秒前缓存的数据；
- **极低读取延迟 (Low Latency)**：Feed 接口元数据返回延迟 $\mathbf{p99 < 200\text{ ms}}$（不含大图片 CDN 网络传输时间）；发帖上传会话创建延迟 $< 100\text{ ms}$；
- **数据鲜度与时序一致性 (Freshness & Consistency)**：
  - 普通创作者发帖后，活跃粉丝在 $\le 5\text{ 秒}$ 内可见；
  - 创作者本人发帖后刷新个人主页，必须严格满足**写后读一致性（Read-Your-Writes Consistency）**；
- **图片持久性与容灾 (Durability)**：已完成 Commit 的原图与成品多规格衍生图跨多可用区（Multi-AZ）冗余持久化，承诺 **$99.999999999\%$（11 个 9）** 数据持久性。

---

## 第二部分：容量评估与量化常数推导 (Back-of-the-Envelope / BOE)

### 2.1 · 用户规模与并发 QPS 估算

| 业务维度 | 基础假设与量化基准 | 吞吐推导与系统规划 |
|---|---|---|
| **用户规模** | 总注册用户 10 亿（1B），日活跃用户（DAU）1 亿（100M） | 典型超大规模消费级社交平台 |
| **发帖写 QPS (Write Path)** | 假设 $20\%$ DAU 每天发布 1 张照片 $\to \mathbf{20\text{M Posts / 天}}$ | 平均写 QPS $\approx 231\text{ QPS}$；常态峰值写 QPS $\approx 1,200\text{ QPS}$；系统按 $\mathbf{5,000\text{ Peak Write QPS}}$ 安全容量规划。 |
| **Feed 刷流读 QPS (Read Path)** | DAU 平均每人每天刷新 Feed 10 次；社交网络读写比通常在 $50:1 \sim 100:1$ | 每日总读请求 $\approx 10\text{ 亿次/天}$；平均读 QPS $\approx 11,600\text{ QPS}$；晚间峰值读 QPS $\approx \mathbf{40,000 \sim 50,000\text{ QPS}}$。 |

#### 吞吐公式定量推导：
- **平均发帖写 QPS**：
  $$\text{平均写 QPS} = \frac{20 \times 10^6}{86,400\text{ s}} \approx 231\text{ QPS}$$
  峰均比按 $4\sim 5$ 倍计，常态峰值写 QPS $\approx 1,200\text{ QPS}$；为应对突发秒级尖峰与客户端重试，微服务层按 $\mathbf{5,000\text{ Peak Write QPS}}$ 部署。
- **平均 Feed 刷流读 QPS**：
  $$\text{每日总读请求} = 100\text{M} \times 10 = \mathbf{1,000,000,000\text{ 次/天}}$$
  $$\text{平均读 QPS} = \frac{10^9}{86,400\text{ s}} \approx 11,600\text{ QPS}$$
  晚间高峰峰均比取 $3\sim 4$ 倍，**峰值读 QPS 达到 $\mathbf{40,000 \sim 50,000\text{ QPS}}$**。

### 2.2 · 存储容量与增长规划

#### 1. 媒体二进制数据（图片存储）
- 单张原图平均体积约为 $2\text{ MB}$；
- 异步媒体流水线对原图进行裁剪与多规格转码，生成 3 种衍生规格：
  - 缩略图（Thumbnail，网格预览）：$50\text{ KB}$；
  - 移动端中图（Medium，Feed 瀑布流）：$200\text{ KB}$；
  - 高清大图（Large，单图详情展示）：$800\text{ KB}$；
  - 多规格衍生图合计体积约为 $1\text{ MB}$。
- **每日新增物理存储增量**：
  $$\text{Daily Media Storage} = 20\text{M} \times (2\text{ MB 原图} + 1\text{ MB 衍生图}) = \mathbf{60\text{ TB / 天}}$$
- **年化物理存储增量**：
  $$\text{Annual Media Storage} = 60\text{ TB/天} \times 365 \approx \mathbf{21.9\text{ PB / 年}}$$

#### 2. 关系型元数据存储 (Metadata Storage)
单条 Post 包含 `(post_id, author_id, caption, cdn_urls, lat, lon, created_at)` 紧凑结构约 $500\text{ Bytes}$：
$$\text{Daily Metadata} = 20\text{M} \times 500\text{ B} = \mathbf{10\text{ GB / 天}} \implies \mathbf{3.65\text{ TB / 年}}$$
单机关系型数据库几年内即会突破物理单盘限制与 B+ 树索引性能拐点，**元数据库必须按 `user_id` 进行水平分库分表（Z 轴 Sharding）**。

### 2.3 · 网络出口带宽与 CDN 流量卸载
- 客户端每次刷新 Feed 拉取 10 条帖子，每条展示 1 张中图（$150\text{ KB}$），单次刷新总数据量约为 $1.5\text{ MB}$；
- **全站峰值读流出带宽**：
  $$\text{Peak Read Egress} = 50,000\text{ QPS} \times 1.5\text{ MB} \times 8\text{ bit} = \mathbf{600\text{ Gbps}}$$
- **CDN 边缘卸载**：
  由于图片具有高度的时间局部性与空间局部性，绝大多数图片直接缓存在全球 CDN 边缘 PoP 节点。**在保证 CDN 缓存命中率 $\ge 95\%$ 的前提下，源站对象存储的真实回源网络带宽被压缩至 $\le \mathbf{30\text{ Gbps}}$**。

---

## 第三部分：全局系统高层架构与组件解耦 (High-Level Architecture)

```photo-sharing-architecture-visual
```

### 3.1 · 核心架构原则：控制面与数据面彻底分离

在海量多媒体系统设计中，**首要铁律是：绝不允许业务微服务（API Gateway / Application Servers）中转代理大文件二进制数据流**。
- **反模式（Bytes through API）**：客户端直接向 API 服务器上传图片，API 服务器将数据流写入内存并转发给对象存储。单次上传霸占一个工作线程与长连接达数秒，极易耗尽网关连接池，且引发严重的内存浅拷贝与频繁 Full GC；
- **工业级正解（Pre-signed Direct Upload）**：控制面（Control Plane）与数据面（Data Plane）物理完全隔离。API 仅负责会话鉴权并返回带有单次写入权限的 **S3 预签名直传 URL（Pre-signed URL）**；客户端绕过所有微服务，直接向分布式对象存储发起 `HTTP PUT`。

```text
                           【控制面 · 协调元数据】
                           Client ──► API Gateway ──► Upload Service ──► Metadata DB (PENDING)
                             │
                             │ (获取短期 Pre-signed URL)
                             ▼
【数据面 · 图片直传与转码】  Client ──────────────────────────────────────────► Raw Storage (S3 Staging)
                                                                                  │
                                                                                  ▼ (S3 Event Notify)
Processed Storage (CDN Origin) ◄── Media Processor (Resize / WebP / NSFW) ◄───────┘
          ▲
          │ (95%+ 边缘命中)
      CDN PoP ◄────────────────────────────────────────────────────────────────── Client (加载图片)
```

### 3.2 · 核心组件职责与数据流闭环

1. **客户端 (Web / iOS / Android)**：
   - 上传时：向网关协商会话 $\to$ 凭 Pre-signed URL 直传原图 $\to$ 向网关 Commit 提交元数据；
   - 刷流时：先向网关发起 `GET /feed` 拉取轻量 Post ID 列表，再按需请求 CDN 边缘加载 WebP 图片；
2. **API Gateway & Load Balancer**：
   - 统一入口，负责 TLS 卸载、动态路由、TraceID 全链路追踪注入；
   - 与 **Authz / Rate Limiter** 协同，通过 Redis 令牌桶算法拦截非法爬虫与高频刷量；
3. **Upload Service (发布微服务)**：
   - 校验文件格式（MIME Type）与元数据参数；
   - 生成 64-bit 全局唯一 Snowflake `post_id`；
   - 在元数据库中插入初始状态为 `PENDING` 的记录，并向客户端返回 15 分钟时效的预签名 URL；
   - 接收客户端上传完成的 Commit 回调，将状态推进为 `PROCESSING`；
4. **Raw Object Storage (S3 Staging 暂存区)**：
   - 专门用于接收客户端直传的高清单图（平均 2MB）；
   - 配置 **24 小时生命周期清理规则（Lifecycle Expiration Rule）**：若客户端直传中断未发送 Commit 回调，对象存储在 24 小时后自动硬删除该文件，规避孤儿脏数据泄露；
5. **Media Processor Pipeline (异步媒体转码集群)**：
   - 监听对象存储发出的 `s3:ObjectCreated:*` 事件通知（通过 SQS / Kafka 缓冲）；
   - 提取图片 EXIF 经纬度与拍摄参数，清除隐私信息；
   - 启动多协程生成 Thumbnail / Medium / Large 三套多尺寸 WebP/AVIF 规格；
   - 调用图像涉黄与违规安全审查模型（NSFW Filter）；
   - 将处理后的成品落盘至成品对象存储，并将元数据库状态正式推进为 `READY`；
6. **Processed Object Storage & CDN Origins**：
   - 存放最终供用户消费的多规格图片，文件命名采用不可变哈希（Content-Addressable）；
   - 作为全球 CDN 的回源站点，多 AZ 物理冗余；
7. **Metadata DB & Transactional Outbox**：
   - 持久化元数据。状态更新为 `READY` 的本地事务内，向 `outbox` 表原子插入一条 `PostReady` 领域事件；
   - CDC 引擎（Debezium）异步拉取 Binlog，将事件投递至 Kafka `post-events` 主题；
8. **View / Feed Service (信息流微服务)**：
   - 承载高并发刷流请求（50,000 Peak QPS）；
   - 采用 **Hybrid 混合推拉模型**：从当前用户的 Redis Inbox 中拉取预计算普通作者流，同时并发 `MGET` 关注列表中大 V 的专属 Outbox；
   - 在内存中执行快速多路堆归并（Online Heap Merge），截取 Top 20，并执行 Hydration 批量元数据水化与权限过滤。

---

## 第四部分：数据实体与存储模型设计 (Data Entities & Storage)

### 4.1 · 核心数据实体与 Schema 设计

```sql
-- 1. 用户表 (按 user_id 水平分库分表)
CREATE TABLE users (
    user_id         BIGINT UNSIGNED NOT NULL PRIMARY KEY,
    username        VARCHAR(32) NOT NULL UNIQUE,
    follower_count  INT UNSIGNED NOT NULL DEFAULT 0,
    is_celebrity    TINYINT(1) NOT NULL DEFAULT 0, -- 粉丝数 >= 50,000 判定为大 V
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 2. 帖子元数据表 (按 user_id 分片，保留单作者单片事务与写后读一致性)
CREATE TABLE posts (
    post_id         BIGINT UNSIGNED NOT NULL PRIMARY KEY, -- Snowflake ID 包含毫秒时间戳
    user_id         BIGINT UNSIGNED NOT NULL,             -- 分片键 Sharding Key
    caption         VARCHAR(1000) DEFAULT '',
    status          VARCHAR(16) NOT NULL DEFAULT 'PENDING', -- PENDING -> PROCESSING -> READY -> DELETED
    latitude        DECIMAL(10, 8) NULL,
    longitude       DECIMAL(11, 8) NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_created (user_id, created_at DESC)
) ENGINE=InnoDB;

-- 3. 媒体成品衍生表 (一对多关联，存储 CDN Key)
CREATE TABLE media_assets (
    asset_id        BIGINT UNSIGNED NOT NULL PRIMARY KEY,
    post_id         BIGINT UNSIGNED NOT NULL,
    variant_type    VARCHAR(16) NOT NULL, -- THUMBNAIL | MEDIUM | LARGE
    cdn_key         VARCHAR(255) NOT NULL,
    width           INT UNSIGNED NOT NULL,
    height          INT UNSIGNED NOT NULL,
    size_bytes      INT UNSIGNED NOT NULL,
    INDEX idx_post_id (post_id)
) ENGINE=InnoDB;

-- 4. 社交关注关系表 (双向切片或全局关系存储)
CREATE TABLE follows (
    follower_id     BIGINT UNSIGNED NOT NULL,
    followee_id     BIGINT UNSIGNED NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    state           TINYINT(1) NOT NULL DEFAULT 1, -- 1: 正常关注, 0: 已取关
    PRIMARY KEY (follower_id, followee_id),
    INDEX idx_followee (followee_id, follower_id)
) ENGINE=InnoDB;
```

### 4.2 · 关系型数据库分库分表 vs NoSQL 选型权衡

| 评估维度 | 关系型分库分表 (Sharded MySQL/PostgreSQL) | 分布式 NoSQL (Cassandra / DynamoDB) | 生产决策结论 |
|---|---|---|---|
| **事务与状态机保障** | 单分片内严格保障 **ACID 本地事务**。状态机（`PENDING` $\to$ `READY`）流转配合 Transactional Outbox 杜绝数据不一致 | 仅提供行级原子性，缺乏复杂的跨表事务与多状态机回滚能力 | **帖子元数据选用 Sharded MySQL**：发帖、修改文案、防重提交需要强一致事务边界 |
| **主键与时序排序** | 原生 B+ 树复合聚簇索引 `(user_id, created_at DESC)` 范围扫描极佳 | 基于 Partition Key 与 Clustering Key，时序倒序检索同样为 $O(1) \sim O(\log N)$ | 均可胜任，但 MySQL 对分页和过滤生态更成熟 |
| **关系关注图谱** | 关系型多表关联需要双向索引（按 `follower_id` 查关注的人，按 `followee_id` 查粉丝列表），跨分片查询复杂 | 需维护两张异构只读表（`user_followings` 与 `user_followers`），依赖应用层双写或 CDC | **关注关系采用关系型双向分片表**；大 V 粉丝列表由专有服务加载至 Redis Set 缓存 |
| **时间线存储** | 关系型单表无法承受亿级用户 Inbox 的毫秒级读写吞吐 | 宽列模型适合存储时序流，但内存缓存性能逊于内存数据库 | **Timeline 收取箱不落关系型 DB**，全部采用 **Redis Sorted Set (ZSet)** 纯内存物化 |

---

## 第五部分：核心技术深度攻坚 (Deep Dives & Engineering Tradeoffs)

### 5.1 · 大文件预签名直传（Pre-signed URL）与完整状态机

```text
Client                  API Gateway / Upload Svc              Raw S3 Staging            Media Workers / DB
  │                                │                                │                           │
  ├─ 1. POST /posts/session ──────►│                                │                           │
  │    (metadata, mime_type)       ├─ 2. 插入 PENDING 状态记录 ──────┼──────────────────────────►│
  │                                ├─ 3. 生成 15min Pre-signed URL   │                           │
  │◄─ 4. 返回 post_id + Signed URL ─┘                                │                           │
  │                                                                 │                           │
  ├─ 5. HTTP PUT image bytes (直传原图，绕过 API 服务器) ───────────►│                           │
  │◄─ 6. HTTP 200 OK ───────────────────────────────────────────────┘                           │
  │                                                                 │                           │
  ├─ 7. POST /posts/commit ───────►│                                │                           │
  │                                ├─ 8. 更新状态为 PROCESSING ──────┼──────────────────────────►│
  │◄─ 9. 返回 202 Accepted ────────┘                                │                           │
  │                                                                 ├─ 10. S3 Event 异步通知 ──►│
  │                                                                 │                           ├─ 11. 裁剪/WebP/NSFW
  │                                                                 │                           ├─ 12. 更新状态 READY
  │                                                                 │                           └─ 13. 发出 PostReady
```

#### 1. 为什么预签名直传是海量吞吐的唯一选择？
1. **彻底消除 API 服务器带宽瓶颈**：
   按峰值写 QPS $1,200$、单图 $2\text{ MB}$ 计算，若经由 API 代理中转，入口瞬时网络吞吐达：
   $$1,200\text{ QPS} \times 2\text{ MB} \times 8\text{ bit} = \mathbf{19.2\text{ Gbps}}$$
   需要数十台大型应用服务器仅充当“字节搬运工”，成本极高且网卡容易拥塞；采用 S3 Pre-signed 直传后，API 仅收发几百字节的 JSON 控制报文，集群规模削减 90%。
2. **连接解耦与防止线程枯竭**：
   移动端上传原图受制于用户弱网环境，耗时通常需 $1\sim 5\text{ 秒}$。直传将慢速长连接完全推至对象存储集群，API 服务器工作线程可在 $< 50\text{ ms}$ 内释放。

#### 2. 上传异常与孤儿文件回收（Garbage Collection）
若客户端在调用 `PUT` 直传成功后突发断网或应用崩溃，未向 Upload Service 触发 `commit`：
- **元数据侧**：Upload Service 定期扫描 DB 中处于 `PENDING` 状态且超过 2 小时的历史记录，标记为 `EXPIRED / FAILED`；
- **对象存储侧**：Raw S3 Bucket 开启 **S3 Lifecycle Rules**，任何对象在创建满 24 小时后由存储引擎自动物理删除，彻底消灭孤儿文件泄露。

---

### 5.2 · Feed 流生成核心——Push、Pull 与 Hybrid 混合推拉模型

Feed 流的本质是：**如何将创作者发布的内容（Outbox），高效、保真、低延迟地投递到所有关注者的视线中（Inbox）**。

#### 1. Push 模型（写扩散 / Fan-out-on-write）
- **实现机制**：系统为每个接收者（Follower）维护一个专属的收取箱（Inbox）。博主发帖时，Worker 查出其全部粉丝，逐一将 `post_id` 写入每个粉丝的 Inbox 中（Redis ZSet，保留最新 800 条）；
- **数学复杂度**：
  - **写入放大**：$\mathbf{O(F)}$（$F$ 为博主粉丝数）；
  - **读取复杂度**：$\mathbf{O(1)}$（直接单点点查自己的 Redis Inbox，延迟 $< 2\text{ ms}$）；
- **致命隐患（大 V 场景雪崩）**：
  如果一个拥有 6000 万粉丝的大 V（如 Cristiano Ronaldo）发帖，单次写操作将向消息队列投递 6000 万个任务。即使 Worker 集群每秒能处理 10 万次写入，**队列也要积压长达 10 分钟**，下游 Redis 集群被写流量瞬间打满；同时，6000 万粉丝中有大量长达半年未登录的“僵尸粉”，为其物化时间线纯属浪费内存。

#### 2. Pull 模型（读扩散 / Fan-out-on-read）
- **实现机制**：系统不维护任何收件箱。每个创作者只维护一个专属的发件箱（Outbox）。用户刷新 Feed 时，系统实时查询其关注的所有人（基数为 $N$），并发拉取这 $N$ 个人的 Outbox 前 20 条记录，并在应用层内存中做多路归并排序（K-way Heap Merge）；
- **数学复杂度**：
  - **写入复杂度**：$\mathbf{O(1)}$（发帖仅写一次，零写放大）；
  - **读取复杂度**：$\mathbf{O(N \log N + K \log N)}$；单次读触发 $N$ 次数据库/缓存网络调用；
- **致命隐患（长尾读延迟与计算风暴）**：
  若用户关注了 1000 人，单次刷流并发打出 1000 次 RPC 查询。整体响应时间取决于最慢的一个节点（$T_{\text{total}} = \max T_i$），在峰值 $50,000\text{ QPS}$ 下，应用服务器的 CPU 将彻底崩溃在并发网络开销与内存堆排序上。

#### 3. 工业级 Hybrid 混合推拉模型（Instagram / Twitter 最优解）

基于社交网络长尾幂律分布，工业界对作者进行**动静分流**：

```text
                                [ 创作者发布新帖 ]
                                        │
                       ┌────────────────┴────────────────┐
                       ▼                                 ▼
             【普通创作者 (粉丝 < 5万)】              【头部大 V (粉丝 ≥ 5万)】
                       │                                 │
                       ▼ (Push 写扩散)                   ▼ (Pull 读扩散)
            遍历粉丝列表 (过滤僵尸粉)                 仅写入大 V 个人 Outbox (Redis/DB)
                       │                                 │ (不扩散，写耗时 < 5ms)
                       ▼                                 │
            异步推入活跃粉丝的专属 Inbox                   │
            (Redis ZSet, 上限 800 条)                     │
                                                         │
─────────────────────────────────────────────────────────┼────────────────────────
                                                         │
                               [ 关注者刷新 Home Feed ]  │
                                        │                │
           ┌────────────────────────────┴────────────────┘
           ▼                                             ▼
  【读取自身专属 Inbox】                        【并发查询关注列表中所有大 V】
  单点点查 Redis ZSet 提取                      MGET 大 V 专属 Outbox 最新发布
  普通创作者预计算队列 (耗时 < 2ms)              (受限常数 M 个，通常 < 30)
           │                                             │
           └────────────────────────────┬────────────────┘
                                        ▼
                       【在线多路双流堆归并 (Heap Merge)】
                       快速归并挑选时间戳最新的 Top 20 条 post_id
                                        │
                                        ▼
                       【元数据水化与权限过滤 (Hydration)】
                       批量 MGET 填充点赞数、图片 CDN Key，过滤黑名单与私密
                                        │
                                        ▼
                                组装最终 JSON 返回客户端
```

- **大 V 动态阈值划分**：
  - 设定固定阈值（如粉丝数 $F_{\text{threshold}} = 50,000$）。在关注计数变更时动态维护标志位 `is_celebrity`；
  - **普通创作者**：走 **Push 模式**，仅写扩散至其关注者的 Inbox；
  - **头部大 V**：走 **Pull 模式**，**严禁写扩散**，仅单点写入该大 V 的个人 Outbox。
- **在线双流堆归并 (Online Merge)**：
  - 用户刷新 Feed 时，应用层先从自身 Redis Inbox 取出候选流（已排好序）；
  - 提取用户关注的大 V 列表（普通人关注的大 V 数量极其有限，通常 $M < 30$ 个），向这 $M$ 个大 V 的 Outbox 发起 `MGET`；
  - 在内存中利用优先级队列（Min-Heap）将 Inbox 流与大 V 增量流进行就地双路归并，耗时 $< 5\text{ ms}$。
- **活跃用户裁剪策略 (Active Follower Inboxing)**：
  - 即使对普通创作者发帖，也**仅向最近 7 天内有登录行为的活跃粉丝进行写扩散**；
  - 沉睡粉丝重新打开 App 时触发**冷启动增量补拉（Lazy Catch-up）**：后台异步任务按 Pull 模式拉取其所有关注者在离线期间发布的新帖，重建其 Redis Inbox。此优化可直接削减时间线缓存集群 **$70\%$ 以上的无效内存开销**。

---

### 5.3 · Timeline 只存 ID 与只读元数据水化模式 (Hydration Pattern)

在时间线存储设计中，**绝对禁止在每个粉丝的 Inbox 中冗余存储帖子的完整 JSON 内容（如 Caption、图片 URL、作者头像）**：
1. **存储体积灾难**：若冗余存储单条 500B 的完整内容，1 亿用户 $\times$ 800 条 Inbox 需消耗数十 TB 的超昂贵 Redis 内存；
2. **数据变更无法同步**：若创作者修改了文案或删除了违规帖子，写扩散无法高效地遍历并更新全网所有粉丝 Inbox 中的历史副本；
3. **权限穿透安全漏洞**：若某创作者将账号设为私密或拉黑了某粉丝，预先物化的冗余内容无法实时收回，造成隐私内容泄露。

#### 工业级水化三步走架构 (Hydration Pipeline)：
- **Step 1 · 纯 ID 候选提取**：
  Inbox 中仅存储轻量元组：`(post_id, author_id, timestamp)`，单条仅占 $24\text{ Bytes}$；
- **Step 2 · 分布式只读缓存批量填充 (MGET)**：
  通过堆归并确定最终展示的 20 个 `post_id` 后，Feed Service 向 Redis / Memcached 集群发起批量 `MGET posts:meta:<post_id>`，一次网络往返批量获取 20 条帖子的完整元数据；
- **Step 3 · 读时安全鉴权 (Read-time Guards)**：
  在最终组装返回前，同步校验当前登录用户与作者的最新关系：检查是否已被作者拉黑、作者是否被系统封禁、帖子是否已被逻辑删除（`status == 'DELETED'`），确保可见性强一致性。

---

### 5.4 · 全球多级缓存与 CDN 边缘回源架构

```text
Client (手机端)
   │
   ├─ 1. 本地磁盘 LRU 缓存 (命中直接展示，零网络开销)
   ▼
Edge CDN PoP (就近边缘节点)
   │
   ├─ 2. 边缘缓存命中率 ≥ 95% (响应延迟 < 20ms)
   ▼
Regional Cache / Shield (区域中心汇聚层)
   │
   ├─ 3. 合并回源请求 (Request Collapsing，防缓存击穿)
   ▼
Processed Object Storage (S3 源站，仅承受 ≤ 5% 回源流量)
```

1. **不可变内容指纹命名（Content-Addressable Hashing）**：
   所有经过转码的成品图片，其存储文件名强制嵌入内容哈希（如 `cdn.domain.com/photos/a8f9c2d1_medium.webp`）。若用户重新上传或图片变更，生成全新 URL，**彻底消灭 CDN 节点执行主动 Cache Purge / Invalidation 的复杂性**；
2. **激进的 HTTP 缓存控制头配置**：
   对象存储返回严格的静态响应头：
   `Cache-Control: public, max-age=31536000, immutable`
   允许客户端与全球边缘 CDN 节点持久缓存 1 年以上；
3. **防击穿请求合并 (Request Collapsing / Singleflight)**：
   当某个大 V 发布爆款图片，全球数万客户端在同一秒内首次向 CDN 请求该图时，CDN 边缘节点与反向代理层启用 Singleflight 机制：**对相同 URL 的未命中请求仅放行 1 个回源连接至 S3 源站**，其余并发请求就地挂起等待结果共享，防止瞬间打爆 S3 的网络出口与 IOPS。

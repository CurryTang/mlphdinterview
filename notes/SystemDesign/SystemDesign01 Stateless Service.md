# 系统设计通用解题框架与实战 (System Design Blueprint: FR + Non-FR + QPS + Diagram + Deep Dive)

```system-design-overview-visual
```

---

## 核心总览：分布式系统设计的统一解题框架 (The Universal 5-Stage Blueprint)

在工业界架构设计与系统设计面试中，最常见的失分与陷阱不是“缺少对某个中间件的了解”，而是**拿到问题后缺乏系统性的推演框架，脱离业务边界与物理数字，过早陷入局部细节或盲目堆砌组件**。

分布式系统设计不是组件的随意拼装，而是**在明确的功能边界、SLA 可用性目标与物理资源硬边界约束下，沿着系统瓶颈演进做出的严密权衡取舍**。

一个工业级标准的架构设计推进过程，严格遵循由浅入深、逻辑闭环的标准阶段：

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        系统设计五步通用解题方法论                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
  1. 需求划界 ──► 2. 容量估算 ──► 3. 架构拓扑与建模 ──► 4. 瓶颈攻坚 ──► 5. 时序闭环
    (FR/Non-FR)     (QPS/Sizing)    (Diagram/Schema)    (Deep Dives)       (Trace)
```

### 标准 45 分钟时间把控节奏

| 阶段 | 核心任务与时间分配 | 产出物与检查标准 | 常见反模式预警 |
|---|---|---|---|
| **Step 1: FR & Scope** | **00 ~ 05 min**：明确核心用例与系统边界 | 核心功能清单（2~3 个核心 API）、Out of Scope 清单 | 切忌直接开始画图；切忌把所有周边辅助功能全盘揽入 |
| **Step 2: Non-FR & SLAs** | **05 ~ 10 min**：可用性、延迟、读写模型与一致性 | 可用性目标（$99.99\%$）、P99 延迟 SLA、读写比（$100:1$）、强/最终一致性界定 | 泛泛而谈“高性能、高可用”，未给出具体可度量的指标数字 |
| **Step 3: QPS & Capacity Sizing** | **10 ~ 15 min**：读写吞吐、存储增量、内存缓存与网络带宽 | 平均/峰值 QPS、5 年存储数据量（TB）、热点缓存容量（80/20 法则）、网卡带宽 | 仅估算 QPS 而忽略存储与内存容量，导致后续选型缺乏推导支撑 |
| **Step 4: Architecture Diagram & Schema** | **15 ~ 25 min**：分层拓扑图、数据表结构、核心网络协议 | 端到端分层架构图（网关/无状态计算/缓存/持久化/队列/Worker）、核心数据表 DDL | 混淆计算与存储边界；把状态保存在应用服务器本地内存 |
| **Step 5: Deep Dives (Trade-offs)** | **25 ~ 40 min**：攻坚核心瓶颈，提供 2+ 方案对比 | 针对 2~3 个核心瓶颈给出方案 A vs 方案 B 的多维度权衡与选型决断 | 宣称某种方案是“唯一完美方案”；避而不谈架构方案的代价与局限 |
| **Step 6: End-to-End Life of a Request** | **40 ~ 45 min**：核心读写时序链路闭环追踪 | 写入与读取时序图、单点故障自愈路径、边界降级兜底 | 流程断裂，组件之间只连线但无法解释具体的网络交互与数据流向 |

为完整展示这一方法论的落地实践，以下以工业界最经典的高并发无状态微服务体系——**高并发分布式短链生成与重定向系统（URL Shortener & Analytics System）**为案例，完整走完每一个标准步骤。

---

## 步骤一：需求分析与系统边界 (Step 1: Functional Requirements & Scope)

系统设计的第一步必须与业务方（或面试官）快速对齐系统要解决的核心问题，明确划定边界，防止架构在后续推演中发散失控。

### 1. 核心功能需求 (Functional Requirements - In Scope)
1. **短链生成 (URL Shortening)**：系统接收用户输入的原始长 URL，生成一个全球唯一的 7 位短编码（如 `https://short.link/a8K9zQ1`），支持用户设置可选的过期时间（TTL）；
2. **极速重定向 (Fast Redirection)**：客户端请求短链接时，系统以极低延迟解析出原始长 URL，并返回 HTTP 重定向响应跳转至目标地址；
3. **点击量与多维数据分析 (Click Analytics Ingestion)**：精准统计每次短链重定向的点击事件（访问时间戳、客户端 IP、地理位置国家/城市、User-Agent、Referer），提供聚合分析能力。

### 2. 明确排除的非核心范围 (Out of Scope)
- 自定义个性化别名修改与防抢注仲裁；
- 链接内容的实时机器学习安全沙箱反爬与反恶意钓鱼过滤；
- 复杂的付费企业级可视化仪表盘。

---

## 步骤二：非功能需求与 SLA 边界 (Step 2: Non-Functional Requirements & SLAs)

非功能需求直接决定了底层架构的选型方向（如选择强一致还是最终一致，单机还是分布式分片）：

1. **超高可用性 (High Availability)**：
   - 核心读重定向服务可用性目标为 **$99.99\%$（四个九）**，年化计划外停机时间不得超过 $52.6\text{ 分钟}$；
   - 读服务必须具备强容灾弹性：在底层数据库甚至出现降级或短暂停机时，核心重定向能力依靠前置分布式缓存维持运转。
2. **极低访问延迟 (Ultra-Low Latency)**：
   - 读重定向链路：$\text{P99 延迟} < 15\text{ ms}$（重定向直接影响用户跳转体验，需极致优化）；
   - 写短链生成链路：$\text{P99 延迟} < 100\text{ ms}$。
3. **高读写比与无状态弹性扩展 (Scale-Out Flexibility)**：
   - 流量模型呈现典型的 **$100:1$ 读多写少** 特征；
   - 计算层必须严格遵循**无状态架构（Stateless）**，实例可在秒级水平扩缩容以承接突发营销洪峰。
4. **数据持久性与唯一性不变量 (Durability & Invariants)**：
   - 已生成的短链映射绝对不可丢失，相同短码绝不允许并发冲突覆盖不同长链接。

---

## 步骤三：规模量化与容量估算 (Step 3: QPS Estimation & Capacity Sizing)

量化估算（Back-of-the-Envelope Estimation）是系统设计的物理锚点。脱离数字的架构讨论无法验证选型的合理性。

### 1. 流量 QPS 精算
- **写入 QPS (Write Traffic)**：
  - 假设系统平均每天生成 $1000\text{ 万} (10^7)$ 条新短链；
  - 1 天按工程近似换算约为 $10^5\text{ 秒} (86{,}400\text{ s})$：
  $$\text{平均写入 QPS} = \frac{10^7\text{ 次}}{10^5\text{ 秒}} = 100\text{ writes/s}$$
  - 设定 $2\times$ 突发峰值系数：
  $$\text{峰值写入 QPS} = 100 \times 2 = 200\text{ writes/s}$$

- **读取 QPS (Read Traffic / Redirection)**：
  - 读写比按 $100:1$ 计算：
  $$\text{平均读取 QPS} = 100\text{ writes/s} \times 100 = 10{,}000\text{ reads/s}$$
  - 设定 $3\times$ 突发峰值系数（应对热点爆款链接）：
  $$\text{峰值读取 QPS} = 10{,}000 \times 3 = 30{,}000\text{ reads/s}$$

### 2. 存储容量精算 (5 年数据持久化)
- **单条记录物理数据体积拆解**：
  - `id`: $8\text{ 字节}$ (BIGINT)
  - `short_code`: $7\text{ 字节}$ (VARCHAR(8))
  - `original_url`: 平均 $512\text{ 字节}$ (VARCHAR(2048))
  - `user_id`: $8\text{ 字节}$ (BIGINT)
  - `created_at`: $8\text{ 字节}$ (TIMESTAMP)
  - `expires_at`: $8\text{ 字节}$ (TIMESTAMP)
  - B+ 树索引与元数据开销预留：$\approx 60\text{ 字节}$
  - **单条记录总计**：$\approx 611\text{ 字节} \approx 0.6\text{ KB}$

- **5 年累计总存储容量推导**：
  - 5 年总生成短链数：
    $$10^7\text{ 条/天} \times 365 \times 5 = 1.825 \times 10^{10}\text{ 条} (182.5\text{ 亿条})$$
  - 5 年累计数据库持久化物理空间：
    $$\text{总存储量} = 1.825 \times 10^{10} \times 0.6\text{ KB} \approx 1.095 \times 10^{10}\text{ KB} \approx 10.95\text{ TB}$$

> **架构结论**：
> 关系型单机单表通常以 $2000\text{ 万行}$ 为 B+ 树性能稳定阈值（3 层索引），$182.5\text{ 亿条}$ 记录远超单表甚至单库物理极限。
> **因此存储层必须引入基于 `short_code` 哈希取模的分库分表（Sharded Relational DB）或原生分布式数据库（TiDB / CockroachDB）**。

### 3. 内存缓存容量精算 (Redis Cache Memory)
- 遵循典型的 **Pareto 80/20 法则**：$20\%$ 的热门短链接贡献了 $80\%$ 的重定向流量；
- 每日重定向访问中涉及的独立热点短链数按每日新增量的 $20\%$ 结合存量估算，约为 $200\text{ 万} (2\times 10^6)$ 条热点短链；
- 缓存单条键值对大小（Key: `short_code` 7B，Value: `original_url` 512B，加上 Redis `dictEntry` 与 `redisObject` 元数据开销 $\approx 600\text{ 字节}$）；
- **单日热点数据内存容量**：
  $$\text{单日热点 Cache} = 2 \times 10^6 \times 600\text{ 字节} \approx 1.2\text{ GB}$$
- 缓存过去 7 天内的高频访问短链并配置 LRU 驱逐策略，所需内存总量为：
  $$1.2\text{ GB} \times 7 \approx 8.4\text{ GB}$$

> **架构结论**：
> $8.4\text{ GB}$ 属于非常紧凑的内存体积，单台 16GB 规格的 Redis 实例即可装下全部热点映射。生产环境采用主从双机 + 哨兵集群（或 Redis Cluster）部署，核心价值在于**抗住 30,000 QPS 的并发读取并保护持久数据库**。

### 4. 网络吞吐与带宽估算 (Network Bandwidth)
- **读带宽（峰值）**：$30{,}000\text{ reads/s} \times 512\text{ 字节} \approx 15.36\text{ MB/s} \approx 123\text{ Mbps}$；
- **写带宽（峰值）**：$200\text{ writes/s} \times 600\text{ 字节} \approx 120\text{ KB/s} \approx 0.96\text{ Mbps}$。
网络带宽开销在现代千兆/万兆数据中心内完全可控，不会成为系统瓶颈。

---

## 步骤四：高层架构拓扑与数据模型 (Step 4: High-Level Architecture & Schema)

### 1. 系统分层架构拓扑
系统严格贯彻“**计算层严格无状态、状态向外分层沉淀**”的架构范式：

```text
[ Clients / Browsers ]
       │
       ▼
[ Anycast DNS / CDN (边缘节点网络加速) ]
       │
       ▼
[ API Gateway / Nginx 反向代理集群 ]
  (负责 TLS 终止、全局速率限制 Rate Limiting、WAF 安全防护)
       │
       ├────────────────────────────────────────┐
       │ (写请求: POST /urls)                   │ (读请求: GET /{short_code})
       ▼                                        ▼
[ Stateless URL-Writer Pods ]            [ Stateless Redirect Pods ]
  - 纯计算无状态                             - 纯计算无状态，毫秒级响应
  - 内存号段预分配 Base62                     - 前置 Bloom Filter 本地校验
       │                                        │
       ├─ (1. 事务落盘)                          ├─ (1. 读热点缓存 Hit: 立即 302)
       │                                        ▼
       ▼                                 [ Redis Cache Cluster ]
[ Sharded Relational Database ]                 │ (Miss 回源从库)
  (MySQL / TiDB 分布式分片)                      ▼
  - 主库承载强一致写入                  [ DB Read Replicas (只读副本) ]
  - 根据 short_code 哈希分片                    │
       │                                        │ (2. 异步发射点击埋点事件)
       │                                        ▼
       │                                 [ Kafka / Event Stream ]
       │                                        │
       ▼                                        ▼
[ Elastic Worker Cluster ]               [ Stream Analytics Workers ]
  (异步清理过期短链)                       (微批聚合点击量，写入 ClickHouse)
```

### 2. 核心数据库表结构设计 (Schema DDL)

#### (1) 短链映射主表 (`urls`)
```sql
CREATE TABLE urls (
    id BIGINT NOT NULL PRIMARY KEY,            -- 全局唯一自增 64 位整数 ID (发号器生成)
    short_code VARCHAR(8) NOT NULL,            -- Base62 编码后的 7 位短码
    original_url VARCHAR(2048) NOT NULL,       -- 原始长链接
    user_id BIGINT DEFAULT NULL,               -- 创建者用户 ID (租户配额与权限)
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT NULL,         -- 可选过期时间戳 (NULL 表示永不过期)
    UNIQUE KEY uk_short_code (short_code),     -- 唯一索引保证短码绝对不冲突
    INDEX idx_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### (2) 写请求幂等防重表 (`idempotency_keys`)
```sql
CREATE TABLE idempotency_keys (
    request_id VARCHAR(64) NOT NULL PRIMARY KEY, -- 客户端提交的意图唯一 UUID
    short_code VARCHAR(8) NOT NULL,              -- 对应生成的短码
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 3. 关键 HTTP 协议决策：HTTP 301 永久重定向 vs HTTP 302 临时重定向
这是短链系统设计中最关键的协议级抉择：

| 维度 | HTTP 301 Moved Permanently | HTTP 302 Found / 307 Temporary Redirect |
|---|---|---|
| **浏览器行为** | 浏览器在本地强持久缓存映射，后续再次点击**不再请求服务端，直接在本地跳转** | 浏览器视为临时重定向，不会强缓存；**每次点击均必须穿透请求服务端** |
| **服务器负载** | 极低（后续请求完全被浏览器本地缓存消化） | 需承受全部 30,000 QPS 重定向流量（由 Redis 消化） |
| **数据分析能力** | **完全丧失**（无法收集后续点击的 PV/UV、时间、IP、Referer） | **绝对精准**（每一次真实重定向均能触发完整埋点日志） |
| **配置变更灵活性**| 极差（短链修改或提前下线无法通知客户端浏览器） | 极佳（服务端随时更新目标长链接或下线短链） |
| **架构选型结论** | ❌ 仅适用于纯静态且无需统计的跳转 | **✅ 唯一正确选型**：数据分析是商业化核心，必须选 302 |

---

## 步骤五：三大核心架构深度权衡 (Step 5: Deep Dives × 3)

### Deep Dive 1：全局无状态发号与短码编码算法 (ID Generation Strategy)

7 位 Base62 编码（由字符集 `[0-9a-zA-Z]` 共 62 个字符构成，理论总容量 $62^7 \approx 3.52\text{ 万亿}$，远超系统 5 年 180 亿的需求）。无状态计算节点如何高效、零冲突地生成这 7 位短码？

```text
方案 A: 长 URL Hash 截断 ──► MurmurHash64 ──► 取前 43 位 Base62 ──► 冲突检查 ──► (若碰撞追加 Salt 重算)
                                                                       │
                                              [随着数据量增长，生日悖论导致碰撞率剧增]

方案 B: 预分配号段发号器 ──► 集中式发号器分配 Segment [10001, 20000] ──► 内存 AtomicLong 自增 ──► 绝对无冲突 Base62
                                                                       │
                                              [Pod 宕机跳号无损，纯内存发号单机百万 QPS]
```

- **方案 A：基于长 URL 哈希截断 + 冲突检测 (Hash Truncation with Conflict Resolution)**
  - *机制*：计算长 URL 的 MD5 或 MurmurHash64 值，截取前 43 位二进制并转为 7 位 Base62。写入数据库时若捕获唯一索引冲突，在原 URL 后追加随机 Salt 重新哈希，直至成功。
  - *致命缺陷*：百亿规模下，**哈希碰撞概率遵循生日悖论（Birthday Paradox）指数级上升**。每次碰撞都会带来昂贵的回表查询与二次哈希重试，导致写入 P99 延迟出现严重的长尾毛刺。
- **方案 B（首选方案）：分布式预分配发号器 + 确定性 Base62 进制转换 (Ticket Range Server + Base62)**
  - *机制*：短码本质上是由一个全局唯一的 64 位单调递增整数 ID 双向确定的数学映射（如 $\text{ID} = 100{,}000{,}000 \iff \text{"a8K9zQ1"}$，**冲突概率绝对为零**）。
  - *无状态架构实现*：
    1. 部署轻量级集中式发号器（如专用自增 MySQL 表或 etcd 原子递增）；
    2. 无状态 API Pod 在启动时申请一段**号段区间（Ticket Segment）**，例如 Pod A 申请到 $[1000001, 1020000]$（2 万个 ID），发号器原子推高全局游标；
    3. Pod A 在本地内存中基于 `AtomicLong` 高速自增，发号吞吐达单机数百万 QPS，**零网络 IO、零线程锁争用**；
    4. 当前号段消耗至 $80\%$ 水位时，后台异步协程向中心发号器预加载下一个号段。
  - *权衡评估*：若 Pod 发生崩溃，内存中未发完的号段会作废跳号。但在 $3.52\text{ 万亿}$ 的巨大空间面前，跳号损失完全可以忽略；换取来的是**所有无状态节点完全解除网络写锁强耦合**。

---

### Deep Dive 2：高并发读缓存、缓存穿透/击穿/雪崩综合治理 (Cache Resilience)

面对 30,000 QPS 的重定向洪峰，如果直接使用朴素 Cache-Aside 架构，系统极易在极端场景下被击垮：

```text
[ 请求进入 ] ──► 1. 本地布隆过滤器 (Bloom Filter) ──► 不存在 ──► [ 直接 404，零穿透开销 ]
                         │ 判定可能存在
                         ▼
                  2. 查询 Redis 集群
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
    [ Cache Hit ]                  [ Cache Miss ]
   (立即返回 302)                         │
                                   3. 争抢分布式互斥锁 (Mutex Lock)
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                  [ 获得锁: 唯次回源 DB ]         [ 未获得锁: 自旋 20ms 重试 ]
                         │
                  4. 回填 Redis (设置随机 Jitter 抖动 TTL)
```

1. **缓存穿透防护 (Cache Penetration Defense)**：
   - 恶意爬虫批量扫描不存在的随机短码（如 `short.link/invalidXXX`），Redis 全部 Miss，流量直接打穿到只读数据库从库；
   - **治理措施**：在无状态 API 内存或 Redis 前置**布隆过滤器（Bloom Filter）**。写短链时同步将短码哈希位标记为 1。重定向请求到达时，若布隆过滤器返回不存在，则**百分之百不存在，直接在网关层返回 404**，杜绝穿透。对偶发误判回源为空的结果，在 Redis 中回填空值哨兵（Null Object）并设置 30 秒短 TTL。
2. **缓存击穿防护 (Cache Breakdown Defense)**：
   - 超级爆款短链（如全网推送消息）的缓存刚好到期失效，成千上万并发重定向请求瞬间全部 Miss，同时发起数据库回源；
   - **治理措施**：引入**分布式互斥锁（Mutex Rebuild）**。仅允许第一个抢到分布式锁（`SET key token NX EX 5`）的请求执行从库查询与缓存回填，其余并发请求自旋等待 20ms 后重新读取 Redis。
3. **缓存雪崩防护 (Cache Avalanche Defense)**：
   - 某一时段批量创建的短链采用固定过期时间（如统一 7 天），导致同一时间点海量键批量失效；
   - **治理措施**：在基础 TTL 上注入 $\pm 10\%$ 的**随机抖动偏差（Jitter）**，使键的失效时间均匀打散在时间轴上。

---

### Deep Dive 3：点击分析统计的异步流式解耦与微批聚合 (Analytics Pipeline)

重定向请求必须记录访问时间、IP、Referer 等详细指标，如何避免复杂的统计逻辑拖垮重定向主链路？

- **方案 A：读请求同步落盘写计数 (Synchronous DB Updates)**
  - *致命缺陷*：每次 302 重定向时同步执行 `UPDATE urls SET clicks = clicks + 1 WHERE short_code = ?`。这使得只读流量瞬间转化为了数据库的**高争用行级排他锁写流量**。30,000 QPS 的并发写锁争用会在数秒内耗尽连接池，导致整个服务雪崩瘫痪。
- **方案 B（首选方案）：事件流解耦 + 滑动窗口微批聚合 (Event Streaming + Micro-batching)**
  - *机制*：
    1. 无状态 Redirect Pod 在向客户端返回 `HTTP 302` 之后，通过非阻塞异步协程向 Kafka 集群的 `click_events` 主题发射一条精简事件：
       `{"code": "a8K9zQ1", "ts": 1789258800, "ip": "1.2.3.4", "ua": "Mobile Safari", "ref": "twitter"}`；
    2. 主链路延迟损耗 $< 1\text{ ms}$；
    3. 后台独立部署分析计算 Worker 集群（如 Flink 或 Go Worker），订阅该主题；
    4. Worker 在内存中开辟 10 秒滑动时间窗口，对同类短码的 PV 点击数做内存累加；
    5. 每 10 秒将微批聚合结果批量写入高性能分析型列式数据库（ClickHouse），并向 Redis 原子更新粗粒度总计数值（用于前端页面展示）。
  - *权衡评估*：点击数据呈现最终一致性（延后数秒可见），但为主业务链路换来了绝对的健壮性与削峰填谷能力。即使下游数据仓库宕机，Kafka 的日志堆积也可以支撑数天，无状态重定向核心服务依然纹丝不动。

---

## 步骤六：端到端请求生命周期追踪 (Step 6: End-to-End Life of a Request)

通过追踪具体请求的时序流转，验证所有组件在运行时如何协同闭环：

### 1. 写入时序流 (Create Short URL)
```text
1. Client                  -> 发送 HTTP POST /api/v1/urls
                              Header: X-Idempotency-Key: "uuid-123"
                              Body: {"url": "https://example.com/very/long/path"}
2. API Gateway             -> 鉴权校验、检查全局令牌桶限流；
                              将请求均匀路由至健康的无状态 URL-Writer Pod A。
3. URL-Writer Pod A        -> 查询数据库 idempotency_keys 表；
                              若已存在，直接返回已绑定的 short_code（幂等命中）。
4. URL-Writer Pod A        -> 从本进程本地预分配号段中原子获取下一个自增整数 ID（如 1000042）；
                              执行纯内存 Base62 转换，生成短码 "a8K9zQ1"。
5. URL-Writer Pod A        -> 开启本地分库事务：
                              ├─ INSERT INTO urls (id, short_code, original_url, ...) VALUES (...);
                              └─ INSERT INTO idempotency_keys (request_id, short_code) VALUES (...);
                              提交事务，权威数据持久化完成。
6. URL-Writer Pod A        -> 异步向 Redis Cache Cluster 预热写入 "a8K9zQ1" -> "https://example.com/..."；
                              将 "a8K9zQ1" 标记入布隆过滤器。
7. URL-Writer Pod A        -> 向客户端返回 HTTP 201 Created，Body: {"short_url": "https://short.link/a8K9zQ1"}。
```

### 2. 读取重定向与分析流 (Redirect & Analytics Ingestion)
```text
1. User Mobile Browser     -> 发起 HTTP GET /a8K9zQ1
2. DNS / CDN               -> Anycast 智能解析至物理距离最近的数据中心边缘节点。
3. API Gateway             -> 负载均衡，随机轮询转发给任意无状态 Redirect Pod B。
4. Redirect Pod B          -> 检查本地布隆过滤器位图：若返回不存在，直接响应 404（无穿透开销）。
5. Redirect Pod B          -> 读取 Redis 集群: GET "a8K9zQ1"
                              ├─ [Cache Hit]: 提取原始长 URL（耗时约 1-2 ms）；
                              └─ [Cache Miss]: 竞争分布式锁，单一请求回源读 Sharded DB 从库并回填 Redis。
6. Redirect Pod B          -> 立即向客户端返回 HTTP 302 Found，
                              Header: Location: "https://example.com/very/long/path"
                              客户端浏览器瞬间自动跳转至长链接落地页。
7. Redirect Pod B          -> 异步并发向 Kafka "click_events" 主题发送埋点结构体。
8. Analytics Worker        -> 后台独立消费 Kafka 消息，滑动窗口微批聚合统计指标，批量写入 ClickHouse。
```

---

## 步骤七：系统设计黄金法则与答题避坑清单 (Step 7: Golden Rules & Anti-Patterns)

### 架构师必须坚持的三大黄金原则
1. **先算后画，数字驱动选型 (Numbers-Driven Architecture)**：
   严禁抛开数据量与 QPS 空谈架构。永远用 QPS 决定是否引入缓存，用 5 年存储量决定是否分库分表，用读写比决定是否读写分离。
2. **主动抛出 Trade-offs，不存在绝对银弹 (Always State Trade-offs)**：
   优秀的架构设计体现为在矛盾中权衡。明确说明“选择了号段发号换取了零网络锁，代价是宕机跳号”、“选择了 302 换取了精准埋点，代价是服务端必须承担全量读 QPS”。
3. **计算严格无状态，状态外置分层管理 (Decouple Compute from State)**：
   计算节点应具备随时被 `kill -9` 的容错性，任何业务持久事实必须交给专用存储基础设施。

### 面试中务必规避的四大致命反模式
- ❌ **反模式 1：开局直接画微服务组件大杂烩**（未对齐需求与规模便列出网关、K8s、Kafka、ES，属于自嗨式设计）；
- ❌ **反模式 2：使用数据库单机单表自增 ID**（在百亿级海量数据下单机自增 ID 会迅速触顶并成为单点写瓶颈）；
- ❌ **反模式 3：忽略网络超时的幂等防重**（认为网络请求一定会成功，缺少 Idempotency Key 导致重试时数据重复创建）；
- ❌ **反模式 4：选错 HTTP 状态码选用 301**（导致浏览器本地强缓存，彻底丧失后续点击数据分析能力）。

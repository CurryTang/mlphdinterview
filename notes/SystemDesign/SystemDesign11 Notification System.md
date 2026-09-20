# System Design 11 · 移动推送与通知系统 (Notification System)

课程位置：[[SystemDesign10 Flash Sale|10 秒杀]] → 本篇 → [[SystemDesign12 Crossword Solver|12 填字游戏求解器]]

接收请求、扇出收件人、经 APNs / FCM 投递至终端，并实现端内收件箱同步。

![[assets/notification-system-whiteboard.png|Notification System 架构白板]]

---

## 1. 功能需求 · Functional Requirements

1. **发送手机推送 (Push Delivery)：** 业务系统按 `user_id` 向用户的 iOS / Android 设备发送通知，支持单用户发送、批量分发（Batch Fanout）与全员广播，支持通知类别、优先级与物理有效期（TTL）。
2. **管理设备与偏好 (Device & Preference Management)：** 移动 App 启动及鉴权后注册/更新系统推送地址（Device Token）；用户配置全局免打扰时段（DND）、各业务分类订阅开关（订单、社交、营销等）及生效设备；用户退出登录后解除设备与账号的绑定关联。
3. **查询通知与状态 (Inbox & Delivery Telemetry)：** 客户端支持分页拉取保留期内的应用内通知收件箱（Notification Inbox）；业务方可查询请求提交、通道接收、失败等全链路流水状态；客户端在前台收到或用户点击展示时异步上报 `received` / `opened` 埋点。

**范围边界与技术假设 (Scope & Assumptions)：**
- 本系统定位为**应用侧通知中心平台**（Application-side Notification Platform）。
- iOS 设备通过 Apple Push Notification service (APNs) 投递，Android 设备通过 Firebase Cloud Messaging (FCM) 投递；移动端与系统级推送网关的长连接保持由操作系统底层及云厂商服务托管。
- 架构核心聚焦于高并发接入、收件人解包扇出、可靠排队削峰、通道协议适配、速率限制与失效回执闭环；MVP 阶段不自建移动操作系统长连通道，暂不覆盖 SMS 短信与 Email 邮件网关。图中的 Bus 概念由多级 Durable Queues 承载，保证削峰与分发解耦。

---

## 2. 非功能需求 → 对应设计

| 非功能需求 | 目标指标 | 对应架构 / 策略设计 |
| :--- | :--- | :--- |
| **延迟 (Latency)** | 高优先级通知（验证码、异地登录、交易安全警报）P95 < 2s 送达第三方 Provider；低优先级/营销广播允许错峰平摊 | 采用**物理优先级队列隔离**：High-Priority Queue 绑定低延迟独立 Worker 池；Bulk Queue 走平滑流控 Worker；批量大广播在 Dispatcher 切片异步入队，严禁堵塞核心通道。 |
| **可用性 (Availability)** | 核心 Ingestion API 达成 99.99%；APNs / FCM 发生网络抖动或服务故障时，业务方写入无阻塞、无感知 | 接收端采用**无状态水平扩展 + Transactional Outbox** 模式；API 仅完成鉴权、参数校验、幂等判定与本地事务写库即返回；外部通信全异步化，彻底解耦。 |
| **一致性与可靠性 (Reliability & Dedup)** | 消息不丢失（Zero Drop）；避免因系统重试造成用户端弹窗轰炸（At-least-once 尝试 + 端到端幂等去重） | 全链路唯一 `notification_id` 与请求幂等键；投递任务基于带租期（Lease）的认领机制；发送前结合 Redis/DB 校验 `(notification_id, device_id)` 幂等锁；APNs 使用 `apns-id` 保证通道层去重。 |
| **规模与削峰 (Scale & Throughput)** | 支撑 100M DAU、单日 10 亿次推送；应对突发热点广播（如大促、突发新闻）导致的瞬时 5~10 倍洪峰 | Dispatcher 执行分页切片（如 1,000 users/batch）；Durable Queues（如 Kafka / RabbitMQ）充当弹性蓄水池；下游针对 APNs / FCM 实施基于令牌桶（Token Bucket）的速率限制与连接复用。 |
| **成本与存储策略 (Cost & Lifecycle)** | 设备 Token 频繁变动、用户收件箱不可无限膨胀 | 存储冷热分层：Device Token 与用户设置采用低延迟主库与 Redis 缓存；通知内容与投递流水设定 TTL（如 30~90 天），过期自动清理或归档至对象存储冷存。 |

---

## 3. QPS 与容量估算 · QPS Estimation

### 3.1 业务体量假设

- **活跃用户规模：** 100M MAU / DAU 约 100M（高黏性移动平台）。
- **人均通知配额：** 平均 10 条 / 天（含系统通知、交易动态、社交互动与运营推荐）。
- **每日通知总请求量：**
  $$100\text{M Users} \times 10\text{ notifs/day} = 10^9\text{ notifs/day (10 亿条/天)}$$
- **峰值放大系数 (Peak Factor)：** 日常高峰按平均吞吐的 5 倍估算；突发重大新闻或运营广播采用队列平滑削峰。
- **设备放大比 (Device Fanout Ratio)：** 平均每位用户注册 1.2 台有效设备（手机、平板双持场景）。
- **重试抖动放大比 (Retry Amplification Ratio)：** 考虑网络偶发瞬断、Provider 429 限流与退避，平均单次设备投递预留 20% 额外重试预算。

### 3.2 吞吐量计算

| 维度 | 计算公式 | 平均吞吐 | 峰值吞吐 (5x) |
| :--- | :--- | :--- | :--- |
| **逻辑通知接收 (Logical Notifications)** | $\frac{10^9\text{ req}}{86,400\text{ s}}$ | **11,574 req/s** ($\approx 11.6\text{K}$) | **57,870 req/s** ($\approx 57.9\text{K}$) |
| **物理设备投递量 (Device Deliveries, 1.2x)** | $11,574 \times 1.2$ | **13,889 delivers/s** ($\approx 13.9\text{K}$) | **69,444 delivers/s** ($\approx 69.4\text{K}$) |
| **Provider 请求尝试 (Attempts, 1.2x 投递 × 1.2 重试)** | $13,889 \times 1.2$ | **16,667 attempts/s** ($\approx 16.7\text{K}$) | **83,333 attempts/s** ($\approx 83.3\text{K}$) |
| **Inbox 查询与回执上报 (Read & Receipts)** | 假设 30% 通知被点击打开，叠加冷启拉取 | **$\approx 3.5\text{K}$ QPS** | **$\approx 17.5\text{K}$ QPS** |

### 3.3 存储与网络带宽估算

- **通知内容载荷 (Payload)：**
  - 单条通知包含标题、文本正文、路由参数（deep-link）与业务扩展元数据，平均按 $1\text{ KB}$ 计。
  - 每日新增逻辑内容：$10^9 \times 1\text{ KB} = 1\text{ TB / day}$。
  - 保留 90 天有效期存储容量：$1\text{ TB/day} \times 90\text{ days} = 90\text{ TB}$。
- **投递流水元数据 (Delivery Metadata)：**
  - 包含 `delivery_id`, `notification_id`, `device_id`, `attempt_count`, `status`, `provider_msg_id`, `updated_at` 等，单行约 $200\text{ Bytes}$。
  - 每日流水记录数：$1.2 \times 10^9$ 行。
  - 每日流水存储增量：$1.2 \times 10^9 \times 200\text{ B} \approx 240\text{ GB / day}$。
  - 30 天保留周期容量：$240\text{ GB/day} \times 30\text{ days} = 7.2\text{ TB}$。
- **设备注册表 (Device Registry Table)：**
  - $100\text{M 用户} \times 1.2\text{ 设备} = 120\text{M}$ 条记录。
  - 单条记录包含 `device_id`, `user_id`, `push_token`, `platform`, `token_status`, `updated_at` 等，平均约 $300\text{ Bytes}$。
  - 总静态容量：$120\text{M} \times 300\text{ B} \approx 36\text{ GB}$。可常驻高效分布式数据库主存并由 Redis 进行高速索引缓存。
- **出口网络带宽 (Egress Bandwidth to APNs / FCM)：**
  - 峰值出口流量：$83,333\text{ attempts/s} \times 1\text{ KB} \approx 83.3\text{ MB/s}$。
  - 折合公网传输带宽：$83.3 \times 8 \approx 666.7\text{ Mbps}$，现代机房千兆/万兆出口可从容承载。

### 3.4 连接池与并发度 (Worker Sizing & Little's Law)

- **Provider 平均响应延迟：** 针对 APNs (HTTP/2 多路复用) 与 FCM (HTTP v1 / gRPC)，平均往返时间（RTT）按 $100\text{ ms} = 0.1\text{ s}$ 估算。
- **峰值在途请求数 (In-flight Concurrency)：**
  由利特尔法则（Little's Law $L = \lambda \times W$）：
  $$L = 83,333\text{ attempts/s} \times 0.1\text{ s} \approx 8,334\text{ concurrent requests}$$
- **连接池规划：**
  - APNs 官方推荐使用 HTTP/2 多路复用，单个长连接并发保持多个流（如单连接 100~500 并发 Streams）。
  - 维持 8,334 个峰值在途请求理论上仅需：
    $$\frac{8,334}{100} \approx 84\text{ 条 HTTP/2 长连接}$$
  - 实际生产中部署 20~40 台无状态 Provider Worker 节点，每台 Worker 维持 2~4 条与 Apple / Google 网关的持久连接池，即可在极低连接维护开销下承载峰值流量。

---

## 4. 架构设计 · High-Level Design

### 4.1 初版设计 (Vanilla Direct Loop) 及其致命缺陷

```text
[Upstream Business Service]
            │
            ▼ (同步遍历 user_id 列表)
      [Device DB] ──▶ 查出所有 device_tokens 
            │
            ▼ (在主线程循环向 APNs / FCM 发送 HTTP POST)
      [APNs / FCM] ──▶ 阻塞等待返回 ──▶ 直接向调用方返回结果
```

初版设计在业务量小时极易实现，但在高并发工业级场景下存在四个致命缺陷：
1. **级联雪崩与线程池耗尽：** 上游业务系统同步等待第三方外部网络响应。一旦 APNs / FCM 出现 300~500ms 网络抖动或接口限流（HTTP 429），上游核心服务的 Web/RPC 线程瞬间占满，导致订单、支付等核心链路被连带拖垮。
2. **大广播引发优先级反转：** 缺少排队调度隔离。当运营系统发送一条面向海量全量用户的活动推送时，单线程循环处理需耗时数小时，后续触发的用户验证码、支付成交通知被完全堵死在后方。
3. **缺乏幂等保护与重试风暴：** 遇网络超时直接粗暴重试，导致同一用户连续收到多条一模一样的骚扰弹窗；而服务崩溃重启时，正在内存中处理的任务直接丢信。
4. **失效 Token 无闭环清理机制：** 设备卸载 App 后，Provider 会返回 `BadDeviceToken` 或 `Unregistered`。初版无异步反馈闭环，持续向无效设备发送无用请求，浪费宝贵的配额与出口带宽。

---

### 4.2 最终生产级分层架构

```text
[Upstream Business Services] (Order / Social / Marketing)
           │
           │ 1. POST /v1/notifications (Idempotency-Key)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Ingestion & Storage Layer                                │
│   [API Gateway] ──(Auth & Rate Limit)                       │
│          │                                                  │
│          ▼                                                  │
│   [Notification Ingestion Service]                          │
│          │                                                  │
│          ├───────────────┬──────────────────┐               │
│          ▼               ▼                  ▼               │
│   [Notification DB] [Outbox Table]    [Inbox Store]         │
│   (Logical metadata) (Unpublished)   (User Pull Cache)      │
└──────────┬──────────────────────────────────────────────────┘
           │ CDC / Poller (Zero-loss publishing)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Fanout & Scheduling Layer                                │
│   [Dispatcher / Fanout Workers]                             │
│          │                                                  │
│          ├── 查 [User Preference Store] (免打扰 / 分类开关) │
│          ├── 查 [Device Registry Store] (有效 Token 列表)   │
│          └── 拆分 Batch (切片为 500~1000 targets/task)      │
└──────────┬──────────────────────────────────────────────────┘
           │ 投递至物理隔离队列
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Durable Messaging Layer (Partitioned by Priority & Chan) │
│   [High-Priority Queue]  ── 验证码、安全警报、即时通讯      │
│   [Bulk/Marketing Queue] ── 运营推广、每日摘要              │
│   [Delayed/Quiet Queue]  ── 免打扰暂存、延迟待发            │
└──────────┬──────────────────────────────────────────────────┘
           │ 并发拉取 (Claim & Lease)
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Delivery & Provider Adapter Layer                        │
│   [Provider Workers]                                        │
│          │                                                  │
│          ├── [Local Deduplication Cache] (Redis 5-min TTL)  │
│          ├── [Token-Bucket Rate Limiter] (per Provider)     │
│          ▼                                                  │
│   [APNs Adapter]          [FCM Adapter]                     │
│   (HTTP/2 keep-alive)     (HTTP v1 / gRPC keep-alive)       │
└──────────┬──────────────────────┬───────────────────────────┘
           │                      │
           ▼                      ▼
      [Apple APNs]          [Google FCM]
           │                      │
           ▼ (OS Managed)         ▼ (OS Managed)
      [iOS Device]          [Android Device]
           │                      │
           └──────────┬───────────┘
                      │ Ack / Open 埋点 (客户端通过 Gateway 异步打点)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Feedback & Cleanup Closed-Loop                           │
│   [Receipt & Cleanup Worker]                                │
│          │                                                  │
│          ├── Provider 返回 410 / BadDeviceToken ──────────┐ │
│          │                                                ▼ │
│          └── 写入 Delivery Attempts 历史流水  ──▶ [Clean Device]
└─────────────────────────────────────────────────────────────┘
```

**分层职责剖析：**
1. **接入与存储层 (Ingestion & Storage)：** API 网关拦截鉴权与限流；Ingestion Service 校验 `idempotency_key`，在单个数据库本地事务中写入 `notification_requests`、`outbox` 与 `inbox_store`，立即向业务方返回 `notification_id`，保证业务请求毫秒级响应。
2. **扇出与过滤层 (Fanout & Scheduling)：** Dispatcher 消费 Outbox 任务。若为批量任务，则拉取目标用户集并按 500~1,000 个收件人切片；读取 Redis 缓存中的用户偏好与设备注册表，过滤处于免打扰或已关闭对应分类的用户，生成设备级物理投递任务。
3. **高可用持久化排队层 (Durable Messaging Layer)：** 物理隔离高优与营销队列，杜绝优先级反转；对于处在免打扰时段但允许推迟的通知，移入延迟队列等待窗口期开启。
4. **适配与投递层 (Delivery & Provider Adapters)：** Worker 基于分布式租约（Lease）认领任务；利用 Redis 幂等锁拦截重复投递；维护与 APNs（HTTP/2 多路复用）及 FCM（HTTP v1 / gRPC）的高性能长连接池；集成令牌桶流控防止被厂商封禁。
5. **回执与失效闭环层 (Feedback & Cleanup)：** 捕获 APNs（HTTP 410 `Unregistered`）与 FCM（`UNREGISTERED`）错误，异步向设备注册表发起软删除或标记失效；收集移动端上报的打开埋点，驱动投递分析报表。

---

### 4.3 核心数据模型 · Data Model

系统采用关系型数据库保证强事务一致性，辅以 Redis 缓存高频热点。

#### 1. `notification_requests` (逻辑通知表)
```sql
CREATE TABLE notification_requests (
    notification_id   VARCHAR(64) PRIMARY KEY,       -- 全局唯一 UUID / SnowFlake
    idempotency_key   VARCHAR(128) UNIQUE NOT NULL,  -- 业务方幂等键 (如 order_12345_shipped)
    source_service    VARCHAR(32) NOT NULL,          -- 来源业务方 (order, auth, marketing)
    recipient_type    VARCHAR(16) NOT NULL,          -- 'SINGLE', 'BATCH', 'BROADCAST'
    recipient_target  TEXT NOT NULL,                 -- 单用户 user_id 或 分群 segment_id
    category          VARCHAR(32) NOT NULL,          -- 'SECURITY', 'ORDER', 'PROMO'
    priority          SMALLINT NOT NULL,             -- 1: High, 2: Normal, 3: Low
    title             VARCHAR(256) NOT NULL,
    body              TEXT NOT NULL,
    payload_json      JSONB,                         -- 业务透传扩展参数 (如 deep-link URI)
    collapse_key      VARCHAR(64),                   -- 折叠聚合键 (同一状态仅保留最新)
    status            VARCHAR(16) NOT NULL,          -- 'ACCEPTED', 'DISPATCHED', 'FAILED'
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at        TIMESTAMP WITH TIME ZONE NOT NULL -- 消息物理过期时间
);

CREATE INDEX idx_notif_created ON notification_requests(created_at);
```

#### 2. `device_registrations` (设备注册表)
```sql
CREATE TABLE device_registrations (
    device_id         VARCHAR(64) PRIMARY KEY,       -- 设备唯一标识 (UUID)
    user_id           VARCHAR(64) NOT NULL,          -- 归属用户
    platform          VARCHAR(16) NOT NULL,          -- 'IOS', 'ANDROID'
    push_token        TEXT NOT NULL,                 -- APNs deviceToken 或 FCM registration_token
    app_version       VARCHAR(32),
    os_version        VARCHAR(32),
    token_status      VARCHAR(16) NOT NULL,          -- 'VALID', 'INVALID', 'UNINSTALLED'
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- 高频查询索引：用户在线设备的有效 Token 查询
CREATE INDEX idx_device_user_valid ON device_registrations(user_id) 
WHERE token_status = 'VALID';
```

#### 3. `user_preferences` (用户偏好设置表)
```sql
CREATE TABLE user_preferences (
    user_id           VARCHAR(64) PRIMARY KEY,
    do_not_disturb    BOOLEAN NOT NULL DEFAULT FALSE,-- 全局免打扰开关
    dnd_start_time    TIME,                          -- 免打扰起始 (如 '22:00:00')
    dnd_end_time      TIME,                          -- 免打扰结束 (如 '08:00:00')
    channel_settings  JSONB NOT NULL DEFAULT '{}',   -- 分类开关: {"PROMO": false, "ORDER": true}
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

#### 4. `delivery_attempts` (物理投递记录表)
```sql
CREATE TABLE delivery_attempts (
    attempt_id        VARCHAR(64) PRIMARY KEY,       -- 投递尝试唯一标识
    notification_id   VARCHAR(64) NOT NULL,          -- 关联逻辑通知
    device_id         VARCHAR(64) NOT NULL,          -- 目标物理设备
    provider          VARCHAR(16) NOT NULL,          -- 'APNS', 'FCM'
    attempt_number    INT NOT NULL DEFAULT 1,        -- 当前重试次数
    status            VARCHAR(16) NOT NULL,          -- 'PENDING', 'SENT', 'FAILED', 'DISCARDED'
    error_code        VARCHAR(64),                   -- 错误码 (BadDeviceToken, RateLimited, etc.)
    provider_msg_id   VARCHAR(128),                  -- APNs apns-id 或 FCM message_id
    lease_until       TIMESTAMP WITH TIME ZONE,      -- 分布式 Worker 抢占租约
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_attempts_notif_device ON delivery_attempts(notification_id, device_id);
CREATE INDEX idx_attempts_lease ON delivery_attempts(status, lease_until) 
WHERE status = 'PENDING';
```

---

### 4.4 偏好一致性与投递边界定义

#### 1. 用户偏好一致性策略 (Preference Consistency)
- **缓存模型：** 用户更新免打扰或分类推送偏好时，通过 Cache-Aside 模式先写入主库并驱逐 Redis 缓存 (`preference:{user_id}`)。
- **一致性权衡：** Dispatcher 消费消息时读取缓存，允许毫秒至秒级的最终一致性（Eventual Consistency）。
- **紧急覆盖规则 (Bypass Override)：** 系统最高优先级（`priority = 1`）通知（如账户异地登录告警、两步验证安全验证码）**强制跳过免打扰校验与退订开关**，直通投递队列。

#### 2. 三级投递边界与 SLA 语义 (Delivery Boundaries & SLA)

```text
[Notification Platform]
         │
  (Boundary 1: Provider Accepted) ──▶ 系统服务等级协议 (SLA) 承诺终点
         ▼
[APNs / FCM Gateway]
         │
  (Boundary 2: Device Displayed)  ──▶ 受终端网络状态、飞行模式、系统推送限额影响
         ▼
[Mobile OS / Notification Tray]
         │
  (Boundary 3: User Opened)       ──▶ 受用户注意力、免打扰主观意愿影响
         ▼
[User Engagement]
```

- **Boundary 1: Provider Accepted (服务商接收)**
  - 定义：APNs 返回 HTTP 200 或 FCM 返回合法的 `name/projects/.../messages/...`。
  - 含义：通知已成功移交 Apple / Google 云端中转队列。
  - **平台 SLA 约束边界**：通知系统的稳定性与延迟承诺（P95 < 2s）**严格以此边界为准**。
- **Boundary 2: Device Displayed (终端展示 / 客户端收到)**
  - 定义：iOS Notification Service Extension 或 Android 客户端后台 Service 接收到数据包，并在本地触发系统托盘通知，上报 `received` 埋点。
  - 约束：若手机断网、关机、进入深度省电休眠或关闭了系统级通知权限，展示延迟不可控。
- **Boundary 3: User Opened (用户交互 / 打开)**
  - 定义：用户点击通知栏卡片启动 App 或触发 Action 按钮，向网关上报 `opened` 埋点。
  - 约束：纯业务指标（CTR），不作为基础设施投递成功的判定条件。

---

## 5. 核心深度权衡 · Deep Dive

### 5.1 分发与缓冲架构 (Dispatch Architecture)

```text
方案 A: 同步链式投递 (Synchronous Direct Dispatch)
[API Service] ──▶ 遍历收件人 ──▶ HTTP POST APNs ──▶ 阻塞等待 ──▶ 返回客户端

方案 B: 事务 Outbox + 多级持久队列 (Transactional Outbox + Durable Queues)
[API Service] ──▶ 本地事务 (DB + Outbox) ──▶ 立即返回
                       │ (CDC / Poller)
                       ▼
                 [Dispatcher] ──▶ 物理隔离队列 (High / Bulk) ──▶ [Provider Workers]
```

| 维度 | 方案 A: 同步链式投递 (Sync Direct) | 方案 B: 事务 Outbox + 多级持久队列 (Durable Queues, 推荐) |
| :--- | :--- | :--- |
| **系统吞吐量** | **受外部 RTT 严格锁死**。若 APNs 延迟 100ms，单线程仅能承载 10 QPS，难以支持海量并发。 | **极高**。本地事务写入仅耗时 2~5ms；后续全链路异步解耦，支持水平横向扩展。 |
| **故障隔离与雪崩防护** | **极差**。外部网络抖动或第三方通道宕机会迅速耗尽服务工作线程，引起全站级联雪崩。 | **完善**。外部故障仅体现为队列排队积压；上游业务请求正常落盘，无阻塞。 |
| **削峰能力** | **无**。瞬时流量脉冲直接透传给 Provider，极易触发 APNs/FCM 的 429 配额限制。 | **具备天然蓄水池效应**。上游 50K QPS 峰值入队，下游 Worker 以平稳恒定速率消费出队。 |
| **数据可靠性** | 进程崩溃或网络中断即丢失正在处理的数据，无补偿回溯机制。 | **零丢失保障**。基于数据库 WAL (Debezium CDC) 或 Outbox 表轮询，严格 At-least-once 投递。 |
| **架构选型建议** | **生产环境禁止采用**；仅适用于测试原型或单机内部自动化脚本。 | **高并发移动平台的唯一标准架构**。 |

---

### 5.2 投递语义与去重防重 (Delivery Semantics & Dedup)

```text
租约认领与幂等投递时序 (Lease Claim & Idempotent Delivery):
[Worker] ──1. Claim (UPDATE delivery_attempts SET lease_until = NOW()+30s)──▶ [DB]
    │
    ├──2. 检查 Redis 幂等锁 SETNX (notif_id + device_id, 5 min)
    │     ├── 锁已存在 ──▶ 跳过该设备 (避免并发重复发送)
    │     └── 获取成功 ──▶ 继续
    │
    ├──3. HTTP/2 POST APNs (携带 Header: apns-id = attempt_id)
    │     ├── 200 OK ──▶ 更新 status = 'SENT'
    │     └── 429 / 5xx ──▶ 退避入重试队列 (Exponential Backoff + Jitter)
```

| 维度 | 方案 A: 至多一次尝试 (At-most-once) | 方案 B: 至少一次投递 + 租约认领 + 分布式去重 (推荐) |
| :--- | :--- | :--- |
| **重复弹窗风险** | **零重复**。消息出队即 Ack；发送失败直接丢弃，绝不触发第二次推送。 | **极低**。通过三层防线（Worker 租约 + Redis 分布式防重锁 + APNs `apns-id` 渠道原生去重）拦截重复。 |
| **消息丢失率** | **高**。任何网络丢包、Provider 503 临时抖动或 Worker 重启都会导致消息静默丢失。 | **零丢失**。未收到成功 Ack 的任务在租约超时后自动被其他健康 Worker 认领重试。 |
| **重试与风暴控制** | 不存在重试机制。 | 配合**指数退避与随机抖动**（Exponential Backoff with Full Jitter）及最大重试次数（如 3 次），超限转入死信队列（DLQ）。 |
| **实现复杂度** | 极低，无需持久化尝试流水与租约状态。 | 中等，需维护尝试流水生命周期、分布式锁与死信监控告警。 |
| **架构选型建议** | 仅容忍用于纯营销类广告广播（宁缺毋滥）。 | **核心业务场景的行业通用规范**，兼顾高到达率与防骚扰体验。 |

---

### 5.3 离线通知积压与成本控制 (Offline Delivery & Cost Optimization)

```text
状态折叠机制 (Coalesce / Collapse Mechanism):
若司机位置频繁变动 (10 秒推送一次):
Request 1: collapse_key = "trip_888", loc = (lat1, lon1)  ──┐
Request 2: collapse_key = "trip_888", loc = (lat2, lon2)  ──┼──▶ [Queue / APNs] 只保留最新一条！
Request 3: collapse_key = "trip_888", loc = (lat3, lon3)  ──┘
```

| 维度 | 方案 A: 全量严格排队落盘 (Retain Every Event Indefinitely) | 方案 B: 状态折叠 (Coalesce by Key) + 有限 TTL (推荐) |
| :--- | :--- | :--- |
| **用户体验** | **恶劣**。用户手机离线数天开机后，瞬间涌入数十条历史陈旧弹窗（如过期的打车位置更新、过期的比赛比分）。 | **优异**。同实体状态仅弹窗展示最新一条；通知具备清晰时效性，消除通知刷屏打扰。 |
| **存储与带宽成本** | **线性激增**。冷数据长期挤占主库与队列磁盘；向 APNs/FCM 发送大量无意义已失效的陈旧数据。 | **显著降低**。大幅缩减队列与数据库体积；利用 APNs `apns-collapse-id` 与 FCM `collapse_key`，云端自动覆盖。 |
| **数据一致性** | 保留全量过程流水，便于完整事后审计。 | 投递层仅保留最新快照；底层事件日志可按需异步归档至低成本冷存。 |
| **业务适配度** | 适用于金融账单扣款明细等每笔都独立的审计流水。 | **最适于状态刷新类场景**（网约车轨迹、外卖配送进度、即时比分、行情波动）。 |
| **架构选型建议** | 不适用于海量高频状态更新。 | **推荐生产架构**。设置合理物理 TTL（如交易类 24h，营销类 6h），配合业务 Key 聚合。 |

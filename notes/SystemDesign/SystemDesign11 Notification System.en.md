# System Design 11 · Mobile Push & Notification System

Course location: [[SystemDesign10 Flash Sale|10 Flash Sale]] → this note → [[SystemDesign99 Glossary|99 Glossary]]

Ingest notification requests, fan out across recipient devices, dispatch reliably via APNs / FCM, and synchronize the in-app notification inbox.

![[assets/notification-system-whiteboard.png|Notification System Architecture Whiteboard]]

---

## 1. Functional Requirements

1. **Mobile Push Delivery:** Business services issue notifications to users' iOS and Android devices by `user_id`, supporting single-recipient dispatch, bulk fanout, and global broadcasts, with granular category classification, priority levels, and physical time-to-live (TTL).
2. **Device & Preference Management:** Mobile apps register and refresh system push tokens upon launch and authentication; users configure global Do-Not-Disturb (DND) windows, category-level subscription toggles (orders, social, marketing), and target devices; user logout severs device-to-account bindings.
3. **Inbox & Delivery Telemetry:** Client apps paginate and pull active notifications from an in-app notification inbox; upstream systems query full-lifecycle statuses (submission, provider receipt, failure); client apps asynchronously report `received` and `opened` telemetry events upon foreground receipt or user interaction.

**Scope & Technical Assumptions:**
- The architecture is scoped as an **Application-Side Notification Platform**.
- Delivery to iOS devices is delegated to the Apple Push Notification service (APNs), and Android devices to Firebase Cloud Messaging (FCM); persistent device socket connections are managed entirely by mobile OS kernels and cloud infrastructure.
- Core architecture focuses on high-concurrency ingestion, recipient fanout, queue leveling, provider protocol adapters, rate limiting, and invalidation feedback loops. MVP scope excludes SMS, email gateways, or self-hosted mobile socket daemons. The bus concept from the whiteboard is implemented via multi-tier durable queues.

---

## 2. Non-Functional Requirements & Architecture Mappings

| Non-Functional Requirement | Target SLA | Architectural & Algorithmic Design |
| :--- | :--- | :--- |
| **Latency** | High-priority notifications (verification codes, login alerts, transaction security) P95 < 2s to external provider; marketing broadcasts allow traffic leveling | **Physical priority queue isolation**: High-priority queues bind to dedicated, low-latency worker pools; bulk queues run rate-limited workers; large broadcast requests are sliced asynchronously by the dispatcher to prevent blocking critical paths. |
| **Availability** | Core Ingestion API reaches 99.99%; upstream business flows remain non-blocking during APNs / FCM network jitter or vendor outages | **Stateless horizontal scale + Transactional Outbox**: Ingestion API performs auth, validation, idempotency checks, and writes to DB/Outbox within a local transaction before immediate return; external vendor communications are fully asynchronous. |
| **Reliability & Dedup** | Zero message loss; eliminate duplicate push notifications during system retries (At-least-once attempts + end-to-end idempotency) | Global unique `notification_id` and client-supplied `idempotency_key`; worker delivery claims guarded by time-bounded leases; pre-send deduplication locks via Redis/DB `(notification_id, device_id)`; native APNs `apns-id` deduplication. |
| **Scale & Throughput** | Support 100M DAU and 1 billion notifications per day; handle 5x to 10x traffic spikes during flash sales and breaking news | Dispatcher paginates recipient lists into slices (500–1,000 users/batch); durable message queues (Kafka / RabbitMQ) absorb bursts; token-bucket rate limiters and HTTP/2 connection pooling govern downstream provider throughput. |
| **Cost & Lifecycle** | Rapid device token churn; unbounded inbox table bloat must be prevented | **Tiered storage**: Device tokens and user preferences reside in low-latency primary stores and Redis caches; notification bodies and attempt audit logs enforce TTLs (30–90 days), with automatic expiration or cold object storage archiving. |

---

## 3. QPS & Capacity Estimation

### 3.1 Business Scale Assumptions

- **Active User Base:** 100M MAU / DAU $pprox$ 100M (high-engagement mobile platform).
- **Per-User Notification Volume:** Average 10 notifications/user/day (system updates, transactions, social interactions, recommendations).
- **Total Daily Notification Ingestion:**
  $$100\text{M Users} \times 10\text{ notifs/day} = 10^9\text{ notifs/day (1 Billion / day)}$$
- **Peak Factor:** Peak hours estimated at 5x average throughput; breaking news broadcasts smoothed via queue buffering.
- **Device Fanout Ratio:** Average 1.2 active devices registered per user (multi-device smartphone + tablet users).
- **Retry Amplification Ratio:** Factoring in network jitter, temporary socket errors, and provider 429 throttling backoff, average delivery incurs 20% extra retry attempts.

### 3.2 Throughput Calculation

| Dimension | Formula | Average Throughput | Peak Throughput (5x) |
| :--- | :--- | :--- | :--- |
| **Logical Ingestion (Notification Requests)** | $\frac{10^9\text{ req}}{86,400\text{ s}}$ | **11,574 req/s** ($\approx 11.6\text{K}$) | **57,870 req/s** ($\approx 57.9\text{K}$) |
| **Device Deliveries (1.2x Fanout)** | $11,574 \times 1.2$ | **13,889 delivers/s** ($\approx 13.9\text{K}$) | **69,444 delivers/s** ($\approx 69.4\text{K}$) |
| **Provider Attempts (1.2x Fanout × 1.2 Retries)** | $13,889 \times 1.2$ | **16,667 attempts/s** ($\approx 16.7\text{K}$) | **83,333 attempts/s** ($\approx 83.3\text{K}$) |
| **Inbox Reads & Telemetry Receipts** | Assuming 30% open rate + cold app launch fetches | **$\approx 3.5\text{K}$ QPS** | **$\approx 17.5\text{K}$ QPS** |

### 3.3 Storage & Bandwidth Estimation

- **Notification Payload Storage:**
  - Each notification contains title, body text, deep-link URI, and JSON metadata, averaging $1\text{ KB}$.
  - Daily raw payload volume: $10^9 \times 1\text{ KB} = 1\text{ TB / day}$.
  - 90-day retention window: $1\text{ TB/day} \times 90\text{ days} = 90\text{ TB}$.
- **Delivery Attempt Metadata:**
  - Records contain `delivery_id`, `notification_id`, `device_id`, `attempt_count`, `status`, `provider_msg_id`, `updated_at`, averaging $200\text{ Bytes}$.
  - Daily attempt rows: $1.2 \times 10^9$ rows.
  - Daily log storage: $1.2 \times 10^9 \times 200\text{ B} \approx 240\text{ GB / day}$.
  - 30-day operational retention: $240\text{ GB/day} \times 30\text{ days} = 7.2\text{ TB}$.
- **Device Registry Table:**
  - $100\text{M users} \times 1.2\text{ devices} = 120\text{M}$ active rows.
  - Row size: $300\text{ Bytes}$ (`device_id`, `user_id`, `push_token`, `platform`, `token_status`, `updated_at`).
  - Total working set: $120\text{M} \times 300\text{ B} \approx 36\text{ GB}$, comfortably fitting within primary database memory and Redis indices.
- **Egress Bandwidth (Provider Outbound):**
  - Peak egress: $83,333\text{ attempts/s} \times 1\text{ KB} \approx 83.3\text{ MB/s}$.
  - Outbound bandwidth: $83.3 \times 8 \approx 666.7\text{ Mbps}$, well within standard datacenter gigabit transit links.

### 3.4 Worker Sizing & Little's Law

- **Provider Round-Trip Latency (RTT):** APNs (HTTP/2 multiplexing) and FCM (HTTP v1 / gRPC) average RTT is estimated at $100\text{ ms} = 0.1\text{ s}$.
- **Peak In-Flight Requests:**
  Applying Little's Law ($L = \lambda \times W$):
  $$L = 83,333\text{ attempts/s} \times 0.1\text{ s} \approx 8,334\text{ concurrent requests}$$
- **Connection Pool Sizing:**
  - APNs multiplexes concurrent streams over persistent HTTP/2 connections (typically 100–500 streams per socket).
  - Theoretical minimum persistent connections needed at peak:
    $$\frac{8,334}{100} \approx 84\text{ persistent HTTP/2 connections}$$
  - In production, deploying 20–40 stateless Provider Workers maintaining 2–4 connections each easily handles peak load with minimal socket overhead.

---

## 4. High-Level Architecture

### 4.1 Vanilla Direct Loop & Failure Analysis

```text
[Upstream Business Service]
            │
            ▼ (Synchronously query user_ids)
      [Device DB] ──▶ Retrieve all device_tokens 
            │
            ▼ (Synchronous loop executing HTTP POSTs in main thread)
      [APNs / FCM] ──▶ Block on external RTT ──▶ Return result to caller
```

**Fatal Production Flaws:**
1. **Cascading Failure & Thread Pool Exhaustion:** Upstream services block synchronously on third-party networks. When APNs/FCM encounters 300–500ms latency spikes or HTTP 429 throttling, upstream application threads are instantly depleted, crashing core order and checkout workflows.
2. **Broadcast Priority Inversion:** Unpartitioned processing means a single 5-million-user marketing campaign blocks thread pools for hours, starving high-priority two-factor auth codes and payment confirmations.
3. **No Idempotency & Retry Storms:** Network timeouts trigger uncoordinated retries, flooding users with duplicate pushes; unpersisted in-memory tasks are lost on worker crashes.
4. **Unclosed Invalidation Loops:** When devices uninstall apps, providers return `BadDeviceToken` or `Unregistered`. Without an asynchronous cleanup loop, the platform wastes bandwidth and quota delivering to dead endpoints indefinitely.

---

### 4.2 Production Layered Architecture

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
│          ├── Query [User Preference Store] (DND / Toggles)  │
│          ├── Query [Device Registry Store] (Active Tokens)  │
│          └── Slice Batches (500–1,000 targets / task chunk) │
└──────────┬──────────────────────────────────────────────────┘
           │ Publish to physically isolated queues
           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Durable Messaging Layer (Partitioned by Priority & Chan) │
│   [High-Priority Queue]  ── 2FA, Security, Direct Messages  │
│   [Bulk/Marketing Queue] ── Campaigns, Daily Digests        │
│   [Delayed/Quiet Queue]  ── DND Buffer, Scheduled Alerts    │
└──────────┬──────────────────────────────────────────────────┘
           │ Concurrent Consumer (Claim & Lease)
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
                      │ Telemetry Acks / Opens (via Gateway async beacon)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Feedback & Cleanup Closed-Loop                           │
│   [Receipt & Cleanup Worker]                                │
│          │                                                  │
│          ├── Provider returns 410 / BadDeviceToken ───────┐ │
│          │                                                ▼ │
│          └── Write Delivery Attempts Audit Log ─▶ [Clean Device]
└─────────────────────────────────────────────────────────────┘
```

**Subsystem Responsibilities:**
1. **Ingestion & Storage Layer:** API Gateway handles authentication and rate limiting; Ingestion Service validates `idempotency_key` and persists `notification_requests`, `outbox`, and `inbox_store` within a single local DB transaction, immediately returning `notification_id` in milliseconds.
2. **Fanout & Scheduling Layer:** Dispatcher consumes Outbox events. For bulk targets, it slices recipient lists into 500–1,000 target chunks, filters users against cached preferences (DND / category toggles) and valid device tokens, and emits device-level delivery tasks.
3. **Durable Messaging Layer:** High-priority and marketing queues are physically isolated to prevent head-of-line blocking. Messages arriving during user DND hours are moved to delayed queues until quiet hours lapse.
4. **Delivery & Provider Adapters:** Workers acquire tasks via distributed leases, verify deduplication locks in Redis, maintain persistent HTTP/2 and gRPC connection pools, and enforce token-bucket throttling.
5. **Feedback & Cleanup Loop:** Captures APNs 410 (`Unregistered`) and FCM (`UNREGISTERED`) responses, triggering asynchronous soft-deletion of dead tokens in the Device Registry; ingests client telemetry for delivery analytics.

---

### 4.3 Core Data Model

A relational store guarantees transactional integrity, paired with Redis for low-latency indexing.

#### 1. `notification_requests` (Logical Notification Table)
```sql
CREATE TABLE notification_requests (
    notification_id   VARCHAR(64) PRIMARY KEY,       -- Global UUID / SnowFlake
    idempotency_key   VARCHAR(128) UNIQUE NOT NULL,  -- Upstream idempotency token (e.g. order_123_shipped)
    source_service    VARCHAR(32) NOT NULL,          -- Calling service (order, auth, marketing)
    recipient_type    VARCHAR(16) NOT NULL,          -- 'SINGLE', 'BATCH', 'BROADCAST'
    recipient_target  TEXT NOT NULL,                 -- user_id or segment_id
    category          VARCHAR(32) NOT NULL,          -- 'SECURITY', 'ORDER', 'PROMO'
    priority          SMALLINT NOT NULL,             -- 1: High, 2: Normal, 3: Low
    title             VARCHAR(256) NOT NULL,
    body              TEXT NOT NULL,
    payload_json      JSONB,                         -- Business metadata (deep-links, routing args)
    collapse_key      VARCHAR(64),                   -- Coalescing key (replace prior stale state)
    status            VARCHAR(16) NOT NULL,          -- 'ACCEPTED', 'DISPATCHED', 'FAILED'
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    expires_at        TIMESTAMP WITH TIME ZONE NOT NULL -- Physical message TTL
);

CREATE INDEX idx_notif_created ON notification_requests(created_at);
```

#### 2. `device_registrations` (Device Registry Table)
```sql
CREATE TABLE device_registrations (
    device_id         VARCHAR(64) PRIMARY KEY,       -- Unique device UUID
    user_id           VARCHAR(64) NOT NULL,          -- Account owner
    platform          VARCHAR(16) NOT NULL,          -- 'IOS', 'ANDROID'
    push_token        TEXT NOT NULL,                 -- APNs deviceToken or FCM registration_token
    app_version       VARCHAR(32),
    os_version        VARCHAR(32),
    token_status      VARCHAR(16) NOT NULL,          -- 'VALID', 'INVALID', 'UNINSTALLED'
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Fast lookup for active tokens per user
CREATE INDEX idx_device_user_valid ON device_registrations(user_id) 
WHERE token_status = 'VALID';
```

#### 3. `user_preferences` (User Preference Table)
```sql
CREATE TABLE user_preferences (
    user_id           VARCHAR(64) PRIMARY KEY,
    do_not_disturb    BOOLEAN NOT NULL DEFAULT FALSE,-- Global quiet hours toggle
    dnd_start_time    TIME,                          -- DND start (e.g. '22:00:00')
    dnd_end_time      TIME,                          -- DND end (e.g. '08:00:00')
    channel_settings  JSONB NOT NULL DEFAULT '{}',   -- Category switches: {"PROMO": false, "ORDER": true}
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
```

#### 4. `delivery_attempts` (Physical Attempt Audit Table)
```sql
CREATE TABLE delivery_attempts (
    attempt_id        VARCHAR(64) PRIMARY KEY,       -- Attempt UUID
    notification_id   VARCHAR(64) NOT NULL,          -- Associated logical notification
    device_id         VARCHAR(64) NOT NULL,          -- Target device
    provider          VARCHAR(16) NOT NULL,          -- 'APNS', 'FCM'
    attempt_number    INT NOT NULL DEFAULT 1,        -- Current retry count
    status            VARCHAR(16) NOT NULL,          -- 'PENDING', 'SENT', 'FAILED', 'DISCARDED'
    error_code        VARCHAR(64),                   -- Error reason (BadDeviceToken, RateLimited, etc.)
    provider_msg_id   VARCHAR(128),                  -- APNs apns-id or FCM message_id
    lease_until       TIMESTAMP WITH TIME ZONE,      -- Distributed worker claim lease
    created_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_attempts_notif_device ON delivery_attempts(notification_id, device_id);
CREATE INDEX idx_attempts_lease ON delivery_attempts(status, lease_until) 
WHERE status = 'PENDING';
```

---

### 4.4 Preference Consistency & Delivery Boundaries

#### 1. Preference Consistency Strategy
- **Cache Architecture:** When users update DND or category toggles, the system uses Cache-Aside: write primary DB first, then invalidate the Redis cache key (`preference:{user_id}`).
- **Consistency Trade-off:** The Dispatcher reads cached preferences, accepting second-level eventual consistency.
- **Emergency Bypass Rule:** Critical security notifications (`priority = 1`, e.g., suspicious login alerts, two-factor auth codes) **bypass DND and category filters**, dispatching immediately.

#### 2. Three Delivery Boundaries & SLA Semantics

```text
[Notification Platform]
         │
  (Boundary 1: Provider Accepted) ──▶ Platform Service Level Agreement (SLA) boundary
         ▼
[APNs / FCM Gateway]
         │
  (Boundary 2: Device Displayed)  ──▶ Subject to battery-saver, flight mode, OS quota
         ▼
[Mobile OS / Notification Tray]
         │
  (Boundary 3: User Opened)       ──▶ Subject to user attention and subjective action
         ▼
[User Engagement]
```

- **Boundary 1: Provider Accepted**
  - Definition: APNs responds with HTTP 200 or FCM returns a valid `projects/.../messages/...` identifier.
  - Meaning: The notification has been durably accepted into Apple / Google push transit pipelines.
  - **Platform SLA Scope**: The notification platform's latency and availability guarantees (P95 < 2s) **strictly terminate at this boundary**.
- **Boundary 2: Device Displayed**
  - Definition: iOS Notification Service Extension or Android background worker receives the push packet, renders the system tray banner, and fires a `received` beacon.
  - Constraints: Uncontrollable if the device is powered off, in airplane mode, or throttled by OS battery management.
- **Boundary 3: User Opened**
  - Definition: User taps the lock-screen notification banner to launch the app, generating an `opened` telemetry event.
  - Constraints: Purely a product conversion metric (CTR), not an infrastructure delivery milestone.

---

## 5. Architectural Deep Dives

### 5.1 Dispatch Architecture

```text
Option A: Synchronous Direct Dispatch
[API Service] ──▶ Iterate recipients ──▶ HTTP POST APNs ──▶ Block on RTT ──▶ Return

Option B: Transactional Outbox + Multi-Tier Durable Queues (Recommended)
[API Service] ──▶ Local Transaction (DB + Outbox) ──▶ Instant Return
                       │ (CDC / Poller)
                       ▼
                 [Dispatcher] ──▶ Priority Queues (High / Bulk) ──▶ [Provider Workers]
```

| Dimension | Option A: Synchronous Direct Dispatch | Option B: Transactional Outbox + Durable Queues (Recommended) |
| :--- | :--- | :--- |
| **System Throughput** | **Tied strictly to external RTT**. With 100ms provider latency, a single thread achieves at most 10 QPS, failing under load. | **Extremely high**. Ingestion commits locally in 2–5ms; all downstream fanout and delivery are asynchronous and scale horizontally. |
| **Fault Isolation** | **Poor**. External latency spikes or vendor outages rapidly exhaust application threads, triggering site-wide cascading outages. | **Complete**. Downstream issues manifest only as queue backlog; upstream ingestion continues normally without blocking callers. |
| **Traffic Smoothing** | **None**. Ingestion bursts pass directly through to providers, immediately hitting APNs/FCM HTTP 429 quota limits. | **Natural buffer reservoir**. Peak 50K QPS ingestion is absorbed by message queues and consumed at a regulated steady rate. |
| **Data Durability** | In-flight tasks are permanently lost if worker processes crash or network sockets reset. | **Guaranteed zero drop**. Upstream writes rely on DB WAL (Debezium CDC) or Outbox polling, ensuring strict At-least-once processing. |
| **Recommendation** | **Never use in production**. Suitable only for internal prototypes and CLI tools. | **Industry standard** for high-volume, resilient mobile platforms. |

---

### 5.2 Delivery Semantics & Deduplication

```text
Lease Claim & Idempotent Delivery Flow:
[Worker] ──1. Claim (UPDATE delivery_attempts SET lease_until = NOW()+30s)──▶ [DB]
    │
    ├──2. Check Redis Idempotency Key SETNX (notif_id + device_id, 5 min)
    │     ├── Key exists  ──▶ Skip device (prevent concurrent duplicate push)
    │     └── Lock acquired ──▶ Proceed
    │
    ├──3. HTTP/2 POST APNs (Header: apns-id = attempt_id)
    │     ├── 200 OK ──▶ Set status = 'SENT'
    │     └── 429 / 5xx ──▶ Exponential Backoff + Jitter into Retry Queue
```

| Dimension | Option A: At-Most-Once Attempts | Option B: At-Least-Once + Leases & Deduplication (Recommended) |
| :--- | :--- | :--- |
| **Duplicate Banner Risk** | **Zero duplicates**. Messages are acknowledged immediately upon dequeuing; failed deliveries are discarded. | **Near zero**. Three defensive layers (Worker task leases, Redis pre-send locks, and native APNs `apns-id` deduplication) stop duplicates. |
| **Message Loss Rate** | **High**. Transient network drops, socket timeouts, or provider 503 errors cause silent push losses. | **Zero loss**. Tasks without successful acks are automatically reclaimed by healthy workers when their lease expires. |
| **Retry & Storm Control** | No retry mechanism exists. | Controlled via **Exponential Backoff with Full Jitter** and bounded retries (e.g. 3 attempts), routing exhausted tasks to a Dead Letter Queue (DLQ). |
| **Implementation Complexity**| Minimal; no attempt state persistence or lease management needed. | Moderate; requires managing attempt lifecycles, distributed locks, and DLQ monitoring. |
| **Recommendation** | Acceptable only for low-value promotional broadcasts where under-delivery is harmless. | **Mandatory production standard** for transactional and security-critical systems. |

---

### 5.3 Offline Delivery & Cost Optimization

```text
State Coalescing Mechanism:
Frequent Driver Location Updates (every 10 seconds):
Request 1: collapse_key = "trip_888", loc = (lat1, lon1)  ──┐
Request 2: collapse_key = "trip_888", loc = (lat2, lon2)  ──┼──▶ [Queue / APNs] Keeps latest state only!
Request 3: collapse_key = "trip_888", loc = (lat3, lon3)  ──┘
```

| Dimension | Option A: Retain Every Event Indefinitely | Option B: State Coalescing + Bounded TTL (Recommended) |
| :--- | :--- | :--- |
| **User Experience** | **Degraded**. A user returning online after days offline receives dozens of stale notification banners (outdated rides, scores). | **Clean**. Devices display only the most recent state; obsolete alerts are collapsed or discarded, eliminating notification spam. |
| **Storage & Bandwidth** | **Linear explosion**. Stale payloads bloat database and queue disks; huge bandwidth is wasted delivering dead notifications. | **Greatly reduced**. Minimizes queue size; leverages APNs `apns-collapse-id` and FCM `collapse_key` to overwrite on provider servers. |
| **Data Traceability** | Preserves full granular historical records for deep auditing. | Delivery tier retains only the latest snapshot; raw business audit logs are archived asynchronously to cold object storage. |
| **Workload Fit** | Best for append-only audit histories (e.g., individual financial debit lines). | **Ideal for continuous state updates** (ride-hailing location, food delivery tracking, live sports scores, market quotes). |
| **Recommendation** | Not viable for high-frequency mobile status pushes. | **Recommended production design**. Enforce physical TTLs (24h for transactions, 6h for marketing) and coalesce by business entity key. |

# System Design 07 · Design Photo Sharing & Home Feed System

Course Position: [[SystemDesign09 Consistent Hashing|09 Consistent Hashing]] → This Note → [[SystemDesign08 LLM Async RL Platform|08 Async LLM RL Platform]]

Designing a hyperscale photo sharing and social feed system (similar to Instagram / Twitter / Weibo) introduces three primary engineering bottlenecks: **extreme read/write skew (100:1 read-heavy workloads), massive unstructured binary image streaming, and power-law follower distributions (celebrity fan-out storms)**. The system must strictly enforce **physical separation between the control plane and the data plane**, and employ a **hybrid push-pull model** for feed generation.

---

## Part 1: Requirements Definition & System Scope

### 1.1 · Functional Requirements
1. **Photo Upload & Publishing**: Users can upload high-resolution photos with textual captions and optional geolocation tags;
2. **Social Follow Graph**: Users can follow and unfollow other creators;
3. **Personalized Home Feed**: Users can fetch and paginate a reverse-chronological timeline of posts published by creators they follow.

```text
[Explicitly Out of Scope]:
- Short-form video / Reels / 24-hour ephemeral Stories / Live streaming;
- Direct Messaging (IM / chat);
- Explore page discovery & machine learning recommendation rankers;
- Complex client-side image editing / filter pipelines;
- Likes and comments (architected as separate read/write microservices).
```

### 1.2 · Non-Functional Requirements & SLA Constraints
- **High Availability**: Target overall availability is $99.99\%$ (< 52 minutes downtime per year). During network partitions or node failures, **the read path (feed consumption) prioritizes availability over strict consistency (AP model under the CAP theorem)**;
- **Ultra-Low Read Latency**: Feed metadata query latency must satisfy $\mathbf{p99 < 200\text{ ms}}$ (excluding CDN image binary transfer time); upload session initialization must complete in $< 100\text{ ms}$;
- **Freshness & Consistency Constraints**:
  - Posts from standard creators must appear in active followers' timelines within $\le 5\text{ seconds}$;
  - Creators viewing their own profile post-upload must observe strict **Read-Your-Writes Consistency**;
- **Durability & Fault Tolerance**: Committed original and transcoded image variants must be replicated across multiple Availability Zones (Multi-AZ), guaranteeing **$99.999999999\%$ (11 nines)** durability.

---

## Part 2: Back-of-the-Envelope (BOE) Capacity Calculations

### 2.1 · Scale & Throughput Numbers

| Dimension | Baseline Assumptions | Throughput Derivation & Capacity Planning |
|---|---|---|
| **User Scale** | 1 Billion (1B) registered accounts; 100 Million (100M) Daily Active Users (DAU) | Hyperscale consumer social network benchmark |
| **Write QPS (Post Uploads)** | Assume $20\%$ of DAU publish 1 photo daily $\to \mathbf{20\text{M Posts / day}}$ | $$\text{Avg Write QPS} = \frac{20 \times 10^6}{86,400\text{ s}} \approx 231\text{ QPS}$$With a peak-to-average ratio of $4\sim 5\times$, **normal peak write QPS $\approx 1,200\text{ QPS}$**; provisioning for retry spikes and bursts, system is dimensioned for $\mathbf{5,000\text{ Peak Write QPS}}$. |
| **Read QPS (Feed Reads)** | Each DAU refreshes feed 10 times daily; social read-to-write ratio is typically $50:1 \sim 100:1$ | $$\text{Total Daily Reads} = 100\text{M} \times 10 = \mathbf{1,000,000,000\text{ Reads / day}}$$$$\text{Avg Read QPS} = \frac{10^9}{86,400\text{ s}} \approx 11,600\text{ QPS}$$Peak read multiplier of $3\sim 4\times$ yields **peak read QPS of $\approx 40,000 \sim 50,000\text{ QPS}$**. |

### 2.2 · Storage Capacity & Physical Footprint

#### 1. Binary Media Storage (Images)
- Uncompressed raw original averages $2\text{ MB}$;
- The asynchronous media pipeline crops and transcodes raw originals into 3 standard variants:
  - Thumbnail (grid preview): $50\text{ KB}$;
  - Medium (mobile feed scroll): $200\text{ KB}$;
  - Large (full-screen detail view): $800\text{ KB}$;
  - Total transcoded variants aggregate to $\approx 1\text{ MB}$.
- **Daily Physical Storage Growth**:
  $$\text{Daily Media Storage} = 20\text{M} \times (2\text{ MB Raw} + 1\text{ MB Variants}) = \mathbf{60\text{ TB / day}}$$
- **Annual Media Storage Footprint**:
  $$\text{Annual Media Storage} = 60\text{ TB/day} \times 365 \approx \mathbf{21.9\text{ PB / year}}$$

#### 2. Relational Metadata Storage
A single post entity `(post_id, author_id, caption, cdn_urls, lat, lon, created_at)` averages $500\text{ Bytes}$:
$$\text{Daily Metadata} = 20\text{M} \times 500\text{ B} = \mathbf{10\text{ GB / day}} \implies \mathbf{3.65\text{ TB / year}}$$
A single-node relational database will hit physical disk limits and B+ tree indexing bottlenecks within several years. **Metadata databases must be horizontally sharded by `user_id` (Z-axis Sharding)**.

### 2.3 · Network Egress Bandwidth & CDN Offload
- Each feed refresh requests 10 posts, each displaying 1 medium variant ($150\text{ KB}$), consuming $\approx 1.5\text{ MB}$ per refresh;
- **Total Peak Read Egress Bandwidth**:
  $$\text{Peak Read Egress} = 50,000\text{ QPS} \times 1.5\text{ MB} \times 8\text{ bits} = \mathbf{600\text{ Gbps}}$$
- **CDN Edge Offloading**:
  Images exhibit high spatial and temporal locality. Over $95\%$ of image requests terminate directly on global CDN edge PoPs. **With a $\ge 95\%$ CDN cache hit rate, actual origin egress traffic is compressed to $\le \mathbf{30\text{ Gbps}}$**.

---

## Part 3: High-Level Architecture & Plane Separation

```photo-sharing-architecture-visual
```

### 3.1 · Core Architectural Principle: Strict Plane Separation

In hyperscale multimedia system design, **Rule #1 is: Application servers and API gateways must never proxy large binary payload streams**.
- **Anti-Pattern (Bytes through API)**: Clients upload images directly to API servers, which buffer payloads into RAM and forward them to object storage. This ties up application threads and TCP connections for seconds, exhausting socket pools, inducing memory shallow-copy overhead, and triggering frequent Stop-the-World Garbage Collection (GC);
- **Production Standard (Pre-Signed Direct Upload)**: The Control Plane is strictly decoupled from the Data Plane. The API only performs authentication and issues a short-lived **S3 Pre-signed URL**; the client bypasses all application microservices and uploads binary bytes directly to distributed object storage via `HTTP PUT`.

```text
                           [CONTROL PLANE · METADATA]
                           Client ──► API Gateway ──► Upload Service ──► Metadata DB (PENDING)
                             │
                             │ (Acquire Short-Lived Pre-signed URL)
                             ▼
[DATA PLANE · DIRECT PUT]  Client ──────────────────────────────────────────► Raw Storage (S3 Staging)
                                                                                  │
                                                                                  ▼ (S3 Event Notify)
Processed Storage (CDN Origin) ◄── Media Processor (Resize / WebP / NSFW) ◄───────┘
          ▲
          │ (95%+ Edge Hit)
      CDN PoP ◄────────────────────────────────────────────────────────────────── Client (Loads Image)
```

### 3.2 · Component Responsibilities & Data Flow

1. **Client (Web / iOS / Android)**:
   - Publishing: Initiates upload session $\to$ uploads binary directly to S3 via pre-signed URL $\to$ commits post metadata;
   - Reading: Dispatches `GET /feed` to fetch lightweight post IDs, then loads WebP images directly from nearest CDN edge nodes;
2. **API Gateway & Load Balancer**:
   - Central entry point handling TLS termination, path routing, and distributed tracing injection;
   - Collaborates with **Authz & Rate Limiter** to throttle scrapers via distributed Redis token buckets;
3. **Upload Service**:
   - Validates MIME types and payload metadata;
   - Generates a 64-bit globally unique Snowflake `post_id`;
   - Persists a `PENDING` post record into the metadata database and issues a 15-minute Pre-signed URL;
   - Receives the client commit callback and advances status to `PROCESSING`;
4. **Raw Object Storage (S3 Staging)**:
   - Dedicated staging bucket for uncompressed client uploads (~2MB);
   - Governed by **24-Hour S3 Lifecycle Expiration Rules**: uncommitted orphan files are automatically deleted by the storage engine after 24 hours;
5. **Media Processor Pipeline**:
   - Asynchronously consumes `s3:ObjectCreated:*` notifications (buffered through Kafka / SQS);
   - Strips sensitive EXIF geo/device tags;
   - Generates Thumbnail, Medium, and Large variants encoded in WebP/AVIF;
   - Executes image safety classifiers (NSFW Filter);
   - Saves variants to processed object storage and promotes DB status to `READY`;
6. **Processed Object Storage & CDN Origins**:
   - Durable store for final transcoded assets using Content-Addressable keys;
   - Serves as origin for global CDN distribution across multiple Availability Zones;
7. **Metadata DB & Transactional Outbox**:
   - Sharded SQL database persisting core entities. In the transaction promoting status to `READY`, an event is inserted into the `outbox` table;
   - A CDC engine (Debezium) tails database binlogs and publishes `PostReady` messages to Kafka;
8. **View / Feed Service**:
   - Handles high-concurrency feed read traffic (50,000 peak QPS);
   - Implements the **Hybrid Fan-out Model**: retrieves the precomputed normal-author stream from the viewer's Redis Inbox, concurrently `MGET`s celebrity outboxes, runs an in-memory Heap Merge for the Top 20, and executes Hydration and authorization checks.

---

## Part 4: Data Entities & Database Architecture

### 4.1 · Core Schema Design

```sql
-- 1. Users table (Horizontally sharded by user_id)
CREATE TABLE users (
    user_id         BIGINT UNSIGNED NOT NULL PRIMARY KEY,
    username        VARCHAR(32) NOT NULL UNIQUE,
    follower_count  INT UNSIGNED NOT NULL DEFAULT 0,
    is_celebrity    TINYINT(1) NOT NULL DEFAULT 0, -- Set to 1 when followers >= 50,000
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- 2. Posts table (Sharded by user_id for single-shard transactions & read-your-writes)
CREATE TABLE posts (
    post_id         BIGINT UNSIGNED NOT NULL PRIMARY KEY, -- 64-bit Snowflake ID with timestamp
    user_id         BIGINT UNSIGNED NOT NULL,             -- Sharding Key
    caption         VARCHAR(1000) DEFAULT '',
    status          VARCHAR(16) NOT NULL DEFAULT 'PENDING', -- PENDING -> PROCESSING -> READY -> DELETED
    latitude        DECIMAL(10, 8) NULL,
    longitude       DECIMAL(11, 8) NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_created (user_id, created_at DESC)
) ENGINE=InnoDB;

-- 3. Media variants table (One-to-many relationship with posts)
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

-- 4. Follow graph table (Dual-index or dual-sharded)
CREATE TABLE follows (
    follower_id     BIGINT UNSIGNED NOT NULL,
    followee_id     BIGINT UNSIGNED NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    state           TINYINT(1) NOT NULL DEFAULT 1, -- 1: active follow, 0: unfollowed
    PRIMARY KEY (follower_id, followee_id),
    INDEX idx_followee (followee_id, follower_id)
) ENGINE=InnoDB;
```

### 4.2 · Sharded SQL vs NoSQL Tradeoffs

| Evaluation Dimension | Sharded SQL (MySQL / PostgreSQL) | Distributed NoSQL (Cassandra / DynamoDB) | Production Architecture Decision |
|---|---|---|---|
| **Transactions & Status Flow** | Strict **ACID local transactions** within a shard. State transitions (`PENDING` $\to$ `READY`) combined with Transactional Outbox guarantee zero inconsistency | Row-level atomicity only; lacks multi-state rollback semantics | **Post metadata adopts Sharded MySQL**: publishing and status machines require strict transactional consistency |
| **Clustering & Time Ordering** | Native composite clustered index `(user_id, created_at DESC)` delivers optimal chronological scans | Partition Key + Clustering Key yields $O(\log N)$ reverse time ordering | Both viable; MySQL offers richer ecosystem for pagination |
| **Follow Social Graph** | Requires dual indexing (querying followees vs querying followers); cross-shard queries are challenging | Requires maintaining two materialized tables (`user_followings` and `user_followers`) via CDC | **Follows table uses dual-sharded relational tables**; celebrity follower sets are cached in Redis Sets |
| **Timeline Storage** | Relational tables cannot handle sub-millisecond writes and reads across billions of inboxes | Wide-column stores handle sequential writes well, but RAM caches outperform disk | **Timeline inboxes bypass relational DB entirely**; materialized in **Redis Sorted Sets (ZSets)** |

---

## Part 5: Deep Dives & Architectural Tradeoffs

### 5.1 · Pre-Signed URL Direct Uploads & Lifecycle State Machine

```text
Client                  API Gateway / Upload Svc              Raw S3 Staging            Media Workers / DB
  │                                │                                │                           │
  ├─ 1. POST /posts/session ──────►│                                │                           │
  │    (metadata, mime_type)       ├─ 2. Insert PENDING record ─────┼──────────────────────────►│
  │                                ├─ 3. Generate 15min Signed URL  │                           │
  │◄─ 4. Return post_id + Signed ──┘                                │                           │
  │                                                                 │                           │
  ├─ 5. HTTP PUT binary bytes (Bypasses API Gateway) ──────────────►│                           │
  │◄─ 6. HTTP 200 OK ───────────────────────────────────────────────┘                           │
  │                                                                 │                           │
  ├─ 7. POST /posts/commit ───────►│                                │                           │
  │                                ├─ 8. Update status PROCESSING ──┼──────────────────────────►│
  │◄─ 9. Return 202 Accepted ──────┘                                │                           │
  │                                                                 ├─ 10. S3 Event Notify ────►│
  │                                                                 │                           ├─ 11. Transcode & NSFW
  │                                                                 │                           ├─ 12. Update READY
  │                                                                 │                           └─ 13. Emit PostReady
```

#### 1. Why Pre-Signed Uploads are Non-Negotiable at Scale
1. **Eliminating Gateway Bandwidth Saturation**:
   At peak write QPS of $1,200$ with $2\text{ MB}$ photos, proxying traffic through API gateways would generate:
   $$1,200\text{ QPS} \times 2\text{ MB} \times 8\text{ bits} = \mathbf{19.2\text{ Gbps}}$$
   Dozens of heavy application servers would be squandered merely moving bytes. With pre-signed URLs, the API handles lightweight JSON (<1 KB), cutting cluster requirements by $90\%$.
2. **Decoupling Slow Network Sockets**:
   Mobile uploads over high-latency cellular networks take $1\sim 5\text{ seconds}$. Direct uploads push slow socket management to object storage, freeing application worker threads in $< 50\text{ ms}$.

#### 2. Garbage Collection for Incomplete Uploads
If a client crashes after `HTTP PUT` without dispatching the `commit` call:
- **Metadata Layer**: Upload Service periodically sweeps `PENDING` records older than 2 hours, marking them `EXPIRED`;
- **Storage Layer**: Raw S3 Bucket runs an **S3 Lifecycle Expiration Rule**, automatically deleting files after 24 hours.

---

### 5.2 · Feed Architecture: Push vs Pull vs Hybrid Fan-Out

The fundamental feed design question: **How to deliver creator outbox posts into follower inboxes with minimal latency and resource cost?**

#### 1. Push Model (Fan-out-on-write)
- **Mechanism**: Every follower maintains an Inbox. When an author posts, background workers fan out and insert the `post_id` into every follower's Redis ZSet (capped at 800 items);
- **Complexity**:
  - **Write Amplification**: $\mathbf{O(F)}$ ($F$ = follower count);
  - **Read Complexity**: $\mathbf{O(1)}$ (point query to viewer's Redis Inbox, latency $< 2\text{ ms}$);
- **Failure Mode (Celebrity Fan-out Storm)**:
  When a celebrity with 60 million followers (e.g. Cristiano Ronaldo) posts, 60 million write tasks are dumped into message queues. Even at 100,000 writes/sec, **the queue backs up for 10 minutes**, saturating Redis cluster bandwidth. Furthermore, millions of inactive/dormant followers consume expensive RAM needlessly.

#### 2. Pull Model (Fan-out-on-read)
- **Mechanism**: No inboxes exist. Authors write once to their Outbox. When a user reads their feed, the system queries all $N$ followees, fetches their recent posts, and performs an in-memory K-way Heap Merge;
- **Complexity**:
  - **Write Complexity**: $\mathbf{O(1)}$ (zero write amplification);
  - **Read Complexity**: $\mathbf{O(N \log N + K \log N)}$ across $N$ network calls;
- **Failure Mode (Tail Latency & CPU Storms)**:
  If a user follows 1,000 authors, a single feed refresh fires 1,000 concurrent network RPCs. Total latency degrades to the slowest straggler node ($T_{\text{total}} = \max T_i$). Under 50,000 peak read QPS, application server CPUs collapse under heap sorting and network context switches.

#### 3. Production Hybrid Fan-Out Model (Instagram / Twitter Standard)

To resolve this conflict, authors are partitioned based on follower thresholds:

```text
                                [ Creator Publishes Post ]
                                            │
                           ┌────────────────┴────────────────┐
                           ▼                                 ▼
               [Standard Creator (F < 50k)]         [Celebrity (F ≥ 50k)]
                           │                                 │
                           ▼ (Push Fan-out)                  ▼ (Pull Mode)
                Query Active Followers              Write ONLY to Outbox
                           │                                 │ (No fan-out, < 5ms)
                           ▼                                 │
                Push post_id to Inboxes                      │
                (Redis ZSet, max 800)                        │
                                                             │
─────────────────────────────────────────────────────────────┼────────────────────────
                                                             │
                                   [ Follower Reads Feed ]   │
                                            │                │
               ┌────────────────────────────┴────────────────┘
               ▼                                             ▼
     [Read Viewer's Own Inbox]                     [Query Followed Celebrities]
     Fetch precomputed candidates from             MGET recent outbox posts from
     Redis ZSet (latency < 2ms)                    celebrities (bounded M < 30)
               │                                             │
               └────────────────────────────┬────────────────┘
                                            ▼
                           [Online Heap Merge (In-Memory)]
                           Merge-sort dual streams for Top 20 post_ids
                                            │
                                            ▼
                           [Hydration & Read-time Guards]
                           Batch MGET captions, CDN keys, filter blocks
                                            │
                                            ▼
                               Return JSON Payload to Client
```

- **Dynamic Celebrity Thresholding**:
  - Maintain a threshold (e.g. $F_{\text{threshold}} = 50,000$). An `is_celebrity` flag is maintained when follower counts update;
  - **Standard Creators**: Routed to **Push Model**, pushing only to active followers;
  - **Celebrities**: Routed to **Pull Model**, writing strictly to their personal Outbox with zero fan-out.
- **Online Dual-Stream Heap Merge**:
  - On feed refresh, the application fetches the precomputed Inbox stream from Redis;
  - Retrieves the small set of celebrities the user follows (typically $M < 30$), and fires an `MGET` against their outboxes;
  - Merges both streams in-memory using a priority queue (heap) in $< 5\text{ ms}$.
- **Active Follower Pruning**:
  - Writes are pushed **only to followers who have logged in within the past 7 days**;
  - Inactive users trigger a **Lazy Catch-up** upon re-opening the app: a background task reconstructs their inbox on demand. This saves **$> 70\%$ of timeline cache memory**.

---

### 5.3 · ID-Only Timeline Storage & Hydration Pattern

In timeline storage, **inboxes must never store redundant post JSON blobs (captions, image URLs, author avatars)**:
1. **Memory Explosion**: Storing 500B JSON objects across 100M users $\times$ 800 items requires tens of terabytes of expensive Redis RAM;
2. **Out-of-Sync Mutations**: Caption edits or post deletions cannot efficiently be updated across millions of follower inboxes;
3. **Privacy & Authorization Leaks**: If an author blocks a follower or sets their account to private, pre-materialized inboxes would leak private content.

#### The Three-Stage Hydration Pipeline:
- **Step 1 · Lightweight ID Candidates**:
  Inboxes store only compact tuples: `(post_id, author_id, timestamp)` ($24\text{ Bytes}$ per item);
- **Step 2 · Distributed Cache Batch MGET**:
  Once the final Top 20 `post_id`s are determined, the Feed Service issues a single `MGET posts:meta:<post_id>` to Redis / Memcached to hydrate captions, author profiles, and CDN URLs;
- **Step 3 · Read-Time Authorization Guards**:
  Before assembling the response, the service evaluates live user-to-author relationships: confirming the viewer is not blocked, the author is not banned, and the post is not marked `DELETED`.

---

### 5.4 · Global CDN Edge Strategy & Caching Architecture

```text
Client (Mobile App)
   │
   ├─ 1. Local Disk LRU Cache (Instant render, 0ms network)
   ▼
Edge CDN PoP (Nearest Edge Location)
   │
   ├─ 2. Edge Cache Hit Rate ≥ 95% (Latency < 20ms)
   ▼
Regional Shield Layer (Origin Request Coalescing)
   │
   ├─ 3. Singleflight Request Collapsing (Prevents Thundering Herd)
   ▼
Processed Object Storage (S3 Origin, absorbs ≤ 5% traffic)
```

1. **Content-Addressable Hashing**:
   All transcoded image variant keys embed content hashes (e.g. `cdn.domain.com/photos/a8f9c2d1_medium.webp`). Updates produce new URLs, **completely removing the operational burden of active CDN cache purges**;
2. **Aggressive HTTP Cache Headers**:
   Responses set immutable cache directives:
   `Cache-Control: public, max-age=31536000, immutable`
   Enabling client devices and edge PoPs to cache images for up to 1 year;
3. **Request Collapsing (Singleflight)**:
   When a celebrity publishes a viral photo and thousands of clients simultaneously request the image on a cold CDN PoP, **Singleflight collapses concurrent misses into a single origin fetch to S3**, preventing origin IOPS saturation.

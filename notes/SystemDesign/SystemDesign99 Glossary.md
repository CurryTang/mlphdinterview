# System Design 99 · 高频术语整合

课程位置：[[SystemDesign08 LLM Async RL Platform|08 异步 LLM RL 平台]] → 本篇；课程入口见 [[SystemDesign00 Overview|00 方法总览]]。

这页放在 System Design 的最后，作为查词和写 design doc 时的速查表。重点不是把词背下来，而是知道一个词在解决什么问题、什么时候值得引入、代价是什么。

## 一、扩展性和容量

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| scale up | 纵向扩展 | 换更强机器，比如更多 CPU、内存、磁盘 | Early stage we can scale up the DB, but eventually we need to scale out. |
| scale out | 横向扩展 | 加更多机器分摊流量 | The API servers should be stateless so we can scale out behind a load balancer. |
| horizontal scaling | 水平扩展 | 同 scale out，更常用于服务层、缓存层、worker 层 | Workers can be horizontally scaled based on queue depth. |
| vertical scaling | 垂直扩展 | 同 scale up，更常用于数据库早期扩容 | Vertical scaling buys time, but it has a ceiling. |
| stateless service | 无状态服务 | 服务实例不保存不可丢失状态 | Make API servers stateless; keep sessions and business state outside the process. |
| stateful service | 有状态服务 | 服务本身持有重要状态，比如 DB、Kafka、Redis cluster | Stateful components need replication, backup, and recovery plans. |
| load balancing | 负载均衡 | 把请求分发给多个实例 | Put a load balancer in front of the API servers. |
| autoscaling | 自动扩缩容 | 按 CPU、QPS、latency、queue depth 增减实例 | Worker pools can autoscale based on queue depth. |
| capacity planning | 容量规划 | 估 QPS、存储、带宽、峰值和增长 | Plan for peak QPS, not just average QPS. |
| bottleneck | 瓶颈 | 系统最先扛不住的部分 | The bottleneck is likely the database write path. |

## 二、性能指标

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| QPS / RPS | 每秒查询数 / 请求数 | 在线服务吞吐 | We need to support 100k QPS at peak. |
| TPS | 每秒事务数 | 数据库、支付、交易系统吞吐 | Payment TPS is much lower than read QPS. |
| throughput | 吞吐量 | 单位时间处理多少请求或数据 | Batching improves throughput but may hurt latency. |
| latency | 延迟 | 单个请求从进入到返回的耗时 | The p99 latency target is under 200 ms. |
| p95 / p99 latency | 分位延迟 | 95% / 99% 请求的延迟上界 | Average latency hides tail behavior. |
| tail latency | 长尾延迟 | 少数慢请求拖垮体验 | Fan-out queries amplify tail latency. |
| availability | 可用性 | 服务能正常响应的比例 | The target is 99.9% availability. |

一个常用估算：

```text
concurrency ~= QPS * average latency in seconds
```

如果 QPS 是 10,000，平均 latency 是 100 ms，那么系统里平均有大约 1,000 个 in-flight requests。

## 三、数据架构

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| hot/cold data separation | 冷热数据分离 | 高频近期数据和低频历史数据分开存 | Recent messages stay in hot storage; old messages move to cold storage. |
| data tiering | 数据分层 | hot / warm / cold 多级存储 | Use data tiering to control storage cost. |
| read-write separation | 读写分离 | 主库写，从库读 | Serve read-heavy traffic from replicas, but account for replication lag. |
| compute-storage separation | 计算存储分离 | 计算层和存储层独立扩展 | Compute-storage separation lets query workers scale independently. |
| sharding | 分片 | 按 user_id、region、time 等把数据拆到不同节点 | Shard by user_id to distribute writes. |
| partitioning | 分区 | 大表按范围、hash、list 拆分 | Partition event tables by time. |
| hot partition / hot shard | 热分区 / 热分片 | 某个分片流量远高于其他分片 | Celebrity users can create hot partitions. |
| consistent hashing | 一致性哈希 | 增减节点时减少数据迁移 | Consistent hashing is useful for cache routing. |
| resharding | 重新分片 | 分片规则变化后的数据迁移 | We need an online resharding plan. |
| data locality | 数据局部性 | 相关数据尽量放近 | Co-locate related data to reduce cross-node reads. |
| denormalization | 反范式化 | 为读性能冗余存储数据 | Denormalize user name into posts to avoid a join on the hot path. |
| materialized view | 物化视图 | 预先计算好的查询结果 | Use materialized views for expensive aggregations. |
| secondary index | 二级索引 | 主键之外的查询索引 | Searching by email needs a secondary index. |
| inverted index | 倒排索引 | 从词到文档的索引 | Full-text search relies on an inverted index. |
| LSM tree | 日志结构合并树 | 写优化存储结构，RocksDB、LevelDB、Cassandra 常见 | LSM trees trade background compaction for high write throughput. |
| B-tree / B+ tree | B 树 / B+ 树 | 传统数据库索引结构，读写较均衡 | InnoDB indexes are based on B+ trees. |
| WAL | 预写日志 | 先写日志再改数据，保证崩溃恢复 | WAL provides durability before pages are flushed. |
| compaction | 合并 / 压缩 | 清理旧版本、合并文件，常见于 LSM/Kafka | Compaction trades background IO for read efficiency. |

## 四、一致性和复制

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| strong consistency | 强一致性 | 写入后读能看到最新值 | Payments and inventory usually need strong consistency. |
| eventual consistency | 最终一致性 | 短时间不一致，但最终收敛 | Feeds and counters often tolerate eventual consistency. |
| read-your-writes consistency | 读己之写 | 用户写完后自己立刻能读到 | Users should see their own post immediately. |
| monotonic reads | 单调读 | 同一用户不会先读新值再读旧值 | Sticky reads can help preserve monotonic reads. |
| quorum | 仲裁 | 多数副本确认后才算成功 | Quorum writes balance consistency and availability. |
| leader-follower replication | 主从复制 | leader 写，followers 复制并可服务读 | Followers can serve reads when stale reads are acceptable. |
| primary-replica | 主副本架构 | 同 leader-follower | Design docs often use primary / replica terminology. |
| sync replication | 同步复制 | 写入等待副本确认 | Sync replication improves durability but increases write latency. |
| async replication | 异步复制 | 主库先返回，副本之后同步 | Read replicas are usually asynchronously replicated. |
| replication lag | 复制延迟 | 从库落后主库的时间 | Read-after-write can fail because of replication lag. |
| consensus | 共识协议 | 多节点对一个值达成一致 | Consensus is needed for leader election and metadata consistency. |
| Raft / Paxos | 共识算法 | 常用于 metadata、leader election、配置变更 | Use Raft-backed metadata when split brain is unacceptable. |
| CAP theorem | CAP 定理 | 网络分区下，一致性和可用性权衡 | CAP is a framing tool, not a full design answer. |
| ACID | 数据库事务属性 | 原子性、一致性、隔离性、持久性 | Orders, payments, and ledgers usually need ACID semantics. |
| BASE | 基本可用、软状态、最终一致 | 大规模 NoSQL 系统常见思想 | BASE systems push more correctness logic into the application. |

## 五、写入语义和消息处理

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| idempotency | 幂等性 | 重试多次和执行一次效果一样 | Payment APIs need idempotency keys. |
| deduplication | 去重 | 避免重复消息造成重复写 | Consumers deduplicate events by event_id. |
| at-most-once | 至多一次 | 可能丢，但不会重复 | At-most-once is acceptable for low-value telemetry. |
| at-least-once | 至少一次 | 不丢，但可能重复 | Most queues provide at-least-once delivery. |
| exactly-once | 精确一次 | 通常依赖事务、幂等、去重共同实现 | Say effectively exactly once unless the system truly guarantees it. |
| DLQ / Dead Letter Queue | 死信队列 | 多次处理失败的消息放到单独队列 | Failed messages go to a DLQ for inspection and replay. |
| replay | 重放 | 重新处理历史消息或事件 | Kafka allows replay from an offset. |
| offset | 消费位点 | 消费者处理到消息流的哪个位置 | Store offsets only after processing succeeds. |
| checkpoint | 检查点 | 保存进度或状态，便于恢复 | Stream jobs checkpoint state and offsets. |

## 六、分布式事务

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| 2PC / Two-Phase Commit | 两阶段提交 | 跨服务或跨库强一致事务协议 | 2PC gives strong semantics but can block and increase latency. |
| Saga pattern | Saga 模式 | 大事务拆成多个本地事务加补偿 | Use Saga for order-payment-inventory workflows. |
| compensating transaction | 补偿事务 | 某步失败后执行反向修正 | If inventory reservation fails after payment, issue a refund. |
| outbox pattern | Outbox 模式 | 本地事务写业务表和 outbox 表，再异步发消息 | Outbox avoids the DB-write-succeeded-but-message-lost problem. |
| CDC / Change Data Capture | 变更数据捕获 | 捕获 binlog 或 change log 推给下游 | CDC keeps search indexes and analytics stores in sync. |

## 七、缓存

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| cache-aside | 旁路缓存 | 应用先查缓存，miss 再查 DB 并回填 | Cache-aside is the default pattern for read-heavy endpoints. |
| lazy loading | 懒加载 | 需要时才加载到缓存 | Lazy loading avoids caching data nobody reads. |
| write-through cache | 写穿缓存 | 写缓存时同步写 DB | Write-through improves consistency at the cost of write latency. |
| write-back cache | 写回缓存 | 先写缓存，异步刷 DB | Write-back is fast but riskier during failures. |
| TTL | 过期时间 | 缓存多久失效 | Add TTL jitter to avoid synchronized expiration. |
| cache invalidation | 缓存失效 | 数据变更后删除或更新缓存 | Invalidation is where most cache bugs happen. |
| cache penetration | 缓存穿透 | 查不存在的数据，每次都打到 DB | Use negative caching or a Bloom filter. |
| negative caching | 空值缓存 | 不存在的结果也短时间缓存 | Cache not-found results briefly. |
| cache breakdown | 缓存击穿 | 热 key 过期，大量请求同时打 DB | Use singleflight or early refresh for hot keys. |
| cache avalanche | 缓存雪崩 | 大量 key 同时过期或缓存集群挂掉 | Randomize TTLs and use multi-level fallback. |
| thundering herd / cache stampede | 惊群 / 缓存踩踏 | 大量请求同时重建同一个缓存 | Coalesce requests for the same key. |
| singleflight / request coalescing | 请求合并 | 同一个 key 只有一个请求去 DB，其余等待 | Singleflight protects the backend on cache miss. |
| stale-while-revalidate | 先返回旧值，后台刷新 | 用一点新鲜度换稳定低延迟 | CDN and feed systems often use stale-while-revalidate. |
| CDN | 内容分发网络 | 静态资源缓存到边缘节点 | Use CDN for images, video, JS, and CSS. |
| edge cache | 边缘缓存 | 离用户更近的缓存 | Edge cache reduces global latency. |

## 八、流量控制

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| buffering | 缓冲 / 削峰填谷 | 用队列吸收峰值，后端慢慢处理 | Use a queue as a buffer to smooth traffic spikes. |
| load leveling | 负载平滑 | 削峰填谷的架构表达 | Queue-based load leveling protects downstream systems. |
| traffic shaping | 流量整形 | 控制流量进入系统的速度和形态 | Gateways can shape traffic before it reaches the service. |
| rate limiting | 限流 | 限制用户或服务单位时间请求数 | Rate limit by user_id, tenant_id, and IP. |
| token bucket | 令牌桶 | 允许一定 burst 的限流算法 | Token bucket allows short bursts while enforcing long-term rate. |
| leaky bucket | 漏桶 | 以固定速率放行请求 | Leaky bucket smooths traffic more aggressively. |
| backpressure | 反压 | 下游处理不过来时通知上游减速 | Apply backpressure when queue length grows. |
| load shedding | 丢弃部分负载 | 过载时主动拒绝低优先级请求 | Shed non-critical traffic to protect core flows. |
| priority queue | 优先级队列 | 高优任务先处理 | Payment jobs should outrank notification jobs. |
| admission control | 准入控制 | 请求进系统前判断能不能接 | Admission control prevents overload before work is admitted. |

## 九、弹性设计

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| timeout | 超时 | 调外部服务不能无限等 | Every RPC should have a timeout. |
| retry | 重试 | 临时失败后再次请求 | Retries must be paired with idempotency. |
| exponential backoff | 指数退避 | 重试间隔逐渐变长 | Backoff prevents retry storms. |
| jitter | 随机抖动 | 给重试或 TTL 加随机性 | Add jitter so clients do not retry at the same time. |
| circuit breaker | 熔断器 | 下游持续失败时短时间停止调用 | Open the circuit to avoid cascading failures. |
| graceful degradation | 优雅降级 | 非核心功能失败时核心功能继续可用 | If recommendations fail, return a default ranking. |
| fallback | 兜底方案 | 主路径失败后的备用结果 | Fallback to cached data. |
| bulkhead isolation | 舱壁隔离 | 不同资源池隔离，避免互相拖垮 | Separate thread pools for payment and analytics. |
| cascading failure | 级联故障 | 一个服务故障导致连锁崩溃 | Timeouts, circuit breakers, and isolation all reduce cascading failures. |

## 十、异步架构

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| message queue | 消息队列 | 解耦生产者和消费者 | Queue writes to decouple the API from slow downstream processing. |
| pub/sub | 发布订阅 | 一个事件被多个消费者订阅 | UserCreated can feed email, analytics, and recommendation systems. |
| event-driven architecture | 事件驱动架构 | 服务通过事件通信 | Event-driven design reduces direct service coupling. |
| stream processing | 流处理 | 实时处理连续事件流 | Use Flink or Kafka Streams for real-time aggregates. |
| batch processing | 批处理 | 定时处理大量历史数据 | Reports and offline features are usually batch processed. |
| event sourcing | 事件溯源 | 保存所有状态变化事件 | Event sourcing helps with auditability. |
| fan-out | 扇出 | 一个请求或事件分发到多个下游 | Feed and notification systems often fan out writes. |
| fan-in | 扇入 | 多个结果聚合成一个结果 | Search services fan in results from multiple shards. |

## 十一、可用性和容灾

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| SPOF / single point of failure | 单点故障 | 某组件挂了整个系统挂 | The primary DB is a potential SPOF. |
| HA / high availability | 高可用 | 单点失败后系统仍能服务 | HA needs replicas, health checks, and failover. |
| DR / disaster recovery | 灾难恢复 | 整个机房或区域故障后的恢复能力 | DR planning covers region-level outages. |
| failover | 故障转移 | 主节点挂了切到备节点 | Automatic failover reduces downtime. |
| active-active | 双活 / 多活 | 多区域同时服务流量 | Active-active is powerful but makes conflict handling harder. |
| active-passive | 主备 | 主区域服务，备区域等待接管 | Active-passive is simpler but uses resources less efficiently. |
| multi-AZ | 多可用区 | 同 region 内跨 AZ 部署 | Multi-AZ protects against single-AZ failure. |
| multi-region | 多区域部署 | 跨地理区域部署 | Multi-region protects against region-level outages. |
| RPO | 恢复点目标 | 最多能丢多少数据 | RPO is 5 minutes. |
| RTO | 恢复时间目标 | 故障后多久恢复服务 | RTO is under 30 minutes. |
| backup / restore | 备份 / 恢复 | 定期备份并验证能恢复 | A backup is not useful until restore has been tested. |
| health check | 健康检查 | 判断服务是否能接流量 | Load balancers rely on health checks. |
| liveness probe | 存活探针 | 判断进程是否需要重启 | Kubernetes uses liveness probes to restart stuck containers. |
| readiness probe | 就绪探针 | 判断实例是否可以接流量 | Readiness prevents traffic from reaching a cold-starting pod. |

## 十二、发布部署

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| canary release | 金丝雀发布 | 先给少量流量发布新版本 | Canary 1% traffic before full rollout. |
| blue-green deployment | 蓝绿部署 | 新旧两套环境切流量 | Blue-green makes rollback fast but costs more capacity. |
| rollback | 回滚 | 新版本有问题切回旧版本 | Keep rollback fast for risky releases. |
| roll-forward | 前滚 | 继续修复发布，不回旧版本 | Roll-forward is common when schema changes are hard to undo. |
| feature flag | 功能开关 | 不重新部署就开关功能 | Feature flags make risky features easier to disable. |
| dark launch | 暗发布 | 后端跑新逻辑但不暴露给用户 | Dark launch validates performance before user exposure. |
| shadow traffic | 影子流量 | 复制真实流量到新系统，不影响用户 | Shadow traffic helps compare a new service with production behavior. |
| A/B testing | A/B 测试 | 不同用户看到不同版本，比较指标 | A/B testing measures product impact. |
| schema migration | 表结构迁移 | 数据库 schema 变更 | Schema migration must be compatible with old and new code. |
| backfill | 回填 | 为历史数据补新字段或新索引 | Run backfill asynchronously. |

## 十三、可观测性

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| observability | 可观测性 | 通过指标、日志、链路追踪理解系统状态 | Observability starts with metrics, logs, and traces. |
| metrics | 指标 | QPS、latency、error rate、CPU 等数字 | Metrics power dashboards and alerts. |
| logs | 日志 | 记录事件和错误上下文 | Logs should include request_id and useful context. |
| tracing | 链路追踪 | 看请求经过哪些服务、每步耗时 | Distributed tracing helps debug slow microservice requests. |
| SLI | 服务水平指标 | 衡量服务质量的指标 | Latency and availability are common SLIs. |
| SLO | 服务水平目标 | 内部承诺的目标 | 99.9% of requests under 200 ms is an SLO. |
| SLA | 服务水平协议 | 对客户承诺，违约可能赔偿 | SLA is external and contractual. |
| error budget | 错误预算 | SLO 允许的失败额度 | Error budget guides whether to ship features or fix reliability. |
| alerting | 报警 | 指标异常时通知 on-call | Alerts should be actionable. |
| runbook | 运维手册 | 故障发生时怎么处理 | Critical alerts should link to a runbook. |

## 十四、API 和安全

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| API gateway | API 网关 | 统一入口，做鉴权、限流、路由 | Put authentication and rate limiting at the gateway. |
| pagination | 分页 | 大结果集分批返回 | Always paginate large list APIs. |
| cursor-based pagination | 游标分页 | 用 cursor 继续取下一页 | Cursor pagination is better for feeds than offset pagination. |
| versioning | API 版本管理 | v1/v2 兼容升级 | API versioning prevents breaking old clients. |
| backward compatibility | 向后兼容 | 新版本不破坏旧客户端 | Keep response fields backward compatible. |
| contract | 接口契约 | 服务之间约定输入输出 | Define a clear API contract between services. |
| authentication / AuthN | 认证 | 你是谁 | AuthN verifies identity. |
| authorization / AuthZ | 授权 | 你能做什么 | AuthZ checks permissions. |
| RBAC | 基于角色的访问控制 | admin / user / viewer 不同权限 | RBAC works well for coarse-grained enterprise permissions. |
| ABAC | 基于属性的访问控制 | 根据用户、资源、环境属性判断权限 | ABAC supports more fine-grained policies. |
| JWT | JSON Web Token | 常见无状态 token 格式 | JWT is stateless, but revocation and key rotation need design. |
| OAuth | 授权协议 | 第三方授权登录或授权访问 | OAuth is used for "login with Google" style flows. |
| mTLS | 双向 TLS | 服务之间互相验证身份 | mTLS is common in service mesh setups. |
| encryption in transit | 传输中加密 | 网络传输用 TLS | Use TLS for data in transit. |
| encryption at rest | 静态加密 | 磁盘或数据库存储加密 | Encrypt sensitive data at rest. |
| PII | 个人可识别信息 | email、phone、SSN、地址等 | PII needs access control, masking, and audit logs. |
| data masking | 数据脱敏 | 隐藏敏感字段 | Mask PII in logs and admin tools. |
| audit log | 审计日志 | 谁在什么时候做了什么 | Audit logs are required for sensitive operations. |

## 十五、权衡词

| 术语 | 中文 | 用来说明什么 | 自然用法 |
| --- | --- | --- | --- |
| trade-off | 权衡 | 不能所有指标同时最优 | This is a latency-consistency trade-off. |
| cost-performance trade-off | 成本性能权衡 | 更快通常更贵 | Caching improves latency but adds consistency complexity. |
| complexity budget | 复杂度预算 | 不为小规模问题过度设计 | At this scale, sharding is premature. |
| over-engineering | 过度设计 | 为不存在的问题引入复杂系统 | Avoid over-engineering before measuring the bottleneck. |
| premature optimization | 过早优化 | 在瓶颈出现前做复杂优化 | Start simple and optimize after measuring. |

## 十六、几组常一起出现的表达

### 削峰填谷

```text
Use a queue as a buffer to smooth traffic spikes and decouple writes from downstream processing.
```

也可以写成：

```text
Use queue-based buffering / load leveling to absorb peak traffic.
```

### 冷热分离

```text
Keep recent, frequently accessed data in hot storage, and move old data to cold storage to reduce cost.
```

### 读写分离

```text
Route writes to the primary database and serve read-heavy traffic from read replicas, while accounting for replication lag.
```

### 计算存储分离

```text
Separate compute from storage so each layer can scale independently.
```

### 至少一次投递下的消费者设计

```text
Since the queue provides at-least-once delivery, consumers must be idempotent and deduplicate events by event_id.
```

### 过载保护

```text
Apply rate limiting at the edge, backpressure between services, circuit breakers for unhealthy dependencies, and graceful degradation for non-critical features.
```

## 十七、使用这些词时先问的问题

```text
这个机制解决的是读瓶颈、写瓶颈、容量瓶颈、延迟问题，还是可用性问题？
它把复杂性放到了哪里？
它会引入什么新问题？
现在的规模是否真的需要它？
```

术语本身不加分，能把机制、瓶颈和代价连起来才有用。

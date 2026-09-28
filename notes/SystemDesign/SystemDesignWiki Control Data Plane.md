# Wiki · Control Plane vs Data Plane (控制面与数据面解耦)

Wiki 词条归属：[[SystemDesign00 Overview|00 系统设计全局蓝图]] → [[SystemDesignWiki Control Data Plane|Wiki · Control Plane vs Data Plane]]

**控制面与数据面解耦（Control Plane vs Data Plane Separation）** 是现代大规模分布式系统、网络虚拟化与云原生基础设施（如 Kubernetes、Service Mesh、SDN）中最关键的高可用与可扩展性设计范式。

---

## 1 · 核心定义与架构职责切分

控制面与数据面在系统的处理路径（Path）上承担截然不同的职责：

```text
       [ Operator / API / GitOps ]
                    │
                    ▼
┌────────────────────────────────────────────────────────┐
│ 控制面 (Control Plane)                                 │
│ 职责：策略计算、元数据仲裁、拓扑编排、声明式收敛        │
│ 关键特性：高一致性 (Raft/etcd)、复杂决策、中低 QPS     │
└──────────────────────────┬─────────────────────────────┘
                           │ (下发指令 / xDS / 本地规则)
                           ▼
┌────────────────────────────────────────────────────────┐
│ 数据面 (Data Plane)                                    │
│ 职责：数据包转发、协议代理、I/O 读写、高频业务计算      │
│ 关键特性：极高吞吐 (百万元 QPS)、微秒级时延、无锁/零拷贝 │
└────────────────────────────────────────────────────────┘
        ▲                                ▲
        │ (Client Traffic)               │ (Client Traffic)
  [ Inbound I/O ]                 [ Outbound I/O ]
```

### 1.1 · 控制面（Control Plane）：系统的“大脑”
- **核心职能**：定义系统“应该处于什么状态”（Desired State），负责策略决策（Routing Policy）、认证鉴权（AuthN/AuthZ）、服务发现拓扑计算、资源调度与故障仲裁。
- **运行特征**：逻辑复杂，重度依赖强一致性存储（如 Raft、etcd、Paxos）进行状态持久化；通常以周期性轮询或事件驱动的“调和循环”（Reconciliation Loop）工作。

### 1.2 · 数据面（Data Plane）：系统的“躯干与四肢”
- **核心职能**：执行控制面下发的策略，对实际经过的业务流量、数据包或存储请求进行高频处理、校验、路由与数据传输。
- **运行特征**：逻辑扁平、高度优化；避免与中央存储通信；采用本地内存规则表、零拷贝（Zero-Copy）、eBPF、DPDK 或无锁并发模型，追求极致的吞吐与确定性超低延迟。

---

## 2 · 核心设计哲学：生存独立性与故障隔离

解耦最根本的工程目标是**爆炸半径隔离（Blast Radius Containment）与数据面自愈运行**：

> [!IMPORTANT]
> **控制面-数据面第一生存法则（Survival Invariant）**：
> **控制面即便全局宕机崩溃，正在运行的数据面也必须完全不受影响，依靠最后已知的有效状态（Last-Known Good State）继续提供 100% 的业务转发服务！**

1. **无控制面依赖的数据转发**：
   - 数据面处理外部业务流量时，绝对禁止以同步 RPC 调用控制面进行决策。
   - 所有路由表、证书、限流阈值必须由控制面提前推送到数据面节点的本地内存缓存中（如 Envoy xDS 协议、Linux 内核 iptables/eBPF 映射表）。
2. **渐进式退化（Graceful Degradation）**：
   - 当控制面不可用时，系统失去“变更能力”（无法部署新服务、无法扩缩容、无法创建新路由），但系统的“服务能力”（现有业务调用、现有连接）保持零损耗。

---

## 3 · 工业级系统实现映射

### 3.1 · Kubernetes 架构中的映射
Kubernetes 是控制面与数据面解耦的教科书式工业实现：

| 层次划分 | 组件清单 | 核心职责与关键特性 |
|---|---|---|
| **K8s 控制面 (Control Plane)** | `kube-apiserver`<br>`etcd`<br>`kube-scheduler`<br>`kube-controller-manager` | · 维护声明式资源配置（CRD、Deployment、Service）。<br>· etcd 运行 Raft 协议保证全局单点强一致。<br>· 调度器进行 Bin-packing 打分计算。<br>· 处理 QPS 通常在 $100 - 1,000\text{ QPS}$ 量级。 |
| **K8s 数据面 (Data Plane)** | `kubelet`<br>`kube-proxy` / Cilium eBPF<br>Container Runtime (containerd)<br>业务 Application Pods | · `kube-proxy` / Cilium 将 Service 映射为 Linux 内核路由规则。<br>· Application Pods 处理来自真实终端用户的业务流量（海量 QPS）。<br>· 若整个 Master 节点集群瘫痪，Worker 节点上的容器与网络转发照常运行。 |

### 3.2 · 服务网格（Service Mesh: Istio + Envoy）
- **控制面（Istiod）**：
  - 监听 K8s API 变更，将 VirtualService / DestinationRule 规则计算并转换为 Envoy 能够理解的标准化 **xDS 协议配置**（LDS, RDS, CDS, EDS）。
  - 下发频率较低（秒级或变更时触发），采用 gRPC 流长连接单向或双向同步。
- **数据面（Envoy Sidecar）**：
  - C++ 编写，基于非阻塞事件驱动模型（epoll）。
  - 拦截所有微服务进出流量（mTLS 加解密、重试、熔断、遥测采集），单核转发能力达数万 RPS，处理延迟 $< 1\text{ ms}$。

### 3.3 · 分布式存储与流系统中的映射
- **Kafka**：
  - **控制面**：KRaft Controller Quorum（基于 Raft 协议维护分区 Leader 分配、Broker 拓扑元数据），低频元数据操作。
  - **数据面**：Kafka Broker 存储引擎（`sendfile` 零拷贝、Page Cache 只追加日志写入），承载数百万 QPS 业务消息流动。
- **TiDB / CockroachDB**：
  - **控制面**：Placement Driver (PD) 负责 Region 分裂感知、调度与 TSO 分布式时间戳分配。
  - **数据面**：TiKV 节点基于 Raft-engine 执行真实的 KV 存储、SSTable 读写与 MVCC 事务并发。

### 3.4 · 分布式计算与 CI/CD 流水线（Workflow vs Job 实体建模解耦）
在任务调度与流水线编排系统（如 GitHub Actions、GitLab CI、Argo Workflows）中，**若将整个流水线设计为一个整体（只有 Workflow 没有 Job），或者扁平化为一堆脚本，系统在调度、扩展性与容错上会迅速崩溃**。

拆分成 **Workflow（图/编排控制）** 与 **Job（节点/计算数据）** 两个独立对象的本质是控制面与数据面的精细化建模：

```text
[ Workflow Orchestrator (控制面) ]
  │
  ├─ DAG 拓扑解析: Job A (编译) ──┬──> Job B (单测) ──┬──> Job D (镜像推送)
  │                              └──> Job C (打包) ──┘
  ├─ 入度监听: In-degree == 0 触发派发
  └─ 调度分流 (派发至异构队列)
         │                         │                         │
         ▼                         ▼                         ▼
   [ Queue: Linux ]          [ Queue: macOS ]          [ Queue: GPU ]
         │                         │                         │
[ Worker: 1-Core Linux ]   [ Worker: Xcode Host ]   [ Worker: A100 Cluster ]
  (数据面: Lint 扫描)        (数据面: iOS 编译)        (数据面: 模型评测)
```

1. **DAG 拓扑编排与状态建模（图 vs 节点）**：
   - **关系抽象**：Workflow 是一张图（DAG），而 Job 是图上的节点（Nodes）。
   - **依赖隔离**：节点之间存在显式的边（如 `job_dependency` 表记录 `from_job_id` 与 `to_job_id`）。只有把 Job 单独抽离为实体表，数据库才能清晰表达出“编译（Job A）通过后，并发触发单测（Job B）与打包镜像（Job C）”的拓扑流转（入度计算与扇出通知）。
2. **异构计算环境与调度隔离（Heterogeneous Runners）**：
   - 现实中，同一个代码提交触发的流水线往往需要截然不同的硬件与运行时：
     - **代码检查/语法扫描（Lint）**：只需要廉价、轻量的 1 核 Linux 容器；
     - **移动端编译**：必须调度到配置了 Xcode 的专属 macOS 物理机；
     - **模型测试**：必须调度到挂载了 GPU（如 NVIDIA A100）的专用实例。
   - 如果没有 Job 这一层抽象，系统就无法将不同计算属性分配给不同的物理宿主机；将 Job 作为调度粒度，Runner Gateway 才能根据 Job 的规格需求（Label / Resource Request）精准派发给匹配的计算节点。
3. **故障爆炸半径与局部断点重试（Partial Retries & Fault Tolerance）**：
   - 若流水线未解耦为独立 Job，当最后一步“镜像推送”因网络超时失败时，整个流水线必须从头重跑，白白浪费前面耗时 40 分钟的编译与测试算力。
   - 拆分为独立 Job 后，每个 Job 拥有独立的状态机（`PENDING`、`RUNNING`、`SUCCESS`、`FAILED`）与产物存储引用（S3 Artifacts）；重试仅针对失败节点，前置节点结果直接复用。

---

## 4 · 量化对比基准（Numbers & Metrics）

```text
┌────────────────────────────────────────────────────────────────────────┐
│                    控制面 vs 数据面 关键量化指标对比                     │
├───────────────────────┬────────────────────────┬───────────────────────┤
│ 指标维度              │ 控制面 (Control Plane)  │ 数据面 (Data Plane)   │
├───────────────────────┼────────────────────────┼───────────────────────┤
│ 典型吞吐量 (QPS)      │ 10 - 2,000 QPS         │ 100,000 - 10,000,000+ │
│ 请求处理时延          │ 10 ms - 数秒 (异步收敛)│ < 100 μs - 5 ms       │
│ 状态存储介质          │ 强一致磁盘 WAL (etcd)   │ 纯内存表 / 寄存器 / DMA│
│ 核心数据结构          │ AST、DAG、B+树、Raft Log│ Hash 表、Radix 树、Ring│
│ 计算模型              │ 阻塞/锁/多副本共识协商 │ 无锁 (Lock-free)/eBPF  │
│ 故障影响范围          │ 暂停管理运维，存量可用 │ 业务直接报错或丢包中断 │
└───────────────────────┴────────────────────────┴───────────────────────┘
```

---

## 5 · 适用场景与面试系统设计决策树

### 5.1 · 何时必须在系统设计中主动提出“两面分离”？
1. **构建 API 网关或反向代理平台**：
   - 规则管理与后台配置（控制面）与网关转发内核（OpenResty / Envoy 数据面）严格物理隔离，禁止网关转发每个请求时实时同步去查管理后台 MySQL。
2. **大规模分布式缓存调度 / 分片代理**：
   - 一致性哈希环的拓扑更新（控制面）广播至客户端 SDK 本地内存（数据面），客户端根据本地视图直连存储分片。
3. **弹性调度与任务分发平台（如 Case 08 / Case 12）**：
   - 调度仲裁器（Master 控制面）负责节点心跳和分发，实际计算 Worker（数据面）以批量本地流水线执行，Worker 崩溃仅触发本地重试或局部任务窃取。

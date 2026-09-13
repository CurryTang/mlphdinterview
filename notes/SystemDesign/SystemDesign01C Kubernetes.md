# System Design 01C · Kubernetes 与 LLM 训练控制面

课程位置：[[SystemDesign01B Virtualization Containers|01B 虚拟化与容器]] → 本篇 → [[SystemDesign01D Redis|01D Redis]]

上一篇介绍 container 是进程隔离与打包。
本篇介绍如何将这些孤立的进程分配到节点。

Kubernetes 是一组声明式控制面。
它在核心的 apiserver 和后端的 etcd 存储中维护期望状态。
分布式的 controller 和居中调度的 scheduler 驱动整个物理集群向该状态推进。

LLM 训练在此之上，需要一层严格的 worker 组语义。
这是原生 Kubernetes 默认不具备的：
- 异构 GPU 设备分配。
- 指定网卡拓扑。
- 多 Worker 同时启动（gang 调度）。
- 防止部分启动。
- 多团队共享资源池的排队、配额管理和动态抢占。
- 故障发生后，整组停止并从外部 checkpoint 进行完整恢复。
- 网络拓扑感知与底层通信路线。

配套实验在本地实现了一个轻量级的训练控制面。
代码位于 `project/LLMTrainLab/`，或 [GitHub: CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)。

```k8s-hierarchy-visual
```

---

## 1 · 集群组件与对象生命周期

一个完整的集群包含中心化的 control plane 和承载计算负载的 worker node。

| 组件 | 职责 | 故障影响 |
|---|---|---|
| kube-apiserver | 对外 HTTP API；集群所有状态读写的入口。 | 控制面瘫痪。无法提交新 Job。已有 Pod 若不依赖外部 API 则继续运行。 |
| etcd | 集群状态的唯一事实来源。 | apiserver 无法读写。控制面操作停止。 |
| kube-scheduler | 给未绑定节点的 Pending Pod 分配节点。 | 新 Pod 处于 Pending 状态。已有 Pod 不受影响。 |
| kube-controller-manager | 执行各类资源对象的 reconcile loop（如 ReplicaSet、Job）。 | Pod 崩溃后，上层控制器无法响应并重建。 |
| kubelet | 节点代理：通过 CRI 接口运行底层容器，并汇报节点状态。 | 节点的心跳丢失。控制面将该节点上的 Pod 状态标记为 Unknown。 |
| container runtime | containerd 或 dockerd 等，负责创建 Linux 容器环境。 | 节点上无法启动新进程。运行中的容器可能仍然存活。 |
| kube-proxy / CNI | 负责 Service 流量转发与跨节点的 Pod 网络底层连通。 | 内部的 ClusterIP 负载均衡失效。跨节点的网络路由中断。 |
| Device Plugin | 向驻留的 kubelet 注册 GPU、RDMA 网卡等扩展资源。 | 调度器无法发现可用的 GPU。容器因挂载硬件失败而无法启动。 |

控制面的职责是存储期望状态与观测实际状态。
数据面承载了业务进程、网卡通信、GPU 计算和磁盘 I/O。
模型训练的 Token 生成以及 all-reduce 同步吞吐，不经过核心的 apiserver。
控制面的短暂不可用，不会直接打断数据面正在运行的计算过程。

一次 4 Worker 的深度学习训练 Job 时间线：
1. 用户向集群提交一个类型为 LLMJob 或 PyTorchJob 的自定义资源。
2. Kueue 拦截并执行准入逻辑。
3. 它验证当前团队的配额和优先级。
4. 将任务置入等待队列。
5. 触发 gang 调度策略。
6. 等待 4 个 GPU slot 在同一时刻全部可用，才会准入。
7. 下游控制器创建 4 个独立的 Pod。
8. scheduler 将它们分别 bind 到具体的物理节点上。
9. 目标节点上的 kubelet 拉取容器镜像。
10. 同时挂载所需的 PVC 存储卷。
11. kubelet 调用底层设备插件，执行 Allocate() 操作。
12. 获取特定的物理 GPU 设备节点。
13. 容器内进程启动。
14. 外层控制器将 RANK、WORLD_SIZE 和 MASTER_ADDR 环境变量注入。
15. 进程的通信库（如 NCCL）执行 init barrier。
16. 进行网络连通性测试。
17. 确认所有 Worker 均已就绪。
18. 开始执行实际的模型训练 step 计算。
19. 在训练过程中，周期性地写入 checkpoint。
20. 将文件并发写入共享的 RWX 存储后端。
21. 任务最终以 Succeeded 状态退出。
22. 如果有部分 Worker 发生硬件故障。
23. 整组停止运行，并在资源重新分配后从最近的 ckpt 恢复。

Pod 生命周期内部的相位（Phase）流转：
```text
Pending
  表示尚未调度到合适节点，或节点正在下载镜像，或资源尚未就绪。
Running
  表示容器内部主进程正在执行。
Succeeded / Failed
  表示主进程以 0 或非 0 状态码终止。
```

CrashLoopBackOff 是 kubelet 不断尝试重启故障容器的行为。
对于无状态 Web 服务，单 Pod 原地重启通常是有效的。
但在并行训练任务中，一旦发生显存溢出或断网导致 NCCL communicator 崩溃，单 Pod 的重启是无效的。
因为通信环内部的网络拓扑逻辑结构已经被破坏。
外层全局的 Job 控制器必须主动介入。
控制器销毁当前组内的所有 Pod，释放资源，并重建整个 Worker 组。

```k8s-lifecycle-visual
```

---

## 2 · 对象层级

在图纸中，对象层级看起来像是包裹的树状嵌套结构。
在代码实现中，它们依靠标签（label）动态筛选和 ownerReference 字段建立关联。
如果通过 API 调用删除父对象，并且没有指定级联删除，失去关联的子对象可能成为游荡的孤儿进程，继续消耗物理资源。

### Container
Container 是基本执行单元，是镜像文件、执行命令和资源限制配置的具体组合。
同一个 Pod 内可以包含多个不同的 container，它们共享同一套网络 namespace（IP 和端口空间）以及挂载的卷空间。
一个训练 Worker 的 Pod 通常包含三个部分：
- 负责主体的容器，执行 `torchrun` 计算。
- 负责观测的 sidecar 容器，收集 exporter 指标或提供网络 debug 工具。
- 负责准备的 init container，拉取数据集或生成网络配置文件。

### Pod
Pod 是 Kubernetes 体系中能够被调度和部署的最小单元。
Pod 内部署的容器运行在同一台物理机或虚拟机节点上。
Pod IP 是由网络插件临时分配的。
一旦 Pod 被销毁，该 IP 就会被回收并改变。
在分布式架构设计中，不能依赖 Pod IP 作为需要跨越生命周期的稳定标识。

### ReplicaSet
ReplicaSet 在后台监控集群状态。
它的任务是确保匹配标签 selector 的 Pod 存活数量等于 `spec.replicas` 声明的数字。
多则删，少则补。

### Deployment
Deployment 构建在 ReplicaSet 之上，增加了变更管理逻辑。
它提供版本滚动发布（Rolling Update）以及版本快速回滚（Rollback）。
Deployment 适合处理 HTTP 请求的无状态服务。
对于多节点并行的训练任务而言，它并不适用。
Deployment 倾向于单独重启发生故障的 Pod，这会破坏全局 NCCL 通信组。
其扩缩容策略不具备 gang 调度语义，会导致部分启动。
训练的各个 rank 具有唯一的 identity（如 Rank 0 必须做统筹）。
Deployment 创建的 Pod 在逻辑上是无差别且可互换的。

### Service
Service 提供虚拟入口 IP（ClusterIP）和内部 DNS 记录。
ClusterIP 模式提供集群内的四层负载均衡，适合对 Web API 进行流量分发。
Headless Service 通过声明 `clusterIP: None`，绕过负载均衡器，直接将 DNS 记录解析为真实的 Pod IP 列表。
在分布式训练启动时，Rank 0 的寻址阶段依赖这种 DNS 解析模式。
训练的 NCCL 梯度流量不应经过 Service 的负载均衡器，否则会成为性能瓶颈。

### Namespace
Namespace 划分资源名称的作用域，避免同名服务冲突。
它提供了资源配额（Quota）、RBAC 权限边界和 NetworkPolicy 网络微隔离。
它是构建多租户集群的基础。
Namespace 在操作系统层面并不提供 CPU 内存的进程隔离。
它仅仅是 API 层面的边界。

---

## 3 · 调度

kube-scheduler 的核心工作循环处理处于 Pending 状态的 Pod。

```text
watch 队列中 unscheduled 的新建 Pod
  -> Filter 阶段（评估候选节点，筛选出满足所有资源约束的节点）
  -> Score 阶段（对筛选出的候选节点执行打分算法，表达偏好）
  -> Bind 阶段（将得分最高的节点名称写入 Pod 对象中）
```

Kubernetes 默认的调度器逐个处理 Pod。
这带来了部分分配死锁风险。
例如，一个任务需要 4 张 GPU，3 个 Pod 进入 Running 状态并锁定了硬件，剩下的 1 个 Pod 卡在 Pending 状态。
这导致 3 张 GPU 被挂起，阻碍其他任务运行，同时无法执行计算。

### request、limit、QoS
在资源定义中，request 是提供给调度器用于全局记账的声明数值，代表理论可分配容量。
limit 是由 kubelet 结合底层的 cgroup 强制执行的物理资源消耗上限。
如果突破 limit，进程会遭遇 OOM Kill。

Kubernetes 将 Pod 划分为三种 QoS 级别：
- Guaranteed：要求 Pod 内每一个容器的 CPU 和内存 request 等于其 limit。
- Burstable：要求容器配置了 request，并且小于 limit。
- BestEffort：未配置 request 和 limit。系统内存紧张时会优先被驱逐。

GPU 在 Kubernetes 内部被归类为扩展资源（Extended Resource）。
对于所有的扩展资源，声明的 request 必须等于 limit。
占用 GPU 的训练 Worker 自动处于 Guaranteed 级别。

### 亲和与拓扑分布
调度器控制负载的物理落点：
- nodeSelector 提供节点标签精确匹配（如 `disk=ssd`）。
- nodeAffinity 和 podAffinity 允许声明节点级别或 Pod 之间的偏好和位置约束。
- taint 和 toleration 形成排斥关系。管理员给节点打上 taint 拒绝默认调度，只有声明了 toleration 的 Pod 才能分配上去。
- topologySpreadConstraints 用于要求副本在不同的可用区（zone）或机柜（rack）之间均匀分布。

拥有 GPU 的节点通常会被打上特殊的 taint。
目的是阻止纯 CPU 微服务占用计算节点。
训练任务要求 Worker 聚集在同一个机柜（rack）或 NVLink 交换域内，以最大化卡间数据传输带宽。
在线推理服务通常利用 topologySpreadConstraints 跨越多个 AZ 进行分布部署。

### Device Plugin
kubelet 依赖外部注册的 Device Plugin 来发现并分配 GPU 资源。

```text
部署在每个节点的 plugin 执行 ListAndWatch
  -> 扫描底层硬件并向 kubelet 汇报实际卡数
kubelet 将汇总后的信息，向中心 API server 汇报
  -> 报告节点 nvidia.com/gpu 的可用容量
Pod 在其 spec 定义中，声明需求
  -> 需要多少个 nvidia.com/gpu 资源
scheduler 在 Filter 阶段
  -> 过滤掉容量不足的节点
kubelet 接收调度结果
  -> 对目标节点上的插件执行 Allocate() 调用
容器 runtime 获取设备信息
  -> 将设备节点（如 /dev/nvidia0）挂载进容器
```

NVIDIA GPU Operator 包含了驱动模块、plugin 组件以及 exporter。
经典的 Device Plugin 模型将 GPU 暴露为离散的整数 slot。
新的 DRA（Dynamic Resource Allocation）框架提供了三维拓扑表达和切片共享分配机制。
目前大多数生产集群仍使用经典的 Device Plugin 模型。
在调度时，`gpuType`（如区分 A100 与 H100）属于硬性的 Filter 规则，而不是 Score 偏好。
针对 H100 优化的算子任务不能降级到 A100 节点上，否则会导致运行期错误。

### Gang 与抢占
Gang 调度的要求是：所有的并行 Worker 必须在同一时刻获得资源分配。
Volcano 通过 PodGroup 概念在调度层植入 gang 语义。
Kueue 在 Webhook 准入层提供同步放行保证。
任务控制器必须具备兜底机制。
发生部分分配时，控制器必须主动删除无法集齐的 Pod 集合，释放资源。

PriorityClass 定义任务的全局优先级。
Kubernetes 原生的抢占机制仅针对单个 Pod 执行。
训练场景需要的是 Job 级别的抢占。
驱逐低优先级 gang，为新的高优先级 gang 腾出连续的资源空缺。

### 队列
Namespace 提供的 resource quota 是资源总量的硬性上限。
它不构成排队系统。
Kueue 引入了 ClusterQueue、LocalQueue 和 ResourceFlavor。
它支持 FIFO、多级优先级、跨团队 fair share 共享，以及跨队列抢占。
在混合集群中，高优先级的在线推理服务可以在扩容时抢占低优先级离线训练任务的 GPU 资源。

### 拓扑标签
常用的拓扑标签包括 `topology.kubernetes.io/zone` 和 `rack`，以及自定义的 `network=rdma`。
硬约束（Hard Constraints）包括指定的 GPU 型号和物理网络条件（如 RDMA）。
软约束（Soft Constraints）表达偏好，如将任务的 Pod 安排在同一个机柜（rack）内。
如果调度器打破跨 rack 的软约束，Pod 仍可以进入 Running 状态。
但这会造成跨机柜的物理 all-reduce 通信延迟，严重拉慢训练的 step 计算时间。

```k8s-gang-visual
```

---

## 4 · 应用管理

### Job
Kubernetes 原生的 Job 控制器确保有指定数量（N）的 Pod 成功运行完毕（状态 0 退出）。
`parallelism` 字段控制并发度，不提供 gang 语义。
Indexed Job 为启动的 Pod 提供确定性的序号（index）。
在发生部分节点失败时，原生 Job 控制器只会重试发生故障的单个 Pod，无法感知整体通信环的崩溃。

### LLMJob / PyTorchJob
现代训练任务使用自定义资源 CRD（如 PyTorchJob）进行定义。
针对训练任务，失败策略配置为 RestartAll。
控制器的逻辑是：等待所有计算 Pod 达到 Running 状态后，获取 rank 0 的 IP 并注入 MASTER_ADDR。
在所有 Worker 到齐并通过网络 barrier 之前，进程不会开始矩阵计算。
这些 CRD 还会自动向容器内注入 RANK 序号和 WORLD_SIZE 环境变量。

### StatefulSet 与 DaemonSet
StatefulSet 提供重启期间稳定的网络主机名标识（如 pod-0, pod-1）和 PVC 存储卷持久绑定。
它适合部署 ZooKeeper、Kafka 等有状态组件。
对于 LLM 训练任务，实践是在节点故障时销毁并重建整个 Worker 组。
它们并不强求每个节点保留前一次运行的 identity 或本地盘数据。

DaemonSet 保证在所有匹配节点上恰好运行一个 Pod。
这用于部署基础设施组件，如 Fluent Bit 日志代理、DCGM exporter 和各类 Device Plugin。

### Operator
Operator 模式将 SRE 知识编码为可运行的软件逻辑。
它在集群内部监听 API server 中特定 CR 的配置变化。
它解析配置，创建基础 Pod 对象和内部 Service 记录。
它接管训练任务的生命周期，编排故障恢复流程。
它将 Pod 的执行进展聚合更新回 CR 的 status phase 字段中。

### 探针
Kubernetes 提供三种健康检查探针：

| 探针类型 | 失败时的执行动作 | 适用场景 |
|---|---|---|
| startupProbe | 不重启底层进程，不引入外部流量路由。 | 保护启动慢的应用，屏蔽后续探针的干扰。 |
| readinessProbe | 将 Pod 从 Service 的 Endpoints 摘除，不重启。 | 用于发布平滑上下线，表明不准备接收新外部流量。 |
| livenessProbe | 直接触发 kubelet 使用 SIGKILL 重启容器。 | 用于打破死锁，挽救假死的进程。 |

在大模型编译或大规模节点 NCCL 网络初始化期间，主线程可能会阻塞。
如果配置了 liveness 探针，它可能会将正常初始化的 rank 节点误判为死锁并击杀。
多节点训练的心跳监控和死锁检测属于高层 Job controller 的职责，不应交给 kubelet 探针执行。

---

## 5 · 持久化

存储后端的选型决定了模型保存的 I/O 吞吐和数据的全局可见性：

| Volume 类型 | 作用域与生命周期 | 典型场景 |
|---|---|---|
| emptyDir | 与挂载它的 Pod 生命周期绑定，销毁时数据清空。 | 临时数据缓存目录、共享内存 shm。 |
| hostPath | 映射节点宿主机的本地物理硬盘路径。 | 节点级缓存数据集。 |
| configMap | 将 etcd 内保存的小型文本对象挂载为配置文件。 | 超参数配置文件、注入启动脚本。 |
| PVC | 独立于 Pod 生命周期之外的高级存储声明。 | 保存 Checkpoint 权重数据、挂载外部训练语料数据集。 |

当使用 emptyDir 并将其 medium 配置为 Memory 时，它被挂载为 tmpfs 虚拟盘。
如果 sizeLimit 限制过小，NCCL 建立共享内存时会抛出 OOM 异常。

### PV 与 PVC

容器级别持久化存储的挂载关系：
```text
Pod (使用者) -> PVC (请求凭证) -> PV (实际资源块) -> 存储后端 (Lustre/Ceph 等物理集群)
```

PVC（PersistentVolumeClaim）是用户对存储空间的声明式请求。
PV（PersistentVolume）是底层集群预先分配或动态创建的存储卷实例。
StorageClass 是连接两者的抽象桥梁，定义了使用的 provisioner 插件并控制自动供给。

底层存储的并发承载能力由访问模式（Access Mode）决定：

| 访问模式 | 并发约束特性 |
|---|---|
| RWO (ReadWriteOnce) | 只能被单个节点挂载为读写模式。 |
| ROX (ReadOnlyMany) | 可以被多个节点挂载，只能执行读取操作。 |
| RWX (ReadWriteMany) | 允许多个节点同时挂载，并发执行读写操作。 |

大规模 LLM 训练的 Worker 进程需要向同一目录写入 checkpoint 文件。
这要求存储基座支持 RWX 挂载模式。
CSI（Container Storage Interface）标准化插件将这些需求对接到底层的 Lustre 集群、Ceph 文件系统或云盘上。

### Checkpoint 恢复路径

故障恢复流水线：
```text
检测到计算 Worker 发生故障
  → 控制器下发中止指令，停止剩余健康 Worker 的前向计算
  → 清理资源，重新申请并重建整个 Worker 组
  → 新 Worker 启动后，加载最后一次完整的 checkpoint
  → 对齐迭代步数，继续模型训练
```

在这个过程中需要追踪核心指标：
- 故障检测时间：从进程死锁到控制层感知该故障的延迟。
- 基础设施重建时间：重新排队、调度和拉取镜像的耗时。
- 丢失步数（Lost Steps）：最后一次保存 checkpoint 到系统崩溃前所计算出的作废轮次。
- 连续故障重试次数：防止在无限失败循环中空转。
- 数据损坏的文件数：监控因停电或强制中断导致的不完整写入。

为了防止写入中途崩溃导致旧数据污染，程序先将新权重写入临时文件。
确认落盘后，通过操作系统级的 rename 操作覆盖旧文件。
业界通常依赖指针文件（如 `latest`）记录最新且完整的检查点位置。
RPO（恢复点目标）对应了写入 checkpoint 的时间间隔。
RTO（恢复时间目标）包含了故障检测、排队、调度、容器冷启动以及权重加载的总耗时。
Kubernetes 负责重建物理 Pod 资源，但无法恢复未保存 checkpoint 而丢失的计算 step。

---

## 6 · 网络

在 Kubernetes 的扁平网络模型中，每一个 Pod 会被分配一个唯一的 IP 地址。
CNI（容器网络接口）插件负责 IP 分配、虚拟网卡配置和底层路由。
CoreDNS 监听 Service 生命周期变化，提供内部集群域名解析服务。
kube-proxy 维护用于 ClusterIP 负载分发的映射转发表。
NetworkPolicy 在内核层面控制不同命名空间和应用之间的通信边界。

在大规模深度学习场景中，梯度同步和 NCCL 通信必须绕过中间件。
直接使用原始的 Pod IP 或底层 RDMA endpoint 进行裸金属连接。
如果数据流向了 ClusterIP 并经过其代理，负载均衡逻辑会将连接分散到不同的后端。
这会破坏 NCCL 内部通信同步组。

Ingress 资源对象和 Gateway API 控制器管理南北向（进出集群）的 HTTP/HTTPS 业务流量。
它们处理七层请求代理，不应出现在 all-reduce 梯度通信路径上。

集群内部东西向（节点之间）数据流：
- 物理节点内部的 GPU 通信依赖专用的 NVLink 进行点对点互联。
- 跨节点的 GPU 通信依赖 RDMA 专线网络（如 RoCEv2 或 InfiniBand）。
- Kubernetes 控制面 API 交互和日志收集使用以太网连接。

调度器必须利用 Filter 规则将依赖 RDMA 的训练任务分配到配置了 RDMA 网卡的节点上。
将高速网卡作为外部设备直接挂载给特定的训练 Pod 使用。
这在隔离性和吞吐性能上优于开启 `hostNetwork: true`。
跨越可用区（AZ）或机柜（rack）的光纤延迟会成为传输瓶颈，拖慢单次 step time。

---

## 7 · 可观测性

系统监控自上而下分层次展开：

| 观测级别 | 核心观测指标对象 | 监控目的 |
|---|---|---|
| 基础设施与集群 | 处于 Pending 的 Pod 数量、节点的 ready 状态比例、kubelet 和 runtime 错误率。 | 确保物理集群具备基础容纳能力与算力底座。 |
| 调度控制面 | Kueue 准入层的排队延迟、gang 调度的等待情况、故障恢复耗时。 | 评估调度分配的效率和公平性。 |
| 上层业务与任务 | 单次 step time、模型 loss 值收敛曲线、NCCL 同步耗时占比、checkpoint 写入耗时与触发频率。 | 将基础设施性能对齐到模型质量和算力利用率。 |

Prometheus 采集并汇聚时间序列指标数据。
NVIDIA 的 DCGM exporter 采集显卡温度、功耗、以及计算利用率等硬件指标。
日志采集方案中，标准输出日志必须打上 job、rank 序号以及 step 轮次字段标签，以排查死锁问题。
KWOK 工具在单台机器上模拟虚拟节点，执行控制面压力极限测试。

---

## 8 · 分层架构

Kubernetes 对象模型和调度逻辑解决微观问题：决定 Pod 放置在具体哪一台机器上。
大规模分布式系统需要根据职责边界和拓扑物理位置进行宏观切分。
这两条轴线是正交的，需独立规划。

```k8s-layered-arch-visual
```

### 职责分层
根据“某一个组件在系统内被允许做什么”划分垂直逻辑架构：
边缘接入层负责外部鉴权、限流和网关路由。
计算逻辑层负责业务编排和发起写指令。
数据持久层保存记录、维护副本数据一致性。
旁路异步处理层负责削峰缓冲、后台重试和保存 checkpoint。

Kubernetes 遵循了这一架构设计：
- apiserver 和 etcd 扮演数据层，存储期望与观测状态。
- Kueue 和 scheduler 扮演管理层，负责准入与放置决策。
- kubelet 和容器 runtime 负责物理节点的执行。
- 跑在容器里的 Pod、GPU 以及 NCCL 库构成数据面。
Token 生成流水不经过控制面的 apiserver。

系统的水平扩展在同一个责任层级内进行。
跨越层级的通信通过定义的契约接口，如 RESTful HTTP、PVC 挂载或 CRD 对象。

### 拓扑分层
按照数据中心物理位置划分物理架构：
地理 Region -> 可用区 AZ -> 机柜 rack -> 宿主机 node -> 容器 Pod -> GPU/NIC。
Kubernetes 控制面依赖 Node 的元数据 label 识别拓扑层次。

| 系统工作负载 | 最优拓扑放置策略 |
|---|---|
| 无状态 API 接口 | 强制跨 AZ 均匀部署，确保多可用区高可用。 |
| 高性能计算 gang | 聚集在同一机柜 rack 内。分配相同 GPU 型号和 RDMA 网络。 |
| Kubernetes 控制面 | 部署在独立的管理节点上，使用千兆以太网。 |
| Checkpoint 存储 | 使用独立的 RWX 存储集群，作为独立故障域。 |

### 常见混用与错误
常见的架构谬误是将物理机房 zone 误认为是软件职责层。
在深度学习集群部署中，如果将一个训练 gang 打散部署在不同的 AZ。
底层通信会受到跨区光纤延迟的影响，使算力闲置等待网络包。

标准的训练任务运行路径：
在职责轴上：Kueue 负责准入 → scheduler 决策调度位置 → kubelet 拉起进程 → NCCL 接管通信。
在拓扑轴上：4 个物理 rank 部署在同一个机柜；checkpoint 持久化写入远端 RWX 存储；apiserver 运行在专属控制面节点上。

---

## 9 · 配套实验

作为容器编排底座，Kubernetes 本身的职责是管理分散的 Pod。
本教学实验中的训练外围控制面组件，用于管理严格同步的 Worker 组。
代码在 `project/LLMTrainLab/`。逐步手打路径写在该目录的 README：landscape recipe → frontier 集群 → LoRA / 70B full / vLLM → gang 排队 → 杀 rank 恢复。

```bash
cd project/LLMTrainLab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
llmctl tutorial print          # 只打印命令
# 逐步敲 README 的 Step 1–9，或一条跑完：
llmctl tutorial run
```

`tutorial run` 走完后应看到：LoRA / 70B full / vLLM 在跑，MoE RLHF 和 PCIe 上的 TP>1 停在 Queued，LoRA 被杀掉一个 rank 后 `retries>=1`。

旧的 8×H100 队列 / 抢占故事仍可用：

```bash
llmctl demo canonical --preempt --virtual-nodes 40
```

---

## 10 · 一手资料

- [Kubernetes 架构与核心组件详尽解析](https://kubernetes.io/docs/concepts/overview/components/)
- [Pod 完整生命周期与各相位流转定义](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/)
- [Scheduling Framework 调度框架底层原理](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/)
- [Device Plugin 设备插件机制与设计模式](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
- [Jobs 任务控制器的适用范围与核心设计思想](https://kubernetes.io/docs/concepts/workloads/controllers/job/)
- [Persistent Volumes 持久化存储声明全貌](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
- [Kueue 高级作业队列准入管理系统概览](https://kueue.sigs.k8s.io/docs/overview/)
- [KWOK 零消耗海量伪节点规模测试工具](https://kwok.sigs.k8s.io/)
- [ByteDance 生产集群: Robust LLM Training Infrastructure 论文](https://www.alphaxiv.org/abs/2509.16293)
- [LLMTrainLab 训练控制面模拟实验台源码](https://github.com/CurryTang/LLMTrainLab) (位于本仓库本地 `project/LLMTrainLab/` 目录中)

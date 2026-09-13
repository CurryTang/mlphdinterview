# LLMTrainLab

训练 / post-training 平台的本地控制面。LLM 权重不加载：Worker 是假的，Kubernetes 的节点 / Device Plugin / kubelet 也是假的。要练的是 **Job 语义、GPU 与网络怎么约束调度、以及故障后 Worker group 怎么活**。

面向的场景是 infra 里的 training / post-training platform：开放模型家族、LoRA / QLoRA / full FT / RLHF、vLLM / SGLang / TensorRT-LLM、H100–B200 与 NVLink / RDMA、DP/TP/PP/EP + NCCL、Kueue 风格准入、gang、GPU operator 账本、Prometheus / DCGM 形指标。控制面是 Python（`llmctl` + 可选 FastAPI）；CRD / Helm / GitOps 清单在 `charts/` 和 `examples/gitops/`。

> Kubernetes 管 Pod 的生命；训练控制面管 Worker group 的生命。推理副本用 KEDA 扩，不要用 Deployment 单独重启一个 TP rank。

配套笔记：上一级 `notes/SystemDesign/SystemDesign01C Kubernetes.md`。独立仓库：[CurryTang/LLMTrainLab](https://github.com/CurryTang/LLMTrainLab)。

```text
landscape recipe (SKU × model × method × engine × fabric)
   ↓
LLMJob CRD
   ↓
LLMJob Controller  (Python, optional FastAPI /metrics)
   ↓
Queue / Admission / Gang  (Kueue-shaped)
   ↓
Kubernetes-shaped Pods + Device Plugin slots
   ↓
Fake workers (NCCL collectives billed as extra ticks)
   ↓
Checkpoint store + Prometheus text + DCGM-shaped gauges
```

不需要 GPU，不需要 kind。压测用 `llmctl scale`（KWOK 风格虚拟节点）。`llmctl kubectl get nodes|pods` 是同一套状态的 kubectl 口令。

## JD 对照（每条都有可跑的东西）

| JD | 在本仓库里 |
|---|---|
| 开放模型、LoRA/QLoRA/full/RLHF、vLLM/SGLang/TRT-LLM | `llmctl landscape models\|methods\|engines`；`examples/jobs/llama8-lora.yaml` 等 |
| H100 / H200 / B200、NVLink、内存层次 | `llmctl landscape gpus`；`examples/cluster-frontier.yaml` |
| DP/TP/PP/EP、NCCL、多机调度 | `llmctl landscape parallelism`；TP>1 必须 NVLink，EP 必须 RDMA |
| 前沿 → 产品决策 | `llmctl landscape recipe --model llama-70b --method full --gpu H200` |
| Helm / CRD / operator / gang / GPU operator | `charts/llmjob/`；`try_place` 原子占坑 |
| kubectl / Pod 生命周期 / 节点 / 网络 | `llmctl kubectl get …`；`llmctl fault pod\|node\|network` |
| GCP / GitOps / Terraform 心态 | `examples/gitops/kustomization.yaml` 是源；集群规格是 YAML |
| Prometheus / Grafana / Loki / OTel / DCGM | `llmctl serve` → `/` dashboard + `/metrics`；`metrics()["dcgm"]` |
| Linux netns / 性能 | 节点 `network` + `interconnect`；fabric_cost 变成 step 耗时 |
| FastAPI 控制面 | `llmtrainlab/api.py`，`pip install -e '.[api]'` |
| 产品面 | `llmctl serve` 打开 `http://127.0.0.1:8080/` |

---

## 1. 安装（5 分钟）

需要 Python 3.10+。建议用虚拟环境（macOS Homebrew Python 会拒绝往系统里装包）：

```bash
cd project/LLMTrainLab          # 若你 clone 的是独立仓库，则 cd LLMTrainLab
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
pytest -q
llmctl tutorial print           # 只打印下面要敲的命令，不改状态
```

状态文件写在当前目录的 `.llmtrainlab/state.json`。命令要在**同一工作目录**里敲，才能串起来。`cluster init` 会覆盖这份状态。

不想逐步敲、只想先看终态：

```bash
llmctl tutorial run
llmctl job list
llmctl kubectl get pods
```

下面 Step 1–9 是同一条路径的手打版。每一步都写出**你该看到什么**。H200 相对 H100 主要是 **141 GB HBM**，NVLink 代数相同；B200 才换 NVLink5 / FP4。Recipe 默认把 Job **塞进一张 NVLink 岛（≤8 GPU）**，这样笔记本上的 frontier 集群就能 admit，而不是一上来要 16 卡。

---

## 2. Tutorial（逐步实操）

### Step 1 — 先看 landscape，再调度

调度器吃的是 Filter，不是「这张卡很快」。先问 recipe：这个模型、这种方法、这块 SKU，要几张卡、要不要 NVLink、要不要 RDMA。

```bash
llmctl landscape gpus
llmctl landscape parallelism
llmctl landscape recipe --model llama-70b --method full --gpu H100
llmctl landscape recipe --model llama-70b --method full --gpu H200
llmctl landscape recipe --model llama-8b --method lora
llmctl landscape recipe --model deepseek-v3 --method rlhf
```

70B full × H100 应是 **TP=8、workers=8、preferSameNode**，每 rank 大约 44 GB，塞进 80 GB。同一 Job 换 H200 变成 **TP=4、workers=4**——HBM 够了，少切一刀。LoRA 8B 落在 L40S、TP=1。DeepSeek RLHF 带 **EP≥2 且 requireRdma**。

`recipe` 打印 workers、TP、是否 `requireRdma`、每 rank 大概要多少 GB。提交时带 `spec.model` + `spec.method`，controller 会填这些字段。

### Step 2 — 初始化 frontier 集群

```bash
llmctl cluster init --config examples/cluster-frontier.yaml
llmctl kubectl get nodes
```

期望五类节点都是 `Ready`：

```text
NAME        STATUS  GPU       NET        XCONN
node-a      Ready   8xA100    ethernet   nvlink
node-h100   Ready   8xH100    rdma       nvlink
node-h200   Ready   8xH200    rdma       nvlink
node-b200   Ready   8xB200    rdma       nvlink
node-l40s   Ready   4xL40S    ethernet   pcie
```

默认 `examples/cluster.yaml` 只有 A100/H100/L40S，给后面的 gang / 抢占 demo 用。这条 tutorial 用 frontier，否则 70B / H200 recipe 没有对应 SKU。

### Step 3 — LoRA 8B 落到 L40S

```bash
llmctl job submit examples/jobs/llama8-lora.yaml
llmctl tick --steps 8
llmctl job list
```

期望：

```text
name         phase    method  engine  gpus  type  tp
llama8-lora  Running  lora    fsdp    1     L40S  1
```

`submit` 会打印 `why:`：adapter FT 不需要 NVLink。一张 PCIe 卡就够。YAML 里没有写 `workers` / `gpuType`，是 recipe 填的。

### Step 4 — 70B full FT，TP 留在一块 H200 里

```bash
llmctl job submit examples/jobs/llama70-full.yaml
llmctl tick --steps 6
llmctl job list
```

期望 `llama70-full` **Running**，`type=H200`，`gpus=4`，`tp=4`。`llmctl job status llama70-full` 里四个 worker 应在 **同一个 `node-h200`**（`preferSameNode`）。

想看「H100 上 TP=8 填满一块 HGX」：

```bash
llmctl job submit examples/jobs/llama70-full-h100.yaml
llmctl tick --steps 4
```

`node-h100` 的 8 张卡会被占满。这条是可选的；不提交也不影响后面的 MoE 排队（MoE 要的是 H200）。

### Step 5 — vLLM 副本占剩余 H200

```bash
llmctl job submit examples/jobs/llama70-vllm.yaml
llmctl tick --steps 4
llmctl kubectl get pods
```

推理 Job 的 qos 是 `inference`。它仍是一组 TP rank（70B 在 H200 上 TP=2），**不是** Deployment 里互不相关的 Pod。KEDA 扩的是整份 replica，kube-scheduler 不能单独重启其中一个 rank。

`kubectl get pods` 里 LoRA 1 个 rank、full FT 4 个 rank、vLLM 2 个 rank，各自绑在不同类型的节点上。

此刻 H200 大约 **6/8 已用**（4 训练 + 2 推理）。

### Step 6 — Gang：MoE 要 8 张 H200，剩 2 张也不给

```bash
llmctl job submit examples/jobs/deepseek-rlhf.yaml
llmctl tick --steps 3
llmctl job list
```

期望 `ds-rlhf` **Queued**，`ep≥2`，H200 used 仍是 6。这是分布式训练最贵的调度错误的反例：3 张 Running、第 4 张 Pending 会让那 3 张卡既没有 step，又堵住别人。`try_place` 一次拿不到全部 GPU 就返回 `None`。

`ds-rlhf` 还带 `requireRdma`。即使你把它丢到 ethernet 节点上，一样 Queued——AllToAll 没有 RDMA 就不要占着卡空转。

### Step 7 — Filter：PCIe 上的 TP>1 不是「会跑得很慢」

```bash
llmctl job submit examples/jobs/bad-tp-l40s.yaml
llmctl tick --steps 2
llmctl job list
```

期望 `bad-tp-l40s` **Queued**，L40S used 仍是 1（只有 LoRA）。L40S 没有 NVLink domain。TP>1 在这里是产品错误，调度器直接拒绝，而不是放上去让 MFU 掉到个位数。

### Step 8 — 杀掉一个 rank：RestartAll + checkpoint

```bash
llmctl tick --steps 20
llmctl fault pod llama8-lora-worker-0
llmctl job status llama8-lora
llmctl tick --steps 8
llmctl job list
```

期望路径：

```text
检测失败
  → 停止剩余 Worker（拆 communicator）
  → 重新创建整个 Worker group
  → 加载最近完整 checkpoint
  → 继续训练
```

`llama8-lora` 的 `retries >= 1`，`lost_steps` ≤ `checkpointEvery`（20）。原生 Kubernetes **不会**做这件事：Deployment 会单独重启那个 Pod，NCCL communicator 已经死了，其余 Worker 空转占着卡。

### Step 9 — 指标，以及可选的控制面 HTTP

```bash
llmctl metrics
llmctl kubectl get jobs
```

`training_throughput` 是假 Worker 的 step/tick。NCCL 在本模拟里不是真的库：TP 跨 NVLink、EP 跨 ethernet、DP 跨 ethernet 会增加 `fabric_cost`，每个 fake step 多走几个 tick。调度成功 ≠ 训练快。

可选：

```bash
python -m pip install -e ".[api]"
llmctl serve
```

浏览器打开 `http://127.0.0.1:8080/` 看 job / DCGM 表；`/metrics` 是 Prometheus text；`/v1/cluster` 是同一份 JSON。`serve` 读的是磁盘上的 state，另开一个终端继续 `llmctl tick` 就能看到数字动。

### 做完你刚走完的语义

| 你敲的 | 对应真实世界 |
|---|---|
| `landscape recipe` | SKU × 模型 × 方法 → Filter（HBM / NVLink / RDMA） |
| `try_place` 一次放下全部 Worker | Volcano PodGroup / scheduler coscheduling |
| `ds-rlhf` 在 2 张空闲卡上 Queued | 部分分配死锁的反面 |
| `bad-tp-l40s` Queued | Device Plugin + 拓扑标签，不是 Score |
| `fault pod` 后 RestartAll | Job 失败策略 + 共享存储上的 ckpt |
| `llmctl serve` `/metrics` | Prometheus + DCGM exporter 的形状 |

对照工业系统：节点 `network=rdma` / `gpu_type=H200` 是 Device Plugin + 拓扑标签；`scale` 出 `kwok-*` 节点是 [KWOK](https://kwok.sigs.k8s.io/)。Kueue 官方定位就是 AI/ML 批任务的队列、配额、优先级和共享。[Kueue 概览](https://kueue.sigs.k8s.io/docs/overview/)。GPU 在 Kubernetes 里通常由 Device Plugin 暴露给 kubelet。[Device Plugin 文档](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)。

真实集群里，一次 4 Worker 训练会经过：

```text
提交 LLMJob
  → 队列 / 配额 / 优先级（Kueue 这一层）
  → Gang：4 张 GPU 同时可分配才 admit
  → 创建 Pod，scheduler bind，Device Plugin Allocate
  → RANK / WORLD_SIZE / MASTER_ADDR
  → rendezvous barrier
  → step + heartbeat + checkpoint
  → 有人挂了：停整组 → 重建 → load ckpt
```

所以本项目的核心对象是 `LLMJob`，不是 `Deployment`。

---

## 3. 附录 · 默认集群上的队列、抢占、KWOK

Tutorial 用完 frontier 之后，如果还想看「6+2 填满 8 张 H100，第 3 个 4 卡 Job 排队 / 被抢」，换回默认异构集群（会清掉当前 state）：

```text
node-a: 4 x A100   rack-1  ethernet  nvlink
node-b: 8 x H100   rack-1  rdma      nvlink
node-c: 2 x L40S   rack-2  ethernet  pcie

team-a  训练   guaranteed 6  burst 10  优先级 80
team-b  实验   guaranteed 2  burst 4   优先级 20
team-c  推理   guaranteed 4  burst 8   优先级 100
```

```bash
llmctl cluster init --config examples/cluster.yaml
llmctl cluster status
llmctl job submit examples/jobs/job-a.yaml    # 6 x H100，低优先级
llmctl tick --steps 8
llmctl job submit examples/jobs/job-b.yaml    # 2 x H100，高优先级，占满剩余
llmctl tick --steps 4
llmctl job submit examples/jobs/job-c.yaml    # 4 x H100 → Queued
llmctl tick --steps 3
llmctl job list
```

期望：A Running，B Running，C **Queued**。H100 used=8。C 的 priority 低于 A，所以它不会去抢 A。若把 C 改成更高优先级再 `cluster init`，会看到 **整组抢占**。

硬抢占 + 故障 + 虚拟节点，一条命令：

```bash
llmctl demo canonical --preempt --virtual-nodes 40
```

只打印命令：`llmctl tutorial print`（landscape 路径）或看上面附录。

### Checkpoint 与故障注入

```bash
llmctl tick --steps 120
llmctl fault pod llama-a-worker-2
llmctl fault node node-b
llmctl recover-node node-b
llmctl fault network --job llama-a
llmctl fault ckpt --duration 30
llmctl fault gpu node-b --remaining 2
llmctl metrics
```

| 字段 | 含义 |
|---|---|
| `retries` | 整组重启次数 |
| `lost_steps` | 上次完整 ckpt 之后丢掉的 step |
| `recovery_p50/p95` | 检测 + 再调度的 tick 数 |
| `gpu_slot_util` | GPU slot 使用率 |
| `queue_wait_p50 / p95` | 队列等待 |
| `training_throughput` | 假 Worker 的 step/tick |

`lost_steps` 应 ≤ `checkpointEvery`。Quota：team-b 的 `burst_gpus=4`，超过 burst 的 Job 会一直 Queued，即使集群空着。这和 Kueue 的「quota 先于 kube-scheduler」是同一层。

---

## 4. 实现顺序（和代码怎么对上）

```text
landscape  SKU / 方法 / 并行轴          landscape.py recommend / apply_recipe
Phase 1    LLMJob CRD + Controller     engine.Job / submit_job / step
Phase 2    Fake GPU + Gang Admission   try_place：不够就返回 None
Phase 3    Worker + Checkpoint         Bound→Running→Training，checkpoint 列表
Phase 4    队列 / 优先级 / Quota / 抢占  _fair_key + _preempt
Phase 5    拓扑 + 故障注入             gpu_type / rdma / nvlink filter，llmctl fault
Phase 6    KWOK 压测 + 指标            scale_virtual_nodes + metrics() + serve
```

核心不变量（测试锁住的）：

1. **没有部分分配。** 4 Worker 的 Job 在 3 张卡上 used 必须是 0。
2. **抢占是整组的。** 高优先级要 8 张时，低优先级 6 Worker 一起变成 Preempted，不会留下 2 个孤儿 rank。
3. **恢复读完整 checkpoint。** `lost_steps <= checkpointEvery`。
4. **类型和网络是 Filter。** H100 不会出现在 A100 节点；`requireRdma` 不会出现在 ethernet；TP>1 不会出现在 PCIe。
5. **同 rack / 同节点是 Score。** 容量够时 TP 组不会跨 NVLink 岛。
6. **Tutorial 路径。** LoRA/70B/vLLM 能 Running，MoE 和 PCIe-TP 保持 Queued，杀 rank 后 retries≥1。

---

## 5. 和 Kueue 的差别

| | LLMTrainLab | Kueue |
|---|---|---|
| 作用层 | 教学用、进程内 | 生产、watch Kubernetes API |
| 队列 | 内存里按 priority / fair / FIFO 排序 | ClusterQueue + LocalQueue |
| 配额 | team guaranteed / burst | ResourceFlavor + borrowed quota |
| Gang | `try_place` 原子占坑 | 需 AllOrNothing 或底层 Volcano/coscheduling |
| 抢占 | Job 级，不够就继续抢下一个低优先级 | 支持，策略可配 |
| 训练语义 | 自带 rank / barrier / ckpt / RestartAll | 不管 PyTorch 生命周期，那是 Training Operator 的事 |

Kueue 解决「这个 workload 现在能不能进集群」；Training Operator / 本项目的 controller 解决「进了之后 Worker group 怎么活、怎么死、怎么恢复」。两者叠在一起才是训练平台。

---

## 6. 读代码的顺序

1. `llmtrainlab/landscape.py` — SKU / 方法 / 并行轴，recipe 怎么变成 Filter
2. `examples/jobs/*.yaml` — CRD 长什么样
3. `llmtrainlab/tutorial.py` — README 这条路径的可测试版本
4. `llmtrainlab/engine.py` — `try_place`、`_preempt`、`_detect_failures`、`_advance_workers`
5. `llmtrainlab/cli.py` — `llmctl landscape` / `kubectl` / `fault` / `tutorial`
6. `llmtrainlab/api.py` — FastAPI `/` `/v1/cluster` `/metrics`
7. `tests/` — 不变量

不要从 CLI 的 argparse 开始。先看 `try_place` 为什么返回 `None` 而不是 3 张卡。

---

## 7. 你做完应该能回答的问题

1. 为什么训练 Job 不能用 Deployment？
2. request 和 Device Plugin 各管哪一层账本？
3. 部分分配死锁是怎么形成的，gang 在 Filter 之前还是之后？
4. Kueue admit 成功之后，kube-scheduler 仍可能把 Pod 拆开放到不同 rack。缺哪一层？
5. 一个 rank 的 livenessProbe 失败，kubelet 重启它，为什么通常是错的？
6. RPO/RTO 分别由 checkpoint 间隔和「检测 + 排队 + 调度 + 拉镜像 + load」决定。Kubernetes 「自动重启 Pod」覆盖了其中哪一段？
7. 跨 rack 的 Job 为什么可能「调度成功、业务失败」？
8. 为什么 H200 上的 70B full 可以比 H100 少用 TP，而 NVLink 代数并没有变？

答得上来，这个项目就没有白做。笔记正文在 `notes/SystemDesign/SystemDesign01C Kubernetes.md`。

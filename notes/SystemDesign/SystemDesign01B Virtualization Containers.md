# System Design 01B · 虚拟化与容器

课程位置：[[SystemDesign01 Stateless Service|01 无状态服务]] → 本篇 → [[SystemDesign02 Database Paradigms|02 数据库基本范式]]

先记住最重要的区别：VM 虚拟出一台机器，container 隔离一组进程。VM 里的 guest 有自己的 kernel；container 仍在使用 host kernel。

```virtualization-container-visual
```

这条边界解释了两者大部分差异。VM 启动重、隔离强，可以运行不同 kernel；container 启动快、密度高，但一次 host kernel 漏洞可能影响同机多个租户。

---

## 1 · 虚拟化到底在虚拟什么

常见手段可以放在一张表里：

| 手段 | 给上层看到什么 | 典型实现 | 代价与用途 |
|---|---|---|---|
| Emulation | 一套完整的虚拟硬件，CPU 架构也可不同 | QEMU TCG | 最灵活，指令翻译慢；适合跨架构和调试 |
| Hardware-assisted VM | vCPU、虚拟内存和虚拟设备 | KVM、Hyper-V、Xen HVM | 云主机的常见基础 |
| Paravirtualization | guest 知道部分设备是虚拟的 | virtio、Xen PV | 少模拟硬件，I/O 更快 |
| OS-level isolation | 同一 kernel 上的隔离进程 | Linux containers | 启动快、密度高，不提供独立 kernel |
| MicroVM | 精简设备模型的硬件辅助 VM | Firecracker | 在 VM 隔离和 container 启动速度之间取折中 |

Emulation 和 virtualization 经常被混着说。QEMU 用 TCG 翻译指令时是在模拟 CPU；QEMU 配合 KVM 时，大部分 guest 指令直接在真实 CPU 上执行，QEMU 主要提供设备模型和 VM 管理。

### Type 1 和 Type 2 hypervisor

教科书常分两类：

```text
Type 1: hypervisor 直接控制硬件
Type 2: hypervisor 作为 host OS 上的程序运行
```

现实没有这么整齐。KVM 是 Linux kernel 的虚拟化能力，QEMU 是 user-space VMM；两者合起来既利用通用 host kernel，又能提供接近 Type 1 的执行路径。面试里解释组件职责，比争论它属于哪一类更有用。

---

## 2 · VM 为什么能接近原生速度

### CPU：guest 直接跑，敏感操作退出

Intel VT-x、AMD-V 等硬件扩展给 guest 提供受控执行模式。普通指令直接运行；guest 执行需要 hypervisor 处理的操作时发生 VM exit，VMM 处理后再恢复 guest。

```text
guest user code
    -> mostly runs on physical CPU

privileged / configured event
    -> VM exit
    -> hypervisor handles it
    -> VM entry back to guest
```

Exit 很贵，所以 hypervisor 会尽量减少它。vCPU 本质上仍由 host scheduler 调度；给 VM 配 8 个 vCPU，不等于它永远独占 8 个物理 core。

### Memory：两级地址翻译

Guest page table 把 guest virtual address 映射到 guest physical address；硬件的 EPT/NPT 再把 guest physical address 映射到 host physical address。

```text
guest virtual
  -> guest physical
  -> host physical
```

Hypervisor 决定 VM 能访问哪些物理页。内存超卖、ballooning 和 live migration 都建立在这层控制上，不过超卖会把性能问题变得很难看，尤其是延迟敏感服务。

### I/O：模拟、半虚拟化或直通

I/O 有三条常见路径：

1. 模拟真实设备。兼容性好，trap 多。
2. Guest 使用 virtio driver，通过共享内存 virtqueue 和 host 交换 buffer。
3. 用 IOMMU + VFIO / SR-IOV 把设备或 virtual function 交给 VM，减少中间层。

性能越接近硬件，迁移、共享和设备管理通常越麻烦。Cloud provider 会按实例类型选择不同组合。

---

## 3 · 云上的 hypervisor 做了什么

用户创建一台 VM 时，control plane 先选 host、分配 vCPU 和内存，再准备虚拟网络、磁盘和启动镜像。Host 上的 hypervisor/VMM 负责执行与隔离，网络和存储数据面可能由独立硬件或 host service 处理。

AWS Nitro 是一个很好的例子。AWS 的文档说明 Nitro Hypervisor 基于 KVM，网络、存储和管理功能被下沉到 Nitro Cards，host CPU 因而少做设备模拟。这里的设计思路比产品名更值得记：

```text
small hypervisor
+ hardware-assisted CPU / memory isolation
+ offloaded network and storage data plane
```

Google Compute Engine 的公开文档也说明其 VM 运行在经过安全加固的 KVM-based hypervisor 上。KVM 是常见基础，但每家云的控制面、设备模型、迁移和安全实现并不相同。

Firecracker 则把 VMM 和设备模型砍到很小，只保留 serverless / container workload 需要的部分。它仍使用 KVM，因此 microVM 仍有独立 guest kernel，只是启动和内存开销比通用 VM 小。

---

## 4 · Container 是怎样拼出来的

Container 不是某一个 kernel object。它是几组 Linux 机制加上镜像、runtime 和网络配置后的产品形态。

### chroot：只换 `/`

`chroot(path)` 改变进程解析绝对路径时使用的根目录。它没有隔离 PID、网络、用户和资源，也不会关闭指向外部的 file descriptor。Linux man page 明确说它不应被当作完整安全沙箱。

```text
before: / -> host filesystem
after:  / -> selected directory tree
```

Container root filesystem 需要 mount namespace、`pivot_root`、只读层和权限限制共同完成。把 container 解释成“高级 chroot”漏掉了大半。

### Namespace：限制进程能看到什么

Linux namespace 把原本全局的资源包装成每组进程各自的视图：

| Namespace | 隔离的视图 |
|---|---|
| PID | process ID tree |
| Mount | mount points 和 filesystem view |
| Network | interface、route、port、socket stack |
| UTS | hostname |
| IPC | shared memory、message queue 等 IPC |
| User | UID/GID 与 capability 映射 |
| Cgroup | cgroup hierarchy 的可见根 |

Namespace 回答“看得见谁”。它不保证某个 container 只能用 2 GB 内存，也不自动阻止所有危险 syscall。

### cgroup：限制进程能用多少

cgroup 把进程组织成层级，并由 controller 做资源统计和控制：

```text
cpu.max       CPU quota
memory.max    memory hard limit
io.max        block I/O limit
pids.max      process count limit
```

它回答“能用多少”。没有 cgroup，一个 container 的进程仍可能吃光整台 host 的内存或 CPU。

### 还缺安全限制

共享 kernel 意味着 syscall 就是攻击面，因此 production container 通常还需要：

- Linux capabilities：把 root 权限拆小，默认删掉不需要的 capability；
- seccomp：允许或拒绝 syscall；
- AppArmor / SELinux：限制进程能访问的对象；
- user namespace / rootless：把 container 内的 root 映射为 host 上的普通 UID；
- read-only filesystem、禁止 privileged mode、少做 hostPath mount。

Namespace 是可见性边界，不等于完整安全边界。

---

## 5 · Docker 在这套机制上加了什么

Docker 主要解决镜像构建、分发和运行体验。一个简化的数据路径是：

```text
Dockerfile
  -> immutable image layers
  -> registry
  -> dockerd / containerd
  -> OCI runtime such as runc
  -> namespaces + cgroups + mount + security policy
  -> application process
```

Image 是只读 layer 加 metadata；运行时再叠一层可写层。多个 container 可以共享只读 layer，所以启动和分发比复制整块 VM disk 便宜。

Docker 不等于 container，container 也不等于 Docker。Kubernetes 常通过 CRI 对接 containerd 或其他 runtime；最底层仍要创建 Linux 进程并配置隔离机制。

### `docker run` 概念上做了什么

```text
pull / locate image
prepare root filesystem
create namespaces
attach cgroups
configure veth, bridge and routes
apply capabilities, seccomp and LSM profile
exec the container entrypoint
```

Container 里看到 PID 1，但 host 上只是一个普通进程。也正因为如此，`kill`、signal handling、zombie reaping 和 graceful shutdown 仍是应用需要处理的问题。

---

## 6 · System design 里容器有什么用

### 统一 deployment artifact

Image 把应用、runtime 和依赖版本固定在一起。Dev、CI 和 production 运行同一个 digest，减少“机器上碰巧装了另一个版本”的问题。

### 让调度器可以安全地搬工作负载

有了 CPU/memory request、limit、health check 和标准启动方式，scheduler 才能做 bin packing、重启、滚动发布和 autoscaling。Container 是交付单元，Kubernetes 一类系统才是调度与编排层。

### 控制 noisy neighbor

cgroup 给多 workload 共机提供最低限度的资源边界。Request 用于调度容量，limit 用于约束使用；两者配错会导致 CPU throttling、OOM kill 或 host 利用率过低。

### 加快发布和回滚

Immutable image 配合版本化部署，可以先启动新 replica，通过 readiness 后再接流量。回滚是切回旧 digest，不是在机器上反向执行一串安装脚本。

### 容器解决不了什么

- 它不让有状态程序自动变无状态；
- writable layer 不是 durable storage；
- 它不替代 replication、backup 和 disaster recovery；
- 共享 kernel 也不是 hostile multi-tenant 的最强隔离。

数据库、queue 和 object storage 仍应保存 source of truth。Pod 被删后能否安全重建，是更实用的检查问题。

---

## 7 · VM、container、microVM 怎么选

| 场景 | 合理起点 |
|---|---|
| 一般 API、worker、内部服务 | Container |
| 需要不同 OS / kernel、强租户隔离 | VM |
| 不可信短任务、serverless、sandbox | MicroVM，或 container + 专用 sandbox runtime |
| 需要完整硬件性能或特殊设备 | Bare metal / device passthrough |
| 开发者本机运行 Linux container | Linux 上直接容器；macOS/Windows 通常先有一层 Linux VM |

安全边界应跟租户信任程度匹配。自家几个微服务共享 kernel 通常可以接受；运行陌生用户上传的代码时，只靠默认 container 隔离就偏冒险。

---

## 8 · 面试记忆版

```text
VM
= virtual hardware + hypervisor/VMM + guest kernel

Container
= host process + namespaces + cgroups + rootfs + security policy

chroot
= change pathname root only

namespace
= what a process can see

cgroup
= how much resource it can use

Docker
= image/build/distribution/runtime UX around containers
```

System design 里最常见的落点：container 提供标准部署单元和资源边界；orchestrator 负责 placement、health、scaling 与 rollout；外部数据系统保存不可丢失状态。

---

## 一手资料

- [Linux namespaces overview](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [Linux `chroot(2)`](https://man7.org/linux/man-pages/man2/chroot.2.html)
- [Linux cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- [Linux KVM documentation](https://docs.kernel.org/virt/kvm/index.html)
- [Linux virtio documentation](https://docs.kernel.org/driver-api/virtio/virtio.html)
- [QEMU system emulation](https://www.qemu.org/docs/master/system/introduction.html)
- [Docker Engine security](https://docs.docker.com/engine/security/)
- [AWS Nitro System architecture](https://docs.aws.amazon.com/whitepapers/latest/security-design-of-aws-nitro-system/the-nitro-system-journey.html)
- [Firecracker microVM](https://firecracker-microvm.github.io/)

# System Design 01B · Virtualization and Containers

Course Location: [[SystemDesign01 Stateless Service|01 Stateless Service]] → This Section → [[SystemDesign02 Database Paradigms|02 Database Paradigms]]

Remember the most important distinction first: A VM virtualizes a machine, while a container isolates a set of processes. The guest inside a VM has its own kernel; a container continues to use the host kernel.

```virtualization-container-visual
```

This boundary explains most of the differences between the two. VMs have heavy startup times and strong isolation, and can run different kernels; containers have fast startup times and high density, but a single host kernel vulnerability could potentially affect multiple tenants on the same machine.

---

## 1 · What exactly is being virtualized?

Common methods can be summarized in a table:

| Method | What the upper layer sees | Typical Implementation | Cost and Use Case |
|---|---|---|---|
| Emulation | A complete virtual hardware set; CPU architecture can differ | QEMU TCG | Most flexible, slow instruction translation; suitable for cross-architecture and debugging |
| Hardware-assisted VM | vCPU, virtual memory, and virtual devices | KVM, Hyper-V, Xen HVM | Common foundation for cloud instances |
| Paravirtualization | Guest is aware that some devices are virtual | virtio, Xen PV | Less hardware simulation, faster I/O |
| OS-level isolation | Isolated processes on the same kernel | Linux containers | Fast startup, high density, no independent kernel |
| MicroVM | Hardware-assisted VM with a stripped-down device model | Firecracker | A trade-off between VM isolation and container startup speed |

Emulation and virtualization are often used interchangeably. When QEMU uses TCG to translate instructions, it is emulating a CPU; when QEMU is used with KVM, most guest instructions execute directly on the physical CPU, with QEMU primarily providing the device model and VM management.

### Type 1 and Type 2 hypervisors

Textbooks often categorize them into two types:

```text
Type 1: Hypervisor controls hardware directly
Type 2: Hypervisor runs as a program on the host OS
```

Reality is not that neat. KVM is the virtualization capability of the Linux kernel, while QEMU is a user-space VMM; together, they leverage the general-purpose host kernel while providing an execution path close to Type 1. In an interview, explaining component responsibilities is more useful than arguing about which category it belongs to.

---

## 2 · Why can VMs approach native speed?

### CPU: Guest runs directly, exits on sensitive operations

Hardware extensions like Intel VT-x and AMD-V provide a controlled execution mode for the guest. Ordinary instructions run directly; when the guest executes an operation requiring hypervisor intervention, a VM exit occurs, the VMM handles it, and then execution returns to the guest.

```text
guest user code
    -> mostly runs on physical CPU

privileged / configured event
    -> VM exit
    -> hypervisor handles it
    -> VM entry back to guest
```

Exits are expensive, so the hypervisor tries to minimize them. A vCPU is still essentially scheduled by the host scheduler; assigning 8 vCPUs to a VM does not mean it exclusively occupies 8 physical cores at all times.

### Memory: Two-level address translation

The guest page table maps guest virtual addresses to guest physical addresses; hardware EPT/NPT then maps guest physical addresses to host physical addresses.

```text
guest virtual
  -> guest physical
  -> host physical
```

The hypervisor determines which physical pages a VM can access. Memory overcommitment, ballooning, and live migration are all built upon this layer of control, though overcommitment can make performance issues look very ugly, especially for latency-sensitive services.

### I/O: Emulation, paravirtualization, or passthrough

There are three common paths for I/O:

1. Emulating real hardware. Good compatibility, many traps.
2. Guest uses a virtio driver to exchange buffers with the host via shared memory (virtqueue).
3. Using IOMMU + VFIO / SR-IOV to pass a device or virtual function to the VM, reducing intermediate layers.

The closer the performance is to hardware, the more troublesome migration, sharing, and device management usually become. Cloud providers choose different combinations based on instance types.

---

## 3 · What does a cloud hypervisor do?

When a user creates a VM, the control plane selects a host, allocates vCPUs and memory, and prepares virtual networking, disks, and boot images. The hypervisor/VMM on the host is responsible for execution and isolation, while the network and storage data planes may be handled by dedicated hardware or host services.

AWS Nitro is a great example. AWS documentation states that the Nitro Hypervisor is based on KVM, while network, storage, and management functions are offloaded to Nitro Cards, allowing the host CPU to do less device simulation. The design philosophy here is more worth remembering than the product name:

```text
small hypervisor
+ hardware-assisted CPU / memory isolation
+ offloaded network and storage data plane
```

Public documentation for Google Compute Engine also indicates that its VMs run on a security-hardened KVM-based hypervisor. KVM is a common foundation, but the control plane, device models, migration, and security implementations differ for every cloud provider.

Firecracker, on the other hand, strips the VMM and device model down to the bare minimum, retaining only what is needed for serverless/container workloads. It still uses KVM, so microVMs still have an independent guest kernel, but with lower startup and memory overhead than general-purpose VMs.

---

## 4 · How are containers assembled?

A container is not a single kernel object. It is a product form resulting from several Linux mechanisms combined with images, runtimes, and network configurations.

### chroot: Changing only `/`

`chroot(path)` changes the root directory used by a process when resolving absolute paths. It does not isolate PIDs, networks, users, or resources, nor does it close file descriptors pointing outside. The Linux man page explicitly states it should not be used as a complete security sandbox.

```text
before: / -> host filesystem
after:  / -> selected directory tree
```

A container root filesystem requires mount namespaces, `pivot_root`, read-only layers, and permission restrictions to be complete. Explaining a container as just an "advanced chroot" misses most of the picture.

### Namespace: Limiting what a process can see

Linux namespaces wrap globally visible resources into individual views for each process group:

| Namespace | Isolated View |
|---|---|
| PID | process ID tree |
| Mount | mount points and filesystem view |
| Network | interface, route, port, socket stack |
| UTS | hostname |
| IPC | shared memory, message queues, etc. |
| User | UID/GID and capability mapping |
| Cgroup | visible root of the cgroup hierarchy |

Namespaces answer "who can be seen." They do not guarantee that a container can only use 2 GB of memory, nor do they automatically block all dangerous syscalls.

### cgroup: Limiting how much a process can use

cgroups organize processes into a hierarchy, with controllers performing resource accounting and control:

```text
cpu.max       CPU quota
memory.max    memory hard limit
io.max        block I/O limit
pids.max      process count limit
```

It answers "how much can be used." Without cgroups, a container's processes could still consume all the memory or CPU of the entire host.

### Security restrictions are still missing

Sharing a kernel means syscalls are an attack surface, so production containers usually also require:

- Linux capabilities: Breaking down root privileges and removing unnecessary capabilities by default;
- seccomp: Allowing or denying syscalls;
- AppArmor / SELinux: Restricting the objects a process can access;
- user namespace / rootless: Mapping root inside the container to a normal UID on the host;
- Read-only filesystems, disabling privileged mode, and minimizing hostPath mounts.

Namespaces are visibility boundaries, not complete security boundaries.

---

## 5 · What did Docker add to this mechanism?

Docker primarily solves image building, distribution, and runtime experience. A simplified data path is:

```text
Dockerfile
  -> immutable image layers
  -> registry
  -> dockerd / containerd
  -> OCI runtime such as runc
  -> namespaces + cgroups + mount + security policy
  -> application process
```

An image is a read-only layer plus metadata; at runtime, a writable layer is added on top. Multiple containers can share read-only layers, making startup and distribution cheaper than copying an entire VM disk.

Docker is not equal to containers, and containers are not equal to Docker. Kubernetes often interfaces with containerd or other runtimes via CRI; at the lowest level, it still needs to create Linux processes and configure isolation mechanisms.

### What `docker run` does conceptually

```text
pull / locate image
prepare root filesystem
create namespaces
attach cgroups
configure veth, bridge and routes
apply capabilities, seccomp and LSM profile
exec the container entrypoint
```

The container sees PID 1, but on the host, it is just a normal process. Precisely because of this, `kill`, signal handling, zombie reaping, and graceful shutdown are still problems that the application needs to handle.

---

## 6 · What is the use of containers in system design?

### Unified deployment artifact

Images fix the application, runtime, and dependency versions together. Dev, CI, and production run the same digest, reducing issues where "the machine happened to have a different version installed."

### Enabling schedulers to safely move workloads

With CPU/memory requests, limits, health checks, and standard startup methods, schedulers can perform bin packing, restarts, rolling releases, and autoscaling. Containers are the unit of delivery; systems like Kubernetes are the scheduling and orchestration layer.

### Controlling noisy neighbors

cgroups provide a minimum resource boundary for multiple workloads sharing a machine. Requests are used for scheduling capacity, while limits are used to constrain usage; misconfiguring these can lead to CPU throttling, OOM kills, or low host utilization.

### Accelerating releases and rollbacks

Immutable images combined with versioned deployments allow for starting new replicas and routing traffic only after readiness checks pass. Rollbacks involve switching back to an old digest, rather than executing a sequence of installation scripts in reverse on the machine.

### What containers cannot solve

- They do not automatically turn stateful programs into stateless ones;
- The writable layer is not durable storage;
- They do not replace replication, backups, and disaster recovery;
- Sharing a kernel is not the strongest isolation for hostile multi-tenancy.

Databases, queues, and object storage should still hold the source of truth. Whether a Pod can be safely rebuilt after being deleted is a more practical check.

---

## 7 · How to choose between VM, container, and microVM

| Scenario | Reasonable Starting Point |
|---|---|
| General APIs, workers, internal services | Container |
| Need different OS / kernel, strong tenant isolation | VM |
| Untrusted short-lived tasks, serverless, sandboxes | MicroVM, or container + dedicated sandbox runtime |
| Need full hardware performance or special devices | Bare metal / device passthrough |
| Developer running Linux containers locally | Containers directly on Linux; macOS/Windows usually have a Linux VM layer first |

Security boundaries should match the level of tenant trust. Sharing a kernel among your own microservices is usually acceptable; relying solely on default container isolation when running code uploaded by strangers is risky.

---

## 8 · Interview Cheat Sheet

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

The most common focus in system design: containers provide standard deployment units and resource boundaries; orchestrators handle placement, health, scaling, and rollouts; external data systems store non-volatile state.

---

## Primary Sources

- [Linux namespaces overview](https://man7.org/linux/man-pages/man7/namespaces.7.html)
- [Linux `chroot(2)`](https://man7.org/linux/man-pages/man2/chroot.2.html)
- [Linux cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- [Linux KVM documentation](https://docs.kernel.org/virt/kvm/index.html)
- [Linux virtio documentation](https://docs.kernel.org/driver-api/virtio/virtio.html)
- [QEMU system emulation](https://www.qemu.org/docs/master/system/introduction.html)
- [Docker Engine security](https://docs.docker.com/engine/security/)
- [AWS Nitro System architecture](https://docs.aws.amazon.com/whitepapers/latest/security-design-of-aws-nitro-system/the-nitro-system-journey.html)
- [Firecracker microVM](https://firecracker-microvm.github.io/)

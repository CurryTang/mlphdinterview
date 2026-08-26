# 性能优化 4 · Coredump 配置、产生与分析

生产环境和量化交易系统里的崩溃往往是偶发的：某个策略进程跑了三天，在某一次行情推送里踩中一个悬垂指针，段错误退出，日志里只留下最后几行毫无信息量的输出，现场早已随进程消失，根本没法用调试器实时"抓包"。Core dump 解决的正是这个问题：操作系统在进程因致命信号异常终止的那一瞬间，把它的完整内存镜像连同寄存器状态一起写入磁盘，事后可以用 `gdb` 反复加载、反复分析，相当于把犯罪现场完整封存下来。这一讲和前面 C++ 系列讲过的内存泄漏、Valgrind/ASan 是两类不同的工具：内存泄漏排查关心的是"内存有没有被正确释放"，而 core dump 关心的是"进程已经死了，怎么从尸检里找到死因"，两者互补但不重叠。本讲按"是什么 → 怎么配置 → 什么原因触发 → 怎么用 gdb 分析 → 生产环境为什么常关闭"的顺序展开，全程配真实命令行操作。

```text
1. 遇到"进程崩溃了但日志没有有效信息"类问题，第一反应是问：core dump 有没有开启（ulimit -c、systemd LimitCORE）、core 文件生成在哪（core_pattern 是否被 systemd-coredump 接管）。
2. 拿到 core 文件后，先用 gdb 加载可执行文件 + core 文件，第一条命令永远是 bt，看崩溃时刻的调用栈长什么样，而不是直接猜代码。
3. 看到崩溃现象要先分类信号：SIGSEGV 对应非法内存访问，SIGABRT 对应主动终止（assert/堆损坏/未捕获异常），SIGBUS 对应对齐或映射文件问题，SIGFPE 对应算术错误（典型是整数除零），不同信号缩小排查范围的方式完全不同。
4. bt 定位到可疑帧之后，用 frame N 切进去，配合 list 看源码、print 看变量值，必要时 bt full 一次性把所有帧的局部变量摊开看。
5. 讨论"为什么线上关 core dump"这类工程权衡题，要同时讲清楚安全（敏感数据落盘）和成本（磁盘占用）两个维度，并知道用 ASan/Valgrind 这类更早期的检测手段作为替代方案。
```

---

## 1. Core dump 是什么、为什么需要它

**核心结论**：core dump（核心转储）是操作系统在进程因致命信号异常终止时，把该进程终止那一刻的完整内存镜像和 CPU 状态写入一个磁盘文件（core 文件）的机制，目的是把"崩溃现场"完整保留下来，供事后离线调试。

一个正常运行的进程如果被 `SIGSEGV`、`SIGABRT` 这类信号杀死，默认情况下进程直接消失，只留下退出码和可能的一行错误日志。内核在终止进程之前，如果判定该信号的默认动作是 "Core"（即会产生核心转储），会先把进程当时的地址空间（包括栈、堆、已加载的共享库映射、寄存器组、信号处理状态）序列化写入一个 core 文件，再真正结束进程。这个文件本质上是进程在死亡瞬间的一张完整快照：

```text
core 文件（概念上包含的内容）
┌───────────────────────────────┐
│  ELF 文件头 + PT_NOTE 段         │  进程 PID、崩溃信号编号、发生崩溃的线程 ID
├───────────────────────────────┤
│  寄存器状态（每个线程一份）        │  PC/RIP、SP/RSP、通用寄存器、标志位
├───────────────────────────────┤
│  栈内存镜像                      │  崩溃时刻每个线程的调用栈内容
├───────────────────────────────┤
│  堆内存镜像                      │  崩溃时刻堆上所有对象的实际字节内容
├───────────────────────────────┤
│  已加载共享库的内存映射信息        │  每个 .so 加载的基址，供符号还原用
└───────────────────────────────┘
```

这个信息量是普通日志远远给不了的：日志只能打印你事先想到要打印的东西，而 core dump 里有崩溃那一刻**全部**的内存状态，包括你完全没想到要检查的某个变量。它的核心价值在于把"必须实时复现才能调试"变成"崩溃发生一次就够了，之后可以慢慢分析"，这对偶发性崩溃（一天几十万次调用只崩一次、依赖特定的行情时序才触发的竞态）是决定性的：这类 bug 在生产环境里用调试器挂着实时抓，命中率极低，而 core dump 是被动收集、不需要提前预判触发条件。

配合 `g++ -g` 编译时保留的调试符号信息，core 文件里的原始地址和寄存器数值可以被 `gdb` 还原成源码行号、函数名、变量名，这也是下文第 4 节要讲的分析流程的前提。

---

## 2. 配置 core dump

**核心结论**：core dump 默认在几乎所有主流 Linux 发行版上都是关闭的（大小限制为 0），要真正拿到 core 文件需要同时确认两件事：`ulimit -c` 允许生成、`core_pattern` 决定生成到哪、以什么名字生成；而在使用了 `systemd-coredump` 的现代发行版上，core 文件根本不会出现在工作目录，需要用 `coredumpctl` 去找。

### 2.1 `ulimit -c`：是否允许生成、生成多大

`ulimit -c` 控制的是当前 shell（以及它派生出的所有子进程）允许生成的 core 文件的最大字节数，单位是 512 字节的块（`ulimit -c unlimited` 表示不限制大小）。

```bash
# 查看当前限制，多数发行版默认是 0，即禁止生成 core 文件
$ ulimit -c
0

# 允许生成，不限制大小
$ ulimit -c unlimited

# 也可以设置一个具体上限，比如 100000（单位 512 字节块，约 48.8MB）
$ ulimit -c 100000
```

这个限制是**按进程继承**的：`ulimit -c` 是当前 shell 进程的一个资源限制（`RLIMIT_CORE`），子进程 fork 出来时会继承这个限制值，所以必须在**启动目标进程之前**、在同一个 shell 会话里设置好，而不能进程跑起来之后再补设置。对于用 `systemd` 管理的服务，`ulimit -c unlimited` 在交互式 shell 里设置了也不会影响 systemd 启动的进程，因为它们的父进程是 `systemd` 而不是你的登录 shell，必须在对应的 unit 文件里显式配置：

```ini
[Service]
ExecStart=/opt/myapp/bin/quant_engine
LimitCORE=infinity
```

`LimitCORE=infinity` 等价于给这个 systemd 服务本身设置 `ulimit -c unlimited`，重启服务（`systemctl daemon-reload && systemctl restart myapp`）后生效。

### 2.2 `core_pattern`：生成到哪、叫什么名字

即使 `ulimit -c` 允许生成 core 文件，具体生成在哪个目录、文件名是什么，由内核参数 `/proc/sys/kernel/core_pattern` 决定：

```bash
$ cat /proc/sys/kernel/core_pattern
core

# 改成带 PID、可执行文件名、时间戳，避免同名覆盖
$ sudo sysctl -w kernel.core_pattern="/var/cores/core.%e.%p.%t"
```

常用占位符：`%p` 进程 PID，`%e` 可执行文件名（截断到 15 字符），`%t` 崩溃时的 Unix 时间戳，`%h` 主机名，`%s` 触发崩溃的信号编号。默认值通常只是 `core` 或 `core.%p`，且相对路径是相对进程当时的**工作目录**而不是固定目录，这也是很多人"core 文件到底生成在哪"疑惑的来源之一。

### 2.3 常见坑：`systemd-coredump` 接管了 `core_pattern`

在 Ubuntu、Debian、Fedora、RHEL 8+ 等安装了 `systemd-coredump` 的现代发行版上，`core_pattern` 默认已经被 systemd 配置成一条**管道命令**，而不是一个文件路径模板：

```bash
$ cat /proc/sys/kernel/core_pattern
|/usr/lib/systemd/systemd-coredump %P %u %g %s %t %c %h %e
```

当 `core_pattern` 第一个字符是 `|` 时，内核不会把 core 内容写成普通文件，而是把它通过管道传给指定的程序处理，这里就是 `systemd-coredump`。它把接收到的 core 数据统一存放在 `/var/lib/systemd/coredump/` 下（通常还会压缩），并登记进日志数据库，此时无论你把 `ulimit -c` 设成多大、无论你在哪个目录下运行程序，当前工作目录下都**不会出现** `core` 或 `core.PID` 文件，这是排查时最容易踩的坑：直接 `ls` 当前目录找不到 core 文件，会误以为 core dump 没有生成，实际上只是被 systemd 接管到别处去了。正确的做法是用 `coredumpctl`：

```bash
# 列出所有已记录的 core dump（按时间倒序）
$ coredumpctl list

# 列出某个可执行文件相关的 core dump
$ coredumpctl list quant_engine

# 把最近一次 core dump 导出成普通文件，供 gdb ./prog core 这种传统方式加载
$ coredumpctl dump quant_engine -o /tmp/quant_engine.core

# 更直接：一条命令自动用 gdb 加载对应可执行文件和 core dump
$ coredumpctl gdb quant_engine
```

另外要注意，管道模式下内核文档明确说明 `ulimit -c` 对管道接收方不完全生效（管道场景下 core 大小限制的语义和写普通文件时不同，systemd-coredump 自身用 `coredump.conf` 里的 `ProcessSizeMax`/`ExternalSizeMax` 等参数控制大小上限），所以在这种系统上想控制 core 文件大小，应该改 `/etc/systemd/coredump.conf`，而不是纠结 `ulimit -c` 的具体数值。

**常见追问 / 面试陷阱**

> 追问"我 `ulimit -c unlimited` 了，为什么还是没有 core 文件"：按顺序排查：① 确认设置是在启动目标进程的**那个** shell 里做的，且进程是从这个 shell 直接 fork 出来的；② 确认程序确实是被会触发 core dump 的信号杀死的，而不是自己正常 `exit()` 或被 `SIGKILL`（`SIGKILL` 不产生 core dump，见下一节）；③ 确认 `/proc/sys/kernel/core_pattern` 是不是被 systemd-coredump 接管了，用 `coredumpctl list` 而不是 `ls` 去找；④ 确认程序所在目录或 `core_pattern` 指定目录有写权限，以及磁盘没有满；⑤ 对于设置了 `setuid`/`setgid` 的可执行文件，内核默认不为它们生成 core dump（除非配置了 `/proc/sys/fs/suid_dumpable`），这是一个专门的安全限制。

---

## 3. Core dump 产生的常见原因

**核心结论**：只有部分信号的默认动作会触发 core dump，其中面试和实际排查里最常见的四个是 `SIGSEGV`、`SIGABRT`、`SIGBUS`、`SIGFPE`；产生 core dump 本身不是 bug，它是操作系统在 bug 导致进程崩溃的那一刻帮你保留下来的现场证据，分析的目标是通过这份证据反推出代码里的 bug 位置和触发路径。

Linux 上信号的默认处理动作在 `signal(7)` 里有明确定义，动作是 "Core" 的信号包括 `SIGQUIT`、`SIGILL`、`SIGABRT`、`SIGFPE`、`SIGSEGV`、`SIGBUS`、`SIGSYS`、`SIGTRAP`、`SIGXCPU`、`SIGXFSZ`，其余会终止进程的信号（比如 `SIGKILL`、`SIGTERM`、`SIGINT`）动作是 "Term"，不产生 core dump，这也是为什么 `kill -9`（`SIGKILL`）杀掉的进程永远不会留下 core 文件。

**`SIGSEGV`（段错误，Segmentation Fault）**：进程访问了它没有权限访问的内存地址，这是 C++ 面试和实际生产事故里出现频率最高的信号。典型触发场景：

```cpp
int* p = nullptr;
*p = 42;                     // 空指针解引用

int arr[10];
arr[100] = 1;                // 数组越界访问到未映射页

int* q = new int(5);
delete q;
*q = 10;                     // 访问已释放内存（use-after-free）
```

**`SIGABRT`（异常终止）**：进程调用了 `abort()`，通常不是内核直接判定内存非法，而是程序自己或运行时库主动判定"不能再往下跑了"。常见来源：

```cpp
#include <cassert>
assert(x > 0);                // 断言失败，assert 内部调用 abort()

// glibc 的堆完整性检查发现堆元数据被破坏（比如 double free、缓冲区溢出踩坏了 chunk header）
free(ptr);
free(ptr);                    // double free，glibc 检测到后 abort

// C++ 异常没有被任何 catch 捕获，触发 std::terminate，默认行为是调用 abort
throw std::runtime_error("oops");   // 如果一路没人 catch
```

**`SIGBUS`（总线错误）**：和 `SIGSEGV` 容易混淆，区别在于 `SIGSEGV` 是访问了权限不允许的合法地址，`SIGBUS` 通常是地址本身或访问方式有问题，最常见的两种场景是访问了未对齐地址（在要求严格对齐的架构上，或者对 `mmap` 映射内存做了非对齐访问）、以及访问了一个通过 `mmap` 映射的文件，但访问的偏移量超出了文件实际大小（文件在映射后被另一个进程截断）。

**`SIGFPE`（浮点异常，Floating-Point Exception）**：名字带"浮点"但最常见的触发原因其实是整数运算错误，尤其是整数除零：

```cpp
int a = 10, b = 0;
int c = a / b;   // 整数除零，触发 SIGFPE，而不是像浮点除零那样得到 inf
```

浮点数的除零在 IEEE 754 标准下默认得到 `inf`/`nan`，不会触发信号；`SIGFPE` 主要覆盖整数除零、整数除法溢出（如 `INT_MIN / -1`）等场景。

**核心认知**：这四类信号触发 core dump 是内核在执行既定策略，进程崩溃的**根因**永远是应用代码或者它依赖的库里的逻辑错误：空指针没判空、下标越界没做边界检查、堆结构被野指针写坏、除数没检查。Core dump 只是把"错误发生瞬间的完整现场"保存了下来，它自己不会告诉你 bug 在哪一行，需要用第 4 节的方法从现场证据反推出触发路径。把"进程 core dump 了"直接等同于"这就是 bug"是不准确的说法，准确的说法是"进程因为某个 bug 触发了致命信号，而这次内核帮我们把现场存了下来"。

---

## 4. 用 gdb 分析 core dump

**核心结论**：分析流程是"编译保留符号 → 复现崩溃拿到 core 文件 → `gdb 可执行文件 core文件` 加载 → `bt` 定位调用栈 → `frame`/`list`/`print` 逐层深挖"，其中 `bt` 是永远第一条要敲的命令。

### 4.1 构造一个会崩溃的例子

下面这段代码里，`getDanglingPointer()` 返回了一个函数返回后就已经销毁的局部变量的地址，调用方后续解引用这个悬垂指针，触发 `SIGSEGV`：

```cpp
// crash_demo.cpp
#include <cstdio>

int* getDanglingPointer() {
    int local_value = 42;
    return &local_value;      // 返回局部变量的地址，函数返回后 local_value 已销毁
}

void useValue(int* p) {
    printf("value = %d\n", *p);   // 解引用悬垂指针，未定义行为，实际运行常常段错误
}

int main() {
    int* dangling = getDanglingPointer();
    useValue(dangling);
    return 0;
}
```

### 4.2 编译、运行、拿到 core 文件

```bash
# -g 保留调试符号（行号、变量名、类型信息），gdb 分析必须有这一步
$ g++ -g -O0 -o crash_demo crash_demo.cpp

# 确认当前 shell 允许生成 core dump
$ ulimit -c unlimited

# 运行触发崩溃
$ ./crash_demo
Segmentation fault (core dumped)

# 确认 core 文件生成位置（假设 core_pattern 是普通文件模式，未被 systemd-coredump 接管）
$ ls core*
core

# 如果是 systemd-coredump 接管的系统，改用：
$ coredumpctl dump crash_demo -o ./core
```

编译时加 `-O0` 是为了避免编译器优化把局部变量优化掉或者重排指令顺序，导致 gdb 里看到的变量值和源码逻辑对不上，实际项目里生产构建通常是 `-O2`，此时 gdb 里能看到的信息会打折扣（部分变量显示 `<optimized out>`），这也是"生产环境该不该带 `-g` 编译"经常被问到的点：**带 `-g` 不会影响运行时性能，只是让二进制文件变大**（调试信息是独立的 section，不影响代码执行路径），生产环境完全可以在优化的同时保留 `-g`。

### 4.3 加载 core 文件

```bash
$ gdb ./crash_demo core
```

gdb 启动后会打印类似下面的信息，指出崩溃的信号类型和触发指令地址：

```text
Program terminated with signal SIGSEGV, Segmentation fault.
#0  0x0000000000401136 in useValue (p=0x7ffd3a8b2c9c) at crash_demo.cpp:9
9	    printf("value = %d\n", *p);
```

### 4.4 常用命令

**`bt`（backtrace）**：打印崩溃时刻完整的调用栈，从崩溃点（第 0 帧）一路到 `main`，是分析的起点。

```text
(gdb) bt
#0  useValue (p=0x7ffd3a8b2c9c) at crash_demo.cpp:9
#1  0x0000000000401160 in main () at crash_demo.cpp:15
```

这里立刻能看出崩溃发生在 `useValue` 内部第 9 行的解引用，调用者是 `main` 第 15 行。虽然还没看到 `getDanglingPointer` 返回的是悬垂指针，但已经把范围收窄到了这两行之间的数据流。

**`bt full`**：在 `bt` 基础上，把每一帧的局部变量也一并打印出来，适合一次性摊开看整条调用链上各层的状态，不用逐帧 `frame` 切换。

```text
(gdb) bt full
#0  useValue (p=0x7ffd3a8b2c9c) at crash_demo.cpp:9
        No locals.
#1  0x0000000000401160 in main () at crash_demo.cpp:15
        dangling = 0x7ffd3a8b2c9c
```

**`frame N` / `f N`**：切换到调用栈的第 N 帧，之后 `list`、`print` 等命令都基于这一帧的上下文。

```text
(gdb) frame 1
#1  0x0000000000401160 in main () at crash_demo.cpp:15
15	    useValue(dangling);
(gdb) print dangling
$1 = (int *) 0x7ffd3a8b2c9c
```

`dangling` 这个地址落在栈区范围内（典型是 `0x7ffd...` 这种高地址），结合源码里 `getDanglingPointer` 返回的是函数内局部变量的地址，可以直接判断这是一个"返回局部变量地址导致的悬垂指针"问题。

**`list`**：显示当前帧对应源码行附近的上下文，不加参数默认显示当前停留位置前后几行；也可以 `list 函数名` 跳到指定函数。

```text
(gdb) list
4	int* getDanglingPointer() {
5	    int local_value = 42;
6	    return &local_value;
7	}
8
9	void useValue(int* p) {
```

**`info registers`**：打印当前帧对应的完整寄存器状态，包括通用寄存器、栈指针 `rsp`、指令指针 `rip`。分析纯 C++ 代码时不常用，但在手写汇编、分析编译器生成代码是否符合调用约定、或者崩溃点在没有调试符号的第三方库里（只能看到裸地址而没有源码行号）时非常关键。

```text
(gdb) info registers
rax            0x0                 0
rbx            0x7ffd3a8b2cc8      140725232235208
rip            0x401136            0x401136 <useValue(int*)+13>
rsp            0x7ffd3a8b2c80      0x7ffd3a8b2c80
```

**`print 变量名`**（可简写 `p`）：打印当前帧任意变量或表达式的值，支持解引用、成员访问、数组下标等 C++ 表达式语法，比如 `print *p`、`print obj.member`、`print arr[3]`。

### 4.5 完整流程小结

```text
g++ -g 编译  →  运行触发崩溃 / coredumpctl dump 取出 core  →  gdb 可执行文件 core文件
        │
        ├─ bt          先看完整调用栈，锁定崩溃发生在哪个函数、哪一行
        ├─ bt full      需要同时看多层局部变量时用这个代替反复 frame 切换
        ├─ frame N      切到可疑的那一帧
        ├─ list         看这一帧对应的源码上下文
        ├─ print 变量    验证具体变量此刻的值是否符合预期
        └─ info registers  裸汇编/无符号库场景下的最后手段
```

**常见追问 / 面试陷阱**

> 追问"gdb 里 `bt` 显示的栈帧是乱码/`??`，怎么办"：多半是符号缺失。检查编译时有没有加 `-g`；如果崩溃发生在第三方共享库内部，需要该库也带调试符号（或者对应的 debuginfo 包）；`gdb` 里可以用 `info sharedlibrary` 查看每个共享库有没有正确加载调试符号。另外要确认 `gdb` 打开的可执行文件和产生 core 的可执行文件是完全同一份二进制（同一次编译产物），如果中间重新编译过、哪怕代码一模一样，地址布局也可能变化，导致符号对不上。

---

## 5. 常见追问：为什么生产环境经常主动关闭 core dump

**核心结论**：生产环境关闭或严格限制 core dump 主要出于两个工程考虑：数据安全（core 文件里可能包含敏感信息）和资源成本（core 文件体积大，容易迅速占满磁盘），实践上通常是预发/测试环境打开、生产环境关闭或限量，并配合 ASan/Valgrind 这类能在崩溃发生前更早暴露问题的工具，减少真正依赖 core dump 事后取证的场景。

**数据安全**：core 文件本质上是进程崩溃那一刻完整的内存镜像，进程运行期间处理过的一切数据（用户密码、API 密钥、数据库连接字符串、交易账户信息、其他敏感的业务数据），只要还留在内存里没被清空，就会原样出现在 core 文件里。如果 core 文件权限设置不当、或者被上传到不受控的日志/监控系统，等于把进程内存里的敏感信息整体导出成了一份静态文件，这是需要专门治理的数据泄露面。金融交易系统里这一点尤其敏感，因为进程内存里往往驻留着账户、持仓、风控参数这类高价值数据。

**磁盘空间**：core 文件大小和进程实际占用的内存大小是同一量级的（尤其是堆内存部分几乎是完整拷贝），一个占用几十 GB 内存的进程崩溃一次，就可能瞬间在磁盘上多出几十 GB 的文件。如果服务在某种异常输入下持续反复崩溃重启（crash loop），每次都留下一个 core 文件而没有清理策略，磁盘可能在很短时间内被写满，进而引发级联故障（日志写不进去、其他服务因磁盘满而异常）。

**工程实践上的折中**：

- 预发布/测试/灰度环境：通常开启 core dump（`ulimit -c unlimited` 或配置 `LimitCORE=infinity`），因为这些环境暴露出的崩溃需要被完整分析，且这些环境的数据敏感度、磁盘规模压力通常小于生产。
- 生产环境：要么完全关闭（`ulimit -c 0`），要么开启但严格限制大小和保留数量（`systemd-coredump` 的 `coredump.conf` 里 `ExternalSizeMax`、`MaxUse`、`KeepFree` 等参数就是为这个场景设计的），并对 core 文件目录做访问控制和自动清理。
- 更根本的思路是把问题检测往前移：用 AddressSanitizer（编译期插桩，`-fsanitize=address`）或 Valgrind 在测试阶段就能在**真正崩溃之前**捕获到越界写、use-after-free 等问题，并直接打印出发生问题时的调用栈，不需要等到进程真的段错误、产生 core 文件、再离线分析。这两个工具在前面 C++ 系列（内存泄漏一讲）里已经介绍过基本用法，这里只强调它们和 core dump 的分工关系：ASan/Valgrind 是"预防性、开发测试阶段的早期检测"，core dump 是"生产环境事后取证的最后一道防线"，两者不是互相替代关系，成熟的工程流程通常两者都配置，只是生效的阶段和环境不同。

**常见追问 / 面试陷阱**

> 追问"关了 core dump，生产环境崩溃了怎么办"：不能完全依赖 core dump 作为唯一手段。实践上会叠加其他机制：崩溃前的详细结构化日志（尤其是在关键路径打点，保证崩溃前最后状态可追溯）、进程管理器自动重启并保留最近若干次崩溃的简要信息（比如只保留 `bt` 的文本摘要而不保留完整 core 文件，用 `coredumpctl` 的 `ProcessSizeMax` 配置成只在必要时才落盘完整文件）、以及事后用测试环境复现问题再分析。也有折中方案：生产环境限制 core 文件大小上限（比如只保留几百 MB），对小进程仍然可以完整保留现场，对超大内存进程则牺牲完整性换取磁盘安全。

---

## 快速选择题

```quiz
title: 快速选择题 1
question: 下列哪个命令能让当前 shell 启动的进程在崩溃时生成不限制大小的 core 文件？
answer: B
A. `ulimit -f unlimited`
B. `ulimit -c unlimited`
C. `sysctl -w kernel.core_pattern=unlimited`
D. `export CORE_DUMP=1`
explanation: `ulimit -c` 控制的正是当前 shell 及其子进程允许生成的 core 文件大小上限，`unlimited` 表示不做限制；`ulimit -f` 控制的是文件大小限制（和 core dump 无关），`core_pattern` 只控制生成路径/命名规则而不是"是否允许"。
```

```quiz
title: 快速选择题 2
question: 一个 systemd 管理的服务，交互式 shell 里已经执行了 `ulimit -c unlimited`，但服务崩溃后仍然没有 core 文件，最可能的原因是：
answer: B
A. `ulimit -c unlimited` 语法写错了
B. systemd 启动的进程不是这个交互式 shell 的子进程，没有继承该限制，需要在 unit 文件里配置 `LimitCORE=infinity`
C. systemd 服务永远不会崩溃
D. `ulimit -c` 只对 `bash` 生效，对 `zsh` 无效
explanation: `RLIMIT_CORE` 是按进程继承的，systemd 启动的服务进程的父进程是 `systemd` 本身，不是登录 shell，因此登录 shell 里设置的 `ulimit -c` 不会传递给它，必须在对应的 `.service` 文件里显式加 `LimitCORE=infinity`。
```

```quiz
title: 快速选择题 3
question: 在一台 `/proc/sys/kernel/core_pattern` 内容为 `|/usr/lib/systemd/systemd-coredump %P %u %g %s %t %c %h %e` 的机器上，进程崩溃后应该：
answer: B
A. 在进程启动时的工作目录下找 `core` 或 `core.PID` 文件
B. 用 `coredumpctl list` / `coredumpctl dump` 查看和导出 core dump
C. 检查 `/tmp` 目录
D. 崩溃不会生成任何 core 相关记录
explanation: `core_pattern` 以 `|` 开头表示 core 数据被管道转交给 `systemd-coredump` 统一处理，不再写成普通文件，工作目录下不会出现 core 文件，需要通过 `coredumpctl` 查询和导出。
```

```quiz
title: 快速选择题 4
question: 下列哪个信号的默认动作不会产生 core dump？
answer: C
A. `SIGSEGV`
B. `SIGABRT`
C. `SIGKILL`
D. `SIGFPE`
explanation: `SIGKILL` 的默认动作是直接终止进程（Term），不产生 core dump，也无法被捕获或忽略，这正是 `kill -9` 常被用来强制杀掉进程但完全拿不到任何崩溃现场信息的原因；`SIGSEGV`/`SIGABRT`/`SIGFPE` 的默认动作都是 Core。
```

```quiz
title: 快速选择题 5
question: 阅读代码：`int a = 5, b = 0; int c = a / b;`，最可能触发哪个信号？
answer: C
A. `SIGSEGV`
B. `SIGBUS`
C. `SIGFPE`
D. `SIGABRT`
explanation: 整数除零触发的是 `SIGFPE`（浮点异常，尽管名字带"浮点"，但整数除零/溢出是它最常见的触发场景），而不是浮点数除零（浮点除零在 IEEE 754 下得到 `inf`，不触发信号）。
```

```quiz
title: 快速选择题 6
question: `assert(x > 0)` 在断言失败时，底层依赖的是哪个机制来终止进程？
answer: B
A. 直接抛出 `SIGSEGV`
B. 调用 `abort()`，触发 `SIGABRT`
C. 调用 `exit(1)`，正常退出不产生 core dump
D. 触发 `SIGBUS`
explanation: `assert` 宏在条件为假时会打印诊断信息并调用 `abort()`，`abort()` 会向进程自身发送 `SIGABRT`，默认动作是终止并产生 core dump。
```

```quiz
title: 快速选择题 7
question: 用 gdb 分析 core 文件时，第一条最应该执行的命令是：
answer: B
A. `print main`
B. `bt`
C. `info registers`
D. `list`
explanation: `bt`（backtrace）打印崩溃时刻完整的调用栈，是定位崩溃发生在哪个函数、哪一层调用的起点；`info registers`、`list`、`print` 都需要先确定关注哪一帧之后才有针对性地使用。
```

```quiz
title: 快速选择题 8
question: 编译时不加 `-g` 直接分析 core 文件，会发生什么？
answer: B
A. gdb 无法打开 core 文件
B. 能看到调用栈的大致地址，但无法直接对应到源码行号和变量名，可读性大幅下降
C. 完全等价于加了 `-g`，因为符号信息总是从 core 文件本身读取
D. core 文件根本不会生成
explanation: 调试符号（行号、变量名、类型信息）来自编译产物里的调试信息 section，不是运行时状态的一部分，core 文件本身不携带这些信息；不加 `-g` 时 gdb 只能显示裸地址和有限的符号（如果没有被 strip），`bt`/`list`/`print` 的可用性会显著下降。
```

```quiz
title: 快速选择题 9
question: 关于生产环境是否应该开启 core dump，下列说法最准确的是：
answer: C
A. 生产环境必须始终关闭 core dump，没有例外
B. 生产环境必须始终开启 core dump，否则无法排查问题
C. 需要在"事后取证能力"与"敏感数据落盘风险、磁盘占用成本"之间权衡，常见做法是关闭或限量，并配合 ASan/Valgrind 等更早期的检测手段减少对 core dump 的依赖
D. 只要磁盘够大，开不开 core dump 没有区别
explanation: 是否开启是一个工程权衡：完整的 core dump 有利于事后排查，但可能包含敏感数据且体积巨大，实践上通常在预发/测试环境开启、生产环境关闭或严格限量，并用更早期的检测工具减少真正需要 core dump 兜底的场景。
```

```quiz
title: 快速选择题 10
question: 一个设置了 `setuid` 位的可执行文件运行时崩溃，即使 `ulimit -c unlimited` 也常常拿不到 core 文件，原因是：
answer: B
A. `setuid` 程序不会崩溃
B. 内核出于安全考虑，默认不为 `setuid`/`setgid` 程序生成 core dump，除非显式配置 `/proc/sys/fs/suid_dumpable`
C. `ulimit -c` 对 `setuid` 程序无效，必须用 `sysctl` 代替
D. 这类程序必须用 `sudo gdb` 才能生成 core
explanation: 这是一条独立于 `ulimit -c` 的安全策略：`setuid`/`setgid` 程序的内存中可能包含提权后才能访问的敏感数据，内核默认禁止为其生成 core dump，避免低权限用户通过诱导其崩溃来窃取高权限进程内存内容；需要调试时可以临时调整 `fs.suid_dumpable`，但这本身是有安全代价的操作。
```

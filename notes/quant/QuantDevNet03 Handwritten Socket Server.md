# 计算机网络 3 · 手撕 Socket Server

前两讲把 IO 模型和 `select`/`poll`/`epoll` 的内部机制讲透了，这一讲把这些原理落到具体能跑起来的代码上——"手撕一个 socket server"是后端/量化基础设施岗位面试里出现频率很高的一道编程题，考察的不是算法技巧，而是对 socket API 调用顺序、每个系统调用的参数含义和错误处理、以及并发模型选择的完整理解。这里给出两个版本：一个最简单的阻塞版 TCP echo server，把 `socket`/`bind`/`listen`/`accept`/`read`/`write`/`close` 这条链路走一遍；一个基于 epoll 边沿触发的进阶版本，直接复用上一讲讲过的 ET 模式必须配合非阻塞 IO、循环读到 `EAGAIN` 为止这条规则，这里不再重复 epoll 红黑树/就绪列表的原理，只讲怎么把它写成能跑的事件循环。最后补两个几乎必问的追问：TIME_WAIT 状态和 `SO_REUSEADDR`、以及粘包拆包问题。

```text
1. 遇到"手写一个echo server"，先在纸上/白板上写清楚这条调用链：socket→setsockopt(SO_REUSEADDR)→bind→listen→accept循环→read/write→close，缺一步就先扣分，尤其容易漏掉SO_REUSEADDR和错误处理。
2. 遇到"listen的第二个参数是什么"，答案要落到"全连接队列(accept queue)长度"这个具体概念上，讲清楚队列里存的是三次握手已完成、等待accept取走的连接，不要只说"最大连接数"这种模糊表述。
3. 遇到"这个阻塞版server能同时服务几个客户端"，答案是同一时刻只能真正服务一个，其他新连接的三次握手可以在内核里独立完成并进入accept队列排队，但只要队列没满就不会被拒绝，一旦队列满了新请求才会被拒绝或丢弃；改进方向要能说出thread-per-connection和IO多路复用两条路线，并说清楚各自的代价。
4. 遇到"用epoll写echo server，ET模式要注意什么"，直接对齐上一讲的结论：非阻塞fd + 循环read/accept到EAGAIN为止，这里补一个上一讲没强调的点——监听socket本身用ET时，accept也必须循环到EAGAIN，否则同一次事件里排队的多个新连接会被漏掉。
5. 遇到"TIME_WAIT为什么要等2MSL"，答案要覆盖两个原因：保证最后一个ACK丢失时能收到对方重传的FIN并正确响应、保证本次连接里滞留在网络中的旧包彻底消失不会污染同四元组的新连接，只说一个原因不算完整。
6. 遇到"粘包怎么处理"，先定性TCP是字节流协议没有消息边界，再给出两种具体方案：长度前缀法和分隔符法，不要只停留在"应用层自己处理"这种空泛的说法。
```

---

## 1. TCP 服务端 socket API 调用序列

**核心结论**：一个最基本的 TCP 服务端总是按照 `socket → bind → listen → accept → read/write → close` 这个固定顺序调用系统调用，每一步都在为下一步准备好必要的状态，顺序不能颠倒。

```text
socket()                  创建一个socket，此时只是内核里分配了一个文件描述符
   │                       和对应的协议控制块，还没有绑定地址、也不能收发数据
   ▼
bind()                    把这个socket和一个本地IP+端口绑定，
   │                       之后到达这个IP+端口的连接请求才会路由到这个socket
   ▼
listen()                  把socket从"主动socket"转换为"监听socket"，
   │                       并声明全连接队列(accept queue)的最大长度
   ▼
accept()  ◄────┐          阻塞等待，从全连接队列里取出一个已完成三次握手的
   │            │          连接，返回一个新的、专门服务这个客户端的socket；
   │            │          原监听socket不受影响，继续留在这里等下一个连接
   ▼            │
read()/write()  │         对accept返回的新socket收发数据，这一步和监听socket
   │            │          完全无关，是两个独立生命周期的fd
   ▼            │
close()         │         关闭这个已连接socket，四次挥手释放这次连接
   │            │
   └────────────┘         回到accept，继续等待/处理下一个连接
```

逐个拆解每个调用的关键参数。

**`socket(AF_INET, SOCK_STREAM, 0)`**：第一个参数 `AF_INET` 指定地址族为 IPv4（`AF_INET6` 对应 IPv6）；第二个参数 `SOCK_STREAM` 指定这是一个面向连接的字节流套接字，对应 TCP（`SOCK_DGRAM` 对应 UDP）；第三个参数 `0` 表示让内核根据前两个参数自动选择协议（对 `AF_INET`+`SOCK_STREAM` 组合就是 TCP，不需要显式传 `IPPROTO_TCP`）。调用成功返回一个新的文件描述符，这个 fd 此时还没有和任何具体的本地地址关联。

**`bind(sockfd, (struct sockaddr *)&addr, sizeof(addr))`**：需要提前填好一个 `struct sockaddr_in` 结构体，其中 `sin_family` 设为 `AF_INET`，`sin_port` 是端口号（要用 `htons` 转换成网络字节序），`sin_addr.s_addr` 是要绑定的本地 IP（服务端通常填 `INADDR_ANY`，表示监听本机所有网卡地址，也可以填一个具体网卡的 IP 只监听那一个网卡）。`bind` 的作用是告诉内核"发到这个 IP+端口的数据包，交给这个 socket 处理"，服务端几乎总是要显式 `bind` 到一个约定好的端口，否则内核会分配一个随机端口，客户端无从事先知道该连哪个端口。

**`listen(sockfd, backlog)`**：这一步把 socket 从可以主动发起连接的普通 socket 转换成被动的监听 socket，之后不能再对这个 fd 调用 `connect`。第二个参数 `backlog` 是**全连接队列（accept queue）**的最大长度，这个队列里存放的是**三次握手已经完成、正在等待应用程序调用 `accept` 把它取走**的连接——注意这和"半连接队列（SYN queue）"是两回事，半连接队列存放的是收到 `SYN` 但三次握手还没完成的连接，这个队列的长度由另一个内核参数（`/proc/sys/net/ipv4/tcp_max_syn_backlog`）控制，不受 `listen` 的 `backlog` 参数影响。全连接队列满了之后，新到达的握手完成的连接会被内核丢弃或者由内核发送 `RST` 拒绝（具体行为取决于 `tcp_abort_on_overflow` 内核参数），这也是为什么应用程序迟迟不调用 `accept` 会导致新客户端连接失败的原因。

**`accept(sockfd, (struct sockaddr *)&cliaddr, &addrlen)`**：这是最容易在概念上搞混的一步。它从监听 socket 的全连接队列里取出队首的一个连接，**返回一个全新的、和监听 socket 不同的 fd**，这个新 fd 专门用来和这一个客户端通信；原来的监听 socket（`sockfd`）完全不受影响，继续留在原地等待下一次 `accept`。如果队列为空，`accept` 默认会阻塞，直到有新连接进队。传入的 `cliaddr`/`addrlen` 用来接收对端（客户端）的地址信息，不是必需参数，传 `NULL` 也合法。

**`read`/`write`（或 `recv`/`send`）**：对 `accept` 返回的已连接 socket 收发数据，和监听 socket 完全无关。这一步是两个独立 fd 的读写，混淆"对监听 socket 读写"和"对已连接 socket 读写"是这道题里最容易出现的低级错误。

**`close(fd)`**：对已连接 socket 调用会触发四次挥手释放这次 TCP 连接；对监听 socket 调用则会让内核停止接受新连接。

**常见追问 / 面试陷阱**

> 追问"全连接队列满了会发生什么"：新到达的、已完成三次握手的连接不会被放进队列，内核会根据 `tcp_abort_on_overflow` 这个参数决定行为——默认是 `0`，直接丢弃这个握手完成的连接（不发送任何响应，客户端表现为连接超时，之后可能收到内核重传的最后一个握手包）；设为 `1` 则会发送 `RST` 主动拒绝，客户端会立刻收到"connection reset"这样的错误而不是超时。这说明"服务端迟迟不调用 accept"在高并发场景下会直接导致新连接失败，不只是响应变慢。

> 追问"半连接队列和全连接队列的区别"：半连接队列（SYN queue）存放收到 `SYN`、发出 `SYN+ACK`、等待客户端最后一个 `ACK` 的连接，处于三次握手中间状态；全连接队列（accept queue）存放三次握手已完成、等待应用程序 `accept` 取走的连接。`listen` 的第二个参数只控制全连接队列的长度，半连接队列的长度由另一个独立的内核参数控制，两者是两条队列，不能混为一谈。

---

## 2. 完整的阻塞版 TCP echo server 代码

**思路**：按照上一节的调用序列，创建监听 socket、设置 `SO_REUSEADDR`、`bind`、`listen`，然后在一个无限循环里反复 `accept`；每 `accept` 到一个连接就在一个内层循环里反复 `read`，读到数据就原样 `write` 回去，直到对端关闭连接（`read` 返回 `0`）或者出错，处理完就 `close` 这个连接的 fd，回到外层循环继续 `accept` 下一个客户端。

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

#define PORT      8888
#define BACKLOG   128
#define BUF_SIZE  4096

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

int main(void) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        die("socket");
    }

    // 允许在TIME_WAIT尚未结束时重新bind同一个地址+端口，
    // 否则服务重启后立刻监听同一端口大概率会遇到 EADDRINUSE
    int opt = 1;
    if (setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        die("setsockopt(SO_REUSEADDR)");
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(PORT);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        die("bind");
    }

    if (listen(listen_fd, BACKLOG) < 0) {
        die("listen");
    }

    printf("blocking echo server listening on port %d\n", PORT);

    for (;;) {
        struct sockaddr_in cli_addr;
        socklen_t cli_len = sizeof(cli_addr);

        int conn_fd = accept(listen_fd, (struct sockaddr *)&cli_addr, &cli_len);
        if (conn_fd < 0) {
            if (errno == EINTR) {
                continue; // 被信号打断，重试即可，不是真正的错误
            }
            perror("accept");
            continue; // 单次accept失败不应让整个服务退出，跳过继续等下一个连接
        }

        char ip_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &cli_addr.sin_addr, ip_str, sizeof(ip_str));
        printf("accepted connection from %s:%d\n", ip_str, ntohs(cli_addr.sin_port));

        char buf[BUF_SIZE];
        for (;;) {
            ssize_t n = read(conn_fd, buf, sizeof(buf));
            if (n < 0) {
                if (errno == EINTR) {
                    continue; // 被信号打断，重试这次read
                }
                perror("read");
                break;
            }
            if (n == 0) {
                // 对端发送了FIN，正常关闭，本次echo结束
                break;
            }

            ssize_t total_written = 0;
            int write_failed = 0;
            while (total_written < n) {
                ssize_t w = write(conn_fd, buf + total_written, (size_t)(n - total_written));
                if (w < 0) {
                    if (errno == EINTR) {
                        continue; // 被信号打断，重试这次write
                    }
                    perror("write");
                    write_failed = 1;
                    break;
                }
                total_written += w;
            }
            if (write_failed) {
                break;
            }
        }

        close(conn_fd);
    }

    close(listen_fd); // 实际不会执行到这里，除非改成可以被信号安全退出的版本
    return 0;
}
```

逐段说明错误处理和关键选项。

**每个系统调用的返回值处理**：`socket`、`bind`、`listen` 这三步失败都是致命错误——地址被占用、权限不足、fd 耗尽等——直接打印错误并退出整个进程，因为服务根本没有进入可以工作的状态。`accept` 失败则不同：单次 `accept` 出错（比如某些瞬时的资源错误）不应该导致整个服务停止接受新连接，这里选择打印错误后 `continue` 回到循环顶部，让服务继续尝试为其他客户端服务。`read`/`write` 返回负值时要先检查 `errno == EINTR`——这表示系统调用是被信号处理函数打断的，不是真正的 IO 错误，语义上应该直接重试这次调用，而不是当成连接出错处理；`read` 返回 `0` 是一个专门的、和负值完全不同的情形，表示对端已经发送了 `FIN`（正常关闭），要和"出错"区分开单独处理。`write` 还要处理**写不完整**的情况：`write` 的返回值可能小于请求写入的字节数（尤其是数据量较大或者发送缓冲区暂时不够时），必须用一个循环持续调用直到把这次读到的数据全部写完，代码里的 `total_written` 循环就是处理这一点。

**为什么要设置 `SO_REUSEADDR`**：服务进程正常关闭已连接 socket 后，主动关闭的一方会进入 `TIME_WAIT` 状态并停留 2×MSL 的时间（下一节详细展开），这段时间里这个"IP+端口"组合在内核看来还没有完全释放。如果服务端进程重启（比如开发时反复重新编译运行、或者生产环境滚动发布），新进程尝试 `bind` 同一个端口就可能因为旧连接还处于 `TIME_WAIT` 而收到 `EADDRINUSE`（"Address already in use"）错误。设置 `SO_REUSEADDR` 后，内核允许 `bind` 一个本地地址端口即使它当前有连接处于 `TIME_WAIT` 状态，这是开发和生产环境里几乎总要加上这一行的原因。需要说明的是 `SO_REUSEADDR` 只影响 `TIME_WAIT` 状态下的地址复用，不是"允许多个进程同时监听同一端口"（那是 `SO_REUSEPORT` 的语义，是另一个选项）。

---

## 3. 阻塞版本的局限性

**核心结论**：上面这份代码在同一时刻只能真正服务一个客户端——外层的 `accept` 循环和内层的 `read`/`write` 循环共用同一个执行流，当前连接的 `read`/`write` 没有返回之前，程序不会回到 `accept`，也就不会去处理任何其他连接。

具体后果是：如果客户端 A 连上之后迟迟不发数据、或者发送速度很慢，服务端会一直阻塞在对 A 的 `read` 调用上；这期间如果客户端 B、C 也发起连接，它们的三次握手仍然可以由内核独立完成（握手不需要应用程序参与），完成后的连接会被放进 `listen` 设置的全连接队列里排队等待，只要队列没满，B、C 的连接请求本身不会被拒绝，但它们要一直等到服务端处理完 A、回到 `accept` 循环顶部才能被取出并开始真正通信；一旦排队的连接数超过 `backlog`，新到达的连接就会被丢弃或者收到 `RST`（对应第一节讲的行为）。也就是说这个模型的并发能力被压缩到了"接受排队"而不是"同时服务"，队列本身不能替代真正的并发处理能力。

要让服务端能够同时服务多个客户端，传统上有两条思路。

**thread-per-connection / process-per-connection**：每次 `accept` 到一个新连接就 `fork` 一个子进程或者创建一个新线程，专门用这个子进程/线程去跑该连接的 `read`/`write` 循环，主进程/主线程立刻回到 `accept` 循环继续接受下一个连接。这个模型编程模型简单直观，每个连接的处理逻辑仍然是最朴素的阻塞读写，不需要处理复杂的状态机；缺点是当并发连接数很大时，创建和销毁进程/线程本身的开销（内存分配、内核调度结构初始化）、以及大量线程之间频繁的上下文切换开销会成为瓶颈，操作系统能同时维护的线程数也存在上限，这个模型在连接数达到一万甚至十万量级（C10K/C10M 问题）时会明显吃力。

**IO 多路复用**：用 `select`/`poll`/`epoll` 在一个或者少数几个线程里同时管理成千上万个连接的 fd，哪个 fd 上有数据就绪就处理哪个，不需要为每个连接单独占用一个线程。三种多路复用机制的内部实现差异、边沿触发/水平触发的区别，前一篇《计算机网络·IO 模型与多路复用》里已经详细讲过，这里不再重复，下一节直接给出基于 `epoll` 的可运行代码。

**常见追问 / 面试陷阱**

> 追问"这个阻塞版本能不能靠增大 listen 的 backlog 解决并发问题"：不能。`backlog` 只影响排队等待被 `accept` 的连接数上限，不改变服务端"同一时刻只能处理一个连接"这个根本瓶颈；调大 `backlog` 只是让更多客户端能排队而不被立刻拒绝，它们仍然要串行等待前面的连接被处理完，对提升实际吞吐没有帮助，反而可能让客户端等待更久才发现服务响应缓慢。

> 追问"thread-per-connection 和 epoll 该怎么选"：连接数不大、单个连接处理逻辑复杂（阻塞式的业务逻辑、需要调用其他阻塞的库）时，thread-per-connection 的简单直观往往是更好的工程选择；连接数巨大但同一时刻真正活跃、有数据可读写的连接只占少数时（典型的网关、长连接推送场景），`epoll` 的优势会随着连接数增大而越来越明显，因为它的开销只和当前就绪的 fd 数相关，不和线程数、也不和总连接数直接挂钩。生产系统里也常见把两者结合：用少数几个 `epoll` 事件循环线程处理网络 IO，配合线程池处理真正耗 CPU 或者可能阻塞的业务逻辑。

---

## 4. 进阶：基于 epoll 的简化 echo server

**思路**：监听 socket 设为非阻塞并用边沿触发（`EPOLLET`）注册进 `epoll`；事件循环里 `epoll_wait` 返回就绪的 fd 列表后分两种情况处理——如果就绪的是监听 socket 本身，说明有新连接到达，要**循环调用 `accept` 直到返回 `EAGAIN`**（这是上一讲的 ET 规则在"接受连接"这个场景下的具体应用：一次事件通知里内核可能已经把好几个新连接放进了全连接队列，如果只 `accept` 一次就转头处理别的事件，排在后面的连接会因为没有新的边沿跳变而永远等不到下一次通知），每个新连接的 socket 也设为非阻塞并注册进 `epoll`；如果就绪的是某个已连接的客户端 socket，说明有数据可读，同样要**循环调用 `read` 直到返回 `EAGAIN`**，把读到的数据原样写回。

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <netinet/in.h>

#define PORT       8888
#define BACKLOG    128
#define MAX_EVENTS 1024
#define BUF_SIZE   4096

static void die(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

// 把fd设置为非阻塞：ET模式下如果fd是阻塞的，循环read/accept到耗尽时
// 最后一次调用会挂起卡死整个事件循环线程，这一点上一讲已经详细讲过
static void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) die("fcntl(F_GETFL)");
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) die("fcntl(F_SETFL)");
}

static void epoll_add(int epfd, int fd, uint32_t events) {
    struct epoll_event ev;
    memset(&ev, 0, sizeof(ev));
    ev.events  = events;
    ev.data.fd = fd;
    if (epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev) < 0) {
        die("epoll_ctl(ADD)");
    }
}

int main(void) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) die("socket");

    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port        = htons(PORT);

    if (bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) die("bind");
    if (listen(listen_fd, BACKLOG) < 0) die("listen");
    set_nonblocking(listen_fd); // 监听socket也必须非阻塞，配合下面的ET+accept循环

    int epfd = epoll_create1(0);
    if (epfd < 0) die("epoll_create1");

    // 监听socket用ET：一次通知里全连接队列可能已经排了多个连接，
    // 必须循环accept到EAGAIN为止，否则会漏掉排在后面的连接
    epoll_add(epfd, listen_fd, EPOLLIN | EPOLLET);

    struct epoll_event events[MAX_EVENTS];
    printf("epoll echo server (ET) listening on port %d\n", PORT);

    for (;;) {
        int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
        if (n < 0) {
            if (errno == EINTR) continue;
            die("epoll_wait");
        }

        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;

            // 简化处理：生产代码应当在这里单独判断EPOLLERR/EPOLLHUP并
            // 立刻close+从epoll移除；本文为了控制篇幅，统一交给下面
            // read返回0或出错的分支兜底处理，实际效果等价但少了一次
            // 显式的错误分类
            if (fd == listen_fd) {
                for (;;) {
                    struct sockaddr_in cli_addr;
                    socklen_t cli_len = sizeof(cli_addr);
                    int conn_fd = accept(listen_fd, (struct sockaddr *)&cli_addr, &cli_len);
                    if (conn_fd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break; // 这一轮通知对应的新连接已经accept干净
                        }
                        if (errno == EINTR) continue;
                        perror("accept");
                        break;
                    }
                    set_nonblocking(conn_fd);
                    epoll_add(epfd, conn_fd, EPOLLIN | EPOLLET);
                }
                continue;
            }

            // 已连接socket就绪：ET模式下必须循环read直到EAGAIN
            char buf[BUF_SIZE];
            int should_close = 0;

            for (;;) {
                ssize_t len = read(fd, buf, sizeof(buf));
                if (len > 0) {
                    ssize_t total_written = 0;
                    while (total_written < len) {
                        ssize_t w = write(fd, buf + total_written, (size_t)(len - total_written));
                        if (w < 0) {
                            if (errno == EINTR) continue;
                            // 简化处理：真实场景里对端接收缓冲区满会让write
                            // 返回EAGAIN，此时应该注册EPOLLOUT等待可写后再继续
                            // 发送剩余数据，而不是直接放弃连接；这里为了控制
                            // 篇幅，遇到写失败统一按连接异常处理
                            should_close = 1;
                            break;
                        }
                        total_written += w;
                    }
                    if (should_close) break;
                } else if (len == 0) {
                    should_close = 1; // 对端发送FIN，正常关闭
                    break;
                } else {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        break; // 数据已读干净，等待下一次跳变通知
                    }
                    if (errno == EINTR) continue;
                    perror("read");
                    should_close = 1;
                    break;
                }
            }

            if (should_close) {
                epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
                close(fd);
            }
        }
    }

    close(listen_fd);
    close(epfd);
    return 0;
}
```

代码里标注了两处有意简化的地方：一是没有单独处理 `EPOLLERR`/`EPOLLHUP`，统一靠 `read` 的返回值兜底；二是 `write` 遇到对端缓冲区满导致的 `EAGAIN` 时直接放弃连接，而不是注册 `EPOLLOUT` 事件等可写后再补发剩余数据。生产级的事件循环通常会给每个连接维护一个独立的发送缓冲区，`write` 没写完的部分先存进这个缓冲区，同时把该 fd 的关注事件改成 `EPOLLIN | EPOLLOUT`，等 `EPOLLOUT` 就绪时再从缓冲区取数据继续写，这一整套"应用层写缓冲"机制是本篇为了聚焦 `epoll` 事件循环本身而略去的部分。

**常见追问 / 面试陷阱**

> 追问"为什么监听 socket 也要用 ET 并且循环 accept，很多示例代码只对已连接 socket 强调 EAGAIN"：因为 ET 是"状态跳变才通知一次"，监听 socket 的可读事件本质上也遵守这个规则——从"队列空"跳变到"队列非空"只通知一次，如果这次通知到达时全连接队列里已经排了不止一个连接（高并发下短时间内多个客户端几乎同时完成握手是常态），只 `accept` 一次就转去处理其他事件，队列里剩下的连接不会再触发新的跳变通知，会一直卡在队列里得不到处理，表现为"服务端明明在跑但部分客户端连接卡住没反应"。

> 追问"如果这里用水平触发（LT）而不是 ET，代码需要怎么改"：把 `EPOLLIN | EPOLLET` 里的 `EPOLLET` 去掉即可，此时不再要求 fd 必须非阻塞，也不再需要循环 `accept`/`read` 到 `EAGAIN`——LT 模式下只要缓冲区还有数据/队列还有连接，下一次 `epoll_wait` 就会再次通知，读一次不完也不会丢事件。用 LT 实现更不容易因为漏写循环而产生 bug，但通知次数会更多，两者的权衡在上一讲已经详细讨论过。

---

## 5. 常见追问：TIME_WAIT 状态

**核心结论**：TCP 四次挥手里，**先发送 `FIN` 的一方（主动关闭方）**在整个挥手流程走完之后不会立刻释放连接资源，而是进入 `TIME_WAIT` 状态并停留 **2×MSL**（Maximum Segment Lifetime，一个 TCP 报文段在网络中可能存在的最长时间）的时长，之后才彻底关闭。被动关闭方（收到对方 `FIN` 的一方）走完挥手后直接进入 `CLOSED`，没有对应的 `TIME_WAIT` 阶段。

```text
主动关闭方(如Client)                        被动关闭方(如Server)
ESTABLISHED                                   ESTABLISHED
     │───────────── FIN ───────────────────▶│
FIN_WAIT_1                                     │ 收到FIN，应用层可能还没调用close，
     │◀───────────── ACK ────────────────────│ 先回一个ACK，进入CLOSE_WAIT
FIN_WAIT_2                                  CLOSE_WAIT
     │                                          │ 应用层处理完数据后调用close，
     │◀───────────── FIN ────────────────────│ 发出自己这一侧的FIN
     │                                       LAST_ACK
     │───────────── ACK ───────────────────▶│
TIME_WAIT                                    CLOSED（收到ACK后直接释放）
     │
     │  停留 2×MSL
     │
  CLOSED（彻底释放这次连接的资源）
```

**为什么需要 `TIME_WAIT`，停留时长为什么恰好是 2×MSL**，有两个独立的原因，缺一个都不完整。

第一个原因是**保证四次挥手能够正确、可靠地结束**。主动关闭方发出的最后一个 `ACK`（对被动方 `FIN` 的确认）有可能在网络中丢失，如果这个 `ACK` 丢了而主动关闭方已经直接释放了连接资源，被动关闭方等不到 `ACK` 就会重传自己的 `FIN`，此时如果连接已经被释放，主动关闭方要么用一个全新的、没有历史状态的连接去响应这个迟到的 `FIN`（协议行为未定义，通常会回一个 `RST`），要么干脆丢弃它，导致被动关闭方一直重传直到超时，无法正常进入 `CLOSED`。停留 `TIME_WAIT` 期间，连接状态还在，如果收到重传的 `FIN`，主动关闭方能够正确地再发一次 `ACK`。`2×MSL` 的具体构成是：一个 `MSL` 覆盖"自己发出的最后一个 `ACK` 在路上丢失、对方因此重传 `FIN`"所需要的最长时间，另一个 `MSL` 覆盖"这个重传的 `FIN` 到达自己这里"所需要的最长时间，两者相加就是等待期间需要覆盖的最坏情况。

第二个原因是**防止旧连接残留的数据包污染使用相同四元组的新连接**。TCP 连接由（源 IP、源端口、目的 IP、目的端口）这个四元组标识，一次连接结束后，如果立刻用同样的四元组建立一次新连接，网络中理论上还可能存在这次旧连接里滞留、尚未送达或者被重传的过期数据包（比如网络路径异常导致的延迟到达）。如果新连接建立得太快，这些迟到的旧数据包可能被新连接错误地当成自己的数据接收，造成数据错乱。`MSL` 定义了一个报文段在网络中能够存在的最长时间，等待 `2×MSL` 足以保证上一次连接里的所有数据包（包括它们的确认包）都已经在网络中消失，新连接不会再受到旧数据的干扰。

**`SO_REUSEADDR` 的作用**：默认情况下，内核不允许 `bind` 一个仍然有 socket 处于 `TIME_WAIT` 状态的本地地址+端口组合，这是为了避免新的 `bind` 意外收到上一条连接残留的数据包。但在实际开发和运维中，服务进程重启是常态，等待几十秒到几分钟的 `TIME_WAIT` 才能重新监听同一个端口在工程上是不可接受的，因此几乎所有服务端程序在 `bind` 之前都会设置 `SO_REUSEADDR`，允许内核放行这次 `bind`。这正是本文示例代码里在 `bind` 之前调用 `setsockopt(..., SO_REUSEADDR, ...)` 的原因；不设置这个选项，服务端重启后立即监听同一端口大概率会直接遇到 `bind: Address already in use` 报错。

**常见追问 / 面试陷阱**

> 追问"客户端和服务端谁会进入 TIME_WAIT"：取决于谁先发起关闭，和角色（客户端/服务端）本身无关。如果是服务端先调用 `close`（比如服务端主动断开空闲连接），那么服务端进入 `TIME_WAIT`；如果是客户端先关闭（更常见的 HTTP 短连接场景，很多情况下是服务端主动关，取决于协议设计），谁先发 `FIN` 谁就进入 `TIME_WAIT`。在需要频繁建立短连接的高并发服务端场景，如果设计上让服务端总是主动关闭连接，会导致服务端堆积大量 `TIME_WAIT` 状态的连接，占用本地端口和内核资源，这是让客户端主动关闭连接、或者使用连接池维持长连接这类优化思路的直接动机。

> 追问"`SO_REUSEADDR` 会不会导致收到上一条连接的脏数据"：不会。`SO_REUSEADDR` 只是放宽了 `bind` 这一步的地址检查，允许在存在 `TIME_WAIT` 状态 socket 的情况下完成 `bind` 和后续的 `listen`/`accept`，内核仍然会用完整的四元组（包括对端 IP+端口，而不仅仅是本地地址+端口）以及序列号范围去区分新旧连接的数据包，不会把新连接建立之后但仍处于 `TIME_WAIT` 窗口内到达的旧连接数据包错误地交给新连接处理。它解决的是"能不能 `bind`"的问题，不改变 TCP 协议本身用于区分新旧连接数据的机制。

---

## 6. 常见追问：粘包与拆包问题

**核心结论**：TCP 是**面向字节流**的协议，只保证发送方写入的字节按顺序被接收方读到，**不保留发送方每次 `write` 调用之间的边界**。因此接收方一次 `read` 读到的数据，既可能是发送方多次 `write` 的内容被合并到了一起（俗称"粘包"），也可能是发送方一次 `write` 的内容被拆成了接收方好几次 `read` 才读完、甚至一次 `read` 只读到半条逻辑消息（俗称"拆包"或者"半包"），这两种情况在使用 TCP 时都是正常且预期内的行为，不是 bug。

产生这个现象的根本原因是 TCP 协议栈内部有发送缓冲区和接收缓冲区，也有 Nagle 算法这类为了提高网络利用率而做的数据合并优化：发送方连续两次很小的 `write` 调用，协议栈可能把它们攒在一起、用一个 TCP 报文段发出去；接收方的一次 `read` 调用则是"缓冲区里当前有多少可读数据就读多少"，和发送方当初调用了几次 `write`、每次写了多长完全没有关系。换句话说，`write`/`read` 的调用次数和调用边界在 TCP 语义里从来没有被保证过对应关系，应用层如果需要"一条条完整的消息"这种语义，必须自己在发送的字节流里设计消息边界，接收方按照约定好的规则去解析这段连续的字节流，不能假设"发送方写一次，接收方就读到完整的一次"。

应用层解决这个问题的常见方案有两类。

**长度前缀法（length-prefixed framing）**：每条消息发送前，先写入一个固定长度的字段（比如 4 字节的整数，通常转换成网络字节序）表示紧跟在后面的消息体一共有多少字节，接收方先读够这个固定长度的字段，解析出消息体长度，再按照这个长度去读取对应字节数的消息体；如果一次 `read` 没有读够，就把已经读到的部分暂存到应用层维护的缓冲区里，等下一次 `read` 补齐。这个方案对消息内容没有任何限制（不需要转义），是使用最广泛的一种做法，很多 RPC 框架（gRPC、Thrift 等）的底层帧格式都是这个思路。

**分隔符法（delimiter-based framing）**：在每条消息末尾（或者消息头结束处）放一个特殊的分隔符，接收方持续读取字节流，扫描到分隔符就认为一条完整的消息（或者消息头）已经到达，切出这部分交给上层处理，分隔符之后剩余的字节留在缓冲区继续拼接下一条消息。HTTP 协议头部用 ` \n \n` 标记头部结束就是这个思路的典型例子；这个方案的限制是消息内容本身不能包含分隔符，否则需要额外的转义机制。

不管用哪种方案，接收方的实现模式是一致的：维护一个应用层的累积缓冲区，每次 `read` 到新数据就追加进这个缓冲区，然后反复尝试按照协议规则从缓冲区头部切出一条完整的消息（长度前缀法就是检查缓冲区长度是否已经够一个完整消息，分隔符法就是在缓冲区里查找分隔符位置）；能切出完整消息就交给上层处理并从缓冲区移除这部分，切不出来（数据还不够）就停止，等待下一次 `read` 补充更多数据后再重试。

**常见追问 / 面试陷阱**

> 追问"UDP 有没有粘包问题"：没有。UDP 是面向消息的协议，发送方一次 `sendto` 对应接收方一次 `recvfrom`，协议栈会保留每个数据报的边界，即使接收方提供的缓冲区大于实际收到的数据报长度，`recvfrom` 也只会返回这一个数据报的内容，不会把多个数据报的数据拼在一起返回，也不存在跨多次调用才读完一个数据报的情况（如果缓冲区小于数据报长度，多出来的部分会被直接丢弃而不是留到下一次读取）。粘包/拆包是字节流协议特有的问题，UDP 不需要考虑。

> 追问"为什么本文前面的 echo server 代码没有处理粘包问题"：因为 echo server 的语义是"原样把收到的字节吐回去"，它并不关心这些字节在应用层构成了几条逻辑消息，直接照抄读到的字节数写回即可，天然不需要消息边界。粘包问题只在应用层需要识别"一条条独立的业务消息"时才会出现，是协议设计层面的问题，不是 TCP 收发数据本身有缺陷。

---

## 快速选择题

```quiz
title: 快速选择题 1
question: `listen(sockfd, backlog)` 的第二个参数 `backlog` 具体控制的是什么？
answer: B
A. 半连接队列（SYN queue）的最大长度
B. 全连接队列（accept queue）的最大长度，即已完成三次握手、等待accept取走的连接数上限
C. 服务端能同时打开的最大文件描述符数
D. 服务端能同时处理的最大并发线程数
explanation: `backlog` 只控制全连接队列长度，队列里存放三次握手已完成、等待应用程序调用 `accept` 取走的连接；半连接队列的长度由另一个独立的内核参数控制，不受这个参数影响。
```

```quiz
title: 快速选择题 2
question: `accept()` 调用成功后返回的新 fd 和原来的监听 fd 是什么关系？
answer: B
A. 是同一个fd，只是内核内部状态发生了变化
B. 是两个完全独立的fd，新fd专门用于和这个客户端通信，监听fd不受影响继续用于接受新连接
C. 新fd会替换监听fd，原监听fd自动关闭
D. 新fd只在这次accept调用期间有效，函数返回后自动失效
explanation: `accept` 返回一个全新的、独立于监听socket的fd，专用于和这一个客户端的数据收发，原监听socket继续留在原地等待下一次 `accept`，两者是完全独立、生命周期不同的两个fd。
```

```quiz
title: 快速选择题 3
question: 阻塞版echo server在处理某个客户端连接的read/write期间，新到达的客户端连接会怎样？
answer: B
A. 立刻被拒绝
B. 三次握手可以正常完成，连接进入全连接队列排队，只要队列未满就不会被拒绝，但要等当前连接处理完才会被accept取走
C. 会触发服务端自动创建新线程处理
D. 服务端会阻塞崩溃
explanation: 握手由内核独立完成不需要应用程序参与，完成后的连接会进入全连接队列排队；只要队列没满就不会被拒绝，但由于服务端此时阻塞在当前连接的read/write上，这些排队的连接要等服务端回到accept循环才会被真正处理。
```

```quiz
title: 快速选择题 4
question: 为什么示例的阻塞版echo server在bind之前要设置`SO_REUSEADDR`？
answer: C
A. 为了提升read/write的吞吐性能
B. 为了允许多个进程同时监听同一个端口
C. 为了允许bind一个仍有连接处于TIME_WAIT状态的本地地址端口，避免服务重启后立刻遇到Address already in use
D. 为了让listen的backlog参数生效
explanation: `SO_REUSEADDR` 放宽了内核对 `bind` 的地址检查，允许绑定一个当前有 `TIME_WAIT` 状态连接占用的本地地址端口，这是服务频繁重启场景下必须设置的选项；"允许多个进程同时监听同一端口"是 `SO_REUSEPORT` 的语义，与此不同。
```

```quiz
title: 快速选择题 5
question: 用epoll边沿触发（ET）模式实现echo server时，已连接socket就绪后应该怎么读数据？
answer: B
A. 只调用一次read，读多少算多少
B. 用非阻塞fd循环调用read，直到某次read返回EAGAIN/EWOULDBLOCK才停止
C. 用阻塞fd循环调用read直到读不到数据为止
D. 调用一次read后立刻关闭连接
explanation: ET模式只在状态跳变时通知一次，必须配合非阻塞fd循环read到返回EAGAIN为止才能保证把这一轮跳变对应的数据读干净，否则剩余数据会因为没有新的跳变而永远等不到下一次通知。
```

```quiz
title: 快速选择题 6
question: 在ET模式下，监听socket本身就绪时为什么也要循环调用accept直到EAGAIN，而不是只accept一次？
answer: B
A. 因为accept在ET模式下必须调用两次才能成功
B. 因为一次通知对应的全连接队列里可能已经排了不止一个连接，只accept一次会漏掉排在后面的连接，且不会再有新的跳变通知它们
C. 因为ET模式下每次accept返回的fd是临时的，需要反复获取才能得到有效fd
D. 这只是一种性能优化，不循环也不会有正确性问题
explanation: ET的"状态跳变才通知一次"规则同样适用于监听socket的可读事件，高并发下一次通知里队列可能已经有多个连接，不循环到EAGAIN会导致排在后面的连接永远得不到处理。
```

```quiz
title: 快速选择题 7
question: TCP的TIME_WAIT状态发生在哪一方，停留时长通常是多少？
answer: B
A. 被动关闭方（后收到FIN的一方），停留1个MSL
B. 主动关闭方（先发送FIN的一方），停留2倍MSL（Maximum Segment Lifetime）
C. 双方都会进入TIME_WAIT，各停留1个MSL
D. 由操作系统随机决定哪一方进入TIME_WAIT
explanation: 先发送FIN的主动关闭方在四次挥手完成后进入TIME_WAIT，停留2×MSL的时长才彻底释放资源；被动关闭方走完挥手直接进入CLOSED。
```

```quiz
title: 快速选择题 8
question: TIME_WAIT状态存在的原因，下列说法完整的是？
answer: C
A. 只是为了防止对端重传的FIN丢失
B. 只是为了防止旧连接残留的数据包污染使用同一四元组的新连接
C. 两者都是：既要保证最后一个ACK丢失时能正确响应对方重传的FIN，也要保证网络中滞留的旧数据包在新连接建立前彻底消失
D. 只是历史遗留的协议设计，现代网络环境下已经没有实际意义
explanation: TIME_WAIT有两个独立且都必要的作用，缺一个都不完整：保证四次挥手在ACK丢失时能正确收尾，以及防止旧连接的延迟数据包被新连接误收。
```

```quiz
title: 快速选择题 9
question: TCP是面向字节流的协议，这直接导致的现象是？
answer: A
A. 发送方多次write的数据，接收方可能一次read全部读到（粘包），也可能被拆成多次甚至读到半条消息（拆包/半包），因为TCP不保留发送方的写入边界
B. 接收方每次read只能读到发送方一次write写入的全部内容，不多不少
C. TCP会自动在应用层数据前加上长度字段，粘包拆包只在UDP中出现
D. 只有当消息内容超过64KB时才会出现粘包拆包
explanation: TCP只保证字节顺序，不保证保留发送方每次write之间的边界，因此接收方read到的数据边界和发送方write的调用次数、长度没有必然对应关系，需要应用层自己设计消息边界。
```

```quiz
title: 快速选择题 10
question: 应用层解决粘包/拆包问题的常见方案是？
answer: B
A. 把每个TCP连接换成UDP连接即可自动解决
B. 在消息前加固定长度字段表示消息体字节数（长度前缀法），或用特殊分隔符标记消息结束（分隔符法），接收方维护缓冲区按协议规则切分字节流
C. 只要接收方的缓冲区足够大，粘包拆包问题就不会出现
D. 把所有write调用替换成一次性write，就不会再拆包
explanation: 长度前缀法和分隔符法是两种最常见的应用层消息边界方案，接收方需要维护一个累积缓冲区，不断尝试按协议规则从中切出完整消息，切不出来的部分留着等待后续数据补齐；缓冲区大小和write调用方式都不能从根本上消除字节流协议本身没有边界这一事实。
```

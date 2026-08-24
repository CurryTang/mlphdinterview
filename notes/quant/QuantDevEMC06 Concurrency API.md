# Effective Modern C++ 6 · 并发 API（条款 35-40）

这一讲对应《Effective Modern C++》第 7 章 "The Concurrency API" 的六个条款。注意这一章的落脚点不是"并发理论"本身——互斥量、条件变量、死锁、线程池大小怎么定这些通用问题在本项目 OS 系列和性能系列里已经讲过——而是 C++11 标准库把这些概念包装成语言级 API 之后，暴露出来的一组极具 C++ 特色的设计取舍和陷阱：`std::thread` 和 `std::async` 该选哪个、`std::async` 的默认调度策略为什么是个坑、`std::thread` 析构为什么会直接终止程序、`std::future` 的析构行为为什么"看情况"、一次性事件通知该用什么原语、以及 `std::atomic` 和 `volatile` 这两个经常被面试题混着问的关键字到底谁该管并发。这些问题的共同特点是：写出来的代码大概率能编译、能跑、甚至测试时表现正常，但在特定调度时序或特定编译器优化下才会暴露出未定义行为或资源泄漏，属于"平时看不出来，出事就是大事"的一类 bug。

```text
看到 C++ 并发 API 问题该检查什么：
1. 任务需要新开执行流吗？先问"我需要的是拿到返回值/异常，还是纯粹的执行"——前者优先 std::async（任务式），只有确需精确控制线程本身（亲和性、优先级、自定义线程池、生命周期跨作用域）才手写 std::thread（线程式）。
2. 用了 std::async 且没指定 launch policy？默认是 std::launch::async | std::launch::deferred 的组合，实现可以自由选择同步执行（deferred，直到 get/wait 才真正跑）还是异步执行。凡是代码依赖"这个任务一定和调用者并发执行"这个假设，必须显式传 std::launch::async。
3. 代码里裸着 wait_for(0s) 判断任务是否完成？检查返回值里有没有处理 std::future_status::deferred 分支，否则轮询 deferred 任务会死循环。
4. 出现裸的 std::thread 局部变量？追踪它离开作用域前的所有路径（包括异常传播路径），确认每条路径上它要么已经 join()，要么已经 detach()，否则析构时直接 std::terminate()。优先用 RAII 包装类兜底。
5. 用 std::future 且它是 std::async 返回的？确认清楚它是不是某个 std::launch::async 任务的"最后一个引用共享状态的 future"——是的话它的析构会阻塞等任务结束，这是唯一会"隐式 join"的 future 场景。
6. 只是要在两个线程间传递一个"发生了/没发生"的一次性信号，不带数据？别用 atomic<bool> 忙等，也别用条件变量硬凑，优先考虑 std::promise<void> / std::future<void>；但记住它只能 set_value 一次，重复事件通知还是条件变量更合适。
7. 看到一个被多个线程读写的变量声明为 volatile？这几乎总是错的——volatile 既不保证原子性也不保证跨线程的内存序，只能防止编译器对该变量做特定优化，正确的并发原语是 std::atomic。
```

---

## 条款 35：优先选用基于任务而非基于线程的程序设计

**核心结论**：如果目的是"异步执行一个函数并（可能）拿到它的结果"，应该用 `std::async` 提交一个任务，而不是自己创建管理 `std::thread`。前者是任务式（task-based）编程——你只描述"要做什么"，把具体在哪个线程、什么时候跑交给库/运行时决定；后者是线程式（thread-based）编程——你直接管理一条具体的执行流，需要自己操心它的调度、同步、生命周期。

任务式编程的核心优势不是风格上的，而是实实在在省掉了两块本来必须手写的同步逻辑：

**拿返回值**。线程函数没有直接的返回值通道——`std::thread` 的构造函数接受的可调用对象即使有返回值，也没有地方能接住它。要把结果从线程函数传回调用者，得自己搭一套 `promise` + 共享状态，或者用条件变量 + 互斥量 + 标志位手动同步。而 `std::async` 返回一个 `std::future`，调用 `get()` 就拿到结果，这套机制是标准库免费给的：

```cpp
// 线程式：结果传递需要手动搭桥
int result = 0;
std::thread t([&result]{ result = computeSomething(); });
t.join();
use(result);   // 必须确保 join 之后才安全读取 result，且异常无法自然传播出来

// 任务式：future 直接承接返回值
std::future<int> fut = std::async(computeSomething);
use(fut.get());   // get() 阻塞直到结果就绪，返回值类型就是 computeSomething 的返回类型
```

**传播异常**。`std::thread` 的线程函数里如果抛出未被捕获的异常，`std::terminate()` 会被调用，程序直接终止，调用者拿不到任何诊断信息。而 `std::async` 会捕获任务内部抛出的异常，存进共享状态，等调用者对 future 调用 `get()` 时把异常在调用者的线程里重新抛出：

```cpp
std::future<int> fut = std::async([]{
    throw std::runtime_error("task failed");
    return 0;
});

try {
    fut.get();          // 异常在这里被重新抛出，可以正常 catch
} catch (const std::exception& e) {
    handle(e);
}
```

除了这两个"免费拿到"的能力，任务式还带来一个资源管理上的好处：如果任务数量很大或者不可控，为每个任务直接 `new` 一个 `std::thread`，很容易导致超订（oversubscription）——同时存在的线程数远超硬件能有效并行执行的数量，结果是频繁的上下文切换反而拖慢整体吞吐。`std::async` 的默认调度策略把"到底开不开新线程"这个决定权交给了实现，实现原则上可以结合当前系统负载做出更聪明的调度（比如复用线程池、延迟执行）。不过这个默认策略本身也有一个明确的副作用——条款 36 会展开讲。

```text
线程式 vs 任务式（分层关系）
应用代码
   │  "我要异步跑这个函数，并且要结果/异常"
   ▼
std::async（任务抽象层）── 库/运行时自行决定调度方式
   │
   ├── 可能：新开一个 std::thread 立即执行
   ├── 可能：复用某个内部线程池里的线程
   └── 可能：推迟到 get()/wait() 被调用时，在调用者线程上同步执行（deferred）

std::thread（线程抽象层，更底层）── 你自己精确控制这一条执行流
   │  需要自己写：结果传递、异常传播、生命周期管理
   ▼
操作系统线程
```

**要记住**
- `std::async` 提交的是任务而非线程，通过 `std::future` 自然拿到返回值和被传播的异常，无需手写同步原语。
- `std::thread` 函数里未捕获的异常直接调用 `std::terminate()`；`std::async` 任务里的异常会被捕获并在 `get()` 处重新抛出。
- 大量或数量不确定的并发工作若逐个开 `std::thread`，容易造成超订，损害性能；`std::async` 把调度决策权交给实现。
- 仍然存在需要直接操作 `std::thread` 的场景：需要访问底层线程 API（如设置线程优先级、CPU 亲和性）、需要精确控制线程数量与生命周期、或者要自己实现线程池等更底层设施时。

---

## 条款 36：如果异步是必要的，则指定 std::launch::async

**核心结论**：不带参数调用 `std::async(f)` 时，实际使用的启动策略是 `std::launch::async | std::launch::deferred` 的组合，这意味着实现可以在运行时自由选择：要么真的开一个新线程立即异步执行（`async` 策略），要么把任务推迟（`deferred` 策略）——推迟意味着任务根本不会主动跑，直到某个线程对返回的 future 调用 `get()` 或 `wait()`，任务才会在**那个调用者的线程上同步执行**。

这个"看实现心情"的默认行为破坏了一整类关于并发执行的隐含假设：

**轮询判断任务是否完成会死循环**。如果任务被推迟执行，调用 `wait_for` 检查状态不会返回 `timeout`，而是返回 `std::future_status::deferred`——因为任务压根还没开始跑，也不会在后台自己完成。如果轮询逻辑只区分 "ready" 和 "timeout" 两种情况，遇到 deferred 的任务就会永远等不到 timeout，陷入死循环：

```cpp
std::future<int> fut = std::async(slowComputation);   // 默认策略，可能被 deferred

// 错误写法：没有处理 deferred 分支
while (fut.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    // 如果 fut 对应的任务被 deferred，这里永远拿到 std::future_status::deferred，
    // 既不是 ready 也不是 timeout，循环体如果只在这两者间做判断就会死循环或行为异常
    doSomethingElseInTheMeantime();
}
```

正确写法必须显式处理 `deferred` 这个第三种状态：

```cpp
auto status = fut.wait_for(std::chrono::milliseconds(0));
if (status == std::future_status::deferred) {
    // 任务还没跑，此时调用 get()/wait() 会在当前线程同步执行它
    fut.get();
} else if (status == std::future_status::ready) {
    fut.get();
} else {
    // 真正的 timeout：任务在异步执行但还没完成
}
```

**依赖"确实并发执行"的代码会悄悄出错**。例如假设某个 `std::async` 任务和调用者线程真的并行运行、依赖线程本地存储（thread-local）按"每个任务一份"的方式隔离，或者根据墙钟时间估算任务与主线程的重叠执行窗口——如果实现选择了 `deferred`，任务其实是在 `get()`/`wait()` 被调用的那一刻，同步跑在调用者自己的线程上，上述假设全部失效，而且这种失效往往只在特定实现或特定负载下才会出现，非常难复现。

修复方式很直接：只要真的需要异步执行，就显式传入 `std::launch::async`，消除这个不确定性：

```cpp
std::future<int> fut = std::async(std::launch::async, slowComputation);
// 保证任务在新线程上立即开始执行，wait_for 不会返回 deferred
```

**常见追问 / 面试陷阱**

> "`std::async` 不传策略参数，默认策略是什么？" 很多人会答"默认异步执行"，这是错的——默认是 `std::launch::async | std::launch::deferred` 的组合，实现可以选同步也可以选异步。面试官经常追问"这个组合策略下 `wait_for(0s)` 可能返回哪三种状态"，标准答案是 `ready`、`timeout`、`deferred`，漏说 `deferred` 就是没理解这条条款的核心。

**要记住**
- `std::async` 的默认启动策略是 `std::launch::async | std::launch::deferred`，是否真正异步执行由实现在运行时决定，不可预测。
- 一个被 `deferred` 的任务只有在 `get()` 或 `wait()` 被调用时才执行，且是同步执行在调用者的线程上。
- 用 `wait_for` 轮询默认策略的 `std::async` 任务时，必须显式处理 `std::future_status::deferred`，否则可能死循环。
- 只要代码的正确性依赖"任务确实与调用者并发执行"这一假设，就必须显式传入 `std::launch::async`。

---

## 条款 37：使 std::thread 在所有路径上都不可结合

**核心结论**：一个 `std::thread` 对象在任意时刻处于两种状态之一——可结合（joinable，关联着一个正在运行或挂起的底层执行流）或不可结合（unjoinable：默认构造、被移动走、已经 `join()` 过、或已经 `detach()` 过）。这条条款要求的纪律是：一个可结合的 `std::thread` 对象，在它的析构函数运行的那一刻，程序必须已经决定好要 `join` 还是 `detach`——否则析构函数会直接调用 `std::terminate()`，程序终止。

这不是一个可以靠"平时测试通过"来豁免的规则，因为触发条件往往藏在异常传播路径或提前返回路径里：

```cpp
void doWork() {
    std::thread t(backgroundTask);

    if (someCondition()) {
        return;              // 提前返回：t 仍然可结合，析构时 std::terminate()
    }

    mayThrow();              // 如果这里抛异常，栈展开经过 t 的析构，同样 std::terminate()

    t.join();                // 只有走到这里才是安全路径
}
```

标准库设计者面对"析构一个仍可结合的 `std::thread` 该怎么办"这个问题时，权衡了两种"隐式兜底"方案，都认为不可接受：隐式 `join`——会在一个意料之外的位置产生一次阻塞等待，可能造成很难定位的性能挂起；隐式 `detach`——分离出去的线程可能继续访问那些在函数返回后已经被销毁的局部对象，产生比挂起更隐蔽、更危险的悬空引用。两种默认行为都是"安静地做错事"，所以标准库选择了第三条路：既不隐式 `join` 也不隐式 `detach`，而是用 `std::terminate()` 制造一个大声、无法忽视的错误，强制程序员在每条路径上都显式做出选择。

推荐的修复方式是把 `std::thread` 包进一个 RAII 类，析构函数里根据构造时选定的策略调用 `join()` 或 `detach()`，确保异常展开路径也能正确处理：

```cpp
class ThreadRAII {
public:
    enum class DtorAction { join, detach };

    ThreadRAII(std::thread&& t, DtorAction a)
        : action(a), t(std::move(t)) {}

    ~ThreadRAII() {
        if (t.joinable()) {
            if (action == DtorAction::join) t.join();
            else t.detach();
        }
    }

    std::thread& get() { return t; }

    // 显式声明移动语义（自定义析构会阻止编译器隐式生成）
    ThreadRAII(ThreadRAII&&) = default;
    ThreadRAII& operator=(ThreadRAII&&) = default;

private:
    DtorAction action;
    std::thread t;
};

void doWorkSafely() {
    ThreadRAII t(std::thread(backgroundTask), ThreadRAII::DtorAction::join);

    if (someCondition()) return;   // 提前返回也安全：ThreadRAII 析构时会 join
    mayThrow();                     // 异常展开同样安全
}
```

**要记住**
- `std::thread` 只有两种状态：joinable 和 unjoinable；对一个 joinable 的 `std::thread` 析构会直接调用 `std::terminate()`。
- 标准库不提供隐式 join 或隐式 detach，因为两者都可能引入难以调试的性能或正确性问题，宁可用 `terminate` 强制暴露错误。
- 保证"所有路径"析构安全，意味着正常返回路径、提前返回路径、异常传播路径都要覆盖，光在函数末尾写一次 `join()` 不够。
- 用 RAII 包装类统一管理析构时的 join/detach 决策，比在每条控制流路径上手动补 `join()`/`detach()` 更可靠。

---

## 条款 38：对变化多端的线程句柄析构函数行为保持关注

**核心结论**：条款 37 讲的是 `std::thread` 析构的单一、粗暴的规则（可结合就 terminate）；`std::future` 的析构行为则复杂得多——按书里的说法是"有时候像隐式 join，有时候像隐式 detach，大多数时候两者都不像"，具体取决于这个 future 关联的共享状态是怎么来的。

绝大多数 `std::future` 的析构都很"老实"：只是销毁 future 对象自身的成员数据，既不阻塞也不分离任何东西。这适用于来自 `std::packaged_task`、来自 `std::promise` 的 future，以及来自 `std::launch::deferred` 且从未被 `get()`/`wait()` 触发过的 `std::async` future。

唯一的例外，也是这条条款要重点提醒的例外：由 `std::async` 启动、且实际使用了 `std::launch::async` 策略（真正跑在新线程上）的任务，如果它对应的 future 是**最后一个引用该共享状态的 future**，并且任务此时还没完成，那么这个 future 的析构函数会阻塞，直到任务跑完——行为上等价于隐式 `join`。

```cpp
{
    std::future<int> fut = std::async(std::launch::async, slowComputation);
    // ... 不调用 fut.get() 或 fut.wait() ...
}   // fut 离开作用域，触发析构：
    // 如果 slowComputation 还没跑完，这里会阻塞，直到它跑完，才能继续往下执行
```

这个特殊行为的存在是标准库的一种设计补丁：`std::async` 启动异步任务后，没有把底层的 `std::thread` 对象暴露给调用者，调用者手上唯一能拿到的句柄就是这个 `future`。如果 future 析构时什么都不做，那么当程序退出、或者这个共享状态占用的资源需要被清理时，没有任何机制能保证那个异步任务真的已经跑完——于是标准库把"隐式等待"的责任压在了这一种特定的 future 析构函数上。

```text
future 析构行为分类
std::future 析构时……
   │
   ├─ 来自 std::promise / std::packaged_task 的 future
   │      → 正常析构，不阻塞、不分离
   │
   ├─ 来自 std::async(std::launch::deferred, ...) 且从未 get()/wait()
   │      → 正常析构，任务从未真正开始，不阻塞
   │
   └─ 来自 std::async(std::launch::async, ...)，且是最后一个引用该共享状态的 future
          │
          ├─ 任务已完成 → 正常析构，不阻塞
          └─ 任务未完成 → 阻塞等待任务完成（行为等价隐式 join）
```

**常见追问 / 面试陷阱**

> "为什么只有 `std::async` 的 future 会有这种特殊析构行为？" 因为其它来源（`promise`、`packaged_task`）的异步执行流是调用者自己创建和管理的 `std::thread`，线程本身的生命周期问题已经由条款 37 的规则处理；只有 `std::async` 把线程创建这一步完全隐藏在库内部，调用者手里没有 `std::thread` 句柄可以 `join`，future 的析构就成了唯一能补上这个洞的地方。

**要记住**
- `std::future` 的析构大多数情况下只是普通析构，不阻塞、不分离。
- 唯一的阻塞例外：`std::async` 以 `std::launch::async` 策略启动、且该 future 是对应共享状态的最后一个引用、且任务尚未完成——此时析构行为等价隐式 join。
- 这个特殊规则的根源是 `std::async` 不暴露底层 `std::thread`，标准库借 future 析构来保证异步任务不会在无人等待的情况下被"扔在后台"。
- 写依赖 `std::async` 的代码时，不要假设 future 离开作用域一定是"轻量、非阻塞"的操作，尤其是在持有互斥量的临界区里让这类 future 析构，可能引入意外的长时间阻塞。

---

## 条款 39：考虑针对一次性事件通信使用 void future

**核心结论**：当一个线程只需要告诉另一个线程"某件事发生了"，不需要传递任何数据负载时，最省心、最不容易出错的工具是 `std::promise<void>` / `std::future<void>` 这一对，而不是忙等标志位或手搓的条件变量方案。

先看两种常见但都有代价的方案。用 `std::atomic<bool>` 标志位配合忙等循环轮询，虽然实现简单，但是纯软件层面的忙等——没有任何硬件支持（不像自旋锁那样在极短临界区内还算合理），会持续占用 CPU 周期，等待时间稍长就是明显的浪费。用条件变量 + 互斥量 + 布尔标志位的经典组合能避免忙等，但需要正确处理一个细节：等待线程开始 `wait()` 之前，事件可能已经发生、标志位已经被设置过了——如果 `wait()` 的实现不先检查标志位就直接挂起，等待线程会永远等不到已经错过的那次通知（丢失唤醒问题的一种变体）。要写对这套组合，代码大致长这样，而且必须搭配谓词形式的 `wait` 才安全：

```cpp
std::mutex m;
std::condition_variable cv;
bool flag = false;

// 通知方
{
    std::lock_guard<std::mutex> lk(m);
    flag = true;
}
cv.notify_one();

// 等待方：必须用带谓词的 wait，防止 flag 已经被设置却错过通知
std::unique_lock<std::mutex> lk(m);
cv.wait(lk, [&flag]{ return flag; });
```

这套方案能正确工作，但对于"只通知一次、不带数据"这么简单的需求而言，配套的互斥量、谓词判断都是额外的心智负担。

`std::promise<void>` / `std::future<void>` 提供了一个更直接的替代：通知方在事件发生时调用 `set_value()`（`void` 特化，不传任何值），等待方对相应的 `future` 调用 `wait()` 或 `get()`：

```cpp
std::promise<void> p;

// 通知方
void notifyOnce(std::promise<void>& p) {
    doSomeSetupWork();
    p.set_value();          // 事件发生，通知等待方
}

// 等待方
void waitForEvent(std::future<void> fut) {
    fut.wait();              // 阻塞直到 set_value() 被调用；
                              // 如果 set_value() 已经先发生了，wait() 立即返回，
                              // 这个"已经发生过再等待"的场景是内建正确处理的，不需要像条件变量那样自己加判断
    proceedAfterEvent();
}
```

这个方案的代价是共享状态需要堆分配，开销比条件变量方案更高，因此更适合"一次性"的事件通知，而不是频繁触发的信号。这里有一个必须明确的限制：一个 `std::promise` 上 `set_value()` 只能调用一次，多次调用会抛 `std::future_error`；如果事件会反复发生，就不能复用同一对 `promise`/`future`，每次都得重新构造一对，这个开销在高频场景下会超过条件变量方案的固定成本。因此结论是：一次性事件用 `promise<void>`/`future<void>`，重复性、高频的信号通知仍然用条件变量。

**要记住**
- `atomic<bool>` 忙等浪费 CPU；条件变量方案能避免忙等，但需要用带谓词的 `wait` 正确处理"通知先于等待发生"的情况。
- `std::promise<void>` / `std::future<void>` 天然处理"事件已经发生再等待"的场景，且不需要额外的互斥量心智负担。
- 代价是共享状态的堆分配开销，比条件变量方案更重。
- `promise` 的 `set_value()` 只能调用一次，因此这个方案只适合一次性事件，反复触发的信号仍然应该用条件变量。

---

## 条款 40：对并发使用 std::atomic，对特殊内存使用 volatile

**核心结论**：`std::atomic<T>` 和 `volatile` 解决的是两个完全不同的问题，唯一的共同点是都会让编译器"不要随便优化这个变量的访问"，除此之外没有任何交集——这也是这两个关键字在面试和代码审查中最容易被混用出错的原因。

`std::atomic<T>` 提供两个并发正确性所必需的保证：一是原子性，任何线程都不可能观察到该变量处于"写了一半"的中间状态；二是内存序约束，默认的顺序一致（sequentially consistent）语义会阻止编译器把其它内存操作重排到这次原子操作的前后，跨线程可见性因此有了明确保证。这两条加起来，使得 `std::atomic` 成为在不引入额外互斥量的前提下，安全地跨线程读写共享变量的唯一正确工具。

`volatile` 提供的保证与并发毫无关系：它只告诉编译器"不要对这个变量的访问做特定优化"，具体是指不要把它的值缓存在寄存器里跨循环复用、不要因为"看不到后续读取"就把一次看似多余的写操作当成死代码删掉。它的正确使用场景是那些**会在 C++ 抽象机的正常执行流程之外被改变**的内存：内存映射的硬件寄存器（值可能因为硬件活动而改变，编译器完全看不到这个变化）、被信号处理函数修改的内存、以及配合 `setjmp`/`longjmp` 使用的内存——这些场景都和多线程同步没有关系。

一个经典的反面例子最能说明问题：

```cpp
volatile bool ready = false;
int data = 0;

// 线程 A
void producer() {
    data = compute();   // (1)
    ready = true;       // (2)
}

// 线程 B
void consumer() {
    while (!ready) { }  // 忙等，看似"能工作"
    use(data);           // (3)
}
```

这段代码编译能过，很多情况下测试也"看起来正常"，但它是不安全的多线程代码，原因有两层：第一，`volatile` 只约束编译器不要重排/优化对 `volatile` 变量本身的访问，完全不管 CPU 层面的乱序执行和多核之间的缓存一致性时序——线程 B 观察到 `ready == true` 时，并不能保证在别的核上 `(1)` 处对 `data` 的写入已经全局可见，`(3)` 处可能读到旧值；第二，`volatile` 不提供原子性保证，如果换成一个多字的类型，读到"写了一半"的撕裂值也是可能的（`bool` 本身在大多数平台恰好是原子的，容易掩盖这个问题）。用 `std::atomic<bool>` 替换 `volatile bool` 才能同时解决这两层问题：原子性和跨线程的内存序都由标准保证，而不是恰好在特定平台特定编译器下"看起来能用"。

**常见追问 / 面试陷阱**

> "`volatile` 是不是能当作一种轻量级的同步原语？" 不能。这是最常见的误用，本项目 OS 系列和 C++ 系列也都提过这个点：`volatile` 从头到尾就没有"跨线程"这个语义，它只影响编译器如何对待单个变量的访问顺序和是否可以优化掉，和 CPU 层面的指令重排、缓存一致性、原子性都无关。凡是面试题里出现"用 `volatile` 实现一个简单的线程同步标志"，正确答案都是指出这个用法本身是错的，应该换成 `std::atomic`。

**要记住**
- `std::atomic<T>` 保证操作的原子性，并按指定内存序（默认顺序一致）约束编译器对其它内存操作的重排，是并发共享变量的正确工具。
- `volatile` 只保证编译器不优化掉对该变量的访问，不提供原子性，也不提供任何跨线程的内存序或可见性保证。
- `volatile` 的正确用途是内存映射寄存器、信号处理函数修改的内存、`setjmp`/`longjmp` 相关内存，这些都与线程无关。
- 一个 `volatile` 标志位在多线程环境下"看起来能用"不等于它是正确的，CPU 乱序执行和缓存可见性问题不会因为测试没暴露就不存在。

---

## 快速选择题

**1. 关于 `std::async` 相比手写 `std::thread` 的优势，以下哪项说法错误？**
A. `std::async` 返回的 `future` 可以直接拿到任务的返回值
B. `std::async` 任务内抛出的未捕获异常会在调用者 `get()` 时被重新抛出
C. `std::async` 保证任务一定会在新线程上并发执行
D. 大量任务用 `std::async` 相比逐个开 `std::thread` 更不容易造成线程超订

**答案：C** — `std::async` 默认策略允许实现选择同步执行（deferred），并不保证一定并发执行；要保证并发执行必须显式指定 `std::launch::async`。

---

**2. `std::thread` 的线程函数中抛出一个未被捕获的异常，会发生什么？**
A. 异常被自动吞掉，线程正常结束
B. 调用 `std::terminate()`，程序终止
C. 异常被存入某个隐式的共享状态，等待 `join()` 时重新抛出
D. 该异常被转换为一个错误码返回

**答案：B** — `std::thread` 的线程函数没有内建的异常传播机制，未捕获异常会导致 `std::terminate()` 被调用；这正是任务式（`std::async`）编程相对线程式的优势之一。

---

**3. `std::async` 不指定启动策略时，默认策略是什么？**
A. `std::launch::async`
B. `std::launch::deferred`
C. `std::launch::async | std::launch::deferred`
D. 由链接的线程库版本决定，标准未规定组合方式

**答案：C** — 默认策略是 `std::launch::async | std::launch::deferred` 的组合，实现可以在运行时自由选择同步或异步执行任务。

---

**4. 一个使用默认启动策略的 `std::async` 任务被实现选择为 deferred 执行，此时对其 future 调用 `wait_for(0s)` 会返回什么？**
A. `std::future_status::timeout`
B. `std::future_status::ready`
C. `std::future_status::deferred`
D. 抛出异常，因为 deferred 任务不支持 `wait_for`

**答案：C** — deferred 任务在被 `get()`/`wait()` 触发之前根本没有开始执行，`wait_for` 会返回 `std::future_status::deferred`，而不是 `timeout` 或 `ready`；忽略这个分支的轮询循环会死循环。

---

**5. 关于 `std::thread` 对象析构时的行为，以下哪项正确？**
A. 如果该对象是可结合（joinable）的，析构函数会自动调用 `join()`
B. 如果该对象是可结合的，析构函数会自动调用 `detach()`
C. 如果该对象是可结合的，析构函数调用 `std::terminate()`
D. 析构函数总是安全的，无论是否可结合

**答案：C** — 标准库既不隐式 `join` 也不隐式 `detach`（两者都可能引入难以调试的问题），而是强制在析构时调用 `std::terminate()`，逼迫程序员在每条代码路径上显式处理线程的结束方式。

---

**6. 为什么标准库不让 `std::thread` 的析构函数自动 `detach()` 来避免 `terminate`？**
A. 因为 `detach()` 本身在标准里就是未定义行为
B. 因为分离出去的线程可能继续访问已经被销毁的局部对象，产生比挂起更隐蔽的错误
C. 因为 `detach()` 的性能开销比 `join()` 高很多
D. 因为 `detach()` 会导致内存泄漏，比 `terminate()` 更危险

**答案：B** — 隐式 `detach` 的问题在于分离后的线程生命周期完全失控，如果它继续访问外层函数已经销毁的栈上对象，会产生比"隐式 join 导致意外阻塞"更难排查的悬空引用问题，所以标准库选择都不做，而是用 `terminate` 强制暴露错误。

---

**7. 用 RAII 包装 `std::thread`（如条款 37 建议的 ThreadRAII）主要解决的是什么问题？**
A. 提升线程创建的性能
B. 保证线程在所有代码路径（包括异常展开路径）上都被正确 join 或 detach
C. 自动限制同时存在的线程数量，防止超订
D. 让 `std::thread` 支持返回值

**答案：B** — RAII 包装类把"析构时该 join 还是 detach"的决策固化在析构函数里，从而保证正常返回、提前返回、异常传播等所有路径都会被覆盖到，不需要在每条路径手动补调用。

---

**8. 关于 `std::future` 的析构行为，以下哪个场景会像"隐式 join"一样阻塞？**
A. 来自 `std::promise` 的 `future` 析构
B. 来自 `std::packaged_task` 的 `future` 析构
C. 来自 `std::async(std::launch::deferred, ...)` 且从未调用 `get()`/`wait()` 的 `future` 析构
D. 来自 `std::async(std::launch::async, ...)` 且是对应共享状态最后一个引用、任务尚未完成的 `future` 析构

**答案：D** — 这是唯一的特殊情况：`std::async` 以 `async` 策略启动的任务，其对应共享状态的最后一个 future 在析构时，如果任务还没完成会阻塞等待，行为上等价隐式 join；其它三种情况都是普通析构，不阻塞。

---

**9. 为什么只有 `std::async` 启动的任务会有这种特殊的 future 析构阻塞行为，而 `std::promise`/`std::packaged_task` 没有？**
A. 因为 `std::promise`/`std::packaged_task` 的共享状态实现方式不同，不支持阻塞
B. 因为 `std::async` 隐藏了底层的 `std::thread`，调用者手上没有别的句柄能保证任务在程序退出前跑完，只能靠 future 析构兜底
C. 因为 `std::promise`/`std::packaged_task` 从不会关联未完成的异步任务
D. 这是标准委员会的历史遗留不一致，没有设计原因

**答案：B** — `std::promise`/`std::packaged_task` 通常搭配调用者自己创建的 `std::thread`，线程本身的 join/detach 责任已经由条款 37 的规则覆盖；而 `std::async` 不暴露底层线程句柄，只能让 future 析构承担"确保任务跑完"的责任。

---

**10. 想用条件变量实现"一次性事件通知"，为什么需要用带谓词形式的 `wait`（如 `cv.wait(lk, pred)`）而不是裸的 `cv.wait(lk)`？**
A. 带谓词形式性能更好
B. 裸 `wait` 不需要配合互斥量使用
C. 防止事件在等待线程开始等待之前就已经发生，导致等待线程错过通知、永远阻塞
D. 带谓词形式可以支持多个事件同时通知

**答案：C** — 如果不检查标志位就直接挂起等待，事件恰好在 `wait()` 调用之前就已经发生并通知过一次的话，这次通知会被错过，等待线程会一直阻塞；谓词形式先检查条件、不满足才挂起，能正确处理这种时序。

---

**11. `std::promise<void>` / `std::future<void>` 用于一次性事件通知时，相比条件变量方案的主要优势是什么？**
A. 性能开销更低，因为不需要堆分配
B. 天然正确处理"事件已经发生、之后才开始等待"的场景，不需要额外的互斥量和谓词判断
C. 支持无限次重复触发同一个事件
D. 不需要任何形式的线程同步机制

**答案：B** — `future.wait()` 内建正确处理"`set_value()` 已经被调用过"的情况，调用即返回，不需要像条件变量那样自己维护标志位和谓词；代价是共享状态需要堆分配，开销比条件变量更高，因此更适合一次性场景。

---

**12. 一个 `std::promise<void>` 对象，其 `set_value()` 可以被调用几次？**
A. 任意多次，多次调用会被忽略
B. 恰好一次，第二次调用会抛出异常
C. 每个关联的 `future` 调用一次
D. 取决于是否使用了 `std::launch::async`

**答案：B** — `set_value()` 在同一个 `promise` 上只能成功调用一次，第二次调用会抛出 `std::future_error`；这也是这个方案只适合一次性事件、不适合重复通知的原因。

---

**13. `volatile` 和 `std::atomic` 在多线程场景下的本质区别是什么？**
A. `volatile` 保证原子性但不保证内存序，`std::atomic` 两者都保证
B. `volatile` 只约束编译器不对该变量做特定优化，不提供原子性也不提供跨线程内存序；`std::atomic` 两者都提供
C. 两者提供的保证完全相同，只是语法不同
D. `volatile` 用于用户态变量，`std::atomic` 用于内核态变量

**答案：B** — `volatile` 与并发正确性无关，只影响编译器的优化决策（不缓存、不消除访问）；`std::atomic` 才同时提供原子性和内存序保证，是并发共享变量的正确工具。

---

**14. `volatile` 的正确使用场景是以下哪一种？**
A. 多个线程共享的计数器
B. 用互斥量保护的临界区内的标志位
C. 内存映射的硬件寄存器，其值可能因硬件活动而在编译器视野之外发生改变
D. 用于替代 `std::atomic<bool>` 实现线程间的忙等信号

**答案：C** — `volatile` 的设计目的是应对"会在 C++ 抽象机正常执行流程之外被改变"的内存，典型例子是内存映射寄存器、信号处理函数修改的内存、`setjmp`/`longjmp` 相关内存，这些都与多线程同步无关；用 `volatile` 实现线程间信号（如 D 选项）是常见但错误的用法。

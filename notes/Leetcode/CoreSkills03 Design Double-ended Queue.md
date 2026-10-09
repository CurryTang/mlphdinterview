# Design Double-ended Queue

## 面试目标

实现双端队列，支持队头和队尾的插入、删除。常见实现是循环数组或双向链表。

## 核心设计

- 循环数组维护 `front`、`size`、`capacity`。
- 队尾位置可以由 `(front + size) % capacity` 计算。
- `pushFront` 让 `front = (front - 1 + capacity) % capacity`。
- `popBack` 只减少 `size`，无需移动元素。

## 复杂度

- 两端插入和删除：`O(1)`
- 随机访问如果实现 `get(i)`：`O(1)`
- 扩容时复制：`O(n)`，摊还后插入仍为 `O(1)`。

## 常见坑

- 负数取模导致 front 变成负数。
- 队尾下标计算 off-by-one。
- 扩容后没有把元素重新排成从 0 开始的逻辑顺序。

## 延伸应用：单调队列

双端队列不只用来练数据结构实现。它还可以维护一组单调的候选值：新元素从队尾进入，失效元素从队首离开。这个模式可以在线性时间内求出每个固定窗口的最大值。

完整推导和代码见 [[CoreSkills16 Sliding Window|Sliding Window 第五题：Sliding Window Maximum]]。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

循环数组版本只保存 `front` 和 `size`。逻辑下标 `i` 映射到物理下标 `(front + i) % capacity`。

```text
pushFront(x):
  grow if full
  front = (front - 1 + capacity) % capacity
  data[front] = x
  size += 1

pushBack(x):
  grow if full
  data[(front + size) % capacity] = x
  size += 1
```

扩容时按逻辑顺序复制 `data[(front+i)%old_capacity]` 到新数组的 `i`，然后把 `front` 重置为 0。

</details>

---

## 实战应用：Trailing Window Maximum（时序流式暖机窗口最大值）

双端队列在工业系统、量化交易与在线时序特征工程中最经典的高频算法应用，正是**单调双端队列（Monotonic Deque）**维护时序滑动窗口极值。

### 题目定义

> **Trailing / Rolling Window Maximum**：
> 给定一个实数时序流 `xs`（长度为 $N$）与最大回看时间窗口大小 $n$。
> 在每一个时间步 $t \in [0, N-1]$，窗口覆盖历史范围为 $[ \max(0, t - n + 1), \; t ]$（窗口尺寸 $\le n$）。
> 要求实时输出截至时间戳 $t$ 为止该回看窗口内的最大值。返回长度同样为 $N$ 的结果列表。

```python
def rolling_max(xs: List[float], n: int) -> List[float]: ...
```

### 与标准 LeetCode 239（固定窗口）的核心区别

| 维度 | 标准 LeetCode 239 (Sliding Window Maximum) | Trailing / Rolling Window Maximum (时序流式) |
|---|---|---|
| **输出时机** | 必须等待窗口填满 $k$ 个元素（$t \ge k - 1$）才开始输出 | **每个时刻 $t$ 都必须立即产生一个输出**（包含暖机阶段） |
| **窗口跨度** | 严格固定为 $k$ | 前 $n-1$ 个点窗口动态从 $1$ 逐渐膨胀至 $n$（Warm-up 暖机阶段） |
| **输出长度** | $N - k + 1$ | 严格等于输入长度 $N$ |
| **工业场景** | 离线批量切片 | 实时在线风控、量化交易信号、特征工程因果无前瞻（Causal Streaming） |

### 核心不变量与操作步骤

双端队列中**仅保存时间戳下标 $t$**，维持两大核心不变量：
1. **时序严格递增**：队首到队尾下标单调递增；
2. **数值单调递减**：队首对应的数值最大，队首 $q[0]$ 恒为当前有效窗口内的最值。

在每个时间戳 $t$ 收到数值 $val = xs[t]$ 时，执行四步标准流程：
1. **队首过期淘汰**：
   当前窗口左界为 $t - n + 1$。因此凡是下标 $\le t - n$（即 $< t - n + 1$）的历史索引均已过期，从队首弹出：`while q and q[0] <= t - n: q.popleft()`；
2. **队尾单调淘汰（支配性原理）**：
   若队尾元素 $xs[q[-1]] \le val$，说明它既比新元素更早进入、数值又小于等于新元素，在未来绝无可能成为窗口最大值，立即从队尾弹出：`while q and xs[q[-1]] <= val: q.pop()`；
3. **当前元素入队**：
   `q.append(t)`；
4. **即时记录当前最大值**：
   队首 $xs[q[0]]$ 即为当前窗口内的最大值，`res.append(float(xs[q[0]]))`。

### 工业级实现代码

```python
from collections import deque
from typing import List

def rolling_max(xs: List[float], n: int) -> List[float]:
    """
    Trailing window maximum with window size <= n.
    
    Parameters:
        xs: 输入浮点时序列表，长度为 N
        n:  最大回看窗口长度 (n >= 1)
        
    Returns:
        与 xs 等长的浮点列表，res[t] 代表 xs[max(0, t - n + 1) : t + 1] 的最大值
    """
    if not xs or n <= 0:
        return []

    q = deque()  # 存放下标，严格维持对应数值自顶向下递减
    res: List[float] = []

    for t, val in enumerate(xs):
        # 1. 队首过期检查：有效窗口左闭右闭区间为 [t - n + 1, t]
        # 下标 <= t - n 已经滑出窗口
        while q and q[0] <= t - n:
            q.popleft()

        # 2. 队尾淘汰劣质候选：比当前值小或相等的历史元素永无翻盘机会
        while q and xs[q[-1]] <= val:
            q.pop()

        # 3. 当前下标入队
        q.append(t)

        # 4. 队首即为当前窗口 [max(0, t - n + 1), t] 内的最大值
        res.append(float(xs[q[0]]))

    return res

if __name__ == "__main__":
    # 示例 1: 基础递增与回撤 (n=3)
    data = [1.0, 3.0, -1.0, -3.0, 5.0, 3.0, 6.0, 7.0]
    expected = [1.0, 3.0, 3.0, 3.0, 5.0, 5.0, 6.0, 7.0]
    assert rolling_max(data, 3) == expected
    print("✅ rolling_max test passed!")
```

### 复杂度分析

- **时间复杂度**：严格 $\mathcal{O}(N)$。每个索引 $t \in [0, N-1]$ 入队恰好一次（`append`），在后续的整个执行过程中最多被队首弹出一次或队尾弹出一次（出队总次数 $\le N$）。因此 `while` 循环的总摊还操作次数为 $\mathcal{O}(N)$，平均每个时间步耗时 $\mathcal{O}(1)$。
- **空间复杂度**：双端队列在任意时刻最多容纳 $\min(n, N)$ 个有效下标，额外空间复杂度严格为 $\mathcal{O}(\min(n, N))$。



## 实战应用二：基于滑动时间窗口的重复记录检测器

### 题目定义
流式处理日志中，判定当前事件在最近 `window_sec` 秒内是否已发生过重复记录。

### 核心心智（双端队列维护时间窗 + 哈希表维护频次）
`deque` 存放 `(timestamp, event_id)`，每次新事件到来时：
1. 从队头 `popleft()` 弹出所有超出时间窗的过期事件，并在哈希频次表中将其计数减 1（计数降为 0 则删除）；
2. 检查当前 `event_id` 是否在哈希表中（存在即为重复）；
3. 将当前事件写入队尾与哈希表。

```python
from collections import deque, defaultdict

class SlidingWindowDuplicateDetector:
    def __init__(self, window_sec: int):
        self.window_sec = window_sec
        self.queue = deque()  # (timestamp, event_id)
        self.counts = defaultdict(int)

    def is_duplicate(self, timestamp: int, event_id: str) -> bool:
        # 1. 淘汰队首过期记录
        while self.queue and self.queue[0][0] <= timestamp - self.window_sec:
            _, old_id = self.queue.popleft()
            self.counts[old_id] -= 1
            if self.counts[old_id] == 0:
                del self.counts[old_id]

        # 2. 检查并记录
        duplicate = event_id in self.counts
        self.queue.append((timestamp, event_id))
        self.counts[event_id] += 1
        return duplicate
```

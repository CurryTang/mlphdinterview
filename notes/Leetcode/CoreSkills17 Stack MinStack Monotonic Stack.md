# Stack · MinStack 与单调栈

栈的核心 API 非常精简，其高频应用主要收敛为两大范式：

```text
MinStack：入栈时保存历史快照，使极值查询无须回溯扫描
单调栈：维护未解元素的候选单调集，等待右侧元素触发消除并确定边界
```

MinStack 是独立的状态快照设计；单调栈则是处理“一维序列最近极值边界”的完整题族。掌握单调栈的关键在于建立**“三问四槽”万能解题模板**，在不同题目中仅需替换比较谓词与结算槽位即可完成映射。

## 学习顺序

题目精选自典型高频核心题型，建立从状态快照到单调栈全域边界的递进理解：

| 顺序 | 原题 | 核心模型 | 解决的关键问题 |
|---:|---|---|---|
| 1 | [155. Min Stack](https://neetcode.io/problems/minimum-stack/question?list=neetcode150) | 状态快照模式 (State-Snapshot) | 每个栈帧绑定 `min_so_far` 实现 $O(1)$ 查询与回滚 |
| 2 | [739. Daily Temperatures](https://neetcode.io/problems/daily-temperatures/question?list=neetcode150) | 单调栈·右侧边界 (Next Greater) | 槽位 3A：弹栈时根据下标差结算等待距离 |
| 3 | [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/) | 单调栈·循环数组 (Circular Array) | 槽位 1：$2n$ 取模倍增，无缝复用单调栈模板 |
| 4 | [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150) | 单调栈·双侧边界 (Dual Boundaries) | 槽位 1 注入尾部哨兵，槽位 3A 弹栈时同步锁定左右边界 |
| 5 | [42. Trapping Rain Water](https://neetcode.io/problems/trapping-rain-water/question?list=neetcode150) | 单调栈·凹槽横向注水 (Trough Fill) | 弹栈确定底部，新栈顶与当前柱围成横向矩形槽 |

---

## 模块一：MinStack

### 1.1 · 状态快照模式：给每个栈帧记录当时的极值

如果仅维护单个全局变量 `minimum`，在 `push` 时更新很容易，但一旦弹出当前的最小值，系统无法获知上一个历史最小值，重新扫描全栈需耗费 $O(n)$ 时间。

工程上最鲁棒的实现是**状态快照模式（State-Snapshot Pattern）**：让每个栈帧存储一对元组：

```text
(当前值 value, 压入该值后的历史最小值 min_so_far)
```

此时栈顶天然携带双重信息：

```text
top()    = stack[-1][0]
getMin() = stack[-1][1]
```

每次 `push` 相当于对当前时刻的极值状态打一张快照；`pop` 弹出快照后，下方栈帧保存的历史极值自然显露，无需任何额外的状态回滚或同步锁机制。

### 1.2 · 为什么重复最小值不会导致状态失真

设依次压入元素 `[2, 1, 1]`：

```text
1. push(2) -> stack: [(2, 2)]
2. push(1) -> stack: [(2, 2), (1, 1)]
3. push(1) -> stack: [(2, 2), (1, 1), (1, 1)]
```

当执行一次 `pop()` 弹出栈顶的 `1` 后，栈变为 `[(2, 2), (1, 1)]`，栈顶快照仍精确记录最小值为 `1`。
若使用独立辅助栈且仅在“严格更小”时入栈，遇到重复最小值时必须在 `pop` 中额外编写条件判断，极易引发同步错位。元组快照法在代数上完全自洽且分支最小。

### 1.3 · Quick Coding：实现 MinStack

实现 `push`、`pop`、`top` 和 `getMin`，要求时间复杂度均为 $O(1)$。

<details>
<summary>参考实现（状态快照法）</summary>

```python
class MinStack:
    def __init__(self):
        # 栈内存储 (val, min_so_far)
        self.stack = []

    def push(self, val: int) -> None:
        current_min = val if not self.stack else min(val, self.stack[-1][1])
        self.stack.append((val, current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

- **时间复杂度**：所有操作均为严格 $O(1)$。
- **空间复杂度**：为每个元素维护快照元组，空间复杂度为 $O(n)$。

</details>

### 1.4 · 进阶扩展：空间最优的差值编码法 (Value-Difference Encoding)

在内存极其严苛的嵌入式或高频数据流场景中，如果不希望为每个元素分配额外的元组对象，可以使用**差值编码法**，仅使用一个栈与单变量 `min_val` 完成：

1. 栈内存储当前值与当前最小值的差值：`diff = val - min_val`。
2. 当 `val < min_val` 时，入栈的 `diff < 0`，随后更新 `min_val = val`；
3. 出栈时若观测到 `diff < 0`，说明弹出的正是当前最小值，前驱最小值为 `min_val = min_val - diff`。

该方法将辅助空间压缩至严格极限，但在高并发多语言实现中需要考虑整数溢出（如 C++ `long long` 防御）。

---

## 模块二：单调栈

### 2.1 · 核心动力学机制与解决的问题

单调栈（Monotonic Stack）最核心解决的问题是：

> **在一维序列中，为每个位置寻找其左侧或右侧“最近”的更大或更小元素（Nearest Greater / Smaller Element）。**

常见题型虽然表述各异，本质上均在寻找这种**局部最近边界**：

| 题目表述 | 实际数学本质 | 对应单调栈行为 |
|---|---|---|
| 还要几天才会升温？ | 右侧第一个严格更大元素（Next Greater） | 答案记录为下标差 $i - j$ |
| 下一个更大元素是谁？ | 循环数组右侧第一个更大元素（Next Greater） | 答案记录为元素值 $nums[i]$ |
| 柱子最多能向两边延伸多远？ | 左右两侧第一个更小元素（Dual Smaller Boundaries） | 弹栈时一次性锁死左右边界 |
| 接雨水能装多少？ | 左右两侧更高的柱子形成封闭凹槽 | 弹栈确定槽底，栈顶与当前柱定高 |
| 连续子数组的最小值之和？ | 以当前值为最小值的极大覆盖区间 $[L+1, R-1]$ | 双侧极值边界与去重防漏乘积 |

#### 为什么暴力法是 $O(n^2)$，而单调栈是 $O(n)$？
暴力法对于每个位置都需要向左或向右线性扫描，最坏情况下（如单调有序数组）产生双重循环 $\sum_{i=1}^n i = O(n^2)$。
单调栈通过维护一个**“未解下标等待集合”**：
- 每个下标**仅入栈一次**；
- 每个下标**至多出栈一次**；
- 整个处理过程中的总 `push` 与 `pop` 操作次数严格 $\le 2n$ 次，因此总时间复杂度严格均摊为 $O(n)$。

#### 栈内为什么必须严格存储“下标”而非“数值”？
在单调栈中，永远优先存储**下标（Index）**，因为一个下标 $j$ 同时具备三项完备信息：
1. **原值寻址**：通过 $nums[j]$ 随时获知其数值大小；
2. **跨度度量**：通过当前坐标 $i$ 计算几何跨度 $\Delta = i - j$ 或矩形宽度 $W = R - L - 1$；
3. **精准写回**：直接定位结果数组槽位 $ans[j]$，无需通过哈希表二次反查，消除键冲突问题。

---

### 2.2 · 单调栈万能通用模板：三问四槽模型 (Universal Monotonic Stack Blueprint)

单调栈所有题型共享同一套解题思维流水线与骨架结构：

```text
                           ┌────────────────────────────┐
                           │   for i, current in arr:   │
                           └─────────────┬──────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
                           │ 槽位 1 [哨兵与初始化]       │
                           │ arr = nums + [0] (可选)    │
                           └─────────────┬──────────────┘
                                         │
                                         ▼
                           ┌────────────────────────────┐
        ┌─────────────────►│ 槽位 2 [弹栈条件断言]       │
        │                  │ while stack and pop_cond:  │
        │                  └──────┬──────────────┬──────┘
        │                         │ 满足         │ 不满足
        │                         ▼              ▼
        │             ┌───────────────────────┐  ┌────────────────────────┐
        │             │ mid = stack.pop()     │  │ 槽位 3B [左侧最近结算]  │
        │             │                       │  │ ans[i] = stack[-1]  │
        │             │ 槽位 3A [右/双侧结算] │  └───────────┬────────────┘
        │             │ ans[mid] = i / 矩形面积│              │
        │             └───────────┬───────────┘              │
        │                         │                          │
        └─────────────────────────┘                          ▼
                                                 ┌────────────────────────┐
                                                 │ 槽位 4 [当前下标入栈]  │
                                                 │ stack.append(i)        │
                                                 └────────────────────────┘
```

#### 1. 三问定型法（Three Clarification Questions）

- **Q1：方向与边界归属（Direction & Attribution）**
  - **右侧最近边界（Next-X）**：当前元素 $i$ 作为“回答者”，当其破坏单调性时，弹出栈顶 $j$，并将当前 $i$ 结算给被弹出元素 $j$（**在 `while` 内部槽位 3A 结算**）。
  - **左侧最近边界（Prev-X）**：当前元素 $i$ 作为“被查询者”，用 `while` 清除所有无法成为有效答案的无效候选；循环结束后，留在栈顶的元素即为 $i$ 的左侧最近边界（**在 `while` 外部槽位 3B 结算**）。
  - **双侧全域边界（Dual Boundaries）**：当 $mid = stack.pop()$ 发生时：
    - 当前扫描元素 $i$ 是其**右侧第一个更小/更大值**（右边界 $R = i$）；
    - 弹出后此时暴露的新栈顶 $stack[-1]$ 则是其**左侧第一个更小/更大值**（左边界 $L = stack[-1]$）；
    - 一次弹栈直接获取 $mid$ 的两端极值边界，覆盖有效区间为 $[L + 1, R - 1]$。

- **Q2：大小关系与去重防漏定理（Strictness & Tie-Breaking Theorem）**
  - **求更大元素**：当前值大于栈顶时弹栈，栈底到栈顶保持单调递减。
  - **求更小元素**：当前值小于栈顶时弹栈，栈底到栈顶保持单调递增。
  - **去重防漏定理（Exact Partitioning Theorem）**：
    - 当原数组存在**重复元素**且题目要求统计所有子数组贡献（如 LC 907、LC 84）时：
      - 若左右两侧均取严格大小关系（`<` 与 `>`），相等元素之间的区间会被**漏算（Under-counting）**；
      - 若左右两侧均取非严格大小关系（`<=` 与 `>=`），相等元素之间的区间会被**重算（Over-counting）**；
      - **黄金原则**：必须且只能设定为**一侧严格（如左侧严格更小 `<`）、另一侧非严格（如右侧小于等于 `<=`）**，从而构成左开右闭或左闭右开的互斥完备子集划分。

- **Q3：存储载体与结算形式（Storage & Answer Form）**
  - 栈内恒存下标。答案槽位依据题意写入：下标值、距离天数 ($i - j$)、扩展宽度 ($R - L - 1$) 或矩形乘积面积。

---

#### 2. 四槽通用代码骨架（Universal Template Code）

```python
from typing import List, Optional


def universal_monotonic_stack(
    nums: List[int],
    mode: str = "next_greater",  # "next_greater" | "next_smaller" | "prev_greater" | "prev_smaller" | "dual_smaller"
    with_sentinel: bool = False,
    sentinel_val: int = 0,
) -> List[int]:
    """单调栈通用解题骨架 (Universal Monotonic Stack Blueprint)

    4 个参数化槽位：
    [槽位 1] 哨兵与初始化: 初始化答案数组与边界哨兵，消除清栈分支
    [槽位 2] 弹栈判定谓词: 当前值与栈顶历史值的关系断言
    [槽位 3] 结算动作:
             - 槽位 3A: 弹栈时结算 (右侧边界或双侧扩展极值)
             - 槽位 3B: 弹栈后结算 (左侧最近有效边界)
    [槽位 4] 入栈等待: 当前下标压入栈中开始等待右侧触发
    """
    n = len(nums)

    # ──────────────────────────────────────────────────────
    # 槽位 1: 哨兵与容器初始化
    # ──────────────────────────────────────────────────────
    arr = nums + [sentinel_val] if with_sentinel else nums
    limit = len(arr)
    ans = [-1] * n
    stack = []  # 严格保存下标

    # ──────────────────────────────────────────────────────
    # 槽位 2: 弹栈比较谓词 (当前值是否打破栈顶单调性)
    # ──────────────────────────────────────────────────────
    def should_pop(top_val: int, curr_val: int) -> bool:
        if mode in ("next_greater", "prev_greater"):
            return curr_val > top_val
        elif mode in ("next_smaller", "prev_smaller", "dual_smaller"):
            return curr_val < top_val
        return False

    for i in range(limit):
        curr_val = arr[i]

        while stack and should_pop(arr[stack[-1]], curr_val):
            mid = stack.pop()

            # ──────────────────────────────────────────────────
            # 槽位 3A: 弹栈时结算 (右侧边界 / 双侧极值边界)
            # ──────────────────────────────────────────────────
            if mode.startswith("next") and mid < n:
                ans[mid] = i  # 或计算跨度: i - mid
            elif mode == "dual_smaller":
                left = stack[-1] if stack else -1
                right = i
                width = right - left - 1
                # 执行双侧几何聚合，如 ans = max(ans, arr[mid] * width)

        # ──────────────────────────────────────────────────────
        # 槽位 3B: 弹栈后结算 (左侧最近有效边界，当前值使用最新栈顶)
        # ──────────────────────────────────────────────────────
        if mode.startswith("prev") and i < n:
            ans[i] = stack[-1] if stack else -1

        # ──────────────────────────────────────────────────────
        # 槽位 4: 当前下标入栈
        # ──────────────────────────────────────────────────────
        stack.append(i)

    return ans
```

---

### 2.3 · 比较符号与状态对照表

记 `top = nums[stack[-1]]` 为栈顶历史值，`current = nums[i]` 为当前扫描值：

| 查找目标 | 弹栈触发条件 (`while`) | 弹栈后栈内值分布（栈底 → 栈顶） | 适用场景 |
|---|---|---|---|
| **右侧第一个严格更大** | `top < current` | 单调不增（从大到小） | 每日温度、下一个更大元素 |
| **右侧第一个大于等于** | `top <= current` | 严格递减 | 消除重复元素的右侧阻挡 |
| **右侧第一个严格更小** | `top > current` | 单调不减（从小到大） | 柱状图最大矩形、子数组极小值 |
| **右侧第一个小于等于** | `top >= current` | 严格递增 | 去重防漏半开半闭区间统计 |

> **核心记忆法则**：
> 不要死记“维护递增还是递减栈”。直接提问：**“当前扫描到的元素，是否已经满足了栈顶元素在等待的目标？”**
> 一旦满足，立即弹栈并执行结算。

下面的交互演示用同一个数组执行“右侧更大”与“右侧更小”的逐步演算：

```monotonic-stack-demo
```

---

### 2.4 · 经典题目万能模板填装对照表 (Slot-Filling Matrix)

面对任何题目，直接将业务参数代入“四槽模型”：

| 经典题目 | 模式分类 | 槽位 1 (哨兵) | 槽位 2 (弹栈谓词) | 槽位 3 (结算时机与计算) | 槽位 4 (入栈) |
|---|---|---|---|---|---|
| **LC 739. Daily Temperatures** | Next Greater | 无 | `top < current` | 槽位 3A：`ans[mid] = i - mid` | `stack.append(i)` |
| **LC 496. Next Greater Element I** | Next Greater | 无 | `top < current` | 槽位 3A：`ans[mid] = current` | `stack.append(i)` |
| **LC 503. Next Greater Element II** | 循环 Next Greater | $2n$ 遍历取模 | `top < nums[i % n]` | 槽位 3A：`ans[mid] = nums[i % n]`（当 `mid < n`） | `if i < n: stack.append(i)` |
| **LC 84. Largest Rectangle** | Dual Smaller | 尾部加 `0` | `top > current` | 槽位 3A：`w = i - stack[-1] - 1`<br>`ans = max(ans, heights[mid] * w)` | `stack.append(i)` |
| **LC 42. Trapping Rain Water** | Dual Greater | 无 | `top < current` | 槽位 3A：凹槽高度差乘以宽度<br>`h = min(top, current) - mid_h` | `stack.append(i)` |
| **LC 907. Subarray Minimums** | Dual Smaller (去重) | 尾部加 `0` | 左严格 `<`，右侧 `<=` | 槽位 3A：乘法原理计算子数组数<br>`count = (mid - left) * (right - mid)` | `stack.append(i)` |

---

### 2.5 · 题型实战与模板代入

#### 实战一：Daily Temperatures（每日温度）
给定每日气温，求出每一天需要等待多少天才会遇到更高气温。
- **模板映射**：属于典型 **Next Greater** 模式，答案形式由下标变为跨度差值 $i - mid$。

```python
from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []  # 保存下标

        for i, temp in enumerate(temperatures):
            # 槽位 2: 当前温度高于栈顶，触发弹栈
            while stack and temperatures[stack[-1]] < temp:
                mid = stack.pop()
                # 槽位 3A: 弹栈结算，答案取距离差
                ans[mid] = i - mid
            # 槽位 4: 当前天数入栈等待
            stack.append(i)

        return ans
```

#### 实战二：Next Greater Element II（循环数组的取模倍增）
给定循环数组，寻找每个元素的下一个更大值。
- **模板映射**：循环数组只需在**槽位 1** 中将遍历范围扩展至 $2n$，下标取模 `i % n`，且仅在第一轮 $i < n$ 时执行压栈。

```python
from typing import List


class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [-1] * n
        stack = []

        # 槽位 1: 虚拟倍增循环 2*n
        for i in range(2 * n):
            val = nums[i % n]
            # 槽位 2: 弹栈比较
            while stack and nums[stack[-1]] < val:
                mid = stack.pop()
                # 槽位 3A: 弹栈结算实际数值
                ans[mid] = val
            # 槽位 4: 仅前 n 个元素需要压栈求解
            if i < n:
                stack.append(i)

        return ans
```

#### 实战三：Largest Rectangle in Histogram（双侧边界与尾部哨兵）
给定柱状图高度数组，求能勾勒出的最大矩形面积。
- **关键推导**：以某根柱子 $heights[mid]$ 作为矩形高度时，其宽度向左右延伸的最大范围受限于**左右两侧第一个严格更矮的柱子**：
  $$W = \text{right} - \text{left} - 1$$
- **哨兵收益**：在末尾追加高度为 `0` 的哨兵，可强制清空栈中所有遗留柱子，避免在循环结束后编写冗长的清栈特判代码。

```largest-rectangle-demo
```

```python
from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        max_area = 0
        stack = []

        # 槽位 1: 尾部注入高度为 0 的哨兵
        for right in range(len(heights) + 1):
            curr_h = 0 if right == len(heights) else heights[right]

            # 槽位 2: 当前高度更矮，破坏单调递增性
            while stack and heights[stack[-1]] > curr_h:
                mid = stack.pop()
                mid_h = heights[mid]
                # 槽位 3A: 双侧边界同时确立
                left = stack[-1] if stack else -1
                width = right - left - 1
                max_area = max(max_area, mid_h * width)

            # 槽位 4: 下标入栈
            stack.append(right)

        return max_area
```

#### 实战四：Trapping Rain Water（接雨水·单调栈横向凹槽法）
- **核心模型**：利用单调栈维护递减序列。当遇到更高柱子时，弹出栈顶作为凹槽底部 $mid$；新的栈顶即为左侧支柱 $left$，当前柱即为右侧支柱 $right$。
- **计算公式**：横向水槽的水位高度取决于木桶效应，宽度为两柱间距：
  $$H = \min(height[left], height[right]) - height[mid]$$
  $$W = right - left - 1$$
  $$\text{Water} = H \times W$$

```python
from typing import List


class Solution:
    def trap(self, height: List[int]) -> int:
        water = 0
        stack = []

        for right, curr_h in enumerate(height):
            # 槽位 2: 遇到更高柱子，破坏递减形成凹槽
            while stack and height[stack[-1]] < curr_h:
                mid = stack.pop()
                if not stack:
                    break  # 左侧无边界，无法积水

                left = stack[-1]
                # 槽位 3A: 横向切片注水
                h = min(height[left], curr_h) - height[mid]
                w = right - left - 1
                water += h * w

            # 槽位 4: 当前柱入栈
            stack.append(right)

        return water
```

---

### 2.6 · 复杂度证明：为什么嵌套 while 循环仍然是 O(n)？

许多初学者容易对嵌套在 `for` 循环内的 `while` 产生 $O(n^2)$ 的误判。

严格的**聚合分析（Aggregate Analysis / 摊还复杂度）**证明如下：
1. 数组长度为 $n$，每个下标进入外层 `for` 循环至多被 `stack.append()` **执行 1 次**；
2. 元素只有存在于栈内时，才可能在 `while` 内部被 `stack.pop()` **执行至多 1 次**；
3. 一旦某个下标被出栈弹出，它便彻底脱离生命周期，后续遍历中绝不可能再次入栈或被重复弹出；
4. 因此，跨越所有 $n$ 次外层迭代，内层 `while` 条件成立并执行 `pop()` 的**物理总次数上限严格为 $n$ 次**。

$$\sum_{i=1}^n (\text{push 次数} + \text{pop 次数}) \le n + n = 2n = O(n)$$

因此，单调栈的全局时间复杂度严格为 $O(n)$，空间复杂度因最坏情况下需保存全量单调下标而为 $O(n)$。

---

### 2.7 · 面试结构化应答清单

在白板或线上编码面试中，单调栈的答题推进可严格按以下 5 步展开：

1. **定型声明**：“本题需要为每个元素寻找单侧/双侧最近极值边界，暴力为 $O(n^2)$，最优结构为单调栈，时间复杂度降至 $O(n)$。”
2. **载体确认**：“栈内保存下标而非数值，因为后续计算跨度 $i - j$ 和面积需要物理距离。”
3. **谓词说明**：“求右侧更大元素，因此栈内维持单调递减；一旦遇到严格更大值即触发弹栈。”
4. **结算归属**：“答案在弹栈时写给旧元素（槽位 3A），因为当前元素扮演的是‘回答者’角色。”
5. **边界哨兵**：“对于柱状图或多区间计算，在数组尾部注入哨兵 `0`，确保栈内遗留元素被强制清空，避免冗余的尾部收敛代码。”

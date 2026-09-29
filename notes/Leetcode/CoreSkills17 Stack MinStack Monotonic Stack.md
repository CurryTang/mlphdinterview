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

### 2.2 · 底层直觉：三大物理模型与顿悟时刻 (Intuitive Mental Models)

许多工程师在初学单调栈时，往往困惑于“为什么偏偏用栈”、“为什么弹出的元素可以永久丢弃”、“为什么一次弹栈能同时得到左右两个边界”。以下三大物理模型与核心顿悟点能够直观解开这些本质机理：

#### 模型一：“天际线视线遮挡”模型（为什么弹出的元素可以永久丢弃？）

以“找右侧第一个更大元素”为例，假设每个数字代表一栋建筑的高度：
- 扫描指针向右推进，当前遇到了一栋高建筑 $B_{\text{current}}$，它比前面几栋矮建筑 $B_{\text{old}}$ 都要高。
- 此时将矮建筑 $B_{\text{old}}$ 弹出淘汰，后续会不会后悔？**绝不会**。
  - **对 $B_{\text{old}}$ 自己而言**：它向右看去，第一眼看到的高于自己的建筑就是当前的 $B_{\text{current}}$，$B_{\text{old}}$ 的问题已被完美回答。
  - **对未来更右侧的任意建筑 $B_{\text{future}}$ 而言**：当它回头往左看时，高耸的 $B_{\text{current}}$ 已经横亘在视野前方，矮建筑 $B_{\text{old}}$ 被彻底遮挡在阴影中（Eclipsed）。任何视线在触及 $B_{\text{old}}$ 之前，都必然先被更高的 $B_{\text{current}}$ 截断！
  - 因此，$B_{\text{old}}$ 在这个序列中已经**永久丧失了作为后续任何元素“左侧最近更高者”的可能性**。它的历史使命彻底终结，将其永久弹出（`pop`）在逻辑上是绝对闭环且完备的。

```text
[视线遮挡原理]
高度:
  6 │                █ (current: 6)
  5 │        █       █
  4 │        █   █   █ ─── 高度 6 横亘在右侧，
  2 │  █     █   █   █     后续任何建筑往左看，
  1 │  █  █  █   █   █     绝对看不见被遮挡在阴影里的 1 和 2！
────┼───────────────────────► 下标推进
下标:  0  1  2   3   4
```

#### 模型二：“悬赏任务看板与最弱防线”模型（为什么偏偏是 LIFO 栈？）

为什么单调栈必须是“后进先出（LIFO）的栈”，而不能是队列或普通哈希表？
- 每一个入栈的下标，都相当于在看板上挂出一张**未决悬赏令**：“我在等待右侧第一个比我强的元素来解答我”。
- 栈底是整个序列中数值最大、最坚固的悬赏；栈顶则是数值最小、最脆弱的**“最弱防线”**。
- 当新元素到来时，它只需要去冲击看板的**最顶端（最弱防线）**：
  - 如果新元素连最矮的栈顶都无法战胜，由于栈底所有老元素都比栈顶更高更强，新元素**绝不可能**解答更深层的任何悬赏！因此根本不需要深入扫描，单次 $O(1)$ 比较后便可立即停止循环，直接挂出自己的悬赏（`push`）。
  - 如果新元素战胜了栈顶，说明栈顶悬赏被当场揭榜（`pop` 并写回答案）；随之下方暴露出来的次弱防线再次接受挑战，直到新元素撞上无法逾越的高墙。
- **结论**：LIFO 栈的精妙之处在于，它让每一次探索都从“阻力最小的破口”切入，彻底消除了无谓的深层回溯。

#### 模型三：“两枚硬币生命周期”模型（为什么嵌套 while 严格是 O(n)？）

代码中两重循环（`for` 嵌套 `while`）往往给人一种 $O(n^2)$ 的错觉。理解线性复杂度的最佳视角是**硬币摊还分析（Amortized Coin Analysis）**：
- 给数组中的每一个元素发放且仅发放**两枚金币**：
  - **第 1 枚金币**：用于支付它被 `push` 进栈时的操作开销；
  - **第 2 枚金币**：用于预付它未来某刻被 `pop` 出栈时的操作开销。
- 每一个元素一旦被弹出，就立刻被销毁，再也没有第三枚金币能让它回到栈中。
- 某一次循环中 `while` 连续弹出了 5 个元素，看似发生了耗时循环，但实际上只是**一口气花掉了这 5 个元素在过去早就预付好的退场金币**。
- 整个程序执行结束，全剧总共消耗的金币数至多为 $2n$ 枚。因此，无论单步循环弹栈多么剧烈，全局的总时间开销严格受控在 $O(n)$。

#### 核心顿悟时刻：弹栈一瞬间的“时空折叠”（同时获取左右双边界）

在柱状图最大矩形（LC 84）或接雨水（LC 42）中，最令人费解的往往是：“为什么仅仅一次弹栈，就能同时确定一个柱子的左边界和右边界？”

当 `mid = stack.pop()` 发生的那一个瞬间，时空事实上发生了精准对齐：
1. **右边界（Right）**：正是当前正在扫描、触发本次弹栈的元素 $i$。因为正是因为 $nums[i]$ 破坏了递增单调性，$i$ 必然是 $mid$ 右侧**第一个更小的障碍**。
2. **左边界（Left）**：弹出 $mid$ 之后，留在栈内的新栈顶 `stack[-1]`，必然是 $mid$ 左侧**第一个更小的障碍**！
   - *为什么中间没有更小的？* 因为在 `stack[-1]` 和 $mid$ 之间如果存在任何更矮的柱子，当 $mid$ 当初入栈时早就已经被淘汰了，绝不可能存活到此刻。
3. **有效跨度一目了然**：以 $mid$ 为高度能够完整延伸的连续开区间严格为 $(Left, Right)$，其物理宽度为：
   $$W = \text{Right} - \text{Left} - 1$$
   一次简单的弹栈，同时锁死了被弹出者向左右两端扩张的极限。

---

### 2.3 · 单调栈万能通用模板：三问四槽模型 (Universal Monotonic Stack Blueprint)

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

### 2.4 · 比较符号与状态对照表

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

### 2.5 · 经典题目万能模板填装对照表 (Slot-Filling Matrix)

面对任何题目，直接将业务参数代入“四槽模型”：

| 经典题目 | 模式分类 | 槽位 1 (哨兵) | 槽位 2 (弹栈谓词) | 槽位 3 (结算时机与计算) | 槽位 4 (入栈) |
|---|---|---|---|---|---|
| **LC 739. Daily Temperatures** | Next Greater | 无 | `top < current` | 槽位 3A：`ans[mid] = i - mid` | `stack.append(i)` |
| **LC 496. Next Greater Element I** | Next Greater | 无 | `top < current` | 槽位 3A：`ans[mid] = current` | `stack.append(i)` |
| **LC 503. Next Greater Element II** | 循环 Next Greater | $2n$ 遍历取模 | `top < nums[i % n]` | 槽位 3A：`ans[mid] = nums[i % n]`（当 `mid < n`） | `if i < n: stack.append(i)` |
| **LC 84. Largest Rectangle** | Dual Smaller | 尾部加 `0` | `top > current` | 槽位 3A：`w = i - stack[-1] - 1`<br>`ans = max(ans, heights[mid] * w)` | `stack.append(i)` |
| **LC 42. Trapping Rain Water** | Dual Greater | 无 | `top < current` | 槽位 3A：凹槽高度差乘以宽度<br>`h = min(top, current) - mid_h` | `stack.append(i)` |
| **LC 907. Subarray Minimums** | Dual Smaller (去重) | 尾部加 `0` | 左严格 `<`，右侧 `<=` | 槽位 3A：乘法原理计算子数组数<br>`count = (mid - left) * (right - mid)` | `stack.append(i)` |
| **LC 1063. Valid Subarrays** | Next Strictly Smaller | 尾部加 `-inf`（或遍历至 $n$） | `top > current` | 槽位 3A：段长累加<br>`ans += i - mid` | `stack.append(i)` |

---

### 2.6 · 哨兵机制全景拆解：什么时候必须用哨兵？(Sentinel Demystified)

哨兵（Sentinel）在单调栈中并非可有可无的代码糖衣，而是用来**彻底消除边界特判、阻断非法越界、并强制完成未决计算的数学屏障**。

#### 1. 哨兵解决的两大根本病态（Dual Failure Modes）

在不使用哨兵时，单调栈在工程实现上面临两个必然出现的边界病态：

- **病态一：头部栈空越界（Head Underflow / Left Boundary Fallback）**
  - **产生时机**：当执行 `mid = stack.pop()` 弹出当前考察元素后，算法需要确定其左侧边界 `left = stack[-1]`。若此时栈已空（说明 $mid$ 左侧没有任何元素比它更小/更大，$mid$ 本身就是左侧全局极值），直接读取 `stack[-1]` 会引发 `IndexError`。
  - **无哨兵妥协**：被迫到处编写条件防御代码：`left = stack[-1] if stack else -1`。
- **病态二：尾部滞留漏算（Tail Stranding / Flush Omission）**
  - **产生时机**：当数组遍历完毕时，若栈内仍有元素残留（例如数组单调递增，或者局部递增序列）。若算法的**核心计算挂载在“弹栈”阶段（槽位 3A）**（如柱状图最大矩形 LC 84、最大矩形 LC 85、子数组最小值累加 LC 907），由于遍历已经结束，后续再无新元素破坏单调性，残留元素将**永远无法被弹出，从而导致严重漏算**！
  - **无哨兵妥协**：在主循环之后，被迫再复制粘贴一段结构极其相似的 `while stack:` 清栈计算逻辑，不仅代码臃肿，且极易在边界索引处理上引入 Off-by-One 缺陷。

```text
[无哨兵 vs 哨兵架构对比]

无哨兵:
遍历原数组 ──────► 栈内遗留未决元素 ──────► 必须外挂 while stack 重复清栈
                      │
                      └─► 每次取 left 必须写: stack[-1] if stack else -1

首尾双哨兵:
[-∞ / 0] + 原始数组 + [-∞ / 0] ──────────► 遍历结束时所有元素已被尾部哨兵自动逼出
   │                       │
   │                       └─► 尾部哨兵: 强制 100% 触发槽位 3A 弹栈结算 (Zero-leak)
   └─────────────────────────► 头部哨兵: 栈永不为空，stack[-1] 恒成立 (No underflow)
```

#### 2. 哨兵流派与选型矩阵（Sentinel Taxonomy & Decision Matrix）

| 哨兵类型 | 典型形式 | 核心作用 | 适用题型特征 | 典型题目 |
|---|---|---|---|---|
| **尾部单哨兵** | `arr = nums + [0]`<br>或 `range(len(nums) + 1)` | 引入**全局破坏者**，强制在最后一步将栈内滞留元素全量弹出结算，消灭循环外重复代码。 | **结算挂载在弹栈阶段（槽位 3A）**，且需要统计全量有效区间的极值。 | **LC 84** (柱状图最大矩形)<br>**LC 907** (子数组最小值之和)<br>**LC 85** (最大矩形) |
| **头部单哨兵** | 预置 `stack = [-1]`<br>或 `arr = [0] + nums` | 充当**开区间天然左支点**，维持栈永非空的不变量，消灭 `if not stack` 特判。 | 需要确定左边界，但尾部不需要强制结算（或由其他机制兜底）。 | **LC 32** (最长有效括号)<br>**LC 84** (单调递增左边界) |
| **首尾双哨兵** | `arr = [0] + heights + [0]`<br>或 `[-inf] + nums + [-inf]` | **同时彻底消灭头部与尾部的所有边界特判**。代码最对称精炼，完全杜绝边界分支。 | 几何面积、双侧连续跨度极值等需要同时查询左右闭环区间的题型。 | **LC 84** (柱状图最大矩形最优解)<br>**LC 85** (最大矩形) |
| **无需哨兵** | 保持原始数组 `nums`<br>结果数组赋默认值 | **天然满足无解定义**，滞留在栈中的元素即代表无答案，无需清栈。 | 1. 结果数组预置 `-1` 或 `0`（如 LC 739，残留元素代表右侧无更大者）；<br>2. 结算挂载在**入栈前的左查询（槽位 3B）**。 | **LC 739** (每日温度)<br>**LC 496** (下一个更大元素 I)<br>**LC 503** (循环数组取模倍增) |

#### 3. 哨兵数值选型法则（Extreme Value Principle）

哨兵的数值绝非随意指定，必须严格服从**空间极值定理**：
- **递增栈（目标：寻找更小元素，如 LC 84、LC 907）**：
  - 尾部哨兵必须**严格小于**数组中可能出现的任何合法元素值，才能确保击穿栈顶所有元素，触发完全清栈。
  - 若数据保证全为非负数（如高度 $heights[i] \ge 0$），取 `0` 即可；
  - 若数组中可能出现负数或任意整数，必须取负无穷大：`float('-inf')`。
- **递减栈（目标：寻找更大元素，如接雨水、最大跨度）**：
  - 尾部哨兵必须**严格大于**数组中可能出现的任何合法元素值，通常取正无穷大：`float('inf')`。

#### 4. 实战代码演进对比（以 LC 84 为例）

```python
# 方案 A：无哨兵（冗长，需处理栈空与循环外清栈，极易遗漏）
class SolutionNoSentinel:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack, max_area = [], 0
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] > h:
                mid = stack.pop()
                left = stack[-1] if stack else -1
                max_area = max(max_area, heights[mid] * (i - left - 1))
            stack.append(i)
        # 必须外挂第二套清栈逻辑处理残留元素！
        while stack:
            mid = stack.pop()
            left = stack[-1] if stack else -1
            max_area = max(max_area, heights[mid] * (len(heights) - left - 1))
        return max_area


# 方案 B：首尾双哨兵（极致优雅，零边界特判，天然闭环）
class SolutionDualSentinels:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 首尾各垫一个高度为 0 的哨兵
        arr = [0] + heights + [0]
        stack, max_area = [], 0

        for i, h in enumerate(arr):
            # 左哨兵保证 stack 永不为空；右哨兵保证最后一步完全出栈
            while stack and arr[stack[-1]] > h:
                mid = stack.pop()
                left = stack[-1]  # 绝对不会越界，无需三元特判
                width = i - left - 1
                max_area = max(max_area, arr[mid] * width)
            stack.append(i)

        return max_area
```

---

### 2.7 · 题型实战与模板代入

#### 实战一：Daily Temperatures（每日温度）
给定每日气温，求出每一天需要等待多少天才会遇到更高气温。
- **模板映射**：属于典型 **Next Greater** 模式，答案形式由下标变为跨度差值 $i - mid$。

```daily-temperatures-demo
```

##### 状态流转推演与关键破链时刻 (Execution Trace & State Transitions)

以输入 `temperatures = [73, 74, 75, 71, 69, 72, 76, 73]` 为例，单调栈的动态演变轨迹如下：

```text
[单调递减栈执行时序流 (Decreasing Stack Execution Flow)]

Day 0 (73°): 栈空 ──► 压入 0(73°)
             栈状态: [ 0(73°) ]

Day 1 (74°): 74° > 73° ──► 弹出 0(73°), ans[0] = 1 - 0 = 1 ──► 压入 1(74°)
             栈状态: [ 1(74°) ]

Day 2 (75°): 75° > 74° ──► 弹出 1(74°), ans[1] = 2 - 1 = 1 ──► 压入 2(75°)
             栈状态: [ 2(75°) ]

Day 3 (71°): 71° < 75° ──► 压入 3(71°)
             栈状态: [ 2(75°), 3(71°) ]

Day 4 (69°): 69° < 71° ──► 压入 4(69°) (持续降温，形成单调递减悬赏链)
             栈状态: [ 2(75°), 3(71°), 4(69°) ] ◄── 栈顶 4(69°) 为最弱防线

Day 5 (72°): 【多级连续破链时刻 1】
             ├─ 72° > 69° ──► 弹出 4(69°), ans[4] = 5 - 4 = 1
             ├─ 72° > 71° ──► 弹出 3(71°), ans[3] = 5 - 3 = 2
             └─ 72° < 75° ──► 触碰坚固防线，停止弹栈！压入 5(72°)
             栈状态: [ 2(75°), 5(72°) ]

Day 6 (76°): 【多级连续破链时刻 2 · 全局突破】
             ├─ 76° > 72° ──► 弹出 5(72°), ans[5] = 6 - 5 = 1
             └─ 76° > 75° ──► 弹出 2(75°), ans[2] = 6 - 2 = 4 (终结长达 4 天的等待！)
             压入 6(76°)
             栈状态: [ 6(76°) ]

Day 7 (73°): 73° < 76° ──► 压入 7(73°)
             栈状态: [ 6(76°), 7(73°) ]

遍历结束:
             栈内遗留 [6, 7] 无后续升温日，ans 保留默认值 0: ans[6] = 0, ans[7] = 0
最终输出:    ans = [1, 1, 4, 2, 1, 1, 0, 0]
```

```python
from typing import List


class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []  # 保存下标，栈底到栈顶对应气温严格单调递减

        for i, temp in enumerate(temperatures):
            # 槽位 2: 当前温度高于栈顶历史温度，触发破链弹栈
            while stack and temperatures[stack[-1]] < temp:
                mid = stack.pop()
                # 槽位 3A: 弹栈结算，答案取时间跨度差值
                ans[mid] = i - mid
            # 槽位 4: 当前天数入栈等待未来升温日
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

#### 实战五：Number of Valid Subarrays（LC 1063，有效子数组个数与单侧延展贡献）

给定整数数组 `nums`，定义一个子数组是“有效”的，当且仅当该子数组的**最左端首元素等于该子数组的最小值（允许相等）**。求有效子数组的总个数。

##### 1. 数学模型与边界判定
- **问题本质转化**：
  若子数组 $nums[i..j]$ 有效，则要求 $\min(nums[i..j]) = nums[i]$，即区间内所有元素均满足：
  $$nums[k] \ge nums[i], \quad \forall k \in [i, j]$$
- **追问一：相等元素会不会截断窗口？**
  - **结论：绝对不会！**
  - **严密证明**：当遇到 $nums[k] == nums[i]$ 时，区间最小值仍然等于 $nums[i]$，并未违背有效性定义。因此，**相等元素不会破坏子数组合法性**，窗口可以继续向右无阻碍延展。截断窗口的**唯一触发条件是遇到严格小于首元素的数值**（即 $nums[k] < nums[i]$）。

##### 2. 左端点、右端点与贡献公式推导
- **左端点（起始点）**：固定考察以某一下标 $i$ 作为子数组的最左元素。
- **右边界 $R_i$**：向右寻找**第一个严格小于 $nums[i]$** 的破坏者下标：
  $$R_i = \min \{ k \mid k > i \text{ 且 } nums[k] < nums[i] \}$$
  若右侧没有任何元素比 $nums[i]$ 更小，则右边界延伸至数组末尾之外，虚拟边界记为 $R_i = n$。
- **合法右端点范围**：既然 $R_i$ 之前的所有元素都 $\ge nums[i]$，则任何以 $j \in [i, R_i - 1]$ 结尾的子数组 $nums[i..j]$ 均合法。
- **单元素贡献公式（Contribution Formula）**：
  以 $i$ 为左端点的有效子数组个数，严格等于合法右端点 $j$ 的可选数量：
  $$\text{Count}(i) = (R_i - 1) - i + 1 = R_i - i$$
- **全局总数**：对所有合法左端点的贡献进行求和：
  $$\text{Total} = \sum_{i=0}^{n-1} (R_i - i)$$

##### 3. 单调栈四槽模型装填与实现

维护一个**单调不减栈（栈底到栈顶 $nums[stack[k]] \le nums[stack[k+1]]$，允许相等元素共存）**：
- **槽位 1（哨兵）**：在末尾虚拟引入一个负无穷大哨兵（或遍历至 $n$），强制将遍历结束后滞留在栈中的所有元素全部弹出；
- **槽位 2（谓词）**：`while stack and nums[stack[-1]] > curr_val:`，遇到严格更小值即触发破链弹栈；
- **槽位 3A（结算）**：当前扫描到的 $i$ 即为被弹出者 $mid$ 的右边界 $R_{mid} = i$，贡献即为 $i - mid$，累加入总答案；
- **槽位 4（入栈）**：`stack.append(i)`。

```python
from typing import List


class Solution:
    def validSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        stack = []  # 栈内下标对应数值单调不减 (nums[stack[k]] <= nums[stack[k+1]])

        # 槽位 1: 遍历至 n (虚拟注入全局最小值哨兵，强制清空栈)
        for i in range(n + 1):
            curr_val = float("-inf") if i == n else nums[i]

            # 槽位 2: 遇到严格更小值，说明打破了以栈顶为最小值的窗口
            while stack and nums[stack[-1]] > curr_val:
                mid = stack.pop()
                # 槽位 3A: 当前 i 即为 mid 右侧首个更小者，贡献段长为 i - mid
                ans += i - mid

            # 槽位 4: 下标入栈
            stack.append(i)

        return ans
```

##### 4. 视角对偶拓展：按右端点存活栈深结算（Active Stack Depth Duality）

除了“以 $mid$ 为左端点，在出栈时结算跨度 $R_{mid} - mid$”之外，本题存在一个极度优美的**数学对偶视角**：
- **以当前 $i$ 为右端点**：
  在执行完 `while stack and nums[stack[-1]] > nums[i]: stack.pop()` 之后，栈中残留的每一个历史元素 $k$，都保证满足 $nums[k] \le nums[i]$，且 $k$ 到 $i$ 之间的所有数值均不小于 $nums[k]$。
- **这意味着**：当前栈内存在的**每一个下标 $k$，都可以作为一个以 $i$ 为结尾的合法有效子数组的左端点**！
- 此时将 $i$ 自身入栈后，以 $i$ 结尾的有效子数组总数，**恰好等于当前栈的物理深度 `len(stack)`**！

```python
class SolutionStackDepth:
    def validSubarrays(self, nums: List[int]) -> int:
        ans = 0
        stack = []

        for num in nums:
            # 弹出所有严格大于当前值的元素（它们无法以当前 num 作为延续）
            while stack and stack[-1] > num:
                stack.pop()
            stack.append(num)
            # 当前栈内所有元素均可作为以当前元素为右端点的有效子数组起点
            ans += len(stack)

        return ans
```

两种视角在数学上恒等：
$$\sum_{i=0}^{n-1} (R_i - i) \equiv \sum_{j=0}^{n-1} \text{StackDepth}(j)$$
- **视角 A（左端点延展跨度）**：体现的是四槽模板中**槽位 3A 弹栈结算**的经典模式，与柱状图最大矩形、每日温度一脉相承；
- **视角 B（右端点栈深存活）**：体现了单调栈内部活跃元素的天然拓扑序。

---

### 2.8 · 复杂度证明：为什么嵌套 while 循环仍然是 O(n)？

许多初学者容易对嵌套在 `for` 循环内的 `while` 产生 $O(n^2)$ 的误判。

严格的**聚合分析（Aggregate Analysis / 摊还复杂度）**证明如下：
1. 数组长度为 $n$，每个下标进入外层 `for` 循环至多被 `stack.append()` **执行 1 次**；
2. 元素只有存在于栈内时，才可能在 `while` 内部被 `stack.pop()` **执行至多 1 次**；
3. 一旦某个下标被出栈弹出，它便彻底脱离生命周期，后续遍历中绝不可能再次入栈或被重复弹出；
4. 因此，跨越所有 $n$ 次外层迭代，内层 `while` 条件成立并执行 `pop()` 的**物理总次数上限严格为 $n$ 次**。

$$\sum_{i=1}^n (\text{push 次数} + \text{pop 次数}) \le n + n = 2n = O(n)$$

因此，单调栈的全局时间复杂度严格为 $O(n)$，空间复杂度因最坏情况下需保存全量单调下标而为 $O(n)$。

---

### 2.9 · 面试结构化应答清单

在白板或线上编码面试中，单调栈的答题推进可严格按以下 5 步展开：

1. **定型声明**：“本题需要为每个元素寻找单侧/双侧最近极值边界，暴力为 $O(n^2)$，最优结构为单调栈，时间复杂度降至 $O(n)$。”
2. **载体确认**：“栈内保存下标而非数值，因为后续计算跨度 $i - j$ 和面积需要物理距离。”
3. **谓词说明**：“求右侧更大元素，因此栈内维持单调递减；一旦遇到严格更大值即触发弹栈。”
4. **结算归属**：“答案在弹栈时写给旧元素（槽位 3A），因为当前元素扮演的是‘回答者’角色。”
5. **边界哨兵**：“对于柱状图或多区间计算，在数组尾部注入哨兵 `0`，确保栈内遗留元素被强制清空，避免冗余的尾部收敛代码。”

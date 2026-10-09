# Stack · MinStack 与单调栈

栈的核心 API 非常精简，其高频应用主要收敛为两大范式：

```text
MinStack：入栈时保存历史快照，使极值查询无须回溯扫描
单调栈：维护未解元素的候选单调集，等待右侧元素触发消除并确定边界
```

MinStack 是独立的状态快照设计；单调栈则是处理“一维序列最近极值边界”的高频通用题族。掌握单调栈的关键在于建立**“谁破坏单调性，谁就负责弹栈结算”的极简心智**，无论题目如何变化，核心骨架仅需 5 行代码。

## 学习顺序

题目精选自典型高频核心题型，建立从状态快照到单调栈全域边界的递进理解：

| 顺序 | 原题 | 核心模型 | 解决的关键问题 |
|---:|---|---|---|
| 1 | [155. Min Stack](https://neetcode.io/problems/minimum-stack/question?list=neetcode150) | 状态快照模式 (State-Snapshot) | 每个栈帧绑定 `min_so_far` 实现 $O(1)$ 查询与回滚 |
| 2 | [739. Daily Temperatures](https://neetcode.io/problems/daily-temperatures/question?list=neetcode150) | 单调栈·右侧边界 (Next Greater) | 弹栈时根据下标差结算等待天数 |
| 3 | [503. Next Greater Element II](https://leetcode.com/problems/next-greater-element-ii/) | 单调栈·循环数组 (Circular Array) | $2n$ 遍历取模倍增，无缝复用单调栈 |
| 4 | [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150) | 单调栈·双侧边界 (Dual Boundaries) | 首尾垫 `0` 哨兵，弹栈时同步锁定左右边界与宽度 |
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

### 2.3 · 极简核心心智与 5 行通用模板

单调栈完全不需要去记复杂的“几问几槽”。在面试中，只要记住**一句话核心心智**：

> **“栈内永远存下标；谁破坏了单调性，谁就是右边界，负责把栈顶弹出来结算。”**

#### 1. 核心口诀（10 秒搞懂递增还是递减）
- **找下一个更大**（Next Greater）$\to$ 维护**单调递减栈**（从底到顶由大到小）：
  - 遇到更小的数：入栈等待；
  - 遇到**更大的数**：破坏了递减，栈顶终于等到了比它大的元素！**立即出栈并写答案**。
- **找下一个更小**（Next Smaller）$\to$ 维护**单调递增栈**（从底到顶由小到大）：
  - 遇到更大的数：入栈等待；
  - 遇到**更小的数**：破坏了递增，栈顶终于等到了比它小的元素！**立即出栈并写答案**。

#### 2. 极简通用模板（闭眼默写这 5 行）

```python
stack = []  # 存下标
ans = [0] * n  # 预设默认答案（未被弹出的元素自动保留 0 或 -1）

for i, x in enumerate(nums):
    # 找下一个更大：来了个更大的 x，栈顶被满足，出栈结算！
    # （若找下一个更小，只需将 < 改为 >）
    while stack and nums[stack[-1]] < x:
        top = stack.pop()
        ans[top] = i - top  # 或 ans[top] = x，根据题意结算
    stack.append(i)
```

下面的交互演示用同一个数组执行“右侧更大”与“右侧更小”的逐步演算：

```monotonic-stack-demo
```

---

### 2.4 · 实战仅有的 3 个变形技巧

所有的单调栈变体，都只是在上面 5 行模板的基础上做极简微调：

1. **变体一：常规单侧边界（如 LC 739 每日温度、LC 496）**
   - 结果数组预先填好默认值（`-1` 或 `0`）；
   - 遍历结束后留在栈里的元素，说明右侧没有比它更优的，天然保留默认值，**不需要任何额外清栈操作**。

2. **变体二：循环数组（如 LC 503）**
   - 数组转两圈：循环写成 `for i in range(2 * n): x = nums[i % n]`；
   - 弹栈正常进行，只有前一圈入栈：`if i < n: stack.append(i)`。

3. **变体三：双侧边界 / 柱状图矩形（如 LC 84）**
   - **首尾各垫一个 0（最省心的哨兵）**：`heights = [0] + heights + [0]`；
   - **左边的 0**：保证栈永远不为空（取左边界 `stack[-1]` 绝不越界，无需 `if stack` 特判）；
   - **右边的 0**：保证遍历结束时，栈内所有残留高度全部被逼出结算（零漏算）；
   - 弹栈时一次性拿到左右两边：
     - 高度：`h = heights[stack.pop()]`
     - 左边界：新栈顶 `left = stack[-1]`
     - 右边界：当前元素 `right = i`
     - 宽度：`w = right - left - 1`，面积：`h * w`。

---

### 2.5 · 核心题型极简速查表

| 经典题目 | 查找目标 | 栈内维护状态 | 弹栈条件 (`while`) | 出栈结算内容 | 面试技巧 |
|---|---|---|---|---|---|
| **LC 739. 每日温度** | 右侧更大 | 递减（大 $\to$ 小） | `nums[stack[-1]] < x` | `ans[top] = i - top`（等待天数） | 默认数组填 0 |
| **LC 503. 下一个更大 II** | 循环右侧更大 | 递减（大 $\to$ 小） | `nums[stack[-1]] < x` | `ans[top] = x`（具体数值） | 遍历 $2n$，下标取模 `i % n` |
| **LC 84. 柱状图最大矩形** | 左右更小（定宽） | 递增（小 $\to$ 大） | `arr[stack[-1]] > x` | `h * (i - stack[-1] - 1)` | 首尾垫 `0`，零越界特判 |
| **LC 42. 接雨水** | 凹槽左右更高 | 递减（大 $\to$ 小） | `height[stack[-1]] < x` | `(min(左, 右) - 槽底) * (右 - 左 - 1)` | 弹出作槽底，新栈顶作左墙 |
| **LC 1063. 有效子数组** | 右侧首个更小 | 递增（小 $\to$ 大） | `nums[stack[-1]] > x` | 贡献段长：`ans += i - top` | 末尾垫 `-inf` 逼出全量结算 |

---

### 2.6 · 哨兵极简法则：什么时候用？

哨兵只需要记住**一句话法则**：
- **单侧查找（如每日温度、下一个更大）**：**完全不用哨兵**。结果数组直接预设默认值 `-1` 或 `0`，留在栈里的自然就是无解，省心省事。
- **双侧矩形 / 凹槽面积（如柱状图最大矩形 LC 84）**：**首尾垫 0**（`[0] + heights + [0]`）。左 0 防止栈空越界，右 0 强制将残留柱子全逼出结算，免去循环后任何补丁代码。

---

### 2.7 · 题型实战与模板代入

#### 实战一：Daily Temperatures（每日温度）
给定每日气温，求出每一天需要等待多少天才会遇到更高气温。
- **核心解法**：属于典型 **Next Greater** 模式，找右侧更大温度，遇到更高温度即弹栈，答案记录天数跨度 $i - mid$。

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
            # 当前温度高于栈顶温度，说明栈顶找到了升温日，触发弹栈
            while stack and temperatures[stack[-1]] < temp:
                mid = stack.pop()
                ans[mid] = i - mid  # 结算等待天数跨度
            stack.append(i)

        return ans
```

##### 右视图（逆序入栈即时结算）实现

若采用**逆序遍历（从右向左 $i = n-1 \to 0$）**，当前扫描的元素 $i$ 直接作为**主角**，栈中维护其右侧所有“潜在的 Next Greater 候选人”。任何气温 $\le temperatures[i]$ 的右侧候选人由于在高度和距离上均劣于当前天（被 $i$ 永久遮挡），直接被 `pop()` 淘汰；淘汰结束后，栈顶即为右侧第一个严格更暖的天数，可在入栈前**即时结算（Slot 3B）**自身答案：

```python
class SolutionRightView:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []  # 栈底到栈顶气温单调递减，维护右侧候选人索引

        for i in range(n - 1, -1, -1):
            temp = temperatures[i]
            # 淘汰右侧更低或相等的被遮挡元素
            while stack and temperatures[stack[-1]] <= temp:
                stack.pop()
            ans[i] = stack[-1] - i if stack else 0  # 即时结算当前天答案
            stack.append(i)

        return ans
```

##### 左视图 vs 右视图本质对比与选用准则

| 维度 | 左视图（正向遍历 + 出栈结算 · Slot 3A） | 右视图（逆序遍历 + 入栈结算 · Slot 3B） |
| :--- | :--- | :--- |
| **主角视角** | 栈内被弹出的元素 `mid` 是主角；当前扫描元素 $i$ 是其右侧终结者 | 当前扫描元素 $i$ 是主角；栈维护其右侧合法候选人骨干链 |
| **结算时机** | **延迟结算**：入栈等待，直到未来某天遇到更高值将其弹出时才写入答案 | **即时结算**：访问到 $i$ 时立刻清算完毕自身答案 |
| **栈内滞留元素** | 属于“未决元素”，若最终未被弹出则保留默认值（或需末尾哨兵清栈） | 属于“全局候选骨干链”，循环结束时所有答案均已确定，天然免除哨兵特判 |
| **何种情况更简单** | 需要同时借助左右两侧边界（如柱状图最大矩形、接雨水） | **流式在线数据**（LC 901）、**连续子数组计数**（LC 1063）、**DP状态转移**（LC 907） |

##### While 条件比较次数聚合分析与严格证明

设数组长度为 $n$。外层循环固定迭代 $n$ 次。考虑内层 `while stack and condition:` 的判定次数：

1. **栈空短路（Short-circuit on empty stack）**：
   当 `stack` 为空时，布尔表达式短路求值，**不发生任何元素数值比较**。
2. **比较成立（Condition evaluates to True）**：
   每次数值比较为 True，必对应一次 `stack.pop()`。由于每个下标至多入栈 1 次，整个算法生命周期内出栈次数至多为 $n$ 次：
   $$\text{Comparisons}_{\text{True}} \le n$$
3. **比较失败（Condition evaluates to False）**：
   每次数值比较为 False，`while` 循环立即终止并跳出。因为每次外层循环至多因 1 次 False 条件退出，外层共有 $n$ 次迭代：
   $$\text{Comparisons}_{\text{False}} \le n$$
4. **总比较次数上界（Tight Amortized Upper Bound）**：
   $$\text{Total Comparisons} = \text{Comparisons}_{\text{True}} + \text{Comparisons}_{\text{False}} \le n + n = 2n$$
   因此单调栈的数值比较总次数严格不超 $2n$ 次，均摊时间复杂度为 $\Theta(n)$。

###### 针对 `temperatures = [73, 74, 75, 71, 69, 72, 76, 73]`（$n=8$）的精确比较追踪

在该 canonical 数组上，左视图与右视图的总比较次数均为 **10 次**（远低于理论上界 $2n = 16$）：

- **左视图（正向）的 10 次比较明细**：
  - Day 0 (73°): 栈空短路，**0 次比较**。压入 0。
  - Day 1 (74°): $73° < 74°$ 为 True（弹出 0，结算），随后栈空短路。**1 次 True**。压入 1。
  - Day 2 (75°): $74° < 75°$ 为 True（弹出 1，结算），随后栈空短路。**1 次 True**。压入 2。
  - Day 3 (71°): $75° < 71°$ 为 False，循环终止。**1 次 False**。压入 3。
  - Day 4 (69°): $71° < 69°$ 为 False，循环终止。**1 次 False**。压入 4。
  - Day 5 (72°): $69° < 72°$ 为 True（弹出 4，结算）；$71° < 72°$ 为 True（弹出 3，结算）；$75° < 72°$ 为 False，循环终止。**2 次 True + 1 次 False = 3 次比较**。压入 5。
  - Day 6 (76°): $72° < 76°$ 为 True（弹出 5，结算）；$75° < 76°$ 为 True（弹出 2，结算）；随后栈空短路。**2 次 True**。压入 6。
  - Day 7 (73°): $76° < 73°$ 为 False，循环终止。**1 次 False**。压入 7。
  - **总计**：$6 \text{ 次 True} + 4 \text{ 次 False} = 10 \text{ 次数值比较}$。

- **右视图（逆序）的 10 次比较明细**：
  - Day 7 (73°): 栈空短路，**0 次比较**。结算 $ans[7]=0$。压入 7。
  - Day 6 (76°): $73° \le 76°$ 为 True（淘汰弹出 7），栈空短路。**1 次 True**。结算 $ans[6]=0$。压入 6。
  - Day 5 (72°): $76° \le 72°$ 为 False，循环终止。**1 次 False**。栈顶为 6，即时结算 $ans[5] = 6 - 5 = 1$。压入 5。
  - Day 4 (69°): $72° \le 69°$ 为 False，循环终止。**1 次 False**。栈顶为 5，即时结算 $ans[4] = 5 - 4 = 1$。压入 4。
  - Day 3 (71°): $69° \le 71°$ 为 True（淘汰弹出 4）；$72° \le 71°$ 为 False，循环终止。**1 次 True + 1 次 False = 2 次比较**。栈顶为 5，即时结算 $ans[3] = 5 - 3 = 2$。压入 3。
  - Day 2 (75°): $71° \le 75°$ 为 True（淘汰弹出 3）；$72° \le 75°$ 为 True（淘汰弹出 5）；$76° \le 75°$ 为 False，循环终止。**2 次 True + 1 次 False = 3 次比较**。栈顶为 6，即时结算 $ans[2] = 6 - 2 = 4$。压入 2。
  - Day 1 (74°): $75° \le 74°$ 为 False，循环终止。**1 次 False**。栈顶为 2，即时结算 $ans[1] = 2 - 1 = 1$。压入 1。
  - Day 0 (73°): $74° \le 73°$ 为 False，循环终止。**1 次 False**。栈顶为 1，即时结算 $ans[0] = 1 - 0 = 1$。压入 0。
  - **总计**：$4 \text{ 次 True} + 6 \text{ 次 False} = 10 \text{ 次数值比较}$。

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

        # 虚拟倍增遍历 2*n 次
        for i in range(2 * n):
            val = nums[i % n]
            # 遇到更大值触发弹栈
            while stack and nums[stack[-1]] < val:
                mid = stack.pop()
                ans[mid] = val  # 结算下一个更大数值
            # 仅前 n 个元素需要压栈等待解答
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

        # 首尾各垫高度为 0 的哨兵：左 0 防栈空，右 0 逼全出栈
        arr = [0] + heights + [0]
        for right, curr_h in enumerate(arr):
            # 当前高度更矮，破坏单调递增性，触发弹栈
            while stack and arr[stack[-1]] > curr_h:
                mid = stack.pop()
                mid_h = arr[mid]
                left = stack[-1]  # 必定非空，无需特判
                width = right - left - 1
                max_area = max(max_area, mid_h * width)
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
            # 遇到更高柱子，破坏递减形成凹槽
            while stack and height[stack[-1]] < curr_h:
                mid = stack.pop()
                if not stack:
                    break  # 左侧无边界，无法积水

                left = stack[-1]
                # 横向切片注水：高度受限于两端较矮者减去槽底
                h = min(height[left], curr_h) - height[mid]
                w = right - left - 1
                water += h * w

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

##### 3. 单调栈极简装填与实现

维护一个**单调不减栈（栈底到栈顶 $nums[stack[k]] \le nums[stack[k+1]]$，允许相等元素共存）**：
- **边界处理**：在末尾虚拟引入一个负无穷大哨兵（遍历至 $n$），强制将留在栈中的元素全部弹出；
- **弹栈条件**：`while stack and nums[stack[-1]] > curr_val:`，遇到严格更小值即触发弹栈；
- **出栈结算**：当前扫描到的 $i$ 即为被弹出者 $mid$ 的右边界，贡献段长为 $i - mid$，累加入总答案；
- **入栈等待**：`stack.append(i)`。

```python
from typing import List


class Solution:
    def validSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        stack = []  # 栈内下标对应数值单调不减 (nums[stack[k]] <= nums[stack[k+1]])

        # 遍历至 n (虚拟注入全局最小值哨兵，强制清空栈)
        for i in range(n + 1):
            curr_val = float("-inf") if i == n else nums[i]

            # 遇到严格更小值，说明打破了以栈顶为最小值的窗口
            while stack and nums[stack[-1]] > curr_val:
                mid = stack.pop()
                # 当前 i 即为 mid 右侧首个更小者，贡献段长为 i - mid
                ans += i - mid

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
- **视角 A（左端点延展跨度）**：体现的是弹栈结算的经典模式，与柱状图最大矩形、每日温度一脉相承；
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

在白板或线上编码面试中，单调栈的答题推进按以下 3 步清晰展开即可：

1. **识别模型**：“本题求一维序列中每个元素的单侧/双侧最近极值边界，暴力扫描需 $O(n^2)$，使用单调栈将均摊时间复杂度降至 $O(n)$。”
2. **栈与状态**：“栈内保存下标以便计算距离和跨度；求下一个更大值维护递减栈，遇到更大值立即出栈结算。”
3. **边界处理**：“单侧求值直接给结果数组预填默认值（-1 或 0）；双侧边界（如柱状图最大矩形）首尾垫 0 哨兵，彻底避免栈空越界与残留漏算。”


## 模块三：栈与单调栈高频扩展真题

### 1. LC 224 / 227 通用表达式计算器 (Universal Basic Calculator via Operator Precedence)

#### 核心心智（双栈模型与运算符优先级）
- 数字栈 `nums`，运算符栈 `ops`，优先级字典 `prec = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}`；
- 遇到新运算符时，若栈顶符号优先级更高或相等，立即弹栈结算；
- 遇到 `(` 入栈；遇到 `)` 循环结算直至遇到 `(`；
- 处理一元负号：若紧跟在开头或 `(` 后面出现负号，向 `nums` 垫入一个 `0`。

```python
class ExpressionCalculator:
    PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2}

    @staticmethod
    def _apply_op(op: str, b: int, a: int) -> int:
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return int(a / b)  # 向零截断
        return 0

    @classmethod
    def calculate(cls, s: str) -> int:
        nums, ops = [], []
        i, n, expect_operand = 0, len(s), True

        def evaluate():
            op = ops.pop()
            b = nums.pop()
            a = nums.pop()
            nums.append(cls._apply_op(op, b, a))

        while i < n:
            ch = s[i]
            if ch == ' ':
                i += 1
                continue
            if ch.isdigit():
                val = 0
                while i < n and s[i].isdigit():
                    val = val * 10 + int(s[i])
                    i += 1
                nums.append(val)
                expect_operand = False
                continue
            if ch == '(':
                ops.append('(')
                expect_operand = True
                i += 1
                continue
            if ch == ')':
                while ops and ops[-1] != '(':
                    evaluate()
                ops.pop()
                expect_operand = False
                i += 1
                continue
            if ch in cls.PRECEDENCE:
                if expect_operand and ch in ('+', '-'):
                    nums.append(0)  # 一元正负号前补 0
                while ops and ops[-1] != '(' and cls.PRECEDENCE.get(ops[-1], 0) >= cls.PRECEDENCE[ch]:
                    evaluate()
                ops.append(ch)
                expect_operand = True
                i += 1

        while ops:
            evaluate()
        return nums[0] if nums else 0
```

---

### 2. LC 316 / 1081 单调栈去重与字典序最小子序列 (Remove Duplicate Letters)

#### 核心心智（单调递增栈 + 字符末次出现位置 + 栈内存在性集合）
- 若字符已在栈中，直接跳过；
- 若栈顶字符比当前字符大，且栈顶在后面还会再次出现（`last_pos[top] > i`），则贪心弹出栈顶换取更小的字典序；
- 将当前字符推入栈并记录至 `in_stack`。

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        last_pos = {ch: i for i, ch in enumerate(s)}
        stack = []
        in_stack = set()

        for i, ch in enumerate(s):
            if ch in in_stack:
                continue
            while stack and stack[-1] > ch and last_pos[stack[-1]] > i:
                in_stack.remove(stack.pop())
            stack.append(ch)
            in_stack.add(ch)

        return "".join(stack)
```

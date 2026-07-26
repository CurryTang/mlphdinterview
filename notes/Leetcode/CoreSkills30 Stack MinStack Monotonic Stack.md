# Stack · MinStack 与单调栈

栈的 API 不难，真正容易卡住的是两种用法：

```text
MinStack：入栈时保存历史，让查询不必回头扫描
单调栈：先把下标放进栈里等待，等右侧第一个合适的元素来回答
```

MinStack 是一道相对独立的设计题，记住一版稳定写法即可。单调栈则是一组题，重点不是背每道题的代码，而是固定同一个模板，再替换比较符号和答案形式。

## 学习顺序

题目从 [NeetCode 150](https://neetcode.io/practice/practice/neetcode150) 中选，但只保留与这两个模块直接相关的三题。

| 顺序 | 原题 | 要掌握的内容 |
|---:|---|---|
| 1 | [155. Min Stack](https://neetcode.io/problems/minimum-stack/question?list=neetcode150) | 每个栈帧保存 `min_so_far` |
| 2 | [739. Daily Temperatures](https://neetcode.io/problems/daily-temperatures/question?list=neetcode150) | 右侧第一个更大值 |
| 3 | [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150) | 单调栈确定左右边界 |

## 模块一：MinStack

### 给每一层保存当时的最小值

如果只额外维护一个全局变量 `minimum`，`push` 和 `getMin` 很简单，但弹出当前最小值后，不知道上一个最小值是什么。重新扫描整个栈需要 $O(n)$。

更稳的写法是让每个栈元素保存一对值：

```text
(当前 value, 压入当前 value 后的 min_so_far)
```

于是栈顶同时包含两份信息：

```text
top()    = stack[-1][0]
getMin() = stack[-1][1]
```

每次 `push` 都给当前状态拍一张快照。`pop` 时快照跟着一起删除，下面一层保存的旧最小值自然恢复，不需要额外回滚逻辑。

### 为什么重复最小值不会出错

依次压入 `2, 1, 1`：

```text
[(2, 2), (1, 1), (1, 1)]
```

弹出一个 `1` 后，下面那个栈帧仍然保存 `(1, 1)`，所以最小值还是 `1`。不要只在“出现更小值”时记录辅助栈，否则重复最小值会让 `pop` 的同步逻辑变复杂。

### Quick Coding：实现 MinStack

实现 `push`、`pop`、`top` 和 `getMin`，四个操作都要求 $O(1)$。

<details>
<summary>参考答案</summary>

```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, value: int) -> None:
        minimum = value if not self.stack else min(
            value,
            self.stack[-1][1],
        )
        self.stack.append((value, minimum))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

每个操作都是 $O(1)$。为每个元素多保存一个最小值，空间复杂度为 $O(n)$。

</details>

这道题建议直接记住 `(value, min_so_far)`。双栈写法也正确，但面试现场需要多维护一次同步关系，没有必要给自己增加分支。

## 模块二：单调栈

### 它主要解决什么问题

单调栈最常处理的是这类问题：

> 对数组中的每个位置，找到它左边或右边第一个（最近的）更大或更小元素。

题目换一种说法，仍然可能是同一个结构：

| 题目问法 | 实际在找什么 |
|---|---|
| 还要几天才会升温 | 右侧第一个严格更大元素，答案取下标差 |
| 下一个更大元素是谁 | 右侧第一个严格更大元素，答案取值 |
| 柱子最多能向两边延伸多远 | 左右两侧第一个更小元素 |
| 当前价格连续支配前面多少天 | 左侧最近的更大元素形成边界 |

它有两个明显信号：

1. 输入通常是一维序列，题目要为**每个位置**找一个边界；
2. 这个边界不是全局最大值或最小值，而是某个方向上**第一个满足大小关系的位置**。

暴力做法会从每个位置向左或向右扫描，最坏需要 $O(n^2)$。单调栈把仍在等待边界的位置留下，让每个下标最多入栈、出栈各一次，把整批查询降到 $O(n)$。

如果题目问的是固定窗口最大值，通常用单调队列；如果只问整个数组的最大值，直接扫描即可。单调栈的价值在于同时回答一批“最近边界”问题。

### 栈里究竟保存什么

以“右侧第一个严格更大元素”为例，栈里不保存答案，而是保存：

> 已经扫描过、但右侧第一个更大元素还没有出现的下标。

通常保存**下标**，而不是只保存数值，因为一个下标同时提供三种信息：

```text
nums[j]：用来比较大小
j：用来计算距离 i - j
answer[j]：知道答案应该写回哪里
```

只存数值会丢掉位置，也无法区分数组中的重复值。

例如扫描 `nums = [5, 2, 4, 6]`：

```text
读到 5：stack = [0]       位置 0 在等更大的数
读到 2：stack = [0, 1]    2 不能回答 5，位置 1 也开始等待
读到 4：弹出 1            4 是位置 1 右侧第一个更大值
        stack = [0, 2]    位置 0 和 2 仍在等待
读到 6：依次弹出 2、0     6 同时回答两个尚未解决的位置
```

所以，栈不是“所有已经看过的元素”，而是经过淘汰后仍未得到答案的候选下标。扫描到下标 `i` 时，当前值 `nums[i]` 会不断回答栈顶：

```python
while stack and nums[i] > nums[stack[-1]]:
    j = stack.pop()
    answer[j] = i
```

弹栈结束后再把 `i` 压入，表示它也开始等待自己的右侧答案。此时栈中对应的值从栈底到栈顶单调不增，但“单调”是上述等待与淘汰过程的结果，不是最终目的。

为什么当前 `i` 一定是被弹出下标 `j` 的**第一个**答案？因为 `j` 入栈以后，扫描指针按顺序经过了 `j + 1, j + 2, ...`。如果中间有任何元素已经满足条件，`j` 当时就会被弹出，不可能等到现在。

### 一个统一模板

先写“右侧第一个满足条件的位置”：

```python
def first_match_on_right(nums):
    answer = [-1] * len(nums)
    stack = []  # 尚未找到答案的下标

    for i, value in enumerate(nums):
        while stack and value_satisfies(nums[stack[-1]], value):
            j = stack.pop()
            answer[j] = i

        stack.append(i)

    return answer
```

真正需要替换的只有两处：

```text
value_satisfies：当前值什么时候能回答栈顶
answer[j]：题目要下标、值，还是距离
```

### 比较符号表

把 `old = nums[stack[-1]]`、`current = nums[i]` 代入：

| 要找的答案 | `while` 弹栈条件 | 处理后栈内值：栈底 → 栈顶 |
|---|---|---|
| 右侧第一个严格更大 | `old < current` | 单调不增 |
| 右侧第一个大于等于 | `old <= current` | 严格递减 |
| 右侧第一个严格更小 | `old > current` | 单调不减 |
| 右侧第一个小于等于 | `old >= current` | 严格递增 |

最不容易写反的记法是：

> 不要先背“我要维护递增栈还是递减栈”。直接问：当前值是否已经满足栈顶等待的答案？满足就弹。

下面的演示用同一个数组跑“右侧更大”和“右侧更小”。切换目标时，模板只改一个比较符号。

```monotonic-stack-demo
```

### Daily Temperatures：答案从下标改成距离

给定每日温度，返回每一天还要等待多少天才会遇到更高温度；之后没有更高温度则返回 `0`。

这就是“右侧第一个严格更大”，只不过答案不是 `i`，而是距离：

```text
answer[j] = i - j
```

### Quick Coding：Daily Temperatures

```python
def dailyTemperatures(temperatures):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def dailyTemperatures(
        self,
        temperatures: List[int],
    ) -> List[int]:
        answer = [0] * len(temperatures)
        stack = []

        for i, temperature in enumerate(temperatures):
            while (
                stack
                and temperatures[stack[-1]] < temperature
            ):
                j = stack.pop()
                answer[j] = i - j

            stack.append(i)

        return answer
```

栈中下标对应的温度从栈底到栈顶单调不增。相等温度不能回答“严格更高”，所以比较符号必须是 `<`，不能写 `<=`。

</details>

### 右侧答案与左侧答案，记录时机不同

单调栈常见的两种问法只差答案属于谁。

### 找右侧第一个答案：回答被弹出的旧下标

```python
for i, value in enumerate(nums):
    while stack and current_answers_top(...):
        j = stack.pop()
        answer[j] = i
    stack.append(i)
```

当前值出现后，可能一次回答多个旧下标，因此答案写在 `while` 里面。

### 找左侧最近答案：弹掉无效候选，再读取栈顶

下面以“左侧最近的严格更小元素”为例：

```python
answer = [-1] * len(nums)
stack = []

for i, value in enumerate(nums):
    while stack and nums[stack[-1]] >= value:
        stack.pop()

    if stack:
        answer[i] = stack[-1]

    stack.append(i)
```

这里答案属于当前下标 `i`。`while` 负责清掉不可能成为答案的候选；清理完成后的栈顶，才是离 `i` 最近的严格更小元素。

| 问法 | `while` 在做什么 | 在哪里写答案 |
|---|---|---|
| 右侧第一个满足条件 | 当前值回答旧下标 | 弹栈时写 `answer[j]` |
| 左侧最近满足条件 | 删除当前下标不能使用的候选 | `while` 后读栈顶，写 `answer[i]` |

### Largest Rectangle：弹栈时确定完整边界

在 [84. Largest Rectangle in Histogram](https://neetcode.io/problems/largest-rectangle-in-histogram/question?list=neetcode150) 中，下标 `j` 被更矮的柱子 `right` 弹出时：

```text
right = 右侧第一个严格更矮的位置
stack[-1] = 弹出后左侧最近的更矮位置
```

因此高度 `heights[j]` 能覆盖的宽度是：

$$
\text{width}
=
\text{right}
-
\text{left}
-
1.
$$

末尾补一个高度为 `0` 的哨兵，可以让所有剩余柱子出栈，不必再复制一段清栈代码。

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        answer = 0
        stack = []

        for right in range(len(heights) + 1):
            current = 0 if right == len(heights) else heights[right]

            while stack and heights[stack[-1]] > current:
                j = stack.pop()
                left = stack[-1] if stack else -1
                width = right - left - 1
                answer = max(answer, heights[j] * width)

            stack.append(right)

        return answer
```

哨兵下标会在最后一次循环中入栈，但循环随即结束，不会再读取 `heights[len(heights)]`。

</details>

### 为什么嵌套 while 仍然是 O(n)

不要把外层 `for` 和内层 `while` 直接相乘。一个下标：

```text
最多入栈一次
最多出栈一次
```

所有 `push` 加起来不超过 $n$ 次，所有 `pop` 加起来也不超过 $n$ 次，所以总操作数是 $O(n)$。栈最多保存 $n$ 个下标，空间复杂度为 $O(n)$。

这和双指针、滑动窗口的线性分析是同一种摊还思路：局部看似有循环，某个元素一旦被删除，就不会回来。

### 面试前最后检查

1. 栈里存值还是下标？需要距离、边界或回看原数组时，优先存下标。
2. 当前元素是在回答旧下标，还是为自己寻找左侧答案？
3. 题目要求严格大于，还是大于等于？这决定 `<` 和 `<=`。
4. `answer[j]` 要保存下标、值还是 `i - j`？
5. 柱状图是否用哨兵清空了剩余下标？

最后只记一句：

> 栈顶下标还没有答案；当前值一旦满足它等待的条件，就弹栈并写答案。

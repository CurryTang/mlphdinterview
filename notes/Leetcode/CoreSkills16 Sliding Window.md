# Sliding Window

滑动窗口本质上是**双指针在单向序列上维护一个动态闭区间 $[\text{left}, \text{right}]$** 的算法模型。所有滑动窗口题目均共用同一个循环骨架，任何题目的解题过程都可以抽象为回答**“三大决策问题”**并填充对应的**“三个代码槽位”**：

```text
【三大决策问题与代码槽位】
1. state & add_right: 窗口内维护什么增量状态？right 进窗如何更新？
2. shrink condition : 什么时候移动 left 恢复不变量？（变长用 while，定长用 if）
3. record answer    : 什么时候记录答案？（最长在收缩后，最短在收缩中，定长在满窗时）
```

---

## 原题与学习顺序

本笔记精选 8 道高频核心题，按**最长合法窗口**、**最短满足窗口**、**固定长度窗口**与**单调队列窗口**四大维度分类递进：

| 顺序 | 原题 | 窗口分类 | 核心维护状态 | 收缩机制 | 记录时机 |
|---:|---|---|---|---|---|
| 1 | [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) | 最长合法窗口 | 字符频次哈希表 | `while` 存在重复 | 收缩后更新 `max` |
| 2 | [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/description/) | 最长合法窗口 | 窗口内 0 的计数 `zeros` | `while zeros > k` | 收缩后更新 `max` |
| 3 | [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/description/) | 最长合法窗口 | 频次表 + `max_freq` | `while len - max_freq > k` | 收缩后更新 `max` |
| 4 | [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/) | 最短满足窗口 | 区间元素累加和 `sum` | `while sum >= target` | 收缩前更新 `min` |
| 5 | [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/) | 最短满足窗口 | `need/window` + `have` | `while have == required` | 收缩前更新 `min` |
| 6 | [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/description/) | 固定长度窗口 | 26 位字符频次表 | `if len > k` | 满窗时比对返回布尔 |
| 7 | [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/) | 固定长度窗口 | 26 位字符频次表 | `if len > k` | 满窗时收集所有 `left` |
| 8 | [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/) | 固定长度窗口 | 单调递减下标双端队列 | `if len > k` | 满窗时读取队首最大值 |

先看经典五题在同一骨架下的模式对照：

```sliding-window-patterns
```

---

## 先固定窗口不变量

窗口数学建模统一采用闭区间：

$$
[\text{left}, \text{right}], \qquad \text{length} = \text{right} - \text{left} + 1.
$$

### 为什么闭区间长度必须 `+ 1`

`left` 与 `right` 均是指向当前窗口内部有效元素的合法数组下标。两下标做差 $\text{right} - \text{left}$ 得到的是两个指针之间的**步长间隔数**，而闭区间所包含的元素个数必须计入起点 `left` 自身，故必须再加 1：

```text
数组下标：  2   3   4
窗口内容： [A   B   C]
指针位置： left = 2, right = 4
下标跨度： 4 - 2 = 2
元素总数： 4 - 2 + 1 = 3
```

边界验证：当单元素窗口 `left == right` 时，闭区间长度恰为 $\text{right} - \text{left} + 1 = 1$；若漏掉 `+ 1` 则错误地得出长度为 0。

| 窗口区间定义 | 是否包含边界 `right` | 长度计算公式 | 适用场景 |
|---|---|---|---|
| **闭区间 $[\text{left}, \text{right}]$** | 是（两端均包含） | $\text{right} - \text{left} + 1$ | **算法推荐（全笔记统一规范）** |
| 半开半闭 $[\text{left}, \text{right})$ | 否（左闭右开） | $\text{right} - \text{left}$ | C++ STL / 切片 `s[left:right]` |

### 窗口状态与指针移动的严格时序

在滑动窗口中，状态维护与指针移动必须严格原子同步。从左端剔除元素时，必须**先从状态中逆向扣除，再递增指针**：

```python
# 正确顺序：先减状态，再动指针
remove_left(state, items[left])
left += 1

# 致命错误：先动指针，会导致从状态中删除了错误的后一个元素
# left += 1
# remove_left(state, items[left])
```

---

## 核心选型：窗口与 State 的数据结构架构

在滑动窗口中，**窗口本身**永远只需要两个整型双指针 `left` 与 `right` 维护一个动态闭区间 $[\text{left}, \text{right}]$，物理指针空间恒为 $O(1)$。

真正决定解题复杂度、算法可行性与工程效率的，是**窗口内部状态（`state`）的数据结构选型**。

### 选型第一性原理：增量推进与逆向撤销

滑动窗口能否在 $O(n)$ 时间内解决问题，取决于状态更新是否满足**低代价的可逆性**：
- **`right` 进窗**：能否基于旧状态，以 $O(1)$ 或 $O(\log k)$ 增量更新？
- **`left` 出窗**：能否以完全对称的逆操作，以 $O(1)$ 或 $O(\log k)$ 将左端元素彻底撤销？
- **有效性查询**：`while` 或 `if` 的判定条件能否在 $O(1)$ 内直接得出，而非遍历整个状态？

### 五大核心状态数据结构选型体系

```text
                      窗口需要维护和查询什么？
                                │
        ┌───────────────────────┼────────────────────────┐
        ▼                       ▼                        ▼
    【标量求和/计数】        【频次与存在性】         【动态极值/序数】
        │                       │                        │
  代数加减可逆运算?       字符集/Key 规模有限?       只需最大/最小值?
  ┌─────┴─────┐           ┌─────┴─────┐           ┌──────┴──────┐
  │ 是        │ 否        │ 是        │ 否        │ 是          │ 否 (中位数/第K大)
  ▼           ▼           ▼           ▼           ▼             ▼
标量变量   无法普通滑窗  定长数组    哈希表+标量  单调双端队列   对顶堆 / 有序集合
(Scalar)  (需前缀和)   (Array[26]) (HashMap+have) (Monotonic Deque) (BST / Dual Heaps)
```

#### 1. 标量变量（Scalar Variables：`int` / `float`）
* **适用场景：** 窗口状态是一个可聚合的标量，且该操作满足**代数加减严格可逆性**。
* **典型指标：**
  - **区间累加和（Sum）：** `curr_sum += nums[right]`，出窗 `curr_sum -= nums[left]`（LC 209）。
  - **特定元素计数（Counter）：** 如翻转 0 的配额 `zeros += 1`，出窗 `zeros -= 1`（LC 1004）。
* **复杂度：** 增减 $O(1)$，空间 $O(1)$。
* **边界警示：** 若数组含**负数**，区间和不再关于双指针单向移动具备单调性，标量滑窗失效，必须改用前缀和 + 哈希/单调队列（如 LC 862）。

#### 2. 定长静态数组（Direct-Mapped Array：`int[26]` / `int[128]`）
* **适用场景：** Key 的定义域极小且固定连续（如全小写英文字母 `a-z`、全大写 `A-Z`、标准 ASCII `128`）。
* **为什么坚决优于哈希表：**
  - **L1 Cache 局部性：** `int count[26]` 占用内存仅 $104$ 字节，单条 CPU L1 Cache Line 即可装载。数组寻址 `ord(c) - ord('a')` 为纯基址偏移，0 次哈希计算、0 次内存分配、0 指针跳跃。
  - **$O(1)$ 数组全量等价比对：** 在固定窗口中，`window == need` 仅需 26 次整型比较，现代编译器可直接向量化（SIMD），耗时几乎为 0。
* **代表题目：** LC 567（排列判定）、LC 438（异位词匹配）、LC 424（最高频次）。

#### 3. 动态哈希表 + 压缩标量（Hash Map + Scalar Counter）
* **适用场景：** Key 属于任意无界离散对象（任意整数、Unicode 字符、稀疏字符串）。
* **核心架构设计（标量降维）：**
  - 单纯使用哈希表时，若每轮都要判断“当前窗口是否包含所有必要字符”，比对整个字典需要 $O(|\Sigma|)$ 耗时。
  - **标量锚定设计**：在底层 `count = defaultdict(int)` 旁并联一个标量计数器（如 `have: int` 表示频次达标的字符种类数；或 `distinct: int` 表示非零键个数）。
  - 只有在频次**刚好达到目标或跌落目标**的瞬间对标量执行 `+= 1` 或 `-= 1`。进入 `while` 的判定直接压成 $O(1)$ 标量比对（如 `have == required`），实现严密降维。
* **代表题目：** LC 3（无重复子串）、LC 76（最小覆盖子串）、LC 992（K个不同整数）。

#### 4. 单调双端队列（Monotonic Deque）
* **适用场景：** 窗口需要动态查询**最值（Maximum / Minimum）**。
* **为什么标量和哈希表无法处理极值？**
  - **极值操作不具备代数逆向可逆性**：当最大值从左端离开窗口时，我们无法通过数值计算得知剩余元素中谁是“次大值”，必须重新扫描窗口（$O(k)$）。
  - 普通堆（Priority Queue）查询最值快，但滑动窗口做任意位置删除极其沉重。
* **核心原则：**
  - **必须存下标（Indices），不能只存数值**：只有存储下标，才能以 $O(1)$ 判定队首极值是否滑出了 $[left, right]$ 的物理范围（`q[0] < left`）。
  - 队列内部维护数值严格递减，进窗淘汰更小队尾（无后效性丢弃），出窗淘汰过期队首。
* **代表题目：** LC 239（滑动窗口最大值）、LC 1438（绝对差限制最长子数组）。

#### 5. 对顶堆 / 平衡二叉搜索树（Dual Heaps / Balanced BST）
* **适用场景：** 窗口需要查询**高级序数统计量（如动态中位数 Median、第 K 大元素）**。
* **为什么单调队列失效：** 单调队列只能保留两极单向支配的候选，但中位数位于集合中央，窗口内所有中间大小的数值都必须精确保留。
* **底层数据结构：**
  - **对顶堆（Dual Heaps + Lazy Deletion）：** 大根堆存较小半区，小根堆存较大半区；出窗元素在哈希表中记录“延迟删除”，在堆顶自然弹出时结算。
  - **平衡二叉搜索树（Balanced BST / `std::multiset` in C++）：** 原生支持 $O(\log k)$ 插入、按迭代器删除与中位数查找。
* **代表题目：** LC 480（滑动窗口中位数）。

---

### 状态数据结构选型全景决策矩阵

| 目标查询属性 | 推荐数据结构 (`state`) | 进窗操作 (`add`) | 出窗操作 (`remove`) | 查询判断复杂度 | 代表题目 |
|---|---|---|---|---|---|
| **区间累加和 / 平均值** | 标量 `curr_sum: int` | `+= nums[right]` | `-= nums[left]` | $O(1)$ | LC 209, LC 643 |
| **特定条件元素个数** | 标量 `count: int` | 命中则 `+ 1` | 命中则 `- 1` | $O(1)$ | LC 1004, LC 1248 |
| **固定字母表频次比对** | 定长数组 `int[26]` | `arr[c] += 1` | `arr[c] -= 1` | $O(1)$ 数组比较 | LC 567, LC 438, LC 424 |
| **任意无界字符/数字频次** | 哈希表 `defaultdict(int)` | `map[x] += 1` | `map[x] -= 1` (为 0 可删) | $O(1)$ 哈希命中 | LC 3, LC 340 |
| **全集覆盖判断（多字符）** | 哈希表 + 标量 `have: int` | 刚达标 `have += 1` | 刚失效 `have -= 1` | $O(1)$ 标量比对 | LC 76 |
| **不同种类数（Distinct $K$）** | 哈希表 + 标量 `distinct: int` | `0 -> 1` 时 `+= 1` | `1 -> 0` 时 `-= 1` | $O(1)$ 标量比对 | LC 992 |
| **动态最大值 / 最小值** | 双端单调队列 `deque`（存下标） | 淘汰队尾弱势元素后入队 | 队首过期则 `popleft()` | $O(1)$ 读队首 | LC 239, LC 1438 |
| **动态中位数 / 第 K 大** | 对顶堆（双堆）+ 延迟删除表 | 入堆并平衡左右堆大小 | 标记进哈希表延迟剔除 | $O(1)$ 读堆顶，更新 $O(\log k)$ | LC 480 |

### 工程实现原则

1. **轻量降级原则：能用标量绝不用集合，能用数组绝不用哈希表。**
   如果只需统计特定元素个数，声明一个 `int count = 0`，切勿用 `set` 存下标。字符集限定在 `a-z` 时，直接声明 `[0] * 26`，执行性能比 `defaultdict` 快 3 至 5 倍。
2. **标量锚定原则：复合状态必须外挂标量，杜绝在 `while` 中做全量遍历。**
   遇到多字符覆盖或种类统计，必须在哈希表旁并联 `have` 或 `distinct` 整数计数器。进入 `while` 的判定必须是 $O(1)$ 的标量关系（如 `have == required`），严禁在每次循环中执行 `all(...)` 或遍历比较。
3. **下标保真原则：极值队列必须存下标，严禁直接存值。**
   单调队列和单调栈如果只存数值，在滑动窗口右移时根本无法判断当前队首是“刚刚滑出窗口的老数据”还是“仍在窗口内部的有效数据”。必须存储下标，通过数值映射 `nums[q[0]]` 使用，边界判定 `q[0] < left` 才能立见分晓。

---

## 通用模板：三问三槽模型

滑动窗口共用的不是某一行固定的代码，而是清晰的解题思维流水线。

```text
                    ┌────────────────────────────┐
                    │  for right, item in ...:   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 1. 进窗: add_right(item)   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 2. 窗口是否固定长度 k ？   │
                    └──────┬──────────────┬──────┘
                           │ 是           │ 否
                           ▼              ▼
         ┌────────────────────────┐  ┌─────────────────────────────────┐
         │ if len > k:            │  │ while 触发收缩条件:             │
         │   remove_left(left)    │  │   [最短窗口: 记录当前 ans]      │
         │   left += 1            │  │   remove_left(left)             │
         └─────────────┬──────────┘  │   left += 1                     │
                       │             └─────────────┬───────────────────┘
                       │                           │
                       ▼                           ▼
         ┌────────────────────────┐  ┌─────────────────────────────────┐
         │ if len == k:           │  │ [最长窗口: 循环结束合规后更新]  │
         │   record_answer(...)   │  │ ans = max(ans, right-left+1)    │
         └────────────────────────┘  └─────────────────────────────────┘
```

### 1. 通用核心骨架代码

```python
def universal_sliding_window(items, k=None):
    left = 0
    state = initialize_state()
    answer = initialize_answer()

    for right, item in enumerate(items):
        # 槽位 1: 进窗更新 (add_right)
        add_right(state, item)

        # 槽位 2: 窗口收缩控制 (shrink)
        if k is not None:
            # 分支 A: 固定大小窗口 (Fixed-Size Window)
            if right - left + 1 > k:
                remove_left(state, items[left])
                left += 1

            # 槽位 3: 记录答案 (当窗口恰好满 k 时)
            if right - left + 1 == k:
                record_fixed(answer, state, left, right)
        else:
            # 分支 B: 变长窗口 (Variable-Size Window)
            # 模式 I: 最短满足窗口 (如 LC 209, LC 76)
            while is_satisfied(state):
                record_min(answer, left, right)   # 满足要求，出窗前先记录最短
                remove_left(state, items[left])
                left += 1

            # 模式 II: 最长合法窗口 (如 LC 3, LC 1004, LC 424)
            # while is_invalid(state):
            #     remove_left(state, items[left]) # 违规，出窗直到恢复合法
            #     left += 1
            # record_max(answer, left, right)     # 恢复合法后更新最长

    return answer
```

### 2. 三大核心题型的决策对比表

| 窗口模式 | 代表例题 | 收缩触发条件 (`shrink`) | 控制语句 | 答案记录时机 (`record`) | 核心机制 |
|---|---|---|---|---|---|
| **最长合法窗口** | LC 3, LC 1004, LC 424 | 窗口出现**非法**违规状态 | `while invalid:` | **`while` 循环结束后**（窗口重获合法） | 非法才缩，合法就扩，循环外更新 `max` |
| **最短满足窗口** | LC 209, LC 76 | 窗口**已满足**目标条件 | `while satisfied:` | **`while` 循环内部**（每次收缩前） | 达标就缩，边缩边测，循环内更新 `min` |
| **固定长度窗口** | LC 567, LC 438, LC 239 | 窗口长度**超过 $k$** | `if len > k:` | **窗口长度恰为 $k$ 时** | 每轮顶多弹 1 个，定长时刻做结算 |

---

## 1. Longest Substring Without Repeating Characters

### 题意

给定字符串 `s`，返回不含重复字符的**最长连续子串**长度。`s` 最长为 $5\times10^4$。

### 填模板

| 槽位 | 本题具体实现 |
|---|---|
| `state` | `count[char]`，哈希表维护当前窗口各字符出现频次 |
| `add_right` | `count[char] += 1` |
| `shrink` 条件 | `while count[char] > 1`（当前新进字符发生重复，窗口非法） |
| `remove_left` | `count[s[left]] -= 1`；`left += 1` |
| `record` 时机 | `while` 结束后（无重复字符，窗口合法），`ans = max(ans, right - left + 1)` |

```python
from collections import defaultdict


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        count = defaultdict(int)
        left = 0
        answer = 0

        for right, char in enumerate(s):
            count[char] += 1

            # 槽位 2: 窗口违规，通过收缩恢复合法
            while count[char] > 1:
                count[s[left]] -= 1
                left += 1

            # 槽位 3: 窗口合法后记录最长长度
            answer = max(answer, right - left + 1)

        return answer
```

```longest-substring-demo
```

### 复杂度分析

外层 `right` 遍历 $n$ 次；内层 `left` 只能单向右移，在整个算法运行周期中累计至多递增 $n$ 次。每个字符进窗 1 次，出窗至多 1 次，摊还时间复杂度为 $O(n)$，空间复杂度为 $O(|\Sigma|) \le O(128) = O(1)$。

---

## 2. Longest Repeating Character Replacement

### 题意

给定只含大写英文字母的字符串 `s` 和整数 `k`。最多替换 `k` 个字符，求能变成同一字符的最长子串长度。`s` 最长为 $10^5$。

一个窗口能否通过最多 $k$ 次替换变成全由同一字符构成，只取决于**窗口长度**与**窗口内最高频字符的出现次数**：

$$
\text{replacements} = \text{window\_length} - \text{max\_frequency} \le k.
$$

只要将除最高频字符之外的所有其他字符替换掉即可。

### 填模板

| 槽位 | 本题具体实现 |
|---|---|
| `state` | `count[char]` 和历史最高频次 `max_freq` |
| `add_right` | `count[char] += 1`；`max_freq = max(max_freq, count[char])` |
| `shrink` 条件 | `while right - left + 1 - max_freq > k`（需要替换的字符数超标） |
| `remove_left` | `count[s[left]] -= 1`；`left += 1` |
| `record` 时机 | 恢复合法后更新 `ans = max(ans, right - left + 1)` |

```python
from collections import defaultdict


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = defaultdict(int)
        left = 0
        max_freq = 0
        answer = 0

        for right, char in enumerate(s):
            count[char] += 1
            max_freq = max(max_freq, count[char])

            # 槽位 2: 替换成本超过 k，收缩左边界
            while right - left + 1 - max_freq > k:
                count[s[left]] -= 1
                left += 1

            # 槽位 3: 记录最大窗口
            answer = max(answer, right - left + 1)

        return answer
```

### 为什么 `max_freq` 在收缩时不需要递减？

这是滑动窗口最具代表性的**历史最优凭证机制**：
我们只关心是否能找到**比历史最大值更长**的窗口。如果收缩后当前窗口内真实最高频次变小了，`max_freq` 仍保留历史峰值：
1. 若后续没有字符打破此峰值，则新窗口绝不可能超越先前的最优解，无需浪费时间更新。
2. 只有当后续进窗字符的真实频次超过了历史 `max_freq`，才会再次刷新更长的有效窗口。

因此 `max_freq` 只增不减完全不影响最终最优解的正确性，避免了每次收缩都要在 26 个字母中找最大值的额外开销。

---

## 3. Permutation in String

### 题意

给定小写英文字符串 `s1` 和 `s2`，判断 `s2` 是否包含 `s1` 的排列。排列的充分必要条件是：**两子串长度相同，且 26 个字母的出现频次完全一致**。因此，这是一道标准的**固定长度窗口**问题，窗口长度恒为 $k = \text{len}(s1)$。

### 填模板

| 槽位 | 本题具体实现 |
|---|---|
| `state` | `need[26]` 和 `window[26]` 两个频次数组 |
| `add_right` | `window[ord(char) - ord('a')] += 1` |
| `shrink` 条件 | `if right - left + 1 > k:`（超长，只需移出 1 个字符） |
| `remove_left` | `window[ord(s2[left]) - ord('a')] -= 1`；`left += 1` |
| `record` 时机 | `if right - left + 1 == k and window == need: return True` |

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        need = [0] * 26
        window = [0] * 26

        for char in s1:
            need[ord(char) - ord('a')] += 1

        k = len(s1)
        left = 0

        for right, char in enumerate(s2):
            window[ord(char) - ord('a')] += 1

            # 槽位 2: 固定窗口超长，单次收缩
            if right - left + 1 > k:
                window[ord(s2[left]) - ord('a')] -= 1
                left += 1

            # 槽位 3: 窗口满 k 时结算比较
            if right - left + 1 == k and window == need:
                return True

        return False
```

每轮比较两个 26 位定长数组需 $O(26)$，总时间复杂度严格为 $O(26 \cdot n) = O(n)$。

---

## 4. Minimum Window Substring

### 题意

给定字符串 `s` 与 `t`，求 `s` 中覆盖 `t` 中所有字符（包含频次）的**最短子串**。若不存在则返回 `""`。

### 种类计数压缩：用 `have` 达到 $O(1)$ 判定

常规思路每次比对整个哈希表需要 $O(|\Sigma|)$。高效解法是引入 `have` 计数：
- `required = len(need)`：表示 `t` 中总共有多少**种不同字符**。
- `have`：表示当前窗口内，已有多少**种不同字符**的出现次数达到了 `t` 的要求。
- 窗口合法当且仅当：

$$
\text{have} == \text{required}.
$$

### 填模板

| 槽位 | 本题具体实现 |
|---|---|
| `state` | `need`、`window`、`have`、`required` |
| `add_right` | `window[c] += 1`；若 `window[c] == need[c]` 则 `have += 1` |
| `shrink` 条件 | `while have == required:`（已满足覆盖要求，尝试左缩求极小值） |
| `record` 时机 | **在 `while` 内部出窗前记录**：先记录当前有效窗口，再收缩 |
| `remove_left` | 若 `window[old] == need[old]` 则 `have -= 1`；`window[old] -= 1`；`left += 1` |

```python
from collections import Counter, defaultdict


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ""

        need = Counter(t)
        window = defaultdict(int)
        required = len(need)
        have = 0

        left = 0
        best_start = 0
        best_len = float('inf')

        for right, char in enumerate(s):
            window[char] += 1
            if char in need and window[char] == need[char]:
                have += 1

            # 槽位 2: 窗口已满足要求，循环收缩寻找最短
            while have == required:
                length = right - left + 1
                # 槽位 3: 收缩前先记录极小值候选
                if length < best_len:
                    best_start = left
                    best_len = length

                old = s[left]
                if old in need and window[old] == need[old]:
                    have -= 1
                window[old] -= 1
                left += 1

        if best_len == float('inf'):
            return ""
        return s[best_start:best_start + best_len]
```

---

## 5. Sliding Window Maximum

### 题意

给定整数数组 `nums` 和滑动窗口大小 `k`。窗口每次向右移动一格，返回每个窗口内的最大值。$n \le 10^5$。

朴素暴力求每个窗口最大值需 $O(nk)$，大根堆需 $O(n\log k)$。利用**双端单调队列（Monotonic Deque）**可在 $O(n)$ 时间内解决。

### 单调队列两大数学不变量

队列中**只保存数组下标**（不单独存值，存下标才能判断是否滑出边界），并严格维持两大不变量：

1. **时序单调性**：队中下标自队首向队尾严格递增：
   $$q[0] < q[1] < \dots < q[m-1]$$
2. **数值单调性**：队中下标对应的数组数值自队首向队尾严格单调递减：
   $$\text{nums}[q[0]] > \text{nums}[q[1]] > \dots > \text{nums}[q[m-1]]$$
3. **极值性质**：**队首元素 $q[0]$ 恒为当前窗口内的最大值**，可在 $O(1)$ 时间直接读取。

### 队列生存竞争法则（为什么可以丢弃队尾与队首？）

- **队尾淘汰机制（Domination Principle）**：
  当遍历到新元素 $\text{nums}[\text{right}]$ 时，若队尾下标 $\text{tail} = q[-1]$ 满足 $\text{nums}[\text{tail}] \le \text{nums}[\text{right}]$，则将 $\text{tail}$ 从队尾弹出（`pop()`）。
  *严谨证明*：对于包含 $\text{right}$ 及其未来所有的滑动窗口，旧下标 $\text{tail}$ 更早离开窗口；且在离开前其数值均不如新元素大。这意味着 $\text{tail}$ 已经**彻底失去成为最大值的机会**（比你年轻还比你强），可以被永久剔除。
- **队首过期机制（Expiration Principle）**：
  当窗口向右滑动使左边界达到 $\text{left}$ 时，若队首元素 $q[0] < \text{left}$（或固定窗口收缩时 $q[0] == \text{left}$），说明先前的最大值已滑出窗口左端，必须从队首弹出（`popleft()`）。

```sliding-window-max-demo
```

### 填模板

| 槽位 | 本题具体实现 |
|---|---|
| `state` | 下标严格递增、对应值严格单调递减的双端队列 `candidates = deque()` |
| `add_right` | 循环弹出队尾所有更小旧元素，然后将当前 `right` 入队 |
| `shrink` 条件 | `if right - left + 1 > k:`（固定窗口超长） |
| `remove_left` | 若队首最大值恰为过期元素 `candidates[0] == left`，则 `popleft()`；`left += 1` |
| `record` 时机 | `if right - left + 1 == k:` 将队首值 `nums[candidates[0]]` 追加至答案 |

```python
from collections import deque
from typing import List


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        candidates = deque()
        answer = []
        left = 0

        for right, value in enumerate(nums):
            # 槽位 1: 进窗。淘汰队尾所有劣势候选，维持严格单调递减
            while candidates and nums[candidates[-1]] <= value:
                candidates.pop()
            candidates.append(right)

            # 槽位 2: 固定窗口收缩。若队首滑出窗口，从队首剔除
            if right - left + 1 > k:
                if candidates[0] == left:
                    candidates.popleft()
                left += 1

            # 槽位 3: 窗口满 k 时记录队首最大值
            if right - left + 1 == k:
                answer.append(nums[candidates[0]])

        return answer
```

### 为什么嵌套 `while` 仍是均摊 $O(n)$？

从宏观生命周期分析，每个数组下标 $i \in [0, n-1]$：
- 入队：通过 `candidates.append(i)` 恰好入队 1 次。
- 出队：被更大新元素从队尾淘汰弹出至多 1 次，或滑出窗口从队首弹出至多 1 次。

全部元素在整个程序运行期间的进出队操作总数上界为 $2n$。因此时间复杂度严格为 $O(n)$，辅助空间复杂度为 $O(k)$。

---

## 6. Minimum Size Subarray Sum

### 题意

给定含有 $n$ 个**正整数**的数组 `nums` 和正整数 `target`。找出该数组中满足其和 $\ge \text{target}$ 的长度最小的连续子数组，并返回其长度。若不存在则返回 0。

这是“最短满足窗口”的基础题，与 LC 76 共享相同的收缩与记录时机。

### 填模板

| 槽位 | 本题具体实现 |
|---|---|
| `state` | `curr_sum`，当前窗口所有元素之和 |
| `add_right` | `curr_sum += nums[right]` |
| `shrink` 条件 | `while curr_sum >= target:`（已满足和的下界，尝试收缩寻求更短长度） |
| `record` 时机 | **在 `while` 内部收缩前记录**：`ans = min(ans, right - left + 1)` |
| `remove_left` | `curr_sum -= nums[left]`；`left += 1` |

```python
from typing import List


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        curr_sum = 0
        ans = float('inf')

        for right, val in enumerate(nums):
            # 槽位 1: 进窗更新和
            curr_sum += val

            # 槽位 2: 窗口达标，循环收缩求最短
            while curr_sum >= target:
                # 槽位 3: 出窗前记录最优解
                ans = min(ans, right - left + 1)
                curr_sum -= nums[left]
                left += 1

        return 0 if ans == float('inf') else ans
```

### 深度考点：为什么必须是“正整数”？负数为何会导致滑窗失效？

滑动窗口能成立的前提是**单调性（Monotonicity）**：
- 当所有元素均为正数时，右指针右移必然导致 `curr_sum` 单调递增；左指针右移必然导致 `curr_sum` 单调递减。
- 若数组存在**负数**，扩大窗口可能使和变小，缩小窗口可能使和变大。此时窗口的收缩条件不再具备无后效性，滑动窗口算法彻底失效！
- **负数场景的解法**：若数组包含负数求最短满足子数组（如 [LC 862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)），必须改用**前缀和数组 + 单调双端队列**在 $O(n)$ 内求解。

---

## 7. Max Consecutive Ones III

### 题意

给定由若干 `0` 和 `1` 组成的二进制数组 `nums`，并可以翻转最多 `k` 个 `0` 为 `1`。返回仅包含 `1` 的最长连续子数组的长度。

### 问题抽象：转化为最长合法窗口

“最多翻转 $k$ 个 0”等价于：**寻找一个最长连续子数组，其中包含 0 的个数不超过 $k$**。
- 合法条件：$\text{zeros} \le k$。
- 违规条件：$\text{zeros} > k$。

### 填模板

| 槽位 | 本题具体实现 |
|---|---|
| `state` | `zeros`，窗口内数字 0 的出现个数 |
| `add_right` | `if nums[right] == 0: zeros += 1` |
| `shrink` 条件 | `while zeros > k:`（0 的个数超过可翻转配额） |
| `remove_left` | `if nums[left] == 0: zeros -= 1`；`left += 1` |
| `record` 时机 | `while` 结束后（合法状态下），`ans = max(ans, right - left + 1)` |

```python
from typing import List


class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        zeros = 0
        ans = 0

        for right, val in enumerate(nums):
            # 槽位 1: 进窗更新 0 的个数
            if val == 0:
                zeros += 1

            # 槽位 2: 违规时收缩
            while zeros > k:
                if nums[left] == 0:
                    zeros -= 1
                left += 1

            # 槽位 3: 恢复合法后更新最大长度
            ans = max(ans, right - left + 1)

        return ans
```

---

## 8. Find All Anagrams in a String

### 题意

给定字符串 `s` 和 `p`，找到 `s` 中所有是 `p` 的异位词的子串，返回这些子串的**起始索引**。

这与 LC 567 完全同构，同属于定长窗口（$k = \text{len}(p)$），区别仅在于 LC 567 命中一次即返回 `True`，而本题需收集所有匹配时刻的 `left`。

### 填模板

```python
from typing import List


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(p) > len(s):
            return []

        need = [0] * 26
        window = [0] * 26

        for char in p:
            need[ord(char) - ord('a')] += 1

        k = len(p)
        left = 0
        result = []

        for right, char in enumerate(s):
            # 槽位 1: 进窗更新
            window[ord(char) - ord('a')] += 1

            # 槽位 2: 定长收缩
            if right - left + 1 > k:
                window[ord(s[left]) - ord('a')] -= 1
                left += 1

            # 槽位 3: 满窗收集结果
            if right - left + 1 == k and window == need:
                result.append(left)

        return result
```

---

## 9. 进阶核心技巧：恰好 K 个（Exact K）的双滑窗转化法

在面对诸如 [LC 992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/) 或 [LC 1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/) 时，直接滑动窗口常常卡壳。因为**“恰好 $k$ 个”不具备单调性**：右移指针可能导致恰好达标，再右移又会超标。

### 降维破局：恒等式转化

利用单调性组合，将非单调的“恰好 $k$ 个”转化为两个**具备严格单调性的“最多 $k$ 个”问题之差**：

$$
\text{Exact}(k) = \text{atMost}(k) - \text{atMost}(k - 1).
$$

对于“至多包含 $k$ 个不同元素”的窗口，若当前窗口 $[\text{left}, \text{right}]$ 合法，则以 $\text{right}$ 结尾且起点落在 $[\text{left}, \text{right}]$ 之间的**所有子数组（共 $\text{right} - \text{left} + 1$ 个）必然均合法**！

```python
from collections import defaultdict
from typing import List


class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def atMost(k_max: int) -> int:
            if k_max == 0:
                return 0
            count = defaultdict(int)
            left = 0
            distinct = 0
            total = 0

            for right, num in enumerate(nums):
                if count[num] == 0:
                    distinct += 1
                count[num] += 1

                while distinct > k_max:
                    count[nums[left]] -= 1
                    if count[nums[left]] == 0:
                        distinct -= 1
                    left += 1

                # 核心贡献：以 right 结尾且包含至多 k_max 个不同元素的子数组个数
                total += right - left + 1

            return total

        return atMost(k) - atMost(k - 1)
```

---

## 题型全景对照与填槽总结

| 题目 | 题号 | 窗口分类 | `state` 维护对象 | 收缩条件与方式 | 记录时机与公式 |
|---|---|---|---|---|---|
| **无重复最长子串** | LC 3 | 最长合法 | 字符出现频次 | `while count[c] > 1` | 循环外 `ans = max(ans, len)` |
| **最大连续1的个数III** | LC 1004 | 最长合法 | 0 的个数 `zeros` | `while zeros > k` | 循环外 `ans = max(ans, len)` |
| **替换后最长重复字符** | LC 424 | 最长合法 | 频次 + `max_freq` | `while len - max_freq > k` | 循环外 `ans = max(ans, len)` |
| **长度最小子数组** | LC 209 | 最短满足 | 区间元素和 `sum` | `while sum >= target` | 循环内 `ans = min(ans, len)` |
| **最小覆盖子串** | LC 76 | 最短满足 | `need/window` + `have` | `while have == required` | 循环内 `ans = min(ans, len)` |
| **字符串排列** | LC 567 | 固定长度 | 26 位定长频次表 | `if len > k` (出窗 1 个) | 满窗时比对频次返回 `True` |
| **找到所有异位词** | LC 438 | 固定长度 | 26 位定长频次表 | `if len > k` (出窗 1 个) | 满窗时匹配追加 `left` |
| **滑动窗口最大值** | LC 239 | 固定长度 | 单调递减下标 deque | `if len > k` (队首过期则出) | 满窗时直接读取 `nums[deque[0]]` |

---

## 常见陷阱与调试清单

| 常见错误 | 根本原因与机制 | 标准正确写法 |
|---|---|---|
| **变长窗口误用 `if`** | 新进元素可能导致严重违规，需要连续收缩多次才能重新恢复合法 | 变长窗口恢复不变量必须使用 `while` |
| **定长窗口机械套 `while`** | 固定窗口每轮进 1 个元素，超长量恒为 1，写 `while` 虽对但遮蔽了定长物理语义 | 固定窗口统一使用 `if right - left + 1 > k` |
| **区间长度漏掉 `+ 1`** | 混淆了两点间跨度与闭区间点集势（元素个数）的区别 | 闭区间元素个数必须为 `right - left + 1` |
| **最长窗口在收缩前记录** | 收缩前窗口处于违规非法状态，此时记录会导致非解混入最优值 | 最长窗口必须在 `while` 收缩完成并重获合法后更新 `max` |
| **最短窗口在收缩后记录** | 收缩后窗口可能已经不再满足条件，错过了刚刚满足时的极短可能 | 最短窗口必须在 `while` 内部收缩动作发生之前更新 `min` |
| **先移动 `left` 后扣除状态** | 指针先递增会导致被减去的是窗口外的下一个元素，引发状态错乱 | 必须先逆向扣除 `items[left]`，再执行 `left += 1` |
| **单调队列只存数值** | 单纯存值无法判断该极大值究竟是何时滑出窗口左端边界 | 单调队列必须存储**下标（indices）**，按需读值 |

---

## 拿到新题怎么套模板

解任何滑动窗口新题，按以下四步顺序推演：

```text
步骤 1: 确定窗口是否固定长度？
        -> 固定长度 k：用 if len > k 控窗，满 k 结算。
        -> 变长窗口：继续执行步骤 2 与 3。

步骤 2: 明确题目目标属性？
        -> 求“最长合法”：while 违规收缩，循环后更新 max。
        -> 求“最短满足”：while 达标收缩，循环内更新 min。
        -> 求“恰好 K 个”：拆解为 atMost(K) - atMost(K-1)。

步骤 3: 确定状态更新与逆向恢复方式？
        -> 新增 items[right] 时，哪些变量可以 O(1) 增量维护？
        -> 剔除 items[left] 时，如何以完全对称的方式逆向撤销？

步骤 4: 检查单调性假设是否成立？
        -> 指针向右滑动时，状态指标是否单调变化？
        -> 如果包含负数或不满足单调性，滑动窗口失效，切换为前缀和/哈希/单调栈。
```

---

## 模板速查

```text
【滑动窗口三槽规则总结】
进窗增量加 right，
超界违规缩 left，
最长合法出圈记，
最短满足进圈记，
固定大小满 k 记！
```

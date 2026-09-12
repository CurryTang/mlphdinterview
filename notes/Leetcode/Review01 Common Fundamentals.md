# 复习卡片：常考基础题 (Review Flashcards · Core Fundamentals)

本篇为算法面试核心复习卡片：精简提炼**题目定义**、**核心思路**、**关键代码**、**复杂度速记**。点击卡片标题即可展开复习。

---

## 模块一：核心底层与高频手撕算法 (Core Fundamentals)

### 1. 归并排序 (Merge Sort)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">基础 01</span>
  <span class="review-card-title">归并排序 (Merge Sort)</span>
  <span class="review-card-tag">分治 · 递归 · 稳定</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

对长度为 $n$ 的无序数组进行排序，要求在最坏情况下时间复杂度仍严格保证为 $O(n \log n)$，且维持相同元素的相对先后次序（具备稳定性）。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

经典分治（Divide & Conquer）三步法：
1. **切分 (Divide)**：计算中点 $mid = \lfloor (l + r) / 2 \rfloor$，将数组均分为左、右两半段。
2. **解决 (Conquer)**：递归对左半段与右半段分别排序，直到子数组长度 $\le 1$。
3. **合并 (Combine)**：双指针从前往后线性扫描，较小者优先放入合并数组；若两数相等，优先取左半段元素以保持稳定性。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def merge_sort(nums: list[int]) -> list[int]:
    if len(nums) <= 1:
        return nums
    mid = len(nums) // 2
    left, right = merge_sort(nums[:mid]), merge_sort(nums[mid:])
    
    # 核心双指针归并 (<= 保证稳定性)
    res, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            res.append(left[i]); i += 1
        else:
            res.append(right[j]); j += 1
    return res + left[i:] + right[j:]
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：最好 $O(n \log n)$ / 最坏 $O(n \log n)$ / 平均 $O(n \log n)$（递归树高 $\log n$，每层所有合并总代价固定为 $O(n)$）
- **辅助空间**：$O(n)$（合并时的临时数组）+ $O(\log n)$（递归调用栈）
- **稳定性**：**稳定**（相等元素左侧优先归并）

</div>

</div>
</details>

---

### 2. 快速排序 (Quick Sort)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">基础 02</span>
  <span class="review-card-title">快速排序 (Quick Sort)</span>
  <span class="review-card-tag">分治 · 原地划分 · 不稳定</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

对长度为 $n$ 的无序数组进行就地升序排序（In-place Sort）。要求平均时间复杂度达到 $O(n \log n)$，且除递归调用栈外不使用额外辅助数据结构。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**划分（Partitioning）先行**：
1. **随机选主元 (Randomized Pivot)**：通过 `random.randint(l, r)` 随机选择一个元素作为 pivot 并与末尾 $r$ 交换，彻底破坏对抗性输入（如升序/降序数组），避免退化为单链表最坏 $O(n^2)$。
2. **独立划分 (Partition 函数)**：Lomuto 划分维护指针 $i$ 作为 $\le pivot$ 区域的右边界。遍历区间 $[l, r - 1]$，将所有 $\le pivot$ 的元素置换到左侧；最后将 pivot 与 $nums[i]$ 交换归位，返回最终切分下标 $p = i$。
3. **分治递归 (Recurse)**：递归处理切分点左右两半：`[l, p - 1]` 与 `[p + 1, r]`。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
import random

def partition(nums: list[int], l: int, r: int) -> int:
    # 1. 随机选主元并与末尾交换，避免最坏 O(n^2) 退化
    rand_idx = random.randint(l, r)
    nums[rand_idx], nums[r] = nums[r], nums[rand_idx]

    # 2. 核心 Lomuto 划分：i 维护 <= pivot 的右边界
    pivot, i = nums[r], l
    for j in range(l, r):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[r] = nums[r], nums[i]
    return i

def quick_sort(nums: list[int], l: int, r: int) -> None:
    if l >= r:
        return
    p = partition(nums, l, r)
    quick_sort(nums, l, p - 1)
    quick_sort(nums, p + 1, r)
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：最好 $O(n \log n)$ / 最坏 $O(n^2)$（极度倾斜切分）/ 平均 $O(n \log n)$
- **辅助空间**：$O(\log n)$（递归栈，最坏 $O(n)$）
- **稳定性**：**不稳定**（长距离跨越式交换破坏相对顺序）

</div>

</div>
</details>

---

### 3. 动态数组实现 (Dynamic Array / Vector)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">基础 03</span>
  <span class="review-card-title">动态数组实现 (Dynamic Array)</span>
  <span class="review-card-tag">连续内存 · 几何倍增 · 均摊分析</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

基于固定大小的连续内存块从零实现可自动扩容的动态数组（类似 Python 的 `list` 或 C++ 的 `std::vector`），支持下标随机访问 `get(i)`、尾部追加 `push_back(val)`、尾部弹出 `pop_back()` 以及容量耗尽时的倍增扩容。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心心智模型与均摊常数级依据：
1. **连续内存与双计数**：底层维护固定容量的数组，用 `cap` 记录物理容量，用 `size` 记录实际有效元素个数。
2. **几何倍增扩容 (Geometric Doubling)**：当 `size == cap` 时触发扩容，申请 $2 \times cap$ 的新连续内存，将旧元素全部拷贝过去，再释放原空间。
3. **均摊分析 (Amortized $O(1)$)**：扩容单次需要 $O(n)$ 数据搬迁，但发生频率呈指数级衰减。从容量 1 扩至 $n$，总复制次数为 $1 + 2 + 4 + \dots + n \le 2n$ 次，分摊到 $n$ 次插入上，单次追加均摊成本严格为 $O(1)$。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
class DynamicArray:
    def __init__(self, capacity: int = 2):
        self.cap, self.size = capacity, 0
        self.arr = [None] * self.cap

    def push_back(self, val: int) -> None:
        # 核心：容量耗尽时几何倍增扩容，均摊 O(1)
        if self.size == self.cap:
            self.cap *= 2
            new_arr = [None] * self.cap
            for i in range(self.size):
                new_arr[i] = self.arr[i]
            self.arr = new_arr
        self.arr[self.size] = val
        self.size += 1

    def pop_back(self) -> int:
        self.size -= 1
        return self.arr[self.size]

    def get(self, i: int) -> int:
        return self.arr[i]
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **随机读写 `get`/`set`**：$O(1)$（连续内存地址公式 $base + i \times size$ 直接寻址）
- **尾部追加 `push_back`**：**均摊 $O(1)$**（最坏扩容时 $O(n)$）
- **尾部弹出 `pop_back`**：$O(1)$
- **空间利用率**：$\ge 50\%$

</div>

</div>
</details>

---

### 4. 二分查找边界模板 (Binary Search Bounds)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">基础 04</span>
  <span class="review-card-title">二分查找边界模板 (Binary Search Bounds)</span>
  <span class="review-card-tag">有序检索 · 开闭区间不变量 · 边界收敛</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

在单调非递减的有序整数数组中，查找目标值 `target` 出现的**首个位置（最左边界 / Lower Bound）**。若目标不存在，返回其应插入的索引位置以维持有序。要求时间复杂度 $O(\log n)$，且绝对杜绝死循环。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心是**严格维护区间不变量（Loop Invariant）**：
1. **区间定义**：采用标准的**左闭右闭区间 $[l, r]$**，初始 $l = 0, r = len(nums) - 1$。
2. **防溢出中点**：使用 $mid = l + \lfloor (r - l) / 2 \rfloor$。
3. **收缩决策（找最左目标）**：
   - 若 $nums[mid] \ge target$：目标可能在 $mid$ 或其左侧，收缩右界 $r = mid - 1$ 继续向左探查。
   - 若 $nums[mid] < target$：目标必然在 $mid$ 右侧，收缩左界 $l = mid + 1$。
4. **收敛性质**：循环条件 $l \le r$，终止时必满足 $l = r + 1$，最终指针 $l$ 恰好收敛在首个满足 $\ge target$ 的索引位置。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def search_lower_bound(nums: list[int], target: int) -> int:
    l, r = 0, len(nums) - 1
    # 严格维护左闭右闭区间 [l, r]
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] >= target:
            r = mid - 1  # 尝试往左寻找更小边界
        else:
            l = mid + 1
    return l  # 终止时 l 落在首个 >= target 的位置
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(\log n)$（每轮直接排除半数候选空间）
- **辅助空间**：$O(1)$（迭代实现无递归栈开销）
- **终止不变量**：循环结束时必定 $l = r + 1$

</div>

</div>
</details>

---

### 5. 拒绝采样：用 Rand7 实现 Rand10 (Rejection Sampling)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">基础 05</span>
  <span class="review-card-title">拒绝采样：用 Rand7 实现 Rand10 (Rejection Sampling)</span>
  <span class="review-card-tag">二维展平 · 能除尽的最大前缀 · 期望调用 2.45 次</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

已知 API `rand7()` 可以等概率返回 $1 \dots 7$ 的随机整数。请设计并实现 `rand10()`，使其等概率生成 $1 \dots 10$ 的随机整数。不得使用任何外部随机库，且必须严格保证生成每个数字的概率均为 $1/10$。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**多维均匀网格展平（2D Grid Flattening）+ 拒绝采样（Rejection Sampling）**：
1. **空间构造（进制/网格展平）**：单次 `rand7()` 仅有 7 种等可能状态，无法直接覆盖 10。调用两次 `rand7()`，构造成 $7 \times 7 = 49$ 种独立且等概率的二元网格状态：
   $$x = (rand7() - 1) \times 7 + rand7() \in [1, 49]$$
   每个离散点发生的概率严格等于 $\frac{1}{7} \times \frac{1}{7} = \frac{1}{49}$。
2. **为什么不能直接取模**：$49$ 不能被 $10$ 整除。若直接取模，前 9 个数字会出现 5 次（概率 $5/49$），而最后一个数字只出现 4 次（概率 $4/49$），破坏等概率性。
3. **截取能被整除的最大前缀（拒绝采样）**：
   - 只保留前缀 $1 \dots 40$（$40$ 是小于 49 且能被 10 整除的最大倍数），每个数字恰好对应 4 种状态，接受概率为 $\frac{40}{49}$，通过 `(x - 1) % 10 + 1` 严格等概率映射到 $1 \dots 10$；
   - 若命中 $41 \dots 49$（共 9 种状态），果断丢弃（Reject），进入下一轮循环重试。
4. **通用套路 ($randM \to randN$)**：选最小 $k$ 使 $M^k \ge N$，展开生成 $1 \dots M^k$，保留 $limit = \lfloor M^k / N \rfloor \times N$，落在 $[1, limit]$ 内即返回，超过则重试。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def rand10() -> int:
    while True:
        # 1. 两次 rand7() 构造 [1, 49] 均匀独立离散空间
        x = (rand7() - 1) * 7 + rand7()
        # 2. 保留能被 10 整除的最大前缀 [1, 40]（每数 4 次，绝对等概）
        if x <= 40:
            return (x - 1) % 10 + 1
        # 41..49 拒绝丢弃，继续下一轮重采
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度（期望）**：$O(1)$，平均调用 $rand7()$ 次数为 $2 \times \frac{49}{40} = 2.45$ 次（几何分布期望 $1/p$）
- **时间复杂度（最坏）**：$O(\infty)$（理论上存在无限连续拒绝的极端情况，发生概率为 0）
- **辅助空间**：$O(1)$（就地变量计算，无额外开销）
- **核心心智模型**：多次采样拼网格，除不尽的前缀丢弃重来

</div>

</div>
</details>

---

## 模块二：数组与哈希核心题组 (Arrays & Hashing)

### 6. 存在重复元素 (Contains Duplicate)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 01</span>
  <span class="review-card-title">存在重复元素 (Contains Duplicate)</span>
  <span class="review-card-tag">哈希集合 · 早期退出 · 一次遍历</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

给定一个整数数组 `nums`。如果任一数值在数组中出现至少两次，返回 `True`；如果数组中每个元素互不相同，返回 `False`。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**哈希集合实时查重与早期退出（Early Exit）**：
1. **哈希集合维护**：初始化空哈希集合 `seen = set()`。
2. **流式遍历判重**：遍历数组中的每个元素 `x`：
   - 若 `x` 已存在于 `seen` 中，说明存在重复，直接返回 `True` 终止遍历；
   - 若 `x` 不在 `seen` 中，将 `x` 加入 `seen`。
3. **兜底返回**：整轮扫描完毕未触发早期退出，说明所有元素互不相同，返回 `False`。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def contains_duplicate(nums: list[int]) -> bool:
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(n)$（单次集合查找与插入平均 $O(1)$，最坏哈希冲突退化为 $O(n)$）
- **辅助空间**：$O(n)$（无重复的最坏情况下集合需容纳全量 $n$ 个元素）
- **关键心智模型**：空间换时间，边查边存，首次碰撞即时返回

</div>

</div>
</details>

---

### 7. 有效的字母异位词 (Valid Anagram)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 02</span>
  <span class="review-card-title">有效的字母异位词 (Valid Anagram)</span>
  <span class="review-card-tag">频次数组 · ASCII 差值 · 长度剪枝</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

给定两个字符串 `s` 和 `t`，判断 `t` 是否是 `s` 的字母异位词（由相同字符以不同排列次序构成，且每个字符出现的频次完全相等）。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**字符计数与差额平衡校验**：
1. **长度预检剪枝**：若 `len(s) != len(t)`，字符总数不对等，必不可能构成异位词，直接返回 `False`。
2. **定长频次数组**：针对小写英文字母表，初始化长度为 26 的计数数组 `counts = [0] * 26`。
3. **加减平衡统计**：同时遍历两个字符串（使用 `zip(s, t)`）。对 `s` 中的字符频次做加法 `+1`，对 `t` 中的字符频次做减法 `-1`。
4. **残差全零断言**：遍历 `counts` 数组。若所有字符频次均平衡归零，则返回 `True`；出现任何非零残差则说明频次不符，返回 `False`。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def is_anagram(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False
    counts = [0] * 26
    for c1, c2 in zip(s, t):
        counts[ord(c1) - ord('a')] += 1
        counts[ord(c2) - ord('a')] -= 1
    return all(c == 0 for c in counts)
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(n)$（单次同步遍历两个长度为 $n$ 的字符串）
- **辅助空间**：$O(1)$（固定 26 槽计数数组，常数级开销；Unicode 字符集使用哈希表时为 $O(k)$，其中 $k$ 为不同字符数）
- **关键心智模型**：加减相互对冲，绝对平衡等价于异位构词

</div>

</div>
</details>

---

### 8. 两数之和 (Two Sum)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 03</span>
  <span class="review-card-title">两数之和 (Two Sum)</span>
  <span class="review-card-tag">哈希查找 · 差值补数 · 前缀存储</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

给定一个整数数组 `nums` 和一个整数目标值 `target`，请在数组中找出和为目标值 `target` 的那两个整数，并返回它们的数组下标。假设每组输入只对应唯一解，且同一个元素不能使用两次。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**前缀哈希映射与补数反向探测**：
1. **差值补数定义**：对于当前扫描的元素 $x = nums[i]$，与其配对的目标值必为 $\text{complement} = target - x$。
2. **前缀字典反查**：维护字典 `lookup`（映射规则为 `数值 -> 历史下标`）。
   - 在将当前数字存入前，优先检查 `complement` 是否已在 `lookup` 中：
     - 若命中，说明先前已遇到过该互补元素，直接返回 `[lookup[complement], i]`；
     - 若未命中，将当前数及其下标写入字典 `lookup[x] = i`，继续向后扫描。
3. **避免自配对**：由于仅在已处理的历史前缀中探查，当前数字尚未存入表内，天然杜绝了取用同一元素两次的情况。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    lookup = {}
    for i, x in enumerate(nums):
        complement = target - x
        if complement in lookup:
            return [lookup[complement], i]
        lookup[x] = i
    return []
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(n)$（单次线性扫描，哈希表平均查找与写入均为 $O(1)$）
- **辅助空间**：$O(n)$（哈希表最多存储 $n - 1$ 个元素与下标）
- **关键心智模型**：记录历史前缀，等待未来补数撞击匹配

</div>

</div>
</details>

---

### 9. 字母异位词分组 (Group Anagrams)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 04</span>
  <span class="review-card-title">字母异位词分组 (Group Anagrams)</span>
  <span class="review-card-tag">频次元组 · 典范哈希键 · 字典聚合</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

给你一个字符串数组 `strs`，请将所有字母异位词组合在一起，并以列表形式返回结果。返回结果的顺序可以是任意的。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**多重集特征向量化与典范哈希键映射**：
1. **不变特征键构建**：异位词的字符构成与各字符频次完全同构，需将其规约为全局唯一的哈希键。
   - 排序键：`"".join(sorted(s))`，单词耗时 $O(k \log k)$；
   - **频次元组键（最优解）**：统计 26 个字符的出现频次，转为不可变元组 `tuple(counts)` 作为字典键，单词处理耗时降为严格线性 $O(k)$。
2. **多对一哈希聚合**：
   - 使用 `collections.defaultdict(list)`，以 26 维频次元组为 key，将原始字符串归并追加到对应的列表中。
   - 最终提取字典的所有 values 列表输出。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
from collections import defaultdict

def group_anagrams(strs: list[str]) -> list[list[str]]:
    groups = defaultdict(list)
    for s in strs:
        # 统计 26 字符频次，以不可变元组为字典键
        counts = [0] * 26
        for c in s:
            counts[ord(c) - ord('a')] += 1
        groups[tuple(counts)].append(s)
    return list(groups.values())
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(n \cdot k)$（$n$ 为字符串个数，$k$ 为单个字符串最大长度，遍历字符计数耗时 $O(k)$）
- **辅助空间**：$O(n \cdot k)$（字典存储全部字符串与元组键）
- **关键心智模型**：字符频次元组标准化，多对一归约聚集

</div>

</div>
</details>

---

### 10. 前 K 个高频元素 (Top K Frequent Elements)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 05</span>
  <span class="review-card-title">前 K 个高频元素 (Top K Frequent Elements)</span>
  <span class="review-card-tag">桶排序 · 频次倒排 · 线性时间</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

给你一个整数数组 `nums` 和一个整数 `k`，请返回其中出现频率前 `k` 高的元素。可以按任意顺序返回答案。算法时间复杂度必须优于 $O(n \log n)$。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**利用频次天然有界特性进行桶排序（倒排索引）**：
1. **哈希频次统计**：使用 `Counter(nums)` 统计每个唯一元素的出现次数。不同元素数量记为 $m \le n$。
2. **三大解法复杂度对比**：
   - 全排序：$O(m \log m)$；
   - 维持大小为 $k$ 的最小堆：$O(n + m \log k)$；
   - **桶排序 / 频次倒排（严格线性 $O(n)$ 最优解）**：
     - 任何元素的出现频次必然介于 $1$ 到 $n$ 之间；
     - 建立 $n + 1$ 个桶 `buckets = [[] for _ in range(n + 1)]`，其中下标 $i$ 存储所有出现频次恰好为 $i$ 的元素列表；
     - 将统计得到的 `(num, freq)` 填入对应的 `buckets[freq]` 中。
3. **逆向贪心提取**：
   - 从最大频次 $n$ 倒序扫描至 1，依次收集桶内元素至结果列表，集满 $k$ 个后直接返回。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
from collections import Counter

def top_k_frequent(nums: list[int], k: int) -> list[int]:
    counts = Counter(nums)
    # 桶下标代表频次，范围 0..len(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, freq in counts.items():
        buckets[freq].append(num)
        
    res = []
    # 从最大频次降序收集 k 个元素
    for freq in range(len(nums), 0, -1):
        for num in buckets[freq]:
            res.append(num)
            if len(res) == k:
                return res
    return res
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(n)$（频次统计 $O(n)$，建桶 $O(m)$，逆向扫描收集 $k$ 个数 $O(n)$，严格线性）
- **辅助空间**：$O(n)$（哈希表与 $n + 1$ 个桶的列表开销）
- **关键心智模型**：频次天然以 $n$ 为界，倒排桶扫描击穿对数瓶颈

</div>

</div>
</details>

---

### 11. 字符串的编码与解码 (Encode and Decode Strings)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 06</span>
  <span class="review-card-title">字符串的编码与解码 (Encode and Decode Strings)</span>
  <span class="review-card-tag">长度前缀 · 字符流分块 · 无歧义边界</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

设计一个算法，将一个字符串列表编码为一个单一的复合字符串，并能够将该复合字符串无损解码还原为原始字符串列表。字符串可能包含任何 256 个可能的 ASCII / Unicode 字符（包括各种分隔符、标点、换行与特殊控制符）。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**网络分块传输协议思想（Length-Prefix Framing）**：
1. **静态分隔符失效陷阱**：使用特定分隔符（如 `,`、`#`）或转义字符时，若原始字符串内恰好包含这些字符，极易引发越界与切分歧义。
2. **长度前缀协议设计**：
   - 编码格式：`"{长度}#{原文}"`。例如 `["neet", "code"] -> "4#neet4#code"`。
   - 解码流程：
     - 维护游标 $i$。从 $i$ 开始定位首个定界符 `'#'` 的下标 $j$；
     - 切片 `s[i:j]` 解析为整数，获取后续子串的确切长度 $L = \text{int}(s[i:j])$；
     - 从 $j + 1$ 向后精确截取 $L$ 个字符：`s[j + 1 : j + 1 + L]` 即为完整原文；
     - 游标跃迁更新为 $i = j + 1 + L$，进入下一块提取流程。
3. **抗干扰特性**：无论原文内部包含多少个 `'#'` 或连续数字，由于解码器依据严格预读取的长度界定内容边界，绝不会发生歧义。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
class Codec:
    def encode(self, strs: list[str]) -> str:
        # 协议格式：长度 + '#' + 原文
        res = []
        for s in strs:
            res.append(f"{len(s)}#{s}")
        return "".join(res)

    def decode(self, s: str) -> list[str]:
        res, i = [], 0
        while i < len(s):
            # 定位长度后的首个定界符 '#'
            j = s.find('#', i)
            length = int(s[i:j])
            # 根据提取出的长度精确切片原文
            res.append(s[j + 1 : j + 1 + length])
            i = j + 1 + length
        return res
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：编码 $O(N)$，解码 $O(N)$（$N$ 为所有字符串字符总和，单趟线性切分）
- **辅助空间**：$O(1)$（除存放编解码结果的缓冲区外，仅使用常数级游标指针）
- **关键心智模型**：元数据（长度）与载荷（内容）解耦，定长切片封死歧义空间

</div>

</div>
</details>

---

### 12. 除自身以外数组的乘积 (Product of Array Except Self)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 07</span>
  <span class="review-card-title">除自身以外数组的乘积 (Product of Array Except Self)</span>
  <span class="review-card-tag">前后缀积分解 · 两次扫描 · 常数空间</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

给你一个整数数组 `nums`，返回一个数组 `res`，其中 `res[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。题目严格要求：**不能使用除法运算**，且时间复杂度必须为 $O(n)$。进阶要求：额外空间复杂度为 $O(1)$（输出数组不计入辅助空间）。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**前后缀乘积正交分解与原地双向扫描**：
1. **对称分解原理**：位置 $i$ 的除自身乘积，结构上等于其**左侧前缀积**与**右侧后缀积**的交乘：
   $$res[i] = \left(\prod_{j=0}^{i-1} nums[j]\right) \times \left(\prod_{j=i+1}^{n-1} nums[j]\right)$$
2. **两遍原地扫描（$O(1)$ 辅助空间）**：
   - **第一遍（前向求前缀积并填入结果数组）**：
     - 初始化 `res = [1] * n`，维护标量 `prefix = 1`。
     - 从左往右扫描：`res[i] = prefix`，随后更新 `prefix *= nums[i]`。此时 `res[i]` 存储位置 $i$ 左边所有数的乘积。
   - **第二遍（反向维护后缀积并滚入结果）**：
     - 维护标量 `postfix = 1`。
     - 从右往左倒序扫描：当前位置左积乘以右积 `res[i] *= postfix`，随后更新 `postfix *= nums[i]`。
3. **消除额外数组**：直接复用返回值数组存储前缀积，后缀积通过单一标量在线累乘，达成严格 $O(1)$ 辅助空间。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def product_except_self(nums: list[int]) -> list[int]:
    n = len(nums)
    res = [1] * n
    
    # 1. 前向扫描：将左侧前缀积填入 res[i]
    prefix = 1
    for i in range(n):
        res[i] = prefix
        prefix *= nums[i]
        
    # 2. 反向扫描：用单变量 postfix 维护后缀积并乘入 res[i]
    postfix = 1
    for i in range(n - 1, -1, -1):
        res[i] *= postfix
        postfix *= nums[i]
        
    return res
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(n)$（正向与反向各扫描一次，总计 $2n$ 次乘法操作）
- **辅助空间**：$O(1)$（输出数组除外，仅使用 `prefix` 与 `postfix` 两个标量指针）
- **关键心智模型**：对称分解左右积，前向落盘，后向动滚合并

</div>

</div>
</details>

---

### 13. 有效的数独 (Valid Sudoku)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 08</span>
  <span class="review-card-title">有效的数独 (Valid Sudoku)</span>
  <span class="review-card-tag">行/列/宫格 · 坐标展平 · 并行判重</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

请你判断一个 $9 \times 9$ 的数独网格是否有效。只需要根据以下规则，验证已经填入的数字（未填入单元格以 `'.'` 表示）：
1. 数字 `1-9` 在每一行只能出现一次。
2. 数字 `1-9` 在每一列只能出现一次。
3. 数字 `1-9` 在每一个以粗实线分隔的 $3 \times 3$ 宫格内只能出现一次。
*注：只需要校验当前已有数字是否合法，无需判断数独是否有解。*

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**宫格坐标一维映射与三向哈希并行判重**：
1. **宫格索引展平映射**：
   - 坐标 $(r, c)$ 所属的 $3 \times 3$ 宫格子网格编号可通过整除 3 计算：
     $$box\_idx = (r // 3) \times 3 + (c // 3) \in [0, 8]$$
2. **三组集合同步维护**：
   - 创建 9 个行集合 `rows`、9 个列集合 `cols`、9 个宫格集合 `boxes`。
   - 单次遍历整个 $9 \times 9$ 网格，跳过空单元格 `'.'`；
   - 对当前提取的数字 `val`，同时校验是否在对应行、列或宫格集合中存在：
     - 若任一集合命中，说明存在同维度数字冲突，立即返回 `False`；
     - 若均未命中，分别向三个集合中注册加入 `val`。
3. 全局扫描无冲突即返回 `True`。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def is_valid_sudoku(board: list[list[str]]) -> bool:
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    
    for r in range(9):
        for c in range(9):
            val = board[r][c]
            if val == '.':
                continue
            box_idx = (r // 3) * 3 + (c // 3)
            # 行、列、3x3 宫格三维并行冲突校验
            if val in rows[r] or val in cols[c] or val in boxes[box_idx]:
                return False
            rows[r].add(val)
            cols[c].add(val)
            boxes[box_idx].add(val)
    return True
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(1)$（固定遍历 $9 \times 9 = 81$ 个单元格）
- **辅助空间**：$O(1)$（$3 \times 9 = 27$ 个集合，每个集合元素上限不超过 9）
- **关键心智模型**：$(r//3) \times 3 + (c//3)$ 展平子宫格，三向哈希协同判重

</div>

</div>
</details>

---

### 14. 最长连续序列 (Longest Consecutive Sequence)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">数组 09</span>
  <span class="review-card-title">最长连续序列 (Longest Consecutive Sequence)</span>
  <span class="review-card-tag">哈希集合 · 前驱探测 · 严格线性</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>

给定一个未排序的整数数组 `nums`，找出数字连续的最长序列（不要求元素在原数组中连续，如 `[100, 4, 200, 1, 3, 2]` 的最长递增连续序列是 `[1, 2, 3, 4]`，长度为 4）的长度。设计并实现时间复杂度为 $O(n)$ 的算法。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>

核心在于**哈希集合 $O(1)$ 寻址与序列头部前驱判定（杜绝重复扫描）**：
1. **哈希集合去重与常数寻址**：
   - 将数组转换为哈希集合 `num_set = set(nums)`，提供 $O(1)$ 的元素存在性检验。
2. **前驱探测剪枝（唯一起点判定）**：
   - 遍历集合中的每一个数字 $x$：
   - **关键判断**：检查 $x - 1$ 是否在 `num_set` 中：
     - 若 $x - 1 \in num\_set$：说明 $x$ 并非连续序列的起始点（其左侧存在更小的前驱），**直接跳过**；
     - 若 $x - 1 \notin num\_set$：说明 $x$ 必然是某条连续链条的**唯一合法起点**！
3. **单向线性延伸**：
   - 确认 $x$ 为起点后，运行 `while current_num + 1 in num_set` 不断向右探索链条长度，更新全局最大长度。
4. **严格线性时间保证**：
   - 每个元素仅在两处被触碰：外层循环做一次前驱剪枝检查；内层循环仅由其所属序列的唯一起点驱动遍历一次。没有任何元素会被多重遍历，总计算步数严格小于等于 $2n$。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def longest_consecutive(nums: list[int]) -> int:
    num_set = set(nums)
    max_len = 0
    
    for x in num_set:
        # 核心剪枝：仅当 x-1 不在集合中时，x 才是连续序列的起点
        if x - 1 not in num_set:
            current_num = x
            current_len = 1
            
            while current_num + 1 in num_set:
                current_num += 1
                current_len += 1
                
            max_len = max(max_len, current_len)
            
    return max_len
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：$O(n)$（构建集合 $O(n)$；起点前驱剪枝确保内层 `while` 总执行次数至多为 $n$）
- **辅助空间**：$O(n)$（哈希集合存储去重后的数组元素）
- **关键心智模型**：无前驱者方为起点，锁定起点单向延伸，杜绝冗余重复扫描

</div>

</div>
</details>




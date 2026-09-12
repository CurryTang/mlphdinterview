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

> 🔗 **LeetCode 链接**：[LeetCode 912 · Sort an Array](https://leetcode.com/problems/sort-an-array/) — `https://leetcode.com/problems/sort-an-array/`

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

if __name__ == "__main__":
    assert merge_sort([5, 2, 3, 1]) == [1, 2, 3, 5]
    assert merge_sort([5, 1, 1, 2, 0, 0]) == [0, 0, 1, 1, 2, 5]
    assert merge_sort([]) == []
    assert merge_sort([42]) == [42]
    print("✅ Card 01 (Merge Sort) all tests passed!")
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

> 🔗 **LeetCode 链接**：
> - [LeetCode 912 · Sort an Array](https://leetcode.com/problems/sort-an-array/) — `https://leetcode.com/problems/sort-an-array/`
> - [LeetCode 215 · Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array/) — `https://leetcode.com/problems/kth-largest-element-in-an-array/`

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

if __name__ == "__main__":
    arr1 = [5, 2, 3, 1]
    quick_sort(arr1, 0, len(arr1) - 1)
    assert arr1 == [1, 2, 3, 5]
    arr2 = [5, 1, 1, 2, 0, 0]
    quick_sort(arr2, 0, len(arr2) - 1)
    assert arr2 == [0, 0, 1, 1, 2, 5]
    print("✅ Card 02 (Quick Sort) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 1929 · Concatenation of Array](https://leetcode.com/problems/concatenation-of-array/) — `https://leetcode.com/problems/concatenation-of-array/`

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

if __name__ == "__main__":
    arr = DynamicArray(capacity=2)
    arr.push_back(10)
    arr.push_back(20)
    assert arr.get(0) == 10 and arr.get(1) == 20
    assert arr.cap == 2
    arr.push_back(30)
    assert arr.cap == 4
    assert arr.pop_back() == 30
    assert arr.size == 2
    print("✅ Card 03 (Dynamic Array) all tests passed!")
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

> 🔗 **LeetCode 链接**：
> - [LeetCode 704 · Binary Search](https://leetcode.com/problems/binary-search/) — `https://leetcode.com/problems/binary-search/`
> - [LeetCode 34 · Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) — `https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/`

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

if __name__ == "__main__":
    nums = [1, 2, 2, 2, 3, 5]
    assert search_lower_bound(nums, 2) == 1
    assert search_lower_bound(nums, 3) == 4
    assert search_lower_bound(nums, 4) == 5
    assert search_lower_bound(nums, 0) == 0
    assert search_lower_bound(nums, 6) == 6
    print("✅ Card 04 (Binary Search Bounds) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 470 · Implement Rand10() Using Rand7()](https://leetcode.com/problems/implement-rand10-using-rand7/) — `https://leetcode.com/problems/implement-rand10-using-rand7/`

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

if __name__ == "__main__":
    import random
    def rand7():
        return random.randint(1, 7)
    samples = [rand10() for _ in range(20000)]
    for num in range(1, 11):
        freq = samples.count(num)
        assert 1600 <= freq <= 2400
    print("✅ Card 05 (Rejection Sampling Rand7->Rand10) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 217 · Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) — `https://leetcode.com/problems/contains-duplicate/`

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

if __name__ == "__main__":
    assert contains_duplicate([1, 2, 3, 1]) is True
    assert contains_duplicate([1, 2, 3, 4]) is False
    assert contains_duplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) is True
    print("✅ Card 06 (Contains Duplicate) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 242 · Valid Anagram](https://leetcode.com/problems/valid-anagram/) — `https://leetcode.com/problems/valid-anagram/`

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

if __name__ == "__main__":
    assert is_anagram("anagram", "nagaram") is True
    assert is_anagram("rat", "car") is False
    assert is_anagram("a", "ab") is False
    print("✅ Card 07 (Valid Anagram) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 1 · Two Sum](https://leetcode.com/problems/two-sum/) — `https://leetcode.com/problems/two-sum/`

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

if __name__ == "__main__":
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]
    print("✅ Card 08 (Two Sum) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 49 · Group Anagrams](https://leetcode.com/problems/group-anagrams/) — `https://leetcode.com/problems/group-anagrams/`

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

if __name__ == "__main__":
    res = group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    sorted_res = sorted([sorted(g) for g in res])
    assert sorted_res == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams([""]) == [[""]]
    assert group_anagrams(["a"]) == [["a"]]
    print("✅ Card 09 (Group Anagrams) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 347 · Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) — `https://leetcode.com/problems/top-k-frequent-elements/`

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

if __name__ == "__main__":
    res1 = top_k_frequent([1, 1, 1, 2, 2, 3], 2)
    assert set(res1) == {1, 2}
    assert top_k_frequent([1], 1) == [1]
    print("✅ Card 10 (Top K Frequent Elements) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 271 · Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/) — `https://leetcode.com/problems/encode-and-decode-strings/`

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

if __name__ == "__main__":
    codec = Codec()
    test_cases = [
        ["lint", "code", "love", "you"],
        [""],
        [],
        ["hello#world", "123#456", "##"],
        ["a" * 100, "b" * 50]
    ]
    for tc in test_cases:
        encoded = codec.encode(tc)
        decoded = codec.decode(encoded)
        assert decoded == tc, f"Failed on {tc}: got {decoded}"
    print("✅ Card 11 (Encode and Decode Strings) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 238 · Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) — `https://leetcode.com/problems/product-of-array-except-self/`

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

if __name__ == "__main__":
    assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]
    assert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]
    print("✅ Card 12 (Product of Array Except Self) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 36 · Valid Sudoku](https://leetcode.com/problems/valid-sudoku/) — `https://leetcode.com/problems/valid-sudoku/`

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

if __name__ == "__main__":
    board = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    assert is_valid_sudoku(board) is True
    board[0][0] = "8"
    assert is_valid_sudoku(board) is False
    print("✅ Card 13 (Valid Sudoku) all tests passed!")
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

> 🔗 **LeetCode 链接**：[LeetCode 128 · Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) — `https://leetcode.com/problems/longest-consecutive-sequence/`

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

if __name__ == "__main__":
    assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4
    assert longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) == 9
    assert longest_consecutive([]) == 0
    print("✅ Card 14 (Longest Consecutive Sequence) all tests passed!")
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



### 15. 有序数组中三分频众数的对数探针检索 (Majority Element in Sorted Array via Sublinear Binary Search Probe)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">卡片 15</span>
  <span class="review-card-title">有序数组中三分频众数的对数探针检索 (Majority Element in Sorted Array via Sublinear Binary Search Probe)</span>
  <span class="review-card-tag">有序数组 · 鸽巢原理 · 探针锚定 · 二分左右边界 · O(log N) 亚线性</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 229 · Majority Element II](https://leetcode.com/problems/majority-element-ii/) — `https://leetcode.com/problems/majority-element-ii/`
> - [LeetCode 1150 · Check If a Number Is Majority Element in a Sorted Array](https://leetcode.com/problems/check-if-a-number-is-majority-element-in-a-sorted-array/) — `https://leetcode.com/problems/check-if-a-number-is-majority-element-in-a-sorted-array/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业场景需求</div>

给定一个**已按升序排列**的整数数组 `nums`（长度 $n \ge 3$），找出所有在数组中出现频次严格大于 $\lfloor n / 3 \rfloor$ 次的元素：

```python
def findMajorityElementsSorted(nums: List[int]) -> List[int]: ...
```

**输入输出示例**：
- `nums = [1, 2, 3]` $\implies$ `[]`
- `nums = [1, 1, 2, 3, 4]` $\implies$ `[1]`
- `nums = [1, 1, 2, 4, 4]` $\implies$ `[1, 4]`
- `nums = [1, 2, 3, 4, 5, 6, 7]` $\implies$ `[]`

**核心追问 (Sublinear Follow-up)**：
- 线性扫描基线：单趟遍历计数相同连续区段耗时 $\mathcal{O}(n)$。
- **进阶要求**：充分利用数组**已经有序**的先验数学性质，在**严格低于线性时间（$\mathcal{O}(\log n)$）的亚线性时间复杂度内**求解！

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与对数探针定位机制</div>

#### 1. 鸽巢原理与候选值锁定 (Candidate Reduction)
- 根据鸽巢原理（Pigeonhole Principle）：在一个长度为 $n$ 的数组中，出现次数严格大于 $n / 3$ 的不同元素**最多只能有 2 个**（因为 $3 \times (\lfloor n/3 \rfloor + 1) > n$）。
- 由于数组已经完全有序，任何连续出现次数 $> n / 3$ 的数字，其在数组中跨越的区间长度必然大于 $n / 3$。
- **神圣探针定理**：
  若某数值 $x$ 的频次 $> n/3$，它在有序数组中的连续覆盖区间**必定至少跨过以下两个探针采样点之一**：
  $$idx_1 = \left\lfloor \frac{n}{3} \right\rfloor, \qquad idx_2 = \left\lfloor \frac{2n}{3} \right\rfloor$$
  因此，全数组中唯一的潜在合格候选人，只可能是候选集合 $\{nums[idx_1], nums[idx_2]\}$ 中的元素！候选空间瞬间从 $n$ 种缩小至最多 2 种。

#### 2. 二分查找边界快速验真 (Binary Search Verification)
- 对候选值 $v \in \{nums[idx_1], nums[idx_2]\}$（注意去重）：
  - 使用两次二分查找分别定位 $v$ 在有序数组中的最左端下标 `bisect_left` 与最右端下标 `bisect_right`；
  - $v$ 的精确物理出现次数为：
    $$\operatorname{count}(v) = \operatorname{bisect\_right}(nums, v) - \operatorname{bisect\_left}(nums, v)$$
  - 若 $\operatorname{count}(v) > \lfloor n / 3 \rfloor$，则 $v$ 确凿合法，加入结果集。
- 两次二分查找耗时为 $\mathcal{O}(\log n)$。总运行时间严格为亚线性的 $\mathcal{O}(\log n)$！

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from bisect import bisect_left, bisect_right
from typing import List

class SortedMajoritySolution:

    @staticmethod
    def findMajorityElementsSorted(nums: List[int]) -> List[int]:
        """
        有序数组 > n/3 众数对数查找
        时间复杂度 O(log N)，空间复杂度 O(1)
        """
        n = len(nums)
        if n < 3:
            threshold = n // 3
            return [x for x in set(nums) if nums.count(x) > threshold]

        threshold = n // 3
        # 依据鸽巢原理提取两个关键探针位置的值
        probe_indices = [n // 3, (2 * n) // 3]
        candidates = set(nums[i] for i in probe_indices)

        res = []
        for cand in sorted(list(candidates)):
            # 利用二分查找左右边界精确统计出现次数
            left = bisect_left(nums, cand)
            right = bisect_right(nums, cand)
            freq = right - left
            if freq > threshold:
                res.append(cand)

        return res

if __name__ == "__main__":
    assert SortedMajoritySolution.findMajorityElementsSorted([1, 1, 2, 3, 4]) == [1]
    assert SortedMajoritySolution.findMajorityElementsSorted([1, 1, 2, 4, 4]) == [1, 4]
    assert SortedMajoritySolution.findMajorityElementsSorted([1, 2, 3]) == []
    assert SortedMajoritySolution.findMajorityElementsSorted([1, 2, 3, 4, 5, 6, 7]) == []
    print("✅ Card 15 (Sorted Array Majority Probe) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：提取采样点 $\mathcal{O}(1)$；对至多 2 个候选数执行二分搜索，每次 $\mathcal{O}(\log n)$，整体时间复杂度为严格 $\mathcal{O}(\log n)$。
- **空间复杂度**：存储候选人与输出，额外空间复杂度严格为 $\mathcal{O}(1)$。
- **高频避坑清单**：
  1. **探针命中同一连续段**：若数组前半部分全是同一个数，`idx_1` 和 `idx_2` 探测出的候选值相同。必须对候选集合使用 `set` 去重，防止同一个数被重复二分并输出两次。

</div>

</div>
</details>

---

### 16. 单词反转与空格排版精确保留 (Reverse Words with Exact Spacing Preservation & In-Place Semantics)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">卡片 16</span>
  <span class="review-card-title">单词反转与空格排版精确保留 (Reverse Words with Exact Spacing Preservation & In-Place Semantics)</span>
  <span class="review-card-tag">双指针 · 局部对称翻转 · 空格间距序列精准回填 · 原地 O(1) 空间</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 151 · Reverse Words in a String](https://leetcode.com/problems/reverse-words-in-a-string/) — `https://leetcode.com/problems/reverse-words-in-a-string/`
> - [LeetCode 186 · Reverse Words in a String II](https://leetcode.com/problems/reverse-words-in-a-string-ii/) — `https://leetcode.com/problems/reverse-words-in-a-string-ii/`

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业变体矩阵</div>

给你一个字符串 `s`，颠倒字符串中**单词**的相对顺序。单词是由非空格字符组成的极大连续子串：

```python
def reverseWords(s: str) -> str: ...
```

在系统大厂面试（如 C++ 架构与内核开发组）中，该题常引申出以下三大高频变体：

| 变体编号 | 核心变体名称 | 核心特征 / 变异约束 | 核心算法与数据结构 |
|---|---|---|---|
| **变体 1** | **经典规整翻转 (LC 151)** | 消除多余前后置空格，单词间仅保留单个空格 | 双反转法（全串翻转 + 单词各自翻转）或快慢双指针就地压缩。 |
| **变体 2** | **空格排版精确保留 (Preserve Spacing)** | **严禁消除空格**：单词位置逆转，但单词之间的**原始空格数量及排版间隙必须 100% 精确保留**！ | **词槽与间隙分离提取**：解析出单词序列 `words` 与空格块长度序列 `spaces`，逆序 `words` 后与原 `spaces` 交织重组。 |
| **变体 3** | **严格 O(1) 额外空间原地翻转** | 针对可变字符数组（如 C++ `std::string` 或 `vector<char>`），严禁开辟新数组 | `std::reverse(s.begin(), s.end())`，随后用双指针定位各个单词首尾并就地局部翻转。 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与算法架构深度剖析</div>

#### 1. 变体 2：空格排版精确保留（Preserve Exact Spacing）算法
- 观察输入字符串：其本质是由“空格串”与“非空格单词串”交替构成的拓扑序列。
  例如 `s = "  hello   world  "`：
  - 单词列表：`["hello", "world"]`，逆转后为 `["world", "hello"]`；
  - 空格槽序列：`["  ", "   ", "  "]`（首部 2 个空格，中间 3 个空格，尾部 2 个空格）。
- 关键重组：**空格槽的数量与位置保持绝对静止，只将逆转后的单词依次填入各个非空词槽中**！
- 耗时严格为单趟扫描 $\mathcal{O}(n)$，空间 $\mathcal{O}(n)$。

#### 2. 变体 3：C++ 级原地 O(1) 经典两趟对称翻转原理
1. **全局逆序**：反转整个字符数组。此时所有单词的位置已经完成逆转，但每个单词内部的字母次序也随之逆序了。
2. **单词局部二次翻转**：使用双指针扫描数组，识别出每个由空格隔开的独立单词区间 $[start, end]$，就地反转该单词内部字符。负负得正，单词恢复正序！

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码（含空格保留版）</div>

```python
from typing import List

class ReverseWordsSolution:

    @staticmethod
    def reverseWordsPreserveSpacing(s: str) -> str:
        """
        变体 2: 单词倒序但 100% 精确保留原有空格拓扑与间距
        时间复杂度 O(N)，空间复杂度 O(N)
        """
        words: List[str] = []
        tokens: List[str] = [] # 记录完整的分词流（包含独立的空格块与单词）
        
        i = 0
        n = len(s)
        while i < n:
            if s[i] == ' ':
                j = i
                while j < n and s[j] == ' ':
                    j += 1
                tokens.append(s[i:j])
                i = j
            else:
                j = i
                while j < n and s[j] != ' ':
                    j += 1
                word = s[i:j]
                tokens.append(word)
                words.append(word)
                i = j

        # 将单词列表逆序
        words.reverse()

        # 将逆序后的单词重新回填到原 tokens 对应的非空格插槽中
        word_idx = 0
        result = []
        for token in tokens:
            if token.startswith(' '):
                result.append(token)
            else:
                result.append(words[word_idx])
                word_idx += 1

        return "".join(result)

    @staticmethod
    def reverseWordsInPlace(chars: List[str]) -> None:
        """
        变体 3: 可变字符数组严格 O(1) 空间原地翻转
        """
        def reverse_sub(l: int, r: int):
            while l < r:
                chars[l], chars[r] = chars[r], chars[l]
                l += 1
                r -= 1

        n = len(chars)
        # 1. 全局逆序
        reverse_sub(0, n - 1)

        # 2. 各单词局部二次逆序
        start = 0
        while start < n:
            if chars[start] == ' ':
                start += 1
                continue
            end = start
            while end < n and chars[end] != ' ':
                end += 1
            reverse_sub(start, end - 1)
            start = end

if __name__ == "__main__":
    assert ReverseWordsSolution.reverseWordsPreserveSpacing("  hello   world  ") == "  world   hello  "
    assert ReverseWordsSolution.reverseWordsPreserveSpacing("the sky is blue") == "blue is sky the"
    chars = list("the sky is blue")
    ReverseWordsSolution.reverseWordsInPlace(chars)
    assert "".join(chars) == "blue is sky the"
    print("✅ Card 16 (Reverse Words Exact Spacing) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：单趟线性扫描与局部反转，整体时间复杂度为严格 $\mathcal{O}(N)$。
- **空间复杂度**：空格保留版提取词元为 $\mathcal{O}(N)$；字符数组原地修改为严格 $\mathcal{O}(1)$。
- **高频避坑清单**：
  1. **首尾空格被静默裁剪**：在空格保留变体中，切勿调用 `s.strip()`，否则首尾连续空格信息永久丢失。

</div>

</div>
</details>

---

### 17. 数据流单调降序邻域局部最大值检索 (Local Maximum on a 1-D Stream with Boundary Degradation)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">ARRAY 17</span>
  <span class="review-card-title">数据流单调降序邻域局部最大值检索 (Local Maximum on a 1-D Stream with Boundary Degradation)</span>
  <span class="review-card-tag">局部极值 · 严格单调邻域 · 边界自适应退化 · O(N) 单趟扫描</span>
</summary>
<div class="review-card-content">

> 🔗 **相关 LeetCode**：[LeetCode 162 · Find Peak Element](https://leetcode.com/problems/find-peak-element/) — `https://leetcode.com/problems/find-peak-element/` (工业界手撕题 / 局部极值变体)

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class LocalMaximaStreamSolution:
    @classmethod
    def findLocalMaxima(cls, rawData: List[float], localArea: int) -> List[int]:
        """
        检索一维数据流中所有满足左右 localArea 邻域严格单调递减的局部波峰最大值索引。
        
        形式化约束:
        对于下标 i:
        - 左侧有效邻居长度 L = min(i, localArea)
          必须满足: rawData[i - j + 1] > rawData[i - j], for all j in [1, L]
        - 右侧有效邻居长度 R = min(len(rawData) - 1 - i, localArea)
          必须满足: rawData[i + j - 1] > rawData[i + j], for all j in [1, R]
        - 若某侧可用邻居不足 localArea 个，自适应退化检查全部可用邻居。
        """
        n = len(rawData)
        if n == 0:
            return []

        result = []

        for i in range(n):
            is_peak = True

            # 1. 检验左侧单调性 (从外侧向 i 递增，即从 i 向外递减)
            left_len = min(i, localArea)
            for j in range(1, left_len + 1):
                if rawData[i - j + 1] <= rawData[i - j]:
                    is_peak = False
                    break

            if not is_peak:
                continue

            # 2. 检验右侧单调性 (从 i 向外递减)
            right_len = min(n - 1 - i, localArea)
            for j in range(1, right_len + 1):
                if rawData[i + j - 1] <= rawData[i + j]:
                    is_peak = False
                    break

            if is_peak:
                result.append(i)

        return result

if __name__ == "__main__":
    assert LocalMaximaStreamSolution.findLocalMaxima([1, 3, 5, 4, 2, 6, 2, 1], 2) == [2]
    assert LocalMaximaStreamSolution.findLocalMaxima([10], 3) == [0]
    assert LocalMaximaStreamSolution.findLocalMaxima([2, 4, 4, 1], 1) == []
    print("✅ Card 17 (Local Maximum 1-D Stream) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **双侧单调递减语义**：
  波峰极大值要求从中心向两侧发散时严格递减。即左侧序列 $rawData[i-L \dots i]$ 必须严格递增，右侧序列 $rawData[i \dots i+R]$ 必须严格递减。任何相邻相等数值（平顶 plateau）均无法满足严格单调性，会被立即滤除。
- **边界自适应退化（Boundary Degradation Handling）**：
  若序列首端 $i=0$，左侧有效邻居为 0，左侧条件平凡满足（Vacuously True），只需验证右侧 $\min(n-1, k)$ 个邻居；同理对于末端 $i=n-1$，只需验证左侧邻居。单元素数组直接返回 `[0]`。
- **流式特征工程与时间序列波峰检出**：
  在金融 Tick 数据流与传感器时序中，该算法用于捕捉支撑阻力位与局部极值事件，常作为上层形态学特征构建的算子基石。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N \cdot K)$。对于长度为 $N$ 的数组，每个点最多向两侧延伸检查 $K = 	ext{localArea}$ 步。若 $K \ll N$，整体逼近 $\mathcal{O}(N)$ 线性时间。
- **空间复杂度**：除存储输出索引外，仅需 $\mathcal{O}(1)$ 额外辅助空间。

</div>

</div>
</details>

---

### 18. 子数组极值和极大化与相邻对偶性规约 (Largest Min+Max in Subarray via Adjacent Pair Reduction)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">ARRAY 18</span>
  <span class="review-card-title">子数组极值和极大化与相邻对偶性规约 (Largest Min+Max in Subarray via Adjacent Pair Reduction)</span>
  <span class="review-card-tag">连续子数组 · 极值和最大化 · 相邻对偶性数学规约 · O(N) 线性最优</span>
</summary>
<div class="review-card-content">

> 🔗 **相关 LeetCode**：[LeetCode 53 · Maximum Subarray](https://leetcode.com/problems/maximum-subarray/) — `https://leetcode.com/problems/maximum-subarray/` (量化面试高频数学归约手撕真题)

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class LargestMinMaxSumSolution:
    @classmethod
    def largestMinMaxSum(cls, nums: List[int]) -> int:
        """
        求解长度 >= 2 的连续子数组中，min(sub) + max(sub) 的全局最大值。
        
        数学定理：
        对于任意长度 >= 2 的连续子数组 nums[i..j]，其 min + max 必小于等于
        该子数组内部某个相邻两元素对 nums[k] + nums[k+1] 的和。
        因此全局最优解必然退化为所有相邻两数之和的最大值。
        """
        n = len(nums)
        if n < 2:
            raise ValueError("数组长度必须至少为 2")

        max_sum = nums[0] + nums[1]
        for i in range(1, n - 1):
            pair_sum = nums[i] + nums[i + 1]
            if pair_sum > max_sum:
                max_sum = pair_sum

        return max_sum

if __name__ == "__main__":
    assert LargestMinMaxSumSolution.largestMinMaxSum([5, 12, 9, 6, 4]) == 21
    assert LargestMinMaxSumSolution.largestMinMaxSum([1, 2]) == 3
    assert LargestMinMaxSumSolution.largestMinMaxSum([10, 1, 10]) == 11
    print("✅ Card 18 (Largest Min+Max Subarray) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **数学反证与规约证明（Reduction Proof）**：
  设区间 $[i, j]$（$j - i \ge 1$）中最小值为 $m = \min(nums[i..j])$，最大值为 $M = \max(nums[i..j])$。
  考察区间内任何相邻元素对 $(nums[k], nums[k+1])$：
  1. 显然有 $nums[k] \ge m$ 且 $nums[k+1] \ge m$；
  2. 必存在某个相邻对包含最大值 $M$（设 $nums[p] = M$，则其相邻元素 $nums[p-1]$ 或 $nums[p+1]$ 至少有一个属于该区间）；
  3. 取该包含 $M$ 的相邻对，其较小元素必然 $\ge m$；
  4. 故该相邻对的和 $M + 	ext{other} \ge M + m$ 恒成立！
  5. **结论**：任意长区间的目标值均被其内部包含最大值的相邻对所支配。因此无需使用线段树或滑动窗口，单次 $\mathcal{O}(N)$ 线性遍历相邻元素即获全局最优解。
- **边界防坑**：
  必须在前置沟通中确认“长度 $\ge 2$”这一刚性约束。若允许长度为 1，则单个元素自身作为子数组的 $min + max = 2 	imes nums[i]$，将改变题目本质。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，单趟扫描数组相邻元素。
- **空间复杂度**：$\mathcal{O}(1)$，仅需常数空间维护当前最大相邻和。

</div>

</div>
</details>

---

### 19. 和为 K 的子数组计数与前缀和哈希映射 (Subarray Sum Equals K via Prefix Sum Hash Map)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">ARRAY 19</span>
  <span class="review-card-title">和为 K 的子数组计数与前缀和哈希映射 (Subarray Sum Equals K via Prefix Sum Hash Map)</span>
  <span class="review-card-tag">前缀和差分 · 频次哈希表 · 负数鲁棒性 · O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 560 · Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) — `https://leetcode.com/problems/subarray-sum-equals-k/`

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List
from collections import defaultdict

class SubarraySumEqualsKSolution:
    @classmethod
    def subarraySum(cls, nums: List[int], k: int) -> int:
        """
        统计数组中所有和等于 k 的连续子数组个数。
        支持正数、负数与零。
        """
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # 初始基准：前缀和恰好为 k 时，差值 0 贡献 1 次匹配
        
        current_sum = 0
        total_valid_subarrays = 0

        for num in nums:
            current_sum += num
            
            # 查找以当前元素结尾且和为 k 的子数组个数
            # sum(nums[i..j]) = current_sum - prefix_sum = k => prefix_sum = current_sum - k
            if (current_sum - k) in prefix_counts:
                total_valid_subarrays += prefix_counts[current_sum - k]

            # 将当前前缀和注册进频次表
            prefix_counts[current_sum] += 1

        return total_valid_subarrays

if __name__ == "__main__":
    assert SubarraySumEqualsKSolution.subarraySum([1, 1, 1], 2) == 2
    assert SubarraySumEqualsKSolution.subarraySum([1, 2, 3], 3) == 2
    assert SubarraySumEqualsKSolution.subarraySum([1, -1, 0], 0) == 3
    print("✅ Card 19 (Subarray Sum Equals K) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **前缀和差分原语**：
  区间 $[i, j]$ 的元素之和可表示为两个前缀和之差：
  $$\sum_{p=i}^j nums[p] = S_j - S_{i-1} = k \iff S_{i-1} = S_j - k$$
  因此，只需维护一个记录历史前缀和出现次数的哈希表。每推进一位 $j$，查询历史中 $S_j - k$ 的频次即为以 $j$ 结尾且和为 $k$ 的合法子数组数量。
- **为何双指针/滑动窗口失效**：
  若数组中包含负数，前缀和序列 $S$ 失去了单调递增性。滑动窗口收缩左边界无法保证区间和单调减小，因此双指针算法彻底失效，哈希映射是 $\mathcal{O}(N)$ 的唯一解法。
- **基准哨兵 `{0: 1}` 的必要性**：
  若某前缀和 $S_j$ 本身恰好等于 $k$，则从第 0 个元素到第 $j$ 个元素构成的完整前缀本身就是一个合法子数组。初始化 `prefix_counts[0] = 1` 确保了此类子数组被正确计数。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，数组单趟遍历，哈希表平均 $\mathcal{O}(1)$ 存取。
- **空间复杂度**：$\mathcal{O}(N)$，哈希表最多存储 $N+1$ 个不同的前缀和数值。

</div>

</div>
</details>

---

### 20. 无重复字符的最长子串与最新索引滑动窗口 (Longest Substring Without Repeating Characters)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STRING 20</span>
  <span class="review-card-title">无重复字符的最长子串与最新索引滑动窗口 (Longest Substring Without Repeating Characters)</span>
  <span class="review-card-tag">滑动窗口 · 字符最新下标表 · 左边界单调跳跃 · O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 3 · Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) — `https://leetcode.com/problems/longest-substring-without-repeating-characters/`

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
class LongestSubstringWithoutRepeatingSolution:
    @classmethod
    def lengthOfLongestSubstring(cls, s: str) -> int:
        """
        计算无重复字符的最长连续子串长度。
        使用字符最后出现索引表，实现左边界 O(1) 单调跳跃。
        """
        char_last_seen = {}
        left = 0
        max_length = 0

        for right, ch in enumerate(s):
            # 若字符重复且上次出现位置在当前窗口内部，直接跳跃左边界至上次位置的右侧一位
            if ch in char_last_seen and char_last_seen[ch] >= left:
                left = char_last_seen[ch] + 1
            
            char_last_seen[ch] = right
            current_window = right - left + 1
            if current_window > max_length:
                max_length = current_window

        return max_length

if __name__ == "__main__":
    assert LongestSubstringWithoutRepeatingSolution.lengthOfLongestSubstring("abcabcbb") == 3
    assert LongestSubstringWithoutRepeatingSolution.lengthOfLongestSubstring("bbbbb") == 1
    assert LongestSubstringWithoutRepeatingSolution.lengthOfLongestSubstring("pwwkew") == 3
    assert LongestSubstringWithoutRepeatingSolution.lengthOfLongestSubstring("") == 0
    print("✅ Card 20 (Longest Substring Without Repeating) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **左边界跨步跳跃（O(1) Jump vs 步进收缩）**：
  若采用基础滑动窗口（维护字符集合），当出现重复字符时，左指针必须单步递增循环删除字符，最坏情况下每个字符出入窗口各一次（总计 $2N$ 步）；
  记录 `char_last_seen[ch]` 后，一旦检测到重复字符，左边界可直接置为 $\max(left, char\_last\_seen[ch] + 1)$，跳过内部冗余收缩。
- **单调性卫语句**：
  必须加入 `char_last_seen[ch] >= left` 判定。因为哈希表中可能记录了窗口左边界之前的陈旧历史索引，左边界绝不可逆流倒退。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，右指针单向推进，左指针单调前进。
- **空间复杂度**：$\mathcal{O}(\min(N, |\Sigma|))$，哈希表大小取决于字符集大小（ASCII 为 128，Unicode 视字符种类而定）。

</div>

</div>
</details>

---

### 21. 8 字节对齐内存分配器仿真 (8-Byte Aligned Memory Allocator Simulation)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">DESIGN 21</span>
  <span class="review-card-title">8 字节对齐内存分配器仿真 (8-Byte Aligned Memory Allocator Simulation)</span>
  <span class="review-card-tag">底层仿真 · 8 字节对齐步进 · 唯一 ID 标记 · O(N / 8 * X)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 2502 · Design Memory Allocator](https://leetcode.com/problems/design-memory-allocator/) — `https://leetcode.com/problems/design-memory-allocator/`

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class AlignedMemoryAllocator:
    """
    支持 8 字节对齐首地址约束的内存分配与按 ID 释放模拟器。
    
    规则:
    - alloc(x): 寻找起始下标为 8 的倍数的最左侧连续 x 个空闲单元 (0)，
                使用自增唯一整数 ID 标记并返回起始下标；无法容纳则返回 -1。
    - erase(id): 释放所有标记为 id 的内存单元，返回清空的单元总数。
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        # 内存单元状态：0 代表空闲，正整数表示分配该单元的块 ID
        self.memory: List[int] = [0] * capacity
        self.next_alloc_id: int = 1

    def alloc(self, x: int) -> int:
        if x <= 0:
            return -1

        current_id = self.next_alloc_id

        # 仅遍历 8 字节对齐的起始候选下标: 0, 8, 16, 24, ...
        for start in range(0, self.capacity, 8):
            if start + x <= self.capacity:
                # 检查连续 x 个单元是否全为 0 (空闲)
                can_fit = True
                for offset in range(x):
                    if self.memory[start + offset] != 0:
                        can_fit = False
                        break

                if can_fit:
                    # 占用内存并打上块 ID 标签
                    for offset in range(x):
                        self.memory[start + offset] = current_id
                    self.next_alloc_id += 1
                    return start

        return -1

    def erase(self, req_id: int) -> int:
        if req_id <= 0:
            return 0

        cleared_count = 0
        for i in range(self.capacity):
            if self.memory[i] == req_id:
                self.memory[i] = 0
                cleared_count += 1

        return cleared_count

if __name__ == "__main__":
    alloc = AlignedMemoryAllocator(24)
    assert alloc.alloc(5) == 0
    assert alloc.alloc(10) == 8
    assert alloc.alloc(5) == -1
    assert alloc.erase(1) == 5
    assert alloc.alloc(6) == 0
    print("✅ Card 21 (8-Byte Aligned Memory Allocator) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **硬件 8 字节对齐约束（Alignment Invariant）**：
  在现代 CPU 架构中，未对齐内存访问（Unaligned Memory Access）会触发额外的总线周期甚至硬件异常。本题强制起始下标必须满足 $start \pmod 8 == 0$，因此外层循环以步长 8 跨步推进，候选点数量缩减至 $\lceil 	ext{capacity} / 8 
ceil$。
- **自动增量分配 ID 与安全擦除**：
  `next_alloc_id` 保证即使连续分配释放后，每一个历史分配块的 ID 绝对唯一，避免因 ID 复用导致释放已销毁块时发生悬垂指针误删。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：
  - `alloc(x)`：最多检查 $\lceil N / 8 
ceil$ 个候选槽位，单次校验 $X$ 步，最坏时间复杂度为 $\mathcal{O}(rac{N}{8} \cdot X)$。
  - `erase(id)`：单趟线性扫描整块内存，时间复杂度为严格 $\mathcal{O}(N)$。
- **空间复杂度**：$\mathcal{O}(N)$，用于维护整块内存状态数组。

</div>

</div>
</details>

---

### 22. 奇偶交替连续子数组极速计数 (Zigzag Alternating-Parity Subarrays)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">ARRAY 22</span>
  <span class="review-card-title">奇偶交替连续子数组极速计数 (Zigzag Alternating-Parity Subarrays)</span>
  <span class="review-card-tag">动态连击增量 · 奇偶模数检验 · 单调推进 · O(N) 时间</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：
> - [LeetCode 2765 · Longest Alternating Subarray](https://leetcode.com/problems/longest-alternating-subarray/) — `https://leetcode.com/problems/longest-alternating-subarray/`
> - [LeetCode 978 · Longest Turbulent Subarray](https://leetcode.com/problems/longest-turbulent-subarray/) — `https://leetcode.com/problems/longest-turbulent-subarray/`

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class AlternatingParitySubarraysSolution:
    @classmethod
    def countAlternatingSubarrays(cls, nums: List[int]) -> int:
        """
        统计数组中所有相邻元素奇偶性互不相同的连续子数组数量。
        单个元素自身视为长度为 1 的合法交替子数组。
        """
        if not nums:
            return 0

        total_subarrays = 1
        current_streak = 1  # 记录以当前元素结尾的奇偶交替最大连续长度

        for i in range(1, len(nums)):
            # 判断与前驱元素的奇偶性是否异号: (nums[i] % 2) != (nums[i-1] % 2)
            if (nums[i] % 2) != (nums[i - 1] % 2):
                current_streak += 1
            else:
                # 奇偶性相同，交替链断裂，当前元素单独作为长度为 1 的交替起点
                current_streak = 1

            # 核心增量：以 nums[i] 结尾的交替子数组个数恰好等于 current_streak
            total_subarrays += current_streak

        return total_subarrays

if __name__ == "__main__":
    assert AlternatingParitySubarraysSolution.countAlternatingSubarrays([1, 2, 3, 4]) == 10
    assert AlternatingParitySubarraysSolution.countAlternatingSubarrays([2, 4, 6]) == 3
    assert AlternatingParitySubarraysSolution.countAlternatingSubarrays([1]) == 1
    print("✅ Card 22 (Zigzag Alternating-Parity Subarrays) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **动态连击增量原语（Running Streak Counting）**：
  若已知以 $nums[i-1]$ 结尾的最长交替连续子数组长度为 $k$，且 $nums[i]$ 与 $nums[i-1]$ 奇偶性不同，则以 $nums[i]$ 结尾的所有交替子数组，正是将前驱的这 $k$ 个子数组全部追加 $nums[i]$（长度从 $2$ 到 $k+1$），外加 $nums[i]$ 单独组成的长度 1 子数组，总数恰为 $k + 1$ 个！
- **杜绝 $\mathcal{O}(N^2)$ 双重遍历**：
  初学者常习惯枚举左右端点 $[i, j]$ 并遍历检验，导致在大规模数据评测中发生超时。通过维护前缀连击长度 `current_streak`，每次仅需累加当前值，单趟扫描即刻出解。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N)$，单次线性扫描。
- **空间复杂度**：$\mathcal{O}(1)$，仅需常数级别的连击累加器。

</div>

</div>
</details>

---

### 23. 双向对齐报纸排版与星号边框渲染 (Two-Direction Justified Newspaper Layout)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STRING 23</span>
  <span class="review-card-title">双向对齐报纸排版与星号边框渲染 (Two-Direction Justified Newspaper Layout)</span>
  <span class="review-card-tag">贪心单词装箱 · 左右动态对齐补齐 · 物理星号边框包裹 · O(Total Words)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 68 · Text Justification](https://leetcode.com/problems/text-justification/) — `https://leetcode.com/problems/text-justification/`

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class NewspaperLayoutSolution:
    @classmethod
    def layoutNewspaper(
        cls,
        paragraphs: List[List[str]],
        alignments: List[str],
        width: int
    ) -> List[str]:
        """
        根据各段落指定的左对齐 (LEFT) 或右对齐 (RIGHT) 标志，以最大行宽 width 贪心排版单词。
        行内单词间用单空格隔开；短行在对应方向补齐空格；整体用 '*' 边框装裱输出。
        """
        content_lines: List[str] = []

        for words, align in zip(paragraphs, alignments):
            current_line_words: List[str] = []
            current_line_len = 0

            for word in words:
                # 计算若加入该单词所需的总长度（非行首单词需追加 1 个间隔空格）
                needed_len = len(word) if not current_line_words else len(word) + 1

                if current_line_len + needed_len <= width:
                    current_line_words.append(word)
                    current_line_len += needed_len
                else:
                    # 缓冲区满，将当前行根据对齐规则输出
                    line_text = " ".join(current_line_words)
                    pad_spaces = " " * (width - len(line_text))
                    
                    if align == "LEFT":
                        formatted_line = line_text + pad_spaces
                    else:  # RIGHT
                        formatted_line = pad_spaces + line_text

                    content_lines.append(f"*{formatted_line}*")
                    # 新行以当前溢出单词起步
                    current_line_words = [word]
                    current_line_len = len(word)

            # 输出段落尾行
            if current_line_words:
                line_text = " ".join(current_line_words)
                pad_spaces = " " * (width - len(line_text))
                if align == "LEFT":
                    formatted_line = line_text + pad_spaces
                else:
                    formatted_line = pad_spaces + line_text
                content_lines.append(f"*{formatted_line}*")

        # 构造顶部与底部星号边框 (边框宽度为 width + 2)
        horizontal_border = "*" * (width + 2)
        return [horizontal_border] + content_lines + [horizontal_border]

if __name__ == "__main__":
    paras = [["Hello", "world"], ["Antigravity", "AI", "news"]]
    aligns = ["LEFT", "RIGHT"]
    rendered = NewspaperLayoutSolution.layoutNewspaper(paras, aligns, 16)
    assert rendered[0] == "******************"
    assert rendered[1] == "*Hello world     *"
    assert rendered[2] == "*  Antigravity AI*"
    assert rendered[3] == "*            news*"
    assert rendered[4] == "******************"
    print("✅ Card 23 (Newspaper Layout) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **贪心贪婪装箱（Greedy Word Packing）**：
  同一段落内单词顺序不可颠倒。每行塞入尽可能多的单词，且行内单词间保持且仅保持一个空格间隔。当且仅当追加新单词后总长超过 `width` 时，触发换行刷盘。
- **动态左右填补（Padding Logic）**：
  - `LEFT` 对齐：文字靠左，剩余空格全部填充在右侧；
  - `RIGHT` 对齐：文字靠右，剩余空格全部填充在左侧。
- **物理边框封闭（Border Framing）**：
  每行内容两侧各贴附一个 `*`，首尾单独追加长度为 `width + 2` 的纯星号行，确保渲染出的字符矩阵绝对平整矩形化。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(L)$，其中 $L$ 为所有段落单词字符与空格的总长度。
- **空间复杂度**：$\mathcal{O}(L)$，存储格式化渲染输出列表。

</div>

</div>
</details>

---

### 24. 最长回文子串与马拉车算法 (Longest Palindromic Substring: Center vs Manacher)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">STRING 24</span>
  <span class="review-card-title">最长回文子串与马拉车算法 (Longest Palindromic Substring: Center vs Manacher)</span>
  <span class="review-card-tag">中心扩散法 · 马拉车 (Manacher) · 回文半径对称映射 · 严格 O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode 链接**：[LeetCode 5 · Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) — `https://leetcode.com/problems/longest-palindromic-substring/`

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
class LongestPalindromeSolution:
    @classmethod
    def longestPalindromeCenterExpand(cls, s: str) -> str:
        """
        解法一：经典中心扩散法
        时间复杂度 O(N^2)，额外空间 O(1)。
        """
        if not s:
            return ""

        start, max_len = 0, 1

        def expand_around_center(left: int, right: int) -> int:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            return right - left - 1

        for i in range(len(s)):
            len1 = expand_around_center(i, i)       # 奇数长度中心
            len2 = expand_around_center(i, i + 1)   # 偶数长度中心
            cur_max = max(len1, len2)
            if cur_max > max_len:
                max_len = cur_max
                start = i - (cur_max - 1) // 2

        return s[start : start + max_len]

    @classmethod
    def longestPalindromeManacher(cls, s: str) -> str:
        """
        解法二：工业级 Manacher 算法（马拉车）
        利用回文对称性与最右边界缓存，时间复杂度严格 O(N)。
        """
        if not s:
            return ""

        # 1. 插入间隔符统一奇偶回文，前后加哨兵杜绝越界检查: "^#a#b#a#$"
        transformed = "^#" + "#".join(s) + "#$"
        m = len(transformed)
        radius = [0] * m  # radius[i] 记录以 i 为中心的最长回文半径
        center = 0
        right = 0

        # 2. 线性推导回文半径
        for i in range(1, m - 1):
            i_mirror = 2 * center - i  # i 关于当前最右边界中心 center 的对称点

            if right > i:
                # 对称加速：初值直接继承对称点的半径，但不能突破已知最右边界
                radius[i] = min(right - i, radius[i_mirror])
            else:
                radius[i] = 0

            # 3. 朴素扩散扩展（仅在突破边界时有效推进）
            while transformed[i + 1 + radius[i]] == transformed[i - 1 - radius[i]]:
                radius[i] += 1

            # 4. 若新回文右翼超越了历史最右边界，更新中心与边界
            if i + radius[i] > right:
                center = i
                right = i + radius[i]

        # 5. 定位最大回文半径与其在原字符串中的起始位置
        best_radius = 0
        best_center = 0
        for i in range(1, m - 1):
            if radius[i] > best_radius:
                best_radius = radius[i]
                best_center = i

        # 关键原串坐标映射: (best_center - best_radius) // 2
        start_orig = (best_center - best_radius) // 2
        return s[start_orig : start_orig + best_radius]

if __name__ == "__main__":
    assert LongestPalindromeSolution.longestPalindromeCenterExpand("babad") in ("bab", "aba")
    assert LongestPalindromeSolution.longestPalindromeCenterExpand("cbbd") == "bb"
    assert LongestPalindromeSolution.longestPalindromeManacher("babad") in ("bab", "aba")
    assert LongestPalindromeSolution.longestPalindromeManacher("cbbd") == "bb"
    assert LongestPalindromeSolution.longestPalindromeManacher("a") == "a"
    print("✅ Card 24 (Longest Palindrome Manacher) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **中心扩散法（$\mathcal{O}(N^2)$ 基准）**：
  每个字符（奇回文，共 $N$ 个）或相邻两字符间隙（偶回文，共 $N-1$ 个）作为扩散核，向双侧线性比对，单次最长扩散耗时 $\mathcal{O}(N)$，最坏情况（如全同一字符 `"aaaaa"`）退化至 $\mathcal{O}(N^2)$。
- **Manacher 算法的 $\mathcal{O}(N)$ 飞跃机理**：
  1. **奇偶同构化**：插入 `#` 后，无论原回文是奇是偶，在变换串中统统归一为**以某个字符或 `#` 为中心的奇数长度回文**；
  2. **对称点映射借力（Mirror Reflection）**：当前点 $i$ 位于已知覆盖范围 $[center - R, right]$ 内部时，由于以 $center$ 为中心的大回文区间是对称的，$i$ 处的回文结构在前半区 $i_{mirror} = 2 \cdot center - i$ 处**早已被完全计算过**！因此 $radius[i]$ 可以直接继承 $\min(right - i, radius[i_{mirror}])$；
  3. **单调前进摊还分析**：由于每一步只有在字符比对成功且拓展出新的 $right$ 边界时才会增加常数操作，$right$ 边界只能单调向右移动至多 $2N$ 次，因此总比对次数被严格限定为 $\mathcal{O}(N)$。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：中心扩散法为 $\mathcal{O}(N^2)$；Manacher 算法为严格 $\mathcal{O}(N)$。
- **空间复杂度**：中心扩散法为 $\mathcal{O}(1)$；Manacher 算法为 $\mathcal{O}(N)$（变换字符串与半径数组）。

</div>

</div>
</details>


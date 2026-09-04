# 复习卡片：常考基础题 (Review Flashcards · Core Fundamentals)

本篇为算法面试复习卡片集第一部分：**常考基础题**。  
每个卡片为一个独立可折叠的复习单元，包含**题目定义**、**核心思路**、**Python 标准实现**、**复杂度速记**以及一个**自测选择题**（点击选项前答案完全保密，点击后即时反馈）。

---

## 🛠️ 可复用复习卡片 Block 结构说明

后续新增卡片可直接复制以下 HTML 模板：

```html
<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">题号 / 模块</span>
  <span class="review-card-title">算法 / 题目名称</span>
  <span class="review-card-tag">分类标签</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>
<div class="review-block-body">题目简述与核心约束...</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>
<div class="review-block-body">心智模型、分步解法...</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 核心 Python 代码</div>
<div class="review-block-body">

```python
# 标准 Python 代码
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与关键性质</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">最好时间</span><span class="complexity-item-value">O(...)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">最坏时间</span><span class="complexity-item-value">O(...)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">辅助空间</span><span class="complexity-item-value">O(...)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">稳定性</span><span class="complexity-item-value">稳定 / 不稳定</span></div>
</div>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看答案与解析)</div>
<div class="review-block-body">

```quiz
title: 自测 · 知识点自测
question: 问题题目描述？
A. 选项 1
B. 选项 2
C. 选项 3
D. 选项 4
答案: A
解析: 为什么选 A 的深度解析...
```

</div>
</div>

</div>
</details>
```

---

## 📚 常考基础题复习卡片

### 1. 归并排序 (Merge Sort)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">基础 01</span>
  <span class="review-card-title">归并排序 (Merge Sort)</span>
  <span class="review-card-tag">分治 · 递归 · 稳定排序</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>
<div class="review-block-body">
给定一个长度为 $n$ 的无序整数数组，将其按非递减顺序排序。要求在最坏情况下时间复杂度仍严格保证为 $O(n \log n)$，且维持相同元素的相对顺序（即具备稳定性）。
</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>
<div class="review-block-body">
归并排序是经典的**分治（Divide and Conquer）**范式，三步流程：
<ol>
  <li><strong>分解 (Divide)</strong>：计算区间中点 <code>mid = len(nums) // 2</code>，将原数组均分为左半段和右半段。</li>
  <li><strong>解决 (Conquer)</strong>：递归对左半段和右半段分别调用归并排序，直到子数组长度为 1（天然有序）。</li>
  <li><strong>合并 (Combine)</strong>：使用双指针遍历两个已排序的子数组，比较当前元素大小，将较小者依次放入新数组；若元素大小相同，优先选取左侧子数组元素以保证稳定性。</li>
</ol>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 核心 Python 代码</div>
<div class="review-block-body">

```python
class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        if len(nums) <= 1:
            return nums

        mid = len(nums) // 2
        left = self.sortArray(nums[:mid])
        right = self.sortArray(nums[mid:])

        return self._merge(left, right)

    def _merge(self, left: list[int], right: list[int]) -> list[int]:
        res = []
        i = j = 0

        # 双指针合并两个有序子数组
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:  # 包含等号保证稳定排序
                res.append(left[i])
                i += 1
            else:
                res.append(right[j])
                j += 1

        # 追加剩余未合并元素
        res.extend(left[i:])
        res.extend(right[j:])
        return res
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与关键性质</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">最好时间</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">最坏时间</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">平均时间</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">辅助空间</span><span class="complexity-item-value">O(n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">稳定性</span><span class="complexity-item-value">稳定 (Stable)</span></div>
</div>
<p style="margin-top: 0.5rem; font-size: 0.88rem; color: #466370;">
<strong>要点解析</strong>：递归树高度严格为 $\lceil \log_2 n \rceil$，每层所有合并操作的总时间固定为 $O(n)$，故时间始终为 $O(n \log n)$。合并时需要 $O(n)$ 缓冲区存储中间结果，递归栈深度为 $O(\log n)$。
</p>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看答案与解析)</div>
<div class="review-block-body">

```quiz
title: 自测题 · 归并排序空间开销
question: 在对一个含有 n 个节点的单链表（Singly Linked List）执行归并排序时，采用自底向上（Bottom-Up）迭代策略的最优辅助空间复杂度（不计常数额外指针）是多少？
A. O(1)
B. O(log n)
C. O(n)
D. O(n log n)
答案: A
解析: 数组归并排序需要 O(n) 额外数组存储合并中间值；但单链表只需要修改节点的 next 指针指向即可完成就地合并。结合自底向上的步长迭代（1, 2, 4, 8...），既不需要递归栈（避免 O(log n) 栈空间），也不需要任何临时数组，因此单链表归并排序的最优空间复杂度是严格的 O(1)。
```

</div>
</div>

</div>
</details>

---

### 2. 快速排序 (Quick Sort)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">基础 02</span>
  <span class="review-card-title">快速排序 (Quick Sort)</span>
  <span class="review-card-tag">分治 · 原地划分 · 不稳定排序</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>
<div class="review-block-body">
给定一个长度为 $n$ 的整数数组，对其进行就地升序排序（In-place Sort）。要求平均时间复杂度达到 $O(n \log n)$，且辅助空间尽可能小（除递归调用栈外不使用额外数据结构）。
</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>
<div class="review-block-body">
快速排序的核心在于**划分（Partitioning）先行**：
<ol>
  <li><strong>选主元 (Pivot Selection)</strong>：随机挑选一个元素作为主元（或三数取中），防止在近乎有序的输入下退化为 $O(n^2)$。</li>
  <li><strong>划分 (Partition)</strong>：调整数组，使所有 $\le pivot$ 的元素位于主元左侧，所有 $\ge pivot$ 的元素位于主元右侧，最终将 pivot 安放到其完全排好序的确定位置 <code>p_idx</code>。</li>
  <li><strong>递归处理 (Recurse)</strong>：对左子区间 <code>[l, p_idx - 1]</code> 与右子区间 <code>[p_idx + 1, r]</code> 分别递归排序。</li>
</ol>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 核心 Python 代码</div>
<div class="review-block-body">

```python
import random

class Solution:
    def sortArray(self, nums: list[int]) -> list[int]:
        self._quick_sort(nums, 0, len(nums) - 1)
        return nums

    def _quick_sort(self, nums: list[int], l: int, r: int) -> None:
        if l >= r:
            return

        # 随机挑选主元并交换到末尾，破坏对抗性输入
        pivot_idx = random.randint(l, r)
        nums[pivot_idx], nums[r] = nums[r], nums[pivot_idx]

        # 双指针对撞划分（Hoare / 对撞法）
        p_idx = self._partition(nums, l, r)
        self._quick_sort(nums, l, p_idx - 1)
        self._quick_sort(nums, p_idx + 1, r)

    def _partition(self, nums: list[int], l: int, r: int) -> int:
        pivot = nums[r]
        i = l
        # Lomuto 划分：i 维护小于等于 pivot 的右边界
        for j in range(l, r):
            if nums[j] <= pivot:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
        nums[i], nums[r] = nums[r], nums[i]
        return i
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与关键性质</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">最好时间</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">最坏时间</span><span class="complexity-item-value">O(n²)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">平均时间</span><span class="complexity-item-value">O(n log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">辅助空间</span><span class="complexity-item-value">O(log n) ~ O(n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">稳定性</span><span class="complexity-item-value">不稳定 (Unstable)</span></div>
</div>
<p style="margin-top: 0.5rem; font-size: 0.88rem; color: #466370;">
<strong>要点解析</strong>：若每次 pivot 划分极不平衡（如退化为 1 和 $n-1$ 两个子区间），递归树深度为 $n$，时间退化为 $O(n^2)$。空间仅消耗递归调用栈，平均为 $O(\log n)$，最坏退化为 $O(n)$。远距离交换可能破坏相同元素的先后次序，故不稳定。
</p>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看答案与解析)</div>
<div class="review-block-body">

```quiz
title: 自测题 · 快速排序重复元素退化
question: 当输入数组中所有元素完全相同（例如包含 10000 个数值为 7 的数组）时，经典单向 Lomuto 划分的快速排序会发生什么？
A. 时间复杂度严重退化至 O(n^2)
B. 时间复杂度保持最优的 O(n log n)
C. 时间复杂度加速至 O(n)
D. 发生数组越界错误
答案: A
解析: 在单向 Lomuto 划分中，条件 nums[j] <= pivot 对于所有元素全部成立，所有的元素都会被连续交换并划入同一侧，导致划分点每次只前进 1 位，递归深度达到 n，时间退化至 O(n^2)。针对此缺陷的标准解法是使用「三路划分（Dutch National Flag 荷兰国旗三向切分）」，将与 pivot 相等的元素整体归入中间并排除出后续递归，此时处理全相同数组仅需 O(n) 时间。
```

</div>
</div>

</div>
</details>

---

### 3. 动态数组实现 (Dynamic Array / Vector)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">基础 03</span>
  <span class="review-card-title">动态数组实现 (Dynamic Array)</span>
  <span class="review-card-tag">数据结构设计 · 均摊分析 · 内存搬迁</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>
<div class="review-block-body">
基于固定大小的连续内存块（固定长度底层数组）从零实现可自动扩容的动态数组（类似 Python 的 <code>list</code> 或 C++ 的 <code>std::vector</code>）。支持以下接口：
<ul>
  <li><code>get(i)</code>：查询索引 $i$ 处的值，索引越界抛出异常。</li>
  <li><code>set(i, val)</code>：修改索引 $i$ 处的值。</li>
  <li><code>push_back(val)</code>：在尾部追加元素；当实际元素个数达到当前物理容量时自动触发倍增扩容。</li>
  <li><code>pop_back()</code>：弹出并返回末尾元素。</li>
  <li><code>resize()</code>：将物理容量扩大为原先的 2 倍并完整迁移原有元素。</li>
</ul>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>
<div class="review-block-body">
动态数组的核心心智模型：
<ol>
  <li><strong>连续内存与双指针计数</strong>：底层分配固定容量的连续空间 <code>capacity</code>，用 <code>size</code> 记录当前有效元素个数。</li>
  <li><strong>几何倍增扩容 (Geometric Doubling)</strong>：当 <code>size == capacity</code> 时触发扩容。新容量设为 <code>max(1, 2 * capacity)</code>，分配新连续内存块，将所有元素依次拷贝过去，再释放原空间。</li>
  <li><strong>均摊常数级（Amortized $O(1)$）证明</strong>：扩容单次虽然需要 $O(n)$ 复制，但发生频率呈指数级衰减。从容量 1 扩至 $n$，总复制次数为 $1 + 2 + 4 + \dots + n \le 2n$ 次，分摊到 $n$ 次插入操作上，单次 <code>push_back</code> 的均摊成本严格为 $O(1)$。</li>
</ol>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 核心 Python 代码</div>
<div class="review-block-body">

```python
class DynamicArray:
    def __init__(self, capacity: int = 2):
        self.capacity = max(1, capacity)
        self.size = 0
        self.array = [None] * self.capacity

    def get(self, i: int) -> int:
        if not (0 <= i < self.size):
            raise IndexError("Index out of bounds")
        return self.array[i]

    def set(self, i: int, val: int) -> None:
        if not (0 <= i < self.size):
            raise IndexError("Index out of bounds")
        self.array[i] = val

    def push_back(self, val: int) -> None:
        # 当容量耗尽时触发几何倍增
        if self.size == self.capacity:
            self._resize(self.capacity * 2)

        self.array[self.size] = val
        self.size += 1

    def pop_back(self) -> int:
        if self.size == 0:
            raise IndexError("Cannot pop from empty array")
        self.size -= 1
        val = self.array[self.size]
        self.array[self.size] = None  # 防止内存泄漏
        return val

    def _resize(self, new_capacity: int) -> None:
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity

    def get_size(self) -> int:
        return self.size

    def get_capacity(self) -> int:
        return self.capacity
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与关键性质</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">随机访问 get/set</span><span class="complexity-item-value">O(1)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">尾部追加 push_back</span><span class="complexity-item-value">均摊 O(1) (最坏 O(n))</span></div>
  <div class="complexity-item"><span class="complexity-item-title">尾部弹出 pop_back</span><span class="complexity-item-value">O(1)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">中间插入/删除</span><span class="complexity-item-value">O(n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">空间利用率</span><span class="complexity-item-value">≥ 50%</span></div>
</div>
<p style="margin-top: 0.5rem; font-size: 0.88rem; color: #466370;">
<strong>要点解析</strong>：随机访问通过物理地址公式 <code>base + i * element_size</code> 直接定位，耗时 $O(1)$。追加操作只在触发扩容的一瞬间耗时 $O(n)$，但其势能均摊后为常数级。中间插入或删除需要逐一平移元素，耗时 $O(n)$。
</p>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看答案与解析)</div>
<div class="review-block-body">

```quiz
title: 自测题 · 动态数组扩容策略
question: 如果动态数组的扩容策略改为固定步长增加（例如每次容量满时仅执行 capacity += 1000），连续执行 N 次 push_back 的总数据搬迁时间与单次均摊复杂度分别是多少？
A. 总时间 O(N^2)，均摊 O(N)
B. 总时间 O(N log N)，均摊 O(log N)
C. 总时间 O(N)，均摊 O(1)
D. 总时间 O(N)，均摊 O(N)
答案: A
解析: 若设固定步长为 C = 1000，插入 N 个元素将触发 N/C 次扩容。每次扩容复制的元素个数分别为 C, 2C, 3C, ..., N，其总复制步数为 C * (1 + 2 + ... + N/C) ≈ C * (N/C)^2 / 2 = O(N^2 / C) = O(N^2)。因此分摊到 N 次插入操作上，每次追加的均摊复杂度会退化为可怕的 O(N)。这也是工业级容器必须采用几何倍增（乘法扩容）而非线性增加的根本原因。
```

</div>
</div>

</div>
</details>

---

### 4. 二分查找边界模板 (Binary Search: Left/Right Bound)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">基础 04</span>
  <span class="review-card-title">二分查找边界模板 (Binary Search Bounds)</span>
  <span class="review-card-tag">有序检索 · 开闭区间不变量 · 边界收敛</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与要求</div>
<div class="review-block-body">
在单调非递减的有序整数数组中，查找目标值 <code>target</code> 出现的**首个位置（最左边界 / Lower Bound）**。若目标值不存在，返回其应插入的索引位置以保持数组有序。要求时间复杂度达到 $O(\log n)$，且绝对杜绝死循环。
</div>
</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心算法</div>
<div class="review-block-body">
二分查找的核心是**严格维护区间不变量（Loop Invariant）**：
<ol>
  <li><strong>区间定义</strong>：采用标准的<strong>左闭右闭区间 <code>[l, r]</code></strong>，初始时 <code>l = 0, r = len(nums) - 1</code>。</li>
  <li><strong>防溢出中点</strong>：使用 <code>mid = l + (r - l) // 2</code> 避免大整数相加溢出，在 Python 中可防止无谓的大数运算。</li>
  <li><strong>收缩决策（找最左目标）</strong>：
    <ul>
      <li>若 <code>nums[mid] >= target</code>：目标可能就是 <code>mid</code> 或在 <code>mid</code> 左侧，为了继续向左探查，收缩右界 <code>r = mid - 1</code>。</li>
      <li>若 <code>nums[mid] < target</code>：目标必然在 <code>mid</code> 右侧，收缩左界 <code>l = mid + 1</code>。</li>
    </ul>
  </li>
  <li><strong>退出收敛</strong>：循环条件为 <code>l <= r</code>，终止时必定满足 <code>l = r + 1</code>，最终指针 <code>l</code> 恰好收敛在第一个满足 $\ge target$ 的索引位置。</li>
</ol>
</div>
</div>

<div class="review-block">
<div class="review-block-label">💻 核心 Python 代码</div>
<div class="review-block-body">

```python
class Solution:
    def searchLowerBound(self, nums: list[int], target: int) -> int:
        l = 0
        r = len(nums) - 1

        # 维护左闭右闭区间 [l, r]
        while l <= r:
            mid = l + (r - l) // 2
            if nums[mid] >= target:
                r = mid - 1  # 尝试往左边找更小的边界
            else:
                l = mid + 1  # 在右半区间继续查找

        # 循环结束时 l == r + 1，l 即为首个 >= target 的位置
        return l
```

</div>
</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与关键性质</div>
<div class="review-block-body">
<div class="complexity-grid">
  <div class="complexity-item"><span class="complexity-item-title">时间复杂度</span><span class="complexity-item-value">O(log n)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">辅助空间</span><span class="complexity-item-value">O(1)</span></div>
  <div class="complexity-item"><span class="complexity-item-title">搜索空间缩减</span><span class="complexity-item-value">每轮折半 (1/2)</span></div>
</div>
<p style="margin-top: 0.5rem; font-size: 0.88rem; color: #466370;">
<strong>要点解析</strong>：每次比较直接将候选空间排除一半，对长度为 $n$ 的数组最多比较 $\lfloor \log_2 n \rfloor + 1$ 次。由于无需递归，空间占用严格为常数级 $O(1)$。
</p>
</div>
</div>

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看答案与解析)</div>
<div class="review-block-body">

```quiz
title: 自测题 · 二分查找死循环陷阱
question: 当使用左闭右闭模板查找右边界时，如果更新逻辑写为 if condition: l = mid else: r = mid - 1，且采用默认向下取整的 mid = l + (r - l) // 2，在搜索区间仅剩 2 个元素（r = l + 1）时会发生什么？
A. 陷入死循环 (Infinite Loop)
B. 正常收敛
C. 发生越界错误
D. 误判目标值不存在
答案: A
解析: 当 r = l + 1 时，mid = l + (1) // 2 = l。若此时命中分支 l = mid，则 l 的值没有发生任何改变（仍然等于原 l），导致区间 [l, r] 的上下界完全未收缩，循环将无限重复执行下去。避开死循环的经典规则：当更新分支存在 l = mid 时，中点计算必须向上取整：mid = l + (r - l + 1) // 2。
```

</div>
</div>

</div>
</details>

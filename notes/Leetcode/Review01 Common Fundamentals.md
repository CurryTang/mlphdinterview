# 复习卡片：常考基础题 (Review Flashcards · Core Fundamentals)

本篇为算法面试核心复习卡片：精简提炼**题目定义**、**核心思路**、**关键代码**、**复杂度速记**以及**自测选择题**。点击卡片标题即可展开复习。

---

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

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看解析)</div>

```quiz
title: 自测题 · 归并排序空间开销
question: 在对一个含有 n 个节点的单链表（Singly Linked List）执行归并排序时，采用自底向上（Bottom-Up）迭代策略的最优辅助空间复杂度是多少？
A. O(1)
B. O(log n)
C. O(n)
D. O(n log n)
答案: A
解析: 数组归并排序需要 O(n) 额外数组存储合并中间值；但单链表只需修改节点的 next 指针指向即可就地归并。结合自底向上的步长迭代（1, 2, 4, 8...），既不需要递归栈，也不需要临时数组，最优辅助空间复杂度是严格的 O(1)。
```

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
1. **选主元 (Pivot)**：选定一个主元（工程上结合随机化或三数取中破坏对抗输入）。
2. **双指针划分 (Partition)**：将所有 $\le pivot$ 的元素归到左侧，所有 $\ge pivot$ 的元素归到右侧，最后将 pivot 就地安置到最终确定位置 $i$。
3. **递归处理 (Recurse)**：分别对左右子区间 $[l, i - 1]$ 与 $[i + 1, r]$ 递归排序。

</div>

<div class="review-block">
<div class="review-block-label">💻 核心代码 (最简 Python 实现)</div>

```python
def quick_sort(nums: list[int], l: int, r: int) -> None:
    if l >= r:
        return
    # 核心 Lomuto 划分：i 维护 <= pivot 的右边界
    pivot, i = nums[r], l
    for j in range(l, r):
        if nums[j] <= pivot:
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
    nums[i], nums[r] = nums[r], nums[i]

    quick_sort(nums, l, i - 1)
    quick_sort(nums, i + 1, r)
```

</div>

<div class="review-block">
<div class="review-block-label">⚡ 复杂度与特性速记</div>

- **时间复杂度**：最好 $O(n \log n)$ / 最坏 $O(n^2)$（极度倾斜切分）/ 平均 $O(n \log n)$
- **辅助空间**：$O(\log n)$（递归栈，最坏 $O(n)$）
- **稳定性**：**不稳定**（长距离跨越式交换破坏相对顺序）

</div>

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看解析)</div>

```quiz
title: 自测题 · 快速排序重复元素退化
question: 当输入数组中所有元素完全相同（例如包含 10000 个数值为 7 的数组）时，经典单向 Lomuto 划分的快速排序会发生什么？
A. 时间复杂度严重退化至 O(n^2)
B. 时间复杂度保持最优的 O(n log n)
C. 时间复杂度加速至 O(n)
D. 发生数组越界错误
答案: A
解析: 在单向 Lomuto 划分中，nums[j] <= pivot 对于所有元素全部成立，所有元素都会被逐个交换并划入同一侧，导致划分点每次仅前进 1 位，递归深度达到 n，时间退化至 O(n^2)。针对此缺陷的标准解法是采用「三路划分（Dutch National Flag 荷兰国旗法）」，将与 pivot 相等的元素整体归入中间并排除出后续递归，此时处理全相同数组仅需 O(n) 时间。
```

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

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看解析)</div>

```quiz
title: 自测题 · 动态数组扩容策略
question: 如果动态数组的扩容策略改为固定步长增加（例如每次容量满时仅执行 capacity += 1000），连续执行 N 次 push_back 的总数据搬迁时间与单次均摊复杂度分别是多少？
A. 总时间 O(N^2)，均摊 O(N)
B. 总时间 O(N log N)，均摊 O(log N)
C. 总时间 O(N)，均摊 O(1)
D. 总时间 O(N)，均摊 O(N)
答案: A
解析: 若设固定步长为 C = 1000，插入 N 个元素将触发 N/C 次扩容。每次扩容复制的元素个数分别为 C, 2C, 3C, ..., N，其总复制步数为 C * (1 + 2 + ... + N/C) ≈ O(N^2)。因此分摊到 N 次插入操作上，每次追加的均摊复杂度会退化为 O(N)。这也是为什么工业级动态数组必须采用几何倍增（乘法扩容）而非固定加法扩容的根本原因。
```

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

<div class="review-block">
<div class="review-block-label">🎯 复习自测单选题 (点击选项查看解析)</div>

```quiz
title: 自测题 · 二分查找死循环陷阱
question: 当使用左闭右闭模板查找右边界时，如果更新逻辑写为 if condition: l = mid else: r = mid - 1，且采用默认向下取整的 mid = l + (r - l) // 2，在搜索区间仅剩 2 个元素（r = l + 1）时会发生什么？
A. 陷入死循环 (Infinite Loop)
B. 正常收敛
C. 发生越界错误
D. 误判目标值不存在
答案: A
解析: 当 r = l + 1 时，向下取整 mid = l + (1) // 2 = l。若此时命中分支 l = mid，则 l 的值未发生改变，导致区间 [l, r] 完全未收缩，循环将无限执行下去。避开死循环的黄金法则：当存在 l = mid 分支时，中点计算必须向上取整：mid = l + (r - l + 1) // 2。
```

</div>

</div>
</details>

# Binary Search 统一模板

二分查找的标准写法不难，容易出错的是变体：精确匹配、答案值域、旋转数组、区间匹配，四类问题的边界处理各不相同。这份笔记把七道题目统一到同一个模板：在一个单调谓词上找边界，题目之间只替换三处内容。

## 学习顺序

题目同样来自 [NeetCode 150](https://neetcode.io/practice/practice/neetcode150) 的 Binary Search 模块。

| 顺序 | 原题 | 要掌握的内容 |
|---:|---|---|
| 1 | [704. Binary Search](https://neetcode.io/problems/binary-search/question?list=neetcode150) | 模板的最基本形式：精确匹配 |
| 2 | [74. Search a 2D Matrix](https://neetcode.io/problems/search-a-2d-matrix/question?list=neetcode150) | 二维下标映射为一维 |
| 3 | [875. Koko Eating Bananas](https://neetcode.io/problems/koko-eating-bananas/question?list=neetcode150) | 搜索空间是答案值域，不是数组下标 |
| 4 | [153. Find Minimum in Rotated Sorted Array](https://neetcode.io/problems/find-minimum-in-rotated-sorted-array/question?list=neetcode150) | 没有目标值时如何定义谓词 |
| 5 | [33. Search in Rotated Sorted Array](https://neetcode.io/problems/find-target-in-rotated-sorted-array/question?list=neetcode150) | 用键值变换把旋转数组线性化 |
| 6 | [981. Time Based Key-Value Store](https://neetcode.io/problems/time-based-key-value-store/question?list=neetcode150) | "最后一个 False"读法 |
| 7 | [4. Median of Two Sorted Arrays](https://neetcode.io/problems/median-of-two-sorted-arrays/question?list=neetcode150) | 搜索空间是分割点，用 `±inf` 做哨兵 |

## 模块一：统一模板

### 搜索空间与谓词

二分查找定位的不是数组本身，而是一个区间 `[lo, hi]`，以及定义在这个区间上的布尔谓词 `check(x)`。模板要求 `check` 在这个区间上单调：存在一个边界 `b`，满足 `x < b` 时 `check(x)` 为 `False`，`x >= b` 时为 `True`。模板返回这个边界 `b`。

区间不一定是数组下标，也可以是答案的取值范围，或者分割点的位置。谓词也不一定和"是否等于目标值"有关，只要求单调。

### 模板代码

```python
def find_first_true(lo, hi, check):
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if check(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

循环不变式：边界 `b` 始终位于 `[lo, hi]` 内。每轮循环后 `hi - lo` 至少减半，`lo == hi` 时循环结束，返回值就是 `b`。

如果 `[lo, hi)` 内没有任何位置满足 `check`，函数返回 `hi`。这不是因为 `check(hi)` 被调用并返回 `True`——`mid` 严格小于 `hi`，`hi` 本身永远不会被传给 `check`。这种情况下 `hi` 起哨兵作用，代表"没有找到"或者"答案落在搜索空间之外"。

七道题目要修改的只有三处：

```text
lo, hi：搜索空间的两端
check(mid)：单调谓词的定义
边界 b 之后的处理：直接使用、验证相等，或者取 b - 1
```

### 两种读法

| 需要的答案 | 处理方式 |
|---|---|
| 第一个满足条件的位置 | 直接返回 `b` |
| 最后一个不满足条件的位置 | 返回 `b - 1`，并检查 `b > 0` |

### 边界哨兵的两种设置方式

| 设置方式 | 含义 | 使用场景 |
|---|---|---|
| 越界一位 | `hi` 设在搜索空间外一格，例如 `len(nums)` | 目标可能不存在，需要一个"未找到"的返回值 |
| 恒真边界 | `hi` 设在搜索空间内一个已知满足 `check` 的位置 | 谓词在该位置成立可以直接证明，不需要额外判断 |

七道题目对应的设置：

| 题目 | 搜索空间 | 哨兵方式 |
|---|---|---|
| Binary Search | 下标 `[0, n]` | 越界一位 |
| Search a 2D Matrix | 展平下标 `[0, m*n]` | 越界一位 |
| Koko Eating Bananas | 速度 `[1, max(piles)]` | 恒真边界：速度为 `max(piles)` 时每堆最多用 1 小时 |
| Find Minimum in Rotated Sorted Array | 下标 `[0, n-1]` | 恒真边界：`nums[n-1] <= nums[n-1]` 恒成立 |
| Search in Rotated Sorted Array | 下标 `[0, n]` | 越界一位 |
| Time Based Key-Value Store | 下标 `[0, len(timestamps)]` | 越界一位 |
| Median of Two Sorted Arrays | 分割点 `[0, m]` | 恒真边界：右边界用 `+inf` 代替越界的一侧 |

### 常见错误

| 问题 | 影响 |
|---|---|
| `check` 在搜索空间上不单调 | 二分查找失效，模板的正确性证明不成立 |
| `mid` 用 `lo + (hi - lo) // 2`，更新 `False` 分支时漏写 `+ 1` | `lo` 不再前进，死循环 |
| 找到边界后不做等值验证 | 精确匹配类问题把"最接近的位置"当成"确实存在" |
| 哨兵设置成越界一位，却直接使用返回值 | 返回值等于 `hi`，指向不存在的位置 |

## 模块二：七道题目的映射

### Binary Search：模板的基本形式

搜索空间是下标 `[0, n]`（`n` 之外一位作为哨兵）。谓词 `check(mid) = nums[mid] >= target`，在有序数组上单调。找到边界 `lo` 后，需要验证 `nums[lo] == target`，否则目标不存在。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 下标 `[0, n]` |
| `check(mid)` | `nums[mid] >= target` |
| 哨兵方式 | 越界一位 |
| 边界处理 | 验证 `nums[lo] == target` |

#### Quick Coding：Binary Search

```python
def search(nums, target):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums)

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] >= target:
                hi = mid
            else:
                lo = mid + 1

        if lo < len(nums) and nums[lo] == target:
            return lo
        return -1
```

</details>

### Search a 2D Matrix：二维下标映射为一维

矩阵满足每行升序、且每行第一个元素大于上一行最后一个元素，因此展平后整体升序。把下标 `k` 映射为 `(k // n, k % n)`，其余与经典二分查找一致。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 展平下标 `[0, m*n]` |
| `check(mid)` | `matrix[mid // n][mid % n] >= target` |
| 哨兵方式 | 越界一位 |
| 边界处理 | 验证展平后对应位置的值等于 `target` |

#### Quick Coding：Search a 2D Matrix

```python
def searchMatrix(matrix, target):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        lo, hi = 0, m * n

        while lo < hi:
            mid = lo + (hi - lo) // 2
            row, col = divmod(mid, n)
            if matrix[row][col] >= target:
                hi = mid
            else:
                lo = mid + 1

        if lo < m * n:
            row, col = divmod(lo, n)
            return matrix[row][col] == target
        return False
```

</details>

### Koko Eating Bananas：搜索空间是答案值域

题目不要求在数组里定位元素，而是要求在速度的取值范围里找一个边界。速度越高，吃完全部香蕉需要的小时数越少，因此"吃完所需小时数 `<= h`"这个谓词随速度单调，可以直接套用模板。

$$
\text{hours\_needed(speed)} = \sum_{\text{pile}} \left\lceil \frac{\text{pile}}{\text{speed}} \right\rceil
$$

| 项目 | 内容 |
|---|---|
| 搜索空间 | 速度 `[1, max(piles)]` |
| `check(mid)` | `hours_needed(mid) <= h` |
| 哨兵方式 | 恒真边界 |
| 边界处理 | 直接返回边界 `b` |

#### Quick Coding：Koko Eating Bananas

```python
def minEatingSpeed(piles, h):
    ...
```

<details>
<summary>参考答案</summary>

```python
import math
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        def hours_needed(speed: int) -> int:
            return sum(math.ceil(pile / speed) for pile in piles)

        lo, hi = 1, max(piles)

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if hours_needed(mid) <= h:
                hi = mid
            else:
                lo = mid + 1

        return lo
```

</details>

题目约束保证 `h >= len(piles)`，所以速度取 `max(piles)` 时，每堆最多用 1 小时，总小时数不超过 `h`，恒真边界成立，不需要额外判断。

### Find Minimum in Rotated Sorted Array：没有目标值的谓词

这道题没有 `target`，谓词要从数组本身的结构里找。旋转后的数组由两段升序区间拼接而成，第一段的值都大于 `nums[-1]`，第二段的值都小于等于 `nums[-1]`。谓词 `check(mid) = nums[mid] <= nums[-1]` 恰好在两段的交界处从 `False` 变为 `True`，边界就是最小值的下标。

数组未旋转时，`nums[0] <= nums[-1]` 本身成立，边界落在下标 `0`，不需要为"未旋转"单独写分支。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 下标 `[0, n-1]` |
| `check(mid)` | `nums[mid] <= nums[-1]` |
| 哨兵方式 | 恒真边界：`nums[n-1] <= nums[n-1]` |
| 边界处理 | 返回 `nums[b]` |

#### Quick Coding：Find Minimum in Rotated Sorted Array

```python
def findMin(nums):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def findMin(self, nums: List[int]) -> int:
        lo, hi = 0, len(nums) - 1

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] <= nums[-1]:
                hi = mid
            else:
                lo = mid + 1

        return nums[lo]
```

</details>

### Search in Rotated Sorted Array：用键值变换线性化

这道题既有旋转结构，又有目标值。直接比较 `nums[mid]` 和 `target` 不再单调，因为数组不是整体有序。做法是给每个值分配一个键：

```text
key(x) = (x <= nums[-1], x)
```

第一段（大于 `nums[-1]` 的值）键的第一个分量是 `False`，第二段（小于等于 `nums[-1]` 的值）是 `True`。按元组比较键值，第一段整体排在第二段之前，段内再按数值比较，因此 `key(nums[i])` 随下标 `i` 严格单调递增，和未旋转数组的效果一致。`target` 按同样规则计算 `key(target)`，谓词改成比较键值即可。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 下标 `[0, n]` |
| `check(mid)` | `key(nums[mid]) >= key(target)` |
| 哨兵方式 | 越界一位 |
| 边界处理 | 验证 `nums[b] == target` |

#### Quick Coding：Search in Rotated Sorted Array

```python
def search(nums, target):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        pivot_value = nums[-1]

        def key(value: int):
            return (value <= pivot_value, value)

        target_key = key(target)
        lo, hi = 0, len(nums)

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if key(nums[mid]) >= target_key:
                hi = mid
            else:
                lo = mid + 1

        if lo < len(nums) and nums[lo] == target:
            return lo
        return -1
```

</details>

题目保证数组元素互不相同，键值比较不会遇到并列的情况。

### Time Based Key-Value Store：最后一个 False

`set` 按时间戳递增写入，同一个 `key` 对应的记录本身有序。`get` 要找的是"时间戳不超过查询值的最后一条记录"，属于"最后一个 False"读法：谓词 `check(mid) = timestamps[mid] > query` 找到第一个时间戳大于查询值的位置，答案下标是这个位置往前一格。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 下标 `[0, len(entries)]` |
| `check(mid)` | `entries[mid].timestamp > query` |
| 哨兵方式 | 越界一位 |
| 边界处理 | 取 `b - 1`，`b == 0` 时没有满足条件的记录 |

#### Quick Coding：Time Based Key-Value Store

```python
class TimeMap:
    def __init__(self):
        ...

    def set(self, key, value, timestamp):
        ...

    def get(self, key, timestamp):
        ...
```

<details>
<summary>参考答案</summary>

```python
from collections import defaultdict


class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)  # key -> [(timestamp, value), ...]

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        entries = self.store[key]
        lo, hi = 0, len(entries)

        while lo < hi:
            mid = lo + (hi - lo) // 2
            if entries[mid][0] > timestamp:
                hi = mid
            else:
                lo = mid + 1

        if lo == 0:
            return ""
        return entries[lo - 1][1]
```

</details>

### Median of Two Sorted Arrays：搜索空间是分割点

这道题的搜索空间既不是数组下标，也不是答案值域，而是"在较短数组里切一刀"的位置。把两个数组各切一刀，左半部分共 `half = (m + n + 1) // 2` 个元素。谓词判断这一刀是否让 `A` 的右半部分足够大：

$$
\text{check}(i) = A[i] \ge B[j-1], \quad j = \text{half} - i
$$

`i` 越大，`A[i]`（或越界时的 `+inf`）不会变小；`j` 越小，`B[j-1]`（或越界时的 `-inf`）不会变大，谓词随 `i` 单调，可以直接二分。用 `±inf` 表示越界，`i == m` 或 `j == 0` 时不需要单独判断。

找到边界 `i` 之后，左半部分最大值 `max_left` 和右半部分最小值 `min_right` 分别是中位数计算所需的两个量：总长度为奇数时中位数是 `max_left`，为偶数时是 `max_left` 和 `min_right` 的平均值。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 分割点 `i ∈ [0, m]`（`m` 为较短数组长度） |
| `check(mid)` | `A[mid] >= B[half - mid - 1]`（越界用 `±inf`） |
| 哨兵方式 | 恒真边界：`i == m` 时 `A` 的右半部分为 `+inf` |
| 边界处理 | 用边界 `i` 处的 `max_left`、`min_right` 计算中位数 |

#### Quick Coding：Median of Two Sorted Arrays

```python
def findMedianSortedArrays(nums1, nums2):
    ...
```

<details>
<summary>参考答案</summary>

```python
import math
from typing import List


class Solution:
    def findMedianSortedArrays(
        self,
        nums1: List[int],
        nums2: List[int],
    ) -> float:
        A, B = nums1, nums2
        if len(A) > len(B):
            A, B = B, A

        m, n = len(A), len(B)
        half = (m + n + 1) // 2

        def a_right_big_enough(i: int) -> bool:
            j = half - i
            a_right = A[i] if i < m else math.inf
            b_left = B[j - 1] if j > 0 else -math.inf
            return a_right >= b_left

        lo, hi = 0, m
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if a_right_big_enough(mid):
                hi = mid
            else:
                lo = mid + 1

        i = lo
        j = half - i
        a_left = A[i - 1] if i > 0 else -math.inf
        a_right = A[i] if i < m else math.inf
        b_left = B[j - 1] if j > 0 else -math.inf
        b_right = B[j] if j < n else math.inf

        max_left = max(a_left, b_left)
        if (m + n) % 2 == 1:
            return float(max_left)

        min_right = min(a_right, b_right)
        return (max_left + min_right) / 2
```

</details>

## 模块三：面试前最后检查

1. 搜索空间是数组下标、答案值域，还是分割点？
2. `check(mid)` 的定义是什么？随 `mid` 单调吗？
3. 需要"第一个 True"还是"最后一个 False"？
4. `hi` 用的是越界一位，还是恒真边界？
5. 找到边界后要不要验证，还是可以直接使用？

最后只记一句：

> 二分查找找的不是目标值，而是一个单调谓词的边界。

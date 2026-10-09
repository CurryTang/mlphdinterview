# Binary Search 统一模板

二分查找的核心本质只有一个：**在单调序列上按条件收缩区间并记录最优解**。

无论题目千变万化（等值匹配、值域二分、旋转数组、分割点），统一使用闭区间 `while left <= right` 探测，达标时存 `ans = mid` 并向目标方向收缩。解题只需明确**搜索区间**、**判定条件**与**移动方向**，直观且不易出错。

## 学习顺序

题目来自 [NeetCode 150](https://neetcode.io/practice/practice/neetcode150) 的 Binary Search 模块：

| 顺序 | 原题 | 考察核心 | 闭区间模板解题要点 |
|---:|---|---|---|
| 1 | [704. Binary Search](https://neetcode.io/problems/binary-search/question?list=neetcode150) | 基础等值查找 | 区间 `[0, n-1]`，命中直接返回，未命中左右收缩 |
| 2 | [74. Search a 2D Matrix](https://neetcode.io/problems/search-a-2d-matrix/question?list=neetcode150) | 二维矩阵二分 | 展平下标 `[0, m*n - 1]`，`(mid // n, mid % n)` 映射 |
| 3 | [875. Koko Eating Bananas](https://neetcode.io/problems/koko-eating-bananas/question?list=neetcode150) | 答案值域求最小值 | 搜索空间 `[1, max]`，达标记 `ans` 并左探（`right = mid - 1`） |
| 4 | [153. Find Minimum in Rotated Sorted Array](https://neetcode.io/problems/find-minimum-in-rotated-sorted-array/question?list=neetcode150) | 旋转数组求极小值 | 比较 `nums[mid] <= nums[-1]`，达标记 `ans` 并左探（`right = mid - 1`） |
| 5 | [33. Search in Rotated Sorted Array](https://neetcode.io/problems/find-target-in-rotated-sorted-array/question?list=neetcode150) | 旋转数组找目标 | 经典分段二分（判断哪半边有序后收缩区间） |
| 6 | [981. Time Based Key-Value Store](https://neetcode.io/problems/time-based-key-value-store/question?list=neetcode150) | 找满足条件的最大时间戳 | 时间戳 `<= query` 达标记 `ans` 并右探（`left = mid + 1`） |
| 7 | [4. Median of Two Sorted Arrays](https://neetcode.io/problems/median-of-two-sorted-arrays/question?list=neetcode150) | 双数组中位数 | 在较短数组上二分分割点，达标记 `ans` 并左探（`right = mid - 1`） |

## 模块一：通用二分模板（ans 记录法）

### 1. 传统模板的问题：为什么求最大值容易写错？

使用 `while lo < hi:`（半开区间）时：
- 求**最小值（如 LC 875 最小速度）**：条件达标往左收 `hi = mid`，退出返回 `lo`，逻辑顺畅。
- 求**最大值（如 LC 2226 最大糖数、切绳子、最大载重）**：
  - 题目求“最后一个可行的”，模板却要求反向找“第一个不可行的”，然后再做 `lo - 1`。
  - 若完全无法满足（如糖果总数小于 $k$），还要额外处理 `lo - 1` 越界的边界情况。

---

### 2. 闭区间 `ans` 记录模板（while left <= right）

使用变量 `ans` 暂存当前最优解，`check(mid)` 始终直接判断“当前值是否可行”：

```python
left, right = 最小可能值, 最大可能值
ans = 默认值  # 无法满足时的兜底，如 LC 2226 无法分配则为 0

while left <= right:
    mid = (left + right) // 2
    if check(mid):  # 当前 mid 可行
        ans = mid   # 记录当前可行解
        # 求最大值 -> 往右找更大值：left = mid + 1
        # 求最小值 -> 往左找更小值：right = mid - 1
    else:
        # 当前 mid 不可行
        # 求最大值 -> 说明数值太大，往左缩：right = mid - 1
        # 求最小值 -> 说明数值太小，往右提：left = mid + 1

return ans
```

规则总结：
- `check(mid)` 只写正向可行条件，不用反转逻辑。
- 可行就存入 `ans = mid`。
- 求最大值往右探（`left = mid + 1`），求最小值往左探（`right = mid - 1`）。
- 退出循环直接返回 `ans`，不需要 `lo - 1` 或加减偏移。

---

### 3. 代码对比：求最大值 (LC 2226) vs 求最小值 (LC 875)

#### 1. 求满足条件的最大值：LC 2226. 每个小孩最多分多少颗糖
> 规则：每堆糖可拆不可合，分给 $k$ 个孩子，每人分相同正整数颗。求最大糖数；分不够返回 0。

```python
class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        left, right = 1, max(candies)
        ans = 0  # 初始为 0，若连 1 颗都分不够直接返回 0

        while left <= right:
            mid = (left + right) // 2
            # 每个人分 mid 颗，是否至少能分给 k 个孩子
            if sum(c // mid for c in candies) >= k:
                ans = mid       # 可行，记录当前值
                left = mid + 1  # 求最大值，继续尝试更大数值
            else:
                right = mid - 1 # 糖数过大无法满足，缩小数值

        return ans
```

#### 2. 求满足条件的最小值：LC 875. 爱吃香蕉的珂珂
> 规则：$h$ 小时内吃完所有堆香蕉，求最小吃香蕉速度。

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        ans = right  # 初始为最大堆大小，必然可行

        while left <= right:
            mid = (left + right) // 2
            # 速度为 mid 时，能否在 h 小时内吃完
            if sum((p + mid - 1) // mid for p in piles) <= h:
                ans = mid        # 可行，记录当前值
                right = mid - 1  # 求最小值，继续尝试更小速度
            else:
                left = mid + 1   # 速度太慢超时，必须提速

        return ans
```

---

### 4. 两种写法的对比

| 问题点 | 传统 `while lo < hi` | `ans` 记录模板 (`left <= right`) |
|---|---|---|
| **求最大值的条件** | 需取反写 `not check` 找第一个不可行点 | 直接写可行条件 `check`，满足即 `ans = mid` |
| **返回值计算** | 需判断返回 `lo` 还是 `lo - 1` | 直接返回 `ans` |
| **无解处理** | 需额外检查 `lo == 0` 等边界 | 直接返回初始 `ans`（如 0 或 -1） |
| **指针移动** | `lo = mid` 时需向上取整防死循环 | 始终移动 `mid + 1` 或 `mid - 1`，保证收敛 |

---

### 5. 常见题目对照表

| 题目场景 | 区间 `[left, right]` | `check(mid)` 条件 | 满足条件时的移动 | 默认值 `ans` |
|---|---|---|---|---|
| **LC 2226 分糖果 (求最大)** | `1, max(candies)` | `sum(c // mid) >= k` | `left = mid + 1` | `0` |
| **LC 875 吃香蕉 (求最小)** | `1, max(piles)` | `hours_needed(mid) <= h` | `right = mid - 1` | `max(piles)` |
| **LC 704 基础查找** | `0, len(nums) - 1` | `nums[mid] == target` | 直接返回 `mid` | `-1` |
| **有序数组第一个 $\ge target$** | `0, len(nums) - 1` | `nums[mid] >= target` | `right = mid - 1` | `-1` 或 `len(nums)` |
| **有序数组最后一个 $\le target$** | `0, len(nums) - 1` | `nums[mid] <= target` | `left = mid + 1` | `-1` |
| **LC 153 旋转数组最小值** | `0, len(nums) - 1` | `nums[mid] <= nums[-1]` | `right = mid - 1` | `nums[-1]` |
| **LC 981 时间戳键值检索** | `0, len(entries) - 1` | `entries[mid].time <= query` | `left = mid + 1` | `""` |

---

## 模块二：七道题目的映射

### Binary Search：基础等值查找

闭区间搜索空间为 `[0, len(nums) - 1]`。通过 `mid` 直接比对目标值，命中直接返回，未命中按大小收缩左右边界。无需任何哨兵。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 下标闭区间 `[0, len(nums) - 1]` |
| `check(mid)` | `nums[mid] == target` |
| 指针移动 | 命中返回 `mid`；偏小 `left = mid + 1`；偏大 `right = mid - 1` |
| 默认返回值 | 退出循环未找到返回 `-1` |

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
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return -1
```

```python
# 库函数 bisect 写法：
from bisect import bisect_left
from typing import List


class SolutionBisect:
    def search(self, nums: List[int], target: int) -> int:
        idx = bisect_left(nums, target)
        return idx if idx < len(nums) and nums[idx] == target else -1
```

</details>

### Search a 2D Matrix：二维下标映射为一维

矩阵展平后是单调有序序列，总元素个数为 $m \times n$。搜索空间为闭区间 `[0, m * n - 1]`。通过 `(mid // n, mid % n)` 获取对应二维元素，直接套用闭区间二分查找模板。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 展平闭区间 `[0, m * n - 1]` |
| 坐标映射 | `row, col = mid // n, mid % n` |
| 指针移动 | 命中返回 `True`；偏小 `left = mid + 1`；偏大 `right = mid - 1` |
| 默认返回值 | 退出循环返回 `False` |

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
        left, right = 0, m * n - 1

        while left <= right:
            mid = (left + right) // 2
            val = matrix[mid // n][mid % n]
            if val == target:
                return True
            elif val < target:
                left = mid + 1
            else:
                right = mid - 1

        return False
```

</details>

### Koko Eating Bananas：答案值域求最小值

题目不要求在数组里定位元素，而是在速度的可能取值范围 `[1, max(piles)]` 内求满足耗时 $\le h$ 的**最小速度**。
- `check(mid)`：速度为 `mid` 时耗时是否达标（`sum((p + mid - 1) // mid for p in piles) <= h`）。
- 达标时：记录 `ans = mid`，并往左探寻找更小速度（`right = mid - 1`）。
- 超时时：速度不够，往右提速（`left = mid + 1`）。

$$
\text{hours\_needed(speed)} = \sum_{\text{pile}} \left\lceil \frac{\text{pile}}{\text{speed}} \right\rceil = \sum_{\text{pile}} \lfloor \frac{\text{pile} + \text{speed} - 1}{\text{speed}} \rfloor
$$

| 项目 | 内容 |
|---|---|
| 搜索空间 | 速度闭区间 `[1, max(piles)]` |
| `check(mid)` | `hours_needed(mid) <= h` |
| 移动策略 | 达标记 `ans = mid`，求最小值往左探 `right = mid - 1`；超时往右提 `left = mid + 1` |
| 默认返回值 | `ans = max(piles)`（最大堆大小必然可行） |

#### Quick Coding：Koko Eating Bananas

```python
def minEatingSpeed(piles, h):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        left, right = 1, max(piles)
        ans = right

        while left <= right:
            mid = (left + right) // 2
            # 向上取整耗时：(p + mid - 1) // mid
            if sum((p + mid - 1) // mid for p in piles) <= h:
                ans = mid        # 达标，记录当前速度
                right = mid - 1  # 求最小值，向左尝试更小速度
            else:
                left = mid + 1   # 超时，向右提速

        return ans
```

```python
# 库函数 bisect 一行写法：
from bisect import bisect_left
from typing import List


class SolutionBisect:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        r = range(1, max(piles) + 1)
        idx = bisect_left(r, True, key=lambda s: sum((p + s - 1) // s for p in piles) <= h)
        return r[idx]
```

</details>

### Find Minimum in Rotated Sorted Array：旋转数组求极小值

旋转数组由两段递增区间组成：左半段所有元素均 $> nums[-1]$，右半段所有元素均 $\le nums[-1]$。
- 判定条件：`nums[mid] <= nums[-1]`。
- 达标时：说明 `mid` 已落在右半段，当前元素是候选最小值，记录 `ans = nums[mid]`，并向左探寻找更小的分界起点（`right = mid - 1`）。
- 未达标时：说明 `mid` 还在左半段，极小值必然在右侧，向右收缩（`left = mid + 1`）。
- 未旋转数组中首个元素就满足 `<= nums[-1]`，逻辑天然兼容。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 下标闭区间 `[0, len(nums) - 1]` |
| `check(mid)` | `nums[mid] <= nums[-1]` |
| 移动策略 | 达标记 `ans = nums[mid]`，向左探 `right = mid - 1`；未达标向右探 `left = mid + 1` |
| 默认返回值 | `ans = nums[-1]`（兜底为末尾元素） |

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
        left, right = 0, len(nums) - 1
        ans = nums[-1]

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] <= nums[-1]:
                ans = nums[mid]  # 落在右半段，记录候选最小值
                right = mid - 1  # 往左探，寻找更小分界点
            else:
                left = mid + 1   # 落在左半段，极小值在右侧

        return ans
```

</details>

### Search in Rotated Sorted Array：旋转数组找目标

在闭区间 `[0, len(nums) - 1]` 内二分。虽然整体非单调，但以 `mid` 切分后，**左右两半必然至少有一半是严格有序的**：
- 若 `nums[left] <= nums[mid]`，左半段有序：若 `target` 落在左半段范围内（`nums[left] <= target < nums[mid]`），收缩右界（`right = mid - 1`），否则搜右半段（`left = mid + 1`）。
- 否则右半段有序：若 `target` 落在右半段范围内（`nums[mid] < target <= nums[right]`），收缩左界（`left = mid + 1`），否则搜左半段（`right = mid - 1`）。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 下标闭区间 `[0, len(nums) - 1]` |
| 核心逻辑 | 判断有序半段，依据 `target` 是否在其中收缩边界 |
| 指针移动 | 命中返回 `mid`；否则按有序区间调整 `left` / `right` |
| 默认返回值 | 退出循环未找到返回 `-1` |

#### Quick Coding：Search in Rotated Sorted Array

```python
def search(nums, target):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


# 解法一：经典分段二分（推荐）
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid

            # 判定左半段是否有序
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:  # 右半段有序
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1

        return -1


# 解法二：键值变换线性化（闭区间模板）
class SolutionKeyTransform:
    def search(self, nums: List[int], target: int) -> int:
        pivot = nums[-1]
        def key(x: int): return (x <= pivot, x)

        target_key = key(target)
        left, right = 0, len(nums) - 1

        while left <= right:
            mid = (left + right) // 2
            k = key(nums[mid])
            if k == target_key:
                return mid
            elif k < target_key:
                left = mid + 1
            else:
                right = mid - 1

        return -1
```

</details>

### Time Based Key-Value Store：求满足条件的最大时间戳

`set` 按时间递增存储，因此每个 `key` 对应的 `entries` 天然按时间戳有序。
`get` 要找时间戳 $\le query$ 的**最新（最大）**一条记录：
- 判定条件：`entries[mid][0] <= timestamp`。
- 达标时：说明该时间戳有效，记录 `ans = entries[mid][1]`，并向右探寻找更新的时间戳（`left = mid + 1`）。
- 未达标时：说明时间戳超出查询时间，向左收缩（`right = mid - 1`）。
- 默认值：`ans = ""`（若所有记录都大于查询时间，直接返回空字符串，无需任何边界特判）。

| 项目 | 内容 |
|---|---|
| 搜索空间 | 下标闭区间 `[0, len(entries) - 1]` |
| `check(mid)` | `entries[mid][0] <= timestamp` |
| 移动策略 | 达标记 `ans = entries[mid][1]`，求最大值向右探 `left = mid + 1`；超时向左收缩 `right = mid - 1` |
| 默认返回值 | `ans = ""`（未找到时直接返回） |

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
        left, right = 0, len(entries) - 1
        ans = ""

        while left <= right:
            mid = (left + right) // 2
            if entries[mid][0] <= timestamp:
                ans = entries[mid][1]  # 达标，记录当前有效值
                left = mid + 1         # 求最大时间戳，向右尝试更新的记录
            else:
                right = mid - 1        # 时间戳过大，向左收缩

        return ans
```

```python
# 库函数 bisect 写法：
from bisect import bisect_right
from collections import defaultdict


class TimeMapBisect:
    def __init__(self):
        self.store = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        entries = self.store[key]
        idx = bisect_right(entries, timestamp, key=lambda x: x[0]) - 1
        return entries[idx][1] if idx >= 0 else ""
```

</details>

### Median of Two Sorted Arrays：在较短数组上二分分割点

搜索空间是在较短数组 $A$（长度 $m$）上的分割线位置 $i \in [0, m]$。两数组左半部分共需 `half = (m + n + 1) // 2` 个元素，此时 $B$ 的分割点 $j = \text{half} - i$。
- 判定条件：当前分割是否满足 $A$ 的右部元素 $\ge B$ 的左部元素（`A[mid] >= B[half - mid - 1]`）。
- 达标时：记录当前分割点 `ans = mid`，求最小分割点往左探（`right = mid - 1`）。
- 未达标时：$A$ 的切分过小导致右部不足，往右探（`left = mid + 1`）。
- 退出循环后直接使用 `ans` 进行中位数计算。

$$
\text{check}(i) = A[i] \ge B[j-1], \quad j = \text{half} - i
$$

| 项目 | 内容 |
|---|---|
| 搜索空间 | 分割点闭区间 `[0, m]`（$m \le n$） |
| `check(mid)` | `a_right >= b_left`（越界取 $\pm\infty$） |
| 移动策略 | 达标记 `ans = mid`，求最小切分向左探 `right = mid - 1`；未达标向右探 `left = mid + 1` |
| 默认返回值 | `ans = m` |

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


# 解法一：闭区间 ans 记录法（手写模板）
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

        left, right = 0, m
        ans = m

        while left <= right:
            mid = (left + right) // 2
            if a_right_big_enough(mid):
                ans = mid        # 达标，记录分割点
                right = mid - 1  # 求最小分割点，向左探
            else:
                left = mid + 1   # 切分不足，向右提

        i = ans
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


# 解法二：库函数 bisect 一行定位分割点（好记易背）
from bisect import bisect_left


class SolutionBisect:
    def findMedianSortedArrays(
        self,
        nums1: List[int],
        nums2: List[int],
    ) -> float:
        A, B = (nums1, nums2) if len(nums1) <= len(nums2) else (nums2, nums1)
        m, n = len(A), len(B)
        half = (m + n + 1) // 2

        def check(i: int) -> bool:
            j = half - i
            a_right = A[i] if i < m else math.inf
            b_left = B[j - 1] if j > 0 else -math.inf
            return a_right >= b_left

        # 在 [0, m] 中二分查找首个让 check(i) 为 True 的切分位置
        i = bisect_left(range(m + 1), True, key=check)
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

三步检查清单：
1. **区间 `[left, right]`**：下标是 `[0, len(nums) - 1]`；答案值域是 `[min_val, max_val]`。
2. **判定 `check(mid)`**：直接写“当前 mid 是否达标/可行”。
3. **移动与记录**：
   - 达标即存 `ans = mid`；
   - 求最大值往右探（`left = mid + 1`），求最小值往左探（`right = mid - 1`）；
   - 退出循环直接返回 `ans`。


## 模块四：二分高频扩展真题

### 有序数组中三分频众数的对数探针检索 (Majority Element in Sorted Array via Sublinear Binary Search Probe)

#### 核心心智（探针采样 + 二分精确验算）
- **问题**：在已排序数组中找出所有频次严格大于 $\lfloor n/3 \rfloor$ 的元素。
- **抽屉原理与探针采样**：
  若元素频次 $> n/3$，该连续段长度至少为 $\lfloor n/3 \rfloor + 1$。它必然会横跨分位点下标 `idx1 = n // 3` 或 `idx2 = 2 * n // 3`。
  因此，全局符合条件的众数至多有 2 个，且必然来自 `nums[idx1]` 或 `nums[idx2]`！
- **对数级快速验算**：
  对提取的去重候选数，使用 `bisect_right(nums, cand) - bisect_left(nums, cand)` 在 $O(\log n)$ 内获知其确切频次。
  **整体时间复杂度严格为 $O(\log n)$，突破线性瓶颈！**

```python
from bisect import bisect_left, bisect_right
from typing import List

class SortedMajoritySolution:
    @classmethod
    def findMajorityElementsSorted(cls, nums: List[int]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        threshold = n // 3

        # 探针采样抽取候选人
        candidates = {nums[n // 3], nums[(2 * n) // 3]}
        res = []

        for cand in candidates:
            left = bisect_left(nums, cand)
            right = bisect_right(nums, cand)
            if right - left > threshold:
                res.append(cand)

        return sorted(res)
```

---

## 模块五：Python 标准库 bisect 使用指南

Python 标准库提供了内置的 `bisect` 模块（底层的 C 实现速度极快）。在面试中，如果二分本身只是解题的一个辅助环节（例如最长递增子序列 LIS、贪心区间调度、时间戳版本检索等），使用 `bisect` 可以大幅缩短编码时间并规避边界错误。

### 1. 函数签名与参数详解

```python
bisect_left(a, x, lo=0, hi=len(a), *, key=None)
bisect_right(a, x, lo=0, hi=len(a), *, key=None)  # 别名 bisect
insort_left(a, x, lo=0, hi=len(a), *, key=None)
insort_right(a, x, lo=0, hi=len(a), *, key=None)  # 别名 insort
```

各参数的作用与注意事项：

| 参数 | 类型与默认值 | 作用说明 | 常见考点与注意事项 |
|---|---|---|---|
| `a` | 序列 (必填) | 待搜索的已排序序列 | 支持随机访问（`list`, `tuple`, `range` 等）；必须按比较键单调升序 |
| `x` | 任意类型 (必填) | 要比对的目标值 (target) | 若指定了 `key`，`x` 直接与 `key(element)` 的比较结果比对 |
| `lo` | `int = 0` | 检索区间的左端点（包含） | 限定在子区间 `[lo, hi)` 中检索；**避免使用切片 `a[lo:hi]` 产生额外内存拷贝** |
| `hi` | `int = len(a)` | 检索区间的右端点（不包含） | 半开区间；返回的索引依然是相对于原序列 `a` 的全局下标 |
| `key` | 单参函数 (Python 3.10+) | 从元素中提取用于比较的键 | **只作用于序列元素 `a[i]`，不作用于目标 `x`**；用于复杂对象或谓词投影 |

#### 参数使用细节：
1. **`lo` 与 `hi` 的优势（避免切片拷贝）**：
   - 如果写 `bisect_left(nums[10:50], target)`，Python 会新建一个长度为 40 的切片子列表，带来 $O(k)$ 的时间和空间开销，且返回的下标需手动加偏移量。
   - 使用 `bisect_left(nums, target, lo=10, hi=50)`，二分直接在原数组的局部区间进行，无切片内存开销，返回的下标直接就是原数组中的绝对下标。
2. **`key` 的单向作用特性**：
   - `key` 函数只被调用在数组元素上（即 `key(a[mid])` 与 `x` 进行比较）。
   - 因此传入的 `x` 必须是已经提取后的比较键类型。例如对 `[(1, 'a'), (3, 'b')]` 二分时间戳时，`x` 传数字 `3`，`key=lambda item: item[0]`。
3. **`insort_left` / `insort_right`**：
   - 查找到位置后就地调用 `a.insert(idx, x)` 将新元素插入并保持序列升序。
   - 注意：`list.insert` 涉及内存平移，单次耗时 $O(n)$。频繁插入且频繁查询建议使用 `heapq` 或平衡树结构。

---

### 2. 数组四大经典查询

`bisect_left` 查找第一个 $\ge target$ 的插入点；`bisect_right`（别名 `bisect`）查找第一个 $> target$ 的插入点：

| 查找目标 | 对应写法 | 说明与边界 |
|---|---|---|
| **第一个 $\ge target$** | `idx = bisect_left(nums, target)` | 若都不满足返回 `len(nums)` |
| **第一个 $> target$** | `idx = bisect_right(nums, target)` | 若都不满足返回 `len(nums)` |
| **最后一个 $\le target$** | `idx = bisect_right(nums, target) - 1` | 若都不满足为 `-1`（即 `idx < 0`） |
| **最后一个 $< target$** | `idx = bisect_left(nums, target) - 1` | 若都不满足为 `-1`（即 `idx < 0`） |
| **查找是否存在** | `i = bisect_left(nums, target)`<br>`found = (i < len(nums) and nums[i] == target)` | LC 704 标准等值判断 |

---

### 3. Python 3.10+ `key=` 参数支持

从 Python 3.10 起，`bisect` 支持 `key=` 参数，可以直接针对对象属性或复合元组进行投影二分。

#### 示例：LC 981 基于时间戳的键值存储
```python
from bisect import bisect_right

# entries 存储有序记录: [(timestamp_1, val_1), (timestamp_2, val_2), ...]
def get(entries, query_time):
    # 查找最后一个 timestamp <= query_time 的记录
    idx = bisect_right(entries, query_time, key=lambda x: x[0]) - 1
    return entries[idx][1] if idx >= 0 else ""
```

---

### 4. 用 `range` + `key` 进行答案值域二分

`range()` 在 Python 中是支持 $O(1)$ 随机访问的虚拟序列（不占用实际数组内存），配合 `key` 参数可直接在答案值域上执行 `bisect`。

Python 中布尔值满足 `False < True`（即 $0 < 1$），因此若可行性谓词序列呈现 `[False, False, ..., True, True]`，`bisect_left(..., True)` 会直接定位到首个 `True`：

#### 示例 A：求最小值 · LC 875 爱吃香蕉的珂珂
```python
from bisect import bisect_left
from typing import List

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        r = range(1, max(piles) + 1)
        # 速度过小耗时超时 (False)，达标后为 True；查找首个 True
        idx = bisect_left(r, True, key=lambda s: sum((p + s - 1) // s for p in piles) <= h)
        return r[idx]
```

#### 示例 B：求最大值 · LC 2226 每个小孩最多分多少颗糖
```python
from bisect import bisect_left
from typing import List

class Solution:
    def maximumCandies(self, candies: List[int], k: int) -> int:
        # 糖数过大导致份数不足时条件为 True (反向判断)，找首个不可行点
        # 其下标恰好等于最大可满足的糖数（从 1 开始计）
        return bisect_left(
            range(1, max(candies) + 1),
            True,
            key=lambda s: sum(c // s for c in candies) < k
        )
```

#### 示例 C：分割点二分 · LC 4 寻找两个正序数组的中位数
```python
from bisect import bisect_left
import math
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        A, B = (nums1, nums2) if len(nums1) <= len(nums2) else (nums2, nums1)
        m, n = len(A), len(B)
        half = (m + n + 1) // 2

        # 判定 A 的右部是否 >= B 的左部
        def check(i: int) -> bool:
            j = half - i
            a_right = A[i] if i < m else math.inf
            b_left = B[j - 1] if j > 0 else -math.inf
            return a_right >= b_left

        # 在 [0, m] 中二分查找首个让 check(i) 为 True 的切分位置
        i = bisect_left(range(m + 1), True, key=check)
        j = half - i

        a_left = A[i - 1] if i > 0 else -math.inf
        a_right = A[i] if i < m else math.inf
        b_left = B[j - 1] if j > 0 else -math.inf
        b_right = B[j] if j < n else math.inf

        max_left = max(a_left, b_left)
        if (m + n) % 2 == 1:
            return float(max_left)
        return (max_left + min(a_right, b_right)) / 2.0
```

---

### 5. 面试中的选择策略

- **二分是题目主要考察点**（如面试官要求“手写二分查找”、“分析旋转数组边界”）：**必须手写**闭区间 `while left <= right` 模板，展示对边界收敛与循环不变量的掌握。
- **二分只是解题辅助步骤**（如 Hard 题的局部优化、求 LIS、贪心调度）：**优先调用 `bisect`**，并向面试官说明“此处是有序序列，使用标准库 `bisect` 保证 $O(\log n)$ 且避免边界越界”，展示对标准库的熟练运用。

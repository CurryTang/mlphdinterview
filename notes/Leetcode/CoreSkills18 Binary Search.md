# Binary Search 统一模板

二分查找的核心本质只有一个：**在单调序列上找分界点（First True）**。

无论题目千变万化（等值匹配、值域二分、旋转数组、分割点），模板代码永远是 6 行，**解题只需填好 3 个空**，10 秒内就能写完且绝无 bug。

## 学习顺序

题目来自 [NeetCode 150](https://neetcode.io/practice/practice/neetcode150) 的 Binary Search 模块：

| 顺序 | 原题 | 考察核心 | 三步填空的关键点 |
|---:|---|---|---|
| 1 | [704. Binary Search](https://neetcode.io/problems/binary-search/question?list=neetcode150) | 基础等值查找 | 找第一个 $\ge target$，最后验证相等 |
| 2 | [74. Search a 2D Matrix](https://neetcode.io/problems/search-a-2d-matrix/question?list=neetcode150) | 二维矩阵二分 | 展平下标 `[0, m*n]`，`(mid // n, mid % n)` 映射 |
| 3 | [875. Koko Eating Bananas](https://neetcode.io/problems/koko-eating-bananas/question?list=neetcode150) | 答案值域二分 | 搜索空间是速度 `[1, max]`，`check` 算耗时是否达标 |
| 4 | [153. Find Minimum in Rotated Sorted Array](https://neetcode.io/problems/find-minimum-in-rotated-sorted-array/question?list=neetcode150) | 旋转数组极值 | 比较对象是末尾元素 `nums[-1]`，直接返回 `nums[lo]` |
| 5 | [33. Search in Rotated Sorted Array](https://neetcode.io/problems/find-target-in-rotated-sorted-array/question?list=neetcode150) | 旋转数组找目标 | 经典分段二分（哪半边有序）/ 键值单调化 |
| 6 | [981. Time Based Key-Value Store](https://neetcode.io/problems/time-based-key-value-store/question?list=neetcode150) | 找“最后一个满足” | 找第一个大于查询值的，返回 `lo - 1` |
| 7 | [4. Median of Two Sorted Arrays](https://neetcode.io/problems/median-of-two-sorted-arrays/question?list=neetcode150) | 双数组中位数 | 在较短数组上二分分割线位置 |

## 模块一：统一模板与极简三步填空法

### 1. 核心模板代码（闭眼默写这 6 行）

二分查找的本质只有一个：**在一个单调分界线上，找第一个满足条件的点（First True）**。无论题目千变万化，核心代码永远是这 6 行：

```python
def find_first_true(lo, hi, check):
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if check(mid):
            hi = mid      # 满足条件，向左收缩找更早的合法点
        else:
            lo = mid + 1  # 不满足，向右排除
    return lo             # 退出时必定 lo == hi，就是第一个满足 check 的位置
```

```binary-search-template-demo
```

> **不变定理**：循环退出时必定 `lo == hi`，它永远精准指向**“第一个使 `check(mid)` 为 True 的位置”**。

---

### 2. 那几个东西怎么填？（极简三步填空心智卡片）

写二分查找时，脑子里不需要背复杂的镜像表或哨兵分类，**只需要按顺序填好 3 个空**：

#### 第一步：区间 `[lo, hi]` 怎么填？

- **普通数组下标（求位置）**：`lo = 0, hi = len(nums)`
  - *为什么 `hi` 是 `len(nums)`？* 因为目标可能根本不存在，此时返回 `len(nums)` 充当“未找到 / 越界”的天然哨兵标记。
- **答案值域（求最小速度、最小容量等）**：`lo = 最小可能值, hi = 最大可能值`
  - *为什么不需要加 1？* 因为已知最大可能值必定满足条件（恒真上界），区间两端都是闭合有效值。

---

#### 第二步：`check(mid)` 怎么填？（4 大边界秒杀口诀）

绝大多数人容易搞混 `>` 还是 `>=`。核心语义永远只有一句话：
> **“当前 `mid` 是否已经满足（或达到）题目要求？”**
> - 一旦满足（`True`），说明我们找到了一个可行解，但想看左边有没有更小/更早的，所以往左收：`hi = mid`。
> - 不满足（`False`），必须往右找：`lo = mid + 1`。

在单调有序数组中找数值时，牢记这套**全网最好记的口诀**：

| 想找的目标 | `check(mid)` 怎么写 | 最终答案 | 秒记口诀 |
|---|---|---|---|
| **第一个 $\ge target$** | `nums[mid] >= target` | `lo` | **找“第一个”，直接写条件** |
| **第一个 $> target$** | `nums[mid] > target` | `lo` | **找“第一个”，直接写条件** |
| **最后一个 $\le target$** | `nums[mid] > target` | `lo - 1` | **找“最后一个”，找右邻居再减一** |
| **最后一个 $< target$** | `nums[mid] >= target` | `lo - 1` | **找“最后一个”，找右邻居再减一** |

> 💡 **两句话口诀**：
> 1. **找“第一个”直接写**：求什么条件 `check` 就写什么条件，答案就是 `lo`。
> 2. **找“最后一个”看右邻居**：求“最后一个 $\le$”，就找“第一个 $>$”，然后 `lo - 1`；求“最后一个 $<$”，就找“第一个 $\ge$”，然后 `lo - 1`。

**遇到反向问题怎么办？（如 Koko 吃香蕉，速度越大耗时越小）**
完全不需要记所谓的“镜像规则”！凭人类直觉即可：
- 问：当前速度 `mid` 耗时是否达标？
- 写：`hours_needed(mid) <= h`。
- 达标就 `hi = mid` 继续向左找更小速度；不达标就 `lo = mid + 1`。一气呵成！

---

#### 第三步：返回值怎么填？

退出循环时必定 `lo == hi`，根据业务目标做最后一步收尾：
1. **精确查找目标值（如 LC 704、LC 74）**：
   - 检查是否越界且命中：`if lo < len(nums) and nums[lo] == target: return lo`，否则返回 `-1`。
2. **值域二分找极值（如 LC 875 吃香蕉、LC 153 旋转最小值）**：
   - 直接返回 `lo`（或 `nums[lo]`），它就是满足条件的最小解。
3. **找最后一个满足的位置（如 LC 981）**：
   - 返回 `lo - 1`（若 `lo == 0` 说明一个都不存在）。

---

### 3. 七道经典题目三步填空速查表

面对任何二分题目，把这 3 个空填进去即可：

| 题目 | 空 1：区间 `[lo, hi]` | 空 2：`check(mid)` 判定条件 | 空 3：返回值处理 |
|---|---|---|---|
| **LC 704. Binary Search** | `0, len(nums)` | `nums[mid] >= target` | 验证 `lo < n and nums[lo] == target`，否则 `-1` |
| **LC 74. Search a 2D Matrix** | `0, m * n`（展平） | `matrix[mid // n][mid % n] >= target` | 验证展平对应元素 `== target`，否则 `False` |
| **LC 875. Koko Eating Bananas** | `1, max(piles)` | `hours_needed(mid) <= h` | 直接返回 `lo`（最小可行速度） |
| **LC 153. Find Min in Rotated Array** | `0, len(nums) - 1` | `nums[mid] <= nums[-1]` | 直接返回 `nums[lo]` |
| **LC 33. Search in Rotated Array** | `0, len(nums)` | 键值变换 `key(nums[mid]) >= key(target)`<br>*(或常规分段二分)* | 验证 `nums[lo] == target`，否则 `-1` |
| **LC 981. Time Based KV Store** | `0, len(entries)` | `entry.time > query`（找首个超时的） | 取前一位 `lo - 1`（若 `lo == 0` 返回空） |
| **LC 4. Median of Two Sorted Arrays** | `0, m`（短数组切分点） | `A[mid] >= B[half - mid - 1]` | 左右边界极值计算中位数 |

---

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


# 解法一：统一模板法（键值变换线性化）
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


# 解法二：面试最常用的经典分段二分（直观易写）
class SolutionClassic:
    def search(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums) - 1

        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if nums[mid] == target:
                return mid

            # 判断哪一半是有序的
            if nums[lo] <= nums[mid]:  # 左半段有序
                if nums[lo] <= target < nums[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
            else:  # 右半段有序
                if nums[mid] < target <= nums[hi]:
                    lo = mid + 1
                else:
                    hi = mid - 1

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

三步填空检查清单：
1. **空 1（区间）**：是下标 `[0, n]`，还是答案值域 `[min, max]`？
2. **空 2（判定）**：找“第一个”直接写；找“最后一个”看右邻居再减一；值域二分直觉问“是否达标”？
3. **空 3（收尾）**：精确查找需验证 `nums[lo] == target`；找最后一个需取 `lo - 1` 并检查 `lo > 0`。

最后只记一句：

> 二分查找找的不是目标值，而是一个单调分界点；只需填好区间、check、返回值 3 个空。


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

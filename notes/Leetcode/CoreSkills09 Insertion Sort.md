# 排序算法

排序题表面上在问六段代码，实际更常在问四条轴：时间复杂度、额外空间、稳定性、是否原地。面试里先判断题目在卡哪一条轴，再决定写哪一种排序；否则很容易把“会背模板”误当成“会选算法”。

## 模块一：六种排序的总览对比

| 算法 | 最好 | 平均 | 最坏 | 额外空间 | 稳定 | 原地 | 核心思路 |
|---|---|---|---|---|---|---|---|
| Insertion Sort | `O(n)` | `O(n^2)` | `O(n^2)` | `O(1)` | 是 | 是 | 维护左侧有序前缀，把当前元素插入正确位置 |
| Selection Sort | `O(n^2)` | `O(n^2)` | `O(n^2)` | `O(1)` | 否 | 是 | 每一轮在未排序后缀中选最小值，放到前面 |
| Bubble Sort | `O(n)`（带提前退出） | `O(n^2)` | `O(n^2)` | `O(1)` | 是 | 是 | 反复交换相邻逆序对，让最大值逐轮冒到右侧 |
| Merge Sort | `O(n log n)` | `O(n log n)` | `O(n log n)` | `O(n)` | 是 | 否 | 分治拆分，分别排好两半，再线性合并 |
| Quick Sort | `O(n log n)` | `O(n log n)` | `O(n^2)` | 平均 `O(log n)` 递归栈，最坏 `O(n)` | 否 | 是 | 选 pivot 做 partition，再递归左右两侧 |
| Heap Sort | `O(n log n)` | `O(n log n)` | `O(n log n)` | `O(1)` | 否 | 是 | 先建最大堆，再把堆顶依次放到数组末尾 |

这张表按本章后续代码口径统计。Insertion 用 `>` 右移，因此稳定；Bubble 只在 `>` 时交换相邻元素，因此稳定；Merge 在相等时先取左边，因此稳定；Selection、Quick、Heap 这版实现都不稳定。原地一列不把递归栈算作额外数组。

## 模块二：`O(n^2)` 家族

这三种排序都只在数组内部做局部操作，但代价结构不同。Insertion 的代价主要是位移，适合近乎有序的输入；Selection 每轮完整扫描后缀，但最多一次交换；Bubble 只做相邻交换，因此最容易保持稳定，也最容易加入提前退出。

```simple-sort-race-demo
```

### Insertion Sort

Insertion sort 维护“左侧前缀已经有序”这个不变式。每一轮取出 `key`，把更大的元素整体右移，再把 `key` 写回空位。它和 Selection、Bubble 的区别在于：它不先找最小值，也不做相邻交换，而是直接为当前元素寻找插入点。

#### Quick Coding: Insertion Sort

```python
def insertion_sort(nums):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


def insertion_sort(nums: List[int]) -> List[int]:
    for i in range(1, len(nums)):
        key = nums[i]
        j = i - 1

        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1

        nums[j + 1] = key

    return nums
```

比较条件必须写成 `>`，不要写成 `>=`。否则相等元素会跨过彼此，稳定性会被破坏。

</details>

### Selection Sort

Selection sort 的不变式是“前缀已经处在最终位置”。第 `i` 轮完整扫描区间 `[i, n - 1]`，找到最小值下标 `min_idx`，再把它放到 `i`。它和 Insertion 的区别是先扫描再交换，和 Bubble 的区别是每轮最多一次交换，但扫描量不会因为输入接近有序而下降。

#### Quick Coding: Selection Sort

```python
def selection_sort(nums):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


def selection_sort(nums: List[int]) -> List[int]:
    n = len(nums)

    for i in range(n - 1):
        min_idx = i

        for j in range(i + 1, n):
            if nums[j] < nums[min_idx]:
                min_idx = j

        if min_idx != i:
            nums[i], nums[min_idx] = nums[min_idx], nums[i]

    return nums
```

这版写法是原地的，但不稳定。原因不是比较条件，而是最终交换：后出现的较小值可能跨过前面的相等元素。

</details>

### Bubble Sort

Bubble sort 每轮扫描当前未排序前缀，只要看到逆序对就交换相邻元素。这样当前最大值会逐步冒到右端。它和 Insertion 的区别是只做相邻交换，和 Selection 的区别是会在一轮内部做多次交换，但因此可以自然支持稳定性和提前退出。

#### Quick Coding: Bubble Sort

```python
def bubble_sort(nums):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


def bubble_sort(nums: List[int]) -> List[int]:
    n = len(nums)

    for end in range(n - 1, 0, -1):
        swapped = False

        for i in range(end):
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                swapped = True

        if not swapped:
            break

    return nums
```

如果一整轮都没有发生交换，数组已经有序，最好情况就是 `O(n)`。

</details>

## 模块三：`O(n log n)` 家族

这三种排序都跳出了平方级，但交换的是不同资源。Merge 用额外数组换稳定性；Quick 用 partition 换很好的实践常数；Heap 用堆结构换最坏情况保证和 `O(1)` 额外空间。

```efficient-sort-race-demo
```

### Merge Sort

Merge sort 的结构是先拆再合。递归返回时，左右两半已经分别有序，只需用双指针做一次线性合并。这版实现的关键点有两个：合并时相等元素优先取左边，因此稳定；合并需要缓冲区，因此不是原地排序。

#### Quick Coding: Merge Sort

```python
def merge_sort(nums):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


def merge_sort(nums: List[int]) -> List[int]:
    if len(nums) <= 1:
        return nums

    temp = [0] * len(nums)

    def sort(lo: int, hi: int) -> None:
        if lo >= hi:
            return

        mid = (lo + hi) // 2
        sort(lo, mid)
        sort(mid + 1, hi)

        i, j, k = lo, mid + 1, lo

        while i <= mid and j <= hi:
            if nums[i] <= nums[j]:
                temp[k] = nums[i]
                i += 1
            else:
                temp[k] = nums[j]
                j += 1
            k += 1

        while i <= mid:
            temp[k] = nums[i]
            i += 1
            k += 1

        while j <= hi:
            temp[k] = nums[j]
            j += 1
            k += 1

        for index in range(lo, hi + 1):
            nums[index] = temp[index]

    sort(0, len(nums) - 1)
    return nums
```

稳定性的关键在 `if nums[i] <= nums[j]` 这一行：相等时先取左边，原始相对顺序才会被保留。

</details>

### Quick Sort

Quick sort 的核心不是递归，而是 partition。先选一个 `pivot`，把 `< pivot` 的元素放到左边，把 `>= pivot` 的元素放到右边，再递归两侧。它和 Merge 的区别是原地、局部性更好；和 Heap 的区别是平均常数更小，但固定选坏 pivot 时会退化到 `O(n^2)`。

#### Quick Coding: Quick Sort

```python
def quick_sort(nums):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


def quick_sort(nums: List[int]) -> List[int]:
    def sort(lo: int, hi: int) -> None:
        if lo >= hi:
            return

        pivot = nums[hi]
        store = lo

        for scan in range(lo, hi):
            if nums[scan] < pivot:
                nums[scan], nums[store] = nums[store], nums[scan]
                store += 1

        nums[store], nums[hi] = nums[hi], nums[store]

        sort(lo, store - 1)
        sort(store + 1, hi)

    sort(0, len(nums) - 1)
    return nums
```

这版是 Lomuto partition，原地、容易讲清楚，但固定选择末尾元素做 pivot 仍然会有最坏 `O(n^2)`。

</details>

### Heap Sort

Heap sort 先把数组原地建成最大堆，再不断把堆顶最大值交换到数组末尾，并对缩小后的堆重新 heapify。它和 Quick 的区别是最坏情况仍然保持 `O(n log n)`；和 Merge 的区别是不需要 `O(n)` 缓冲区，但缓存局部性通常更差，也不稳定。

#### Quick Coding: Heap Sort

```python
def heap_sort(nums):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


def heap_sort(nums: List[int]) -> List[int]:
    def heapify(heap_size: int, root: int) -> None:
        while True:
            largest = root
            left = 2 * root + 1
            right = 2 * root + 2

            if left < heap_size and nums[left] > nums[largest]:
                largest = left

            if right < heap_size and nums[r\right] > nums[largest]:
                largest = right

            if largest == root:
                return

            nums[root], nums[largest] = nums[largest], nums[root]
            root = largest

    n = len(nums)

    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)

    for end in range(n - 1, 0, -1):
        nums[0], nums[end] = nums[end], nums[0]
        heapify(end, 0)

    return nums
```

建堆阶段是 `O(n)`，不是 `O(n log n)`。总复杂度来自后续 `n - 1` 次抽取，每次都可能触发一次 `O(log n)` 的下沉。

</details>

## 模块四：如何选择

### 为什么 quicksort 往往比 merge sort 和 heap sort 更快

quicksort 在数组上通常有更好的缓存局部性，partition 只在当前区间内顺序扫描，常数也更小。即使它的最坏情况更差，平均实践表现仍然经常最好。

### 怎么让 quicksort 更稳健

固定取首元素或末尾元素，在已排序输入上容易退化。常见修正是随机 pivot、median-of-three，或者在重复值很多时改成三路快排。

### 为什么 merge sort 适合链表和外部排序

merge sort 只要求顺序访问，不依赖随机访问。链表上做“从中间切开、分别排序、再合并”很自然；外部排序也可以把大文件分块排序后再流式合并。稳定性在多关键字排序里也经常是硬需求。

### 什么时候还应该主动选 insertion sort

当数组很小、近乎有序，或者你在写 hybrid sort 的 base case 时，Insertion 常常比更复杂的 `O(n log n)` 算法更合适。它的代码短、常数小、局部位移代价低。

### 什么时候 heap sort 更有意义

当题目同时要求最坏情况 `O(n log n)` 和 `O(1)` 额外空间时，Heap 是明确候选。它的劣势是访问模式跳跃，实践常数通常不如 quicksort。

### 稳定性为什么重要

稳定排序会保留相等元素的原始相对顺序。标准多关键字做法是：先按次关键字排序，再按主关键字做一次稳定排序。第二次排序时，相等主关键字之间的次关键字顺序才不会丢失。

## 最后检查清单

1. 题目是否明确要求稳定性？
2. 题目是否允许 `O(n)` 额外空间？
3. 输入是否近乎有序，或者规模是否很小？
4. 是否需要最坏情况 `O(n log n)` 保证？
5. 数据结构是数组、链表，还是外部文件？

> 先识别约束轴，再从六种排序里选最合适的那一个。

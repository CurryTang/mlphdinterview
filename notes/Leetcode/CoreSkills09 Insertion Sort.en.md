# Sorting Algorithms

Sorting questions look like six different code templates, but interviews usually test four axes instead: time complexity, extra space, stability, and whether the algorithm is in-place. The practical task is to identify which axis the problem actually cares about, then pick the matching sort.

## Module 1: A Side-by-Side Comparison

| Algorithm | Best | Average | Worst | Extra space | Stable | In-place | Core idea |
|---|---|---|---|---|---|---|---|
| Insertion Sort | `O(n)` | `O(n^2)` | `O(n^2)` | `O(1)` | Yes | Yes | keep a sorted prefix on the left and insert the current value into it |
| Selection Sort | `O(n^2)` | `O(n^2)` | `O(n^2)` | `O(1)` | No | Yes | repeatedly choose the minimum from the unsorted suffix |
| Bubble Sort | `O(n)` (with early exit) | `O(n^2)` | `O(n^2)` | `O(1)` | Yes | Yes | repeatedly swap adjacent out-of-order pairs |
| Merge Sort | `O(n log n)` | `O(n log n)` | `O(n log n)` | `O(n)` | Yes | No | divide, sort both halves, then merge them linearly |
| Quick Sort | `O(n log n)` | `O(n log n)` | `O(n^2)` | `O(log n)` average recursion stack, `O(n)` worst | No | Yes | partition around a pivot, then recurse on both sides |
| Heap Sort | `O(n log n)` | `O(n log n)` | `O(n log n)` | `O(1)` | No | Yes | build a max heap, then move the root to the end repeatedly |

This table matches the exact implementations in this chapter. Insertion uses `>` when shifting, so it is stable. Bubble swaps adjacent values only when `>`, so it is stable. Merge prefers the left side on ties, so it is stable. The Selection, Quick, and Heap implementations here are not stable. The in-place column ignores recursion stack space.

## Module 2: The `O(n^2)` Family

All three of these algorithms work entirely inside the array, but they spend their effort differently. Insertion pays mostly in shifts and benefits from nearly sorted input. Selection scans a full suffix, then performs at most one swap per pass. Bubble performs only adjacent swaps, which makes stability and early exit easy.

```simple-sort-race-demo
```

### Insertion Sort

Insertion sort maintains the invariant that the prefix on the left is already sorted. Each pass removes `key`, shifts larger values rightward, and writes `key` into the gap. Unlike Selection and Bubble, it does not first search for a global minimum or perform adjacent swaps only; it directly finds the insertion point for the current value.

#### Quick Coding: Insertion Sort

```python
def insertion_sort(nums):
    ...
```

<details>
<summary>Reference answer</summary>

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

The comparison has to stay `>`, not `>=`. Otherwise equal values can cross and stability is lost.

</details>

### Selection Sort

Selection sort maintains a different invariant: the prefix already contains final positions. On pass `i`, it scans `[i, n - 1]`, finds `min_idx`, and places that minimum into `i`. Unlike Insertion, it scans first and swaps later. Unlike Bubble, it performs at most one swap per pass, but its scan cost does not improve on nearly sorted input.

#### Quick Coding: Selection Sort

```python
def selection_sort(nums):
    ...
```

<details>
<summary>Reference answer</summary>

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

This implementation is in-place but not stable. The issue is the final swap: a later smaller value can jump ahead of equal values that appeared earlier.

</details>

### Bubble Sort

Bubble sort scans the current unsorted prefix and swaps every adjacent inversion it sees. That makes the current maximum drift to the right end. Unlike Insertion, it never shifts a block. Unlike Selection, it may swap many times inside one pass, but that same structure gives it stability and a natural early-exit optimization.

#### Quick Coding: Bubble Sort

```python
def bubble_sort(nums):
    ...
```

<details>
<summary>Reference answer</summary>

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

If an entire pass makes no swap, the array is already sorted and the algorithm stops in `O(n)` time.

</details>

## Module 3: The `O(n log n)` Family

These three algorithms leave the quadratic family, but each pays for that in a different way. Merge uses extra memory to stay stable. Quick uses partitioning and usually wins on constant factors. Heap uses a heap structure to keep worst-case guarantees and `O(1)` extra space.

```efficient-sort-race-demo
```

### Merge Sort

Merge sort first splits, then merges. When recursion returns, the two halves are already sorted, so a two-pointer linear merge is enough. Two details matter in this implementation: ties go left first, which preserves stability; and merging uses a buffer, which means the algorithm is not in-place.

#### Quick Coding: Merge Sort

```python
def merge_sort(nums):
    ...
```

<details>
<summary>Reference answer</summary>

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

Stability depends on `if nums[i] <= nums[j]`: ties must take the left side first.

</details>

### Quick Sort

Quicksort is about partition, not recursion. Choose a pivot, place `< pivot` on the left and `>= pivot` on the right, then recurse on both sides. Compared with Merge, it is in-place and usually has better locality. Compared with Heap, it usually has smaller constants, but a bad fixed pivot can still produce the `O(n^2)` worst case.

#### Quick Coding: Quick Sort

```python
def quick_sort(nums):
    ...
```

<details>
<summary>Reference answer</summary>

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

This is Lomuto partition: concise, in-place, and easy to explain, but still vulnerable to the `O(n^2)` worst case with a fixed last-element pivot.

</details>

### Heap Sort

Heap sort first turns the array into a max heap in-place, then repeatedly swaps the root maximum with the last element of the heap and restores the heap with `heapify`. Compared with Quick, it keeps the `O(n log n)` worst-case guarantee. Compared with Merge, it avoids `O(n)` extra memory, but it is not stable and usually has worse cache locality.

#### Quick Coding: Heap Sort

```python
def heap_sort(nums):
    ...
```

<details>
<summary>Reference answer</summary>

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

The build-heap phase is `O(n)`, not `O(n log n)`. The total cost comes from the later extract-and-heapify rounds.

</details>

## Module 4: How to Choose

### Why quicksort is often faster in practice than merge sort or heap sort

Quicksort usually benefits from better cache locality and smaller constant factors. Partition scans a contiguous range in-place, which is often enough to beat stronger worst-case guarantees.

### How to make quicksort more robust

Always choosing the first or last element can degenerate badly on sorted input. Common fixes are a random pivot, median-of-three, or three-way partitioning when duplicates are common.

### Why merge sort is a good fit for linked lists and external sorting

Merge sort only needs sequential access. That makes “split, sort, merge” natural on linked lists, and it makes streaming merges natural for large files on disk. Stability is also a direct advantage in multi-key sorting.

### When insertion sort is still the right answer

For very small arrays, nearly sorted arrays, or the base case inside a hybrid sort, Insertion is often the right choice. The code is short, the constant factor is small, and the number of shifts can stay low.

### When heap sort is the better answer

If the problem wants both worst-case `O(n log n)` and `O(1)` extra space, Heap is an obvious candidate. Its downside is the access pattern: it jumps around the array more than quicksort does.

### Why stability matters

A stable sort preserves the relative order of equal values. The standard multi-key pattern is: sort by the secondary key first, then do a stable sort by the primary key. The second sort keeps the secondary-key ordering inside equal primary keys.

## Final Checklist

1. Does the problem explicitly require stability?
2. Is `O(n)` extra space allowed?
3. Is the input nearly sorted, or very small?
4. Do you need a worst-case `O(n log n)` guarantee?
5. Is the data stored in an array, a linked list, or external files?

> Identify the active constraint first, then choose the sort that matches it.

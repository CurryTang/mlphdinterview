# Design Segment Tree

## Interview Goal

A segment tree solves a very specific class of problems: the array is updated repeatedly, and at the same time you must repeatedly query the aggregate value of a contiguous range. That aggregate can be `sum`, `min`, `max`, `gcd`, a count, maximum subarray information, and so on. As long as the answers for two adjacent ranges can be merged into the answer for the larger range, a segment tree is worth considering.

In an interview, you should be able to explain three things clearly:

- Each node represents a contiguous interval, and the root represents the entire array.
- During a range query, the target interval is decomposed into `O(log n)` non-overlapping tree nodes.
- During a point update, only one leaf changes, and then parent nodes are recomputed along the path from that leaf to the root.

## First Translate the Diagram Below into Plain Language

This diagram is not showing the usual `tails` array for LIS. It is showing a DP table:

```text
leaf[rank] = the length of the longest increasing subsequence ending at this value so far
```

For example, after coordinate compression of the value range:

```text
rank 0  1  2  3  4  5  6   7
value 2  3  5  7  9  10 18  101
```

If the current value being processed is `7`, its rank is `3`. Because the subsequence must be strictly increasing, the previous value must be smaller than `7`, so we can only look at ranks `0..2`, which correspond to values `2, 3, 5`. What the segment tree does is answer this quickly:

```text
Among all ending values smaller than 7, what is the maximum length?
```

If the answer is `2`, that means there is already an increasing subsequence of length 2 that can be extended by `7`, so:

```text
length ending at 7 = 2 + 1 = 3
```

Then update the leaf at rank `3` to `3`. Every frame in the full diagram repeats only these three steps:

1. Find the rank of the current number.
2. Query the maximum DP value among all smaller ranks.
3. Write back `maximum DP value + 1` to the current rank.

```segment-tree-demo
```

## What Exactly Does a Segment Tree Store?

Suppose the array is `arr[0..n-1]`. A segment-tree node does not store a single element. It stores the aggregated result for an interval.

For example, for range maximum:

- The leaf node `[i, i]` stores `arr[i]`.
- The parent node `[l, r]` stores `max(left_child, right_child)`.
- The root node `[0, n-1]` stores the maximum over the whole array.

So the essence of a segment tree is not "a tree," but a divide-and-conquer cache that can be recomputed quickly. It cuts the array into many standard intervals, reads cached answers directly during queries, and fixes only the affected cached values during updates.

## Array Implementation: Why `2 * n` Is Common

There are two common implementations.

The first is the recursive segment tree, which usually allocates `4 * n` space. If the node index is `idx`, its left child is `2 * idx` and its right child is `2 * idx + 1`.

The second is the iterative segment tree, which first places the leaves in the second half of the array:

```text
tree[n + i] = arr[i]
tree[i] = merge(tree[2*i], tree[2*i + 1])
```

If `n` is padded up to a power of 2, the whole tree becomes more regular:

```text
Leaves: tree[n], tree[n+1], ..., tree[2n-1]
Parent: i // 2
Left child: 2 * i
Right child: 2 * i + 1
```

The implementation you posted is exactly this iterative style, and the merge operation is `max`:

```python
class SegmentTree:
    def __init__(self, N):
        self.n = N
        while (self.n & (self.n - 1)) != 0:
            self.n += 1
        self.tree = [0] * (2 * self.n)
```

After `self.n` is padded to a power of 2, the original position `i` corresponds to the leaf `self.n + i`. The extra leaves default to `0`, because in this LIS problem the segment tree stores lengths, and the empty-state length is exactly `0`.

## Point Update

A point update affects only one path from a leaf to the root.

```python
def update(self, i, val):
    self.tree[self.n + i] = val
    j = (self.n + i) >> 1
    while j >= 1:
        self.tree[j] = max(self.tree[j << 1], self.tree[j << 1 | 1])
        j >>= 1
```

The logic of this code is:

1. First change leaf `i` to `val`.
2. Let `j` become its parent.
3. Recompute the parent each time using the maximum of its left and right children.
4. Continue updating all the way to the root node `1`.

The complexity is the height of the tree, which is `O(log n)`.

In an interview, you can add one more sentence: if the same position may be updated multiple times and you want to preserve the maximum value, you can write `self.tree[self.n + i] = max(self.tree[self.n + i], val)`. In this left-to-right LIS process, the best length at the same rank never decreases, so direct assignment also works.

## Range Query

Iterative range queries are the easiest part to get visually confused by, because they do not recursively test the three overlap cases. Instead, they shrink two pointers upward from the leaf layer.

```python
def query(self, l, r):
    if l > r:
        return 0
    res = float('-inf')
    l += self.n
    r += self.n + 1
    while l < r:
        if l & 1:
            res = max(res, self.tree[l])
            l += 1
        if r & 1:
            r -= 1
            res = max(res, self.tree[r])
        l >>= 1
        r >>= 1
    return res
```

Key points:

- The query interval is converted to the half-open interval `[l, r)`, so the original right endpoint must be written as `r += self.n + 1`.
- If `l` is a right child, it cannot be merged upward together with its left sibling, or else extra elements outside the query range on the left would be included. So first put `tree[l]` into the answer, then do `l += 1`.
- If `r` is in the position of a right child on the right boundary, first do `r -= 1` and include that valid node in the answer.
- After processing the boundary nodes in each round, do `l >>= 1` and `r >>= 1` to move to the parent layer.

At most two boundary nodes are taken at each level, so this is also `O(log n)`.

## The Identity Element Must Match the Merge Operation

There is an easy mistake in segment-tree queries: what should a non-contributing interval return?

- Range sum: return `0`.
- Range maximum: if the array may contain negative numbers, the strict identity should be `-inf`.
- Range minimum: return `+inf`.
- gcd: return `0`, because `gcd(0, x) = x`.

In this LIS problem, the segment tree stores "length," and all valid lengths are at least `0`, so returning `0` for an empty query is reasonable.

## Example Problem: Longest Increasing Subsequence

Problem: given `nums`, find the length of the longest strictly increasing subsequence.

The traditional `O(n log n)` patience-sorting solution is convenient when you only need the length. The segment-tree solution looks more like dynamic programming and is easier to extend to problems like "LIS with constraints," "value-range queries," or "DP with updates."

Define:

```text
dp[x] = the length of the longest increasing subsequence ending with value x
```

When processing the current number `num`, it can only extend values smaller than it:

```text
cur = max(dp[value] for value < num) + 1
dp[num] = cur
answer = max(answer, cur)
```

If you allocate an array directly by real values, the values may be large or even negative. So first perform coordinate compression and map values to ranks:

```python
def compress(arr):
    sortedArr = sorted(set(arr))
    order = []
    for num in arr:
        order.append(bisect_left(sortedArr, num))
    return order
```

After compression, a smaller `rank` means a smaller original value. Strictly increasing means the current rank can query only `[0, rank - 1]`, and cannot include itself.

Full idea:

```python
from bisect import bisect_left
from typing import List

class SegmentTree:
    def __init__(self, N):
        self.n = N
        while (self.n & (self.n - 1)) != 0:
            self.n += 1
        self.tree = [0] * (2 * self.n)

    def update(self, i, val):
        self.tree[self.n + i] = val
        j = (self.n + i) >> 1
        while j >= 1:
            self.tree[j] = max(self.tree[j << 1], self.tree[j << 1 | 1])
            j >>= 1

    def query(self, l, r):
        if l > r:
            return 0
        res = float('-inf')
        l += self.n
        r += self.n + 1
        while l < r:
            if l & 1:
                res = max(res, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = max(res, self.tree[r])
            l >>= 1
            r >>= 1
        return res

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        def compress(arr):
            sortedArr = sorted(set(arr))
            order = []
            for num in arr:
                order.append(bisect_left(sortedArr, num))
            return order

        nums = compress(nums)
        n = len(nums)
        segTree = SegmentTree(n)

        LIS = 0
        for num in nums:
            curLIS = segTree.query(0, num - 1) + 1
            segTree.update(num, curLIS)
            LIS = max(LIS, curLIS)
        return LIS
```

Using `[10, 9, 2, 5, 3, 7, 101, 18]` as an example:

```text
sorted values = [2, 3, 5, 7, 9, 10, 18, 101]
compressed    = [5, 4, 0, 2, 1, 3, 7, 6]
```

When processing `7`, its rank is `3`, so we query `[0, 2]`, which means looking only at the best ending lengths for values `{2, 3, 5}`. The maximum at that point is `2`, so `curLIS = 3`, representing a possible sequence such as `[2, 5, 7]` or `[2, 3, 7]`.

When processing `18`, the rank is `6`, so we query `[0, 5]`. The maximum is `3`, so `curLIS = 4`. The final answer is `4`.

## Why Not Query `[0, rank]`

LIS is strictly increasing, so equal values cannot be chained together. If you query `[0, rank]`, the same value may contribute to itself, and the result becomes non-strictly increasing.

For example, for `[2, 2, 2]`:

- The correct answer is `1`.
- Each step should query values smaller than `2`, which is an empty interval.
- If the query includes the current rank, the length from a previous `2` may be extended by a later `2`, which breaks the intended meaning.

## Complexity

Let the array length be `n`, and the number of distinct values be `m`.

- Coordinate compression: `O(n log n)`.
- One range-maximum query and one point update per number: `O(n log m)`.
- Segment-tree space: `O(m)`, and after padding to a power of 2 in the implementation, it is still linear space.

## When Is Lazy Propagation Needed?

This problem uses only point updates, so lazy propagation is not needed.

Lazy propagation is used for range updates, for example:

- Add `x` to every element in `[l, r]`.
- Assign every element in `[l, r]` to `x`.
- Perform frequent range updates together with range queries.

If every range update recursively reaches all leaves, the worst case degrades to `O(n)`. The idea of lazy propagation is: when a node interval is fully covered by an update, first update that node's aggregate value and record a lazy tag; only when its children are visited later do you push the tag downward.

## Common Pitfalls

- Forgetting coordinate compression and using the raw value directly as the array index.
- Querying `[0, rank]` in a strictly increasing problem and accidentally including equal elements.
- Using a no-overlap return value for a max segment tree that does not match the data range.
- Writing `r += self.n + 1` incorrectly, so the right endpoint is not included.
- Updating only the leaf after a point update and failing to recompute all the way to the root.
- Missing the correct timing for pushing down lazy tags in a range-update problem.

## Interview Answer Template

<details class="solution">
<summary>Expand Template</summary>

A segment tree splits the array into a binary interval tree. Each node maintains the aggregate value of an interval, such as the maximum. When querying `[l, r]`, we decompose it into several already-cached node intervals and merge their answers; during a point update, only one path from the leaf to the root is affected, so the update complexity is `O(log n)`.

For the LIS problem, I coordinate-compress the values into ranks. The leaf at rank `rank` in the segment tree maintains "the length of the longest increasing subsequence ending at this rank." When processing the current `rank`, I first query the maximum over all smaller ranks, which is `query(0, rank - 1)`, then add one to get the current length, and finally update the current rank. This makes each element `O(log n)`, for a total complexity of `O(n log n)`.

</details>

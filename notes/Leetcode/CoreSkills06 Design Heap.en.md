# Heap and Priority Queue

## Prerequisite: Design Heap

### Interview Goal

Implement a heap, covering the array representation of a complete binary tree, bubbling up, sifting down, and priority-queue operations.

### Core Design

- A min-heap requires every parent's value to be no greater than its children's.
- Array index relationships: `parent=(i-1)//2`, `left=2*i+1`, `right=2*i+2`.
- Insertion: append to the end, then bubble up.
- Removing the top: move the last element to the root, then sift down.

### Complexity

- peek: `O(1)`
- push/pop: `O(log n)`
- heapify: `O(n)`

### Common Pitfalls

- Sifting down without picking the smaller of the two children.
- Forgetting to return the original top after popping.
- Not handling operations on an empty heap.

### Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

The array represents a complete binary tree. After insertion, compare upward against the parent; after removing the root, move the last element to the root and swap downward against the smaller child.

```text
push(x):
  data.append(x)
  i = last index
  while i > 0 and data[i] < data[parent(i)]:
    swap(i, parent(i))
    i = parent(i)

pop():
  ans = data[0]
  data[0] = data.pop()
  heapify_down(0)
  return ans
```

`heapify_down` picks the smaller of the two children each time, swapping only when that child is smaller than the current node.

</details>

The heap ADT above is the prerequisite for this chapter. The 7 problems below no longer care how a heap is implemented; they care about one thing: when a problem only needs the extreme value or the top `k` elements of a set, not a full sort, can a heap cut the complexity or the amount of code needed?

## Learning Order

Problems are drawn from the Heap / Priority Queue module of [NeetCode 150](https://neetcode.io/practice/practice/neetcode150).

| Order | Problem | What to Master |
|---:|---|---|
| 1 | [703. Kth Largest Element In a Stream](https://neetcode.io/problems/kth-largest-element-in-a-stream/question?list=neetcode150) | A fixed-size heap whose top is the answer |
| 2 | [1046. Last Stone Weight](https://neetcode.io/problems/last-stone-weight/question?list=neetcode150) | Negating values to simulate a max-heap |
| 3 | [973. K Closest Points to Origin](https://neetcode.io/problems/k-closest-points-to-origin/question?list=neetcode150) | The same fixed-size heap, with distance as the comparison key |
| 4 | [215. Kth Largest Element In an Array](https://neetcode.io/problems/kth-largest-element-in-an-array/question?list=neetcode150) | The same template, plus quickselect as a follow-up |
| 5 | [621. Task Scheduler](https://neetcode.io/problems/task-scheduler/question?list=neetcode150) | Frequency counting plus a formula, easier to remember than heap simulation |
| 6 | [355. Design Twitter](https://neetcode.io/problems/design-twitter/question?list=neetcode150) | Merging bounded candidate sets to get the k most recent |
| 7 | [295. Find Median From Data Stream](https://neetcode.io/problems/find-median-from-a-data-stream/question?list=neetcode150) | Two heaps maintaining a running median |

## Module 1: The Mental Model for Heaps

A heap is a complete binary tree stored as an array: the parent of index `i` is `(i-1)//2`, and its two children are `2*i+1` and `2*i+2`. This array representation is what makes `push`/`pop` both `O(log n)`: a new or replaced element only needs to bubble up or sift down once, along the height of the tree.

`heapify` (turning an entire array into a valid heap in place) is `O(n)`, not `O(n log n)`. Most nodes sit near the bottom, so they sift down only a short distance; only a few nodes near the root travel the full height, and averaged across all nodes the total work is linear. This complexity is asked about often enough that the result is worth memorizing directly.

Python's `heapq` implements a min-heap only. When max-heap semantics are needed, negate the values on the way in, or store tuples like `(-key, ...)`, and negate again on the way out.

| Need | How |
|---|---|
| Turn a list into a heap in place | `heapq.heapify(data)` |
| Insert | `heapq.heappush(data, x)` |
| Pop the minimum | `heapq.heappop(data)` |
| Peek without removing | `data[0]` |
| Max-heap semantics | Store `-x`, negate again on read |
| Get the top `k` directly | `heapq.nlargest(k, iterable)` / `heapq.nsmallest(k, iterable)` |

## Module 2: Core Solving Techniques

### 1. A Fixed-Size Bounded Heap

When a problem asks for "the top `k`" or "the `k`th largest," the full dataset never needs to be sorted. Maintain a min-heap whose size is always `k`: compare each new element against the top, and only replace the top if the new element is larger. The top is always the weakest of the `k` current candidates, which is exactly the boundary of the answer.

```python
def top_k(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap
```

This runs in `O(n log k)` instead of the `O(n log n)` a full sort would take. The difference is large whenever `k` is much smaller than `n`.

Used by: Kth Largest Element In a Stream, K Closest Points to Origin, Kth Largest Element In an Array.

### 2. Negating Values to Simulate a Max-Heap

`heapq` only provides a min-heap. When a problem repeatedly needs the maximum, store negated values; the top of the heap is the smallest negated value, corresponding to the largest original value, and it needs to be negated again on the way out.

```python
max_heap = []
heapq.heappush(max_heap, -x)
largest = -heapq.heappop(max_heap)
```

Used by: Last Stone Weight.

### 3. Merging Bounded Candidate Sets

When the k most recent items need to be found across multiple sources, there is no need to merge every source's full history. Take only the k most recent items from each source as candidates; the total candidate count is "number of sources times k," and the true top k can then be picked out of those candidates with a heap or `heapq.nlargest`. This is still a fixed-size heap underneath, with the candidate set already pruned by one round first.

Used by: Design Twitter.

### 4. Two Heaps Maintaining a Running Median

A median only depends on the one or two values in the middle, not the full sorted sequence. Keep a max-heap `small` holding the lower half and a min-heap `large` holding the upper half, with their sizes never differing by more than one. On insertion, push into one heap first, then transfer its top to the other heap once, which guarantees `small`'s maximum never exceeds `large`'s minimum.

```python
def add_num(num, small, large):
    heapq.heappush(small, -num)
    heapq.heappush(large, -heapq.heappop(small))
    if len(large) > len(small):
        heapq.heappush(small, -heapq.heappop(large))
```

When the two heaps are equal in size, the median is the average of both tops; when `small` has one more element, the median is simply `small`'s top.

Used by: Find Median From Data Stream.

### 5. Check the Frequency Count First; a Heap Is Not Always Optimal

Being in the Heap category does not mean every problem must use a heap. Task Scheduler looks like it is asking for "greedily pick the highest-frequency task each round," which a max-heap can simulate correctly. But counting frequencies once and plugging the result into a formula is shorter and easier to get right under interview pressure. Recognizing that a problem does not actually need a heap is itself part of the judgment this category is meant to train.

## Module 3: Mapping the Seven Problems

### 1. Kth Largest Element In a Stream

A direct application of the fixed-size heap. The constructor loads the initial array into a min-heap of size `k`; `add` inserts the new value and pops the top if the heap grows past `k`, so the top is always the current `k`th largest value.

| Item | Value |
|---|---|
| Composed technique | Fixed-size bounded heap |
| Invariant | Heap size stays `k`; the top is the `k`th largest |
| Time / Space | Init `O(n log k)`, each `add` is `O(log k)`, space `O(k)` |

#### Quick Coding: Kth Largest Element In a Stream

```python
class KthLargest:
    def __init__(self, k, nums):
        ...

    def add(self, val):
        ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq
from typing import List


class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]
```

</details>

### 2. Last Stone Weight

A direct application of max-heap-by-negation. Each round removes the two heaviest stones; if their weights differ, the difference goes back into the heap, until at most one stone remains.

| Item | Value |
|---|---|
| Composed technique | Negating values to simulate a max-heap |
| Invariant | The negation of the top is always the current heaviest stone |
| Time / Space | `O(n log n) / O(n)` |

#### Quick Coding: Last Stone Weight

```python
def lastStoneWeight(stones):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq
from typing import List


class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = [-stone for stone in stones]
        heapq.heapify(heap)

        while len(heap) > 1:
            first = -heapq.heappop(heap)
            second = -heapq.heappop(heap)
            if first != second:
                heapq.heappush(heap, -(first - second))

        return -heap[0] if heap else 0
```

</details>

### 3. K Closest Points to Origin

The comparison key changes from a plain value to the squared distance from the origin; the template is unchanged: maintain a max-heap of size `k`, where the top is the farthest point among current candidates, and replace it whenever a closer point arrives. Squaring preserves the comparison order, so using squared distance avoids an unnecessary square root.

| Item | Value |
|---|---|
| Composed technique | Fixed-size bounded heap, keyed by squared distance |
| Invariant | The heap always holds the `k` closest points seen so far |
| Time / Space | `O(n log k) / O(k)` |

#### Quick Coding: K Closest Points to Origin

```python
def kClosest(points, k):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq
from typing import List


class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap = []
        for x, y in points:
            dist = -(x * x + y * y)
            if len(heap) < k:
                heapq.heappush(heap, (dist, x, y))
            elif dist > heap[0][0]:
                heapq.heapreplace(heap, (dist, x, y))
        return [[x, y] for _, x, y in heap]
```

`heapq.heapreplace` pops the top and pushes the new value in a single call, saving one comparison compared to calling `heappop`/`heappush` separately.

</details>

### 4. Kth Largest Element In an Array

The same template as problem 1, except the input is an array instead of a stream: maintain a min-heap of size `k`, and the final top is the `k`th largest element.

| Item | Value |
|---|---|
| Composed technique | Fixed-size bounded heap |
| Invariant | Heap size stays `k`; the top is the `k`th largest |
| Time / Space | `O(n log k) / O(k)` |

#### Quick Coding: Kth Largest Element In an Array

```python
def findKthLargest(nums, k):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq
from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]
```

`heapq.nlargest` maintains exactly a size-`k` heap internally, so this one line is equivalent to writing the bounded-heap template by hand.

Interviewers often follow up by asking for an average-`O(n)` solution: quickselect, which reuses partition and only recurses into one side each round.

```python
import random
from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        target = len(nums) - k

        def partition(left: int, right: int) -> int:
            pivot_index = random.randint(left, right)
            nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
            store = left
            for i in range(left, right):
                if nums[i] < nums[right]:
                    nums[i], nums[store] = nums[store], nums[i]
                    store += 1
            nums[store], nums[right] = nums[right], nums[store]
            return store

        left, right = 0, len(nums) - 1
        while True:
            pivot_index = partition(left, right)
            if pivot_index == target:
                return nums[pivot_index]
            if pivot_index < target:
                left = pivot_index + 1
            else:
                right = pivot_index - 1
```

A random pivot keeps the worst case unlikely to show up within the scope of an interview, and the average time is `O(n)`. The heap version is shorter and easier to get right under pressure; quickselect is a follow-up answer worth knowing, not the version to lead with.

</details>

### 5. Task Scheduler

Count how often each task occurs. Let `max_freq` be the highest frequency and `max_count` be the number of task types that reach it. Lay out the most frequent tasks first, filling the cooldown gaps between them with other tasks or idle slots; in the ideal case the total time is `(max_freq - 1) * (n + 1) + max_count`: there are `max_freq - 1` gaps between occurrences of the most frequent tasks, each gap has length `n + 1` (including one execution slot), and the final round of `max_count` most-frequent tasks is appended at the end. If there are enough distinct task types to fill every gap without idling, the actual time is just the total number of tasks; the answer is the larger of the two.

| Item | Value |
|---|---|
| Composed technique | Frequency counting plus a formula (easier to remember than heap simulation) |
| Key state | `max_freq`, `max_count` |
| Time / Space | `O(n) / O(1)` (the number of task types is bounded by 26) |

#### Quick Coding: Task Scheduler

```python
def leastInterval(tasks, n):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import Counter
from typing import List


class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts = Counter(tasks)
        max_freq = max(counts.values())
        max_count = sum(1 for freq in counts.values() if freq == max_freq)

        idle_slots = (max_freq - 1) * (n + 1) + max_count
        return max(idle_slots, len(tasks))
```

With `n=2` and tasks `AAABBB`: `max_freq=3`, `max_count=2` (A and B tie), and the formula gives `(3-1)*(2+1)+2=8`, matching the schedule `A B _ A B _ A B`.

</details>

Simulating "pick the current highest-frequency task each round" with a max-heap also produces the correct answer, and it is the more general version of this technique: under more complex constraints (for example, if each task type had a different cooldown), the formula would no longer hold, but heap simulation would still work.

```python
import heapq
from collections import Counter, deque
from typing import List


class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        heap = [-freq for freq in Counter(tasks).values()]
        heapq.heapify(heap)

        time = 0
        cooldown = deque()  # (available_time, remaining_count)
        while heap or cooldown:
            time += 1
            if heap:
                remaining = heapq.heappop(heap) + 1
                if remaining < 0:
                    cooldown.append((time + n, remaining))
            if cooldown and cooldown[0][0] == time:
                heapq.heappush(heap, cooldown.popleft()[1])

        return time
```

### 6. Design Twitter

Each user's tweets are stored as a list in posting order, with a global, only-increasing counter serving as a timestamp. `getNewsFeed` only needs the most recent 10 tweets from the user and each followee: the candidate count is at most "number of followees plus one" times 10, and `heapq.nlargest` picks the 10 most recent out of those candidates, with no need to scan anyone's full history.

| Item | Value |
|---|---|
| Composed technique | Bounded candidate sets plus a fixed-size heap |
| Key state | A global increasing timestamp, plus each user's tweet list |
| Time / Space | `getNewsFeed` is `O(f log 10)`, where `f` is the number of followees; space is `O(total tweets)` |

#### Quick Coding: Design Twitter

```python
class Twitter:
    def __init__(self):
        ...

    def postTweet(self, userId, tweetId):
        ...

    def getNewsFeed(self, userId):
        ...

    def follow(self, followerId, followeeId):
        ...

    def unfollow(self, followerId, followeeId):
        ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq
from collections import defaultdict
from typing import List


class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)  # userId -> [(time, tweetId), ...]
        self.following = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        candidates = []
        for uid in self.following[userId] | {userId}:
            candidates.extend(self.tweets[uid][-10:])
        top = heapq.nlargest(10, candidates)
        return [tweetId for _, tweetId in top]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.following[followerId].discard(followeeId)
```

`self.time` increments with every posted tweet, so timestamps naturally increase in posting order, and taking the 10 largest timestamps with `heapq.nlargest` is exactly the 10 most recent tweets.

</details>

### 7. Find Median From Data Stream

A direct application of the two-heap template. Every `addNum` strictly follows the order "push into `small` first, transfer its top to `large`, then transfer back if needed," which guarantees that `small`'s maximum never exceeds `large`'s minimum at any point, and that the two heaps' sizes never differ by more than one.

| Item | Value |
|---|---|
| Composed technique | Two heaps maintaining a running median |
| Invariant | `len(small) - len(large)` is always `0` or `1`; `small`'s maximum `<=` `large`'s minimum |
| Time / Space | `addNum` is `O(log n)`, `findMedian` is `O(1)`; space is `O(n)` |

The demo below inserts a sequence of numbers one at a time, showing how the two heaps stay balanced after each step.

```median-two-heaps-demo
```

#### Quick Coding: Find Median From Data Stream

```python
class MedianFinder:
    def __init__(self):
        ...

    def addNum(self, num):
        ...

    def findMedian(self):
        ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq


class MedianFinder:
    def __init__(self):
        self.small = []  # max-heap (negated), holds the lower half
        self.large = []  # min-heap, holds the upper half

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

The new value is always pushed into `small` unconditionally first, then `small`'s maximum is transferred to `large`; this step guarantees every value remaining in `small` is no greater than the one just transferred. If that transfer left `large` with one more element than `small`, transfer `large`'s minimum back so the size difference returns to at most one. This order avoids having to decide up front which heap the new value belongs in, which is the easiest part of this template to remember.

</details>

## Module 4: Final Checks Before an Interview

1. Does the problem need "extreme values" or "a full sort"? If only extremes or the top `k` are needed, a heap is usually the better fit than sorting.
2. Is a max-heap or min-heap needed? `heapq` only provides a min-heap; max-heap semantics require negation.
3. Should the heap size be fixed at `k`? A fixed size drops the complexity from `O(n log n)` to `O(n log k)`.
4. Does the data arrive all at once, or does it keep coming in (a stream, or multiple sources)? Ongoing arrival usually signals that one or more heaps should be maintained incrementally, rather than re-sorting every time.
5. Does this problem actually need a heap? Counting frequencies or applying a formula is sometimes shorter and less error-prone than heap simulation.

One sentence to keep:

> A heap solves problems that repeatedly need an extreme value without needing full ordering; identifying which extreme value the problem wants, and how many, fixes the template to use.

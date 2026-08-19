# Heap 与 Priority Queue

## 前置：Design Heap

### 面试目标

实现堆，掌握数组表示完全二叉树、上浮、下沉和优先队列操作。

### 核心设计

- 最小堆满足父节点值不大于子节点。
- 数组下标关系：`parent=(i-1)//2`，`left=2*i+1`，`right=2*i+2`。
- 插入：放到末尾后 bubble up。
- 删除堆顶：末尾元素换到根，再 bubble down。

### 复杂度

- peek：`O(1)`
- push/pop：`O(log n)`
- heapify：`O(n)`

### 常见坑

- 下沉时没有选择更小的子节点。
- pop 后忘记返回原堆顶。
- 空堆操作没有处理。

### 参考解法

<details class="solution">
<summary>展开解法</summary>

数组表示完全二叉树。插入后向上和父节点比较；删除根后把最后一个元素放到根，再向下和更小的子节点交换。

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

`heapify_down` 每次选择左右孩子中更小的那个，只有它比当前节点更小时才交换。

</details>

上面的堆 ADT 是这一章的前置。后面的 7 道题不再关心怎么实现堆本身，而是关心一件事：题目只需要集合中的最值或前 `k` 个，不需要完整排序时，堆能不能把复杂度或代码量降下来。

## 学习顺序

题目来自 [NeetCode 150](https://neetcode.io/practice/practice/neetcode150) 的 Heap / Priority Queue 模块。

| 顺序 | 原题 | 要掌握的内容 |
|---:|---|---|
| 1 | [703. Kth Largest Element In a Stream](https://neetcode.io/problems/kth-largest-element-in-a-stream/question?list=neetcode150) | 固定大小的堆，堆顶就是答案 |
| 2 | [1046. Last Stone Weight](https://neetcode.io/problems/last-stone-weight/question?list=neetcode150) | 取负数模拟最大堆 |
| 3 | [973. K Closest Points to Origin](https://neetcode.io/problems/k-closest-points-to-origin/question?list=neetcode150) | 固定大小的堆，比较键换成距离 |
| 4 | [215. Kth Largest Element In an Array](https://neetcode.io/problems/kth-largest-element-in-an-array/question?list=neetcode150) | 同一个模板，外加快速选择作为进阶写法 |
| 5 | [621. Task Scheduler](https://neetcode.io/problems/task-scheduler/question?list=neetcode150) | 频率统计 + 公式，比堆模拟更好记 |
| 6 | [355. Design Twitter](https://neetcode.io/problems/design-twitter/question?list=neetcode150) | 用有界候选集合做多路合并取最近 k 条 |
| 7 | [295. Find Median From Data Stream](https://neetcode.io/problems/find-median-from-a-data-stream/question?list=neetcode150) | 双堆维护中位数 |

## 模块一：堆的心智模型

堆是一棵完全二叉树，用数组存储：下标 `i` 的父节点是 `(i-1)//2`，两个子节点是 `2*i+1` 和 `2*i+2`。这个数组表示保证了 `push`/`pop` 都是 `O(log n)`：新元素或替换元素只需要沿着树高上浮或下沉一次。

`heapify`（把整个数组原地变成合法堆）是 `O(n)`，不是 `O(n log n)`。原因是大多数节点靠近底部，下沉的距离很短；只有少数靠近根的节点需要走满树高，均摊下来是线性的。这个复杂度经常被问到，值得直接记住结论。

Python 的 `heapq` 只实现最小堆。需要最大堆语义时，把存入的值取负，或者存 `(-key, ...)` 这样的元组，读出来再取负还原。

| 需求 | 写法 |
|---|---|
| 把列表原地变成堆 | `heapq.heapify(data)` |
| 插入 | `heapq.heappush(data, x)` |
| 取出最小值 | `heapq.heappop(data)` |
| 只看不取 | `data[0]` |
| 最大堆语义 | 存入 `-x`，取出后再取负 |
| 直接要前 `k` 大/小 | `heapq.nlargest(k, iterable)` / `heapq.nsmallest(k, iterable)` |

## 模块二：核心解题技巧

### 1. 固定大小的有界堆

题目要"前 `k` 个"或"第 `k` 大"时，不需要对全部数据排序。维护一个大小恒为 `k` 的最小堆：新元素来了就和堆顶比较，只有比堆顶更大才替换。堆顶始终是这 `k` 个候选里最弱的一个，也就是答案的边界。

```python
def top_k(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap
```

这样只需要 `O(n log k)`，而不是排序全部数据的 `O(n log n)`。`k` 远小于 `n` 时差别很大。

使用题目：Kth Largest Element In a Stream、K Closest Points to Origin、Kth Largest Element In an Array。

### 2. 取负数模拟最大堆

`heapq` 只有最小堆。题目需要反复取最大值时，把值取负存入，堆顶的最小负数对应原始的最大值，取出后记得再取负一次。

```python
max_heap = []
heapq.heappush(max_heap, -x)
largest = -heapq.heappop(max_heap)
```

使用题目：Last Stone Weight。

### 3. 用有界候选集合做多路合并

需要从多个来源里找"最近的 `k` 个"时，不必合并全部历史数据。每个来源只取自己最近的 `k` 个作为候选，候选总数是"来源数 `×` k"，再从候选里用堆或 `heapq.nlargest` 选出真正的前 `k` 个。这本质上还是固定大小的堆，只是候选集合已经先做过一轮裁剪。

使用题目：Design Twitter。

### 4. 双堆维护中位数

中位数只和中间的一两个值有关，不需要维护整个有序序列。用一个最大堆 `small` 存较小的一半，一个最小堆 `large` 存较大的一半，让两堆大小最多相差一。插入时先放进某一堆，再把它的堆顶转移给另一堆一次，保证 `small` 的最大值不超过 `large` 的最小值。

```python
def add_num(num, small, large):
    heapq.heappush(small, -num)
    heapq.heappush(large, -heapq.heappop(small))
    if len(large) > len(small):
        heapq.heappush(small, -heapq.heappop(large))
```

两堆大小相等时中位数是两个堆顶的平均值；`small` 多一个时中位数就是 `small` 的堆顶。

使用题目：Find Median From Data Stream。

### 5. 先看频率统计，堆不一定是最优解

Heap 分类下的题目不代表每道题都必须用堆。Task Scheduler 表面在问"每轮贪心选频率最高的任务"，这确实可以用最大堆模拟；但只统计一次频率、代入一个公式，比堆模拟更短也更容易在白板上写对。识别"这道题其实不需要堆"，本身也是这一类题目要练的判断力。

## 模块三：7 道题目的映射

### 1. Kth Largest Element In a Stream

这道题直接套用固定大小的堆。构造函数把初始数组灌进一个大小为 `k` 的最小堆；`add` 每次插入新值，超出大小就弹出堆顶，堆顶始终是当前第 `k` 大的值。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 固定大小的有界堆 |
| 关键不变量 | 堆大小恒为 `k`，堆顶是第 `k` 大 |
| 时间 / 空间 | 初始化 `O(n log k)`，每次 `add` 是 `O(log k)`，空间 `O(k)` |

#### Quick Coding：Kth Largest Element In a Stream

```python
class KthLargest:
    def __init__(self, k, nums):
        ...

    def add(self, val):
        ...
```

<details>
<summary>参考答案</summary>

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

最大堆语义的直接应用。每轮取出最大的两块石头，差值非零就放回堆里，直到堆里最多剩一块。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 取负数模拟最大堆 |
| 关键不变量 | 堆顶的相反数始终是当前最大的石头 |
| 时间 / 空间 | `O(n log n) / O(n)` |

#### Quick Coding：Last Stone Weight

```python
def lastStoneWeight(stones):
    ...
```

<details>
<summary>参考答案</summary>

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

比较键从数值本身换成到原点的距离，模板不变：维护大小为 `k` 的最大堆，堆顶是当前候选里距离最远的点，新点更近就替换掉它。开平方不影响大小比较，直接用距离平方即可，省掉一次开方运算。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 固定大小的有界堆，比较键是距离平方 |
| 关键不变量 | 堆里始终是目前见过的 `k` 个最近点 |
| 时间 / 空间 | `O(n log k) / O(k)` |

#### Quick Coding：K Closest Points to Origin

```python
def kClosest(points, k):
    ...
```

<details>
<summary>参考答案</summary>

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

`heapq.heapreplace` 一次完成"弹出堆顶再插入新值"，比分开调用 `heappop`/`heappush` 少一次比较。

</details>

### 4. Kth Largest Element In an Array

和第 1 题同一个模板，只是输入是数组而不是数据流：维护大小为 `k` 的最小堆，最终堆顶就是第 `k` 大的元素。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 固定大小的有界堆 |
| 关键不变量 | 堆大小恒为 `k`，堆顶是第 `k` 大 |
| 时间 / 空间 | `O(n log k) / O(k)` |

#### Quick Coding：Kth Largest Element In an Array

```python
def findKthLargest(nums, k):
    ...
```

<details>
<summary>参考答案</summary>

```python
import heapq
from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k, nums)[-1]
```

`heapq.nlargest` 内部维护的正是一个大小为 `k` 的堆，这一行等价于手写的有界堆模板。

面试官经常会追问平均 `O(n)` 的写法：快速选择复用 partition，每轮只递归一侧；partition 本身的原地划分模板见 [[CoreSkills10 Insertion Sort#Quick Sort|Quick Sort]]。

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

随机 pivot 让最坏情况在面试考察范围内不太可能出现，平均时间是 `O(n)`。堆的写法更短、更容易在压力下写对，快速选择是知道就能加分的进阶答案，不是必须先写的版本。

下面的演示把 partition 的每一步拆开：pivot 怎么选、scan 和 store 两个指针怎么移动、pivot 最终换到哪个位置，以及每一轮为什么只需要继续递归一侧。

```quickselect-partition-demo
```

</details>

### 5. Task Scheduler

先统计每种任务出现的次数。设最高频率为 `max_freq`，达到这个频率的任务种类数为 `max_count`。把最高频任务排开，中间的冷却间隔用其他任务或空闲填充，理想情况下总时间是 `(max_freq - 1) * (n + 1) + max_count`：最高频任务之间有 `max_freq - 1`个间隔，每个间隔长度 `n + 1`（包含一个执行槽），最后再补上最后一轮的 `max_count` 个最高频任务。如果任务种类够多，足以把所有间隔填满而不需要空闲，实际时间就是任务总数本身；两者取较大值。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 频率统计 + 公式（比堆模拟更好记） |
| 关键状态 | `max_freq`、`max_count` |
| 时间 / 空间 | `O(n) / O(1)`（任务种类数是常数 26） |

#### Quick Coding：Task Scheduler

```python
def leastInterval(tasks, n):
    ...
```

<details>
<summary>参考答案</summary>

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

`n=2`、任务是 `AAABBB` 时：`max_freq=3`，`max_count=2`（A 和 B 并列），公式给出 `(3-1)*(2+1)+2=8`，对应排列 `A B _ A B _ A B`。

</details>

用最大堆模拟"每轮取当前频率最高的任务"也能得到正确答案，是这一类问题更通用的写法：面对更复杂的约束（比如每种任务恢复时间不同）时公式不一定还成立，堆模拟仍然适用。

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

每个用户的推文按时间顺序存成一个列表，全局用一个只增不减的计数器充当时间戳。`getNewsFeed` 只需要看自己和关注对象各自最近的 10 条：候选数量最多是"关注数 `+1`"乘以 10，用 `heapq.nlargest` 从候选里取最近的 10 条即可，不需要扫描每个人的完整历史。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 有界候选集合 + 固定大小的堆 |
| 关键状态 | 全局递增时间戳，每个用户的推文列表 |
| 时间 / 空间 | `getNewsFeed` 是 `O(f log 10)`，`f` 是关注数；空间 `O(推文总数)` |

#### Quick Coding：Design Twitter

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
<summary>参考答案</summary>

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

`self.time` 每发一条推文加一，时间戳天然按发布顺序递增，`heapq.nlargest` 按时间戳取最大的 10 个就是最近的 10 条。

</details>

#### 另一种写法：K 路归并版本

`getNewsFeed` 也可以用 K 路归并实现：把每个关注对象最近 10 条推文的迭代器交给 `heapq.merge`，惰性地归并出前 10 条。

```python
import heapq
import itertools
from collections import defaultdict
from typing import List


class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)
        self.follow_map = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1

    def getNewsFeed(self, userId: int) -> List[int]:
        followed = self.follow_map[userId] | {userId}
        user_tweets = [reversed(self.tweets[u][-10:]) for u in followed if self.tweets[u]]
        merged = heapq.merge(*user_tweets, key=lambda x: x[0], reverse=True)
        return [tweetId for _, tweetId in itertools.islice(merged, 10)]

    def follow(self, followerId: int, followeeId: int) -> None:
        self.follow_map[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.follow_map[followerId].discard(followeeId)
```

这版把属性名从 `following` 改成了 `follow_map`，为的是避免和 `follow` 方法同名；如果直接叫 `self.follow`，实例属性会覆盖同名方法。

两种写法看起来是"惰性 K 路归并"和"全量候选加堆"的对比，但实测结果和直觉相反：在关注数（`F`）明显大于 10 的场景下，K 路归并版本更快，差距随 `F` 增大而扩大。原因在于两者真正触达的数据量不同：

- `heapq.nlargest(10, candidates)` 必须完整遍历 `candidates` 这个长度为 `10F` 的列表，对每个元素都执行一次比较，是一次完整的线性扫描。
- `heapq.merge` 只需要为每一路建一个大小为 `F` 的初始堆（每一路贡献当前最新的一条），之后每弹出一个结果就从对应的那一路补一个新元素，只要 10 次弹出。整个过程只触达大约 `F + 10 log F`个元素，不需要看到全部 `10F` 条候选。
- `F` 越大，`10F`（nlargest 要扫描的元素数）和 `F + 10 log F`（merge 实际触达的元素数）之间的差距越大，K 路归并的优势也越明显。

`key=lambda x: x[0]` 这一步其实是多余的：`(time, tweetId)` 这样的 tuple 本身就按时间优先比较，去掉 `key` 能再省一点调用开销，但不影响上面的结论。

判断标准是候选流的路数（这里是关注数）和每一路的长度的关系：路数远小于总候选数（也就是每一路都比较长）时，一次性物化再用 `heapq.nlargest`/`heapq.nsmallest` 更划算；路数本身较大、每一路又很短（就像这道题，每一路最多 10 条）时，K 路归并能避免扫描全部候选，通常更快。这也是外部归并排序、合并多个有序日志或数据库分页结果时优先选择 `heapq.merge` 的原因。

### 7. Find Median From Data Stream

双堆模板的直接应用。`addNum` 每次都严格按"先入 `small`，转移堆顶给 `large`，必要时再转移回来"的顺序执行，保证任何时候 `small` 的最大值不超过 `large` 的最小值，且两堆大小最多相差一。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 双堆维护中位数 |
| 关键不变量 | `len(small) - len(large)` 恒为 `0` 或 `1`，`small` 的最大值 `<=` `large` 的最小值 |
| 时间 / 空间 | `addNum` 是 `O(log n)`，`findMedian` 是 `O(1)`；空间 `O(n)` |

下面的演示逐个插入一串数字，展示两堆如何在每一步之后保持平衡。

```median-two-heaps-demo
```

#### Quick Coding：Find Median From Data Stream

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
<summary>参考答案</summary>

```python
import heapq


class MedianFinder:
    def __init__(self):
        self.small = []  # 最大堆（取负），保存较小的一半
        self.large = []  # 最小堆，保存较大的一半

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

先无条件把新值推进 `small`，再把 `small` 的最大值转移给 `large`，这一步保证了 `small` 剩下的元素都不大于刚转移过去的这个值。如果转移之后 `large` 比 `small` 多了一个，再把 `large` 的最小值转移回来，让两堆大小差恢复到最多为一。这个顺序不需要先判断新值该进哪一堆，是这个模板最容易记住的部分。

</details>

## 模块四：面试前最后检查

1. 题目要的是"最值"还是"完整排序"？只要最值或前 `k` 个，堆通常比排序更合适。
2. 需要最大堆还是最小堆？`heapq` 只有最小堆，最大堆语义要靠取负实现。
3. 堆的大小是否应该固定为 `k`？固定大小能把复杂度从 `O(n log n)` 降到 `O(n log k)`。
4. 数据是一次性给定还是持续到来？持续到来（数据流、多个来源）通常提示需要维护一个或多个堆，而不是每次重新排序。
5. 这道题真的需要堆吗？统计频率或一个公式有时候比堆模拟更短、更不容易写错。

最后只记一句：

> 堆解决的是"反复要最值，不需要完整顺序"这一类问题；看清楚题目要哪种最值、要几个，模板就定了。

# Linked Lists

## 面试目标

实现单链表，重点是指针更新、头节点处理、按下标遍历和插入删除边界。

## 核心设计

- 节点保存 `val` 和 `next`。
- 可以使用 dummy head 简化头部插入和删除。
- `get(index)` 从头开始走 `index` 步。
- `insertHead(val)` 新节点指向旧头，再更新 head。
- `remove(index)` 找到前驱节点后跳过目标节点。

## 复杂度

- 头部插入：`O(1)`
- 按下标访问/插入/删除：`O(n)`
- 额外空间：`O(1)`，不计新节点。

## 常见坑

- 删除第一个节点时忘记更新 head。
- 遍历条件多走或少走一步。
- 删除尾节点时没有正确断开前驱的 `next`。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

用 dummy head 可以让插入和删除头节点不需要单独分支。`getPrev(index)` 返回目标位置前一个节点。

```text
insert(index, val):
  prev = dummy
  repeat index times:
    prev = prev.next
  node = Node(val)
  node.next = prev.next
  prev.next = node

remove(index):
  prev = getPrev(index)
  prev.next = prev.next.next
```

头插就是 `insert(0, val)`；尾插可以维护 `tail` 优化，也可以遍历到末尾。

</details>

上面的单链表 ADT 是这一章的前置。后面的 11 道题都在反复回答同一组问题：头节点会不会变，几个指针以什么相对速度移动，是否需要重连 `next`，是否要用哈希表补一个 `O(1)` 查找。

## 学习顺序

题目同样来自 [NeetCode 150](https://neetcode.io/practice/practice/neetcode150)。这里按依赖关系排序：先把反转、归并和快慢指针写稳，再扩展到随机指针、隐式图和双向链表设计。

| 顺序 | 原题 | 要掌握的内容 |
|---:|---|---|
| 1 | [206. Reverse Linked List](https://neetcode.io/problems/reverse-a-linked-list/question?list=neetcode150) | 迭代三指针反转 |
| 2 | [21. Merge Two Sorted Lists](https://neetcode.io/problems/merge-two-sorted-linked-lists/question?list=neetcode150) | dummy head + 双指针归并 |
| 3 | [141. Linked List Cycle](https://neetcode.io/problems/linked-list-cycle-detection/question?list=neetcode150) | Floyd 快慢指针 |
| 4 | [143. Reorder List](https://neetcode.io/problems/reorder-linked-list/question?list=neetcode150) | 找中点 + 反转后半段 + 交错归并 |
| 5 | [19. Remove Nth Node From End of List](https://neetcode.io/problems/remove-node-from-end-of-linked-list/question?list=neetcode150) | 固定间距双指针 |
| 6 | [138. Copy List With Random Pointer](https://neetcode.io/problems/copy-linked-list-with-random-pointer/question?list=neetcode150) | `old -> new` 哈希映射 |
| 7 | [2. Add Two Numbers](https://neetcode.io/problems/add-two-numbers/question?list=neetcode150) | 逐位相加与进位传播 |
| 8 | [287. Find The Duplicate Number](https://neetcode.io/problems/find-the-duplicate-number/question?list=neetcode150) | 把数组当作隐式链表做 Floyd |
| 9 | [146. LRU Cache](https://neetcode.io/problems/lru-cache/question?list=neetcode150) | 双向链表 + 哈希表 |
| 10 | [23. Merge K Sorted Lists](https://neetcode.io/problems/merge-k-sorted-linked-lists/question?list=neetcode150) | 两两归并 / divide and conquer |
| 11 | [25. Reverse Nodes In K Group](https://neetcode.io/problems/reverse-nodes-in-k-group/question?list=neetcode150) | 分段反转与重新接回 |

## 模块一：写代码前先过五问

链表题的表面题意差异很大，底层选择并不多。先把下面五件事问清楚，很多实现会自动收敛到固定模板。

1. 先在纸上画出节点、`next` 方向，以及每个指针在每一轮要停在哪里。
2. 头节点会不会变化？如果删除头节点、在头前面插入、或者从空结果链表开始构造，优先加 dummy head。
3. 需要几个移动指针，速度关系是什么？单指针遍历、固定间距双指针、快慢指针、还是多路合并。
4. 是否需要额外内存做 `O(1)` 查找？常见是 `old node -> new node`、`key -> node`。
5. 这道题只读取结构，还是要改写 `next`？读取类题目更强调停止条件；改写类题目更强调先保存后继，再改线。

| 检查项 | 典型信号 | 常用模板 |
|---|---|---|
| 头节点是否变化 | 删除头、头插、构造新链表 | dummy head |
| 指针如何同步移动 | 中点、环、倒数第 `n` 个 | 单指针 / 固定间距 / 快慢指针 |
| 是否需要批量归并 | 两条或多条有序链表 | dummy head + merge |
| 是否需要随机访问旧节点 | `random` 指针、缓存键值 | 哈希表 |
| 是否要重连 `next` | 反转、重排、分组反转 | 先保存后继，再改线 |

## 模块二：常用模板与惯用写法

### 哨兵头节点（dummy head）

只要答案链表的头部可能变化，或者你不想为“删头节点 / 插头节点”单独写分支，dummy head 都是最稳定的起点。做法是让所有修改统一发生在某个前驱节点的 `next` 上。

```python
dummy = ListNode(0, head)

# 所有操作都尽量改 dummy.next 或某个前驱的 next
prev = dummy
...

return dummy.next
```

常见用途有三类：

- 删除头节点，例如 Remove Nth Node From End of List。
- 从空链表构造结果，例如 Merge Two Sorted Lists、Add Two Numbers。
- 需要拿到某一段的“前一个节点”，例如 Reverse Nodes in K Group。

### 反转

链表反转的主模板是迭代三指针。顺序固定：先保存 `next_node`，再把 `curr.next` 指回 `prev`，最后推进三个变量。

```python
def reverse(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

递归版也常见，但这一章后面的题主要复用迭代版，因为它更容易嵌入“只反转一段”的场景。

```python
def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

下面的演示把“保存后继、翻转箭头、推进指针”拆成独立步骤。

```linked-list-reversal-demo
```

### 快慢指针

快慢指针这一组模板有三种常用速度关系。题面不同，核心结构相同。

| 用法 | 速度关系 | 典型停止条件 | 代表题 |
|---|---|---|---|
| 找中点 | `slow +1`，`fast +2` | `fast` 或 `fast.next` 为空 | Reorder List |
| 检测环 / 找环入口 | `slow +1`，`fast +2` | `slow == fast` | Linked List Cycle、Find The Duplicate Number |
| 倒数第 `n` 个 | `lead` 先走固定步数，再同步 | `lead` 走到结尾 | Remove Nth Node From End of List |

找中点：

```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

# slow 停在中点
```

Floyd 检测环：

```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        break
```

找到环入口：

```python
finder = head
while finder != slow:
    finder = finder.next
    slow = slow.next

# finder / slow 都停在环入口
```

固定间距找倒数第 `n` 个节点本体：

```python
lead = follow = head
for _ in range(n):
    lead = lead.next

while lead:
    lead = lead.next
    follow = follow.next

# follow 停在倒数第 n 个节点
```

删除类题通常把这套模板和 dummy head 一起用，让跟随指针停在待删节点的前驱。

```fast-slow-pointer-demo
```

### 归并

两条有序链表的归并模板是：dummy head 作为结果前缀，`tail` 始终指向当前结果链表的末尾，每次接上两条链表当前头节点中较小的那个。

```python
dummy = tail = ListNode(0)

while l1 and l2:
    if l1.val <= l2.val:
        tail.next = l1
        l1 = l1.next
    else:
        tail.next = l2
        l2 = l2.next
    tail = tail.next

tail.next = l1 or l2
return dummy.next
```

这个模板直接推广到 `k` 条有序链表。常见写法有两种：

- 两两归并，按 merge sort 的方式分治，时间复杂度 `O(N log k)`。
- 最小堆，每次取最小头节点，时间复杂度同样是 `O(N log k)`。

这章的参考解法选择两两归并，因为它直接复用 Merge Two Sorted Lists。

### 从旧节点映射到新节点

Copy List With Random Pointer 的核心困难不是复制 `next`，而是复制 `random`。`random` 可以指向任意旧节点，因此需要先建立 `old -> new` 的映射，再回头接线。

```python
old_to_new = {None: None}

cur = head
while cur:
    old_to_new[cur] = Node(cur.val)
    cur = cur.next

cur = head
while cur:
    copy = old_to_new[cur]
    copy.next = old_to_new[cur.next]
    copy.random = old_to_new[cur.random]
    cur = cur.next
```

还有一种 `O(1)` 额外空间的“穿插复制”写法，但两趟哈希表版本更直接，也更适合先把思路写对。

### 进位传播

Add Two Numbers 的结构和手算加法一致：逐位读取、计算 `sum`、拆成 `digit` 与 `carry`，再把新节点接到结果链表末尾。

```python
dummy = tail = ListNode(0)
carry = 0

while l1 or l2 or carry:
    v1 = l1.val if l1 else 0
    v2 = l2.val if l2 else 0
    total = v1 + v2 + carry
    carry, digit = divmod(total, 10)
    tail.next = ListNode(digit)
    tail = tail.next
    l1 = l1.next if l1 else None
    l2 = l2.next if l2 else None
```

这里 dummy head 的作用是把“第一位结果”和“后续结果”的连接逻辑完全统一。

### 分段反转

Reverse Nodes in K Group 不是整条链表反转，而是重复做“检查这一段是否够长，够长就只反转这一段，再接回去”。模板比普通反转多了三个边界对象：

- `group_prev`：当前分组前一个节点。
- `kth`：这一组的最后一个节点。
- `group_next`：这一组后面第一个节点。

```python
kth = get_kth(group_prev, k)
if not kth:
    break

group_next = kth.next
prev = group_next
curr = group_prev.next

while curr != group_next:
    next_node = curr.next
    curr.next = prev
    prev = curr
    curr = next_node
```

反转完成后，原分组头节点会变成新分组尾节点，再把它接到下一段的起点即可。

### Floyd 在隐式图上的复用

Find The Duplicate Number 的关键观察是：数组也可以视为一张“每个点只有一条出边”的函数图，其中

```text
next(i) = nums[i]
```

因为长度是 `n + 1`，值域是 `[1, n]`，至少有两个位置会指向同一个后继，因此这张图一定有环。重复值就是环入口。

```python
slow = fast = 0
while True:
    slow = nums[slow]
    fast = nums[nums[fast]]
    if slow == fast:
        break

finder = 0
while finder != slow:
    finder = nums[finder]
    slow = nums[slow]

return slow
```

这道题没有 `ListNode`，但它和 Linked List Cycle 用的是完全同一个 Floyd 模板：只要能定义 `next(i)`，快慢指针就能用，不需要真的是链表节点。

### 为什么第二阶段是必须的：同余论证

记从下标 `0` 出发的访问序列为 $x_0, x_1, x_2, \ldots$，其中 $x_0 = 0$，$x_{i+1} = \text{nums}[x_i]$。设尾长为 $\mu$（$x_0, \ldots, x_{\mu-1}$ 在环外，各不相同），环长为 $\lambda$（$x_\mu$ 是环入口，也就是重复值本身）。对 $k \ge \mu$，$x_k$ 在环上的位置是 $(k - \mu) \bmod \lambda$。

第一阶段循环第 $k$ 轮后，`slow` 停在 $x_k$，`fast` 停在 $x_{2k}$。两者相遇当且仅当 $x_k = x_{2k}$，也就是

$$
(k - \mu) \equiv (2k - \mu) \pmod \lambda
\quad\Longleftrightarrow\quad
\lambda \mid k.
$$

第一次相遇发生在第一个满足 $\lambda \mid k$ 且 $k \ge \mu$ 的 $k$，记作 $k^*$。相遇点在环上的位置是

$$
(k^* - \mu) \bmod \lambda = (-\mu) \bmod \lambda.
$$

这个位置只有在 $\lambda \mid \mu$ 时才是 `0`（环入口）；一般情况下它是环上任意一点。这就是第一阶段结束后不能直接返回 `slow` 的原因：`slow` 停在的是"某个满足 $\lambda \mid k$ 的相遇点"，不是"环入口"，两者只在特殊情况下重合。

第二阶段把 `finder` 放回 `x_0`，`slow` 留在相遇点，两者都以每轮一步的速度前进。设两者各走了 $s$ 步：

- `finder` 走到环入口需要 $s = \mu$ 步：它还在环外的尾部，要先走完尾长。
- `slow` 的位置是 $\big((-\mu \bmod \lambda) + s\big) \bmod \lambda$；当 $s = \mu$ 时这个值恒等于 $0$，与 $\mu \bmod \lambda$ 具体是多少无关。

`finder` 和 `slow`在 $s = \mu$ 步后同时到达环入口。$s < \mu$ 时 `finder` 还在尾部，和环上的节点不相交，不可能提前相遇，所以 $s = \mu$ 是它们第一次相遇的时刻——这是第二阶段返回值一定正确的原因。

#### 具体例子：`nums = [1, 3, 4, 2, 2]`

访问序列：$x_0=0, x_1=1, x_2=3, x_3=2, x_4=4, x_5=2, x_6=4, \ldots$。尾部是 $x_0, x_1, x_2$（$\mu = 3$），环是 $2 \to 4 \to 2$（$\lambda = 2$），环入口是 `2`，也就是重复值。

第一阶段：

| 轮次 | `slow` | `fast` | 是否相遇 |
|---:|---|---|---|
| 1 | 1 | 3 | 否 |
| 2 | 3 | 4 | 否 |
| 3 | 2 | 4 | 否 |
| 4 | 4 | 4 | 是 |

第一阶段在 `slow = 4` 处相遇（$k^* = 4$，是不小于 $\mu=3$ 的最小 `2` 的倍数）。如果直接返回 `slow`，答案会是 `4`，但真正的重复值是 `2`。验证一下论证：$(-\mu) \bmod \lambda = (-3) \bmod 2 = 1$，对应环入口之后第 `1` 个位置，正是 `4`，不是环入口本身。

第二阶段：

| 轮次 | `finder` | `slow` | 是否相遇 |
|---:|---|---|---|
| 1 | 1 | 2 | 否 |
| 2 | 3 | 4 | 否 |
| 3 | 2 | 2 | 是 |

恰好 $\mu = 3$ 轮后两者在真正的环入口 `2` 相遇，返回 `2`。

```array-duplicate-demo
```

### 双向链表 + 哈希表

LRU Cache 要求 `get` / `put` 都是 `O(1)`。这意味着：

- `key -> node` 需要哈希表；
- 任意节点的删除与尾部插入也必须是 `O(1)`，因此链表必须是双向链表。

头尾两端都放哨兵节点后，删除和插入都可以写成固定的四次指针改动：

```python
def remove(node):
    prev_node, next_node = node.prev, node.next
    prev_node.next = next_node
    next_node.prev = prev_node

def insert_before_tail(node):
    prev_node = tail.prev
    prev_node.next = node
    node.prev = prev_node
    node.next = tail
    tail.prev = node
```

这里 `head.next` 始终是最久未使用节点，`tail.prev` 始终是最新使用节点。

## 模块三：11 道题目的映射

### 1. Reverse Linked List

这道题单独考反转模板本身。结构最纯，目标只有一个：把“保存后继、改写 `next`、推进指针”的顺序写稳。

| 项目 | 内容 |
|---|---|
| 组合模板 | 反转 |
| 关键不变量 | `prev` 始终是已经反转好的前缀头节点 |
| 时间 / 空间 | `O(n) / O(1)` |

#### Quick Coding：Reverse Linked List

```python
def reverseList(head):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, curr = None, head

        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node

        return prev
```

</details>

### 2. Merge Two Sorted Lists

这道题把 dummy head 和双指针归并放在一起。链表本身已经有序，因此每一轮只需要比较两个当前头节点，接上较小的那个。

| 项目 | 内容 |
|---|---|
| 组合模板 | dummy head + 归并 |
| 关键不变量 | `tail.next` 之前的部分始终保持有序 |
| 时间 / 空间 | `O(n + m) / O(1)` |

#### Quick Coding：Merge Two Sorted Lists

```python
def mergeTwoLists(list1, list2):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def mergeTwoLists(
        self,
        list1: Optional[ListNode],
        list2: Optional[ListNode],
    ) -> Optional[ListNode]:
        dummy = tail = ListNode(0)

        while list1 and list2:
            if list1.val <= list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next

        tail.next = list1 or list2
        return dummy.next
```

</details>

### 3. Linked List Cycle

这道题只读取结构，不改写 `next`。重点是停止条件：只要 `fast` 和 `fast.next` 都存在，就可以让 `slow` 走一步、`fast` 走两步。

| 项目 | 内容 |
|---|---|
| 组合模板 | Floyd 快慢指针 |
| 关键不变量 | 有环时，`fast` 最终会在环内追上 `slow` |
| 时间 / 空间 | `O(n) / O(1)` |

#### Quick Coding：Linked List Cycle

```python
def hasCycle(head):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True

        return False
```

</details>

### 4. Reorder List

这是一道组合题，完整流程是：先用快慢指针找到中点，再反转后半段，最后把前半段和反转后的后半段交错归并。三个子步骤都不复杂，难点在于把它们按正确顺序接起来。

| 项目 | 内容 |
|---|---|
| 组合模板 | 找中点 + 反转后半段 + 交错归并 |
| 关键不变量 | 拆分后前后两段各自独立，归并时先保存两边后继 |
| 时间 / 空间 | `O(n) / O(1)` |

#### Quick Coding：Reorder List

```python
def reorderList(head):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head or not head.next:
            return

        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        second = slow.next
        slow.next = None

        prev = None
        while second:
            next_node = second.next
            second.next = prev
            prev = second
            second = next_node

        first, second = head, prev
        while second:
            next_first = first.next
            next_second = second.next
            first.next = second
            second.next = next_first
            first = next_first
            second = next_second
```

</details>

### 5. Remove Nth Node From End of List

固定间距双指针负责定位，dummy head 负责统一删除头节点的边界。把 `fast` 从 dummy 开始先走 `n + 1` 步后，`slow` 会停在待删节点的前驱。

| 项目 | 内容 |
|---|---|
| 组合模板 | dummy head + 固定间距双指针 |
| 关键不变量 | `fast` 和 `slow` 始终相差 `n + 1` 个节点 |
| 时间 / 空间 | `O(n) / O(1)` |

#### Quick Coding：Remove Nth Node From End of List

```python
def removeNthFromEnd(head, n):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def removeNthFromEnd(
        self,
        head: Optional[ListNode],
        n: int,
    ) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        fast = slow = dummy

        for _ in range(n + 1):
            fast = fast.next

        while fast:
            fast = fast.next
            slow = slow.next

        slow.next = slow.next.next
        return dummy.next
```

</details>

### 6. Copy List With Random Pointer

这道题的 `next` 结构很普通，真正的信息在 `random`。清晰写法是两趟扫描：第一趟只创建新节点并记录映射，第二趟再把 `next` 和 `random` 都接好。

| 项目 | 内容 |
|---|---|
| 组合模板 | `old -> new` 哈希映射 |
| 关键不变量 | 每个旧节点只创建一个新节点，所有指针都从映射表读取 |
| 时间 / 空间 | `O(n) / O(n)` |

#### Quick Coding：Copy List With Random Pointer

```python
def copyRandomList(head):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        if not head:
            return None

        old_to_new = {None: None}

        cur = head
        while cur:
            old_to_new[cur] = Node(cur.val)
            cur = cur.next

        cur = head
        while cur:
            copy = old_to_new[cur]
            copy.next = old_to_new[cur.next]
            copy.random = old_to_new[cur.random]
            cur = cur.next

        return old_to_new[head]
```

</details>

### 7. Add Two Numbers

这道题的链表只是数字的存储形式，核心运算是进位传播。dummy head 负责构造结果，`carry` 负责把低位信息传到下一位。

| 项目 | 内容 |
|---|---|
| 组合模板 | dummy head + 进位传播 |
| 关键不变量 | `carry` 保存上一位产生的进位 |
| 时间 / 空间 | `O(n) / O(n)`，结果链表不计为额外空间时辅助空间是 `O(1)` |

#### Quick Coding：Add Two Numbers

```python
def addTwoNumbers(l1, l2):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def addTwoNumbers(
        self,
        l1: Optional[ListNode],
        l2: Optional[ListNode],
    ) -> Optional[ListNode]:
        dummy = tail = ListNode(0)
        carry = 0

        while l1 or l2 or carry:
            v1 = l1.val if l1 else 0
            v2 = l2.val if l2 else 0
            total = v1 + v2 + carry
            carry, digit = divmod(total, 10)

            tail.next = ListNode(digit)
            tail = tail.next

            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return dummy.next
```

</details>

### 8. Find The Duplicate Number

这道题是 Floyd 模板的迁移题。数组下标扮演“节点位置”，`nums[i]` 扮演“下一跳指针”，重复值对应环入口。

| 项目 | 内容 |
|---|---|
| 组合模板 | Floyd 快慢指针，但指针来自 `nums[i]` |
| 关键不变量 | `next(i) = nums[i]` 构成一张带环的函数图 |
| 时间 / 空间 | `O(n) / O(1)` |

#### Quick Coding：Find The Duplicate Number

```python
def findDuplicate(nums):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        slow = fast = 0

        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break

        finder = 0
        while finder != slow:
            finder = nums[finder]
            slow = nums[slow]

        return slow
```

</details>

### 9. LRU Cache

这道题把“链表题”扩展成数据结构设计题。哈希表给出 `key -> node` 的 `O(1)` 查找，双向链表负责 `O(1)` 删除任意节点和把节点移到最新位置。

| 项目 | 内容 |
|---|---|
| 组合模板 | 双向链表 + 哈希表 + 头尾哨兵 |
| 关键不变量 | `left.next` 始终是 LRU，`r\right.prev` 始终是 MRU |
| 时间 / 空间 | `O(1)` 平均时间每次操作，`O(capacity)` 空间 |

#### Quick Coding：LRU Cache

```python
class LRUCache:
    def __init__(self, capacity):
        ...

    def get(self, key):
        ...

    def put(self, key, value):
        ...
```

<details>
<summary>参考答案</summary>

```python
class Node:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.left = Node()   # LRU sentinel
        self.right = Node()  # MRU sentinel
        self.left.next = self.right
        self.r\right.prev = self.left

    def remove(self, node: Node) -> None:
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def insert_before_tail(self, node: Node) -> None:
        prev_node = self.r\right.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.right
        self.r\right.prev = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self.remove(node)
        self.insert_before_tail(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.remove(self.cache[key])

        node = Node(key, value)
        self.cache[key] = node
        self.insert_before_tail(node)

        if len(self.cache) > self.capacity:
            lru = self.left.next
            self.remove(lru)
            del self.cache[lru.key]
```

</details>

下面的演示固定 `capacity = 2`，逐步执行一串 `put`/`get`，展示双向链表顺序和哈希表如何随每次访问变化，包括淘汰发生的那一步。

```lru-cache-demo
```

### 10. Merge K Sorted Lists

这道题直接复用两路归并，但把它提升到多路场景。最顺手的链表写法是两两归并：每一轮把列表数减半，直到只剩一条。

| 项目 | 内容 |
|---|---|
| 组合模板 | Merge Two Sorted Lists + divide and conquer |
| 关键不变量 | 每一轮合并后，每条中间结果都仍然有序 |
| 时间 / 空间 | `O(N log k) / O(1)` 额外链表空间，若不计中间列表容器 |

#### Quick Coding：Merge K Sorted Lists

```python
def mergeKLists(lists):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List, Optional


class Solution:
    def mergeKLists(
        self,
        lists: List[Optional[ListNode]],
    ) -> Optional[ListNode]:
        if not lists:
            return None

        def merge_two(
            l1: Optional[ListNode],
            l2: Optional[ListNode],
        ) -> Optional[ListNode]:
            dummy = tail = ListNode(0)

            while l1 and l2:
                if l1.val <= l2.val:
                    tail.next = l1
                    l1 = l1.next
                else:
                    tail.next = l2
                    l2 = l2.next
                tail = tail.next

            tail.next = l1 or l2
            return dummy.next

        while len(lists) > 1:
            merged = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i + 1] if i + 1 < len(lists) else None
                merged.append(merge_two(l1, l2))
            lists = merged

        return lists[0]
```

</details>

### 11. Reverse Nodes In K Group

这道题可以看成”把普通反转模板限定在长度为 `k` 的窗口里，再反复执行”。反转前先确认这一组确实有 `k` 个节点；不满足就保持后半段原样。

| 项目 | 内容 |
|---|---|
| 组合模板 | dummy head + 分段反转 |
| 关键不变量 | `group_prev.next` 始终指向当前分组头节点 |
| 时间 / 空间 | `O(n) / O(1)` |

#### Quick Coding：Reverse Nodes In K Group

```python
def reverseKGroup(head, k):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def reverseKGroup(
        self,
        head: Optional[ListNode],
        k: int,
    ) -> Optional[ListNode]:
        def get_kth(node: Optional[ListNode], steps: int) -> Optional[ListNode]:
            while node and steps > 0:
                node = node.next
                steps -= 1
            return node

        dummy = ListNode(0, head)
        group_prev = dummy

        while True:
            kth = get_kth(group_prev, k)
            if not kth:
                break

            group_next = kth.next
            prev = group_next
            curr = group_prev.next

            while curr != group_next:
                next_node = curr.next
                curr.next = prev
                prev = curr
                curr = next_node

            new_group_tail = group_prev.next
            group_prev.next = kth
            group_prev = new_group_tail

        return dummy.next
```

</details>

## 模块四：面试前最后检查

1. 头节点会不会变化？如果会，先决定是否加 dummy head。
2. 指针每一轮各走几步？停止条件是否在空指针之前就做了保护？
3. 是否要改写 `next`？如果要，是否总是先保存后继，再改线？
4. 题目要求的是节点本体、节点前驱，还是某个由链表诱导出的值？
5. 哈希表里存的键是什么：旧节点对象、数组下标，还是缓存键值？
6. 组合题是否已经拆成独立步骤，例如 Reorder List 的“找中点 -> 反转 -> 交错归并”？

最后只记一句：

> 链表题通常不是新算法；它更像是在 dummy head、反转、快慢指针、哈希映射这几块稳定模板之间做组合。

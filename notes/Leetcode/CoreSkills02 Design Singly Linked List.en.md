# Design Singly Linked List

## Interview Goal

Implement a singly linked list, with the focus on pointer updates, head-node handling, index-based traversal, and insertion/deletion boundaries.

## Core Design

- Each node stores `val` and `next`.
- A dummy head can be used to simplify insertion and deletion at the head.
- `get(index)` walks `index` steps from the head.
- `insertHead(val)` makes the new node point to the old head, then updates `head`.
- `remove(index)` finds the predecessor node and then skips over the target node.

## Complexity

- Insertion at the head: `O(1)`
- Index-based access/insertion/deletion: `O(n)`
- Extra space: `O(1)`, excluding the new node itself.

## Common Pitfalls

- Forgetting to update `head` when deleting the first node.
- Traversal conditions that move one step too far or one step too few.
- Failing to correctly detach the predecessor's `next` when deleting the tail node.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

Using a dummy head avoids special branching when inserting or deleting the head node. `getPrev(index)` returns the node immediately before the target position.

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

Insertion at the head is just `insert(0, val)`; insertion at the tail can be optimized by maintaining `tail`, or you can simply traverse to the end.

</details>

The singly linked-list ADT above is the prerequisite for this chapter. The next 11 problems keep asking the same questions in different forms: can the head change, how many pointers move and at what relative speed, do we need to rewire `next`, and do we need a hash map for `O(1)` lookup.

## Learning Order

These problems also come from [NeetCode 150](https://neetcode.io/practice/practice/neetcode150). The order here follows dependency: stabilize reversal, merge, and fast-slow pointers first, then extend them to random pointers, implicit graphs, and doubly linked-list design.

| Order | Problem | What to Master |
|---:|---|---|
| 1 | [206. Reverse Linked List](https://neetcode.io/problems/reverse-a-linked-list/question?list=neetcode150) | Iterative three-pointer reversal |
| 2 | [21. Merge Two Sorted Lists](https://neetcode.io/problems/merge-two-sorted-linked-lists/question?list=neetcode150) | Dummy head + two-pointer merge |
| 3 | [141. Linked List Cycle](https://neetcode.io/problems/linked-list-cycle-detection/question?list=neetcode150) | Floyd fast-slow pointers |
| 4 | [143. Reorder List](https://neetcode.io/problems/reorder-linked-list/question?list=neetcode150) | Find middle + reverse second half + interleave merge |
| 5 | [19. Remove Nth Node From End of List](https://neetcode.io/problems/remove-node-from-end-of-linked-list/question?list=neetcode150) | Fixed-gap two pointers |
| 6 | [138. Copy List With Random Pointer](https://neetcode.io/problems/copy-linked-list-with-random-pointer/question?list=neetcode150) | `old -> new` hash mapping |
| 7 | [2. Add Two Numbers](https://neetcode.io/problems/add-two-numbers/question?list=neetcode150) | Digit-by-digit addition with carry propagation |
| 8 | [287. Find The Duplicate Number](https://neetcode.io/problems/find-the-duplicate-number/question?list=neetcode150) | Treat the array as an implicit linked list and apply Floyd |
| 9 | [146. LRU Cache](https://neetcode.io/problems/lru-cache/question?list=neetcode150) | Doubly linked list + hash map |
| 10 | [23. Merge K Sorted Lists](https://neetcode.io/problems/merge-k-sorted-linked-lists/question?list=neetcode150) | Pairwise merge / divide and conquer |
| 11 | [25. Reverse Nodes In K Group](https://neetcode.io/problems/reverse-nodes-in-k-group/question?list=neetcode150) | Segment reversal and reconnection |

## Module 1: Ask These Five Questions Before Coding

Linked-list problems vary in wording, but the implementation choices are limited. Once these five questions are settled, the code usually collapses to a known template.

1. Draw the nodes and the `next` arrows first, and decide where each pointer must stand after every loop.
2. Can the head change? If you may delete the head, insert before the head, or build a result list from scratch, a dummy head is usually the cleanest start.
3. How many moving pointers are needed, and what is their relative speed? A single traversal pointer, a fixed-gap pair, fast-slow pointers, or multiple pointers for merging.
4. Is extra memory required for `O(1)` lookup? Common examples are `old node -> new node` and `key -> node`.
5. Does the problem only read the structure, or does it rewire `next`? Read-only problems emphasize stopping rules; rewiring problems emphasize saving the successor before changing pointers.

| Check | Typical Signal | Common Template |
|---|---|---|
| Can the head change? | Delete head, insert at head, build a new list | Dummy head |
| How do pointers move together? | Middle, cycle, nth from end | Single pointer / fixed gap / fast-slow pointers |
| Is there batch merging? | Two or more sorted lists | Dummy head + merge |
| Is random access to old nodes required? | `random` pointer, cache keys | Hash map |
| Does `next` get rewritten? | Reversal, reorder, k-group reversal | Save successor, then rewire |

## Module 2: Common Templates and Idioms

### Dummy Head

Whenever the answer list may get a new head, or you do not want a special branch for "delete the head" or "insert at the head," a dummy head is the most reliable starting point. The idea is to make all structural changes happen through some predecessor node's `next`.

```python
dummy = ListNode(0, head)

# Modify dummy.next or the next of some predecessor
prev = dummy
...

return dummy.next
```

There are three common uses:

- Removing the head, as in Remove Nth Node From End of List.
- Building a result list from scratch, as in Merge Two Sorted Lists and Add Two Numbers.
- Needing "the node before a segment," as in Reverse Nodes in K Group.

### Reversal

The main linked-list reversal template is the iterative three-pointer pattern. The order is fixed: save `next_node`, point `curr.next` backward to `prev`, then advance all three variables.

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

The recursive version also appears in interviews, but this chapter reuses the iterative version throughout because it is easier to embed inside "reverse only one segment."

```python
def reverse_recursive(head):
    if not head or not head.next:
        return head
    new_head = reverse_recursive(head.next)
    head.next.next = head
    head.next = None
    return new_head
```

The demo below separates "save successor," "flip one arrow," and "advance pointers" into distinct steps.

```linked-list-reversal-demo
```

### Fast and Slow Pointers

This family has three common relative-speed patterns. The wording changes; the structure does not.

| Use | Relative Speed | Typical Stop Condition | Representative Problems |
|---|---|---|---|
| Find the middle | `slow +1`, `fast +2` | `fast` or `fast.next` becomes empty | Reorder List |
| Detect a cycle / find cycle entry | `slow +1`, `fast +2` | `slow == fast` | Linked List Cycle, Find The Duplicate Number |
| Find the nth node from the end | `lead` moves first, then both move together | `lead` reaches the end | Remove Nth Node From End of List |

Find the middle:

```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next

# slow is at the middle
```

Floyd cycle detection:

```python
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        break
```

Find the cycle entry:

```python
finder = head
while finder != slow:
    finder = finder.next
    slow = slow.next

# finder / slow are both at the entry
```

Fixed-gap search for the actual nth node from the end:

```python
lead = follow = head
for _ in range(n):
    lead = lead.next

while lead:
    lead = lead.next
    follow = follow.next

# follow is at the nth node from the end
```

Deletion problems usually combine this with a dummy head so the trailing pointer lands on the predecessor of the target node.

```fast-slow-pointer-demo
```

### Merging

The template for merging two sorted linked lists is: use a dummy head for the result, keep `tail` at the current end of the merged list, and attach the smaller of the two current heads on each step.

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

This generalizes directly to `k` sorted lists. The two standard implementations are:

- Pairwise divide-and-conquer merging, with time complexity `O(N log k)`.
- A min-heap over list heads, with the same `O(N log k)` complexity.

This chapter uses pairwise merging because it directly reuses Merge Two Sorted Lists.

### Mapping Old Nodes to New Nodes

The main difficulty in Copy List With Random Pointer is not copying `next`, but copying `random`. Since `random` can point to any old node, build the `old -> new` mapping first, then wire everything afterward.

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

An `O(1)` extra-space weaving solution also exists, but the two-pass hash-map version is clearer and easier to make correct on the first try.

### Carry Propagation

Add Two Numbers follows the same structure as grade-school addition: read the current digits, compute `sum`, split it into `digit` and `carry`, and append a new node to the result list.

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

The dummy head removes the distinction between "the first result digit" and "every later digit."

### Segment Reversal

Reverse Nodes in K Group is not full-list reversal. It repeatedly checks whether the next segment really has `k` nodes, reverses only that segment, and reconnects it. The extra boundary objects are:

- `group_prev`: the node before the current segment.
- `kth`: the last node of the current segment.
- `group_next`: the first node after the current segment.

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

After the reversal, the original segment head becomes the new segment tail and is reconnected to the next segment.

### Floyd on an Implicit Graph

The key observation in Find The Duplicate Number is that an array can also be treated as a functional graph where

```text
next(i) = nums[i]
```

The array length is `n + 1`, but every value lies in `[1, n]`, so at least two positions must point into the same chain. That creates a cycle, and the duplicate value is the entry of that cycle.

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

This problem has no `ListNode`, but it uses exactly the same Floyd template as Linked List Cycle: fast-slow pointers work on any structure with a well-defined `next(i)`, not only literal linked-list nodes.

### Doubly Linked List + Hash Map

LRU Cache requires both `get` and `put` to run in `O(1)`. That means:

- `key -> node` must be a hash lookup.
- Removing an arbitrary node and reinserting it at the "most recent" end must also be `O(1)`, which requires a doubly linked list.

Once sentinels are placed at both ends, removal and insertion are always the same four pointer writes:

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

Here `head.next` is always the least recently used node, and `tail.prev` is always the most recently used node.

## Module 3: Mapping the 11 Problems

### 1. Reverse Linked List

This problem isolates the reversal template itself. The structure is minimal: stabilize the order "save successor, rewrite `next`, advance pointers."

| Item | Value |
|---|---|
| Composed idioms | Reversal |
| Invariant | `prev` is always the head of the already reversed prefix |
| Time / Space | `O(n) / O(1)` |

#### Quick Coding: Reverse Linked List

```python
def reverseList(head):
    ...
```

<details>
<summary>Reference answer</summary>

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

This combines a dummy head with two-pointer merging. Since both lists are already sorted, each step only needs to compare the two current heads and attach the smaller one.

| Item | Value |
|---|---|
| Composed idioms | Dummy head + merge |
| Invariant | The prefix before `tail.next` is always sorted |
| Time / Space | `O(n + m) / O(1)` |

#### Quick Coding: Merge Two Sorted Lists

```python
def mergeTwoLists(list1, list2):
    ...
```

<details>
<summary>Reference answer</summary>

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

This problem reads the structure without changing `next`. The main concern is the stopping condition: as long as both `fast` and `fast.next` exist, move `slow` by one and `fast` by two.

| Item | Value |
|---|---|
| Composed idioms | Floyd fast-slow pointers |
| Invariant | If a cycle exists, `fast` eventually catches `slow` inside the cycle |
| Time / Space | `O(n) / O(1)` |

#### Quick Coding: Linked List Cycle

```python
def hasCycle(head):
    ...
```

<details>
<summary>Reference answer</summary>

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

This problem combines three idioms: find the middle with fast-slow pointers, reverse the second half, then interleave the first half with the reversed second half.

| Item | Value |
|---|---|
| Composed idioms | Find middle + reverse second half + interleave merge |
| Invariant | After the split, both halves are independent; merging always saves both successors first |
| Time / Space | `O(n) / O(1)` |

#### Quick Coding: Reorder List

```python
def reorderList(head):
    ...
```

<details>
<summary>Reference answer</summary>

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

The fixed-gap two-pointer technique handles the position, and the dummy head handles the head-deletion boundary. Once `fast` starts from the dummy and moves `n + 1` steps first, `slow` lands on the predecessor of the node to delete.

| Item | Value |
|---|---|
| Composed idioms | Dummy head + fixed-gap two pointers |
| Invariant | `fast` and `slow` remain `n + 1` nodes apart |
| Time / Space | `O(n) / O(1)` |

#### Quick Coding: Remove Nth Node From End of List

```python
def removeNthFromEnd(head, n):
    ...
```

<details>
<summary>Reference answer</summary>

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

The `next` structure is simple; the real information sits in `random`. The clearest implementation is two passes: first create every new node and record the mapping, then wire both `next` and `random`.

| Item | Value |
|---|---|
| Composed idioms | `old -> new` hash mapping |
| Invariant | Every old node creates exactly one copy; all copied pointers are read from the map |
| Time / Space | `O(n) / O(n)` |

#### Quick Coding: Copy List With Random Pointer

```python
def copyRandomList(head):
    ...
```

<details>
<summary>Reference answer</summary>

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

Here the linked lists are just the storage format for digits; the core operation is carry propagation. The dummy head builds the result, and `carry` transports overflow to the next digit.

| Item | Value |
|---|---|
| Composed idioms | Dummy head + carry propagation |
| Invariant | `carry` stores the overflow produced by the previous digit |
| Time / Space | `O(n) / O(n)`; auxiliary space is `O(1)` if the output list is excluded |

#### Quick Coding: Add Two Numbers

```python
def addTwoNumbers(l1, l2):
    ...
```

<details>
<summary>Reference answer</summary>

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

This is the transfer problem for Floyd's algorithm. Array indices act as node positions, and `nums[i]` acts as the next pointer, so the duplicate value becomes the cycle entry.

| Item | Value |
|---|---|
| Composed idioms | Floyd fast-slow pointers with `nums[i]` as the pointer |
| Invariant | `next(i) = nums[i]` forms a functional graph with a cycle |
| Time / Space | `O(n) / O(1)` |

#### Quick Coding: Find The Duplicate Number

```python
def findDuplicate(nums):
    ...
```

<details>
<summary>Reference answer</summary>

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

This extends "linked-list problems" into data-structure design. The hash map provides `key -> node` in `O(1)`, and the doubly linked list provides `O(1)` deletion of an arbitrary node plus `O(1)` insertion at the "most recent" end.

| Item | Value |
|---|---|
| Composed idioms | Doubly linked list + hash map + head/tail sentinels |
| Invariant | `left.next` is always the LRU node, and `right.prev` is always the MRU node |
| Time / Space | `O(1)` average time per operation, `O(capacity)` space |

#### Quick Coding: LRU Cache

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
<summary>Reference answer</summary>

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
        self.right.prev = self.left

    def remove(self, node: Node) -> None:
        prev_node, next_node = node.prev, node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def insert_before_tail(self, node: Node) -> None:
        prev_node = self.right.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.right
        self.right.prev = node

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

### 10. Merge K Sorted Lists

This directly reuses two-list merging, but lifts it to multiple lists. The most natural linked-list implementation is pairwise merging: halve the number of lists on each pass until only one remains.

| Item | Value |
|---|---|
| Composed idioms | Merge Two Sorted Lists + divide and conquer |
| Invariant | After each merge round, every intermediate list remains sorted |
| Time / Space | `O(N log k) / O(1)` extra linked-list space, excluding the temporary list container |

#### Quick Coding: Merge K Sorted Lists

```python
def mergeKLists(lists):
    ...
```

<details>
<summary>Reference answer</summary>

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

This is the full reversal template constrained to a window of length `k`, repeated across the list. Before reversing, verify that the segment really contains `k` nodes; otherwise leave the remaining suffix unchanged.

| Item | Value |
|---|---|
| Composed idioms | Dummy head + segment reversal |
| Invariant | `group_prev.next` always points to the head of the current segment |
| Time / Space | `O(n) / O(1)` |

#### Quick Coding: Reverse Nodes In K Group

```python
def reverseKGroup(head, k):
    ...
```

<details>
<summary>Reference answer</summary>

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

## Module 4: Final Checks Before an Interview

1. Can the head change? If yes, decide whether a dummy head should be added before coding anything else.
2. How many steps does each pointer move per iteration? Is the stopping condition protected before any null access?
3. Will `next` be rewritten? If yes, is the successor always saved before the rewiring?
4. Does the problem ask for the node itself, the predecessor of the node, or some value induced by the list structure?
5. What is the key stored in the hash map: an old node object, an array index, or a cache key?
6. In a synthesis problem, has the flow been split into stable substeps, such as "find middle -> reverse -> interleave" for Reorder List?

Keep one sentence in memory:

> Linked-list problems are usually not new algorithms; they are combinations of a few stable templates: dummy head, reversal, fast-slow pointers, and hash-based node lookup.

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

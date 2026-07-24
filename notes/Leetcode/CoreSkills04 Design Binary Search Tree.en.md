# Design Binary Search Tree

## Interview Goal

Implement a binary search tree and master the ordered properties behind insertion, search, deletion, and in-order traversal.

## Core Design

- For any node, values in the left subtree are smaller, and values in the right subtree are larger.
- During search, move left or right based on the value comparison.
- Node deletion falls into three cases: leaf, single subtree, and two subtrees.
- For deletion with two subtrees, a common replacement is the minimum node in the right subtree or the maximum node in the left subtree.

## Complexity

- Search/insertion/deletion when balanced: `O(log n)`
- `O(n)` in the extreme case when it degenerates into a linked list
- In-order traversal: `O(n)`

## Common Pitfalls

- Forgetting to delete the replacement node from its original position after deleting a node with two subtrees.
- Failing to return the updated subtree root.
- Ignoring the strategy for duplicate values.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

Insertion and search both move left or right according to value comparisons. During deletion, recursively return the new subtree root so the parent can reconnect to the updated subtree.

```text
delete(root, key):
  if root is null: return null
  if key < root.val: root.left = delete(root.left, key)
  else if key > root.val: root.right = delete(root.right, key)
  else:
    if root.left is null: return root.right
    if root.right is null: return root.left
    succ = minNode(root.right)
    root.val = succ.val
    root.right = delete(root.right, succ.val)
  return root
```

Deletion with two subtrees uses the in-order successor as the replacement. After replacement, you still need to delete the successor node from the right subtree.

</details>

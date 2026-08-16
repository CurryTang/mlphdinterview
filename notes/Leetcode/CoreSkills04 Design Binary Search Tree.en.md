# Trees

## Prerequisite: Design Binary Search Tree

### Interview Goal

Implement a binary search tree and master the ordered properties behind insertion, search, deletion, and in-order traversal.

### Core Design

- For any node, values in the left subtree are smaller, and values in the right subtree are larger.
- During search, move left or right based on the value comparison.
- Node deletion falls into three cases: leaf, single subtree, and two subtrees.
- For deletion with two subtrees, a common replacement is the minimum node in the right subtree or the maximum node in the left subtree.

### Complexity

- Search/insertion/deletion when balanced: `O(log n)`
- `O(n)` in the extreme case when it degenerates into a linked list
- In-order traversal: `O(n)`

### Common Pitfalls

- Forgetting to delete the replacement node from its original position after deleting a node with two subtrees.
- Failing to return the updated subtree root.
- Ignoring the strategy for duplicate values.

### Reference Solution

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

The BST ADT above is the prerequisite for this chapter. The remaining material organizes tree problems around four kinds of information: traversal order, recursive return values, path state, and BST ordering constraints. All 15 Trees problems combine these fixed structures.

## Learning Order

These problems come from the Trees module of [NeetCode 150](https://neetcode.io/practice/practice/neetcode150). The order establishes traversal and height recursion first, then proceeds through structural comparison, BST constraints, reconstruction, and path aggregation.

| Order | Problem | What to Master |
|---:|---|---|
| 1 | [226. Invert Binary Tree](https://neetcode.io/problems/invert-a-binary-tree/question?list=neetcode150) | Preorder or postorder recursion that swaps child pointers |
| 2 | [104. Maximum Depth of Binary Tree](https://neetcode.io/problems/depth-of-binary-tree/question?list=neetcode150) | The basic height return value |
| 3 | [543. Diameter of Binary Tree](https://neetcode.io/problems/binary-tree-diameter/question?list=neetcode150) | Return height while updating the global diameter |
| 4 | [110. Balanced Binary Tree](https://neetcode.io/problems/balanced-binary-tree/question?list=neetcode150) | A height sentinel and early termination |
| 5 | [100. Same Tree](https://neetcode.io/problems/same-binary-tree/question?list=neetcode150) | Compare two trees node by node |
| 6 | [572. Subtree of Another Tree](https://neetcode.io/problems/subtree-of-a-binary-tree/question?list=neetcode150) | Reuse Same Tree at every candidate node |
| 7 | [235. Lowest Common Ancestor of a BST](https://neetcode.io/problems/lowest-common-ancestor-in-binary-search-tree/question?list=neetcode150) | Descend one direction using BST values |
| 8 | [102. Binary Tree Level Order Traversal](https://neetcode.io/problems/level-order-traversal-of-binary-tree/question?list=neetcode150) | Queue processing with per-level snapshots |
| 9 | [199. Binary Tree Right Side View](https://neetcode.io/problems/binary-tree-right-side-view/question?list=neetcode150) | Keep the final node from each level |
| 10 | [1448. Count Good Nodes in Binary Tree](https://neetcode.io/problems/count-good-nodes-in-binary-tree/question?list=neetcode150) | Pass the path maximum downward |
| 11 | [98. Validate Binary Search Tree](https://neetcode.io/problems/valid-binary-search-tree/question?list=neetcode150) | Pass an allowed open interval downward |
| 12 | [230. Kth Smallest Element in a BST](https://neetcode.io/problems/kth-smallest-integer-in-bst/question?list=neetcode150) | Inorder traversal and the kth visit |
| 13 | [105. Construct Binary Tree from Preorder and Inorder Traversal](https://neetcode.io/problems/binary-tree-from-preorder-and-inorder-traversal/question?list=neetcode150) | Choose roots from preorder and split with inorder |
| 14 | [124. Binary Tree Maximum Path Sum](https://neetcode.io/problems/binary-tree-maximum-path-sum/question?list=neetcode150) | Downward path gain and global path sum |
| 15 | [297. Serialize and Deserialize Binary Tree](https://neetcode.io/problems/serialize-and-deserialize-binary-tree/question?list=neetcode150) | Preorder encoding with explicit null markers |

## Module 1: Four Traversal Templates

The fixed example tree below produces preorder `1, 2, 4, 5, 3, 6`, inorder `4, 2, 5, 1, 3, 6`, postorder `4, 5, 2, 6, 3, 1`, and level order `1, 2, 3, 4, 5, 6`.

```text
        1
      /   \
     2     3
    / \     \
   4   5     6
```

```tree-traversal-demo
```

### Preorder: root → left → right

The recursive form concatenates results in visit order.

```python
def preorder(root):
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []
```

The iterative form pushes the right child before the left child. The stack's last-in, first-out order processes the left subtree first.

```python
def preorder_iterative(root):
    if not root:
        return []

    stack = [root]
    order = []
    while stack:
        node = stack.pop()
        order.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return order
```

### Inorder: left → root → right

The recursive form completes the left subtree before visiting the root. A BST with distinct keys produces a strictly increasing inorder sequence.

```python
def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []
```

The iterative form repeatedly pushes the left chain. When `current` becomes empty, pop and visit one node, then move to its right subtree.

```python
def inorder_iterative(root):
    stack, order = [], []
    current = root

    while stack or current:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        order.append(current.val)
        current = current.right

    return order
```

### Postorder: left → right → root

The recursive form places the root after both subtrees.

```python
def postorder(root):
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []
```

The iterative form first produces modified preorder `root → right → left`, then reverses the complete sequence. Push the left child before the right child so the right child pops first. A single stack with `last_visited` also works, but it requires more state branches.

```python
def postorder_iterative(root):
    if not root:
        return []

    stack = [root]
    reverse_order = []
    while stack:
        node = stack.pop()
        reverse_order.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return reverse_order[::-1]
```

### Level Order: a BFS queue

The queue processes nodes in first-in, first-out order. Enqueueing the left and right children in that order preserves left-to-right order within each level.

```python
from collections import deque


def level_order(root):
    if not root:
        return []

    queue = deque([root])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return order
```

Every traversal visits each node once, for `O(n)` time. Recursive DFS and iterative DFS use `O(h)` call-stack or explicit-stack space. BFS stores up to one level at a time, for `O(w)` space.

## Module 2: Five Core Recursive Patterns

### 1. Bottom-Up Return Value + Global Best

Iterative postorder processes both children before computing the parent. A cache stores the quantity that each parent can continue composing, while a separate variable records the answer across the full tree.

```python
def solve(root):
    value = {None: 0}
    best = 0
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if node is None:
            continue
        if processed:
            left = value[node.left]
            right = value[node.right]
            best = max(best, combine_for_answer(left, right, node))
            value[node] = value_for_parent(left, right, node)
        else:
            stack.append((node, True))
            stack.append((node.left, False))
            stack.append((node.right, False))
    return best
```

`processed=False` expands the children first. `processed=True` means both child values are finalized in the cache, so the current node can be computed. Diameter caches height and records `left_height + right_height` as the answer. Maximum Path Sum caches a one-sided downward path sum and records a complete path that may use both branches. The whole-tree answer and the quantity required by the parent differ; conflating them is a common implementation error.

Count Good Nodes applies the same separation between state and answer aggregation. Its explicit stack passes the path maximum from parent to child, while a separate variable records the count, making it a top-down variant of the pattern.

Used by: Diameter of Binary Tree, Binary Tree Maximum Path Sum, Count Good Nodes in Binary Tree.

### 2. Structural Comparison Recursion

The explicit stack stores corresponding nodes from the two trees together. The null-node combination and value equality determine the result.

```python
def same(a, b):
    stack = [(a, b)]
    while stack:
        x, y = stack.pop()
        if not x and not y:
            continue
        if not x or not y or x.val != y.val:
            return False
        stack.append((x.left, y.left))
        stack.append((x.right, y.right))
    return True
```

Same Tree uses this template directly. Subtree of Another Tree calls `same(node, subRoot)` at every node in the larger tree and continues into both subtrees after a failed match.

Used by: Same Tree, Subtree of Another Tree.

### 3. The BST Ordering Invariant

The most useful BST-specific fact is that inorder traversal is sorted. It supports three common implementations.

| Problem | Use of the invariant |
|---|---|
| Kth Smallest | The kth inorder visit is the kth smallest value; traversal can stop early |
| Validate BST | Each node carries an open interval `(low, high)`; the left upper bound and right lower bound tighten to the current value |
| LCA of a BST | Move left when both targets are smaller, move right when both are larger, and stop at the split point |

Checking only `node.left.val < node.val < node.right.val` misses violations that cross multiple levels. The allowed interval must include constraints from every ancestor. LCA follows one path in `O(h)` time and does not search both subtrees.

Validate BST uses an explicit stack to carry the ancestor bounds for each node:

```python
import math


def valid(root):
    stack = [(root, -math.inf, math.inf)]
    while stack:
        node, low, high = stack.pop()
        if not node:
            continue
        if not low < node.val < high:
            return False
        stack.append((node.left, low, node.val))
        stack.append((node.right, node.val, high))
    return True
```

Iterative inorder traversal can also check that values are strictly increasing. The bounds stack more directly shows the constraints imposed by all ancestors.

### 4. Traversal-Sequence Reconstruction

Preorder supplies each new node in sequence. The stack stores the current path of nodes still waiting for a right child, and `j` points to the next node to complete in inorder.

```python
def build_tree(preorder, inorder):
    root = TreeNode(preorder[0])
    stack = [root]
    j = 0
    for i in range(1, len(preorder)):
        node = TreeNode(preorder[i])
        parent = None
        while stack and stack[-1].val == inorder[j]:
            parent = stack.pop()
            j += 1
        if parent:
            parent.right = node
        else:
            stack[-1].left = node
        stack.append(node)
    return root
```

When the next preorder value differs from `inorder[j]`, it is the left child of the stack top. When the values match, the current left subtree is complete. Pop consecutive stack entries that match inorder, then attach the new node as the right child of the last popped node. Each node enters and leaves the stack once, giving `O(n)` time and `O(n)` auxiliary space.

Serialize and Deserialize uses preorder with explicit null markers and a stack of nodes with pending children. Preorder supplies the current root first, and the remaining markers are consumed in the fixed order of left subtree followed by right subtree. Inorder alone places left-side content before the root and cannot identify the root position without another traversal.

Used by: Construct Binary Tree from Preorder and Inorder Traversal, Serialize and Deserialize Binary Tree.

### 5. Recomputed Heights and the Sentinel Fix

A direct Balanced Binary Tree implementation calls a separate `height()` function at every node. On a skewed tree, the same descendants are revisited many times, producing `O(n^2)` time.

One iterative postorder traversal can compute height and detect imbalance together. A cache stores the heights of processed subtrees, and the function returns `False` immediately when it finds an imbalance.

```python
def is_balanced(root):
    height = {None: 0}
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if node is None:
            continue
        if processed:
            left_h = height[node.left]
            right_h = height[node.right]
            if abs(left_h - right_h) > 1:
                return False
            height[node] = 1 + max(left_h, right_h)
        else:
            stack.append((node, True))
            stack.append((node.left, False))
            stack.append((node.right, False))
    return True
```

The recursive version needs `-1` to carry the "subtree is already unbalanced" signal across call frames. The iterative version can return `False` directly from the function when it finds an imbalance, so it does not need a sentinel value. Both mechanisms apply the same principle: stop as soon as the answer is known. The recursive version's `-1` still follows the same interface pattern as a binary-search boundary sentinel and the trailing sentinel in Largest Rectangle: a special value carries control information and shortens later processing.

## Module 3: Self-Balancing Tree Fundamentals

Inserting sorted values into a plain BST can leave every node with only a right child. The height becomes `n`, and search, insertion, and deletion become `O(n)`. Self-balancing trees constrain height through local restructuring or multi-way nodes.

### AVL Tree

An AVL tree maintains a balance factor at every node:

$$
\text{balance}(node) = \text{height}(node.left) - \text{height}(node.right)
$$

Every node requires `|balance| <= 1`. After an insertion or deletion creates an imbalance, the two directions along the heavy path select the rotation.

| Case | Heavy path | Repair |
|---|---|---|
| LL | left → left | Right-rotate the unbalanced node |
| RR | right → right | Left-rotate the unbalanced node |
| LR | left → right | Left-rotate the left child, then right-rotate the unbalanced node |
| RL | right → left | Right-rotate the right child, then left-rotate the unbalanced node |

The LL example uses nodes `30, 20, 10, 25`. After the right rotation, `20` becomes the new root and `30` moves to the right. The previous `20.right = 25` is reattached as `30.left`. The inorder sequence remains `10, 20, 25, 30`.

```text
        30                 20
       /                  /  \
     20        ->        10   30
    /  \                    /
   10  25                  25
       T2                  T2
```

The LR example uses `30, 10, 20`. First rotate left around `10`, converting the shape to LL. Then rotate right around `30`.

```text
      30              30              20
     /               /               /  \
   10       ->      20      ->      10   30
     \             /
     20           10
```

```avl-rotation-demo
```

### Red-Black Tree

A Red-Black tree maintains these coloring invariants:

1. Every node is red or black.
2. The root and all null leaves are black.
3. Every red node has black children.
4. Every path from a node to a descendant null leaf contains the same number of black nodes.

These constraints prevent consecutive red nodes and fix the black height of every path. The longest path is at most about twice the shortest path, keeping tree height in `O(log n)`. Red-Black balance is looser than AVL balance and usually requires fewer rotations during updates. C++ `std::map` and Java `TreeMap` commonly use Red-Black trees.

### B-Tree

A B-Tree node stores several sorted keys and several child pointers. The high branching factor reduces tree height substantially. One disk page can hold a complete node, so one page read returns multiple separator keys. Database indexes and filesystems use this structure to reduce disk or storage-page reads.

### Comparison

| Structure | Balance criterion | Insert / delete rebalance cost | Typical use |
|---|---|---|---|
| AVL | Left and right heights differ by at most 1 at every node | Few rotations with strict query height; deletion may continue repairing ancestors | Read-heavy in-memory ordered structures |
| Red-Black | Color, red-node, and black-height invariants | Constant rotations plus recoloring; update cost is usually lower | General ordered maps such as `std::map` and `TreeMap` |
| B-Tree | Each multi-way node keeps its key count within capacity bounds; all leaves share a depth | Split, merge, or borrow keys from a sibling | Database indexes, filesystems, block storage |

## Module 4: Mapping the 15 Problems

### 1. Invert Binary Tree

This problem combines iterative DFS with a local pointer swap. Swap each node's children, then push its non-null children onto the stack.

| Item | Value |
|---|---|
| Composed patterns | Tree traversal + local pointer update |
| State | The current node's child pointers |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Invert Binary Tree

```python
def invertTree(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        stack = [root]
        while stack:
            node = stack.pop()
            node.left, node.right = node.right, node.left
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return root
```

</details>

### 2. Maximum Depth of Binary Tree

Level-order BFS increments the depth after processing each level. When the queue is empty, the accumulated level count is the maximum depth.

| Item | Value |
|---|---|
| Composed patterns | Level order + level count |
| State | The current queue length is the level size |
| Time / Space | `O(n) / O(w)` |

#### Quick Coding: Maximum Depth of Binary Tree

```python
def maxDepth(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import deque
from typing import Optional


class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        depth = 0
        queue = deque([root])
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return depth
```

</details>

### 3. Diameter of Binary Tree

Iterative postorder finalizes both child heights before computing the current node's height. A separate variable records `left_height + right_height` at every node. The diameter is measured in edges.

| Item | Value |
|---|---|
| Composed patterns | Postorder stack + height cache + global best |
| Cached / answer quantity | Subtree height / maximum diameter |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Diameter of Binary Tree

```python
def diameterOfBinaryTree(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        height = {None: 0}
        diameter = 0
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if node is None:
                continue
            if processed:
                left_h, right_h = height[node.left], height[node.right]
                diameter = max(diameter, left_h + right_h)
                height[node] = 1 + max(left_h, right_h)
            else:
                stack.append((node, True))
                stack.append((node.left, False))
                stack.append((node.right, False))
        return diameter
```

Each `(node, processed)` pair records a processing stage. `processed=False` expands the node's children first. `processed=True` means both child heights are finalized in `height`, so the current height can be computed. This is a standard iterative postorder simulation, and Maximum Path Sum reuses the same structure.

</details>

### 4. Balanced Binary Tree

Iterative postorder caches the height of every subtree. The function returns `False` immediately when a node's child heights differ by more than `1`.

| Item | Value |
|---|---|
| Composed patterns | Postorder stack + height cache + early termination |
| Condition | `abs(left - right) <= 1` |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Balanced Binary Tree

```python
def isBalanced(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        height = {None: 0}
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if node is None:
                continue
            if processed:
                left_h, right_h = height[node.left], height[node.right]
                if abs(left_h - right_h) > 1:
                    return False
                height[node] = 1 + max(left_h, right_h)
            else:
                stack.append((node, True))
                stack.append((node.left, False))
                stack.append((node.right, False))
        return True
```

</details>

### 5. Same Tree

This is the base structural-comparison problem. Each stack element contains two nodes at the same structural position. Two null nodes match; one null node or unequal values fail immediately.

| Item | Value |
|---|---|
| Composed patterns | Pair stack + structural comparison |
| State | Two nodes at the same structural position |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Same Tree

```python
def isSameTree(p, q):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        stack = [(p, q)]
        while stack:
            a, b = stack.pop()
            if not a and not b:
                continue
            if not a or not b or a.val != b.val:
                return False
            stack.append((a.left, b.left))
            stack.append((a.right, b.right))
        return True
```

</details>

### 6. Subtree of Another Tree

This problem uses iterative Same Tree as a subroutine. After a failed match at the current node, the main stack continues searching both subtrees for a candidate root.

| Item | Value |
|---|---|
| Composed patterns | Structural-comparison stack + candidate-root stack |
| Operation | `same(node, subRoot)` |
| Time / Space | Worst case `O(mn) / O(h)` |

#### Quick Coding: Subtree of Another Tree

```python
def isSubtree(root, subRoot):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def same(a: Optional[TreeNode], b: Optional[TreeNode]) -> bool:
            stack = [(a, b)]
            while stack:
                x, y = stack.pop()
                if not x and not y:
                    continue
                if not x or not y or x.val != y.val:
                    return False
                stack.append((x.left, y.left))
                stack.append((x.right, y.right))
            return True

        if not subRoot:
            return True
        stack = [root]
        while stack:
            node = stack.pop()
            if not node:
                continue
            if same(node, subRoot):
                return True
            stack.append(node.left)
            stack.append(node.right)
        return False
```

</details>

### 7. Lowest Common Ancestor of a BST

The BST ordering invariant limits the search to one root-to-leaf path. When the target values lie on different sides of the current value, the current node is the split point.

| Item | Value |
|---|---|
| Composed patterns | BST ordering invariant |
| Branches | Both left, both right, split |
| Time / Space | `O(h) / O(1)` |

#### Quick Coding: Lowest Common Ancestor of a BST

```python
def lowestCommonAncestor(root, p, q):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def lowestCommonAncestor(
        self,
        root: TreeNode,
        p: TreeNode,
        q: TreeNode,
    ) -> Optional[TreeNode]:
        current = root
        low, high = sorted((p.val, q.val))

        while current:
            if high < current.val:
                current = current.left
            elif low > current.val:
                current = current.right
            else:
                return current
        return None
```

</details>

### 8. Binary Tree Level Order Traversal

Add one inner loop to the basic BFS. Snapshot the current queue length before each level; that length is the number of nodes in the level.

| Item | Value |
|---|---|
| Composed patterns | Level order + level-size snapshot |
| State | `level_size = len(queue)` |
| Time / Space | `O(n) / O(w)` |

#### Quick Coding: Binary Tree Level Order Traversal

```python
def levelOrder(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import deque
from typing import List, Optional


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result
```

</details>

### 9. Binary Tree Right Side View

This reuses per-level BFS. The final node removed from each level is visible from the right side.

| Item | Value |
|---|---|
| Composed patterns | Level order + final visit per level |
| Condition | `i == level_size - 1` |
| Time / Space | `O(n) / O(w)` |

#### Quick Coding: Binary Tree Right Side View

```python
def rightSideView(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import deque
from typing import List, Optional


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []
        queue = deque([root])
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if i == level_size - 1:
                    result.append(node.val)
        return result
```

</details>

### 10. Count Good Nodes in Binary Tree

Each `(node, max_seen)` pair on the stack passes the path maximum from the root downward. Increment the count when the current value is at least that maximum, then update the value passed to both children.

| Item | Value |
|---|---|
| Composed patterns | Path state + outer aggregate count |
| State | `max_seen` |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Count Good Nodes in Binary Tree

```python
def goodNodes(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        good = 0
        stack = [(root, root.val)]
        while stack:
            node, max_seen = stack.pop()
            if node.val >= max_seen:
                good += 1
            next_max = max(max_seen, node.val)
            if node.left:
                stack.append((node.left, next_max))
            if node.right:
                stack.append((node.right, next_max))
        return good
```

</details>

### 11. Validate Binary Search Tree

The explicit stack carries the open interval `(low, high)` established by all ancestors for each node. The open interval also excludes duplicate keys.

| Item | Value |
|---|---|
| Composed patterns | BST ordering + interval propagation |
| Condition | `low < node.val < high` |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Validate Binary Search Tree

```python
def isValidBST(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import math
from typing import Optional


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = [(root, -math.inf, math.inf)]
        while stack:
            node, low, high = stack.pop()
            if not node:
                continue
            if not low < node.val < high:
                return False
            stack.append((node.left, low, node.val))
            stack.append((node.right, node.val, high))
        return True
```

Iterative inorder traversal can also check that values are strictly increasing. The bounds stack remains the primary solution because it directly shows that the allowed range contains constraints from every ancestor.

</details>

### 12. Kth Smallest Element in a BST

Iterative inorder traversal pops nodes in ascending order. Return the current value on the kth pop-and-visit operation.

| Item | Value |
|---|---|
| Composed patterns | BST ordering + iterative inorder |
| State | Remaining visit count `k` |
| Time / Space | `O(h + k) / O(h)` |

#### Quick Coding: Kth Smallest Element in a BST

```python
def kthSmallest(root, k):
    ...
```

<details>
<summary>Reference answer</summary>

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        current = root

        while stack or current:
            while current:
                stack.append(current)
                current = current.left

            current = stack.pop()
            k -= 1
            if k == 0:
                return current.val
            current = current.right

        raise ValueError("k exceeds the number of nodes")
```

</details>

### 13. Construct Binary Tree from Preorder and Inorder Traversal

Preorder creates each node in sequence. The stack stores nodes on the current path that are still waiting for a right child, and `j` advances in lockstep with inorder.

| Item | Value |
|---|---|
| Composed patterns | Traversal-sequence reconstruction |
| State | Pending-node stack + inorder pointer |
| Time / Space | `O(n) / O(n)` |

#### Quick Coding: Construct Binary Tree from Preorder and Inorder Traversal

```python
def buildTree(preorder, inorder):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List, Optional


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        root = TreeNode(preorder[0])
        stack = [root]
        j = 0
        for i in range(1, len(preorder)):
            node = TreeNode(preorder[i])
            parent = None
            while stack and stack[-1].val == inorder[j]:
                parent = stack.pop()
                j += 1
            if parent:
                parent.right = node
            else:
                stack[-1].left = node
            stack.append(node)
        return root
```

The invariant is that `stack` holds the current right spine of nodes still waiting for a right child, while `j` points to the next unfinished position in inorder. Before attaching the next preorder node, compare the stack top with `inorder[j]`. If they differ, the new node must be the left child of the stack top, so construction continues down a left spine. If they match, that node's left subtree is complete. Pop consecutive entries that match `inorder[j]` and advance `j` after each pop. The new node becomes the right child of the last node popped. If nothing was popped, it remains the left child of the current stack top.

For `preorder=[3,9,20,15,7]` and `inorder=[9,3,15,20,7]`, the execution is:

1. Create root `3`: `stack=[3]`, `j=0`, and the next inorder value is `9`.
2. Process `9`: stack top `3 != 9`, so set `3.left=9`. The stack becomes `[3,9]`.
3. Process `20`: stack top `9` matches inorder value `9`, so pop `9`. Then `3` matches inorder value `3`, so pop `3`. Now `j=2`; set `3.right=20`, and the stack becomes `[20]`.
4. Process `15`: stack top `20 != 15`, so set `20.left=15`. The stack becomes `[20,15]`.
5. Process `7`: pop `15` and `20` as they match the next inorder values. Now `j=4`; set `20.right=7`.

The result is `3(9, 20(15,7))`.

</details>

The demo below steps through this exact example, showing the stack, the `j` pointer, and the tree under construction changing together.

```build-tree-demo
```

### 14. Binary Tree Maximum Path Sum

A parent can continue through one downward branch, so the cache stores `node.val + max(left_gain, right_gain)`. The complete candidate at the current node may join both branches and updates the global answer. Negative gains are clamped to `0`.

| Item | Value |
|---|---|
| Composed patterns | Postorder stack + gain cache + global best |
| Cached / answer quantity | One-sided downward gain / maximum path with any endpoints |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Binary Tree Maximum Path Sum

```python
def maxPathSum(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import math


class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        gain = {None: 0}
        best = -math.inf
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if node is None:
                continue
            if processed:
                left_gain = max(0, gain[node.left])
                right_gain = max(0, gain[node.right])
                best = max(best, node.val + left_gain + right_gain)
                gain[node] = node.val + max(left_gain, right_gain)
            else:
                stack.append((node, True))
                stack.append((node.left, False))
                stack.append((node.right, False))
        return best
```

The `(node, processed)` state is identical to Diameter: expand the children first, then combine their cached results at the current node. Both problems use the same iterative postorder pattern with different combine functions for height and path gain.

</details>

### 15. Serialize and Deserialize Binary Tree

The preorder sequence records every node value and writes `#` for every null child. Serialization and deserialization both use explicit stacks to track pending children.

| Item | Value |
|---|---|
| Composed patterns | Preorder reconstruction with null markers |
| Invariant | Serialization and deserialization use the same token order |
| Time / Space | `O(n) / O(n)` |

#### Quick Coding: Serialize and Deserialize Binary Tree

```python
class Codec:
    def serialize(self, root):
        ...

    def deserialize(self, data):
        ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        tokens = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node is None:
                tokens.append("#")
                continue
            tokens.append(str(node.val))
            stack.append(node.right)
            stack.append(node.left)
        return ",".join(tokens)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        tokens = data.split(",")
        idx = 0
        root_token = tokens[idx]
        idx += 1
        if root_token == "#":
            return None
        root = TreeNode(int(root_token))
        stack = [[root, 0]]
        while idx < len(tokens):
            token = tokens[idx]
            idx += 1
            entry = stack[-1]
            parent = entry[0]
            new_node = None if token == "#" else TreeNode(int(token))
            if entry[1] == 0:
                parent.left = new_node
                entry[1] = 1
            else:
                parent.right = new_node
                entry[1] = 2
            while stack and stack[-1][1] == 2:
                stack.pop()
            if new_node is not None:
                stack.append([new_node, 0])
        return root
```

`serialize` is iterative preorder with null markers. Push the right child before the left child so the left child pops first, and emit `"#"` for every null child.

Each `deserialize` stack entry is `[node, fill_count]`, where `fill_count` records how many of the node's two children have been assigned: `0`, `1`, or `2`. Each token becomes the next pending child of the current stack top. When `fill_count` reaches `2`, the node is complete and is popped. This can pop several ancestors in sequence when the new assignment also completes them. A non-null new node is then pushed to await its own two children.

This stack and the Preorder+Inorder construction stack both track pending children. Here, null markers state whether each child exists, so no inorder cross-reference is required.

</details>

## Module 5: Final Checks Before an Interview

1. Does the problem need preorder, inorder, postorder, or level order? Does the visit timing match the target quantity?
2. What does the helper return to its parent? Does the full-tree answer require a separate aggregate?
3. Does path state flow downward, such as an allowed interval or a path maximum?
4. Has a BST problem used sorted inorder traversal or one-direction descent?
5. Are heights recomputed? Can one DFS and a sentinel combine the states?
6. Does reconstruction record null nodes explicitly or use a second traversal to define subtree boundaries?
7. Have the worst-case recursion depth, explicit stack, or BFS queue space been stated?

Keep one sentence in memory:

> Tree problems reduce to the visit time at each node, the state passed downward, and the value returned to the parent.

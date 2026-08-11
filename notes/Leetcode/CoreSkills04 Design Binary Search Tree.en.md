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

The helper returns a quantity the parent can continue composing, while an outer variable records the answer across the full tree.

```python
best = 0

def dfs(node):
    nonlocal best
    if not node:
        return 0

    left = dfs(node.left)
    right = dfs(node.right)
    best = max(best, combine_for_answer(left, right, node))
    return value_for_parent(left, right, node)
```

Diameter returns height and records `left_height + right_height` outside the helper. Maximum Path Sum returns a one-sided downward path sum and records a complete path that may use both branches. The whole-tree answer and the recursive quantity required by the parent differ; conflating them is the common implementation error.

Count Good Nodes applies the same separation between recursive state and answer aggregation. Its path maximum flows from parent to child, while an outer variable records the count, making it a top-down variant of the pattern.

Used by: Diameter of Binary Tree, Binary Tree Maximum Path Sum, Count Good Nodes in Binary Tree.

### 2. Structural Comparison Recursion

Two nodes enter each recursive call together. The null-node combination and value equality determine the result.

```python
def same(a, b):
    if not a or not b:
        return a is b
    return (
        a.val == b.val
        and same(a.left, b.left)
        and same(a.right, b.right)
    )
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

### 4. Traversal-Sequence Reconstruction

The first preorder value identifies the root of the current subtree. Its position in inorder splits the nodes into left and right subtrees. A hash map from value to inorder index gives `O(1)` splits.

```python
preorder_index = 0
inorder_index = {value: i for i, value in enumerate(inorder)}

def build(left, right):
    nonlocal preorder_index
    if left > right:
        return None
    root_value = preorder[preorder_index]
    preorder_index += 1
    root = TreeNode(root_value)
    split = inorder_index[root_value]
    root.left = build(left, split - 1)
    root.right = build(split + 1, right)
    return root
```

Serialize and Deserialize uses preorder with an explicit null marker. Preorder supplies the current root first, and the remaining markers can be consumed in the fixed order of left subtree followed by right subtree. Inorder alone places left-side content before the root and cannot identify the root position without another traversal.

Used by: Construct Binary Tree from Preorder and Inorder Traversal, Serialize and Deserialize Binary Tree.

### 5. Recomputed Heights and the Sentinel Fix

A direct Balanced Binary Tree implementation calls a separate `height()` function at every node. On a skewed tree, the same descendants are revisited many times, producing `O(n^2)` time.

One bottom-up DFS can compute height and detect imbalance together. Return `-1` as soon as a subtree is unbalanced, then propagate that sentinel upward.

```python
def height_or_unbalanced(node):
    if not node:
        return 0

    left = height_or_unbalanced(node.left)
    if left == -1:
        return -1

    right = height_or_unbalanced(node.right)
    if right == -1 or abs(left - right) > 1:
        return -1

    return 1 + max(left, right)
```

This `-1` follows the same interface pattern as a binary-search boundary sentinel and the trailing sentinel in Largest Rectangle: one special value carries control information and shortens later processing.

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

This problem combines basic DFS with a local pointer swap. Swap each node's children, then recurse into both subtrees. Preorder and postorder both work.

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

        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

</details>

### 2. Maximum Depth of Binary Tree

This is the basic height recursion. An empty tree has height `0`; a nonempty node has one plus the larger child height.

| Item | Value |
|---|---|
| Composed patterns | Bottom-up height return |
| Recurrence | `1 + max(left_depth, right_depth)` |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Maximum Depth of Binary Tree

```python
def maxDepth(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

</details>

### 3. Diameter of Binary Tree

The helper returns height to its parent, while an outer variable records `left_height + right_height` at every node. The diameter is measured in edges.

| Item | Value |
|---|---|
| Composed patterns | Return value + global best |
| Returned / answer quantity | Subtree height / maximum diameter |
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
        diameter = 0

        def height(node: Optional[TreeNode]) -> int:
            nonlocal diameter
            if not node:
                return 0

            left = height(node.left)
            right = height(node.right)
            diameter = max(diameter, left + right)
            return 1 + max(left, right)

        height(root)
        return diameter
```

</details>

### 4. Balanced Binary Tree

The height DFS returns `-1` when a subtree is already unbalanced. The sentinel propagates up the call stack, so the tree is traversed once.

| Item | Value |
|---|---|
| Composed patterns | Height recursion + imbalance sentinel |
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
        def height(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            left = height(node.left)
            if left == -1:
                return -1

            right = height(node.right)
            if right == -1 or abs(left - right) > 1:
                return -1

            return 1 + max(left, right)

        return height(root) != -1
```

</details>

### 5. Same Tree

This is the base structural-comparison problem. Two null nodes match; one null node or unequal values fail immediately.

| Item | Value |
|---|---|
| Composed patterns | Structural comparison recursion |
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
    def isSameTree(
        self,
        p: Optional[TreeNode],
        q: Optional[TreeNode],
    ) -> bool:
        if not p or not q:
            return p is q
        return (
            p.val == q.val
            and self.isSameTree(p.left, q.left)
            and self.isSameTree(p.right, q.right)
        )
```

</details>

### 6. Subtree of Another Tree

This problem uses Same Tree as a subroutine. After a failed match at the current node, continue searching for a candidate root in both subtrees.

| Item | Value |
|---|---|
| Composed patterns | Structural comparison + candidate-root enumeration |
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
    def isSubtree(
        self,
        root: Optional[TreeNode],
        subRoot: Optional[TreeNode],
    ) -> bool:
        def same(a: Optional[TreeNode], b: Optional[TreeNode]) -> bool:
            if not a or not b:
                return a is b
            return (
                a.val == b.val
                and same(a.left, b.left)
                and same(a.right, b.right)
            )

        if not subRoot:
            return True
        if not root:
            return False
        return (
            same(root, subRoot)
            or self.isSubtree(root.left, subRoot)
            or self.isSubtree(root.right, subRoot)
        )
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

Pass the path maximum from the root downward. Increment the count when the current value is at least that maximum, then update the value passed to both children.

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

        def dfs(node: TreeNode, max_seen: int) -> None:
            nonlocal good
            if not node:
                return

            if node.val >= max_seen:
                good += 1
            next_max = max(max_seen, node.val)
            dfs(node.left, next_max)
            dfs(node.right, next_max)

        dfs(root, root.val)
        return good
```

</details>

### 11. Validate Binary Search Tree

Every node must lie inside the open interval `(low, high)` established by all ancestors. The open interval also excludes duplicate keys.

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
        def valid(node: Optional[TreeNode], low: float, high: float) -> bool:
            if not node:
                return True
            if not low < node.val < high:
                return False
            return (
                valid(node.left, low, node.val)
                and valid(node.right, node.val, high)
            )

        return valid(root, -math.inf, math.inf)
```

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

The preorder pointer selects each subtree root. The inorder hash map gives the split position, and recursive boundaries define both subtree ranges.

| Item | Value |
|---|---|
| Composed patterns | Traversal-sequence reconstruction |
| State | Preorder pointer + inorder interval |
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
    def buildTree(
        self,
        preorder: List[int],
        inorder: List[int],
    ) -> Optional[TreeNode]:
        inorder_index = {value: i for i, value in enumerate(inorder)}
        preorder_index = 0

        def build(left: int, right: int) -> Optional[TreeNode]:
            nonlocal preorder_index
            if left > right:
                return None

            root_value = preorder[preorder_index]
            preorder_index += 1
            root = TreeNode(root_value)
            split = inorder_index[root_value]
            root.left = build(left, split - 1)
            root.right = build(split + 1, right)
            return root

        return build(0, len(inorder) - 1)
```

</details>

### 14. Binary Tree Maximum Path Sum

A parent can continue through one downward branch, so the helper returns `node.val + max(left_gain, right_gain)`. The complete candidate at the current node may join both branches and updates the global answer. Negative gains are clamped to `0`.

| Item | Value |
|---|---|
| Composed patterns | Return value + global best |
| Returned / answer quantity | One-sided downward gain / maximum path with any endpoints |
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
        best = -math.inf

        def gain(node: TreeNode) -> int:
            nonlocal best
            if not node:
                return 0

            left = max(0, gain(node.left))
            right = max(0, gain(node.right))
            best = max(best, node.val + left + right)
            return node.val + max(left, right)

        gain(root)
        return best
```

</details>

### 15. Serialize and Deserialize Binary Tree

The preorder sequence records every node value and writes `#` for every null child. Deserialization consumes tokens in the same order, so each recursive call can determine whether its current subtree is empty.

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

        def encode(node: Optional[TreeNode]) -> None:
            if not node:
                tokens.append("#")
                return
            tokens.append(str(node.val))
            encode(node.left)
            encode(node.right)

        encode(root)
        return ",".join(tokens)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        tokens = iter(data.split(","))

        def decode() -> Optional[TreeNode]:
            token = next(tokens)
            if token == "#":
                return None
            node = TreeNode(int(token))
            node.left = decode()
            node.right = decode()
            return node

        return decode()
```

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

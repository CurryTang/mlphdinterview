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

In tree interview problems, **recursion** is almost always the preferred approach. Recursive solutions are concise (typically 5–15 lines) and naturally align with the recursive definition of trees. On a whiteboard, explicit stack simulations (especially postorder state machines with backtracing) are tedious to write and error-prone.

Mastering the following 5 recursive patterns covers the vast majority of binary tree problems.

---

### 1. Bottom-Up Return Value + Global Best (Tree DP)

A child node reports its "single-branch maximum gain or height" up to its parent. The parent combines both reports to update a global optimal answer (spanning both left and right subtrees through the current node), and then returns the single-branch contribution to its own parent.

**Core Intuition**:
- **Return upward**: Can only pick one downward path to return to the parent (no branching).
- **Update globally**: Left and right branches can be joined at the current node to form an inverted-V path (global optimum).

```python
def solve(root):
    best = 0  # Global optimum across the entire tree

    def dfs(node):
        nonlocal best
        if not node:
            return 0  # Base case: null node contributes 0

        # 1. Obtain bottom-up single-branch contributions from children
        left = dfs(node.left)
        right = dfs(node.right)

        # 2. Update global answer (e.g., diameter left + right, or path sum left + right + val)
        best = max(best, combine_for_answer(left, right, node))

        # 3. Return the maximum single-branch contribution to the parent (cannot branch)
        return value_for_parent(left, right, node)

    dfs(root)
    return best
```

> **Common Pitfall**: Conflating the single-branch value returned upward with the global answer aggregated across both branches.

Used by: Diameter of Binary Tree (543), Binary Tree Maximum Path Sum (124).

---

### 2. Structural Comparison Recursion (Dual-Tree Traversal)

Traverse corresponding nodes in two trees simultaneously. Determine whether the structures and values match using straightforward base cases and parallel recursion.

```python
def is_same(a, b):
    # 1. Both null -> structural match
    if not a and not b:
        return True
    # 2. Exactly one null, or values differ -> mismatch
    if not a or not b or a.val != b.val:
        return False
    # 3. Both left and right subtrees must match
    return is_same(a.left, b.left) and is_same(a.right, b.right)
```

- **Core Intuition**: Two trees are identical if and only if their root values match and both their left and right subtrees are identical.
- **Variant**: Subtree of Another Tree (572) calls `is_same(node, subRoot)` at each node, falling back to `isSubtree(node.left, subRoot) or isSubtree(node.right, subRoot)` upon a mismatch.

Used by: Same Tree (100), Subtree of Another Tree (572).

---

### 3. The BST Ordering Invariant (Top-Down Range Propagation)

The fundamental property of a Binary Search Tree (BST) is that its **inorder traversal is strictly increasing** (every node is strictly greater than all nodes in its left subtree, and strictly less than all nodes in its right subtree).

Checking only `node.left.val < node.val < node.right.val` locally is a **classic bug**, as it misses multi-level ancestor violations (e.g., a left grandchild in the right subtree smaller than the root). The correct approach **propagates a valid open interval `(low, high)` top-down**.

```python
def is_valid_bst(root):
    def validate(node, low, high):
        if not node:
            return True
        # Current node value must lie strictly within (low, high)
        if not (low < node.val < high):
            return False
        # Going left tightens upper bound; going right tightens lower bound
        return validate(node.left, low, node.val) and validate(node.right, node.val, high)

    return validate(root, float('-inf'), float('inf'))
```

Three canonical BST patterns:

| Problem | Use of the invariant | Complexity |
|---|---|---|
| Validate BST (98) | Top-down `(low, high)` range propagation | `O(n) / O(h)` |
| Kth Smallest (230) | Inorder traversal is sorted; kth visited node is the answer | `O(h + k) / O(h)` |
| LCA of a BST (235) | Binary search path: both < root move left, both > root move right; split point is LCA | `O(h) / O(1)` |

Used by: Validate Binary Search Tree (98), Kth Smallest Element in a BST (230), Lowest Common Ancestor of a BST (235).

---

### 4. Traversal-Sequence Reconstruction (Preorder Finds Root, Inorder Splits)

Iterative stack backtracing with `parent` tracking is notoriously difficult to write without bugs under interview pressure. The most intuitive, memorable, and standard approach for LeetCode 105 is **divide-and-conquer recursion with a hash map**.

The essence of this problem is simply: **"Preorder gives the root; inorder splits into left and right subtrees."**

#### Core Intuition (Remember in 3 Seconds)

* **Preorder:** `[root, ...all left subtree nodes..., ...all right subtree nodes...]`
  * Role: **The first element is always the root of the current subtree.**
* **Inorder:** `[...all left subtree nodes..., root, ...all right subtree nodes...]`
  * Role: **Locating the root splits the array into left and right subtrees.**

Once we know the size of the left subtree ($k$ nodes), we can partition the preorder array into left and right subtree segments and recursively build the tree.

```text
preorder: [ Root |  --- Left Subtree (k nodes) ---  |  --- Right Subtree ---  ]
              ↓
inorder:  [ --- Left Subtree (k nodes) --- | Root | --- Right Subtree ---  ]
```

#### Standard Optimal Solution (Pointers + Hash Map)

A hash map precomputes indices in `inorder` for $O(1)$ root lookups; passing index ranges avoids array copying, achieving optimal $O(n)$ time and $O(n)$ space.

```python
def build_tree(preorder, inorder):
    # 1. Map values to indices in inorder for O(1) root lookups
    val_to_idx = {val: i for i, val in enumerate(inorder)}

    def helper(pre_left, pre_right, in_left, in_right):
        if pre_left > pre_right:
            return None

        # The first element in the preorder segment is the root
        root_val = preorder[pre_left]
        root = TreeNode(root_val)

        # Locate root in inorder and determine left subtree size
        in_root_idx = val_to_idx[root_val]
        left_size = in_root_idx - in_left

        # Recursively construct left and right subtrees
        root.left = helper(
            pre_left + 1, pre_left + left_size,
            in_left, in_root_idx - 1
        )
        root.right = helper(
            pre_left + left_size + 1, pre_right,
            in_root_idx + 1, in_right
        )

        return root

    return helper(0, len(preorder) - 1, 0, len(inorder) - 1)
```

#### Whiteboard 5-Line Slice Version

If slicing overhead is acceptable, Python slices yield an ultra-compact 5-line implementation:

```python
def build_tree_slice(preorder, inorder):
    if not preorder:
        return None

    root_val = preorder[0]
    root = TreeNode(root_val)

    # Index in inorder equals the number of nodes in the left subtree
    k = inorder.index(root_val)

    root.left = build_tree_slice(preorder[1 : 1 + k], inorder[:k])
    root.right = build_tree_slice(preorder[1 + k :], inorder[k + 1 :])

    return root
```

> **3-Step Memory Formula**:
> 1. Take the first preorder element as the root.
> 2. Find the root in inorder to get left subtree length $k$.
> 3. Skip the root in preorder and take $k$ elements for left, remainder for right.

Used by: Construct Binary Tree from Preorder and Inorder Traversal (105), Serialize and Deserialize Binary Tree (297) (preorder with explicit null markers naturally defines boundaries).

---

### 5. Recomputed Heights and Sentinel Pruning (Bottom-Up Check)

A naive Balanced Binary Tree solution recalculates `height()` at every node, degrading to $O(n^2)$ time on skewed trees.

The optimal approach uses **bottom-up postorder recursion**: compute heights while simultaneously verifying balance. Use `-1` as a **sentinel signal** for imbalance. As soon as any subtree is unbalanced, `-1` propagates immediately upward, aborting redundant work and reducing time to $O(n)$.

```python
def is_balanced(root):
    def check(node):
        if not node:
            return 0  # Empty node has height 0

        left_h = check(node.left)
        if left_h == -1:  # Left subtree unbalanced -> prune
            return -1

        right_h = check(node.right)
        if right_h == -1:  # Right subtree unbalanced -> prune
            return -1

        if abs(left_h - right_h) > 1:  # Current node unbalanced
            return -1

        return 1 + max(left_h, right_h)  # Return true height when balanced

    return check(root) != -1
```

- **Core Intuition**: Bottom-up postorder traversal naturally has access to all subtree information; returning `-1` acts as a circuit-breaker as soon as a violation is found.

Used by: Balanced Binary Tree (110).

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

Inverting a binary tree is most natural with recursion: swap the current node's left and right children, where each child is itself recursively inverted.

| Item | Value |
|---|---|
| Composed patterns | Tree traversal + local pointer swap |
| Key operation | `root.left, root.right = invert(root.right), invert(root.left)` |
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
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

> **Note**: In interviews, 4-line recursion is the most concise and clean. If an iterative approach is explicitly asked, use a BFS queue or stack to swap children level by level or node by node.

</details>

### 2. Maximum Depth of Binary Tree

The maximum depth of a binary tree equals 1 (for the root) plus the maximum of the left and right subtree depths. Base case: an empty node has depth 0.

| Item | Value |
|---|---|
| Composed patterns | Bottom-up height recursion |
| Key recurrence | `1 + max(maxDepth(root.left), maxDepth(root.right))` |
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

> **Note**: Level-order BFS can also be used (incrementing `depth += 1` per level), using `O(w)` space.

</details>

### 3. Diameter of Binary Tree

Use postorder DFS to compute subtree depth. For any node, the longest path passing through it has `left_depth + right_depth` edges. Maintain the global maximum diameter in an instance variable while returning the single-branch maximum depth `1 + max(left_depth, right_depth)` up to the parent.

| Item | Value |
|---|---|
| Composed patterns | Bottom-up return value + global best |
| Return / Answer quantity | Single-branch maximum depth / Maximum tree diameter |
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
        self.diameter = 0

        def depth(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left = depth(node.left)
            right = depth(node.right)
            # Update global diameter with the longest path passing through this node
            self.diameter = max(self.diameter, left + right)
            # Return single-branch maximum depth to parent
            return 1 + max(left, right)

        depth(root)
        return self.diameter
```

</details>

### 4. Balanced Binary Tree

Compute tree height bottom-up. If either subtree is unbalanced (returns `-1`) or the height difference exceeds 1, return `-1` immediately to prune; otherwise return the true height `1 + max(left, right)`.

| Item | Value |
|---|---|
| Composed patterns | Postorder recursion + sentinel pruning |
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
        def check(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            left = check(node.left)
            if left == -1:
                return -1

            right = check(node.right)
            if right == -1:
                return -1

            if abs(left - right) > 1:
                return -1

            return 1 + max(left, right)

        return check(root) != -1
```

</details>

### 5. Same Tree

This is the base structural-comparison problem. Each stack element contains two nodes at the same structural position. Two null nodes match; one null node or unequal values fail immediately.

| Item | Value |
|---|---|
### 5. Same Tree

Compare two trees recursively: root values must match, and left subtrees must match left subtrees, right subtrees must match right subtrees.

| Item | Value |
|---|---|
| Composed patterns | Dual-tree parallel recursion |
| Key condition | `p.val == q.val and isSame(p.left, q.left) and isSame(p.right, q.right)` |
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
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

</details>

### 6. Subtree of Another Tree

Check recursively: is the current tree identical to `subRoot`? If not, search in the left subtree or right subtree.

| Item | Value |
|---|---|
| Composed patterns | Double recursion (Tree search + structural comparison) |
| Operation | `self.isSame(root, subRoot)` |
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
        if not root:
            return False
        if self.isSame(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def isSame(self, s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:
        if not s and not t:
            return True
        if not s or not t or s.val != t.val:
            return False
        return self.isSame(s.left, t.left) and self.isSame(s.right, t.right)
```

</details>

### 7. Lowest Common Ancestor of a BST

Use the BST ordering invariant to descend in one direction: if both `p` and `q` are smaller than the current node, the LCA must lie in the left subtree; if both are larger, it must lie in the right subtree; otherwise, the current node is the split point (LCA).

| Item | Value |
|---|---|
| Composed patterns | BST ordering invariant + single-path descent |
| Branches | Both left, both right, split point |
| Time / Space | `O(h) / O(1)` |

#### Quick Coding: Lowest Common Ancestor of a BST

```python
def lowestCommonAncestor(root, p, q):
    ...
```

<details>
<summary>Reference answer</summary>

```python
class Solution:
    def lowestCommonAncestor(
        self, root: TreeNode, p: TreeNode, q: TreeNode
    ) -> TreeNode:
        curr = root
        while curr:
            if p.val < curr.val and q.val < curr.val:
                curr = curr.left
            elif p.val > curr.val and q.val > curr.val:
                curr = curr.right
            else:
                return curr
        return root
```

> **Note**: The recursive version is also extremely clean (4 lines):

```python
if p.val < root.val and q.val < root.val:
    return self.lowestCommonAncestor(root.left, p, q)
if p.val > root.val and q.val > root.val:
    return self.lowestCommonAncestor(root.right, p, q)
return root
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

Traverse downward recursively from the root while tracking `max_val` seen along the current path. If `node.val >= max_val`, count it as a good node and propagate the updated maximum downward.

| Item | Value |
|---|---|
| Composed patterns | Top-down path state propagation |
| Key state | `cur_max = max(max_val, node.val)` |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Count Good Nodes in Binary Tree

```python
def goodNodes(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node: Optional[TreeNode], max_val: int) -> int:
            if not node:
                return 0
            count = 1 if node.val >= max_val else 0
            cur_max = max(max_val, node.val)
            return count + dfs(node.left, cur_max) + dfs(node.right, cur_max)

        return dfs(root, root.val)
```

</details>

### 11. Validate Binary Search Tree

Propagate the open interval `(low, high)` that each node must satisfy top-down. Going left tightens the upper bound; going right tightens the lower bound.

| Item | Value |
|---|---|
| Composed patterns | BST ordering invariant + interval propagation |
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
from typing import Optional


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node: Optional[TreeNode], low: float, high: float) -> bool:
            if not node:
                return True
            if not (low < node.val < high):
                return False
            return validate(node.left, low, node.val) and validate(node.right, node.val, high)

        return validate(root, float('-inf'), float('inf'))
```

</details>

### 12. Kth Smallest Element in a BST

A BST's inorder traversal is strictly ascending. Through inorder traversal, the kth node visited is the kth smallest element.

| Item | Value |
|---|---|
| Composed patterns | BST inorder monotonicity |
| State | Decrement counter `k` to 0 for early stopping |
| Time / Space | `O(h + k) / O(h)` |

#### Quick Coding: Kth Smallest Element in a BST

```python
def kthSmallest(root, k):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.k = k
        self.res = None

        def inorder(node: Optional[TreeNode]):
            if not node or self.res is not None:
                return
            inorder(node.left)
            self.k -= 1
            if self.k == 0:
                self.res = node.val
                return
            inorder(node.right)

        inorder(root)
        return self.res
```

> **Note**: An explicit stack can also be used for early exit on the kth pop:

```python
stack = []
curr = root
while stack or curr:
    while curr:
        stack.append(curr)
        curr = curr.left
    curr = stack.pop()
    k -= 1
    if k == 0:
        return curr.val
    curr = curr.right
```

</details>

### 13. Construct Binary Tree from Preorder and Inorder Traversal

The most intuitive and memorable approach for this problem is **divide-and-conquer recursion with a hash map**.
Iterative stack backtracing with `parent` tracking is notoriously prone to pointer bugs under interview pressure; the essence of the recursive approach is simply: **"Preorder gives the root; inorder splits into left and right subtrees."**

| Item | Value |
|---|---|
| Composed patterns | Divide-and-conquer recursion + hash map lookup |
| State | Preorder range `[pre_left, pre_right]` and Inorder range `[in_left, in_right]` |
| Time / Space | `O(n) / O(n)` |

#### Core Intuition (Remember in 3 Seconds)

* **Preorder:** `[root, ...all left subtree nodes..., ...all right subtree nodes...]`
  * Role: **The first element is always the root of the current subtree.**
* **Inorder:** `[...all left subtree nodes..., root, ...all right subtree nodes...]`
  * Role: **Locating the root splits the array into left and right subtrees.**

Once we know the size of the left subtree ($k$ nodes), we can partition the preorder array into left and right subtree segments and recursively construct both subtrees.

```text
preorder: [ Root |  --- Left Subtree (k nodes) ---  |  --- Right Subtree ---  ]
              ↓
inorder:  [ --- Left Subtree (k nodes) --- | Root | --- Right Subtree ---  ]
```

#### Quick Coding: Construct Binary Tree from Preorder and Inorder Traversal

```python
def buildTree(preorder, inorder):
    ...
```

<details>
<summary>Reference answer</summary>

##### 1. Standard Optimal Solution (Pointers + Hash Map)

A hash map precomputes indices in `inorder` for $O(1)$ root lookups; passing index ranges avoids array copying, achieving optimal $O(n)$ time and $O(n)$ space.

```python
from typing import List, Optional


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # 1. Precompute root positions in inorder for O(1) lookups
        val_to_idx = {val: i for i, val in enumerate(inorder)}

        def helper(pre_left: int, pre_right: int, in_left: int, in_right: int) -> Optional[TreeNode]:
            if pre_left > pre_right:
                return None

            # First element in the preorder range is the root
            root_val = preorder[pre_left]
            root = TreeNode(root_val)

            # Locate root in inorder and determine left subtree size
            in_root_idx = val_to_idx[root_val]
            left_size = in_root_idx - in_left

            # Recursively construct left and right subtrees
            root.left = helper(
                pre_left + 1, pre_left + left_size,
                in_left, in_root_idx - 1
            )
            root.right = helper(
                pre_left + left_size + 1, pre_right,
                in_root_idx + 1, in_right
            )

            return root

        return helper(0, len(preorder) - 1, 0, len(inorder) - 1)
```

##### 2. Whiteboard 5-Line Slice Version

If slicing overhead is acceptable, Python slices yield an ultra-compact 5-line implementation:

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None

        root_val = preorder[0]
        root = TreeNode(root_val)

        # Index in inorder equals the number of nodes in the left subtree
        k = inorder.index(root_val)

        root.left = self.buildTree(preorder[1 : 1 + k], inorder[:k])
        root.right = self.buildTree(preorder[1 + k :], inorder[k + 1 :])

        return root
```

> **3-Step Memory Formula**:
> 1. Take the first preorder element as the root.
> 2. Find the root in inorder to get left subtree length $k$.
> 3. Skip the root in preorder and take $k$ elements for left, remainder for right.

</details>

The demo below steps through the monotonic stack iterative construction (provided for deeper understanding of stack state transformations; in interviews, the recursive divide-and-conquer approach above is strongly recommended).

```build-tree-demo
```

### 14. Binary Tree Maximum Path Sum

This belongs to the same "Bottom-Up Return Value + Global Best" Tree DP pattern as Diameter of Binary Tree. The helper returns the single-branch downward gain (cannot branch), while `self.max_sum` records the complete path joining both branches at the current node. Negative gains are clamped to `0`.

| Item | Value |
|---|---|
| Composed pattern | Bottom-up return value + global best |
| Return / answer quantity | One-sided downward gain / maximum path with any endpoints |
| Time / Space | `O(n) / O(h)` |

#### Quick Coding: Binary Tree Maximum Path Sum

```python
def maxPathSum(root):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import Optional


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')

        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            # Clamp negative gains to 0 (omit negative subtrees)
            max_left = max(0, dfs(node.left))
            max_right = max(0, dfs(node.right))

            # Update global maximum path turning at the current node
            path_sum = node.val + max_left + max_right
            self.max_sum = max(self.max_sum, path_sum)

            # Return single-branch maximum gain to parent
            return node.val + max(max_left, max_right)

        dfs(root)
        return self.max_sum
```

Each call to `dfs` naturally manages its own stack frame, so `max_left`/`max_right` are already the values computed by its children. `self.max_sum` updates whenever a new path sum exceeds the previous record, and once `dfs` returns, it holds the final answer.

</details>

### 15. Serialize and Deserialize Binary Tree

The preorder sequence records every node value and writes a null marker for every null child. Recursion reads much more clearly than an iterative stack: `serialize` is just a preorder traversal, and `deserialize` consumes tokens in the same order using an iterator.

| Item | Value |
|---|---|
| Composed pattern | Preorder reconstruction with null markers |
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
        res = []

        def dfs(node: Optional[TreeNode]):
            if not node:
                res.append("N")  # "N" marks a null node
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(res)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        vals = iter(data.split(","))

        def dfs():
            val = next(vals)
            if val == "N":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()
```

The `dfs` in `serialize` is an ordinary preorder traversal, except null children also get written; otherwise `deserialize` would have no way to tell whether a child exists.

`deserialize` wraps the token list in an iterator with `iter()`, so each `next(vals)` pulls the next token in exactly the order `serialize` produced it. It builds the current node first, then recurses into the left subtree, then the right, the same order `serialize` used, so every `next()` call is guaranteed to return the correct next token.

This problem and Preorder+Inorder reconstruction are the same broad idea: preorder fixes the next node, and some mechanism tells you where its subtree ends. There, that mechanism is the split position in the inorder sequence, which needs an explicit stack to track. Here, the mechanism is an explicit null marker, and the recursive call itself already knows where the subtree ends, so no extra state is required.

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

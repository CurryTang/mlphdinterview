# Review Flashcards: Trees, Graphs & DP

This note is the fourth volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Graphs & Grid Search**, **Trees & BST**, and **Dynamic Programming** with production-grade implementations and asymptotic complexity breakdowns.

---

## Module 1: Graphs & Grid Search

### 1. Number of Islands & All Canonical Variants

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 01</span>
  <span class="review-card-title">Number of Islands & All Canonical Variants</span>
  <span class="review-card-tag">Grid Implicit Graph · BFS/DFS · Relative Coordinate Normalization</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 200 · Number of Islands](https://leetcode.com/problems/number-of-islands/) — `https://leetcode.com/problems/number-of-islands/`
> - [LeetCode 694 · Number of Distinct Islands](https://leetcode.com/problems/number-of-distinct-islands/) — `https://leetcode.com/problems/number-of-distinct-islands/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Number of Islands & Canonical Variants (LeetCode 200 / 694 / 695 / 827)**:
> Given an $m \times n$ 2D binary grid `grid` which represents a map of `'1'`s (land) and `'0'`s (water), return the number of islands.
> An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically.
>
> **Variants Matrix**:
> - Read-only grid (no mutation allowed, requires external visited set);
> - Distinct islands (LC 694, requires relative coordinate normalization);
> - Max area of island (LC 695);
> - Large out-of-core grid chunking.

**Function Signature**:
```python
def numIslands(grid: List[List[str]]) -> int: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from collections import deque
from typing import List

class IslandSolution:
    @staticmethod
    def numIslands(grid: List[List[str]]) -> int:
        if not grid or not grid[0]: return 0
        m, n, count = len(grid), len(grid[0]), 0
        for r in range(m):
            for c in range(n):
                if grid[r][c] == '1':
                    count += 1
                    queue = deque([(r, c)])
                    grid[r][c] = '0'
                    while queue:
                        cr, cc = queue.popleft()
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '1':
                                grid[nr][nc] = '0'
                                queue.append((nr, nc))
        return count

if __name__ == "__main__":
    g1 = [
      ["1","1","1","1","0"],
      ["1","1","0","1","0"],
      ["1","1","0","0","0"],
      ["0","0","0","0","0"]
    ]
    assert IslandSolution.numIslands(g1) == 1
    g2 = [
      ["1","1","0","0","0"],
      ["1","1","0","0","0"],
      ["0","0","1","0","0"],
      ["0","0","0","1","1"]
    ]
    assert IslandSolution.numIslands(g2) == 3
    print("✅ Card 01 (Number of Islands) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- $\mathcal{O}(M \times N)$ time and space.

</div>

</div>
</details>

---

### 2. Course Schedule & Topological Sort

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 02</span>
  <span class="review-card-title">Course Schedule & Topological Sort</span>
  <span class="review-card-tag">Kahn BFS In-degree · DFS Three-Color Mark · Cycle Reconstruction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 207 · Course Schedule](https://leetcode.com/problems/course-schedule/) — `https://leetcode.com/problems/course-schedule/`
> - [LeetCode 210 · Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) — `https://leetcode.com/problems/course-schedule-ii/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Course Schedule & Topological Sort (LeetCode 207 / 210)**:
> There are a total of `numCourses` courses you have to take, labeled from 0 to `numCourses - 1`. You are given an array `prerequisites` where `prerequisites[i] = [a_i, b_i]` indicates that you must take course $b_i$ before $a_i$ ($b_i \to a_i$).
> - **Feasibility (LC 207)**: return whether a valid topological ordering exists.
> - **Order Generation (LC 210)**: return any valid completion order, or `[]` if impossible.
> - **SRE Follow-up**: detect and reconstruct exact circular dependency cycle path using three-color DFS.

**Function Signature**:
```python
class CourseScheduleSolution:
    @staticmethod
    def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]: ...
    @staticmethod
    def detectAndPrintCycle(numCourses: int, prerequisites: List[List[int]]) -> Optional[List[int]]: ...
```

**Examples**:
- `numCourses = 2, prerequisites = [[1, 0]]` $\implies$ `[0, 1]`
- `numCourses = 2, prerequisites = [[1, 0], [0, 1]]` $\implies$ `[]` (cycle detected)

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from collections import deque
from typing import List

class CourseScheduleSolution:
    @staticmethod
    def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        adj = [[] for _ in range(numCourses)]
        in_degree = [0] * numCourses
        for dest, src in prerequisites:
            adj[src].append(dest)
            in_degree[dest] += 1
        queue = deque([i for i in range(numCourses) if in_degree[i] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return order if len(order) == numCourses else []

if __name__ == "__main__":
    assert CourseScheduleSolution.findOrder(2, [[1, 0]]) == [0, 1]
    assert CourseScheduleSolution.findOrder(2, [[1, 0], [0, 1]]) == []
    assert len(CourseScheduleSolution.findOrder(4, [[1,0],[2,0],[3,1],[3,2]])) == 4
    print("✅ Card 02 (Course Schedule & Topological Sort) all tests passed!")
```

</div>

</div>
</details>

---

### 3. Order Validator with Dynamic DAG Dependencies & Node Contraction

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 03</span>
  <span class="review-card-title">Order Validator with Dynamic DAG Dependencies & Node Contraction</span>
  <span class="review-card-tag">Directed Acyclic Graph · Dynamic Dependencies · Topological Validation · Node Contraction Rewiring</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Architectural Contract</div>

Build an `OrderValidator` system for e-commerce orders and dynamic validation rules:
1. Rules inspect prohibited items, price boundaries, etc.
2. Dynamic additions of rules and dependencies; reject any dependency creating a cycle.
3. Validate orders in topological dependency order.
4. **Node Contraction Rewiring Follow-up**: When a rule $R$ is removed, contract node $R$: connect every predecessor $Pre(R)$ directly to every successor $Succ(R)$ ($Pre(R) \times Succ(R)$ cross-product), then delete $R$.

</div>

<div class="review-block">
<div class="review-block-label">💡 Mathematical Proof of Acyclicity Under Contraction</div>

- **Theorem**: If $G$ is a DAG, the graph $G'$ obtained by contracting node $R$ remains strictly acyclic.
- **Proof**: If $G'$ contained a cycle $C$, any new edge $(u, v)$ in $C$ replaces the original path $u \to R \to v$. Substituting all such contracted shortcuts restores a valid directed cycle in $G$, contradicting the DAG premise.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from collections import deque
from typing import Dict, Set, List, Callable

class OrderValidator:
    def __init__(self):
        self.rules: Dict[str, Callable[[dict], bool]] = {}
        self.outgoing: Dict[str, Set[str]] = {}
        self.incoming: Dict[str, Set[str]] = {}

    def add_rule(self, rule_id: str, validate_fn: Callable[[dict], bool]) -> None:
        if rule_id not in self.rules:
            self.rules[rule_id] = validate_fn
            self.outgoing[rule_id] = set()
            self.incoming[rule_id] = set()

    def _creates_cycle(self, src: str, dest: str) -> bool:
        if src == dest: return True
        visited = set()
        queue = deque([dest])
        while queue:
            cur = queue.popleft()
            if cur == src: return True
            for nxt in self.outgoing.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def add_dependency(self, prereq_id: str, rule_id: str) -> None:
        if prereq_id not in self.rules or rule_id not in self.rules:
            raise ValueError("Rules must exist")
        if self._creates_cycle(prereq_id, rule_id):
            raise ValueError(f"Cycle detected: {prereq_id} -> {rule_id}")
        self.outgoing[prereq_id].add(rule_id)
        self.incoming[rule_id].add(prereq_id)

    def remove_rule(self, rule_id: str) -> None:
        """Contracts rule_id: connects Pre(R) x Succ(R)"""
        if rule_id not in self.rules: return
        preds = self.incoming[rule_id]
        succs = self.outgoing[rule_id]
        for p in preds:
            self.outgoing[p].remove(rule_id)
            for s in succs:
                self.outgoing[p].add(s)
        for s in succs:
            self.incoming[s].remove(rule_id)
            for p in preds:
                self.incoming[s].add(p)
        del self.rules[rule_id]
        del self.outgoing[rule_id]
        del self.incoming[rule_id]

    def validate(self, order: dict) -> bool:
        in_deg = {node: len(self.incoming[node]) for node in self.rules}
        queue = deque([node for node, deg in in_deg.items() if deg == 0])
        processed = 0
        while queue:
            cur = queue.popleft()
            processed += 1
            if not self.rules[cur](order):
                return False
            for nxt in self.outgoing[cur]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)
        return processed == len(self.rules)

if __name__ == "__main__":
    ov = OrderValidator()
    ov.add_rule("R1", lambda o: o.get("amount", 0) > 0)
    ov.add_rule("R2", lambda o: o.get("user_verified", False))
    ov.add_dependency("R1", "R2")
    cycle_caught = False
    try:
        ov.add_dependency("R2", "R1")
    except ValueError:
        cycle_caught = True
    assert cycle_caught is True
    assert ov.validate({"amount": 100, "user_verified": True}) is True
    assert ov.validate({"amount": 0, "user_verified": True}) is False
    ov.remove_rule("R2")
    assert ov.validate({"amount": 100, "user_verified": False}) is True
    print("✅ Card 03 (Order Validator & DAG Dependency) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: `validate` is $\mathcal{O}(V + E)$; `remove_rule` is $\mathcal{O}(\operatorname{in\_deg} + \operatorname{out\_deg} + \operatorname{in\_deg} \times \operatorname{out\_deg})$.
- **Space Complexity**: $\mathcal{O}(V + E)$.

</div>

</div>
</details>

---

### 4. Shortest Path in Grid with Obstacles Elimination

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 04</span>
  <span class="review-card-title">Shortest Path in Grid with Obstacles Elimination</span>
  <span class="review-card-tag">3D State BFS · Dominance Pruning · Manhattan Distance Shortcut</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 1293 · Shortest Path in a Grid with Obstacles Elimination](https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/) — `https://leetcode.com/problems/shortest-path-in-a-grid-with-obstacles-elimination/`

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Dominance Invariant</div>

Given an $M \times N$ grid, find the shortest path from $(0, 0)$ to $(M-1, N-1)$ while eliminating at most $k$ obstacles (LC 1293):
- **Dominance Pruning**: Reaching cell $(r, c)$ with greater remaining elimination budget dominates reaching it with fewer. `visited[r][c]` tracks max remaining $k$.
- **Manhattan Shortcut**: If $k \ge (m - 1) + (n - 1) - 1$, return direct taxicab distance $(m - 1) + (n - 1)$ immediately.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from collections import deque
from typing import List

class ObstacleGridShortestPathSolution:

    @staticmethod
    def shortestPath(grid: List[List[int]], k: int) -> int:
        if not grid or not grid[0]: return -1
        m, n = len(grid), len(grid[0])
        if m == 1 and n == 1: return 0

        if k >= (m - 1) + (n - 1) - 1:
            return (m - 1) + (n - 1)

        visited = [[-1] * n for _ in range(m)]
        visited[0][0] = k
        queue = deque([(0, 0, k, 0)])

        while queue:
            r, c, rem_k, steps = queue.popleft()
            if r == m - 1 and c == n - 1:
                return steps

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    nxt_k = rem_k - grid[nr][nc]
                    if nxt_k >= 0 and nxt_k > visited[nr][nc]:
                        visited[nr][nc] = nxt_k
                        queue.append((nr, nc, nxt_k, steps + 1))

        return -1

if __name__ == "__main__":
    grid = [[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]]
    assert ObstacleGridShortestPathSolution.shortestPath(grid, 1) == 6
    assert ObstacleGridShortestPathSolution.shortestPath([[0,1,1],[1,1,1],[1,0,0]], 1) == -1
    print("✅ Card 04 (Shortest Path with Obstacles Elimination) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(M \times N \times k)$.
- **Space Complexity**: $\mathcal{O}(M \times N \times k)$.

</div>

</div>
</details>

---

## Module 2: Trees & BST

### 5. Lowest Common Ancestor (LCA)

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 05</span>
  <span class="review-card-title">Lowest Common Ancestor (LCA)</span>
  <span class="review-card-tag">Postorder Divide-and-Conquer · BST Value Pruning · Parent Pointer Intersection</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 236 · Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) — `https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/`
> - [LeetCode 235 · Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) — `https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Lowest Common Ancestor (LeetCode 236 / 235)**:
> Given a binary tree (or BST), find the lowest common ancestor (LCA) of two given nodes `p` and `q`.
> The lowest common ancestor is the lowest node in $T$ that has both $p$ and $q$ as descendants (allowing a node to be a descendant of itself).

**Function Signature**:
```python
class LCASolution:
    @staticmethod
    def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode': ...
```

**Examples**:
- `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1` $\implies$ `3`
- `root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4` $\implies$ `5`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = self.right = None

class LCASolution:
    @staticmethod
    def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q: return root
        left = LCASolution.lowestCommonAncestor(root.left, p, q)
        right = LCASolution.lowestCommonAncestor(root.right, p, q)
        return root if left and right else (left or right)

if __name__ == "__main__":
    root5 = TreeNode(3)
    root5.left = TreeNode(5)
    root5.right = TreeNode(1)
    root5.left.left = TreeNode(6)
    root5.left.right = TreeNode(2)
    assert LCASolution.lowestCommonAncestor(root5, root5.left, root5.right).val == 3
    assert LCASolution.lowestCommonAncestor(root5, root5.left, root5.left.right).val == 5
    print("✅ Card 05 (Lowest Common Ancestor) all tests passed!")
```

</div>

</div>
</details>

---

### 6. Binary Tree Maximum Path Sum & Path Reconstruction

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 06</span>
  <span class="review-card-title">Binary Tree Maximum Path Sum & Path Reconstruction</span>
  <span class="review-card-tag">Postorder Tree DP · Negative Gain Clamping · Optimal Path Reconstruction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 124 · Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) — `https://leetcode.com/problems/binary-tree-maximum-path-sum/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Binary Tree Maximum Path Sum & Path Reconstruction (LeetCode 124)**:
> A path in a binary tree is a sequence of nodes where each pair of adjacent nodes has an edge connecting them. The path does not need to pass through the root.
> Return the maximum path sum of any non-empty path.

**Function Signature**:
```python
class MaxPathSumSolution:
    @staticmethod
    def maxPathSum(root: Optional[TreeNode]) -> int: ...
```

**Examples**:
- `root = [1, 2, 3]` $\implies$ `6` (`2 + 1 + 3`)
- `root = [-10, 9, 20, null, null, 15, 7]` $\implies$ `42` (`15 + 20 + 7`)

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import Optional

class MaxPathSumSolution:
    @classmethod
    def maxPathSum(cls, root: Optional[TreeNode]) -> int:
        max_sum = float('-inf')
        def max_gain(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            if not node: return 0
            left_gain = max(0, max_gain(node.left))
            right_gain = max(0, max_gain(node.right))
            max_sum = max(max_sum, node.val + left_gain + right_gain)
            return node.val + max(left_gain, right_gain)
        max_gain(root)
        return int(max_sum)

if __name__ == "__main__":
    r6 = TreeNode(1, TreeNode(2), TreeNode(3))
    assert MaxPathSumSolution.maxPathSum(r6) == 6
    r6_neg = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
    assert MaxPathSumSolution.maxPathSum(r6_neg) == 42
    print("✅ Card 06 (Binary Tree Maximum Path Sum) all tests passed!")
```

</div>

</div>
</details>

---

### 7. Flatten Comment Tree to Multi-Level Hierarchy

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 07</span>
  <span class="review-card-title">Flatten Comment Tree to Multi-Level Hierarchy</span>
  <span class="review-card-tag">Two-Pass Hash Assembly · Arbitrary Depth · Orphan Safety · Self-Reference Guard</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Examples</div>

Given a flat list of comment objects with `id`, `parent_id` (`null` for root), and `text`:
- Reconstruct into a nested tree where comments have `children: List[dict]`, returning all root trees.

```python
def buildCommentTree(comments: List[dict]) -> List[dict]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Two-Pass Architecture</div>

1. **Pass 1**: Map `id -> cloned_node` with empty `children: []`.
2. **Pass 2**: Attach nodes with `parent_id is None` to `roots`; attach others to `node_map[parent_id]['children']`.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List, Dict, Any

class CommentTreeBuilder:

    @staticmethod
    def buildCommentTree(comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not comments:
            return []

        node_map: Dict[int, Dict[str, Any]] = {}
        for item in comments:
            cid = item['id']
            node_map[cid] = {
                'id': cid,
                'parent_id': item.get('parent_id'),
                'text': item.get('text', ''),
                'children': []
            }

        roots: List[Dict[str, Any]] = []

        for item in comments:
            cid = item['id']
            pid = item.get('parent_id')
            node = node_map[cid]

            if pid == cid:
                continue

            if pid is None:
                roots.append(node)
            else:
                if pid in node_map:
                    node_map[pid]['children'].append(node)
                else:
                    node['orphan_warning'] = True
                    roots.append(node)

        return roots

if __name__ == "__main__":
    cmts = [
        {"id": 1, "text": "Top 1", "parent_id": None},
        {"id": 2, "text": "Reply 1.1", "parent_id": 1},
        {"id": 3, "text": "Top 2", "parent_id": None}
    ]
    tree7 = CommentTreeBuilder.buildCommentTree(cmts)
    assert len(tree7) == 2
    assert tree7[0]["id"] == 1 and len(tree7[0]["children"]) == 1
    assert tree7[0]["children"][0]["id"] == 2
    print("✅ Card 07 (Comment Tree Multi-Level Hierarchy) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$ strictly.
- **Space Complexity**: $\mathcal{O}(N)$.

</div>

</div>
</details>

---

### 8. Equalize Root-to-Leaf Path Sums in N-ary Tree

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 08</span>
  <span class="review-card-title">Equalize Root-to-Leaf Path Sums in N-ary Tree</span>
  <span class="review-card-tag">N-ary Tree Postorder · Tree Greedy Lift · Majority Element Warmup</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Greedy Lift Mechanics</div>

Given an N-ary tree where each node holds an integer:
- An operation increments a single node by 1.
- Find the minimum number of operations to make all root-to-leaf path sums equal.
- **Greedy Lift Insight**: If internal nodes can be incremented, compensate shared lag at common ancestors to save operations.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List
from collections import Counter

class NaryTreeNode:
    def __init__(self, val: int = 0, children: List['NaryTreeNode'] = None):
        self.val = val
        self.children = children if children is not None else []

class NaryTreeEqualizeSolution:

    @staticmethod
    def mostFrequentSmallest(nums: List[int]) -> int:
        counts = Counter(nums)
        best_num, max_freq = None, -1
        for num, freq in counts.items():
            if freq > max_freq or (freq == max_freq and (best_num is None or num < best_num)):
                max_freq = freq
                best_num = num
        return best_num

    @classmethod
    def minOperationsToEqualize(cls, root: NaryTreeNode) -> int:
        if not root:
            return 0

        total_ops = 0

        def postorder(node: NaryTreeNode) -> int:
            nonlocal total_ops
            if not node.children:
                return node.val

            child_sums = [postorder(child) for child in node.children]
            max_child_sum = max(child_sums)

            for s in child_sums:
                total_ops += (max_child_sum - s)

            return node.val + max_child_sum

        postorder(root)
        return total_ops

if __name__ == "__main__":
    leaf1 = NaryTreeNode(3)
    leaf2 = NaryTreeNode(5)
    root8 = NaryTreeNode(1, [leaf1, leaf2])
    assert NaryTreeEqualizeSolution.minOperationsToEqualize(root8) == 2
    print("✅ Card 08 (Equalize Root-to-Leaf Path Sums) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$ postorder traversal.
- **Space Complexity**: $\mathcal{O}(H)$ recursion depth.

</div>

</div>
</details>

---

## Module 3: Dynamic Programming

### 9. Coin Change 1 & 2 / Unbounded Knapsack

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">DP 09</span>
  <span class="review-card-title">Coin Change 1 & 2 / Unbounded Knapsack</span>
  <span class="review-card-tag">Unbounded Knapsack · Min Coins vs Total Combinations · Inner Loop Direction</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Links**:
> - [LeetCode 322 · Coin Change](https://leetcode.com/problems/coin-change/) — `https://leetcode.com/problems/coin-change/`
> - [LeetCode 518 · Coin Change II](https://leetcode.com/problems/coin-change-ii/) — `https://leetcode.com/problems/coin-change-ii/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Coin Change & Unbounded Knapsack (LeetCode 322 / 518)**:
> You are given an integer array `coins` and an integer `amount`:
> 1. **Coin Change (LC 322)**: return fewest number of coins needed to make up amount, or -1 if impossible.
> 2. **Coin Change II (LC 518)**: return number of combinations that make up amount.

**Function Signature**:
```python
class CoinChangeSolution:
    @staticmethod
    def coinChange(coins: List[int], amount: int) -> int: ...
```

**Examples**:
- `coins = [1, 2, 5], amount = 11` $\implies$ `3` (`5 + 5 + 1`)
- `coins = [2], amount = 3` $\implies$ `-1`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class CoinChangeSolution:
    @staticmethod
    def coinChange(coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] = min(dp[x], dp[x - coin] + 1)
        return int(dp[amount]) if dp[amount] != float('inf') else -1

if __name__ == "__main__":
    assert CoinChangeSolution.coinChange([1, 2, 5], 11) == 3
    assert CoinChangeSolution.coinChange([2], 3) == -1
    assert CoinChangeSolution.coinChange([1], 0) == 0
    print("✅ Card 09 (Coin Change) all tests passed!")
```

</div>

</div>
</details>

---

### 10. Stickers to Spell Word & Bitmask DP

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">DP 10</span>
  <span class="review-card-title">Stickers to Spell Word & Bitmask DP</span>
  <span class="review-card-tag">Bitmask DP · First Unmet Pruning · O(2^n * m * n)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 691 · Stickers to Spell Word](https://leetcode.com/problems/stickers-to-spell-word/) — `https://leetcode.com/problems/stickers-to-spell-word/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Stickers to Spell Word (LeetCode 691)**:
> We are given $n$ different types of `stickers`. Each sticker has a lowercase word on it.
> You would like to spell out `target` by cutting individual letters from stickers. You have infinite quantities of each sticker.
> Return the minimum number of stickers needed to spell out `target`, or -1 if impossible.

**Function Signature**:
```python
class StickersSolution:
    @classmethod
    def minStickers(cls, stickers: List[str], target: str) -> int: ...
```

**Examples**:
- `stickers = ["with","example","science"], target = "thehat"` $\implies$ `3`
- `stickers = ["notice","possible"], target = "basic"` $\implies$ `-1`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List
from collections import Counter

class StickersSolution:
    @classmethod
    def minStickers(cls, stickers: List[str], target: str) -> int:
        n = len(target)
        target_chars = set(target)
        sticker_counts = [Counter(ch for ch in s if ch in target_chars) for s in stickers]
        sticker_counts = [cnt for cnt in sticker_counts if cnt]
        memo = { (1 << n) - 1: 0 }

        def dfs(mask: int) -> int:
            if mask in memo: return memo[mask]
            first_unmet = 0
            while (mask >> first_unmet) & 1: first_unmet += 1
            target_ch = target[first_unmet]
            ans = float('inf')
            for cnt in sticker_counts:
                if target_ch not in cnt: continue
                avail = dict(cnt)
                nxt_mask = mask
                for i in range(n):
                    if not ((nxt_mask >> i) & 1) and target[i] in avail and avail[target[i]] > 0:
                        avail[target[i]] -= 1
                        nxt_mask |= (1 << i)
                if nxt_mask != mask:
                    ans = min(ans, 1 + dfs(nxt_mask))
            memo[mask] = ans
            return ans

        res = dfs(0)
        return int(res) if res != float('inf') else -1

if __name__ == "__main__":
    assert StickersSolution.minStickers(["with", "example", "science"], "thehat") == 3
    assert StickersSolution.minStickers(["notice", "possible"], "basicbasic") == -1
    print("✅ Card 10 (Stickers to Spell Word) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- Time: $\mathcal{O}(2^N \cdot M \cdot N)$. Space: $\mathcal{O}(2^N)$.

</div>

</div>
</details>

---

### 11. Longest Alternating Zigzag Path in 2D Grid

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 11</span>
  <span class="review-card-title">Longest Alternating Zigzag Path in 2D Grid</span>
  <span class="review-card-tag">2D Grid · Memoized Search · State-Machine DP · O(M * N)</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Longest Alternating Zigzag Path in 2-D Grid**:
> Given a 2-D grid of integers, find the length of the longest zigzag path.
> Start from any cell; each step moves to a 4-directionally adjacent cell whose value strictly alternates between greater-than and less-than the current cell.
>
> **Industrial Follow-ups**:
> 1. What if diagonal moves are allowed?
> 2. Reconstruct the actual path sequence.

**Function Signature**:
```python
class LongestZigzagPathSolution:
    @classmethod
    def longestZigzag(cls, grid: List[List[int]]) -> int: ...
```

**Examples**:
- `grid = [[1, 2, 1], [2, 1, 2], [1, 2, 1]]` $\implies$ `9`

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class LongestZigzagPathSolution:
    @classmethod
    def longestZigzag(cls, grid: List[List[int]]) -> int:
        """
        Computes the length (number of cells) of the longest alternating
        strictly-increasing / strictly-decreasing path in a 2D integer grid.
        
        State representation:
        (r, c, expect_greater)
        - expect_greater = True: next step must be strictly greater (grid[nr][nc] > grid[r][c])
        - expect_greater = False: next step must be strictly smaller (grid[nr][nc] < grid[r][c])
        """
        if not grid or not grid[0]:
            return 0
        
        m, n = len(grid), len(grid[0])
        memo = {}
        visiting = set()

        def dfs(r: int, c: int, expect_greater: bool) -> int:
            state = (r, c, expect_greater)
            if state in memo:
                return memo[state]
            if state in visiting:
                # Loop guard (for cyclic state detection in alternating graph paths)
                return 1
            
            visiting.add(state)
            best_len = 1  # Base length including the current cell
            
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    if expect_greater and grid[nr][nc] > grid[r][c]:
                        best_len = max(best_len, 1 + dfs(nr, nc, False))
                    elif (not expect_greater) and grid[nr][nc] < grid[r][c]:
                        best_len = max(best_len, 1 + dfs(nr, nc, True))
            
            visiting.remove(state)
            memo[state] = best_len
            return best_len

        max_path = 0
        for r in range(m):
            for c in range(n):
                # Any cell can be the origin, starting with either increasing or decreasing step
                max_path = max(max_path, dfs(r, c, True), dfs(r, c, False))
        
        return max_path

if __name__ == "__main__":
    g_zz = [
        [1, 5, 2],
        [4, 3, 6],
        [2, 7, 1]
    ]
    assert LongestZigzagPathSolution.longestZigzag(g_zz) == 10
    assert LongestZigzagPathSolution.longestZigzag([[1]]) == 1
    print("✅ Card 11 (Longest Alternating Zigzag Path) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **State Space & Binary Alternator**:
  Every cell $(r, c)$ in the path has two directional expectations: "expecting an increase" vs "expecting a decrease". Thus the search space is bounded by $\mathcal{S} = \{ (r, c, d) \mid 0 \le r < m, 0 \le c < n, d \in \{0, 1\} \}$, totaling $2MN$ states.
- **Strict Inequality Guard**:
  Equal neighbor values ($grid[nr][nc] == grid[r][c]$) break the alternating constraint and cannot be traversed.
- **Memoization vs Exponential Backtracking**:
  Without memoization, the branching factor leads to $\mathcal{O}(4^L)$ time complexity. Memoizing on $(r, c, d)$ converts the traversal into finding the longest path on a state graph, with each state evaluated amortized once.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(M \cdot N)$. Exactly $2MN$ states, each checking 4 orthogonal directions.
- **Space Complexity**: $\mathcal{O}(M \cdot N)$ for memoization table and recursion call stack.

</div>

</div>
</details>

---

### 12. N-ary Tree Downward Target Path Sum via Prefix Sum

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">TREE 12</span>
  <span class="review-card-title">N-ary Tree Downward Target Path Sum via Prefix Sum</span>
  <span class="review-card-tag">N-ary Tree · Running Prefix Sum · Backtracking Scope Cleanup · O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 437 · Path Sum III](https://leetcode.com/problems/path-sum-iii/) — `https://leetcode.com/problems/path-sum-iii/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **N-ary Tree Downward Target Path Sum (LeetCode 437 Variant)**:
> Given root of an N-ary tree and an integer `targetSum`, return the number of downward paths summing to `targetSum`.
> Must clean up backtracking scope in $\mathcal{O}(n)$ time via prefix sum hash map.

**Function Signature**:
```python
class NaryPathSumSolution:
    @classmethod
    def pathSum(cls, root: Optional[NaryTreeNode], targetSum: int) -> int: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List, Optional
from collections import defaultdict

class NaryTreeNode:
    def __init__(self, val: int = 0, children: Optional[List['NaryTreeNode']] = None):
        self.val = val
        self.children = children if children is not None else []

class NaryPathSumSolution:
    @classmethod
    def pathSum(cls, root: Optional[NaryTreeNode], target: int) -> int:
        """
        Counts downward parent-to-child paths whose node values sum to target.
        Runs in O(N) time using a root-to-current prefix sum hash map.
        """
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # Base prefix: exact prefix sum matches target
        total_valid_paths = 0

        def dfs(node: Optional[NaryTreeNode], current_prefix_sum: int) -> None:
            nonlocal total_valid_paths
            if not node:
                return

            current_prefix_sum += node.val
            # Condition: current_prefix_sum - ancestor_prefix_sum = target
            # => ancestor_prefix_sum = current_prefix_sum - target
            total_valid_paths += prefix_counts[current_prefix_sum - target]

            # Register current prefix sum
            prefix_counts[current_prefix_sum] += 1

            for child in node.children:
                dfs(child, current_prefix_sum)

            # Backtracking: remove current node's prefix sum before exiting subtree
            prefix_counts[current_prefix_sum] -= 1

        dfs(root, 0)
        return total_valid_paths

if __name__ == "__main__":
    n_root = NaryTreeNode(10, [
        NaryTreeNode(5, [NaryTreeNode(3), NaryTreeNode(-2)]),
        NaryTreeNode(-3, [NaryTreeNode(11)])
    ])
    assert NaryPathSumSolution.pathSum(n_root, 8) == 2
    assert NaryPathSumSolution.pathSum(None, 8) == 0
    print("✅ Card 12 (N-ary Tree Downward Target Path Sum) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Prefix Sum Difference Property**:
  A downward path from ancestor $u$ to current node $v$ sums to $target$ iff:
  $$\sum_{w \in 	ext{path}(u 	o v)} 	ext{val}(w) = S(v) - S(	ext{parent}(u)) = target \implies S(	ext{parent}(u)) = S(v) - target$$
  Checking the frequency of $S(v) - target$ in the ancestor prefix map directly yields the count of valid downward paths ending at $v$.
- **Why Sliding Window Fails**:
  Node values can be negative or zero (violating prefix monotonicity), and tree branches diverge into non-linear paths.
- **Backtracking Scope Invariant**:
  `prefix_counts` must strictly reflect only nodes along the current path from root to node. Decrementing before returning preserves subtree isolation.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(V)$. Every tree node is visited exactly once with $\mathcal{O}(1)$ amortized map operations.
- **Space Complexity**: $\mathcal{O}(H)$, where $H$ is the maximum tree depth ($\mathcal{O}(\log V)$ balanced, $\mathcal{O}(V)$ degenerate).

</div>

</div>
</details>

---

### 13. Word Search II with Trie & Backtracking Pruning

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 13</span>
  <span class="review-card-title">Word Search II with Trie & Backtracking Pruning</span>
  <span class="review-card-tag">Trie · Grid Backtracking · Dynamic Leaf Pruning · In-Place Visited Sentinel</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 212 · Word Search II](https://leetcode.com/problems/word-search-ii/) — `https://leetcode.com/problems/word-search-ii/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Word Search II with Trie & Backtracking Pruning (LeetCode 212)**:
> Given an $m \times n$ `board` of characters and a list of strings `words`, return all words on the board.
> Each word must be constructed from letters of sequentially adjacent cells (4-directionally).
> Must optimize with Trie dynamic leaf pruning and in-place visited sentinel.

**Function Signature**:
```python
class WordSearchIISolution:
    @classmethod
    def findWords(cls, board: List[List[str]], words: List[str]) -> List[str]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List, Dict, Any

class WordSearchIISolution:
    @classmethod
    def findWords(cls, board: List[List[str]], words: List[str]) -> List[str]:
        """
        Finds all dictionary words constructible on a 2D character board.
        Uses Trie prefix tree with dynamic on-the-fly leaf pruning.
        """
        if not board or not board[0] or not words:
            return []

        # 1. Build Trie
        root: Dict[str, Any] = {}
        for word in words:
            curr = root
            for ch in word:
                curr = curr.setdefault(ch, {})
            curr['$'] = word  # Terminal sentinel storing full word

        m, n = len(board), len(board[0])
        result = []

        # 2. Backtracking DFS with dynamic leaf node pruning
        def dfs(r: int, c: int, parent_node: Dict[str, Any]) -> None:
            ch = board[r][c]
            curr_node = parent_node[ch]

            # Match found
            matched_word = curr_node.pop('$', None)
            if matched_word is not None:
                result.append(matched_word)

            # In-place sentinel to prevent revisiting on current path
            board[r][c] = '#'

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and board[nr][nc] in curr_node:
                    dfs(nr, nc, curr_node)

            # Restore original character
            board[r][c] = ch

            # Dynamic pruning: if current Trie branch becomes empty, unlink from parent
            if not curr_node:
                parent_node.pop(ch)

        for r in range(m):
            for c in range(n):
                if board[r][c] in root:
                    dfs(r, c, root)

        return result

if __name__ == "__main__":
    b13 = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
    w13 = ["oath","pea","eat","rain"]
    assert sorted(WordSearchIISolution.findWords(b13, w13)) == ["eat", "oath"]
    print("✅ Card 13 (Word Search II with Trie) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Prefix Sharing via Trie**:
  Searching words individually yields $\mathcal{O}(W \cdot M \cdot N \cdot 4^L)$. Packing $W$ words into a Trie amortizes prefix traversals so shared prefixes are explored simultaneously.
- **On-the-fly Leaf Pruning**:
  Popping `$` prevents duplicates. If a Trie node subsequently has no child branches, removing it via `parent_node.pop(ch)` permanently terminates redundant branch exploration in subsequent board scans.
- **In-Place Board Sentinel**:
  Overwriting `board[r][c] = '#'` eliminates `visited` set allocation and hashing overheads.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(\sum |W_i|)$ for Trie construction. Grid DFS worst case $\mathcal{O}(M \cdot N \cdot 4 \cdot 3^{L-1})$ ($L$ = max word length), but dynamic leaf pruning collapses runtime close to linear in practice.
- **Space Complexity**: $\mathcal{O}(\sum |W_i|)$ for the Trie structure.

</div>

</div>
</details>

---

### 14. Grid Shortest Path with Fuel Tank & Recharge Stations

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 14</span>
  <span class="review-card-title">Grid Shortest Path with Fuel Tank & Recharge Stations</span>
  <span class="review-card-tag">State Expansion · Dijkstra Shortest Path · Recharge State Collapse · Large-K Supergraph</span>
</summary>
<div class="review-card-content">

> 💡 **Problem Type**: Standalone Algorithm Implementation (No direct LeetCode equivalent; logic and test specifications are standalone, please run the self-contained test suite below for local verification).

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Grid Shortest Path with Fuel Tank & Recharge Stations (LeetCode 864 Variant)**:
> Find shortest path from start to target on a grid where a vehicle has a limited fuel tank capacity $C$.
> Moving to an adjacent cell consumes 1 unit of fuel. Recharge stations replenish fuel to $C$.
> Return minimum steps to destination, or -1 if unreachable.

**Function Signature**:
```python
class FuelGridShortestPathSolution:
    @classmethod
    def shortestPathWithFuel(cls, grid: List[List[str]], capacity: int) -> int: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
import heapq
from typing import List

class FuelGridShortestPathSolution:
    @classmethod
    def minCost(
        cls,
        grid_cost: List[List[int]],
        blocked: List[List[bool]],
        recharge: List[List[bool]],
        K: int
    ) -> int:
        """
        Finds the minimum entry cost to reach (m-1, n-1) from (0, 0) with a fuel cap K.
        
        State: (cost, r, c, fuel)
        Entering a recharge cell immediately resets fuel to K.
        """
        m, n = len(grid_cost), len(grid_cost[0])
        if blocked[0][0] or blocked[m - 1][n - 1]:
            return -1

        start_cost = grid_cost[0][0]
        start_fuel = K
        
        # dist[(r, c, fuel)] tracks minimum recorded cost to reach state
        dist = {}
        dist[(0, 0, start_fuel)] = start_cost
        
        pq = [(start_cost, 0, 0, start_fuel)]

        while pq:
            cost, r, c, fuel = heapq.heappop(pq)

            if r == m - 1 and c == n - 1:
                return cost

            if cost > dist.get((r, c, fuel), float('inf')):
                continue

            if fuel == 0:
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and not blocked[nr][nc]:
                    next_fuel = K if recharge[nr][nc] else fuel - 1
                    next_cost = cost + grid_cost[nr][nc]

                    if next_cost < dist.get((nr, nc, next_fuel), float('inf')):
                        dist[(nr, nc, next_fuel)] = next_cost
                        heapq.heappush(pq, (next_cost, nr, nc, next_fuel))

        return -1

if __name__ == "__main__":
    grid_cost = [[1, 2], [3, 4]]
    blocked = [[False, False], [False, False]]
    recharge = [[False, False], [False, False]]
    assert FuelGridShortestPathSolution.minCost(grid_cost, blocked, recharge, 2) == 7
    assert FuelGridShortestPathSolution.minCost(grid_cost, blocked, recharge, 1) == -1
    print("✅ Card 14 (Grid Shortest Path with Fuel Tank) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Layered Graph / State Space Expansion**:
  Fuel remaining dictates future reachability. The state space is expanded to $(r, c, 	ext{fuel})$, with directed edges weighted by destination `grid_cost[nr][nc]`.
- **Recharge State Collapse**:
  At any cell with `recharge[nr][nc] == True`, fuel resets to $K$, collapsing all arriving fuel states into $(nr, nc, K)$.
- **Large-K Optimization & Supergraph Condensation**:
  1. If $K \ge m + n - 2$, fuel never bounds the optimal path, reducing to 2D grid Dijkstra in $\mathcal{O}(MN \log(MN))$.
  2. For sparse recharge cells ($R \ll MN$), construct a **Recharge Supergraph** with vertices $\{	ext{Start}, 	ext{Goal}\} \cup \{	ext{Recharge Stations}\}$. Run pair-wise fuel-constrained shortest paths between stations, then execute Dijkstra over the condensed $\mathcal{O}(R)$-node graph.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(M \cdot N \cdot K \log(M \cdot N \cdot K))$. At most $\mathcal{O}(MNK)$ states pushed into the priority queue.
- **Space Complexity**: $\mathcal{O}(M \cdot N \cdot K)$ for distance dictionary and priority queue.

</div>

</div>
</details>

---

### 15. Photo Similarity Groups via Union-Find

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 15</span>
  <span class="review-card-title">Photo Similarity Groups via Union-Find</span>
  <span class="review-card-tag">Disjoint Set Union · Connected Components · Upper-Triangle Scan · O(N^2 * α(N))</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 547 · Number of Provinces](https://leetcode.com/problems/number-of-provinces/) — `https://leetcode.com/problems/number-of-provinces/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Photo Similarity Groups via Union-Find (LeetCode 547 Variant)**:
> Given $N$ photos and an adjacency similarity matrix `is_connected`, group photos that are directly or transitively similar into connected clusters using Disjoint Set Union (DSU).
> Return the number of unique similarity groups.

**Function Signature**:
```python
class PhotoSimilarityGroupsSolution:
    @classmethod
    def findGroups(cls, n: int, is_connected: List[List[int]]) -> int: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class PhotoSimilarityGroupsSolution:
    @classmethod
    def findGroups(cls, isSimilar: List[List[int]]) -> int:
        """
        Computes the number of connected similarity components among N photos.
        """
        if not isSimilar:
            return 0

        n = len(isSimilar)
        parent = list(range(n))
        rank = [0] * n
        components_count = n

        def find(i: int) -> int:
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> bool:
            nonlocal components_count
            root_i, root_j = find(i), find(j)
            if root_i == root_j:
                return False
            
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1

            components_count -= 1
            return True

        # Scan strictly upper triangle (j > i) using symmetric relation
        for i in range(n):
            for j in range(i + 1, n):
                if isSimilar[i][j] == 1:
                    union(i, j)

        return components_count

if __name__ == "__main__":
    sim_mat = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ]
    assert PhotoSimilarityGroupsSolution.findGroups(sim_mat) == 2
    assert PhotoSimilarityGroupsSolution.findGroups([[1, 0], [0, 1]]) == 2
    print("✅ Card 15 (Photo Similarity Groups via Union-Find) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Equivalence Relation & Connected Components**:
  Symmetry and transitivity form an equivalence relation. Counting similarity groups is isomorphic to finding connected components in an undirected graph $G = (V, E)$.
- **Strict Upper-Triangle Scan**:
  Since $isSimilar[i][j] == isSimilar[j][i]$ and the diagonal is reflexive, scanning $j \in [i+1, n-1]$ halves the iterations to $rac{N(N-1)}{2}$.
- **Streaming Adaptability**:
  While BFS and DSU both run in $\mathcal{O}(N^2)$ on dense matrices, DSU supports online streaming updates in $\mathcal{O}(lpha(N))$ per new edge without re-traversing.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N^2 \cdot lpha(N))$, dominated by matrix scanning with near-constant DSU operations.
- **Space Complexity**: $\mathcal{O}(N)$ for parent and rank arrays.

</div>

</div>
</details>

---

### 16. Binary Tree Right Side View with Custom Tree Scaffolding

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">TREE 16</span>
  <span class="review-card-title">Binary Tree Right Side View with Custom Tree Scaffolding</span>
  <span class="review-card-tag">Binary Tree · Level-Order BFS · Right-First DFS · Test Scaffolding</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 199 · Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) — `https://leetcode.com/problems/binary-tree-right-side-view/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Binary Tree Right Side View (LeetCode 199)**:
> Given `root` of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
> Includes self-contained scaffolding for tree reconstruction.

**Function Signature**:
```python
class BinaryTreeScaffolding:
    @staticmethod
    def rightSideViewBFS(root: Optional[TreeNode]) -> List[int]: ...
    @staticmethod
    def rightSideViewDFS(root: Optional[TreeNode]) -> List[int]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List, Optional
from collections import deque

class TreeNode:
    """Production-grade binary tree node definition."""
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTreeScaffolding:
    """Helper to reconstruct binary trees from level-order arrays with None placeholders."""
    @classmethod
    def build_tree(cls, values: List[Optional[int]]) -> Optional[TreeNode]:
        if not values or values[0] is None:
            return None
        
        root = TreeNode(values[0])
        queue = deque([root])
        idx = 1
        n = len(values)

        while queue and idx < n:
            curr = queue.popleft()
            
            if idx < n and values[idx] is not None:
                curr.left = TreeNode(values[idx])
                queue.append(curr.left)
            idx += 1
            
            if idx < n and values[idx] is not None:
                curr.right = TreeNode(values[idx])
                queue.append(curr.right)
            idx += 1

        return root

class RightSideViewSolution:
    @classmethod
    def rightSideViewBFS(cls, root: Optional[TreeNode]) -> List[int]:
        """Approach 1: BFS level-order traversal taking the last node per layer."""
        if not root:
            return []
        
        result = []
        queue = deque([root])

        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if i == level_size - 1:
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return result

    @classmethod
    def rightSideViewDFS(cls, root: Optional[TreeNode]) -> List[int]:
        """Approach 2: Right-first DFS (root -> right -> left) recording first visit per depth."""
        result = []

        def dfs(node: Optional[TreeNode], depth: int) -> None:
            if not node:
                return
            if depth == len(result):
                result.append(node.val)
            dfs(node.right, depth + 1)
            dfs(node.left, depth + 1)

        dfs(root, 0)
        return result

if __name__ == "__main__":
    tree16 = BinaryTreeScaffolding.build_tree([1, 2, 3, None, 5, None, 4])
    assert RightSideViewSolution.rightSideViewBFS(tree16) == [1, 3, 4]
    assert RightSideViewSolution.rightSideViewDFS(tree16) == [1, 3, 4]
    print("✅ Card 16 (Binary Tree Right Side View) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **BFS vs Right-First DFS Trade-offs**:
  - **BFS Queue**: Natural per-level grouping; the final node in each level queue is directly the rightmost visible node. Space scales with max layer width $W$.
  - **Right-First DFS**: Order `Root -> Right -> Left`. Condition `depth == len(result)` guarantees that the first node reaching any new depth is the rightmost node. Space scales with tree height $H$.
- **Right Side View vs Rightmost Branch**:
  The right side view is NOT just the branch of right children. If the right subtree terminates early, deeper nodes from the left subtree become visible from the right.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$ for both approaches, visiting each node once.
- **Space Complexity**:
  - BFS: $\mathcal{O}(W)$ where $W$ is the maximum tree level width ($\mathcal{O}(N)$ for full binary tree).
  - DFS: $\mathcal{O}(H)$ where $H$ is the maximum tree depth ($\mathcal{O}(\log N)$ balanced, $\mathcal{O}(N)$ skewed).

</div>

</div>
</details>

---

### 17. Construct Binary Tree from Preorder and Postorder Traversal

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">TREE 17</span>
  <span class="review-card-title">Construct Binary Tree from Preorder and Postorder Traversal</span>
  <span class="review-card-tag">Recursive Reconstruction · Postorder Index Map · Subtree Size Partitioning · O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 889 · Construct Binary Tree from Preorder and Postorder Traversal](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/) — `https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Construct Binary Tree from Preorder and Postorder Traversal (LeetCode 889)**:
> Given two integer arrays, `preorder` and `postorder` of a binary tree of distinct values, reconstruct and return the binary tree.
> If multiple answers exist, return any of them.

**Function Signature**:
```python
class ConstructFromPrePostSolution:
    @classmethod
    def constructFromPrePost(cls, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List, Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ConstructFromPrePostSolution:
    @classmethod
    def constructFromPrePost(
        cls, preorder: List[int], postorder: List[int]
    ) -> Optional[TreeNode]:
        """
        Reconstructs a binary tree from preorder and postorder traversals (distinct values).
        If ambiguous single-child nodes exist, returns any valid configuration.
        """
        if not preorder or not postorder:
            return None

        post_idx = {val: i for i, val in enumerate(postorder)}

        def build(pre_start: int, pre_end: int, post_start: int, post_end: int) -> Optional[TreeNode]:
            if pre_start > pre_end:
                return None

            root = TreeNode(preorder[pre_start])
            if pre_start == pre_end:
                return root

            # Key pivot: preorder[pre_start + 1] is the left subtree root
            left_root_val = preorder[pre_start + 1]
            left_post_idx = post_idx[left_root_val]

            left_size = left_post_idx - post_start + 1

            root.left = build(
                pre_start + 1, pre_start + left_size,
                post_start, left_post_idx
            )
            root.right = build(
                pre_start + left_size + 1, pre_end,
                left_post_idx + 1, post_end - 1
            )
            return root

        n = len(preorder)
        return build(0, n - 1, 0, n - 1)

if __name__ == "__main__":
    r17 = ConstructFromPrePostSolution.constructFromPrePost([1,2,4,5,3,6,7], [4,5,2,6,7,3,1])
    assert r17.val == 1 and r17.left.val == 2 and r17.right.val == 3
    print("✅ Card 17 (Construct Binary Tree from Pre/Postorder) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Partitioning Pivot**:
  Preorder layout is `[Root, Left..., Right...]` and postorder is `[Left..., Right..., Root]`.
  The element `preorder[pre_start + 1]` must be the left-child root. Finding its index in postorder yields `left_size = left_post_idx - post_start + 1`, splitting index boundaries recursively in $\mathcal{O}(1)$.
- **Single-Child Ambiguity**:
  Preorder and postorder cannot uniquely determine tree topology when a node has a single child. The algorithm canonically roots the child as a left subtree, satisfying consistency requirements.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$ using hash-indexed lookups.
- **Space Complexity**: $\mathcal{O}(N)$ for hash map and recursive call stack.

</div>

</div>
</details>

---

### 18. Construct Binary Tree from Descriptions via Child Set Deduction

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">TREE 18</span>
  <span class="review-card-title">Construct Binary Tree from Descriptions via Child Set Deduction</span>
  <span class="review-card-tag">Node Registry · Child Set Set-Difference · Topological Root · O(N)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 2196 · Create Binary Tree From Descriptions](https://leetcode.com/problems/create-binary-tree-from-descriptions/) — `https://leetcode.com/problems/create-binary-tree-from-descriptions/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Create Binary Tree From Descriptions (LeetCode 2196)**:
> You are given a 2D integer array `descriptions` where `descriptions[i] = [parent_i, child_i, isLeft_i]`.
> - `isLeft_i == 1`: left child; `isLeft_i == 0`: right child.
> Construct the binary tree and return its root.

**Function Signature**:
```python
class ConstructTreeFromDescriptionsSolution:
    @classmethod
    def createBinaryTree(cls, descriptions: List[List[int]]) -> Optional[TreeNode]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import List, Optional

class ConstructTreeFromDescriptionsSolution:
    @classmethod
    def createBinaryTree(cls, descriptions: List[List[int]]) -> Optional[TreeNode]:
        """
        descriptions[i] = [parent, child, isLeft]
        Reconstructs the binary tree and returns its unique root.
        """
        nodes = {}
        children = set()

        for parent_val, child_val, is_left in descriptions:
            if parent_val not in nodes:
                nodes[parent_val] = TreeNode(parent_val)
            if child_val not in nodes:
                nodes[child_val] = TreeNode(child_val)

            if is_left == 1:
                nodes[parent_val].left = nodes[child_val]
            else:
                nodes[parent_val].right = nodes[child_val]

            children.add(child_val)

        root_val = None
        for parent_val, _, _ in descriptions:
            if parent_val not in children:
                root_val = parent_val
                break

        return nodes[root_val] if root_val is not None else None

if __name__ == "__main__":
    desc = [[20,15,1],[20,17,0],[50,20,1],[50,80,0],[80,19,1]]
    r18 = ConstructTreeFromDescriptionsSolution.createBinaryTree(desc)
    assert r18.val == 50 and r18.left.val == 20 and r18.right.val == 80
    print("✅ Card 18 (Construct Binary Tree from Descriptions) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Zero-Indegree Root Property**:
  In a valid tree, every non-root node has in-degree exactly 1 (it appears in `child` positions). The root is the unique node that never appears as a child.
- **Object Identity Caching**:
  `nodes` dictionary ensures nodes referenced across multiple parent/child entries maintain consistent object identity and pointers.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(N)$ single pass over edge descriptions.
- **Space Complexity**: $\mathcal{O}(N)$ storing node references and child set.

</div>

</div>
</details>

---

### 19. Restore IP Addresses & Generalized K-Segment Partition

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">BT 19</span>
  <span class="review-card-title">Restore IP Addresses & Generalized K-Segment Partition</span>
  <span class="review-card-tag">Backtracking · Boundary Validation · Leading Zero Guard · Pigeonhole Pruning · O(1)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 93 · Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/) — `https://leetcode.com/problems/restore-ip-addresses/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Restore IP Addresses & Generalized K-Segment Partition (LeetCode 93)**:
> A valid IP address consists of exactly four integers, each between 0 and 255, separated by single dots and without leading zeros.
> Given a string `s` containing only digits, return all possible valid IP addresses.
>
> **Industrial Follow-up**: Generalize to arbitrary $K$-segment partitioning.

**Function Signature**:
```python
class RestoreIPSolution:
    @classmethod
    def restoreIpAddresses(cls, s: str) -> List[str]: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List

class RestoreIPSolution:
    @classmethod
    def restoreIpAddresses(cls, s: str) -> List[str]:
        return cls.partitionStringIntoSegments(s, k=4, max_val=255)

    @classmethod
    def partitionStringIntoSegments(cls, s: str, k: int = 4, max_val: int = 255) -> List[str]:
        """Partitions digit string into k valid integer segments in range [0, max_val]."""
        n = len(s)
        if n < k or n > k * 3:
            return []

        result = []
        path: List[str] = []

        def backtrack(start_idx: int, segments_left: int) -> None:
            if segments_left == 0:
                if start_idx == n:
                    result.append(".".join(path))
                return

            remaining_chars = n - start_idx
            if remaining_chars < segments_left or remaining_chars > segments_left * 3:
                return

            for length in range(1, 4):
                if start_idx + length > n:
                    break

                segment_str = s[start_idx : start_idx + length]

                if length > 1 and segment_str[0] == '0':
                    break

                val = int(segment_str)
                if val > max_val:
                    break

                path.append(segment_str)
                backtrack(start_idx + length, segments_left - 1)
                path.pop()

        backtrack(0, k)
        return result

if __name__ == "__main__":
    assert sorted(RestoreIPSolution.restoreIpAddresses("25525511135")) == sorted(["255.255.11.135", "255.255.111.35"])
    assert RestoreIPSolution.restoreIpAddresses("0000") == ["0.0.0.0"]
    assert RestoreIPSolution.restoreIpAddresses("101023") == ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
    print("✅ Card 19 (Restore IP Addresses) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **Validation Triad**:
  1. No leading zero: multi-digit segment cannot start with `'0'`;
  2. Range bound: $0 \le val \le 255$;
  3. Total consumption: all input characters must be utilized.
- **Pigeonhole Pruning**:
  Remaining characters must satisfy $segments\_left \le rem \le 3 	imes segments\_left$, collapsing the recursion tree.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(1)$ for fixed $k=4$, bounded by $3^4 = 81$ states.
- **Space Complexity**: $\mathcal{O}(k)$ recursion depth.

</div>

</div>
</details>

---

### 20. Alien Dictionary via Directed Graph Topological Sort

<details class="review-card">
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 20</span>
  <span class="review-card-title">Alien Dictionary via Directed Graph Topological Sort</span>
  <span class="review-card-tag">Directed Graph Topological Sort · Kahn's BFS · Prefix Trap Defense · O(C)</span>
</summary>
<div class="review-card-content">

> 🔗 **LeetCode Link**: [LeetCode 269 · Alien Dictionary](https://leetcode.com/problems/alien-dictionary/) — `https://leetcode.com/problems/alien-dictionary/`

<div class="review-block">
<div class="review-block-label">📌 Problem Statement & Requirements</div>

**Original Problem Statement**:
> **Alien Dictionary via Directed Graph Topological Sort (LeetCode 269)**:
> Given a list of strings `words` from the alien dictionary, sorted lexicographically by the rules of this new language, derive the order of letters in this language.
> Return `""` if the order is invalid (cycle or invalid prefix order such as `["abc", "ab"]`).

**Function Signature**:
```python
class AlienDictionarySolution:
    @classmethod
    def alienOrder(cls, words: List[str]) -> str: ...
```

</div>

<div class="review-block">
<div class="review-block-label">📌 Core Implementation</div>

```python
from typing import List
from collections import defaultdict, deque

class AlienDictionarySolution:
    @classmethod
    def alienOrder(cls, words: List[List[str]]) -> str:
        """
        Infers valid alphabet ordering from adjacent sorted words.
        Returns empty string if order is contradictory (cyclic) or invalid.
        """
        if not words:
            return ""

        adj = defaultdict(set)
        in_degree = {ch: 0 for word in words for ch in word}

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            min_len = min(len(w1), len(w2))
            found_diff = False

            for j in range(min_len):
                c1, c2 = w1[j], w2[j]
                if c1 != c2:
                    if c2 not in adj[c1]:
                        adj[c1].add(c2)
                        in_degree[c2] += 1
                    found_diff = True
                    break

            # Prefix trap: w2 is strict prefix of w1 but w1 is longer (e.g. ["abc", "ab"])
            if not found_diff and len(w1) > len(w2):
                return ""

        queue = deque([ch for ch, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            curr = queue.popleft()
            order.append(curr)

            for nxt in adj[curr]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) < len(in_degree):
            return ""

        return "".join(order)

if __name__ == "__main__":
    assert AlienDictionarySolution.alienOrder(["wrt","wrf","er","ett","rftt"]) == "wertf"
    assert AlienDictionarySolution.alienOrder(["z","x"]) == "zx"
    assert AlienDictionarySolution.alienOrder(["z","x","z"]) == ""
    print("✅ Card 20 (Alien Dictionary via Topological Sort) all tests passed!")
```

</div>

<div class="review-block">
<div class="review-block-label">💡 Mechanism & Invariants</div>

- **First Differing Character**:
  Only the first mismatch between adjacent words provides valid relative ordering $c_1 	o c_2$.
- **Crucial Pitfalls**:
  1. **Prefix Violation Trap**: If $w1$ contains $w2$ as prefix with $len(w1) > len(w2)$, it violates lexicographical ordering; return `""`;
  2. **Isolated Nodes**: Ensure all characters in any word are registered in `in_degree`;
  3. **Duplicate Edges**: Store edges in a `set` to prevent incrementing in-degree multiple times for the same character pair.

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity Analysis</div>

- **Time Complexity**: $\mathcal{O}(C)$ where $C$ is total characters across all words.
- **Space Complexity**: $\mathcal{O}(|\Sigma| + |E|)$, bounded by unique alphabet size ($\le 26$).

</div>

</div>
</details>


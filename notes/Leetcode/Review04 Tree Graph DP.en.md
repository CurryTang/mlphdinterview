# Review Flashcards: Trees, Graphs & DP

This note is the fourth volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Graphs & Grid Search**, **Trees & BST**, and **Dynamic Programming** with production-grade implementations and asymptotic complexity breakdowns.

---

## Module 1: Graphs & Grid Search

### 1. Number of Islands & All Canonical Variants

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 01</span>
  <span class="review-card-title">Number of Islands & All Canonical Variants</span>
  <span class="review-card-tag">Grid Implicit Graph · BFS/DFS · Relative Coordinate Normalization</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 02</span>
  <span class="review-card-title">Course Schedule & Topological Sort</span>
  <span class="review-card-tag">Kahn BFS In-degree · DFS Three-Color Mark · Cycle Reconstruction</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

### 3. Order Validator with Dynamic DAG Dependencies & Node Contraction

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 03</span>
  <span class="review-card-title">Order Validator with Dynamic DAG Dependencies & Node Contraction</span>
  <span class="review-card-tag">Directed Acyclic Graph · Dynamic Dependencies · Topological Validation · Node Contraction Rewiring</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 04</span>
  <span class="review-card-title">Shortest Path in Grid with Obstacles Elimination</span>
  <span class="review-card-tag">3D State BFS · Dominance Pruning · Manhattan Distance Shortcut</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 05</span>
  <span class="review-card-title">Lowest Common Ancestor (LCA)</span>
  <span class="review-card-tag">Postorder Divide-and-Conquer</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

### 6. Binary Tree Maximum Path Sum & Path Reconstruction

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 06</span>
  <span class="review-card-title">Binary Tree Maximum Path Sum & Path Reconstruction</span>
  <span class="review-card-tag">Postorder Tree DP · Negative Gain Clamping</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

```python
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
```

</div>

</div>
</details>

---

### 7. Flatten Comment Tree to Multi-Level Hierarchy

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 07</span>
  <span class="review-card-title">Flatten Comment Tree to Multi-Level Hierarchy</span>
  <span class="review-card-tag">Two-Pass Hash Assembly · Arbitrary Depth · Orphan Safety · Self-Reference Guard</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 08</span>
  <span class="review-card-title">Equalize Root-to-Leaf Path Sums in N-ary Tree</span>
  <span class="review-card-tag">N-ary Tree Postorder · Tree Greedy Lift · Majority Element Warmup</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">DP 09</span>
  <span class="review-card-title">Coin Change 1 & 2 / Unbounded Knapsack</span>
  <span class="review-card-tag">Unbounded Knapsack</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

</div>
</details>

---

### 10. Stickers to Spell Word & Bitmask DP

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">DP 10</span>
  <span class="review-card-title">Stickers to Spell Word & Bitmask DP</span>
  <span class="review-card-tag">Bitmask DP · First Unmet Pruning · O(2^n * m * n)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

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
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- Time: $\mathcal{O}(2^N \cdot M \cdot N)$. Space: $\mathcal{O}(2^N)$.

</div>

</div>
</details>

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

---

### 11. Longest Alternating Zigzag Path in 2D Grid

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 11</span>
  <span class="review-card-title">Longest Alternating Zigzag Path in 2D Grid</span>
  <span class="review-card-tag">2D Grid · Memoized Search · State-Machine DP · O(M * N)</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">TREE 12</span>
  <span class="review-card-title">N-ary Tree Downward Target Path Sum via Prefix Sum</span>
  <span class="review-card-tag">N-ary Tree · Running Prefix Sum · Backtracking Scope Cleanup · O(N)</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 13</span>
  <span class="review-card-title">Word Search II with Trie & Backtracking Pruning</span>
  <span class="review-card-tag">Trie · Grid Backtracking · Dynamic Leaf Pruning · In-Place Visited Sentinel</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 14</span>
  <span class="review-card-title">Grid Shortest Path with Fuel Tank & Recharge Stations</span>
  <span class="review-card-tag">State Expansion · Dijkstra Shortest Path · Recharge State Collapse · Large-K Supergraph</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 15</span>
  <span class="review-card-title">Photo Similarity Groups via Union-Find</span>
  <span class="review-card-tag">Disjoint Set Union · Connected Components · Upper-Triangle Scan · O(N^2 * α(N))</span>
</summary>
<div class="review-card-content">

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

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">TREE 16</span>
  <span class="review-card-title">Binary Tree Right Side View with Custom Tree Scaffolding</span>
  <span class="review-card-tag">Binary Tree · Level-Order BFS · Right-First DFS · Test Scaffolding</span>
</summary>
<div class="review-card-content">

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


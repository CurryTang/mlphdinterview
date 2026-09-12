# Review Flashcards: Trees, Graphs & DP

This note is the fourth volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Graphs & Grid Search**, **Trees & BST**, and **Dynamic Programming** with production-grade implementations and asymptotic complexity breakdowns.

---

## Module 1: Graphs & Grid Search

### 1. Number of Islands & All Canonical Variants

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 01</span>
  <span class="review-card-title">Number of Islands & All Canonical Variants</span>
  <span class="review-card-tag">Grid Implicit Graph · BFS/DFS · Disjoint Set · Relative Coordinate Normalization · Out-of-Core Partition</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Master Variant Matrix</div>

Given an $m \times n$ 2D binary grid `grid` representing a map of `'1'`s (land) and `'0'`s (water), return the number of islands:

| Variant | Variant Name | Core Restriction | Algorithmic Strategy |
|---|---|---|---|
| **Variant 1** | **Classic Count** | Count 4-connected components | DFS / BFS Flood Fill |
| **Variant 2** | **No-Modify** | Grid is read-only | BFS explicit queue + visited set, **mark upon enqueue** |
| **Variant 3** | **Same-Shape (LC 694)** | Count distinct island geometries | **Relative coordinate translation normalization** |
| **Variant 4** | **Huge-Map** | Grid exceeds single-machine RAM | **Tile blocks + local connectivity + cross-border DSU stitching** |
| **Variant 5** | **Water-Flow (LC 417)** | Flow downhill to boundary | Multi-source reverse BFS/DFS from ocean borders |
| **Variant 6** | **2D Runs** | Largest 1-region | General component (DFS size sum) vs rectangle (monotonic stack) |
| **Variant 7** | **Perimeter (LC 463)** | Total perimeter | $\text{Perimeter} = 4 \times \text{land} - 2 \times \text{shared edges}$ |
| **Variant 8** | **Single-Pass Aggregation** | Return island count and max area simultaneously | Aggregate local area counter alongside global count |

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from collections import deque
from typing import List, Tuple

class IslandSolution:
    @staticmethod
    def numIslands(grid: List[List[str]]) -> int:
        if not grid or not grid[0]: return 0
        m, n = len(grid), len(grid[0])
        count = 0
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

    @staticmethod
    def numDistinctIslands(grid: List[List[int]]) -> int:
        if not grid or not grid[0]: return 0
        m, n = len(grid), len(grid[0])
        visited = set()
        unique_shapes = set()
        for r in range(m):
            for c in range(n):
                if grid[r][c] == 1 and (r, c) not in visited:
                    shape = []
                    queue = deque([(r, c)])
                    visited.add((r, c))
                    while queue:
                        cr, cc = queue.popleft()
                        shape.append((cr - r, cc - c))
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == 1 and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                    shape.sort()
                    unique_shapes.add(tuple(shape))
        return len(unique_shapes)
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(M \times N)$.
- **Space Complexity**: $\mathcal{O}(M \times N)$.
- **Critical Pitfall**: Always mark nodes as visited **immediately upon enqueue**.

</div>

</div>
</details>

---

### 2. Course Schedule & Topological Sort

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Graph 02</span>
  <span class="review-card-title">Course Schedule & Topological Sort</span>
  <span class="review-card-tag">Kahn BFS In-degree · DFS Three-Color Mark · Directed Cycle Path Reconstruction · SRE DAG · Word Ladder BFS</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Master Variant Matrix</div>

Given `numCourses` and prerequisite pairs where `[a, b]` means $b \to a$:
- **Feasibility (LC 207)**: `canFinish(numCourses, prerequisites) -> bool`
- **Order (LC 210)**: `findOrder(numCourses, prerequisites) -> List[int]`

| Variant | Variant Name | Core Requirement | Architectural Resolution |
|---|---|---|---|
| **Variant 1** | **Topological Feasibility (LC 207)** | Check if a valid curriculum exists | **Kahn's BFS In-Degree Algorithm**: In-degree 0 queue; check if popped count equals $V$. |
| **Variant 2** | **Full Order Reconstruction (LC 210)** | Return valid course order, empty if cyclic | Record popped nodes into `order`; return `[]` if count $< V$. |
| **Variant 3** | **SRE Directed Graph Cycle Check** | Prerequisite list given as custom service edges | Parse edges, apply Kahn's algorithm to isolate cyclic deadlocks. |
| **Variant 4** | **DAG Order + DFS Cycle Path Print** | Output valid DAG order, or **print the exact cyclic path** | **DFS Three-Color Marking** (0 White, 1 Gray, 2 Black): on Gray node, trace `parent` chain to reconstruct the exact cycle loop. |
| **Variant 5** | **Word Ladder Implicit Graph (LC 127)** | Produce shortest transformation sequence | **On-the-fly neighbor generation BFS**: Replace each position with 26 letters; do NOT precompute $O(N^2)$ full edge list. |

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

#### 1. Kahn's Algorithm vs DFS Three-Color Marking
- **Kahn's Algorithm (BFS)**:
  - Natural fit when you need a valid topological ordering.
  - Decrements in-degrees of neighbors. Nodes reaching in-degree 0 join the queue.
- **DFS Three-Color Marking**:
  - `0 (White)`: Unvisited.
  - `1 (Gray)`: Active in current recursion stack.
  - `2 (Black)`: Completely explored and certified acyclic.
  - Encountering a **Gray** node indicates a back-edge into an ancestor $\implies$ cycle detected!
  - Preferred when you need to **print the cycle**, because the recursion stack / `parent` pointer chain holds the exact loop.

#### 2. Word Ladder Implicit Graph BFS
- Rather than constructing an $O(N^2 L)$ edge list, generate candidates dynamically by replacing characters across 'a'-'z'.
- Remove matched words from `wordSet` upon enqueue to prevent cycles. Time complexity is $\mathcal{O}(N \times 26 \times L)$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from collections import deque
from typing import List, Optional

class CourseScheduleSolution:

    @staticmethod
    def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """LC 210: Kahn's algorithm for topological order. O(V + E)"""
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

    @staticmethod
    def detectAndPrintCycle(numCourses: int, prerequisites: List[List[int]]) -> Optional[List[int]]:
        """DFS three-color marking: detect and print exact cycle path."""
        adj = [[] for _ in range(numCourses)]
        for dest, src in prerequisites:
            adj[src].append(dest)

        color = [0] * numCourses
        parent = [-1] * numCourses
        cycle = []

        def dfs(u: int) -> bool:
            color[u] = 1
            for v in adj[u]:
                if color[v] == 1:
                    cur = u
                    cycle.append(v)
                    while cur != v:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(v)
                    cycle.reverse()
                    return True
                elif color[v] == 0:
                    parent[v] = u
                    if dfs(v):
                        return True
            color[u] = 2
            return False

        for i in range(numCourses):
            if color[i] == 0:
                if dfs(i):
                    return cycle
        return None

    @staticmethod
    def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
        """LC 127: Word Ladder BFS with on-the-fly neighbor generation."""
        word_set = set(wordList)
        if endWord not in word_set:
            return 0

        queue = deque([(beginWord, 1)])
        if beginWord in word_set:
            word_set.remove(beginWord)

        L = len(beginWord)
        while queue:
            word, step = queue.popleft()
            if word == endWord:
                return step

            for i in range(L):
                for ch in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + ch + word[i+1:]
                    if next_word in word_set:
                        word_set.remove(next_word)
                        queue.append((next_word, step + 1))

        return 0
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(V + E)$ for topological sort; $\mathcal{O}(N \cdot 26 \cdot L)$ for Word Ladder.
- **Space Complexity**: $\mathcal{O}(V + E)$ for adjacency list.
- **Critical Pitfalls**: Edge direction inversion (`[a, b]` means $b \to a$); forgetting to delete matched words in Word Ladder causing infinite queue expansion.

</div>

</div>
</details>

---

## Module 2: Trees & BST

### 3. Lowest Common Ancestor (LCA)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 03</span>
  <span class="review-card-title">Lowest Common Ancestor (LCA)</span>
  <span class="review-card-tag">Postorder Divide-and-Conquer · BST Value Pruning · Parent Pointer Intersection · Node Existence Guard</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Variants</div>

- **Binary Tree LCA (LC 236)**: Postorder divide-and-conquer in $\mathcal{O}(N)$.
- **BST LCA (LC 235)**: Binary search tree monotonicity pruning in $\mathcal{O}(H)$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class LCASolution:
    @staticmethod
    def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root
        left = LCASolution.lowestCommonAncestor(root.left, p, q)
        right = LCASolution.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right

    @staticmethod
    def lowestCommonAncestorBST(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        cur = root
        while cur:
            if p.val < cur.val and q.val < cur.val:
                cur = cur.left
            elif p.val > cur.val and q.val > cur.val:
                cur = cur.right
            else:
                return cur
        return None
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$ for binary tree, $\mathcal{O}(H)$ for BST.
- **Space Complexity**: $\mathcal{O}(H)$.

</div>

</div>
</details>

---

### 4. Binary Tree Maximum Path Sum & Path Reconstruction

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Tree 04</span>
  <span class="review-card-title">Binary Tree Maximum Path Sum & Path Reconstruction</span>
  <span class="review-card-tag">Postorder Tree DP · Single-Arm Contribution · Negative Gain Clamping · Path Reconstruction · Downward Constraints</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Master Follow-up Matrix</div>

Given the root of a binary tree where node values may be negative, find the maximum path sum:

```python
def maxPathSum(root: Optional[TreeNode]) -> int: ...
```

| Variant | Variant Name | Core Follow-up Condition | Recursion Contract & Strategy |
|---|---|---|---|
| **Variant 1** | **Classic Max Path Sum (LC 124)** | Arbitrary single bend, negative values allowed | **Tree DP**: Return single-arm downward gain to parent; side-effect updates global best bent path sum. |
| **Variant 2** | **Return Path Itself** | Return ordered list of node values along the best path | Recursion returns `(gain, arm_path)`; reconstruct full path as `left_arm[::-1] + [node.val] + right_arm`. |
| **Variant 3** | **Downward-Only Path Sum** | Path may only travel downward (parent-to-leaf) | **Prefix sum hash map**: Track running sums on path down to find sub-path summing to target in $O(N)$. |
| **Variant 4** | **Non-Root / Non-Leaf Constraint** | Path must not bend at root or must involve internal nodes | Gate the global best update with conditional checks `if node != root`. |

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

#### 1. Single-Arm Contribution vs Full Curved Path Sum
- **Gain to parent**: The parent can only extend through one downward arm (either left or right):
  $$\operatorname{gain}(node) = node.val + \max(0, \;\max(\operatorname{gain}(node.left), \;\operatorname{gain}(node.right)))$$
- **Full local curved path**: Bends at `node` by uniting both arms:
  $$\operatorname{curve\_sum}(node) = node.val + \max(0, \operatorname{gain}(node.left)) + \max(0, \operatorname{gain}(node.right))$$
- **Negative gain clamping**: Any sub-tree yielding negative gain must be clamped to 0 (`max(0, gain)`).

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import Optional, List, Tuple

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class MaxPathSumSolution:

    @classmethod
    def maxPathSum(cls, root: Optional[TreeNode]) -> int:
        """LC 124: Classic score output, O(N) time, O(H) space."""
        max_sum = float('-inf')

        def max_gain(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            if not node:
                return 0

            left_gain = max(0, max_gain(node.left))
            right_gain = max(0, max_gain(node.right))

            price_newpath = node.val + left_gain + right_gain
            max_sum = max(max_sum, price_newpath)

            return node.val + max(left_gain, right_gain)

        max_gain(root)
        return int(max_sum)

    @classmethod
    def maxPathSumWithPath(cls, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """Variant 2: Simultaneously outputs best score and node sequence."""
        best_score = float('-inf')
        best_path: List[int] = []

        def dfs(node: Optional[TreeNode]) -> Tuple[int, List[int]]:
            nonlocal best_score, best_path
            if not node:
                return 0, []

            left_gain, left_arm = dfs(node.left)
            right_gain, right_arm = dfs(node.right)

            valid_left = left_gain > 0
            valid_right = right_gain > 0

            cur_sum = node.val + (left_gain if valid_left else 0) + (right_gain if valid_right else 0)

            if cur_sum > best_score:
                best_score = cur_sum
                l_part = left_arm[::-1] if valid_left else []
                r_part = right_arm if valid_right else []
                best_path = l_part + [node.val] + r_part

            if valid_left and (not valid_right or left_gain >= right_gain):
                return node.val + left_gain, [node.val] + left_arm
            elif valid_right:
                return node.val + right_gain, [node.val] + right_arm
            else:
                return node.val, [node.val]

        dfs(root)
        return int(best_score), best_path
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(N)$.
- **Space Complexity**: Recursion stack depth $\mathcal{O}(H)$.
- **Critical Pitfalls**: Initializing `max_sum` to 0 instead of `float('-inf')` produces incorrect answers on all-negative trees (e.g. `[-3]`).

</div>

</div>
</details>

---

## Module 3: Dynamic Programming

### 5. Coin Change 1 & 2 / Unbounded Knapsack

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">DP 05</span>
  <span class="review-card-title">Coin Change 1 & 2 / Unbounded Knapsack</span>
  <span class="review-card-tag">Unbounded Knapsack · 0/1 Knapsack · Optimization vs Combinations · Order of Loops</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Dual Models</div>

- **Coin Change I (LC 322)**: Minimum coins to reach amount.
- **Coin Change II (LC 518)**: Total number of combinations.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

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

    @staticmethod
    def change(amount: int, coins: List[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:
            for x in range(coin, amount + 1):
                dp[x] += dp[x - coin]
        return dp[amount]
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(\text{amount} \times |\text{coins}|)$.
- **Space Complexity**: $\mathcal{O}(\text{amount})$.

</div>

</div>
</details>

---

### 6. Stickers to Spell Word & Bitmask DP

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">DP 06</span>
  <span class="review-card-title">Stickers to Spell Word & Bitmask DP</span>
  <span class="review-card-tag">Bitmask DP · Memoized DFS · First-Unmet Character Pruning · Letter Multiset · O(2^n * m * n)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Constraints</div>

Given $M$ sticker types and a target string `target` (length $N \le 15$), return the minimum number of stickers needed to assemble every character of the target:

```python
def minStickers(stickers: List[str], target: str) -> int: ...
```

- Target length is bounded in the low teens ($N \le 15$), making compact **bitmask state representation** central to the problem.
- An explicit pruning argument is essential to avoid TLE within interview time limits.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Deep Dive Mechanics</div>

#### 1. Bitmask State Representation
- With $N \le 15$, represent satisfied characters with an integer bitmask `mask \in [0, 2^N - 1]`.
- State count: $2^N = 2^{15} = 32768$.

#### 2. Golden Pruning Rule: Branch on First Unmet Character
- Blindly testing all $M$ stickers creates massive permutation redundancy (choosing sticker A then B vs B then A).
- **Pruning**: Identify the first position $k$ where `(mask >> k) & 1 == 0`.
- **Enforce**: In this recursion step, **only branch on stickers that supply `target[k]`**!
- Because `target[k]` must be satisfied at some point, enforcing this canonical order eliminates duplicate branches without omitting the optimal solution.

#### 3. Rigorous Complexity Derivation
- States: $2^N$.
- Transitions per state: $M$ stickers, each scanning $N$ characters.
- Time Complexity: strictly $\mathcal{O}(2^N \cdot M \cdot N)$.
- Space Complexity: $\mathcal{O}(2^N)$ for memoization.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List
from collections import Counter

class StickersSolution:
    """
    Bitmask DP with first-unmet-character branch pruning.
    Time: O(2^N * M * N), Space: O(2^N)
    """

    @classmethod
    def minStickers(cls, stickers: List[str], target: str) -> int:
        n = len(target)
        target_chars = set(target)

        # Precompute character frequencies for target-relevant characters
        sticker_counts = []
        for s in stickers:
            cnt = Counter(ch for ch in s if ch in target_chars)
            if cnt:
                sticker_counts.append(cnt)

        memo = { (1 << n) - 1: 0 }

        def dfs(mask: int) -> int:
            if mask in memo:
                return memo[mask]

            # Find first position not yet satisfied
            first_unmet = 0
            while (mask >> first_unmet) & 1:
                first_unmet += 1
            target_ch = target[first_unmet]

            ans = float('inf')

            # Branch ONLY on stickers containing target_ch
            for cnt in sticker_counts:
                if target_ch not in cnt:
                    continue

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
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(2^N \cdot M \cdot N)$.
- **Space Complexity**: $\mathcal{O}(2^N)$.
- **Critical Pitfalls**: Omitting the first-unmet character prune triggers TLE.

</div>

</div>
</details>

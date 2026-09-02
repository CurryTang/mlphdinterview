# Graphs

## Module 1: Graph Representation

Graphs are commonly stored in two ways: adjacency lists and adjacency matrices. An adjacency list suits sparse graphs, using a `node -> neighbors` mapping to hold each node's neighbors. An adjacency matrix suits graphs with few nodes and frequent edge queries, using an `n x n` 2D array to record whether an edge (or its weight) exists between any two nodes.

- Adjacency list space: `O(V + E)`; adjacency matrix space: `O(V^2)`.
- Traversing every node and edge takes `O(V + E)` either way; under an adjacency matrix this bound degrades to `O(V^2)`, since scanning one node's neighbors already costs `O(V)`.

Adding an edge to an undirected graph requires writing both `u -> v` and `v -> u`. A weighted graph's neighbor list usually stores `(neighbor, weight)` tuples.

```text
addEdge(u, v, w=1):
  adj[u].append((v, w))
  if undirected:
    adj[v].append((u, w))

neighbors(u):
  return adj.get(u, [])
```

Common pitfalls:

- Adding only one direction of an undirected edge, so a neighbor lookup from the other direction misses it.
- Forgetting `visited` during traversal, causing an infinite loop on a graph with a cycle.
- Using array indices for nodes that are not contiguous integers (string nodes, or integer nodes with gaps); a dictionary-based adjacency list should be used instead.

---

## Universal Graph Problem-Solving Blueprint & Master Templates

When facing graph problems in technical interviews, every question maps cleanly to one of **6 universal archetypes** by analyzing:
1. **Graph representation** (Implicit 2D grid vs Explicit adjacency list),
2. **Edge weights** (Unweighted vs Non-negative vs Negative / Step-bounded), and
3. **Target semantics** (Connectivity / Shortest path / Topological order / Eulerian path).

```text
Graph 4-Step Decision Flow:
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ 1. Graph Structure    │ ➔   │ 2. Edge Weighting     │ ➔   │ 3. Connectivity/Order │ ➔   │ 4. Master Template    │
│ Implicit Grid vs List │     │ Unweighted vs Weighted│     │ Dynamic DSU vs Topo   │     │ Drop-in Skeleton Code │
└───────────────────────┘     └───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

### 1. The 6 Master Graph Code Templates

#### Template 1: Implicit Grid Graph (Matrix DFS & Multi-Source BFS)

- **Target Problems**: Number of Islands, Max Area of Island, Surrounded Regions, Pacific Atlantic Water Flow, Rotting Oranges, Walls and Gates.

```python
from collections import deque
from typing import List

# 1. Grid DFS for Connected Components, Area, and Flood Fill
def solve_grid_dfs(grid: List[List[str]]) -> int:
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    visited = set()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    def dfs(r: int, c: int) -> int:
        if not in_bounds(r, c) or (r, c) in visited or grid[r][c] == "0":
            return 0
        visited.add((r, c))
        area = 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            area += dfs(r + dr, c + dc)
        return area

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                dfs(r, c)
                count += 1
    return count


# 2. Grid Multi-Source BFS for Unweighted Shortest Path / Time Diffusion
def solve_grid_multisource_bfs(grid: List[List[int]]) -> int:
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    visited = set()

    # Step 1: Seed all source cells simultaneously into queue
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:  # Starting sources (e.g. rotten oranges / gates)
                queue.append((r, c))
                visited.add((r, c))

    dist = 0
    while queue:
        # Step 2: Layered snapshot progression via len(queue)
        for _ in range(len(queue)):
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] == 1:
                    visited.add((nr, nc))  # Mark visited at ENQUEUE time to prevent duplicates!
                    queue.append((nr, nc))
        if queue:
            dist += 1
    return dist
```

---

#### Template 2: Explicit Graph Traversal & Hash Map Cloning (Graph DFS / BFS)

- **Target Problems**: Clone Graph, Word Ladder (implicit state transformation BFS).

```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    def cloneGraph(self, node: "Node") -> "Node":
        if not node:
            return None
        clones = {}  # Original node -> Cloned node mapping

        def dfs(curr: "Node") -> "Node":
            if curr in clones:
                return clones[curr]
            copy = Node(curr.val)
            clones[curr] = copy
            for neighbor in curr.neighbors:
                copy.neighbors.append(dfs(neighbor))
            return copy

        return dfs(node)
```

---

#### Template 3: Production-Grade Disjoint Set Union (Universal DSU / Union-Find)

- **Target Problems**: Number of Connected Components, Graph Valid Tree, Redundant Connection, Kruskal's MST.

```python
class UnionFind:
    """High-performance Disjoint Set Union with Path Compression and Union by Rank."""
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.count = n  # Dynamic connected components counter

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return False  # Already connected; cycle / redundant connection detected!
        # Union by rank: attach smaller tree under larger root
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)
```

---

#### Template 4: Topological Sort & Cycle Detection via Kahn's Algorithm

- **Target Problems**: Course Schedule (cycle check), Course Schedule II (topological order), Alien Dictionary (character precedence).

```python
from collections import defaultdict, deque
from typing import List, Optional

def solve_topological_sort(num_nodes: int, prerequisites: List[List[int]]) -> Optional[List[int]]:
    graph = defaultdict(list)
    in_degree = [0] * num_nodes

    # 1. Build directed adjacency list & in-degree array (prereq -> course)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # 2. Enqueue all 0-in-degree nodes
    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    topo_order = []

    # 3. BFS traversal & in-degree deduction
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 4. Cycle check: valid if topo_order contains all nodes; else graph has cycles
    return topo_order if len(topo_order) == num_nodes else None
```

---

#### Template 5: Shortest Path Suite (Dijkstra & Bellman-Ford)

- **Dijkstra**: For **non-negative edge weights** single-source shortest path (Network Delay Time, Swim in Rising Water).
- **Bellman-Ford**: For **hop-bounded ($k$ stops)** or negative edge weights (Cheapest Flights Within K Stops).

```python
import heapq
from collections import defaultdict
from typing import List, Dict

# 1. Dijkstra: Min-Heap Non-Negative Shortest Path
def dijkstra(n: int, times: List[List[int]], start: int) -> Dict[int, int]:
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    dist = {}
    heap = [(0, start)]  # (cumulative_distance, node)

    while heap:
        d, u = heapq.heappop(heap)
        if u in dist:
            continue
        dist[u] = d
        for v, w in graph[u]:
            if v not in dist:
                heapq.heappush(heap, (d + w, v))

    return dist  # Dictionary of all reachable shortest distances


# 2. Bellman-Ford: k-Stops Bounded Shortest Path
def bellman_ford_k_stops(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    INF = float("inf")
    prices = [INF] * n
    prices[src] = 0

    # At most k stops = at most k + 1 flights/rounds
    for _ in range(k + 1):
        next_prices = prices.copy()  # Must read old and write new to prevent multi-hop cascading!
        for u, v, w in flights:
            if prices[u] != INF and prices[u] + w < next_prices[v]:
                next_prices[v] = prices[u] + w
        prices = next_prices

    return -1 if prices[dst] == INF else prices[dst]
```

---

#### Template 6: Eulerian Path via Hierholzer's Algorithm

- **Target Problems**: Reconstruct Itinerary (visit every edge exactly once in lexicographical order).

```python
import heapq
from collections import defaultdict
from typing import List

def solve_eulerian_path(tickets: List[List[str]], start: str = "JFK") -> List[str]:
    graph = defaultdict(list)
    # Min-heap ensures smallest destination is greedily explored first
    for src, dst in tickets:
        heapq.heappush(graph[src], dst)

    route = []

    def dfs(curr: str) -> None:
        while graph[curr]:
            nxt = heapq.heappop(graph[curr])  # Consume this edge
            dfs(nxt)
        route.append(curr)  # Crucial: post-order records dead ends and terminal nodes first

    dfs(start)
    return route[::-1]  # Reverse to obtain forward itinerary
```

---

### 2. Master Graph Problem-Solving Decision Matrix

| Pattern Archetype | Core Concept | Recommended Template | Canonical Problems | Time Complexity | Space Complexity |
|---|---|---|---|---|---|
| **Grid Connectivity** | Flood Fill / Connected Area | **Template 1 (Grid DFS)** | Number of Islands, Max Area of Island | $O(mn)$ | $O(mn)$ |
| **Grid Shortest Path** | Multi-Source Layered Diffusion | **Template 1 (Multi-Source BFS)** | Rotting Oranges, Islands and Treasure | $O(mn)$ | $O(mn)$ |
| **State Deep Copy** | Adjacency List Hash Clone | **Template 2 (Graph DFS + Map)** | Clone Graph | $O(V + E)$ | $O(V)$ |
| **Implicit Transformation**| Single-char neighbor substitution | **Template 1/2 (State BFS)** | Word Ladder | $O(N \cdot L^2)$ | $O(N \cdot L)$ |
| **Dynamic Connectivity** | Component count / Cycle check | **Template 3 (Union-Find / DSU)** | Graph Valid Tree, Redundant Connection | $O(E \alpha(V))$ | $O(V)$ |
| **Minimum Spanning Tree** | Minimum total connection weight | **Template 3 (Kruskal + DSU)** / Prim | Min Cost to Connect All Points | $O(E \log E)$ | $O(V + E)$ |
| **Topological Precedence**| Course deps / Alien Dictionary | **Template 4 (Kahn BFS In-degree)** | Course Schedule I/II, Alien Dictionary | $O(V + E)$ | $O(V + E)$ |
| **Non-negative Shortest Path**| Network delay / Minimax path | **Template 5 (Dijkstra Min-Heap)** | Network Delay Time, Swim in Rising Water | $O(E \log V)$ | $O(V + E)$ |
| **Hop-Bounded Shortest Path**| At most $k$ intermediate stops | **Template 5 (Bellman-Ford $k+1$ rounds)**| Cheapest Flights Within K Stops | $O(k \cdot E)$ | $O(V)$ |
| **Traverse Every Edge Once** | Lexicographical Eulerian Path | **Template 6 (Hierholzer Post-Order)** | Reconstruct Itinerary | $O(E \log E)$ | $O(V + E)$ |

---

## Learning Order

| Order | Problem | What to Master |
|---:|---|---|
| 1 | [200. Number of Islands](https://neetcode.io/problems/number-of-islands/question?list=neetcode150) | Matrix DFS/BFS flood fill to count connected components |
| 2 | [695. Max Area of Island](https://neetcode.io/problems/max-area-of-island/question?list=neetcode150) | Accumulating area through DFS's recursive return value |
| 3 | [417. Pacific Atlantic Water Flow](https://neetcode.io/problems/pacific-atlantic-water-flow/question?list=neetcode150) | Two-source DFS/BFS with a reversed flow condition |
| 4 | [130. Surrounded Regions](https://neetcode.io/problems/surrounded-regions/question?list=neetcode150) | Marking the border-reachable safe region first, then flipping the rest |
| 5 | [994. Rotting Oranges](https://neetcode.io/problems/rotting-oranges/question?list=neetcode150) | Multi-source layered BFS |
| 6 | [Islands and Treasure (286. Walls and Gates)](https://neetcode.io/problems/islands-and-treasure/question?list=neetcode150) | Multi-source BFS filling in distance to the nearest source (no visited set needed) |
| 7 | [133. Clone Graph](https://neetcode.io/problems/clone-graph/question?list=neetcode150) | Adjacency-list DFS plus a hash map tracking the clone mapping |
| 8 | [127. Word Ladder](https://neetcode.io/problems/word-ladder/question?list=neetcode150) | BFS plus per-position letter substitution to generate neighbors |
| 9 | [323. Number of Connected Components in an Undirected Graph](https://neetcode.io/problems/count-connected-components/question?list=neetcode150) | Union-Find counting connected components |
| 10 | [261. Graph Valid Tree](https://neetcode.io/problems/valid-tree/question?list=neetcode150) | Edge-count check plus Union-Find cycle detection |
| 11 | [684. Redundant Connection](https://neetcode.io/problems/redundant-connection/question?list=neetcode150) | Processing edges in order; the first failed union is the answer |
| 12 | [1584. Min Cost to Connect All Points](https://neetcode.io/problems/min-cost-to-connect-points/question?list=neetcode150) | Minimum spanning tree, Kruskal or Prim |
| 13 | [743. Network Delay Time](https://neetcode.io/problems/network-delay-time/question?list=neetcode150) | Standard Dijkstra |
| 14 | [778. Swim in Rising Water](https://neetcode.io/problems/swim-in-rising-water/question?list=neetcode150) | Min-heap plus minimax path |
| 15 | [787. Cheapest Flights Within K Stops](https://neetcode.io/problems/cheapest-flight-path/question?list=neetcode150) | Bellman-Ford with an edge-count limit |
| 16 | [332. Reconstruct Itinerary](https://neetcode.io/problems/reconstruct-itinerary/question?list=neetcode150) | Hierholzer's algorithm for an Eulerian path |
| 17 | [207. Course Schedule](https://neetcode.io/problems/course-schedule/question?list=neetcode150) | Cycle detection via Kahn's algorithm |
| 18 | [210. Course Schedule II](https://neetcode.io/problems/course-schedule-ii/question?list=neetcode150) | Kahn's algorithm producing the topological order |
| 19 | [269. Alien Dictionary](https://neetcode.io/problems/foreign-dictionary/question?list=neetcode150) | Extracting character partial order from adjacent word pairs |

## Module 2: Grid as Graph, the Matrix DFS/BFS Templates

### 1. How a Grid Maps to a Graph

A 2D grid is a special kind of graph: each cell `(r, c)` is a node, and its up/down/left/right neighbors are its edges. The edges are computed implicitly from row/column arithmetic rather than stored in an adjacency list. Connected-component, shortest-path, and region-filling problems on a grid are instances of the same graph-theory problems (connected components, unweighted shortest path, reachability) applied to this implicit adjacency structure.

Both DFS and BFS traverse a grid correctly. The choice depends on what quantity the problem asks for. When the question is only "can this be reached" or "how large is this region" (connectivity, area, region marking), DFS is more direct: it recurses to the bottom and backtracks, with no need to track layer information. When the question is "what is the minimum number of steps/minutes to reach this" (shortest distance, simultaneous multi-source spread), BFS is the only naturally correct choice. The order in which BFS expands layer by layer is exactly the shortest-path order on an unweighted graph; the first path DFS happens to find is not guaranteed to be the shortest.

### 2. DFS Flood Fill Template

The DFS version centers on a function that handles one cell at a time: check bounds and termination conditions first, mark the current cell visited, then recurse into the four directions.

```python
def dfs(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return
    if (r, c) in visited or grid[r][c] == blocked_value:
        return
    visited.add((r, c))
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        dfs(r + dr, c + dc)
```

An outer double loop scans every cell; whenever it finds an unvisited target cell, it starts a new DFS, and each new DFS call corresponds to one new connected component. Visited state can be tracked with a separate `visited` set, or by mutating the grid in place (overwriting target cells with a non-target value) to save memory. The in-place approach makes "blocked cell" and "visited cell" share the same marker, so it stops working once a problem needs to distinguish the two, for example when the same grid must be traversed independently in two separate passes.

### 3. BFS Layered Template

The BFS version keeps a queue for the current frontier and uses a `len(queue)` snapshot to cut the traversal into layers, where each layer corresponds to one unit of distance or time.

```python
from collections import deque

queue = deque(start_cells)          # a single start cell for single-source, all start cells for multi-source
visited = set(start_cells)          # marked at enqueue time, not at dequeue time
dist = 0
while queue:
    for _ in range(len(queue)):     # fix the size of this layer before processing it
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] != blocked_value:
                visited.add((nr, nc))
                queue.append((nr, nc))
    dist += 1
```

Multi-source BFS only changes the initialization step, from enqueueing one start cell to enqueueing all of them at once; the main loop is unchanged. The layered structure does not care which source a cell in a given layer came from, only that it was discovered during the same round of expansion.

### 4. Common Pitfalls

- Marking `visited` at dequeue time instead of enqueue time lets the same cell be discovered and enqueued again by another path while it is still waiting in the queue. This wastes work and can miscount layers. The mark must happen at the moment of enqueueing.
- Checking bounds after accessing the cell: reading `grid[r][c]` before confirming the coordinates are in range triggers an index error directly. The bounds check must run first and short-circuit.
- Reusing the same marker for cells that are inherently impassable (obstacle/water) and cells that are merely visited. This corrupts results whenever the same grid needs multiple independent passes, for example checking reachability from two different regions in sequence.
- Miscounting the layer/distance: `dist` should increment only after an entire layer has been processed. Incrementing it partway through the layer over- or under-counts the distance by one.

### 5. Demo: Multi-Source BFS on Rotting Oranges

Take `grid = [[2,1,1],[1,1,0],[0,1,1]]` as the example (`2` is a rotten orange, `1` is fresh, `0` is empty). All rotten oranges are enqueued together as layer 0. At minute 1, the two fresh neighbors of `(0,0)`, namely `(0,1)` and `(1,0)`, rot together. At minute 2, their fresh neighbors `(0,2)` and `(1,1)` rot. `(2,1)` rots at minute 3, and `(2,2)` rots at minute 4. Every fresh orange finishes rotting after 4 minutes, which is the standard answer for this grid.

```grid-multi-source-bfs-demo
```

## Module 3: Six Problems

### 1. Number of Islands

[NeetCode problem link](https://neetcode.io/problems/number-of-islands/question?list=neetcode150)

The outer loop scans the whole grid. Every time it finds an unvisited land cell (`'1'`), that marks the discovery of a new connected component, so the count increments and a DFS (or BFS) runs to mark every land cell in that component visited, so the outer loop never counts it twice.

| Item | Detail |
|---|---|
| Technique | DFS/BFS flood fill to count connected components |
| Key invariant | Whenever the outer loop triggers a new traversal, the component containing the current cell has never been counted before |
| Time / Space | `O(mn) / O(mn)` |

#### Quick Coding: Number of Islands

```python
def numIslands(grid):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        rows, cols = len(grid), len(grid[0])
        visited = set()

        def dfs(r, c):
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= cols
                or grid[r][c] == "0"
                or (r, c) in visited
            ):
                return
            visited.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dfs(r + dr, c + dc)

        count = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == "1" and (r, c) not in visited:
                    dfs(r, c)
                    count += 1
        return count
```

On `grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]` there are three components: the 2x2 block in the top-left, the single isolated land cell in the middle, and the pair of adjacent cells in the bottom-right. The function returns 3. The `visited` set combined with flood fill guarantees each component is counted exactly once, at the moment it is first encountered.

</details>

### 2. Max Area of Island

[NeetCode problem link](https://neetcode.io/problems/max-area-of-island/question?list=neetcode150)

The template matches Number of Islands exactly. The only difference is that DFS no longer just marks cells, it returns the count of cells reachable from the current cell through recursion, and the outer loop keeps the maximum area seen across all components.

| Item | Detail |
|---|---|
| Technique | DFS flood fill, accumulating component area through the recursive return value |
| Key invariant | `dfs(r, c)` returns exactly the number of still-unvisited cells in the component containing the current cell |
| Time / Space | `O(mn) / O(mn)` |

#### Quick Coding: Max Area of Island

```python
def maxAreaOfIsland(grid):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        visited = set()

        def dfs(r, c):
            if (
                r < 0
                or r >= rows
                or c < 0
                or c >= cols
                or grid[r][c] == 0
                or (r, c) in visited
            ):
                return 0
            visited.add((r, c))
            area = 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                area += dfs(r + dr, c + dc)
            return area

        best = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1 and (r, c) not in visited:
                    best = max(best, dfs(r, c))
        return best
```

On the standard 8-row by 13-column example grid, the largest component covers 6 cells, so the function returns 6. Each cell contributes to the area exactly once through the recursion, and the order in which the contributions are summed does not affect the result since addition is commutative.

</details>

### 3. Pacific Atlantic Water Flow

[NeetCode problem link](https://neetcode.io/problems/pacific-atlantic-water-flow/question?list=neetcode150)

#### 1. In-Depth Strategy: Why Does Forward Search Cause TLE?

- **The Forward Simulation Trap**:
  If you test each inland cell $(r, c)$ by flowing downhill (only moving to neighbors with height $\le$ current cell) toward the Pacific (top/left) and Atlantic (bottom/\right):
  - With $m \times n$ cells, each search may explore up to $O(mn)$ cells in the worst case;
  - Path memoization is complicated by directed flow cycles on plateaus;
  - The naive forward simulation runs in $O((mn)^2)$. For $m, n = 200$, this demands $1.6 \times 10^9$ operations $\implies$ guaranteed **Time Limit Exceeded (TLE)**.

- **The Paradigm Shift: Reverse Flow from the Ocean Borders**:
  Flip the perspective 180 degrees: **Instead of asking where inland water flows down to, ask where ocean water can flow uphill into!**
  - **Reversed Flow Condition**: Water can flow downhill from $A$ to $B$ ($h(A) \ge h(B)$) if and only if water can climb uphill from $B$ to $A$ ($h(A) \ge h(B)$, meaning **neighbor height $\ge$ current height**);
  - Launch a full traversal from the **Pacific borders** (row 0 and col 0), recording all reachable cells in `pacific`;
  - Launch a full traversal from the **Atlantic borders** (row $m-1$ and col $n-1$), recording all reachable cells in `atlantic`;
  - The **set intersection `pacific & atlantic`** yields all coordinates that can flow into both oceans!
  - Each cell is visited at most twice $\implies$ linear $O(mn)$ time complexity.

```text
Reverse Uphill Infiltration Topology:
┌───────────────── Pacific Ocean (Top & Left Borders) ─────────────────┐
│                                                                       │
│  (0,0)  ────► (0,1) ────► (0,2) ... Climbing uphill (height >= prev) │
│   │                                                                   │
│   ▼                                                                   │
│  (1,0) ... Pacific_set (Reverse-reachable cells)                      │
│                                                                       │
│               [ Intersection: Pacific ∩ Atlantic ]                    │
│                                                                       │
│                                  Atlantic_set (Reverse-reachable) ... │
│                                                                   ▲   │
│                                                                   │   │
│            ... (m-1, n-3) ◄──── (m-1, n-2) ◄──── (m-1, n-1)          │
│                                                                       │
└───────────────── Atlantic Ocean (Bottom & Right Borders) ─────────────┘
```

---

#### 2. Key Interview Question: Why is DFS Strongly Preferred Over BFS Here?

While both DFS and BFS achieve the same $O(mn)$ asymptotic time and space bounds, **DFS is overwhelmingly preferred in coding interviews and production implementations**:

| Dimension | DFS (Strongly Recommended ⭐⭐⭐⭐⭐) | BFS (Acceptable but Verbose ⚠️) | Detailed Rationale |
|---|---|---|---|
| **Problem Match** | **Pure Reachability / Connectivity** | Shortest path / layer count | We only care whether a cell is reachable, *not* how many steps or minutes it took. BFS's core level-synchronization strength is completely unused. |
| **Heap / Memory Overhead** | **Zero Queue Overhead** | Requires instantiating two `deque` objects | DFS executes directly on the native call stack, eliminating heap allocation and pointer indirection overhead of `deque`. |
| **Code Conciseness** | **Ultra-Compact (~12 lines core helper)** | Verbose queue setup and loops | DFS recursion is clean; the `visited` set serves as both loop guard and final answer collector. |
| **Cache Locality** | **High (Spatial Locality on Grid)** | Lower (wavefronts jump across grid) | DFS traverses along mountain ridges contiguously in memory, yielding better hardware cache locality. |

---

#### 3. Code Implementations

##### Recommended Solution: Recursive DFS (Clean & Fast)

```python
from typing import List


class Solution:

  def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    rows, cols = len(heights), len(heights[0])
    pacific, atlantic = set(), set()

    def dfs(r, c, visited, prev_height):
      # Out-of-bounds, visited pruning, and uphill flow check (heights[r][c] >= prev_height)
      if (
          r < 0
          or r >= rows
          or c < 0
          or c >= cols
          or (r, c) in visited
          or heights[r][c] < prev_height
      ):
        return

      visited.add((r, c))
      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        dfs(r + dr, c + dc, visited, heights[r][c])

    # 1. Flow uphill from Pacific borders (Top & Left)
    for c in range(cols):
      dfs(0, c, pacific, heights[0][c])
    for r in range(rows):
      dfs(r, 0, pacific, heights[r][0])

    # 2. Flow uphill from Atlantic borders (Bottom & Right)
    for c in range(cols):
      dfs(rows - 1, c, atlantic, heights[rows - 1][c])
    for r in range(rows):
      dfs(r, cols - 1, atlantic, heights[r][cols - 1])

    # 3. Intersection of cells reachable from both oceans
    return [list(coord) for coord in (pacific & atlantic)]
```

##### Alternative Solution: Iterative BFS (Equivalent but Verbose)

<details>
<summary>Click to view BFS implementation</summary>

```python
from collections import deque
from typing import List


class SolutionBFS:

  def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
    rows, cols = len(heights), len(heights[0])

    def get_reachable_bfs(starts):
      queue = deque(starts)
      visited = set(starts)
      while queue:
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
          nr, nc = r + dr, c + dc
          if (
              0 <= nr < rows
              and 0 <= nc < cols
              and (nr, nc) not in visited
              and heights[nr][nc] >= heights[r][c]
          ):
            visited.add((nr, nc))
            queue.append((nr, nc))
      return visited

    pac_starts = [(0, c) for c in range(cols)] + [
        (r, 0) for r in range(1, rows)
    ]
    atl_starts = [(rows - 1, c) for c in range(cols)] + [
        (r, cols - 1) for r in range(rows - 1)
    ]

    pacific = get_reachable_bfs(pac_starts)
    atlantic = get_reachable_bfs(atl_starts)

    return [list(coord) for coord in (pacific & atlantic)]
```

</details>

---

#### 4. Common Pitfalls & Edge Cases

1. **Plateaus and Equal Heights**:
   - The reversed flow condition is `heights[next] >= heights[curr]` with strict equality allowed;
   - Water can flow in both directions between adjacent cells of equal height.
2. **Infinite Recursion on Flat Terrain**:
   - Because adjacent cells of the same height can flow both ways ($A \to B$ and $B \to A$), without checking `(r, c) in visited` at the start of `dfs`, execution would infinite-loop. Marking visited immediately at function entry is mandatory.

| Item | Detail |
|---|---|
| Technique | Two-source reverse uphill DFS with `heights[next] >= heights[curr]` |
| Key invariant | Every cell in the reverse set has a path with monotonically non-decreasing height back to that ocean |
| Time / Space | `O(mn) / O(mn)` (strictly two full traversals; call stack + set memory) |

### 4. Surrounded Regions

[NeetCode problem link](https://neetcode.io/problems/surrounded-regions/question?list=neetcode150)

#### 1. In-Depth Strategy: Why is Forward Search from Inland 'O' Flawed?

- **The Forward Exploration Dilemma**:
  If you start exploring from an inland `'O'`:
  - You must traverse its entire connected component while tracking whether **any cell** touches one of the 4 outer borders;
  - If no cell touched a border after exploration, you must trigger a **second pass** to flip the entire component from `'O'` to `'X'`;
  - If a cell touched a border, you must cancel the flip and mark all visited cells as "safe" to avoid redundant work;
  - This 2-phase state machine (explore-and-verify $\to$ backtrack-or-flip) is error-prone, verbose, and easily leaks state.

- **The Paradigm Shift: Boundary Inoculation (Escape-Route Protection)**:
  Look at the **core mathematical invariant**:
  $$\text{A cell with 'O' is captured (flipped to 'X')} \iff \text{It has NO 4-directional path to ANY of the 4 outer borders}$$
  Conversely: **Only `'O'` cells located on the 4 outer borders, along with inland `'O'` cells connected to them, possess immunity from capture!**

```text
Boundary Inoculation Topology:
┌─────────────────────────── 4 Outer Borders ───────────────────────────┐
│                                                                       │
│  [Border 'O'] ──(Infiltration)──► ['T'] ──(Infiltration)──► ['T']    │
│                                            (Immune / Saved)           │
│                                                                       │
│        ══════════════ Isolated Interior Region ═══════════════        │
│                                                                       │
│                 ['O'] ──► ['O'] ──► ['O']                             │
│             (Trapped / Cut off from borders -> Flips to 'X')          │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

#### 2. The Standard 3-Step Production Pipeline (In-Place)

1. **Step 1: Border Multi-Source Infiltration (Inoculation / Marking Immunity)**:
   - Scan exclusively the 4 outer borders (row 0, row $m-1$, col 0, col $n-1$);
   - Whenever an `'O'` is encountered, run DFS or Multi-Source BFS to temporarily overwrite all connected `'O'`s with `'T'` (Temporary Safe);
   - The board is now partitioned into 3 distinct character classes:
     - `'X'`: Original boundary walls / obstacles;
     - `'T'`: Border-connected survivors possessing immunity;
     - `'O'`: **Trapped interior cells** (isolated from all borders, untouched by border traversals, retaining their original `'O'` tag).
2. **Step 2: Single-Pass Linear Settle**:
   - Iterate through every cell $(r, c)$ in the $m \times n$ grid:
     - If `board[r][c] == 'O'`: It is a trapped interior cell $\to$ **flip in-place to `'X'`**;
     - If `board[r][c] == 'T'`: It is an immune survivor $\to$ **restore in-place to `'O'`**;
     - If `board[r][c] == 'X'`: Leave untouched.
3. **Zero Extra Auxiliary Space Optimization**:
   - **Zero `visited = set()` needed**: Overwriting `'O'` with `'T'` in-place acts as the natural visited guard, cutting auxiliary space overhead down to $O(1)$ (beyond the recursion stack or queue)!

---

#### 3. Dual Implementation Comparison: DFS vs. Multi-Source BFS

##### Recommended Solution 1: Recursive DFS (Cleanest for Whiteboard Interviews, ~6 lines recursion)

```python
from typing import List


class Solution:

  def solve(self, board: List[List[str]]) -> None:
    if not board or not board[0]:
      return

    rows, cols = len(board), len(board[0])

    def dfs(r, c):
      # Out-of-bounds or non-'O' (including 'X' and already visited 'T') returns immediately
      if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != "O":
        return
      board[r][c] = "T"  # In-place temporary mark as immune
      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        dfs(r + dr, c + dc)

    # 1. Inoculate left and right vertical borders
    for r in range(rows):
      dfs(r, 0)
      dfs(r, cols - 1)
    # 2. Inoculate top and bottom horizontal borders
    for c in range(cols):
      dfs(0, c)
      dfs(rows - 1, c)

    # 3. Single-pass in-place resolution: 'O' -> 'X' (captured), 'T' -> 'O' (restored)
    for r in range(rows):
      for c in range(cols):
        if board[r][c] == "O":
          board[r][c] = "X"
        elif board[r][c] == "T":
          board[r][c] = "O"
```

##### Recommended Solution 2: Multi-Source BFS (Production-Grade Stack Safety)

<details>
<summary>Click to view Multi-Source BFS implementation</summary>

```python
from collections import deque
from typing import List


class SolutionBFS:

  def solve(self, board: List[List[str]]) -> None:
    if not board or not board[0]:
      return

    rows, cols = len(board), len(board[0])
    queue = deque()

    # 1. Push all border 'O's into queue and in-place rewrite to 'T'
    for r in range(rows):
      for c in (0, cols - 1):
        if board[r][c] == "O":
          board[r][c] = "T"
          queue.append((r, c))
    for c in range(1, cols - 1):
      for r in (0, rows - 1):
        if board[r][c] == "O":
          board[r][c] = "T"
          queue.append((r, c))

    # 2. Multi-source BFS wavefront expanding inward
    while queue:
      r, c = queue.popleft()
      for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] == "O":
          board[nr][nc] = "T"  # In-place tag prevents duplicate enqueues
          queue.append((nr, nc))

    # 3. Single-pass settle
    for r in range(rows):
      for c in range(cols):
        if board[r][c] == "O":
          board[r][c] = "X"
        elif board[r][c] == "T":
          board[r][c] = "O"
```

</details>

---

#### 4. Edge Cases & Interview Insights

1. **Tiny Grids ($m \le 2$ or $n \le 2$)**:
   - Every cell in the grid lies on the outer boundary. No cell can possibly be surrounded;
   - All `'O'`s are tagged `'T'` in Step 1 and restored to `'O'` in Step 2, running cleanly with zero edge-case branches.
2. **In-Place Mutation Memory**:
   - The extra auxiliary space is strictly $O(1)$ because the character grid itself serves as the visited table.

| Item | Detail |
|---|---|
| Technique | Reverse boundary infiltration tagging (In-place) + single-pass settle |
| Key invariant | After Step 1, cells retaining `'O'` are strictly and exclusively isolated trapped regions |
| Time / Space | `O(mn) / O(1)` auxiliary space (only call stack or queue memory) |

### 5. Rotting Oranges

[NeetCode problem link](https://neetcode.io/problems/rotting-oranges/question?list=neetcode150)

Every rotten orange (value `2`) is enqueued as layer 0 of a multi-source BFS, and the initial count of fresh oranges (value `1`) is recorded. Each layer processes all cells currently in the queue, rots any fresh neighbor and enqueues it while decrementing the fresh count, and the minute counter increments only after the full layer finishes. If the queue empties while fresh oranges remain, some oranges are unreachable and the function returns `-1`.

| Item | Detail |
|---|---|
| Technique | Multi-source layered BFS |
| Key invariant | Cells dequeued at layer `t` are exactly the oranges that rotted during minute `t`; the `len(queue)` snapshot guarantees cells produced during the same minute are never processed early |
| Time / Space | `O(mn) / O(mn)` |

#### Quick Coding: Rotting Oranges

```python
def orangesRotting(grid):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import deque
from typing import List


class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh = 0

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh += 1

        minutes = 0
        while queue and fresh > 0:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        grid[nr][nc] = 2
                        fresh -= 1
                        queue.append((nr, nc))
            minutes += 1

        return minutes if fresh == 0 else -1
```

On `grid = [[2,1,1],[1,1,0],[0,1,1]]`, `fresh` starts at 5. Minute 1 rots `(0,1)` and `(1,0)`. Minute 2 rots `(0,2)` and `(1,1)`. Minute 3 rots `(2,1)`. Minute 4 rots `(2,2)`, bringing `fresh` to zero and ending the loop. The function returns `4`, matching the animation walked through in Module 2.

</details>

### 6. Islands and Treasure (Walls and Gates)

[NeetCode problem link](https://neetcode.io/problems/islands-and-treasure/question?list=neetcode150)

In NeetCode 150, this problem is named **Islands and Treasure** (matching LeetCode classic **286. Walls and Gates**): fill every empty room (`INF` / `2147483647`) with its shortest distance to the nearest gate (`0`), where water/walls (`-1`) block movement.

**Why is a `visited` set completely unnecessary here?**
- All empty land rooms are initialized with `INF`;
- When multi-source BFS reaches an unvisited neighboring cell `(nr, nc)`, `grid[nr][nc] == INF` is satisfied, and we **immediately modify in-place** `grid[nr][nc] = grid[r][c] + 1`;
- The newly written value is strictly less than `INF`. Thus, any future BFS wavefront trying to visit `(nr, nc)` will fail the `grid[nr][nc] == INF` check, **naturally acting as the visited guard with zero extra $O(mn)$ hash set space overhead**!

| Item | Detail |
|---|---|
| Technique | Multi-source in-place BFS (no `visited` set required) |
| Key invariant | The moment an empty room is set to `grid[r][c] + 1`, that value is its true shortest distance to the nearest gate |
| Time / Space | `O(mn) / O(mn)` (only queue memory; zero auxiliary set overhead) |

#### Quick Coding: Islands and Treasure

```python
def islandsAndTreasure(grid):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import deque
from typing import List

INF = 2147483647


class Solution:
    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        rows = len(grid)
        cols = len(grid[0])
        queue = deque()

        # 1. Enqueue all treasure chests / gates (0) as the initial layer
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0:
                    queue.append((i, j))

        # 2. Multi-source in-place BFS diffusion without a visited set
        while queue:
            r, c = queue.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr >= 0 and nr < rows) and (nc >= 0 and nc < cols) and grid[nr][nc] == INF:
                    grid[nr][nc] = grid[r][c] + 1
                    queue.append((nr, nc))
```

On the standard 4x4 example grid, the two gates sit at `(0,2)` and `(3,0)`. BFS expands from both gates concurrently, in-place filling the grid to `[[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]`. Every cell holds its exact Manhattan shortest distance to the closest gate.

</details>

Grid problems are graph problems where the adjacency list is implicit: no graph needs to be built ahead of time, since each cell's neighbors are computed on the fly from row/column arithmetic. The only two decisions that actually matter each time are whether the problem needs connectivity/area or shortest distance/layer count (which decides DFS versus BFS), and whether there is one starting point or many (which decides single-source versus multi-source). Once those two are settled, one of the four templates (single-source DFS, multi-source DFS, single-source BFS, multi-source BFS) applies directly.
## Module 4: General Traversal on Adjacency-List Graphs

The previous module worked on an implicit graph over a grid: nodes are `(row, col)` pairs, edges are always the four up/down/left/right directions, and no separate adjacency structure is needed. This module covers explicit graphs where the problem hands over an adjacency list directly (or one needs to be built first): nodes can be integers, strings, or custom objects such as the ones in Clone Graph, and edges come from a mapping like `adj[node]`, with no implicit direction structure left. The DFS/BFS skeleton stays the same; what changes is that `visited` is no longer a 2D array but a set keyed by the node itself (or some hashable identity of the node), and neighbors are looked up in `adj[node]` instead of computed from coordinate offsets.

### The DFS / BFS Template on Adjacency-List Graphs

DFS can be written recursively or with an explicit stack. Either way, a node must be added to `visited` the moment it is entered, so a cycle in the graph does not cause infinite recursion or an infinite loop.

```python
def dfs(start, adj):
    visited = {start}

    def visit(node):
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                visit(neighbor)

    visit(start)
    return visited
```

BFS uses a queue, and marks `visited` at enqueue time rather than at dequeue time, since otherwise the same node could be discovered by multiple neighbors and enqueued more than once.

```python
from collections import deque


def bfs(start, adj):
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
```

### State Beyond Plain Traversal

A plain `visited` set can only answer "has this node been seen." Some problems need to carry more information along the traversal:

- Clone Graph needs to know which clone corresponds to each original node, so a hash map `old_to_new` replaces the plain `visited` set. If a node already appears in the map, it has already been cloned, and the existing clone is returned directly; this single check handles both deduplication and cycles.
- Word Ladder needs to know how many steps the current word is from `beginWord`, so the BFS queue holds `(word, distance)` pairs instead of bare words. The distance travels with the node through enqueue and dequeue, with no need for a separate pass to track BFS layers.

The two problems below correspond to these two kinds of extra state.

### 7. Clone Graph

DFS from each node, using a hash map `old_to_new` from original node to its clone. The recursive function checks this map first: if the node is already present, it has already been cloned, and the existing clone is returned with no further recursion; if not, a new clone is created, registered in the map, and then each neighbor is cloned recursively and appended to the current clone's `neighbors`.

| Item | Value |
|---|---|
| Composed technique | Adjacency-list DFS plus a hash map from original node to clone |
| Invariant | Each original node in `old_to_new` maps to at most one clone; this same map doubles as the visited check |
| Time / Space | `O(V + E) / O(V)` |

#### Quick Coding: Clone Graph

```python
class Solution:
    def cloneGraph(self, node):
        ...
```

<details>
<summary>Reference answer</summary>

```python
# Definition for a Node.
# class Node:
#     def __init__(self, val=0, neighbors=None):
#         self.val = val
#         self.neighbors = neighbors if neighbors is not None else []


class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if node is None:
            return None

        old_to_new = {}

        def dfs(n):
            if n in old_to_new:
                return old_to_new[n]
            clone = Node(n.val)
            old_to_new[n] = clone
            for neighbor in n.neighbors:
                clone.neighbors.append(dfs(neighbor))
            return clone

        return dfs(node)
```

`old_to_new[n] = clone` must be registered before recursing into the neighbors. If it were registered afterward, a cycle in the graph would cause the same node to be cloned repeatedly, and `dfs` would never terminate.

</details>

### 8. Word Ladder

#### Problem Description (LeetCode 127)
Given two words, `beginWord` and `endWord`, and a dictionary `wordList` of unique words, return the **number of words in the shortest transformation sequence** from `beginWord` to `endWord`, or `0` if no such sequence exists.

**Transformation Rules**:
1. Only **one letter** can be changed at a time;
2. Each transformed word must exist in the word list `wordList` (`beginWord` does not need to be in `wordList`);
3. The sequence length counts all words from start to finish (e.g., `"hit" -> "hot" -> "dot" -> "dog" -> "cog"` has length 5).

```text
Implicit Unweighted Graph & Shortest Path (Example: begin="hit", end="cog")
       [hit] (dist=1, Source)
         │  (substitute 2nd char 'i'->'o')
         ▼
       [hot] (dist=2)
      ┌──┴───────────────┐
 (substitute 1st char) (substitute 1st char)
      ▼                  ▼
    [dot] (dist=3)     [lot] (dist=3)
      │                  │
 (substitute 3rd char) (substitute 3rd char)
      ▼                  ▼
    [dog] (dist=4)     [log] (dist=4)
      └──┬───────────────┘
         │  (substitute 1st char)
         ▼
       [cog] (dist=5, Target reached!)
```

#### Graph Modeling & Core Engineering Takeaways

1. **Graph Theoretical Formulation**:
   - **Vertices $V$**: `beginWord` and all words in `wordList`;
   - **Unweighted Edges $E$**: An undirected edge exists between two words if their Hamming distance is exactly 1 (differ by 1 character);
   - **Target Algorithm**: Shortest path on an unweighted graph $\implies$ **Breadth-First Search (BFS)** guarantees finding the global shortest sequence on first arrival.

2. **Takeaway 1: Neighbor Generation Complexity ($\mathcal{O}(26 \cdot L^2)$ vs $\mathcal{O}(N \cdot L)$)**:
   - **Strategy A (Scan Dictionary)**: Compare the current word against every word in `wordList` character-by-character. Cost per step is $\mathcal{O}(N \cdot L)$. For $N = 5000, L = 5$, this requires $5000 \times 5 = 25,000$ comparisons per step!
   - **Strategy B (Enumerate 26 Letters + Set Lookup)**: Enumerate $L$ positions, substitute 26 letters ('a'~'z'), slice the string in $\mathcal{O}(L)$, and check existence in `word_set` in $\mathcal{O}(L)$. Cost per step is $\mathcal{O}(26 \cdot L^2)$. For $L = 5$, this is $26 \times 25 = 650$ operations!
   - **Conclusion**: $650 \ll 25000$, Strategy B is almost 40x faster!

3. **Takeaway 2: In-place Set Deletion instead of Visited Set**:
   - Instead of maintaining a separate `visited = set()`, we can directly remove `next_word` from `word_set` upon enqueueing (`word_set.remove(next_word)`).
   - Because BFS explores level-by-level, any subsequent attempt to visit this word in a deeper level can never yield a shorter path. Removing it in-place achieves validity checking, deduplication, and pruning in a single $\mathcal{O}(1)$ step!

| Item | Value |
|---|---|
| Composed technique | Implicit Graph BFS + 26-Letter Substitution Neighbor Generation + In-place Set Pruning |
| Invariant | The `dist` of the popped word is strictly the shortest distance from `beginWord` (guaranteed by BFS layer-by-layer expansion) |
| Time / Space | Time $\mathcal{O}(26 \cdot N \cdot L^2)$, space $\mathcal{O}(N \cdot L)$ ($N$ = dictionary size, $L$ = word length) |

#### Quick Coding: Word Ladder

```python
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        ...
```

<details>
<summary>Reference answer</summary>

```python
import string
from collections import deque
from typing import List


class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        word_set = set(wordList)
        if endWord not in word_set:
            return 0

        queue = deque([(beginWord, 1)])
        word_set.discard(beginWord)

        while queue:
            word, dist = queue.popleft()
            if word == endWord:
                return dist
            for i in range(len(word)):
                for c in string.ascii_lowercase:
                    if c == word[i]:
                        continue
                    next_word = word[:i] + c + word[i + 1:]
                    if next_word in word_set:
                        word_set.remove(next_word)
                        queue.append((next_word, dist + 1))
        return 0
```

Generating neighbors by substitution produces `O(L · 26)` candidates per word, and each candidate requires building a new string and a hash-set lookup at `O(L)`; comparing directly against every word in the list would cost `O(N · L)` per word instead, which is clearly worse once `N` is much larger than 26. When `endWord` is not in the given word list, `0` is returned immediately without entering the BFS.

</details>

---

## Module 5: Disjoint Set Union (Union-Find & Cycle Detection)

> **🎯 Two Primary Pillars of Union-Find**:
> 1. **Dynamic Connected Components**: Performing `union(u, v)` for each edge dynamically maintains connectivity clusters without building adjacency lists or running BFS/DFS traversals.
> 2. **Undirected Cycle Detection**: If `find(u) == find(v)` when adding edge $(u, v)$, both endpoints are already connected. **Adding this edge strictly forms a closed cycle**! This is the foundational criterion for Kruskal's MST and Redundant Connection.

Union-Find (also called disjoint set union) maintains a collection of disjoint sets and supports two core operations:
- `find(x)`: Returns the representative (root) of the set containing $x$, amortized to $\mathcal{O}(\alpha(n)) \approx \mathcal{O}(1)$ with path compression;
- `union(a, b)`: Merges the two sets containing $a$ and $b$; **returns `False` if $a$ and $b$ already share the same root (Cycle Alarm!)**.

### The Core Design of Union-Find

An array `parent` represents all the sets: `parent[x]` stores the parent of `x`, and a root's `parent` points to itself. On initialization, every node is its own set: `parent[x] = x`, `size[x] = 1`.

`find(x)` walks up the `parent` chain until it reaches a root `r` where `parent[r] == r`. Without any optimization, this chain can degenerate into a single path of length `n`, making `find` cost `O(n)`. Path compression reattaches every node on the search path directly to the root, so a later `find` on any of those nodes takes just one hop. The version used below is iterative: it first walks to the root, then walks the path a second time to attach every node directly to it.

```python
def find(self, x):
    root = x
    while root != self.parent[root]:
        root = self.parent[root]
    while x != root:
        self.parent[x], x = root, self.parent[x]
    return root
```

A recursive version is shorter (`parent[x] = find(parent[x]); return parent[x]`) and behaves the same, but each `find` call consumes one level of the Python call stack; once the number of nodes reaches the tens of thousands, this risks hitting the recursion limit. The three problems in this module all use the iterative version above.

`union(a, b)` first finds the roots `ra` and `rb` of `a` and `b`. If they are equal, `a` and `b` are already in the same set, so no merge happens, and the connected-component count must not be decremented again. Otherwise, the tree with the smaller `size` is attached under the root of the tree with the larger `size`, and the new root's `size` is updated:

```python
def union(self, a, b):
    ra, rb = self.find(a), self.find(b)
    if ra == rb:
        return False
    if self.size[ra] < self.size[rb]:
        ra, rb = rb, ra
    self.parent[rb] = ra
    self.size[ra] += self.size[rb]
    return True
```

Union by rank (an estimate of tree height) is an equivalent alternative strategy; both share the same idea of attaching the smaller tree under the larger one so trees do not grow deeper than necessary. Using path compression alone, or union by size/rank alone, bounds a single operation at `O(log n)` in the worst case. Combining both bounds any sequence of `n` operations to an amortized `O(α(n))` per operation, where `α` is the inverse Ackermann function, which stays at or below 4 for any realistic `n` and can be treated as approximately `O(1)`.

Common pitfalls:

- Forgetting path compression, or forgetting union by size/rank: Union-Find degenerates into a linked list, and `find` becomes `O(n)`.
- When counting connected components, decrementing the count on every `union` call without first checking whether the two ends are already connected: merging an already-connected pair again decrements the count an extra time, undercounting the result.
- Not aligning 1-indexed versus 0-indexed nodes: problems such as Redundant Connection number nodes starting at 1, so the `parent` array must be sized `n + 1` and used starting at index 1; initializing it as if the nodes were 0-indexed either introduces a spurious node at index 0 or runs out of bounds at node `n`.

The demo below runs a sequence of `union` calls over 8 nodes (indexed 0 through 7). The first 6 `union` calls deliberately skip the by-size optimization, simply attaching one root under the other, which builds two chains of different lengths. The 7th step calls `find` on the tail of the longer chain, showing path compression flattening every node on that chain straight onto the root. The final step unions the two chains' sets by size, showing exactly which root union-by-size keeps.

```union-find-demo
```

### 9. Number of Connected Components in an Undirected Graph

Run `union` once per edge. A `count` that starts at `n` and decrements on every successful `union` tracks the current number of connected components. An equivalent approach is to call `find` once per node after processing all edges and collect the distinct roots in a set, whose size is the component count; the decrementing `count` saves that extra pass.

| Item | Value |
|---|---|
| Composed technique | Union-Find, running `union` over every edge |
| Invariant | `count` always equals the current number of connected components: it starts at `n` and decreases by one on every successful `union` |
| Time / Space | `O((V + E) · α(V)) / O(V)` |

#### Quick Coding: Number of Connected Components in an Undirected Graph

```python
class Solution:
    def countComponents(self, n, edges):
        ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def find(self, x):
        root = x
        while root != self.parent[root]:
            root = self.parent[root]
        while x != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.count -= 1
        return True


class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        uf = UnionFind(n)
        for a, b in edges:
            uf.union(a, b)
        return uf.count
```

`uf.count` only decreases when `union` actually merges two distinct sets, so processing an edge inside an already-merged component has no effect on it, and the order in which edges are processed does not change the final result.

</details>

### 10. Graph Valid Tree

A graph is a valid tree if and only if it has exactly `n - 1` edges and is fully connected (once the edge count is fixed at `n - 1`, being acyclic and being connected are equivalent: one more edge always creates a cycle, one fewer always leaves the graph disconnected). The edge count is checked first, returning `False` immediately if it is wrong. When the count is correct, Union-Find processes every edge: any `union` that fails (its two endpoints are already in the same set) means a cycle exists, so `False` is returned immediately; after all edges are processed, the component count should be exactly `1`.

| Item | Value |
|---|---|
| Composed technique | Edge-count check (`== n - 1`) plus Union-Find cycle detection |
| Invariant | Any failed `union` implies a cycle; once the edge count already equals `n - 1`, being acyclic is equivalent to being fully connected |
| Time / Space | `O(V · α(V)) / O(V)` |

#### Quick Coding: Graph Valid Tree

```python
class Solution:
    def validTree(self, n, edges):
        ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def find(self, x):
        root = x
        while root != self.parent[root]:
            root = self.parent[root]
        while x != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.count -= 1
        return True


class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1:
            return False

        uf = UnionFind(n)
        for a, b in edges:
            if not uf.union(a, b):
                return False

        return uf.count == 1
```

The edge-count check is a fast-fail pruning step, not a logically independent requirement: without it, "no `union` ever fails" together with "the final `count` equals `1`" already forces the edge count to equal `n - 1`, since every successful `union` decrements `count` by exactly one, and going from `n` down to `1` takes exactly `n - 1` successful merges. Checking the edge count up front just returns immediately when it is obviously wrong, without running Union-Find at all.

</details>

### 11. Redundant Connection

Edges are processed in input order, with `union` run on each one. The problem guarantees that removing one specific edge leaves a tree, so the first edge whose `union` fails (its two endpoints are already in the same set) is the extra edge that turned the tree into a graph with a cycle.

| Item | Value |
|---|---|
| Composed technique | Union-Find over edges in input order; the first failed `union` is the answer |
| Invariant | After processing the first `i` edges, Union-Find always reflects the connectivity of the graph formed by those `i` edges |
| Time / Space | `O(E · α(V)) / O(V)` |

#### Quick Coding: Redundant Connection

```python
class Solution:
    def findRedundantConnection(self, edges):
        ...
```

<details>
<summary>Reference answer</summary>

```python
from typing import List


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        root = x
        while root != self.parent[root]:
            root = self.parent[root]
        while x != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


class Solution:
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        uf = UnionFind(n + 1)  # nodes are numbered from 1, so size the array to n + 1 and leave index 0 unused

        for a, b in edges:
            if not uf.union(a, b):
                return [a, b]

        return []
```

Nodes are numbered from 1 to `n`, so `UnionFind` is sized `n + 1` with index 0 left unused, avoiding a misalignment between node 1 and array index 0. The problem guarantees exactly one redundant edge exists, so `union` is guaranteed to fail on some edge, and the trailing empty-list branch is never actually reached.

</details>
## Module 6: Minimum Spanning Tree, Prim and Kruskal

Given a weighted, connected, undirected graph, a minimum spanning tree (MST) is a tree that includes all `n` nodes with the smallest possible total edge weight. It has exactly `n-1` edges and contains no cycle. MST is the standard model for "connect everything at minimum total cost." Two algorithms dominate: Prim and Kruskal. Both rely on greedily adding edges, differing in the order and criteria used to add them.

### Prim's Algorithm

Prim starts from an arbitrary node and grows a single connected tree outward. A min-heap holds candidate edges that connect the current tree to nodes outside it. Each step pops the minimum-weight candidate: if it connects to a node not yet in the tree, both the edge and the node are added, and the new node's outgoing edges are pushed onto the heap. If the popped edge connects to a node already in the tree (a stale entry left in the heap), it is discarded.

```text
visited = {start}
heap = edges out of start
cost = 0
while heap and len(visited) < n:
    w, u, v = heappop(heap)
    if v in visited:
        continue
    visited.add(v)
    cost += w
    for nxt, nw in adj[v]:
        if nxt not in visited:
            heappush(heap, (nw, v, nxt))
```

Each edge is pushed onto the heap at most once (from whichever endpoint gets visited first). Heap push and pop are `O(log E)`, giving an overall time complexity of `O(E log E)`.

### Kruskal's Algorithm

Kruskal sorts every edge in the graph by weight ascending, then processes them one at a time. Whether an edge can be added is decided by Union-Find: if the two endpoints are not yet in the same component, adding the edge cannot create a cycle, so the algorithm unions them and adds the weight to the running cost. If the endpoints are already connected, adding the edge would only create a cycle, so it is skipped. The algorithm stops once `n-1` edges have been added, since at that point every node is connected.

```text
sort edges by weight
dsu = UnionFind(n)
cost = 0
used = 0
for u, v, w in edges:
    if dsu.union(u, v):
        cost += w
        used += 1
        if used == n - 1:
            break
```

Sorting is `O(E log E)`. Union-Find operations (with path compression and union by rank) are close to `O(1)` each. The sort dominates, so the total complexity is also `O(E log E)`.

### Comparing the Two

| Dimension | Prim | Kruskal |
|---|---|---|
| Growth pattern | Grows a single tree outward from a seed node | Sorts all edges globally, merges components via Union-Find |
| Data structures | Adjacency list + min-heap of candidate edges | Sorted edge list + Union-Find |
| Fits best when | The graph is given as an adjacency list, especially when dense (distances can be computed on the fly instead of listing every edge up front) | The graph is already given as a flat edge list |
| Disconnected graphs | Not detected directly; if `visited` ends up smaller than `n`, the graph is disconnected | Detected naturally: if fewer than `n-1` edges get added, the result is a minimum spanning forest |

### Common Pitfalls

- In Prim, forgetting to check `visited` before counting a popped edge toward the cost. The same node can then be added more than once, introducing a cycle and inflating the total cost.
- In Kruskal, not using the return value of the union operation to decide whether an edge would form a cycle, so cycle-forming edges are not skipped.
- Assuming a spanning tree always exists. If the graph is disconnected, Prim's heap empties before `visited` reaches size `n`, and Kruskal can only add fewer than `n-1` edges, producing a spanning forest. Both cases need explicit handling, typically returning `-1` or the forest itself.

### 12. Min Cost to Connect All Points

The input is `n` points on a plane, any two of which can be connected, with edge weight equal to their Manhattan distance. This is an implicit complete graph: listing every edge explicitly produces `n*(n-1)/2` edges.

The Kruskal approach builds all `n*(n-1)/2` edges, sorts them, and adds them via Union-Find, a direct application of the template above. The Prim approach never materializes an edge list: starting from an arbitrary point (say index `0`), it computes the distance from the current tree to every unvisited point on the fly and pushes those distances onto a heap, with the heap top giving the next edge to add.

Both approaches produce `O(n^2)`-scale edge or heap operations in the worst case. Kruskal is `O(n^2 log n)`, dominated by the sort. Heap-based Prim is also `O(n^2 log n)`, but it never builds and sorts all `n*(n-1)/2` edges up front; it computes distances incrementally as the tree grows and pushes them onto the heap. For larger `n`, Prim skips the global sort and the up-front edge-list materialization, making it the more common choice in practice.

| Item | Content |
|---|---|
| Combined technique | Kruskal (build all edges + sort + Union-Find) or Prim (min-heap, computing Manhattan distances on the fly as the tree grows) |
| Key invariant | Kruskal: points inside the same Union-Find component are already connected. Prim: for every point outside `visited`, the candidate distance held is the current shortest known distance from the tree to that point |
| Time / space | Both `O(n^2 log n) / O(n^2)`; Prim skips the explicit sort and edge-list materialization |

#### Quick Coding: Min Cost to Connect All Points

```python
def minCostConnectPoints(points):
    ...
```

<details>
<summary>Reference answer</summary>

Kruskal version:

```python
from typing import List


class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        edges = []
        for i in range(n):
            xi, yi = points[i]
            for j in range(i + 1, n):
                xj, yj = points[j]
                dist = abs(xi - xj) + abs(yi - yj)
                edges.append((dist, i, j))
        edges.sort()

        total = 0
        used = 0
        for dist, u, v in edges:
            if union(u, v):
                total += dist
                used += 1
                if used == n - 1:
                    break
        return total
```

Prim version:

```python
import heapq
from typing import List


class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        visited = [False] * n
        min_heap = [(0, 0)]  # (dist, point_index), starting from point 0
        total = 0
        edges_used = 0

        while min_heap and edges_used < n:
            dist, u = heapq.heappop(min_heap)
            if visited[u]:
                continue
            visited[u] = True
            total += dist
            edges_used += 1
            xu, yu = points[u]
            for v in range(n):
                if not visited[v]:
                    xv, yv = points[v]
                    nd = abs(xu - xv) + abs(yu - yv)
                    heapq.heappush(min_heap, (nd, v))

        return total
```

Both versions return `20` on `points = [[0,0],[2,2],[3,10],[5,2],[7,0]]` and `18` on `points = [[3,12],[-2,5],[-4,1]]`, confirmed by running both against these inputs. Kruskal builds and sorts every candidate edge first, the most direct application of the module's template. Prim only computes distances between the current tree and the remaining unvisited points, avoiding a global sort; at most `O(n)` candidate distances for unvisited points sit in the heap at any point (stale entries for already-visited points are discarded on pop). It is the more common choice as `n` grows.

</details>

## Module 7: Shortest Path, Dijkstra and Bellman-Ford

For shortest-path problems, do not start by memorizing a template. First determine the constraints:

- If all edge weights are non-negative and there are no extra restrictions: prioritize Dijkstra.
- If there are negative edges, or the path length / number of edges has an explicit limit: consider Bellman-Ford.
- If the graph is unweighted: BFS is the shortest-path algorithm.
- If the state includes dimensions beyond the node, such as "how many transfers have been used": expand the state to `(node, state)`.

### Dijkstra Review

Dijkstra applies to graphs with non-negative weights. The core idea is to use a min-heap to pop the state with the smallest current cost each time, then relax its neighbors.

```text
dist[start] = 0
heap = [(0, start)]

while heap:
  d, u = heappop(heap)
  if d != dist[u]:
    continue
  for v, w in graph[u]:
    if d + w < dist[v]:
      dist[v] = d + w
      heappush(heap, (dist[v], v))
```

For Cheapest Flights, standard Dijkstra is not enough, because "the lowest price to the same city" is not necessarily the final best state. A more expensive state that used fewer flights may still be feasible later. Therefore, the Dijkstra formulation must expand the state to `(cost, city, stopsUsed)`.

### 13. Network Delay Time

The input is a directed weighted graph `times`, where each edge `[u, v, w]` means a signal takes `w` time units to travel from node `u` to node `v`, and all weights are non-negative. A signal starts at node `k`. Find the minimum time for all `n` nodes to receive the signal; if some node is unreachable, return `-1`. This is the worked example missing from the Dijkstra review above: a single source, one shortest distance per node, with no need to track an extra state dimension such as the number of edges used, unlike Cheapest Flights.

| Item | Detail |
|---|---|
| Technique combination | Standard Dijkstra, single-source shortest path |
| Key invariant | The first time a node is popped from the heap, its `d` is the final shortest distance; any other copy of the same node still in the heap is a stale candidate |
| Time / Space | Time `O(E log V)`, space `O(V + E)` |

#### Quick Coding: Network Delay Time

```python
def networkDelayTime(times, n, k):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq
from collections import defaultdict
from typing import List


class Solution:
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        graph = defaultdict(list)
        for u, v, w in times:
            graph[u].append((v, w))

        INF = float("inf")
        dist = [INF] * (n + 1)
        dist[k] = 0
        heap = [(0, k)]

        while heap:
            d, u = heapq.heappop(heap)
            if d != dist[u]:
                continue
            for v, w in graph[u]:
                if d + w < dist[v]:
                    dist[v] = d + w
                    heapq.heappush(heap, (dist[v], v))

        farthest = max(dist[1:n + 1])
        return farthest if farthest < INF else -1
```

The initialization of `dist` and `heap`, and the relax step, match the pseudocode in the Dijkstra review: `if d != dist[u]: continue` skips a stale copy left in the heap, and relaxation only happens when `d + w < dist[v]`. Once every reachable node has been relaxed, the maximum value in `dist[1:n+1]` is the time for all nodes to receive the signal, bounded by whichever node gets notified last. If that maximum is still `INF`, some node is unreachable from `k`, and the answer is `-1`.

</details>

### 14. Swim in Rising Water

The input is an `n x n` elevation grid `grid`. Water starts at the top-left cell and rises over time; a cell becomes enterable once the current water level is at least its elevation. The time to complete a path equals the maximum elevation visited along that path. The goal is the minimum value of that path bottleneck over all paths from the top-left cell to the bottom-right cell, a minimax path problem.

This can be solved with the same heap-driven relaxation loop as Dijkstra or Prim's algorithm: the heap holds `(bottleneck, r, c)`, where `bottleneck` is the largest elevation seen so far on the best known path to `(r, c)`. Each pop takes the state with the smallest bottleneck, and relaxing a neighbor sets its bottleneck to `max(bottleneck, grid[neighbor])`. The first time the bottom-right cell is popped, its bottleneck is the answer. An equivalent approach binary-searches the water level `t` and uses BFS/DFS to check whether cells with elevation `<= t` connect the two corners; this section focuses on the heap-based version.

| Item | Detail |
|---|---|
| Technique combination | Min-heap + minimax path, a Dijkstra/Prim variant |
| Key invariant | The heap top always holds the state with the smallest path bottleneck seen so far; relaxation updates a neighbor's bottleneck with `max` |
| Time / Space | Time `O(n^2 log n)`, space `O(n^2)` |

#### Quick Coding: Swim in Rising Water

```python
def swimInWater(grid):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq
from typing import List


class Solution:
    def swimInWater(self, grid: List[List[int]]) -> int:
        n = len(grid)
        visited = [[False] * n for _ in range(n)]
        heap = [(grid[0][0], 0, 0)]
        visited[0][0] = True

        while heap:
            bottleneck, r, c = heapq.heappop(heap)
            if r == n - 1 and c == n - 1:
                return bottleneck
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                    visited[nr][nc] = True
                    heapq.heappush(heap, (max(bottleneck, grid[nr][nc]), nr, nc))

        return -1
```

`visited` plays the same role as `dist` in Dijkstra: a cell is pushed onto the heap at most once, because the heap guarantees that the first time it is popped, its `bottleneck` is already the smallest achievable value for reaching it. Relaxing a neighbor sets its state to `max(bottleneck, grid[nr][nc])`, tracking the largest elevation seen along the path. The function returns as soon as the bottom-right cell is popped, without waiting for the heap to empty.

</details>

### Why Bellman-Ford Is More Natural

The semantics of Bellman-Ford fit the next problem exactly:

> After the `i`-th relaxation round, `prices[x]` represents the lowest price to reach city `x` using at most `i` edges.

The problem allows at most `k` stops, which means at most `k + 1` flights. Therefore, only `k + 1` relaxation rounds are needed.

The key point is that each round must read from the previous round's `prices` and write into `nextPrices`. Updating in place directly would chain multiple edges together within the same round, equivalent to secretly using more flights than the current round allows.

### 15. Cheapest Flights Within K Stops

Input:

```text
n = 4
flights = [
  [0, 1, 100],
  [1, 2, 100],
  [2, 0, 100],
  [1, 3, 600],
  [2, 3, 200],
]
src = 0
dst = 3
k = 1
```

At most `1` stop is allowed, so at most `2` flights can be taken. The valid answer is `0 -> 1 -> 3`, with total price `700`. Is the path `0 -> 1 -> 2 -> 3` cheaper? It costs `400`, but it requires `3` flights, which exceeds the limit, so it cannot be used.

```bellman-demo
cheapest-flights
```

#### Optimized Bellman-Ford

The basic formulation is to perform a fixed `k + 1` rounds. There are two optimization points:

1. In each round, use `nextPrices = prices.copy()` to guarantee that this round only transitions from the previous round's states.
2. If a round makes no updates at all, further relaxations cannot improve anything, so it can stop early.

```python
from typing import List

class Solution:
    def findCheapestPrice(
        self,
        n: int,
        flights: List[List[int]],
        src: int,
        dst: int,
        k: int,
    ) -> int:
        INF = float("inf")
        prices = [INF] * n
        prices[src] = 0

        for _ in range(k + 1):
            next_prices = prices.copy()
            changed = False

            for start, end, cost in flights:
                if prices[start] == INF:
                    continue

                candidate = prices[start] + cost
                if candidate < next_prices[end]:
                    next_prices[end] = candidate
                    changed = True

            prices = next_prices
            if not changed:
                break

        return -1 if prices[dst] == INF else prices[dst]
```

#### Correctness Intuition

At the start of round `0`, only `src` has price `0`, representing that "without taking any flight, you can only reach the starting point."

Round `1` only allows transitions from the results of round `0`, so it can only reach cities accessible with one flight.

Round `2` only allows transitions from the results of round `1`, so it reaches cities accessible with at most two flights.

Continuing until round `k + 1` exactly covers the maximum number of flights allowed by the problem. Because each round reads the old array and writes the new array, continuous chaining such as `0 -> 1 -> 2 -> 3` cannot occur within the same round.

#### Complexity

- Time: `O((k + 1) * E)`, where `E` is the number of flights.
- Space: `O(V)`, keeping only `prices` and `next_prices`.

This is usually more direct than pushing all `(city, stops)` states into a heap, and it is also better suited to explaining the constraint of "at most how many edges."

#### Common Pitfalls

- Treating `k stops` as at most `k` edges. In reality it means at most `k + 1` flights.
- Updating `prices` in place, causing multiple edges to be chained within the same round.
- Forgetting to skip flights where `prices[start] == INF`.
- Returning early using the current price of `dst`; Bellman-Ford must wait until the current round of relaxation is complete.
- Using standard Dijkstra's `dist[city]` to discard a state that is more expensive but uses fewer stops.
## Module 8: Eulerian Path, Reconstruct Itinerary

Most earlier graph problems ask for visiting every node once (topological sort, number of islands, course schedule). Reconstruct Itinerary asks for using every ticket once. A ticket is an edge, so this is an Eulerian path problem: find a path that uses every edge in the graph exactly once, revisiting nodes as needed.

In general, an Eulerian path exists only when at most one node has out-degree exceeding in-degree by 1 (the start), at most one node has in-degree exceeding out-degree by 1 (the end), and every other node has equal in-degree and out-degree. When every node has equal in-degree and out-degree, the Eulerian path degenerates into an Eulerian circuit (start and end coincide). LeetCode guarantees a valid itinerary exists for the given input, so the existence check is not needed in code.

The problem also requires returning the lexicographically smallest itinerary when multiple valid ones exist. The standard algorithm for this is Hierholzer's algorithm.

### Hierholzer's Algorithm

Each departure airport keeps a min-heap (or sorted list) of destinations, so the lexicographically smallest unused destination is always available first. DFS starts at `JFK`. Each step pops the smallest unused destination from the current airport and recurses into it. Once an airport's heap is exhausted (no unused ticket leaves it), the airport is appended to `route`. This append happens after the recursive call returns, in postorder.

The postorder append is what makes the algorithm correct. DFS advances greedily along the lexicographically smallest choice at each step. If that greedy path runs into a dead end (an airport with unused tickets elsewhere in the graph, but no outgoing edge left in the current branch), postorder guarantees the dead-end node gets recorded exactly where the dead end occurred. Once the recursive call returns, the outer call resumes trying the departure airport's remaining unused tickets, splicing the skipped sub-loop back into the main path. By the time DFS finishes, the order of nodes in `route` is their finishing order: earliest dead end first, true starting point last. Reversing `route` produces a valid itinerary in the order tickets were actually used.

```python
import heapq
from collections import defaultdict
from typing import List


class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = defaultdict(list)
        for src, dst in tickets:
            heapq.heappush(graph[src], dst)

        route = []

        def dfs(node: str) -> None:
            while graph[node]:
                nxt = heapq.heappop(graph[node])
                dfs(nxt)
            route.append(node)

        dfs("JFK")
        return route[::-1]
```

### Trace by Hand

`tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]`

After building the heaps, each airport's sorted candidate destinations are: `JFK -> [ATL, SFO]`, `SFO -> [ATL]`, `ATL -> [JFK, SFO]`.

Calling `dfs("JFK")`:
1. `JFK` pops `ATL` (leaving `[SFO]`), recurses into `dfs("ATL")`.
2. `ATL` pops `JFK` (leaving `[SFO]`), recurses into `dfs("JFK")`.
3. `JFK` pops `SFO` (leaving `[]`), recurses into `dfs("SFO")`.
4. `SFO` pops `ATL` (leaving `[]`), recurses into `dfs("ATL")`.
5. `ATL` pops `SFO` (leaving `[]`), recurses into `dfs("SFO")`.
6. `SFO` has no candidates left, appends `SFO` in postorder: `route = [SFO]`.
7. Back in step 5's `ATL`, no candidates left, appends `ATL`: `route = [SFO, ATL]`.
8. Back in step 3's `SFO`, no candidates left, appends `SFO`: `route = [SFO, ATL, SFO]`.
9. Back in step 2's `JFK`, no candidates left, appends `JFK`: `route = [SFO, ATL, SFO, JFK]`.
10. Back in step 1's `ATL`, no candidates left, appends `ATL`: `route = [SFO, ATL, SFO, JFK, ATL]`.
11. Back in the outermost `JFK`, no candidates left, appends `JFK`: `route = [SFO, ATL, SFO, JFK, ATL, JFK]`.

Reversing `route` gives `["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"]`, matching the expected output, confirmed by running the script.

### Common Pitfalls

- Not sorting each airport's destination list lexicographically (or not maintaining it as a heap). DFS then pops a candidate that is not the smallest available, and even if every ticket eventually gets used, the resulting itinerary may not be the lexicographically smallest one.
- Appending a node to the route on entry to an airport, before its heap of outgoing tickets is exhausted. This preorder placement records an airport's position before its dead-end sub-loops get resolved, so once a dead-end branch is hit, there is no way to splice the remaining tickets back in. A concrete case is `tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]`: `KUL` is the lexicographically smallest option out of `JFK`. Appending it on entry would end the route after a single ticket, since `KUL` has no outgoing edge. With postorder, `KUL` is still chosen first by the greedy rule, but it gets recorded immediately because it has no outgoing edge (`route = [KUL]` at that point), and DFS backtracks to try `JFK`'s other ticket, `NRT`. The final reversed result is `["JFK", "NRT", "JFK", "KUL"]`, with `KUL` correctly placed at the end, confirmed by running the script.
- Forgetting to reverse `route` at the end. The unreversed order has the endpoint first and the starting airport last, the opposite of the actual itinerary.
- Using a plain list with `list.remove` or index-based deletion to simulate consuming a ticket. Without sorting first, or with an incorrect deletion, an already-used ticket can get picked again, or the destination chosen is not the lexicographically smallest one available.

### 16. Reconstruct Itinerary

| Item | Content |
|---|---|
| Combined technique | Hierholzer's algorithm: a min-heap of outgoing destinations per node, DFS appends nodes in postorder, the whole result is reversed at the end |
| Key invariant | A node is appended to `route` only once its heap is exhausted (every ticket leaving it has been used), guaranteeing dead ends and sub-loops get spliced into the correct position |
| Time / space | Building the heaps is `O(E log E)` (`E` is the number of tickets), DFS visits every edge once for `O(E)`, total `O(E log E)`; space `O(E)` |

#### Quick Coding: Reconstruct Itinerary

```python
def findItinerary(tickets):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import heapq
from collections import defaultdict
from typing import List


class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        graph = defaultdict(list)
        for src, dst in tickets:
            heapq.heappush(graph[src], dst)

        route = []

        def dfs(node: str) -> None:
            while graph[node]:
                nxt = heapq.heappop(graph[node])
                dfs(nxt)
            route.append(node)

        dfs("JFK")
        return route[::-1]
```

`graph[node]` is a min-heap holding the still-unused ticket destinations leaving `node`, with the heap top always the lexicographically smallest candidate. DFS pops and recurses into a destination until an airport's heap is exhausted, at which point the airport is appended to `route`. This exhaust-then-append (postorder) ordering guarantees dead ends and sub-loops are recorded at the correct position without any extra backtracking logic. `route` is reversed at the end because postorder records the endpoint first and the starting airport last.

</details>

---

## Module 9: Topological Sort (DAG, Kahn's Algorithm & Cycle Detection)

> **Core Definition**: A topological sort of a **Directed Acyclic Graph (DAG)** is a linear ordering of its vertices such that for every directed edge $(u, v)$, vertex $u$ appears before $v$ in the ordering.
>
> **4-Step Mental Model**: **1. Count Indegrees -> 2. Enqueue In-0 Nodes -> 3. Relax Successors -> 4. Check Node Count (|V|)**.

```topo-demo
```

Topological sorting is not an algorithm for "sorting an array." It is a graph algorithm that arranges a set of dependencies into a legal execution order. Whenever a problem mentions "precedence, dependencies, courses, build order, letter order, or task scheduling," the first question should be: can these relationships be modeled as a directed graph?

If each edge `u -> v` means `u` must come before `v`, then topological sorting outputs a sequence such that every edge in the graph points from left to right in that sequence. Only a **DAG (Directed Acyclic Graph)** is guaranteed to have a topological order; if there is a cycle, such as `a -> b -> c -> a`, it represents mutual dependency and no legal order can be given.

### What Is Topological Sorting

Given a directed graph `G = (V, E)`:

- `V` is the set of nodes, such as courses, tasks, or characters.
- `E` is the set of constraints, such as `pre -> course`, `dependency -> target`, or `smaller letter -> larger letter`.
- A topological order is a linear ordering containing all nodes and satisfying: for every edge `u -> v`, `u` appears before `v`.

It solves the problem of turning a **partial order into a linear order**. A partial order tells you only some precedence relationships, such as `h < e` and `e < r`, but it does not say that all characters can be compared directly. Topological sorting gives any legal linear answer without violating the known constraints.

### Kahn's Algorithm: Start from Nodes with No Prerequisites

Kahn's algorithm is the most intuitive BFS formulation in interviews.

1. Build the graph and count the indegree `indegree` of each node.
2. Put all nodes with indegree `0` into a queue.
3. Each time, pop a node `u` and add it to the answer.
4. Traverse all successors `v` of `u` and do `indegree[v] -= 1`.
5. If a successor's indegree drops to `0`, it means all of its prerequisites have been completed, so it can be enqueued.
6. If the final answer length is smaller than the number of nodes, the graph contains a cycle.

```text
queue = all nodes with indegree 0
order = []

while queue:
  u = queue.pop_front()
  order.append(u)

  for v in graph[u]:
    indegree[v] -= 1
    if indegree[v] == 0:
      queue.push_back(v)

if len(order) != len(nodes):
  there is a cycle
else:
  order is a valid topological ordering
```

The complexity is `O(V + E)`, because each node is enqueued and dequeued once, and each edge is processed only once.

### DFS Formulation: Use Three Colors to Detect Cycles

DFS can also perform topological sorting. The core idea is "append in postorder":

- `0 = unvisited`: not visited yet.
- `1 = visiting`: currently on the DFS path.
- `2 = visited`: this node and all its successors have already been fully processed.

If DFS encounters a `visiting` node, the current path forms a cycle. If all successors of a node have been processed, then add the node to the answer. Finally, reverse the answer to obtain a topological order.

Kahn's is better for explaining "indegree and dependency release"; DFS is better for recursive implementations and cycle detection. Either is fine in an interview, but you must be able to explain the edge direction clearly.

### 17. Course Schedule

The input is `numCourses` courses, numbered `0` to `numCourses - 1`, and a list of prerequisite pairs `prerequisites`, where `[a, b]` means course `a` requires course `b` first, an edge `b -> a`. Deciding whether all courses can be finished is the same question as deciding whether this prerequisite graph is a DAG.

This is exactly what Kahn's algorithm answers: a topological order exists if and only if every node can be placed into `order`. Running Kahn's algorithm with the indegree array and queue introduced above, then comparing the number of processed nodes against `numCourses`, gives the answer directly; if fewer nodes were processed, the rest are stuck in a cycle and their indegree never reaches `0`. Three-color DFS finds the same answer through cycle detection: encountering a node still marked `visiting` (color `1`) on the current path means a cycle exists. The version below follows the same Kahn's-BFS style already used for Foreign Dictionary.

| Item | Detail |
|---|---|
| Technique combination | Cycle detection via Kahn's algorithm |
| Key invariant | The number of processed nodes equals `numCourses` if and only if the graph has no cycle |
| Time / Space | Time `O(V + E)`, space `O(V + E)` |

#### Quick Coding: Course Schedule

```python
def canFinish(numCourses, prerequisites):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import defaultdict, deque
from typing import List


class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graph = defaultdict(list)
        indegree = [0] * numCourses
        for course, pre in prerequisites:
            graph[pre].append(course)
            indegree[course] += 1

        queue = deque(node for node in range(numCourses) if indegree[node] == 0)
        visited_count = 0

        while queue:
            node = queue.popleft()
            visited_count += 1
            for nxt in graph[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        return visited_count == numCourses
```

`graph[pre].append(course)` encodes the prerequisite pair as the edge `pre -> course`, the same relationship `[a, b]` expresses as `b -> a`, just stored directly in edge direction. `visited_count` increases by one each time a node is popped, equivalent to the length of `order` in Kahn's algorithm. This problem only needs a yes/no answer for whether the courses can be finished; the actual order is the subject of Course Schedule II below.

</details>

### 18. Course Schedule II

Same input as Course Schedule, but the answer is a valid course order rather than a boolean; return an empty array if no valid order exists. This asks directly for the `order` that Kahn's algorithm produces, with no additional check needed, since `order` already doubles as both the topological order and the cycle check.

| Item | Detail |
|---|---|
| Technique combination | Kahn's algorithm, returning the topological order directly |
| Key invariant | `order` is a valid topological order when `len(order) == numCourses`; otherwise a cycle exists and the answer is `[]` |
| Time / Space | Time `O(V + E)`, space `O(V + E)` |

#### Quick Coding: Course Schedule II

```python
def findOrder(numCourses, prerequisites):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import defaultdict, deque
from typing import List


class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        indegree = [0] * numCourses
        for course, pre in prerequisites:
            graph[pre].append(course)
            indegree[course] += 1

        queue = deque(node for node in range(numCourses) if indegree[node] == 0)
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for nxt in graph[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        return order if len(order) == numCourses else []
```

`order` here follows the same logic as `order` in Foreign Dictionary, with nodes being course numbers instead of characters: a node with indegree `0` is enqueued, popped into the answer, and its successors are relaxed in turn. The cycle check before returning, `len(order) == numCourses`, mirrors `len(order) != len(indegree)` in Foreign Dictionary. A valid topological order is usually not unique; for `numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]`, both `[0,1,2,3]` and `[0,2,1,3]` are valid, since every prerequisite edge places the prerequisite course before the dependent one.

</details>

### 19. Alien Dictionary (Foreign Dictionary)

The key to Foreign Dictionary / Alien Dictionary is not string processing itself, but extracting partial-order relationships between characters from adjacent words. Take `words = ["hrn", "hrf", "er", "enn", "rfnn"]` as an example:

The essential process of this problem is:

1. Initialize all characters that appear, to avoid missing isolated nodes with no edges.
2. Compare only adjacent words, because when the whole dictionary is sorted, each adjacent pair provides the minimum necessary constraint.
3. Find the first different characters `a` and `b` in adjacent words, add the edge `a -> b`, and then stop comparing that pair.
4. If no differing character is found but the previous word is longer, for example `["abc", "ab"]`, this is invalid input, so return an empty string directly.
5. Perform topological sorting on the character graph; if a cycle is detected, also return an empty string.

The problem gives an ordered dictionary `words` in an alien language. These words are still composed of English letters, but the relative order of the letters is unknown. Return one legal letter order; if no legal order exists, return an empty string. Problem source: <https://neetcode.io/problems/foreign-dictionary/question?list=neetcode150>

```topo-demo
foreign-dictionary
```

| Item | Detail |
|---|---|
| Technique combination | Extracting character partial order from adjacent word pairs, Kahn's algorithm producing the topological order |
| Key invariant | The partial order provided by the first differing character in each adjacent word pair is equivalent to every necessary constraint implied by the whole dictionary |
| Time / Space | Time `O(N + V + E)` (`N` is the total length of all words), space `O(V + E)` |

#### Graph Construction Rule

For any adjacent words `word1` and `word2`:

```text
word1 = h r n
word2 = h r f
             ^
the first different characters are n and f, so n must come before f; add edge n -> f
```

Note that you look only at the first different character. Later characters cannot be used to add more edges, because lexicographic order is determined at the first difference.

#### Python Solution: Kahn BFS

```python
from collections import defaultdict, deque
from typing import List

class Solution:
    def foreignDictionary(self, words: List[str]) -> str:
        graph = {char: set() for word in words for char in word}
        indegree = {char: 0 for char in graph}

        for first, second in zip(words, words[1:]):
            min_len = min(len(first), len(second))

            if len(first) > len(second) and first[:min_len] == second[:min_len]:
                return ""

            for i in range(min_len):
                if first[i] != second[i]:
                    src, dst = first[i], second[i]
                    if dst not in graph[src]:
                        graph[src].add(dst)
                        indegree[dst] += 1
                    break

        queue = deque([char for char, degree in indegree.items() if degree == 0])
        order = []

        while queue:
            char = queue.popleft()
            order.append(char)

            for nxt in graph[char]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(indegree):
            return ""

        return "".join(order)
```

The time complexity is `O(N + V + E)`, where `N` is the total length of all words, `V` is the number of distinct characters, and `E` is the number of character partial-order edges. The space complexity is `O(V + E)`.

#### Why This Code Passes

- `graph = {char: set() ...}` registers every character as a node first, ensuring the answer includes isolated characters.
- `if dst not in graph[src]` prevents duplicate edges from increasing indegree multiple times.
- The prefix-invalid case must be handled before comparing characters; otherwise `["abc", "ab"]` would be incorrectly treated as adding no new constraint.
- `len(order) != len(indegree)` is Kahn's cycle detection. If there is a cycle, nodes in the cycle can never drop to indegree `0`.

### Common Pitfalls

- Reversing the edge direction: if `word1` comes before `word2`, and the first differing characters are `a/b`, the edge should be `a -> b`.
- Comparing characters beyond the first difference: lexicographic order is determined only by the first differing character.
- Not handling the prefix-invalid case: `["abc", "ab"]` must return `""`.
- Using a list to store neighbors without deduplication, causing indegree to be increased repeatedly.
- Forgetting to put all characters into the graph, which causes missing characters in the answer.
- Assuming the answer must be unique; the problem usually allows any valid topological order.
## Module 10: Final Checklist Before an Interview

1. Is this an implicit graph (a grid) or an explicit graph (an adjacency list or edge list)? An implicit graph needs no construction; adjacency comes directly from coordinate arithmetic.
2. Does the problem only care about connectivity/area, or does it need shortest distance/layer count? The former calls for DFS; the latter requires BFS, or Dijkstra on a weighted graph.
3. Is there one starting point or many? Multiple starting points can all be seeded together as BFS layer 0, with no need to run several single-source BFS passes and take the minimum.
4. Are all edge weights non-negative? If so, use Dijkstra. If there are negative weights, or an explicit limit on path length/edge count, use Bellman-Ford instead.
5. Does the problem ask for the number of connected components, cycle detection, or a spanning tree? Union-Find handles the first two well; a spanning tree is a choice between Union-Find-driven Kruskal and heap-driven Prim.
6. Does the problem ask to "visit every node once" or "use every edge once"? The former is ordinary traversal or topological sort; the latter is an Eulerian path, which needs Hierholzer's algorithm.
7. If the problem mentions "precedence, dependency, or lexicographic order," can the relationships be built into a directed graph and solved with topological sort?

One sentence to keep in mind:

> The first step in a graph problem is not choosing an algorithm, it is figuring out what the graph actually looks like: what the nodes are, what the edges are, whether the edges are weighted or directed, and how many starting points there are. Once that is settled, the usable algorithms narrow down to one or two.

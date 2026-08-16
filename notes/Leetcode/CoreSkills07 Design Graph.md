# Graphs：图论

## 模块一：图的表示与建图

图有两种常见的存储方式：邻接表和邻接矩阵。邻接表适合稀疏图，用 `node -> neighbors` 的映射保存每个节点的邻居；邻接矩阵适合点数较小、边查询频繁的场景，用一个 `n x n` 的二维数组保存任意两点间是否有边（或边权）。

- 邻接表空间：`O(V + E)`；邻接矩阵空间：`O(V^2)`。
- 遍历全部点边的时间都是 `O(V + E)`；邻接矩阵下这个上界会退化到 `O(V^2)`，因为扫描每个点的邻居本身要 `O(V)`。

无向图加一条边时需要同时写入 `u -> v` 和 `v -> u`；加权图的邻居列表通常保存 `(neighbor, weight)` 这样的元组。

```text
addEdge(u, v, w=1):
  adj[u].append((v, w))
  if undirected:
    adj[v].append((u, w))

neighbors(u):
  return adj.get(u, [])
```

常见坑：

- 无向边只加了一边，另一个方向的邻居查询会漏掉这条边。
- 遍历时忘记 `visited`，在有环图上陷入死循环。
- 节点编号不连续（比如字符串节点，或编号有跳号）时仍强行用数组下标，应改用字典存储邻接表。

这一章接下来的九个模块，覆盖 NeetCode 150 里 Graphs 和 Advanced Graphs 两个分类的全部 19 道题目，按解题技巧分组：网格上的隐式图遍历、邻接表上的显式图遍历、并查集、最小生成树、最短路、欧拉路径、拓扑排序。

## 学习顺序

| 顺序 | 原题 | 要掌握的内容 |
|---:|---|---|
| 1 | [200. Number of Islands](https://neetcode.io/problems/number-of-islands/question?list=neetcode150) | Matrix DFS/BFS flood fill 统计连通块 |
| 2 | [695. Max Area of Island](https://neetcode.io/problems/max-area-of-island/question?list=neetcode150) | DFS 递归返回值累加面积 |
| 3 | [417. Pacific Atlantic Water Flow](https://neetcode.io/problems/pacific-atlantic-water-flow/question?list=neetcode150) | 双源 DFS/BFS，流动条件反转 |
| 4 | [130. Surrounded Regions](https://neetcode.io/problems/surrounded-regions/question?list=neetcode150) | 先标记边界安全区域，再翻转其余部分 |
| 5 | [994. Rotting Oranges](https://neetcode.io/problems/rotting-oranges/question?list=neetcode150) | 多源分层 BFS |
| 6 | [286. Walls and Gates](https://neetcode.io/problems/islands-and-treasure/question?list=neetcode150) | 多源 BFS 填充到最近源点的距离 |
| 7 | [133. Clone Graph](https://neetcode.io/problems/clone-graph/question?list=neetcode150) | 邻接表 DFS + 哈希表记录克隆映射 |
| 8 | [127. Word Ladder](https://neetcode.io/problems/word-ladder/question?list=neetcode150) | BFS + 按位替换生成邻居 |
| 9 | [323. Number of Connected Components in an Undirected Graph](https://neetcode.io/problems/count-connected-components/question?list=neetcode150) | 并查集统计连通分量 |
| 10 | [261. Graph Valid Tree](https://neetcode.io/problems/valid-tree/question?list=neetcode150) | 边数校验 + 并查集环检测 |
| 11 | [684. Redundant Connection](https://neetcode.io/problems/redundant-connection/question?list=neetcode150) | 按顺序处理边，首个失败的 union 即答案 |
| 12 | [1584. Min Cost to Connect All Points](https://neetcode.io/problems/min-cost-to-connect-points/question?list=neetcode150) | 最小生成树，Kruskal 或 Prim |
| 13 | [743. Network Delay Time](https://neetcode.io/problems/network-delay-time/question?list=neetcode150) | 标准 Dijkstra |
| 14 | [778. Swim in Rising Water](https://neetcode.io/problems/swim-in-rising-water/question?list=neetcode150) | 最小堆 + minimax path |
| 15 | [787. Cheapest Flights Within K Stops](https://neetcode.io/problems/cheapest-flight-path/question?list=neetcode150) | 带边数限制的 Bellman-Ford |
| 16 | [332. Reconstruct Itinerary](https://neetcode.io/problems/reconstruct-itinerary/question?list=neetcode150) | Hierholzer 算法求欧拉路径 |
| 17 | [207. Course Schedule](https://neetcode.io/problems/course-schedule/question?list=neetcode150) | Kahn 算法判环 |
| 18 | [210. Course Schedule II](https://neetcode.io/problems/course-schedule-ii/question?list=neetcode150) | Kahn 算法输出拓扑序 |
| 19 | [269. Alien Dictionary](https://neetcode.io/problems/foreign-dictionary/question?list=neetcode150) | 从相邻单词对抽取字符偏序 |

## 模块二：网格即图，Matrix DFS/BFS 模板

### 1. 网格与图的对应关系

二维网格是一种特殊图：每个格子 `(r, c)` 是一个节点，它的上下左右四个格子是它的邻居，边由行列坐标的加减运算隐式给出，不需要显式建立邻接表。网格上的连通块、最短路径、区域填充问题，本质上都是图论里连通分量、无权图最短路、可达性问题在这种隐式邻接结构上的实例。

DFS 和 BFS 都能正确遍历网格，选择依据是题目要的是什么量。只关心"能不能到达"或"这一片区域有多大"（连通性、面积、区域标记）时，DFS 更直接，一路递归到底再回溯，不需要维护额外的层级信息。题目要"最少几步/最少几分钟能到达"（最短距离、多源同时扩散）时，BFS 是唯一天然正确的选择：BFS 逐层扩展的顺序本身就是无权图上的最短路顺序，DFS 找到的第一条路径不保证是最短的。

### 2. DFS Flood Fill 模板

DFS 版本的核心是一个只处理单个格子的函数：先做越界和终止条件检查，再标记当前格子已访问，最后递归四个方向。

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

外层再套一层双重循环遍历所有格子，遇到未访问的目标格子就启动一次 DFS，每启动一次对应一个新的连通块。标记访问状态可以用独立的 `visited` 集合，也可以直接把 `grid` 里的目标格子原地改写成非目标值，省掉额外空间，但后一种写法会让"障碍格子"和"已访问格子"共用同一个标记，一旦题目后续还需要区分这两者（比如要在同一份网格上做两轮独立的遍历），原地改写就不再适用。

### 3. BFS 分层扩散模板

BFS 版本用队列维护当前边界，用 `len(queue)` 的快照把遍历切成一层一层，每一层对应距离或时间加一。

```python
from collections import deque

queue = deque(start_cells)          # 单源就放一个起点，多源就把所有起点一次性放入
visited = set(start_cells)          # 入队时立刻标记，不是出队时再标记
dist = 0
while queue:
    for _ in range(len(queue)):     # 这一层剩余的格子数量，遍历前先固定住
        r, c = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] != blocked_value:
                visited.add((nr, nc))
                queue.append((nr, nc))
    dist += 1
```

多源 BFS 只是把初始化步骤从"一个起点入队"换成"把所有起点一次性入队"，主循环逻辑完全不变。BFS 的分层结构不关心某一层的格子来自哪个源，只要它们是在同一轮扩展中被发现的，就属于同一层。

### 4. 常见坑

- BFS 里在出队时才标记 `visited`，会让同一个格子在它还排队等待处理的时候被其他路径重复发现、重复入队，既浪费时间也可能把层数算错。标记必须发生在入队的那一刻。
- 边界检查和访问/类型检查的顺序颠倒：先访问 `grid[r][c]` 再判断坐标是否越界，会直接触发数组越界。越界检查必须最先执行并且短路返回。
- 把"障碍/水"这类本身就不可通行的格子和"已访问"的格子混用同一个标记，会在需要多轮独立遍历同一份网格时（比如先后判断两片区域的可达性）互相污染彼此的结果。
- 层数/距离统计的位置错误：`dist` 应该在处理完整一层之后才加一，如果在遍历这一层格子的过程中提前累加，会把距离多算或少算一。

### 5. 演示：多源 BFS 处理 Rotting Oranges

以 `grid = [[2,1,1],[1,1,0],[0,1,1]]` 为例（`2` 是腐烂橘子，`1` 是新鲜橘子，`0` 是空格子），所有腐烂橘子同时作为第 0 层的起点入队。第 1 分钟，`(0,0)` 的两个新鲜邻居 `(0,1)`、`(1,0)` 一起变腐烂；第 2 分钟，它们各自的新鲜邻居 `(0,2)`、`(1,1)` 变腐烂；第 3 分钟轮到 `(2,1)`，第 4 分钟轮到 `(2,2)`。全部新鲜橘子腐烂完毕耗时 4 分钟，这也是这道题的标准答案。

```grid-multi-source-bfs-demo
```

## 模块三：6 道题目的映射

### 1. Number of Islands

[NeetCode 题目链接](https://neetcode.io/problems/number-of-islands/question?list=neetcode150)

外层遍历整张网格，每遇到一个未访问的陆地格子（`'1'`），说明发现了一个新的连通块，计数加一，同时启动一次 DFS（或 BFS）把这个连通块里所有陆地格子标记为已访问，避免它们在外层循环里被重复计数。

| 项目 | 内容 |
|---|---|
| 组合技巧 | DFS/BFS flood fill 统计连通块个数 |
| 关键不变量 | 外层循环每次触发新的一次遍历时，当前格子所在的连通块此前一定没有被计数过 |
| 时间 / 空间 | `O(mn) / O(mn)` |

#### Quick Coding：Number of Islands

```python
def numIslands(grid):
    ...
```

<details>
<summary>参考答案</summary>

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

`grid = [["1","1","0","0","0"],["1","1","0","0","0"],["0","0","1","0","0"],["0","0","0","1","1"]]` 上有三个连通块：左上角的 2x2 陆地、中间单独的一格陆地、右下角相邻的两格陆地，函数返回 3。`visited` 集合和 flood fill 保证同一个连通块只会在第一次遇到它时触发计数。

</details>

### 2. Max Area of Island

[NeetCode 题目链接](https://neetcode.io/problems/max-area-of-island/question?list=neetcode150)

模板和 Number of Islands 完全一致，唯一的区别是 DFS 不再只做标记，还要返回"以当前格子为起点、递归覆盖到的格子总数"，外层遍历时取所有连通块面积的最大值。

| 项目 | 内容 |
|---|---|
| 组合技巧 | DFS flood fill，递归返回值累加连通块面积 |
| 关键不变量 | `dfs(r, c)` 的返回值恰好等于当前格子所在连通块中、尚未被访问部分的格子数 |
| 时间 / 空间 | `O(mn) / O(mn)` |

#### Quick Coding：Max Area of Island

```python
def maxAreaOfIsland(grid):
    ...
```

<details>
<summary>参考答案</summary>

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

标准示例网格（8 行 13 列）上最大的连通块有 6 个格子，函数返回 6。递归让每个格子只贡献一次面积，`area` 的累加顺序不影响结果，加法满足交换律。

</details>

### 3. Pacific Atlantic Water Flow

[NeetCode 题目链接](https://neetcode.io/problems/pacific-atlantic-water-flow/question?list=neetcode150)

水从高处或平地流向不高于自己的邻居，正向判断"某个格子能否流到太平洋"需要从这个格子出发试探所有下降路径，每个格子都试一遍是 `O((mn)^2)`。从太平洋和大西洋各自的边界出发，按照"邻居高度大于等于当前高度"这个反转后的条件做 DFS/BFS，能找出所有能够顺流而下到达该边界的格子：水能从 A 流到 B，当且仅当能够从 B 沿着高度不下降的方向反向走到 A。分别求出从两条边界出发的可达集合，取交集就是答案。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 双源 DFS/BFS，从边界反向出发，流动条件反转为"邻居高度 >= 当前高度" |
| 关键不变量 | 反向可达集合中的每个格子，都存在一条真实水流方向上高度不上升的路径通向对应边界 |
| 时间 / 空间 | `O(mn) / O(mn)` |

#### Quick Coding：Pacific Atlantic Water Flow

```python
def pacificAtlantic(heights):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        rows, cols = len(heights), len(heights[0])
        pacific, atlantic = set(), set()

        def dfs(r, c, visited, prev_height):
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

        for c in range(cols):
            dfs(0, c, pacific, heights[0][c])
            dfs(rows - 1, c, atlantic, heights[rows - 1][c])
        for r in range(rows):
            dfs(r, 0, pacific, heights[r][0])
            dfs(r, cols - 1, atlantic, heights[r][cols - 1])

        return [
            [r, c]
            for r in range(rows)
            for c in range(cols)
            if (r, c) in pacific and (r, c) in atlantic
        ]
```

标准示例的 5x5 高度矩阵上，两个海洋都能到达的格子共有 7 个：`[0,4] [1,3] [1,4] [2,2] [3,0] [3,1] [4,0]`。DFS 的终止条件里用 `heights[r][c] < prev_height` 而不是 `<=`，允许在平地之间继续扩展，水在平地上也能流动。

</details>

### 4. Surrounded Regions

[NeetCode 题目链接](https://neetcode.io/problems/surrounded-regions/question?list=neetcode150)

直接判断每个 `'O'` 是否被 `'X'` 包围，需要对每个 `'O'` 反复检查是否存在到边界的路径，容易重复计算。等价的判断条件是：只有连接到边界的 `'O'` 不会被翻转，其余 `'O'` 一定被完全围住。因此先从四条边界出发，对边界上的每个 `'O'` 做 DFS/BFS，把所有能从边界到达的 `'O'` 标记成一个临时符号（比如 `'#'`）；遍历结束后，剩下还是 `'O'` 的格子就是真正被包围的，翻转成 `'X'`，再把临时符号还原成 `'O'`。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 先标记边界可达的安全区域，再翻转其余部分 |
| 关键不变量 | 两轮遍历结束后，仍保留原始标记的 `'O'` 一定不存在到边界的路径 |
| 时间 / 空间 | `O(mn) / O(mn)` |

#### Quick Coding：Surrounded Regions

```python
def solve(board):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        rows, cols = len(board), len(board[0])

        def dfs(r, c):
            if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != "O":
                return
            board[r][c] = "#"
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                dfs(r + dr, c + dc)

        for r in range(rows):
            dfs(r, 0)
            dfs(r, cols - 1)
        for c in range(cols):
            dfs(0, c)
            dfs(rows - 1, c)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == "#":
                    board[r][c] = "O"
```

`board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]` 里，`(3,1)` 的 `'O'` 本身就在最后一行的边界上，DFS 会把它连同它能到达的邻居一起标记为 `'#'`，最终还原为 `'O'`；中间那两个 `'O'` 没有任何路径通向边界，被翻转成 `'X'`。结果是 `[["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]`，只有边界行的那个 `'O'` 保留。

</details>

### 5. Rotting Oranges

[NeetCode 题目链接](https://neetcode.io/problems/rotting-oranges/question?list=neetcode150)

所有腐烂橘子（值为 `2`）作为多源 BFS 的第 0 层同时入队，统计初始新鲜橘子（值为 `1`）的数量。每一层出队处理完所有当前队列里的格子，把它们四周新鲜的邻居变腐烂并入队，同时新鲜橘子计数减一，处理完一整层后分钟数加一。队列为空但仍有新鲜橘子未变腐烂，说明存在无法被感染的橘子，返回 `-1`。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 多源分层 BFS |
| 关键不变量 | 第 `t` 层出队的格子，恰好是第 `t` 分钟新变腐烂的橘子，`len(queue)` 快照保证同一分钟内产生的格子不会被提前处理 |
| 时间 / 空间 | `O(mn) / O(mn)` |

#### Quick Coding：Rotting Oranges

```python
def orangesRotting(grid):
    ...
```

<details>
<summary>参考答案</summary>

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

`grid = [[2,1,1],[1,1,0],[0,1,1]]` 上，`fresh` 初始为 5。第 1 分钟腐烂 `(0,1)`、`(1,0)`；第 2 分钟腐烂 `(0,2)`、`(1,1)`；第 3 分钟腐烂 `(2,1)`；第 4 分钟腐烂 `(2,2)`，`fresh` 归零，循环结束，返回 `4`，和模块二演示的动画过程完全对应。

</details>

### 6. Walls and Gates

[NeetCode 题目链接](https://neetcode.io/problems/islands-and-treasure/question?list=neetcode150)

经典 LeetCode 题目名是 Walls and Gates，NeetCode 上重命名为 Islands and Treasure，题意不变：把每个空房间（`INF`）填成到最近一个门（`0`）的最短距离，墙（`-1`）不可通行。所有门作为多源 BFS 的起点一次性入队，按层扩展，第一次到达某个空房间时所处的层数就是它到最近门的最短距离，BFS 在无权图上第一次到达即最短。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 多源 BFS 填充到最近源点的距离 |
| 关键不变量 | 空房间第一次被访问的那一层，就是它到所有门中最近一个的最短距离 |
| 时间 / 空间 | `O(mn) / O(mn)` |

#### Quick Coding：Walls and Gates

```python
def islandsAndTreasure(grid):
    ...
```

<details>
<summary>参考答案</summary>

```python
from collections import deque
from typing import List

INF = 2147483647


class Solution:
    def islandsAndTreasure(self, grid: List[List[int]]) -> None:
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        visited = set()

        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 0:
                    queue.append((r, c))
                    visited.add((r, c))

        dist = 0
        while queue:
            dist += 1
            for _ in range(len(queue)):
                r, c = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and (nr, nc) not in visited
                        and grid[nr][nc] == INF
                    ):
                        grid[nr][nc] = dist
                        visited.add((nr, nc))
                        queue.append((nr, nc))
```

标准示例的 4x4 网格上，两个门分别在 `(0,2)` 和 `(3,0)`，BFS 从这两个门同时展开，填充结果是 `[[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]`，每个空房间的最终值都对应它到最近门的真实最短距离。`visited` 在入队时就标记，避免同一个房间被两个门的 BFS 波前重复计算成两个不同的距离。

</details>

网格题目本质上都是图论题目，只是邻接表是隐式的：不需要提前建图，每次访问一个格子时用行列坐标的加减法现算它的邻居。做这类题时真正要做的判断只有两个：这道题要连通性/面积还是要最短距离/层数（决定用 DFS 还是 BFS），起点是一个还是很多个（决定单源还是多源）。想清楚这两点之后，四个模板（DFS 单源、DFS 多源、BFS 单源、BFS 多源）里对应的那一个基本就能直接套用。
## 模块四：邻接表上的通用遍历

前面处理的是网格上的隐式图：节点是 `(row, col)`，边固定是上下左右四个方向，不需要额外的邻接结构。这一节处理题目直接给出邻接表(或者需要先自己建出邻接表)的显式图：节点可以是整数、字符串，甚至是 Clone Graph 里那样的自定义对象，边由 `adj[node]` 这样的映射给出，不再有方向这个隐含结构。DFS/BFS 的骨架不变，变的是两处：`visited` 不再是二维数组，而是以节点本身(或节点的某种可哈希标识)为键的集合；取邻居不再是加减坐标，而是查 `adj[node]`。

### 邻接表上的 DFS / BFS 模板

DFS 可以写成递归，也可以用显式栈，两种写法都需要在进入一个节点时立刻把它加进 `visited`，避免图里的环导致无限递归或死循环。

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

BFS 用队列，同样在入队时立刻标记 `visited`，而不是等到出队才标记，否则同一个节点可能因为被多个邻居同时发现而重复入队。

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

### 遍历之外的额外状态

单纯的 `visited` 集合只能回答"这个节点有没有来过"。有些题目还需要在遍历过程中携带更多信息：

- Clone Graph 需要知道每个原节点对应的克隆节点是谁，所以用一个哈希表 `old_to_new` 代替单纯的 `visited` 集合：一个节点在表里出现过，就说明它已经被克隆过，直接返回已有的克隆节点，同一个判断同时完成去重和处理环。
- Word Ladder 需要知道当前单词是从 `beginWord` 走了几步过来的，所以 BFS 队列里存的不是单个单词，而是 `(word, 距离)` 这样的二元组，距离随节点一起入队、出队，不需要按层单独扫描一遍队列。

下面两道题分别对应这两种额外状态。

### 7. Clone Graph

对每个节点做 DFS，用哈希表 `old_to_new` 把原节点映射到它的克隆节点。递归函数一开始先查这个表：查到了，说明这个节点已经克隆过，直接返回克隆节点，不再往下递归；查不到，才新建一个克隆节点、登记进表，再递归克隆它的每个邻居，把克隆邻居接到当前克隆节点的 `neighbors` 上。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 邻接表 DFS + 哈希表记录原节点到克隆节点的映射 |
| 关键不变量 | `old_to_new` 中每个原节点最多对应一个克隆节点，这个映射本身兼任 visited 检查 |
| 时间 / 空间 | `O(V + E) / O(V)` |

#### Quick Coding：Clone Graph

```python
class Solution:
    def cloneGraph(self, node):
        ...
```

<details>
<summary>参考答案</summary>

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

`old_to_new[n] = clone` 必须在递归克隆邻居之前完成登记。如果放在递归之后，图里的环会导致同一个节点被反复克隆，`dfs` 也不会终止。

</details>

### 8. Word Ladder

BFS 求最短变换序列，队列里的每一项是 `(word, dist)`。给定单词的邻居不是拿它跟单词表里每一个词逐个比较，而是枚举它的每一个字符位置，把这一位换成 26 个字母中的每一个，拼出新词后检查是否在单词集合里。距离从 1 开始(`beginWord` 自身算第一步)，每往外扩一层加一。一个单词被访问后直接从集合里删除，代替单独维护一个 `visited` 集合，删除和查重共用同一个操作。

| 项目 | 内容 |
|---|---|
| 组合技巧 | BFS + 按位替换生成邻居 + 访问后从单词集合中移除 |
| 关键不变量 | 队首单词的 `dist` 就是从 `beginWord` 到它的最短步数(BFS 逐层扩展保证) |
| 时间 / 空间 | `O(26 · N · L²) / O(N · L)`，`N` 为单词数，`L` 为单词长度 |

#### Quick Coding：Word Ladder

```python
class Solution:
    def ladderLength(self, beginWord, endWord, wordList):
        ...
```

<details>
<summary>参考答案</summary>

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

按位替换生成邻居是每个单词 `O(L · 26)` 次候选，每次候选构造新字符串并做哈希集合查找是 `O(L)`；直接和单词表中每个词比较则是 `O(N · L)`，当 `N` 远大于 26 时按位替换明显更快。`endWord` 不在给定的单词表里时直接返回 `0`，不需要进入 BFS。

</details>

## 模块五：并查集 Union-Find

并查集(Disjoint Set Union，也叫 Union-Find)维护一组不相交的集合，支持两个操作：`find(x)` 返回 `x` 所在集合的代表元(根)，`union(a, b)` 把 `a`、`b` 所在的两个集合合并成一个。放到图问题里，"集合"对应一个连通分量：对每条边执行一次 `union`，最终共享同一个根的节点就在同一个连通分量里，不需要显式建邻接表，也不需要遍历。

### 并查集的核心设计

用一个数组 `parent` 表示所有集合，`parent[x]` 存 `x` 的父节点，根节点的 `parent` 指向自己。初始化时每个节点自成一个集合：`parent[x] = x`，`size[x] = 1`。

`find(x)` 沿着 `parent` 链条往上走，直到找到满足 `parent[r] == r` 的根 `r`。不做优化时，链条可能退化成长度为 `n` 的一条链，`find` 变成 `O(n)`。路径压缩把查找路径上的每个节点都重新接到根下面，之后再 `find` 这些节点只需要一跳。下面用的是迭代版本，先走到根，再回头把路径上的每个节点直接接到根：

```python
def find(self, x):
    root = x
    while root != self.parent[root]:
        root = self.parent[root]
    while x != root:
        self.parent[x], x = root, self.parent[x]
    return root
```

递归版本更短(`parent[x] = find(parent[x]); return parent[x]`)，效果等价，但每次 `find` 都会占用一层 Python 调用栈；节点数达到几万时有触发递归深度限制的风险，本节三道题统一使用上面的迭代版本。

`union(a, b)` 先分别 `find` 出 `a`、`b` 的根 `ra`、`rb`。如果两者相等，说明 `a`、`b` 已经在同一集合里，不能再合并，也不能再把连通分量计数减一。否则把 `size` 较小的树挂到 `size` 较大的树的根下面，同时把新根的 `size` 累加：

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

按 rank(树高的估计值)合并是另一种等价策略，两者的共同点都是让小树挂到大树下面，避免树越合并越深。单独使用路径压缩，或单独使用按大小/按秩合并，单次操作最坏是 `O(log n)`；两者一起使用时，`n` 个节点上做任意多次操作，均摊到每次操作是 `O(α(n))`，`α` 是反阿克曼函数，对任何现实规模的 `n` 都不超过 4，可以当作近似 `O(1)`。

常见的坑：

- 忘记路径压缩，或者忘记按大小/按秩合并：并查集退化成链表，`find` 变成 `O(n)`。
- 统计连通分量数时，`union` 之前不检查两端是否已经连通就直接把计数减一：同一对已连通的节点被重复合并，会把连通分量数多减，答案偏小。
- 节点编号是 1-indexed 还是 0-indexed 没对齐：Redundant Connection 等题目的节点从 1 开始编号，`parent` 数组要开到 `n + 1` 并且从下标 1 开始使用；直接套用 0-indexed 的初始化会让下标 0 变成一个从未出现过的多余节点，或者在节点 `n` 上越界。

下面的演示对 8 个节点(编号 0 到 7)执行一段 `union` 序列。前 6 次 `union` 故意不按大小合并，只是把一个根直接挂到另一个根下面，制造出两条长度不同的链条；第 7 步对链尾节点调用一次 `find`，展示路径压缩如何把这条链条上的每个节点直接拉平到根；最后一步 `union` 按大小合并两条链所在的集合，展示按大小合并具体保留了哪一个根。

```union-find-demo
```

### 9. Number of Connected Components in an Undirected Graph

对每条边执行一次 `union`。用一个随 `union` 成功而递减的 `count`(初始为 `n`)记录当前连通分量数。等价的写法是遍历完所有边之后，对每个节点调用一次 `find`，用一个集合收集出现过的不同的根，集合大小就是连通分量数；用递减的 `count` 省掉这次额外遍历。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 并查集，遍历所有边执行 `union` |
| 关键不变量 | `count` 恒等于当前连通分量数：初始为 `n`，每次 `union` 成功就减一 |
| 时间 / 空间 | `O((V + E) · α(V)) / O(V)` |

#### Quick Coding：Number of Connected Components in an Undirected Graph

```python
class Solution:
    def countComponents(self, n, edges):
        ...
```

<details>
<summary>参考答案</summary>

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

`uf.count` 只在 `union` 真正合并了两个不同集合时才减一，重复处理同一个连通分量内部的边不会影响它，边的处理顺序也就不影响最终结果。

</details>

### 10. Graph Valid Tree

一个图是合法的树，当且仅当边数恰好是 `n - 1` 且整个图连通(边数已经固定为 `n - 1` 时，无环和连通是等价的：多一条边必然出现环，少一条边必然不连通)。先检查边数，边数不对直接返回 `False`。边数正确时用并查集处理每条边：只要某次 `union` 失败(两端点已经在同一集合)，就说明存在环，直接返回 `False`；处理完所有边后集合数应该恰好是 `1`。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 边数校验(`== n - 1`) + 并查集环检测 |
| 关键不变量 | 任意一次 `union` 失败即存在环；边数已经等于 `n - 1` 时，无环等价于全连通 |
| 时间 / 空间 | `O(V · α(V)) / O(V)` |

#### Quick Coding：Graph Valid Tree

```python
class Solution:
    def validTree(self, n, edges):
        ...
```

<details>
<summary>参考答案</summary>

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

边数检查是一次快速失败的剪枝，不是逻辑上必需的独立步骤：如果没有它，"没有任何一次 `union` 失败"加上"最终 `count == 1`" 这两个条件本身就能推出边数必然等于 `n - 1`，因为每次成功的 `union` 恰好让 `count` 减一，`count` 从 `n` 降到 `1` 必须经过恰好 `n - 1` 次成功合并。提前检查边数能在边数明显不对时立刻返回，不需要真的跑一遍并查集。

</details>

### 11. Redundant Connection

按输入顺序处理每条边，对每条边执行 `union`。题目保证去掉某一条边之后原图是一棵树，第一条使 `union` 失败的边，即它的两个端点已经在同一集合里的边，就是那条把树变成带环图的多余边。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 按输入顺序处理边的并查集，首个失败的 `union` 即答案 |
| 关键不变量 | 处理到第 `i` 条边为止，并查集始终反映前 `i` 条边构成的图的连通状态 |
| 时间 / 空间 | `O(E · α(V)) / O(V)` |

#### Quick Coding：Redundant Connection

```python
class Solution:
    def findRedundantConnection(self, edges):
        ...
```

<details>
<summary>参考答案</summary>

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
        uf = UnionFind(n + 1)  # 节点从 1 开始编号，数组开到 n + 1，下标 0 不用

        for a, b in edges:
            if not uf.union(a, b):
                return [a, b]

        return []
```

节点从 1 编号到 `n`，`UnionFind` 初始化成 `n + 1` 大小、下标 0 空置不用，避免把节点 1 错误地对齐到数组下标 0。题目保证恰好存在一条冗余边，`union` 一定会在某条边上失败，最后的空列表分支不会被实际用到。

</details>
## 模块六：最小生成树，Prim 与 Kruskal

给定一个带权无向连通图，最小生成树（Minimum Spanning Tree，MST）是一棵包含图中全部 `n` 个节点、总边权最小的树。它恰好有 `n-1` 条边，不含环，是"用最少总代价把所有节点连通"这一类问题的标准模型。求 MST 主要有两种算法：Prim 和 Kruskal，两者都基于贪心加边的思路，具体的加边顺序和判断方式不同。

### Prim 算法

Prim 从任意一个起点出发，维护一棵不断扩张的连通树。算法用一个最小堆保存"树内节点到树外节点"的候选边，每次弹出权重最小的候选边：如果这条边连到一个还没加入树的节点，就把这条边和节点都加入树，并把新节点发出的边继续推入堆；如果候选边连到的节点已经在树里（堆中残留的旧候选边），直接丢弃。

```text
visited = {start}
heap = start 发出的所有边
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

每条边最多被推入堆一次（由某个已访问节点发出），堆的插入和弹出是 `O(log E)`，整体时间复杂度 `O(E log E)`。

### Kruskal 算法

Kruskal 先把图里全部的边按权重从小到大排序，再依次处理每条边。判断一条边是否可以加入用并查集（Union-Find）：如果边的两个端点当前不在同一个连通分量里，加入这条边不会成环，就执行 union 并把边权计入总代价；如果两个端点已经连通，加入这条边只会形成环，直接跳过。加入 `n-1` 条边后所有节点已经连通，算法结束。

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

排序是 `O(E log E)`，并查集操作（带路径压缩和按秩合并）近似 `O(1)`，总复杂度由排序主导，同样是 `O(E log E)`。

### 两种算法的对比

| 维度 | Prim | Kruskal |
|---|---|---|
| 扩张方式 | 从一个种子节点出发，始终维护单棵树 | 全局按边权排序，用并查集合并多个分量 |
| 数据结构 | 邻接表 + 候选边最小堆 | 排序后的边列表 + 并查集 |
| 适合的输入形式 | 图以邻接表给出，尤其稠密图（可以边扩张边现算距离，不需要先列出全部边） | 图本身就以扁平边列表给出 |
| 处理不连通图 | 无法直接判断，`visited` 收尾时小于 `n` 说明不连通 | 天然支持：加入的边数收尾时小于 `n-1`，得到的是最小生成森林 |

### 常见坑

- Prim 忘记检查 `visited` 就直接把弹出的候选边计入代价，导致同一个节点被重复加入，生成树里出现环，总代价也被算错。
- Kruskal 没有用并查集的返回值判断两个端点是否已经连通，遇到会成环的边没有跳过。
- 默认图一定连通，直接假设一定能找到覆盖全部节点的生成树；图不连通时 Prim 会在 `visited` 小于 `n` 时提前耗尽堆，Kruskal 只能得到边数小于 `n-1` 的生成森林，两种情况都需要显式处理（通常是返回 `-1` 或森林本身）。

### 12. Min Cost to Connect All Points

题目给出平面上 `n` 个点，任意两点间都可以连边，边权是两点间的曼哈顿距离。这是一个隐式的完全图：显式列出全部边有 `n*(n-1)/2` 条。

Kruskal 的做法是先把全部 `n*(n-1)/2` 条边构造出来并排序，再用并查集加边，是模块开头模板的直接套用。Prim 的做法不需要预先构造边列表：从任意一点（比如下标 `0`）出发，每次把"当前树到所有未访问点"的距离现算出来推入堆，堆顶给出下一条要加入的最小边。

两种做法在这道题上最坏情况都会产生 `O(n^2)` 规模的边或堆操作，Kruskal 是 `O(n^2 log n)`（排序主导），基于堆的 Prim 同样是 `O(n^2 log n)`，但 Prim 不需要一次性构造并排序全部 `n*(n-1)/2` 条边，只在扩张当前树时现算距离、增量推入堆。`n` 较大时，Prim 省掉了整体排序和预先物化边表的开销，是这道题更常见的推荐写法。

| 项目 | 内容 |
|---|---|
| 组合技巧 | Kruskal（构造全部边 + 排序 + 并查集）或 Prim（最小堆，边扩张边现算曼哈顿距离） |
| 关键不变量 | Kruskal：并查集里同一分量内的点已经连通；Prim：`visited` 之外的候选距离始终是"当前树到该点的已知最短距离" |
| 时间 / 空间 | 两者均为 `O(n^2 log n) / O(n^2)`；Prim 省去显式排序和边表物化 |

#### Quick Coding：Min Cost to Connect All Points

```python
def minCostConnectPoints(points):
    ...
```

<details>
<summary>参考答案</summary>

Kruskal 版本：

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

Prim 版本：

```python
import heapq
from typing import List


class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        n = len(points)
        visited = [False] * n
        min_heap = [(0, 0)]  # (dist, point_index)，从点 0 出发
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

两个版本在示例 `points = [[0,0],[2,2],[3,10],[5,2],[7,0]]` 上都返回 `20`，`points = [[3,12],[-2,5],[-4,1]]` 上都返回 `18`，已用脚本验证一致。Kruskal 需要把全部候选边先构造并排序好，是套用模块开头模板最直接的写法；Prim 每次只对"当前树"和"剩余未访问点"现算距离，不需要整体排序，堆里最多同时存在 `O(n)` 个未访问点的候选距离（弹出后若已访问会直接跳过），面对更大的 `n` 时更常作为首选实现。

</details>

## 模块七：最短路，Dijkstra 与 Bellman-Ford

最短路题不要先背模板，而是先判断约束：

- 边权都非负，并且没有额外限制：优先考虑 Dijkstra。
- 有负权边，或者路径长度/边数有显式限制：考虑 Bellman-Ford。
- 图是无权图：BFS 就是最短路。
- 状态里除了节点还有别的维度，例如"用了几次中转"：要把状态扩展成 `(node, state)`。

### Dijkstra 回顾

Dijkstra 适用于非负权图。核心是用最小堆每次弹出当前代价最小的状态，再松弛邻居。

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

对于 Cheapest Flights，普通 Dijkstra 不够，因为"到同一个城市的最低价格"不一定是最终最好状态。一个更贵但用了更少航班的状态，后面可能仍然可行。因此 Dijkstra 写法需要把状态扩展为 `(cost, city, stopsUsed)`。

### 13. Network Delay Time

题目给出有向带权图 `times`，每条边 `[u, v, w]` 表示信号从节点 `u` 传到节点 `v` 需要 `w` 时间，边权非负。信号从节点 `k` 发出，求所有 `n` 个节点都收到信号所需的最短时间；如果有节点始终收不到信号，返回 `-1`。这是上面 Dijkstra 回顾缺的那道例题：单一起点，每个节点只关心一个最短距离，状态里不需要像 Cheapest Flights 那样额外记录用了几条边。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 标准 Dijkstra，单源最短路 |
| 关键不变量 | 节点第一次从堆中弹出时，对应的 `d` 就是它的最终最短距离；堆里同一节点的其余副本都是过期候选 |
| 时间 / 空间 | 时间 `O(E log V)`，空间 `O(V + E)` |

#### Quick Coding：Network Delay Time

```python
def networkDelayTime(times, n, k):
    ...
```

<details>
<summary>参考答案</summary>

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

`dist` 数组和 `heap` 的初始化、松弛逻辑，和 Dijkstra 回顾里的伪代码完全一致：`if d != dist[u]: continue` 跳过堆里的过期副本，松弛只在 `d + w < dist[v]` 时发生。全部松弛完成后，`dist[1:n+1]` 里的最大值就是所有节点收到信号所需的时间，因为这个时间取决于最慢被通知到的那个节点；如果最大值仍是 `INF`，说明存在从 `k` 到不了的节点，返回 `-1`。

</details>

### 14. Swim in Rising Water

题目给出 `n x n` 的高程网格 `grid`。水从左上角开始随时间线性上涨，进入某个格子的前提是当前水位不低于这个格子的高程；一条路径能通过的时间等于路径上出现过的最大高程。目标是求这个路径瓶颈在所有从左上角到右下角的路径里的最小值，这是一个 minimax path 问题。

这可以按 Dijkstra/Prim 的思路改写成同一套堆驱动的松弛流程：堆里存 `(bottleneck, r, c)`，`bottleneck` 表示从起点到 `(r, c)` 这条已知路径上出现过的最大高程；每次弹出 `bottleneck` 最小的状态，松弛邻居时把它的瓶颈更新为 `max(bottleneck, grid[neighbor])`。第一次把右下角弹出堆时，对应的 `bottleneck` 就是答案。另一种等价思路是对水位 `t` 做二分，用 BFS/DFS 检查只经过高程 `<= t` 的格子能否从左上角连到右下角；这一节以堆的写法为主。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 最小堆 + minimax path，Dijkstra/Prim 的变体 |
| 关键不变量 | 堆顶维护的是当前已知路径里瓶颈最小的状态，松弛时用 `max` 更新邻居的瓶颈 |
| 时间 / 空间 | 时间 `O(n^2 log n)`，空间 `O(n^2)` |

#### Quick Coding：Swim in Rising Water

```python
def swimInWater(grid):
    ...
```

<details>
<summary>参考答案</summary>

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

`visited` 起的作用和 Dijkstra 里的 `dist` 数组类似：一个格子只入堆一次，因为堆保证它第一次被弹出时，对应的 `bottleneck` 已经是能到达它的最小路径瓶颈。松弛时把邻居的新状态设为 `max(bottleneck, grid[nr][nc])`，这个量对应路径上出现过的最大高程。第一次弹出终点格子时直接返回，不必等堆清空。

</details>

### 为什么 Bellman-Ford 更自然

Bellman-Ford 的语义刚好适合下面这道题：

> 第 `i` 轮松弛之后，`prices[x]` 表示使用最多 `i` 条边到达城市 `x` 的最低价格。

题目允许最多 `k` 个 stops，也就是最多 `k + 1` 条 flight。因此我们只需要做 `k + 1` 轮松弛。

关键点是：每一轮必须从上一轮的 `prices` 读，写入 `nextPrices`。如果直接原地更新，就会在同一轮里把多条边串起来，等价于偷偷使用了超过当前轮数的航班数。

### 15. Cheapest Flights Within K Stops

输入：

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

最多 `1` 个中转，所以最多可以坐 `2` 段航班。合法答案是 `0 -> 1 -> 3`，价格是 `700`。路径 `0 -> 1 -> 2 -> 3` 价格更低吗？它是 `400`，但需要 `3` 段航班，超过限制，不能用。

```bellman-demo
cheapest-flights
```

#### 优化版 Bellman-Ford

基础写法是做固定的 `k + 1` 轮。优化点有两个：

1. 每轮用 `nextPrices = prices.copy()`，保证本轮只从上一轮状态转移。
2. 如果某一轮没有任何更新，说明继续松弛也不会变好，可以提前停止。

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

#### 正确性直觉

第 `0` 轮开始时，只有 `src` 的价格是 `0`，表示"不坐任何航班只能到达起点"。

第 `1` 轮只允许从第 `0` 轮的结果出发，所以只能得到所有一段航班能到达的城市。

第 `2` 轮只允许从第 `1` 轮的结果出发，所以得到最多两段航班能到达的城市。

一直做到第 `k + 1` 轮，就正好覆盖了题目允许的最大航班数。因为每轮读旧数组、写新数组，同一轮内不会发生 `0 -> 1 -> 2 -> 3` 这种连续串联。

#### 复杂度

- 时间：`O((k + 1) * E)`，其中 `E` 是航班数量。
- 空间：`O(V)`，只保留 `prices` 和 `next_prices`。

这通常比把所有 `(city, stops)` 状态丢进堆里更直接，也更适合解释"最多几条边"的约束。

#### 常见坑

- 把 `k stops` 当成最多 `k` 条边。实际是最多 `k + 1` 条 flight。
- 原地更新 `prices`，导致同一轮串联多条边。
- 忘记跳过 `prices[start] == INF` 的航班。
- 提前返回 `dst` 的当前价格；Bellman-Ford 要等当前轮松弛完成。
- 用普通 Dijkstra 的 `dist[city]` 压掉了"更贵但 stops 更少"的状态。
## 模块八：欧拉路径，Reconstruct Itinerary

前面的图论题目大多要求"访问每个节点一次"（比如拓扑排序、岛屿数量、课程表）。Reconstruct Itinerary 要求的是"用掉每一张机票一次"，机票对应图里的边，这是欧拉路径（Eulerian Path）问题：找一条路径，用光图中每一条边恰好一次，节点可以重复经过。

一般图论中，欧拉路径存在需要满足：出度比入度多 1 的节点最多一个（作为起点），入度比出度多 1 的节点最多一个（作为终点），其余节点出度等于入度；如果所有节点出入度都相等，欧拉路径退化为欧拉回路（起点和终点相同）。LeetCode 保证输入一定存在合法行程，代码里不需要再判断是否存在。

题目还要求：如果存在多种合法行程，返回字典序最小的一种。求解的标准算法是 Hierholzer 算法。

### Hierholzer 算法

每个出发机场维护一个目的地的最小堆（或排序后的列表），保证每次都能取出字典序最小的未使用目的地。DFS 从 `JFK` 出发，每一步弹出当前机场堆里最小的目的地并递归深入；当某个机场的堆已经耗尽（没有未使用的机票可以继续走）时，把这个机场追加进 `route`。这个追加动作发生在递归调用返回之后，也就是后序位置。

后序追加是这个算法能正确工作的关键。DFS 沿着字典序最小的选择贪心前进，如果提前走进了一个死胡同（机场还有票没用完，但当前分支已经没有出边可走），后序保证这个死胡同节点会在它所在的死胡同被记录，而递归返回后外层调用会继续尝试同一机场的其它未用机票，把之前跳过的子回路接回主路径。整个 DFS 结束后 `route` 里节点的顺序是"完成时间"从早到晚，终点在最前、起点在最后，把 `route` 整体反转就是从起点开始、按机票使用顺序排列的合法行程。

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

### 手工推演

`tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]`

建堆后每个机场的候选目的地（已排序）：`JFK -> [ATL, SFO]`，`SFO -> [ATL]`，`ATL -> [JFK, SFO]`。

调用 `dfs("JFK")`：
1. `JFK` 弹出 `ATL`（剩 `[SFO]`），进入 `dfs("ATL")`。
2. `ATL` 弹出 `JFK`（剩 `[SFO]`），进入 `dfs("JFK")`。
3. `JFK` 弹出 `SFO`（剩 `[]`），进入 `dfs("SFO")`。
4. `SFO` 弹出 `ATL`（剩 `[]`），进入 `dfs("ATL")`。
5. `ATL` 弹出 `SFO`（剩 `[]`），进入 `dfs("SFO")`。
6. `SFO` 已无候选，后序追加 `SFO`，`route = [SFO]`。
7. 回到第 5 步的 `ATL`，已无候选，后序追加 `ATL`，`route = [SFO, ATL]`。
8. 回到第 3 步的 `SFO`，已无候选，后序追加 `SFO`，`route = [SFO, ATL, SFO]`。
9. 回到第 2 步的 `JFK`，已无候选，后序追加 `JFK`，`route = [SFO, ATL, SFO, JFK]`。
10. 回到第 1 步的 `ATL`，已无候选，后序追加 `ATL`，`route = [SFO, ATL, SFO, JFK, ATL]`。
11. 回到最外层的 `JFK`，已无候选，后序追加 `JFK`，`route = [SFO, ATL, SFO, JFK, ATL, JFK]`。

反转 `route` 得到 `["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"]`，与预期输出一致，已用脚本验证。

### 常见坑

- 每个机场的目的地列表没有按字典序排序（或没有用堆维护），DFS 弹出的不是当前最小的候选目的地，即便最终用光了所有机票，得到的也可能不是字典序最小的那一种合法行程。
- 把追加动作写在进入机场时（前序），没有等到这个机场的堆耗尽后再追加。这样在遇到死胡同子回路时会提前把还没走完的机场记进结果，之后再也没有机会把剩下的机票接回来。典型例子是 `tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]`：`JFK` 的候选里 `KUL` 字典序最小，若一进入 `KUL` 就前序追加，代码会在只用掉一张票时因为 `KUL` 无出边而提前结束；用后序写法，`KUL` 虽然最先被贪心选中，也会因为没有出边而立刻在后序里被记录（此时 `route = [KUL]`），DFS 转而回溯继续 `JFK` 的另一张票 `NRT`，最终反转后得到 `["JFK", "NRT", "JFK", "KUL"]`，`KUL` 被正确地放在了终点位置，已用脚本验证。
- 最后忘记反转 `route`，直接返回后序序列，得到的顺序是终点在前、起点在后，和实际的行程顺序正好相反。
- 用普通列表存目的地并用 `list.remove` 或下标删除模拟"取用"，如果没有先排序或者删除逻辑写错，容易重复选中同一张已用过的机票，或者没有优先选中字典序最小的目的地。

### 16. Reconstruct Itinerary

| 项目 | 内容 |
|---|---|
| 组合技巧 | Hierholzer 算法：每个节点的出边用最小堆维护，DFS 按后序把节点加入结果，最后整体反转 |
| 关键不变量 | 只有当前机场堆耗尽（这个机场发出的所有机票都已使用）时才把它加入 `route`，保证死胡同和子回路能正确接回主路径 |
| 时间 / 空间 | 建堆 `O(E log E)`（`E` 是机票数），DFS 访问每条边一次是 `O(E)`，总计 `O(E log E)`；空间 `O(E)` |

#### Quick Coding：Reconstruct Itinerary

```python
def findItinerary(tickets):
    ...
```

<details>
<summary>参考答案</summary>

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

`graph[node]` 是一个最小堆，保存 `node` 出发但还没用掉的机票目的地，堆顶始终是字典序最小的候选。DFS 每次弹出并深入一个目的地，直到某个机场的堆耗尽，才把这个机场追加进 `route`；这个"耗尽后再追加"的后序写法保证死胡同或子回路总能在恰当的位置被记录下来，不需要额外的回溯撤销逻辑。最后反转 `route`，因为后序记录的顺序是终点在前、起点在后。

</details>
## 模块九：拓扑排序

拓扑排序不是一种"排序数组"的算法，而是把一组依赖关系排成合法执行顺序的图算法。只要题目里出现了"先后关系、依赖、课程、构建顺序、字母顺序、任务调度"，就应该先问自己：这些关系能不能建成一个有向图？

如果每条边 `u -> v` 表示 `u` 必须排在 `v` 前面，那么拓扑排序要输出一个序列，使得图里每条边都从序列左边指向右边。只有 **DAG（Directed Acyclic Graph，有向无环图）** 才一定存在拓扑序；如果存在环，比如 `a -> b -> c -> a`，就代表互相依赖，无法给出合法顺序。

### 什么是拓扑排序

给定一个有向图 `G = (V, E)`：

- `V` 是节点，例如课程、任务、字符。
- `E` 是约束，例如 `pre -> course`、`dependency -> target`、`smaller letter -> larger letter`。
- 拓扑序是一个包含所有节点的线性排列，并满足：对每条边 `u -> v`，`u` 都出现在 `v` 前面。

它解决的是 **偏序到线性序** 的问题。偏序只告诉你一部分先后关系，例如 `h < e`、`e < r`，但没有说所有字符之间都能直接比较。拓扑排序会在不违反已知约束的前提下，给出任意一个合法线性答案。

### Kahn 算法：从"没有前置依赖"的节点开始

Kahn 算法是面试里最直观的 BFS 写法。

1. 建图，并统计每个节点的入度 `indegree`。
2. 把所有入度为 `0` 的节点放入队列。
3. 每次弹出一个节点 `u`，把它加入答案。
4. 遍历 `u` 的所有后继 `v`，把 `indegree[v] -= 1`。
5. 如果某个后继入度降到 `0`，说明它的前置依赖已经全部完成，可以入队。
6. 最后如果答案长度小于节点数，说明图里有环。

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

复杂度是 `O(V + E)`，因为每个节点入队出队一次，每条边只被处理一次。

### DFS 写法：用三色标记找环

DFS 也能做拓扑排序，核心是"后序加入答案"：

- `0 = unvisited`：还没访问过。
- `1 = visiting`：正在当前 DFS 路径上。
- `2 = visited`：这个节点和它的后继都已经处理完。

如果 DFS 时遇到 `visiting` 节点，说明当前路径形成了环；如果一个节点的所有后继都处理完，再把它加入答案。最后反转答案即可得到拓扑序。

Kahn 更适合解释"入度、依赖释放"；DFS 更适合写递归和检测环。面试中两种都可以，但要确保你能清楚解释边方向。

### 17. Course Schedule

题目给出 `numCourses` 门课程，编号 `0` 到 `numCourses - 1`，以及一组先修关系 `prerequisites`，其中 `[a, b]` 表示要学 `a` 必须先学 `b`，也就是一条边 `b -> a`。判断能否把所有课程都学完，等价于判断这张先修关系图是不是 DAG。

这正是 Kahn 算法要回答的问题：图里存在拓扑序，当且仅当能把所有节点都排进 `order`。用上文的入度和队列跑一遍 Kahn 算法，最后比较处理过的节点数和 `numCourses`，处理数不足就说明剩下的节点都困在环里，永远等不到入度降为 `0`。三色 DFS 也能找到同样的答案：只要在当前路径上遇到 `visiting`（颜色 `1`）的节点，就说明存在环。下面给出和 Foreign Dictionary 一致的 Kahn BFS 写法。

| 项目 | 内容 |
|---|---|
| 组合技巧 | Kahn 算法判环 |
| 关键不变量 | 处理过的节点数等于 `numCourses`，当且仅当图中无环 |
| 时间 / 空间 | 时间 `O(V + E)`，空间 `O(V + E)` |

#### Quick Coding：Course Schedule

```python
def canFinish(numCourses, prerequisites):
    ...
```

<details>
<summary>参考答案</summary>

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

`graph[pre].append(course)` 把先修关系翻译成边 `pre -> course`，`[a, b]` 表示 `b -> a` 是同一件事，只是这里直接按边方向存储。`visited_count` 每弹出一个节点就加一，等价于 Kahn 算法里 `order` 的长度。这道题只需要判断课程能否学完，`order` 的具体内容留给下一题 Course Schedule II。

</details>

### 18. Course Schedule II

题目和 Course Schedule 输入相同，但要返回一个具体的课程学习顺序；不存在合法顺序时返回空数组。这就是直接要 Kahn 算法产出的 `order`，不需要额外再判断一次，因为 `order` 本身既是拓扑序，也自带了环检测。

| 项目 | 内容 |
|---|---|
| 组合技巧 | Kahn 算法，直接输出拓扑序 |
| 关键不变量 | `len(order) == numCourses` 时 `order` 是合法拓扑序；否则说明有环，返回 `[]` |
| 时间 / 空间 | 时间 `O(V + E)`，空间 `O(V + E)` |

#### Quick Coding：Course Schedule II

```python
def findOrder(numCourses, prerequisites):
    ...
```

<details>
<summary>参考答案</summary>

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

`order` 和 Foreign Dictionary 里的 `order` 是同一套逻辑，节点从字符换成了课程编号：入度为 `0` 的节点入队，弹出即加入答案，再松弛它的后继。返回前用 `len(order) == numCourses` 判断是否有环，思路和 Foreign Dictionary 里的 `len(order) != len(indegree)` 一致。合法拓扑序通常不止一个，`numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]` 除了 `[0,1,2,3]`，`[0,2,1,3]` 同样合法，只要每条先修边在 `order` 里都是先修课排在前面。

</details>

### 19. Alien Dictionary（Foreign Dictionary）

Foreign Dictionary / Alien Dictionary 的关键不是字符串处理本身，而是从相邻单词里抽出字符之间的偏序关系。以 `words = ["hrn", "hrf", "er", "enn", "rfnn"]` 为例：

这道题的本质流程是：

1. 初始化所有出现过的字符，避免漏掉没有边的孤立节点。
2. 只比较相邻单词，因为字典整体有序时，相邻对提供的是最小必要约束。
3. 找到相邻单词的第一个不同字符 `a` 和 `b`，加入边 `a -> b`，然后停止比较这一对。
4. 如果没有找到不同字符，但前一个单词更长，例如 `["abc", "ab"]`，这是非法输入，直接返回空字符串。
5. 对字符图做拓扑排序；如果检测到环，也返回空字符串。

题目给出一个外星语言的有序词典 `words`。这些单词仍由英文字母组成，但字母大小顺序未知。你需要返回一个合法的字母顺序；如果不存在合法顺序，返回空字符串。题目源：<https://neetcode.io/problems/foreign-dictionary/question?list=neetcode150>

```topo-demo
foreign-dictionary
```

| 项目 | 内容 |
|---|---|
| 组合技巧 | 从相邻单词对抽取字符偏序，Kahn 算法输出拓扑序 |
| 关键不变量 | 相邻单词对里第一个不同字符提供的偏序，等价于整本词典蕴含的全部必要约束 |
| 时间 / 空间 | 时间 `O(N + V + E)`（`N` 为单词总长度），空间 `O(V + E)` |

#### 建图规则

对任意相邻单词 `word1` 和 `word2`：

```text
word1 = h r n
word2 = h r f
             ^
第一个不同字符是 n 和 f，所以 n 必须排在 f 前面，建边 n -> f
```

注意，只看第一个不同字符。后面的字符不能继续拿来建边，因为词典序在第一个差异处就已经决定了两个单词的大小。

#### Python 解法：Kahn BFS

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

时间复杂度是 `O(N + V + E)`，其中 `N` 是所有单词总长度，`V` 是不同字符数，`E` 是字符偏序边数。空间复杂度是 `O(V + E)`。

#### 为什么这段代码能过

- `graph = {char: set() ...}` 先把所有字符注册成节点，保证答案包含孤立字符。
- `if dst not in graph[src]` 防止重复边把入度加多次。
- prefix invalid case 必须在比较字符前处理，否则 `["abc", "ab"]` 会被错误当成没有新约束。
- `len(order) != len(indegree)` 是 Kahn 算法的环检测；如果有环，环内节点永远不会降到入度 `0`。

### 常见坑

- 边方向写反：如果 `word1` 在 `word2` 前面，且第一个不同字符是 `a/b`，应该建 `a -> b`。
- 比较了非首个不同字符：词典序只由第一个不同字符决定。
- 没有处理 prefix invalid：`["abc", "ab"]` 必须返回 `""`。
- 用 list 存邻居但没有去重，导致入度被重复增加。
- 忘记把所有字符放进图，导致答案缺字符。
- 认为答案必须唯一；题目通常允许返回任意一个合法拓扑序。
## 模块十：面试前最后检查

1. 这是隐式图（网格）还是显式图（邻接表/边列表）？隐式图不需要建图，邻接关系由坐标运算给出。
2. 只关心连通性/面积，还是要最短距离/层数？前者用 DFS，后者必须用 BFS，带权图下用 Dijkstra。
3. 起点是一个还是很多个？多个起点直接把它们一起作为 BFS 第 0 层，不需要分别跑多次单源 BFS 再取最小值。
4. 边权是否非负？非负用 Dijkstra；有负权，或者路径长度/边数有显式上限，改用 Bellman-Ford。
5. 题目要连通分量数、判环，还是要生成树？并查集擅长前两者；生成树在并查集驱动的 Kruskal 和堆驱动的 Prim 之间选。
6. 题目要"访问每个节点一次"还是"用掉每条边一次"？前者是普通遍历或拓扑排序，后者是欧拉路径，需要 Hierholzer 算法。
7. 题目里出现"先后关系、依赖、字典序"这类词，能不能建成有向图，再跑拓扑排序？

最后只记一句：

> 图论的第一步不是选算法，而是先想清楚图长什么样：节点是什么，边是什么，边有没有权重和方向，起点有几个。想清楚这些，能用的算法基本就剩一两个了。

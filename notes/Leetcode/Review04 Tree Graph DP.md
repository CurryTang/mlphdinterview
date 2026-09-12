# 复习卡片：树、图与动态规划 (Review Flashcards · Trees, Graphs & DP)

本篇为算法面试高频复习卡片第四辑：系统整理**图论与网格搜索 (Graphs & Grid Search)**、**树与二叉搜索树 (Trees & BST)** 以及**动态规划 (Dynamic Programming)** 的高频核心考题、拓扑变体、生产级实现与时空复杂度全景。

---

## 模块一：图论与网格搜索核心题组 (Graphs & Grid Search)

### 1. 岛屿数量与全景变体全家桶 (Number of Islands & All Canonical Variants)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">图论 01</span>
  <span class="review-card-title">岛屿数量与全景变体全家桶 (Number of Islands & All Canonical Variants)</span>
  <span class="review-card-tag">网格隐式图 · BFS/DFS · 并查集 · 相对坐标归一化 · 外存分块</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与八大变体全景矩阵</div>

给定一个由 `'1'`（陆地）和 `'0'`（水）组成的二维网格 `grid`，计算网格中独立连通岛屿的数量：

| 变体编号 | 核心变体名称 | 核心限制 / 变异条件 | 考察核心与解法突破口 |
|---|---|---|---|
| **变体 1** | **经典计数 (Classic Count)** | 无特殊限制，统计 4 连通块数量 | DFS / BFS Flood Fill 染色遍历 |
| **变体 2** | **不可修改网格 (No-Modify)** | 网格只读（不可就地改写为 `'0'`） | BFS 显式队列 + 外部 `visited`，恪守**入队即标记** |
| **变体 3** | **同形岛屿判重 (Same-Shape)** | 统计形状各异的岛屿数量（去重） | **相对坐标平移归一化** |
| **变体 4** | **海量超大地图 (Huge-Map)** | 地图远超单机内存（如 $10^6 \times 10^6$） | **分块外存切分 + 局部连通 + 跨块边界并查集 (DSU) 缝合** |
| **变体 5** | **水流倾泻可达 (Water-Flow)** | 单元格水流严格向低处流动至边界 | **逆向思维**：从边界海洋出发沿高度非递减方向多源反向扩散 |
| **变体 6** | **2D 连续全 1 扩展 (2D Runs)** | 1D 最长连续 1 推广至 2D 连续全 1 区域 | 任意形状最大面积（DFS 面积累加）vs 最大全 1 矩形（直方图单调栈） |
| **变体 7** | **边界周长追问 (Perimeter)** | 不求连通块数，求岛屿总周长 | 几何代数解：$\text{周长} = 4 \times \text{陆地数} - 2 \times \text{相邻共享边数}$ |
| **变体 8** | **单趟指标聚合 (Aggregation)** | 同一趟遍历同时返回岛屿总数与最大面积 | 遍历连通块的同时用局部计数器累加格子数 |

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

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
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间复杂度严格为 $\mathcal{O}(M \times N)$，空间复杂度 $\mathcal{O}(M \times N)$。

</div>

</div>
</details>

---

### 2. 课程表与拓扑排序全家桶 (Course Schedule & Topological Sort)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">图论 02</span>
  <span class="review-card-title">课程表与拓扑排序全家桶 (Course Schedule & Topological Sort)</span>
  <span class="review-card-tag">Kahn 入度队列 · DFS 三色标记 · 有向环路径提取 · SRE 依赖排查 · 词梯隐式图 BFS</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与实现代码</div>

```python
from collections import deque
from typing import List, Optional

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

    @staticmethod
    def detectAndPrintCycle(numCourses: int, prerequisites: List[List[int]]) -> Optional[List[int]]:
        adj = [[] for _ in range(numCourses)]
        for dest, src in prerequisites: adj[src].append(dest)
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
                    if dfs(v): return True
            color[u] = 2
            return False
        for i in range(numCourses):
            if color[i] == 0 and dfs(i): return cycle
        return None
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与理论依据</div>

- 拓扑排序 $\mathcal{O}(V + E)$；词梯隐式图 $\mathcal{O}(N \times 26 \times L)$。

</div>

</div>
</details>

---

### 3. 带动态 DAG 依赖与节点收缩的规则校验系统 (Order Validator with Dynamic DAG Dependencies & Node Contraction)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">图论 03</span>
  <span class="review-card-title">带动态 DAG 依赖与节点收缩的规则校验系统 (Order Validator with Dynamic DAG Dependencies & Node Contraction)</span>
  <span class="review-card-tag">有向无环图 (DAG) · 动态增删依赖 · 拓扑校验流 · 节点收缩 (Contraction) 邻接重组 · 环检测</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业系统契约</div>

设计并实现一个企业级电商订单校验引擎 `OrderValidator`，能够根据一组动态配置且存在依赖约束的业务规则校验订单数据：

**核心业务需求**：
1. **基础规则能力**：拦截包含违禁品的订单、校验价格区间、发货地址合规性等。
2. **规则动态管理**：支持在线动态添加规则 `add_rule`、添加依赖 `add_dependency(prereq_id, rule_id)`、移除规则 `remove_rule`。
3. **依赖序拓扑执行**：规则之间的前置依赖关系构成有向无环图 (DAG)。在对订单执行校验时，**任何规则必须在其所有前置依赖规则成功通过后才允许执行**。添加依赖时若检测到将形成有向环，必须拒绝该依赖并抛出异常。
4. **核心进阶变体（节点收缩与邻接重组 Node Contraction Rewiring）**：
   - 当某条规则 $R$ 被下线移除时，系统不能粗暴斩断上下游拓扑，而必须进行**图收缩 (Contraction)**：
   - 将 $R$ 的每一个直接前置规则（Predecessors），与 $R$ 的每一个直接后续规则（Successors）之间直接建立新的有向依赖边（即 $Pre(R) \times Succ(R)$ 全连接跨接）；
   - 随后安全删除规则 $R$ 及其所有关联边。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与节点收缩数学性质深度证明</div>

#### 1. 双向邻接表模型设计 (Dual Adjacency Sets)
为了支持极速的拓扑遍历、成环校验与局域边重构，对每个规则节点维护：
- `outgoing[u]: Set[str]`：从 $u$ 出发的后继节点集合；
- `incoming[u]: Set[str]`：指向 $u$ 的前驱依赖集合。
- 采用哈希集合 `set` 能在 $\mathcal{O}(1)$ 内查重、添加与删除边，彻底杜绝重复边的产生。

#### 2. 节点收缩（Node Contraction）为何绝对不会引入有向环？
- **数学定理**：若原图 $G$ 是 DAG（无环），对任意节点 $R$ 进行收缩（将 $R$ 的入边点与出边点直连并删除 $R$）得到的新图 $G'$ **依然严格是 DAG**。
- **反证法证明**：
  - 假设收缩后新图 $G'$ 中产生了有向回路 $C$。
  - 若回路 $C$ 不包含任何新增加的跨接边 $(u, v)$（其中 $u \in Pre(R), v \in Succ(R)$），则 $C$ 在原图 $G$ 中就已经存在，与原图是 DAG 矛盾。
  - 若回路 $C$ 包含了某条新边 $(u, v)$，则在原图 $G$ 中必定存在替代路径：$u \to R \to v$。
  - 将 $C$ 中的所有新边 $(u_i, v_i)$ 均还原为经过 $R$ 的两步路径 $u_i \to R \to v_i$，我们将在原图 $G$ 中构造出一个合法的闭合有向回路！这与原图 $G$ 无环的假设彻底矛盾。
  - 证毕：**节点收缩绝对不会凭空创造有向环**！

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码（含收缩重组与单元测试桩）</div>

```python
from collections import deque
from typing import Dict, Set, List, Callable, Any

class OrderValidator:
    """
    带 DAG 依赖与节点收缩重组的企业级订单校验器
    """
    def __init__(self):
        # 规则函数映射: rule_id -> validate_fn(order) -> bool
        self.rules: Dict[str, Callable[[dict], bool]] = {}
        # 出边: u -> set of v (u 是 v 的前置)
        self.outgoing: Dict[str, Set[str]] = {}
        # 入边: v -> set of u (u 是 v 的前置)
        self.incoming: Dict[str, Set[str]] = {}

    def add_rule(self, rule_id: str, validate_fn: Callable[[dict], bool]) -> None:
        """注册单个校验规则"""
        if rule_id not in self.rules:
            self.rules[rule_id] = validate_fn
            self.outgoing[rule_id] = set()
            self.incoming[rule_id] = set()

    def _creates_cycle(self, src: str, dest: str) -> bool:
        """检查添加边 src -> dest 是否会导致成环（即检验 dest 是否能通过已有路径到达 src）"""
        if src == dest:
            return True
        visited = set()
        queue = deque([dest])
        while queue:
            cur = queue.popleft()
            if cur == src:
                return True
            for nxt in self.outgoing.get(cur, ()):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def add_dependency(self, prereq_id: str, rule_id: str) -> None:
        """添加依赖: prereq_id 必须先于 rule_id 执行"""
        if prereq_id not in self.rules or rule_id not in self.rules:
            raise ValueError("Both rules must exist before adding dependency")
        if self._creates_cycle(prereq_id, rule_id):
            raise ValueError(f"Adding dependency {prereq_id} -> {rule_id} creates a cycle")

        self.outgoing[prereq_id].add(rule_id)
        self.incoming[rule_id].add(prereq_id)

    def remove_rule(self, rule_id: str) -> None:
        """
        核心追问: 移除规则并收缩节点
        将 Pre(rule_id) 中的所有前驱与 Succ(rule_id) 中的所有后继两两直连
        """
        if rule_id not in self.rules:
            return

        preds = self.incoming[rule_id]
        succs = self.outgoing[rule_id]

        # 1. 在前驱与后继之间跨接建立直连依赖边
        for p in preds:
            self.outgoing[p].remove(rule_id)
            for s in succs:
                self.outgoing[p].add(s)

        for s in succs:
            self.incoming[s].remove(rule_id)
            for p in preds:
                self.incoming[s].add(p)

        # 2. 物理销毁规则
        del self.rules[rule_id]
        del self.outgoing[rule_id]
        del self.incoming[rule_id]

    def validate(self, order: dict) -> bool:
        """
        按拓扑排序流依次执行全部规则
        若某规则校验失败立即返回 False
        """
        # 计算当前图内部顶点的入度
        in_deg = {node: len(self.incoming[node]) for node in self.rules}
        queue = deque([node for node, deg in in_deg.items() if deg == 0])
        processed = 0

        while queue:
            cur = queue.popleft()
            processed += 1

            # 执行当前规则业务逻辑
            if not self.rules[cur](order):
                return False

            for nxt in self.outgoing[cur]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)

        if processed != len(self.rules):
            raise RuntimeError("Corrupted DAG state: cycle exists during validation")

        return True
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：
  - 拓扑校验 `validate`：严格 $\mathcal{O}(V + E)$；
  - 边成环检测 `_creates_cycle`：遍历下游可达点，最坏 $\mathcal{O}(V + E)$；
  - 节点删除收缩 `remove_rule`：耗时 $\mathcal{O}(\operatorname{in\_deg} + \operatorname{out\_deg} + \operatorname{in\_deg} \times \operatorname{out\_deg})$，仅涉及局部顶点的笛卡尔积边连接。
- **空间复杂度**：存储双向邻接集合，额外空间为 $\mathcal{O}(V + E)$。
- **核心避坑**：收缩连边时必须同时更新 `outgoing` 与 `incoming` 双向引用，遗漏其一将导致入度统计与逆向拓扑错位。

</div>

</div>
</details>

---

### 4. 带消除障碍物预算的网格最短路径 (Shortest Path in Grid with Obstacles Elimination)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">图论 04</span>
  <span class="review-card-title">带消除障碍物预算的网格最短路径 (Shortest Path in Grid with Obstacles Elimination)</span>
  <span class="review-card-tag">3D 状态空间 BFS · 支配性剪枝 (Dominance Pruning) · 曼哈顿直通捷径</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与状态空间扩增</div>

给定一个 $M \times N$ 的二维网格 `grid`，每个单元格为 `0`（空地）或 `1`（障碍物）。你可以向上下左右四个方向移动。

你拥有最多消除 `k` 个障碍物的预算配额。求从左上角 $(0, 0)$ 到达右下角 $(M-1, N-1)$ 的**最少移动步数**。若无法到达返回 `-1`（LC 1293）：

```python
def shortestPath(grid: List[List[int]], k: int) -> int: ...
```

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与支配性剪枝核心</div>

#### 1. 状态升维与绝对支配性法则 (Dominance Invariant)
- 单纯记录 `(r, c)` 是否访问过会导致致命错误：一条后来到达 $(r, c)$ 的路径虽然步数可能多几步，但它可能保留了更多的消除预算 `remaining_k`，从而能穿过后续更密集的障碍直达终点！
- **支配性剪枝数组**：用二维数组 `visited[r][c]` 记录到达坐标 $(r, c)$ 时**历史上观察到的最大剩余预算**（初值为 -1）。
- 当新状态 $(r, c, k_{cur})$ 到达时：
  - 若 $k_{cur} \le visited[r][c]$：当前状态在预算上被历史最优严格支配，直接剪枝！
  - 若 $k_{cur} > visited[r][c]$：更新 `visited[r][c] = k_{cur}`，并将该状态压入 BFS 队列。

#### 2. 曼哈顿捷径 (Taxicab Shortcut)
- 从起点到终点的最少移动曼哈顿距离为 $(M - 1) + (N - 1)$。在这一最捷径路径上，至多经过 $(M - 1) + (N - 1) - 1$ 个中间格子。
- 若预算 $k \ge (M - 1) + (N - 1) - 1$，无论中间全是障碍还是全是空地，预算都足以一路铲平所有阻碍直达终点！可直接在 $O(1)$ 时间返回曼哈顿距离。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from collections import deque
from typing import List

class ObstacleGridShortestPathSolution:

    @staticmethod
    def shortestPath(grid: List[List[int]], k: int) -> int:
        if not grid or not grid[0]:
            return -1

        m, n = len(grid), len(grid[0])
        if m == 1 and n == 1:
            return 0

        # 曼哈顿捷径加速
        if k >= (m - 1) + (n - 1) - 1:
            return (m - 1) + (n - 1)

        # visited[r][c] 记录抵达该格子的历史最大剩余消除配额
        visited = [[-1] * n for _ in range(m)]
        visited[0][0] = k

        # 队列元素: (r, c, remaining_k, steps)
        queue = deque([(0, 0, k, 0)])

        while queue:
            r, c, rem_k, steps = queue.popleft()

            if r == m - 1 and c == n - 1:
                return steps

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n:
                    nxt_k = rem_k - grid[nr][nc]

                    # 只有当预算非负且严格优于历史最大剩余配额时才拓展
                    if nxt_k >= 0 and nxt_k > visited[nr][nc]:
                        visited[nr][nc] = nxt_k
                        queue.append((nr, nc, nxt_k, steps + 1))

        return -1
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：每个单元格至多被访问 $k$ 次，总体时间复杂度为 $\mathcal{O}(M \times N \times k)$。
- **空间复杂度**：队列与状态记录数组占用 $\mathcal{O}(M \times N \times k)$。

</div>

</div>
</details>

---

## 模块二：树与二叉搜索树 (Trees & BST)

### 5. 最近公共祖先与全景变体全家桶 (Lowest Common Ancestor / LCA)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">树 05</span>
  <span class="review-card-title">最近公共祖先与全景变体全家桶 (Lowest Common Ancestor / LCA)</span>
  <span class="review-card-tag">递归后序分治 · 二叉搜索树数值剪枝 · 父指针哈希交汇</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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
        if left and right: return root
        return left if left else right
```

</div>

</div>
</details>

---

### 6. 二叉树最大路径和全景与路径重构 (Binary Tree Maximum Path Sum & Path Reconstruction)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">树 06</span>
  <span class="review-card-title">二叉树最大路径和全景与路径重构 (Binary Tree Maximum Path Sum & Path Reconstruction)</span>
  <span class="review-card-tag">后序树形 DP · 单侧最大贡献 · 负增益截断 · 全局最优路径重构</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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

### 7. 扁平化多级评论数据转换为嵌套层级树 (Flatten Comment Tree to Multi-Level Hierarchy)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">树 07</span>
  <span class="review-card-title">扁平化多级评论数据转换为嵌套层级树 (Flatten Comment Tree to Multi-Level Hierarchy)</span>
  <span class="review-card-tag">哈希字典映射 · 两趟单线性构建 · 孤儿节点防御 · 环形自引用拦截</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业输入输出规范</div>

在社交媒体、论坛及电商商品详情中，评论数据在数据库中通常以扁平记录表存储。每条记录包含：
`id` (唯一标识), `parent_id` (指向父评论的标识，根评论为 `None`), 以及 `text` (正文内容)。

**任务目标**：将输入的扁平字典列表转换为完整的具有任意深度的**嵌套树状结构数组**，每个评论节点新增 `children: List[dict]` 字段容纳其直接回复子评论，最终按顶层根评论列表的形式返回：

```python
def buildCommentTree(comments: List[dict]) -> List[dict]: ...
```

**示例数据**：
```json
[
  {"id": 1, "parent_id": null, "text": "这是第一条根评论"},
  {"id": 2, "parent_id": null, "text": "这是第二条根评论"},
  {"id": 3, "parent_id": 1, "text": "回复第一条根评论"},
  {"id": 4, "parent_id": 2, "text": "回复第二条根评论"},
  {"id": 5, "parent_id": 3, "text": "孙评论：回复评论3"}
]
```
**期望层级结果**：
- 根评论 1 包含子评论 3，子评论 3 包含子评论 5；
- 根评论 2 包含子评论 4。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与生产级防御架构剖析</div>

#### 1. 两趟哈希线性映射构建法 (Two-Pass O(N) Building)
很多初学者尝试用递归不断在数组中检索子节点，复杂度恶化至 $\mathcal{O}(N^2)$。生产级通用标准是两趟线性字典组装：
- **第一趟 (Pass 1: Node Instantiation)**：
  - 遍历扁平列表，对每个条目创建深克隆或初始化结构体：
    `node_map[item['id']] = {**item, "children": []}`。
  - 无论原数据是否有序或父子是否倒置，第一步保证所有节点对象在内存中已独立就绪。
- **第二趟 (Pass 2: Tree Assembly)**：
  - 再次线性遍历所有原始记录：
    - 若 `parent_id is None`：说明是顶层根节点，加入全局 `roots` 数组；
    - 若 `parent_id` 存在：通过字典常数时间定位其父节点，调用 `node_map[parent_id]["children"].append(cur_node)`！
  - 耗时严格为 $\mathcal{O}(N)$，空间 $\mathcal{O}(N)$，与输入数组的初始顺序完全无关。

#### 2. 面试工业陷阱与防御式校验 (Defensive Sanitization)
- **孤儿评论 (Missing Parent / Orphan)**：若某条记录的 `parent_id` 在全表中根本不存在，直接访问会导致 `KeyError`。生产解法：将其收容至孤儿列表或作为无头根节点暂存。
- **自引用死锁 (Self-Referential Cycle)**：若恶意输入 `parent_id == id`，会导致节点自环造成 JSON 序列化无限递归。必须在第二趟前置条件门禁：`if item['parent_id'] == item['id']: raise ValueError("Self reference detected")`。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List, Dict, Any, Optional

class CommentTreeBuilder:

    @staticmethod
    def buildCommentTree(comments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        扁平评论对象转换为嵌套树：
        时间复杂度 O(N)，空间复杂度 O(N)
        """
        if not comments:
            return []

        # 1. 第一趟：映射构建并赋空 children
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

        # 2. 第二趟：组装层级拓扑关系
        for item in comments:
            cid = item['id']
            pid = item.get('parent_id')
            node = node_map[cid]

            # 自引用安全拦截
            if pid == cid:
                continue

            if pid is None:
                roots.append(node)
            else:
                if pid in node_map:
                    node_map[pid]['children'].append(node)
                else:
                    # 孤儿记录降级为根节点并附带标记
                    node['orphan_warning'] = True
                    roots.append(node)

        return roots
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：两次顺序字典遍历，严格为 $\mathcal{O}(N)$。
- **空间复杂度**：存储字典映射节点，额外空间为 $\mathcal{O}(N)$。

</div>

</div>
</details>

---

### 8. N 叉树根到叶路径和拉平的最小操作数 (Equalize Root-to-Leaf Path Sums in N-ary Tree)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">树 08</span>
  <span class="review-card-title">N 叉树根到叶路径和拉平的最小操作数 (Equalize Root-to-Leaf Path Sums in N-ary Tree)</span>
  <span class="review-card-tag">N 叉树后序遍历 · 树形贪心 · 公共祖先提升 (Greedy Lift) · 自测验证桩</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与操作模型</div>

给定一棵 N 叉树（任意节点可以有任意多个子节点），每个节点内部持有一个整数数值。

我们每次操作可以**选择单个节点并将其数值加 1**。

请问最少需要执行多少次操作，才能使得**从根节点出发到达任意叶子节点的路径上所有节点权值之和完全相等**？

**示例对照**：
- 示例 1：根为 2，拥有两个叶子子节点 3 和 4。
  - 路径和分别为 $2+3=5$ 与 $2+4=6$。
  - 答案：`1`（将节点 3 增加 1 变为 4，两路径和均为 6）。
- 示例 2：
  ```
        1
     2     3
    2 2   3 3
  ```
  - 答案：`4`（若仅允许在叶子节点操作，将左侧两个叶子 2 均增加 2 变为 4，共 4 次操作）。

**面试前置热身小题**：
给定整数数组，求出现频次最高的元素（众数），若有多个并列，返回**数值较小的那一个**。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与树形 DP 贪心提升深度剖析</div>

#### 1. 后序遍历树形 DP 核心状态转移
- 考虑任一节点 $u$ 及其所有子节点集合 $children(u)$：
  - 递归求得每个子节点 $c$ 从自身到其子树叶子节点的最大可能路径和 $\operatorname{sub\_sum}(c)$。
  - 为了让所有从 $u$ 出发流经不同子树到达叶子的路径和拉平，**所有子树的路径和必须统一对齐到当前各子树的最大峰值**：
    $$M = \max_{c \in children(u)} \operatorname{sub\_sum}(c)$$
  - 对于每一个子树 $c$，其落后于峰值的差额 $M - \operatorname{sub\_sum}(c)$ **必须被无条件填平**，该差额直接累加进全局总操作计数器中！
  - 当前节点 $u$ 向其父节点汇报的子树总路径和为：
    $$\operatorname{sub\_sum}(u) = u.val + M$$

#### 2. 内部节点提升 (Greedy Lift) vs 仅叶子节点操作
- **关键面试澄清点**：内部节点允许被加 1 吗？
  - 若**允许增加内部节点**：在示例 2 中，左子树的两个叶子均为 2，父节点为 2（左侧两个总路径和均为 5）；右侧两个叶子均为 3，父节点为 3（右侧两个总路径和均为 7）。我们**直接将左侧父节点 2 增加 2 变为 4**，只需 2 次操作即可同时将左侧两条路径和从 5 提升至 7！操作数从 4 次锐减为 2 次。
  - 必须在编码前与面试官主动澄清这一物理约束。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码（含前置热身与自测试桩）</div>

```python
from typing import List, Dict
from collections import Counter

class NaryTreeNode:
    def __init__(self, val: int = 0, children: List['NaryTreeNode'] = None):
        self.val = val
        self.children = children if children is not None else []

class NaryTreeEqualizeSolution:

    # -------------------------------------------------------------
    # 面试热身小题: 众数检索 (Tie-breaking 向数值较小者倾斜)
    # -------------------------------------------------------------
    @staticmethod
    def mostFrequentSmallest(nums: List[int]) -> int:
        """单趟哈希统计，O(N) 时间，O(N) 空间"""
        if not nums:
            raise ValueError("Array must not be empty")
        counts = Counter(nums)
        best_num = None
        max_freq = -1
        for num, freq in counts.items():
            if freq > max_freq or (freq == max_freq and (best_num is None or num < best_num)):
                max_freq = freq
                best_num = num
        return best_num

    # -------------------------------------------------------------
    # 核心题: 路径和拉平最小递增操作数 (支持内部节点贪心提升)
    # -------------------------------------------------------------
    @classmethod
    def minOperationsToEqualize(cls, root: NaryTreeNode) -> int:
        if not root:
            return 0

        total_ops = 0

        def postorder(node: NaryTreeNode) -> int:
            """返回以 node 为根的子树中，从 node 到其叶子的最大路径和"""
            nonlocal total_ops
            if not node.children:
                return node.val

            # 递归计算所有子节点的子树路径和
            child_sums = [postorder(child) for child in node.children]
            max_child_sum = max(child_sums)

            # 将每个落后的子树补齐至 max_child_sum
            for s in child_sums:
                total_ops += (max_child_sum - s)

            return node.val + max_child_sum

        postorder(root)
        return total_ops
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：树中每个节点被后序遍历访问恰好一次，时间复杂度为严格 $\mathcal{O}(N)$。
- **空间复杂度**：递归调用栈深度为树高度 $\mathcal{O}(H)$。

</div>

</div>
</details>

---

## 模块三：动态规划核心题组 (Dynamic Programming)

### 9. 零钱兑换与完全背包模型全景 (Coin Change 1 & 2 / Unbounded Knapsack)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">DP 09</span>
  <span class="review-card-title">零钱兑换与完全背包模型全景 (Coin Change 1 & 2 / Unbounded Knapsack)</span>
  <span class="review-card-tag">完全背包 · 最值模型 vs 组合数模型 · 循环顺序本质</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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

### 10. 贴纸拼词与状态压缩动态规划 (Stickers to Spell Word & Bitmask DP)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">DP 10</span>
  <span class="review-card-title">贴纸拼词与状态压缩动态规划 (Stickers to Spell Word & Bitmask DP)</span>
  <span class="review-card-tag">状态压缩 · 记忆化搜索 · 首个未满足字符剪枝 · O(2^n * m * n)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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
<div class="review-block-label">⏱️ 复杂度分析</div>

- 时间 $\mathcal{O}(2^N \cdot M \cdot N)$，空间 $\mathcal{O}(2^N)$。

</div>

</div>
</details>

---

### 11. 网格最长交替折线路径 (Longest Alternating Zigzag Path in 2D Grid)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 11</span>
  <span class="review-card-title">网格最长交替折线路径 (Longest Alternating Zigzag Path in 2D Grid)</span>
  <span class="review-card-tag">二维网格 · 记忆化搜索 · 状态机DP · O(M * N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class LongestZigzagPathSolution:
    @classmethod
    def longestZigzag(cls, grid: List[List[int]]) -> int:
        """
        计算二维网格中数值交替严格递增与递减的最长路径节点数。
        
        状态设计：
        (r, c, expect_greater)
        - expect_greater = True: 下一步必须移动到数值严格更大的邻居 (grid[nr][nc] > grid[r][c])
        - expect_greater = False: 下一步必须移动到数值严格更小的邻居 (grid[nr][nc] < grid[r][c])
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
                # 环路保护 (针对一般图回溯防护，网格交替状态图拓扑基底)
                return 1
            
            visiting.add(state)
            best_len = 1  # 至少包含当前单元格自身
            
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
                # 任何格子都可以作为起点，首步既可尝试递增起步，亦可尝试递减起步
                max_path = max(max_path, dfs(r, c, True), dfs(r, c, False))
        
        return max_path
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **状态空间与二元交替机**：
  每一个单元格 $(r, c)$ 在路径上具有两种入轨意图——“要求下一跳变大”或“要求下一跳变小”。因此整个搜索空间被规范为有界离散状态集合 $\mathcal{S} = \{ (r, c, d) \mid 0 \le r < m, 0 \le c < n, d \in \{0, 1\} \}$，总状态数为 $2MN$。
- **严格不等与等值阻断**：
  当相邻格子数值相等时（$grid[nr][nc] == grid[r][c]$），交替关系破裂，不可作为后继延伸。
- **记忆化 vs 递归回溯**：
  若不进行状态记忆，最坏情况下路径分支将发生指数级退化 $\mathcal{O}(4^L)$；通过对 $(r, c, d)$ 建立记忆表，将问题转化为状态图上的最长路求解，平摊每个状态仅深度遍历一次。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(M \cdot N)$。总共有 $2MN$ 个离散状态，每个状态仅被精确求值一次，遍历 4 个正交方向分支。
- **空间复杂度**：$\mathcal{O}(M \cdot N)$。记忆化哈希表与递归调用栈开销均为 $\mathcal{O}(M \cdot N)$。

</div>

</div>
</details>

---

### 12. 多叉树垂直自顶向下目标路径和 (N-ary Tree Downward Target Path Sum)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">TREE 12</span>
  <span class="review-card-title">多叉树垂直自顶向下目标路径和 (N-ary Tree Downward Target Path Sum)</span>
  <span class="review-card-tag">多叉树 · 前缀和哈希表 · 回溯作用域清理 · O(N)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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
        统计 N 叉树中所有自顶向下连续路径，使得节点值之和等于 target。
        利用根到节点的前缀和哈希表在 O(N) 时间内完成计数。
        """
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # 基准前缀和：若某前缀和刚好等于 target，其差值为 0
        total_valid_paths = 0

        def dfs(node: Optional[NaryTreeNode], current_prefix_sum: int) -> None:
            nonlocal total_valid_paths
            if not node:
                return

            current_prefix_sum += node.val
            # 满足: current_prefix_sum - ancestor_prefix_sum = target
            # 即: ancestor_prefix_sum = current_prefix_sum - target
            total_valid_paths += prefix_counts[current_prefix_sum - target]

            # 将当前前缀和注册入哈希表
            prefix_counts[current_prefix_sum] += 1

            # 深入遍历所有子节点分支
            for child in node.children:
                dfs(child, current_prefix_sum)

            # 回溯关键：离开当前节点子树时，撤销当前节点前缀和，严格保持祖先路径作用域
            prefix_counts[current_prefix_sum] -= 1

        dfs(root, 0)
        return total_valid_paths
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **前缀和差分原语**：
  若存在祖先节点 $u$ 到当前节点 $v$ 的路径和等于 $target$，根据前缀和定义：
  $$\sum_{w \in 	ext{path}(u 	o v)} 	ext{val}(w) = S(v) - S(	ext{parent}(u)) = target \implies S(	ext{parent}(u)) = S(v) - target$$
  只需在进入 $v$ 时查询当前作用域内值为 $S(v) - target$ 的祖先节点个数即可。
- **为何双指针/滑动窗口彻底失效**：
  1. 树节点数值可能包含负数与零，前缀和序列失去了单调递增性；
  2. 多叉树具有树状分支分叉，不能沿一维双指针进行首尾收缩。
- **回溯作用域单调性（Backtracking Scope Cleanliness）**：
  哈希表 `prefix_counts` 仅反映**当前遍历路径中各祖先节点**的前缀和。递归返回前必须执行 `prefix_counts[current_prefix_sum] -= 1`，避免跨分支产生脏数据污染。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(V)$。每个树节点恰好被访问一次，哈希表的查询与插入均为平摊 $\mathcal{O}(1)$。
- **空间复杂度**：$\mathcal{O}(H)$。递归调用栈与哈希表存储的有效祖先节点数量受限于树的最大深度 $H$（最坏退化链表为 $\mathcal{O}(V)$，平衡树为 $\mathcal{O}(\log V)$）。

</div>

</div>
</details>

---

### 13. 字典树加速网格单词搜寻 (Word Search II with Trie & Backtracking Pruning)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 13</span>
  <span class="review-card-title">字典树加速网格单词搜寻 (Word Search II with Trie & Backtracking Pruning)</span>
  <span class="review-card-tag">Trie 前缀树 · 网格回溯 · 动态叶节点剪枝 · 原地状态置换</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List, Dict, Any

class WordSearchIISolution:
    @classmethod
    def findWords(cls, board: List[List[str]], words: List[str]) -> List[str]:
        """
        在字符网格中查找给定词表的所有合法拼写单词。
        结合 Trie 前缀树与动态叶节点剔除剪枝。
        """
        if not board or not board[0] or not words:
            return []

        # 1. 建立 Trie
        root: Dict[str, Any] = {}
        for word in words:
            curr = root
            for ch in word:
                curr = curr.setdefault(ch, {})
            curr['$'] = word  # 终端标识直接存储完整单词，避免字符串累加开销

        m, n = len(board), len(board[0])
        result = []

        # 2. 网格深度优先回溯与动态 Trie 剪枝
        def dfs(r: int, c: int, parent_node: Dict[str, Any]) -> None:
            ch = board[r][c]
            curr_node = parent_node[ch]

            # 命中有完整单词
            matched_word = curr_node.pop('$', None)
            if matched_word is not None:
                result.append(matched_word)

            # 原地标记访问，避免二次使用
            board[r][c] = '#'

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and board[nr][nc] in curr_node:
                    dfs(nr, nc, curr_node)

            # 恢复现场
            board[r][c] = ch

            # 动态剪枝：若当前节点成为无分支叶子节点，从父节点中移除，杜绝后续多余搜索
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
<div class="review-block-label">💡 机制剖析</div>

- **全局共享前缀加速**：
  若对每个单词分别在网格中执行 DFS，时间复杂度为 $\mathcal{O}(W \cdot M \cdot N \cdot 4^L)$，将引发超时。将 $W$ 个单词合并构造成 Trie，所有前缀重叠部分仅在网格中走一次。
- **终端叶节点实时卸载（On-the-fly Trie Leaf Removal）**：
  当一个单词被找到后，首先通过 `curr_node.pop('$', None)` 防止同词重复加入结果集；随后若该字典树节点没有任何其余子字符分支，直接在父节点调用 `parent_node.pop(ch)` 彻底将其摘除。这使得已被消耗完毕的单词分支在后续扫描中以 $\mathcal{O}(1)$ 阻断，显著收敛搜索分支。
- **原地修改标记与零额外空间**：
  借由将当前单元格设为 `'#'` 并在返回时复原，免除了创建和维护庞大 `visited` 哈希集合的昂贵开销。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：建树开销 $\mathcal{O}(\sum |W_i|)$。网格回溯最坏上界为 $\mathcal{O}(M \cdot N \cdot 4 \cdot 3^{L-1})$（$L$ 为词表中单词的最大长度），但在动态修剪下实际速度逼近 $\mathcal{O}(M \cdot N + \sum |W_i|)$。
- **空间复杂度**：$\mathcal{O}(\sum |W_i|)$，用于存储字典树树形节点字典。

</div>

</div>
</details>

---

### 14. 带油箱与充能站的网格最短路径 (Grid Shortest Path with Fuel Tank & Recharge Stations)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 14</span>
  <span class="review-card-title">带油箱与充能站的网格最短路径 (Grid Shortest Path with Fuel Tank & Recharge Stations)</span>
  <span class="review-card-tag">状态空间扩展 · Dijkstra 最短路 · 充能状态坍缩 · 充能超图优化</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

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
        在带障碍物、异构单元格进入代价和充能站的网格中，求解抵达右下角的最小花费。
        
        状态设计: (cost, r, c, fuel)
        到达 (r, c) 时剩余油量为 fuel，若该格为充能站，fuel 立即强制重置为 K。
        """
        m, n = len(grid_cost), len(grid_cost[0])
        if blocked[0][0] or blocked[m - 1][n - 1]:
            return -1

        start_cost = grid_cost[0][0]
        start_fuel = K
        
        # dist[(r, c, fuel)] 记录到达该扩展状态的最小已付代价
        dist = {}
        dist[(0, 0, start_fuel)] = start_cost
        
        # 优先队列维护 (cost, r, c, fuel)
        pq = [(start_cost, 0, 0, start_fuel)]

        while pq:
            cost, r, c, fuel = heapq.heappop(pq)

            # 目标检测：由于 Dijkstra 单调出队性质，首次弹出终点即为最小代价
            if r == m - 1 and c == n - 1:
                return cost

            if cost > dist.get((r, c, fuel), float('inf')):
                continue

            # 若油量耗尽，且当前格子非终点，则无法继续下一步移动
            if fuel == 0:
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and not blocked[nr][nc]:
                    # 状态转移：进入充能站则充满至 K，否则油量减 1
                    next_fuel = K if recharge[nr][nc] else fuel - 1
                    next_cost = cost + grid_cost[nr][nc]

                    if next_cost < dist.get((nr, nc, next_fuel), float('inf')):
                        dist[(nr, nc, next_fuel)] = next_cost
                        heapq.heappush(pq, (next_cost, nr, nc, next_fuel))

        return -1
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **分层图（Layered Graph / State Expansion）建模**：
  普通的网格最短路状态仅包含 $(r, c)$，但在带油量约束时，同一个坐标在持有不同油量下的后续可达性存在本质区别。因此将物理坐标扩展为三维状态 $(r, c, 	ext{fuel})$，边权为目标格子的 `grid_cost[nr][nc]`。
- **充能状态坍缩（State Space Collapse）**：
  若某一非障碍格子是充能站（`recharge[nr][nc] == True`），进入后剩余油量无条件补满至 $K$。此时所有流入该格子的前驱状态，在出格转移时均统一坍缩为 $(nr, nc, K)$ 唯一状态。
- **极端大 $K$ 场景优化策略（Large-K Supergraph Reduction）**：
  1. 若 $K \ge m + n - 2$：油量无法对路径构成任何约束，状态第三维 $	ext{fuel}$ 可直接舍弃，退化为经典二维网格 Dijkstra，复杂度降为 $\mathcal{O}(MN \log(MN))$。
  2. 若充能站数量稀疏（$R \ll MN$）且 $K$ 较大：可构造**充能站超图（Recharge Supergraph）**。顶点集合为 $\{	ext{Start}, 	ext{Goal}\} \cup \{	ext{All Recharge Stations}\}$。利用网格 BFS/Dijkstra 预处理各顶点在 $K$ 步以内的成对最短可达距离，在仅含 $\mathcal{O}(R)$ 顶点的紧凑超图上执行最短路搜索。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(M \cdot N \cdot K \log(M \cdot N \cdot K))$。扩展状态空间节点数为 $\mathcal{O}(MNK)$，每条转移边至多入堆一次。
- **空间复杂度**：$\mathcal{O}(M \cdot N \cdot K)$。距离表与优先队列最大容纳的状态数。

</div>

</div>
</details>

---

### 15. 图片相似度聚类与并查集连通分量 (Photo Similarity Groups via Union-Find)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">GRAPH 15</span>
  <span class="review-card-title">图片相似度聚类与并查集连通分量 (Photo Similarity Groups via Union-Find)</span>
  <span class="review-card-tag">并查集 (DSU) · 连通分量计数 · 上三角矩阵遍历 · O(N^2 * α(N))</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List

class PhotoSimilarityGroupsSolution:
    @classmethod
    def findGroups(cls, isSimilar: List[List[int]]) -> int:
        """
        计算 N 张图片通过直接/间接相似关系形成的独立图片组（连通分量）数量。
        """
        if not isSimilar:
            return 0

        n = len(isSimilar)
        parent = list(range(n))
        rank = [0] * n
        components_count = n

        def find(i: int) -> int:
            # 路径压缩 (Path Compression)
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> bool:
            nonlocal components_count
            root_i, root_j = find(i), find(j)
            if root_i == root_j:
                return False
            
            # 按秩合并 (Union by Rank)
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1

            components_count -= 1
            return True

        # 仅需遍历严格上三角矩阵，利用对称性降低常数开销
        for i in range(n):
            for j in range(i + 1, n):
                if isSimilar[i][j] == 1:
                    union(i, j)

        return components_count
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **无向图连通分支的代数同构**：
  矩阵的对称性与相似的传递性构成了标准的等价关系。求解相似组数等价于无向图 $G = (V, E)$ 中独立极大连通子图（Connected Components）的数量。
- **严格上三角遍历**：
  由于相似关系具有对称性（$isSimilar[i][j] == isSimilar[j][i]$），且对角线元素自反恒为 1，只需检查 $j > i$ 的上三角区域，有效循环次数由 $N^2$ 减半为 $rac{N(N-1)}{2}$。
- **并查集 vs 广度优先搜索 (BFS/DFS)**：
  - 在密集邻接矩阵输入下，无论采用 DSU 还是 BFS 均受限于 $\mathcal{O}(N^2)$ 的矩阵元素扫描下界；
  - 但并查集具有优良的**流式（Streaming/Online）扩展性**：若后续动态新增相似图片对，并查集仅需 $\mathcal{O}(lpha(N))$ 即可增量合并，无需重新遍历整图。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：$\mathcal{O}(N^2 \cdot lpha(N))$，其中 $lpha$ 为反阿克曼函数。由上三角扫描主导，整体在常数上优于全矩阵扫描。
- **空间复杂度**：$\mathcal{O}(N)$，维护长度为 $N$ 的父指针与秩数组。

</div>

</div>
</details>

---

### 16. 二叉树右视图与自建树脚手架 (Binary Tree Right Side View with Custom Tree Scaffolding)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">TREE 16</span>
  <span class="review-card-title">二叉树右视图与自建树脚手架 (Binary Tree Right Side View with Custom Tree Scaffolding)</span>
  <span class="review-card-tag">二叉树 · 层序遍历 BFS · 逆先序 DFS · 测试树自动构建</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 核心代码</div>

```python
from typing import List, Optional
from collections import deque

class TreeNode:
    """工程规范二叉树节点定义"""
    def __init__(self, val: int = 0, left: Optional['TreeNode'] = None, right: Optional['TreeNode'] = None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTreeScaffolding:
    """自建树脚手架：从层序数组（包含 None）构建标准二叉树"""
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
            
            # 挂载左子节点
            if idx < n and values[idx] is not None:
                curr.left = TreeNode(values[idx])
                queue.append(curr.left)
            idx += 1
            
            # 挂载右子节点
            if idx < n and values[idx] is not None:
                curr.right = TreeNode(values[idx])
                queue.append(curr.right)
            idx += 1

        return root

class RightSideViewSolution:
    @classmethod
    def rightSideViewBFS(cls, root: Optional[TreeNode]) -> List[int]:
        """解法一：BFS 层序遍历，每层记录最后一个出队元素"""
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
        """解法二：逆先序 DFS (根 -> 右 -> 左)，按深度首次命中记录"""
        result = []

        def dfs(node: Optional[TreeNode], depth: int) -> None:
            if not node:
                return
            # 当当前深度等于当前结果集长度时，说明该深度首次被访问（即为该层最右节点）
            if depth == len(result):
                result.append(node.val)
            dfs(node.right, depth + 1)
            dfs(node.left, depth + 1)

        dfs(root, 0)
        return result
```

</div>

<div class="review-block">
<div class="review-block-label">💡 机制剖析</div>

- **解法对比（BFS vs 逆先序 DFS）**：
  - **BFS 队列法**：直接按物理层推进，天然隔离各深度，每层最后一个访问的节点即为右侧视线落点。直观且不易出错，空间复杂度取决于最大层宽 $W$。
  - **逆先序 DFS**：遍历顺序固定为 `根 -> 右孩子 -> 左孩子`。通过 `depth == len(result)` 条件判定当前层是否已落库，空间复杂度由树高 $H$ 决定（平衡树下仅需 $\mathcal{O}(\log N)$）。
- **右视图语义纠偏**：
  右视图绝不等于“从根节点一直向 `right` 走的叶节点分支”。当右子树在某一层缺失而左子树存在更深节点时，左子树的边缘节点仍会在该深度对右视线可见。

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度分析</div>

- **时间复杂度**：两种方法均为 $\mathcal{O}(N)$，每个节点遍历一次。
- **空间复杂度**：
  - BFS：$\mathcal{O}(W)$，其中 $W$ 为二叉树单层最大节点数（满二叉树叶层为 $\mathcal{O}(N)$）。
  - DFS：$\mathcal{O}(H)$，其中 $H$ 为树的高度（平衡树为 $\mathcal{O}(\log N)$，退化链表为 $\mathcal{O}(N)$）。

</div>

</div>
</details>


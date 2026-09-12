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

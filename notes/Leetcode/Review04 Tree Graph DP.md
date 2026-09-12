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
| **变体 3** | **同形岛屿判重 (Same-Shape)** | 统计形状各异的岛屿数量（去重） | **相对坐标平移归一化** 或 **带回溯标记 'B' 的 DFS 路径签名** |
| **变体 4** | **海量超大地图 (Huge-Map)** | 地图远超单机内存（如 $10^6 \times 10^6$） | **分块外存切分 + 局部连通 + 跨块边界并查集 (DSU) 缝合** |
| **变体 5** | **水流倾泻可达 (Water-Flow)** | 单元格水流严格向低处流动至边界 | **逆向思维**：从边界海洋出发沿高度非递减方向多源反向扩散 |
| **变体 6** | **2D 连续全 1 扩展 (2D Runs)** | 1D 最长连续 1 推广至 2D 连续全 1 区域 | 任意形状最大面积（DFS 面积累加）vs 最大全 1 矩形（直方图单调栈） |
| **变体 7** | **边界周长追问 (Perimeter)** | 不求连通块数，求岛屿总周长 | 几何代数解：$\text{周长} = 4 \times \text{陆地数} - 2 \times \text{相邻共享边数}$，做到 $O(1)$ 空间 |
| **变体 8** | **单趟指标聚合 (Aggregation)** | 同一趟遍历同时返回岛屿总数与最大面积 | 遍历连通块的同时用局部计数器累加格子数，单趟双更新 |

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：每个单元格进出队列各一次，严格 $\mathcal{O}(M \times N)$。
- **空间复杂度**：最坏情况下队列空间为 $\mathcal{O}(M \times N)$。
- **高频避坑**：BFS 遍历必须**入队即标记**，出队再标记会导致同一节点被多个邻居重复入队引发指数爆炸。

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
<div class="review-block-label">📌 题目定义与五大变体矩阵</div>

你这个学期必须选修 `numCourses` 门课程，记为 `0` 到 `numCourses - 1`。在选修某些课程之前需要先修前置课程，给定前置关系数组 `prerequisites`，其中 `[a, b]` 表示要想修课程 `a` 必须先修课程 `b`（即存在有向边 $b \to a$）。

主要有两种经典标准考法与三大工业变体：
- **可行性判定 (Feasibility, LC 207)**：`canFinish(numCourses: int, prerequisites: List[List[int]]) -> bool`
- **拓扑序输出 (Order, LC 210)**：`findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]`

| 变体编号 | 核心变体名称 | 核心特征 / 变异条件 | 算法架构与破局关键 |
|---|---|---|---|
| **变体 1** | **拓扑排序可行性 (LC 207)** | 判断是否存在有效的学业完成计划 | **Kahn 算法 (BFS 入度表)**：统计各点入度，入度为 0 压队；统计出队总数是否等于 $V$。 |
| **变体 2** | **拓扑序全量重构 (LC 210)** | 返回任意一种合法的拓扑排序列表，若成环返回空列表 | Kahn 算法中将出队元素顺序记录进结果集 `order`；若成环清空返回 `[]`。 |
| **变体 3** | **SRE 架构有向依赖成环排查** | 输入格式为自定义服务的远程调用有向边，排查微服务依赖死锁 | 解析自定义 edge 结构构图，应用 Kahn 算法定位所有入度 $> 0$ 陷入环中的故障服务集。 |
| **变体 4** | **DAG 拓扑输出 + DFS 环路径捕获打印** | 若存在有效顺序输出 DAG 拓扑；若存在环，**必须打印出该环的完整节点回路** | **DFS 三色标记法**（0 白、1 灰、2 黑）：遇到灰色节点时回溯父节点指针链，精确打印出闭合环回路。 |
| **变体 5** | **多步延伸：词梯隐式图最短路径 (LC 127)** | 拓扑热身通过后，追加求从初始词到目标词的最短单字符变换步数 | **隐式图即时生成 (On-the-fly) BFS**：绝不预建 $O(N^2)$ 全量边表，而是对每个出队单词动态替换 26 个字母并在哈希字典中检索。 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与算法架构深度抉择</div>

#### 1. Kahn 算法 (BFS 入度队列) vs DFS 三色标记法深度辩护
- **Kahn 算法（首选：直观、拓扑序生成天然、无递归栈溢出）**：
  1. 统计每个顶点的入度 $\operatorname{in\_degree}[u]$ 与邻接表 `adj[u]`。
  2. 初始化队列，将所有入度为 0 的顶点入队（表示没有任何先决依赖，可立即执行）。
  3. 循环出队 $u$，将 $u$ 追加至拓扑序列 `order`；遍历 $u$ 的后继邻居 $v$，令其入度减 1（$\operatorname{in\_degree}[v] \gets \operatorname{in\_degree}[v] - 1$）；若入度降为 0，将 $v$ 入队。
  4. 最终若 `len(order) == numCourses`，说明所有节点均安全消除，无环；否则说明图中存在有向环。
- **DFS 三色标记法（首选：精准捕获并打印有向环回路）**：
  - 节点着色契约：
    - `0` (白色 White)：尚未被访问过的节点；
    - `1` (灰色 Gray)：正在当前递归调用路径上（祖先链条）的活跃节点；
    - `2` (黑色 Black)：该节点及其所有下游子树均已遍历完毕，已确认安全无环。
  - **成环判据**：若 DFS 访问到某个邻接点处于灰色（状态为 1），代表撞上了当前递归栈上的直系祖先，**必定抓到了有向环**！
  - **为何打印环选 DFS？** 因为 DFS 的递归栈（或维护的 `parent` 映射）完整持有了从环起点到当前节点的闭合路径，只需沿 `parent` 回溯即可完美还原环的节点全貌。

#### 2. 词梯 (Word Ladder) 隐式图 BFS 剪枝关键
- 字典单词规模较大时（$N = 5000$），若两两比对单词字符差异构建显式邻接图，耗时为 $\mathcal{O}(N^2 \cdot L)$，极易超时。
- **工业级标准：按位变换 26 个字母动态查找**：
  - 对当前出队单词 `word`，遍历长度 $L$ 的每个位置，用 `'a'` 到 `'z'` 替换；
  - 检查替换后的单词是否存在于 `wordSet` 中；若存在则加入 BFS 队列并**立刻从 `wordSet` 中物理删除**（充当 `visited` 集合，杜绝重复搜索）。
  - 单次查找耗时 $\mathcal{O}(26 \times L \times 1)$，总体复杂度严格为 $\mathcal{O}(N \times 26 \times L)$。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from collections import deque
from typing import List, Optional

class CourseScheduleSolution:
    """
    拓扑排序与环检测核心体系：
    1. canFinish / findOrder: Kahn BFS
    2. detectAndPrintCycle: DFS 三色标记环提取
    3. ladderLength: 词梯隐式图即时生成 BFS
    """

    @staticmethod
    def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        """LC 210: Kahn 算法输出拓扑排序，若有环返回 []，O(V + E)"""
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
        """DFS 三色标记法：检测环并完整还原并打印环的节点回路"""
        adj = [[] for _ in range(numCourses)]
        for dest, src in prerequisites:
            adj[src].append(dest)

        # 0: white (unvisited), 1: gray (visiting), 2: black (visited)
        color = [0] * numCourses
        parent = [-1] * numCourses
        cycle = []

        def dfs(u: int) -> bool:
            color[u] = 1  # 标记为灰色
            for v in adj[u]:
                if color[v] == 1:
                    # 发现环！回溯 parent 链提取闭合回路
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
            color[u] = 2  # 标记为黑色
            return False

        for i in range(numCourses):
            if color[i] == 0:
                if dfs(i):
                    return cycle
        return None

    @staticmethod
    def ladderLength(beginWord: str, endWord: str, wordList: List[str]) -> int:
        """LC 127: 词梯最短路径，隐式图即时生成 BFS，O(N * 26 * L)"""
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

            # 动态替换 26 个字母查找邻居
            for i in range(L):
                for ch in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + ch + word[i+1:]
                    if next_word in word_set:
                        word_set.remove(next_word)  # 物理摘除防止成环
                        queue.append((next_word, step + 1))

        return 0
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：
  - 拓扑排序 (Kahn 与 DFS)：构图与遍历时间严格为 $\mathcal{O}(V + E)$（$V$ 为课程数，$E$ 为依赖对数）。
  - 词梯：$\mathcal{O}(N \times 26 \times L)$，其中 $N$ 为字典大小，$L$ 为单词长度。
- **空间复杂度**：
  - 邻接表、入度表与队列均为 $\mathcal{O}(V + E)$。
- **高频避坑清单**：
  1. **建图方向倒置**：输入 `[a, b]` 代表 $b \to a$。若误写为 $a \to b$，拓扑序完全逆反。
  2. **词梯重复入队引发爆炸**：生成合法新单词加入队列时，必须**立即从 `word_set` 中剔除**。若等到出队时再剔除，同一单词会被多个路径重复压入队列导致 TLE / MLE。

</div>

</div>
</details>

---

## 模块二：树与二叉搜索树 (Trees & BST)

### 3. 最近公共祖先与全景变体全家桶 (Lowest Common Ancestor / LCA)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">树 03</span>
  <span class="review-card-title">最近公共祖先与全景变体全家桶 (Lowest Common Ancestor / LCA)</span>
  <span class="review-card-tag">递归后序分治 · 二叉搜索树数值剪枝 · 父指针哈希交汇 · 节点存在性校验</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与核心变体全景矩阵</div>

给定一棵二叉树的根节点 `root` 以及两个指定节点 `p` 和 `q`，找到该树中两节点的最近公共祖先 (LCA)：

```python
def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode': ...
```

| 变体编号 | 核心变体名称 | 核心特征 / 变异条件 | 算法架构与破局关键 |
|---|---|---|---|
| **变体 1** | **通用二叉树 LCA (LC 236)** | 无序任意二叉树，保证 $p, q$ 均存在 | **后序分治递归**：左右子树各自搜寻，两边均有则当前为 LCA，单边有则返回该边。 |
| **变体 2** | **二叉搜索树 LCA (LC 235)** | 树满足 BST 性质（左小右大） | **数值区间剪枝**：若 $p, q$ 均小于 root 走左边，均大于走右边，分岔点即为 LCA。$O(H)$ 时间 $O(1)$ 空间。 |
| **变体 3** | **节点可能不存在 (LC 1644)** | 树中可能根本没有 $p$ 或 $q$ | 必须完整后序遍历全树并统计发现计数 `count == 2`，不可提前返回剪枝。 |
| **变体 4** | **带父指针节点 (LC 1650)** | 节点包含 `parent` 指针，不给根节点 | **相交链表求交点模型**：双指针追赶法，走完自己走对方，步数相同时相遇即为 LCA。 |

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class LCASolution:
    @staticmethod
    def lowestCommonAncestor(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """LC 236: 通用二叉树 LCA，O(N) 时间，O(H) 空间"""
        if not root or root == p or root == q:
            return root
        left = LCASolution.lowestCommonAncestor(root.left, p, q)
        right = LCASolution.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        return left if left else right

    @staticmethod
    def lowestCommonAncestorBST(root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """LC 235: BST LCA，数值单调剪枝，O(H) 时间，O(1) 空间"""
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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：二叉树为 $\mathcal{O}(N)$，BST 为 $\mathcal{O}(H)$。
- **空间复杂度**：递归栈深度 $\mathcal{O}(H)$。

</div>

</div>
</details>

---

### 4. 二叉树最大路径和全景与路径重构 (Binary Tree Maximum Path Sum & Path Reconstruction)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">树 04</span>
  <span class="review-card-title">二叉树最大路径和全景与路径重构 (Binary Tree Maximum Path Sum & Path Reconstruction)</span>
  <span class="review-card-tag">后序树形 DP · 单侧最大贡献 · 负增益截断 · 全局最优路径重构 · 向下单向约束</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与核心演进变体矩阵</div>

二叉树中的**路径**被定义为一条节点序列，序列中每对相邻节点在树中都存在一条边相连。同一个节点在一条路径序列中**至多出现一次**。该路径**至少包含一个节点**，且不一定经过根节点。

求该树中所有可能路径的**最大路径和**：

```python
def maxPathSum(root: Optional[TreeNode]) -> int: ...
```

在系统大厂（如 Meta / Google）多轮代码考核中，该题常引申出以下四大高频追问：

| 变体编号 | 核心变体名称 | 核心变异约束 / 面试官追问 | 递归契约与解题突破口 |
|---|---|---|---|
| **变体 1** | **经典最大路径和 (LC 124)** | 节点可能为负数，路径可任意弯折一次 | **树形后序 DP**：递归返回以当前节点为端点向父节点延伸的“单侧最大增益”；副作用更新全局最大曲折和。 |
| **变体 2** | **还原最优路径自身 (Return Path)** | 不仅要输出最大得分，还需**输出最优路径上的有序节点值列表** | 递归时同步返回 `(gain, arm_path)`；更新全局最优时将左单臂反转 + `[node.val]` + 右单臂拼接。 |
| **变体 3** | **仅向下单向路径 (Downward-Only)** | 路径必须严格自顶向下（父到子），且要求判断是否存在某路径和等于目标值 `target` | **前缀和哈希表 (Prefix Sum)**：在自顶向下的 DFS 路径上维护 `prefix_sums[sum - target]`，降维至 $O(N)$。 |
| **变体 4** | **严禁穿过根节点 / 必须为非叶节点** | 额外增加拓扑约束，破坏对经典原题的机械记忆 | 在副作用更新全局最优时，依据约束增加 `if node != root` 或 `if node.left or node.right` 的条件门禁。 |

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与递归契约深度剖析</div>

#### 1. 递归契约的设计灵魂：单侧延伸贡献 (Gain) vs 全局曲折路径和 (Path Sum)
初学者最容易混淆的致命概念：
- **向父节点报告的贡献值 `gain(node)`**：父节点如果想把路径经由 `node` 串联起来，`node` 只能在其左子树或右子树中**二选一**提供一条单向下垂分支！因为路径不能存在分叉：
  $$\operatorname{gain}(node) = node.val + \max(0, \;\max(\operatorname{gain}(node.left), \;\operatorname{gain}(node.right)))$$
- **以当前节点为最高拐弯点的局部曲折路径和**：在当前节点内部，可以将左右两侧分支同时揽入怀中，形成一个以 `node` 为最高穹顶的马鞍形完整路径：
  $$\operatorname{curve\_sum}(node) = node.val + \max(0, \operatorname{gain}(node.left)) + \max(0, \operatorname{gain}(node.right))$$
- **负增益强制截断保护（核心黄金法则）**：
  若某棵子树计算出的最大贡献值小于 0，其加入只会拉低总和。必须通过 $\max(0, gain)$ 将其彻底置为 0（代表抛弃该分支）。

#### 2. 最优路径自身节点序列重构 (Path Reconstruction)
- 递归函数返回类型升级为元组：`Tuple[int, List[int]]`，分别代表 `(max_arm_gain, best_arm_path)`。
- 当前节点的最长单臂延伸路径构建：
  - 若左右单臂增益均 $\le 0$，单臂仅包含 `[node.val]`；
  - 若左单臂增益更大，单臂为 `[node.val] + left_arm`；
  - 若右单臂增益更大，单臂为 `[node.val] + right_arm`。
- 当计算以 `node` 为穹顶的最高曲折路径时：
  $$\text{full\_path} = \text{left\_arm}[::-1] + [node.val] + \text{right\_arm}$$
  若其曲折和打破全局记录，同步覆写 `best_score` 与 `best_path`！

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码（含路径重构）</div>

```python
from typing import Optional, List, Tuple

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class MaxPathSumSolution:
    """二叉树最大路径和全景方案"""

    @classmethod
    def maxPathSum(cls, root: Optional[TreeNode]) -> int:
        """LC 124: 经典单值输出，O(N) 时间，O(H) 空间"""
        max_sum = float('-inf')

        def max_gain(node: Optional[TreeNode]) -> int:
            nonlocal max_sum
            if not node:
                return 0

            # 负数增益截断为 0
            left_gain = max(0, max_gain(node.left))
            right_gain = max(0, max_gain(node.right))

            # 以当前节点为折弯顶点的总路径和
            price_newpath = node.val + left_gain + right_gain
            max_sum = max(max_sum, price_newpath)

            # 向父节点返回单侧最大贡献
            return node.val + max(left_gain, right_gain)

        max_gain(root)
        return int(max_sum)

    @classmethod
    def maxPathSumWithPath(cls, root: Optional[TreeNode]) -> Tuple[int, List[int]]:
        """变体 2: 同时输出最优分数与具体的路径节点序列"""
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

            # 计算以当前节点为顶点的弯折路径
            cur_sum = node.val + (left_gain if valid_left else 0) + (right_gain if valid_right else 0)

            if cur_sum > best_score:
                best_score = cur_sum
                # 拼接完整路径：左臂逆序 + 当前节点 + 右臂正序
                l_part = left_arm[::-1] if valid_left else []
                r_part = right_arm if valid_right else []
                best_path = l_part + [node.val] + r_part

            # 构造提供给父节点的单侧最优手臂
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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：每个节点被访问常数次，时间复杂度为严格 $\mathcal{O}(N)$。路径重构在最优更新时执行列表拼接，最坏为 $\mathcal{O}(N)$，均摊仍极其高效。
- **空间复杂度**：递归调用栈占用 $\mathcal{O}(H)$ 空间（最差退化链表为 $\mathcal{O}(N)$，平衡二叉树为 $\mathcal{O}(\log N)$）。
- **高频避坑清单**：
  1. **全负数树初始值陷阱**：`max_sum` 必须初始化为 `float('-inf')`，绝不能初始化为 `0`！若树中仅有一个节点 `[-3]`，初始为 0 会导致错误输出 0。
  2. **向父节点返回了弯折路径**：向父节点只能返回单臂 `node.val + max(left, right)`，如果返回了 `node.val + left + right`，则路径在父节点处发生二次分叉，彻底违反单链定义。

</div>

</div>
</details>

---

## 模块三：动态规划核心题组 (Dynamic Programming)

### 5. 零钱兑换与完全背包模型全景 (Coin Change 1 & 2 / Unbounded Knapsack)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">DP 05</span>
  <span class="review-card-title">零钱兑换与完全背包模型全景 (Coin Change 1 & 2 / Unbounded Knapsack)</span>
  <span class="review-card-tag">完全背包 · 0/1 背包 · 最值 vs 组合数 · 循环顺序本质 · 空间滚动压缩</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与双核变体对照</div>

给定不同面额的硬币 `coins` 和一个总金额 `amount`，硬币数量无限：
- **零钱兑换 I (LC 322 最值模型)**：计算凑成总金额所需的**最少硬币个数**。无法凑成返回 `-1`。
- **零钱兑换 II (LC 518 组合数模型)**：计算凑成总金额的**组合总数**。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

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
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：$\mathcal{O}(\text{amount} \times |\text{coins}|)$。
- **空间复杂度**：一维滚动数组压缩后为 $\mathcal{O}(\text{amount})$。

</div>

</div>
</details>

---

### 6. 贴纸拼词与状态压缩动态规划 (Stickers to Spell Word & Bitmask DP)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">DP 06</span>
  <span class="review-card-title">贴纸拼词与状态压缩动态规划 (Stickers to Spell Word & Bitmask DP)</span>
  <span class="review-card-tag">状态压缩 · 记忆化搜索 · 首个未满足字符剪枝 · 字符多重集 · O(2^n * m * n)</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 题目定义与工业考点切入</div>

我们有 $M$ 种不同类型的贴纸 `stickers`。每个贴纸上都有一个小写的英文单词。

你想要拼写出给定的字符串 `target`，方法是从贴纸中切割单个字母并重新排列它们。如果你愿意，你可以无限次复用任意类型的贴纸，每个贴纸也可以只使用其中一部分字母。

计算拼出目标字符串 `target` 所需的**最少贴纸数量**。如果任务不可能完成，返回 `-1`：

```python
def minStickers(stickers: List[str], target: str) -> int: ...
```

**关键数据规模与面试破局约束**：
- `target` 的长度 $N \in [1, 15]$（典型低十几范围，强烈暗示**状态压缩 (Bitmask)**）。
- `stickers` 的种类 $M \in [1, 50]$。
- **45 分钟编码的核心评判点**：如果不加剪枝做纯暴力搜索，分支状态空间发生阶乘级爆炸直接 TLE。必须展示出**极具说服力的剪枝依据 (Explicit Pruning Argument)**。

</div>

<div class="review-block">
<div class="review-block-label">💡 大致思路与核心剪枝机制深度剖析</div>

#### 1. 状态压缩设计 (Bitmask Representation)
- 设 $N = \operatorname{len}(target)$。我们用一个长度为 $N$ 位的二进制整数 `mask` 表示 `target` 中各位置字符的满足情况：
  - 若 `(mask >> i) & 1 == 1`：表示 `target[i]` 已经被某张贴纸中的字母满足覆盖；
  - 若 `(mask >> i) & 1 == 0`：表示 `target[i]` 尚未被覆盖。
- 初始状态：`mask = 0`（全未覆盖）；目标终止状态：`mask = (1 << N) - 1`（全被满足）。
- 状态总空间：$2^N$。当 $N = 15$ 时，$2^{15} = 32768$，状态空间极小，完全契合数组/字典记忆化。

#### 2. 致命搜索冗余与“首个未满足字符”黄金剪枝法则
- **暴搜为何必挂？**
  - 假设我们需要覆盖的集合需要贴纸 A 和贴纸 B。先选 A 再选 B，与先选 B 再选 A 达到完全相同的 `mask` 状态。如果对每个状态盲目枚举所有 $M$ 张贴纸，会生成海量排列等价树，造成巨大的分支冗余。
- **黄金剪枝原则：只分支能满足当前“首个空缺字符”的贴纸！**
  1. 对于当前未满状态 `mask`，找到**最低位的未满足位置** $k$（即 `(mask >> k) & 1 == 0` 的最小 $k$）；
  2. 此时待满足的字符为 $c = target[k]$；
  3. **强制规则**：在当前步骤中，**只尝试那些自身包含了字符 $c$ 的贴纸**！
  4. **正确性证明**：因为目标字符串中的字符 $target[k]$ 迟早必须被某一张贴纸覆盖，我们约定“谁先提供 $target[k]$ 谁就在当前层转移”，这样规定了选取贴纸的固定偏序，彻底消除了贴纸选取顺序不同带来的排列重复，但**绝对不会丢失全局最优解**！搜索树分支直接萎缩一个数量级。

#### 3. 复杂度严密理论推导
- **状态数**：$2^N$ 个不同的掩码。
- **单状态转移代价**：尝试 $M$ 种贴纸，对每张贴纸比对 $target$ 的 $N$ 个字符，位运算模拟耗时 $\mathcal{O}(N)$。
- **总体时间复杂度**：严格为 $\mathcal{O}(2^N \cdot M \cdot N)$。当 $N=15, M=50$ 时，计算步数约为 $32768 \times 50 \times 15 \approx 2.4 \times 10^7$，在 1 秒以内极速完成！
- **空间复杂度**：记忆化哈希表或数组存储 $2^N$ 个状态，空间为 $\mathcal{O}(2^N)$。

</div>

<div class="review-block">
<div class="review-block-label">💻 完整生产级实现代码</div>

```python
from typing import List
from collections import Counter

class StickersSolution:
    """
    贴纸拼词生产级状压 DP 实现：
    记忆化搜索 + 首字符剪枝优化
    时间复杂度 O(2^N * M * N)，空间复杂度 O(2^N)
    """

    @classmethod
    def minStickers(cls, stickers: List[str], target: str) -> int:
        n = len(target)
        target_chars = set(target)

        # 1. 预统计每张贴纸在 target 字符集内的有效词频（过滤无关字符）
        sticker_counts = []
        for s in stickers:
            cnt = Counter(ch for ch in s if ch in target_chars)
            if cnt:
                sticker_counts.append(cnt)

        # 记忆化缓存: mask -> min_stickers_needed
        # mask 的第 i 位为 1 表示 target[i] 已满足
        memo = { (1 << n) - 1: 0 }

        def dfs(mask: int) -> int:
            if mask in memo:
                return memo[mask]

            # 2. 找到第一个尚未被满足的位置 k
            first_unmet = 0
            while (mask >> first_unmet) & 1:
                first_unmet += 1
            target_ch = target[first_unmet]

            ans = float('inf')

            # 3. 仅枚举能够提供 target_ch 的贴纸（核心剪枝）
            for cnt in sticker_counts:
                if target_ch not in cnt:
                    continue

                # 模拟用当前贴纸尽可能多地覆盖 target 的未满位置
                avail = dict(cnt)
                nxt_mask = mask
                for i in range(n):
                    if not ((nxt_mask >> i) & 1) and target[i] in avail and avail[target[i]] > 0:
                        avail[target[i]] -= 1
                        nxt_mask |= (1 << i)

                # 若状态发生推进，继续向下递归
                if nxt_mask != mask:
                    ans = min(ans, 1 + dfs(nxt_mask))

            memo[mask] = ans
            return ans

        res = dfs(0)
        return int(res) if res != float('inf') else -1
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ 复杂度与核心避坑清单</div>

- **时间复杂度**：状态总数 $\mathcal{O}(2^N)$，每个状态最多尝试 $M$ 张贴纸并在 $N$ 位上做掩码转移，理论上界为严格 $\mathcal{O}(2^N \cdot M \cdot N)$。
- **空间复杂度**：递归栈深 $\mathcal{O}(N)$，记忆化哈希表占用 $\mathcal{O}(2^N)$。
- **高频避坑清单**：
  1. **遗漏首字符剪枝导致 TLE**：若取消 `if target_ch not in cnt: continue`，在 LeetCode 上会直接超时报错。
  2. **无解死循环防御**：如果所有贴纸合并起来都无法凑齐 `target` 中的某个字符，函数最终返回 `float('inf')`，必须被捕获并规约为 `-1`。

</div>

</div>
</details>

# -*- coding: utf-8 -*-

# -------------------------------------------------------------
# 1. Update Chinese note
# -------------------------------------------------------------
with open("notes/Leetcode/CoreSkills07 Design Graph.md", "r", encoding="utf-8") as f:
    zh = f.read()

# Insert the detailed Union-Find cycle detection section into Module 5
uf_cycle_section_zh = r"""### 并查集检测无向图环的核心原理与复杂度推导（Cycle Detection）

在无向图理论中，树与环具有如下严格的拓扑不变性：

1. **无环连通分支（树）**：包含 $k$ 个节点的连通分支恰好拥有 $k - 1$ 条边，且分支内任意两点之间**有且仅有一条简单路径**；
2. **成环的充要条件**：若两个节点 $u$ 和 $v$ 在添加边 $(u, v)$ **之前就已经处于同一个连通分量内**（即 `find(u) == find(v)`，它们之间已经存在路径 $u \leadsto v$），那么此时再加入边 $(u, v)$，这条边就会与原有的路径闭合，**形成一个环（Cycle）**！

```text
并查集加边判环状态转移过程：

场景 1：find(u) != find(v) ➔ 无环（起“桥梁”合并作用）
   Component A          Component B
     ( u )                ( v )
       \                    /
        \──[ 加入边 (u,v) ]──/   ==> 两分量合并为单一连通块，连通分量计数 count 减 1

场景 2：find(u) == find(v) ➔ 发现环！（该边为成环冗余边）
        ┌──────────────┐
        │  Component   │
        │   (u) ~~~ (v)│  <── 之前已经通过内部路径连通
        └─▲──────────▲─┘
          └──[加边 (u,v)]┘  ==> 形成闭合环路：u ~~~ v ➔ u (Cycle Detected!)
```

#### 算法标准执行流程（3 步法）

1. **初始化**：每个节点自成一个独立的集合（$parent[i] = i, size[i] = 1$）；
2. **流式遍历每条无向边 $(u, v)$**：
   - 查询节点 $u$ 和 $v$ 所在集合的根节点（代表元）：
     $$root_u = \text{find}(u), \quad root_v = \text{find}(v)$$
   - **分支 A（发生冲突，环触发）**：若 $root_u == root_v$，说明 $u$ 和 $v$ 早已连通，**边 $(u, v)$ 必定是闭环边，立刻报警或返回该边**；
   - **分支 B（无冲突，合并分量）**：若 $root_u \ne root_v$，执行 `union` 将较小集合的根挂到较大集合的根下，继续考察下一条边；
3. **结束判定**：若扫描完全部 $E$ 条边均未触发分支 A，则该无向图**严格无环**。

#### 时空复杂度严格推导

- **时间复杂度**：$\mathcal{O}(E \cdot \alpha(V)) \approx \mathcal{O}(E)$。遍历 $E$ 条边，每条边执行 2 次 `find` 和至多 1 次 `union`。在**路径压缩（Path Compression）+ 按大小/秩合并（Union by Size/Rank）**双优化下，单次操作均摊时间为 $\mathcal{O}(\alpha(V))$，其中 $\alpha$ 是**反阿克曼函数**。对于现实中所有的计算规模（即便 $V = 10^{80}$），$\alpha(V) \le 4$ 为极小常数，因此整体时间为线性 $\mathcal{O}(E)$。
- **空间复杂度**：$\mathcal{O}(V)$。仅需维护长度为 $V$ 的父节点数组 `parent` 和集合大小数组 `size`，无需显式构建邻接表或矩阵。

#### 面试必问高频避坑点：无向图 vs 有向图判环对比

| 判环算法 | 适用图类型 | 空间开销 | 核心原理 | 为什么普通并查集**不能**直接测有向图环？ |
|---|---|---|---|---|
| **并查集 (Union-Find)** | **无向图** | $\mathcal{O}(V)$ | 检测加边两端是否已处于同一连通分支 | 有向图具有**方向性**。对于“菱形汇聚”结构（如 $A \to B, A \to C, B \to D, C \to D$），该图完全无环，但并查集在处理最后一条边 $C \to D$ 时会发现 $C$ 和 $D$ 已连通，从而**错误报告有环（False Positive）**。 |
| **Kahn 算法 (入度 BFS)** | **有向图 (DAG)** | $\mathcal{O}(V + E)$ | 拓扑排序出队节点数是否等于 $\|V\|$ | 利用入度消减检测循环等待（死锁）。 |
| **三色标记法 (DFS)** | **有向图 / 无向图** | $\mathcal{O}(V + E)$ | 0=未访问, 1=回溯栈中, 2=已完成 | DFS 递归路径遇到状态 1（灰色节点）即发现后向边（Back-edge）。 |
"""

# Replace in Module 5
target_m5_pos = zh.find("常见的坑：\n\n- 忘记路径压缩")
if target_m5_pos != -1:
    zh = zh[:target_m5_pos] + uf_cycle_section_zh + "\n### 常见坑\n\n- 忘记路径压缩" + zh[target_m5_pos + len("常见的坑：\n\n- 忘记路径压缩"):]
    print("Added Union-Find Cycle Detection section to Chinese note!")

# Standardize Module headers in Chinese note
module_headers_zh = [
    ("## 模块一：图的表示与建图", "---\n\n## 模块一：图的表示与建图"),
    ("## 模块二：网格即图，Matrix DFS/BFS 模板", "---\n\n## 模块二：网格即图，Matrix DFS/BFS 模板"),
    ("## 模块三：6 道题目的映射", "---\n\n## 模块三：6 道题目的映射"),
    ("## 模块四：邻接表上的通用遍历", "---\n\n## 模块四：邻接表上的通用遍历"),
    ("## 模块五：并查集 Union-Find", "---\n\n## 模块五：并查集 Union-Find（连通分量与环检测）"),
    ("## 模块六：最小生成树，Prim 与 Kruskal", "---\n\n## 模块六：最小生成树（Prim 与 Kruskal 算法）"),
    ("## 模块七：最短路，Dijkstra 与 Bellman-Ford", "---\n\n## 模块七：最短路（Dijkstra 与 Bellman-Ford 算法）"),
    ("## 模块八：欧拉路径，Reconstruct Itinerary", "---\n\n## 模块八：欧拉路径（Hierholzer 算法与 Reconstruct Itinerary）"),
    ("## 模块十：面试前最后检查", "---\n\n## 模块十：图论终极决策图与面试自查清单"),
]

# Clean redundant --- if already present
for old_h, new_h in module_headers_zh:
    if ("---\n\n" + old_h) in zh:
        zh = zh.replace("---\n\n" + old_h, new_h)
    elif old_h in zh:
        zh = zh.replace(old_h, new_h)

with open("notes/Leetcode/CoreSkills07 Design Graph.md", "w", encoding="utf-8") as f:
    f.write(zh)
print("Updated Chinese CoreSkills07 successfully!")

# -------------------------------------------------------------
# 2. Update English note
# -------------------------------------------------------------
with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

uf_cycle_section_en = r"""### Union-Find Cycle Detection Principle & Complexity Analysis

In undirected graph theory, trees and cycles exhibit strict topological invariants:

1. **Acyclic Connected Component (Tree)**: A connected component with $k$ nodes contains exactly $k - 1$ edges, with **exactly one unique simple path** between any pair of nodes;
2. **Necessary & Sufficient Condition for Cycles**: If two nodes $u$ and $v$ **already belong to the same connected component** prior to adding edge $(u, v)$ (i.e. `find(u) == find(v)` via an existing path $u \leadsto v$), adding edge $(u, v)$ closes the loop and **forms a cycle**!

```text
Union-Find Edge Addition & Cycle Detection State Machine:

Scenario 1: find(u) != find(v) ➔ No Cycle (Acts as a Bridge)
   Component A          Component B
     ( u )                ( v )
       \                    /
        \──[ Add edge (u,v) ]──/   ==> Two components merge into one; count decrements by 1.

Scenario 2: find(u) == find(v) ➔ Cycle Detected! (Redundant Edge)
        ┌──────────────┐
        │  Component   │
        │   (u) ~~~ (v)│  <── Already connected via internal path
        └─▲──────────▲─┘
          └──[Add (u,v)]┘  ==> Closed circuit formed: u ~~~ v ➔ u (Cycle Detected!)
```

#### Standard 3-Step Execution Procedure

1. **Initialize**: Each node forms its own singleton set ($parent[i] = i, size[i] = 1$);
2. **Stream through each undirected edge $(u, v)$**:
   - Find set roots for endpoints $u$ and $v$:
     $$root_u = \text{find}(u), \quad root_v = \text{find}(v)$$
   - **Branch A (Conflict / Cycle Triggered)**: If $root_u == root_v$, $u$ and $v$ are already connected. **Edge $(u, v)$ is a cycle-forming redundant edge; alert or return immediately**;
   - **Branch B (No Conflict / Merge)**: If $root_u \ne root_v$, perform `union` by attaching the smaller tree's root under the larger tree's root;
3. **Termination**: If all $E$ edges are processed without triggering Branch A, the graph is **strictly acyclic**.

#### Rigorous Complexity Analysis

- **Time Complexity**: $\mathcal{O}(E \cdot \alpha(V)) \approx \mathcal{O}(E)$. Processing $E$ edges performs $2E$ `find` calls and at most $E$ `union` calls. With **Path Compression + Union by Size/Rank**, the amortized cost per operation is $\mathcal{O}(\alpha(V))$, where $\alpha$ is the **Inverse Ackermann Function** ($\alpha(V) \le 4$ for all realistic universe values). Total time is practically linear $\mathcal{O}(E)$.
- **Space Complexity**: $\mathcal{O}(V)$. Requires only `parent` and `size` arrays of length $V$, avoiding explicit graph adjacency representations.

#### High-Frequency Interview Comparison: Undirected vs. Directed Graph Cycle Detection

| Algorithm | Applicable Graphs | Space Overhead | Core Invariant | Why can't standard Union-Find detect directed cycles? |
|---|---|---|---|---|
| **Union-Find** | **Undirected Graphs** | $\mathcal{O}(V)$ | Checks if endpoints belong to the same component | Directed graphs possess **edge directionality**. In a diamond convergence DAG ($A \to B, A \to C, B \to D, C \to D$), there is no cycle, but undirected DSU would treat $C$ and $D$ as already connected when adding $C \to D$, producing a **False Positive**! |
| **Kahn's BFS** | **Directed Graphs (DAG)** | $\mathcal{O}(V + E)$ | Checks if processed count equals $\|V\|$ | Detects circular wait (deadlock) via in-degree reduction. |
| **Three-Color DFS** | **Directed / Undirected** | $\mathcal{O}(V + E)$ | 0=Unvisited, 1=Visiting (Stack), 2=Visited | Detects back-edges to nodes currently in the recursion stack (color 1). |
"""

target_m5_pos_en = en.find("Common traps:\n\n- Forgetting path compression")
if target_m5_pos_en != -1:
    en = en[:target_m5_pos_en] + uf_cycle_section_en + "\n### Common Pitfalls\n\n- Forgetting path compression" + en[target_m5_pos_en + len("Common traps:\n\n- Forgetting path compression"):]
    print("Added Union-Find Cycle Detection section to English note!")

# Standardize Module headers in English note
module_headers_en = [
    ("## Module 1: Graph Representation and Construction", "---\n\n## Module 1: Graph Representation and Construction"),
    ("## Module 2: Grids as Graphs, Matrix DFS/BFS Templates", "---\n\n## Module 2: Grids as Graphs, Matrix DFS/BFS Templates"),
    ("## Module 3: Mapping Six Core Problems", "---\n\n## Module 3: Mapping Six Core Problems"),
    ("## Module 4: General Traversals on Adjacency Lists", "---\n\n## Module 4: General Traversals on Adjacency Lists"),
    ("## Module 5: Disjoint Set Union (Union-Find)", "---\n\n## Module 5: Disjoint Set Union (Union-Find & Cycle Detection)"),
    ("## Module 6: Minimum Spanning Trees, Prim and Kruskal", "---\n\n## Module 6: Minimum Spanning Trees (Prim & Kruskal Algorithms)"),
    ("## Module 7: Shortest Paths, Dijkstra and Bellman-Ford", "---\n\n## Module 7: Shortest Paths (Dijkstra & Bellman-Ford Algorithms)"),
    ("## Module 8: Eulerian Paths, Reconstruct Itinerary", "---\n\n## Module 8: Eulerian Paths (Hierholzer's Algorithm & Reconstruct Itinerary)"),
    ("## Module 10: Final Pre-Interview Checklist", "---\n\n## Module 10: Final Pre-Interview Checklist & Decision Map"),
]

for old_h, new_h in module_headers_en:
    if ("---\n\n" + old_h) in en:
        en = en.replace("---\n\n" + old_h, new_h)
    elif old_h in en:
        en = en.replace(old_h, new_h)

with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
    f.write(en)
print("Updated English CoreSkills07 successfully!")


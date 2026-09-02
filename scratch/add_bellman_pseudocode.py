# -*- coding: utf-8 -*-

# 1. Update Chinese note
with open("notes/Leetcode/CoreSkills07 Design Graph.md", "r", encoding="utf-8") as f:
    zh = f.read()

bellman_section_zh = r"""### Bellman-Ford 算法回顾与标准伪代码

Bellman-Ford 是基于**动态规划（DP）**思想的单源最短路算法。

#### 1. 核心数学原理与状态转移
- **状态定义**：$dist^{(k)}[v]$ 表示从源点 $src$ 出发，**最多经过 $k$ 条边**到达顶点 $v$ 的最短路径长度。
- **状态转移方程（边松弛 Relaxation）**：
  $$dist^{(k)}[v] = \min \left( dist^{(k-1)}[v], \min_{(u, v) \in E} \left( dist^{(k-1)}[u] + w(u, v) \right) \right)$$
- **收敛性定理**：在一个包含 $V$ 个顶点且**无负权回路（Negative Cycle）**的图中，任意两点间的最短简单路径最多只包含 $V - 1$ 条边。因此只需执行 $V - 1$ 轮全边松弛，所有节点的最短路必然收敛！

```text
Bellman-Ford 标准版伪代码 (O(V · E) 时间, 支持负权边与负权环检测):
--------------------------------------------------------------------------------
def bellman_ford(V, edges, src):
    # 1. 初始化距离表
    dist = [float("inf")] * V
    dist[src] = 0

    # 2. 进行 V - 1 轮松弛 (每轮遍历所有边)
    for i in range(V - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:  # 提前收敛优化
            break

    # 3. 第 V 轮检测负权环: 若仍能松弛变小，则说明存在负权回路！
    for u, v, w in edges:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            raise ValueError("图中存在可达的负权回路 (Negative Cycle Detected!)")

    return dist
```

```text
限制最多经过 K 条边的 Bellman-Ford 伪代码 (Cheapest Flights 核心):
--------------------------------------------------------------------------------
def bellman_ford_k_edges(V, edges, src, k_edges):
    dist = [float("inf")] * V
    dist[src] = 0

    # 进行恰好 k_edges 轮松弛
    for _ in range(k_edges):
        next_dist = dist.copy()  # 必须严格从上一轮状态读取，禁止本轮内部串联！
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < next_dist[v]:
                next_dist[v] = dist[u] + w
        dist = next_dist

    return dist
```

#### 最短路三大核心算法深度对比

| 算法 | 核心机制 | 适用边权 | 显式限制步数 | 时间复杂度 | 空间复杂度 | 工业界与面试定位 |
|---|---|---|---|---|---|---|
| **Dijkstra** | 贪心 + 最小堆优先队列 | **仅非负权** | 否（需扩充状态） | $\mathcal{O}(E \log V)$ | $\mathcal{O}(V + E)$ | **单源非负权最短路绝对主力** |
| **Bellman-Ford** | 动态规划 + $V-1$ 轮全边松弛 | **支持负权边** | **天然支持（$K$ 轮松弛）** | $\mathcal{O}(V \cdot E)$ | $\mathcal{O}(V)$ | **带步数约束 / 负权边 / 判负环** |
| **SPFA (队列优化 BF)** | 队列记录发生松弛的节点 | **支持负权边** | 否 | 平均 $\mathcal{O}(E)$，最坏 $\mathcal{O}(V \cdot E)$ | $\mathcal{O}(V)$ | 稀疏图常数级优化（卡常时慎用） |
"""

target_zh = "### Dijkstra 回顾"
target_zh_end = "状态里不需要像 Cheapest Flights 那样额外记录用了几条边。"

if target_zh in zh:
    # insert before ### 13. Network Delay Time
    pos_13 = zh.find("### 13. Network Delay Time")
    zh = zh[:pos_13] + bellman_section_zh + "\n" + zh[pos_13:]
    with open("notes/Leetcode/CoreSkills07 Design Graph.md", "w", encoding="utf-8") as f:
        f.write(zh)
    print("Added Bellman-Ford Pseudocode to Chinese note!")

# 2. Update English note
with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

bellman_section_en = r"""### Bellman-Ford Algorithm Review & Standard Pseudocode

Bellman-Ford is a single-source shortest path algorithm founded on **Dynamic Programming (DP)** principles.

#### 1. Mathematical Invariant & State Transitions
- **State Definition**: $dist^{(k)}[v]$ represents the shortest distance from source $src$ to vertex $v$ using **at most $k$ edges**.
- **State Transition Equation (Edge Relaxation)**:
  $$dist^{(k)}[v] = \min \left( dist^{(k-1)}[v], \min_{(u, v) \in E} \left( dist^{(k-1)}[u] + w(u, v) \right) \right)$$
- **Convergence Theorem**: In a graph with $V$ vertices and **no negative-weight cycles**, any simple shortest path contains at most $V - 1$ edges. Thus, performing $V - 1$ rounds of all-edge relaxation guarantees global convergence!

```text
Standard Bellman-Ford Pseudocode (O(V · E) Time, Supports Negative Weights & Negative Cycle Detection):
--------------------------------------------------------------------------------
def bellman_ford(V, edges, src):
    # 1. Initialize distance array
    dist = [float("inf")] * V
    dist[src] = 0

    # 2. Perform V - 1 relaxation rounds (iterate all edges each round)
    for i in range(V - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                updated = True
        if not updated:  # Early termination optimization
            break

    # 3. Round V check for negative cycles: if relaxation still happens, a negative cycle exists!
    for u, v, w in edges:
        if dist[u] != float("inf") and dist[u] + w < dist[v]:
            raise ValueError("Graph contains a reachable negative weight cycle!")

    return dist
```

```text
Bounded-Step Bellman-Ford Pseudocode (Cheapest Flights Within K Stops):
--------------------------------------------------------------------------------
def bellman_ford_k_edges(V, edges, src, k_edges):
    dist = [float("inf")] * V
    dist[src] = 0

    # Perform exactly k_edges relaxation passes
    for _ in range(k_edges):
        next_dist = dist.copy()  # Strictly read from previous round to prevent intra-round chaining!
        for u, v, w in edges:
            if dist[u] != float("inf") and dist[u] + w < next_dist[v]:
                next_dist[v] = dist[u] + w
        dist = next_dist

    return dist
```

#### Shortest Path Algorithms Comparison Matrix

| Algorithm | Paradigm | Edge Weights | Step / Hop Constraints | Time Complexity | Space Complexity | Practical / Interview Application |
|---|---|---|---|---|---|---|
| **Dijkstra** | Greedy + Min-Heap Priority Queue | **Non-negative only** | No (requires state expansion) | $\mathcal{O}(E \log V)$ | $\mathcal{O}(V + E)$ | **Standard choice for non-negative SSSP** |
| **Bellman-Ford** | Dynamic Programming ($V-1$ rounds) | **Supports negative edges** | **Native support ($K$ rounds)** | $\mathcal{O}(V \cdot E)$ | $\mathcal{O}(V)$ | **Step limits / Negative weights / Negative cycle detection** |
| **SPFA (Queue-Optimized BF)** | Queue-driven relaxation tracker | **Supports negative edges** | No | Avg $\mathcal{O}(E)$, Worst $\mathcal{O}(V \cdot E)$ | $\mathcal{O}(V)$ | Sparse graph constant-factor optimization |
"""

if "### 13. Network Delay Time" in en:
    pos_13_en = en.find("### 13. Network Delay Time")
    en = en[:pos_13_en] + bellman_section_en + "\n" + en[pos_13_en:]
    with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Added Bellman-Ford Pseudocode to English note!")


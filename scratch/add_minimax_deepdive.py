# -*- coding: utf-8 -*-

# 1. Update Chinese note
with open("notes/Leetcode/CoreSkills07 Design Graph.md", "r", encoding="utf-8") as f:
    zh = f.read()

minimax_zh = r"""### 14. Swim in Rising Water（水位上升的泳池中游泳 · LeetCode 778）

#### 题目描述与 Minimax 本质
题目给出 `n x n` 的高程网格 `grid`。水从左上角 `(0, 0)` 开始随时间 $t$ 线性上涨，在时刻 $t$，你只能游进高程不超过 $t$ 的相邻格子（即 $t \ge \text{grid}[r][c]$）。
一条从起点到终点 `(n-1, n-1)` 的路径所耗费的最早到达时间，等于该路径上所有格子高程的**最大值（路径瓶颈）**。
目标是：在所有连通起点与终点的可行路径中，找出耗时最小（即路径最大值最小）的那条路径。这是一个经典的 **Minimax Path（最小化最大值 / 最小瓶颈路）** 问题。

#### 专题透视：Minimax / Maximin 路径问题三大解法全景

在算法竞赛与大厂高频面试中，形如“求最大值的最小值（Minimax）”或“求最小值的最大值（Maximin / 瓶颈容量）”的图论问题层出不穷：

```text
Minimax / Maximin 数学形式对比:
  1. 普通最短路 (Sum-Metric):      min_{p ∈ P} ∑_{e ∈ p} weight(e)
  2. 最小瓶颈路 (Minimax Path):    min_{p ∈ P} max_{e ∈ p} weight(e)   <── (如 LeetCode 778 / 1631)
  3. 最大瓶颈路 (Maximin / 宽度):  max_{p ∈ P} min_{e ∈ p} capacity(e) <── (如 LeetCode 1102 / 2812)
```

这类问题在图论中拥有三大通用黄金解法（面试三板斧）：

| 解法体系 | 核心机制 | 时间复杂度 | 空间复杂度 | 适用场景与优缺点 |
|---|---|---|---|---|
| **方法 1：改写 Dijkstra 堆贪心（本节推荐）** | 将松弛操作由 $d + w$ 改为 $\max(d, w)$。最小堆每次弹出当前瓶颈最小的状态。 | $\mathcal{O}(N^2 \log N)$ | $\mathcal{O}(N^2)$ | **单次查询最直接高效**；支持浮点与任意实数，首次弹出终点即为全局最优解。 |
| **方法 2：二分答案 + BFS/DFS 连通性验证** | 二分猜一个瓶颈值 $mid$（范围 $[0, \max]$），把网格中 $\le mid$ 的格子视为可走通路，用 BFS/DFS 检查起点能否走到终点。 | $\mathcal{O}(N^2 \log(\text{Max} - \text{Min}))$ | $\mathcal{O}(N^2)$ | **通用降维打击法**；单调性极强时代码极不易出错，容易泛化到复杂多重约束。 |
| **方法 3：Kruskal 最小生成树 / 并查集动态加边** | 将所有相邻单元格的边（权值为 $\max(u, v)$）按权值升序排序，从小到大执行 `union`。当起点与终点**首次连通**时，当前加入的边权即为答案！ | $\mathcal{O}(N^2 \log N)$ | $\mathcal{O}(N^2)$ | **数学结构最优雅**；最小生成树（MST）在树上的唯一路径天然就是瓶颈最短路。 |

#### 为什么 Dijkstra 贪心在 Minimax 下依然 100% 成立？
传统 Dijkstra 依赖于路径权值的**非负累加单调性**（$d(u) + w \ge d(u)$）。在 Minimax 问题中：
$$\text{new\_bottleneck} = \max(\text{bottleneck}(u), \text{grid}[v]) \ge \text{bottleneck}(u)$$
该松弛算子同样具备**严格的单调不减性（Monotonicity）**！
当状态 $(bottleneck, r, c)$ 第一次从最小堆中弹出时，它记录的瓶颈值已经是全局所有可能到达 $(r, c)$ 的路径中的最小瓶颈。任何未来从堆中弹出的更大瓶颈状态，后续扩展无论怎么取 $\max$，瓶颈值都绝不可能小于当前值。因此**堆贪心最优子结构严格成立**！

#### 高频同类题谱推荐
1. **LeetCode 1631. Path With Minimum Effort（最小体力消耗路径）**：边权为相邻格子绝对高度差 $|\Delta h|$ 的 Minimax 问题；
2. **LeetCode 1102. Path With Maximum Minimum Value（得分最高的路径）**：求路径最小值的最大化（Maximin），用最大堆贪心；
3. **LeetCode 2812. Find the Safest Path in a Grid（找出安全度最大的路径）**：多源 BFS 计算曼哈顿距离场 + Maximin 堆贪心。"""

target_zh = "### 14. Swim in Rising Water"
target_zh_end = "这一节以堆的写法为主。"

pos_14 = zh.find(target_zh)
pos_14_end = zh.find(target_zh_end, pos_14)

if pos_14 != -1 and pos_14_end != -1:
    zh = zh[:pos_14] + minimax_zh + zh[pos_14_end + len(target_zh_end):]
    with open("notes/Leetcode/CoreSkills07 Design Graph.md", "w", encoding="utf-8") as f:
        f.write(zh)
    print("Enhanced Chinese Minimax Section!")
else:
    print("Failed to find Chinese boundaries")

# 2. Update English note
with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

minimax_en = r"""### 14. Swim in Rising Water (LeetCode 778)

#### Problem Formulation & Minimax Essence
You are given an `n x n` integer matrix `grid` where each cell represents the elevation at that point. Rain starts falling at time $t = 0$. At time $t$, you can swim to any 4-directionally adjacent cell if and only if the elevation of that cell is at most $t$ (i.e. $t \ge \text{grid}[r][c]$).
The time required to traverse a path from `(0, 0)` to `(n-1, n-1)` equals the **maximum elevation encountered along that path (the path bottleneck)**.
The goal is to find the minimum time needed to reach the bottom-right cell among all valid paths. This is a classic **Minimax Path (Minimum Bottleneck Path)** problem.

#### Deep Dive: The 3 Universal Paradigms for Minimax / Maximin Path Problems

In competitive programming and technical interviews, problems asking for "minimizing the maximum value (Minimax)" or "maximizing the minimum capacity (Maximin / Widest Path)" are extremely common:

```text
Mathematical Formulations:
  1. Standard Shortest Path (Sum-Metric):  min_{p ∈ P} ∑_{e ∈ p} weight(e)
  2. Minimax Path (Minimum Bottleneck):     min_{p ∈ P} max_{e ∈ p} weight(e)   <── (e.g. LeetCode 778 / 1631)
  3. Maximin Path (Maximum Bottleneck):     max_{p ∈ P} min_{e ∈ p} capacity(e) <── (e.g. LeetCode 1102 / 2812)
```

There are three universal algorithmic paradigms to solve this class of problems:

| Algorithmic Paradigm | Core Mechanism | Time Complexity | Space Complexity | Practical Pros & Cons |
|---|---|---|---|---|
| **Method 1: Modified Dijkstra (Min-Heap Greedy)** | Replace the additive relaxation $d + w$ with $\max(d, w)$. The min-heap always pops the state with the minimum bottleneck so far. | $\mathcal{O}(N^2 \log N)$ | $\mathcal{O}(N^2)$ | **Most direct and fast for single query**; handles continuous/discrete weights seamlessly; first pop of destination is globally optimal. |
| **Method 2: Binary Search + BFS/DFS Connectivity** | Binary search the bottleneck threshold $mid \in [0, \max]$. Treat cells with $\text{grid}[r][c] \le mid$ as passable, then check start-to-end reachability via BFS/DFS. | $\mathcal{O}(N^2 \log(\text{Max} - \text{Min}))$ | $\mathcal{O}(N^2)$ | **Highly versatile reduction**; robust against edge cases; naturally generalizes to complex multi-constraint scenarios. |
| **Method 3: Kruskal's MST / Disjoint Set Union (DSU)** | Sort all adjacent cell-to-cell transitions (with weight $\max(u, v)$) in ascending order. Incrementally `union` endpoints. When source and destination **first become connected**, that edge's weight is the exact Minimax answer! | $\mathcal{O}(N^2 \log N)$ | $\mathcal{O}(N^2)$ | **Mathematically elegant**; the unique tree path in a Minimum Spanning Tree is inherently the minimum bottleneck path. |

#### Why Dijkstra's Greedy Choice Remains 100% Valid for Minimax
Standard Dijkstra relies on the **non-negative additive monotonicity** ($d(u) + w \ge d(u)$). Under the Minimax metric:
$$\text{new\_bottleneck} = \max(\text{bottleneck}(u), \text{grid}[v]) \ge \text{bottleneck}(u)$$
This relaxation operator satisfies **strict monotonicity**!
When $(bottleneck, r, c)$ is first popped from the min-heap, its recorded bottleneck is guaranteed to be the minimum possible among all paths connecting the source to $(r, c)$. Any future path expanded from a larger bottleneck state will always yield $\ge \text{larger bottleneck}$ under the $\max$ operation. Thus, **optimal substructure and greedy choice hold rigorously**!

#### Related High-Frequency Problems
1. **LeetCode 1631. Path With Minimum Effort**: Minimax path where edge weights are absolute height differences $|\Delta h|$;
2. **LeetCode 1102. Path With Maximum Minimum Value**: Maximin path on grid values using max-heap greedy;
3. **LeetCode 2812. Find the Safest Path in a Grid**: Multi-source BFS distance field + Maximin heap/binary search."""

target_en = "### 14. Swim in Rising Water"
target_en_end = "this section focuses on the heap approach."
if target_en_end not in en:
    target_en_end = "this section focuses on the heap-based implementation."

pos_14_en = en.find(target_en)
pos_14_en_end = en.find(target_en_end, pos_14_en)
if pos_14_en_end == -1:
    pos_14_en_end = en.find("| Item |", pos_14_en)

if pos_14_en != -1 and pos_14_en_end != -1:
    en = en[:pos_14_en] + minimax_en + "\n\n" + en[pos_14_en_end:]
    with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Enhanced English Minimax Section!")
else:
    print(f"Failed to find English boundaries: pos_14={pos_14_en}, pos_14_en_end={pos_14_en_end}")


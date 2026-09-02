# -*- coding: utf-8 -*-

# 1. Update Chinese note
with open("notes/Leetcode/CoreSkills07 Design Graph.md", "r", encoding="utf-8") as f:
    zh = f.read()

zh_target = r"""## 模块五：并查集 Union-Find（连通分量与环检测）

并查集(Disjoint Set Union，也叫 Union-Find)维护一组不相交的集合，支持两个操作：`find(x)` 返回 `x` 所在集合的代表元(根)，`union(a, b)` 把 `a`、`b` 所在的两个集合合并成一个。放到图问题里，"集合"对应一个连通分量：对每条边执行一次 `union`，最终共享同一个根的节点就在同一个连通分量里，不需要显式建邻接表，也不需要遍历。"""

zh_replacement = r"""## 模块五：并查集 Union-Find（连通分量与环检测）

> **🎯 并查集两大黄金应用场景**：
> 1. **动态维护连通分量**：每加入一条边执行一次 `union(u, v)`，最终共享同一个根的节点就在同一个连通分量里，不需要显式建邻接表，也不需要 BFS/DFS 遍历。
> 2. **无向图成环检测（Cycle Detection）**：若在加边 $(u, v)$ 时发现 `find(u) == find(v)`，说明两点此前早已连通，**该边加入后必定与已有路径闭合形成环（Cycle）**！这是 Kruskal 最小生成树与 LeetCode 684 冗余连接的第一核心判定准则。

并查集（Disjoint Set Union，也叫 Union-Find）维护一组不相交的集合，支持两个基础操作：
- `find(x)`：返回 $x$ 所在集合的代表元（根节点），结合路径压缩均摊时间仅 $\mathcal{O}(\alpha(n)) \approx \mathcal{O}(1)$；
- `union(a, b)`：合并 $a$ 和 $b$ 所在的两个集合；**若两者已处于同一集合则返回 `False`（成环报警！）**。"""

if zh_target in zh:
    zh = zh.replace(zh_target, zh_replacement)
    with open("notes/Leetcode/CoreSkills07 Design Graph.md", "w", encoding="utf-8") as f:
        f.write(zh)
    print("Enhanced Chinese Module 5 Intro!")
else:
    print("Chinese target not found")

# 2. Update English note
with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

en_target = r"""## Module 5: Disjoint Set Union (Union-Find & Cycle Detection)

Disjoint Set Union (DSU, also called Union-Find) maintains a collection of disjoint sets, supporting two operations: `find(x)` returns the representative (root) of the set containing `x`, and `union(a, b)` merges the sets containing `a` and `b`. In graph problems, a "set" corresponds to a connected component: by performing a `union` for every edge, nodes sharing the same root end up in the same connected component, without the need to build an explicit adjacency list or perform traversals."""

en_replacement = r"""## Module 5: Disjoint Set Union (Union-Find & Cycle Detection)

> **🎯 Two Primary Pillars of Union-Find**:
> 1. **Dynamic Connected Components**: Performing `union(u, v)` for each edge dynamically maintains connectivity clusters without building adjacency lists or running BFS/DFS.
> 2. **Undirected Cycle Detection**: If `find(u) == find(v)` when adding edge $(u, v)$, both endpoints are already connected. **Adding this edge strictly forms a closed cycle**! This is the foundational criterion for Kruskal's MST and Redundant Connection.

Disjoint Set Union (DSU, also called Union-Find) maintains a collection of disjoint sets, supporting two core operations:
- `find(x)`: Returns the representative (root) of the set containing $x$, amortized to $\mathcal{O}(\alpha(n)) \approx \mathcal{O}(1)$ with path compression;
- `union(a, b)`: Merges sets containing $a$ and $b$; **returns `False` if $a$ and $b$ already share the same root (Cycle Alarm!)**."""

if en_target in en:
    en = en.replace(en_target, en_replacement)
    with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Enhanced English Module 5 Intro!")
else:
    print("English target not found")


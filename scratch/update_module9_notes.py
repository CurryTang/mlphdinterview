# -*- coding: utf-8 -*-

# 1. Update Chinese note
with open("notes/Leetcode/CoreSkills07 Design Graph.md", "r", encoding="utf-8") as f:
    zh = f.read()

zh_target = r"""</details>
## 模块九：拓扑排序"""

zh_replacement = r"""</details>

---

## 模块九：拓扑排序（Topological Sort）

> **核心定义**：拓扑排序（Topological Sort）是将**有向无环图（DAG）**的所有顶点排成一个线性序列，使得图中任意一条有向边 $(u, v)$，顶点 $u$ 在序列中均出现在 $v$ 之前。
>
> **解题口诀**：**一数入度、二入零度、三砍后继、四比数量**。

```topo-demo
```"""

if zh_target in zh:
    zh = zh.replace(zh_target, zh_replacement)
    with open("notes/Leetcode/CoreSkills07 Design Graph.md", "w", encoding="utf-8") as f:
        f.write(zh)
    print("Updated Chinese CoreSkills07 Module 9 header and demo block!")
else:
    print("Chinese target not found")

# 2. Update English note
with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

en_target = r"""</details>
## Module 9: Topological Sort"""

en_replacement = r"""</details>

---

## Module 9: Topological Sort (DAG, Kahn's Algorithm & Cycle Detection)

> **Core Definition**: A topological sort of a **Directed Acyclic Graph (DAG)** is a linear ordering of its vertices such that for every directed edge $(u, v)$, vertex $u$ appears before $v$ in the ordering.
>
> **4-Step Mental Model**: **1. Count Indegrees -> 2. Enqueue In-0 Nodes -> 3. Relax Successors -> 4. Check Node Count (|V|)**.

```topo-demo
```"""

if en_target in en:
    en = en.replace(en_target, en_replacement)
    with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Updated English CoreSkills07 Module 9 header and demo block!")
else:
    print("English target not found")


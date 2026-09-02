# -*- coding: utf-8 -*-

# 1. Update Chinese note
with open("notes/Leetcode/CoreSkills07 Design Graph.md", "r", encoding="utf-8") as f:
    zh = f.read()

euler_section_zh = r"""## 模块八：欧拉路径（Hierholzer 算法与 Reconstruct Itinerary）

### 16. Reconstruct Itinerary（重建行程 · LeetCode 332）

#### 题目详细描述
给定一份航班机票列表 `tickets`，其中 `tickets[i] = [from_i, to_i]` 表示一张从机场 `from_i` 飞往 `to_i` 的单程机票。请你按照机票信息重建并输出这份飞行行程。

**核心约束与规则**：
1. **强制固定起点**：所有行程必须从肯尼迪国际机场 `"JFK"` 出发；
2. **一票恰好用一次（边完全覆盖）**：所有机票必须**恰好且全部用掉一次**（若存在完全重复的多张机票，每张都必须作为一条独立的边被飞行一次）；
3. **允许机场重复进出（节点可重复）**：同一个机场可以被多次起降访问（入度、出度均可 $> 1$）；
4. **字典序最小贪心规则（Lexicographical Tie-breaker）**：如果存在多条能完整用完全部机票的合法行程，必须返回**字符字典序最小**的那一条（例如 `["JFK", "ATL", "JFK"]` 字典序优于 `["JFK", "SFO", "JFK"]`）；
5. **合法性保证**：题目保证输入的机票组合至少存在一种合法的飞行行程。

```text
欧拉路径图示与死胡同陷阱 (示例: tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]])
                ┌────────────────┐
                │                ▼
             [JFK] (起点) ⇄ [NRT] (子回路)
                │
                ▼ (死胡同分支: 出度为 0)
              [KUL] (终点)

★ 贪心陷阱: JFK 的邻居有 'KUL' 和 'NRT'。若直接前序贪心选择字典序较小的 'KUL'，飞入 KUL 后由于无出边陷入死胡同，导致 NRT 回路无法被用掉！
★ Hierholzer 破局法: 走到尽头才入栈（后序遍历）。KUL 率先走到尽头先入栈，回溯后走完 NRT 回路再入栈 JFK，最终反转得到正确序列：["JFK", "NRT", "JFK", "KUL"]！
```

#### 图论本质剖析：欧拉路径 vs 哈密顿路径

前面的图论题目大多要求“访问每个节点恰好一次”（如拓扑排序、岛屿遍历、二叉树遍历），这是经典的**点覆盖**逻辑。而 Reconstruct Itinerary 要求的是“用掉每一张机票恰好一次”，机票对应图中的有向边，这属于经典的**欧拉路径（Eulerian Path）**：

| 对比维度 | 欧拉路径 / 欧拉回路（Eulerian Path / Circuit） | 哈密顿路径 / 哈密顿回路（Hamiltonian Path） |
|---|---|---|
| **核心定义** | 遍历图中的**每一条边恰好一次**，节点可重复访问 | 遍历图中的**每一个顶点恰好一次**，边不一定全用 |
| **存在性充要条件** | **有向图**：至多一个点出度比入度多 1（起点），至多一个点入度比出度多 1（终点），其余点出度等于入度 | **NP-Hard**，无简易充要条件 |
| **经典实际问题** | 柯尼斯堡七桥问题、一笔画问题、机票全部核销（LeetCode 332） | 旅行商问题（TSP）、骑士周游问题 |
| **求解算法与复杂度** | **Hierholzer 算法：$\mathcal{O}(E \log E)$**（线性多项式时间） | 指数级回溯搜索：$\mathcal{O}(2^V \cdot V^2)$（NP 难） |

#### Hierholzer 算法核心原理：后序遍历与死胡同缝合

1. **邻接表数据结构**：每个出发机场维护一个目的地的**最小堆（Min-Heap）**，保证每次贪心选出的都是字典序最小的候选目的地；
2. **边走边删边**：DFS 每次从堆顶弹出一条出边并递归深入（即消耗掉这张机票）；
3. **后序入栈（Post-order Recording）**：
   - 当某个机场的堆耗尽（当前分支已经没有未使用的机票可走）时，才把该机场追加进 `route` 列表；
   - 走到死胡同的终点机场（如 `KUL`）会最先结束递归并最先进入 `route`；
   - 递归回溯时，外层机场会继续调用其它未走完的子回路，待所有子回路走完后才将自身入栈；
4. **终点反转**：DFS 结束后 `route` 中的顺序是“逆向记录”（终点在最前、起点在最后），将 `route` 整体翻转（`route[::-1]`）即得到完美拼接了所有子回路的合法行程！"""

target_start_zh = "## 模块八：欧拉路径（Hierholzer 算法与 Reconstruct Itinerary）"
target_end_zh = "### Hierholzer 算法"

pos_start = zh.find(target_start_zh)
pos_end = zh.find(target_end_zh)

if pos_start != -1 and pos_end != -1:
    zh = zh[:pos_start] + euler_section_zh + "\n\n" + zh[pos_end:]
    with open("notes/Leetcode/CoreSkills07 Design Graph.md", "w", encoding="utf-8") as f:
        f.write(zh)
    print("Enhanced Chinese Eulerian Path section in CoreSkills07.md!")
else:
    print("Failed to find boundaries in Chinese note")

# 2. Update English note
with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

euler_section_en = r"""## Module 8: Eulerian Paths (Hierholzer's Algorithm & Reconstruct Itinerary)

### 16. Reconstruct Itinerary (LeetCode 332)

#### Detailed Problem Description
You are given a list of airline tickets `tickets` where `tickets[i] = [from_i, to_i]` represents the departure and arrival airports of a one-way flight. Reconstruct the itinerary in order and return it.

**Core Rules & Invariants**:
1. **Mandatory Departure**: All itineraries must begin from `"JFK"`;
2. **Every Ticket Used Exactly Once (Full Edge Coverage)**: You must use all the tickets **exactly once** (if duplicate tickets exist, each ticket represents a distinct directed edge and must be flown separately);
3. **Nodes Can Be Re-visited**: Airports can be visited multiple times (in-degrees and out-degrees can be $> 1$);
4. **Lexicographically Smallest Order**: If there are multiple valid itineraries, return the itinerary that has the **smallest lexical order** when read as a single string (e.g., `["JFK", "ATL", "JFK"]` has a smaller lexical order than `["JFK", "SFO", "JFK"]`);
5. **Guarantee of Solution**: You may assume all inputs have at least one valid itinerary.

```text
Eulerian Path & Dead-End Traps (Example: tickets = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]])
                ┌────────────────┐
                │                ▼
             [JFK] (Start) ⇄ [NRT] (Sub-circuit)
                │
                ▼ (Dead-end branch: out-degree = 0)
              [KUL] (Destination)

★ Greedy Trap: JFK has destinations 'KUL' and 'NRT'. If we greedily take smaller 'KUL' in preorder, we get trapped in KUL (no outgoing edges), leaving the NRT circuit stranded!
★ Hierholzer Resolution: Post-order recording. KUL hits the dead end first and gets pushed to route first. Backtracking traverses the remaining NRT circuit and pushes JFK last. Reversing gives the correct path: ["JFK", "NRT", "JFK", "KUL"]!
```

#### Graph Theoretical Formulation: Eulerian Path vs. Hamiltonian Path

Most graph problems (e.g. topological sort, island traversal, tree search) focus on visiting each **vertex** once (vertex coverage). Reconstruct Itinerary requires using every **ticket (edge)** once, which is an **Eulerian Path** problem:

| Dimension | Eulerian Path / Circuit | Hamiltonian Path / Cycle |
|---|---|---|
| **Core Definition** | Traverses **every edge exactly once** (vertices can be revisited) | Traverses **every vertex exactly once** (edges may be skipped) |
| **Existence Condition** | **Directed Graph**: At most one vertex with $\text{out} - \text{in} = 1$ (start), at most one with $\text{in} - \text{out} = 1$ (end), all others $\text{out} == \text{in}$ | **NP-Hard** (no polynomial necessary & sufficient condition) |
| **Classic Problems** | Seven Bridges of Königsberg, Line Tracing, Ticket Itinerary (LeetCode 332) | Travelling Salesperson Problem (TSP), Knight's Tour |
| **Algorithm & Complexity** | **Hierholzer's Algorithm: $\mathcal{O}(E \log E)$** (polynomial time) | Exponential backtracking / DP: $\mathcal{O}(2^V \cdot V^2)$ (NP-Hard) |

#### Hierholzer's Algorithm Mechanics: Post-order DFS & Sub-circuit Splicing

1. **Adjacency Representation**: Store destinations for each airport in a **Min-Heap**, ensuring greedy selection of lexicographically smallest candidate destinations;
2. **In-place Edge Removal**: Each DFS step pops the smallest edge from the heap (consuming the ticket);
3. **Post-order Appending**:
   - Append an airport to `route` only when its heap is exhausted (no more tickets left from this airport);
   - Dead-end destination airports (such as `KUL`) exhaust their tickets first and get appended first;
   - As recursion unwinds, outer airports traverse any remaining sub-circuits before appending themselves;
4. **Final Reverse**: Reversing the post-order sequence `route[::-1]` yields the complete, correctly spliced itinerary starting from `"JFK"`.

### Hierholzer Algorithm Implementation"""

target_start_en = "## Module 8: Eulerian Paths (Hierholzer's Algorithm & Reconstruct Itinerary)"
target_end_en = "### Hierholzer's Algorithm"
if target_end_en not in en:
    target_end_en = "### Hierholzer Algorithm"

pos_start_en = en.find(target_start_en)
pos_end_en = en.find("### Hierholzer", pos_start_en)

if pos_start_en != -1 and pos_end_en != -1:
    en = en[:pos_start_en] + euler_section_en + "\n\n" + en[pos_end_en + len("### Hierholzer Algorithm Implementation") if en[pos_end_en:].startswith("### Hierholzer Algorithm Implementation") else pos_end_en:]
    with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Enhanced English Eulerian Path section in CoreSkills07.en.md!")
else:
    print("Failed to find boundaries in English note")


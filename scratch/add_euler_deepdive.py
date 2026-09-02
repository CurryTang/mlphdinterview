# -*- coding: utf-8 -*-

# 1. Update Chinese note
with open("notes/Leetcode/CoreSkills07 Design Graph.md", "r", encoding="utf-8") as f:
    zh = f.read()

deepdive_zh = r"""#### 深度剖析：为什么“后序入栈 + 逆序翻转”百分之百正确？

很多初学者容易产生疑问：*为什么不能一进入机场就记录（前序），而必须等出边耗尽在后序位置才追加进 `route` 并最后反转？*

##### 1. 核心矛盾：前序遍历在遇到死胡同时必定导致路径断裂
设机票组合为 `tickets = [["JFK","KUL"], ["JFK","NRT"], ["NRT","JFK"]]`：
- $JFK$ 出发有两条路：走向死胡同 $KUL$（出度为 0，终点），或走向闭合子回路 $JFK \to NRT \to JFK$；
- 题目要求字典序最小，由于 `"KUL" < "NRT"`，贪心搜索**必定优先选择飞往 $KUL$**；
- **若使用前序遍历（一进就记）**：立即记录 `["JFK", "KUL"]`，到达 $KUL$ 后发现无路可走被迫终止，导致 $NRT \to JFK$ 子回路的机票被永久遗弃，算法直接失败！

##### 2. 拓扑本质：欧拉图的“终点唯一性”定理
在存在欧拉路径的有向图中：
- **中间节点**：$\text{出度} == \text{入度}$。只要能进入该节点，就必定有路能走出来；若存在环路，也必然是能绕回该节点自身的**闭合子回路**；
- **全图终点（Dead-end）**：$\text{入度} - \text{出度} == 1$。全图中**唯一一个一旦走进去就可能“无路可走”的点，只有真正的终点**！

##### 3. 后序入栈如何自动缝合（Splice）子回路？
Hierholzer 的核心哲学是：**“我不记录我什么时候到达，我只记录我什么时候再也出不去了（出边全部耗尽）。”**

```text
后序记录与回路缝合全流程 (tickets = [["JFK","KUL"], ["JFK","NRT"], ["NRT","JFK"]]):

1. DFS("JFK") 贪心弹出 "KUL" ➔ 进入 DFS("KUL");
2. KUL 无出边可走 (while 循环直接结束) ➔ 触发后序追加: route = ["KUL"] (★ 终点最先入栈!);
3. 递归回溯回到 DFS("JFK") ➔ JFK 出边还没空 (还有 "NRT"), while 循环继续!
4. 弹出 "NRT" ➔ 进入 DFS("NRT") ➔ 弹出 "JFK" ➔ 进入 DFS("JFK");
5. 此时所有机票全部消耗完毕，递归栈从最底层依次返回并后序追加:
   - DFS("JFK") 结束 ➔ route = ["KUL", "JFK"]
   - DFS("NRT") 结束 ➔ route = ["KUL", "JFK", "NRT"]
   - DFS("JFK") 最外层结束 ➔ route = ["KUL", "JFK", "NRT", "JFK"]
```

##### 4. 终点反转的数学必然性
观察 `route` 中记录的顺序：
$$\text{route} = [ \mathbf{KUL} \text{ (最先无路可走的终点)}, \quad JFK, \quad NRT, \quad \mathbf{JFK} \text{ (最外层主干起点)} ]$$
这相当于**从终点倒着往回穿线**。当我们将 `route` 整体翻转（`route[::-1]`）后：
$$\text{最终行程} = [ \mathbf{JFK} \text{ (起点)}, \quad NRT, \quad JFK, \quad \mathbf{KUL} \text{ (终点)} ]$$
- **死胡同节点必定沉淀到翻转后的最末尾（终点）**；
- **任何中途展开的子回路，都会被完完整整地缝合（Spliced）在经过它的那个主干节点之后**；
- **字典序贪心全局成立**：因为后序机制免疫死胡同陷阱，局部选字典序最小的边，翻转后该小分支自然排在前面，得到全局字典序最小解。"""

target_zh = "#### 手工推演示例"
if target_zh in zh:
    zh = zh.replace(target_zh, deepdive_zh + "\n\n" + target_zh)
    with open("notes/Leetcode/CoreSkills07 Design Graph.md", "w", encoding="utf-8") as f:
        f.write(zh)
    print("Added deep dive to Chinese note!")

# 2. Update English note
with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

deepdive_en = r"""#### Deep Dive: Why is Post-order Recording + Reversing Strictly Correct?

A common question is: *Why can't we record airports upon entry (pre-order), and why must we wait until all outgoing edges are exhausted in post-order before appending to `route` and reversing?*

##### 1. Core Conflict: Pre-order Traversal Collapses on Dead Ends
Consider `tickets = [["JFK","KUL"], ["JFK","NRT"], ["NRT","JFK"]]`:
- From $JFK$, there are two outgoing paths: the dead-end branch to $KUL$ ($\text{out-degree} = 0$, the final destination), and the closed sub-circuit $JFK \to NRT \to JFK$;
- Due to the tie-breaking rule, since `"KUL" < "NRT"`, greedy search **must pick $KUL$ first**;
- **If using Pre-order**: We immediately record `["JFK", "KUL"]`. Upon reaching $KUL$, no outgoing flights exist, forcing premature termination and stranding the $NRT \to JFK$ sub-circuit tickets!

##### 2. Topological Invariant: The "Unique Dead-End" Theorem
In any directed Eulerian graph:
- **Intermediate Nodes**: $\text{out-degree} == \text{in-degree}$. Any entry into the node guarantees a way out; any local cycles are closed sub-circuits returning to this node;
- **The Final Destination (Dead-end)**: $\text{in-degree} - \text{out-degree} == 1$. The **only node in the entire graph that can run out of exits is the true destination**!

##### 3. How Post-order DFS Automatically Splices Sub-circuits
Hierholzer's core philosophy: **"I don't record when I arrive; I only record when I can never leave again (all outgoing edges exhausted)."**

```text
Post-order Recording & Circuit Splicing (tickets = [["JFK","KUL"], ["JFK","NRT"], ["NRT","JFK"]]):

1. DFS("JFK") greedily pops "KUL" ➔ enters DFS("KUL");
2. KUL has no outgoing edges (while-loop empty) ➔ post-order append: route = ["KUL"] (★ Destination appended first!);
3. Unwinds back to DFS("JFK") ➔ JFK still has tickets left ("NRT"), while-loop continues!
4. Pops "NRT" ➔ enters DFS("NRT") ➔ pops "JFK" ➔ enters DFS("JFK");
5. All tickets now exhausted. Recursion unwinds and appends in post-order:
   - DFS("JFK") ends ➔ route = ["KUL", "JFK"]
   - DFS("NRT") ends ➔ route = ["KUL", "JFK", "NRT"]
   - DFS("JFK") outermost ends ➔ route = ["KUL", "JFK", "NRT", "JFK"]
```

##### 4. Mathematical Inevitability of Reversing
The recorded sequence in `route` is:
$$\text{route} = [ \mathbf{KUL} \text{ (first dead-end destination)}, \quad JFK, \quad NRT, \quad \mathbf{JFK} \text{ (outermost start)} ]$$
This represents **threading the needle backwards from the destination**. Reversing `route` (`route[::-1]`):
$$\text{Final Itinerary} = [ \mathbf{JFK} \text{ (Start)}, \quad NRT, \quad JFK, \quad \mathbf{KUL} \text{ (Destination)} ]$$
- **The dead-end airport is guaranteed to settle at the very end of the itinerary**;
- **Any sub-circuits explored along the way are spliced into the main path**;
- **Global Lexicographical Optimality**: Because the post-order splicing is immune to dead-end traps, picking the smallest candidate edge at each step guarantees the smallest sub-path appears first after reversal."""

target_en = "#### Trace by Hand"
if target_en in en:
    en = en.replace(target_en, deepdive_en + "\n\n" + target_en)
    with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Added deep dive to English note!")


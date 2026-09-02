# -*- coding: utf-8 -*-

# 1. Update Chinese note
with open("notes/Leetcode/CoreSkills07 Design Graph.md", "r", encoding="utf-8") as f:
    zh = f.read()

zh_target = r"""### 8. Word Ladder

BFS 求最短变换序列，队列里的每一项是 `(word, dist)`。给定单词的邻居不是拿它跟单词表里每一个词逐个比较，而是枚举它的每一个字符位置，把这一位换成 26 个字母中的每一个，拼出新词后检查是否在单词集合里。距离从 1 开始(`beginWord` 自身算第一步)，每往外扩一层加一。一个单词被访问后直接从集合里删除，代替单独维护一个 `visited` 集合，删除和查重共用同一个操作。

| 项目 | 内容 |
|---|---|
| 组合技巧 | BFS + 按位替换生成邻居 + 访问后从单词集合中移除 |
| 关键不变量 | 队首单词的 `dist` 就是从 `beginWord` 到它的最短步数(BFS 逐层扩展保证) |
| 时间 / 空间 | `O(26 · N · L²) / O(N · L)`，`N` 为单词数，`L` 为单词长度 |"""

zh_replacement = r"""### 8. Word Ladder（单词接龙）

#### 题目描述（LeetCode 127）
字典 `wordList` 中包含一组唯一的单词。给定两个单词 `beginWord`（起始单词）和 `endWord`（目标单词），要求找出从 `beginWord` 转换到 `endWord` 的**最短转换序列中的单词数目**。如果不存在这样的转换序列，返回 `0`。

**转换规则**：
1. 每一步只能改变单词中的**恰好一个字母**；
2. 每次转换后的新单词必须存在于字典 `wordList` 中（`beginWord` 本身可以不在字典中）；
3. 序列长度计算包含起点和终点（如 `"hit" -> "hot" -> "dot" -> "dog" -> "cog"` 包含 5 个单词，返回 `5`）。

```text
隐式无向图与最短路径图示 (示例: begin="hit", end="cog")
       [hit] (dist=1, 起点)
         │  (换第2位 'i'->'o')
         ▼
       [hot] (dist=2)
      ┌──┴───────────────┐
 (换第1位 'h'->'d')  (换第1位 'h'->'l')
      ▼                  ▼
    [dot] (dist=3)     [lot] (dist=3)
      │                  │
 (换第3位 't'->'g')  (换第3位 't'->'g')
      ▼                  ▼
    [dog] (dist=4)     [log] (dist=4)
      └──┬───────────────┘
         │  (换第1位 'd'/'l'->'c')
         ▼
       [cog] (dist=5, 命中终点!)
```

#### 图论本质与两大工程考点剖析

1. **图模型映射**：
   - **顶点 $V$**：`beginWord` 及 `wordList` 中的每一个有效单词；
   - **无权边 $E$**：若两个单词的汉明距离（Hamming Distance）为 1（只差 1 个字母），则存在一条长度为 1 的无向边；
   - **目标算法**：无向无权图上的最短路径 $\implies$ **必须使用 BFS 广度优先搜索（分层扩散，首次遇到终点即为全局全局最优最短路）**。

2. **考点一：邻居生成策略的性能抉择（为什么枚举 26 个字母远快于遍历字典？）**：
   - **策略 A（暴力遍历字典）**：遍历字典中的每个单词 $w$，逐字符比对与当前词是否只差 1 位。单步耗时 $\mathcal{O}(N \cdot L)$。当字典单词数 $N = 5000$、长度 $L = 5$ 时，每一步需比对 $5000 \times 5 = 25,000$ 次！
   - **策略 B（枚举 $L$ 个位置替换 26 个字母 + 哈希查表）**：枚举当前单词的 $L$ 个字符位置，每个位置尝试替换为 'a'~'z' 的 26 个字母，拼出新字符串并在 `word_set`（哈希集合）中以 $\mathcal{O}(L)$ 进行查找。单步耗时 $\mathcal{O}(26 \cdot L^2)$。当 $L = 5$ 时，每一步仅需 $26 \times 5^2 = 650$ 次！
   - **结论**：$650 \ll 25000$，策略 B 性能高出近 40 倍！

3. **考点二：原地删除（In-place Set Removal）替代 visited 集合**：
   - 传统 BFS 额外维护 `visited = set()` 记录已走过的节点；
   - 优化技巧：由于 `word_set` 仅作为有效字典使用，一旦某个单词被入队，后续任何更深层级再次到达该词都绝不可能得到更短路径。因此**入队时直接调用 `word_set.remove(next_word)`**，既完成了合法性检查，又同时完成了剪枝与防重复访问，免去了维护额外 visited 集合的双重哈希开销。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 隐式图 BFS + 按位枚举 26 字母生成邻居 + 原地集合删除防重剪枝 |
| 关键不变量 | 队首单词的 `dist` 就是从 `beginWord` 到它的最短步数（BFS 逐层扩散单调递增性保证） |
| 时间 / 空间 | 时间 $\mathcal{O}(26 \cdot N \cdot L^2)$，空间 $\mathcal{O}(N \cdot L)$（$N$ 为单词总数，$L$ 为单词长度） |"""

if zh_target in zh:
    zh = zh.replace(zh_target, zh_replacement)
    with open("notes/Leetcode/CoreSkills07 Design Graph.md", "w", encoding="utf-8") as f:
        f.write(zh)
    print("Enhanced Chinese Word Ladder Section!")
else:
    print("Chinese target not found")

# 2. Update English note
with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "r", encoding="utf-8") as f:
    en = f.read()

en_target = r"""### 8. Word Ladder

BFS for shortest transformation sequence, queue holds `(word, dist)`. The neighbors of a given word are not found by comparing against every word in the dictionary; instead, enumerate every character position, replace that position with each of the 26 letters, form the new word, and check whether it is in the word set. Distance starts from 1 (`beginWord` itself counts as step 1) and increments by one with each layer. Once visited, a word is deleted directly from the set instead of maintaining a separate `visited` set, combining removal and dedup into one operation.

| Item | Detail |
|---|---|
| Technique combination | BFS + generate neighbors by character substitution + remove from word set after visit |
| Key invariant | The `dist` of the word at the head of the queue is the shortest distance from `beginWord` to it (guaranteed by BFS layer-by-layer expansion) |
| Time / Space | `O(26 · N · L²) / O(N · L)`, where `N` is the number of words, `L` is the word length |"""

en_replacement = r"""### 8. Word Ladder

#### Problem Description (LeetCode 127)
Given two words, `beginWord` and `endWord`, and a dictionary `wordList` of unique words, return the **number of words in the shortest transformation sequence** from `beginWord` to `endWord`, or `0` if no such sequence exists.

**Transformation Rules**:
1. Only **one letter** can be changed at a time;
2. Each transformed word must exist in the word list `wordList` (`beginWord` does not need to be in `wordList`);
3. The sequence length counts all words from start to finish (e.g., `"hit" -> "hot" -> "dot" -> "dog" -> "cog"` has length 5).

```text
Implicit Unweighted Graph & Shortest Path (Example: begin="hit", end="cog")
       [hit] (dist=1, Source)
         │  (substitute 2nd char 'i'->'o')
         ▼
       [hot] (dist=2)
      ┌──┴───────────────┐
 (substitute 1st char) (substitute 1st char)
      ▼                  ▼
    [dot] (dist=3)     [lot] (dist=3)
      │                  │
 (substitute 3rd char) (substitute 3rd char)
      ▼                  ▼
    [dog] (dist=4)     [log] (dist=4)
      └──┬───────────────┘
         │  (substitute 1st char)
         ▼
       [cog] (dist=5, Target reached!)
```

#### Graph Modeling & Core Engineering Takeaways

1. **Graph Theoretical Formulation**:
   - **Vertices $V$**: `beginWord` and all words in `wordList`;
   - **Unweighted Edges $E$**: An undirected edge exists between two words if their Hamming distance is exactly 1 (differ by 1 character);
   - **Target Algorithm**: Shortest path on an unweighted graph $\implies$ **Breadth-First Search (BFS)** guarantees finding the global shortest sequence on first arrival.

2. **Takeaway 1: Neighbor Generation Complexity ($\mathcal{O}(26 \cdot L^2)$ vs $\mathcal{O}(N \cdot L)$)**:
   - **Strategy A (Scan Dictionary)**: Compare the current word against every word in `wordList` character-by-character. Cost per step is $\mathcal{O}(N \cdot L)$. For $N = 5000, L = 5$, this requires $5000 \times 5 = 25,000$ comparisons per step!
   - **Strategy B (Enumerate 26 Letters + Set Lookup)**: Enumerate $L$ positions, substitute 26 letters ('a'~'z'), slice the string in $\mathcal{O}(L)$, and check existence in `word_set` in $\mathcal{O}(L)$. Cost per step is $\mathcal{O}(26 \cdot L^2)$. For $L = 5$, this is $26 \times 25 = 650$ operations!
   - **Conclusion**: $650 \ll 25000$, Strategy B is almost 40x faster!

3. **Takeaway 2: In-place Set Deletion instead of Visited Set**:
   - Instead of maintaining a separate `visited = set()`, we can directly remove `next_word` from `word_set` upon enqueueing (`word_set.remove(next_word)`).
   - Because BFS explores level-by-level, any subsequent attempt to visit this word in a deeper level can never yield a shorter path. Removing it in-place achieves validity checking, deduplication, and pruning in a single $\mathcal{O}(1)$ step!

| Item | Detail |
|---|---|
| Technique combination | Implicit Graph BFS + 26-Letter Substitution Neighbor Generation + In-place Set Pruning |
| Key invariant | The `dist` of the popped word is strictly the shortest distance from `beginWord` (guaranteed by BFS layer-by-layer expansion) |
| Time / Space | Time $\mathcal{O}(26 \cdot N \cdot L^2)$, space $\mathcal{O}(N \cdot L)$ ($N$ = dictionary size, $L$ = word length) |"""

if en_target in en:
    en = en.replace(en_target, en_replacement)
    with open("notes/Leetcode/CoreSkills07 Design Graph.en.md", "w", encoding="utf-8") as f:
        f.write(en)
    print("Enhanced English Word Ladder Section!")
else:
    print("English target not found")


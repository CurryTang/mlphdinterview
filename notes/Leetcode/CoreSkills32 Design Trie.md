# Tries

## 前置：Design Trie

### 面试目标

Trie（前缀树）是一种按字符逐层展开的树形结构，用来高效存储和查询一组字符串，尤其擅长"是否存在某个前缀"这类查询。核心不是某个具体算法，而是一种数据组织方式：把字符串拆成字符序列，让共享同一段前缀的字符串共享同一段路径。

### 核心设计

每个节点包含两部分：一个从字符到子节点的映射 `children`，以及一个布尔标记 `is_end`，表示"从根节点走到这里，正好拼出一个完整插入过的单词"。根节点本身不代表任何字符，只是所有单词共享的起点。

`children` 常见有两种实现：字符集不确定或很大时用哈希表（`dict`），只处理小写英文字母时可以用长度 26 的数组，用 `ord(ch) - ord('a')` 做下标，省掉哈希开销。

三个基本操作共享同一套移动规则：从根节点出发，沿着字符串的每个字符找对应的子节点。

- `insert(word)`：逐字符往下走，子节点不存在就新建，最后把终点节点的 `is_end` 置为真。
- `search(word)`：逐字符往下走，子节点不存在就直接判否；走到终点后还要检查 `is_end`，因为路径存在只说明这是某个单词的前缀，不代表它本身是一个完整单词。
- `startsWith(prefix)`：和 `search` 用同一套移动逻辑，区别是走到终点就直接判是，不需要检查 `is_end`。

### 复杂度

设字符串长度为 `L`，字符集大小为 `Σ`。`insert`、`search`、`startsWith` 都是 `O(L)` 时间，和字典里存了多少个单词无关。空间是 `O(所有单词的总字符数 × Σ)` 的量级，实际远小于这个上界，因为共享前缀不会重复占用节点。

### 常见坑

- 把"路径存在"和"这是一个完整单词"混为一谈：`search("ca")` 在插入过 `"cat"` 之后，路径是存在的，但如果 `"ca"` 本身没有被单独插入过，`is_end` 是假，`search` 必须返回否。
- 用数组实现 `children` 时没有处理字符集之外的输入，或者数组大小和实际字符集对不上。
- 插入空字符串或空前缀查询时忘记处理边界：空字符串对应根节点本身。

### 参考解法

<details class="solution">
<summary>展开解法</summary>

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _traverse(self, s: str):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word: str) -> bool:
        node = self._traverse(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._traverse(prefix) is not None
```

`search` 和 `startsWith` 共用 `_traverse`，两者的差异只在最后一步要不要检查 `is_end`，这也是 Trie 上几乎所有查询类操作的通用写法：先复用同一套移动逻辑走到终点节点（或提前判否），再根据题目要求决定终点节点本身要满足什么条件。

</details>

下面的演示插入 `cat`、`car`、`card`、`dog` 四个单词，先看它们怎么共享节点长成一棵树，再看 `search` 和 `startsWith` 在这棵树上具体怎么走。

```trie-core-demo
```

## 学习顺序

题目来自 [NeetCode 150](https://neetcode.io/practice/practice/neetcode150) 的 Tries 模块。

| 顺序 | 原题 | 要掌握的内容 |
|---:|---|---|
| 1 | [208. Implement Trie (Prefix Tree)](https://neetcode.io/problems/implement-prefix-tree/question?list=neetcode150) | Trie 的标准实现，直接套用前置里的模板 |
| 2 | [211. Design Add and Search Words Data Structure](https://neetcode.io/problems/design-word-search-data-structure/question?list=neetcode150) | 通配符 `.` 触发的多分支 DFS |
| 3 | [212. Word Search II](https://neetcode.io/problems/search-for-word-ii/question?list=neetcode150) | 用一棵共享 Trie 替代逐词网格搜索，并在回溯时剪枝 |

## 模块二：3 道题目的映射

### 1. Implement Trie Prefix Tree

直接实现前置里的 `Trie` 类。这道题本身就是在考察对 Trie 核心设计的掌握程度，不需要额外的技巧。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 标准 Trie：`children` 哈希表 + `is_end` 标记 |
| 关键不变量 | 节点的 `is_end` 为真，当且仅当从根到这个节点的路径拼出过一个完整插入的单词 |
| 时间 / 空间 | 三个操作均为 `O(L) / O(总字符数 × Σ)` |

#### Quick Coding：Implement Trie Prefix Tree

```python
class Trie:
    def __init__(self):
        ...

    def insert(self, word):
        ...

    def search(self, word):
        ...

    def startsWith(self, prefix):
        ...
```

<details>
<summary>参考答案</summary>

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def _traverse(self, s: str):
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node

    def search(self, word: str) -> bool:
        node = self._traverse(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        return self._traverse(prefix) is not None
```

`insert("apple")` 之后，`search("apple")` 为真，`search("app")` 为假，`startsWith("app")` 为真；再 `insert("app")` 之后，`search("app")` 变为真。这组行为覆盖了"前缀存在但单词不存在"和"补插入后单词也存在"两种情况，是这道题最常用来验证实现是否正确的用例。

</details>

### 2. Design Add and Search Words Data Structure

在标准 Trie 上加一种通配符查询：`search(word)` 里的 `.` 可以匹配任意一个字符。普通字符只需要沿着一个子节点继续；遇到 `.` 时，当前节点的每一个子节点都是候选，需要对每一个候选分别递归，只要有一个分支最终成功，整体就返回真。这是深度优先搜索里"多分支探索"的直接应用：普通字符对应"只有一条路可走"，通配符对应"当前节点有几个子节点就分几条路"。

| 项目 | 内容 |
|---|---|
| 组合技巧 | Trie 插入不变，查询改成允许通配符分支的 DFS |
| 关键不变量 | `dfs(node, i)` 为真，当且仅当从 `node` 出发能够匹配 `word[i:]` 剩余部分 |
| 时间 / 空间 | 最坏情况（`word` 全是 `.`）是 `O(Σ^L)`，`Σ` 是字符集大小；不含通配符时退化为 `O(L)` |

下面的演示展示查询 `.at` 在 `bad`、`dad`、`cat` 三个单词组成的 Trie 上如何分支：第 0 个字符是通配符，会依次尝试根节点的每一个子节点。

```trie-wildcard-demo
```

#### Quick Coding：Design Add and Search Words Data Structure

```python
class WordDictionary:
    def __init__(self):
        ...

    def addWord(self, word):
        ...

    def search(self, word):
        ...
```

<details>
<summary>参考答案</summary>

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        def dfs(node: TrieNode, i: int) -> bool:
            if i == len(word):
                return node.is_end
            ch = word[i]
            if ch == '.':
                return any(dfs(child, i + 1) for child in node.children.values())
            if ch not in node.children:
                return False
            return dfs(node.children[ch], i + 1)

        return dfs(self.root, 0)
```

`any(...)` 在生成器上短路：一旦某个子节点分支返回真，后面还没试过的子节点就不会再递归，不需要额外写提前退出的逻辑。`i == len(word)` 时不再看字符本身，只看 `node.is_end`，这一点和标准 Trie 的 `search` 完全一致，通配符只改变了"怎么走到终点"，没有改变"终点要满足什么条件"。

</details>

### 3. Word Search II

题目给出一个字符网格和一组目标单词，要求返回所有能在网格里拼出来的目标单词，路径要求水平或竖直相邻、同一个格子不能在一个单词内重复使用。

对每个目标单词单独在网格上跑一次搜索，需要为每个单词各自做一次完整的网格 DFS/回溯。如果目标单词有 `words` 个、平均长度为 `L`，网格有 `mn` 个格子，这个做法的开销是"单词数"和"网格搜索"两者的乘积，而且不同单词之间共享的前缀会被重复搜索多次，比如 `"cat"` 和 `"car"` 会各自把 `c -> a` 这段路径搜一遍。

更好的做法是把全部目标单词合并建成一棵 Trie，只对网格做一次统一的 DFS：从每个格子出发，DFS 的每一步不再单独判断"当前字符是不是某个单词的下一个字符"，而是判断"当前字符在 Trie 当前节点的 `children` 里有没有对应的子节点"。多个单词共享的前缀在网格 DFS 里也只会沿着同一条 Trie 路径走一次，天然消除了重复搜索。网格本身的遍历规则和 [[CoreSkills07 Design Graph|Graphs 章节的 Matrix DFS 模板]] 完全一致：越界检查、`visited` 标记、四方向递归，Word Search II 只是在这个模板基础上，把"访问节点"的判断依据换成了 Trie 的 `children`。

DFS 过程中走到 Trie 的某个终点节点（`word` 字段非空）时，说明拼出了一个完整目标单词，记录下来；为了避免同一个单词被网格里的多条不同路径重复记录，记录后立刻把这个终点节点的 `word` 字段清空。另一处剪枝是：如果当前 Trie 节点的 `children` 已经空了（这条路径能拼出的单词都已经找到或者根本不存在能继续匹配的单词），就把它从父节点的 `children` 里删除，后续经过同一个前缀的 DFS 调用会更早终止。

| 项目 | 内容 |
|---|---|
| 组合技巧 | 把目标单词合并成一棵共享 Trie，配合网格回溯，用 `children` 判断能否继续 |
| 关键不变量 | DFS 走到的 Trie 节点，恰好对应网格路径已经匹配上的前缀 |
| 时间 / 空间 | 时间 `O(mn · 4 · 3^(L-1))`（`L` 是最长单词长度，回溯时最多 3 个方向可继续），空间 `O(所有目标单词的总字符数)` |

#### Quick Coding：Word Search II

```python
def findWords(board, words):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List


class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = None


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for word in words:
            node = root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.word = word

        rows, cols = len(board), len(board[0])
        result = []

        def dfs(r: int, c: int, node: TrieNode) -> None:
            ch = board[r][c]
            if ch not in node.children:
                return
            child = node.children[ch]
            if child.word is not None:
                result.append(child.word)
                child.word = None

            board[r][c] = '#'
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                    dfs(nr, nc, child)
            board[r][c] = ch

            if not child.children:
                del node.children[ch]

        for r in range(rows):
            for c in range(cols):
                dfs(r, c, root)

        return result
```

`node.word` 直接存完整单词而不是布尔标记，是因为 DFS 走到终点节点时，节点本身的路径信息已经不在手边（只有当前字符和 Trie 节点），直接存字符串省去了重新拼接路径的麻烦。`board[r][c] = '#'` 和递归结束后的还原，是这道题里访问标记的写法：把当前格子标成一个不会出现在任何单词里的字符，代替额外的 `visited` 集合。最后的 `del node.children[ch]` 是可选的剪枝，不加也能得到正确答案，但去掉之后同一个死胡同前缀会在网格的每一次起点尝试里被重复走一遍。

</details>

## 模块三：面试前最后检查

1. 题目要不要按前缀批量查询，或者反复问"是否存在某个前缀/单词"？这是 Trie 最基本的适用信号。
2. 查询里是否出现通配符或者"匹配任意字符"这类要求？如果有，`search` 要从单一路径的移动改写成多分支 DFS。
3. 是不是要在网格或者字符串数组里同时找多个目标模式？多个模式要找的话，先把它们合并成一棵共享 Trie，再统一遍历，比对每个模式单独跑一遍更省时间。
4. Trie 节点要不要在插入完成后清理？Word Search II 这类"边遍历边消耗"的场景，及时删除耗尽的分支能让后续查询更快退出。

最后只记一句：

> Trie 解决的是"很多字符串共享前缀"这一类问题；看到"前缀""字典""通配符匹配""网格里同时找多个单词"，先想能不能把这些字符串合并成一棵树。

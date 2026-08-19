# Tries

## Prerequisite: Design Trie

### Interview Goal

A trie (prefix tree) is a tree structure that expands one character at a time, built for efficiently storing and querying a set of strings, particularly well suited to "does this prefix exist" queries. The core idea is not a specific algorithm but a way of organizing data: strings are broken into character sequences, and strings that share a prefix share the path representing that prefix.

### Core Design

Each node holds two things: a mapping `children` from character to child node, and a boolean flag `is_end` marking that the path from the root to this node spells out a complete word that was inserted. The root itself represents no character; it is only the shared starting point for every word.

`children` is commonly implemented one of two ways: a hash map (`dict`) when the character set is uncertain or large, or a fixed-length array of size 26 when only lowercase English letters are involved, indexed with `ord(ch) - ord('a')` to skip hashing overhead.

The three basic operations share the same movement rule: start at the root and follow the child corresponding to each character of the string.

- `insert(word)`: walk down character by character, creating a child whenever one does not exist, then set `is_end` to true on the final node.
- `search(word)`: walk down character by character, returning false immediately if a child is missing; once the walk reaches the final node, `is_end` still has to be checked, since the path existing only means this string is a prefix of some inserted word, not necessarily a complete word on its own.
- `startsWith(prefix)`: uses the same movement logic as `search`; the only difference is that reaching the final node is enough to return true, with no `is_end` check.

### Complexity

Let `L` be the string length and `Σ` the size of the character set. `insert`, `search`, and `startsWith` all run in `O(L)` time, independent of how many words are stored. Space is on the order of `O(total characters across all words × Σ)`; the actual usage is well below that bound, since shared prefixes never occupy a node twice.

### Common Pitfalls

- Conflating "the path exists" with "this is a complete word": after inserting `"cat"`, `search("ca")` follows an existing path, but if `"ca"` itself was never inserted, `is_end` is false and `search` must return false.
- Implementing `children` as a fixed array without handling input outside the assumed character set, or sizing the array to something that does not match the actual character set.
- Forgetting the boundary case of an empty string being inserted or queried as a prefix: an empty string corresponds to the root itself.

### Reference Solution

<details class="solution">
<summary>Expand solution</summary>

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

`search` and `startsWith` share `_traverse`; the only difference between them is whether the last step checks `is_end`. This is also the general pattern for almost every query-style operation on a trie: reuse the same movement logic to reach the final node (or fail early), then decide what condition that final node needs to satisfy based on what the problem asks for.

</details>

The demo below inserts the four words `cat`, `car`, `card`, and `dog`. It first shows how they share nodes to form one tree, then shows exactly how `search` and `startsWith` walk that same tree.

```trie-core-demo
```

## Learning Order

Problems come from the Tries module of [NeetCode 150](https://neetcode.io/practice/practice/neetcode150).

| Order | Problem | What to Master |
|---:|---|---|
| 1 | [208. Implement Trie (Prefix Tree)](https://neetcode.io/problems/implement-prefix-tree/question?list=neetcode150) | The standard trie implementation, a direct application of the template above |
| 2 | [211. Design Add and Search Words Data Structure](https://neetcode.io/problems/design-word-search-data-structure/question?list=neetcode150) | A multi-branch DFS triggered by the wildcard `.` |
| 3 | [212. Word Search II](https://neetcode.io/problems/search-for-word-ii/question?list=neetcode150) | Replacing per-word grid search with one shared trie, plus pruning during backtracking |

## Module 2: Three Problems

### 1. Implement Trie Prefix Tree

A direct implementation of the `Trie` class from the prerequisite section above. This problem tests understanding of the core trie design itself, with no additional technique needed.

| Item | Detail |
|---|---|
| Technique | A standard trie: a `children` hash map plus an `is_end` flag |
| Key invariant | A node's `is_end` is true if and only if the path from the root to that node spells out a complete word that was inserted |
| Time / Space | All three operations are `O(L) / O(total characters × Σ)` |

#### Quick Coding: Implement Trie Prefix Tree

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
<summary>Reference answer</summary>

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

After `insert("apple")`, `search("apple")` is true, `search("app")` is false, and `startsWith("app")` is true. Inserting `"app"` afterward flips `search("app")` to true. This sequence covers both "a prefix exists but the word does not" and "the word exists once it is inserted too," which is the pair of cases most commonly used to check whether an implementation is correct.

</details>

### 2. Design Add and Search Words Data Structure

Adds a wildcard query on top of a standard trie: a `.` inside `search(word)` can match any single character. An ordinary character continues along one child; a `.` makes every child of the current node a candidate, each explored recursively, and the overall call returns true as soon as any one branch succeeds. This is a direct application of multi-branch depth-first search: an ordinary character means there is exactly one path forward, while a wildcard means there are as many paths forward as the current node has children.

| Item | Detail |
|---|---|
| Technique | Trie insertion is unchanged; the query becomes a DFS that allows wildcard branching |
| Key invariant | `dfs(node, i)` is true if and only if starting from `node`, the remaining suffix `word[i:]` can be matched |
| Time / Space | Worst case (`word` made entirely of `.`) is `O(Σ^L)`, where `Σ` is the character set size; with no wildcards it degrades to `O(L)` |

The demo below traces the query `.at` on a trie built from `bad`, `dad`, and `cat`. Character 0 is a wildcard, so it tries every child of the root in turn.

```trie-wildcard-demo
```

#### Quick Coding: Design Add and Search Words Data Structure

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
<summary>Reference answer</summary>

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

`any(...)` short-circuits over the generator: as soon as one child branch returns true, the remaining, not-yet-tried children are never recursed into, with no extra early-exit logic needed. When `i == len(word)`, the character itself is no longer examined, only `node.is_end`, matching a standard trie's `search` exactly. The wildcard only changes how the final node is reached, not what condition that final node has to satisfy.

</details>

### 3. Word Search II

The input is a character grid and a list of target words. The task is to return every target word that can be spelled out on the grid, where the path must move horizontally or vertically between adjacent cells and the same cell cannot be reused within one word.

Running a separate grid search for each target word means a full grid DFS/backtrack per word. With `words` target words of average length `L` and a grid of `mn` cells, this approach costs the product of "number of words" and "grid search," and a prefix shared between words gets searched over and over: `"cat"` and `"car"`, for example, would each search the `c -> a` segment separately.

A better approach merges all target words into one shared trie and runs a single unified DFS over the grid: instead of asking "is the current character the next character of some specific word," each DFS step asks "does the current Trie node's `children` have an entry for the current character." A prefix shared by multiple words is walked along the same trie path only once during the grid DFS, which removes the repeated searching by construction. The grid traversal rule itself is identical to the [[CoreSkills07 Design Graph|Matrix DFS template in the Graphs chapter]]: bounds checking, a `visited` marker, and recursion into the four directions. Word Search II only replaces the "should this node be visited" check with a lookup into the trie's `children`.

When the DFS reaches a trie node that terminates a word (its `word` field is not empty), a complete target word has been spelled out and gets recorded; to avoid the same word being recorded again through a different path on the grid, the `word` field on that node is cleared immediately after recording. A second pruning step: once a trie node's `children` becomes empty (every word reachable along that path has already been found, or none ever existed to continue matching), it is removed from its parent's `children`, so a later DFS call passing through the same prefix terminates sooner.

| Item | Detail |
|---|---|
| Technique | Merge target words into one shared trie, combine with grid backtracking, and use `children` to decide whether to continue |
| Key invariant | The trie node the DFS has reached corresponds exactly to the prefix the grid path has matched so far |
| Time / Space | Time `O(mn · 4 · 3^(L-1))` (`L` is the longest word's length; backtracking has at most 3 directions to continue into), space `O(total characters across all target words)` |

#### Quick Coding: Word Search II

```python
def findWords(board, words):
    ...
```

<details>
<summary>Reference answer</summary>

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

`node.word` stores the complete word directly instead of a boolean flag, because by the time the DFS reaches the terminal node, the path information itself is no longer at hand (only the current character and trie node are), so storing the string directly avoids reconstructing the path. Setting `board[r][c] = '#'` and restoring it after the recursive calls return is this problem's way of marking visited cells: the current cell is overwritten with a character that never appears in any word, replacing a separate `visited` set. The final `del node.children[ch]` is an optional pruning step; the answer is correct without it, but leaving it out means the same dead-end prefix gets walked again from every new starting cell on the grid.

</details>

## Module 3: Final Checklist Before an Interview

1. Does the problem need batched prefix queries, or does it repeatedly ask "does this prefix/word exist"? This is the most basic signal that a trie applies.
2. Does the query involve a wildcard or "match any character" requirement? If so, `search` needs to change from moving along a single path to a multi-branch DFS.
3. Does the problem need to find several target patterns at once, on a grid or in an array of strings? When there are multiple patterns to find, merging them into one shared trie first and traversing once is faster than running a separate search per pattern.
4. Should trie nodes be cleaned up after insertion? In "consume while traversing" scenarios like Word Search II, removing exhausted branches promptly lets later queries terminate sooner.

One sentence to keep in mind:

> A trie solves problems where many strings share prefixes. Seeing "prefix," "dictionary," "wildcard matching," or "find several words on a grid at once" is the cue to ask whether those strings can be merged into one tree.

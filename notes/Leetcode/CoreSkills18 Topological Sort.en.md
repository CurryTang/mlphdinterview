# Topological Sort: Turn Partial-Order Constraints into a Linear Answer

## Interview Goal

Topological sorting is not an algorithm for "sorting an array." It is a graph algorithm that arranges a set of dependencies into a legal execution order. Whenever a problem mentions "precedence, dependencies, courses, build order, letter order, or task scheduling," you should first ask yourself: can these relationships be modeled as a directed graph?

If each edge `u -> v` means `u` must come before `v`, then topological sorting outputs a sequence such that every edge in the graph points from left to right in that sequence. Only a **DAG (Directed Acyclic Graph)** is guaranteed to have a topological order; if there is a cycle, such as `a -> b -> c -> a`, it represents mutual dependency and no legal order can be given.

## What Is Topological Sorting

Given a directed graph `G = (V, E)`:

- `V` is the set of nodes, such as courses, tasks, or characters.
- `E` is the set of constraints, such as `pre -> course`, `dependency -> target`, or `smaller letter -> larger letter`.
- A topological order is a linear ordering containing all nodes and satisfying: for every edge `u -> v`, `u` appears before `v`.

It solves the problem of turning a **partial order into a linear order**. A partial order tells you only some precedence relationships, such as `h < e` and `e < r`, but it does not say that all characters can be compared directly. Topological sorting gives any legal linear answer without violating the known constraints.

## Kahn's Algorithm: Start from Nodes with No Prerequisites

Kahn's algorithm is the most intuitive BFS formulation in interviews.

1. Build the graph and count the indegree `indegree` of each node.
2. Put all nodes with indegree `0` into a queue.
3. Each time, pop a node `u` and add it to the answer.
4. Traverse all successors `v` of `u` and do `indegree[v] -= 1`.
5. If a successor's indegree drops to `0`, it means all of its prerequisites have been completed, so it can be enqueued.
6. If the final answer length is smaller than the number of nodes, the graph contains a cycle.

```text
queue = all nodes with indegree 0
order = []

while queue:
  u = queue.pop_front()
  order.append(u)

  for v in graph[u]:
    indegree[v] -= 1
    if indegree[v] == 0:
      queue.push_back(v)

if len(order) != len(nodes):
  there is a cycle
else:
  order is a valid topological ordering
```

The complexity is `O(V + E)`, because each node is enqueued and dequeued once, and each edge is processed only once.

## DFS Formulation: Use Three Colors to Detect Cycles

DFS can also perform topological sorting. The core idea is "append in postorder":

- `0 = unvisited`: not visited yet.
- `1 = visiting`: currently on the DFS path.
- `2 = visited`: this node and all its successors have already been fully processed.

If DFS encounters a `visiting` node, the current path forms a cycle. If all successors of a node have been processed, then add the node to the answer. Finally, reverse the answer to obtain a topological order.

Kahn is better for explaining "indegree and dependency release"; DFS is better for recursive implementations and cycle detection. Either is fine in an interview, but you must be able to explain the edge direction clearly.

## Visualization: How Foreign Dictionary Becomes Topological Sorting

The key to Foreign Dictionary / Alien Dictionary is not string processing itself, but extracting partial-order relationships between characters from adjacent words. Take `words = ["hrn", "hrf", "er", "enn", "rfnn"]` as an example:

The essential process of this problem is:

1. Initialize all characters that appear, to avoid missing isolated nodes with no edges.
2. Compare only adjacent words, because when the whole dictionary is sorted, each adjacent pair provides the minimum necessary constraint.
3. Find the first different characters `a` and `b` in adjacent words, add the edge `a -> b`, and then stop comparing that pair.
4. If no differing character is found but the previous word is longer, for example `["abc", "ab"]`, this is invalid input, so return an empty string directly.
5. Perform topological sorting on the character graph; if a cycle is detected, also return an empty string.

## Foreign Dictionary as the Example Problem

The problem gives an ordered dictionary `words` in an alien language. These words are still composed of English letters, but the relative order of the letters is unknown. You need to return one legal letter order; if no legal order exists, return an empty string. Problem source: <https://neetcode.io/problems/foreign-dictionary/question?list=neetcode150>

```topo-demo
foreign-dictionary
```

### Graph Construction Rule

For any adjacent words `word1` and `word2`:

```text
word1 = h r n
word2 = h r f
             ^
the first different characters are n and f, so n must come before f; add edge n -> f
```

Note that you look only at the first different character. Later characters cannot be used to add more edges, because lexicographic order is determined at the first difference.

### Python Solution: Kahn BFS

```python
from collections import defaultdict, deque
from typing import List

class Solution:
    def foreignDictionary(self, words: List[str]) -> str:
        graph = {char: set() for word in words for char in word}
        indegree = {char: 0 for char in graph}

        for first, second in zip(words, words[1:]):
            min_len = min(len(first), len(second))

            if len(first) > len(second) and first[:min_len] == second[:min_len]:
                return ""

            for i in range(min_len):
                if first[i] != second[i]:
                    src, dst = first[i], second[i]
                    if dst not in graph[src]:
                        graph[src].add(dst)
                        indegree[dst] += 1
                    break

        queue = deque([char for char, degree in indegree.items() if degree == 0])
        order = []

        while queue:
            char = queue.popleft()
            order.append(char)

            for nxt in graph[char]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(indegree):
            return ""

        return "".join(order)
```

The time complexity is `O(N + V + E)`, where `N` is the total length of all words, `V` is the number of distinct characters, and `E` is the number of character partial-order edges. The space complexity is `O(V + E)`.

### Why This Code Passes

- `graph = {char: set() ...}` registers every character as a node first, ensuring the answer includes isolated characters.
- `if dst not in graph[src]` prevents duplicate edges from increasing indegree multiple times.
- The prefix-invalid case must be handled before comparing characters; otherwise `["abc", "ab"]` would be incorrectly treated as adding no new constraint.
- `len(order) != len(indegree)` is Kahn's cycle detection. If there is a cycle, nodes in the cycle can never drop to indegree `0`.

## Common Pitfalls

- Reversing the edge direction: if `word1` comes before `word2`, and the first differing characters are `a/b`, you should add `a -> b`.
- Comparing characters beyond the first difference: lexicographic order is determined only by the first differing character.
- Not handling the prefix-invalid case: `["abc", "ab"]` must return `""`.
- Using a list to store neighbors without deduplication, causing indegree to be increased repeatedly.
- Forgetting to put all characters into the graph, which causes missing characters in the answer.
- Assuming the answer must be unique; the problem usually allows any valid topological order.

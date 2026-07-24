# Prim's Algorithm

## Interview Goal

Master Prim's algorithm for the minimum spanning tree: start from one node, and each time choose the minimum edge that connects the visited set to the unvisited set.

## Core Process

1. Choose any starting point.
2. Add the starting point's edges to a min-heap.
3. Each time, pop the edge with the smallest weight.
4. If the edge connects to an unvisited node, add it to the MST, then continue adding that node's edges.

## Complexity

- Adjacency list + heap: `O(E log E)`, also commonly written as `O(E log V)`.
- Space: `O(V + E)`.

## Common Pitfalls

- If the graph is disconnected, it is impossible to obtain an MST covering all nodes.
- Without `visited`, nodes can be added repeatedly and cycles will form.
- Confusing the meaning of `dist` in Prim and Dijkstra.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

Prim maintains the set of nodes already added to the MST and a min-heap of candidate edges connecting to outside nodes.

```text
visited = set([start])
heap = edges from start
cost = 0
while heap and len(visited) < n:
  w, u, v = heappop(heap)
  if v in visited: continue
  visited.add(v)
  cost += w
  for next, nw in adj[v]:
    if next not in visited:
      heappush(heap, (nw, v, next))
```

If `visited` contains fewer than `n` nodes at the end, the graph is disconnected, so there is no MST covering all nodes.

</details>

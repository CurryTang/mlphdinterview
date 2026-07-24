# Kruskal's Algorithm

## Interview Goal

Master Kruskal's algorithm for the minimum spanning tree: try adding edges in order of increasing weight, and use Union-Find to avoid cycles.

## Core Process

1. Sort all edges by weight.
2. Initialize Union-Find.
3. Traverse edges `(u, v, w)` one by one.
4. If `u` and `v` are not connected, add the edge to the MST and union them.
5. Stop after adding `V - 1` edges.

## Complexity

- Dominated by sorting: `O(E log E)`
- Union-Find operations are approximately `O(1)`.

## Common Pitfalls

- Forgetting to skip edges that would form a cycle.
- Repeating undirected edges, which doubles the size of the sorted list.
- If the graph is disconnected, you can only obtain a minimum spanning forest.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

Kruskal first sorts all edges globally by weight, then uses Union-Find to determine whether an edge would create a cycle.

```text
sort edges by weight
dsu = UnionFind(n)
cost = 0
for u, v, w in edges:
  if dsu.union(u, v):
    cost += w
    used += 1
    if used == n - 1: break
```

Only when `union` returns true does the edge connect two different connected components, so it can be added to the MST.

</details>

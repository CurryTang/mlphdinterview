# Design Graph

## Interview Goal

Implement a graph structure and understand adjacency lists, adjacency matrices, directed/undirected graphs, weighted edges, and traversal interfaces.

## Core Design

- Adjacency lists are suitable for sparse graphs: `node -> neighbors`.
- Adjacency matrices are suitable when the number of nodes is small and edge queries are frequent.
- In an undirected graph, adding an edge requires writing both `u -> v` and `v -> u`.
- In a weighted graph, neighbors usually store `(neighbor, weight)`.

## Complexity

- Adjacency list space: `O(V + E)`
- Adjacency matrix space: `O(V^2)`
- Traversing all nodes and edges: `O(V + E)`

## Common Pitfalls

- Adding only one direction of an undirected edge.
- Forgetting `visited` during traversal and getting stuck in an infinite loop on a cycle.
- Forcing array indices even when node IDs are not contiguous.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

An adjacency list is the most general approach. For an unweighted graph, store neighbor nodes; for a weighted graph, store `(neighbor, weight)`.

```text
addEdge(u, v, w=1):
  adj[u].append((v, w))
  if undirected:
    adj[v].append((u, w))

neighbors(u):
  return adj.get(u, [])
```

Use `visited` during traversal. If the nodes are not contiguous integers, storing the adjacency list in a map/dictionary is safer.

</details>

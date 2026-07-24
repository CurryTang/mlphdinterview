# Design Disjoint Set (Union-Find)

## Interview Goal

Implement a disjoint set union structure for dynamically maintaining connected components. The core operations are `find` and `union`.

## Core Design

- `parent[x]` points to the representative node of `x`.
- `find(x)` follows parent pointers until it reaches the root.
- Path compression makes nodes on the search path point directly to the root.
- Union by rank/size attaches the smaller tree under the larger one.

## Complexity

- With path compression and union by rank: approximately `O(1)`, more precisely at the inverse Ackermann level.
- Space: `O(n)`.

## Common Pitfalls

- Not finding the roots first before calling `union`.
- Having `find` return only the parent node instead of recursively reaching the representative.
- Decreasing the connected-component count even on a duplicate union.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

Disjoint set union uses `parent` and `size/rank`. `find` applies path compression, and `union` first finds the two roots, then attaches the smaller set under the larger set.

```text
find(x):
  if parent[x] != x:
    parent[x] = find(parent[x])
  return parent[x]

union(a, b):
  ra, rb = find(a), find(b)
  if ra == rb: return false
  if size[ra] < size[rb]: swap(ra, rb)
  parent[rb] = ra
  size[ra] += size[rb]
  return true
```

Returning `false` means the two points were already in the same set, so the connected-component count must not be decreased again.

</details>

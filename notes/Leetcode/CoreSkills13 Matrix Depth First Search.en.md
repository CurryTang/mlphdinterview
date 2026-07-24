# Matrix Depth-First Search

## Interview Goal

Master DFS on a 2D grid, commonly used for number of islands, connected components, path search, and flood fill.

## Core Template

- Traverse every cell, and when you encounter an unvisited target cell, start DFS.
- DFS checks boundaries, visited state, and cell type.
- Move in four directions: up, down, left, and right.
- You can use recursion, or use an explicit stack to avoid recursion depth issues.

## Complexity

- Time: `O(mn)`
- Space: `O(mn)`, for the worst-case recursion stack or `visited`.

## Common Pitfalls

- Not marking `visited` immediately after visiting, which causes repeated revisits.
- Swapping row and column boundaries.
- Whether diagonals count as adjacent depends on the problem definition.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

The DFS function handles only one cell: first do out-of-bounds and `visited` checks, then mark it, and finally recurse in four directions.

```text
dfs(r, c):
  if r/c out of bounds: return
  if visited[r][c] or grid[r][c] is blocked: return
  visited[r][c] = true
  for (dr, dc) in directions:
    dfs(r + dr, c + dc)
```

If the problem asks for the number of connected components, the outer loop traverses every cell; when it encounters a new target cell, increment the count by one and start DFS.

</details>

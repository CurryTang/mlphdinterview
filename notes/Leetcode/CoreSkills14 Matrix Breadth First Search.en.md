# Matrix Breadth-First Search

## Interview Goal

Master BFS on a 2D grid, commonly used for shortest steps, multi-source diffusion, and level-by-level propagation.

## Core Template

- Use a queue to store the current frontier.
- Mark `visited` immediately when enqueuing.
- Each level represents an increase of one in distance.
- In multi-source BFS, enqueue all starting points together at the beginning.

## Complexity

- Time: `O(mn)`
- Space: `O(mn)`.

## Common Pitfalls

- Marking `visited` only when dequeuing causes the same cell to be enqueued repeatedly.
- Counting levels at the wrong position.
- Forgetting to initialize all source nodes in multi-source BFS.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

BFS expands level by level with a queue. Mark immediately when enqueuing to avoid the same cell being enqueued repeatedly by multiple parent nodes.

```text
queue = all start cells
mark all starts visited
dist = 0
while queue not empty:
  repeat len(queue) times:
    r, c = pop front
    push unvisited valid neighbors
  dist += 1
```

For single-source shortest path, return the level count when the destination is reached for the first time; for multi-source diffusion, treat all source nodes as level `0`.

</details>

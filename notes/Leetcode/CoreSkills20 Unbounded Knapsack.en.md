# Unbounded Knapsack

## Interview Goal

Master unbounded knapsack: each item can be chosen infinitely many times, and maximize value or count solutions under a capacity limit.

## State Design

- `dp[c]` represents the best result for capacity `c`.
- For each item `(weight, value)`, traverse capacity from small to large.
- Maximum-value transition: `dp[c] = max(dp[c], dp[c - weight] + value)`.

## Why Forward Order

Traversing capacity in forward order allows the current item to be reused within the same round, which matches unlimited selection.

## Complexity

- Time: `O(nW)`
- Space: `O(W)`.

## Common Pitfalls

- Confusing the loop direction with 0/1 knapsack.
- For combination-counting and permutation-counting problems, the loop order is different.
- Initialization must be adjusted based on whether the goal is maximum value, feasibility, or counting.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

Unbounded knapsack allows the same item to be chosen repeatedly, so capacity is traversed in forward order, allowing the newly updated `dp[c-weight]` from the current round to continue participating in transitions.

```text
dp = [0] * (W + 1)
for weight, value in items:
  for c in range(weight, W + 1):
    dp[c] = max(dp[c], dp[c - weight] + value)
return dp[W]
```

If the problem asks for the number of solutions, you still need to clarify first whether it asks for combinations or permutations; the loop order affects the counting semantics.

</details>

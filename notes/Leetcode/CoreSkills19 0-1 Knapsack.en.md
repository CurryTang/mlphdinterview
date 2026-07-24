# 0 / 1 Knapsack

## Interview Goal

Master 0/1 knapsack: each item can be chosen at most once, and maximize value under a capacity limit.

## State Design

- `dp[c]` represents the maximum value obtainable with capacity `c`.
- For each item `(weight, value)`, capacity must be traversed from large to small.
- Transition: `dp[c] = max(dp[c], dp[c - weight] + value)`.

## Why Reverse Order

Traversing capacity in reverse prevents the same item from being reused within the current round. Forward order turns it into the unbounded-knapsack effect.

## Complexity

- Time: `O(nW)`
- Space: `O(W)`, where `W` is the capacity.

## Common Pitfalls

- Writing the capacity loop in the wrong direction.
- Initializing in a way that does not match whether the problem requires filling the capacity exactly.
- Not skipping when `weight` exceeds the capacity.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

In 1D DP, `dp[c]` represents the maximum value with capacity not exceeding `c`. Each item can be used only once, so capacity must be traversed in reverse.

```text
dp = [0] * (W + 1)
for weight, value in items:
  for c in range(W, weight - 1, -1):
    dp[c] = max(dp[c], dp[c - weight] + value)
return dp[W]
```

Reverse order guarantees that `dp[c - weight]` is still the state after processing the previous round of items, so the current item will not be reused.

</details>

# 0 / 1 Knapsack

## 面试目标

掌握 0/1 背包：每个物品最多选一次，在容量限制下最大化价值。

## 状态设计

- `dp[c]` 表示容量为 c 时能得到的最大价值。
- 对每个物品 `(weight, value)`，容量需要从大到小遍历。
- 转移：`dp[c] = max(dp[c], dp[c - weight] + value)`。

## 为什么倒序

倒序容量可以避免同一物品在当前轮被重复使用。正序会变成完全背包效果。

## 复杂度

- 时间：`O(nW)`
- 空间：`O(W)`，W 是容量。

## 常见坑

- 容量循环方向写反。
- 初始化不符合题目是否要求刚好装满。
- weight 大于容量时没有跳过。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

一维 DP 中 `dp[c]` 表示容量不超过 `c` 的最大价值。每个物品只能用一次，所以容量倒序。

```text
dp = [0] * (W + 1)
for weight, value in items:
  for c in range(W, weight - 1, -1):
    dp[c] = max(dp[c], dp[c - weight] + value)
return dp[W]
```

倒序保证 `dp[c - weight]` 还是上一轮物品处理完的状态，不会重复使用当前物品。

</details>

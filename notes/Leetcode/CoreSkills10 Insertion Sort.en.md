# Insertion Sort

## Interview Goal

Understand the locally ordered idea behind insertion sort: keep the left side sorted, then insert the current element into the correct position.

## Core Process

1. Start iterating from index 1.
2. Save the current value `key`.
3. Move larger elements backward from the end of the sorted left portion.
4. Place `key` into the vacated position.

## Complexity

- Best case: `O(n)`, when the array is already sorted.
- Average/worst case: `O(n^2)`.
- Space: `O(1)`.
- Stability: stable.

## Common Pitfalls

- Not saving `key` before overwriting it.
- A `while` condition that causes an out-of-bounds access.
- Writing the comparison as `>=`, which breaks stability.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

The left side `[0, i)` always remains sorted. Take out `arr[i]`, shift larger elements to the right as a block, and finally place `key` into the empty slot.

```text
for i in range(1, n):
  key = arr[i]
  j = i - 1
  while j >= 0 and arr[j] > key:
    arr[j + 1] = arr[j]
    j -= 1
  arr[j + 1] = key
```

Use `>` in the comparison condition, not `>=`, so equal elements do not swap and the sort remains stable.

</details>

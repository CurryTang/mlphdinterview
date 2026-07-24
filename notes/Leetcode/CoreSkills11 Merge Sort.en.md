# Merge Sort

## Interview Goal

Master divide-and-conquer sorting: recursively split the array, sort the left and right halves, then merge them linearly.

## Core Process

1. If the interval length is less than or equal to 1, return directly.
2. Recursively sort the left half and the right half.
3. Use two pointers to merge the two sorted arrays.
4. Write the merged result back to the original array or return a new array.

## Complexity

- Time: `O(n log n)`
- Space: `O(n)`
- Stability: Stable, depending on preferring the left side first when equal elements are merged.

## Common Pitfalls

- `mid` computation and interval boundaries are not handled consistently.
- Forgetting to copy the remaining elements after merging.
- In-place merging is complicated; in interviews, write the extra-array version first.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

Recursively sort the left and right halves, then merge them with two pointers. When elements are equal during merging, preferring the left side preserves stability.

```text
mergeSort(arr):
  if len(arr) <= 1: return arr
  mid = len(arr) // 2
  left = mergeSort(arr[:mid])
  right = mergeSort(arr[mid:])
  return merge(left, right)

merge(left, right):
  use two pointers, append smaller one
  append remaining suffix
```

If the problem requires sorting the original array, write the temporary array back to the original interval at the end.

</details>

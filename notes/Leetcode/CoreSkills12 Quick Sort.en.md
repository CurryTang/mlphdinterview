# Quick Sort

## Interview Goal

Master the partition step of quicksort: choose a pivot, place elements smaller than the pivot on one side, and elements larger than the pivot on the other side.

## Core Process

1. Choose a pivot.
2. Reorder the interval with two pointers or Lomuto/Hoare partition.
3. Recursively sort the left and right sides of the pivot.
4. Using a random pivot can reduce the chance of worst-case degeneration.

## Complexity

- Average time: `O(n log n)`
- Worst-case time: `O(n^2)`
- Recursive stack space: average `O(log n)`, worst-case `O(n)`.
- Stability: Usually unstable.

## Common Pitfalls

- The recursive interval after partition still includes the pivot, causing infinite recursion.
- With many duplicate elements, performance can degenerate; three-way quicksort can be used.
- Always choosing the first element on an already sorted array can easily degenerate.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

Partition first, then recursively process both sides. The Lomuto version is the easiest to explain clearly: `store` points to the next position for an element smaller than the pivot.

```text
quickSort(l, r):
  if l >= r: return
  p = partition(l, r)
  quickSort(l, p - 1)
  quickSort(p + 1, r)

partition(l, r):
  pivot = arr[r]
  store = l
  for i in [l, r):
    if arr[i] < pivot:
      swap(arr[i], arr[store])
      store += 1
  swap(arr[store], arr[r])
  return store
```

In a real interview, you can first choose a random pivot and swap it to `r` to reduce the chance of degeneration.

</details>

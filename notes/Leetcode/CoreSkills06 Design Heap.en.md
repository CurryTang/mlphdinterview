# Design Heap

## Interview Goal

Implement a heap and master the array representation of a complete binary tree, bubble-up, bubble-down, and priority queue operations.

## Core Design

- In a min-heap, the parent value is no greater than its children.
- Array index relationships: `parent=(i-1)//2`, `left=2*i+1`, `right=2*i+2`.
- Insertion: place the element at the end, then bubble it up.
- Delete the heap top: move the last element to the root, then bubble it down.

## Complexity

- peek: `O(1)`
- push/pop: `O(log n)`
- heapify: `O(n)`

## Common Pitfalls

- Failing to choose the smaller child during bubble-down.
- Forgetting to return the original heap top after `pop`.
- Not handling operations on an empty heap.

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

An array represents a complete binary tree. After insertion, compare upward with the parent; after deleting the root, move the last element to the root and then swap downward with the smaller child.

```text
push(x):
  data.append(x)
  i = last index
  while i > 0 and data[i] < data[parent(i)]:
    swap(i, parent(i))
    i = parent(i)

pop():
  ans = data[0]
  data[0] = data.pop()
  heapify_down(0)
  return ans
```

`heapify_down` chooses the smaller of the left and right children each time, and swaps only if that child is smaller than the current node.

</details>

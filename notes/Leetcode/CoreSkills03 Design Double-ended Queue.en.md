# Design Double-ended Queue

## Interview Goal

Implement a deque that supports insertion and deletion at both the front and the back. Common implementations use either a circular array or a doubly linked list.

## Core Design

- A circular array maintains `front`, `size`, and `capacity`.
- The tail position can be computed as `(front + size) % capacity`.
- `pushFront` updates `front = (front - 1 + capacity) % capacity`.
- `popBack` only decreases `size`; no elements need to be moved.

## Complexity

- Insertion and deletion at both ends: `O(1)`
- Random access, if `get(i)` is implemented: `O(1)`
- Copying during resize: `O(n)`, but insertion is still amortized `O(1)` afterward.

## Common Pitfalls

- Negative modulo causing `front` to become negative.
- An off-by-one error when computing the tail index.
- Failing to rearrange elements into logical order starting from 0 after resizing.

## Extended Application: Monotonic Queue

A deque is not only used to practice implementing data structures. It can also maintain a monotonic set of candidate values: new elements enter from the back, and expired elements leave from the front. This pattern can compute the maximum value of every fixed-size window in linear time.

For the full derivation and code, see [[CoreSkills29 Sliding Window|Sliding Window Problem 5: Sliding Window Maximum]].

## Reference Solution

<details class="solution">
<summary>Expand Solution</summary>

The circular-array version only stores `front` and `size`. A logical index `i` maps to the physical index `(front + i) % capacity`.

```text
pushFront(x):
  grow if full
  front = (front - 1 + capacity) % capacity
  data[front] = x
  size += 1

pushBack(x):
  grow if full
  data[(front + size) % capacity] = x
  size += 1
```

When resizing, copy `data[(front+i)%old_capacity]` into position `i` of the new array in logical order, then reset `front` to 0.

</details>

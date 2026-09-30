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

For the full derivation and code, see [[CoreSkills16 Sliding Window|Sliding Window Problem 5: Sliding Window Maximum]].

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

---

## Practical Application: Trailing Window Maximum (Streaming Window with Warm-up)

The most prominent high-frequency application of a double-ended queue in quantitative trading, streaming systems, and online feature engineering is using a **Monotonic Deque** to maintain rolling extremums.

### Problem Definition

> **Trailing / Rolling Window Maximum**:
> Given a time series of real numbers `xs` (length $N$) and a maximum lookback window size $n$.
> At each timestamp $t \in [0, N-1]$, the window spans $[\max(0, t - n + 1), \; t]$ (effective window size $\le n$).
> Output the maximum value in the trailing window at every timestamp $t$. Return a list of length $N$.

```python
def rolling_max(xs: List[float], n: int) -> List[float]: ...
```

### Key Differences from Standard LeetCode 239 (Fixed Window)

| Dimension | Standard LeetCode 239 (Sliding Window Maximum) | Trailing / Rolling Window Maximum (Streaming) |
|---|---|---|
| **Output Timing** | Waits until window reaches full size $k$ ($t \ge k - 1$) | **Emits an output at every timestamp $t$** (including warm-up) |
| **Window Span** | Strictly fixed at size $k$ | First $n-1$ points grow dynamically from $1$ to $n$ (Warm-up phase) |
| **Output Length** | $N - k + 1$ | Strictly equals input length $N$ |
| **Industry Scenario**| Batch slice processing | Real-time risk engines, causal quantitative trading signals (no lookahead bias) |

### Invariants & Operational Lifecycle

The deque stores **only timestamp indices $t$**, preserving two core invariants:
1. **Strictly Increasing Timestamp**: Indices from front to back are monotonically increasing;
2. **Strictly Decreasing Value**: Values corresponding to indices are strictly descending; the front element $q[0]$ is always the index of the maximum value in the active window.

At each timestamp $t$ with incoming value $val = xs[t]$:
1. **Front Expiry**:
   The active window is $[t - n + 1, t]$. Any index $\le t - n$ has slid out of view and is popped: `while q and q[0] <= t - n: q.popleft()`;
2. **Back Domination (Purging Inferiors)**:
   If the element at the back $xs[q[-1]] \le val$, it arrived earlier and has a smaller or equal value. It can never become the maximum in any future window. Pop it: `while q and xs[q[-1]] <= val: q.pop()`;
3. **Push Current Index**:
   `q.append(t)`;
4. **Immediate Recording**:
   $xs[q[0]]$ is the maximum of the current valid window, so append `res.append(float(xs[q[0]]))`.

### Production-Grade Implementation

```python
from collections import deque
from typing import List

def rolling_max(xs: List[float], n: int) -> List[float]:
    """
    Trailing window maximum with window size <= n.
    
    Parameters:
        xs: Input list of floats representing a time series of length N
        n:  Maximum lookback window size (n >= 1)
        
    Returns:
        A list of floats of length N where res[t] is max(xs[max(0, t - n + 1) : t + 1])
    """
    if not xs or n <= 0:
        return []

    q = deque()  # stores indices with values monotonically decreasing
    res: List[float] = []

    for t, val in enumerate(xs):
        # 1. Purge expired indices from the front: valid range is [t - n + 1, t]
        while q and q[0] <= t - n:
            q.popleft()

        # 2. Purge dominated candidates from the back
        while q and xs[q[-1]] <= val:
            q.pop()

        # 3. Append current timestamp index
        q.append(t)

        # 4. Front element is the maximum of window [max(0, t - n + 1), t]
        res.append(float(xs[q[0]]))

    return res

if __name__ == "__main__":
    data = [1.0, 3.0, -1.0, -3.0, 5.0, 3.0, 6.0, 7.0]
    expected = [1.0, 3.0, 3.0, 3.0, 5.0, 5.0, 6.0, 7.0]
    assert rolling_max(data, 3) == expected
    print("✅ rolling_max test passed!")
```

### Complexity Analysis

- **Time Complexity**: Strictly $\mathcal{O}(N)$. Each index $t \in [0, N-1]$ is pushed into the deque exactly once and popped at most once from either end. The amortized number of deque operations across the entire stream is at most $2N$.
- **Space Complexity**: The deque holds at most $\min(n, N)$ indices at any given moment, yielding $\mathcal{O}(\min(n, N))$ auxiliary space.


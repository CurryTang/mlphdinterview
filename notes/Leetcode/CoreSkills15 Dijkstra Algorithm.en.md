# Shortest Path: Dijkstra and Bellman-Ford with an Edge-Count Limit

## Interview Goal

For shortest-path problems, do not start by memorizing a template. First determine the constraints:

- If all edge weights are non-negative and there are no extra restrictions: prioritize Dijkstra.
- If there are negative edges, or the path length / number of edges has an explicit limit: consider Bellman-Ford.
- If the graph is unweighted: BFS is the shortest-path algorithm.
- If the state includes dimensions beyond the node, such as "how many transfers have been used": expand the state to `(node, state)`.

This section uses **Cheapest Flights Within K Stops** as the example, focusing on an optimized Bellman-Ford. It is not finding an arbitrary shortest path among all paths, but the shortest path under the constraint of "at most `k` stops, that is, at most `k + 1` edges."

## Dijkstra Review

Dijkstra applies to graphs with non-negative weights. The core idea is to use a min-heap to pop the state with the smallest current cost each time, then relax its neighbors.

```text
dist[start] = 0
heap = [(0, start)]

while heap:
  d, u = heappop(heap)
  if d != dist[u]:
    continue
  for v, w in graph[u]:
    if d + w < dist[v]:
      dist[v] = d + w
      heappush(heap, (dist[v], v))
```

For Cheapest Flights, standard Dijkstra is not enough, because "the lowest price to the same city" is not necessarily the final best state. A more expensive state that used fewer flights may still be feasible later. Therefore, the Dijkstra formulation must expand the state to `(cost, city, stopsUsed)`.

## Why Bellman-Ford Is More Natural

The semantics of Bellman-Ford fit this problem exactly:

> After the `i`-th relaxation round, `prices[x]` represents the lowest price to reach city `x` using at most `i` edges.

The problem allows at most `k` stops, which means at most `k + 1` flights. Therefore, we only need to perform `k + 1` relaxation rounds.

The key point is that each round must read from the previous round's `prices` and write into `nextPrices`. If you update in place directly, multiple edges will be chained together within the same round, which is equivalent to secretly using more flights than the current round allows.

## Example Problem: Cheapest Flights Within K Stops

Input:

```text
n = 4
flights = [
  [0, 1, 100],
  [1, 2, 100],
  [2, 0, 100],
  [1, 3, 600],
  [2, 3, 200],
]
src = 0
dst = 3
k = 1
```

At most `1` stop is allowed, so at most `2` flights can be taken. The valid answer is `0 -> 1 -> 3`, with total price `700`. Is the path `0 -> 1 -> 2 -> 3` cheaper? It costs `400`, but it requires `3` flights, which exceeds the limit, so it cannot be used.

```bellman-demo
cheapest-flights
```

## Optimized Bellman-Ford

The basic formulation is to perform a fixed `k + 1` rounds. There are two optimization points:

1. In each round, use `nextPrices = prices.copy()` to guarantee that this round only transitions from the previous round's states.
2. If a round makes no updates at all, further relaxations cannot improve anything, so we can stop early.

```python
from typing import List

class Solution:
    def findCheapestPrice(
        self,
        n: int,
        flights: List[List[int]],
        src: int,
        dst: int,
        k: int,
    ) -> int:
        INF = float("inf")
        prices = [INF] * n
        prices[src] = 0

        for _ in range(k + 1):
            next_prices = prices.copy()
            changed = False

            for start, end, cost in flights:
                if prices[start] == INF:
                    continue

                candidate = prices[start] + cost
                if candidate < next_prices[end]:
                    next_prices[end] = candidate
                    changed = True

            prices = next_prices
            if not changed:
                break

        return -1 if prices[dst] == INF else prices[dst]
```

## Correctness Intuition

At the start of round `0`, only `src` has price `0`, representing that "without taking any flight, you can only reach the starting point."

Round `1` only allows transitions from the results of round `0`, so it can only reach cities accessible with one flight.

Round `2` only allows transitions from the results of round `1`, so it reaches cities accessible with at most two flights.

Continuing until round `k + 1` exactly covers the maximum number of flights allowed by the problem. Because each round reads the old array and writes the new array, continuous chaining such as `0 -> 1 -> 2 -> 3` cannot occur within the same round.

## Complexity

- Time: `O((k + 1) * E)`, where `E` is the number of flights.
- Space: `O(V)`, keeping only `prices` and `next_prices`.

This is usually more direct than pushing all `(city, stops)` states into a heap, and it is also better suited to explaining the constraint of "at most how many edges."

## Common Pitfalls

- Treating `k stops` as at most `k` edges. In reality it means at most `k + 1` flights.
- Updating `prices` in place, causing multiple edges to be chained within the same round.
- Forgetting to skip flights where `prices[start] == INF`.
- Returning early using the current price of `dst`; Bellman-Ford must wait until the current round of relaxation is complete.
- Using standard Dijkstra's `dist[city]` to discard a state that is more expensive but uses fewer stops.

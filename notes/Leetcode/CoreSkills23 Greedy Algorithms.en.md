# Greedy Algorithms: From Forced Choices to Invariants

## The Core Test for Greedy

Greedy is not "do whatever feels best at each step." Truly reliable greedy algorithms usually come from two types of signals:

1. **Forced choice**: the current smallest, earliest, or most urgent element has no other legal place to go.
2. **Dominance relation**: after making a certain local choice, the future does not get worse, and may even become easier.

When explaining Greedy in an interview, it is better not to say only "intuitively this should work." A better phrasing is:

```text
I first identify an element that must be handled.
In any legal solution, this element can only appear in a certain way.
So handling it now does not discard any feasible solution.
After handling it, the remaining problem has the same shape as the original one.
```

That is the idea behind an exchange argument / forced move.

## Hand of Straights: Consecutive Groups

Problem:

```text
Given hand and groupSize.
Can all cards be divided into several groups, where each group has length groupSize,
and every group consists of consecutive integers?
```

Example:

```text
hand = [1, 2, 3, 6, 2, 3, 4, 7, 8]
groupSize = 3

It can be divided into:
[1, 2, 3]
[2, 3, 4]
[6, 7, 8]

answer = true
```

Counterexample:

```text
hand = [1, 2, 3, 4, 5]
groupSize = 4

The total number of cards, 5, is not divisible by 4.
answer = false
```

## Breakthrough: The Smallest Card Has No Choice

If the current smallest remaining card is `x`, where can it go?

It cannot be the second card of some straight:

```text
x - 1, x, x + 1, ...
```

because `x - 1` is smaller than `x`, but there are no smaller cards left now.

It also cannot be the third card or the fourth card. The reason is the same: that would require smaller predecessor cards.

So:

> The current smallest remaining card `x` must be the start of some straight.

This is the greedy choice in this problem.

If there are `count[x]` copies of `x`, then all `count[x]` copies must be starts. So afterward we must simultaneously have:

```text
count[x] copies of x + 1
count[x] copies of x + 2
...
count[x] copies of x + groupSize - 1
```

If any one of those cards is insufficient, the answer is immediately `False`.

## Why We Can Subtract count[x] Copies at Once

Suppose the current smallest card is `x`, and:

```text
count[x] = 3
groupSize = 4
```

All 3 copies of `x` must start 3 different straights:

```text
x, x+1, x+2, x+3
x, x+1, x+2, x+3
x, x+1, x+2, x+3
```

So instead of subtracting one group at a time, we can subtract in bulk:

```text
count[x] -= 3
count[x + 1] -= 3
count[x + 2] -= 3
count[x + 3] -= 3
```

This is not an optimization trick. It is a direct consequence of the correctness of the greedy argument:

```text
The smallest card has no room for choice, so all of its copies are forced to start new straights.
```

## Standard Solution: Counter + Min Heap

Algorithm flow:

1. If `len(hand) % groupSize != 0`, return `False` immediately.
2. Use `Counter` to count the frequency of each card.
3. Put all distinct card values into a min-heap so we can find the current smallest card.
4. Each time we inspect the heap top `first`:
   - If `count[first] == 0`, it means it has already been fully consumed by previous straights, so pop it.
   - Otherwise it is the current smallest card that is not fully processed yet, and it must start new straights.
5. Let `need = count[first]`, and subtract `need` from every card from `first` to `first + groupSize - 1`.
6. If any card does not have enough copies, return `False`.

Code:

```python
from collections import Counter
from heapq import heapify, heappop
from typing import List

class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False

        count = Counter(hand)
        min_heap = list(count)
        heapify(min_heap)

        while min_heap:
            first = min_heap[0]

            if count[first] == 0:
                heappop(min_heap)
                continue

            need = count[first]
            for card in range(first, first + groupSize):
                if count[card] < need:
                    return False
                count[card] -= need

        return True
```

## Key Details in This Code

### 1. Why Check Divisibility First

Each group has fixed length `groupSize`, and all cards must be used up.

So the total number of cards must satisfy:

```text
len(hand) % groupSize == 0
```

Otherwise there is no need to enter the greedy logic.

### 2. Why the Heap May Contain Cards with Count 0

For example:

```text
hand = [1, 2, 3, 2, 3, 4]
groupSize = 3
```

When processing `1`, we subtract:

```text
1, 2, 3
```

This may cause `2` or `3` to become `0` early. But they still remain in the heap.

So each time we inspect the heap top, we must skip cards that are already exhausted:

```python
if count[first] == 0:
    heappop(min_heap)
    continue
```

### 3. Why We Do Not Need to Literally Build One Group at a Time

If `first` has `need` copies, then we must create `need` groups starting from `first`.

Subtracting in bulk is more direct than creating them one by one:

```python
for card in range(first, first + groupSize):
    count[card] -= need
```

What this expresses is:

```text
Settle all straights that start with first in one shot.
```

## Correctness Proof

Maintain an invariant:

```text
At the start of each while-loop iteration, all cards smaller than the heap top have already been consumed by legal groups.
```

Now consider the current smallest remaining card `first`.

Because there are no cards smaller than `first`, in any legal solution `first` cannot appear in the middle or at the end of a straight.

Therefore it must be the start of a straight.

If `first` has `need` copies, then we must open `need` straights. Each straight requires:

```text
first, first + 1, ..., first + groupSize - 1
```

So each of those card values must have at least `need` copies.

If any one of them is insufficient, then no solution can possibly exist.

If they are all sufficient, we subtract them. After subtraction, all copies of `first` have been used legally, and the remaining cards are still the same problem:

```text
Can the remaining cards be divided into consecutive groups of fixed length?
```

So we just continue processing the new smallest card.

When all cards have been subtracted away, it means every forced choice succeeded, so the answer is `True`.

## Complexity

Let:

```text
n = hand.length
u = number of distinct card values
g = groupSize
```

Counter requires:

```text
O(n)
```

Building the heap requires:

```text
O(u)
```

Each distinct card value is popped from the heap at most once:

```text
O(u log u)
```

The bulk subtraction part scans at most `groupSize` cards for each "straight start", so in the worst case it can be written as:

```text
O(u * groupSize)
```

Because `u <= n`, in an interview you can usually say:

```text
Time:  O(n log n)
Space: O(n)
```

More precisely:

```text
Time:  O(n + u log u + u * groupSize)
Space: O(u)
```

## Another Style: Sort the Distinct Values and Scan

If you do not want to use a heap, you can also directly sort all distinct card values.

The idea is the same:

```text
Process each card value from small to large.
If count[x] > 0, it must start count[x] straights.
```

Code:

```python
from collections import Counter
from typing import List

class Solution:
    def isNStraightHand(self, hand: List[int], groupSize: int) -> bool:
        if len(hand) % groupSize != 0:
            return False

        count = Counter(hand)

        for first in sorted(count):
            need = count[first]
            if need == 0:
                continue

            for card in range(first, first + groupSize):
                if count[card] < need:
                    return False
                count[card] -= need

        return True
```

This version is shorter and also works very well in interviews.

Complexity:

```text
Time:  O(n + u log u + u * groupSize)
Space: O(u)
```

## Walk Through It Manually

Input:

```text
hand = [1, 2, 3, 6, 2, 3, 4, 7, 8]
groupSize = 3
```

Frequencies:

```text
1:1, 2:2, 3:2, 4:1, 6:1, 7:1, 8:1
```

The current smallest card is `1`:

```text
need = 1
subtract 1, 2, 3
```

Remaining:

```text
2:1, 3:1, 4:1, 6:1, 7:1, 8:1
```

The current smallest card is `2`:

```text
need = 1
subtract 2, 3, 4
```

Remaining:

```text
6:1, 7:1, 8:1
```

The current smallest card is `6`:

```text
need = 1
subtract 6, 7, 8
```

Everything is used up, so return `True`.

## Common Pitfalls

- Forgetting to first check whether the total length is divisible by `groupSize`.
- Subtracting only one `first`, instead of subtracting all `count[first]` copies in bulk.
- Forgetting to skip stale heap tops where `count[first] == 0` when using a heap.
- Mistakenly thinking the smallest card can appear in the middle of a straight. The smallest card has no predecessor, so it can only be the start.
- Using a normal list and calling `pop(0)` each time to find the smallest value, causing unnecessary `O(n^2)` overhead.
- Returning false only when `count[card] == 0`, even though `count[card] < need` is already insufficient.

## Interview Answer Template

You can explain it like this:

1. First check whether `len(hand)` is divisible by `groupSize`.
2. Use Counter to count the number of each card.
3. The greedy point is: the current smallest remaining card cannot come after any smaller card, so it must be the start of a straight.
4. If the smallest card `first` has `need` copies, then we must open `need` straights starting from `first`.
5. Therefore each of `first, first + 1, ..., first + groupSize - 1` must have at least `need` copies.
6. If not enough, return `False`; if enough, subtract in bulk.
7. Keep processing until all cards are used up, then return `True`.

One-sentence summary:

> The Greedy in Hand of Straights is not "just start from the smallest at random," but rather "the smallest card has no predecessor, so it is forced to start from the smallest."

# Sliding Window · Universal Template & Core Interview Archetypes

The sliding window technique fundamentally operates as a **two-pointer model maintaining a dynamic closed interval $[\text{left}, \text{right}]$ over a 1D sequence**. All sliding window problems share an identical loop skeleton. Any sliding window problem can be solved systematically by answering **"Three Decision Questions"** and filling **"Three Code Slots"**:

```text
[Three Decision Questions & Code Slots]
1. state & add_right: What incremental state is maintained? How does right entering update it?
2. shrink condition : When should left advance to restore the invariant? (while for variable, if for fixed)
3. record answer    : When should the answer be recorded? (after shrinking for longest, inside shrinking for shortest, when full for fixed)
```

---

## Original Problems and Learning Order

This note selects 8 high-frequency problems categorized into four archetypes: **Longest Valid Window**, **Shortest Satisfying Window**, **Fixed-Size Window**, and **Monotonic Deque Window**:

| Order | Original Problem | Window Archetype | Core Maintained State | Shrink Mechanism | Record Timing |
|---:|---|---|---|---|---|
| 1 | [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) | Longest Valid Window | Character frequency map | `while` duplicate exists | Update `max` after shrink |
| 2 | [1004. Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/description/) | Longest Valid Window | Zero counter `zeros` | `while zeros > k` | Update `max` after shrink |
| 3 | [424. Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/description/) | Longest Valid Window | Frequency map + `max_freq` | `while len - max_freq > k` | Update `max` after shrink |
| 4 | [209. Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/description/) | Shortest Satisfying Window | Running sum `sum` | `while sum >= target` | Update `min` inside shrink |
| 5 | [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/description/) | Shortest Satisfying Window | `need/window` + `have` | `while have == required` | Update `min` inside shrink |
| 6 | [567. Permutation in String](https://leetcode.com/problems/permutation-in-string/description/) | Fixed-Size Window | Two 26-element frequency arrays | `if len > k` | Compare at full size and return bool |
| 7 | [438. Find All Anagrams in a String](https://leetcode.com/problems/find-all-anagrams-in-a-string/description/) | Fixed-Size Window | Two 26-element frequency arrays | `if len > k` | Collect matching `left` indices at full size |
| 8 | [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/description/) | Fixed-Size Window | Decreasing monotonic index deque | `if len > k` | Read front element maximum at full size |

Compare how the core patterns map to the unified skeleton:

```sliding-window-patterns
```

---

## First, Fix the Window Invariant

The window interval is mathematically modeled as a closed interval:

$$
[\text{left}, \text{right}], \qquad \text{length} = \text{right} - \text{left} + 1.
$$

### Why the Closed Interval Length Requires `+ 1`

Both `left` and `right` point to valid elements currently included inside the window. The difference $\text{right} - \text{left}$ measures the **index step distance** between the two pointers. The number of elements contained in the closed interval must also count the starting index `left` itself, requiring the additional $+ 1$:

```text
Array Indices:  2   3   4
Window Items:  [A   B   C]
Pointers:      left = 2, right = 4
Index Gap:     4 - 2 = 2
Element Count: 4 - 2 + 1 = 3
```

Boundary check: When a single-element window has `left == right`, the closed interval length is $\text{right} - \text{left} + 1 = 1$; omitting $+ 1$ would incorrectly yield a length of 0.

| Window Convention | Includes Boundary `right`? | Length Formula | Usage |
|---|---|---|---|
| **Closed Interval $[\text{left}, \text{right}]$** | Yes (both ends inclusive) | $\text{right} - \text{left} + 1$ | **Standard algorithmic convention (used throughout this note)** |
| Half-open Interval $[\text{left}, \text{right})$ | No (left inclusive, right exclusive) | $\text{right} - \text{left}$ | C++ STL iterators / slicing `s[left:right]` |

### Strict Timing Between State Mutation and Pointer Advance

In sliding window algorithms, state updates and pointer movements must be atomically synchronized. When evicting an element from the left, you must **decrement/remove the element from the state first, before advancing the pointer**:

```python
# Correct order: revert state first, then increment pointer
remove_left(state, items[left])
left += 1

# Fatal bug: advancing pointer first deletes the wrong subsequent element!
# left += 1
# remove_left(state, items[left])
```

The interactive demo below demonstrates the four-phase cycle (expand, maintain, shrink, record) for the "Longest Valid Window" archetype:

```sliding-window-demo
```

---

## Universal Template: Three Questions & Three Slots Framework

What all sliding window problems truly share is not an arbitrary line of code, but a structured mental decision pipeline:

```text
                    ┌────────────────────────────┐
                    │  for right, item in ...:   │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 1. Enter: add_right(item)  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ 2. Is window size fixed k? │
                    └──────┬──────────────┬──────┘
                           │ Yes          │ No
                           ▼              ▼
         ┌────────────────────────┐  ┌─────────────────────────────────┐
         │ if len > k:            │  │ while shrink condition met:     │
         │   remove_left(left)    │  │   [Shortest: record min ans]    │
         │   left += 1            │  │   remove_left(left)             │
         └─────────────┬──────────┘  │   left += 1                     │
                       │             └─────────────┬───────────────────┘
                       │                           │
                       ▼                           ▼
         ┌────────────────────────┐  ┌─────────────────────────────────┐
         │ if len == k:           │  │ [Longest: update after shrink]  │
         │   record_answer(...)   │  │ ans = max(ans, right-left+1)    │
         └────────────────────────┘  └─────────────────────────────────┘
```

### 1. Universal Skeleton Code

```python
def universal_sliding_window(items, k=None):
    left = 0
    state = initialize_state()
    answer = initialize_answer()

    for right, item in enumerate(items):
        # Slot 1: Add new element and update state (add_right)
        add_right(state, item)

        # Slot 2: Window shrinkage control (shrink)
        if k is not None:
            # Branch A: Fixed-Size Window
            if right - left + 1 > k:
                remove_left(state, items[left])
                left += 1

            # Slot 3: Record answer when window reaches fixed size k
            if right - left + 1 == k:
                record_fixed(answer, state, left, right)
        else:
            # Branch B: Variable-Size Window
            # Pattern I: Shortest Satisfying Window (e.g., LC 209, LC 76)
            while is_satisfied(state):
                record_min(answer, left, right)   # Record min before evicting
                remove_left(state, items[left])
                left += 1

            # Pattern II: Longest Valid Window (e.g., LC 3, LC 1004, LC 424)
            # while is_invalid(state):
            #     remove_left(state, items[left]) # Evict until restored to valid
            #     left += 1
            # record_max(answer, left, right)     # Update max once valid
    return answer
```

### 2. Decision Matrix for the Three Archetypes

| Window Archetype | Classic Problems | Shrink Condition (`shrink`) | Control Statement | Record Answer Timing (`record`) | Core Principle |
|---|---|---|---|---|---|
| **Longest Valid Window** | LC 3, LC 1004, LC 424 | Window violates rules (**invalid**) | `while invalid:` | **After `while` loop completes** (when restored to valid) | Expand while valid, shrink when invalid, record max outside loop |
| **Shortest Satisfying Window** | LC 209, LC 76 | Window **satisfies** target goal | `while satisfied:` | **Inside `while` loop** (before each eviction) | Shrink when satisfied to minimize, record min inside loop |
| **Fixed-Size Window** | LC 567, LC 438, LC 239 | Window length **exceeds $k$** | `if len > k:` | **When window length equals $k$** | At most 1 element evicted per turn, evaluate at size $k$ |

---

## 1. Longest Substring Without Repeating Characters

### Problem Description

Given a string `s`, find the length of the **longest continuous substring** without duplicate characters. $s$ length is up to $5\times10^4$.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `count[char]`, hash map tracking character frequencies in the current window |
| `add_right` | `count[char] += 1` |
| `shrink` Condition | `while count[char] > 1` (current incoming character creates a duplicate) |
| `remove_left` | `count[s[left]] -= 1`; `left += 1` |
| `record` Timing | After `while` finishes (window is valid without duplicates), `ans = max(ans, right - left + 1)` |

```python
from collections import defaultdict


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        count = defaultdict(int)
        left = 0
        answer = 0

        for right, char in enumerate(s):
            count[char] += 1

            # Slot 2: When duplicate occurs, shrink until valid
            while count[char] > 1:
                count[s[left]] -= 1
                left += 1

            # Slot 3: Record max length once window is restored to valid
            answer = max(answer, right - left + 1)

        return answer
```

```longest-substring-demo
```

### Complexity Analysis

The outer pointer `right` advances $n$ times. The inner pointer `left` only advances monotonically to the right, incrementing at most $n$ times in total. Each character enters the window once and exits at most once. Amortized time complexity is $O(n)$, and space complexity is $O(|\Sigma|) \le O(128) = O(1)$.

---

## 2. Longest Repeating Character Replacement

### Problem Description

Given a string `s` consisting of uppercase English letters and an integer `k`. You can choose at most `k` characters and replace them with any other uppercase character. Return the length of the longest substring containing the same letter you can achieve. $s$ length is up to $10^5$.

Whether a window can be converted into a uniform character string with at most $k$ replacements depends solely on the **window length** and the **frequency of the most frequent character**:

$$
\text{replacements} = \text{window\_length} - \text{max\_frequency} \le k.
$$

We simply preserve the highest-frequency character and replace all other characters.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `count[char]` and historical peak frequency `max_freq` |
| `add_right` | `count[char] += 1`; `max_freq = max(max_freq, count[char])` |
| `shrink` Condition | `while right - left + 1 - max_freq > k` (required replacements exceed budget $k$) |
| `remove_left` | `count[s[left]] -= 1`; `left += 1` |
| `record` Timing | Once valid, update `ans = max(ans, right - left + 1)` |

```python
from collections import defaultdict


class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = defaultdict(int)
        left = 0
        max_freq = 0
        answer = 0

        for right, char in enumerate(s):
            count[char] += 1
            max_freq = max(max_freq, count[char])

            # Slot 2: Shrink window when replacements exceed k
            while right - left + 1 - max_freq > k:
                count[s[left]] -= 1
                left += 1

            # Slot 3: Record max window length
            answer = max(answer, right - left + 1)

        return answer
```

### Why `max_freq` Does Not Need to Decrement During Shrink

This is a classic **historical proof-of-work optimization**:
We only care whether a window can beat the historical maximum length. If after shrinking the actual highest character frequency in the current window decreases, keeping `max_freq` at its historical peak causes no false positives:
1. If no incoming character ever breaks this historical peak, the window can never exceed our already discovered optimum anyway.
2. Only when an incoming character's frequency strictly surpasses `max_freq` will the window legitimately expand to record a larger answer.

Thus, allowing `max_freq` to remain monotonic non-decreasing preserves mathematical correctness while eliminating the $O(26)$ search on each shrink.

---

## 3. Permutation in String

### Problem Description

Given two strings `s1` and `s2`, return `True` if `s2` contains a permutation of `s1`. A permutation requires: **identical length and identical character frequencies across all 26 lowercase letters**. This is a canonical **Fixed-Size Window** problem where window size is fixed at $k = \text{len}(s1)$.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | Two 26-element arrays: `need[26]` and `window[26]` |
| `add_right` | `window[ord(char) - ord('a')] += 1` |
| `shrink` Condition | `if right - left + 1 > k:` (exceeds length $k$, single eviction) |
| `remove_left` | `window[ord(s2[left]) - ord('a')] -= 1`; `left += 1` |
| `record` Timing | `if right - left + 1 == k and window == need: return True` |

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        if len(s1) > len(s2):
            return False

        need = [0] * 26
        window = [0] * 26

        for char in s1:
            need[ord(char) - ord('a')] += 1

        k = len(s1)
        left = 0

        for right, char in enumerate(s2):
            window[ord(char) - ord('a')] += 1

            # Slot 2: Fixed-size window eviction (at most 1 element)
            if right - left + 1 > k:
                window[ord(s2[left]) - ord('a')] -= 1
                left += 1

            # Slot 3: Evaluate comparison when window reaches full size k
            if right - left + 1 == k and window == need:
                return True

        return False
```

Comparing two 26-element arrays takes $O(26)$. Total time complexity is strictly $O(26 \cdot n) = O(n)$.

---

## 4. Minimum Window Substring

### Problem Description

Given strings `s` and `t`, return the **minimum window substring** of `s` that covers all characters in `t` (including duplicate frequencies). If no such substring exists, return `""`.

### Type Count Compression: Achieving $O(1)$ Validity via `have`

Directly comparing frequency dictionaries on each step takes $O(|\Sigma|)$. Instead, introduce a match counter `have`:
- `required = len(need)`: total number of **distinct characters** in `t`.
- `have`: number of distinct characters in the current window whose frequency has reached or exceeded `need[c]`.
- The window is valid if and only if:

$$
\text{have} == \text{required}.
$$

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `need`, `window`, `have`, `required` |
| `add_right` | `window[c] += 1`; if `window[c] == need[c]` then `have += 1` |
| `shrink` Condition | `while have == required:` (target satisfied, shrink to minimize length) |
| `record` Timing | **Inside `while` loop before evicting**: record candidate minimum first |
| `remove_left` | If `window[old] == need[old]` then `have -= 1`; `window[old] -= 1`; `left += 1` |

```python
from collections import Counter, defaultdict


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ""

        need = Counter(t)
        window = defaultdict(int)
        required = len(need)
        have = 0

        left = 0
        best_start = 0
        best_len = float('inf')

        for right, char in enumerate(s):
            window[char] += 1
            if char in need and window[char] == need[char]:
                have += 1

            # Slot 2: Window satisfies target, shrink to find minimum
            while have == required:
                length = right - left + 1
                # Slot 3: Record optimal candidate before evicting left
                if length < best_len:
                    best_start = left
                    best_len = length

                old = s[left]
                if old in need and window[old] == need[old]:
                    have -= 1
                window[old] -= 1
                left += 1

        if best_len == float('inf'):
            return ""
        return s[best_start:best_start + best_len]
```

---

## 5. Sliding Window Maximum

### Problem Description

Given an integer array `nums` and a sliding window size `k`. The window slides from left to right one position at a time. Return the max sliding window. $n \le 10^5$.

Naive search takes $O(nk)$. A max heap takes $O(n\log k)$. Using a **Double-Ended Monotonic Queue (Monotonic Deque)** achieves optimal $O(n)$ time complexity.

### Monotonic Deque Mathematical Invariants

The deque **stores array indices only** (indices are required to determine whether an entry has expired beyond the left boundary), maintaining two strict mathematical invariants:

1. **Index Monotonicity**: Indices in the deque strictly increase from head to tail:
   $$q[0] < q[1] < \dots < q[m-1]$$
2. **Value Monotonicity**: Corresponding values strictly decrease from head to tail:
   $$\text{nums}[q[0]] > \text{nums}[q[1]] > \dots > \text{nums}[q[m-1]]$$
3. **Extremum Property**: **The front element $q[0]$ is unconditionally the maximum of the current window**, accessible in $O(1)$ time.

### Queue Elimination Mechanics (Why Tail and Head Evictions are Safe)

- **Tail Elimination (Domination Principle)**:
  When examining incoming element $\text{nums}[\text{right}]$, if deque tail $\text{tail} = q[-1]$ satisfies $\text{nums}[\text{tail}] \le \text{nums}[\text{right}]$, pop $\text{tail}$ from the back (`pop()`).
  *Proof*: For any current or future window containing $\text{right}$, index $\text{tail}$ will expire earlier, and its value is no greater than $\text{nums}[\text{right}]$. Thus $\text{tail}$ can **never again become a window maximum** (it is both older and weaker). It is permanently dominated and can be discarded.
- **Head Expiration (Expiration Principle)**:
  When the window advances its left boundary to $\text{left}$, if the front element $q[0] < \text{left}$ (or in fixed window logic $q[0] == \text{left}$), the old maximum has fallen outside the window boundary. It must be popped from the front (`popleft()`).

```sliding-window-max-demo
```

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | Double-ended queue storing indices with strictly decreasing values: `candidates = deque()` |
| `add_right` | Pop all dominated smaller tail elements, then append `right` |
| `shrink` Condition | `if right - left + 1 > k:` (fixed-size window exceeds $k$) |
| `remove_left` | If the front maximum expired (`candidates[0] == left`), pop it (`popleft()`); `left += 1` |
| `record` Timing | `if right - left + 1 == k:` append `nums[candidates[0]]` to answer |

```python
from collections import deque
from typing import List


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        candidates = deque()
        answer = []
        left = 0

        for right, value in enumerate(nums):
            # Slot 1: Add right. Pop weaker tail candidates to preserve strict monotonic decrease
            while candidates and nums[candidates[-1]] <= value:
                candidates.pop()
            candidates.append(right)

            # Slot 2: Fixed window shrink. Evict front if it expired
            if right - left + 1 > k:
                if candidates[0] == left:
                    candidates.popleft()
                left += 1

            # Slot 3: When window size reaches k, read front maximum
            if right - left + 1 == k:
                answer.append(nums[candidates[0]])

        return answer
```

### Why the Nested `while` is Strictly Amortized $O(n)$

Analyzing the macro lifetime of each index $i \in [0, n-1]$:
- Push: `candidates.append(i)` executes exactly once per index.
- Pop: Either evicted from tail by a larger element (at most once) or evicted from head due to expiration (at most once).

Total push and pop operations across the entire execution cannot exceed $2n$. Thus time complexity is strictly $O(n)$, with auxiliary space $O(k)$.

---

## 6. Minimum Size Subarray Sum

### Problem Description

Given an array of $n$ **positive integers** `nums` and a positive integer `target`. Return the minimal length of a contiguous subarray whose sum is greater than or equal to `target`. If there is no such subarray, return 0.

This is the quintessential baseline for mastering the **Shortest Satisfying Window** archetype, sharing the exact same shrink and record timing as LC 76.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `curr_sum`, running sum of elements in the window |
| `add_right` | `curr_sum += nums[right]` |
| `shrink` Condition | `while curr_sum >= target:` (target reached, shrink left to minimize length) |
| `record` Timing | **Inside `while` loop before evicting**: `ans = min(ans, right - left + 1)` |
| `remove_left` | `curr_sum -= nums[left]`; `left += 1` |

```python
from typing import List


class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        curr_sum = 0
        ans = float('inf')

        for right, val in enumerate(nums):
            # Slot 1: Add element to window sum
            curr_sum += val

            # Slot 2: Target met, shrink left to find minimal length
            while curr_sum >= target:
                # Slot 3: Record min length inside loop before evicting
                ans = min(ans, right - left + 1)
                curr_sum -= nums[left]
                left += 1

        return 0 if ans == float('inf') else ans
```

### In-Depth Theory: Why Must Elements be Strictly Positive?

Sliding window algorithms require **Monotonicity**:
- When all elements are positive, advancing `right` strictly increases `curr_sum`, and advancing `left` strictly decreases `curr_sum`.
- If the array contains **negative numbers**, expanding the window might decrease the sum, and shrinking the window might increase the sum. Monotonicity collapses, causing standard two-pointer sliding window to fail!
- **Solution for Negative Numbers**: To find the shortest subarray with sum $\ge target$ on general arrays with negatives (such as [LC 862. Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)), one must use **Prefix Sums + Monotonic Deque** in $O(n)$ time.

---

## 7. Max Consecutive Ones III

### Problem Description

Given a binary array `nums` and an integer `k`, return the maximum number of consecutive `1`'s in the array if you can flip at most `k` `0`'s.

### Problem Abstraction: Translating to Longest Valid Window

"Flipping at most $k$ zeros" is equivalent to: **Find the longest contiguous subarray containing at most $k$ zeros**.
- Valid Condition: $\text{zeros} \le k$.
- Invalid Condition: $\text{zeros} > k$.

### Filling the Template Slots

| Slot | Concrete Implementation |
|---|---|
| `state` | `zeros`, count of zero elements within the window |
| `add_right` | `if nums[right] == 0: zeros += 1` |
| `shrink` Condition | `while zeros > k:` (zero count exceeds flip budget $k$) |
| `remove_left` | `if nums[left] == 0: zeros -= 1`; `left += 1` |
| `record` Timing | After `while` loop (restored to valid), `ans = max(ans, right - left + 1)` |

```python
from typing import List


class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = 0
        zeros = 0
        ans = 0

        for right, val in enumerate(nums):
            # Slot 1: Update zero count
            if val == 0:
                zeros += 1

            # Slot 2: Shrink when zero count exceeds budget k
            while zeros > k:
                if nums[left] == 0:
                    zeros -= 1
                left += 1

            # Slot 3: Record max length once restored to valid
            ans = max(ans, right - left + 1)

        return ans
```

---

## 8. Find All Anagrams in a String

### Problem Description

Given two strings `s` and `p`, return an array of all the **start indices** of `p`'s anagrams in `s`.

This is isomorphic to LC 567 (Fixed-Size Window of length $k = \text{len}(p)$). The only difference is that LC 567 returns boolean `True` on the first match, whereas LC 438 appends `left` to a results list on every match.

### Filling the Template Slots

```python
from typing import List


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(p) > len(s):
            return []

        need = [0] * 26
        window = [0] * 26

        for char in p:
            need[ord(char) - ord('a')] += 1

        k = len(p)
        left = 0
        result = []

        for right, char in enumerate(s):
            # Slot 1: Increment frequency of incoming character
            window[ord(char) - ord('a')] += 1

            # Slot 2: Fixed-size window eviction
            if right - left + 1 > k:
                window[ord(s[left]) - ord('a')] -= 1
                left += 1

            # Slot 3: Collect match index at size k
            if right - left + 1 == k and window == need:
                result.append(left)

        return result
```

---

## 9. Advanced Master Technique: Exact K via Dual Sliding Window

When confronting problems like [LC 992. Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/) or [LC 1248. Count Number of Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/), a standard sliding window gets stuck because **"exactly $k$" is non-monotonic**: expanding the window can enter or leave the exact $k$ state intermittently.

### Dimension Reduction: The Identity Decomposition

Convert the non-monotonic "exactly $k$" problem into the difference of two **strictly monotonic "at most $k$" problems**:

$$
\text{Exact}(k) = \text{atMost}(k) - \text{atMost}(k - 1).
$$

For an "at most $k$ distinct elements" window, if $[\text{left}, \text{right}]$ is valid, then **all subarrays ending at $\text{right}$ with start $\ge \text{left}$ (exactly $\text{right} - \text{left} + 1$ subarrays) are unconditionally valid**!

```python
from collections import defaultdict
from typing import List


class Solution:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def atMost(k_max: int) -> int:
            if k_max == 0:
                return 0
            count = defaultdict(int)
            left = 0
            distinct = 0
            total = 0

            for right, num in enumerate(nums):
                if count[num] == 0:
                    distinct += 1
                count[num] += 1

                while distinct > k_max:
                    count[nums[left]] -= 1
                    if count[nums[left]] == 0:
                        distinct -= 1
                    left += 1

                # Number of valid subarrays ending at right with <= k_max distinct elements
                total += right - left + 1

            return total

        return atMost(k) - atMost(k - 1)
```

---

## Complete Problem Matrix & Slot Comparison

| Problem | LC # | Window Archetype | `state` Tracked | Shrink Condition & Control | Record Timing & Formula |
|---|---|---|---|---|---|
| **Longest Substring** | LC 3 | Longest Valid | Char frequencies | `while count[c] > 1` | Outside loop: `ans = max(ans, len)` |
| **Max Consecutive Ones III** | LC 1004 | Longest Valid | Zero count `zeros` | `while zeros > k` | Outside loop: `ans = max(ans, len)` |
| **Character Replacement** | LC 424 | Longest Valid | Frequencies + `max_freq` | `while len - max_freq > k` | Outside loop: `ans = max(ans, len)` |
| **Min Size Subarray Sum** | LC 209 | Shortest Satisfying | Running sum `sum` | `while sum >= target` | Inside loop: `ans = min(ans, len)` |
| **Minimum Window Substring** | LC 76 | Shortest Satisfying | `need/window` + `have` | `while have == required` | Inside loop: `ans = min(ans, len)` |
| **Permutation in String** | LC 567 | Fixed-Size | 26-elem frequency array | `if len > k` (evict 1) | At size $k$: compare and return `True` |
| **Find All Anagrams** | LC 438 | Fixed-Size | 26-elem frequency array | `if len > k` (evict 1) | At size $k$: append matching `left` |
| **Sliding Window Maximum** | LC 239 | Fixed-Size | Monotonic index deque | `if len > k` (evict head) | At size $k$: read `nums[deque[0]]` |

---

## Common Pitfalls & Debugging Checklist

| Common Mistake | Root Cause & Failure Mechanism | Correct Practice |
|---|---|---|
| **Using `if` instead of `while` in variable window** | An incoming element can cause multiple violations, requiring continuous eviction | Always use `while` for variable window invariant restoration |
| **Using `while` blindly in fixed window** | A fixed window adds 1 element per step, exceeding by at most 1 | Use `if right - left + 1 > k` to clearly express fixed-window mechanics |
| **Omitting `+ 1` in window length** | Confuses interval distance with element cardinality | Closed interval element count is strictly `right - left + 1` |
| **Recording longest window before shrinking** | Window is currently in an invalid violation state, polluting the optimal answer | For longest valid windows, update `max` **after** `while` loop completes |
| **Recording shortest window after shrinking** | The window may no longer satisfy the target after eviction, missing the minimal valid state | For shortest satisfying windows, update `min` **inside** `while` before eviction |
| **Advancing `left` before mutating state** | Pointer increment shifts `left` to the subsequent element, corrupting state synchronization | Always subtract `items[left]` from state **first**, then `left += 1` |
| **Storing values instead of indices in deque** | Values alone cannot indicate when an element has fallen outside window bounds | Monotonic queues must store **indices**, referencing values as needed |

---

## How to Apply the Template to New Problems

Follow this 4-step deduction checklist on any new sliding window problem:

```text
Step 1: Is the window length fixed?
        -> Fixed length k: Control with `if len > k`, record at size k.
        -> Variable length: Proceed to Step 2 and 3.

Step 2: What is the optimization goal?
        -> "Longest valid": while invalid shrink, update max outside loop.
        -> "Shortest satisfying": while satisfied shrink, update min inside loop.
        -> "Exact K count": Decompose into `atMost(K) - atMost(K-1)`.

Step 3: How is state incrementally updated and reverted?
        -> Entering items[right]: What state variables can be maintained in O(1)?
        -> Evicting items[left]: How can the state be reversed symmetrically?

Step 4: Verify the monotonicity assumption!
        -> Does moving pointers monotonically change the validity metrics?
        -> If negative values exist (e.g. subarray sum with negatives), sliding window fails;
           switch to Prefix Sums + Monotonic Deque / Hash Map.
```

---

## Template Quick-Memorization Card

```text
[Universal 3-Slot Mantra]
Add right to state upon entry,
Shrink left to maintain the boundary,
Record longest after loop is done,
Record shortest before eviction run,
Record fixed window when size is won!
```

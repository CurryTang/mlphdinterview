# Review Flashcards: Intervals, Sliding Window & Backtracking

This note is the third volume of the high-frequency algorithmic interview review flashcards: systematically organizing **Intervals & Sweep Line**, **Two Pointers & Sliding Window**, and **Backtracking & Digit Greedy** with production-grade implementations and asymptotic complexity breakdowns.

---

## Module 1: Intervals & Sweep Line

### 1. Merge Intervals & Interval Topology Variants

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Interval 01</span>
  <span class="review-card-title">Merge Intervals & Interval Topology Variants</span>
  <span class="review-card-tag">Closed Interval Semantics · Presorted Acceleration · Nested Absorption · Insert Interval · Non-Overlapping Greedy · Meeting Rooms Sweep Line</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

```python
from typing import List

class IntervalSolution:
    @staticmethod
    def merge(intervals: List[List[int]]) -> List[List[int]]:
        if not intervals: return []
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for i in range(1, len(intervals)):
            cur = intervals[i]
            if cur[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], cur[1])
            else:
                merged.append(cur)
        return merged
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- $\mathcal{O}(N \log N)$ sorting, $\mathcal{O}(1)$ space.

</div>

</div>
</details>

---

## Module 2: Two Pointers & Sliding Window

### 2. Pythagorean Triplet & Multiplicity 2-Pointer Search

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Two Pointers 02</span>
  <span class="review-card-title">Pythagorean Triplet & Multiplicity 2-Pointer Search</span>
  <span class="review-card-tag">Square Mapping · Multiplicity Preservation · 3SUM Reduction · Two Pointers Convergence · O(N^2) Optimality Proof</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

```python
from typing import List

class PythagoreanTripletSolution:
    @staticmethod
    def judgePythagoreanTriplet(nums: List[int]) -> bool:
        n = len(nums)
        if n < 3: return False
        squares = sorted([x * x for x in nums])
        for k in range(n - 1, 1, -1):
            target = squares[k]
            i, j = 0, k - 1
            while i < j:
                cur_sum = squares[i] + squares[j]
                if cur_sum == target:
                    return True
                elif cur_sum < target:
                    i += 1
                else:
                    j -= 1
        return False
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity</div>

- $\mathcal{O}(N^2)$ time. In the comparison model, no $\mathcal{O}(N^{2-\varepsilon})$ algorithm exists by 3SUM conjecture.

</div>

</div>
</details>

---

### 3. Minimum Window Substring & Multi-Candidate Expansion

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Window 03</span>
  <span class="review-card-title">Minimum Window Substring & Multi-Candidate Expansion</span>
  <span class="review-card-tag">Variable Window · O(|S| + |T|) Rigorous Proof · k-Factor Frequency · All Tied Minimum Windows · Fixed-Size Array Constant Bound</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Follow-ups</div>

Given strings $S$ and $T$, find the shortest substring in $S$ containing every character in $T$, respecting multiplicities:
- **Linear complexity proof**: Explain $\mathcal{O}(|S| + |T|)$.
- **k-factor follow-up**: Every character in $T$ must appear at least $k$ times (or $k$ times its required frequency).
- **All tied minimum windows**: Return **every valid substring tied for the minimum length**, rather than only one.
- **Fixed array bound**: Fixed-size counter array (ASCII 128) guarantees strict constant lookup without hash collision overhead.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Invariants</div>

- `formed_kinds` increments strictly when a character count first matches target requirement; extra copies do not re-increment.
- **Contract and record before removal**: When valid, update the answer list before incrementing `left`, because removing the leftmost character can immediately invalidate the window.
- When finding a strictly smaller window, clear the answer list; append on ties.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List
from collections import Counter

class MinWindowSolution:

    @classmethod
    def minWindowAll(cls, s: str, t: str, k_scale: int = 1) -> List[str]:
        """
        Collects all tied shortest substrings containing T (scaled by k_scale).
        Time: O(|S| + |T|), Space: O(1) with 128-byte array.
        """
        if not s or not t or k_scale <= 0:
            return []

        target_counts = Counter(t)
        req = [0] * 128
        for ch, count in target_counts.items():
            req[ord(ch)] = count * k_scale

        required_kinds = len(target_counts)
        formed_kinds = 0

        win = [0] * 128
        min_len = float('inf')
        ans_list: List[str] = []

        left = 0
        s_len = len(s)

        for right in range(s_len):
            r_code = ord(s[right])
            win[r_code] += 1

            if req[r_code] > 0 and win[r_code] == req[r_code]:
                formed_kinds += 1

            while left <= right and formed_kinds == required_kinds:
                cur_len = right - left + 1

                if cur_len < min_len:
                    min_len = cur_len
                    ans_list = [s[left : right + 1]]
                elif cur_len == min_len:
                    ans_list.append(s[left : right + 1])

                l_code = ord(s[left])
                win[l_code] -= 1
                if req[l_code] > 0 and win[l_code] < req[l_code]:
                    formed_kinds -= 1

                left += 1

        return ans_list
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(|S| + |T|)$. Each character enters and leaves the window at most once.
- **Space Complexity**: $\mathcal{O}(1)$ with fixed-size ASCII array.
- **Critical Pitfalls**: Recording answers after incrementing `left` loses valid boundary data.

</div>

</div>
</details>

---

### 4. Longest Repeating Character Replacement & Max-Frequency Invariant

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Window 04</span>
  <span class="review-card-title">Longest Repeating Character Replacement & Max-Frequency Invariant</span>
  <span class="review-card-tag">Sliding Window · Max-Frequency Invariant · Non-Decreasing Window Size · O(N) Single-Pass</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Examples</div>

Given an uppercase string `s` and integer `k`, replace at most `k` characters to form the longest substring with identical characters (LC 424):
- `s = "ABAB", k = 2` $\implies 4$
- `s = "AABABBA", k = 1` $\implies 4$

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & Max-Frequency Monotonicity</div>

- Window condition: `(length - max_freq) <= k`.
- **Key Insight**: `max_freq` does NOT need to be recomputed when shrinking `left`! A smaller frequency cannot yield a larger valid window than the historical best.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
class CharacterReplacementSolution:

    @staticmethod
    def characterReplacement(s: str, k: int) -> int:
        counts = [0] * 26
        left = max_freq = max_len = 0

        for right, ch in enumerate(s):
            idx = ord(ch) - ord('A')
            counts[idx] += 1
            max_freq = max(max_freq, counts[idx])

            while (right - left + 1) - max_freq > k:
                counts[ord(s[left]) - ord('A')] -= 1
                left += 1

            max_len = max(max_len, right - left + 1)

        return max_len
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: Strictly $\mathcal{O}(N)$.
- **Space Complexity**: $\mathcal{O}(1)$ for 26 letters.

</div>

</div>
</details>

---

### 5. Variable & Fixed Sliding Window Patterns

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Window 05</span>
  <span class="review-card-title">Variable & Fixed Sliding Window Patterns</span>
  <span class="review-card-tag">Longest Substring Without Repeating Characters · Hash Jump Optimization</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

```python
class SlidingWindowSolution:
    @staticmethod
    def lengthOfLongestSubstring(s: str) -> int:
        last_seen = {}
        max_len = left = 0
        for right, ch in enumerate(s):
            if ch in last_seen:
                left = max(left, last_seen[ch] + 1)
            last_seen[ch] = right
            max_len = max(max_len, right - left + 1)
        return max_len
```

</div>

</div>
</details>

---

## Module 3: Backtracking & Digit Greedy

### 6. Subsets, Permutations & Combinations

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Backtrack 06</span>
  <span class="review-card-title">Subsets, Permutations & Combinations</span>
  <span class="review-card-tag">State-Space Tree · Pruning Deduplication</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Implementation</div>

```python
from typing import List

class BacktrackSolution:
    @staticmethod
    def subsetsWithDup(nums: List[int]) -> List[List[int]]:
        nums.sort()
        res, path = [], []
        def backtrack(start: int):
            res.append(list(path))
            for i in range(start, len(nums)):
                if i > start and nums[i] == nums[i - 1]: continue
                path.append(nums[i])
                backtrack(i + 1)
                path.pop()
        backtrack(0)
        return res
```

</div>

</div>
</details>

---

### 7. Largest Number Smaller than N from Digits A (Digit Greedy Backtracking)

<details class="review-card" open>
<summary class="review-card-summary">
  <span class="review-card-badge">Digit 07</span>
  <span class="review-card-title">Largest Number Smaller than N from Digits A (Digit Greedy Backtracking)</span>
  <span class="review-card-tag">Digit Backtracking · Greedy Prefix Match · Downgrade Suffix Max Fill · Shorter Length Fallback</span>
</summary>
<div class="review-card-content">

<div class="review-block">
<div class="review-block-label">📌 Problem Definition & Examples</div>

Given a positive integer $N$ and a set of decimal digits $A \subseteq \{0, \dots, 9\}$ reusable indefinitely:
- Construct the largest integer strictly smaller than $N$ using only digits from $A$.
- Example: $N = 23415, A = [2, 4, 9] \implies 22999$.
- Example: $N = 222, A = [2] \implies 22$.

</div>

<div class="review-block">
<div class="review-block-label">💡 Intuition & State Machine</div>

- Sort $A$ ascending.
- At position $i$:
  1. Try $d == N[i]$, recurse.
  2. If recursion fails, pick largest $d < N[i]$, and immediately fill all remaining suffix positions with $\max(A)$.
  3. If no same-length candidate exists, fallback to length $L - 1$ filled entirely with $\max(A)$.

</div>

<div class="review-block">
<div class="review-block-label">💻 Production-Grade Implementations</div>

```python
from typing import List

class DigitConstructionSolution:

    @classmethod
    def findLargestSmaller(cls, N: int, A: List[int]) -> int:
        if N <= 0 or not A:
            return -1

        digits = sorted(list(set(A)))
        max_d = digits[-1]
        s_N = str(N)
        L = len(s_N)
        res_digits: List[int] = []

        def backtrack(idx: int, is_less: bool) -> bool:
            if idx == L:
                return is_less

            cur_target = int(s_N[idx])

            if is_less:
                res_digits.append(max_d)
                if backtrack(idx + 1, True):
                    return True
                res_digits.pop()
                return False

            for d in reversed(digits):
                if d == cur_target:
                    res_digits.append(d)
                    if backtrack(idx + 1, False):
                        return True
                    res_digits.pop()
                elif d < cur_target:
                    res_digits.append(d)
                    if backtrack(idx + 1, True):
                        return True
                    res_digits.pop()

            return False

        if backtrack(0, False):
            return int("".join(map(str, res_digits)))

        if L > 1:
            if max_d == 0:
                return -1
            return int(str(max_d) * (L - 1))

        return -1
```

</div>

<div class="review-block">
<div class="review-block-label">⏱️ Complexity & Common Pitfalls</div>

- **Time Complexity**: $\mathcal{O}(L \cdot |A|)$ where $L = \operatorname{len}(str(N))$.
- **Space Complexity**: $\mathcal{O}(L)$ recursion depth.
- **Critical Pitfalls**: Forgetting shorter-length fallback (e.g. $N=222, A=[3] \implies 33$).

</div>

</div>
</details>

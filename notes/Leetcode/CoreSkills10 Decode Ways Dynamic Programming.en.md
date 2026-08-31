# Dynamic Programming: From Recurrence to Space Optimization

## Module 1: Core DP Mental Model & 6 Skeletons Taxonomy

### 1. Interview Goal & Mindset Refactoring

Dynamic Programming (DP) is never about "memorizing templates." Its fundamental essence is: **decomposing a complex, large-scale decision problem into overlapping isomorphic subproblems, and computing/caching answers along their topological dependency order**.

A robust, reusable 3-step DP formulation for whiteboard interviews consists of:

```text
Step 1: Mathematically Define State ➔ Clarify the physical meaning of dp[...] & Base Cases
Step 2: Enumerate Decision Branches (Transition) ➔ Formulate recurrence & determine Iteration Order
Step 3: Analyze Dependency Radius (Space Optimization) ➔ Apply rolling variables / array space reduction
```

> [!IMPORTANT]
> **Golden Interview Rule**: In whiteboard coding interviews, clearly communicating the State Definition, Base Cases, and Transitions to the interviewer *before* writing code is far more critical than jumping straight to implementation. Once the recurrence is rigorous, writing code is mere mechanical translation.

---

### 2. Identifying DP Signals

Consider DP whenever a problem exhibits the following signals:

- **Optimization/Counting Goal**: Asks for "number of ways / minimum cost / maximum profit / feasibility (True/False)".
- **Staged Decisions**: Progresses in stages (e.g., prefix of `i` characters, first `i` items, day `i`, interval `[i, j]`).
- **Overlapping Subproblems**: Naive recursive search repeatedly explores identical subproblems with identical arguments.
- **Optimal Substructure & No Aftereffect (Markov Property)**: The optimal solution to the current state depends strictly on optimal solutions to smaller subproblems, completely independent of the historical path taken to achieve them.

The four pillars of Dynamic Programming:

$$\text{DP Formula} = \mathbf{State} + \mathbf{Transition} + \mathbf{Base\ Case} + \mathbf{Iteration\ Order}$$

---

### 3. Deep Dive into the 3-Step Workflow

#### Step 1: Formulate Recurrence (State & Transition)
Before writing any loops, define `dp[...]` with one unambiguous sentence:
- `dp[i]`: Answer for prefix `arr[:i]` or suffix starting at index `i`.
- `dp[i][j]`: Answer for double sequence prefixes `(i, j)` or closed interval `[i, j]`.
- `dp[i][state]`: Answer at stage `i` under a discrete named state (e.g., holding stock / cooldown).

Combine previous sub-states according to the problem's objective:
- **Counting (Ways)**: $\text{dp}[i] = \sum \text{dp}[\text{prev}]$
- **Optimization (Min/Max)**: $\text{dp}[i] = \min / \max(\text{dp}[\text{prev}] + \text{cost})$
- **Feasibility (Boolean)**: $\text{dp}[i] = \bigvee \text{dp}[\text{prev}]$

#### Step 2: Determine Iteration Order from Dependencies
Loop directions are dictated solely by **topological dependencies**:
- If $dp[i]$ depends on $dp[i-1]:$ Traverse forward ($i: 1 \to n$).
- If $dp[i]$ depends on $dp[i+1]:$ Traverse backward ($i: n-1 \to 0$).
- If $dp[i][j]$ depends on $dp[i+1][j-1]$ (Interval DP): Traverse by increasing length ($L: 1 \to n$) or bottom-up ($i: n-1 \to 0$).

#### Step 3: Space Optimization
- Depends only on $dp[i-1]:$ Reduce to a single variable ($O(1)$ space).
- Depends on $dp[i-1], dp[i-2]:$ Reduce to two rolling variables (e.g., Fibonacci / House Robber / Decode Ways).
- 2D table $dp[i][j]$ depends only on row $dp[i-1][\dots]:$ Reduce to a 1D rolling array ($O(nm) \to O(m)$).

---

### 4. 6 DP Skeletons Taxonomy

```text
What is the structural dimension of the state?
│
├── 1. Single index i (prefix / suffix / linear progression)
│     └── Skeleton 1: 1D Linear DP
│         Key problems: Climbing Stairs, Min Cost Climbing Stairs, House Robber I & II,
│                       Decode Ways, Word Break, LIS, Maximum Product Subarray
│
├── 2. Two prefix pointers (i, j) across two sequences
│     └── Skeleton 2: Two Sequences & String Matching DP
│         Key problems: Longest Common Subsequence (LCS), Edit Distance,
│                       Interleaving String, Distinct Subsequences, Regular Expression Matching
│
├── 3. Closed interval [i, j] (substring / peeling onion / length expansion)
│     └── Skeleton 3: Interval DP
│         Key problems: Longest Palindromic Substring, Palindromic Substrings, Burst Balloons
│
├── 4. Items + Capacity (0/1 vs Complete, Combinations vs Permutations)
│     └── Skeleton 4: Knapsack Family
│         Key problems: Partition Equal Subset Sum (0/1 Feasibility), Target Sum (0/1 Count),
│                       Coin Change (Complete Min Coins), Coin Change II (Complete Combinations)
│
├── 5. 2D grid coordinates (r, c)
│     └── Skeleton 5: Grid & DAG DP
│         Key problems: Unique Paths (deterministic sweep),
│                       Longest Increasing Path in a Matrix (arbitrary directions ➔ Memoized DFS)
│
└── 6. Stage + discrete named states (holding / sold cooldown / free rest)
      └── Skeleton 6: State Machine DP
          Key problems: Best Time to Buy and Sell Stock with Cooldown
```

---

## Module 2: Skeleton 1 · 1D Linear DP & Prefix States

1D Linear DP represents the foundational cornerstone of dynamic programming, where states are defined over prefixes or suffixes of an array or string.

### 1. Deep Dive: Decode Ways

**Problem Statement**: A message containing letters from `A-Z` is encoded as `'A' -> 1, 'B' -> 2, ..., 'Z' -> 26`. Given a digit string `s`, compute the total number of valid decoding configurations.

#### Recurrence & Boundary Design
Define $dp[i]$ as the number of decoding ways for suffix string $s[i:]$.
- **Base Case**: $dp[n] = 1$ (empty suffix represents 1 valid completed decoding path).
- **Leading `'0'` Trap**: If $s[i] == '0'$, digit `'0'` cannot map to any letter on its own and leading zeros are invalid $\implies dp[i] = 0$.
- **1-Digit Choice**: Single digit $s[i] \in ['1'..'9'] \implies$ contributes $dp[i+1]$.
- **2-Digit Choice**: Two digits $s[i..i+1] \in [10..26] \implies$ additionally contributes $dp[i+2]$.

$$dp[i] = \begin{cases} 0 & s[i] = '0' \\ dp[i+1] + [10 \le s[i..i+1] \le 26] \cdot dp[i+2] & s[i] \neq '0' \end{cases}$$

#### Full Array to Space-Optimized Implementation

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0
            
        n = len(s)
        # Space Optimization: dp[i] only depends on dp[i+1] (one) and dp[i+2] (two)
        one = 1  # Corresponds to dp[i+1], initialized as dp[n] = 1
        two = 0  # Corresponds to dp[i+2]
        
        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                cur = 0
            else:
                cur = one
                if i + 1 < n and (s[i] == '1' or (s[i] == '2' and s[i + 1] in '0123456')):
                    cur += two
            two = one
            one = cur
            
        return one
```

- **Complexity**: Time $O(n)$, Space $O(1)$.
- **Pitfalls**: Leading `'0'` cannot decode; boundary check requires `i + 1 < n`.

---

### 2. Stair Climbing Family: Climbing Stairs & Min Cost Climbing Stairs

#### Climbing Stairs
- **State**: $dp[i]$ = Number of ways to reach step $i$.
- **Transition**: $dp[i] = dp[i-1] + dp[i-2]$ (Fibonacci isomorphism).
- **Implementation**:

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        a, b = 1, 2
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b
```

#### Min Cost Climbing Stairs
- **State**: $dp[i]$ = Minimum cumulative cost to reach step $i$ and pay `cost[i]` to step off.
- **Top Floor Semantic**: The top floor has no cost, so the final answer is $\min(dp[n-1], dp[n-2])$.
- **Transition**: $dp[i] = cost[i] + \min(dp[i-1], dp[i-2])$.

```python
class Solution:
    def minCostClimbingStairs(self, cost: list[int]) -> int:
        a, b = cost[0], cost[1]
        for i in range(2, len(cost)):
            a, b = b, cost[i] + min(a, b)
        return min(a, b)
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

### 3. Robbery Family: House Robber & House Robber II

#### House Robber I (Linear Street)
- **Constraint**: Cannot rob two adjacent houses on the same night.
- **State**: $dp[i]$ = Max money robbed considering first $i$ houses.
- **Transition**: $dp[i] = \max(dp[i-1], dp[i-2] + nums[i-1])$.

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        prev2, prev1 = 0, 0
        for x in nums:
            prev2, prev1 = prev1, max(prev1, prev2 + x)
        return prev1
```

#### House Robber II (Circular Street)
- **Circular Constraint**: House $0$ and House $n-1$ are adjacent and cannot both be robbed.
- **Dual Linear Subsegment Decomposition**:
  1. Case A: Skip last house $\implies$ Run linear Robber on `nums[0..n-2]`;
  2. Case B: Skip first house $\implies$ Run linear Robber on `nums[1..n-1]`;
  3. Global Max: $\max(\text{rob}(nums[0..n-2]), \text{rob}(nums[1..n-1]))$.

```python
class Solution:
    def rob(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]
            
        def rob_linear(arr: list[int]) -> int:
            a = b = 0
            for x in arr:
                a, b = b, max(b, a + x)
            return b
            
        return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

- **Complexity**: Time $O(n)$, Space $O(1)$.
- **Pitfall**: Do not forget the $n=1$ edge case.

---

### 4. String Prefix Partition: Word Break

#### Approach 1: Standard Set Hash Slice DP (Baseline)

- **State**: $dp[i]$ = Boolean flag indicating if prefix $s[:i]$ can be segmented into dictionary words.
- **Transition**: Enumerate the split point $j \in [0, i)$ for the last word:

$$dp[i] = \bigvee_{j=0}^{i-1} \bigl(dp[j] \land (s[j:i] \in \text{wordDict})\bigr)$$

```python
class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        words = set(wordDict)
        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Base case: empty prefix is trivially segmentable
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j] and s[j:i] in words:
                    dp[i] = True
                    break
        return dp[n]
```

- **Complexity**: Time $O(n^2 \cdot L)$ ($L$ is substring slice length), Space $O(n + \sum \text{len}(\text{words}))$.
- **Performance Bottleneck**:
  1. **Substring slicing overhead**: Each inner iteration generates a new substring `s[j:i]`, causing heap allocation and string copy ($O(i - j)$);
  2. **No prefix pruning**: Even if `s[j:j+2]` is not a prefix of any dictionary word, standard DP blindly continues enumerating all $j \dots i$.

---

#### Approach 2: Prefix Tree (Trie) + DP Optimization (Forward Matching & Instant Pruning)

In production text tokenization or large dictionary workloads, **indexing `wordDict` into a Trie coupled with forward-driven DP matching** achieves superior theoretical and benchmark performance:

- **Key Optimization Mechanisms**:
  1. **Trie Construction**: Insert all words from `wordDict` into a Trie, tracking `max_len`;
  2. **Forward Matching Driver**: Whenever $dp[i] == \text{True}$, start traversing the Trie forward matching characters $s[j]$ for $j \ge i$;
  3. **Instant Branch Pruning**: As soon as a character $s[j]$ is missing in the current Trie node's children, **break immediately** (no word in the dictionary can match any longer substring starting at $i$);
  4. **Zero Slicing Overhead**: Pure character pointer advancement with zero substring memory allocations.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Solution:
    def wordBreak(self, s: str, wordDict: list[str]) -> bool:
        # 1. Build Prefix Tree (Trie)
        root = TrieNode()
        max_len = 0
        for word in wordDict:
            node = root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.is_word = True
            max_len = max(max_len, len(word))

        n = len(s)
        dp = [False] * (n + 1)
        dp[0] = True  # Base case: empty prefix is valid

        # 2. Forward Trie Traversal with Instant Branch Pruning
        for i in range(n):
            if not dp[i]:
                continue  # Previous prefix is invalid, skip
            
            node = root
            # Traverse Trie forward matching characters starting from index i
            for j in range(i, min(n, i + max_len)):
                ch = s[j]
                if ch not in node.children:
                    break  # Key Pruning: Trie branch miss, terminate inner loop immediately!
                node = node.children[ch]
                if node.is_word:
                    dp[j + 1] = True

        return dp[n]
```

- **Comparative Architectural Matrix**:

| Dimension | Approach 1: Standard Set DP | Approach 2: Trie Forward DP |
|---|---|---|
| **String Slice Overhead** | Creates new `s[j:i]` objects ($O(L)$ allocation/copy) | **Zero allocation** (direct char traversal) |
| **Prefix Pruning** | None (blindly tests all $j \in [0, i)$) | **Instant pruning** on Trie branch mismatch |
| **Worst-case Time** | $O(n^2 \cdot L)$ | $O(n \cdot \min(n, L_{\max}) + \sum \text{len})$ |
| **Space Complexity** | $O(n + \sum \text{len})$ | $O(n + \Sigma \cdot \text{Nodes})$ |
| **Production Target** | Small dictionaries, short text | Large corpora, NLP tokenization, high-throughput |

---

### 5. Historical Sweep: Longest Increasing Subsequence (LIS)

- **State**: $dp[i]$ = Length of longest strictly increasing subsequence **ending strictly at $nums[i]$**.
- **Transition**: Scan all smaller predecessors $j < i$:

$$dp[i] = 1 + \max_{j < i, nums[j] < nums[i]} dp[j]$$
- **Global Answer**: $\max_{0 \le i < n} dp[i]$ (not necessarily at the last cell).

```python
class Solution:
    def lengthOfLIS(self, nums: list[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
```

- **Complexity**: Time $O(n^2)$, Space $O(n)$. (Note: Binary search / Patience sorting achieves $O(n \log n)$).

---

### 6. Sign Flipping & Dual Extrema: Maximum Product Subarray

- **Core Challenge**: Multiplying negative numbers flips signs; a local minimum (most negative) flips into a global maximum upon meeting another negative number!
- **Dual Extremum Tracking**: Maintain both `max_here` and `min_here` ending at index $i$:

$$\text{max\_here}' = \max(x, \text{max\_here} \cdot x, \text{min\_here} \cdot x)$$

$$\text{min\_here}' = \min(x, \text{max\_here} \cdot x, \text{min\_here} \cdot x)$$

```python
class Solution:
    def maxProduct(self, nums: list[int]) -> int:
        ans = max_here = min_here = nums[0]
        for x in nums[1:]:
            candidates = (x, max_here * x, min_here * x)
            max_here, min_here = max(candidates), min(candidates)
            ans = max(ans, max_here)
        return ans
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

## Module 3: Skeleton 2 · Two Sequences & String Matching DP

Two-sequence DP handles alignment, edit operations, interleaving, and subsequence matching between two strings `s1` and `s2`. States are indexed by $(i, j)$ representing prefixes `s1[:i]` and `s2[:j]`.

```text
2D Two-Sequence Dependency Topology:
         dp[i-1][j-1] (Diagonal: match / substitute) ────> dp[i-1][j] (Top: delete / skip s1)
              │                                                │
              ▼                                                ▼
         dp[i][j-1] (Left: insert / skip s2)         ────> dp[i][j] (Current State)
```

---

### 1. Alignment & Common Subsequence: Longest Common Subsequence (LCS)

- **State**: $dp[i][j]$ = Length of LCS between `text1[:i]` and `text2[:j]`.
- **Transition**:

$$dp[i][j] = \begin{cases} dp[i-1][j-1] + 1 & \text{text1}[i-1] == \text{text2}[j-1] \\ \max(dp[i-1][j], dp[i][j-1]) & \text{text1}[i-1] \neq \text{text2}[j-1] \end{cases}$$

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$ (rolling array reduces to $O(\min(m, n))$).
- **Pitfall**: When characters match, you must take the diagonal $+ 1$, not $\max(\text{top}, \text{left}) + 1$.

---

### 2. Tri-Directional Decision: Edit Distance

- **State**: $dp[i][j]$ = Minimum operations (insert, delete, replace) to convert `word1[:i]` to `word2[:j]`.
- **Transition**:

$$dp[i][j] = \begin{cases} dp[i-1][j-1] & \text{word1}[i-1] == \text{word2}[j-1] \\ 1 + \min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) & \text{otherwise} \end{cases}$$

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$.

---

### 3. Dual-Source Feasibility: Interleaving String

- **State**: $dp[i][j]$ = Whether `s1[:i]` and `s2[:j]` interleave to form `s3[:i+j]`.
- **Transition**:

$$dp[i][j] = (dp[i-1][j] \land s1[i-1] == s3[i+j-1]) \lor (dp[i][j-1] \land s2[j-1] == s3[i+j-1])$$

```python
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        m, n = len(s1), len(s2)
        if m + n != len(s3):
            return False
            
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] and s1[i - 1] == s3[i - 1]
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] and s2[j - 1] == s3[j - 1]
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = (
                    (dp[i - 1][j] and s1[i - 1] == s3[i + j - 1])
                    or (dp[i][j - 1] and s2[j - 1] == s3[i + j - 1])
                )
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$.

---

### 4. Subsequence Counting: Distinct Subsequences

- **State**: $dp[i][j]$ = Number of subsequences of `s[:i]` that equal `t[:j]`.
- **Transition**:

$$dp[i][j] = \begin{cases} dp[i-1][j-1] + dp[i-1][j] & s[i-1] == t[j-1] \\ dp[i-1][j] & s[i-1] \neq t[j-1] \end{cases}$$

```python
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = 1  # Empty target t can always be matched once (by picking nothing)
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s[i - 1] == t[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$.

---

### 5. Wildcard Branches: Regular Expression Matching

- **State**: $dp[i][j]$ = Whether `s[:i]` matches pattern `p[:j]`.
- **Transition**:
  1. $p[j-1] \neq '*'$: If $p[j-1] == s[i-1] \lor p[j-1] == '.' \implies dp[i][j] = dp[i-1][j-1]$.
  2. $p[j-1] == '*'$:
     - **Match 0 times**: Discard the `x*` token completely $\implies dp[i][j] = dp[i][j-2]$;
     - **Match 1 or more times**: If preceding char matches ($p[j-2] == s[i-1] \lor p[j-2] == '.'$) $\implies$ consume char from $s$ and keep pattern $\implies dp[i-1][j]$.

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        
        # Base Cases for patterns like a*, a*b*, a*b*c* matching empty string
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[0][j] = dp[0][j - 2]
                
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    dp[i][j] = dp[i][j - 2]  # Match 0 times
                    if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                        dp[i][j] = dp[i][j] or dp[i - 1][j]  # Match 1+ times
                elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    
        return dp[m][n]
```

- **Complexity**: Time $O(mn)$, Space $O(mn)$.

---

## Module 4: Skeleton 4 · Interval DP & Matrix Evolution

Interval DP states are defined over a closed interval $[i, j]$. Because larger intervals strictly depend on their shorter sub-intervals, iteration order must proceed **by increasing interval length ($L: 1 \to n$)** or **bottom-up ($i: n-1 \to 0$)**.

---

### 1. Palindrome Family: Longest Palindromic Substring & Palindromic Substrings

#### Recurrence & Onion Peeling Principle
Define $dp[i][j]$ as whether substring $s[i..j]$ is a palindrome:

$$dp[i][j] = (s[i] == s[j]) \land (j - i < 2 \lor dp[i+1][j-1])$$

1. **Boundary Mismatch** ($s[i] \neq s[j]$): Cannot be a palindrome $\implies dp[i][j] = False$;
2. **Length $\le 2$** ($j - i < 2$): If endpoints match, single char ($i=j$) or adjacent pair ($j=i+1$) are base cases $\implies dp[i][j] = True$;
3. **Length $\ge 3$** ($j - i \ge 2$): Check the **southwest neighbor $\swarrow$** $dp[i+1][j-1]$ (inner peeled core).

#### 2D State Transition Matrix Interactive Visualization

```palindrome-dp-demo
```

#### Code Implementation

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        ans = 0
        
        # Bottom-up sweep guarantees dp[i+1][j-1] is computed before dp[i][j]
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    ans += 1
        return ans

    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False] * n for _ in range(n)]
        start, max_len = 0, 1
        
        for i in range(n - 1, -1, -1):
            for j in range(i, n):
                if s[i] == s[j] and (j - i < 2 or dp[i + 1][j - 1]):
                    dp[i][j] = True
                    if j - i + 1 > max_len:
                        start, max_len = i, j - i + 1
        return s[start:start + max_len]
```

- **Complexity**: Time $O(n^2)$, Space $O(n^2)$.

---

### 2. Reverse Thinking of "Last Burst": Burst Balloons

- **Why "first burst" fails**: Popping $k$ first merges its left and right neighbors, dynamically entangling subproblem boundaries!
- **Reverse Formulation**: Enumerate the **last balloon $k$ to burst** in open interval $(i, j)$. When $k$ is burst last, all balloons in $(i, k)$ and $(k, j)$ are already cleared, leaving $k$'s neighbors fixed as the boundary elements $a[i]$ and $a[j]$!

$$dp[i][j] = \max_{i < k < j} \bigl(dp[i][k] + a[i] \cdot a[k] \cdot a[j] + dp[k][j]\bigr)$$

```python
class Solution:
    def maxCoins(self, nums: list[int]) -> int:
        a = [1] + nums + [1]
        n = len(a)
        dp = [[0] * n for _ in range(n)]
        
        for length in range(2, n):
            for i in range(0, n - length):
                j = i + length
                for k in range(i + 1, j):
                    dp[i][j] = max(
                        dp[i][j],
                        dp[i][k] + a[i] * a[k] * a[j] + dp[k][j]
                    )
        return dp[0][n - 1]
```

- **Complexity**: Time $O(n^3)$, Space $O(n^2)$.

---

## Module 5: Skeleton 4 · Knapsack Family (0/1, Complete & Counting)

Knapsack problems handle constrained optimization and counting. The defining difference in 1D space compression is the inner loop direction:

```text
Knapsack Traversal Direction:
┌─────────────────┬───────────────────┬───────────────────────────────────────┐
## Module 5: Skeleton 4 · Knapsack Family (0/1, Complete & Counting)

The knapsack family represents discrete choice optimization under resource budget constraints. Whether solving for maximum value, minimum items, feasibility, or combinations/permutations, all variations map into a rigorous, unified mathematical framework.

---

### 1. General Knapsack Problem-Solving Strategy Guide

When facing any knapsack variation in whiteboard interviews, follow the **4-Step Identification Framework** and **Algebraic Reduction Patterns**:

#### Step 1: The 4-Step Identification Framework

```text
Knapsack 4-Step Decision Flow:
┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ 1. Identify Capacity  │ ➔   │ 2. Item Reusability   │ ➔   │ 3. Target Operator    │ ➔   │ 4. Loop Hierarchy     │
│ Budget W & Cost weight│     │ 0/1 vs Complete       │     │ Min/Max vs Feasible   │     │ Combinations vs Perms │
└───────────────────────┘     └───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

1. **Identify Resource Constraint & Item Cost (Capacity & Cost)**:
   - What represents capacity $W$? (Sum limit, amount, string character limits, budget);
   - What is an "item"? Each item consumes cost $weight_i$ and yields benefit $value_i$.
2. **Determine Item Reusability (0/1 vs Complete vs Bounded)**:
   - **At most 1 use per item** $\implies$ **0/1 Knapsack**: 1D capacity loop **MUST be backward** ($W \to weight_i$) to prevent multi-selection within the same round;
   - **Unlimited reuse per item** $\implies$ **Complete Knapsack**: 1D capacity loop **MUST be forward** ($weight_i \to W$) to deliberately leverage current-round updates;
   - **Bounded count $k_i$ per item** $\implies$ **Multiple Knapsack**: Convert to 0/1 Knapsack via binary decomposition ($1, 2, 4, \dots, k_i - 2^p + 1$).
3. **Determine Target Metric & Operator**:
   - **Optimization**: Use $\max$ (max value) or $\min$ (Coin Change minimum count);
   - **Feasibility (Boolean)**: Use $\lor$ (Partition Equal Subset Sum);
   - **Counting Ways**: Use $+$ (Target Sum, Coin Change II).
4. **Determine Loop Nesting Hierarchy (Outer vs Inner)**:
   - **Items in Outer, Capacity in Inner** $\implies$ **Combinations**: `[1, 2]` equals `[2, 1]` (each item processed once in fixed order, e.g. Coin Change II);
   - **Capacity in Outer, Items in Inner** $\implies$ **Permutations**: `[1, 2]` differs from `[2, 1]` (any item can be chosen as the terminal addition at capacity $j$, e.g. Combination Sum IV).

---

#### Step 2: High-Frequency Algebraic Reduction Patterns

Real interview problems rarely say "this is knapsack" directly—they hide behind algebraic equivalence reductions:

```text
4 Canonical Algebraic Reductions:
1. Equal Subset Sum Partition (Partition Sum):
   Check if a subset sums to total / 2 ➔ 0/1 Knapsack Feasibility (target = total / 2)

2. Sign Partition & Summing (Target Sum):
   P - N = target and P + N = total ➔ P = (target + total) / 2 ➔ 0/1 Knapsack Subset Counting

3. Minimal Partition Difference (Last Stone Weight II):
   Find max subset sum <= floor(total / 2) ➔ Min difference is total - 2 * dp[floor(total / 2)]

4. Multi-dimensional Cost Knapsack (Ones and Zeroes):
   Bounded by <= M zeros AND <= N ones ➔ 2D state dp[j][k] with double backward loops
```

---

#### Step 3: Initialization & Sentinel Matrix

| Objective | Exact Fullness Required | `dp[0]` Base | Rest `dp[1..W]` | Core Rationale |
|---|---|---|---|---|
| **Max Value ($\max$)** | No ($\le W$) | `0` | `0` | Empty knapsack is valid with 0 value |
| **Max Value ($\max$)** | Yes ($= W$ exact) | `0` | `-inf` | Only capacity 0 is reachable initially |
| **Min Count ($\min$)** | Yes ($= W$ exact) | `0` | `+inf` (or `amount + 1`) | $dp[0]=0$ (0 coins for 0 sum), others $\infty$ |
| **Count Ways ($+$)** | Yes ($= W$ exact) | `1` | `0` | $dp[0]=1$ (choosing empty set is the 1 valid way) |
| **Feasibility ($\lor$)** | Yes ($= W$ exact) | `True` | `False` | Only sum 0 is reachable at start |

---

#### Step 4: Universal Knapsack Blueprint & Code Templates

Every knapsack problem reduces to a single unified framework tuned by **3 control knobs**:

```text
Universal Knapsack 3-Knob System:
┌───────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Control Knob          │ Branch Choices & Engineering Meaning                                   │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. Loop Hierarchy     │ Items Outer, Capacity Inner ➔ Combinations / Extremum / Feasibility   │
│                       │ Capacity Outer, Items Inner ➔ Permutations (e.g. Combination Sum IV)   │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. Capacity Direction │ Backward range(W, w-1, -1) ➔ 0/1 Knapsack (Prevent item multi-use)     │
│                       │ Forward  range(w, W+1)     ➔ Complete Knapsack (Allow item reuse)      │
├───────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. Target Operator    │ Extremum    ➔ dp[j] = min/max(...)                                     │
│                       │ Counting    ➔ dp[j] += dp[j - w]                                       │
│                       │ Feasible    ➔ dp[j] = dp[j] or dp[j - w]                               │
└───────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

##### 1. Master Parameterized Knapsack Function

```python
def universal_knapsack(
    items: list[tuple[int, int]],  # [(weight, value), ...]
    capacity: int,
    problem_type: str = "01_max",
    # Types: "01_max" | "01_min" | "01_feas" | "01_count" | "complete_min" | "complete_combo" | "complete_perm"
) -> int | bool:
    """
    Universal Knapsack Master Framework
    """
    # 1. Base case setup
    if problem_type in ("01_max", "complete_max"):
        dp = [0] * (capacity + 1)
    elif problem_type == "01_feas":
        dp = [True] + [False] * capacity
    elif problem_type in ("01_count", "complete_combo", "complete_perm"):
        dp = [1] + [0] * capacity
    elif problem_type in ("01_min", "complete_min"):
        INF = capacity + 1
        dp = [0] + [INF] * capacity

    # 2. State transition execution
    if problem_type == "complete_perm":
        # Permutations: Capacity in outer loop, Items in inner loop
        for j in range(1, capacity + 1):
            for weight, val in items:
                if j >= weight:
                    dp[j] += dp[j - weight]
    else:
        # Standard: Items in outer loop, Capacity in inner loop
        for weight, val in items:
            step_range = (
                range(capacity, weight - 1, -1)
                if problem_type.startswith("01")
                else range(weight, capacity + 1)
            )
            for j in step_range:
                if problem_type in ("01_max", "complete_max"):
                    dp[j] = max(dp[j], dp[j - weight] + val)
                elif problem_type in ("01_min", "complete_min"):
                    dp[j] = min(dp[j], dp[j - weight] + 1)
                elif problem_type in ("01_count", "complete_combo"):
                    dp[j] += dp[j - weight]
                elif problem_type == "01_feas":
                    dp[j] = dp[j] or dp[j - weight]

    return dp[capacity]
```

##### 2. Top-5 High-Frequency Interview Quick-Snippets

```python
# 1. 0/1 Knapsack · Feasibility (Partition Equal Subset Sum)
dp = [True] + [False] * target
for x in nums:
    for j in range(target, x - 1, -1):  # Backward
        dp[j] = dp[j] or dp[j - x]

# 2. 0/1 Knapsack · Counting Ways (Target Sum)
dp = [1] + [0] * bag
for x in nums:
    for j in range(bag, x - 1, -1):     # Backward
        dp[j] += dp[j - x]

# 3. Complete Knapsack · Minimum Count (Coin Change)
dp = [0] + [amount + 1] * amount
for c in coins:
    for j in range(c, amount + 1):      # Forward
        dp[j] = min(dp[j], dp[j - c] + 1)

# 4. Complete Knapsack · Combinations (Coin Change II)
dp = [1] + [0] * amount
for c in coins:                         # Items Outer
    for j in range(c, amount + 1):      # Forward Inner
        dp[j] += dp[j - c]

# 5. Complete Knapsack · Permutations (Combination Sum IV)
dp = [1] + [0] * target
for j in range(1, target + 1):          # Capacity Outer
    for x in nums:                      # Items Inner
        if j >= x:
            dp[j] += dp[j - x]
```

##### 3. Whiteboard 3-Second Mental Mnemonics

> 💡 **3-Second Interview Mnemonics**:
> - **"0/1 loops backward to prevent multi-use; Complete loops forward for compounding reuse."**
> - **"Items outer produces Combinations; Capacity outer produces Permutations."**
> - **"Feasibility uses OR; Counting uses ADD with base $dp[0]=1$."**

---

### 2. 0/1 Knapsack: Partition Equal Subset Sum (Feasibility)

- **Reduction**: Determine if a subset can be chosen (each element at most once) summing to $target = \text{total} / 2$.
- **Transition**: $dp[j] = dp[j] \lor dp[j - x]$ (traversed backward).

#### 0/1 Knapsack 1D State Evolution, Backward Traversal & Reachability Walkthrough

```subset-sum-demo
```

#### Code Implementation

```python
class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False
        target = total // 2
        
        dp = [False] * (target + 1)
        dp[0] = True
        
        for x in nums:
            for j in range(target, x - 1, -1):  # 0/1 Knapsack MUST traverse backward!
                dp[j] = dp[j] or dp[j - x]
        return dp[target]
```

- **Complexity**: Time $O(n \cdot target)$, Space $O(target)$.

---

### 3. 0/1 Knapsack: Target Sum (Algebraic Reduction & Counting)

- **Algebraic Derivation**:

$$P - N = target,\quad P + N = total \implies 2P = target + total \implies P = \frac{target + total}{2}$$
- **Reduction**: Count subset combinations summing to $bag = (target + total) // 2$.
- **Transition**: $dp[s] += dp[s - num]$ (backward).

```python
class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        total = sum(nums)
        if abs(target) > total or (target + total) % 2 != 0:
            return 0
            
        bag = (target + total) // 2
        dp = [0] * (bag + 1)
        dp[0] = 1
        
        for num in nums:
            for s in range(bag, num - 1, -1):  # Backward
                dp[s] += dp[s - num]
        return dp[bag]
```

- **Complexity**: Time $O(n \cdot bag)$, Space $O(bag)$.

---

### 4. Complete Knapsack: Coin Change (Min Coins)

- **Semantic**: Each coin denomination is available in infinite supply. Find the **minimum total number of coins** to make up total `amount`.
- **State**: $dp[a]$ = Minimum number of coins needed to make amount $a$.
- **Transition**: For each coin $coin$:

$$dp[a] = \min(dp[a], dp[a - coin] + 1) \quad (a \ge coin)$$

- **Why Complete Knapsack MUST Traverse Forward**:
  - In a 1D rolling array, forward traversal ($a: coin \to amount$) ensures that when we compute $dp[a]$, $dp[a - coin]$ has **already been updated within the current round**. This inherently allows the same coin to be reused multiple times (e.g. using three 1-dollar coins to make 3 dollars);
  - In 0/1 knapsack (at most 1 per item), backward traversal is mandatory to force referencing the un-updated state from the previous item.

#### Complete Knapsack 1D DP State Evolution & Path Replay Visualizer

```coin-change-demo
```

#### Code Implementation

```python
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        inf = amount + 1  # Sentinel upper bound (at most amount 1-dollar coins; amount+1 represents unreachable)
        dp = [inf] * (amount + 1)
        dp[0] = 0  # Base case: 0 coins needed for amount 0
        
        for coin in coins:
            for s in range(coin, amount + 1):  # Forward traversal ➔ allows unlimited reuse
                dp[s] = min(dp[s], dp[s - coin] + 1)
                
        return -1 if dp[amount] == inf else dp[amount]
```

- **Complexity**: Time $O(n \cdot amount)$, Space $O(amount)$.
- **Interview Pitfall**: Why doesn't greedy work?
  - E.g. `coins = [1, 3, 4], amount = 6`:
    - **Greedy** grabs the largest denomination first: $4 + 1 + 1 \implies 3$ coins;
    - **DP optimal** finds global coordination: $3 + 3 \implies 2$ coins!
  - Greedy only holds if denominations form a canonical coin system (like US/Euro coins); arbitrary denominations require DP.

---

### 5. Complete Knapsack: Coin Change II (Combinations vs Permutations)

- **Combinations vs Permutations Loop Order**:
  - **Combinations (`{1, 2}` same as `{2, 1}`)**: **Items in outer loop, Capacity in inner loop**. Each coin type is processed once in fixed order.
  - **Permutations (`{1, 2}` different from `{2, 1}`)**: **Capacity in outer loop, Items in inner loop**.

```python
class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        dp = [0] * (amount + 1)
        dp[0] = 1
        
        # Combinations: Coins in outer loop
        for c in coins:
            for a in range(c, amount + 1):
                dp[a] += dp[a - c]
        return dp[amount]
```

- **Complexity**: Time $O(n \cdot amount)$, Space $O(amount)$.

---

### 6. Multi-dimensional Cost 0/1 Knapsack: Ones and Zeroes

- **Problem Characteristic**: Each string costs $zeros$ zeros and $ones$ ones. Choose the maximum number of strings under budget $\le m$ zeros and $\le n$ ones.
- **Formulation**: 2D capacity state $dp[j][k]$, nested double backward loop!

```python
class Solution:
    def findMaxForm(self, strs: list[str], m: int, n: int) -> int:
        # dp[j][k] = max subset size with at most j zeros and k ones
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for s in strs:
            zeros = s.count('0')
            ones = len(s) - zeros
            
            # 0/1 Knapsack: Both dimensions must traverse backward!
            for j in range(m, zeros - 1, -1):
                for k in range(n, ones - 1, -1):
                    dp[j][k] = max(dp[j][k], dp[j - zeros][k - ones] + 1)
                    
        return dp[m][n]
```

- **Complexity**: Time $O(L \cdot m \cdot n)$ ($L$ is array length), Space $O(m \cdot n)$.

---

### 7. Knapsack Master Decision Matrix

| Knapsack Pattern | Canonical Problem | Loop Nesting | Traversal Direction | Core Recurrence |
|---|---|---|---|---|
| **0/1 Feasibility** | Partition Equal Subset Sum | Items Outer, Capacity Inner | Capacity **Backward** ($W \to w$) | $dp[j] = dp[j] \lor dp[j - w]$ |
| **0/1 Counting** | Target Sum | Items Outer, Capacity Inner | Capacity **Backward** ($W \to w$) | $dp[j] += dp[j - w]$ |
| **0/1 Multi-dim** | Ones and Zeroes | Items Outer, Capacity Inner | Multi-dim **Backward** | $dp[j][k] = \max(dp[j][k], dp[j-z][k-o] + 1)$ |
| **Complete Min Count** | Coin Change (Min Coins) | Items Outer, Capacity Inner | Capacity **Forward** ($w \to W$) | $dp[j] = \min(dp[j], dp[j - c] + 1)$ |
| **Complete Combinations**| Coin Change II | **Items Outer, Capacity Inner** | Capacity **Forward** ($w \to W$) | $dp[j] += dp[j - c]$ |
| **Complete Permutations**| Combination Sum IV | **Capacity Outer, Items Inner** | Capacity **Forward** ($1 \to W$) | $dp[j] += dp[j - num]$ |

---

## Module 6: Skeleton 5 · Grid & DAG Memoization

### 1. Deterministic Grid Sweep: Unique Paths

- **Rule**: Only move down or right $\implies$ Dependency is purely top and left.
- **Transition**: $dp[j] = dp[j] + dp[j-1]$ (single rolling row).

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for _ in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[-1]
```

- **Complexity**: Time $O(mn)$, Space $O(n)$.

### 2. Arbitrary Grid & Implicit DAG: Longest Increasing Path in a Matrix

- **Why nested loops fail**: Values are arbitrarily distributed without a static geometric sweep order.
- **DAG Property**: Strictly increasing paths form a Directed Acyclic Graph (DAG).
- **Solution**: Memoized DFS computes topological DP over the DAG.

```python
class Solution:
    def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
        if not matrix:
            return 0
        m, n = len(matrix), len(matrix[0])
        memo = {}
        
        def dfs(r: int, c: int) -> int:
            if (r, c) in memo:
                return memo[(r, c)]
            best = 1
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n and matrix[nr][nc] > matrix[r][c]:
                    best = max(best, 1 + dfs(nr, nc))
            memo[(r, c)] = best
            return best
            
        return max(dfs(r, c) for r in range(m) for c in range(n))
```

- **Complexity**: Time $O(mn)$ (each cell computed exactly once), Space $O(mn)$.

---

## Module 7: Skeleton 6 · State Machine DP

When a stage can occupy one of several mutually exclusive named discrete states, State Machine DP decouples transition logic cleanly.

### Classic: Best Time to Buy and Sell Stock with Cooldown

```text
3-State Finite State Machine (FSM):
           ┌─────────────────────────── buy ───────────────────────────┐
           │                                                           │
           ▼                                                           │
      ┌─────────┐                  sell                    ┌───────────────┐
      │  Hold   │ ───────────────────────────────────────> │     Sold      │
      │ (owned) │                                          │(just sold/cool)│
      └─────────┘                                          └───────────────┘
           │                                                           │
          rest                                                    cooldown
           │                                                           │
           ▼                                                           ▼
      ┌─────────┐                                          ┌───────────────┐
      │  Hold   │ <──────────────────── rest ───────────── │     Rest      │
      └─────────┘                                          │  (free cash)  │
                                                           └───────────────┘
```

#### States & Transitions
- `hold`: Max profit holding stock at end of day $\implies \text{hold} = \max(\text{hold}, \text{rest} - price)$
- `sold`: Max profit selling stock today (cooldown next day) $\implies \text{sold} = \text{hold} + price$
- `rest`: Max profit holding no stock & ready to buy $\implies \text{rest} = \max(\text{rest}, \text{sold})$

#### Implementation (Avoiding Variable Dirty-Read Overwrite)

```python
class Solution:
    def maxProfit(self, prices: list[int]) -> int:
        hold = float("-inf")
        sold = float("-inf")
        rest = 0
        
        for price in prices:
            next_hold = max(hold, rest - price)
            next_sold = hold + price
            next_rest = max(rest, sold)
            # Synchronous update prevents new hold from corrupting sold calculation
            hold, sold, rest = next_hold, next_sold, next_rest
            
        return max(sold, rest)
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

## Module 8: Advanced Insight · DP to Greedy Compression

Many greedy algorithms are mathematically derived from **compressing full DP tables via monotonicity and dominance relations**.

---

### 1. Kadane's Algorithm: Maximum Subarray

- **DP Formulation**: $dp[i] = \max(nums[i], dp[i-1] + nums[i])$.
- **Greedy Compression**: If $curSum < 0$, keeping it drags down all future extensions; thus reset $curSum = 0$ immediately!

```python
class Solution:
    def maxSubArray(self, nums: list[int]) -> int:
        cur_sum = 0
        max_sum = nums[0]
        for x in nums:
            if cur_sum < 0:
                cur_sum = 0
            cur_sum += x
            max_sum = max(max_sum, cur_sum)
        return max_sum
```

- **Complexity**: Time $O(n)$, Space $O(1)$.
- **Pitfall**: In an all-negative array (e.g. `[-3, -1, -2]`), `max_sum` must initialize to `nums[0]`, not `0`.

### 2. Jump Game: Leftmost Goal Compression

- **Full DP Formulation**: $dp[i] = \bigvee_{j=i+1}^{i+nums[i]} dp[j]$ in $O(n^2)$ time.
- **Greedy Reduction**:
  - The left-most good position dominates all positions to its right (easier to reach from earlier indices).
  - Hence, compress the entire boolean array into a single scalar `goal` (leftmost reachable position).
  - If $i + nums[i] \ge goal$, update $goal = i$.

```python
class Solution:
    def canJump(self, nums: list[int]) -> bool:
        goal = len(nums) - 1
        for i in range(len(nums) - 2, -1, -1):
            if i + nums[i] >= goal:
                goal = i
        return goal == 0
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

#### Criteria for DP ➔ Greedy Compression:
1. **Monotonic Boundary**: State set can be represented by a single extremum (e.g. leftmost `goal` or furthest `reach`).
2. **State Dominance**: The chosen representative state strictly outperforms all dominated states across all future branches.
3. **Choice Safety**: Local greedy decisions never eliminate any globally optimal possibilities.

---

## Module 9: Interview Pitfalls, Complexity Matrix & Communication Template

### 1. Complexity & Space Optimization Summary

| Skeleton | Problem | Standard Time | Space (Standard $\to$ Optimized) | Key Transitions & Order |
|---|---|---|---|---|
| **1D Linear** | Decode Ways | $O(n)$ | $O(n) \to O(1)$ | Backward sweep, handle `'0'`, 2 rolling variables |
| **1D Linear** | Climbing Stairs | $O(n)$ | $O(n) \to O(1)$ | Fibonacci structure, 2 rolling variables |
| **1D Linear** | Min Cost Climbing Stairs | $O(n)$ | $O(n) \to O(1)$ | Top floor has no cost, take $\min(dp[n-1], dp[n-2])$ |
| **1D Linear** | House Robber I / II | $O(n)$ | $O(n) \to O(1)$ | Pick vs skip; II splits into 2 linear subsegments |
| **1D Linear** | Word Break | $O(n^2 \cdot L) \xrightarrow{\text{Trie}} O(n \cdot L_{\max})$ | $O(n)$ | Set lookup / Forward Trie traversal with instant pruning |
| **1D Linear** | LIS | $O(n^2)$ | $O(n)$ | Global predecessor sweep; answer is $\max(dp)$ |
| **1D Linear** | Maximum Product Subarray | $O(n)$ | $O(n) \to O(1)$ | Track `max_here` and `min_here` simultaneously |
| **Two Sequences** | LCS | $O(mn)$ | $O(mn) \to O(\min(m, n))$ | Match $\to$ diagonal $+1$; else $\max(\text{top}, \text{left})$ |
| **Two Sequences** | Edit Distance | $O(mn)$ | $O(mn)$ | Match $\to$ diagonal; else $1 + \min(\text{insert, delete, replace})$ |
| **Two Sequences** | Interleaving String | $O(mn)$ | $O(mn)$ | Top or left matching `s3[i+j-1]` |
| **Two Sequences** | Distinct Subsequences | $O(mn)$ | $O(mn)$ | Match $\to$ sum of pick and skip; $dp[i][0]=1$ |
| **Two Sequences** | Regex Matching | $O(mn)$ | $O(mn)$ | `*` branches into $dp[i][j-2]$ (0x) and $dp[i-1][j]$ (1+x) |
| **Interval DP** | Longest Palindrome / Count | $O(n^2)$ | $O(n^2)$ | Southwest neighbor $dp[i+1][j-1]$, bottom-up sweep |
| **Interval DP** | Burst Balloons | $O(n^3)$ | $O(n^2)$ | Reverse formulation: last balloon $k$ to burst |
| **0/1 Knapsack** | Partition Equal Subset Sum | $O(n \cdot \frac{\text{sum}}{2})$ | $O(\frac{\text{sum}}{2})$ | Feasibility knapsack, capacity must loop **backward** |
| **0/1 Knapsack** | Target Sum | $O(n \cdot bag)$ | $O(bag)$ | Algebraic reduction to subset count, **backward** |
| **Complete** | Coin Change (Min Coins) | $O(n \cdot amount)$ | $O(amount)$ | Complete knapsack, capacity loops **forward**, $\min$ |
| **Complete** | Coin Change II (Combinations)| $O(n \cdot amount)$ | $O(amount)$ | Combinations: **Coins in outer loop, capacity forward in inner** |
| **Grid DP** | Unique Paths | $O(mn)$ | $O(n)$ | Top + Left, single rolling row |
| **Grid DP** | Longest Increasing Path | $O(mn)$ | $O(mn)$ | Implicit DAG over grid, Memoized DFS |
| **State Machine**| Stock with Cooldown | $O(n)$ | $O(n) \to O(1)$ | Hold / Sold / Rest 3-state synchronous rotation |
| **DP ➔ Greedy**| Kadane Max Subarray | $O(n)$ | $O(1)$ | Negative prefix reset, 1 rolling variable |
| **DP ➔ Greedy**| Jump Game | $O(n)$ | $O(1)$ | Maintain leftmost `goal` or furthest `reach` |

---

### 2. High-Frequency Interview Pitfalls

1. **Writing code without clarifying Base Cases**: E.g. $dp[0]=1$, empty string match, top floor having no cost.
2. **Iteration order violating topological dependencies**: In Interval DP, sweeping without length-increasing/bottom-up order reads uninitialized $dp[i+1][j-1]$.
3. **Reversing knapsack loop direction**:
   - 0/1 Knapsack in forward order allows the same item to be reused multiple times within the same step, degenerating into complete knapsack.
   - Permutations vs Combinations: Coins in outer loop yields combinations (Coin Change II); capacity in outer loop yields permutations (Combination Sum IV).
4. **State Machine dirty-read overwrites during space optimization**: Modifying `hold` before computing `sold = hold + price` incorrectly reads the updated `hold`. Always use temporary variables for synchronous updates!
5. **Wrong answer location in extremum DP**: Assuming the final answer is always at $dp[n-1]$ (in LIS and Maximum Product Subarray, the answer is the global maximum over all cells).

---

### 3. Structured Interview Communication Template

When presenting your solution in a whiteboard interview, follow these 6 structured steps:

1. **State Definition**: "I define $dp[i][j]$ as the optimal value / number of ways for prefix/interval/stage..."
2. **Decision Branches & Transition**: "At the current position, there are $k$ choices, which map to sub-states..."
3. **Base Cases**: "The base cases are $dp[0]$ / the main diagonal, representing..."
4. **Iteration Order**: "Because the current state depends on earlier states, we iterate in forward/backward/length-increasing order."
5. **Implementation & Dry Run**: "Let's implement the 2D/1D DP and dry-run with a small example."
6. **Complexity & Space Optimization**: "The time complexity is $O(\dots)$ and space is $O(\dots)$. Observing that dependencies only reach the previous row/two variables, we can compress space to $O(1)$ / $O(m)$."


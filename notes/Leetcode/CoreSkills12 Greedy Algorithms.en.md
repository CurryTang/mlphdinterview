# Core Skills 12 · Greedy Algorithms: Forced Moves, Reachable Envelopes, and Invariant Proofs

## 1. The Core Mental Model of Greedy Algorithms

In technical coding interviews, many candidates assume greedy algorithms simply mean "greedily pick whatever looks best right now." However, in competitive interviews, **writing greedy code based on sheer intuition often leads directly into traps where local optimality fails to yield global optimality**.

A sound greedy strategy is always grounded in rigorous mathematical guarantees:
1. **Greedy Choice Property**: A globally optimal solution can be reached by making a sequence of locally optimal choices.
2. **Optimal Substructure**: An optimal solution to the problem contains within it optimal solutions to subproblems, reducing the problem to an identical smaller subproblem.

In interviews, the two standard arguments used to prove greedy correctness are:
- **Forced Move / Unique Placement**: The most extreme element (the smallest item, earliest deadline, or strictest constraint) has **no other valid placement** in any legitimate solution. Therefore, resolving it immediately discards zero valid solutions.
- **Exchange Argument / Dominance**: Assume an optimal solution $OPT$ does not make the greedy choice. We can exchange one of its decisions with the greedy choice to construct a new valid solution $OPT'$ that is at least as good as $OPT$.

```text
4-Step Thinking Framework for Greedy Problems:
1. Inspect Extremes and Boundaries: Is there an element that must be handled first (e.g. minimum value, earliest deadline, rightmost boundary)?
2. Verify Lack of Aftereffect: Does the current local choice impose irreversible negative restrictions on future choices? If yes, backtrack or use dynamic programming instead.
3. Maintain a Monotonic Invariant: E.g., "the farthest reachable index max_reach", "a non-negative running subarray prefix", or "valid unclosed parentheses count range [cmin, cmax]".
4. Discard Negative Drags: Once a prefix's net contribution drops below zero, carrying it forward only drags future subarrays down—reset the starting point immediately.
```

---

## 2. Four Universal Greedy Templates

| Template Category | Invariant & Operational Logic | Classic Representative Problems |
| :--- | :--- | :--- |
| **1. Prefix Reset** | When running sum `< 0`, history is a pure drag on the future; discard and reset start immediately. | **Maximum Subarray** (LC 53)<br>**Gas Station** (LC 134) |
| **2. Reachable Envelope & BFS Window** | Maintain farthest reachable boundary `max_reach`; advance layer-by-layer for minimum steps. | **Jump Game** (LC 55)<br>**Jump Game II** (LC 45) |
| **3. Forced Choice & Disqualification** | Minimum element has no predecessor, forced to start a sequence; disqualify violating candidates upfront. | **Hand of Straights** (LC 846)<br>**Merge Triplets** (LC 1899) |
| **4. Boundary Merge & Range Tracking** | Cut immediately when reaching `max(last[c])`; track unclosed bracket range `[min, max]` under wildcards. | **Partition Labels** (LC 763)<br>**Valid Parenthesis String** (LC 678) |

```greedy-patterns
```

---

## 3. Deep Dive into the 8 NeetCode 150 Greedy Problems

---

### Problem 1: Maximum Subarray (LC 53)

#### Problem Statement
Given an integer array `nums`, find the subarray with the largest sum, and return its sum.

#### Greedy Choice & Invariant Proof
Let `cur_sum` be the sum of the current contiguous subarray. When iterating over element `x`:
- If `cur_sum > 0`, appending `x` benefits `x` (even if `x` is negative, the prefix still contributes positive value).
- If `cur_sum <= 0`, the historical prefix has become a "liability". Adding a negative prefix to `x` only reduces the sum of any subarray starting at `x`. **We must discard the negative prefix and restart the subarray from `x`**.

```text
Kadane's State Transition:
    cur_sum = max(x, cur_sum + x)
    max_sum = max(max_sum, cur_sum)
```

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        cur_sum = 0
        
        for num in nums:
            cur_sum = max(num, cur_sum + num)
            max_sum = max(max_sum, cur_sum)
            
        return max_sum
```

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int maxSubArray(const std::vector<int>& nums) {
        int max_sum = nums[0];
        int cur_sum = 0;
        for (int num : nums) {
            cur_sum = std::max(num, cur_sum + num);
            max_sum = std::max(max_sum, cur_sum);
        }
        return max_sum;
    }
};
```

- **Complexity**: Time $O(n)$, Space $O(1)$.
- **Trap**: When all numbers are negative (e.g. `[-3, -2, -1]`), `max_sum` must not be initialized to `0`; initialize it to `nums[0]`.

---

### Problem 2: Jump Game (LC 55)

#### Problem Statement
You are given an integer array `nums`. You are initially positioned at the array's first index. Each element represents your maximum jump length at that position. Return `true` if you can reach the last index.

#### Greedy Choice & Invariant Proof
We do not need to branch over every jump size ($O(2^n)$). **We only maintain a global invariant: the farthest reachable index `max_reach`**.
- For each index `i`: if `i > max_reach`, index `i` is unreachable from any previous jump—return `False` immediately.
- Otherwise, `i` is reachable, so we extend the boundary: `max_reach = max(max_reach, i + nums[i])`.
- If `max_reach >= n - 1`, we can reach the end—return `True` early.

```jump-game-demo
```

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach = 0
        n = len(nums)
        
        for i, jump in enumerate(nums):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i + jump)
            if max_reach >= n - 1:
                return True
                
        return True
```

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    bool canJump(const std::vector<int>& nums) {
        int max_reach = 0;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            if (i > max_reach) return false;
            max_reach = std::max(max_reach, i + nums[i]);
            if (max_reach >= n - 1) return true;
        }
        return true;
    }
};
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

### Problem 3: Jump Game II (LC 45)

#### Problem Statement
Given a 0-indexed array of integers `nums` of length `n`, return the minimum number of jumps to reach `nums[n - 1]`.

#### Greedy Choice: Implicit BFS Level Window
Finding the minimum steps is equivalent to unweighted shortest path BFS. All indices reachable in $k$ jumps form a contiguous window `[cur_start, cur_end]`.
- While scanning within `[cur_start, cur_end]`, track the **farthest boundary reachable in the next jump: `farthest = max(farthest, i + nums[i])`**.
- When pointer `i` reaches `cur_end`, the current jump layer is exhausted:
  - `steps += 1`
  - Advance the window: `cur_end = farthest`.

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
            
        steps = 0
        cur_end = 0
        farthest = 0
        
        # Note: Loop only up to n - 2! Looping up to n - 1 causes an extra redundant step increment.
        for i in range(n - 1):
            farthest = max(farthest, i + nums[i])
            if i == cur_end:
                steps += 1
                cur_end = farthest
                if cur_end >= n - 1:
                    break
                    
        return steps
```

```cpp
#include <vector>
#include <algorithm>

class Solution {
public:
    int jump(const std::vector<int>& nums) {
        int n = nums.size();
        if (n <= 1) return 0;
        
        int steps = 0;
        int cur_end = 0;
        int farthest = 0;
        
        for (int i = 0; i < n - 1; ++i) {
            farthest = std::max(farthest, i + nums[i]);
            if (i == cur_end) {
                ++steps;
                cur_end = farthest;
                if (cur_end >= n - 1) break;
            }
        }
        return steps;
    }
};
```

- **Complexity**: Time $O(n)$, Space $O(1)$.
- **Crucial Boundary Detail**: Loop up to `n - 2` (`range(n - 1)`). Once `cur_end >= n - 1`, we have already reached the last index.

---

### Problem 4: Gas Station (LC 134)

#### Problem Statement
There are `n` gas stations along a circular route, where the amount of gas at station `i` is `gas[i]`. The cost to travel from station `i` to station `i+1` is `cost[i]`. Find the starting station index to complete the full circuit once, or `-1` if impossible.

#### Greedy Theorems & Proof
1. **Global Solvability Theorem**: If $\sum gas[i] < \sum cost[i]$, total gas is insufficient, return `-1`. If $\sum gas[i] \ge \sum cost[i]$, **a unique valid starting station is guaranteed to exist**.
2. **Deficit Reset Theorem**:
   - Suppose we start at `start` and travel smoothly until reaching station `j`, where cumulative fuel first drops below zero (`tank < 0`).
   - **Theorem: No station $k$ in the range $[start, j]$ can serve as a valid starting station!**
   - **Proof**: Since we successfully reached $k$ from `start`, our remaining fuel at $k$ was $\ge 0$. If we started from $k$ directly with $0$ initial fuel, our fuel at $j$ would be even less, causing fuel exhaustion at or before $j$.
   - **Greedy Action**: Jump the next candidate start directly to `start = j + 1`, and reset `tank = 0`.

```gas-station-demo
```

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        total_surplus = 0
        cur_tank = 0
        start = 0
        
        for i in range(len(gas)):
            net = gas[i] - cost[i]
            total_surplus += net
            cur_tank += net
            
            if cur_tank < 0:
                start = i + 1
                cur_tank = 0
                
        return start if total_surplus >= 0 else -1
```

```cpp
#include <vector>

class Solution {
public:
    int canCompleteCircuit(const std::vector<int>& gas, const std::vector<int>& cost) {
        int total_surplus = 0;
        int cur_tank = 0;
        int start = 0;
        
        for (int i = 0; i < gas.size(); ++i) {
            int net = gas[i] - cost[i];
            total_surplus += net;
            cur_tank += net;
            if (cur_tank < 0) {
                start = i + 1;
                cur_tank = 0;
            }
        }
        
        return total_surplus >= 0 ? start : -1;
    }
};
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

### Problem 5: Hand of Straights (LC 846)

#### Problem Statement
Given an integer array `hand` and an integer `groupSize`, determine whether it is possible to rearrange the cards into groups of size `groupSize` consisting of consecutive integers.

#### Greedy Choice: Global Minimum Has No Predecessor
If `x` is the smallest remaining card:
- Can `x` be the 2nd or 3rd card of a straight? **No!** That would require a smaller card `x - 1`, which does not exist because `x` is the global minimum.
- **Therefore, `x` must be the first card of `count[x]` straights: `[x, x+1, ..., x + groupSize - 1]`.**
- If `need = count[x]`, we must deduct `count[x + k] -= need` for each $0 \le k < groupSize$. If any required card count is $< need$, return `False`.

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

```cpp
#include <vector>
#include <map>

class Solution {
public:
    bool isNStraightHand(const std::vector<int>& hand, int groupSize) {
        if (hand.size() % groupSize != 0) return false;
        
        std::map<int, int> count;
        for (int card : hand) {
            count[card]++;
        }
        
        for (auto [first, freq] : count) {
            if (freq == 0) continue;
            
            for (int k = 0; k < groupSize; ++k) {
                int card = first + k;
                if (count[card] < freq) return false;
                count[card] -= freq;
            }
        }
        return true;
    }
};
```

- **Complexity**: Time $O(u \log u + u \cdot groupSize)$ where $u$ is the number of distinct cards, Space $O(u)$.

---

### Problem 6: Merge Triplets to Form Target Triplet (LC 1899)

#### Problem Statement
Given `triplets` where `triplets[i] = [ai, bi, ci]` and `target = [x, y, z]`, return `true` if you can obtain `target` by merging triplets using component-wise `max([a1, a2], [b1, b2], [c1, c2])`.

#### Greedy Choice: Coordinate Independence & Disqualification
The `max` operator is **monotonically non-decreasing**. If any component exceeds the target, it can never be reduced.
1. **Safety Filter**: If any component of a triplet `t` exceeds `target` (`t[0] > target[0] or t[1] > target[1] or t[2] > target[2]`), this triplet is disqualified and discarded.
2. **Greedy Merge**: Merging all remaining safe triplets will never exceed `target`.
3. **Validation**: We only need to check if the safe triplets collectively contain values equal to `target[0]`, `target[1]`, and `target[2]`.

```python
class Solution:
    def mergeTriplets(self, triplets: List[List[int]], target: List[int]) -> bool:
        tx, ty, tz = target
        has_x = has_y = has_z = False
        
        for a, b, c in triplets:
            if a > tx or b > ty or c > tz:
                continue
                
            if a == tx: has_x = True
            if b == ty: has_y = True
            if c == tz: has_z = True
            
            if has_x and has_y and has_z:
                return True
                
        return False
```

```cpp
#include <vector>

class Solution {
public:
    bool mergeTriplets(const std::vector<std::vector<int>>& triplets, const std::vector<int>& target) {
        int tx = target[0], ty = target[1], tz = target[2];
        bool has_x = false, has_y = false, has_z = false;
        
        for (const auto& t : triplets) {
            if (t[0] > tx || t[1] > ty || t[2] > tz) continue;
            
            if (t[0] == tx) has_x = true;
            if (t[1] == ty) has_y = true;
            if (t[2] == tz) has_z = true;
            
            if (has_x && has_y && has_z) return true;
        }
        return false;
    }
};
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

### Problem 7: Partition Labels (LC 763)

#### Problem Statement
Given a string `s`, partition it into as many parts as possible so that each letter appears in at most one part. Return a list of the partition sizes.

#### Greedy Choice: Last-Occurrence Envelope Cut
1. **Precomputation**: Record the last occurrence index `last[c]` for every character in `s`.
2. **Greedy Scan**: Maintain current partition start `start` and required boundary `end`:
   - At character `s[i]`, update `end = max(end, last[s[i]])`.
   - When `i == end`, all characters inside `[start, end]` will **never appear again in the remainder of the string**.
   - Cut immediately at `i == end`, record size `i - start + 1`, and start the next partition at `start = i + 1`.

```partition-labels-demo
```

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = {c: i for i, c in enumerate(s)}
        
        partitions = []
        start = 0
        end = 0
        
        for i, c in enumerate(s):
            end = max(end, last[c])
            if i == end:
                partitions.append(i - start + 1)
                start = i + 1
                
        return partitions
```

```cpp
#include <vector>
#include <string>
#include <algorithm>

class Solution {
public:
    std::vector<int> partitionLabels(const std::string& s) {
        int last[26] = {0};
        for (int i = 0; i < s.size(); ++i) {
            last[s[i] - 'a'] = i;
        }
        
        std::vector<int> partitions;
        int start = 0;
        int end = 0;
        
        for (int i = 0; i < s.size(); ++i) {
            end = std::max(end, last[s[i] - 'a']);
            if (i == end) {
                partitions.push_back(i - start + 1);
                start = i + 1;
            }
        }
        return partitions;
    }
};
```

- **Complexity**: Time $O(n)$, Space $O(1)$ (26 English characters).

---

### Problem 8: Valid Parenthesis String (LC 678)

#### Problem Statement
Given a string `s` containing `'('`, `')'`, and `'*'`, where `'*'` can be treated as `'('`, `')'`, or an empty string `""`, return `true` if `s` is valid.

#### Greedy Choice: Range Balance Tracking `[cmin, cmax]`
Instead of $O(3^n)$ recursion, track the closed interval of possible open parenthesis counts `[cmin, cmax]`:
- `cmax`: Maximum open parentheses if all `*` are treated as `'('`.
- `cmin`: Minimum open parentheses if all `*` are treated as `')'` (clamped at `0`).

State transitions:
1. `'('`: `cmin += 1, cmax += 1`
2. `')'`: `cmin = max(0, cmin - 1), cmax -= 1`
3. `'*'`: `cmin = max(0, cmin - 1), cmax += 1`
4. Validation:
   - If `cmax < 0`, excess `')'` cannot be matched even if all `*` become `'('`—return `False`.
   - At the end, return `cmin == 0`.

```python
class Solution:
    def checkValidString(self, s: str) -> bool:
        cmin = 0
        cmax = 0
        
        for ch in s:
            if ch == '(':
                cmin += 1
                cmax += 1
            elif ch == ')':
                cmin -= 1
                cmax -= 1
            else: # '*'
                cmin -= 1
                cmax += 1
                
            if cmax < 0:
                return False
                
            cmin = max(cmin, 0)
            
        return cmin == 0
```

```cpp
#include <string>
#include <algorithm>

class Solution {
public:
    bool checkValidString(const std::string& s) {
        int cmin = 0;
        int cmax = 0;
        
        for (char ch : s) {
            if (ch == '(') {
                ++cmin;
                ++cmax;
            } else if (ch == ')') {
                --cmin;
                --cmax;
            } else {
                --cmin;
                ++cmax;
            }
            
            if (cmax < 0) return false;
            cmin = std::max(cmin, 0);
        }
        
        return cmin == 0;
    }
};
```

- **Complexity**: Time $O(n)$, Space $O(1)$.

---

## 4. Summary & Top Interview Pitfall Matrix

| Problem | Core Invariant | Fatal Trap | Complexity |
| :--- | :--- | :--- | :--- |
| **53. Maximum Subarray** | Reset running sum when `cur_sum < 0` | Initializing `max_sum = 0` for all-negative arrays | Time $O(n)$<br>Space $O(1)$ |
| **55. Jump Game** | Maintain `max_reach` envelope monotonically | Blind backtracking on 0 (only need `max_reach` to cross) | Time $O(n)$<br>Space $O(1)$ |
| **45. Jump Game II** | Implicit BFS level window (`steps += 1` when `i == cur_end`) | Looping to `n - 1` triggers an extra step at the finish line | Time $O(n)$<br>Space $O(1)$ |
| **134. Gas Station** | Deficit eliminates prefix; jump candidate start to `i + 1` | Missing global total surplus validation (`total_surplus >= 0`) | Time $O(n)$<br>Space $O(1)$ |
| **846. Hand of Straights** | Minimum card has no predecessor, forced to start straights | Missing `len(hand) % groupSize == 0` validation | Time $O(n \log n)$<br>Space $O(n)$ |
| **1899. Merge Triplets** | Disqualify violating triplets upfront, merge all safe items | Expecting a single triplet to match all 3 target components | Time $O(n)$<br>Space $O(1)$ |
| **763. Partition Labels** | Cut partition immediately when `i == max(last[c])` | Slicing on the fly without precomputing last occurrence table | Time $O(n)$<br>Space $O(1)$ |
| **678. Valid Parenthesis String** | Track unclosed left bracket balance range `[cmin, cmax]` | Forgetting to clamp `cmin` lower bound at 0 | Time $O(n)$<br>Space $O(1)$ |

**Key Takeaways**:
- Always present the invariant and exchange argument to the interviewer before writing code.
- Whenever negative running sum degrades future subarrays, apply the prefix reset model (Kadane / Gas Station).
- For step-minimizing jump problems, use the implicit BFS level window model.
- For multidimensional max-merging problems, disqualify violating candidates upfront.
- For wildcard bracket matching, never write exponential backtracking when tracking the `[cmin, cmax]` range provides an exact $O(n)$ solution.

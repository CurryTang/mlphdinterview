# Bit Manipulation: LeetCode & NeetCode Patterns Cheat Sheet

The core essence of bit manipulation: **Leverage bitwise parallelism to eliminate branching and loops, achieving $O(1)$ auxiliary space or optimal runtime bounds.**  
This cheat sheet consolidates high-frequency bit manipulation patterns across LeetCode and NeetCode 150: the left column provides problem statements and core invariants; the right column provides idiomatic, minimal Python implementations.

---

## 0. Bit Manipulation Primitives (Quick Reference)

<table>
<thead>
  <tr>
    <th style="width: 38%;">Pattern / Target Operation</th>
    <th>Python Core Expression / Statement</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><strong>Get Bit $i$</strong><br/>Extract the $i$-th bit of integer $n$ (0-indexed)</td>
    <td>

```python
bit = (n >> i) & 1
```

</td>
  </tr>
  <tr>
    <td><strong>Set Bit $i$</strong><br/>Force the $i$-th bit to 1, leaving other bits untouched</td>
    <td>

```python
n = n | (1 << i)
```

</td>
  </tr>
  <tr>
    <td><strong>Clear Bit $i$</strong><br/>Force the $i$-th bit to 0, leaving other bits untouched</td>
    <td>

```python
n = n & ~(1 << i)
```

</td>
  </tr>
  <tr>
    <td><strong>Toggle Bit $i$</strong><br/>Flip the $i$-th bit: 0 becomes 1, 1 becomes 0</td>
    <td>

```python
n = n ^ (1 << i)
```

</td>
  </tr>
  <tr>
    <td><strong>Clear Lowest 1-Bit (Brian Kernighan)</strong><br/>Clears the rightmost set bit in binary representation</td>
    <td>

```python
n = n & (n - 1)
```

</td>
  </tr>
  <tr>
    <td><strong>Extract Lowest 1-Bit (lowbit)</strong><br/>Isolates the rightmost set bit, zeroing all others</td>
    <td>

```python
lowbit = n & (-n)
```

</td>
  </tr>
  <tr>
    <td><strong>Parity Check (Is Odd)</strong><br/>Evaluates whether the lowest bit is 1</td>
    <td>

```python
is_odd = bool(n & 1)
```

</td>
  </tr>
</tbody>
</table>

---

## 1. XOR Cancellation Family

Leveraging key XOR properties: $x \oplus x = 0$, $x \oplus 0 = x$, commutative and associative.

<table>
<thead>
  <tr>
    <th style="width: 38%;">Problem &amp; Pattern</th>
    <th>Core Bit Manipulation Implementation (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 136. Single Number</strong><br/><br/>
      • <strong>Problem</strong>: Every element appears twice except for one unique element.<br/>
      • <strong>Pattern</strong>: XOR all numbers; duplicate pairs cancel via $x \oplus x = 0$.<br/>
      • <strong>Complexity</strong>: Time $O(n)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def singleNumber(self, nums: list[int]) -> int:
        res = 0
        for x in nums:
            res ^= x
        return res
```

</td>
  </tr>
  <tr>
    <td>
      <strong>LeetCode 268. Missing Number</strong><br/><br/>
      • <strong>Problem</strong>: Array of length $n$ containing $n$ distinct numbers from $[0, n]$. Find the one missing.<br/>
      • <strong>Pattern</strong>: XOR the full range $[0..n]$ against all array elements; matching numbers cancel.<br/>
      • <strong>Complexity</strong>: Time $O(n)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def missingNumber(self, nums: list[int]) -> int:
        res = len(nums)
        for i, x in enumerate(nums):
            res ^= i ^ x
        return res
```

</td>
  </tr>
  <tr>
    <td>
      <strong>LeetCode 389. Find the Difference</strong><br/><br/>
      • <strong>Problem</strong>: String $t$ is formed by shuffling $s$ and adding one random character. Find that character.<br/>
      • <strong>Pattern</strong>: XOR ASCII character values over $s + t$; pairs cancel, leaving the extra char.<br/>
      • <strong>Complexity</strong>: Time $O(n)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        ch = 0
        for c in s + t:
            ch ^= ord(c)
        return chr(ch)
```

</td>
  </tr>
  <tr>
    <td>
      <strong>LeetCode 260. Single Number III</strong><br/><br/>
      • <strong>Problem</strong>: Exactly two elements appear once; all other elements appear twice.<br/>
      • <strong>Pattern</strong>: XOR all numbers to obtain $a \oplus b$. Extract `lowbit = diff & (-diff)` as a differentiating bit. Partition into two groups and XOR separately.<br/>
      • <strong>Complexity</strong>: Time $O(n)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def singleNumber(self, nums: list[int]) -> list[int]:
        diff = 0
        for x in nums:
            diff ^= x
        # Extract lowest differentiating bit between a and b
        lowbit = diff & (-diff)
        a = b = 0
        for x in nums:
            if x & lowbit:
                a ^= x
            else:
                b ^= x
        return [a, b]
```

</td>
  </tr>
</tbody>
</table>

---

## 2. Lowest Set-Bit Clearing & Hamming Weight (Brian Kernighan & Lowbit)

Using `n & (n - 1)` to clear the lowest 1-bit, looping strictly $k$ times ($k =$ count of set bits) instead of 32 iterations.

<table>
<thead>
  <tr>
    <th style="width: 38%;">Problem &amp; Pattern</th>
    <th>Core Bit Manipulation Implementation (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 191. Number of 1 Bits (Hamming Weight)</strong><br/><br/>
      • <strong>Problem</strong>: Return the number of set bits (Hamming weight) of an integer.<br/>
      • <strong>Pattern</strong>: Repeat `n &= n - 1` to peel off the lowest 1 until zero.<br/>
      • <strong>Complexity</strong>: Time $O(k)$ ($k \le 32$ is the number of 1-bits), Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n &= n - 1
            count += 1
        return count
```

</td>
  </tr>
  <tr>
    <td>
      <strong>LeetCode 231. Power of Two</strong><br/><br/>
      • <strong>Problem</strong>: Determine if integer $n$ is a power of two.<br/>
      • <strong>Pattern</strong>: Must be positive ($n > 0$) and contain exactly one set bit (`n & (n - 1) == 0`).<br/>
      • <strong>Complexity</strong>: Time $O(1)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0
```

</td>
  </tr>
  <tr>
    <td>
      <strong>LeetCode 342. Power of Four</strong><br/><br/>
      • <strong>Problem</strong>: Determine if integer $n$ is a power of four.<br/>
      • <strong>Pattern</strong>: Must be a power of two and its sole 1-bit must reside at an odd position (`n & 0x55555555 != 0`).<br/>
      • <strong>Complexity</strong>: Time $O(1)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0 and (n & 0x55555555) != 0
```

</td>
  </tr>
  <tr>
    <td>
      <strong>LeetCode 338. Counting Bits</strong><br/><br/>
      • <strong>Problem</strong>: Return an array of length $n + 1$ with set-bit counts for each $0 \le i \le n$.<br/>
      • <strong>Pattern</strong>: DP relation `dp[i] = dp[i & (i - 1)] + 1`: bit count equals subproblem without lowest bit plus 1.<br/>
      • <strong>Complexity</strong>: Time $O(n)$, Space $O(1)$ (excluding output array).
    </td>
    <td>

```python
class Solution:
    def countBits(self, n: int) -> list[int]:
        dp = [0] * (n + 1)
        for i in range(1, n + 1):
            dp[i] = dp[i & (i - 1)] + 1
        return dp
```

</td>
  </tr>
  <tr>
    <td>
      <strong>LeetCode 461. Hamming Distance</strong><br/><br/>
      • <strong>Problem</strong>: Return the number of positions at which the corresponding bits are different.<br/>
      • <strong>Pattern</strong>: `x ^ y` isolates differing bits, then apply Kernighan's trick.<br/>
      • <strong>Complexity</strong>: Time $O(k)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        xor = x ^ y
        res = 0
        while xor:
            xor &= xor - 1
            res += 1
        return res
```

</td>
  </tr>
</tbody>
</table>

---

## 3. Modulo-K State Machine & Bitwise Counting

<table>
<thead>
  <tr>
    <th style="width: 38%;">Problem &amp; Pattern</th>
    <th>Core Bit Manipulation Implementation (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 137. Single Number II</strong><br/><br/>
      • <strong>Problem</strong>: Every element appears 3 times except for one unique element appearing once.<br/>
      • <strong>Pattern</strong>: Finite state machine modulo 3 ($00 \to 01 \to 10 \to 00$). `ones` and `twos` track bits appearing 1 and 2 times, resetting when reaching 3.<br/>
      • <strong>Complexity</strong>: Time $O(n)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def singleNumber(self, nums: list[int]) -> int:
        ones = twos = 0
        for x in nums:
            ones = (ones ^ x) & ~twos
            twos = (twos ^ x) & ~ones
        return ones
```

</td>
  </tr>
</tbody>
</table>

---

## 4. Bit Reversal & Arithmetic Without Addition Operators

<table>
<thead>
  <tr>
    <th style="width: 38%;">Problem &amp; Pattern</th>
    <th>Core Bit Manipulation Implementation (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 190. Reverse Bits</strong><br/><br/>
      • <strong>Problem</strong>: Reverse bits of a 32-bit unsigned integer.<br/>
      • <strong>Pattern</strong>: Iterate 32 times: shift `res` left, insert `n & 1`, shift `n` right.<br/>
      • <strong>Complexity</strong>: Time $O(1)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res << 1) | (n & 1)
            n >>= 1
        return res
```

</td>
  </tr>
  <tr>
    <td>
      <strong>LeetCode 371. Sum of Two Integers</strong><br/><br/>
      • <strong>Problem</strong>: Add two integers without `+` or `-` operators.<br/>
      • <strong>Pattern</strong>: Sum without carry is `a ^ b`; carry is `(a & b) << 1`. In Python, clamp to 32 bits using mask `0xFFFFFFFF` to avoid arbitrary-precision integer expansion.<br/>
      • <strong>Complexity</strong>: Time $O(1)$, Space $O(1)$.
    </td>
    <td>

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF
        max_int = 0x7FFFFFFF
        while b != 0:
            carry = ((a & b) << 1) & mask
            a = (a ^ b) & mask
            b = carry
        return a if a <= max_int else ~(a ^ mask)
```

</td>
  </tr>
</tbody>
</table>

---

## 5. Bitmask State Compression & Submask Enumeration

<table>
<thead>
  <tr>
    <th style="width: 38%;">Problem &amp; Pattern</th>
    <th>Core Bit Manipulation Implementation (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 78. Subsets (Power Set via Bitmask)</strong><br/><br/>
      • <strong>Problem</strong>: Return all subsets of an array of distinct integers.<br/>
      • <strong>Pattern</strong>: Numbers $0 \sim 2^n - 1$ each represent a subset mask; the $i$-th bit indicates inclusion of `nums[i]`.<br/>
      • <strong>Complexity</strong>: Time $O(n \cdot 2^n)$, Space $O(1)$ (excluding output).
    </td>
    <td>

```python
class Solution:
    def subsets(self, nums: list[int]) -> list[list[int]]:
        n = len(nums)
        res = []
        for mask in range(1 << n):
            sub = [nums[i] for i in range(n) if (mask >> i) & 1]
            res.append(sub)
        return res
```

</td>
  </tr>
  <tr>
    <td>
      <strong>Submask Fast Enumeration Trick</strong><br/><br/>
      • <strong>Pattern</strong>: Given a binary `mask`, enumerate all its non-empty submasks in strictly decreasing order in $O(2^k)$.<br/>
      • <strong>Mechanism</strong>: `sub = (sub - 1) & mask`: subtracting 1 borrows bits while `& mask` filters out non-member bits.<br/>
      • <strong>Application</strong>: Bitmask DP state transitions (TSP, partition DP).
    </td>
    <td>

```python
class Solution:
    def enumerateSubmasks(self, mask: int) -> list[int]:
        submasks = []
        sub = mask
        while sub > 0:
            submasks.append(sub)
            sub = (sub - 1) & mask
        submasks.append(0)  # Append the empty subset mask 0
        return submasks
```

</td>
  </tr>
</tbody>
</table>

---

## 6. Quick-Recall Bit Manipulation Formula Card

| Core Pattern | Canonical Expression | Representative Problems |
|---|---|---|
| **XOR Cancellation** | `res ^= x` | LC 136, LC 268, LC 389 |
| **Clear Lowest 1-Bit** | `n &= n - 1` | LC 191, LC 231, LC 338 |
| **Extract Lowest 1-Bit** | `lowbit = n & (-n)` | LC 260, Fenwick Tree |
| **Modulo-3 State Machine** | `ones = (ones ^ x) & ~twos; twos = (twos ^ x) & ~ones` | LC 137 |
| **Carryless Addition** | `carry = (a & b) << 1; a = a ^ b` | LC 371 |
| **Submask Enumeration** | `sub = (sub - 1) & mask` | Bitmask DP |

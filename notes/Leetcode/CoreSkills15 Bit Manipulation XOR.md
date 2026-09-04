# Bit Manipulation：LeetCode & NeetCode 常见位运算套路速查

位运算的核心：**利用位级并行操作消除条件分支与循环，实现 $O(1)$ 空间或极值时间复杂度。**  
本速查表整理 LeetCode 与 NeetCode 150 高频位运算模式：左侧为题目与核心考点，右侧为核心 Python 实现。

---

## 0. 基础位操作原子速查 (Bit Primitives)

<table>
<thead>
  <tr>
    <th style="width: 38%;">操作模式 / 技巧目标</th>
    <th>Python 核心表达式 / 操作</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><strong>获取第 $i$ 位 (Get Bit $i$)</strong><br/>提取整数 $n$ 的第 $i$ 个二进制位（0-indexed）</td>
    <td>

```python
bit = (n >> i) & 1
```

</td>
  </tr>
  <tr>
    <td><strong>置位 (Set Bit $i$)</strong><br/>将第 $i$ 位强制设为 1，其余位不变</td>
    <td>

```python
n = n | (1 << i)
```

</td>
  </tr>
  <tr>
    <td><strong>清零 (Clear Bit $i$)</strong><br/>将第 $i$ 位强制清零为 0，其余位不变</td>
    <td>

```python
n = n & ~(1 << i)
```

</td>
  </tr>
  <tr>
    <td><strong>翻转 (Toggle Bit $i$)</strong><br/>第 $i$ 位 0 变 1、1 变 0</td>
    <td>

```python
n = n ^ (1 << i)
```

</td>
  </tr>
  <tr>
    <td><strong>抹去最低位的 1 (Brian Kernighan)</strong><br/>消去二进制中最右侧的 1，其余位不变</td>
    <td>

```python
n = n & (n - 1)
```

</td>
  </tr>
  <tr>
    <td><strong>提取最低位的 1 (lowbit)</strong><br/>仅保留最右侧的 1，其余位全部清零</td>
    <td>

```python
lowbit = n & (-n)
```

</td>
  </tr>
  <tr>
    <td><strong>判断奇偶性 (Parity Check)</strong><br/>最低位为 1 是奇数，为 0 是偶数</td>
    <td>

```python
is_odd = bool(n & 1)
```

</td>
  </tr>
</tbody>
</table>

---

## 1. 成对异或消除族 (XOR Cancellation)

利用异或核心性质：$x \oplus x = 0$，$x \oplus 0 = x$，满足交换律与结合律。

<table>
<thead>
  <tr>
    <th style="width: 38%;">题目与考点 (Problem & Pattern)</th>
    <th>核心位运算实现 (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 136. Single Number (只出现一次的数字)</strong><br/><br/>
      • <strong>题意</strong>：其余元素均出现 2 次，恰好 1 个元素出现 1 次。<br/>
      • <strong>套路</strong>：全员异或，成对元素 $x \oplus x = 0$ 抵消。<br/>
      • <strong>复杂度</strong>：时间 $O(n)$，空间 $O(1)$。
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
      <strong>LeetCode 268. Missing Number (丢失的数字)</strong><br/><br/>
      • <strong>题意</strong>：长度为 $n$ 的数组包含 $[0, n]$ 内 $n$ 个数，求缺失的 1 个数。<br/>
      • <strong>套路</strong>：将完整集合 $[0..n]$ 与数组所有数异或，成对抵消。<br/>
      • <strong>复杂度</strong>：时间 $O(n)$，空间 $O(1)$。
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
      <strong>LeetCode 389. Find the Difference (找不同)</strong><br/><br/>
      • <strong>题意</strong>：字符串 $t$ 由 $s$ 随机打乱并随机插入 1 个字母组成，求该字母。<br/>
      • <strong>套路</strong>：遍历拼接字符串 $s + t$，按 ASCII 码累积异或。<br/>
      • <strong>复杂度</strong>：时间 $O(n)$，空间 $O(1)$。
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
      <strong>LeetCode 260. Single Number III (只出现一次的数字 III)</strong><br/><br/>
      • <strong>题意</strong>：恰好两个元素各出现 1 次，其余元素均出现 2 次。<br/>
      • <strong>套路</strong>：全量异或得 $a \oplus b$；取 `lowbit = diff & (-diff)` 区分二者；按该位是 0 还是 1 分组异或。<br/>
      • <strong>复杂度</strong>：时间 $O(n)$，空间 $O(1)$。
    </td>
    <td>

```python
class Solution:
    def singleNumber(self, nums: list[int]) -> list[int]:
        diff = 0
        for x in nums:
            diff ^= x
        # 提取 a 和 b 最低不同位
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

## 2. 最低位消除与汉明统计族 (Brian Kernighan & Lowbit)

利用 `n & (n - 1)` 消除最低 1 位，避开 32 次全量扫描，循环步数严格等于 1 的个数。

<table>
<thead>
  <tr>
    <th style="width: 38%;">题目与考点 (Problem & Pattern)</th>
    <th>核心位运算实现 (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 191. Number of 1 Bits (位 1 的个数)</strong><br/><br/>
      • <strong>题意</strong>：求无符号整数二进制中 1 的个数（汉明重量）。<br/>
      • <strong>套路</strong>：每次执行 `n &= n - 1` 抹去最右侧 1，循环次数等于 1 的数目。<br/>
      • <strong>复杂度</strong>：时间 $O(k)$（$k \le 32$ 为 1 的个数），空间 $O(1)$。
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
      <strong>LeetCode 231. Power of Two (2 的幂)</strong><br/><br/>
      • <strong>题意</strong>：判断整数 $n$ 是否为 2 的幂次方。<br/>
      • <strong>套路</strong>：2 的幂必为正整数且二进制表示中仅有唯一的 1。<br/>
      • <strong>复杂度</strong>：时间 $O(1)$，空间 $O(1)$。
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
      <strong>LeetCode 342. Power of Four (4 的幂)</strong><br/><br/>
      • <strong>题意</strong>：判断整数 $n$ 是否为 4 的幂次方。<br/>
      • <strong>套路</strong>：需满足为 2 的幂，且唯一的 1 必须位于奇数索引位（与 `0x55555555` 相与不为 0）。<br/>
      • <strong>复杂度</strong>：时间 $O(1)$，空间 $O(1)$。
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
      <strong>LeetCode 338. Counting Bits (比特位计数)</strong><br/><br/>
      • <strong>题意</strong>：计算区间 $[0, n]$ 内所有整数二进制 1 的个数。<br/>
      • <strong>套路</strong>：DP 递推 `dp[i] = dp[i & (i - 1)] + 1`，消除最低 1 后查表。<br/>
      • <strong>复杂度</strong>：时间 $O(n)$，空间 $O(1)$（不计输出数组）。
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
      <strong>LeetCode 461. Hamming Distance (汉明距离)</strong><br/><br/>
      • <strong>题意</strong>：计算两数二进制位不同的位置数量。<br/>
      • <strong>套路</strong>：`x ^ y` 提取不同位，再用 Kernighan 统计 1 的数目。<br/>
      • <strong>复杂度</strong>：时间 $O(k)$，空间 $O(1)$。
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

## 3. 模 K 状态机与逐位统计族 (Modulo K & Bit State Machine)

<table>
<thead>
  <tr>
    <th style="width: 38%;">题目与考点 (Problem & Pattern)</th>
    <th>核心位运算实现 (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 137. Single Number II (只出现一次的数字 II)</strong><br/><br/>
      • <strong>题意</strong>：其余所有元素均出现 3 次，恰好 1 个元素出现 1 次。<br/>
      • <strong>套路</strong>：构建模 3 状态机（$00 \to 01 \to 10 \to 00$）。`ones` 与 `twos` 分别追踪位出现 1 次与 2 次的状态，满 3 次自动复位。<br/>
      • <strong>复杂度</strong>：时间 $O(n)$，空间 $O(1)$。
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

## 4. 位逆序与无加号算术 (Bit Reversal & Arithmetic)

<table>
<thead>
  <tr>
    <th style="width: 38%;">题目与考点 (Problem & Pattern)</th>
    <th>核心位运算实现 (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 190. Reverse Bits (颠倒二进制位)</strong><br/><br/>
      • <strong>题意</strong>：颠倒 32 位无符号整数的所有二进制位。<br/>
      • <strong>套路</strong>：固定 32 次循环，结果左移腾位后装入 `n & 1`，输入 `n` 右移。<br/>
      • <strong>复杂度</strong>：时间 $O(1)$，空间 $O(1)$。
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
      <strong>LeetCode 371. Sum of Two Integers (两整数之和)</strong><br/><br/>
      • <strong>题意</strong>：不使用 `+` 和 `-` 计算两整数之和。<br/>
      • <strong>套路</strong>：无进位和为 `a ^ b`，进位为 `(a & b) << 1`。Python 需加 `0xFFFFFFFF` 掩码限制在 32 位防止大数扩张。<br/>
      • <strong>复杂度</strong>：时间 $O(1)$，空间 $O(1)$。
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

## 5. 状态压缩与子集遍历 (Bitmask & Submask Enumeration)

<table>
<thead>
  <tr>
    <th style="width: 38%;">题目与考点 (Problem & Pattern)</th>
    <th>核心位运算实现 (Python)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>
      <strong>LeetCode 78. Subsets (二进制掩码枚举子集)</strong><br/><br/>
      • <strong>题意</strong>：生成不含重复元素的数组的所有子集（幂集）。<br/>
      • <strong>套路</strong>：$0 \sim 2^n - 1$ 每个数值的第 $i$ 个 bit 代表是否选取 `nums[i]`。<br/>
      • <strong>复杂度</strong>：时间 $O(n \cdot 2^n)$，空间 $O(1)$（不计输出）。
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
      <strong>子掩码快速遍历模板 (Submask Enumeration Trick)</strong><br/><br/>
      • <strong>模式</strong>：给定二进制状态 `mask`，在 $O(2^k)$ 内严格降序遍历所有非空子掩码。<br/>
      • <strong>套路</strong>：`sub = (sub - 1) & mask`，借位减 1 配合与运算跳过无效状态。<br/>
      • <strong>应用</strong>：状压 DP 转移（旅行商 TSP、划分型 DP）。
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
        submasks.append(0)  # 补充空集状态 0
        return submasks
```

</td>
  </tr>
</tbody>
</table>

---

## 6. 位运算口诀速记卡

| 核心模式 | 关键代码公式 | 典型题目 |
|---|---|---|
| **成对异或消除** | `res ^= x` | LC 136, LC 268, LC 389 |
| **最低 1 位消除** | `n &= n - 1` | LC 191, LC 231, LC 338 |
| **最低 1 位提取** | `lowbit = n & (-n)` | LC 260, 树状数组 (Fenwick) |
| **模 3 状态机** | `ones = (ones ^ x) & ~twos; twos = (twos ^ x) & ~ones` | LC 137 |
| **无加号进位加** | `carry = (a & b) << 1; a = a ^ b` | LC 371 |
| **子掩码降序遍历** | `sub = (sub - 1) & mask` | 状压 DP |

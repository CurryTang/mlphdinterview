# Bit Manipulation：XOR / Single Number & Missing Number

位运算题的核心不是记很多符号，而是知道每个操作在二进制位上到底做了什么。

这一页从两个最经典的 XOR 例题开始：

```text
LeetCode 136: Single Number
LeetCode 268: Missing Number
```

第一题给定一个数组：

- 只有一个数字出现一次
- 其他数字都出现两次

返回那个只出现一次的数字。

第二题给定长度为 `n` 的数组，元素来自区间 `[0, n]`，其中恰好缺失一个数字。要求返回缺失值。

---

## 目录

1. [XOR 是什么](#xor-是什么)
2. [Single Number](#single-number)
3. [为什么 XOR 可以消掉重复数字](#为什么-xor-可以消掉重复数字)
4. [可视化 walkthrough](#可视化-walkthrough)
5. [Single Number 代码](#single-number-代码)
6. [Missing Number](#missing-number)
7. [为什么 Missing Number 也能用 XOR](#为什么-missing-number-也能用-xor)
8. [Missing Number 可视化 walkthrough](#missing-number-可视化-walkthrough)
9. [Missing Number 代码](#missing-number-代码)
10. [复杂度](#复杂度)
11. [常见坑](#常见坑)
12. [一句话记忆](#一句话记忆)

---

## XOR 是什么

XOR 写作 `^`。

它的规则是：

```text
相同为 0，不同为 1
```

| a | b | a ^ b |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

所以：

```text
5 = 0101
3 = 0011
---------
5 ^ 3 = 0110 = 6
```

XOR 有三个最重要的性质：

### 1. 自己和自己异或会消掉

```text
x ^ x = 0
```

因为每一位都相同。

例子：

```text
7 ^ 7 = 0
```

### 2. 任何数和 0 异或还是自己

```text
x ^ 0 = x
```

因为 `0` 不会改变任何 bit。

例子：

```text
7 ^ 0 = 7
```

### 3. XOR 满足交换律和结合律

```text
a ^ b = b ^ a
(a ^ b) ^ c = a ^ (b ^ c)
```

这意味着顺序不重要。

所以：

```text
4 ^ 1 ^ 2 ^ 1 ^ 2
= 4 ^ (1 ^ 1) ^ (2 ^ 2)
= 4 ^ 0 ^ 0
= 4
```

这就是 Single Number 的全部核心。

---

## Single Number

### 题目

给定整数数组 `nums`，其中恰好一个元素只出现一次，其余每个元素都出现两次。

返回只出现一次的元素。

例子：

```text
Input: nums = [2, 2, 1]
Output: 1
```

```text
Input: nums = [4, 1, 2, 1, 2]
Output: 4
```

---

## 为什么 XOR 可以消掉重复数字

如果用哈希表，当然可以做：

```text
统计每个数字出现次数
返回 count == 1 的数字
```

但这需要 `O(n)` 额外空间。

XOR 的做法更优雅：

```text
把所有数字全部 XOR 起来
成对出现的数字会变成 0
最后剩下的就是 single number
```

对 `[4, 1, 2, 1, 2]`：

```text
res = 0

res = 0 ^ 4
res = 4 ^ 1
res = 4 ^ 1 ^ 2
res = 4 ^ 1 ^ 2 ^ 1
res = 4 ^ 1 ^ 2 ^ 1 ^ 2
```

因为 XOR 可以重排：

```text
4 ^ 1 ^ 2 ^ 1 ^ 2
= 4 ^ (1 ^ 1) ^ (2 ^ 2)
= 4 ^ 0 ^ 0
= 4
```

所以答案是 `4`。

---

## 可视化 walkthrough

以：

```text
nums = [4, 1, 2, 1, 2]
```

为例。

| step | num | res before | res = res ^ num | 为什么 |
|---:|---:|---:|---:|---|
| 0 | 4 | 0 | 4 | `0 ^ 4 = 4` |
| 1 | 1 | 4 | 5 | 先暂时混在一起 |
| 2 | 2 | 5 | 7 | 继续累积 bit 信息 |
| 3 | 1 | 7 | 6 | 第二个 `1` 把第一个 `1` 消掉 |
| 4 | 2 | 6 | 4 | 第二个 `2` 把第一个 `2` 消掉 |

用二进制看最后两步更直观：

```text
7 = 0111
1 = 0001
---------
6 = 0110
```

这里 `1` 的最低位被消掉了。

再 XOR `2`：

```text
6 = 0110
2 = 0010
---------
4 = 0100
```

`2` 对应的 bit 也被消掉，最后只剩 `4`。

---

## Single Number 代码

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res = res ^ num
        return res
```

也可以写成：

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res
```

`res ^= num` 等价于：

```python
res = res ^ num
```

## Missing Number

### 题目

给定一个长度为 `n` 的数组 `nums`，其中包含 `[0, n]` 范围内的 `n` 个不同数字。

因为 `[0, n]` 一共有 `n + 1` 个数字，而数组里只有 `n` 个数字，所以恰好缺失一个数字。返回缺失的那个数字。

例子：

```text
Input: nums = [3, 0, 1]
Output: 2
```

```text
Input: nums = [0, 1]
Output: 2
```

---

## 为什么 Missing Number 也能用 XOR

Missing Number 的本质也是“成对抵消”。

完整集合应该是：

```text
0, 1, 2, ..., n
```

数组里实际出现的是：

```text
nums[0], nums[1], ..., nums[n - 1]
```

如果把完整集合和数组里的所有数字都 XOR 在一起：

```text
0 ^ 1 ^ 2 ^ ... ^ n ^ nums[0] ^ nums[1] ^ ... ^ nums[n - 1]
```

出现过的数字会在两边各出现一次，因此全部抵消：

```text
x ^ x = 0
```

最后只剩下那个没有在 `nums` 里出现的数字。

代码里不需要真的先构造 `[0, 1, ..., n]`。可以一边遍历，一边把索引 `i` 和数字 `num` 都 XOR 进去：

```text
res = n
for each i, num:
  res = res ^ i ^ num
```

为什么一开始是 `res = n`？

因为遍历数组时，索引只会覆盖：

```text
0, 1, ..., n - 1
```

完整集合还差最后一个边界值 `n`，所以先把 `n` 放进 `res`。

---

## Missing Number 可视化 walkthrough

以：

```text
nums = [3, 0, 1]
n = 3
```

完整集合应该是：

```text
0, 1, 2, 3
```

数组里有：

```text
3, 0, 1
```

缺的是 `2`。

用代码的扫描方式：

| step | i | num | res before | res = res ^ i ^ num | 抵消关系 |
|---:|---:|---:|---:|---:|---|
| init | - | - | - | 3 | 先放入边界值 `n` |
| 0 | 0 | 3 | 3 | 0 | `3 ^ 3 = 0`，`0` 不改变结果 |
| 1 | 1 | 0 | 0 | 1 | 加入索引 `1`，`0` 不改变结果 |
| 2 | 2 | 1 | 1 | 2 | `1 ^ 1 = 0`，剩下 `2` |

整体看就是：

```text
res = 3 ^ (0 ^ 3) ^ (1 ^ 0) ^ (2 ^ 1)
    = (3 ^ 3) ^ (0 ^ 0) ^ (1 ^ 1) ^ 2
    = 0 ^ 0 ^ 0 ^ 2
    = 2
```

这个写法的关键不是“索引有什么特殊魔法”，而是索引 `0..n-1` 加上初始值 `n`，刚好组成完整集合 `0..n`。

---

## Missing Number 代码

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        res = len(nums)

        for i, num in enumerate(nums):
            res ^= i ^ num

        return res
```

也可以写得更展开：

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        res = n

        for i, num in enumerate(nums):
            res = res ^ i
            res = res ^ num

        return res
```

两段代码完全等价。

---

## 复杂度

两题都只扫描数组一次。

```text
Time:  O(n)
Space: O(1)
```

这个 `O(1)` 是 XOR 做法的关键优势：不需要哈希表，不需要排序，也不需要额外数组。

---

## 常见坑

### 1. 把 XOR 理解成加法

XOR 不是加法。

```text
1 ^ 1 = 0
```

而不是 `2`。

XOR 是逐 bit 比较：

```text
same -> 0
different -> 1
```

### 2. 担心负数不能用 XOR

Python 的负数也可以做 XOR。

LeetCode 这题不需要你手动处理二进制补码细节。只要题目保证其他数字都出现两次，`x ^ x = 0` 仍然成立。

### 3. 不理解为什么顺序无所谓

因为 XOR 满足交换律和结合律。

数组原顺序是：

```text
4 ^ 1 ^ 2 ^ 1 ^ 2
```

但数学上可以看成：

```text
4 ^ (1 ^ 1) ^ (2 ^ 2)
```

这就是为什么一次线性扫描就够。

### 4. Missing Number 忘记先 XOR `n`

Missing Number 的数组索引只到 `n - 1`。

如果初始化成：

```python
res = 0
```

然后只遍历 `i` 和 `num`，完整集合里最后的 `n` 就没有被放进去。

所以要写：

```python
res = len(nums)
```

### 5. 把 Missing Number 和 Single Number 混成同一个输入条件

Single Number 是：

```text
一个数字出现一次，其他数字出现两次
```

Missing Number 是：

```text
数组长度为 n，数字来自 0..n，少一个数字
```

它们都用 XOR，但配对对象不同：

```text
Single Number: 数字和数字配对
Missing Number: 完整索引集合和数组数字配对
```

### 6. Single Number 的输入条件变了不能直接套

Single Number 的简单 XOR 代码只适用于：

```text
一个数字出现一次，其他数字都出现两次
```

如果题目变成：

```text
一个数字出现一次，其他数字都出现三次
```

就不能只用简单 XOR，需要统计每一位出现次数并对 `3` 取模。

---

## 一句话记忆

```text
Single Number = 数组全部 XOR 一遍。
成对数字 x ^ x = 0，
0 ^ single = single。
```

```text
Missing Number = 完整集合 0..n 和 nums 全部 XOR。
出现过的数字会两两抵消，
最后剩下缺失值。
```

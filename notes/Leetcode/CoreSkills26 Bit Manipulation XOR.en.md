# Bit Manipulation: XOR / Single Number & Missing Number

The core of bit manipulation problems is not memorizing many symbols, but understanding exactly what each operation does at the binary-bit level.

This page starts with the two most classic XOR examples:

```text
LeetCode 136: Single Number
LeetCode 268: Missing Number
```

The first problem gives an array:

- only one number appears once
- all other numbers appear twice

Return the number that appears only once.

The second problem gives an array of length `n`, whose elements come from the range `[0, n]`, with exactly one number missing. You need to return the missing value.

---

## Table of Contents

1. [What XOR Is](#what-xor-is)
2. [Single Number](#single-number)
3. [Why XOR Can Cancel Out Duplicate Numbers](#why-xor-can-cancel-out-duplicate-numbers)
4. [Visual Walkthrough](#visual-walkthrough)
5. [Single Number Code](#single-number-code)
6. [Missing Number](#missing-number)
7. [Why Missing Number Can Also Use XOR](#why-missing-number-can-also-use-xor)
8. [Missing Number Visual Walkthrough](#missing-number-visual-walkthrough)
9. [Missing Number Code](#missing-number-code)
10. [Complexity](#complexity)
11. [Common Pitfalls](#common-pitfalls)
12. [One-Sentence Memory Aid](#one-sentence-memory-aid)

---

## What XOR Is

XOR is written as `^`.

Its rule is:

```text
same gives 0, different gives 1
```

| a | b | a ^ b |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

So:

```text
5 = 0101
3 = 0011
---------
5 ^ 3 = 0110 = 6
```

XOR has three most important properties:

### 1. XOR with itself cancels out

```text
x ^ x = 0
```

Because every bit is the same.

Example:

```text
7 ^ 7 = 0
```

### 2. XOR with 0 leaves the number unchanged

```text
x ^ 0 = x
```

Because `0` does not change any bit.

Example:

```text
7 ^ 0 = 7
```

### 3. XOR satisfies commutativity and associativity

```text
a ^ b = b ^ a
(a ^ b) ^ c = a ^ (b ^ c)
```

This means the order does not matter.

So:

```text
4 ^ 1 ^ 2 ^ 1 ^ 2
= 4 ^ (1 ^ 1) ^ (2 ^ 2)
= 4 ^ 0 ^ 0
= 4
```

That is the entire core of Single Number.

---

## Single Number

### Problem

Given an integer array `nums`, in which exactly one element appears once and every other element appears twice.

Return the element that appears only once.

Example:

```text
Input: nums = [2, 2, 1]
Output: 1
```

```text
Input: nums = [4, 1, 2, 1, 2]
Output: 4
```

---

## Why XOR Can Cancel Out Duplicate Numbers

If you use a hash table, of course you can do:

```text
count the occurrences of each number
return the number with count == 1
```

But this requires `O(n)` extra space.

The XOR approach is more elegant:

```text
XOR all numbers together
numbers that appear in pairs become 0
the remaining value is the single number
```

For `[4, 1, 2, 1, 2]`:

```text
res = 0

res = 0 ^ 4
res = 4 ^ 1
res = 4 ^ 1 ^ 2
res = 4 ^ 1 ^ 2 ^ 1
res = 4 ^ 1 ^ 2 ^ 1 ^ 2
```

Because XOR can be reordered:

```text
4 ^ 1 ^ 2 ^ 1 ^ 2
= 4 ^ (1 ^ 1) ^ (2 ^ 2)
= 4 ^ 0 ^ 0
= 4
```

So the answer is `4`.

---

## Visual Walkthrough

Take:

```text
nums = [4, 1, 2, 1, 2]
```

as the example.

| step | num | res before | res = res ^ num | Why |
|---:|---:|---:|---:|---|
| 0 | 4 | 0 | 4 | `0 ^ 4 = 4` |
| 1 | 1 | 4 | 5 | temporarily mixed together |
| 2 | 2 | 5 | 7 | continue accumulating bit information |
| 3 | 1 | 7 | 6 | the second `1` cancels out the first `1` |
| 4 | 2 | 6 | 4 | the second `2` cancels out the first `2` |

Looking at the last two steps in binary makes it more intuitive:

```text
7 = 0111
1 = 0001
---------
6 = 0110
```

Here, the lowest bit of `1` is canceled out.

Then XOR with `2`:

```text
6 = 0110
2 = 0010
---------
4 = 0100
```

The bit corresponding to `2` is also canceled out, leaving only `4` in the end.

---

## Single Number Code

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res = res ^ num
        return res
```

It can also be written as:

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for num in nums:
            res ^= num
        return res
```

`res ^= num` is equivalent to:

```python
res = res ^ num
```

## Missing Number

### Problem

Given an array `nums` of length `n`, containing `n` distinct numbers from the range `[0, n]`.

Since `[0, n]` contains `n + 1` numbers in total while the array contains only `n` numbers, exactly one number is missing. Return that missing number.

Example:

```text
Input: nums = [3, 0, 1]
Output: 2
```

```text
Input: nums = [0, 1]
Output: 2
```

---

## Why Missing Number Can Also Use XOR

The essence of Missing Number is also "pairwise cancellation."

The complete set should be:

```text
0, 1, 2, ..., n
```

What actually appears in the array is:

```text
nums[0], nums[1], ..., nums[n - 1]
```

If we XOR the complete set and all numbers in the array together:

```text
0 ^ 1 ^ 2 ^ ... ^ n ^ nums[0] ^ nums[1] ^ ... ^ nums[n - 1]
```

The numbers that have appeared will each appear once on both sides, so they all cancel out:

```text
x ^ x = 0
```

In the end, only the number that does not appear in `nums` remains.

In code, you do not need to literally construct `[0, 1, ..., n]` first. You can traverse once and XOR both the index `i` and the number `num` into the result:

```text
res = n
for each i, num:
  res = res ^ i ^ num
```

Why do we start with `res = n`?

Because when traversing the array, the indices cover only:

```text
0, 1, ..., n - 1
```

The complete set is still missing the final boundary value `n`, so we put `n` into `res` first.

---

## Missing Number Visual Walkthrough

Take:

```text
nums = [3, 0, 1]
n = 3
```

The complete set should be:

```text
0, 1, 2, 3
```

The array contains:

```text
3, 0, 1
```

The missing value is `2`.

Using the scan method from the code:

| step | i | num | res before | res = res ^ i ^ num | Cancellation relation |
|---:|---:|---:|---:|---:|---|
| init | - | - | - | 3 | put the boundary value `n` in first |
| 0 | 0 | 3 | 3 | 0 | `3 ^ 3 = 0`, and `0` does not change the result |
| 1 | 1 | 0 | 0 | 1 | add index `1`, and `0` does not change the result |
| 2 | 2 | 1 | 1 | 2 | `1 ^ 1 = 0`, leaving `2` |

Viewed as a whole:

```text
res = 3 ^ (0 ^ 3) ^ (1 ^ 0) ^ (2 ^ 1)
    = (3 ^ 3) ^ (0 ^ 0) ^ (1 ^ 1) ^ 2
    = 0 ^ 0 ^ 0 ^ 2
    = 2
```

The key point of this form is not that "indices have some special magic," but that indices `0..n-1` plus the initial value `n` happen to form the complete set `0..n`.

---

## Missing Number Code

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        res = len(nums)

        for i, num in enumerate(nums):
            res ^= i ^ num

        return res
```

It can also be written in a more expanded form:

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

The two code blocks are completely equivalent.

---

## Complexity

Both problems scan the array only once.

```text
Time:  O(n)
Space: O(1)
```

This `O(1)` is the key advantage of the XOR approach: no hash table, no sorting, and no extra array are needed.

---

## Common Pitfalls

### 1. Treating XOR as addition

XOR is not addition.

```text
1 ^ 1 = 0
```

not `2`.

XOR compares bit by bit:

```text
same -> 0
different -> 1
```

### 2. Worrying that negative numbers cannot use XOR

Negative numbers in Python can also use XOR.

This LeetCode problem does not require you to manually handle details of binary two's complement. As long as the problem guarantees that all other numbers appear twice, `x ^ x = 0` still holds.

### 3. Not understanding why the order does not matter

Because XOR satisfies commutativity and associativity.

The original array order is:

```text
4 ^ 1 ^ 2 ^ 1 ^ 2
```

But mathematically it can be viewed as:

```text
4 ^ (1 ^ 1) ^ (2 ^ 2)
```

That is why a single linear scan is enough.

### 4. Forgetting to XOR `n` first in Missing Number

The array indices in Missing Number only go up to `n - 1`.

If you initialize with:

```python
res = 0
```

and then only traverse `i` and `num`, the final `n` in the complete set is never included.

So you should write:

```python
res = len(nums)
```

### 5. Mixing up the input conditions of Missing Number and Single Number

Single Number is:

```text
one number appears once, and all other numbers appear twice
```

Missing Number is:

```text
the array length is n, the numbers come from 0..n, and one number is missing
```

They both use XOR, but the pairing targets are different:

```text
Single Number: numbers are paired with numbers
Missing Number: the complete index set is paired with array numbers
```

### 6. If Single Number changes its input condition, you cannot apply the same pattern directly

The simple XOR code for Single Number only applies to:

```text
one number appears once, and all other numbers appear twice
```

If the problem changes to:

```text
one number appears once, and all other numbers appear three times
```

then simple XOR alone no longer works. You need to count the occurrences of each bit and take modulo `3`.

---

## One-Sentence Memory Aid

```text
Single Number = XOR the entire array once.
Paired numbers satisfy x ^ x = 0,
and 0 ^ single = single.
```

```text
Missing Number = XOR the complete set 0..n and nums together.
Numbers that appeared will cancel in pairs,
and the missing value is what remains in the end.
```

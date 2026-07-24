# Math: Fast Power / Binary Exponentiation

Math problems are not about memorizing formulas, but about breaking a big problem into repeatable small structures.

This page starts with the most common first problem:

```text
LeetCode 50: Pow(x, n)
```

The problem asks you to implement:

```text
pow(x, n) = x^n
```

Here, `n` may be very large, and it may also be negative.

---

## Problem 1: Pow(x, n)

### Problem

Given a floating-point number `x` and an integer `n`, return `x^n`.

Example:

```text
Input:  x = 2.00000, n = 10
Output: 1024.00000
```

```text
Input:  x = 2.00000, n = -2
Output: 0.25000
```

Because:

```text
2^-2 = 1 / 2^2 = 1 / 4
```

## Why You Cannot Brute-Force Multiply n Times

The most direct approach is:

```python
ans = 1
for _ in range(n):
    ans *= x
```

This requires `O(n)` multiplications.

If `n = 2^31 - 1`, this approach is too slow.

The goal of binary exponentiation is to reduce the complexity to:

```text
O(log n)
```

The core reason is:

```text
x^10 = x^(8 + 2)
```

And the binary form of `10` is:

```text
10 = 1010₂ = 8 + 2
```

So we do not need to multiply `x` one by one. We only need to determine which binary bits are `1`.

## Core Intuition

In each round, maintain two variables:

```text
base = the power represented by the current bit
res  = the product of the powers already selected
```

Read the binary bits of `n` from right to left.

For `n = 10 = 1010₂`:

```text
bit weights: 8 4 2 1
bit values:  1 0 1 0
```

Starting from the lowest bit:

```text
do not select the 1 position
select the 2 position
do not select the 4 position
select the 8 position
```

So:

```text
x^10 = x^2 * x^8
```

That is why there are two things in the code:

```python
if power & 1:
    res *= x
```

If the lowest bit is `1`, it means the current power contribution of `x` needs to be multiplied into the answer.

Then:

```python
x *= x
power >>= 1
```

`x *= x` means the current power doubles:

```text
x^1 -> x^2 -> x^4 -> x^8 -> ...
```

`power >>= 1` means dropping the lowest binary bit and continuing to inspect the next bit.

## Visualization: Why It Is `res *= base`, Then `base *= base`

```pow-demo
```

Expand `pow(2, 10)`:

```text
10 = 1010₂
```

We read the bits from right to left.

| Current `power` | Lowest bit | Current `base` | Action | `res` |
|---|---:|---:|---|---:|
| 10 | 0 | 2 | Do not multiply; square `base` | 1 |
| 5 | 1 | 4 | Multiply into `res` | 4 |
| 2 | 0 | 16 | Do not multiply; square `base` | 4 |
| 1 | 1 | 256 | Multiply into `res` | 1024 |

Finally:

```text
res = 4 * 256 = 2^2 * 2^8 = 2^10
```

## Iterative Binary Exponentiation

### Intuition

We want to compute `x^n` efficiently, even when `n` is very large.

The brute force approach multiplies `x` repeatedly and takes `O(n)` time.

Binary exponentiation uses the binary representation of `n`.

At each step:

- if the current lowest bit is `1`, multiply the current base into `res`
- square the base
- shift the exponent right by one bit

For negative powers:

```text
x^(-n) = 1 / x^n
```

So we compute with `abs(n)` and return the reciprocal at the end.

### Algorithm

1. If `x == 0`, return `0`.
2. If `n == 0`, return `1`.
3. Set `res = 1`.
4. Set `power = abs(n)`.
5. While `power > 0`:
   - if `power & 1`, do `res *= x`
   - do `x *= x`
   - do `power >>= 1`
6. If `n < 0`, return `1 / res`; otherwise return `res`.

### Code

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0:
            return 0
        if n == 0:
            return 1

        res = 1
        power = abs(n)

        while power:
            if power & 1:
                res *= x
            x *= x
            power >>= 1

        return res if n >= 0 else 1 / res
```

## Complexity

In every round, `power` is divided by 2.

So the number of iterations is:

```text
log2(|n|)
```

Complexity:

```text
Time:  O(log n)
Space: O(1)
```

## Common Pitfalls

### 1. Forgetting to handle negative exponents

Incorrect version:

```python
power = n
```

If `n < 0`, the loop logic is incorrect.

The correct approach is:

```python
power = abs(n)
return res if n >= 0 else 1 / res
```

### 2. Not understanding `power & 1`

`power & 1` checks whether the current lowest bit is `1`.

Equivalent to:

```python
power % 2 == 1
```

But bit operations are closer to the essence of this algorithm: reading the binary representation one bit at a time.

### 3. Not understanding why `x *= x`

Each time we right-shift `power`, we move on to the next binary bit.

The weight of the next bit doubles:

```text
1 -> 2 -> 4 -> 8 -> 16
```

So the current base must also be squared:

```text
x^1 -> x^2 -> x^4 -> x^8 -> x^16
```

### 4. `x == 0` and negative exponents

LeetCode usually will not require you to handle mathematically undefined cases like a negative exponent for `0`.

If an interviewer follows up:

```text
0^-1 = 1 / 0
```

This is undefined / division by zero and should be handled according to the problem's contract.

## One-Sentence Memory Aid

```text
Binary exponentiation = read the binary representation of n from right to left.
If the bit is 1, multiply the current base into res;
in each round, square the base and right-shift power.
```

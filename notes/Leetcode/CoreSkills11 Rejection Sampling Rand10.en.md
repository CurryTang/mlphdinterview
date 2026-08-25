# Rejection Sampling: Implementing randN with randM

## Interview Objective

The representative problem in this category is LeetCode 470: `Implement Rand10() Using Rand7()`.

Problem statement:

```text
Given that rand7() returns 1..7 with equal probability.
Implement rand10(), which must return 1..10 with equal probability.
```

The core method is rejection sampling.

You should be able to explain three things clearly:

- Multiple calls to `randM()` can be used to construct a larger uniform integer space.
- You cannot directly use `% N` unless the space size is divisible by `N`.
- Keep the largest prefix that is divisible by `N`, and discard the remaining samples and resample.

## Standard Solution: Implement rand10 with rand7

Call `rand7()` twice:

```python
x = (rand7() - 1) * 7 + rand7()
```

This generates `1..49` with equal probability.

Why is it uniform?

The first `rand7()` chooses the row, and the second `rand7()` chooses the column:

```text
7 x 7 = 49 cells
```

The probability of each cell is:

```text
1/7 * 1/7 = 1/49
```

So `x` is uniform over `1..49`.

But `49` is not divisible by `10`. If we directly do:

```python
return x % 10
```

some results will appear 5 times and some will appear 4 times, so the probabilities are not uniform.

Therefore, only keep `1..40`:

```text
40 is divisible by 10
Within 1..40, each rand10 result appears exactly 4 times
Discard 41..49 and resample
```

Code:

```python
class Solution:
    def rand10(self):
        while True:
            x = (rand7() - 1) * 7 + rand7()  # uniform 1..49
            if x <= 40:
                return (x - 1) % 10 + 1
```

Note that the return value is written as:

```python
(x - 1) % 10 + 1
```

This makes the result `1..10`, not `0..9`.

## Why You Cannot Take Modulo Directly

Suppose we directly map `1..49` to `1..10`.

```text
1, 11, 21, 31, 41 -> 1
2, 12, 22, 32, 42 -> 2
...
9, 19, 29, 39, 49 -> 9
10, 20, 30, 40    -> 10
```

The first 9 results each appear 5 times, while result `10` appears only 4 times.

The probabilities are:

```text
P(1..9) = 5/49
P(10)  = 4/49
```

This is not uniform.

The essence of rejection sampling is: only take results from the region where probability can be distributed evenly.

## General Template: Implement randN with randM

If we have:

```text
randM() -> uniform 1..M
```

and want to implement:

```text
randN() -> uniform 1..N
```

The pattern is:

1. Use `k` calls to `randM()` to construct a sufficiently large uniform space `1..M^k`.
2. Find `limit = floor(M^k / N) * N`.
3. If the sampled value `x <= limit`, return `(x - 1) % N + 1`.
4. Otherwise reject and resample.

Code template:

```python
def randN():
    while True:
        x = 1
        for _ in range(k):
            x = (x - 1) * M + randM()

        limit = (M ** k // N) * N
        if x <= limit:
            return (x - 1) % N + 1
```

Here `k` should be chosen so that:

```text
M^k >= N
```

But larger is not always better. If `k` is larger, each round calls `randM()` more times; if `k` is too small, the rejection probability may be high or it may not even cover `N` at all.

## How to Choose k

The simplest rule:

```text
Choose the smallest k such that M^k >= N
```

For example, `rand7 -> rand10`:

```text
7^1 = 7  < 10
7^2 = 49 >= 10
```

So use `rand7()` twice.

Rejection probability:

```text
usable = floor(49 / 10) * 10 = 40
reject = 9 / 49
accept = 40 / 49
```

The average number of rounds needed per success is:

```text
1 / accept = 49 / 40 rounds
```

Each round calls `rand7()` twice, so the expected number of calls is:

```text
2 * 49 / 40 = 2.45
```

## Variation 1: Optimize rand10 by Reusing Rejected Randomness

The standard solution discards `41..49`, but those 9 numbers are themselves still uniform. They can be remapped to `1..9`, then multiplied by a new `rand7()` to construct `1..63`.

Code:

```python
class Solution:
    def rand10(self):
        while True:
            x = (rand7() - 1) * 7 + rand7()  # 1..49
            if x <= 40:
                return (x - 1) % 10 + 1

            x = (x - 40 - 1) * 7 + rand7()  # 1..63
            if x <= 60:
                return (x - 1) % 10 + 1

            x = (x - 60 - 1) * 7 + rand7()  # 1..21
            if x <= 20:
                return (x - 1) % 10 + 1
```

Why is this still correct?

- `41..49` contains 9 equally likely outcomes.
- After mapping them to `1..9`, they are still uniform.
- Multiplying by `rand7()` again gives `9 * 7 = 63` equally likely outcomes.
- `60` is divisible by `10`, so we can take `1..60`.

This version reduces waste, but the code is more complex. In an interview, write the standard solution first, then mention this optimization.

## Variation 2: Implement rand7 with rand5

`5^1 = 5 < 7`, so at least two calls are needed:

```text
5^2 = 25
limit = floor(25 / 7) * 7 = 21
```

Code:

```python
def rand7():
    while True:
        x = (rand5() - 1) * 5 + rand5()  # 1..25
        if x <= 21:
            return (x - 1) % 7 + 1
```

## Variation 3: Implement rand3 with rand2

Two calls to `rand2()` can generate `1..4`:

```text
2^2 = 4
limit = floor(4 / 3) * 3 = 3
```

Code:

```python
def rand3():
    while True:
        x = (rand2() - 1) * 2 + rand2()  # 1..4
        if x <= 3:
            return x
```

Here, when `x <= 3`, return `x` directly because the target is exactly `1..3`.

## Variation 4: Implement rand7 with rand10

If `M >= N`, one call may already be enough.

`rand10()` generates `1..10`, and we want to implement `rand7()`:

```text
limit = floor(10 / 7) * 7 = 7
```

Code:

```python
def rand7():
    while True:
        x = rand10()
        if x <= 7:
            return x
```

This is more direct than `rand7 -> rand10`, because the original space is already larger than the target space.

## Variation 5: Use randM to Implement a Specific Range [a, b]

If you want to generate `[a, b]`:

```text
N = b - a + 1
```

First implement `randN()`, then shift:

```python
return randN() + a - 1
```

For example, if you want to generate `5..14`, generate `1..10` first and then add `4`.

## Variation 6: What If the Probability Is Not Uniform?

The premise of this type of problem is usually:

```text
randM() itself is uniform
```

If the input is a biased coin, you cannot directly apply the formula above. You first need to use the Von Neumann trick to construct fair random bits:

```text
Flip the biased coin twice in a row:
HT -> 0
TH -> 1
HH / TT -> discard and retry
```

Because:

```text
P(HT) = p(1-p)
P(TH) = (1-p)p
```

the two probabilities are equal.

This is another common variation: use a biased coin to implement a fair coin.

## Correctness Proof Template

In an interview, you can prove it like this:

1. `k` calls to `randM()` generate `M^k` combinations, and each combination has the same probability, so `x` is uniform over `1..M^k`.
2. Take `limit = floor(M^k / N) * N`, so `1..limit` can be evenly divided into `N` groups.
3. For `x <= limit`, use `(x - 1) % N + 1` for the mapping, and each result appears `limit / N` times.
4. Reject and resample for `x > limit`; this does not bias any result, it only increases running time.

Therefore, the output is uniformly distributed over `1..N`.

## Complexity

For the standard `rand7 -> rand10` solution:

```text
accept probability = 40 / 49
expected rounds = 49 / 40
expected rand7 calls = 2 * 49 / 40 = 2.45
```

In the general case:

```text
space size = M^k
limit = floor(M^k / N) * N
accept probability = limit / M^k
expected rounds = M^k / limit
expected randM calls = k * M^k / limit
```

The space complexity is `O(1)`.

## Common Pitfalls

- Directly using `% N` without checking whether the original space is divisible by `N`.
- Forgetting that `randM()` returns `1..M`, so failing to subtract 1 first when constructing the space.
- Returning `(x % N) + 1`, which causes a boundary distribution error; it is better to consistently write `(x - 1) % N + 1`.
- Sampling only once and not handling the rejected region.
- Thinking rejection sampling can loop forever; as long as the acceptance probability is greater than 0, the expected running time is finite.
- When reusing the leftover space in the optimized version, failing to verify that the leftover space is still uniform.

## Interview Answer Template

<details class="solution">
<summary>Expand Template</summary>

You cannot directly take modulo on the result of `rand7()` in this problem, because `7` or `49` may not be divisible by the target range. Taking modulo would make some results appear more often.

I would first call `rand7()` twice to construct a uniform `1..49` space:

```python
x = (rand7() - 1) * 7 + rand7()
```

Then I only accept `1..40`, because `40` is divisible by `10`. For accepted values, I use:

```python
(x - 1) % 10 + 1
```

to map into `1..10`. I reject `41..49` and resample. This way, each output value appears exactly 4 times within `1..40`, so the result is uniform.

</details>

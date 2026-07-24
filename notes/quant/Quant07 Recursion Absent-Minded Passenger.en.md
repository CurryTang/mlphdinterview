# Quant 7 · Recurrence: The Absent-Minded Passenger

Recurrence is well suited to problems in which a random choice leaves behind a smaller problem with the same structure. The key is usually not to enumerate the entire process, but to:

1. define the target quantity for a problem of size $n$;
2. classify the possible first-step outcomes;
3. identify the isomorphic subproblem left after each outcome;
4. write a recurrence and solve it using the boundary conditions.

This note demonstrates the method with the classic absent-minded passenger problem.

---

## Problem

An airplane has $n$ seats, numbered $1$ through $n$, and $n$ passengers board in order.

Each passenger is supposed to sit in the seat with the same number as their own. If a passenger finds that seat occupied, they choose uniformly at random from all currently empty seats.

Passenger 1 is absent-minded and initially chooses a seat uniformly at random from seats $1$ through $n$. All other passengers follow the rule above.

What is the probability that passenger $n$ ultimately sits in their own seat?

For $n\ge 2$, the answer is:

$$
\boxed{\frac12}
$$

For the degenerate case $n=1$, the only passenger must sit in the only seat, so the probability is $1$.

---

## 1. Define the recurrence state

Let:

$$
p_n=P(\text{with }n\text{ passengers, passenger }n\text{ sits in seat }n).
$$

We only need to consider which seat passenger 1 initially chooses.

When $n=2$, passenger 1 chooses seat 1 or seat 2 with equal probability:

- if passenger 1 chooses seat 1, passenger 2 succeeds;
- if passenger 1 chooses seat 2, passenger 2 fails.

Therefore, the boundary condition is:

$$
p_2=\frac12.
$$

---

## 2. Classify the first choice

Passenger 1 chooses uniformly from $n$ seats, so each seat is selected with probability $1/n$.

### Case A: Seat 1 is selected

This occurs with probability $1/n$. Every later passenger can then sit in their own seat, so passenger $n$ certainly succeeds.

This case contributes:

$$
\frac1n\cdot 1.
$$

### Case B: Seat $n$ is selected

This occurs with probability $1/n$. Passenger $n$'s seat is already occupied, so passenger $n$ certainly fails.

This case contributes:

$$
\frac1n\cdot 0.
$$

### Case C: An intermediate seat $k$ is selected

Here $2\le k\le n-1$.

Passengers $2,3,\ldots,k-1$ all sit normally. When passenger $k$ boards, they find that passenger 1 has occupied their seat, so they must choose uniformly from the remaining empty seats.

The empty seats that can affect the subsequent conflict are:

$$
1,k+1,k+2,\ldots,n.
$$

The passengers who may still be forced to choose randomly are:

$$
k,k+1,k+2,\ldots,n.
$$

Treat passenger $k$ as the "first absent-minded passenger" in a new problem and renumber these relevant seats. The remaining process is identical to the original problem of size:

$$
n-k+1
$$

Therefore, conditional on passenger 1 choosing seat $k$, the probability that the final passenger succeeds is:

$$
p_{n-k+1}.
$$

---

## 3. Write the recurrence

By the law of total probability:

$$
p_n
=\frac1n
+\frac1n\sum_{k=2}^{n-1}p_{n-k+1}.
$$

Let $m=n-k+1$. As $k$ ranges from $2$ to $n-1$, $m$ runs through $n-1,n-2,\ldots,2$. Therefore:

$$
\boxed{
p_n=\frac{1+\sum_{m=2}^{n-1}p_m}{n}
}.
$$

Equivalently:

$$
np_n=1+\sum_{m=2}^{n-1}p_m.
$$

For $n-1$, we also have:

$$
(n-1)p_{n-1}=1+\sum_{m=2}^{n-2}p_m.
$$

Subtracting the two equations:

$$
np_n-(n-1)p_{n-1}=p_{n-1}.
$$

Rearranging gives:

$$
np_n=np_{n-1},
$$

and hence:

$$
p_n=p_{n-1}.
$$

Together with $p_2=1/2$, induction gives, for every $n\ge2$:

$$
\boxed{p_n=\frac12}.
$$

---

## 4. Why the subproblem is truly isomorphic

Suppose passenger 1 sits in seat $k$. After that:

```text
Passengers 2 through k-1: their seats are free, so they all sit normally.
Passenger k: their seat is occupied, so they become the next person to choose randomly.
Seats 2 through k-1: they are now correctly occupied and no longer affect what follows.
```

After removing these passengers and seats whose outcomes are fixed, the remaining rules are still:

```text
The first active passenger chooses uniformly from the remaining seats.
Every other active passenger first tries to sit in their own seat.
If their own seat is occupied, they choose uniformly from the empty seats.
```

The earlier history can therefore be compressed away. The subsequent success probability depends only on the size of the remaining problem, not on how the conflict propagated to this point. This is **state compression** in a recurrence.

---

## 5. A faster symmetry argument

Only two special seats determine the final outcome of the conflict chain:

```text
Seat 1: the conflict chain ends here, and passenger n succeeds.
Seat n: the conflict chain ends here, and passenger n fails.
```

If a passenger choosing randomly selects an intermediate seat, the conflict merely transfers to another passenger and does not immediately determine the result. The outcome is fixed only when the conflict chain first reaches seat 1 or seat $n$.

Whenever a random choice is made while seats 1 and $n$ are both empty, those two seats are completely symmetric. Therefore, each is equally likely to be selected first:

$$
P(\text{seat }1\text{ is selected first})
=P(\text{seat }n\text{ is selected first})
=\frac12.
$$

Thus, passenger $n$ sits in their own seat with probability $1/2$.

This argument is shorter, but the recurrence is easier to generalize to variants that are no longer fully symmetric.

---

## 6. Common mistakes

### Mistake 1: The answer is $1/n$

$1/n$ is only the probability that passenger 1 directly chooses a particular seat. If passenger 1 chooses an intermediate seat, the conflict continues to propagate, and passenger $n$ may still succeed.

### Mistake 2: Assuming every later passenger chooses randomly

Only a passenger who finds their own seat occupied chooses randomly. Most passengers still sit directly in their assigned seats.

### Mistake 3: Using the wrong subproblem size

If passenger 1 occupies seat $k$, the relevant passengers are $k,k+1,\ldots,n$, whose number is:

$$
n-k+1,
$$

so the corresponding term is $p_{n-k+1}$, not $p_{n-k}$.

### Mistake 4: Simply claiming that the result is "obviously symmetric"

The symmetry that needs to be explained is between the two special empty seats, 1 and $n$, at which the conflict chain can terminate. It is not a symmetry of every passenger or every seat in the final configuration.

---

## 7. General recurrence template

For a random-process problem, organize the solution in this order:

```text
1. Define p_n: the probability of the target event for a problem of size n.
2. Find a boundary condition: solve the smallest size directly, such as p_2.
3. Examine the first step: list all mutually exclusive cases and their probabilities.
4. Compress the state: determine whether the first step leaves a smaller isomorphic problem.
5. Write the recurrence using the law of total probability.
6. Solve the recurrence by subtracting adjacent equations, induction, or generating functions.
7. Check small cases and degenerate boundaries.
```

The structure worth remembering from this problem is:

$$
\text{randomly select an intermediate state}
\quad\Longrightarrow\quad
\text{remove the fixed part and obtain a smaller isomorphic problem}.
$$

---

```quiz
title: Recurrence Quick Check
question: If passenger 1 initially sits in seat k, where 2 ≤ k ≤ n-1, what is the size of the isomorphic problem that follows?
answer: C
A. k-1
B. n-k
C. n-k+1
D. n-1
explanation: The relevant passengers are k,k+1,...,n, for a total of n-k+1 passengers; passenger k becomes the new random seat chooser.
```

## One-sentence summary

An intermediate seat merely passes the role of the "absent-minded passenger" farther back. The conflict chain is equally likely to reach seat 1 or seat $n$ first, so the final passenger sits in their own seat with probability $1/2$.

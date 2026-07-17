# Quant 1 · 期望与计数：Indicator、Records 与 Multinomial

这篇放在 Quant 概率部分的开头。它复习三件经常一起出现的工具：

```text
1. 把计数变量拆成 indicator，再用线性期望。
2. 在随机顺序里识别 record high / record low。
3. 遇到多项分布计数乘积，用 falling factorial moment。
```

前半篇从骰子计数讲 multinomial moment，后半篇用“单行徒步队伍”讲 record minimum。两题表面不同，落笔时都先把目标数量拆成一串 0/1 事件。

---

## 1. 题目

一枚公平的六面骰子被掷 10 次。对每个点数 $i \in \{1,2,\ldots,6\}$，令 $N_i$ 表示点数 $i$ 出现的次数。

求：

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
$$

---

## 2. 核心理解

这题的随机变量不是单个 $N_i$，而是六个计数的乘积。因为 $N_1,\ldots,N_6$ 来自同一组 10 次投掷，它们不是独立的：

```text
某个点数出现得多，
其他点数能出现的总次数就会变少。
```

所以不能写成：

$$
\mathbb{E}[N_1]\mathbb{E}[N_2]\cdots\mathbb{E}[N_6]
$$

正确工具是 multinomial distribution 的 factorial moment，或者等价地用 indicator expansion 数组合。

---

## 3. 方法一：Multinomial Falling Factorial Moment

10 次公平骰子投掷后：

$$
(N_1,\ldots,N_6) \sim \text{Multinomial}\left(10;\frac16,\ldots,\frac16\right)
$$

Multinomial 有一个很重要的公式：

$$
\mathbb{E}\left[\prod_{j=1}^k (N_j)_{a_j}\right]
=
(n)_{a_1+\cdots+a_k}\prod_{j=1}^k p_j^{a_j}
$$

其中：

$$
(x)_a = x(x-1)\cdots(x-a+1)
$$

叫 falling factorial。

这题里每个点数只取一次，所以 $a_1=\cdots=a_6=1$。因为 $(N_i)_1=N_i$，所以：

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
=
(10)_6\left(\frac16\right)^6
$$

展开：

$$
(10)_6 = 10\cdot 9\cdot 8\cdot 7\cdot 6\cdot 5
$$

因此答案是：

$$
\boxed{\frac{10\cdot9\cdot8\cdot7\cdot6\cdot5}{6^6}}
$$

也可以写成：

$$
\boxed{\frac{151200}{46656}=\frac{175}{54}}
$$

---

## 4. 方法二：Indicator Expansion

这个方法更适合面试时解释为什么公式成立。

把 $N_i$ 写成 indicator 之和。设第 $t$ 次投掷是否为点数 $i$：

$$
X_{t,i} =
\mathbf{1}\{\text{第 }t\text{ 次掷出 }i\}
$$

那么：

$$
N_i = \sum_{t=1}^{10} X_{t,i}
$$

所以：

$$
N_1N_2N_3N_4N_5N_6
=
\sum_{t_1,\ldots,t_6}
X_{t_1,1}X_{t_2,2}X_{t_3,3}X_{t_4,4}X_{t_5,5}X_{t_6,6}
$$

关键点：如果两个 $t$ 相同，比如 $t_1=t_2$，那么同一次投掷不可能同时是点数 1 和点数 2，所以这一项一定是 0。

只有当：

```text
t_1, t_2, t_3, t_4, t_5, t_6
全部互不相同
```

这一项才可能非零。

有多少种互不相同的有序选择？

$$
10\cdot9\cdot8\cdot7\cdot6\cdot5 = (10)_6
$$

对每一种选择，对应 6 次指定投掷分别掷出 $1,2,3,4,5,6$，概率是：

$$
\left(\frac16\right)^6
$$

所以：

$$
\mathbb{E}[N_1N_2N_3N_4N_5N_6]
=
(10)_6\left(\frac16\right)^6
=
\frac{175}{54}
$$

---

## 5. 为什么不是把期望相乘

每个 $N_i$ 的期望都是：

$$
\mathbb{E}[N_i] = \frac{10}{6}
$$

但不能因此得到：

$$
\left(\frac{10}{6}\right)^6
$$

原因是 $N_i$ 之间有负相关。总次数固定为 10：

$$
N_1+\cdots+N_6=10
$$

如果 $N_1$ 很大，剩下五个计数的总空间就变小。乘积期望需要处理这些依赖关系。

---

## 6. 第一题小结

多项分布里，普通计数乘积经常转成 falling factorial moment：

$$
\mathbb{E}[N_1N_2\cdots N_k]
=
(n)_k p_1p_2\cdots p_k
$$

前提是每个计数只出现一次。直觉上就是：

```text
先选出 k 个互不相同的 trial，
再要求它们分别落到指定的 k 个类别。
```

---

## 7. 第二题：单行徒步队伍

$n$ 名徒步者在一条无限长的狭窄山路上同向前进，起点从前到后编号为 $1,2,\ldots,n$。每个人的速度独立地来自同一个连续分布，山路上不能超越。

如果后方较快的人追上前方较慢的队伍，他会留在队尾，并从此按前方队伍的速度行走。时间足够长后，期望还剩多少支队伍？

连续分布这个条件排除了速度相同的情况。至于速度究竟服从 uniform、exponential、lognormal 还是别的连续速度分布，并不影响答案。

---

## 8. 先把动态过程压成静态条件

设 $V_i$ 是第 $i$ 位徒步者的速度，顺序仍是从前到后。第 $i$ 位最终成为一支队伍的领队，当且仅当：

$$
V_i < \min(V_1,\ldots,V_{i-1})
$$

也就是说，他的速度是到目前为止的新低点，称为 **prefix minimum** 或 **record low**。

- 如果 $V_i$ 比前面所有人都慢，他不可能追上前面的任何队伍，因此会保留一支新队伍。
- 如果前面已经有人更慢，他迟早会追上那个更慢的队伍；中间发生多少次合并都不改变结局。

路程和追赶时间因此退出了问题，只剩速度的相对排名。

```record-minimum-demo
```

图中的速度只是一次具体样本。它可能产生 4 支队伍，但题目问的是对所有随机排列取平均。

---

## 9. Indicator + 对称性

令：

$$
I_i
=
\mathbf{1}\{V_i \text{ 是 }V_1,\ldots,V_i\text{ 中的最小值}\}
$$

最终队伍数为：

$$
K_n=\sum_{i=1}^n I_i
$$

观察前 $i$ 个速度。因为它们 i.i.d. 且没有 ties，最小值落在 $i$ 个位置中的概率完全相同，所以：

$$
\mathbb{P}(I_i=1)=\frac{1}{i}
$$

使用线性期望：

$$
\mathbb{E}[K_n]
=
\sum_{i=1}^n \mathbb{E}[I_i]
=
\sum_{i=1}^n \frac1i
=
\boxed{H_n}
$$

其中 $H_n$ 是第 $n$ 个 harmonic number。规模很大时：

$$
H_n
=
\log n+\gamma+\frac{1}{2n}+O(n^{-2})
$$

所以人数增加十倍，队伍数不会增加十倍，只会增加大约 $\log 10$。

---

## 10. 面试里容易走偏的地方

### 只比较相邻两个人

写成 $\mathbb{P}(V_i<V_{i-1})=1/2$ 会得到约 $n/2$ 支队伍，但领队必须慢过前面的**所有人**，不是只慢过紧邻的一位。

### 试图模拟每次追赶

初始距离会改变谁先追上谁，却不改变无限时间后的领队集合。先寻找终态的静态刻画，通常比给每次碰撞写递推简单。

### 先研究分布的密度

这题只使用相对排名。i.i.d.、连续性和交换对称性已经给出 $1/i$，不需要积分。

### 忘记模型边界

如果道路有限、只观察有限时间、允许超越，或者速度有大量 ties，prefix minimum 的结论就需要重新检查。

---

## 11. 经常一起出现的面试题

| 题型 | 隐藏的 indicator | 结论或联系 |
|---|---|---|
| 从左侧能看见多少栋随机高度的楼 | 第 $i$ 栋高于前面所有楼 | 期望为 $H_n$，把 record low 换成 record high |
| 每遇到历史最佳候选人就更换员工 | 第 $i$ 位是 best-so-far | 期望更换 $H_n$ 次 |
| 扫描随机数组时，running max 更新几次 | 第 $i$ 项刷新前缀最大值 | 期望为 $H_n$ |
| 随机排列有多少个 cycle | 每个 cycle 贡献一个代表元素 | 期望也是 $H_n$；cycle 数与 record 数甚至同分布 |
| 随机 BST 中某个节点的深度 | 哪些区间元素会成为它的 ancestor | 拆开后出现两段 harmonic sum |
| Secretary problem | 只有 record candidate 值得录用 | 仍用 record，但目标变成最大化选中全局最优的概率 |

Coupon collector 也会出现 $H_n$，但原因不同：它来自“尚未收集的类别越来越少”，不是前缀纪录。看到 harmonic number 时，最好说明分母 $1/i$ 究竟从哪里来。

---

## 12. 可继续追问

### 恰好剩 $k$ 支队伍的概率

Record 数与随机排列的 cycle 数同分布，因此：

$$
\mathbb{P}(K_n=k)
=
\frac{\left[{n\atop k}\right]}{n!}
$$

$\left[{n\atop k}\right]$ 是第一类无符号 Stirling number。

### 方差

连续 i.i.d. 样本的 record indicators 相互独立，因此：

$$
\operatorname{Var}(K_n)
=
\sum_{i=1}^n \frac1i\left(1-\frac1i\right)
=
H_n-H_n^{(2)}
$$

其中 $H_n^{(2)}=\sum_{i=1}^n 1/i^2$。

---

## 13. 一分钟答题顺序

```text
把人按前后顺序编号
→ 最终领队等价于 prefix minimum
→ K_n = Σ I_i
→ 第 i 个位置成为前缀最小值的概率是 1/i
→ E[K_n] = H_n ≈ log n + γ
```

这题不需要追逐过程的递推。最难的一步只是认出：看起来在不断合并，实际是在数随机序列里的 records。

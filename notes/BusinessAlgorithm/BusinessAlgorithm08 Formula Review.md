# 业务算法公式速查

## 公式速查

这页只收高频公式。复习时别只背符号：至少要能说出它在哪一层使用、分母为零怎么办、线上如何近似。

### 召回与相似度

余弦相似度：

```math
\cos(a,b)=\frac{a^\top b}{\|a\|_2\|b\|_2}.
```

ItemCF：

```math
\operatorname{sim}(i,j)
=\frac{|U_i\cap U_j|}
{\sqrt{|U_i||U_j|}}.
```

矩阵分解：

```math
\hat r_{ui}=u_u^\top v_i.
```

双塔对比损失：

```math
\mathcal L
=-\log
\frac{e^{s(q,i^+)/\tau}}
{\sum_{j\in\{i^+\}\cup\mathcal N_q}e^{s(q,j)/\tau}}.
```

BM25：

```math
\sum_{t\in q}\operatorname{IDF}(t)
\frac{tf(t,d)(k_1+1)}
{tf(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
```

### 排序

LogLoss：

```math
-y\log p-(1-y)\log(1-p).
```

Pairwise logistic loss：

```math
-\log\sigma(s^+-s^-).
```

DCG：

```math
\operatorname{DCG@K}
=\sum_{i=1}^{K}
\frac{2^{rel_i}-1}{\log_2(i+1)}.
```

MMoE：

```math
h_t(x)=\sum_e g_{t,e}(x)f_e(x).
```

FM：

```math
\hat y
=w_0+\sum_iw_ix_i
+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j.
```

DIN：

```math
u(q)=\sum_j\alpha(h_j,q)h_j.
```

### 列表与生成

MMR：

```math
\arg\max_i
\left[\theta r_i-(1-\theta)
\max_{j\in S}\operatorname{sim}(i,j)\right].
```

DPP：

```math
P(S)\propto\det(L_S).
```

生成式检索：

```math
P(d\mid q)=\prod_tP(d_t\mid d_{<t},q).
```

列表生成：

```math
P(i_1,\ldots,i_m\mid H)
=\prod_tP(i_t\mid i_{<t},H).
```

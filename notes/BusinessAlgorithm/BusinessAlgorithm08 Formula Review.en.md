# Business Algorithm Formula Quick Reference

## Formula Quick Reference

This page contains only high-frequency formulas. When reviewing, do not just memorize the symbols: you should at least be able to explain which layer they are used in, how to handle division by zero, and how to approximate them in production.

### Recall and Similarity

Cosine Similarity:

```math
\cos(a,b)=\frac{a^\top b}{\|a\|_2\|b\|_2}.
```

ItemCF:

```math
\operatorname{sim}(i,j)
=\frac{|U_i\cap U_j|}
{\sqrt{|U_i||U_j|}}.
```

Matrix Factorization:

```math
\hat r_{ui}=u_u^\top v_i.
```

Two-Tower Contrastive Loss:

```math
\mathcal L
=-\log
\frac{e^{s(q,i^+)/\tau}}
{\sum_{j\in\{i^+\}\cup\mathcal N_q}e^{s(q,j)/\tau}}.
```

BM25:

```math
\sum_{t\in q}\operatorname{IDF}(t)
\frac{tf(t,d)(k_1+1)}
{tf(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
```

### Ranking

LogLoss:

```math
-y\log p-(1-y)\log(1-p).
```

Pairwise logistic loss:

```math
-\log\sigma(s^+-s^-).
```

DCG:

```math
\operatorname{DCG@K}
=\sum_{i=1}^{K}
\frac{2^{rel_i}-1}{\log_2(i+1)}.
```

MMoE:

```math
h_t(x)=\sum_e g_{t,e}(x)f_e(x).
```

FM:

```math
\hat y
=w_0+\sum_iw_ix_i
+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j.
```

DIN:

```math
u(q)=\sum_j\alpha(h_j,q)h_j.
```

### List and Generation

MMR:

```math
\arg\max_i
\left[\theta r_i-(1-\theta)
\max_{j\in S}\operatorname{sim}(i,j)\right].
```

DPP:

```math
P(S)\propto\det(L_S).
```

Generative Retrieval:

```math
P(d\mid q)=\prod_tP(d_t\mid d_{<t},q).
```

List Generation:

```math
P(i_1,\ldots,i_m\mid H)
=\prod_tP(i_t\mid i_{<t},H).
```

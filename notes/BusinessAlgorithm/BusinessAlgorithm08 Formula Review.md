# 公式速查与两周复习表

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

---

## 两周复习安排

### 第一周：传统链路

| 天 | 内容 | 当天产出 |
| --- | --- | --- |
| 1 | 第 1-2 章 | 照总图讲完一次请求，并说明日志怎样变成样本 |
| 2 | 第 3 章 | 手写 BM25、ItemCF、矩阵分解，讲清离线索引 |
| 3 | 第 4-5 章 | 解释负样本、ANN、多路召回和 query 改写 |
| 4 | 第 6 章 | 对比 pointwise、pairwise、listwise，手算 NDCG |
| 5 | 第 7-8 章 | 画 MMoE、FM/DCN 和粗排位置 |
| 6 | 第 9-11 章 | 串起 DIN/SIM、MMR/DPP、冷启动与探索 |
| 7 | 系统复盘 | 闭卷画短视频推荐和电商搜索，再补离线验收与 A/B |

### 第二周：生成式与系统题

| 天 | 内容 | 当天产出 |
| --- | --- | --- |
| 8 | 第 12 章 | 对比 DSI、NCI、SEAL、TIGER，手写残差量化 |
| 9 | 第 13 章前半 | 对比 pointwise、pairwise、listwise LLM reranking |
| 10 | HSTU 与 OneRec | 说明它们分别替换了传统链路的哪一段 |
| 11 | 第 14 章 | 画标准 RAG、Self-RAG 和 search agent |
| 12 | 生成式搜索案例 | 补齐引用、评价、安全和成本 |
| 13 | 生成式串讲 | 闭卷连接检索、推荐、RL 和 agentic search |
| 14 | 完整模拟 | 先定业务与约束，沿数据流设计系统，模型放在最后定 |

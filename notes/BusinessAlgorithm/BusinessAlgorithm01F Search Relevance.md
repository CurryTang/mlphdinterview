# 搜索相关性与 BERT

## 第 8 章 搜索相关性与 BERT

相关性只回答一个问题：文档是否满足 query 的需求。内容质量、时效、个性化和商业目标会影响最终排序，但不应偷偷混进相关性标签。

### 8.1 相关性分档

常见分档可以写成 0–3：

| 档位 | 含义 |
| --- | --- |
| 3 | 直接满足主要意图，信息完整 |
| 2 | 相关且有帮助，但不够完整 |
| 1 | 只覆盖次要意图或局部信息 |
| 0 | 语义不相关，或没有回答问题 |

多义 query 要在标注规范里列出主要意图。`"苹果"` 的文档可以命中水果或品牌；是否算高相关取决于意图占比和产品场景，不能由标注者临场猜。

### 8.2 三种离线评价口径

Pointwise 指标看单个 `(q,d)` 的预测是否接近标签，可用 MSE、LogLoss、AUC 或分档准确率。

Pairwise 指标看同一 query 下文档对的顺序。若 `y_i > y_j`，希望 `s_i > s_j`。正逆序比可写为：

```math
\frac{\#\{(i,j):y_i>y_j,\ s_i>s_j\}}
{\#\{(i,j):y_i>y_j\}}.
```

Listwise 指标看整个结果列表，常用 DCG/NDCG。三类指标分别关心保值、保序和列表头部，没有一个指标能替代另外两个。

### 8.3 文本匹配分数

倒排索引阶段仍依赖传统文本信号：

- TF-IDF；
- BM25；
- query term 覆盖率；
- 标题、正文、anchor 等字段匹配；
- term 顺序和距离。

BM25 对 term frequency 做饱和，并修正文档长度：

```math
\operatorname{BM25}(q,d)
=\sum_{t\in q}
\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}
{f(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
```

词袋模型不看上下文。`"阿迪跑步鞋"` 与包含 `"阿迪王跑步鞋"` 的文档有很强字面匹配，语义却可能不满足品牌意图。Term proximity 能补充词距，仍解决不了所有语义问题。

传统分数没有过时。它们便宜、可解释，适合召回和轻量粗排，也可作为 BERT 排序模型的输入特征。

### 8.4 Cross-BERT 与双塔 BERT

Cross-BERT 联合编码：

```text
[CLS] query [SEP] title [SEP] body [SEP] -> relevance
```

Query token 可以直接与文档 token 交互，准确率高，但每个 `(q,d)` 都要推理。

双塔分别编码：

```math
z_q=f(q),\qquad z_d=g(d),\qquad s(q,d)=z_q^\top z_d.
```

文档向量可离线计算。双塔适合向量召回或粗排，Cross-BERT 适合候选更少的后级。实际链路常按成本逐层加深交互，而不是二选一。

### 8.5 Token 粒度和长文档

中文可以使用字粒度，也可以使用字词混合粒度。混合粒度序列更短，能减少 attention 成本，并在固定 token 上限内保留更多文档信息。

长文档不能直接粗暴截断。可组合：

- 标题和首段优先；
- query-aware passage selection；
- 离线抽取式摘要；
- 分段打分后聚合；
- Anchor Query 作为文档的短标签。

Anchor Query 是从文档反向生成并经过相关性过滤的 query。它对双塔尤其有用，因为短 query 风格文本能缩小 query 与长文档表示之间的结构差异。

### 8.6 推理降本

Cross-BERT 常见手段：

- 缓存 `(query_id, document_id, model_version) -> score`；
- INT8 PTQ 或 QAT；
- teacher-student 蒸馏；
- 减少 token 上限；
- passage 预筛；
- batching 和动态 padding。

缓存 key 必须包含模型、分词和文档版本。否则更新模型后仍在读旧分数，灰度结果会被缓存污染。

双塔的成本主要在 query encoder、文档向量存储与 ANN。量化和索引参数会损失 recall，验收时要把模型损失与 ANN 近似损失分开。

### 8.7 四阶段训练

课程把相关性 BERT 训练拆成四步：

```text
预训练 -> 后预训练 -> 人工标注微调 -> 蒸馏
```

预训练使用 MLM 等任务学习通用语言表示。

后预训练利用大规模搜索日志。先从日志中得到 `(q,d,x)`，其中 `x` 是点击和交互统计；再用一小批人工相关性数据训练映射 `t(x)`，生成弱标签：

```math
\tilde y=t(x).
```

然后用海量 `(q,d,\tilde y)` 继续训练 BERT，同时保留 MLM，减轻语言能力遗忘。

这里有一条重要禁区：`t(x)` 不能把旧相关性模型分数当输入。否则旧模型决定曝光和分数，新模型又学习这个分数，反馈回路会不断复制旧模型偏差。

### 8.8 微调：同时保值和保序

人工标注数据量较小但标签干净。回归或 soft-label CE 让预测值接近标注：

```math
\mathcal L_{\text{CE}}
=-y\log p-(1-y)\log(1-p).
```

Pairwise logistic 让高相关文档排在低相关文档前：

```math
\mathcal L_{\text{pair}}
=\log\left(1+\exp[-\gamma(s_i-s_j)]\right),
\qquad y_i>y_j.
```

工业训练常把 pointwise、pairwise 和 MLM 目标加权。只做多分类会把相关性档位当成彼此无序的类别，不符合 `"3 比 2 更相关"` 的结构。

### 8.9 蒸馏

大 teacher 提供更准确的 soft label，小 student 满足线上延迟预算。典型流程：

1. 完整训练 teacher；
2. 用 teacher 给大规模 `(q,d)` 打分；
3. 先用正常流程预热 student；
4. 让 student 拟合 teacher 分数和顺序；
5. 在人工标注集上确认没有只学会 teacher 的偏差。

逐层拟合 teacher hidden states 并非必需。最终目标是相关性分数与排序，投入同样算力时，直接增加高质量蒸馏 pair 往往更实用。

### 8.10 本章自测

1. 为什么相关性标签不应混入内容质量？
2. Pointwise、pairwise 和 listwise 指标各自看什么？
3. BM25 在 BERT 时代为什么仍有用？
4. Cross-BERT 的缓存 key 为什么需要模型版本？
5. 后预训练怎样从用户行为得到弱相关性标签？
6. 为什么不能用旧相关性模型分数训练弱标签映射？
7. 蒸馏前为什么要先预热 student？

<details>
<summary>参考答案</summary>

1. 两者含义和更新方式不同。分开后才能判断文档是"相关但低质"还是"高质但不相关"，并按场景融合。
2. Pointwise 看单样本分值，pairwise 看文档对顺序，listwise 看整个列表尤其是头部位置。
3. 它计算便宜、可解释，适合倒排召回和粗筛，还能作为神经模型特征；神经语义模型没有消除大规模候选生成的成本。
4. 模型、分词或文档变化后，同一 `(q,d)` 的正确分数可能改变。不带版本会读到陈旧缓存。
5. 用少量人工样本学习点击/交互统计到相关性标签的映射，再给海量日志样本生成带噪声的弱标签。
6. 旧模型已经影响曝光，再把它的分数当监督信号会形成自我复制的反馈回路，压制旧模型没发现的相关文档。
7. 预热让 student 先具备语言和相关性基础，再学习 teacher 的细粒度分布；从随机或仅通用预训练状态直接蒸馏更难稳定收敛。

</details>

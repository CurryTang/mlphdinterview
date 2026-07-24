# 双塔、负样本与向量检索

## 第 4 章 双塔、负样本与向量检索

### 4.1 双塔把匹配变成最近邻

双塔分别编码 query/user 和 document/item：

```math
z_q=f_\theta(x_q), \qquad z_i=g_\phi(x_i),
```

```math
s(q,i)
=\frac{z_q^\top z_i}{\|z_q\|_2\|z_i\|_2}.
```

推荐侧 `x_q` 是用户、历史与上下文，`x_i` 是物品特征。搜索侧 `x_q` 是 query，`x_i` 是文档。

两边独立编码的最大好处是 item/document 向量可离线计算。线上只算 query 向量，然后做 ANN。代价是 query 与候选无法在编码阶段进行细粒度 token/feature 交互。

### 4.2 训练目标

一组正样本和负样本上常用 softmax 对比目标：

```math
\mathcal L
=-\log
\frac{\exp(s(q,i^+)/\tau)}
{\exp(s(q,i^+)/\tau)+
\sum_{j\in\mathcal N_q}\exp(s(q,j)/\tau)}.
```

温度 `τ` 控制分布尖锐程度。负样本集合 `N_q` 往往比网络结构更决定效果。

### 4.3 负样本不是随便采

随机负样本容易到模型一眼就能区分，训练很快收敛，线上却分不清真正相似的候选。

常见来源：

- 全库随机负样本：便宜，通常太简单；
- 批内负样本：其他样本的正例当当前 query 的负例，吞吐高；
- 曝光未点击：接近线上分布，但有位置偏差；
- hard negative：由旧模型或 BM25 召回、语义相近但不相关的候选；
- 混合负样本：兼顾覆盖、难度和稳定性。

批内负样本有 false negative。两个用户可能都喜欢同一物品，或者一个 query 的负例其实也相关。可通过去重、同 query mask、流行度修正和软标签缓解。

采样还改变了先验分布。若热门物品更容易进入 batch，模型会偏热门；若对热门项做过度降采样，线上概率又可能失真。训练目标、采样分布和线上打分要一起设计。

### 4.4 ANN 与向量库

全库做精确 top-k 内积仍然昂贵。近似最近邻用少量召回损失换速度。

常见思路：

- IVF：先把向量分桶，查询只扫最接近的若干桶；
- PQ：把向量分段量化，压缩存储并近似计算距离；
- HNSW：构建多层邻接图，通过图搜索靠近 query。

索引调优要同时看：

- recall 与延迟；
- 内存与量化误差；
- 建库时间与增量更新；
- top-k 大小与后续排序成本。

物品向量更新后，索引是否支持增量写入也很关键。一天全量重建一次可能跟不上新闻、商品库存或短视频热点。

验收 ANN 时要固定同一批 query，画 `Recall@K - P95/P99 延迟 - 内存` 曲线，而不是只报一个 top-K Recall。模型 embedding 不变、索引参数变化时，这条曲线才能说明近似检索本身损失了多少。

### 4.5 线上服务

典型双塔链路：

```text
离线：
item 特征 -> item tower -> item embedding -> ANN index

在线：
用户/query 特征 -> query tower -> query embedding
                               -> ANN top-k
                               -> 过滤与后续排序
```

需要关注：

- query tower 的 P99 延迟；
- embedding 版本与索引版本一致；
- 新物品向量生成和入库延迟；
- 特征缺失的默认值；
- 向量范数、量化和相似度口径；
- 索引故障时的降级通道。

### 4.6 双塔与 Cross-Encoder

Cross-encoder 把 query 和 candidate 拼在一起：

```text
[CLS] query [SEP] document [SEP] -> relevance score
```

它能做细粒度交互，通常更准，但每个 query-candidate 对都要过模型，不能预计算文档侧。最常见的组合是双塔召回、cross-encoder 重排。

这也是后面理解 LLM ranker 的基础。LLM ranker 并没有让计算成本消失，只是把 cross-encoder 的语义能力放大了。

### 4.7 本章自测

1. 双塔为什么适合召回，不适合直接替代所有精排？
2. 批内负样本为什么高效，false negative 从哪里来？
3. hard negative 越难越好吗？
4. HNSW、IVF、PQ 分别在利用什么结构？
5. item embedding 更新后，线上还要同步哪些版本？

<details>
<summary>参考答案</summary>

1. 双塔把 item 表示离线计算后做 ANN，适合大规模候选生成；但 user 与 item 在打分前不做细粒度交互，难以替代 cross-encoder 或复杂精排。
2. 一个 batch 内其他正例可直接充当 negatives，因此不用额外编码。若两个用户都喜欢同一 item，或语义相近 item 被当作负例，就会产生 false negative。
3. 不是。过难样本可能是错标、假负例或业务上不可区分的候选；应选择模型当前能学到、标签又可靠的难例。
4. HNSW 用分层近邻图导航；IVF 先按 coarse centroid 缩小搜索分区；PQ 把向量分块量化，用压缩码近似距离。
5. 需要同步模型、向量、ANN 索引、特征 schema 和 item 可用状态版本，并保证灰度期间 query tower 与索引向量来自兼容版本。

</details>

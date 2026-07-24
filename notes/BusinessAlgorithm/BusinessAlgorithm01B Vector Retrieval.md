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

双塔也可以用 pointwise、pairwise 或 listwise 方式训练。Pointwise 独立判断一个 user-item pair；pairwise 比较一个正例和一个负例；上面的 sampled softmax 属于常见的 listwise 写法，让一个正例同时和多条负例竞争。三种形式都能用，工业召回更常采用批内负样本的 sampled softmax，因为一次编码就能组成大量比较。

### 4.3 负样本不是随便采

随机负样本容易到模型一眼就能区分，训练很快收敛，线上却分不清真正相似的候选。

常见来源：

- 全库随机负样本：便宜，通常太简单；
- 批内负样本：其他样本的正例当当前 query 的负例，吞吐高；
- 曝光未点击：很难，且包含位置偏差和大量假负例；
- hard negative：由旧模型或 BM25 召回、语义相近但不相关的候选；
- 混合负样本：兼顾覆盖、难度和稳定性。

批内负样本有 false negative。两个用户可能都喜欢同一物品，或者一个 query 的负例其实也相关。可通过去重、同 query mask、流行度修正和软标签缓解。

对召回模型，曝光未点击通常不宜作为默认负例。能进入曝光，说明旧召回和排序已经认为它有一定价值；没有点击还可能只是位置、时间或偶然行为。更稳的组合是全库随机/批内负例加上被粗排或精排淘汰的 hard negative，再单独验证曝光未点击是否真的带来收益。

采样还改变了先验分布。若物品 `j` 进入负样本的概率为 `p_j`，批内 softmax 可把 logit 修正为：

```math
s'(q,j)=s(q,j)-\log p_j.
```

热门物品更常进入 batch，不修正会把“被抽得多”误当成“模型应该压得更低”。反过来，对热门项过度降采样也会让线上分布失真。采样概率、损失修正和线上打分需要一起记录。

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

模型更新通常同时走两条节奏。凌晨用前一天完整日志随机打散训练全量模型，发布两座塔和全部物品向量；白天按小时或分钟消费新日志，重点更新用户 ID embedding，让近期兴趣更快进入用户塔。增量数据按时间到达，分布偏、标签也更不成熟，不能长期替代全量训练。线上应分别记录全量 checkpoint、增量 offset 和物品索引版本，任一环节异常都能回退到最近一次完整发布。

### 4.6 双塔与 Cross-Encoder

Cross-encoder 把 query 和 candidate 拼在一起：

```text
[CLS] query [SEP] document [SEP] -> relevance score
```

它能做细粒度交互，通常更准，但每个 query-candidate 对都要过模型，不能预计算文档侧。最常见的组合是双塔召回、cross-encoder 重排。

这也是后面理解 LLM ranker 的基础。LLM ranker 并没有让计算成本消失，只是把 cross-encoder 的语义能力放大了。

### 4.7 离散特征怎样进入模型

用户 ID、物品 ID、类目和城市先通过字典映射为整数，再查 embedding 表。类别很少时 one-hot 尚可，用户或物品达到亿级后只能使用稠密 embedding。

线上故障更多出在映射和版本管理：

- 未登录用户和 OOV 类别使用哪个默认 ID；
- 新物品何时分配 ID、生成向量并写入索引；
- 训练与服务是否加载同一份词典；
- 高频类别是否独占 ID，长尾是否哈希；
- 哈希冲突率和 embedding 表扩容怎样监控。

ID embedding 记住协同信息，内容特征帮助长尾和新品。只用 ID 的模型通常更准地拟合活跃物品，也更容易让冷启动彻底失效。

### 4.8 双塔加自监督学习

头部物品点击多，监督信号足；长尾物品的向量常学不好。自监督训练为同一个物品生成两种特征视图：

- 随机 mask 一部分 field；
- 对多值类目或关键词做 dropout；
- 把特征拆成两组互补视图；
- 按 field 互信息 mask 一组强关联特征。

两种视图的向量应接近，不同物品的向量应分开：

```math
\mathcal L_{\text{ssl}}(i)
=-\log
\frac{\exp(\operatorname{sim}(z_i^{(1)},z_i^{(2)})/\tau)}
{\sum_j\exp(\operatorname{sim}(z_i^{(1)},z_j^{(2)})/\tau)}.
```

最终目标把点击监督和自监督相加：

```math
\mathcal L
=\mathcal L_{\text{click}}
+\alpha\mathcal L_{\text{ssl}}.
```

增强不能破坏物品身份。把品牌和型号同时 mask 后，两个商品可能变得不可区分；自监督损失再强，也只会教模型忽略真正有用的字段。

### 4.9 Deep Retrieval

Deep Retrieval 不把物品只表示成一个向量，而是让物品关联一到多条离散路径，例如 `(2,4,1)`。系统维护双向索引：

```text
item -> paths
path -> items
```

给定用户特征 `x`，模型自回归预估路径：

```math
p(a,b,c\mid x)
=p_1(a\mid x)
\cdot p_2(b\mid a,x)
\cdot p_3(c\mid a,b,x).
```

路径总数随深度指数增长，线上用 beam search 找高概率路径，再通过 `path -> items` 取回候选。这条链路是：

```text
user -> paths -> items
```

训练需要交替学习两类关系：

1. 用户点击某 item 后，提高该 item 所属路径的概率；
2. 根据"喜欢该 item 的用户也喜欢哪些路径"更新 item-path 关联。

还要加入负载正则，避免大量热门 item 挤在少数路径上。与双塔 ANN 相比，它把可检索结构本身也放进训练；代价是路径版本、beam 搜索和双向索引更难维护。

### 4.10 本章自测

1. 双塔为什么适合召回，不适合直接替代所有精排？
2. 批内负样本为什么高效，false negative 从哪里来？
3. hard negative 越难越好吗？
4. HNSW、IVF、PQ 分别在利用什么结构？
5. item embedding 更新后，线上还要同步哪些版本？
6. 为什么长尾物品更需要自监督目标？
7. Deep Retrieval 为什么要限制一条路径上的物品数？

<details>
<summary>参考答案</summary>

1. 双塔把 item 表示离线计算后做 ANN，适合大规模候选生成；但 user 与 item 在打分前不做细粒度交互，难以替代 cross-encoder 或复杂精排。
2. 一个 batch 内其他正例可直接充当 negatives，因此不用额外编码。若两个用户都喜欢同一 item，或语义相近 item 被当作负例，就会产生 false negative。
3. 不是。过难样本可能是错标、假负例或业务上不可区分的候选；应选择模型当前能学到、标签又可靠的难例。
4. HNSW 用分层近邻图导航；IVF 先按 coarse centroid 缩小搜索分区；PQ 把向量分块量化，用压缩码近似距离。
5. 需要同步模型、向量、ANN 索引、特征 schema 和 item 可用状态版本，并保证灰度期间 query tower 与索引向量来自兼容版本。
6. 长尾物品的点击监督很少，自监督可以从内容字段的不同视图获得额外训练信号，让同一物品在特征缺失或扰动后仍有稳定表示。
7. 若大量 item 集中在少数路径，热门路径返回的 posting 过长，召回成本和候选拥塞都会上升，其他路径也学不到有效分工。

</details>

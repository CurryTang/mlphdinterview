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

#### 工业级原型：YouTube DNN (2016) 召回双塔与 Example Age
YouTube 2016 奠定了现代双塔召回的基准架构：
- **用户塔输入**：观看历史 ID 序列（平均池化 Embedding）+ 搜索词 Query 序列（平均池化 Embedding）+ 人口统计学特征与地理静态特征；
- **时效性连续特征 Example Age**：
  - **问题**：推荐系统存在强烈的时效性偏置（Recency Bias）。老视频上传时间久、累计播放量大，其在正样本中占据统治地位；新发布的优质视频由于没有历史播放，模型打分天然偏低，难以获得曝光机会；
  - **训练期建模**：将样本生成时间距离视频上传时间的差值作为连续特征输入网络：$x_{\text{age}} = t_{\text{event}} - t_{\text{upload}}$；网络学习到视频热度随年龄衰减的规律；
  - **线上推理 Trick**：在线预估候选新视频向量时，将所有候选的 `example_age` 统一人为置为 0（甚至微小负数，等价于“假设该视频刚刚上传即刻”），彻底消除了历史马太效应与时效性偏置，新视频能仅凭内容质量与用户兴趣获得公平的召回竞争力。

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
- 批内负样本 (In-batch Negatives)：同 Batch 内其他用户的正样本作为当前用户的负样本，吞吐极高；
- 曝光未点击：很难，且包含位置偏差和大量假负例；
- hard negative：由旧模型或 BM25 召回、语义相近但不相关的候选；
- 混合负样本：兼顾覆盖、难度和稳定性。

批内负样本有 false negative。两个用户可能都喜欢同一物品，或者一个 query 的负例其实也相关。可通过去重、同 query mask、流行度修正和软标签缓解。

对召回模型，曝光未点击通常不宜作为默认负例。能进入曝光，说明旧召回和排序已经认为它有一定价值；没有点击还可能只是位置、时间或偶然行为。更稳的组合是全库随机/批内负例加上被粗排或精排淘汰的 hard negative，再单独验证曝光未点击是否真的带来收益。

Hard negative 也会过期。模型修复一批错误后，旧难例可能已经变得太简单；长期只训练固定 mined set，又会过拟合少数错误模式。常见做法是周期性用当前 checkpoint 重挖，混入一部分稳定随机负例，并人工抽查 top hard negatives 中的假负例比例。被旧模型淘汰只说明"旧模型不选它"，不自动构成可靠负标签。

#### 批内负采样流行度偏置与 Google logQ 修正 (Sampling-Bias Correction)
批内负采样虽然计算极快（一个 Batch 内 $B$ 个样本无需额外计算负样本 Embedding，直接通过矩阵乘法 $\mathbf{U} \mathbf{V}^T$ 获得 $B \times B$ 的相似度矩阵），但引入了严重的**采样分布偏差**：
- 在-batch 样本中，物品 $j$ 被选为负样本的边缘概率正比于其全站展现频次：$p_j \propto \text{frequency}_j$；
- **病态后果**：高曝光的热门头部物品被当做负样本惩罚的概率远高于长尾冷门物品！模型为了最小化损失，会无意识地大幅压低头部物品的向量模长与内积分数，线上推理时导致热门优质物品全面失效，长尾长尾异常飙升。

**Google logQ 修正的严格数学推导：**
全库全量 Softmax 的真实交叉熵目标为：
$$\mathcal{L} = -\sum_{i=1}^B \log \frac{\exp(s(u_i, y_i))}{\sum_{j \in \mathcal{V}} \exp(s(u_i, j))}$$

利用重要性采样（Importance Sampling），当负样本从分布 $P$ 中按概率 $p_j$ 独立抽取时，分母的全库配分函数是其重要性权重的无偏估计：
$$\sum_{j \in \mathcal{V}} \exp(s(u_i, j)) = \mathbb{E}_{j \sim P}\left[ \frac{\exp(s(u_i, j))}{p_j} \right] \approx \sum_{j \in \mathcal{B}} \exp\left(s(u_i, j) - \log p_j\right)$$

因此，将批内打分 Logits 显式减去 $\log p_j$：

```math
s'(u_i, j) = s(u_i, j) - \log p_j
```

训练损失转化为：

```math
\mathcal{L}_{\text{in-batch}} = -\sum_{i=1}^B \log \frac{\exp\left(s(u_i, y_i) - \log p_{y_i}\right)}{\exp\left(s(u_i, y_i) - \log p_{y_i}\right) + \sum_{j \in \mathcal{B}, j \ne y_i} \exp\left(s(u_i, j) - \log p_j\right)}
```

- **训练与线上服务解耦**：训练时通过 $-\log p_j$ 抵消热门物品被频繁采样的梯度惩罚；**在线检索时直接移除 $-\log p_j$ 项**，还原为纯净的向量内积 $\langle \mathbf{u}, \mathbf{v}_j \rangle$ 运行 ANN 索引，从理论上保证了在线相似度打分的无偏性。

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

过滤顺序会改变实际 Recall。先取 ANN top-100，再过滤库存、地域和类目，可能一条也不剩；盲目把 top-k 放大到 1000 又会增加索引与后级成本。可选方案包括按强过滤条件分片建索引、使用支持 metadata filter 的 ANN、先做粗过滤再向量检索，或根据历史过滤率动态 over-fetch。离线 ANN Recall 很高但线上经常回填热门，往往是过滤后的 Recall 出了问题。

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

增量更新必须守住向量空间兼容性。若白天修改了 item tower、共享底层或会影响两侧坐标系的参数，新的 query embedding 会与旧 ANN 中的 item embedding 失配。只更新用户侧独有参数时可以复用旧索引；动到共享空间时，要同步重算 item 向量并切换索引，或继续使用旧 query tower 直到新索引就绪。

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

### 4.11 核心攻坚 Q&A：双塔计算开销拆解与 ANN 索引底层原理

<details>
<summary>Q1: 双塔在线检索系统的端到端时延由哪些核心开销构成？各环节存在哪些系统级瓶颈与工程优化策略？</summary>

双塔在线检索（Two-Tower Online Retrieval）的端到端时延（End-to-End Latency）主要由**Query 塔在线推理**、**ANN 向量检索**以及**系统、网络与后处理开销**三大部分决定。整套系统的架构设计本质上是在**召回率（Recall）、显存/内存带宽（Memory Bandwidth）与长尾延迟（P99 Latency）**之间寻找最佳平衡点。

#### 一、 Query 塔在线推理开销（Query Tower Inference）

1. **核心计算瓶颈**
   - 耗时直接取决于 Query 塔的深度与算子复杂度（如深度全连接层 DNN、长行为序列 Transformer/Self-Attention、交叉特征网络）。
   - 用户侧特征（实时交互序列、动态上下文、统计画像）高度动态，无法像 Item 塔那样离线预计算，必须在请求到达时实时完成前向推理。

2. **工业级工程优化策略**
   - **高频用户特征与向量缓存（Embedding Caching）**：
     - 在内存（如本地 LRU Cache）或分布式高速缓存（如 Redis Cluster）中缓存高频活跃用户的 Query Embedding。在短时间窗口（如 5~15 分钟）内，若用户无新行为发生，直接命中 Cache 跳过神经网络前向推理，将 P50 推理开销降为 0ms。
   - **服务层动态批处理（Dynamic Batching）**：
     - 在推理网关或 Serving 框架（如 NVIDIA Triton、TorchServe）设置微秒级排队窗口（`max_queue_delay_microseconds = 1000 ~ 2000`）与最大批尺寸（`max_batch_size = 32 ~ 64`）。利用 GPU 的高并发矩阵计算能力摊平访存开销，在单请求轻微增加 1~2ms 延迟的代价下，将系统吞吐量（QPS）提升数倍。

---

#### 二、 ANN 向量检索核心开销（Vector Search Bottlenecks）

ANN 阶段是整个召回系统的计算与访存重灾区，主要受三大物理约束影响：**向量维度 $d$**、**候选集全量规模 $N$** 以及 **ANN 索引结构与检索超参数**。

1. **计算与显存带宽瓶颈（Memory Bandwidth Bound）**
   - 向量检索虽然涉及浮点内积乘加，但在大规模候选池场景下，瓶颈通常不在算力（ALU），而在**内存/显存读取带宽**。
   - 假设全库有 $N = 10^7$（1000 万）个物品，每个物品向量维度 $d = 128$（FP32 占用 512 字节），全量索引占用内存达 $5.12\text{ GB}$。若单次查询需扫描大量候选，高并发下显存带宽会被瞬间打满。

2. **向量量化（Quantization: INT8 / PQ）降维与降带宽**
   - **标量量化（Scalar Quantization, SQ8 / INT8）**：将 4 字节 Float32 映射为 1 字节 Int8，内存带宽与存储直接缩小至原来的 $25\%$，并可调用 CPU AVX-512 或 GPU DP4A 向量化整数指令加速内积计算。
   - **乘积量化（Product Quantization, PQ）**：将 128 维向量切分为 8 个 16 维子向量，每个子向量量化为 256 个簇心之一的 1 字节索引。单个向量压缩至仅 8 字节，压缩比高达 $64\times$，大幅缓解总线带宽压力。

3. **超参对 Recall 与延迟的敏感性调优**
   - **HNSW 索引**：核心超参为搜索候选池大小 `efSearch` 和构建邻居度数 `M`。增大 `efSearch` 能显著提升多层图跳跃时的贪婪路径搜索质量，使 Recall@K 提升，但单次查询的内积计算次数线性上升，导致 P99 延迟劣化。
   - **IVF 索引**：核心超参为探测聚类中心数 `nprobe` 与总分桶数 `nlist`。增大 `nprobe` 允许扫描更多相关倒排桶，减少漏召回边界样本，但扫描的 Posting 链表长度成倍增长。

4. **分片检索与 RPC 放大效应（Index Sharding & Scatter-Gather）**
   - 当候选集达到几千万到数亿规模时，单个节点内存与算力无法承载，必须采用水平分片（Sharding），将索引均摊在 $S$ 个检索节点上。
   - **Scatter-Gather 风险**：查询由聚合网关并发广播（Scatter）到 $S$ 个 Shard，各分片独立检索出 Top-$K$ 候选项后汇聚（Gather）。
   - **长尾延迟放大（The Tail at Scale）**：根据木桶效应，端到端延迟取决于最慢的节点：
     $$P99_{\text{overall}} \approx 1 - (1 - P99_{\text{single}})^S$$
     随着分片数 $S$ 增加，网络 RPC 抖动、节点垃圾回收（GC）或瞬时负载都会被指数级放大，推高系统 P99 与 P999 延迟。

---

#### 三、 过滤、合并与重排系统开销（Filtering & Rescoring）

1. **业务规则过滤（Hard Filtering）的系统两难**
   - 推荐系统存在库存售罄、下架、用户已读、类目拉黑等严格业务过滤规则。
   - **前置过滤（Pre-Filtering）**：若在 ANN 检索前按布尔条件过滤，会直接破坏 HNSW 等图索引的连通性拓扑，导致图遍历陷入死胡同或断崖式早停；
   - **后置过滤（Post-Filtering）**：若在 ANN 检索后过滤，为了防止最终满足条件的候选数不足 $K$ 个，必须进行激进的过度召回（Over-fetching，如将候选数放大至 $K' = 10 \times K$），这会成倍加重 ANN 遍历与后续链路的传输负担；
   - **折中优化**：工业界常采用**带元数据过滤的迭代图搜索（AC-HNSW）**或基于高频类目的分片独立建库。

2. **分片合并与精确重排（Top-K Reranking / Exact Rescoring）**
   - **堆排序合并**：聚合网关收集 $S$ 个分片返回的 $S \times K'$ 个候选，利用最小堆（Min-Heap）进行多路归并，提取全局粗筛 Top-$K'$。
   - **回表精确打分（FP32 Rescoring）**：量化索引（PQ/SQ8）存在量化误差（Quantization Loss），在粗筛截断后，系统常从内存或 SSD 中根据 Item ID 回表读取原始 FP32 稠密向量，对 Top-$K'$ 个候选重新执行高精度向量内积，按真实几何相似度重新排序。该回表访存与二次打分显著提升了最终 Recall，但也增加了端到端的内存读取开销。

</details>

<details>
<summary>Q2: 向量索引（Index）在工程底层到底长什么样？为什么拿到 Query Embedding 就可以“直接 Top-K”？</summary>

在工程实现中（如 Faiss、ScaNN、Milvus 等向量检索引擎），Index 并不是玄学黑盒，本质上就是**特定组织形式的内存数据结构（倒排链表、邻接图或压缩表）**。双塔模型之能够“直接 Top-K”，是由其数学打分机制与底层 C++ 剪枝逻辑共同保障的。

#### 一、 工业界两大主流 Index 的物理内存形态

---

##### 1. 倒排索引形态：IVF-PQ（Inverted File with Product Quantization）

IVF-PQ 是工业界最经典、内存占用极低的倒排压缩索引，其物理组织结构类似于一本带有两级目录的字典：

```text
[聚类中心表 (Centroids Table)]
  Cluster 0: [ 0.12, -0.45,  0.88, ... ] (d 维 Float32 向量)
  Cluster 1: [ 0.81,  0.03, -0.21, ... ]
  ...
  Cluster K: [ ... ]

[倒排链表 (Inverted Lists / Postings)]
  Key: Cluster 0 ──► List: [ 
                             (Item_102, [PQ 压缩码: 0x1A, 0x3F, 0x09, 0xB2, ...]),
                             (Item_589, [PQ 压缩码: 0x02, 0x8C, 0xF1, 0x4D, ...]),
                             ...
                           ]
  Key: Cluster 1 ──► List: [ (Item_12, ...), (Item_441, ...), ... ]
  ...
```

* **离线构建流程（Offline Indexing）**：
  1. **粗聚类（Coarse Quantization）**：对全库千万级 Item 向量执行 K-Means 聚类，产生 $K$ 个粗聚类中心（Centroids，例如 $K = 4096$）。
  2. **倒排分桶（Inverted Posting Assignment）**：将每个 Item 根据最近邻原则归入与其距离最近的簇心，挂载至对应的倒排链表下。
  3. **细粒度残差量化（Product Quantization）**：计算 Item 向量与其所属簇心的残差向量 $\mathbf{r} = \mathbf{v} - \mathbf{c}_k$，将残差切分为 $M$ 个子向量（如 8 个），每个子向量通过子码本量化为 1 字节的整数索引（Byte Code）。原本 128 维 Float32（耗费 512 字节）被压缩至 8 字节。
* **物理内存占用**：千万级物品库仅需数百 MB 到几 GB 内存，可完整常驻 RAM。

---

##### 2. 图索引形态：HNSW（Hierarchical Navigable Small World）

HNSW 是当前召回精度（Recall）最高的主流索引，其结构是**多层跳表（Skip-List）思想在多维空间图拓扑（Graph）上的推广**：

```text
[Layer 2 (高速路 / 稀疏长跳跃)]     Node_A ───────────────────────────► Node_K
                                    │                                  │
                                    ▼                                  ▼
[Layer 1 (中速路 / 次级细化)]       Node_A ────────► Node_D ─────────► Node_K ────► Node_M
                                    │                │                 │            │
                                    ▼                ▼                 ▼            ▼
[Layer 0 (底层 / 全量节点邻接图)]   全量 Item 构成的邻接表（每个节点维护指向多维邻近的 16~64 个邻居指针）
```

* **底层的物理数据结构**：
  1. **稠密节点数据数组**：连续存储的 `Item_ID -> Vector` 数组（可保留 FP32 或经 SQ8 量化）。
  2. **分层邻接表（Multi-Layer Adjacency List）**：每个节点在每一层维护一个变长数组 `neighbors: List[int]`，记录其在几何空间中距离最近的 $M$ 个双向连接邻居节点的 Item ID。顶层稀疏分布，底层覆盖全量节点。

---

#### 二、 为什么拿到了 Query Embedding 就可以“直接 Top-K”？

很多人初学推荐系统时会疑惑：*“拿到用户向量后，难道不需要再跑一个模型对候选 Item 挨个计算并打分吗？”*

**完全不需要。** 这是双塔模型在数学设计与工程架构上刻意实现的结果：

##### 1. 数学本质：打分函数被严格定义为了“纯几何运算”

精排模型（如 DCN、DIN）之所以无法直接 Top-K，是因为其打分函数存在强非线性交叉：
$$\text{Score} = \text{MLP}\Big(\text{Concat}(\mathbf{u}, \mathbf{v}, \mathbf{u} \times \mathbf{v})\Big)$$
这必须把 $\mathbf{u}$ 和千万个候选 $\mathbf{v}$ 一对一拼接起来，执行千万次深度神经网络前向传播，在线算力完全无法支持。

而双塔模型在建模时，**强制将打分函数定义为纯几何内积（Dot Product）或余弦相似度（Cosine Similarity）**：
$$\text{Score}(u, v) = \mathbf{u}^\top \mathbf{v} \quad \left(\text{若经 } L_2 \text{ 归一化，即 } \cos(\mathbf{u}, \mathbf{v}) = 1 - \frac{1}{2}\Vert \mathbf{u} - \mathbf{v} \Vert_2^2\right)$$

- **数学等价性**：
  - 用户所有复杂的历史交互、画像与长程兴趣，被用户塔深度压缩封装至单一一枚 $d$ 维稠密向量 $\mathbf{u}$ 中；
  - 物品的所有类目、价格、文本多模态信息，也已提前被物品塔离线压缩进静态向量 $\mathbf{v}$ 中；
  - 寻找“预估分数最高的 Top-$K$ 物品”，在数学上完全等价于**在多维几何空间中寻找距离向量 $\mathbf{u}$ 最近（夹角最小 / 内积最大）的 Top-$K$ 个点**（即 MIPS: Maximum Inner Product Search）。

---

##### 2. 工程实现：ANN 索引如何在 5ms 内完成 Top-K 检索？

因为 Top-K 被成功降维为纯几何剪枝问题，就不再需要经过任何深度学习框架的 Forward 计算，而是依靠底层 C++ / 汇编级优化完成毫秒级剪枝：

- **若使用 IVF-PQ 检索**：
  1. **粗聚类中心初筛（微秒级）**：将在线生成的 $\mathbf{u}$ 与 $K=4096$ 个聚类中心做内积，仅挑出距离最近的 $n_{\text{probe}}$ 个中心（例如最近的 8 个簇）；
  2. **空间剪枝（毫秒级）**：99% 以上的无关物品被直接跳过，仅需扫描这 8 个倒排链表内的候选（约数万个 Item）；
  3. **非对称距离查表加速（ADC, Asymmetric Distance Computation）**：Query 向量保持 FP32 不量化，预先计算 Query 子向量与 PQ 码本中所有聚类中心的内积查找表（Lookup Table，尺寸仅 $M \times 256$）。随后遍历链表时**无需对 Item 向量执行反量化**，只需根据 Item 的 8 字节 PQ Code 直接查表求和。借助 CPU AVX-512 向量化整数指令，在 1~3ms 内完成几万次内积累加，并利用大小为 $K$ 的最小堆（Min-Heap）维护最高得分，输出 Top-K。

- **若使用 HNSW 检索**：
  1. **顶层快速导航**：从最顶层（Layer 2）入口点进入，计算与当前邻居的内积，沿着“内积单调递增”的方向像下山一样贪婪跳转；
  2. **逐层下潜（Layer Descent）**：当顶层邻居无法找到更大内积时，以下潜点为基准降至下一层（Layer 1），在更密集的图拓扑中继续贪婪游走；
  3. **底层收敛与候选堆维护**：最终落入全量 Layer 0，仅计算数百到上千次向量内积，就能以 $95\%+$ 的召回精度锁定空间中最靠近 $\mathbf{u}$ 的 Top-K 节点。

---

#### 三、 一句话总结

双塔模型通过将复杂特征交互退化为**线性内积**，成功将“在线复杂概率推断”降维为了“空间几何邻近搜索”。一旦生成 Query 向量，后续的 Top-K 计算仅包含高密度底层 C++ 向量点积、SIMD 查表剪枝与最小堆归并，因此能在 5ms 内从千万级候选池中极速召回。

</details>


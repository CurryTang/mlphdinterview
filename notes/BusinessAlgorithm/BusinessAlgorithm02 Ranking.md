# 排序目标与离线评价

## 第 9 章 排序目标与离线评价

排序模型看到的不是全库，而是上游策略筛过的一小批候选。它学到的是"在这套召回、过滤和曝光策略给出的候选中，谁更值得靠前"，不是脱离候选分布的全局偏好函数。召回版本一变，负例难度、类目比例和分数分布都会跟着变；旧测试集上的提升未必还能保留。

这也是排序评价必须固定候选集的原因。同一批候选上比较，才能隔离模型的调序能力；让新模型同时换召回，再报一个 NDCG，无法判断收益来自候选变好还是排序变好。上线时两项都重要，诊断时必须拆开。

排序还要先决定分数的含义。CTR 模型的分数可以校准成点击概率，相关性模型的分数可能只保证档位或顺序，多目标模型的最终分数甚至只是业务效用。分数含义不清，后面的阈值、融合和实验结论都会变得含糊。

### 9.1 排序先定义"相关"

搜索相关性通常是分档标签：

- 高度相关：直接满足 query 的主要意图；
- 相关：能解决问题，但不够完整或不够精确；
- 弱相关：只覆盖部分意图；
- 不相关。

真实标注比这麻烦。query 可能多义，文档可能质量差但主题相关，时效性也会改变答案。标注规范应把"主题相关性"和"最终是否值得展示"分开，否则模型会把质量、权威性和相关性搅在一起。

推荐标签来自行为。点击、有效播放、点赞、关注和购买代表不同强度，也有不同延迟。模型目标必须对应产品目标，不能因为点击样本多就永远只学 CTR。

### 9.2 Pointwise

Pointwise 把每个候选当作独立分类或回归样本：

$$
\mathcal{L}_{\text{point}} = -\left[ y\log p + (1-y)\log(1-p) \right]
$$

优点是样本和训练都简单，预测概率还能做校准。缺点是它没有直接表达同一个 query/user 下候选之间的顺序。

CTR、CVR 和时长预估多从 pointwise 开始。搜索相关性若有分档，也可做多分类或回归。

### 9.3 Pairwise

Pairwise 构造正负候选对，希望正例分数更高：

$$
\mathcal{L}_{\text{pair}} = -\log \sigma(s^+ - s^-)
$$

它更接近"谁排在谁前面"，但 pair 数量可能爆炸。负例怎么选会明显改变训练：随机负例简单，当前模型排错的 hard negative 更有信息，也更容易包含标注噪声。

RankNet 属于 pairwise。LambdaRank/LambdaMART 会根据交换两个候选对 NDCG 的影响调整梯度权重，让列表头部错误更受重视。

### 9.4 Listwise

Listwise 把整个候选列表作为训练对象。一个简单形式是对列表做 softmax：

$$
P(i \mid q) = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

再用目标分布做交叉熵。也可以直接优化 NDCG 的近似目标或生成候选排列。

Listwise 更贴近最终任务，却要求同 query 的候选成组进入训练，显存、采样和实现都更复杂。LLM 排序常使用 listwise prompt，但长候选列表会遇到上下文窗口和位置偏差，第 18 章会展开。

### 9.5 搜索相关性模型

传统文本分数包括：

- BM25；
- query term 覆盖；
- 词距和顺序；
- 标题、正文、锚文本等字段匹配；
- 点击与跳出统计。

Cross-BERT 把 query 与文档一同编码，能识别深层语义和否定关系。它的服务成本高，常放在后级，只处理召回后的 top-N。

双塔 BERT 可预计算文档表示，适合召回；Cross-BERT 适合精排。比较两者时要把调用次数算进去：top-200 精排意味着每个 query 要执行 200 个 query-document 前向。

搜索链路中的召回截断、粗排和精排都在做融合，只是候选量和模型成本不同。三层都可能读取相关性、CTR/交互率、内容质量、时效、地域与统计特征：召回截断偏向双塔和线性模型，粗排用较小 Cross-BERT/双塔和树模型，精排才承担最重的交叉模型。

搜索建设早期通常先用融合规则：按相关性分档，档内融合点击、质量和时效。这样遇到标题党或视频占比异常时可以直接调权重。数据积累后再训练融合模型，但要继续监控 top-k 各相关性档位占比，防止模型靠弱化相关性换点击。

融合模型的监督可结合人工综合满意度和用户行为。人工标注只看 query、document、时间和地点，给出包含相关性、质量和时效的综合档位；点击与交互补充个性化偏好。标注量不够时，先用非个性化特征训练小 teacher，给海量日志 pair 估计满意度，再与行为组成目标训练包含用户特征的线上融合模型。Teacher 不能使用未来信息，文档年龄也必须按请求时刻计算。

### 9.6 工业级推荐与搜索全链路评估指标体系

在工业级推荐系统（RecSys）与搜索系统中，离线评估不能只靠单一指标（如全局 AUC 或 NDCG）包打天下。完整的推荐流水线呈现多阶段漏斗结构：**召回 ➔ 粗排 ➔ 精排 ➔ 重排/端排 ➔ 校准与竞价出价 ➔ 在线 A/B 实验与长期商业生态**。每一个阶段的候选规模、特征丰富度、算力时延约束（SLA）以及任务目标截然不同，对应着独特的评估维度与防坑红线。

#### 9.6.1 全链路各阶段评估维度与指标速查表

| 链路阶段 | 核心指标 | 核心评估意图 / 典型业务场景 | 正常基准与读取方式 | 侧重点与核心局限（避坑指南） |
|---|---|---|---|---|
| **① 召回阶段**<br>(候选生成/向量双塔/Graph) | **Recall@K** | 衡量千万级全库粗筛至数千时，正反馈物品被捞回的比例。 | $0 \sim 1$，$K \in [200, 1000]$。固定候选池下越高越好。 | **漏斗天花板**：召回漏掉的样本下游无法挽救；但不保证排在顶部位。 |
| | **HitRate@K** | Top-$K$ 列表中是否至少命中 1 个正例。适合“单次只要 1 个满意即算成功”场景。 | $0 \sim 1$。首屏大卡或单列流核心指标。 | 二值统计，对命中多个正例的丰富度不敏感，无法度量多兴趣覆盖。 |
| | **MRR@K** | 系统找回**首个相关正例**的速度与展现位次。 | $(0, 1]$。第 1 位命中得 1.0，第 10 位仅 0.1。 | 强惩罚首位未命中；但完全忽略第 2、第 3 个正例排位（适合问答搜索）。 |
| | **Coverage (覆盖率)** | 衡量召回池能激活全库多少比例的长尾与冷启动内容。 | 百分比。越高代表长尾挖掘越充分。 | 必须与相关性/CTR 联合看，单纯追求高覆盖会拉入大量低质噪音。 |
| **② 粗排阶段**<br>(轻量网络/大规模初筛) | **Kendall's $\tau$ / Spearman's $\rho$** | 粗排轻量模型与大精排打分排序的相对次序一致性。 | $[-1, 1]$。要求与精排维持高正相关（通常 $\tau > 0.6$）。 | 粗排受 P99 $\le 5\text{ms}$ 强约束；使命是**不误杀精排眼中的 Top 候选**。 |
| | **Recall@Top-M against Ranker** | 精排判定为 Top-$N$ 的优质候选，粗排 Top-$M$ 捞回了百分之多少。 | 衡量粗排截断造成的**精排上限损失率**。 | 依赖“精排打分为伪真值”；若精排本身学偏，粗排会同向偏离。 |
| **③ 精排阶段**<br>(CTR / CVR / 多任务) | **Request-GAUC** | **精排第一黄金离线指标**：单次刷新同屏展现内正例排在负例前面的概率。 | 按 RequestID 分组计算 AUC 后按曝光加权。$\Delta\text{GAUC} \ge +0.003$ 线上显著。 | **彻底切断跨请求/跨用户先验偏倚与辛普森悖论**；单刷全负例不参与计算。 |
| | **User-GAUC** | 同一用户历史曝光集合内正例排在负例前面的能力。 | 按 UserID 分组做 AUC 加权。通常高于 Request-GAUC。 | 易受用户早晚意图漂移（白天工作、夜晚娱乐）干扰，组内负例包含不同时段。 |
| | **Global ROC-AUC** | 粗略观测模型在全量正负样本上的全局统计区分度。 | 0.5 为随机，1.0 完美。一般作为底线防御指标。 | **虚假繁荣陷阱**：极易被大活跃用户或热门 Item 抬高，存在辛普森悖论。 |
| | **PR-AUC (AP)** | **极度稀疏转化场景（如大额 CVR、高危风控拦截）核心指标**。 | PR 曲线下面积。**必须与正例先验基线 $\pi = P(Y=1)$ 一起对比**。 | 负例极大时规避海量 TN 稀释 FPR 的假象；但跨先验数据集不可直接对比绝对值。 |
| | **LogLoss** | 评估预测概率值的绝对拟合精度；对自信犯错严惩。 | 越小越好。通常看相对基线的降幅。 | 绝对值极易受大盘自然转化率波动干扰；负采样训练下必须做重要性加权。 |
| | **NE / RIG** | Meta/TikTok 工业标准 CTR 评价指标。衡量相对背景熵的不确定性压缩比。 | NE 越小越好，RIG 越高越好（RIG 提升 1% 即为显著技术突破）。 | 严格剔除了节假日/大促等背景 CTR 波动对 Loss 的干扰，跨数据集可比。 |
| **④ 重排与端排**<br>(整页呈现/序列决策) | **NDCG@K** | 衡量列表头部展示位的多级相关度满意度，带**位置递减对数折扣**。 | $0 \sim 1$。$K$ 必须严格对齐真实终端屏幕视窗坑位数（如 $K=4 \sim 6$）。 | 对列表尾部的排序颠倒完全不敏感；未显式考虑候选间的相互替代效应。 |
| | **ILD (列表内多样性)** | 屏内或单次会话推荐内容在 Embedding 或类目维度的丰富度。 | 平均两两语义距离。越大说明同屏推荐越丰富。 | 多样性与短期 CTR 存在帕累托博弈；过度多样化可能打断沉浸式体验。 |
| | **Novelty / Serendipity** | 新颖性自信息量：惩罚无脑推全站热门，奖励发掘个性化小众精品。 | 越大说明新颖度越高。 | 必须以用户满意为前提，否则容易推怪异生僻内容损伤体验。 |
| **⑤ 校准与出价**<br>(广告/电商/变现) | **PCOC** | 预测概率和与真实观测转化数之比（全盘高低估比）。 | **完美校准为 1.000**。1.10 代表高估 10%，0.90 代表低估 10%。 | **商业化出价生命线**：高估导致广告主预算超扣赔付，低估导致跑不出量。 |
| | **ECE (分桶校准误差)** | 将预测概率分桶，衡量不同置信区间内局部预估概率与后验频率的加权绝对差。 | 越小越好（理想为 0）。 | 仅做单调变换（如温度调节）能显著优化 ECE，但不会改变 AUC。 |
| | **Brier Score** | 预测概率与真实二值标签之间的均方误差。 | 越小越好。可正交分解为可靠性、分辨力与不确定性。 | 综合反映排序与校准，但不如 AUC 对排序敏感，也不如 PCOC 直观。 |
| **⑥ 在线 A/B 与生态**<br>(业务最终真值) | **长线北极星指标** | DAU、MAU、人均停留时长 (Dwell Time)、7D/30D 留存率、GMV、eCPM。 | 衡量推荐系统对平台整体商业变现与用户心智黏性的终极因果贡献。 | 需跑满 7~14 天消除星期效应与新奇效应；配合 CUPED 缩减方差。 |
| | **过程与护栏指标** | 点击率、完播率、**负反馈率 (Dislike/举报，坚决零退化)**、P99 推理时延。 | 监控短期交互与系统红线防御底线。 | **警惕标题党陷阱**：CTR 大涨但停留时长或留存暴跌说明引入了低质诱导内容。 |
| | **SRM (样本比例失衡)** | 检验 A/B 实验分流机制是否公正、无偏。 | 卡方检验 $p < 0.001$ 判定为 SRM 污染报警。 | **实验有效性基石**：一旦触发 SRM，说明分流受污染，所有指标结论全部作废。 |

---

#### 9.6.2 全链路核心数学公式与形式化推导

##### 1. 召回阶段公式 (Candidate Retrieval)

- **Recall@K (召回率)**：
  $$
  \operatorname{Recall@K} = \frac{\left| \mathcal{R}_K \cap \mathcal{G}^+ \right|}{\left| \mathcal{G}^+ \right|}
  $$
  其中 $\mathcal{R}_K$ 为召回模块返回的 Top-$K$ 候选集合，$\mathcal{G}^+$ 为用户真实交互过的正反馈物品全集。
- **HitRate@K (命中率)**：
  $$
  \operatorname{HR@K} = \mathbb{I}\left( \left| \mathcal{R}_K \cap \mathcal{G}^+ \right| \ge 1 \right)
  $$
  其中 $\mathbb{I}(\cdot)$ 为指示函数，只要前 $K$ 个候选中出现至少一个正例即记为 1，否则为 0。
- **MRR@K (Mean Reciprocal Rank，平均倒数排名)**：
  $$
  \operatorname{MRR@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\operatorname{rank}_q^{(1)}}
  $$
  其中 $\operatorname{rank}_q^{(1)}$ 为第 $q$ 个请求中首个命中正例的展示位次；若前 $K$ 位均未命中，则该项记为 0。
- **全库覆盖率 (Catalog Coverage)**：
  $$
  \operatorname{Coverage} = \frac{\left| \bigcup_{u \in \mathcal{U}} \mathcal{R}_K(u) \right|}{\left| \mathcal{I}_{\text{total}} \right|}
  $$
  衡量全体用户测试集 $\mathcal{U}$ 在 Top-$K$ 召回中被激活的不同物品并集占全库总物品集 $\mathcal{I}_{\text{total}}$ 的比例。

##### 2. 粗排阶段公式 (Pre-ranking & Lightweight Filtering)

- **Kendall's $\tau$ 秩相关系数 (粗排与精排打分保序一致性)**：
  $$
  \tau = \frac{P - Q}{\frac{1}{2} M (M - 1)}
  $$
  其中 $M$ 为送入粗排的候选总量，$P$ 为粗排与精排打分顺序一致的 candidate pair 数量，$Q$ 为顺序相反的逆序对数量。
- **截断召回率 (Recall@Top-M against Heavy Ranker)**：
  $$
  \operatorname{Recall@Top\text{-}M}_{\text{coarse}} = \frac{\left| \mathcal{C}_{\text{coarse\_M}} \cap \mathcal{C}_{\text{fine\_topN}} \right|}{N}
  $$
  衡量在完整精排模型打分最高的 Top-$N$ 黄金候选中，粗排截断 Top-$M$ 成功保留的比例。

##### 3. 精排阶段公式 (Heavy Ranking & Scoring)

- **Request-GAUC (一刷内请求级分组 AUC，精排第一黄金指标)**：
  $$
  \operatorname{Request-GAUC} = \frac{\sum_{r \in \mathcal{R}, \, n_r^+ > 0, \, n_r^- > 0} w_r \cdot \operatorname{AUC}_r}{\sum_{r \in \mathcal{R}, \, n_r^+ > 0, \, n_r^- > 0} w_r}
  $$
  其中 $\operatorname{AUC}_r$ 是严格在单次刷新请求 $r$ 的同屏曝光集合内部计算的局部 AUC，权重 $w_r = n_r$ 取该次请求的有效曝光展现量。单刷内全为点击或全为未点击的请求由于无正负对，分母为 0，需显式跳过。
- **User-GAUC (用户级分组 AUC)**：
  $$
  \operatorname{User-GAUC} = \frac{\sum_{u \in \mathcal{U}, \, n_u^+ > 0, \, n_u^- > 0} w_u \cdot \operatorname{AUC}_u}{\sum_{u \in \mathcal{U}, \, n_u^+ > 0, \, n_u^- > 0} w_u}
  $$
  其中 $\operatorname{AUC}_u$ 为单用户 $u$ 历史所有曝光聚合集合内的局部 AUC，权重 $w_u$ 通常取该用户的总曝光数。
- **全局 ROC-AUC (Global AUC)**：
  $$
  \operatorname{AUC}_{\text{global}} = \frac{1}{\left| \mathcal{D}^+ \right| \cdot \left| \mathcal{D}^- \right|} \sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \left( \mathbb{I}(p_i > p_j) + \frac{1}{2}\mathbb{I}(p_i = p_j) \right)
  $$
  在整个测试集跨用户混合比对正负样本对。
- **PR-AUC (Precision-Recall AUC / Average Precision)**：
  $$
  \operatorname{PR-AUC} = \sum_{k=1}^N \left( \operatorname{Recall}_k - \operatorname{Recall}_{k-1} \right) \cdot \operatorname{Precision}_k
  $$
  按预测得分降序排列后，以查全率步进加权查准率积分面积，专注于稀疏正例的检出质量。
- **LogLoss (二进制交叉熵损失)**：
  $$
  \operatorname{LogLoss} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
  $$
- **NE (Normalized Cross Entropy / RIG - 相对信息增益)**：
  $$
  \operatorname{NE} = \frac{\operatorname{LogLoss}(p, y)}{H(\bar{p})} = \frac{-\frac{1}{N}\sum_{i=1}^N \left[ y_i \log p_i + (1-y_i)\log(1-p_i) \right]}{-\bar{p}\log \bar{p} - (1-\bar{p})\log(1-\bar{p})}
  $$
  $$
  \operatorname{RIG} = 1 - \operatorname{NE}
  $$
  其中 $\bar{p} = \frac{1}{N}\sum y_i$ 为数据集的全局经验背景正例率，$H(\bar{p})$ 为其香农信息熵。

##### 4. 重排与端排公式 (Re-ranking & Slate Optimization)

- **NDCG@K (Normalized Discounted Cumulative Gain)**：
  $$
  \operatorname{DCG@K} = \sum_{i=1}^K \frac{2^{\operatorname{rel}_i} - 1}{\log_2(i + 1)}, \quad \operatorname{NDCG@K} = \frac{\operatorname{DCG@K}}{\operatorname{IDCG@K}}
  $$
  其中 $\operatorname{rel}_i$ 为排在第 $i$ 位的相关性等级分数，$\operatorname{IDCG@K}$ 是将当前候选按相关性理想降序排列计算得到的最大理论 DCG 分数。
- **ILD (Intra-List Diversity，列表内多样性)**：
  $$
  \operatorname{ILD} = \frac{2}{K(K - 1)} \sum_{i=1}^K \sum_{j=i+1}^K \operatorname{dist}(\text{item}_i, \text{item}_j)
  $$
  其中 $\operatorname{dist}(i, j) = 1 - \cos(\mathbf{e}_i, \mathbf{e}_j)$ 为物品 Embedding 的余弦距离或类目差异指示变量。
- **新颖度自信息量 (Novelty / Serendipity)**：
  $$
  \operatorname{Novelty} = \frac{1}{K} \sum_{i=1}^K -\log_2 P(\text{item}_i)
  $$
  其中 $P(\text{item}_i)$ 为物品在全站全局曝光中的边缘先验概率，长尾冷门物品产生更高的自信息量。

##### 5. 校准度与竞价出价公式 (Calibration & Bidding)

- **PCOC (Predictive-over-Observed Ratio，全盘高低估比)**：
  $$
  \operatorname{PCOC} = \frac{\sum_{i=1}^N \hat{p}_i}{\sum_{i=1}^N y_i} \quad (\text{完美校准 } = 1.000)
  $$
- **ECE (Expected Calibration Error，分桶期望校准误差)**：
  $$
  \operatorname{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|
  $$
  将样本按预测概率划入 $M$ 个区间桶 $B_m$，$\operatorname{conf}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m}\hat{p}_i$ 为桶内平均预测置信度，$\operatorname{acc}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m} y_i$ 为桶内真实经验发生率。
- **Brier Score 及其正交三项分解**：
  $$
  \operatorname{BS} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2 = \operatorname{Reliability} - \operatorname{Resolution} + \operatorname{Uncertainty}
  $$
  其中 $\operatorname{Reliability} = \sum_m \frac{|B_m|}{N}(\operatorname{conf}(B_m) - \operatorname{acc}(B_m))^2$ 度量校准偏差，$\operatorname{Resolution} = \sum_m \frac{|B_m|}{N}(\operatorname{acc}(B_m) - \bar{y})^2$ 度量排序分辨力，$\operatorname{Uncertainty} = \bar{y}(1 - \bar{y})$ 为数据固有方差。
- **负采样几率比反解公式 (Odds Ratio Inversion Formula)**：
  在负采样率 $w \in (0, 1)$ 下，模型直接输出的得分记为 $p_{\text{sampled}}$，线上真实物理概率还原公式为：
  $$
  \operatorname{Odds}_{\text{real}} = \operatorname{Odds}_{\text{sampled}} \cdot w \implies \frac{p_{\text{real}}}{1 - p_{\text{real}}} = \frac{p_{\text{sampled}}}{1 - p_{\text{sampled}}} \cdot w
  $$
  $$
  p_{\text{real}} = \frac{p_{\text{sampled}}}{p_{\text{sampled}} + \frac{1 - p_{\text{sampled}}}{w}}
  $$

##### 6. 在线 A/B 实验与因果检验公式 (Online Experimentation & Causality)

- **SRM 卡方拟合优度检验 (Sample Ratio Mismatch Test)**：
  $$
  \chi^2 = \sum_{k=1}^C \frac{(O_k - E_k)^2}{E_k}, \quad \text{自由度 } df = C - 1
  $$
  其中 $O_k$ 为第 $k$ 组实际观测进流样本数，$E_k$ 为按分流哈希配置计算的理论期望样本数。
- **CUPED 方差缩减估计量 (Controlled-experiment Using Pre-Experiment Data)**：
  $$
  \hat{Y}_{\text{CUPED}} = \bar{Y} - \theta (\bar{X} - \mathbb{E}[X]), \quad \text{其中 } \theta^* = \frac{\operatorname{Cov}(Y, X)}{\operatorname{Var}(X)}
  $$
  $$
  \operatorname{Var}\left( \hat{Y}_{\text{CUPED}} \right) = \operatorname{Var}(\bar{Y}) \cdot \left( 1 - \rho_{XY}^2 \right)
  $$
  利用实验前用户历史指标 $X$（与实验干预严格正交）吸收指标方差，等效大幅缩减所需样本量与实验运行天数。

---

#### 9.6.3 工业级离线评价五大深水区面试考点深度辨析

##### 1. 为什么“全局 ROC-AUC 涨了，但线上 CTR 却跌了”？（辛普森悖论与 Request-GAUC）
- **底层因果机制**：全局 ROC-AUC 混杂了**跨用户的先验偏好差异**。
  - 设用户 A 为极重度用户（天然点击率 40%），模型打分集中在 $[0.6, 0.9]$；
  - 设用户 B 为极轻度用户（天然点击率 2%），模型打分集中在 $[0.1, 0.3]$；
  - 在全局 AUC 的二重求和计算中，正例绝大部分来自用户 A，负例绝大部分来自用户 B。模型甚至无需学会“用户 A 究竟对哪篇内容更感兴趣”，只需学会“用户 A 的特征打高分，用户 B 的特征打低分”，**全局 ROC-AUC 即可轻松冲到 0.85 以上**。
- **线上服务实相**：在手机客户端，用户 A 触发一次刷新，客户端向推荐引擎请求 10 条内容（同屏展示列表）。由于模型在用户 A 内部打分完全随机（局部 $\text{AUC}_A = 0.50$），排在首屏的恰好是用户 A 最厌恶的内容，导致用户直接关掉 App。
- **面试考核切入点**：
  > 为什么推荐精排必须看 **Request-GAUC**？因为同屏候选才构成真实博弈。Request-GAUC 冻结了单次网络请求的上下文，只度量同屏曝光集合内部的相对序，彻底剥离了跨请求、跨用户的混淆变量。

##### 2. 为什么 CVR 与低频风控任务中，必须看 PR-AUC 而非 ROC-AUC？
- **混淆矩阵分母的“稀释效应”数学证明**：
  $$
  \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}, \quad \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$
  在极度不平衡场景（如正负样本比 $1:10000$ 的电商购买转化或高危黑产风控）：
  - 真实负样本基数 $\text{TN}$ 极其庞大。即便模型产生了 5000 个误报（$\text{FP} = 5000$，而实际正例检出 $\text{TP} = 100$），由于分母中 $\text{TN} \approx 10^6$，$\text{FPR} \approx \frac{5000}{1000000} = 0.005$ 极度接近 0！
  - 此时绘制出的 ROC 曲线近乎完美贴合左上角，**ROC-AUC 高达 0.98+（虚假繁荣）**。
  - 然而在实际业务落地中，精确率 $\text{Precision} = \frac{100}{100 + 5000} \approx 1.96\%$！这意味着推给用户的商品 98% 都是误判，或者风控审核人员审查 100 个报警只有不到 2 个是对的，系统彻底瘫痪。
- **先验基线陷阱**：
  - ROC 曲线的随机猜测基线永远是连接 $(0,0)$ 到 $(1,1)$ 的对角线，面积为 **0.5**。
  - PR 曲线的随机猜测基线是一条水平线 $y = \pi = P(Y=1)$（即正例先验率）。如果转化率只有 $0.1\%$，随机模型的 PR-AUC 是 **0.001**。若模型 PR-AUC 达到 0.15，代表相比基线实现了 150 倍的信噪比提升。

##### 3. 为什么工业大厂（Meta/TikTok）模型监控首选 NE (Normalized Cross Entropy / RIG)？
- **公式解析**：
  $$
  \text{NE} = \frac{\operatorname{LogLoss}(p, y)}{- \bar{p}\log\bar{p} - (1-\bar{p})\log(1-\bar{p})}, \quad \text{RIG} = 1 - \text{NE}
  $$
  其中分母是整个测试集依据背景平均转化率 $\bar{p}$ 计算出的**背景香农信息熵 $H(\bar{p})$**。
- **为什么要消除背景熵？**
  - 当周末、节假日或平台大促到来时，大盘自然点击率 $\bar{p}$ 发生天然跳变（例如从 3% 飙升至 7%）。此时即使模型参数分毫不变，由于样本标签的信息熵增加，绝对 LogLoss 也会发生剧烈波动。
  - 研发人员无法区分“LogLoss 上升是模型变差了，还是大盘流量结构变了”。
  - **NE 衡量的是模型比“无脑常数预估器”压缩了百分之多少的相对信息不确定性**。NE 排除了自然流量分布的波动，使得不同时期、不同流量切片之间的模型效果具备严格可比性。

##### 4. 排序分辨力（AUC）与概率校准（PCOC/ECE）的本质解耦与出价风险
- **保序变换对指标的冲击差异**：
  - 若将预估得分通过任意严格单调递增函数变换（例如 $g(s) = s^2$ 或 $g(s) = \sigma(10s)$），所有样本的相对大小次序完全不变，**AUC 和 NDCG@K 保持 100% 恒定不变**。
  - 但是预测概率的物理数值被大幅扭曲，**PCOC 会发生剧烈偏离（如从 1.0 飙升至 2.5），ECE 大幅恶化**。
- **广告与电商出价的灾难性后果**：
  - 现代在线广告出价公式为：$\text{Bid}_{\text{actual}} = p\text{CTR} \times p\text{CVR} \times \text{Bid}_{\text{target}}$。
  - 如果一个模型的 GAUC 从 0.75 提升到 0.78，但 PCOC 从 1.02 恶化到 1.30（系统性高估 30%）：线上竞价时，平台会误以为点击率很高而报出巨额天价，导致广告主预算在 10 分钟内被快速耗尽而无法达成实际转化，触发严重的商业赔付报警。
- **工业界标准架构解法**：
  - **解耦“排序学习”与“概率校准”**：主精排模型（如 DeepFM / DLRM）专注于最大化 GAUC 与排序分辨力；在精排输出层之后，串联一层**参数量极小且保序的独立校准器**（如 100-Bin Isotonic Regression、Platt Scaling 或分桶多项式校准），强制将输出值拉回真实物理概率对角线。

##### 5. 负采样（Negative Downsampling）下的保序性与先验反解校正公式
在广告和信息流精排中，海量未点击曝光负样本极度消耗算力，通常进行负样本下采样（采样率 $w \in (0, 1)$，如 $w=0.1$）：
- **AUC 的不变性原理**：AUC 衡量的是正例打分高于随机负例的期望。均匀随机下采样不会改变负样本内部得分的概率分布形状，正负例对的大小关系数学期望保持不变。因此 **均匀负采样下 AUC 是全量真实 AUC 的渐近无偏估计**。
- **概率失真与 Odds 反解校正**：
  - 下采样导致训练集中正样本比例被人为放大了 $\frac{1}{w}$ 倍。模型直接输出的打分 $p_{\text{sampled}}$ 会严重偏高。
  - 若要还原线上真实的物理点击概率 $p_{\text{real}}$，必须使用 **Odds Ratio（几率比）反解公式**：
    $$
    \text{Odds}_{\text{real}} = \text{Odds}_{\text{sampled}} \cdot w \implies \frac{p_{\text{real}}}{1 - p_{\text{real}}} = \frac{p_{\text{sampled}}}{1 - p_{\text{sampled}}} \cdot w
    $$
    解得：
    $$
    p_{\text{real}} = \frac{p_{\text{sampled}}}{p_{\text{sampled}} + \frac{1 - p_{\text{sampled}}}{w}}
    $$
  - 或者在训练时，直接对保留下来的负样本赋予权重 $\frac{1}{w}$ 进行加权交叉熵训练。

### 9.7 训练和评价的错位

常见错位：

- 用 BCE 训练 CTR，却用 NDCG 评价排序；
- 用曝光日志训练，却在人工构造候选集上测试；
- 训练候选来自旧模型，线上候选来自新召回；
- 搜索标注只看相关性，线上排序还混入质量和时效；
- 推荐离线只留一个正例，系统实际要生成多兴趣列表。

代理目标可以与最终指标不同，但实验记录要说明这段差距。候选集、标签口径或流量分布发生变化时，离线增益不能直接外推到线上。

### Quick Coding：NDCG@K

输入按预测顺序排列的分级相关性，计算 DCG、IDCG 和 NDCG。边界条件包括 `k <= 0`、列表短于 `k`，以及所有相关性都为零。

实现：

```python
def ndcg_at_k(relevances, k):
    ...
```

`relevances` 已按模型预测顺序排列，每个值是非负相关性等级。使用：

$$
\operatorname{DCG@K} = \sum_{i=1}^{K}\frac{2^{\operatorname{rel}_i}-1}{\log_2(i+1)}
$$

无相关结果时返回 `0.0`，`k <= 0` 时也返回 `0.0`。

<details>
<summary>参考答案</summary>

```python
from math import log2


def ndcg_at_k(relevances, k):
    if k <= 0:
        return 0.0

    all_values = list(relevances)

    def dcg(items):
        return sum(
            (2 ** rel - 1) / log2(rank + 1)
            for rank, rel in enumerate(items, start=1)
        )

    actual = dcg(all_values[:k])
    ideal = dcg(sorted(all_values, reverse=True)[:k])
    return 0.0 if ideal == 0 else actual / ideal
```

排序 ideal list 需要 `O(n log n)`；若相关性等级范围很小，可以用计数把它降到线性。

</details>

### Quick Coding：计算 GAUC (Group AUC)

根据日志样本计算曝光加权分组 AUC。处理单用户全正/全负边界条件。

```python
def calculate_gauc(user_ids, labels, preds):
    ...
```

<details>
<summary>参考答案</summary>

```python
from collections import defaultdict


def calculate_gauc(user_ids, labels, preds):
    # 1. 按 user 分组归集 (label, pred)
    user_data = defaultdict(lambda: ([], []))
    for u, y, p in zip(user_ids, labels, preds):
        user_data[u][0].append(y)
        user_data[u][1].append(p)

    total_weight = 0
    weighted_auc_sum = 0.0

    for u, (u_labels, u_preds) in user_data.items():
        n_pos = sum(u_labels)
        n_neg = len(u_labels) - n_pos

        # 边界条件：若该用户全为正例或全为负例，AUC 无定义，过滤跳过
        if n_pos == 0 or n_neg == 0:
            continue

        # 基于 Wilcoxon-Mann-Whitney 秩和快速计算单用户局部 AUC
        ranked = sorted(zip(u_preds, u_labels), key=lambda x: x[0])
        rank_sum = 0
        for rank, (_, y) in enumerate(ranked, start=1):
            if y == 1:
                rank_sum += rank

        auc_u = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

        weight = len(u_labels)  # 以单用户曝光数作为加权依据
        weighted_auc_sum += auc_u * weight
        total_weight += weight

    return weighted_auc_sum / total_weight if total_weight > 0 else 0.5
```

时间复杂度为 $\sum O(N_u \log N_u) \le O(N \log N)$，空间复杂度为 $O(N)$。

</details>

### 9.8 本章自测

1. Pointwise 分数能校准，为什么排序仍可能不好？
2. 为什么在工业推荐中必须看 GAUC 而不是单纯看全局 AUC？
3. Pairwise 的 hard negative 应该怎样产生？
4. NDCG 为什么对列表头部更敏感？
5. Cross-BERT 和双塔 BERT 应放在哪一层？
6. 相关性、内容质量和最终排序分数应该怎样解耦？
7. 搜索融合为什么常从"相关性分档 + 档内规则"开始，之后才换融合模型？
8. 为什么大厂 CTR 模型通常以 NE (Normalized Cross Entropy / RIG) 为核心优化指标，而不是单纯看 LogLoss？
9. 在极度稀疏的 CVR（如大额电商成单）或低频风控拦截任务中，为什么 ROC-AUC 会出现“虚假繁荣”？此时应看什么？
10. 如果离线训练对负样本进行了采样率 $w=0.1$ 的随机降采样，预估分送入广告 eCPM 出价模块前必须做什么数学处理？

<details>
<summary>参考答案</summary>

1. 校准只保证同一分数对应的平均概率接近真实频率，不保证相近候选的相对次序正确；特征不足或损失与 NDCG 不一致时仍会排错。
2. 全局 AUC 跨用户混合比对正负例。高活用户底色 CTR 高、低活用户底色 CTR 低，模型若仅拟合人群先验，全局 AUC 很高，但单个用户内部排序全错。线上推荐是单用户单屏展现，GAUC 消除人群先验偏置，真实度量单用户内的个性化调序能力。
3. 从同一次请求中选择旧模型排得高但标签为负的曝光候选，或从 ANN/BM25 top 结果中采样。还要过滤假负例和未成熟标签。
4. 它使用对数折扣，越靠前的位置权重越大；高相关候选从第 1 位跌到第 2 位的损失高于尾部相同位移。
5. 双塔 BERT 适合大规模语义召回或粗排；Cross-BERT 需要联合编码 query-document，只适合候选较少的精排或重排。
6. 分别产出相关性、质量和业务目标分数，先校准再按场景融合，并保留可解释的硬护栏。不要让一个总分同时承担所有含义。
7. 规则容易解释和快速纠错，适合数据少、链路还在频繁变化的阶段。融合模型能利用更多交叉关系，但需要可靠的综合满意度与行为标签，还要用相关性分档监控防止它用不相关结果换点击。
8. 绝对 LogLoss 极易受大盘自然转化率波动（如大促、节假日、昼夜流量切换）影响，使得研发无法区分“Loss 上涨是模型退化还是流量结构改变”。NE 除以了背景平均香农熵，衡量模型相比“常数先验基准”压缩的不确定性比例，具备严格跨流量切片与跨周期的鲁棒可比性。
9. 极度不平衡时负例基数 TN 极大，会极度稀释假阳率公式 $\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$ 的分母。即便产生海量误报（FP 远超 TP），FPR 依然趋近于 0，使 ROC 曲线紧贴左上角呈现虚假繁荣。此时必须看 **PR-AUC（Average Precision）**，并与正例先验基线 $\pi = P(Y=1)$ 对比。
10. 负采样保持了 AUC 的单调性，但破坏了绝对物理概率与校准度，使模型输出打分 $p_{\text{sampled}}$ 严重虚高。送入出价前必须通过几率比反解公式还原真实点击率：$p_{\text{real}} = \frac{p_{\text{sampled}}}{p_{\text{sampled}} + \frac{1 - p_{\text{sampled}}}{w}}$，或者在训练时将负样本赋予 $\frac{1}{w}$ 的加权损失。

</details>

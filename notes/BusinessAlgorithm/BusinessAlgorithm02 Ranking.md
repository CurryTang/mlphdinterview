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

```math
\mathcal L_{\text{point}}
=-\left[ y\log p+(1-y)\log(1-p) \right].
```

优点是样本和训练都简单，预测概率还能做校准。缺点是它没有直接表达同一个 query/user 下候选之间的顺序。

CTR、CVR 和时长预估多从 pointwise 开始。搜索相关性若有分档，也可做多分类或回归。

### 9.3 Pairwise

Pairwise 构造正负候选对，希望正例分数更高：

```math
\mathcal L_{\text{pair}}
=-\log \sigma(s^+-s^-).
```

它更接近"谁排在谁前面"，但 pair 数量可能爆炸。负例怎么选会明显改变训练：随机负例简单，当前模型排错的 hard negative 更有信息，也更容易包含标注噪声。

RankNet 属于 pairwise。LambdaRank/LambdaMART 会根据交换两个候选对 NDCG 的影响调整梯度权重，让列表头部错误更受重视。

### 9.4 Listwise

Listwise 把整个候选列表作为训练对象。一个简单形式是对列表做 softmax：

```math
P(i\mid q)=\frac{e^{s_i}}{\sum_j e^{s_j}},
```

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

### 9.6 排序效果怎样衡量

CTR、CVR 等 pointwise 预估先看 LogLoss：

```math
\operatorname{LogLoss}
=-\frac{1}{N}\sum_i
\left[ y_i\log p_i+(1-y_i)\log(1-p_i) \right].
```

它关心概率本身与绝对校准质量。

#### 全局 AUC (Global AUC) 及其在推荐中的缺陷
AUC 衡量随机抽取一个正样本其预估得分高于随机抽取一个负样本的概率：
$$\operatorname{AUC} = \frac{\sum_{i \in \mathcal{D}^+} \sum_{j \in \mathcal{D}^-} \left[ \mathbb{I}(p_i > p_j) + 0.5 \times \mathbb{I}(p_i = p_j) \right]}{|\mathcal{D}^+| \times |\mathcal{D}^-|}$$

**为什么全局 AUC 会误导推荐排序？**
全局 AUC 在全量测试集上混合跨用户比对正负样本。如果活跃用户 A 偏好点击（底色 CTR 30%），低活用户 B 极少点击（底色 CTR 1%），若模型单纯拟合了用户群体先验，给用户 A 的所有物品都预测高分、给用户 B 的所有物品都预测低分：
- 跨用户比对时，用户 A 的正例会压制用户 B 的负例，**全局 AUC 可能高达 0.85+**；
- 但在实际线上服务中，**系统永远是在单用户单次 Session 内对专属候选集排序**，不同用户的候选永远不会在同一屏竞争；
- 若模型在用户 A 内部或用户 B 内部的排序完全随机（$\text{AUC}_A = 0.5, \text{AUC}_B = 0.5$），线上推荐给单用户的全是不相关结果，用户体验完全崩溃。

#### 分组 AUC (Group AUC, GAUC) —— 工业界核心离线指标
为了消除用户先验偏差、真实衡量模型在单个用户内的个性化调序能力，工业界推荐系统（如字节跳动、Meta、阿里）统一采用 **GAUC (Group AUC)**：

```math
\operatorname{GAUC}
=\frac{\sum_{u \in \mathcal{U}} w_u \times \operatorname{AUC}_u}{\sum_{u \in \mathcal{U}} w_u}
```

其中：
- $\operatorname{AUC}_u$ 表示模型仅在用户 $u$ 自己的曝光集合内计算的局部 AUC；
- 权重 $w_u$ 通常取用户 $u$ 的曝光展现量（$\text{impressions}_u$）或点击量；
- **边界过滤条件**：若用户 $u$ 在测试样本中**全为正样本（全点）**或**全为负样本（全未点）**，此时正负例对数为 0，$\operatorname{AUC}_u$ 无定义，累加时必须显式跳过该用户。

**工业界经验法则**：在召回和粗排不变的前提下，精排模型的 $\Delta\operatorname{GAUC} \ge +0.003$（千分之三）通常可稳定支撑在线 A/B 实验产生具有统计显著性的真实 CTR、完播率或留存正向收益。

搜索和推荐列表还常看 DCG/NDCG：

```math
\operatorname{DCG@K}
=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)},
```

```math
\operatorname{NDCG@K}
=\frac{\operatorname{DCG@K}}{\operatorname{IDCG@K}}.
```

位置越靠前折损越小，高相关候选排错的代价更大。搜索的 relevance grade 可以来自人工标注；推荐常把点击、购买或行为强度映射成等级，要说明这种映射代表什么。

若任务主要关心第一个正确结果，可看 MRR：

```math
\operatorname{MRR}
=\frac{1}{|Q|}\sum_{q\in Q}\frac{1}{rank_q}.
```

这些指标要在同一候选集上比较。召回集合变了，NDCG 的变化可能来自候选变好，也可能来自排序模型本身，二者要分开做实验。

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

```math
DCG@K=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}.
```

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

<details>
<summary>参考答案</summary>

1. 校准只保证同一分数对应的平均概率接近真实频率，不保证相近候选的相对次序正确；特征不足或损失与 NDCG 不一致时仍会排错。
2. 全局 AUC 跨用户混合比对正负例。高活用户底色 CTR 高、低活用户底色 CTR 低，模型若仅拟合人群先验，全局 AUC 很高，但单个用户内部排序全错。线上推荐是单用户单屏展现，GAUC 消除人群先验偏置，真实度量单用户内的个性化调序能力。
3. 从同一次请求中选择旧模型排得高但标签为负的曝光候选，或从 ANN/BM25 top 结果中采样。还要过滤假负例和未成熟标签。
4. 它使用对数折扣，越靠前的位置权重越大；高相关候选从第 1 位跌到第 2 位的损失高于尾部相同位移。
5. 双塔 BERT 适合大规模语义召回或粗排；Cross-BERT 需要联合编码 query-document，只适合候选较少的精排或重排。
6. 分别产出相关性、质量和业务目标分数，先校准再按场景融合，并保留可解释的硬护栏。不要让一个总分同时承担所有含义。
7. 规则容易解释和快速纠错，适合数据少、链路还在频繁变化的阶段。融合模型能利用更多交叉关系，但需要可靠的综合满意度与行为标签，还要用相关性分档监控防止它用不相关结果换点击。

</details>

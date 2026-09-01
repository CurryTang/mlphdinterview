# 稀疏检索与协同召回

## 第 3 章 稀疏检索与协同召回

召回不负责把最终第一名选出来。它要在固定延迟和候选预算内，从全库捞出一批"后面可能用得上"的内容。此时漏掉正例比候选顺序不够准更麻烦：排序模型可以调整已召回候选的位置，却无法给一个从未进入候选集的 item 打分。

因此召回输出不应只有 `item_id`。至少还要带通道、原始分数和触发它的 seed，后级才能做融合，离线也才能回答"这个候选为什么会出现"。单路召回的价值不是独立准确率最高，而是在总候选数不变时，能补回多少其他通道没有找到的正例。

本章先看两类不依赖重型在线交互的证据：搜索中的词项关系，以及推荐中的行为共现。它们便宜、稳定，也有共同的硬边界：词没有出现在 posting 中，或用户与物品之间没有形成共现边，系统就看不到这条关系。第 4 章的向量召回正是用连续表示补这部分空白。

### 3.1 两类稀疏证据

搜索里的词项共现和推荐里的行为共现看起来不一样，计算习惯却很接近。

搜索问："哪些文档出现了 query 中的重要词？"

推荐问："哪些物品与用户交互过的物品被相似人群共同消费？"

二者都会提前建立倒排关系，线上从少量 key 找到一批候选。区别在于 key 的语义：搜索用 term，ItemCF 用 item，UserCF 用相似用户。

### 3.2 倒排索引

正排索引是：

```text
doc_id -> 文档内容
```

倒排索引是：

```text
term -> [(doc_id, tf, positions), ...]
```

`tf` 是词在文档中的出现次数，`positions` 记录位置，可用于词距和短语匹配。线上查询先分词，再取多个 posting list 做交、并或更复杂的遍历。

倒排索引强在：

- 精确词匹配稳定，可解释；
- 索引压缩和跳表技术成熟；
- 文档增量更新相对直接；
- 对实体名、型号、代码等精确 query 很可靠。

它的短板也直白：字面不重合时很难召回。"LV 包"和"LOUIS VUITTON 包包"语义接近，纯词项检索未必知道。

线上倒排索引最常见的问题往往不在公式。分词器、词典和索引必须来自兼容版本；服务把 `"iPhone15"` 切成一个词，旧索引却按 `"iPhone" + "15"` 建 posting，会直接造成空召回。高频词的 posting 可能长到无法完整遍历，需要跳表、WAND/Block-Max WAND 或更严格的 term 组合提前剪枝。文档删除通常先写 tombstone，再由 segment merge 回收空间；只删正排内容却没让倒排失效，会继续召回已经下架的文档。

AND 与 OR 也不是实现细节。AND 精度高，却可能让长 query 一个候选都没有；OR 覆盖高，却会引入只命中弱修饰词的文档。成熟系统会根据核心词、实体和召回量动态选择，并在日志里记录实际执行的 query plan。

### Quick Coding：倒排索引

给定已经分词的若干文档，构建 `term -> posting list`。每条 posting 需要保存 `doc_id`、词频和位置，并按文档 ID 排序。这道题能顺手检查你是否真的理解 TF、词距和短语匹配需要哪些数据。

输入是已经分词的文档：

```python
documents = {
    1: ["deep", "learning", "deep"],
    2: ["learning", "system"],
}
```

实现：

```python
def build_inverted_index(documents):
    ...
```

输出每个 term 的 posting list，每条 posting 为：

```text
(doc_id, term_frequency, zero_based_positions)
```

例如 `deep` 对应 `[(1, 2, [0, 2])]`。posting 按 `doc_id` 升序。

<details>
<summary>参考答案</summary>

```python
from collections import defaultdict


def build_inverted_index(documents):
    index = defaultdict(list)

    for doc_id in sorted(documents):
        positions = defaultdict(list)
        for position, term in enumerate(documents[doc_id]):
            positions[term].append(position)

        for term, term_positions in positions.items():
            index[term].append(
                (doc_id, len(term_positions), term_positions)
            )

    return dict(index)
```

若总 token 数是 `T`，时间复杂度为 `O(T + D log D)`，其中 `D` 是文档数；索引空间为 `O(T)`。

</details>

### 3.3 TF-IDF 与 BM25

TF-IDF 同时考虑词在当前文档中的频率和全库稀有度：

```math
\operatorname{tfidf}(t,d)
=\operatorname{tf}(t,d)
\log\frac{N}{df(t)}.
```

常见词区分度低，IDF 会给它较小权重。BM25 在此基础上加入词频饱和与文档长度归一化：

```math
\operatorname{BM25}(q,d)
=\sum_{t\in q}
\operatorname{IDF}(t)
\frac{tf(t,d)(k_1+1)}
{tf(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
```

词出现 20 次不该比出现 10 次重要一倍，这就是饱和项的作用。长文档天然更容易命中词，长度归一化负责修正。

BM25 快、稳定、可解释。评估稠密召回或生成式检索时，先与它比较；只和较弱的神经基线比较，很难判断新增复杂度是否值得。

文档字段通常不会同权。标题命中、正文命中、anchor 命中和结构化属性命中可分别计算 BM25，再用 BM25F 或线性融合组合。标题权重过高会奖励关键词堆砌，正文权重过高又会让长文档占便宜，所以字段权重需要在人工相关性切片上调，而不是只看点击。

BM25 分数还不适合直接跨 query 比较。一个包含多个稀有词的 query 天然可能得到更大分数；阈值、缓存或多路融合若需要跨请求口径，应做 query 内排序、分桶归一化或校准。

### 3.4 ItemCF

ItemCF 先根据用户-物品交互计算物品相似度，再从用户历史扩展候选。

最朴素的共现相似度：

```math
\operatorname{sim}(i,j)
=\frac{|U_i\cap U_j|}
{\sqrt{|U_i||U_j|}}.
```

`U_i` 是交互过物品 `i` 的用户集合。分母抑制热门物品。

用户 `u` 对候选 `i` 的分数可以写成：

```math
\operatorname{score}(u,i)
=\sum_{j\in H_u}
w(u,j)\operatorname{sim}(j,i),
```

`H_u` 是用户历史，`w(u,j)` 可加入行为类型、时间衰减和观看深度。

共现表的质量先取决于哪些行为被写进去。一次有效观看、主动收藏和购买可以使用不同权重；自动连播、误触和机器人流量应降权或剔除。两个 item 在同一短 session 内连续出现，通常比同一用户半年内各点过一次更能说明局部关联。计算相似表时还会限制用户活跃度和单 session 长度，避免超级活跃用户给任意 item pair 都制造一条边。

完整工程流程：

1. 离线统计用户到物品的行为表；
2. 生成物品共现；
3. 计算并截断每个物品的 top 相似物品；
4. 线上读取用户历史；
5. 查相似表、聚合、过滤已曝光内容；
6. 输出 top 候选。

ItemCF 的好处是解释容易、服务便宜。坏处是新品没有共现，热门物品容易占优势，兴趣跨类目时也可能扩不出去。

### Quick Coding：ItemCF

从 `user -> interacted items` 的小型数据集出发，计算共现余弦相似度，为指定用户召回未交互物品。要求过滤历史、稳定处理同分候选，并说明为什么这份面试代码不能直接拿去服务全量流量。

实现：

```python
def item_cf_recommend(user_items, target_user, k):
    ...
```

使用物品共现余弦相似度：

```math
sim(i,j)=\frac{|U_i\cap U_j|}{\sqrt{|U_i||U_j|}}.
```

候选分数是目标用户已交互物品与候选物品相似度之和。过滤已经交互过的物品，按 `score` 降序、`item_id` 升序返回前 `k` 个 `(item_id, score)`。

<details>
<summary>参考答案</summary>

```python
from collections import defaultdict
from math import sqrt


def item_cf_recommend(user_items, target_user, k):
    item_users = defaultdict(set)
    for user, items in user_items.items():
        for item in set(items):
            item_users[item].add(user)

    seen = set(user_items.get(target_user, []))
    scores = []

    for candidate, candidate_users in item_users.items():
        if candidate in seen:
            continue

        score = 0.0
        for item in seen:
            users = item_users.get(item, set())
            if users and candidate_users:
                score += len(users & candidate_users) / sqrt(
                    len(users) * len(candidate_users)
                )
        if score > 0:
            scores.append((candidate, score))

    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return scores[:k]
```

这是面试用直接实现。线上系统会离线生成 item-to-item top 相似表，避免请求时扫描全部物品。

</details>

### 3.5 UserCF 与 Swing

UserCF 先找相似用户，再召回他们喜欢而当前用户未见过的物品。它在用户兴趣较稳定、用户数没大到无法维护相似表时有用。大规模内容平台更常使用 ItemCF，因为物品相似关系通常比用户相似关系稳定。

Swing 针对共现中的"小圈子噪声"。如果两个物品只被一小群高度重合的用户共同交互，普通共现可能把它们判得过分相似。Swing 对用户对的共同物品数做惩罚，大意是：

```math
\operatorname{sim}(i,j)
=\sum_{u,v\in U_i\cap U_j}
\frac{1}{\alpha+|I_u\cap I_v|}.
```

共同兴趣极多的用户对贡献会下降。公式不是重点，重点是它修正了哪种统计偏差。

### 3.6 矩阵分解

把用户-物品矩阵 `R` 近似为两个低秩矩阵：

```math
R\approx UV^\top,
\qquad
\hat r_{ui}=u_u^\top v_i.
```

矩阵分解把离散共现压进连续向量。它比简单 ItemCF 更容易表达隐含兴趣，但仍主要依赖交互，冷启动问题没有凭空消失。

隐式反馈通常使用带置信度的目标，而不是把未点击全部当明确负例。训练时也要注意负采样，否则海量零项会淹没正样本。

一种 weighted matrix factorization 写法是：

```math
\min_{U,V}
\sum_{u,i} c_{ui}
\left( p_{ui}-u_u^\top v_i \right)^2
\lambda(\|U\|_F^2+\|V\|_F^2),
```

其中 `p_ui` 只表示是否观察到偏好，`c_ui` 表示置信度。未交互项仍参与目标，但权重低于明确正反馈，不能把"没有观察到"解释成同等强度的厌恶。

线上可以用用户向量对物品向量做 ANN，也可以预先生成 top item。它与双塔都使用低维内积，差别在输入：经典矩阵分解主要学习 ID 与交互，加入文本、图像或上下文不自然；双塔从特征生成向量，因此对新品和跨场景迁移更方便。

### 3.7 什么时候用哪一个

| 方法 | 强项 | 明显短板 |
| --- | --- | --- |
| 倒排索引/BM25 | 精确文本、成熟高效、易更新 | 语义不重合 |
| ItemCF | 便宜、可解释、行为相关性强 | 冷启动、热门偏置 |
| UserCF | 借相似人群发现候选 | 用户关系不稳定、规模大 |
| Swing | 抑制小圈子和过强共现 | 计算与参数更复杂 |
| 矩阵分解 | 连续隐空间、泛化较好 | 仍依赖行为、特征利用有限 |

线上系统通常会保留多种通道。传统方法即使单路指标不占优，也可能找到深度模型漏掉的候选。

### 3.8 本章自测

1. 倒排索引为什么要保存词位置？
2. BM25 的饱和项与长度归一化各解决什么？
3. ItemCF 线上服务需要哪些预计算表？
4. Swing 想修正普通共现的哪种问题？
5. 矩阵分解和双塔有什么联系，又有什么区别？

<details>
<summary>参考答案</summary>

1. 位置支持 phrase query、邻近匹配和高亮；只保存文档 ID 与词频无法判断多个词是否按顺序相邻出现。
2. 饱和项避免词频线性增长压倒其他信号；长度归一化避免长文档仅因包含更多词而获得过高分数。
3. 至少需要 item-to-item top 相似表、用户近期交互列表和 item 可用状态。相似表通常离线或准实时构建，线上只做查表与聚合。
4. 普通共现容易被活跃用户和热门物品制造的偶然重叠放大。Swing 会降低被大量用户共同连接的 item pair 和过度相似用户对的权重。
5. 两者都学习低维 user/item 表示并用内积打分。矩阵分解主要从交互矩阵学习 ID 向量；双塔还能接入内容和上下文特征，并支持对新 item 生成表示。

</details>

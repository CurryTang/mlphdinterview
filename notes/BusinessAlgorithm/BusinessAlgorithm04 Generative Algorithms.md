# 生成式检索与 Semantic ID

## 第 17 章 生成式检索与 Semantic ID

### 17.1 从"算相似度"到"生成标识"

传统稠密检索做两件事：

1. 把 query 和文档编码到同一向量空间；
2. 用 ANN 找最近邻。

生成式检索换了一种形式。给定 query，模型自回归生成目标文档的标识：

```math
P(d\mid q)
=\prod_{t=1}^{L}
P(d_t\mid d_{<t},q).
```

`d_1...d_L` 是文档或物品的离散 token 序列。beam search 产生多个标识，映射回候选。

这种写法允许索引标识、匹配目标和排序一起训练，beam search 直接给出 top-k。代价转移到了别处：标识设计、语料更新、解码延迟和无效 ID 都要由系统处理。

### 17.2 DSI

[DSI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html)（Tay et al., NeurIPS 2022）把 Transformer 当成可微搜索索引。模型先通过文档文本学习"文档 -> docid"，再通过 query 学习"query -> docid"。

若 docid 是无结构的随机整数，模型要记住大量任意映射。若 docid 带语义或层级，相近文档可共享前缀，解码更容易泛化。

DSI 测试了把部分外部索引写进模型参数的可能性。它没有解决动态大语料的维护问题：新增、删除和纠错比更新外部索引麻烦，模型容量还要承担语料记忆。

### 17.3 NCI

[NCI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a46156bd3579c3b268108ea6aca71d13-Abstract-Conference.html)（Wang et al., NeurIPS 2022）进一步使用语义文档标识。常见构造方式是对文档向量做层次聚类：

```text
文档
  -> 一级簇 token
  -> 二级子簇 token
  -> ...
  -> 叶子标识
```

前缀代表粗语义，后缀逐步定位具体文档。解码器使用 prefix-aware 结构，训练还加入自动生成 query 和一致性正则。

树状 ID 把全库选择拆成多步小分类，适合 beam search，也允许同前缀文档共享训练信号。"语义更好"只是其中一面。

### 17.4 SEAL

[SEAL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cd88d62a2063fdaf7ce6f9068fb15dcd-Abstract-Conference.html)（Bevilacqua et al., NeurIPS 2022）不为每篇文档发一个任意编号，而是生成文档中真实出现的 n-gram。生成出的 n-gram 再通过 FM-index 映射回包含它的文档。

SEAL 使用约束解码：下一 token 必须能在语料中继续形成合法 n-gram，无效路径直接 mask。模型生成有区分度的文本标识，传统索引验证标识存在并完成定位。

SEAL 说明生成式检索不一定要把外部索引彻底删掉。生成模型和经典数据结构可以协作，这个思路比"一切都放进参数"更接近可维护系统。

### 17.5 Semantic ID

推荐物品数量可能上亿。直接把每个 item_id 当独立 token，词表巨大，新物品没有语义。Semantic ID 先把连续内容向量离散化为一串 code。

设物品内容向量为 `e_i`，残差量化逐层选择 codeword：

```math
r_i^{(0)}=e_i,
```

```math
c_i^{(l)}
=\arg\min_{c\in\mathcal C_l}
\|r_i^{(l-1)}-c\|_2^2,
```

```math
r_i^{(l)}
=r_i^{(l-1)}-c_i^{(l)}.
```

最终：

```text
SID(i) = [code_1, code_2, ..., code_L]
```

前几个 code 表示粗语义，后面的 code 补残差。若每层有 `K` 个 code，长度 `L` 的组合空间可达 `K^L`，词表本身却只需每层 `K` 个 token。

这和普通 item embedding 的关系是：

- embedding 是连续向量，用于相似度或作为模型输入；
- Semantic ID 是离散 token 序列，用于自回归生成；
- Semantic ID 往往由 embedding 量化得到，但二者不等价。

### Quick Coding：残差量化

给定一个向量和多层 codebook，每层选择离当前 residual 最近的 codeword，再更新 residual。返回 codeword 下标序列和最终残差。这个小题正好对应 Semantic ID 生成的最小骨架。

题目：[[BusinessAlgorithm09 Quick Coding.md#QC08 Semantic ID 的残差量化|QC08 Semantic ID 的残差量化]]。

### 17.6 TIGER

[TIGER](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)（Rajput et al., NeurIPS 2023）把 Semantic ID 用到序列推荐：

1. 用预训练文本 encoder 得到 item 内容向量；
2. 用残差量化生成 Semantic ID；
3. 把用户历史物品改写成 Semantic ID 序列；
4. Transformer 生成下一物品的 Semantic ID；
5. beam search 得到 top-k 候选。

语义相近物品共享部分 code，新物品即使没有行为，也能凭内容获得有意义的 ID。这解释了论文中冷启动和长尾上的改善来源。

但要小心一件事：如果两个物品量化到同一 SID，生成正确 SID 不等于找到了唯一物品。工程上需要碰撞处理，例如额外叶子 token、后置候选消歧或保证编码器的唯一映射。

### 17.7 搜索与推荐能否共享 Semantic ID

2025 年 Spotify 等作者研究了[联合生成式搜索与推荐的 Semantic ID](https://arxiv.org/abs/2508.10478)。搜索 embedding 学 query-item 匹配，推荐 embedding 学 item-item 行为共现。分别量化会得到两套 token 空间，共享量化又可能牺牲单任务效果。

这类工作讨论的是：

```text
内容语义 + 搜索匹配 + 协同信号
             ↓
       一套可生成的离散物品语言
```

目前更适合作为前沿方向，而不是默认工程方案。搜索和推荐的训练分布、目标和更新频率都不同，共享 ID 需要证明确实带来系统级收益。

### 17.8 生成式检索的工程账单

| 风险 | 上线前要验证什么 |
| --- | --- |
| 无效解码 | trie/FM-index 约束是否覆盖全部合法 ID；后置校验失败时怎样回填 |
| ID 碰撞 | 一个 SID 对应几个 item；离线评估按 SID 还是按真实 item 计分 |
| 目录更新 | 新 item 如何分配 ID；旧 ID 会不会变化；多久增量训练一次 |
| beam search | top-k 的 P95/P99 延迟；层数、beam width 和 Recall 的关系 |
| 热门偏置 | 高频前缀是否压住长尾；采样、重加权或校准是否有效 |
| 排障与回滚 | 能否回放每步 token 概率；传统召回通道能否独立接管流量 |

### 17.9 与双塔怎么比较

| 维度 | 双塔召回 | 生成式召回 |
| --- | --- | --- |
| 表示 | 连续向量 | 离散 ID 序列 |
| 训练 | 对比学习 | 序列生成 |
| 在线计算 | query tower + ANN | 自回归解码 + 合法路径约束 |
| 新 item | 生成 embedding 并写入索引 | 分配 ID，必要时更新模型 |
| 冷启动 | 取决于内容 encoder | 取决于内容 encoder 和 codebook |
| 排障 | 检查近邻、索引和过滤 | 检查 token 概率、beam 与物化结果 |

两者可以并行召回，也可以互为 teacher。是否替换现有通道，要看固定延迟预算下的增量 Recall，而不是只看离线全量结果。

### 17.10 本章自测

1. DSI、NCI 和 SEAL 生成的 document identifier 有什么不同？
2. Semantic ID 和普通 item embedding 是什么关系？
3. 残差量化为什么可能产生 ID 碰撞，怎样处理？
4. 生成式检索为什么需要约束解码？
5. 怎样在同一延迟预算下比较双塔和生成式召回？

<details>
<summary>参考答案</summary>

1. DSI 直接生成预先分配的文档 ID；NCI 使用结构化的层级 ID；SEAL 生成文档中可检索的 n-gram，再通过 FM-index 约束到合法文档。
2. embedding 是连续向量，用于相似度计算或模型输入；Semantic ID 是离散 token 序列，通常由 embedding 量化得到，但不能互换。
3. 多个相近向量可能选择相同 code 序列。可增加叶子 token、扩大或重训 codebook、后置物化多个候选再消歧，或显式保证唯一映射。
4. 自回归模型可能输出不存在的 ID 或无效前缀。trie、FM-index 等约束能把每一步限制在合法路径，并减少无效 beam。
5. 固定 P95/P99、候选数和硬件，比较增量 Recall、长尾覆盖、索引/目录更新成本与失败率；不能拿全量离线生成结果对比受延迟限制的 ANN。

</details>

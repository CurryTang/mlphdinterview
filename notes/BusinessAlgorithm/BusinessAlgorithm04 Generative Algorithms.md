# 第五部分：生成式方法到底改了什么

## 第 12 章 生成式检索与 Semantic ID

### 12.1 从"算相似度"到"生成标识"

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

它吸引人的地方很明确：索引、匹配和排序可以共同训练；模型能共享不同文档前缀上的统计；解码本身就产生 top-k。麻烦也同样明确：标识怎样设计，语料更新怎样处理，beam search 是否够快，生成不存在的 ID 怎么办。

### 12.2 DSI

[DSI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html)（Tay et al., NeurIPS 2022）把 Transformer 当成可微搜索索引。模型先通过文档文本学习"文档 -> docid"，再通过 query 学习"query -> docid"。

若 docid 是无结构的随机整数，模型要记住大量任意映射。若 docid 带语义或层级，相近文档可共享前缀，解码更容易泛化。

DSI 提出了一个很大胆的问题：传统外部索引能否部分写进模型参数？答案并不是"已经可以完全替代倒排和 ANN"。对动态大语料，新增、删除和纠错都比更新外部索引麻烦；模型容量也要承担语料记忆。

### 12.3 NCI

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

### 12.4 SEAL

[SEAL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cd88d62a2063fdaf7ce6f9068fb15dcd-Abstract-Conference.html)（Bevilacqua et al., NeurIPS 2022）不为每篇文档发一个任意编号，而是生成文档中真实出现的 n-gram。生成出的 n-gram 再通过 FM-index 映射回包含它的文档。

这里最值得记的是约束解码。下一 token 必须能在语料中继续形成合法 n-gram，无效路径直接 mask。模型负责生成有区分度的文本标识，传统索引负责保证它确实存在并快速定位。

SEAL 说明生成式检索不一定要把外部索引彻底删掉。生成模型和经典数据结构可以协作，这个思路比"一切都放进参数"更接近可维护系统。

### 12.5 Semantic ID

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

这也回答了一个常见问题：推荐里的 Semantic ID 与一致性哈希没有直接关系。一致性哈希解决节点变化时的数据分片迁移；Semantic ID 解决物品如何被生成模型表示。它们都使用"ID"或哈希式映射的语言，目标完全不同。

### Quick Coding：残差量化

给定一个向量和多层 codebook，每层选择离当前 residual 最近的 codeword，再更新 residual。返回 codeword 下标序列和最终残差。这个小题正好对应 Semantic ID 生成的最小骨架。

题目与参考答案：[[BusinessAlgorithm06 Quick Coding.md#QC08 Semantic ID 的残差量化|QC08 Semantic ID 的残差量化]]。

### 12.6 TIGER

[TIGER](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)（Rajput et al., NeurIPS 2023）把 Semantic ID 用到序列推荐：

1. 用预训练文本 encoder 得到 item 内容向量；
2. 用残差量化生成 Semantic ID；
3. 把用户历史物品改写成 Semantic ID 序列；
4. Transformer 生成下一物品的 Semantic ID；
5. beam search 得到 top-k 候选。

语义相近物品共享部分 code，新物品即使没有行为，也能凭内容获得有意义的 ID。这解释了论文中冷启动和长尾上的改善来源。

但要小心一件事：如果两个物品量化到同一 SID，生成正确 SID 不等于找到了唯一物品。工程上需要碰撞处理，例如额外叶子 token、后置候选消歧或保证编码器的唯一映射。

### 12.7 搜索与推荐能否共享 Semantic ID

2025 年 Spotify 等作者研究了[联合生成式搜索与推荐的 Semantic ID](https://arxiv.org/abs/2508.10478)。搜索 embedding 学 query-item 匹配，推荐 embedding 学 item-item 行为共现。分别量化会得到两套 token 空间，共享量化又可能牺牲单任务效果。

这类工作讨论的是：

```text
内容语义 + 搜索匹配 + 协同信号
             ↓
       一套可生成的离散物品语言
```

目前更适合作为前沿方向，而不是默认工程方案。搜索和推荐的训练分布、目标和更新频率都不同，共享 ID 需要证明确实带来系统级收益。

### 12.8 生成式检索的工程账单

必须检查以下问题。

无效解码：模型可能生成库中不存在的 code 序列，需要 trie/FM-index 约束或后置校验。

ID 碰撞：多个物品共享 SID 时，离线指标可能被错误计算，需要一对一映射或明确消歧。

目录更新：新文档加入后如何分配 ID，是否改变旧 ID，模型是否需要增量训练。

beam search 成本：top-k 解码是串行的，多层 code 会增加延迟。可用浅层树、并行多 token 预测、蒸馏或两阶段候选生成。

热门偏置：高频 item 的 token 前缀出现更多，模型更容易生成。需要采样、重加权或校准。

可解释和回滚：传统索引可直接检查 posting 或邻居，参数化索引更难定位漏召回原因。工业系统通常会保留传统通道作为保底与对照。

### 12.9 面试回答框架

被问"生成式召回和双塔哪个好"时，不要只比 Recall。

可以这样回答：

1. 表示：连续向量最近邻 vs 离散 ID 自回归；
2. 训练：对比学习 vs 序列生成；
3. 服务：ANN 查询 vs beam search；
4. 更新：索引增量写入 vs ID 分配和模型更新；
5. 冷启动：内容 encoder 与 Semantic ID 是否可泛化；
6. 排障：邻居可查 vs 解码路径可查；
7. 级联：两者是否可以并行召回或互为 teacher。

---

## 第 13 章 LLM 排序与生成式推荐

### 13.1 先分清三种任务

这三个系统都可能使用 decoder-only Transformer，目标却不同。

LLM reranker：输入 query 和已有候选，输出分数、偏好或排列。

生成式召回：输入 query/用户历史，输出 item/doc 的 ID。

生成式列表推荐：输入用户历史，一次生成一组有顺序的物品。

第一种不解决全库检索，第二种通常只产生候选，第三种试图把召回、排序和列表决策合并。

### 13.2 Pointwise LLM 排序

对每个 query-document 独立提问：

```text
Query: ...
Document: ...
Is the document relevant? yes/no
```

可以读取 `yes` token 的 logit 作为分数，而不是只解析自然语言输出。

优点是简单、可并行，缺点是候选之间不比较，分数还受 prompt 和 token 偏好影响。若每个候选都调用大模型，成本也很高。

### 13.3 Pairwise Ranking Prompting

给模型两个候选，让它判断哪个更相关：

```text
对于 query q，A 与 B 哪个更相关？
```

[PRP](https://aclanthology.org/2024.findings-naacl.97/)（Qin et al., Findings of NAACL 2024）说明，中等规模开源模型在 pairwise 形式下能取得很强效果。成对比较比一次理解整张排序表容易。

朴素 pairwise 要比较 `O(n^2)` 对。可用冒泡式 pass、锦标赛或局部比较把成本降下来。还应交换 A/B 位置重复询问，缓解位置偏差。

### 13.4 Listwise 与 RankGPT

Listwise prompt 给模型一组候选，要求输出：

```text
[4] > [1] > [3] > [2]
```

[RankGPT](https://aclanthology.org/2023.emnlp-main.923/)（Sun et al., EMNLP 2023）用滑动窗口处理长列表，并研究把大模型排列蒸馏到 440M 小模型。

Listwise 能直接比较多文档，问题有：

- 候选顺序影响结果；
- 长文档挤占上下文；
- 模型可能漏 ID、重复 ID 或输出非法格式；
- 滑动窗口只看到局部；
- 自回归生成完整排列慢。

测试时应随机打乱候选顺序，统计排序稳定性。单次 prompt 的高 NDCG 不能证明没有位置偏差。

### 13.5 FIRST

[FIRST](https://aclanthology.org/2024.emnlp-main.491/)（Gangi Reddy et al., EMNLP 2024）不再生成完整 ID 排列，而是读取第一个生成位置上各候选 ID 的 logits，用它们直接得到排序。论文同时加入 learning-to-rank loss，让高相关候选的错误更受惩罚。

这个设计很有代表性：使用生成模型的表示能力，不一定非要付出完整自回归解码成本。论文报告在保持效果的同时将推理加速约 50%，但实际收益仍取决于候选长度、模型和部署方式。

### 13.6 LLM 排序放在哪里

比较现实的几种用法：

- top-20/50 的末级重排；
- 难 query 路由到 LLM，普通 query 用小模型；
- 作为 teacher 生成软标签或候选排列；
- 为 cross-encoder 产生 hard negative；
- 离线标注复杂相关性；
- RAG 中筛选最终上下文。

全量高 QPS 搜索直接让大模型排 top-100，通常不划算。模型量化、KV cache、批处理和蒸馏能降成本，但级联仍然重要。

### 13.7 HSTU 与 Generative Recommenders

[HSTU](https://proceedings.mlr.press/v235/zhai24a.html)（Zhai et al., ICML 2024）把推荐表述为序列转导：输入用户动作序列，预测后续动作/内容。它针对推荐数据的高基数、非平稳和超长序列设计 HSTU，而不是直接照搬标准 Transformer。

论文报告：

- 公共和合成数据上最高 65.8% NDCG 相对提升；
- 长度 8192 上相对 FlashAttention2 Transformer 有 5.3 至 15.2 倍速度优势；
- 工业 1.5 万亿参数模型在多个场景部署，并报告 12.4% 的线上指标改善。

这些数字来自论文所述数据与平台，不能直接外推到别的业务。更值得记住的是它的建模方向：将大量异构推荐特征和行为序列整理成可扩展的序列模型，并观察到随计算量增加的 scaling 行为。

HSTU 的"generative"也不等于生成自然语言。它生成的是推荐序列中的目标事件或 item。

### 13.8 OneRec

[OneRec](https://arxiv.org/abs/2502.18965)（Deng et al., 2025 预印本）更激进地尝试统一召回与排序：

- encoder 读取用户历史；
- decoder 逐步生成一个 session 的视频列表；
- 分层离散 code 表示物品；
- sparse MoE 扩容量；
- reward model 和迭代 DPO 做偏好对齐。

传统系统逐项预估，再用规则拼列表。OneRec 直接学习：

```math
P(i_1,\ldots,i_m\mid H_u)
=\prod_{t=1}^{m}
P(i_t\mid i_{<t},H_u).
```

后一个物品以已生成列表为条件，因此模型有机会学习列表内互补和重复。预印本报告快手主场景 watch-time 上升 1.6%。

这是作者报告的工业结果，论文在本手册整理时仍按预印本处理。统一模型还要回答回滚、规则、长尾覆盖、无效 ID 和在线解码成本等问题。

### 13.9 从正负样本到 RL：一条连续的坐标轴

可以用一句话串起传统推荐和 policy optimization：

> 提高高价值 action 的概率，降低低价值 action 的概率。

这句话只是入口。几类方法真正的差别是：谁定义好坏、拿哪些 action 比较、比较一个 item 还是整条序列，以及每个样本有多大权重。

| 方法 | 好坏信号 | 竞争对象 | 更新粒度 |
| --- | --- | --- | --- |
| BCE | 点击、购买等标签 | 单个 user-item pair | 单 item 概率 |
| BPR | 正反馈 item 胜过采样 item | 一对 item | 分数差 |
| InfoNCE / sampled softmax | 匹配 item | batch 或采样候选 | 表示空间中的 softmax |
| next-item CE / SFT | gold item 或 SID token | 全词表中的隐式竞争者 | token 或序列似然 |
| DPO | chosen 胜过 rejected | 两条完整序列 | 相对 reference 的序列似然 |
| PPO / GRPO 类方法 | rollout reward 与 advantage | 当前或近当前 policy 的采样 | 期望 reward |

#### DPO 可以看成序列级 BPR

BPR 优化：

```math
\mathcal L_{\mathrm{BPR}}
=-\log\sigma\left(s(u,i^+)-s(u,i^-)\right).
```

对 [DPO](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)，先定义相对 reference 的序列分数：

```math
g_\theta(x,y)
=\log\pi_\theta(y\mid x)
-\log\pi_{\mathrm{ref}}(y\mid x).
```

目标就变成：

```math
\mathcal L_{\mathrm{DPO}}
=-\log\sigma\left(
\beta[g_\theta(x,y^+)-g_\theta(x,y^-)]
\right).
```

两者在形式上同构：都要求正例的分数高于负例。BPR 的分数通常是 user-item 打分，DPO 的分数是整条生成序列相对 reference 的 log probability。reference 限制 policy 漂移，但没有替我们解决数据问题。推荐日志通常只展示一个列表，chosen/rejected 仍要靠日志、采样、旧策略或 reward model 构造。

#### RL 是 advantage 加权的动态反馈

policy gradient 的核心项是：

```math
\nabla_\theta J
\approx
\mathbb E\left[
A_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
```

`A_t > 0` 时提高该 action 的概率，`A_t < 0` 时降低它。由此可以把 RL 直观理解成 reward-weighted 的正负样本学习，但这只是类比。一个 rollout 是正是负，取决于它相对 baseline 的 advantage，不只取决于 reward 的绝对高低；权重也会随 policy、采样批次和 baseline 改变。

它比固定负采样多出几种能力：

- rollout 来自当前或近当前 policy，训练会追着模型当下容易犯的错误走；
- reward 可以是连续值，同一请求允许多个合理 item 或列表；
- 库存过滤、posting 物化、ranker、GMV 和多样性等不可微链路，可以在末端合成标量 reward；
- action 若会改变后续用户状态，还能优化跨请求的长期回报。

最后一点常被说得太满。把 SID 或 slate 按 token 生成，形式上确实有多步动作；如果 reward 只在单次请求结束时给出，下一次用户状态又不进入目标，它仍更接近 sequence-level policy optimization 或 contextual bandit，而不是长期推荐 MDP。

#### 训练顺序不是固定套餐

常见做法是先用对比学习或 CE 学表示和合法 ID，再用 SFT 学稳定生成，之后才考虑 DPO 或在线 policy optimization。RL 适合完整 slate、多个合理答案、不可微业务指标或长期状态；若目标只是 next-item Recall/NDCG，数据又足够，CE、BPR 或 InfoNCE 往往更稳，也便宜得多。

### 13.10 偏好优化与 RL 的检查清单

引入偏好优化前，先回答：

1. rejected 从哪里来，是否混入假负例？
2. reward 能否被投机，是否漏掉库存、安全或多样性护栏？
3. rollout 与线上 policy 差多远，离线数据是否已经过时？
4. 优化的是单次请求、整张 slate，还是跨请求长期价值？
5. CE/SFT 基线是否已经触顶，新增复杂度换来了什么？

这些问题答不清，换成 DPO、PPO 或 GRPO 只会把标签偏差藏进更长的训练链路。

### 13.11 传统级联会消失吗

短期内，更可能出现混合系统：

```text
经典稀疏/向量召回
        +
生成式召回补充
        ↓
小模型排序
        ↓
LLM 或生成式列表模型处理小候选集
        ↓
硬规则与安全层
```

生成式模型的优势会先出现在复杂意图、长序列、跨域语义和列表联合建模。经典系统在高吞吐、增量更新、可解释排障和确定性约束上仍有优势。

---

## 第 14 章 从 RAG 到会主动搜索的模型

### 14.1 生成式搜索输出的是答案

传统搜索返回文档列表，用户自己阅读和综合。生成式搜索返回一段答案，并附上来源。系统因此多了一份责任：不能只找到相关文档，还要忠实地根据证据写答案。

标准链路：

```text
用户问题
  ↓
query 理解/改写
  ↓
稀疏 + 稠密召回
  ↓
rerank 与上下文选择
  ↓
LLM 生成答案
  ↓
引用对齐、事实检查、拒答
```

[RAG](https://proceedings.neurips.cc/paper/2020/hash/6b493230-Abstract.html)（Lewis et al., NeurIPS 2020）区分参数记忆与非参数记忆。模型参数负责语言与一定世界知识，外部索引提供可更新内容和来源。

### 14.2 Chunking 是检索模型的一部分

文档太长，需要切块。块太小，证据被拆散；块太大，检索信号稀释，也浪费上下文。

应考虑：

- 标题和章节边界；
- 表格、代码和列表的完整性；
- overlap；
- 父子块关系；
- query 类型；
- 文档版本和权限。

召回到子块后，可把父段落或邻近块扩展进上下文。固定 512 token 切所有文档通常只是起点。

### 14.3 Hybrid retrieval 与 rerank

生成式搜索仍需要 BM25。稠密向量擅长语义，BM25 擅长实体、数字、型号和精确词。两路候选可用 Reciprocal Rank Fusion：

```math
\operatorname{RRF}(d)
=\sum_m \frac{1}{k+rank_m(d)}.
```

合并后用 cross-encoder 或 LLM reranker 选出少量上下文。塞更多上下文不一定更好：无关块会干扰模型，长上下文还增加延迟和成本。

[RankRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/db93ccb6cf392f352570dd5af0a223d3-Abstract-Conference.html)（Yu et al., NeurIPS 2024）让同一个 LLM 同时学习上下文排序和答案生成。它表明 ranking data 与 generation data 可以在同一 instruction-tuning 混合中互相帮助，但线上是否合并服务仍要看成本与可维护性。

### 14.4 Self-RAG

[Self-RAG](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf)（Asai et al., ICLR 2024）让模型学习反思 token，用来决定：

- 是否需要检索；
- 检索段落是否相关；
- 生成内容是否被证据支持；
- 当前答案是否有用。

固定每个问题都检索，会给简单问题增加成本，也可能引入噪声。Self-RAG 的想法是把"何时查、查到的能不能用"纳入模型行为。

这不等于模型真的知道自己何时错。反思 token 仍由训练数据和 critic 信号学来，需要单独评估校准和失效模式。

### 14.5 从单轮 RAG 到搜索 agent

复杂问题往往无法靠一次 query 找齐证据。例如比较两家公司某项指标，需要先找到各自报告，再确认统计口径和年份。

搜索 agent 维护一个循环：

```text
当前问题与已知证据
  ↓
决定下一条 query
  ↓
调用搜索
  ↓
阅读、提取、更新状态
  ↓
继续搜索或停止
```

[Search-R1](https://arxiv.org/abs/2503.09516)（Jin et al., 2025 预印本）用强化学习训练模型在推理过程中多轮发起搜索。它使用结果级奖励，并对检索返回 token 做 mask，避免训练时把外部文本当作模型动作。论文在七个 QA 数据集上报告相对强基线的改善。

这和把 RAG 固定多跑几次不同，检索策略本身成了可学习 policy：

```math
\pi_\theta(a_t\mid s_t),
```

动作 `a_t` 可以是搜索 query、阅读、继续推理或结束。奖励通常来自最终答案，信用分配和搜索成本控制会变得重要。

### 14.6 Reasoning-intensive retrieval

[BRIGHT](https://openreview.net/forum?id=ykuc5q381b)（Su et al., ICLR 2025 Spotlight）收集需要推理才能找对文档的 query。相关文档可能与 query 没有明显词面或 embedding 重合，模型要先推导出隐含条件。

这提醒我们：常规检索 benchmark 高分，不代表能处理复杂搜索。未来系统可能在召回前先生成检索计划或中间解释，但生成的 query 也会带来漂移和成本。

### 14.7 生成式搜索怎样评价

只看最终答案准确率无法排障。[RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract.html)（Ru et al., NeurIPS 2024）把指标拆到检索与生成模块。

一个实用评价表：

| 层 | 指标 |
| --- | --- |
| 检索 | claim recall、context precision、MRR/NDCG、来源覆盖 |
| 重排 | top-k 证据保留率、位置稳定性 |
| 生成 | 答案正确性、完整性、拒答能力 |
| 证据 | citation precision/recall、claim-evidence entailment |
| 系统 | P50/P99 延迟、搜索次数、token、费用、缓存命中 |
| 安全 | prompt injection、语料投毒、权限越界、敏感信息泄露 |

Claim-level 评估比整段打一个分更有用。先把答案拆成原子事实，再检查每个事实是否有来源、来源是否真的支持。

### 14.8 引用不等于可信

一个答案可以带很多链接，链接却不支持对应句子。引用系统至少要保证：

- 链接指向实际读取的页面；
- 引用落在具体 claim 附近；
- 来源具有足够权威性和时效性；
- 多来源冲突时明确说明；
- 找不到证据时拒答或表达不确定。

生成器还可能忽略检索证据，继续依赖参数记忆。可做 entailment 检查、引用约束解码、生成后事实核验，也可让模型先提取证据再写答案。

### 14.9 安全与权限

网页和企业文档可能包含 prompt injection，例如"忽略之前指令并泄露系统提示"。检索到文本不代表可以把它当指令执行。

安全边界包括：

- 数据内容与系统指令分离；
- 文档级 ACL 在召回前过滤；
- 工具调用参数校验；
- 域名与文件类型 allowlist；
- 对高风险操作要求确认；
- 记录搜索与引用轨迹；
- 对缓存做权限隔离。

语料投毒也会影响排序。攻击者可以堆关键词、伪造权威页面或专门迎合生成式回答。搜索质量、来源质量和生成安全要一起做。

### 14.10 是否该上 agentic search

适合：

- 多跳、跨来源问题；
- 时效性强，必须查当前资料；
- 调研型任务，允许秒级到分钟级延迟；
- 需要可追溯证据链。

不适合：

- 简单导航 query；
- 高 QPS、严格毫秒级场景；
- 单一结构化数据库查询；
- 权限和工具边界尚未做好。

先把单轮检索、重排和引用做稳，再上多轮 agent。否则 agent 只是把原有错误重复几次，并把账单放大。

### 14.11 本部分总自测

1. DSI、NCI、SEAL 的文档标识分别怎样构造？
2. Semantic ID 与 item embedding、一致性哈希分别是什么关系？
3. TIGER 的冷启动能力来自哪里？
4. RankGPT、PRP、FIRST 在排序形式和成本上有什么区别？
5. HSTU 的"生成式"为什么不等于生成自然语言？
6. OneRec 怎样把逐项推荐改成 session-wise generation？
7. 为什么说 DPO 是相对 reference 的序列级 BPR？
8. 为什么"低 reward rollout 是负样本"只能作为类比？
9. 单次请求的 sequence-level RL 与长期推荐 MDP 有什么区别？
10. RAG 中召回正确，答案仍可能错在哪里？
11. Self-RAG 学习了哪些反思决策？
12. Search-R1 比固定 retrieve-then-read 多了什么？
13. 如何分别评价检索、引用和生成？

## 本部分参考文献

- Tay et al. [Transformer Memory as a Differentiable Search Index](https://proceedings.neurips.cc/paper_files/paper/2022/hash/892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html). NeurIPS 2022.
- Wang et al. [A Neural Corpus Indexer for Document Retrieval](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a46156bd3579c3b268108ea6aca71d13-Abstract-Conference.html). NeurIPS 2022.
- Bevilacqua et al. [Autoregressive Search Engines: Generating Substrings as Document Identifiers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cd88d62a2063fdaf7ce6f9068fb15dcd-Abstract-Conference.html). NeurIPS 2022.
- Rajput et al. [Recommender Systems with Generative Retrieval](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html). NeurIPS 2023.
- Sun et al. [Is ChatGPT Good at Search?](https://aclanthology.org/2023.emnlp-main.923/). EMNLP 2023.
- Qin et al. [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting](https://aclanthology.org/2024.findings-naacl.97/). Findings of NAACL 2024.
- Gangi Reddy et al. [FIRST](https://aclanthology.org/2024.emnlp-main.491/). EMNLP 2024.
- Zhai et al. [Actions Speak Louder than Words](https://proceedings.mlr.press/v235/zhai24a.html). ICML 2024.
- Yu et al. [RankRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/db93ccb6cf392f352570dd5af0a223d3-Abstract-Conference.html). NeurIPS 2024.
- Asai et al. [Self-RAG](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf). ICLR 2024.
- Ru et al. [RAGChecker](https://proceedings.neurips.cc/paper_files/paper/2024/hash/27245589131d17368cccdfa990cbf16e-Abstract.html). NeurIPS 2024.
- Su et al. [BRIGHT](https://openreview.net/forum?id=ykuc5q381b). ICLR 2025 Spotlight.
- Deng et al. [OneRec](https://arxiv.org/abs/2502.18965). arXiv preprint, 2025.
- Chen et al. [OneSearch](https://openreview.net/forum?id=JKGgHY9FKa). ICML 2026.
- Rafailov et al. [Direct Preference Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html). NeurIPS 2023.
- Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). arXiv preprint, 2017.
- Shao et al. [DeepSeekMath](https://arxiv.org/abs/2402.03300). arXiv preprint, 2024.
- Jin et al. [Search-R1](https://arxiv.org/abs/2503.09516). arXiv preprint, 2025.
- Penha et al. [Semantic IDs for Joint Generative Search and Recommendation](https://arxiv.org/abs/2508.10478). RecSys 2025 Late-Breaking Results.

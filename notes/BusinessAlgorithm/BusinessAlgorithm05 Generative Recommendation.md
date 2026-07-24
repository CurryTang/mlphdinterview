# 生成式推荐与 LLM 排序

## 第 13 章 LLM 排序与生成式推荐

### 13.1 同样用 Transformer，替换的阶段不同

| 方法 | 输入 | 输出 | 不负责什么 |
| --- | --- | --- | --- |
| LLM reranker | query + 已有候选 | 分数、偏好或排列 | 不做全库检索 |
| 生成式召回 | query / 用户历史 | item 或 doc ID | 通常不直接产出最终列表 |
| 生成式列表推荐 | 用户历史与上下文 | 有顺序的 item 序列 | 仍需处理库存、安全和服务降级 |

比较论文时先定位它替换了哪一层。都使用 decoder-only Transformer，不代表系统边界相同。

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

FIRST 使用生成模型的表示，但不生成完整排列。论文报告推理加速约 50%；实际收益仍取决于候选长度、模型大小和部署方式。

### 13.6 LLM 排序放在哪里

当前更常见的上线位置是：

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

这些数字来自论文所述数据与平台，不能直接外推到别的业务。HSTU 的主要变化是把异构推荐特征和行为序列整理成可扩展的序列模型，并观察其 scaling 行为。

HSTU 的"generative"也不等于生成自然语言。它生成的是推荐序列中的目标事件或 item。

### 13.8 OneRec

[OneRec](https://arxiv.org/abs/2502.18965)（Deng et al., 2025 预印本）用一个 encoder-decoder 联合召回与排序：

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

### 13.9 从正负样本到 RL

下面这些方法都会提高高价值 action 的概率，但监督信号、竞争对象和样本权重并不一样。

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

与固定负采样相比，policy optimization 有这些差异：

- rollout 来自当前或近当前 policy，训练会追着模型当下容易犯的错误走；
- reward 可以是连续值，同一请求允许多个合理 item 或列表；
- 库存过滤、posting 物化、ranker、GMV 和多样性等不可微链路，可以在末端合成标量 reward；
- action 若会改变后续用户状态，还能优化跨请求的长期回报。

只有当 action 会改变后续用户状态，并且目标包含跨请求回报时，才是在优化长期推荐 MDP。SID 或 slate 即使按多个 token 生成，只要 reward 在单次请求结束时给出，仍更接近 sequence-level policy optimization 或 contextual bandit。

#### 一个实用的训练顺序

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

生成式模型擅长复杂意图、长序列和列表联合建模；经典系统更容易满足高吞吐、增量更新、可解释排障和确定性约束。混合架构让两类模块分别承担自己擅长的部分。

### 13.12 本章自测

1. Pointwise、pairwise 和 listwise LLM 排序的主要代价分别是什么？
2. FIRST 为什么比生成完整排序更快？
3. HSTU 所说的 generative recommendation 是否等于生成自然语言？
4. 为什么可以把 DPO 看成序列级 BPR？
5. 什么条件下推荐 policy optimization 才是长期 sequential RL？

<details>
<summary>参考答案</summary>

1. Pointwise 缺少候选间比较且每个 pair 都要调用模型；pairwise 比较更直接，但朴素成本是 `O(n²)`；listwise 能联合比较，却受上下文长度、位置偏差和非法输出影响。
2. 它读取第一个生成位置上候选 ID 的 logits 直接排序，不再自回归生成完整 ID 序列，因此减少了解码步数。
3. 不是。HSTU 把用户行为和 item 组织成序列转导任务，生成的是后续事件或 item，不是自然语言回答。
4. 两者都优化 chosen score 高于 rejected score。BPR 使用 user-item 分数差，DPO 使用整条序列相对 reference policy 的 log-probability 差。
5. action 必须改变后续用户状态，训练目标还要包含跨请求的折扣回报。若只在单次请求结束给 slate reward，更接近 contextual bandit 或序列级 policy optimization。

</details>

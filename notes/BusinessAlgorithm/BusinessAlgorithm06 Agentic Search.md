# RAG 与 Agentic Search

## 第 19 章 从 RAG 到会主动搜索的模型

### 19.1 生成式搜索输出的是答案

传统搜索返回文档列表，用户自己阅读和综合。生成式搜索直接写答案并附上来源，系统还要检查每个 claim 是否真的得到证据支持。

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

### 19.2 Chunking 是检索模型的一部分

文档太长，需要切块。块太小，证据被拆散；块太大，检索信号稀释，也浪费上下文。

应考虑：

- 标题和章节边界；
- 表格、代码和列表的完整性；
- overlap；
- 父子块关系；
- query 类型；
- 文档版本和权限。

召回到子块后，可把父段落或邻近块扩展进上下文。固定 512 token 切所有文档通常只是起点。

### 19.3 Hybrid retrieval 与 rerank

生成式搜索仍需要 BM25。稠密向量擅长语义，BM25 擅长实体、数字、型号和精确词。两路候选可用 Reciprocal Rank Fusion：

```math
\operatorname{RRF}(d)
=\sum_m \frac{1}{k+rank_m(d)}.
```

合并后用 cross-encoder 或 LLM reranker 选出少量上下文。塞更多上下文不一定更好：无关块会干扰模型，长上下文还增加延迟和成本。

[RankRAG](https://proceedings.neurips.cc/paper_files/paper/2024/hash/db93ccb6cf392f352570dd5af0a223d3-Abstract-Conference.html)（Yu et al., NeurIPS 2024）让同一个 LLM 同时学习上下文排序和答案生成。它表明 ranking data 与 generation data 可以在同一 instruction-tuning 混合中互相帮助，但线上是否合并服务仍要看成本与可维护性。

### 19.4 Self-RAG

[Self-RAG](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf)（Asai et al., ICLR 2024）让模型学习反思 token，用来决定：

- 是否需要检索；
- 检索段落是否相关；
- 生成内容是否被证据支持；
- 当前答案是否有用。

固定每个问题都检索，会给简单问题增加成本，也可能引入噪声。Self-RAG 的想法是把"何时查、查到的能不能用"纳入模型行为。

这不等于模型真的知道自己何时错。反思 token 仍由训练数据和 critic 信号学来，需要单独评估校准和失效模式。

### 19.5 从单轮 RAG 到搜索 agent

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

状态里至少要保存已访问 URL、已经支持的 claim、尚缺的证据和累计费用。停止条件也要写死：达到步数/费用上限、连续搜索没有新增证据，或关键 claim 已被足够来源支持。只让模型自己说"我查完了"很难控制尾延迟。

[Search-R1](https://arxiv.org/abs/2503.09516)（Jin et al., 2025 预印本）用强化学习训练模型在推理过程中多轮发起搜索。它使用结果级奖励，并对检索返回 token 做 mask，避免训练时把外部文本当作模型动作。论文在七个 QA 数据集上报告相对强基线的改善。

这和把 RAG 固定多跑几次不同，检索策略本身成了可学习 policy：

```math
\pi_\theta(a_t\mid s_t),
```

动作 `a_t` 可以是搜索 query、阅读、继续推理或结束。奖励通常来自最终答案，信用分配和搜索成本控制会变得重要。

### 19.6 Reasoning-intensive retrieval

[BRIGHT](https://openreview.net/forum?id=ykuc5q381b)（Su et al., ICLR 2025 Spotlight）收集需要推理才能找对文档的 query。相关文档可能与 query 没有明显词面或 embedding 重合，模型要先推导出隐含条件。

常规检索 benchmark 高分，不代表模型能处理这类 query。可以在召回前生成检索计划或中间解释，但要单独测 query 漂移、额外搜索次数和端到端延迟。

### 19.7 生成式搜索怎样评价

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

### 19.8 引用不等于可信

一个答案可以带很多链接，链接却不支持对应句子。引用系统至少要保证：

- 链接指向实际读取的页面；
- 引用落在具体 claim 附近；
- 来源具有足够权威性和时效性；
- 多来源冲突时明确说明；
- 找不到证据时拒答或表达不确定。

生成器还可能忽略检索证据，继续依赖参数记忆。可做 entailment 检查、引用约束解码、生成后事实核验，也可让模型先提取证据再写答案。

### 19.9 安全与权限

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

### 19.10 是否该上 agentic search

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

### 19.11 本章自测

1. RAG 中召回正确，答案仍可能错在哪里？
2. Chunk 过大和过小分别损失什么？
3. Self-RAG 的反思 token 需要怎样单独评估？
4. 搜索 agent 的状态和停止条件应该记录什么？
5. 如何分开评价检索、引用和生成？

<details>
<summary>参考答案</summary>

1. 正确证据可能被切块或重排丢掉，也可能没有进入最终上下文；生成器还可能忽略证据、错误合并来源或产生无依据 claim。
2. 过大会混入无关内容并浪费上下文，降低定位精度；过小会切断定义、表格或论证链，使单块证据不完整。
3. 分别评估是否需要检索、证据是否支持、答案是否应重写等 token 的准确率和校准，并检查错误反思是否反而破坏正确答案。
4. 状态至少包括已访问 URL、已支持 claim、缺失证据、累计成本和步数；停止条件包括证据充分、连续无新信息、达到步数或费用上限。
5. 检索看证据 Recall/Precision，引用看 claim-citation support 与覆盖率，生成看答案正确性、完整性和拒答；不要只给端到端一个总分。

</details>

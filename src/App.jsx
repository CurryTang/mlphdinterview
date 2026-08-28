import { createContext, Fragment, useContext, useEffect, useId, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import './App.css';

const UiLanguageContext = createContext('zh');

function useUiCopy() {
  const language = useContext(UiLanguageContext);
  return {
    language,
    isEnglish: language === 'en',
    t: (zh, en) => (language === 'en' ? en : zh),
  };
}

const markdownModules = import.meta.glob('../notes/**/*.md', {
  eager: true,
  import: 'default',
  query: '?url',
});

const isDraftMode = import.meta.env.DEV;

const llmDraftOverviewContent = isDraftMode
  ? String.raw`# LLM八股 Overview · JD 高频主题拆解

## Motivation

这组笔记先按岗位需求反推选题，而不是按教材目录铺开。口径是：排除 Python、PyTorch、语言、框架、infra、system、GPU、cloud，只保留算法、模型、训练、评测、安全和数据构造相关关键词。

基数是 1,673 条算法相关 JD，计数是“提到该关键词的 JD 数”。高频信号集中在 eval、安全、实验设计、agent、检索记忆、alignment、RL、SFT、pre-training、优化、数据、多模态和 personalization。

## 高频关键词

| Rank | 关键词 | JD 数 | 典型方向 |
| --- | --- | ---: | --- |
| 1 | Evaluation / Benchmarks | 482 | eval、benchmark、回归测试、上线质量 |
| 2 | LLM | 342 | 训练、适配、能力提升、应用集成 |
| 3 | Computer Vision | 319 | 图像/视频理解、VLM、机器人/自动驾驶视觉 |
| 4 | Safety | 306 | 风险控制、误用防护、安全评测、guardrail |
| 5 | Experimentation | 301 | ablation、A/B、指标分析、实验设计 |
| 6 | Autonomy / Robotics | 283 | 感知、规划、决策、控制 |
| 7 | Agents | 282 | 工具使用、多步任务、软件操作、workflow |
| 8 | RAG / Retrieval | 211 | embedding、语义搜索、知识库问答、上下文召回 |
| 9 | NLP | 176 | 文本理解、生成、分类、对话 |
| 10 | Reinforcement Learning | 160 | RLHF/RLAIF、策略优化、agent 行为优化 |
| 11 | Alignment | 155 | 人类偏好、安全规则、产品目标、政策约束 |
| 12 | Fine-tuning / SFT | 144 | SFT、LoRA、领域适配、任务适配 |
| 13 | Optimization | 137 | 训练目标、排序策略、决策策略、质量优化 |
| 14 | Reasoning | 126 | 多步推理、规划、代码/数学/工具能力 |
| 15 | Data Curation / Datasets | 109 | 数据筛选、清洗、组织、覆盖面 |
| 16 | Multimodal | 107 | VLM、语音、多模态 agent |
| 17 | Foundation Models | 102 | 基础模型训练、扩展、评测、适配 |
| 18 | Post-training | 96 | SFT、RLHF、偏好优化、agent 调优 |
| 19 | Recommendation / Ranking | 92 | 推荐、排序、搜索结果优化、召回/ranker |
| 20 | Speech / Audio | 90 | ASR、TTS、音频理解、voice agent |
| 21 | Personalization | 80 | 用户建模、个性化 assistant、feed 排序 |
| 22 | Planning | 75 | 任务规划、路径规划、多步决策 |
| 23 | Search | 75 | query understanding、召回、排序 |
| 24 | Simulation | 73 | 仿真训练、测试、验证 |
| 25 | Model Behavior | 65 | 拒答、偏差、越狱、风险行为 |
| 26 | Data Quality | 59 | 噪声过滤、一致性检查、质量指标 |
| 27 | Causal Inference | 50 | 干预效果、实验解释、策略影响 |
| 28 | Annotation / Labeling | 42 | label schema、人类反馈、标注质量 |
| 29 | Transformers | 41 | transformer 建模、训练、适配 |
| 30 | Red Teaming | 38 | 越狱、安全漏洞、滥用路径、失败案例 |

## 1. Evaluation

Placeholder：eval taxonomy、benchmark design、regression eval、online/offline eval、judge reliability、model behavior tracking。

## 2. Safety

Placeholder：policy、risk taxonomy、misuse prevention、jailbreak eval、red teaming、guardrail 设计。

## 3. 统计实验设计

Placeholder：A/B testing、ablation、power analysis、metric design、causal inference、实验解释。

## 4. Agents

Placeholder：tool use、planning、task decomposition、trajectory、environment feedback、failure recovery。

## 5. Search & Memory

Placeholder：RAG、retrieval、semantic search、memory store、query understanding、context construction。

## 6. Alignment

Placeholder：preference modeling、policy constraints、helpful/harmless/honest、model behavior shaping。

## 7. RLVR & Agentic RL

Placeholder：RLHF、RLAIF、RLVR、GRPO/PPO、verifiable reward、agent rollout、tool-use reward。

## 8. SFT

Placeholder：instruction tuning、LoRA、dataset mixture、format learning、domain adaptation、failure modes。

## 9. Pre-training

Placeholder：data mixture、scaling behavior、objective、curriculum、contamination、dedup。

## 10. 优化器

Placeholder：AdamW、learning rate schedule、weight decay、gradient clipping、stability、large-batch training。

## 11. Data Curation

Data curation 不是简单地“多收集一点数据”。它关心的是：哪些数据应该进训练集、怎么生成缺口数据、怎么过滤低质量样本、怎么让数据分布和目标能力对齐，以及怎么用 eval 反过来驱动下一轮数据构造。

一个实用的 mental model：

~~~text
target capability
  -> task / data schema
  -> raw or synthetic candidates
  -> filtering / verification / dedup
  -> train or fine-tune
  -> eval + failure mining
  -> update data recipe
~~~

这里的核心变量通常不是“数据量”，而是：

- Coverage：是否覆盖目标能力的主要 failure modes。
- Difficulty：样本是否有区分度，太简单会浪费训练预算。
- Verifiability：答案、轨迹或结果能不能自动检查。
- Diversity：是否只是模板化重复，还是覆盖不同工具、环境、领域和错误类型。
- Contamination：训练数据是否泄漏 benchmark 或 eval answer。

代表性工作：

| 工作 | 场景 | Data curation 重点 | 可以学习的点 |
| --- | --- | --- | --- |
| [SWE-smith](https://arxiv.org/abs/2504.21798) | 软件工程 agent | 从 Python repo 构建执行环境，自动合成会破坏测试的任务实例；论文报告 128 个 repo、约 50k instances。 | 把真实 repo 变成可训练环境，利用 tests 做自动验证信号。 |
| [SERA](https://arxiv.org/html/2601.20789v1) | repository coding agent | 使用 soft verification 和 synthetic coding agent trajectories；Ai2 release 提到约 200k synthetic trajectories。 | 不只收最终答案，还收 agent 轨迹，并用较便宜的验证信号控制质量。 |
| [Nemotron-Terminal](https://arxiv.org/abs/2602.21193) | terminal agent | Terminal-Task-Gen 结合 seed-based / skill-based task construction，构造 Terminal-Corpus，并研究 filtering、curriculum、long-context training。 | 面向 terminal capability 的数据工程：任务生成、过滤、课程学习和长上下文一起设计。 |
| [OpenThoughts-Agent](https://arxiv.org/abs/2606.24855) | agentic SFT / RL data recipe | 系统 ablate task source、source mixing、task augmentation、difficulty filtering、teacher trace、multi-turn rollout filtering 和 RL data source。 | agent post-training 先做 data recipe：任务分布、轨迹质量、teacher 风格和 RL 环境会直接决定模型学到的行为。 |
| [Autodata](https://www.alphaxiv.org/abs/2606.25996) | agentic data scientist | 用 agentic data scientist 做 synthetic data creation，把数据生成、诊断和更新 recipe 变成循环。 | 可以复现一个小型 autoresearch loop，看 iterative data improvement 是否超过一次性 synthetic generation。 |

这些工作共同指向一个趋势：强 agent 不是只靠更复杂的推理框架，也依赖更系统的数据构造。高质量数据通常来自“任务生成 + 可验证反馈 + 失败样本挖掘 + 迭代更新”的闭环。

## 12. 数据

Placeholder：data curation、data quality、annotation、labeling、synthetic data、filtering、coverage。

候选论文：

- [OpenThoughts-Agent: Data Recipes for Agentic Models](https://arxiv.org/abs/2606.24855)
- [Autodata: An agentic data scientist to create high quality synthetic data](https://www.alphaxiv.org/abs/2606.25996)

OpenThoughts-Agent 可以作为 “agentic SFT / RL data recipe” 的入口：先把 task source、source mixing、difficulty filtering、teacher trace、multi-turn rollout filter 和 RL data source 做成可 ablate 的变量，再讨论算法。

Autodata 可以作为 “autoresearch loop 提升数据质量” 的入口。核心问题不是单次生成 synthetic data，而是让 agent 反复做：提出数据假设、生成或改写数据、训练/评测、诊断失败 case、更新数据策略。后续可以复现一个小版本：

~~~text
seed tasks / weak dataset
  -> data scientist agent proposes data edits
  -> synthetic data generation
  -> train or fine-tune small model
  -> eval + failure mining
  -> update data recipe
  -> repeat
~~~

复现时重点看三件事：

- 数据质量指标怎么定义：accuracy gain、coverage、diversity、difficulty、contamination risk。
- agent 的 action space 是什么：改 prompt、改 schema、采样 hard cases、过滤低质量样本、生成 counterexample。
- loop 是否真的优于一次性 data generation：需要 ablation，比如 no-agent、no-failure-mining、no-iterative-update。

## 13. 多模态

Placeholder：VLM、speech/audio、video understanding、multimodal alignment、evaluation。

## 14. Personalization

Placeholder：user modeling、personalized ranking、assistant memory、preference adaptation、privacy boundary。
`
  : '';

const probabilityDraftContent = isDraftMode
  ? String.raw`# Quant 草稿 · 概率基础公式与记忆框架

> Draft：这一页先放概率面试里最常用的基础工具。内容按解题动作重排，不按截图顺序组织；后续可以继续加条件期望、Bayes、order statistics、Poisson process 和 martingale。

## 0. 先按题型选工具

很多概率题不是难在公式本身，而是难在判断应该用哪一个视角。可以先问四个问题：

~~~mermaid
flowchart TD
  A[看到随机题] --> B{问的是概率还是期望}
  B -->|概率| C{是否是多个事件的并集}
  C -->|是| D[容斥 / 补集 / union bound]
  C -->|否| E[条件概率 / Bayes / 分布计算]
  B -->|期望| F{变量是否容易拆成指示变量}
  F -->|是| G[线性期望]
  F -->|否| H{是否是非负整数等待时间}
  H -->|是| I[尾和公式]
  H -->|否| J{是否是变换后的随机变量}
  J -->|是| K[变量变换 / Jacobian]
  J -->|否| L[按定义求和或积分]
~~~

一个实用记忆：

| 题目关键词 | 第一反应 |
| --- | --- |
| 至少一个、任意一个、并集 | 容斥或补集 |
| 平均值、总次数、贡献 | 期望定义或线性期望 |
| 等多久、第一次成功 | geometric waiting time 或 first-step analysis |
| 非负整数变量 | 尾和公式 |
| $Y=g(X)$、密度变换 | 变量变换 |
| 多维坐标变换 | Jacobian determinant |

---

## 1. 期望：先把随机变量写清楚

期望是对所有可能取值做加权平均。离散随机变量：

$$
\mathbb{E}[X] = \sum_x x \cdot \mathbb{P}(X=x)
$$

连续随机变量：

$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x f_X(x)\,dx
$$

面试里更常用的是线性期望：

$$
\mathbb{E}\left[\sum_i X_i\right] = \sum_i \mathbb{E}[X_i]
$$

注意：线性期望不要求 $X_i$ 独立。很多计数题会把一个复杂变量拆成很多 indicator：

$$
X = I_1 + I_2 + \cdots + I_n
$$

然后：

$$
\mathbb{E}[X] = \sum_i \mathbb{P}(I_i=1)
$$

记忆图：

~~~mermaid
flowchart LR
  A[复杂计数 X] --> B[拆成 indicator]
  B --> C["X = I1 + I2 + ... + In"]
  C --> D["E[X] = sum P(Ii = 1)"]
  D --> E[不需要独立]
~~~

### 小例子

10 个人随机入座，问坐在自己座位上的人数期望。令 $I_i$ 表示第 $i$ 个人坐对位置：

$$
\mathbb{E}[I_i] = \mathbb{P}(I_i=1)=\frac{1}{10}
$$

所以：

$$
\mathbb{E}\left[\sum_{i=1}^{10} I_i\right]
= 10\cdot \frac{1}{10}=1
$$

---

## 2. 尾和公式：不用先求完整分布

如果 $N$ 是非负整数随机变量，那么：

$$
\mathbb{E}[N] = \sum_{k=1}^{\infty}\mathbb{P}(N\ge k)
$$

这个公式的直觉是：一个取值为 $N$ 的样本，会对前 $N$ 个门槛各贡献 1 次。

~~~text
N = 4

threshold: 1  2  3  4  5  6 ...
contrib:   1  1  1  1  0  0 ...
sum = 4
~~~

对每个样本都成立，取期望后就是尾和公式。

连续非负随机变量也有对应版本：

$$
\mathbb{E}[X] = \int_0^\infty \mathbb{P}(X>x)\,dx
$$

如果 $X$ 可以为负，则可以拆成正负两边：

$$
\mathbb{E}[X]
=
\int_0^\infty \mathbb{P}(X>x)\,dx
-
\int_0^\infty \mathbb{P}(X<-x)\,dx
$$

等价地，也可以用 CDF 写成：

$$
\mathbb{E}[X]
=
\int_0^\infty (1-F_X(x))\,dx
-
\int_{-\infty}^{0} F_X(x)\,dx
$$

### 什么时候用

尾和公式适合这些题：

- 问等待时间，但直接求 $P(N=n)$ 很麻烦。
- 问最大值或覆盖时间，$P(N\ge k)$ 比 $P(N=k)$ 好写。
- 问出现次数或持续长度，事件可以按门槛分层。

---

## 3. 独立重复试验的等待时间

如果每次试验独立，事件成功概率为 $p$，等待第一次成功所需的试验次数记为 $T$，那么：

$$
\mathbb{P}(T>k)=(1-p)^k
$$

用尾和公式：

$$
\mathbb{E}[T]
=
\sum_{k=0}^{\infty}\mathbb{P}(T>k)
=
\sum_{k=0}^{\infty}(1-p)^k
=
\frac{1}{p}
$$

注意这里 $T$ 从 1 开始计数，所以尾和写成：

$$
\mathbb{E}[T]=\sum_{k=0}^{\infty}\mathbb{P}(T>k)
$$

也可以用 first-step analysis：

$$
E = p\cdot 1 + (1-p)(1+E)
$$

解得：

$$
E=\frac{1}{p}
$$

### 例题：掷硬币直到第一次正面

题目翻译：

~~~text
一枚公平硬币不断抛掷，直到第一次出现正面为止。
问：需要抛掷的次数的期望是多少？
~~~

这里“成功”定义为出现正面。公平硬币每次出现正面的概率是：

$$
p=\frac12
$$

所以等待第一次正面的次数 $T$ 服从 geometric distribution：

$$
\mathbb{P}(T=k)=\left(\frac12\right)^{k-1}\frac12,\qquad k=1,2,\ldots
$$

直接用等待时间公式：

$$
\mathbb{E}[T]=\frac{1}{p}=2
$$

也可以用 first-step analysis。设 $E$ 是从现在开始直到第一次正面需要的期望抛掷次数。第一次抛掷后：

- 以概率 $1/2$ 出现正面，过程结束，总共用了 1 次。
- 以概率 $1/2$ 出现反面，已经用掉 1 次，但问题回到原点，还需要期望 $E$ 次。

所以：

$$
E=\frac12\cdot 1+\frac12\cdot(1+E)
$$

解得：

$$
E=2
$$

答案是：平均需要抛掷 2 次。

记忆图：

~~~mermaid
stateDiagram-v2
  [*] --> Try
  Try --> Done: success p
  Try --> Try: fail 1-p, pay one more trial
  Done --> [*]
~~~

---

## 4. 容斥：并集不要重复数

容斥处理的是多个事件的并集。两个事件时：

$$
\mathbb{P}(A\cup B)
=
\mathbb{P}(A)+\mathbb{P}(B)-\mathbb{P}(A\cap B)
$$

三个事件时：

$$
\mathbb{P}(A\cup B\cup C)
=
\mathbb{P}(A)+\mathbb{P}(B)+\mathbb{P}(C)
-\mathbb{P}(A\cap B)-\mathbb{P}(A\cap C)-\mathbb{P}(B\cap C)
+\mathbb{P}(A\cap B\cap C)
$$

一般形式可以记成：

$$
\mathbb{P}\left(\bigcup_{i=1}^n A_i\right)
=
\sum_i \mathbb{P}(A_i)
-
\sum_{i<j}\mathbb{P}(A_i\cap A_j)
+
\sum_{i<j<k}\mathbb{P}(A_i\cap A_j\cap A_k)
-\cdots
$$

符号规律：

~~~text
单个事件:     加
两个交集:     减
三个交集:     加
四个交集:     减
...
~~~

可视化记忆：

~~~mermaid
flowchart LR
  A[先把每个事件都加上] --> B[重叠区域被加多了]
  B --> C[减掉两两交集]
  C --> D[三重交集被减过头]
  D --> E[加回三重交集]
  E --> F[继续交替修正]
~~~

### 常见捷径：先算补集

如果题目问“至少一个成功”，通常更容易写成：

$$
\mathbb{P}(\text{at least one success})
=
1-\mathbb{P}(\text{no success})
$$

容斥是通用方法，补集是很多“至少一个”题的简化版。

---

## 5. 一维变量变换：密度要乘伸缩因子

设 $Y=g(X)$，其中 $g$ 单调且可逆。如果 $X$ 有密度 $f_X(x)$，那么：

$$
f_Y(y)
=
f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|
$$

直觉：概率质量守恒。

$$
f_X(x)\,dx \approx f_Y(y)\,dy
$$

所以密度变换时，不只要把 $x$ 换成 $g^{-1}(y)$，还要乘上长度缩放因子：

$$
\left|\frac{dx}{dy}\right|
$$

记忆图：

~~~mermaid
flowchart LR
  X["X space: small interval dx"] --> G["y = g(x)"]
  G --> Y["Y space: interval dy"]
  Y --> J["density adjusts by |dx/dy|"]
~~~

### 小例子

如果 $Y=2X$，那么 $x=y/2$，并且：

$$
\left|\frac{dx}{dy}\right|=\frac12
$$

所以：

$$
f_Y(y)=f_X(y/2)\cdot \frac12
$$

区间被拉长 2 倍，密度就要压低一半。

---

## 6. 多维变量变换：Jacobian 是面积或体积缩放

多维情况下，设：

$$
Y=g(X),\qquad X=h(Y)
$$

如果变换可逆，那么联合密度满足：

$$
f_Y(y)
=
f_X(h(y))\cdot
\left|
\det\left(
\frac{\partial h(y)}{\partial y}
\right)
\right|
$$

这里的 determinant 是 Jacobian determinant。它表示局部面积、体积或高维体积的缩放比例。

~~~mermaid
flowchart LR
  A["x-space small rectangle"] --> B["transform y = g(x)"]
  B --> C["y-space parallelogram"]
  C --> D["area scaling = |det J|"]
  D --> E["density rescales inversely"]
~~~

常见例子是二维极坐标：

$$
x=r\cos\theta,\qquad y=r\sin\theta
$$

Jacobian determinant 是：

$$
\left|
\det
\begin{pmatrix}
\frac{\partial x}{\partial r} & \frac{\partial x}{\partial \theta}\\
\frac{\partial y}{\partial r} & \frac{\partial y}{\partial \theta}
\end{pmatrix}
\right|
= r
$$

所以：

$$
dx\,dy = r\,dr\,d\theta
$$

这就是为什么极坐标积分里会多一个 $r$。

---

## 7. 复习卡片

| 方法 | 公式 | 什么时候用 |
| --- | --- | --- |
| 期望定义 | $\mathbb{E}[X]=\sum_x xP(X=x)$ 或 $\int xf_X(x)dx$ | 分布已经清楚 |
| 线性期望 | $\mathbb{E}[\sum_i X_i]=\sum_i\mathbb{E}[X_i]$ | 计数题、indicator 拆解 |
| 尾和公式 | $\mathbb{E}[N]=\sum_{k\ge1}P(N\ge k)$ | 非负整数、等待时间、最大值 |
| 几何等待 | $\mathbb{E}[T]=1/p$ | 独立重复试验直到成功 |
| 容斥 | 加单项、减两两、加三重、交替 | 多事件并集 |
| 一维变换 | $f_Y(y)=f_X(g^{-1}(y))|(g^{-1})'(y)|$ | $Y=g(X)$ |
| 多维变换 | $f_Y(y)=f_X(h(y))|\det(\partial h/\partial y)|$ | 坐标变换、联合密度 |

## 8. 最短记忆版

~~~text
期望题:
  能拆 indicator 就拆 indicator。
  不能拆但变量非负，就试 tail sum。

等待题:
  独立重复成功概率 p -> 1/p。
  状态依赖 -> first-step analysis。

并集题:
  先想补集。
  补集不好算，再用容斥。

密度变换:
  先反解原变量。
  再乘 Jacobian。
~~~
`
  : '';

const systemDesignDbScalingContent = String.raw`# System Design 03 · 数据库扩展三件套

课程位置：[[SystemDesign02 Database Paradigms|02 数据库基本范式]] → 本篇 → [[SystemDesign04 Storage Systems|04 存储系统]]

这篇只从数据库扩展角度讨论三件事：读压力如何分给 replica，写入和容量如何分片，以及两者怎样组合。故障检测、fencing、热备、跨区 RPO/RTO 统一放在 [[SystemDesign05 Reliability Replication|05 可靠性与复制]]，这里不重复讲容灾流程。

## 0. 基础概念：QPS、IOPS、吞吐和延迟

做数据库扩展题之前，先把几个指标说清楚。很多面试回答的问题不是“方案错了”，而是没有先估算系统到底卡在 CPU、网络、磁盘、数据库连接数，还是单机容量。

### 0.1 QPS / RPS / TPS

QPS 是 queries per second，通常表示每秒查询数。RPS 是 requests per second，通常表示服务每秒请求数。TPS 是 transactions per second，常用于数据库事务或支付交易。

它们经常接近，但不完全一样：

| 指标 | 常见含义 | 例子 |
| --- | --- | --- |
| RPS | 服务入口请求数 | API Gateway 每秒收到 10k 个 HTTP request |
| QPS | 查询请求数 | Search service 每秒处理 20k 次 query |
| TPS | 成功事务数 | Payment service 每秒完成 500 笔交易 |
| DB QPS | 数据库查询次数 | 一个 API 请求打 5 次 DB，则 DB QPS 可能是 API RPS 的 5 倍 |

一个常见坑：

~~~text
用户 QPS != 数据库 QPS

1 个 API request
  -> 读 user profile
  -> 读 feature flags
  -> 查订单列表
  -> 写 audit log

入口 RPS = 1
DB operations = 4
DB QPS 约等于 4
~~~

### 0.2 Throughput、Latency 和 Concurrency

Throughput 是单位时间完成多少工作；latency 是单个请求花多久；concurrency 是同一时刻有多少请求在系统内。

三者可以用 Little's Law 做粗估：

$$
\text{concurrency} \approx \text{QPS} \times \text{latency}
$$

注意 latency 要换成秒。

例子：

~~~text
QPS = 10,000 requests/s
平均 latency = 100 ms = 0.1 s

系统内平均并发请求数约为:
10,000 * 0.1 = 1,000
~~~

这说明即使每秒 1 万请求，如果每个请求在系统里停留 100ms，系统同时要承载大约 1000 个 in-flight requests。

记忆图：

~~~mermaid
flowchart LR
  A["QPS: 每秒进来多少"] --> D["Concurrency: 同时在系统里多少"]
  B["Latency: 每个请求待多久"] --> D
  D --> C["线程 / 连接 / 队列 / 内存压力"]
~~~

### 0.3 平均 QPS 和峰值 QPS

日活、月活、请求总量通常只能给平均 QPS。系统设计时要估峰值。

一天有：

$$
24\times 60\times 60 = 86400 \approx 10^5
$$

所以：

$$
\text{avg QPS} \approx \frac{\text{daily requests}}{10^5}
$$

峰值通常可以粗略乘一个系数：

~~~text
peak QPS = avg QPS * peak factor

普通业务: peak factor 3~5
明显潮汐业务: peak factor 5~10
秒杀/热点事件: 可能 10~100+
~~~

例子：

~~~text
每天 1 亿次请求
avg QPS ≈ 100,000,000 / 100,000 = 1,000

如果 peak factor = 5
peak QPS ≈ 5,000
~~~

面试里更重要的是说明假设，而不是死背某个倍数。

### 0.4 IOPS 和磁盘带宽

IOPS 是 input/output operations per second，表示存储系统每秒能处理多少次 I/O 操作。它主要用于估算随机读写压力。

Bandwidth / throughput 表示每秒能传多少数据，常用于大块顺序读写。

| 指标 | 关注点 | 典型瓶颈 |
| --- | --- | --- |
| IOPS | 每秒多少次读写操作 | 小块随机读写、索引 lookup、KV get |
| Bandwidth | 每秒多少 MB/GB | 扫描大文件、备份、日志传输 |
| Latency | 单次 I/O 等多久 | tail latency、同步写路径 |

一个粗略估算：

$$
\text{required IOPS}
\approx
\text{QPS} \times \text{I/O ops per request}
$$

如果每个请求需要 3 次随机读、1 次随机写：

~~~text
API peak QPS = 5,000
I/O per request = 4

required IOPS ≈ 20,000
~~~

如果每个请求还要读取 20KB 数据，那么网络或磁盘带宽约为：

$$
\text{bandwidth} \approx \text{QPS} \times \text{bytes per request}
$$

~~~text
5,000 QPS * 20 KB ≈ 100 MB/s
~~~

这两个估算回答的是不同问题：

~~~text
小对象随机读很多:
  看 IOPS

大对象连续读很多:
  看 bandwidth
~~~

### 0.5 常见容量估算模板

#### 存储容量

~~~text
daily data = daily writes * average record size
retention storage = daily data * retention days * replication factor
~~~

例子：

~~~text
每天 1 亿条 event
每条 500 bytes
保留 30 天
3 副本

raw daily data = 100,000,000 * 500B = 50GB/day
total storage ≈ 50GB * 30 * 3 = 4.5TB
~~~

#### 数据库读写拆分

~~~text
read QPS = total QPS * read ratio
write QPS = total QPS * write ratio
replica count ≈ read QPS / safe read QPS per replica
~~~

例子：

~~~text
peak QPS = 20,000
读写比 = 90% read, 10% write

read QPS = 18,000
write QPS = 2,000

如果单个 replica 安全承载 4,000 read QPS
至少需要 5 个 read replicas
~~~

#### Cache 命中后端压力

~~~text
backend QPS = total QPS * (1 - cache hit rate)
~~~

例子：

~~~text
total QPS = 100,000
cache hit rate = 95%

backend QPS = 100,000 * 5% = 5,000
~~~

这就是为什么高 QPS 系统里，cache hit rate 从 95% 掉到 90% 会很严重：后端压力直接翻倍。

#### 队列和 worker 数

如果任务平均处理时间是 $T$ 秒，每个 worker 一次处理一个任务，那么单 worker 吞吐约为：

$$
\text{worker throughput} \approx \frac{1}{T}
$$

所需 worker 数：

$$
\text{workers} \approx \text{arrival QPS} \times T
$$

例子：

~~~text
每秒进入 200 个任务
每个任务平均处理 0.5 秒

需要并发 worker ≈ 200 * 0.5 = 100
~~~

### 0.6 面试里怎么用这些数字

系统设计里，估算不是为了精确，而是为了决定架构方向。

~~~mermaid
flowchart TD
  A["估 QPS / storage / bandwidth / IOPS"] --> B{"单机能否承受"}
  B -->|读压力大| C["replica / cache / read pool"]
  B -->|写压力大| D["partition / queue / batch"]
  B -->|容量大| E["sharding / cold storage / retention"]
  B -->|延迟高| F["index / cache / async / locality"]
  B -->|峰值高| G["autoscale / rate limit / backpressure"]
~~~

一个比较稳的回答顺序：

~~~text
1. 先估入口 QPS 和峰值 QPS。
2. 再估每个请求会打多少 DB / cache / storage。
3. 把入口 QPS 转成后端 QPS、IOPS 和 bandwidth。
4. 判断读瓶颈、写瓶颈、容量瓶颈还是延迟瓶颈。
5. 再选择复制、分片、缓存、队列或异步化。
~~~

---

## 0.7 先判断压力来自哪里

数据库扩展题不要一上来就说“加缓存”或“上分片”。先判断系统瓶颈：

~~~mermaid
flowchart TD
  A["数据库压力"] --> B{"主要压力是什么"}
  B -->|读请求太多| C["主从复制 + 读写分离"]
  B -->|主库不可用风险| D["主主 / 主备 / 自动故障切换"]
  B -->|写请求太多| E["数据分区 / Sharding"]
  B -->|数据量太大| E
  C --> F["代价: replication lag / stale read"]
  D --> G["代价: 冲突处理 / failover 复杂"]
  E --> H["代价: 跨分片查询 / rebalancing / hot shard"]
~~~

可以先记住一句：

| 模式 | 解决什么 | 不解决什么 |
| --- | --- | --- |
| 主从复制 | 读扩展、为故障接管保留副本 | 不扩展主库写入能力，也不代替 backup |
| 主主复制 | 多个写入口或更灵活的接管 | 不让写入能力线性翻倍，还会引入冲突 |
| 数据分区 | 容量扩展、写入扩展、索引变小 | 增加查询路由和跨分片复杂度 |

---

## 1. 主从复制：这里先解决读扩展

主从复制的基本结构是：

~~~text
Primary / Master  ->  Replica / Slave
~~~

主库接收写入，从库复制主库数据。所有会修改数据的操作都进入主库：

~~~text
INSERT
UPDATE
DELETE
CREATE TABLE
ALTER TABLE
~~~

从库通常不直接接收业务写入，而是跟随主库的变更日志更新本地数据。

### 1.1 复制链路怎么工作

以 MySQL 为例，主从复制的核心是 binlog。主库执行数据修改后，会把变更写进 binary log；从库持续拉取主库 binlog 的增量，写入 relay log，再在本地重放这些操作。

~~~mermaid
sequenceDiagram
  participant C as Client
  participant P as Primary
  participant B as Binlog
  participant R as Replica
  participant L as Relay Log

  C->>P: write request
  P->>P: execute mutation
  P->>B: append change event
  R->>B: pull changes after known position
  R->>L: write relay log
  R->>R: replay relay log
~~~

本质上是三步：

~~~text
主库记录变更；
从库拉取变更；
从库本地重放变更。
~~~

### 1.2 主从复制的用途

主从复制最常见的用途有四个。

第一，读写分离。写请求走主库，读请求分摊到多个从库：

~~~text
Write  -> Primary
Read   -> Replica 1 / Replica 2 / Replica 3
~~~

例如：

| 请求 | 路由 |
| --- | --- |
| 用户浏览商品 | Replica |
| 用户查看订单列表 | Replica |
| 用户修改地址 | Primary |
| 用户下单付款 | Primary |

这适合读多写少系统。需要强调：主从复制扩展的是读能力，不是写能力。

第二，查询隔离。不同从库可以承担不同类型的读任务：

~~~text
Replica 1: 线上普通查询
Replica 2: 报表查询
Replica 3: 备份任务
~~~

这样慢查询、报表、备份不直接拖垮主库。

第三，零停机备份。主库继续服务线上请求，从库执行备份任务。

第四，从库故障转移。某个从库挂了，可以从读流量池摘掉，请求转向其他副本。

### 1.3 核心代价：复制延迟

主从复制通常是异步的。主库完成写入后可以先返回成功，不必等待所有从库都重放完成。

这会带来 stale read：

~~~text
用户把昵称 Alice 改成 Bob
  -> 写入 Primary 成功
  -> 用户立刻刷新页面
  -> 读请求被路由到 Replica
  -> Replica 还没同步
  -> 页面仍显示 Alice
~~~

所以主从复制的核心 trade-off 是：

~~~text
提高读取能力和可用性；
但读副本可能短暂落后。
~~~

常见处理方式：

| 场景 | 策略 |
| --- | --- |
| 刚写完立刻读自己的数据 | read-your-writes：短时间强制读主库 |
| 可接受短暂旧数据 | 读从库 |
| 强一致关键路径 | 写主库后读主库，或使用同步复制/多数派协议 |
| 副本落后严重 | 从读池摘掉落后 replica |

---

## 2. 主主复制：理解写拓扑，不把它当写入翻倍

主从复制里，主库是唯一写入口。主库挂了以后，需要选新主库、切流量、处理旧主库恢复后的状态。主主复制把两个节点都做成可写节点：

~~~text
Primary A  <->  Primary B
~~~

A 的写入复制到 B，B 的写入也复制到 A。这通常用于双机热备。

~~~mermaid
flowchart LR
  C["Client"] --> A["Primary A"]
  C --> B["Primary B"]
  A -->|replicate changes| B
  B -->|replicate changes| A
~~~

### 2.1 如何避免复制循环

如果 A 的写入复制到 B，B 又原样复制回 A，就会无限循环。MySQL 复制里每台服务器有 server-id，变更日志会记录事件来源。

因此：

~~~text
A 产生事件 e
  -> B 收到 e
  -> B 看到 e 来自 A
  -> B 不再把 e 当作自己的新事件复制回 A
~~~

### 2.2 为什么它不等于写扩展

主主复制看起来有两个主库，但并不意味着写入能力翻倍。原因是两个节点最终都要保存完整数据集，也都要执行对方复制来的写入。

写到 A：

~~~text
Client -> A
A 执行写入
A 写 binlog
B 拉取并重放
~~~

写到 B：

~~~text
Client -> B
B 执行写入
B 写 binlog
A 拉取并重放
~~~

最终 A 和 B 都要承担完整数据、完整索引、完整存储和复制写入。因此主主复制更适合被理解为高可用方案，而不是水平写扩展方案。

### 2.3 主主复制的代价

主主复制会引入：

~~~text
双边都保存完整数据；
双边都执行所有写入；
复制增加磁盘和网络 I/O；
两边同时写同一行可能冲突；
failover 和旧主恢复更复杂。
~~~

如果业务真的需要写扩展，通常要进入数据分区，而不是只靠两个 master。

---

## 3. 数据分区：用 Sharding 扩展容量和写入

复制保存多份相同数据；分区保存不同数据。

~~~text
Replication: 每台机器有完整副本
Sharding: 每台机器只保存一部分数据
~~~

如果单库数据太大，或者单个主库写入压力太高，就需要分片。

### 3.1 基本思想

假设用户表按 user_id 分为 4 个 shard：

~~~text
user_id % 4 = 0  ->  Shard 0
user_id % 4 = 1  ->  Shard 1
user_id % 4 = 2  ->  Shard 2
user_id % 4 = 3  ->  Shard 3
~~~

访问 user_id = 123：

~~~text
123 % 4 = 3
访问 Shard 3
~~~

这样每台机器只负责一部分用户，容量和写入压力都会被分散。

### 3.2 Sharding key 是设计核心

好的 sharding key 需要满足三件事。

第一，查询时经常带这个 key。

如果大部分查询是：

~~~sql
SELECT * FROM orders WHERE user_id = ?
~~~

那么 user_id 很自然。如果 sharding key 在查询里很少出现，就只能广播到所有 shard。

第二，分布要均匀。

按国家分片可能导致 US shard 过热；hash(user_id) 通常更均匀。

第三，减少跨分片查询。

理想查询：

~~~text
定位 shard -> 查询 -> 返回
~~~

跨分片查询：

~~~mermaid
flowchart TD
  Q["Query"] --> S0["Shard 0"]
  Q --> S1["Shard 1"]
  Q --> S2["Shard 2"]
  Q --> S3["Shard 3"]
  S0 --> M["Merge / Sort / Aggregate"]
  S1 --> M
  S2 --> M
  S3 --> M
  M --> R["Response"]
~~~

跨分片 join、跨分片事务、全局排序都会明显变复杂。

### 3.3 分片的收益和代价

收益：

~~~text
数据量分散到多台机器；
写入压力分散到多个 shard；
每个节点维护更小索引；
单机存储和内存压力下降；
可以通过增加 shard 扩容。
~~~

代价：

~~~text
跨分片查询复杂；
跨分片事务复杂；
全局唯一 ID 需要设计；
扩容和数据迁移困难；
应用层要处理路由、重试和部分失败。
~~~

一句话：

~~~text
复制让相同数据有更多副本；
分片让不同机器承担不同数据。
~~~

---

## 4. 复制和分片通常一起用

真实系统里经常是每个 shard 内部再做主从复制：

~~~mermaid
flowchart TD
  Router["Query Router"] --> S0P["Shard 0 Primary"]
  Router --> S1P["Shard 1 Primary"]
  Router --> S2P["Shard 2 Primary"]

  S0P --> S0R1["Shard 0 Replica"]
  S0P --> S0R2["Shard 0 Replica"]
  S1P --> S1R1["Shard 1 Replica"]
  S1P --> S1R2["Shard 1 Replica"]
  S2P --> S2R1["Shard 2 Replica"]
  S2P --> S2R2["Shard 2 Replica"]
~~~

这样同时获得：

~~~text
分片带来的容量和写入扩展；
复制带来的读扩展、备份和高可用。
~~~

总结：

~~~text
复制解决多读、多副本、高可用；
分片解决大数据量、高写入、单机容量瓶颈。
~~~

---

## 5. Feature Store 里的对应设计

Feature Store 可以理解为给模型服务提供在线特征的分布式状态系统。

在线预测时，模型不只需要当前请求字段，还需要历史上下文，例如：

| 场景 | 需要的特征 |
| --- | --- |
| 风控 | 用户过去 5 分钟交易次数、设备关联用户数、商户拒付率 |
| 推荐 | 用户最近点击、物品曝光点击统计、用户物品交互历史 |
| 广告 | 用户兴趣、广告主预算状态、实时点击率 |

这些特征不能在请求时临时扫描日志计算，通常要提前 materialize 到 Online Feature Store。

### 5.1 Feature Store 中的复制

类比数据库：

| 数据库 | Feature Store |
| --- | --- |
| Primary 接收写入 | feature primary 接收特征更新 |
| Replica 复制数据 | feature replica 复制特征状态 |
| 应用读副本 | model serving 读副本 |

写入路径：

~~~text
feature computation / materialization -> feature primary
~~~

读取路径：

~~~text
model serving / feature service -> feature replicas
~~~

这就是特征系统里的读写分离：特征计算负责写，模型服务负责读。

### 5.2 Feature Store 中的 stale feature

数据库里有 stale read，Feature Store 里有 stale feature。

~~~text
用户刚连续支付失败 5 次；
风险特征应该升高；
feature primary 已更新；
replica 尚未同步；
模型从 replica 读到旧特征；
风险被低估。
~~~

因此需要监控 freshness：

~~~text
特征最后更新时间；
特征落后多久；
是否超过模型可接受延迟；
哪些 replica 已经落后。
~~~

不同特征 freshness 要求不同：

| 特征类型 | 常见 freshness |
| --- | --- |
| 风控短窗口特征 | 秒级到几十秒 |
| 推荐行为特征 | 分钟级 |
| 商户长期统计 | 小时级 |
| 用户画像 | 天级 |

### 5.3 Feature Store 中的数据分区

Feature Store 的 sharding key 通常是 entity key：

~~~text
user_id
item_id
merchant_id
device_id
tenant_id
session_id
~~~

例如：

| Feature group | Sharding key |
| --- | --- |
| user_features | user_id |
| item_features | item_id |
| merchant_features | merchant_id |
| device_features | device_id |
| user_item_features | hash(user_id, item_id) |

一次风控请求可能需要：

~~~text
user_features:user_id=123
merchant_features:merchant_id=888
device_features:device_id=abc
user_merchant_features:user_id=123,merchant_id=888
~~~

Feature Service 必须能根据请求里的 key 直接定位 shard，不能每次都广播所有节点。

### 5.4 Feature Store 分片的代价

第一，跨 entity 特征不适合在线临时聚合。

例如：

~~~text
某城市过去 1 小时所有用户的平均交易金额；
某品类最近 30 分钟整体点击率；
全站最近 10 分钟支付失败率。
~~~

这些通常要通过 batch 或 streaming job 提前算好，再写回 Online Feature Store。

第二，hot key 会导致负载不均衡：

~~~text
超级热门商品；
大型商户；
超大企业客户；
高活跃用户。
~~~

hot key 的本质是：分片规则可能平均，但访问流量不平均。一个热门商品、一个大商户、一个超活跃用户可能把某个 shard 或 replica 打满。

处理方式可以先分成读路径和更新路径。

读路径上，常见办法是：

~~~text
增加 replica；
加缓存；
热点 key 特殊拆分；
热门 item feature 预加载到模型服务本地；
对多次读取做 batch 和合并。
~~~

更新路径上，面试里可以讲 push 模式和 pull 模式。

| 模式 | 怎么做 | 适合场景 | 代价 |
| --- | --- | --- | --- |
| Push / active update | 上游 feature computation 产出新值后，主动把 hot feature 推到 cache / serving replica / local cache | 热点少、更新频率可控、freshness 要求高 | 写放大；需要 fanout、版本号和失败重试 |
| Pull / lazy update | serving 侧读到缺失或过期 feature 时，再去 feature store / source of truth 拉取并刷新本地缓存 | 热点变化快、长尾 key 多、允许短暂 stale | 第一次 miss 慢；需要 TTL、singleflight、防止 cache stampede |

可以这样理解：

~~~mermaid
flowchart LR
  A["Feature Update"] --> B{"更新模式"}
  B -->|Push active| C["主动刷新 hot cache / replica"]
  B -->|Pull lazy| D["请求 miss / TTL 过期时再刷新"]
  C --> E["低读延迟 + 更高写放大"]
  D --> F["低写放大 + miss 时延迟更高"]
~~~

在推荐系统里，热门 item feature 往往适合 push 到 serving local cache；用户长尾特征更适合 pull + TTL，因为主动推所有用户特征会造成大量无效写入。

第三，多类特征来自不同 shard，更新时间可能不同。模型拿到的通常是近似一致的特征快照，而不是严格同一时刻的全局状态。

### 5.5 更新日志和 checkpoint

MySQL 主从复制依赖 binlog position。Feature Store 也有类似的增量同步位置：

~~~text
streaming job 处理到 Kafka offset X；
batch materialization 处理到某个时间分区；
feature group v7 同步到 checkpoint Y。
~~~

更新链路可以横向记：

~~~mermaid
flowchart LR
  A["Raw Events"] --> B["Feature Computation"]
  B --> C["Feature Update Log / Checkpoint"]
  C --> D["Online Feature Store Primary"]
  D --> E["Online Feature Store Replicas"]
  E --> F["Model Serving"]
~~~

这个 update log / checkpoint 在概念上类似数据库里的 binlog position：不是每次全量同步，而是记录处理进度，持续增量更新。

---

## 6. 面试回答模板

如果题目问“数据库怎么扩展”，可以按这个顺序回答：

~~~text
1. 先判断瓶颈：读多、写多、数据大、还是可用性问题。
2. 读多：主从复制 + 读写分离，但要处理 replication lag。
3. 主库故障恢复：主备或主主，重点是 failover，不是写扩展。
4. 写多或数据大：按业务访问模式选择 sharding key。
5. 分片后要讨论跨分片查询、事务、全局 ID、迁移和热点。
6. 真实系统通常是 shard 内复制，复制和分片组合使用。
~~~

展开回答时可以按这几个层次讲：

| 层次 | 要说清楚什么 | 常见追问 |
| --- | --- | --- |
| 负载估算 | 入口 QPS、读写比、峰值系数、单请求 DB/Cache 次数 | 平均 QPS 和峰值 QPS 差多少？ |
| 复制 | 主从复制扩读，主备/主主解决 failover | replication lag 怎么处理？ |
| 分片 | sharding key 贴近主查询路径，避免广播查询 | hot shard、跨分片 join、全局 ID 怎么办？ |
| 一致性 | 哪些读可以 stale，哪些必须 read-your-writes | 下单、支付、库存这类路径能不能读从库？ |
| 运维 | rebalancing、backup、schema migration、observability | 扩 shard 时怎么迁移数据？ |

如果题目是 Feature Store / Online KV / Embedding Store：

~~~text
1. 把它看成服务模型的分布式状态系统。
2. entity key 决定分片。
3. replica 承担低延迟读和高可用。
4. freshness 等价于 ML 系统里的 replication lag。
5. hot key 要拆读路径和更新路径：cache/replica/push/pull。
6. update log / checkpoint 决定增量同步和故障恢复能力。
~~~

面试里不要只说“加缓存”。更好的表述是：

~~~text
我会先估读写压力和数据规模。
如果主要是读压力，用 replica 和 cache；
如果主要是写压力或容量压力，用 sharding；
如果是可用性问题，用 failover 和复制；
如果是 Feature Store，还要额外讨论 freshness、hot key、update log 和 checkpoint。
~~~

### 6.1 选择题：读扩展应该先想到什么？

~~~quiz
title: Database Scaling Check 1
question: 一个服务读请求很多、写请求相对少，最直接的数据库扩展手段是什么？
answer: B
A. 主主复制，因为两个主库可以把所有写入吞吐翻倍
B. 主从复制加读写分离，让多个 replica 承担读请求
C. 立刻按随机 key 分片，不考虑查询路径
D. 把所有请求都改成异步队列
explanation: 读多写少时，主从复制和读写分离通常是第一步；它扩展读能力，但不扩展单主写能力。
~~~

### 6.2 选择题：Feature Store 的 hot key 怎么分析？

~~~quiz
title: Feature Store Check 1
question: 一个热门 item 的 feature 被大量请求读取，同时该 feature 更新不算频繁。哪种说法更合理？
answer: C
A. 只能把这个 item 按 item_id 重新 hash 到另一个 shard
B. 必须每次请求都从 source of truth 读取，避免 stale
C. 可以把热门 item feature push 到 serving local cache 或更多 replica，降低读路径压力
D. hot key 只影响写入，不影响读取
explanation: 热门 item 的问题主要是读流量集中。若 freshness 要求较高且热点集合较小，push/active update 到本地 cache 或 replica 很合适。
~~~

### 6.3 选择题：update log / checkpoint 解决什么？

~~~quiz
title: Feature Store Check 2
question: Feature Store 里的 update log / checkpoint 最核心的作用是什么？
answer: B
A. 让每次同步都重新扫描全量历史数据
B. 记录处理进度，支持增量更新、故障恢复和判断副本落后程度
C. 替代 sharding key，让查询不用路由
D. 保证所有 feature 在严格同一时刻更新
explanation: update log / checkpoint 类似 binlog position 或 Kafka offset，核心是记录已经处理到哪里，从而持续增量同步和恢复。
~~~

## 7. 最短记忆版

~~~text
主从复制:
  读扩展 + 备份 + 容灾
  代价是 stale read

主主复制:
  高可用 + 快速切换
  不是写入翻倍

分片:
  容量扩展 + 写扩展
  代价是跨分片复杂

Feature Store:
  传统数据库扩展思想在 ML online state 上的复用
~~~
`;

const mlsysNoteDefinitions = [
  createTutorialDefinition('MLSYS1 · GPU 体系结构入门', 'MLSYS1.md', 'MLSYS1.en.md', {
    titleEn: 'MLSYS1 · GPU Architecture Basics',
  }),
  createTutorialDefinition('MLSYS2 · CUDA 编程模型与 GPU 组件', 'MLSYS2.md', 'MLSYS2.en.md', {
    titleEn: 'MLSYS2 · CUDA Programming Model & GPU Architecture',
  }),
  createTutorialDefinition('MLSYS3 · Roofline Analysis', 'MLSYS3.md', 'MLSYS3.en.md', {
    titleEn: 'MLSYS3 · Roofline Analysis',
  }),
  createTutorialDefinition('MLSYS4 · CUDA Reduce Kernel 完全指南', 'MLSYS4.md', 'MLSYS4.en.md', {
    titleEn: 'MLSYS4 · Complete Guide to CUDA Reduce Kernels',
  }),
  createTutorialDefinition('MLSYS5 · CUDA Parallel Primitives: Histogram & Scan', 'MLSYS5.md', 'MLSYS5.en.md', {
    titleEn: 'MLSYS5 · CUDA Parallel Primitives: Histogram & Scan',
  }),
  createTutorialDefinition('MLSYS6 · Memory-Bound Kernel 优化', 'MLSYS6.md', 'MLSYS6.en.md', {
    titleEn: 'MLSYS6 · Memory-Bound Kernel Optimization',
  }),
  createTutorialDefinition(
    'MLSYS7 · Compute-Bound Kernel (1)',
    'MLSYS7 Compute-Bound Kernel (1).md',
    'MLSYS7 Compute-Bound Kernel (1).en.md',
    { titleEn: 'MLSYS7 · Compute-Bound Kernel (1)' },
  ),
  createTutorialDefinition(
    'MLSYS8 · Compute-Bound Kernel (2)',
    'MLSYS8 Compute-Bound Kernel (2).md',
    'MLSYS8 Compute-Bound Kernel (2).en.md',
    { titleEn: 'MLSYS8 · Compute-Bound Kernel (2)' },
  ),
  createTutorialDefinition(
    'MLSYS9 · Compute-Bound Kernel (3)',
    'MLSYS9 Compute-bound kernel (3).md',
    'MLSYS9 Compute-bound kernel (3).en.md',
    { titleEn: 'MLSYS9 · Compute-Bound Kernel (3)' },
  ),
  createTutorialDefinition('MLSYS10 · 分布式训练并行范式', 'MLSYS10 parallelism.md', 'MLSYS10 parallelism.en.md', {
    titleEn: 'MLSYS10 · Distributed Training Parallelism Paradigms',
  }),
  createTutorialDefinition('MLSYS11 · nano-vllm 精读 (1)', 'MLSYS11 nano-vllm-1.md', 'MLSYS11 nano-vllm-1.en.md', {
    titleEn: 'MLSYS11 · nano-vllm Code Walkthrough (1)',
  }),
  createTutorialDefinition('MLSYS12 · nano-vllm 精读 (2)', 'MLSYS12 nano-vllm-2.md', 'MLSYS12 nano-vllm-2.en.md', {
    titleEn: 'MLSYS12 · nano-vllm Code Walkthrough (2)',
  }),
  createTutorialDefinition(
    'MLSYS13 · Low-bit Quantization 核心方法详解',
    'MLSYS13 Quantization and precision.md',
    'MLSYS13 Quantization and precision.en.md',
    { titleEn: 'MLSYS13 · Low-Bit Quantization Methods' },
  ),
  createTutorialDefinition(
    'MLSYS14 · Post-Training Infra：从 TRL 到 Forge',
    'MLSYS14 Post-Training Infra.md',
    'MLSYS14 Post-Training Infra.en.md',
    { titleEn: 'MLSYS14 · Post-Training Infra: TRL to Forge' },
  ),
  createTutorialDefinition(
    'MLSYS15 · Efficient Attention：现代长上下文架构',
    'MLSYS15 Efficient Attention Modern Architectures.md',
    'MLSYS15 Efficient Attention Modern Architectures.en.md',
    { titleEn: 'MLSYS15 · Efficient Attention: Modern Long-Context Architectures' },
  ),
  createTutorialDefinition(
    'MLSYS16 · KV Cache：内存管理与前缀复用',
    'MLSYS15 KV Cache Prefix Caching IndexShare.md',
    'MLSYS15 KV Cache Prefix Caching IndexShare.en.md',
    { titleEn: 'MLSYS16 · KV Cache: Memory Management & Prefix Reuse' },
  ),
  createTutorialDefinition(
    'MLSYS17 · Inference：并行解码与草稿验证',
    'MLSYS15 LLM Inference Speculative Decoding DFlash.md',
    'MLSYS15 LLM Inference Speculative Decoding DFlash.en.md',
    { titleEn: 'MLSYS17 · LLM Inference: Speculative Decoding & Verification' },
  ),
  createTutorialDefinition(
    'MLSYS18 · MoE Systems：路由、通信与 Kernel',
    'MLSYS16 Modern MoE SonicMoE.md',
    'MLSYS16 Modern MoE SonicMoE.en.md',
    { titleEn: 'MLSYS18 · MoE Systems: Routing, Communication & Kernels' },
  ),
];

const mlsysNotes = mlsysNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const leetcodeNoteDefinitions = [
  createTutorialDefinition(
    'Core Skills 1 · Design Dynamic Array',
    'CoreSkills01 Design Dynamic Array.md',
    'CoreSkills01 Design Dynamic Array.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 1 · Design Dynamic Array', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 2 · Linked Lists',
    'CoreSkills02 Design Singly Linked List.md',
    'CoreSkills02 Design Singly Linked List.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 2 · Linked Lists', category: 'Linked Lists', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 3 · Design Double-ended Queue',
    'CoreSkills03 Design Double-ended Queue.md',
    'CoreSkills03 Design Double-ended Queue.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 3 · Design Double-ended Queue', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 4 · Trees',
    'CoreSkills04 Design Binary Search Tree.md',
    'CoreSkills04 Design Binary Search Tree.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 4 · Trees', category: 'Trees', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 5 · Design Hash Table',
    'CoreSkills05 Design Hash Table.md',
    'CoreSkills05 Design Hash Table.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 5 · Design Hash Table', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 6 · Heap & Priority Queue',
    'CoreSkills06 Design Heap.md',
    'CoreSkills06 Design Heap.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 6 · Heap & Priority Queue', category: 'Heap', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 7 · Graphs',
    'CoreSkills07 Design Graph.md',
    'CoreSkills07 Design Graph.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 7 · Graphs', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 8 · Design Segment Tree',
    'CoreSkills08 Design Segment Tree.md',
    'CoreSkills08 Design Segment Tree.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 8 · Design Segment Tree', category: 'Implement Data Structures', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 9 · Sorting Algorithms',
    'CoreSkills09 Insertion Sort.md',
    'CoreSkills09 Insertion Sort.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 9 · Sorting Algorithms', category: 'Sorting', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 10 · Dynamic Programming',
    'CoreSkills10 Decode Ways Dynamic Programming.md',
    'CoreSkills10 Decode Ways Dynamic Programming.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 10 · Dynamic Programming', category: 'Dynamic Programming', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 11 · Rejection Sampling / Rand10',
    'CoreSkills11 Rejection Sampling Rand10.md',
    'CoreSkills11 Rejection Sampling Rand10.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 11 · Rejection Sampling / Rand10', category: 'Math & Probability', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 12 · Greedy Algorithms',
    'CoreSkills12 Greedy Algorithms.md',
    'CoreSkills12 Greedy Algorithms.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 12 · Greedy Algorithms', category: 'Greedy', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 13 · Interval Problems',
    'CoreSkills13 Interval Problems.md',
    'CoreSkills13 Interval Problems.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 13 · Interval Problems', category: 'Intervals', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 14 · Math: Fast Power',
    'CoreSkills14 Math Binary Exponentiation.md',
    'CoreSkills14 Math Binary Exponentiation.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 14 · Math: Fast Power', category: 'Math', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 15 · Bit Manipulation: XOR',
    'CoreSkills15 Bit Manipulation XOR.md',
    'CoreSkills15 Bit Manipulation XOR.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 15 · Bit Manipulation: XOR', category: 'Math', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 16 · String Basics',
    'CoreSkills16 String Basics Encode Decode.md',
    'CoreSkills16 String Basics Encode Decode.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 16 · String Basics', category: 'Strings', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 17 · Two Pointers',
    'CoreSkills17 Two Pointers.md',
    'CoreSkills17 Two Pointers.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 17 · Two Pointers', category: 'Two Pointers', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 18 · Sliding Window',
    'CoreSkills18 Sliding Window.md',
    'CoreSkills18 Sliding Window.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 18 · Sliding Window', category: 'Sliding Window', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 19 · Stack & Monotonic Stack',
    'CoreSkills19 Stack MinStack Monotonic Stack.md',
    'CoreSkills19 Stack MinStack Monotonic Stack.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 19 · Stack & Monotonic Stack', category: 'Stacks', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 20 · Binary Search',
    'CoreSkills20 Binary Search.md',
    'CoreSkills20 Binary Search.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 20 · Binary Search', category: 'Binary Search', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 21 · Tries',
    'CoreSkills21 Design Trie.md',
    'CoreSkills21 Design Trie.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 21 · Tries', category: 'Tries', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 22 · Backtracking',
    'CoreSkills22 Backtracking.md',
    'CoreSkills22 Backtracking.en.md',
    { directory: 'Leetcode', titleEn: 'Core Skills 22 · Backtracking', category: 'Backtracking', difficulty: 'Medium' },
  ),
];

const leetcodeNotes = leetcodeNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const draftNoteDefinitions = isDraftMode
  ? [
      createDraftTutorialDefinition(
        'LLM八股 Overview · JD 高频主题拆解',
        'Draft LLM Interview Overview.md',
        llmDraftOverviewContent,
        { titleEn: 'LLM Interview Overview · Job Description Key Themes' },
      ),
      createDraftTutorialDefinition(
        'Quant 草稿 · 概率基础公式与记忆框架',
        'Draft Probability Basics.md',
        probabilityDraftContent,
        { titleEn: 'Quant Draft · Probability Formulas & Memory Framework' },
      ),
    ]
  : [];

const draftNotes = draftNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createInlineVariant(definition.zhFileName, definition.content),
    en: createInlineVariant('', undefined),
  },
}));

const quantNoteDefinitions = [
  createTutorialDefinition(
    'Quant 1 · 期望与计数：指示变量、前缀极值与多项分布',
    'Quant01 Expectation Counting Multinomial.md',
    'Quant01 Expectation Counting Multinomial.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 1 · Expectation & Counting: Indicator Variables & Multinomial',
      category: 'Expectation & Counting',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Quant 2 · Markov Chains：状态压缩与期望时间',
    'Quant02 Markov Chains Expected Time.md',
    'Quant02 Markov Chains Expected Time.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 2 · Markov Chains: State Compression & Expected Time',
      category: 'Markov',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Quant 3 · 连续分布：CDF、几何区域与变量变换',
    'Quant03 Continuous Distribution Geometry Transform.md',
    'Quant03 Continuous Distribution Geometry Transform.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 3 · Continuous Distributions: CDF, Geometry & Transformations',
      category: 'Distribution',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Quant 4 · 协方差、相关系数与相关矩阵 PSD',
    'Quant04 Correlation Matrix PSD.md',
    'Quant04 Correlation Matrix PSD.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 4 · Covariance, Correlation & PSD Matrices',
      category: 'Probability',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Quant 5 · 正态分布：二维正态、Cholesky 与符号相关',
    'Quant05 Normal Sign Correlation.md',
    'Quant05 Normal Sign Correlation.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 5 · Normal Distributions: Bivariate, Cholesky & Sign Correlation',
      category: 'Normal Distribution',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Quant 6 · 高维积分：大数定律与控制收敛',
    'Quant06 High Dimensional Integral Dominated Convergence.md',
    'Quant06 High Dimensional Integral Dominated Convergence.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 6 · High-Dimensional Integrals: LLN & Dominated Convergence',
      category: 'Analysis & Probability',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Quant 7 · 递推法：健忘乘客与状态压缩',
    'Quant07 Recursion Absent-Minded Passenger.md',
    'Quant07 Recursion Absent-Minded Passenger.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 7 · Recursion: Absent-Minded Passenger & State Compression',
      category: 'Recursion',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Quant 8 · 顺序统计量：CDF 求导与条件截断',
    'Quant08 Order Statistics Conditional Truncation.md',
    'Quant08 Order Statistics Conditional Truncation.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 8 · Order Statistics: CDF Derivatives & Conditional Truncation',
      category: 'Order Statistics',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Quant 9 · 假设检验与最大似然估计：方向、边界与偏差方差',
    'Quant09 Hypothesis Testing Maximum Likelihood.md',
    'Quant09 Hypothesis Testing Maximum Likelihood.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 9 · Hypothesis Testing & MLE: Boundaries & Bias-Variance',
      category: 'Estimation & Testing',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Quant 10 · 风险中性定价与最优下注策略：鞅论、Problem of Points 与效用函数',
    'Quant10 Betting Risk Neutral Pricing Martingales.md',
    'Quant10 Betting Risk Neutral Pricing Martingales.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 10 · Risk-Neutral Pricing & Optimal Betting: Martingales & Utility',
      category: 'Martingales & Betting',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Quant 11 · 鞅、停时与随机游走：Wald 等式、鞅构造与最优时停',
    'Quant11 Martingales Stopping Times Random Walks.md',
    'Quant11 Martingales Stopping Times Random Walks.en.md',
    {
      directory: 'quant',
      titleEn: "Quant 11 · Martingales, Stopping Times & Random Walks: Wald's Identity, Martingale Construction & Optimal Stopping",
      category: 'Martingales & Random Walks',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Quant 12 · 布朗运动、伊藤微积分、停时与期权交易应用',
    'Quant12 Brownian Motion Ito Calculus Stopping Times and Options.md',
    'Quant12 Brownian Motion Ito Calculus Stopping Times and Options.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 12 · Brownian Motion, Itô Calculus, Stopping Times & Option Trading',
      category: 'Stochastic Calculus & Trading',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Quant 13 · 博弈论与策略性决策：纳什均衡、逆向归纳与华尔街量化经典',
    'Quant13 Game Theory and Strategic Decision Making.md',
    'Quant13 Game Theory and Strategic Decision Making.en.md',
    {
      directory: 'quant',
      titleEn: 'Quant 13 · Game Theory & Strategic Decision Making: Nash Equilibrium, Backward Induction & Wall Street Classics',
      category: 'Game Theory & Strategy',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'C++ 面经 1 · 面向对象基础与类设计',
    'QuantDevCPP01 OOP Fundamentals Class Design.md',
    null,
    {
      directory: 'quant',
      titleEn: 'C++ Interview 1 · OOP Fundamentals & Class Design',
      category: 'C++ 面经',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'C++ 面经 2 · 内存模型与指针陷阱',
    'QuantDevCPP02 Memory Model Pointers.md',
    null,
    {
      directory: 'quant',
      titleEn: 'C++ Interview 2 · Memory Model & Pointer Pitfalls',
      category: 'C++ 面经',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'C++ 面经 3 · 多态、虚函数与关键字',
    'QuantDevCPP03 Polymorphism Virtual Functions Keywords.md',
    null,
    {
      directory: 'quant',
      titleEn: 'C++ Interview 3 · Polymorphism, Virtual Functions & Keywords',
      category: 'C++ 面经',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'C++ 面经 4 · 现代 C++：移动语义与智能指针',
    'QuantDevCPP04 Move Semantics Smart Pointers.md',
    null,
    {
      directory: 'quant',
      titleEn: 'C++ Interview 4 · Modern C++: Move Semantics & Smart Pointers',
      category: 'C++ 面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'C++ 面经 5 · STL 容器与底层实现',
    'QuantDevCPP05 STL Containers Internals.md',
    null,
    {
      directory: 'quant',
      titleEn: 'C++ Interview 5 · STL Containers & Internals',
      category: 'C++ 面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'C++ 面经 6 · 手撕代码与高级数据结构',
    'QuantDevCPP06 Coding Practice Advanced Data Structures.md',
    null,
    {
      directory: 'quant',
      titleEn: 'C++ Interview 6 · Coding Practice & Advanced Data Structures',
      category: 'C++ 面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    '性能优化 1 · 性能分析方法论与工具链',
    'QuantDevPerf01 Profiling Methodology Toolchain.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Perf Optimization 1 · Profiling Methodology & Toolchain',
      category: '性能优化面经',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    '性能优化 2 · 多线程与 CPU 优化',
    'QuantDevPerf02 Multithreading CPU Optimization.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Perf Optimization 2 · Multithreading & CPU Optimization',
      category: '性能优化面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    '性能优化 3 · 内存性能优化',
    'QuantDevPerf03 Memory Optimization.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Perf Optimization 3 · Memory Performance Optimization',
      category: '性能优化面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    '性能优化 4 · Coredump 配置、产生与分析',
    'QuantDevPerf04 Coredump Configuration Analysis.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Perf Optimization 4 · Coredump Configuration & Analysis',
      category: '性能优化面经',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    '操作系统 1 · 进程与线程基础',
    'QuantDevOS01 Process Thread Basics.md',
    null,
    {
      directory: 'quant',
      titleEn: 'OS 1 · Process & Thread Basics',
      category: '操作系统面经',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    '操作系统 2 · 线程同步与死锁',
    'QuantDevOS02 Synchronization Deadlock.md',
    null,
    {
      directory: 'quant',
      titleEn: 'OS 2 · Thread Synchronization & Deadlocks',
      category: '操作系统面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    '操作系统 3 · 进程间通信与调度',
    'QuantDevOS03 IPC Scheduling.md',
    null,
    {
      directory: 'quant',
      titleEn: 'OS 3 · Inter-Process Communication & Scheduling',
      category: '操作系统面经',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    '操作系统 4 · 虚拟内存与链接',
    'QuantDevOS04 Virtual Memory Linking.md',
    null,
    {
      directory: 'quant',
      titleEn: 'OS 4 · Virtual Memory & Linking',
      category: '操作系统面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    '计算机网络 1 · IO 模型与多路复用',
    'QuantDevNet01 IO Models Multiplexing.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Computer Networks 1 · I/O Models & Multiplexing',
      category: '计算机网络面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    '计算机网络 2 · Reactor、Proactor 与线程池',
    'QuantDevNet02 Reactor Proactor ThreadPool.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Computer Networks 2 · Reactor, Proactor & Thread Pools',
      category: '计算机网络面经',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    '计算机网络 3 · 手撕 Socket Server',
    'QuantDevNet03 Handwritten Socket Server.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Computer Networks 3 · Handwritten Socket Server',
      category: '计算机网络面经',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Effective Modern C++ 1 · 类型推导与 auto（条款 1-6）',
    'QuantDevEMC01 Deducing Types Auto.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Effective Modern C++ 1 · Type Deduction & auto (Items 1-6)',
      category: 'Effective Modern C++',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Effective Modern C++ 2 · 迈向现代 C++（条款 7-17）',
    'QuantDevEMC02 Moving To Modern Cpp.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Effective Modern C++ 2 · Moving to Modern C++ (Items 7-17)',
      category: 'Effective Modern C++',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Effective Modern C++ 3 · 智能指针（条款 18-22）',
    'QuantDevEMC03 Smart Pointers.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Effective Modern C++ 3 · Smart Pointers (Items 18-22)',
      category: 'Effective Modern C++',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Effective Modern C++ 4 · 右值引用、移动语义与完美转发（条款 23-30）',
    'QuantDevEMC04 Rvalue References Move Semantics.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Effective Modern C++ 4 · Rvalues, Move Semantics & Perfect Forwarding (Items 23-30)',
      category: 'Effective Modern C++',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Effective Modern C++ 5 · Lambda 表达式与工程细节（条款 31-34、41-42）',
    'QuantDevEMC05 Lambda Expressions Tweaks.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Effective Modern C++ 5 · Lambdas & Engineering Details (Items 31-34, 41-42)',
      category: 'Effective Modern C++',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'Effective Modern C++ 6 · 并发 API（条款 35-40）',
    'QuantDevEMC06 Concurrency API.md',
    null,
    {
      directory: 'quant',
      titleEn: 'Effective Modern C++ 6 · Concurrency API (Items 35-40)',
      category: 'Effective Modern C++',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'Effective Modern C++ 7 · C++17 与 C++20 核心新特性深度解构',
    'QuantDevEMC07 C++17 and C++20 Key Modern Features.md',
    'QuantDevEMC07 C++17 and C++20 Key Modern Features.en.md',
    {
      directory: 'quant',
      titleEn: 'Effective Modern C++ 7 · C++17 & C++20 Core Modern Features',
      category: 'Effective Modern C++',
      difficulty: 'Hard',
    },
  ),
];

const quantNotes = quantNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const mlCodingNoteDefinitions = [
  createTutorialDefinition(
    'ML Coding · 从零实现 LLM',
    'MLCoding01 Unicode Pretokenization.md',
    'MLCoding01 Unicode Pretokenization.en.md',
    {
      directory: 'MLCoding',
      titleEn: 'ML Coding · From-Scratch LLM Implementation',
      category: 'From Scratch',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'ML Coding 02 · 基础算子补完：GELU、BatchNorm、Kaiming Init、Dropout、Conv2d、线性回归、梯度累积',
    'MLCoding02 GELU BatchNorm Conv2d Linear Regression.md',
    'MLCoding02 GELU BatchNorm Conv2d Linear Regression.en.md',
    {
      directory: 'MLCoding',
      titleEn: 'ML Coding 02 · Fundamentals Roundup: GELU, BatchNorm, Kaiming Init, Dropout, Conv2d, Linear Regression',
      category: 'From Scratch',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'ML Coding 03 · 注意力机制全家桶：从 MHA 到 GQA、滑动窗口、线性注意力、KV Cache 与 Flash Attention',
    'MLCoding03 Attention Variants GQA Sliding Window KV Cache.md',
    'MLCoding03 Attention Variants GQA Sliding Window KV Cache.en.md',
    {
      directory: 'MLCoding',
      titleEn: 'ML Coding 03 · Attention Zoo: From MHA to GQA, Sliding Window, Linear Attention, KV Cache, and Flash Attention',
      category: 'From Scratch',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'ML Coding 04 · 架构组件：LoRA 低秩微调、ViT Patch Embedding 与 Mixture of Experts',
    'MLCoding04 LoRA ViT Patch Embedding MoE.md',
    'MLCoding04 LoRA ViT Patch Embedding MoE.en.md',
    {
      directory: 'MLCoding',
      titleEn: 'ML Coding 04 · Architecture Extensions: LoRA, ViT Patch Embedding, and Mixture of Experts',
      category: 'From Scratch',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    'ML Coding 05 · 推理解码策略：Top-k/Top-p 采样、Beam Search 与投机解码',
    'MLCoding05 Sampling Beam Search Speculative Decoding.md',
    'MLCoding05 Sampling Beam Search Speculative Decoding.en.md',
    {
      directory: 'MLCoding',
      titleEn: 'ML Coding 05 · Decoding Strategies: Top-k/Top-p Sampling, Beam Search, and Speculative Decoding',
      category: 'From Scratch',
      difficulty: 'Medium',
    },
  ),
  createTutorialDefinition(
    'ML Coding 06 · 量化与对齐：INT8 量化和 DPO / GRPO / PPO / OPD 损失函数',
    'MLCoding06 INT8 Quantization DPO GRPO PPO OPD Loss.md',
    'MLCoding06 INT8 Quantization DPO GRPO PPO OPD Loss.en.md',
    {
      directory: 'MLCoding',
      titleEn: 'ML Coding 06 · Quantization & Alignment: INT8 Quantization and DPO / GRPO / PPO / OPD Losses',
      category: 'From Scratch',
      difficulty: 'Hard',
    },
  ),
  createTutorialDefinition(
    '强化学习练习 · RL Infra 自测 35 问',
    'MLSYS15 RL Infra 自测 35 问.md',
    'MLSYS15 RL Infra 自测 35 问.en.md',
    {
      directory: 'Mlsys',
      titleEn: 'RL Practice · 35 Questions on RL Infra',
      category: 'Reinforcement Learning',
      difficulty: 'Medium',
    },
  ),
];

const mlCodingNotes = mlCodingNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const systemDesignNoteDefinitions = [
  createTutorialDefinition(
    'System Design 00 · 方法总览',
    'SystemDesign00 Overview.md',
    'SystemDesign00 Overview.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 00 · Overview & Methodology', category: 'Overview', difficulty: 'Intro' },
  ),
  createTutorialDefinition(
    'System Design 01 · 无状态设计范式',
    'SystemDesign01 Stateless Service.md',
    'SystemDesign01 Stateless Service.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 01 · Stateless Service Patterns', category: 'Design Pattern', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 01B · 虚拟化与容器',
    'SystemDesign01B Virtualization Containers.md',
    'SystemDesign01B Virtualization Containers.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 01B · Virtualization & Containers', category: 'Compute Isolation', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 02 · 数据库基本范式',
    'SystemDesign02 Database Paradigms.md',
    'SystemDesign02 Database Paradigms.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 02 · Database Paradigms', category: 'Database', difficulty: 'Medium' },
  ),
  {
    id: 'SystemDesign03 Database Scaling.md',
    title: 'System Design 03 · 数据库扩展三件套',
    titleEn: 'System Design 03 · Database Scaling Trio',
    fileName: 'SystemDesign03 Database Scaling.md',
    zhFileName: 'SystemDesign03 Database Scaling.md',
    enFileName: 'SystemDesign03 Database Scaling.en.md',
    directory: 'SystemDesign',
    category: 'Design Pattern',
    difficulty: 'Medium',
    content: systemDesignDbScalingContent,
  },
  createTutorialDefinition(
    'System Design 04 · 存储系统',
    'SystemDesign04 Storage Systems.md',
    'SystemDesign04 Storage Systems.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 04 · Storage Systems', category: 'Storage', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 05 · 可靠性与复制',
    'SystemDesign05 Reliability Replication.md',
    'SystemDesign05 Reliability Replication.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 05 · Reliability & Replication', category: 'Reliability', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 06 · 异步消息系统',
    'SystemDesign06 Async Messaging Systems.md',
    'SystemDesign06 Async Messaging Systems.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 06 · Async Messaging Systems', category: 'Messaging', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'System Design 07 · 图片分享与 Feed',
    'SystemDesign07 Photo Sharing Feed.md',
    'SystemDesign07 Photo Sharing Feed.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 07 · Photo Sharing & Feed', category: 'Case Study', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'System Design 08 · 异步 LLM RL 训练平台',
    'SystemDesign08 LLM Async RL Platform.md',
    'SystemDesign08 LLM Async RL Platform.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 08 · Async LLM RL Training Platform', category: 'ML Infrastructure', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'System Design 09 · 一致性哈希',
    'SystemDesign09 Consistent Hashing.md',
    'SystemDesign09 Consistent Hashing.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 09 · Consistent Hashing', category: 'Distributed Systems', difficulty: 'Medium' },
  ),
  // Keep the glossary as the final System Design note even when new chapters are inserted.
  createTutorialDefinition(
    'System Design 99 · 高频术语整合',
    'SystemDesign99 Glossary.md',
    'SystemDesign99 Glossary.en.md',
    { directory: 'SystemDesign', titleEn: 'System Design 99 · Glossary & Key Concepts', category: 'Glossary', difficulty: 'Reference' },
  ),
];

const systemDesignNotes = systemDesignNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: definition.content
      ? createInlineVariant(definition.zhFileName, definition.content)
      : createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const businessAlgorithmNoteDefinitions = [
  createTutorialDefinition(
    '第 1 章 · 推荐与搜索的多阶段链路',
    'Business Algorithm TODO.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'System Map', difficulty: 'Start Here' },
  ),
  createTutorialDefinition(
    '第 2 章 · 数据、样本与特征流',
    'BusinessAlgorithm00 Data Foundations.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Data', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 3 章 · 稀疏检索与协同召回',
    'BusinessAlgorithm01 Retrieval.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Retrieval', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 4 章 · 双塔、负样本与向量检索',
    'BusinessAlgorithm01B Vector Retrieval.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Vector Retrieval', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 5 章 · Query 理解与改写',
    'BusinessAlgorithm01D Query Understanding.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Query Understanding', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 6 章 · Query、内容和多路召回',
    'BusinessAlgorithm01C Multi-Channel Retrieval.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Retrieval Fusion', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 7 章 · 搜索体验与评价',
    'BusinessAlgorithm01E Search Quality.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Search Quality', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 8 章 · 搜索相关性与 BERT',
    'BusinessAlgorithm01F Search Relevance.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Search Relevance', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 9 章 · 排序目标与离线评价',
    'BusinessAlgorithm02 Ranking.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Ranking', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 10 章 · 多目标学习与分数融合',
    'BusinessAlgorithm02B Multi-Objective Ranking.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Multi-Objective', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 11 章 · 特征交叉、粗排与个性化',
    'BusinessAlgorithm02C Feature Interaction.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Feature Interaction', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 12 章 · 用户行为序列',
    'BusinessAlgorithm02D User Sequences.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'User Modeling', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 13 章 · 重排、多样性与规则',
    'BusinessAlgorithm03 List Decision.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Reranking', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 14 章 · 冷启动、探索与长期反馈',
    'BusinessAlgorithm03B Exploration Cold Start.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Exploration', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 15 章 · 在线实验与涨指标',
    'BusinessAlgorithm03C Experimentation Growth.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Experimentation', difficulty: 'Applied' },
  ),
  createTutorialDefinition(
    '第 16 章 · 查询词推荐',
    'BusinessAlgorithm03D Query Recommendation.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Query Recommendation', difficulty: 'Core' },
  ),
  createTutorialDefinition(
    '第 17 章 · 生成式检索与 Semantic ID',
    'BusinessAlgorithm04 Generative Algorithms.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Generative Retrieval', difficulty: 'Frontier' },
  ),
  createTutorialDefinition(
    '第 18 章 · LLM 排序与生成式推荐',
    'BusinessAlgorithm05 Generative Recommendation.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Generative Rec', difficulty: 'Frontier' },
  ),
  createTutorialDefinition(
    '第 19 章 · RAG 与 Agentic Search',
    'BusinessAlgorithm06 Agentic Search.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Generative Search', difficulty: 'Frontier' },
  ),
  createTutorialDefinition(
    '第 20 章 · 系统设计与上线验证',
    'BusinessAlgorithm07 System Design.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Production', difficulty: 'Applied' },
  ),
];

const businessAlgorithmEnglishTitles = {
  'Business Algorithm TODO.md': 'Chapter 1 · Multi-Stage Recommendation and Search Pipeline',
  'BusinessAlgorithm00 Data Foundations.md': 'Chapter 2 · Data, Samples, and Feature Streams',
  'BusinessAlgorithm01 Retrieval.md': 'Chapter 3 · Sparse and Collaborative Retrieval',
  'BusinessAlgorithm01B Vector Retrieval.md': 'Chapter 4 · Two-Tower Models, Negatives, and Vector Retrieval',
  'BusinessAlgorithm01D Query Understanding.md': 'Chapter 5 · Query Understanding and Rewriting',
  'BusinessAlgorithm01C Multi-Channel Retrieval.md': 'Chapter 6 · Query, Content, and Multi-Channel Retrieval',
  'BusinessAlgorithm01E Search Quality.md': 'Chapter 7 · Search Experience and Evaluation',
  'BusinessAlgorithm01F Search Relevance.md': 'Chapter 8 · Search Relevance and BERT',
  'BusinessAlgorithm02 Ranking.md': 'Chapter 9 · Ranking Objectives and Offline Evaluation',
  'BusinessAlgorithm02B Multi-Objective Ranking.md': 'Chapter 10 · Multi-Objective Learning and Score Fusion',
  'BusinessAlgorithm02C Feature Interaction.md': 'Chapter 11 · Feature Interaction, Coarse Ranking, and Personalization',
  'BusinessAlgorithm02D User Sequences.md': 'Chapter 12 · User Behavior Sequences',
  'BusinessAlgorithm03 List Decision.md': 'Chapter 13 · Reranking, Diversity, and Rules',
  'BusinessAlgorithm03B Exploration Cold Start.md': 'Chapter 14 · Cold Start, Exploration, and Long-Term Feedback',
  'BusinessAlgorithm03C Experimentation Growth.md': 'Chapter 15 · Online Experimentation and Metric Growth',
  'BusinessAlgorithm03D Query Recommendation.md': 'Chapter 16 · Query Recommendation',
  'BusinessAlgorithm04 Generative Algorithms.md': 'Chapter 17 · Generative Retrieval and Semantic IDs',
  'BusinessAlgorithm05 Generative Recommendation.md': 'Chapter 18 · LLM Ranking and Generative Recommendation',
  'BusinessAlgorithm06 Agentic Search.md': 'Chapter 19 · RAG and Agentic Search',
  'BusinessAlgorithm07 System Design.md': 'Chapter 20 · System Design and Production Validation',
};

const businessAlgorithmNotes = businessAlgorithmNoteDefinitions.map((definition) => ({
  ...definition,
  titleEn: businessAlgorithmEnglishTitles[definition.id] ?? definition.titleEn,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const noteSections = [
  {
    id: 'mlsys',
    title: 'MLSYS',
    description: 'GPU kernels, training systems, inference systems, and performance notes',
    notes: mlsysNotes,
  },
  {
    id: 'quant',
    title: 'Quant',
    description: 'Probability, Markov chains, expectation, and interview math drills',
    notes: quantNotes,
  },
  {
    id: 'mlcoding',
    title: 'ML Coding & 八股',
    description: 'From-scratch machine learning implementation exercises, plus ML interview drills',
    notes: mlCodingNotes,
  },
  {
    id: 'system-design',
    title: 'System Design',
    description: 'Backend system design, LLM serving, agent workflows, and infra interview drills',
    notes: systemDesignNotes,
  },
  {
    id: 'business-algorithm',
    title: '业务算法',
    description: '从一次线上请求出发，拆解召回、排序、列表决策、生成式方法与实验闭环',
    notes: businessAlgorithmNotes,
  },
  {
    id: 'leetcode',
    title: 'LeetCode',
    description: 'Core data structure and algorithm interview drills',
    notes: leetcodeNotes,
  },
  ...(isDraftMode
    ? [
        {
          id: 'drafts',
          title: '草稿区',
          description: 'Local-only drafts. Visible in dev, hidden from production builds.',
          notes: draftNotes,
        },
      ]
    : []),
];

const tutorials = noteSections.flatMap((section) =>
  section.notes.map((note) => ({
    ...note,
    sectionId: section.id,
    sectionTitle: section.title,
  })),
);

const noteIdByAlias = buildNoteAliasMap(tutorials);
const mediaModules = import.meta.glob('../notes/**/assets/**/*.{png,jpg,jpeg,gif,webp,svg,avif,bmp}', {
  eager: true,
  import: 'default',
  query: '?url',
});
const mediaUrlByAlias = buildMediaAliasMap(mediaModules);
const languageOptions = [
  { id: 'zh', label: '中文' },
  { id: 'en', label: 'English' },
];

const homeStats = [
  { value: noteSections.length, id: 'sections' },
  { value: tutorials.length, id: 'notes' },
  { value: '2', id: 'languages' },
];

const homeCopy = {
  zh: {
    brandSubtitle: '系统 · 基础设施 · 算法练习',
    mainNavigation: '主导航',
    home: '首页',
    about: '关于',
    languageSelector: '首页语言',
    heroEyebrow: 'Interview Notes',
    heroTitle: 'ML / LLM 技术复习笔记',
    siteSummary: '站点概览',
    stats: {
      sections: '板块',
      notes: '篇笔记',
      languages: '语言',
    },
    sectionsAria: '笔记板块',
    sectionsEyebrow: 'Sections',
    sectionsHeading: '笔记板块',
    noteCount: (count) => `${count} 篇笔记`,
    aboutEyebrow: 'About',
    aboutHeading: '关于作者',
    aboutBody: [
      '这个网站由 Zhikai Chen 维护，收录我平时复习 MLSYS、LLM infra、ML coding、quant 和 LeetCode 时留下的笔记。',
      '我目前在找工作，研究和工程兴趣集中在 agent memory、agentic reinforcement learning、predictive foundation models 和 agentic security。如果你有合适的机会，欢迎通过 GitHub、LinkedIn 或 Email 联系。',
    ],
    contactLinks: '作者联系方式',
  },
  en: {
    brandSubtitle: 'systems · infrastructure · practice',
    mainNavigation: 'Main navigation',
    home: 'Home',
    about: 'About',
    languageSelector: 'Homepage language',
    heroEyebrow: 'Interview Notes',
    heroTitle: 'ML / LLM interview notes',
    siteSummary: 'Site summary',
    stats: {
      sections: 'Sections',
      notes: 'Notes',
      languages: 'Languages',
    },
    sectionsAria: 'Interview note sections',
    sectionsEyebrow: 'Sections',
    sectionsHeading: 'Browse the notes',
    noteCount: (count) => `${count} ${count === 1 ? 'note' : 'notes'}`,
    aboutEyebrow: 'About',
    aboutHeading: 'About the author',
    aboutBody: [
      'This site is maintained by Zhikai Chen. It collects the notes I use to review MLSYS, LLM infrastructure, ML coding, quant, and LeetCode.',
      "I'm currently looking for new opportunities. My work spans agent memory, agentic reinforcement learning, predictive foundation models, and agentic security. You can reach me on GitHub, LinkedIn, or by email.",
    ],
    contactLinks: 'Author contact links',
  },
};

const homeSectionCopy = {
  zh: {
    mlsys: {
      title: 'MLSYS',
      description: 'GPU kernel、训练系统、推理系统与性能分析',
    },
    quant: {
      title: 'Quant',
      description: '概率、马尔可夫链、期望与面试数学题',
    },
    mlcoding: {
      title: 'ML Coding & 八股',
      description: '从零实现 tokenizer、attention、训练循环等机器学习组件，附强化学习自测与面试题',
    },
    'system-design': {
      title: 'System Design',
      description: '后端系统设计、LLM serving、Agent workflow 与基础设施面试题',
    },
    'business-algorithm': {
      title: '业务算法',
      description: '沿一次线上请求拆解召回、排序、列表决策、生成式方法与实验闭环',
    },
    leetcode: {
      title: 'LeetCode',
      description: '数据结构、算法模式与 LeetCode 练习',
    },
    drafts: {
      title: '草稿区',
      description: '仅在本地开发环境可见的草稿',
    },
  },
  en: {
    mlsys: {
      title: 'MLSYS',
      description: 'GPU kernels, training and inference systems, and performance analysis',
    },
    quant: {
      title: 'Quant',
      description: 'Probability, Markov chains, expectation, and interview math',
    },
    mlcoding: {
      title: 'ML Coding & Interview',
      description: 'From-scratch implementations of tokenizers, attention, training loops, and other ML components, plus an RL self-test and interview questions',
    },
    'system-design': {
      title: 'System Design',
      description: 'Backend design, LLM serving, agent workflows, and infrastructure interviews',
    },
    'business-algorithm': {
      title: 'Business Algorithms',
      description: 'Retrieval, ranking, list decisions, generative methods, and experimentation along one production request',
    },
    leetcode: {
      title: 'LeetCode',
      description: 'Data structures, algorithm patterns, and LeetCode practice',
    },
    drafts: {
      title: 'Drafts',
      description: 'Local drafts shown in development builds only',
    },
  },
};

const authorLinks = [
  {
    label: 'GitHub',
    href: 'https://github.com/CurryTang',
    value: 'github.com/CurryTang',
  },
  {
    label: 'LinkedIn',
    href: 'https://www.linkedin.com/in/zhikai-chen-435252129',
    value: 'Zhikai Chen',
  },
  {
    label: 'Email',
    href: 'mailto:chenzh85@msu.edu',
    value: 'chenzh85@msu.edu',
  },
];

function createTutorialDefinition(title, zhFileName, enFileName, options = {}) {
  const directory = options.directory ?? 'Mlsys';
  const resolvedEnglishFileName = enFileName ?? zhFileName.replace(/\.md$/i, '.en.md');
  return {
    id: zhFileName,
    title,
    titleEn: options.titleEn ?? title,
    fileName: zhFileName,
    zhFileName,
    enFileName: resolvedEnglishFileName,
    directory,
    category: options.category ?? '',
    difficulty: options.difficulty ?? '',
  };
}

function createVariant(fileName, directory) {
  if (!fileName) {
    return {
      fileName: '',
      url: null,
    };
  }

  const modulePath = `../notes/${directory}/${fileName}`;
  const url = markdownModules[modulePath];
  return {
    fileName,
    url: typeof url === 'string' ? url : null,
  };
}

function createInlineVariant(fileName, content) {
  return {
    fileName,
    url: null,
    content,
  };
}

function createDraftTutorialDefinition(title, zhFileName, content, options = {}) {
  return {
    id: zhFileName,
    title,
    titleEn: options.titleEn ?? title,
    fileName: zhFileName,
    zhFileName,
    enFileName: '',
    directory: 'Drafts',
    category: 'Draft',
    difficulty: 'Draft',
    content,
  };
}

function variantHasContent(variant) {
  return Boolean(variant?.url || typeof variant?.content === 'string');
}

function normalizePathToken(rawValue) {
  if (!rawValue) {
    return '';
  }

  let value = rawValue.trim().replace(/\\/g, '/');
  try {
    value = decodeURIComponent(value);
  } catch {
    // Ignore malformed URI fragments and keep the original token.
  }

  value = value.replace(/^\.\//, '');
  value = value.replace(/^\//, '');
  value = value.replace(/^notes\//i, '');

  return value.toLowerCase();
}

function buildNoteAliasMap(tutorialList) {
  const map = new Map();

  const addAlias = (alias, id) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, id);
    }
  };

  tutorialList.forEach((tutorial) => {
    const fileNames = [tutorial.variants.zh.fileName, tutorial.variants.en.fileName].filter(Boolean);

    addAlias(tutorial.id, tutorial.id);
    addAlias(tutorial.fileName, tutorial.id);
    addAlias(`${tutorial.directory}/${tutorial.fileName}`, tutorial.id);
    addAlias(`notes/${tutorial.directory}/${tutorial.fileName}`, tutorial.id);

    fileNames.forEach((fileName) => {
      const withoutMd = fileName.replace(/\.md$/i, '');
      const withoutLang = withoutMd.replace(/\.en$/i, '');
      addAlias(fileName, tutorial.id);
      addAlias(`${tutorial.directory}/${fileName}`, tutorial.id);
      addAlias(`notes/${tutorial.directory}/${fileName}`, tutorial.id);
      addAlias(withoutMd, tutorial.id);
      addAlias(withoutLang, tutorial.id);
    });
  });

  return map;
}

function buildMediaAliasMap(modules) {
  const map = new Map();

  const addAlias = (alias, url) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, url);
    }
  };

  Object.entries(modules).forEach(([modulePath, assetUrl]) => {
    if (typeof assetUrl !== 'string') {
      return;
    }

    const relativePath = modulePath.replace('../notes/', '');
    const fileName = relativePath.split('/').at(-1) ?? relativePath;
    addAlias(relativePath, assetUrl);
    addAlias(`notes/${relativePath}`, assetUrl);
    addAlias(fileName, assetUrl);
    addAlias(`assets/${fileName}`, assetUrl);
    addAlias(`./assets/${fileName}`, assetUrl);
  });

  return map;
}

function splitObsidianTarget(rawContent) {
  const [targetPart, ...aliasParts] = rawContent.split('|');
  const target = targetPart?.trim() ?? '';
  const aliasRaw = aliasParts.join('|').trim();

  if (!aliasRaw || /^\d+$/.test(aliasRaw)) {
    return { target, alias: '' };
  }

  return { target, alias: aliasRaw };
}

function prettyLabel(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const token = withoutAnchor.split('/').at(-1) ?? withoutAnchor;
  const anchor = rawTarget.includes('#') ? cleanHeadingText(rawTarget.split('#').slice(1).join('#')) : '';
  return token.replace(/\.en\.md$/i, '').replace(/\.md$/i, '').trim() || anchor || rawTarget.trim();
}

function resolveNoteId(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const normalized = normalizePathToken(withoutAnchor);

  if (!normalized) {
    return null;
  }

  const basename = normalized.split('/').at(-1) ?? normalized;
  const candidates = [
    normalized,
    normalized.endsWith('.md') ? normalized.slice(0, -3) : `${normalized}.md`,
    basename,
    basename.endsWith('.md') ? basename.slice(0, -3) : `${basename}.md`,
  ];

  for (const candidate of candidates) {
    const match = noteIdByAlias.get(candidate);
    if (match) {
      return match;
    }
  }

  return null;
}

function resolveObsidianLink(target, alias) {
  if (target.startsWith('#')) {
    const heading = cleanHeadingText(target.slice(1));
    if (!heading) {
      return alias || '';
    }

    return `[${alias || heading}](#${slugify(heading)})`;
  }

  const noteId = resolveNoteId(target);
  if (!noteId) {
    return null;
  }

  const targetHeading = target.includes('#')
    ? cleanHeadingText(target.split('#').slice(1).join('#'))
    : '';
  const routeTarget = targetHeading
    ? `${noteId}::${slugify(targetHeading)}`
    : noteId;

  return `[${alias || prettyLabel(target)}](#${encodeURIComponent(routeTarget)})`;
}

function resolveMediaUrl(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const normalized = normalizePathToken(withoutAnchor);

  if (!normalized) {
    return null;
  }

  const basename = normalized.split('/').at(-1) ?? normalized;
  const candidates = [normalized, basename, `mlsys/assets/${basename}`, `assets/${basename}`];

  for (const candidate of candidates) {
    const match = mediaUrlByAlias.get(candidate);
    if (match) {
      return match;
    }
  }

  return null;
}

function slugify(text) {
  return String(text)
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\w一-龥-]/g, '')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '');
}

function HeadingWithAnchor({ level, children }) {
  const Tag = `h${level}`;
  const text = extractPlainText(children);
  const id = slugify(text);
  const scrollToSection = (event) => {
    event.preventDefault();
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <Tag id={id} className="heading-anchor-host">
      {children}
      <a href={`#${id}`} className="heading-anchor" aria-label="Link to section" onClick={scrollToSection}>¶</a>
    </Tag>
  );
}

function extractPlainText(value) {
  if (typeof value === 'string' || typeof value === 'number') {
    return String(value);
  }

  if (Array.isArray(value)) {
    return value.map(extractPlainText).join('');
  }

  if (value?.props?.children) {
    return extractPlainText(value.props.children);
  }

  return '';
}

function extractMarkdownHeadings(markdownText) {
  if (!markdownText) {
    return [];
  }

  return markdownText
    .replace(/```[\s\S]*?```/g, '')
    .split('\n')
    .map((line) => {
      const match = /^(#{1,3})\s+(.+?)\s*$/.exec(line);
      if (!match) {
        return null;
      }

      const text = cleanHeadingText(match[2]);
      if (!text) {
        return null;
      }

      return {
        id: slugify(text),
        level: match[1].length,
        text,
      };
    })
    .filter(Boolean);
}

function cleanHeadingText(text) {
  return text
    .replace(/<[^>]+>/g, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[`*_~]/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function normalizeAnswerToken(rawValue) {
  const value = String(rawValue ?? '')
    .trim()
    .replace(/^[([{\s]+/g, '')
    .replace(/[\])}\s.。:：]+$/g, '');
  if (!value) {
    return '';
  }

  if (/^\d+$/.test(value)) {
    return String(Number(value) - 1);
  }

  return value.charAt(0).toUpperCase();
}

function parseQuizSource(rawSource) {
  const lines = String(rawSource ?? '')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);

  const quiz = {
    title: 'Practice',
    question: '',
    answer: '',
    explanation: '',
    options: [],
  };

  lines.forEach((line) => {
    const fieldMatch = line.match(/^(title|question|answer|correct|explanation|解析|答案)\s*[:：]\s*(.+)$/i);
    if (fieldMatch) {
      const [, rawKey, rawValue] = fieldMatch;
      const key = rawKey.toLowerCase();
      if (key === 'correct' || key === 'answer' || rawKey === '答案') {
        quiz.answer = normalizeAnswerToken(rawValue);
      } else if (key === 'explanation' || rawKey === '解析') {
        quiz.explanation = rawValue.trim();
      } else {
        quiz[key] = rawValue.trim();
      }
      return;
    }

    const optionMatch = line.match(/^(?:[-*]\s*)?([A-Ha-h]|\d+)[).、:：]\s+(.+)$/);
    if (optionMatch) {
      const [, rawKey, text] = optionMatch;
      quiz.options.push({
        id: normalizeAnswerToken(rawKey),
        label: /^[A-Ha-h]$/.test(rawKey) ? rawKey.toUpperCase() : String(quiz.options.length + 1),
        text: text.trim(),
      });
      return;
    }

    if (!quiz.question) {
      quiz.question = line;
    }
  });

  if (!quiz.answer && quiz.options.some((option) => /^\*/.test(option.text))) {
    const correctOption = quiz.options.find((option) => /^\*/.test(option.text));
    quiz.answer = correctOption.id;
    quiz.options = quiz.options.map((option) => ({
      ...option,
      text: option.text.replace(/^\*\s*/, ''),
    }));
  }

  return quiz;
}

function QuizBlock({ source }) {
  const { t } = useUiCopy();
  const quiz = useMemo(() => parseQuizSource(source), [source]);
  const [collapsed, setCollapsed] = useState(false);
  const [selectedOption, setSelectedOption] = useState('');

  const isAnswered = Boolean(selectedOption);
  const isCorrect = selectedOption === quiz.answer;

  if (!quiz.question || quiz.options.length === 0 || !quiz.answer) {
    return (
      <pre>
        <code>{source}</code>
      </pre>
    );
  }

  return (
    <section className={`practice-card ${collapsed ? 'collapsed' : ''}`}>
      <button
        className="practice-card-toggle"
        type="button"
        onClick={() => setCollapsed((current) => !current)}
        aria-expanded={!collapsed}
        aria-label={`${collapsed ? t('展开', 'Show') : t('收起', 'Hide')} ${quiz.title}`}
      >
        <span>{quiz.title}</span>
        <span aria-hidden="true">{collapsed ? t('展开', 'Show') : t('收起', 'Hide')}</span>
      </button>

      {!collapsed && (
        <div className="practice-card-body">
          <p className="practice-question">{quiz.question}</p>
          <div className="practice-options" role="group" aria-label={quiz.question}>
            {quiz.options.map((option) => {
              const optionSelected = selectedOption === option.id;
              const optionCorrect = option.id === quiz.answer;
              const stateClass = isAnswered && optionSelected
                ? isCorrect
                  ? 'correct'
                  : 'incorrect'
                : isAnswered && optionCorrect
                  ? 'correct'
                  : '';

              return (
                <button
                  key={option.id}
                  className={`practice-option ${stateClass}`}
                  type="button"
                  onClick={() => setSelectedOption(option.id)}
                  aria-pressed={optionSelected}
                >
                  <span className="practice-option-key">{option.label}</span>
                  <span>{option.text}</span>
                </button>
              );
            })}
          </div>
          {isAnswered && (
            <p className={`practice-feedback ${isCorrect ? 'correct' : 'incorrect'}`} role="status">
              {isCorrect ? t('回答正确。', 'Correct.') : t('再想一下。', 'Not quite.')}
              {quiz.explanation ? ` ${quiz.explanation}` : ''}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function ForeignDictionaryTopoVisual() {
  const { isEnglish, t } = useUiCopy();
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackKey, setPlaybackKey] = useState(0);
  const comparisons = [
    ['hrn', 'hrf', 'n -> f', 'pair-1'],
    ['hrf', 'er', 'h -> e', 'pair-2'],
    ['er', 'enn', 'r -> n', 'pair-3'],
    ['enn', 'rfnn', 'e -> r', 'pair-4'],
  ];

  const nodes = [
    ['h', '0', 'node-h'],
    ['e', '1', 'node-e'],
    ['r', '1', 'node-r'],
    ['n', '1', 'node-n'],
    ['f', '1', 'node-f'],
  ];

  const playAnimation = () => {
    setIsPlaying(false);
    window.requestAnimationFrame(() => {
      setPlaybackKey((current) => current + 1);
      setIsPlaying(true);
    });
  };

  return (
    <section
      className={`topo-visual ${isPlaying ? 'is-playing' : ''}`}
      aria-label={t('外星文字典拓扑排序可视化', 'Foreign Dictionary topological sorting visualization')}
    >
      <div className="topo-visual-copy">
        <div>
          <p className="topo-kicker">{t('动画演示', 'Animated walkthrough')}</p>
          <h2>{t('从相邻单词比较，到 Kahn 拓扑序', 'From adjacent words to a Kahn topological order')}</h2>
          <p>{t(
            '每一组相邻单词只看第一个不同字符；这个字符对就是一条有向边。边建完后，入度为 0 的字符先进入队列。',
            'For each adjacent pair, inspect only the first differing character. That pair gives one directed edge; after building the graph, enqueue characters with indegree 0.',
          )}</p>
        </div>
        <button className="topo-play-button" type="button" onClick={playAnimation}>
          {isPlaying ? t('重播', 'Replay') : t('播放', 'Play')}
        </button>
      </div>

      <div className="topo-stage" key={playbackKey}>
        <div className="topo-words" aria-label={t('相邻单词比较', 'Adjacent word comparisons')}>
          {comparisons.map(([first, second, edge, pairClass]) => (
            <span className={`topo-word-pair ${pairClass}`} key={edge}>
              <b>{first}</b>
              <b>{second}</b>
              <em>{edge}</em>
            </span>
          ))}
        </div>

        <div className="topo-graph-board" aria-label={t('有向图 h 到 e 到 r 到 n 到 f', 'Directed graph h to e to r to n to f')}>
          <div className="topo-chain">
            {nodes.map(([label, indegree, nodeClass], index) => (
              <Fragment key={label}>
                <span className={`topo-node ${nodeClass}`}>
                  {label}
                  <small>{indegree}</small>
                </span>
                {index < nodes.length - 1 && (
                  <span
                    className={`topo-edge-link ${['edge-he', 'edge-er', 'edge-rn', 'edge-nf'][index]}`}
                    aria-hidden="true"
                  />
                )}
              </Fragment>
            ))}
          </div>
        </div>

        <div className="topo-output" aria-label={t('拓扑排序输出', 'Topological output order')}>
          {['h', 'e', 'r', 'n', 'f'].map((char, index) => (
            <span className={`out-${index + 1}`} key={char}>{char}</span>
          ))}
        </div>

        <ol className="topo-timeline">
          {isEnglish ? (
            <>
              <li className="step-1">Compare <code>hrn</code> with <code>hrf</code>. The first difference is <code>n/f</code>, giving <code>n -&gt; f</code>.</li>
              <li className="step-2">Compare <code>hrf</code> with <code>er</code>. The first difference is <code>h/e</code>, giving <code>h -&gt; e</code>.</li>
              <li className="step-3">Compare <code>er</code> with <code>enn</code>. The first difference is <code>r/n</code>, giving <code>r -&gt; n</code>.</li>
              <li className="step-4">Compare <code>enn</code> with <code>rfnn</code>. The first difference is <code>e/r</code>, giving <code>e -&gt; r</code>.</li>
              <li className="step-5">Kahn&apos;s algorithm starts from <code>h</code>, whose indegree is 0, then releases <code>e</code>, <code>r</code>, <code>n</code>, and <code>f</code> to output <code>hernf</code>.</li>
            </>
          ) : (
            <>
              <li className="step-1">比较 <code>hrn</code> 和 <code>hrf</code>，第一个不同字符是 <code>n/f</code>，得到 <code>n -&gt; f</code>。</li>
              <li className="step-2">比较 <code>hrf</code> 和 <code>er</code>，第一个不同字符是 <code>h/e</code>，得到 <code>h -&gt; e</code>。</li>
              <li className="step-3">比较 <code>er</code> 和 <code>enn</code>，第一个不同字符是 <code>r/n</code>，得到 <code>r -&gt; n</code>。</li>
              <li className="step-4">比较 <code>enn</code> 和 <code>rfnn</code>，第一个不同字符是 <code>e/r</code>，得到 <code>e -&gt; r</code>。</li>
              <li className="step-5">Kahn 算法从入度为 0 的 <code>h</code> 开始，依次释放 <code>e</code>、<code>r</code>、<code>n</code>、<code>f</code>，输出 <code>hernf</code>。</li>
            </>
          )}
        </ol>
      </div>
    </section>
  );
}

function CheapestFlightsBellmanVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeRound, setActiveRound] = useState(0);
  const rounds = (isEnglish ? [
    {
      label: 'init',
      title: 'Round 0 / source only',
      prices: ['0', '∞', '∞', '∞'],
      activeEdges: [],
      note: 'Only src=0 is reachable before taking any flight.',
    },
    {
      label: '1 edge',
      title: 'Round 1 / at most 1 flight',
      prices: ['0', '100', '∞', '∞'],
      activeEdges: ['flight-0-1'],
      note: 'Use the previous prices array. Flight 0 -> 1 relaxes city 1 to 100.',
    },
    {
      label: '2 edges',
      title: 'Round 2 / at most 2 flights',
      prices: ['0', '100', '200', '700'],
      activeEdges: ['flight-1-2', 'flight-1-3'],
      note: 'Copy before relaxing, so 1 -> 2 and 1 -> 3 are allowed, but 2 -> 3 cannot chain inside this same round.',
    },
  ] : [
    {
      label: '初始化',
      title: '第 0 轮 / 只有起点',
      prices: ['0', '∞', '∞', '∞'],
      activeEdges: [],
      note: '还没有乘坐任何航班时，只有 src=0 可达。',
    },
    {
      label: '1 条边',
      title: '第 1 轮 / 最多 1 趟航班',
      prices: ['0', '100', '∞', '∞'],
      activeEdges: ['flight-0-1'],
      note: '只读取上一轮的 prices。航班 0 -> 1 将城市 1 的价格更新为 100。',
    },
    {
      label: '2 条边',
      title: '第 2 轮 / 最多 2 趟航班',
      prices: ['0', '100', '200', '700'],
      activeEdges: ['flight-1-2', 'flight-1-3'],
      note: '松弛前先复制数组，因此可以使用 1 -> 2 和 1 -> 3，但 2 -> 3 不能在同一轮继续串联。',
    },
  ]);
  const cities = [
    ['0', 'src', 'city-0'],
    ['1', '', 'city-1'],
    ['2', '', 'city-2'],
    ['3', 'dst', 'city-3'],
  ];
  const flights = [
    ['flight-0-1', '0 -> 1', '$100'],
    ['flight-1-2', '1 -> 2', '$100'],
    ['flight-2-0', '2 -> 0', '$100'],
    ['flight-1-3', '1 -> 3', '$600'],
    ['flight-2-3', '2 -> 3', '$200'],
  ];
  const round = rounds[activeRound];

  const nextRound = () => {
    setActiveRound((current) => Math.min(current + 1, rounds.length - 1));
  };

  const previousRound = () => {
    setActiveRound((current) => Math.max(current - 1, 0));
  };

  return (
    <section className="bf-visual" aria-label={t('K 次中转内最便宜航班的 Bellman-Ford 可视化', 'Optimized Bellman-Ford visualization for Cheapest Flights Within K Stops')}>
      <div className="bf-header">
        <div>
          <p className="bf-kicker">{t('带边数限制的 Bellman-Ford', 'Bellman-Ford with edge budget')}</p>
          <h2>Cheapest Flights Within K Stops</h2>
          <p>{t('例子：', 'Example:')} <code>n=4</code>, <code>src=0</code>, <code>dst=3</code>, <code>k=1</code>. {t('最多可以乘坐', 'We may use at most')} <code>k + 1 = 2</code> {t('趟航班。', 'flights.')}</p>
        </div>
        <div className="bf-controls" aria-label={t('Bellman-Ford 轮次控制', 'Bellman-Ford round controls')}>
          <button type="button" onClick={previousRound} disabled={activeRound === 0} aria-label={t('上一轮', 'Previous round')}>{t('上一轮', 'Prev')}</button>
          <span>{round.title}</span>
          <button type="button" onClick={nextRound} disabled={activeRound === rounds.length - 1} aria-label={t('下一轮', 'Next round')}>{t('下一轮', 'Next')}</button>
        </div>
      </div>

      <div className="bf-stage">
        <div className="bf-round-tabs" role="tablist" aria-label={t('Bellman-Ford 轮次', 'Bellman-Ford rounds')}>
          {rounds.map((candidate, index) => (
            <button
              key={candidate.label}
              className={index === activeRound ? 'active' : ''}
              type="button"
              onClick={() => setActiveRound(index)}
              role="tab"
              aria-selected={index === activeRound}
            >
              {candidate.label}
            </button>
          ))}
        </div>

        <div className="bf-layout">
          <div className="bf-graph" aria-label={t('带权有向航班图', 'Weighted directed flights')}>
            {cities.map(([id, tag, className]) => (
              <div className={`bf-city ${className}`} key={id}>
                <strong>{id}</strong>
                {tag && <small>{tag}</small>}
              </div>
            ))}
            {flights.map(([className, route, price]) => (
              <div
                className={`bf-flight ${className} ${round.activeEdges.includes(className) ? 'active' : ''}`}
                key={className}
              >
                <span>{route}</span>
                <em>{price}</em>
              </div>
            ))}
          </div>

          <div className="bf-prices" aria-label={t('价格数组', 'Prices array')}>
            <div className="bf-prices-title">
              <span>prices</span>
              <small>{t('只读取上一轮', 'from previous round only')}</small>
            </div>
            <div className="bf-price-grid">
              {round.prices.map((price, index) => (
                <div className={`bf-price ${price !== '∞' ? 'reachable' : ''}`} key={`${round.label}-${index}`}>
                  <span>{t('城市', 'city')} {index}</span>
                  <strong>{price}</strong>
                </div>
              ))}
            </div>
            <p>{round.note}</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function SegmentTreeLISVisual() {
  const { t } = useUiCopy();
  const values = [2, 3, 5, 7, 9, 10, 18, 101];
  const steps = [
    { input: 10, rank: 5, query: '0..4', beforeBest: 0, current: 1, lis: 1, after: [0, 0, 0, 0, 0, 1, 0, 0] },
    { input: 9, rank: 4, query: '0..3', beforeBest: 0, current: 1, lis: 1, after: [0, 0, 0, 0, 1, 1, 0, 0] },
    { input: 2, rank: 0, query: 'empty', beforeBest: 0, current: 1, lis: 1, after: [1, 0, 0, 0, 1, 1, 0, 0] },
    { input: 5, rank: 2, query: '0..1', beforeBest: 1, current: 2, lis: 2, after: [1, 0, 2, 0, 1, 1, 0, 0] },
    { input: 3, rank: 1, query: '0..0', beforeBest: 1, current: 2, lis: 2, after: [1, 2, 2, 0, 1, 1, 0, 0] },
    { input: 7, rank: 3, query: '0..2', beforeBest: 2, current: 3, lis: 3, after: [1, 2, 2, 3, 1, 1, 0, 0] },
    { input: 101, rank: 7, query: '0..6', beforeBest: 3, current: 4, lis: 4, after: [1, 2, 2, 3, 1, 1, 0, 4] },
    { input: 18, rank: 6, query: '0..5', beforeBest: 3, current: 4, lis: 4, after: [1, 2, 2, 3, 1, 1, 4, 4] },
  ];
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const step = steps[activeStep];

  useEffect(() => {
    if (!isPlaying) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setActiveStep((current) => (current + 1) % steps.length);
    }, 1900);

    return () => window.clearInterval(timer);
  }, [isPlaying, steps.length]);

  const previousStep = () => {
    setIsPlaying(false);
    setActiveStep((current) => (current === 0 ? steps.length - 1 : current - 1));
  };

  const nextStep = () => {
    setIsPlaying(false);
    setActiveStep((current) => (current + 1) % steps.length);
  };

  const treeLevels = buildSegmentTreeLevels(step.after);
  const smallerValues = values.slice(0, step.rank);
  const queryDescription = smallerValues.length > 0
    ? `${t('rank', 'ranks')} 0..${step.rank - 1}${t('，也就是值', ', representing values')} ${smallerValues.join(', ')}`
    : t('没有更小的压缩值', 'there is no smaller compressed value');

  return (
    <section className="seg-visual" aria-label={t('最长递增子序列的线段树可视化', 'Segment tree visualization for longest increasing subsequence')}>
      <div className="seg-header">
        <div>
          <p className="seg-kicker">{t('线段树演示', 'Segment tree walkthrough')}</p>
          <h2>{t('LIS：先查更小值的最好结果，再更新当前值', 'LIS: query the best smaller value, then update the current value')}</h2>
          <p>
            {t('例子输入', 'Example input:')} <code>[10, 9, 2, 5, 3, 7, 101, 18]</code>。
            {t('坐标压缩后，每个叶子存', 'After coordinate compression, each leaf stores the ')}
            <strong>{t('以这个值结尾的最长递增子序列长度', 'longest increasing subsequence ending at that value')}</strong>。
          </p>
        </div>

        <div className="seg-controls" aria-label={t('线段树动画控制', 'Segment tree animation controls')}>
          <button type="button" onClick={previousStep} aria-label={t('上一个 LIS 步骤', 'Previous LIS step')}>{t('上一步', 'Prev')}</button>
          <button type="button" onClick={() => setIsPlaying((current) => !current)} aria-label={t('播放线段树动画', 'Play segment tree animation')}>
            {isPlaying ? t('暂停', 'Pause') : t('播放', 'Play')}
          </button>
          <button type="button" onClick={nextStep} aria-label={t('下一个 LIS 步骤', 'Next LIS step')}>{t('下一步', 'Next')}</button>
        </div>
      </div>

      <div className="seg-stage">
        <div className="seg-explainer">
          <div>
            <span>{t('这一帧怎么看', 'How to read this frame')}</span>
            <p>
              {t('现在处理输入里的第', 'We are processing input number')} <strong>{activeStep + 1}</strong>{t(' 个数：', ': ')}<strong>{step.input}</strong>。
              {t(`因为 LIS 要严格递增，它只能接在比 ${step.input} 更小的值后面。`, `Because the LIS must be strictly increasing, ${step.input} can only follow a smaller value.`)}
            </p>
          </div>
          <ol>
            <li>{t('蓝色叶子是本轮查询范围：', 'Blue leaves form this query range: ')}<code>{queryDescription}</code>。</li>
            <li>{t('线段树返回这些更小值里的最大 LIS 长度：', 'The segment tree returns the largest LIS among those smaller values: ')}<code>{step.beforeBest}</code>。</li>
            <li>{t('当前数自己的长度就是', 'The current value therefore has length')} <code>{step.beforeBest} + 1 = {step.current}</code>{t('，写到绿色叶子。', ', which is written to the green leaf.')}</li>
          </ol>
        </div>

        <div className="seg-step-summary">
          <div>
            <span>current num</span>
            <strong>{step.input}</strong>
            <small>rank {step.rank}</small>
          </div>
          <div>
            <span>query</span>
            <strong>{step.query}</strong>
            <small>best smaller = {step.beforeBest}</small>
          </div>
          <div>
            <span>update</span>
            <strong>{step.current}</strong>
            <small>tree[{step.rank}] = {step.current}</small>
          </div>
          <div>
            <span>LIS so far</span>
            <strong>{step.lis}</strong>
            <small>global answer</small>
          </div>
        </div>

        <div className="seg-board-title">
          <span>{t('压缩后的叶子', 'Compressed leaves')}</span>
          <small>{t('叶子里的数字 = 以该值结尾的最佳 LIS 长度', 'Leaf value = best LIS length ending at this value')}</small>
        </div>
        <div className="seg-rank-board" aria-label={t('压缩值叶子', 'Compressed value leaves')}>
          {values.map((value, index) => {
            const inQuery = step.rank > 0 && index < step.rank;
            const isUpdated = index === step.rank;

            return (
              <div
                className={`seg-leaf ${inQuery ? 'in-query' : ''} ${isUpdated ? 'updated' : ''}`}
                key={value}
              >
                <span>rank {index}</span>
                <strong>{value}</strong>
                <em>{step.after[index]}</em>
              </div>
            );
          })}
        </div>

        <div className="seg-board-title">
          <span>{t('线段树缓存', 'Segment tree cache')}</span>
          <small>{t('每个内部节点保存自己区间里的最大叶子值', 'Each internal node stores the maximum leaf value in its range')}</small>
        </div>
        <div className="seg-tree-board" aria-label={t('线段树最大值', 'Segment tree max values')}>
          {treeLevels.map((level, levelIndex) => (
            <div className="seg-tree-level" key={`level-${levelIndex}`}>
              {level.map((node) => {
                const intersectsQuery = step.rank > 0 && node.left < step.rank;
                const containsUpdate = node.left <= step.rank && step.rank <= node.right;

                return (
                  <div
                    className={`seg-tree-node ${intersectsQuery ? 'touches-query' : ''} ${containsUpdate ? 'update-path' : ''}`}
                    key={`${node.left}-${node.right}`}
                  >
                    <span>[{node.left}, {node.right}]</span>
                    <strong>{node.value}</strong>
                  </div>
                );
              })}
            </div>
          ))}
        </div>

        <div className="seg-board-title">
          <span>{t('输入顺序', 'Input order')}</span>
          <small>{t('点击任意一步，观察一个数如何改变整棵树', 'Select any step to see how one value changes the tree')}</small>
        </div>
        <ol className="seg-timeline">
          {steps.map((candidate, index) => (
            <li className={index === activeStep ? 'active' : ''} key={`${candidate.input}-${index}`}>
              <button type="button" onClick={() => { setIsPlaying(false); setActiveStep(index); }}>
                <span>{index + 1}</span>
                <strong>{candidate.input}</strong>
                <em>LIS {candidate.lis}</em>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

function buildSegmentTreeLevels(leaves) {
  const levels = [
    leaves.map((value, index) => ({ left: index, right: index, value })),
  ];

  while (levels[0].length > 1) {
    const previous = levels[0];
    const next = [];
    for (let index = 0; index < previous.length; index += 2) {
      const left = previous[index];
      const right = previous[index + 1] ?? left;
      next.push({
        left: left.left,
        right: right.right,
        value: Math.max(left.value, right.value),
      });
    }
    levels.unshift(next);
  }

  return levels;
}

const INTERVAL_VISUALS = {
  'interval-merge-demo': {
    title: 'Merge Intervals',
    subtitle: 'Sort by start, then keep extending the current merged interval.',
    domain: [0, 18],
    intervals: [
      { id: 'a', label: '[1, 3]', start: 1, end: 3 },
      { id: 'b', label: '[2, 6]', start: 2, end: 6 },
      { id: 'c', label: '[8, 10]', start: 8, end: 10 },
      { id: 'd', label: '[15, 18]', start: 15, end: 18 },
    ],
    steps: [
      {
        title: 'Step 1 · sort by start',
        note: '先按 start 排序，保证只需要和当前 merged interval 比较。',
        active: ['a'],
        result: [{ id: 'm1', label: 'current [1, 3]', start: 1, end: 3 }],
        stats: [['current', '[1, 3]'], ['output', '[]']],
      },
      {
        title: 'Step 2 · overlap, extend end',
        note: '[2, 6] 的 start <= current end 3，所以合并成 [1, 6]。',
        active: ['a', 'b'],
        muted: ['a'],
        result: [{ id: 'm1', label: 'merged [1, 6]', start: 1, end: 6 }],
        stats: [['condition', '2 <= 3'], ['current', '[1, 6]']],
      },
      {
        title: 'Step 3 · gap, flush current',
        note: '[8, 10] 的 start > current end 6，说明前一段结束，输出 [1, 6]。',
        active: ['c'],
        result: [
          { id: 'm1', label: 'output [1, 6]', start: 1, end: 6 },
          { id: 'm2', label: 'current [8, 10]', start: 8, end: 10 },
        ],
        stats: [['condition', '8 > 6'], ['output', '[[1, 6]]']],
      },
      {
        title: 'Step 4 · finish',
        note: '最后没有重叠，依次输出剩余 current。',
        active: ['d'],
        result: [
          { id: 'm1', label: '[1, 6]', start: 1, end: 6 },
          { id: 'm2', label: '[8, 10]', start: 8, end: 10 },
          { id: 'm3', label: '[15, 18]', start: 15, end: 18 },
        ],
        stats: [['answer', '[[1,6],[8,10],[15,18]]']],
      },
    ],
  },
  'interval-insert-demo': {
    title: 'Insert Interval',
    subtitle: 'Three zones: before newInterval, overlapping block, after newInterval.',
    domain: [0, 17],
    intervals: [
      { id: 'a', label: '[1, 2]', start: 1, end: 2 },
      { id: 'b', label: '[3, 5]', start: 3, end: 5 },
      { id: 'c', label: '[6, 7]', start: 6, end: 7 },
      { id: 'd', label: '[8, 10]', start: 8, end: 10 },
      { id: 'e', label: '[12, 16]', start: 12, end: 16 },
      { id: 'new', label: 'new [4, 8]', start: 4, end: 8, kind: 'new' },
    ],
    steps: [
      {
        title: 'Step 1 · append before zone',
        note: '[1, 2] 完全在 newInterval 左边，直接进 output。',
        active: ['a', 'new'],
        result: [{ id: 'o1', label: 'output [1, 2]', start: 1, end: 2 }],
        stats: [['rule', 'end < new.start'], ['output', '[[1,2]]']],
      },
      {
        title: 'Step 2 · merge overlap block',
        note: '[3,5], [6,7], [8,10] 都和 [4,8] 有交集，持续扩张 newInterval。',
        active: ['b', 'c', 'd', 'new'],
        muted: ['b', 'c', 'd'],
        result: [
          { id: 'o1', label: 'output [1, 2]', start: 1, end: 2 },
          { id: 'm1', label: 'merged [3, 10]', start: 3, end: 10 },
        ],
        stats: [['merged start', 'min(4,3)=3'], ['merged end', 'max(8,10)=10']],
      },
      {
        title: 'Step 3 · append after zone',
        note: '[12,16] 完全在合并结果右边，先放入 [3,10]，再追加剩余区间。',
        active: ['e'],
        result: [
          { id: 'o1', label: '[1, 2]', start: 1, end: 2 },
          { id: 'm1', label: '[3, 10]', start: 3, end: 10 },
          { id: 'o2', label: '[12, 16]', start: 12, end: 16 },
        ],
        stats: [['answer', '[[1,2],[3,10],[12,16]]']],
      },
    ],
  },
  'interval-rooms-demo': {
    title: 'Meeting Rooms II',
    subtitle: 'Sweep starts and ends; the answer is max active meetings.',
    domain: [0, 30],
    intervals: [
      { id: 'a', label: '[0, 30]', start: 0, end: 30 },
      { id: 'b', label: '[5, 10]', start: 5, end: 10 },
      { id: 'c', label: '[15, 20]', start: 15, end: 20 },
    ],
    events: [
      { time: 0, label: '+1' },
      { time: 5, label: '+1' },
      { time: 10, label: '-1' },
      { time: 15, label: '+1' },
      { time: 20, label: '-1' },
      { time: 30, label: '-1' },
    ],
    steps: [
      {
        title: 't = 0 · first meeting starts',
        note: 'active 从 0 变成 1，需要 1 个房间。',
        active: ['a'],
        marker: 0,
        stats: [['active', '1'], ['max rooms', '1']],
      },
      {
        title: 't = 5 · overlap appears',
        note: '[5,10] 开始时 [0,30] 还没结束，active = 2。',
        active: ['a', 'b'],
        marker: 5,
        stats: [['active', '2'], ['max rooms', '2']],
      },
      {
        title: 't = 10 · one room freed',
        note: '[5,10] 结束，active 回到 1。',
        active: ['a'],
        marker: 10,
        stats: [['active', '1'], ['max rooms', '2']],
      },
      {
        title: 't = 15 · another overlap',
        note: '[15,20] 开始时 [0,30] 仍在进行，max rooms 仍然是 2。',
        active: ['a', 'c'],
        marker: 15,
        stats: [['active', '2'], ['answer', '2']],
      },
    ],
  },
  'interval-query-demo': {
    title: 'Minimum Interval to Include Each Query',
    subtitle: 'Sort queries; push candidate intervals into a min heap by length.',
    domain: [0, 7],
    intervals: [
      { id: 'a', label: '[1, 4] len 4', start: 1, end: 4 },
      { id: 'b', label: '[2, 4] len 3', start: 2, end: 4 },
      { id: 'c', label: '[3, 6] len 4', start: 3, end: 6 },
      { id: 'd', label: '[4, 4] len 1', start: 4, end: 4 },
    ],
    queries: [2, 3, 4, 5],
    steps: [
      {
        title: 'query = 2',
        note: '加入 start <= 2 的区间：[1,4], [2,4]。最短覆盖区间是 [2,4]，长度 3。',
        active: ['a', 'b'],
        marker: 2,
        result: [{ id: 'best', label: 'best [2,4]', start: 2, end: 4 }],
        stats: [['heap top', 'len 3 [2,4]'], ['ans[2]', '3']],
      },
      {
        title: 'query = 3',
        note: '加入 [3,6]。heap 顶仍是 [2,4]，它覆盖 3。',
        active: ['b', 'c'],
        marker: 3,
        result: [{ id: 'best', label: 'best [2,4]', start: 2, end: 4 }],
        stats: [['heap top', 'len 3 [2,4]'], ['ans[3]', '3']],
      },
      {
        title: 'query = 4',
        note: '加入 [4,4]，长度 1，立刻成为最优答案。',
        active: ['a', 'b', 'c', 'd'],
        marker: 4,
        result: [{ id: 'best', label: 'best [4,4]', start: 4, end: 4 }],
        stats: [['heap top', 'len 1 [4,4]'], ['ans[4]', '1']],
      },
      {
        title: 'query = 5',
        note: '弹掉 end < 5 的区间，剩下 [3,6] 覆盖 5，长度 4。',
        active: ['c'],
        marker: 5,
        result: [{ id: 'best', label: 'best [3,6]', start: 3, end: 6 }],
        stats: [['removed', 'end < 5'], ['ans[5]', '4']],
      },
    ],
  },
};

const INTERVAL_ZH_COPY = {
  'interval-merge-demo': {
    title: '合并区间',
    subtitle: '先按起点排序，再不断扩展当前合并区间。',
    steps: ['第 1 步 · 按起点排序', '第 2 步 · 有重叠，扩展终点', '第 3 步 · 出现间隔，输出当前区间', '第 4 步 · 收尾'],
  },
  'interval-insert-demo': {
    title: '插入区间',
    subtitle: '分成三段：新区间左侧、重叠区间、新区间右侧。',
    steps: ['第 1 步 · 追加左侧区间', '第 2 步 · 合并重叠区间', '第 3 步 · 追加右侧区间'],
  },
  'interval-rooms-demo': {
    title: '会议室 II',
    subtitle: '扫描所有开始与结束事件，答案是同时进行的会议数峰值。',
    steps: ['t = 0 · 第一场会议开始', 't = 5 · 出现重叠', 't = 10 · 释放一个房间', 't = 15 · 再次重叠'],
  },
  'interval-query-demo': {
    title: '包含每个查询的最小区间',
    subtitle: '按顺序处理查询，把候选区间按长度放入最小堆。',
    steps: ['query = 2', 'query = 3', 'query = 4', 'query = 5'],
  },
};

const INTERVAL_EN_NOTES = {
  'interval-merge-demo': [
    'Sort by start so each interval only needs to be compared with the current merged interval.',
    'The start of [2, 6] is at most the current end 3, so extend the interval to [1, 6].',
    'The start of [8, 10] is greater than the current end 6. Flush [1, 6] and start a new interval.',
    'No remaining intervals overlap, so flush each current interval in order.',
  ],
  'interval-insert-demo': [
    '[1, 2] lies completely before newInterval, so append it directly to the output.',
    '[3,5], [6,7], and [8,10] all overlap [4,8], so keep expanding newInterval.',
    '[12,16] lies completely after the merged interval. Append [3,10], then append the remaining interval.',
  ],
  'interval-rooms-demo': [
    'active grows from 0 to 1, so one room is needed.',
    'When [5,10] starts, [0,30] is still active, so active = 2.',
    '[5,10] ends and active returns to 1.',
    'When [15,20] starts, [0,30] is still active, so the maximum remains 2 rooms.',
  ],
  'interval-query-demo': [
    'Add intervals with start <= 2: [1,4] and [2,4]. The shortest covering interval is [2,4], with length 3.',
    'Add [3,6]. The heap top remains [2,4], which covers query 3.',
    'Add [4,4]. Its length is 1, so it immediately becomes the best answer.',
    'Remove intervals with end < 5. The remaining [3,6] covers 5 and has length 4.',
  ],
};

const INTERVAL_STAT_ZH = {
  current: '当前区间',
  output: '输出',
  condition: '条件',
  answer: '答案',
  rule: '规则',
  'merged start': '合并后起点',
  'merged end': '合并后终点',
  active: '进行中',
  'max rooms': '最多房间',
  'heap top': '堆顶',
  removed: '已弹出',
};

function IntervalPatternVisual({ kind }) {
  const { isEnglish, t } = useUiCopy();
  const baseVisual = INTERVAL_VISUALS[kind];
  const zhCopy = INTERVAL_ZH_COPY[kind];
  const visual = isEnglish
    ? {
      ...baseVisual,
      steps: baseVisual.steps.map((step, index) => ({
        ...step,
        note: INTERVAL_EN_NOTES[kind][index],
      })),
    }
    : {
      ...baseVisual,
      title: zhCopy.title,
      subtitle: zhCopy.subtitle,
      steps: baseVisual.steps.map((step, index) => ({ ...step, title: zhCopy.steps[index] })),
    };
  const [activeStep, setActiveStep] = useState(0);
  const step = visual.steps[activeStep];
  const active = new Set(step.active ?? []);
  const muted = new Set(step.muted ?? []);
  const domain = visual.domain;
  const ticks = buildIntervalTicks(domain);

  return (
    <section className="interval-visual">
      <header className="interval-visual-header">
        <div>
          <p className="eyebrow">{t('区间题可视化', 'Interval visual')}</p>
          <h2>{visual.title}</h2>
          <p>{visual.subtitle}</p>
        </div>
        <div className="interval-step-counter">
          {activeStep + 1}<span>/ {visual.steps.length}</span>
        </div>
      </header>

      <div className="interval-step-note">
        <strong>{step.title}</strong>
        <span>{step.note}</span>
      </div>

      <div className="interval-axis" aria-label={`${visual.title} timeline`}>
        <div className="interval-axis-line">
          {ticks.map((tick) => (
            <span
              className="interval-tick"
              key={tick}
              style={{ left: `${intervalPercent(tick, domain)}%` }}
            >
              {tick}
            </span>
          ))}
          {step.marker !== undefined && (
            <span
              className="interval-marker"
              style={{ left: `${intervalPercent(step.marker, domain)}%` }}
            >
              q={step.marker}
            </span>
          )}
        </div>

        <div className="interval-lanes">
          {visual.intervals.map((interval) => (
            <IntervalBar
              domain={domain}
              interval={interval}
              isActive={active.has(interval.id)}
              isMuted={muted.has(interval.id)}
              key={interval.id}
            />
          ))}
        </div>

        {visual.queries && (
          <div className="interval-query-row">
            {visual.queries.map((query) => (
              <span
                className={query === step.marker ? 'active' : ''}
                key={query}
                style={{ left: `${intervalPercent(query, domain)}%` }}
              >
                {query}
              </span>
            ))}
          </div>
        )}

        {step.result && (
          <div className="interval-result-lanes">
            {step.result.map((interval) => (
              <IntervalBar
                domain={domain}
                interval={{ ...interval, kind: 'result' }}
                isActive
                key={interval.id}
              />
            ))}
          </div>
        )}

        {visual.events && (
          <div className="interval-events">
            {visual.events.map((event) => (
              <span
                className={event.time === step.marker ? 'active' : ''}
                key={`${event.time}-${event.label}`}
                style={{ left: `${intervalPercent(event.time, domain)}%` }}
              >
                {event.label}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="interval-stat-grid">
        {(step.stats ?? []).map(([label, value]) => (
          <div key={label}>
            <span>{isEnglish ? label : (INTERVAL_STAT_ZH[label] ?? label)}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>

      <ol className="interval-step-list">
        {visual.steps.map((candidate, index) => (
          <li className={index === activeStep ? 'active' : ''} key={candidate.title}>
            <button type="button" onClick={() => setActiveStep(index)}>
              <span>{index + 1}</span>
              {candidate.title.replace(/^(?:Step|第)\s*\d+\s*(?:步)?\s*·\s*/, '')}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

const POW_STEPS = [
  {
    title: 'Init',
    power: 10,
    base: 2,
    res: 1,
    bit: 0,
    action: 'power = 10, binary = 1010. Lowest bit is 0, so this round does not contribute to the answer yet.',
    next: 'Square base to 4, shift power right to 5.',
  },
  {
    title: 'Read bit 1',
    power: 5,
    base: 4,
    res: 1,
    bit: 1,
    action: 'power is odd. The current base represents x^2, so multiply it into res.',
    next: 'res = 1 * 4 = 4. Square base to 16, shift power right to 2.',
  },
  {
    title: 'Read bit 0',
    power: 2,
    base: 16,
    res: 4,
    bit: 0,
    action: 'Lowest bit is 0. x^4 is not needed for n = 10, so res stays unchanged.',
    next: 'Square base to 256, shift power right to 1.',
  },
  {
    title: 'Read bit 1',
    power: 1,
    base: 256,
    res: 4,
    bit: 1,
    action: 'power is odd again. The current base represents x^8, and n = 10 includes this bit.',
    next: 'res = 4 * 256 = 1024. Shift power to 0, stop.',
  },
  {
    title: 'Done',
    power: 0,
    base: 65536,
    res: 1024,
    bit: null,
    action: 'All bits have been consumed from right to left: 10 = 8 + 2.',
    next: 'Return 1024 for pow(2, 10).',
  },
];

const POW_STEPS_ZH = [
  {
    title: '初始化',
    action: 'power = 10，二进制是 1010。最低位为 0，这一轮暂时不计入答案。',
    next: 'base 平方得到 4，power 右移得到 5。',
  },
  {
    title: '读取 bit 1',
    action: 'power 是奇数。当前 base 代表 x²，因此把它乘进 res。',
    next: 'res = 1 × 4 = 4。base 平方得到 16，power 右移得到 2。',
  },
  {
    title: '读取 bit 0',
    action: '最低位为 0。n = 10 不需要 x⁴，因此 res 保持不变。',
    next: 'base 平方得到 256，power 右移得到 1。',
  },
  {
    title: '读取 bit 1',
    action: 'power 再次为奇数。当前 base 代表 x⁸，而 n = 10 包含这一位。',
    next: 'res = 4 × 256 = 1024。power 右移到 0，停止。',
  },
  {
    title: '完成',
    action: '所有二进制位都已从右向左处理完：10 = 8 + 2。',
    next: 'pow(2, 10) 返回 1024。',
  },
];

function BinaryPowVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = isEnglish
    ? POW_STEPS
    : POW_STEPS.map((step, index) => ({ ...step, ...POW_STEPS_ZH[index] }));
  const step = steps[activeStep];
  const binaryBits = ['1', '0', '1', '0'];
  const consumedFromRight = Math.min(activeStep, binaryBits.length);

  return (
    <section className="pow-visual" aria-label={t('二进制快速幂演示', 'Binary exponentiation walkthrough')}>
      <header className="pow-header">
        <div>
          <p className="eyebrow">{t('数学可视化', 'Math visual')}</p>
          <h2>Binary Exponentiation: pow(2, 10)</h2>
          <p>{t(
            '每一轮只看 power 的最低位：bit 为 1 才把当前 base 乘进 res。',
            'Each round reads only the lowest bit of power. Multiply the current base into res only when that bit is 1.',
          )}</p>
        </div>
        <div className="pow-counter">{activeStep + 1}<span>/ {POW_STEPS.length}</span></div>
      </header>

      <div className="pow-board">
        <div className="pow-bits" aria-label={t('指数 10 的二进制位', 'Binary bits of exponent 10')}>
          {binaryBits.map((bit, index) => {
            const fromRight = binaryBits.length - 1 - index;
            const isCurrent = fromRight === consumedFromRight && activeStep < binaryBits.length;
            const isConsumed = fromRight < consumedFromRight;
            return (
              <span
                className={`${isCurrent ? 'current' : ''} ${isConsumed ? 'consumed' : ''}`}
                key={`${bit}-${index}`}
              >
                {bit}
                <small>{[8, 4, 2, 1][index]}</small>
              </span>
            );
          })}
        </div>

        <div className="pow-state-grid">
          <div>
            <span>x / base</span>
            <strong>{step.base}</strong>
          </div>
          <div>
            <span>power</span>
            <strong>{step.power}</strong>
          </div>
          <div>
            <span>power & 1</span>
            <strong>{step.bit === null ? '-' : step.bit}</strong>
          </div>
          <div>
            <span>res</span>
            <strong>{step.res}</strong>
          </div>
        </div>

        <div className="pow-explain">
          <strong>{step.title}</strong>
          <p>{step.action}</p>
          <p>{step.next}</p>
        </div>
      </div>

      <ol className="pow-timeline">
        {steps.map((candidate, index) => (
          <li className={index === activeStep ? 'active' : ''} key={candidate.title}>
            <button type="button" onClick={() => setActiveStep(index)}>
              <span>{index + 1}</span>
              {candidate.title}
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

const SLIDING_WINDOW_VALUES = ['A', 'B', 'C', 'A', 'D', 'B'];

const SLIDING_WINDOW_STEPS = [
  {
    phase: 'expand',
    title: '右扩：加入 A',
    detail: 'right 向右走一格，把新元素加入窗口状态。',
    left: 0,
    right: 0,
    valid: true,
    state: 'A × 1',
    best: '—',
  },
  {
    phase: 'record',
    title: '合法：记录 A',
    detail: '窗口合法；最长模板在收缩循环之后更新答案。',
    left: 0,
    right: 0,
    valid: true,
    state: 'A × 1',
    best: 'A',
  },
  {
    phase: 'expand',
    title: '右扩：加入 B',
    detail: 'right 永远只向右，增量加入 B，不重新扫描整个窗口。',
    left: 0,
    right: 1,
    valid: true,
    state: 'A × 1 · B × 1',
    best: 'A',
  },
  {
    phase: 'record',
    title: '合法：记录 AB',
    detail: '当前窗口 [left, right] 合法，best 从 1 更新为 2。',
    left: 0,
    right: 1,
    valid: true,
    state: 'A × 1 · B × 1',
    best: 'AB',
  },
  {
    phase: 'expand',
    title: '右扩：加入 C',
    detail: '状态仍然合法，窗口继续扩大。',
    left: 0,
    right: 2,
    valid: true,
    state: 'A × 1 · B × 1 · C × 1',
    best: 'AB',
  },
  {
    phase: 'record',
    title: '合法：记录 ABC',
    detail: '窗口长度是 right - left + 1 = 3。',
    left: 0,
    right: 2,
    valid: true,
    state: 'A × 1 · B × 1 · C × 1',
    best: 'ABC',
  },
  {
    phase: 'validate',
    title: '加入 A 后条件失效',
    detail: 'A 的频次变成 2。不要移动 right，也不要立刻记录答案；进入收缩循环。',
    left: 0,
    right: 3,
    valid: false,
    state: 'A × 2 · B × 1 · C × 1',
    best: 'ABC',
  },
  {
    phase: 'shrink',
    title: '左缩：移除旧 A',
    detail: '先从状态中删除 nums[left]，再执行 left += 1，直到窗口重新合法。',
    left: 1,
    right: 3,
    removedIndex: 0,
    valid: true,
    state: 'A × 1 · B × 1 · C × 1',
    best: 'ABC',
  },
  {
    phase: 'record',
    title: '恢复合法：窗口 BCA',
    detail: '最长模板在 while invalid 结束后记录；长度仍是 3，best 不变。',
    left: 1,
    right: 3,
    valid: true,
    state: 'A × 1 · B × 1 · C × 1',
    best: 'ABC',
  },
  {
    phase: 'expand',
    title: '右扩：加入 D',
    detail: '窗口 BCAD 合法，right 再次只向右前进。',
    left: 1,
    right: 4,
    valid: true,
    state: 'A × 1 · B × 1 · C × 1 · D × 1',
    best: 'ABC',
  },
  {
    phase: 'record',
    title: '记录新的最优 BCAD',
    detail: '当前长度 4 大于旧答案 3，更新 best。完整循环随后继续处理下一个 right。',
    left: 1,
    right: 4,
    valid: true,
    state: 'A × 1 · B × 1 · C × 1 · D × 1',
    best: 'BCAD',
  },
];

const SLIDING_PHASES = [
  ['expand', '1 · 右扩', '加入 nums[right]'],
  ['validate', '2 · 判断', '检查窗口条件'],
  ['shrink', '3 · 左缩', 'while 触发就删除'],
  ['record', '4 · 记录', '在正确时机更新'],
];

const SLIDING_WINDOW_STEPS_EN = [
  ['Expand right: add A', 'Move right one position and add the new element to the window state.'],
  ['Valid: record A', 'The window is valid. For the longest-window template, update the answer after the shrinking loop.'],
  ['Expand right: add B', 'right only moves forward. Add B incrementally instead of rescanning the window.'],
  ['Valid: record AB', 'The current [left, right] window is valid, so best grows from 1 to 2.'],
  ['Expand right: add C', 'The state remains valid, so the window keeps growing.'],
  ['Valid: record ABC', 'The window length is right - left + 1 = 3.'],
  ['Adding A breaks the condition', 'The count of A becomes 2. Keep right fixed and enter the shrinking loop before recording an answer.'],
  ['Shrink left: remove the old A', 'Remove nums[left] from the state, then increment left until the window is valid again.'],
  ['Valid again: window BCA', 'For the longest-window template, record only after the invalid loop ends. The length is still 3.'],
  ['Expand right: add D', 'Window BCAD is valid, and right continues moving only forward.'],
  ['Record the new best BCAD', 'The current length 4 exceeds the previous best 3, so update best before processing the next right.'],
];

const SLIDING_PHASES_EN = [
  ['expand', '1 · Expand right', 'add nums[right]'],
  ['validate', '2 · Validate', 'check the window condition'],
  ['shrink', '3 · Shrink left', 'remove while invalid'],
  ['record', '4 · Record', 'update at the right moment'],
];

function SlidingWindowVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = isEnglish
    ? SLIDING_WINDOW_STEPS.map((step, index) => ({
      ...step,
      title: SLIDING_WINDOW_STEPS_EN[index][0],
      detail: SLIDING_WINDOW_STEPS_EN[index][1],
    }))
    : SLIDING_WINDOW_STEPS;
  const phases = isEnglish ? SLIDING_PHASES_EN : SLIDING_PHASES;
  const step = steps[activeStep];
  const windowText = SLIDING_WINDOW_VALUES.slice(step.left, step.right + 1).join('');

  return (
    <section className="sliding-window-visual" aria-label={t('滑动窗口万能模板演示', 'General sliding-window template walkthrough')}>
      <header className="sliding-window-header">
        <div>
          <p className="eyebrow">{t('滑动窗口可视化', 'Sliding window visual')}</p>
          <h2>{t('右扩、维护、左缩、记录', 'Expand, maintain, shrink, record')}</h2>
          <p>{t('示例状态是“窗口内不能出现重复字符”，但四拍循环可以替换成任何可增量维护的条件。', 'This example forbids duplicate characters, but the four-beat loop works with any condition that can be maintained incrementally.')}</p>
        </div>
        <div className="sliding-window-counter">{activeStep + 1}<span>/ {steps.length}</span></div>
      </header>

      <div className="sliding-window-phases">
        {phases.map(([id, label, detail]) => (
          <div className={step.phase === id ? `active ${id}` : ''} key={id}>
            <strong>{label}</strong>
            <span>{detail}</span>
          </div>
        ))}
      </div>

      <div className="sliding-window-step-copy">
        <strong>{step.title}</strong>
        <span>{step.detail}</span>
      </div>

      <div className="sliding-window-array-wrap">
        <div className="sliding-window-array" aria-label={t('滑动窗口数组', 'Sliding-window array')}>
          {SLIDING_WINDOW_VALUES.map((value, index) => {
            const inWindow = step.left <= index && index <= step.right;
            const isLeft = index === step.left;
            const isRight = index === step.right;
            const isRemoved = index === step.removedIndex;
            return (
              <div
                className={`sliding-window-cell ${inWindow ? 'in-window' : ''} ${isRemoved ? 'removed' : ''}`}
                key={`${value}-${index}`}
              >
                <small>{index}</small>
                <strong>{value}</strong>
                <span>
                  {isLeft && <b className="left">L</b>}
                  {isRight && <b className="right">R</b>}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="sliding-window-state">
        <div>
          <span>{t('当前窗口', 'Current window')}</span>
          <strong>{windowText || '∅'}</strong>
          <small>[{step.left}, {step.right}]</small>
        </div>
        <div>
          <span>{t('增量状态', 'Incremental state')}</span>
          <strong>{step.state}</strong>
          <small>{t('只 add / remove 边界元素', 'add or remove boundary elements only')}</small>
        </div>
        <div className={step.valid ? 'valid' : 'invalid'}>
          <span>{t('条件', 'Condition')}</span>
          <strong>{step.valid ? t('合法', 'valid') : t('不合法', 'invalid')}</strong>
          <small>{step.valid ? t('可以考虑记录', 'ready to record') : t('必须继续左缩', 'keep shrinking left')}</small>
        </div>
        <div>
          <span>best</span>
          <strong>{step.best}</strong>
          <small>{t('最长合法窗口', 'longest valid window')}</small>
        </div>
      </div>

      <div className="sliding-window-timing">
        <div>
          <strong>{t('求最长合法窗口', 'Longest valid window')}</strong>
          <span>{t('while 不合法：左缩', 'while invalid: shrink left')}</span>
          <em>{t('while 结束后 update max', 'update max after the loop')}</em>
        </div>
        <div>
          <strong>{t('求最短满足窗口', 'Shortest satisfying window')}</strong>
          <span>{t('while 合法：先记录，再左缩', 'while valid: record, then shrink')}</span>
          <em>{t('在 while 内 update min', 'update min inside the loop')}</em>
        </div>
      </div>

      <div className="sliding-window-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择滑动窗口演示步骤', 'Select a sliding-window step')}
        />
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          disabled={activeStep === steps.length - 1}
        >
          {t('下一步', 'Next')}
        </button>
      </div>
    </section>
  );
}

const LONGEST_SUBSTRING_VALUES = ['a', 'b', 'c', 'a'];

const LONGEST_SUBSTRING_TEMPLATE_LINES = [
  ['loop', 'for right in range(len(items)):'],
  ['add', '    add(state, items[right])'],
  ['while', '    while window_is_invalid(state):'],
  ['remove', '        remove(state, items[left])'],
  ['left', '        left += 1'],
  ['record', '    answer = max(answer, right - left + 1)'],
];

const LONGEST_SUBSTRING_FILLED_LINES = [
  ['loop', 'for right in range(len(s)):'],
  ['add', '    count[s[right]] += 1'],
  ['while', '    while count[s[right]] > 1:'],
  ['remove', '        count[s[left]] -= 1'],
  ['left', '        left += 1'],
  ['record', '    max_length = max(max_length, right - left + 1)'],
];

const LONGEST_SUBSTRING_STEPS = [
  {
    phase: '扩张',
    title: 'right = 0：加入 a',
    detail: '先执行 count[s[right]] += 1。a 的频次变成 1，窗口仍然合法。',
    left: 0,
    right: 0,
    includedRight: 0,
    state: 'a × 1',
    best: 0,
    activeLines: ['loop', 'add'],
  },
  {
    phase: '记录',
    title: '记录窗口 a',
    detail: '窗口合法，长度是 0 - 0 + 1 = 1，max_length 更新为 1。',
    left: 0,
    right: 0,
    includedRight: 0,
    state: 'a × 1',
    best: 1,
    activeLines: ['record'],
  },
  {
    phase: '扩张',
    title: 'right = 1：复用窗口并加入 b',
    detail: 'count 没有清空；b 的频次从 0 变成 1。',
    left: 0,
    right: 1,
    includedRight: 1,
    state: 'a × 1 · b × 1',
    best: 1,
    activeLines: ['loop', 'add'],
  },
  {
    phase: '记录',
    title: '记录窗口 ab',
    detail: '所有频次都不超过 1。长度是 1 - 0 + 1 = 2。',
    left: 0,
    right: 1,
    includedRight: 1,
    state: 'a × 1 · b × 1',
    best: 2,
    activeLines: ['record'],
  },
  {
    phase: '扩张',
    title: 'right = 2：继续加入 c',
    detail: 'c 的频次从 0 变成 1，旧状态继续复用。',
    left: 0,
    right: 2,
    includedRight: 2,
    state: 'a × 1 · b × 1 · c × 1',
    best: 2,
    activeLines: ['loop', 'add'],
  },
  {
    phase: '记录',
    title: '记录窗口 abc',
    detail: '长度是 2 - 0 + 1 = 3，max_length 更新为 3。',
    left: 0,
    right: 2,
    includedRight: 2,
    state: 'a × 1 · b × 1 · c × 1',
    best: 3,
    activeLines: ['record'],
  },
  {
    phase: '扩张',
    title: 'right = 3：先加入第二个 a',
    detail: 'a 的频次变成 2，while count[s[right]] > 1 被触发。',
    left: 0,
    right: 3,
    includedRight: 3,
    invalid: true,
    state: 'a × 2 · b × 1 · c × 1',
    best: 3,
    activeLines: ['loop', 'add', 'while'],
  },
  {
    phase: '收缩',
    title: '移除旧 a，left 从 0 变成 1',
    detail: 'count[s[left]] 减 1 后，a 的频次恢复为 1；再让 left += 1。',
    left: 1,
    right: 3,
    includedRight: 3,
    removedIndex: 0,
    state: 'a × 1 · b × 1 · c × 1',
    best: 3,
    activeLines: ['remove', 'left'],
  },
  {
    phase: '记录',
    title: '记录窗口 bca，最优值仍是 3',
    detail: 'while 已结束，窗口重新合法。right 没有回头，count 也没有重建。',
    left: 1,
    right: 3,
    includedRight: 3,
    state: 'a × 1 · b × 1 · c × 1',
    best: 3,
    activeLines: ['record'],
  },
];

const LONGEST_SUBSTRING_STEPS_EN = [
  ['Expand', 'right = 0: add a', 'First run count[s[right]] += 1. The count of a becomes 1, so the window stays valid.'],
  ['Record', 'Record window a', 'The valid window has length 0 - 0 + 1 = 1, so max_length becomes 1.'],
  ['Expand', 'right = 1: reuse the window and add b', 'Do not reset count. The count of b changes from 0 to 1.'],
  ['Record', 'Record window ab', 'Every count is at most 1. The length is 1 - 0 + 1 = 2.'],
  ['Expand', 'right = 2: add c', 'The count of c changes from 0 to 1 while the previous state remains in place.'],
  ['Record', 'Record window abc', 'The length is 2 - 0 + 1 = 3, so max_length becomes 3.'],
  ['Expand', 'right = 3: add the second a first', 'The count of a becomes 2, which triggers while count[s[right]] > 1.'],
  ['Shrink', 'Remove the old a; left moves from 0 to 1', 'After decrementing count[s[left]], the count of a returns to 1. Then increment left.'],
  ['Record', 'Record window bca; the best remains 3', 'The loop has ended and the window is valid again. right never moved backward, and count was not rebuilt.'],
];

function LongestSubstringVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = isEnglish
    ? LONGEST_SUBSTRING_STEPS.map((step, index) => ({
      ...step,
      phase: LONGEST_SUBSTRING_STEPS_EN[index][0],
      title: LONGEST_SUBSTRING_STEPS_EN[index][1],
      detail: LONGEST_SUBSTRING_STEPS_EN[index][2],
    }))
    : LONGEST_SUBSTRING_STEPS;
  const step = steps[activeStep];
  const windowText = step.includedRight >= step.left
    ? LONGEST_SUBSTRING_VALUES.slice(step.left, step.includedRight + 1).join('')
    : '∅';

  const renderCode = (lines) => lines.map(([id, code]) => (
    <span className={step.activeLines.includes(id) ? 'active' : ''} key={id}>
      {code}
    </span>
  ));

  return (
    <section className="longest-substring-visual" aria-label={t('最长无重复子串代码映射演示', 'Longest-substring code mapping walkthrough')}>
      <header className="longest-substring-header">
        <div>
          <p className="eyebrow">{t('模板到具体代码', 'Template → concrete code')}</p>
          <h2>{t('同一行骨架，逐项填入本题条件', 'Fill one shared skeleton with this problem’s condition')}</h2>
          <p>{t('拖动步骤，左边的抽象操作与右边的实际代码会同时高亮。', 'Move through the steps to highlight the abstract operation and its concrete code together.')}</p>
        </div>
        <div className="longest-substring-counter">
          {activeStep + 1}<span>/ {steps.length}</span>
        </div>
      </header>

      <div className="longest-substring-code-map">
        <div>
          <strong>{t('万能模板骨架', 'General template')}</strong>
          <pre><code>{renderCode(LONGEST_SUBSTRING_TEMPLATE_LINES)}</code></pre>
        </div>
        <div>
          <strong>{t('Longest Substring 填空结果', 'Longest Substring specialization')}</strong>
          <pre><code>{renderCode(LONGEST_SUBSTRING_FILLED_LINES)}</code></pre>
        </div>
      </div>

      <div className="longest-substring-step-copy">
        <span>{step.phase}</span>
        <strong>{step.title}</strong>
        <p>{step.detail}</p>
      </div>

      <div className="longest-substring-array" aria-label={t('字符串 abca 的窗口状态', 'Window state for string abca')}>
        {LONGEST_SUBSTRING_VALUES.map((value, index) => {
          const inWindow = step.left <= index && index <= step.includedRight;
          const isCandidate = step.candidate && index === step.right;
          const isRemoved = index === step.removedIndex;
          return (
            <div
              className={`longest-substring-cell ${inWindow ? 'in-window' : ''} ${isCandidate ? 'candidate' : ''} ${isRemoved ? 'removed' : ''}`}
              key={`${value}-${index}`}
            >
              <small>{index}</small>
              <strong>{value}</strong>
              <span>
                {index === step.left && <b className="left">L</b>}
                {index === step.right && <b className="right">R</b>}
              </span>
            </div>
          );
        })}
      </div>

      <div className="longest-substring-state">
        <div>
          <span>window</span>
          <strong>{windowText}</strong>
          <small>s[{step.left}:{step.includedRight + 1}]</small>
        </div>
        <div className={step.invalid ? 'invalid' : ''}>
          <span>frequency state</span>
          <strong>{step.state}</strong>
          <small>{step.invalid ? t('存在频次大于 1', 'a count exceeds 1') : t('所有频次都不超过 1', 'all counts are at most 1')}</small>
        </div>
        <div>
          <span>max_length</span>
          <strong>{step.best}</strong>
          <small>{t('只在窗口合法时记录', 'record only when the window is valid')}</small>
        </div>
      </div>

      <div className="longest-substring-comparison">
        <div>
          <strong>{t('外层 loop left', 'Outer loop over left')}</strong>
          <code>abc… · bc… · c…</code>
          <span>{t('每个起点重建 count，right 反复扫描：O(n²)', 'Rebuild count for every start and rescan with right: O(n²)')}</span>
        </div>
        <div>
          <strong>{t('外层 loop right', 'Outer loop over right')}</strong>
          <code>{t('R → n 次 · L → 最多 n 次', 'R → n moves · L → at most n moves')}</code>
          <span>{t('窗口和 count 跨轮复用，总移动不超过 2n：O(n)', 'Reuse the window and count across rounds; total pointer movement is at most 2n: O(n)')}</span>
        </div>
      </div>

      <div className="longest-substring-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择最长无重复子串演示步骤', 'Select a longest-substring step')}
        />
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          disabled={activeStep === steps.length - 1}
        >
          {t('下一步', 'Next')}
        </button>
      </div>
    </section>
  );
}

const SLIDING_WINDOW_PATTERNS = [
  {
    id: 'unique',
    number: '3',
    title: 'Longest Substring',
    shape: '变长 · 求最长合法',
    state: 'count',
    add: 'count[s[right]] += 1',
    control: 'while：count[s[right]] > 1',
    shrinkStep: '可能连续移出左端字符',
    shrink: 'count[s[left]] -= 1，再移动 left',
    beforeRecord: '不记录',
    afterRecord: '窗口合法后更新 max',
    invariant: '窗口内每个字符至多出现一次',
    formula: 'invalid = count[s[right]] > 1',
    tone: 'green',
  },
  {
    id: 'replace',
    number: '424',
    title: 'Character Replacement',
    shape: '变长 · 求最长合法',
    state: 'count + max_freq',
    add: '更新字符频次与最高频次',
    control: 'while：所需替换数 > k',
    shrinkStep: '可能连续移出左端字符',
    shrink: '左端字符频次减 1',
    beforeRecord: '不记录',
    afterRecord: '窗口合法后更新 max',
    invariant: 'len(window) - max_freq <= k',
    formula: 'replacements = length - max_freq',
    tone: 'amber',
  },
  {
    id: 'permutation',
    number: '567',
    title: 'Permutation in String',
    shape: '定长 · 长度为 |s1|',
    state: 'need[26] + window[26]',
    add: '加入 s2[right]',
    control: 'if：窗口长度 > |s1|',
    shrinkStep: '最多移出一个左端字符',
    shrink: 'window[s2[left]] -= 1，再移动 left',
    beforeRecord: '不记录',
    afterRecord: '窗口满 |s1| 时比较频次表',
    invariant: '窗口始终不长于 |s1|',
    formula: 'match = window == need',
    tone: 'blue',
  },
  {
    id: 'minimum',
    number: '76',
    title: 'Minimum Window',
    shape: '变长 · 求最短满足',
    state: 'need/window + have',
    add: '达到某字符阈值时 have += 1',
    control: 'while：have == required',
    shrinkStep: '每轮记录候选后移出左端字符',
    shrink: '更新 have/window，再删除左端字符',
    beforeRecord: '更新 min',
    afterRecord: '不记录，候选已在调整前保存',
    invariant: 'have 只数已经达到所需频次的字符种类',
    formula: 'valid = have == len(need)',
    tone: 'rose',
  },
  {
    id: 'maximum',
    number: '239',
    title: 'Sliding Window Maximum',
    shape: '定长 · 每窗求最大值',
    state: '递减 deque，存下标',
    add: '删弱势队尾，再 append right',
    control: 'if：窗口长度 > k',
    shrinkStep: '最多移出一个过期位置',
    shrink: '若队首是 left 就删除，再移动 left',
    beforeRecord: '不记录',
    afterRecord: '窗口满 k 时读取 deque[0]',
    invariant: '下标递增，值递减，队首是最大值',
    formula: 'maximum = nums[deque[0]]',
    tone: 'violet',
  },
];

const SLIDING_WINDOW_PATTERNS_EN = {
  unique: {
    shape: 'Variable length · longest valid',
    add: 'count[s[right]] += 1',
    control: 'while count[s[right]] > 1',
    shrinkStep: 'May remove several characters from the left',
    shrink: 'count[s[left]] -= 1, then move left',
    beforeRecord: 'do not record',
    afterRecord: 'update max after the window is valid',
    invariant: 'Each character appears at most once in the window',
  },
  replace: {
    shape: 'Variable length · longest valid',
    add: 'Update the character count and max_freq',
    control: 'while replacements needed > k',
    shrinkStep: 'May remove several characters from the left',
    shrink: 'Decrement the count of the leftmost character',
    beforeRecord: 'do not record',
    afterRecord: 'update max after the window is valid',
    invariant: 'len(window) - max_freq <= k',
  },
  permutation: {
    shape: 'Fixed length · |s1|',
    add: 'Add s2[right]',
    control: 'if window length > |s1|',
    shrinkStep: 'Remove at most one character from the left',
    shrink: 'window[s2[left]] -= 1, then move left',
    beforeRecord: 'do not record',
    afterRecord: 'compare frequency tables when the window reaches |s1|',
    invariant: 'The window never grows beyond |s1|',
  },
  minimum: {
    shape: 'Variable length · shortest satisfying',
    add: 'Increment have when a character reaches its target count',
    control: 'while have == required',
    shrinkStep: 'Record a candidate, then remove one character from the left',
    shrink: 'Update have and window, then remove the leftmost character',
    beforeRecord: 'update min',
    afterRecord: 'do not record; the candidate was saved before shrinking',
    invariant: 'have counts only character types that meet their required frequency',
  },
  maximum: {
    shape: 'Fixed length · maximum per window',
    add: 'Remove weaker tail entries, then append right',
    control: 'if window length > k',
    shrinkStep: 'Remove at most one expired position',
    shrink: 'If deque[0] equals left, remove it; then move left',
    beforeRecord: 'do not record',
    afterRecord: 'read deque[0] when the window reaches k',
    invariant: 'Indices increase, values decrease, and the front is the maximum',
  },
};

function SlidingWindowPatternAtlas() {
  const { isEnglish, t } = useUiCopy();
  const [activePattern, setActivePattern] = useState('unique');
  const basePattern = SLIDING_WINDOW_PATTERNS.find(({ id }) => id === activePattern)
    ?? SLIDING_WINDOW_PATTERNS[0];
  const pattern = isEnglish
    ? { ...basePattern, ...SLIDING_WINDOW_PATTERNS_EN[basePattern.id] }
    : basePattern;

  return (
    <section className={`sliding-pattern-atlas ${pattern.tone}`} aria-label={t('五道滑动窗口题模板对照', 'Five sliding-window patterns compared')}>
      <header className="sliding-pattern-header">
        <div>
          <p className="eyebrow">{t('同一套骨架，两种调整规则', 'One skeleton · two resize rules')}</p>
          <h2>{t('先分定长与变长，再选择 if 或 while', 'Choose fixed or variable length first, then choose if or while')}</h2>
          <p>{t('共同顺序是右端加入、调整左端、记录答案；调整次数由窗口类型决定。', 'Every pattern adds on the right, adjusts the left, and records an answer. The window type determines how often adjustment runs.')}</p>
        </div>
        <code>{pattern.formula}</code>
      </header>

      <div className="sliding-pattern-tabs" role="tablist" aria-label={t('选择滑动窗口题目', 'Choose a sliding-window problem')}>
        {SLIDING_WINDOW_PATTERNS.map((candidate) => (
          <button
            type="button"
            className={candidate.id === activePattern ? 'active' : ''}
            onClick={() => setActivePattern(candidate.id)}
            role="tab"
            aria-selected={candidate.id === activePattern}
            key={candidate.id}
          >
            <span>LC {candidate.number}</span>
            <strong>{candidate.title}</strong>
          </button>
        ))}
      </div>

      <div className="sliding-pattern-summary">
        <div>
          <span>{t('窗口形状', 'Window shape')}</span>
          <strong>{pattern.shape}</strong>
        </div>
        <div>
          <span>{t('增量状态', 'Incremental state')}</span>
          <strong>{pattern.state}</strong>
        </div>
        <div>
          <span>{t('窗口不变量', 'Window invariant')}</span>
          <strong>{pattern.invariant}</strong>
        </div>
      </div>

      <div className="sliding-pattern-flow">
        <div>
          <span>{t('1 · right 右扩', '1 · Expand right')}</span>
          <strong>{pattern.add}</strong>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('2 · 选择调整规则', '2 · Choose the adjustment rule')}</span>
          <strong>{pattern.control}</strong>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('3 · 调整 left', '3 · Adjust left')}</span>
          <strong>{pattern.shrinkStep}</strong>
          <small>{t('移动前记录：', 'Before moving: ')}{pattern.beforeRecord}；remove: {pattern.shrink}</small>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('4 · 窗口调整后', '4 · After adjustment')}</span>
          <strong>{pattern.afterRecord}</strong>
        </div>
      </div>
    </section>
  );
}

const THREE_SUM_VALUES = [-4, -1, -1, 0, 1, 2];

const THREE_SUM_STEPS = [
  {
    title: '排序并初始化',
    i: 0,
    left: 1,
    right: 5,
    sum: -3,
    tone: 'low',
    action: '-3 < 0，和太小；固定 i 和 right，left 右移。',
    results: [],
  },
  {
    title: 'left 继续右移',
    i: 0,
    left: 2,
    right: 5,
    sum: -3,
    tone: 'low',
    action: '仍然小于 0。虽然值还是 -1，但这一轮尚未命中，继续移动 left。',
    results: [],
  },
  {
    title: '排除更小的组合',
    i: 0,
    left: 3,
    right: 5,
    sum: -2,
    tone: 'low',
    action: '-2 < 0。排序保证 left 左边的候选都不会更大，可以安全排除。',
    results: [],
  },
  {
    title: '第一个锚点结束',
    i: 0,
    left: 4,
    right: 5,
    sum: -1,
    tone: 'low',
    action: '-1 < 0，left 再右移就会与 right 相遇；固定 -4 时没有答案。',
    results: [],
  },
  {
    title: '固定 -1，命中第一组',
    i: 1,
    left: 2,
    right: 5,
    sum: 0,
    tone: 'hit',
    action: '-1 + -1 + 2 = 0，记录答案，然后两端跳过重复值并同时内收。',
    results: [[-1, -1, 2]],
  },
  {
    title: '同一锚点命中第二组',
    i: 1,
    left: 3,
    right: 4,
    sum: 0,
    tone: 'hit',
    action: '-1 + 0 + 1 = 0，再记录一组；随后 left 与 right 交错。',
    results: [[-1, -1, 2], [-1, 0, 1]],
  },
  {
    title: '跳过重复锚点',
    i: 2,
    left: null,
    right: null,
    sum: null,
    tone: 'skip',
    action: 'nums[2] == nums[1]，若再次固定 -1 只会生成重复答案，直接 continue。',
    skippedAnchor: 2,
    results: [[-1, -1, 2], [-1, 0, 1]],
  },
  {
    title: '扫描完成',
    i: 3,
    left: 4,
    right: 5,
    sum: 3,
    tone: 'high',
    action: '0 + 1 + 2 = 3，和太大；right 左移后指针相遇，全部搜索结束。',
    results: [[-1, -1, 2], [-1, 0, 1]],
  },
];

const THREE_SUM_STEPS_EN = [
  ['Sort and initialize', '-3 < 0, so the sum is too small. Hold i and right fixed, then move left rightward.'],
  ['Keep moving left rightward', 'The sum is still below 0. The value is still -1, but this pair has not produced a hit, so continue moving left.'],
  ['Eliminate smaller combinations', '-2 < 0. Sorting guarantees that candidates to the left cannot be larger, so this block can be discarded safely.'],
  ['Finish the first anchor', '-1 < 0. One more move would make left meet right, so anchor -4 has no solution.'],
  ['Anchor -1 and find the first triplet', '-1 + -1 + 2 = 0. Record it, skip duplicates on both ends, and move both pointers inward.'],
  ['Find the second triplet with the same anchor', '-1 + 0 + 1 = 0. Record it; then left and right cross.'],
  ['Skip the duplicate anchor', 'nums[2] == nums[1]. Anchoring -1 again would only duplicate previous answers, so continue.'],
  ['Scan complete', '0 + 1 + 2 = 3, so the sum is too large. Move right leftward; the pointers meet and the search ends.'],
];

function ThreeSumVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = isEnglish
    ? THREE_SUM_STEPS.map((step, index) => ({
      ...step,
      title: THREE_SUM_STEPS_EN[index][0],
      action: THREE_SUM_STEPS_EN[index][1],
    }))
    : THREE_SUM_STEPS;
  const step = steps[activeStep];
  const selectedValues = step.left === null
    ? [THREE_SUM_VALUES[step.i]]
    : [THREE_SUM_VALUES[step.i], THREE_SUM_VALUES[step.left], THREE_SUM_VALUES[step.right]];

  return (
    <section className="three-sum-visual" aria-label={t('3Sum 双指针演示', '3Sum two-pointer walkthrough')}>
      <header className="three-sum-header">
        <div>
          <p className="eyebrow">{t('双指针可视化', 'Two pointers visual')}</p>
          <h2>{t('3Sum：固定 i，收缩 left / right', '3Sum: fix i and move left / right inward')}</h2>
          <p>{t('输入 [-1, 0, 1, 2, -1, -4]，先排序，再观察每次移动为什么能排除一批组合。', 'Sort input [-1, 0, 1, 2, -1, -4], then see why each pointer move eliminates a block of combinations.')}</p>
        </div>
        <div className="three-sum-counter">{activeStep + 1}<span>/ {steps.length}</span></div>
      </header>

      <div className="three-sum-step-copy">
        <strong>{step.title}</strong>
        <span>{step.action}</span>
      </div>

      <div className="three-sum-array-wrap">
      <div className="three-sum-array" aria-label={t('排序后的数组', 'Sorted array')}>
          {THREE_SUM_VALUES.map((value, index) => {
            const roles = [];
            if (index === step.i) roles.push('i');
            if (index === step.left) roles.push('L');
            if (index === step.right) roles.push('R');
            const isSkipped = index === step.skippedAnchor;

            return (
              <div
                className={`three-sum-cell ${index === step.i ? 'anchor' : ''} ${index === step.left ? 'left' : ''} ${index === step.right ? 'right' : ''} ${isSkipped ? 'skipped' : ''}`}
                key={`${value}-${index}`}
              >
                <small>index {index}</small>
                <strong>{value}</strong>
                <span>{roles.join(' · ') || '·'}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="three-sum-state">
        <div>
          <span>{t('当前选择', 'Current selection')}</span>
          <strong>{selectedValues.join(' + ')}</strong>
        </div>
        <div className={`three-sum-sum ${step.tone}`}>
          <span>sum</span>
          <strong>{step.sum === null ? 'skip' : step.sum}</strong>
        </div>
        <div>
          <span>{t('动作', 'Action')}</span>
          <strong>{step.tone === 'low' ? 'left →' : step.tone === 'high' ? '← right' : step.tone === 'hit' ? t('记录 + 内收', 'record + move inward') : t('跳过重复', 'skip duplicate')}</strong>
        </div>
      </div>

      <div className="three-sum-results">
        <span>{t('已找到', 'Found')}</span>
        <div>
          {step.results.length > 0
            ? step.results.map((triplet) => <strong key={triplet.join(',')}>[{triplet.join(', ')}]</strong>)
            : <em>{t('尚未命中', 'No hit yet')}</em>}
        </div>
      </div>

      <div className="three-sum-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          {t('上一步', 'Previous')}
        </button>
        <div className="three-sum-dots">
          {steps.map((candidate, index) => (
            <button
              type="button"
              className={index === activeStep ? 'active' : ''}
              aria-label={`${t('跳到步骤', 'Go to step')} ${index + 1}: ${candidate.title}`}
              onClick={() => setActiveStep(index)}
              key={candidate.title}
            />
          ))}
        </div>
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          disabled={activeStep === steps.length - 1}
        >
          {t('下一步', 'Next')}
        </button>
      </div>
    </section>
  );
}

const RAIN_WATER_HEIGHTS = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1];

function buildRainWaterSteps(heights) {
  const steps = [];
  const resolved = Array(heights.length).fill(false);
  const waterByIndex = Array(heights.length).fill(0);
  let left = 0;
  let right = heights.length - 1;
  let leftMax = 0;
  let rightMax = 0;
  let total = 0;

  while (left <= right) {
    leftMax = Math.max(leftMax, heights[left]);
    rightMax = Math.max(rightMax, heights[right]);

    if (leftMax <= rightMax) {
      const current = left;
      const added = leftMax - heights[current];
      total += added;
      resolved[current] = true;
      waterByIndex[current] = added;
      steps.push({
        side: 'left',
        current,
        left,
        right,
        leftMax,
        rightMax,
        added,
        total,
        resolved: [...resolved],
        waterByIndex: [...waterByIndex],
        title: `结算左侧 index ${current}`,
        titleEn: `Resolve left index ${current}`,
        note: leftMax === rightMax
          ? `leftMax = rightMax = ${leftMax}，任选一侧都安全；这里先处理左侧。`
          : `leftMax ${leftMax} < rightMax ${rightMax}，右边已有足够高的墙，左侧水位已经确定。`,
        noteEn: leftMax === rightMax
          ? `leftMax = rightMax = ${leftMax}. Either side is safe; resolve the left side first.`
          : `leftMax ${leftMax} < rightMax ${rightMax}. The right side already has a tall enough wall, so the left water level is fixed.`,
      });
      left += 1;
    } else {
      const current = right;
      const added = rightMax - heights[current];
      total += added;
      resolved[current] = true;
      waterByIndex[current] = added;
      steps.push({
        side: 'right',
        current,
        left,
        right,
        leftMax,
        rightMax,
        added,
        total,
        resolved: [...resolved],
        waterByIndex: [...waterByIndex],
        title: `结算右侧 index ${current}`,
        titleEn: `Resolve right index ${current}`,
        note: `rightMax ${rightMax} < leftMax ${leftMax}，左边已有足够高的墙，右侧水位已经确定。`,
        noteEn: `rightMax ${rightMax} < leftMax ${leftMax}. The left side already has a tall enough wall, so the right water level is fixed.`,
      });
      right -= 1;
    }
  }

  return steps;
}

const RAIN_WATER_STEPS = buildRainWaterSteps(RAIN_WATER_HEIGHTS);

function RainWaterVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const step = RAIN_WATER_STEPS[activeStep];
  const maxHeight = Math.max(...RAIN_WATER_HEIGHTS);

  useEffect(() => {
    if (!isPlaying) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setActiveStep((current) => {
        if (current >= RAIN_WATER_STEPS.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 950);

    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const jumpToStep = (index) => {
    setIsPlaying(false);
    setActiveStep(index);
  };

  return (
    <section className="rain-water-visual" aria-label={t('接雨水双指针演示', 'Trapping Rain Water two-pointer walkthrough')}>
      <header className="rain-water-header">
        <div>
          <p className="eyebrow">{t('双指针可视化', 'Two pointers visual')}</p>
          <h2>Trapping Rain Water</h2>
          <p>{t('较低的历史最高墙先结算：它这一侧的水位已经被另一侧兜住。', 'Resolve the side with the lower running maximum first; the opposite side already guarantees its water level.')}</p>
        </div>
        <div className="rain-water-total">
          <span>{t('累计水量', 'Total water')}</span>
          <strong>{step.total}</strong>
          <small>/ 6</small>
        </div>
      </header>

      <div className="rain-water-rule">
        <span className={step.side === 'left' ? 'active left' : 'left'}>leftMax = {step.leftMax}</span>
        <strong>{step.leftMax <= step.rightMax ? '≤' : '>'}</strong>
        <span className={step.side === 'right' ? 'active right' : 'right'}>rightMax = {step.rightMax}</span>
        <em>→ {step.side === 'left' ? t('结算左侧', 'resolve left') : t('结算右侧', 'resolve right')}</em>
      </div>

      <div className="rain-water-chart-wrap">
        <div className="rain-water-chart" aria-label={t('柱状高度与已结算雨水', 'Bar heights and resolved water')}>
          {RAIN_WATER_HEIGHTS.map((height, index) => {
            const water = step.waterByIndex[index];
            const isResolved = step.resolved[index];
            const isCurrent = index === step.current;

            return (
              <div className={`rain-water-column ${isResolved ? 'resolved' : ''} ${isCurrent ? `current ${step.side}` : ''}`} key={`${height}-${index}`}>
                <div className="rain-water-cells">
                  {Array.from({ length: maxHeight }, (_, rowIndex) => {
                    const level = maxHeight - rowIndex;
                    const isBar = level <= height;
                    const isWater = isResolved && level > height && level <= height + water;
                    return (
                      <span className={isBar ? 'bar' : isWater ? 'water' : 'empty'} key={level} />
                    );
                  })}
                </div>
                <strong>{height}</strong>
                <small>{index}</small>
                <div className="rain-water-pointers">
                  {index === step.left && <b className="left">L</b>}
                  {index === step.right && <b className="right">R</b>}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="rain-water-explain">
        <div>
          <span>{isEnglish ? step.titleEn : step.title}</span>
          <strong>{isEnglish ? step.noteEn : step.note}</strong>
        </div>
        <div className="rain-water-formula">
          <span>{t('本格水量', 'Water at this index')}</span>
          <strong>
            {step.side === 'left' ? step.leftMax : step.rightMax}
            {' - '}{RAIN_WATER_HEIGHTS[step.current]} = {step.added}
          </strong>
        </div>
      </div>

      <div className="rain-water-legend">
        <span><i className="bar" />{t('柱子', 'bar')}</span>
        <span><i className="water" />{t('已确定的水', 'resolved water')}</span>
        <strong>{t('未处理区域保持空白', 'Unresolved cells stay blank')}</strong>
      </div>

      <div className="rain-water-controls">
        <button type="button" onClick={() => jumpToStep(Math.max(0, activeStep - 1))} disabled={activeStep === 0}>
          {t('上一步', 'Previous')}
        </button>
        <button
          type="button"
          className="rain-water-play"
          onClick={() => {
            if (activeStep === RAIN_WATER_STEPS.length - 1) {
              setActiveStep(0);
              setIsPlaying(true);
            } else {
              setIsPlaying((current) => !current);
            }
          }}
        >
          {isPlaying ? t('暂停', 'Pause') : activeStep === RAIN_WATER_STEPS.length - 1 ? t('重新播放', 'Replay') : t('播放', 'Play')}
        </button>
        <input
          type="range"
          min="0"
          max={RAIN_WATER_STEPS.length - 1}
          value={activeStep}
          onChange={(event) => jumpToStep(Number(event.target.value))}
          aria-label={t('选择接雨水演示步骤', 'Select a trapping-rain-water step')}
        />
        <span>{activeStep + 1} / {RAIN_WATER_STEPS.length}</span>
        <button
          type="button"
          onClick={() => jumpToStep(Math.min(RAIN_WATER_STEPS.length - 1, activeStep + 1))}
          disabled={activeStep === RAIN_WATER_STEPS.length - 1}
        >
          {t('下一步', 'Next')}
        </button>
      </div>
    </section>
  );
}

function IntervalBar({ domain, interval, isActive = false, isMuted = false }) {
  const { isEnglish } = useUiCopy();
  const left = intervalPercent(interval.start, domain);
  const right = intervalPercent(interval.end, domain);
  const width = Math.max(right - left, 1.4);
  const label = isEnglish
    ? interval.label
    : interval.label
      .replace(/^current /, '当前 ')
      .replace(/^merged /, '合并后 ')
      .replace(/^output /, '输出 ')
      .replace(/^new /, '新区间 ')
      .replace(/^best /, '最优 ')
      .replace(/ len /, ' 长度 ');

  return (
    <div
      className={`interval-bar ${interval.kind ?? ''} ${isActive ? 'active' : ''} ${isMuted ? 'muted' : ''}`}
      style={{ left: `${left}%`, width: `${width}%` }}
    >
      <span>{label}</span>
    </div>
  );
}

function intervalPercent(value, [min, max]) {
  if (max === min) {
    return 0;
  }

  return ((value - min) / (max - min)) * 100;
}

function buildIntervalTicks([min, max]) {
  const width = max - min;
  const step = width <= 8 ? 1 : Math.ceil(width / 6);
  const ticks = [];
  for (let value = min; value <= max; value += step) {
    ticks.push(value);
  }
  if (ticks[ticks.length - 1] !== max) {
    ticks.push(max);
  }
  return ticks;
}

const INTEGRAL_N_LEVELS = [2, 4, 8, 16, 32, 64, 128];

function projectIntegralPoint(x, y, z) {
  return {
    x: 100 + 430 * x + 150 * y,
    y: 405 + 40 * x - 90 * y - 310 * z,
  };
}

function ratioHeight(values) {
  let sum = 0;
  let squareSum = 0;

  values.forEach((value) => {
    sum += value;
    squareSum += value * value;
  });

  return sum === 0 ? 0 : squareSum / sum;
}

function seededUniform(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function buildIntegralCloud(n, count = 190) {
  const points = [];
  let ratioTotal = 0;

  for (let sampleIndex = 0; sampleIndex < count; sampleIndex += 1) {
    const random = seededUniform(91_337 + sampleIndex * 7_919);
    const values = Array.from({ length: n }, () => random());
    const mean = values.reduce((total, value) => total + value, 0) / n;
    const secondMoment = values.reduce((total, value) => total + value * value, 0) / n;
    const ratio = ratioHeight(values);
    const projected = projectIntegralPoint(mean, secondMoment, ratio);
    ratioTotal += ratio;
    points.push({
      ...projected,
      mean,
      secondMoment,
      ratio,
      key: sampleIndex,
    });
  }

  return {
    points,
    estimate: ratioTotal / count,
  };
}

function buildIntegralSurface(gridSize = 12) {
  const cells = [];

  for (let yIndex = gridSize - 1; yIndex >= 0; yIndex -= 1) {
    for (let xIndex = gridSize - 1; xIndex >= 0; xIndex -= 1) {
      const x0 = xIndex / gridSize;
      const x1 = (xIndex + 1) / gridSize;
      const y0 = yIndex / gridSize;
      const y1 = (yIndex + 1) / gridSize;
      const corners = [
        [x0, y0],
        [x1, y0],
        [x1, y1],
        [x0, y1],
      ];
      const points = corners.map(([x, y]) => {
        const height = x + y === 0 ? 0 : (x * x + y * y) / (x + y);
        return projectIntegralPoint(x, y, height);
      });
      const averageHeight = corners.reduce((total, [x, y]) => (
        total + (x + y === 0 ? 0 : (x * x + y * y) / (x + y))
      ), 0) / corners.length;

      cells.push({
        key: `${xIndex}-${yIndex}`,
        points: points.map((point) => `${point.x},${point.y}`).join(' '),
        averageHeight,
      });
    }
  }

  return cells;
}

function IntegralAxis({ from, to, label, labelX, labelY }) {
  return (
    <g className="integral-axis">
      <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} />
      <circle cx={to.x} cy={to.y} r="3" />
      <text x={labelX} y={labelY}>{label}</text>
    </g>
  );
}

function HighDimensionalIntegralVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('surface');
  const [levelIndex, setLevelIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const n = INTEGRAL_N_LEVELS[levelIndex];
  const surface = useMemo(() => buildIntegralSurface(), []);
  const cloud = useMemo(() => buildIntegralCloud(n), [n]);
  const origin = projectIntegralPoint(0, 0, 0);
  const xEnd = projectIntegralPoint(1, 0, 0);
  const yEnd = projectIntegralPoint(0, 1, 0);
  const zEnd = projectIntegralPoint(0, 0, 1);
  const target = projectIntegralPoint(0.5, 1 / 3, 2 / 3);

  useEffect(() => {
    if (!isPlaying || mode !== 'cloud') {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setLevelIndex((current) => (current + 1) % INTEGRAL_N_LEVELS.length);
    }, 1300);

    return () => window.clearInterval(timer);
  }, [isPlaying, mode]);

  const switchMode = (nextMode) => {
    setMode(nextMode);
    if (nextMode === 'surface') {
      setIsPlaying(false);
    }
  };

  return (
    <section className="integral-visual" aria-label={t('高维积分动态三维可视化', 'Dynamic 3D visualization of a high-dimensional integral')}>
      <header className="integral-visual-header">
        <div>
          <p className="eyebrow">{t('动态三维直觉', 'Dynamic 3D intuition')}</p>
          <h2>{t('把积分看成“随机高度的平均值”', 'View the integral as an average random height')}</h2>
          <p>{t('先看二维曲面的平均高度，再观察维度增加时随机点如何收缩到极限点。', 'Start with the average height of a 2D surface, then watch random points concentrate around a limit as dimension grows.')}</p>
        </div>
        <div className="integral-mode-toggle" role="group" aria-label={t('可视化视图', 'Visualization view')}>
          <button
            type="button"
            className={mode === 'surface' ? 'active' : ''}
            aria-pressed={mode === 'surface'}
            onClick={() => switchMode('surface')}
          >
            n = 2 {t('曲面', 'surface')}
          </button>
          <button
            type="button"
            className={mode === 'cloud' ? 'active' : ''}
            aria-pressed={mode === 'cloud'}
            onClick={() => switchMode('cloud')}
          >
            n → ∞ {t('云团', 'cloud')}
          </button>
        </div>
      </header>

      <div className="integral-stage">
        <svg viewBox="0 0 760 500" role="img" aria-labelledby="integral-visual-title integral-visual-desc">
          <title id="integral-visual-title">
            {mode === 'surface' ? t('二元积分曲面', 'Two-variable integral surface') : t(`${n} 维随机样本的统计量点云`, `Statistic cloud from ${n}-dimensional random samples`)}
          </title>
          <desc id="integral-visual-desc">
            {mode === 'surface'
              ? t(
                '曲面高度是 x1 平方加 x2 平方除以 x1 加 x2，积分是单位正方形上曲面的平均高度。',
                'The surface height is (x1 squared plus x2 squared) divided by (x1 plus x2). The integral is its average height over the unit square.',
              )
              : t(
                `每个点由 ${n} 个独立均匀随机数生成，维数增加时点云趋近均值二分之一、二阶矩三分之一、比值三分之二。`,
                `Each point comes from ${n} independent uniform samples. As dimension grows, the cloud approaches mean 1/2, second moment 1/3, and ratio 2/3.`,
              )}
          </desc>

          <g className="integral-base-grid" aria-hidden="true">
            {[0.25, 0.5, 0.75, 1].map((tick) => {
              const xFrom = projectIntegralPoint(tick, 0, 0);
              const xTo = projectIntegralPoint(tick, 1, 0);
              const yFrom = projectIntegralPoint(0, tick, 0);
              const yTo = projectIntegralPoint(1, tick, 0);
              return (
                <Fragment key={tick}>
                  <line x1={xFrom.x} y1={xFrom.y} x2={xTo.x} y2={xTo.y} />
                  <line x1={yFrom.x} y1={yFrom.y} x2={yTo.x} y2={yTo.y} />
                </Fragment>
              );
            })}
          </g>

          {mode === 'surface' ? (
            <g className="integral-surface">
              {surface.map((cell) => (
                <polygon
                  key={cell.key}
                  points={cell.points}
                  style={{ '--surface-height': cell.averageHeight }}
                />
              ))}
            </g>
          ) : (
            <g className="integral-cloud">
              <line className="integral-target-guide" x1={target.x} y1={target.y} x2={target.x} y2={origin.y} />
              {cloud.points.map((point) => (
                <circle
                  key={point.key}
                  cx={point.x}
                  cy={point.y}
                  r={n >= 32 ? 2.6 : 3.1}
                  style={{ '--point-ratio': point.ratio }}
                />
              ))}
              <circle className="integral-target-halo" cx={target.x} cy={target.y} r="15" />
              <circle className="integral-target" cx={target.x} cy={target.y} r="5.5" />
              <g className="integral-target-label" transform={`translate(${target.x + 18} ${target.y - 14})`}>
                <rect x="0" y="-23" width="174" height="46" rx="7" />
                <text x="10" y="-4">{t('极限点', 'Limit point')} (1/2, 1/3, 2/3)</text>
                <text x="10" y="14">{t('大数定律下的集中', 'LLN concentration')}</text>
              </g>
            </g>
          )}

          <IntegralAxis
            from={origin}
            to={xEnd}
            label={mode === 'surface' ? 'x₁' : t('样本均值  x̄ₙ', 'sample mean  x̄ₙ')}
            labelX={xEnd.x + 10}
            labelY={xEnd.y + 8}
          />
          <IntegralAxis
            from={origin}
            to={yEnd}
            label={mode === 'surface' ? 'x₂' : t('二阶矩  qₙ', 'second moment  qₙ')}
            labelX={yEnd.x + 8}
            labelY={yEnd.y - 8}
          />
          <IntegralAxis
            from={origin}
            to={zEnd}
            label={mode === 'surface' ? t('高度 f₂', 'height f₂') : t('比值  qₙ / x̄ₙ', 'ratio  qₙ / x̄ₙ')}
            labelX={zEnd.x - 4}
            labelY={zEnd.y - 12}
          />
        </svg>

        <aside className="integral-stage-note">
          {mode === 'surface' ? (
            <>
              <span>{t('二维切入', 'Start in 2D')}</span>
              <strong>{t('积分 = 曲面的平均高度', 'Integral = average surface height')}</strong>
              <p>{t('在单位正方形均匀撒点，每个点的高度是 f₂(x₁,x₂)。所有高度取平均，就是二重积分。', 'Sample uniformly on the unit square. Each point has height f₂(x₁,x₂), and the average of those heights is the double integral.')}</p>
            </>
          ) : (
            <>
              <span>{t('当前维度', 'Current dimension')}</span>
              <strong>n = {n}</strong>
              <p>{t('点云平均高度（固定随机样本）', 'Mean cloud height (fixed random samples)')}</p>
              <b>{cloud.estimate.toFixed(4)}</b>
              <small>{t('目标：', 'Target: ')}2/3 ≈ 0.6667</small>
            </>
          )}
        </aside>
      </div>

      {mode === 'cloud' && (
        <div className="integral-controls">
          <button
            type="button"
            className="integral-play-button"
            onClick={() => setIsPlaying((current) => !current)}
            aria-label={isPlaying ? t('暂停维度动画', 'Pause dimension animation') : t('播放维度动画', 'Play dimension animation')}
          >
            {isPlaying ? t('暂停', 'Pause') : t('播放', 'Play')}
          </button>
          <label>
            <span>{t('维度 n', 'Dimension n')}</span>
            <input
              type="range"
              min="0"
              max={INTEGRAL_N_LEVELS.length - 1}
              step="1"
              value={levelIndex}
              onChange={(event) => {
                setLevelIndex(Number(event.target.value));
                setIsPlaying(false);
              }}
              aria-label={t('选择积分维度', 'Select integral dimension')}
            />
          </label>
          <div className="integral-levels" aria-hidden="true">
            {INTEGRAL_N_LEVELS.map((level, index) => (
              <span className={index === levelIndex ? 'active' : ''} key={level}>{level}</span>
            ))}
          </div>
        </div>
      )}

      <footer className="integral-visual-footer">
        <span><i className="surface-key" /> {t('函数高度 / 样本点', 'function height / sample point')}</span>
        <span><i className="target-key" /> {t('大数定律极限', 'law-of-large-numbers limit')}</span>
        <strong>{mode === 'surface' ? t('先理解“平均高度”', 'Start with average height') : t('n 越大，云团越集中', 'The cloud tightens as n grows')}</strong>
      </footer>
    </section>
  );
}

const MESSAGE_QUEUE_STEPS = [
  {
    phase: 'produce',
    title: 'Producer 构造应用消息',
    detail: 'Envelope 和 payload 已经生成，但 broker 还没有接管责任。此时进程崩溃，消息仍可能丢失。',
    status: 'NEW',
    location: 'producer',
    position: '未分配',
    deliveryCount: 0,
    handle: '无',
    lease: '无',
    activePart: 'message',
  },
  {
    phase: 'store',
    title: 'Broker 持久化并放入 Ready index',
    detail: 'Body bytes 写入持久化 segment，broker 分配位置 184233。Durable ack 之后，API 才能安全返回已接收。',
    status: 'READY',
    location: 'ready',
    position: '184233',
    deliveryCount: 0,
    handle: '无',
    lease: '无',
    activePart: 'broker',
  },
  {
    phase: 'deliver',
    title: 'Worker A 领取消息',
    detail: '业务 body 没变。Broker 生成本次投递使用的 handle，并在 lease 到期前把消息放进 in-flight 集合。',
    status: 'IN_FLIGHT',
    location: 'inflight',
    position: '184233',
    deliveryCount: 1,
    handle: 'rh_A7',
    lease: '30s',
    activePart: 'delivery',
  },
  {
    phase: 'timeout',
    title: 'Worker A 崩溃，没有 ack',
    detail: '数据库可能尚未提交，也可能已经提交。Lease 到期后 broker 只能把消息重新交付，因此 consumer 必须幂等。',
    status: 'RETRY_WAIT',
    location: 'retry',
    position: '184233',
    deliveryCount: 1,
    handle: 'rh_A7 失效',
    lease: '已超时',
    activePart: 'delivery',
  },
  {
    phase: 'requeue',
    title: '消息重新变成 READY',
    detail: '同一条 message body 回到可领取集合。Broker 保留重投信息，下一次领取会得到新的 delivery handle。',
    status: 'READY',
    location: 'ready',
    position: '184233',
    deliveryCount: 1,
    handle: '等待新 handle',
    lease: '无',
    activePart: 'broker',
  },
  {
    phase: 'redeliver',
    title: 'Worker B 收到重投',
    detail: 'Position 和业务 ID 仍相同，handle 变成 rh_B2，delivery count 增加。Worker B 先用 event_id 去重。',
    status: 'IN_FLIGHT',
    location: 'inflight',
    position: '184233',
    deliveryCount: 2,
    handle: 'rh_B2',
    lease: '30s',
    activePart: 'delivery',
  },
  {
    phase: 'ack',
    title: '业务提交成功，再发送 ack',
    detail: 'Broker 收到当前 handle 的确认后删除 queue entry 或推进消费位置。Envelope 和 payload 不需要被修改。',
    status: 'DONE',
    location: 'done',
    position: '184233',
    deliveryCount: 2,
    handle: 'rh_B2 已确认',
    lease: '结束',
    activePart: 'delivery',
  },
];

const MESSAGE_QUEUE_PHASES = [
  ['produce', '构造'],
  ['store', '持久化'],
  ['deliver', '首次投递'],
  ['timeout', '超时'],
  ['requeue', '重新入队'],
  ['redeliver', '再次投递'],
  ['ack', '确认完成'],
];

const MESSAGE_QUEUE_STEPS_EN = [
  {
    title: 'The producer constructs an application message',
    detail: 'The envelope and payload exist, but the broker has not accepted responsibility. A process crash can still lose the message.',
    position: 'unassigned',
    handle: 'none',
    lease: 'none',
  },
  {
    title: 'The broker persists the message and adds it to the ready index',
    detail: 'Body bytes are written to persistent segment 184233. The API can safely report acceptance only after the durable acknowledgment.',
    position: '184233',
    handle: 'none',
    lease: 'none',
  },
  {
    title: 'Worker A claims the message',
    detail: 'The business body is unchanged. The broker creates a delivery handle and keeps the message in flight until its lease expires.',
    position: '184233',
    handle: 'rh_A7',
    lease: '30s',
  },
  {
    title: 'Worker A crashes without acknowledging',
    detail: 'The database may or may not have committed. After the lease expires, the broker can only redeliver, so the consumer must be idempotent.',
    position: '184233',
    handle: 'rh_A7 expired',
    lease: 'timed out',
  },
  {
    title: 'The message becomes ready again',
    detail: 'The same message body returns to the claimable set. The broker retains redelivery metadata, and the next claim receives a new delivery handle.',
    position: '184233',
    handle: 'awaiting a new handle',
    lease: 'none',
  },
  {
    title: 'Worker B receives the redelivery',
    detail: 'The position and business ID stay the same. The handle changes to rh_B2 and the delivery count increases. Worker B deduplicates with event_id first.',
    position: '184233',
    handle: 'rh_B2',
    lease: '30s',
  },
  {
    title: 'Commit the business transaction, then acknowledge',
    detail: 'After receiving an acknowledgment for the current handle, the broker deletes the queue entry or advances the consumer position. The envelope and payload stay unchanged.',
    position: '184233',
    handle: 'rh_B2 acknowledged',
    lease: 'closed',
  },
];

const MESSAGE_QUEUE_PHASES_EN = [
  ['produce', 'Construct'],
  ['store', 'Persist'],
  ['deliver', 'First delivery'],
  ['timeout', 'Timeout'],
  ['requeue', 'Requeue'],
  ['redeliver', 'Redeliver'],
  ['ack', 'Acknowledge'],
];

function MessageQueueVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = isEnglish
    ? MESSAGE_QUEUE_STEPS.map((step, index) => ({ ...step, ...MESSAGE_QUEUE_STEPS_EN[index] }))
    : MESSAGE_QUEUE_STEPS;
  const phases = isEnglish ? MESSAGE_QUEUE_PHASES_EN : MESSAGE_QUEUE_PHASES;
  const step = steps[activeStep];
  const lanes = isEnglish ? [
    ['ready', 'Ready', 'available for a consumer to claim'],
    ['inflight', 'In-flight', 'delivered and awaiting acknowledgment'],
    ['retry', 'Retry wait', 'waiting for lease expiry or backoff'],
    ['done', 'Done', 'the queue entry is complete'],
  ] : [
    ['ready', 'Ready', '可以被 consumer 领取'],
    ['inflight', 'In-flight', '已交付，等待 ack'],
    ['retry', 'Retry wait', '等待 lease / backoff'],
    ['done', 'Done', 'entry 已确认完成'],
  ];

  return (
    <section className="message-queue-visual" aria-label={t('消息队列数据与投递生命周期演示', 'Message data and delivery lifecycle walkthrough')}>
      <header className="message-queue-header">
        <div>
          <p className="eyebrow">{t('消息结构与投递状态', 'Message anatomy + delivery state')}</p>
          <h2>{t('业务内容保持不变，Broker 状态不断变化', 'The business payload stays fixed while broker state changes')}</h2>
          <p>{t('逐步查看同一条 OrderPaid 消息如何从 producer 进入 queue，超时后重投，最后被确认。', 'Follow one OrderPaid message from the producer into the queue, through timeout and redelivery, and finally to acknowledgment.')}</p>
        </div>
        <div className="message-queue-counter">
          {activeStep + 1}<span>/ {MESSAGE_QUEUE_STEPS.length}</span>
        </div>
      </header>

      <div className="message-queue-phases" aria-label={t('消息投递阶段', 'Message delivery stages')}>
        {phases.map(([id, label], index) => (
          <button
            type="button"
            className={`${index === activeStep ? 'active' : ''} ${index < activeStep ? 'complete' : ''}`}
            onClick={() => setActiveStep(index)}
            aria-pressed={index === activeStep}
            key={id}
          >
            <span>{index + 1}</span>
            {label}
          </button>
        ))}
      </div>

      <div className="message-queue-step-copy">
        <span>{step.status}</span>
        <strong>{step.title}</strong>
        <p>{step.detail}</p>
      </div>

      <div className="message-queue-stage">
        <div className={`message-record ${step.activePart === 'message' ? 'active' : ''}`}>
          <div className="message-record-title">
            <span>{t('应用消息', 'Application message')}</span>
            <strong>evt_01J...</strong>
          </div>
          <div className="message-envelope">
            <span>{t('消息头', 'Envelope')}</span>
            <code>event_type</code><strong>order.paid</strong>
            <code>schema_version</code><strong>3</strong>
            <code>aggregate_id</code><strong>order_918</strong>
            <code>traceparent</code><strong>00-a81...</strong>
          </div>
          <div className="message-payload">
            <span>{t('载荷字节', 'Payload bytes')}</span>
            <pre>{`{
  "order_id": "order_918",
  "amount_cents": 2599,
  "currency": "USD"
}`}</pre>
          </div>
        </div>

        <div className="message-queue-arrow" aria-hidden="true">
          <span className={step.location === 'producer' ? '' : 'active'}>→</span>
          <small>{step.location === 'producer' ? t('等待发布', 'publish pending') : t('消息正文不变', 'same body bytes')}</small>
        </div>

        <div className={`broker-record ${step.activePart !== 'message' ? 'active' : ''}`}>
          <div className="broker-record-title">
            <span>{t('Broker 元数据', 'Broker metadata')}</span>
            <strong>{step.status}</strong>
          </div>
          <dl>
            <div><dt>queue</dt><dd>billing.v1</dd></div>
            <div><dt>position</dt><dd>{step.position}</dd></div>
            <div><dt>delivery_count</dt><dd>{step.deliveryCount}</dd></div>
            <div className={step.activePart === 'delivery' ? 'hot' : ''}><dt>handle</dt><dd>{step.handle}</dd></div>
            <div className={step.activePart === 'delivery' ? 'hot' : ''}><dt>lease</dt><dd>{step.lease}</dd></div>
          </dl>
        </div>
      </div>

      <div className="message-queue-lanes" aria-label={t('Broker 中的消息状态集合', 'Message state sets in the broker')}>
        {lanes.map(([id, label, detail]) => (
          <div className={step.location === id ? `active ${id}` : id} key={id}>
            <span><strong>{label}</strong><small>{detail}</small></span>
            <div className="message-queue-slot">
              {step.location === id ? (
                <b>
                  <i />
                  evt_01J...
                  <em>#{step.position}</em>
                </b>
              ) : (
                <small>{t('空', 'empty')}</small>
              )}
            </div>
          </div>
        ))}
      </div>

      {step.location === 'producer' && (
        <div className="message-queue-producer-note">{t('消息还在 producer 内存中，broker 尚未接管', 'The message is still in producer memory; the broker has not accepted responsibility')}</div>
      )}

      <div className="message-queue-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择消息队列生命周期步骤', 'Select a message lifecycle step')}
        />
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          disabled={activeStep === steps.length - 1}
        >
          {t('下一步', 'Next')}
        </button>
      </div>
    </section>
  );
}

const BUSINESS_ALGORITHM_PATHS = {
  cascade: {
    label: '传统级联',
    eyebrow: 'MULTI-STAGE FUNNEL',
    title: '亿级候选，沿延迟预算逐层收窄',
    summary: '前层处理更多候选，使用便宜特征；后层候选变少，才加入实时交叉和列表约束。',
    stages: [
      {
        id: 'recall',
        step: '01',
        title: '多路召回',
        short: 'Recall',
        volume: '10⁸ → 3k',
        latency: '10–30 ms',
        input: '全量 item、query / 用户历史、倒排与向量索引',
        output: '带召回通道和原始分数的数千候选',
        compute: 'BM25、ItemCF、双塔 ANN、热门与关注通道并行取回',
        failure: '正例没进候选，后续再强的排序也救不回来。',
        chapter: '第 6 章 · Query、内容和多路召回',
        noteId: 'BusinessAlgorithm01C Multi-Channel Retrieval.md',
      },
      {
        id: 'filter',
        step: '02',
        title: '合并与过滤',
        short: 'Merge',
        volume: '3k → 1.8k',
        latency: '5–15 ms',
        input: '多路候选、库存、地域、安全、已看记录',
        output: '去重后的合法候选与通道归因',
        compute: '去重、配额、硬规则、轻量特征补齐',
        failure: '过滤过严会形成隐蔽误杀；过滤过松会浪费后级预算。',
        chapter: '第 20 章 · 系统设计',
        noteId: 'BusinessAlgorithm07 System Design.md',
      },
      {
        id: 'prerank',
        step: '03',
        title: '粗排',
        short: 'Pre-rank',
        volume: '1.8k → 300',
        latency: '10–25 ms',
        input: '候选、低成本用户与 item 特征',
        output: '保留给精排的数百候选',
        compute: '蒸馏模型、轻量 DNN / GBDT、分数校准',
        failure: '粗排与精排目标错位时，会提前删掉精排本来会保留的 item。',
        chapter: '第 11 章 · 特征交叉、粗排与个性化',
        noteId: 'BusinessAlgorithm02C Feature Interaction.md',
      },
      {
        id: 'rank',
        step: '04',
        title: '精排',
        short: 'Rank',
        volume: '300 → 80',
        latency: '25–60 ms',
        input: '实时特征、交叉特征和候选集合',
        output: 'CTR、CVR、时长等多目标分数',
        compute: 'Wide & Deep、DeepFM、DCN、多任务学习与分数融合',
        failure: '训练标签、曝光偏差或线上特征错位会直接扭曲最终顺序。',
        chapter: '第 10 章 · 多目标学习与分数融合',
        noteId: 'BusinessAlgorithm02B Multi-Objective Ranking.md',
      },
      {
        id: 'slate',
        step: '05',
        title: '列表决策',
        short: 'Slate',
        volume: '80 → 20',
        latency: '5–20 ms',
        input: '排序结果、规则、探索预算和列表上下文',
        output: '最终展示列表与完整曝光日志',
        compute: 'MMR / DPP、去重、频控、业务规则、bandit 探索',
        failure: '逐 item 最优不等于整页最优；重复、疲劳和规则冲突都在这里暴露。',
        chapter: '第 13 章 · 重排、多样性与规则',
        noteId: 'BusinessAlgorithm03 List Decision.md',
      },
    ],
  },
  generative: {
    label: '端到端生成',
    eyebrow: 'GENERATIVE PATH',
    title: '把检索与排序目标并入一次序列生成',
    summary: '模型可以统一更多阶段，但 SID、约束解码、库存安全规则和反馈闭环仍然存在。',
    stages: [
      {
        id: 'context',
        step: '01',
        title: '统一上下文',
        short: 'Context',
        volume: 'query + history',
        latency: 'online',
        input: 'query、行为序列、场景、用户与 item 表示',
        output: '可供序列模型消费的统一 token / embedding',
        compute: '序列化用户行为，融合搜索意图与上下文',
        failure: '上下文过长、时间信息丢失或训练服务格式不一致都会污染生成。',
        chapter: '第 18 章 · 生成式推荐',
        noteId: 'BusinessAlgorithm05 Generative Recommendation.md',
      },
      {
        id: 'generator',
        step: '02',
        title: '统一生成器',
        short: 'Generate',
        volume: 'one model',
        latency: 'decode budget',
        input: '统一上下文和当前策略',
        output: 'item token、Semantic ID 或整个推荐 slate',
        compute: 'HSTU / OneRec / OneSearch 类序列建模与自回归解码',
        failure: '"端到端"范围因系统而异，不能默认所有在线服务和规则都消失。',
        chapter: '第 18 章 · 生成式推荐',
        noteId: 'BusinessAlgorithm05 Generative Recommendation.md',
      },
      {
        id: 'materialize',
        step: '03',
        title: '标识物化',
        short: 'Materialize',
        volume: 'SID → items',
        latency: 'index lookup',
        input: '生成的 item ID / Semantic ID',
        output: '真实、可展示且可追踪版本的候选',
        compute: 'SID codebook、posting、版本对齐与冲突处理',
        failure: '量化冲突、空 posting 或索引版本错位会让合法 token 找不到真实 item。',
        chapter: '第 17 章 · Semantic ID',
        noteId: 'BusinessAlgorithm04 Generative Algorithms.md',
      },
      {
        id: 'align',
        step: '04',
        title: '偏好对齐',
        short: 'Align',
        volume: 'CE → DPO / RL',
        latency: 'offline train',
        input: '正负偏好、rollout 和下游 reward',
        output: '更符合列表与业务目标的生成策略',
        compute: 'SFT、DPO、GRPO / PPO 与不可微系统指标',
        failure: 'reward 设计不完整会诱发投机；off-policy 数据会放大分布偏移。',
        chapter: '第 18 章 · 偏好与 RL',
        noteId: 'BusinessAlgorithm05 Generative Recommendation.md',
      },
      {
        id: 'guardrail',
        step: '05',
        title: '约束与服务',
        short: 'Serve',
        volume: 'valid top N',
        latency: 'P99 budget',
        input: '生成结果、库存、安全和业务规则',
        output: '最终列表、降级结果和曝光日志',
        compute: '约束解码、过滤、缓存、fallback 与观测',
        failure: '模型统一了目标，不代表确定性约束、容灾和线上验证可以省略。',
        chapter: '第 20 章 · 系统设计',
        noteId: 'BusinessAlgorithm07 System Design.md',
      },
    ],
  },
};

const BUSINESS_ALGORITHM_PATHS_EN = {
  cascade: {
    label: 'Traditional cascade',
    eyebrow: 'MULTI-STAGE FUNNEL',
    title: 'Narrow hundreds of millions of candidates within a fixed latency budget',
    summary: 'Early stages handle more candidates with cheaper features. Real-time crosses and slate constraints appear only after the candidate set is small.',
    stages: [
      {
        id: 'recall',
        step: '01',
        title: 'Multi-channel retrieval',
        short: 'Recall',
        volume: '10⁸ → 3k',
        latency: '10–30 ms',
        input: 'The full item corpus, query or user history, inverted indexes, and vector indexes',
        output: 'A few thousand candidates with channel attribution and raw retrieval scores',
        compute: 'Run BM25, ItemCF, two-tower ANN, trending, and following channels in parallel',
        failure: 'If a relevant item never enters the candidate set, no downstream ranker can recover it.',
        chapter: 'Chapter 6 · Query, content, and multi-channel retrieval',
        noteId: 'BusinessAlgorithm01C Multi-Channel Retrieval.md',
      },
      {
        id: 'filter',
        step: '02',
        title: 'Merge and filter',
        short: 'Merge',
        volume: '3k → 1.8k',
        latency: '5–15 ms',
        input: 'Candidates from every channel, inventory, region, safety, and seen-item history',
        output: 'Deduplicated eligible candidates with channel attribution',
        compute: 'Deduplication, quotas, hard rules, and lightweight feature hydration',
        failure: 'Over-filtering silently drops good items; under-filtering wastes the budget of later stages.',
        chapter: 'Chapter 20 · System design',
        noteId: 'BusinessAlgorithm07 System Design.md',
      },
      {
        id: 'prerank',
        step: '03',
        title: 'Pre-rank',
        short: 'Pre-rank',
        volume: '1.8k → 300',
        latency: '10–25 ms',
        input: 'Candidates plus low-cost user and item features',
        output: 'A few hundred candidates reserved for the full ranker',
        compute: 'Distilled models, lightweight DNNs or GBDTs, and score calibration',
        failure: 'If pre-rank and rank optimize different targets, pre-rank may discard items the ranker would have kept.',
        chapter: 'Chapter 11 · Feature crosses, pre-rank, and personalization',
        noteId: 'BusinessAlgorithm02C Feature Interaction.md',
      },
      {
        id: 'rank',
        step: '04',
        title: 'Rank',
        short: 'Rank',
        volume: '300 → 80',
        latency: '25–60 ms',
        input: 'Real-time features, cross features, and the candidate set',
        output: 'Multi-objective scores such as CTR, CVR, and watch time',
        compute: 'Wide & Deep, DeepFM, DCN, multi-task learning, and score fusion',
        failure: 'Biased labels, exposure bias, or online feature skew directly distort the final order.',
        chapter: 'Chapter 10 · Multi-objective learning and score fusion',
        noteId: 'BusinessAlgorithm02B Multi-Objective Ranking.md',
      },
      {
        id: 'slate',
        step: '05',
        title: 'Slate decision',
        short: 'Slate',
        volume: '80 → 20',
        latency: '5–20 ms',
        input: 'Ranked items, rules, exploration budget, and slate context',
        output: 'The final slate and a complete exposure log',
        compute: 'MMR or DPP, deduplication, frequency caps, business rules, and bandit exploration',
        failure: 'The best items individually may form a poor page. Repetition, fatigue, and rule conflicts surface here.',
        chapter: 'Chapter 13 · Re-ranking, diversity, and rules',
        noteId: 'BusinessAlgorithm03 List Decision.md',
      },
    ],
  },
  generative: {
    label: 'End-to-end generation',
    eyebrow: 'GENERATIVE PATH',
    title: 'Fold retrieval and ranking objectives into sequence generation',
    summary: 'A single model can unify more stages, but SID materialization, constrained decoding, inventory and safety rules, and the feedback loop still remain.',
    stages: [
      {
        id: 'context',
        step: '01',
        title: 'Unified context',
        short: 'Context',
        volume: 'query + history',
        latency: 'online',
        input: 'Query, behavior sequence, scenario, and user and item representations',
        output: 'A unified token or embedding sequence for the sequence model',
        compute: 'Serialize user behavior and combine search intent with current context',
        failure: 'Overlong context, lost temporal information, or train-serve formatting skew can corrupt generation.',
        chapter: 'Chapter 18 · Generative recommendation',
        noteId: 'BusinessAlgorithm05 Generative Recommendation.md',
      },
      {
        id: 'generator',
        step: '02',
        title: 'Unified generator',
        short: 'Generate',
        volume: 'one model',
        latency: 'decode budget',
        input: 'Unified context and the current policy',
        output: 'Item tokens, Semantic IDs, or an entire recommendation slate',
        compute: 'HSTU-, OneRec-, or OneSearch-style sequence modeling and autoregressive decoding',
        failure: '“End to end” has different boundaries across systems; it does not imply that every online service and rule disappears.',
        chapter: 'Chapter 18 · Generative recommendation',
        noteId: 'BusinessAlgorithm05 Generative Recommendation.md',
      },
      {
        id: 'materialize',
        step: '03',
        title: 'ID materialization',
        short: 'Materialize',
        volume: 'SID → items',
        latency: 'index lookup',
        input: 'Generated item IDs or Semantic IDs',
        output: 'Real, displayable, versioned, and traceable candidates',
        compute: 'SID codebooks, postings, version alignment, and collision handling',
        failure: 'Quantization collisions, empty postings, or index-version skew can leave a valid token with no real item.',
        chapter: 'Chapter 17 · Semantic ID',
        noteId: 'BusinessAlgorithm04 Generative Algorithms.md',
      },
      {
        id: 'align',
        step: '04',
        title: 'Preference alignment',
        short: 'Align',
        volume: 'CE → DPO / RL',
        latency: 'offline train',
        input: 'Preference pairs, rollouts, and downstream rewards',
        output: 'A generation policy better aligned with slate and business objectives',
        compute: 'SFT, DPO, GRPO or PPO, and non-differentiable system metrics',
        failure: 'An incomplete reward invites gaming; off-policy data can amplify distribution shift.',
        chapter: 'Chapter 18 · Preference learning and RL',
        noteId: 'BusinessAlgorithm05 Generative Recommendation.md',
      },
      {
        id: 'guardrail',
        step: '05',
        title: 'Constraints and serving',
        short: 'Serve',
        volume: 'valid top N',
        latency: 'P99 budget',
        input: 'Generated results, inventory, safety, and business rules',
        output: 'The final slate, fallback results, and exposure logs',
        compute: 'Constrained decoding, filtering, caching, fallbacks, and observability',
        failure: 'A unified model objective does not remove deterministic constraints, resilience, or online validation.',
        chapter: 'Chapter 20 · System design',
        noteId: 'BusinessAlgorithm07 System Design.md',
      },
    ],
  },
};

function BusinessAlgorithmMap() {
  const { isEnglish, t } = useUiCopy();
  const [mode, setMode] = useState('cascade');
  const [activeStageId, setActiveStageId] = useState('recall');
  const paths = isEnglish ? BUSINESS_ALGORITHM_PATHS_EN : BUSINESS_ALGORITHM_PATHS;
  const path = paths[mode];
  const activeStage = path.stages.find((stage) => stage.id === activeStageId) ?? path.stages[0];

  const selectMode = (nextMode) => {
    setMode(nextMode);
    setActiveStageId(paths[nextMode].stages[0].id);
  };

  return (
    <section
      className="biz-map"
      data-mode={mode}
      data-system-label={t('系统 / 01', 'SYSTEM / 01')}
      aria-label={t('推荐与搜索业务算法系统地图', 'Recommendation and search algorithm system map')}
    >
      <header className="biz-map-header">
        <div className="biz-map-title">
          <p className="eyebrow">{mode === 'cascade' ? t('多阶段漏斗', path.eyebrow) : t('生成式链路', path.eyebrow)}</p>
          <h2>{path.title}</h2>
          <p>{path.summary}</p>
        </div>
        <div className="biz-mode-switch" role="group" aria-label={t('选择系统架构', 'Choose a system architecture')}>
          {Object.entries(paths).map(([key, option]) => (
            <button
              type="button"
              key={key}
              className={mode === key ? 'active' : ''}
              aria-pressed={mode === key}
              onClick={() => selectMode(key)}
            >
              <span>{key === 'cascade' ? '01' : '02'}</span>
              {option.label}
            </button>
          ))}
        </div>
      </header>

      <div className="biz-request-strip">
        <div>
          <small>{t('请求', 'REQUEST')}</small>
          <strong>query · user · context</strong>
        </div>
        <span className="biz-pulse" aria-hidden="true" />
        <p>{mode === 'cascade'
          ? t('候选漏斗在每一层显式收窄', 'The candidate funnel narrows explicitly at every stage')
          : t('统一模型生成，外部系统负责物化与约束', 'One model generates; external systems materialize and enforce constraints')}</p>
        <div>
          <small>{t('响应', 'RESPONSE')}</small>
          <strong>top 20 + trace</strong>
        </div>
      </div>

      <div className="biz-stage-flow" aria-label={`${path.label} ${t('阶段', 'stages')}`}>
        {path.stages.map((stage, index) => (
          <button
            type="button"
            key={stage.id}
            className={`biz-stage ${activeStage.id === stage.id ? 'active' : ''}`}
            onClick={() => setActiveStageId(stage.id)}
            aria-pressed={activeStage.id === stage.id}
            style={{ '--stage-index': index }}
          >
            <span className="biz-stage-number">{stage.step}</span>
            <strong>{stage.title}</strong>
            <small>{stage.short}</small>
            <b>{stage.volume}</b>
          </button>
        ))}
      </div>

      <div className="biz-inspector" aria-live="polite">
        <div className="biz-inspector-lead">
          <span>{activeStage.step} / {activeStage.short}</span>
          <h3>{activeStage.title}</h3>
          <p>{activeStage.compute}</p>
          <a href={`#${encodeURIComponent(activeStage.noteId)}`}>{activeStage.chapter} →</a>
        </div>
        <dl className="biz-io-grid">
          <div>
            <dt>{t('输入', 'INPUT')}</dt>
            <dd>{activeStage.input}</dd>
          </div>
          <div>
            <dt>{t('输出', 'OUTPUT')}</dt>
            <dd>{activeStage.output}</dd>
          </div>
          <div>
            <dt>{t('预算', 'BUDGET')}</dt>
            <dd>{activeStage.latency}</dd>
          </div>
        </dl>
        <div className="biz-failure-card">
          <small>{t('需要警惕的失效点', 'FAILURE TO WATCH')}</small>
          <p>{activeStage.failure}</p>
        </div>
      </div>

      <footer className="biz-feedback-loop">
        <div className="biz-feedback-label">
          <span>↺</span>
          <div><small>{t('共享反馈闭环', 'SHARED FEEDBACK LOOP')}</small><strong>{t('模型之外，系统仍要闭环', 'The system still needs a feedback loop beyond the model')}</strong></div>
        </div>
        <ol>
          <li><span>01</span>{t('曝光与交互日志', 'Exposure and interaction logs')}</li>
          <li><span>02</span>{t('样本与特征', 'Samples and features')}</li>
          <li><span>03</span>{t('训练与评估', 'Training and evaluation')}</li>
          <li><span>04</span>{t('模型 / 索引版本', 'Model and index versions')}</li>
        </ol>
      </footer>
    </section>
  );
}

const OVERVIEW_STAGES = {
  edge: {
    label: '入口层',
    title: '先把流量接稳',
    body: 'Load Balancer 负责健康检查和分流；API Gateway 负责鉴权、限流与路由。这里不放业务重计算。',
    check: '估算峰值 QPS、连接数、请求大小与突发系数。',
  },
  service: {
    label: '计算层',
    title: '无状态服务承载业务规则',
    body: '实例可以水平扩容，也可以随时被替换。长任务交给队列，热点读取交给缓存。',
    check: '根据单实例安全 QPS 计算副本数，并预留 30% 左右余量。',
  },
  data: {
    label: '数据层',
    title: '先明确 source of truth',
    body: '主存储保存事实数据；副本、缓存和物化视图都是可重建的派生状态。',
    check: '估算读写比、数据量、索引大小、复制带宽与恢复目标。',
  },
  async: {
    label: '异步层',
    title: '把慢工作移出请求路径',
    body: 'Queue / Event Log 接管任务后，Worker 可以独立扩缩容、重试和削峰。',
    check: '估算生产速率、消费速率、积压时间与消息保留空间。',
  },
};

const OVERVIEW_STAGES_EN = {
  edge: {
    label: 'Edge layer',
    title: 'Stabilize incoming traffic first',
    body: 'The load balancer handles health checks and traffic distribution. The API gateway handles authentication, rate limits, and routing. Heavy business computation stays out of this layer.',
    check: 'Estimate peak QPS, connection count, request size, and burst factor.',
  },
  service: {
    label: 'Compute layer',
    title: 'Stateless services run business rules',
    body: 'Instances can scale horizontally and be replaced at any time. Move long-running work to a queue and serve hot reads from a cache.',
    check: 'Divide peak traffic by safe per-instance QPS, then keep roughly 30% headroom.',
  },
  data: {
    label: 'Data layer',
    title: 'Identify the source of truth',
    body: 'The primary store holds factual state. Replicas, caches, and materialized views are rebuildable derived state.',
    check: 'Estimate the read/write ratio, data volume, index size, replication bandwidth, and recovery targets.',
  },
  async: {
    label: 'Async layer',
    title: 'Move slow work off the request path',
    body: 'Once a queue or event log accepts the work, workers can scale, retry, and absorb bursts independently.',
    check: 'Estimate production and consumption rates, backlog time, and message retention storage.',
  },
};

function SystemDesignOverviewVisual() {
  const { isEnglish, t } = useUiCopy();
  const [active, setActive] = useState('service');
  const detail = (isEnglish ? OVERVIEW_STAGES_EN : OVERVIEW_STAGES)[active];

  return (
    <section className="arch-visual overview-arch" aria-label={t('系统设计基础架构图', 'System design overview diagram')}>
      <header className="arch-header">
        <div>
          <p className="eyebrow">{t('系统整体架构', 'High-level architecture')}</p>
          <h2>{t('先跑通同步闭环，再按指标加组件', 'Start with a complete synchronous path, then add components for measured needs')}</h2>
          <p>{t('点击节点查看它解决的问题。蓝色是同步请求，橙色是异步工作，绿色是数据访问。', 'Select a node to see what it solves. Blue marks synchronous requests, orange asynchronous work, and green data access.')}</p>
        </div>
        <div className="arch-legend" aria-label={t('连线图例', 'Connection legend')}>
          <span><i className="sync" />{t('同步', 'sync')}</span>
          <span><i className="async" />{t('异步', 'async')}</span>
          <span><i className="data" />{t('数据', 'data')}</span>
        </div>
      </header>

      <div className="overview-board">
        <div className="arch-lane-label">{t('请求链路', 'REQUEST PATH')}</div>
        <div className="arch-flow overview-main-flow">
          <div className="arch-node neutral"><small>01</small><strong>User / Client</strong><span>{t('发起请求', 'send request')}</span></div>
          <span className="arch-connector sync" aria-hidden="true">→</span>
          <button type="button" className={`arch-node edge ${active === 'edge' ? 'active' : ''}`} onClick={() => setActive('edge')}>
            <small>02 · EDGE</small><strong>LB / API Gateway</strong><span>auth · rate limit · routing</span>
          </button>
          <span className="arch-connector sync" aria-hidden="true">→</span>
          <button type="button" className={`arch-node service ${active === 'service' ? 'active' : ''}`} onClick={() => setActive('service')}>
            <small>03 · COMPUTE</small><strong>Stateless Service</strong><span>{t('业务规则与编排', 'business rules and orchestration')}</span>
          </button>
          <span className="arch-connector data" aria-hidden="true">→</span>
          <button type="button" className={`arch-node store ${active === 'data' ? 'active' : ''}`} onClick={() => setActive('data')}>
            <small>04 · SOURCE OF TRUTH</small><strong>Primary Store</strong><span>{t('事实数据与事务边界', 'factual state and transaction boundary')}</span>
          </button>
        </div>

        <div className="arch-lane-label">{t('支撑链路', 'SUPPORTING PATHS')}</div>
        <div className="overview-support-grid">
          <div className="overview-support-card data-card">
            <span className="support-origin">Service</span><span className="support-arrow data">↓ {t('热点读取', 'hot reads')}</span>
            <div className="arch-node compact store"><strong>Cache</strong><span>{t('可丢、可重建、带 TTL', 'disposable · rebuildable · TTL')}</span></div>
          </div>
          <button type="button" className={`overview-support-card async-card ${active === 'async' ? 'active' : ''}`} onClick={() => setActive('async')}>
            <span className="support-origin">Service</span><span className="support-arrow async">↓ {t('入队', 'enqueue')}</span>
            <div className="arch-node compact queue"><strong>Queue / Event Log</strong><span>{t('持久交接 · 缓冲', 'durable handoff · buffer')}</span></div>
            <span className="support-arrow async">↓ {t('消费', 'consume')}</span>
            <div className="arch-node compact worker"><strong>Workers</strong><span>{t('重试 · 批处理 · 扩缩容', 'retry · batch · scale')}</span></div>
          </button>
          <div className="overview-support-card data-card">
            <span className="support-origin">Primary Store</span><span className="support-arrow data">↓ {t('复制 / 分片', 'replicate / shard')}</span>
            <div className="arch-node compact store"><strong>Replica / Shard</strong><span>{t('读扩展与故障恢复', 'read scaling and recovery')}</span></div>
          </div>
        </div>
      </div>

      <aside className="arch-inspector" aria-live="polite">
        <span>{detail.label}</span>
        <div><strong>{detail.title}</strong><p>{detail.body}</p></div>
        <div className="arch-estimate"><small>{t('面试时估算', 'Estimate in the interview')}</small><b>{detail.check}</b></div>
      </aside>
    </section>
  );
}

const PHOTO_PATHS = {
  upload: {
    eyebrow: 'UPLOAD PATH',
    title: '大文件直传，API 只走控制流',
    note: '图片 bytes 不经过业务服务；PostReady 事件驱动处理和 feed 分发。',
  },
  feed: {
    eyebrow: 'READ PATH',
    title: '先取 post_id，再批量补齐内容',
    note: 'Timeline 是可重建索引，metadata 才是事实数据；图片由 CDN 返回。',
  },
};

const PHOTO_PATHS_EN = {
  upload: {
    eyebrow: 'UPLOAD PATH',
    title: 'Upload large files directly; keep bytes off the API path',
    note: 'Image bytes bypass the application service. A PostReady event drives media processing and feed distribution.',
  },
  feed: {
    eyebrow: 'READ PATH',
    title: 'Fetch post IDs first, then hydrate content in batches',
    note: 'The timeline is a rebuildable index; metadata is factual state, and the CDN serves image bytes.',
  },
};

function PhotoSharingArchitectureVisual() {
  const { isEnglish, t } = useUiCopy();
  const [mode, setMode] = useState('upload');
  const copy = (isEnglish ? PHOTO_PATHS_EN : PHOTO_PATHS)[mode];

  return (
    <section className="arch-visual photo-arch" aria-label={t('图片分享系统架构图', 'Photo sharing system architecture')}>
      <header className="arch-header split">
        <div>
          <p className="eyebrow">{t('图片分享系统', 'Photo sharing system')}</p>
          <h2>{copy.title}</h2>
          <p>{copy.note}</p>
        </div>
        <div className="arch-tabs" role="group" aria-label={t('选择图片系统链路', 'Choose a photo-system path')}>
          <button type="button" className={mode === 'upload' ? 'active' : ''} onClick={() => setMode('upload')}>{t('发布图片', 'Publish photo')}</button>
          <button type="button" className={mode === 'feed' ? 'active' : ''} onClick={() => setMode('feed')}>{t('读取 Feed', 'Read feed')}</button>
        </div>
      </header>

      <div className="photo-stage" data-mode={mode}>
        <div className="photo-stage-label">{mode === 'upload' ? t('上传链路', copy.eyebrow) : t('读取链路', copy.eyebrow)}</div>
        {mode === 'upload' ? (
          <>
            <div className="photo-control-row arch-flow">
              <div className="arch-node neutral"><small>CLIENT</small><strong>App</strong><span>create post</span></div>
              <span className="arch-connector sync">→</span>
              <div className="arch-node edge"><small>CONTROL</small><strong>Post API</strong><span>auth + signed URL</span></div>
              <span className="arch-connector data">→</span>
              <div className="arch-node store"><small>STATE</small><strong>Metadata DB</strong><span>PENDING → READY</span></div>
            </div>
            <div className="photo-branch-grid">
              <div className="photo-branch media">
                <span className="branch-kicker">{t('数据面 · 图片字节', 'DATA PLANE · BYTES')}</span>
                <div className="arch-node compact neutral"><strong>App</strong><span>PUT signed URL</span></div>
                <span className="support-arrow data">↓</span>
                <div className="arch-node compact blob"><strong>Object Storage</strong><span>original image</span></div>
                <span className="support-arrow async">↓ object event</span>
                <div className="arch-node compact worker"><strong>Media Workers</strong><span>resize · scan · encode</span></div>
                <span className="support-arrow data">↓</span>
                <div className="arch-node compact blob"><strong>CDN Origins</strong><span>optimized variants</span></div>
              </div>
              <div className="photo-branch events">
                <span className="branch-kicker">{t('事件面 · 标识符', 'EVENT PLANE · IDS')}</span>
                <div className="arch-node compact store"><strong>Outbox</strong><span>PostReady(post_id)</span></div>
                <span className="support-arrow async">↓</span>
                <div className="arch-node compact queue"><strong>Event Log</strong><span>partition by author_id</span></div>
                <span className="support-arrow async">↓</span>
                <div className="arch-node compact worker"><strong>Fan-out Workers</strong><span>push ordinary authors</span></div>
                <span className="support-arrow data">↓</span>
                <div className="arch-node compact store"><strong>Home Timelines</strong><span>bounded post_id list</span></div>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="photo-control-row arch-flow">
              <div className="arch-node neutral"><small>CLIENT</small><strong>App</strong><span>GET /feed</span></div>
              <span className="arch-connector sync">→</span>
              <div className="arch-node edge"><small>EDGE</small><strong>Gateway</strong><span>auth · rate limit</span></div>
              <span className="arch-connector sync">→</span>
              <div className="arch-node service"><small>READ</small><strong>Feed Service</strong><span>merge + paginate</span></div>
            </div>
            <div className="feed-read-grid">
              <div className="read-source"><small>1 · CANDIDATES</small><strong>Timeline Store</strong><span>home list + celebrity outbox</span></div>
              <span className="read-arrow">→</span>
              <div className="read-source"><small>2 · HYDRATE</small><strong>Metadata Cache / DB</strong><span>batch-get posts and authors</span></div>
              <span className="read-arrow">→</span>
              <div className="read-source"><small>3 · MEDIA</small><strong>CDN</strong><span>return image variants</span></div>
            </div>
            <div className="feed-safety-strip">
              <span>{t('读取时校验', 'READ-TIME GUARDS')}</span>
              <b>{t('隐私', 'privacy')}</b><i>·</i><b>{t('屏蔽列表', 'block list')}</b><i>·</i><b>{t('已删除内容', 'deleted posts')}</b><i>·</i><b>{t('排序策略', 'ranking policy')}</b>
            </div>
          </>
        )}
      </div>

      <footer className="arch-footnote"><span><i className="sync" />{t('控制流', 'control')}</span><span><i className="data" />{t('字节 / 读取', 'bytes / reads')}</span><span><i className="async" />{t('事件', 'events')}</span><strong>{t('当前视图：', 'Current view: ')}{mode === 'upload' ? t('写入与派生', 'write and derive') : t('读取与补齐', 'read and hydrate')}</strong></footer>
    </section>
  );
}

const ASYNC_PATTERNS = {
  queue: {
    label: 'Task Queue',
    title: '一条任务，只交给一个 worker',
    description: '同一组 worker 竞争领取任务。扩容 worker 可以提高消费速率，但不会复制业务动作。',
  },
  pubsub: {
    label: 'Pub/Sub',
    title: '一个事件，多份独立处理',
    description: '每个订阅拥有自己的进度和重试。增加订阅者时，上游 producer 不需要改代码。',
  },
  kafka: {
    label: 'Kafka groups',
    title: '系统是实现，group 决定语义',
    description: '同一个 consumer group 内是抢单；不同 group 各读一份，就是发布订阅。',
  },
};

const ASYNC_PATTERNS_EN = {
  queue: {
    label: 'Task Queue',
    title: 'One task goes to exactly one worker',
    description: 'Workers in the same group compete for tasks. Adding workers increases throughput without duplicating the business action.',
  },
  pubsub: {
    label: 'Pub/Sub',
    title: 'One event drives several independent handlers',
    description: 'Each subscription owns its progress and retries. Adding a subscriber does not require changing the upstream producer.',
  },
  kafka: {
    label: 'Kafka groups',
    title: 'Kafka is the system; consumer groups define the semantics',
    description: 'Consumers within one group compete for work. Different groups each receive a copy, which gives publish-subscribe semantics.',
  },
};

function AsyncMessagingArchitectureVisual() {
  const { isEnglish, t } = useUiCopy();
  const [pattern, setPattern] = useState('queue');
  const patterns = isEnglish ? ASYNC_PATTERNS_EN : ASYNC_PATTERNS;
  const copy = patterns[pattern];

  return (
    <section className="arch-visual async-arch" aria-label={t('异步消息模式架构图', 'Asynchronous messaging pattern diagram')}>
      <header className="arch-header split">
        <div>
          <p className="eyebrow">{t('消息消费语义', 'Messaging semantics')}</p>
          <h2>{copy.title}</h2>
          <p>{copy.description}</p>
        </div>
        <div className="arch-tabs" role="group" aria-label={t('选择消息模式', 'Choose a messaging pattern')}>
          {Object.entries(patterns).map(([id, item]) => (
            <button type="button" className={pattern === id ? 'active' : ''} onClick={() => setPattern(id)} key={id}>{item.label}</button>
          ))}
        </div>
      </header>

      <div className={`messaging-pattern pattern-${pattern}`}>
        <div className="message-producer arch-node neutral"><small>PRODUCER</small><strong>Order Service</strong><span>OrderPaid</span></div>
        <span className="pattern-arrow async">→</span>
        <div className="message-broker arch-node queue"><small>{pattern === 'kafka' ? 'KAFKA TOPIC' : pattern === 'queue' ? 'DURABLE QUEUE' : 'TOPIC'}</small><strong>orders.paid.v1</strong><span>key = order_id</span></div>
        <span className="pattern-arrow async">→</span>

        {pattern === 'queue' && (
          <div className="consumer-cluster queue-consumers">
            <span>{t('一个消费者组', 'ONE CONSUMER GROUP')}</span>
            <div><div className="consumer active"><b>Worker A</b><small>{t('处理 evt_42', 'handles evt_42')}</small></div><div className="consumer"><b>Worker B</b><small>{t('等待下一条', 'waits for the next task')}</small></div><div className="consumer"><b>Worker C</b><small>{t('等待下一条', 'waits for the next task')}</small></div></div>
            <p><strong>{t('竞争消费', 'Competing consumers')}</strong> · {t('evt_42 只会被其中一个 worker 领取', 'only one worker claims evt_42')}</p>
          </div>
        )}

        {pattern === 'pubsub' && (
          <div className="consumer-cluster subscription-consumers">
            <span>{t('三个订阅', 'THREE SUBSCRIPTIONS')}</span>
            <div><div className="consumer active"><b>Billing</b><small>sub_billing</small></div><div className="consumer active"><b>CRM</b><small>sub_crm</small></div><div className="consumer active"><b>Analytics</b><small>sub_analytics</small></div></div>
            <p><strong>{t('各自一份', 'One copy each')}</strong> · {t('三个订阅分别保存 offset、重试与 DLQ', 'each subscription keeps its own offset, retries, and DLQ')}</p>
          </div>
        )}

        {pattern === 'kafka' && (
          <div className="consumer-cluster kafka-consumers">
            <span>{t('两个消费者组', 'TWO CONSUMER GROUPS')}</span>
            <div className="kafka-group"><b>group: billing</b><div><div className="consumer active"><small>consumer 1</small></div><div className="consumer"><small>consumer 2</small></div></div><em>{t('组内竞争', 'compete within the group')}</em></div>
            <div className="kafka-group"><b>group: analytics</b><div><div className="consumer active"><small>consumer 1</small></div><div className="consumer"><small>consumer 2</small></div></div><em>{t('另一份事件', 'a separate copy')}</em></div>
          </div>
        )}
      </div>

      <footer className="messaging-rule"><span>{t('记忆规则', 'Rule of thumb')}</span><strong>{t('Queue / PubSub 是消费语义；Kafka、RabbitMQ、SQS 是承载语义的系统。', 'Queue and Pub/Sub describe consumption semantics; Kafka, RabbitMQ, and SQS are systems that implement them.')}</strong></footer>
    </section>
  );
}

function VirtualizationContainerVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('vm');
  const isVm = mode === 'vm';

  return (
    <section className="isolation-visual" aria-label={t('虚拟机与容器隔离边界对比', 'VM and container isolation boundary comparison')}>
      <header className="isolation-header">
        <div>
          <p className="eyebrow">{t('隔离边界', 'Isolation boundary')}</p>
          <h2>{isVm
            ? t('VM：每个 guest 有自己的 kernel', 'VM: each guest has its own kernel')
            : t('Container：多个进程共享 host kernel', 'Container: processes share the host kernel')}</h2>
          <p>{isVm
            ? t('Hypervisor 提供虚拟 CPU、内存与设备。', 'The hypervisor provides virtual CPUs, memory, and devices.')
            : t('Namespace 改变可见范围，cgroup 约束资源使用。', 'Namespaces change what a process can see; cgroups limit its resource use.')}</p>
        </div>
        <div className="isolation-tabs" role="group" aria-label="选择隔离方式">
          <button type="button" className={isVm ? 'active' : ''} onClick={() => setMode('vm')}>Virtual machine</button>
          <button type="button" className={!isVm ? 'active' : ''} onClick={() => setMode('container')}>Container</button>
        </div>
      </header>

      <div className={`isolation-stage ${isVm ? 'vm-mode' : 'container-mode'}`}>
        <div className="isolation-workloads">
          {(isVm ? ['Guest A', 'Guest B'] : ['Container A', 'Container B', 'Container C']).map((label, index) => (
            <div className="isolation-workload" key={label}>
              <span>{label}</span>
              <strong>{index === 0 ? 'API' : index === 1 ? 'Worker' : 'Sidecar'}</strong>
              <small>{t('应用 + 依赖库', 'app + libraries')}</small>
              {isVm ? <b>Guest kernel</b> : <b>rootfs + ns + cgroup</b>}
            </div>
          ))}
        </div>

        <div className={`isolation-boundary ${isVm ? 'vm' : 'container'}`}>
          <span>{isVm ? 'HARDWARE VIRTUALIZATION BOUNDARY' : 'PROCESS ISOLATION BOUNDARY'}</span>
        </div>

        {isVm ? (
          <div className="isolation-platform hypervisor">
            <strong>Hypervisor / VMM</strong>
            <span>vCPU · second-level page tables · virtual devices</span>
          </div>
        ) : (
          <>
            <div className="isolation-platform runtime">
              <strong>Container runtime</strong>
              <span>image · rootfs · network · security policy</span>
            </div>
            <div className="isolation-platform kernel">
              <strong>Shared host kernel</strong>
              <span>scheduler · namespaces · cgroups · syscalls</span>
            </div>
          </>
        )}

        <div className="isolation-hardware"><strong>Physical host</strong><span>CPU · memory · NIC · storage</span></div>
      </div>

      <footer className="isolation-memory">
        <span>{t('记忆', 'Remember')}</span>
        <strong>{isVm
          ? t('隔离一台机器，guest kernel 也被隔开。', 'A VM isolates a machine, including its guest kernel.')
          : t('隔离进程视图和资源，kernel 仍然共享。', 'A container isolates process views and resources but still shares the kernel.')}</strong>
        <small>{isVm
          ? t('边界更强 · 启动更重', 'stronger boundary · heavier startup')
          : t('密度更高 · 共享内核风险', 'higher density · shared-kernel risk')}</small>
      </footer>
    </section>
  );
}

function VtableDispatchVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('dynamic');
  const isDynamic = mode === 'dynamic';

  return (
    <section className="vtable-visual" aria-label={t('虚函数动态绑定与非虚函数静态绑定对比', 'Virtual dynamic dispatch vs non-virtual static dispatch')}>
      <header className="vtable-header">
        <div>
          <p className="eyebrow">{t('同一个指针，同一个对象', 'Same pointer, same object')}</p>
          <h2>{isDynamic
            ? t('speak() 是虚函数：运行时查表', 'speak() is virtual: resolved at runtime')
            : t('speak() 不是虚函数：编译期写死', 'speak() is non-virtual: resolved at compile time')}</h2>
          <p>{t('Base* ptr 实际指向一个 Derived 对象，speak() 最终调用谁只取决于它是不是虚函数。', 'Base* ptr actually points at a Derived object. Which speak() runs depends only on whether it is virtual.')}</p>
        </div>
        <div className="vtable-tabs" role="group" aria-label={t('选择是否为虚函数', 'Choose virtual or non-virtual')}>
          <button type="button" className={isDynamic ? 'active' : ''} onClick={() => setMode('dynamic')}>virtual</button>
          <button type="button" className={!isDynamic ? 'active' : ''} onClick={() => setMode('static')}>non-virtual</button>
        </div>
      </header>

      <div className="vtable-code">
        <code>Base* ptr = new Derived();</code>
        <code>ptr-&gt;speak();</code>
      </div>

      <div className="vtable-flow">
        <div className="vtable-node object">
          <span>{t('内存中的对象', 'Object in memory')}</span>
          <strong>Derived</strong>
          <div className={`vtable-vptr ${isDynamic ? 'active' : 'unused'}`}>
            <b>vptr</b>
            <small>{isDynamic ? t('本次调用会用到', 'used this call') : t('本次调用不会用到', 'unused this call')}</small>
          </div>
          <div className="vtable-field">int age</div>
        </div>

        <div className="vtable-arrow" aria-hidden="true">
          <span className={isDynamic ? 'active' : ''}>{isDynamic ? '⇢' : '⇢'}</span>
        </div>

        <div className={`vtable-node table ${isDynamic ? 'active' : 'skipped'}`}>
          <span>{isDynamic ? t("Derived 的 vtable", "Derived's vtable") : t('vtable（未被查询）', 'vtable (never consulted)')}</span>
          <div className="vtable-slot hit"><code>speak</code><b>Derived::speak</b></div>
          <div className="vtable-slot"><code>eat</code><b>Base::eat</b></div>
        </div>

        <div className="vtable-arrow" aria-hidden="true"><span className="active">→</span></div>

        <div className={`vtable-node result ${isDynamic ? 'correct' : 'surprise'}`}>
          <span>{t('实际执行', 'Actually runs')}</span>
          <strong>{isDynamic ? 'Derived::speak()' : 'Base::speak()'}</strong>
          <small>{isDynamic
            ? t('符合直觉：调用了对象真正类型的版本', "matches intuition: the object's real type wins")
            : t('容易被忽略的坑：编译器只看指针的声明类型', "the easy-to-miss trap: the compiler only looks at the pointer's declared type")}</small>
        </div>
      </div>

      <footer className="vtable-footer">
        <span>{t('记住', 'Remember')}</span>
        <strong>{t('只有虚函数才会经过 vptr 查表；非虚函数在编译期就已经绑定到指针的声明类型上，与指针实际指向的对象类型无关。', "Only virtual functions go through the vptr lookup; non-virtual functions are bound at compile time to the pointer's declared type, regardless of the object it actually points at.")}</strong>
      </footer>
    </section>
  );
}

function FalseSharingVisual() {
  const { t } = useUiCopy();
  const [padded, setPadded] = useState(false);

  return (
    <section className="false-sharing-visual" aria-label={t('伪共享与 cache line 填充对比', 'False sharing vs cache-line padding')}>
      <header className="false-sharing-header">
        <div>
          <p className="eyebrow">{t('两个线程，两个互不相关的变量', 'Two threads, two unrelated variables')}</p>
          <h2>{padded
            ? t('填充后：a、b 各自占一条 cache line', 'Padded: a and b each own a cache line')
            : t('未填充：a、b 挤在同一条 cache line 里', 'Unpadded: a and b share one cache line')}</h2>
          <p>{t('线程 0 只写 a，线程 1 只写 b，逻辑上完全独立。', 'Thread 0 only writes a, thread 1 only writes b — logically independent.')}</p>
        </div>
        <div className="false-sharing-tabs" role="group" aria-label={t('选择是否填充', 'Choose padded or not')}>
          <button type="button" className={!padded ? 'active' : ''} onClick={() => setPadded(false)}>{t('未填充', 'unpadded')}</button>
          <button type="button" className={padded ? 'active' : ''} onClick={() => setPadded(true)}>alignas(64)</button>
        </div>
      </header>

      <div className="false-sharing-cores">
        <div className="false-sharing-core">
          <span>{t('核心 0', 'Core 0')}</span>
          <strong>{t('反复写 a', 'repeatedly writes a')}</strong>
        </div>
        <div className="false-sharing-core">
          <span>{t('核心 1', 'Core 1')}</span>
          <strong>{t('反复写 b', 'repeatedly writes b')}</strong>
        </div>
      </div>

      <div className={`false-sharing-lines ${padded ? 'padded' : 'unpadded'}`}>
        {padded ? (
          <>
            <div className="false-sharing-line a-only">
              <span>{t('cache line #1（64 字节）', 'cache line #1 (64 bytes)')}</span>
              <div className="false-sharing-bytes">
                <b className="byte-a">a</b>
                {Array.from({ length: 7 }).map((_, i) => <i key={i} />)}
              </div>
              <small>{t('只被核心 0 缓存', 'cached only by core 0')}</small>
            </div>
            <div className="false-sharing-line b-only">
              <span>{t('cache line #2（64 字节）', 'cache line #2 (64 bytes)')}</span>
              <div className="false-sharing-bytes">
                <b className="byte-b">b</b>
                {Array.from({ length: 7 }).map((_, i) => <i key={i} />)}
              </div>
              <small>{t('只被核心 1 缓存', 'cached only by core 1')}</small>
            </div>
          </>
        ) : (
          <div className="false-sharing-line shared">
            <span>{t('同一条 cache line（64 字节）', 'the same cache line (64 bytes)')}</span>
            <div className="false-sharing-bytes">
              <b className="byte-a">a</b>
              <b className="byte-b">b</b>
              {Array.from({ length: 6 }).map((_, i) => <i key={i} />)}
            </div>
            <small className="warn">{t('两个核心反复互相 invalidate 对方的缓存副本', "each core keeps invalidating the other's cached copy")}</small>
          </div>
        )}
      </div>

      <footer className={`false-sharing-footer ${padded ? 'good' : 'bad'}`}>
        <span>{t('结果', 'Result')}</span>
        <strong>{padded
          ? t('两个核心各自访问自己的 cache line，没有额外的一致性流量。', "each core hits its own cache line — no extra coherence traffic.")
          : t('MESI 协议不断在两个核心间搬运这条 cache line，性能明显下降。', 'MESI keeps bouncing this cache line between cores, and performance drops sharply.')}</strong>
      </footer>
    </section>
  );
}

function ForkCowVisual() {
  const { t } = useUiCopy();
  const [written, setWritten] = useState(false);

  return (
    <section className="cow-visual" aria-label={t('fork 写时复制机制演示', 'fork copy-on-write mechanism')}>
      <header className="cow-header">
        <div>
          <p className="eyebrow">{t('fork() 之后', 'After fork()')}</p>
          <h2>{written
            ? t('子进程写入触发缺页，内核复制出独立页面', 'A write by the child triggers a page fault and a private copy')
            : t('父子进程共享同一块物理页，都只读', 'Parent and child share one physical page, both read-only')}</h2>
        </div>
        <div className="cow-tabs" role="group" aria-label={t('选择是否已写入', 'Choose before or after the write')}>
          <button type="button" className={!written ? 'active' : ''} onClick={() => setWritten(false)}>{t('刚 fork', 'just forked')}</button>
          <button type="button" className={written ? 'active' : ''} onClick={() => setWritten(true)}>{t('子进程写入后', 'after child writes')}</button>
        </div>
      </header>

      <div className="cow-stage">
        <div className="cow-process">
          <span>{t('父进程页表', "Parent's page table")}</span>
          <strong>{t('堆页 P1 → 只读', 'heap page P1 → read-only')}</strong>
        </div>
        <div className="cow-process">
          <span>{t('子进程页表', "Child's page table")}</span>
          <strong className={written ? 'changed' : ''}>
            {written ? t('堆页 P1 → 可写（新副本）', 'heap page P1 → writable (new copy)') : t('堆页 P1 → 只读', 'heap page P1 → read-only')}
          </strong>
        </div>

        <div className="cow-frames">
          <div className="cow-frame original">
            <span>{t('物理页 A（原始内容）', 'physical page A (original content)')}</span>
            <small>{written ? t('父进程独占', 'owned by the parent now') : t('父子共同指向', 'pointed to by both')}</small>
          </div>
          {written && (
            <div className="cow-frame copy">
              <span>{t('物理页 B（写入触发的副本）', 'physical page B (copy made on write)')}</span>
              <small>{t('子进程独占', 'owned by the child')}</small>
            </div>
          )}
        </div>
      </div>

      <footer className="cow-footer">
        <span>{t('记住', 'Remember')}</span>
        <strong>{t('复制不是在 fork() 那一刻发生的，而是在真正写入的那一刻由缺页异常触发。', 'The copy does not happen at fork() itself — it happens at the moment of the actual write, triggered by a page fault.')}</strong>
      </footer>
    </section>
  );
}

const EPOLL_FD_COUNT = 8;
const EPOLL_ACTIVE_FDS = [2, 5];

function EpollVsSelectVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('epoll');
  const isEpoll = mode === 'epoll';

  return (
    <section className="epoll-visual" aria-label={t('select/poll 线性扫描与 epoll 就绪列表对比', 'select/poll linear scan vs epoll ready list')}>
      <header className="epoll-header">
        <div>
          <p className="eyebrow">{t('8 个已注册的 fd，只有 2 个真正有数据', '8 registered fds, only 2 actually have data')}</p>
          <h2>{isEpoll
            ? t('epoll：内核直接把就绪的 fd 放进列表', 'epoll: the kernel pushes ready fds straight into a list')
            : t('select / poll：每次调用都要挨个问一遍', 'select / poll: every call scans every fd')}</h2>
        </div>
        <div className="epoll-tabs" role="group" aria-label={t('选择多路复用方式', 'Choose the multiplexing mechanism')}>
          <button type="button" className={!isEpoll ? 'active' : ''} onClick={() => setMode('select')}>select / poll</button>
          <button type="button" className={isEpoll ? 'active' : ''} onClick={() => setMode('epoll')}>epoll</button>
        </div>
      </header>

      <div className="epoll-fds">
        {Array.from({ length: EPOLL_FD_COUNT }).map((_, i) => {
          const isActive = EPOLL_ACTIVE_FDS.includes(i);
          const stateClass = !isEpoll ? 'scanned' : isActive ? 'pushed' : '';
          return (
            <div className={`epoll-fd ${isActive ? 'ready' : 'idle'} ${stateClass}`} key={i}>
              <span>fd{i}</span>
              <small>{isActive ? t('有数据', 'has data') : t('无数据', 'no data')}</small>
            </div>
          );
        })}
      </div>

      <div className="epoll-note">
        {isEpoll
          ? t('只有 fd2、fd5 触发回调直接进入就绪列表；其余 6 个 fd 完全不参与这次 epoll_wait。', 'Only fd2 and fd5 fire their callback and land in the ready list; the other 6 fds are never touched by this epoll_wait.')
          : t('内核（或库）必须遍历全部 8 个 fd 才能知道哪些就绪，复杂度是 O(n)，与真正就绪的数量无关。', 'The kernel (or library) must scan all 8 fds to find the ready ones — O(n), regardless of how many are actually ready.')}
      </div>

      <div className="epoll-result">
        <span>{t('epoll_wait / select 返回', 'epoll_wait / select returns')}</span>
        <strong>fd2, fd5</strong>
      </div>
    </section>
  );
}

function SharedPtrCycleVisual() {
  const { t } = useUiCopy();
  const [fixed, setFixed] = useState(false);

  return (
    <section className="sp-cycle-visual" aria-label={t('shared_ptr 循环引用与 weak_ptr 修复对比', 'shared_ptr cycle vs the weak_ptr fix')}>
      <header className="sp-cycle-header">
        <div>
          <p className="eyebrow">{t('A 持有指向 B 的指针，B 也持有指向 A 的指针', 'A holds a pointer to B, and B holds a pointer to A')}</p>
          <h2>{fixed
            ? t('B → A 换成 weak_ptr：外部引用释放后两者都能正确析构', 'B → A becomes weak_ptr: releasing the external reference destroys both correctly')
            : t('两端都用 shared_ptr：外部引用释放后谁都不会被析构', 'Both sides use shared_ptr: releasing the external reference destroys neither')}</h2>
        </div>
        <div className="sp-cycle-tabs" role="group" aria-label={t('选择是否修复循环引用', 'Choose broken or fixed')}>
          <button type="button" className={!fixed ? 'active' : ''} onClick={() => setFixed(false)}>{t('循环引用', 'cycle')}</button>
          <button type="button" className={fixed ? 'active' : ''} onClick={() => setFixed(true)}>{t('打破循环', 'break the cycle')}</button>
        </div>
      </header>

      <div className="sp-cycle-external">
        <span>{t('外部作用域', 'external scope')}</span>
        <b>shared_ptr&lt;A&gt; outer;</b>
      </div>

      <div className="sp-cycle-stage">
        <div className="sp-cycle-node">
          <span>A</span>
          <small>use_count = {fixed ? 1 : 2}</small>
          <em>{t('含 1 个外部 shared_ptr', 'includes 1 external shared_ptr')}</em>
        </div>
        <div className="sp-cycle-links">
          <div className="sp-cycle-arrow forward">
            <span>shared_ptr</span>
            <b>A → B</b>
          </div>
          <div className={`sp-cycle-arrow backward ${fixed ? 'weak' : ''}`}>
            <span>{fixed ? 'weak_ptr' : 'shared_ptr'}</span>
            <b>B → A</b>
          </div>
        </div>
        <div className="sp-cycle-node">
          <span>B</span>
          <small>use_count = 1</small>
          <em>{t('只被 A 持有', 'held only by A')}</em>
        </div>
      </div>

      <div className={`sp-cycle-outcome ${fixed ? 'good' : 'bad'}`}>
        <strong>{t('外部持有 A 的 shared_ptr 释放后：', 'After the external shared_ptr to A is released:')}</strong>
        <p>{fixed
          ? t("A 的 use_count 归零 → A 析构 → A 持有的 shared_ptr<B> 释放 → B 的 use_count 归零 → B 析构。", "A's use_count hits 0 → A destructs → A's shared_ptr<B> is released → B's use_count hits 0 → B destructs.")
          : t('A 的 use_count 变成 1（仍被 B 持有），B 的 use_count 仍是 1（被 A 持有），两者永远等不到 0，内存泄露。', "A's use_count drops to 1 (still held by B), B's use_count stays 1 (held by A) — neither ever reaches 0, and both leak.")}</p>
      </div>
    </section>
  );
}

const RANDOM_WALK_RUIN_PARAMS = {
  symmetric: { p: 0.5, a: 3, b: 5 },
  asymmetric: { p: 0.4, a: 2, b: 3 },
};

function simulateRandomWalkRuin(p, a, b, maxSteps = 500) {
  const path = [0];
  let position = 0;
  for (let step = 0; step < maxSteps; step += 1) {
    position += Math.random() < p ? 1 : -1;
    path.push(position);
    if (position === b || position === -a) break;
  }
  return path;
}

function RandomWalkRuinVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('symmetric');
  const [path, setPath] = useState(() => {
    const { p, a, b } = RANDOM_WALK_RUIN_PARAMS.symmetric;
    return simulateRandomWalkRuin(p, a, b);
  });

  const params = RANDOM_WALK_RUIN_PARAMS[mode];
  const { a, b } = params;

  const resimulate = (nextMode) => {
    const { p, a: nextA, b: nextB } = RANDOM_WALK_RUIN_PARAMS[nextMode ?? mode];
    setPath(simulateRandomWalkRuin(p, nextA, nextB));
  };

  const width = 560;
  const height = 260;
  const marginX = 40;
  const marginY = 20;
  const yRange = a + b;
  const plotWidth = width - marginX * 2;
  const plotHeight = height - marginY * 2;

  const xForStep = (step) => marginX + (path.length <= 1 ? 0 : (step / (path.length - 1)) * plotWidth);
  const yForPos = (pos) => marginY + ((b - pos) / yRange) * plotHeight;

  const points = path.map((pos, step) => `${xForStep(step).toFixed(1)},${yForPos(pos).toFixed(1)}`).join(' ');
  const finalPos = path[path.length - 1];
  const hitUpper = finalPos === b;
  const hitLower = finalPos === -a;
  const steps = path.length - 1;

  return (
    <section className="rw-ruin-visual" aria-label={t('一维随机游走吸收边界模拟', 'One-dimensional random walk with absorbing barriers')}>
      <header className="rw-ruin-header">
        <div>
          <p className="eyebrow">{t('单次路径模拟', 'A single simulated path')}</p>
          <h2>{mode === 'symmetric'
            ? t('对称随机游走：p = 0.5，a = 3，b = 5', 'Symmetric random walk: p = 0.5, a = 3, b = 5')
            : t('不对称随机游走：p = 0.4，a = 2，b = 3（向下概率更大）', 'Asymmetric random walk: p = 0.4, a = 2, b = 3 (biased downward)')}</h2>
        </div>
        <div className="rw-ruin-tabs" role="group" aria-label={t('选择参数', 'Choose parameters')}>
          <button type="button" className={mode === 'symmetric' ? 'active' : ''} onClick={() => { setMode('symmetric'); resimulate('symmetric'); }}>{t('对称', 'symmetric')}</button>
          <button type="button" className={mode === 'asymmetric' ? 'active' : ''} onClick={() => { setMode('asymmetric'); resimulate('asymmetric'); }}>{t('不对称', 'asymmetric')}</button>
        </div>
      </header>

      <svg className="rw-ruin-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t('随机游走路径图', 'Random walk path chart')}>
        <line x1={marginX} y1={yForPos(b)} x2={width - marginX} y2={yForPos(b)} className="rw-ruin-barrier upper" />
        <line x1={marginX} y1={yForPos(-a)} x2={width - marginX} y2={yForPos(-a)} className="rw-ruin-barrier lower" />
        <line x1={marginX} y1={yForPos(0)} x2={width - marginX} y2={yForPos(0)} className="rw-ruin-zero" />
        <text x={width - marginX + 6} y={yForPos(b) + 4} className="rw-ruin-label upper">b={b}</text>
        <text x={width - marginX + 6} y={yForPos(-a) + 4} className="rw-ruin-label lower">-a={-a}</text>
        <text x={width - marginX + 6} y={yForPos(0) + 4} className="rw-ruin-label zero">0</text>
        <polyline points={points} className={`rw-ruin-path ${hitUpper ? 'won' : hitLower ? 'lost' : ''}`} />
      </svg>

      <div className={`rw-ruin-outcome ${hitUpper ? 'good' : hitLower ? 'bad' : ''}`}>
        <span>{t('本次模拟结果', 'This simulation')}</span>
        <strong>
          {t('共 ', 'Total ')}{steps}{t(' 步，', ' steps, ')}
          {hitUpper
            ? t('撞到上边界 b', 'hit the upper barrier b')
            : hitLower
              ? t('撞到下边界 -a', 'hit the lower barrier -a')
              : t('模拟步数上限已到，尚未分出胜负', 'reached the simulation step cap without absorption')}
        </strong>
        <button type="button" onClick={() => resimulate()}>{t('重新模拟', 'Resimulate')}</button>
      </div>

      <p className="rw-ruin-note">{t(
        '这是单次随机实现，会因为随机性而波动；上面推导出的 P(先到 b) 和 E[T] 是对所有可能路径取平均之后的理论值，不是某一次具体路径的结果。',
        'This is a single random realization and will vary from run to run; the P(hit b first) and E[T] derived above are theoretical averages over all possible paths, not the result of any one specific path.',
      )}</p>
    </section>
  );
}

function bmNormalRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function bmNormalCDF(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp(-x * x / 2);
  let p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  if (x > 0) p = 1 - p;
  return p;
}

function bmBSCall(S, K, T, r, sigma) {
  if (T <= 1e-6) return { price: Math.max(0, S - K), delta: S > K ? 1 : 0, gamma: 0 };
  const d1 = (Math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  const delta = bmNormalCDF(d1);
  const gamma = (Math.exp(-0.5 * d1 * d1) / Math.sqrt(2 * Math.PI)) / (S * sigma * Math.sqrt(T));
  const price = S * delta - K * Math.exp(-r * T) * bmNormalCDF(d2);
  return { price, delta, gamma };
}

function BrownianMotionVisual() {
  const { t } = useUiCopy();
  const [steps, setSteps] = useState(500);
  const [seed, setSeed] = useState(1);

  const { path, qv, tv, maxVal, minVal } = useMemo(() => {
    const T = 1.0;
    const dt = T / steps;
    const sqrtDt = Math.sqrt(dt);
    const pts = [{ t: 0, w: 0, qv: 0 }];
    let curW = 0;
    let curQv = 0;
    let curTv = 0;
    let maxW = 0;
    let minW = 0;

    for (let i = 1; i <= steps; i++) {
      const dw = bmNormalRandom() * sqrtDt;
      curW += dw;
      curQv += dw * dw;
      curTv += Math.abs(dw);
      if (curW > maxW) maxW = curW;
      if (curW < minW) minW = curW;
      pts.push({ t: i * dt, w: curW, qv: curQv });
    }

    return { path: pts, qv: curQv, tv: curTv, maxVal: Math.max(maxW, 1.2), minVal: Math.min(minW, -1.2) };
  }, [steps, seed]);

  const width = 580;
  const height = 280;
  const marginX = 45;
  const marginY = 25;
  const plotW = width - marginX * 2;
  const plotH = height - marginY * 2;

  const xForT = (time) => marginX + time * plotW;
  const yForW = (val) => marginY + ((maxVal - val) / (maxVal - minVal)) * plotH;
  const yForZero = yForW(0);

  const pointsW = path.map((p) => `${xForT(p.t).toFixed(1)},${yForW(p.w).toFixed(1)}`).join(' ');
  const pointsQv = path.map((p) => `${xForT(p.t).toFixed(1)},${(marginY + plotH - (p.qv / 1.8) * plotH).toFixed(1)}`).join(' ');

  return (
    <section className="bm-demo-container" aria-label={t('布朗运动轨道与二次变差演示', 'Brownian Motion Path & Quadratic Variation Demo')}>
      <header className="bm-demo-header">
        <div>
          <p className="eyebrow">{t('样本轨道与几何性质', 'Sample Path & Geometric Properties')}</p>
          <h2>{t('布朗运动轨道：处处不可微与二次变差收敛', 'Brownian Motion: Nowhere Differentiable & Quadratic Variation')}</h2>
        </div>
        <div className="bm-demo-controls">
          <label>
            {t('分段步数 N = ', 'Steps N = ')}<strong>{steps}</strong>
            <input type="range" min="100" max="2000" step="100" value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          </label>
          <button type="button" onClick={() => setSeed((s) => s + 1)}>{t('重新生成路径', 'New Path')}</button>
        </div>
      </header>

      <svg className="bm-demo-svg" viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1={marginX} y1={yForZero} x2={width - marginX} y2={yForZero} className="bm-axis-zero" />
        <polyline points={pointsW} className="bm-path-wt" />
        <polyline points={pointsQv} className="bm-path-qv" />
        <text x={marginX + 6} y={marginY + 14} className="bm-legend-w">W_t (Path)</text>
        <text x={marginX + 6} y={height - marginY - 6} className="bm-legend-qv">[W]_t ≈ t (QV)</text>
      </svg>

      <div className="bm-metrics-grid">
        <div className="bm-metric-badge">
          <span>{t('二次变差 [W]_T（理论值 = 1.000）', 'Quadratic Variation [W]_T (Theory = 1.000)')}</span>
          <strong style={{ color: '#a855f7' }}>{qv.toFixed(4)}</strong>
        </div>
        <div className="bm-metric-badge">
          <span>{t('一阶全变差 Σ|ΔW|（当 N→∞ 时发散）', 'Total Variation Σ|ΔW| (Diverges as N→∞)')}</span>
          <strong style={{ color: '#f43f5e' }}>{tv.toFixed(2)}</strong>
        </div>
      </div>
    </section>
  );
}

function TwoDRandomWalkVisual() {
  const { t } = useUiCopy();
  const [steps, setSteps] = useState(2500);
  const [seed, setSeed] = useState(1);

  const { path, maxRadius, returnsToOrigin } = useMemo(() => {
    const dt = 1.0 / 100;
    const sqrtDt = Math.sqrt(dt);
    const pts = [{ x: 0, y: 0 }];
    let curX = 0;
    let curY = 0;
    let maxR = 0;
    let returns = 0;

    for (let i = 1; i <= steps; i++) {
      curX += bmNormalRandom() * sqrtDt;
      curY += bmNormalRandom() * sqrtDt;
      pts.push({ x: curX, y: curY });
      const r = Math.sqrt(curX * curX + curY * curY);
      if (r > maxR) maxR = r;
      if (r < 1.5 && i > 30) returns++;
    }

    return { path: pts, maxRadius: Math.max(maxR, 4), returnsToOrigin: returns };
  }, [steps, seed]);

  const size = 340;
  const center = size / 2;
  const scale = (size * 0.42) / maxRadius;

  const points = path.map((p) => `${(center + p.x * scale).toFixed(1)},${(center - p.y * scale).toFixed(1)}`).join(' ');
  const endPoint = path[path.length - 1];

  return (
    <section className="bm-demo-container" aria-label={t('2D 随机游走与布朗运动极限演示', '2D Random Walk & Brownian Limit Demo')}>
      <header className="bm-demo-header">
        <div>
          <p className="eyebrow">{t('高维拓扑与 Donsker 不变原理', 'High-D Topology & Donsker Limit')}</p>
          <h2>{t('二维布朗运动：平面常返性与原点邻域缠绕', '2D Brownian Motion: Planar Recurrence & Neighborhood Winding')}</h2>
        </div>
        <div className="bm-demo-controls">
          <label>
            {t('步数 K = ', 'Steps K = ')}<strong>{steps}</strong>
            <input type="range" min="500" max="6000" step="500" value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          </label>
          <button type="button" onClick={() => setSeed((s) => s + 1)}>{t('重新模拟游走', 'Resimulate Walk')}</button>
        </div>
      </header>

      <div className="bm-2d-layout">
        <svg className="bm-2d-svg" viewBox={`0 0 ${size} ${size}`} role="img">
          <circle cx={center} cy={center} r={1.5 * scale} className="bm-2d-eps-disk" />
          <line x1={0} y1={center} x2={size} y2={center} className="bm-axis-zero" />
          <line x1={center} y1={0} x2={center} y2={size} className="bm-axis-zero" />
          <polyline points={points} className="bm-2d-path" />
          <circle cx={center} cy={center} r="4" fill="#22c55e" />
          <circle cx={center + endPoint.x * scale} cy={center - endPoint.y * scale} r="4" fill="#f43f5e" />
        </svg>

        <div className="bm-2d-stats">
          <div className="bm-metric-badge">
            <span>{t('原点邻域 B_ε(0) 访问次数', 'Visits to Neighborhood B_ε(0)')}</span>
            <strong style={{ color: '#22c55e' }}>{returnsToOrigin}</strong>
          </div>
          <div className="bm-metric-badge">
            <span>{t('最大游走半径 max ||B_t||', 'Max Radial Distance')}</span>
            <strong style={{ color: '#38bdf8' }}>{maxRadius.toFixed(2)}</strong>
          </div>
          <p className="bm-demo-tip">
            {t('Pólya 定理：1D/2D 游走概率 1 常返；3D 游走瞬变（回原点概率 ≈ 34%）。2D 连续布朗运动单点瞬变（不撞单点），但邻域常返（任意小圆盘必进无限次）。',
               'Pólya Theorem: 1D & 2D walks are recurrent (P=1); 3D walk is transient (P≈34%). 2D continuous Brownian motion is point-transient but neighborhood-recurrent.')}
          </p>
        </div>
      </div>
    </section>
  );
}

function ItoGeometryVisual() {
  const { t } = useUiCopy();
  const [partitions, setPartitions] = useState(15);
  const [seed, setSeed] = useState(1);

  const { path, itoSum, stratSum, exactIto, finalW } = useMemo(() => {
    const T = 1.0;
    const dt = T / partitions;
    const sqrtDt = Math.sqrt(dt);
    const pts = [{ t: 0, w: 0 }];
    let curW = 0;
    let iSum = 0;
    let sSum = 0;

    for (let i = 1; i <= partitions; i++) {
      const dw = bmNormalRandom() * sqrtDt;
      const prevW = curW;
      curW += dw;
      pts.push({ t: i * dt, w: curW });
      iSum += prevW * dw; // Left point
      sSum += 0.5 * (prevW + curW) * dw; // Midpoint
    }

    const exact = 0.5 * curW * curW - 0.5 * T;
    return { path: pts, itoSum: iSum, stratSum: sSum, exactIto: exact, finalW: curW };
  }, [partitions, seed]);

  const width = 580;
  const height = 260;
  const marginX = 45;
  const marginY = 25;
  const plotW = width - marginX * 2;
  const plotH = height - marginY * 2;

  const minW = Math.min(...path.map((p) => p.w), -1.2);
  const maxW = Math.max(...path.map((p) => p.w), 1.2);

  const xForT = (time) => marginX + time * plotW;
  const yForW = (val) => marginY + ((maxW - val) / (maxW - minW)) * plotH;
  const yZero = yForW(0);

  const pointsW = path.map((p) => `${xForT(p.t).toFixed(1)},${yForW(p.w).toFixed(1)}`).join(' ');

  return (
    <section className="bm-demo-container" aria-label={t('伊藤几何与斯特拉托诺维奇积分对比演示', 'Itô Geometry vs. Stratonovich Integral Demo')}>
      <header className="bm-demo-header">
        <div>
          <p className="eyebrow">{t('积分逼近与几何差异', 'Integration Scheme & Geometry')}</p>
          <h2>{t('伊藤积分（左端点/鞅） vs 斯特拉托诺维奇积分（中点/普通微积分）', 'Itô Integral (Left Endpoint) vs. Stratonovich (Midpoint)')}</h2>
        </div>
        <div className="bm-demo-controls">
          <label>
            {t('分割数 N = ', 'Partitions N = ')}<strong>{partitions}</strong>
            <input type="range" min="5" max="50" step="5" value={partitions} onChange={(e) => setPartitions(Number(e.target.value))} />
          </label>
          <button type="button" onClick={() => setSeed((s) => s + 1)}>{t('重新模拟', 'Resimulate')}</button>
        </div>
      </header>

      <svg className="bm-demo-svg" viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1={marginX} y1={yZero} x2={width - marginX} y2={yZero} className="bm-axis-zero" />
        {path.slice(0, -1).map((p, i) => {
          const next = path[i + 1];
          const x0 = xForT(p.t);
          const x1 = xForT(next.t);
          const yLeft = yForW(p.w);
          return (
            <rect
              key={i}
              x={x0}
              y={Math.min(yLeft, yZero)}
              width={x1 - x0}
              height={Math.abs(yLeft - yZero)}
              className="bm-ito-rect"
            />
          );
        })}
        <polyline points={pointsW} className="bm-path-wt" />
      </svg>

      <div className="bm-metrics-grid">
        <div className="bm-metric-badge">
          <span>{t('伊藤黎曼和 Σ W_{t_i} ΔW_i（逼近 1/2 W_T^2 - 1/2 T）', 'Itô Sum Σ W_ti ΔW_i')}</span>
          <strong style={{ color: '#38bdf8' }}>{itoSum.toFixed(4)} (理论 {exactIto.toFixed(4)})</strong>
        </div>
        <div className="bm-metric-badge">
          <span>{t('斯特拉托诺维奇中点和 Σ W̄_i ΔW_i（逼近 1/2 W_T^2）', 'Stratonovich Sum Σ W_mid ΔW_i')}</span>
          <strong style={{ color: '#a855f7' }}>{stratSum.toFixed(4)} (理论 {(0.5 * finalW * finalW).toFixed(4)})</strong>
        </div>
      </div>
    </section>
  );
}

function ReflectionPrincipleVisual() {
  const { t } = useUiCopy();
  const [barrier, setBarrier] = useState(1.5);
  const [seed, setSeed] = useState(1);

  const { path, reflectedPath, hitTime, theoreticalProb } = useMemo(() => {
    const T = 2.0;
    const steps = 400;
    const dt = T / steps;
    const sqrtDt = Math.sqrt(dt);
    const orig = [{ t: 0, w: 0 }];
    const refl = [{ t: 0, w: 0 }];
    let curW = 0;
    let hitT = -1;

    for (let i = 1; i <= steps; i++) {
      const dw = bmNormalRandom() * sqrtDt;
      curW += dw;
      const time = i * dt;
      orig.push({ t: time, w: curW });

      if (hitT === -1 && curW >= barrier) {
        hitT = time;
      }
    }

    for (let i = 0; i <= steps; i++) {
      const p = orig[i];
      if (hitT !== -1 && p.t >= hitT) {
        refl.push({ t: p.t, w: 2 * barrier - p.w });
      } else {
        refl.push(p);
      }
    }

    const prob = 2 * (1 - bmNormalCDF(barrier / Math.sqrt(T)));
    return { path: orig, reflectedPath: refl, hitTime: hitT, theoreticalProb: prob };
  }, [barrier, seed]);

  const width = 580;
  const height = 260;
  const marginX = 45;
  const marginY = 25;
  const plotW = width - marginX * 2;
  const plotH = height - marginY * 2;

  const maxVal = Math.max(barrier * 1.4, ...path.map((p) => p.w), ...reflectedPath.map((p) => p.w), 2.2);
  const minVal = Math.min(-1.5, ...path.map((p) => p.w), ...reflectedPath.map((p) => p.w));

  const xForT = (time) => marginX + (time / 2.0) * plotW;
  const yForW = (val) => marginY + ((maxVal - val) / (maxVal - minVal)) * plotH;

  const pointsOrig = path.map((p) => `${xForT(p.t).toFixed(1)},${yForW(p.w).toFixed(1)}`).join(' ');
  const pointsRefl = reflectedPath.map((p) => `${xForT(p.t).toFixed(1)},${yForW(p.w).toFixed(1)}`).join(' ');
  const yBarrier = yForW(barrier);
  const yZero = yForW(0);

  return (
    <section className="bm-demo-container" aria-label={t('停时与反射原理演示', 'Stopping Time & Reflection Principle Demo')}>
      <header className="bm-demo-header">
        <div>
          <p className="eyebrow">{t('强马尔可夫性与极值分布', 'Strong Markov & Extremum Distribution')}</p>
          <h2>{t('反射原理：首达水平 a 之后的空间镜像对称', 'Reflection Principle: Mirror Symmetry After Hitting Barrier a')}</h2>
        </div>
        <div className="bm-demo-controls">
          <label>
            {t('边界水平 a = ', 'Barrier a = ')}<strong>{barrier.toFixed(2)}</strong>
            <input type="range" min="0.5" max="2.5" step="0.25" value={barrier} onChange={(e) => setBarrier(Number(e.target.value))} />
          </label>
          <button type="button" onClick={() => setSeed((s) => s + 1)}>{t('重新模拟路径', 'New Sample Path')}</button>
        </div>
      </header>

      <svg className="bm-demo-svg" viewBox={`0 0 ${width} ${height}`} role="img">
        <line x1={marginX} y1={yZero} x2={width - marginX} y2={yZero} className="bm-axis-zero" />
        <line x1={marginX} y1={yBarrier} x2={width - marginX} y2={yBarrier} className="bm-barrier-line" />
        <text x={width - marginX - 90} y={yBarrier - 6} className="bm-barrier-text">Barrier a = {barrier.toFixed(2)}</text>
        <polyline points={pointsOrig} className="bm-path-wt" />
        {hitTime !== -1 && <polyline points={pointsRefl} className="bm-path-refl" />}
      </svg>

      <div className="bm-metrics-grid">
        <div className="bm-metric-badge">
          <span>{t('首达停时 τ_a', 'First Hitting Time τ_a')}</span>
          <strong style={{ color: hitTime !== -1 ? '#f59e0b' : '#94a3b8' }}>
            {hitTime !== -1 ? `${hitTime.toFixed(3)}s` : t('未触碰', 'Not Reached')}
          </strong>
        </div>
        <div className="bm-metric-badge">
          <span>{t('理论触碰概率 P(M_T >= a) = 2(1 - Φ(a/√T))', 'Theoretical Hit Probability')}</span>
          <strong style={{ color: '#38bdf8' }}>{(theoreticalProb * 100).toFixed(2)}%</strong>
        </div>
      </div>
    </section>
  );
}

function DeltaHedgingVisual() {
  const { t } = useUiCopy();
  const [volImplied, setVolImplied] = useState(0.20);
  const [volRealized, setVolRealized] = useState(0.35);
  const [seed, setSeed] = useState(1);

  const { pathS, pathPnl, finalPnL, theoGammaPnL } = useMemo(() => {
    const T = 1.0;
    const steps = 100;
    const dt = T / steps;
    const sqrtDt = Math.sqrt(dt);
    const r = 0.03;
    const S0 = 100;
    const K = 100;

    const sArr = [S0];
    const pnlArr = [0];
    let curS = S0;
    const initOpt = bmBSCall(S0, K, T, r, volImplied);
    let cash = initOpt.price - initOpt.delta * S0;
    let curDelta = initOpt.delta;
    let theoPnL = 0;

    for (let i = 1; i <= steps; i++) {
      const time = i * dt;
      const remT = Math.max(0, T - time);
      const dw = bmNormalRandom() * sqrtDt;
      const prevS = curS;
      curS = curS * Math.exp((r - 0.5 * volRealized * volRealized) * dt + volRealized * dw);
      sArr.push(curS);

      const opt = bmBSCall(curS, K, remT, r, volImplied);
      theoPnL += 0.5 * prevS * prevS * opt.gamma * (volRealized * volRealized - volImplied * volImplied) * dt;

      cash = cash * Math.exp(r * dt) - (opt.delta - curDelta) * curS;
      curDelta = opt.delta;
      const pnl = opt.price - curDelta * curS + cash;
      pnlArr.push(pnl);
    }

    return { pathS: sArr, pathPnl: pnlArr, finalPnL: pnlArr[steps], theoGammaPnL: theoPnL };
  }, [volImplied, volRealized, seed]);

  const width = 580;
  const height = 260;
  const marginX = 45;
  const marginY = 25;
  const plotW = width - marginX * 2;
  const plotH = height - marginY * 2;

  const minS = Math.min(...pathS, 80);
  const maxS = Math.max(...pathS, 120);
  const maxAbsPnl = Math.max(...pathPnl.map(Math.abs), 3);

  const xForI = (i) => marginX + (i / 100) * plotW;
  const yForS = (val) => marginY + ((maxS - val) / (maxS - minS)) * (plotH * 0.55);
  const yForPnl = (val) => marginY + plotH * 0.8 - (val / (maxAbsPnl * 1.5)) * (plotH * 0.22);
  const yPnlZero = yForPnl(0);

  const pointsS = pathS.map((s, i) => `${xForI(i).toFixed(1)},${yForS(s).toFixed(1)}`).join(' ');
  const pointsPnl = pathPnl.map((p, i) => `${xForI(i).toFixed(1)},${yForPnl(p).toFixed(1)}`).join(' ');

  return (
    <section className="bm-demo-container" aria-label={t('期权 Delta 对冲与 Gamma 损益演示', 'Option Delta Hedging & Gamma PnL Demo')}>
      <header className="bm-demo-header">
        <div>
          <p className="eyebrow">{t('做市对冲与波动率套利', 'Market Making & Volatility Arbitrage')}</p>
          <h2>{t('动态 Delta 对冲损益：dΠ = 1/2 S² Γ (σ_R² - σ_I²) dt', 'Dynamic Delta Hedging PnL')}</h2>
        </div>
        <div className="bm-demo-controls">
          <label>
            {t('隐含波动率 σ_I = ', 'Implied Vol σ_I = ')}<strong>{(volImplied * 100).toFixed(0)}%</strong>
            <input type="range" min="0.10" max="0.50" step="0.05" value={volImplied} onChange={(e) => setVolImplied(Number(e.target.value))} />
          </label>
          <label>
            {t('已实现波动率 σ_R = ', 'Realized Vol σ_R = ')}<strong>{(volRealized * 100).toFixed(0)}%</strong>
            <input type="range" min="0.10" max="0.60" step="0.05" value={volRealized} onChange={(e) => setVolRealized(Number(e.target.value))} />
          </label>
          <button type="button" onClick={() => setSeed((s) => s + 1)}>{t('模拟市场路径', 'Simulate Market')}</button>
        </div>
      </header>

      <svg className="bm-demo-svg" viewBox={`0 0 ${width} ${height}`} role="img">
        <polyline points={pointsS} className="bm-path-wt" />
        <line x1={marginX} y1={yPnlZero} x2={width - marginX} y2={yPnlZero} className="bm-axis-zero" />
        <polyline points={pointsPnl} className={finalPnL >= 0 ? 'bm-path-pnl-pos' : 'bm-path-pnl-neg'} />
        <text x={marginX + 6} y={marginY + 14} className="bm-legend-w">Stock Price S_t</text>
        <text x={marginX + 6} y={height - marginY - 6} className="bm-legend-qv">Hedging PnL (Long Gamma)</text>
      </svg>

      <div className="bm-metrics-grid">
        <div className="bm-metric-badge">
          <span>{t('累计对冲 PnL（Long Gamma）', 'Cumulative Hedging PnL')}</span>
          <strong style={{ color: finalPnL >= 0 ? '#22c55e' : '#f43f5e' }}>
            {finalPnL >= 0 ? '+$' : '-$'}{Math.abs(finalPnL).toFixed(2)}
          </strong>
        </div>
        <div className="bm-metric-badge">
          <span>{t('理论 Gamma 收益 ∫ 1/2 S² Γ (σ_R² - σ_I²) dt', 'Theoretical Gamma Alpha')}</span>
          <strong style={{ color: theoGammaPnL >= 0 ? '#22c55e' : '#f43f5e' }}>
            {theoGammaPnL >= 0 ? '+$' : '-$'}{Math.abs(theoGammaPnL).toFixed(2)}
          </strong>
        </div>
      </div>
    </section>
  );
}

function GameTheoryVisual() {
  const { t } = useUiCopy();
  const [activeTab, setActiveTab] = useState('pirates');

  // Pirates State
  const [numPirates, setNumPirates] = useState(5);
  const [coins, setCoins] = useState(100);

  // Truel State
  const [pA, setPA] = useState(0.3333);
  const [pB, setPB] = useState(0.6667);
  const [pC, setPC] = useState(1.0);
  const [aStrategy, setAStrategy] = useState('pass');

  // Auctions State
  const [valuation, setValuation] = useState(80);
  const [numBidders, setNumBidders] = useState(2);

  // Computed Pirates Backward Induction
  const pirateTable = useMemo(() => {
    let curAlloc = [coins];
    const history = [{ k: 1, alloc: [coins], votes: [0], isSurvived: true }];

    for (let k = 2; k <= numPirates; k++) {
      const prev = history[k - 2].alloc;
      const neededBribes = Math.ceil(k / 2) - 1;
      const candidates = [];
      for (let i = 0; i < k - 1; i++) {
        candidates.push({ idx: i + 1, prevPayout: prev[i], cost: prev[i] + 1 });
      }
      candidates.sort((a, b) => a.cost - b.cost || a.idx - b.idx);
      const chosen = new Set(candidates.slice(0, neededBribes).map(c => c.idx));

      const newAlloc = new Array(k).fill(0);
      let totalSpent = 0;
      const votes = [0];
      for (let i = 0; i < k - 1; i++) {
        const pId = i + 1;
        if (chosen.has(pId)) {
          newAlloc[pId] = prev[i] + 1;
          totalSpent += newAlloc[pId];
          votes.push(pId);
        } else {
          newAlloc[pId] = 0;
        }
      }
      newAlloc[0] = Math.max(0, coins - totalSpent);
      history.push({ k, alloc: newAlloc, votes, isSurvived: totalSpent <= coins });
    }
    return history;
  }, [numPirates, coins]);

  // Truel Win Rates
  const truelWinRates = useMemo(() => {
    const denomAB = 1 - (1 - pA) * (1 - pB);
    const pAB = denomAB > 0 ? pA / denomAB : 0.5;
    const pBA = 1 - pAB;

    const denomAC = 1 - (1 - pA) * (1 - pC);
    const pAC = denomAC > 0 ? pA / denomAC : 0.5;

    let winA = 0, winB = 0, winC = 0;

    if (aStrategy === 'pass') {
      winA = pB * pAB + (1 - pB) * pC * pAC;
      winB = pB * (1 - pAB);
      winC = (1 - pB) * pC * (1 - pAC);
    } else if (aStrategy === 'shootC') {
      const aKillsC_winA = (1 - pB) * pAB;
      const aKillsC_winB = pB + (1 - pB) * pBA;

      const aMissC_winA = pB * pAB + (1 - pB) * pC * pAC;
      const aMissC_winB = pB * (1 - pAB);
      const aMissC_winC = (1 - pB) * pC * (1 - pAC);

      winA = pA * aKillsC_winA + (1 - pA) * aMissC_winA;
      winB = pA * aKillsC_winB + (1 - pA) * aMissC_winB;
      winC = (1 - pA) * aMissC_winC;
    } else {
      const aKillsB_winA = 0;
      const aKillsB_winC = 1;

      const aMissB_winA = pB * pAB + (1 - pB) * pC * pAC;
      const aMissB_winB = pB * (1 - pAB);
      const aMissB_winC = (1 - pB) * pC * (1 - pAC);

      winA = pA * aKillsB_winA + (1 - pA) * aMissB_winA;
      winB = (1 - pA) * aMissB_winB;
      winC = pA * aKillsB_winC + (1 - pA) * aMissB_winC;
    }

    return { winA, winB, winC };
  }, [pA, pB, pC, aStrategy]);

  // Auction Bidding Values
  const auctionData = useMemo(() => {
    const v = valuation;
    const n = numBidders;
    const bidFPA = ((n - 1) / n) * v;
    const bidSPA = v;
    const bidAllPay = ((n - 1) / n) * Math.pow(v / 100, n) * 100;
    const revSeller = ((n - 1) / (n + 1)) * 100;
    const winnersCurseExp = (v / 100) * (1.5 * (v / 2) - v);

    return { bidFPA, bidSPA, bidAllPay, revSeller, winnersCurseExp };
  }, [valuation, numBidders]);

  return (
    <section className="bm-card" aria-label={t('博弈论与策略性决策演示', 'Game Theory & Strategic Decision Making Demo')}>
      <div className="bm-card-header">
        <div>
          <h3 className="bm-card-title">{t('量化博弈论与经典策略交互模拟器', 'Quant Game Theory & Strategic Decision Simulator')}</h3>
          <p className="bm-card-subtitle">
            {t('动态逆向归纳 (SPE) · 纳什均衡 (NE) · 贝叶斯拍卖与胜者诅咒', 'Dynamic Backward Induction · Nash Equilibrium · Auctions & Winner\'s Curse')}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '0.4rem' }}>
          <button
            type="button"
            className={`bm-btn ${activeTab === 'pirates' ? 'bm-btn-primary' : ''}`}
            onClick={() => setActiveTab('pirates')}
          >
            {t('海盗分金', 'Pirates Gold')}
          </button>
          <button
            type="button"
            className={`bm-btn ${activeTab === 'truel' ? 'bm-btn-primary' : ''}`}
            onClick={() => setActiveTab('truel')}
          >
            {t('三方决斗', 'The Truel')}
          </button>
          <button
            type="button"
            className={`bm-btn ${activeTab === 'auctions' ? 'bm-btn-primary' : ''}`}
            onClick={() => setActiveTab('auctions')}
          >
            {t('拍卖与胜者诅咒', 'Auctions')}
          </button>
        </div>
      </div>

      {activeTab === 'pirates' && (
        <div className="bm-controls-row">
          <div className="bm-control-group">
            <label className="bm-label">{t(`海盗人数 N = ${numPirates}`, `Pirates N = ${numPirates}`)}</label>
            <input
              type="range"
              min="2"
              max="10"
              step="1"
              value={numPirates}
              onChange={(e) => setNumPirates(Number(e.target.value))}
              className="bm-slider"
            />
          </div>
          <div className="bm-control-group">
            <label className="bm-label">{t(`金币总数 M = ${coins}`, `Gold Coins M = ${coins}`)}</label>
            <input
              type="range"
              min="10"
              max="300"
              step="10"
              value={coins}
              onChange={(e) => setCoins(Number(e.target.value))}
              className="bm-slider"
            />
          </div>
        </div>
      )}

      {activeTab === 'truel' && (
        <div className="bm-controls-row">
          <div className="bm-control-group">
            <label className="bm-label">{t(`枪手 A 命中率 p_A: ${(pA * 100).toFixed(1)}%`, `Gunfighter A Accuracy: ${(pA * 100).toFixed(1)}%`)}</label>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={pA}
              onChange={(e) => setPA(Number(e.target.value))}
              className="bm-slider"
            />
          </div>
          <div className="bm-control-group">
            <label className="bm-label">{t(`枪手 B 命中率 p_B: ${(pB * 100).toFixed(1)}%`, `Gunfighter B Accuracy: ${(pB * 100).toFixed(1)}%`)}</label>
            <input
              type="range"
              min="0.3"
              max="0.95"
              step="0.05"
              value={pB}
              onChange={(e) => setPB(Number(e.target.value))}
              className="bm-slider"
            />
          </div>
          <div className="bm-control-group">
            <label className="bm-label">{t('枪手 A 第一枪策略', 'Player A Strategy')}</label>
            <select
              value={aStrategy}
              onChange={(e) => setAStrategy(e.target.value)}
              className="bm-slider"
              style={{ background: '#1e293b', color: '#f8fafc', padding: '0.3rem', borderRadius: '4px' }}
            >
              <option value="pass">{t('故意朝天放枪 (Pass)', 'Shoot into air (Pass)')}</option>
              <option value="shootC">{t('射击神枪手 C (Shoot C)', 'Shoot Gunfighter C')}</option>
              <option value="shootB">{t('射击枪手 B (Shoot B)', 'Shoot Gunfighter B')}</option>
            </select>
          </div>
        </div>
      )}

      {activeTab === 'auctions' && (
        <div className="bm-controls-row">
          <div className="bm-control-group">
            <label className="bm-label">{t(`真实估值 v = $${valuation}`, `True Valuation v = $${valuation}`)}</label>
            <input
              type="range"
              min="10"
              max="100"
              step="5"
              value={valuation}
              onChange={(e) => setValuation(Number(e.target.value))}
              className="bm-slider"
            />
          </div>
          <div className="bm-control-group">
            <label className="bm-label">{t(`竞拍人数 n = ${numBidders}`, `Bidders n = ${numBidders}`)}</label>
            <input
              type="range"
              min="2"
              max="10"
              step="1"
              value={numBidders}
              onChange={(e) => setNumBidders(Number(e.target.value))}
              className="bm-slider"
            />
          </div>
        </div>
      )}

      {/* Visual Renderings */}
      {activeTab === 'pirates' && (
        <div style={{ background: '#0f172a', borderRadius: '8px', padding: '1rem', border: '1px solid #334155' }}>
          <div style={{ fontWeight: 600, color: '#38bdf8', marginBottom: '0.5rem' }}>
            {t(`子博弈逆向归纳 (N = ${numPirates} 人最优分配方案)`, `Subgame Backward Induction (N = ${numPirates} Pirates)`)}
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.85rem', borderCollapse: 'collapse', color: '#cbd5e1' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #475569', textAlign: 'left' }}>
                  <th style={{ padding: '0.4rem' }}>{t('子博弈规模', 'Subgame')}</th>
                  <th style={{ padding: '0.4rem' }}>{t('金币分配 (提议者 → 最弱者)', 'Allocation (Proposer → Weakest)')}</th>
                  <th style={{ padding: '0.4rem' }}>{t('投赞成票的编号', 'Supporting Voters')}</th>
                  <th style={{ padding: '0.4rem' }}>{t('提议者所得', 'Proposer Share')}</th>
                </tr>
              </thead>
              <tbody>
                {pirateTable.map((step) => (
                  <tr key={step.k} style={{ borderBottom: '1px solid #1e293b' }}>
                    <td style={{ padding: '0.4rem', fontWeight: 600, color: step.k === numPirates ? '#f59e0b' : '#94a3b8' }}>
                      {step.k} {t('人局', 'Pirates')}
                    </td>
                    <td style={{ padding: '0.4rem', fontFamily: 'monospace' }}>
                      ({step.alloc.join(', ')})
                    </td>
                    <td style={{ padding: '0.4rem', color: '#22c55e' }}>
                      {step.votes.map(v => `P${numPirates - step.k + 1 + v}`).join(', ')} ({step.votes.length}/{step.k} {t('票通过', 'votes')})
                    </td>
                    <td style={{ padding: '0.4rem', fontWeight: 700, color: '#38bdf8' }}>
                      {step.alloc[0]} {t('枚', 'coins')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'truel' && (
        <div className="bm-metrics-grid" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
          <div className="bm-metric-badge" style={{ borderLeft: '4px solid #38bdf8' }}>
            <span>{t('枪手 A 最终胜率', 'Gunfighter A Win Rate')}</span>
            <strong style={{ color: '#38bdf8', fontSize: '1.2rem' }}>{(truelWinRates.winA * 100).toFixed(2)}%</strong>
          </div>
          <div className="bm-metric-badge" style={{ borderLeft: '4px solid #a855f7' }}>
            <span>{t('枪手 B 最终胜率', 'Gunfighter B Win Rate')}</span>
            <strong style={{ color: '#a855f7', fontSize: '1.2rem' }}>{(truelWinRates.winB * 100).toFixed(2)}%</strong>
          </div>
          <div className="bm-metric-badge" style={{ borderLeft: '4px solid #f43f5e' }}>
            <span>{t('枪手 C (神枪手) 胜率', 'Gunfighter C Win Rate')}</span>
            <strong style={{ color: '#f43f5e', fontSize: '1.2rem' }}>{(truelWinRates.winC * 100).toFixed(2)}%</strong>
          </div>
        </div>
      )}

      {activeTab === 'auctions' && (
        <div className="bm-metrics-grid" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
          <div className="bm-metric-badge">
            <span>{t('一阶密封出价 FPA b*(v)', 'First-Price Bid')}</span>
            <strong style={{ color: '#38bdf8' }}>${auctionData.bidFPA.toFixed(2)}</strong>
          </div>
          <div className="bm-metric-badge">
            <span>{t('二阶密封出价 SPA b*(v)', 'Second-Price Bid')}</span>
            <strong style={{ color: '#22c55e' }}>${auctionData.bidSPA.toFixed(2)}</strong>
          </div>
          <div className="bm-metric-badge">
            <span>{t('全支付拍卖出价 All-Pay', 'All-Pay Bid')}</span>
            <strong style={{ color: '#a855f7' }}>${auctionData.bidAllPay.toFixed(2)}</strong>
          </div>
          <div className="bm-metric-badge">
            <span>{t('胜者诅咒收购期望利润', 'Winner\'s Curse Profit')}</span>
            <strong style={{ color: '#f43f5e' }}>-${Math.abs(auctionData.winnersCurseExp).toFixed(2)}</strong>
          </div>
        </div>
      )}
    </section>
  );
}

const RECORD_EXAMPLE_SPEEDS = [7, 4, 6, 2, 5, 1, 3];


const RECORD_EXAMPLE_STATES = (() => {
  let prefixMinimum = Infinity;
  let leaderIndex = -1;
  let groupNumber = 0;

  return RECORD_EXAMPLE_SPEEDS.map((speed, index) => {
    const previousMinimum = prefixMinimum;
    const isRecord = speed < prefixMinimum;

    if (isRecord) {
      prefixMinimum = speed;
      leaderIndex = index;
      groupNumber += 1;
    }

    return {
      index,
      speed,
      previousMinimum,
      prefixMinimum,
      isRecord,
      leaderIndex,
      groupNumber,
    };
  });
})();

function RecordMinimumVisual() {
  const { isEnglish, t } = useUiCopy();
  const [step, setStep] = useState(3);
  const current = step > 0 ? RECORD_EXAMPLE_STATES[step - 1] : null;
  const processed = RECORD_EXAMPLE_STATES.slice(0, step);
  const groups = processed.reduce((result, walker) => {
    if (walker.isRecord) {
      result.push({
        leaderIndex: walker.index,
        speed: walker.speed,
        members: [walker.index + 1],
      });
    } else {
      result[result.length - 1].members.push(walker.index + 1);
    }
    return result;
  }, []);

  let decision = t(
    '从最前方开始，维护目前见过的最低速度。',
    'Scan from the front and keep the lowest speed seen so far.',
  );
  if (current?.index === 0) {
    decision = t(
      '第 1 位在最前方，自然成为第一支队伍的领队。',
      'Walker 1 starts at the front, so they lead the first group.',
    );
  } else if (current?.isRecord) {
    decision = isEnglish
      ? `v${current.index + 1} = ${current.speed} < ${current.previousMinimum}, so it sets a new prefix minimum and becomes a leader.`
      : `v${current.index + 1} = ${current.speed} < ${current.previousMinimum}，刷新前缀最小值，成为新领队。`;
  } else if (current) {
    decision = isEnglish
      ? `v${current.index + 1} = ${current.speed} > ${current.previousMinimum}, so it eventually catches leader ${current.leaderIndex + 1}.`
      : `v${current.index + 1} = ${current.speed} > ${current.previousMinimum}，最终会追上第 ${current.leaderIndex + 1} 位领队。`;
  }

  return (
    <section className="record-visual" aria-label={t('前缀最小值与最终队伍可视化', 'Prefix minimum and final groups visualization')}>
      <header className="record-header">
        <div>
          <p className="eyebrow">{t('前缀最小值', 'Prefix minimum')}</p>
          <h2>{t('从前往后，只保留新的最低速度', 'Scan left to right and keep only new minimum speeds')}</h2>
          <p>{t('速度样本固定为', 'The speed sample is fixed at')} [7, 4, 6, 2, 5, 1, 3].</p>
        </div>
        <strong className="record-step">{t('位置', 'Position')} {step} / {RECORD_EXAMPLE_STATES.length}</strong>
      </header>

      <div className="record-stage">
        <div className="record-direction" aria-hidden="true">
          <span>{t('前方', 'Front')}</span>
          <b>← {t('行进方向', 'Direction of travel')}</b>
          <span>{t('后方', 'Back')}</span>
        </div>

        <div className="record-walkers">
          {RECORD_EXAMPLE_STATES.map((walker) => {
            const isProcessed = walker.index < step;
            const isCurrent = walker.index === step - 1;
            const stateClass = !isProcessed
              ? 'pending'
              : walker.isRecord
                ? 'leader'
                : 'follower';

            return (
              <div
                className={`record-walker ${stateClass}${isCurrent ? ' current' : ''}`}
                aria-current={isCurrent ? 'step' : undefined}
                key={walker.index}
              >
                <small>{t('位置', 'Position')} {walker.index + 1}</small>
                <strong>v = {walker.speed}</strong>
                <span>
                  {!isProcessed
                    ? t('待检查', 'Pending')
                    : walker.isRecord
                      ? `${t('新领队', 'New leader')} · ${t('组', 'Group')} ${walker.groupNumber}`
                      : `${t('并入位置', 'Joins position')} ${walker.leaderIndex + 1}`}
                </span>
              </div>
            );
          })}
        </div>

        <p className="record-decision" aria-live="polite">{decision}</p>

        <div className="record-groups" aria-label={t('当前形成的队伍', 'Groups formed so far')}>
          {groups.length === 0
            ? <span className="record-empty">{t('尚未开始扫描', 'The scan has not started')}</span>
            : groups.map((group) => (
              <div className="record-group" key={group.leaderIndex}>
                <small>{t('领队', 'Leader')} {group.leaderIndex + 1} · {t('速度', 'speed')} {group.speed}</small>
                <strong>[{group.members.join(', ')}]</strong>
              </div>
            ))}
        </div>
      </div>

      <footer className="record-footer">
        <div>
          <span>{t('这组样本', 'This sample')}</span>
          <strong>{groups.length} {t('支队伍', groups.length === 1 ? 'group' : 'groups')}</strong>
        </div>
        <code>{t('P(位置 i 刷新最低值)', 'P(position i sets a new minimum)')} = 1 / i</code>
        <div>
          <span>{t('随机期望', 'Expected value')}</span>
          <strong>H₇ ≈ 2.593</strong>
        </div>
      </footer>

      <div className="record-controls">
        <button
          type="button"
          onClick={() => setStep((value) => Math.max(0, value - 1))}
          disabled={step === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <button
          type="button"
          className="primary"
          onClick={() => setStep((value) => Math.min(RECORD_EXAMPLE_STATES.length, value + 1))}
          disabled={step === RECORD_EXAMPLE_STATES.length}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const SORT_RACE_EXAMPLE = [5, 2, 8, 1, 6, 3];
const SORT_RACE_MAX = Math.max(...SORT_RACE_EXAMPLE);

function rangeInclusive(start, end) {
  if (start > end) {
    return [];
  }

  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}

function sortedIndicesFromSet(indices) {
  return Array.from(indices).sort((left, right) => left - right);
}

function isIndexInRange(range, index) {
  return Boolean(range) && index >= range[0] && index <= range[1];
}

function formatSortRaceRange(range) {
  return range ? `[${range[0]}, ${range[1]}]` : '—';
}

function formatSortRaceIndex(index) {
  return index === null || index === undefined ? '—' : index;
}

function snapshotSortRaceStep(step) {
  return {
    ...step,
    array: [...step.array],
    finalIndices: [...(step.finalIndices ?? [])],
    sortedPrefixIndices: [...(step.sortedPrefixIndices ?? [])],
    compareIndices: [...(step.compareIndices ?? [])],
    moveIndices: [...(step.moveIndices ?? [])],
    activeRange: step.activeRange ? [...step.activeRange] : null,
    leftRange: step.leftRange ? [...step.leftRange] : null,
    rightRange: step.rightRange ? [...step.rightRange] : null,
    sortedRange: step.sortedRange ? [...step.sortedRange] : null,
    mergePointers: step.mergePointers ? { ...step.mergePointers } : null,
    quickPointers: step.quickPointers ? { ...step.quickPointers } : null,
    heapPointers: step.heapPointers ? { ...step.heapPointers } : null,
  };
}

function buildInsertionSortRaceSteps() {
  const array = [...SORT_RACE_EXAMPLE];
  const steps = [
    snapshotSortRaceStep({
      kind: 'start',
      activeLine: 'outer',
      array,
      currentPass: 0,
      keyValue: null,
      keyIndex: null,
      insertIndex: null,
      probeIndex: null,
      finalIndices: [],
      sortedPrefixIndices: [0],
      compareIndices: [],
      moveIndices: [],
      activeRange: [0, 0],
    }),
  ];

  for (let i = 1; i < array.length; i += 1) {
    const key = array[i];
    let j = i - 1;

    steps.push(snapshotSortRaceStep({
      kind: 'pick',
      activeLine: 'save',
      array,
      currentPass: i,
      keyValue: key,
      keyIndex: i,
      insertIndex: i,
      probeIndex: j,
      finalIndices: [],
      sortedPrefixIndices: rangeInclusive(0, i - 1),
      compareIndices: [],
      moveIndices: [i],
      activeRange: [0, i],
    }));

    while (j >= 0) {
      steps.push(snapshotSortRaceStep({
        kind: 'compare',
        activeLine: 'shift',
        array,
        currentPass: i,
        keyValue: key,
        keyIndex: j + 1,
        insertIndex: j + 1,
        probeIndex: j,
        finalIndices: [],
        sortedPrefixIndices: rangeInclusive(0, i - 1),
        compareIndices: [j, j + 1],
        moveIndices: [],
        activeRange: [0, i],
      }));

      if (array[j] <= key) {
        break;
      }

      array[j + 1] = array[j];
      steps.push(snapshotSortRaceStep({
        kind: 'shift',
        activeLine: 'shift',
        array,
        currentPass: i,
        keyValue: key,
        keyIndex: j,
        insertIndex: j,
        probeIndex: j,
        finalIndices: [],
        sortedPrefixIndices: [],
        compareIndices: [j, j + 1],
        moveIndices: [j, j + 1],
        activeRange: [0, i],
      }));

      j -= 1;
    }

    array[j + 1] = key;
    steps.push(snapshotSortRaceStep({
      kind: 'insert',
      activeLine: 'insert',
      array,
      currentPass: i,
      keyValue: key,
      keyIndex: null,
      insertIndex: j + 1,
      probeIndex: null,
      finalIndices: [],
      sortedPrefixIndices: rangeInclusive(0, i),
      compareIndices: [],
      moveIndices: [j + 1],
      activeRange: [0, i],
    }));
  }

  steps.push(snapshotSortRaceStep({
    kind: 'finish',
    activeLine: 'finish',
    array,
    currentPass: array.length - 1,
    keyValue: null,
    keyIndex: null,
    insertIndex: null,
    probeIndex: null,
    finalIndices: rangeInclusive(0, array.length - 1),
    sortedPrefixIndices: rangeInclusive(0, array.length - 1),
    compareIndices: [],
    moveIndices: [],
    activeRange: [0, array.length - 1],
  }));

  return steps;
}

function buildSelectionSortRaceSteps() {
  const array = [...SORT_RACE_EXAMPLE];
  const steps = [
    snapshotSortRaceStep({
      kind: 'start',
      activeLine: 'outer',
      array,
      passIndex: 0,
      minIndex: null,
      scanIndex: null,
      didSwap: false,
      finalIndices: [],
      compareIndices: [],
      moveIndices: [],
      activeRange: [0, array.length - 1],
    }),
  ];
  const finalIndices = [];

  for (let i = 0; i < array.length - 1; i += 1) {
    let minIndex = i;

    steps.push(snapshotSortRaceStep({
      kind: 'pick',
      activeLine: 'initMin',
      array,
      passIndex: i,
      minIndex,
      scanIndex: null,
      didSwap: false,
      finalIndices: [...finalIndices],
      compareIndices: [],
      moveIndices: [i],
      activeRange: [i, array.length - 1],
    }));

    for (let j = i + 1; j < array.length; j += 1) {
      steps.push(snapshotSortRaceStep({
        kind: 'compare',
        activeLine: 'scan',
        array,
        passIndex: i,
        minIndex,
        scanIndex: j,
        didSwap: false,
        finalIndices: [...finalIndices],
        compareIndices: [minIndex, j],
        moveIndices: [],
        activeRange: [i, array.length - 1],
      }));

      if (array[j] < array[minIndex]) {
        minIndex = j;
        steps.push(snapshotSortRaceStep({
          kind: 'updateMin',
          activeLine: 'updateMin',
          array,
          passIndex: i,
          minIndex,
          scanIndex: j,
          didSwap: false,
          finalIndices: [...finalIndices],
          compareIndices: [i, j],
          moveIndices: [j],
          activeRange: [i, array.length - 1],
        }));
      }
    }

    const didSwap = minIndex !== i;
    if (didSwap) {
      [array[i], array[minIndex]] = [array[minIndex], array[i]];
    }
    finalIndices.push(i);

    steps.push(snapshotSortRaceStep({
      kind: 'place',
      activeLine: 'swap',
      array,
      passIndex: i,
      minIndex,
      scanIndex: null,
      didSwap,
      finalIndices: [...finalIndices],
      compareIndices: didSwap ? [i, minIndex] : [i],
      moveIndices: didSwap ? [i, minIndex] : [i],
      activeRange: [i, array.length - 1],
    }));
  }

  finalIndices.push(array.length - 1);
  steps.push(snapshotSortRaceStep({
    kind: 'finish',
    activeLine: 'finish',
    array,
    passIndex: array.length - 1,
    minIndex: null,
    scanIndex: null,
    didSwap: false,
    finalIndices: [...finalIndices],
    compareIndices: [],
    moveIndices: [],
    activeRange: [0, array.length - 1],
  }));

  return steps;
}

function buildBubbleSortRaceSteps() {
  const array = [...SORT_RACE_EXAMPLE];
  const steps = [
    snapshotSortRaceStep({
      kind: 'start',
      activeLine: 'outer',
      array,
      passEnd: array.length - 1,
      passNumber: 0,
      finalIndices: [],
      compareIndices: [],
      moveIndices: [],
      activeRange: [0, array.length - 1],
    }),
  ];

  let finalIndices = [];
  let passNumber = 1;

  for (let end = array.length - 1; end > 0; end -= 1, passNumber += 1) {
    let swapped = false;

    steps.push(snapshotSortRaceStep({
      kind: 'passStart',
      activeLine: 'reset',
      array,
      passEnd: end,
      passNumber,
      finalIndices: [...finalIndices],
      compareIndices: [],
      moveIndices: [],
      activeRange: [0, end],
    }));

    for (let i = 0; i < end; i += 1) {
      steps.push(snapshotSortRaceStep({
        kind: 'compare',
        activeLine: 'scan',
        array,
        passEnd: end,
        passNumber,
        finalIndices: [...finalIndices],
        compareIndices: [i, i + 1],
        moveIndices: [],
        activeRange: [0, end],
      }));

      if (array[i] > array[i + 1]) {
        [array[i], array[i + 1]] = [array[i + 1], array[i]];
        swapped = true;

        steps.push(snapshotSortRaceStep({
          kind: 'swap',
          activeLine: 'swap',
          array,
          passEnd: end,
          passNumber,
          finalIndices: [...finalIndices],
          compareIndices: [i, i + 1],
          moveIndices: [i, i + 1],
          activeRange: [0, end],
        }));
      }
    }

    if (!swapped) {
      finalIndices = rangeInclusive(0, array.length - 1);
      steps.push(snapshotSortRaceStep({
        kind: 'earlyStop',
        activeLine: 'check',
        array,
        passEnd: end,
        passNumber,
        finalIndices: [...finalIndices],
        compareIndices: [],
        moveIndices: [],
        activeRange: [0, end],
      }));
      break;
    }

    finalIndices = rangeInclusive(end, array.length - 1);
    steps.push(snapshotSortRaceStep({
      kind: 'passDone',
      activeLine: 'check',
      array,
      passEnd: end,
      passNumber,
      finalIndices: [...finalIndices],
      compareIndices: [],
      moveIndices: [end],
      activeRange: [0, end],
    }));
  }

  steps.push(snapshotSortRaceStep({
    kind: 'finish',
    activeLine: 'finish',
    array,
    passEnd: 0,
    passNumber,
    finalIndices: rangeInclusive(0, array.length - 1),
    compareIndices: [],
    moveIndices: [],
    activeRange: [0, array.length - 1],
  }));

  return steps;
}

const SIMPLE_SORT_RACE_STEPS = {
  insertion: buildInsertionSortRaceSteps(),
  selection: buildSelectionSortRaceSteps(),
  bubble: buildBubbleSortRaceSteps(),
};

const SIMPLE_SORT_RACE_CODE_LINES = {
  insertion: [
    { id: 'outer', code: ['for i in range(1, len(arr)):', '    key = arr[i]; j = i - 1'] },
    { id: 'save', code: ['key = arr[i]', 'j = i - 1'] },
    { id: 'shift', code: ['while j >= 0 and arr[j] > key:', '    arr[j + 1] = arr[j]; j -= 1'] },
    { id: 'insert', code: ['arr[j + 1] = key'] },
    { id: 'finish', code: ['return arr'] },
  ],
  selection: [
    { id: 'outer', code: ['for i in range(len(arr) - 1):'] },
    { id: 'initMin', code: ['min_idx = i'] },
    { id: 'scan', code: ['for j in range(i + 1, len(arr)):', '    if arr[j] < arr[min_idx]: ...'] },
    { id: 'updateMin', code: ['if arr[j] < arr[min_idx]:', '    min_idx = j'] },
    { id: 'swap', code: ['if min_idx != i:', '    arr[i], arr[min_idx] = arr[min_idx], arr[i]'] },
    { id: 'finish', code: ['return arr'] },
  ],
  bubble: [
    { id: 'outer', code: ['for end in range(len(arr) - 1, 0, -1):'] },
    { id: 'reset', code: ['swapped = False'] },
    { id: 'scan', code: ['for i in range(end):', '    if arr[i] > arr[i + 1]: ...'] },
    { id: 'swap', code: ['arr[i], arr[i + 1] = arr[i + 1], arr[i]', 'swapped = True'] },
    { id: 'check', code: ['if not swapped:', '    break'] },
    { id: 'finish', code: ['return arr'] },
  ],
};

function describeSimpleSortStep(mode, step, t) {
  if (mode === 'insertion') {
    if (step.kind === 'start') {
      return {
        activeLineLabel: t('初始化外层循环', 'Initialize the outer loop'),
        title: t('开始：把长度 1 的前缀视为已排序', 'Start: treat the length-1 prefix as sorted'),
        detail: t(
          'Insertion sort 只维护“左侧前缀有序”。它不会提前保证任何位置已经是全局最终位置。',
          'Insertion sort only maintains a sorted prefix on the left. It does not guarantee any globally final position early.',
        ),
        cards: [
          { label: 'i', value: '—' },
          { label: 'key', value: '—' },
          { label: t('不变式', 'Invariant'), value: t('前缀 [0, 0] 已排序', 'prefix [0, 0] is sorted') },
        ],
      };
    }

    if (step.kind === 'pick') {
      return {
        activeLineLabel: t('取出 key', 'Save key'),
        title: t(
          `第 ${step.currentPass} 轮：取出 key = ${step.keyValue}`,
          `Pass ${step.currentPass}: take key = ${step.keyValue}`,
        ),
        detail: t(
          `进入本轮前，前缀 [0, ${step.currentPass - 1}] 已排序。接下来只需要为 key 找到插入位置。`,
          `Before this pass, prefix [0, ${step.currentPass - 1}] is sorted. The only task now is to find the insertion position for the key.`,
        ),
        cards: [
          { label: 'i', value: step.currentPass },
          { label: 'key', value: step.keyValue },
          { label: t('不变式', 'Invariant'), value: t(`前缀 [0, ${step.currentPass - 1}] 已排序`, `prefix [0, ${step.currentPass - 1}] is sorted`) },
        ],
      };
    }

    if (step.kind === 'compare') {
      const probeValue = step.array[step.probeIndex];
      const shouldShift = probeValue > step.keyValue;
      return {
        activeLineLabel: t('比较并判断是否右移', 'Compare and decide whether to shift'),
        title: t(
          `比较 arr[${step.probeIndex}] = ${probeValue} 与 key = ${step.keyValue}`,
          `Compare arr[${step.probeIndex}] = ${probeValue} with key = ${step.keyValue}`,
        ),
        detail: shouldShift
          ? t('当前值更大，需要继续右移，为 key 腾出位置。', 'The current value is larger, so it must shift right to open a slot for the key.')
          : t('当前值不大于 key，可以停止右移。', 'The current value is not larger than the key, so the shifts can stop.'),
        cards: [
          { label: 'i', value: step.currentPass },
          { label: 'key', value: step.keyValue },
          { label: t('待比较下标', 'Probe index'), value: step.probeIndex },
        ],
      };
    }

    if (step.kind === 'shift') {
      return {
        activeLineLabel: t('右移较大元素', 'Shift the larger value to the right'),
        title: t(
          `右移 arr[${step.probeIndex}]，空位移动到 ${step.insertIndex}`,
          `Shift arr[${step.probeIndex}] to the right; the gap moves to ${step.insertIndex}`,
        ),
        detail: t(
          'Insertion sort 的代价主要来自位移，不是交换。对近乎有序的数据，这种位移通常很少。',
          'Insertion sort pays mostly in shifts, not swaps. On nearly sorted input, those shifts are often rare.',
        ),
        cards: [
          { label: 'i', value: step.currentPass },
          { label: 'key', value: step.keyValue },
          { label: t('当前空位', 'Current gap'), value: step.insertIndex },
        ],
      };
    }

    if (step.kind === 'insert') {
      return {
        activeLineLabel: t('写回 key', 'Write the key back'),
        title: t(
          `把 key = ${step.keyValue} 放到下标 ${step.insertIndex}`,
          `Write key = ${step.keyValue} into index ${step.insertIndex}`,
        ),
        detail: t(
          `这一轮结束后，前缀 [0, ${step.currentPass}] 已重新恢复为有序。`,
          `After this pass, prefix [0, ${step.currentPass}] becomes sorted again.`,
        ),
        cards: [
          { label: 'i', value: step.currentPass },
          { label: 'key', value: step.keyValue },
          { label: t('不变式', 'Invariant'), value: t(`前缀 [0, ${step.currentPass}] 已排序`, `prefix [0, ${step.currentPass}] is sorted`) },
        ],
      };
    }

    return {
      activeLineLabel: t('结束', 'Finish'),
      title: t('结束：整个数组有序', 'Finish: the whole array is sorted'),
      detail: t(
        'Insertion sort 直到最后一轮才把“局部有序前缀”扩展成“整个数组有序”。',
        'Insertion sort turns the locally sorted prefix into a fully sorted array only at the last pass.',
      ),
      cards: [
        { label: 'i', value: 'done' },
        { label: 'key', value: '—' },
        { label: t('最终状态', 'Final state'), value: t('所有位置已确定', 'all positions are sorted') },
      ],
    };
  }

  if (mode === 'selection') {
    if (step.kind === 'start') {
      return {
        activeLineLabel: t('初始化外层循环', 'Initialize the outer loop'),
        title: t('开始：每一轮只确定一个最小值的位置', 'Start: each pass fixes exactly one minimum'),
        detail: t(
          'Selection sort 的不变式是“前缀已经是最终位置”。它会完整扫描后缀，再做至多一次交换。',
          'The invariant in selection sort is that the prefix already contains final positions. It scans the whole suffix, then performs at most one swap.',
        ),
        cards: [
          { label: 'i', value: '—' },
          { label: 'min_idx', value: '—' },
          { label: t('不变式', 'Invariant'), value: t('当前没有最终前缀', 'no final prefix yet') },
        ],
      };
    }

    if (step.kind === 'pick') {
      return {
        activeLineLabel: t('初始化最小值下标', 'Initialize the minimum index'),
        title: t(`第 ${step.passIndex} 轮：先令 min_idx = ${step.minIndex}`, `Pass ${step.passIndex}: start with min_idx = ${step.minIndex}`),
        detail: t(
          '接下来会线性扫描未排序后缀，看看是否存在更小的候选值。',
          'The next step scans the unsorted suffix linearly to see whether a smaller candidate exists.',
        ),
        cards: [
          { label: 'i', value: step.passIndex },
          { label: 'min_idx', value: step.minIndex },
          { label: t('已确定前缀', 'Final prefix'), value: step.passIndex === 0 ? '—' : `[0, ${step.passIndex - 1}]` },
        ],
      };
    }

    if (step.kind === 'compare') {
      return {
        activeLineLabel: t('扫描未排序后缀', 'Scan the unsorted suffix'),
        title: t(
          `比较 arr[${step.scanIndex}] = ${step.array[step.scanIndex]} 与 arr[${step.minIndex}] = ${step.array[step.minIndex]}`,
          `Compare arr[${step.scanIndex}] = ${step.array[step.scanIndex]} with arr[${step.minIndex}] = ${step.array[step.minIndex]}`,
        ),
        detail: t(
          'Selection sort 不立即交换。它先扫完整个后缀，再在本轮结束时统一放置最小值。',
          'Selection sort does not swap immediately. It finishes scanning the suffix first, then places the minimum once at the end of the pass.',
        ),
        cards: [
          { label: 'i', value: step.passIndex },
          { label: 'min_idx', value: step.minIndex },
          { label: t('扫描位置', 'Scan index'), value: step.scanIndex },
        ],
      };
    }

    if (step.kind === 'updateMin') {
      return {
        activeLineLabel: t('更新最小值下标', 'Update the minimum index'),
        title: t(
          `更新：min_idx = ${step.minIndex}`,
          `Update: min_idx = ${step.minIndex}`,
        ),
        detail: t(
          '后缀里出现了更小的值，因此本轮最终交换的目标位置也随之改变。',
          'A smaller value appears in the suffix, so the target of the final swap changes as well.',
        ),
        cards: [
          { label: 'i', value: step.passIndex },
          { label: 'min_idx', value: step.minIndex },
          { label: t('当前最小值', 'Current minimum'), value: step.array[step.minIndex] },
        ],
      };
    }

    if (step.kind === 'place') {
      return {
        activeLineLabel: t('把最小值放到前面', 'Place the minimum at the front'),
        title: step.didSwap
          ? t(`交换下标 ${step.passIndex} 与 ${step.minIndex}`, `Swap indices ${step.passIndex} and ${step.minIndex}`)
          : t(`本轮无需交换，下标 ${step.passIndex} 已经是最小值`, `No swap this pass; index ${step.passIndex} is already the minimum`),
        detail: t(
          `本轮结束后，下标 ${step.passIndex} 进入最终位置。`,
          `After this pass, index ${step.passIndex} is in its final position.`,
        ),
        cards: [
          { label: 'i', value: step.passIndex },
          { label: 'min_idx', value: step.minIndex },
          { label: t('最终前缀', 'Final prefix'), value: `[0, ${step.passIndex}]` },
        ],
      };
    }

    return {
      activeLineLabel: t('结束', 'Finish'),
      title: t('结束：所有位置都已确定', 'Finish: every position is final'),
      detail: t(
        'Selection sort 的优点是每轮最多一次交换，缺点是无论输入如何都要做完整扫描。',
        'Selection sort performs at most one swap per pass, but it pays for a full scan regardless of input order.',
      ),
      cards: [
        { label: 'i', value: 'done' },
        { label: 'min_idx', value: '—' },
        { label: t('最终状态', 'Final state'), value: t('整个数组有序', 'the array is sorted') },
      ],
    };
  }

  if (step.kind === 'start') {
    return {
      activeLineLabel: t('初始化外层循环', 'Initialize the outer loop'),
      title: t('开始：每一轮把当前最大值冒到右端', 'Start: each pass bubbles the current maximum to the right'),
      detail: t(
        'Bubble sort 只做相邻交换，因此最容易保持稳定性，也最容易加入提前退出优化。',
        'Bubble sort only uses adjacent swaps, so it is easy to keep stable and easy to optimize with an early exit.',
      ),
      cards: [
        { label: t('未排序右端', 'Unsorted end'), value: step.passEnd },
        { label: t('交换状态', 'Swap state'), value: '—' },
        { label: t('不变式', 'Invariant'), value: t('当前没有最终后缀', 'no final suffix yet') },
      ],
    };
  }

  if (step.kind === 'passStart') {
    return {
      activeLineLabel: t('重置 swapped 标记', 'Reset the swapped flag'),
      title: t(
        `第 ${step.passNumber} 轮：处理区间 [0, ${step.passEnd}]`,
        `Pass ${step.passNumber}: process [0, ${step.passEnd}]`,
      ),
      detail: t(
        '这一轮结束后，下标 end 会成为当前未排序区间中的最大值位置。',
        'At the end of this pass, index end will contain the maximum of the current unsorted range.',
      ),
      cards: [
        { label: t('end', 'end'), value: step.passEnd },
        { label: t('交换状态', 'Swap state'), value: t('尚未交换', 'no swap yet') },
        { label: t('已确定后缀', 'Final suffix'), value: step.passEnd === step.array.length - 1 ? '—' : `[${step.passEnd + 1}, ${step.array.length - 1}]` },
      ],
    };
  }

  if (step.kind === 'compare') {
    return {
      activeLineLabel: t('比较相邻元素', 'Compare adjacent values'),
      title: t(
        `比较 arr[${step.compareIndices[0]}] = ${step.array[step.compareIndices[0]]} 与 arr[${step.compareIndices[1]}] = ${step.array[step.compareIndices[1]]}`,
        `Compare arr[${step.compareIndices[0]}] = ${step.array[step.compareIndices[0]]} with arr[${step.compareIndices[1]}] = ${step.array[step.compareIndices[1]]}`,
      ),
      detail: t(
        '只有发生逆序时才交换。相等值不会互换，因此这版 bubble sort 是稳定的。',
        'A swap happens only on an inversion. Equal values never cross, so this bubble sort remains stable.',
      ),
      cards: [
        { label: t('end', 'end'), value: step.passEnd },
        { label: t('比较对', 'Compared pair'), value: `[${step.compareIndices.join(', ')}]` },
        { label: t('已确定后缀', 'Final suffix'), value: step.finalIndices.length ? `[${step.finalIndices[0]}, ${step.array.length - 1}]` : '—' },
      ],
    };
  }

  if (step.kind === 'swap') {
    return {
      activeLineLabel: t('交换逆序对', 'Swap the inversion'),
      title: t(
        `交换相邻逆序对 [${step.moveIndices.join(', ')}]`,
        `Swap the adjacent inversion [${step.moveIndices.join(', ')}]`,
      ),
      detail: t(
        '最大值会沿着相邻交换逐步向右移动。这也是“冒泡”这个名字的由来。',
        'The maximum drifts right through adjacent swaps. That is the mechanical pattern behind the name.',
      ),
      cards: [
        { label: t('end', 'end'), value: step.passEnd },
        { label: t('交换状态', 'Swap state'), value: t('本轮已发生交换', 'a swap happened this pass') },
        { label: t('已确定后缀', 'Final suffix'), value: step.finalIndices.length ? `[${step.finalIndices[0]}, ${step.array.length - 1}]` : '—' },
      ],
    };
  }

  if (step.kind === 'passDone') {
    return {
      activeLineLabel: t('检查是否提前结束', 'Check for an early exit'),
      title: t(
        `本轮结束：下标 ${step.passEnd} 进入最终位置`,
        `Pass complete: index ${step.passEnd} is now final`,
      ),
      detail: t(
        '这一轮至少发生过一次交换，因此还需要继续处理更短的未排序前缀。',
        'At least one swap happened this pass, so a shorter unsorted prefix still remains.',
      ),
      cards: [
        { label: t('end', 'end'), value: step.passEnd },
        { label: t('交换状态', 'Swap state'), value: t('继续下一轮', 'continue to the next pass') },
        { label: t('最终后缀', 'Final suffix'), value: `[${step.passEnd}, ${step.array.length - 1}]` },
      ],
    };
  }

  if (step.kind === 'earlyStop') {
    return {
      activeLineLabel: t('检查是否提前结束', 'Check for an early exit'),
      title: t('本轮没有交换，数组已经有序', 'No swap this pass; the array is already sorted'),
      detail: t(
        '这就是 bubble sort 的最好情况 `O(n)`：扫描一轮后直接退出。',
        'This is the `O(n)` best case in bubble sort: one full scan, then an immediate exit.',
      ),
      cards: [
        { label: t('end', 'end'), value: step.passEnd },
        { label: t('交换状态', 'Swap state'), value: t('提前结束', 'early exit') },
        { label: t('最终状态', 'Final state'), value: t('所有位置已确定', 'all positions are sorted') },
      ],
    };
  }

  return {
    activeLineLabel: t('结束', 'Finish'),
    title: t('结束：整个数组有序', 'Finish: the whole array is sorted'),
    detail: t(
      'Bubble sort 的主要价值是相邻交换和提前退出这两个性质，而不是理论复杂度。',
      'The main value of bubble sort is its adjacent-swap structure and early-exit rule, not its asymptotic complexity.',
    ),
    cards: [
      { label: t('end', 'end'), value: 'done' },
      { label: t('交换状态', 'Swap state'), value: '—' },
      { label: t('最终状态', 'Final state'), value: t('整个数组有序', 'the array is sorted') },
    ],
  };
}

function SimpleSortRaceVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('insertion');
  const [activeStep, setActiveStep] = useState(0);
  const steps = SIMPLE_SORT_RACE_STEPS[mode];
  const step = steps[activeStep];
  const modeLabels = {
    insertion: 'Insertion',
    selection: 'Selection',
    bubble: 'Bubble',
  };
  const copy = describeSimpleSortStep(mode, step, t);

  return (
    <section
      className="simple-sort-race-visual"
      aria-label={t('简单排序对比演示', 'Simple sorting comparison walkthrough')}
    >
      <header className="simple-sort-race-header">
        <div>
          <p className="eyebrow">{t('O(n²) 家族', 'O(n²) family')}</p>
          <h2>{t('同一组输入，比较位移、选最小值、相邻交换', 'One input, compared through shifts, min-selection, and adjacent swaps')}</h2>
          <p>{t(
            `固定数组 [${SORT_RACE_EXAMPLE.join(', ')}]。切换模式时，不改变输入，只改变“本轮不变式”和局部操作。`,
            `The array stays fixed at [${SORT_RACE_EXAMPLE.join(', ')}]. Switching modes changes only the invariant and the local operation.`,
          )}</p>
        </div>
        <div className="simple-sort-race-mode" role="group" aria-label={t('选择简单排序算法', 'Choose the simple sort')}>
          {Object.entries(modeLabels).map(([key, label]) => (
            <button
              type="button"
              className={mode === key ? 'active' : ''}
              aria-pressed={mode === key}
              key={key}
              onClick={() => {
                setMode(key);
                setActiveStep(0);
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      <div className={`simple-sort-race-step-copy ${step.kind}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{copy.title}</strong>
        <p>{copy.detail}</p>
      </div>

      <div className="simple-sort-race-workspace">
        <div className="simple-sort-race-stage-card">
          <div className="simple-sort-race-stage-heading">
            <span>{t('固定样例', 'Fixed sample')}</span>
            <strong>[{SORT_RACE_EXAMPLE.join(', ')}]</strong>
          </div>

          <div className="simple-sort-race-chart" aria-label={t('简单排序数组状态', 'Simple sort array state')}>
            {step.array.map((value, index) => {
              const isCompared = step.compareIndices.includes(index);
              const isMoved = step.moveIndices.includes(index);
              const isSettled = step.finalIndices.includes(index);
              const isSortedPrefix = mode === 'insertion' && step.sortedPrefixIndices.includes(index) && !isSettled;
              const isKey = mode === 'insertion'
                && step.insertIndex === index
                && ['pick', 'compare', 'shift', 'insert'].includes(step.kind);
              const isMinimum = mode === 'selection'
                && step.minIndex === index
                && ['pick', 'compare', 'updateMin', 'place'].includes(step.kind);
              const inActiveRange = isIndexInRange(step.activeRange, index);
              const tags = [];

              if (isSettled) tags.push(t('final', 'final'));
              else if (isSortedPrefix) tags.push(t('prefix', 'prefix'));
              if (isKey) tags.push('key');
              if (isMinimum) tags.push('min');

              return (
                <div
                  className={[
                    'simple-sort-race-column',
                    inActiveRange ? 'active-range' : '',
                    isCompared ? 'compare' : '',
                    isMoved ? 'move' : '',
                    isSettled ? 'settled' : '',
                    isSortedPrefix ? 'sorted-prefix' : '',
                    isKey ? 'key-slot' : '',
                    isMinimum ? 'minimum' : '',
                  ].filter(Boolean).join(' ')}
                  key={index}
                >
                  <div className="simple-sort-race-track">
                    <div
                      className="simple-sort-race-bar"
                      style={{ height: `${(value / SORT_RACE_MAX) * 100}%` }}
                    >
                      <span>{value}</span>
                    </div>
                  </div>
                  <small>i = {index}</small>
                  <div className="simple-sort-race-tags">
                    {tags.length === 0
                      ? <i>{t('—', '—')}</i>
                      : tags.map((tag) => <i key={tag}>{tag}</i>)}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="simple-sort-race-cards">
            {copy.cards.map((card) => (
              <div className="simple-sort-race-card" key={card.label}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="simple-sort-race-code" aria-label={t('当前排序代码', 'Active sorting code')}>
          <div className="simple-sort-race-code-heading">
            <span>{modeLabels[mode]} sort</span>
            <strong>{t('当前执行', 'Now')}: {copy.activeLineLabel}</strong>
          </div>
          <div className="simple-sort-race-code-lines">
            {SIMPLE_SORT_RACE_CODE_LINES[mode].map((block) => (
              <div
                className={step.activeLine === block.id ? 'active' : ''}
                aria-current={step.activeLine === block.id ? 'step' : undefined}
                key={block.id}
              >
                {block.code.map((line, lineIndex) => (
                  <code key={lineIndex}>{line}</code>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="simple-sort-race-legend">
        <span><i className="compare" />{t('本步比较', 'compared this step')}</span>
        <span><i className="move" />{t('本步移动或交换', 'moved or swapped this step')}</span>
        <span><i className="settled" />{t('已在最终位置', 'final position')}</span>
        <span><i className="prefix" />{t('局部有序 / 当前角色', 'local invariant / current role')}</span>
        <strong>{t('Insertion 只标记有序前缀；Selection / Bubble 会逐步标记最终位置。', 'Insertion marks a sorted prefix; Selection and Bubble gradually mark final positions.')}</strong>
      </div>

      <div className="simple-sort-race-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择简单排序步骤', 'Select a simple-sort step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          disabled={activeStep === steps.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

function buildMergeSortRaceSteps() {
  const array = [...SORT_RACE_EXAMPLE];
  const steps = [
    snapshotSortRaceStep({
      kind: 'start',
      activeLine: 'split',
      array,
      mid: null,
      activeRange: [0, array.length - 1],
      leftRange: null,
      rightRange: null,
      sortedRange: null,
      finalIndices: [],
      compareIndices: [],
      moveIndices: [],
      mergePointers: null,
    }),
  ];

  function sort(lo, hi) {
    if (lo >= hi) {
      steps.push(snapshotSortRaceStep({
        kind: 'base',
        activeLine: 'base',
        array,
        mid: lo,
        activeRange: [lo, hi],
        leftRange: null,
        rightRange: null,
        sortedRange: [lo, hi],
        finalIndices: [],
        compareIndices: [],
        moveIndices: [],
        mergePointers: null,
      }));
      return;
    }

    const mid = Math.floor((lo + hi) / 2);
    steps.push(snapshotSortRaceStep({
      kind: 'split',
      activeLine: 'split',
      array,
      mid,
      activeRange: [lo, hi],
      leftRange: [lo, mid],
      rightRange: [mid + 1, hi],
      sortedRange: null,
      finalIndices: [],
      compareIndices: [],
      moveIndices: [],
      mergePointers: null,
    }));

    sort(lo, mid);
    sort(mid + 1, hi);

    const left = array.slice(lo, mid + 1);
    const right = array.slice(mid + 1, hi + 1);
    let i = 0;
    let j = 0;
    let k = lo;

    steps.push(snapshotSortRaceStep({
      kind: 'prepare',
      activeLine: 'prepare',
      array,
      mid,
      activeRange: [lo, hi],
      leftRange: [lo, mid],
      rightRange: [mid + 1, hi],
      sortedRange: null,
      finalIndices: [],
      compareIndices: [],
      moveIndices: [],
      mergePointers: { left: lo, right: mid + 1, write: lo },
    }));

    while (i < left.length && j < right.length) {
      const leftIndex = lo + i;
      const rightIndex = mid + 1 + j;

      steps.push(snapshotSortRaceStep({
        kind: 'compare',
        activeLine: 'compare',
        array,
        mid,
        activeRange: [lo, hi],
        leftRange: [lo, mid],
        rightRange: [mid + 1, hi],
        sortedRange: null,
        finalIndices: [],
        compareIndices: [leftIndex, rightIndex],
        moveIndices: [],
        mergePointers: { left: leftIndex, right: rightIndex, write: k },
      }));

      let source = 'left';
      if (left[i] <= right[j]) {
        array[k] = left[i];
        i += 1;
      } else {
        array[k] = right[j];
        j += 1;
        source = 'right';
      }

      steps.push(snapshotSortRaceStep({
        kind: 'write',
        activeLine: 'write',
        array,
        mid,
        activeRange: [lo, hi],
        leftRange: [lo, mid],
        rightRange: [mid + 1, hi],
        sortedRange: [lo, k],
        finalIndices: [],
        compareIndices: [],
        moveIndices: [k],
        mergePointers: {
          left: i < left.length ? lo + i : null,
          right: j < right.length ? mid + 1 + j : null,
          write: k,
        },
        writeIndex: k,
        writtenValue: array[k],
        source,
      }));

      k += 1;
    }

    while (i < left.length) {
      array[k] = left[i];
      i += 1;

      steps.push(snapshotSortRaceStep({
        kind: 'drainLeft',
        activeLine: 'drain',
        array,
        mid,
        activeRange: [lo, hi],
        leftRange: [lo, mid],
        rightRange: [mid + 1, hi],
        sortedRange: [lo, k],
        finalIndices: [],
        compareIndices: [],
        moveIndices: [k],
        mergePointers: {
          left: i < left.length ? lo + i : null,
          right: j < right.length ? mid + 1 + j : null,
          write: k,
        },
        writeIndex: k,
        writtenValue: array[k],
        source: 'left',
      }));

      k += 1;
    }

    while (j < right.length) {
      array[k] = right[j];
      j += 1;

      steps.push(snapshotSortRaceStep({
        kind: 'drainRight',
        activeLine: 'drain',
        array,
        mid,
        activeRange: [lo, hi],
        leftRange: [lo, mid],
        rightRange: [mid + 1, hi],
        sortedRange: [lo, k],
        finalIndices: [],
        compareIndices: [],
        moveIndices: [k],
        mergePointers: {
          left: i < left.length ? lo + i : null,
          right: j < right.length ? mid + 1 + j : null,
          write: k,
        },
        writeIndex: k,
        writtenValue: array[k],
        source: 'right',
      }));

      k += 1;
    }

    steps.push(snapshotSortRaceStep({
      kind: 'merged',
      activeLine: 'drain',
      array,
      mid,
      activeRange: [lo, hi],
      leftRange: null,
      rightRange: null,
      sortedRange: [lo, hi],
      finalIndices: [],
      compareIndices: [],
      moveIndices: [],
      mergePointers: null,
    }));
  }

  sort(0, array.length - 1);
  steps.push(snapshotSortRaceStep({
    kind: 'finish',
    activeLine: 'finish',
    array,
    mid: null,
    activeRange: [0, array.length - 1],
    leftRange: null,
    rightRange: null,
    sortedRange: [0, array.length - 1],
    finalIndices: rangeInclusive(0, array.length - 1),
    compareIndices: [],
    moveIndices: [],
    mergePointers: null,
  }));

  return steps;
}

function buildQuickSortRaceSteps() {
  const array = [...SORT_RACE_EXAMPLE];
  const finalIndices = new Set();
  const steps = [
    snapshotSortRaceStep({
      kind: 'start',
      activeLine: 'pivot',
      array,
      activeRange: [0, array.length - 1],
      finalIndices: [],
      compareIndices: [],
      moveIndices: [],
      quickPointers: null,
      partitionBoundary: null,
      pivotIndex: null,
      pivotValue: null,
    }),
  ];

  function sort(lo, hi) {
    if (lo > hi) {
      return;
    }

    if (lo === hi) {
      finalIndices.add(lo);
      steps.push(snapshotSortRaceStep({
        kind: 'base',
        activeLine: 'base',
        array,
        activeRange: [lo, hi],
        finalIndices: sortedIndicesFromSet(finalIndices),
        compareIndices: [lo],
        moveIndices: [],
        quickPointers: { store: lo, scan: null },
        partitionBoundary: lo,
        pivotIndex: lo,
        pivotValue: array[lo],
      }));
      return;
    }

    let store = lo;
    const pivotValue = array[hi];

    steps.push(snapshotSortRaceStep({
      kind: 'pivot',
      activeLine: 'pivot',
      array,
      activeRange: [lo, hi],
      finalIndices: sortedIndicesFromSet(finalIndices),
      compareIndices: [hi],
      moveIndices: [],
      quickPointers: { store, scan: null },
      partitionBoundary: store,
      pivotIndex: hi,
      pivotValue,
    }));

    for (let scan = lo; scan < hi; scan += 1) {
      steps.push(snapshotSortRaceStep({
        kind: 'scan',
        activeLine: 'scan',
        array,
        activeRange: [lo, hi],
        finalIndices: sortedIndicesFromSet(finalIndices),
        compareIndices: [scan, hi],
        moveIndices: [],
        quickPointers: { store, scan },
        partitionBoundary: store,
        pivotIndex: hi,
        pivotValue,
      }));

      if (array[scan] < pivotValue) {
        [array[scan], array[store]] = [array[store], array[scan]];
        steps.push(snapshotSortRaceStep({
          kind: 'swap',
          activeLine: 'swap',
          array,
          activeRange: [lo, hi],
          finalIndices: sortedIndicesFromSet(finalIndices),
          compareIndices: [scan, hi],
          moveIndices: [scan, store],
          quickPointers: { store, scan },
          partitionBoundary: store + 1,
          pivotIndex: hi,
          pivotValue,
        }));
        store += 1;
      }
    }

    [array[store], array[hi]] = [array[hi], array[store]];
    finalIndices.add(store);

    steps.push(snapshotSortRaceStep({
      kind: 'placePivot',
      activeLine: 'place',
      array,
      activeRange: [lo, hi],
      finalIndices: sortedIndicesFromSet(finalIndices),
      compareIndices: [],
      moveIndices: [store, hi],
      quickPointers: { store, scan: null },
      partitionBoundary: store,
      pivotIndex: store,
      pivotValue: array[store],
    }));

    steps.push(snapshotSortRaceStep({
      kind: 'recurse',
      activeLine: 'recurse',
      array,
      activeRange: [lo, hi],
      finalIndices: sortedIndicesFromSet(finalIndices),
      compareIndices: [],
      moveIndices: [],
      quickPointers: { store, scan: null },
      partitionBoundary: store,
      pivotIndex: store,
      pivotValue: array[store],
    }));

    sort(lo, store - 1);
    sort(store + 1, hi);
  }

  sort(0, array.length - 1);
  steps.push(snapshotSortRaceStep({
    kind: 'finish',
    activeLine: 'finish',
    array,
    activeRange: [0, array.length - 1],
    finalIndices: rangeInclusive(0, array.length - 1),
    compareIndices: [],
    moveIndices: [],
    quickPointers: null,
    partitionBoundary: null,
    pivotIndex: null,
    pivotValue: null,
  }));

  return steps;
}

function buildHeapSortRaceSteps() {
  const array = [...SORT_RACE_EXAMPLE];
  const finalIndices = new Set();
  const steps = [
    snapshotSortRaceStep({
      kind: 'start',
      activeLine: 'build',
      array,
      heapEnd: array.length - 1,
      phase: 'build',
      activeRange: [0, array.length - 1],
      finalIndices: [],
      compareIndices: [],
      moveIndices: [],
      heapPointers: null,
    }),
  ];

  function heapify(heapSize, root, phase, extractedEnd = null) {
    let parent = root;

    while (true) {
      const left = 2 * parent + 1;
      const right = 2 * parent + 2;
      let largest = parent;
      const compared = [parent];
      if (left < heapSize) compared.push(left);
      if (right < heapSize) compared.push(right);

      steps.push(snapshotSortRaceStep({
        kind: 'compare',
        activeLine: 'compare',
        array,
        heapEnd: heapSize - 1,
        phase,
        extractedEnd,
        activeRange: [0, heapSize - 1],
        finalIndices: sortedIndicesFromSet(finalIndices),
        compareIndices: compared,
        moveIndices: [],
        heapPointers: {
          parent,
          left: left < heapSize ? left : null,
          right: right < heapSize ? right : null,
          largest,
        },
      }));

      if (left < heapSize && array[left] > array[largest]) {
        largest = left;
        steps.push(snapshotSortRaceStep({
          kind: 'select',
          activeLine: 'select',
          array,
          heapEnd: heapSize - 1,
          phase,
          extractedEnd,
          activeRange: [0, heapSize - 1],
          finalIndices: sortedIndicesFromSet(finalIndices),
          compareIndices: [parent, left],
          moveIndices: [left],
          heapPointers: {
            parent,
            left,
            right: right < heapSize ? right : null,
            largest,
          },
        }));
      }

      if (right < heapSize && array[right] > array[largest]) {
        largest = right;
        steps.push(snapshotSortRaceStep({
          kind: 'select',
          activeLine: 'select',
          array,
          heapEnd: heapSize - 1,
          phase,
          extractedEnd,
          activeRange: [0, heapSize - 1],
          finalIndices: sortedIndicesFromSet(finalIndices),
          compareIndices: [parent, right],
          moveIndices: [right],
          heapPointers: {
            parent,
            left: left < heapSize ? left : null,
            right,
            largest,
          },
        }));
      }

      if (largest === parent) {
        steps.push(snapshotSortRaceStep({
          kind: 'heapifyDone',
          activeLine: 'done',
          array,
          heapEnd: heapSize - 1,
          phase,
          extractedEnd,
          activeRange: [0, heapSize - 1],
          finalIndices: sortedIndicesFromSet(finalIndices),
          compareIndices: [parent],
          moveIndices: [],
          heapPointers: {
            parent,
            left: left < heapSize ? left : null,
            right: right < heapSize ? right : null,
            largest,
          },
        }));
        return;
      }

      [array[parent], array[largest]] = [array[largest], array[parent]];
      steps.push(snapshotSortRaceStep({
        kind: 'swap',
        activeLine: 'swap',
        array,
        heapEnd: heapSize - 1,
        phase,
        extractedEnd,
        activeRange: [0, heapSize - 1],
        finalIndices: sortedIndicesFromSet(finalIndices),
        compareIndices: [parent, largest],
        moveIndices: [parent, largest],
        heapPointers: {
          parent,
          left: left < heapSize ? left : null,
          right: right < heapSize ? right : null,
          largest,
        },
      }));

      parent = largest;
    }
  }

  for (let i = Math.floor(array.length / 2) - 1; i >= 0; i -= 1) {
    steps.push(snapshotSortRaceStep({
      kind: 'buildNode',
      activeLine: 'build',
      array,
      heapEnd: array.length - 1,
      phase: 'build',
      activeRange: [0, array.length - 1],
      finalIndices: sortedIndicesFromSet(finalIndices),
      compareIndices: [i],
      moveIndices: [],
      heapPointers: {
        parent: i,
        left: 2 * i + 1 < array.length ? 2 * i + 1 : null,
        right: 2 * i + 2 < array.length ? 2 * i + 2 : null,
        largest: i,
      },
    }));
    heapify(array.length, i, 'build');
  }

  steps.push(snapshotSortRaceStep({
    kind: 'heapBuilt',
    activeLine: 'build',
    array,
    heapEnd: array.length - 1,
    phase: 'build',
    activeRange: [0, array.length - 1],
    finalIndices: sortedIndicesFromSet(finalIndices),
    compareIndices: [0],
    moveIndices: [],
    heapPointers: { parent: 0, left: 1, right: 2, largest: 0 },
  }));

  for (let end = array.length - 1; end > 0; end -= 1) {
    [array[0], array[end]] = [array[end], array[0]];
    finalIndices.add(end);

    steps.push(snapshotSortRaceStep({
      kind: 'extract',
      activeLine: 'extract',
      array,
      heapEnd: end - 1,
      phase: 'extract',
      extractedEnd: end,
      activeRange: end > 1 ? [0, end - 1] : [0, 0],
      finalIndices: sortedIndicesFromSet(finalIndices),
      compareIndices: [0, end],
      moveIndices: [0, end],
      heapPointers: {
        parent: 0,
        left: 1 < end ? 1 : null,
        right: 2 < end ? 2 : null,
        largest: 0,
      },
    }));

    heapify(end, 0, 'extract', end);
  }

  finalIndices.add(0);
  steps.push(snapshotSortRaceStep({
    kind: 'finish',
    activeLine: 'finish',
    array,
    heapEnd: -1,
    phase: 'finish',
    activeRange: null,
    finalIndices: rangeInclusive(0, array.length - 1),
    compareIndices: [],
    moveIndices: [],
    heapPointers: null,
  }));

  return steps;
}

const EFFICIENT_SORT_RACE_STEPS = {
  merge: buildMergeSortRaceSteps(),
  quick: buildQuickSortRaceSteps(),
  heap: buildHeapSortRaceSteps(),
};

const EFFICIENT_SORT_RACE_CODE_LINES = {
  merge: [
    { id: 'base', code: ['if lo >= hi:', '    return'] },
    { id: 'split', code: ['mid = (lo + hi) // 2', 'sort(lo, mid); sort(mid + 1, hi)'] },
    { id: 'prepare', code: ['left = arr[lo:mid + 1]', 'right = arr[mid + 1:hi + 1]'] },
    { id: 'compare', code: ['while i < len(left) and j < len(right):'] },
    { id: 'write', code: ['    arr[k] = left[i] if left[i] <= right[j] else right[j]'] },
    { id: 'drain', code: ['copy the remaining suffix from left or right'] },
    { id: 'finish', code: ['return arr'] },
  ],
  quick: [
    { id: 'base', code: ['if lo >= hi:', '    return'] },
    { id: 'pivot', code: ['pivot = arr[hi]', 'store = lo'] },
    { id: 'scan', code: ['for scan in range(lo, hi):'] },
    { id: 'swap', code: ['if arr[scan] < pivot:', '    arr[scan], arr[store] = arr[store], arr[scan]', '    store += 1'] },
    { id: 'place', code: ['arr[store], arr[hi] = arr[hi], arr[store]'] },
    { id: 'recurse', code: ['quick_sort(lo, store - 1)', 'quick_sort(store + 1, hi)'] },
    { id: 'finish', code: ['return arr'] },
  ],
  heap: [
    { id: 'build', code: ['for i in range(n // 2 - 1, -1, -1):', '    heapify(n, i)'] },
    { id: 'compare', code: ['left = 2 * i + 1', 'right = 2 * i + 2'] },
    { id: 'select', code: ['largest = argmax(i, left, right) inside the heap'] },
    { id: 'swap', code: ['if largest != i:', '    arr[i], arr[largest] = arr[largest], arr[i]', '    heapify(heap_size, largest)'] },
    { id: 'done', code: ['if largest == i:', '    return'] },
    { id: 'extract', code: ['for end in range(n - 1, 0, -1):', '    arr[0], arr[end] = arr[end], arr[0]', '    heapify(end, 0)'] },
    { id: 'finish', code: ['return arr'] },
  ],
};

function describeEfficientSortStep(mode, step, t) {
  if (mode === 'merge') {
    if (step.kind === 'start') {
      return {
        activeLineLabel: t('拆分当前区间', 'Split the current range'),
        title: t('开始：对整个数组做分治', 'Start: apply divide and conquer to the full array'),
        detail: t(
          'Merge sort 先递归把子区间排好，再在返回阶段做线性合并。',
          'Merge sort recursively sorts subranges first, then performs linear merges on the way back up.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: 'mid', value: '—' },
          { label: t('不变式', 'Invariant'), value: t('子问题先有序，再合并', 'subproblems first, then merge') },
        ],
      };
    }

    if (step.kind === 'base') {
      return {
        activeLineLabel: t('命中长度 1 的基线', 'Hit the length-1 base case'),
        title: t(
          `基线：区间 ${formatSortRaceRange(step.activeRange)} 不再拆分`,
          `Base case: range ${formatSortRaceRange(step.activeRange)} stops splitting`,
        ),
        detail: t(
          '单个元素天然有序，但它还不是全局最终位置，只是一个已经排好的子问题。',
          'A single element is already sorted, but it is not yet a globally final position; it is only a solved subproblem.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: 'mid', value: step.mid },
          { label: t('状态', 'State'), value: t('局部已排序', 'locally sorted') },
        ],
      };
    }

    if (step.kind === 'split') {
      return {
        activeLineLabel: t('拆成左右两半', 'Split into left and right halves'),
        title: t(
          `拆分区间 ${formatSortRaceRange(step.activeRange)}，mid = ${step.mid}`,
          `Split range ${formatSortRaceRange(step.activeRange)}, mid = ${step.mid}`,
        ),
        detail: t(
          `左半是 ${formatSortRaceRange(step.leftRange)}，右半是 ${formatSortRaceRange(step.rightRange)}。`,
          `The left half is ${formatSortRaceRange(step.leftRange)} and the right half is ${formatSortRaceRange(step.rightRange)}.`,
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: 'mid', value: step.mid },
          { label: t('下一步', 'Next'), value: t('先递归左右子区间', 'recurse into both halves') },
        ],
      };
    }

    if (step.kind === 'prepare') {
      return {
        activeLineLabel: t('准备左右缓冲区', 'Prepare the left and right buffers'),
        title: t(
          `开始合并 ${formatSortRaceRange(step.activeRange)}`,
          `Start merging ${formatSortRaceRange(step.activeRange)}`,
        ),
        detail: t(
          '此时左右两半已经分别有序。接下来只需要用两个指针做线性合并。',
          'At this point the two halves are already sorted individually. Only a two-pointer linear merge remains.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('指针', 'Pointers'), value: `i = ${step.mergePointers.left}, j = ${step.mergePointers.right}, k = ${step.mergePointers.write}` },
          { label: t('不变式', 'Invariant'), value: t('left / right 各自有序', 'left / right are individually sorted') },
        ],
      };
    }

    if (step.kind === 'compare') {
      return {
        activeLineLabel: t('比较左右指针', 'Compare the two merge pointers'),
        title: t(
          `比较 arr[${step.compareIndices[0]}] = ${step.array[step.compareIndices[0]]} 与 arr[${step.compareIndices[1]}] = ${step.array[step.compareIndices[1]]}`,
          `Compare arr[${step.compareIndices[0]}] = ${step.array[step.compareIndices[0]]} with arr[${step.compareIndices[1]}] = ${step.array[step.compareIndices[1]]}`,
        ),
        detail: t(
          '稳定版本在相等时优先取左边，因此相等元素的原始相对顺序会被保留。',
          'A stable implementation takes the left side first on ties, preserving the original order of equal values.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('指针', 'Pointers'), value: `i = ${step.mergePointers.left}, j = ${step.mergePointers.right}, k = ${step.mergePointers.write}` },
          { label: t('比较结果', 'Comparison'), value: step.array[step.compareIndices[0]] <= step.array[step.compareIndices[1]] ? 'left <= right' : 'right < left' },
        ],
      };
    }

    if (step.kind === 'write') {
      return {
        activeLineLabel: t('把更小值写回数组', 'Write the smaller value back'),
        title: t(
          `把 ${step.source === 'left' ? 'left' : 'right'} 的值 ${step.writtenValue} 写到 arr[${step.writeIndex}]`,
          `Write ${step.writtenValue} from the ${step.source} buffer into arr[${step.writeIndex}]`,
        ),
        detail: t(
          '蓝绿色高亮表示已经合并好的前缀。写指针 `k` 只向右前进，不会回退。',
          'The blue-green highlight marks the merged prefix. The write pointer `k` moves only to the right.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('写入位置', 'Write index'), value: step.writeIndex },
          { label: t('已合并前缀', 'Merged prefix'), value: formatSortRaceRange(step.sortedRange) },
        ],
      };
    }

    if (step.kind === 'drainLeft' || step.kind === 'drainRight') {
      return {
        activeLineLabel: t('拷贝剩余后缀', 'Copy the remaining suffix'),
        title: t(
          `继续拷贝 ${step.source === 'left' ? 'left' : 'right'} 的剩余值到 arr[${step.writeIndex}]`,
          `Copy the remaining ${step.source} value into arr[${step.writeIndex}]`,
        ),
        detail: t(
          '一侧缓冲区耗尽后，另一侧的剩余元素已经保持有序，直接顺序写回即可。',
          'Once one buffer is exhausted, the other side is already ordered and can be copied back directly.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('写入位置', 'Write index'), value: step.writeIndex },
          { label: t('已合并前缀', 'Merged prefix'), value: formatSortRaceRange(step.sortedRange) },
        ],
      };
    }

    if (step.kind === 'merged') {
      return {
        activeLineLabel: t('本轮合并结束', 'Finish the current merge'),
        title: t(
          `区间 ${formatSortRaceRange(step.activeRange)} 已经局部有序`,
          `Range ${formatSortRaceRange(step.activeRange)} is now locally sorted`,
        ),
        detail: t(
          '之后它会作为更大子问题的一半继续参与上一层合并。',
          'This range will now serve as one sorted half inside a larger merge.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('已排序子区间', 'Sorted subrange'), value: formatSortRaceRange(step.sortedRange) },
          { label: t('状态', 'State'), value: t('等待上一层合并', 'waiting for the parent merge') },
        ],
      };
    }

    return {
      activeLineLabel: t('结束', 'Finish'),
      title: t('结束：整个数组已经排好序', 'Finish: the whole array is sorted'),
      detail: t(
        'Merge sort 的时间复杂度稳定在 `O(n log n)`，代价是这版实现需要 `O(n)` 额外空间。',
        'Merge sort stays at `O(n log n)` time, at the cost of `O(n)` extra space in this implementation.',
      ),
      cards: [
        { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
        { label: t('额外空间', 'Extra space'), value: 'O(n)' },
        { label: t('最终状态', 'Final state'), value: t('稳定且有序', 'stable and sorted') },
      ],
    };
  }

  if (mode === 'quick') {
    const store = step.quickPointers?.store ?? null;
    const scan = step.quickPointers?.scan ?? null;

    if (step.kind === 'start') {
      return {
        activeLineLabel: t('初始化 partition', 'Initialize the partition'),
        title: t('开始：对整个数组做第一次 partition', 'Start: perform the first partition on the full array'),
        detail: t(
          '这版 quicksort 使用 Lomuto partition：`store` 指向下一个“小于 pivot”的写入位置。',
          'This quicksort uses Lomuto partition: `store` points to the next slot for a value smaller than the pivot.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('pivot', 'pivot'), value: '—' },
          { label: t('不变式', 'Invariant'), value: t('先 partition，再递归两侧', 'partition first, recurse later') },
        ],
      };
    }

    if (step.kind === 'base') {
      return {
        activeLineLabel: t('命中基线', 'Hit the base case'),
        title: t(
          `基线：单点区间 ${formatSortRaceRange(step.activeRange)} 已经到位`,
          `Base case: single-index range ${formatSortRaceRange(step.activeRange)} is already final`,
        ),
        detail: t(
          '快排的“最终位置”来自 partition。单点区间不需要再做任何工作。',
          'Quicksort gets final positions through partitioning. A single-index range needs no more work.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('pivot', 'pivot'), value: step.pivotValue },
          { label: t('状态', 'State'), value: t('该位置已最终确定', 'this position is final') },
        ],
      };
    }

    if (step.kind === 'pivot') {
      return {
        activeLineLabel: t('选择 pivot 并初始化 store', 'Choose the pivot and initialize store'),
        title: t(
          `选择 pivot = arr[${step.pivotIndex}] = ${step.pivotValue}`,
          `Choose pivot = arr[${step.pivotIndex}] = ${step.pivotValue}`,
        ),
        detail: t(
          '扫描区间时，`store` 左边保持 `< pivot`，`store` 到 `scan - 1` 保持 `>= pivot`。',
          'During the scan, everything left of `store` stays `< pivot`, while `store .. scan - 1` stays `>= pivot`.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('pivot', 'pivot'), value: `${step.pivotValue} @ ${step.pivotIndex}` },
          { label: t('store', 'store'), value: store },
        ],
      };
    }

    if (step.kind === 'scan') {
      return {
        activeLineLabel: t('扫描 partition 区间', 'Scan the partition range'),
        title: t(
          `比较 arr[${scan}] = ${step.array[scan]} 与 pivot = ${step.pivotValue}`,
          `Compare arr[${scan}] = ${step.array[scan]} with pivot = ${step.pivotValue}`,
        ),
        detail: t(
          '如果当前值小于 pivot，就把它交换到 `store` 位置，并把 `store` 右移一格。',
          'If the current value is smaller than the pivot, swap it into the `store` position and move `store` one step right.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('pivot', 'pivot'), value: `${step.pivotValue} @ ${step.pivotIndex}` },
          { label: t('指针', 'Pointers'), value: `store = ${store}, scan = ${scan}` },
        ],
      };
    }

    if (step.kind === 'swap') {
      return {
        activeLineLabel: t('把较小值交换到左边', 'Swap a smaller value into the left partition'),
        title: t(
          `arr[${scan}] < pivot，交换下标 ${step.moveIndices[0]} 与 ${step.moveIndices[1]}`,
          `arr[${scan}] < pivot, so swap indices ${step.moveIndices[0]} and ${step.moveIndices[1]}`,
        ),
        detail: t(
          '交换后，左侧分区边界右移一格。绿色区域就是已经确认 `< pivot` 的部分。',
          'After the swap, the left-partition boundary moves one step right. The green region marks values already known to be `< pivot`.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('pivot', 'pivot'), value: `${step.pivotValue} @ ${step.pivotIndex}` },
          { label: t('下一条边界', 'Next boundary'), value: step.partitionBoundary },
        ],
      };
    }

    if (step.kind === 'placePivot') {
      return {
        activeLineLabel: t('把 pivot 放回最终位置', 'Place the pivot into its final position'),
        title: t(
          `交换 pivot 到下标 ${step.pivotIndex}`,
          `Swap the pivot into index ${step.pivotIndex}`,
        ),
        detail: t(
          'partition 完成后，pivot 左边都 `< pivot`，右边都 `>= pivot`，因此 pivot 的位置已经最终确定。',
          'Once partition completes, everything left of the pivot is `< pivot` and everything right is `>= pivot`, so the pivot is now final.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('pivot', 'pivot'), value: `${step.pivotValue} @ ${step.pivotIndex}` },
          { label: t('状态', 'State'), value: t('pivot 已最终确定', 'pivot is now final') },
        ],
      };
    }

    if (step.kind === 'recurse') {
      return {
        activeLineLabel: t('递归左右两侧', 'Recurse on both sides'),
        title: t(
          `递归处理 ${formatSortRaceRange([step.activeRange[0], step.pivotIndex - 1])} 和 ${formatSortRaceRange([step.pivotIndex + 1, step.activeRange[1]])}`,
          `Recurse on ${formatSortRaceRange([step.activeRange[0], step.pivotIndex - 1])} and ${formatSortRaceRange([step.pivotIndex + 1, step.activeRange[1]])}`,
        ),
        detail: t(
          '快排的最终有序性来自“每个 pivot 都被放到最终位置”，然后对子区间重复同样的动作。',
          'Quicksort becomes sorted because each pivot is placed in its final position, and the same action repeats on both subranges.',
        ),
        cards: [
          { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
          { label: t('pivot', 'pivot'), value: `${step.pivotValue} @ ${step.pivotIndex}` },
          { label: t('已确定位置数', 'Final positions'), value: step.finalIndices.length },
        ],
      };
    }

    return {
      activeLineLabel: t('结束', 'Finish'),
      title: t('结束：整个数组已经排好序', 'Finish: the whole array is sorted'),
      detail: t(
        '这版 quicksort 是原地的，但固定取末尾元素做 pivot 仍然有最坏 `O(n^2)` 的情况。',
        'This quicksort is in-place, but choosing the last element as the pivot still leaves a worst-case `O(n^2)` path.',
      ),
      cards: [
        { label: t('当前区间', 'Current range'), value: formatSortRaceRange(step.activeRange) },
        { label: t('额外空间', 'Extra space'), value: t('平均 O(log n) 栈', 'O(log n) average stack') },
        { label: t('最终状态', 'Final state'), value: t('原地且有序', 'in-place and sorted') },
      ],
    };
  }

  const parent = step.heapPointers?.parent ?? null;
  const left = step.heapPointers?.left ?? null;
  const right = step.heapPointers?.right ?? null;
  const largest = step.heapPointers?.largest ?? null;
  const heapLabel = step.heapEnd >= 0 ? `[0, ${step.heapEnd}]` : '—';
  const finalSuffixLabel = step.heapEnd >= 0 && step.heapEnd < step.array.length - 1
    ? `[${step.heapEnd + 1}, ${step.array.length - 1}]`
    : (step.heapEnd < 0 ? `[0, ${step.array.length - 1}]` : '—');

  if (step.kind === 'start') {
    return {
      activeLineLabel: t('开始建堆', 'Start building the heap'),
      title: t('开始：先把数组原地建成最大堆', 'Start: build an in-place max heap first'),
      detail: t(
        'Heap sort 先完成一次 `O(n)` 建堆，再不断把堆顶最大值交换到数组末尾。',
        'Heap sort first builds a max heap in `O(n)`, then repeatedly swaps the maximum root to the end of the array.',
      ),
      cards: [
        { label: t('当前堆', 'Current heap'), value: heapLabel },
        { label: t('最终后缀', 'Final suffix'), value: '—' },
        { label: t('不变式', 'Invariant'), value: t('堆顶始终是当前最大值', 'the heap root is the current maximum') },
      ],
    };
  }

  if (step.kind === 'buildNode') {
    return {
      activeLineLabel: t('从最后一个非叶子节点开始 heapify', 'Heapify from the last internal node'),
      title: t(
        `建堆：对下标 ${parent} 做 heapify`,
        `Build heap: heapify index ${parent}`,
      ),
      detail: t(
        '建堆阶段从最后一个非叶子节点向前走，因为叶子节点天然满足堆性质。',
        'The build phase walks backward from the last internal node because leaves already satisfy the heap property.',
      ),
      cards: [
        { label: t('当前堆', 'Current heap'), value: heapLabel },
        { label: t('节点', 'Nodes'), value: `p = ${formatSortRaceIndex(parent)}, l = ${formatSortRaceIndex(left)}, r = ${formatSortRaceIndex(right)}` },
        { label: t('阶段', 'Phase'), value: t('建堆', 'build heap') },
      ],
    };
  }

  if (step.kind === 'compare') {
    return {
      activeLineLabel: t('比较父节点与孩子节点', 'Compare the parent with its children'),
      title: t(
        `检查 parent = ${parent}，left = ${formatSortRaceIndex(left)}，right = ${formatSortRaceIndex(right)}`,
        `Inspect parent = ${parent}, left = ${formatSortRaceIndex(left)}, right = ${formatSortRaceIndex(right)}`,
      ),
      detail: t(
        '孩子下标来自固定公式：`left = 2*i + 1`，`right = 2*i + 2`。只有堆区间内的下标才参与比较。',
        'Child indices follow the fixed formulas `left = 2*i + 1` and `right = 2*i + 2`. Only indices inside the heap participate.',
      ),
      cards: [
        { label: t('当前堆', 'Current heap'), value: heapLabel },
        { label: t('节点', 'Nodes'), value: `p = ${formatSortRaceIndex(parent)}, l = ${formatSortRaceIndex(left)}, r = ${formatSortRaceIndex(right)}` },
        { label: t('最终后缀', 'Final suffix'), value: finalSuffixLabel },
      ],
    };
  }

  if (step.kind === 'select') {
    return {
      activeLineLabel: t('更新当前最大值候选', 'Update the current largest candidate'),
      title: t(
        `largest 更新为下标 ${largest}`,
        `Update largest to index ${largest}`,
      ),
      detail: t(
        'heapify 的核心不是全局扫描，而是沿着一条从根到叶的路径不断下沉较小的值。',
        'The core of heapify is not a global scan. It sinks the smaller value along one root-to-leaf path.',
      ),
      cards: [
        { label: t('当前堆', 'Current heap'), value: heapLabel },
        { label: t('largest', 'largest'), value: largest },
        { label: t('阶段', 'Phase'), value: step.phase === 'build' ? t('建堆', 'build heap') : t('抽取后恢复堆', 'restore heap after extraction') },
      ],
    };
  }

  if (step.kind === 'swap') {
    return {
      activeLineLabel: t('交换父节点与更大孩子', 'Swap the parent with the larger child'),
      title: t(
        `交换下标 ${step.moveIndices[0]} 与 ${step.moveIndices[1]}`,
        `Swap indices ${step.moveIndices[0]} and ${step.moveIndices[1]}`,
      ),
      detail: t(
        '交换后，较小的值继续向下沉，直到该子树重新满足最大堆性质。',
        'After the swap, the smaller value keeps sinking until the subtree satisfies the max-heap property again.',
      ),
      cards: [
        { label: t('当前堆', 'Current heap'), value: heapLabel },
        { label: t('交换节点', 'Swapped nodes'), value: `[${step.moveIndices.join(', ')}]` },
        { label: t('最终后缀', 'Final suffix'), value: finalSuffixLabel },
      ],
    };
  }

  if (step.kind === 'heapifyDone') {
    return {
      activeLineLabel: t('当前子树恢复堆性质', 'The current subtree satisfies the heap property'),
      title: t(
        `以 ${parent} 为根的子树已经满足最大堆`,
        `The subtree rooted at ${parent} now satisfies the max-heap property`,
      ),
      detail: t(
        '如果 `largest == parent`，说明父节点已经不小于两个孩子，可以结束当前 heapify。',
        'If `largest == parent`, the parent is already at least as large as both children, so the current heapify is done.',
      ),
      cards: [
        { label: t('当前堆', 'Current heap'), value: heapLabel },
        { label: t('largest', 'largest'), value: largest },
        { label: t('状态', 'State'), value: t('当前子树已修复', 'current subtree restored') },
      ],
    };
  }

  if (step.kind === 'heapBuilt') {
    return {
      activeLineLabel: t('建堆完成', 'Finish building the heap'),
      title: t('最大堆已经建好', 'The max heap is ready'),
      detail: t(
        '现在根节点就是整个未排序区间中的最大值，下一步可以把它交换到数组末尾。',
        'Now the root is the maximum of the unsorted range, so the next step can swap it to the end of the array.',
      ),
      cards: [
        { label: t('当前堆', 'Current heap'), value: heapLabel },
        { label: t('根节点', 'Root'), value: step.array[0] },
        { label: t('下一步', 'Next'), value: t('抽取最大值', 'extract the maximum') },
      ],
    };
  }

  if (step.kind === 'extract') {
    return {
      activeLineLabel: t('把堆顶最大值放到末尾', 'Move the maximum root to the end'),
      title: t(
        `交换堆顶与下标 ${step.extractedEnd}，该位置进入最终状态`,
        `Swap the heap root with index ${step.extractedEnd}; that index is now final`,
      ),
      detail: t(
        '抽取后，堆区间缩短一格。接下来只需要对根节点重新做一次 heapify。',
        'After extraction, the heap shrinks by one position. Only the root needs another heapify.',
      ),
      cards: [
        { label: t('当前堆', 'Current heap'), value: heapLabel },
        { label: t('最终后缀', 'Final suffix'), value: finalSuffixLabel },
        { label: t('阶段', 'Phase'), value: t('抽取最大值', 'extract maximum') },
      ],
    };
  }

  return {
    activeLineLabel: t('结束', 'Finish'),
    title: t('结束：整个数组已经排好序', 'Finish: the whole array is sorted'),
    detail: t(
      'Heap sort 保证最坏 `O(n log n)` 且额外空间 `O(1)`，但顺序访问和缓存局部性通常不如 quicksort。',
      'Heap sort guarantees `O(n log n)` worst-case time with `O(1)` extra space, but its access pattern is usually less cache-friendly than quicksort.',
    ),
    cards: [
      { label: t('当前堆', 'Current heap'), value: '—' },
      { label: t('最终后缀', 'Final suffix'), value: finalSuffixLabel },
      { label: t('最终状态', 'Final state'), value: t('所有位置已确定', 'all positions are final') },
    ],
  };
}

function EfficientSortRaceVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('merge');
  const [activeStep, setActiveStep] = useState(0);
  const steps = EFFICIENT_SORT_RACE_STEPS[mode];
  const step = steps[activeStep];
  const modeLabels = {
    merge: 'Merge',
    quick: 'Quick',
    heap: 'Heap',
  };
  const copy = describeEfficientSortStep(mode, step, t);

  return (
    <section
      className="efficient-sort-race-visual"
      aria-label={t('高效排序对比演示', 'Efficient sorting comparison walkthrough')}
    >
      <header className="efficient-sort-race-header">
        <div>
          <p className="eyebrow">{t('O(n log n) 家族', 'O(n log n) family')}</p>
          <h2>{t('同一组输入，比较分治、partition、heapify', 'One input, compared through divide-and-conquer, partitioning, and heapify')}</h2>
          <p>{t(
            `固定数组 [${SORT_RACE_EXAMPLE.join(', ')}]。Merge 看子区间与双指针，Quick 看 pivot 与边界，Heap 看 parent / child 下沉。`,
            `The array stays fixed at [${SORT_RACE_EXAMPLE.join(', ')}]. Merge exposes subranges and merge pointers, Quick exposes the pivot and boundary, and Heap exposes parent / child sift-down steps.`,
          )}</p>
        </div>
        <div className="efficient-sort-race-mode" role="group" aria-label={t('选择高效排序算法', 'Choose the efficient sort')}>
          {Object.entries(modeLabels).map(([key, label]) => (
            <button
              type="button"
              className={mode === key ? 'active' : ''}
              aria-pressed={mode === key}
              key={key}
              onClick={() => {
                setMode(key);
                setActiveStep(0);
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      <div className={`efficient-sort-race-step-copy ${step.kind}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{copy.title}</strong>
        <p>{copy.detail}</p>
      </div>

      <div className="efficient-sort-race-workspace">
        <div className="efficient-sort-race-stage-card">
          <div className="efficient-sort-race-stage-heading">
            <span>{t('固定样例', 'Fixed sample')}</span>
            <strong>[{SORT_RACE_EXAMPLE.join(', ')}]</strong>
          </div>

          <div className="efficient-sort-race-chart" aria-label={t('高效排序数组状态', 'Efficient sort array state')}>
            {step.array.map((value, index) => {
              const isSettled = step.finalIndices.includes(index);
              const isCompared = step.compareIndices.includes(index);
              const isMoved = step.moveIndices.includes(index);
              const inActiveRange = isIndexInRange(step.activeRange, index);
              const inLeftRange = isIndexInRange(step.leftRange, index);
              const inRightRange = isIndexInRange(step.rightRange, index);
              const inSortedRange = isIndexInRange(step.sortedRange, index);
              const isPivot = mode === 'quick' && step.pivotIndex === index;
              const isScan = mode === 'quick' && step.quickPointers?.scan === index;
              const isStore = mode === 'quick' && step.quickPointers?.store === index;
              const isPartitionLeft = mode === 'quick'
                && step.partitionBoundary !== null
                && inActiveRange
                && index < step.partitionBoundary;
              const isPartitionRight = mode === 'quick'
                && step.partitionBoundary !== null
                && inActiveRange
                && index > step.partitionBoundary
                && !isPivot;
              const isHeapZone = mode === 'heap' && step.heapEnd >= 0 && index <= step.heapEnd;
              const isParent = mode === 'heap' && step.heapPointers?.parent === index;
              const isLeftChild = mode === 'heap' && step.heapPointers?.left === index;
              const isRightChild = mode === 'heap' && step.heapPointers?.right === index;
              const isMergeLeftPointer = mode === 'merge' && step.mergePointers?.left === index;
              const isMergeRightPointer = mode === 'merge' && step.mergePointers?.right === index;
              const isWriteIndex = mode === 'merge' && step.mergePointers?.write === index;
              const tags = [];

              if (isSettled) tags.push(t('final', 'final'));
              if (mode === 'merge') {
                if (inLeftRange) tags.push('L');
                if (inRightRange) tags.push('R');
                if (isMergeLeftPointer) tags.push('i');
                if (isMergeRightPointer) tags.push('j');
                if (isWriteIndex) tags.push('k');
              } else if (mode === 'quick') {
                if (isPivot) tags.push('pivot');
                if (isStore) tags.push('store');
                if (isScan) tags.push('scan');
              } else {
                if (isParent) tags.push('parent');
                if (isLeftChild) tags.push('left');
                if (isRightChild) tags.push('right');
              }

              return (
                <div
                  className={[
                    'efficient-sort-race-column',
                    inActiveRange ? 'active-range' : '',
                    inLeftRange ? 'left-range' : '',
                    inRightRange ? 'right-range' : '',
                    inSortedRange ? 'sorted-range' : '',
                    isCompared ? 'compare' : '',
                    isMoved ? 'move' : '',
                    isSettled ? 'settled' : '',
                    isPivot ? 'pivot' : '',
                    isScan ? 'scan' : '',
                    isStore ? 'store' : '',
                    isPartitionLeft ? 'partition-left' : '',
                    isPartitionRight ? 'partition-right' : '',
                    isHeapZone ? 'heap-zone' : '',
                    isParent ? 'parent' : '',
                    isLeftChild || isRightChild ? 'child' : '',
                  ].filter(Boolean).join(' ')}
                  key={index}
                >
                  <div className="efficient-sort-race-track">
                    <div
                      className="efficient-sort-race-bar"
                      style={{ height: `${(value / SORT_RACE_MAX) * 100}%` }}
                    >
                      <span>{value}</span>
                    </div>
                  </div>
                  <small>i = {index}</small>
                  <div className="efficient-sort-race-tags">
                    {tags.length === 0
                      ? <i>{t('—', '—')}</i>
                      : tags.map((tag) => <i key={tag}>{tag}</i>)}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="efficient-sort-race-cards">
            {copy.cards.map((card) => (
              <div className="efficient-sort-race-card" key={card.label}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="efficient-sort-race-code" aria-label={t('当前排序代码', 'Active sorting code')}>
          <div className="efficient-sort-race-code-heading">
            <span>{modeLabels[mode]} sort</span>
            <strong>{t('当前执行', 'Now')}: {copy.activeLineLabel}</strong>
          </div>
          <div className="efficient-sort-race-code-lines">
            {EFFICIENT_SORT_RACE_CODE_LINES[mode].map((block) => (
              <div
                className={step.activeLine === block.id ? 'active' : ''}
                aria-current={step.activeLine === block.id ? 'step' : undefined}
                key={block.id}
              >
                {block.code.map((line, lineIndex) => (
                  <code key={lineIndex}>{line}</code>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="efficient-sort-race-legend">
        <span><i className="range" />{t('当前工作区间 / 堆区间', 'active subrange / heap')}</span>
        <span><i className="compare" />{t('本步比较', 'compared this step')}</span>
        <span><i className="move" />{t('本步写回或交换', 'written or swapped this step')}</span>
        <span><i className="settled" />{t('已在最终位置', 'final position')}</span>
        <strong>{t('Merge 标记 L / R / i / j / k；Quick 标记 pivot / store / scan；Heap 标记 parent / left / right。', 'Merge marks L / R / i / j / k; Quick marks pivot / store / scan; Heap marks parent / left / right.')}</strong>
      </div>

      <div className="efficient-sort-race-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择高效排序步骤', 'Select an efficient-sort step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          disabled={activeStep === steps.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const MONOTONIC_STACK_VALUES = [2, 1, 2, 4, 3];

function buildMonotonicStackSteps(mode) {
  const stack = [];
  const answers = Array(MONOTONIC_STACK_VALUES.length).fill(null);
  const steps = [{
    action: 'start',
    current: null,
    popped: null,
    stack: [],
    answers: [...answers],
  }];
  const resolves = mode === 'greater'
    ? (waiting, current) => current > waiting
    : (waiting, current) => current < waiting;

  MONOTONIC_STACK_VALUES.forEach((value, index) => {
    steps.push({
      action: 'scan',
      current: index,
      popped: null,
      stack: [...stack],
      answers: [...answers],
    });

    while (stack.length > 0 && resolves(MONOTONIC_STACK_VALUES[stack.at(-1)], value)) {
      const popped = stack.pop();
      answers[popped] = index;
      steps.push({
        action: 'resolve',
        current: index,
        popped,
        stack: [...stack],
        answers: [...answers],
      });
    }

    stack.push(index);
    steps.push({
      action: 'push',
      current: index,
      popped: null,
      stack: [...stack],
      answers: [...answers],
    });
  });

  steps.push({
    action: 'finish',
    current: null,
    popped: null,
    stack: [...stack],
    answers: [...answers],
  });
  return steps;
}

const MONOTONIC_STACK_STEPS = {
  greater: buildMonotonicStackSteps('greater'),
  smaller: buildMonotonicStackSteps('smaller'),
};

function MonotonicStackVisual() {
  const { isEnglish, t } = useUiCopy();
  const [mode, setMode] = useState('greater');
  const [activeStep, setActiveStep] = useState(0);
  const steps = MONOTONIC_STACK_STEPS[mode];
  const step = steps[activeStep];
  const isGreater = mode === 'greater';
  const directionWord = t(isGreater ? '更大' : '更小', isGreater ? 'greater' : 'smaller');
  const stackOrder = t(
    isGreater ? '栈底 → 栈顶：单调不增' : '栈底 → 栈顶：单调不减',
    isGreater ? 'bottom → top: non-increasing' : 'bottom → top: non-decreasing',
  );

  let title = t(
    '下标都在等待右侧第一个答案',
    'Every index is waiting for its first answer on the right',
  );
  let detail = t(
    `目标：找到右侧第一个严格${directionWord}的元素。栈里只放尚未得到答案的下标。`,
    `Goal: find the first strictly ${directionWord} value to the right. The stack stores only unresolved indices.`,
  );
  let activeLine = 'init';

  if (step.action === 'scan') {
    const value = MONOTONIC_STACK_VALUES[step.current];
    const top = step.stack.at(-1);
    title = isEnglish
      ? `Scan i = ${step.current}, value = ${value}`
      : `扫描 i = ${step.current}，当前值 = ${value}`;
    detail = top === undefined
      ? t('栈为空，当前下标还不能回答任何人。', 'The stack is empty, so the current index cannot resolve anyone yet.')
      : isEnglish
        ? `Compare ${value} with the waiting top value ${MONOTONIC_STACK_VALUES[top]}.`
        : `拿 ${value} 和栈顶等待中的 ${MONOTONIC_STACK_VALUES[top]} 比较。`;
    activeLine = 'loop';
  } else if (step.action === 'resolve') {
    const currentValue = MONOTONIC_STACK_VALUES[step.current];
    const poppedValue = MONOTONIC_STACK_VALUES[step.popped];
    const symbol = isGreater ? '>' : '<';
    title = isEnglish
      ? `${currentValue} ${symbol} ${poppedValue}: pop index ${step.popped}`
      : `${currentValue} ${symbol} ${poppedValue}，弹出下标 ${step.popped}`;
    detail = isEnglish
      ? `Index ${step.current} is the first qualifying value to the right of index ${step.popped}, so answer[${step.popped}] = ${step.current}.`
      : `下标 ${step.current} 是下标 ${step.popped} 右侧第一个满足条件的位置，因此 answer[${step.popped}] = ${step.current}。`;
    activeLine = 'resolve';
  } else if (step.action === 'push') {
    title = isEnglish
      ? `Push index ${step.current}`
      : `压入下标 ${step.current}`;
    detail = isEnglish
      ? `Its answer is still unknown. After the push, ${stackOrder}.`
      : `它自己的答案还不知道。入栈后，${stackOrder}。`;
    activeLine = 'push';
  } else if (step.action === 'finish') {
    title = t('扫描结束，栈中元素没有右侧答案', 'The scan is done; remaining indices have no answer to the right');
    detail = t(
      '这些下标从未被后来的元素弹出，答案保留为 -1。',
      'No later value ever popped these indices, so their answers remain -1.',
    );
    activeLine = 'finish';
  }

  const templateLines = [
    ['init', 'answer = [-1] * n', 'stack = []'],
    ['loop', 'for i, x in enumerate(nums):', t('    # x 尝试回答栈顶', '    # x tries to resolve the top')],
    [
      'resolve',
      isGreater
        ? '    while stack and nums[stack[-1]] < x:'
        : '    while stack and nums[stack[-1]] > x:',
      '        j = stack.pop(); answer[j] = i',
    ],
    ['push', '    stack.append(i)', t('    # i 开始等待自己的答案', '    # i starts waiting for its own answer')],
    ['finish', 'return answer', ''],
  ];
  const activeLineLabel = {
    init: t('初始化', 'Initialize'),
    loop: t('读取当前元素', 'Read current value'),
    resolve: t('弹栈并写答案', 'Pop and write answer'),
    push: t('压入未决下标', 'Push unresolved index'),
    finish: t('返回答案', 'Return answers'),
  }[activeLine];

  return (
    <section
      className="monotonic-stack-visual"
      aria-label={t('单调栈统一模板演示', 'Unified monotonic-stack template walkthrough')}
    >
      <header className="monotonic-stack-header">
        <div>
          <p className="eyebrow">{t('单调栈', 'Monotonic stack')}</p>
          <h2>{t('当前元素负责回答谁？', 'Whose question can the current value answer?')}</h2>
          <p>{t(
            '固定数组 [2, 1, 2, 4, 3]。切换目标时，只改变 while 的比较符号。',
            'The array is fixed at [2, 1, 2, 4, 3]. Switching the target changes only the while-loop comparator.',
          )}</p>
        </div>
        <div className="monotonic-stack-mode" role="group" aria-label={t('选择单调栈目标', 'Choose the monotonic-stack target')}>
          <button
            type="button"
            className={isGreater ? 'active' : ''}
            aria-pressed={isGreater}
            onClick={() => {
              setMode('greater');
              setActiveStep(0);
            }}
          >
            {t('找右侧更大', 'Next greater')}
          </button>
          <button
            type="button"
            className={!isGreater ? 'active' : ''}
            aria-pressed={!isGreater}
            onClick={() => {
              setMode('smaller');
              setActiveStep(0);
            }}
          >
            {t('找右侧更小', 'Next smaller')}
          </button>
        </div>
      </header>

      <div className={`monotonic-stack-step-copy ${step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="monotonic-stack-array" aria-label={t('输入数组与已确定答案', 'Input array and resolved answers')}>
        {MONOTONIC_STACK_VALUES.map((value, index) => {
          const isCurrent = index === step.current;
          const isPopped = index === step.popped;
          const answer = step.answers[index];
          const answerLabel = answer === null
            ? (step.action === 'finish' ? '-1' : t('等待', 'waiting'))
            : `→ ${answer}`;
          return (
            <div
              className={`monotonic-stack-cell${isCurrent ? ' current' : ''}${isPopped ? ' popped' : ''}${answer !== null ? ' resolved' : ''}`}
              key={index}
            >
              <small>i = {index}</small>
              <strong>{value}</strong>
              <span>{answerLabel}</span>
            </div>
          );
        })}
      </div>

      <div className="monotonic-stack-workspace">
        <div className="monotonic-stack-lane">
          <div className="monotonic-stack-lane-heading">
            <span>{t('未决下标栈', 'Unresolved index stack')}</span>
            <strong>{stackOrder}</strong>
          </div>
          <div className="monotonic-stack-items">
            <span className="monotonic-stack-bottom">{t('栈底', 'bottom')}</span>
            {step.stack.length === 0
              ? <em>{t('空栈', 'empty')}</em>
              : step.stack.map((index, position) => (
                <div
                  className={`monotonic-stack-item${position === step.stack.length - 1 ? ' top' : ''}`}
                  key={index}
                >
                  <small>i = {index}</small>
                  <strong>{MONOTONIC_STACK_VALUES[index]}</strong>
                  {position === step.stack.length - 1 && <span>{t('栈顶', 'top')}</span>}
                </div>
              ))}
          </div>
        </div>

        <div className="monotonic-stack-code" aria-label={t('当前模板代码', 'Active template code')}>
          <div className="monotonic-stack-code-heading">
            <span>{t('统一模板', 'Unified template')}</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>
          <div className="monotonic-stack-code-lines">
            {templateLines.map(([id, first, second]) => (
              <div
                className={activeLine === id ? 'active' : ''}
                aria-current={activeLine === id ? 'step' : undefined}
                key={id}
              >
                <code>{first}</code>
                {second && <code>{second}</code>}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="monotonic-stack-answer">
        <strong>answer</strong>
        {step.answers.map((answer, index) => (
          <span className={answer !== null ? 'resolved' : ''} key={index}>
            <small>{index}</small>
            {answer ?? (step.action === 'finish' ? '-1' : '−')}
          </span>
        ))}
        <em>{t('保存答案下标；题目也可能要求值或距离', 'Stores indices; a problem may instead ask for values or distances')}</em>
      </div>

      <div className="monotonic-stack-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择单调栈演示步骤', 'Select a monotonic-stack step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          disabled={activeStep === steps.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const BINARY_SEARCH_TEMPLATE_CODE_LINES = [
  { id: 'init', code: ['def find_first_true(lo, hi, check):', '    while lo < hi:'] },
  { id: 'mid', code: ['        mid = lo + (hi - lo) // 2'] },
  { id: 'true', code: ['        if check(mid):', '            hi = mid'] },
  { id: 'false', code: ['        else:', '            lo = mid + 1'] },
  { id: 'finish', code: ['    return lo'] },
];

const BINARY_SEARCH_TEMPLATE_NUMS = [1, 3, 5, 7, 9, 11, 13];
const BINARY_SEARCH_TEMPLATE_PILES = [3, 6, 7, 11];
const BINARY_SEARCH_TEMPLATE_HOURS_LIMIT = 8;
const BINARY_SEARCH_TEMPLATE_TIMESTAMPS = [1, 4, 7, 10];
const BINARY_SEARCH_TEMPLATE_QUERY = 8;

function buildBinarySearchTemplateSteps(lo, hi, evaluate, finish) {
  const steps = [{
    action: 'start',
    lo,
    hi,
    rangeLo: lo,
    rangeHi: hi,
    mid: null,
    checkResult: null,
    activeLine: 'init',
  }];

  let currentLo = lo;
  let currentHi = hi;
  let iteration = 1;

  while (currentLo < currentHi) {
    const mid = currentLo + Math.floor((currentHi - currentLo) / 2);
    const probe = evaluate(mid);
    const nextLo = probe.checkResult ? currentLo : mid + 1;
    const nextHi = probe.checkResult ? mid : currentHi;

    steps.push({
      action: 'scan',
      iteration,
      lo: currentLo,
      hi: currentHi,
      rangeLo: currentLo,
      rangeHi: currentHi,
      mid,
      nextLo,
      nextHi,
      activeLine: 'mid',
      ...probe,
    });

    steps.push({
      action: 'shrink',
      iteration,
      lo: currentLo,
      hi: currentHi,
      rangeLo: nextLo,
      rangeHi: nextHi,
      mid,
      nextLo,
      nextHi,
      activeLine: probe.checkResult ? 'true' : 'false',
      ...probe,
    });

    currentLo = nextLo;
    currentHi = nextHi;
    iteration += 1;
  }

  steps.push({
    action: 'finish',
    lo: currentLo,
    hi: currentHi,
    rangeLo: currentLo,
    rangeHi: currentHi,
    mid: null,
    activeLine: 'finish',
    ...finish(currentLo),
  });

  return steps;
}

function buildBinarySearchExactScenario(target) {
  const positions = BINARY_SEARCH_TEMPLATE_NUMS.map((value, index) => ({
    index,
    label: `i = ${index}`,
    displayValue: value,
    predicate: value >= target,
    sentinel: false,
  }));
  positions.push({
    index: BINARY_SEARCH_TEMPLATE_NUMS.length,
    label: `i = ${BINARY_SEARCH_TEMPLATE_NUMS.length}`,
    displayValue: '∅',
    predicate: null,
    sentinel: true,
  });

  return {
    mode: 'exact',
    target,
    positions,
    initialLo: 0,
    initialHi: BINARY_SEARCH_TEMPLATE_NUMS.length,
    checkRule: `nums[mid] >= ${target}`,
    steps: buildBinarySearchTemplateSteps(
      0,
      BINARY_SEARCH_TEMPLATE_NUMS.length,
      (mid) => {
        const value = BINARY_SEARCH_TEMPLATE_NUMS[mid];
        return {
          probeValue: value,
          checkResult: value >= target,
        };
      },
      (boundary) => {
        const inBounds = boundary < BINARY_SEARCH_TEMPLATE_NUMS.length;
        const boundaryValue = inBounds ? BINARY_SEARCH_TEMPLATE_NUMS[boundary] : null;
        const found = inBounds && boundaryValue === target;
        return {
          boundary,
          boundaryValue,
          found,
          answerIndex: found ? boundary : null,
          returnValue: found ? boundary : -1,
        };
      },
    ),
  };
}

function buildBinarySearchRangeScenario() {
  const maxSpeed = Math.max(...BINARY_SEARCH_TEMPLATE_PILES);
  const positions = [];
  for (let speed = 1; speed <= maxSpeed; speed += 1) {
    const hours = BINARY_SEARCH_TEMPLATE_PILES.reduce((sum, pile) => sum + Math.ceil(pile / speed), 0);
    positions.push({
      index: speed,
      label: `speed = ${speed}`,
      displayValue: speed,
      predicate: hours <= BINARY_SEARCH_TEMPLATE_HOURS_LIMIT,
      auxiliary: `${hours}h`,
      sentinel: false,
    });
  }

  return {
    mode: 'range',
    piles: BINARY_SEARCH_TEMPLATE_PILES,
    hoursLimit: BINARY_SEARCH_TEMPLATE_HOURS_LIMIT,
    positions,
    initialLo: 1,
    initialHi: maxSpeed,
    checkRule: `hours_needed(speed) <= ${BINARY_SEARCH_TEMPLATE_HOURS_LIMIT}`,
    steps: buildBinarySearchTemplateSteps(
      1,
      maxSpeed,
      (speed) => {
        const parts = BINARY_SEARCH_TEMPLATE_PILES.map((pile) => Math.ceil(pile / speed));
        const hours = parts.reduce((sum, value) => sum + value, 0);
        return {
          probeValue: speed,
          checkResult: hours <= BINARY_SEARCH_TEMPLATE_HOURS_LIMIT,
          hours,
          hoursBreakdown: parts.join(' + '),
        };
      },
      (boundary) => ({
        boundary,
        answerIndex: boundary,
        returnValue: boundary,
      }),
    ),
  };
}

function buildBinarySearchLastFalseScenario() {
  const positions = BINARY_SEARCH_TEMPLATE_TIMESTAMPS.map((value, index) => ({
    index,
    label: `i = ${index}`,
    displayValue: value,
    predicate: value > BINARY_SEARCH_TEMPLATE_QUERY,
    sentinel: false,
  }));
  positions.push({
    index: BINARY_SEARCH_TEMPLATE_TIMESTAMPS.length,
    label: `i = ${BINARY_SEARCH_TEMPLATE_TIMESTAMPS.length}`,
    displayValue: '∅',
    predicate: null,
    sentinel: true,
  });

  return {
    mode: 'lastFalse',
    query: BINARY_SEARCH_TEMPLATE_QUERY,
    positions,
    initialLo: 0,
    initialHi: BINARY_SEARCH_TEMPLATE_TIMESTAMPS.length,
    checkRule: `timestamps[mid] > ${BINARY_SEARCH_TEMPLATE_QUERY}`,
    steps: buildBinarySearchTemplateSteps(
      0,
      BINARY_SEARCH_TEMPLATE_TIMESTAMPS.length,
      (mid) => {
        const value = BINARY_SEARCH_TEMPLATE_TIMESTAMPS[mid];
        return {
          probeValue: value,
          checkResult: value > BINARY_SEARCH_TEMPLATE_QUERY,
        };
      },
      (boundary) => {
        const answerIndex = boundary > 0 ? boundary - 1 : null;
        return {
          boundary,
          boundaryValue: boundary < BINARY_SEARCH_TEMPLATE_TIMESTAMPS.length
            ? BINARY_SEARCH_TEMPLATE_TIMESTAMPS[boundary]
            : null,
          answerIndex,
          answerValue: answerIndex === null ? null : BINARY_SEARCH_TEMPLATE_TIMESTAMPS[answerIndex],
          returnValue: answerIndex,
        };
      },
    ),
  };
}

const BINARY_SEARCH_TEMPLATE_SCENARIOS = {
  exact: {
    found: buildBinarySearchExactScenario(9),
    missing: buildBinarySearchExactScenario(6),
  },
  range: buildBinarySearchRangeScenario(),
  lastFalse: buildBinarySearchLastFalseScenario(),
};

function BinarySearchTemplateVisual() {
  const { isEnglish, t } = useUiCopy();
  const [mode, setMode] = useState('exact');
  const [exactExample, setExactExample] = useState('found');
  const [activeStep, setActiveStep] = useState(0);
  const scenario = mode === 'exact'
    ? BINARY_SEARCH_TEMPLATE_SCENARIOS.exact[exactExample]
    : BINARY_SEARCH_TEMPLATE_SCENARIOS[mode];
  const step = scenario.steps[activeStep];

  const activeLineLabel = {
    init: t('初始化', 'Initialize'),
    mid: t('计算 mid', 'Compute mid'),
    true: t('True 分支：hi = mid', 'True branch: hi = mid'),
    false: t('False 分支：lo = mid + 1', 'False branch: lo = mid + 1'),
    finish: t('返回边界', 'Return boundary'),
  }[step.activeLine];

  let title = '';
  let detail = '';

  if (step.action === 'start') {
    if (mode === 'exact') {
      title = t('初始化：边界在 [0, 7] 内', 'Initialize: the boundary lies in [0, 7]');
      detail = t(
        `check(mid) = nums[mid] >= ${scenario.target}。模板返回边界 b，之后再验证 nums[b] == target。`,
        `check(mid) = nums[mid] >= ${scenario.target}. The template returns boundary b, then verifies nums[b] == target.`,
      );
    } else if (mode === 'range') {
      title = t('初始化：速度边界在 [1, 11] 内', 'Initialize: the speed boundary lies in [1, 11]');
      detail = t(
        'check(speed) = hours_needed(speed) <= 8。hi = 11 是已知 True 的边界。',
        'check(speed) = hours_needed(speed) <= 8. hi = 11 is a known True boundary.',
      );
    } else {
      title = t('初始化：边界在 [0, 4] 内', 'Initialize: the boundary lies in [0, 4]');
      detail = t(
        'check(mid) = timestamps[mid] > 8。模板返回第一个 True，下标答案是 b - 1。',
        'check(mid) = timestamps[mid] > 8. The template returns the first True; the answer index is b - 1.',
      );
    }
  } else if (step.action === 'scan') {
    title = isEnglish
      ? `Iteration ${step.iteration}: mid = ${step.mid}`
      : `第 ${step.iteration} 轮：mid = ${step.mid}`;
    if (mode === 'exact') {
      detail = isEnglish
        ? `lo = ${step.lo}, hi = ${step.hi}, nums[${step.mid}] = ${step.probeValue}.`
        : `lo = ${step.lo}，hi = ${step.hi}，nums[${step.mid}] = ${step.probeValue}。`;
    } else if (mode === 'range') {
      detail = isEnglish
        ? `lo = ${step.lo}, hi = ${step.hi}, speed = ${step.mid}, hours_needed(${step.mid}) = ${step.hours}.`
        : `lo = ${step.lo}，hi = ${step.hi}，speed = ${step.mid}，hours_needed(${step.mid}) = ${step.hours}。`;
    } else {
      detail = isEnglish
        ? `lo = ${step.lo}, hi = ${step.hi}, timestamps[${step.mid}] = ${step.probeValue}.`
        : `lo = ${step.lo}，hi = ${step.hi}，timestamps[${step.mid}] = ${step.probeValue}。`;
    }
  } else if (step.action === 'shrink') {
    if (step.checkResult) {
      title = t(`check(${step.mid}) = True，执行 hi = mid`, `check(${step.mid}) = True, execute hi = mid`);
      detail = isEnglish
        ? `The candidate interval shrinks to [${step.nextLo}, ${step.nextHi}].`
        : `候选区间收缩到 [${step.nextLo}, ${step.nextHi}]。`;
    } else {
      title = t(`check(${step.mid}) = False，执行 lo = mid + 1`, `check(${step.mid}) = False, execute lo = mid + 1`);
      detail = isEnglish
        ? `The candidate interval shrinks to [${step.nextLo}, ${step.nextHi}].`
        : `候选区间收缩到 [${step.nextLo}, ${step.nextHi}]。`;
    }
  } else if (mode === 'exact') {
    title = isEnglish
      ? `Finish: boundary b = ${step.boundary}`
      : `结束：边界 b = ${step.boundary}`;
    detail = step.found
      ? t(
        `nums[${step.boundary}] = ${step.boundaryValue}，验证通过，返回下标 ${step.returnValue}。`,
        `nums[${step.boundary}] = ${step.boundaryValue}; verification passes, so return index ${step.returnValue}.`,
      )
      : t(
        `nums[${step.boundary}] = ${step.boundaryValue}，验证失败，返回 -1。`,
        `nums[${step.boundary}] = ${step.boundaryValue}; verification fails, so return -1.`,
      );
  } else if (mode === 'range') {
    title = t(`结束：边界 b = ${step.boundary}`, `Finish: boundary b = ${step.boundary}`);
    detail = t(
      `最小可行速度是 ${step.returnValue}。`,
      `The minimum feasible speed is ${step.returnValue}.`,
    );
  } else {
    title = t(`结束：边界 b = ${step.boundary}`, `Finish: boundary b = ${step.boundary}`);
    detail = t(
      `答案下标 = b - 1 = ${step.answerIndex}，时间戳 = ${step.answerValue}。`,
      `The answer index is b - 1 = ${step.answerIndex}, whose timestamp is ${step.answerValue}.`,
    );
  }

  const summaryCards = step.action === 'start'
    ? [
      { label: 'lo', value: scenario.initialLo },
      { label: 'hi', value: scenario.initialHi },
      { label: t('谓词', 'Predicate'), value: scenario.checkRule },
      {
        label: t('哨兵', 'Sentinel'),
        value: mode === 'range'
          ? t('hi 已知为 True', 'hi is known True')
          : t('hi 是越界一位', 'hi is one past the end'),
      },
      {
        label: t('边界之后', 'After boundary'),
        value: mode === 'exact'
          ? t('验证相等', 'verify equality')
          : mode === 'range'
            ? t('直接返回 b', 'return b directly')
            : t('返回 b - 1', 'return b - 1'),
      },
    ]
    : step.action === 'finish'
      ? [
        { label: 'b', value: step.boundary },
        { label: t('模板返回', 'Template return'), value: step.boundary },
        {
          label: t('后处理', 'Post-process'),
          value: mode === 'exact'
            ? t(`nums[b] == ${scenario.target}`, `nums[b] == ${scenario.target}`)
            : mode === 'range'
              ? t('直接使用 b', 'use b directly')
              : 'b - 1',
        },
        {
          label: t('最终答案', 'Final answer'),
          value: mode === 'lastFalse' ? step.answerIndex : step.returnValue,
        },
        {
          label: t('状态', 'Status'),
          value: mode === 'exact'
            ? (step.found ? t('找到', 'found') : t('未找到', 'not found'))
            : t('完成', 'done'),
        },
      ]
      : [
        { label: 'lo', value: step.lo },
        { label: 'hi', value: step.hi },
        { label: 'mid', value: step.mid },
        { label: 'check(mid)', value: step.checkResult ? 'True' : 'False' },
        { label: t('下一步', 'Next'), value: step.checkResult ? 'hi = mid' : 'lo = mid + 1' },
      ];

  const currentRangeLabel = mode === 'range'
    ? t(`当前候选速度 [${step.rangeLo}, ${step.rangeHi}]`, `Current candidate speeds [${step.rangeLo}, ${step.rangeHi}]`)
    : t(`当前候选边界 [${step.rangeLo}, ${step.rangeHi}]`, `Current candidate boundaries [${step.rangeLo}, ${step.rangeHi}]`);

  const formulaItems = [];
  let formulaNote = '';

  if (step.action !== 'start' && step.action !== 'finish') {
    if (mode === 'exact') {
      formulaItems.push(
        { label: 'nums[mid]', value: `nums[${step.mid}] = ${step.probeValue}` },
        { label: 'target', value: scenario.target },
        { label: 'check(mid)', value: `${step.probeValue} ${step.checkResult ? '≥' : '<'} ${scenario.target} → ${step.checkResult ? 'True' : 'False'}` },
      );
    } else if (mode === 'range') {
      formulaItems.push(
        { label: 'speed', value: step.mid },
        { label: 'hours_needed(speed)', value: `${step.hoursBreakdown} = ${step.hours}` },
        { label: 'check(mid)', value: `${step.hours} ${step.checkResult ? '≤' : '>'} ${scenario.hoursLimit} → ${step.checkResult ? 'True' : 'False'}` },
      );
    } else {
      formulaItems.push(
        { label: 'timestamps[mid]', value: `timestamps[${step.mid}] = ${step.probeValue}` },
        { label: 'query', value: scenario.query },
        { label: 'check(mid)', value: `${step.probeValue} ${step.checkResult ? '>' : '≤'} ${scenario.query} → ${step.checkResult ? 'True' : 'False'}` },
      );
    }
    formulaNote = step.checkResult
      ? t('下一步进入 True 分支。', 'The next step enters the True branch.')
      : t('下一步进入 False 分支。', 'The next step enters the False branch.');
  } else if (step.action === 'finish') {
    if (mode === 'exact') {
      formulaItems.push(
        { label: 'b', value: step.boundary },
        { label: 'nums[b]', value: `nums[${step.boundary}] = ${step.boundaryValue}` },
        { label: 'verify', value: `${step.boundaryValue} ${step.found ? '=' : '≠'} ${scenario.target}` },
        { label: 'return', value: step.returnValue },
      );
    } else if (mode === 'range') {
      formulaItems.push(
        { label: 'b', value: step.boundary },
        { label: 'hours_needed(b)', value: `${BINARY_SEARCH_TEMPLATE_PILES.map((pile) => Math.ceil(pile / step.boundary)).join(' + ')} = ${BINARY_SEARCH_TEMPLATE_PILES.reduce((sum, pile) => sum + Math.ceil(pile / step.boundary), 0)}` },
        { label: 'answer', value: step.returnValue },
      );
    } else {
      formulaItems.push(
        { label: 'b', value: step.boundary },
        { label: 'b - 1', value: step.answerIndex },
        { label: 'timestamps[b - 1]', value: step.answerValue },
        { label: 'return', value: step.answerIndex },
      );
    }
    formulaNote = mode === 'exact'
      ? (step.found ? t('边界位置同时是目标位置。', 'The boundary position is also the target position.') : t('边界存在，但目标值不存在。', 'The boundary exists, but the target value does not.'))
      : mode === 'range'
        ? t('边界本身就是答案。', 'The boundary itself is the answer.')
        : t('模板返回的是边界，题目答案在前一格。', 'The template returns the boundary; the problem answer is one position earlier.');
  } else {
    formulaItems.push(
      { label: 'lo', value: scenario.initialLo },
      { label: 'hi', value: scenario.initialHi },
      { label: 'check(mid)', value: scenario.checkRule },
    );
    formulaNote = mode === 'exact'
      ? t('精确匹配还需要边界之后的相等验证。', 'Exact match still needs an equality check after the boundary is found.')
      : mode === 'range'
        ? t('这次搜索的对象是速度，不是数组下标。', 'This search runs over speeds, not array indices.')
        : t('这次读取的是最后一个 False。', 'This reading asks for the last False.');
  }

  return (
    <section
      className="binary-search-template-visual"
      aria-label={t('二分查找统一模板演示', 'Unified binary-search template walkthrough')}
    >
      <header className="binary-search-template-header">
        <div>
          <p className="eyebrow">{t('二分查找', 'Binary search')}</p>
          <h2>{t('同一个 find_first_true，只替换 check(mid)', 'One find_first_true; only check(mid) changes')}</h2>
          <p>{mode === 'exact'
            ? t(
              `nums = [${BINARY_SEARCH_TEMPLATE_NUMS.join(', ')}]，target = ${scenario.target}。`,
              `nums = [${BINARY_SEARCH_TEMPLATE_NUMS.join(', ')}], target = ${scenario.target}.`,
            )
            : mode === 'range'
              ? t(
                `piles = [${BINARY_SEARCH_TEMPLATE_PILES.join(', ')}]，h = ${scenario.hoursLimit}。`,
                `piles = [${BINARY_SEARCH_TEMPLATE_PILES.join(', ')}], h = ${scenario.hoursLimit}.`,
              )
              : t(
                `timestamps = [${BINARY_SEARCH_TEMPLATE_TIMESTAMPS.join(', ')}]，query = ${scenario.query}。`,
                `timestamps = [${BINARY_SEARCH_TEMPLATE_TIMESTAMPS.join(', ')}], query = ${scenario.query}.`,
              )}</p>
        </div>
        <div className="binary-search-template-mode" role="group" aria-label={t('选择二分查找模式', 'Choose the binary-search mode')}>
          <button
            type="button"
            className={mode === 'exact' ? 'active' : ''}
            aria-pressed={mode === 'exact'}
            onClick={() => {
              setMode('exact');
              setActiveStep(0);
            }}
          >
            {t('精确匹配', 'Exact match')}
          </button>
          <button
            type="button"
            className={mode === 'range' ? 'active' : ''}
            aria-pressed={mode === 'range'}
            onClick={() => {
              setMode('range');
              setActiveStep(0);
            }}
          >
            {t('答案值域', 'Answer range')}
          </button>
          <button
            type="button"
            className={mode === 'lastFalse' ? 'active' : ''}
            aria-pressed={mode === 'lastFalse'}
            onClick={() => {
              setMode('lastFalse');
              setActiveStep(0);
            }}
          >
            {t('最后一个 False', 'Last False')}
          </button>
        </div>
      </header>

      {mode === 'exact' && (
        <div className="binary-search-template-example" role="group" aria-label={t('选择精确匹配示例', 'Choose the exact-match example')}>
          <button
            type="button"
            className={exactExample === 'found' ? 'active' : ''}
            aria-pressed={exactExample === 'found'}
            onClick={() => {
              setExactExample('found');
              setActiveStep(0);
            }}
          >
            {t('target = 9（找到）', 'target = 9 (found)')}
          </button>
          <button
            type="button"
            className={exactExample === 'missing' ? 'active' : ''}
            aria-pressed={exactExample === 'missing'}
            onClick={() => {
              setExactExample('missing');
              setActiveStep(0);
            }}
          >
            {t('target = 6（未找到）', 'target = 6 (not found)')}
          </button>
        </div>
      )}

      <div className={`binary-search-template-step-copy ${step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {scenario.steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="binary-search-template-stats" aria-label={t('当前状态', 'Current state')}>
        {summaryCards.map((item) => (
          <div className="binary-search-template-stat" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>

      <div className="binary-search-template-workspace">
        <div className="binary-search-template-search">
          <div className="binary-search-template-search-heading">
            <span>{t('搜索空间', 'Search space')}</span>
            <strong>{currentRangeLabel}</strong>
          </div>
          <p className="binary-search-template-rule">
            <code>{scenario.checkRule}</code>
          </p>
          <div className={`binary-search-template-track${mode === 'range' ? ' range' : ''}`}>
            {scenario.positions.map((point) => {
              const inRange = point.index >= step.rangeLo && point.index <= step.rangeHi;
              const isCurrent = point.index === step.mid;
              const isBoundary = step.action === 'finish' && point.index === step.boundary;
              const isAnswer = step.action === 'finish' && step.answerIndex === point.index;
              const pointLabel = mode === 'range'
                ? `${t('速度', 'speed')} = ${point.index}`
                : `i = ${point.index}`;
              const flags = [];

              if (step.action === 'finish') {
                if (isBoundary && isAnswer) flags.push(t('b = 答案', 'b = answer'));
                else if (isBoundary) flags.push('b');
                else if (isAnswer) flags.push(t('答案', 'answer'));
              } else {
                if (step.rangeLo === step.rangeHi && point.index === step.rangeLo) {
                  flags.push('lo = hi');
                } else {
                  if (point.index === step.rangeLo) flags.push('lo');
                  if (point.index === step.rangeHi) flags.push('hi');
                }
                if (isCurrent) flags.push('mid');
              }

              return (
                <div
                  className={[
                    'binary-search-template-point',
                    inRange ? 'candidate' : 'trimmed',
                    isCurrent ? 'current' : '',
                    isBoundary ? 'boundary' : '',
                    isAnswer ? 'answer' : '',
                    point.sentinel ? 'sentinel' : '',
                  ].filter(Boolean).join(' ')}
                  key={point.index}
                >
                  <small>{pointLabel}</small>
                  <strong>{point.displayValue}</strong>
                  <div className="binary-search-template-meta">
                    {point.auxiliary && <span>{point.auxiliary}</span>}
                    {point.predicate !== null && (
                      <em className={point.predicate ? 'true' : 'false'}>{point.predicate ? 'T' : 'F'}</em>
                    )}
                    {point.sentinel && <span>{t('哨兵', 'sentinel')}</span>}
                  </div>
                  {flags.length > 0 && (
                    <div className="binary-search-template-flags">
                      {flags.map((flag) => (
                        <i key={flag}>{flag}</i>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        <div className="binary-search-template-code" aria-label={t('统一模板代码', 'Unified template code')}>
          <div className="binary-search-template-code-heading">
            <span>find_first_true(lo, hi, check)</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>
          <div className="binary-search-template-code-lines">
            {BINARY_SEARCH_TEMPLATE_CODE_LINES.map((block) => (
              <div
                className={step.activeLine === block.id ? 'active' : ''}
                aria-current={step.activeLine === block.id ? 'step' : undefined}
                key={block.id}
              >
                {block.code.map((line, lineIndex) => (
                  <code key={lineIndex}>{line}</code>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className={`binary-search-template-formula${step.action === 'finish' ? ' finish' : ''}`}>
        {formulaItems.map((item) => (
          <div className="binary-search-template-formula-card" key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </div>
        ))}
        <em>{formulaNote}</em>
      </div>

      <div className="binary-search-template-legend">
        <span><i className="candidate" />{t('候选区间', 'candidate range')}</span>
        <span><i className="current" />{t('当前 mid', 'current mid')}</span>
        <span><i className="boundary" />{t('返回边界 b', 'returned boundary b')}</span>
        <span><i className="answer" />{t('题目答案', 'problem answer')}</span>
      </div>

      <div className="binary-search-template-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={scenario.steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择二分查找演示步骤', 'Select a binary-search step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(scenario.steps.length - 1, current + 1))}
          disabled={activeStep === scenario.steps.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const LARGEST_RECTANGLE_HEIGHTS = [2, 1, 5, 6, 2, 3];
const LARGEST_RECTANGLE_DISPLAY_HEIGHTS = [...LARGEST_RECTANGLE_HEIGHTS, 0];

function buildLargestRectangleSteps(heights) {
  const stack = [];
  const steps = [];
  let answer = 0;
  let best = null;

  for (let right = 0; right <= heights.length; right += 1) {
    const isSentinel = right === heights.length;
    const currentHeight = isSentinel ? 0 : heights[right];

    steps.push({
      action: 'scan',
      right,
      isSentinel,
      currentHeight,
      stack: [...stack],
      popped: null,
      answer,
      updated: false,
    });

    while (stack.length > 0 && heights[stack[stack.length - 1]] > currentHeight) {
      const j = stack.pop();
      const left = stack.length > 0 ? stack[stack.length - 1] : -1;
      const width = right - left - 1;
      const area = heights[j] * width;
      const updated = area > answer;
      if (updated) {
        answer = area;
        best = { j, height: heights[j], left, right, width, area };
      }

      steps.push({
        action: 'pop',
        right,
        isSentinel,
        currentHeight,
        stack: [...stack],
        popped: j,
        height: heights[j],
        left,
        width,
        area,
        answer,
        updated,
      });
    }

    if (isSentinel) {
      steps.push({
        action: 'finish',
        right,
        isSentinel,
        currentHeight,
        stack: [...stack],
        popped: null,
        answer,
        best,
      });
    } else {
      stack.push(right);
      steps.push({
        action: 'push',
        right,
        isSentinel,
        currentHeight,
        stack: [...stack],
        popped: null,
        answer,
        updated: false,
      });
    }
  }

  return steps;
}

const LARGEST_RECTANGLE_STEPS = buildLargestRectangleSteps(LARGEST_RECTANGLE_HEIGHTS);

const LARGEST_RECTANGLE_CODE_LINES = [
  { id: 'init', code: ['answer = 0', 'stack = []'] },
  {
    id: 'loop',
    code: [
      'for right in range(len(heights) + 1):',
      '    current = 0 if right == len(heights) else heights[right]',
    ],
  },
  {
    id: 'resolve',
    code: [
      '    while stack and heights[stack[-1]] > current:',
      '        j = stack.pop()',
      '        left = stack[-1] if stack else -1',
      '        width = right - left - 1',
      '        answer = max(answer, heights[j] * width)',
    ],
  },
  { id: 'push', code: ['    stack.append(right)'] },
  { id: 'finish', code: ['return answer'] },
];

function LargestRectangleVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = LARGEST_RECTANGLE_STEPS;
  const step = steps[activeStep];
  const heights = LARGEST_RECTANGLE_HEIGHTS;
  const displayHeights = LARGEST_RECTANGLE_DISPLAY_HEIGHTS;
  const maxHeight = Math.max(...displayHeights);

  const stackOrder = t('栈底 → 栈顶：高度单调不减', 'bottom → top: heights are non-decreasing');

  let activeLine = 'init';
  if (step.action === 'scan') activeLine = 'loop';
  else if (step.action === 'pop') activeLine = 'resolve';
  else if (step.action === 'push') activeLine = 'push';
  else if (step.action === 'finish') activeLine = 'finish';

  let title = '';
  let detail = '';

  if (step.action === 'scan') {
    const top = step.stack.length ? step.stack[step.stack.length - 1] : undefined;
    title = step.isSentinel
      ? t(`扫描哨兵，right = ${step.right}，高度 = 0`, `Scan sentinel, right = ${step.right}, height = 0`)
      : t(`扫描 right = ${step.right}，高度 = ${step.currentHeight}`, `Scan right = ${step.right}, height = ${step.currentHeight}`);
    if (top === undefined) {
      detail = t('栈为空，本步不发生弹出。', 'The stack is empty; no pop happens this step.');
    } else if (heights[top] > step.currentHeight) {
      detail = t(
        `栈顶下标 ${top} 高度 ${heights[top]} > ${step.currentHeight}，将触发弹出。`,
        `Top index ${top} has height ${heights[top]} > ${step.currentHeight}; a pop follows.`,
      );
    } else {
      detail = t(
        `栈顶下标 ${top} 高度 ${heights[top]} ≤ ${step.currentHeight}，不发生弹出。`,
        `Top index ${top} has height ${heights[top]} ≤ ${step.currentHeight}; no pop happens.`,
      );
    }
  } else if (step.action === 'pop') {
    title = t(`弹出下标 ${step.popped}`, `Pop index ${step.popped}`);
    detail = t(
      `left = ${step.left}，right = ${step.right}，width = ${step.width}，area = ${step.area}。`,
      `left = ${step.left}, right = ${step.right}, width = ${step.width}, area = ${step.area}.`,
    );
  } else if (step.action === 'push') {
    title = t(`压入下标 ${step.right}`, `Push index ${step.right}`);
    detail = t(`入栈后，${stackOrder}。`, `After the push, ${stackOrder}.`);
  } else {
    title = t('哨兵处理完毕，栈已清空', 'The sentinel is processed; the stack is empty');
    detail = t(
      '哨兵下标本身会照代码入栈，但循环随即结束，不会再被读取。',
      'Per the code, the sentinel index would still be pushed, but the loop ends immediately, so it is never read.',
    );
  }

  const activeLineLabel = {
    init: t('初始化', 'Initialize'),
    loop: t('读取当前高度', 'Read current height'),
    resolve: t('弹栈并结算矩形', 'Pop and settle a rectangle'),
    push: t('压入待定下标', 'Push a pending index'),
    finish: t('返回答案', 'Return the answer'),
  }[activeLine];

  return (
    <section
      className="largest-rectangle-visual"
      aria-label={t('柱状图最大矩形演示', 'Largest Rectangle in Histogram walkthrough')}
    >
      <header className="largest-rectangle-header">
        <div>
          <p className="eyebrow">{t('哨兵单调栈', 'Sentinel monotonic stack')}</p>
          <h2>{t('弹出下标时结算它的矩形', 'Settle a rectangle whenever an index is popped')}</h2>
          <p>{t(
            `固定数组 heights = [${heights.join(', ')}]，末尾补一个高度 0 的哨兵。`,
            `Fixed array heights = [${heights.join(', ')}], with a trailing sentinel of height 0.`,
          )}</p>
        </div>
        <div className="largest-rectangle-total">
          <span>{t('当前最优面积', 'Best area so far')}</span>
          <strong className={step.action === 'pop' && step.updated ? 'updated' : ''}>{step.answer}</strong>
        </div>
      </header>

      <div className={`largest-rectangle-step-copy ${step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="largest-rectangle-chart-wrap">
        <div className="largest-rectangle-chart" aria-label={t('柱状图与被测量的矩形', 'Bar heights and the rectangle under measurement')}>
          {displayHeights.map((height, index) => {
            const isPoppedNow = step.action === 'pop' && index === step.popped;
            const isInStack = step.stack.includes(index);
            const isStackTop = isInStack && index === step.stack[step.stack.length - 1];
            const isCurrent = index === step.right
              && (step.action === 'scan' || step.action === 'pop' || step.action === 'push' || step.action === 'finish');
            const isFuture = index > step.right;

            let state = 'resolved';
            if (isPoppedNow) state = 'popped-now';
            else if (isStackTop) state = 'stack-top';
            else if (isInStack) state = 'in-stack';
            else if (isCurrent) state = 'current';
            else if (isFuture) state = 'future';

            const isMeasuring = step.action === 'pop' && index >= step.left + 1 && index <= step.right - 1;
            const isBest = step.action === 'finish' && step.best
              && index >= step.best.left + 1 && index <= step.best.right - 1;
            const rectHeight = isMeasuring ? step.height : (isBest ? step.best.height : 0);
            const isSentinelBar = index === heights.length;

            return (
              <div className={`largest-rectangle-column ${state}`} key={index}>
                <div className="largest-rectangle-track">
                  {(isMeasuring || isBest) && (
                    <div
                      className={`largest-rectangle-rect-fill ${isBest ? 'best' : 'measuring'}`}
                      style={{ height: `${(rectHeight / maxHeight) * 100}%` }}
                    />
                  )}
                  <div
                    className={`largest-rectangle-bar${isSentinelBar ? ' sentinel' : ''}`}
                    style={{ height: `${(height / maxHeight) * 100}%` }}
                  >
                    <span>{height}</span>
                  </div>
                </div>
                <small>{isSentinelBar ? t('哨兵', 'sentinel') : index}</small>
              </div>
            );
          })}
        </div>
      </div>

      {step.action === 'pop' && (
        <div className="largest-rectangle-formula">
          <div>
            <span>left</span>
            <strong>stack[-1] if stack else -1 = {step.left}</strong>
          </div>
          <div>
            <span>width</span>
            <strong>{step.right} − {step.left} − 1 = {step.width}</strong>
          </div>
          <div className={step.updated ? 'result updated' : 'result'}>
            <span>area</span>
            <strong>{step.height} × {step.width} = {step.area}</strong>
          </div>
          <em>{step.updated
            ? t(`answer 更新为 ${step.answer}`, `answer updates to ${step.answer}`)
            : t(`area ≤ answer(${step.answer})，不更新`, `area ≤ answer (${step.answer}); no update`)}</em>
        </div>
      )}

      {step.action === 'finish' && step.best && (
        <div className="largest-rectangle-formula finish">
          <div>
            <span>{t('最优矩形', 'Best rectangle')}</span>
            <strong>j = {step.best.j}, height = {step.best.height}</strong>
          </div>
          <div>
            <span>{t('覆盖区间', 'Covered span')}</span>
            <strong>[{step.best.left + 1}, {step.best.right - 1}] ({t('宽度', 'width')} {step.best.width})</strong>
          </div>
          <div className="result updated">
            <span>answer</span>
            <strong>{step.best.height} × {step.best.width} = {step.answer}</strong>
          </div>
        </div>
      )}

      <div className="largest-rectangle-workspace">
        <div className="largest-rectangle-lane">
          <div className="largest-rectangle-lane-heading">
            <span>{t('待定下标栈', 'Pending index stack')}</span>
            <strong>{stackOrder}</strong>
          </div>
          <div className="largest-rectangle-items">
            <span className="largest-rectangle-bottom">{t('栈底', 'bottom')}</span>
            {step.stack.length === 0
              ? <em>{t('空栈', 'empty')}</em>
              : step.stack.map((index, position) => (
                <div
                  className={`largest-rectangle-item${position === step.stack.length - 1 ? ' top' : ''}`}
                  key={index}
                >
                  <small>i = {index}</small>
                  <strong>{displayHeights[index]}</strong>
                  {position === step.stack.length - 1 && <span>{t('栈顶', 'top')}</span>}
                </div>
              ))}
          </div>
        </div>

        <div className="largest-rectangle-code" aria-label={t('参考代码', 'Reference code')}>
          <div className="largest-rectangle-code-heading">
            <span>{t('参考代码', 'Reference code')}</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>
          <div className="largest-rectangle-code-lines">
            {LARGEST_RECTANGLE_CODE_LINES.map((block) => (
              <div
                className={activeLine === block.id ? 'active' : ''}
                aria-current={activeLine === block.id ? 'step' : undefined}
                key={block.id}
              >
                {block.code.map((line, lineIndex) => (
                  <code key={lineIndex}>{line || ' '}</code>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="largest-rectangle-legend">
        <span><i className="current" />{t('当前扫描', 'scanning')}</span>
        <span><i className="in-stack" />{t('栈中待定', 'in stack')}</span>
        <span><i className="popped-now" />{t('本步弹出', 'popped now')}</span>
        <span><i className="resolved" />{t('已结算', 'resolved')}</span>
        <span><i className="rect" />{t('测量矩形', 'measured rectangle')}</span>
      </div>

      <div className="largest-rectangle-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择演示步骤', 'Select a step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          disabled={activeStep === steps.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const LINKED_LIST_REVERSAL_NODES = [
  { id: 0, value: 1, x: 90, y: 142 },
  { id: 1, value: 2, x: 220, y: 142 },
  { id: 2, value: 3, x: 350, y: 142 },
  { id: 3, value: 4, x: 480, y: 142 },
  { id: 4, value: 5, x: 610, y: 142 },
];

const FAST_SLOW_CYCLE_NODES = [
  { id: 0, value: 3, x: 110, y: 134 },
  { id: 1, value: 2, x: 270, y: 134 },
  { id: 2, value: 0, x: 430, y: 134 },
  { id: 3, value: -4, x: 590, y: 134 },
];

const LINKED_LIST_REVERSAL_CODE_LINES = [
  { id: 'init', code: ['prev = None', 'curr = head'] },
  { id: 'loop', code: ['while curr:'] },
  { id: 'save', code: ['    next_node = curr.next'] },
  { id: 'rewire', code: ['    curr.next = prev'] },
  { id: 'advance', code: ['    prev = curr', '    curr = next_node'] },
  { id: 'finish', code: ['return prev'] },
];

const FAST_SLOW_POINTER_CODE_LINES = {
  middle: [
    { id: 'init', code: ['slow = head', 'fast = head'] },
    { id: 'move', code: ['while fast and fast.next:', '    slow = slow.next', '    fast = fast.next.next'] },
    { id: 'finish', code: ['return slow'] },
  ],
  cycle: [
    { id: 'init', code: ['slow = head', 'fast = head'] },
    { id: 'detect', code: ['while True:', '    slow = slow.next', '    fast = fast.next.next', '    if slow == fast: break'] },
    { id: 'reset', code: ['finder = head'] },
    { id: 'locate', code: ['while finder != slow:', '    finder = finder.next', '    slow = slow.next'] },
    { id: 'finish', code: ['return finder'] },
  ],
  gap: [
    { id: 'init', code: ['lead = head', 'follow = head'] },
    { id: 'advance', code: ['for _ in range(n):', '    lead = lead.next'] },
    { id: 'move', code: ['while lead:', '    lead = lead.next', '    follow = follow.next'] },
    { id: 'finish', code: ['return follow'] },
  ],
};

function buildLinkedListHighlights(entries) {
  const highlights = {};
  entries.forEach(([nodeId, tone]) => {
    if (nodeId !== null && nodeId !== undefined) {
      highlights[nodeId] = tone;
    }
  });
  return highlights;
}

function buildForwardEdges(nodes) {
  return nodes.slice(0, -1).map((node, index) => ({
    from: node.id,
    to: nodes[index + 1].id,
    type: 'next',
  }));
}

function buildLinkedListReversalSteps() {
  const nextMap = [1, 2, 3, 4, null];
  const steps = [{
    action: 'start',
    iteration: 0,
    prev: null,
    curr: 0,
    nextNode: null,
    nextMap: [...nextMap],
    activeLine: 'init',
    nodeHighlights: buildLinkedListHighlights([[0, 'current']]),
  }];

  let prev = null;
  let curr = 0;
  let iteration = 1;

  while (curr !== null) {
    const nextNode = nextMap[curr];
    steps.push({
      action: 'save',
      iteration,
      prev,
      curr,
      nextNode,
      nextMap: [...nextMap],
      activeLine: 'save',
      nodeHighlights: buildLinkedListHighlights([
        [prev, 'anchor'],
        [curr, 'current'],
        [nextNode, 'support'],
      ]),
    });

    nextMap[curr] = prev;
    steps.push({
      action: 'rewire',
      iteration,
      prev,
      curr,
      nextNode,
      nextMap: [...nextMap],
      activeLine: 'rewire',
      nodeHighlights: buildLinkedListHighlights([
        [prev, 'anchor'],
        [curr, 'flip'],
        [nextNode, 'support'],
      ]),
    });

    prev = curr;
    curr = nextNode;
    steps.push({
      action: 'advance',
      iteration,
      prev,
      curr,
      nextNode,
      nextMap: [...nextMap],
      activeLine: 'advance',
      nodeHighlights: buildLinkedListHighlights([
        [prev, 'anchor'],
        [curr, 'current'],
      ]),
    });
    iteration += 1;
  }

  steps.push({
    action: 'finish',
    iteration: iteration - 1,
    prev,
    curr,
    nextNode: null,
    nextMap: [...nextMap],
    activeLine: 'finish',
    nodeHighlights: buildLinkedListHighlights([[prev, 'result']]),
  });

  return steps;
}

function buildFastSlowMiddleSteps() {
  return [
    {
      action: 'start',
      slow: 0,
      fast: 0,
      activeLine: 'init',
      nodeHighlights: buildLinkedListHighlights([[0, 'current']]),
    },
    {
      action: 'move',
      iteration: 1,
      slow: 1,
      fast: 2,
      activeLine: 'move',
      nodeHighlights: buildLinkedListHighlights([
        [1, 'anchor'],
        [2, 'current'],
      ]),
    },
    {
      action: 'move',
      iteration: 2,
      slow: 2,
      fast: 4,
      activeLine: 'move',
      nodeHighlights: buildLinkedListHighlights([
        [2, 'anchor'],
        [4, 'current'],
      ]),
    },
    {
      action: 'finish',
      slow: 2,
      fast: 4,
      activeLine: 'finish',
      nodeHighlights: buildLinkedListHighlights([
        [2, 'result'],
        [4, 'support'],
      ]),
    },
  ];
}

function buildFastSlowCycleSteps() {
  return [
    {
      action: 'start',
      slow: 0,
      fast: 0,
      finder: null,
      activeLine: 'init',
      nodeHighlights: buildLinkedListHighlights([[0, 'support']]),
    },
    {
      action: 'detect',
      iteration: 1,
      slow: 1,
      fast: 2,
      finder: null,
      activeLine: 'detect',
      nodeHighlights: buildLinkedListHighlights([
        [1, 'anchor'],
        [2, 'current'],
      ]),
    },
    {
      action: 'detect',
      iteration: 2,
      slow: 2,
      fast: 1,
      finder: null,
      activeLine: 'detect',
      nodeHighlights: buildLinkedListHighlights([
        [2, 'anchor'],
        [1, 'current'],
      ]),
    },
    {
      action: 'meet',
      iteration: 3,
      slow: 3,
      fast: 3,
      finder: null,
      activeLine: 'detect',
      nodeHighlights: buildLinkedListHighlights([[3, 'meeting']]),
    },
    {
      action: 'reset',
      slow: 3,
      fast: null,
      finder: 0,
      activeLine: 'reset',
      nodeHighlights: buildLinkedListHighlights([
        [3, 'meeting'],
        [0, 'current'],
      ]),
    },
    {
      action: 'locate',
      iteration: 1,
      slow: 1,
      fast: null,
      finder: 1,
      activeLine: 'locate',
      nodeHighlights: buildLinkedListHighlights([[1, 'result']]),
    },
    {
      action: 'finish',
      slow: 1,
      fast: null,
      finder: 1,
      activeLine: 'finish',
      nodeHighlights: buildLinkedListHighlights([[1, 'result']]),
    },
  ];
}

function buildFastSlowGapSteps() {
  return [
    {
      action: 'start',
      lead: 0,
      follow: 0,
      activeLine: 'init',
      nodeHighlights: buildLinkedListHighlights([[0, 'current']]),
    },
    {
      action: 'advance',
      progress: 1,
      total: 2,
      lead: 1,
      follow: 0,
      activeLine: 'advance',
      nodeHighlights: buildLinkedListHighlights([
        [1, 'current'],
        [0, 'anchor'],
      ]),
    },
    {
      action: 'advance',
      progress: 2,
      total: 2,
      lead: 2,
      follow: 0,
      activeLine: 'advance',
      nodeHighlights: buildLinkedListHighlights([
        [2, 'current'],
        [0, 'anchor'],
      ]),
    },
    {
      action: 'move',
      iteration: 1,
      lead: 3,
      follow: 1,
      activeLine: 'move',
      nodeHighlights: buildLinkedListHighlights([
        [3, 'current'],
        [1, 'anchor'],
      ]),
    },
    {
      action: 'move',
      iteration: 2,
      lead: 4,
      follow: 2,
      activeLine: 'move',
      nodeHighlights: buildLinkedListHighlights([
        [4, 'current'],
        [2, 'anchor'],
      ]),
    },
    {
      action: 'move',
      iteration: 3,
      lead: null,
      follow: 3,
      activeLine: 'move',
      nodeHighlights: buildLinkedListHighlights([[3, 'anchor']]),
    },
    {
      action: 'finish',
      lead: null,
      follow: 3,
      activeLine: 'finish',
      nodeHighlights: buildLinkedListHighlights([[3, 'result']]),
    },
  ];
}

const LINKED_LIST_REVERSAL_STEPS = buildLinkedListReversalSteps();

const FAST_SLOW_POINTER_SCENARIOS = {
  middle: {
    key: 'middle',
    nodes: LINKED_LIST_REVERSAL_NODES,
    edges: buildForwardEdges(LINKED_LIST_REVERSAL_NODES),
    steps: buildFastSlowMiddleSteps(),
  },
  cycle: {
    key: 'cycle',
    nodes: FAST_SLOW_CYCLE_NODES,
    edges: [
      ...buildForwardEdges(FAST_SLOW_CYCLE_NODES),
      { from: 3, to: 1, type: 'cycle' },
    ],
    steps: buildFastSlowCycleSteps(),
  },
  gap: {
    key: 'gap',
    nodes: LINKED_LIST_REVERSAL_NODES,
    edges: buildForwardEdges(LINKED_LIST_REVERSAL_NODES),
    steps: buildFastSlowGapSteps(),
  },
};

const ARRAY_DUPLICATE_NODES = [
  { id: 0, value: 0, x: 96, y: 182, next: 1, region: 'tail', nextLabelOffset: 52 },
  { id: 1, value: 1, x: 236, y: 182, next: 3, region: 'tail', nextLabelOffset: 52 },
  { id: 3, value: 3, x: 376, y: 182, next: 2, region: 'tail', nextLabelOffset: 52 },
  { id: 2, value: 2, x: 556, y: 118, next: 4, region: 'cycle', nextLabelOffset: 50, isEntry: true },
  { id: 4, value: 4, x: 556, y: 246, next: 2, region: 'cycle', nextLabelOffset: 52 },
];

const ARRAY_DUPLICATE_EDGE_PATHS = {
  '0-1': 'M 138 182 L 194 182',
  '1-3': 'M 278 182 L 334 182',
  '3-2': 'M 418 182 C 468 182, 492 138, 514 118',
  '2-4': 'M 588 132 C 640 146, 646 218, 588 232',
  '4-2': 'M 524 232 C 448 214, 446 148, 524 132',
};

const ARRAY_DUPLICATE_EDGES = [
  { from: 0, to: 1, tone: 'tail' },
  { from: 1, to: 3, tone: 'tail' },
  { from: 3, to: 2, tone: 'tail' },
  { from: 2, to: 4, tone: 'cycle' },
  { from: 4, to: 2, tone: 'cycle-back' },
];

const ARRAY_DUPLICATE_CODE_LINES = [
  { id: 'init', code: ['slow = fast = 0'] },
  { id: 'phase1', code: ['while True:', '    slow = nums[slow]', '    fast = nums[nums[fast]]', '    if slow == fast:', '        break'] },
  { id: 'reset', code: ['finder = 0'] },
  { id: 'phase2', code: ['while finder != slow:', '    finder = nums[finder]', '    slow = nums[slow]'] },
  { id: 'finish', code: ['return slow'] },
];

const ARRAY_DUPLICATE_STEPS = [
  {
    action: 'start',
    slow: 0,
    fast: 0,
    finder: null,
    phase: 1,
    round: 0,
    k: 0,
    activeLine: 'init',
    headerTone: 'init',
    nodeHighlights: buildLinkedListHighlights([[0, 'current']]),
  },
  {
    action: 'phase1',
    slow: 1,
    fast: 3,
    finder: null,
    phase: 1,
    round: 1,
    k: 1,
    activeLine: 'phase1',
    headerTone: 'phase1',
    nodeHighlights: buildLinkedListHighlights([
      [1, 'anchor'],
      [3, 'current'],
    ]),
  },
  {
    action: 'phase1',
    slow: 3,
    fast: 4,
    finder: null,
    phase: 1,
    round: 2,
    k: 2,
    activeLine: 'phase1',
    headerTone: 'phase1',
    nodeHighlights: buildLinkedListHighlights([
      [3, 'anchor'],
      [4, 'current'],
    ]),
  },
  {
    action: 'phase1',
    slow: 2,
    fast: 4,
    finder: null,
    phase: 1,
    round: 3,
    k: 3,
    activeLine: 'phase1',
    headerTone: 'phase1',
    nodeHighlights: buildLinkedListHighlights([
      [2, 'anchor'],
      [4, 'current'],
    ]),
  },
  {
    action: 'meet',
    slow: 4,
    fast: 4,
    finder: null,
    phase: 1,
    round: 4,
    k: 4,
    activeLine: 'phase1',
    headerTone: 'phase1',
    nodeHighlights: buildLinkedListHighlights([[4, 'meeting']]),
  },
  {
    action: 'warning',
    slow: 4,
    fast: 4,
    finder: null,
    phase: 1,
    round: 4,
    k: 4,
    activeLine: 'phase1',
    headerTone: 'warning',
    nodeHighlights: buildLinkedListHighlights([
      [4, 'wrong'],
      [2, 'result'],
    ]),
  },
  {
    action: 'reset',
    slow: 4,
    fast: null,
    finder: 0,
    phase: 2,
    round: 0,
    s: 0,
    activeLine: 'reset',
    headerTone: 'phase2',
    nodeHighlights: buildLinkedListHighlights([
      [4, 'meeting'],
      [0, 'current'],
    ]),
  },
  {
    action: 'phase2',
    slow: 2,
    fast: null,
    finder: 1,
    phase: 2,
    round: 1,
    s: 1,
    activeLine: 'phase2',
    headerTone: 'phase2',
    nodeHighlights: buildLinkedListHighlights([
      [2, 'anchor'],
      [1, 'current'],
    ]),
  },
  {
    action: 'phase2',
    slow: 4,
    fast: null,
    finder: 3,
    phase: 2,
    round: 2,
    s: 2,
    activeLine: 'phase2',
    headerTone: 'phase2',
    nodeHighlights: buildLinkedListHighlights([
      [4, 'anchor'],
      [3, 'current'],
    ]),
  },
  {
    action: 'meet-entry',
    slow: 2,
    fast: null,
    finder: 2,
    phase: 2,
    round: 3,
    s: 3,
    activeLine: 'phase2',
    headerTone: 'phase2',
    nodeHighlights: buildLinkedListHighlights([[2, 'result']]),
  },
  {
    action: 'finish',
    slow: 2,
    fast: null,
    finder: 2,
    phase: 2,
    round: 3,
    s: 3,
    activeLine: 'finish',
    headerTone: 'finish',
    nodeHighlights: buildLinkedListHighlights([[2, 'result']]),
  },
];

const LRU_CACHE_CAPACITY = 2;

const LRU_CACHE_OPERATIONS = [
  { type: 'put', key: 1, value: 1, result: null },
  { type: 'put', key: 2, value: 2, result: null },
  { type: 'get', key: 1, result: 1 },
  { type: 'put', key: 3, value: 3, result: null },
  { type: 'get', key: 2, result: -1 },
  { type: 'put', key: 4, value: 4, result: null },
  { type: 'get', key: 1, result: -1 },
  { type: 'get', key: 3, result: 3 },
  { type: 'get', key: 4, result: 4 },
];

const LRU_CACHE_CODE_LINES = [
  { id: 'class-def', code: ['class LRUCache:'] },
  { id: 'remove-def', code: ['    def remove(self, node: Node) -> None:'] },
  { id: 'remove-neighbors', code: ['        prev_node, next_node = node.prev, node.next'] },
  { id: 'remove-prev-next', code: ['        prev_node.next = next_node'] },
  { id: 'remove-next-prev', code: ['        next_node.prev = prev_node'] },
  { id: 'insert-def', code: ['    def insert_before_tail(self, node: Node) -> None:'] },
  { id: 'insert-prev', code: ['        prev_node = self.right.prev'] },
  { id: 'insert-prev-next', code: ['        prev_node.next = node'] },
  { id: 'insert-node-prev', code: ['        node.prev = prev_node'] },
  { id: 'insert-node-next', code: ['        node.next = self.right'] },
  { id: 'insert-right-prev', code: ['        self.right.prev = node'] },
  { id: 'get-def', code: ['    def get(self, key: int) -> int:'] },
  { id: 'get-lookup', code: ['        if key not in self.cache:'] },
  { id: 'get-miss', code: ['            return -1'] },
  { id: 'get-node', code: ['        node = self.cache[key]'] },
  { id: 'get-remove', code: ['        self.remove(node)'] },
  { id: 'get-insert', code: ['        self.insert_before_tail(node)'] },
  { id: 'get-return', code: ['        return node.value'] },
  { id: 'put-def', code: ['    def put(self, key: int, value: int) -> None:'] },
  { id: 'put-lookup', code: ['        if key in self.cache:'] },
  { id: 'put-remove-existing', code: ['            self.remove(self.cache[key])'] },
  { id: 'put-node', code: ['        node = Node(key, value)'] },
  { id: 'put-cache', code: ['        self.cache[key] = node'] },
  { id: 'put-insert', code: ['        self.insert_before_tail(node)'] },
  { id: 'put-capacity', code: ['        if len(self.cache) > self.capacity:'] },
  { id: 'put-lru', code: ['            lru = self.left.next'] },
  { id: 'put-remove-lru', code: ['            self.remove(lru)'] },
  { id: 'put-delete', code: ['            del self.cache[lru.key]'] },
];

function buildLRUCacheSteps() {
  const steps = [];
  const cache = new Map();
  const order = [];

  const snapshot = (operationIndex, action, activeLine, extra = {}) => {
    steps.push({
      operationIndex,
      action,
      activeLine,
      order: [...order],
      cacheEntries: [...cache.entries()].sort(([leftKey], [rightKey]) => leftKey - rightKey),
      ...extra,
    });
  };

  LRU_CACHE_OPERATIONS.forEach((operation, operationIndex) => {
    const { key, type, value } = operation;

    if (type === 'get') {
      if (!cache.has(key)) {
        snapshot(operationIndex, 'get-miss', 'get-miss', { lookupKey: key, lookupTone: 'miss' });
        return;
      }

      snapshot(operationIndex, 'get-hit', 'get-node', { lookupKey: key, lookupTone: 'hit' });
      order.splice(order.indexOf(key), 1);
      snapshot(operationIndex, 'get-remove', 'get-remove', { detachedKey: key, lookupKey: key, lookupTone: 'hit' });
      order.push(key);
      snapshot(operationIndex, 'get-insert', 'get-insert', { movedKey: key, lookupKey: key, lookupTone: 'hit' });
      snapshot(operationIndex, 'get-return', 'get-return', { movedKey: key, lookupKey: key, lookupTone: 'hit' });
      return;
    }

    const existed = cache.has(key);
    snapshot(operationIndex, 'put-lookup', 'put-lookup', {
      lookupKey: key,
      lookupTone: existed ? 'hit' : 'miss',
    });

    if (existed) {
      order.splice(order.indexOf(key), 1);
      snapshot(operationIndex, 'put-remove-existing', 'put-remove-existing', { detachedKey: key, lookupKey: key, lookupTone: 'hit' });
    }

    snapshot(operationIndex, 'put-new-node', 'put-node', { detachedKey: key, detachedValue: value });
    cache.set(key, value);
    snapshot(operationIndex, 'put-map', 'put-cache', { detachedKey: key });
    order.push(key);
    snapshot(operationIndex, 'put-insert', 'put-insert', { movedKey: key });

    if (cache.size <= LRU_CACHE_CAPACITY) {
      snapshot(operationIndex, 'capacity-ok', 'put-capacity');
      return;
    }

    snapshot(operationIndex, 'capacity-over', 'put-capacity');
    const lruKey = order[0];
    snapshot(operationIndex, 'evict-target', 'put-lru', { evictKey: lruKey });
    order.shift();
    snapshot(operationIndex, 'evict-remove', 'put-remove-lru', { detachedKey: lruKey, evictKey: lruKey });
    cache.delete(lruKey);
    snapshot(operationIndex, 'evict-delete', 'put-delete', { evictedKey: lruKey });
  });

  return steps;
}

const LRU_CACHE_STEPS = buildLRUCacheSteps();

const TREE_TRAVERSAL_NODES = [
  { id: 1, value: 1, x: 350, y: 54, left: 2, right: 3 },
  { id: 2, value: 2, x: 210, y: 145, left: 4, right: 5 },
  { id: 3, value: 3, x: 490, y: 145, left: null, right: 6 },
  { id: 4, value: 4, x: 130, y: 238, left: null, right: null },
  { id: 5, value: 5, x: 290, y: 238, left: null, right: null },
  { id: 6, value: 6, x: 570, y: 238, left: null, right: null },
];

const TREE_TRAVERSAL_NODE_MAP = new Map(TREE_TRAVERSAL_NODES.map((node) => [node.id, node]));

const TREE_TRAVERSAL_EDGES = TREE_TRAVERSAL_NODES.flatMap((node) => [node.left, node.right]
  .filter((childId) => childId !== null)
  .map((childId) => ({ from: node.id, to: childId })));

const TREE_TRAVERSAL_CODE_LINES = {
  preorder: [
    { id: 'init', code: ['if not root: return []', 'stack = [root]', 'order = []'] },
    { id: 'loop', code: ['while stack:'] },
    { id: 'pop', code: ['    node = stack.pop()'] },
    { id: 'visit', code: ['    order.append(node.val)'] },
    { id: 'push-right', code: ['    if node.right:', '        stack.append(node.right)'] },
    { id: 'push-left', code: ['    if node.left:', '        stack.append(node.left)'] },
    { id: 'finish', code: ['return order'] },
  ],
  inorder: [
    { id: 'init', code: ['stack, order = [], []', 'current = root'] },
    { id: 'loop', code: ['while stack or current:'] },
    { id: 'descend', code: ['    while current:', '        stack.append(current)', '        current = current.left'] },
    { id: 'pop', code: ['    current = stack.pop()'] },
    { id: 'visit', code: ['    order.append(current.val)'] },
    { id: 'move-right', code: ['    current = current.right'] },
    { id: 'finish', code: ['return order'] },
  ],
  postorder: [
    { id: 'init', code: ['if not root: return []', 'stack = [root]', 'reverse_order = []'] },
    { id: 'loop', code: ['while stack:'] },
    { id: 'pop', code: ['    node = stack.pop()'] },
    { id: 'visit', code: ['    reverse_order.append(node.val)'] },
    { id: 'push-left', code: ['    if node.left:', '        stack.append(node.left)'] },
    { id: 'push-right', code: ['    if node.right:', '        stack.append(node.right)'] },
    { id: 'finish', code: ['return reverse_order[::-1]'] },
  ],
  level: [
    { id: 'init', code: ['if not root: return []', 'queue = deque([root])', 'order = []'] },
    { id: 'loop', code: ['while queue:'] },
    { id: 'pop', code: ['    node = queue.popleft()'] },
    { id: 'visit', code: ['    order.append(node.val)'] },
    { id: 'add-left', code: ['    if node.left:', '        queue.append(node.left)'] },
    { id: 'add-right', code: ['    if node.right:', '        queue.append(node.right)'] },
    { id: 'finish', code: ['return order'] },
  ],
};

function buildPreorderTraversalSteps() {
  const steps = [];
  const stack = [1];
  const output = [];
  const snapshot = (action, activeLine, current = null) => steps.push({
    action,
    activeLine,
    current,
    container: [...stack],
    output: [...output],
    buffer: [],
  });

  snapshot('start', 'init');
  while (stack.length) {
    const nodeId = stack.pop();
    const node = TREE_TRAVERSAL_NODE_MAP.get(nodeId);
    snapshot('pop', 'pop', nodeId);
    output.push(nodeId);
    snapshot('visit', 'visit', nodeId);
    if (node.right !== null) {
      stack.push(node.right);
      snapshot('push-right', 'push-right', nodeId);
    }
    if (node.left !== null) {
      stack.push(node.left);
      snapshot('push-left', 'push-left', nodeId);
    }
  }
  snapshot('finish', 'finish');
  return steps;
}

function buildInorderTraversalSteps() {
  const steps = [];
  const stack = [];
  const output = [];
  let current = 1;
  const snapshot = (action, activeLine, focus = current) => steps.push({
    action,
    activeLine,
    current: focus,
    container: [...stack],
    output: [...output],
    buffer: [],
  });

  snapshot('start', 'init');
  while (stack.length || current !== null) {
    while (current !== null) {
      const pushed = current;
      stack.push(pushed);
      current = TREE_TRAVERSAL_NODE_MAP.get(pushed).left;
      snapshot('descend', 'descend', pushed);
    }
    current = stack.pop();
    snapshot('pop', 'pop', current);
    output.push(current);
    snapshot('visit', 'visit', current);
    current = TREE_TRAVERSAL_NODE_MAP.get(current).right;
    snapshot('move-right', 'move-right', current);
  }
  snapshot('finish', 'finish', null);
  return steps;
}

function buildPostorderTraversalSteps() {
  const steps = [];
  const stack = [1];
  const reverseOrder = [];
  const snapshot = (action, activeLine, current = null, finished = false) => steps.push({
    action,
    activeLine,
    current,
    container: [...stack],
    output: finished ? [...reverseOrder].reverse() : [],
    buffer: [...reverseOrder],
  });

  snapshot('start', 'init');
  while (stack.length) {
    const nodeId = stack.pop();
    const node = TREE_TRAVERSAL_NODE_MAP.get(nodeId);
    snapshot('pop', 'pop', nodeId);
    reverseOrder.push(nodeId);
    snapshot('visit-buffer', 'visit', nodeId);
    if (node.left !== null) {
      stack.push(node.left);
      snapshot('push-left', 'push-left', nodeId);
    }
    if (node.right !== null) {
      stack.push(node.right);
      snapshot('push-right', 'push-right', nodeId);
    }
  }
  snapshot('finish', 'finish', null, true);
  return steps;
}

function buildLevelOrderTraversalSteps() {
  const steps = [];
  const queue = [1];
  const output = [];
  const snapshot = (action, activeLine, current = null) => steps.push({
    action,
    activeLine,
    current,
    container: [...queue],
    output: [...output],
    buffer: [],
  });

  snapshot('start', 'init');
  while (queue.length) {
    const nodeId = queue.shift();
    const node = TREE_TRAVERSAL_NODE_MAP.get(nodeId);
    snapshot('pop', 'pop', nodeId);
    output.push(nodeId);
    snapshot('visit', 'visit', nodeId);
    if (node.left !== null) {
      queue.push(node.left);
      snapshot('add-left', 'add-left', nodeId);
    }
    if (node.right !== null) {
      queue.push(node.right);
      snapshot('add-right', 'add-right', nodeId);
    }
  }
  snapshot('finish', 'finish');
  return steps;
}

const TREE_TRAVERSAL_SCENARIOS = {
  preorder: { steps: buildPreorderTraversalSteps(), container: 'stack' },
  inorder: { steps: buildInorderTraversalSteps(), container: 'stack' },
  postorder: { steps: buildPostorderTraversalSteps(), container: 'stack' },
  level: { steps: buildLevelOrderTraversalSteps(), container: 'queue' },
};

const TREE_TRAVERSAL_RECURSIVE_CODE_LINES = {
  preorder: [
    { id: 'call', code: ['def preorder(node, order):', '    if not node: return'] },
    { id: 'visit', code: ['    order.append(node.val)'] },
    { id: 'recurse-left', code: ['    preorder(node.left, order)'] },
    { id: 'recurse-right', code: ['    preorder(node.right, order)'] },
    { id: 'return', code: ['    # control returns to the caller'] },
  ],
  inorder: [
    { id: 'call', code: ['def inorder(node, order):', '    if not node: return'] },
    { id: 'recurse-left', code: ['    inorder(node.left, order)'] },
    { id: 'visit', code: ['    order.append(node.val)'] },
    { id: 'recurse-right', code: ['    inorder(node.right, order)'] },
    { id: 'return', code: ['    # control returns to the caller'] },
  ],
  postorder: [
    { id: 'call', code: ['def postorder(node, order):', '    if not node: return'] },
    { id: 'recurse-left', code: ['    postorder(node.left, order)'] },
    { id: 'recurse-right', code: ['    postorder(node.right, order)'] },
    { id: 'visit', code: ['    order.append(node.val)'] },
    { id: 'return', code: ['    # control returns to the caller'] },
  ],
};

function buildPreorderRecursiveSteps() {
  const steps = [];
  const callStack = [];
  const output = [];
  const snapshot = (action, activeLine, current = null) => steps.push({
    action,
    activeLine,
    current,
    container: [...callStack],
    output: [...output],
    buffer: [],
  });

  const recurse = (nodeId) => {
    if (nodeId === null) return;
    callStack.push(nodeId);
    snapshot('call', 'call', nodeId);
    output.push(nodeId);
    snapshot('visit', 'visit', nodeId);
    const node = TREE_TRAVERSAL_NODE_MAP.get(nodeId);
    snapshot('recurse-left', 'recurse-left', nodeId);
    recurse(node.left);
    snapshot('recurse-right', 'recurse-right', nodeId);
    recurse(node.right);
    callStack.pop();
    snapshot('return', 'return', nodeId);
  };

  snapshot('start', 'call');
  recurse(1);
  snapshot('finish', 'return');
  return steps;
}

function buildInorderRecursiveSteps() {
  const steps = [];
  const callStack = [];
  const output = [];
  const snapshot = (action, activeLine, current = null) => steps.push({
    action,
    activeLine,
    current,
    container: [...callStack],
    output: [...output],
    buffer: [],
  });

  const recurse = (nodeId) => {
    if (nodeId === null) return;
    callStack.push(nodeId);
    snapshot('call', 'call', nodeId);
    const node = TREE_TRAVERSAL_NODE_MAP.get(nodeId);
    snapshot('recurse-left', 'recurse-left', nodeId);
    recurse(node.left);
    output.push(nodeId);
    snapshot('visit', 'visit', nodeId);
    snapshot('recurse-right', 'recurse-right', nodeId);
    recurse(node.right);
    callStack.pop();
    snapshot('return', 'return', nodeId);
  };

  snapshot('start', 'call');
  recurse(1);
  snapshot('finish', 'return');
  return steps;
}

function buildPostorderRecursiveSteps() {
  const steps = [];
  const callStack = [];
  const output = [];
  const snapshot = (action, activeLine, current = null) => steps.push({
    action,
    activeLine,
    current,
    container: [...callStack],
    output: [...output],
    buffer: [],
  });

  const recurse = (nodeId) => {
    if (nodeId === null) return;
    callStack.push(nodeId);
    snapshot('call', 'call', nodeId);
    const node = TREE_TRAVERSAL_NODE_MAP.get(nodeId);
    snapshot('recurse-left', 'recurse-left', nodeId);
    recurse(node.left);
    snapshot('recurse-right', 'recurse-right', nodeId);
    recurse(node.right);
    output.push(nodeId);
    snapshot('visit', 'visit', nodeId);
    callStack.pop();
    snapshot('return', 'return', nodeId);
  };

  snapshot('start', 'call');
  recurse(1);
  snapshot('finish', 'return');
  return steps;
}

const TREE_TRAVERSAL_RECURSIVE_SCENARIOS = {
  preorder: { steps: buildPreorderRecursiveSteps(), container: 'stack' },
  inorder: { steps: buildInorderRecursiveSteps(), container: 'stack' },
  postorder: { steps: buildPostorderRecursiveSteps(), container: 'stack' },
};

const AVL_ROTATION_CODE_LINES = {
  ll: [
    { id: 'detect', code: ['# balance(30) = +2: LL case'] },
    { id: 'promote', code: ['new_root = node.left       # 20', 'transfer = new_root.right  # 25'] },
    { id: 'rotate', code: ['new_root.right = node       # 20 -> 30', 'node.left = transfer        # 30 -> 25'] },
    { id: 'finish', code: ['return new_root             # 20'] },
  ],
  lr: [
    { id: 'detect', code: ['# balance(30) = +2: LR case'] },
    { id: 'left-rotate', code: ['node.left = rotate_left(node.left)', '# 20 becomes the left-subtree root'] },
    { id: 'right-rotate', code: ['new_root = rotate_right(node)', '# 20 becomes the subtree root'] },
    { id: 'finish', code: ['return new_root             # 20'] },
  ],
};

function buildAVLLLSteps() {
  const beforeNodes = [
    { id: 'n30', value: 30, x: 350, y: 54 },
    { id: 'n20', value: 20, x: 225, y: 145 },
    { id: 'n10', value: 10, x: 135, y: 238 },
    { id: 'n25', value: 25, x: 315, y: 238 },
  ];
  const beforeEdges = [
    { from: 'n30', to: 'n20' },
    { from: 'n20', to: 'n10' },
    { from: 'n20', to: 'n25', label: 'T2' },
  ];
  const afterNodes = [
    { id: 'n20', value: 20, x: 350, y: 54 },
    { id: 'n10', value: 10, x: 225, y: 145 },
    { id: 'n30', value: 30, x: 475, y: 145 },
    { id: 'n25', value: 25, x: 405, y: 238 },
  ];
  const afterEdges = [
    { from: 'n20', to: 'n10' },
    { from: 'n20', to: 'n30' },
    { from: 'n30', to: 'n25', label: 'T2' },
  ];

  return [
    { action: 'detect', activeLine: 'detect', nodes: beforeNodes, edges: beforeEdges, root: 30, moved: null, highlights: { n30: 'pivot' } },
    { action: 'promote', activeLine: 'promote', nodes: beforeNodes, edges: beforeEdges, root: 30, moved: 20, highlights: { n30: 'pivot', n20: 'promoted', n25: 'transfer' } },
    { action: 'rotate', activeLine: 'rotate', nodes: afterNodes, edges: afterEdges, root: 20, moved: 25, highlights: { n20: 'promoted', n30: 'pivot', n25: 'transfer' } },
    { action: 'finish', activeLine: 'finish', nodes: afterNodes, edges: afterEdges, root: 20, moved: 25, highlights: { n20: 'result', n25: 'transfer' } },
  ];
}

function buildAVLLRSteps() {
  const beforeNodes = [
    { id: 'n30', value: 30, x: 350, y: 54 },
    { id: 'n10', value: 10, x: 225, y: 145 },
    { id: 'n20', value: 20, x: 315, y: 238 },
  ];
  const beforeEdges = [
    { from: 'n30', to: 'n10' },
    { from: 'n10', to: 'n20' },
  ];
  const intermediateNodes = [
    { id: 'n30', value: 30, x: 350, y: 54 },
    { id: 'n20', value: 20, x: 225, y: 145 },
    { id: 'n10', value: 10, x: 135, y: 238 },
  ];
  const intermediateEdges = [
    { from: 'n30', to: 'n20' },
    { from: 'n20', to: 'n10' },
  ];
  const afterNodes = [
    { id: 'n20', value: 20, x: 350, y: 54 },
    { id: 'n10', value: 10, x: 225, y: 145 },
    { id: 'n30', value: 30, x: 475, y: 145 },
  ];
  const afterEdges = [
    { from: 'n20', to: 'n10' },
    { from: 'n20', to: 'n30' },
  ];

  return [
    { action: 'detect', activeLine: 'detect', nodes: beforeNodes, edges: beforeEdges, root: 30, moved: null, highlights: { n30: 'pivot', n20: 'promoted' } },
    { action: 'left-rotate', activeLine: 'left-rotate', nodes: intermediateNodes, edges: intermediateEdges, root: 30, moved: 20, highlights: { n20: 'promoted', n10: 'pivot' } },
    { action: 'right-rotate', activeLine: 'right-rotate', nodes: afterNodes, edges: afterEdges, root: 20, moved: 20, highlights: { n20: 'promoted', n30: 'pivot' } },
    { action: 'finish', activeLine: 'finish', nodes: afterNodes, edges: afterEdges, root: 20, moved: 20, highlights: { n20: 'result' } },
  ];
}

const AVL_ROTATION_SCENARIOS = {
  ll: { steps: buildAVLLLSteps() },
  lr: { steps: buildAVLLRSteps() },
};

function buildLinkedListEdgePath(source, target, type) {
  if (type === 'cycle') {
    return `M ${source.x + 42} ${source.y} C ${source.x + 100} ${source.y + 102}, ${target.x + 72} ${target.y + 102}, ${target.x} ${target.y + 30}`;
  }
  if (source.x < target.x) {
    return `M ${source.x + 42} ${source.y} L ${target.x - 42} ${target.y}`;
  }
  return `M ${source.x - 42} ${source.y} C ${source.x - 82} ${source.y - 74}, ${target.x + 82} ${target.y - 74}, ${target.x + 42} ${target.y}`;
}

function LinkedListDiagram({
  ariaLabel,
  annotations = [],
  edges,
  nodes,
  nodeHighlights = {},
  nullPointers = [],
  pointerLabels,
  viewBoxHeight = 280,
}) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const labelsByNode = new Map(nodes.map((node) => [node.id, []]));
  const annotationsByNode = new Map(nodes.map((node) => [node.id, []]));

  pointerLabels.forEach((label) => {
    if (label.nodeId !== null && label.nodeId !== undefined) {
      labelsByNode.get(label.nodeId)?.push(label);
    }
  });

  annotations.forEach((annotation) => {
    if (annotation.nodeId !== null && annotation.nodeId !== undefined) {
      annotationsByNode.get(annotation.nodeId)?.push(annotation);
    }
  });

  return (
    <div className="linked-list-diagram">
      <svg
        aria-label={ariaLabel}
        className="linked-list-diagram-svg"
        role="img"
        viewBox={`0 0 700 ${viewBoxHeight}`}
      >
        <defs>
          <marker
            id="linked-list-arrow-next"
            markerHeight="8"
            markerWidth="8"
            orient="auto"
            refX="8"
            refY="4"
            viewBox="0 0 8 8"
          >
            <path d="M 0 0 L 8 4 L 0 8 z" fill="#1d596d" />
          </marker>
          <marker
            id="linked-list-arrow-reversed"
            markerHeight="8"
            markerWidth="8"
            orient="auto"
            refX="8"
            refY="4"
            viewBox="0 0 8 8"
          >
            <path d="M 0 0 L 8 4 L 0 8 z" fill="#c96e27" />
          </marker>
          <marker
            id="linked-list-arrow-cycle"
            markerHeight="8"
            markerWidth="8"
            orient="auto"
            refX="8"
            refY="4"
            viewBox="0 0 8 8"
          >
            <path d="M 0 0 L 8 4 L 0 8 z" fill="#157158" />
          </marker>
        </defs>

        {edges.map((edge) => {
          const source = nodeMap.get(edge.from);
          const target = nodeMap.get(edge.to);
          return (
            <path
              className={`linked-list-edge ${edge.type}`}
              d={buildLinkedListEdgePath(source, target, edge.type)}
              key={`${edge.from}-${edge.to}-${edge.type}`}
              markerEnd={`url(#linked-list-arrow-${edge.type === 'reversed' ? 'reversed' : edge.type === 'cycle' ? 'cycle' : 'next'})`}
            />
          );
        })}

        {nodes.map((node) => {
          const nodeLabels = labelsByNode.get(node.id) ?? [];
          return nodeLabels.map((label, index) => {
            const width = Math.max(56, label.text.length * 7 + 20);
            const labelX = node.x - width / 2;
            const labelY = node.y - 74 - index * 28;
            return (
              <g className={`linked-list-pointer-label ${label.tone}`} key={`${node.id}-${label.text}`}>
                <line x1={node.x} x2={node.x} y1={labelY + 20} y2={node.y - 28} />
                <rect height="20" rx="9" ry="9" width={width} x={labelX} y={labelY} />
                <text dominantBaseline="middle" textAnchor="middle" x={node.x} y={labelY + 10}>
                  {label.text}
                </text>
              </g>
            );
          });
        })}

        {nodes.map((node) => {
          const annotationItems = annotationsByNode.get(node.id) ?? [];
          return annotationItems.map((annotation, index) => (
            <g className={`linked-list-annotation ${annotation.tone ?? 'muted'}`} key={`${node.id}-${annotation.label}`}>
              <rect
                height="18"
                rx="8"
                ry="8"
                width={Math.max(60, annotation.label.length * 7 + 22)}
                x={node.x - Math.max(60, annotation.label.length * 7 + 22) / 2}
                y={node.y + 42 + index * 22}
              />
              <text dominantBaseline="middle" textAnchor="middle" x={node.x} y={node.y + 51 + index * 22}>
                {annotation.label}
              </text>
            </g>
          ));
        })}

        {nodes.map((node) => (
          <g
            className={`linked-list-node${nodeHighlights[node.id] ? ` ${nodeHighlights[node.id]}` : ''}`}
            key={node.id}
          >
            <rect height="50" rx="12" ry="12" width="84" x={node.x - 42} y={node.y - 25} />
            <text dominantBaseline="middle" textAnchor="middle" x={node.x} y={node.y}>
              {node.value}
            </text>
          </g>
        ))}
      </svg>

      {nullPointers.length > 0 && (
        <div className="linked-list-null-pointers">
          {nullPointers.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      )}
    </div>
  );
}

function LinkedListReversalVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const step = LINKED_LIST_REVERSAL_STEPS[activeStep];

  const pointerLabels = [];
  const nullPointers = [];

  if (step.prev !== null) {
    pointerLabels.push({ nodeId: step.prev, text: 'prev', tone: 'green' });
  } else {
    nullPointers.push('prev = None');
  }

  if (step.curr !== null) {
    pointerLabels.push({ nodeId: step.curr, text: 'curr', tone: 'orange' });
  } else {
    nullPointers.push('curr = None');
  }

  if (step.action === 'save' || step.action === 'rewire' || step.action === 'advance') {
    if (step.nextNode !== null) {
      pointerLabels.push({ nodeId: step.nextNode, text: 'next_node', tone: 'blue' });
    } else {
      nullPointers.push('next_node = None');
    }
  }

  if (step.action === 'finish' && step.prev !== null) {
    pointerLabels.push({ nodeId: step.prev, text: 'return', tone: 'gold' });
  }

  const edges = [];
  step.nextMap.forEach((nextIndex, index) => {
    if (nextIndex !== null) {
      edges.push({
        from: index,
        to: nextIndex,
        type: nextIndex > index ? 'next' : 'reversed',
      });
    }
  });

  const pointerCards = [
    {
      label: 'prev',
      value: step.prev === null ? 'None' : LINKED_LIST_REVERSAL_NODES[step.prev].value,
      tone: 'green',
    },
    {
      label: 'curr',
      value: step.curr === null ? 'None' : LINKED_LIST_REVERSAL_NODES[step.curr].value,
      tone: 'orange',
    },
    {
      label: 'next_node',
      value: step.action === 'save' || step.action === 'rewire' || step.action === 'advance'
        ? (step.nextNode === null ? 'None' : LINKED_LIST_REVERSAL_NODES[step.nextNode].value)
        : '-',
      tone: 'blue',
    },
  ];

  const activeLineLabel = {
    init: t('初始化', 'Initialize'),
    loop: t('进入循环', 'Enter loop'),
    save: t('保存 next_node', 'Save next_node'),
    rewire: t('翻转一条 next', 'Flip one next'),
    advance: t('推进三个指针', 'Advance the pointers'),
    finish: t('返回新头节点', 'Return the new head'),
  }[step.activeLine];

  let title = '';
  let detail = '';

  if (step.action === 'start') {
    title = t('初始化：prev 为空，curr 指向头节点', 'Initialize: prev is empty and curr points to the head');
    detail = t(
      '反转从空前缀开始。真正变化发生在每一轮的保存、改线、推进三步里。',
      'The reversed prefix starts empty. Each loop performs the same three operations: save, rewire, then advance.',
    );
  } else if (step.action === 'save') {
    title = t(`第 ${step.iteration} 轮：先保存 next_node`, `Iteration ${step.iteration}: save next_node first`);
    detail = step.nextNode === null
      ? t('当前节点已经是原链表尾部，保存结果是 `None`。', 'The current node is the original tail, so the saved next node is `None`.')
      : t(
        `保存值为 ${LINKED_LIST_REVERSAL_NODES[step.nextNode].value} 的节点，后面改写 curr.next 时不会丢链。`,
        `Save the node with value ${LINKED_LIST_REVERSAL_NODES[step.nextNode].value} so rewiring curr.next does not lose the suffix.`,
      );
  } else if (step.action === 'rewire') {
    title = t(`第 ${step.iteration} 轮：执行 curr.next = prev`, `Iteration ${step.iteration}: execute curr.next = prev`);
    detail = step.prev === null
      ? t('第一条被翻转的边直接指向 `None`，原头节点会成为新尾节点。', 'The first flipped edge points to `None`, so the original head becomes the new tail.')
      : t(
        `当前节点的箭头改为指向值为 ${LINKED_LIST_REVERSAL_NODES[step.prev].value} 的前驱，橙色边表示已经翻转。`,
        `The current node now points to the node with value ${LINKED_LIST_REVERSAL_NODES[step.prev].value}. Orange edges mark the reversed prefix.`,
      );
  } else if (step.action === 'advance') {
    title = t(`第 ${step.iteration} 轮：推进 prev 和 curr`, `Iteration ${step.iteration}: advance prev and curr`);
    detail = step.curr === null
      ? t('`curr` 已经走到 `None`，下一步可以直接返回 `prev`。', '`curr` has reached `None`, so the next step can return `prev` directly.')
      : t(
        `新的 prev 停在值为 ${LINKED_LIST_REVERSAL_NODES[step.prev].value} 的节点，curr 继续处理值为 ${LINKED_LIST_REVERSAL_NODES[step.curr].value} 的节点。`,
        `The new prev stays at value ${LINKED_LIST_REVERSAL_NODES[step.prev].value}, and curr continues with value ${LINKED_LIST_REVERSAL_NODES[step.curr].value}.`,
      );
  } else {
    title = t('结束：prev 就是新的头节点', 'Finish: prev is the new head');
    detail = t(
      '整条链的 `next` 都已经翻转完成，返回值从最右侧节点开始向左遍历。',
      'Every `next` pointer has been reversed. Traversal now starts at the rightmost node and proceeds left.',
    );
  }

  return (
    <section
      className="linked-list-reversal-visual"
      aria-label={t('链表反转过程演示', 'Linked-list reversal walkthrough')}
    >
      <header className="linked-list-reversal-header">
        <div>
          <p className="eyebrow">{t('链表反转', 'Linked-list reversal')}</p>
          <h2>{t('三指针模板拆成可视步骤', 'Split the three-pointer template into visible steps')}</h2>
          <p>{t(
            '固定链表 `1 -> 2 -> 3 -> 4 -> 5`。蓝色是尚未改写的 `next`，橙色是已经翻转的 `next`。',
            'The list is fixed at `1 -> 2 -> 3 -> 4 -> 5`. Blue edges are still original `next` pointers; orange edges have already been reversed.',
          )}</p>
        </div>
        <div className="linked-list-reversal-legend">
          <span><i className="next" /> next</span>
          <span><i className="reversed" /> reversed next</span>
          <span><i className="result" /> return head</span>
        </div>
      </header>

      <div className={`linked-list-reversal-step-copy ${step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {LINKED_LIST_REVERSAL_STEPS.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="linked-list-reversal-workspace">
        <div className="linked-list-reversal-stage-card">
          <div className="linked-list-reversal-stage-heading">
            <span>{t('当前链表状态', 'Current list state')}</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>
          <LinkedListDiagram
            ariaLabel={t('链表反转节点与指针状态', 'Linked-list reversal nodes and pointers')}
            edges={edges}
            nodes={LINKED_LIST_REVERSAL_NODES}
            nodeHighlights={step.nodeHighlights}
            nullPointers={nullPointers}
            pointerLabels={pointerLabels}
          />
          <div className="linked-list-reversal-pointer-cards">
            {pointerCards.map((card) => (
              <div className={`linked-list-reversal-pointer-card ${card.tone}`} key={card.label}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="linked-list-reversal-code" aria-label={t('当前模板代码', 'Active template code')}>
          <div className="linked-list-reversal-code-heading">
            <span>{t('迭代模板', 'Iterative template')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="linked-list-reversal-code-lines">
            {LINKED_LIST_REVERSAL_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => (
                  <code key={code}>{code}</code>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="linked-list-reversal-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={LINKED_LIST_REVERSAL_STEPS.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择链表反转步骤', 'Select a linked-list reversal step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(LINKED_LIST_REVERSAL_STEPS.length - 1, current + 1))}
          disabled={activeStep === LINKED_LIST_REVERSAL_STEPS.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

function FastSlowPointerVisual() {
  const { isEnglish, t } = useUiCopy();
  const [mode, setMode] = useState('middle');
  const [activeStep, setActiveStep] = useState(0);
  const scenario = FAST_SLOW_POINTER_SCENARIOS[mode];
  const step = scenario.steps[activeStep];

  const pointerLabels = [];
  const nullPointers = [];
  const annotations = [];
  let pointerCards = [];

  if (mode === 'middle') {
    pointerLabels.push({ nodeId: step.slow, text: 'slow', tone: 'blue' });
    pointerLabels.push({ nodeId: step.fast, text: 'fast', tone: 'orange' });
    if (step.action === 'finish') {
      pointerLabels.push({ nodeId: step.slow, text: 'middle', tone: 'green' });
      annotations.push({ nodeId: step.slow, label: t('中点', 'middle'), tone: 'green' });
      nullPointers.push('fast.next = None');
    }
    pointerCards = [
      {
        label: 'slow',
        value: scenario.nodes[step.slow].value,
        tone: 'blue',
      },
      {
        label: 'fast',
        value: scenario.nodes[step.fast].value,
        tone: 'orange',
      },
    ];
  } else if (mode === 'cycle') {
    pointerLabels.push({ nodeId: 0, text: 'head', tone: 'gray' });
    if (step.slow !== null) {
      pointerLabels.push({ nodeId: step.slow, text: 'slow', tone: 'blue' });
    }
    if (step.fast !== null) {
      pointerLabels.push({ nodeId: step.fast, text: 'fast', tone: 'orange' });
    }
    if (step.finder !== null) {
      pointerLabels.push({ nodeId: step.finder, text: 'finder', tone: 'gold' });
    }
    if (step.action === 'finish') {
      pointerLabels.push({ nodeId: step.finder, text: 'entry', tone: 'green' });
      annotations.push({ nodeId: step.finder, label: t('入口', 'entry'), tone: 'green' });
    }
    pointerCards = [
      {
        label: 'slow',
        value: step.slow === null ? 'None' : scenario.nodes[step.slow].value,
        tone: 'blue',
      },
      {
        label: step.fast === null ? 'finder' : 'fast',
        value: step.fast === null
          ? (step.finder === null ? 'None' : scenario.nodes[step.finder].value)
          : scenario.nodes[step.fast].value,
        tone: step.fast === null ? 'gold' : 'orange',
      },
    ];
  } else {
    if (step.lead !== null) {
      pointerLabels.push({ nodeId: step.lead, text: 'lead', tone: 'orange' });
    } else {
      nullPointers.push('lead = None');
    }
    pointerLabels.push({ nodeId: step.follow, text: 'follow', tone: 'blue' });
    if (step.action === 'finish') {
      pointerLabels.push({ nodeId: step.follow, text: 'nth', tone: 'green' });
      annotations.push({ nodeId: step.follow, label: t('第 2 个倒数节点', '2nd from end'), tone: 'green' });
    }
    pointerCards = [
      {
        label: 'lead',
        value: step.lead === null ? 'None' : scenario.nodes[step.lead].value,
        tone: 'orange',
      },
      {
        label: 'follow',
        value: scenario.nodes[step.follow].value,
        tone: 'blue',
      },
    ];
  }

  const activeLineLabel = {
    init: t('初始化', 'Initialize'),
    move: t('同步推进', 'Move together'),
    detect: t('检测相遇', 'Detect a meeting'),
    reset: t('重置到头节点', 'Reset one pointer to head'),
    locate: t('同步寻找入口', 'Walk to the entry together'),
    advance: t('先拉开固定距离', 'Create the fixed gap'),
    finish: t('读出答案', 'Read the answer'),
  }[step.activeLine];

  let title = '';
  let detail = '';

  if (mode === 'middle') {
    if (step.action === 'start') {
      title = t('初始化：slow 和 fast 都从头出发', 'Initialize: both slow and fast start at the head');
      detail = t(
        '每一轮 `slow` 走一步，`fast` 走两步。停止时，`slow` 落在中点。',
        'Each loop moves slow by one and fast by two. When the loop stops, slow is at the middle.',
      );
    } else if (step.action === 'move') {
      title = isEnglish
        ? `Iteration ${step.iteration}: slow +1, fast +2`
        : `第 ${step.iteration} 轮：slow 走 1，fast 走 2`;
      detail = isEnglish
        ? `slow is now at ${scenario.nodes[step.slow].value}; fast is now at ${scenario.nodes[step.fast].value}.`
        : `slow 现在在值为 ${scenario.nodes[step.slow].value} 的节点；fast 现在在值为 ${scenario.nodes[step.fast].value} 的节点。`;
    } else {
      title = t('结束：slow 就是中点', 'Finish: slow is the middle');
      detail = t(
        '这条链长度为奇数，因此 `fast.next` 为空时，`slow` 正好落在值为 3 的中点。',
        'This list has odd length, so when `fast.next` becomes empty, `slow` lands exactly on the middle node with value 3.',
      );
    }
  } else if (mode === 'cycle') {
    if (step.action === 'start') {
      title = t('初始化：先只做环检测', 'Initialize: start with cycle detection');
      detail = t(
        '尾节点的 `next` 回到值为 2 的节点，因此链表里存在一个环。',
        'The tail points back to the node with value 2, so the list contains a cycle.',
      );
    } else if (step.action === 'detect') {
      title = isEnglish
        ? `Detect step ${step.iteration}: fast closes the gap`
        : `检测阶段第 ${step.iteration} 轮：fast 在缩小距离`;
      detail = isEnglish
        ? `slow is at ${scenario.nodes[step.slow].value}; fast is at ${scenario.nodes[step.fast].value}.`
        : `slow 在值为 ${scenario.nodes[step.slow].value} 的节点；fast 在值为 ${scenario.nodes[step.fast].value} 的节点。`;
    } else if (step.action === 'meet') {
      title = t('相遇：两指针在环内碰头', 'Meeting point: the two pointers collide inside the cycle');
      detail = t(
        '这里只能说明“有环”，还不能直接说明入口位置。下一步要把一个指针重置回头节点。',
        'This proves the cycle exists, but it does not yet reveal the entry. The next step resets one pointer to the head.',
      );
    } else if (step.action === 'reset') {
      title = t('重置：finder 回到 head', 'Reset: send finder back to head');
      detail = t(
        '之后 `finder` 和 `slow` 都改为每次走一步。再次相遇的位置就是环入口。',
        'From here, finder and slow both move one step at a time. Their next meeting point is the cycle entry.',
      );
    } else if (step.action === 'locate') {
      title = t('再次相遇：定位到环入口', 'Second meeting: locate the cycle entry');
      detail = t(
        '两指针同步前进后一起到达值为 2 的节点，这正是入口。',
        'After moving together, both pointers arrive at the node with value 2, which is the entry.',
      );
    } else {
      title = t('结束：返回环入口', 'Finish: return the cycle entry');
      detail = t(
        'Floyd 的第二阶段把“是否有环”提升为“入口在哪里”。这和数组版 Duplicate Number 完全同构。',
        'Floyd’s second phase upgrades cycle detection into cycle-entry location. The same structure appears in Find the Duplicate Number.',
      );
    }
  } else {
    if (step.action === 'start') {
      title = t('初始化：lead 和 follow 从同一位置出发', 'Initialize: lead and follow start together');
      detail = t(
        '先让 `lead` 单独向前走 `n = 2` 步，之后保持固定间距同步移动。',
        'First move `lead` forward by `n = 2` steps. Then keep the fixed gap while moving both pointers together.',
      );
    } else if (step.action === 'advance') {
      title = t(
        `拉开间距：第 ${step.progress} / ${step.total} 步`,
        `Create the gap: step ${step.progress} / ${step.total}`,
      );
      detail = isEnglish
        ? `lead is now at ${scenario.nodes[step.lead].value}; follow still waits at ${scenario.nodes[step.follow].value}.`
        : `lead 现在在值为 ${scenario.nodes[step.lead].value} 的节点；follow 仍停在值为 ${scenario.nodes[step.follow].value} 的节点。`;
    } else if (step.action === 'move') {
      title = isEnglish
        ? `Move together ${step.iteration}: keep the two-node gap`
        : `同步前进第 ${step.iteration} 轮：保持 2 个节点的间距`;
      detail = step.lead === null
        ? t('lead 已经走到 `None`，因此 follow 停在倒数第 2 个节点。', 'lead has fallen off the list, so follow is now at the 2nd node from the end.')
        : isEnglish
          ? `lead is at ${scenario.nodes[step.lead].value}; follow is at ${scenario.nodes[step.follow].value}.`
          : `lead 在值为 ${scenario.nodes[step.lead].value} 的节点；follow 在值为 ${scenario.nodes[step.follow].value} 的节点。`;
    } else {
      title = t('结束：follow 就是倒数第 2 个节点', 'Finish: follow is the 2nd node from the end');
      detail = t(
        '删除题通常会把这套模板和 dummy 头节点一起用，让跟随指针停在待删节点的前驱。',
        'Removal problems usually combine this gap template with a dummy head so the trailing pointer lands on the predecessor of the node to delete.',
      );
    }
  }

  const modeLabels = {
    middle: t('找中点', 'Find middle'),
    cycle: t('找环入口', 'Cycle entry'),
    gap: t('倒数第 n 个', 'Nth from end'),
  };

  return (
    <section
      className="fast-slow-pointer-visual"
      aria-label={t('快慢指针模式演示', 'Fast-slow pointer walkthrough')}
    >
      <header className="fast-slow-pointer-header">
        <div>
          <p className="eyebrow">{t('快慢指针', 'Fast and slow pointers')}</p>
          <h2>{t('同一种结构，三种相对速度', 'One structure, three relative-speed patterns')}</h2>
          <p>{t(
            '切换模式时，链表保持固定，变化的是指针速度和停止条件。',
            'The lists stay fixed. What changes between modes is the relative speed and the stopping rule.',
          )}</p>
        </div>
        <div className="fast-slow-pointer-mode" role="group" aria-label={t('选择快慢指针模式', 'Choose the fast-slow pointer mode')}>
          {Object.entries(modeLabels).map(([key, label]) => (
            <button
              type="button"
              className={mode === key ? 'active' : ''}
              aria-pressed={mode === key}
              key={key}
              onClick={() => {
                setMode(key);
                setActiveStep(0);
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      <div className={`fast-slow-pointer-step-copy ${step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {scenario.steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="fast-slow-pointer-workspace">
        <div className="fast-slow-pointer-stage-card">
          <div className="fast-slow-pointer-stage-heading">
            <span>{modeLabels[mode]}</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>
          <LinkedListDiagram
            ariaLabel={t('快慢指针节点与指针状态', 'Fast-slow pointer nodes and pointers')}
            annotations={annotations}
            edges={scenario.edges}
            nodes={scenario.nodes}
            nodeHighlights={step.nodeHighlights}
            nullPointers={nullPointers}
            pointerLabels={pointerLabels}
            viewBoxHeight={mode === 'cycle' ? 304 : 280}
          />
          <div className="fast-slow-pointer-state-cards">
            {pointerCards.map((card) => (
              <div className={`fast-slow-pointer-state-card ${card.tone}`} key={card.label}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="fast-slow-pointer-code" aria-label={t('当前模板代码', 'Active template code')}>
          <div className="fast-slow-pointer-code-heading">
            <span>{t('模板代码', 'Template code')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="fast-slow-pointer-code-lines">
            {FAST_SLOW_POINTER_CODE_LINES[mode].map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => (
                  <code key={code}>{code}</code>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="fast-slow-pointer-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={scenario.steps.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择快慢指针演示步骤', 'Select a fast-slow pointer step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(scenario.steps.length - 1, current + 1))}
          disabled={activeStep === scenario.steps.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

function ArrayDuplicateDiagram({ annotations, ariaLabel, pointerLabels, step, zoneLabels }) {
  const labelsByNode = new Map(ARRAY_DUPLICATE_NODES.map((node) => [node.id, []]));
  const annotationsByNode = new Map(ARRAY_DUPLICATE_NODES.map((node) => [node.id, []]));

  pointerLabels.forEach((label) => {
    if (label.nodeId !== null && label.nodeId !== undefined) {
      labelsByNode.get(label.nodeId)?.push(label);
    }
  });

  annotations.forEach((annotation) => {
    if (annotation.nodeId !== null && annotation.nodeId !== undefined) {
      annotationsByNode.get(annotation.nodeId)?.push(annotation);
    }
  });

  return (
    <svg
      aria-label={ariaLabel}
      className="array-duplicate-diagram-svg"
      role="img"
      viewBox="0 0 720 340"
    >
      <defs>
        <marker
          id="array-duplicate-arrow-tail"
          markerHeight="8"
          markerWidth="8"
          orient="auto"
          refX="8"
          refY="4"
          viewBox="0 0 8 8"
        >
          <path d="M 0 0 L 8 4 L 0 8 z" fill="#1d596d" />
        </marker>
        <marker
          id="array-duplicate-arrow-cycle"
          markerHeight="8"
          markerWidth="8"
          orient="auto"
          refX="8"
          refY="4"
          viewBox="0 0 8 8"
        >
          <path d="M 0 0 L 8 4 L 0 8 z" fill="#8a6d0c" />
        </marker>
        <marker
          id="array-duplicate-arrow-cycle-back"
          markerHeight="8"
          markerWidth="8"
          orient="auto"
          refX="8"
          refY="4"
          viewBox="0 0 8 8"
        >
          <path d="M 0 0 L 8 4 L 0 8 z" fill="#8a6d0c" />
        </marker>
      </defs>

      <g className="array-duplicate-zone-labels" aria-hidden="true">
        <text x="148" y="82">{zoneLabels.tail}</text>
        <text x="548" y="74">{zoneLabels.cycle}</text>
      </g>

      {ARRAY_DUPLICATE_EDGES.map((edge) => (
        <path
          className={`array-duplicate-edge ${edge.tone}`}
          d={ARRAY_DUPLICATE_EDGE_PATHS[`${edge.from}-${edge.to}`]}
          key={`${edge.from}-${edge.to}`}
          markerEnd={`url(#array-duplicate-arrow-${edge.tone})`}
        />
      ))}

      {ARRAY_DUPLICATE_NODES.map((node) => {
        const nodeLabels = labelsByNode.get(node.id) ?? [];
        return nodeLabels.map((label, index) => {
          const width = Math.max(58, label.text.length * 7 + 22);
          const labelX = node.x - width / 2;
          const labelY = node.y - 78 - index * 28;
          return (
            <g className={`array-duplicate-pointer ${label.tone}`} key={`${node.id}-${label.text}`}>
              <line x1={node.x} x2={node.x} y1={labelY + 20} y2={node.y - 28} />
              <rect height="20" rx="9" ry="9" width={width} x={labelX} y={labelY} />
              <text dominantBaseline="middle" textAnchor="middle" x={node.x} y={labelY + 10}>
                {label.text}
              </text>
            </g>
          );
        });
      })}

      {ARRAY_DUPLICATE_NODES.map((node) => {
        const nodeAnnotations = annotationsByNode.get(node.id) ?? [];
        return nodeAnnotations.map((annotation, index) => {
          const width = Math.max(82, annotation.label.length * 7 + 24);
          return (
            <g className={`array-duplicate-annotation ${annotation.tone}`} key={`${node.id}-${annotation.label}`}>
              <rect
                height="20"
                rx="9"
                ry="9"
                width={width}
                x={node.x - width / 2}
                y={node.y + 52 + index * 24}
              />
              <text dominantBaseline="middle" textAnchor="middle" x={node.x} y={node.y + 62 + index * 24}>
                {annotation.label}
              </text>
            </g>
          );
        });
      })}

      {ARRAY_DUPLICATE_NODES.map((node) => {
        const highlight = step.nodeHighlights[node.id] ?? '';
        return (
          <g
            className={`array-duplicate-node ${node.region}${node.isEntry ? ' entry' : ''}${highlight ? ` ${highlight}` : ''}`}
            key={node.id}
          >
            <rect height="50" rx="12" ry="12" width="84" x={node.x - 42} y={node.y - 25} />
            <text className="array-duplicate-node-value" dominantBaseline="middle" textAnchor="middle" x={node.x} y={node.y - 2}>
              {node.value}
            </text>
            <text className="array-duplicate-node-meta" dominantBaseline="middle" textAnchor="middle" x={node.x} y={node.y + node.nextLabelOffset}>
              {`nums[${node.id}] = ${node.next}`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function ArrayDuplicateVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const step = ARRAY_DUPLICATE_STEPS[activeStep];

  const pointerLabels = [];
  const annotations = [];

  if (step.slow !== null && step.slow !== undefined) {
    pointerLabels.push({ nodeId: step.slow, text: 'slow', tone: 'blue' });
  }
  if (step.fast !== null && step.fast !== undefined) {
    pointerLabels.push({ nodeId: step.fast, text: 'fast', tone: 'orange' });
  }
  if (step.finder !== null && step.finder !== undefined) {
    pointerLabels.push({ nodeId: step.finder, text: 'finder', tone: 'gold' });
  }

  if (step.action === 'warning') {
    annotations.push({ nodeId: 4, label: t('若直接返回：4', 'returning now: 4'), tone: 'warning' });
    annotations.push({ nodeId: 2, label: t('真正入口：2', 'true entrance: 2'), tone: 'success' });
  }

  if (step.action === 'meet-entry' || step.action === 'finish') {
    annotations.push({ nodeId: 2, label: t('入口 = 重复值 = 2', 'entrance = duplicate = 2'), tone: 'success' });
  }

  const activeLineLabel = {
    init: t('初始化', 'Initialize'),
    phase1: t('第一阶段：找相遇点', 'Phase 1: find a meeting'),
    reset: t('第二阶段：重置 finder', 'Phase 2: reset finder'),
    phase2: t('第二阶段：同步找入口', 'Phase 2: walk to the entrance'),
    finish: t('返回入口', 'Return the entrance'),
  }[step.activeLine];

  let title = '';
  let detail = '';

  if (step.action === 'start') {
    title = t('把数组画成函数图', 'Render the array as a functional graph');
    detail = t(
      '固定示例 `nums = [1, 3, 4, 2, 2]`。尾部是 `0 -> 1 -> 3`，然后进入环 `2 -> 4 -> 2`。',
      'The example is fixed at `nums = [1, 3, 4, 2, 2]`. The tail is `0 -> 1 -> 3`, which feeds into the cycle `2 -> 4 -> 2`.',
    );
  } else if (step.action === 'phase1') {
    title = isEnglish
      ? `Phase 1, round ${step.round}: slow +1, fast +2`
      : `第一阶段第 ${step.round} 轮：slow 走 1，fast 走 2`;
    detail = isEnglish
      ? `The table in the note gives (slow, fast) = (${step.slow}, ${step.fast}) after this round.`
      : `本轮结束后，表格中的位置正是 (slow, fast) = (${step.slow}, ${step.fast})。`;
  } else if (step.action === 'meet') {
    title = t('第一阶段结束：相遇点在 4', 'Phase 1 ends: the meeting point is 4');
    detail = t(
      '`slow` 和 `fast` 第一次在节点 `4` 相遇。这一步只说明“有环”，还没有得到真正入口。',
      '`slow` and `fast` meet for the first time at node `4`. This only proves a cycle exists; it does not yet give the entrance.',
    );
  } else if (step.action === 'warning') {
    title = t('这里直接返回会错', 'Returning here would be wrong');
    detail = t(
      '如果把第一阶段的相遇点 `4` 当答案返回，就会错过真正的重复值 `2`。这正是第二阶段必须存在的原因。',
      'If you return the phase-one meeting point `4` directly, you miss the actual duplicate `2`. This is exactly why the second phase is necessary.',
    );
  } else if (step.action === 'reset') {
    title = t('第二阶段开始：finder 回到 0', 'Phase 2 starts: reset finder to 0');
    detail = t(
      '`slow` 留在相遇点 `4`，`finder` 回到起点 `0`。接下来两者都改成每轮走一步。',
      '`slow` stays at the meeting point `4`, and `finder` resets to the start `0`. From here, both move one step per round.',
    );
  } else if (step.action === 'phase2') {
    title = isEnglish
      ? `Phase 2, round ${step.round}: walk one step each`
      : `第二阶段第 ${step.round} 轮：两者都走 1 步`;
    detail = isEnglish
      ? `This matches the note's second table: (finder, slow) = (${step.finder}, ${step.slow}).`
      : `这一步与文中的第二张表一致： (finder, slow) = (${step.finder}, ${step.slow})。`;
  } else if (step.action === 'meet-entry') {
    title = t('恰好在第 3 轮到达入口 2', 'They reach the entrance 2 exactly on round 3');
    detail = t(
      '`s = μ = 3` 时，`finder` 和 `slow` 同时落在真正的环入口 `2`。',
      'At `s = μ = 3`, `finder` and `slow` land on the true cycle entrance `2` at the same time.',
    );
  } else {
    title = t('正确答案：返回 2', 'Correct answer: return 2');
    detail = t(
      '第二阶段把“相遇点 `4`”修正为“入口 `2`”。数组版 Duplicate Number 返回的是入口，而不是第一阶段的相遇点。',
      'The second phase corrects “meeting point `4`” into “entrance `2`.” In Find The Duplicate Number, the returned value is the entrance, not the first meeting point.',
    );
  }

  const phaseValue = step.phase === 1 ? t('第一阶段', 'Phase 1') : t('第二阶段', 'Phase 2');
  const countValue = step.phase === 1
    ? (step.k === 0 ? 'k = 0' : `k = ${step.k}`)
    : `s = ${step.s} / μ = 3`;

  const stateCards = [
    { label: 'slow', value: String(step.slow), tone: 'blue' },
    { label: 'fast', value: step.fast === null ? '—' : String(step.fast), tone: 'orange' },
    { label: 'finder', value: step.finder === null ? '—' : String(step.finder), tone: 'gold' },
    {
      label: step.phase === 1 ? t('阶段 / k', 'phase / k') : t('阶段 / s', 'phase / s'),
      value: `${phaseValue} · ${countValue}`,
      tone: step.action === 'warning' ? 'warning' : step.action === 'finish' ? 'success' : 'gray',
    },
  ];

  return (
    <section
      className="array-duplicate-visual"
      aria-label={t('数组版 Floyd 第二阶段必要性演示', 'Why Floyd needs phase two on the array duplicate graph')}
    >
      <header className="array-duplicate-header">
        <div>
          <p className="eyebrow">{t('数组判重', 'Find The Duplicate Number')}</p>
          <h2>{t('第一阶段只会到相遇点，不会直接到入口', 'Phase 1 reaches a meeting point, not necessarily the entrance')}</h2>
          <p>{t(
            '这个固定例子只演示一个 correctness gotcha：`slow = fast` 时停在 `4`，真正要返回的是第二阶段找到的入口 `2`。',
            'This fixed example isolates one correctness gotcha: the `slow = fast` stop is at `4`, but the value to return is the entrance `2` found in phase 2.',
          )}</p>
        </div>
        <div className="array-duplicate-summary" aria-label={t('示例参数', 'Example parameters')}>
          <span>nums = [1, 3, 4, 2, 2]</span>
          <span>μ = 3</span>
          <span>λ = 2</span>
          <span>{t('入口 = 2', 'entrance = 2')}</span>
        </div>
      </header>

      <div className={`array-duplicate-step ${step.headerTone}`} aria-live="polite">
        <span>{activeStep + 1} / {ARRAY_DUPLICATE_STEPS.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="array-duplicate-workspace">
        <div className="array-duplicate-stage-card">
          <div className="array-duplicate-stage-heading">
            <span>{t('隐式图状态', 'Implicit graph state')}</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>

          <div className="array-duplicate-diagram-shell">
            <ArrayDuplicateDiagram
              annotations={annotations}
              ariaLabel={t('数组隐式图与指针状态', 'Implicit array graph and pointer state')}
              pointerLabels={pointerLabels}
              step={step}
              zoneLabels={{ tail: t('尾部', 'tail'), cycle: t('环', 'cycle') }}
            />
          </div>

          <div className="array-duplicate-state-cards">
            {stateCards.map((card) => (
              <div className={`array-duplicate-state-card ${card.tone}`} key={card.label}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
              </div>
            ))}
          </div>
        </div>

        <div className="array-duplicate-code" aria-label={t('当前代码', 'Current code')}>
          <div className="array-duplicate-code-heading">
            <span>{t('模板代码', 'Template code')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="array-duplicate-code-lines">
            {ARRAY_DUPLICATE_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => (
                  <code key={code}>{code}</code>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="array-duplicate-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={ARRAY_DUPLICATE_STEPS.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择数组版 Floyd 演示步骤', 'Select an array-duplicate demo step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(ARRAY_DUPLICATE_STEPS.length - 1, current + 1))}
          disabled={activeStep === ARRAY_DUPLICATE_STEPS.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

function LRUCacheVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const step = LRU_CACHE_STEPS[activeStep];
  const operation = LRU_CACHE_OPERATIONS[step.operationIndex];
  const operationCall = operation.type === 'put'
    ? `put(${operation.key}, ${operation.value})`
    : `get(${operation.key})`;
  const operationResult = operation.type === 'put'
    ? t('(无返回值)', '(no return value)')
    : String(operation.result);
  const cacheValues = new Map(step.cacheEntries);
  const valueForKey = (key) => cacheValues.get(key)
    ?? LRU_CACHE_OPERATIONS.find((item) => item.type === 'put' && item.key === key)?.value;
  const mruToLru = [...step.order].reverse();

  const copyByAction = {
    'put-lookup': {
      title: step.lookupTone === 'hit'
        ? t('哈希查找命中已有键', 'Hash lookup finds the existing key')
        : t('哈希查找未找到该键', 'Hash lookup does not find the key'),
      detail: step.lookupTone === 'hit'
        ? t('先从双向链表删除旧节点，再用新值创建并插入节点。', 'Remove the old node from the list before creating and inserting the replacement.')
        : t('该键尚不在 cache 中，下一步创建新节点。', 'The key is not in cache, so the next step creates a fresh node.'),
    },
    'put-remove-existing': {
      title: t('删除已有节点', 'Remove the existing node'),
      detail: t('哈希映射暂时保留，节点已从双向链表断开。', 'The hash entry remains temporarily while the node is detached from the doubly linked list.'),
    },
    'put-new-node': {
      title: t('创建新节点', 'Create a fresh node'),
      detail: t('新节点当前还未接入链表，也还未写入哈希表。', 'The fresh node is not linked into the list or stored in the hash map yet.'),
    },
    'put-map': {
      title: t('写入 key → node 映射', 'Store the key → node mapping'),
      detail: t('哈希表已经指向新节点；节点仍处于未接入链表的中间状态。', 'The hash map now points to the fresh node, which is still detached from the list.'),
    },
    'put-insert': {
      title: t('insert_before_tail：插到 MRU 端', 'insert_before_tail: insert at the MRU end'),
      detail: t('节点被插到 right 前面，因此 right.prev 再次指向最新访问节点。', 'The node is inserted immediately before right, so right.prev again identifies the most recent node.'),
    },
    'capacity-ok': {
      title: t('容量检查通过，无需淘汰', 'Capacity check passes; no eviction'),
      detail: t('cache 大小没有超过 capacity = 2，本次 put 完成。', 'The cache size does not exceed capacity = 2, so this put is complete.'),
    },
    'capacity-over': {
      title: t('容量超限，必须淘汰一个节点', 'Capacity exceeded; one node must be evicted'),
      detail: t('插入发生在检查之前，所以此刻短暂出现 3 个节点。', 'Insertion happens before the check, so the cache temporarily contains three nodes.'),
    },
    'evict-target': {
      title: t(`选择 left.next：键 ${step.evictKey}`, `Select left.next: key ${step.evictKey}`),
      detail: t('left.next 是 LRU 节点；不能从 right.prev 淘汰。', 'left.next is the LRU node; eviction must not use right.prev.'),
    },
    'evict-remove': {
      title: t(`从链表删除 LRU 节点 ${step.evictKey}`, `Remove LRU node ${step.evictKey} from the list`),
      detail: t('remove(lru) 已修复前后指针，但哈希映射还存在，下一行才删除。', 'remove(lru) has repaired both neighbor links, but the hash entry remains until the next line.'),
    },
    'evict-delete': {
      title: t(`从哈希表删除键 ${step.evictedKey}`, `Delete key ${step.evictedKey} from the hash map`),
      detail: t('链表与哈希表再次同步，淘汰完成。', 'The list and hash map are synchronized again; eviction is complete.'),
    },
    'get-hit': {
      title: t(`哈希查找命中键 ${operation.key}`, `Hash lookup hits key ${operation.key}`),
      detail: t('命中后不能直接返回；还要把该节点移动到 MRU 端。', 'A hit cannot return immediately; the node must first move to the MRU end.'),
    },
    'get-miss': {
      title: t(`哈希查找未命中键 ${operation.key}`, `Hash lookup misses key ${operation.key}`),
      detail: t('cache 中没有该键，立即返回 -1，链表顺序不变。', 'The key is absent, so get returns -1 immediately and the list order stays unchanged.'),
    },
    'get-remove': {
      title: t(`remove：断开键 ${operation.key} 的节点`, `remove: detach the node for key ${operation.key}`),
      detail: t('哈希表仍指向这个临时断开的节点。', 'The hash map still points to this temporarily detached node.'),
    },
    'get-insert': {
      title: t('insert_before_tail：移动到 MRU 端', 'insert_before_tail: move to the MRU end'),
      detail: t('同一个节点被重新插到 right 前面，哈希映射无需更改。', 'The same node is reinserted immediately before right; the hash entry does not change.'),
    },
    'get-return': {
      title: t(`返回节点值 ${operation.result}`, `Return node value ${operation.result}`),
      detail: t('返回前的移动已经完成，当前顺序就是本次 get 之后的顺序。', 'The move completes before returning, so the displayed order is the final order after this get.'),
    },
  };

  const activeCopy = copyByAction[step.action];
  const activeLineLabel = {
    'put-lookup': t('检查键是否存在', 'Check whether the key exists'),
    'put-remove-existing': t('删除旧节点', 'Remove the old node'),
    'put-node': t('创建节点', 'Create the node'),
    'put-cache': t('更新哈希表', 'Update the hash map'),
    'put-insert': t('插到 MRU 端', 'Insert at the MRU end'),
    'put-capacity': t('检查容量', 'Check capacity'),
    'put-lru': t('读取 left.next', 'Read left.next'),
    'put-remove-lru': t('从链表淘汰', 'Evict from the list'),
    'put-delete': t('从哈希表淘汰', 'Evict from the hash map'),
    'get-node': t('哈希查找命中', 'Hash lookup hits'),
    'get-miss': t('哈希查找未命中', 'Hash lookup misses'),
    'get-remove': t('从链表断开', 'Detach from the list'),
    'get-insert': t('重新插到 MRU 端', 'Reinsert at the MRU end'),
    'get-return': t('返回节点值', 'Return the node value'),
  }[step.activeLine];

  const stepTone = step.action.startsWith('evict') || step.action === 'capacity-over'
    ? 'eviction'
    : step.action === 'get-miss'
      ? 'miss'
      : step.movedKey !== undefined
        ? 'moved'
        : 'normal';

  const renderNode = (key, className = '') => (
    <div className={`lru-cache-node ${className}`.trim()} key={key}>
      <span>{t('键', 'key')} {key}</span>
      <strong>{t('值', 'value')} {valueForKey(key)}</strong>
    </div>
  );

  return (
    <section className="lru-cache-visual" aria-label={t('LRU 缓存逐步演示', 'LRU cache walkthrough')}>
      <header className="lru-cache-header">
        <div>
          <p className="eyebrow">LRU Cache · LeetCode 146</p>
          <h2>{t('双向链表顺序与哈希表同步变化', 'Doubly linked-list order and hash map in sync')}</h2>
          <p>{t(
            '固定执行 9 个操作。链表从左到右始终是 LRU 端到 MRU 端，淘汰只删除 left.next。',
            'The walkthrough runs nine fixed operations. The list always reads from the LRU end to the MRU end, and eviction only removes left.next.',
          )}</p>
        </div>
        <div className="lru-cache-summary" aria-label={t('缓存参数', 'Cache parameters')}>
          <span>capacity = {LRU_CACHE_CAPACITY}</span>
          <span>{t('当前大小', 'current size')} = {step.cacheEntries.length}</span>
          <span>left.next = {step.order[0] ?? 'right'}</span>
          <span>right.prev = {step.order.at(-1) ?? 'left'}</span>
        </div>
      </header>

      <div className={`lru-cache-step ${stepTone}`} aria-live="polite">
        <span>{activeStep + 1} / {LRU_CACHE_STEPS.length}</span>
        <strong>{activeCopy.title}</strong>
        <p>{activeCopy.detail}</p>
      </div>

      <div className="lru-cache-operation-bar">
        <span>{t('操作', 'operation')} {step.operationIndex + 1} / {LRU_CACHE_OPERATIONS.length}</span>
        <code>{operationCall}</code>
        <strong>→ {operationResult}</strong>
        <i className={operation.type === 'get' ? (operation.result === -1 ? 'miss' : 'hit') : 'put'}>
          {operation.type === 'get'
            ? (operation.result === -1 ? t('未命中', 'miss') : t('命中', 'hit'))
            : 'put'}
        </i>
      </div>

      <div className="lru-cache-workspace">
        <div className="lru-cache-stage-card">
          <div className="lru-cache-stage-heading">
            <span>{t('双向链表与哈希表', 'Doubly linked list and hash map')}</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>

          <div className="lru-cache-order-pill">
            <span>{t('操作后顺序', 'Order after this micro-step')}</span>
            <code>MRU → LRU [{mruToLru.join(', ')}]</code>
          </div>

          <div className="lru-cache-list-shell">
            <div className="lru-cache-list-direction">
              <span>{t('较久未使用', 'less recent')}</span>
              <strong>{t('LRU 端 → MRU 端', 'LRU end → MRU end')}</strong>
              <span>{t('最近使用', 'more recent')}</span>
            </div>
            <div className="lru-cache-list" aria-label={t('LRU 到 MRU 的双向链表', 'Doubly linked list from LRU to MRU')}>
              <div className="lru-cache-sentinel left">
                <strong>left</strong>
                <span>{t('LRU 哨兵', 'LRU sentinel')}</span>
              </div>
              <span className="lru-cache-edge" aria-hidden="true">⇄</span>
              {step.order.map((key) => (
                <Fragment key={key}>
                  {renderNode(
                    key,
                    key === step.evictKey
                      ? 'evict-target'
                      : key === step.movedKey
                        ? 'moved'
                        : '',
                  )}
                  <span className="lru-cache-edge" aria-hidden="true">⇄</span>
                </Fragment>
              ))}
              <div className="lru-cache-sentinel right">
                <strong>right</strong>
                <span>{t('MRU 哨兵', 'MRU sentinel')}</span>
              </div>
            </div>
          </div>

          {(step.detachedKey !== undefined || step.evictedKey !== undefined) && (
            <div className={`lru-cache-detached ${step.evictKey !== undefined || step.evictedKey !== undefined ? 'eviction' : ''}`}>
              <span>{step.evictedKey !== undefined
                ? t('已淘汰', 'evicted')
                : step.evictKey !== undefined
                  ? t('已从链表移除，等待删除哈希映射', 'removed from list; hash deletion pending')
                  : t('临时断开的节点', 'temporarily detached node')}</span>
              {renderNode(step.detachedKey ?? step.evictedKey, step.evictedKey !== undefined ? 'evicted' : 'detached')}
            </div>
          )}

          <div className="lru-cache-map-panel">
            <div className="lru-cache-map-heading">
              <strong>{t('哈希表 cache', 'Hash map cache')}</strong>
              {step.lookupKey !== undefined && (
                <span className={step.lookupTone}>
                  {t('查找', 'lookup')} {step.lookupKey} → {step.lookupTone === 'hit' ? t('命中', 'hit') : t('未命中', 'miss')}
                </span>
              )}
            </div>
            <div className="lru-cache-map-entries">
              {step.cacheEntries.length === 0 ? (
                <span className="empty">{t('空', 'empty')}</span>
              ) : step.cacheEntries.map(([key, value]) => (
                <div
                  className={`${key === step.evictKey ? 'evict-target' : ''}${key === step.lookupKey ? ` ${step.lookupTone}` : ''}`.trim()}
                  key={key}
                >
                  <code>{key}</code>
                  <span>→</span>
                  <strong>Node({key}, {value})</strong>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="lru-cache-code" aria-label={t('当前 LRU 代码', 'Current LRU code')}>
          <div className="lru-cache-code-heading">
            <span>{t('参考实现', 'Reference implementation')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="lru-cache-code-lines">
            {LRU_CACHE_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="lru-cache-controls">
        <button
          type="button"
          onClick={() => setActiveStep((current) => Math.max(0, current - 1))}
          disabled={activeStep === 0}
        >
          ← {t('上一步', 'Previous')}
        </button>
        <input
          type="range"
          min="0"
          max={LRU_CACHE_STEPS.length - 1}
          value={activeStep}
          onChange={(event) => setActiveStep(Number(event.target.value))}
          aria-label={t('选择 LRU 缓存演示步骤', 'Select an LRU cache walkthrough step')}
        />
        <button
          type="button"
          className="primary"
          onClick={() => setActiveStep((current) => Math.min(LRU_CACHE_STEPS.length - 1, current + 1))}
          disabled={activeStep === LRU_CACHE_STEPS.length - 1}
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

function TreeTraversalDiagram({ step, t }) {
  const visited = new Set([...step.output, ...step.buffer]);
  const frontier = new Set(step.container);
  const nodeMap = new Map(TREE_TRAVERSAL_NODES.map((node) => [node.id, node]));

  return (
    <svg
      aria-label={t('固定二叉树的遍历状态', 'Traversal state on the fixed binary tree')}
      className="tree-traversal-diagram"
      role="img"
      viewBox="0 0 700 292"
    >
      {TREE_TRAVERSAL_EDGES.map((edge) => {
        const source = nodeMap.get(edge.from);
        const target = nodeMap.get(edge.to);
        return (
          <line
            className="tree-traversal-edge"
            key={`${edge.from}-${edge.to}`}
            x1={source.x}
            x2={target.x}
            y1={source.y + 27}
            y2={target.y - 27}
          />
        );
      })}
      {TREE_TRAVERSAL_NODES.map((node) => {
        const classes = [
          'tree-traversal-node',
          visited.has(node.id) ? 'visited' : '',
          frontier.has(node.id) ? 'frontier' : '',
          step.current === node.id ? 'current' : '',
        ].filter(Boolean).join(' ');
        return (
          <g className={classes} key={node.id}>
            <rect height="54" rx="14" ry="14" width="70" x={node.x - 35} y={node.y - 27} />
            <text dominantBaseline="middle" textAnchor="middle" x={node.x} y={node.y}>{node.value}</text>
          </g>
        );
      })}
    </svg>
  );
}

function TreeTraversalVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('preorder');
  const [execMode, setExecMode] = useState('iterative');
  const [activeStep, setActiveStep] = useState(0);
  const isRecursive = execMode === 'recursive' && mode !== 'level';
  const scenario = isRecursive ? TREE_TRAVERSAL_RECURSIVE_SCENARIOS[mode] : TREE_TRAVERSAL_SCENARIOS[mode];
  const step = scenario.steps[activeStep];
  const isQueue = scenario.container === 'queue';
  const codeLines = isRecursive ? TREE_TRAVERSAL_RECURSIVE_CODE_LINES[mode] : TREE_TRAVERSAL_CODE_LINES[mode];
  const modeLabels = {
    preorder: t('前序', 'Preorder'),
    inorder: t('中序', 'Inorder'),
    postorder: t('后序', 'Postorder'),
    level: t('层序', 'Level order'),
  };
  const expected = {
    preorder: [1, 2, 4, 5, 3, 6],
    inorder: [4, 2, 5, 1, 3, 6],
    postorder: [4, 5, 2, 6, 3, 1],
    level: [1, 2, 3, 4, 5, 6],
  }[mode];
  const activeLineLabel = {
    init: t('初始化容器', 'Initialize the container'),
    loop: t('检查容器', 'Check the container'),
    pop: isQueue ? t('队首出队', 'Dequeue the front') : t('栈顶出栈', 'Pop the stack'),
    visit: mode === 'postorder' && !isRecursive ? t('写入反向缓冲区', 'Append to the reverse buffer') : t('访问节点', 'Visit the node'),
    'push-right': t('右子节点入栈', 'Push the right child'),
    'push-left': t('左子节点入栈', 'Push the left child'),
    descend: t('沿左链入栈', 'Push the left chain'),
    'move-right': t('转向右子树', 'Move to the right subtree'),
    'add-left': t('左子节点入队', 'Enqueue the left child'),
    'add-right': t('右子节点入队', 'Enqueue the right child'),
    call: t('进入递归调用', 'Enter the recursive call'),
    'recurse-left': t('递归处理左子树', 'Recurse into the left subtree'),
    'recurse-right': t('递归处理右子树', 'Recurse into the right subtree'),
    return: t('返回上一层调用', 'Return to the caller'),
    finish: mode === 'postorder' && !isRecursive ? t('反转缓冲区', 'Reverse the buffer') : t('返回序列', 'Return the sequence'),
  }[step.activeLine];
  const actionCopy = {
    start: {
      title: isRecursive
        ? t('从根节点开始第一次调用', 'Start with the first call on the root')
        : isQueue ? t('根节点进入队列', 'Place the root in the queue') : t('初始化显式栈', 'Initialize the explicit stack'),
      detail: isRecursive
        ? t('调用栈保存尚未返回的调用，输出序列当前为空。', 'The call stack holds calls that have not returned yet; the output is empty.')
        : t('容器保存后续需要处理的节点，输出序列当前为空。', 'The container holds nodes that still need processing; the output is empty.'),
    },
    pop: {
      title: isQueue ? t('取出队首节点', 'Remove the front node') : t('弹出栈顶节点', 'Pop the top node'),
      detail: t('橙色节点是当前节点。下一步根据遍历顺序访问或扩展它。', 'The orange node is current. The next step visits or expands it according to the traversal order.'),
    },
    call: {
      title: t('进入新的递归调用', 'Enter a new recursive call'),
      detail: t('橙色节点是当前调用处理的节点；蓝色区域是仍未返回的调用序列。', 'The orange node is what the current call is processing; the blue area shows calls that have not returned yet.'),
    },
    visit: {
      title: t('把当前值追加到输出', 'Append the current value to the output'),
      detail: t('绿色节点已经被访问，输出序列按执行顺序增长。', 'Green nodes have been visited, and the output grows in execution order.'),
    },
    'visit-buffer': {
      title: t('写入 root-right-left 反向缓冲区', 'Append to the root-right-left reverse buffer'),
      detail: t('后序的最终结果会在遍历结束后把这个缓冲区整体反转。', 'Postorder reverses this complete buffer after the traversal finishes.'),
    },
    'push-right': {
      title: mode === 'preorder' ? t('先压入右子节点', 'Push the right child first') : t('压入右子节点', 'Push the right child'),
      detail: mode === 'preorder'
        ? t('随后再压入左子节点，左子节点会先出栈。', 'The left child is pushed afterward, so it will pop first.')
        : t('后序的修改前序要求右子节点先出栈。', 'The modified preorder for postorder needs the right child to pop first.'),
    },
    'push-left': {
      title: mode === 'preorder' ? t('再压入左子节点', 'Push the left child second') : t('先压入左子节点', 'Push the left child first'),
      detail: mode === 'preorder'
        ? t('栈是后进先出，因此下一次优先处理左子树。', 'The stack is last-in, first-out, so the left subtree is processed next.')
        : t('右子节点随后入栈并先处理，缓冲区顺序保持 root-right-left。', 'The right child is pushed next and processed first, preserving root-right-left in the buffer.'),
    },
    descend: {
      title: t('当前节点入栈并继续向左', 'Push the current node and continue left'),
      detail: t('中序遍历先保存祖先，直到当前指针到达空节点。', 'Inorder saves ancestors until the current pointer reaches an empty child.'),
    },
    'move-right': {
      title: t('访问后转向右子树', 'Move to the right subtree after visiting'),
      detail: t('右子树仍按相同规则先走到最左端。', 'The same rule descends to the leftmost node of the right subtree.'),
    },
    'add-left': {
      title: t('左子节点加入队尾', 'Enqueue the left child'),
      detail: t('同一层的节点会在下一层节点之前出队。', 'Nodes on the current level leave the queue before nodes on the next level.'),
    },
    'add-right': {
      title: t('右子节点加入队尾', 'Enqueue the right child'),
      detail: t('左子节点先入队，因此同一层保持从左到右的顺序。', 'The left child entered first, preserving left-to-right order within the level.'),
    },
    'recurse-left': {
      title: t('调用左子树的递归', 'Call the recursion on the left subtree'),
      detail: t('当前调用会等待左子树的递归完全返回，再继续往下执行。', 'The current call waits for the left-subtree recursion to fully return before continuing.'),
    },
    'recurse-right': {
      title: t('调用右子树的递归', 'Call the recursion on the right subtree'),
      detail: t('左子树已经返回；现在对右子树做同样的递归调用。', 'The left subtree has already returned; the same recursive call now runs on the right subtree.'),
    },
    return: {
      title: t('当前调用返回上一层', 'The current call returns to its caller'),
      detail: t('这个节点从调用栈中移除，控制权交还给调用它的父节点。', 'This node is removed from the call stack, and control returns to the parent call that invoked it.'),
    },
    finish: {
      title: t(`完成：${expected.join(' → ')}`, `Complete: ${expected.join(' → ')}`),
      detail: mode === 'postorder' && !isRecursive
        ? t('反向缓冲区整体翻转后得到 left-right-root 的后序序列。', 'Reversing the complete buffer produces the left-right-root postorder sequence.')
        : isRecursive
          ? t('所有调用都已返回，输出序列是最终结果。', 'Every call has returned, and the output sequence is the final result.')
          : t('容器为空，所有节点都已按当前模式访问。', 'The container is empty, and every node has been visited in the selected order.'),
    },
  };
  const copy = actionCopy[step.action];
  const displayContainer = isQueue ? step.container : [...step.container].reverse();
  const sequence = (values) => values.length ? values.join(' → ') : '—';

  return (
    <section className="tree-traversal-visual" aria-label={t('二叉树遍历逐步演示', 'Binary-tree traversal walkthrough')}>
      <header className="tree-traversal-header">
        <div>
          <p className="eyebrow">{t('遍历模板', 'Traversal templates')}</p>
          <h2>{t('同一棵树的遍历：递归与迭代对照', 'Traversals on one tree: recursive vs. iterative')}</h2>
          <p>{t(
            '切换遍历方式后，节点结构保持固定；再切换递归或迭代，容器规则、访问时机和输出顺序随之变化。',
            'The node structure stays fixed when you switch traversal order; switching recursive vs. iterative then changes the container rule, visit timing, and output order.',
          )}</p>
        </div>
        <div className="tree-traversal-mode" role="group" aria-label={t('选择遍历模式', 'Choose a traversal mode')}>
          {Object.entries(modeLabels).map(([key, label]) => (
            <button
              aria-pressed={mode === key}
              className={mode === key ? 'active' : ''}
              key={key}
              onClick={() => {
                setMode(key);
                setActiveStep(0);
                if (key === 'level') setExecMode('iterative');
              }}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      {mode !== 'level' && (
        <div className="tree-traversal-exec-row">
          <span>{t('执行方式', 'Execution style')}</span>
          <div className="tree-traversal-mode" role="group" aria-label={t('选择递归或迭代', 'Choose recursive or iterative')}>
            <button
              aria-pressed={execMode === 'recursive'}
              className={execMode === 'recursive' ? 'active' : ''}
              onClick={() => { setExecMode('recursive'); setActiveStep(0); }}
              type="button"
            >
              {t('递归', 'Recursive')}
            </button>
            <button
              aria-pressed={execMode === 'iterative'}
              className={execMode === 'iterative' ? 'active' : ''}
              onClick={() => { setExecMode('iterative'); setActiveStep(0); }}
              type="button"
            >
              {t('迭代', 'Iterative')}
            </button>
          </div>
        </div>
      )}

      <div className={`tree-traversal-step ${step.action === 'finish' ? 'finish' : step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {scenario.steps.length}</span>
        <strong>{copy.title}</strong>
        <p>{copy.detail}</p>
      </div>

      <div className="tree-traversal-workspace">
        <div className="tree-traversal-stage-card">
          <div className="tree-traversal-stage-heading">
            <span>{modeLabels[mode]}</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>
          <TreeTraversalDiagram step={step} t={t} />
          <div className="tree-traversal-container-panel">
            <div>
              <span>
                {isQueue
                  ? t('队列：队首在左', 'Queue: front at left')
                  : isRecursive
                    ? t('调用栈：栈顶在左', 'Call stack: top at left')
                    : t('栈：栈顶在左', 'Stack: top at left')}
              </span>
              <strong>{isQueue ? 'FIFO' : 'LIFO'}</strong>
            </div>
            <div className="tree-traversal-container">
              {displayContainer.length ? displayContainer.map((nodeId, index) => (
                <span className={index === 0 ? 'next' : ''} key={`${nodeId}-${index}`}>
                  {nodeId}
                  {index === 0 && <i>{isQueue ? t('队首', 'front') : t('栈顶', 'top')}</i>}
                </span>
              )) : <em>{t('空', 'empty')}</em>}
            </div>
          </div>
          <div className="tree-traversal-state-cards">
            <div><span>{t('当前节点', 'current')}</span><strong>{step.current ?? '—'}</strong></div>
            {mode === 'postorder' && !isRecursive && <div className="buffer"><span>{t('反向缓冲区', 'reverse buffer')}</span><strong>{sequence(step.buffer)}</strong></div>}
            <div className="output"><span>{t('输出', 'output')}</span><strong>{sequence(step.output)}</strong></div>
          </div>
        </div>

        <div className="tree-traversal-code" aria-label={t('当前遍历代码', 'Current traversal code')}>
          <div className="tree-traversal-code-heading">
            <span>{isRecursive ? t('递归模板', 'Recursive template') : t('迭代模板', 'Iterative template')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="tree-traversal-code-lines">
            {codeLines.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="tree-traversal-legend">
        <span><i className="current" />{t('当前节点', 'current')}</span>
        <span><i className="frontier" />{isRecursive ? t('在调用栈中', 'on the call stack') : t('栈或队列中', 'in stack or queue')}</span>
        <span><i className="visited" />{t('已访问', 'visited')}</span>
      </div>

      <div className="tree-traversal-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择树遍历演示步骤', 'Select a tree-traversal step')}
          max={scenario.steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === scenario.steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(scenario.steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

function AVLRotationDiagram({ step, t }) {
  const nodeMap = new Map(step.nodes.map((node) => [node.id, node]));
  return (
    <svg aria-label={t('AVL 旋转中的节点位置', 'Node positions during the AVL rotation')} className="avl-rotation-diagram" role="img" viewBox="0 0 700 292">
      {step.edges.map((edge) => {
        const source = nodeMap.get(edge.from);
        const target = nodeMap.get(edge.to);
        return (
          <g className={edge.label ? 'avl-rotation-edge transfer' : 'avl-rotation-edge'} key={`${edge.from}-${edge.to}`}>
            <line x1={source.x} x2={target.x} y1={source.y + 27} y2={target.y - 27} />
            {edge.label && <text textAnchor="middle" x={(source.x + target.x) / 2 + 18} y={(source.y + target.y) / 2}>{edge.label}</text>}
          </g>
        );
      })}
      {step.nodes.map((node) => (
        <g className={`avl-rotation-node ${step.highlights[node.id] ?? ''}`} key={node.id}>
          <rect height="54" rx="14" ry="14" width="76" x={node.x - 38} y={node.y - 27} />
          <text dominantBaseline="middle" textAnchor="middle" x={node.x} y={node.y}>{node.value}</text>
        </g>
      ))}
    </svg>
  );
}

function AVLRotationVisual() {
  const { t } = useUiCopy();
  const [mode, setMode] = useState('ll');
  const [activeStep, setActiveStep] = useState(0);
  const scenario = AVL_ROTATION_SCENARIOS[mode];
  const step = scenario.steps[activeStep];
  const modeLabels = {
    ll: t('LL：单右旋', 'LL: single right rotation'),
    lr: t('LR：先左后右', 'LR: left then right'),
  };
  const lineLabels = {
    detect: t('识别失衡类型', 'Identify the imbalance'),
    promote: t('保存新根与转移子树', 'Save the new root and transfer subtree'),
    rotate: t('执行右旋并重新连接', 'Rotate right and reconnect'),
    'left-rotate': t('先对左子节点左旋', 'First rotate the left child left'),
    'right-rotate': t('再对失衡节点右旋', 'Then rotate the unbalanced node right'),
    finish: t('返回新的子树根', 'Return the new subtree root'),
  };
  const copy = {
    ll: {
      detect: {
        title: t('LL 失衡：30 的左侧高出两层', 'LL imbalance: the left side of 30 is two levels taller'),
        detail: t('较重路径是 30 → 20 → 10，需要围绕 30 做一次右旋。', 'The heavy path is 30 → 20 → 10, so one right rotation around 30 is required.'),
      },
      promote: {
        title: t('20 将成为新的子树根', '20 will become the new subtree root'),
        detail: t('旋转前先保存 20.right，也就是标为 T2 的节点 25。', 'Before rotating, save 20.right: node 25, labeled T2.'),
      },
      rotate: {
        title: t('30 下移，T2 接到 30.left', '30 moves down, and T2 attaches to 30.left'),
        detail: t('20.right 指向 30；25 保持有序关系并成为 30 的左子节点。', '20.right points to 30; node 25 preserves ordering as the left child of 30.'),
      },
      finish: {
        title: t('LL 修复完成：新根是 20', 'LL repair complete: the new root is 20'),
        detail: t('中序顺序仍为 10, 20, 25, 30，子树高度差恢复到允许范围。', 'The inorder sequence remains 10, 20, 25, 30, and the subtree height difference returns to the allowed range.'),
      },
    },
    lr: {
      detect: {
        title: t('LR 失衡：较重路径先向左，再向右', 'LR imbalance: the heavy path goes left, then right'),
        detail: t('路径 30 → 10 → 20 需要两次旋转。', 'The path 30 → 10 → 20 requires two rotations.'),
      },
      'left-rotate': {
        title: t('第一步：围绕 10 左旋', 'Step 1: rotate left around 10'),
        detail: t('20 成为 30 的左子节点，10 成为 20 的左子节点。LR 已转换为 LL。', '20 becomes the left child of 30, and 10 becomes the left child of 20. The LR case is now an LL case.'),
      },
      'right-rotate': {
        title: t('第二步：围绕 30 右旋', 'Step 2: rotate right around 30'),
        detail: t('20 上移为子树根，10 和 30 分别位于左右两侧。', '20 moves up as the subtree root, with 10 on the left and 30 on the right.'),
      },
      finish: {
        title: t('LR 修复完成：新根是 20', 'LR repair complete: the new root is 20'),
        detail: t('中序顺序仍为 10, 20, 30，两次局部旋转保持 BST 有序性质。', 'The inorder sequence remains 10, 20, 30; both local rotations preserve BST ordering.'),
      },
    },
  }[mode][step.action];

  return (
    <section className="avl-rotation-visual" aria-label={t('AVL 单旋与双旋逐步演示', 'AVL single- and double-rotation walkthrough')}>
      <header className="avl-rotation-header">
        <div>
          <p className="eyebrow">{t('AVL 再平衡', 'AVL rebalancing')}</p>
          <h2>{t('旋转只修改局部连接', 'Rotations update a local set of links')}</h2>
          <p>{t(
            'LL 模式展示转移子树 T2 的重新连接；LR 模式展示连续的左旋与右旋。',
            'LL mode shows transfer subtree T2 being reattached; LR mode shows the left and right rotations in sequence.',
          )}</p>
        </div>
        <div className="avl-rotation-mode" role="group" aria-label={t('选择 AVL 旋转案例', 'Choose an AVL rotation case')}>
          {Object.entries(modeLabels).map(([key, label]) => (
            <button
              aria-pressed={mode === key}
              className={mode === key ? 'active' : ''}
              key={key}
              onClick={() => {
                setMode(key);
                setActiveStep(0);
              }}
              type="button"
            >
              {label}
            </button>
          ))}
        </div>
      </header>

      <div className={`avl-rotation-step ${step.action === 'finish' ? 'finish' : step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {scenario.steps.length}</span>
        <strong>{copy.title}</strong>
        <p>{copy.detail}</p>
      </div>

      <div className="avl-rotation-workspace">
        <div className="avl-rotation-stage-card">
          <div className="avl-rotation-stage-heading">
            <span>{modeLabels[mode]}</span>
            <strong>{t('当前执行', 'Now')}: {lineLabels[step.activeLine]}</strong>
          </div>
          <AVLRotationDiagram step={step} t={t} />
          <div className="avl-rotation-state-cards">
            <div><span>{t('当前子树根', 'subtree root')}</span><strong>{step.root}</strong></div>
            <div><span>{t('移动节点 / 子树', 'moved node / subtree')}</span><strong>{step.moved ?? '—'}</strong></div>
            <div><span>{t('旋转类型', 'rotation')}</span><strong>{mode === 'll' ? t('右旋', 'right') : t('左旋 → 右旋', 'left → right')}</strong></div>
          </div>
        </div>

        <div className="avl-rotation-code" aria-label={t('当前 AVL 旋转伪代码', 'Current AVL rotation pseudocode')}>
          <div className="avl-rotation-code-heading">
            <span>{t('局部更新', 'Local update')}</span>
            <strong>{lineLabels[step.activeLine]}</strong>
          </div>
          <div className="avl-rotation-code-lines">
            {AVL_ROTATION_CODE_LINES[mode].map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="avl-rotation-legend">
        <span><i className="pivot" />{t('失衡节点', 'unbalanced node')}</span>
        <span><i className="promoted" />{t('上移节点', 'promoted node')}</span>
        <span><i className="transfer" />{t('转移子树', 'transfer subtree')}</span>
      </div>

      <div className="avl-rotation-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择 AVL 旋转步骤', 'Select an AVL rotation step')}
          max={scenario.steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === scenario.steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(scenario.steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const BUILD_TREE_PREORDER = [3, 9, 20, 15, 7];
const BUILD_TREE_INORDER = [9, 3, 15, 20, 7];
const BUILD_TREE_POSITIONS = {
  3: { x: 350, y: 54 },
  9: { x: 210, y: 145 },
  20: { x: 490, y: 145 },
  15: { x: 420, y: 238 },
  7: { x: 560, y: 238 },
};

const BUILD_TREE_CODE_LINES = [
  { id: 'init', code: ['root = TreeNode(preorder[0])', 'stack = [root]', 'j = 0'] },
  { id: 'create', code: ['for i in range(1, len(preorder)):', '    node = TreeNode(preorder[i])', '    parent = None'] },
  { id: 'compare', code: ['    while stack and stack[-1].val == inorder[j]:'] },
  { id: 'pop', code: ['        parent = stack.pop()', '        j += 1'] },
  { id: 'attach-right', code: ['    if parent:', '        parent.right = node'] },
  { id: 'attach-left', code: ['    else:', '        stack[-1].left = node'] },
  { id: 'push', code: ['    stack.append(node)'] },
  { id: 'finish', code: ['return root'] },
];

function buildBuildTreeSteps() {
  const preorder = BUILD_TREE_PREORDER;
  const inorder = BUILD_TREE_INORDER;
  const steps = [];
  const stack = [preorder[0]];
  const edges = [];
  const createdValues = new Set([preorder[0]]);
  let j = 0;

  const snapshot = (action, activeLine, extra = {}) => steps.push({
    action,
    activeLine,
    i: extra.i ?? null,
    j,
    current: extra.current ?? null,
    compareTop: extra.compareTop ?? null,
    popped: extra.popped ?? null,
    stack: [...stack],
    edges: edges.map((edge) => ({ ...edge })),
    created: [...createdValues],
  });

  snapshot('init', 'init', { current: preorder[0] });

  for (let i = 1; i < preorder.length; i++) {
    const nodeVal = preorder[i];
    createdValues.add(nodeVal);
    snapshot('create', 'create', { i, current: nodeVal });

    let parent = null;
    while (true) {
      if (!stack.length) break;
      const top = stack[stack.length - 1];
      snapshot('compare', 'compare', { i, current: nodeVal, compareTop: top });
      if (top !== inorder[j]) break;
      parent = stack.pop();
      j += 1;
      snapshot('pop', 'pop', { i, current: nodeVal, popped: parent });
    }

    if (parent !== null) {
      edges.push({ parent, child: nodeVal, side: 'right' });
      snapshot('attach-right', 'attach-right', { i, current: nodeVal });
    } else {
      edges.push({ parent: stack[stack.length - 1], child: nodeVal, side: 'left' });
      snapshot('attach-left', 'attach-left', { i, current: nodeVal });
    }

    stack.push(nodeVal);
    snapshot('push', 'push', { i, current: nodeVal });
  }

  snapshot('finish', 'finish', {});
  return steps;
}

const BUILD_TREE_STEPS = buildBuildTreeSteps();

function BuildTreeDiagram({ step, t }) {
  const placed = new Set([BUILD_TREE_PREORDER[0], ...step.edges.map((edge) => edge.child)]);
  return (
    <svg
      aria-label={t('从前序和中序构建的二叉树', 'The binary tree built from preorder and inorder')}
      className="build-tree-diagram"
      role="img"
      viewBox="0 0 700 292"
    >
      {step.edges.map((edge) => {
        const source = BUILD_TREE_POSITIONS[edge.parent];
        const target = BUILD_TREE_POSITIONS[edge.child];
        return (
          <line
            className="build-tree-edge"
            key={`${edge.parent}-${edge.child}`}
            x1={source.x}
            x2={target.x}
            y1={source.y + 27}
            y2={target.y - 27}
          />
        );
      })}
      {Object.entries(BUILD_TREE_POSITIONS).map(([value, pos]) => {
        const nodeVal = Number(value);
        if (!placed.has(nodeVal)) {
          return (
            <g className="build-tree-node ghost" key={nodeVal}>
              <rect height="54" rx="14" ry="14" width="70" x={pos.x - 35} y={pos.y - 27} />
            </g>
          );
        }
        const classes = [
          'build-tree-node',
          nodeVal === step.current ? 'current' : '',
          nodeVal === step.popped ? 'popped' : '',
          nodeVal === step.compareTop ? 'comparing' : '',
          nodeVal !== step.current && step.stack.includes(nodeVal) ? 'frontier' : '',
          nodeVal !== step.current && !step.stack.includes(nodeVal) ? 'visited' : '',
        ].filter(Boolean).join(' ');
        return (
          <g className={classes} key={nodeVal}>
            <rect height="54" rx="14" ry="14" width="70" x={pos.x - 35} y={pos.y - 27} />
            <text dominantBaseline="middle" textAnchor="middle" x={pos.x} y={pos.y}>{nodeVal}</text>
          </g>
        );
      })}
    </svg>
  );
}

function BuildTreeVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = BUILD_TREE_STEPS;
  const step = steps[activeStep];
  const pendingValue = step.current !== null
    && step.current !== BUILD_TREE_PREORDER[0]
    && !step.edges.some((edge) => edge.child === step.current)
    ? step.current
    : null;
  const displayStack = [...step.stack].reverse();

  const activeLineLabel = {
    init: t('创建根节点并初始化栈', 'Create the root and initialize the stack'),
    create: t('用下一个前序值创建新节点', 'Create a new node from the next preorder value'),
    compare: t('比较栈顶值与 inorder[j]', 'Compare the stack top with inorder[j]'),
    pop: t('弹出栈顶，j 前进一位', 'Pop the stack top and advance j'),
    'attach-right': t('接到最后弹出节点的右侧', 'Attach as the right child of the last popped node'),
    'attach-left': t('接到当前栈顶的左侧', 'Attach as the left child of the current stack top'),
    push: t('新节点入栈，等待右子节点', 'Push the new node, awaiting a right child'),
    finish: t('返回根节点', 'Return the root'),
  }[step.activeLine];

  const actionCopy = {
    init: {
      title: t('根节点来自 preorder[0]', 'The root comes from preorder[0]'),
      detail: t('根节点入栈，j 从 0 开始指向 inorder 中下一个待完成的位置。', 'The root is pushed onto the stack; j starts at 0, pointing at the next position inorder must resolve.'),
    },
    create: {
      title: t(`创建节点 ${step.current}`, `Create node ${step.current}`),
      detail: t('新节点还没有连接到树上，先看它应该接在哪里。', 'The new node is not connected to the tree yet — the next steps decide where it attaches.'),
    },
    compare: {
      title: t(`栈顶 ${step.compareTop} 是否等于 inorder[${step.j}]？`, `Does the stack top ${step.compareTop} equal inorder[${step.j}]?`),
      detail: t('相等说明栈顶节点的左子树已经在中序序列中完整出现，可以确定它没有更多待定的左侧内容。', 'Equality means the stack-top node’s left subtree has fully appeared in the inorder sequence — nothing about it is still pending on the left.'),
    },
    pop: {
      title: t(`弹出 ${step.popped}，j 变为 ${step.j}`, `Pop ${step.popped}; j becomes ${step.j}`),
      detail: t('这个节点暂时没有更多子节点等待判断；如果后面没有节点接到它右侧，它就保持只有左子树。', 'This node has nothing further pending for now; unless a later node attaches to its right, it keeps only its left subtree.'),
    },
    'attach-right': {
      title: t(`${step.current} 成为 ${step.edges[step.edges.length - 1]?.parent} 的右子节点`, `${step.current} becomes the right child of ${step.edges[step.edges.length - 1]?.parent}`),
      detail: t('至少发生过一次弹栈，说明新节点应该接在最后一个弹出节点的右侧。', 'At least one pop happened, so the new node attaches to the right of the last node popped.'),
    },
    'attach-left': {
      title: t(`${step.current} 成为 ${step.stack[step.stack.length - 1]} 的左子节点`, `${step.current} becomes the left child of ${step.stack[step.stack.length - 1]}`),
      detail: t('没有发生弹栈，说明当前栈顶还在等待左子节点。', 'No pop happened, so the current stack top is still waiting for its left child.'),
    },
    push: {
      title: t(`${step.current} 入栈`, `Push ${step.current}`),
      detail: t('新节点也可能还有自己的右子节点，所以先入栈等待。', 'The new node may still need a right child of its own, so it waits on the stack.'),
    },
    finish: {
      title: t('构建完成：3(9, 20(15, 7))', 'Build complete: 3(9, 20(15, 7))'),
      detail: t('每个节点入栈、出栈各一次，时间和额外空间都是 O(n)。', 'Every node is pushed and popped exactly once, so time and extra space are both O(n).'),
    },
  };
  const copy = actionCopy[step.action];

  return (
    <section className="build-tree-visual" aria-label={t('前序加中序重建二叉树逐步演示', 'Step-through: rebuilding a binary tree from preorder and inorder')}>
      <header className="build-tree-header">
        <div>
          <p className="eyebrow">{t('遍历序列重建', 'Traversal-sequence reconstruction')}</p>
          <h2>{t('用前序定根，用中序找分界', 'Preorder picks the root, inorder finds the split')}</h2>
          <p>{t(
            '固定示例 preorder = [3, 9, 20, 15, 7]，inorder = [9, 3, 15, 20, 7]。栈保存等待右子节点的节点，j 跟随中序序列前进。',
            'Fixed example preorder = [3, 9, 20, 15, 7], inorder = [9, 3, 15, 20, 7]. The stack holds nodes still waiting for a right child; j advances through the inorder sequence.',
          )}</p>
        </div>
      </header>

      <div className={`build-tree-step ${step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{copy.title}</strong>
        <p>{copy.detail}</p>
      </div>

      <div className="build-tree-workspace">
        <div className="build-tree-stage-card">
          <div className="build-tree-arrays">
            <div className="build-tree-array-row">
              <span>{t('preorder', 'preorder')}</span>
              <div>
                {BUILD_TREE_PREORDER.map((value, index) => (
                  <em className={index === step.i ? 'pointer' : ''} key={index}>
                    {value}
                    {index === step.i && <i>i</i>}
                  </em>
                ))}
              </div>
            </div>
            <div className="build-tree-array-row">
              <span>{t('inorder', 'inorder')}</span>
              <div>
                {BUILD_TREE_INORDER.map((value, index) => (
                  <em className={index === step.j ? 'pointer' : ''} key={index}>
                    {value}
                    {index === step.j && <i>j</i>}
                  </em>
                ))}
              </div>
            </div>
          </div>

          <div className="build-tree-stage-heading">
            <span>{t('构建中的树', 'Tree under construction')}</span>
            <strong>{t('当前执行', 'Now')}: {activeLineLabel}</strong>
          </div>
          <BuildTreeDiagram step={step} t={t} />

          {pendingValue !== null && (
            <div className="build-tree-pending">
              <span>{t('待连接节点', 'Node awaiting attachment')}</span>
              <div className="build-tree-pending-node">{pendingValue}</div>
            </div>
          )}

          <div className="build-tree-container-panel">
            <div>
              <span>{t('栈：等待右子节点，栈顶在左', 'Stack: awaiting a right child, top at left')}</span>
              <strong>LIFO</strong>
            </div>
            <div className="build-tree-container">
              {displayStack.length ? displayStack.map((value, index) => (
                <span className={index === 0 ? 'next' : ''} key={`${value}-${index}`}>
                  {value}
                  {index === 0 && <i>{t('栈顶', 'top')}</i>}
                </span>
              )) : <em>{t('空', 'empty')}</em>}
            </div>
          </div>
        </div>

        <div className="build-tree-code" aria-label={t('当前重建代码', 'Current reconstruction code')}>
          <div className="build-tree-code-heading">
            <span>{t('迭代模板', 'Iterative template')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="build-tree-code-lines">
            {BUILD_TREE_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="build-tree-legend">
        <span><i className="current" />{t('本轮新节点', 'this round’s new node')}</span>
        <span><i className="comparing" />{t('正在比较', 'being compared')}</span>
        <span><i className="popped" />{t('刚刚弹出', 'just popped')}</span>
        <span><i className="frontier" />{t('栈中等待', 'waiting on the stack')}</span>
        <span><i className="visited" />{t('已确定', 'resolved')}</span>
      </div>

      <div className="build-tree-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择构建步骤', 'Select a build step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const MEDIAN_STREAM = [5, 15, 1, 3];

const MEDIAN_CODE_LINES = [
  { id: 'push-small', code: ['heapq.heappush(self.small, -num)'] },
  { id: 'transfer-to-large', code: ['heapq.heappush(self.large, -heapq.heappop(self.small))'] },
  { id: 'check-balance', code: ['if len(self.large) > len(self.small):'] },
  { id: 'transfer-back', code: ['    heapq.heappush(self.small, -heapq.heappop(self.large))'] },
  { id: 'median', code: ['# findMedian()', 'return (-self.small[0] if len(self.small) > len(self.large)', '        else (-self.small[0] + self.large[0]) / 2)'] },
];

function buildMedianStreamSteps() {
  const steps = [];
  let small = []; // sorted desc, front is the max
  let large = []; // sorted asc, front is the min

  const snapshot = (action, activeLine, extra = {}) => steps.push({
    action,
    activeLine,
    num: extra.num ?? null,
    moved: extra.moved ?? null,
    balanced: extra.balanced ?? null,
    small: [...small],
    large: [...large],
    median: extra.median ?? null,
  });

  snapshot('start', 'push-small');

  MEDIAN_STREAM.forEach((num) => {
    snapshot('incoming', 'push-small', { num });

    small.push(num);
    small.sort((a, b) => b - a);
    snapshot('push-small', 'push-small', { num, moved: num });

    const movedToLarge = small.shift();
    large.push(movedToLarge);
    large.sort((a, b) => a - b);
    snapshot('transfer-to-large', 'transfer-to-large', { num, moved: movedToLarge });

    if (large.length > small.length) {
      const movedToSmall = large.shift();
      small.push(movedToSmall);
      small.sort((a, b) => b - a);
      snapshot('transfer-back', 'transfer-back', { num, moved: movedToSmall, balanced: false });
    } else {
      snapshot('balanced', 'check-balance', { num, balanced: true });
    }

    const median = small.length > large.length ? small[0] : (small[0] + large[0]) / 2;
    snapshot('median', 'median', { num, median });
  });

  return steps;
}

const MEDIAN_STREAM_STEPS = buildMedianStreamSteps();

function MedianTwoHeapsVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = MEDIAN_STREAM_STEPS;
  const step = steps[activeStep];

  const activeLineLabel = {
    'push-small': t('新数字入 small', 'Push the new number into small'),
    'transfer-to-large': t('转移 small 的最大值给 large', "Transfer small's maximum to large"),
    'check-balance': t('检查两堆大小是否失衡', 'Check whether the two heaps are unbalanced'),
    'transfer-back': t('转移 large 的最小值回 small', "Transfer large's minimum back to small"),
    median: t('计算中位数', 'Compute the median'),
  }[step.activeLine];

  const actionCopy = {
    start: {
      title: t('两个空堆，还没有插入任何数字', 'Two empty heaps, before any number is inserted'),
      detail: t('small 是最大堆，保存较小的一半；large 是最小堆，保存较大的一半。', 'small is a max-heap holding the lower half; large is a min-heap holding the upper half.'),
    },
    incoming: {
      title: t(`下一个数字：${step.num}`, `Next number: ${step.num}`),
      detail: t('先不判断它该进哪一堆，统一按固定顺序处理。', 'There is no need to decide which heap it belongs in first; the fixed order below handles that automatically.'),
    },
    'push-small': {
      title: t(`把 ${step.num} 推入 small`, `Push ${step.num} into small`),
      detail: t('不管这个数字最终该属于哪一半，都先无条件放进 small。', 'Regardless of which half it actually belongs to, it always goes into small first.'),
    },
    'transfer-to-large': {
      title: t(`把 small 的最大值 ${step.moved} 转移给 large`, `Transfer small's maximum, ${step.moved}, to large`),
      detail: t('这一步保证 small 剩下的元素都不超过刚转移过去的这个值。', 'This step guarantees every value remaining in small is no greater than the value just transferred.'),
    },
    balanced: {
      title: t('两堆大小差不超过一，不需要转移', 'The size difference is at most one; no transfer is needed'),
      detail: t('large 并没有比 small 多，跳过转移回 small 的一步。', 'large did not end up larger than small, so the transfer-back step is skipped.'),
    },
    'transfer-back': {
      title: t(`把 large 的最小值 ${step.moved} 转移回 small`, `Transfer large's minimum, ${step.moved}, back to small`),
      detail: t('转移之后 large 比 small 多了一个，用这一步把大小差恢复到最多为一。', 'The previous transfer left large with one more element than small; this step restores the size difference to at most one.'),
    },
    median: {
      title: t(`中位数是 ${step.median}`, `The median is ${step.median}`),
      detail: step.small.length > step.large.length
        ? t('small 比 large 多一个，中位数就是 small 的堆顶。', 'small has one more element than large, so the median is simply the top of small.')
        : t('两堆大小相等，中位数是两个堆顶的平均值。', 'The two heaps are equal in size, so the median is the average of both tops.'),
    },
  };
  const copy = actionCopy[step.action];

  const renderHeap = (values, tone) => (
    <div className={`median-heaps-list ${tone}`}>
      {values.length ? values.map((value, index) => (
        <span
          className={[
            index === 0 ? 'top' : '',
            value === step.moved ? 'moved' : '',
            step.action !== 'start' && step.action !== 'median' && value === step.num && index !== 0 ? 'incoming' : '',
          ].filter(Boolean).join(' ')}
          key={`${value}-${index}`}
        >
          {value}
          {index === 0 && <i>{t('堆顶', 'top')}</i>}
        </span>
      )) : <em>{t('空', 'empty')}</em>}
    </div>
  );

  return (
    <section className="median-heaps-visual" aria-label={t('双堆维护中位数逐步演示', 'Step-through: maintaining a running median with two heaps')}>
      <header className="median-heaps-header">
        <div>
          <p className="eyebrow">{t('双堆维护中位数', 'Two heaps maintaining a running median')}</p>
          <h2>{t('固定流：5, 15, 1, 3', 'Fixed stream: 5, 15, 1, 3')}</h2>
          <p>{t(
            '每个数字都按同一个顺序处理：先入 small，转移堆顶给 large，必要时再转移回来。',
            'Every number goes through the same order: push into small, transfer its top to large, then transfer back if needed.',
          )}</p>
        </div>
      </header>

      <div className={`median-heaps-step ${step.action}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{copy.title}</strong>
        <p>{copy.detail}</p>
      </div>

      <div className="median-heaps-workspace">
        <div className="median-heaps-stage-card">
          <div className="median-heaps-stream">
            <span>{t('数据流', 'Stream')}</span>
            <div>
              {MEDIAN_STREAM.map((value, index) => (
                <em className={step.num === value && step.action !== 'median' && step.action !== 'start' ? 'pointer' : ''} key={index}>
                  {value}
                </em>
              ))}
            </div>
          </div>

          <div className="median-heaps-pair">
            <div className="median-heaps-panel">
              <div className="median-heaps-panel-heading">
                <span>small</span>
                <strong>{t('最大堆，较小的一半', 'max-heap, lower half')}</strong>
              </div>
              {renderHeap(step.small, 'small')}
            </div>
            <div className="median-heaps-panel">
              <div className="median-heaps-panel-heading">
                <span>large</span>
                <strong>{t('最小堆，较大的一半', 'min-heap, upper half')}</strong>
              </div>
              {renderHeap(step.large, 'large')}
            </div>
          </div>

          <div className="median-heaps-median">
            <span>{t('当前中位数', 'Current median')}</span>
            <strong>{step.median ?? '—'}</strong>
          </div>
        </div>

        <div className="median-heaps-code" aria-label={t('当前双堆代码', 'Current two-heap code')}>
          <div className="median-heaps-code-heading">
            <span>{t('addNum 模板', 'addNum template')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="median-heaps-code-lines">
            {MEDIAN_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="median-heaps-legend">
        <span><i className="top" />{t('堆顶', 'top')}</span>
        <span><i className="moved" />{t('本步转移的值', 'value moved this step')}</span>
      </div>

      <div className="median-heaps-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择中位数演示步骤', 'Select a median demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const ROTTING_GRID_INITIAL = [
  [2, 1, 1],
  [1, 1, 0],
  [0, 1, 1],
];

const ROTTING_CODE_LINES = [
  { id: 'seed', code: ['for r in range(rows):', '    for c in range(cols):', '        if grid[r][c] == 2: q.append((r, c))', '        elif grid[r][c] == 1: fresh += 1'] },
  { id: 'loop', code: ['while q and fresh:', '    minutes += 1'] },
  { id: 'relax', code: ['    for _ in range(len(q)):', '        r, c = q.popleft()', '        for dr, dc in DIRECTIONS:', '            nr, nc = r + dr, c + dc', '            if valid(nr, nc) and grid[nr][nc] == 1:', '                grid[nr][nc] = 2', '                fresh -= 1', '                q.append((nr, nc))'] },
  { id: 'return', code: ['return minutes if fresh == 0 else -1'] },
];

function buildRottingStepsData() {
  const grid = ROTTING_GRID_INITIAL.map((row) => [...row]);
  const rows = grid.length;
  const cols = grid[0].length;
  const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  let fresh = 0;
  let queue = [];

  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      if (grid[r][c] === 1) fresh += 1;
      if (grid[r][c] === 2) queue.push([r, c]);
    }
  }

  const steps = [{
    minute: 0,
    grid: grid.map((row) => [...row]),
    newlyRotten: [],
    freshRemaining: fresh,
    activeLine: 'seed',
    done: false,
  }];

  let minute = 0;
  while (queue.length > 0 && fresh > 0) {
    minute += 1;
    const next = [];
    const newlyRotten = [];
    queue.forEach(([r, c]) => {
      directions.forEach(([dr, dc]) => {
        const nr = r + dr;
        const nc = c + dc;
        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc] === 1) {
          grid[nr][nc] = 2;
          fresh -= 1;
          next.push([nr, nc]);
          newlyRotten.push([nr, nc]);
        }
      });
    });
    queue = next;
    steps.push({
      minute,
      grid: grid.map((row) => [...row]),
      newlyRotten,
      freshRemaining: fresh,
      activeLine: 'relax',
      done: false,
    });
  }

  steps.push({
    minute,
    grid: grid.map((row) => [...row]),
    newlyRotten: [],
    freshRemaining: fresh,
    activeLine: 'return',
    done: true,
    answer: fresh === 0 ? minute : -1,
  });

  return steps;
}

const ROTTING_STEPS = buildRottingStepsData();

function RottingOrangesBFSVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = ROTTING_STEPS;
  const step = steps[activeStep];

  const isNewlyRotten = (r, c) => step.newlyRotten.some(([nr, nc]) => nr === r && nc === c);

  const activeLineLabel = {
    seed: t('初始化：所有腐烂格子入队，统计新鲜橘子数量', 'Initialize: seed every rotten cell into the queue and count fresh oranges'),
    loop: t('队列非空且还有新鲜橘子时继续', 'Continue while the queue is non-empty and fresh oranges remain'),
    relax: t('按当前队列长度整体弹出，检查四个方向并把新鲜橘子变腐烂入队', 'Pop the entire current layer, check all four directions, and turn fresh neighbors rotten'),
    return: t('返回耗费的分钟数，如果还有新鲜橘子剩下就返回 -1', 'Return the elapsed minutes, or -1 if fresh oranges remain'),
  }[step.activeLine];

  let title;
  let detail;
  if (step.minute === 0 && !step.done) {
    title = t('第 0 分钟：起点是所有腐烂橘子', 'Minute 0: the sources are every already-rotten orange');
    detail = t('多源 BFS 把所有腐烂格子同时放入队列，而不是只从一个格子出发。', 'Multi-source BFS seeds every rotten cell into the queue at once, instead of starting from a single cell.');
  } else if (step.done) {
    title = step.answer === -1
      ? t('还有新鲜橘子无法到达，返回 -1', 'Some fresh oranges are unreachable, return -1')
      : t(`全部腐烂，耗费 ${step.answer} 分钟`, `Every orange is rotten, elapsed time is ${step.answer} minutes`);
    detail = t('BFS 按层扩展，最后一层的编号就是答案。', 'BFS expands layer by layer, and the final layer number is exactly the answer.');
  } else {
    title = t(`第 ${step.minute} 分钟：新增 ${step.newlyRotten.length} 个腐烂橘子`, `Minute ${step.minute}: ${step.newlyRotten.length} orange(s) turn rotten`);
    detail = t('这一层的所有新鲜邻居同时变腐烂，它们与起点的 BFS 距离相同。', 'Every fresh neighbor discovered in this layer turns rotten at once, since they share the same BFS distance from the sources.');
  }

  return (
    <section className="mbfs-visual" aria-label={t('多源 BFS：Rotting Oranges 逐步演示', 'Step-through: multi-source BFS on Rotting Oranges')}>
      <header className="mbfs-header">
        <div>
          <p className="eyebrow">{t('多源 BFS，按层扩展', 'Multi-source BFS, layer by layer')}</p>
          <h2>{t('Rotting Oranges：grid = [[2,1,1],[1,1,0],[0,1,1]]', 'Rotting Oranges: grid = [[2,1,1],[1,1,0],[0,1,1]]')}</h2>
          <p>{t(
            '所有腐烂橘子同时是 BFS 的起点，入队时立刻标记，一层代表一分钟。',
            'Every rotten orange starts as a BFS source at once. Cells are marked the moment they are enqueued, and one layer equals one minute.',
          )}</p>
        </div>
      </header>

      <div className={`mbfs-step ${step.done ? 'done' : ''}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="mbfs-workspace">
        <div className="mbfs-stage-card">
          <div className="mbfs-grid" style={{ '--mbfs-cols': step.grid[0].length }}>
            {step.grid.map((row, r) => row.map((value, c) => {
              const highlight = isNewlyRotten(r, c);
              const cellClass = value === 2 ? 'rotten' : value === 1 ? 'fresh' : 'empty';
              return (
                <div className={`mbfs-cell ${cellClass} ${highlight ? 'newly' : ''}`} key={`${r}-${c}`}>
                  <span>{value}</span>
                </div>
              );
            }))}
          </div>
          <div className="mbfs-meta">
            <div>
              <span>{t('分钟', 'Minute')}</span>
              <strong>{step.minute}</strong>
            </div>
            <div>
              <span>{t('剩余新鲜', 'Fresh remaining')}</span>
              <strong>{step.freshRemaining}</strong>
            </div>
          </div>
        </div>

        <div className="mbfs-code" aria-label={t('当前 BFS 代码', 'Current BFS code')}>
          <div className="mbfs-code-heading">
            <span>{t('多源 BFS 模板', 'Multi-source BFS template')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="mbfs-code-lines">
            {ROTTING_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="mbfs-legend">
        <span><i className="fresh" />{t('新鲜（1）', 'Fresh (1)')}</span>
        <span><i className="rotten" />{t('腐烂（2）', 'Rotten (2)')}</span>
        <span><i className="newly" />{t('本层新增腐烂', 'Turned rotten this layer')}</span>
        <span><i className="empty" />{t('空格（0）', 'Empty (0)')}</span>
      </div>

      <div className="mbfs-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择 BFS 层数', 'Select a BFS layer')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const UNION_FIND_N = 8;

const UNION_FIND_OPERATIONS = [
  { type: 'union', a: 0, b: 1, mode: 'naive' },
  { type: 'union', a: 1, b: 2, mode: 'naive' },
  { type: 'union', a: 2, b: 3, mode: 'naive' },
  { type: 'union', a: 3, b: 4, mode: 'naive' },
  { type: 'union', a: 5, b: 6, mode: 'naive' },
  { type: 'union', a: 6, b: 7, mode: 'naive' },
  { type: 'find', x: 0 },
  { type: 'union', a: 0, b: 5, mode: 'size' },
];

const UNION_FIND_CODE_LINES = [
  { id: 'union-naive', code: ['def union(a, b):  # 暂不按大小合并', '    ra, rb = find(a), find(b)', '    if ra != rb:', '        parent[ra] = rb'] },
  { id: 'find-compress', code: ['def find(x):', '    if parent[x] != x:', '        parent[x] = find(parent[x])  # 路径压缩', '    return parent[x]'] },
  { id: 'union-size', code: ['def union(a, b):  # 按集合大小合并', '    ra, rb = find(a), find(b)', '    if size[ra] < size[rb]:', '        ra, rb = rb, ra', '    parent[rb] = ra', '    size[ra] += size[rb]'] },
];

function ufFindNoCompress(parent, x) {
  let cur = x;
  while (parent[cur] !== cur) cur = parent[cur];
  return cur;
}

function ufComponentsOf(parent) {
  const groups = new Map();
  parent.forEach((_, i) => {
    const root = ufFindNoCompress(parent, i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(i);
  });
  return [...groups.entries()]
    .map(([root, members]) => ({ root, members }))
    .sort((a, b) => a.root - b.root);
}

function buildUnionFindSteps() {
  const parent = Array.from({ length: UNION_FIND_N }, (_, i) => i);
  const steps = [{
    kind: 'init',
    activeLine: 'union-naive',
    parent: [...parent],
    changed: [],
    components: ufComponentsOf(parent),
    op: null,
  }];

  UNION_FIND_OPERATIONS.forEach((op) => {
    if (op.type === 'union' && op.mode === 'naive') {
      const ra = ufFindNoCompress(parent, op.a);
      const rb = ufFindNoCompress(parent, op.b);
      parent[ra] = rb;
      steps.push({
        kind: 'union-naive',
        activeLine: 'union-naive',
        parent: [...parent],
        changed: [ra],
        components: ufComponentsOf(parent),
        op: { type: 'union', a: op.a, b: op.b, ra, rb },
      });
    } else if (op.type === 'find') {
      const path = [];
      let cur = op.x;
      while (parent[cur] !== cur) {
        path.push(cur);
        cur = parent[cur];
      }
      const root = cur;
      path.forEach((node) => { parent[node] = root; });
      steps.push({
        kind: 'find-compress',
        activeLine: 'find-compress',
        parent: [...parent],
        changed: [...path],
        components: ufComponentsOf(parent),
        op: { type: 'find', x: op.x, path, root },
      });
    } else if (op.type === 'union' && op.mode === 'size') {
      const beforeComponents = ufComponentsOf(parent);
      const ra = ufFindNoCompress(parent, op.a);
      const rb = ufFindNoCompress(parent, op.b);
      const sizeA = beforeComponents.find((group) => group.root === ra).members.length;
      const sizeB = beforeComponents.find((group) => group.root === rb).members.length;
      let winner = ra;
      let loser = rb;
      if (sizeA < sizeB) {
        winner = rb;
        loser = ra;
      }
      parent[loser] = winner;
      steps.push({
        kind: 'union-size',
        activeLine: 'union-size',
        parent: [...parent],
        changed: [loser],
        components: ufComponentsOf(parent),
        op: {
          type: 'union',
          a: op.a,
          b: op.b,
          ra: winner,
          rb: loser,
          winnerSize: Math.max(sizeA, sizeB),
          loserSize: Math.min(sizeA, sizeB),
        },
      });
    }
  });

  return steps;
}

const UNION_FIND_STEPS = buildUnionFindSteps();

function UnionFindVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = UNION_FIND_STEPS;
  const step = steps[activeStep];

  const activeLineLabel = {
    'union-naive': t('先不比较大小，直接把一个根挂到另一个根下', 'Attach one root under the other without comparing sizes yet'),
    'find-compress': t('find 时把路径上的每个节点直接指向根', 'find rewires every node on the path to point straight at the root'),
    'union-size': t('比较两个集合大小，把小集合的根挂到大集合的根下', 'Compare the two set sizes and attach the smaller root under the larger one'),
  }[step.activeLine];

  let title;
  let detail;
  if (step.kind === 'init') {
    title = t('8 个节点，初始时各自是自己的根', '8 nodes, each initially its own root');
    detail = t('parent[i] = i，还没有发生任何合并。', 'parent[i] = i for every node; no union has happened yet.');
  } else if (step.kind === 'union-naive') {
    title = t(`union(${step.op.a}, ${step.op.b})：把根 ${step.op.ra} 挂到根 ${step.op.rb} 下`, `union(${step.op.a}, ${step.op.b}): attach root ${step.op.ra} under root ${step.op.rb}`);
    detail = t('这里故意不按大小合并，让链条越接越长，才能看出路径压缩的效果。', 'This union deliberately skips the by-size optimization so the chain keeps growing, which is what makes path compression visible in the next step.');
  } else if (step.kind === 'find-compress') {
    title = t(`find(${step.op.x})：路径压缩`, `find(${step.op.x}): path compression`);
    detail = t(
      `find(${step.op.x}) 原本要沿着 ${step.op.path.join(' → ')} → ${step.op.root} 走 ${step.op.path.length} 步，压缩后这些节点全部直接指向根 ${step.op.root}。`,
      `find(${step.op.x}) originally has to walk ${step.op.path.join(' → ')} → ${step.op.root}, ${step.op.path.length} hop(s). After compression every node on that path points straight at root ${step.op.root}.`,
    );
  } else {
    title = t(`union(${step.op.a}, ${step.op.b})：按大小合并`, `union(${step.op.a}, ${step.op.b}): union by size`);
    detail = t(
      `根 ${step.op.ra} 所在集合有 ${step.op.winnerSize} 个节点，根 ${step.op.rb} 所在集合只有 ${step.op.loserSize} 个，小集合挂到大集合下面。`,
      `Root ${step.op.ra}'s set has ${step.op.winnerSize} node(s) while root ${step.op.rb}'s set has only ${step.op.loserSize}; the smaller set attaches under the larger one.`,
    );
  }

  return (
    <section className="uf-visual" aria-label={t('并查集：路径压缩与按大小合并演示', 'Step-through: union-find with path compression and union by size')}>
      <header className="uf-header">
        <div>
          <p className="eyebrow">{t('并查集，find 与 union', 'Union-Find, find and union')}</p>
          <h2>{t('8 个节点：先建链条，再看路径压缩', '8 nodes: build a chain, then watch path compression')}</h2>
          <p>{t(
            '前 6 次 union 故意不按大小合并，制造出两条链条；第 7 步的 find 触发路径压缩；最后一次 union 按大小合并。',
            'The first 6 union calls skip the by-size optimization on purpose, building two chains. The 7th step is a find call that triggers path compression. The final union merges the two components by size.',
          )}</p>
        </div>
      </header>

      <div className={`uf-step ${step.kind}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="uf-workspace">
        <div className="uf-stage-card">
          <div className="uf-array">
            {step.parent.map((p, i) => {
              const isRoot = p === i;
              const changed = step.changed.includes(i);
              return (
                <div className={`uf-node ${isRoot ? 'root' : ''} ${changed ? 'changed' : ''}`} key={i}>
                  <span>{i}</span>
                  <strong>{t(`父 ${p}`, `parent ${p}`)}</strong>
                </div>
              );
            })}
          </div>

          <div className="uf-components">
            <span>{t('当前连通分量', 'Current connected components')}</span>
            <div>
              {step.components.map((group) => (
                <div className="uf-component" key={group.root}>
                  <em>{t(`根 ${group.root}`, `root ${group.root}`)}</em>
                  <div>
                    {group.members.map((m) => <i key={m}>{m}</i>)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="uf-code" aria-label={t('当前并查集代码', 'Current union-find code')}>
          <div className="uf-code-heading">
            <span>{t('find / union 模板', 'find / union template')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="uf-code-lines">
            {UNION_FIND_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="uf-legend">
        <span><i className="root" />{t('根节点（parent[i] = i）', 'Root node (parent[i] = i)')}</span>
        <span><i className="changed" />{t('本步更新的父指针', 'Parent pointer updated this step')}</span>
      </div>

      <div className="uf-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择并查集演示步骤', 'Select a union-find demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const QUICKSELECT_NUMS = [3, 2, 1, 5, 6, 4];
const QUICKSELECT_K = 2;

const QUICKSELECT_CODE_LINES = [
  { id: 'setup', code: ['target = len(nums) - k'] },
  { id: 'pick-pivot', code: ['pivot = nums[right]  # 演示固定选最右元素，代码里用随机 pivot'] },
  { id: 'init-store', code: ['store = left'] },
  { id: 'scan-loop', code: ['for i in range(left, right):'] },
  { id: 'compare', code: ['    if nums[i] < pivot:'] },
  { id: 'swap', code: ['        nums[i], nums[store] = nums[store], nums[i]', '        store += 1'] },
  { id: 'place-pivot', code: ['nums[store], nums[right] = nums[right], nums[store]', 'pivot_index = store'] },
  { id: 'check-target', code: ['if pivot_index == target:', '    return nums[pivot_index]'] },
  { id: 'go-right', code: ['if pivot_index < target:', '    left = pivot_index + 1'] },
  { id: 'go-left', code: ['else:', '    right = pivot_index - 1'] },
];

function buildQuickSelectSteps() {
  const nums = [...QUICKSELECT_NUMS];
  const n = nums.length;
  const target = n - QUICKSELECT_K;
  const steps = [];
  const settledPivots = [];
  let left = 0;
  let right = n - 1;
  let round = 0;

  steps.push({
    phase: 'init',
    activeLine: 'setup',
    array: [...nums],
    left,
    right,
    target,
    round,
    settledPivots: [...settledPivots],
  });

  while (true) {
    round += 1;
    const pivot = nums[right];
    const pivotAt = right;
    let store = left;

    steps.push({
      phase: 'round-start',
      activeLine: 'pick-pivot',
      array: [...nums],
      left,
      right,
      target,
      round,
      pivot,
      pivotAt,
      storeIndex: store,
      scanIndex: null,
      settledPivots: [...settledPivots],
    });

    for (let i = left; i < right; i += 1) {
      const willSwap = nums[i] < pivot;
      steps.push({
        phase: 'compare',
        activeLine: 'compare',
        array: [...nums],
        left,
        right,
        target,
        round,
        pivot,
        pivotAt,
        storeIndex: store,
        scanIndex: i,
        willSwap,
        settledPivots: [...settledPivots],
      });
      if (willSwap) {
        const storeBefore = store;
        const tmp = nums[i];
        nums[i] = nums[store];
        nums[store] = tmp;
        store += 1;
        steps.push({
          phase: 'swap',
          activeLine: 'swap',
          array: [...nums],
          left,
          right,
          target,
          round,
          pivot,
          pivotAt,
          storeIndex: store,
          scanIndex: i,
          swappedIndices: [i, storeBefore],
          settledPivots: [...settledPivots],
        });
      }
    }

    const storeVal = nums[store];
    nums[store] = nums[right];
    nums[right] = storeVal;
    const pivotIndex = store;

    steps.push({
      phase: 'place-pivot',
      activeLine: 'place-pivot',
      array: [...nums],
      left,
      right,
      target,
      round,
      pivotIndex,
      swappedIndices: [store, right],
      settledPivots: [...settledPivots],
    });

    settledPivots.push(pivotIndex);

    if (pivotIndex === target) {
      steps.push({
        phase: 'done',
        activeLine: 'check-target',
        array: [...nums],
        left,
        right,
        target,
        round,
        pivotIndex,
        answer: nums[pivotIndex],
        settledPivots: [...settledPivots],
      });
      break;
    }

    if (pivotIndex < target) {
      steps.push({
        phase: 'go-right',
        activeLine: 'go-right',
        array: [...nums],
        left,
        right,
        target,
        round,
        pivotIndex,
        settledPivots: [...settledPivots],
      });
      left = pivotIndex + 1;
    } else {
      steps.push({
        phase: 'go-left',
        activeLine: 'go-left',
        array: [...nums],
        left,
        right,
        target,
        round,
        pivotIndex,
        settledPivots: [...settledPivots],
      });
      right = pivotIndex - 1;
    }
  }

  return steps;
}

const QUICKSELECT_STEPS = buildQuickSelectSteps();

function QuickSelectPartitionVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = QUICKSELECT_STEPS;
  const step = steps[activeStep];

  const activeLineLabel = {
    setup: t('计算目标下标：第 k 大对应升序排序后的第几个位置', 'Compute the target index: which position the kth largest lands on after ascending sort'),
    'pick-pivot': t('选取 pivot（演示固定选最右元素，便于复现）', 'Choose the pivot (fixed to the rightmost element here so the trace is reproducible)'),
    'init-store': t('store 指针从区间左端开始', 'The store pointer starts at the left end of the range'),
    'scan-loop': t('scan 指针从左到右扫描区间', 'The scan pointer sweeps the range left to right'),
    compare: t('比较当前元素与 pivot', 'Compare the current element with the pivot'),
    swap: t('比 pivot 小就换到 store 位置，store 前进一位', 'If it is smaller than the pivot, swap it to the store position and advance store'),
    'place-pivot': t('把 pivot 换到 store 位置，完成这一轮划分', 'Swap the pivot into the store position, finishing this round of partitioning'),
    'check-target': t('pivot 落位下标是否正好是 target', 'Check whether the pivot landed exactly on the target index'),
    'go-right': t('target 在右边，丢弃左边，只递归右侧', 'The target is on the right; discard the left side and recurse only into the right'),
    'go-left': t('target 在左边，丢弃右边，只递归左侧', 'The target is on the left; discard the right side and recurse only into the left'),
  }[step.activeLine];

  let title;
  let detail;
  if (step.phase === 'init') {
    title = t(
      `目标下标 target = ${step.array.length} - ${QUICKSELECT_K} = ${step.target}`,
      `Target index: target = ${step.array.length} - ${QUICKSELECT_K} = ${step.target}`,
    );
    detail = t(
      'quickselect 只需要让 pivot 落在 target 位置，不需要把整个数组排好序。',
      'Quickselect only needs the pivot to land on the target index; it never needs to fully sort the array.',
    );
  } else if (step.phase === 'round-start') {
    title = t(
      `第 ${step.round} 轮：区间 [${step.left}, ${step.right}]，选 nums[${step.pivotAt}] = ${step.pivot} 作为 pivot`,
      `Round ${step.round}: range [${step.left}, ${step.right}], pivot = nums[${step.pivotAt}] = ${step.pivot}`,
    );
    detail = t(
      '为了让演示可复现，这里固定选最右边的元素作为 pivot；实际代码用随机 pivot 是为了避免特定输入构造出最坏情况。',
      'To keep the trace reproducible, this demo always picks the rightmost element as the pivot; the real code uses a random pivot to avoid a worst case built from a specific input.',
    );
  } else if (step.phase === 'compare') {
    title = t(
      `比较 nums[${step.scanIndex}] = ${step.array[step.scanIndex]} 与 pivot ${step.pivot}`,
      `Compare nums[${step.scanIndex}] = ${step.array[step.scanIndex]} with pivot ${step.pivot}`,
    );
    detail = step.willSwap
      ? t('比 pivot 小，需要换到 store 指针位置。', 'Smaller than the pivot, so it needs to move to the store position.')
      : t('不小于 pivot，留在原地，scan 继续前进。', 'Not smaller than the pivot, so it stays put while scan keeps moving.');
  } else if (step.phase === 'swap') {
    title = t(
      `交换下标 ${step.swappedIndices[0]} 与 ${step.swappedIndices[1]}，store 前进到 ${step.storeIndex}`,
      `Swap indices ${step.swappedIndices[0]} and ${step.swappedIndices[1]}; store advances to ${step.storeIndex}`,
    );
    detail = t(
      'store 指针始终指向"已确认小于 pivot 的区域"的右边界。',
      'The store pointer always marks the right boundary of the region confirmed to be smaller than the pivot.',
    );
  } else if (step.phase === 'place-pivot') {
    title = t(
      `把 pivot 换到 store 位置：交换下标 ${step.swappedIndices[0]} 与 ${step.swappedIndices[1]}，pivot 落位在下标 ${step.pivotIndex}`,
      `Swap the pivot into the store position: swap indices ${step.swappedIndices[0]} and ${step.swappedIndices[1]}; the pivot lands at index ${step.pivotIndex}`,
    );
    detail = t(
      '这一步之后，下标左边的元素都比 pivot 小，右边的都不小于 pivot，这是 Lomuto partition 的核心不变量。',
      'After this step, everything left of the index is smaller than the pivot and everything right is not smaller, the core invariant of Lomuto partition.',
    );
  } else if (step.phase === 'go-right') {
    title = t(
      `pivot_index(${step.pivotIndex}) < target(${step.target})，只递归右侧`,
      `pivot_index(${step.pivotIndex}) < target(${step.target}); recurse only into the right side`,
    );
    detail = t(
      '左边的元素已经确定不是答案，不会再被访问；下一轮区间变成右侧剩余部分。',
      'The left side is already confirmed not to hold the answer and will never be visited again; the next round works only on the remaining right side.',
    );
  } else if (step.phase === 'go-left') {
    title = t(
      `pivot_index(${step.pivotIndex}) > target(${step.target})，只递归左侧`,
      `pivot_index(${step.pivotIndex}) > target(${step.target}); recurse only into the left side`,
    );
    detail = t(
      '右边的元素已经确定不是答案，不会再被访问；下一轮区间变成左侧剩余部分。',
      'The right side is already confirmed not to hold the answer and will never be visited again; the next round works only on the remaining left side.',
    );
  } else {
    title = t(
      `pivot_index 恰好等于 target，返回 nums[${step.pivotIndex}] = ${step.answer}`,
      `pivot_index equals target exactly; return nums[${step.pivotIndex}] = ${step.answer}`,
    );
    detail = t(
      'quickselect 不需要把整个数组排完，只要 pivot 落在 target 位置就可以直接返回，这是它平均比堆排序更快的原因。',
      'Quickselect never needs to finish sorting the array; once the pivot lands on the target index it can return immediately, which is why it is faster on average than the heap version.',
    );
  }

  const isSettled = (index) => index < step.left || index > step.right;
  const isFinalPivot = (index) => (step.settledPivots ?? []).includes(index) || (step.phase === 'done' && index === step.pivotIndex);

  return (
    <section className="qs-visual" aria-label={t('Quickselect Partition 逐步演示', 'Step-through: quickselect partition')}>
      <header className="qs-header">
        <div>
          <p className="eyebrow">{t('Quickselect，复用 partition，每轮只递归一侧', 'Quickselect, reusing partition and recursing into only one side per round')}</p>
          <h2>{t(
            `nums = [${QUICKSELECT_NUMS.join(', ')}], k = ${QUICKSELECT_K}`,
            `nums = [${QUICKSELECT_NUMS.join(', ')}], k = ${QUICKSELECT_K}`,
          )}</h2>
          <p>{t(
            '和 Quick Sort 用的是同一个 Lomuto partition：区别只在于每轮只需要沿着 pivot 落位的方向继续，不需要两侧都递归。',
            'This uses the exact same Lomuto partition as Quick Sort. The only difference is that each round only continues toward the side where the pivot landed, instead of recursing into both.',
          )}</p>
        </div>
      </header>

      <div className={`qs-step ${step.phase}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="qs-workspace">
        <div className="qs-stage-card">
          <div className="qs-array">
            {step.array.map((value, index) => {
              const classes = [
                index === step.scanIndex ? 'scan' : '',
                index === step.storeIndex ? 'store' : '',
                index === step.pivotAt ? 'pivot' : '',
                (step.phase === 'place-pivot' && index === step.pivotIndex) ? 'pivot' : '',
                index === step.target ? 'target' : '',
                isFinalPivot(index) ? 'final' : '',
                isSettled(index) ? 'settled' : '',
              ].filter(Boolean).join(' ');
              return (
                <div className={`qs-cell ${classes}`} key={index}>
                  <span>{value}</span>
                  <em>{index}</em>
                </div>
              );
            })}
          </div>
          <div className="qs-meta">
            <div>
              <span>{t('区间', 'Range')}</span>
              <strong>[{step.left}, {step.right}]</strong>
            </div>
            <div>
              <span>{t('target', 'target')}</span>
              <strong>{step.target}</strong>
            </div>
            <div>
              <span>{t('轮次', 'Round')}</span>
              <strong>{step.round ?? 0}</strong>
            </div>
          </div>
        </div>

        <div className="qs-code" aria-label={t('当前 quickselect 代码', 'Current quickselect code')}>
          <div className="qs-code-heading">
            <span>{t('quickselect 模板', 'Quickselect template')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="qs-code-lines">
            {QUICKSELECT_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="qs-legend">
        <span><i className="pivot" />{t('pivot', 'pivot')}</span>
        <span><i className="scan" />{t('scan（正在比较）', 'scan (being compared)')}</span>
        <span><i className="store" />{t('store（小于 pivot 的边界）', 'store (boundary of values smaller than the pivot)')}</span>
        <span><i className="target" />{t('target（目标下标）', 'target (goal index)')}</span>
        <span><i className="settled" />{t('已排除，不会再访问', 'excluded, never visited again')}</span>
      </div>

      <div className="qs-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择 quickselect 演示步骤', 'Select a quickselect demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

function triePathPrefixes(path) {
  const prefixes = [''];
  for (let i = 1; i <= path.length; i += 1) prefixes.push(path.slice(0, i));
  return prefixes;
}

function TrieDiagram({ positions, visibleNodes, endNodes, nodeClass, edgeClass, ariaLabel, viewBox }) {
  const paths = Object.keys(positions);
  return (
    <svg aria-label={ariaLabel} className="trie-diagram" role="img" viewBox={viewBox}>
      {paths.filter((path) => path !== '').map((path) => {
        const parent = path.slice(0, -1);
        if (!visibleNodes.has(path) || !visibleNodes.has(parent)) return null;
        const source = positions[parent];
        const target = positions[path];
        return (
          <line
            className={`trie-edge ${edgeClass ? edgeClass(path) : ''}`}
            key={path}
            x1={source.x}
            x2={target.x}
            y1={source.y + 22}
            y2={target.y - 22}
          />
        );
      })}
      {paths.map((path) => {
        const pos = positions[path];
        const visible = visibleNodes.has(path);
        const isEnd = endNodes.has(path);
        const char = path === '' ? '•' : path[path.length - 1];
        const classes = [
          'trie-node',
          visible ? '' : 'ghost',
          isEnd ? 'end' : '',
          nodeClass ? nodeClass(path) : '',
        ].filter(Boolean).join(' ');
        return (
          <g className={classes} key={path || 'root'}>
            <circle cx={pos.x} cy={pos.y} r={isEnd ? 23 : 19} />
            {visible && <text dominantBaseline="middle" textAnchor="middle" x={pos.x} y={pos.y}>{char}</text>}
          </g>
        );
      })}
    </svg>
  );
}

const TRIE_CORE_WORDS = ['cat', 'car', 'card', 'dog'];
const TRIE_CORE_QUERIES = [
  { type: 'search', term: 'car' },
  { type: 'search', term: 'ca' },
  { type: 'startsWith', term: 'ca' },
  { type: 'search', term: 'cars' },
];
const TRIE_CORE_POSITIONS = {
  '': { x: 400, y: 40 },
  c: { x: 250, y: 120 },
  ca: { x: 250, y: 200 },
  cat: { x: 150, y: 280 },
  car: { x: 350, y: 280 },
  card: { x: 350, y: 360 },
  d: { x: 550, y: 120 },
  do: { x: 550, y: 200 },
  dog: { x: 550, y: 280 },
};

const TRIE_CORE_CODE_LINES = [
  { id: 'root', code: ['def insert(self, word):', '    node = self.root'] },
  { id: 'create-child', code: ['    for ch in word:', '        if ch not in node.children:', '            node.children[ch] = TrieNode()'] },
  { id: 'reuse-child', code: ['    for ch in word:', '        node = node.children[ch]'] },
  { id: 'mark-end', code: ['    node.is_end = True'] },
  { id: 'traverse-char', code: ['def _traverse(self, s):', '    node = self.root', '    for ch in s:', '        node = node.children[ch]'] },
  { id: 'traverse-fail', code: ['        if ch not in node.children:', '            return None'] },
  { id: 'check-end', code: ['def search(self, word):', '    node = self._traverse(word)', '    return node is not None and node.is_end'] },
  { id: 'check-prefix', code: ['def startsWith(self, prefix):', '    return self._traverse(prefix) is not None'] },
];

function buildTrieCoreSteps() {
  const steps = [];
  const nodes = new Set(['']);
  const endWords = new Set();

  steps.push({
    phase: 'intro',
    activeLine: 'root',
    nodes: [...nodes],
    endWords: [...endWords],
    highlightPath: [''],
    currentPath: '',
  });

  TRIE_CORE_WORDS.forEach((word) => {
    let path = '';
    for (let i = 0; i < word.length; i += 1) {
      const nextPath = path + word[i];
      const isNew = !nodes.has(nextPath);
      if (isNew) nodes.add(nextPath);
      steps.push({
        phase: 'insert-char',
        activeLine: isNew ? 'create-child' : 'reuse-child',
        word,
        char: word[i],
        isNew,
        nodes: [...nodes],
        endWords: [...endWords],
        highlightPath: triePathPrefixes(nextPath),
        currentPath: nextPath,
      });
      path = nextPath;
    }
    endWords.add(path);
    steps.push({
      phase: 'mark-end',
      activeLine: 'mark-end',
      word,
      nodes: [...nodes],
      endWords: [...endWords],
      highlightPath: triePathPrefixes(path),
      currentPath: path,
    });
  });

  TRIE_CORE_QUERIES.forEach((query) => {
    let path = '';
    let broken = false;
    steps.push({
      phase: 'query-start',
      activeLine: query.type === 'search' ? 'check-end' : 'check-prefix',
      query,
      nodes: [...nodes],
      endWords: [...endWords],
      highlightPath: [''],
      currentPath: '',
    });
    for (let i = 0; i < query.term.length; i += 1) {
      const nextPath = path + query.term[i];
      if (!nodes.has(nextPath)) {
        steps.push({
          phase: 'query-fail',
          activeLine: 'traverse-fail',
          query,
          failChar: query.term[i],
          nodes: [...nodes],
          endWords: [...endWords],
          highlightPath: triePathPrefixes(path),
          currentPath: path,
        });
        broken = true;
        break;
      }
      path = nextPath;
      steps.push({
        phase: 'query-step',
        activeLine: 'traverse-char',
        query,
        nodes: [...nodes],
        endWords: [...endWords],
        highlightPath: triePathPrefixes(path),
        currentPath: path,
      });
    }
    const result = broken ? false : (query.type === 'search' ? endWords.has(path) : true);
    steps.push({
      phase: 'query-result',
      activeLine: query.type === 'search' ? 'check-end' : 'check-prefix',
      query,
      result,
      nodes: [...nodes],
      endWords: [...endWords],
      highlightPath: broken ? [] : triePathPrefixes(path),
      currentPath: broken ? null : path,
    });
  });

  return steps;
}

const TRIE_CORE_STEPS = buildTrieCoreSteps();

function TrieCoreVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = TRIE_CORE_STEPS;
  const step = steps[activeStep];
  const visibleNodes = new Set(step.nodes);
  const endNodes = new Set(step.endWords);
  const highlightSet = new Set(step.highlightPath);

  const activeLineLabel = {
    root: t('从根节点开始', 'Start from the root'),
    'create-child': t('这个字符对应的子节点不存在，新建一个', "This character's child does not exist yet, so create one"),
    'reuse-child': t('这个字符对应的子节点已经存在，直接复用', "This character's child already exists, reuse it"),
    'mark-end': t('把当前节点标记为一个完整单词的结尾', 'Mark the current node as the end of a complete word'),
    'traverse-char': t('沿着已有的子节点继续前进', 'Continue along an existing child'),
    'traverse-fail': t('当前字符没有对应的子节点，路径中断', 'No child exists for this character; the path breaks'),
    'check-end': t('路径完整之后，还要检查 is_end 是否为真', 'Once the path completes, also check whether is_end is true'),
    'check-prefix': t('路径存在就够了，不需要检查 is_end', 'Existence of the path is enough; is_end is never checked'),
  }[step.activeLine];

  let title;
  let detail;
  if (step.phase === 'intro') {
    title = t('空 Trie，只有一个根节点', 'An empty trie, just the root node');
    detail = t('根节点不代表任何字符，它是所有单词共享的起点。', 'The root represents no character; it is the shared starting point for every word.');
  } else if (step.phase === 'insert-char') {
    title = step.isNew
      ? t(`insert("${step.word}")：字符 '${step.char}' 新建节点`, `insert("${step.word}"): character '${step.char}' creates a new node`)
      : t(`insert("${step.word}")：字符 '${step.char}' 复用已有节点`, `insert("${step.word}"): character '${step.char}' reuses an existing node`);
    detail = step.isNew
      ? t('这条路径第一次出现这个字符，必须新建节点才能继续往下走。', 'This character has never appeared on this path before, so a new node has to be created to continue.')
      : t('之前插入的单词已经建过这个节点，共享前缀不需要重复创建。', 'An earlier insert already created this node; a shared prefix never needs to be built twice.');
  } else if (step.phase === 'mark-end') {
    title = t(`insert("${step.word}")：标记结尾`, `insert("${step.word}"): mark the end`);
    detail = t('节点本身只代表一个字符，is_end 才是"这里是一个完整单词"的标记。', 'A node only represents one character; is_end is the marker that says "a complete word ends here."');
  } else if (step.phase === 'query-start') {
    const label = step.query.type === 'search' ? 'search' : 'startsWith';
    title = t(`${label}("${step.query.term}")：从根节点开始查找`, `${label}("${step.query.term}"): start the lookup from the root`);
    detail = t('查找和插入用同一套移动规则，区别只在没有子节点时要不要新建。', 'Lookup follows the same movement rule as insert; the only difference is that a missing child is never created.');
  } else if (step.phase === 'query-step') {
    title = t(`匹配到 '${step.currentPath[step.currentPath.length - 1]}'，继续前进`, `Matched '${step.currentPath[step.currentPath.length - 1]}', continue`);
    detail = t('子节点存在，指针移动到它，继续处理下一个字符。', 'The child exists, so the pointer moves onto it and the next character is processed.');
  } else if (step.phase === 'query-fail') {
    title = t(`没有 '${step.failChar}' 这个子节点，直接判否`, `No child for '${step.failChar}'; the answer is immediately no`);
    detail = t('路径在这里断掉，后面不管还剩多少字符都不用再看了。', 'The path breaks here; whatever characters remain in the query never need to be examined.');
  } else {
    const label = step.query.type === 'search' ? 'search' : 'startsWith';
    title = t(
      `${label}("${step.query.term}") 返回 ${step.result ? 'True' : 'False'}`,
      `${label}("${step.query.term}") returns ${step.result ? 'True' : 'False'}`,
    );
    detail = step.query.type === 'search'
      ? t(
        step.result
          ? '路径完整存在，并且终点节点的 is_end 为真。'
          : step.currentPath === null
            ? '路径在中途断开了，说明这个字符串根本没有被插入过，也不是任何单词的前缀。'
            : '路径完整存在，但终点节点的 is_end 为假：这里只是别的单词的前缀，不是一个完整单词。',
        step.result
          ? 'The path exists in full, and the node it ends on has is_end set to true.'
          : step.currentPath === null
            ? 'The path broke partway through, which means this string was never inserted and is not a prefix of anything either.'
            : 'The path exists in full, but the node it ends on has is_end set to false: it is only a prefix of some other word, not a complete word on its own.',
      )
      : t(
        step.result ? '路径存在，不需要关心终点是不是完整单词。' : '路径在中途断开，不存在这个前缀。',
        step.result ? 'The path exists; whether it happens to end a complete word is irrelevant.' : 'The path breaks partway through; this prefix does not exist.',
      );
  }

  return (
    <section aria-label={t('Trie 插入与查找逐步演示', 'Step-through: trie insertion and lookup')} className="tc-visual">
      <header className="tc-header">
        <div>
          <p className="eyebrow">{t('Trie，共享前缀 + 结尾标记', 'Trie: shared prefixes plus an end-of-word marker')}</p>
          <h2>{t('插入 cat / car / card / dog', 'Insert cat / car / card / dog')}</h2>
          <p>{t(
            '先看这四个单词怎么共享节点建成一棵树，再看 search 和 startsWith 怎么复用同一套移动规则。',
            'First watch how these four words share nodes to form one tree, then watch how search and startsWith reuse the exact same movement rule.',
          )}</p>
        </div>
      </header>

      <div className={`tc-step ${step.phase}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="tc-workspace">
        <div className="tc-stage-card">
          <TrieDiagram
            ariaLabel={t('Trie 结构图', 'Trie structure diagram')}
            edgeClass={(path) => (highlightSet.has(path) ? 'active' : '')}
            endNodes={endNodes}
            nodeClass={(path) => [
              path === step.currentPath ? 'current' : '',
              path !== step.currentPath && highlightSet.has(path) ? 'active' : '',
              step.phase === 'insert-char' && step.isNew && path === step.currentPath ? 'new' : '',
              step.phase === 'query-fail' ? 'fail' : '',
            ].filter(Boolean).join(' ')}
            positions={TRIE_CORE_POSITIONS}
            viewBox="0 0 700 400"
            visibleNodes={visibleNodes}
          />
        </div>

        <div aria-label={t('当前代码', 'Current code')} className="tc-code">
          <div className="tc-code-heading">
            <span>{t('insert / search / startsWith', 'insert / search / startsWith')}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="tc-code-lines">
            {TRIE_CORE_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="tc-legend">
        <span><i className="current" />{t('当前节点', 'Current node')}</span>
        <span><i className="active" />{t('本次经过的路径', 'Path visited this step')}</span>
        <span><i className="new" />{t('本步新建', 'Created this step')}</span>
        <span><i className="end" />{t('完整单词结尾（is_end）', 'End of a complete word (is_end)')}</span>
        <span><i className="ghost" />{t('还未创建', 'Not created yet')}</span>
      </div>

      <div className="tc-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择 Trie 演示步骤', 'Select a trie demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const TRIE_WILDCARD_WORDS = ['bad', 'dad', 'cat'];
const TRIE_WILDCARD_QUERY = '.at';
const TRIE_WILDCARD_POSITIONS = {
  '': { x: 400, y: 40 },
  b: { x: 200, y: 120 },
  ba: { x: 200, y: 200 },
  bad: { x: 200, y: 280 },
  d: { x: 400, y: 120 },
  da: { x: 400, y: 200 },
  dad: { x: 400, y: 280 },
  c: { x: 600, y: 120 },
  ca: { x: 600, y: 200 },
  cat: { x: 600, y: 280 },
};

const TRIE_WILDCARD_CODE_LINES = [
  { id: 'base-case', code: ['def dfs(node, i):', '    if i == len(word):', '        return node.is_end'] },
  { id: 'branch', code: ['    ch = word[i]', '    if ch == ".":', '        return any(', '            dfs(child, i + 1)', '            for child in node.children.values()', '        )'] },
  { id: 'literal-miss', code: ['    if ch not in node.children:', '        return False'] },
  { id: 'literal-continue', code: ['    return dfs(node.children[ch], i + 1)'] },
];

function buildTrieWildcardSteps() {
  const nodes = new Set(['']);
  const endWords = new Set();
  TRIE_WILDCARD_WORDS.forEach((word) => {
    let path = '';
    for (const ch of word) {
      path += ch;
      nodes.add(path);
    }
    endWords.add(path);
  });

  const rootChildren = [];
  TRIE_WILDCARD_WORDS.forEach((word) => {
    if (!rootChildren.includes(word[0])) rootChildren.push(word[0]);
  });

  const steps = [];
  const branchStatus = {};
  rootChildren.forEach((c) => { branchStatus[c] = 'pending'; });

  steps.push({
    phase: 'intro',
    activeLine: 'base-case',
    highlightPath: [''],
    currentPath: null,
    branchStatus: { ...branchStatus },
  });

  steps.push({
    phase: 'wildcard-branch',
    activeLine: 'branch',
    highlightPath: ['', ...rootChildren],
    currentPath: null,
    branchStatus: { ...branchStatus, ...Object.fromEntries(rootChildren.map((c) => [c, 'active'])) },
  });

  let solved = false;

  rootChildren.forEach((c0) => {
    if (solved) {
      return;
    }
    branchStatus[c0] = 'active';
    let path = c0;
    steps.push({
      phase: 'branch-enter',
      activeLine: 'literal-continue',
      branch: c0,
      highlightPath: triePathPrefixes(path),
      currentPath: path,
      branchStatus: { ...branchStatus },
    });

    let failed = false;
    for (let i = 1; i < TRIE_WILDCARD_QUERY.length; i += 1) {
      const ch = TRIE_WILDCARD_QUERY[i];
      const nextPath = path + ch;
      if (!nodes.has(nextPath)) {
        steps.push({
          phase: 'branch-fail',
          activeLine: 'literal-miss',
          branch: c0,
          failChar: ch,
          highlightPath: triePathPrefixes(path),
          currentPath: path,
          branchStatus: { ...branchStatus },
        });
        failed = true;
        break;
      }
      path = nextPath;
      steps.push({
        phase: 'branch-step',
        activeLine: 'literal-continue',
        branch: c0,
        highlightPath: triePathPrefixes(path),
        currentPath: path,
        branchStatus: { ...branchStatus },
      });
    }

    const success = !failed && endWords.has(path);
    branchStatus[c0] = success ? 'success' : 'failed';
    steps.push({
      phase: 'branch-result',
      activeLine: failed ? 'literal-miss' : 'base-case',
      branch: c0,
      result: success,
      highlightPath: triePathPrefixes(path),
      currentPath: path,
      branchStatus: { ...branchStatus },
    });

    if (success) {
      solved = true;
    }
  });

  steps.push({
    phase: 'done',
    activeLine: 'branch',
    highlightPath: [],
    currentPath: null,
    branchStatus: { ...branchStatus },
    result: solved,
  });

  return steps;
}

const TRIE_WILDCARD_STEPS = buildTrieWildcardSteps();

function TrieWildcardVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = TRIE_WILDCARD_STEPS;
  const step = steps[activeStep];
  const visibleNodes = new Set(Object.keys(TRIE_WILDCARD_POSITIONS));
  const endNodes = new Set(
    TRIE_WILDCARD_WORDS.map((word) => word),
  );
  const highlightSet = new Set(step.highlightPath);

  const activeLineLabel = {
    'base-case': t('字符已经用完，检查这个节点是不是单词结尾', 'The characters are exhausted; check whether this node ends a word'),
    branch: t('当前字符是通配符，遍历所有子节点分别递归', 'The current character is a wildcard; recurse into every child'),
    'literal-miss': t('普通字符，但没有对应的子节点', 'An ordinary character, but no matching child exists'),
    'literal-continue': t('普通字符，沿着对应的子节点继续递归', 'An ordinary character; recurse into the matching child'),
  }[step.activeLine];

  let title;
  let detail;
  if (step.phase === 'intro') {
    title = t(`WordDictionary 里已有 bad / dad / cat，查询 ".at"`, `WordDictionary already holds bad / dad / cat; query ".at"`);
    detail = t('第 0 个字符是通配符，可以匹配任意一个子节点。', 'Character 0 is a wildcard, which can match any child.');
  } else if (step.phase === 'wildcard-branch') {
    title = t(`遇到 "."：分别尝试 ${Object.keys(step.branchStatus).map((c) => `'${c}'`).join(', ')}`, `Hitting ".": try ${Object.keys(step.branchStatus).map((c) => `'${c}'`).join(', ')} in turn`);
    detail = t(
      '这一步对应代码里的 any(...)：只要有一个分支返回 True，整体就返回 True。',
      'This corresponds to any(...) in the code: if a single branch returns True, the whole call returns True.',
    );
  } else if (step.phase === 'branch-enter') {
    title = t(`尝试分支 '${step.branch}'`, `Trying branch '${step.branch}'`);
    detail = t('从这个子节点开始，继续匹配查询串里剩下的普通字符。', "Starting from this child, continue matching the rest of the query's ordinary characters.");
  } else if (step.phase === 'branch-step') {
    title = t(`分支 '${step.branch}'：匹配到 '${step.currentPath[step.currentPath.length - 1]}'`, `Branch '${step.branch}': matched '${step.currentPath[step.currentPath.length - 1]}'`);
    detail = t('这个位置不是通配符，只需要检查这一个子节点是否存在。', 'This position is not a wildcard, so only this one child needs to be checked.');
  } else if (step.phase === 'branch-fail') {
    title = t(`分支 '${step.branch}'：没有 '${step.failChar}' 这个子节点，本分支返回 False`, `Branch '${step.branch}': no child for '${step.failChar}'; this branch returns False`);
    detail = t('这个分支走不通，但不影响其他分支继续尝试。', "This branch is a dead end, but it does not stop the other branches from being tried.");
  } else if (step.phase === 'branch-result') {
    title = step.result
      ? t(`分支 '${step.branch}'：到达单词结尾，返回 True`, `Branch '${step.branch}': reached the end of a word, returns True`)
      : t(`分支 '${step.branch}'：路径存在但不是完整单词，返回 False`, `Branch '${step.branch}': the path exists but is not a complete word, returns False`);
    detail = step.result
      ? t('any(...) 拿到一个 True，短路返回，不用再尝试剩下的分支。', 'any(...) receives a True and short-circuits; the remaining branches never run.')
      : t('这个分支的结果是 False，继续看下一个分支。', 'This branch resolves to False, so the next branch is tried.');
  } else {
    title = t(
      `search(".at") 返回 ${step.result ? 'True' : 'False'}`,
      `search(".at") returns ${step.result ? 'True' : 'False'}`,
    );
    detail = t(
      '"bad" 和 "dad" 的分支都在第二个字符处失配，"cat" 的分支走到底并且是完整单词，所以整体是 True。',
      'The "bad" and "dad" branches both fail at the second character, while the "cat" branch reaches a complete word, so the overall result is True.',
    );
  }

  return (
    <section aria-label={t('通配符查找逐步演示', 'Step-through: wildcard search')} className="ws-visual">
      <header className="ws-header">
        <div>
          <p className="eyebrow">{t('通配符 "." 触发多分支 DFS', 'A wildcard "." triggers a multi-branch DFS')}</p>
          <h2>{t('search(".at")', 'search(".at")')}</h2>
          <p>{t(
            '普通字符只走一条路径，通配符会在当前节点的所有子节点上分别递归，任意一个分支成功就返回 True。',
            'An ordinary character follows a single path. A wildcard recurses into every child of the current node, and the call returns True as soon as any branch succeeds.',
          )}</p>
        </div>
      </header>

      <div className={`ws-step ${step.phase}`} aria-live="polite">
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="ws-workspace">
        <div className="ws-stage-card">
          <TrieDiagram
            ariaLabel={t('Trie 结构图', 'Trie structure diagram')}
            edgeClass={(path) => (highlightSet.has(path) ? 'active' : '')}
            endNodes={endNodes}
            nodeClass={(path) => {
              const branchChar = path.length > 0 ? path[0] : null;
              const status = branchChar ? step.branchStatus[branchChar] : null;
              return [
                path === step.currentPath ? 'current' : '',
                path !== step.currentPath && highlightSet.has(path) ? 'active' : '',
                status === 'success' ? 'branch-success' : '',
                status === 'failed' && highlightSet.has(path) ? 'branch-failed' : '',
              ].filter(Boolean).join(' ');
            }}
            positions={TRIE_WILDCARD_POSITIONS}
            viewBox="0 0 800 340"
            visibleNodes={visibleNodes}
          />
          <div className="ws-branches">
            {Object.entries(step.branchStatus).map(([char, status]) => (
              <div className={`ws-branch ${status}`} key={char}>
                <span>'{char}'</span>
                <strong>{{
                  pending: t('待定', 'pending'),
                  active: t('尝试中', 'trying'),
                  success: t('成功', 'success'),
                  failed: t('失败', 'failed'),
                }[status]}</strong>
              </div>
            ))}
          </div>
        </div>

        <div aria-label={t('当前代码', 'Current code')} className="ws-code">
          <div className="ws-code-heading">
            <span>{t('search 的 DFS 模板', "search's DFS template")}</span>
            <strong>{activeLineLabel}</strong>
          </div>
          <div className="ws-code-lines">
            {TRIE_WILDCARD_CODE_LINES.map((line) => (
              <div
                aria-current={step.activeLine === line.id ? 'step' : undefined}
                className={step.activeLine === line.id ? 'active' : ''}
                key={line.id}
              >
                {line.code.map((code) => <code key={code}>{code}</code>)}
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="ws-legend">
        <span><i className="current" />{t('当前节点', 'Current node')}</span>
        <span><i className="active" />{t('本次经过的路径', 'Path visited this step')}</span>
        <span><i className="branch-success" />{t('成功分支', 'Successful branch')}</span>
        <span><i className="branch-failed" />{t('失败分支', 'Failed branch')}</span>
      </div>

      <div className="ws-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择通配符演示步骤', 'Select a wildcard demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const BACKTRACKING_PATTERNS = [
  {
    id: 'subsets',
    number: 78,
    title: 'Subsets',
    tone: 'subset',
    pattern: '子集型',
    signature: 'backtrack(start)',
    choices: 'nums[start:]',
    recurse: 'backtrack(i + 1)',
    collect: '每进入一个节点就收一次，没有 base case',
    guard: '输入互不相同，不需要去重',
    prune: '无',
    size: '2^n 个节点',
  },
  {
    id: 'permutations',
    number: 46,
    title: 'Permutations',
    tone: 'permute',
    pattern: '排列型',
    signature: 'backtrack() + used[]',
    choices: '所有 used[i] == False 的下标',
    recurse: 'backtrack()，靠 used 排除已选',
    collect: 'len(path) == len(nums) 时收',
    guard: '输入互不相同，不需要去重',
    prune: '无',
    size: 'n! 个叶子',
  },
  {
    id: 'combination-sum',
    number: 39,
    title: 'Combination Sum',
    tone: 'combo',
    pattern: '组合型（元素可复用）',
    signature: 'backtrack(start, remain)',
    choices: 'candidates[start:]',
    recurse: 'backtrack(i, ...)，传 i 允许再选自己',
    collect: 'remain == 0 时收',
    guard: '输入互不相同，不需要去重',
    prune: '排序后 candidates[i] > remain 直接 break',
    size: '深度约 target / min(candidates)',
  },
  {
    id: 'subsets-ii',
    number: 90,
    title: 'Subsets II',
    tone: 'dedup',
    pattern: '子集型 + 去重',
    signature: 'backtrack(start)，先 nums.sort()',
    choices: 'nums[start:]，跳过同层重复值',
    recurse: 'backtrack(i + 1)',
    collect: '每进入一个节点就收一次',
    guard: 'if i > start and nums[i] == nums[i-1]: continue',
    prune: '同层去重本身就是剪枝',
    size: '小于 2^n',
  },
  {
    id: 'combination-sum-ii',
    number: 40,
    title: 'Combination Sum II',
    tone: 'dedup',
    pattern: '组合型 + 去重',
    signature: 'backtrack(start, remain)，先排序',
    choices: 'candidates[start:]，跳过同层重复值',
    recurse: 'backtrack(i + 1, ...)，每个下标只用一次',
    collect: 'remain == 0 时收',
    guard: 'if i > start and 值与前一个相同: continue',
    prune: 'candidates[i] > remain 时 break',
    size: '小于 2^n',
  },
  {
    id: 'generate-parentheses',
    number: 22,
    title: 'Generate Parentheses',
    tone: 'bracket',
    pattern: '约束构造型 / 前缀平衡',
    signature: 'backtrack(open, close)',
    choices: "两个选择：'(' 或 ')'",
    recurse: '放 ( 则 open + 1，放 ) 则 close + 1',
    collect: 'open == n and close == n 时收',
    guard: 'open < n 才能放 (；close < open 才能放 )',
    prune: '只要 close == open 就绝不能放 )（避免前缀失衡）',
    size: '第 n 个卡特兰数 C_n = (2n)! / ((n+1)! n!)',
  },
  {
    id: 'letter-combinations',
    number: 17,
    title: 'Letter Combinations',
    tone: 'product',
    pattern: '笛卡尔积型',
    signature: 'backtrack(index)',
    choices: 'keypad[digits[index]]',
    recurse: 'backtrack(index + 1)，层号即下标',
    collect: 'index == len(digits) 时收',
    guard: '不涉及去重',
    prune: '无，整棵树都是答案',
    size: '最多 4^n 个叶子',
  },
  {
    id: 'palindrome-partitioning',
    number: 131,
    title: 'Palindrome Partitioning',
    tone: 'cut',
    pattern: '切割型',
    signature: 'backtrack(start)',
    choices: '所有以 start 开头的子串 s[start:end+1]',
    recurse: 'backtrack(end + 1)，下一刀接着切',
    collect: 'start == len(s) 时收',
    guard: '不涉及去重',
    prune: '这一段不是回文就不递归',
    size: '最多 2^(n-1) 种切法',
  },
  {
    id: 'word-search',
    number: 79,
    title: 'Word Search',
    tone: 'grid',
    pattern: '网格型',
    signature: 'backtrack(r, c, index)',
    choices: '上下左右四个方向',
    recurse: 'backtrack(nr, nc, index + 1)',
    collect: 'index == len(word) 时返回 True',
    guard: '同一格子不能重复使用',
    prune: '字符不匹配、越界、已占用都直接返回 False',
    size: '每个起点一棵四叉树',
  },
  {
    id: 'n-queens',
    number: 51,
    title: 'N-Queens',
    tone: 'board',
    pattern: '棋盘型',
    signature: 'backtrack(row) + cols/diag/anti',
    choices: '这一行的 n 个列',
    recurse: 'backtrack(row + 1)，一层一行',
    collect: 'row == n 时收整张棋盘',
    guard: '按行放置，行冲突天然不存在',
    prune: '三个集合把冲突检查降到 O(1)',
    size: '上界 n!，实际远小于此',
  },
];

const BACKTRACKING_PATTERNS_EN = {
  subsets: {
    pattern: 'Subset',
    choices: 'nums[start:]',
    recurse: 'backtrack(i + 1)',
    collect: 'Collect once on entering every node; no base case',
    guard: 'Input is distinct, so no dedup needed',
    prune: 'None',
    size: '2^n nodes',
  },
  permutations: {
    pattern: 'Permutation',
    signature: 'backtrack() + used[]',
    choices: 'Every index where used[i] == False',
    recurse: 'backtrack(), with used excluding what is taken',
    collect: 'Collect when len(path) == len(nums)',
    guard: 'Input is distinct, so no dedup needed',
    prune: 'None',
    size: 'n! leaves',
  },
  'combination-sum': {
    pattern: 'Combination with reuse',
    choices: 'candidates[start:]',
    recurse: 'backtrack(i, ...) — passing i allows re-picking',
    collect: 'Collect when remain == 0',
    guard: 'Input is distinct, so no dedup needed',
    prune: 'After sorting, candidates[i] > remain allows break',
    size: 'Depth about target / min(candidates)',
  },
  'subsets-ii': {
    pattern: 'Subset + dedup',
    signature: 'backtrack(start) after nums.sort()',
    choices: 'nums[start:], skipping same-level repeats',
    recurse: 'backtrack(i + 1)',
    collect: 'Collect once on entering every node',
    guard: 'if i > start and nums[i] == nums[i-1]: continue',
    prune: 'Same-level dedup is itself the pruning',
    size: 'Fewer than 2^n',
  },
  'combination-sum-ii': {
    pattern: 'Combination + dedup',
    signature: 'backtrack(start, remain) after sorting',
    choices: 'candidates[start:], skipping same-level repeats',
    recurse: 'backtrack(i + 1, ...) — each index used once',
    collect: 'Collect when remain == 0',
    guard: 'if i > start and value equals the previous: continue',
    prune: 'break once candidates[i] > remain',
    size: 'Fewer than 2^n',
  },
  'generate-parentheses': {
    pattern: 'Constrained construction / Prefix balance',
    signature: 'backtrack(open, close)',
    choices: "Two choices: '(' or ')'",
    recurse: 'open + 1 on adding (, close + 1 on adding )',
    collect: 'Collect when open == n and close == n',
    guard: 'open < n to add (; close < open to add )',
    prune: 'Never add ) when close == open (avoids illegal prefixes)',
    size: 'Catalan number C_n = (2n)! / ((n+1)! n!)',
  },
  'letter-combinations': {
    pattern: 'Cartesian product',
    signature: 'backtrack(index)',
    choices: 'keypad[digits[index]]',
    recurse: 'backtrack(index + 1) — the level is the index',
    collect: 'Collect when index == len(digits)',
    guard: 'No dedup involved',
    prune: 'None; the whole tree is answers',
    size: 'At most 4^n leaves',
  },
  'palindrome-partitioning': {
    pattern: 'Partition',
    signature: 'backtrack(start)',
    choices: 'Every substring s[start:end+1]',
    recurse: 'backtrack(end + 1) — the next cut follows',
    collect: 'Collect when start == len(s)',
    guard: 'No dedup involved',
    prune: 'A non-palindromic piece is never recursed into',
    size: 'At most 2^(n-1) partitions',
  },
  'word-search': {
    pattern: 'Grid',
    signature: 'backtrack(r, c, index)',
    choices: 'The four neighbouring cells',
    recurse: 'backtrack(nr, nc, index + 1)',
    collect: 'Return True when index == len(word)',
    guard: 'A cell cannot be reused within one path',
    prune: 'Mismatch, out of bounds, or occupied returns False',
    size: 'One 4-ary tree per start cell',
  },
  'n-queens': {
    pattern: 'Board',
    signature: 'backtrack(row) + cols/diag/anti',
    choices: 'The n columns of this row',
    recurse: 'backtrack(row + 1) — one level per row',
    collect: 'Collect the whole board when row == n',
    guard: 'Row conflicts are impossible by construction',
    prune: 'Three sets bring conflict checks down to O(1)',
    size: 'Bounded by n!, far smaller in practice',
  },
};

const GREEDY_PATTERNS = [
  {
    id: 'max-subarray',
    number: '53',
    title: 'Maximum Subarray',
    tone: 'kadane',
    signature: 'cur_sum = max(num, cur_sum + num)',
    pattern: '前缀重置 / Kadane 算法',
    invariant: '若历史累加和 cur_sum < 0，对后续子数组只有负贡献，必须果断归零重置',
    whyGreedy: '抛弃负前缀绝不会漏掉全局最大子数组（任何包含负前缀的解都可以通过切除该前缀变得更大）',
    coreState: 'cur_sum = max(x, cur_sum + x); max_sum = max(max_sum, cur_sum)',
    timeSpace: 'Time: O(n) | Space: O(1)',
    trap: '全为负数时不能初始化 max_sum = 0，必须初始化为 -infinity 或 nums[0]',
  },
  {
    id: 'jump-game',
    number: '55',
    title: 'Jump Game',
    tone: 'reach',
    signature: 'max_reach = max(max_reach, i + nums[i])',
    pattern: '最远可达边界单调维护',
    invariant: '只要当前下标 i <= max_reach，说明 i 是可达的；更新 max_reach 即可吞吐全部可能跳跃',
    whyGreedy: '我们不需要穷举每一步跳多少，只需维护“最远能跳到哪”这一外包络线（Envelope）',
    coreState: 'if i > max_reach: return False; max_reach = max(max_reach, i + nums[i])',
    timeSpace: 'Time: O(n) | Space: O(1)',
    trap: '遇到 0 不必特殊回溯，只要 max_reach 跨过该 0 即可继续前进；若 i > max_reach 则被困死',
  },
  {
    id: 'jump-game-ii',
    number: '45',
    title: 'Jump Game II',
    tone: 'window',
    signature: 'steps += 1; cur_end = farthest',
    pattern: '隐式 BFS / 层次最远窗口贪心',
    invariant: '第 k 步能覆盖的区间为 [cur_start, cur_end]；在当前区间内求出第 k+1 步的最远边界 farthest',
    whyGreedy: '每一步都贪心地在该层覆盖范围内收集能跳到下一层的最大上限，步数必然最少',
    coreState: 'farthest = max(farthest, i + nums[i]); if i == cur_end: steps += 1; cur_end = farthest',
    timeSpace: 'Time: O(n) | Space: O(1)',
    trap: '循环只需遍历到 n - 2，若遍历到 n - 1 会在刚好到达终点时多触发一次无意义的 steps += 1',
  },
  {
    id: 'gas-station',
    number: '134',
    title: 'Gas Station',
    tone: 'circuit',
    signature: 'if tank < 0: start = i + 1; tank = 0',
    pattern: '总净赤字校验 + 局部断点跳跃',
    invariant: '若 sum(gas) >= sum(cost)，必存在唯一解；从 start 出发若在 i 处断油，则 [start, i] 内所有点都无法作为起点',
    whyGreedy: '从 start 走到中间任意点 k 时剩余油量 >= 0；若从 k 出发相当于少了 start 积累的油，只会更早断油，因此起点直接跳至 i+1',
    coreState: 'tank += gas[i] - cost[i]; if tank < 0: start = i + 1; tank = 0',
    timeSpace: 'Time: O(n) | Space: O(1)',
    trap: '必须同时记录 total_surplus += gas[i] - cost[i]，遍历结束后若 total_surplus < 0 必须返回 -1',
  },
  {
    id: 'hand-of-straights',
    number: '846',
    title: 'Hand of Straights',
    tone: 'forced',
    signature: 'count[first + k] -= count[first]',
    pattern: '最小元素强制开顺子 / 频次切片',
    invariant: '当前剩余的全局最小牌 x 在任何合法顺子里都绝不可能出现在中间或末尾，必须作为顺子起点',
    whyGreedy: '因为不存在 x - 1，所以所有 count[x] 张牌都必须开新顺子，具有 100% 强制性',
    coreState: 'for card in range(first, first + groupSize): if count[card] < need: return False',
    timeSpace: 'Time: O(n log n) | Space: O(n)',
    trap: '总牌数无法整除 groupSize 时直接返回 False；遍历堆顶时需跳过 count 已被减为 0 的历史牌',
  },
  {
    id: 'merge-triplets',
    number: '1899',
    title: 'Merge Triplets to Form Target',
    tone: 'filter',
    signature: 'has_a |= (t[0]==target[0]) ...',
    pattern: '坐标独立性 + 单调超限剔除',
    invariant: '任何分量超过 target[0..2] 的三元组永久禁用；其余安全三元组各分量独立，取 max 绝不会超标',
    whyGreedy: 'max 操作单调不减；只要能分别在安全三元组中找到匹配 target[0], target[1], target[2] 的三元组，全合并即可',
    coreState: 'if t[0] <= target[0] and t[1] <= target[1] and t[2] <= target[2]: match |= (t == target)',
    timeSpace: 'Time: O(n) | Space: O(1)',
    trap: '误以为需要严格找到单个三元组全匹配，实际上只要安全三元组在 3 个坐标上分别达标即可合并',
  },
  {
    id: 'partition-labels',
    number: '763',
    title: 'Partition Labels',
    tone: 'partition',
    signature: 'end = max(end, last[char]); if i == end: cut()',
    pattern: '字符最后出现位置与区间合并',
    invariant: '当前片段必须延伸到其中所有出现过的字符的最后一次出现位置的最大值 end',
    whyGreedy: '当扫描到达 i == end 时，当前片段内所有字符在后续都不会再出现，此时切断保证片段数量最多且每个片段最短',
    coreState: 'end = max(end, last[s[i]]); if i == end: res.append(i - start + 1); start = i + 1',
    timeSpace: 'Time: O(n) | Space: O(1) (26 chars)',
    trap: '必须先完整做一遍预处理得到每个字符的 last 索引，不能在一次遍历中边猜边切',
  },
  {
    id: 'valid-parenthesis-string',
    number: '678',
    title: 'Valid Parenthesis String',
    tone: 'range',
    signature: 'cmin = max(0, cmin - 1); cmax += 1',
    pattern: '未匹配左括号数量范围追踪 [cmin, cmax]',
    invariant: '遇到通配符 * 时，左括号需求量从单一确定值变为连续区间 [cmin, cmax]',
    whyGreedy: '将 * 分别视为 )、空字符、(，只需追踪可能的最少未匹配左括号 cmin 和最多 cmax；cmin 下限截断至 0',
    coreState: 'if cmax < 0: return False; cmin = max(cmin, 0); # return cmin == 0',
    timeSpace: 'Time: O(n) | Space: O(1)',
    trap: '中途 cmax < 0 说明把所有 * 当做 ( 都不够抵消右括号，必不合法；遍历结束只有 cmin == 0 才能完全闭合',
  },
];

const GREEDY_PATTERNS_EN = {
  'max-subarray': {
    pattern: 'Prefix Reset / Kadane',
    invariant: 'If running sum cur_sum < 0, it only drags future subarrays down; reset to zero immediately',
    whyGreedy: 'Discarding negative prefixes never misses the optimal subarray (chopping off a negative prefix only increases subarray sum)',
    coreState: 'cur_sum = max(x, cur_sum + x); max_sum = max(max_sum, cur_sum)',
    trap: 'For all-negative arrays, do not initialize max_sum = 0; initialize to -infinity or nums[0]',
  },
  'jump-game': {
    pattern: 'Monotonic Reachable Envelope',
    invariant: 'As long as i <= max_reach, index i is reachable; extending max_reach encapsulates all valid paths',
    whyGreedy: 'No need to branch over every jump step; maintaining the farthest envelope suffices',
    coreState: 'if i > max_reach: return False; max_reach = max(max_reach, i + nums[i])',
    trap: 'Zeros do not require backtracking unless i > max_reach (i.e. stuck completely)',
  },
  'jump-game-ii': {
    pattern: 'Implicit BFS Level Window',
    invariant: 'k-th jump covers [cur_start, cur_end]; find the farthest reach for the (k+1)-th jump within this window',
    whyGreedy: 'Greedily collecting the maximum reach per jump level guarantees minimum jump count',
    coreState: 'farthest = max(farthest, i + nums[i]); if i == cur_end: steps += 1; cur_end = farthest',
    trap: 'Loop up to n - 2; looping to n - 1 causes an extra false jump at the finish line',
  },
  'gas-station': {
    pattern: 'Total Deficit Check + Candidate Jump',
    invariant: 'If sum(gas) >= sum(cost), a unique start exists; if fuel drops < 0 at i, no index in [start, i] can be the start',
    whyGreedy: 'From start to any intermediate k had >= 0 fuel; starting from k with 0 fuel exhausts even sooner, so skip to i + 1',
    coreState: 'tank += gas[i] - cost[i]; if tank < 0: start = i + 1; tank = 0',
    trap: 'Must record total_surplus; if total_surplus < 0 after the full loop, return -1',
  },
  'hand-of-straights': {
    pattern: 'Forced Move / Minimum Key Slicing',
    invariant: 'The minimum remaining card x has no predecessor x-1, so it must start count[x] new straights',
    whyGreedy: 'Because x cannot fit anywhere else in any valid configuration, treating it as straight start is 100% forced',
    coreState: 'for card in range(first, first + groupSize): if count[card] < need: return False',
    trap: 'Return False early if len % groupSize != 0; skip heap tops whose count is already 0',
  },
  'merge-triplets': {
    pattern: 'Coordinate Independence + Disqualification',
    invariant: 'Triplets with any component > target[k] are permanently disqualified; safe triplets never exceed target under max',
    whyGreedy: 'max is non-decreasing; safe triplets matching target[0], target[1], target[2] can all be merged together safely',
    coreState: 'if t[0] <= target[0] and t[1] <= target[1] and t[2] <= target[2]: match |= (t == target)',
    trap: 'Do not search for a single triplet matching all 3 coordinates; components can be matched from separate safe triplets',
  },
  'partition-labels': {
    pattern: 'Last Occurrence & Interval Merging',
    invariant: 'Current partition must stretch to cover the maximum last[c] of all characters seen so far',
    whyGreedy: 'When i == end, all characters in [start, end] will never appear again; cutting now yields the maximum count of valid segments',
    coreState: 'end = max(end, last[s[i]]); if i == end: res.append(i - start + 1); start = i + 1',
    trap: 'Precompute last occurrence map upfront before making partition decisions',
  },
  'valid-parenthesis-string': {
    pattern: 'Unclosed Left-Bracket Range Tracking',
    invariant: 'Wildcard * turns the count of unclosed ( into a continuous range [cmin, cmax]',
    whyGreedy: 'Track [cmin, cmax] bounds where * acts as ), empty, or (; clamp cmin at 0',
    coreState: 'if cmax < 0: return False; cmin = max(cmin, 0); # return cmin == 0',
    trap: 'cmax < 0 means even all * as ( cannot balance ); at the end only cmin == 0 is valid',
  },
};

function GreedyPatternAtlas() {
  const { isEnglish, t } = useUiCopy();
  const [activePattern, setActivePattern] = useState('max-subarray');
  const basePattern = GREEDY_PATTERNS.find(({ id }) => id === activePattern)
    ?? GREEDY_PATTERNS[0];
  const pattern = isEnglish
    ? { ...basePattern, ...GREEDY_PATTERNS_EN[basePattern.id] }
    : basePattern;

  return (
    <section
      aria-label={t('八道贪心题全景对照', 'Eight greedy problems compared')}
      className={`gp-atlas ${pattern.tone}`}
    >
      <header className="gp-header">
        <div>
          <p className="eyebrow">{t('贪心核心心智模型', 'Core Greedy Mental Model')}</p>
          <h2>{t('强制选择 · 最远包络 · 前缀断点重置', 'Forced Move · Reachable Envelope · Prefix Reset')}</h2>
          <p>{t(
            '可靠的贪心永远建立在不变量与替换论证上，绝非局部盲目求快。',
            'Sound greedy choices always rely on invariants and exchange arguments, never blind local haste.',
          )}</p>
        </div>
        <code>{pattern.signature}</code>
      </header>

      <div aria-label={t('选择贪心题目', 'Choose a greedy problem')} className="gp-tabs" role="tablist">
        {GREEDY_PATTERNS.map((candidate) => (
          <button
            aria-selected={candidate.id === activePattern}
            className={candidate.id === activePattern ? 'active' : ''}
            key={candidate.id}
            onClick={() => setActivePattern(candidate.id)}
            role="tab"
            type="button"
          >
            <span>LC {candidate.number}</span>
            <strong>{candidate.title}</strong>
          </button>
        ))}
      </div>

      <div className="gp-summary">
        <div>
          <span>{t('决策模式', 'Decision pattern')}</span>
          <strong>{pattern.pattern}</strong>
        </div>
        <div>
          <span>{t('复杂度', 'Complexity')}</span>
          <strong>{pattern.timeSpace}</strong>
        </div>
      </div>

      <div className="gp-flow">
        <div>
          <span>{t('1 · 核心不变量', '1 · Core Invariant')}</span>
          <strong>{pattern.invariant}</strong>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('2 · 为什么贪心不漏解', '2 · Why Greedy Works')}</span>
          <strong>{pattern.whyGreedy}</strong>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('3 · 状态转移', '3 · State Transition')}</span>
          <code>{pattern.coreState}</code>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('4 · 易错陷阱', '4 · Common Pitfall')}</span>
          <strong className="gp-trap">{pattern.trap}</strong>
        </div>
      </div>
    </section>
  );
}

const KADANE_STEPS = [
  { i: 0, val: -2, curSum: -2, maxSum: -2, reset: true, subStart: 0, subEnd: 0, desc: '初始元素 -2：cur_sum = -2 < 0。前缀和为负，立即止损归零重置，max_sum = -2。', descEn: 'Initial element -2: cur_sum = -2 < 0. Negative prefix causes drag; reset to 0 immediately, max_sum = -2.' },
  { i: 1, val: 1, curSum: 1, maxSum: 1, reset: false, subStart: 1, subEnd: 1, desc: '遇到 1：从 0 开始加上 1 -> cur_sum = 1 > 0。形成有效正收益，更新 max_sum = 1。', descEn: 'Element 1: start fresh from 0 + 1 -> cur_sum = 1 > 0. Positive momentum formed, update max_sum = 1.' },
  { i: 2, val: -3, curSum: -2, maxSum: 1, reset: true, subStart: 1, subEnd: 2, desc: '遇到 -3：cur_sum = 1 + (-3) = -2 < 0。前缀再次沦为负资产，果断归零重置！max_sum 维持 1。', descEn: 'Element -3: cur_sum = 1 + (-3) = -2 < 0. Prefix becomes a liability again; reset to 0! max_sum remains 1.' },
  { i: 3, val: 4, curSum: 4, maxSum: 4, reset: false, subStart: 3, subEnd: 3, desc: '遇到 4：从 0 开始加上 4 -> cur_sum = 4。开启全新的优质子数组 [4]，更新 max_sum = 4。', descEn: 'Element 4: 0 + 4 -> cur_sum = 4. Start a promising new subarray [4], update max_sum = 4.' },
  { i: 4, val: -1, curSum: 3, maxSum: 4, reset: false, subStart: 3, subEnd: 4, desc: '遇到 -1：cur_sum = 4 + (-1) = 3 > 0。尽管遇到负数，但历史前缀仍有 +3 净利润，继续保留子数组 [4, -1]！', descEn: 'Element -1: cur_sum = 4 + (-1) = 3 > 0. Despite negative num, historical profit remains +3, keep subarray [4, -1]!' },
  { i: 5, val: 2, curSum: 5, maxSum: 5, reset: false, subStart: 3, subEnd: 5, desc: '遇到 2：cur_sum = 3 + 2 = 5 > 0。正收益持续扩张，更新 max_sum = 5（子数组 [4, -1, 2]）。', descEn: 'Element 2: cur_sum = 3 + 2 = 5 > 0. Momentum expands, update max_sum = 5 for subarray [4, -1, 2].' },
  { i: 6, val: 1, curSum: 6, maxSum: 6, reset: false, subStart: 3, subEnd: 6, desc: '遇到 1：cur_sum = 5 + 1 = 6 > 0。达到全局峰值！更新 max_sum = 6（子数组 [4, -1, 2, 1]）。', descEn: 'Element 1: cur_sum = 5 + 1 = 6 > 0. Hits global maximum! Update max_sum = 6 for [4, -1, 2, 1].' },
  { i: 7, val: -5, curSum: 1, maxSum: 6, reset: false, subStart: 3, subEnd: 7, desc: '遇到 -5：cur_sum = 6 + (-5) = 1 > 0。虽然利润缩减为 1，但尚未变成负资产，max_sum 保持 6。', descEn: 'Element -5: cur_sum = 6 + (-5) = 1 > 0. Profit drops to 1 but not negative, max_sum remains 6.' },
  { i: 8, val: 4, curSum: 5, maxSum: 6, reset: false, subStart: 3, subEnd: 8, desc: '遇到 4：cur_sum = 1 + 4 = 5。全数组扫描结束，全局最大连续子数组为 [4, -1, 2, 1]，最大和为 6！', descEn: 'Element 4: cur_sum = 1 + 4 = 5. Full array scanned, global max subarray is [4, -1, 2, 1] with sum = 6!' },
];

function KadaneVisual() {
  const { isEnglish, t } = useUiCopy();
  const [stepIndex, setStepIndex] = useState(0);
  const step = KADANE_STEPS[stepIndex] ?? KADANE_STEPS[0];
  const nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4];

  return (
    <section aria-label={t('Kadane 算法前缀动量与重置演示', 'Kadane algorithm momentum and reset walkthrough')} className="kadane-vis">
      <header className="kadane-header">
        <div>
          <p className="eyebrow">{t('前缀动量 vs 负债归零', 'Prefix Momentum vs Liability Reset')}</p>
          <h2>{t('Kadane 算法：正向利润累加与负前缀即时止损', 'Kadane\'s Algorithm: Accumulate Profit & Reset Negative Drag')}</h2>
          <p>{t(
            '当 cur_sum > 0 时带入下一项是有益资本；当 cur_sum < 0 时沦为负资产，切除后未来总和必更大。',
            'When cur_sum > 0, it is beneficial asset; when cur_sum < 0, it becomes a liability—cutting it increases future sums.',
          )}</p>
        </div>
        <div className="kadane-scoreboard">
          <div><span>cur_sum:</span> <strong className={step.curSum >= 0 ? 'pos' : 'neg'}>{step.curSum}</strong></div>
          <div><span>max_sum:</span> <strong className="max-tag">{step.maxSum}</strong></div>
        </div>
      </header>

      <div className="kadane-track">
        {nums.map((val, idx) => {
          const isInspected = idx === step.i;
          const isInSubarray = idx >= step.subStart && idx <= step.subEnd;
          const isPeakSubarray = idx >= 3 && idx <= 6 && stepIndex >= 6;
          return (
            <div
              key={idx}
              className={`kadane-cell ${isInspected ? 'inspect' : ''} ${isInSubarray ? 'active-sub' : ''} ${isPeakSubarray ? 'peak' : ''}`}
            >
              <span className="kadane-idx">i={idx}</span>
              <strong className={`kadane-val ${val >= 0 ? 'pos' : 'neg'}`}>{val >= 0 ? `+${val}` : val}</strong>
              {isInspected && <span className="kadane-badge current">{t('当前项', 'Current')}</span>}
              {idx === step.subStart && isInSubarray && <span className="kadane-badge start">{t('起点 L', 'Start L')}</span>}
            </div>
          );
        })}
      </div>

      <div className="kadane-control-bar">
        <label>
          <span>{t('步进步骤', 'Step')}: {stepIndex + 1} / {KADANE_STEPS.length} (Index {step.i})</span>
          <input
            aria-label={t('选择 Kadane 演示步骤', 'Select Kadane step')}
            max={KADANE_STEPS.length - 1}
            min="0"
            onChange={(e) => setStepIndex(Number(e.target.value))}
            type="range"
            value={stepIndex}
          />
        </label>
        <button
          className="kadane-next-btn"
          disabled={stepIndex >= KADANE_STEPS.length - 1}
          onClick={() => setStepIndex((prev) => Math.min(KADANE_STEPS.length - 1, prev + 1))}
          type="button"
        >
          {t('下一步 →', 'Next Step →')}
        </button>
      </div>

      <div className="kadane-desc-card">
        <strong>{t('状态决策', 'Decision')}: </strong>
        <span>{isEnglish ? step.descEn : step.desc}</span>
      </div>
    </section>
  );
}

const JUMP_GAME_CASES = [
  {
    id: 'solvable',
    title: 'nums = [2, 3, 1, 1, 4] (可达终点 · True)',
    titleEn: 'nums = [2, 3, 1, 1, 4] (Reachable · True)',
    nums: [2, 3, 1, 1, 4],
    steps: [
      { i: 0, jump: 2, reach: 2, desc: '初始 i=0：从 0 跳最多 2 步，max_reach = max(0, 0+2) = 2', descEn: 'Start i=0: max_reach = max(0, 0+2) = 2' },
      { i: 1, jump: 3, reach: 4, desc: 'i=1 <= 2 (可达)：从 1 跳最多 3 步，max_reach = max(2, 1+3) = 4 >= 4 (已达终点！)', descEn: 'i=1 <= 2 (reachable): jump 3 reaches max_reach = max(2, 1+3) = 4 >= 4 (Finish!)' },
      { i: 2, jump: 1, reach: 4, desc: 'i=2 <= 4 (可达)：max_reach = max(4, 2+1) = 4', descEn: 'i=2 <= 4: max_reach = max(4, 2+1) = 4' },
      { i: 3, jump: 1, reach: 4, desc: 'i=3 <= 4 (可达)：max_reach = max(4, 3+1) = 4', descEn: 'i=3 <= 4: max_reach = max(4, 3+1) = 4' },
      { i: 4, jump: 4, reach: 8, desc: '到达终点 index 4，返回 True！', descEn: 'Reached final index 4, return True!' },
    ],
  },
  {
    id: 'blocked',
    title: 'nums = [3, 2, 1, 0, 4] (被困 0 处 · False)',
    titleEn: 'nums = [3, 2, 1, 0, 4] (Trapped by 0 · False)',
    nums: [3, 2, 1, 0, 4],
    steps: [
      { i: 0, jump: 3, reach: 3, desc: '初始 i=0：max_reach = max(0, 0+3) = 3', descEn: 'Start i=0: max_reach = max(0, 0+3) = 3' },
      { i: 1, jump: 2, reach: 3, desc: 'i=1 <= 3：max_reach = max(3, 1+2) = 3', descEn: 'i=1 <= 3: max_reach = max(3, 1+2) = 3' },
      { i: 2, jump: 1, reach: 3, desc: 'i=2 <= 3：max_reach = max(3, 2+1) = 3', descEn: 'i=2 <= 3: max_reach = max(3, 2+1) = 3' },
      { i: 3, jump: 0, reach: 3, desc: 'i=3 <= 3：此时 nums[3]=0，max_reach = max(3, 3+0) = 3', descEn: 'i=3 <= 3: nums[3]=0, max_reach = max(3, 3+0) = 3' },
      { i: 4, jump: 4, reach: 3, desc: 'i=4 > max_reach (3)！当前位置不可达，被困死，返回 False！', descEn: 'i=4 > max_reach (3)! Index 4 is unreachable, trapped, return False!' },
    ],
  },
];

function JumpGameVisual() {
  const { isEnglish, t } = useUiCopy();
  const [activeCaseId, setActiveCaseId] = useState('solvable');
  const [stepIndex, setStepIndex] = useState(0);

  const activeCase = JUMP_GAME_CASES.find((c) => c.id === activeCaseId) ?? JUMP_GAME_CASES[0];
  const step = activeCase.steps[stepIndex] ?? activeCase.steps[0];

  return (
    <section aria-label={t('跳跃游戏贪心包络线演示', 'Jump Game greedy envelope walkthrough')} className="jump-vis">
      <header className="jump-header">
        <div>
          <p className="eyebrow">{t('最远可达包络线机制', 'Farthest Reachable Envelope')}</p>
          <h2>{t('Jump Game：维护 max_reach 消除回溯', 'Jump Game: Maintain max_reach without backtracking')}</h2>
          <p>{t(
            '不需要穷举跳 1 步还是 2 步，只需维护历史能覆盖的最远边界 max_reach。',
            'No need to branch over every jump length; simply maintain the farthest envelope max_reach.',
          )}</p>
        </div>
        <div className="jump-case-tabs" role="tablist">
          {JUMP_GAME_CASES.map((c) => (
            <button
              key={c.id}
              className={c.id === activeCaseId ? 'active' : ''}
              onClick={() => { setActiveCaseId(c.id); setStepIndex(0); }}
              role="tab"
              type="button"
            >
              {isEnglish ? c.titleEn : c.title}
            </button>
          ))}
        </div>
      </header>

      <div className="jump-grid">
        {activeCase.nums.map((val, idx) => {
          const isCurrent = idx === step.i;
          const isReachable = idx <= step.reach;
          const isFarthest = idx === Math.min(step.reach, activeCase.nums.length - 1);
          return (
            <div
              key={idx}
              className={`jump-cell ${isCurrent ? 'current' : ''} ${isReachable ? 'reachable' : 'unreachable'} ${isFarthest ? 'farthest' : ''}`}
            >
              <span className="jump-idx">i={idx}</span>
              <strong className="jump-val">{val}</strong>
              <span className="jump-reach">→ {idx + val}</span>
              {isCurrent && <span className="jump-badge current">{t('当前位置 i', 'Current i')}</span>}
              {isFarthest && <span className="jump-badge reach">{t('最远 reach', 'max_reach')}</span>}
            </div>
          );
        })}
      </div>

      <div className="jump-control-bar">
        <label>
          <span>{t('执行步骤', 'Step')}: {stepIndex + 1} / {activeCase.steps.length}</span>
          <input
            aria-label={t('选择跳跃游戏演示步骤', 'Select Jump Game step')}
            max={activeCase.steps.length - 1}
            min="0"
            onChange={(e) => setStepIndex(Number(e.target.value))}
            type="range"
            value={stepIndex}
          />
        </label>
        <button
          className="jump-next-btn"
          disabled={stepIndex >= activeCase.steps.length - 1}
          onClick={() => setStepIndex((prev) => Math.min(activeCase.steps.length - 1, prev + 1))}
          type="button"
        >
          {t('下一步 →', 'Next Step →')}
        </button>
      </div>

      <div className="jump-desc-card">
        <strong>{t('不变量判定', 'Invariant State')}: </strong>
        <span>{isEnglish ? step.descEn : step.desc}</span>
      </div>
    </section>
  );
}

const GAS_STATION_STEPS = [
  {
    i: 0,
    gas: 1,
    cost: 3,
    net: -2,
    tank: -2,
    start: 0,
    nextStart: 1,
    desc: '从起点 0 出发：net = 1 - 3 = -2。cur_tank = -2 < 0 断油！起点 0 废弃，下一候选起点跳至 start = 1，重置 cur_tank = 0。',
    descEn: 'Start at station 0: net = 1 - 3 = -2. cur_tank = -2 < 0 (Deficit)! Station 0 disqualified, next candidate start = 1, reset cur_tank = 0.',
  },
  {
    i: 1,
    gas: 2,
    cost: 4,
    net: -2,
    tank: -2,
    start: 1,
    nextStart: 2,
    desc: '从起点 1 出发：net = 2 - 4 = -2。cur_tank = -2 < 0 断油！起点 1 废弃，下一候选起点跳至 start = 2，重置 cur_tank = 0。',
    descEn: 'Start at station 1: net = 2 - 4 = -2. cur_tank = -2 < 0 (Deficit)! Station 1 disqualified, next candidate start = 2, reset cur_tank = 0.',
  },
  {
    i: 2,
    gas: 3,
    cost: 5,
    net: -2,
    tank: -2,
    start: 2,
    nextStart: 3,
    desc: '从起点 2 出发：net = 3 - 5 = -2。cur_tank = -2 < 0 断油！起点 2 废弃，下一候选起点跳至 start = 3，重置 cur_tank = 0。',
    descEn: 'Start at station 2: net = 3 - 5 = -2. cur_tank = -2 < 0 (Deficit)! Station 2 disqualified, next candidate start = 3, reset cur_tank = 0.',
  },
  {
    i: 3,
    gas: 4,
    cost: 1,
    net: 3,
    tank: 3,
    start: 3,
    nextStart: 3,
    desc: '从起点 3 出发：net = 4 - 1 = +3。cur_tank = +3 >= 0 顺畅通行！继续保留 start = 3。',
    descEn: 'Start at station 3: net = 4 - 1 = +3. cur_tank = +3 >= 0 (Surplus)! Keep start = 3 and advance.',
  },
  {
    i: 4,
    gas: 5,
    cost: 2,
    net: 3,
    tank: 6,
    start: 3,
    nextStart: 3,
    desc: '从起点 3 继续到达 4：net = 5 - 2 = +3。cur_tank = 3 + 3 = 6 >= 0。总净油量 total_surplus = (-2)*3 + 3*2 = 0 >= 0。全局有解，唯一有效起点为 index 3！',
    descEn: 'Continue from station 3 to station 4: net = 5 - 2 = +3. cur_tank = 3 + 3 = 6 >= 0. Total surplus = 0 >= 0. Unique valid start is index 3!',
  },
];

function GasStationVisual() {
  const { isEnglish, t } = useUiCopy();
  const [stepIndex, setStepIndex] = useState(0);
  const step = GAS_STATION_STEPS[stepIndex] ?? GAS_STATION_STEPS[0];

  const stations = [
    { idx: 0, gas: 1, cost: 3, net: -2 },
    { idx: 1, gas: 2, cost: 4, net: -2 },
    { idx: 2, gas: 3, cost: 5, net: -2 },
    { idx: 3, gas: 4, cost: 1, net: 3 },
    { idx: 4, gas: 5, cost: 2, net: 3 },
  ];

  return (
    <section aria-label={t('加油站断点重置演示', 'Gas Station deficit reset walkthrough')} className="gas-vis">
      <header className="gas-header">
        <div>
          <p className="eyebrow">{t('断点跳跃定理', 'Deficit Reset Theorem')}</p>
          <h2>{t('Gas Station：排除负累赘与候选点跳跃', 'Gas Station: Deficit Disqualification and Candidate Jump')}</h2>
          <p>{t(
            '若在 j 处断油，则 [start, j] 内所有加油站都不可能作为起点，直接跳到 j + 1。',
            'If fuel runs out at j, no station in [start, j] can be the start; jump straight to j + 1.',
          )}</p>
        </div>
      </header>

      <div className="gas-track">
        {stations.map((st) => {
          const isInspected = st.idx === step.i;
          const isCandidate = st.idx === step.start;
          const isFailed = st.idx < step.nextStart && step.tank < 0;
          return (
            <div
              key={st.idx}
              className={`gas-station-card ${isInspected ? 'inspect' : ''} ${isCandidate ? 'candidate' : ''} ${isFailed ? 'failed' : ''}`}
            >
              <div className="gas-card-top">
                <span>Station {st.idx}</span>
                {isCandidate && <strong className="cand-tag">{t('候选起点', 'Candidate')}</strong>}
              </div>
              <div className="gas-stats">
                <div><span>Gas:</span> <strong>{st.gas}</strong></div>
                <div><span>Cost:</span> <strong>{st.cost}</strong></div>
                <div><span>Net:</span> <strong className={st.net >= 0 ? 'pos' : 'neg'}>{st.net >= 0 ? `+${st.net}` : st.net}</strong></div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="gas-control-bar">
        <label>
          <span>{t('探索步数', 'Step')}: {stepIndex + 1} / {GAS_STATION_STEPS.length} (Station {step.i})</span>
          <input
            aria-label={t('选择加油站演示步骤', 'Select Gas Station step')}
            max={GAS_STATION_STEPS.length - 1}
            min="0"
            onChange={(e) => setStepIndex(Number(e.target.value))}
            type="range"
            value={stepIndex}
          />
        </label>
        <button
          className="gas-next-btn"
          disabled={stepIndex >= GAS_STATION_STEPS.length - 1}
          onClick={() => setStepIndex((prev) => Math.min(GAS_STATION_STEPS.length - 1, prev + 1))}
          type="button"
        >
          {t('下一步 →', 'Next Step →')}
        </button>
      </div>

      <div className="gas-desc-card">
        <strong>{t('状态快照', 'State Snapshot')}: </strong>
        <span>{isEnglish ? step.descEn : step.desc}</span>
      </div>
    </section>
  );
}

const PARTITION_STEPS = [
  { i: 0, char: 'a', lastIdx: 8, end: 8, cuts: [], desc: 'i=0 (\'a\')：last[\'a\'] = 8 -> 初始 end = 8。当前片段必须至少延伸到 8。', descEn: 'i=0 (\'a\'): last[\'a\'] = 8 -> end = 8. Partition must reach at least index 8.' },
  { i: 4, char: 'c', lastIdx: 7, end: 8, cuts: [], desc: 'i=4 (\'c\')：last[\'c\'] = 7 <= 8 -> end 保持 8。', descEn: 'i=4 (\'c\'): last[\'c\'] = 7 <= 8 -> end remains 8.' },
  { i: 8, char: 'a', lastIdx: 8, end: 8, cuts: [9], desc: 'i=8 (\'a\')：i == end (8)！当前片段内所有字符在后续不再出现，切出第一段长度 9 ("ababcbaca")！下一段从 9 开始。', descEn: 'i=8 (\'a\'): i == end (8)! All chars inside never appear again. Cut Part 1 of length 9 ("ababcbaca")! Start next at 9.' },
  { i: 11, char: 'e', lastIdx: 15, end: 15, cuts: [9], desc: 'i=11 (\'e\')：last[\'e\'] = 15 -> end 扩展到 15。', descEn: 'i=11 (\'e\'): last[\'e\'] = 15 -> end expanded to 15.' },
  { i: 15, char: 'e', lastIdx: 15, end: 15, cuts: [9, 7], desc: 'i=15 (\'e\')：i == end (15)！切出第二段长度 7 ("defegde")！下一段从 16 开始。', descEn: 'i=15 (\'e\'): i == end (15)! Cut Part 2 of length 7 ("defegde")! Start next at 16.' },
  { i: 23, char: 'j', lastIdx: 23, end: 23, cuts: [9, 7, 8], desc: 'i=23 (\'j\')：i == end (23)！切出第三段长度 8 ("hijhklij")！最终答案 [9, 7, 8]。', descEn: 'i=23 (\'j\'): i == end (23)! Cut Part 3 of length 8 ("hijhklij")! Final answer: [9, 7, 8].' },
];

function PartitionLabelsVisual() {
  const { isEnglish, t } = useUiCopy();
  const [stepIndex, setStepIndex] = useState(0);
  const step = PARTITION_STEPS[stepIndex] ?? PARTITION_STEPS[0];

  const rawStr = 'ababcbacadefegdehijhklij';

  return (
    <section aria-label={t('划分字母区间贪心切分演示', 'Partition Labels greedy cut walkthrough')} className="part-vis">
      <header className="part-header">
        <div>
          <p className="eyebrow">{t('最后出现位置包络切割', 'Last-Occurrence Envelope')}</p>
          <h2>{t('Partition Labels：贪心扩展右边界并在 i == end 处切断', 'Partition Labels: Expand boundary and cut when i == end')}</h2>
          <p>{t(
            '预处理每个字符的最后出现位置 last[c]，当 i 追上当前片段的最大 last 时立即切断。',
            'Precompute last[c] for every char; cut immediately when pointer i reaches max last index.',
          )}</p>
        </div>
      </header>

      <div className="part-str-track">
        {rawStr.split('').map((ch, idx) => {
          const isInspected = idx === step.i;
          const isInWindow = idx <= step.end;
          return (
            <div
              key={idx}
              className={`part-char-box ${isInspected ? 'inspect' : ''} ${isInWindow ? 'in-window' : ''}`}
            >
              <span className="part-idx">{idx}</span>
              <strong className="part-char">{ch}</strong>
            </div>
          );
        })}
      </div>

      <div className="part-cuts-summary">
        <span>{t('当前已切割片段', 'Current Cut Partitions')}: </span>
        {step.cuts.length === 0 ? <em>{t('暂未切出完整片段', 'None yet')}</em> : (
          <strong>[{step.cuts.join(', ')}]</strong>
        )}
      </div>

      <div className="part-control-bar">
        <label>
          <span>{t('演示关键帧', 'Keyframe')}: {stepIndex + 1} / {PARTITION_STEPS.length} (Index {step.i})</span>
          <input
            aria-label={t('选择划分字母区间演示步骤', 'Select Partition Labels step')}
            max={PARTITION_STEPS.length - 1}
            min="0"
            onChange={(e) => setStepIndex(Number(e.target.value))}
            type="range"
            value={stepIndex}
          />
        </label>
        <button
          className="part-next-btn"
          disabled={stepIndex >= PARTITION_STEPS.length - 1}
          onClick={() => setStepIndex((prev) => Math.min(PARTITION_STEPS.length - 1, prev + 1))}
          type="button"
        >
          {t('下一步 →', 'Next Step →')}
        </button>
      </div>

      <div className="part-desc-card">
        <strong>{t('切分决策', 'Cut Decision')}: </strong>
        <span>{isEnglish ? step.descEn : step.desc}</span>
      </div>
    </section>
  );
}

function BacktrackingPatternAtlas() {
  const { isEnglish, t } = useUiCopy();
  const [activePattern, setActivePattern] = useState('subsets');
  const basePattern = BACKTRACKING_PATTERNS.find(({ id }) => id === activePattern)
    ?? BACKTRACKING_PATTERNS[0];
  const pattern = isEnglish
    ? { ...basePattern, ...BACKTRACKING_PATTERNS_EN[basePattern.id] }
    : basePattern;

  return (
    <section
      aria-label={t('十道回溯题模板对照', 'Ten backtracking problems compared')}
      className={`bp-atlas ${pattern.tone}`}
    >
      <header className="bp-header">
        <div>
          <p className="eyebrow">{t('同一个骨架，三个槽位', 'One skeleton · three slots')}</p>
          <h2>{t('先认决策模式，再决定传 start 还是开 used', 'Identify the decision pattern first, then choose start or used')}</h2>
          <p>{t(
            '每道题都是 make / backtrack / undo 这三行，差别只在 choices 怎么枚举、答案什么时候收、以及要不要去重。',
            'Every problem is the same make / backtrack / undo trio. Only the enumeration of choices, the moment an answer is collected, and the need for dedup differ.',
          )}</p>
        </div>
        <code>{pattern.signature}</code>
      </header>

      <div aria-label={t('选择回溯题目', 'Choose a backtracking problem')} className="bp-tabs" role="tablist">
        {BACKTRACKING_PATTERNS.map((candidate) => (
          <button
            aria-selected={candidate.id === activePattern}
            className={candidate.id === activePattern ? 'active' : ''}
            key={candidate.id}
            onClick={() => setActivePattern(candidate.id)}
            role="tab"
            type="button"
          >
            <span>LC {candidate.number}</span>
            <strong>{candidate.title}</strong>
          </button>
        ))}
      </div>

      <div className="bp-summary">
        <div>
          <span>{t('决策模式', 'Decision pattern')}</span>
          <strong>{pattern.pattern}</strong>
        </div>
        <div>
          <span>{t('树的规模', 'Tree size')}</span>
          <strong>{pattern.size}</strong>
        </div>
      </div>

      <div className="bp-flow">
        <div>
          <span>{t('1 · choices 枚举什么', '1 · What choices enumerates')}</span>
          <strong>{pattern.choices}</strong>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('2 · 剪枝与去重', '2 · Pruning and dedup')}</span>
          <strong>{pattern.prune}</strong>
          <small>{pattern.guard}</small>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('3 · 递归怎么传参', '3 · What the recursion passes')}</span>
          <strong>{pattern.recurse}</strong>
        </div>
        <b aria-hidden="true">→</b>
        <div>
          <span>{t('4 · 什么时候收答案', '4 · When an answer is collected')}</span>
          <strong>{pattern.collect}</strong>
        </div>
      </div>
    </section>
  );
}

function BacktrackTreeDiagram({ ariaLabel, edgeClass, nodeClass, nodes, viewBox }) {
  const byId = {};
  nodes.forEach((node) => { byId[node.id] = node; });

  return (
    <svg aria-label={ariaLabel} className="bt-tree" role="img" viewBox={viewBox}>
      {nodes.filter((node) => node.id !== '').map((node) => {
        const parentId = node.parentId ?? node.id.slice(0, -1);
        const parent = byId[parentId];
        if (!parent) return null;
        return (
          <line
            className={`bt-edge ${edgeClass(node.id)}`}
            key={`edge-${node.id}`}
            x1={parent.x}
            x2={node.x}
            y1={parent.y + 16}
            y2={node.y - 16}
          />
        );
      })}
      {nodes.map((node) => {
        const width = Math.max(52, node.label.length * 8.5 + 18);
        return (
          <g className={`bt-node ${nodeClass(node.id)}`} key={`node-${node.id}`}>
            <rect height="32" rx="8" width={width} x={node.x - width / 2} y={node.y - 16} />
            <text textAnchor="middle" x={node.x} y={node.y + 5}>{node.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

const BT_SUBSET_NUMS = [1, 2, 3];

const BT_SUBSET_NODES = [
  { id: '', label: '[ ]', x: 366, y: 44 },
  { id: '0', label: '[1]', x: 170, y: 142 },
  { id: '1', label: '[2]', x: 434, y: 142 },
  { id: '2', label: '[3]', x: 634, y: 142 },
  { id: '01', label: '[1,2]', x: 92, y: 240 },
  { id: '02', label: '[1,3]', x: 258, y: 240 },
  { id: '12', label: '[2,3]', x: 434, y: 240 },
  { id: '012', label: '[1,2,3]', x: 92, y: 338 },
];

const BT_SUBSET_CODE_LINES = [
  { id: 'collect', code: ['def backtrack(start):', '    result.append(path[:])'] },
  { id: 'loop', code: ['    for i in range(start, len(nums)):'] },
  { id: 'choose', code: ['        path.append(nums[i])'] },
  { id: 'recurse', code: ['        backtrack(i + 1)'] },
  { id: 'undo', code: ['        path.pop()'] },
];

function buildSubsetTreeSteps() {
  const nums = BT_SUBSET_NUMS;
  const steps = [];
  const path = [];
  const result = [];
  const visited = [];
  const stack = [];

  const snap = (kind, node, extra) => steps.push({
    kind,
    node,
    path: [...path],
    result: result.map((entry) => [...entry]),
    visited: [...visited],
    stack: [...stack],
    ...extra,
  });

  const walk = (start, nodeId) => {
    stack.push({ node: nodeId, start });
    result.push([...path]);
    visited.push(nodeId);
    snap('collect', nodeId, { start });

    for (let i = start; i < nums.length; i += 1) {
      const childId = nodeId + String(i);
      path.push(nums[i]);
      snap('choose', childId, { index: i, value: nums[i], parent: nodeId });
      walk(i + 1, childId);
      path.pop();
      snap('undo', nodeId, { index: i, value: nums[i], child: childId });
    }

    stack.pop();
  };

  snap('start', '', { start: 0 });
  walk(0, '');
  snap('done', '', { start: 0 });
  return steps;
}

const BT_SUBSET_STEPS = buildSubsetTreeSteps();

function formatList(values) {
  return `[${values.join(', ')}]`;
}

function BacktrackingTreeVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = BT_SUBSET_STEPS;
  const step = steps[activeStep];
  const visited = new Set(step.visited);
  const collected = new Set(step.visited);
  const onPath = new Set();
  for (let i = 0; i <= step.node.length; i += 1) {
    onPath.add(step.node.slice(0, i));
  }

  const activeLine = {
    start: 'collect',
    collect: 'collect',
    choose: 'choose',
    undo: 'undo',
    done: 'loop',
  }[step.kind];

  const lineLabel = {
    collect: t('每个节点一进来就收一次答案', 'Every node collects an answer on entry'),
    choose: t('path.append：进入子树前做选择', 'path.append: choose before entering the subtree'),
    undo: t('path.pop：离开子树后撤销选择', 'path.pop: un-choose after leaving the subtree'),
    loop: t('循环走完，函数返回', 'The loop finished and the function returns'),
  }[activeLine];

  let title;
  let detail;
  if (step.kind === 'start') {
    title = t('从根节点出发，path 是空的', 'Start at the root with an empty path');
    detail = t(
      '根节点代表"什么都没选"。子集型的答案不是叶子，而是树上的每一个节点，所以空集也是一个答案。',
      'The root means "nothing chosen yet." In a subset problem the answers are not the leaves but every node in the tree, so the empty set is an answer too.',
    );
  } else if (step.kind === 'collect') {
    title = t(
      `收答案：result 现在有 ${step.result.length} 项`,
      `Collect: result now holds ${step.result.length} entries`,
    );
    detail = t(
      `result.append(path[:]) 拷贝了当前的 ${formatList(step.path)}。写成 result.append(path) 会存进引用，后面 pop 掉之后这一项会跟着变空。`,
      `result.append(path[:]) copies the current ${formatList(step.path)}. Writing result.append(path) would store a reference, and later pops would empty this entry.`,
    );
  } else if (step.kind === 'choose') {
    title = t(
      `选择 nums[${step.index}] = ${step.value}，进入子树`,
      `Choose nums[${step.index}] = ${step.value} and enter the subtree`,
    );
    detail = t(
      `path 变成 ${formatList(step.path)}，递归调用 backtrack(${step.index + 1})。传 i + 1 而不是 start + 1，意思是"接着我刚选的这个元素往后挑，不回头"。`,
      `path becomes ${formatList(step.path)} and backtrack(${step.index + 1}) is called. Passing i + 1 rather than start + 1 means "keep picking after the element I just chose, never turning back."`,
    );
  } else if (step.kind === 'undo') {
    title = t(
      `子树走完，撤销 ${step.value}`,
      `Subtree finished, undo ${step.value}`,
    );
    detail = t(
      `path.pop() 把 path 改回 ${formatList(step.path)}。不撤销的话，下一个兄弟分支会看到上一个分支留下的元素，答案直接错。`,
      `path.pop() restores path to ${formatList(step.path)}. Without the undo the next sibling branch would inherit the previous branch's leftovers and the answers would be wrong.`,
    );
  } else {
    title = t('搜索结束，8 个子集全部收齐', 'Search complete: all 8 subsets collected');
    detail = t(
      'path 回到空、递归栈清空，正好说明每一次 append 都配上了一次 pop。',
      'path is empty again and the recursion stack is clear, which is exactly the evidence that every append was matched by a pop.',
    );
  }

  return (
    <section aria-label={t('Subsets 决策树逐步演示', 'Step-through: the Subsets decision tree')} className="bt-visual">
      <header className="bt-header">
        <div>
          <p className="eyebrow">{t('LC 78 · Subsets，nums = [1, 2, 3]', 'LC 78 · Subsets with nums = [1, 2, 3]')}</p>
          <h2>{t('choose / recurse / undo 在树上是什么动作', 'What choose / recurse / undo look like on the tree')}</h2>
          <p>{t(
            '看三件事：path 怎么随深度变化、答案在哪一刻被收走、以及返回父节点时 pop 撤销了什么。',
            'Watch three things: how path changes with depth, the moment each answer is collected, and what pop undoes when control returns to the parent.',
          )}</p>
        </div>
      </header>

      <div aria-live="polite" className={`bt-step ${step.kind}`}>
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="bt-workspace">
        <div className="bt-stage-card">
          <BacktrackTreeDiagram
            ariaLabel={t('Subsets 决策树', 'The Subsets decision tree')}
            edgeClass={(id) => (onPath.has(id) ? 'active' : visited.has(id) ? 'done' : 'ghost')}
            nodeClass={(id) => [
              id === step.node ? 'current' : '',
              id !== step.node && onPath.has(id) ? 'active' : '',
              !onPath.has(id) && collected.has(id) ? 'done' : '',
              !visited.has(id) ? 'ghost' : '',
            ].filter(Boolean).join(' ')}
            nodes={BT_SUBSET_NODES}
            viewBox="0 0 720 380"
          />
        </div>

        <div className="bt-side">
          <div aria-label={t('当前代码', 'Current code')} className="bt-code">
            <div className="bt-code-heading">
              <span>subsets</span>
              <strong>{lineLabel}</strong>
            </div>
            <div className="bt-code-lines">
              {BT_SUBSET_CODE_LINES.map((line) => (
                <div
                  aria-current={activeLine === line.id ? 'step' : undefined}
                  className={activeLine === line.id ? 'active' : ''}
                  key={line.id}
                >
                  {line.code.map((code) => <code key={code}>{code}</code>)}
                </div>
              ))}
            </div>
          </div>

          <div className="bt-state">
            <div>
              <span>path</span>
              <strong>{formatList(step.path)}</strong>
            </div>
            <div>
              <span>{t('递归栈深度', 'Stack depth')}</span>
              <strong>{step.stack.length}</strong>
            </div>
          </div>

          <div className="bt-results">
            <span>result（{step.result.length}）</span>
            <div>
              {step.result.map((entry, index) => (
                <code
                  className={index === step.result.length - 1 && step.kind === 'collect' ? 'fresh' : ''}
                  key={`${entry.join('-')}-${index}`}
                >
                  {formatList(entry)}
                </code>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bt-legend">
        <span><i className="current" />{t('当前节点', 'Current node')}</span>
        <span><i className="active" />{t('当前 path 经过的节点', 'Nodes on the current path')}</span>
        <span><i className="done" />{t('已经收过答案', 'Already collected')}</span>
        <span><i className="ghost" />{t('还没访问到', 'Not visited yet')}</span>
      </div>

      <div className="bt-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择决策树演示步骤', 'Select a decision-tree demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const BT_DEDUP_NUMS = [1, 2, 2];

const BT_DEDUP_NODES = [
  { id: '', label: '[ ]', x: 380, y: 44 },
  { id: '0', label: '[1]', x: 176, y: 142 },
  { id: '1', label: '[2]', x: 452, y: 142 },
  { id: '2', label: '[2]', x: 654, y: 142 },
  { id: '01', label: '[1,2]', x: 96, y: 242 },
  { id: '02', label: '[1,2]', x: 264, y: 242 },
  { id: '12', label: '[2,2]', x: 452, y: 242 },
  { id: '012', label: '[1,2,2]', x: 96, y: 340 },
];

const BT_DEDUP_CODE_LINES = [
  { id: 'collect', code: ['nums.sort()', '', 'def backtrack(start):', '    result.append(path[:])'] },
  { id: 'loop', code: ['    for i in range(start, len(nums)):'] },
  { id: 'guard', code: ['        if i > start and nums[i] == nums[i - 1]:', '            continue'] },
  { id: 'choose', code: ['        path.append(nums[i])'] },
  { id: 'recurse', code: ['        backtrack(i + 1)'] },
  { id: 'undo', code: ['        path.pop()'] },
];

function buildDedupTreeSteps() {
  const nums = BT_DEDUP_NUMS;
  const steps = [];
  const path = [];
  const result = [];
  const visited = [];
  const skipped = [];
  let currentStart = 0;

  const snap = (kind, node, extra) => steps.push({
    kind,
    node,
    start: currentStart,
    path: [...path],
    result: result.map((entry) => [...entry]),
    visited: [...visited],
    skipped: [...skipped],
    ...extra,
  });

  const walk = (start, nodeId) => {
    currentStart = start;
    result.push([...path]);
    visited.push(nodeId);
    snap('collect', nodeId);

    for (let i = start; i < nums.length; i += 1) {
      const childId = nodeId + String(i);
      if (i > start && nums[i] === nums[i - 1]) {
        skipped.push(childId);
        snap('skip', nodeId, { index: i, value: nums[i], child: childId });
        continue;
      }
      path.push(nums[i]);
      snap('choose', childId, { index: i, value: nums[i] });
      walk(i + 1, childId);
      path.pop();
      currentStart = start;
      snap('undo', nodeId, { index: i, value: nums[i], child: childId });
    }
  };

  snap('start', '');
  walk(0, '');
  currentStart = 0;
  snap('done', '');
  return steps;
}

const BT_DEDUP_STEPS = buildDedupTreeSteps();

function BacktrackingDedupVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = BT_DEDUP_STEPS;
  const step = steps[activeStep];
  const visited = new Set(step.visited);
  const skipped = new Set(step.skipped);
  const onPath = new Set();
  for (let i = 0; i <= step.node.length; i += 1) {
    onPath.add(step.node.slice(0, i));
  }

  const activeLine = {
    start: 'collect',
    collect: 'collect',
    skip: 'guard',
    choose: 'choose',
    undo: 'undo',
    done: 'loop',
  }[step.kind];

  const lineLabel = {
    collect: t('子集型每个节点都收答案', 'A subset problem collects at every node'),
    guard: t('同层重复值，跳过整棵子树', 'A same-level duplicate: skip the whole subtree'),
    choose: t('这一层第一次选这个值，保留', 'First pick of this value at this level: keep it'),
    undo: t('撤销选择，回到父节点', 'Un-choose and return to the parent'),
    loop: t('循环走完，函数返回', 'The loop finished and the function returns'),
  }[activeLine];

  let title;
  let detail;
  if (step.kind === 'start') {
    title = t('nums 排序后是 [1, 2, 2]', 'After sorting, nums is [1, 2, 2]');
    detail = t(
      '排序不是为了让答案有序，而是让相同的值相邻。否则 nums[i] == nums[i-1] 这个判断根本抓不到重复。',
      'Sorting is not about ordering the output; it makes equal values adjacent. Without it the nums[i] == nums[i-1] check catches nothing.',
    );
  } else if (step.kind === 'collect') {
    title = t(
      `收答案 ${formatList(step.path)}，result 有 ${step.result.length} 项`,
      `Collect ${formatList(step.path)}; result holds ${step.result.length} entries`,
    );
    detail = t(
      '子集型的答案是每一个节点，所以进入节点就收，不需要 base case。',
      'In a subset problem every node is an answer, so collection happens on entry and no base case is needed.',
    );
  } else if (step.kind === 'skip') {
    title = t(
      `i = ${step.index} > start = ${step.start}，且 nums[${step.index}] == nums[${step.index - 1}]，剪掉`,
      `i = ${step.index} > start = ${step.start} and nums[${step.index}] == nums[${step.index - 1}], so prune`,
    );
    detail = t(
      `这是这一层第二次遇到值 ${step.value}。它展开出来的子树和前一个分支完全一样，留着就会产生重复答案，所以整棵子树都不进。`,
      `This is the second time value ${step.value} appears at this level. The subtree it would expand is identical to the previous branch, so keeping it would duplicate answers and the whole subtree is skipped.`,
    );
  } else if (step.kind === 'choose') {
    title = step.index === step.start
      ? t(
        `i = ${step.index} 等于 start，是这一层第一次选 ${step.value}，保留`,
        `i = ${step.index} equals start: the first pick of ${step.value} at this level, so keep it`,
      )
      : t(
        `选择 nums[${step.index}] = ${step.value}`,
        `Choose nums[${step.index}] = ${step.value}`,
      );
    detail = step.index === step.start
      ? t(
        '注意这里的值和上一个分支相同，但 i == start 说明它是这一层的第一个候选，代表"父节点第一次选这个值"，必须保留。把条件写成 i > 0 就会把它一起剪掉。',
        'The value matches the previous branch, but i == start marks it as this level\'s first candidate, meaning "the parent picks this value for the first time," so it must stay. Writing the guard as i > 0 would cut it too.',
      )
      : t(
        `path 变成 ${formatList(step.path)}，递归调用 backtrack(${step.index + 1})。`,
        `path becomes ${formatList(step.path)} and backtrack(${step.index + 1}) is called.`,
      );
  } else if (step.kind === 'undo') {
    title = t(`撤销 ${step.value}，回到 ${formatList(step.path)}`, `Undo ${step.value}, back to ${formatList(step.path)}`);
    detail = t(
      '和不带去重的版本完全一样：append 和 pop 严格成对。',
      'Identical to the version without dedup: every append is strictly matched by a pop.',
    );
  } else {
    title = t('结束：6 个不重复子集，剪掉 2 棵重复子树', 'Done: 6 distinct subsets, 2 duplicate subtrees pruned');
    detail = t(
      '不加那一行 continue 会得到 8 个结果，其中 [2] 和 [2,2] 各出现两次。剪掉的正好是这两棵子树。',
      'Without that continue the search returns 8 results, with [2] and [2,2] appearing twice each. The two pruned subtrees are exactly those duplicates.',
    );
  }

  return (
    <section aria-label={t('Subsets II 同层去重逐步演示', 'Step-through: same-level dedup in Subsets II')} className="bd-visual">
      <header className="bd-header">
        <div>
          <p className="eyebrow">{t('LC 90 · Subsets II，nums = [1, 2, 2]', 'LC 90 · Subsets II with nums = [1, 2, 2]')}</p>
          <h2>{t('i > start 到底剪掉了哪两棵子树', 'Which two subtrees i > start actually removes')}</h2>
          <p>{t(
            '同一层的两个相同值会展开出完全一样的子树，必须剪；不同层的两个相同值是"选了两个 2"，是合法答案。',
            'Two equal values at the same level expand identical subtrees and must be cut. Two equal values at different levels mean "two 2s were picked" and form a valid answer.',
          )}</p>
        </div>
      </header>

      <div aria-live="polite" className={`bd-step ${step.kind}`}>
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="bd-workspace">
        <div className="bd-stage-card">
          <BacktrackTreeDiagram
            ariaLabel={t('Subsets II 决策树', 'The Subsets II decision tree')}
            edgeClass={(id) => (
              skipped.has(id) ? 'cut' : onPath.has(id) ? 'active' : visited.has(id) ? 'done' : 'ghost'
            )}
            nodeClass={(id) => [
              skipped.has(id) ? 'cut' : '',
              !skipped.has(id) && id === step.node ? 'current' : '',
              !skipped.has(id) && id !== step.node && onPath.has(id) ? 'active' : '',
              !skipped.has(id) && !onPath.has(id) && visited.has(id) ? 'done' : '',
              !skipped.has(id) && !visited.has(id) ? 'ghost' : '',
            ].filter(Boolean).join(' ')}
            nodes={BT_DEDUP_NODES}
            viewBox="0 0 740 382"
          />
        </div>

        <div className="bd-side">
          <div aria-label={t('当前代码', 'Current code')} className="bd-code">
            <div className="bd-code-heading">
              <span>subsetsWithDup</span>
              <strong>{lineLabel}</strong>
            </div>
            <div className="bd-code-lines">
              {BT_DEDUP_CODE_LINES.map((line) => (
                <div
                  aria-current={activeLine === line.id ? 'step' : undefined}
                  className={activeLine === line.id ? 'active' : ''}
                  key={line.id}
                >
                  {line.code.map((code, index) => <code key={`${line.id}-${index}`}>{code || ' '}</code>)}
                </div>
              ))}
            </div>
          </div>

          <div className="bd-state">
            <div>
              <span>path</span>
              <strong>{formatList(step.path)}</strong>
            </div>
            <div>
              <span>start</span>
              <strong>{step.start ?? '—'}</strong>
            </div>
            <div>
              <span>{t('已剪子树', 'Subtrees cut')}</span>
              <strong>{step.skipped.length}</strong>
            </div>
          </div>

          <div className="bd-results">
            <span>result（{step.result.length}）</span>
            <div>
              {step.result.map((entry, index) => (
                <code
                  className={index === step.result.length - 1 && step.kind === 'collect' ? 'fresh' : ''}
                  key={`${entry.join('-')}-${index}`}
                >
                  {formatList(entry)}
                </code>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bd-legend">
        <span><i className="current" />{t('当前节点', 'Current node')}</span>
        <span><i className="active" />{t('当前 path', 'Current path')}</span>
        <span><i className="done" />{t('已访问', 'Visited')}</span>
        <span><i className="cut" />{t('同层重复，已剪', 'Same-level duplicate, pruned')}</span>
      </div>

      <div className="bd-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择去重演示步骤', 'Select a dedup demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const NQUEENS_N = 4;

const NQUEENS_CODE_LINES = [
  { id: 'base', code: ['def backtrack(row):', '    if row == n:', '        result.append(snapshot())', '        return'] },
  { id: 'check', code: ['    for col in range(n):', '        if (col in cols or (row - col) in diag', '                or (row + col) in anti):', '            continue'] },
  { id: 'make', code: ['        cols.add(col)', '        diag.add(row - col)', '        anti.add(row + col)'] },
  { id: 'recurse', code: ['        backtrack(row + 1)'] },
  { id: 'undo', code: ['        cols.remove(col)', '        diag.remove(row - col)', '        anti.remove(row + col)'] },
];

function buildNQueensSteps() {
  const n = NQUEENS_N;
  const cols = new Set();
  const diag = new Set();
  const anti = new Set();
  const queens = [];
  const solutions = [];
  const steps = [];

  const sorted = (set) => [...set].sort((a, b) => a - b);
  const snap = (kind, extra) => steps.push({
    kind,
    queens: [...queens],
    cols: sorted(cols),
    diag: sorted(diag),
    anti: sorted(anti),
    solutions: solutions.map((entry) => [...entry]),
    ...extra,
  });

  const backtrack = (row) => {
    if (row === n) {
      solutions.push([...queens]);
      snap('solution', { row });
      return;
    }

    let rejected = [];
    const rowRejects = [];
    let tried = 0;
    for (let col = 0; col < n; col += 1) {
      let reason = null;
      if (cols.has(col)) reason = 'col';
      else if (diag.has(row - col)) reason = 'diag';
      else if (anti.has(row + col)) reason = 'anti';

      if (reason) {
        rejected.push({ col, reason });
        rowRejects.push({ col, reason });
        continue;
      }

      if (rejected.length > 0) {
        snap('reject', { row, rejected });
        rejected = [];
      }

      tried += 1;

      cols.add(col);
      diag.add(row - col);
      anti.add(row + col);
      queens.push(col);
      snap('place', { row, col });

      backtrack(row + 1);

      queens.pop();
      cols.delete(col);
      diag.delete(row - col);
      anti.delete(row + col);
      snap('remove', { row, col });
    }

    if (rejected.length > 0) {
      snap('reject', { row, rejected });
    }
    snap('exhausted', { row, rejected: rowRejects, tried });
  };

  snap('start', { row: 0 });
  backtrack(0);
  snap('done', { row: 0 });
  return steps;
}

const NQUEENS_STEPS = buildNQueensSteps();

function NQueensVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = NQUEENS_STEPS;
  const step = steps[activeStep];
  const n = NQUEENS_N;
  const colSet = new Set(step.cols);
  const diagSet = new Set(step.diag);
  const antiSet = new Set(step.anti);
  const rejectedCols = new Map((step.rejected ?? []).map(({ col, reason }) => [col, reason]));

  const reasonLabel = {
    col: t('列冲突', 'column conflict'),
    diag: t('主对角线冲突', 'main-diagonal conflict'),
    anti: t('副对角线冲突', 'anti-diagonal conflict'),
  };

  const reasonMark = { col: '│', diag: '╲', anti: '╱' };

  const activeLine = {
    start: 'base',
    reject: 'check',
    place: 'make',
    remove: 'undo',
    exhausted: 'check',
    solution: 'base',
    done: 'base',
  }[step.kind];

  const lineLabel = {
    base: t('row == n：整张棋盘填满，收答案', 'row == n: the board is full, collect the answer'),
    check: t('三个集合各查一次，O(1) 判冲突', 'One lookup in each of the three sets: an O(1) conflict check'),
    make: t('放下皇后，同时把三个坐标加进集合', 'Place the queen and add all three coordinates to the sets'),
    undo: t('拿走皇后，三个集合各删一项', 'Remove the queen and delete one entry from each set'),
  }[activeLine];

  let title;
  let detail;
  if (step.kind === 'start') {
    title = t('空棋盘，三个集合都是空的', 'An empty board with all three sets empty');
    detail = t(
      '按行放置，第 row 层负责第 row 行，行冲突天然不可能发生，只剩列、主对角线、副对角线要查。',
      'Queens go row by row, so level row handles row row. Row conflicts are impossible by construction, leaving only column, main diagonal, and anti-diagonal to check.',
    );
  } else if (step.kind === 'reject') {
    const parts = step.rejected.map(({ col, reason }) => `col ${col}（${reasonLabel[reason]}）`);
    const partsEn = step.rejected.map(({ col, reason }) => `col ${col} (${reasonLabel[reason]})`);
    title = t(`第 ${step.row} 行：跳过 ${step.rejected.length} 个列`, `Row ${step.row}: skipping ${step.rejected.length} column(s)`);
    detail = t(
      `${parts.join('，')}。这些列被已放置的皇后控制着，continue 掉就等于剪掉了它们下面的整棵子树。`,
      `${partsEn.join(', ')}. These columns are controlled by queens already placed, so the continue prunes every subtree beneath them.`,
    );
  } else if (step.kind === 'place') {
    title = t(
      `第 ${step.row} 行放在 col ${step.col}`,
      `Row ${step.row}: place at col ${step.col}`,
    );
    detail = t(
      `make 一次做三件事：cols 加 ${step.col}，diag 加 row - col = ${step.row - step.col}，anti 加 row + col = ${step.row + step.col}。同一条对角线上的格子，这两个差值和和值分别相同，所以集合查一次就够。`,
      `One make does three things: add ${step.col} to cols, add row - col = ${step.row - step.col} to diag, and add row + col = ${step.row + step.col} to anti. Cells on one diagonal share that difference or that sum, so a single set lookup suffices.`,
    );
  } else if (step.kind === 'remove') {
    title = t(
      `撤销第 ${step.row} 行的 col ${step.col}`,
      `Undo col ${step.col} in row ${step.row}`,
    );
    detail = t(
      '这棵子树已经走完，三个集合各删掉一项，棋盘恢复到放这个皇后之前的样子，然后继续试这一行的下一列。',
      'That subtree is finished, so one entry is deleted from each of the three sets, the board returns to its state before this queen, and the next column of this row gets tried.',
    );
  } else if (step.kind === 'exhausted') {
    title = step.tried === 0
      ? t(`第 ${step.row} 行一个合法列都没有，直接回溯`, `Row ${step.row} has no legal column at all, so backtrack`)
      : t(
        `第 ${step.row} 行的 ${step.tried} 个合法列都试过了，全部失败，回溯`,
        `All ${step.tried} legal column(s) in row ${step.row} were tried and failed, so backtrack`,
      );
    detail = step.tried === 0
      ? t(
        '这一行每一列都被前面的皇后控制住了。for 循环一次都没进到 make，函数直接返回，控制权回到上一行。',
        'Every column in this row is controlled by an earlier queen. The loop never reaches make, the function returns, and control goes back to the previous row.',
      )
      : t(
        '标记出来的列一开始就被控制住，能放的列往下走也没走通。for 循环结束，函数返回，上一行继续试它的下一个列。',
        'The marked columns were controlled from the start, and the columns that could be filled led nowhere deeper. The loop ends, the function returns, and the previous row tries its next column.',
      );
  } else if (step.kind === 'solution') {
    title = t(
      `找到第 ${step.solutions.length} 个解：${formatList(step.queens)}`,
      `Solution ${step.solutions.length} found: ${formatList(step.queens)}`,
    );
    detail = t(
      '数组里第 row 项是第 row 行皇后所在的列。收完答案马上 return，回溯继续找下一个解。',
      'Entry row of the array is the column of the queen in row row. The answer is collected, the function returns immediately, and backtracking continues looking for the next one.',
    );
  } else {
    title = t('搜索结束，n = 4 一共 2 个解', 'Search complete: n = 4 has exactly 2 solutions');
    detail = t(
      '第一层的 4 个分支里，只有 col = 1 和 col = 2 能走到底。三个集合此刻都空了，说明每次 add 都配上了 remove。',
      'Of the four first-level branches, only col = 1 and col = 2 reach the bottom. All three sets are empty again, which shows every add was matched by a remove.',
    );
  }

  return (
    <section aria-label={t('4 皇后回溯逐步演示', 'Step-through: backtracking on 4-Queens')} className="nq-visual">
      <header className="nq-header">
        <div>
          <p className="eyebrow">{t('LC 51 · N-Queens，n = 4', 'LC 51 · N-Queens with n = 4')}</p>
          <h2>{t('三个集合就是这道题的 state', 'The three sets are this problem\'s entire state')}</h2>
          <p>{t(
            'cols / diag / anti 分别记录被占用的列、主对角线和副对角线。make 是三个 add，undo 是三个 remove。',
            'cols / diag / anti record the occupied columns, main diagonals, and anti-diagonals. make is three adds and undo is three removes.',
          )}</p>
        </div>
      </header>

      <div aria-live="polite" className={`nq-step ${step.kind}`}>
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="nq-workspace">
        <div className="nq-stage-card">
          <div className="nq-board" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
            {Array.from({ length: n * n }, (unused, cell) => {
              const row = Math.floor(cell / n);
              const col = cell % n;
              const hasQueen = step.queens[row] === col;
              const attacked = !hasQueen
                && (colSet.has(col) || diagSet.has(row - col) || antiSet.has(row + col));
              const isCursorRow = step.row === row && (step.kind === 'reject' || step.kind === 'exhausted' || step.kind === 'place' || step.kind === 'remove');
              const rejectedHere = isCursorRow && rejectedCols.has(col);
              const focused = (step.kind === 'place' || step.kind === 'remove') && step.row === row && step.col === col;
              return (
                <div
                  className={[
                    'nq-cell',
                    (row + col) % 2 === 0 ? 'light' : 'dark',
                    hasQueen ? 'queen' : '',
                    attacked ? 'attacked' : '',
                    isCursorRow ? 'cursor-row' : '',
                    rejectedHere ? 'rejected' : '',
                    focused ? 'focused' : '',
                  ].filter(Boolean).join(' ')}
                  key={cell}
                >
                  {hasQueen
                    ? <b>Q</b>
                    : rejectedHere
                      ? <s title={reasonLabel[rejectedCols.get(col)]}>{reasonMark[rejectedCols.get(col)]}</s>
                      : null}
                </div>
              );
            })}
          </div>
          <div className="nq-sets">
            <div>
              <span>cols</span>
              <strong>{`{${step.cols.join(', ')}}`}</strong>
            </div>
            <div>
              <span>diag (row − col)</span>
              <strong>{`{${step.diag.join(', ')}}`}</strong>
            </div>
            <div>
              <span>anti (row + col)</span>
              <strong>{`{${step.anti.join(', ')}}`}</strong>
            </div>
          </div>
        </div>

        <div className="nq-side">
          <div aria-label={t('当前代码', 'Current code')} className="nq-code">
            <div className="nq-code-heading">
              <span>solveNQueens</span>
              <strong>{lineLabel}</strong>
            </div>
            <div className="nq-code-lines">
              {NQUEENS_CODE_LINES.map((line) => (
                <div
                  aria-current={activeLine === line.id ? 'step' : undefined}
                  className={activeLine === line.id ? 'active' : ''}
                  key={line.id}
                >
                  {line.code.map((code, index) => <code key={`${line.id}-${index}`}>{code}</code>)}
                </div>
              ))}
            </div>
          </div>

          <div className="nq-solutions">
            <span>{t(`已找到的解（${step.solutions.length}）`, `Solutions found (${step.solutions.length})`)}</span>
            <div>
              {step.solutions.length === 0
                ? <em>{t('还没有', 'none yet')}</em>
                : step.solutions.map((entry, index) => (
                  <code key={`${entry.join('-')}-${index}`}>{formatList(entry)}</code>
                ))}
            </div>
          </div>
        </div>
      </div>

      <div className="nq-legend">
        <span><i className="queen" />{t('皇后', 'Queen')}</span>
        <span><i className="attacked" />{t('被现有皇后控制', 'Controlled by a placed queen')}</span>
        <span><i className="rejected" />{t('本行被 continue 跳过的列（│ 列 ╲ 主对角线 ╱ 副对角线）', 'Column skipped by continue in this row (│ column, ╲ main diagonal, ╱ anti-diagonal)')}</span>
        <span><i className="free" />{t('可放置', 'Free')}</span>
      </div>

      <div className="nq-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择 N 皇后演示步骤', 'Select an N-Queens demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const PM_NUMS = [1, 2, 3];

const PM_NODES = [
  { id: '', label: '[ ]', x: 390, y: 36 },
  { id: '0', label: '[1]', x: 135, y: 118 },
  { id: '1', label: '[2]', x: 390, y: 118 },
  { id: '2', label: '[3]', x: 645, y: 118 },
  { id: '01', label: '[1,2]', x: 80, y: 200 },
  { id: '02', label: '[1,3]', x: 190, y: 200 },
  { id: '10', label: '[2,1]', x: 335, y: 200 },
  { id: '12', label: '[2,3]', x: 445, y: 200 },
  { id: '20', label: '[3,1]', x: 590, y: 200 },
  { id: '21', label: '[3,2]', x: 700, y: 200 },
  { id: '012', label: '[1,2,3]', x: 80, y: 290 },
  { id: '021', label: '[1,3,2]', x: 190, y: 290 },
  { id: '102', label: '[2,1,3]', x: 335, y: 290 },
  { id: '120', label: '[2,3,1]', x: 445, y: 290 },
  { id: '201', label: '[3,1,2]', x: 590, y: 290 },
  { id: '210', label: '[3,2,1]', x: 700, y: 290 },
];

const PM_CODE_LINES = [
  { id: 'base', code: ['def backtrack():', '    if len(path) == len(nums):', '        result.append(path[:])', '        return'] },
  { id: 'loop', code: ['    for i in range(len(nums)):'] },
  { id: 'guard', code: ['        if used[i]:', '            continue'] },
  { id: 'choose', code: ['        used[i] = True', '        path.append(nums[i])'] },
  { id: 'recurse', code: ['        backtrack()'] },
  { id: 'undo', code: ['        path.pop()', '        used[i] = False'] },
];

function buildPermutationsSteps() {
  const nums = PM_NUMS;
  const steps = [];
  const path = [];
  const used = [false, false, false];
  const result = [];
  const visited = [];

  const snap = (kind, node, extra) => steps.push({
    kind,
    node,
    path: [...path],
    used: [...used],
    result: result.map((entry) => [...entry]),
    visited: [...visited],
    ...extra,
  });

  const walk = (nodeId) => {
    visited.push(nodeId);
    if (path.length === nums.length) {
      result.push([...path]);
      snap('collect', nodeId);
      return;
    }

    for (let i = 0; i < nums.length; i += 1) {
      if (used[i]) {
        snap('skip', nodeId, { index: i, value: nums[i] });
        continue;
      }

      const childId = nodeId + String(i);
      used[i] = true;
      path.push(nums[i]);
      snap('choose', childId, { index: i, value: nums[i] });

      walk(childId);

      path.pop();
      used[i] = false;
      snap('undo', nodeId, { index: i, value: nums[i], child: childId });
    }
  };

  snap('start', '');
  walk('');
  snap('done', '');
  return steps;
}

const PM_STEPS = buildPermutationsSteps();

function PermutationsVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = PM_STEPS;
  const step = steps[activeStep];
  const visited = new Set(step.visited);
  const collected = new Set(step.result.map((r) => r.map((val) => PM_NUMS.indexOf(val)).join('')));
  const onPath = new Set();
  for (let i = 0; i <= step.node.length; i += 1) {
    onPath.add(step.node.slice(0, i));
  }

  const activeLine = {
    start: 'base',
    collect: 'base',
    skip: 'guard',
    choose: 'choose',
    undo: 'undo',
    done: 'loop',
  }[step.kind];

  const lineLabel = {
    base: t('len(path) == len(nums) 触发叶子收集', 'len(path) == len(nums) triggers leaf collection'),
    guard: t('used[i] 为 True，跳过已占用元素', 'used[i] is True, skip occupied element'),
    choose: t('标记 used[i] 并加入 path', 'Mark used[i] and append to path'),
    undo: t('回溯：pop 并释放 used[i]', 'Backtrack: pop and release used[i]'),
    loop: t('遍历完成，函数返回', 'Loop finished, function returns'),
  }[activeLine];

  let title;
  let detail;
  if (step.kind === 'start') {
    title = t('从根节点出发，used 全为 False', 'Start at root with used array all False');
    detail = t(
      '全排列问题顺序重要，每一层都从 0 遍历到 n-1，通过 used 数组判断哪些元素已被放入当前排列中。',
      'In permutations order matters. Every level iterates 0 to n-1, using the used array to check which elements are already in the current path.',
    );
  } else if (step.kind === 'choose') {
    title = t(
      `选中 nums[${step.index}] = ${step.value}，used[${step.index}] 设为 True`,
      `Choose nums[${step.index}] = ${step.value}, set used[${step.index}] = True`,
    );
    detail = t(
      `path 变为 ${formatList(step.path)}，进入下一层递归。used[${step.index}] 锁定该元素，防止更深层再次选中它。`,
      `path becomes ${formatList(step.path)}, entering next recursion level. used[${step.index}] locks this element so deeper levels do not pick it again.`,
    );
  } else if (step.kind === 'skip') {
    title = t(
      `nums[${step.index}] = ${step.value} 已在 used 中（True），跳过`,
      `nums[${step.index}] = ${step.value} already used (True), skip`,
    );
    detail = t(
      '排列型每一层从 i = 0 扫起，遇到已在当前路径使用的元素直接 continue。',
      'Permutation loops start from i = 0 at every level; used elements in current path are skipped with continue.',
    );
  } else if (step.kind === 'collect') {
    title = t(
      `叶子节点！收集全排列 ${formatList(step.path)}（第 ${step.result.length}/6 个）`,
      `Leaf node! Collected permutation ${formatList(step.path)} (${step.result.length}/6)`,
    );
    detail = t(
      'len(path) == len(nums) 触发 base case，result.append(path[:]) 拷贝当前排列并立即 return。',
      'len(path) == len(nums) triggers the base case. result.append(path[:]) copies the permutation and returns immediately.',
    );
  } else if (step.kind === 'undo') {
    title = t(
      `回溯：pop 掉 ${step.value}，释放 used[${step.index}] = False`,
      `Backtrack: pop ${step.value} and reset used[${step.index}] = False`,
    );
    detail = t(
      `状态恢复为 path = ${formatList(step.path)}。同层的下一个 i 可以继续尝试其他可用元素。`,
      `State restored to path = ${formatList(step.path)}. The next i at this level can now try other available elements.`,
    );
  } else {
    title = t('搜索完成：共找到 3! = 6 个全排列', 'Search complete: found 3! = 6 permutations');
    detail = t(
      '所有 used 状态均已还原为 False，递归栈完全清空。',
      'All used states are reset to False and the recursion stack is empty.',
    );
  }

  return (
    <section aria-label={t('Permutations 决策树逐步演示', 'Step-through: the Permutations decision tree')} className="pm-visual">
      <header className="pm-header bt-header">
        <div>
          <p className="eyebrow">{t('LC 46 · Permutations，nums = [1, 2, 3]', 'LC 46 · Permutations with nums = [1, 2, 3]')}</p>
          <h2>{t('全排列决策树与 used 数组状态', 'Permutations decision tree and used array state')}</h2>
          <p>{t(
            '看三件事：每层从 0 开始扫但受 used 限制、每次进入/离开子树时 used 标志的置位与复位、以及答案只在最深处的叶子节点收集。',
            'Watch three things: looping from 0 at every level gated by used, setting/resetting used flags on entry/exit, and collecting answers only at the leaf nodes.',
          )}</p>
        </div>
      </header>

      <div aria-live="polite" className={`bt-step ${step.kind}`}>
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="bt-workspace">
        <div className="bt-stage-card">
          <div className="pm-used-strip">
            <span className="title">used[]:</span>
            {PM_NUMS.map((num, i) => {
              const isUsed = step.used[i];
              const isCurrent = step.index === i && (step.kind === 'choose' || step.kind === 'skip' || step.kind === 'undo');
              return (
                <div
                  className={`pm-used-item ${isUsed ? 'is-used' : 'is-free'} ${isCurrent ? 'is-current' : ''}`}
                  key={i}
                >
                  <span>nums[{i}]={num}</span>
                  <code>{isUsed ? 'T (Used)' : 'F (Free)'}</code>
                </div>
              );
            })}
          </div>

          <BacktrackTreeDiagram
            ariaLabel={t('Permutations 决策树', 'The Permutations decision tree')}
            edgeClass={(id) => (onPath.has(id) ? 'active' : visited.has(id) ? 'done' : 'ghost')}
            nodeClass={(id) => [
              id === step.node ? 'current' : '',
              id !== step.node && onPath.has(id) ? 'active' : '',
              !onPath.has(id) && collected.has(id) ? 'done' : '',
              !visited.has(id) ? 'ghost' : '',
            ].filter(Boolean).join(' ')}
            nodes={PM_NODES}
            viewBox="0 0 780 340"
          />
        </div>

        <div className="bt-side">
          <div aria-label={t('当前代码', 'Current code')} className="bt-code">
            <div className="bt-code-heading">
              <span>permute</span>
              <strong>{lineLabel}</strong>
            </div>
            <div className="bt-code-lines">
              {PM_CODE_LINES.map((line) => (
                <div
                  aria-current={activeLine === line.id ? 'step' : undefined}
                  className={activeLine === line.id ? 'active' : ''}
                  key={line.id}
                >
                  {line.code.map((code) => <code key={code}>{code || ' '}</code>)}
                </div>
              ))}
            </div>
          </div>

          <div className="bt-state">
            <div>
              <span>path</span>
              <strong>{formatList(step.path)}</strong>
            </div>
            <div>
              <span>{t('当前 i', 'Current i')}</span>
              <strong>{step.index !== undefined ? `${step.index} (nums[${step.index}]=${PM_NUMS[step.index]})` : '—'}</strong>
            </div>
          </div>

          <div className="bt-results">
            <span>result（{step.result.length} / 6）</span>
            <div>
              {step.result.map((entry, index) => (
                <code
                  className={index === step.result.length - 1 && step.kind === 'collect' ? 'fresh' : ''}
                  key={`${entry.join('-')}-${index}`}
                >
                  {formatList(entry)}
                </code>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bt-legend">
        <span><i className="current" />{t('当前节点', 'Current node')}</span>
        <span><i className="active" />{t('当前 path', 'Current path')}</span>
        <span><i className="done" />{t('已探索分支 / 已收排列', 'Explored branch / Collected permutation')}</span>
        <span><i className="ghost" />{t('尚未访问', 'Not visited yet')}</span>
      </div>

      <div className="bt-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择全排列演示步骤', 'Select a permutations demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

const CS_CANDIDATES = [2, 3, 6, 7];
const CS_TARGET = 7;

const CS_NODES = [
  { id: '', label: '[ ] rem:7', x: 390, y: 35 },
  { id: '0', label: '[2] rem:5', x: 175, y: 115 },
  { id: '1', label: '[3] rem:4', x: 400, y: 115 },
  { id: '2', label: '[6] rem:1', x: 580, y: 115 },
  { id: '3', label: '[7] rem:0 ★', x: 700, y: 115 },
  { id: '00', label: '[2,2] rem:3', x: 95, y: 200 },
  { id: '01', label: '[2,3] rem:2', x: 255, y: 200 },
  { id: '11', label: '[3,3] rem:1', x: 400, y: 200 },
  { id: '000', label: '[2,2,2] rem:1', x: 45, y: 285 },
  { id: '001', label: '[2,2,3] rem:0 ★', x: 155, y: 285 },
  // Cut nodes
  { id: '000_cut_0', parentId: '000', label: '✂ 2>1 (break)', x: 45, y: 345 },
  { id: '00_cut_2', parentId: '00', label: '✂ 6>3 (break)', x: 115, y: 252 },
  { id: '01_cut_1', parentId: '01', label: '✂ 3>2 (break)', x: 255, y: 255 },
  { id: '0_cut_2', parentId: '0', label: '✂ 6>5 (break)', x: 275, y: 155 },
  { id: '11_cut_1', parentId: '11', label: '✂ 3>1 (break)', x: 400, y: 255 },
  { id: '1_cut_2', parentId: '1', label: '✂ 6>4 (break)', x: 490, y: 155 },
  { id: '2_cut_2', parentId: '2', label: '✂ 6>1 (break)', x: 580, y: 175 },
];

const CS_CODE_LINES = [
  { id: 'base', code: ['def backtrack(start, remain):', '    if remain == 0:', '        result.append(path[:])', '        return'] },
  { id: 'loop', code: ['    for i in range(start, len(candidates)):'] },
  { id: 'prune', code: ['        if candidates[i] > remain:', '            break  # 排序后提前终止整个循环'] },
  { id: 'choose', code: ['        path.append(candidates[i])'] },
  { id: 'recurse', code: ['        backtrack(i, remain - candidates[i])  # 传 i 允许同元素复用'] },
  { id: 'undo', code: ['        path.pop()'] },
];

function buildCombinationSumSteps() {
  const candidates = CS_CANDIDATES;
  const target = CS_TARGET;
  const steps = [];
  const path = [];
  const result = [];
  const visited = [];
  const pruned = [];

  const snap = (kind, node, extra) => steps.push({
    kind,
    node,
    path: [...path],
    result: result.map((entry) => [...entry]),
    visited: [...visited],
    pruned: [...pruned],
    ...extra,
  });

  const walk = (start, remain, nodeId) => {
    visited.push(nodeId);
    if (remain === 0) {
      result.push([...path]);
      snap('collect', nodeId, { start, remain });
      return;
    }

    for (let i = start; i < candidates.length; i += 1) {
      const val = candidates[i];
      if (val > remain) {
        const cutId = `${nodeId}_cut_${i}`;
        pruned.push(cutId);
        snap('prune', nodeId, {
          start,
          remain,
          index: i,
          value: val,
          cutId,
        });
        break;
      }

      const childId = nodeId + String(i);
      path.push(val);
      snap('choose', childId, {
        start,
        remain: remain - val,
        index: i,
        value: val,
        parent: nodeId,
      });

      walk(i, remain - val, childId);

      path.pop();
      snap('undo', nodeId, {
        start,
        remain,
        index: i,
        value: val,
        child: childId,
      });
    }
  };

  snap('start', '', { start: 0, remain: target });
  walk(0, target, '');
  snap('done', '', { start: 0, remain: 0 });
  return steps;
}

const CS_STEPS = buildCombinationSumSteps();

function CombinationSumVisual() {
  const { t } = useUiCopy();
  const [activeStep, setActiveStep] = useState(0);
  const steps = CS_STEPS;
  const step = steps[activeStep];
  const visited = new Set(step.visited);
  const pruned = new Set(step.pruned);
  const collected = new Set(step.result.map((r) => {
    if (r.join(',') === '2,2,3') return '001';
    if (r.join(',') === '7') return '3';
    return '';
  }));
  const onPath = new Set();
  for (let i = 0; i <= step.node.length; i += 1) {
    onPath.add(step.node.slice(0, i));
  }

  const sumPath = step.path.reduce((acc, curr) => acc + curr, 0);
  const progressPercent = Math.min(100, Math.round((sumPath / CS_TARGET) * 100));

  const activeLine = {
    start: 'base',
    collect: 'base',
    prune: 'prune',
    choose: 'choose',
    undo: 'undo',
    done: 'loop',
  }[step.kind];

  const lineLabel = {
    base: t('remain == 0 命中目标和', 'remain == 0 target sum reached'),
    prune: t('candidates[i] > remain：break 剪掉后续所有较大分支', 'candidates[i] > remain: break cuts all larger branches'),
    choose: t('选入数字，remain 相应扣减', 'Choose candidate and deduct from remain'),
    undo: t('回溯：pop 并恢复 remain', 'Backtrack: pop and restore remain'),
    loop: t('循环结束，返回上一层', 'Loop finished, returns to caller'),
  }[activeLine];

  let title;
  let detail;
  if (step.kind === 'start') {
    title = t('初始状态：candidates = [2, 3, 6, 7], target = 7', 'Initial state: candidates = [2, 3, 6, 7], target = 7');
    detail = t(
      '数组已升序排序。从 backtrack(0, 7) 开始递归搜索，remain 实时记录距目标总和的差额。',
      'Array is sorted ascending. backtrack(0, 7) starts the search, remain tracks distance to target.',
    );
  } else if (step.kind === 'choose') {
    title = t(
      `选 candidates[${step.index}] = ${step.value}，remain 降为 ${step.remain}`,
      `Choose candidates[${step.index}] = ${step.value}, remain becomes ${step.remain}`,
    );
    detail = t(
      `path 变为 ${formatList(step.path)}。递归调用 backtrack(${step.index}, ${step.remain})：传 i 而不是 i+1，允许当前元素在下一层继续复用！`,
      `path becomes ${formatList(step.path)}. Recurse backtrack(${step.index}, ${step.remain}): passing i allows this number to be reused!`,
    );
  } else if (step.kind === 'prune') {
    title = t(
      `candidates[${step.index}] = ${step.value} > remain (${step.remain})，执行 break 剪枝！`,
      `candidates[${step.index}] = ${step.value} > remain (${step.remain}), executing break pruning!`,
    );
    detail = t(
      '因为 candidates 已升序排序，当前元素超额意味着后续所有元素（更大）必然超额，直接 break 结束循环，剪掉整批分支！',
      'Because candidates is sorted, if this element exceeds remain, all subsequent larger elements will too. break terminates the loop and cuts all further branches!',
    );
  } else if (step.kind === 'collect') {
    title = t(
      `命中目标！remain == 0，收答案 ${formatList(step.path)}（第 ${step.result.length}/2 个）`,
      `Target met! remain == 0, collected ${formatList(step.path)} (${step.result.length}/2)`,
    );
    detail = t(
      '刚好凑齐 target = 7！result.append(path[:]) 拷贝快照并 return。',
      'Exactly summed to target = 7! result.append(path[:]) copies snapshot and returns.',
    );
  } else if (step.kind === 'undo') {
    title = t(
      `回溯：pop 掉 ${step.value}，恢复 remain 为 ${step.remain}`,
      `Backtrack: pop ${step.value}, restore remain to ${step.remain}`,
    );
    detail = t(
      `path 改回 ${formatList(step.path)}，for 循环继续探索下一个候选数。`,
      `path restored to ${formatList(step.path)}. The loop proceeds to test the next candidate.`,
    );
  } else {
    title = t('搜索完成：共找到 2 组组合解 [[2, 2, 3], [7]]', 'Search complete: found 2 combination solutions [[2, 2, 3], [7]]');
    detail = t(
      '通过"传 i 允许复用"和"有序 break 剪枝"，高效搜索且不生成任何重复组合（如 [3,2,2]）。',
      'With "pass i for reuse" and "sorted break pruning", it searches efficiently without duplicating combinations like [3,2,2].',
    );
  }

  return (
    <section aria-label={t('Combination Sum 决策树逐步演示', 'Step-through: the Combination Sum decision tree')} className="cs-visual">
      <header className="cs-header bt-header">
        <div>
          <p className="eyebrow">{t('LC 39 · Combination Sum，candidates = [2, 3, 6, 7], target = 7', 'LC 39 · Combination Sum with [2, 3, 6, 7], target = 7')}</p>
          <h2>{t('预算扣减、同元素复用与 break 剪枝', 'Budget countdown, candidate reuse, and break pruning')}</h2>
          <p>{t(
            '看三件事：递归传 i 允许数字重复选取（[2, 2, 3]）、remain 预算如何随选择递减、以及 candidates[i] > remain 时 break 如何瞬间斩断后续分支。',
            'Watch three things: recursion passing i for repetition ([2, 2, 3]), remain decreasing with choices, and break instantly pruning all larger candidates.',
          )}</p>
        </div>
      </header>

      <div aria-live="polite" className={`bt-step ${step.kind}`}>
        <span>{activeStep + 1} / {steps.length}</span>
        <strong>{title}</strong>
        <p>{detail}</p>
      </div>

      <div className="bt-workspace">
        <div className="bt-stage-card">
          <div className="cs-budget-bar">
            <div className="cs-budget-meta">
              <span>{t('目标 target: 7', 'Target: 7')}</span>
              <span>{t(`已选总和: ${sumPath} (${step.path.join('+') || '0'})`, `Sum: ${sumPath} (${step.path.join('+') || '0'})`)}</span>
              <span>{t(`剩余预算 remain: ${step.remain}`, `Remain: ${step.remain}`)}</span>
            </div>
            <div className="cs-budget-track">
              <div
                className={`cs-budget-fill ${step.remain === 0 ? 'zero' : ''}`}
                style={{ width: `${progressPercent}%` }}
              />
            </div>
          </div>

          <BacktrackTreeDiagram
            ariaLabel={t('Combination Sum 决策树', 'The Combination Sum decision tree')}
            edgeClass={(id) => (
              pruned.has(id) ? 'cut' : onPath.has(id) ? 'active' : visited.has(id) ? 'done' : 'ghost'
            )}
            nodeClass={(id) => [
              pruned.has(id) ? 'cut' : '',
              !pruned.has(id) && id === step.node ? 'current' : '',
              !pruned.has(id) && id !== step.node && onPath.has(id) ? 'active' : '',
              !pruned.has(id) && !onPath.has(id) && visited.has(id) ? 'done' : '',
              !pruned.has(id) && collected.has(id) ? 'done solution' : '',
              !pruned.has(id) && !visited.has(id) ? 'ghost' : '',
            ].filter(Boolean).join(' ')}
            nodes={CS_NODES}
            viewBox="0 0 780 370"
          />
        </div>

        <div className="bt-side">
          <div aria-label={t('当前代码', 'Current code')} className="bt-code">
            <div className="bt-code-heading">
              <span>combinationSum</span>
              <strong>{lineLabel}</strong>
            </div>
            <div className="bt-code-lines">
              {CS_CODE_LINES.map((line) => (
                <div
                  aria-current={activeLine === line.id ? 'step' : undefined}
                  className={activeLine === line.id ? 'active' : ''}
                  key={line.id}
                >
                  {line.code.map((code) => <code key={code}>{code || ' '}</code>)}
                </div>
              ))}
            </div>
          </div>

          <div className="bt-state">
            <div>
              <span>path</span>
              <strong>{formatList(step.path)}</strong>
            </div>
            <div>
              <span>remain</span>
              <strong style={{ color: step.remain === 0 ? '#18775a' : undefined }}>{step.remain}</strong>
            </div>
            <div>
              <span>start (i)</span>
              <strong>{step.start ?? 0}</strong>
            </div>
            <div>
              <span>{t('已剪枝数', 'Pruned')}</span>
              <strong>{step.pruned.length}</strong>
            </div>
          </div>

          <div className="bt-results">
            <span>result（{step.result.length} / 2）</span>
            <div>
              {step.result.map((entry, index) => (
                <code
                  className={index === step.result.length - 1 && step.kind === 'collect' ? 'fresh' : ''}
                  key={`${entry.join('-')}-${index}`}
                >
                  {formatList(entry)}
                </code>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="bt-legend">
        <span><i className="current" />{t('当前节点', 'Current node')}</span>
        <span><i className="active" />{t('当前 path', 'Current path')}</span>
        <span><i className="done" />{t('已探索 / 解', 'Explored / Solution')}</span>
        <span><i className="cut" />{t('> remain，break 剪枝', '> remain, break pruned')}</span>
        <span><i className="ghost" />{t('尚未访问', 'Not visited yet')}</span>
      </div>

      <div className="bt-controls">
        <button disabled={activeStep === 0} onClick={() => setActiveStep((current) => Math.max(0, current - 1))} type="button">
          ← {t('上一步', 'Previous')}
        </button>
        <input
          aria-label={t('选择组合总和演示步骤', 'Select a combination sum demo step')}
          max={steps.length - 1}
          min="0"
          onChange={(event) => setActiveStep(Number(event.target.value))}
          type="range"
          value={activeStep}
        />
        <button
          className="primary"
          disabled={activeStep === steps.length - 1}
          onClick={() => setActiveStep((current) => Math.min(steps.length - 1, current + 1))}
          type="button"
        >
          {t('下一步', 'Next')} →
        </button>
      </div>
    </section>
  );
}

function MartingaleRandomWalkVisual() {
  const { t } = useUiCopy();
  const [activeTab, setActiveTab] = useState('random-walk');

  // Tab 1: Random Walk & Wald Martingales state
  const [upperA, setUpperA] = useState(5);
  const [lowerB, setLowerB] = useState(5);
  const [probP, setProbP] = useState(0.5);
  const [simPath, setSimPath] = useState(null);

  // Computations for 1D Random Walk
  const q = 1 - probP;
  let probA = 0;
  let expectedTime = 0;

  if (Math.abs(probP - 0.5) < 1e-6) {
    probA = lowerB / (upperA + lowerB);
    expectedTime = upperA * lowerB;
  } else {
    const qOverP = q / probP;
    probA = (Math.pow(qOverP, lowerB) - 1) / (Math.pow(qOverP, upperA + lowerB) - 1);
    expectedTime = (upperA * probA - lowerB * (1 - probA)) / (probP - q);
  }

  const handleSimulate = () => {
    let current = 0;
    const path = [0];
    let steps = 0;
    const maxSteps = 1500;
    while (current < upperA && current > -lowerB && steps < maxSteps) {
      const step = Math.random() < probP ? 1 : -1;
      current += step;
      path.push(current);
      steps++;
    }
    setSimPath({
      path,
      outcome: current >= upperA ? 'hit_a' : current <= -lowerB ? 'hit_b' : 'max_steps',
      steps,
    });
  };

  // Tab 2: Secretary Problem state
  const [totalN, setTotalN] = useState(30);
  const [cutoffK, setCutoffK] = useState(11);

  const secretaryData = useMemo(() => {
    const arr = [];
    let bestK = 1;
    let maxP = 0;
    for (let k = 1; k <= totalN; k++) {
      let sumH = 0;
      for (let j = k; j <= totalN; j++) {
        sumH += 1 / (j - 1 || 1);
      }
      const pSuccess = k === 1 ? 1 / totalN : ((k - 1) / totalN) * sumH;
      if (pSuccess > maxP) {
        maxP = pSuccess;
        bestK = k;
      }
      arr.push({ k, p: pSuccess });
    }
    return { arr, bestK, maxP };
  }, [totalN]);

  const currentCutoffProb = secretaryData.arr.find((item) => item.k === cutoffK)?.p ?? 0;

  // Tab 3: Pattern Waiting Times & Li's Martingale
  const [patternA, setPatternA] = useState('HTTH');
  const [patternB, setPatternB] = useState('HTHT');

  const computePatternInfo = (pattern) => {
    const p = pattern.toUpperCase().replace(/[^HT]/g, '');
    const m = p.length || 1;
    const overlaps = [];
    let expectedT = 0;
    for (let len = 1; len <= m; len++) {
      const prefix = p.slice(0, len);
      const suffix = p.slice(m - len);
      const isMatch = prefix === suffix;
      const val = isMatch ? Math.pow(2, len) : 0;
      expectedT += val;
      overlaps.push({ len, prefix, suffix, isMatch, val });
    }
    return { pattern: p, m, overlaps, expectedT };
  };

  const computeCrossOverlap = (p1, p2) => {
    const minLen = Math.min(p1.length, p2.length);
    let sum = 0;
    for (let len = 1; len <= minLen; len++) {
      const prefix1 = p1.slice(0, len);
      const suffix2 = p2.slice(p2.length - len);
      if (prefix1 === suffix2) {
        sum += Math.pow(2, len);
      }
    }
    return sum;
  };

  const infoA = useMemo(() => computePatternInfo(patternA), [patternA]);
  const infoB = useMemo(() => computePatternInfo(patternB), [patternB]);

  const penneyAoverB = useMemo(() => {
    const aa = computeCrossOverlap(infoA.pattern, infoA.pattern);
    const bb = computeCrossOverlap(infoB.pattern, infoB.pattern);
    const ab = computeCrossOverlap(infoA.pattern, infoB.pattern);
    const ba = computeCrossOverlap(infoB.pattern, infoA.pattern);
    const num = bb - ba;
    const den = aa - ab + (bb - ba);
    return den !== 0 ? Math.max(0, Math.min(1, num / den)) : 0.5;
  }, [infoA.pattern, infoB.pattern]);

  return (
    <section aria-label={t('鞅论、Wald 等式与最优时停交互演示', 'Martingale, Wald equations, and optimal stopping visual lab')} className="mrw-vis">
      <header className="mrw-header">
        <div>
          <p className="eyebrow">{t('鞅论与最优时停交互实验室', 'Martingale & Optimal Stopping Interactive Lab')}</p>
          <h2>{t('Wald 等式、1D 随机游走与最优决策', 'Wald\'s Identities, 1D Random Walks & Optimal Stopping')}</h2>
          <p>{t(
            '切换不同模块，实时验证随机游走吸收概率、期望停止时间、秘书问题 37% 法则与 Li\'s 赌场鞅模式计算。',
            'Switch tabs to interactively verify random walk hitting probabilities, expected exit times, the 37% rule, and pattern waiting times.',
          )}</p>
        </div>
        <div className="mrw-tabs" role="tablist">
          <button
            aria-selected={activeTab === 'random-walk'}
            className={activeTab === 'random-walk' ? 'active' : ''}
            onClick={() => setActiveTab('random-walk')}
            role="tab"
            type="button"
          >
            {t('1D 随机游走与鞅', '1D Random Walk')}
          </button>
          <button
            aria-selected={activeTab === 'secretary'}
            className={activeTab === 'secretary' ? 'active' : ''}
            onClick={() => setActiveTab('secretary')}
            role="tab"
            type="button"
          >
            {t('秘书问题 37% 法则', 'Secretary Problem (37%)')}
          </button>
          <button
            aria-selected={activeTab === 'pattern'}
            className={activeTab === 'pattern' ? 'active' : ''}
            onClick={() => setActiveTab('pattern')}
            role="tab"
            type="button"
          >
            {t('模式等待与赌场鞅', 'Pattern Waiting & Li\'s Martingale')}
          </button>
        </div>
      </header>

      {activeTab === 'random-walk' && (
        <div className="mrw-body">
          <div className="mrw-controls-grid">
            <label>
              <span>{t('上吸收边界 +a', 'Upper Barrier +a')}: <strong>+{upperA}</strong></span>
              <input
                max="20"
                min="1"
                onChange={(e) => setUpperA(Number(e.target.value))}
                type="range"
                value={upperA}
              />
            </label>
            <label>
              <span>{t('下吸收边界 -b', 'Lower Barrier -b')}: <strong>-{lowerB}</strong></span>
              <input
                max="20"
                min="1"
                onChange={(e) => setLowerB(Number(e.target.value))}
                type="range"
                value={lowerB}
              />
            </label>
            <label>
              <span>{t('步长向上概率 p', 'Upward Probability p')}: <strong>{probP.toFixed(2)}</strong> (q = {q.toFixed(2)})</span>
              <input
                max="0.9"
                min="0.1"
                step="0.05"
                onChange={(e) => setProbP(Number(e.target.value))}
                type="range"
                value={probP}
              />
            </label>
          </div>

          <div className="mrw-metrics-cards">
            <div className="mrw-card">
              <span>{t('胜率 P(到达 +a)', 'Win Probability P(Hit +a)')}</span>
              <strong className="accent-green">{(probA * 100).toFixed(2)}%</strong>
              <small>{probP === 0.5 ? t(`对称公式: ${lowerB} / (${upperA}+${lowerB}) = b/(a+b)`, `Symmetric: ${lowerB}/(${upperA}+${lowerB}) = b/(a+b)`) : t(`指数鞅公式: ((q/p)^b - 1) / ((q/p)^(a+b) - 1)`, `Exp Martingale: ((q/p)^b - 1) / ((q/p)^(a+b) - 1)`)}</small>
            </div>
            <div className="mrw-card">
              <span>{t('败率 P(到达 -b)', 'Loss Probability P(Hit -b)')}</span>
              <strong className="accent-red">{((1 - probA) * 100).toFixed(2)}%</strong>
              <small>{probP === 0.5 ? t(`对称公式: ${upperA} / (${upperA}+${lowerB}) = a/(a+b)`, `Symmetric: ${upperA}/(${upperA}+${lowerB}) = a/(a+b)`) : t(`余概率: 1 - P(Hit +a)`, `Complement: 1 - P(Hit +a)`)}</small>
            </div>
            <div className="mrw-card">
              <span>{t('期望停止时间 E[T]', 'Expected Exit Time E[T]')}</span>
              <strong className="accent-blue">{expectedTime.toFixed(2)} {t('步', 'steps')}</strong>
              <small>{probP === 0.5 ? t(`二阶 Wald / 方差鞅: ${upperA} × ${lowerB} = ab`, `Wald 2nd / Var Martingale: ${upperA} × ${lowerB} = ab`) : t(`一阶 Wald / 漂移鞅: (a·Pa - b·Pb) / (p-q)`, `Wald 1st / Drift Martingale: (a·Pa - b·Pb)/(p-q)`)}</small>
            </div>
          </div>

          <div className="mrw-sim-section">
            <div className="mrw-sim-header">
              <button className="primary-btn" onClick={handleSimulate} type="button">
                🎲 {t('单次蒙特卡洛模拟游走', 'Simulate Single Random Walk')}
              </button>
              {simPath && (
                <span className="mrw-sim-status">
                  {simPath.outcome === 'hit_a' ? (
                    <strong className="pos">🎯 {t(`成功命中 +${upperA} (耗时 ${simPath.steps} 步)`, `Reached +${upperA} in ${simPath.steps} steps`)}</strong>
                  ) : (
                    <strong className="neg">💥 {t(`触碰下界 -${lowerB} (耗时 ${simPath.steps} 步)`, `Hit lower -${lowerB} in ${simPath.steps} steps`)}</strong>
                  )}
                </span>
              )}
            </div>

            {simPath && (
              <div className="mrw-path-vis">
                <div className="mrw-axis-label top">+{upperA} ({t('上界 a', 'Upper a')})</div>
                <div className="mrw-canvas-wrap">
                  <svg className="mrw-svg" preserveAspectRatio="none" viewBox={`0 0 ${Math.max(simPath.path.length - 1, 1)} ${upperA + lowerB}`}>
                    <line
                      className="mrw-zero-line"
                      x1="0"
                      x2={simPath.path.length - 1}
                      y1={upperA}
                      y2={upperA}
                    />
                    <polyline
                      className="mrw-path-line"
                      points={simPath.path.map((val, idx) => `${idx},${upperA - val}`).join(' ')}
                    />
                  </svg>
                </div>
                <div className="mrw-axis-label bottom">-{lowerB} ({t('下界 -b', 'Lower -b')})</div>
              </div>
            )}
          </div>
        </div>
      )}

      {activeTab === 'secretary' && (
        <div className="mrw-body">
          <div className="mrw-controls-grid">
            <label>
              <span>{t('候选人总人数 n', 'Total Candidates n')}: <strong>{totalN}</strong></span>
              <input
                max="50"
                min="5"
                onChange={(e) => {
                  const val = Number(e.target.value);
                  setTotalN(val);
                  setCutoffK(Math.min(cutoffK, val));
                }}
                type="range"
                value={totalN}
              />
            </label>
            <label>
              <span>{t('当前设定样本观察区 k', 'Rejection Threshold k')}: <strong>{cutoffK}</strong> ({((cutoffK / totalN) * 100).toFixed(1)}%)</span>
              <input
                max={totalN}
                min="1"
                onChange={(e) => setCutoffK(Number(e.target.value))}
                type="range"
                value={cutoffK}
              />
            </label>
          </div>

          <div className="mrw-metrics-cards">
            <div className="mrw-card">
              <span>{t('当前设定命中率 P(k)', 'Current Hit Rate P(k)')}</span>
              <strong className={cutoffK === secretaryData.bestK ? 'accent-green' : 'accent-blue'}>
                {(currentCutoffProb * 100).toFixed(2)}%
              </strong>
              <small>{t(`前 ${cutoffK - 1} 人仅观察，从第 ${cutoffK} 人开始选优`, `Observe first ${cutoffK - 1}, select first better from ${cutoffK}`)}</small>
            </div>
            <div className="mrw-card">
              <span>{t('离散最优阈值 k*', 'Discrete Optimal Cutoff k*')}</span>
              <strong className="accent-green">k* = {secretaryData.bestK}</strong>
              <small>{t(`离散最高概率: ${(secretaryData.maxP * 100).toFixed(2)}%`, `Discrete Peak: ${(secretaryData.maxP * 100).toFixed(2)}%`)}</small>
            </div>
            <div className="mrw-card">
              <span>{t('连续渐近极限 (n -> ∞)', 'Asymptotic Limit (n -> ∞)')}</span>
              <strong className="accent-purple">1/e ≈ 36.79%</strong>
              <small>{t('最优拒绝比例 x* = 1/e ≈ 36.8%', 'Optimal rejection fraction x* = 1/e ≈ 36.8%')}</small>
            </div>
          </div>

          <div className="mrw-sec-chart">
            <div className="mrw-sec-bars">
              {secretaryData.arr.map(({ k, p }) => {
                const isSelected = k === cutoffK;
                const isBest = k === secretaryData.bestK;
                const heightPct = Math.max(4, (p / 0.45) * 100);
                return (
                  <button
                    aria-label={`k=${k}, P=${(p * 100).toFixed(2)}%`}
                    className={`mrw-bar-col ${isSelected ? 'selected' : ''} ${isBest ? 'best' : ''}`}
                    key={k}
                    onClick={() => setCutoffK(k)}
                    type="button"
                  >
                    <div className="mrw-bar-fill" style={{ height: `${heightPct}%` }} />
                    <span className="mrw-bar-lbl">{k % 5 === 0 || k === 1 || isBest ? k : ''}</span>
                  </button>
                );
              })}
            </div>
            <div className="mrw-chart-hint">
              <span>{t('柱状图展示各个候选拒绝阈值 k 对应的全局最优命中概率 P(k)，点击柱子可直接切换阈值。', 'Bar chart shows success probability P(k) across thresholds k. Click any bar to select.')}</span>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'pattern' && (
        <div className="mrw-body">
          <div className="mrw-pattern-selector">
            <span className="mrw-lbl">{t('选择预设或自定义模式 A / B:', 'Select preset or custom pattern A / B:')}</span>
            <div className="mrw-preset-btns">
              <button
                className={patternA === 'HTTH' && patternB === 'HTHT' ? 'active' : ''}
                onClick={() => { setPatternA('HTTH'); setPatternB('HTHT'); }}
                type="button"
              >
                HTTH vs HTHT (18 vs 20)
              </button>
              <button
                className={patternA === 'HHHH' && patternB === 'THHH' ? 'active' : ''}
                onClick={() => { setPatternA('HHHH'); setPatternB('THHH'); }}
                type="button"
              >
                HHHH vs THHH (30 vs 16)
              </button>
              <button
                className={patternA === 'HHTT' && patternB === 'HTHH' ? 'active' : ''}
                onClick={() => { setPatternA('HHTT'); setPatternB('HTHH'); }}
                type="button"
              >
                HHTT vs HTHH (16 vs 18)
              </button>
            </div>
          </div>

          <div className="mrw-pat-grid">
            <div className="mrw-pat-card">
              <div className="mrw-pat-title">
                <h3>{t('模式 A: ', 'Pattern A: ')} <code>{infoA.pattern}</code></h3>
                <span className="mrw-exp-tag">E[T_A] = {infoA.expectedT} {t('次', 'tosses')}</span>
              </div>
              <table className="mrw-table">
                <thead>
                  <tr>
                    <th>{t('前缀长度 k', 'Length k')}</th>
                    <th>{t('前缀 Prefix', 'Prefix')}</th>
                    <th>{t('后缀 Suffix', 'Suffix')}</th>
                    <th>{t('重合判定', 'Match?')}</th>
                    <th>{t('奖金 2^k', 'Payout')}</th>
                  </tr>
                </thead>
                <tbody>
                  {infoA.overlaps.map((row) => (
                    <tr className={row.isMatch ? 'match-row' : ''} key={row.len}>
                      <td>{row.len}</td>
                      <td><code>{row.prefix}</code></td>
                      <td><code>{row.suffix}</code></td>
                      <td>{row.isMatch ? '✅ MATCH' : '❌ NO'}</td>
                      <td>{row.isMatch ? `+$${row.val}` : '$0'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mrw-math-sum">
                (A * A)_2 = {infoA.overlaps.filter((r) => r.isMatch).map((r) => `2^${r.len}`).join(' + ')} = <strong>{infoA.expectedT}</strong>
              </p>
            </div>

            <div className="mrw-pat-card">
              <div className="mrw-pat-title">
                <h3>{t('模式 B: ', 'Pattern B: ')} <code>{infoB.pattern}</code></h3>
                <span className="mrw-exp-tag">E[T_B] = {infoB.expectedT} {t('次', 'tosses')}</span>
              </div>
              <table className="mrw-table">
                <thead>
                  <tr>
                    <th>{t('前缀长度 k', 'Length k')}</th>
                    <th>{t('前缀 Prefix', 'Prefix')}</th>
                    <th>{t('后缀 Suffix', 'Suffix')}</th>
                    <th>{t('重合判定', 'Match?')}</th>
                    <th>{t('奖金 2^k', 'Payout')}</th>
                  </tr>
                </thead>
                <tbody>
                  {infoB.overlaps.map((row) => (
                    <tr className={row.isMatch ? 'match-row' : ''} key={row.len}>
                      <td>{row.len}</td>
                      <td><code>{row.prefix}</code></td>
                      <td><code>{row.suffix}</code></td>
                      <td>{row.isMatch ? '✅ MATCH' : '❌ NO'}</td>
                      <td>{row.isMatch ? `+$${row.val}` : '$0'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mrw-math-sum">
                (B * B)_2 = {infoB.overlaps.filter((r) => r.isMatch).map((r) => `2^${r.len}`).join(' + ')} = <strong>{infoB.expectedT}</strong>
              </p>
            </div>
          </div>

          <div className="mrw-penney-box">
            <h4>{t('Penney\'s Game 竞速胜率 (谁先出现谁获胜)', 'Penney\'s Game Race (First to appear wins)')}</h4>
            <div className="mrw-penney-bar">
              <div className="mrw-penney-a" style={{ width: `${penneyAoverB * 100}%` }}>
                {infoA.pattern}: {(penneyAoverB * 100).toFixed(1)}%
              </div>
              <div className="mrw-penney-b" style={{ width: `${(1 - penneyAoverB) * 100}%` }}>
                {infoB.pattern}: {((1 - penneyAoverB) * 100).toFixed(1)}%
              </div>
            </div>
            <small>{t('注：由 Conway 算法与 Li\'s 赌场鞅推导，模式间的非传递博弈使得后手总能构造胜率 > 50% 的模式。', 'Note: Derived from Conway\'s algorithm & Li\'s martingale, Penney\'s game is non-transitive—Player 2 can always counter with > 50% odds.')}</small>
          </div>
        </div>
      )}
    </section>
  );
}

function MarkdownPre({ children, ...props }) {
  const child = Array.isArray(children) ? children[0] : children;
  const className = child?.props?.className ?? '';
  const match = /language-(quiz|mcq|mermaid|topo-demo|bellman-demo|segment-tree-demo|interval-merge-demo|interval-insert-demo|interval-rooms-demo|interval-query-demo|pow-demo|sliding-window-demo|longest-substring-demo|sliding-window-patterns|monotonic-stack-demo|largest-rectangle-demo|binary-search-template-demo|linked-list-reversal-demo|fast-slow-pointer-demo|array-duplicate-demo|lru-cache-demo|tree-traversal-demo|avl-rotation-demo|build-tree-demo|median-two-heaps-demo|three-sum-demo|rain-water-demo|simple-sort-race-demo|efficient-sort-race-demo|high-dimensional-integral-demo|record-minimum-demo|message-queue-demo|business-algorithm-map|system-design-overview-visual|photo-sharing-architecture-visual|async-messaging-architecture-visual|virtualization-container-visual|grid-multi-source-bfs-demo|union-find-demo|quickselect-partition-demo|trie-core-demo|trie-wildcard-demo|backtracking-patterns|backtracking-tree-demo|permutations-demo|combination-sum-demo|backtracking-dedup-demo|n-queens-demo|greedy-patterns|kadane-demo|jump-game-demo|gas-station-demo|partition-labels-demo|vtable-dispatch-demo|false-sharing-demo|fork-cow-demo|epoll-vs-select-demo|shared-ptr-cycle-demo|martingale-rw-demo|random-walk-ruin-demo|brownian-motion-demo|two-d-walk-demo|ito-geometry-demo|reflection-principle-demo|delta-hedging-demo|game-theory-interactive-demo)/.exec(className);

  if (match?.[1] === 'mermaid') {
    return <MermaidDiagram chart={extractPlainText(child.props.children).replace(/\n$/, '')} />;
  }

  if (match?.[1] === 'topo-demo') {
    return <ForeignDictionaryTopoVisual />;
  }

  if (match?.[1] === 'bellman-demo') {
    return <CheapestFlightsBellmanVisual />;
  }

  if (match?.[1] === 'segment-tree-demo') {
    return <SegmentTreeLISVisual />;
  }

  if (match?.[1]?.startsWith('interval-')) {
    return <IntervalPatternVisual kind={match[1]} />;
  }

  if (match?.[1] === 'pow-demo') {
    return <BinaryPowVisual />;
  }

  if (match?.[1] === 'sliding-window-demo') {
    return <SlidingWindowVisual />;
  }

  if (match?.[1] === 'longest-substring-demo') {
    return <LongestSubstringVisual />;
  }

  if (match?.[1] === 'sliding-window-patterns') {
    return <SlidingWindowPatternAtlas />;
  }

  if (match?.[1] === 'monotonic-stack-demo') {
    return <MonotonicStackVisual />;
  }

  if (match?.[1] === 'largest-rectangle-demo') {
    return <LargestRectangleVisual />;
  }

  if (match?.[1] === 'binary-search-template-demo') {
    return <BinarySearchTemplateVisual />;
  }

  if (match?.[1] === 'linked-list-reversal-demo') {
    return <LinkedListReversalVisual />;
  }

  if (match?.[1] === 'fast-slow-pointer-demo') {
    return <FastSlowPointerVisual />;
  }

  if (match?.[1] === 'array-duplicate-demo') {
    return <ArrayDuplicateVisual />;
  }

  if (match?.[1] === 'lru-cache-demo') {
    return <LRUCacheVisual />;
  }

  if (match?.[1] === 'tree-traversal-demo') {
    return <TreeTraversalVisual />;
  }

  if (match?.[1] === 'avl-rotation-demo') {
    return <AVLRotationVisual />;
  }

  if (match?.[1] === 'build-tree-demo') {
    return <BuildTreeVisual />;
  }

  if (match?.[1] === 'median-two-heaps-demo') {
    return <MedianTwoHeapsVisual />;
  }

  if (match?.[1] === 'grid-multi-source-bfs-demo') {
    return <RottingOrangesBFSVisual />;
  }

  if (match?.[1] === 'union-find-demo') {
    return <UnionFindVisual />;
  }

  if (match?.[1] === 'quickselect-partition-demo') {
    return <QuickSelectPartitionVisual />;
  }

  if (match?.[1] === 'trie-core-demo') {
    return <TrieCoreVisual />;
  }

  if (match?.[1] === 'trie-wildcard-demo') {
    return <TrieWildcardVisual />;
  }

  if (match?.[1] === 'greedy-patterns') {
    return <GreedyPatternAtlas />;
  }

  if (match?.[1] === 'kadane-demo') {
    return <KadaneVisual />;
  }

  if (match?.[1] === 'jump-game-demo') {
    return <JumpGameVisual />;
  }

  if (match?.[1] === 'gas-station-demo') {
    return <GasStationVisual />;
  }

  if (match?.[1] === 'partition-labels-demo') {
    return <PartitionLabelsVisual />;
  }

  if (match?.[1] === 'backtracking-patterns') {
    return <BacktrackingPatternAtlas />;
  }

  if (match?.[1] === 'backtracking-tree-demo') {
    return <BacktrackingTreeVisual />;
  }

  if (match?.[1] === 'permutations-demo') {
    return <PermutationsVisual />;
  }

  if (match?.[1] === 'combination-sum-demo') {
    return <CombinationSumVisual />;
  }

  if (match?.[1] === 'backtracking-dedup-demo') {
    return <BacktrackingDedupVisual />;
  }

  if (match?.[1] === 'n-queens-demo') {
    return <NQueensVisual />;
  }

  if (match?.[1] === 'three-sum-demo') {
    return <ThreeSumVisual />;
  }

  if (match?.[1] === 'rain-water-demo') {
    return <RainWaterVisual />;
  }

  if (match?.[1] === 'simple-sort-race-demo') {
    return <SimpleSortRaceVisual />;
  }

  if (match?.[1] === 'efficient-sort-race-demo') {
    return <EfficientSortRaceVisual />;
  }

  if (match?.[1] === 'high-dimensional-integral-demo') {
    return <HighDimensionalIntegralVisual />;
  }

  if (match?.[1] === 'record-minimum-demo') {
    return <RecordMinimumVisual />;
  }

  if (match?.[1] === 'message-queue-demo') {
    return <MessageQueueVisual />;
  }

  if (match?.[1] === 'business-algorithm-map') {
    return <BusinessAlgorithmMap />;
  }

  if (match?.[1] === 'system-design-overview-visual') {
    return <SystemDesignOverviewVisual />;
  }

  if (match?.[1] === 'photo-sharing-architecture-visual') {
    return <PhotoSharingArchitectureVisual />;
  }

  if (match?.[1] === 'async-messaging-architecture-visual') {
    return <AsyncMessagingArchitectureVisual />;
  }

  if (match?.[1] === 'virtualization-container-visual') {
    return <VirtualizationContainerVisual />;
  }

  if (match?.[1] === 'vtable-dispatch-demo') {
    return <VtableDispatchVisual />;
  }

  if (match?.[1] === 'false-sharing-demo') {
    return <FalseSharingVisual />;
  }

  if (match?.[1] === 'fork-cow-demo') {
    return <ForkCowVisual />;
  }

  if (match?.[1] === 'epoll-vs-select-demo') {
    return <EpollVsSelectVisual />;
  }

  if (match?.[1] === 'shared-ptr-cycle-demo') {
    return <SharedPtrCycleVisual />;
  }

  if (match?.[1] === 'martingale-rw-demo' || match?.[1] === 'random-walk-ruin-demo') {
    return <MartingaleRandomWalkVisual />;
  }

  if (match?.[1] === 'brownian-motion-demo') {
    return <BrownianMotionVisual />;
  }

  if (match?.[1] === 'two-d-walk-demo') {
    return <TwoDRandomWalkVisual />;
  }

  if (match?.[1] === 'ito-geometry-demo') {
    return <ItoGeometryVisual />;
  }

  if (match?.[1] === 'reflection-principle-demo') {
    return <ReflectionPrincipleVisual />;
  }

  if (match?.[1] === 'delta-hedging-demo') {
    return <DeltaHedgingVisual />;
  }

  if (match?.[1] === 'game-theory-interactive-demo') {
    return <GameTheoryVisual />;
  }

  if (match) {
    return <QuizBlock source={extractPlainText(child.props.children).replace(/\n$/, '')} />;
  }

  return <CodeBlock className={className} source={extractPlainText(child?.props?.children)} {...props} />;
}

let mermaidLoader = null;

async function getMermaid() {
  if (!mermaidLoader) {
    mermaidLoader = import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'base',
        flowchart: {
          curve: 'basis',
          htmlLabels: true,
          nodeSpacing: 70,
          rankSpacing: 85,
          padding: 24,
        },
        themeVariables: {
          background: 'transparent',
          primaryColor: '#e9f3f5',
          primaryTextColor: '#102735',
          primaryBorderColor: '#2f7b94',
          lineColor: '#315568',
          secondaryColor: '#fff7e3',
          tertiaryColor: '#f4fbf7',
          fontFamily: '"IBM Plex Mono", "Courier New", monospace',
        },
      });
      return mermaid;
    });
  }

  return mermaidLoader;
}

function MermaidDiagram({ chart }) {
  const containerRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const reactId = useId();
  const diagramId = useMemo(() => `mermaid-${reactId.replace(/[^a-zA-Z0-9_-]/g, '')}`, [reactId]);
  const zoomPercent = Math.round(zoom * 100);
  const changeZoom = (delta) => {
    setZoom((current) => Math.min(2.25, Math.max(0.75, Number((current + delta).toFixed(2)))));
  };

  useEffect(() => {
    let cancelled = false;

    async function renderDiagram() {
      if (!containerRef.current) {
        return;
      }

      const mermaid = await getMermaid();
      if (cancelled || !containerRef.current) {
        return;
      }
      containerRef.current.innerHTML = '';

      try {
        const { svg } = await mermaid.render(diagramId, chart);
        if (!cancelled && containerRef.current) {
          containerRef.current.innerHTML = svg;
        }
      } catch {
        if (!cancelled && containerRef.current) {
          containerRef.current.textContent = 'Diagram failed to render.';
        }
      }
    }

    renderDiagram();

    return () => {
      cancelled = true;
    };
  }, [chart, diagramId]);

  return (
    <figure className="mermaid-frame">
      <figcaption className="mermaid-toolbar">
        <span className="mermaid-label">Diagram</span>
        <span className="mermaid-zoom-controls" aria-label="Diagram zoom controls">
          <button
            type="button"
            className="diagram-zoom-button"
            onClick={() => changeZoom(-0.15)}
            disabled={zoom <= 0.75}
            aria-label="Zoom out diagram"
            title="Zoom out"
          >
            -
          </button>
          <button
            type="button"
            className="diagram-zoom-reset"
            onClick={() => setZoom(1)}
            aria-label="Reset diagram zoom"
            title="Reset zoom"
          >
            {zoomPercent}%
          </button>
          <button
            type="button"
            className="diagram-zoom-button"
            onClick={() => changeZoom(0.15)}
            disabled={zoom >= 2.25}
            aria-label="Zoom in diagram"
            title="Zoom in"
          >
            +
          </button>
        </span>
      </figcaption>
      <div className="mermaid-diagram">
        <div
          className="mermaid-canvas"
          ref={containerRef}
          role="img"
          aria-label="Mermaid diagram"
          style={{ '--diagram-zoom': zoom }}
        />
      </div>
    </figure>
  );
}

function CodeBlock({ className = '', source = '' }) {
  const [copied, setCopied] = useState(false);
  const language = className.match(/language-([\w-]+)/)?.[1] ?? 'text';
  const label = formatCodeLanguage(language);
  const code = source.replace(/\n$/, '');

  const copyCode = async () => {
    if (!navigator?.clipboard) {
      return;
    }

    await navigator.clipboard.writeText(code);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1400);
  };

  return (
    <figure className="code-frame">
      <figcaption className="code-frame-header">
        <span>{label}</span>
        <button type="button" onClick={copyCode}>
          {copied ? 'Copied' : 'Copy'}
        </button>
      </figcaption>
      <pre>
        <code className={className}>
          <HighlightedCode code={code} language={language} />
        </code>
      </pre>
    </figure>
  );
}

function HighlightedCode({ code, language }) {
  const tokens = tokenizeCode(code, language);
  return tokens.map((token, index) => (
    token.type === 'text'
      ? <Fragment key={index}>{token.value}</Fragment>
      : <span className={`code-token ${token.type}`} key={index}>{token.value}</span>
  ));
}

function formatCodeLanguage(language) {
  const labels = {
    js: 'JavaScript',
    jsx: 'React JSX',
    py: 'Python',
    python: 'Python',
    text: 'Text',
  };

  return labels[language] ?? language.toUpperCase();
}

function tokenizeCode(code, language) {
  if (!['py', 'python', 'js', 'jsx'].includes(language)) {
    return [{ type: 'text', value: code }];
  }

  const keywordPattern = language === 'python'
    || language === 'py'
    ? 'False|None|True|and|as|break|class|continue|def|elif|else|for|from|if|import|in|is|not|or|return|while|with'
    : 'const|let|var|function|return|if|else|for|while|import|from|export|class|new|true|false|null|undefined|await|async';
  const builtinPattern = language === 'python' || language === 'py'
    ? 'Counter|List|abs|bool|dict|enumerate|float|heapify|heappop|int|len|list|max|min|range|set|sorted|sum'
    : 'Array|Boolean|Map|Math|Number|Object|Promise|Set|String|console';
  const tokenPattern = new RegExp(
    `(#.*|//.*|"""[\\s\\S]*?"""|'''[\\s\\S]*?'''|"(?:\\\\.|[^"\\\\])*"|'(?:\\\\.|[^'\\\\])*'|\\b(?:${keywordPattern})\\b|\\b(?:${builtinPattern})\\b|\\b\\d+(?:\\.\\d+)?\\b)`,
    'g',
  );

  const tokens = [];
  let cursor = 0;
  for (const match of code.matchAll(tokenPattern)) {
    if (match.index > cursor) {
      tokens.push({ type: 'text', value: code.slice(cursor, match.index) });
    }

    const value = match[0];
    let type = 'number';
    if (value.startsWith('#') || value.startsWith('//')) {
      type = 'comment';
    } else if (value.startsWith('"') || value.startsWith("'")) {
      type = 'string';
    } else if (new RegExp(`^(?:${keywordPattern})$`).test(value)) {
      type = 'keyword';
    } else if (new RegExp(`^(?:${builtinPattern})$`).test(value)) {
      type = 'builtin';
    }

    tokens.push({ type, value });
    cursor = match.index + value.length;
  }

  if (cursor < code.length) {
    tokens.push({ type: 'text', value: code.slice(cursor) });
  }

  return tokens;
}

const legacyRoutes = {
  'SystemDesign05 Interview Flow.md': 'SystemDesign00 Overview.md',
  'SystemDesign06 Photo Sharing Feed.md': 'SystemDesign07 Photo Sharing Feed.md',
  'SystemDesign07 Async Messaging Systems.md': 'SystemDesign06 Async Messaging Systems.md',
  'CoreSkills09 Design Segment Tree.md': 'CoreSkills08 Design Segment Tree.md',
  'CoreSkills10 Insertion Sort.md': 'CoreSkills09 Insertion Sort.md',
  'CoreSkills21 Decode Ways Dynamic Programming.md': 'CoreSkills10 Decode Ways Dynamic Programming.md',
  'CoreSkills22 Rejection Sampling Rand10.md': 'CoreSkills11 Rejection Sampling Rand10.md',
  'CoreSkills23 Dynamic Programming Patterns.md': 'CoreSkills10 Decode Ways Dynamic Programming.md',
  'CoreSkills23 Greedy Algorithms.md': 'CoreSkills12 Greedy Algorithms.md',
  'CoreSkills24 Interval Problems.md': 'CoreSkills13 Interval Problems.md',
  'CoreSkills25 Math Binary Exponentiation.md': 'CoreSkills14 Math Binary Exponentiation.md',
  'CoreSkills26 Bit Manipulation XOR.md': 'CoreSkills15 Bit Manipulation XOR.md',
  'CoreSkills27 String Basics Encode Decode.md': 'CoreSkills16 String Basics Encode Decode.md',
  'CoreSkills28 Two Pointers.md': 'CoreSkills17 Two Pointers.md',
  'CoreSkills29 Sliding Window.md': 'CoreSkills18 Sliding Window.md',
  'CoreSkills30 Stack MinStack Monotonic Stack.md': 'CoreSkills19 Stack MinStack Monotonic Stack.md',
  'CoreSkills31 Binary Search.md': 'CoreSkills20 Binary Search.md',
  'CoreSkills32 Design Trie.md': 'CoreSkills21 Design Trie.md',
  'CoreSkills33 Backtracking.md': 'CoreSkills22 Backtracking.md',
};

function parseHashRoute(rawHash) {
  const hashValue = decodeURIComponent(String(rawHash ?? '').replace(/^#/, '')).replace(/^\/+/, '');

  if (!hashValue || hashValue === 'home') {
    return { view: 'home', noteId: null, sectionId: null, headingId: null };
  }

  const [rawNoteId, ...headingParts] = hashValue.split('::');
  const headingId = headingParts.join('::') || null;
  const resolvedNoteId = legacyRoutes[rawNoteId] ?? rawNoteId;
  const noteMatch = tutorials.find((tutorial) => tutorial.id === resolvedNoteId);
  if (noteMatch) {
    return {
      view: 'reader',
      noteId: noteMatch.id,
      sectionId: noteMatch.sectionId,
      headingId,
    };
  }

  const sectionMatch = noteSections.find((section) => section.id === rawNoteId);
  if (sectionMatch) {
    return {
      view: 'reader',
      noteId: sectionMatch.notes[0]?.id ?? null,
      sectionId: sectionMatch.id,
      headingId: null,
    };
  }

  return null;
}

function replaceObsidianHighlights(markdownText) {
  const codeSegments = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g;

  return markdownText
    .split(codeSegments)
    .map((segment, index) => (
      index % 2 === 1
        ? segment
        : segment.replace(/==([^=\n][^=\n]*?)==/g, '<mark>$1</mark>')
    ))
    .join('');
}

function normalizeObsidianMarkdown(markdownText) {
  if (!markdownText) {
    return '';
  }

  let normalized = markdownText;

  normalized = normalized.replace(/%%[\s\S]*?%%/g, '');

  normalized = normalized.replace(/^>\s*\[!([^\]\n+-]+)(?:[+-])?\](.*)$/gim, (_, type, rawTitle) => {
    const label = type.trim();
    const title = rawTitle.trim().replace(/^[-:\s]+/, '');
    const heading = title || (label.charAt(0).toUpperCase() + label.slice(1).toLowerCase());
    return `> **${heading}:**`;
  });

  normalized = normalized.replace(/!\[\[([^\]\n]+)\]\]/g, (_, body) => {
    const { target, alias } = splitObsidianTarget(body);
    if (!target) {
      return '';
    }

    const mediaUrl = resolveMediaUrl(target);
    if (mediaUrl) {
      return `![${alias || prettyLabel(target)}](${mediaUrl})`;
    }

    const noteId = resolveNoteId(target);
    if (noteId) {
      return `[Embedded note: ${alias || prettyLabel(target)}](#${encodeURIComponent(noteId)})`;
    }

    return `*Embedded asset not found: ${alias || prettyLabel(target)}*`;
  });

  normalized = normalized.replace(/\[\[([^\]\n]+)\]\]/g, (_, body) => {
    const { target, alias } = splitObsidianTarget(body);
    if (!target) {
      return '';
    }

    const resolvedLink = resolveObsidianLink(target, alias);
    if (resolvedLink) {
      return resolvedLink;
    }

    if (/^https?:\/\//i.test(target)) {
      return `[${alias || target}](${target})`;
    }

    return alias || prettyLabel(target);
  });

  normalized = replaceObsidianHighlights(normalized);

  return normalized;
}

function App() {
  const initialRoute = parseHashRoute(window.location.hash) ?? { view: 'home', noteId: null, sectionId: null };
  const initialId = initialRoute.noteId ?? tutorials[0]?.id ?? '';

  const [currentView, setCurrentView] = useState(initialRoute.view);
  const [selectedTutorialId, setSelectedTutorialId] = useState(initialId);
  const [pendingHeadingId, setPendingHeadingId] = useState(initialRoute.headingId ?? null);
  const [language, setLanguage] = useState('zh');
  const [query, setQuery] = useState('');
  const [contentByKey, setContentByKey] = useState({});
  const [errorByKey, setErrorByKey] = useState({});
  const inFlightRef = useRef(new Set());
  const selectedSection = noteSections.find((section) =>
    section.notes.some((note) => note.id === selectedTutorialId),
  ) ?? noteSections[0];
  const localizedHome = homeCopy[language] ?? homeCopy.zh;
  const localizeHomeSection = (section) => (
    homeSectionCopy[language]?.[section.id]
    ?? homeSectionCopy.zh[section.id]
    ?? { title: section.title, description: section.description }
  );
  const activeSectionNotes = tutorials.filter((tutorial) => tutorial.sectionId === selectedSection?.id);

  const filteredTutorials = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return activeSectionNotes;
    }

    return activeSectionNotes.filter((tutorial) =>
      [tutorial.title, tutorial.titleEn, tutorial.fileName, tutorial.enFileName]
        .filter(Boolean)
        .some((field) => field.toLowerCase().includes(normalizedQuery)),
    );
  }, [activeSectionNotes, query]);

  const selectedTutorial =
    tutorials.find((tutorial) => tutorial.id === selectedTutorialId) ?? filteredTutorials[0] ?? tutorials[0] ?? null;

  const activeLanguage =
    variantHasContent(selectedTutorial?.variants[language]) ? language : 'zh';
  const localizedSelectedSection = (
    homeSectionCopy[activeLanguage]?.[selectedSection?.id]
    ?? homeSectionCopy.zh[selectedSection?.id]
    ?? { title: selectedSection?.title ?? 'Notes', description: selectedSection?.description ?? '' }
  );
  const selectedVariant = selectedTutorial?.variants[activeLanguage] ?? null;
  const contentKey =
    selectedTutorial && selectedVariant ? `${selectedTutorial.id}:${activeLanguage}` : '';

  useEffect(() => {
    if (!selectedVariant?.url || selectedVariant?.content !== undefined || !contentKey) {
      return;
    }

    const isLoaded = Object.prototype.hasOwnProperty.call(contentByKey, contentKey);
    if (isLoaded || errorByKey[contentKey] || inFlightRef.current.has(contentKey)) {
      return;
    }

    inFlightRef.current.add(contentKey);

    fetch(selectedVariant.url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load markdown (${response.status})`);
        }
        return response.text();
      })
      .then((content) => {
        setContentByKey((prev) => ({
          ...prev,
          [contentKey]: content,
        }));
      })
      .catch((error) => {
        setErrorByKey((prev) => ({
          ...prev,
          [contentKey]: error.message,
        }));
      })
      .finally(() => {
        inFlightRef.current.delete(contentKey);
      });
  }, [contentByKey, contentKey, errorByKey, selectedVariant]);

  const selectedInlineContent = selectedVariant?.content;
  const hasSelectedContent = selectedInlineContent !== undefined || (contentKey
    ? Object.prototype.hasOwnProperty.call(contentByKey, contentKey)
    : false);
  const selectedContent = selectedInlineContent !== undefined
    ? selectedInlineContent
    : hasSelectedContent
      ? contentByKey[contentKey]
      : '';
  const selectedError = contentKey ? errorByKey[contentKey] : '';
  const selectedIsLoading = Boolean(
    selectedTutorial && selectedVariant?.url && selectedInlineContent === undefined && !hasSelectedContent && !selectedError,
  );

  const normalizedSelectedContent = useMemo(
    () => normalizeObsidianMarkdown(selectedContent),
    [selectedContent],
  );
  const sectionHeadings = useMemo(
    () => extractMarkdownHeadings(normalizedSelectedContent).filter((heading) => heading.level <= 3),
    [normalizedSelectedContent],
  );

  const scrollToHeading = (headingId) => {
    const target = document.getElementById(headingId);
    target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const navigateHome = () => {
    setCurrentView('home');
    setQuery('');
    setPendingHeadingId(null);
    if (window.location.hash) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    }
  };

  const navigateToSection = (sectionId) => {
    const section = noteSections.find((candidate) => candidate.id === sectionId);
    const nextId = section?.notes[0]?.id ?? tutorials[0]?.id ?? '';
    setCurrentView('reader');
    setQuery('');
    setPendingHeadingId(null);
    setSelectedTutorialId(nextId);
  };

  const navigateToTutorial = (tutorialId) => {
    setCurrentView('reader');
    setPendingHeadingId(null);
    setSelectedTutorialId(tutorialId);
  };

  const navigateToAbout = () => {
    setCurrentView('home');
    setQuery('');
    if (window.location.hash) {
      window.history.replaceState(null, '', window.location.pathname + window.location.search);
    }
    window.requestAnimationFrame(() => {
      document.getElementById('about')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  };

  useEffect(() => {
    if (currentView !== 'reader' || !selectedTutorial) {
      return;
    }

    const encoded = `#${encodeURIComponent(selectedTutorial.id)}`;
    if (window.location.hash !== encoded) {
      window.history.replaceState(null, '', encoded);
    }
  }, [currentView, selectedTutorial]);

  useEffect(() => {
    const handleHashChange = () => {
      const route = parseHashRoute(window.location.hash);
      if (!route) {
        return;
      }

      setCurrentView(route.view);
      if (route.noteId) {
        setSelectedTutorialId(route.noteId);
      }
      setPendingHeadingId(route.headingId ?? null);
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    if (!pendingHeadingId || selectedIsLoading || selectedError || !normalizedSelectedContent.trim()) {
      return undefined;
    }

    const frameId = window.requestAnimationFrame(() => {
      document.getElementById(pendingHeadingId)?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
      setPendingHeadingId(null);
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [normalizedSelectedContent, pendingHeadingId, selectedError, selectedIsLoading]);

  return (
    <div className={`site-shell ${currentView === 'home' ? 'home-view' : 'reader-view'}`}>
      <header className="top-nav">
        <button className="brand-lockup" type="button" onClick={navigateHome}>
          <span className="brand-mark">IN</span>
          <span>
            <strong>Interview Notes</strong>
            <small>{localizedHome.brandSubtitle}</small>
          </span>
        </button>

        <nav className="top-nav-links" aria-label={localizedHome.mainNavigation}>
          <button
            className={`top-nav-link ${currentView === 'home' ? 'active' : ''}`}
            type="button"
            onClick={navigateHome}
          >
            {localizedHome.home}
          </button>
          {noteSections.map((section) => {
            const sectionCopy = localizeHomeSection(section);
            return (
              <button
                key={section.id}
                className={`top-nav-link ${currentView === 'reader' && selectedSection?.id === section.id ? 'active' : ''}`}
                type="button"
                onClick={() => navigateToSection(section.id)}
              >
                {sectionCopy.title}
              </button>
            );
          })}
          <button
            className="top-nav-link"
            type="button"
            onClick={navigateToAbout}
          >
            {localizedHome.about}
          </button>
        </nav>
      </header>

      {currentView === 'home' ? (
        <main className="home-page">
          <section className="home-hero">
            <div className="home-hero-copy">
              <div className="home-hero-kicker">
                <p className="eyebrow">{localizedHome.heroEyebrow}</p>
                <div
                  className="language-toggle home-language-toggle"
                  aria-label={localizedHome.languageSelector}
                  role="group"
                >
                  {languageOptions.map((option) => (
                    <button
                      key={option.id}
                      className={`language-button ${language === option.id ? 'active' : ''}`}
                      onClick={() => setLanguage(option.id)}
                      type="button"
                      aria-pressed={language === option.id}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
              <h1>{localizedHome.heroTitle}</h1>
            </div>

            <div className="home-hero-panel" aria-label={localizedHome.siteSummary}>
              {homeStats.map((stat) => (
                <div className="home-stat" key={stat.id}>
                  <strong>{stat.value}</strong>
                  <span>{localizedHome.stats[stat.id]}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="home-sections" aria-label={localizedHome.sectionsAria}>
            <div className="section-heading">
              <p className="eyebrow">{localizedHome.sectionsEyebrow}</p>
              <h2>{localizedHome.sectionsHeading}</h2>
            </div>

            <div className="section-card-grid">
              {noteSections.map((section) => {
                const sectionCopy = localizeHomeSection(section);
                return (
                  <button
                    key={section.id}
                    className="home-section-card"
                    type="button"
                    onClick={() => navigateToSection(section.id)}
                  >
                    <span className="section-card-kicker">{localizedHome.noteCount(section.notes.length)}</span>
                    <strong>{sectionCopy.title}</strong>
                    <span>{sectionCopy.description}</span>
                  </button>
                );
              })}
            </div>
          </section>

          <section className="home-about" id="about" aria-labelledby="about-title">
            <div className="section-heading">
              <p className="eyebrow">{localizedHome.aboutEyebrow}</p>
              <h2 id="about-title">{localizedHome.aboutHeading}</h2>
            </div>

            <div className="about-panel">
              <div className="about-copy">
                {localizedHome.aboutBody.map((paragraph) => (
                  <p key={paragraph}>{paragraph}</p>
                ))}
              </div>

              <div className="about-links" aria-label={localizedHome.contactLinks}>
                {authorLinks.map((link) => (
                  <a
                    className="about-link"
                    href={link.href}
                    key={link.label}
                    target={link.href.startsWith('http') ? '_blank' : undefined}
                    rel={link.href.startsWith('http') ? 'noreferrer' : undefined}
                  >
                    <span>{link.label}</span>
                    <strong>{link.value}</strong>
                  </a>
                ))}
              </div>
            </div>
          </section>
        </main>
      ) : (
        <div className="app-shell">
      <aside className="notes-panel">
        <header className="panel-header">
          <p className="eyebrow">{language === 'en' ? 'Current section' : '当前板块'}</p>
          <h1>{localizedSelectedSection.title}</h1>
          <p className="panel-meta">
            {language === 'en'
              ? `${activeSectionNotes.length} notes in this section`
              : `本板块共 ${activeSectionNotes.length} 篇笔记`}
          </p>
          {localizedSelectedSection.description && (
            <p className="panel-description">{localizedSelectedSection.description}</p>
          )}
        </header>

        <label className="search">
          <span>{language === 'en' ? 'Search' : '搜索'} {localizedSelectedSection.title}</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={language === 'en' ? 'Type a note title or filename' : '输入笔记标题或文件名'}
          />
        </label>

        <div className="note-list">
          {filteredTutorials.map((tutorial) => (
            <button
              key={tutorial.id}
              className={`note-button ${selectedTutorial?.id === tutorial.id ? 'active' : ''}`}
              onClick={() => navigateToTutorial(tutorial.id)}
              type="button"
            >
              <span className="note-title">
                {language === 'en' ? (tutorial.titleEn ?? tutorial.title) : tutorial.title}
              </span>
              <span className="note-subtitle">
                {language === 'en' ? (tutorial.enFileName || tutorial.fileName) : tutorial.zhFileName}
              </span>
            </button>
          ))}
          {filteredTutorials.length === 0 && (
            <p className="list-empty">{language === 'en' ? 'No notes matched your search.' : '没有匹配的笔记。'}</p>
          )}
        </div>
      </aside>

      <main className="reader-panel">
        {selectedTutorial ? (
          <>
            <header className="reader-header">
              <div className="reader-header-top">
                <div>
                  <p className="reader-label">{localizedSelectedSection.title} / {language === 'en' ? 'Interview Notes' : '面试笔记'}</p>
                  <h2>
                    {language === 'en'
                      ? (selectedTutorial.titleEn ?? selectedTutorial.title)
                      : selectedTutorial.title}
                  </h2>
                  <p>{language === 'en' ? (selectedTutorial.enFileName || selectedTutorial.fileName) : selectedTutorial.zhFileName}</p>
                </div>

                <div className="reader-controls">
                  <div className="language-toggle" aria-label={language === 'en' ? 'Language selector' : '语言选择'} role="group">
                    {languageOptions.map((option) => (
                      <button
                        key={option.id}
                        className={`language-button ${language === option.id ? 'active' : ''}`}
                        onClick={() => setLanguage(option.id)}
                        type="button"
                        aria-pressed={language === option.id}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </header>

            <div className="reader-content-grid">
              <article className="markdown-body">
                {selectedError && <p className="empty-note">{activeLanguage === 'en' ? 'Load failed' : '加载失败'}: {selectedError}</p>}
                {selectedIsLoading && !selectedError && <p className="empty-note">{activeLanguage === 'en' ? 'Loading markdown...' : '正在加载 Markdown…'}</p>}
                {!selectedIsLoading && !selectedError && normalizedSelectedContent?.trim() && (
                  <UiLanguageContext.Provider value={activeLanguage}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeRaw, rehypeKatex]}
                    components={{
                      a: ({ href, children, ...props }) => {
                        const external = href?.startsWith('http');
                        return (
                          <a
                            href={href}
                            target={external ? '_blank' : undefined}
                            rel={external ? 'noreferrer' : undefined}
                            {...props}
                          >
                            {children}
                          </a>
                        );
                      },
                      h1: ({ children }) => <HeadingWithAnchor level={1}>{children}</HeadingWithAnchor>,
                      h2: ({ children }) => <HeadingWithAnchor level={2}>{children}</HeadingWithAnchor>,
                      h3: ({ children }) => <HeadingWithAnchor level={3}>{children}</HeadingWithAnchor>,
                      h4: ({ children }) => <HeadingWithAnchor level={4}>{children}</HeadingWithAnchor>,
                      pre: MarkdownPre,
                      code: ({ className, children, ...props }) => (
                        <code className={className} {...props}>
                          {children}
                        </code>
                      ),
                    }}
                  >
                    {normalizedSelectedContent}
                  </ReactMarkdown>
                  </UiLanguageContext.Provider>
                )}
                {!selectedIsLoading && !selectedError && selectedContent !== undefined && !selectedContent.trim() && (
                  <p className="empty-note">{activeLanguage === 'en' ? 'This file is empty and ready for future notes.' : '这个文件暂时为空，可以继续补充笔记。'}</p>
                )}
              </article>

              {sectionHeadings.length > 0 && (
                <aside className="section-toc" aria-label={activeLanguage === 'en' ? 'Section navigation' : '章节导航'}>
                  <div className="section-toc-inner">
                    <div className="section-toc-heading">
                      <p className="eyebrow">{activeLanguage === 'en' ? 'Sections' : '本页目录'}</p>
                      <span>{sectionHeadings.length}</span>
                    </div>
                    <nav>
                      {sectionHeadings.map((heading, index) => (
                        <a
                          className={`toc-link level-${heading.level}`}
                          href={`#${heading.id}`}
                          key={`${heading.id}-${index}`}
                          onClick={(event) => {
                            event.preventDefault();
                            scrollToHeading(heading.id);
                          }}
                        >
                          {heading.text}
                        </a>
                      ))}
                    </nav>
                  </div>
                </aside>
              )}
            </div>
          </>
        ) : (
          <section className="reader-empty">
            <h2>{activeLanguage === 'en' ? 'No published Markdown files found' : '没有找到已发布的 Markdown 文件'}</h2>
            <p>{activeLanguage === 'en' ? 'Add ready notes to an interview section and refresh.' : '把整理好的笔记加入对应板块后刷新页面。'}</p>
          </section>
        )}
      </main>
    </div>
      )}
    </div>
  );
}

export default App;

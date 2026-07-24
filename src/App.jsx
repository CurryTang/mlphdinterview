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
  createTutorialDefinition('MLSYS1 · GPU 体系结构入门', 'MLSYS1.md', 'MLSYS1.en.md'),
  createTutorialDefinition('MLSYS2 · CUDA 编程模型与 GPU 组件', 'MLSYS2.md', 'MLSYS2.en.md'),
  createTutorialDefinition('MLSYS3 · Roofline Analysis', 'MLSYS3.md', 'MLSYS3.en.md'),
  createTutorialDefinition('MLSYS4 · CUDA Reduce Kernel 完全指南', 'MLSYS4.md', 'MLSYS4.en.md'),
  createTutorialDefinition('MLSYS5 · CUDA Parallel Primitives: Histogram & Scan', 'MLSYS5.md', 'MLSYS5.en.md'),
  createTutorialDefinition('MLSYS6 · Memory-Bound Kernel 优化', 'MLSYS6.md', 'MLSYS6.en.md'),
  createTutorialDefinition(
    'MLSYS7 · Compute-Bound Kernel (1)',
    'MLSYS7 Compute-Bound Kernel (1).md',
    'MLSYS7 Compute-Bound Kernel (1).en.md',
  ),
  createTutorialDefinition(
    'MLSYS8 · Compute-Bound Kernel (2)',
    'MLSYS8 Compute-Bound Kernel (2).md',
    'MLSYS8 Compute-Bound Kernel (2).en.md',
  ),
  createTutorialDefinition(
    'MLSYS9 · Compute-Bound Kernel (3)',
    'MLSYS9 Compute-bound kernel (3).md',
    'MLSYS9 Compute-bound kernel (3).en.md',
  ),
  createTutorialDefinition('MLSYS10 · 分布式训练并行范式', 'MLSYS10 parallelism.md', 'MLSYS10 parallelism.en.md'),
  createTutorialDefinition('MLSYS11 · nano-vllm 精读 (1)', 'MLSYS11 nano-vllm-1.md', 'MLSYS11 nano-vllm-1.en.md'),
  createTutorialDefinition('MLSYS12 · nano-vllm 精读 (2)', 'MLSYS12 nano-vllm-2.md', 'MLSYS12 nano-vllm-2.en.md'),
  createTutorialDefinition(
    'MLSYS13 · Low-bit Quantization 核心方法详解',
    'MLSYS13 Quantization and precision.md',
    'MLSYS13 Quantization and precision.en.md',
  ),
  createTutorialDefinition(
    'MLSYS14 · Post-Training Infra：从 TRL 到 Forge',
    'MLSYS14 Post-Training Infra.md',
    'MLSYS14 Post-Training Infra.en.md',
  ),
  createTutorialDefinition(
    'MLSYS15 · Efficient Attention：现代长上下文架构',
    'MLSYS15 Efficient Attention Modern Architectures.md',
    null,
  ),
  createTutorialDefinition(
    'MLSYS16 · KV Cache：内存管理与前缀复用',
    'MLSYS15 KV Cache Prefix Caching IndexShare.md',
    null,
  ),
  createTutorialDefinition(
    'MLSYS17 · Inference：并行解码与草稿验证',
    'MLSYS15 LLM Inference Speculative Decoding DFlash.md',
    null,
  ),
  createTutorialDefinition(
    'MLSYS18 · MoE Systems：路由、通信与 Kernel',
    'MLSYS16 Modern MoE SonicMoE.md',
    null,
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
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 2 · Design Singly Linked List',
    'CoreSkills02 Design Singly Linked List.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 3 · Design Double-ended Queue',
    'CoreSkills03 Design Double-ended Queue.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 4 · Design Binary Search Tree',
    'CoreSkills04 Design Binary Search Tree.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 5 · Design Hash Table',
    'CoreSkills05 Design Hash Table.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 6 · Design Heap',
    'CoreSkills06 Design Heap.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 7 · Design Graph',
    'CoreSkills07 Design Graph.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 8 · Design Disjoint Set',
    'CoreSkills08 Design Disjoint Set Union Find.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 9 · Design Segment Tree',
    'CoreSkills09 Design Segment Tree.md',
    null,
    { directory: 'Leetcode', category: 'Implement Data Structures', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 10 · Insertion Sort',
    'CoreSkills10 Insertion Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 11 · Merge Sort',
    'CoreSkills11 Merge Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 12 · Quick Sort',
    'CoreSkills12 Quick Sort.md',
    null,
    { directory: 'Leetcode', category: 'Sorting', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 13 · Matrix DFS',
    'CoreSkills13 Matrix Depth First Search.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 14 · Matrix BFS',
    'CoreSkills14 Matrix Breadth First Search.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    "Core Skills 15 · Shortest Path: Dijkstra & Bellman-Ford",
    'CoreSkills15 Dijkstra Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    "Core Skills 16 · Prim's Algorithm",
    'CoreSkills16 Prim Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    "Core Skills 17 · Kruskal's Algorithm",
    'CoreSkills17 Kruskal Algorithm.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 18 · Topological Sort / Foreign Dictionary',
    'CoreSkills18 Topological Sort.md',
    null,
    { directory: 'Leetcode', category: 'Graphs', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Core Skills 19 · 0 / 1 Knapsack',
    'CoreSkills19 0-1 Knapsack.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 20 · Unbounded Knapsack',
    'CoreSkills20 Unbounded Knapsack.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 21 · Dynamic Programming',
    'CoreSkills21 Decode Ways Dynamic Programming.md',
    null,
    { directory: 'Leetcode', category: 'Dynamic Programming', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 22 · Rejection Sampling / Rand10',
    'CoreSkills22 Rejection Sampling Rand10.md',
    null,
    { directory: 'Leetcode', category: 'Math & Probability', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 23 · Greedy Algorithms',
    'CoreSkills23 Greedy Algorithms.md',
    null,
    { directory: 'Leetcode', category: 'Greedy', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 24 · Interval Problems',
    'CoreSkills24 Interval Problems.md',
    null,
    { directory: 'Leetcode', category: 'Intervals', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 25 · Math: Fast Power',
    'CoreSkills25 Math Binary Exponentiation.md',
    null,
    { directory: 'Leetcode', category: 'Math', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 26 · Bit Manipulation: XOR',
    'CoreSkills26 Bit Manipulation XOR.md',
    null,
    { directory: 'Leetcode', category: 'Math', difficulty: 'Easy' },
  ),
  createTutorialDefinition(
    'Core Skills 27 · String Basics',
    'CoreSkills27 String Basics Encode Decode.md',
    null,
    { directory: 'Leetcode', category: 'Strings', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 28 · Two Pointers',
    'CoreSkills28 Two Pointers.md',
    null,
    { directory: 'Leetcode', category: 'Two Pointers', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Core Skills 29 · Sliding Window',
    'CoreSkills29 Sliding Window.md',
    null,
    { directory: 'Leetcode', category: 'Sliding Window', difficulty: 'Medium' },
  ),
];

const leetcodeNotes = leetcodeNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const llmNoteDefinitions = [
  createTutorialDefinition(
    'LLM八股 1 · World Model',
    'LLMInterview01 World Model.md',
    null,
    { directory: 'LLMInterview', category: 'Models & Agents', difficulty: 'Placeholder' },
  ),
  createTutorialDefinition(
    '强化学习练习',
    'MLSYS15 RL Infra 自测 35 问.md',
    'MLSYS15 RL Infra 自测 35 问.en.md',
  ),
];

const llmNotes = llmNoteDefinitions.map((definition) => ({
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
      ),
      createDraftTutorialDefinition(
        'Quant 草稿 · 概率基础公式与记忆框架',
        'Draft Probability Basics.md',
        probabilityDraftContent,
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
    'Quant 1 · 期望与计数：Indicator、Records 与 Multinomial',
    'Quant01 Expectation Counting Multinomial.md',
    null,
    { directory: 'quant', category: 'Expectation', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Quant 2 · Markov Chains：状态压缩与期望时间',
    'Quant02 Markov Chains Expected Time.md',
    null,
    { directory: 'quant', category: 'Markov', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Quant 3 · 连续分布：CDF、几何区域与变量变换',
    'Quant03 Continuous Distribution Geometry Transform.md',
    null,
    { directory: 'quant', category: 'Distribution', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Quant 4 · 协方差、相关系数与相关矩阵 PSD',
    'Quant04 Correlation Matrix PSD.md',
    null,
    { directory: 'quant', category: 'Probability', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Quant 5 · 正态分布：二维正态、Cholesky 与符号相关',
    'Quant05 Normal Sign Correlation.md',
    null,
    { directory: 'quant', category: 'Normal Distribution', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'Quant 6 · 高维积分：大数定律与控制收敛',
    'Quant06 High Dimensional Integral Dominated Convergence.md',
    null,
    { directory: 'quant', category: 'Analysis & Probability', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'Quant 7 · 递推法：健忘乘客与状态压缩',
    'Quant07 Recursion Absent-Minded Passenger.md',
    null,
    { directory: 'quant', category: 'Recursion', difficulty: 'Medium' },
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
    'ML Coding 1 · Unicode & Pretokenization',
    'MLCoding01 Unicode Pretokenization.md',
    null,
    { directory: 'MLCoding', category: 'Tokenizer', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'ML Coding 2 · BPE Training',
    'MLCoding02 BPE Training.md',
    null,
    { directory: 'MLCoding', category: 'Tokenizer', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 3 · Tokenizer Runtime',
    'MLCoding03 Tokenizer Runtime.md',
    null,
    { directory: 'MLCoding', category: 'Tokenizer', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 4 · Tensor Modules',
    'MLCoding04 Tensor Modules.md',
    null,
    { directory: 'MLCoding', category: 'Transformer LM', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'ML Coding 5 · Attention & Transformer',
    'MLCoding05 Attention Transformer.md',
    null,
    { directory: 'MLCoding', category: 'Transformer LM', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 6 · Training Components',
    'MLCoding06 Training Components.md',
    null,
    { directory: 'MLCoding', category: 'Training', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 7 · Training Loop & Generation',
    'MLCoding07 Training Loop Generation.md',
    null,
    { directory: 'MLCoding', category: 'Training', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'ML Coding 8 · Experiments & Ablations',
    'MLCoding08 Experiments Ablations.md',
    null,
    { directory: 'MLCoding', category: 'Experiments', difficulty: 'Hard' },
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
    null,
    { directory: 'SystemDesign', category: 'Overview', difficulty: 'Intro' },
  ),
  createTutorialDefinition(
    'System Design 01 · 无状态设计范式',
    'SystemDesign01 Stateless Service.md',
    null,
    { directory: 'SystemDesign', category: 'Design Pattern', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 01B · 虚拟化与容器',
    'SystemDesign01B Virtualization Containers.md',
    null,
    { directory: 'SystemDesign', category: 'Compute Isolation', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 02 · 数据库基本范式',
    'SystemDesign02 Database Paradigms.md',
    null,
    { directory: 'SystemDesign', category: 'Database', difficulty: 'Medium' },
  ),
  {
    id: 'SystemDesign03 Database Scaling.md',
    title: 'System Design 03 · 数据库扩展三件套',
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
    null,
    { directory: 'SystemDesign', category: 'Storage', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 05 · 可靠性与复制',
    'SystemDesign05 Reliability Replication.md',
    null,
    { directory: 'SystemDesign', category: 'Reliability', difficulty: 'Medium' },
  ),
  createTutorialDefinition(
    'System Design 06 · 异步消息系统',
    'SystemDesign06 Async Messaging Systems.md',
    null,
    { directory: 'SystemDesign', category: 'Messaging', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'System Design 07 · 图片分享与 Feed',
    'SystemDesign07 Photo Sharing Feed.md',
    null,
    { directory: 'SystemDesign', category: 'Case Study', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'System Design 08 · 异步 LLM RL 训练平台',
    'SystemDesign08 LLM Async RL Platform.md',
    null,
    { directory: 'SystemDesign', category: 'ML Infrastructure', difficulty: 'Hard' },
  ),
  createTutorialDefinition(
    'System Design 09 · 一致性哈希',
    'SystemDesign09 Consistent Hashing.md',
    null,
    { directory: 'SystemDesign', category: 'Distributed Systems', difficulty: 'Medium' },
  ),
  // Keep the glossary as the final System Design note even when new chapters are inserted.
  createTutorialDefinition(
    'System Design 99 · 高频术语整合',
    'SystemDesign99 Glossary.md',
    null,
    { directory: 'SystemDesign', category: 'Glossary', difficulty: 'Reference' },
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
  createTutorialDefinition(
    '附录 · 王树森课程覆盖索引',
    'BusinessAlgorithm10 Course Coverage.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Coverage', difficulty: 'Audit' },
  ),
  createTutorialDefinition(
    '附录 · 公式速查',
    'BusinessAlgorithm08 Formula Review.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Review', difficulty: 'Reference' },
  ),
  createTutorialDefinition(
    '附录 · Quick Coding',
    'BusinessAlgorithm09 Quick Coding.md',
    null,
    { directory: 'BusinessAlgorithm', category: 'Practice', difficulty: '8 Problems' },
  ),
];

const businessAlgorithmNotes = businessAlgorithmNoteDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName, definition.directory),
    en: createVariant(definition.enFileName, definition.directory),
  },
}));

const mlInterviewNoteDefinitions = [
  createTutorialDefinition(
    'ML八股 · TODO',
    'ML Interview TODO.md',
    null,
    { directory: 'MLInterview', category: 'TODO', difficulty: 'TODO' },
  ),
];

const mlInterviewNotes = mlInterviewNoteDefinitions.map((definition) => ({
  ...definition,
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
    id: 'llm',
    title: 'LLM八股',
    description: 'LLM concepts, agent training, RL practice, and interview drills',
    notes: llmNotes,
  },
  {
    id: 'quant',
    title: 'Quant',
    description: 'Probability, Markov chains, expectation, and interview math drills',
    notes: quantNotes,
  },
  {
    id: 'mlcoding',
    title: 'ML Coding',
    description: 'From-scratch machine learning implementation exercises',
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
    id: 'ml-interview',
    title: 'ML八股',
    description: 'TODO: machine learning fundamentals and interview drills',
    notes: mlInterviewNotes,
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
    heroDescription:
      '这是一套供自己反复复习的 ML / LLM 技术笔记，内容包括 MLSYS、CUDA kernel、分布式训练、LLM inference、ML coding、quant 和算法练习。',
    startMlsys: '开始读 MLSYS',
    tryPractice: '做一道练习',
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
    heroDescription:
      'The notes I use to review ML systems, CUDA kernels, distributed training, LLM inference, ML coding, quant, and algorithm problems.',
    startMlsys: 'Start MLSYS',
    tryPractice: 'Try a practice problem',
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
    llm: {
      title: 'LLM八股',
      description: 'LLM 基础、Agent 训练、强化学习与面试题',
    },
    quant: {
      title: 'Quant',
      description: '概率、马尔可夫链、期望与面试数学题',
    },
    mlcoding: {
      title: 'ML Coding',
      description: '从零实现 tokenizer、attention、训练循环等机器学习组件',
    },
    'system-design': {
      title: 'System Design',
      description: '后端系统设计、LLM serving、Agent workflow 与基础设施面试题',
    },
    'business-algorithm': {
      title: '业务算法',
      description: '沿一次线上请求拆解召回、排序、列表决策、生成式方法与实验闭环',
    },
    'ml-interview': {
      title: 'ML八股',
      description: '机器学习基础与面试题，正在整理',
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
    llm: {
      title: 'LLM Interview',
      description: 'LLM fundamentals, agent training, reinforcement learning, and interview review',
    },
    quant: {
      title: 'Quant',
      description: 'Probability, Markov chains, expectation, and interview math',
    },
    mlcoding: {
      title: 'ML Coding',
      description: 'From-scratch implementations of tokenizers, attention, training loops, and other ML components',
    },
    'system-design': {
      title: 'System Design',
      description: 'Backend design, LLM serving, agent workflows, and infrastructure interviews',
    },
    'business-algorithm': {
      title: 'Business Algorithms',
      description: 'Retrieval, ranking, list decisions, generative methods, and experimentation along one production request',
    },
    'ml-interview': {
      title: 'ML Interview',
      description: 'Machine learning fundamentals and interview questions. Work in progress.',
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

function createDraftTutorialDefinition(title, zhFileName, content) {
  return {
    id: zhFileName,
    title,
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
    title: 'Narrow billions of candidates within a fixed latency budget',
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

function MarkdownPre({ children, ...props }) {
  const child = Array.isArray(children) ? children[0] : children;
  const className = child?.props?.className ?? '';
  const match = /language-(quiz|mcq|mermaid|topo-demo|bellman-demo|segment-tree-demo|interval-merge-demo|interval-insert-demo|interval-rooms-demo|interval-query-demo|pow-demo|sliding-window-demo|longest-substring-demo|sliding-window-patterns|three-sum-demo|rain-water-demo|high-dimensional-integral-demo|record-minimum-demo|message-queue-demo|business-algorithm-map|system-design-overview-visual|photo-sharing-architecture-visual|async-messaging-architecture-visual|virtualization-container-visual)/.exec(className);

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

  if (match?.[1] === 'three-sum-demo') {
    return <ThreeSumVisual />;
  }

  if (match?.[1] === 'rain-water-demo') {
    return <RainWaterVisual />;
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

const legacySystemDesignRoutes = {
  'SystemDesign05 Interview Flow.md': 'SystemDesign00 Overview.md',
  'SystemDesign06 Photo Sharing Feed.md': 'SystemDesign07 Photo Sharing Feed.md',
  'SystemDesign07 Async Messaging Systems.md': 'SystemDesign06 Async Messaging Systems.md',
};

function parseHashRoute(rawHash) {
  const hashValue = decodeURIComponent(String(rawHash ?? '').replace(/^#/, '')).replace(/^\/+/, '');

  if (!hashValue || hashValue === 'home') {
    return { view: 'home', noteId: null, sectionId: null, headingId: null };
  }

  const [rawNoteId, ...headingParts] = hashValue.split('::');
  const headingId = headingParts.join('::') || null;
  const resolvedNoteId = legacySystemDesignRoutes[rawNoteId] ?? rawNoteId;
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
      [tutorial.title, tutorial.fileName].some((field) => field.toLowerCase().includes(normalizedQuery)),
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
              <p>{localizedHome.heroDescription}</p>
              <div className="home-actions">
                <button className="primary-action" type="button" onClick={() => navigateToSection('mlsys')}>
                  {localizedHome.startMlsys}
                </button>
                <button className="secondary-action" type="button" onClick={() => navigateToTutorial('MLSYS1.md')}>
                  {localizedHome.tryPractice}
                </button>
              </div>
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
          <p className="eyebrow">{activeLanguage === 'en' ? 'Current section' : '当前板块'}</p>
          <h1>{localizedSelectedSection.title}</h1>
          <p className="panel-meta">
            {activeLanguage === 'en'
              ? `${activeSectionNotes.length} notes in this section`
              : `本板块共 ${activeSectionNotes.length} 篇笔记`}
          </p>
          {localizedSelectedSection.description && (
            <p className="panel-description">{localizedSelectedSection.description}</p>
          )}
        </header>

        <label className="search">
          <span>{activeLanguage === 'en' ? 'Search' : '搜索'} {localizedSelectedSection.title}</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={activeLanguage === 'en' ? 'Type a note title or filename' : '输入笔记标题或文件名'}
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
              <span className="note-title">{tutorial.title}</span>
              <span className="note-subtitle">{tutorial.fileName}</span>
            </button>
          ))}
          {filteredTutorials.length === 0 && (
            <p className="list-empty">{activeLanguage === 'en' ? 'No notes matched your search.' : '没有匹配的笔记。'}</p>
          )}
        </div>
      </aside>

      <main className="reader-panel">
        {selectedTutorial ? (
          <>
            <header className="reader-header">
              <div className="reader-header-top">
                <div>
                  <p className="reader-label">{localizedSelectedSection.title} / {activeLanguage === 'en' ? 'Interview Notes' : '面试笔记'}</p>
                  <h2>{selectedTutorial.title}</h2>
                  <p>{selectedVariant?.fileName ?? selectedTutorial.fileName}</p>
                </div>

                <div className="reader-controls">
                  <div className="language-toggle" aria-label={activeLanguage === 'en' ? 'Language selector' : '语言选择'} role="group">
                    {languageOptions.map((option) => (
                      <button
                        key={option.id}
                        className={`language-button ${activeLanguage === option.id ? 'active' : ''}`}
                        onClick={() => setLanguage(option.id)}
                        type="button"
                        aria-pressed={activeLanguage === option.id}
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

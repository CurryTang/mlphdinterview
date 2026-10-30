# 10 · 经典 Tech Report 精读：MAI-Thinking-1 体系解构与 Slides 笔记

大模型前沿已全面进入“预训练物理系统协同设计”与“大规模强化学习长期攀爬”的深度整合阶段。2026 年 6 月，微软人工智能团队（Microsoft AI, MAI）发布了长达 109 页的技术报告 **《MAI-Thinking-1: Building a Hill-Climbing Machine》**。该报告全面披露了一个万亿级稀疏专家混合模型（MoE）从 29.2T 纯人类原创语料预训练、Mid-Training 配比倾斜，到由 4,096 张推理卡与 768 张训练卡驱动的千步异步强化学习（RL）集群的完整工程实现细节。

本章系统解构这份经典技术报告及其配套 80 页精读 Slides，涵盖：
1. **模型架构与物理互联协同设计**：Dense/MoE 交替层级、LatentMoE 压缩投射与 $23\%$ 全互联通信量压缩。
2. **预训练消融与跨 Scale 迁移法则**：The Ladder 阶梯架构、能效增益标尺（Efficiency Gain, EG）与度量反转陷阱。
3. **真实生产训练法则**：为何真实系统 $TPP \gg 20$、消融实验中的数据重复率对齐与 Muennighoff 4 轮重复价值衰减律。
4. **数据工程与混合律理论**：29.2T 纯人类原创语料治理、Mixing Laws 混合律解析与狭窄数据饱和效应。
5. **Mid-Training 与训练 Recipe**：基座采样阈值理论、Attention 零初始化破除路由塌陷、AdamW 与 Muon 谱球几何权衡。
6. **YOLO 训练基础设施**：8,192 张 GB200 集群下 $90.0\%$ Goodput、位精确（Bit-Exact）确定性故障重放。
7. **强化学习关键技术**：Token-level GRPO 目标函数完全推导、宽剪切区间防陈旧性漂移、动态熵调控与自蒸馏崩溃自愈。
8. **分布式强化学习引擎 Rocket**：4,096 推理 + 768 训练异构池解耦、采样-训练严格数值重放、编译式 RDMA 权重下发。

---

## 幻灯片原件在线阅读与资源索引

```slide-viewer
file: slides/mai_thinking_1_v2.pdf
slidesDir: slides/mai-thinking-1
totalSlides: 80
title: MAI-Thinking-1: Building a Hill-Climbing Machine (Microsoft AI Technical Report · June 2026)
```

---

## 报告总览与三大工程核心命题

### 一、模型规格与同量级 MoE 模型全景对比

MAI-Thinking-1 基于基座模型 MAI-Base-1 构建，采用 78 层 Transformer 架构，上下文窗口为 256K tokens。其总参数量为 962B，每个 token 激活参数为 34.7B（激活比例仅为 $3.6\%$）。预训练消耗 30T tokens，Mid-Training 消耗 3.55T tokens，均在 8,192 块 NVIDIA GB200 上完成；后续异步强化学习阶段峰值部署 4,096 张推理卡 + 768 张训练卡（GB300）。

| 模型 | 发布时间 | 总参数量 | 激活参数量 | 激活比例 | 路由专家 + 共享专家 | 专家稀疏度 | MoE 层占比 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MAI-Thinking-1** | 2026.06 | **962B** | **34.7B** | **3.6%** | **8 / 512 + 0** | **64×** | **39 / 78 (50%)** |
| Kimi K2 | 2025.07 | 1T | 32B | 3.2% | 8 / 384 + 1 | 48× | 60 / 61 (98%) |
| DeepSeek-V3.2-Exp | 2025.09 | 671B | 37B | 5.5% | 8 / 256 + 1 | 32× | 58 / 61 (95%) |
| DeepSeek-V4-Pro | 2026.04 | 1.6T | 49B | 3.1% | 6 / 384 + 1 | 64× | ~ / 61 |
| Llama 4 Maverick | 2025.04 | 400B | 17B | 4.3% | 1 / 128 + 1 | 128× | 24 / 48 (50%) |
| Qwen3-235B-A22B | 2025.04 | 235B | 22B | 9.4% | 8 / 128 + 0 | 16× | 94 / 94 (100%) |

在基准评测上（Pass@1, 4 次独立采样均值），MAI-Thinking-1 展现出强劲的推理与长思考性能：
- **AIME 2025**: $97.0$（超越 Claude 3.7 Sonnet 4.6 的 $95.6$）
- **GPQA Diamond**: $84.2$（Sonnet 4.6 为 $89.9$）
- **LiveCodeBench v6 (LCB v6)**: $87.7$
- **SWE-bench Verified**: $73.5$（Sonnet 4.6 为 $79.6$）
- **SWE-bench Pro**: $52.8$
- **Terminal-Bench**: $46.0$（Sonnet 4.6 为 $59.1$）

![MAI-Thinking-1 Overview](/slides/mai-thinking-1/slide-02.png)

### 二、三大核心工程命题 (Three Core Claims)

MAI 技术报告的核心主张贯穿了整个系统研发生命周期：

![Three Claims](/slides/mai-thinking-1/slide-05.png)

1. **小模型决策，大尺度校验，采用正确的能效度量尺（Decide small, check big, with the right yardstick）**
   - 所有的预训练架构改动、数据配比与超参数调节均在小模型阶梯（Ladder）上以“能效增益”（Efficiency Gain, EG，即节省的等效算力倍率）统一定价。
   - 必须警惕小模型上的“伪信号”：部分在小型模型或每 FLOP 维度获胜的策略，在扩展至 1T 参数或以物理壁钟时间/强化学习最终效果度量时发生反转。
2. **为物理机器设计，而非为理论 FLOPs 设计（Design for the machine, not for FLOPs）**
   - 传统 MoE 架构往往过度追求 FLOPs 视角的计算效率，忽视了跨 GPU 的 All-to-All 物理互联网络瓶颈。
   - MAI 通过“密集/稀疏交替堆叠”与“分发前压缩投影”，将网络通信量缩减至同类架构的 $23\%$。通过位精确（Bit-Exact）确定性训练保证数千卡集群的快速、可校验故障重启。
3. **保持异步强化学习持续攀爬数千步而不塌陷（Keep RL alive for thousands of steps）**
   - 从未见过任何合成 Reasoning Trace 的纯净基座出发，摒弃传统的 KL 散度惩罚与 Entropy Bonus，采用宽剪切区间（$\epsilon=0.6$）、非对称上限参数 $k$、强行硬截断、采样器行为精确重放以及策略崩溃后的自蒸馏机制，支撑模型在数千步强化学习中持续登顶。

---

## 第一部分：模型架构与物理互联协同设计

### 一、MoE 运行本质与 All-to-All 通信瓶颈

在规模达万亿参数的模型中，单张 GPU 无法容纳所有专家的权重参数。采用专家并行（Expert Parallelism, EP）时，各 GPU 分别持有若干专家。当输入 Token 经过注意力计算后，Router 根据当前特征为每个 Token 选定 Top-K 个专家。由于所选专家分布在不同的 GPU 上，系统必须通过全互联通信网络进行 Token 分发（Dispatch）与结果收集（Combine）。

![EP Toy Example](/slides/mai-thinking-1/slide-08.png)

以 4 张 GPU、8 个专家、每个 Token 选取 1 个专家的最小系统为例：
1. **分发前（Before Dispatch）**：每个 GPU 持有本地序列的激活值，Router 计算各 Token 的目标专家及对应 GPU。
2. **分发阶段（All-to-All Dispatch）**：每张 GPU 向其他所有 GPU 发送特定 Token 负载，跨卡网络流量爆发。
3. **专家前向（Experts Compute）**：各 GPU 本地专家并行计算。
4. **收集阶段（All-to-All Combine）**：专家输出结果原路返回所属 GPU，加权聚合。

**物理时间痛点**：通信传输消耗的是实际壁钟时间，而非计算 FLOPs。在标准 FLOPs 统计公式中，网络通信并不计入计算量，但后续层级必须阻塞等待通信完成。当层层堆叠 MoE 时，All-to-All 通信时延迅速成为整个集群算力利用率（MFU）的核心瓶颈。

![Communication Bottleneck](/slides/mai-thinking-1/slide-09.png)

### 二、两大降通信杠杆：密集/稀疏交替与 LatentMoE 投影

为彻底打破网络互联瓶颈，MAI 提出了协同设计的两大杠杆：

![Two Levers for Communication Reduction](/slides/mai-thinking-1/slide-10.png)

1. **密集层（Dense FFN）与稀疏层（MoE FFN）交替堆叠**
   - 传统模型（如 DeepSeek-V3、Kimi K2）在几乎所有层级（58~60 层）均采用 MoE。
   - MAI 将 78 层 Transformer 划分为 39 组结构，每组由一个 Dense FFN 层和一个 MoE FFN 层交替组成（MoE 层仅占 39 层）。Dense FFN 完全在 Token 本地 GPU 运行，通信量直接为 0。仅此一项改动，全局通信量直接降低 $50\%$。
   - 为在减半 MoE 层数的同时保持 962B 总参数量，MAI 单个 MoE 层的专家数量扩大至 512 个，每个 Token 路由至 8 个专家。
2. **分发前降维投影（LatentMoE）**
   - 模型的隐藏层主维度为 $d_{\text{model}} = 6656$。如果直接分发 6656 维向量，通信负载极其沉重。
   - MAI 在进入 All-to-All 分发前，首先通过本地线性变换将 Token 向量投影压缩至 $d_{\text{latent}} = 3072$。跨 GPU 网络仅传输 3072 维向量；远端专家内部直接在 3072 维度进行中间升维计算并投影回 3072；在 Combine 返回所属 GPU 后，再通过本地变换投射回 6656 维。
   - Router 在计算专家打分时，依然基于完整的 6656 维上下文向量决策。

通信量量化测算（以单 Token 往返字节数计）：
- **基线方案（全 78 层 MoE，6656 维传输）**：$100\%$ 通信负载（$78 \times 6656$ 相对单位）。
- **引入密集层交替后**：$39 \times 6656 = 50\%$ 通信负载。
- **再叠加 LatentMoE 压缩后**：$39 \times 3072 = 23.08\%$ 通信负载。

整体通信量被压缩至基准的近四分之一（$23\%$）。此外，MAI 将 All-to-All 通信域严格限制在单个 64-GPU NVLink Rack 内部，彻底避开了跨机柜 InfiniBand/RoCE 网络的通信争抢。

### 三、消融证据：舍弃 Shared Expert

业界经典架构（DeepSeek-V3、Kimi K2）普遍引入了常驻激活的共享专家（Shared Expert）。MAI 对此进行了细致的对照实验（Table 2）：

![Ablation on MoE Layout](/slides/mai-thinking-1/slide-11.png)

- **方案 A（全层 MoE，8/384）**：在单步 FLOPs 能效与每小时训练吞吐上均落后。
- **方案 B（全层 MoE + 共享专家，7/384 + 1）**：在每 FLOP 能效上与 MAI 持平（$1.03$ 对 $1.0$），但在实际物理壁钟时间（每小时训练步数）上暴跌至 $0.82$（损失 $18\%$ 吞吐）。其根源正是全层 MoE 带来了 2 倍的跨卡通信量。
- **结论**：MAI 的交替 Dense 层本身已经充当了始终开启的共享通道，额外添加 Shared Expert 毫无增益。最终确立了 8/512 纯路由专家、无共享专家的精炼布局。

---

## 第二部分：预训练消融实验与跨 Scale 迁移法则

### 一、评估阶梯（The Ladder）与平滑信号设计

在万亿参数上直接进行全量消融是算力不可承受的。MAI 构建了名为 **The Ladder** 的 8 档评测模型矩阵，激活参数覆盖 0.37B 至 35.6B（L12、L18、L24、L30、L36、L42、L66、L78）。

![The Ladder](/slides/mai-thinking-1/slide-13.png)

1. **几何形状统一**：所有层数 $L$ 均为 6 的倍数（与 5 local + 1 global 注意力周期严格对齐）。模型宽度满足：
   $$\text{Width} = \frac{256}{3} \times L \approx 85 L$$
   且保证为 512 的整数倍。所有大小的模型在统一的 Tokens Per Parameter (TPP) 下训练，确保模型规模是沿曲线变化的唯一变量。
2. **为什么弃用 Benchmark 准确率**：在小模型阶段，大部分标准学术基准（如 MMLU、GSM8k）的得分均在随机猜测线（Chance Level）附近徘徊，两个方案之间的微小差异完全被噪声淹没（Schaeffer 2023 所指出的所谓“湧现特性”大多源于全有或全无的离散指标设计）。
3. **基于留出集 Loss 的平滑度量**：评测集由涵盖专有代码库、历史日志、学术数学与 STEM 留出题库的约 40 个 held-out loss 数据集组成，其中代码占比 $50\%$，STEM 占比 $20\%$。

### 二、能效增益标尺（Efficiency Gain, EG）

单纯比较 Loss 差值（例如 $2.31 \to 2.30$）在直觉上难以量化，且该差距随着模型增大必然收窄。MAI 将所有改进统一定价为等效算力倍率：

![Efficiency Gain](/slides/mai-thinking-1/slide-15.png)

$$\text{EG} = \frac{C_{\text{baseline}}(L)}{C_{\text{candidate}}(L)}$$

若新方案在消耗 100 单位算力下的 Loss，基线模型必须耗费 119 单位算力才能达到，则其 $\text{EG} = 1.19$（相当于提升了 $19\%$ 的等效算力）。**工程准则：唯有在 Ladder 的所有尺度下均稳定满足 $\text{EG} > 1$ 的设计，才被允许晋级。**

### 三、跨 Scale 保持与反转的设计经验

![Sparsity at Scale](/slides/mai-thinking-1/slide-16.png)

1. **成功保持有效的设计**：
   - **专家稀疏度持续扩张**：从 256 专家扩展到 512、1024 专家，EG 曲线随训练 FLOPs 的增加而持续上扬（在代码和加权综合集上表现尤甚）。最终选择 512 专家是权衡了机柜内通信带宽与部署成本。
   - **全局批次负载均衡（Global Batch Load Balancing）**：传统的辅助平衡损失 $\mathcal{L}_{\text{aux}} = N \sum_{i=1}^N f_i P_i$ 若在微批次（Micro-batch）内强制计算，会剥夺模型在局部序列中的专业化表现。MAI 改为跨全局大 Batch 统计 $f_i$，允许微批次内部出现局部倾斜。
   - **无丢弃路由（Dropless Routing）**：允许单卡容量在运行时弹性动态变化，彻底摒弃 Token Dropping，避免了截断导致的信息破损。
2. **发生反转的陷阱（Did NOT Transfer）**：
   - **数据配比排序反转**：在小模型上通过网格搜索评出的“最佳数据配比”，在大模型上 Loss 曲线发生相交反转。例如过于重仓代码的数据配比在小模型上收敛极快，但在大模型后程由于代码分布单调而迅速遭遇边际效应递减。
   - **标尺不一致陷阱**：在小模型每 FLOP 评估中获胜的超参（如学习率余弦衰减至峰值的 $0.01\times$），在最终强化学习阶段被证明会过早耗尽模型策略熵；相反，衰减至 $0.1\times$ 的模型在 RL 阶段表现出压倒性优势。

---

## 第三部分：从 Ladder 到真实大规模训练

### 一、真实工业训练中为何 $TPP \gg 20$

Chinchilla 缩放律指出，在给定计算预算下，训练 token 数量与模型参数量的最优比值约为 $TPP \approx 20$。但在工业级大模型实践中，该比值被推高至数十倍甚至上百倍：

![Why Real TPP >> 20](/slides/mai-thinking-1/slide-23.png)

- Llama 3 8B: $TPP = 1875$ (15T / 8B)
- Qwen3-235B: $TPP = 1636$ (36T / 22B active)
- **MAI-Thinking-1**: $TPP = 865$ (30T / 34.7B active)
- DeepSeek-V3: $TPP = 400$ (14.8T / 37B active)

其核心系统与业务动因包括：
1. **全生命周期 Serving 成本分摊**：Chinchilla 优化目标仅局限于模型训练的一次性算力支出，完全忽略了推理服务成本。模型部署后的推理吞吐与显存占用正比于参数量；长达数月的强化学习 Rollout 采样本质上也是庞大的推理集群服务。训练一个体积更小、但训练更充分（Over-trained）的模型，在综合财务与系统能耗上远胜过训练一个体积庞大但仅见识过 20 TPP 的脆弱模型（Sardana 2024）。
2. **Loss 优化盆地极为平缓（The Valley is Flat）**：偏离 Chinchilla 最优点对最终 Loss 的惩罚极小，而后续收益持续存在。
3. **MoE 激活参数向右偏移**：在 MoE 结构中，总参数池远大于激活参数量，计算最优的 TPP 本身自然右移至 100~200 之间。

### 二、消融实验中的数据重复率对齐陷阱

在小模型短程训练与超大规模长程训练之间存在一个隐蔽的数据陷阱：**重复轮次（Epochs）不对称**。

![Data Repetition in Ablations](/slides/mai-thinking-1/slide-24.png)

- 设有一份高质量但存量稀缺的数学数据集（1T tokens，占总混合比例的 $5\%$）。
- 在 30T 的真实生产训练中，$30\text{T} \times 5\% = 1.5\text{T}$ tokens，该数据集被反复读取了 **1.5 轮**。
- 若直接在小型 Ladder 模型上消融（例如 4B 激活模型跑 150 TPP，总计 600B tokens），该数据源仅需消费 $600\text{B} \times 5\% = 30\text{B}$ tokens，相当于仅被读取了该数据集的 **0.03 轮**（模型见到的永远是全新的新鲜样本）。
- **缺陷**：由于新鲜样本的有效学习价值远高于重复样本，常规小模型消融会严重高估高质量小数据集的边际增益，给出脱离实际的过高权重。
- **MAI 矫正策略**：在小模型 Ladder 消融中，人为将该数据集截断至 20B 样本，强制其在小模型训练中同样被重复读取 1.5 轮，从而使数据重复衰减曲线与真实大规模运行精确对齐。

### 三、重复数据衰减律（Muennighoff et al., NeurIPS 2023）

当全网高质量文本存量耗尽时，重复读取数据是否依然有效？Muennighoff 等人通过 400 余次跨规模实验给出了精确定量模型：

![Repeated Data Laws](/slides/mai-thinking-1/slide-25.png)

第 $R$ 轮重复时，数据相当于新鲜样本的有效价值衰减遵循指数规律：
$$\text{Value}(R) \approx e^{-R / 15}$$
- 在 **$\le 4$ 轮重复**内，损失函数下降效果与全新数据高度接近（保留了 $93\%$ 以上的新鲜价值）。
- 超过 16 轮后边际收益急剧收缩；达到 40 轮后几乎不再贡献泛化知识，反而引发严重的记忆与过拟合。
- **MAI 落地配置**：在 30T 预训练中，最高频的数学数据重读轮次为 **5.28 轮**，全部外部数据集严格受限于 **8 轮硬封顶**，在充分挖掘稀缺数据价值的同时彻底杜绝了模型记忆。

---

## 第四部分：预训练数据工程与混合律

### 一、语料治理：纯人类原创与全栈式去重

MAI 坚持不依赖第三方模型蒸馏或合成数据，构建了 29.2T 的纯人类原创语料库：

![Data Sources and Cleaning](/slides/mai-thinking-1/slide-27.png)

1. **多层级严厉去重**：
   - 抓取的 1.2T 网页在经过合规策略过滤后剩 794B；
   - 通过哈希计算去除完全相同副本后降至 423B；
   - 采用局部敏感哈希（MinHash）指纹对相似度达 $80\%$ 以上的近似文档聚类去重，最终保留 190B 网页（仅占初筛的 1/4）；
   - 代码语料则进一步通过 Embedding 聚类剔除功能等价的模板文件。
2. **严防基准污染**：全面过滤来自 huggingface.co 及其镜像站点的全部抓取内容；对已知公开评测集（包括 SWE-bench Verified PR、HumanEval 等）采用 20-gram 模糊重叠过滤，确保预训练阶段彻底与评测样本绝缘。

### 二、30T 混合配比设计

语料配比并非由原始存量驱动，而是严格基于留出集 Loss 优化加权：

![30T Data Mix](/slides/mai-thinking-1/slide-28.png)

- **代码（Code）**：重复 2.22 轮，占总训练量高达 **$55\%$**（而 Llama 3 中代码占比仅约 $17\%$）。
- **STEM 科学数据**：重复 2.17 轮。
- **数学（Math）**：重复 5.28 轮（全场最高重复密度）。
- **专著与期刊（Books, Journals）**：重复 1.65 轮。
- **通用网络文本（Web text）**：仅读取 0.55 轮（大部分低质网页仅浏览一次或未入选）。
- **多语言文本（Multilingual）**：仅读取 0.06 轮（高纯度筛选）。

### 三、混合律（Mixing Laws）理论建模

传统混合比例搜索往往依赖大量小规模试错。理论界（Ye et al., Fudan, ICLR 2025; RegMix）提出了混合律函数，将配比搜索转化为平滑凸优化问题：

![Mixing Laws](/slides/mai-thinking-1/slide-30.png)

$$\mathcal{L}_i(r) = c_i + k_i \exp\left( \sum_{j} t_{ij} r_j \right)$$

- $r_j$：领域 $j$ 在训练混合集中的比例权重（满足 $\sum r_j = 1$）。
- $c_i$：领域 $i$ 无法通过数据混合消除的固有不可约误差（Irreducible loss）。
- $t_{ij}$：迁移影响矩阵。当 $t_{ij} < 0$ 时，表示增加领域 $j$ 的语料有助于降低领域 $i$ 的损失（正迁移）；当 $t_{ij} > 0$ 时，表示领域 $j$ 挤占了领域 $i$ 的表征容量（负迁移）。

实验观测到强烈的交叉迁移效应：增加学术数学与科技 PDF 语料能显著降低通用物理学领域的 Loss（$t < 0$）；但若无限堆砌代码语料，在超过临界点后将对自然语言常识推理产生压制。

**专用狭窄领域（Narrow Data）快速饱和现象**：随着模型参数与训练算力总量的上升，特定小领域的边际增益曲线下降极其陡峭。这意味着在小模型上最优的配比方案，在万亿参数模型上必须根据领域饱和规律动态下调其相对份额。

---

## 第五部分：Mid-Training 阶段的设计哲学

### 一、Mid-Training 的定位与理论边界

在 30T 预训练结束后，MAI 安排了 3.55T tokens 的 Mid-Training 阶段。

![What Mid-Training Is](/slides/mai-thinking-1/slide-34.png)

1. **不引入新语料，仅做分布倾斜（Tilting the Mix）**：Mid-Training 并没有摄入任何此前未曾见过的奇异数据，而是从原有的高质量代码、数学与推理子集中重采样，将高质量领域的混合比率拉升至极高水平。
2. **核心理论命题：RL 只能强化基座模型能够采样的解空间**：
   $$\text{RL sharpens what the base can already sample.}$$
   强化学习的策略梯度更新完全依赖于模型在探索过程中命中至少一个获得正向奖励（Reward $>0$）的候选序列。如果基座模型在某个问题上的初始采样概率恒为 0（即在温度采样下尝试数千次均无法命中任何正确步骤），则该轨迹生成的 Advantage 恒为 0，RL 无法提供任何有效梯度。Mid-Training 的核心使命，正是将关键推理与复杂任务的初始生成概率拔高到采样阈值之上。
3. **上下文窗口外推**：在前 30T 预训练与 Mid-Training 前半程，序列长度严格维持在 16K，以最大化硬件利用率；仅在 Mid-Training 尾端阶段平滑拉升至 256K 完整上下文。

---

## 第六部分：优化器与训练 Recipe

### 一、AdamW 超参数配置与学习率底线

与部分社区模型全面转向新优化器不同，MAI 在万亿模型上继续坚守了深度工程验证过的 AdamW：

![Optimizer Recipe](/slides/mai-thinking-1/slide-40.png)

- **峰值学习率**：$2 \times 10^{-4}$，采用余弦衰减调度。
- **衰减底线（LR Floor）**：衰减至峰值的 **$0.1\times$**（即 $2 \times 10^{-5}$），而非业界通用的 $0.01\times$。消融结果表明，保留较高的学习率底线在预训练末期虽然使 Loss 看起来略高，但在后续 RL 阶段极大地延缓了策略熵的崩塌。
- **一阶与二阶动量系数**：$\beta_1 = 0.95, \beta_2 = 0.925$（二者更为接近，平滑了梯度估计的方差）。RL 阶段的 Learner 则进一步采用 $\beta_1 = 0.95, \beta_2 = 0.95$。
- **残差 Dropout**：在每个 Transformer block 输出并入残差连接之前，引入 $p = 0.15$ 的 Dropout。这在 PaLM、Llama 等大模型中极其罕见。实验证明，由于高价值数学语料重复读取了 5.28 轮，Dropout 显著抑制了 MoE 专家的记忆化偏置，并为后续 RL 阶段的自蒸馏保留了宝贵的特征多样性。

### 二、Attention 零初始化（Zero-Init Attention）

在万亿 MoE 架构初始启动时，MAI 观察到严重的专家负载失衡现象：最繁忙的专家负载偏差高达 **$+298\%$**，大量专家陷入饥饿。

![Zero-Init Attention](/slides/mai-thinking-1/slide-41.png)

1. **机理分析**：在初始化 Step 0 时，Attention 投影权重为随机初始高斯分布，Softmax 计算呈现平坦均匀分布。导致每个 Token 对前文各位置几乎进行等权重平均，退化为因果平均池化（Causal Mean Pooling）。每个 Token 叠加了几乎相同的上下文平均增量，导致送入深层 Router 的隐藏向量高度同质化。相似的向量必然激发相同的专家集合，造成网络早期路由崩溃。
2. **解决方案**：将 Attention 输出层 RMSNorm 的可学习增益参数（Gain）初始化为 **0**。在初始阶段，Attention 残差分支的输出恒为 0，整个网络在 Step 0 表现为纯粹的按 Token 独立运算的前馈网络，所有 Token 保持原生的差异化特征。随着训练推进，Attention 增益逐渐学习放大，平滑淡入。
3. **效果**：最繁忙专家的负载偏差从 $+298\%$ 骤降至 **$+59\%$**，模型跨尺度能效增益（EG）显著提升。

### 三、优化器的几何视界：SGD、Adam 与 Muon

针对学术界近期热议的 Muon 优化器，技术报告从受限最优化的一阶几何视角给出了统一剖析：

![Geometry of Optimizers](/slides/mai-thinking-1/slide-42.png)

更新量定义为：
$$\Delta W = \arg\min_{\Delta} \left\langle G, \Delta \right\rangle + \frac{\lambda}{2} \|\Delta\|^2$$

不同的优化器本质上是选择了不同的几何约束球（Norm Ball）：
1. **SGD（欧氏球，$\ell_2$ 范数）**：
   $$\Delta W = -G$$
   梯度绝对值大的分量剧烈更新，微小的分量几乎静止。更新方向沿最陡下降线。
2. **Adam（超立方体盒约束，$\ell_\infty$ 范数）**：
   忽略动量平滑项，Adam 的有效更新步长正比于 $-1 / \sqrt{v} \cdot g \approx -\text{sign}(g)$。更新量被约束在超立方体的各个顶点上，所有分量无论原始绝对值大小均以固定步长推进。Adam 将权重张量视作“一袋互不相干的独立数字（Bag of numbers）”。
3. **Muon（谱范数球，Spectral Ball）**：
   $$\Delta W = -U V^T$$
   其中 $G = U \Sigma V^T$ 为梯度的奇异值分解。更新量被强制约束在正交矩阵空间，使得矩阵沿所有主奇异方向具有均等的影响力。Muon 将权重张量视作一个不可分割的“线性映射算子”。

**为什么 MAI 在生产集群中维持 AdamW**：
- **微观增益随规模衰减**：Stanford 实验（Wen et al. 2025）表明，当基线 AdamW 超参经过充分精细调优后，Muon 带来的等效算力收益从 0.1B 模型的 $2.0\times$ 迅速缩水至 1.2B 模型的 $1.1\times$。
- **集群状态切分兼容性**：YOLO 训练架构深度依赖 ZeRO-2/ZeRO-3 对优化器状态进行分片（Sharding）。AdamW 的逐元素状态天生支持任意粒度分片；而 Muon 依赖全量梯度的全局 SVD / Newton-Schulz 迭代，跨节点收集与重构全矩阵会引入沉重的通信开销。
- **阶段平滑演进**：后续异步 RL 阶段采用 AdamW，避免了切换优化器引发的表征震荡。

---

## 第七部分：YOLO 训练基础设施与高 Goodput 体系

在 8,192 块 GB200 规模下连续运行数月，硬件单点故障（XID 错误、网络降速、静默内存位翻转）不可避免。MAI 打造了名为 YOLO 的训练底座，在整个 30T 预训练中实现了高达 **$90.0\%$** 的工业级有效训练时间（Goodput = 理想训练时间 / 实际壁钟时间）。

![Reliability and Goodput](/slides/mai-thinking-1/slide-45.png)

### 一、位精确（Bit-Exact）确定性训练

大部分大模型训练为了极限速度，开启了 GPU 硬件原子加（Atomic Add）以及网络交换机内部的 SHARP 聚合。这导致浮点数加法顺序受微秒级网络抖动影响而随机变化，单步训练失去了确定性。

MAI 反其道而行之，在底层彻底关闭了非确定性原子操作与交换机聚合，强制保持固定的 Batch 加载顺序与稳定的 Top-K 路由：
- **可复现校验**：当某个节点故障重启后，通过广播 Checkpoint 重新加载，重放的历史步数所产生的 Loss 日志必须与崩溃前记录的 Loss 在 **位级别（Bit-level）完全一致**。
- **排障价值**：一旦重启重放的 Loss 产生偏离，监控系统便能瞬时定位出隐蔽的数据静默错误或特定坏卡，避免了带有逻辑缺陷的坏节点在集群中长期污染模型权重。

### 二、架构-系统协同瓶颈演进

从初代到第 5 代架构，模型能效增益（EG）从 $1.0\times$ 提升至 $1.69\times$，系统瓶颈随之不断迁移：

![Architecture System Co-Design](/slides/mai-thinking-1/slide-46.png)

1. **v2（首次 GB200 运行）**：启用 GPU-Direct RDMA、优化 Attention 内核与 ZeRO-2，MFU 从 $18\%$ 提升至 $22\%$。
2. **v3（引入 Dropless 动态路由，EG $1.40\times$）**：Token 在各专家的分布在运行时动态漂移。通过重叠（Overlap）通信调度，成功维持了 $22\%$ MFU。
3. **v4（512 专家，Top-8，LatentMoE，8K GPU 规模，EG $1.69\times$）**：专家细粒度化导致单个专家的 GEMM 计算块过小，GPU 计算时间过短，CPU 发射调度严重滞后，MFU 跌至 $16\%$。团队通过适配 FlashAttention-4 并优化 CPU 发射队列，将 MFU 拉回至 $20\%$。
4. **v5（35B 激活，1T 总参数）**：ZeRO-3 在反向传播中密集的权重 Gather 操作造成总线拥塞，MFU 再次降至 $18\%$。团队通过激活值卸载（Activation Offload）回退并常驻 ZeRO-2，将 MFU 稳定在 $20\%$。

---

## 第八部分：强化学习关键技术与防崩溃机制

在后训练阶段，MAI-Thinking-1 摒弃了一切外部蒸馏数据，直接在纯人类数据基座上启动异步大规模强化学习。

### 一、Token-Level GRPO 目标函数完全解析

模型采用 Token 级别的 Group Relative Policy Optimization (GRPO)，其训练目标函数为：

![The RL Objective](/slides/mai-thinking-1/slide-48.png)

$$\mathcal{J}(\theta) = \frac{1}{\sum_{i=1}^G |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \min \left( \hat{r}_{i,t} A_i, \; \text{clip}\left(\hat{r}_{i,t}, \, 1 - \epsilon, \, (1 - \epsilon)^{-1} + k\right) A_i \right)$$

其中：
1. **统一归一化因子**：分母除以当前 Batch 内所有候选回答的总 Token 数量 $\sum |y_i|$（借鉴 DAPO），避免长回答因 Token 数量多而主导梯度。
2. **免 Critic 优势估计**：同一个 Prompt 采样出 $G$ 个候选回答，利用各回答的总奖励 $R_i$ 直接计算归一化优势：
   $$A_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$
3. **重要性采样比率（Importance Ratio）**：
   $$r_{i,t} = \frac{\pi_\theta(y_{i,t} \mid q, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid q, y_{i,<t})}$$
4. **强行硬截断保护**：
   $$\hat{r}_{i,t} = \text{clip}(r_{i,t}, \, 0, \, 50)$$
   标准 PPO 在 $A_i < 0$ 且 $r > 1$ 时未设上限，容易在探索剧烈时产生数值溢出。硬截断消除了致命的梯度尖峰。
5. **宽剪切区间设计**：选取 $\epsilon = 0.6$，此时剪切下界为 $1 - \epsilon = 0.4$，上界基线为 $(1 - \epsilon)^{-1} = 2.5$。宽阔的接受区间允许落后训练端达 40 步以上的陈旧 Rollout 轨迹依然能够贡献有效梯度。
6. **彻底剔除 KL 惩罚与 Entropy Bonus**：由于异步环境下策略分布快速前行，施加刚性的 KL 惩罚会严重压制长链条推理能力；而传统 Entropy Bonus 会导致模型在无关虚词上发散。

### 二、异步陈旧性（Staleness）与重要性比率漂移

在 4,096 推理卡与 768 训练卡并行的异步架构下，采样器（Sampler）生成样本时所依据的权重版本必然落后于当前训练器（Learner）的最新参数：

![Staleness](/slides/mai-thinking-1/slide-49.png)

- 推理卡生成的长思考链样本耗时较长，送达训练器时可能已经落后 5~8 个权重版本。
- 设定权重版本落后超过 8 步的轨迹直接丢弃；落后 1~8 步的样本则通过宽区间重要性加权 $r_{i,t}$ 充分利用。

### 三、熵塌陷机理与非对称剪切参数 $k$ 调控

在强化学习多轮迭代中，高奖励往往倾向于赋予那些高置信度、相对保守的解题路径。策略网络会自发缩小探索分布，导致策略熵 $H(\pi_\theta)$ 指数级下跌，最终退化为单一的模态坍塌（Entropy Collapse）。

![Entropy Dynamics](/slides/mai-thinking-1/slide-50.png)

MAI 引入了不对称剪切上限调节参数 $k \in [0, 2.5]$：
- 当观测到策略熵跌幅过快时，增大 $k$ 值，拔高剪切上界 $(1 - \epsilon)^{-1} + k$，允许概率激增的新探索动作获得更大的正向梯度更新幅度；
- 当策略熵企稳后，收拢 $k$ 值恢复常规约束。

![Entropy Control via k](/slides/mai-thinking-1/slide-51.png)

### 四、崩溃后的自蒸馏重启（Self-Distillation）

若在漫长的训练中不可避免地遭遇了局部的探索塌陷或严重遗忘，传统做法往往是耗费巨大代价回滚至数天前的历史 Checkpoint。MAI 提出了自蒸馏（Self-Distillation）恢复技术：

![Self-Distillation](/slides/mai-thinking-1/slide-53.png)

- 从历史训练池中筛选出塌陷前生成的高质量正向轨迹，重新构建包含代码、数学与长思考的多任务混合集；
- 以塌陷前的健康权重为教师，开启带有 $p=0.15$ 残差 Dropout 的自蒸馏监督微调；
- 经过短暂的自蒸馏，模型的策略熵与表征多样性迅速复苏，重构基石后继续攀爬，避免了昂贵的训练中断。

---

## 第九部分：三大 Specialist 攀爬流水线

MAI 将后训练拆解为三大独立演进的专家领域分支，各分支在独立的强化学习环境中并行攀爬，最终通过蒸馏统一：

![Three Climbs Pipeline](/slides/mai-thinking-1/slide-55.png)

1. **STEM Climb（理科推理攀爬）**：
   - 核心原则：奖励信号的质量取决于答案真值校验器（Answer Key Verifier）的刚性。拒绝使用大模型自我评判（LLM-as-a-Judge），全面采用符号计算、单元测试断言与等价形式验证。
2. **Agentic Climb（智能体工具环境攀爬）**：
   - 本质是环境工程：构建隔离的 Docker 容器沙箱、Linux Terminal 环境与真实编译执行器。为模型提供真实的执行标准输出与报错 Traceback，驱动其习得自我反思与多轮纠错能力。
3. **Helpfulness & Safety（有用性与安全门控）**：
   - **安全作为硬门控（Gate），而非软惩罚权重（Weight）**：若将安全与有用性混杂在单一损失函数中加权，模型极易习得在安全边界上的“投机迎合”。MAI 将安全策略设计为独立的通过性过滤器（Pass/Fail Gate），唯有通过安全准入的轨迹才能进入推理能力奖励分配流程。

---

## 第十部分：分布式强化学习引擎 Rocket

在大规模强化学习集群中，推理吞吐与显存利用率的瓶颈往往卡在推理服务与训练更新之间的异构调度上。MAI 开发了定制化的强化学习引擎 **Rocket**。

![Rocket Architecture](/slides/mai-thinking-1/slide-60.png)

### 一、推理与训练物理计算池解耦

- **推理计算池（Sampler Pool）**：4,096 张 GPU，专职负责根据 Prompt 进行高并发流式自回归解码。
- **训练计算池（Learner Pool）**：768 张 GPU，专职负责吞吐 Batch 梯度的反向传播与参数更新。
- 物理池隔离消除了传统单池共享架构在推理 Prefill/Decode 与训练反向传播之间的显存抢占与算子重新编译开销。

### 二、采样-训练高保真数值重放（Numerics Replay）

在绝大多数开源强化学习框架（如 verl、slime、SkyRL）中，Learner 在执行更新时，通常在本地重新计算一遍旧策略概率 $\pi_{\text{old}}$。然而，由于推理引擎（如 vLLM/SGLang）与训练框架（如 Megatron-LM）采用不同的底层 CUDA 核函数、KV Cache 精度或 Softmax 归约顺序，两者的概率输出存在微小的浮点偏差。在动辄数万步的自回归长序列中，该微弱偏差会被累积放大，导致重要性比率 $\pi_\theta / \pi_{\text{old}}$ 严重失真，诱发训练发散。

![Numerics Replay](/slides/mai-thinking-1/slide-61.png)

Rocket 采取了极致的“对齐重放，而非校正”原则：
1. **直接沿用采样概率**：Learner 计算重要性比率时，分母直接采用 Sampler 在前向生成时记录的原始对数概率。
2. **BF16 精度全链条对齐**：统一推理与训练计算核的内部浮点标准。
3. **路由选择严格重放（Routing Replay）**：MoE 专家的门控选择直接锁定为推理采样时选取的专家 ID，Learner 严禁在后向重新计算 Top-K，防止因微弱浮点浮动导致专家切换。
4. **尾部概率截断重放（Tail Mask Replay）**：推理采样时通常开启了 Top-P（如 $97\%$）截断。Learner 在计算 Softmax 损失时必须精准重现该截断掩码，防止未被采样过的长尾无效 Token 注入有害梯度。

### 三、编译式点对点权重传输计划（Compiled Weight Transfer Plan）

每隔 5 个训练步骤，Learner 必须将数百 GB 的最新权重同步至 4,096 张推理卡。

![Weight Transfer Plan](/slides/mai-thinking-1/slide-62.png)

- **传统开源方案**：先在训练端执行 All-Gather 拼接成完整张量，再通过 NCCL 全量广播（Broadcast）至所有推理节点。该方案数据冗余极其严重，全网带宽被瞬间打爆。
- **Rocket 编译式分发方案**：
  - 在作业启动初期，系统分析 Learner 端的 ZeRO/TP 显存切分拓扑与 Sampler 端的 Tensor Parallel / Pipeline Parallel 切分拓扑；
  - 静态编译出一张最优的内存重叠映射表（Intersection Map），精确指明每个 GPU 的具体内存偏移量；
  - 在同步触发时，通过点对点 GPU-Direct RDMA 仅传输必要的重叠张量切片，彻底避开了冗余广播。通过将 4 个数据并行组与 36 个推理服务集群并行对接，单次全局权重下发时间缩短了数倍。

### 四、分布式强化学习系统横向对比矩阵

| 系统维度 | 社区开源栈（verl / slime / AReaL / SkyRL） | 微软定制化架构 Rocket |
| :--- | :--- | :--- |
| **计算池形态** | 普遍默认单池时分复用；部分前沿栈支持分离 | **严格物理分池**（4,096 推理 + 768 训练） |
| **陈旧性控制** | 刚性阻塞等待（限制落后 2~8 版本） | **自由流式生成**，硬丢弃 $>8$ 步陈旧样本 |
| **$\pi_{\text{old}}$ 概率计算** | 在 Learner 端重新做一次完整前向计算 | **直接复用 Sampler 记录的真实概率** |
| **内核数值差异** | 通过截断重要性采样权重进行数学校正 | **内核硬性重放**（路由与 Top-P 97% 掩码严格重放） |
| **长序列开销** | 额外消耗 1T 参数在 128K 序列上的前向显存 | **0 额外前向开销**，显存与计算高度节省 |
| **权重同步拓扑** | 全量 Gather 后 NCCL 广播 / CUDA IPC | **静态编译的点对点 RDMA 差异分片下发** |

---

## 批判性思考与开放问题

MAI-Thinking-1 展现了工业界在万亿 MoE 与 RL 领域的卓越工程实力，但技术报告也存在着若干未充分消融的叙事与开放疑问：

![Open Questions](/slides/mai-thinking-1/slide-64.png)

1. **强化学习超参数的消融透明度**：在预训练阶段，团队提供了详尽的 Ladder 与 EG 能效数据；但在后训练强化学习部分，诸多关键决策（如 $\epsilon=0.6$、截断上界 50、不对称参数 $k$ 的动态调度）更多依赖于经验叙述（“We found”），缺乏严格的对照消融实验。
2. **自蒸馏重启的真实代价**：报告多次提到在策略塌陷后启用自蒸馏，但未披露十次自蒸馏所消耗的 SFT 训练算力以及恢复正常 RL 攀爬所需的追加步数。
3. **混合律的外推极限**：最终入围的数据配比仅在 $2.8\times$ 算力下进行了验证，而理论显示最佳混合点随算力指数增长持续偏移。
4. **未尝试的潜在技术路线**：架构阶梯主要探索了深度方向（Depth），并未引入 $\mu\text{P}$（Maximal Update Parametrization）跨宽度学习率迁移技术；同时在优化器层面完全绕过了正交化谱优化的实际尝试。

---

## 附录：80 张 Slide 核心技术索引速查

| Slide 编号 | 主题分类 | 核心技术要点与系统结论 |
| :--- | :--- | :--- |
| **01 - 05** | Overview & Thesis | 962B/34.7B MoE 规格、评测对比、Learned-not-inherited 哲学、三大核心命题 |
| **06 - 11** | Part 1: Architecture | Dense/MoE 39 组交替堆叠、EP All-to-All 瓶颈、LatentMoE 6656→3072 投影、$23\%$ 通信压减、舍弃 Shared Expert |
| **12 - 20** | Part 2: Ablations | The Ladder 8 档测试阵列、held-out loss 平滑信号、EG 能效标尺、全局批平衡与 Dropless 路由、数据排序反转 |
| **21 - 25** | Part 3: Real Run | $TPP \gg 20$ 工业考量、小模型数据重复率对齐策略、Muennighoff $e^{-R/15}$ 4 轮重复衰减律 |
| **26 - 32** | Part 4: Pre-training Data | 29.2T 纯人类语料、MinHash/聚类去重、55% 代码加权、Mixing Laws 混合律公式与狭窄数据饱和 |
| **33 - 38** | Part 5: Mid-Training | 不加新数据、仅倾斜配比、RL 只能强化基座有概率采出的解空间、16K 到 256K 序列拉伸 |
| **39 - 43** | Part 6: Recipe & Optim | AdamW 0.1x LR Floor、Attention 零初始化破除路由塌陷、SGD/Adam/Muon 谱球几何对照与生产权衡 |
| **44 - 46** | Part 7: YOLO Infra | 90.0% Goodput、Bit-Exact 位精确确定性训练、8K GB200 集群 MFU 瓶颈迁移协同演进 |
| **47 - 54** | Part 8: RL Tech | Token-level GRPO 目标函数、$\epsilon=0.6$ 宽剪切应对异步陈旧性、非对称参数 $k$ 控熵、自蒸馏恢复 |
| **55 - 58** | Part 9: Three Climbs | STEM 刚性验题器、Agentic 容器环境工程、Safety 作为独立硬门控、多专长蒸馏融合 |
| **59 - 63** | Part 10: Rocket RL | 4096 推理 + 768 训练异构分池、采样概率与路由/Top-P 掩码严格数值重放、编译式 RDMA 权重传输 |
| **64 - 65** | Closing & Takeaways | 批判性开放问题审视、三大系统工程师必借核心方法论总结 |
| **66 - 80** | Backup Slides | 并行范式对比、优化器谱范数几何补充推导、Muon 工业界落地阻碍、FP8 矩阵计算与 FP32 敏感边界 |

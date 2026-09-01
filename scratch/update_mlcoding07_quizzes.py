import re

# 1. Update Chinese note
with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md", "r", encoding="utf-8") as f:
    content_07 = f.read()

target_m8_zh = r"""## 模块八：工业级核心技术高频自测单选题（Interactive Self-Test Quizzes）

<details class="exercise">
<summary><span class="q-label">Q1 · 精排负采样与概率还原</span> <span class="q-text">某信息流推荐精排模型真实点击率 $p = 1\%$。为了加速训练，对未点击负样本以保留率 $w = 10\%$（$w=0.1$）进行均匀下采样，正样本全部保留。在模型输出预测值 $\hat{p} = 0.0917$（$9.17\%$）后，线上竞价与融分系统应如何精确还原真实自然概率 $p$？</span></summary>

- [ ] **A.** 直接使用 $\hat{p} \times w = 0.0917 \times 0.1 = 0.00917$ 进行粗略估计。
- [ ] **B.** 使用线性放缩公式 $p = \frac{\hat{p}}{w} = 0.917$。
- [x] **C.** 使用解析还原公式 $p = \frac{\hat{p}}{\hat{p} + \frac{1 - \hat{p}}{w}}$，计算得出精准自然概率 $p = 0.01$（$1.0\%$）。
- [ ] **D.** 下采样仅改变 LogLoss 绝对值，不影响输出概率的绝对标定，无需做任何概率还原。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **先验扭曲**：在负采样率 $w$ 下，模型学到的条件概率被拉高为 $\hat{p} = \frac{p}{p + w(1-p)}$；
>   2. **精准逆变换**：解反函数可得 $p = \frac{\hat{p}}{\hat{p} + \frac{1 - \hat{p}}{w}}$。代入 $\hat{p} = 0.0917$ 与 $w = 0.1$：
>      $$p = \frac{0.0917}{0.0917 + \frac{1 - 0.0917}{0.1}} = \frac{0.0917}{0.0917 + 9.083} = \frac{0.0917}{9.1747} = 0.01 \quad (1.0\%)$$
>   3. **业务后果**：如果不做还原直接传入 $eCPM = pCTR \times \text{Bid}$ 竞价排序，出价与预估 CTR 将虚高近 10 倍，瞬间击穿广告主预算。
</details>

<details class="exercise">
<summary><span class="q-label">Q2 · 多目标学习与负迁移治理</span> <span class="q-text">在电商多目标精排模型中（同时预测点击 CTR 与购买 CVR），频繁出现“点击率显著上涨但下单成交量严重下滑”的跷跷板效应（Negative Transfer）。下列哪种架构与优化方案在理论与工程上<strong>最能彻底阻断不同任务私有特征间的梯度撕裂</strong>？</span></summary>

- [ ] **A.** Shared-Bottom 架构：让底层所有参数由两个任务均摊反向传播梯度。
- [ ] **B.** 静态加权 Loss 调参：通过人工微调 Loss 权重（如 $\mathcal{L} = 0.8\mathcal{L}_{\text{CTR}} + 0.2\mathcal{L}_{\text{CVR}}$）。
- [x] **C.** PLE (Progressive Layered Extraction) 架构结合 PCGrad 梯度正交投影：物理隔离任务私有专家与共享专家，并在梯度负冲突（$\cos(\mathbf{g}_i, \mathbf{g}_j) < 0$）时进行法平面正交投影。
- [ ] **D.** 采用单任务训练并在离线阶段直接相乘（$p\text{CTCVR} = p\text{CTR} \times p\text{CVR}$）。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **Shared-Bottom 的缺陷**：底层硬共享导致点击梯度与转化梯度相互抵消（$\cos(\mathbf{g}_A, \mathbf{g}_B) < 0$），高频 CTR 完全主导参数更新；
>   2. **PLE 的物理隔离优势**：PLE 显式将 Experts 划分为 Task-Specific 独占专家与 Shared 共享专家，各任务私有表征不再受到其他任务梯度的直接污染；
>   3. **PCGrad 动态保障**：当不同任务在共享层出现破坏性负方向分量时，将梯度正交投影到冲突梯度的法平面，彻底消除梯度对抗。
</details>

<details class="exercise">
<summary><span class="q-label">Q3 · 离线评估指标 GAUC 解析</span> <span class="q-text">在评估工业级推荐精排模型时，为什么算法团队通常以 <strong>Request-Grouped GAUC（按单次刷新请求分组的 GAUC）</strong> 作为第一核心排序北极星，而非全局 Global AUC 或 User-Grouped GAUC？</span></summary>

- [ ] **A.** Request-GAUC 的数值通常比 Global AUC 更高，便于向上汇报展示涨点效果。
- [ ] **B.** Global AUC 会受到不同用户样本量不平衡的影响，而 User-GAUC 完全没有这一问题。
- [x] **C.** Global AUC 混合了跨用户的全局打分偏倚（辛普森悖论）；User-GAUC 混杂了用户跨时段的意图漂移；而 Request-GAUC 严格评估单次刷新同屏候选物（Within-Slate）内部的精准相对排序。
- [ ] **D.** Request-GAUC 能够直接替代 PCOC 和 ECE 来评估模型的概率绝对校准能力。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **Global AUC 的致命缺陷**：若模型学会“给高活用户打高分、低活用户打低分”，Global AUC 很高但单次列表内部排序完全随机；
>   2. **User-GAUC vs. Request-GAUC**：User-GAUC 将用户早上的未点击项与晚上的点击项混合计算，引入了昼夜意图漂移噪声；而 Request-GAUC 严格限制在单次刷新展现的 6~10 个同屏物料中，**百分之百对齐了用户决策瞬间面临的真实选择现场**；
>   3. **D 选项错误**：GAUC 具有单调保序不变性，只能衡量排序分辨力（Discrimination），无法评估绝对概率校准度（Calibration）。
</details>

<details class="exercise">
<summary><span class="q-label">Q4 · 在线实验与小样本因果推断</span> <span class="q-text">电商商详页（PDP）UI 改版 A/B 实验显示：列表到商详页点击率 CTR 提升 $+12\%$（$p < 0.01$），但商详页内转化率 CVR 保持平稳（$\Delta \text{CVR} \approx 0\%$，$p = 0.65$）。在黑金大客户分层（$N \approx 100$）样本量极小时，下列决策与推断方法中<strong>最科学严谨的是</strong>：</span></summary>

- [ ] **A.** 因为大客户层 $t$ 检验 $p > 0.05$，说明新 UI 对高价值客户无效，应立即下线改版。
- [ ] **B.** 条件 CVR 没有提升说明改版失败，单次曝光带来的实际商业回报没有增长。
- [x] **C.** 全盘总成交量 $\text{CTCVR} = \text{CTR} \times \text{CVR}$ 实际上净增长了 $+12\%$；CVR 持平源于新增边缘访客的“流量稀释效应”；大客户小样本层应采用<strong>经验贝叶斯部分池化（Empirical Bayes Shrinkage）</strong>与前置消费协变量 CUPED 进行稳健推断。
- [ ] **D.** 延长实验至 6 个月以上，直到大客户层样本量达到数十万后自然拒绝原假设。

> 💡 **答案解析**：
> - **正确选项：C**。
>   1. **漏斗全局因果净增**：$\text{CTCVR} = \text{CTR} \times \text{CVR} = 1.12 \times 1.0 = 1.12$（全盘总订单量净增 $+12\%$）；
>   2. **流量稀释因果机制**：改版吸引了额外原本不点击的低意图用户，在分母稀释下仍维持原有高转化率，证明落地页承接能力强劲；
>   3. **小样本推断**：$p > 0.05$ 仅代表统计功效不足（$\text{Power} < 20\%$），绝非无效证明。采用经验贝叶斯收缩模型 $\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \mu_{\text{grand}} + (1-B) \bar{Y}_{\text{small}}$ 跨层借力，是工业界防误判的金标准。
</details>
"""

m8_pos = content_07.find("## 模块八")
if m8_pos != -1:
    content_07 = content_07[:m8_pos] + target_m8_zh
else:
    content_07 += "\n\n" + target_m8_zh

with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.md", "w", encoding="utf-8") as f:
    f.write(content_07)

# 2. Update English note
with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md", "r", encoding="utf-8") as f:
    content_07_en = f.read()

target_m8_en = r"""## Module 8: High-Yield Industrial ML Multiple-Choice Quizzes

<details class="exercise">
<summary><span class="q-label">Q1 · Negative Downsampling & Probability Recovery</span> <span class="q-text">A feed ranking model operates with natural CTR $p = 1\%$. Negative samples are uniformly downsampled with retention rate $w = 10\%$ ($w=0.1$). If the model outputs $\hat{p} = 0.0917$ ($9.17\%$), how should downstream auction bidding restore natural probability $p$?</span></summary>

- [ ] **A.** Approximate via linear scaling: $\hat{p} \times w = 0.00917$.
- [ ] **B.** Rescale via $p = \hat{p} / w = 0.917$.
- [x] **C.** Apply exact closed-form probability recovery: $p = \frac{\hat{p}}{\hat{p} + \frac{1-\hat{p}}{w}}$, producing $p = 0.01$ ($1.0\%$).
- [ ] **D.** Negative downsampling only affects LogLoss, so no recovery is required for predicted probabilities.

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. **Prior Distortion**: Under negative downsampling rate $w$, model predictions are distorted to $\hat{p} = \frac{p}{p + w(1-p)}$.
>   2. **Exact Inverse**: Solving for $p$ yields $p = \frac{\hat{p}}{\hat{p} + \frac{1-\hat{p}}{w}}$. Substituting $\hat{p}=0.0917$ and $w=0.1$ exactly recovers natural probability $p = 0.01$ ($1.0\%$).
</details>

<details class="exercise">
<summary><span class="q-label">Q2 · Multi-Objective Learning & Seesaw Mitigation</span> <span class="q-text">In a multi-objective ranking model (joint CTR & CVR prediction), which architecture and optimization protocol <strong>most effectively isolates task-specific representations from negative gradient conflicts ($\cos(\mathbf{g}_i, \mathbf{g}_j) < 0$)</strong>?</span></summary>

- [ ] **A.** Shared-Bottom architecture: all shared representations average gradients across tasks.
- [ ] **B.** Static loss weight grid search (e.g. $\mathcal{L} = 0.8\mathcal{L}_{\text{CTR}} + 0.2\mathcal{L}_{\text{CVR}}$).
- [x] **C.** PLE (Progressive Layered Extraction) with PCGrad orthogonal projection: physical decoupling of task-specific and shared experts with orthogonal gradient projection.
- [ ] **D.** Independent single-task training combined with offline probability multiplication ($p\text{CTCVR} = p\text{CTR} \times p\text{CVR}$).

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. **Shared-Bottom Flaw**: Shared parameter layers suffer direct destructive interference from opposing gradient vectors.
>   2. **PLE Advantage**: PLE enforces structural physical isolation between task-specific and shared expert modules.
>   3. **PCGrad**: Projects conflicting gradient components onto the normal plane, eliminating gradient destruction.
</details>

<details class="exercise">
<summary><span class="q-label">Q3 · Grouped AUC (GAUC) Invariance</span> <span class="q-text">Why is <strong>Request-Grouped GAUC</strong> favored over Global ROC-AUC as the primary offline ranking benchmark in precision rankers?</span></summary>

- [ ] **A.** Request-GAUC is mathematically larger, providing inflated metrics for management presentations.
- [ ] **B.** Global AUC is sensitive to negative sampling, whereas GAUC is not.
- [x] **C.** Global AUC conflates cross-user activity baselines (Simpson's paradox); Request-GAUC measures within-slate discrimination on a single refresh, directly matching real user decisions.
- [ ] **D.** Request-GAUC can substitute for PCOC and ECE in evaluating absolute probability calibration.

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. **Global AUC Bias**: A model that simply assigns higher scores to active users achieves high Global AUC while failing at within-session ranking.
>   2. **Request-GAUC Alignment**: Evaluates pairwise rankings strictly within the 6~10 items presented on a single screen refresh, eliminating diurnal drift.
</details>

<details class="exercise">
<summary><span class="q-label">Q4 · A/B Testing Attribution & Small-Sample Inference</span> <span class="q-text">An e-commerce PDP UI test increases list CTR by $+12\%$ ($p < 0.01$) while PDP CVR is flat ($\Delta \text{CVR} \approx 0\%$, $p = 0.65$). For an enterprise VIP tier with $N \approx 100$, what is the <strong>most rigorous statistical approach</strong>?</span></summary>

- [ ] **A.** Since $p > 0.05$ in the VIP slice, conclude the UI harms high-value users and abort the rollout.
- [ ] **B.** Conclude the experiment failed because within-PDP conversion rate did not increase.
- [x] **C.** Total orders per impression $\text{CTCVR} = \text{CTR} \times \text{CVR}$ increased by $+12\%$; flat CVR is driven by traffic dilution from marginal users; use <strong>Empirical Bayes Partial Pooling / Shrinkage</strong> and CUPED baseline covariates for VIP tier inference.
- [ ] **D.** Extend the test for 6 months until sample size in the VIP tier reaches hundreds of thousands.

> 💡 **Explanation**:
> - **Correct Answer: C**.
>   1. **Net Funnel Lift**: $\text{CTCVR} = 1.12 \times 1.0 = 1.12$ ($+12\%$ net order volume per impression).
>   2. **Traffic Dilution**: Lower click friction brings in marginal lower-intent visitors. Maintaining flat conversion confirms strong page performance.
>   3. **Small-Sample Inference**: $p > 0.05$ reflects low power ($\text{Power} < 20\%$), not evidence of absence. Empirical Bayes shrinkage $\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \mu_{\text{grand}} + (1-B) \bar{Y}_{\text{small}}$ borrows statistical strength across tiers.
</details>
"""

m8_pos_en = content_07_en.find("## Module 8")
if m8_pos_en != -1:
    content_07_en = content_07_en[:m8_pos_en] + target_m8_en
else:
    content_07_en += "\n\n" + target_m8_en

with open("notes/MLCoding/MLCoding07 Industrial Machine Learning System RecSys Reranking ABTesting.en.md", "w", encoding="utf-8") as f:
    f.write(content_07_en)

print("Updated MLCoding07 quizzes zh and en")

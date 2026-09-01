import os

with open('notes/BusinessAlgorithm/BusinessAlgorithm08 Industrial ML AB Testing Experimentation.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace header overview
old_overview = """本篇系统构建工业级 A/B 实验知识体系：
1. **工业 ML 与 A/B 测试的核心概念：为什么离线指标高不等于线上收益？**
2. **工业级 A/B 实验全生命周期规范（设计、分流、CUPED 方差缩减、SRM 检验与决策准则）**
3. **经典案例深度剖析：注册漏斗（Sign-Up Funnel）按钮颜色与位置的全因子实验设计**"""

new_overview = """本篇系统构建工业级 A/B 实验知识体系：
1. **工业 ML 与 A/B 测试的核心概念：为什么离线指标高不等于线上收益？**
2. **工业级 A/B 实验全生命周期规范（设计、分流、CUPED 方差缩减、SRM 检验与决策准则）**
3. **经典案例深度剖析一：注册漏斗（Sign-Up Funnel）按钮颜色与位置的全因子实验设计**
4. **经典案例深度剖析二：电商商详页 (PDP) 改版转化归因（CTR 涨，CVR 平）与小样本分层因果推断**"""

content = content.replace(old_overview, new_overview)

# Replace Module 4
target_module4 = """## 模块四：工业级 A/B 实验核心自测题

### Q1：为什么在 A/B 测试中，即使第 3 天 p-value 已经达到了 0.001，也坚决不能提前关停并宣布胜利（Peeking Problem）？
> **答**：
> 1. **多重假设检验假阳性膨胀（Alpha Inflation via Continuous Monitoring）**：传统的假设检验假设样本量是固定的。如果在实验运行期间每天甚至每小时去“偷看（Peek）”数据并在显著时立即早停，本质上是在进行数十次重复假设检验。真实的全局第一类错误率（False Positive Rate）会从 $5\%$ 迅速飙升至 $20\%\sim40\%$；
> 2. **周期性偏差（Day-of-Week Seasonality）**：工作日与周末的用户画像、决策耐心存在显著周期性波动，未跑满完整周周期（如 14 天）的样本无法代表长期稳态；
> 3. **新奇效应尚未消退**：早期显著往往由用户的新奇感引起，随着时间推移效应可能衰减归零甚至反转。若确需动态早停，必须采用严格的 **Sequential Testing 序列检验（如 mSPRT / Alpha-Spending Function）** 调整拒绝域阈值。

### Q2：CUPED（Controlled-experiment Using Pre-Experiment Data）方差缩减的底层数学原理是什么？
> **答**：
> 1. **核心思想**：利用实验前用户已有的基线特征 $X$（与实验后目标指标 $Y$ 强相关，且由于发生在实验前，与实验处理完全正交独立），消除指标中原有的用户固有方差；
> 2. **修正指标公式**：
>    $$\hat{Y}_{\text{CUPED}} = Y - \theta (X - \mathbb{E}[X]), \quad \text{其中 } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
> 3. **方差压缩倍率**：
>    $$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - \rho^2)$$
>    若实验前后的相关系数 $\rho = 0.8$，则方差直接骤降 $64\%$（仅剩原方差的 $36\%$），在不增加任何线上流量和实验时间的前提下，直接将实验所需的样本量降低近 3 倍！"""

new_module4_and_5 = """## 模块四：实战案例二：电商商详页 (PDP) 改版转化归因与极小样本分层因果推断

> 📌 **业务场景**：电商平台对商品详情页（Product Detail Page, PDP）的头图与 CTA 布局进行了 UI 改版。A/B 实验结果显示：
> - **列表页到商详页的点击率（CTR）显著上涨 $+12\%$ ($p < 0.01$)**；
> - **进入商详页后的购买转化率（CVR）保持平稳，无显著变化（$\Delta \text{CVR} \approx 0\%, p = 0.65$）**。

---

### 1. 现象归因与统计学机制深度拆解

```text
电商转化漏斗分解与流量稀释效应:
【全盘列表曝光 Impression】 (100,000)
       │
       ▼  CTR (Control: 5.0% ➔ Treatment: 5.6%, 相对 +12%)
【进入商详页 PDP Pageviews】 (Control: 5,000 ➔ Treatment: 5,600)
       │
       ▼  CVR (Control: 10.0% ➔ Treatment: 10.0%, 相对 +0%)
【最终下单成交 Purchases】 (Control: 500 ➔ Treatment: 560, 净增 +60 单! +12%)
```

#### (1) 为什么说“CTR 涨，CVR 平”实际上是一个强烈的正向商业胜利？
根据全链路因果分解定理：

$$\text{CTCVR} = \frac{\text{总成交购买数}}{\text{总曝光展示数}} = \text{CTR} \times \text{CVR}$$

- 若 $\text{CTR}$ 提升 $+12\%$，而条件转化率 $\text{CVR}$ 保持不变（$10\% \to 10\%$），则**单次曝光带来的全盘总成交订单量实际上净增长了 $+12\%$**！

#### (2) 为什么商详页改版没有带来 CVR 的上涨？（流量稀释效应与边际意图递减）
在因果推断中，这被称为**自选择偏差与流量稀释效应（Traffic Dilution / Selection Effect）**：
- 改版后的 UI 降低了用户的进入门槛，不仅吸引了原本就会买的核心高意向用户，还**额外吸纳了一批原本不会点击的“低意向/摇摆边缘用户（Marginal Users）”**；
- 边缘用户的固有购买意愿天然偏低。**在涌入大量低潜用户稀释了分母的前提下，商详页依然能够维持原有的高 CVR，说明改版后的页面对不同意图的用户均具备强大的承接转化能力**。

---

### 2. 统计与实验诊断排查流水线（Diagnostic Protocol）

为了排除“诱导点击/货不对板”的假阳性风险，必须执行以下 4 项诊断：

```text
商详页诊断分析四大排查工具:
┌─────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 排查维度                │ 核心分析工具与判定阈值                                                 │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. 停留时长与跳出率     │ 统计 PDP 停留时长分布（KS 检验）。若停留时长中位数暴跌、秒退率激增，   │
│ (Dwell Time & Bounce)   │ 说明存在诱导点击（Clickbait），用户进页后失望离开。                   │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. 中间步骤加购率       │ 分析加购率（Add-to-Cart）与“立即购买”点击率。若加购率涨但结算页跳出，  │
│ (Cart & Checkout Funnel)│ 说明转化瓶颈在下游支付链路，而非商详页。                               │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. 逆向售后指标         │ 监控 7 天退货率（Return Rate）与纠纷率。防范误导性视觉导致的冲动消费。 │
├─────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 4. 新奇效应队列跟踪     │ 按用户访问天数做 Cohort Analysis，观察 Week 2 增益是否衰减。           │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 3. 用户分层异质性分析（HTE）与极小样本分层因果推断

当按用户价值分层（User Tiers: 普通用户、新用户、高价值黑金 VIP）切片时，经常面临**高价值黑金 VIP 层样本量极小（如 $N_{\text{Treatment}} = 120, N_{\text{Control}} = 115$）**的严峻统计挑战。

```text
小样本分层的统计学困境:
• 样本量 N ≈ 100 ➔ 标准误差 SE 极高 ➔ 统计功效 Power < 20%
• 经典双样本 t 检验的 95% 置信区间宽达 [-18%, +22%] (p = 0.45 无法拒绝原假设)
• 致命风险: 错误地将"没有证据证明有效 (Failure to reject)" 武断判定为 "该层级无效"!
```

---

### 4. 极小样本分层合理推断的三大科学统计方案

#### 方案一：贝叶斯分层模型与经验贝叶斯部分池化（Empirical Bayes Partial Pooling / Shrinkage）★推荐

不要将小样本层级作为孤立孤岛单独做 t 检验，而是建立**分层贝叶斯模型（Hierarchical Bayesian Model）**，在保持各层特异性的同时，跨层“借力（Borrowing Strength）”：

$$\theta_k \sim \mathcal{N}(\mu_{\text{grand}}, \tau^2), \quad \bar{Y}_k \sim \mathcal{N}(\theta_k, \sigma_k^2)$$

- **收缩估计量（Shrinkage Estimator）**：
  $$\hat{\theta}_{\text{small}}^{\text{shrunk}} = B \cdot \mu_{\text{grand}} + (1 - B) \cdot \bar{Y}_{\text{small}}, \quad \text{其中收缩权重 } B = \frac{\sigma_{\text{small}}^2}{\sigma_{\text{small}}^2 + \tau^2}$$
  - 当小样本层自身的方差 $\sigma_{\text{small}}^2$ 极大时，$B \to 1$，估计值自适应向大盘全局均值 $\mu_{\text{grand}}$ 靠拢（收缩），消除小样本极端噪点；
  - 当样本量充分时，$B \to 0$，估计值还原为该层自身观测值。

#### 方案二：CUPED 结合前置历史消费协变量（CUPED Variance Reduction）
利用黑金 VIP 用户在**实验前 30 天的历史下单额与访问频次作为前置协变量 $X$**。
由于高净值用户的历史消费习惯具备极高的跨周期稳定性（相关系数 $\rho \approx 0.85 \sim 0.90$）：
- 经 CUPED 调整后，方差削减 $\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - 0.90^2) = 0.19 \cdot \text{Var}(Y)$；
- **等效于将小样本层的有效样本量瞬间放大 5 倍以上**，直接将原本不可检测的微弱信号拉入可检验置信区间。

#### 方案三：非参数精确置换检验（Exact Permutation Test / Fisher-Pitman Test）
当样本量过小导致中心极限定理（CLT）失效、正态分布假设不成立时，严禁使用经典渐近 z/t 检验。采用 **Exact Permutation Test**：
- 在 235 个样本的全部可能分组排列中遍历或进行 $100,000$ 次随机重抽样打散，计算经验零分布下的精确 $p$-value；
- 结合**贝叶斯后验超越概率** $P(\theta_{\text{Treatment}} > \theta_{\text{Control}} \mid \text{Data}) > 0.90$ 给出有把握的置信度决策。

---

## 模块五：工业级 A/B 实验核心自测题

### Q1：为什么在 A/B 测试中，即使第 3 天 p-value 已经达到了 0.001，也坚决不能提前关停并宣布胜利（Peeking Problem）？
> **答**：
> 1. **多重假设检验假阳性膨胀（Alpha Inflation via Continuous Monitoring）**：传统的假设检验假设样本量是固定的。如果在实验运行期间每天甚至每小时去“偷看（Peek）”数据并在显著时立即早停，本质上是在进行数十次重复假设检验。真实的全局第一类错误率（False Positive Rate）会从 $5\%$ 迅速飙升至 $20\%\sim40\%$；
> 2. **周期性偏差（Day-of-Week Seasonality）**：工作日与周末的用户画像、决策耐心存在显著周期性波动，未跑满完整周周期（如 14 天）的样本无法代表长期稳态；
> 3. **新奇效应尚未消退**：早期显著往往由用户的新奇感引起，随着时间推移效应可能衰减归零甚至反转。若确需动态早停，必须采用严格的 **Sequential Testing 序列检验（如 mSPRT / Alpha-Spending Function）** 调整拒绝域阈值。

### Q2：CUPED（Controlled-experiment Using Pre-Experiment Data）方差缩减的底层数学原理是什么？
> **答**：
> 1. **核心思想**：利用实验前用户已有的基线特征 $X$（与实验后目标指标 $Y$ 强相关，且由于发生在实验前，与实验处理完全正交独立），消除指标中原有的用户固有方差；
> 2. **修正指标公式**：
>    $$\hat{Y}_{\text{CUPED}} = Y - \theta (X - \mathbb{E}[X]), \quad \text{其中 } \theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$
> 3. **方差压缩倍率**：
>    $$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - \rho^2)$$
>    若实验前后的相关系数 $\rho = 0.8$，则方差直接骤降 $64\%$（仅剩原方差的 $36\%$），在不增加任何线上流量和实验时间的前提下，直接将实验所需的样本量降低近 3 倍！

### Q3：如果某次商详页改版导致列表 CTR $+15\%$，但商详页 CVR 相对下跌 $-3\%$，该如何综合评估是否全量上线？
> **答**：
> 1. **全盘净产出检验（CTCVR 净效应）**：
>    $$\text{New CTCVR} = 1.15 \times 0.97 = 1.1155 \quad (\text{整体曝光到购买依然净增长 } +11.55\%)$$
> 2. **排查下跌是否由流量稀释导致**：若新增流量主要来自低意向人群，则 CVR 小幅稀释属于自然物理规律；
> 3. **核查核心护栏指标**：若跳出率、退货率无异常，且高价值 VIP 核心客群的绝对 GMV 无下跌，则应当**果断全量上线**。"""

content = content.replace(target_module4, new_module4_and_5)

with open('notes/BusinessAlgorithm/BusinessAlgorithm08 Industrial ML AB Testing Experimentation.md', 'w', encoding='utf-8') as f:
    f.write(content)
print("Successfully updated BusinessAlgorithm08 Chinese note")

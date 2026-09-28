# 21 · 推荐与营销中的因果推断与增量建模（Causal Inference & Uplift Modeling）

在工业级搜索、推荐与用户增长营销体系中，机器学习模型正经历从**相关性预测（Correlation）**到**因果增量决策（Causation & Uplift）**的根本范式跃迁。

传统推荐精排模型（如 CTR/CVR 预估）本质是观测数据上的条件概率拟合：$P(Y=1 \mid X, T)$。它回答的是：*“如果给该用户展示某推荐位或发放一张满减券，他最终发生转化的概率是多少？”* 在高意向老客群体中，该概率天然高达 90% 以上——无论是否干预，用户都会购买。若仅按 CVR 排序进行预算倾斜，算法实质上是在系统性地捕获“自然转化者（Free Riders / 搭便车用户）”，造成巨额营销预算浪费。

**因果推断与增量建模（Uplift Modeling）** 则从反事实视角切入，精确量化因干预行为自身引发的**净增量效应（Treatment Effect）**：

$$
\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]
$$

本章系统梳理工业界因果推断数学基础、AB 实验陷阱治理、五大 Uplift Meta-Learners 与因果树模型数学机理、离线评估体系以及受限预算下的 ROI 决策闭环。

---

## 模块一：因果推断数学基础（Causal Inference Fundamentals）

### 1. 潜在结果框架（Neyman-Rubin Potential Outcomes Framework）

对于总体中的任意独立样本个体 $i$：
- **处理状态（Treatment Indicator）**：$T_i \in \{0, 1\}$。$T_i = 1$ 表示施加干预（如弹出补贴弹窗、推送重定向 Push、推荐特定类目）；$T_i = 0$ 表示维持现状对照。
- **潜在结果（Potential Outcomes）**：$Y_i(1)$ 表示个体接受干预时的潜在转化反应；$Y_i(0)$ 表示个体不接受干预时的潜在转化反应。

#### 因果推断根本难题（Fundamental Problem of Causal Inference）

在现实物理世界中，对任意单一样本 $i$，处理状态 $T_i$ 具有排他性。我们**只能观测到其中一个潜在状态的结果**，而未发生的另一个状态被称为**反事实（Counterfactual）**，天然永久缺失：

$$
Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)
$$

> **📊 因果推断核心矛盾：反事实缺失矩阵（The Counterfactual Missing Data Problem）**
>
> 📌 **核心结论**：个体因果效应（ITE）不可观测。因果推断的一切统计建模本质，是利用实验随机性或特征条件独立假设，通过群体统计期望去稳健“推补（Impute）”反事实均值。

| 样本 $i$ | 特征 $X_i$ | 干预分配 $T_i$ | 观测结果 $Y_i$ | $Y_i(1)$ (干预潜在结果) | $Y_i(0)$ (对照潜在结果) | 真实增量 $\tau_i$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 用户 1 | 高活跃/一线城市 | **1 (实验组)** | **1** | 1 (已观测) | ? (反事实缺失) | 无法单点直接计算 |
| 用户 2 | 低活跃/三线城市 | **0 (对照组)** | **0** | ? (反事实缺失) | 0 (已观测) | 无法单点直接计算 |
| 用户 3 | 中活跃/二线城市 | **1 (实验组)** | **1** | 1 (已观测) | ? (反事实缺失) | 无法单点直接计算 |

#### 因果效应的三大度量分层（ATE vs. ATT vs. CATE）

在工业界实际落地中，因果推断绝非抽象的数学符号，而是直接映射到大数据平台（Hive / ClickHouse）的数据字段与在线决策引擎中：

---

##### 1. 工业实战中的变量实体映射（Feature $X$、Treatment $T$、Outcome $Y$ 到底是什么？）

以最典型的大厂**“电商/外卖大促智能发券补贴”**与**“App 消息推送（Push）召回”**为例：

> **📋 工业大厂样本宽表数据结构定义（Hive / Feature Store 真实字段切片）**

| 变量类别 | 数仓具体字段名 | 数据类型与业务含义 | 在因果建模中的实战作用 |
| :--- | :--- | :--- | :--- |
| **特征向量 $X$**<br>(Covariates) | `user_age_group`<br>`city_tier`<br>`device_brand_price`<br>`active_days_30d`<br>`pay_gmv_30d`<br>`cart_unpaid_cnt_7d`<br>`hist_coupon_use_rate`<br>`cur_browse_cate_l1` | • 用户静态画像（年龄/一二线或下沉/千元机或旗舰机）<br>• RFM 历史消费统计（近30天活跃天数、消费金额）<br>• 强意向未转化信号（近7天加购但未支付次数）<br>• 价格敏感度指标（历史领券核销率、折扣偏好）<br>• 实时上下文（当前访问品类：高毛利美妆 vs 低毛利日百） | 作为模型的输入协变量。因果模型（如 Causal Forest / X-Learner）基于 $X$ 捕捉人群异质性（Heterogeneity），区分出“对价格极度敏感但有购买意向”的人群与“无论如何都会买”的老客。 |
| **干预动作 $T$**<br>(Treatment) | `is_coupon_issued`<br>`treatment_type` | • 离散二值：`1`（发放满 100 减 20 促销券），`0`（对照组，纯自然推荐不下发任何补贴）<br>• 多门槛干预：`T ∈ {0: 无券, 1: 满50-5, 2: 满100-20}` | 工程侧实际施加的物理干预信号。在随机 AB 实验中由服务端随机分流系统（如 Hash 落桶）决定。 |
| **目标响应 $Y$**<br>(Outcome) | `is_order_paid_24h`<br>`pay_gmv_24h`<br>`net_profit_24h` | • 分类目标：干预后 24 小时内是否支付成功（`0 或 1`）<br>• 回归目标：干预后成交的总 GMV（连续实数）<br>• 净利润：`GMV * 毛利率 - (补贴面额 * 核销标记)` | 模型希望通过干预拉动的业务结果。工业界若仅优化 GMV 极易产生巨额补贴亏损，因此成熟业务通常将 $Y$ 建模为净商业增量（Net Profit）或带成本约束的转化率。 |

---

---

##### 2. ATE、ATT、CATE 在业务生命周期中的应用与落地决策

| 效应度量 | 数学形式 | 工业界的通俗本质 | 在大厂实习/工程中怎么计算？ | 核心业务应用场景与决策局限 |
| :--- | :--- | :--- | :--- | :--- |
| **ATE**<br>(Average Treatment Effect) | $$\mathbb{E}[Y(1) - Y(0)]$$ | **“大盘普惠发券，总账到底赚不赚？”**<br><br>衡量若将该策略全量推向全平台所有用户，人均能够带来的净增量期望。 | **随机 AB 实验全量均值差**：<br>在无偏随机分流桶中，统计实验组与对照组全样本均值之差：<br>$$\widehat{\text{ATE}} = \bar{Y}_{T=1} - \bar{Y}_{T=0}$$<br>通过双样本 t-test / Welch t-test 计算 p-value。 | **大盘立项与方案上线底线评审**：<br>• 用于高管汇报：“该功能全量后能为大盘提供多大增益”。<br>• **危险陷阱**：若 ATE = +0.01（转化率提升 1%），若对 1000 万全量用户普发 10 元券（成本 1 亿元），1% 转化带来的平台增量抽成只有 200 万元，**全量推行 ATE 会造成 9800 万元巨额亏损！** 因此 ATE 绝不能用于受限资源的投放分配。 |
| **ATT**<br>(Average Treatment Effect on the Treated) | $$\mathbb{E}[Y(1) - Y(0) \mid T = 1]$$ | **“运营经验选出的这波人，发券到底有没有增量？”**<br><br>衡量在**当前已经被规则或算法选中干预**的特定客群内部，干预相对于不干预的净收益。 | **规则客群 Holdout 对照实验**：<br>对运营圈定的人群（如“近 30 天未登录的高净值流失老客”），预留 5%~10% 的反向切流对照组（符合条件但故意不发）：<br>$$\widehat{\text{ATT}} = \bar{Y}_{T=1, \text{rule}} - \bar{Y}_{T=0, \text{rule}}$$ | **既有运营策略审计与去水排查**：<br>• 实习中最常见任务：审计运营规则是否有效。<br>• 若某条运营规则圈定人群的 ATT 为 0 甚至为负，证明这部分用户要么本就会自然买单（给羊毛党送钱），要么是对优惠券免疫的无效用户，可据此下线低效规则。 |
| **CATE**<br>(Conditional Average Treatment Effect) | $$\tau(X) = \mathbb{E}[Y(1) - Y(0) \mid X]$$ | **“针对画像为 $X$ 的具体用户，发券能净增多少下单概率？”**<br><br>量化特定画像特征人群的异质性净增益，是 **Uplift 建模的核心标的**。 | **因果机器学习模型单点预估**：<br>训练 X-Learner、Causal Forest 或 DR-Learner，对到访用户的特征向量 $X_i$ 打分，输出预测连续值：<br>$$\hat{\tau}(X_i) \in (-\infty, +\infty)$$ | **算法在线精准决策与受限预算最优化**：<br>1. **识别四象限人群**：过滤自然转化者（不发）与勿扰人群（严禁打扰）；<br>2. **单位补贴边际 ROI 贪心截断**：在有限预算内最大化增量回报。 |

---

##### 3. CATE 在工程中的端到端落地四步法（从模型打分到实时投放）

在实际工业界落地中，算法工程师如何把训练好的 CATE 模型转化为真实的线上收入？完整工程闭环分为四步：

1. **第一步：全量用户增量打分与人群四象限归类**
   用户请求到访时，特征服务实时提取特征向量 $X_i$，CATE 模型输出预估净增量 $\hat{\tau}(X_i)$。结合用户基线转化率 $P(Y=1 \mid X_i, T=0)$，系统可自动将用户划分入四象限：
   - **Persuadables（可说服人群，$\hat{\tau}(X_i) \gg 0$）**：本来在犹豫，发券后立刻买单 $\implies$ **重点发券干预群体**。
   - **Sure Things（自然转化人群，$\hat{\tau}(X_i) \approx 0, Y_i(0)=1$）**：本来就会全价购买的高粘性老客 $\implies$ **绝不下发补贴，截流净省预算**。
   - **Lost Causes（无动于衷人群，$\hat{\tau}(X_i) \approx 0, Y_i(0)=0$）**：对该类目毫无意向的沉睡用户 $\implies$ **不发券，避免浪费资源**。
   - **Sleeping Dogs（勿扰敏感人群，$\hat{\tau}(X_i) < 0$）**：发券弹窗或 Push 频繁打扰导致用户反感卸载或关闭通知 $\implies$ **加入免打扰黑名单**。

2. **第二步：离线 Decile 分桶单调性校验**
   在测试集上按 $\hat{\tau}(X_i)$ 降序排列分成 10 等份（Decile 1 到 10）。统计每个分桶内部真实实验组与对照组的实际转化率差 $\bar{Y}_{T=1} - \bar{Y}_{T=0}$。
   合格上线的 CATE 模型必须呈现**严格递减的单调性阶梯**：Decile 1 桶的实际净增量最高（如 +8%），Decile 10 桶必须接近 0 甚至是负数。

3. **第三步：受限预算下的边际 ROI 贪心截断（Marginal ROI Thresholding）**
   假设运营活动总预算为 $B$（如 50 万元），对用户 $i$ 发放补贴的面额成本为 $c(X_i)$（如 20 元），单笔订单平台预估抽佣毛利为 $v$。
   系统计算每个用户的单位补贴边际期望净回报：
   
   $$
   \text{Marginal\_ROI}_i = \frac{\hat{\tau}(X_i) \times v}{c(X_i)}
   $$
   
   将所有候选用户按照 $\text{Marginal\_ROI}_i$ 从高到低排序，从第一名开始顺次发放，直到累积成本 $\sum c(X_i) = B$ 耗尽预算。此时边界上最后一名用户的得分即为线上判定阈值 $\theta$。

4. **第四步：在线实时网关 Serving 拦截**
   在线请求进入推荐或补贴微服务时：
   
   $$
   \text{Decision}(X_i) = 
   \begin{cases} 
   \text{弹出 20 元神券弹窗 (Treatment)}, & \text{if } \frac{\hat{\tau}(X_i) \times v}{c(X_i)} \ge \theta \\
   \text{维持自然推荐 (Control)}, & \text{if } \frac{\hat{\tau}(X_i) \times v}{c(X_i)} < \theta
   \end{cases}
   $$

---

---

### 2. 因果效应可识别性的三大公理假设

为了使用历史观测数据或实验数据推断真实因果效应 $\tau(X)$，数据生成过程必须满足三大因果识别假设：

<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 20px 0;">
  <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: #ffffff; border-top: 4px solid #3b82f6;">
    <div style="font-weight: bold; font-size: 14px; color: #1e293b; margin-bottom: 8px;">1. 无混杂性 (Unconfoundedness)</div>
    <div style="font-family: monospace; font-size: 13px; color: #2563eb; background: #eff6ff; padding: 6px; border-radius: 4px; margin-bottom: 8px;">
      (Y(1), Y(0)) ⟂ T | X
    </div>
    <div style="font-size: 13px; color: #475569; line-height: 1.5;">
      亦称强可忽略性（Ignorability）。要求给定可观测特征 X 后，干预分配 T 与潜在结果完全独立。换言之，不存在同时操纵分配与结果的未观测隐变量（No Unobserved Confounder）。
    </div>
  </div>
  <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: #ffffff; border-top: 4px solid #10b981;">
    <div style="font-weight: bold; font-size: 14px; color: #1e293b; margin-bottom: 8px;">2. 重叠性 / 共同支撑域 (Overlap)</div>
    <div style="font-family: monospace; font-size: 13px; color: #059669; background: #ecfdf5; padding: 6px; border-radius: 4px; margin-bottom: 8px;">
      0 &lt; P(T=1 | X=x) &lt; 1, ∀x
    </div>
    <div style="font-size: 13px; color: #475569; line-height: 1.5;">
      对特征空间中的任意可能人群，接受与不接受干预的概率均严格介于 0 和 1 之间。若某类人群 P(T=1|X)=0（如某特定城市永不补贴），则无法推断该人群的干预反事实。
    </div>
  </div>
  <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; background: #ffffff; border-top: 4px solid #8b5cf6;">
    <div style="font-weight: bold; font-size: 14px; color: #1e293b; margin-bottom: 8px;">3. 稳定单元处理值 (SUTVA)</div>
    <div style="font-family: monospace; font-size: 13px; color: #7c3aed; background: #f5f3ff; padding: 6px; border-radius: 4px; margin-bottom: 8px;">
      Y_i(T_1,...,T_n) = Y_i(T_i)
    </div>
    <div style="font-size: 13px; color: #475569; line-height: 1.5;">
      ① <b>无个体间干扰（No Interference）</b>：个体 i 的结果不受个体 j 是否被干预影响；<br>
      ② <b>处理无隐式多版本（No Multiple Versions）</b>：同一干预标签代表完全均一的物理干预强度。
    </div>
  </div>
</div>

---

## 模块二：工业界实验设计与因果样本陷阱（RCT & Causal Pitfalls）

### 1. 依从性与分析样本界定（Compliance: ITT vs. As-Treated / CACE）

在工业推荐与营销 AB 实验中，算法拥有分流决策权，但无法强迫用户执行交互。此时产生**意向分流（Randomized Assignment, $Z$）**与**实际接受干预（Actual Treatment, $T$）**的割裂：

<div style="margin: 24px 0; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; background: #ffffff;">
  <div style="font-weight: bold; font-size: 14px; color: #1e293b; margin-bottom: 12px;">
    📐 因果拓扑图（DAG）：依从性断层与混杂偏倚的生成机理
  </div>
  <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-around; gap: 12px; padding: 12px 0;">
    <div style="border: 2px solid #3b82f6; border-radius: 6px; padding: 12px 16px; text-align: center; background: #eff6ff;">
      <div style="font-weight: bold; color: #1e40af;">分流意向 Z ∈ {0, 1}</div>
      <div style="font-size: 12px; color: #3b82f6;">服务端随机 Hash 落桶</div>
    </div>
    <div style="font-size: 20px; font-weight: bold; color: #94a3b8;">➔</div>
    <div style="border: 2px solid #f59e0b; border-radius: 6px; padding: 12px 16px; text-align: center; background: #fffbeb;">
      <div style="font-weight: bold; color: #92400e;">实际接受 T ∈ {0, 1}</div>
      <div style="font-size: 12px; color: #d97706;">用户实际曝光/点击/核销</div>
    </div>
    <div style="font-size: 20px; font-weight: bold; color: #94a3b8;">➔</div>
    <div style="border: 2px solid #10b981; border-radius: 6px; padding: 12px 16px; text-align: center; background: #ecfdf5;">
      <div style="font-weight: bold; color: #065f46;">转化结果 Y ∈ {0, 1}</div>
      <div style="font-size: 12px; color: #059669;">最终下单 / 留存时长</div>
    </div>
  </div>
  <div style="margin-top: 12px; border-top: 1px dashed #cbd5e1; padding-top: 12px; font-size: 13px; color: #475569; display: flex; align-items: center; gap: 8px;">
    <span style="color: #dc2626; font-weight: bold;">⚠️ 隐式混杂 U (用户内在购买意愿)</span>
    <span>同时作用于实际核销行为 <i>T</i> 与购买转化 <i>Y</i>，形成不可阻断的后门路径（Backdoor Path）！</span>
  </div>
</div>

#### 三大分析口径比较

1. **意向性分析（ITT, Intention-to-Treat）**：
   $$\text{ITT} = \mathbb{E}[Y \mid Z=1] - \mathbb{E}[Y \mid Z=0]$$
   - **机制**：完全按照算法底层的“分配决策 $Z$”划分样本，严格保留初始随机性。
   - **工业价值**：**系统决策的标准口径**。真实业务预算是按“发了多少曝光/券”结算的，ITT 准确评估了算法投放策略全链路的净拉动产出。
2. **实际接受分析（AT, As-Treated / Naive Comparison）及其致命缺陷**：
   $$\text{AT} = \mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]$$
   - **致命陷阱**：直接拿“真正领券用了的人”对比“没用券的人”。高价值活跃用户本就具备极高的核销意愿，使得 $T=1$ 人群自然转化率极高。这种对比引入了**严重的内生性自选择偏差（Selection Bias）**，导致干预效应被严重虚假高估！
3. **依从者局部平均处理效应（CACE / LATE）**：
   利用随机分流 $Z$ 作为实际干预 $T$ 的**工具变量（Instrumental Variable, IV）**，测算在真正受干预影响的依从群体（Compliers）上的真实效应：
   $$\text{CACE} = \frac{\text{ITT}_Y}{\text{ITT}_T} = \frac{\mathbb{E}[Y \mid Z=1] - \mathbb{E}[Y \mid Z=0]}{\mathbb{E}[T \mid Z=1] - \mathbb{E}[T \mid Z=0]}$$

---

### 2. 倾向得分（Propensity Score）与加权方差控制

在非完全随机的定向实验（如仅针对特定低频用户灰度发券）中，样本的干预概率随特征变化。
定义倾向得分（Propensity Score）为：

$$
e(X) = P(T=1 \mid X)
$$

利用**逆倾向得分加权（IPW, Inverse Probability Weighting）**可构建无偏估计量：

$$
\hat{\tau}_{\text{IPW}} = \frac{1}{N} \sum_{i=1}^N \left[ \frac{T_i Y_i}{e(X_i)} - \frac{(1 - T_i) Y_i}{1 - e(X_i)} \right]
$$

#### 方差爆炸与权重截断（Propensity Trimming / Clipping）
- **数学本质**：当某些特定特征人群被干预的概率极低（例如 $e(X_i) = 0.001$）时，权重 $\frac{1}{e(X_i)} = 1000$。单个孤立正样本就会让方差彻底失控，淹没全盘信号。
- **工业治理方案**：对倾向得分进行对称截断：
  $$\tilde{e}(X) = \max(\alpha, \min(1 - \alpha, e(X))), \quad \alpha \in [0.01, 0.05]$$

---

### 3. SUTVA 假设破坏与网络溢出治理（Interference & Spillover）

在工业推荐与双边市场中，个体间独立假设经常面临系统性击穿：
- **双边资源供给挤占（负溢出）**：网约车补贴实验中，实验组乘客打车需求暴增，抢空了同区域内的可用运力，导致对照组乘客排队时间拉长、取消率攀升。实验测得的 $\Delta$ 包含了对对照组的“掠夺偷跑”，导致全量推广后实际大盘增量大幅衰减。
- **社交网络传播（正溢出）**：短视频推荐算法中，实验组用户接收到趣味内容后转发到微信群，使得对照组用户提前感知并点击，稀释实验组的相对优势。

<div style="border-left: 4px solid #0284c7; background: #f0f9ff; padding: 12px 16px; border-radius: 4px; margin: 16px 0; font-size: 13px; color: #0369a1;">
  <b>🚀 工业级防溢出实验方案：</b><br>
  1. <b>集群随机分流（Cluster Randomization）</b>：以地理物理蜂窝网格（Uber H3 / 滴滴蜂窝网格）或社交图谱 Louvain 社区发现算法作为随机化单元，确保同一紧密集群内的交互全部落在同一实验组内。<br>
  2. <b>时间轮替实验（Switchback Experiments）</b>：全城所有司机与乘客在同一时刻处于同一策略下，以 30 分钟或 1 小时为时间切片交替轮转策略 A 与 B，切断物理时空上的资源争夺。
</div>

---

## 模块三：Uplift 算法族（Meta-Learners & Tree Approaches）

增量建模的核心任务是基于有限样本拟合 $\hat{\tau}(X)$。工业界主流形成了 Meta-Learners（元学习器体系）与 Causal Tree / Forest（因果树体系）两大路线。

> **🧩 五大 Uplift 核心建模范式架构全景矩阵**

| 算法范式 | 核心预估架构 | 目标因果效应 $\tau(X)$ 计算公式 | 工业界核心优势 | 工业界致命缺陷 / 风险 |
| :--- | :--- | :--- | :--- | :--- |
| **S-Learner**<br>(Single Learner) | 单模型 $\mu(X, T)$，预估 $\hat{\tau} = \mu(X, 1) - \mu(X, 0)$ | $\mu(X, T) = \mathbb{E}[Y \mid X, T]$<br>$\hat{\tau}(X) = \hat{\mu}(X, 1) - \hat{\mu}(X, 0)$ | 结构最简单，直接复用既有精排模型，训练成本最低 | **正则化偏倚**：树分裂或 L1/L2 极易将 1 维的 $T$ 吞噬归零，预估增量严重缩水偏向 0 |
| **T-Learner**<br>(Two Learners) | 独立训练 $\mu_1(X)$ 与 $\mu_0(X)$，预估 $\mu_1 - \mu_0$ | $\mu_1(X) = \mathbb{E}[Y \mid X, T=1]$<br>$\mu_0(X) = \mathbb{E}[Y \mid X, T=0]$<br>$\hat{\tau}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$ | 强制保留 $T$ 效应，不受特征正则化挤压 | **样本量失衡方差爆炸**：工业界实验组 $T=1$ 常仅占 5%~10%，$\mu_1(X)$ 严重欠拟合导致残差方差放大 |
| **X-Learner**<br>(Crossover Learner) | 两阶段交叉推补反事实残差，倾向得分自适应加权 | Stage 1: 反事实残差推补<br>$D_1 = Y_1 - \hat{\mu}_0(X_1)$<br>$D_0 = \hat{\mu}_1(X_0) - Y_0$<br>Stage 2: 倾向得分融合<br>$\hat{\tau}(X) = e(X)\hat{\tau}_0(X) + (1-e(X))\hat{\tau}_1(X)$ | **专治样本极度不平衡**（$P(T=1) \ll P(T=0)$），方差最小，大厂营销补贴首选 Meta-Learner | 需训练 4 个基模型 + 1 个倾向模型，离线离线训练与维护成本最高 |
| **DR-Learner**<br>(Doubly Robust) | 构造 AIPW 伪标签 $Y^{\text{DR}}$，直接回归拟合增量 | $Y^{\text{DR}} = \hat{\mu}_1(X) - \hat{\mu}_0(X) + \frac{T(Y - \hat{\mu}_1(X))}{e(X)} - \frac{(1-T)(Y - \hat{\mu}_0(X))}{1-e(X)}$<br>$\hat{\tau} = \arg\min_f \sum (Y_i^{\text{DR}} - f(X_i))^2$ | **双重稳健性 + Neyman 正交性**：响应模型 $\mu$ 或倾向模型 $e$ 任一正确即无偏；收敛速度达 $\sqrt{N}$ | 分母含倾向得分 $e(X)$，极端重叠度下伪标签有震荡离群点，需截断 |
| **Causal Forest**<br>(Honest Forest) | 诚实树分裂（Honest Splitting）自适应近邻匹配 | $\hat{\tau}(X) = \sum_{i=1}^n \alpha_i(X) Y_i$<br>权重 $\alpha_i(X)$ 由样本在同一叶节点的共现频率决定 | 局部非参数估计，具备严格渐近正态性与统计置信区间输出 | 特征维度极高（$D > 500$）或稀疏 ID 特征下树模型分裂退化，推理延迟较高 |

---

### 1. S-Learner（Single Learner）及其正则化偏倚

将干预状态 $T$ 仅仅视作一个普通输入特征，与所有用户特征 $X$ 拼接在一起：

$$\mu(X, T) = \mathbb{E}[Y \mid X, T]$$

模型预测出的增量为：

$$\hat{\tau}_{\text{S}}(X) = \mu(X, T=1) - \mu(X, T=0)$$

```text
[S-Learner 结构]
(Features X, Treatment T) ───> [ Single Model μ(X, T) ] ───> 预测 Y
增量计算: μ(X, 1) - μ(X, 0)
```

#### 致命缺陷：正则化偏倚（Regularization Bias）
在工业级特征工程中，$X$ 通常包含成百上千维用户特征（行为序列、离散 Embedding、统计频次）。而干预变量 $T$ 仅有 1 维。
当使用 GBDT（如 LightGBM）或带有 L1/L2 正则化的神经网络时，贪心分裂优先选择能大幅降低全局训练误差的特征，**1 维的 $T$ 很难在贪心策略下被选中分裂**。正则化惩罚会将 $T$ 的权重收缩压制，导致 $\mu(X, 1) \approx \mu(X, 0)$，模型的增量预估值系统性坍塌为 0！

---

### 2. T-Learner（Two Learners）及其样本不平衡方差

为了阻止干预信号被高维特征湮灭，T-Learner 彻底切断参数共享，训练两个独立的回归模型：
- 对照组模型：$\mu_0(X) = \mathbb{E}[Y \mid X, T=0]$，在对照组样本 $\{X_i, Y_i\}_{T_i=0}$ 上拟合；
- 实验组模型：$\mu_1(X) = \mathbb{E}[Y \mid X, T=1]$，在实验组样本 $\{X_i, Y_i\}_{T_i=1}$ 上拟合。

增量预测即为两模型之差：

$$\hat{\tau}_{\text{T}}(X) = \hat{\mu}_1(X) - \hat{\mu}_0(X)$$

```text
[T-Learner 结构]
T=0 样本 ───> [ Model μ₀(X) ] ───> 预测 Y(0)
                                        │
                                        ├───> 差值 τ = μ₁(X) - μ₀(X)
                                        │
T=1 样本 ───> [ Model μ₁(X) ] ───> 预测 Y(1)
```

#### 核心局限：
1. **两模型估计误差独立相加**：$\text{Var}(\hat{\tau}) = \text{Var}(\hat{\mu}_1) + \text{Var}(\hat{\mu}_0)$，两组误差没有协同抵消机制；
2. **样本极度不平衡时崩溃**：在工业营销中，补贴资源有限，发券用户可能仅占大盘的 3%（实验组 3 万，对照组 97 万）。此时 $\hat{\mu}_1(X)$ 在小样本上极度欠拟合或过拟合，估计方差极大。

---

### 3. X-Learner（Cross Learner）：针对样本不平衡的两阶段推补

由 Künzel et al. (2019) 提出，专为工业界**处理组与对照组样本量悬殊**场景设计的核心利器。

#### 算法执行四步曲

```text
第一阶段 (Base Models):
  训练基础模型 μ₀(X) 与 μ₁(X)

第二阶段 (Counterfactual Imputation & Residuals):
  实验组推补反事实: D₁ = Y₁ - μ₀(X₁)  (实验组在对照下的净增益真实残差)
  对照组推补反事实: D₀ = μ₁(X₀) - Y₀  (对照组在实验下的净增益真实残差)

第三阶段 (Imputed Effect Models):
  用 X₁ 拟合 D₁ 得到 τ₁(X)
  用 X₀ 拟合 D₀ 得到 τ₀(X)

第四阶段 (Propensity Weighted Fusion):
  τ(X) = e(X) · τ₀(X) + (1 - e(X)) · τ₁(X)
```

#### 为什么倾向得分加权在样本不平衡下具备极强鲁棒性？
考虑实验组极小的情形（$P(T=1) \to 0$，即 $e(X) \to 0$）：
- 此时 $1 - e(X) \approx 1$，加权公式几乎完全由 $\hat{\tau}_1(X)$ 决定；
- 而 $\hat{\tau}_1(X)$ 的目标变量 $D_1 = Y_1 - \hat{\mu}_0(X_1)$，其减数 $\hat{\mu}_0$ 是在**海量对照组（占 97% 样本）上精调拟合的高精度稳定模型**！
- 这一精妙的代数构造充分利用了大样本对照组学到的先验，使得小样本实验组的残差估计方差得到严格控制。

---

### 4. DR-Learner（Doubly Robust Learner / AIPW）

基于半参数效率理论中的 **增广逆倾向加权（AIPW, Augmented Inverse Probability Weighting）**。

#### 核心公式：伪标签（Pseudo-Outcome）的构造

首先训练倾向得分模型 $\hat{e}(X)$ 以及条件均值模型 $\hat{\mu}_0(X), \hat{\mu}_1(X)$。随后为每个样本构造一个具有 Neyman 正交性质的因果伪结果 $Y_i^{\text{DR}}$：

$$
Y_i^{\text{DR}} = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i (Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1 - T_i) (Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)}
$$

最后，直接以特征 $X_i$ 为输入，$Y_i^{\text{DR}}$ 为标量监督目标，训练任意无约束回归器（如 GBDT 或 MLP）拟合 $\hat{\tau}(X)$：

$$
\min_{\tau} \sum_{i=1}^N \left( Y_i^{\text{DR}} - \tau(X_i) \right)^2
$$

#### 双重稳健性（Double Robustness Property）的严谨数学证明

考察 $Y_i^{\text{DR}}$ 在真实数据生成过程下的条件数学期望 $\mathbb{E}[Y^{\text{DR}} \mid X]$：

$$
\begin{aligned}
\mathbb{E}[Y^{\text{DR}} \mid X] &= \hat{\mu}_1(X) - \hat{\mu}_0(X) \\
&\quad + \mathbb{E}\left[ \frac{T}{\hat{e}(X)} (Y - \hat{\mu}_1(X)) \;\middle|\; X \right] - \mathbb{E}\left[ \frac{1 - T}{1 - \hat{e}(X)} (Y - \hat{\mu}_0(X)) \;\middle|\; X \right]
\end{aligned}
$$

由于 $T \perp\!\!\perp (Y(1), Y(0)) \mid X$，且 $\mathbb{E}[T \mid X] = e(X)$（真实倾向得分）：

$$
\mathbb{E}\left[ \frac{T}{\hat{e}(X)} (Y - \hat{\mu}_1(X)) \;\middle|\; X \right] = \frac{e(X)}{\hat{e}(X)} \left( \mu_1(X) - \hat{\mu}_1(X) \right)
$$

同理对于对照部分：

$$
\mathbb{E}\left[ \frac{1 - T}{1 - \hat{e}(X)} (Y - \hat{\mu}_0(X)) \;\middle|\; X \right] = \frac{1 - e(X)}{1 - \hat{e}(X)} \left( \mu_0(X) - \hat{\mu}_0(X) \right)
$$

将两项代入展开合并后，其估计误差满足：

$$
\mathbb{E}[Y^{\text{DR}} \mid X] - \tau(X) = \left( \frac{e(X) - \hat{e}(X)}{\hat{e}(X)} \right) \left( \mu_1(X) - \hat{\mu}_1(X) \right) - \left( \frac{\hat{e}(X) - e(X)}{1 - \hat{e}(X)} \right) \left( \mu_0(X) - \hat{\mu}_0(X) \right)
$$

**双重保护结论立现：**
- **情形 1：倾向得分模型完美（$\hat{e}(X) = e(X)$）**：无论结果回归模型 $\hat{\mu}_0, \hat{\mu}_1$ 拟合得多么糟糕，误差项乘数恒为 0，$\mathbb{E}[Y^{\text{DR}} \mid X] = \tau(X)$！
- **情形 2：结果回归模型完美（$\hat{\mu}_0 = \mu_0, \hat{\mu}_1 = \mu_1$）**：无论倾向得分模型预测有多大偏差，括号内残差恒为 0，$\mathbb{E}[Y^{\text{DR}} \mid X] = \tau(X)$！
- **半参数收敛速度**：当两个模型各自以 $O(n^{-1/4})$ 的慢速非参数收敛时，二者误差的乘积达到 $O(n^{-1/2})$ 的极速参数渐近收敛率（Neyman 正交性）。

---

### 5. Causal Forest（因果森林与诚实切分，Athey & Imbens）

基于广义随机森林（Generalized Random Forests, GRF）框架，直接以**因果效应异质性**为优化目标分裂树节点。

#### 1. 节点分裂准则：最大化增量方差（Treatment Effect Heterogeneity）

传统 CART 树最小化预测残差方差 $\sum (Y - \bar{Y})^2$。而在因果森林中，节点分裂的目标是**拉大子节点之间因果效应的差距**。
设父节点 $P$ 尝试切分为左子节点 $L$ 与右子节点 $R$：

$$
\Delta(L, R) = \frac{N_L \cdot N_R}{N_P^2} \left( \hat{\tau}_L - \hat{\tau}_R \right)^2
$$

其中 $\hat{\tau}_L = \bar{Y}_{1, L} - \bar{Y}_{0, L}$ 为左子节点的均值差。通过最大化子节点间的因果方差，森林能自动聚类出对干预最敏感或最迟钝的特征细分人群。

#### 2. 诚实切分（Honest Splitting）架构设计

在经典机器学习中，使用同一批样本做树结构搜索与叶子节点参数评估会导致严重的自适应过拟合，且无法输出置信区间。因果森林引入样本独立划分机制：

> **🌲 因果森林：诚实切分（Honest Splitting）样本流动机制**
>
> * **子样本集 A: $S_{\text{split}}$（结构训练集）**：
>   仅用于计算切分指标 $\Delta(L, R)$，决定树在哪个特征、哪个阈值处分裂。树结构确定后，该集合样本立即被彻底丢弃。
> * **子样本集 B: $S_{\text{est}}$（效应估计集）**：
>   将未参与树分裂的独立干净样本落入叶子节点，在各个叶子内部计算真实因果效应：
>   $$\hat{\tau}_{\text{leaf}} = \bar{Y}_{1, \text{leaf}} - \bar{Y}_{0, \text{leaf}}$$
>
> 📌 **统计学证明**：由于叶子内的样本没有参与该叶子边界的形成（零信息泄露），根据中心极限定理，叶子估计量无偏，且渐近服从高斯正态分布，原生支持输出稳健的置信区间与 $p$ 值！

---

## 模块四：离线因果评估体系与业务决策闭环（Evaluation & Policy Value）

因为真实单样本反事实 $\tau_i$ 永不可见，传统的分类/回归指标（AUC、ROC、LogLoss、RMSE）在因果增量领域全部失效。工业界必须依赖**排序累积增益**与**分桶单调性检验**。

### 1. Uplift 专属排序指标：AUUC 与 Qini 曲线

将测试集样本按照模型预估增量 $\hat{\tau}(X)$ 降序排列。如果模型有效，排在头部的高分样本在干预下的净拉动必须显著高于尾部样本。

#### 1. 累积增益曲线（Cumulative Gain / Uplift Curve）

在排名前 $k$ 个样本的截断截面上，定义累积增量为：

$$
\text{Gain}(k) = \left( \frac{\sum_{i=1}^k Y_i \cdot T_i}{N_T(k)} - \frac{\sum_{i=1}^k Y_i \cdot (1 - T_i)}{N_C(k)} \right) \cdot (N_T(k) + N_C(k))
$$

- 曲线在坐标系中横轴为触达用户比例 $k/N \in [0, 1]$，纵轴为累积增量转化数；
- **AUUC（Area Under Uplift Curve）**：累积增益曲线下的积分面积。

#### 2. Qini 曲线与 Qini 系数（针对组间样本量不平衡的无偏修正）

在真实营销实验中，每个局部切片内实验组和对照组的人数很难保持严格等比。Radcliffe (2007) 提出以对照组为基准进行标准化缩放的 Qini 曲线：

$$
\text{Qini}(k) = \sum_{i=1}^k Y_i \cdot T_i - \left( \sum_{i=1}^k Y_i \cdot (1 - T_i) \right) \cdot \frac{N_T(k)}{N_C(k)}
$$

- **Qini 系数（Qini Score / Normalized Qini）**：
  $$Q = \frac{\text{AUUC}_{\text{model}} - \text{AUUC}_{\text{random}}}{\text{AUUC}_{\text{oracle}} - \text{AUUC}_{\text{random}}}$$
  将模型增益曲线与“随机均匀投放基线（斜线）”之间的面积，除以“完美因果神谕（Oracle）”超出随机基线的面积，取值介于 $[0, 1]$，是衡量增量排序保序度的第一金指标。

---

### 2. 分组检验（Decile Uplift Monotonicity Validation）

在工业落地验收中，算法工程师会输出模型在独立测试集上的 **10 分位桶图谱（Decile Chart）**：
1. 测试集样本按 $\hat{\tau}(X)$ 降序等分为 10 桶（Decile 1 为头部前 10% 用户，Decile 10 为尾部 10% 用户）；
2. 分别计算每个桶内部真实的实验组转化率与对照组转化率之差：
   $$\Delta \bar{Y}_k = \bar{Y}_{1, k} - \bar{Y}_{0, k}, \quad k \in \{1, 2, \dots, 10\}$$

<div style="margin: 24px 0; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; background: #ffffff;">
  <div style="background: #f8fafc; padding: 12px 16px; border-bottom: 1px solid #cbd5e1; font-weight: bold; font-size: 14px;">
    📈 理想工业 Uplift 模型：10 分位桶增量柱状分布图（Decile Uplift Distribution）
  </div>
  <div style="padding: 20px; display: flex; align-items: flex-end; justify-content: space-between; height: 180px; gap: 8px; border-bottom: 1px solid #e2e8f0;">
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+18.5%</div>
      <div style="width: 100%; height: 130px; background: #22c55e; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D1</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+14.2%</div>
      <div style="width: 100%; height: 100px; background: #4ade80; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D2</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+10.1%</div>
      <div style="width: 100%; height: 75px; background: #86efac; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D3</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+7.4%</div>
      <div style="width: 100%; height: 55px; background: #bbf7d0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D4</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #16a34a; margin-bottom: 4px;">+4.8%</div>
      <div style="width: 100%; height: 38px; background: #bbf7d0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D5</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #64748b; margin-bottom: 4px;">+2.3%</div>
      <div style="width: 100%; height: 22px; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D6</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #64748b; margin-bottom: 4px;">+0.8%</div>
      <div style="width: 100%; height: 12px; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D7</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #64748b; margin-bottom: 4px;">+0.1%</div>
      <div style="width: 100%; height: 5px; background: #e2e8f0; border-radius: 4px 4px 0 0;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D8</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #ef4444; margin-bottom: 4px;">-1.2%</div>
      <div style="width: 100%; height: 15px; background: #fca5a5; border-radius: 0 0 4px 4px; margin-top: 15px;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D9</div>
    </div>
    <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
      <div style="font-size: 11px; font-weight: bold; color: #b91c1c; margin-bottom: 4px;">-4.6%</div>
      <div style="width: 100%; height: 35px; background: #ef4444; border-radius: 0 0 4px 4px; margin-top: 35px;"></div>
      <div style="font-size: 11px; color: #64748b; margin-top: 6px;">D10</div>
    </div>
  </div>
  <div style="padding: 10px 16px; font-size: 12px; color: #475569; background: #f8fafc;">
    🔍 <b>合格判断两要素：</b>① 柱高从 D1 到 D10 呈<b>严格单调递减趋势（Monotonicity）</b>；② 成功在 D9/D10 处探底呈现负增益，证明模型成功挖掘出了“受打扰反感人群（Sleeping Dogs）”。
  </div>
</div>

---

### 3. Uplift 四象限人群分类与受限预算下的 ROI 决策闭环

根据个体潜在结果 $(Y(1), Y(0))$ 的不同组合，全量用户可在理论上被划分进四大象限：

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0;">
  <div style="border: 2px solid #22c55e; border-radius: 8px; padding: 14px; background: #f0fdf4;">
    <div style="font-weight: bold; color: #15803d; font-size: 14px;">🎯 1. 说服型人群 (Persuadables)</div>
    <div style="font-family: monospace; font-size: 12px; color: #166534; margin: 4px 0;">Y(1) = 1,  Y(0) = 0  ➔  τ(X) &gt; 0</div>
    <div style="font-size: 13px; color: #374151; line-height: 1.4;">
      不干预就不买，给了干预（如优惠券或强曝光）就下单。<br>
      <b>业务行动策略</b>：<b>核心投放主力人群，预算重点覆盖！</b>
    </div>
  </div>
  <div style="border: 2px solid #94a3b8; border-radius: 8px; padding: 14px; background: #f8fafc;">
    <div style="font-weight: bold; color: #475569; font-size: 14px;">☕ 2. 确定型人群 (Sure Things)</div>
    <div style="font-family: monospace; font-size: 12px; color: #475569; margin: 4px 0;">Y(1) = 1,  Y(0) = 1  ➔  τ(X) = 0</div>
    <div style="font-size: 13px; color: #374151; line-height: 1.4;">
      高忠诚老客或刚需用户，不给券也必然买，给券只会被其白白占便宜。<br>
      <b>业务行动策略</b>：<b>坚决不发券、不补贴，节省营销毛利！</b>
    </div>
  </div>
  <div style="border: 2px solid #cbd5e1; border-radius: 8px; padding: 14px; background: #f8fafc;">
    <div style="font-weight: bold; color: #64748b; font-size: 14px;">🧱 3. 无望型人群 (Lost Causes)</div>
    <div style="font-family: monospace; font-size: 12px; color: #64748b; margin: 4px 0;">Y(1) = 0,  Y(0) = 0  ➔  τ(X) = 0</div>
    <div style="font-size: 13px; color: #374151; line-height: 1.4;">
      非目标受众或完全休眠用户，无论怎么施加干预都不可能转化。<br>
      <b>业务行动策略</b>：<b>不浪费任何营销资源与推荐曝光槽位！</b>
    </div>
  </div>
  <div style="border: 2px solid #ef4444; border-radius: 8px; padding: 14px; background: #fef2f2;">
    <div style="font-weight: bold; color: #b91c1c; font-size: 14px;">⛔ 4. 反感型人群 (Sleeping Dogs / 沉睡者)</div>
    <div style="font-family: monospace; font-size: 12px; color: #991b1b; margin: 4px 0;">Y(1) = 0,  Y(0) = 1  ➔  τ(X) &lt; 0</div>
    <div style="font-size: 13px; color: #374151; line-height: 1.4;">
      平时安稳使用，一旦推送营销通知或频控过度，反而觉得被打扰而退订或卸载。<br>
      <b>业务行动策略</b>：<b>严格设立黑名单，绝对禁止打扰！</b>
    </div>
  </div>
</div>

#### 有限预算约束下的背包决策优化（Knapsack ROI Optimization）

在真实的推荐位变现或营销发券系统中，单个干预有确定的成本开销 $c(X)$（如面额 5 元的补贴券、短信通知费 0.05 元或 Push 挤占的疲劳度成本），业务总预算上限为 $B$。单次成功转化的期望商业毛利价值为 $V$。
决策变量为针对每个候选用户 $i$ 的二值决策 $a_i \in \{0, 1\}$：

$$
\max_{\{a_i \in \{0, 1\}\}} \sum_{i=1}^N a_i \cdot \Big[ V \cdot \hat{\tau}(X_i) - c(X_i) \Big] \quad \text{s.t.} \quad \sum_{i=1}^N a_i \cdot c(X_i) \le B
$$

#### 工业贪心近似解法：单位成本增量比（Incremental ROI Ranking）

当用户量 $N$ 达到千万至亿级时，上述 0-1 背包问题直接通过单位边际回报比贪心降序排序求解：

$$
\text{Priority}(X_i) = \frac{\hat{\tau}(X_i)}{c(X_i)}
$$

1. 过滤掉所有 $\hat{\tau}(X_i) \le 0$ 的非正向人群（直接剔除 Sleeping Dogs、Lost Causes、Sure Things）；
2. 过滤掉单位收益无法覆盖成本的人群：$V \cdot \hat{\tau}(X_i) < c(X_i)$；
3. 将剩余候选人群按照 $\frac{\hat{\tau}(X_i)}{c(X_i)}$ 严格降序排列，自上而下顺次分配干预，直至总预算 $B$ 消耗完毕；
4. 截断阈值处的边际 ROI 满足：
   $$\text{Marginal ROI} = \frac{V \cdot \hat{\tau}(X_{\text{cutoff}})}{c(X_{\text{cutoff}})} \ge 1$$

---

## 模块五：生产级纯 Python 实现参考（Production Code Blueprint）

以下给出轻量级、自包含且无外部重型黑盒依赖的工业级 X-Learner 与离线增量评估器实现：

```python
import numpy as np


class XLearner:
    """
    X-Learner 实现：两阶段残差推补与倾向得分融合
    基学习器可任意替换为 Ridge、LightGBM 或 MLP
    """
    def __init__(self, base_model_cls, **model_params):
        self.base_model_cls = base_model_cls
        self.model_params = model_params
        # 第一阶段基模型
        self.m0 = base_model_cls(**model_params)
        self.m1 = base_model_cls(**model_params)
        # 第二阶段残差模型
        self.tau0 = base_model_cls(**model_params)
        self.tau1 = base_model_cls(**model_params)
        self.propensity_model = None

    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray, p: np.ndarray = None):
        """
        X: (N, D) 特征矩阵
        T: (N,) 干预标签 0/1
        y: (N,) 观测指标
        p: (N,) 可选先验倾向得分，若无则使用样本均值兜底
        """
        idx_0 = np.where(T == 0)[0]
        idx_1 = np.where(T == 1)[0]

        X0, y0 = X[idx_0], y[idx_0]
        X1, y1 = X[idx_1], y[idx_1]

        # Stage 1: 拟合结果均值基模型
        self.m0.fit(X0, y0)
        self.m1.fit(X1, y1)

        # Stage 2: 反事实残差推补 (Counterfactual Imputation)
        # 对实验组: 用 m0 预测其对照态反事实，残差 D1 = y1 - m0(X1)
        D1 = y1 - self.m0.predict(X1)
        # 对对照组: 用 m1 预测其实验态反事实，残差 D0 = m1(X0) - y0
        D0 = self.m1.predict(X0) - y0

        # Stage 3: 拟合残差模型
        self.tau1.fit(X1, D1)
        self.tau0.fit(X0, D0)

        # 记录或估算倾向得分
        if p is not None:
            self.fixed_p = p
        else:
            self.fixed_p = float(len(idx_1)) / len(T)

    def predict(self, X: np.ndarray, p: np.ndarray = None) -> np.ndarray:
        """
        Stage 4: 倾向得分加权组合
        """
        tau0_pred = self.tau0.predict(X)
        tau1_pred = self.tau1.predict(X)

        propensity = p if p is not None else self.fixed_p
        # 倾向得分加权: e(X)*tau0 + (1-e(X))*tau1
        return propensity * tau0_pred + (1.0 - propensity) * tau1_pred


class UpliftEvaluator:
    """
    Uplift 离线综合评估器：计算 Decile 单调性、AUUC 与 Qini 系数
    """
    @staticmethod
    def evaluate_deciles(y_true: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray, n_bins: int = 10):
        N = len(y_true)
        # 按预估增量降序排序
        order = np.argsort(-uplift_score)
        y_sorted = y_true[order]
        t_sorted = treatment[order]

        bin_size = N // n_bins
        decile_results = []

        for b in range(n_bins):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_bins - 1 else N
            
            y_b = y_sorted[start:end]
            t_b = t_sorted[start:end]
            
            n_t = np.sum(t_b == 1)
            n_c = np.sum(t_b == 0)
            
            y1_mean = np.sum(y_b[t_b == 1]) / n_t if n_t > 0 else 0.0
            y0_mean = np.sum(y_b[t_b == 0]) / n_c if n_c > 0 else 0.0
            lift = y1_mean - y0_mean
            
            decile_results.append({
                "decile": b + 1,
                "n_treated": int(n_t),
                "n_control": int(n_c),
                "mean_treated": float(y1_mean),
                "mean_control": float(y0_mean),
                "lift": float(lift)
            })

        return decile_results

    @staticmethod
    def calculate_qini_score(y_true: np.ndarray, treatment: np.ndarray, uplift_score: np.ndarray) -> float:
        """
        计算标准归一化 Qini 系数: (AUUC_model - AUUC_rand) / (AUUC_oracle - AUUC_rand)
        """
        order = np.argsort(-uplift_score)
        y_s = y_true[order]
        t_s = treatment[order]

        total_t = np.sum(t_s == 1)
        total_c = np.sum(t_s == 0)

        # 累积实验组正例与对照组正例
        cum_y1 = np.cumsum(y_s * t_s)
        cum_y0 = np.cumsum(y_s * (1 - t_s))

        # Qini 累积曲线
        qini_curve = cum_y1 - cum_y0 * (total_t / total_c if total_c > 0 else 1.0)
        qini_model_area = np.sum(qini_curve)

        # 随机基线面积
        qini_random_area = qini_curve[-1] * len(y_s) / 2.0

        return float(qini_model_area - qini_random_area)
```

---

## 模块六：面试高频追问与决策速查总表

| 考察维度 | 典型面试追问 | 核心底层机制与标准答题策略 |
| :--- | :--- | :--- |
| **理论基石** | 为什么不能直接拿实际领券用户的点击率做监督训练？ | **内生性自选择偏差**：领券行为受用户内在转化意愿 $U$ 驱动，直接比较混杂了用户自身偏好与发券的真实激励，造成因果效应严重虚高。必须坚持 ITT 口径或使用分流标识作为工具变量（IV）。 |
| **算法选型** | 在优惠券补贴场景（发券组 5%，未发券组 95%），选 T-Learner 还是 X-Learner？为什么？ | **坚决选择 X-Learner**：T-Learner 的发券组模型样本极小、方差极大；X-Learner 通过两阶段残差推补，利用海量对照组训练的高精度基模型作为反事实锚点，辅以倾向得分加权，方差得到严格控制。 |
| **因果森林** | Causal Forest 的“诚实切分（Honest Splitting）”为什么能产生置信区间？ | 结构分裂集 $S_{\text{split}}$ 与叶子估计集 $S_{\text{est}}$ 物理隔离。叶子内的样本没有参与该叶子空间边界的选择过程，消除了自适应过拟合，依据中心极限定理（CLT）估计量服从渐近正态分布。 |
| **业务决策** | 离线 Qini 表现很好，线上投放为什么经常达不到预期收益？ | 检查三大现实陷阱：① **SUTVA 击穿**：双边运力或库存被实验组争抢挤占，对照组被负向吸血；② **倾向得分漂移**：线上灰度人群与离线训练人群特征重叠度不足；③ **非依从稀释**：实际曝光率/触达率过低导致 ITT 稀释。 |
| **评估指标** | 为什么在营销场景必须坚决检查 Decile 尾部是否为负？ | 识别 **Sleeping Dogs（反感型用户）**。若尾部出现显著负增益，证明过度触达诱发了用户反感与流失，必须在生产线上对尾部桶执行强制拦截黑名单。 |

# ML Coding 00 · ML 基础：数据预处理、数据泄露与经典损失函数

在机器学习系统设计与算法工程实践中，扎实的统计学基础与严密的数据管道工程是构建高可用模型的基石。许多模型在离线评测中指标优异，上线后效果却断崖式下跌，其根源往往不在于复杂的模型架构，而在于数据泄露（Data Leakage）、不恰当的缺失值处理（Missing Data Imputation）、样本不平衡的评估陷阱或对损失函数（Loss Functions）统计假设的认知偏差。

本篇系统梳理工业界机器学习基础与工程落地的 6 大核心模块：
1. **数据泄露（Data Leakage）机理与全方位防御体系**
2. **缺失值机制（MCAR / MAR / MNAR）与处理策略权衡**
3. **样本不平衡处理体系、表征学习与 VAE 核心价值**
4. **分类、排序、校准与业务评估指标全景体系（含可折叠代码实现）**
5. **经典损失函数推导：线性回归 vs 逻辑回归，MSE vs MAE 及统计学收敛特性**
6. **核心机制辨析与高频问题清单**

---

## 模块一：数据泄露（Data Leakage）机理与防御体系

### 1. 数据泄露的本质与危害

数据泄露（Data Leakage）是指**在模型训练过程中，非预期地引入了训练集外部的信息（尤其是目标变量或未来测试数据）**。

```text
数据泄露生命周期与危害：
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ 训练/离线验证阶段       │ ───> │ 离线指标虚假繁荣        │ ───> │ 生产线上真实部署        │
│ 意外窥探未来/目标信息   │      │ 验证集 AUC 0.98+        │      │ 无法获取泄露特征        │
│ 产生虚假强相关性特征    │      │ (Overly Optimistic)     │      │ 线上效果断崖式崩塌 💥   │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

数据泄露会导致严重的过拟合与“虚假繁荣”——模型在训练集和验证集上表现完美，但由于泄露的信息在真实的生产推理环境中根本不存在，模型在线上部署时性能会发生灾难性衰退。

---

### 2. 四大高频数据泄露场景与典型案例

#### 场景 1：目标泄露 / 代理特征（Target Leakage / Proxy Features）

**核心机制**：特征本身是在**目标事件发生之后**才被生成、更新或记录的，但在离线回溯构建样本时被误作为输入特征。

- **典型案例 1（贷款违约预测）**：用“账户注销日期（`account_closed_date`）”或“催收退款状态码（`refund_status_code`）”来预测用户是否会违约。在现实业务流中，只有用户发生违约并进入催收流程后，这些字段才会被写入数据库。
- **典型案例 2（疾病诊断）**：在预测患者是否患有某种罕见病时，把“是否开具了该病的专属处方特效药（`prescribed_treatment_drug`）”作为特征。医生是在确诊后才开药的，将其作为预测特征属于本末倒置。

#### 场景 2：预处理泄露 / 全局统计量污染（Preprocessing Leakage）

**核心机制**：在划分训练集/测试集之前，在**全量数据集（Global Dataset）**上统一计算了全局统计量并完成了数据转换。

- **典型案例 1（特征缩放与归一化）**：在 `train_test_split` 之前，直接对全量数据调用 `StandardScaler().fit_transform(X)`。测试集的均值和方差提前渗透进了训练集，导致测试集分布发生信息外泄。
- **典型案例 2（文本特征词表与 TF-IDF）**：在全量语料上拟合 `TfidfVectorizer`，使得词表（Vocabulary）和逆文档频率（IDF）包含了测试集的信息。
- **典型案例 3（高基数目标编码 Target Encoding）**：在没有按折（Out-of-Fold）隔离的情况下，直接用全量数据的目标均值替换类别特征，导致模型直接“背诵”了测试集的目标分布。

#### 场景 3：时间序列的时间泄露（Temporal / Look-Ahead Leakage）

**核心机制**：用“未来时间戳”的数据来预测“过去”发生的事件，破坏了时序数据的因果律（Causality）。

- **典型案例 1（金融量化 / 股票预测）**：使用未来 5 天的滚动移动平均线（Rolling SMA centered）作为今日交易信号的特征。
- **典型案例 2（错误的交叉验证切分）**：对时序/用户行为日志采用随机 K 折交叉验证（Random K-Fold）。第 1 天的测试样本可能被第 5 天的训练样本“剧透”，完全掩盖了概念漂移（Concept Drift）与时序因果性。

#### 场景 4：样本组 / 重复实体泄露（Group / Duplication Leakage）

**核心机制**：属于**同一个实体（Entity / Subject）**的多条强相关或重复样本，被随机拆分到了训练集与测试集两端。

- **典型案例 1（医学图像诊断）**：同一位患者拍摄了 10 张不同角度的胸透 CT 切片。如果随机划分，该患者的 8 张切片在训练集，2 张在测试集。卷积神经网络可能会记住该患者独特的骨骼阴影或设备伪影，而不是泛化的病理特征。
- **典型案例 2（多会话用户推荐）**：同一用户在同一天内的 20 次点击行为被随机分散到训练和测试集中。

---

### 3. 工业级数据泄露防御策略

| 防御策略 | 核心实施要点 | 关键工具 / 库支持 | 解决的泄露类型 |
|---|---|---|---|
| **先拆分，后拟合（Split First, Fit Later）** | 必须在数据集划分后，仅在训练集上调用 `fit()`，测试集仅调用 `transform()`。严禁在切分前做全局缩放或插补。 | `sklearn.pipeline.Pipeline`, `ColumnTransformer` | 预处理泄露 |
| **时序前向链式切分（Time-Based Splitting）** | 严格基于时间戳排序，仅使用历史时间窗口预测未来，使用滚动切分而非随机打乱。 | `TimeSeriesSplit`, `PurgedGroupTimeSeriesSplit` | 时间泄露 |
| **实体分组隔离（Group-Aware Splitting）** | 确保同一患者、同一设备或同一用户的所有数据严格锁定在单侧（同在训练集或同在测试集）。 | `GroupKFold`, `GroupShuffleSplit`, `StratifiedGroupKFold` | 实体分组泄露 |
| **推理时间线可用性审计（Inference Timeline Audit）** | 针对每个特征严格提问：“在生产环境发起预测请求的毫秒瞬间，该字段在数据库中是否已经生成并可用？” | 特征元数据注册表（Feature Store 如 Feast）、数据血缘系统 | 目标与代理特征泄露 |

---

### 4. Quick Coding：防泄露 Pipeline 与 GroupKFold 实战

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. 模拟生成带有缺失值、分组实体的数据
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
groups = np.repeat(np.arange(100), 10)  # 100 个独立用户，每个用户 10 条记录
X[np.random.rand(*X.shape) < 0.1] = np.nan  # 注入 10% 缺失值

# 2. 构建严密的防泄露管道 (Pipeline 封装 Imputer + Scaler + Model)
# 管道确保所有转换步骤仅在每一折的训练集上 fit，绝不窥探测试集
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42))
])

# 3. 使用 GroupKFold 确保同一用户数据不跨折泄露
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # 核心：fit 仅接触当前折的训练数据
    model_pipeline.fit(X_train, y_train)
    oof_preds[val_idx] = model_pipeline.predict_proba(X_val)[:, 1]

cv_auc = roc_auc_score(y, oof_preds)
print(f"严格防泄露 GroupKFold 5-Fold OOF AUC: {cv_auc:.4f}")
```

---

## 模块二：缺失值处理策略与统计权衡（Handling Missing Data）

### 1. 三大统计缺失机制（Missingness Mechanisms）

统计学家 Rubin 将数据缺失机制划分为以下三类：

```text
数据缺失机制分类：
┌──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ 缺失机制类别                         │ 统计学数学定义与核心特征                                               │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. 完全随机缺失 (MCAR)               │ P(M | Y_obs, Y_mis) = P(M)                                             │
│    Missing Completely at Random      │ 缺失与任何已观测或未观测变量均无关（如传感器偶然丢包、问卷纸张偶发破损）│
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. 随机缺失 (MAR)                    │ P(M | Y_obs, Y_mis) = P(M | Y_obs)                                     │
│    Missing at Random                 │ 缺失倾向依赖于其他已观测特征，但与缺失值本身无关（如老年人更少填写手机号│
│                                      │ 但在已知年龄的情况下，手机号缺失概率与手机号本身取值无关）             │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. 非随机缺失 (MNAR)                 │ P(M | Y_obs, Y_mis) 依赖于 Y_mis 本身                                  │
│    Missing Not at Random             │ 缺失本身携带强烈的未观测业务信号（如超高收入者或极低收入者更倾向于拒填│
│                                      │ 收入字段，缺失事实本身具有极高信息量）                                 │
└──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 2. 五大缺失值处理策略综合对比表

| 处理策略 | 适用场景 | 优势（Pros） | 劣势与权衡（Cons / Trade-offs） |
|---|---|---|---|
| **行/列直接删除（Listwise / Column Deletion）** | MCAR 机制且缺失率极低（<3%~5%）；或整列缺失率超过 80%~90%。 | 实现极简；若符合 MCAR 则不会引入人为合成的分布偏差。 | 严重损失样本量；若实际为 MAR 或 MNAR 会导致剧烈的**样本选择偏差（Selection Bias）**。 |
| **简单统计量填充（Mean / Median / Mode）** | 快速基线；数值型（中位数抗偏态）或类别型（众数/常量）；缺失率较低。 | 计算开销极低；在线实时推理部署成本低，易于持久化。 | **扭曲特征原有分布**，人为低估特征方差，完全破坏变量之间的协方差与相关性。 |
| **缺失指示变量（Missing Indicator: `is_missing`）** | MNAR 场景；“缺失这一事实本身”具有极强业务预测信号（如用户跳过可选收入填报）。 | 保留了“缺失行为”所蕴含的原生业务信号。 | 若盲目应用于所有特征会导致特征维度翻倍；可能在共线性与稀疏度上引入挑战。 |
| **基于模型的插补（Model-Based: KNN, MICE / IterativeImputer, MissForest）** | 特征间存在复杂的非线性交互；中等规模的高价值表格数据集。 | 充分保留特征间的多变量相关性、协方差与方差分布。 | 计算复杂度高；在线推理部署困难（需加载插补模型）；存在多级误差级联风险。 |
| **树模型原生默认路径路由（Native Tree Handling）** | LightGBM, XGBoost, CatBoost 等基于决策树的梯度提升模型。 | 无需手工插补；树分裂时通过评估将缺失值分配到左/右子树的最优增益自动选择默认路由。 | 仅限特定树模型使用；无法直接推广到线性模型、SVM 或深度神经网络。 |

---

### 3. 统计学深度权衡剖析

1. **方差收缩与分布扭曲（Variance Shrinkage）**：
   若使用均值填充 $x_{\text{imputed}} = \bar{x}$，填充后的样本方差计算为：

$$\text{Var}(X_{\text{imputed}}) = \frac{N_{\text{obs}}}{N_{\text{total}}} \text{Var}(X_{\text{obs}}) < \text{Var}(X_{\text{obs}})$$

   人为压低了特征方差，使后续基于方差的特征选择或线性模型权重估计产生统计偏差。
2. **多重插补（MICE: Multivariate Imputation by Chained Equations）的优势**：
   通过链式方程循环回归，针对每个缺失特征以其他特征作为自变量进行多轮迭代建模预测，并注入适度扰动残差，从而真实还原特征间的相关矩阵。

---

### 4. Quick Coding：带 Missing Indicator 的鲁棒插补 Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 1. 构造包含 MNAR 信号的样本数据
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, np.nan, 60],
    'income': [50000, np.nan, 120000, 80000, np.nan, 200000, 95000],  # 高收入倾向于缺失 (MNAR)
    'credit_score': [650, 700, 750, 680, 710, 790, 720]
})
y = np.array([0, 1, 0, 1, 0, 1, 0])

# 2. 构建组合插补器：同时获得 (中位数填充值 + 缺失二值指示标记)
numeric_features = ['age', 'income', 'credit_score']

numeric_transformer = FeatureUnion([
    ('imputed_features', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])),
    ('missing_indicators', MissingIndicator())  # 自动提取布尔标志列
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)]
)

full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', Ridge())
])

full_pipeline.fit(df, y)
print("Pipeline 训练完成，转换后特征维度（含 is_missing 指示列）:", 
      full_pipeline.named_steps['preprocess'].transform(df).shape)
```

---

## 模块三：样本不平衡处理体系、表征学习与 VAE 核心价值

### 1. 不平衡数据处理的三个层级

在实际机器学习系统中，处理类别不平衡主要分布在三个层级：

* **数据层**：
  * **欠采样（Under-sampling）**：Random Under-sampling、Tomek Links（识别并移除异类最近邻对中的多数类以清晰类别分界）、ENN（Edited Nearest Neighbours，清理边界噪点）；
  * **过采样（Over-sampling）**：SMOTE（在特征空间通过 $k$ 近邻线性插值合成少数类样本）、ADASYN（根据少数类样本周围多数类的密集程度自适应分配插值权重）；
  * **针对性特征/样本增强**：长尾特征扰动与数据扩增。
* **算法与损失层**：
  * **代价敏感加权（Cost-sensitive / Class Weights）**：在损失函数中按类别频次反比赋予样本权重 $w_c \propto \frac{1}{N_c}$；
  * **聚焦损失（Focal Loss）**：引入调制因子 $(1 - p_t)^\gamma$，自适应衰减易分类样本对梯度的贡献；
  * **任务重构为单分类或异常检测**：One-Class SVM、Isolation Forest，避开极度不平衡的有监督分类直接学习正常样本分布支持集。
* **决策与后处理层**：
  * **动态阈值微调（Threshold Moving / Threshold Tuning）**：不直接采用 0.5 默认截断，依据验证集上的特定业务效用函数（如最大化 $F_\beta$ 或收益总和）搜索最优决策阈值；
  * **概率校准（Platt Scaling, Isotonic Regression）**：纠正因采样或加权导致的输出后验概率偏离真实经验发生率的问题；
  * **业务容量限制下的 Top-$k$ 截断**：按预测概率从高到低排序，仅截取前 $k$ 个最高风险/收益样本进入下游执行流。

---

### 2. 为什么严重不平衡有时对业务“没关系”？

1. **ROC-AUC 的排序不变性**：
   ROC-AUC 的统计学本质是 Wilcoxon-Mann-Whitney 统计量，等价于从正负样本中各随机抽取一个样本，正样本预测概率大于负样本预测概率的先验期望：
   $$ \text{AUC} = P(S^+ > S^-) $$
   排序关系仅取决于条件分布 $P(X \mid Y=1)$ 与 $P(X \mid Y=0)$ 在投影方向上的可分离度，在数学上完全独立于类别先验概率 $P(Y)$。即便负样本数量增加数倍，只要正负样本内部的分数分布未变，ROC-AUC 保持数学恒定。
2. **决策场景只依赖 Top 排序**：
   在推荐系统召回排序、量化多因子选股或风控初筛中，业务逻辑往往是选取固定容量的头部样本（如每日做多 Top 1% 股票，或人工审核前 500 笔可疑交易）。此时只要模型对头部的相对排序准确，绝对先验概率的偏移不影响最终决策集的构成。
3. **ROC-AUC 的“虚假繁荣”陷阱与 PR-AUC 边界**：
   虽然 ROC-AUC 对先验不敏感，但在极度不平衡下（如正例比例为 0.1%），假阳率公式为：
   $$ \text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}} $$
   庞大的 $\text{TN}$ 会极度稀释 $\text{FPR}$ 的分母。模型即便产生数千个误报（$\text{FP}$ 远超 $\text{TP}$），$\text{FPR}$ 依然极低，表现为 ROC-AUC 高达 0.98，但线上真实精确率（Precision）可能不足 5%。这也是为何实际反欺诈与故障诊断更推荐 **PR-AUC (Average Precision)**。

---

### 3. 对比学习 (SupCon)、SMOTE 与 Focal Loss 机制对比

| 方法 | 基本思想 | 适用范围 | 局限与边界 |
|---|---|---|---|
| **SMOTE** | 在特征空间中寻找少数类样本的 $k$ 近邻，通过线性插值合成新样本：$x_{\text{new}} = x + \lambda(x_{nn} - x)$。 | 中低维结构化表格数据；少数类分布连续且无大量边界重叠的场景。 | 高维稀疏特征下失效（维数灾难）；会盲目插值噪声与离群点，加剧类别混淆。 |
| **Focal Loss** | 在交叉熵损失上引入动态衰减因子 $(1 - p_t)^\gamma$，自适应降低易分类样本对梯度的贡献，强迫网络聚焦于难分样本。 | 密集预测（如目标检测）、高容量神经网络、样本极度不平衡且不希望修改采样率的端到端训练。 | 对标签噪声（Label Noise）极度敏感，因错误标注的样本天然会被视作“极难样本”而赋予极高权重。 |
| **对比学习 (SupCon)** | 利用样本间成对约束，拉近同类嵌入距离、推远异类嵌入距离，学习具有判别性的低维几何流形。 | 高维复杂表征学习（文本、时序、图表征）；长尾分布（Long-tailed recognition）；少样本冷启动。 | 训练开销大（依赖大 Batch Size 或 Memory Bank），需要精细构建正负样本对与表征投影头。 |

---

### 4. 变分自编码器（VAE）在噪声、不平衡与低信噪比数据中的价值

在量化金融高频时序与强噪声表格建模中，数据通常具有极低信噪比（$\text{SNR} < 0.05$）与非平稳性。变分自编码器（VAE，包括 CVAE、Bottleneck Autoencoder）展现出以下核心价值：

* **隐式高斯扰动与去噪正则化**：
  传统 Autoencoder 或 MLP 容易记忆微观结构中的高频随机噪声导致过拟合。VAE 通过将输入编码为均值 $\mu$ 与方差 $\sigma$，并引入重参数化技巧（Reparameterization Trick）采样：
  $$ z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) $$
  强制隐空间满足高斯先验 $\mathcal{N}(0, I)$。这种随机扰动相当于在隐特征上施加连续的数据增强，迫使解码器或下游预测头只关注宏观拓扑流形，过滤高频噪声。
* **非线性宏观状态因子（Market Regime）抽取**：
  复杂系统往往由少数不可观测的潜变量驱动（如系统状态切换、流动性枯竭）。线性主成分分析（PCA）无法捕获跨维度的非线性协同交互。VAE 的低维 Bottleneck $z$ 能提取正交、连续的非线性因子，作为下游树模型（LightGBM）或时序模型（Transformer）的稳健输入。
* **多任务联合预训练（End-to-End Joint Loss）**：
  将 VAE 特征重构损失与下游任务预测损失联合端到端优化：
  $$ \mathcal{L} = \mathcal{L}_{\text{prediction}}(y, \hat{y}) + \lambda_1 \mathcal{L}_{\text{recon}}(x, \hat{x}) + \lambda_2 D_{\text{KL}}(q(z \mid x) \parallel p(z)) $$
  无监督重构项充当强正则化约束，防止模型参数过早坍缩到局部假相关（Spurious Correlation）中。
* **不平衡与异常检测**：
  极端异常模式（系统崩盘、离群故障）在历史样本中极度稀缺。基于 VAE 的重构误差 $\|x - \hat{x}\|_2^2$ 或边缘似然估计，可以直接作为样本的“异常度评分”（Anomaly Score），用于头寸保护或风险兜底。

<details>
<summary><b>实现代码与解析：带下游联合预测头的 VAE 完整架构（PyTorch）</b></summary>

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TabularVAEWithJointHead(nn.Module):
    """带下游联合预测头的变分自编码器架构"""
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        # 编码器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器 (Decoder: 特征重构去噪)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 联合预测头 (Prediction Head: 下游回归/分类任务)
        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧：z = mu + sigma * eps"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        pred_y = self.pred_head(z)
        return recon_x, pred_y, mu, logvar

def compute_joint_vae_loss(
    x: torch.Tensor, recon_x: torch.Tensor, 
    y_true: torch.Tensor, y_pred: torch.Tensor, 
    mu: torch.Tensor, logvar: torch.Tensor,
    lambda_recon: float = 1.0, lambda_kl: float = 0.01
):
    """端到端联合优化损失计算：预测 MSE + 重构 MSE + KL 散度约束"""
    pred_loss = F.mse_loss(y_pred.squeeze(-1), y_true)
    recon_loss = F.mse_loss(recon_x, x)
    # KL 散度：-0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    
    total_loss = pred_loss + lambda_recon * recon_loss + lambda_kl * kl_loss
    return total_loss, {
        "pred_loss": pred_loss.item(),
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item()
    }
```
</details>

---

## 模块四：分类、排序、校准与业务评估指标全景体系

```ml-metrics-demo
```

### 1. 十大常用评估指标全景比对矩阵

| 指标类型 | 指标名称 | 核心定义 / 计算公式 | 核心适用场景 | 盲区与潜在陷阱 |
|---|---|---|---|---|
| **排序类（阈值无关）** | **ROC-AUC** | TPR 对 FPR 的积分曲线下宽度，统计本质为 $P(S^+ > S^-)$。 | 评估模型全局分离能力；类别分布相对稳定或需要与先验解耦的模型横向对比。 | 负样本巨大时对假阳率钝化，在极度不平衡下易产生虚假繁荣。 |
|  | **PR-AUC (AP)** | Precision 对 Recall 积分曲线下宽度，加权阶梯面积：$\sum (R_k - R_{k-1})P_k$。 | **极度不平衡分类（反欺诈、违约预测、故障排查）核心指标**；聚焦于正样本检出率与查准率。 | 对负样本纯度不敏感；若无统一基准线（随机猜测基准为正类先验比例 $\pi = P(Y=1)$），跨数据集难以横向比较。 |
| **决策类（阈值相关）** | **F1-Score / $F_\beta$** | $F_\beta = (1 + \beta^2)\frac{P \cdot R}{\beta^2 P + R}$，调和平均数。 | 单一阈值上线决策；根据业务诉求微调检出偏好（如漏报代价高设 $\beta=2$）。 | 强依赖所选固定截断阈值；未考虑不同预测概率区间的风险分布。 |
|  | **Precision@k / Recall@k** | 预测置信度最高的前 $k$ 个样本中的查准率或查全率。 | 生产端具有严格吞吐上限（如人工复审团队每日限额、推荐系统前 $k$ 位展现）。 | 仅衡量头部排序质量，对 $k$ 之后的长尾分布完全盲区。 |
|  | **Balanced Accuracy** | $\frac{\text{TPR} + \text{TNR}}{2} = \frac{\text{Recall}_{\text{pos}} + \text{Recall}_{\text{neg}}}{2}$。 | 需要兼顾每一个类别的准确性，避免模型全部预测为多数类。 | 对极端分类器容易给出钝化评分，忽略了正负类在业务侧的不对称成本。 |
| **概率质量与校准** | **Log-Loss (Cross-Entropy)** | $-\frac{1}{N}\sum [y \ln p + (1-y)\ln(1-p)]$。 | 概率预测敏感任务（如点击率预估 CTR、期望收益定价）。 | 极易受高置信度错误分类的剧烈惩罚；受类别不平衡先验漂移影响极大。 |
|  | **Brier Score** | $\frac{1}{N}\sum (p_i - y_i)^2$，概率空间的均方误差。 | 衡量校准质量；可严格分解为可靠性（Reliability）、分辨率（Resolution）和不确定性。 | 无法直接替代分类决策阈值设计。 |
|  | **ECE (Expected Calibration Error)** | 概率分桶后，桶内置信度与真实标签比例的加权差绝对值。 | 风险定价系统、安全关键系统（医疗诊断、信贷授信）中的模型可信度验证。 | 结果受分桶策略（固定宽度 vs 等频分桶）影响较大，且不衡量区分能力。 |
| **量化 / 业务类** | **Rank IC (Information Coefficient)** | 预测打分排名与实际未来收益排名的 Spearman 秩相关系数。 | **量化截面 Alpha 因子有效性评价指标**；评估截面相对强弱。 | 无法衡量收益的非线性厚尾特征以及实际扣减交易滑点/换手率后的表现。 |
|  | **Expected Business Cost** | $\sum_{i,j} C_{ij} \cdot P(\hat{Y}=i, Y=j)$，业务代价矩阵。 | 生产线上业务决策：为误报（FP）和漏报（FN）赋予显式资金损失函数。 | 业务损失矩阵的量化成本往往难以动态精确建模。 |

---

### 2. 十大评估指标底层实现与解析（可折叠代码块）

<details>
<summary><b>实现 1：ROC-AUC（基于 Wilcoxon-Mann-Whitney 秩和检验算法）</b></summary>

```python
import numpy as np

def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算二分类 ROC-AUC
    
    数学原理：Wilcoxon-Mann-Whitney 统计量
    AUC = (sum(rank(S_pos)) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    支持平局分数（Tied Scores）的平均秩次处理。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = np.sum(pos_mask)
    n_neg = np.sum(neg_mask)
    
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true 必须同时包含正例与负例样本")
        
    # 计算升序排列索引
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    
    # 平局分数的平均化（Fractional Ranking）
    sorted_scores = y_score[order]
    unique_scores, inverse_indices, counts = np.unique(sorted_scores, return_inverse=True, return_counts=True)
    if len(unique_scores) < len(y_score):
        tie_ranks = np.cumsum(counts) - (counts - 1) / 2.0
        ranks = tie_ranks[inverse_indices][np.argsort(order)]
    
    sum_pos_ranks = np.sum(ranks[pos_mask])
    u_stat = sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))

# 测试验证
y_t = np.array([0, 0, 1, 1])
y_s = np.array([0.1, 0.4, 0.35, 0.8])
print("ROC-AUC:", compute_roc_auc(y_t, y_s))  # 0.75
```
</details>

<details>
<summary><b>实现 2：PR-AUC / Average Precision（梯步加权面积法）</b></summary>

```python
import numpy as np

def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """计算 PR-AUC / Average Precision (AP)
    
    数学定义：AP = sum_k (R_k - R_{k-1}) * P_k
    按预测分数降序排列，逐点计算 Precision 与 Recall 的梯步变化。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    n_pos = tp[-1]
    
    if n_pos == 0:
        return 0.0
        
    precision = tp / (tp + fp)
    recall = tp / n_pos
    
    # 前驱召回率点 (R_0 = 0)
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    recall_diff = recall - recall_prev
    
    return float(np.sum(precision * recall_diff))

# 测试验证
print("Average Precision:", compute_average_precision(y_t, y_s))
```
</details>

<details>
<summary><b>实现 3：F1-Score 与 F-beta 评分（阈值决策指标）</b></summary>

```python
import numpy as np

def compute_f_beta(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    threshold: float = 0.5, 
    beta: float = 1.0, 
    eps: float = 1e-12
) -> float:
    """计算二分类在指定决策阈值下的 F-beta 评分
    
    数学公式：F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    beta = 1.0: F1-Score (平衡精确率与召回率)
    beta = 2.0: 偏向召回率 (如反欺诈、重大疾病筛查，漏报成本极高)
    beta = 0.5: 偏向精确率 (如垃圾邮件拦截，误报成本极高)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = (np.asarray(y_score).ravel() >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    
    beta_sq = beta ** 2
    f_beta = (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + eps)
    return float(f_beta)

# 测试验证
print("F1-Score:", compute_f_beta(y_t, y_s, threshold=0.5, beta=1.0))
print("F2-Score:", compute_f_beta(y_t, y_s, threshold=0.5, beta=2.0))
```
</details>

<details>
<summary><b>实现 4：Precision@k 与 Recall@k（容量受限 Top-k 截断）</b></summary>

```python
import numpy as np

def compute_precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> tuple[float, float]:
    """计算置信度最高的 Top-k 样本中的查准率与查全率
    
    适用场景：人工复审名额受限、推荐系统前 k 个曝光位等。
    使用 argpartition 保证 O(N) 的选择复杂度。
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    n = len(y_true)
    k = min(max(1, k), n)
    
    # 快速获取 Top-k 索引
    top_k_indices = np.argpartition(-y_score, k - 1)[:k]
    
    hits = np.sum(y_true[top_k_indices] == 1)
    total_positives = np.sum(y_true == 1)
    
    p_at_k = float(hits / k)
    r_at_k = float(hits / total_positives) if total_positives > 0 else 0.0
    return p_at_k, r_at_k

# 测试验证
print("P@2, R@2:", compute_precision_recall_at_k(y_t, y_s, k=2))
```
</details>

<details>
<summary><b>实现 5：Balanced Accuracy（各类别召回率宏平均）</b></summary>

```python
import numpy as np

def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平衡准确率 Balanced Accuracy
    
    数学定义：各类别召回率（Sensitivity / Specificity）的未加权平均：
    Balanced_Acc = 0.5 * (TPR + TNR)
    完全克服多数类掩盖少数类预测失败的虚高准确率陷阱。
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(y_true)
    
    recalls = []
    for c in classes:
        mask = (y_true == c)
        total_c = np.sum(mask)
        if total_c > 0:
            tp_c = np.sum((y_pred == c) & mask)
            recalls.append(tp_c / total_c)
            
    return float(np.mean(recalls)) if len(recalls) > 0 else 0.0

# 测试验证
print("Balanced Acc:", compute_balanced_accuracy(y_t, (y_s >= 0.5).astype(int)))
```
</details>

<details>
<summary><b>实现 6：Log-Loss / Binary Cross-Entropy（数值稳定对数损失）</b></summary>

```python
import numpy as np

def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """计算数值稳定的二元对数损失（Log-Loss / BCE）
    
    数学公式：-1/N * sum(y * ln(p) + (1-y) * ln(1-p))
    边界保护：将预测概率 clip 到 [eps, 1-eps] 防止 log(0) 产生 NaN。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.clip(np.asarray(y_prob, dtype=float).ravel(), eps, 1.0 - eps)
    loss = -np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss)

# 测试验证
print("Log Loss:", compute_log_loss(y_t, y_s))
```
</details>

<details>
<summary><b>实现 7：Brier Score（概率空间均方误差与校准质量）</b></summary>

```python
import numpy as np

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """计算二分类 Brier Score
    
    数学公式：BS = 1/N * sum (p_i - y_i)^2
    性质：
    1. 取值范围 [0, 1]，0 表示完美校准与完美分类；
    2. 可严格分解为：Brier = Reliability - Resolution + Uncertainty
       - Reliability (可靠性/校准误差)：概率预测是否匹配真实发生频率；
       - Resolution (分辨率)：模型区分类别的能力；
       - Uncertainty (固有不确定性)：事件先验方差 p*(1-p)。
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    return float(np.mean((y_prob - y_true) ** 2))

# 测试验证
print("Brier Score:", compute_brier_score(y_t, y_s))
```
</details>

<details>
<summary><b>实现 8：Expected Calibration Error / ECE（等宽分桶期望校准误差）</b></summary>

```python
import numpy as np

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """计算期望校准误差 ECE（Expected Calibration Error）
    
    数学公式：ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    衡量模型输出概率与真实观测经验频率之间的绝对校准差距。
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n = len(y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

# 测试验证
print("ECE (10 bins):", compute_ece(y_t, y_s, n_bins=10))
```
</details>

<details>
<summary><b>实现 9：Rank IC（截面 Spearman 秩相关系数）</b></summary>

```python
import numpy as np

def compute_rank_ic(pred_scores: np.ndarray, true_returns: np.ndarray) -> float:
    """计算量化 Alpha 因子截面 Rank IC（Spearman 秩相关系数）
    
    数学定义：预测得分排序向量与实际未来收益排序向量的 Pearson 线性相关系数。
    评估因子对全市场资产相对表现的单调排序能力。
    """
    pred_scores = np.asarray(pred_scores).ravel()
    true_returns = np.asarray(true_returns).ravel()
    
    def rank_array(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a))
        return ranks
        
    rank_p = rank_array(pred_scores)
    rank_y = rank_array(true_returns)
    
    # 协方差与方差计算
    cov = np.cov(rank_p, rank_y)[0, 1]
    std_p = np.std(rank_p, ddof=1)
    std_y = np.std(rank_y, ddof=1)
    
    if std_p == 0 or std_y == 0:
        return 0.0
    return float(cov / (std_p * std_y))

# 测试验证
print("Rank IC:", compute_rank_ic(y_s, y_t))
```
</details>

<details>
<summary><b>实现 10：Expected Business Cost（业务代价矩阵加权损失）</b></summary>

```python
import numpy as np

def compute_expected_business_cost(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    cost_matrix: np.ndarray
) -> float:
    """计算线上决策的期望业务代价
    
    参数定义：
    cost_matrix: 二维数组 C[pred_class, true_class]
    例如风控二分类代价矩阵：
    cost_matrix = [[C_00 (正常判为正常: 0元),   C_01 (盗刷漏报: 损失1000元)],
                   [C_10 (误封正常用户: 损失50元), C_11 (盗刷拦截: 挽损成本5元)]]
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    
    # 向量化直接索引对应成本
    costs = cost_matrix[y_pred, y_true]
    return float(np.mean(costs))

# 测试验证
cost_mat = np.array([
    [0.0, 1000.0],  # 预测为 0
    [50.0, 5.0]     # 预测为 1
])
y_p_hard = (y_s >= 0.5).astype(int)
print("平均业务单笔损失:", compute_expected_business_cost(y_t, y_p_hard, cost_mat))
```
</details>

---

## 模块五：经典损失函数剖析：线性回归 vs 逻辑回归，MSE vs MAE

### 1. 线性回归目标函数与高斯 MLE 概率推导

线性回归使用均方误差（Mean Squared Error, MSE）或普通最小二乘法（Ordinary Least Squares, OLS）作为目标函数：

$$\mathcal{L}_{\text{Linear}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

#### 概率论推导（Gaussian MLE Derivation）

假设目标值 $y_i$ 与模型预测值 $\mathbf{w}^T \mathbf{x}_i$ 之间的残差 $\epsilon_i$ 独立同分布于均值为 0、方差为 $\sigma^2$ 的一维高斯分布：

$$y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2) \implies y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^T \mathbf{x}_i, \sigma^2)$$

其样本条件概率密度为：

$$p(y_i \mid \mathbf{x}_i; \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - \mathbf{w}^T \mathbf{x}_i)^2}{2\sigma^2} \right)$$

构建全样本的对数似然函数 $\ell(\mathbf{w})$：

$$\ell(\mathbf{w}) = \sum_{i=1}^N \ln p(y_i \mid \mathbf{x}_i; \mathbf{w}, \sigma^2) = -\frac{N}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

最大化对数似然 $\max_{\mathbf{w}} \ell(\mathbf{w})$ 等价于最小化负对数似然，常数项舍去后即精确等价于**最小化均方误差（MSE）**：

$$\arg\max_{\mathbf{w}} \ell(\mathbf{w}) \iff \arg\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

---

### 2. 逻辑回归目标函数与伯努利 MLE 推导

对于二分类问题 $y_i \in \{0, 1\}$，逻辑回归通过 Sigmoid 函数将线性输出映射为后验概率 $\hat{p}_i$：

$$\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}_i}}$$

假设 $y_i \mid \mathbf{x}_i$ 服从伯努利分布 $\text{Bernoulli}(\hat{p}_i)$，其概率质量函数为：

$$P(y_i \mid \mathbf{x}_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

全样本对数似然函数为：

$$\ell(\mathbf{w}) = \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

取负均值得到二元交叉熵损失（Binary Cross-Entropy / Log Loss）：

$$\mathcal{L}_{\text{Logistic}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

---

### 3. 为什么逻辑回归分类不能使用 MSE 损失？

许多初学者会问：“既然 MSE 能衡量误差，为什么不能直接在逻辑回归的 $\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i)$ 上使用 MSE 损失？”

$$\mathcal{L}_{\text{MSE-Logistic}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i))^2$$

**不能使用 MSE 的三大根本原因**：

#### 原因 1：非凸性（Non-Convexity）与局部极小值陷阱

- **Log Loss** 与线性参数 $\mathbf{w}$ 结合是严格的**凸函数（Convex Function）**，其 Hessian 矩阵半正定，保证任意梯度下降算法都能收敛到全局全局最优解。
- **MSE** 与非线性的 Sigmoid 函数复合后，损失函数曲面变得高度**非凸（Non-Convex）**，存在大量平坦区域（Platoons）、鞍点（Saddle Points）和局部极小值（Local Minima），梯度下降极易卡死。

#### 原因 2：梯度消失与错误惩罚软弱（Vanishing Gradient on Severe Errors）

对比两者的参数梯度对残差的响应：

1. **MSE 损失关于参数 $\mathbf{w}$ 的梯度**：
   令 $z_i = \mathbf{w}^T \mathbf{x}_i$，根据链式法则：

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \mathbf{w}} = \frac{2}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \cdot \sigma'(z_i) \cdot \mathbf{x}_i = \frac{2}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \cdot \hat{p}_i(1 - \hat{p}_i) \cdot \mathbf{x}_i$$

   **致命缺陷**：当模型发生**严重错误预测**时（例如真实标签 $y_i = 1$，但模型输出 $\hat{p}_i = 0.0001$）：
   - 项 $(\hat{p}_i - y_i) \approx -1$（误差极大，理应受到剧烈惩罚）；
   - 但导数项 $\hat{p}_i(1 - \hat{p}_i) = 0.0001 \times 0.9999 \approx 0.0001 \to 0$！
   - 两者相乘导致**梯度几乎为 0**！模型在犯下大错时反而失去了学习动力，更新停滞。

2. **Log Loss 损失关于参数 $\mathbf{w}$ 的梯度**：

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \mathbf{x}_i$$

   **完美性质**：Sigmoid 的导数项 $\hat{p}_i(1-\hat{p}_i)$ 与 Log Loss 对 $\hat{p}$ 求导的分母**精准抵消**！梯度严格正比于预测误差 $(\hat{p}_i - y_i)$。当预测错得越离谱时，梯度越大，反向传播纠错越迅速。

#### 原因 3：概率校准（Well-Calibrated Probabilities）

Log Loss 源自伯努利最大似然估计，能够驱动模型输出真正收敛到真实的后验条件概率 $P(Y=1 \mid X)$；而 MSE 无法提供这种严格的概率校准保证。

---

### 4. MSE vs MAE 深度权衡与统计学收敛特性

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \quad \text{vs.} \quad \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

| 核心考量维度 | 均方误差（MSE / L2 Loss） | 平均绝对误差（MAE / L1 Loss） |
|---|---|---|
| **对离群异常值的敏感度** | **极度敏感**。残差被平方放大，单个极端异常值会产生巨大梯度，拉偏整个回归超平面。 | **高度鲁棒（Robust）**。误差按线性比例惩罚，受极端离群点的影响显著减小。 |
| **可导性与优化便利度** | **处处连续可导**。梯度 $\nabla_{\hat{y}} = -2(y - \hat{y})$ 平滑且随接近最优解自动缩小，易于梯度下降稳定收敛。 | **在 $e=0$ 处不可导**。梯度为固定符号阶跃函数（$\pm 1$），在极小值附近容易震荡，需使用次梯度或衰减学习率。 |
| **统计学收敛目标** | 最小化经验风险收敛到**条件均值（Conditional Mean）**：<br>$$\hat{y}^* = \mathbb{E}[y \mid \mathbf{x}]$$ | 最小化经验风险收敛到**条件中位数（Conditional Median）**：<br>$$\hat{y}^* = \text{Median}(y \mid \mathbf{x})$$ |

#### 数学证明：为什么 MSE 对应条件均值，而 MAE 对应条件中位数？

1. **MSE 的最优解是条件期望**：
   求期望风险极小值：$\min_c \mathbb{E}[(Y - c)^2]$
   对常数 $c$ 求导并令导数为 0：

$$\frac{d}{dc} \mathbb{E}[(Y - c)^2] = \mathbb{E}[-2(Y - c)] = -2\mathbb{E}[Y] + 2c = 0 \implies c^* = \mathbb{E}[Y]$$

2. **MAE 的最优解是中位数**：
   求期望风险极小值：$\min_c \mathbb{E}[|Y - c|]$
   对 $c$ 求导（利用 Leibniz 积分法则）：

$$\frac{d}{dc} \left( \int_{-\infty}^c (c - y) p(y)dy + \int_c^{\infty} (y - c) p(y)dy \right) = P(Y \le c) - P(Y > c) = 0$$

$$P(Y \le c) = P(Y > c) = 0.5 \implies c^* = \text{Median}(Y)$$

---

### 5. 折中方案：Huber Loss 与 Smooth L1 Loss

为了兼顾 MSE 的平滑可导性与 MAE 的抗离群鲁棒性，工业界常使用 **Huber Loss**：

$$\mathcal{L}_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{for } |e| \le \delta \\ \delta \left( |e| - \frac{1}{2}\delta \right) & \text{for } |e| > \delta \end{cases}$$

- **小误差区间（$|e| \le \delta$）**：表现为 MSE，梯度为 $e$，连续平滑，便于微调收敛；
- **大误差区间（$|e| > \delta$）**：平滑过渡为 MAE，梯度被截断为固定的 $\pm \delta$，防止异常值梯度爆炸。

```text
损失函数梯度行为对比：
      误差 e 趋向无穷大时:
      • MSE 梯度: 2e ──> 趋向无穷 (梯度爆炸风险)
      • MAE 梯度: ±1 ──> 恒定常数 (零点不连续)
      • Huber 梯度: ±δ ──> 恒定有界且零点平滑！
```

---

### 6. Quick Coding：手写常用损失函数与导数验证

```python
import torch
import torch.nn as nn
import numpy as np

def custom_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """手写 MSE 损失"""
    return torch.mean((y_pred - y_true) ** 2)

def custom_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """手写 MAE 损失"""
    return torch.mean(torch.abs(y_pred - y_true))

def custom_bce_loss(y_prob: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """手写数值稳定的二元交叉熵损失"""
    y_prob = torch.clamp(y_prob, min=eps, max=1.0 - eps)  # 防止 log(0) 溢出
    return -torch.mean(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))

def custom_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """手写 Huber Loss"""
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)

# 单元测试与对齐验证
y_t = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)
y_p = torch.tensor([0.9, 0.2, 0.8, 0.4], dtype=torch.float32)

# 验证与 PyTorch 官方原生实现严格数值等价
assert torch.allclose(custom_mse_loss(y_p, y_t), nn.MSELoss()(y_p, y_t))
assert torch.allclose(custom_mae_loss(y_p, y_t), nn.L1Loss()(y_p, y_t))
assert torch.allclose(custom_bce_loss(y_p, y_t), nn.BCELoss()(y_p, y_t))
assert torch.allclose(custom_huber_loss(y_p, y_t, delta=1.0), nn.HuberLoss(delta=1.0)(y_p, y_t))

print("✅ 所有损失函数数值测试均通过验证！")
```

---

## 模块六：核心机制辨析与系统问答清单

### Q1：如果训练集和测试集的分布不一致（Covariate Shift），如何设计交叉验证？
> **答**：
> 1. 先进行对抗验证（Adversarial Validation）：将训练集打标为 0，测试集打标为 1，训练一个二分类器（如 LightGBM）。若 AUC 远大于 0.5，说明存在明显的协变量偏移。
> 2. 利用对抗验证分类器的预测概率对训练样本计算重要性权重（Importance Weighting $w(x) = \frac{p_{\text{test}}(x)}{p_{\text{train}}(x)}$），或者选择与测试集概率分布最接近的训练样本构建验证集。

### Q2：为什么目标编码（Target Encoding）极易发生数据泄露？如何彻底防范？
> **答**：
> 1. 直接计算全量类别的目标均值会把样本自身的标签反哺给自己，产生严重的自相关泄露。
> 2. **防范标准**：采用 **K 折袋外目标编码（Out-of-Fold Target Encoding）**，计算当前样本所属类别的编码均值时，必须严格排除当前折（甚至排除当前样本自身），并施加经验贝叶斯平滑（Smoothing with prior mean）和高斯噪声扰动。

### Q3：为什么说最小化 MAE 比 MSE 更适合存在大量错误标记（Label Noise）的回归任务？
> **答**：
> 因为 MSE 会将离群点的巨大残差进行平方放大，导致模型被少数几个具有大标注错误的噪声样本“绑架”，过度扭曲模型拟合方向；而 MAE 的惩罚上限是线性的，对应的最优解是条件中位数，中位数对单侧尾部的极端噪声拥有天然的崩溃点（Breakdown Point）免疫力。

### Q4：在极度不平衡业务中，为什么即便 ROC-AUC 达到 0.98，模型上线后查准率依然可能崩溃？
> **答**：
> 核心根源在于假阳率公式 $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$。当负样本基数极大（如正负比 1:1000）时，巨大的 $\text{TN}$ 会稀释分母，使得即便模型产生了大量误报（例如 $\text{FP} = 1000$ 对比 $\text{TP} = 50$），$\text{FPR}$ 依然仅有千分之几，ROC 曲线显得极为优异。然而在实际业务中，查准率 $\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{50}{1050} \approx 4.76\%$，导致人工审核资源被海量误报完全瘫痪。因此极端不平衡场景必须以 **PR-AUC（Average Precision）** 或 **Precision@k** 作为核心评估基准。

---

## 模块七：模型训练诊断、前向手算推导与无监督经典算法从零实现

### 1. 混淆矩阵严选：满足 Recall > 90% 与 FPR < 10% 的多矩阵筛选决策

在分类系统评估与风控排查中，常需要在给定的多个候选模型混淆矩阵（Confusion Matrices）中，筛选出同时满足**高查全率（Recall > 90%）**与**低假阳率（FPR < 10%）**的有效模型。

#### (1) 混淆矩阵基础指标与定义规范

设真实类别为行（Rows），预测类别为列（Columns）：

$$egin{pmatrix} 	ext{TN} & 	ext{FP} \ 	ext{FN} & 	ext{TP} \end{pmatrix}$$

- **真正例（TP）**：真实为正，预测为正；
- **假负例（FN）**：真实为正，预测为负（漏报）；
- **假正例（FP）**：真实为负，预测为正（误报）；
- **真负例（TN）**：真实为负，预测为负。

核心评估公式：
1. **召回率 / 真正率（Recall / Sensitivity / TPR）**：
   $$	ext{Recall} = rac{	ext{TP}}{	ext{TP} + 	ext{FN}} = rac{	ext{TP}}{	ext{Actual Positives}} > 0.90$$
2. **假阳率（False Positive Rate / FPR / Fall-out）**：
   $$	ext{FPR} = rac{	ext{FP}}{	ext{FP} + 	ext{TN}} = rac{	ext{FP}}{	ext{Actual Negatives}} = 1 - 	ext{Specificity} < 0.10$$

#### (2) 候选矩阵多维对比与判定实例

假设测试集包含 100 个真实正样本与 100 个真实负样本，考察以下 4 个候选混淆矩阵：

| 矩阵 | TP | FN | FP | TN | Recall (TPR) | FPR | 满足 Recall > 90% 且 FPR < 10%？ |
|---|---|---|---|---|---|---|---|
| **Matrix A** | 95 | 5 | 8 | 92 | $rac{95}{100} = 95.0\%$ | $rac{8}{100} = 8.0\%$ | **合格 (Pass)**：Recall=95% > 90%, FPR=8% < 10% |
| **Matrix B** | 88 | 12 | 4 | 96 | $rac{88}{100} = 88.0\%$ | $rac{4}{100} = 4.0\%$ | **淘汰 (Fail)**：Recall=88% 未达标 (< 90%) |
| **Matrix C** | 98 | 2 | 15 | 85 | $rac{98}{100} = 98.0\%$ | $rac{15}{100} = 15.0\%$ | **淘汰 (Fail)**：FPR=15% 超标 (> 10%) |
| **Matrix D** | 92 | 8 | 7 | 93 | $rac{92}{100} = 92.0\%$ | $rac{7}{100} = 7.0\%$ | **合格 (Pass)**：Recall=92% > 90%, FPR=7% < 10% |

#### (3) 验证实现代码

```python
from typing import List, Tuple, Dict

def filter_confusion_matrices(
    matrices: List[Dict[str, int]],
    min_recall: float = 0.90,
    max_fpr: float = 0.10
) -> List[Tuple[Dict[str, int], float, float]]:
    """
    筛选满足 Recall > min_recall 且 FPR < max_fpr 的候选混淆矩阵。
    """
    valid_matrices = []
    for m in matrices:
        tp, fn = m['TP'], m['FN']
        fp, tn = m['FP'], m['TN']
        
        positives = tp + fn
        negatives = fp + tn
        
        recall = tp / positives if positives > 0 else 0.0
        fpr = fp / negatives if negatives > 0 else 0.0
        
        if recall > min_recall and fpr < max_fpr:
            valid_matrices.append((m, recall, fpr))
            
    return valid_matrices

# 单元测试
candidates = [
    {'name': 'A', 'TP': 95, 'FN': 5, 'FP': 8, 'TN': 92},
    {'name': 'B', 'TP': 88, 'FN': 12, 'FP': 4, 'TN': 96},
    {'name': 'C', 'TP': 98, 'FN': 2, 'FP': 15, 'TN': 85},
    {'name': 'D', 'TP': 92, 'FN': 8, 'FP': 7, 'TN': 93},
]
passed = filter_confusion_matrices(candidates, min_recall=0.90, max_fpr=0.10)
assert [m[0]['name'] for m in passed] == ['A', 'D']
```

---

### 2. 训练损失发散（Training Loss Divergence）多维根因排查与多选辨析

当模型训练过程中，训练损失（Training Loss）出现异常暴增、震荡发散至无穷大（`inf`）或出现 `NaN` 时，需从数值计算、优化超参数和网络架构三个层面进行系统性多维归因：

| 潜在成因 | 是否会导致发散？ | 理论机制与物理证明 |
|---|---|---|
| **学习率 / 步长过大 (Step Size / Learning Rate Too Large)** | **是 [核心主因]** | 在二次目标函数 $rac{1}{2} x^T H x$ 中，若学习率 $\eta > rac{2}{\lambda_{\max}(H)}$，梯度更新将在曲率最大方向发生几何级发散：$\|w_{t+1} - w^*\| > \|w_t - w^*\|$，导致损失指数级爆炸。 |
| **未对输入特征进行归一化 / 标准化 (Unnormalized Features)** | **是 [核心主因]** | 特征尺度差异巨大导致损失曲面的 Hessian 矩阵条件数极度恶化（$\kappa(H) = rac{\lambda_{\max}}{\lambda_{\min}} \gg 1$），形成狭长病态狭谷。固定学习率在陡峭方向震荡跳出边界。 |
| **损失函数缺乏数值截断保护 (Unstable Loss Formulation)** | **是 [核心主因]** | 交叉熵损失中未对预测概率施加防溢出截断（如缺少 `clamp(p, eps, 1-eps)`），当 $p 	o 0$ 时 $\log(p) 	o -\infty$，乘法溢出直接产生 `NaN`。 |
| **深度网络中缺乏梯度裁剪 (Exploding Gradients without Clipping)** | **是 [核心主因]** | 反向传播中长程矩阵连乘导致梯度范数 $\|
abla_	heta \mathcal{L}\|$ 突破浮点数表示上限，引发参数剧烈外弹。 |
| **正则化系数过高 (Regularization Too High)** | **否 [典型误选]** | 正则化系数 $\lambda$ 过高会把权重强行压制向 0，导致**严重欠拟合（Underfitting）**，损失维持在较大的恒定非零值，但**绝不会引起损失发散至无穷大**。 |
| **病态问题中未施加正则化 (Zero Regularization in Ill-Posed Problems)** | **是** | 当特征高度共线性或样本数少于特征数时，$X^T X$ 不可逆或奇异，缺少 $L_2$ 正则化导致权重参数毫无约束地膨胀发散。 |

---

### 3. 拟合-验证曲线过拟合诊断与针对性缓解对策 (Overfitting Diagnosis & Mitigations)

#### (1) 学习曲线（Fit-vs-Validation Curve）病理诊断

```text
损失 (Loss)
 ▲
 │   \              验证集损失 (Validation Loss)
 │    \             /-------------------- (泛化差距急剧扩大: 高方差 / 过拟合)
 │     \   最低点  /
 │      \───★───/
 │        │        \─────── 训练集损失 (Training Loss) 逼近 0
 └──────────────────────────────────────────► 训练轮次 (Epochs)
```

- **过拟合识别特征**：训练集损失持续单调下降逼近 0，而验证集损失在越过鞍部拐点后不降反升，泛化鸿沟（Generalization Gap: $\mathcal{L}_{	ext{val}} - \mathcal{L}_{	ext{train}}$）持续发散。此时模型进入“背诵训练样本噪声”的高方差阶段。

#### (2) 缓解对策多选有效性判定

| 调优策略 | 是否有效缓解过拟合？ | 作用机理 |
|---|---|---|
| **早停机制 (Early Stopping)** | **有效 [核心对策]** | 监控验证集损失，在验证损失达到局部极小值并连续 $P$ 轮不再改善时提前终止训练，阻止模型学习样本噪声。 |
| **引入 / 增加正则化惩罚项 ($L_1 / L_2$ Regularization)** | **有效 [核心对策]** | 约束参数空间的 $L_2$ 范数（Weight Decay）或 $L_1$ 稀疏性，压缩模型有效容量与函数假设空间。 |
| **引入 Dropout / DropPath** | **有效 [核心对策]** | 在前向传播中以概率 $p$ 随机阻断神经元连接，阻止复杂的协同适应（Co-adaptation），等价于隐式海量子网集成。 |
| **数据增强 (Data Augmentation) 与扩充样本** | **有效 [核心对策]** | 引入人工领域扰动，增加样本多样性，提高训练数据分布对真实总体分布的覆盖率。 |
| **降低模型复杂度 (Reduce Model Capacity)** | **有效 [核心对策]** | 减少网络层数、隐层通道数、降低决策树最大深度 `max_depth`，从结构上限制模型的拟合能力。 |
| **特征筛选与降维 (Feature Selection / Pruning)** | **有效 [核心对策]** | 剔除与目标无关、信噪比极低的高方差噪声特征，降低特征空间维度。 |
| **继续训练更多 Epoch** | **无效 [反向恶化]** | 在过拟合拐点之后继续训练，只会促使模型更深地拟合噪声，加剧泛化差距恶化。 |
| **增加网络深度与隐藏单元** | **无效 [反向恶化]** | 增大假设空间容量，使模型更容易在训练集上形成过参数化过拟合。 |

---

### 4. 3 层神经网络前向传播纯手算数值推导 (3-Layer NN Forward Pass by Hand)

#### (1) 网络架构与参数设定

考虑一个 3 层全连接前向神经网络，输入维度 2，隐层维度 2，输出维度 1，激活函数为 **Linear - Linear - Sigmoid**：

- **输入**：$x = egin{pmatrix} 0.5 \ -0.2 \end{pmatrix}$
- **第 1 层 (Linear)**：
  $$W_1 = egin{pmatrix} 0.4 & -0.5 \ 0.2 & 0.8 \end{pmatrix}, \quad b_1 = egin{pmatrix} 0.1 \ -0.1 \end{pmatrix}$$
- **第 2 层 (Linear)**：
  $$W_2 = egin{pmatrix} 0.5 & 0.3 \ -0.2 & 0.4 \end{pmatrix}, \quad b_2 = egin{pmatrix} -0.05 \ 0.15 \end{pmatrix}$$
- **第 3 层 (Linear + Sigmoid)**：
  $$W_3 = egin{pmatrix} 1.2 & -0.8 \end{pmatrix}, \quad b_3 = 0.05$$
  激活函数：$\sigma(z) = rac{1}{1 + e^{-z}}$

#### (2) 纯手算分步推导

1. **第 1 层计算**：
   $$z_1 = W_1 x + b_1 = egin{pmatrix} 0.4(0.5) + (-0.5)(-0.2) + 0.1 \ 0.2(0.5) + 0.8(-0.2) + (-0.1) \end{pmatrix} = egin{pmatrix} 0.20 + 0.10 + 0.10 \ 0.10 - 0.16 - 0.10 \end{pmatrix} = egin{pmatrix} 0.400 \ -0.160 \end{pmatrix}$$

2. **第 2 层计算**：
   $$z_2 = W_2 z_1 + b_2 = egin{pmatrix} 0.5(0.400) + 0.3(-0.160) - 0.05 \ -0.2(0.400) + 0.4(-0.160) + 0.15 \end{pmatrix} = egin{pmatrix} 0.200 - 0.048 - 0.050 \ -0.080 - 0.064 + 0.150 \end{pmatrix} = egin{pmatrix} 0.102 \ 0.006 \end{pmatrix}$$

3. **第 3 层计算与 Sigmoid**：
   $$z_3 = W_3 z_2 + b_3 = 1.2(0.102) + (-0.8)(0.006) + 0.050 = 0.1224 - 0.0048 + 0.0500 = 0.1676$$
   $$\hat{y} = \sigma(0.1676) = rac{1}{1 + e^{-0.1676}} pprox rac{1}{1 + 0.84569} pprox rac{1}{1.84569} pprox 0.54180 pprox \mathbf{0.542}$$

#### (3) 验证代码

```python
import numpy as np

def manual_forward_pass() -> float:
    x = np.array([0.5, -0.2])
    W1 = np.array([[0.4, -0.5], [0.2, 0.8]])
    b1 = np.array([0.1, -0.1])
    
    W2 = np.array([[0.5, 0.3], [-0.2, 0.4]])
    b2 = np.array([-0.05, 0.15])
    
    W3 = np.array([1.2, -0.8])
    b3 = 0.05
    
    z1 = W1 @ x + b1
    z2 = W2 @ z1 + b2
    z3 = float(W3 @ z2 + b3)
    out = 1.0 / (1.0 + np.exp(-z3))
    return round(out, 3)

assert manual_forward_pass() == 0.542
```

---

### 5. 一维数据流局部极大值检出与边界退化处理 (Local Maximum on a 1-D Stream)

在时间序列信号处理与流式特征工程中，需实时检出一维序列中的局部波峰极大值点：

#### (1) 问题形式化与边界退化规则

给定一维数值序列 `rawData` 与局部考察半径 `localArea`（记为 $k$）：
- 对于索引 $i$，左侧有效邻居长度为 $L = \min(i, k)$，右侧有效邻居长度为 $R = \min(N - 1 - i, k)$；
- 若左侧存在邻居，从外侧向 $i$ 必须**严格单调递增**（即从 $i$ 向左看必须严格单调递减）：
  $$rawData[i - j + 1] > rawData[i - j], \quad orall j \in [1, L]$$
- 若右侧存在邻居，从 $i$ 向右看必须**严格单调递减**：
  $$rawData[i + j - 1] > rawData[i + j], \quad orall j \in [1, R]$$
- **边界退化原则**：当某一侧有效邻居不足 $k$ 个时，直接使用现有存在的全部邻居进行校验。若长度为 1，直接判定为极大值；若存在平顶（相邻相等数值），严格打破递减关系，不判定为极值。

#### (2) 算法实现与测试

```python
from typing import List

def find_local_maxima(rawData: List[float], localArea: int) -> List[int]:
    """
    检出一维数据流中所有满足左右 localArea 邻域严格递减的局部最大值索引。
    若一侧不足 localArea 个邻居，则退化检查所有可用邻居。
    """
    n = len(rawData)
    if n == 0:
        return []

    local_max_indices = []

    for i in range(n):
        is_peak = True

        # 检查左侧: 从 i 向左必须严格单调递减 (即从左向右严格单调递增至 i)
        left_bound = min(i, localArea)
        for j in range(1, left_bound + 1):
            if rawData[i - j + 1] <= rawData[i - j]:
                is_peak = False
                break

        if not is_peak:
            continue

        # 检查右侧: 从 i 向右必须严格单调递减
        right_bound = min(n - 1 - i, localArea)
        for j in range(1, right_bound + 1):
            if rawData[i + j - 1] <= rawData[i + j]:
                is_peak = False
                break

        if is_peak:
            local_max_indices.append(i)

    return local_max_indices

# 测试用例
arr = [1, 3, 5, 4, 2, 6, 2, 1]
assert find_local_maxima(arr, 2) == [2, 5]
# 边界单元素退化
assert find_local_maxima([10], 3) == [0]
# 平台相等打破严格极大值
assert find_local_maxima([2, 4, 4, 1], 1) == []
```

- **复杂度**：时间复杂度 $\mathcal{O}(N \cdot k)$，额外空间复杂度 $\mathcal{O}(1)$。

---

### 6. 从零纯手写 K-Means 聚类器 (k-Means from Scratch: Assign-Update Loop)

K-Means 是最经典的无监督聚类算法，核心遵循 Lloyd 算法的交替最小化（Alternating Minimization）框架：

#### (1) 数学优化目标

最小化样本点到对应聚类中心的簇内平方误差和（Inertia / WCSS）：

$$rg\min_{\mathcal{S}, oldsymbol{\mu}} \sum_{j=1}^K \sum_{\mathbf{x} \in S_j} \|\mathbf{x} - oldsymbol{\mu}_j\|^2$$

两步迭代交替推进：
1. **样本簇分配（Assignment Step）**：
   $$c_i^{(t)} = rg\min_{j \in \{1, \dots, K\}} \|\mathbf{x}_i - oldsymbol{\mu}_j^{(t)}\|^2$$
2. **质心重算更新（Update Step）**：
   $$oldsymbol{\mu}_j^{(t+1)} = rac{1}{|S_j|} \sum_{i \in S_j} \mathbf{x}_i$$

#### (2) 生产级实现

```python
import numpy as np
from typing import List, Union

class ScratchKMeans:
    def __init__(self, k: int, initial_centroids: Union[List[List[float]], np.ndarray], max_iters: int = 100):
        self.k = k
        self.centroids = np.asarray(initial_centroids, dtype=float)
        self.max_iters = max_iters

    def fit_predict(self, data: Union[List[List[float]], np.ndarray]) -> List[int]:
        """
        执行标准交替指派-更新循环，返回各样本所属簇标签。
        """
        X = np.asarray(data, dtype=float)
        n_samples = len(X)
        labels = np.zeros(n_samples, dtype=int)

        for _ in range(self.max_iters):
            # 1. 指派步: 计算所有点到所有质心的欧氏距离平方
            # 广播计算: X[:, None, :] shape (N, 1, D), centroids[None, :, :] shape (1, K, D)
            distances = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # 收敛判断：若所有样本聚类分配不再变化，提前退出
            if np.array_equal(labels, new_labels) and _ > 0:
                break
            labels = new_labels

            # 2. 更新步: 重算各簇质心均值
            for c in range(self.k):
                cluster_members = X[labels == c]
                if len(cluster_members) > 0:
                    self.centroids[c] = np.mean(cluster_members, axis=0)

        return labels.tolist()

# 验证测试
X_pts = [[1.0, 2.0], [1.5, 1.8], [5.0, 8.0], [8.0, 8.0], [1.0, 0.6], [9.0, 11.0]]
init_centers = [[1.0, 2.0], [8.0, 8.0]]
kmeans = ScratchKMeans(k=2, initial_centroids=init_centers)
cluster_res = kmeans.fit_predict(X_pts)
assert cluster_res == [0, 0, 1, 1, 0, 1]
```

- **复杂度**：时间复杂度 $\mathcal{O}(T \cdot N \cdot K \cdot D)$，空间复杂度 $\mathcal{O}(N \cdot D + K \cdot D)$。

---

### 7. 缩放点积自注意力与第一性原理二元交叉熵损失 (Scaled Dot-Product Attention & First-Principles BCE)

在现代机器学习工程面试中，注意力算子与损失函数的第一性原理推导常作为高频联立考核题：

#### (1) 缩放点积自注意力（Scaled Dot-Product Attention）机制实现

公式原语：
$$	ext{Attention}(Q, K, V) = 	ext{softmax}\left(rac{Q K^T}{\sqrt{d_k}} + Might) V$$

- **输入张量维度**：$Q, K, V \in \mathbb{R}^{B 	imes L 	imes d_k}$（$B$ 为 batch size，$L$ 为序列长度，$d_k$ 为头维度）；
- **掩码（Mask）语义**：在因果自注意力或填充位处，对于无效位置赋予 $-10^9$ 或 $-\infty$，确保经 softmax 后注意力权重绝对归零；
- **数值稳定性保证（Log-Sum-Exp Trick）**：在执行 $\exp$ 之前，减去每行的最大值（Row-Max Subtraction），杜绝浮点数上溢（Overflow）。

```python
import numpy as np
from typing import Optional

def self_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    纯 NumPy 实现的数值稳定缩放点积自注意力。
    
    参数:
    - Q, K, V: shape (batch_size, seq_len, d_k)
    - mask: shape (seq_len, seq_len) 或 (batch_size, seq_len, seq_len)，0 为屏蔽位，1 为保留位
    
    返回:
    - output: shape (batch_size, seq_len, d_k)
    """
    d_k = Q.shape[-1]
    scale = 1.0 / np.sqrt(d_k)

    # 1. 批量矩阵乘法计算原始注意力分数: Q @ K^T -> (B, L, L)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale

    # 2. 掩码注入 (填充位或因果下三角掩码)
    if mask is not None:
        # 将掩码为 0 的位置替换为极大负数
        scores = np.where(mask == 0, -1e9, scores)

    # 3. 数值稳定 Softmax: 减去行最大值防止 exp 溢出
    row_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - row_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. 加权聚合 Value: Attention @ V -> (B, L, d_k)
    output = np.matmul(attention_weights, V)
    return output
```

#### (2) 第一性原理推导：Logits 版二元交叉熵损失（BCEWithLogits）

设逻辑回归模型的未归一化对数几率（Logit）为 $z \in \mathbb{R}$，则预测概率由 Sigmoid 函数给出：
$$p = \sigma(z) = rac{1}{1 + e^{-z}}$$

根据伯努利分布假设，单个样本的目标似然函数为：
$$P(y \mid z) = p^y (1 - p)^{1 - y}$$

取负对数似然（Negative Log-Likelihood）得到单个样本的损失函数：
$$\ell(z, y) = - ig[ y \log(p) + (1 - y) \log(1 - p) ig]$$

**代入 $p = \sigma(z)$ 进行代数化简**：
$$\log(p) = \log\left(rac{1}{1 + e^{-z}}ight) = -\log(1 + e^{-z})$$
$$\log(1 - p) = \log\left(rac{e^{-z}}{1 + e^{-z}}ight) = -z - \log(1 + e^{-z})$$

将上述两式代入 $\ell(z, y)$：
$$\ell(z, y) = - ig[ -y \log(1 + e^{-z}) + (1 - y)(-z - \log(1 + e^{-z})) ig] = (1 - y)z + \log(1 + e^{-z}) = z - yz + \log(1 + e^{-z})$$

为了保证在 $z > 0$ 与 $z < 0$ 时均不会发生指数爆炸，转化为数值稳定的等价形式：
$$\ell(z, y) = \max(z, 0) - z \cdot y + \log(1 + e^{-|z|})$$

批次经验损失为全部样本的算术平均：
$$\mathcal{L}(Z, Y) = rac{1}{N} \sum_{i=1}^N ig[ \max(z_i, 0) - z_i y_i + \log(1 + e^{-|z_i|}) ig]$$

```python
def binary_cross_entropy_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    第一性原理实现的数值稳定二元交叉熵损失（带 logits）。
    等价于 PyTorch 的 nn.BCEWithLogitsLoss()。
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    
    # 稳定计算: max(z, 0) - z * y + log(1 + exp(-|z|))
    max_z = np.maximum(logits, 0.0)
    abs_z = np.abs(logits)
    loss = max_z - logits * labels + np.log(1.0 + np.exp(-abs_z))
    
    return float(np.mean(loss))
```

#### (3) 核心原理追问（Oral Deep-Dive）

1. **为什么点积注意力要除以 $\sqrt{d_k}$？（Variance Scaling）**
   - 假设向量 $Q$ 与 $K$ 的各个分量独立同分布，均值为 0，方差为 1。
   - 点积为 $d_k$ 个独立变量之积的累加：$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$。
   - 由独立随机变量方差的可加性：
     $$\mathbb{E}[q \cdot k] = 0, \quad 	ext{Var}(q \cdot k) = \sum_{i=1}^{d_k} 	ext{Var}(q_i k_i) = \sum_{i=1}^{d_k} 	ext{Var}(q_i) 	ext{Var}(k_i) = d_k$$
   - 点积的标准差为 $\sqrt{d_k}$。若不除以 $\sqrt{d_k}$，当维度 $d_k$ 很大时（如 $d_k = 128$），点积结果方差高达 128，绝大多数注意力分数值会被推入 Softmax 函数两端的极端饱和区，导致反向传播梯度极其接近于 0（梯度消失）。除以 $\sqrt{d_k}$ 将方差重新缩放归一至 1，保证 Softmax 工作在灵敏活跃区间。
2. **前馈神经网络（FFN）子层在 Attention 模块后的核心作用是什么？**
   - Self-Attention 是**跨 Token 的全局线性加权混叠**（Token Mixing），它实现了全序列上下文信息的聚合，但其本身主要提供线性重组；
   - FFN 是**位置级非线性特征投影**（Channel / Feature Mixing），通常采用两层全连接与激活函数（如 GELU/SwiGLU），中间隐层升维至 $4	imes d_{	ext{model}}$。它为每个 Token 独立提供丰富的非线性函数拟合能力与高维语义记忆存储。


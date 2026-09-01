# ML Coding 00 · ML 基础：数据预处理、数据泄露与经典损失函数

在机器学习系统设计与算法工程实践中，扎实的统计学基础与严密的数据管道工程是构建高可用模型的基石。许多模型在离线评测中指标优异，上线后效果却断崖式下跌，其根源往往不在于复杂的模型架构，而在于数据泄露（Data Leakage）、不恰当的缺失值处理（Missing Data Imputation）或对损失函数（Loss Functions）统计假设的认知偏差。

本篇系统梳理工业界最高频的 3 大 ML 基础核心模块：
1. **数据泄露（Data Leakage）机理与全方位防御体系**
2. **缺失值机制（MCAR / MAR / MNAR）与处理策略权衡**
3. **经典损失函数推导：线性回归 vs 逻辑回归，MSE vs MAE 及统计学收敛特性**

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

## 模块三：经典损失函数剖析：线性回归 vs 逻辑回归，MSE vs MAE

### 1. 线性回归目标函数与高斯 MLE 概率推导

线性回归使用均方误差（Mean Squared Error, MSE）或普通最小二乘法（Ordinary Least Squares, OLS）作为目标函数：

$$\mathcal{L}_{\text{Linear}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

#### 概率论推导（Gaussian MLE Derivation）

假设目标值 $y_i$ 与模型预测值 $\mathbf{w}^T \mathbf{x}_i$ 之间的残差 $\epsilon_i$ 独立同分布于均值为 0、方差为 $\sigma^2$ 的一维高斯分布：

$$y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2) \implies y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^T \mathbf{x}_i, \sigma^2)$$

其样本条件概率密度为：

$$p(y_i \mid \mathbf{x}_i; \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - \mathbf{w}^T \mathbf{x}_i)^2}{2\sigma^2} \r\right)$$

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

$$\ell(\mathbf{w}) = \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \r\right]$$

取负均值得到二元交叉熵损失（Binary Cross-Entropy / Log Loss）：

$$\mathcal{L}_{\text{Logistic}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \r\right]$$

---

### 3. 深度面试考点：为什么逻辑回归不能使用 MSE？

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

$$\frac{d}{dc} \left( \int_{-\infty}^c (c - y) p(y)dy + \int_c^{\infty} (y - c) p(y)dy \r\right) = P(Y \le c) - P(Y > c) = 0$$

$$P(Y \le c) = P(Y > c) = 0.5 \implies c^* = \text{Median}(Y)$$

---

### 5. 折中方案：Huber Loss 与 Smooth L1 Loss

为了兼顾 MSE 的平滑可导性与 MAE 的抗离群鲁棒性，工业界常使用 **Huber Loss**：

$$\mathcal{L}_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{for } |e| \le \delta \\ \delta \left( |e| - \frac{1}{2}\delta \r\right) & \text{for } |e| > \delta \end{cases}$$

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

## 模块四：面试前速记与核心问答清单（Interview FAQ）

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

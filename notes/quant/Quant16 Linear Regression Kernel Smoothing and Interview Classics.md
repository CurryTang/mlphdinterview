# Quant 16 · 线性回归与核平滑：OLS、Gauss–Markov、Ridge/Lasso 与经典问题推导

线性回归是量化研究与统计建模的核心基石。深入掌握线性模型不仅要求熟练推导 OLS 估计量的解析闭式解，更在于系统理解高维欧氏几何投影机制、Gauss–Markov 定理的数学边界，以及在实际金融时间序列违背球形扰动与外生性假定（异方差、自相关、多重共线性、测量误差、遗漏变量）时的理论修正与稳健估计方法。

```text
核心理论心智模型（Core Mental Models）：
1. 单变量 OLS 关键恒等式：样本估计量 \hat\beta_1 = \hat\rho_{XY} (s_Y / s_X)（总体形式 \beta_1 = \rho (\sigma_Y / \sigma_X)）以及 R^2 = \hat\rho^2。
2. 回归的不可逆性：y 对 x 的回归斜率与 x 对 y 的回归斜率乘积为 \rho^2 \le 1，不可直接取倒数。
3. 几何正交投影：将 OLS 视作 y 在 X 列空间上的正交投影（Orthogonal Projection）。正交性是推导残差性质的核心。
4. BLUE 不依赖正态性：Gauss-Markov 定理证明 OLS 是最佳线性无偏估计量时不包含正态性假设。正态性仅用于精确的小样本 t/F 检验。
5. 正则化几何效应：Lasso 的 \ell_1 菱形诱导稀疏性（变量选择），Ridge 的 \ell_2 球形诱导谱收缩（控制共线性方差）。
```

> 🧭 **核心知识全景导览**
> - **模块一：OLS 几何与代数**：正规方程 ｜ 残差五大正交性质与方差分解 ｜ 回归系数与协方差本质 ｜ 逆向回归陷阱
> - **模块二：Gauss–Markov、统计推断与核心 Lemma 全景清单**：估计量性质 ｜ t/F 检验与受限模型 ｜ 预测 vs 置信区间 ｜ 留一法与杠杆 ｜ 测量误差与 OVB
> - **模块三：变量选择与收缩（Shrinkage）**：子集选择 ｜ 岭回归（Ridge） ｜ Lasso ｜ 几何直觉与比较
> - **模块四：核平滑与局部回归**：条件期望与核的本质 ｜ Nadaraya-Watson ｜ 边界偏差与局部线性回归 ｜ 维数灾难与破局
> - **模块五：核心经典问题与定理推导（绿皮书 + HOTS + ESL 计算精选）**：相关系数极值 ｜ 等相关矩阵半正定下界 ｜ Cholesky 模拟 ｜ CAPM 与逆向回归 ｜ 仿射变换 ｜ 遗漏变量偏差 ｜ 测量误差 ｜ 多重共线性与 VIF ｜ 最优套保比率 ｜ FWL 定理与两阶段残差回归陷阱（求 β₁/β₂ 比值） ｜ 无截距回归陷阱 ｜ R² 与实盘 IC ｜ 正交设计下四大模型显式解推导 ｜ 岭回归 SVD 谱收缩与 MSE 恒优证明 ｜ 局部线性回归等价核与边界无偏证明 ｜ 平滑矩阵性质与两类有效自由度
> - **模块六：知识结构梳理与核心要点清单**

---

## 模块一：OLS 几何与代数（ESL 3.2）

### 1. 一元线性回归模型与 OLS 估计量

#### 模型设定与符号定义（严格约定）
设我们观测到 $n$ 组独立样本观测对 $\{(X_i, Y_i)\}_{i=1}^n$：
- **$X_i \in \mathbb{R}$**：自变量（解释变量 / 输入特征 / Regressor），为给定的观测特征数值；
- **$Y_i \in \mathbb{R}$**：因变量（被解释变量 / 真实响应值 / Ground Truth / Target），代表我们要预测或解释的客观观测值；
- **$\beta_0, \beta_1 \in \mathbb{R}$**：未知的真实总体回归参数（$\beta_0$ 为总体截距，$\beta_1$ 为总体斜率）；
- **$\varepsilon_i \in \mathbb{R}$**：不可观测的总体随机扰动项（Error Term / Noise），代表所有未被自变量捕捉的随机环境噪声。

一元线性回归的总体真实数据生成过程为：

$$
Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i \quad (i = 1, 2, \dots, n)
$$

#### 估计量、拟合值与残差（“帽子”符号 $\ \hat{}\ $ 的严格含义）
在统计学规范中，字母上方的**“帽子”（Hat, $\ \hat{}\ $）统一定义为“由样本数据计算得出的估计值或预测值”（Estimator / Prediction）**：
- **$\hat\beta_0, \hat\beta_1$**：通过样本数据求解得出的截距与斜率估计值；
- **$\hat{Y}_i$（读作 Y-hat，拟合值 / 预测值 Fitted Value）**：模型根据特征 $X_i$ 与估计参数给出的线性预测输出：
  $$\hat{Y}_i = \hat\beta_0 + \hat\beta_1 X_i$$
- **$\hat\varepsilon_i$（读作 epsilon-hat，样本残差 Residual）**：真实观测值 $Y_i$ 与模型拟合值 $\hat{Y}_i$ 之间的偏差（即模型在第 $i$ 个样本上的实际预测误差）：
  $$\hat\varepsilon_i = Y_i - \hat{Y}_i = Y_i - (\hat\beta_0 + \hat\beta_1 X_i)$$

#### 样本与总体统计量符号（$s_X, s_Y, \sigma_X, \sigma_Y, \rho$）的严格约定
在统计建模与量化推导中，严格遵循“**拉丁字母表示有限样本统计量，希腊字母表示数据生成过程（DGP）底层真实总体参数**”的规范约定：

1. **样本统计量（由 $n$ 组有限样本观测值计算得到，为样本空间中的随机变量）**：
   设样本算术均值为 $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$，$\bar{Y} = \frac{1}{n}\sum_{i=1}^n Y_i$：
   - **样本方差 $s_X^2 = \widehat{\operatorname{Var}}(X)$ 与 $s_Y^2 = \widehat{\operatorname{Var}}(Y)$**：
     $$s_X^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2, \quad s_Y^2 = \frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2$$
   - **样本标准差 $s_X$ 与 $s_Y$**（亦常记为小写 $s_x, s_y$）：
     $$s_X = \sqrt{s_X^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2}, \quad s_Y = \sqrt{s_Y^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2}$$
     *物理含义*：度量样本数据点围绕均值的物理离散尺度，与原始变量具有完全相同的物理量纲（单位）。
   - **样本协方差 $\widehat{\operatorname{Cov}}(X, Y)$**：
     $$\widehat{\operatorname{Cov}}(X, Y) = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})$$
   - **样本 Pearson 相关系数 $\hat\rho_{XY}$**（亦常记为 $r_{XY}$ 或 $r$）：
     $$\hat\rho_{XY} = \frac{\widehat{\operatorname{Cov}}(X, Y)}{s_X s_Y} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2}\sqrt{\sum_{i=1}^n (Y_i - \bar{Y})^2}} \in [-1, 1]$$
     *物理含义*：消除自变量与因变量各自的物理量纲后，度量两者纯粹的无量纲协同线性相关程度。

2. **总体理论参数（客观数据生成过程 DGP 的固有静态特征，不存在样本抽样波动）**：
   - **总体方差 $\sigma_X^2 = \operatorname{Var}(X)$ 与 $\sigma_Y^2 = \operatorname{Var}(Y)$**：$\sigma_X^2 = \mathbb{E}[(X - \mathbb{E}[X])^2]$，$\sigma_Y^2 = \mathbb{E}[(Y - \mathbb{E}[Y])^2]$。
   - **总体标准差 $\sigma_X$ 与 $\sigma_Y$**（亦常记为小写 $\sigma_x, \sigma_y$）：总体分布的真实波动率，$\sigma_X = \sqrt{\operatorname{Var}(X)}$，$\sigma_Y = \sqrt{\operatorname{Var}(Y)}$。
   - **总体协方差与相关系数**：$\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$，$\rho = \frac{\operatorname{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$。
   - **扰动项总体方差 $\sigma^2$**（亦记为 $\sigma_\varepsilon^2$）：回归未观测物理白噪声的真实方差，$\operatorname{Var}(\varepsilon_i \mid X) = \sigma^2$。
   - **渐近收敛性**：由弱大数定律（WLLN），当样本量 $n \to \infty$ 时，样本统计量依概率收敛于总体参数：$s_X \xrightarrow{p} \sigma_X$，$s_Y \xrightarrow{p} \sigma_Y$，$\hat\rho_{XY} \xrightarrow{p} \rho$。

#### OLS 目标函数与闭式估计量推导
普通最小二乘法（OLS）的核心准则是：寻找使全样本残差平方和（Residual Sum of Squares, $RSS$）达到全局最小的参数 $(\hat\beta_0, \hat\beta_1)$：

$$
\min_{\hat\beta_0, \hat\beta_1} \sum_{i=1}^n \hat\varepsilon_i^2 = \min_{\hat\beta_0, \hat\beta_1} \sum_{i=1}^n \left( Y_i - \hat\beta_0 - \hat\beta_1 X_i \right)^2
$$

分别对 $\hat\beta_0$ 与 $\hat\beta_1$ 求偏导并令其为 0（一阶驻点条件）：
1. $\frac{\partial}{\partial \hat\beta_0} = -2\sum_{i=1}^n (Y_i - \hat\beta_0 - \hat\beta_1 X_i) = 0 \implies \sum_{i=1}^n \hat\varepsilon_i = 0 \implies \hat\beta_0 = \bar{Y} - \hat\beta_1 \bar{X}$；
2. 代入对 $\hat\beta_1$ 的导数方程：$\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y} - \hat\beta_1(X_i - \bar{X})) = 0$，解得：

$$
\hat\beta_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2} = \frac{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2} = \frac{\widehat{\operatorname{Cov}}(X, Y)}{\widehat{\operatorname{Var}}(X)}, \quad \hat\beta_0 = \bar{Y} - \hat\beta_1 \bar{X}
$$

分母与分子的样本容量归一化因子 $\frac{1}{n-1}$（或 $\frac{1}{n}$）完全抵消。**一元 OLS 的斜率本质上就是自变量与因变量的样本协方差除以自变量的样本方差**。

- **几何质心定锚（Center-of-Mass Pivot）**：由 $\bar{Y} = \hat\beta_0 + \hat\beta_1 \bar{X}$ 可知，样本重心 $(\bar{X}, \bar{Y})$ 是拟合线的固定刚性支点（Pivot）。无论斜率如何变动，回归直线必强制穿过质心。
- **物理力矩平衡（Torque & Spring Equilibrium）**：最小化 $\sum \hat\varepsilon_i^2$ 物理上等价于每个样本点通过一根垂直弹簧（弹性势能 $E_p \propto \Delta y^2$）拉拽一根刚性杠杆。当杠杆处于总弹性势能最低的静力学平衡态时，所有垂直拉力之和为零（$\sum \hat\varepsilon_i = 0$），绕质心的合力矩亦为零（$\sum (X_i - \bar{X})\hat\varepsilon_i = 0$）。

---

### 2. 判定系数 $R^2$ 与方差分解（ANOVA）

#### 标量与 $n$ 维向量符号的严格对应
为了从全局几何视角透彻审视方差分解，我们将全样本标量堆叠为 $n$ 维样本空间 $\mathbb{R}^n$ 中的列向量：
- **真实观测向量 $Y \in \mathbb{R}^n$**：全样本真实目标值的列向量 $Y = (Y_1, Y_2, \dots, Y_n)^\top$；
- **模型拟合向量 $\hat{Y} \in \mathbb{R}^n$**：全样本模型拟合值的列向量 $\hat{Y} = (\hat{Y}_1, \hat{Y}_2, \dots, \hat{Y}_n)^\top$；
- **样本均值基线向量 $\bar{Y}\mathbf{1} \in \mathbb{R}^n$**：标量均值 $\bar{Y}$ 乘以全 1 向量 $\mathbf{1} = (1, 1, \dots, 1)^\top$，即 $\bar{Y}\mathbf{1} = (\bar{Y}, \bar{Y}, \dots, \bar{Y})^\top$。它代表完全不用任何特征自变量时的“盲猜常数基准”；
- **残差向量 $\hat\varepsilon \in \mathbb{R}^n$**：全样本预测误差列向量 $\hat\varepsilon = Y - \hat{Y} = (\hat\varepsilon_1, \hat\varepsilon_2, \dots, \hat\varepsilon_n)^\top$。

#### 三大平方和（TSS, ESS, RSS）的物理定义
$R^2$ 衡量回归模型相对于朴素均值基线的拟合优度（Goodness of fit），定义为：

$$
R^2 = \frac{ESS}{TSS} = 1 - \frac{RSS}{TSS}
$$

其中三大平方和分别对应样本向量的欧氏范数平方：
- **总离差平方和 $TSS$（Total Sum of Squares）**：
  $$TSS = \sum_{i=1}^n (Y_i - \bar{Y})^2 = \|Y - \bar{Y}\mathbf{1}\|_2^2$$
  *物理意义*：真实数据围绕其均值基线的总变异量（如果不看特征 $X$，盲猜平均值 $\bar{Y}$ 时犯下的总平方误差）。
- **回归解释平方和 $ESS$（Explained Sum of Squares）**：
  $$ESS = \sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2 = \|\hat{Y} - \bar{Y}\mathbf{1}\|_2^2$$
  *物理意义*：模型拟合值围绕均值基线的波动，代表**自变量 $X$ 成功捕捉并解释出来的波动量**。
- **残差平方和 $RSS$（Residual Sum of Squares）**：
  $$RSS = \sum_{i=1}^n (Y_i - \hat{Y}_i)^2 = \sum_{i=1}^n \hat\varepsilon_i^2 = \hat\varepsilon^\top \hat\varepsilon = \|Y - \hat{Y}\|_2^2$$
  *物理意义*：真实值与模型拟合值的偏离程度，代表**模型用尽了特征也无法解释的纯剩余误差**。

#### 方差分解恒等式（ANOVA）的严格代数证明
考察样本点 $i$ 的总离差拆分恒等式：
$$(Y_i - \bar{Y}) = (\hat{Y}_i - \bar{Y}) + (Y_i - \hat{Y}_i) = (\hat{Y}_i - \bar{Y}) + \hat\varepsilon_i$$
对所有样本两边平方并求和：
$$\sum_{i=1}^n (Y_i - \bar{Y})^2 = \sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2 + \sum_{i=1}^n \hat\varepsilon_i^2 + 2\sum_{i=1}^n (\hat{Y}_i - \bar{Y})\hat\varepsilon_i$$
展开交叉项：
$$\sum_{i=1}^n (\hat{Y}_i - \bar{Y})\hat\varepsilon_i = \sum_{i=1}^n \hat{Y}_i \hat\varepsilon_i - \bar{Y}\sum_{i=1}^n \hat\varepsilon_i$$
1. 由一阶条件，残差和为零：$\sum_{i=1}^n \hat\varepsilon_i = 0$；
2. 将 $\hat{Y}_i = \hat\beta_0 + \hat\beta_1 X_i$ 代入前项：
   $$\sum_{i=1}^n \hat{Y}_i \hat\varepsilon_i = \hat\beta_0 \sum_{i=1}^n \hat\varepsilon_i + \hat\beta_1 \sum_{i=1}^n X_i \hat\varepsilon_i = \hat\beta_0 \cdot 0 + \hat\beta_1 \cdot 0 = 0$$
两项均严格为零，交叉项完全消失！由此导出**方差分解恒等式**：

$$
TSS = ESS + RSS
$$

- **高维欧氏勾股定理（Pythagorean Theorem in $\mathbb{R}^n$）**：在 $n$ 维去中心化样本空间中，交叉项为零在几何上等价于两向量正交垂直：$(\hat{Y} - \bar{Y}\mathbf{1}) \perp (Y - \hat{Y})$。观测向量 $Y - \bar{Y}\mathbf{1}$ 为**直角三角形斜边**，拟合向量 $\hat{Y} - \bar{Y}\mathbf{1}$ 为落在特征子空间上的**邻边**，残差向量 $\hat\varepsilon = Y - \hat{Y}$ 为垂直于特征子空间的**对边**。两直角边严格正交，故斜边模长平方恒等于两直角边模长平方之和（$\|Y - \bar{Y}\mathbf{1}\|_2^2 = \|\hat{Y} - \bar{Y}\mathbf{1}\|_2^2 + \|\hat\varepsilon\|_2^2$）。
- **子空间夹角余弦平方（Squared Cosine of Subspace Angle）**：

  $$
  R^2 = \cos^2(\theta)
  $$

  其中 $\theta$ 是观测向量与特征超平面之间的几何空间夹角。若 $Y$ 完全躺在特征子空间内，$\theta = 0^\circ \implies R^2 = 1$（完全拟合）；若 $Y$ 垂直于特征子空间，$\theta = 90^\circ \implies R^2 = 0$（自变量无线性解释力）。

```anova-variance-demo
```

---

### 3. 多元线性回归的矩阵表达与正规方程
多元线性回归模型的矩阵形式：

$$
Y = X\beta + \varepsilon
$$

其中因变量向量 $Y \in \mathbb{R}^{n \times 1}$，设计矩阵 $X \in \mathbb{R}^{n \times k}$（假定列满秩 $\operatorname{rank}(X) = k \le n$），参数列向量 $\beta \in \mathbb{R}^{k \times 1}$。

OLS 最小化残差平方和：

$$
RSS(\beta) = \|Y - X\beta\|_2^2 = (Y - X\beta)^\top (Y - X\beta) = Y^\top Y - 2\beta^\top X^\top Y + \beta^\top X^\top X \beta
$$

一阶驻点条件令梯度为零：

$$
\nabla_\beta RSS(\beta) = -2 X^\top Y + 2 X^\top X \beta = \mathbf{0}
$$

导出**正规方程（Normal Equations）**：

$$
X^\top X \hat\beta = X^\top Y
$$

由于 $X$ 满列秩，$X^\top X$ 严格对称正定可逆，得到唯一解析封闭解：

$$
\hat\beta = (X^\top X)^{-1} X^\top Y
$$

- **正交投影最短距离原理（Orthogonal Projection Principle）**：$Y \in \mathbb{R}^n$ 是悬浮在 $n$ 维空间中的目标点，$X\beta$ 是由 $X$ 的 $k$ 个列向量张成的 $k$ 维超平面 $\operatorname{Col}(X)$（“地面”）。在地面上寻找距离 $Y$ 欧氏距离最近的点 $\hat{Y} = X\hat\beta$，连接两点的误差线段 $Y - \hat{Y}$ 必须是垂直于地面的垂线。垂线垂直于地面上的每一根基底向量 $X_j$（即 $X_j^\top (Y - X\hat\beta) = 0$），将所有列向量堆叠即得正规方程 $X^\top(Y - X\hat\beta) = \mathbf{0}$。

---

### 4. 残差的正交性与投影算子（Residual Orthogonality & Projection Operators）
定义拟合值向量 $\hat{Y} = X\hat\beta$ 与样本残差向量 $\hat\varepsilon = Y - \hat{Y} = Y - X\hat\beta$。

- **帽子矩阵 $H$（正交投影算子）**：

  $$
  H = X(X^\top X)^{-1} X^\top
  $$

  - **几何直观（垂直聚光灯）**：将空间中任意向量正交压向特征超平面 $\operatorname{Col}(X)$，使得 $HY = \hat{Y}$。对称幂等性（$H^2 = H, H^\top = H$）表明：落到地面的点，再次投影坐标保持不变。
  - **迹与几何维数**：$\operatorname{tr}(H) = \operatorname{tr}(X(X^\top X)^{-1} X^\top) = \operatorname{tr}((X^\top X)^{-1} X^\top X) = \operatorname{tr}(I_k) = k$。特征空间的物理维数（自由度）即为 $k$。

- **消除矩阵 $M$（残差投影算子）**：

  $$
  M = I - H
  $$

  - **几何直观（垂直分量提取器）**：将任意向量投影至正交补空间 $\operatorname{Col}(X)^\perp$，滤除所有平行于地面的分量，仅提取纯垂直残差 $MY = \hat\varepsilon$。
  - **正交互补性**：$H + M = I, HM = \mathbf{0}$，且 $\operatorname{tr}(M) = n - k$（正交补空间的几何维数）。

#### 残差正交性的五大代数与几何性质

1. **残差与所有解释变量正交（$X^\top \hat\varepsilon = \mathbf{0}$）**：
   由正规方程直接给出：$X^\top(Y - X\hat\beta) = \mathbf{0} \implies X^\top \hat\varepsilon = \mathbf{0}$。
   - **几何含义**：误差向量 $\hat\varepsilon$ 垂直于由 $X$ 的所有列向量张成的超平面 $\operatorname{Col}(X)$。自变量中的所有线性信号已被 $\hat\beta$ 榨取殆尽，残差中不存在任何沿 $X$ 的投影分量。
2. **截距项的力学平衡：残差和恒为 0（$\mathbf{1}^\top \hat\varepsilon = 0 \implies \bar{\hat\varepsilon} = 0$）**：
   若模型包含常数截距项 $\beta_0$，则 $X$ 的第一列为全 1 向量 $X_0 = \mathbf{1}$。
   由 $X_0^\top \hat\varepsilon = 0$ 立即导出：

   $$
   \mathbf{1}^\top \hat\varepsilon = \sum_{i=1}^n \hat\varepsilon_i = 0 \implies \bar{\hat\varepsilon} = \frac{1}{n} \sum_{i=1}^n \hat\varepsilon_i \equiv 0
   $$

   - **几何与物理含义**：全 1 向量位于特征超平面内，垂直于该平面的残差向量与全 1 向量内积必为 0。物理上对应系统总力矩平衡，回归超平面必然精确穿透样本重心 $(\bar{X}, \bar{Y})$。
   - **注**：若无截距项（强制拟合过原点），$\mathbf{1} \notin \operatorname{Col}(X)$，残差和通常不为 0。
3. **残差与拟合值正交（$\hat{Y}^\top \hat\varepsilon = 0$）**：
   因为 $\hat{Y} = X\hat\beta \in \operatorname{Col}(X)$：

   $$
   \hat{Y}^\top \hat\varepsilon = (X\hat\beta)^\top \hat\varepsilon = \hat\beta^\top (X^\top \hat\varepsilon) = \hat\beta^\top \mathbf{0} = 0
   $$

   - **几何含义**：地面的向量（拟合值）与天花板垂线（残差）在空间中必然垂直。
4. **高维勾股定理与方差分解推导（$TSS = ESS + RSS$）**：
   由 $Y = \hat{Y} + \hat\varepsilon$ 且 $\hat{Y} \perp \hat\varepsilon$，去中心化后：

   $$
   (Y - \bar{Y}\mathbf{1}) = (\hat{Y} - \bar{Y}\mathbf{1}) + \hat\varepsilon
   $$

   两部分内积交叉项精确为零：

   $$
   (\hat{Y} - \bar{Y}\mathbf{1})^\top \hat\varepsilon = \hat{Y}^\top \hat\varepsilon - \bar{Y}(\mathbf{1}^\top \hat\varepsilon) = 0 - 0 = 0
   $$

   直接导出方差分解恒等式：

   $$
   \underbrace{\sum_{i=1}^n (Y_i - \bar{Y})^2}_{TSS} = \underbrace{\sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2}_{ESS} + \underbrace{\sum_{i=1}^n \hat\varepsilon_i^2}_{RSS} \implies R^2 = \frac{ESS}{TSS} = 1 - \frac{RSS}{TSS} \in [0, 1]
   $$

5. **核心辨析：样本残差代数正交 vs. 总体误差外生性假定**：
   - **样本残差代数正交（$X^\top \hat\varepsilon = \mathbf{0}$）**：纯代数数值恒等式。只要执行 OLS 求解，正规方程机械保证残差垂直于自变量，与真实物理规律是否线性、是否存在异方差或测量误差完全无关。
   - **总体误差外生性（$E(\varepsilon \mid X) = \mathbf{0} \implies E(X^\top \varepsilon) = \mathbf{0}$）**：总体因果统计假设。要求不可观测的真实误差中不包含与 $X$ 相关的隐藏变量。
   - **核心推论**：在存在遗漏变量（OVB）的模型中，计算出的样本残差 $\hat\varepsilon$ 依然与纳入模型的自变量**代数正交**；但不可观测的真实扰动 $\varepsilon$ 与自变量已经**不再正交**，导致估计系数产生内生性偏误。

---

### 5. 回归系数与协方差（Covariance）的深刻内在联系

#### （1）单变量回归：协方差与自变量方差之商

在有限样本 OLS 估计层面，参数解析解为：

$$
\hat\beta_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2} = \frac{\widehat{\operatorname{Cov}}(X, Y)}{\widehat{\operatorname{Var}}(X)} = \hat\rho_{XY} \frac{s_Y}{s_X}
$$

$$
\hat\beta_0 = \bar{Y} - \hat\beta_1 \bar{X}, \quad R^2 = \hat\rho_{XY}^2
$$

在理论总体分布（总体最佳线性预测 BLP）层面，对应参数关系为：

$$
\beta_1 = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)} = \rho \frac{\sigma_Y}{\sigma_X}
$$

- **物理含义与量纲分析**：相关系数 $\hat\rho_{XY} \in [-1, 1]$（或总体 $\rho$）度量纯粹的无量纲线性关联紧密性；而标准差之比 $\frac{s_Y}{s_X}$（或总体 $\frac{\sigma_Y}{\sigma_X}$）提供物理量纲转换因子。回归斜率 $\beta_1$ 带有确切的物理单位 $[Y]/[X]$，表示自变量每变动一个单位时因变量的边际变动响应率。
- **标准化数据**：当特征与目标被标准化处理（即样本 $s_X = s_Y = 1$ 或总体 $\sigma_X = \sigma_Y = 1$）时，标准差之比退化为 1，回归斜率与相关系数严格重合：$\hat\beta_1 = \hat\rho_{XY}$（总体 $\beta_1 = \rho$）。

#### （2）回归的非对称性与均值回归（Regression to the Mean）
分别考察总体理论分布与有限样本估计量下的双向回归关系：

- **总体理论关系**：
  $$\beta_{Y \sim X} = \rho \frac{\sigma_Y}{\sigma_X}, \quad \beta_{X \sim Y} = \rho \frac{\sigma_X}{\sigma_Y} \implies \beta_{Y \sim X} \times \beta_{X \sim Y} = \rho^2 \le 1$$
- **有限样本估计关系**：
  $$\hat\beta_{Y \sim X} = \hat\rho_{XY} \frac{s_Y}{s_X}, \quad \hat\beta_{X \sim Y} = \hat\rho_{XY} \frac{s_X}{s_Y} \implies \hat\beta_{Y \sim X} \times \hat\beta_{X \sim Y} = \hat\rho_{XY}^2 \le 1$$

- **物理直观（噪声稀释效应与均值收缩）**：反向回归斜率绝不是正向斜率的简单倒数，而是 $\beta_{X \sim Y} = \frac{\rho^2}{\beta_{Y \sim X}} < \frac{1}{\beta_{Y \sim X}}$（当且仅当数据存在不可预测的噪声导致 $|\rho| < 1$ 时）。测量与数据生成中客观存在的随机波动稀释了确定性因果信号；因此从任意一侧反推另一侧时，极值观测的期望预测均被系统性拉向总体中心均值（高个子父母的子女倾向于矮于父母，极端收益率资产倾向于回归平均）。

#### （3）多元回归的协方差表达：线性白化去相关

对自变量与因变量中心化后：

$$
\hat\beta = (X^\top X)^{-1} X^\top Y = \hat{\boldsymbol{\Sigma}}_{XX}^{-1} \hat{\boldsymbol{\Sigma}}_{XY}
$$

- **物理直观（去相关白化滤镜）**：若特征互不相关（$\hat{\boldsymbol{\Sigma}}_{XX}$ 为对角阵），多元回归解耦为多个独立的单变量回归；若特征相互混杂，$\hat{\boldsymbol{\Sigma}}_{XX}^{-1}$ 扮演**线性去相关（Whitening）算子**，剔除所有间接共动路径，精准剥离出各个特征独占的净边际贡献。

#### （4）偏协方差与 Frisch–Waugh–Lovell (FWL) 定理

多元回归中单个特征 $X_j$ 的系数满足：

$$
\hat\beta_j = \frac{\operatorname{Cov}(\tilde{X}_j, Y)}{\operatorname{Var}(\tilde{X}_j)} = \frac{\operatorname{Cov}(\tilde{X}_j, \tilde{Y})}{\operatorname{Var}(\tilde{X}_j)}
$$

其中 $\tilde{X}_j$ 是 $X_j$ 对其余所有特征 $X_{-j}$ 回归后的正交残差，$\tilde{Y}$ 是 $Y$ 对 $X_{-j}$ 回归后的正交残差。

- **几何直观（子空间正交解耦 Subspace De-aliasing）**：要想探知 $X_j$ 对 $Y$ 的纯净边际作用，必须先将混杂特征 $X_{-j}$ 张成的子空间从 $X_j$ 和 $Y$ 中分别投影剔除（滤清间接混杂），再拿两者剥离出的纯净正交分量做单变量回归。
- **方差膨胀因子（Variance Inflation Factor, VIF）**：

  $$
  \operatorname{Var}(\hat\beta_j \mid X) = \frac{\sigma^2}{(n-1)\operatorname{Var}(X_j)} \cdot \underbrace{\frac{1}{1 - R_{j \mid -j}^2}}_{\mathrm{VIF}_j}
  $$

  - **几何直观（极短力臂放大抖动）**：$1 - R_{j \mid -j}^2 = \sin^2(\theta_j)$，其中 $\theta_j$ 为 $X_j$ 与其余特征子平面的空间夹角。当多重共线性极高时，$\theta_j \to 0$，残差垂直力臂 $\tilde{X}_j$ 长度急剧萎缩至接近 0。用极其短小的力臂去杠杆平衡输出响应，数据中的微小扰动会导致回归平面沿该轴剧烈晃动，估计方差发生灾难性膨胀。

#### （5）量化金融典型应用映射
1. **CAPM 资产 Beta**：$\beta_i = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)}$；
2. **方差最小化最优套保比率（Optimal Hedge Ratio）**：$\min_h \operatorname{Var}(\Delta S - h\Delta F) \implies h^* = \frac{\operatorname{Cov}(\Delta S, \Delta F)}{\operatorname{Var}(\Delta F)} \equiv \beta_{\Delta S \sim \Delta F}$；
3. **遗漏变量偏差（OVB）**：真实模型 $Y = \beta_1 X_1 + \beta_2 X_2 + \varepsilon$，遗漏 $X_2$ 的短回归估计量期望为 $E(\hat\beta_1^{\text{short}} \mid X) = \beta_1 + \beta_2 \frac{\operatorname{Cov}(X_1, X_2)}{\operatorname{Var}(X_1)}$；
4. **Barra 风险因子正交中性化**：$F_{\text{raw}} = X_{\text{risk}} \gamma + F_{\text{neutral}}$，利用正交投影 $F_{\text{neutral}} \perp X_{\text{risk}}$ 彻底剥离行业与风格风险暴露。

---

## 模块二：Gauss–Markov 定理、统计推断与核心 Lemma 全景清单

本模块系统梳理 Gauss–Markov 假定、推断分布及 7 大核心 Lemma 的数学推导与物理/几何直觉。

---

### 1. Gauss-Markov 假设与 BLUE 本质
Gauss-Markov 定理指出，在线性模型基本假设满足时，OLS 估计量是**最佳线性无偏估计量（Best Linear Unbiased Estimator, BLUE）**，即在所有线性无偏估计量中，OLS 的协方差矩阵在半正定意义下达到最小方差。

1. **参数线性（Linearity in parameters）**：模型形式为 $Y = X\beta + \varepsilon$；
2. **严格外生性（Strict Exogeneity）**：$E(\varepsilon \mid X) = \mathbf{0}$；
3. **球形扰动项（Spherical Errors）**：
   - **同方差性（Homoskedasticity）**：$\operatorname{Var}(\varepsilon_i \mid X) = \sigma^2$；
   - **无自相关性（No Autocorrelation）**：$\operatorname{Cov}(\varepsilon_i, \varepsilon_j \mid X) = 0 \quad (i \ne j)$；
   - 矩阵统一形式：$\operatorname{Var}(\varepsilon \mid X) = \sigma^2 I_n$；
4. **无完全多重共线性（No Full Multicollinearity）**：$\operatorname{rank}(X) = k \le n$。

- **核心辨析：正态性（Normality）的作用边界**：
  OLS 成为 BLUE **完全不需要假设误差项服从正态分布**。该定理仅依赖一阶矩（外生性）与二阶矩（球形扰动）假定。正态性假定仅在**小样本有限自由度下进行精确 $t$ 检验、$F$ 检验**，以及证明 OLS 达到 Cramér–Rao 下界（成为一致最小方差无偏估计量 UMVUE）时才需要。

---

### 2. 核心分析 Lemma 清单

#### 【Lemma 1】OLS 估计量的代数与矩性质
- **线性形式**：$\hat\beta = (X^\top X)^{-1} X^\top Y = C Y$，其中线性权重矩阵 $C = (X^\top X)^{-1} X^\top$ 满足 $C X = I_k$。
- **条件无偏性（Unbiasedness）**：

  $$
  E(\hat\beta \mid X) = E(C(X\beta + \varepsilon) \mid X) = \beta + C E(\varepsilon \mid X) = \beta
  $$

- **条件协方差矩阵（Variance-Covariance Matrix）**：

  $$
  \operatorname{Var}(\hat\beta \mid X) = \operatorname{Var}(CY \mid X) = C \operatorname{Var}(\varepsilon \mid X) C^\top = C (\sigma^2 I_n) C^\top = \sigma^2 (X^\top X)^{-1}
  $$

  - 单系数估计方差：$\operatorname{Var}(\hat\beta_j \mid X) = \sigma^2 [(X^\top X)^{-1}]_{jj}$；
  - 双系数估计协方差：$\operatorname{Cov}(\hat\beta_j, \hat\beta_m \mid X) = \sigma^2 [(X^\top X)^{-1}]_{jm}$。
- **残差方差的无偏估计量与二次型期望引理**：

  $$
  \hat\sigma^2 = \frac{\hat\varepsilon^\top \hat\varepsilon}{n - k} = \frac{\sum_{i=1}^n \hat\varepsilon_i^2}{n - k}
  $$

  - **几何与物理直观（正交补空间维数容量）**：残差为 $\hat\varepsilon = (I - H)\varepsilon$。原始 $n$ 维观测空间被拟合超平面占去了 $k$ 个自由度，残差被严格禁锢在维度为 $n - k$ 的正交补空间内自由摆动。二次型期望公式：

    $$
    E(\hat\varepsilon^\top \hat\varepsilon \mid X) = E(\varepsilon^\top (I - H) \varepsilon \mid X) = \operatorname{tr}\left( (I - H) \sigma^2 I_n \right) = \sigma^2 \operatorname{tr}(I - H) = \sigma^2 (n - k)
    $$

    除以 $n - k$ 恰好归一化正交补空间的几何容量，从而给出每个自由度上真实物理噪声方差 $\sigma^2$ 的无偏估计。

#### 【Lemma 2】正态假定下的统计分布引理
当误差项满足球形高斯假定 $\varepsilon \mid X \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_n)$ 时：
- **统计独立性引理（Cochran 定理子空间解耦）**：

  $$
  \hat\beta \text{ 与样本残差 } \hat\varepsilon \text{（以及方差估计 } \hat\sigma^2 \text{）严格统计独立！}
  $$

  - **几何与物理直观**：在球形高斯测度下，两个几何相互正交的子空间在统计上必然相互独立。由于 $\hat\beta$ 仅取决于投影点 $\hat{Y} \in \operatorname{Col}(X)$，而 $\hat\varepsilon \in \operatorname{Col}(X)^\perp$ 位于完全正交的补空间中，互协方差恒为零：

    $$
    \operatorname{Cov}(\hat\beta, \hat\varepsilon \mid X) = \sigma^2 C (I - H)^\top = \sigma^2 \left( (X^\top X)^{-1}X^\top - (X^\top X)^{-1}X^\top H \right) = \mathbf{0}
    $$

    多元正态分布下协方差为零等价于严格独立，因此模型拟合系数与残差波动统计脱耦。
- **残差平方和卡方分布**：

  $$
  \frac{\hat\varepsilon^\top \hat\varepsilon}{\sigma^2} = \frac{(n - k)\hat\sigma^2}{\sigma^2} \sim \chi^2(n - k)
  $$

- **单参数 $t$ 检验统计量**：
  检验假设 $H_0: \beta_j = \beta_{j,0}$：

  $$
  t = \frac{\hat\beta_j - \beta_{j,0}}{\sqrt{\hat\sigma^2 [(X^\top X)^{-1}]_{jj}}} \sim t_{n-k}
  $$

- **联合线性约束 $F$ 检验统计量**：
  检验 $q$ 个线性联合假设 $H_0: R\beta = r$（$R \in \mathbb{R}^{q \times k}$ 满行秩）：

  $$
  F = \frac{(R\hat\beta - r)^\top [R(X^\top X)^{-1} R^\top]^{-1} (R\hat\beta - r) / q}{\hat\sigma^2} \sim F_{q, n-k}
  $$

  - **受限模型与非受限模型残差比形式**：

    $$
    F = \frac{(RSS_R - RSS_{UR}) / q}{RSS_{UR} / (n - k)} = \frac{(R_{UR}^2 - R_R^2) / q}{(1 - R_{UR}^2) / (n - k)} \sim F_{q, n-k}
    $$

  - **单约束代数恒等**：当约束数 $q = 1$ 时，$t^2 \equiv F$。

#### 【Lemma 3】预测区间 vs. 置信区间引理
针对新查询特征点 $X_0 \in \mathbb{R}^k$：
- **条件均值响应的置信区间（Confidence Interval for $E(Y_0 \mid X_0) = X_0^\top \beta$）**：
  估计值 $\hat{Y}_0 = X_0^\top \hat\beta$，方差仅源自参数估计抽样误差：

  $$
  \operatorname{Var}(\hat{Y}_0 \mid X) = X_0^\top \operatorname{Var}(\hat\beta \mid X) X_0 = \sigma^2 X_0^\top (X^\top X)^{-1} X_0
  $$

  $1-\alpha$ 置信区间：

  $$
  \hat{Y}_0 \pm t_{n-k, 1-\alpha/2} \cdot \sqrt{\hat\sigma^2 X_0^\top (X^\top X)^{-1} X_0}
  $$

- **单个新样本值的预测区间（Prediction Interval for $Y_0 = X_0^\top \beta + \varepsilon_0$）**：
  预测误差为 $e_0 = Y_0 - \hat{Y}_0 = \varepsilon_0 - X_0^\top(\hat\beta - \beta)$。由于未来样本独立扰动 $\varepsilon_0$ 独立于历史样本：

  $$
  \operatorname{Var}(e_0 \mid X) = \operatorname{Var}(\varepsilon_0) + \operatorname{Var}(\hat{Y}_0 \mid X) = \sigma^2 \left[ 1 + X_0^\top (X^\top X)^{-1} X_0 \right]
  $$

  $1-\alpha$ 预测区间：

  $$
  \hat{Y}_0 \pm t_{n-k, 1-\alpha/2} \cdot \sqrt{\hat\sigma^2 \left[ 1 + X_0^\top (X^\top X)^{-1} X_0 \right]}
  $$

- **物理本质辨析（群体均值中心 vs 单个随机粒子）**：置信区间量化的是“测量群体均值超平面的位置摆动”，随着样本容量 $n \to \infty$，估计方差收敛至 0；而预测区间面对的是“未来新到来的单个离散粒子”，粒子自带不可消除的物理热运动白噪声 $\varepsilon_0 \sim \mathcal{N}(0, \sigma^2)$。因此预测区间方差恒比置信区间多出不可消除的 $\sigma^2$ 项，即使 $n \to \infty$，其宽度亦存在 $\pm z_{\alpha/2}\sigma$ 的物理极限底线。

#### 【Lemma 4】留一法与杠杆值引理（LOOCV & Leverage）
- **帽子矩阵对角元（杠杆值 Leverage $H_{ii}$）**：

  $$
  H_{ii} = X_i^\top (X^\top X)^{-1} X_i
  $$

  满足 $0 \le H_{ii} \le 1$，$\sum_{i=1}^n H_{ii} = k$，平均杠杆值 $\bar{H} = k/n$。
  - **几何直观（阿基米德杠杆臂）**：$H_{ii}$ 本质是样本点 $X_i$ 在特征空间中距离质心的马氏距离（Mahalanobis Distance）。离群样本点拥有超长力矩臂（$H_{ii} \to 1$），单手即可拉偏整个拟合平面的倾角。
- **留一残差解析捷径（Sherman–Morrison 公式）**：
  无需重新训练 $n$ 次模型，剔除第 $i$ 个样本后的模型在 $X_i$ 处的测试残差由原样本残差闭式缩放得到：

  $$
  \hat\varepsilon_{(-i)} = Y_i - \hat{Y}_{(-i)} = \frac{\hat\varepsilon_i}{1 - H_{ii}}
  $$

  由此一步得出严谨的留一交叉验证（LOOCV）误差：

  $$
  \mathrm{LOOCV} = \frac{1}{n} \sum_{i=1}^n \left( \frac{\hat\varepsilon_i}{1 - H_{ii}} \right)^2
  $$

  - **物理直观（回弹力矩释放）**：样本点利用自身杠杆力臂将拟合线硬拉近了自己（原残差 $\hat\varepsilon_i$ 被人为压小）；将其剔除后，回归线瞬间释放弹性势能向反方向回弹，真实留一误差正是原残差以 $\frac{1}{1 - H_{ii}}$ 倍数剧烈放大。

#### 【Lemma 5】遗漏变量与多余变量引理（OVB & Irrelevant Regressors）
- **遗漏变量偏差（Omitted Variable Bias, OVB）**：
  若真实数据生成机制为 $Y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon$。若遗漏 $X_2$ 仅对 $X_1$ 进行短回归：

  $$
  E(\hat\beta_1^{\text{short}} \mid X) = \beta_1 + \underbrace{(X_1^\top X_1)^{-1} X_1^\top X_2}_{\hat\Gamma_{2 \sim 1}} \beta_2
  $$

  - **几何直观（正交投影的阴影污染）**：若遗漏的 $X_2$ 与入选的 $X_1$ 存在空间共线性（$X_1^\top X_2 \ne \mathbf{0}$），$X_2$ 对 $Y$ 的真实作用力会在 $X_1$ 上投射下一道“因果投影阴影”。短回归无法辨别阴影来源，错误地把阴影份额强加在 $\beta_1$ 头上。只有当 $\beta_2 = \mathbf{0}$（遗漏特征无影响）或 $X_1 \perp X_2$（投影阴影垂直为 0）时，短回归才无偏。
- **纳入无关冗余变量（Overfitting / Irrelevant Regressor）**：
  若真实模型不含 $X_2$（$\beta_2 = \mathbf{0}$），但人为纳入 $X_2$ 回归：
  - 参数估计依然无偏：$E(\hat\beta_1^{\text{long}}) = \beta_1$；
  - 但估计方差必然膨胀：$\operatorname{Var}(\hat\beta_1^{\text{long}}) \ge \operatorname{Var}(\hat\beta_1^{\text{short}})$，且相等当且仅当 $X_1 \perp X_2$。

#### 【Lemma 6】测量误差引理（Errors-in-Variables / Attenuation Bias）
- **自变量测量误差与衰减偏误**：
  真实模型 $Y_i = \beta_0 + \beta_1 X_i^* + \varepsilon_i$，观测值掺入加性测量白噪声 $X_i = X_i^* + u_i$（$u_i \sim (0, \sigma_u^2)$ 独立于 $X_i^*, \varepsilon_i$）：

  $$
  \operatorname{plim}_{n \to \infty} \hat\beta_1 = \beta_1 \cdot \frac{\sigma_{X^*}^2}{\sigma_{X^*}^2 + \sigma_u^2} < \beta_1
  $$

  - **物理直观（信噪比稀释机理）**：自变量掺入噪声相当于在纯净信号中混入泥沙，将数据点在横轴方向人为吹散，使得真实的斜率被硬生生拉平，导致估计系数向 0 系统性收缩衰减。
- **因变量测量误差**：
  若因变量观测存在噪声 $Y_i = Y_i^* + v_i$（$v_i$ 独立于 $X_i$），则 $\hat\beta_1$ **仍然无偏且一致**，物理噪声仅被并入扰动方差 $\sigma^2 + \sigma_v^2$，带来标准误变大与检验功效下降。

#### 【Lemma 7】变量尺度与仿射变换引理（Scale & Affine Invariance）
- **自变量缩放**：若 $X_{\text{new}} = c \cdot X$，则 $\hat\beta_{\text{new}} = \frac{1}{c} \hat\beta$；
- **因变量缩放**：若 $Y_{\text{new}} = d \cdot Y$，则 $\hat\beta_{\text{new}} = d \cdot \hat\beta$；
- **自变量/因变量平移**：若 $X$ 或 $Y$ 增加平移常数，**斜率 $\hat\beta_1$ 绝对不变**，仅截距 $\hat\beta_0$ 发生对应平移；
- **不变性定律**：非零尺度缩放与平移变换下，**$t$ 检验量、$F$ 检验量、$R^2$、回归 $p$ 值完全保持数值不变**。
  - **几何直观（欧氏保角性）**：度量衡单位改变（如米改毫米）本质只是改变了坐标轴的刻度，并未改变高维空间中向量之间的夹角 $\theta$。所有基于夹角余弦的统计量（$R^2 = \cos^2\theta$，$t \propto \cot\theta$）均属于空间几何无量纲不变量。

---

### 3. 违反假设的后果与补救（White / Newey-West / GLS）
当真实数据违背 Gauss-Markov 假设时：
- **异方差（Heteroskedasticity）与自相关（Autocorrelation）**：
  OLS 估计量**依然无偏且一致**，但不再是 BLUE。普通标准误公式 $\sigma^2 (X^\top X)^{-1}$ 被严重低估，产生虚假统计显著性。
- **补救方案**：
  1. **White 异方差稳健标准误（HC0 / Sandwich Estimator）**：

     $$
     \operatorname{Var}_{\text{White}}(\hat\beta) = (X^\top X)^{-1} \left( \sum_{i=1}^n \hat\varepsilon_i^2 X_i X_i^\top \right) (X^\top X)^{-1}
     $$

     - **物理直观（三明治构造）**：外层的“两片面包” $(X^\top X)^{-1}$ 负责坐标空间的基底投影变换，中间的“夹心肉” $X^\top \hat{\Omega} X = \sum_{i=1}^n \hat\varepsilon_i^2 X_i X_i^\top$ 捕获每个样本点真实的局部异方差能量。
  2. **Newey–West 异方差自相关稳健标准误（HAC）**：
     引入 Bartlett 滞后三角核函数修正自相关截断，为时间序列金融数据标准配置。
  3. **广义最小二乘法（GLS / WLS, Aitken 定理）**：
     已知协方差结构 $\operatorname{Var}(\varepsilon \mid X) = \sigma^2 \boldsymbol{\Omega}$ 时，令空间白化矩阵 $P = \boldsymbol{\Omega}^{-1/2}$ 对原方程预乘变换：

     $$
     \hat\beta_{\text{GLS}} = (X^\top \boldsymbol{\Omega}^{-1} X)^{-1} X^\top \boldsymbol{\Omega}^{-1} Y
     $$

     - **几何直观（空间白化与马氏度量）**：当扰动项在空间中呈倾斜或拉伸的椭球状分布时，普通欧氏距离失效。预乘 $\boldsymbol{\Omega}^{-1/2}$ 将椭球逆向旋转压缩为标准正球体，在该白化坐标系下运行标准 OLS 正交投影，重新达到最佳线性无偏（BLUE）。

---

## 模块三：变量选择与收缩（ESL 3.3–3.4）

面对大量可能存在共线性的特征因子，我们需要对模型进行限制（Regularization）。

### 1. 传统方法（Subset Selection）
- **最优子集（Best Subset）** / **逐步回归（Forward / Backward Stepwise）**：在高层面上，这些离散的选择过程能够选出较好的变量，但是由于选择的离散性，方差通常较大。

### 2. 岭回归（Ridge Regression）
引入 $\ell_2$ 范数惩罚项来控制系数大小：

$$
\hat\beta^{\mathrm{ridge}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2
$$

其封闭解为：

$$
\hat\beta^{\mathrm{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y
$$

**特点**：极好地处理多重共线性问题（引入偏差，降低方差）；**不会将任何系数精确收缩到 0**。

#### 数据增广（Data Augmentation）推导闭式解：将 Ridge 等价还原为标准 OLS

在量化金融与统计学习推导中，一个极为优美且实用的代数技巧是：**无需对目标函数求矩阵导数，仅通过在原数据矩阵后追加单位矩阵构建“虚拟观测数据”（Data Augmentation），即可直接将岭回归完全还原为普通的 OLS 问题并导出其闭式解**。

##### 1. 增广矩阵构造（Augmented System Formulation）

注意到目标函数中的 $\ell_2$ 惩罚项可以写为以 0 为目标、以单位阵为特征的残差平方和形式：

$$
\lambda \|\beta\|_2^2 = \|\mathbf{0}_{p \times 1} - \sqrt{\lambda} I_p \beta\|_2^2
$$

因此，我们将原始设计矩阵 $X \in \mathbb{R}^{N \times p}$ 与目标向量 $y \in \mathbb{R}^{N \times 1}$ 垂直拼接（Vertical Concatenation），构造增广设计矩阵 $\tilde{X}$ 与增广目标向量 $\tilde{y}$：

$$
\tilde{X} = \begin{bmatrix} X \\ \sqrt{\lambda} I_p \end{bmatrix} \in \mathbb{R}^{(N + p) \times p}, \quad \tilde{y} = \begin{bmatrix} y \\ \mathbf{0}_{p \times 1} \end{bmatrix} \in \mathbb{R}^{(N + p) \times 1}
$$

##### 2. 目标函数严格恒等性证明

在增广数据集 $(\tilde{X}, \tilde{y})$ 上定义标准未加惩罚的 OLS 残差平方和目标函数：

$$
\begin{aligned}
\mathcal{L}_{\text{OLS}}(\beta; \tilde{X}, \tilde{y}) &= \|\tilde{y} - \tilde{X}\beta\|_2^2 \\
&= \left\| \begin{bmatrix} y \\ \mathbf{0} \end{bmatrix} - \begin{bmatrix} X \\ \sqrt{\lambda} I_p \end{bmatrix} \beta \right\|_2^2 \\
&= \left\| \begin{bmatrix} y - X\beta \\ -\sqrt{\lambda} I_p \beta \end{bmatrix} \right\|_2^2 \\
&= \|y - X\beta\|_2^2 + \|-\sqrt{\lambda} I_p \beta\|_2^2 \\
&= \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2
\end{aligned}
$$

该增广系统的残差平方和与岭回归的优化目标**在数学上完全恒等**！

##### 3. 利用 OLS 正规方程直接写出闭式解

由于增广问题是一个标准的无约束 OLS 回归，其全局最优解直接由经典的 OLS 正规方程（Normal Equations）给出：

$$
\hat\beta^{\mathrm{ridge}} = (\tilde{X}^\top \tilde{X})^{-1} \tilde{X}^\top \tilde{y}
$$

分别展开两项分块矩阵乘法：

$$
\tilde{X}^\top \tilde{X} = \begin{bmatrix} X^\top & \sqrt{\lambda} I_p \end{bmatrix} \begin{bmatrix} X \\ \sqrt{\lambda} I_p \end{bmatrix} = X^\top X + (\sqrt{\lambda} I_p)(\sqrt{\lambda} I_p) = X^\top X + \lambda I_p
$$

$$
\tilde{X}^\top \tilde{y} = \begin{bmatrix} X^\top & \sqrt{\lambda} I_p \end{bmatrix} \begin{bmatrix} y \\ \mathbf{0} \end{bmatrix} = X^\top y + \sqrt{\lambda} I_p \mathbf{0} = X^\top y
$$

代入即直接得到岭回归的显式闭式解：

$$
\hat\beta^{\mathrm{ridge}} = (X^\top X + \lambda I_p)^{-1} X^\top y
$$

##### 4. 增广数据视角的统计、几何与工程深刻洞见

- **物理与几何直觉（Virtual Observations Pulling to Zero）**：
  数据增广等价于在真实数据之外，人为追加了 $p$ 个**虚拟单变量探测实验**。第 $j$ 个虚拟样本的特征为 $x_{\text{pseudo}, j} = \sqrt{\lambda} \mathbf{e}_j$（仅第 $j$ 个特征为 $\sqrt{\lambda}$，其余全为 0），其观测响应为 $y_{\text{pseudo}, j} = 0$。这 $p$ 个锚点在空间中对回归超平面产生刚性约束，一旦某个系数 $\beta_j$ 偏离 0，就会在虚拟样本上产生残差惩罚，从而将所有系数平滑拉向 0；
- **满秩与可逆性保证（Guaranteed Invertibility Even When $N < p$）**：
  当 $N < p$（高维小样本场景）或特征存在严重多重共线性时，原始矩阵 $X$ 的行数少于列数，$\text{rank}(X) \le N < p$，$X^\top X$ 必然奇异不可逆，标准 OLS 产生无穷多解。而在增广矩阵 $\tilde{X}$ 中，底部拼接的 $\sqrt{\lambda} I_p$ 拥有 $p$ 个严格线性无关的正交行，确保 $\text{rank}(\tilde{X}) = p$ 恒成立。
  对任意非零向量 $v \ne \mathbf{0}$：

  $$
  v^\top (X^\top X + \lambda I_p) v = \|Xv\|_2^2 + \lambda \|v\|_2^2 \ge \lambda \|v\|_2^2 > 0 \quad (\forall \lambda > 0)
  $$

  因此 $X^\top X + \lambda I_p$ 严格对称正定，保证闭式解必然存在且唯一；
- **数值工程优势（Numerical Stability via QR Decomposition）**：
  在工程实现中，直接显式计算正规方程中的 $X^\top X + \lambda I$ 会导致**矩阵条件数平方**（$\kappa(X^\top X + \lambda I) \approx \kappa(\tilde{X})^2$），在接近病态时引起严重的浮点舍入精度损失。借助数据增广形式，生产级线性代数库可以直接对 $(N+p) \times p$ 的增广矩阵 $\tilde{X}$ 执行**经济型 QR 分解**（Thin QR Decomposition）：$\tilde{X} = \tilde{Q} \tilde{R}$，然后通过回代求解上三角系统 $\tilde{R} \beta = \tilde{Q}^\top \tilde{y}$。这样条件数保持为 $\kappa(\tilde{X})$，彻底避免了显式求逆与条件数平方恶化；
- **贝叶斯先验与虚拟数据的对偶性（Bayesian Fictitious Data Duality）**：
  在贝叶斯线性回归中，高斯先验 $\beta \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$ 下的极大后验估计（MAP）完全等价于 Ridge 回归（$\lambda = \sigma^2 / \tau^2$）。数据增广证明了经典统计学与贝叶斯统计学的深层对偶：**对参数的高斯先验信念，完全等价于在样本空间中观测到了 $p$ 个均值为 0、精度由先验方差决定的虚拟先验数据（Fictitious Data）**。



### 3. Lasso 回归
引入 $\ell_1$ 范数惩罚项：
$$
\hat\beta^{\mathrm{lasso}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_1
$$
**特点**：因为 $\ell_1$ 惩罚的几何形状是尖锐的“菱形（Diamond）”，在等高线相切时极易切在坐标轴或顶点上，从而能够将部分系数**精确收缩到 0**，起到**内建的特征选择（Sparsity）**作用。

### 4. 方法对比矩阵

| 方法 | 惩罚项 | 偏差与方差 | 产生稀疏性？ | 能处理多重共线性？ |
| :--- | :--- | :--- | :---: | :--- |
| **最优子集** | 限制变量数 | 离散过程，高方差 | 是 | 视保留的子集而定 |
| **Ridge** | $\lambda \|\beta\|_2^2$ (圆球) | 引入偏差，降低方差 | 否 | 能极好地处理，解唯一 |
| **Lasso** | $\lambda \|\beta\|_1$ (菱形) | 引入偏差，降低方差 | 是 | 能，但对高度相关的变量随机选一个 |

*(注：PCR / PLS 等降维方法本质上是生成少量“衍生方向”（Derived Directions）进行回归，与带惩罚的变量选择思路不同，无需长篇赘述。)*

---

## 模块四：核平滑与局部回归（ESL 6.1–6.3）

在前三个模块中，我们深入剖析了线性回归（OLS、Ridge、Lasso）。这些经典方法都建立在**全局参数假定（Global Parametric Assumption）**之上：即假设真实函数在全空间满足 $f(X) = X\beta$。然而，在量化金融的诸多前沿场景（如期权隐含波动率曲面 Volatility Smile/Surface 拟合、高频订单流不平衡的价格冲击非线性曲线、以及局部 Alpha 因子挖掘）中，真实的函数关系往往呈现出高度弯曲或状态依赖性。

当我们希望摆脱全局线性的强加假设时，便走到了经典统计学与机器学习的交叉路口：**非参数平滑（Nonparametric Smoothing）**。本模块将循着 ESL 第 6 章的理论脉络，从最底层的条件期望出发，阐明“核（Kernel）”如何天然成为连接概率密度与回归的桥梁，并系统推导局部多项式回归的核心机理。

---

### 1. 理论根基：回归目标、核（Kernel）的本质与两大学派连接

#### （1）回归的统计本质：条件期望函数
在概率统计中，回归问题的终极目标是找到一个预测函数 $f(X)$，使得均方预测误差 $\mathbb{E}[(Y - f(X))^2]$ 最小化。根据全期望公式与正交投影性质，该问题的最优理论解唯一确定为**条件期望函数（Regression Function）**：
$$
f(x_0) = \mathbb{E}[Y \mid X = x_0] = \int y \, p(y \mid x_0) \, dy = \frac{\int y \, p(x_0, y) \, dy}{p(x_0)}
$$
- **全局参数学派（模块一至三）**：强行猜测 $f(x) \approx x^\top \beta$，用全体样本求解一组全局固定的权重 $\hat\beta$。优点是方差极小、计算快，但存在巨大的**模型设定偏误（Model Misspecification Bias）**。
- **非参数局域学派（本模块）**：完全不对 $f(x)$ 预设全局形式，而是遵循**记忆型学习（Memory-Based Learning / Lazy Learning）**——“想预测哪一点 $x_0$，就只看 $x_0$ 附近的邻居”。

#### （2）从条件期望到 Nadaraya–Watson：核密度估计的自然代入
既然条件期望是联合密度与边缘密度的积分商，统计学家 Nadaraya (1964) 与 Watson (1964) 提出了一个极具开创性的思想：**能否用非参数核密度估计（Parzen Window KDE）直接估计分子与分母？**
设核函数为 $K_\lambda(x_0, x) = \frac{1}{\lambda} D\left(\frac{|x - x_0|}{\lambda}\right)$：
1. **分母（输入边缘密度 $\hat{p}(x_0)$）**：
   $$ \hat{p}(x_0) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) $$
2. **分子（联合密度积分 $\int y \hat{p}(x_0, y) dy$）**：用二维独立乘积核估计联合密度 $\hat{p}(x_0, y) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) K_{h_y}(y, y_i)$，将其代入关于 $y$ 的积分：
   $$ \int y \, \hat{p}(x_0, y) \, dy = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) \underbrace{\int y K_{h_y}(y, y_i) \, dy}_{= y_i} = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) y_i $$
将分子与分母相除，便极其自然、毫无违和感地**精确推导出了 Nadaraya–Watson 核回归公式**：
$$
\hat{f}(x_0) = \frac{\int y \hat{p}(x_0, y) dy}{\hat{p}(x_0)} = \frac{\sum_{i=1}^N K_\lambda(x_0, x_i) y_i}{\sum_{i=1}^N K_\lambda(x_0, x_i)}
$$
**核心启示**：核回归并非人为拼凑的加权平均经验公式，它是概率论中**条件期望 $\mathbb{E}[Y \mid X=x]$ 在无参数假设下的 Plug-in（代入式）最优估计量**！

#### （3）机器学习经典辨析：局部化核（Localization Kernel） vs. 再生核（Mercer / RKHS Kernel）
ESL 第 6 章开篇特别强调：**切勿将本章的核（Kernel）与 SVM 中的“核技巧（Kernel Trick）”相混淆！**

| 比较维度 | 局部平滑核（Localization Kernel, ESL 第6章） | 再生核 / 算子核（Mercer / RKHS Kernel, ESL 第5.8/12章） |
| :--- | :--- | :--- |
| **数学定义** | 局域权重衰减窗函数 $K_\lambda(x_0, x_i) = D\left(\frac{\|x_i - x_0\|}{\lambda}\right)$ | 半正定连续核函数 $K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$ |
| **核心机制** | **在原始输入空间进行局部化邻域加权**（Memory-Based Localization） | **将特征隐式映射到高维/无穷维再生核希尔伯特空间（RKHS）** |
| **计算模式** | **惰性求值（Lazy Learning）**，训练期几乎零计算，计算全部发生在查询时刻 | **积极求值（Eager Learning）**，需在训练期求解全局对偶二次规划或核矩阵求逆 |
| **典型应用** | Nadaraya-Watson、局部线性回归（LOESS/Lowess）、波动率曲面平滑 | 支持向量机（SVM）、Kernel Ridge Regression、高斯过程（GP） |

#### （4）连续谱（Continuum Spectrum）：全局 OLS 到局部近邻的平滑过渡
局部加权最小二乘的目标函数为：
$$
\min_{\beta(x_0)} \sum_{i=1}^N K_\lambda(x_0, x_i) \left[ y_i - b(x_i)^\top \beta(x_0) \right]^2
$$
带宽 $\lambda$ 充当了调节全局刚性与局部柔性的“旋钮”：
- 当 **$\lambda \to \infty$** 时：核权重退化为均匀常数 $K_\lambda \to \text{const}$，局部回归**严格退化为全局普通最小二乘法（Global OLS）**（方差最低，但偏差受制于线性假设）；
- 当 **$\lambda \to 0$** 时：核权重仅在最接近 $x_0$ 的极少数样本点非零，局部回归**退化为最近邻插值（1-NN Interpolation）**（完全零偏差，但方差无限放大）；
- **有限带宽 $\lambda \in (0, \infty)$**：在全局模型（高偏差）与局部极值（高方差）之间构筑了一条完美的平滑连续过渡谱。

---

### 2. 从 k-NN 到 Nadaraya–Watson 核加权平均（ESL 6.1）
- **k-NN 局部均值的缺陷**：
  在点 $x$ 处取 $k$ 近邻平均 $\hat{f}(x) = \frac{1}{k}\sum_{x_i \in N_k(x)} y_i$。当查询点 $x$ 连续移动时，边界样本点以离散阶跃（0-1 权重突变）进出邻域 $N_k(x)$，导致拟合出的 $\hat{f}(x)$ 呈现不自然的锯齿状断裂（Bumpy & Discontinuous）。
- **Nadaraya–Watson 核估计量（1964）**：
  引入平滑衰减的**核权重函数** $K_\lambda(x_0, x_i) = D\left(\frac{|x_i - x_0|}{\lambda}\right)$，使得邻域样本权重随距离平滑衰减：
  $$
  \hat{f}(x_0) = \frac{\sum_{i=1}^N K_\lambda(x_0, x_i) y_i}{\sum_{i=1}^N K_\lambda(x_0, x_i)} = \sum_{i=1}^N l_i(x_0) y_i
  $$
  其中等价权重 $l_i(x_0) = \frac{K_\lambda(x_0, x_i)}{\sum_{j=1}^N K_\lambda(x_0, x_j)}$ 满足非负性且归一化 $\sum_{i=1}^N l_i(x_0) = 1$。
  - **局部常数（Local Constant）等价性**：Nadaraya-Watson 估计量严格等价于在 $x_0$ 邻域内求解一个加权最小二乘常数：
    $$
    \hat{f}(x_0) = \arg\min_c \sum_{i=1}^N K_\lambda(x_0, x_i)(y_i - c)^2
    $$
- **三大常用核函数对比**：
  1. **Epanechnikov 二次核**：$D(t) = \frac{3}{4}(1 - t^2) \cdot \mathbb{I}(|t| \le 1)$。紧支集（Compact Support）；在渐近均方误差（AMSE）意义下是方差最小的最优核，但在支集边界处一阶不可导。
  2. **Tri-cube 三次核（Cleveland LOESS 默认核）**：$D(t) = (1 - |t|^3)^3 \cdot \mathbb{I}(|t| \le 1)$。紧支集；在支集边界具有二阶连续导数，顶部更平坦，过渡更平滑。
  3. **高斯核（Gaussian Kernel）**：$D(t) = \frac{1}{\sqrt{2\pi}} e^{-t^2/2}$。全域无限支集，处处无限可微；以标准差充当带宽 $\lambda$。
- **带宽 $\lambda$ 与偏差-方差权衡（Bias-Variance Tradeoff）**：
  - $\lambda \to 0$（极窄窗口）：仅受极少数甚至单个点主导，$\hat{f}(x_0) \approx y_i$，**低偏差、高方差**（插值样本点，严重过拟合）；
  - $\lambda \to \infty$（极宽窗口）：全样本均匀加权，$\hat{f}(x_0) \to \bar{y}$，**高偏差、低方差**（欠拟合，退化为全局常数均值）；
  - **度量带宽（Metric Bandwidth） vs. k 近邻自适应带宽（Adaptive Bandwidth）**：
    - 固定度量带宽 $\lambda$（如 $\lambda = 0.2$）：邻域物理宽度恒定，保持局部偏差基本恒定，但在样本稀疏区域（数据点极少）估计方差会剧烈上升；
    - $k$ 近邻自适应宽度 $h_k(x_0) = |x_0 - x_{[k]}|$：保证估计方差处处恒定，但在稀疏区域邻域被迫变宽，导致偏差增大。

### 3. 局部常数的致命弱点：边界偏差（Boundary Bias）与数学机理
为什么 Nadaraya–Watson 核估计在实际应用中存在严重缺陷（边界偏差）？
- **直观缺陷**：
  在数据内部，查询点 $x_0$ 的左右两侧通常有对称分布的数据点，高估和低估相互抵消。
  然而在数据边界处（例如在定义域 $[0, 1]$ 的左端点 $x_0 = 0$），邻域内的样本全部落在 $x_0$ 的右侧（$x_i > x_0$）。若真实函数在边界处有明显斜率（$f'(x_0) > 0$），右侧样本点的函数值系统性地高于 $f(x_0)$，因此局部加权平均必然**系统性向上产生严重偏差**！
- **泰勒展开严格量化偏差阶数**：
  将真实函数 $f(x_i)$ 在 $x_0$ 处展开：
  $$
  f(x_i) = f(x_0) + f'(x_0)(x_i - x_0) + \frac{f''(x_0)}{2}(x_i - x_0)^2 + O((x_i - x_0)^3)
  $$
  代入估计量的条件期望 $\mathbb{E}[\hat{f}(x_0) \mid X] = \sum_{i=1}^N l_i(x_0) f(x_i)$，由于 $\sum l_i(x_0) = 1$：
  $$
  \operatorname{Bias}(\hat{f}(x_0)) = \mathbb{E}[\hat{f}(x_0)] - f(x_0) = f'(x_0) \underbrace{\sum_{i=1}^N l_i(x_0)(x_i - x_0)}_{\text{一阶矩（First Moment）}} + \frac{f''(x_0)}{2} \sum_{i=1}^N l_i(x_0)(x_i - x_0)^2 + O(h^3)
  $$
  - **内部对称区域**：由于 $x_i - x_0$ 左右对称抵消，一阶矩 $\sum l_i(x_0)(x_i - x_0) = 0$，一阶偏差自发消除，剩余偏差为主阶 **$O(h^2) f''(x_0)$**；
  - **边界不对称区域**：单侧样本导致一阶矩 $\sum l_i(x_0)(x_i - x_0) = O(h) \ne 0$，偏差急剧恶化为 **$O(h) f'(x_0)$**！收敛速度比内部慢整整一个数量级。

### 4. 局部线性回归与“自动核修缮”（Local Linear Regression & Automatic Kernel Carpentry，ESL 6.1.1）
为消除 $O(h)$ 边界偏差，局部线性回归（Local Linear Regression）不再局限于局部常数，而是在每个点 $x_0$ 拟合一条局部切线。

- **加权最小二乘目标（WLS）**：
  在查询点 $x_0$ 处求解：
  $$
  \min_{\alpha(x_0), \beta(x_0)} \sum_{i=1}^N K_\lambda(x_0, x_i) \left[ y_i - \alpha(x_0) - \beta(x_0)(x_i - x_0) \right]^2
  $$
  注意：由于自变量采用了中心化 $(x_i - x_0)$，在 $x = x_0$ 处的拟合值恰好就是截距：$\hat{f}(x_0) = \hat{\alpha}(x_0)$。

- **矩阵封闭解与等价核（Equivalent Kernel）**：
  定义基向量 $b(x) = (1, x - x_0)^\top$，设计矩阵 $\mathbf{B}_{N \times 2}$ 的第 $i$ 行为 $(1, x_i - x_0)$。令对角权重阵 $\mathbf{W}(x_0) = \operatorname{diag}(K_\lambda(x_0, x_1), \dots, K_\lambda(x_0, x_N))$。
  根据加权最小二乘正规方程：
  $$
  \begin{pmatrix} \hat{\alpha}(x_0) \\ \hat{\beta}(x_0) \end{pmatrix} = \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{y}
  $$
  因此，拟合值依然是 $y$ 的线性组合：
  $$
  \hat{f}(x_0) = e_1^\top \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{y} = \sum_{i=1}^N l_i(x_0) y_i
  $$
  其中行向量 $l(x_0)^\top = e_1^\top \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0)$ 被称为**等价核（Equivalent Kernel）**。

- **为什么被称为“自动核修缮”（Automatic Kernel Carpentry）？**
  由矩阵正规方程基本性质 $\left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right) \cdot \left[ \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} e_1 \right] = e_1$，即：
  $$
  \mathbf{B}^\top \mathbf{W}(x_0) l(x_0) = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
  $$
  将 $\mathbf{B}$ 代入展开两行：
  1. 第 1 行（零阶矩）：$\sum_{i=1}^N l_i(x_0) = 1$（保持无偏水平）
  2. 第 2 行（一阶矩）：$\sum_{i=1}^N l_i(x_0)(x_i - x_0) = 0$（**一阶矩在任何位置、包括边界，严格恒等于 0！**）
  
  代回泰勒展开偏差公式，一阶项 $f'(x_0) \sum l_i(x_0)(x_i - x_0) \equiv 0$ 被**精确消除**！
  在边界处，等价核 $l_i(x_0)$ 会自动自适应变形（靠近边界侧权重升高，甚至在远端产生微小负权进行外推修正），**使边界偏差自动从 $O(h)$ 降至与内部同阶的 $O(h^2)$**。这一完美性质完全由 WLS 机制自动实现，不需要研究者手动做复杂的边界截断修剪。

- **局部多项式阶数 $d$ 的权衡法则（ESL 6.1.2）**：
  - **局部二次回归（$d=2$）**：若在内部区域真实函数曲率很大（$f''(x)$ 剧烈弯曲），局部线性会出现“削平峰顶、填平谷底（trimming hills and filling valleys）”的曲率偏差。局部二次拟合能消除二阶曲率偏差（偏差降为 $O(h^4)$），但在边界处方差增大显著。
  - **奇数阶占优准则（Odd vs. Even Degree）**：
    渐近理论证明，**奇数阶多项式在均方误差（MSE）上严格占优于相邻的偶数阶**。例如：从 $d=0$（常数）升级到 $d=1$（线性），边界偏差大幅消除且方差几乎不增加；但从 $d=1$ 到 $d=2$（二次），边界偏差阶数并未提升，方差却急剧增大。
    $\implies$ **工程准则：绝大多数场景首选局部线性拟合（$d=1$）**。

### 5. 核带宽选择与有效自由度（ESL 6.2 / Ch.7）
- **线性平滑算子（Linear Smoother）与平滑矩阵**：
  所有 $N$ 个训练样本点的预测值可写为矩阵形式：$\hat{\mathbf{y}} = \mathbf{S}_\lambda \mathbf{y}$，其中平滑矩阵第 $i$ 行为 $l(x_i)^\top$。
- **有效自由度（Effective Degrees of Freedom）**：
  类比线性回归帽子矩阵的自由度 $p+1 = \operatorname{tr}(H)$，核平滑的有效模型复杂度定义为：
  $$
  \operatorname{df}_\lambda = \operatorname{tr}(\mathbf{S}_\lambda)
  $$
  - 当 $\lambda \to 0$ 时，$\mathbf{S}_\lambda \to \mathbf{I}_N \implies \operatorname{df}_\lambda = N$（每个样本自成参数，完全过拟合）；
  - 当 $\lambda \to \infty$ 时，局部线性回归退化为全局 OLS 回归 $\implies \operatorname{df}_\lambda = 2$（截距 + 斜率）。
- **留一交叉验证（LOOCV）解析捷径**：
  对于线性平滑算子，无需真正循环训练 $N$ 次模型，利用平滑矩阵主对角线元素 $S_{\lambda, ii}$ 即可一步得出严格的留一误差：
  $$
  \operatorname{CV}(\lambda) = \frac{1}{N} \sum_{i=1}^N \left( \frac{y_i - \hat{f}_\lambda(x_i)}{1 - S_{\lambda, ii}} \right)^2
  $$
  若计算全部对角线过慢，可采用广义交叉验证（GCV）：
  $$
  \operatorname{GCV}(\lambda) = \frac{1}{N} \sum_{i=1}^N \left( \frac{y_i - \hat{f}_\lambda(x_i)}{1 - \operatorname{tr}(\mathbf{S}_\lambda)/N} \right)^2
  $$

### 6. 高维推广、维数灾难与结构化破局（ESL 6.3–6.4）
- **多元局部回归在 $\mathbb{R}^p$**：
  基向量拓展为 $b(x) = (1, (x - x_0)^\top)^\top \in \mathbb{R}^{p+1}$，采用径向核 $K_\lambda(x_0, x) = D\left(\frac{\|x - x_0\|_2}{\lambda}\right)$。在 2 到 3 维（如对冲期权隐含波动率曲面的“行权价 $\times$ 到期期限”）表现出色。
- **高维空间的维数灾难（Curse of Dimensionality）**：
  当维度 $p \ge 4$ 时，局部平滑全面失效，根源在于两大几何事实：
  1. **体积空旷性**：在 $p$ 维单位超球中，若要捕获比例为 $r$ 的局部样本点，邻域半径必须达到 $e_p(r) = r^{1/p}$。
     - $p=1$ 时，若抓取 $1\%$ 的样本，$e_1(0.01) = 0.01$（真正意义上的局部）；
     - $p=10$ 时，同样要抓取 $1\%$ 的样本，$e_{10}(0.01) = (0.01)^{0.1} \approx 0.63$（邻域半径已跨越超立方体整个特征范围的 $63\%$，“局部”荡然无存！）；
     - 此时非参数回归的均方误差收敛速度恶化为 $O(N^{-4/(4+p)})$，需要天文数字级的样本量。
  2. **边界泛滥**：高维超球体中几乎所有体积都聚集在表面薄壳上（距离边界厚度为 $\epsilon$ 的外壳体积占比为 $1 - (1-\epsilon)^p \to 1$）。在高维中几乎每一个点都是“边界点”，导致边界偏差无处不在。
- **高维非参数建模方案：结构化模型（ESL 6.4）**：
  面对维数灾难，统计建模引入**结构化先验**：
  1. **结构化马氏度量核（Structured Kernels）**：
     引入半正定权重阵 $\mathbf{A} \succeq 0$：$K_{\lambda, \mathbf{A}}(x_0, x) = D\left(\frac{(x - x_0)^\top \mathbf{A} (x - x_0)}{\lambda}\right)$。通过特征协方差或稀疏先验剔除噪声维度、压缩有效搜索子空间。
  2. **广义可加模型（Generalized Additive Models, GAM / ESL Ch.9）**：
     假设函数由各个因子的单变量非参数曲线相加而成：
     $$f(X) = \alpha + \sum_{j=1}^p g_j(X_j)$$
     使用 **Backfitting（交替迭代平滑算法）**，在每一步固定其他分量，对偏残差 $y - \alpha - \sum_{k \ne j} g_k(x_k)$ 关于 $X_j$ 单独做一维局部线性回归。这样既保留了非线性灵活性，又将估计收敛速度牢牢锁死在单变量的 $O(N^{-2/5})$，彻底化解维数灾难。
  3. **变系数模型（Varying-Coefficient Models，量化金融核心武器）**：
     $$f(X, Z) = \sum_{j=1}^q \beta_j(Z) X_j$$
     将解释变量分为两组：核心多因子特征 $X$（维度可较高）与宏观状态/体制变量 $Z$（极低维，如宏观波动率 VIX、资金利率或换手率）。对于给定的状态 $Z = z_0$，模型关于因子 $X$ 是线性的；但因子载荷 $\beta(z_0)$ 随状态 $z_0$ 进行局部核加权拟合。这正是量化投资中**状态依赖因子回归（Regime-Switching Factor Pricing）**的理论基石！

---

## 模块五：核心经典问题与定理推导（绿皮书 + HOTS + ESL）

本模块系统收录周新丰《绿皮书》（A Practical Guide to Quantitative Finance Interviews）、Crack《Heard on the Street》（HOTS）以及 ESL 中的核心线性回归、协方差分析与谱分解计算与证明问题，逐题给出严密代数推导与几何/物理直觉剖析。

---

### 1. 三变量相关系数极值推导（Gram 矩阵半正定与欧氏几何角）

> **问题定义（Green Book 3.6 / 三变量相关系数边界）**：
> 设随机变量 $X, Y, Z$ 均值为 0、方差为 1。已知 $X$ 与 $Y$ 的相关系数为 $\rho_{xy} = 0.8$，$X$ 与 $Z$ 的相关系数为 $\rho_{xz} = 0.8$。
> 1. 求 $Y$ 与 $Z$ 的相关系数 $\rho_{yz}$ 的最大可能值 $\rho_{\max}$ 与最小可能值 $\rho_{\min}$；
> 2. 推广到一般情形：若 $\rho_{xy} = a, \rho_{xz} = b$，求 $\rho_{yz}$ 的取值区间。

**思路拆解与核心直觉**：
相关系数在代数上受制于**协方差矩阵的半正定性（Positive Semi-Definite, PSD）**；在几何上，零均值单位方差随机变量在 Hilbert 空间中对应单位向量，相关系数就是向量夹角的余弦值 $\rho = \cos\theta$。两种视角均可严格求得取值区间。

**严密推导与分步求解**：

**方法一：相关系数矩阵半正定（Gram 矩阵法）**
由于 $X, Y, Z$ 的相关系数矩阵 $\mathbf{R}$ 必须半正定（$\mathbf{R} \succeq 0$），其行列式必须非负：
$$
\mathbf{R} = \begin{pmatrix} 1 & 0.8 & 0.8 \\ 0.8 & 1 & \rho \\ 0.8 & \rho & 1 \end{pmatrix}
$$
按第一行展开行列式：
$$
\begin{aligned}
\det(\mathbf{R}) &= 1 \cdot (1 - \rho^2) - 0.8 \cdot (0.8 - 0.8\rho) + 0.8 \cdot (0.8\rho - 0.8) \\
&= 1 - \rho^2 - 0.64 + 0.64\rho + 0.64\rho - 0.64 \\
&= -\rho^2 + 1.28\rho - 0.28 \ge 0
\end{aligned}
$$
将不等式两边同乘 $-1$：
$$
\rho^2 - 1.28\rho + 0.28 \le 0
$$
求解二次方程 $\rho^2 - 1.28\rho + 0.28 = 0$ 的两根：
$$
\rho = \frac{1.28 \pm \sqrt{1.28^2 - 4 \times 1 \times 0.28}}{2} = \frac{1.28 \pm \sqrt{1.6384 - 1.12}}{2} = \frac{1.28 \pm \sqrt{0.5184}}{2} = \frac{1.28 \pm 0.72}{2}
$$
得到：
- $\rho_{\max} = \frac{1.28 + 0.72}{2} = \boxed{1.0}$
- $\rho_{\min} = \frac{1.28 - 0.72}{2} = \boxed{0.28}$

**方法二：欧氏空间向量夹角法（几何三角不等式）**
将随机变量视作内积空间（$L^2$ 空间）中的单位向量，内积为相关系数：$\langle U, V \rangle = \operatorname{Corr}(U, V) = \cos\theta$。
- 由 $\rho_{xy} = 0.8$，向量 $X$ 与 $Y$ 的夹角为 $\theta_{xy} = \theta_0 = \arccos(0.8)$；
- 由 $\rho_{xz} = 0.8$，向量 $X$ 与 $Z$ 的夹角同样为 $\theta_{xz} = \theta_0 = \arccos(0.8)$。
根据三维空间中两向量夹角的三角不等式，向量 $Y$ 与 $Z$ 的夹角 $\theta_{yz}$ 必须满足：
$$
|\theta_{xy} - \theta_{xz}| \le \theta_{yz} \le \theta_{xy} + \theta_{xz} \implies 0 \le \theta_{yz} \le 2\theta_0
$$
因为余弦函数在 $[0, \pi]$ 上单调递减：
1. **最大相关性（夹角最小）**：当 $\theta_{yz} = 0$ 时，向量 $Y$ 与 $Z$ 完全同向共线：
   $$ \rho_{\max} = \cos(0) = \boxed{1.0} $$
2. **最小相关性（夹角最大）**：当 $\theta_{yz} = 2\theta_0$ 时，$Y$ 与 $Z$ 分列在 $X$ 的两侧且共面：
   $$ \rho_{\min} = \cos(2\theta_0) = 2\cos^2\theta_0 - 1 = 2(0.8)^2 - 1 = 2(0.64) - 1 = \boxed{0.28} $$

**参数化推广**：
若 $\rho_{xy} = a, \rho_{xz} = b$，令 $\theta_a = \arccos a, \theta_b = \arccos b$，则 $\theta_{yz} \in [|\theta_a - \theta_b|, \theta_a + \theta_b]$。利用和差化积公式：
$$
\rho_{yz} \in \left[ ab - \sqrt{(1 - a^2)(1 - b^2)},\; ab + \sqrt{(1 - a^2)(1 - b^2)} \right]
$$

---

### 2. 两两等相关矩阵的半正定下界（Equicorrelated Matrix Bound）

> **问题定义（Green Book 3.6 / 等相关矩阵半正定条件）**：
> 假设有 $n$ 个资产 $X_1, X_2, \dots, X_n$，具有相同的方差 $\sigma^2 > 0$。任意两个不同资产之间的相关系数全部相等，均为 $\rho$（即 $\operatorname{Corr}(X_i, X_j) = \rho, \forall i \ne j$）。
> 1. 为了使该相关系数矩阵合法（即半正定），$\rho$ 的理论取值范围是多少？
> 2. 当资产数量 $n \to \infty$ 时，该下界趋近于何值？这对投资组合分散化（Portfolio Diversification）有何启示？

**思路拆解与严格推导**：

**方法一：特征值分析法**
该相关系数矩阵 $\mathbf{R}_{n \times n}$ 具有如下结构：
$$
\mathbf{R} = \begin{pmatrix} 1 & \rho & \cdots & \rho \\ \rho & 1 & \cdots & \rho \\ \vdots & \vdots & \ddots & \vdots \\ \rho & \rho & \cdots & 1 \end{pmatrix} = (1 - \rho)\mathbf{I}_n + \rho \mathbf{1}\mathbf{1}^\top
$$
其中 $\mathbf{1} = (1, 1, \dots, 1)^\top \in \mathbb{R}^n$。
考察特征向量与特征值：
1. 取向量 $\mathbf{1}$：
   $$ \mathbf{R}\mathbf{1} = (1 - \rho)\mathbf{1} + \rho \mathbf{1}(\mathbf{1}^\top \mathbf{1}) = (1 - \rho)\mathbf{1} + n\rho \mathbf{1} = [1 + (n - 1)\rho]\mathbf{1} $$
   因此，$\lambda_1 = 1 + (n - 1)\rho$，其代数重数为 1。
2. 取任意与 $\mathbf{1}$ 正交的向量 $v \perp \mathbf{1}$（满足 $\mathbf{1}^\top v = 0$，此类线性无关向量共有 $n - 1$ 个）：
   $$ \mathbf{R}v = (1 - \rho)v + \rho \mathbf{1}(\mathbf{1}^\top v) = (1 - \rho)v $$
   因此，$\lambda_2 = \lambda_3 = \dots = \lambda_n = 1 - \rho$，其代数重数为 $n - 1$。

矩阵半正定（$\mathbf{R} \succeq 0$）等价于所有特征值非负：
$$
\begin{cases}
1 - \rho \ge 0 \implies \rho \le 1 \\
1 + (n - 1)\rho \ge 0 \implies \rho \ge -\frac{1}{n - 1}
\end{cases}
$$
因此，合法的取值范围为：
$$
\boxed{-\frac{1}{n - 1} \le \rho \le 1}
$$

**方法二：等权重组合方差非负法（代数分解法）**
构造一个等权重资产组合的总和 $S = \sum_{i=1}^n X_i$。该组合的总方差必须非负：
$$
\begin{aligned}
\operatorname{Var}(S) &= \sum_{i=1}^n \operatorname{Var}(X_i) + \sum_{i \ne j} \operatorname{Cov}(X_i, X_j) \\
&= n\sigma^2 + n(n - 1)\rho\sigma^2 = n\sigma^2 [1 + (n - 1)\rho] \ge 0
\end{aligned}
$$
因为 $n\sigma^2 > 0$，直接得出 $1 + (n - 1)\rho \ge 0 \implies \rho \ge -\frac{1}{n - 1}$。

**金融学意义与极限**：
- 当 $n = 2$ 时，$\rho \ge -1$，两个资产可以完全负相关（对冲风险归零）；
- 当 $n = 3$ 时，$\rho \ge -1/2 = -0.5$；
- 当 $n \to \infty$ 时，$\lim_{n \to \infty} \left(-\frac{1}{n - 1}\right) = 0$。
这意味着：**在由大量资产组成的大市场中，所有资产两两之间不可能普遍为负相关**。如果相关性均为负，组合总方差将不可避免地变成负数，违背概率公理。

---

### 3. 相关矩阵合法性与 Cholesky 分解模拟

> **问题定义（Green Book 3.6 / 协方差奇异性与模拟）**：
> 现有三个资产的成对相关系数：$\rho_{12} = 0.6, \rho_{23} = 0.8, \rho_{13} = 0$。
> 1. 这个相关矩阵是否合法（Valid）？
> 2. 若合法，如何在量化蒙特卡洛引擎中生成服从该相关结构的资产回报路径？

**思路拆解与严格推导**：

**步骤 1：检验半正定性**
构建相关矩阵 $\mathbf{R}$：
$$
\mathbf{R} = \begin{pmatrix} 1 & 0.6 & 0 \\ 0.6 & 1 & 0.8 \\ 0 & 0.8 & 1 \end{pmatrix}
$$
计算所有顺序主子式（Sylvester 准则检验半正定）：
- 1 阶主子式：$1 > 0$
- 2 阶主子式：$1 - 0.6^2 = 0.64 > 0$
- 3 阶主子式（行列式）：
  $$
  \det(\mathbf{R}) = 1 \cdot (1 - 0.8^2) - 0.6 \cdot (0.6 - 0) + 0 = (1 - 0.64) - 0.36 = 0.36 - 0.36 = 0
  $$
因为所有主子式均 $\ge 0$ 且 $\det(\mathbf{R}) = 0$，**该矩阵是合法的半正定矩阵**（处于共面的退化边界，最小特征值为 0）。

**步骤 2：Cholesky 分解与随机数模拟**
欲生成均值为 0、协方差为 $\mathbf{R}$ 的随机向量 $X = (X_1, X_2, X_3)^\top$。对 $\mathbf{R}$ 作下三角 Cholesky 分解 $\mathbf{R} = \mathbf{L}\mathbf{L}^\top$：
设 $\mathbf{L} = \begin{pmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{pmatrix}$：
1. $l_{11} = \sqrt{1} = 1$
2. $l_{21} = 0.6 / 1 = 0.6$；$l_{22} = \sqrt{1 - 0.6^2} = 0.8$
3. $l_{31} = 0 / 1 = 0$；$l_{32} = (0.8 - 0 \times 0.6) / 0.8 = 1.0$；$l_{33} = \sqrt{1 - 0^2 - 1.0^2} = 0$

得到下三角矩阵：
$$
\mathbf{L} = \begin{pmatrix} 1 & 0 & 0 \\ 0.6 & 0.8 & 0 \\ 0 & 1 & 0 \end{pmatrix}
$$
**模拟执行算法**：
先抽取 3 个独立的标准正态伪随机数 $Z = (Z_1, Z_2, Z_3)^\top \sim \mathcal{N}(0, \mathbf{I})$，令：
$$
\begin{pmatrix} X_1 \\ X_2 \\ X_3 \end{pmatrix} = \mathbf{L} \begin{pmatrix} Z_1 \\ Z_2 \\ Z_3 \end{pmatrix} = \begin{pmatrix} Z_1 \\ 0.6 Z_1 + 0.8 Z_2 \\ Z_2 \end{pmatrix}
$$
验证协方差：
- $\operatorname{Corr}(X_1, X_2) = \mathbb{E}[Z_1(0.6Z_1 + 0.8Z_2)] = 0.6$
- $\operatorname{Corr}(X_2, X_3) = \mathbb{E}[(0.6Z_1 + 0.8Z_2)Z_2] = 0.8$
- $\operatorname{Corr}(X_1, X_3) = \mathbb{E}[Z_1 Z_2] = 0$
完全满足要求！注意因为 $\det(\mathbf{R}) = 0$，$X_3$ 严格等于用于构造 $X_2$ 的第二个正交基 $Z_2$。

---

### 4. CAPM Beta、方差分解与逆向回归

> **问题定义（Heard on the Street / 条件期望与反向回归）**：
> 某股票 A 的日收益率波动率为 $\sigma_A = 2\%$，市场基准 M 的波动率为 $\sigma_M = 1\%$，两者相关系数为 $\rho = 0.5$。
> 1. 计算股票 A 对市场基准 M 回归的 $\beta$、模型的解释度 $R^2$ 以及残差波动率 $\sigma_\varepsilon$；
> 2. 若今天股票 A 暴涨了 $+4\%$，预测今天市场基准 M 的收益率；
> 3. 若收益率满足独立同分布（IID）假设，预测股票 A 明天的收益率。

**思路拆解与严格推导**：

**第 1 问：前向回归各项指标计算**
根据单变量 OLS 核心公式：
- **市场 Beta**：
  $$ \beta_{A \sim M} = \rho \frac{\sigma_A}{\sigma_M} = 0.5 \times \frac{2\%}{1\%} = \boxed{1.0} $$
- **决定系数 $R^2$**：
  $$ R^2 = \rho^2 = 0.5^2 = \boxed{0.25 = 25\%} $$
- **残差方差与残差波动率**：
  由方差正交分解 $\sigma_A^2 = \beta^2 \sigma_M^2 + \sigma_\varepsilon^2 = R^2 \sigma_A^2 + (1 - R^2)\sigma_A^2$：
  $$ \sigma_\varepsilon = \sigma_A \sqrt{1 - \rho^2} = 2\% \times \sqrt{1 - 0.25} = 2\% \times \frac{\sqrt{3}}{2} = \boxed{\sqrt{3}\% \approx 1.732\%} $$

**第 2 问：逆向回归（Reverse Regression）**
> **常见直觉误区**：直接移项变形得到“既然 $\beta = 1.0$，当股票涨 $4\%$ 时市场也涨 $4\%$”。
> **误区根源**：直接将前向回归方程代数移项，忽视了投影方向改变后的条件期望非对称性。

**正确推导**：
当条件变量变成 $R_A = 4\%$ 时，我们要解决的是在给定 $R_A$ 下对 $R_M$ 的条件期望预测 $\mathbb{E}[R_M \mid R_A = 4\%]$。
此时因变量是 $M$，自变量是 $A$，必须建立**逆向回归（Reverse Regression）**模型：
$$
\beta_{M \sim A} = \rho \frac{\sigma_M}{\sigma_A} = 0.5 \times \frac{1\%}{2\%} = \boxed{0.25}
$$
因此，最佳无偏线性预测值为：
$$
\mathbb{E}[R_M \mid R_A = 4\%] = \beta_{M \sim A} \times 4\% = 0.25 \times 4\% = \boxed{+1\%}
$$
**标准化变量视角（均值回归的本质）**：
将股票收益率标准化为 $Z$-score：$z_A = \frac{+4\%}{\sigma_A} = \frac{4\%}{2\%} = +2$（股票上涨了 $2$ 个标准差）。
根据二元正态分布条件期望：$\hat{z}_M = \rho \cdot z_A = 0.5 \times 2 = +1$（市场仅上涨 $1$ 个标准差）。
市场收益率预测值即为 $1 \times \sigma_M = 1 \times 1\% = +1\%$。
由于 $|\rho| = 0.5 < 1$，极端表现的自变量所预测的因变量一定会向均值收缩（Regression to the Mean），乘积恒满足 $\beta_{\text{forward}} \times \beta_{\text{reverse}} = \rho^2 \le 1$！

**第 3 问：IID 假定下的跨期预测**
> **概念辨析**：在收益率 IID 假定下，明天股票的预期收益率为何？
> **分析**：明天预期收益率为无条件均值（**近似为 0%**）。
因为题设明确假定收益率是 **IID（独立同分布）**。过去的价格和今天的 $+4\%$ 对未来的表现不提供任何信息（$\operatorname{Cov}(R_{t+1}, R_t) = 0$）。
横截面上的高斯均值回归（Regression to the Mean，由确定性信号被噪声稀释所致）与时间序列上的均值回归（Mean Reversion，由负自相关性所致）属于两个完全不同的统计物理概念，不可混淆。

---

### 5. 仿射变换对协方差与相关系数的影响

> **问题定义（Heard on the Street 4.5）**：
> 已知随机变量 $X$ 与 $Y$ 的相关系数为 $\operatorname{Corr}(X, Y) = \rho$。
> 1. 求 $\operatorname{Corr}(X + 5, Y)$；
> 2. 求 $\operatorname{Corr}(5X, Y)$；
> 3. 求 $\operatorname{Corr}(-5X + 3, 2Y - 7)$。

**思路拆解与严格推导**：
根据协方差和方差在仿射变换下的基本代数性质：
- 协方差的双线性性：$\operatorname{Cov}(aX + b, cY + d) = ac \operatorname{Cov}(X, Y)$（常数平移量 $b, d$ 不影响波动）
- 方差的齐次性：$\operatorname{Var}(aX + b) = a^2 \operatorname{Var}(X) \implies \sigma_{aX+b} = |a| \sigma_X$

代入相关系数定义：
$$
\operatorname{Corr}(aX + b, cY + d) = \frac{\operatorname{Cov}(aX + b, cY + d)}{\sigma_{aX+b} \sigma_{cY+d}} = \frac{ac \operatorname{Cov}(X, Y)}{|a|\sigma_X |c|\sigma_Y} = \frac{ac}{|a||c|} \operatorname{Corr}(X, Y) = \operatorname{sgn}(ac) \rho
$$
**结论直接代入**：
1. $\operatorname{Corr}(X + 5, Y)$：$a = 1, c = 1 \implies \operatorname{sgn}(1) \rho = \boxed{\rho}$（**平移严格不变**）
2. $\operatorname{Corr}(5X, Y)$：$a = 5, c = 1 \implies \operatorname{sgn}(5) \rho = \boxed{\rho}$（**正数缩放严格不变**）
3. $\operatorname{Corr}(-5X + 3, 2Y - 7)$：$a = -5, c = 2 \implies ac = -10 < 0 \implies \operatorname{sgn}(-10)\rho = \boxed{-\rho}$（**异号缩放产生负号**）

---

### 6. 遗漏变量偏差（Omitted Variable Bias, OVB）代数推导

> **问题定义（多因子模型中的遗漏变量偏差）**：
> 假设资产真实的数据生成过程（DGP）包含两个因子：
> $$ y = \beta_1 x_1 + \beta_2 x_2 + \varepsilon, \qquad \mathbb{E}[\varepsilon \mid x_1, x_2] = 0 $$
> 但研究者在回归时遗漏了变量 $x_2$，仅对 $x_1$ 拟合了单变量回归：$y = \alpha x_1 + u$。
> 1. 严格推导 OLS 估计量 $\hat\alpha$ 的大样本概率极限 $\operatorname{plim}\hat\alpha$，并给出遗漏变量偏差表达式；
> 2. **量化案例分析**：若 $x_1$ 为某股票的高频动量因子，遗漏的 $x_2$ 为全行业景气度因子（已知 $\beta_2 > 0$），且动量越好的股票往往属于高景气行业（$\operatorname{Cov}(x_1, x_2) > 0$），请问单变量动量因子的回归斜率是被高估还是低估？

**思路拆解与严格推导**：

单变量 OLS 估计量为：
$$
\hat\alpha = \frac{\sum_{i=1}^N x_{1i} y_i}{\sum_{i=1}^N x_{1i}^2}
$$
将真实的 $y_i = \beta_1 x_{1i} + \beta_2 x_{2i} + \varepsilon_i$ 代入分子：
$$
\begin{aligned}
\hat\alpha &= \frac{\sum_{i=1}^N x_{1i}(\beta_1 x_{1i} + \beta_2 x_{2i} + \varepsilon_i)}{\sum_{i=1}^N x_{1i}^2} \\
&= \beta_1 \frac{\sum x_{1i}^2}{\sum x_{1i}^2} + \beta_2 \frac{\sum x_{1i} x_{2i}}{\sum x_{1i}^2} + \frac{\sum x_{1i} \varepsilon_i}{\sum x_{1i}^2} \\
&= \beta_1 + \beta_2 \frac{\sum x_{1i} x_{2i}}{\sum x_{1i}^2} + \frac{\frac{1}{N}\sum x_{1i}\varepsilon_i}{\frac{1}{N}\sum x_{1i}^2}
\end{aligned}
$$
当 $N \to \infty$ 时，由大数定律及外生性假定 $\mathbb{E}[x_1 \varepsilon] = 0$，最后一项依概率收敛于 0：
$$
\operatorname{plim}\hat\alpha = \beta_1 + \beta_2 \frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}
$$
**遗漏变量偏差公式**为：
$$
\operatorname{Bias} = \operatorname{plim}\hat\alpha - \beta_1 = \boxed{\beta_2 \frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}}
$$
**量化实战定性结论**：
- $\beta_2 > 0$（行业景气度带来正收益）；
- $\operatorname{Cov}(x_1, x_2) > 0$（高动量股集中在高景气行业）；
- 因此 $\operatorname{Bias} > 0$，单变量动量因子的斜率被**严重高估（向上偏差）**。
在多因子量化中，若未做行业中性化（Industry Neutralization），研究员误以为自己找到了强大的个股动量 Alpha，实则只是被动承担了未对冲的行业 Beta 风险。

---

### 7. 自变量测量误差（Measurement Error）与衰减偏差

> **问题定义（自变量含测量噪声时的衰减偏差）**：
> 假设真实收益率模型为 $y = \beta x^* + \varepsilon$（其中 $\beta \ne 0$），$\mathbb{E}[\varepsilon \mid x^*] = 0$。但由于微观结构噪音（如买卖价差跳价、延迟行情或估计误差），真实的因子 $x^*$ 无法被直接观测，研究者只能观测到带有噪音的指标 $x = x^* + u$，其中测量误差 $u \sim \mathcal{N}(0, \sigma_u^2)$，且 $u$ 与真实值 $x^*$ 及扰动项 $\varepsilon$ 完全独立。
> 1. 推导使用观测指标 $x$ 进行 OLS 回归时的斜率概率极限 $\operatorname{plim}\hat\beta$；
> 2. 解释为何这会导致“衰减偏差（Attenuation Bias / Regression Dilution）”？

**思路拆解与严格推导**：

单变量 OLS 斜率估计量为：
$$
\hat\beta = \frac{\widehat{\operatorname{Cov}}(x, y)}{\widehat{\operatorname{Var}}(x)}
$$
大样本下分别推导分子与分母的概率极限：
1. **分子（样本协方差极限）**：
   $$
   \begin{aligned}
   \operatorname{Cov}(x, y) &= \operatorname{Cov}(x^* + u, \beta x^* + \varepsilon) \\
   &= \operatorname{Cov}(x^*, \beta x^*) + \operatorname{Cov}(x^*, \varepsilon) + \operatorname{Cov}(u, \beta x^*) + \operatorname{Cov}(u, \varepsilon) \\
   &= \beta \operatorname{Var}(x^*) + 0 + 0 + 0 = \beta \sigma_{x^*}^2
   \end{aligned}
   $$
2. **分母（样本方差极限）**：
   $$
   \operatorname{Var}(x) = \operatorname{Var}(x^* + u) = \operatorname{Var}(x^*) + \operatorname{Var}(u) + 2\operatorname{Cov}(x^*, u) = \sigma_{x^*}^2 + \sigma_u^2
   $$
代入比值：
$$
\operatorname{plim}\hat\beta = \frac{\beta \sigma_{x^*}^2}{\sigma_{x^*}^2 + \sigma_u^2} = \beta \cdot \boxed{\frac{1}{1 + \frac{\sigma_u^2}{\sigma_{x^*}^2}}}
$$
定义**信噪比可靠性系数** $\lambda = \frac{\sigma_{x^*}^2}{\sigma_{x^*}^2 + \sigma_u^2} \in (0, 1)$，则：
$$
\operatorname{plim}\hat\beta = \beta \cdot \lambda < \beta \quad (\text{若 } \beta > 0)
$$
**结论与意义**：
自变量带有测量噪音会使 OLS 斜率**严格向 0 衰减（收缩）**。在量化实盘中，订单流不平衡（OFI）或高频信号若包含大量微观结构白噪音，会导致模型严重低估信号对未来价格的边际驱动力。即使样本量 $N \to \infty$，该衰减偏差也无法消除（OLS 估计量不一致）。通常必须引入工具变量（IV）或状态空间卡尔曼滤波进行纠偏。

---

### 8. 多重共线性（Multicollinearity）、VIF 与预测/解释悖论

> **问题定义（高维共线性与方差膨胀因子）**：
> 1. 写出多元线性回归中第 $j$ 个回归系数方差 $\operatorname{Var}(\hat\beta_j)$ 的解析公式，并定义方差膨胀因子（VIF）；
> 2. 为什么多重共线性会严重破坏因子的经济学解释性，但对模型整体的预测精度通常影响微弱？

**思路拆解与严格推导**：

在多元回归 $y = X\beta + \varepsilon$ 中，参数协方差矩阵为 $\operatorname{Var}(\hat\beta) = \sigma^2 (X^\top X)^{-1}$。
对其主对角线元素展开，第 $j$ 个系数的方差可严格写为：
$$
\operatorname{Var}(\hat\beta_j) = \frac{\sigma^2}{\sum_{i=1}^N (x_{ij} - \bar{x}_j)^2 (1 - R_j^2)} = \frac{\sigma^2}{\operatorname{TSS}_j} \cdot \operatorname{VIF}_j
$$
其中：
- $R_j^2$ 为将特征 $x_j$ 对其余所有解释变量做辅助 OLS 回归得到的决定系数；
- $\operatorname{VIF}_j = \frac{1}{1 - R_j^2}$ 被称为**方差膨胀因子（Variance Inflation Factor）**。

**预测 vs. 解释的几何悖论**：
- **解释力崩塌**：当 $x_j$ 与其他特征高度线性相关时，$R_j^2 \to 1 \implies \operatorname{VIF}_j \to \infty$。导致 $\hat\beta_j$ 的抽样方差爆炸，标准误极大，$t$ 统计量骤降，甚至系数正负号发生剧烈翻转，单因子完全失去解释价值。
- **预测力稳健**：在几何上，$X$ 的列向量所张成的子空间 $\mathrm{Col}(X)$ 是高度稳定的超平面。虽然在子空间内部难以区分各个基底方向的独立贡献（矩阵 $(X^\top X)$ 接近奇异），但因变量 $y$ 向整个超平面的正交投影 $\hat{y} = H y$ 是唯一确定的。只要测试集数据的协方差结构与训练集一致，拟合值 $\hat{y}$ 的预测方差依然很小。

---

### 9. 最优期货套期保值比率（Optimal Hedge Ratio）推导

> **问题定义（Green Book 4.5 / 最小方差套期保值）**：
> 某量化对冲基金持有价值现货头寸 $S$，计划使用股指期货 $F$ 进行风险对冲。设在对冲期内，现货价值变动量为 $\Delta S$，期货价值变动量为 $\Delta F$。构建对冲组合 $\Delta \Pi = \Delta S - h \Delta F$，其中 $h$ 为单位现货对应的期货对冲比率。
> 1. 求解使对冲组合价值波动方差最小化的最优对冲比率 $h^*$；
> 2. 证明该最优比率严格等价于单变量 OLS 回归斜率，并给出对冲后的方差缩减比例。

**思路拆解与严格推导**：

**第 1 问：组合方差极小化**
对冲组合的方差为：
$$
\operatorname{Var}(\Delta \Pi) = \operatorname{Var}(\Delta S - h \Delta F) = \operatorname{Var}(\Delta S) + h^2 \operatorname{Var}(\Delta F) - 2h \operatorname{Cov}(\Delta S, \Delta F)
$$
记 $\sigma_S^2 = \operatorname{Var}(\Delta S)$，$\sigma_F^2 = \operatorname{Var}(\Delta F)$，相关系数为 $\rho$。方差函数为关于 $h$ 的开口向上的凸二次函数：
$$
f(h) = \sigma_S^2 + h^2 \sigma_F^2 - 2h \rho \sigma_S \sigma_F
$$
对 $h$ 求一阶导数并令其为 0：
$$
\frac{d f(h)}{dh} = 2h \sigma_F^2 - 2\operatorname{Cov}(\Delta S, \Delta F) = 0
$$
解得最优对冲比率：
$$
h^* = \frac{\operatorname{Cov}(\Delta S, \Delta F)}{\operatorname{Var}(\Delta F)} = \rho \frac{\sigma_S}{\sigma_F}
$$

**第 2 问：OLS 等价性与方差缩减**
- **OLS 等价性**：若建立线性回归模型 $\Delta S = \alpha + h \Delta F + \varepsilon$，最小化残差平方和 $\sum \varepsilon_i^2$ 本质上就是最小化对冲组合的残差方差。其正规方程解恰好就是 $h^* = \frac{\operatorname{Cov}(\Delta S, \Delta F)}{\operatorname{Var}(\Delta F)}$！
- **最小残差方差**：将 $h^*$ 代回方差公式：
  $$
  \operatorname{Var}^*(\Delta \Pi) = \sigma_S^2 + \left( \rho \frac{\sigma_S}{\sigma_F} \right)^2 \sigma_F^2 - 2\left( \rho \frac{\sigma_S}{\sigma_F} \right) \rho \sigma_S \sigma_F = \sigma_S^2 + \rho^2 \sigma_S^2 - 2\rho^2 \sigma_S^2 = \sigma_S^2(1 - \rho^2)
  $$
- **方差缩减比例**：
  $$ \frac{\operatorname{Var}(\Delta S) - \operatorname{Var}^*(\Delta \Pi)}{\operatorname{Var}(\Delta S)} = \frac{\sigma_S^2 - \sigma_S^2(1 - \rho^2)}{\sigma_S^2} = \boxed{\rho^2 = R^2} $$
  通过期货对冲能消灭的现货风险比例，严格等于回归模型的判定系数 $R^2$！

---

### 10. Frisch–Waugh–Lovell (FWL) 定理与两阶段残差回归（求 $\beta_1 / \beta_2$ 比值）

> **问题定义（FWL 定理与两阶段残差回归的系数比值）**：
> 在多元线性回归中，考虑以下三组回归（为简化推导，假设所有变量均已去中心化，中心化不改变方差、协方差与斜率）：
> 1. **$Y$ on $X_1$（一元回归提取残差）**：
>    $$ \varepsilon = Y - \gamma X_1, \quad \text{其中 } \gamma = \frac{\operatorname{Cov}(Y, X_1)}{\operatorname{Var}(X_1)}, \quad \text{且满足残差正交 } \operatorname{Cov}(\varepsilon, X_1) = 0 $$
> 2. **$\varepsilon$ on $X_2$（残差对未正交化特征的一元回归）**：
>    $$ \beta_1 = \frac{\operatorname{Cov}(\varepsilon, X_2)}{\operatorname{Var}(X_2)} $$
> 3. **$Y$ on $(X_1, X_2)$（标准二元联合回归）**：
>    $$ Y = b_1 X_1 + \beta_2 X_2 + u, \quad \text{其中多元残差 } u \text{ 满足 } \operatorname{Cov}(u, X_1) = 0 \text{ 且 } \operatorname{Cov}(u, X_2) = 0 $$
> 已知 $X_1$ 与 $X_2$ 的相关系数为 $\rho = \operatorname{Corr}(X_1, X_2)$。
>
> **核心追问**：
> 1. 试求斜率 $\beta_1$ 与多元联合回归系数 $\beta_2$ 的数学关系与比值 $\frac{\beta_1}{\beta_2}$；
> 2. 很多人凭直觉误以为 $\beta_1 = \beta_2$。请从 Frisch–Waugh–Lovell (FWL) 定理与几何正交投影的本质，深入剖析为什么直接将 $\varepsilon$ 对原变量 $X_2$ 回归会导致估计量发生压缩（Attenuation），真正的 FWL 应当如何操作？
> 3. 请阐述该结论在量化多因子模型中“因子行业/风格中性化（Neutralization）”与增量因子有效性检验中的实战指导意义。

**思路拆解与严格推导**：

#### 1. 核心推导：建立 $\varepsilon$ 与多元回归的关系消元

核心区别在于：题目中给出的是相关系数 $\rho = \operatorname{Corr}(X_1, X_2)$，而一元线性回归斜率使用的是协方差与自变量方差之比 $\frac{\operatorname{Cov}}{\operatorname{Var}}$。

处理时的本质推导流程只需要在涉及 $X_1$ 和 $X_2$ 互投时引入方差归一化：

**第一步：定义各回归表达式**
- $Y$ on $X_1$：
  $$ \varepsilon = Y - \gamma X_1, \quad \gamma = \frac{\operatorname{Cov}(Y, X_1)}{\operatorname{Var}(X_1)}, \quad \operatorname{Cov}(\varepsilon, X_1) = 0 $$
- $\varepsilon$ on $X_2$：
  $$ \beta_1 = \frac{\operatorname{Cov}(\varepsilon, X_2)}{\operatorname{Var}(X_2)} $$
- $Y$ on $(X_1, X_2)$：
  $$ Y = b_1 X_1 + \beta_2 X_2 + u, \quad \operatorname{Cov}(u, X_1) = 0 \text{ 且 } \operatorname{Cov}(u, X_2) = 0 $$

**第二步：建立 $\varepsilon$ 与多元回归的关系**
将多元回归方程 $Y = b_1 X_1 + \beta_2 X_2 + u$ 代入 $\varepsilon = Y - \gamma X_1$ 中：
$$ \varepsilon = (b_1 - \gamma) X_1 + \beta_2 X_2 + u $$
计算残差 $\varepsilon$ 与 $X_2$ 的协方差：
$$
\begin{aligned}
\operatorname{Cov}(\varepsilon, X_2) &= \operatorname{Cov}\left( (b_1 - \gamma) X_1 + \beta_2 X_2 + u, \, X_2 \right) \\
&= (b_1 - \gamma)\operatorname{Cov}(X_1, X_2) + \beta_2 \operatorname{Var}(X_2) + \underbrace{\operatorname{Cov}(u, X_2)}_{= 0} \\
&= (b_1 - \gamma)\operatorname{Cov}(X_1, X_2) + \beta_2 \operatorname{Var}(X_2)
\end{aligned}
$$

**第三步：利用正交条件消去 $(b_1 - \gamma)$**
由一元 OLS 的正规方程性质，残差 $\varepsilon$ 必须正交于回归自变量 $X_1$，即 $\operatorname{Cov}(\varepsilon, X_1) = 0$：
$$
\begin{aligned}
\operatorname{Cov}(\varepsilon, X_1) &= (b_1 - \gamma)\operatorname{Var}(X_1) + \beta_2 \operatorname{Cov}(X_2, X_1) + \underbrace{\operatorname{Cov}(u, X_1)}_{= 0} = 0
\end{aligned}
$$
由此精确解得未知系数差 $(b_1 - \gamma)$：
$$ b_1 - \gamma = -\beta_2 \frac{\operatorname{Cov}(X_1, X_2)}{\operatorname{Var}(X_1)} $$

**第四步：带入求 $\beta_1$ 并代换出相关系数 $\rho$**
将 $(b_1 - \gamma)$ 的表达式代回 $\operatorname{Cov}(\varepsilon, X_2)$：
$$
\begin{aligned}
\operatorname{Cov}(\varepsilon, X_2) &= \left( -\beta_2 \frac{\operatorname{Cov}(X_1, X_2)}{\operatorname{Var}(X_1)} \right) \operatorname{Cov}(X_1, X_2) + \beta_2 \operatorname{Var}(X_2) \\
&= -\beta_2 \frac{\operatorname{Cov}(X_1, X_2)^2}{\operatorname{Var}(X_1)} + \beta_2 \operatorname{Var}(X_2) \\
&= \beta_2 \operatorname{Var}(X_2) \left( 1 - \frac{\operatorname{Cov}(X_1, X_2)^2}{\operatorname{Var}(X_1)\operatorname{Var}(X_2)} \right)
\end{aligned}
$$
注意到括号中的第二项正好是相关系数平方 $\rho^2 = \frac{\operatorname{Cov}(X_1, X_2)^2}{\operatorname{Var}(X_1)\operatorname{Var}(X_2)}$：
$$ \operatorname{Cov}(\varepsilon, X_2) = \beta_2 \operatorname{Var}(X_2)(1 - \rho^2) $$
两边同除以 $\operatorname{Var}(X_2)$，即得 $\beta_1$ 的显式闭式解：
$$ \beta_1 = \frac{\operatorname{Cov}(\varepsilon, X_2)}{\operatorname{Var}(X_2)} = \beta_2 (1 - \rho^2) $$
两者的比值直接写为：
$$ \boxed{\frac{\beta_1}{\beta_2} = 1 - \rho^2} $$

---

#### 2. 几何与 Frisch–Waugh–Lovell (FWL) 定理视角

这本质上是 **Frisch–Waugh–Lovell (FWL) 定理**最经典的高频变体与几何陷阱：

- **真 FWL 定理的核心操作**：
  FWL 定理指出，多元回归系数 $\beta_2$ 对应将 $Y$ 投影到 $X_2$ **剔除 $X_1$ 后的净残差空间**上：
  $$ \beta_2 = \frac{\operatorname{Cov}(\varepsilon, \tilde{X}_2)}{\operatorname{Var}(\tilde{X}_2)} $$
  其中 $\tilde{X}_2 = X_2 - \operatorname{Proj}_{X_1}(X_2) = M_1 X_2$ 是自变量 $X_2$ 剥离掉与 $X_1$ 线性共线性后的纯净特征增量。
- **题目中两阶段回归的致命疏漏**：
  题目里的 $\beta_1$ 仅仅将因变量 $Y$ 做了正交化（得到残差 $\varepsilon$），但**忘记了将自变量 $X_2$ 也做正交化**，直接将 $\varepsilon$ 投在了未净化的原变量 $X_2$ 上：
  $$ \beta_1 = \frac{\operatorname{Cov}(\varepsilon, X_2)}{\operatorname{Var}(X_2)} $$
- **为什么两者差了一个因子 $(1 - \rho^2)$？**
  1. **分子内积恒等**：因为 $\varepsilon \perp X_1$ 且 $X_2 = \operatorname{Proj}_{X_1}(X_2) + \tilde{X}_2$：
     $$ \operatorname{Cov}(\varepsilon, X_2) = \underbrace{\operatorname{Cov}(\varepsilon, \operatorname{Proj}_{X_1}(X_2))}_{= 0} + \operatorname{Cov}(\varepsilon, \tilde{X}_2) = \operatorname{Cov}(\varepsilon, \tilde{X}_2) $$
     分子内积在几何上绝对相同！
  2. **分母方差缩减比例**：
     $\beta_2$ 的分母是净特征方差 $\operatorname{Var}(\tilde{X}_2) = \operatorname{Var}(X_2)(1 - R_{X_2 \sim X_1}^2) = \operatorname{Var}(X_2)(1 - \rho^2)$；
     而 $\beta_1$ 的分母错误地使用了包含大量冗余共线信息的全方差 $\operatorname{Var}(X_2)$。
     因此两者之比恰好等于方差缩减比例：
     $$ \frac{\beta_1}{\beta_2} = \frac{\operatorname{Var}(\tilde{X}_2)}{\operatorname{Var}(X_2)} = 1 - R_{X_2 \sim X_1}^2 = 1 - \rho^2 $$

```fwl-geometry-demo
```

---

#### 3. 量化投资多因子实战启示

1. **行业与风格中性化（Neutralization）必须“两端正交”**：
   在多因子 Alpha 模型中，有两种做法：
   - **正确做法（FWL 标准流）**：不仅将收益率 $Y$ 对行业/风格风险因子 $X_1$ 做截面回归取残差 $\varepsilon$，**还必须将原始候选因子 $X_2$ 也对 $X_1$ 做截面回归取残差 $\tilde{X}_2$**，再求因子收益率斜率；
   - **错误做法**：只对收益率剔除行业影响，却直接用未中性化的原始因子去测 IC 或回归。由于行业敞口 $\rho \ne 0$，测得的纯净因子收益率将被虚假压缩 $(1 - \rho^2)$ 倍，导致优秀增量因子被系统性低估甚至误杀！
2. **增量因子有效性检验（Incremental Alpha Test）**：
   当要验证一个新的 Alpha 因子 $X_{\text{new}}$ 在既有数百个基准因子库 $X_{\text{base}}$ 之外是否具有纯净增量贡献时：
   必须先将 $X_{\text{new}}$ 对整个矩阵 $X_{\text{base}}$ 做投影消去（$\tilde{X}_{\text{new}} = M_{\text{base}} X_{\text{new}}$），再测试净残差的统计显著性。

---

### 11. 无截距回归（Regression Without Intercept）与负 R²

> **问题定义（无截距回归对残差均值与判定系数的代数影响）**：
> 在 CAPM 或套利定价理论测试中，若强行令截距项为零进行回归：$y = X\beta + \varepsilon$。
> 1. 为什么无截距时，残差之和 $\sum_{i=1}^N \hat\varepsilon_i$ 通常不等于零？
> 2. 为什么常规计算的决定系数 $R^2$ 可能会出现负数？

**思路拆解与严格推导**：

**第 1 问：残差和为零的真正来源**
OLS 正规方程为 $X^\top \hat\varepsilon = 0$。
- 当回归模型**包含截距项**时，$X$ 的第一列为全 1 向量 $\mathbf{1} = (1, 1, \dots, 1)^\top$。正规方程的第一行即为：
  $$ \mathbf{1}^\top \hat\varepsilon = \sum_{i=1}^N \hat\varepsilon_i = 0 $$
- 当回归模型**强制无截距**时，列向量全为具体的特征数值，没有任何线性组合保证能构造出常数向量 $\mathbf{1}$。因此 $\mathbf{1}$ 不垂直于残差向量 $\hat\varepsilon$，**残差均值通常不为零（$\sum \hat\varepsilon_i \ne 0$）**！

**第 2 问：平方和分解公式崩溃与负 $R^2$**
总离差平方和定义为 $\operatorname{TSS} = \sum_{i=1}^N (y_i - \bar{y})^2$。展开分解：
$$
\begin{aligned}
\operatorname{TSS} &= \sum_{i=1}^N (y_i - \hat{y}_i + \hat{y}_i - \bar{y})^2 \\
&= \sum_{i=1}^N \hat\varepsilon_i^2 + \sum_{i=1}^N (\hat{y}_i - \bar{y})^2 + 2\sum_{i=1}^N \hat\varepsilon_i (\hat{y}_i - \bar{y}) \\
&= \operatorname{RSS} + \operatorname{ESS} + 2\underbrace{\sum_{i=1}^N \hat\varepsilon_i \hat{y}_i}_{= 0} - 2\bar{y}\underbrace{\sum_{i=1}^N \hat\varepsilon_i}_{\ne 0}
\end{aligned}
$$
注意：因为正规方程保证 $\hat{y}^\top \hat\varepsilon = \hat\beta^\top X^\top \hat\varepsilon = 0$，但无截距时 $\sum \hat\varepsilon_i \ne 0$，因此**交叉项 $-2\bar{y}\sum \hat\varepsilon_i$ 无法消除**！
$$ \operatorname{TSS} \ne \operatorname{ESS} + \operatorname{RSS} $$
若统计软件依然盲目套用公式：
$$ R^2 = 1 - \frac{\operatorname{RSS}}{\operatorname{TSS}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2} $$
当无截距拟合线（强行穿过原点）的表现比水平基准线 $\bar{y}$ 还要糟糕时，残差平方和 $\operatorname{RSS} > \operatorname{TSS}$，从而计算出 **$R^2 < 0$**！

---

### 12. 日频收益率 $R^2 \approx 1\%$ 与信息比率（IR）的数学映射

> **问题定义（横截面 R² 与实盘信息系数 IC 的数学对应）**：
> 某模型回测股票 Alpha 信号时，信号对次日收益率的回归 $R^2$ 仅为 $1\%$（即 $0.01$）。
> 请使用**主动管理基本法则（Fundamental Law of Active Management）**严格分析该预测能力的商业价值与年化信息比率（IR）。

**思路拆解与严格推导**：

单变量回归中，判定系数与相关系数满足：
$$ R^2 = \rho^2 \implies |\rho| = \sqrt{R^2} $$
当日频 $R^2 = 1\% = 0.01$ 时，信号与次日收益率的信息系数（Information Coefficient, IC）为：
$$ \operatorname{IC} = \rho = \sqrt{0.01} = \boxed{0.10} $$
**主动管理基本法则（Grinold & Kahn）**：
$$
\operatorname{IR} \approx \operatorname{IC} \times \sqrt{\text{Breadth}}
$$
其中：
- $\operatorname{IR}$ 为投资组合的信息比率（近似等于年化夏普比率 Sharpe Ratio）；
- $\text{Breadth}$ 为一年内独立投资决策的广度。

**量化实盘参数代入**：
假设该多因子策略在全市场跟踪 $N = 1000$ 只活跃股票，一年约有 $T = 252$ 个交易日：
- 即使因股票之间存在截面相关性，我们将每期的有效独立股票数保守折算为 $N_{\text{eff}} = 100$；
- 则全年的有效决策广度为 $\text{Breadth} = 252 \times 100 = 25,200$。
计算信息比率：
$$
\operatorname{IR} \approx 0.10 \times \sqrt{25,200} \approx 0.10 \times 158.7 = \boxed{15.87}
$$
退一步，哪怕仅考虑时间序列维度的广度（$\text{Breadth} = 252$，完全不考虑横截面分散）：
$$
\operatorname{IR} \approx 0.10 \times \sqrt{252} \approx 0.10 \times 15.87 \approx \boxed{1.59}
$$
在实盘量化多空组合中，年化夏普比率达到 1.5 ~ 2.0 即具备极高的配置价值。
**金融市场信噪比特征**：
金融时间序列的信噪比极低（日频大部分波动为随机噪声）。宏观经济模型中常见的高 $R^2$ 在二级市场资产定价中并不存在（若出现高 $R^2$ 通常提示存在**前瞻偏差或信息泄露**）。基于主动管理基本法则，$R^2 = 1\%$（对应 $\operatorname{IC} = 0.10$）在大广度投资组合中足以产生显著的风险调整后收益。

---

### 13. ESL 3.4.1：正交设计下 OLS、Ridge、Lasso 与 Best Subset 显式闭式解推导

> **问题定义（ESL Ex 3.12 / 正交设计矩阵下四大估计量的显式解推导）**：
> 设特征矩阵 $X \in \mathbb{R}^{n \times p}$ 各列已中心化且相互正交规范化，即满足：
> $$ X^\top X = I_p $$
> 记单变量 OLS 估计量为 $\hat\beta_j^{\text{ols}} = X_j^\top Y$。
> 1. 请分别推导并写出以下四种回归方法在该正交设定下的**显式参数解析解（Closed-form Solutions）**：
>    - 普通最小二乘（OLS）；
>    - 岭回归（Ridge Regression, $\ell_2$ 惩罚）；
>    - Lasso 回归（$\ell_1$ 惩罚）；
>    - 最优子集选择（Best Subset Selection, $\ell_0$ 惩罚）；
> 2. 请比较这四种估计量关于单变量 OLS 解 $\hat\beta_j^{\text{ols}}$ 的响应函数形态，并从优化一阶条件与次梯度角度深入解释：为什么 Lasso 能够产生稀疏解（精确压缩为 0），而 Ridge 只能产生收缩解？

**思路拆解与严格推导**：

#### 1. 损失函数在正交条件下的解耦分解
对于任意回归模型，误差平方和项展开为：
$$
\begin{aligned}
\|Y - X\beta\|_2^2 &= Y^\top Y - 2\beta^\top X^\top Y + \beta^\top X^\top X \beta \\
&= Y^\top Y - 2\sum_{j=1}^p \beta_j (X_j^\top Y) + \sum_{j=1}^p \beta_j^2 \quad (\because X^\top X = I_p) \\
&= Y^\top Y - \sum_{j=1}^p (\hat\beta_j^{\text{ols}})^2 + \sum_{j=1}^p (\beta_j - \hat\beta_j^{\text{ols}})^2
\end{aligned}
$$
因为 $X^\top X = I_p$，**联合优化目标完全解耦为 $p$ 个相互独立的一维标量优化问题**：
$$ \min_\beta \sum_{j=1}^p \left[ \frac{1}{2}(\beta_j - \hat\beta_j^{\text{ols}})^2 + g(\beta_j) \right] $$

#### 2. 四大估计量的显式闭式解推导
1. **OLS（无惩罚，$g(\beta_j) = 0$）**：
   $$ \min_{\beta_j} \frac{1}{2}(\beta_j - \hat\beta_j^{\text{ols}})^2 \implies \boxed{\hat\beta_j^{\text{ols}} = X_j^\top Y} $$
2. **岭回归（Ridge，$\ell_2$ 惩罚：$g(\beta_j) = \frac{1}{2}\lambda \beta_j^2$）**：
   目标函数对 $\beta_j$ 求导令其为零：
   $$ (\beta_j - \hat\beta_j^{\text{ols}}) + \lambda \beta_j = 0 \implies (1 + \lambda)\beta_j = \hat\beta_j^{\text{ols}} \implies \boxed{\hat\beta_j^{\text{ridge}} = \frac{1}{1 + \lambda} \hat\beta_j^{\text{ols}}} $$
   **几何性质**：**线性同比例缩放（Linear Shrinkage）**。系数关于原 OLS 解处处平滑按比例缩小，斜率为 $\frac{1}{1+\lambda} < 1$，但**永不精确为零**（除非 $\hat\beta_j^{\text{ols}} = 0$）。
3. **Lasso 回归（$\ell_1$ 惩罚：$g(\beta_j) = \lambda |\beta_j|$）**：
   目标函数为不可导的凸优化问题：$\min_{\beta_j} \frac{1}{2}(\beta_j - \hat\beta_j^{\text{ols}})^2 + \lambda |\beta_j|$。利用**次梯度（Subgradient）KKT 条件**：
   $$ 0 \in (\beta_j - \hat\beta_j^{\text{ols}}) + \lambda \, \partial |\beta_j| $$
   - 若 $\beta_j > 0$，次微分 $\partial |\beta_j| = \{1\}$：$\beta_j - \hat\beta_j^{\text{ols}} + \lambda = 0 \implies \beta_j = \hat\beta_j^{\text{ols}} - \lambda$（要求 $\hat\beta_j^{\text{ols}} > \lambda$）；
   - 若 $\beta_j < 0$，次微分 $\partial |\beta_j| = \{-1\}$：$\beta_j - \hat\beta_j^{\text{ols}} - \lambda = 0 \implies \beta_j = \hat\beta_j^{\text{ols}} + \lambda$（要求 $\hat\beta_j^{\text{ols}} < -\lambda$）；
   - 若 $\beta_j = 0$，次微分 $\partial |\beta_j| = [-1, 1]$：$-\hat\beta_j^{\text{ols}} + \lambda s = 0$ 存在 $s \in [-1, 1]$ 成立 $\iff |\hat\beta_j^{\text{ols}}| \le \lambda$。
   综合得到**软阈值算子（Soft-Thresholding Operator $\mathcal{S}_\lambda$）**：
   $$ \boxed{\hat\beta_j^{\text{lasso}} = \operatorname{sign}(\hat\beta_j^{\text{ols}}) \max\left( 0, \, |\hat\beta_j^{\text{ols}}| - \lambda \right)} $$
   **几何性质**：将幅值较小（$|\hat\beta_j^{\text{ols}}| \le \lambda$）的噪声系数**精确截断为 0（稀疏性 Sparsity）**；将较强的信号向 0 平移常数 $\lambda$。
4. **最优子集选择（Best Subset，$\ell_0$ 惩罚：$g(\beta_j) = \frac{1}{2}\lambda \mathbb{I}(\beta_j \ne 0)$）**：
   - 若取 $\beta_j = 0$，损失为 $\frac{1}{2}(\hat\beta_j^{\text{ols}})^2$；
   - 若取 $\beta_j \ne 0$，最优选择为 $\beta_j = \hat\beta_j^{\text{ols}}$，损失为 $\frac{1}{2}\lambda$。
   - 两者比较：当 $\frac{1}{2}(\hat\beta_j^{\text{ols}})^2 > \frac{1}{2}\lambda \iff |\hat\beta_j^{\text{ols}}| > \sqrt{\lambda}$ 时保留原值，否则置零。
   导出**硬阈值算子（Hard-Thresholding Operator $\mathcal{H}_{\sqrt{\lambda}}$）**：
   $$ \boxed{\hat\beta_j^{\text{subset}} = \hat\beta_j^{\text{ols}} \cdot \mathbb{I}(|\hat\beta_j^{\text{ols}}| > \sqrt{\lambda})} $$

#### 3. 四大估计量对比矩阵

| 方法 | 惩罚项 | 估计量数学闭式解 $\hat\beta_j$ | 连续性 | 稀疏性（精确置零） |
| :--- | :--- | :--- | :---: | :---: |
| **OLS** | 无 | $\hat\beta_j^{\text{ols}}$ | 连续恒等映射 | 否 |
| **Ridge** | $\frac{1}{2}\lambda \beta_j^2$ | $\frac{1}{1 + \lambda}\hat\beta_j^{\text{ols}}$ | 连续平滑缩放 | 否（永不为 0） |
| **Lasso** | $\lambda \lVert\beta\rVert_1$ | $\operatorname{sign}(\hat\beta_j^{\text{ols}})(\lvert\hat\beta_j^{\text{ols}}\rvert - \lambda)_+$ | 处处连续 | **是**（小于 $\lambda$ 置零） |
| **Best Subset** | $\frac{1}{2}\lambda \mathbb{I}(\beta_j \ne 0)$ | $\hat\beta_j^{\text{ols}} \cdot \mathbb{I}(\lvert\hat\beta_j^{\text{ols}}\rvert > \sqrt{\lambda})$ | **不连续（有跳跃）** | **是**（小于 $\sqrt{\lambda}$ 置零） |

> **核心性质对比**：最优子集不连续，导致微小的样本扰动会引发变量进入/退出的剧烈跳跃（极高估计方差）；Lasso 既保留了截断为 0 的变量选择能力，又保持了响应函数的连续性，因此方差显著低于最优子集。

---

### 14. ESL 3.4.1 / Ex 3.8：岭回归 SVD 谱收缩、有效自由度与 MSE 严格优于 OLS 证明

> **定理推导（Theobald 1974 定理 / 岭回归 MSE 严格优于 OLS）**：
> 设中心化设计矩阵 $X \in \mathbb{R}^{n \times p}$（满列秩 $\operatorname{rank}(X) = p \le n$）的奇异值分解（SVD）为：
> $$ X = U D V^\top $$
> 其中 $U \in \mathbb{R}^{n \times p}$ 满足 $U^\top U = I_p$，$V \in \mathbb{R}^{p \times p}$ 为正交矩阵，$D = \operatorname{diag}(d_1, \dots, d_p)$，$d_1 \ge d_2 \ge \dots \ge d_p > 0$。
> 1. 用奇异值 $d_j$ 和左奇异向量 $u_j$ 显式展开岭回归拟合值向量 $\hat{Y}^{\text{ridge}} = X\hat\beta^{\text{ridge}}$，并分析岭回归对不同主成分因子的收缩特性；
> 2. 证明岭回归的有效自由度 $\operatorname{df}(\lambda) = \operatorname{tr}(H_\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$，并证明其关于 $\lambda \ge 0$ 是严格单调递减的；
> 3. **Theobald (1974) 定理**：无论真实参数 $\beta$ 和扰动方差 $\sigma^2$ 为何值，**严格证明总存在 $\lambda^* > 0$，使得岭回归估计量的总均方误差（Total MSE）严格小于 OLS 估计量**：
>    $$ \operatorname{MSE}(\hat\beta^{\text{ridge}}(\lambda^*)) < \operatorname{MSE}(\hat\beta^{\text{ols}}) $$

**思路拆解与严格推导**：

#### 1. SVD 谱收缩展开式
由 $X = U D V^\top$ 可得：$X^\top X = V D^2 V^\top$。
岭回归封闭解代入 SVD：
$$
\begin{aligned}
\hat\beta^{\text{ridge}} &= (X^\top X + \lambda I)^{-1} X^\top Y \\
&= \left[ V (D^2 + \lambda I) V^\top \right]^{-1} V D U^\top Y \\
&= V (D^2 + \lambda I)^{-1} D U^\top Y = V \operatorname{diag}\left( \frac{d_j}{d_j^2 + \lambda} \right) U^\top Y
\end{aligned}
$$
拟合向量 $\hat{Y}^{\text{ridge}} = X\hat\beta^{\text{ridge}}$ 为：
$$
\hat{Y}^{\text{ridge}} = (U D V^\top) V (D^2 + \lambda I)^{-1} D U^\top Y = U \operatorname{diag}\left( \frac{d_j^2}{d_j^2 + \lambda} \right) U^\top Y = \sum_{j=1}^p u_j \left( \frac{d_j^2}{d_j^2 + \lambda} \right) u_j^\top Y
$$
- **与 OLS 对比**：OLS 对应 $\lambda = 0$，$\hat{Y}^{\text{ols}} = \sum_{j=1}^p u_j (u_j^\top Y)$。
- **谱收缩物理意义**：每个主成分方向 $u_j$ 的收缩因子为 $f_j = \frac{d_j^2}{d_j^2 + \lambda}$。
  - 对于方差最大的主成分（$d_1^2 \gg \lambda$），$f_1 \approx 1$，基本不压缩；
  - 对于方差最小的主成分（$d_p^2 \ll \lambda$，共线性严重的方向），$f_p \to 0$，**被剧烈压缩归零**！
  - 岭回归本质上是在主成分坐标系下对“低信噪比、共线性强”的微弱奇异方向进行保护性滤波。

#### 2. 有效自由度推导
帽子矩阵为 $H_\lambda = U \operatorname{diag}\left( \frac{d_j^2}{d_j^2 + \lambda} \right) U^\top$。
$$ \operatorname{df}(\lambda) = \operatorname{tr}(H_\lambda) = \operatorname{tr}\left( \operatorname{diag}\left( \frac{d_j^2}{d_j^2 + \lambda} \right) U^\top U \right) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} $$
对 $\lambda$ 求一阶导数：
$$ \frac{d}{d\lambda} \operatorname{df}(\lambda) = -\sum_{j=1}^p \frac{d_j^2}{(d_j^2 + \lambda)^2} < 0 \quad (\forall \lambda \ge 0) $$
因此，$\operatorname{df}(\lambda)$ 随惩罚强度 $\lambda$ 的增大而严格单调递减：$\operatorname{df}(0) = p$，$\lim_{\lambda \to \infty} \operatorname{df}(\lambda) = 0$。

#### 3. Theobald 定理严格证明：MSE 必然可以被 Ridge 改善
均方误差（MSE）定义为：
$$ \operatorname{MSE}(\hat\beta) = E[\|\hat\beta - \beta\|_2^2] = \operatorname{tr}(\operatorname{Var}(\hat\beta)) + \|\operatorname{Bias}(\hat\beta)\|_2^2 $$
- **方差项（Variance）**：
  $$ \operatorname{Var}(\hat\beta^{\text{ridge}}) = \sigma^2 (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1} = \sigma^2 V \operatorname{diag}\left( \frac{d_j^2}{(d_j^2 + \lambda)^2} \right) V^\top $$
  其迹为：$\operatorname{tr}(\operatorname{Var}) = \sigma^2 \sum_{j=1}^p \frac{d_j^2}{(d_j^2 + \lambda)^2}$。
- **偏差项（Bias）**：
  $$ \operatorname{Bias}(\hat\beta^{\text{ridge}}) = E[\hat\beta^{\text{ridge}}] - \beta = -\lambda (X^\top X + \lambda I)^{-1} \beta $$
  令正交坐标系下的真实参数为 $\alpha = V^\top \beta = (\alpha_1, \dots, \alpha_p)^\top$：
  $$ \|\operatorname{Bias}\|^2 = \lambda^2 \beta^\top V (D^2 + \lambda I)^{-2} V^\top \beta = \lambda^2 \sum_{j=1}^p \frac{\alpha_j^2}{(d_j^2 + \lambda)^2} $$
- **总 MSE 关于 $\lambda$ 的导数分析**：
  $$ \operatorname{MSE}(\lambda) = \sum_{j=1}^p \frac{\sigma^2 d_j^2 + \lambda^2 \alpha_j^2}{(d_j^2 + \lambda)^2} $$
  求导：
  $$
  \begin{aligned}
  \frac{d}{d\lambda} \operatorname{MSE}(\lambda) &= \sum_{j=1}^p \frac{2\lambda \alpha_j^2 (d_j^2 + \lambda)^2 - 2(d_j^2 + \lambda)(\sigma^2 d_j^2 + \lambda^2 \alpha_j^2)}{(d_j^2 + \lambda)^4} \\
  &= \sum_{j=1}^p \frac{2\lambda \alpha_j^2 (d_j^2 + \lambda) - 2(\sigma^2 d_j^2 + \lambda^2 \alpha_j^2)}{(d_j^2 + \lambda)^3} \\
  &= \sum_{j=1}^p \frac{2\lambda d_j^2 \alpha_j^2 - 2\sigma^2 d_j^2}{(d_j^2 + \lambda)^3}
  \end{aligned}
  $$
  计算在 $\lambda = 0$（即 OLS 处）的导数值：
  $$ \left. \frac{d}{d\lambda} \operatorname{MSE}(\lambda) \right|_{\lambda = 0} = \sum_{j=1}^p \frac{-2\sigma^2 d_j^2}{d_j^6} = -2\sigma^2 \sum_{j=1}^p \frac{1}{d_j^4} < 0 $$
  **核心结论**：在 $\lambda = 0$ 处，总均方误差对 $\lambda$ 的导数**严格小于零**！
  由于 $\operatorname{MSE}(\lambda)$ 在 $[0, \infty)$ 上连续可微，根据极限定义，必存在充分小的 $\lambda^* > 0$，使得：
  $$ \operatorname{MSE}(\hat\beta^{\text{ridge}}(\lambda^*)) < \operatorname{MSE}(\hat\beta^{\text{ols}}) $$
  **证毕！** 这证明了引入适量偏误（$\lambda > 0$）所换取的方差削减率严格大于偏差增加率，OLS 在无偏估计中虽是方差最小的（Gauss-Markov），但在放宽无偏限制后，其均方误差必然可以被正则化收缩改进。

---

### 15. ESL 6.1.1 / Ex 6.1–6.2：局部线性回归等价核闭式解、一阶矩条件与边界偏差消除

> **定理推导（ESL Ch.6 / 局部线性回归等价核与边界无偏性）**：
> 在非参数回归中，给定样本 $(X_i, Y_i)_{i=1}^n$。局部线性回归在查询点 $x_0$ 处求解加权最小二乘：
> $$ \min_{\alpha, \beta} \sum_{i=1}^n K_h(X_i - x_0) \left[ Y_i - \alpha - \beta(X_i - x_0) \right]^2 $$
> 其中 $K(u)$ 是对称概率核，$K_h(u) = \frac{1}{h} K(u/h)$。估计值为 $\hat{f}(x_0) = \hat\alpha$。
> 1. 求解加权正规方程，证明拟合值可表达为线性平滑器 $\hat{f}(x_0) = \sum_{i=1}^n l_i(x_0) Y_i$，并求出等价核权重 $l_i(x_0)$ 的显式闭式解（用核样本矩 $s_r(x_0) = \sum_{i=1}^n K_h(X_i - x_0)(X_i - x_0)^r$ 表示）；
> 2. 严格证明等价核权重 $l_i(x_0)$ 自动满足零阶矩与一阶矩条件：
>    $$ \sum_{i=1}^n l_i(x_0) = 1, \quad \sum_{i=1}^n (X_i - x_0) l_i(x_0) = 0 $$
> 3. 设真实条件均值 $f(x)$ 二阶连续可导。推导为什么 Nadaraya–Watson 局部常数回归在定义域边界处的偏差为 $O(h)$，而局部线性回归能自动消除一阶导数偏差，使边界偏差达到与内部同阶的 $O(h^2)$？

**思路拆解与严格推导**：

#### 1. 加权最小二乘求解与等价核解析式
令 $z_i = X_i - x_0$，$w_i = K_h(z_i)$。局部设计矩阵与加权对角阵为：
$$ B = \begin{pmatrix} 1 & z_1 \\ 1 & z_2 \\ \vdots & \vdots \\ 1 & z_n \end{pmatrix} \in \mathbb{R}^{n \times 2}, \quad W = \operatorname{diag}(w_1, \dots, w_n) $$
参数向量 $(\hat\alpha, \hat\beta)^\top = (B^\top W B)^{-1} B^\top W Y$。
计算加权 Gram 矩阵：
$$ B^\top W B = \begin{pmatrix} \sum_{i=1}^n w_i & \sum_{i=1}^n w_i z_i \\ \sum_{i=1}^n w_i z_i & \sum_{i=1}^n w_i z_i^2 \end{pmatrix} = \begin{pmatrix} s_0(x_0) & s_1(x_0) \\ s_1(x_0) & s_2(x_0) \end{pmatrix} $$
行列式为 $D = s_0 s_2 - s_1^2$。求 $2 \times 2$ 逆矩阵：
$$ (B^\top W B)^{-1} = \frac{1}{s_0 s_2 - s_1^2} \begin{pmatrix} s_2 & -s_1 \\ -s_1 & s_0 \end{pmatrix} $$
拟合值 $\hat{f}(x_0) = \hat\alpha = e_1^\top (B^\top W B)^{-1} B^\top W Y$。取第一行内积：
$$
\begin{aligned}
\hat{f}(x_0) &= \frac{1}{s_0 s_2 - s_1^2} \begin{pmatrix} s_2 & -s_1 \end{pmatrix} \begin{pmatrix} \sum w_i Y_i \\ \sum w_i z_i Y_i \end{pmatrix} \\
&= \sum_{i=1}^n \left[ \frac{w_i (s_2 - s_1 z_i)}{s_0 s_2 - s_1^2} \right] Y_i
\end{aligned}
$$
因此，等价核权重函数 $l_i(x_0)$ 的闭式解析式为：
$$ \boxed{l_i(x_0) = \frac{K_h(X_i - x_0) \left[ s_2(x_0) - s_1(x_0)(X_i - x_0) \right]}{s_0(x_0) s_2(x_0) - s_1^2(x_0)}} $$

#### 2. 矩条件的严格代数证明
- **零阶矩条件（加和为 1）**：
  $$ \sum_{i=1}^n l_i(x_0) = \frac{s_2 \sum w_i - s_1 \sum w_i z_i}{s_0 s_2 - s_1^2} = \frac{s_2 s_0 - s_1 s_1}{s_0 s_2 - s_1^2} = \frac{s_0 s_2 - s_1^2}{s_0 s_2 - s_1^2} \equiv \boxed{1} $$
- **一阶矩条件（正交为 0）**：
  $$ \sum_{i=1}^n (X_i - x_0) l_i(x_0) = \sum_{i=1}^n z_i l_i(x_0) = \frac{s_2 \sum w_i z_i - s_1 \sum w_i z_i^2}{s_0 s_2 - s_1^2} = \frac{s_2 s_1 - s_1 s_2}{s_0 s_2 - s_1^2} \equiv \boxed{0} $$

#### 3. 边界偏差机理与数学展开
对真实函数 $f(X_i)$ 在 $x_0$ 处做二阶泰勒展开：
$$ f(X_i) = f(x_0) + f'(x_0)(X_i - x_0) + \frac{1}{2} f''(x_0)(X_i - x_0)^2 + o((X_i - x_0)^2) $$
条件期望值为：
$$
\begin{aligned}
E[\hat{f}(x_0) \mid X] &= \sum_{i=1}^n l_i(x_0) f(X_i) \\
&= f(x_0) \underbrace{\sum_{i=1}^n l_i(x_0)}_{= 1} + f'(x_0) \underbrace{\sum_{i=1}^n (X_i - x_0) l_i(x_0)}_{= 0} + \frac{1}{2} f''(x_0) \sum_{i=1}^n (X_i - x_0)^2 l_i(x_0) + \dots \\
&= f(x_0) + \frac{1}{2} f''(x_0) \sum_{i=1}^n (X_i - x_0)^2 l_i(x_0) + O(h^3)
\end{aligned}
$$
**边界偏差对比**：
- **Nadaraya-Watson 局部常数回归**：
  权重为 $l_i^{\text{NW}}(x_0) = \frac{w_i}{s_0}$。
  一阶矩项为 $\sum z_i l_i^{\text{NW}} = \frac{s_1(x_0)}{s_0(x_0)}$。
  在定义域内部，由于核对称且数据均匀，$s_1 \approx 0$；
  但在边界点（如 $x_0 = 0$，数据全部位于 $X_i \ge 0$ 右侧），核被截断，一阶矩 $s_1 = \sum w_i z_i \sim O(h)$ 严重不为 0！
  导致 Nadaraya-Watson 局部常数在边界处的偏差为主项：
  $$ \operatorname{Bias}_{\text{NW}}(0) = f'(0) \frac{s_1(0)}{s_0(0)} = \boxed{O(h)} $$
- **局部线性回归（Local Linear Regression）**：
  通过在拟合中内建局部斜率参数 $\beta$，构造出的等价核 $l_i(x_0)$ **无论在内部还是边界，一阶矩恒等为 0**！
  一阶导数偏差项被代数结构完全抵消，因此边界偏差直接提升至：
  $$ \operatorname{Bias}_{\text{LLR}}(0) = \frac{1}{2} f''(0) \sum_{i=1}^n z_i^2 l_i(0) = \boxed{O(h^2)} $$
  这就是 ESL 中著名的“**自动核修缮（Automatic Kernel Carpentry）**”！

---

### 16. ESL 6.2 / Ex 6.3：核平滑矩阵 $S_\lambda$ 性质辨析、两类有效自由度与波动率曲面拟合

> **定理推导（ESL Ch.6 / 线性平滑矩阵性质与两类有效自由度）**：
> 将所有线性平滑器统一写作矩阵形式：$\hat{Y} = S_\lambda Y$，其中 $S_\lambda \in \mathbb{R}^{n \times n}$ 为平滑矩阵（Smoother Matrix）。
> 1. 证明对非均匀分布的数据点，局部多项式回归的平滑矩阵 $S_\lambda$ 满足行和为 1（$S_\lambda \mathbf{1} = \mathbf{1}$），但**通常不对称**（$S_\lambda^\top \ne S_\lambda$），且**不幂等**（$S_\lambda^2 \ne S_\lambda$）；
> 2. 统计学中定义了两类有效自由度：$\operatorname{df}_{\text{fit}} = \operatorname{tr}(S_\lambda)$ 与 $\operatorname{df}_{\text{var}} = \operatorname{tr}(S_\lambda S_\lambda^\top)$。解释两者的统计含义，并证明对于对称平滑矩阵恒有 $\operatorname{df}_{\text{var}} \le \operatorname{df}_{\text{fit}}$；
> 3. 在量化金融中拟合期权隐含波动率曲面（Implied Volatility Surface）时，若直接使用传统回归参数个数计算 AIC/BIC，会导致什么陷阱？如何利用广义交叉验证（GCV）科学控制模型复杂度？

**思路拆解与严格推导**：

#### 1. 平滑矩阵的三大核心性质
1. **行和为 1（保留常数）**：
   若因变量为常数向量 $Y = c \mathbf{1}$，局部多项式拟合中多项式可以完全无误差拟合常数，得到预测向量 $\hat{Y} = c \mathbf{1}$。
   因此 $S_\lambda (c \mathbf{1}) = c (S_\lambda \mathbf{1}) = c \mathbf{1} \implies S_\lambda \mathbf{1} = \mathbf{1}$。
2. **不对称性（$S_\lambda^\top \ne S_\lambda$）**：
   矩阵元素 $S_{ij} = l_j(X_i)$ 表示第 $j$ 个样本观测值对第 $i$ 个位置拟合值的权重贡献。
   $l_j(X_i)$ 依赖于以 $X_i$ 为中心的核权重归一化因子 $\sum_k K_h(X_k - X_i)$；而 $l_i(X_j)$ 依赖于以 $X_j$ 为中心的归一化因子。除非样本点在网格上严格均匀周期分布，否则由于样本密度不同，$S_{ij} \ne S_{ji}$。
3. **非幂等性（$S_\lambda^2 \ne S_\lambda$）**：
   正交投影矩阵（如 OLS 的帽子矩阵 $H = X(X^\top X)^{-1}X^\top$）满足 $H^2 = H$；
   而平滑矩阵 $S_\lambda$ 并不对应向特定有限维子空间的正交投影，对已平滑的序列再次平滑（$S_\lambda (S_\lambda Y)$）相当于执行二次低通滤波，拟合曲线会进一步被抹平，$S_\lambda^2 \ne S_\lambda$。

#### 2. 两类有效自由度的统计本质与大小不等式证明
- **$\operatorname{df}_{\text{fit}} = \operatorname{tr}(S_\lambda)$（拟合自由度 / Efron 自由度）**：
  在误差同方差且独立假定下（$\operatorname{Var}(Y) = \sigma^2 I$），考察拟合值与真实观测值的总协方差：
  $$ \sum_{i=1}^n \frac{\operatorname{Cov}(\hat{Y}_i, Y_i)}{\sigma^2} = \sum_{i=1}^n \frac{\operatorname{Cov}\left( \sum_{j=1}^n S_{ij} Y_j, \, Y_i \right)}{\sigma^2} = \sum_{i=1}^n \frac{S_{ii} \sigma^2}{\sigma^2} = \sum_{i=1}^n S_{ii} = \operatorname{tr}(S_\lambda) $$
  它度量了模型预测对训练数据自身波动的**平均敏感度（自相关联程度）**。
- **$\operatorname{df}_{\text{var}} = \operatorname{tr}(S_\lambda S_\lambda^\top)$（方差自由度）**：
  计算所有拟合点估计方差的总和：
  $$ \sum_{i=1}^n \frac{\operatorname{Var}(\hat{Y}_i)}{\sigma^2} = \frac{1}{\sigma^2} \operatorname{tr}(\operatorname{Var}(S_\lambda Y)) = \frac{1}{\sigma^2} \operatorname{tr}(S_\lambda (\sigma^2 I) S_\lambda^\top) = \operatorname{tr}(S_\lambda S_\lambda^\top) $$
  它度量了模型预测的总波动消耗。

**不等式 $\operatorname{df}_{\text{var}} \le \operatorname{df}_{\text{fit}}$ 证明（以对称平滑器为例）**：
若平滑器对称（如平滑样条 Smoothing Splines），$S_\lambda$ 实对称矩阵可对角化，其特征值为 $\gamma_1, \dots, \gamma_n$。
因为平滑算子具有收缩滤波特性（Shrinkage），其所有特征值满足 $0 \le \gamma_i \le 1$。
$$ \operatorname{df}_{\text{fit}} = \operatorname{tr}(S_\lambda) = \sum_{i=1}^n \gamma_i $$
$$ \operatorname{df}_{\text{var}} = \operatorname{tr}(S_\lambda S_\lambda^\top) = \operatorname{tr}(S_\lambda^2) = \sum_{i=1}^n \gamma_i^2 $$
由于 $\gamma_i \in [0, 1]$，显然 $\gamma_i^2 \le \gamma_i$。因此：
$$ \operatorname{df}_{\text{var}} = \sum_{i=1}^n \gamma_i^2 \le \sum_{i=1}^n \gamma_i = \operatorname{df}_{\text{fit}} $$
等号成立当且仅当所有非零特征值均为 1（即 $S_\lambda$ 是正交投影矩阵，退化为普通无偏 OLS）！

#### 3. 期权波动率曲面平滑的量化实战启示
- **离散参数计数陷阱**：在期权做市中，沿执行价 $K$ 和到期时间 $T$ 平滑隐含波动率（IV）曲面时，局部多项式并没有传统意义上的“显式参数个数 $k$”。如果错误地将参数个数设为常数（如 3），计算得到的赤池信息量（AIC）将失效；
- **GCV 自动化正则化**：在波动率微观结构噪声下，必须使用基于有效自由度的**广义交叉验证（Generalized Cross-Validation, GCV）**选取最优核带宽 $h$：
  $$ \mathrm{GCV}(h) = \frac{\frac{1}{n} \|Y - \hat{Y}\|_2^2}{\left( 1 - \frac{\operatorname{tr}(S_h)}{n} \right)^2} $$
  分母中的 $\operatorname{tr}(S_h)$ 严格充当了“模型有效参数量”，避免了带宽过小引发波动率微结构假套利峰谷，确保了期权无套利平滑曲面的稳健性。

---

## 模块六：知识结构梳理与核心要点清单

```text
回归与平滑模型核心要点清单：
1. 单变量 OLS 估计量：斜率 \hat\beta = \rho \cdot (\sigma_y / \sigma_x)，拟合优度 R^2 = \rho^2。
2. 逆向回归与均值回归：正向与逆向回归斜率乘积为 \rho^2 \le 1；受随机噪声稀释，不可直接取倒数。
3. BLUE 条件与正态性边界：Gauss-Markov 定理仅要求一阶外生性与二阶球形扰动；正态性仅在有限样本精确 t/F 检验与达到 UMVUE 时需要。
4. 违背球形扰动的后果：在异方差或自相关下，OLS 估计量依然无偏且一致，但普通协方差被低估（产生虚假显著）；需采用 White (HC0) 或 Newey-West (HAC) 稳健标准误。
5. 正则化几何机制：Lasso 的 \ell_1 等值线具备非光滑尖角，易与残差等高线切于坐标轴（产生稀疏解）；Ridge 的 \ell_2 等值线为光滑超球体，沿低方差主成分方向进行平滑谱收缩。
6. 核平滑边界偏差与高维拓展：Nadaraya-Watson（局部常数核估计）在边界处存在 O(h) 阶偏差；局部线性回归自动满足一阶正交矩，使边界偏差阶数降至 O(h^2)；应对维数灾难可采用可加模型（GAM）或状态依赖变系数模型。
```

---

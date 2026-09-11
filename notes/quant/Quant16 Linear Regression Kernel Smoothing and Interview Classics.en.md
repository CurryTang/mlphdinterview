# Quant 16 · Linear Regression and Kernel Smoothing: OLS, Gauss–Markov, Ridge/Lasso, and Theoretical Foundations

Linear regression forms the foundational bedrock of quantitative research and statistical modeling. Mastering linear models requires not only deriving the closed-form analytical solutions for OLS estimators, but also systematically understanding the geometry of high-dimensional Euclidean projections, the mathematical boundaries of the Gauss–Markov theorem, and the theoretical remedies and robust estimation procedures when financial time series violate spherical disturbance and exogeneity assumptions (heteroskedasticity, autocorrelation, multicollinearity, measurement errors, and omitted variables).

```text
Core Theoretical Mental Models:
1. Univariate OLS Fundamental Identities: Sample estimator \hat\beta_1 = \hat\rho_{XY} (s_Y / s_X) (population form \beta_1 = \rho (\sigma_Y / \sigma_X)) and R^2 = \hat\rho^2.
2. Regression Asymmetry: The product of the forward slope of y on x and the reverse slope of x on y is \rho^2 \le 1; never invert directly.
3. Geometric Orthogonal Projection: View OLS as the orthogonal projection of y onto the column space of X. Orthogonality is the algebraic bedrock of residual properties.
4. BLUE Does Not Require Normality: The Gauss-Markov theorem proves OLS is BLUE under moment assumptions alone without assuming normal errors. Normality is required only for exact finite-sample t and F tests.
5. Regularization Geometry: Lasso's \ell_1 diamond induces sparsity (variable selection), while Ridge's \ell_2 sphere induces spectral shrinkage (mitigating multicollinearity variance).
```

> 🧭 **Core Knowledge Landscape**
> - **Module 1: OLS Geometry & Algebra**: Normal Equations | 5 Dimensions of Residual Orthogonality & ANOVA | Coefficients vs. Covariance | Reverse Regression Trap
> - **Module 2: Gauss–Markov, Statistical Inference & Core Analytical Lemmas**: Estimator Properties | t/F Tests & Restricted Models | Prediction vs Confidence Intervals | LOOCV & Leverage | Measurement Errors & OVB
> - **Module 3: Variable Selection & Shrinkage**: Best Subset | Ridge Regression | Lasso | Geometric Intuition & Comparison
> - **Module 4: Kernel Smoothing & Local Regression**: Conditional Expectation & Essence of Kernels | Nadaraya-Watson | Boundary Bias & Local Linear | Curse of Dimensionality
> - **Module 5: Core Classical Problems and Analytical Proofs (Green Book + HOTS + ESL Calculations)**: Correlation Bounds | Equicorrelated Matrix Lower Bound | Cholesky Simulation | CAPM & Reverse Regression | Affine Invariance | Omitted Variable Bias | Measurement Error | Multicollinearity & VIF | Optimal Futures Hedge Ratio | FWL Theorem & Two-Stage Residual Regression Trap (Ratio β₁ / β₂) | Regression Without Intercept Trap | R² vs. Real-World IC | Closed-Form Derivations under Orthogonal Designs | Ridge SVD Spectral Shrinkage & MSE Strict Dominance | Local Linear Equivalent Kernel & Boundary Bias Removal | Smoother Matrix Properties & Two Types of Effective Degrees of Freedom
> - **Module 6: Core Knowledge Checklist & Diagnostic Traps**

---

## Module 1: OLS Geometry and Algebra (ESL 3.2)

### 1. Simple Linear Regression Model and OLS Estimator

#### Model Formulation & Notation Conventions (Rigorous Definitions)
Suppose we observe $n$ independent paired samples $\{(X_i, Y_i)\}_{i=1}^n$:
- **$X_i \in \mathbb{R}$**: Independent variable (explanatory variable / regressor / feature), the known input feature value;
- **$Y_i \in \mathbb{R}$**: Dependent variable (response variable / ground truth / target), the observed outcome we seek to predict or explain;
- **$\beta_0, \beta_1 \in \mathbb{R}$**: Unknown true population regression parameters ($\beta_0$ is population intercept, $\beta_1$ is population slope);
- **$\varepsilon_i \in \mathbb{R}$**: Unobservable population random error term (disturbance / noise), representing omitted factors and stochastic noise.

The true population data-generating process is:

$$
Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i \quad (i = 1, 2, \dots, n)
$$

#### Estimators, Predictions, and Residuals (Meaning of the "Hat" Symbol $\ \hat{}\ $)
In standard statistical convention, a **"hat" ($\ \hat{}\ $) signifies a sample estimate or model prediction calculated from empirical data** (distinguishing it from the unobservable population parameter):
- **$\hat\beta_0, \hat\beta_1$**: Sample estimates of the intercept and slope coefficients;
- **$\hat{Y}_i$ (pronounced "Y-hat", Fitted Value / Prediction)**: The model's linear point prediction for sample $i$:
  $$\hat{Y}_i = \hat\beta_0 + \hat\beta_1 X_i$$
- **$\hat\varepsilon_i$ (pronounced "epsilon-hat", Sample Residual)**: The deviation between observed target $Y_i$ and fitted prediction $\hat{Y}_i$ (the empirical prediction error on sample $i$):
  $$\hat\varepsilon_i = Y_i - \hat{Y}_i = Y_i - (\hat\beta_0 + \hat\beta_1 X_i)$$

#### Formal Conventions for Sample Statistics vs. Population Parameters ($s_X, s_Y, \sigma_X, \sigma_Y, \rho$)
In statistical modeling and quantitative analysis, mathematical rigor requires adhering to the universal convention: **Latin letters denote finite-sample estimators (sample statistics computed from observed data), while Greek letters denote the unobserved, true parameters of the underlying data-generating process (DGP)**:

1. **Finite-Sample Statistics (Computed from $n$ sample pairs; random variables subject to sampling variation)**:
   Let $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ and $\bar{Y} = \frac{1}{n}\sum_{i=1}^n Y_i$ denote sample arithmetic means:
   - **Sample Variances $s_X^2 = \widehat{\operatorname{Var}}(X)$ and $s_Y^2 = \widehat{\operatorname{Var}}(Y)$**:
     $$s_X^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2, \quad s_Y^2 = \frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2$$
   - **Sample Standard Deviations $s_X$ and $s_Y$** (often written in lowercase $s_x, s_y$):
     $$s_X = \sqrt{s_X^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2}, \quad s_Y = \sqrt{s_Y^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (Y_i - \bar{Y})^2}$$
     *Physical Meaning*: The characteristic dispersion scale of observed sample points around their mean, possessing the identical physical units/dimension as the raw variable.
   - **Sample Covariance $\widehat{\operatorname{Cov}}(X, Y)$**:
     $$\widehat{\operatorname{Cov}}(X, Y) = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})$$
   - **Sample Pearson Correlation Coefficient $\hat\rho_{XY}$** (commonly denoted $r_{XY}$ or $r$):
     $$\hat\rho_{XY} = \frac{\widehat{\operatorname{Cov}}(X, Y)}{s_X s_Y} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2}\sqrt{\sum_{i=1}^n (Y_i - \bar{Y})^2}} \in [-1, 1]$$
     *Physical Meaning*: A dimensionless scalar measuring the pure linear co-dependence between $X$ and $Y$ isolated from physical scale.

2. **Population Parameters (Inherent theoretical properties of the DGP; constant and free from sampling noise)**:
   - **Population Variances $\sigma_X^2 = \operatorname{Var}(X)$ and $\sigma_Y^2 = \operatorname{Var}(Y)$**: $\sigma_X^2 = \mathbb{E}[(X - \mathbb{E}[X])^2]$, $\sigma_Y^2 = \mathbb{E}[(Y - \mathbb{E}[Y])^2]$.
   - **Population Standard Deviations $\sigma_X$ and $\sigma_Y$** (often written in lowercase $\sigma_x, \sigma_y$): $\sigma_X = \sqrt{\operatorname{Var}(X)}$, $\sigma_Y = \sqrt{\operatorname{Var}(Y)}$.
   - **Population Covariance & Correlation**: $\operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]$, $\rho = \frac{\operatorname{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]$.
   - **Population Noise Variance $\sigma^2$** (or $\sigma_\varepsilon^2$): The intrinsic physical white noise variance of the unobserved disturbance, $\operatorname{Var}(\varepsilon_i \mid X) = \sigma^2$.
   - **Asymptotic Convergence**: By the Weak Law of Large Numbers (WLLN), as sample size $n \to \infty$, sample statistics converge in probability to population parameters: $s_X \xrightarrow{p} \sigma_X$, $s_Y \xrightarrow{p} \sigma_Y$, $\hat\rho_{XY} \xrightarrow{p} \rho$.

#### OLS Objective and Closed-Form Derivation
Ordinary Least Squares (OLS) chooses estimates $(\hat\beta_0, \hat\beta_1)$ that globally minimize the Residual Sum of Squares ($RSS$):

$$
\min_{\hat\beta_0, \hat\beta_1} \sum_{i=1}^n \hat\varepsilon_i^2 = \min_{\hat\beta_0, \hat\beta_1} \sum_{i=1}^n \left( Y_i - \hat\beta_0 - \hat\beta_1 X_i \right)^2
$$

Setting the first-order partial derivatives to zero (FOC):
1. $\frac{\partial}{\partial \hat\beta_0} = -2\sum_{i=1}^n (Y_i - \hat\beta_0 - \hat\beta_1 X_i) = 0 \implies \sum_{i=1}^n \hat\varepsilon_i = 0 \implies \hat\beta_0 = \bar{Y} - \hat\beta_1 \bar{X}$;
2. Substituting into the derivative with respect to $\hat\beta_1$: $\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y} - \hat\beta_1(X_i - \bar{X})) = 0$, yielding:

$$
\hat\beta_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2} = \frac{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2} = \frac{\widehat{\operatorname{Cov}}(X, Y)}{\widehat{\operatorname{Var}}(X)}, \quad \hat\beta_0 = \bar{Y} - \hat\beta_1 \bar{X}
$$

The degree-of-freedom normalization factor $\frac{1}{n-1}$ cancels out identically. **The univariate OLS slope is precisely the ratio of the sample covariance $\widehat{\operatorname{Cov}}(X, Y)$ to the sample variance $\widehat{\operatorname{Var}}(X)$**.

- **Center-of-Mass Pivot**: By $\bar{Y} = \hat\beta_0 + \hat\beta_1 \bar{X}$, the sample center of mass $(\bar{X}, \bar{Y})$ serves as a rigid rotational pivot. Regardless of how the slope varies, the fitted line is strictly constrained to pass through the centroid.
- **Torque and Spring Equilibrium**: Minimizing $\sum \hat\varepsilon_i^2$ is physically equivalent to each data point pulling a rigid lever via a vertical spring (with potential energy $E_p \propto \Delta y^2$). At the static equilibrium state of minimum total potential energy, the net vertical force vanishes ($\sum \hat\varepsilon_i = 0$), and the net torque about the center of mass also vanishes ($\sum (X_i - \bar{X})\hat\varepsilon_i = 0$).

---

### 2. Coefficient of Determination $R^2$ and Variance Decomposition (ANOVA)

#### Scalar-to-Vector Mapping in Sample Space $\mathbb{R}^n$
To view variance decomposition through high-dimensional Euclidean geometry, we stack sample observations into $n$-dimensional vectors:
- **Observed Target Vector $Y \in \mathbb{R}^n$**: $Y = (Y_1, Y_2, \dots, Y_n)^\top$;
- **Fitted Prediction Vector $\hat{Y} \in \mathbb{R}^n$**: $\hat{Y} = (\hat{Y}_1, \hat{Y}_2, \dots, \hat{Y}_n)^\top$;
- **Sample Mean Baseline Vector $\bar{Y}\mathbf{1} \in \mathbb{R}^n$**: scalar mean $\bar{Y}$ multiplied by the all-ones vector $\mathbf{1} = (1, 1, \dots, 1)^\top$, representing the zero-predictor naive baseline;
- **Residual Vector $\hat\varepsilon \in \mathbb{R}^n$**: $\hat\varepsilon = Y - \hat{Y} = (\hat\varepsilon_1, \hat\varepsilon_2, \dots, \hat\varepsilon_n)^\top$.

#### Physical Definitions of the Three Sums of Squares
$R^2$ quantifies the goodness of fit of the regression model relative to the naive mean baseline:

$$
R^2 = \frac{ESS}{TSS} = 1 - \frac{RSS}{TSS}
$$

The three sums of squares correspond to squared Euclidean norms in $\mathbb{R}^n$:
- **Total Sum of Squares ($TSS$)**:
  $$TSS = \sum_{i=1}^n (Y_i - \bar{Y})^2 = \|Y - \bar{Y}\mathbf{1}\|_2^2$$
  *Physical Meaning*: Total variation of observed responses around their mean baseline (the total squared prediction error incurred if one blindly predicts $\bar{Y}$ without features).
- **Explained Sum of Squares ($ESS$)**:
  $$ESS = \sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2 = \|\hat{Y} - \bar{Y}\mathbf{1}\|_2^2$$
  *Physical Meaning*: Variation of fitted values around the mean baseline, representing the **variance successfully accounted for by features $X$**.
- **Residual Sum of Squares ($RSS$)**:
  $$RSS = \sum_{i=1}^n (Y_i - \hat{Y}_i)^2 = \sum_{i=1}^n \hat\varepsilon_i^2 = \hat\varepsilon^\top \hat\varepsilon = \|Y - \hat{Y}\|_2^2$$
  *Physical Meaning*: Variation remaining in the residuals, representing **unexplained error variance**.

#### Algebraic Proof of the ANOVA Identity
Consider the identity splitting the total deviation for each sample:
$$(Y_i - \bar{Y}) = (\hat{Y}_i - \bar{Y}) + (Y_i - \hat{Y}_i) = (\hat{Y}_i - \bar{Y}) + \hat\varepsilon_i$$
Squaring both sides and summing over all $n$ observations:
$$\sum_{i=1}^n (Y_i - \bar{Y})^2 = \sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2 + \sum_{i=1}^n \hat\varepsilon_i^2 + 2\sum_{i=1}^n (\hat{Y}_i - \bar{Y})\hat\varepsilon_i$$
Expanding the cross term:
$$\sum_{i=1}^n (\hat{Y}_i - \bar{Y})\hat\varepsilon_i = \sum_{i=1}^n \hat{Y}_i \hat\varepsilon_i - \bar{Y}\sum_{i=1}^n \hat\varepsilon_i$$
1. By OLS first-order conditions, the residuals sum to zero: $\sum_{i=1}^n \hat\varepsilon_i = 0$;
2. Substituting $\hat{Y}_i = \hat\beta_0 + \hat\beta_1 X_i$ into the first term:
   $$\sum_{i=1}^n \hat{Y}_i \hat\varepsilon_i = \hat\beta_0 \sum_{i=1}^n \hat\varepsilon_i + \hat\beta_1 \sum_{i=1}^n X_i \hat\varepsilon_i = \hat\beta_0 \cdot 0 + \hat\beta_1 \cdot 0 = 0$$
Both terms vanish identically! The cross term is strictly zero, establishing the **ANOVA Variance Decomposition Identity**:

$$
TSS = ESS + RSS
$$

- **High-Dimensional Pythagorean Theorem in $\mathbb{R}^n$**: In the centered sample space $\mathbb{R}^n$, the vanishing cross product implies that vectors $(\hat{Y} - \bar{Y}\mathbf{1})$ and $(Y - \hat{Y})$ are strictly orthogonal: $(\hat{Y} - \bar{Y}\mathbf{1}) \perp \hat\varepsilon$. The total deviation vector $Y - \bar{Y}\mathbf{1}$ is the **hypotenuse**, the fitted deviation vector $\hat{Y} - \bar{Y}\mathbf{1}$ is the **adjacent leg** lying in the feature subspace, and the residual vector $\hat\varepsilon = Y - \hat{Y}$ is the **opposite leg** strictly perpendicular to the feature subspace. By the Pythagorean theorem, the squared hypotenuse length identically equals the sum of the squared leg lengths: $\|Y - \bar{Y}\mathbf{1}\|_2^2 = \|\hat{Y} - \bar{Y}\mathbf{1}\|_2^2 + \|\hat\varepsilon\|_2^2$.
- **Squared Cosine of Subspace Angle**:

  $$
  R^2 = \cos^2(\theta)
  $$

  where $\theta$ is the geometric angle between the observation vector and the feature hyperplane. If $Y$ lies entirely in the feature subspace, $\theta = 0^\circ \implies R^2 = 1$ (perfect fit); if $Y$ is perpendicular to the feature subspace, $\theta = 90^\circ \implies R^2 = 0$ (no linear explanatory power).

```anova-variance-demo
```

---

### 3. Multivariate Linear Regression: Matrix Formulation and Normal Equations
Matrix formulation of multivariate linear regression:

$$
Y = X\beta + \varepsilon
$$

where target vector $Y \in \mathbb{R}^{n \times 1}$, design matrix $X \in \mathbb{R}^{n \times k}$ (assuming full column rank $\operatorname{rank}(X) = k \le n$), and parameter vector $\beta \in \mathbb{R}^{k \times 1}$.

OLS minimizes the residual sum of squares:

$$
RSS(\beta) = \|Y - X\beta\|_2^2 = (Y - X\beta)^\top (Y - X\beta) = Y^\top Y - 2\beta^\top X^\top Y + \beta^\top X^\top X \beta
$$

Setting the gradient with respect to $\beta$ to zero:

$$
\nabla_\beta RSS(\beta) = -2 X^\top Y + 2 X^\top X \beta = \mathbf{0}
$$

yields the **Normal Equations**:

$$
X^\top X \hat\beta = X^\top Y
$$

Because $X$ has full column rank, $X^\top X$ is strictly symmetric positive-definite and invertible, giving the unique analytical closed-form solution:

$$
\hat\beta = (X^\top X)^{-1} X^\top Y
$$

- **Orthogonal Projection Shortest Distance Principle**: $Y \in \mathbb{R}^n$ is a point suspended in $n$-dimensional space, while $X\beta$ represents the $k$-dimensional hyperplane $\operatorname{Col}(X)$ ("the floor") spanned by the $k$ column vectors of $X$. To find the point $\hat{Y} = X\hat\beta$ on the floor closest in Euclidean distance to $Y$, the error segment connecting them $Y - \hat{Y}$ must be a vertical drop perpendicular to the floor. The vertical drop is orthogonal to every basis column $X_j$ on the floor ($X_j^\top (Y - X\hat\beta) = 0$). Stacking all column vectors yields the normal equations $X^\top (Y - X\hat\beta) = \mathbf{0}$.

---

### 4. Residual Orthogonality and Projection Operators
Define the fitted vector $\hat{Y} = X\hat\beta$ and the sample residual vector $\hat\varepsilon = Y - \hat{Y} = Y - X\hat\beta$.

- **Hat Matrix $H$ (Orthogonal Projection Operator)**:

  $$
  H = X(X^\top X)^{-1} X^\top
  $$

  - **Geometric Intuition (Vertical Spotlight)**: Projects any vector in space orthogonally onto the feature hyperplane $\operatorname{Col}(X)$, such that $HY = \hat{Y}$. Symmetry and idempotence ($H^2 = H, H^\top = H$) imply that a point already on the floor does not move under repeated projection.
  - **Trace and Geometric Dimension**: $\operatorname{tr}(H) = \operatorname{tr}(X(X^\top X)^{-1} X^\top) = \operatorname{tr}((X^\top X)^{-1} X^\top X) = \operatorname{tr}(I_k) = k$. The geometric dimension (degrees of freedom) of the feature subspace is exactly $k$.

- **Annihilator Matrix $M$ (Residual Projection Operator)**:

  $$
  M = I - H
  $$

  - **Geometric Intuition (Vertical Component Extractor)**: Projects any vector onto the orthogonal complement subspace $\operatorname{Col}(X)^\perp$, filtering out all components parallel to the floor and extracting only the pure vertical residual $MY = \hat\varepsilon$.
  - **Orthogonal Complementarity**: $H + M = I, HM = \mathbf{0}$, and $\operatorname{tr}(M) = n - k$ (the geometric dimension of the orthogonal complement subspace).

#### Five Fundamental Algebraic and Geometric Properties of Residual Orthogonality

1. **Residuals are Orthogonal to Every Regressor ($X^\top \hat\varepsilon = \mathbf{0}$)**:
   Follows directly from the normal equations: $X^\top (Y - X\hat\beta) = \mathbf{0} \implies X^\top \hat\varepsilon = \mathbf{0}$.
   - **Geometric Meaning**: The residual vector $\hat\varepsilon$ is perpendicular to the subspace $\operatorname{Col}(X)$ spanned by all columns of $X$. All linear signals present in the regressors have been completely extracted by $\hat\beta$, leaving zero projection component along $X$ in the residuals.
2. **Mechanical Equilibrium of the Intercept: Residual Sum Vanishes ($\mathbf{1}^\top \hat\varepsilon = 0 \implies \bar{\hat\varepsilon} = 0$)**:
   If the model includes a constant intercept $\beta_0$, the first column of $X$ is the vector of ones $X_0 = \mathbf{1}$.
   Evaluating $X_0^\top \hat\varepsilon = 0$ directly yields:

   $$
   \mathbf{1}^\top \hat\varepsilon = \sum_{i=1}^n \hat\varepsilon_i = 0 \implies \bar{\hat\varepsilon} = \frac{1}{n} \sum_{i=1}^n \hat\varepsilon_i \equiv 0
   $$

   - **Geometric and Physical Meaning**: The vector of ones $\mathbf{1}$ lies within the feature hyperplane; hence, the residual vector perpendicular to this hyperplane must have zero inner product with $\mathbf{1}$. Physically, this corresponds to net torque balance, forcing the regression hyperplane to pass exactly through the sample center of mass $(\bar{X}, \bar{Y})$.
   - **Note**: If the intercept is omitted (regression forced through the origin), $\mathbf{1} \notin \operatorname{Col}(X)$, and the residual sum is generally non-zero.
3. **Residuals are Orthogonal to Fitted Values ($\hat{Y}^\top \hat\varepsilon = 0$)**:
   Because $\hat{Y} = X\hat\beta \in \operatorname{Col}(X)$:

   $$
   \hat{Y}^\top \hat\varepsilon = (X\hat\beta)^\top \hat\varepsilon = \hat\beta^\top (X^\top \hat\varepsilon) = \hat\beta^\top \mathbf{0} = 0
   $$

   - **Geometric Meaning**: The vector lying on the floor (fitted values) and the vertical drop from the ceiling (residuals) are strictly perpendicular in $\mathbb{R}^n$.
4. **High-Dimensional Pythagorean Theorem and Variance Decomposition ($TSS = ESS + RSS$)**:
   From $Y = \hat{Y} + \hat\varepsilon$ and $\hat{Y} \perp \hat\varepsilon$, centering around $\bar{Y}\mathbf{1}$:

   $$
   (Y - \bar{Y}\mathbf{1}) = (\hat{Y} - \bar{Y}\mathbf{1}) + \hat\varepsilon
   $$

   The inner product cross-term vanishes identically:

   $$
   (\hat{Y} - \bar{Y}\mathbf{1})^\top \hat\varepsilon = \hat{Y}^\top \hat\varepsilon - \bar{Y}(\mathbf{1}^\top \hat\varepsilon) = 0 - 0 = 0
   $$

   Yielding the ANOVA identity:

   $$
   \underbrace{\sum_{i=1}^n (Y_i - \bar{Y})^2}_{TSS} = \underbrace{\sum_{i=1}^n (\hat{Y}_i - \bar{Y})^2}_{ESS} + \underbrace{\sum_{i=1}^n \hat\varepsilon_i^2}_{RSS} \implies R^2 = \frac{ESS}{TSS} = 1 - \frac{RSS}{TSS} \in [0, 1]
   $$

5. **Core Distinction: Sample Residual Algebraic Orthogonality vs. Population Error Exogeneity**:
   - **Sample Residual Algebraic Orthogonality ($X^\top \hat\varepsilon = \mathbf{0}$)**: A pure algebraic numerical identity. As long as OLS is computed, the normal equations mechanically enforce orthogonality between residuals and regressors, regardless of whether the true relationship is linear, or whether heteroskedasticity or measurement errors exist.
   - **Population Error Exogeneity ($E(\varepsilon \mid X) = \mathbf{0} \implies E(X^\top \varepsilon) = \mathbf{0}$)**: A structural statistical assumption regarding the data-generating process. It requires that unobserved true disturbances contain no omitted factors correlated with $X$.
   - **Core Implication**: In a model with omitted variable bias (OVB), the computed sample residuals $\hat\varepsilon$ remain **algebraically orthogonal** to the included regressors; however, the unobserved true disturbances $\varepsilon$ are **no longer orthogonal** to the regressors, producing structural endogeneity bias in the parameter estimates.

---

### 5. Deep Connection Between Regression Coefficients and Covariance

#### (1) Univariate Regression: Ratio of Covariance to Regressor Variance

At the finite-sample OLS estimation level, the closed-form analytical solutions are:

$$
\hat\beta_1 = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sum_{i=1}^n (X_i - \bar{X})^2} = \frac{\widehat{\operatorname{Cov}}(X, Y)}{\widehat{\operatorname{Var}}(X)} = \hat\rho_{XY} \frac{s_Y}{s_X}
$$

$$
\hat\beta_0 = \bar{Y} - \hat\beta_1 \bar{X}, \quad R^2 = \hat\rho_{XY}^2
$$

At the theoretical population level (Population Best Linear Predictor, BLP), the structural parameter relationship is:

$$
\beta_1 = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)} = \rho \frac{\sigma_Y}{\sigma_X}
$$

- **Physical Meaning & Dimensional Analysis**: The correlation coefficient $\hat\rho_{XY} \in [-1, 1]$ (or population $\rho$) quantifies pure dimensionless linear association; the standard deviation ratio $\frac{s_Y}{s_X}$ (or population $\frac{\sigma_Y}{\sigma_X}$) provides the physical dimensional conversion factor. The regression slope $\beta_1$ carries exact physical units $[Y]/[X]$, representing the marginal response rate in $Y$ per unit displacement in $X$.
- **Standardized Data**: When features and targets are normalized ($s_X = s_Y = 1$ in sample, or $\sigma_X = \sigma_Y = 1$ in population), the dispersion scale ratio collapses to 1, and the regression slope strictly coincides with the correlation coefficient: $\hat\beta_1 = \hat\rho_{XY}$ (population $\beta_1 = \rho$).

#### (2) Asymmetry of Regression and Regression to the Mean
Examining bidirectional regressions at both the population theoretical level and the finite-sample estimation level:

- **Population Theoretical Relation**:
  $$\beta_{Y \sim X} = \rho \frac{\sigma_Y}{\sigma_X}, \quad \beta_{X \sim Y} = \rho \frac{\sigma_X}{\sigma_Y} \implies \beta_{Y \sim X} \times \beta_{X \sim Y} = \rho^2 \le 1$$
- **Finite-Sample Estimation Relation**:
  $$\hat\beta_{Y \sim X} = \hat\rho_{XY} \frac{s_Y}{s_X}, \quad \hat\beta_{X \sim Y} = \hat\rho_{XY} \frac{s_X}{s_Y} \implies \hat\beta_{Y \sim X} \times \hat\beta_{X \sim Y} = \hat\rho_{XY}^2 \le 1$$

- **Physical Intuition (Noise Dilution Effect & Mean Contraction)**: The reverse regression slope is never the algebraic reciprocal of the forward slope; rather, $\beta_{X \sim Y} = \frac{\rho^2}{\beta_{Y \sim X}} < \frac{1}{\beta_{Y \sim X}}$ whenever noise exists ($|\rho| < 1$). Random fluctuations dilute deterministic signals; predicting either variable from the other systematically contracts extreme values toward the unconditional center mean (e.g., tall parents' children tend to be shorter, extreme return assets tend to revert to the cross-sectional mean).

#### (3) Multivariate Regression Covariance Formulation: Linear Whitening Decorrelation

Centering both regressors and target:

$$
\hat\beta = (X^\top X)^{-1} X^\top Y = \hat{\boldsymbol{\Sigma}}_{XX}^{-1} \hat{\boldsymbol{\Sigma}}_{XY}
$$

- **Physical Intuition (Whitening Decorrelation Filter)**: If features are mutually uncorrelated ($\hat{\boldsymbol{\Sigma}}_{XX}$ is diagonal), multivariate regression decouples into independent univariate regressions. When features are correlated, $\hat{\boldsymbol{\Sigma}}_{XX}^{-1}$ serves as a **linear decorrelation (whitening) operator**, eliminating indirect co-movement pathways and isolating each feature's net marginal contribution.

#### (4) Partial Covariance and the Frisch–Waugh–Lovell (FWL) Theorem

##### 1. Core Theoretical Motivation (Why Do We Need Partial Covariance & the FWL Theorem?)
In empirical statistics, quantitative finance, and causal inference, partial covariance and the FWL theorem were developed to resolve four fundamental pain points:
1. **Pain Point 1: Confounder Collinearity and the Mathematical Realization of "Ceteris Paribus" (All Else Equal)**:
   Observed features in real-world data are almost never orthogonal (e.g., tenure and education, market capitalization and turnover are deeply intertwined). In a simple univariate regression $Y \sim X_j$, the ordinary covariance $\operatorname{Cov}(X_j, Y)$ conflates $X_j$'s direct marginal contribution with all indirect, channeled influences from correlated confounders (omitted variable bias, OVB).
   - *Motivation*: Formulate a rigorous algebraic mechanism to purge the confounding influences of all other features $X_{-j}$, quantifying the pure, ceteris paribus marginal impact of $X_j$ on $Y$.
2. **Pain Point 2: Computational Inversion Bottlenecks (Frisch & Waugh's 1933 Historical Dilemma)**:
   In 1933, econometric pioneers Ragnar Frisch and Frederick Waugh analyzed agricultural supply-demand series with extensive polynomial time trends and seasonal dummies. In the pre-computer era, inverting an $(X^\top X)$ matrix with dozens of control columns was a prohibitive $O(p^3)$ manual chore.
   - *Motivation*: Is it possible to avoid inverting the massive full design matrix and isolate the parameter of primary interest (e.g., price elasticity) by stripping out nuisance trends and fixed effects? The FWL theorem proved that residualizing both target and primary regressor against nuisance variables yields the identical coefficient and standard error in a univariate regression.
3. **Pain Point 3: High-Dimensional Fixed Effects Absorption (Panel & Clustered Data)**:
   Microeconometric panel datasets and cross-sectional equity universes often feature thousands of entity fixed effects (e.g., 5,000 equity dummy columns). Direct inversion crashes memory and compute.
   - *Motivation*: The FWL projection operator $M_{\text{FE}} = I - D(D^\top D)^{-1}D^\top$ proves that high-dimensional fixed effects are mathematically equivalent to simple within-group demeaning (the Within Transformation), absorbing thousands of nuisance parameters without explicitly inverting them.
4. **Pain Point 4: Foundation of Modern Causal Inference and Double/Debiased Machine Learning (DML)**:
   Chernozhukov et al. (2018) established Double Machine Learning by generalizing FWL orthogonalization from linear subspaces to arbitrary non-parametric ML models (residualizing treatment $D$ and outcome $Y$ via complex ML on high-dimensional confounders $W$, followed by Neyman-orthogonal regression to eliminate regularization bias and recover $\sqrt{N}$ asymptotic normality).

##### 2. Theorem Statement and Formal Definition of Partial Covariance
Partition the design matrix as $X = [X_1, X_2]$ in the linear model $Y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon$:
- Define the **orthogonal projection matrix** onto $\operatorname{Col}(X_1)$ as $P_1 = X_1 (X_1^\top X_1)^{-1} X_1^\top$;
- Define the **annihilator matrix (orthogonal complement projector)** as $M_1 = I - P_1$ (symmetric and idempotent: $M_1^\top = M_1$, $M_1^2 = M_1$, $M_1 X_1 = \mathbf{0}$).

Applying $M_1$ to both $Y$ and $X_2$ isolates the **purified orthogonal increment residuals**:
$$\tilde{Y} = M_1 Y, \quad \tilde{X}_2 = M_1 X_2$$

The **Frisch–Waugh–Lovell (FWL) Theorem** establishes that the multivariate OLS estimator $\hat\beta_2$ is strictly identical to the univariate regression of residual $\tilde{Y}$ on residual $\tilde{X}_2$:

$$
\hat\beta_2 = (X_2^\top M_1 X_2)^{-1} X_2^\top M_1 Y = \frac{\widehat{\operatorname{Cov}}(\tilde{X}_2, \tilde{Y})}{\widehat{\operatorname{Var}}(\tilde{X}_2)} = \frac{\widehat{\operatorname{Cov}}(\tilde{X}_2, Y)}{\widehat{\operatorname{Var}}(\tilde{X}_2)}
$$

- **Partial Covariance**: $\widehat{\operatorname{Cov}}(\tilde{X}_2, \tilde{Y})$ defines the sample partial covariance between $X_2$ and $Y$ conditional on $X_1$. It captures the pure co-movement between the remaining unexplained components after exhausting $X_1$'s linear predictive power.
- **One-Sided Projection Invariance**: By the symmetry and idempotence of $M_1$:
  $$\tilde{X}_2^\top \tilde{Y} = (M_1 X_2)^\top (M_1 Y) = X_2^\top M_1^2 Y = X_2^\top M_1 Y = \tilde{X}_2^\top Y$$
  The numerator inner product is identical whether or not $Y$ is pre-projected; however, the denominator regressor **must be strictly orthogonalized** ($\|\tilde{X}_2\|_2^2 \ne \|X_2\|_2^2$).

##### 3. Three Core Practical Utilities of the FWL Theorem
1. **High-Dimensional Regression Visualization: Added Variable Plots (Partial Residual Plots)**:
   When $p > 2$, multivariate regression cannot be directly visualized. Using FWL, plot $(\tilde{X}_{j, i}, \tilde{Y}_i)$ on a 2D scatter plot:
   - The univariate OLS slope of this scatter plot **identically equals the multivariate partial regression coefficient $\hat\beta_j$**;
   - Non-linear curvature, heteroskedastic fan shapes, and influential high-leverage outliers conditional on all other covariates become immediately apparent.
2. **Geometric Origin of Multicollinearity: Variance Inflation Factor (VIF)**:
   Expanding the sampling variance of the FWL estimator:
   $$
   \operatorname{Var}(\hat\beta_j \mid X) = \frac{\sigma^2}{\|\tilde{X}_j\|_2^2} = \frac{\sigma^2}{(n-1)\operatorname{Var}(X_j)} \cdot \underbrace{\frac{1}{1 - R_{j \mid -j}^2}}_{\mathrm{VIF}_j}
   $$
   - **Geometric Intuition (Collapsed Lever Arm Amplifies Jitter)**: $1 - R_{j \mid -j}^2 = \sin^2(\theta_j)$, where $\theta_j$ is the angle between $X_j$ and the hyperplane spanned by remaining features. Under extreme multicollinearity ($R_{j \mid -j}^2 \to 1$), $\theta_j \to 0$, shrinking the orthogonal lever arm $\tilde{X}_j$ toward zero. Balancing output responses with an infinitesimal lever arm violently amplifies minor perturbations, driving parameter estimation variance to infinity.
3. **Two-Sided Factor Neutralization in Quantitative Alpha Modeling**:
   When evaluating whether candidate alpha signal $X_2$ delivers incremental return beyond benchmark risk factors $X_1$ (e.g., industry, size):
   - **Correct Method (FWL Stream)**: Both asset return $Y$ and candidate factor $X_2$ **must be simultaneously orthogonalized against $X_1$** (two-sided neutralization);
   - **The Single-Sided Trap**: If one only neutralizes returns $Y$ to obtain residual $\varepsilon$ but regresses directly onto raw factor $X_2$, the measured factor return is systematically compressed by $(1 - \rho^2)$ ($\frac{\beta_{\text{naive}}}{\beta_{\text{FWL}}} = 1 - \rho^2$), severely penalizing and misdiagnosing promising alpha signals!

```fwl-geometry-demo
```

#### (5) Canonical Quantitative Finance Mappings
1. **CAPM Asset Beta**: $\beta_i = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)}$;
2. **Variance-Minimizing Optimal Hedge Ratio**: $\min_h \operatorname{Var}(\Delta S - h\Delta F) \implies h^* = \frac{\operatorname{Cov}(\Delta S, \Delta F)}{\operatorname{Var}(\Delta F)} \equiv \beta_{\Delta S \sim \Delta F}$;
3. **Omitted Variable Bias (OVB)**: If the true model is $Y = \beta_1 X_1 + \beta_2 X_2 + \varepsilon$, omitting $X_2$ in a short regression yields $E(\hat\beta_1^{\text{short}} \mid X) = \beta_1 + \beta_2 \frac{\operatorname{Cov}(X_1, X_2)}{\operatorname{Var}(X_1)}$;
4. **Barra Factor Risk Neutralization**: $F_{\text{raw}} = X_{\text{risk}} \gamma + F_{\text{neutral}}$, using orthogonal projection $F_{\text{neutral}} \perp X_{\text{risk}}$ to neutralize industry and style risk exposures.

---

## Module 2: Gauss–Markov Theorem, Statistical Inference, and Core Analytical Lemmas

This module systematically details the Gauss–Markov assumptions, inference distributions, and mathematical derivations alongside physical and geometric intuitions for the 7 core lemmas.

---

### 1. Gauss–Markov Assumptions and the Essence of BLUE
The Gauss–Markov theorem states that under standard linear model assumptions, the OLS estimator is the **Best Linear Unbiased Estimator (BLUE)**—meaning that among all linear unbiased estimators, OLS achieves the minimum variance in the positive semi-definite ordering.

1. **Linearity in Parameters**: The model takes the form $Y = X\beta + \varepsilon$;
2. **Strict Exogeneity**: $E(\varepsilon \mid X) = \mathbf{0}$;
3. **Spherical Errors**:
   - **Homoskedasticity**: $\operatorname{Var}(\varepsilon_i \mid X) = \sigma^2$;
   - **No Autocorrelation**: $\operatorname{Cov}(\varepsilon_i, \varepsilon_j \mid X) = 0 \quad (i \ne j)$;
   - Unified matrix form: $\operatorname{Var}(\varepsilon \mid X) = \sigma^2 I_n$;
4. **No Full Multicollinearity**: $\operatorname{rank}(X) = k \le n$.

- **Boundary of Normality**:
  OLS being BLUE **does not require the error terms to follow a normal distribution**. The theorem depends solely on first-moment (exogeneity) and second-moment (spherical errors) conditions. Normality is required only for **exact finite-sample $t$-tests and $F$-tests**, and for proving that OLS achieves the Cramér–Rao lower bound (making it the Uniformly Minimum-Variance Unbiased Estimator, UMVUE).

---

### 2. Core Analytical Lemmas

#### [Lemma 1] Algebraic and Moment Properties of OLS
- **Linear Form**: $\hat\beta = (X^\top X)^{-1} X^\top Y = C Y$, where the linear weight matrix $C = (X^\top X)^{-1} X^\top$ satisfies $C X = I_k$.
- **Conditional Unbiasedness**:

  $$
  E(\hat\beta \mid X) = E(C(X\beta + \varepsilon) \mid X) = \beta + C E(\varepsilon \mid X) = \beta
  $$

- **Conditional Covariance Matrix**:

  $$
  \operatorname{Var}(\hat\beta \mid X) = \operatorname{Var}(CY \mid X) = C \operatorname{Var}(\varepsilon \mid X) C^\top = C (\sigma^2 I_n) C^\top = \sigma^2 (X^\top X)^{-1}
  $$

  - Single-coefficient variance: $\operatorname{Var}(\hat\beta_j \mid X) = \sigma^2 [(X^\top X)^{-1}]_{jj}$;
  - Pairwise covariance: $\operatorname{Cov}(\hat\beta_j, \hat\beta_m \mid X) = \sigma^2 [(X^\top X)^{-1}]_{jm}$.
- **Unbiased Residual Variance Estimator and Quadratic Form Expectation Lemma**:

  $$
  \hat\sigma^2 = \frac{\hat\varepsilon^\top \hat\varepsilon}{n - k} = \frac{\sum_{i=1}^n \hat\varepsilon_i^2}{n - k}
  $$

  - **Geometric and Physical Intuition (Orthogonal Complement Capacity)**: Residuals are $\hat\varepsilon = (I - H)\varepsilon$. The original $n$-dimensional observation space loses $k$ degrees of freedom to the fitted hyperplane, confining residuals to freely fluctuate within the $(n - k)$-dimensional orthogonal complement subspace. By the quadratic form expectation identity:

    $$
    E(\hat\varepsilon^\top \hat\varepsilon \mid X) = E(\varepsilon^\top (I - H) \varepsilon \mid X) = \operatorname{tr}\left( (I - H) \sigma^2 I_n \right) = \sigma^2 \operatorname{tr}(I - H) = \sigma^2 (n - k)
    $$

    Dividing by $n - k$ normalizes by the geometric capacity of the orthogonal complement subspace, yielding an unbiased estimator of the true physical noise variance $\sigma^2$.

#### [Lemma 2] Distributional Properties under Normality
When error terms satisfy spherical Gaussianity $\varepsilon \mid X \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_n)$:
- **Statistical Independence (Cochran's Subspace Decoupling)**:

  $$
  \hat\beta \text{ and the sample residuals } \hat\varepsilon \text{ (as well as } \hat\sigma^2 \text{) are strictly statistically independent!}
  $$

  - **Geometric and Physical Intuition**: Under spherical Gaussian measure, geometrically orthogonal subspaces are statistically independent. Because $\hat\beta$ depends solely on the projection $\hat{Y} \in \operatorname{Col}(X)$, while $\hat\varepsilon \in \operatorname{Col}(X)^\perp$ resides in the orthogonal complement subspace, their cross-covariance vanishes identically:

    $$
    \operatorname{Cov}(\hat\beta, \hat\varepsilon \mid X) = \sigma^2 C (I - H)^\top = \sigma^2 \left( (X^\top X)^{-1}X^\top - (X^\top X)^{-1}X^\top H \right) = \mathbf{0}
    $$

    In a multivariate Gaussian distribution, zero covariance implies strict independence; hence, parameter estimates and residual fluctuations are statistically decoupled.
- **Chi-Square Distribution of Residual Sum of Squares**:

  $$
  \frac{\hat\varepsilon^\top \hat\varepsilon}{\sigma^2} = \frac{(n - k)\hat\sigma^2}{\sigma^2} \sim \chi^2(n - k)
  $$

- **Single-Coefficient $t$-Test Statistic**:
  Testing $H_0: \beta_j = \beta_{j,0}$:

  $$
  t = \frac{\hat\beta_j - \beta_{j,0}}{\sqrt{\hat\sigma^2 [(X^\top X)^{-1}]_{jj}}} \sim t_{n-k}
  $$

- **Multiple Linear Restrictions $F$-Test Statistic**:
  Testing $q$ joint linear restrictions $H_0: R\beta = r$ ($R \in \mathbb{R}^{q \times k}$ with full row rank):

  $$
  F = \frac{(R\hat\beta - r)^\top [R(X^\top X)^{-1} R^\top]^{-1} (R\hat\beta - r) / q}{\hat\sigma^2} \sim F_{q, n-k}
  $$

  - **Restricted vs. Unrestricted Residual Sum of Squares Form**:

    $$
    F = \frac{(RSS_R - RSS_{UR}) / q}{RSS_{UR} / (n - k)} = \frac{(R_{UR}^2 - R_R^2) / q}{(1 - R_{UR}^2) / (n - k)} \sim F_{q, n-k}
    $$

  - **Single Restriction Equivalence**: When $q = 1$, $t^2 \equiv F$.

#### [Lemma 3] Prediction Interval vs. Confidence Interval
For a new query feature point $X_0 \in \mathbb{R}^k$:
- **Confidence Interval for Conditional Mean Response ($E(Y_0 \mid X_0) = X_0^\top \beta$)**:
  Point estimate $\hat{Y}_0 = X_0^\top \hat\beta$, with variance arising purely from parameter sampling error:

  $$
  \operatorname{Var}(\hat{Y}_0 \mid X) = X_0^\top \operatorname{Var}(\hat\beta \mid X) X_0 = \sigma^2 X_0^\top (X^\top X)^{-1} X_0
  $$

  $1-\alpha$ Confidence Interval:

  $$
  \hat{Y}_0 \pm t_{n-k, 1-\alpha/2} \cdot \sqrt{\hat\sigma^2 X_0^\top (X^\top X)^{-1} X_0}
  $$

- **Prediction Interval for an Individual Observation ($Y_0 = X_0^\top \beta + \varepsilon_0$)**:
  Prediction error is $e_0 = Y_0 - \hat{Y}_0 = \varepsilon_0 - X_0^\top(\hat\beta - \beta)$. Because the future disturbance $\varepsilon_0$ is independent of past data:

  $$
  \operatorname{Var}(e_0 \mid X) = \operatorname{Var}(\varepsilon_0) + \operatorname{Var}(\hat{Y}_0 \mid X) = \sigma^2 \left[ 1 + X_0^\top (X^\top X)^{-1} X_0 \right]
  $$

  $1-\alpha$ Prediction Interval:

  $$
  \hat{Y}_0 \pm t_{n-k, 1-\alpha/2} \cdot \sqrt{\hat\sigma^2 \left[ 1 + X_0^\top (X^\top X)^{-1} X_0 \right]}
  $$

- **Physical Distinction (Population Centroid vs. Individual Particle)**: The confidence interval quantifies the positional oscillation of the population mean hyperplane; as sample size $n \to \infty$, estimation variance collapses to 0. The prediction interval addresses a single incoming particle carrying irreducible thermal white noise $\varepsilon_0 \sim \mathcal{N}(0, \sigma^2)$. Consequently, the prediction interval variance always exceeds that of the confidence interval by the irreducible $\sigma^2$ baseline; even as $n \to \infty$, its width is bounded below by $\pm z_{\alpha/2}\sigma$.

#### [Lemma 4] Leave-One-Out Cross-Validation and Leverage
- **Hat Matrix Diagonal (Leverage $H_{ii}$)**:

  $$
  H_{ii} = X_i^\top (X^\top X)^{-1} X_i
  $$

  satisfying $0 \le H_{ii} \le 1$, $\sum_{i=1}^n H_{ii} = k$, and average leverage $\bar{H} = k/n$.
  - **Geometric Intuition (Archimedean Lever Arm)**: $H_{ii}$ measures the Mahalanobis distance of sample point $X_i$ from the centroid in feature space. Extreme outliers possess long leverage arms ($H_{ii} \to 1$), capable of single-handedly tilting the entire regression plane.
- **Leave-One-Out Residual Shortcut (Sherman–Morrison Formula)**:
  Without retraining $n$ separate times, the out-of-sample error when leaving out sample $i$ is obtained via exact closed-form scaling:

  $$
  \hat\varepsilon_{(-i)} = Y_i - \hat{Y}_{(-i)} = \frac{\hat\varepsilon_i}{1 - H_{ii}}
  $$

  yielding the exact Leave-One-Out Cross-Validation (LOOCV) error:

  $$
  \mathrm{LOOCV} = \frac{1}{n} \sum_{i=1}^n \left( \frac{\hat\varepsilon_i}{1 - H_{ii}} \right)^2
  $$

  - **Physical Intuition (Elastic Rebound Release)**: The sample point uses its leverage to pull the regression plane toward itself (artificially deflating the in-sample residual $\hat\varepsilon_i$). Removing the point instantly releases the elastic strain, causing the regression plane to rebound away; the true out-of-sample error is inflated precisely by the factor $\frac{1}{1 - H_{ii}}$.

#### [Lemma 5] Omitted Variable Bias and Irrelevant Regressors
- **Omitted Variable Bias (OVB)**:
  If the true data-generating process is $Y = X_1 \beta_1 + X_2 \beta_2 + \varepsilon$, omitting $X_2$ in a short regression on $X_1$ yields:

  $$
  E(\hat\beta_1^{\text{short}} \mid X) = \beta_1 + \underbrace{(X_1^\top X_1)^{-1} X_1^\top X_2}_{\hat\Gamma_{2 \sim 1}} \beta_2
  $$

  - **Geometric Intuition (Projection Shadow Contamination)**: If the omitted $X_2$ is non-orthogonal to the included $X_1$ ($X_1^\top X_2 \ne \mathbf{0}$), the true physical force of $X_2$ on $Y$ casts a causal projection shadow onto $X_1$. The short regression cannot identify the origin of the shadow, attributing the projected component to $\beta_1$. Only if $\beta_2 = \mathbf{0}$ (omitted feature has zero true effect) or $X_1 \perp X_2$ (shadow projection is zero) is the short regression unbiased.
- **Including Irrelevant Regressors (Overfitting)**:
  If the true model does not contain $X_2$ ($\beta_2 = \mathbf{0}$), but $X_2$ is erroneously included:
  - The parameter estimate remains unbiased: $E(\hat\beta_1^{\text{long}}) = \beta_1$;
  - But variance strictly inflates: $\operatorname{Var}(\hat\beta_1^{\text{long}}) \ge \operatorname{Var}(\hat\beta_1^{\text{short}})$, with equality holding if and only if $X_1 \perp X_2$.

#### [Lemma 6] Measurement Error (Errors-in-Variables / Attenuation Bias)
- **Regressor Measurement Error (Attenuation Bias)**:
  True model $Y_i = \beta_0 + \beta_1 X_i^* + \varepsilon_i$, observed with additive white noise $X_i = X_i^* + u_i$ ($u_i \sim (0, \sigma_u^2)$ independent of $X_i^*, \varepsilon_i$):

  $$
  \operatorname{plim}_{n \to \infty} \hat\beta_1 = \beta_1 \cdot \frac{\sigma_{X^*}^2}{\sigma_{X^*}^2 + \sigma_u^2} < \beta_1
  $$

  - **Physical Intuition (Signal-to-Noise Dilution)**: Noise in the regressor diffuses data points horizontally along the $x$-axis, flattening the slope and systematically shrinking the estimated coefficient toward zero.
- **Target Measurement Error**:
  If the target has measurement noise $Y_i = Y_i^* + v_i$ ($v_i$ independent of $X_i$), $\hat\beta_1$ **remains unbiased and consistent**; the measurement noise simply folds into the residual variance $\sigma^2 + \sigma_v^2$, increasing standard errors and reducing test power.

#### [Lemma 7] Scale and Affine Invariance
- **Regressor Scaling**: If $X_{\text{new}} = c \cdot X$, then $\hat\beta_{\text{new}} = \frac{1}{c} \hat\beta$;
- **Target Scaling**: If $Y_{\text{new}} = d \cdot Y$, then $\hat\beta_{\text{new}} = d \cdot \hat\beta$;
- **Shifting / Centering**: Adding constants to $X$ or $Y$ leaves the slope $\hat\beta_1$ **strictly invariant**, shifting only the intercept $\hat\beta_0$;
- **Invariance Law**: Under non-zero affine scaling and shifts, **$t$-statistics, $F$-statistics, $R^2$, and regression $p$-values remain strictly unchanged**.
  - **Geometric Intuition (Euclidean Conformal Preservation)**: Changing units of measurement alters coordinate axes scaling without changing the spatial angle $\theta$ between high-dimensional vectors. All statistics derived from the angle cosine ($R^2 = \cos^2\theta$, $t \propto \cot\theta$) are dimensionless geometric invariants.

---

### 3. Violations of Assumptions and Remedies (White / Newey–West / GLS)
When empirical data violates Gauss–Markov conditions:
- **Heteroskedasticity and Autocorrelation**:
  The OLS estimator **remains unbiased and consistent**, but is no longer BLUE. The standard OLS covariance $\sigma^2 (X^\top X)^{-1}$ is severely underestimated, creating spurious statistical significance.
- **Remedies**:
  1. **White Heteroskedasticity-Consistent Standard Errors (HC0 / Sandwich Estimator)**:

     $$
     \operatorname{Var}_{\text{White}}(\hat\beta) = (X^\top X)^{-1} \left( \sum_{i=1}^n \hat\varepsilon_i^2 X_i X_i^\top \right) (X^\top X)^{-1}
     $$

     - **Physical Intuition (Sandwich Structure)**: The outer "two slices of bread" $(X^\top X)^{-1}$ handle coordinate basis projection, while the middle "meat" $X^\top \hat{\Omega} X = \sum_{i=1}^n \hat\varepsilon_i^2 X_i X_i^\top$ captures empirical point-by-point local heteroskedastic energy.
  2. **Newey–West Heteroskedasticity and Autocorrelation Consistent (HAC)**:
     Incorporates a Bartlett triangular lag kernel to correct for serial autocorrelation, standard for financial time series.
  3. **Generalized Least Squares (GLS / WLS, Aitken's Theorem)**:
     When error covariance $\operatorname{Var}(\varepsilon \mid X) = \sigma^2 \boldsymbol{\Omega}$ is known, pre-multiplying by the whitening matrix $P = \boldsymbol{\Omega}^{-1/2}$ transforms the system:

     $$
     \hat\beta_{\text{GLS}} = (X^\top \boldsymbol{\Omega}^{-1} X)^{-1} X^\top \boldsymbol{\Omega}^{-1} Y
     $$

     - **Geometric Intuition (Spatial Whitening and Mahalanobis Metric)**: When disturbances form a tilted or elongated ellipsoid in space, standard Euclidean distance fails. Pre-multiplying by $\boldsymbol{\Omega}^{-1/2}$ rotates and compresses the ellipsoid into a standard sphere, in which standard OLS orthogonal projection recovers BLUE optimality.

---

## Module 3: Variable Selection and Shrinkage (ESL 3.3–3.4)

When faced with numerous potentially collinear predictors, we must restrict or regularize the model.

### 1. Traditional Methods (Subset Selection)
- **Best Subset / Forward & Backward Stepwise**: At a high level, these discrete selection procedures can find good subsets of variables, but the discrete nature of the selection process often leads to high variance.

### 2. Ridge Regression
Introduces an $\ell_2$ norm penalty to shrink coefficients:

$$
\hat\beta^{\mathrm{ridge}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2
$$

The closed-form solution is:

$$
\hat\beta^{\mathrm{ridge}} = (X^\top X + \lambda I)^{-1}X^\top y
$$

**Key Traits**: Excellent at handling multicollinearity (trades a little bias for a large reduction in variance). However, **it does not shrink any coefficient exactly to zero**.

#### Closed-Form Derivation via Data Augmentation: Reducing Ridge to Standard OLS

In quantitative finance and statistical learning theory, an exceptionally elegant and practical algebraic formulation is: **without taking matrix derivatives of the penalized objective, one can reduce Ridge regression entirely to standard Ordinary Least Squares (OLS) simply by appending an identity matrix to the design matrix (Data Augmentation)**.

##### 1. Augmented System Formulation

Notice that the $\ell_2$ regularization penalty can be rewritten as a sum of squared residuals with a target of zero and an identity predictor matrix:

$$
\lambda \|\beta\|_2^2 = \|\mathbf{0}_{p \times 1} - \sqrt{\lambda} I_p \beta\|_2^2
$$

We stack the original design matrix $X \in \mathbb{R}^{N \times p}$ and response vector $y \in \mathbb{R}^{N \times 1}$ vertically with pseudo-data, constructing the augmented design matrix $\tilde{X}$ and augmented response vector $\tilde{y}$:

$$
\tilde{X} = \begin{bmatrix} X \\ \sqrt{\lambda} I_p \end{bmatrix} \in \mathbb{R}^{(N + p) \times p}, \quad \tilde{y} = \begin{bmatrix} y \\ \mathbf{0}_{p \times 1} \end{bmatrix} \in \mathbb{R}^{(N + p) \times 1}
$$

##### 2. Mathematical Equivalence Proof

Formulate the standard unpenalized OLS residual sum of squares loss on the augmented dataset $(\tilde{X}, \tilde{y})$:

$$
\begin{aligned}
\mathcal{L}_{\text{OLS}}(\beta; \tilde{X}, \tilde{y}) &= \|\tilde{y} - \tilde{X}\beta\|_2^2 \\
&= \left\| \begin{bmatrix} y \\ \mathbf{0} \end{bmatrix} - \begin{bmatrix} X \\ \sqrt{\lambda} I_p \end{bmatrix} \beta \right\|_2^2 \\
&= \left\| \begin{bmatrix} y - X\beta \\ -\sqrt{\lambda} I_p \beta \end{bmatrix} \right\|_2^2 \\
&= \|y - X\beta\|_2^2 + \|-\sqrt{\lambda} I_p \beta\|_2^2 \\
&= \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2
\end{aligned}
$$

This objective function is **algebraically identical to the Ridge regression loss**!

##### 3. Deriving the Closed-Form Solution via OLS Normal Equations

Because the augmented problem is a standard unconstrained OLS regression, its optimal solution is immediately governed by the classical OLS Normal Equations:

$$
\hat\beta^{\mathrm{ridge}} = (\tilde{X}^\top \tilde{X})^{-1} \tilde{X}^\top \tilde{y}
$$

Evaluating the block matrix products:

$$
\tilde{X}^\top \tilde{X} = \begin{bmatrix} X^\top & \sqrt{\lambda} I_p \end{bmatrix} \begin{bmatrix} X \\ \sqrt{\lambda} I_p \end{bmatrix} = X^\top X + (\sqrt{\lambda} I_p)(\sqrt{\lambda} I_p) = X^\top X + \lambda I_p
$$

$$
\tilde{X}^\top \tilde{y} = \begin{bmatrix} X^\top & \sqrt{\lambda} I_p \end{bmatrix} \begin{bmatrix} y \\ \mathbf{0} \end{bmatrix} = X^\top y + \sqrt{\lambda} I_p \mathbf{0} = X^\top y
$$

Substituting these blocks directly recovers the explicit Ridge closed-form estimator:

$$
\hat\beta^{\mathrm{ridge}} = (X^\top X + \lambda I_p)^{-1} X^\top y
$$

##### 4. Geometric, Theoretical, and Engineering Insights

- **Physical & Geometric Intuition (Virtual Observations Pulling to Zero)**:
  Data augmentation corresponds to adding $p$ **virtual single-variable probing experiments** to the real dataset. The $j$-th virtual observation has features $x_{\text{pseudo}, j} = \sqrt{\lambda} \mathbf{e}_j$ (where only the $j$-th predictor equals $\sqrt{\lambda}$ and all others are zero) and observed response $y_{\text{pseudo}, j} = 0$. These $p$ anchor points penalize any coefficient $\beta_j$ that strays from zero, smoothly shrinking all estimates toward the origin;
- **Guaranteed Full Rank & Invertibility (Even When $N < p$)**:
  When $N < p$ (high-dimensional regimes) or features are collinear, $\text{rank}(X) \le N < p$, rendering $X^\top X$ singular with infinitely many OLS solutions. In the augmented matrix $\tilde{X}$, appending $\sqrt{\lambda} I_p$ guarantees $p$ linearly independent rows, enforcing $\text{rank}(\tilde{X}) = p$.
  For any non-zero vector $v \ne \mathbf{0}$:

  $$
  v^\top (X^\top X + \lambda I_p) v = \|Xv\|_2^2 + \lambda \|v\|_2^2 \ge \lambda \|v\|_2^2 > 0 \quad (\forall \lambda > 0)
  $$

  Thus, $X^\top X + \lambda I_p$ is strictly symmetric positive-definite (SPD), ensuring the unique existence of the closed-form inverse;
- **Numerical Stability via QR Decomposition (Avoiding Matrix Squaring)**:
  Forming the normal matrix $X^\top X + \lambda I$ directly squares the condition number ($\kappa(X^\top X + \lambda I) \approx \kappa(\tilde{X})^2$), accelerating floating-point roundoff errors. Using data augmentation, production linear solvers compute the **Thin QR Decomposition** of the $(N+p) \times p$ augmented matrix $\tilde{X} = \tilde{Q} \tilde{R}$ and solve the triangular system $\tilde{R} \beta = \tilde{Q}^\top \tilde{y}$ via backward substitution. This maintains condition number $\kappa(\tilde{X})$ and completely avoids explicit matrix inversion;
- **Bayesian Fictitious Data Duality**:
  In Bayesian linear regression with a Gaussian prior $\beta \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$ and likelihood $\mathcal{N}(X\beta, \sigma^2 I)$, the Maximum A Posteriori (MAP) estimate is mathematically identical to Ridge with $\lambda = \sigma^2 / \tau^2$. Data augmentation reveals that **a Gaussian parameter prior is mathematically indistinguishable from observing $p$ fictitious data points centered at zero with precision scaled by the prior variance**.



### 3. Lasso Regression
Introduces an $\ell_1$ norm penalty:
$$
\hat\beta^{\mathrm{lasso}} = \arg\min_\beta \|y - X\beta\|_2^2 + \lambda \|\beta\|_1
$$
**Key Traits**: The geometric shape of the $\ell_1$ penalty is a sharp "diamond". The elliptical contours of the sum of squares are highly likely to hit the corners of the diamond on the axes, allowing the Lasso to shrink some coefficients **exactly to zero**, effectively performing **built-in variable selection (sparsity)**.

### 4. Method Comparison Matrix

| Method | Penalty | Bias vs. Variance | Produces Sparsity? | Handles Multicollinearity? |
| :--- | :--- | :--- | :---: | :--- |
| **Best Subset** | Restricts # of variables | Discrete, high variance | Yes | Depends on the subset kept |
| **Ridge** | $\lambda \|\beta\|_2^2$ (Sphere) | Increases bias, lowers var | No | Yes, beautifully; unique solution |
| **Lasso** | $\lambda \|\beta\|_1$ (Diamond) | Increases bias, lowers var | Yes | Yes, but picks randomly among highly correlated features |

*(Note: Dimensionality reduction techniques like PCR/PLS essentially create "derived directions" to regress on, differing from penalized regression frameworks and omitted here for conciseness.)*

---

## Module 4: Kernel Smoothing & Local Regression (ESL 6.1–6.3)

In the preceding three modules, we thoroughly examined linear regression (OLS, Ridge, Lasso). All these classical models rest upon a **Global Parametric Assumption**: namely, that the true underlying data-generating function satisfies $f(X) = X\beta$ globally across the entire input domain. However, across modern quantitative finance—such as fitting option implied volatility smiles/surfaces, capturing nonlinear price impact curves from high-frequency order flow imbalance, or mining localized alpha signals—true relationships are inherently curved, regime-dependent, or state-contingent.

When we seek to discard rigid global linear assumptions, we arrive at the intersection of classical statistics and machine learning: **Nonparametric Smoothing**. Following the theoretical architecture of ESL Chapter 6, this module begins from first principles with conditional expectation, demonstrates how "kernels" naturally emerge as the bridge connecting density estimation to regression, and rigorously derives the mechanics of local polynomial regression.

---

### 1. Theoretical Foundations: The Regression Objective, The Essence of "Kernel", and Bridging Two Paradigms

#### (1) The Statistical Essence of Regression: The Conditional Expectation Function
In probability and statistics, the ultimate objective of regression is finding a predictor function $f(X)$ that minimizes the expected mean squared prediction error $\mathbb{E}[(Y - f(X))^2]$. By the law of total expectation and the orthogonal projection theorem in $L^2$ probability space, the unique theoretical minimizer is the **conditional expectation function (regression function)**:
$$
f(x_0) = \mathbb{E}[Y \mid X = x_0] = \int y \, p(y \mid x_0) \, dy = \frac{\int y \, p(x_0, y) \, dy}{p(x_0)}
$$
- **Global Parametric School (Modules 1–3)**: Imposes a rigid global functional assumption $f(x) \approx x^\top \beta$, estimating a single fixed set of parameters $\hat\beta$ across all samples. Its strengths are low estimation variance and high computational efficiency, but it suffers from severe **model misspecification bias**.
- **Nonparametric Local School (This Module)**: Presumes no global parametric structure on $f(x)$. Instead, it adheres to **memory-based learning (lazy learning)**—"to predict at query point $x_0$, inspect only the observations in the local neighborhood of $x_0$".

#### (2) From Conditional Expectation to Nadaraya–Watson: Natural Plug-in via Kernel Density Estimation
Since conditional expectation is the ratio between the integrated joint density and the marginal density, statisticians Nadaraya (1964) and Watson (1964) proposed an elegant, foundational breakthrough: **can we directly estimate both the numerator and denominator using nonparametric Parzen window Kernel Density Estimation (KDE)?**

Let the kernel function with bandwidth $\lambda$ be $K_\lambda(x_0, x) = \frac{1}{\lambda} D\left(\frac{|x - x_0|}{\lambda}\right)$:
1. **Denominator (Marginal input density $\hat{p}(x_0)$)**:
   $$ \hat{p}(x_0) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) $$
2. **Numerator (Joint density integral $\int y \hat{p}(x_0, y) dy$)**:
   Using a 2D independent product kernel to estimate the joint density $\hat{p}(x_0, y) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) K_{h_y}(y, y_i)$, substitute this into the integral over $y$:
   $$ \int y \, \hat{p}(x_0, y) \, dy = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) \underbrace{\int y K_{h_y}(y, y_i) \, dy}_{= y_i} = \frac{1}{N} \sum_{i=1}^N K_\lambda(x_0, x_i) y_i $$

Dividing the estimated numerator by the estimated denominator yields the **Nadaraya–Watson kernel regression estimator** directly and unconditionally:
$$
\hat{f}(x_0) = \frac{\int y \hat{p}(x_0, y) dy}{\hat{p}(x_0)} = \frac{\sum_{i=1}^N K_\lambda(x_0, x_i) y_i}{\sum_{i=1}^N K_\lambda(x_0, x_i)}
$$
**Key Takeaway**: Kernel regression is not an ad-hoc heuristic weighting rule; it is the mathematically exact **nonparametric plug-in estimator of the true conditional expectation $\mathbb{E}[Y \mid X = x]$**.

#### (3) Machine Learning Clarification: Localization Kernels vs. Mercer / RKHS Kernels
ESL Chapter 6 explicitly warns: **Do not confuse the localization kernels in this chapter with the "kernel trick" used in Support Vector Machines!**

| Dimension | Localization Kernel (ESL Chapter 6) | Mercer / RKHS Kernel (ESL Chapters 5.8 & 12) |
| :--- | :--- | :--- |
| **Mathematical Definition** | Local neighborhood decay window $K_\lambda(x_0, x_i) = D\left(\frac{\|x_i - x_0\|}{\lambda}\right)$ | Positive semi-definite continuous kernel $K(x, x') = \langle \phi(x), \phi(x') \rangle_\mathcal{H}$ |
| **Core Mechanism** | **Neighborhood weighting in the original input space** (memory-based localization) | **Implicit mapping into a high/infinite-dimensional Reproducing Kernel Hilbert Space (RKHS)** |
| **Computation Paradigm** | **Lazy Learning**: Virtually zero offline training; all computation occurs at query time | **Eager Learning**: Solves a global dual quadratic program or kernel matrix inversion during training |
| **Canonical Applications** | Nadaraya-Watson, local linear regression (LOESS/Lowess), volatility surface smoothing | Support Vector Machines (SVM), Kernel Ridge Regression, Gaussian Processes (GP) |

#### (4) The Continuum Spectrum: Continuous Transition from Global OLS to Local Nearest Neighbors
The objective function of local weighted least squares is:
$$
\min_{\beta(x_0)} \sum_{i=1}^N K_\lambda(x_0, x_i) \left[ y_i - b(x_i)^\top \beta(x_0) \right]^2
$$
The bandwidth $\lambda$ acts as a continuous dial between global rigidity and local flexibility:
- When **$\lambda \to \infty$**: Kernel weights become uniform constants $K_\lambda \to \text{const}$, and local regression **strictly degenerates to Global Ordinary Least Squares (OLS)** (minimum variance, but high potential misspecification bias; $\mathrm{df} = 2$);
- When **$\lambda \to 0$**: Kernel weights vanish everywhere except at the single observation closest to $x_0$, and local regression **degenerates to 1-Nearest Neighbor (1-NN) interpolation** (zero bias, but unbounded variance; $\mathrm{df} = N$);
- **Finite Bandwidth $\lambda \in (0, \infty)$**: Forms a continuous spectrum balancing the bias-variance tradeoff between the global parametric extreme and the local nonparametric extreme.

---

### 2. From k-NN to Nadaraya–Watson Kernel Regression (ESL 6.1)
- **Defects of the k-NN Running Mean**:
  A simple $k$-nearest-neighbor running mean estimates $\hat{f}(x) = \frac{1}{k}\sum_{x_i \in N_k(x)} y_i$. As $x$ shifts smoothly, observations enter and leave the neighborhood $N_k(x)$ abruptly in discrete steps, yielding an unnaturally jagged and discontinuous curve $\hat{f}(x)$.
- **The Nadaraya–Watson Kernel Estimator (1964)**:
  Replace the 0-1 indicator weights with a smoothly decaying **kernel weighting function** $K_\lambda(x_0, x_i) = D\left(\frac{|x_i - x_0|}{\lambda}\right)$:
  $$
  \hat{f}(x_0) = \frac{\sum_{i=1}^N K_\lambda(x_0, x_i) y_i}{\sum_{i=1}^N K_\lambda(x_0, x_i)} = \sum_{i=1}^N l_i(x_0) y_i
  $$
  where the normalized equivalent weights $l_i(x_0) = \frac{K_\lambda(x_0, x_i)}{\sum_{j=1}^N K_\lambda(x_0, x_j)}$ satisfy non-negativity and $\sum_{i=1}^N l_i(x_0) = 1$.
  - **Local Constant Fit Equivalence**: The Nadaraya–Watson estimate is mathematically equivalent to solving a local weighted least squares problem for a constant:
    $$
    \hat{f}(x_0) = \arg\min_c \sum_{i=1}^N K_\lambda(x_0, x_i)(y_i - c)^2
    $$
- **Comparison of Three Common Kernels**:
  1. **Epanechnikov Kernel**: $D(t) = \frac{3}{4}(1 - t^2) \cdot \mathbb{I}(|t| \le 1)$. Compact support. Asymptotically optimal in the sense of minimizing mean squared error (AMSE) among nonnegative kernels, though its derivative is discontinuous at the support boundaries.
  2. **Tri-cube Kernel (Default in Cleveland's LOESS)**: $D(t) = (1 - |t|^3)^3 \cdot \mathbb{I}(|t| \le 1)$. Compact support, twice continuously differentiable at the support boundaries, with a flatter peak and smoother transitions.
  3. **Gaussian Kernel**: $D(t) = \frac{1}{\sqrt{2\pi}} e^{-t^2/2}$. Infinite support, infinitely differentiable everywhere, with bandwidth parameter $\lambda$ acting as the standard deviation.
- **Bandwidth $\lambda$ & The Bias-Variance Tradeoff**:
  - $\lambda \to 0$ (narrow window): Dominated by only one or very few points $\implies$ **low bias, high variance** (interpolates data, extreme overfitting).
  - $\lambda \to \infty$ (wide window): All points receive equal weight $\implies$ **high bias, low variance** (degenerates to the global sample mean $\bar{y}$, severe underfitting).
  - **Metric Bandwidth vs. k-NN Adaptive Bandwidth**:
    - Constant metric bandwidth $\lambda$ maintains a constant neighborhood radius, keeping bias roughly uniform across space, but causes variance to spike in sparse data regions.
    - $k$-NN adaptive bandwidth $h_k(x_0) = |x_0 - x_{[k]}|$ fixes the effective sample size $k$, ensuring uniform variance, but broadens the window in sparse regions, increasing bias.

### 3. The Fatal Flaw: Boundary Bias & Mathematical Analysis
Why is the Nadaraya–Watson estimator often rejected as an inadequate baseline in quantitative research?
- **Intuitive Flaw**:
  In the interior of the data cloud, neighbors are balanced symmetrically to the left and right of $x_0$, so overestimates and underestimates cancel out.
  At the boundary of the support (e.g., $x_0 = 0$ on domain $[0, 1]$), all available neighbors lie strictly on one side ($x_i > x_0$). If the true underlying function has a non-zero slope ($f'(x_0) > 0$), all neighboring points systematically evaluate above $f(x_0)$, causing the weighted average to **systematically overestimate the true value**.
- **Taylor Series Derivation of Bias Orders**:
  Expanding the true function $f(x_i)$ around $x_0$:
  $$
  f(x_i) = f(x_0) + f'(x_0)(x_i - x_0) + \frac{f''(x_0)}{2}(x_i - x_0)^2 + O((x_i - x_0)^3)
  $$
  Taking the conditional expectation $\mathbb{E}[\hat{f}(x_0) \mid X] = \sum_{i=1}^N l_i(x_0) f(x_i)$, and noting $\sum l_i(x_0) = 1$:
  $$
  \operatorname{Bias}(\hat{f}(x_0)) = \mathbb{E}[\hat{f}(x_0)] - f(x_0) = f'(x_0) \underbrace{\sum_{i=1}^N l_i(x_0)(x_i - x_0)}_{\text{First Moment}} + \frac{f''(x_0)}{2} \sum_{i=1}^N l_i(x_0)(x_i - x_0)^2 + O(h^3)
  $$
  - **Interior Region**: Symmetric support leads to $\sum l_i(x_0)(x_i - x_0) = 0$, so the first-order term cancels. The bias is dominated by curvature: **$O(h^2) f''(x_0)$**.
  - **Boundary Region**: One-sided support leaves $\sum l_i(x_0)(x_i - x_0) = O(h) \ne 0$. The boundary bias degrades to **$O(h) f'(x_0)$**—an entire order of magnitude worse in convergence speed!

### 4. Local Linear Regression & "Automatic Kernel Carpentry" (ESL 6.1.1)
To eliminate the $O(h)$ boundary bias, local linear regression upgrades the model from a local constant to a local tangent line at every query point $x_0$.

- **Weighted Least Squares (WLS) Formulation**:
  At query point $x_0$, solve:
  $$
  \min_{\alpha(x_0), \beta(x_0)} \sum_{i=1}^N K_\lambda(x_0, x_i) \left[ y_i - \alpha(x_0) - \beta(x_0)(x_i - x_0) \right]^2
  $$
  Because regressors are centered at $(x_i - x_0)$, the evaluated prediction at $x = x_0$ is simply the intercept: $\hat{f}(x_0) = \hat{\alpha}(x_0)$.

- **Matrix Solution & The Equivalent Kernel**:
  Let basis vector $b(x) = (1, x - x_0)^\top$, and let design matrix $\mathbf{B}_{N \times 2}$ have row $i$ as $(1, x_i - x_0)$. Let weight diagonal matrix $\mathbf{W}(x_0) = \operatorname{diag}(K_\lambda(x_0, x_1), \dots, K_\lambda(x_0, x_N))$.
  From standard weighted normal equations:
  $$
  \begin{pmatrix} \hat{\alpha}(x_0) \\ \hat{\beta}(x_0) \end{pmatrix} = \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{y}
  $$
  The point prediction is linear in $y$:
  $$
  \hat{f}(x_0) = e_1^\top \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{y} = \sum_{i=1}^N l_i(x_0) y_i
  $$
  where row vector $l(x_0)^\top = e_1^\top \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} \mathbf{B}^\top \mathbf{W}(x_0)$ is the **equivalent kernel**.

- **Why is it called "Automatic Kernel Carpentry"?**
  By the matrix identity $\left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right) \cdot \left[ \left( \mathbf{B}^\top \mathbf{W}(x_0) \mathbf{B} \right)^{-1} e_1 \right] = e_1$:
  $$
  \mathbf{B}^\top \mathbf{W}(x_0) l(x_0) = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
  $$
  Writing out the two rows explicitly:
  1. Row 1 (Zeroth Moment): $\sum_{i=1}^N l_i(x_0) = 1$ (preserves level unbiasedness)
  2. Row 2 (First Moment): $\sum_{i=1}^N l_i(x_0)(x_i - x_0) = 0$ (**the first-order moment is strictly zero everywhere, even on asymmetric boundaries!**)
  
  Substituting this back into the Taylor bias formula, the term $f'(x_0) \sum l_i(x_0)(x_i - x_0) \equiv 0$ **vanishes identically**!
  At boundary points, the equivalent kernel $l_i(x_0)$ automatically adapts its shape (increasing weights on near points and dipping slightly negative on distant points) to cancel the first-order slope bias, reducing boundary bias from $O(h)$ to $O(h^2)$ with zero manual tuning.

- **Polynomial Degree Tradeoffs (ESL 6.1.2)**:
  - **Local Quadratic ($d=2$)**: In regions of high interior curvature ($|f''(x)| \gg 0$), local linear fits exhibit "trimming hills and filling valleys" bias. Local quadratic fits remove this curvature bias (bias becomes $O(h^4)$), but increase variance considerably at boundaries.
  - **Odd-Degree Dominance**: Asymptotic MSE is dominated by boundary behavior. Moving from degree 0 to degree 1 drastically reduces boundary bias with negligible variance penalty. Moving from degree 1 to degree 2 does not improve boundary bias order while inflating boundary variance.
  - $\implies$ **Industry Rule of Thumb: Default to local linear ($d=1$)**.

### 5. Bandwidth Selection & Effective Degrees of Freedom (ESL 6.2 / Ch.7)
- **Linear Smoother & Smoother Matrix**:
  Predictions across all training points form a linear mapping: $\hat{\mathbf{y}} = \mathbf{S}_\lambda \mathbf{y}$, where row $i$ of $\mathbf{S}_\lambda$ is $l(x_i)^\top$.
- **Effective Degrees of Freedom**:
  Paralleling the projection hat matrix in OLS where $\operatorname{df} = \operatorname{tr}(H) = p+1$, the effective degrees of freedom for a kernel smoother is:
  $$
  \operatorname{df}_\lambda = \operatorname{tr}(\mathbf{S}_\lambda)
  $$
  - As $\lambda \to 0$, $\mathbf{S}_\lambda \to \mathbf{I}_N \implies \operatorname{df}_\lambda = N$ (full interpolation, maximal overfitting).
  - As $\lambda \to \infty$, local linear regression converges to global OLS $\implies \operatorname{df}_\lambda = 2$ (intercept + slope).
- **Leave-One-Out Cross-Validation (LOOCV) Shortcut**:
  For any linear smoother, LOOCV requires no expensive retraining loops:
  $$
  \operatorname{CV}(\lambda) = \frac{1}{N} \sum_{i=1}^N \left( \frac{y_i - \hat{f}_\lambda(x_i)}{1 - S_{\lambda, ii}} \right)^2
  $$
  Or via Generalized Cross-Validation (GCV):
  $$
  \operatorname{GCV}(\lambda) = \frac{1}{N} \sum_{i=1}^N \left( \frac{y_i - \hat{f}_\lambda(x_i)}{1 - \operatorname{tr}(\mathbf{S}_\lambda)/N} \right)^2
  $$

### 6. Multidimensional Smoothing & Escaping the Curse of Dimensionality (ESL 6.3–6.4)
- **Multivariate Local Linear Regression in $\mathbb{R}^p$**:
  Basis vector expands to $b(x) = (1, (x - x_0)^\top)^\top \in \mathbb{R}^{p+1}$ with radial kernel $K_\lambda(x_0, x) = D\left(\frac{\|x - x_0\|_2}{\lambda}\right)$. Excellent for 2D/3D applications such as implied volatility surface calibration across strike and maturity.
- **The Curse of Dimensionality**:
  For dimensions $p \ge 4$, local smoothing collapses due to two geometric realities:
  1. **Empty Space Phenomenon**: To enclose a fraction $r$ of sample observations in a unit hypercube in $\mathbb{R}^p$, the required neighborhood radius is $e_p(r) = r^{1/p}$.
     - For $p=1$, capturing $1\%$ of the sample requires radius $e = 0.01$ (truly localized).
     - For $p=10$, capturing $1\%$ requires $e = (0.01)^{0.1} \approx 0.63$ (covers over $60\%$ of each feature axis; completely non-local!). Nonparametric MSE convergence slows to $O(N^{-4/(4+p)})$.
  2. **Boundary Proliferation**: In high-dimensional hyperspheres, almost all volume resides in a thin outer shell ($1 - (1-\epsilon)^p \to 1$). Every sample point is effectively on the boundary.
- **Escape Routes: Structured Nonparametric Models (ESL 6.4)**:
  Top quantitative desks avoid unstructured high-dimensional smoothing by imposing domain structure:
  1. **Structured Kernels**: Use a positive semi-definite metric matrix $\mathbf{A} \succeq 0$: $K_{\lambda, \mathbf{A}}(x_0, x) = D\left(\frac{(x - x_0)^\top \mathbf{A} (x - x_0)}{\lambda}\right)$ to eliminate noise dimensions.
  2. **Generalized Additive Models (GAM / ESL Ch.9)**:
     Decompose the target into additive univariate functions:
     $$f(X) = \alpha + \sum_{j=1}^p g_j(X_j)$$
     Solved via the **Backfitting Algorithm**: iteratively smooth partial residuals $y - \alpha - \sum_{k \ne j} g_k(x_k)$ against $X_j$ using 1D local linear regression. Preserves 1D nonparametric convergence rates $O(N^{-2/5})$ while modeling nonlinear factor dependencies.
  3. **Varying-Coefficient Models (The Quantitative Finance Standard)**:
     $$f(X, Z) = \sum_{j=1}^q \beta_j(Z) X_j$$
     Factors $X$ enter linearly, but their factor loadings $\beta(Z)$ vary smoothly according to low-dimensional macroeconomic regime indicators $Z$ (e.g., market volatility, interest rate levels, liquidity spreads). Fits separate local WLS regressions conditional on $Z=z_0$, creating **regime-switching dynamic factor models**.

---

## Module 5: Core Classical Problems and Analytical Proofs (Green Book + HOTS + ESL Calculations)

This module systematically compiles core linear regression, covariance analysis, and spectral decomposition problems from Xinfeng Zhou's *A Practical Guide to Quantitative Finance Interviews* (the "Green Book"), Timothy Crack's *Heard on the Street* (HOTS), and *The Elements of Statistical Learning* (ESL), providing rigorous algebraic derivations and geometric/physical intuitions for each problem.

---

### 1. Three-Variable Correlation Bound Derivation (Gram Matrix PSD & Geometric Angles)

> **Problem Definition (Green Book 3.6 / Three-Variable Correlation Bounds)**:
> Let $X, Y, Z$ be zero-mean, unit-variance random variables. The correlation between $X$ and $Y$ is $\rho_{xy} = 0.8$, and the correlation between $X$ and $Z$ is $\rho_{xz} = 0.8$.
> 1. Find the maximum and minimum possible values of the correlation between $Y$ and $Z$, $\rho_{yz}$;
> 2. Generalize to arbitrary correlations $\rho_{xy} = a$ and $\rho_{xz} = b$.

**Intuition & Mental Model**:
Correlation matrices must be **Positive Semi-Definite (PSD)**. Geometrically, centered unit-variance random variables form unit vectors in Hilbert space ($L^2$), where the correlation equals the cosine of the angle between vectors: $\rho = \cos\theta$.

**Step-by-Step Derivation**:

**Method 1: Positive Semi-Definite Gram Matrix**
The correlation matrix $\mathbf{R}$ must satisfy $\mathbf{R} \succeq 0$, which requires $\det(\mathbf{R}) \ge 0$:
$$
\mathbf{R} = \begin{pmatrix} 1 & 0.8 & 0.8 \\ 0.8 & 1 & \rho \\ 0.8 & \rho & 1 \end{pmatrix}
$$
Expanding along the first row:
$$
\begin{aligned}
\det(\mathbf{R}) &= 1 \cdot (1 - \rho^2) - 0.8 \cdot (0.8 - 0.8\rho) + 0.8 \cdot (0.8\rho - 0.8) \\
&= 1 - \rho^2 - 0.64 + 0.64\rho + 0.64\rho - 0.64 \\
&= -\rho^2 + 1.28\rho - 0.28 \ge 0
\end{aligned}
$$
Multiplying by $-1$:
$$
\rho^2 - 1.28\rho + 0.28 \le 0
$$
Solving the quadratic equation $\rho^2 - 1.28\rho + 0.28 = 0$:
$$
\rho = \frac{1.28 \pm \sqrt{1.28^2 - 4(0.28)}}{2} = \frac{1.28 \pm \sqrt{1.6384 - 1.12}}{2} = \frac{1.28 \pm \sqrt{0.5184}}{2} = \frac{1.28 \pm 0.72}{2}
$$
Hence:
- $\rho_{\max} = \frac{1.28 + 0.72}{2} = \boxed{1.0}$
- $\rho_{\min} = \frac{1.28 - 0.72}{2} = \boxed{0.28}$

**Method 2: Euclidean Vector Angle (Triangle Inequality)**
In $L^2$, $\langle U, V \rangle = \operatorname{Corr}(U, V) = \cos\theta$:
- $\cos\theta_{xy} = 0.8 \implies \theta_{xy} = \theta_0 = \arccos(0.8)$;
- $\cos\theta_{xz} = 0.8 \implies \theta_{xz} = \theta_0 = \arccos(0.8)$.
By spherical/Euclidean triangle inequalities, the angle $\theta_{yz}$ between $Y$ and $Z$ satisfies:
$$
|\theta_{xy} - \theta_{xz}| \le \theta_{yz} \le \theta_{xy} + \theta_{xz} \implies 0 \le \theta_{yz} \le 2\theta_0
$$
Since $\cos\theta$ is monotonically decreasing on $[0, \pi]$:
1. **Maximum correlation (minimal angle)**: When $\theta_{yz} = 0$, $Y$ and $Z$ are collinear:
   $$ \rho_{\max} = \cos(0) = \boxed{1.0} $$
2. **Minimum correlation (maximal angle)**: When $\theta_{yz} = 2\theta_0$, $Y$ and $Z$ lie on opposite sides of $X$ in the same plane:
   $$ \rho_{\min} = \cos(2\theta_0) = 2\cos^2\theta_0 - 1 = 2(0.8)^2 - 1 = 2(0.64) - 1 = \boxed{0.28} $$

**General Formula**:
For $\rho_{xy} = a, \rho_{xz} = b$, let $\theta_a = \arccos a, \theta_b = \arccos b$:
$$
\rho_{yz} \in \left[ ab - \sqrt{(1 - a^2)(1 - b^2)},\; ab + \sqrt{(1 - a^2)(1 - b^2)} \right]
$$

---

### 2. Pairwise Equicorrelated Matrix Positive Semi-Definite Bound

> **Problem Definition (Green Book 3.6 / Equicorrelated Matrix PSD Condition)**:
> Suppose there are $n$ assets $X_1, X_2, \dots, X_n$, each with variance $\sigma^2 > 0$. The pairwise correlation between any two distinct assets is identical: $\operatorname{Corr}(X_i, X_j) = \rho, \forall i \ne j$.
> 1. Find the theoretical admissible range of $\rho$ such that the correlation matrix is valid (positive semi-definite);
> 2. What happens to the lower bound as $n \to \infty$? What is the fundamental takeaway for portfolio diversification?

**Step-by-Step Derivation**:

**Method 1: Eigenvalue Decomposition**
The equicorrelated matrix $\mathbf{R}_{n \times n}$ has the algebraic structure:
$$
\mathbf{R} = (1 - \rho)\mathbf{I}_n + \rho \mathbf{1}\mathbf{1}^\top
$$
where $\mathbf{1} = (1, 1, \dots, 1)^\top \in \mathbb{R}^n$.
1. For eigenvector $\mathbf{1}$:
   $$ \mathbf{R}\mathbf{1} = (1 - \rho)\mathbf{1} + \rho \mathbf{1}(\mathbf{1}^\top \mathbf{1}) = (1 - \rho)\mathbf{1} + n\rho \mathbf{1} = [1 + (n - 1)\rho]\mathbf{1} $$
   Thus, $\lambda_1 = 1 + (n - 1)\rho$ (multiplicity 1).
2. For any vector $v$ orthogonal to $\mathbf{1}$ ($v \perp \mathbf{1}$, spanning an $(n-1)$-dimensional subspace):
   $$ \mathbf{R}v = (1 - \rho)v + \rho \mathbf{1}(\mathbf{1}^\top v) = (1 - \rho)v $$
   Thus, $\lambda_2 = \dots = \lambda_n = 1 - \rho$ (multiplicity $n - 1$).

Positive semi-definiteness ($\mathbf{R} \succeq 0$) requires all eigenvalues to be non-negative:
$$
\begin{cases}
1 - \rho \ge 0 \implies \rho \le 1 \\
1 + (n - 1)\rho \ge 0 \implies \rho \ge -\frac{1}{n - 1}
\end{cases}
$$
Thus, the exact valid range is:
$$
\boxed{-\frac{1}{n - 1} \le \rho \le 1}
$$

**Method 2: Equal-Weighted Portfolio Variance (Algebraic Decomposition)**
Consider the sum portfolio $S = \sum_{i=1}^n X_i$. Total variance must be non-negative:
$$
\operatorname{Var}(S) = \sum_{i=1}^n \operatorname{Var}(X_i) + \sum_{i \ne j} \operatorname{Cov}(X_i, X_j) = n\sigma^2 + n(n - 1)\rho\sigma^2 = n\sigma^2[1 + (n - 1)\rho] \ge 0
$$
Since $n\sigma^2 > 0$, this immediately yields $1 + (n - 1)\rho \ge 0 \implies \rho \ge -\frac{1}{n - 1}$.

**Financial Takeaway**:
- For $n = 2$: $\rho \ge -1$ (two assets can be perfectly negatively correlated);
- For $n = 3$: $\rho \ge -1/2 = -0.5$;
- As $n \to \infty$: $\lim_{n \to \infty} \left(-\frac{1}{n - 1}\right) = 0$.
**Conclusion**: In an infinite universe of assets, they cannot all be mutually negatively correlated. Systematic market risk forces the average pairwise correlation to be bounded below by 0.

---

### 3. Correlation Matrix Validity and Cholesky Simulation

> **Problem Definition (Green Book 3.6 / Covariance Singularity & Simulation)**:
> Given pairwise correlations among three assets: $\rho_{12} = 0.6, \rho_{23} = 0.8, \rho_{13} = 0$.
> 1. Is this correlation matrix mathematically valid?
> 2. If valid, describe how to generate correlated Monte Carlo asset paths using the Cholesky decomposition.

**Step-by-Step Solution**:

**Step 1: Verify Positive Semi-Definiteness**
Construct the matrix:
$$
\mathbf{R} = \begin{pmatrix} 1 & 0.6 & 0 \\ 0.6 & 1 & 0.8 \\ 0 & 0.8 & 1 \end{pmatrix}
$$
Check principal minors:
- $1 \times 1$ minor: $1 > 0$
- $2 \times 2$ minor: $1 - 0.6^2 = 0.64 > 0$
- Determinant:
  $$ \det(\mathbf{R}) = 1(1 - 0.8^2) - 0.6(0.6 - 0) + 0 = 0.36 - 0.36 = 0 $$
All principal minors are $\ge 0$ and $\det(\mathbf{R}) = 0$. Hence, $\mathbf{R}$ is a **valid positive semi-definite matrix** (residing on the boundary of degeneracy where the vectors are coplanar).

**Step 2: Cholesky Factorization & Simulation**
Compute lower-triangular $\mathbf{L}$ such that $\mathbf{R} = \mathbf{L}\mathbf{L}^\top$:
1. $l_{11} = \sqrt{1} = 1$;
2. $l_{21} = 0.6 / 1 = 0.6$, $l_{22} = \sqrt{1 - 0.6^2} = 0.8$;
3. $l_{31} = 0 / 1 = 0$, $l_{32} = (0.8 - 0 \times 0.6) / 0.8 = 1.0$, $l_{33} = \sqrt{1 - 0^2 - 1.0^2} = 0$.

$$
\mathbf{L} = \begin{pmatrix} 1 & 0 & 0 \\ 0.6 & 0.8 & 0 \\ 0 & 1 & 0 \end{pmatrix}
$$
**Simulation Algorithm**:
Sample independent standard normals $Z = (Z_1, Z_2, Z_3)^\top \sim \mathcal{N}(0, \mathbf{I})$. The correlated variables are generated via $X = \mathbf{L}Z$:
$$
\begin{pmatrix} X_1 \\ X_2 \\ X_3 \end{pmatrix} = \begin{pmatrix} Z_1 \\ 0.6 Z_1 + 0.8 Z_2 \\ Z_2 \end{pmatrix}
$$
Verifying covariances: $\mathbb{E}[X_1 X_2] = 0.6$, $\mathbb{E}[X_2 X_3] = 0.8$, $\mathbb{E}[X_1 X_3] = 0$. Exact match!

---

### 4. CAPM Beta, Variance Decomposition, and Reverse Regression

> **Problem Definition (Heard on the Street / Conditional Expectation & Reverse Regression)**:
> Stock A has daily volatility $\sigma_A = 2\%$, market index M has volatility $\sigma_M = 1\%$, and their correlation is $\rho = 0.5$.
> 1. Calculate stock A's CAPM $\beta$ against M, model $R^2$, and residual idiosyncratic volatility $\sigma_\varepsilon$;
> 2. If stock A surged $+4\%$ today, what is your best estimate of market M's return today?
> 3. Under the IID return assumption, what is your prediction for stock A's return tomorrow?

**Step-by-Step Derivation & Pitfalls**:

**Part 1: Forward Regression Parameters**
- **CAPM Beta**:
  $$ \beta_{A \sim M} = \rho \frac{\sigma_A}{\sigma_M} = 0.5 \times \frac{2\%}{1\%} = \boxed{1.0} $$
- **Coefficient of Determination $R^2$**:
  $$ R^2 = \rho^2 = 0.5^2 = \boxed{0.25 = 25\%} $$
- **Residual Volatility**:
  $$ \sigma_\varepsilon = \sigma_A \sqrt{1 - R^2} = 2\% \times \sqrt{1 - 0.25} = 2\% \times \frac{\sqrt{3}}{2} = \boxed{\sqrt{3}\% \approx 1.732\%} $$

**Part 2: The Reverse Regression Trap**
> **Common Intuitive Pitfall**: Directly rearranging the forward equation to conclude "since $\beta = 1.0$, when the stock rises $4\%$, the market also rises $4\%$".
> **Root Cause**: Algebraically inverting the forward regression equation ignores the asymmetry of conditional expectations when the direction of projection changes.

**Correct Derivation**:
To predict $R_M$ given $R_A = +4\%$, we must construct the reverse regression conditioning on $R_A$:
$$
\beta_{M \sim A} = \rho \frac{\sigma_M}{\sigma_A} = 0.5 \times \frac{1\%}{2\%} = \boxed{0.25}
$$
Thus, the expected market return is:
$$
\mathbb{E}[R_M \mid R_A = 4\%] = \beta_{M \sim A} \times 4\% = 0.25 \times 4\% = \boxed{+1\%}
$$
**Standardized Variable Intuition (Regression to the Mean)**:
In $Z$-scores, $z_A = \frac{+4\%}{\sigma_A} = \frac{4\%}{2\%} = +2$ ($2\sigma$ shock).
Conditioning yields: $\hat{z}_M = \rho z_A = 0.5 \times 2 = +1$ ($1\sigma$ shock).
Converting back: $1 \times \sigma_M = +1\%$. Since $|\rho| < 1$, extreme observations always predict less extreme partners!

**Part 3: IID Tomorrow Forecast**
> **Conceptual Clarification**: Under the IID return assumption, what is the expected return of the stock tomorrow?
> **Analysis**: Expected return tomorrow is the unconditional mean (**approximately 0%**).
Returns were specified as **IID**. Cross-sectional regression to the mean is purely a property of bivariate conditioning at a single snapshot, NOT negative time-series autocorrelation.

---

### 5. Effect of Affine Transformations on Covariance and Correlation

> **Problem Definition (Heard on the Street 4.5)**:
> Given $\operatorname{Corr}(X, Y) = \rho$:
> 1. Find $\operatorname{Corr}(X + 5, Y)$;
> 2. Find $\operatorname{Corr}(5X, Y)$;
> 3. Find $\operatorname{Corr}(-5X + 3, 2Y - 7)$.

**Step-by-Step Derivation**:
Using the bilinear property of covariance $\operatorname{Cov}(aX + b, cY + d) = ac \operatorname{Cov}(X, Y)$ and scale property of standard deviations $\sigma_{aX+b} = |a|\sigma_X$:
$$
\operatorname{Corr}(aX + b, cY + d) = \frac{ac \operatorname{Cov}(X, Y)}{|a|\sigma_X |c|\sigma_Y} = \frac{ac}{|a||c|} \rho = \operatorname{sgn}(ac) \rho
$$
1. **Translation ($a=1, c=1$)**: $\operatorname{Corr}(X + 5, Y) = \boxed{\rho}$ (translation invariant);
2. **Positive Scaling ($a=5, c=1$)**: $\operatorname{Corr}(5X, Y) = \boxed{\rho}$ (scale invariant);
3. **Opposite Sign Scaling ($a=-5, c=2$)**: $ac = -10 < 0 \implies \operatorname{Corr}(-5X + 3, 2Y - 7) = \boxed{-\rho}$.

---

### 6. Omitted Variable Bias (OVB) Analytical Derivation

> **Problem Definition (Omitted Variable Bias in Multi-Factor Models)**:
> Suppose the true data generating process (DGP) contains two factors:
> $$ y = \beta_1 x_1 + \beta_2 x_2 + \varepsilon, \qquad \mathbb{E}[\varepsilon \mid x_1, x_2] = 0 $$
> A researcher mistakenly omits $x_2$ and estimates $y = \alpha x_1 + u$.
> 1. Derive the large-sample probability limit $\operatorname{plim}\hat\alpha$ and state the omitted variable bias formula;
> 2. **Quantitative Case Study**: If $x_1$ is a short-term momentum factor and the omitted $x_2$ is an industry sentiment factor ($\beta_2 > 0$), and high-momentum stocks cluster in high-sentiment industries ($\operatorname{Cov}(x_1, x_2) > 0$), is the univariate momentum slope overestimated or underestimated?

**Step-by-Step Derivation**:
The univariate OLS estimator is:
$$
\hat\alpha = \frac{\sum x_{1i} y_i}{\sum x_{1i}^2} = \beta_1 + \beta_2 \frac{\sum x_{1i} x_{2i}}{\sum x_{1i}^2} + \frac{\sum x_{1i} \varepsilon_i}{\sum x_{1i}^2}
$$
Taking the probability limit as $N \to \infty$:
$$
\operatorname{plim}\hat\alpha = \beta_1 + \beta_2 \frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}
$$
The **Omitted Variable Bias** is:
$$
\operatorname{Bias} = \operatorname{plim}\hat\alpha - \beta_1 = \boxed{\beta_2 \frac{\operatorname{Cov}(x_1, x_2)}{\operatorname{Var}(x_1)}}
$$
**Quantitative Takeaway**:
Since $\beta_2 > 0$ and $\operatorname{Cov}(x_1, x_2) > 0$, $\operatorname{Bias} > 0$. The univariate momentum exposure is **substantially overestimated**, confusing industry Beta risk with idiosyncratic stock Alpha.

---

### 7. Regressor Measurement Error and Attenuation Bias

> **Problem Definition (Attenuation Bias under Measurement Noise in Regressors)**:
> The true economic model is $y = \beta x^* + \varepsilon$ with $\beta \ne 0$ and $\mathbb{E}[\varepsilon \mid x^*] = 0$. Due to market microstructure noise (bid-ask bounce, stale quotes), $x^*$ cannot be directly observed. Instead, the researcher observes $x = x^* + u$, where $u \sim (0, \sigma_u^2)$ is white noise independent of $x^*$ and $\varepsilon$.
> 1. Derive the probability limit $\operatorname{plim}\hat\beta$ when regressing $y$ on the noisy proxy $x$;
> 2. Explain why this causes "attenuation bias" and why increasing sample size $N \to \infty$ does NOT fix it.

**Step-by-Step Derivation**:
$$
\hat\beta = \frac{\widehat{\operatorname{Cov}}(x, y)}{\widehat{\operatorname{Var}}(x)}
$$
1. **Numerator**: $\operatorname{Cov}(x^* + u, \beta x^* + \varepsilon) = \beta \operatorname{Var}(x^*) = \beta \sigma_{x^*}^2$;
2. **Denominator**: $\operatorname{Var}(x^* + u) = \sigma_{x^*}^2 + \sigma_u^2$.
$$
\operatorname{plim}\hat\beta = \beta \cdot \boxed{\frac{\sigma_{x^*}^2}{\sigma_{x^*}^2 + \sigma_u^2}} = \beta \cdot \frac{1}{1 + \frac{\sigma_u^2}{\sigma_{x^*}^2}}
$$
**Conclusion**:
The reliability ratio $\frac{\sigma_{x^*}^2}{\sigma_{x^*}^2 + \sigma_u^2} < 1$ shrinks the slope **toward zero**. OLS is inconsistent under measurement error in $x$. Remedy requires Instrumental Variables (IV) or Kalman filtering.

---

### 8. Multicollinearity, VIF, and the Prediction vs. Interpretation Paradox

> **Problem Definition (High-Dimensional Collinearity and Variance Inflation Factor)**:
> 1. State the analytical formula for the variance of the $j$-th regression coefficient $\operatorname{Var}(\hat\beta_j)$ and define the Variance Inflation Factor (VIF);
> 2. Explain why severe multicollinearity destroys factor interpretation but leaves in-sample predictions virtually unharmed.

**Step-by-Step Derivation**:
In multivariate OLS, $\operatorname{Var}(\hat\beta) = \sigma^2 (X^\top X)^{-1}$. Expanding the $j$-th diagonal entry:
$$
\operatorname{Var}(\hat\beta_j) = \frac{\sigma^2}{\sum_{i=1}^N (x_{ij} - \bar{x}_j)^2 (1 - R_j^2)} = \frac{\sigma^2}{\operatorname{TSS}_j} \cdot \operatorname{VIF}_j
$$
where $R_j^2$ is the $R^2$ from regressing $x_j$ on all remaining regressors, and $\operatorname{VIF}_j = \frac{1}{1 - R_j^2}$.

**The Geometric Paradox**:
- **Interpretation collapses**: As $R_j^2 \to 1$, $\operatorname{VIF}_j \to \infty$. Standard errors blow up, $t$-statistics drop to zero, and individual coefficient signs become erratic.
- **Prediction remains solid**: Geometrically, the subspace $\mathrm{Col}(X)$ is well-defined. The orthogonal projection $\hat{y} = Hy$ onto the subspace is unique and numerically stable, even if the individual basis vectors spanning that plane are nearly collinear.

---

### 9. Optimal Futures Hedge Ratio Derivation

> **Problem Definition (Green Book 4.5 / Minimum Variance Hedging)**:
> An asset manager holds spot asset $S$ and hedges using futures contracts $F$. Over the hedging period, spot price change is $\Delta S$ and futures price change is $\Delta F$. The hedged portfolio change is $\Delta \Pi = \Delta S - h \Delta F$, where $h$ is the futures hedge ratio per unit of spot.
> 1. Find the hedge ratio $h^*$ that minimizes portfolio variance;
> 2. Show that $h^*$ is identical to the univariate OLS slope and derive the percentage variance reduction.

**Step-by-Step Derivation**:
1. **Minimize Portfolio Variance**:
   $$ \operatorname{Var}(\Delta \Pi) = \sigma_S^2 + h^2 \sigma_F^2 - 2h \operatorname{Cov}(\Delta S, \Delta F) $$
   Setting the derivative with respect to $h$ to zero:
   $$ \frac{d \operatorname{Var}(\Delta \Pi)}{dh} = 2h \sigma_F^2 - 2\operatorname{Cov}(\Delta S, \Delta F) = 0 \implies h^* = \frac{\operatorname{Cov}(\Delta S, \Delta F)}{\operatorname{Var}(\Delta F)} = \rho \frac{\sigma_S}{\sigma_F} $$
2. **OLS Equivalence & Variance Reduction**:
   This is mathematically identical to the OLS slope of regressing $\Delta S$ on $\Delta F$.
   Substituting $h^*$ back into the variance:
   $$ \operatorname{Var}^*(\Delta \Pi) = \sigma_S^2(1 - \rho^2) $$
   Percentage risk reduction is $\frac{\sigma_S^2 - \sigma_S^2(1 - \rho^2)}{\sigma_S^2} = \boxed{\rho^2 = R^2}$.

---

### 10. Frisch–Waugh–Lovell (FWL) Theorem and Two-Stage Residual Regression (Ratio $\beta_1 / \beta_2$)

> **Problem Definition (FWL Theorem and Coefficient Ratio in Two-Stage Residual Regression)**:
> Consider the standard multivariate linear regression setting. To simplify algebra without loss of generality, assume all variables are mean-centered (centering does not alter variances, covariances, or regression slopes).
> Suppose an analyst performs three regressions:
> 1. **$Y$ on $X_1$ (Univariate regression isolating residual $\varepsilon$)**:
>    $$ \varepsilon = Y - \gamma X_1, \quad \text{where } \gamma = \frac{\operatorname{Cov}(Y, X_1)}{\operatorname{Var}(X_1)}, \quad \text{with residual orthogonality } \operatorname{Cov}(\varepsilon, X_1) = 0 $$
> 2. **$\varepsilon$ on $X_2$ (Univariate regression of residual on unorthogonalized raw $X_2$)**:
>    $$ \beta_1 = \frac{\operatorname{Cov}(\varepsilon, X_2)}{\operatorname{Var}(X_2)} $$
> 3. **$Y$ on $(X_1, X_2)$ (Standard joint bivariate regression)**:
>    $$ Y = b_1 X_1 + \beta_2 X_2 + u, \quad \text{where multivariate residual } u \text{ satisfies } \operatorname{Cov}(u, X_1) = 0 \text{ and } \operatorname{Cov}(u, X_2) = 0 $$
> Given that the sample correlation between $X_1$ and $X_2$ is $\rho = \operatorname{Corr}(X_1, X_2)$.
>
> **Core Questions**:
> 1. Find the mathematical relationship and ratio $\frac{\beta_1}{\beta_2}$ between slope $\beta_1$ and joint regression coefficient $\beta_2$;
> 2. Many people intuitively assume $\beta_1 = \beta_2$. From the Frisch–Waugh–Lovell (FWL) theorem and geometric orthogonal projections, explain why regressing $\varepsilon$ directly on raw $X_2$ attenuates the estimate, and specify the correct FWL procedure;
> 3. Explain the practical guidance of this conclusion for factor neutralization (industry/style) and incremental factor efficacy testing in multi-factor alpha models.

**Step-by-Step Derivation**:

#### 1. Algebraic Derivation: Relating $\varepsilon$ to the Multivariate Model

The core nuance lies in the fact that the correlation $\rho = \operatorname{Corr}(X_1, X_2)$ is given, while univariate regression slopes use the ratio of covariance to regressor variance $\frac{\operatorname{Cov}}{\operatorname{Var}}$.

**Step 1: Define Regression Specifications and Orthogonality**
- $Y$ on $X_1$:
  $$ \varepsilon = Y - \gamma X_1, \quad \gamma = \frac{\operatorname{Cov}(Y, X_1)}{\operatorname{Var}(X_1)}, \quad \operatorname{Cov}(\varepsilon, X_1) = 0 $$
- $\varepsilon$ on $X_2$:
  $$ \beta_1 = \frac{\operatorname{Cov}(\varepsilon, X_2)}{\operatorname{Var}(X_2)} $$
- $Y$ on $(X_1, X_2)$:
  $$ Y = b_1 X_1 + \beta_2 X_2 + u, \quad \operatorname{Cov}(u, X_1) = 0, \quad \operatorname{Cov}(u, X_2) = 0 $$

**Step 2: Connect Residual $\varepsilon$ to the Joint Model**
Substitute the bivariate equation $Y = b_1 X_1 + \beta_2 X_2 + u$ into $\varepsilon = Y - \gamma X_1$:
$$ \varepsilon = (b_1 - \gamma) X_1 + \beta_2 X_2 + u $$
Compute the covariance between $\varepsilon$ and $X_2$:
$$
\begin{aligned}
\operatorname{Cov}(\varepsilon, X_2) &= \operatorname{Cov}\left( (b_1 - \gamma) X_1 + \beta_2 X_2 + u, \, X_2 \right) \\
&= (b_1 - \gamma)\operatorname{Cov}(X_1, X_2) + \beta_2 \operatorname{Var}(X_2) + \underbrace{\operatorname{Cov}(u, X_2)}_{= 0} \\
&= (b_1 - \gamma)\operatorname{Cov}(X_1, X_2) + \beta_2 \operatorname{Var}(X_2)
\end{aligned}
$$

**Step 3: Eliminate $(b_1 - \gamma)$ via Residual Orthogonality**
By the first-order condition of OLS, the residual $\varepsilon$ is strictly orthogonal to $X_1$, i.e., $\operatorname{Cov}(\varepsilon, X_1) = 0$:
$$
\begin{aligned}
\operatorname{Cov}(\varepsilon, X_1) &= (b_1 - \gamma)\operatorname{Var}(X_1) + \beta_2 \operatorname{Cov}(X_2, X_1) + \underbrace{\operatorname{Cov}(u, X_1)}_{= 0} = 0
\end{aligned}
$$
Solving for $(b_1 - \gamma)$:
$$ b_1 - \gamma = -\beta_2 \frac{\operatorname{Cov}(X_1, X_2)}{\operatorname{Var}(X_1)} $$

**Step 4: Substitute and Express in Terms of Correlation $\rho$**
Substitute $(b_1 - \gamma)$ back into $\operatorname{Cov}(\varepsilon, X_2)$:
$$
\begin{aligned}
\operatorname{Cov}(\varepsilon, X_2) &= \left( -\beta_2 \frac{\operatorname{Cov}(X_1, X_2)}{\operatorname{Var}(X_1)} \right) \operatorname{Cov}(X_1, X_2) + \beta_2 \operatorname{Var}(X_2) \\
&= -\beta_2 \frac{\operatorname{Cov}(X_1, X_2)^2}{\operatorname{Var}(X_1)} + \beta_2 \operatorname{Var}(X_2) \\
&= \beta_2 \operatorname{Var}(X_2) \left( 1 - \frac{\operatorname{Cov}(X_1, X_2)^2}{\operatorname{Var}(X_1)\operatorname{Var}(X_2)} \right)
\end{aligned}
$$
The bracketed term is precisely $1 - \rho^2$, where $\rho^2 = \frac{\operatorname{Cov}(X_1, X_2)^2}{\operatorname{Var}(X_1)\operatorname{Var}(X_2)}$:
$$ \operatorname{Cov}(\varepsilon, X_2) = \beta_2 \operatorname{Var}(X_2)(1 - \rho^2) $$
Dividing both sides by $\operatorname{Var}(X_2)$ yields the explicit closed form for $\beta_1$:
$$ \beta_1 = \frac{\operatorname{Cov}(\varepsilon, X_2)}{\operatorname{Var}(X_2)} = \beta_2 (1 - \rho^2) $$
Thus, the exact ratio is:
$$ \boxed{\frac{\beta_1}{\beta_2} = 1 - \rho^2} $$

---

#### 2. Geometric & Frisch–Waugh–Lovell (FWL) Theorem Perspective

This problem illuminates the subtle, fundamental geometric distinction at the heart of the **Frisch–Waugh–Lovell (FWL) Theorem**:

- **True FWL Procedure**:
  The FWL theorem proves that the multivariate regression slope $\beta_2$ represents the projection of $Y$ onto $X_2$ **after purging both variables of the linear influence of $X_1$**:
  $$ \beta_2 = \frac{\operatorname{Cov}(\varepsilon, \tilde{X}_2)}{\operatorname{Var}(\tilde{X}_2)} $$
  where $\tilde{X}_2 = X_2 - \operatorname{Proj}_{X_1}(X_2) = M_1 X_2$ is the net innovation in $X_2$ orthogonal to $X_1$.
- **The Naive Two-Stage Trap**:
  In the question, the researcher orthogonalized $Y$ (producing $\varepsilon$), but **forgot to orthogonalize $X_2$**, mistakenly regressing $\varepsilon$ onto raw $X_2$:
  $$ \beta_1 = \frac{\operatorname{Cov}(\varepsilon, X_2)}{\operatorname{Var}(X_2)} $$
- **Why Do They Differ by Exactly $(1 - \rho^2)$?**
  1. **Numerator Inner Products are Identical**:
     Since $\varepsilon \perp X_1$ and $X_2 = \operatorname{Proj}_{X_1}(X_2) + \tilde{X}_2$:
     $$ \operatorname{Cov}(\varepsilon, X_2) = \underbrace{\operatorname{Cov}(\varepsilon, \operatorname{Proj}_{X_1}(X_2))}_{= 0} + \operatorname{Cov}(\varepsilon, \tilde{X}_2) = \operatorname{Cov}(\varepsilon, \tilde{X}_2) $$
     The numerator dot product is geometrically identical!
  2. **Denominator Variances Differ**:
     True FWL divides by the net variance $\operatorname{Var}(\tilde{X}_2) = \operatorname{Var}(X_2)(1 - R_{X_2 \sim X_1}^2) = \operatorname{Var}(X_2)(1 - \rho^2)$;
     Naive regression divides by the full variance $\operatorname{Var}(X_2)$, which is inflated by redundant collinear variance shared with $X_1$.
     Therefore, the ratio strictly equals the variance retention fraction:
     $$ \frac{\beta_1}{\beta_2} = \frac{\operatorname{Var}(\tilde{X}_2)}{\operatorname{Var}(X_2)} = 1 - R_{X_2 \sim X_1}^2 = 1 - \rho^2 $$

```fwl-geometry-demo
```

---

#### 3. Practical Implications in Quantitative Multi-Factor Modeling

1. **Both Sides Must be Neutralized**:
   In factor research, researchers often want to test a new factor $X_2$ controlling for risk factors $X_1$ (e.g., industry and size):
   - **Correct (FWL)**: Regress returns on industry dummies to get residual return $\varepsilon$, AND regress $X_2$ on industry dummies to get net factor $\tilde{X}_2$. Then test the slope of $\varepsilon$ on $\tilde{X}_2$.
   - **Incorrect**: Regressing residual return $\varepsilon$ on raw factor $X_2$. If $X_2$ correlates with industry ($\rho \ne 0$), the measured factor return slope is artificially attenuated by $(1 - \rho^2)$, underestimating true factor efficacy!
2. **Incremental Alpha Testing**:
   To establish whether a proposed alpha signal $X_{\text{new}}$ contains non-redundant predictive power beyond a library of existing factors $X_{\text{base}}$, one must project $X_{\text{new}}$ onto $X_{\text{base}}^\perp$ ($\tilde{X}_{\text{new}} = M_{\text{base}} X_{\text{new}}$) and evaluate the significance of the residual signal.

---

### 11. Regression Without Intercept and Negative R²

> **Problem Definition (Algebraic Impact of Omitting the Intercept on Residual Mean and R²)**:
> In empirical tests of CAPM or arbitrage pricing models, if the intercept is forced to zero: $y = X\beta + \varepsilon$.
> 1. Why does the sum of residuals $\sum_{i=1}^N \hat\varepsilon_i$ no longer equal zero?
> 2. Why can the standard coefficient of determination $R^2$ become negative?

**Step-by-Step Derivation**:
1. **Residual Sum Zero Condition**:
   Normal equations state $X^\top \hat\varepsilon = 0$. When an intercept is included, $X$ contains the constant vector $\mathbf{1}$, so $\mathbf{1}^\top \hat\varepsilon = \sum \hat\varepsilon_i = 0$. Without an intercept, $\mathbf{1} \notin \mathrm{Col}(X)$, so the residuals do NOT sum to zero.
2. **Breakdown of TSS Decomposition**:
   $$ \operatorname{TSS} = \sum (y_i - \bar{y})^2 = \operatorname{RSS} + \operatorname{ESS} - 2\bar{y}\sum_{i=1}^N \hat\varepsilon_i $$
   Since $\sum \hat\varepsilon_i \ne 0$, the cross-term does not vanish: $\operatorname{TSS} \ne \operatorname{ESS} + \operatorname{RSS}$.
   If the zero-intercept line fits worse than the horizontal line $y = \bar{y}$, $\operatorname{RSS} > \operatorname{TSS}$, producing **$R^2 = 1 - \frac{\operatorname{RSS}}{\operatorname{TSS}} < 0$**.

---

### 12. Mathematical Mapping Between Daily Return R² ≈ 1% and Information Ratio (IR)

> **Problem Definition (Mathematical Mapping Between Cross-Sectional R² and Realized IC)**:
> In a backtest of an equity alpha signal, the regression $R^2$ of the signal against next-day returns is only $1\%$ (i.e., $0.01$).
> Using the **Fundamental Law of Active Management**, rigorously analyze the commercial value of this predictive power and its annualized Information Ratio (IR).

**Step-by-Step Derivation**:
In a univariate regression, $R^2 = \rho^2 \implies |\rho| = \sqrt{R^2} = \sqrt{0.01} = \boxed{0.10}$.
The signal possesses an **Information Coefficient (IC) of 0.10**.

**Fundamental Law of Active Management (Grinold & Kahn)**:
$$ \operatorname{IR} \approx \operatorname{IC} \times \sqrt{\text{Breadth}} $$
For a universe of $N = 1000$ stocks over $T = 252$ trading days:
- Even conservatively assuming effective independent cross-sectional breadth of $N_{\text{eff}} = 100$:
  $$ \text{Breadth} = 252 \times 100 = 25,200 \implies \operatorname{IR} \approx 0.10 \times \sqrt{25,200} \approx 15.87 $$
- Even considering only time-series breadth ($T = 252$, single-stock portfolio):
  $$ \operatorname{IR} \approx 0.10 \times \sqrt{252} \approx 1.59 $$
In systematic equity market-neutral portfolios, an annualized Sharpe ratio of 1.5 to 2.0 provides substantial allocation value.

**Signal-to-Noise Ratio Characteristics of Financial Markets**:
Financial time series have extremely low signal-to-noise ratios (most daily fluctuations are random noise). The high $R^2$ values common in macroeconomic models do not exist in secondary market asset pricing (a high $R^2$ usually indicates lookahead bias or data leakage). Based on the Fundamental Law of Active Management, $R^2 = 1\%$ (corresponding to $\operatorname{IC} = 0.10$) is sufficient to generate significant risk-adjusted returns in portfolios with broad investment breadth.

---

### 13. ESL 3.4.1: Closed-Form Solutions of OLS, Ridge, Lasso, and Best Subset under Orthogonal Designs

> **Problem Definition (ESL Ex 3.12 / Closed-Form Solutions under Orthonormal Design)**:
> Suppose the feature matrix $X \in \mathbb{R}^{n \times p}$ has centered, orthonormal columns, i.e.,
> $$ X^\top X = I_p $$
> Let the univariate OLS estimator for each coordinate be $\hat\beta_j^{\text{ols}} = X_j^\top Y$.
> 1. Derive and write down the **closed-form parameter solutions** under this orthogonal design for the following four regression methods:
>    - Ordinary Least Squares (OLS);
>    - Ridge Regression ($\ell_2$ penalty);
>    - Lasso Regression ($\ell_1$ penalty);
>    - Best Subset Selection ($\ell_0$ penalty);
> 2. Compare the response function shapes of these four estimators with respect to the univariate OLS estimate $\hat\beta_j^{\text{ols}}$. Using first-order optimality and subgradient conditions, explain why Lasso produces sparse solutions (exact zeros) while Ridge only produces shrinkage.

**Step-by-Step Derivation**:

#### 1. Decoupling the Loss Function under Orthogonality
For any linear regression, expanding the sum of squared errors yields:
$$
\begin{aligned}
\|Y - X\beta\|_2^2 &= Y^\top Y - 2\beta^\top X^\top Y + \beta^\top X^\top X \beta \\
&= Y^\top Y - 2\sum_{j=1}^p \beta_j (X_j^\top Y) + \sum_{j=1}^p \beta_j^2 \quad (\because X^\top X = I_p) \\
&= Y^\top Y - \sum_{j=1}^p (\hat\beta_j^{\text{ols}})^2 + \sum_{j=1}^p (\beta_j - \hat\beta_j^{\text{ols}})^2
\end{aligned}
$$
Because $X^\top X = I_p$, **the joint optimization problem decouples completely into $p$ independent 1-dimensional scalar optimization problems**:
$$ \min_\beta \sum_{j=1}^p \left[ \frac{1}{2}(\beta_j - \hat\beta_j^{\text{ols}})^2 + g(\beta_j) \right] $$

#### 2. Derivation of the Four Closed-Form Estimators
1. **OLS (No penalty, $g(\beta_j) = 0$)**:
   $$ \min_{\beta_j} \frac{1}{2}(\beta_j - \hat\beta_j^{\text{ols}})^2 \implies \boxed{\hat\beta_j^{\text{ols}} = X_j^\top Y} $$
2. **Ridge Regression ($\ell_2$ penalty: $g(\beta_j) = \frac{1}{2}\lambda \beta_j^2$)**:
   Differentiating with respect to $\beta_j$ and setting to zero:
   $$ (\beta_j - \hat\beta_j^{\text{ols}}) + \lambda \beta_j = 0 \implies (1 + \lambda)\beta_j = \hat\beta_j^{\text{ols}} \implies \boxed{\hat\beta_j^{\text{ridge}} = \frac{1}{1 + \lambda} \hat\beta_j^{\text{ols}}} $$
   **Geometric Property**: **Linear Proportional Shrinkage**. The coefficient is smoothly scaled down by a factor of $\frac{1}{1+\lambda} < 1$, but is **never exactly zero** (unless $\hat\beta_j^{\text{ols}} = 0$).
3. **Lasso Regression ($\ell_1$ penalty: $g(\beta_j) = \lambda |\beta_j|$)**:
   The objective is non-differentiable at $\beta_j = 0$. By the **subgradient KKT conditions**:
   $$ 0 \in (\beta_j - \hat\beta_j^{\text{ols}}) + \lambda \, \partial |\beta_j| $$
   - If $\beta_j > 0$, the subdifferential $\partial |\beta_j| = \{1\}$: $\beta_j - \hat\beta_j^{\text{ols}} + \lambda = 0 \implies \beta_j = \hat\beta_j^{\text{ols}} - \lambda$ (requires $\hat\beta_j^{\text{ols}} > \lambda$);
   - If $\beta_j < 0$, the subdifferential $\partial |\beta_j| = \{-1\}$: $\beta_j - \hat\beta_j^{\text{ols}} - \lambda = 0 \implies \beta_j = \hat\beta_j^{\text{ols}} + \lambda$ (requires $\hat\beta_j^{\text{ols}} < -\lambda$);
   - If $\beta_j = 0$, the subdifferential $\partial |\beta_j| = [-1, 1]$: $-\hat\beta_j^{\text{ols}} + \lambda s = 0$ holds for some $s \in [-1, 1] \iff |\hat\beta_j^{\text{ols}}| \le \lambda$.
   Combining these yields the **Soft-Thresholding Operator $\mathcal{S}_\lambda$**:
   $$ \boxed{\hat\beta_j^{\text{lasso}} = \operatorname{sign}(\hat\beta_j^{\text{ols}}) \max\left( 0, \, |\hat\beta_j^{\text{ols}}| - \lambda \right)} $$
   **Geometric Property**: Coefficients with small magnitudes ($|\hat\beta_j^{\text{ols}}| \le \lambda$) are **set strictly to zero (Sparsity)**, while stronger signals are shifted towards zero by a constant amount $\lambda$.
4. **Best Subset Selection ($\ell_0$ penalty: $g(\beta_j) = \frac{1}{2}\lambda \mathbb{I}(\beta_j \ne 0)$)**:
   - If $\beta_j = 0$, loss is $\frac{1}{2}(\hat\beta_j^{\text{ols}})^2$;
   - If $\beta_j \ne 0$, optimal $\beta_j = \hat\beta_j^{\text{ols}}$, loss is $\frac{1}{2}\lambda$.
   - Comparing both: keep the OLS value when $\frac{1}{2}(\hat\beta_j^{\text{ols}})^2 > \frac{1}{2}\lambda \iff |\hat\beta_j^{\text{ols}}| > \sqrt{\lambda}$, otherwise zero it out.
   This yields the **Hard-Thresholding Operator $\mathcal{H}_{\sqrt{\lambda}}$**:
   $$ \boxed{\hat\beta_j^{\text{subset}} = \hat\beta_j^{\text{ols}} \cdot \mathbb{I}(|\hat\beta_j^{\text{ols}}| > \sqrt{\lambda})} $$

#### 3. Comparison Matrix of the Four Estimators

| Method | Penalty | Mathematical Closed-Form Solution $\hat\beta_j$ | Continuity | Sparsity (Exact Zero) |
| :--- | :--- | :--- | :---: | :---: |
| **OLS** | None | $\hat\beta_j^{\text{ols}}$ | Continuous identity | No |
| **Ridge** | $\frac{1}{2}\lambda \beta_j^2$ | $\frac{1}{1 + \lambda}\hat\beta_j^{\text{ols}}$ | Continuous smooth shrinkage | No (never zero) |
| **Lasso** | $\lambda \lVert\beta\rVert_1$ | $\operatorname{sign}(\hat\beta_j^{\text{ols}})(\lvert\hat\beta_j^{\text{ols}}\rvert - \lambda)_+$ | Everywhere continuous | **Yes** (zero if $\le \lambda$) |
| **Best Subset** | $\frac{1}{2}\lambda \mathbb{I}(\beta_j \ne 0)$ | $\hat\beta_j^{\text{ols}} \cdot \mathbb{I}(\lvert\hat\beta_j^{\text{ols}}\rvert > \sqrt{\lambda})$ | **Discontinuous (step jump)** | **Yes** (zero if $\le \sqrt{\lambda}$) |

> **Core Comparison**: Best subset selection has a jump discontinuity at the threshold, causing high variance (small perturbations in data can abruptly drop or retain variables). Lasso achieves variable selection via exact truncation at zero while preserving continuous transitions, resulting in significantly lower variance than best subset.

---

### 14. ESL 3.4.1 / Ex 3.8: Ridge SVD Spectral Shrinkage, Effective Degrees of Freedom, and Proof of Strict MSE Dominance over OLS

> **Theorem Derivation (Theobald 1974 Theorem / Ridge Strict MSE Dominance over OLS)**:
> Let the centered design matrix $X \in \mathbb{R}^{n \times p}$ (with full column rank $\operatorname{rank}(X) = p \le n$) have Singular Value Decomposition (SVD):
> $$ X = U D V^T $$
> where $U \in \mathbb{R}^{n \times p}$ satisfies $U^T U = I_p$, $V \in \mathbb{R}^{p \times p}$ is orthogonal, and $D = \operatorname{diag}(d_1, \dots, d_p)$ with $d_1 \ge d_2 \ge \dots \ge d_p > 0$.
> 1. Expand the Ridge fitted vector $\hat{Y}^{\text{ridge}} = X\hat\beta^{\text{ridge}}$ explicitly in terms of singular values $d_j$ and left singular vectors $u_j$, and analyze how Ridge shrinks different principal component directions;
> 2. Prove that the effective degrees of freedom $\operatorname{df}(\lambda) = \operatorname{tr}(H_\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$, and prove that it is strictly monotonically decreasing for $\lambda \ge 0$;
> 3. **Theobald (1974) Theorem**: For any true parameter vector $\beta$ and disturbance variance $\sigma^2$, **rigorously prove that there always exists a $\lambda^* > 0$ such that the total Mean Squared Error (MSE) of Ridge is strictly smaller than that of OLS**:
>    $$ \operatorname{MSE}(\hat\beta^{\text{ridge}}(\lambda^*)) < \operatorname{MSE}(\hat\beta^{\text{ols}}) $$

**Step-by-Step Derivation**:

#### 1. SVD Spectral Shrinkage Expansion
From $X = U D V^T$, we have $X^T X = V D^2 V^T$.
Substituting into the closed-form Ridge estimator:
$$
\begin{aligned}
\hat\beta^{\text{ridge}} &= (X^T X + \lambda I)^{-1} X^T Y \\
&= \left[ V (D^2 + \lambda I) V^T \right]^{-1} V D U^T Y \\
&= V (D^2 + \lambda I)^{-1} D U^T Y = V \operatorname{diag}\left( \frac{d_j}{d_j^2 + \lambda} \right) U^T Y
\end{aligned}
$$
The fitted value vector $\hat{Y}^{\text{ridge}} = X\hat\beta^{\text{ridge}}$ is:
$$
\hat{Y}^{\text{ridge}} = (U D V^T) V (D^2 + \lambda I)^{-1} D U^T Y = U \operatorname{diag}\left( \frac{d_j^2}{d_j^2 + \lambda} \right) U^T Y = \sum_{j=1}^p u_j \left( \frac{d_j^2}{d_j^2 + \lambda} \right) u_j^T Y
$$
- **Comparison with OLS**: OLS corresponds to $\lambda = 0$, where $\hat{Y}^{\text{ols}} = \sum_{j=1}^p u_j (u_j^T Y)$.
- **Physical Interpretation of Spectral Shrinkage**: The shrinkage factor along each principal component direction $u_j$ is $f_j = \frac{d_j^2}{d_j^2 + \lambda}$.
  - For high-variance principal components ($d_1^2 \gg \lambda$), $f_1 \approx 1$, virtually uncompressed;
  - For low-variance components ($d_p^2 \ll \lambda$, collinear directions), $f_p \to 0$, **heavily suppressed towards zero**;
  - Ridge regression acts as an adaptive low-pass filter in the principal component coordinate system, filtering out high-variance, collinear noise directions.

#### 2. Effective Degrees of Freedom
The hat matrix is $H_\lambda = U \operatorname{diag}\left( \frac{d_j^2}{d_j^2 + \lambda} \right) U^T$.
$$ \operatorname{df}(\lambda) = \operatorname{tr}(H_\lambda) = \operatorname{tr}\left( \operatorname{diag}\left( \frac{d_j^2}{d_j^2 + \lambda} \right) U^T U \right) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} $$
Differentiating with respect to $\lambda$:
$$ \frac{d}{d\lambda} \operatorname{df}(\lambda) = -\sum_{j=1}^p \frac{d_j^2}{(d_j^2 + \lambda)^2} < 0 \quad (\forall \lambda \ge 0) $$
Thus, $\operatorname{df}(\lambda)$ is strictly monotonically decreasing in $\lambda \ge 0$, with $\operatorname{df}(0) = p$ and $\lim_{\lambda \to \infty} \operatorname{df}(\lambda) = 0$.

#### 3. Proof of Theobald's Theorem: Ridge Strictly Dominates OLS in MSE
Mean Squared Error is decomposed into:
$$ \operatorname{MSE}(\hat\beta) = E[\|\hat\beta - \beta\|_2^2] = \operatorname{tr}(\operatorname{Var}(\hat\beta)) + \|\operatorname{Bias}(\hat\beta)\|_2^2 $$
- **Variance Term**:
  $$ \operatorname{Var}(\hat\beta^{\text{ridge}}) = \sigma^2 (X^T X + \lambda I)^{-1} X^T X (X^T X + \lambda I)^{-1} = \sigma^2 V \operatorname{diag}\left( \frac{d_j^2}{(d_j^2 + \lambda)^2} \right) V^T $$
  Its trace is: $\operatorname{tr}(\operatorname{Var}) = \sigma^2 \sum_{j=1}^p \frac{d_j^2}{(d_j^2 + \lambda)^2}$.
- **Bias Term**:
  $$ \operatorname{Bias}(\hat\beta^{\text{ridge}}) = E[\hat\beta^{\text{ridge}}] - \beta = -\lambda (X^T X + \lambda I)^{-1} \beta $$
  Let $\alpha = V^T \beta = (\alpha_1, \dots, \alpha_p)^T$ in the orthonormal eigenbasis:
  $$ \|\operatorname{Bias}\|^2 = \lambda^2 \beta^T V (D^2 + \lambda I)^{-2} V^T \beta = \lambda^2 \sum_{j=1}^p \frac{\alpha_j^2}{(d_j^2 + \lambda)^2} $$
- **Derivative Analysis of Total MSE with respect to $\lambda$**:
  $$ \operatorname{MSE}(\lambda) = \sum_{j=1}^p \frac{\sigma^2 d_j^2 + \lambda^2 \alpha_j^2}{(d_j^2 + \lambda)^2} $$
  Computing the derivative:
  $$
  \begin{aligned}
  \frac{d}{d\lambda} \operatorname{MSE}(\lambda) &= \sum_{j=1}^p \frac{2\lambda \alpha_j^2 (d_j^2 + \lambda)^2 - 2(d_j^2 + \lambda)(\sigma^2 d_j^2 + \lambda^2 \alpha_j^2)}{(d_j^2 + \lambda)^4} \\
  &= \sum_{j=1}^p \frac{2\lambda \alpha_j^2 (d_j^2 + \lambda) - 2(\sigma^2 d_j^2 + \lambda^2 \alpha_j^2)}{(d_j^2 + \lambda)^3} \\
  &= \sum_{j=1}^p \frac{2\lambda d_j^2 \alpha_j^2 - 2\sigma^2 d_j^2}{(d_j^2 + \lambda)^3}
  \end{aligned}
  $$
  Evaluating at $\lambda = 0$ (the OLS estimator):
  $$ \left. \frac{d}{d\lambda} \operatorname{MSE}(\lambda) \right|_{\lambda = 0} = \sum_{j=1}^p \frac{-2\sigma^2 d_j^2}{d_j^6} = -2\sigma^2 \sum_{j=1}^p \frac{1}{d_j^4} < 0 $$
  **Key Conclusion**: At $\lambda = 0$, the derivative of MSE with respect to $\lambda$ is **strictly negative**!
  Since $\operatorname{MSE}(\lambda)$ is continuously differentiable on $[0, \infty)$, by the definition of limits, there must exist some sufficiently small $\lambda^* > 0$ such that:
  $$ \operatorname{MSE}(\hat\beta^{\text{ridge}}(\lambda^*)) < \operatorname{MSE}(\hat\beta^{\text{ols}}) $$
  **Q.E.D.** While Gauss-Markov establishes that OLS has the lowest variance among all *unbiased* linear estimators, allowing a minute amount of bias ($\lambda > 0$) yields a variance reduction that strictly outweighs the bias penalty, guaranteeing MSE superiority.

---

### 15. ESL 6.1.1 / Ex 6.1–6.2: Local Linear Regression Equivalent Kernel Closed Form, Moment Conditions, and Boundary Bias Elimination

> **Theorem Derivation (ESL Ch.6 / Local Linear Regression Equivalent Kernel and Boundary Unbiasedness)**:
> In nonparametric regression with sample $(X_i, Y_i)_{i=1}^n$, local linear regression at query point $x_0$ minimizes:
> $$ \min_{\alpha, \beta} \sum_{i=1}^n K_h(X_i - x_0) \left[ Y_i - \alpha - \beta(X_i - x_0) \right]^2 $$
> where $K(u)$ is a symmetric probability kernel, $K_h(u) = \frac{1}{h} K(u/h)$, and the estimate is $\hat{f}(x_0) = \hat\alpha$.
> 1. Solve the weighted normal equations, prove that $\hat{f}(x_0) = \sum_{i=1}^n l_i(x_0) Y_i$, and derive the closed-form expression for the equivalent kernel weights $l_i(x_0)$ in terms of kernel moments $s_r(x_0) = \sum_{i=1}^n K_h(X_i - x_0)(X_i - x_0)^r$;
> 2. Rigorously prove that $l_i(x_0)$ automatically satisfies the zeroth and first moment conditions:
>    $$ \sum_{i=1}^n l_i(x_0) = 1, \quad \sum_{i=1}^n (X_i - x_0) l_i(x_0) = 0 $$
> 3. Assuming the true function $f(x)$ is twice continuously differentiable, prove why Nadaraya–Watson local constant regression suffers from an $O(h)$ boundary bias, whereas local linear regression automatically eliminates first-derivative bias, maintaining $O(h^2)$ bias even at domain boundaries.

**Step-by-Step Derivation**:

#### 1. Weighted Least Squares and Closed-Form Equivalent Kernel
Let $z_i = X_i - x_0$ and $w_i = K_h(z_i)$. The local design matrix and weighting diagonal matrix are:
$$ B = \begin{pmatrix} 1 & z_1 \\ 1 & z_2 \\ \vdots & \vdots \\ 1 & z_n \end{pmatrix} \in \mathbb{R}^{n \times 2}, \quad W = \operatorname{diag}(w_1, \dots, w_n) $$
The parameter vector is $(\hat\alpha, \hat\beta)^T = (B^T W B)^{-1} B^T W Y$.
Compute the weighted Gram matrix:
$$ B^T W B = \begin{pmatrix} \sum_{i=1}^n w_i & \sum_{i=1}^n w_i z_i \\ \sum_{i=1}^n w_i z_i & \sum_{i=1}^n w_i z_i^2 \end{pmatrix} = \begin{pmatrix} s_0(x_0) & s_1(x_0) \\ s_1(x_0) & s_2(x_0) \end{pmatrix} $$
Determinant is $D = s_0 s_2 - s_1^2$. The $2 \times 2$ inverse matrix is:
$$ (B^T W B)^{-1} = \frac{1}{s_0 s_2 - s_1^2} \begin{pmatrix} s_2 & -s_1 \\ -s_1 & s_0 \end{pmatrix} $$
The estimate is $\hat{f}(x_0) = \hat\alpha = e_1^T (B^T W B)^{-1} B^T W Y$. Taking the first row inner product:
$$
\begin{aligned}
\hat{f}(x_0) &= \frac{1}{s_0 s_2 - s_1^2} \begin{pmatrix} s_2 & -s_1 \end{pmatrix} \begin{pmatrix} \sum w_i Y_i \\ \sum w_i z_i Y_i \end{pmatrix} \\
&= \sum_{i=1}^n \left[ \frac{w_i (s_2 - s_1 z_i)}{s_0 s_2 - s_1^2} \right] Y_i
\end{aligned}
$$
Therefore, the closed-form equivalent kernel weights $l_i(x_0)$ are:
$$ \boxed{l_i(x_0) = \frac{K_h(X_i - x_0) \left[ s_2(x_0) - s_1(x_0)(X_i - x_0) \right]}{s_0(x_0) s_2(x_0) - s_1^2(x_0)}} $$

#### 2. Algebraic Proof of Moment Conditions
- **Zeroth Moment (Sum to 1)**:
  $$ \sum_{i=1}^n l_i(x_0) = \frac{s_2 \sum w_i - s_1 \sum w_i z_i}{s_0 s_2 - s_1^2} = \frac{s_2 s_0 - s_1 s_1}{s_0 s_2 - s_1^2} = \frac{s_0 s_2 - s_1^2}{s_0 s_2 - s_1^2} \equiv \boxed{1} $$
- **First Moment (Orthogonality to $(X_i - x_0)$)**:
  $$ \sum_{i=1}^n (X_i - x_0) l_i(x_0) = \sum_{i=1}^n z_i l_i(x_0) = \frac{s_2 \sum w_i z_i - s_1 \sum w_i z_i^2}{s_0 s_2 - s_1^2} = \frac{s_2 s_1 - s_1 s_2}{s_0 s_2 - s_1^2} \equiv \boxed{0} $$

#### 3. Boundary Bias Analysis via Taylor Expansion
Expand $f(X_i)$ in a second-order Taylor series around $x_0$:
$$ f(X_i) = f(x_0) + f'(x_0)(X_i - x_0) + \frac{1}{2} f''(x_0)(X_i - x_0)^2 + o((X_i - x_0)^2) $$
The conditional expectation is:
$$
\begin{aligned}
E[\hat{f}(x_0) \mid X] &= \sum_{i=1}^n l_i(x_0) f(X_i) \\
&= f(x_0) \underbrace{\sum_{i=1}^n l_i(x_0)}_{= 1} + f'(x_0) \underbrace{\sum_{i=1}^n (X_i - x_0) l_i(x_0)}_{= 0} + \frac{1}{2} f''(x_0) \sum_{i=1}^n (X_i - x_0)^2 l_i(x_0) + \dots \\
&= f(x_0) + \frac{1}{2} f''(x_0) \sum_{i=1}^n (X_i - x_0)^2 l_i(x_0) + O(h^3)
\end{aligned}
$$
**Comparison at Boundaries**:
- **Nadaraya-Watson (Local Constant)**:
  Weights are $l_i^{\text{NW}}(x_0) = \frac{w_i}{s_0}$.
  The first-moment term is $\sum z_i l_i^{\text{NW}} = \frac{s_1(x_0)}{s_0(x_0)}$.
  In the interior, symmetry implies $s_1 \approx 0$;
  At a boundary point (e.g. $x_0 = 0$ with data only on $X_i \ge 0$), the kernel is asymmetric and truncated, so $s_1 = \sum w_i z_i \sim O(h) \ne 0$.
  This produces a dominant $O(h)$ boundary bias:
  $$ \operatorname{Bias}_{\text{NW}}(0) = f'(0) \frac{s_1(0)}{s_0(0)} = \boxed{O(h)} $$
- **Local Linear Regression**:
  By embedding the local slope $\beta$, the equivalent kernel $l_i(x_0)$ **guarantees $\sum (X_i - x_0) l_i(x_0) \equiv 0$ everywhere**, including the boundaries!
  The first-derivative bias term vanishes algebraically, improving boundary bias to:
  $$ \operatorname{Bias}_{\text{LLR}}(0) = \frac{1}{2} f''(0) \sum_{i=1}^n z_i^2 l_i(0) = \boxed{O(h^2)} $$
  This is the celebrated property known in ESL as "**Automatic Kernel Carpentry**".

---

### 16. ESL 6.2 / Ex 6.3: Properties of Smoother Matrix $S_\lambda$, Two Types of Effective Degrees of Freedom, and Volatility Surface Fitting

> **Theorem Derivation (ESL Ch.6 / Linear Smoother Matrix Properties and Two Types of Effective Degrees of Freedom)**:
> All linear smoothers can be unified in matrix form: $\hat{Y} = S_\lambda Y$, where $S_\lambda \in \mathbb{R}^{n \times n}$ is the smoother matrix.
> 1. Prove that for non-uniformly distributed design points, the local polynomial smoother matrix $S_\lambda$ has row sums equal to 1 ($S_\lambda \mathbf{1} = \mathbf{1}$), but is **generally asymmetric** ($S_\lambda^T \ne S_\lambda$) and **non-idempotent** ($S_\lambda^2 \ne S_\lambda$);
> 2. Statistics defines two types of effective degrees of freedom: $\operatorname{df}_{\text{fit}} = \operatorname{tr}(S_\lambda)$ and $\operatorname{df}_{\text{var}} = \operatorname{tr}(S_\lambda S_\lambda^T)$. Explain their statistical meanings and prove that $\operatorname{df}_{\text{var}} \le \operatorname{df}_{\text{fit}}$ for symmetric smoothers;
> 3. When fitting option implied volatility surfaces in quantitative finance, what pitfalls arise from counting explicit parameters in AIC/BIC? How does Generalized Cross-Validation (GCV) resolve this?

**Step-by-Step Derivation**:

#### 1. Three Core Properties of the Smoother Matrix
1. **Row Sums Equal to 1 (Preserves Constants)**:
   If the response is a constant vector $Y = c \mathbf{1}$, local polynomial regression fits a constant perfectly without error, producing $\hat{Y} = c \mathbf{1}$.
   Hence $S_\lambda (c \mathbf{1}) = c (S_\lambda \mathbf{1}) = c \mathbf{1} \implies S_\lambda \mathbf{1} = \mathbf{1}$.
2. **Asymmetry ($S_\lambda^T \ne S_\lambda$)**:
   The entry $S_{ij} = l_j(X_i)$ is the weight of observation $j$ on the fit at $X_i$.
   $l_j(X_i)$ involves normalization centered at $X_i$ ($\sum_k K_h(X_k - X_i)$), while $l_i(X_j)$ involves normalization centered at $X_j$. Unless design points are uniformly spaced on a periodic grid, sample density variations ensure $S_{ij} \ne S_{ji}$.
3. **Non-Idempotence ($S_\lambda^2 \ne S_\lambda$)**:
   An orthogonal projection matrix (e.g., OLS hat matrix $H = X(X^T X)^{-1}X^T$) satisfies $H^2 = H$.
   A kernel smoother does not project onto a finite-dimensional subspace; smoothing an already smoothed sequence ($S_\lambda(S_\lambda Y)$) applies a second low-pass filter, flattening the curve further: $S_\lambda^2 \ne S_\lambda$.

#### 2. Statistical Meanings of the Two Degrees of Freedom and Inequality Proof
- **$\operatorname{df}_{\text{fit}} = \operatorname{tr}(S_\lambda)$ (Fit / Efron's Degrees of Freedom)**:
  Under independent homoskedastic errors ($\operatorname{Var}(Y) = \sigma^2 I$), consider the total covariance between predictions and observed targets:
  $$ \sum_{i=1}^n \frac{\operatorname{Cov}(\hat{Y}_i, Y_i)}{\sigma^2} = \sum_{i=1}^n \frac{\operatorname{Cov}\left( \sum_{j=1}^n S_{ij} Y_j, \, Y_i \right)}{\sigma^2} = \sum_{i=1}^n \frac{S_{ii} \sigma^2}{\sigma^2} = \sum_{i=1}^n S_{ii} = \operatorname{tr}(S_\lambda) $$
  It measures the **average self-sensitivity / degrees of freedom consumed in fitting data**.
- **$\operatorname{df}_{\text{var}} = \operatorname{tr}(S_\lambda S_\lambda^T)$ (Variance Degrees of Freedom)**:
  Summing the variances of all fitted values:
  $$ \sum_{i=1}^n \frac{\operatorname{Var}(\hat{Y}_i)}{\sigma^2} = \frac{1}{\sigma^2} \operatorname{tr}(\operatorname{Var}(S_\lambda Y)) = \frac{1}{\sigma^2} \operatorname{tr}(S_\lambda (\sigma^2 I) S_\lambda^T) = \operatorname{tr}(S_\lambda S_\lambda^T) $$
  It measures the total variance consumption of the estimator.

**Proof that $\operatorname{df}_{\text{var}} \le \operatorname{df}_{\text{fit}}$ (for Symmetric Smoothers)**:
For symmetric smoothers (such as smoothing splines), $S_\lambda$ is real symmetric with eigenvalues $\gamma_1, \dots, \gamma_n$.
Because smoothing operations are shrinkage filters, all eigenvalues satisfy $0 \le \gamma_i \le 1$.
$$ \operatorname{df}_{\text{fit}} = \operatorname{tr}(S_\lambda) = \sum_{i=1}^n \gamma_i $$
$$ \operatorname{df}_{\text{var}} = \operatorname{tr}(S_\lambda S_\lambda^T) = \operatorname{tr}(S_\lambda^2) = \sum_{i=1}^n \gamma_i^2 $$
Since $\gamma_i \in [0, 1]$, we have $\gamma_i^2 \le \gamma_i$. Thus:
$$ \operatorname{df}_{\text{var}} = \sum_{i=1}^n \gamma_i^2 \le \sum_{i=1}^n \gamma_i = \operatorname{df}_{\text{fit}} $$
Equality holds if and only if every non-zero eigenvalue equals 1 (i.e. $S_\lambda$ is an orthogonal projection matrix, degenerating to OLS)!

#### 3. Quant Finance Implication: Implied Volatility Surface Fitting
- **Parameter Counting Pitfall**: In option market making, when smoothing the implied volatility (IV) surface across strike $K$ and maturity $T$, local polynomials do not have an explicit parameter count $k$. Setting $k$ to an arbitrary integer causes standard AIC/BIC to fail completely.
- **Automated GCV Regularization**: To handle microstructure noise in quote data, one uses **Generalized Cross-Validation (GCV)** to select the optimal bandwidth $h$:
  $$ \mathrm{GCV}(h) = \frac{\frac{1}{n} \|Y - \hat{Y}\|_2^2}{\left( 1 - \frac{\operatorname{tr}(S_h)}{n} \right)^2} $$
  The trace $\operatorname{tr}(S_h)$ serves as the effective parameter count, penalizing undersmoothed models and preventing spurious arbitrage humps in the fitted IV surface.

---

## Module 6: Core Knowledge Checklist & Diagnostic Traps

```text
Core Knowledge Checklist for Regression & Smoothing Models:
1. Univariate OLS Estimator: Slope \hat\beta = \rho \cdot (\sigma_y / \sigma_x), goodness of fit R^2 = \rho^2.
2. Reverse Regression & Mean Reversion: The product of forward and reverse regression slopes is \rho^2 \le 1; diluted by random noise, never invert directly.
3. BLUE Conditions & Normality Boundaries: The Gauss-Markov theorem requires only first-order exogeneity and second-order spherical disturbances; normality is required only for finite-sample exact t/F tests and achieving UMVUE.
4. Consequences of Spherical Violation: Under heteroskedasticity or autocorrelation, OLS estimators remain unbiased and consistent, but standard covariance is underestimated (creating spurious significance); White (HC0) or Newey-West (HAC) robust standard errors must be used.
5. Regularization Geometric Mechanisms: Lasso's \ell_1 contours feature non-smooth vertices that tend to intersect residual contours on coordinate axes (producing sparse solutions); Ridge's \ell_2 contours form smooth hyperspheres that smoothly shrink spectral components along low-variance principal directions.
6. Kernel Smoothing Boundary Bias & High-Dimensional Extension: Nadaraya-Watson (local constant kernel estimation) suffers from O(h) boundary bias; local linear regression automatically satisfies the first-order orthogonality moment, reducing boundary bias order to O(h^2); high-dimensional extensions mitigate the curse of dimensionality via Generalized Additive Models (GAM) or varying-coefficient models.
```

---


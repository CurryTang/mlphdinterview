# ML Coding 00 · ML Basics: Data Preprocessing, Data Leakage & Loss Functions

In machine learning system design and production engineering, a solid statistical foundation and rigorous data pipeline practices are essential prerequisites for building dependable models. Many machine learning models exhibit stellar offline evaluation metrics only to degrade catastrophically upon production rollout. The root causes rarely lie in model architectures, but rather in insidious data leakage, flawed missing data imputation, or a misalignment between loss function assumptions and problem characteristics.

This note systematically covers 3 foundational pillars of practical machine learning:
1. **Data Leakage Mechanisms & End-to-End Prevention Strategies**
2. **Missing Data Mechanisms (MCAR / MAR / MNAR) & Imputation Trade-Offs**
3. **Loss Function Derivations: Linear vs. Logistic Regression, MSE vs. MAE & Statistical Convergence Targets**

---

## Module 1: Data Leakage Mechanisms & Prevention

### 1. The Nature and Danger of Data Leakage

**Data leakage** occurs when **information from outside the training dataset (especially from the target variable or future test data) is inadvertently used to train a machine learning model**.

```text
Data Leakage Lifecycle & Impact:
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Training / Offline Eval │ ───> │ Overly Optimistic Eval  │ ───> │ Production Deployment   │
│ Inadvertent look-ahead  │      │ Validation AUC 0.98+    │      │ Leaked signal missing   │
│ or target contamination │      │ (Spurious Correlations) │      │ Catastrophic drop 💥    │
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

Leakage creates an illusion of high predictive performance during offline experimentation. However, because the leaked information is unavailable at inference time in real-world systems, the model suffers a severe drop in production accuracy.

---

### 2. Four Canonical Data Leakage Scenarios

#### Scenario 1: Target Leakage / Proxy Features

**Mechanism**: Including a feature that is created, computed, or updated only **after** the target event occurs, but is mistakenly included in historical training records.

- **Case 1 (Loan Default Prediction)**: Using `account_closed_date` or `refund_status_code` as predictive features for whether a borrower will default. In real banking workflows, these columns are populated only after default proceedings and collections take place.
- **Case 2 (Medical Diagnosis)**: Using `prescribed_treatment_drug` to predict if a patient has a rare disease. Doctors prescribe the drug only after diagnosis; using it as an input inverts the causal graph.

#### Scenario 2: Preprocessing Leakage (Global Scaling & Imputation)

**Mechanism**: Computing preprocessing statistics (mean, variance, min-max bounds, TF-IDF vocabulary, or target encoding) across the **entire dataset** prior to splitting into train/test sets.

- **Case 1 (Feature Standardization)**: Calling `StandardScaler().fit_transform(X)` on the global dataset before `train_test_split`. Test set distribution parameters leak into training features.
- **Case 2 (NLP Vocabulary & TF-IDF)**: Fitting `TfidfVectorizer` globally, allowing inverse document frequency (IDF) and vocabulary tokens from the test split to contaminate training representations.
- **Case 3 (Target Encoding)**: Calculating category-level target averages on the full dataset without out-of-fold isolation, enabling the model to directly memorize test set target distributions.

#### Scenario 3: Temporal / Look-Ahead Leakage in Time Series

**Mechanism**: Using future timestamps to predict past events, violating the temporal arrow of causality.

- **Case 1 (Quantitative Finance / Market Prediction)**: Using a centered 5-day rolling moving average as a feature for today's trading signal.
- **Case 2 (Flawed Cross-Validation)**: Applying standard randomized K-Fold cross-validation to user activity logs or financial time series. Testing data from Day 1 is evaluated using training data from Day 5, obscuring non-stationarity and concept drift.

#### Scenario 4: Group / Duplication Leakage

**Mechanism**: Multiple highly correlated or repeated samples belonging to the **same entity (subject, patient, user session)** are randomly distributed across both train and test splits.

- **Case 1 (Medical Imaging)**: A patient undergoes 10 CT scans from varying angles. A randomized split assigns 8 scans to train and 2 to test. The CNN memorizes patient-specific artifacts (e.g., bone density, scanner noise) rather than generalized pathology.
- **Case 2 (Multi-Session Recommenders)**: User clicks from the same session are partitioned across train and test.

---

### 3. Industrial Data Leakage Prevention Blueprint

| Prevention Strategy | Implementation Principle | Recommended Tooling | Leakage Type Addressed |
|---|---|---|---|
| **Split First, Fit Later** | Enforce dataset partitioning before computing any statistical transformations. `fit()` exclusively on training data; `transform()` on test data. | `sklearn.pipeline.Pipeline`, `ColumnTransformer` | Preprocessing Leakage |
| **Time-Based Splitting** | Enforce strict chronological ordering. Train on historical windows; validate strictly on forward-looking out-of-time (OOT) sets. | `TimeSeriesSplit`, `PurgedGroupTimeSeriesSplit` | Temporal Leakage |
| **Group-Aware Partitioning** | Keep all records belonging to a given entity (user, patient, device) strictly inside either the train or test set. | `GroupKFold`, `GroupShuffleSplit`, `StratifiedGroupKFold` | Group / Subject Leakage |
| **Inference Timeline Audit** | Ask: "At the exact millisecond an inference request arrives in production, is this feature available in the feature store/DB?" | Feature Store (e.g., Feast), Data Lineage systems | Target & Proxy Leakage |

---

### 4. Quick Coding: Leak-Free Pipeline with GroupKFold

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# 1. Generate synthetic grouped tabular data with missing values
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
groups = np.repeat(np.arange(100), 10)  # 100 distinct entities, 10 records each
X[np.random.rand(*X.shape) < 0.1] = np.nan  # Inject 10% missingness

# 2. Build leak-free pipeline encapsulating imputation, scaling, and estimator
model_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42))
])

# 3. Perform GroupKFold cross-validation
gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(y))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # fit only touches the current training fold
    model_pipeline.fit(X_train, y_train)
    oof_preds[val_idx] = model_pipeline.predict_proba(X_val)[:, 1]

cv_auc = roc_auc_score(y, oof_preds)
print(f"Leak-Free GroupKFold 5-Fold OOF AUC: {cv_auc:.4f}")
```

---

## Module 2: Handling Missing Data: Strategies & Trade-Offs

### 1. Statistical Missingness Mechanisms

Rubin's classification framework categorizes missingness into three distinct regimes:

```text
Missing Data Regimes:
┌──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Mechanism Category                   │ Mathematical Definition & Interpretation                               │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 1. MCAR (Missing Completely at Random)│ P(M | Y_obs, Y_mis) = P(M)                                             │
│                                      │ Missingness is entirely independent of observed and unobserved data.    │
│                                      │ (e.g., dropped sensor packets, randomly damaged paper forms).          │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 2. MAR (Missing at Random)           │ P(M | Y_obs, Y_mis) = P(M | Y_obs)                                     │
│                                      │ Missingness depends on observed features, but not on missing value.    │
│                                      │ (e.g., older patients less often report phone numbers, but conditional │
│                                      │ on age, missingness is independent of the phone number itself).        │
├──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 3. MNAR (Missing Not at Random)      │ P(M | Y_obs, Y_mis) depends on Y_mis                                   │
│                                      │ Missingness directly depends on the unobserved value itself.           │
│                                      │ (e.g., very high/low income earners declining to disclose income).     │
└──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

### 2. Comprehensive Missing Data Strategy Comparison

| Strategy | When to Use | Advantages (Pros) | Trade-Offs & Disadvantages (Cons) |
|---|---|---|---|
| **Listwise / Column Deletion** | MCAR with low missingness (<3%~5%); or when a column is >80% empty. | Simple; introduces no synthetic bias if truly MCAR. | Severe data loss; induces strong **selection bias** if data is MAR or MNAR. |
| **Simple Imputation (Mean / Median / Mode)** | Fast baseline; low missing rate on numeric/categorical features. | Extremely lightweight; easy online deployment. | **Distorts distribution**, artificially deflates feature variance, ignores cross-feature covariance. |
| **Missing Indicator (`is_missing`)** | MNAR scenarios where omission carries predictive signal (e.g., skipped optional credit check). | Explicitly preserves the informational signal of missingness. | Doubles the feature dimensionality if applied naively; potential collinearity. |
| **Model-Based Imputation (KNN, MICE / Iterative, MissForest)** | High-value tabular data with non-linear feature correlations. | Preserves multivariate distributions, variance, and feature interactions. | Computationally expensive; complex inference deployment; risk of cascading errors. |
| **Native Tree Routing (LightGBM, XGBoost, CatBoost)** | Gradient boosted decision trees. | Zero manual imputation; finds optimal split direction for missing values natively. | Restricted to specific tree libraries; unusable for neural nets or linear models. |

---

### 3. Deep Statistical Trade-Off Analysis

1. **Variance Deflation under Mean Imputation**:
   When replacing missing entries with the sample mean $x_{\text{imputed}} = \bar{x}$, the imputed variance drops:

$$\text{Var}(X_{\text{imputed}}) = \frac{N_{\text{obs}}}{N_{\text{total}}} \text{Var}(X_{\text{obs}}) < \text{Var}(X_{\text{obs}})$$

   This deflation produces biased standard errors, narrower confidence intervals, and skewed regression weights.
2. **MICE (Multivariate Imputation by Chained Equations)**:
   MICE iteratively specifies a univariate conditional model for each variable with missing data given all other variables, adding residual stochastic noise to preserve true uncertainty.

---

### 4. Quick Coding: Missing Indicator Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 1. Synthetic dataset with MNAR income missingness
df = pd.DataFrame({
    'age': [25, 30, np.nan, 45, 50, np.nan, 60],
    'income': [50000, np.nan, 120000, 80000, np.nan, 200000, 95000],
    'credit_score': [650, 700, 750, 680, 710, 790, 720]
})
y = np.array([0, 1, 0, 1, 0, 1, 0])

# 2. Composite transformer: Imputed values + Boolean Missing Indicators
numeric_features = ['age', 'income', 'credit_score']

numeric_transformer = FeatureUnion([
    ('imputed_features', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])),
    ('missing_indicators', MissingIndicator())
])

preprocessor = ColumnTransformer(
    transformers=[('num', numeric_transformer, numeric_features)]
)

full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', Ridge())
])

full_pipeline.fit(df, y)
print("Pipeline fitted successfully. Output shape (with is_missing flags):", 
      full_pipeline.named_steps['preprocess'].transform(df).shape)
```

---

## Module 3: Loss Functions: Linear vs. Logistic Regression, and MSE vs. MAE

### 1. Linear Regression Objective & Gaussian MLE Derivation

Linear regression optimizes the Ordinary Least Squares (OLS) / Mean Squared Error (MSE) loss:

$$\mathcal{L}_{\text{Linear}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

#### Probabilistic Derivation via Gaussian MLE

Assume the true target $y_i$ is generated by a linear function with additive Gaussian residual noise $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:

$$y_i = \mathbf{w}^T \mathbf{x}_i + \epsilon_i \implies y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^T \mathbf{x}_i, \sigma^2)$$

The likelihood of observing sample $y_i$ is:

$$p(y_i \mid \mathbf{x}_i; \mathbf{w}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(y_i - \mathbf{w}^T \mathbf{x}_i)^2}{2\sigma^2} \right)$$

The total log-likelihood $\ell(\mathbf{w})$ across $N$ i.i.d. observations is:

$$\ell(\mathbf{w}) = -\frac{N}{2} \ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

Maximizing log-likelihood $\max_{\mathbf{w}} \ell(\mathbf{w})$ is mathematically equivalent to **minimizing the Mean Squared Error (MSE)**:

$$\arg\max_{\mathbf{w}} \ell(\mathbf{w}) \iff \arg\min_{\mathbf{w}} \frac{1}{N} \sum_{i=1}^N (y_i - \mathbf{w}^T \mathbf{x}_i)^2$$

---

### 2. Logistic Regression Objective & Bernoulli MLE Derivation

For binary classification $y_i \in \{0, 1\}$, logistic regression models posterior probability via the sigmoid function:

$$\hat{p}_i = \sigma(\mathbf{w}^T \mathbf{x}_i) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x}_i}}$$

Assuming $y_i \mid \mathbf{x}_i \sim \text{Bernoulli}(\hat{p}_i)$, the probability mass function is:

$$P(y_i \mid \mathbf{x}_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}$$

The dataset log-likelihood is:

$$\ell(\mathbf{w}) = \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

Negating and scaling by $1/N$ gives the Binary Cross-Entropy / Log Loss objective:

$$\mathcal{L}_{\text{Logistic}}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

---

### 3. Why Not MSE for Logistic Regression?

A common interview question asks why we avoid using MSE directly on sigmoid probabilities:

$$\mathcal{L}_{\text{MSE-Logistic}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N (y_i - \sigma(\mathbf{w}^T \mathbf{x}_i))^2$$

**Three Fundamental Reasons**:

#### Reason 1: Non-Convex Optimization Landscape

- **Log Loss** with linear parameters $\mathbf{w}$ is strictly **convex**, guaranteeing that gradient descent reaches the global minimum.
- **MSE** compounded with the non-linear sigmoid $\sigma(z)$ produces a **non-convex** error surface with flat plateaus, saddle points, and suboptimal local minima.

#### Reason 2: Vanishing Gradients on Severe Errors

Let $z_i = \mathbf{w}^T \mathbf{x}_i$. Compare parameter gradients under both objectives:

1. **MSE Gradient**:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial \mathbf{w}} = \frac{2}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \cdot \hat{p}_i(1 - \hat{p}_i) \mathbf{x}_i$$

   **Critical Flaw**: If the model makes a severe error (e.g., true $y_i = 1$, but predicted $\hat{p}_i = 0.0001$):
   - The error term $(\hat{p}_i - y_i) \approx -1$ is large.
   - However, the derivative term $\hat{p}_i(1 - \hat{p}_i) \approx 0.0001 \to 0$.
   - Their product causes the gradient to **vanish to zero**! The model stops learning precisely when it is most wrong.

2. **Log Loss (BCE) Gradient**:

$$\frac{\partial \mathcal{L}_{\text{BCE}}}{\partial \mathbf{w}} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i) \mathbf{x}_i$$

   **Optimal Behavior**: The sigmoid derivative in the chain rule cancels out the denominator of the log derivative. The gradient is directly proportional to the residual $(\hat{p}_i - y_i)$. Large errors yield large gradients.

#### Reason 3: Probability Calibration

Log loss corresponds to the proper scoring rule under Bernoulli MLE, producing outputs that converge to true posterior probabilities $P(Y=1 \mid X)$.

---

### 4. MSE vs. MAE: Statistical Convergence & Trade-Offs

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2 \quad \text{vs.} \quad \text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

| Dimension | Mean Squared Error (MSE / L2) | Mean Absolute Error (MAE / L1) |
|---|---|---|
| **Outlier Sensitivity** | **Highly sensitive**. Squares residuals, disproportionately penalizing large deviations. | **Robust**. Scales linearly with error; outliers do not dominate the gradient. |
| **Optimization & Smoothness** | **Smooth & differentiable everywhere**. Gradients shrink gracefully near the minimum. | **Non-differentiable at $e=0$**. Gradient is a step function ($\pm 1$), causing oscillations around the optimum without learning rate decay. |
| **Statistical Target** | Converges to the **Conditional Mean**: <br>$$\hat{y}^* = \mathbb{E}[y \mid \mathbf{x}]$$ | Converges to the **Conditional Median**: <br>$$\hat{y}^* = \text{Median}(y \mid \mathbf{x})$$ |

#### Proof: MSE Minimizes to Mean, MAE Minimizes to Median

1. **MSE converges to Mean**:
   Minimize $\min_c \mathbb{E}[(Y - c)^2]$. Differentiating w.r.t. $c$:

$$\frac{d}{dc} \mathbb{E}[(Y - c)^2] = \mathbb{E}[-2(Y - c)] = -2\mathbb{E}[Y] + 2c = 0 \implies c^* = \mathbb{E}[Y]$$

2. **MAE converges to Median**:
   Minimize $\min_c \mathbb{E}[|Y - c|]$. Applying Leibniz's rule:

$$\frac{d}{dc} \left( \int_{-\infty}^c (c - y) p(y)dy + \int_c^{\infty} (y - c) p(y)dy \right) = P(Y \le c) - P(Y > c) = 0 \implies c^* = \text{Median}(Y)$$

---

### 5. The Hybrid Compromise: Huber Loss

**Huber Loss** combines the smoothness of MSE near zero with the outlier robustness of MAE for large errors:

$$\mathcal{L}_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{for } |e| \le \delta \\ \delta \left(|e| - \frac{1}{2}\delta\right) & \text{for } |e| > \delta \end{cases}$$

```text
Gradient behavior comparison as error e -> infinity:
• MSE Gradient:   2e  ──> Grows unboundedly (risk of exploding gradients)
• MAE Gradient:   ±1  ──> Constant (discontinuous at 0)
• Huber Gradient: ±δ  ──> Bounded and smooth at 0!
```

---

### 6. Quick Coding: Custom Loss Functions & PyTorch Alignment

```python
import torch
import torch.nn as nn

def custom_mse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)

def custom_mae_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_pred - y_true))

def custom_bce_loss(y_prob: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    y_prob = torch.clamp(y_prob, min=eps, max=1.0 - eps)
    return -torch.mean(y_true * torch.log(y_prob) + (1.0 - y_true) * torch.log(1.0 - y_prob))

def custom_huber_loss(y_pred: torch.Tensor, y_true: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.tensor(delta))
    linear = abs_error - quadratic
    return torch.mean(0.5 * quadratic ** 2 + delta * linear)

# Verification against native PyTorch implementations
y_t = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)
y_p = torch.tensor([0.9, 0.2, 0.8, 0.4], dtype=torch.float32)

assert torch.allclose(custom_mse_loss(y_p, y_t), nn.MSELoss()(y_p, y_t))
assert torch.allclose(custom_mae_loss(y_p, y_t), nn.L1Loss()(y_p, y_t))
assert torch.allclose(custom_bce_loss(y_p, y_t), nn.BCELoss()(y_p, y_t))
assert torch.allclose(custom_huber_loss(y_p, y_t, delta=1.0), nn.HuberLoss(delta=1.0)(y_p, y_t))

print("✅ All custom loss functions matched PyTorch native references!")
```

---

## Module 4: Interview Cheatsheet & Rapid FAQ

### Q1: How do you design cross-validation when train and test distributions differ (Covariate Shift)?
> **Answer**:
> 1. Run **Adversarial Validation**: Train a binary classifier (e.g., LightGBM) to distinguish between train ($y=0$) and test ($y=1$).
> 2. If AUC $\gg 0.5$, use the predicted probabilities for **Importance Weighting** ($w(x) = \frac{p_{\text{test}}(x)}{p_{\text{train}}(x)}$) or select training instances with the highest test similarity to build the validation set.

### Q2: Why is Target Encoding prone to catastrophic leakage, and how is it prevented?
> **Answer**:
> Using the global target average for a category leaks a sample's own ground-truth label into its feature representation. Standard prevention requires **Out-of-Fold (OOF) Target Encoding** (calculating target statistics strictly on the remaining $K-1$ folds) combined with empirical Bayes smoothing and additive Gaussian jitter.

### Q3: Why is MAE preferred over MSE for training models on datasets with severe label noise?
> **Answer**:
> MSE squares errors, allowing mislabeled samples with huge residuals to dominate the loss and pull the regression curve off trajectory. MAE penalizes errors linearly and optimizes for the conditional median, which has a higher statistical breakdown point against extreme outliers.

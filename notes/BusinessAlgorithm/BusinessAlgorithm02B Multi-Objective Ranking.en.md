# Multi-Objective Learning and Score Fusion

## Chapter 10: Multi-Objective Learning and Score Fusion

### 10.1 Why Multi-Objective?

Short-video platforms may simultaneously care about clicks, watch time, completion, likes, follows, and negative feedback. E-commerce cares about clicks, add-to-cart, orders, and GMV. Search must also consider relevance, quality, timeliness, geography, and personalization.

Roughly summing all objectives into one label loses structure. Training multiple models separately leads to redundant computation and causes low-frequency tasks to lack data. Multi-task learning finds a balance between these two extremes.

### 10.2 Shared-Bottom

The simplest structure shares the bottom layers:

```text
features -> shared network -> task A tower
                           -> task B tower
                           -> task C tower
```

Total loss:

```math
\mathcal L=\sum_t \lambda_t\mathcal L_t.
```

The problem is that task gradients may conflict. Click preference favors title attractiveness, while long watch time favors sustained content value; they do not always update shared parameters in the same direction.

### 10.3 MMoE

MMoE uses multiple experts to generate representations, with each task having its own gate:

```math
h_t(x)
=\sum_{e=1}^{E}g_{t,e}(x)f_e(x),
```

```math
g_t(x)=\operatorname{softmax}(W_t x).
```

Task `t` dynamically combines experts based on the sample. It is more flexible than Shared-Bottom, but do not interpret the gate as a stable business division of labor. A specific expert does not necessarily represent "clicks" permanently, and the gate may collapse into a few experts.

A small amount of expert dropout on the gate output can reduce polarization: mask some experts, renormalize the gate, and stop a task from always following one path. This does not replace load monitoring, and aggressive dropout can destroy useful specialization.

When diagnosing MMoE, one can look at:

- The entropy of each task's gate;
- Whether expert usage is balanced;
- Task gradient cosine similarity;
- Gains from single-task vs. multi-task grouping;
- Whether low-frequency tasks are suppressed by high-frequency tasks.

### 10.4 ESMM and the Conversion Funnel

E-commerce CVR is only observable after a click. If CVR is trained using only click samples, the training distribution differs from the full exposure distribution.

ESMM utilizes:

```math
P(\text{click and conversion})
=P(\text{click})P(\text{conversion}\mid\text{click}),
```

It jointly learns CTR and CTCVR in the full exposure space, then constrains CVR through their relationship. It alleviates sample selection bias and conversion sparsity, but still relies on model assumptions and data definitions, and does not mean the counterfactual problem is completely solved.

### 10.5 Duration Modeling

Watch time is zero-inflated and influenced by video length. Directly regressing seconds will bias toward long videos.

Optional approaches:

- Predict effective views and conditional duration;
- Log-transform or bucketize duration;
- Predict the watch ratio;
- Calibrate by video length;
- Model exit using survival/hazard analysis.

One YouTube-style objective maps observed watch seconds `t` to a soft label:

```math
y=\frac{t}{1+t},\qquad p=\sigma(z).
```

Train `p` against `y` with binary cross-entropy. When `p=y`, `e^z=t`, so `e^z` is the duration estimate at inference. Completion can instead regress watch ratio or classify an event such as "watched more than 80%." Both need length-based calibration because short videos are easier to complete.

When evaluating, bucket by content length, user activity, and scenario. An increase in average duration might just mean the system pushed more long videos.

### 10.6 Score Fusion

Model outputs usually cannot be added linearly. CTR might be in `[0, 0.2]`, duration prediction is in seconds, and CVR is even sparser. Calibrate first, then discuss fusion.

Common form:

```math
S
=w_1f_1(\hat p_{\text{click}})
+w_2f_2(\hat t)
+w_3f_3(\hat p_{\text{conversion}})
-w_4\hat p_{\text{negative}}.
```

`f_t` can be a log, power function, piecewise function, or quantile mapping. Weights do not rely solely on offline search; they ultimately require online experimentation.

Several common fusion forms behave differently:

```math
S_{\text{add}}
=p_{\text{click}}+w_1p_{\text{like}}+w_2p_{\text{share}}+\cdots,
```

```math
S_{\text{rank}}
=\sum_j\frac{w_j}{r_j+\beta_j},
```

```math
S_{\text{commerce}}
=p_{\text{click}}^{\alpha}
\times p_{\text{cart}}^{\beta}
\times p_{\text{pay}}^{\gamma}
\times \operatorname{price}^{\delta}.
```

The additive form depends on calibrated scales. Rank fusion is more scale-robust but discards score gaps. The multiplicative e-commerce form follows the exposure-to-payment funnel and strongly suppresses an item when any stage is near zero.

Another path is to learn a fusion model, taking scores from each objective and context as input. However, it still requires training labels and is harder to interpret regarding objective trade-offs. Strong business constraints are best kept in the re-ranking or rule layer.

### 10.7 Calibration

If a model says 0.2, and the samples have approximately 20% actual clicks, the score is calibrated. Common methods:

- Platt scaling;
- Isotonic regression;
- Temperature scaling;
- Scenario-based or population-based calibration.

Ranking only requires relative order, but fusion often requires comparable probabilities. Calibration changes do not necessarily change AUC, but they can significantly change the results of multi-objective fusion.

Negative downsampling also requires probability correction. If only an `\alpha` fraction of negatives is retained and the sampled-data estimate is `p_s`, the original-distribution probability is:

```math
p
=\frac{\alpha p_s}
{1-p_s+\alpha p_s}.
```

Downsampling without this correction systematically inflates CTR and downstream rates, and makes fusion weights depend on the sampling ratio.

### 10.8 From Ranking Loss to Preference Optimization

BCE judges a single pair, BPR compares a pair of items, and InfoNCE makes one positive example compete with a set of candidates. All three utilize positive/negative feedback, but differ in comparison granularity and negative sample sources.

Generative recommendation extends the comparison unit to tokens or complete sequences. Next-token CE competes with the entire vocabulary, DPO compares chosen/rejected sequences, and policy gradient uses advantage to weight rollouts. Low-advantage RL rollouts cannot simply be treated as fixed negative samples because candidates are generated by the current policy, and sample weights change with training. For details, see [[BusinessAlgorithm05 Generative Recommendation.md#18.9 From Positive/Negative Samples to RL|Preference Optimization in Generative Recommendation]].

### 10.9 Chapter Self-Test

1. Where does negative transfer in Shared-Bottom come from?
2. How can MMoE gates be diagnosed?
3. Which two problems does ESMM solve for CVR?
4. Why does directly predicting watch seconds bias toward long videos?
5. When AUC is unchanged, why might calibration still improve online fusion?
6. How do the comparison granularities of BCE, BPR, and InfoNCE differ?

<details>
<summary>Reference answers</summary>

1. Shared parameters receive gradients from multiple tasks; gradients may conflict, and a high-volume task may dominate updates.
2. Inspect gate distributions, entropy, expert load, and specialization by task and slice. Constant or identical gates are warning signs.
3. ESMM models exposure-to-click and click-to-conversion jointly, reducing CVR sample-selection bias and sparsity from clicked-only training.
4. Watch-time scale grows with video length, so the model may learn length instead of satisfaction. Use completion rate, buckets, or normalization.
5. AUC only measures order. Fusion needs comparable probability scales, and calibration stops one head from dominating through numerical scale alone.
6. BCE classifies one sample, BPR compares one positive-negative pair, and InfoNCE compares the positive against a set of candidates.

</details>

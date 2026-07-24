# Cold Start, Exploration, and Long-Term Feedback

## Chapter 11: Cold Start, Exploration, and Long-term Feedback

### 11.1 Embeddings Only Provide Initial Estimates

New items have no interactions, so collaborative filtering cannot find them; new users have no history, so personalized models have no basis for judgment. Even if content encoders are used to generate embeddings, the system still lacks real consumption feedback.

Cold start requires solving two problems:

1. Providing initial estimates using existing information;
2. Obtaining real feedback under controlled risk.

Doing only the first without exploration makes it difficult for the model to correct initial misjudgments.

### 11.2 New Item Candidates

Common channels include:

- Category, tags, author, and region;
- Text/image/multimodal similarity;
- Clustering and placement into matching interest clusters;
- Look-alike audience expansion;
- New item pools after operational or quality review.

Look-alike starts from a small number of seed users to find populations with similar profiles or behaviors. It depends on seed quality and can easily amplify early biases.

### 11.3 New Users

For new users, one can utilize:

- Region, language, device, and entry point;
- First-screen interactions;
- Explicit interest selection;
- Contextual popularity;
- Similar anonymous sessions.

The most valuable data for a new user is their first few actions. The system should quickly update short-term interests and avoid showing default popular content for too long.

### 11.4 Where to Inject Traffic

Cold-start candidates can be injected at different layers:

- Recall injection: Low risk, but likely to be eliminated by coarse and fine ranking;
- Post-coarse-ranking quota: Easier to gain exposure, but requires higher quality control;
- Post-fine-ranking insertion: Stable exposure, but may significantly harm the main list;
- Reranking quota: Most controllable, but requires clear budgets and guardrails.

The later the injection, the more guaranteed the exposure, but the higher the risk of damaging the main list. High-quality review pools can be guaranteed in reranking; new content of unknown origin is better suited for small-scale traffic testing at the recall layer.

### 11.5 Exploration vs. Exploitation

Exploitation selects the content currently estimated to be the best, while exploration selects content that is uncertain but potentially promising.

A simple UCB form:

```math
\operatorname{UCB}_i
=\hat\mu_i
+c\sqrt{\frac{\log t}{n_i}}.
```

`μ_i` is the current utility estimate, and `n_i` is the number of exposures. Items with fewer exposures receive an uncertainty bonus.

Utility in recommendations depends on user context, items expire, and feedback has delays and position effects, so the UCB formula only indicates the direction of exploration. Even without bandits, one must provide a small amount of controlled traffic to high-uncertainty candidates; otherwise, the system cannot collect the data needed to correct initial estimates.

### 11.6 Cold-Start Evaluation

Overall CTR will be drowned out by mature items. At least look at the following separately:

- Proportion of new items entering recall;
- Waiting time for first exposure;
- Traffic required to reach minimum feedback volume;
- Stratified CTR/duration/negative feedback for new items;
- Damage to the main list caused by exploration;
- Proportion of high-quality new items retained after exploration.

Also, stratify by author, category, and quality priors. Mixing all new items together and averaging them will make the strategy appear erratic.

### 11.7 Feedback Loops

Recommendation systems are trained on their own exposures. Models favor popular items, popular items get more exposure, and they become even more popular. The long-term result can be a narrowing of content and difficulty for new authors to enter.

Mitigation methods include:

- Random or uncertainty-based exploration;
- Popularity debiasing and propensity weighting;
- New item quotas;
- Long-term holdouts;
- Creator-side ecosystem metrics;
- Exposure correction in training samples.

These methods sacrifice some short-term clicks. Whether this is worth it is a product and ecosystem decision, not purely a model problem.

### 11.8 Chapter Self-Test

1. Why does the cold-start problem persist even after having content embeddings?
2. What are the risks of injecting cold-start candidates after recall versus after fine-ranking?
3. Why does the UCB uncertainty bonus decrease as exposure increases?
4. Which metrics reflect that new items have truly been given an opportunity?
5. How can one determine if the system is forming a popularity feedback loop?

<details>
<summary>Reference answers</summary>

1. Content embeddings provide initial relevance but not CTR, satisfaction, audience differences, or quality feedback. New items still need traffic to collect those signals.
2. Injection after retrieval gives coverage but consumes ranking capacity and may add low-quality items. Injection after fine ranking is cheaper but may bypass quality control.
3. More exposure reduces estimator variance, so uncertainty shrinks and traffic should move toward items with demonstrated value.
4. Track the share reaching a minimum exposure threshold, time to first useful feedback, category coverage, and post-exposure quality.
5. Inspect exposure concentration and whether past exposure drives future inclusion after controlling for quality; compare with a random or exploration bucket.

</details>

# Cold Start, Exploration, and Long-Term Feedback

## Chapter 14: Cold Start, Exploration, and Long-term Feedback

### 14.1 Embeddings Only Provide Initial Estimates

New items have no interactions, so collaborative filtering cannot find them; new users have no history, so personalized models have no basis for judgment. Even if content encoders are used to generate embeddings, the system still lacks real consumption feedback.

Cold start requires solving two problems:

1. Providing initial estimates using existing information;
2. Obtaining real feedback under controlled risk.

Doing only the first without exploration makes it difficult for the model to correct initial misjudgments.

### 14.2 New Item Candidates

Common channels include:

- Category, tags, author, and region;
- Text/image/multimodal similarity;
- Clustering and placement into matching interest clusters;
- Look-alike audience expansion;
- New item pools after operational or quality review.

Look-alike starts from a small number of seed users to find populations with similar profiles or behaviors. It depends on seed quality and can easily amplify early biases.

Simple channels first make the item retrievable:

- same author: show new work to users who recently consumed that author;
- category and tags: match against historical category interests;
- content vectors: retrieve neighbors from title, image, or video representations;
- new-item pools: sample by context or cohort after quality review.

Two-tower serving also needs a policy for unseen IDs. Until a new item has its own ID embedding, use a shared default embedding or average the ID embeddings of several high-exposure, content-similar items. Separate ANN pools for items younger than one hour, six hours, or one day can share one model; they add index and retrieval cost, not another training job.

Cluster retrieval first trains a multimodal content-similarity encoder, then runs k-means on its item vectors. Positive pairs can be high-exposure items in the same fine category with strong ItemCF similarity; negatives come from other items with enough text and acceptable quality. Online, each Last-N seed item selects a nearby cluster, then retrieves recent new items from that cluster's posting. Coarse clusters mix intent, while overly fine clusters become sparse IDs.

Look-Alike can seed from users who already gave high-quality feedback. A direct implementation maintains the mean of their user embeddings as the new item's audience vector, then searches a new-item vector index with the current user's embedding. Update the mean nearline after new interactions. With few seeds, profile rules or a small classifier are more stable, and expansion needs a cap so accidental early users do not define the full audience.

### 14.3 New Users

For new users, one can utilize:

- Region, language, device, and entry point;
- First-screen interactions;
- Explicit interest selection;
- Contextual popularity;
- Similar anonymous sessions.

The most valuable data for a new user is their first few actions. The system should quickly update short-term interests and avoid showing default popular content for too long.

### 14.4 Where to Inject Traffic

Cold-start candidates can be injected at different layers:

- Recall injection: Low risk, but likely to be eliminated by coarse and fine ranking;
- Post-coarse-ranking quota: Easier to gain exposure, but requires higher quality control;
- Post-fine-ranking insertion: Stable exposure, but may significantly harm the main list;
- Reranking quota: Most controllable, but requires clear budgets and guardrails.

The later the injection, the more guaranteed the exposure, but the higher the risk of damaging the main list. High-quality review pools can be guaranteed in reranking; new content of unknown origin is better suited for small-scale traffic testing at the recall layer.

Traffic control often progresses from score boosts to dynamic guarantees and differentiated guarantees. A fixed boost is easy but exposure is sensitive to its coefficient. Dynamic guarantees increase the boost as `elapsed time / target time` grows while `current exposure / target exposure` remains low. Differentiated guarantees set larger targets for items with stronger content or creator-quality priors instead of giving every new item the same traffic.

### 14.5 Exploration vs. Exploitation

Exploitation selects the content currently estimated to be the best, while exploration selects content that is uncertain but potentially promising.

A simple UCB form:

```math
\operatorname{UCB}_i
=\hat\mu_i
+c\sqrt{\frac{\log t}{n_i}}.
```

`μ_i` is the current utility estimate, and `n_i` is the number of exposures. Items with fewer exposures receive an uncertainty bonus.

Utility in recommendations depends on user context, items expire, and feedback has delays and position effects, so the UCB formula only indicates the direction of exploration. Even without bandits, one must provide a small amount of controlled traffic to high-uncertainty candidates; otherwise, the system cannot collect the data needed to correct initial estimates.

### 14.6 Cold-Start Evaluation

Overall CTR will be drowned out by mature items. At least look at the following separately:

- Proportion of new items entering recall;
- Waiting time for first exposure;
- Traffic required to reach minimum feedback volume;
- Stratified CTR/duration/negative feedback for new items;
- Damage to the main list caused by exploration;
- Proportion of high-quality new items retained after exploration.

Also, stratify by author, category, and quality priors. Mixing all new items together and averaging them will make the strategy appear erratic.

The course groups metrics by three stakeholders. Creator metrics include publishing penetration (daily publishers divided by DAU) and items published per DAU. Consumer metrics include stratified new-item CTR and engagement plus overall time, DAU, and MAU. Content metrics include the fraction of new items that cross a defined popularity threshold within a fixed window. A policy can improve creator supply while hurting consumption, so report all three.

### 14.7 Feedback Loops

Recommendation systems are trained on their own exposures. Models favor popular items, popular items get more exposure, and they become even more popular. The long-term result can be a narrowing of content and difficulty for new authors to enter.

Mitigation methods include:

- Random or uncertainty-based exploration;
- Popularity debiasing and propensity weighting;
- New item quotas;
- Long-term holdouts;
- Creator-side ecosystem metrics;
- Exposure correction in training samples.

These methods sacrifice some short-term clicks. Whether this is worth it is a product and ecosystem decision, not purely a model problem.

### 14.8 Chapter Self-Test

1. Why does the cold-start problem persist even after having content embeddings?
2. What are the risks of injecting cold-start candidates after recall versus after fine-ranking?
3. Why does the UCB uncertainty bonus decrease as exposure increases?
4. Which metrics reflect that new items have truly been given an opportunity?
5. How can one determine if the system is forming a popularity feedback loop?
6. Which sides of new-item cold start do cluster retrieval and Look-Alike address?

<details>
<summary>Reference answers</summary>

1. Content embeddings provide initial relevance but not CTR, satisfaction, audience differences, or quality feedback. New items still need traffic to collect those signals.
2. Injection after retrieval gives coverage but consumes ranking capacity and may add low-quality items. Injection after fine ranking is cheaper but may bypass quality control.
3. More exposure reduces estimator variance, so uncertainty shrinks and traffic should move toward items with demonstrated value.
4. Track the share reaching a minimum exposure threshold, time to first useful feedback, category coverage, and post-exposure quality.
5. Inspect exposure concentration and whether past exposure drives future inclusion after controlling for quality; compare with a random or exploration bucket.
6. Cluster retrieval assigns an item-side interest cluster from content. Look-Alike expands from early responding users to similar audiences. The first provides retrievability; the second uses early audience evidence.

</details>

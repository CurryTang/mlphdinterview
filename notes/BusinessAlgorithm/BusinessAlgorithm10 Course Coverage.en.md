# Wang Shusen Course Coverage Index

This index is an audit aid, not a second textbook. Coverage is checked against the two primary public course repositories:

- [Industrial Recommender Systems](https://github.com/wangshusen/RecommenderSystem)
- [Search Engine Technology](https://github.com/wangshusen/SearchEngine)

The notes follow a production request path, so their order differs from the courses. Every row points to substantive coverage. A topic listed only here without an explanation in the target chapter does not count as covered.

## Industrial Recommender Systems

| Course module | Lecture | Notes |
| --- | --- | --- |
| Overview | Recommender-system concepts and metrics | Chapter 1; Chapter 15.1 |
| Overview | Recommendation pipeline | Chapter 1.1–1.4 |
| Overview | A/B testing | Chapter 15.1–15.7 |
| Retrieval | ItemCF | Chapter 3.4 |
| Retrieval | Swing | Chapter 3.5 |
| Retrieval | UserCF | Chapter 3.5 |
| Retrieval | Discrete features | Chapter 4.7 |
| Retrieval | Matrix completion / factorization | Chapter 3.6 |
| Retrieval | Two-tower model and training | Chapter 4.1–4.2 |
| Retrieval | Two-tower positives and negatives | Chapter 4.3 |
| Retrieval | Two-tower serving | Chapter 4.4–4.5 |
| Retrieval | Self-supervised two-tower training | Chapter 4.8 |
| Retrieval | Deep Retrieval | Chapter 4.9 |
| Retrieval | Other retrieval channels | Chapter 6.3–6.4 |
| Retrieval | Exposure filtering | Chapter 6.5 |
| Ranking | Multi-objective ranking | Chapter 10.1–10.4 |
| Ranking | MMoE | Chapter 10.3 |
| Ranking | Score fusion | Chapter 10.6–10.7 |
| Ranking | Watch-time and completion modeling | Chapter 10.5 |
| Ranking | Recommendation features | Chapter 11.6 |
| Ranking | Three-tower pre-ranking | Chapter 11.5 |
| Feature crossing | FM | Chapter 11.2; Quick Coding QC06 |
| Feature crossing | DCN | Chapter 11.3 |
| Feature crossing | LHUC / PPNet | Chapter 11.4 |
| Feature crossing | SENet and FiBiNET | Chapter 11.4 |
| Behavior sequence | Last-N | Chapter 12.1–12.2 |
| Behavior sequence | DIN | Chapter 12.3 |
| Behavior sequence | SIM | Chapter 12.4 |
| Diversity | Similarity and diversity metrics | Chapter 13.1–13.2 |
| Diversity | MMR | Chapter 13.3; Quick Coding QC07 |
| Diversity | Rule constraints | Chapter 13.5–13.6 |
| Diversity | DPP foundations | Chapter 13.4 |
| Diversity | DPP reranking | Chapter 13.4 |
| Item cold start | Evaluation | Chapter 14.6 |
| Item cold start | Simple retrieval channels | Chapter 14.2 |
| Item cold start | Cluster retrieval | Chapter 14.2 |
| Item cold start | Look-Alike expansion | Chapter 14.2 |
| Item cold start | Traffic allocation | Chapter 14.4–14.5 |
| Item cold start | Cold-start A/B tests | Chapter 15.7 |
| Metric growth | Overview | Chapter 15.8 |
| Metric growth | Retrieval improvements | Chapter 4.8–4.9; Chapter 15.8 |
| Metric growth | Ranking improvements | Chapters 11–12; Chapter 15.8 |
| Metric growth | Diversity improvements | Chapter 13; Chapter 15.8 |
| Metric growth | Special populations | Chapter 15.9 |
| Metric growth | Follows, shares, and comments | Chapter 15.10 |

## Search Engine Technology

| Source chapter | Topic | Notes |
| --- | --- | --- |
| 1 | Search concepts, satisfaction, and pipeline | Chapter 1; Chapter 7.1 |
| 2 | Search metrics and human evaluation | Chapter 7.6–7.9; Chapter 15 |
| 3 | Binary, multiclass, regression, and ranking tasks | Chapter 5.4–5.5; Chapter 9.2–9.4 |
| 4 | Pointwise, pairwise, and listwise evaluation | Chapter 8.2; Chapter 9.6 |
| 5 | NLP pretraining, post-pretraining, fine-tuning, distillation | Chapter 8.7–8.9 |
| 6 | Relevance, lexical matching, and BERT | Chapter 8 |
| 7 | EAT, text, and image quality | Chapter 7.2 |
| 8 | Breaking, general, and periodic freshness | Chapter 7.3 |
| 9 | POIs, geographic retrieval, and ranking | Chapter 7.4 |
| 10 | Personalization, CTR models, and features | Chapter 7.5; Chapter 11 |
| 11 | Dictionary/neural segmentation and NER | Chapter 5.1–5.2 |
| 12 | Term weights | Chapter 5.3 |
| 13 | Multi-label category prediction | Chapter 5.4 |
| 14 | Intent prediction and routing | Chapter 5.5 |
| 15 | Segmentation-, relevance-, and intent-driven rewriting | Chapter 5.6–5.7 |
| 16 | Inverted indexes and lexical retrieval | Chapter 3.2–3.3; Chapter 6 |
| 17 | Relevance and personalized vector retrieval | Chapter 4; Chapter 8.4 |
| 18 | Log mining, offline, reverse, and cached retrieval | Chapter 6.5–6.6 |
| 19 | Fusion features, rules, and training data | Chapter 9.5; Chapters 10–11 |
| 20 | Pointwise, pairwise, and listwise ranking training | Chapter 9.2–9.4 |
| 21 | Pre-search, SUG, SERP, and in-document scenarios | Chapter 16.1–16.2 |
| 22 | SUG, Q2Q, D2Q, and scenario-specific retrieval | Chapter 16.3–16.6 |
| 23 | Query click, conversion, and diversity ranking | Chapter 16.7–16.9 |

## Material beyond the original courses

The course foundations are connected to newer system designs:

- Chapter 17: DSI, NCI, SEAL, Semantic IDs, and TIGER;
- Chapter 18: LLM ranking, HSTU, OneRec, DPO, and RL;
- Chapter 19: RAG, Self-RAG, and Agentic Search;
- Chapter 20: recommendation, search, and generative-search system design.

These additions do not erase the traditional pipeline. First identify which retrieval, ranking, list, or generative stage a method replaces, then compare its objective, serving cost, and launch evidence.

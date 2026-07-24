# Query Understanding and Rewriting

## Chapter 5: Query Understanding and Rewriting

User queries are short and often irregular. The first search stages turn raw characters into signals that retrieval and ranking can use:

```text
raw query
  -> normalization / segmentation / entity recognition
  -> term weights / categories / intents
  -> original and rewritten queries
  -> retrieval routing, filters, and ranking features
```

An error here gives retrieval and ranking the wrong structured signal to work from.

### 5.1 Segmentation defines retrieval boundaries

Chinese text has no natural word boundaries. Dictionary-based segmentation finds valid words and uses dynamic programming to select the lowest-cost path:

```math
\operatorname{cost}(0,t)
=\min_{s<t}
\left[
\operatorname{cost}(0,s)
-\log P(x_{s:t})
\right].
```

Frequent words have lower costs. Production dictionaries also boost brands, product models, names, places, and business terms so that phrases such as `"iPhone 15 Pro Max"` are not split incorrectly.

A dictionary is always behind new language. Vocabulary construction commonly combines:

- frequencies from queries and documents;
- left and right entropy for boundary freedom;
- pointwise mutual information for internal cohesion;
- curated entity, brand, and safety dictionaries.

Neural segmentation is usually a BMES sequence-labeling task. BERT encodes each character, and a classifier or CRF predicts `B/M/E/S`. It uses context well, while dictionaries remain useful for new model numbers, strict entities, and boundaries that must be controllable.

### 5.2 Named entity recognition

NER identifies people, places, brands, works, organizations, product models, and POIs. For example:

```text
"Apple 15 repair in Beijing"
brand=Apple
model=iPhone 15
location=Beijing
intent=repair service
```

Entities affect several downstream stages:

- vertical-index selection;
- entity normalization and alias expansion;
- brand, location, or model filters;
- knowledge-base, product, and POI retrieval;
- query-document interaction features.

NER should be evaluated with exact-span precision, recall, and F1. Character accuracy hides boundary errors. Recognizing only `"Beijing"` in `"Peking University"` gets many characters right but changes the entity.

### 5.3 Term weights

Terms within a query are not equally important. One annotation method masks each term and asks how much the intent changed. A model can predict term weights from BERT outputs:

```math
\alpha_j
=\frac{\exp(w^\top h_j)}
{\sum_k \exp(w^\top h_k)}.
```

Term weights support:

- boosts in inverted-index retrieval;
- core-term preservation during rewriting;
- core-term SUG retrieval;
- relevance features;
- category and intent classification.

IDF is not a semantic term weight. IDF measures rarity in a document collection; a query term weight measures importance to this particular intent.

### 5.4 Category prediction

A query may belong to several categories. `"children's sun-protection jacket"` involves childrenswear, sun protection, and outdoor products, so multi-label classification is appropriate:

```math
p_c=\sigma(w_c^\top h_q), \qquad c=1,\ldots,C.
```

A single global threshold rarely works. Head categories have more data and higher priors; tail categories need separate calibration. Report micro-F1, macro-F1, and stratified recall:

- micro-F1 is dominated by head categories;
- macro-F1 exposes failures on tail categories;
- top-k recall tells whether the correct vertical is included.

Category predictions should route retrieval and provide features before they become hard filters. A low-confidence mistake used as a hard filter removes relevant documents before later stages can recover them.

### 5.5 Intent prediction

An intent label is useful only when it changes downstream execution:

| Intent | Possible routing |
| --- | --- |
| Product purchase | Product index, price and inventory filters |
| Local service | POI recognition, geographic retrieval and ranking |
| Breaking event | Trending index and freshness boost |
| Medical or financial | Trusted sources and stronger safety gates |
| Person or organization | Entity and account indexes |
| Navigation | Direct answer or high-precision rules |

Multi-intent queries should not be forced into one class. The model can emit multiple calibrated probabilities and let the router invoke several paths.

### 5.6 Query rewriting

Rewriting brings relevant documents into the candidate set when the original query cannot retrieve them. Common rewrites include:

- spelling and phonetic correction;
- script, case, and unit normalization;
- aliases, abbreviations, and synonyms;
- corrected segmentation;
- entity linking and product-model normalization;
- omitted-context completion;
- intent-based expansion or narrowing.

Dictionary-based rewriting is cheap and controllable. Relevance-based rewriting retrieves candidates and uses query-query or query-document relevance to reject drift. Generative rewriting is more flexible, but it must preserve core entities and constraints.

Production systems usually retain the original query path:

```text
original-query retrieval ─┐
high-confidence rewrites  ├─> merge, deduplicate, keep rewrite_source
exploratory rewrites      ┘
```

Each candidate must retain its rewrite source. Otherwise a bad result cannot be traced to the rewrite that changed the intent.

### 5.7 Evaluating query understanding

Component metrics answer whether each model completed its local task:

- segmentation and NER: exact-span precision, recall, and F1;
- category and intent: stratified precision/recall and micro/macro-F1;
- rewriting: judged relevance, coverage, and rejection rate;
- routing: correct-index coverage and empty-retrieval rate.

End-to-end checks include:

- no-result rate;
- added relevant candidates;
- relevance regressions;
- active reformulation rate;
- improvement on tail queries.

More candidates with worse relevance is not a successful rewrite system.

### 5.8 Chapter self-test

1. Why is frequency alone insufficient for discovering new words?
2. Why should NER use exact-span evaluation?
3. How do IDF and query term weights differ?
4. Why is query category prediction usually multi-label?
5. How can rewriting control semantic drift?
6. When does an intent label actually close the business loop?

<details>
<summary>Reference answers</summary>

1. A frequent fragment may be only part of a longer phrase. Internal cohesion, boundary freedom, entity dictionaries, and review are also needed.
2. The business needs the complete entity. A one-character boundary error can link to a different brand, place, or organization.
3. IDF measures collection rarity; a query term weight measures importance to the current search intent.
4. One query may express multiple categories and attributes needed by parallel retrieval paths.
5. Retain the original query, preserve high-weight terms and entities, assign confidence, and filter with relevance models and stratified human evaluation.
6. It must alter index selection, retrieval, filtering, ranking features, or safety policy. Producing an unused offline label does not close the loop.

</details>

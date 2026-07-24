# Generative Retrieval and Semantic ID

## Chapter 12: Generative Retrieval and Semantic ID

### 12.1 From "Calculating Similarity" to "Generating Identifiers"

Traditional dense retrieval performs two tasks:

1. Encoding queries and documents into the same vector space;
2. Using ANN to find the nearest neighbors.

Generative retrieval adopts a different form. Given a query, the model autoregressively generates the identifier of the target document:

```math
P(d\mid q)
=\prod_{t=1}^{L}
P(d_t\mid d_{<t},q).
```

`d_1...d_L` is a discrete token sequence representing a document or item. Beam search produces multiple identifiers, which are then mapped back to candidates.

This approach allows index identifiers, matching targets, and ranking to be trained together, with beam search directly providing the top-k results. The cost is shifted elsewhere: identifier design, corpus updates, decoding latency, and invalid IDs must all be handled by the system.

### 12.2 DSI

[DSI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html) (Tay et al., NeurIPS 2022) treats a Transformer as a differentiable search index. The model first learns "document -> docid" through document text, and then learns "query -> docid" through queries.

If the docid is an unstructured random integer, the model must memorize a vast number of arbitrary mappings. If the docid carries semantic or hierarchical information, similar documents can share prefixes, making decoding easier to generalize.

DSI tested the possibility of writing part of an external index into model parameters. It did not solve the maintenance problem for large, dynamic corpora: adding, deleting, and correcting entries is more cumbersome than updating an external index, and the model capacity must also bear the burden of corpus memorization.

### 12.3 NCI

[NCI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a46156bd3579c3b268108ea6aca71d13-Abstract-Conference.html) (Wang et al., NeurIPS 2022) further utilizes semantic document identifiers. A common construction method is to perform hierarchical clustering on document vectors:

```text
Document
  -> Level-1 cluster token
  -> Level-2 sub-cluster token
  -> ...
  -> Leaf identifier
```

The prefix represents coarse semantics, while the suffix gradually locates the specific document. The decoder uses a prefix-aware structure, and training also incorporates automatic query generation and consistency regularization.

Tree-based IDs break down full-corpus selection into multiple small classification steps, which is suitable for beam search and allows documents with the same prefix to share training signals. "Better semantics" is only one aspect of this.

### 12.4 SEAL

[SEAL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/cd88d62a2063fdaf7ce6f9068fb15dcd-Abstract-Conference.html) (Bevilacqua et al., NeurIPS 2022) does not issue an arbitrary ID for each document; instead, it generates n-grams that actually appear in the document. The generated n-grams are then mapped back to the documents containing them via an FM-index.

SEAL uses constrained decoding: the next token must be able to continue forming a valid n-gram in the corpus, and invalid paths are directly masked. The model generates discriminative text identifiers, while the traditional index verifies the existence of the identifier and completes the localization.

SEAL demonstrates that generative retrieval does not necessarily require the complete removal of external indices. Generative models and classic data structures can collaborate, an approach closer to a maintainable system than "putting everything into parameters."

### 12.5 Semantic ID

The number of recommended items can reach hundreds of millions. Treating each item_id as an independent token results in a massive vocabulary, and new items lack semantics. Semantic ID first discretizes continuous content vectors into a sequence of codes.

Let the item content vector be `e_i`, and residual quantization selects codewords layer by layer:

```math
r_i^{(0)}=e_i,
```

```math
c_i^{(l)}
=\arg\min_{c\in\mathcal C_l}
\|r_i^{(l-1)}-c\|_2^2,
```

```math
r_i^{(l)}
=r_i^{(l-1)}-c_i^{(l)}.
```

Finally:

```text
SID(i) = [code_1, code_2, ..., code_L]
```

The first few codes represent coarse semantics, while subsequent codes compensate for the residual. If there are `K` codes per layer, the combination space for length `L` can reach `K^L`, yet the vocabulary itself only requires `K` tokens per layer.

Its relationship with ordinary item embeddings is:

- Embeddings are continuous vectors used for similarity or as model input;
- Semantic IDs are discrete token sequences used for autoregressive generation;
- Semantic IDs are often obtained by quantizing embeddings, but the two are not equivalent.

### Quick Coding: Residual Quantization

Given a vector and multi-layer codebooks, select the codeword closest to the current residual at each layer, then update the residual. Return the sequence of codeword indices and the final residual. This exercise corresponds exactly to the minimal skeleton of Semantic ID generation.

Problem: [[BusinessAlgorithm09 Quick Coding.md#QC08 Residual Quantization for Semantic IDs|QC08 Residual Quantization for Semantic IDs]].

### 12.6 TIGER

[TIGER](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html) (Rajput et al., NeurIPS 2023) applies Semantic ID to sequential recommendation:

1. Use a pre-trained text encoder to obtain item content vectors;
2. Use residual quantization to generate Semantic IDs;
3. Rewrite user historical items into Semantic ID sequences;
4. The Transformer generates the Semantic ID of the next item;
5. Beam search obtains top-k candidates.

Semantically similar items share some codes, so even if a new item has no interaction history, it can obtain a meaningful ID based on its content. This explains the source of improvements in cold-start and long-tail scenarios mentioned in the paper.

However, one must be careful: if two items are quantized to the same SID, generating the correct SID does not equate to finding a unique item. In engineering, collision handling is required, such as additional leaf tokens, post-hoc candidate disambiguation, or ensuring a unique mapping from the encoder.

### 12.7 Can Search and Recommendation Share Semantic IDs?

In 2025, authors from Spotify and others studied [Joint Generative Search and Recommendation with Semantic IDs](https://arxiv.org/abs/2508.10478). Search embeddings learn query-item matching, while recommendation embeddings learn item-item behavioral co-occurrence. Quantizing them separately results in two sets of token spaces, while shared quantization may sacrifice single-task performance.

This type of work discusses:

```text
Content Semantics + Search Matching + Collaborative Signals
             ↓
       A unified, generatable discrete item language
```

Currently, this is more suitable as a frontier research direction rather than a default engineering solution. The training distributions, objectives, and update frequencies of search and recommendation differ; sharing IDs requires proof of genuine system-level benefits.

### 12.8 Engineering Bill for Generative Retrieval

| Risk | What to Validate Before Launch |
| --- | --- |
| Invalid Decoding | Do trie/FM-index constraints cover all valid IDs? How to backfill when post-hoc verification fails? |
| ID Collision | How many items correspond to one SID? Is offline evaluation scored by SID or by actual item? |
| Catalog Updates | How are IDs assigned to new items? Will old IDs change? How often is incremental training performed? |
| Beam Search | P95/P99 latency for top-k; relationship between depth, beam width, and Recall |
| Popularity Bias | Do high-frequency prefixes suppress the long tail? Are sampling, re-weighting, or calibration effective? |
| Debugging & Rollback | Can token probabilities be replayed at each step? Can traditional retrieval channels independently take over traffic? |

### 12.9 Comparison with Two-Tower Models

| Dimension | Two-Tower Retrieval | Generative Retrieval |
| --- | --- | --- |
| Representation | Continuous vector | Discrete ID sequence |
| Training | Contrastive learning | Sequence generation |
| Online Computation | Query tower + ANN | Autoregressive decoding + valid path constraints |
| New Items | Generate embedding and write to index | Assign ID, update model if necessary |
| Cold Start | Depends on content encoder | Depends on content encoder and codebook |
| Debugging | Check nearest neighbors, index, and filtering | Check token probabilities, beam, and materialized results |

The two can perform retrieval in parallel, or serve as teachers for each other. Whether to replace an existing channel depends on the incremental Recall under a fixed latency budget, rather than just looking at offline full-scale results.

---

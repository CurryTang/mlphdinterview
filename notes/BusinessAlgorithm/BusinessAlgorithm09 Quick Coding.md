# Quick Coding：8 道短题

这些题只检查核心逻辑，直接用 Python 标准库即可。答案放在题目下方，默认折叠。

## QC01 稳定 A/B 分桶

实现：

```python
def assign_bucket(user_id: str, salt: str, num_buckets: int) -> int:
    ...
```

要求：

- 相同 `user_id + salt` 永远得到同一桶；
- 返回值落在 `[0, num_buckets)`；
- 更换 `salt` 可以得到独立实验分桶；
- 不能使用 Python 内置 `hash()`，因为它默认不保证跨进程稳定；
- `num_buckets <= 0` 时抛出 `ValueError`。

<details>
<summary>参考答案</summary>

```python
import hashlib


def assign_bucket(user_id: str, salt: str, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")

    payload = f"{salt}\0{user_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % num_buckets
```

时间复杂度是 `O(len(user_id) + len(salt))`，额外空间为 `O(1)`。

</details>

## QC02 构建倒排索引

输入是已经分词的文档：

```python
documents = {
    1: ["deep", "learning", "deep"],
    2: ["learning", "system"],
}
```

实现：

```python
def build_inverted_index(documents):
    ...
```

输出每个 term 的 posting list，每条 posting 为：

```text
(doc_id, term_frequency, zero_based_positions)
```

例如 `deep` 对应 `[(1, 2, [0, 2])]`。posting 按 `doc_id` 升序。

<details>
<summary>参考答案</summary>

```python
from collections import defaultdict


def build_inverted_index(documents):
    index = defaultdict(list)

    for doc_id in sorted(documents):
        positions = defaultdict(list)
        for position, term in enumerate(documents[doc_id]):
            positions[term].append(position)

        for term, term_positions in positions.items():
            index[term].append(
                (doc_id, len(term_positions), term_positions)
            )

    return dict(index)
```

若总 token 数是 `T`，时间复杂度为 `O(T + D log D)`，其中 `D` 是文档数；索引空间为 `O(T)`。

</details>

## QC03 ItemCF 推荐

实现：

```python
def item_cf_recommend(user_items, target_user, k):
    ...
```

使用物品共现余弦相似度：

```math
sim(i,j)=\frac{|U_i\cap U_j|}{\sqrt{|U_i||U_j|}}.
```

候选分数是目标用户已交互物品与候选物品相似度之和。过滤已经交互过的物品，按 `score` 降序、`item_id` 升序返回前 `k` 个 `(item_id, score)`。

<details>
<summary>参考答案</summary>

```python
from collections import defaultdict
from math import sqrt


def item_cf_recommend(user_items, target_user, k):
    item_users = defaultdict(set)
    for user, items in user_items.items():
        for item in set(items):
            item_users[item].add(user)

    seen = set(user_items.get(target_user, []))
    scores = []

    for candidate, candidate_users in item_users.items():
        if candidate in seen:
            continue

        score = 0.0
        for item in seen:
            users = item_users.get(item, set())
            if users and candidate_users:
                score += len(users & candidate_users) / sqrt(
                    len(users) * len(candidate_users)
                )
        if score > 0:
            scores.append((candidate, score))

    scores.sort(key=lambda pair: (-pair[1], pair[0]))
    return scores[:k]
```

这是面试用直接实现。线上系统会离线生成 item-to-item top 相似表，避免请求时扫描全部物品。

</details>

## QC04 Reciprocal Rank Fusion

实现：

```python
def reciprocal_rank_fusion(rankings, rrf_k=60, top_n=None):
    ...
```

每个 ranking 是一个候选 ID 列表。融合分数：

```math
score(d)=\sum_m\frac{1}{rrf\_k+rank_m(d)}.
```

rank 从 1 开始。同一 ranking 中重复候选只计算第一次出现。按分数降序、候选 ID 升序返回。

<details>
<summary>参考答案</summary>

```python
from collections import defaultdict


def reciprocal_rank_fusion(rankings, rrf_k=60, top_n=None):
    scores = defaultdict(float)

    for ranking in rankings:
        seen = set()
        for rank, candidate in enumerate(ranking, start=1):
            if candidate in seen:
                continue
            seen.add(candidate)
            scores[candidate] += 1.0 / (rrf_k + rank)

    result = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return result if top_n is None else result[:top_n]
```

若所有 ranking 的总长度为 `N`，计分是 `O(N)`，最终排序是 `O(M log M)`，`M` 为去重候选数。

</details>

## QC05 NDCG@K

实现：

```python
def ndcg_at_k(relevances, k):
    ...
```

`relevances` 已按模型预测顺序排列，每个值是非负相关性等级。使用：

```math
DCG@K=\sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}.
```

无相关结果时返回 `0.0`，`k <= 0` 时也返回 `0.0`。

<details>
<summary>参考答案</summary>

```python
from math import log2


def ndcg_at_k(relevances, k):
    if k <= 0:
        return 0.0

    all_values = list(relevances)

    def dcg(items):
        return sum(
            (2 ** rel - 1) / log2(rank + 1)
            for rank, rel in enumerate(items, start=1)
        )

    actual = dcg(all_values[:k])
    ideal = dcg(sorted(all_values, reverse=True)[:k])
    return 0.0 if ideal == 0 else actual / ideal
```

排序 ideal list 需要 `O(n log n)`；若相关性等级范围很小，可以用计数把它降到线性。

</details>

## QC06 FM 前向计算

实现：

```python
def fm_predict(x, bias, linear_weights, factors):
    ...
```

其中 `factors[i]` 是第 `i` 个特征的 `latent_dim` 维向量。要求使用下面的等价式，把二阶交叉从 `O(d²k)` 降到 `O(dk)`：

```math
\frac{1}{2}\sum_f
\left[
\left(\sum_i v_{i,f}x_i\right)^2
-\sum_i(v_{i,f}x_i)^2
\right].
```

输入维度不一致时抛出 `ValueError`。

<details>
<summary>参考答案</summary>

```python
def fm_predict(x, bias, linear_weights, factors):
    if len(x) != len(linear_weights) or len(x) != len(factors):
        raise ValueError("feature dimensions do not match")
    if not factors:
        return float(bias)

    latent_dim = len(factors[0])
    if any(len(vector) != latent_dim for vector in factors):
        raise ValueError("factor dimensions do not match")

    linear = sum(weight * value for weight, value in zip(linear_weights, x))
    interaction = 0.0

    for latent in range(latent_dim):
        summed = sum(factors[i][latent] * x[i] for i in range(len(x)))
        squared = sum(
            (factors[i][latent] * x[i]) ** 2
            for i in range(len(x))
        )
        interaction += 0.5 * (summed ** 2 - squared)

    return bias + linear + interaction
```

设特征数为 `d`、隐向量维度为 `m`，时间复杂度是 `O(dm)`，额外空间为 `O(1)`。

</details>

## QC07 MMR 重排

实现：

```python
def mmr_rerank(candidates, relevance, similarities, k, theta):
    ...
```

每轮选择：

```math
\arg\max_i
\left[
\theta r_i-(1-\theta)\max_{j\in S}sim(i,j)
\right].
```

`similarities` 使用 `(item_a, item_b) -> score` 的字典，可只提供一个方向。相同 MMR 分数时按候选 ID 升序。

<details>
<summary>参考答案</summary>

```python
def mmr_rerank(candidates, relevance, similarities, k, theta):
    remaining = set(candidates)
    selected = []

    def similarity(a, b):
        return similarities.get(
            (a, b),
            similarities.get((b, a), 0.0),
        )

    while remaining and len(selected) < k:
        def score(item):
            redundancy = max(
                (similarity(item, chosen) for chosen in selected),
                default=0.0,
            )
            return theta * relevance[item] - (1 - theta) * redundancy

        chosen = min(remaining, key=lambda item: (-score(item), item))
        selected.append(chosen)
        remaining.remove(chosen)

    return selected
```

直接实现是 `O(k²N)`；缓存每个候选与已选集合的最大相似度后可做到 `O(kN)` 次更新。

</details>

## QC08 Semantic ID 的残差量化

实现：

```python
def residual_quantize(vector, codebooks):
    ...
```

每层 codebook 选择与当前 residual 欧氏距离最近的 codeword，然后减去该 codeword。返回：

```text
([每层选中的 codeword 下标], 最终 residual)
```

所有 codeword 必须与输入向量同维；空 codebook 或维度错误时抛出 `ValueError`。

<details>
<summary>参考答案</summary>

```python
def residual_quantize(vector, codebooks):
    residual = list(vector)
    codes = []

    for codebook in codebooks:
        if not codebook:
            raise ValueError("codebook must not be empty")
        if any(len(codeword) != len(residual) for codeword in codebook):
            raise ValueError("codeword dimensions do not match")

        best_index = min(
            range(len(codebook)),
            key=lambda index: sum(
                (residual[d] - codebook[index][d]) ** 2
                for d in range(len(residual))
            ),
        )
        codes.append(best_index)
        residual = [
            value - code
            for value, code in zip(residual, codebook[best_index])
        ]

    return codes, residual
```

若有 `L` 层、每层 `K` 个 codeword、向量维度为 `d`，时间复杂度是 `O(LKd)`。

</details>

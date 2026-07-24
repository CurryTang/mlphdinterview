# Quick Coding 题单

这些题用于复习完一个小模块后立即动手。建议先只看题目，控制在 10-20 分钟。完整离线版附有只依赖 Python 标准库的参考实现和单元测试；网页先保留题目，适合白板或在线编辑器计时练习。

本地完整版的测试命令：

```bash
python3 -m unittest handbook/quick_coding/test_solutions.py -v
```

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

时间目标：10 分钟。

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

时间目标：15 分钟。

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

时间目标：20 分钟。

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

时间目标：10 分钟。

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

时间目标：10 分钟。

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

时间目标：15 分钟。

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

时间目标：15 分钟。

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

时间目标：15 分钟。

## 建议复习方式

第一次只求写对。第二次补边界条件和复杂度。第三次口头解释这段代码在线上系统的哪一步运行、数据量增大后要换成什么实现。

例如 ItemCF 题只适合小数据面试题。真实系统会离线构造共现和 top 相似表，线上不会每次遍历所有用户与物品。

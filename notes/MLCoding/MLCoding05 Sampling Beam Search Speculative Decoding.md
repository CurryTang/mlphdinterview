# ML Coding 05 · 推理解码策略：Top-k/Top-p 采样、Beam Search 与投机解码

MLCoding01 的生成循环已经给出了最简版本的 top-p 采样。这一篇把解码策略补完成面试里常问的完整集合：把 top-k 和 top-p 组合起来用，理解 beam search 的长度偏置陷阱，以及投机解码这个"无损加速"技巧背后真正的正确性论证。三道题都作用在 logits/概率向量上，不涉及模型内部结构，所以代码全部用 NumPy 写，和白板面试的形态一致。

## 模块九：解码策略

### Exercise 1 · Top-k / Top-p 组合采样

单独的 top-k 会在概率分布很平（比如开放式续写）时砍掉太多合理选项，单独的 top-p 会在分布很尖（比如代码补全，只有一两个token说得通）时保留一堆几乎不可能的噪声。工程上通常把两者串起来用：先用 top-k 做一个粗筛（防止长尾噪声进入候选池），再在剩下的 k 个里面用 top-p 做自适应截断（分布尖锐时只留 1-2 个，分布平坦时留更多）。

两个容易在面试里漏掉的边界：

- `temperature = 0` 必须走单独的 argmax 分支，因为 `logits / 0` 会产生 `inf`/`nan`，不能指望除法自然退化成贪心。
- top-p 截断之后概率之和不再是 1，必须重新归一化，否则 `np.random.choice` 或 `torch.multinomial` 会报错或采样偏差。

#### Quick Coding：`sample_top_k_top_p`

```python
def sample_top_k_top_p(logits, k=None, p=None, temperature=1.0, rng=None):
    ...
```

<details>
<summary>参考答案</summary>

```python
import numpy as np

def sample_top_k_top_p(logits, k=None, p=None, temperature=1.0, rng=None):
    rng = rng or np.random.default_rng()
    logits = np.asarray(logits, dtype=np.float64)

    if temperature == 0:
        idx = int(np.argmax(logits))
        probs = np.zeros_like(logits)
        probs[idx] = 1.0
        return idx, probs

    scaled = (logits / temperature) - (logits / temperature).max()
    probs = np.exp(scaled)
    probs /= probs.sum()

    if k is not None and k < len(probs):
        keep = np.argsort(-probs)[:k]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    if p is not None and p < 1.0:
        order = np.argsort(-probs)
        cdf = np.cumsum(probs[order])
        cutoff = max(int(np.searchsorted(cdf, p)) + 1, 1)
        keep = order[:cutoff]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()

    idx = rng.choice(len(probs), p=probs)
    return idx, probs
```

```python
rng = np.random.default_rng(0)
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.1, -1.0, -2.0])

_, probs_k = sample_top_k_top_p(logits, k=3, rng=rng)
assert np.count_nonzero(probs_k) <= 3
assert abs(probs_k.sum() - 1.0) < 1e-9

full_probs = np.exp(logits - logits.max())
full_probs /= full_probs.sum()
_, probs_p = sample_top_k_top_p(logits, p=0.8, rng=rng)
kept = probs_p > 0
assert full_probs[kept].sum() >= 0.8 - 1e-9  # 保留集合在原分布下的累计概率确实 >= p

idx0, onehot = sample_top_k_top_p(logits, temperature=0)
assert idx0 == int(np.argmax(logits))
```

实测（`np.random.default_rng(0)`，7-token 玩具分布）：`k=3` 时非零项数为 3；`p=0.8` 时保留集合在原分布下的累计概率约为 0.817（大于等于 0.8，因为累计概率只能"跨过"阈值而不能精确停在阈值上）；`temperature=0` 精确退化为 `argmax`。

</details>

### Exercise 2 · Beam Search 与长度偏置

Beam search 维护 `beam_width` 条假设，每一步把每条假设展开成词表大小种可能，再从全部候选里按**累计 log 概率**保留全局 top `beam_width` 条。它不是精确搜索：beam 宽度不够时会漏掉全局最优序列，这一点很多人以为"beam 更宽=更慢但更准"，却说不出具体在哪一步、为什么会漏。

还有两个经典坑：

- **长度偏置**：每多生成一个 token，累计 log 概率就再乘上一个 `<1` 的概率（log 空间是再加一个负数），所以不加处理的累计分数天然偏向短序列。标准修正是长度归一化，把分数除以 `len(seq) ** alpha`（`alpha` 常取 0.6-1.0），或者等价地比较"平均 log 概率"而不是"总 log 概率"。
- **eos 处理**：一条假设生成 eos 之后就"完成"了，不应该继续展开，但它的最终分数仍然要和其余还在扩展的活跃假设放在同一个候选池里比较，不能提前把已完成的假设摘出去不参与后续几轮的比较，否则可能过早收敛到一个次优的短序列。

#### Quick Coding：`beam_search`

```python
def beam_search(log_prob_fn, start_token, beam_width, max_len, eos_id):
    ...
```

<details>
<summary>参考答案</summary>

```python
def beam_search(log_prob_fn, vocab, start_token, beam_width, max_len, eos_id):
    # log_prob_fn(prev_token) -> {token: log_prob}，玩具版本只依赖上一个 token
    beams = [([start_token], 0.0, False)]
    for _ in range(max_len):
        candidates = []
        for seq, score, done in beams:
            if done:
                candidates.append((seq, score, True))
                continue
            next_logp = log_prob_fn(seq[-1])
            for tok in vocab:
                candidates.append((seq + [tok], score + next_logp[tok], tok == eos_id))
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]
        if all(done for _, _, done in beams):
            break
    beams.sort(key=lambda x: x[1], reverse=True)
    return beams[0][0], beams[0][1]

def length_normalized_score(seq, cum_logprob, alpha=0.7):
    return cum_logprob / (len(seq) ** alpha)
```

用一个可以暴力枚举验证的玩具转移表（词表只有 `{X, Y}`，深度固定为 2）：

```python
import itertools
import numpy as np

VOCAB = ["X", "Y"]
TABLE = {
    "<s>": {"X": 0.6, "Y": 0.4},
    "X":   {"X": 0.5, "Y": 0.5},
    "Y":   {"X": 0.9, "Y": 0.1},
}
LOGTABLE = {prev: {t: np.log(p) for t, p in d.items()} for prev, d in TABLE.items()}

def score(seq):
    prev, s = "<s>", 0.0
    for tok in seq:
        s += LOGTABLE[prev][tok]
        prev = tok
    return s

def brute_force_best(depth):
    return max(
        (list(c) for c in itertools.product(VOCAB, repeat=depth)),
        key=score,
    )

def toy_beam_search(beam_width, depth):
    beams = [([], 0.0)]
    for _ in range(depth):
        cand = []
        for seq, sc in beams:
            prev = seq[-1] if seq else "<s>"
            for tok in VOCAB:
                cand.append((seq + [tok], sc + LOGTABLE[prev][tok]))
        cand.sort(key=lambda x: x[1], reverse=True)
        beams = cand[:beam_width]
    return max(beams, key=lambda x: x[1])

best_seq = brute_force_best(depth=2)                      # ['Y', 'X'], 概率 0.36
wide_seq, _ = toy_beam_search(beam_width=2, depth=2)
narrow_seq, _ = toy_beam_search(beam_width=1, depth=2)     # 贪心 = beam_width=1

assert wide_seq == best_seq
assert narrow_seq != best_seq   # 贪心只看第一步的 0.6 > 0.4，选了 X，从此再也回不到全局最优
```

实测结果：`<s> -> X` 的单步概率（0.6）比 `<s> -> Y`（0.4）高，但 `X` 分支的最佳延续只有 `0.5`，而 `Y` 分支的最佳延续高达 `0.9`。全局最优序列是 `[Y, X]`（联合概率 0.36），束宽为 2 的 beam search 能找到它；束宽为 1（等价于贪心解码）第一步就锁死在 `X`，永远找不回来，最终只能得到 `[X, X]`（联合概率 0.30）。这正是"贪心解码不是最优解码"的最小反例。

</details>

### Exercise 3 · 投机解码（Speculative Decoding）

投机解码的加速来源很直接：小而便宜的 draft 模型自回归地提出 `k` 个候选 token，大而昂贵的 target 模型只需要**一次并行前向**（而不是 `k` 次串行前向）就能对这 `k` 个位置同时打分。真正的难点不是"怎么并行"，而是"怎么在使用 draft 模型采样的情况下，保证最终吐出来的 token 分布和只用 target 模型采样时完全一样"。这是一个无损加速，不是近似加速，面试里经常会追问这一点怎么证明。

接受规则（modified rejection sampling）：对 draft 提出的 token `x`，以概率 `min(1, p_target(x) / p_draft(x))` 接受；一旦被拒绝，不能简单地退回去从 `p_target` 里重新采样了事，而要从修正后的残差分布 `max(0, p_target - p_draft)`（重新归一化）里重新采样。

证明接受后的边际分布恰好等于 `p_target`（离散情形，`x` 取遍词表）：

```text
P(接受 x) = p_draft(x) · min(1, p_target(x)/p_draft(x)) = min(p_draft(x), p_target(x))
P(总体拒绝) = 1 - sum_x min(p_draft(x), p_target(x))
P(拒绝后重采样得到 x) = P(总体拒绝) · residual(x)，其中 residual(x) = max(0, p_target(x)-p_draft(x)) / sum_x max(0, p_target(x)-p_draft(x))

注意 sum_x max(0, p_target(x)-p_draft(x)) 恰好等于 P(总体拒绝)（因为对每个 x，
p_target(x) - min(p_draft(x), p_target(x)) 要么是 0，要么是 p_target(x)-p_draft(x)，
两边对 x 求和都等于 1 - sum_x min(...)）。

所以 P(拒绝后重采样得到 x) = max(0, p_target(x) - p_draft(x))

最终 P(输出 x) = P(接受 x) + P(拒绝后重采样得到 x)
             = min(p_draft(x), p_target(x)) + max(0, p_target(x) - p_draft(x))
             = p_target(x)   对任意 x 成立（分两种情况代入即可验证：
               p_draft(x) <= p_target(x) 时是 p_draft(x) + (p_target(x)-p_draft(x)) = p_target(x)；
               p_draft(x) >  p_target(x) 时是 p_target(x) + 0 = p_target(x)）
```

也就是说，不管 draft 模型多不准，只要接受/拒绝规则用对了，最终的输出分布严格等于 target 模型的分布；draft 模型质量只影响接受率（进而影响加速比），不影响正确性。

#### Quick Coding：`speculative_accept_step`

```python
def speculative_accept_step(p_target, p_draft, rng):
    ...
```

<details>
<summary>参考答案</summary>

```python
import numpy as np

def speculative_accept_step(p_target, p_draft, rng):
    vocab_size = len(p_target)
    x = rng.choice(vocab_size, p=p_draft)          # draft 提议一个 token
    accept_prob = min(1.0, p_target[x] / p_draft[x])
    if rng.random() < accept_prob:
        return x, True
    residual = np.maximum(0.0, p_target - p_draft)
    residual /= residual.sum()
    return rng.choice(vocab_size, p=residual), False

def speculative_decode(target_logits_fn, draft_logits_fn, prompt, num_speculative_tokens, rng):
    # target_logits_fn / draft_logits_fn: 给定当前上下文，返回下一个 token 的完整概率分布
    tokens = list(prompt)
    proposals = []
    for _ in range(num_speculative_tokens):
        p_draft = draft_logits_fn(tokens + proposals)
        proposals.append(int(rng.choice(len(p_draft), p=p_draft)))

    # target 模型对 prompt + 全部候选 token 做一次并行前向，拿到每个位置的分布
    accepted = []
    for i, x in enumerate(proposals):
        p_target = target_logits_fn(tokens + proposals[:i])
        p_draft_i = draft_logits_fn(tokens + proposals[:i])
        accept_prob = min(1.0, p_target[x] / p_draft_i[x])
        if rng.random() < accept_prob:
            accepted.append(x)
            continue
        residual = np.maximum(0.0, p_target - p_draft_i)
        residual /= residual.sum()
        accepted.append(int(rng.choice(len(p_target), p=residual)))
        break  # 第一次拒绝之后，后面的 draft token 全部作废，从残差重采样这一个就结束本轮
    return tokens + accepted
```

Monte Carlo 验证（5-token 玩具词表，`np.random.default_rng(0)`，200,000 次试验）：

```python
p_target = np.array([0.05, 0.35, 0.30, 0.20, 0.10])
p_draft  = np.array([0.30, 0.10, 0.10, 0.40, 0.10])

rng = np.random.default_rng(0)
samples = np.array([speculative_accept_step(p_target, p_draft, rng)[0] for _ in range(200_000)])
empirical = np.bincount(samples, minlength=5) / 200_000

assert np.max(np.abs(empirical - p_target)) < 0.01       # 收敛到 target 分布
assert np.max(np.abs(empirical - p_draft)) > 0.05         # 明显不是 draft 分布
```

实测经验分布为 `[0.0499, 0.3497, 0.3018, 0.1990, 0.0996]`，与 `p_target = [0.05, 0.35, 0.30, 0.20, 0.10]` 的最大误差约 `0.0018`（在 200,000 次采样的蒙特卡洛噪声范围内），与 `p_draft` 的最大误差约 `0.25`（明显不是 draft 分布）。接受率 = `sum(min(p_target, p_draft))`，在这组数字下约为 `0.55`，也就是平均每两个 draft token 大约有一个多能被直接接受。

</details>

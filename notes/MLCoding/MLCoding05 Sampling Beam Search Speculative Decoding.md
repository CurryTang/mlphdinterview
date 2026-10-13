# 05 · 推理解码：采样与投机解码

MLCoding01 的生成循环已经给出了最简版本的 top-p 采样。这一篇把解码策略补完成面试里常问的完整集合：把 top-k 和 top-p 组合起来用，理解 beam search 的长度偏置陷阱，以及投机解码这个"无损加速"技巧背后真正的正确性论证。三道题都作用在 logits/概率向量上，不涉及模型内部结构。代码全部采用 PyTorch (`torch`) 实现，力求步骤清晰、直观易记、不跳步且避免晦涩语法，贴合高频面试白板手撕的标准形态。

## 模块九：解码策略

### Exercise 1 · Top-k / Top-p 组合采样

单独的 top-k 会在概率分布很平（比如开放式续写）时砍掉太多合理选项，单独的 top-p 会在分布很尖（比如代码补全，只有一两个 token 说得通）时保留一堆几乎不可能的噪声。工程上通常把两者串起来用：先用 top-k 做一个粗筛（防止长尾噪声进入候选池），再在剩下的 k 个里面用 top-p 做自适应截断（分布尖锐时只留 1-2 个，分布平坦时留更多）。

两个容易在面试里漏掉的边界：

- `temperature = 0` 必须走单独的 argmax 分支，因为 `logits / 0` 会产生 `inf`/`nan`，不能指望除法自然退化成贪心。
- top-p 截断之后概率之和不再是 1，必须重新归一化，否则 `torch.multinomial` 会报错或采样偏差。

#### Quick Coding：`sample_top_k_top_p`

```python
def sample_top_k_top_p(logits, k=None, p=None, temperature=1.0, generator=None):
    ...
```

<details>
<summary>参考答案</summary>

```python
import torch

def sample_top_k_top_p(logits, k=None, p=None, temperature=1.0, generator=None):
    logits = torch.as_tensor(logits, dtype=torch.float32)

    # 1. temperature = 0: 贪心选择 (argmax)
    if temperature == 0:
        idx = torch.argmax(logits).item()
        probs = torch.zeros_like(logits)
        probs[idx] = 1.0
        return idx, probs

    # 2. 温度调节并计算初始概率
    probs = torch.softmax(logits / temperature, dim=-1)

    # 3. Top-k 过滤：只保留概率最高的 k 个候选
    if k is not None and k < len(probs):
        topk_probs, topk_indices = torch.topk(probs, k)
        new_probs = torch.zeros_like(probs)
        new_probs[topk_indices] = topk_probs
        probs = new_probs / new_probs.sum()

    # 4. Top-p 过滤 (Nucleus Sampling)：累积概率截断
    if p is not None and p < 1.0:
        # 按概率从大到小排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        # 计算累积分布函数 (CDF)
        cdf = torch.cumsum(sorted_probs, dim=0)

        # 找到累积概率首次达到或超过 p 的截断点（至少保留 1 个）
        cutoff = torch.searchsorted(cdf, torch.tensor(p)).item() + 1
        cutoff = min(max(cutoff, 1), len(probs))

        keep_indices = sorted_indices[:cutoff]
        keep_probs = sorted_probs[:cutoff]

        new_probs = torch.zeros_like(probs)
        new_probs[keep_indices] = keep_probs
        probs = new_probs / new_probs.sum()

    # 5. 按最终分布进行多项式采样
    idx = torch.multinomial(probs, num_samples=1, generator=generator).item()
    return idx, probs
```

```python
gen = torch.Generator().manual_seed(0)
logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.1, -1.0, -2.0])

_, probs_k = sample_top_k_top_p(logits, k=3, generator=gen)
assert torch.count_nonzero(probs_k) <= 3
assert abs(probs_k.sum().item() - 1.0) < 1e-6

full_probs = torch.softmax(logits, dim=-1)
_, probs_p = sample_top_k_top_p(logits, p=0.8, generator=gen)
kept = probs_p > 0
assert full_probs[kept].sum().item() >= 0.8 - 1e-6  # 保留集合在原分布下的累计概率确实 >= p

idx0, onehot = sample_top_k_top_p(logits, temperature=0)
assert idx0 == int(torch.argmax(logits).item())
```

实测（`torch.Generator().manual_seed(0)`，7-token 玩具分布）：`k=3` 时非零项数为 3；`p=0.8` 时保留集合在原分布下的累计概率约为 0.817（大于等于 0.8，因为累计概率只能"跨过"阈值而不能精确停在阈值上）；`temperature=0` 精确退化为 `argmax`。

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
import torch

def length_normalized_score(seq, cum_logprob, alpha=0.7):
    # 长度惩罚公式：Score = log_prob / (len(seq) ** alpha)
    return cum_logprob / (len(seq) ** alpha)

def beam_search(log_prob_fn, start_token, beam_width=3, max_len=20, eos_id=None, alpha=0.0):
    # initial_seq: 支持传入单个 int 或列表
    initial_seq = [start_token] if isinstance(start_token, int) else list(start_token)
    # 每条 beam 维护三元组: (已生成的 token 列表, 累积 log 概率, 是否已遇到 eos 结束)
    beams = [(initial_seq, 0.0, False)]

    for _ in range(max_len):
        candidates = []
        for seq, score, done in beams:
            # 已结束的序列不再展开，原样保留参与本轮竞争
            if done:
                candidates.append((seq, score, True))
                continue

            # 调用模型获取下一个 token 的 log 概率分布 (1D Tensor, shape: [vocab_size])
            next_logp = log_prob_fn(seq)
            if not isinstance(next_logp, torch.Tensor):
                next_logp = torch.tensor(next_logp, dtype=torch.float32)

            # 遍历词表逐一扩展
            for tok_id in range(len(next_logp)):
                tok_score = next_logp[tok_id].item()
                is_eos = (eos_id is not None and tok_id == eos_id)
                candidates.append((seq + [tok_id], score + tok_score, is_eos))

        # 按得分从高到低排序候选（若开启长度归一化则按归一化得分排序）
        candidates.sort(
            key=lambda x: length_normalized_score(x[0], x[1], alpha) if alpha > 0 else x[1],
            reverse=True
        )
        # 只保留最优的 beam_width 条路径
        beams = candidates[:beam_width]

        # 若当前所有 beam 均已生成 eos，提前终止搜索
        if all(done for _, _, done in beams):
            break

    # 最终选取得分最高的假设
    beams.sort(
        key=lambda x: length_normalized_score(x[0], x[1], alpha) if alpha > 0 else x[1],
        reverse=True
    )
    return beams[0][0], beams[0][1]
```

用一个可以暴力枚举验证的玩具转移表（词表大小为 2: 0 代表 X，1 代表 Y，搜索深度固定为 2）：

```python
import itertools
import torch

# 词表: 0 代表 X, 1 代表 Y; -1 代表起始符 <s>
# 转移概率:
# <s> -> X: 0.6, Y: 0.4
# X   -> X: 0.5, Y: 0.5
# Y   -> X: 0.9, Y: 0.1
LOG_TABLE = {
    -1: torch.log(torch.tensor([0.6, 0.4])),
    0:  torch.log(torch.tensor([0.5, 0.5])),
    1:  torch.log(torch.tensor([0.9, 0.1])),
}

def toy_log_prob_fn(seq):
    prev_tok = seq[-1]
    return LOG_TABLE[prev_tok]

def score(seq):
    prev, s = -1, 0.0
    for tok in seq:
        s += LOG_TABLE[prev][tok].item()
        prev = tok
    return s

def brute_force_best(depth):
    return max(
        (list(c) for c in itertools.product([0, 1], repeat=depth)),
        key=score,
    )

best_seq = brute_force_best(depth=2)                                   # [1, 0] 即 [Y, X], 联合概率 0.36
wide_seq, _ = beam_search(toy_log_prob_fn, start_token=-1, beam_width=2, max_len=2)
narrow_seq, _ = beam_search(toy_log_prob_fn, start_token=-1, beam_width=1, max_len=2)  # beam_width=1 等价于贪心

assert wide_seq[1:] == best_seq
assert narrow_seq[1:] != best_seq  # 贪心第一步只看 0.6 > 0.4 选择了 0 (X)，从此错过全局最优
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
def speculative_accept_step(p_target, p_draft, generator=None):
    ...
```

<details>
<summary>参考答案</summary>

```python
import torch

def speculative_accept_step(p_target, p_draft, generator=None):
    p_target = torch.as_tensor(p_target, dtype=torch.float32)
    p_draft = torch.as_tensor(p_draft, dtype=torch.float32)

    # 1. Draft 模型根据自身概率分布提议一个 token
    x = torch.multinomial(p_draft, num_samples=1, generator=generator).item()

    # 2. 计算接受概率 min(1, p_target(x) / p_draft(x))
    accept_prob = min(1.0, (p_target[x] / p_draft[x]).item())

    # 3. 接受/拒绝判定
    u = torch.rand(1, generator=generator).item()
    if u < accept_prob:
        return x, True

    # 4. 若拒绝，从修正残差分布 max(0, p_target - p_draft) 重采样
    residual = torch.clamp(p_target - p_draft, min=0.0)
    residual = residual / residual.sum()
    resampled_x = torch.multinomial(residual, num_samples=1, generator=generator).item()
    return resampled_x, False

def speculative_decode(target_prob_fn, draft_prob_fn, prompt, num_speculative_tokens, generator=None):
    tokens = list(prompt)
    proposals = []
    draft_probs_list = []

    # 1. Draft 模型自回归串行提出 K 个候选
    for _ in range(num_speculative_tokens):
        p_draft = draft_prob_fn(tokens + proposals)
        if not isinstance(p_draft, torch.Tensor):
            p_draft = torch.tensor(p_draft, dtype=torch.float32)
        tok = torch.multinomial(p_draft, num_samples=1, generator=generator).item()
        proposals.append(tok)
        draft_probs_list.append(p_draft)

    # 2. Target 模型并行验证（此处模拟按位置比对概率）
    accepted = []
    for i, x in enumerate(proposals):
        p_target = target_prob_fn(tokens + proposals[:i])
        if not isinstance(p_target, torch.Tensor):
            p_target = torch.tensor(p_target, dtype=torch.float32)
        p_draft_i = draft_probs_list[i]

        accept_prob = min(1.0, (p_target[x] / p_draft_i[x]).item())
        u = torch.rand(1, generator=generator).item()
        if u < accept_prob:
            accepted.append(x)
            continue

        # 第一次拒绝后，后面的 draft token 全部作废，从残差重采样这一个后结束本轮
        residual = torch.clamp(p_target - p_draft_i, min=0.0)
        residual = residual / residual.sum()
        resampled_tok = torch.multinomial(residual, num_samples=1, generator=generator).item()
        accepted.append(resampled_tok)
        break

    return tokens + accepted
```

Monte Carlo 验证（5-token 玩具词表，`torch.Generator().manual_seed(0)`，200,000 次试验）：

```python
p_target = torch.tensor([0.05, 0.35, 0.30, 0.20, 0.10])
p_draft  = torch.tensor([0.30, 0.10, 0.10, 0.40, 0.10])

gen = torch.Generator().manual_seed(0)
N = 200_000
samples = torch.tensor([speculative_accept_step(p_target, p_draft, generator=gen)[0] for _ in range(N)])
empirical = torch.bincount(samples, minlength=5).float() / N

assert (empirical - p_target).abs().max().item() < 0.01  # 收敛到 target 分布
assert (empirical - p_draft).abs().max().item() > 0.05   # 明显不是 draft 分布
```

实测经验分布为 `[0.0501, 0.3499, 0.3007, 0.1992, 0.1001]`，与 `p_target = [0.05, 0.35, 0.30, 0.20, 0.10]` 的最大误差约 `0.0008`（在 200,000 次采样的蒙特卡洛噪声范围内），与 `p_draft` 的最大误差约 `0.25`（明显不是 draft 分布）。接受率 = `sum(min(p_target, p_draft))`，在这组数字下约为 `0.55`，也就是平均每两个 draft token 大约有一个多能被直接接受。

</details>

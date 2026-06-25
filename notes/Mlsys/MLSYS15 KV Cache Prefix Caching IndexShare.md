# MLSYS16 · KV Cache：内存管理、前缀复用与 GLM-5.2 IndexShare

这篇补上推理系统里最常被追问的一层：KV cache 到底在存什么，为什么它会变成长上下文推理的主瓶颈，以及 GLM-5.2 的 IndexShare 到底共享了什么。

先给结论：

```text
KV cache 省的是重复计算。
PagedAttention 解决的是 KV cache 动态内存管理。
Prefix cache 解决的是不同请求之间的公共前缀复用。
IndexShare / IndexCache 解决的是 DSA sparse attention 里 indexer 重复选 top-k 的成本。
```

它们不是同一个问题，但真实 runtime 会把它们放在同一条 decode 路径里。

## 目录

1. [[#一、为什么 decode 必须有 KV cache]]
2. [[#二、KV cache 里具体存什么]]
3. [[#三、PagedAttention：把 KV cache 当显存页表管理]]
4. [[#四、Prefix cache：同一段 prompt 不要反复 prefill]]
5. [[#五、KV cache 的容量账]]
6. [[#六、Kernel 视角：paged decode attention 怎么读 cache]]
7. [[#七、KV cache 压缩、淘汰与传输]]
8. [[#八、DSA sparse attention：先选 token，再做 attention]]
9. [[#九、GLM-5.2 IndexShare / IndexCache 源码精读]]
10. [[#十、KV cache 与 MTP / speculative decoding]]
11. [[#十一、面试题]]
12. [[#参考资料]]

---

## 一、为什么 decode 必须有 KV cache

Transformer 第 `l` 层 attention 的核心是：

$$
Q_l = X_l W_Q,\quad K_l = X_l W_K,\quad V_l = X_l W_V
$$

自回归 decode 第 `t` 步只新增一个 token。如果没有 KV cache，每一步都要把从 `0` 到 `t` 的 token 重新投影成 K/V：

```text
step 1: recompute K/V for token 0
step 2: recompute K/V for token 0..1
step 3: recompute K/V for token 0..2
...
```

这会把历史 token 的 K/V projection 重复做很多遍。KV cache 的做法是：

```text
prefill:
  一次性算 prompt 所有 token 的 K/V，写入 cache

decode step t:
  只为新 token 算 K/V
  把新 K/V append 到 cache
  当前 query 读取历史 K/V 做 attention
```

所以 decode 的 attention 变成：

```text
q_t attends to cached K[0:t] and V[0:t]
```

注意这个优化没有让 attention 对历史长度免费。每生成一个 token，query 仍然要读越来越长的 K/V。KV cache 只是避免重复生成历史 K/V。

可视化：

```text
without KV cache

step t:
  X[0:t] -> K[0:t], V[0:t]  # repeat projection
  q_t    -> attention

with KV cache

prefill:
  X[0:p] -> KV cache

decode step t:
  x_t -> k_t, v_t -> append
  q_t -> read KV cache[0:t]
```

## 二、KV cache 里具体存什么

普通 MHA 的每层 cache 可以粗略写成：

```text
K cache: [num_tokens, num_kv_heads, head_dim]
V cache: [num_tokens, num_kv_heads, head_dim]
```

如果 batch 里有多个请求，runtime 通常不会真的存成一个规整的 `[batch, max_seq, ...]` 大矩阵，因为每个请求长度不同，而且请求会不断进出。

### MHA / GQA / MQA 的区别

| 结构 | KV heads | cache 体积 | 典型影响 |
|---|---:|---:|---|
| MHA | 等于 query heads | 最大 | attention 表达最完整，但 cache 压力最大 |
| GQA | 少于 query heads | 中等 | Llama 系列常见，cache 大幅下降 |
| MQA | 1 个 KV head | 最小 | cache 最省，但模型结构约束更强 |

单层 KV cache 大小近似：

$$
2 \times T \times H_{kv} \times D \times \text{bytes}
$$

全模型再乘以层数 `L`。`2` 是 K 和 V。

举例：

```text
layers = 80
seq_len = 128K
kv_heads = 8
head_dim = 128
dtype = bf16 = 2 bytes

KV size ~= 2 * 80 * 128K * 8 * 128 * 2
        ~= 40 GB
```

这只是单请求，不含 allocator 碎片、block table、prefix cache、spec decode 候选 token、并发 batch 等额外开销。

### MLA 的 cache 形态

DeepSeek-V2/V3、GLM-5 这类 MLA 结构不直接缓存完整 K/V head，而是缓存低秩 latent KV 和 RoPE 部分。以 ATOM 的 GLM path 为例，主 MLA cache 存的是：

```text
kv_lora_rank + qk_rope_head_dim
```

decode 时再通过矩阵吸收或投影把 query 与 cached latent 结合。这样可以显著降低 KV cache 体积，但 kernel 更复杂。

## 三、PagedAttention：把 KV cache 当显存页表管理

vLLM 的 PagedAttention 借鉴 OS virtual memory。每个请求的逻辑 token 序列被切成固定大小的 block：

```text
request A logical tokens:

[0..15] [16..31] [32..47] [48..]
  blk0    blk1     blk2    blk3

block table:
logical blk0 -> physical block 91
logical blk1 -> physical block 18
logical blk2 -> physical block 37
logical blk3 -> physical block 44
```

这样做解决三个问题：

1. 请求长度动态增长，不需要预分配最大长度。
2. 请求结束后释放 block，显存可以被其他请求复用。
3. 多个请求共享前缀时，可以让 block table 指向同一批 physical blocks，并用引用计数管理。

它的代价是 attention kernel 不能再假设 K/V 在内存里连续。kernel 读第 `j` 个历史 token 时，需要通过 block table 找到物理地址：

```text
logical_token_id -> logical_block_id, offset_in_block
logical_block_id -> physical_block_id
physical_block_id, offset_in_block -> K/V address
```

### PagedAttention 与 contiguous cache 的取舍

| 方案 | 优点 | 代价 |
|---|---|---|
| contiguous cache | kernel 简单，连续读更直接 | 动态长度下容易浪费和碎片化 |
| PagedAttention | 分配灵活，适合 continuous batching | kernel 多一次 block table 间接寻址 |
| vAttention | 保持虚拟地址连续，物理页按需分配 | 依赖 CUDA virtual memory 管理，系统层复杂 |

PagedAttention 不是 attention 数学变了，而是 K/V 的内存布局和访问方式变了。

## 四、Prefix cache：同一段 prompt 不要反复 prefill

很多线上请求有相同前缀：

```text
system prompt
tool definitions
few-shot examples
repo context header
```

如果每个请求都重新 prefill 这些 token，TTFT 会被浪费掉。prefix cache 的做法是把已经算好的 KV block 保留下来，下次新请求命中同样 token 前缀时直接复用。

SGLang 的 RadixAttention 用 radix tree 管理这些前缀：

```text
root
 └── system prompt
      ├── user A history
      └── user B history
```

一个请求进来时：

```text
1. tokenizer 后得到 token ids
2. 在 radix tree 中找最长公共前缀
3. 命中的 KV blocks 增加引用计数
4. 未命中的 suffix 才需要 prefill
```

prefix cache 的正确性要求很严格：

```text
必须是 token-level prefix 完全一致。
同一句文本只要 tokenizer 结果不同，就不能复用同一段 KV。
```

它也会改变 scheduler。高命中率请求应该尽量被路由到已有 cache 的 worker；否则 cache 在另一个 worker 上，仍然要重新 prefill 或跨机传输 KV。

## 五、KV cache 的容量账

推理系统经常看起来是算力问题，最后却卡在 KV 容量上。

一个实用估算：

```text
KV bytes = 2 * layers * tokens * kv_heads * head_dim * bytes_per_element
```

对 MLA，可以换成：

```text
MLA KV bytes = layers * tokens * (kv_lora_rank + qk_rope_head_dim) * bytes_per_element
```

这里没有乘 `2`，因为 cache 形态已经把 latent KV 和 RoPE 部分合在一起。不同实现会有 padding、scale、block metadata，需要看具体 runtime。

### 为什么长上下文下 KV cache 比权重更麻烦

权重大小是固定的：

```text
model weights: fixed
```

KV cache 随并发和上下文长度增长：

```text
KV cache ~= active_requests * sequence_length
```

所以同一张卡上能跑多大 batch，常常由 KV cache 决定，而不是由权重决定。

## 六、Kernel 视角：paged decode attention 怎么读 cache

一个简化的 paged decode attention kernel 可以理解成：

```python
def paged_decode_attention(q, block_table, k_cache, v_cache, seq_len):
    scores = []
    for token_id in range(seq_len):
        logical_block = token_id // BLOCK_SIZE
        offset = token_id % BLOCK_SIZE
        physical_block = block_table[logical_block]

        k = k_cache[physical_block, offset]
        score = dot(q, k)
        scores.append(score)

    probs = softmax(scores)

    out = 0
    for token_id in range(seq_len):
        logical_block = token_id // BLOCK_SIZE
        offset = token_id % BLOCK_SIZE
        physical_block = block_table[logical_block]
        v = v_cache[physical_block, offset]
        out += probs[token_id] * v

    return out
```

真实 kernel 会做这些优化：

- 一个 CTA 处理一个或多个 query head。
- K/V 以 vectorized load 读取，尽量 coalesced。
- 对长上下文分块做 online softmax，避免保存完整 score 向量。
- block table、sequence length、slot mapping 都来自 runtime metadata。
- CUDA graph 要求 batch shape 稳定，因此 runtime 常把请求 padding 到固定 capture size。

这解释了为什么 FlashAttention、FlashInfer、vLLM paged kernel 不是简单替换关系。它们都在处理 attention，但输入布局、metadata、batch 动态性不同。

## 七、KV cache 压缩、淘汰与传输

KV cache 优化大致有四类。

### 1. 更小的结构

GQA、MQA、MLA 直接减少每个 token 需要缓存的向量维度。

### 2. 低精度 KV

FP8 KV cache 可以减半显存，但要处理 scale、quantization error、kernel 支持。对长上下文，K 的误差会影响 attention score，V 的误差会影响 value aggregation。

### 3. 淘汰或稀疏保留

H2O 保留 heavy hitter token 和最近 token。StreamingLLM 保留 attention sink 和滑动窗口。SnapKV 在 prefill 后根据 prompt attention 选择每个 head 更重要的 KV positions。

这些方法能省显存，但通常不再是严格等价推理。面试里要说清楚：

```text
PagedAttention / prefix cache 是 exact memory management。
KV eviction / compression 通常是近似方法，除非只是无损编码或纯 dtype 改写。
```

### 4. 跨机传输和持久化

CacheGen、LMCache 这类工作关注的是复用长上下文时，KV cache 很大，直接从远端拉 KV 可能比重算还慢。因此它们会压缩 KV bitstream、异步加载、边传输边解码或在带宽不够时选择部分重算。

## 八、DSA sparse attention：先选 token，再做 attention

DeepSeek Sparse Attention 和 GLM-5 的 DSA 不再让每层 attention 对所有历史 token 做完整 attention。它加了一个轻量 indexer：

```text
hidden/query -> indexer -> top-k token indices
top-k token indices -> sparse MLA attention
```

标准 DSA 每层都做：

```text
for layer in layers:
  indices = indexer_l(query_l, cached_keys_l)
  output = sparse_attention_l(query_l, KV_l, indices)
```

indexer 比主 attention 便宜，但它仍然要对长上下文打分并做 top-k。上下文到 200K、1M 后，这个成本不可忽略。

IndexCache 论文的观察是：

```text
相邻层选出来的 top-k token 很像。
如果一组层都在看差不多的历史 token，就没必要每层都重新跑 indexer。
```

于是把层分成两类：

```text
F = Full layer：运行自己的 indexer，产生新的 top-k indices
S = Shared layer：不运行 indexer，复用前一个 F layer 的 top-k indices
```

可视化：

```text
standard DSA

L0: indexer -> indices0 -> sparse attention
L1: indexer -> indices1 -> sparse attention
L2: indexer -> indices2 -> sparse attention
L3: indexer -> indices3 -> sparse attention

IndexShare / IndexCache

L0: indexer -> indices0 -> sparse attention
L1: reuse indices0 -> sparse attention
L2: reuse indices0 -> sparse attention
L3: reuse indices0 -> sparse attention
L4: indexer -> indices4 -> sparse attention
```

重要区别：

```text
共享的是 selected token indices，不是共享每层的主 KV cache。
每层 attention 仍然有自己的 hidden state、projection、MLP、residual。
```

## 九、GLM-5.2 IndexShare / IndexCache 源码精读

官方 GLM-5.2 文档说 IndexShare 每 4 个 sparse attention layer 共享一个轻量 indexer，1M context 下 per-token FLOPs 降低 2.9 倍。论文名是 IndexCache，方法名强调 cross-layer index reuse。

### 1. Transformers reference path

Hugging Face Transformers 里的 `glm_moe_dsa` config 直接暴露 per-layer schedule：

```python
indexer_types = ["full", "shared", "shared", "shared", ...]
```

核心逻辑：

```python
self.skip_topk = config.indexer_types[layer_idx] == "shared"
self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)
```

forward 时：

```python
if self.indexer is not None:
    topk_indices = self.indexer(...)
else:
    topk_indices = prev_topk_indices
```

模型主循环维护一个 `topk_indices` 变量：

```python
topk_indices = None
for decoder_layer in self.layers:
    hidden_states, topk_indices = decoder_layer(
        hidden_states,
        prev_topk_indices=topk_indices,
    )
```

所以 `shared` 层没有自己的 indexer 权重，也不会产生新的 indices。它拿上一层 Full indexer 的结果继续跑 sparse attention。

### 2. ATOM / vLLM serving path

ATOM 的 GLM-5.2 recipe 明确写了：

```text
"full" attention layers compute DSA indexer.
"shared" layers reuse previous full layer.
shared layers carry no indexer weights.
```

对应源码有三个关键点。

第一，`_should_skip_index_topk` 根据 config 决定是否跳过 top-k：

```python
if indexer_types[layer_id] == "shared":
    return True
```

它还处理 MTP layer：

```python
if layer_id >= num_hidden_layers and index_share_for_mtp_iteration:
    return True
```

第二，`_indexer_weights_shared` 让 shared 层不构建 indexer 参数：

```python
if indexer_types[layer_id] == "shared":
    self.indexer = None
```

第三，vLLM plugin 会注册 indexer cache，让 vLLM 的 KV cache allocator 给 indexer cache 分配显存：

```python
AttentionLayerBase.register(DeepseekV32IndexerCache)
vllm_sfc[prefix] = module
```

这是容易漏掉的系统细节：DSA 除了主 MLA KV cache，还需要 indexer 的 key cache。ATOM recipe 也建议 GLM-5.2 用 `--kv_cache_dtype bf16`，并把 `--gpu-memory-utilization` 设到 `0.8` 左右，给 DSA index cache 留空间。

### 3. indexer 怎么把 top-k 交给 sparse MLA kernel

ATOM/vLLM sparse MLA metadata builder 会分配一块 buffer：

```python
self.paged_kv_indices = torch.zeros(
    [max_num_batched_tokens * topk_tokens],
    dtype=torch.int32,
    device=device,
)
```

然后把同一块 buffer 绑定给 indexer 和 sparse attention：

```python
indexer.sparse_kv_indices_buffer = self.paged_kv_indices
sparse_attn.sparse_kv_indices_buffer = self.paged_kv_indices
```

indexer forward 做三件事：

```text
1. 把当前 token 的 indexer K 写进 indexer cache
2. 对历史 indexer K 做 FP8 MQA logits
3. top-k 后把 request-local indices 转成 paged global indices
```

最后写入：

```text
sparse_kv_indices_buffer / paged_kv_indices
```

sparse MLA decode kernel 再读：

```python
mla_decode_fwd(
    q,
    kv_buffer,
    output,
    qo_indptr,
    paged_kv_indptr,
    paged_kv_indices,
    paged_kv_last_page_len,
    ...
)
```

也就是说，IndexShare 在 serving 里的真实数据流是：

```text
Full layer:
  hidden -> indexer -> top-k local indices
  local indices -> paged global indices
  write paged_kv_indices
  sparse MLA reads paged_kv_indices

Shared layer:
  skip indexer
  sparse MLA reads reused paged_kv_indices
```

### 4. 为什么不能简单“每 4 层固定共享”就完事

IndexCache 论文给了两个版本：

| 版本 | 做法 | 适用场景 |
|---|---|---|
| training-free | 冻结模型，用校准集 LM loss 贪心搜索哪些层保留 indexer | 已有 DSA 模型快速改造 |
| training-aware | retained indexer 用 multi-layer distillation 同时服务多层 | 从训练中就让 indexer 适应共享 |

GLM-5.2 是训练中引入 IndexShare。官方博客写到从 mid-training 的 128K sequence length 开始训练，这比事后硬改 schedule 更稳。

论文里还有一个值得记住的负结果：

```text
只看 top-k overlap 或 attention output similarity，不足以找到最优 sharing pattern。
最终质量要看 end-to-end LM loss 或下游任务。
```

原因是 shared 层如果漏掉少量关键 token，误差会沿后续层传播。早期层尤其敏感。

## 十、KV cache 与 MTP / speculative decoding

speculative decoding 会让一次 target verify 接受多个 token。runtime 因此要处理：

```text
decode query length > 1
candidate tokens 的 temporary KV
accepted token 的 KV commit
rejected suffix 的 rollback
```

GLM-5.2 的 MTP 还把 IndexShare 用到 MTP layer。官方博客的关键点是：

```text
MTP 第一步运行 indexer。
后续 MTP step 复用第一步的 top-k indices。
KV cache 只保留来自 target model hidden states 的 kv1:4。
训练时复用第一步的 KV cache 和 top-k indices。
```

这样做有两个目的：

1. MTP draft 成本更低。
2. 减少训练和推理不一致。否则后续 MTP step 的 KV 会混入 MTP 自己生成的 hidden states。

ATOM 的 indexer metadata 里也能看到多 token decode 的处理：如果 `max_decode_len > 1`，它会把 multi-token decode request 展平成多个 single-token batch entry，再构造对应的 seq_lens 和 block table。这样底层 paged MQA logits / top-k kernel 仍然能按统一接口跑。

## 十一、面试题

### 1. KV cache 降低了什么复杂度？

它避免重复计算历史 token 的 K/V projection。decode 每步仍然要让当前 query 读历史 K/V，所以 attention 的历史读流量仍随 sequence length 增长。

### 2. 为什么 PagedAttention 能提高吞吐？

它减少 KV cache 内存浪费，让同样显存容纳更多活跃请求。吞吐提升主要来自更大 effective batch 和更少碎片，不是 attention 数学更快。

### 3. Prefix cache 和 PagedAttention 有什么区别？

PagedAttention 管理单个或多个请求的 KV block 分配。prefix cache 判断不同请求是否有相同 token 前缀，并复用已经算好的 KV blocks。

### 4. 为什么 prefix cache 必须按 token 匹配？

模型看到的是 token ids 和 positions。文本看起来一样不代表 tokenization 一样；position、RoPE offset、chat template 差异也会让 KV 不可复用。

### 5. GQA/MQA 为什么能省 KV cache？

它减少 KV heads。query heads 可以多，KV heads 可以少，多个 query heads 共享同一组 K/V。

### 6. MLA cache 和普通 KV cache 的差别？

普通 cache 存每个 KV head 的 K/V。MLA 存低秩 latent KV 和 RoPE slice，decode 时通过吸收或投影恢复 attention 所需计算。

### 7. IndexShare 共享的是 KV cache 吗？

不是。它共享 DSA indexer 选出的 top-k token indices。主 MLA KV cache 仍按层维护；shared 层只是跳过自己的 indexer forward。

### 8. IndexShare 为什么对 1M context 特别有用？

DSA indexer 仍要对长上下文打分并 top-k。context 越长，indexer 成本越明显。跨层复用 indices 后，多数层可以跳过这段成本。

### 9. IndexShare 的风险是什么？

某些层可能确实需要不同 token。简单均匀共享可能伤质量，所以 IndexCache 论文用校准集 loss 搜索 pattern，或者在训练中用 multi-layer distillation 让 retained indexer 服务多层。

### 10. KV cache 量化为什么不总是免费收益？

低精度会影响 attention score 或 value aggregation，还需要 scale 存储和支持对应 dtype 的 kernel。长上下文和 retrieval 任务对误差更敏感。

### 11. spec decode 为什么会让 KV cache 管理更复杂？

候选 token 可能被拒绝。runtime 需要区分临时 KV、已接受 KV 和需要回滚的 KV，还要让 verify forward 支持多 token query。

### 12. 面试里如何一句话解释 GLM-5.2 IndexShare？

GLM-5.2 在 DSA sparse attention 中让一组连续层复用同一个 Full layer 的 top-k sparse indices，shared 层不再运行自己的 indexer，从而降低长上下文下 indexer dot product 和 top-k 的重复成本。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM PagedAttention design doc](https://docs.vllm.ai/en/latest/design/paged_attention/)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference](https://www.usenix.org/conference/osdi24/presentation/agrawal)
- [DistServe: Disaggregating Prefill and Decoding](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)
- [FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving](https://arxiv.org/abs/2501.01005)
- [vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention](https://arxiv.org/abs/2405.04437)
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference](https://arxiv.org/abs/2306.14048)
- [StreamingLLM: Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [SnapKV: LLM Knows What You are Looking for Before Generation](https://arxiv.org/abs/2404.14469)
- [CacheGen: KV Cache Compression and Streaming for Fast LLM Serving](https://arxiv.org/abs/2310.07240)
- [IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse](https://arxiv.org/abs/2603.12201)
- [GLM-5.2 official blog](https://huggingface.co/blog/zai-org/glm-52-blog)
- [GLM-5 official repository](https://github.com/zai-org/GLM-5)
- [ATOM GLM-5 recipe](https://github.com/ROCm/ATOM/blob/main/recipes/GLM-5.md)

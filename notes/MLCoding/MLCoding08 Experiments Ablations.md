# ML Coding · Experiments & Ablations

对应 CS336 Assignment 1：Section 7。

## Exercise 1 · Experiment Logger

对应 PDF：`experiment_log`

记录字段：

```text
run name
git commit / config hash
model config
optimizer config
dataset / tokenizer
train loss by step
validation loss by step
wall-clock time
tokens processed
checkpoint path
generated samples
```

输出：

```text
CSV / JSONL / wandb run
learning curve plot
experiment log markdown
```

## Exercise 2 · TinyStories LR Sweep

对应 PDF：`learning_rate`

实验设计：

```text
fixed model config
fixed total tokens processed
vary learning rate logarithmically
include at least one divergent run
```

输出：

```text
learning curves
best validation loss
divergence threshold
edge-of-stability discussion
```

## Exercise 3 · Batch Size Sweep

对应 PDF：`batch_size_experiment`

实验矩阵：

```text
batch_size = 1
batch_size = 64
batch_size = 128
batch_size near memory limit
```

记录：

```text
tokens/sec
step time
validation loss
GPU memory
best LR per batch size
```

## Exercise 4 · Generate TinyStories Samples

对应 PDF：`generate`

变量：

```text
temperature
top_p
prompt
checkpoint step
```

输出：

```text
sample text, at least 256 tokens or until EOS
sampling config
fluency comment
two factors affecting quality
```

## Exercise 5 · Remove RMSNorm

对应 PDF：`layer_norm_ablation`

实验：

```text
baseline pre-norm
no RMSNorm at previous best LR
no RMSNorm at lower LR
```

输出：

```text
learning curves
best stable LR
activation / gradient norm observations
normalization commentary
```

## Exercise 6 · Post-Norm Transformer

对应 PDF：`pre_norm_ablation`

对比：

```text
pre-norm:
  z = x + MHA(RMSNorm(x))
  y = z + FFN(RMSNorm(z))

post-norm:
  z = RMSNorm(x + MHA(x))
  y = RMSNorm(z + FFN(z))
```

输出：

```text
learning curve comparison
stability notes
best LR if retuned
```

## Exercise 7 · NoPE vs RoPE

对应 PDF：`no_pos_emb`

实验：

```text
baseline RoPE
NoPE model
same training budget
same tokenizer / data
```

输出：

```text
validation loss curve
sample generation
position information discussion
```

## Exercise 8 · SwiGLU vs SiLU FFN

对应 PDF：`swiglu_ablation`

约束：

```text
SwiGLU: three matrices, d_ff around 8/3 d_model
SiLU baseline: two matrices, d_ff around 4 d_model
parameter counts approximately matched
```

输出：

```text
learning curves
parameter count comparison
validation loss comparison
gating discussion
```

## Exercise 9 · OpenWebText Run

对应 PDF：`main_experiment`

输出：

```text
OpenWebText learning curve
generated sample
TinyStories vs OWT loss comparison
fluency analysis
```

分析重点：

OpenWebText 更杂、更长尾、更难压缩；同样模型和 compute budget 下，loss 和 generation quality 不能直接和 TinyStories 等价比较。

## Exercise 10 · Leaderboard-Style Modification

对应 PDF：`leaderboard`

可选方向：

```text
weight tying
better initialization
LR / batch size schedule
architecture small changes
tokenizer changes within allowed data
throughput optimization
```

输出：

```text
final validation loss
wall-clock-bounded learning curve
description of modification
evidence the modification helped
```

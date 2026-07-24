# 多目标学习与分数融合

## 第 7 章 多目标学习与分数融合

### 7.1 为什么会有多目标

短视频平台可能同时关心点击、播放时长、完播、点赞、关注和负反馈。电商关心点击、加购、下单和成交额。搜索还要考虑相关性、质量、时效、地域和个性化。

把所有目标粗暴加成一个标签，会丢掉结构。分别训练多个模型又会重复计算，并让低频任务缺少数据。多任务学习就在这两端之间找平衡。

### 7.2 Shared-Bottom

最简单的结构共享底层：

```text
features -> shared network -> task A tower
                           -> task B tower
                           -> task C tower
```

总损失：

```math
\mathcal L=\sum_t \lambda_t\mathcal L_t.
```

问题是任务梯度可能冲突。点击偏好标题吸引力，长时长偏好内容持续价值，它们不总朝同一方向更新共享参数。

### 7.3 MMoE

MMoE 用多个 expert 产生表示，每个任务有自己的 gate：

```math
h_t(x)
=\sum_{e=1}^{E}g_{t,e}(x)f_e(x),
```

```math
g_t(x)=\operatorname{softmax}(W_t x).
```

任务 `t` 根据样本动态组合 experts。它比 Shared-Bottom 更灵活，但不要把 gate 解释成稳定的业务分工。某个 expert 不一定永久代表"点击"，gate 也可能塌缩到少数 experts。

诊断 MMoE 时可以看：

- 各任务 gate 的熵；
- expert 使用是否均衡；
- 任务梯度余弦相似度；
- 单任务与多任务的分群收益；
- 低频任务是否被高频任务压制。

### 7.4 ESMM 与转化漏斗

电商 CVR 只在点击后可观察。若只用点击样本训练 CVR，训练分布与全量曝光分布不同。

ESMM 利用：

```math
P(\text{click and conversion})
=P(\text{click})P(\text{conversion}\mid\text{click}),
```

在全量曝光空间联合学习 CTR 和 CTCVR，再由二者关系约束 CVR。它缓解样本选择偏差与转化稀疏，但仍依赖模型假设和数据口径，不代表反事实问题完全解决。

### 7.5 时长建模

播放时长既有零膨胀，又受视频长度影响。直接回归秒数会偏向长视频。

可选做法：

- 预测有效播放和条件时长；
- 对时长做 log 或分桶；
- 预测播放比例；
- 分视频长度校准；
- 用 survival/hazard 思路建模退出。

评价时要按内容长度、用户活跃度和场景分桶。平均时长上涨可能只是系统多推了长视频。

### 7.6 分数融合

模型输出通常不能直接线性相加。CTR 可能在 `[0, 0.2]`，时长预测是秒，CVR 更稀疏。先做校准，再讨论融合。

常见形式：

```math
S
=w_1f_1(\hat p_{\text{click}})
+w_2f_2(\hat t)
+w_3f_3(\hat p_{\text{conversion}})
-w_4\hat p_{\text{negative}}.
```

`f_t` 可以是 log、幂函数、分段函数或分位数映射。权重不只靠离线搜索，最终需要在线实验。

另一条路是学习融合模型，把各目标分数和上下文作为输入。但它仍要有训练标签，且更难解释目标权衡。业务强约束最好保留在重排或规则层。

### 7.7 校准

如果模型说 0.2 的样本约有 20% 真正点击，分数就是校准的。常用方法：

- Platt scaling；
- isotonic regression；
- temperature scaling；
- 分场景或分人群校准。

排序只要求相对顺序，融合却经常需要可比较的概率。校准变化不一定改变 AUC，却可能大幅改变多目标融合的结果。

### 7.8 从排序损失到偏好优化

BCE 判断单个 pair，BPR 比较一对 item，InfoNCE 让一个正例与一组候选竞争。三者都利用正负反馈，比较粒度和负样本来源不同。

生成式推荐把比较单位扩展到 token 或完整序列。next-token CE 与整个词表竞争，DPO 比较 chosen/rejected 序列，policy gradient 用 advantage 给 rollout 加权。RL 的低 advantage rollout 不能简单当成固定负样本，因为候选由当前 policy 产生，样本权重也会随训练变化。细节见 [[BusinessAlgorithm05 Generative Recommendation.md#13.9 从正负样本到 RL：一条连续的坐标轴|生成式推荐中的偏好优化]]。

### 7.9 本章自测

1. Shared-Bottom 的负迁移从哪里来？
2. MMoE 的 gate 可以怎样诊断？
3. ESMM 解决了 CVR 的哪两个问题？
4. 为什么直接预测播放秒数会偏长视频？
5. AUC 不变时，校准为什么仍可能改善线上融合？
6. BCE、BPR、InfoNCE 的比较粒度有什么不同？

<details>
<summary>参考答案</summary>

1. 多个任务共享同一表示时，梯度方向可能冲突，数据量大的任务还会主导参数更新，使其他任务变差。
2. 看不同任务和样本上的 gate 分布、expert 使用率、熵与负载；长期只选一个 expert 或所有 gate 完全相同都值得检查。
3. 它用曝光→点击和点击→转化的联合建模缓解 CVR 样本选择偏差，并利用全曝光空间缓解只在点击样本上训练造成的数据稀疏。
4. 秒数上界随视频长度增长，模型容易把长度当作收益。可预测完播率、分桶时长或使用带长度归一化的目标。
5. AUC 只看相对顺序。融合多个目标时需要概率尺度可比，校准能避免某个头仅因分数偏大而压过其他目标。
6. BCE 判断单个样本；BPR 比较一对正负 item；InfoNCE 让正例与一组 batch 或采样候选共同竞争。

</details>

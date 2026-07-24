# 特征交叉、粗排与个性化

## 第 11 章 特征交叉、粗排与个性化

### 11.1 业务规律常在交叉项里

单看"用户年龄"或"内容类目"都不够，模型需要捕捉"某年龄段在某时段对某类内容的偏好"。稀疏特征组合数量巨大，人工枚举很快失控。

### 11.2 FM

Factorization Machine 对二阶交叉使用低秩向量：

```math
\hat y
=w_0+\sum_i w_ix_i
+\sum_{i<j}\langle v_i,v_j\rangle x_ix_j.
```

若 `x` 是 one-hot，多数维度为零，计算只涉及非零特征。内积共享统计强度，即使某个特征对很少共同出现，也能通过各自 embedding 学到合理交互。

FM 表达的是二阶交叉。更复杂关系需要深层网络或显式高阶结构。

### Quick Coding：FM 前向

实现一次 FM 前向计算，但不能写双重特征循环。使用平方和恒等式，把二阶交叉从 `O(d²k)` 降到 `O(dk)`，并用朴素两两计算验证结果。

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

### 11.3 DCN

DCN 的 cross layer 常写成：

```math
x_{l+1}
=x_0(x_l^\top w_l)+b_l+x_l.
```

每层都保留原始输入 `x_0`，逐层构造更高阶显式交叉。并行的 deep network 学习隐式关系，最后拼接。

DCN 给模型加入了有结构的乘性交互，参数量比暴力枚举高阶组合小。把它记成"比 MLP 多一个层"会漏掉这一点。

### 11.4 LHUC、SENet 与 FiBiNET

LHUC 对隐藏单元做条件化缩放：

```math
h' = a(u, c)\odot h,
```

其中 `a` 由用户或场景产生。它让同一底座在不同用户、域或场景下使用不同容量。

LHUC 最早用于语音识别；快手把这种按用户条件化隐藏单元的做法用于推荐精排，称为 PPNet。名字不同，先看清 gate 由哪些个性化特征产生、它缩放哪一层，以及是否与主干参数共同训练。

SENet 先根据整组特征生成权重，重标定各 field。FiBiNET 再对 field embedding 做双线性交互。它们适合 field 较多、重要性随样本变化的 CTR 场景。

别把这些模型当成必须背的缩写清单。记住三个问题更实用：

1. 它显式建模了什么交互？
2. 参数怎样在特征或人群间共享？
3. 线上增加了多少计算和特征依赖？

### 11.5 粗排

粗排夹在召回和精排之间。候选仍多，模型必须便宜；但只用召回分数又会误杀。

常见策略：

- 小型 MLP/三塔；
- 精排蒸馏；
- 只使用可批量读取的特征；
- 特征选择与低精度推理；
- 分层 top-k 或早退。

三塔粗排把计算拆成：

```text
用户塔：用户画像 + 场景，只算一次
物品塔：静态物品特征，输出尽量缓存
交叉塔：实时统计与交叉特征，对每个候选计算，但网络要小
```

三塔并没有让每个候选的计算消失。交叉塔和上层多目标网络仍要跑 `N` 次；收益来自把最重的用户计算摊到整次请求，并缓存较静态的物品表示。物品画像更新后要主动失效缓存，统计特征则不能假装成静态特征长期缓存。

评价粗排时，单独 AUC 的信息不够。更该问的是："在给定算力下，它保住了多少精排会选中的候选？"可用 top-k 一致性、精排 top-N recall 和线上误杀分析来回答。

### 11.6 个性化特征怎样进入模型

user_id embedding 对活跃用户很强，对新用户和跨域用户无能为力。稳健的个性化通常结合：

- ID 与统计特征；
- 长短期行为；
- 人群与场景；
- 内容语义；
- 实时意图。

排序特征可以按来源检查：

- 用户画像：ID、活跃度、人口属性、长期类目兴趣；
- 物品画像：ID、作者、类目、文本、多模态质量、发布时间；
- 用户统计：不同时间窗、内容类型和类目下的曝光与交互；
- 物品/作者统计：不同人群和时间窗下的曝光与转化；
- 场景：时间、地域、设备、入口；
- 交叉：用户对当前类目/作者的历史偏好。

连续计数常做 `log(1+x)`，转化率需要平滑，缺失特征要有覆盖率报表。一个特征只覆盖 20% 用户时，离线总体增益会被稀释；线上缺失默认值若与训练不同，还会直接制造分布偏移。

个性化越强，越要留意隐私、过滤气泡和反馈闭环。搜索中还需把显式 query 放在用户历史之前，不能让长期画像压过当前需求。

### 11.7 本章自测

1. FM 为什么能处理稀疏特征二阶交叉？
2. DCN 的 cross layer 与普通 MLP 有什么不同？
3. LHUC 的条件化缩放适合哪些场景？
4. 粗排为什么不应只看自己的 AUC？
5. 搜索个性化为什么要让 query 保持主导？

<details>
<summary>参考答案</summary>

1. 每个特征拥有低维向量，二阶交叉用向量内积共享统计强度，因此未被频繁共同观察的稀疏组合也能泛化。
2. Cross layer 显式构造有界阶数的特征乘积，并保留原输入；MLP 通过非线性隐式学习交互，结构约束更弱。
3. 适合用用户、场景或域信息对共享网络做轻量条件化，例如多场景排序和用户群适配；它不适合替代缺失的核心特征。
4. 粗排目标是保住精排会选中的候选。即使自身 AUC 高，只要误杀精排 top item，整条链路仍会退化。
5. 搜索表达的是当前明确意图。个性化只能在相关候选内调序，不能让历史偏好把不相关商品推到前面。

</details>

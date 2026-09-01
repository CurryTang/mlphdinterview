# 特征交叉、粗排与个性化

## 第 11 章 特征交叉、粗排与个性化

### 11.1 业务规律常在交叉项里

单看"用户年龄"或"内容类目"都不够，模型需要捕捉"某年龄段在某时段对某类内容的偏好"。稀疏特征组合数量巨大，人工枚举很快失控。

交叉项表达的是条件关系。`video_length` 的全局均值意义有限，`video_length × user_session_depth` 才可能说明用户在本次 session 还有没有耐心看长内容；`query_term × title_term` 表达文本是否对齐；`user_category_history × item_category` 表达当前候选是否命中已有兴趣。

这类关系有两种难点。第一是稀疏：`城市 × 品牌 × 时间段` 的组合数增长很快，许多组合在训练中只出现几次。第二是更新：交叉里若包含实时统计，训练和服务必须使用同一窗口与默认值，否则模型学到的不是业务规律，而是数据管道差异。

选择结构时先看要表达什么，不要从缩写表倒推问题：

- 已知少量稳定组合，可直接构造显式交叉，解释和上线都简单；
- 大量稀疏二阶组合适合 FM 一类低秩共享；
- 需要可控的高阶乘性交互时，可用 DCN；
- 关系复杂且数据足够，MLP 能学隐式交互，但更难判断它究竟用了什么。

ID 交叉很容易记住头部用户和物品。离线随机切分时它可能表现很好，换到新品、长尾或未来时间窗口就失效。因此交叉模型除了总体 AUC，还应按新老 item、特征覆盖率和时间切片检查泛化。

### 11.2 FM

Factorization Machine 对二阶交叉使用低秩向量：

```math
\hat y
=w_0+\sum_i w_ix_i
+\sum_{i<j}\langle v_i,v_j angle x_ix_j.
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
\left[ \left( \sum_i v_{i,f}x_i \right)^2
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

### 11.3 DCN 与 DCN-v2 (Deep & Cross Network)

DCN 的核心是显式 Cross Layer：

$$\mathbf{x}_{l+1} = \mathbf{x}_0 \odot (\mathbf{W}_l \mathbf{x}_l) + \mathbf{b}_l + \mathbf{x}_l$$

在 **DCN-v2** 中，为了降低全连接矩阵 $\mathbf{W}_l \in \mathbb{R}^{d \times d}$ 的计算开销并捕捉低秩子空间，引入了**低秩矩阵分解（Low-Rank Matrix Decomposition: $\mathbf{W}_l = \mathbf{U}_l \mathbf{V}_l^T$）与混合门控路由（MoE-style Subspaces）**：

#### 伪代码实现：DCN-v2 Cross Network 前向
```python
import torch
import torch.nn as nn

class CrossNetworkV2(nn.Module):
    def __init__(self, in_features, num_layers=3, low_rank=32, num_experts=4):
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        # 低秩分解: W_l = U_l @ V_l.T, U in R^(d x r), V in R^(d x r)
        self.u_list = nn.ParameterList([
            nn.Parameter(torch.randn(num_experts, in_features, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.v_list = nn.ParameterList([
            nn.Parameter(torch.randn(num_experts, in_features, low_rank) * 0.01)
            for _ in range(num_layers)
        ])
        self.gate_list = nn.ModuleList([
            nn.Linear(in_features, num_experts, bias=False) for _ in range(num_layers)
        ])
        self.bias_list = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features)) for _ in range(num_layers)
        ])

    def forward(self, x0):
        # x0: [batch_size, in_features]
        xl = x0
        for l in range(self.num_layers):
            # 1. 动态计算 Expert 门控权重: [batch_size, num_experts, 1]
            gate = torch.softmax(self.gate_list[l](xl), dim=-1).unsqueeze(-1)
            # 2. 低秩矩阵变换: (xl @ V) @ U.T
            # xl: [B, 1, d] -> xl @ V: [B, E, r] -> (xl @ V) @ U.T: [B, E, d]
            xl_expanded = xl.unsqueeze(1) # [B, 1, d]
            v = self.v_list[l] # [E, d, r]
            u = self.u_list[l] # [E, d, r]
            low_rank_out = torch.einsum('bmd,edr->ber', xl_expanded, v) # [B, E, r]
            expert_out = torch.einsum('ber,edr->bed', low_rank_out, u)   # [B, E, d]
            # 3. MoE 专家加权汇聚: [B, d]
            moe_out = (expert_out * gate).sum(dim=1)
            # 4. 显式特征交叉与残差相加
            xl = x0 * (moe_out + self.bias_list[l]) + xl
        return xl
```

---

### 11.4 DLRM (Deep Learning Recommendation Model)

Meta DLRM 架构将特征解耦为数值 Dense 特征与稀疏 Sparse 类别特征：
1. **Bottom MLP**：将连续稠密特征映射为 $d$ 维稠密向量；
2. **Embedding Tables**：为各 Sparse 类别 ID 查表得到 $d$ 维嵌入向量；
3. **Explicit Dot-Product Triu Interaction**：计算所有 $d$ 维向量之间的两两点积（Dot Product），提取上三角矩阵元素并拉平；
4. **Top MLP**：将点积交叉结果与原始 Bottom 特征拼接后送入 Top MLP 输出预估概率。

#### 伪代码实现：DLRM 特征交互与前向
```python
class DLRM(nn.Module):
    def __init__(self, embedding_sizes, dense_in_dim, embed_dim=64):
        super().__init__()
        # 1. Sparse 嵌入表
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim) for num_classes in embedding_sizes
        ])
        # 2. Bottom MLP 处理 Dense 数值特征
        self.bottom_mlp = nn.Sequential(
            nn.Linear(dense_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        # 3. 计算两两点积组合数: (num_sparse + 1) * num_sparse / 2
        num_fields = len(embedding_sizes) + 1
        num_interactions = (num_fields * (num_fields - 1)) // 2
        # 4. Top MLP 最终分类
        self.top_mlp = nn.Sequential(
            nn.Linear(num_interactions + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, dense_x, sparse_x):
        # dense_x: [B, dense_in_dim], sparse_x: [B, num_sparse]
        v_dense = self.bottom_mlp(dense_x).unsqueeze(1) # [B, 1, embed_dim]
        v_sparse = [emb(sparse_x[:, i]).unsqueeze(1) for i, emb in enumerate(self.embeddings)]
        # 全部字段向量拼接: [B, num_fields, embed_dim]
        all_embeddings = torch.cat([v_dense] + v_sparse, dim=1)
        # 批量矩阵乘法计算两两内积: [B, num_fields, num_fields]
        dot_interactions = torch.bmm(all_embeddings, all_embeddings.transpose(1, 2))
        # 提取上三角非对角线元素 (去冗余): [B, num_interactions]
        triu_indices = torch.triu_indices(all_embeddings.size(1), all_embeddings.size(1), offset=1)
        flat_interactions = dot_interactions[:, triu_indices[0], triu_indices[1]]
        # 拼接原始 Bottom 特征后喂入 Top MLP
        top_input = torch.cat([flat_interactions, v_dense.squeeze(1)], dim=-1)
        return self.top_mlp(top_input)
```


### 11.5 LHUC、SENet 与 FiBiNET
、SENet 与 FiBiNET

LHUC 对隐藏单元做条件化缩放：

```math
h' = a(u, c)\odot h,
```

其中 `a` 由用户或场景产生。它让同一底座在不同用户、域或场景下使用不同容量。

LHUC 最早用于语音识别；快手把这种按用户条件化隐藏单元的做法用于推荐精排，称为 PPNet。名字不同，先看清 gate 由哪些个性化特征产生、它缩放哪一层，以及是否与主干参数共同训练。

SENet 先根据整组特征生成权重，重标定各 field。FiBiNET 再对 field embedding 做双线性交互。它们适合 field 较多、重要性随样本变化的 CTR 场景。

Gate 也会失败。用户历史很少时，个性化 gate 可能只是在放大噪声；sigmoid 长期饱和后，一些隐藏单元几乎收不到梯度；线上 field 缺失率变化，又会让 SENet 的权重分布整体漂移。服务侧要记录 gate 分布、特征覆盖率和各分群收益，并保留不经过条件化分支的底座分数作为回退。

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

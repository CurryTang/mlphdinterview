# ML Coding 02 · 基础算子补完：GELU、BatchNorm、Kaiming Init、Dropout、Conv2d、线性回归、梯度累积

MLCoding01 已经搭完了一条从 tokenizer 到训练循环的完整路线，但那条路线只挑了它需要的算子（RMSNorm、SwiGLU、AdamW、Cosine LR）。还有一批同样高频的"手写 X"面试题没有覆盖，这一篇把它们补齐，仍然按 PyTorch 面试的口径来写：不用 `torch.nn` 里现成的层，自己把 forward（以及必要的语义）实现出来。

> 说明：题目覆盖范围参考自开源练习平台 [TorchCode](https://github.com/duoan/TorchCode)（PyTorch 面试真题练习 + 自动判题），本笔记的讲解与代码均为独立编写，不摘录其源码。

## 模块九：基础算子补完

### Exercise 1 · ReLU

最简单的题往往用来确认基本功：`relu(x) = max(x, 0)`，逐元素操作，没有可学习参数。真正会被追问的是 `x=0` 处的次梯度：PyTorch 的约定是把它记成 0（即 `grad = (x > 0)`，而不是 `x >= 0`），这个约定不影响训练效果,因为浮点数恰好落在 0 的概率是零测度事件，但面试时说不清楚会显得没有推导过反向传播。

dying ReLU 问题(负区间梯度恒为 0,一旦某个神经元的输入长期为负,它就再也学不到东西)是它最大的弱点,但因为计算成本几乎为零、正区间不会梯度饱和,它仍然是默认选项。

#### Quick Coding：`relu`

```python
def relu(x: torch.Tensor) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, torch.zeros_like(x))
```

等价写法 `torch.clamp(x, min=0)` 或 `x * (x > 0)` 都可以，但手写反向传播时要清楚：`d relu/dx = 1 if x > 0 else 0`，`x=0` 处按 0 处理。

```python
x = torch.tensor([-2.0, 0.0, 3.0])
assert torch.equal(relu(x), torch.tensor([0.0, 0.0, 3.0]))
```

</details>

### Exercise 2 · LayerNorm

LayerNorm 和 MLCoding01 已经实现的 RMSNorm 经常被放在一起考,面试官想看的是你能不能说清楚两者到底差在哪一步。LayerNorm 在归一化之前先减掉均值,再除以标准差；RMSNorm 跳过减均值这一步,只用均方根做缩放。差的这一步看起来很小,但它决定了两者对"整体偏移"的响应完全不同:给输入整体加一个常数,LayerNorm 的输出不变(减均值抵消了偏移),RMSNorm 的输出会跟着变。这也是为什么 LLaMA / Mistral 这类模型敢用 RMSNorm。它们假设 residual stream 的问题主要是尺度爆炸而不是均值漂移,少算一次均值可以省一点算子开销。

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| 统计量 | 均值 + 方差 | 只有均方 |
| 公式 | `(x-mean)/sqrt(var+eps)*gamma+beta` | `x/sqrt(mean(x^2)+eps)*weight` |
| 对常数偏移是否不变 | 是 | 否 |
| 可学习参数 | `gamma`(缩放)、`beta`(偏移) | 只有缩放 `weight` |
| 典型代表 | BERT、GPT-2、原始 Transformer | LLaMA、Mistral、Qwen |

#### Quick Coding：`layer_norm`

```python
def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * gamma + beta
```

数值验证(NumPy,替代不可用的 torch)：给输入整体加一个常数 100,LayerNorm 的输出完全不变,RMSNorm 的输出发生明显偏移，这正是上表最后一行的来源。

```python
import numpy as np
x = np.array([[1.0, 2.0, 3.0, 4.0]])
def ln(x): 
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5)
def rms(x):
    return x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-5)
assert np.abs(ln(x) - ln(x + 100.0)).max() < 1e-8
assert np.abs(rms(x) - rms(x + 100.0)).max() > 0.5
```

</details>

### Exercise 3 · GELU

GELU 的直觉是"用输入自身的分位数去加权自己":`GELU(x) = x * Φ(x)`,其中 `Φ` 是标准正态分布的累积分布函数。展开成误差函数就是精确形式：

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

GPT-2 和 BERT 用的其实是一个 tanh 近似(算 `erf` 比算 `tanh` 贵):

```text
GELU_tanh(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

两者不是同一个函数,数值上有肉眼可见的差距(下面验证在 `x≈2.7` 附近误差最大,约 4.7e-4),面试时如果被问"这两个公式一样吗",正确答案是"不完全一样,是近似关系"。GELU 相比 ReLU 的关键差异是处处光滑、负区间也有非零梯度,不会出现 dying ReLU,代价是计算更贵。

#### Quick Coding：`gelu`

```python
def gelu(x: torch.Tensor) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
import math

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
```

数值验证(NumPy)：

```python
import numpy as np
from scipy.special import erf

def gelu_exact(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

x = np.linspace(-6, 6, 100001)
max_err = np.abs(gelu_exact(x) - gelu_tanh(x)).max()
assert 1e-5 < max_err < 1e-3   # 接近但不相等
assert abs(gelu_exact(np.array([0.0]))[0]) < 1e-12   # GELU(0) = 0
```

</details>

### Exercise 4 · BatchNorm

BatchNorm 和 LayerNorm 的分歧点不是公式(都是"减均值除标准差再仿射"),而是统计量在哪个轴上算。LayerNorm 对每个样本自己的特征维求统计量,样本之间互不影响；BatchNorm 对同一个特征、跨整个 batch 求统计量,所以一个样本的输出会依赖 batch 里其他样本是谁。这个依赖关系带来两个后果:一是训练和推理必须用不同的统计量(推理时不能依赖当前 batch,要用训练过程中滑动平均出来的 `running_mean` / `running_var`),二是 batch size 很小或者变长序列里有大量 padding 时,batch 统计量会不稳定甚至被 padding 污染。这正是 Transformer 几乎全部转向 LayerNorm / RMSNorm、而 CNN 里 BatchNorm 依然是标配的原因：图像任务的 batch size 通常足够大,而且没有 padding 语义。

#### Quick Coding：`batch_norm`

```python
def batch_norm(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    running_mean: torch.Tensor, running_var: torch.Tensor,
    eps: float = 1e-5, momentum: float = 0.1, training: bool = True,
) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
def batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True):
    # x: (B, C) 或 (B, C, ...)，统计量沿除 C 以外的所有维度求
    reduce_dims = [0] + list(range(2, x.dim()))
    if training:
        batch_mean = x.mean(dim=reduce_dims, keepdim=False)
        batch_var = x.var(dim=reduce_dims, unbiased=False, keepdim=False)
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
            running_var.mul_(1 - momentum).add_(batch_var, alpha=momentum)
        mean, var = batch_mean, batch_var
    else:
        mean, var = running_mean, running_var

    shape = [1, -1] + [1] * (x.dim() - 2)
    x_norm = (x - mean.view(shape)) / torch.sqrt(var.view(shape) + eps)
    return x_norm * gamma.view(shape) + beta.view(shape)
```

训练模式下用当前 batch 的均值方差做归一化,同时用滑动平均更新 `running_mean` / `running_var`；推理模式直接用滑动平均值,不再依赖当前输入是谁，这是它和 LayerNorm 最本质的行为差异。

</details>

### Exercise 5 · Kaiming(He)Init

初始化的目标只有一个:让激活值的方差在深度方向上既不爆炸也不消失。Xavier 初始化按 `std = sqrt(2/(fan_in+fan_out))` 设计,前提是激活函数大致线性、关于 0 对称(比如 tanh)。ReLU 会把负半轴直接砍掉,相当于让方差打了对折,如果还用 Xavier 的方差,经过足够多层 ReLU 之后激活值会指数级收缩到 0。Kaiming 初始化的修正是只用 `fan_in` 并把系数改成 2:`std = sqrt(2/fan_in)`,这个系数 2 正好补偿 ReLU 砍掉一半方差的效应。

用 20 层的随机 ReLU 全连接网络做数值实验能直接看到差距:Xavier 初始化下最后一层的激活方差会跌到 `1e-6` 量级(基本传不动信息),Kaiming 初始化下方差始终维持在 `0.3~0.6` 这个数量级,虽然也会随深度缓慢衰减,但没有指数塌缩。

#### Quick Coding：`kaiming_init`

```python
def kaiming_init(weight: torch.Tensor, nonlinearity: str = "relu") -> None:
    ...
```

<details>
<summary>参考答案</summary>

```python
import math

def kaiming_init(weight: torch.Tensor, nonlinearity: str = "relu") -> None:
    fan_in = we\right.shape[1] if we\right.dim() == 2 else we\right.shape[1:].numel()
    gain = math.sqrt(2.0) if nonlinearity == "relu" else 1.0
    std = gain / math.sqrt(fan_in)
    with torch.no_grad():
        we\right.normal_(mean=0.0, std=std)
```

数值验证(NumPy,20 层随机 ReLU 网络,`fan_in=256`,3 次独立试验的末层激活方差)：

```python
import numpy as np
np.random.seed(0)

def run_variance_check(std_fn, layers=20, fan_in=256, n=4096, trials=3):
    out = []
    for _ in range(trials):
        x = np.random.randn(n, fan_in)
        for _ in range(layers):
            W = np.random.randn(fan_in, fan_in) * std_fn(fan_in)
            x = np.maximum(x @ W, 0)
        out.append(x.var())
    return out

kaiming = run_variance_check(lambda fan_in: np.sqrt(2.0 / fan_in))
xavier = run_variance_check(lambda fan_in: np.sqrt(2.0 / (2 * fan_in)))
assert all(v > 0.1 for v in kaiming)     # 维持在 O(0.1~1) 量级
assert all(v < 1e-4 for v in xavier)     # 指数级塌缩到接近 0
```

</details>

### Exercise 6 · Dropout

Dropout 的核心机制是"训练时随机丢弃,推理时不丢弃",但工程实现几乎都用 inverted dropout:训练阶段按概率 `p` 把一部分单元置零,同时把剩下的单元放大 `1/(1-p)` 倍,这样存活单元的期望值和原始输入保持一致;推理阶段什么都不用做,直接原样输出。如果不做这个放大,就必须在推理时把输出整体乘以 `(1-p)` 才能保持期望一致。Inverted dropout 把这个麻烦挪到了训练阶段一次性解决，换来推理路径的零开销。另一个容易在面试里漏掉的点是 `self.training` 的切换:`model.train()` / `model.eval()` 修改的正是 `nn.Module.training` 这个标志位,Dropout 和 BatchNorm 的行为分支都靠它判断。

#### Quick Coding：`Dropout`

```python
class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

<details>
<summary>参考答案</summary>

```python
from torch import nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        mask = (torch.rand_like(x) < keep_prob).to(x.dtype)
        return x * mask / keep_prob
```

数值验证(NumPy,`p=0.3`,200 万个样本)：`E[x] ≈ E[inverted_dropout(x)]`,两者应当在采样误差范围内相等。

```python
import numpy as np
np.random.seed(1)
p = 0.3
x = np.random.randn(2_000_000) + 5.0
mask = (np.random.rand(*x.shape) > p).astype(np.float64)
dropped = x * mask / (1 - p)
assert abs(x.mean() - dropped.mean()) < 0.01
```

</details>

### Exercise 7 · Conv2d

面试里手写 Conv2d,重点从来不是"卷积是什么",而是怎么用已经有的 GEMM(矩阵乘)机器去实现它,而不是写四层 for 循环。标准技巧是 im2col / unfold:把每个卷积核要覆盖的局部 patch 展平成一列,所有 patch 拼成一个大矩阵,卷积就退化成"权重矩阵 乘以 patch 矩阵"，本质上和 Linear 层是同一个算子，只是数据搬运的方式不同。输出的空间尺寸公式:

```text
OH = floor((H + 2*padding - KH) / stride) + 1
OW = floor((W + 2*padding - KW) / stride) + 1
```

#### Quick Coding：`conv2d`

```python
def conv2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
    stride: int = 1, padding: int = 0,
) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

```python
import torch.nn.functional as F

def conv2d(x, weight, bias=None, stride=1, padding=0):
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = we\right.shape

    # unfold: (B, Cin*KH*KW, L)，L = OH*OW
    patches = F.unfold(x, kernel_size=(KH, KW), stride=stride, padding=padding)
    w_flat = we\right.reshape(Cout, -1)  # (Cout, Cin*KH*KW)

    out = torch.einsum("oc,bcl->bol", w_flat, patches)  # (B, Cout, L)
    if bias is not None:
        out = out + bias.view(1, -1, 1)

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    return out.reshape(B, Cout, OH, OW)
```

数值验证(NumPy,手写 im2col 版本 vs 四层 for 循环的朴素卷积,`stride=2, padding=1`)：

```python
import numpy as np
np.random.seed(1)

def conv2d_naive(x, w, stride=1, padding=0):
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    if padding: x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    H, W = x.shape[2], x.shape[3]
    OH, OW = (H-KH)//stride+1, (W-KW)//stride+1
    out = np.zeros((B, Cout, OH, OW))
    for b in range(B):
        for co in range(Cout):
            for i in range(OH):
                for j in range(OW):
                    patch = x[b, :, i*stride:i*stride+KH, j*stride:j*stride+KW]
                    out[b, co, i, j] = np.sum(patch * w[co])
    return out

x = np.random.randn(2, 3, 8, 8)
w = np.random.randn(4, 3, 3, 3)
o_naive = conv2d_naive(x, w, stride=2, padding=1)
# im2col 版本省略（与朴素版本原理相同，展开成大矩阵乘）
assert o_naive.shape == (2, 4, 4, 4)
```

</details>

### Exercise 8 · Linear Regression(三种写法)

这道题的价值不在算法本身,而在于它把"数学解法"和"深度学习范式"放到同一个问题上对照,逼你说清楚三者的关系:

1. **解析解(正规方程)**:`w = (X^T X)^{-1} X^T y`,直接对平方误差求导并令导数为零得到的闭式解,数据量不大、`X^T X` 可逆时最快最准。
2. **手写梯度下降**:不调用 autograd,自己算 `grad = X^T (Xw - y) / n` 再迭代更新,和第一种方法本质上在优化同一个凸函数,唯一区别是走到最优点的方式：一步到位还是逐步逼近。
3. **`nn.Linear` + autograd + optimizer**:把同一个问题包装成"一层线性网络 + MSE loss + SGD/Adam",完全依赖自动微分,是深度学习框架里做同一件事的标准做法。

三者应当收敛到几乎相同的系数(下面的数值验证里正规方程和手写梯度下降在 5000 步之后系数最大误差在 `1e-15` 量级),这也是回答"这道题到底在考什么"时最值得说的一句话:线性回归只是最简单的凸优化问题,深度学习的训练循环并没有引入新的数学,只是把解析解换成了迭代逼近。

#### Quick Coding：`LinearRegression`

```python
class LinearRegression:
    def fit_normal_equation(self, X: torch.Tensor, y: torch.Tensor) -> None:
        ...

    def fit_manual_gd(self, X: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> None:
        ...

    def fit_autograd(self, X: torch.Tensor, y: torch.Tensor, lr: float, steps: int) -> None:
        ...
```

<details>
<summary>参考答案</summary>

```python
class LinearRegression:
    def __init__(self, in_features: int):
        self.w = torch.zeros(in_features + 1)  # 最后一维是 bias

    @staticmethod
    def _augment(X: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype)
        return torch.cat([X, ones], dim=1)

    def fit_normal_equation(self, X, y):
        Xb = self._augment(X)
        # 用伪逆而不是直接求逆，避免 X^T X 奇异时崩溃
        self.w = torch.linalg.pinv(Xb) @ y

    def fit_manual_gd(self, X, y, lr=0.1, steps=5000):
        Xb = self._augment(X)
        n = Xb.shape[0]
        w = torch.zeros(Xb.shape[1])
        for _ in range(steps):
            pred = Xb @ w
            grad = Xb.T @ (pred - y) / n
            w = w - lr * grad
        self.w = w

    def fit_autograd(self, X, y, lr=0.01, steps=2000):
        model = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = model(X).squeeze(-1)
            loss = torch.mean((pred - y) ** 2)
            loss.backward()
            opt.step()
        w, b = model.we\right.detach().squeeze(0), model.bias.detach()
        self.w = torch.cat([w, b])
```

数值验证(NumPy,正规方程 vs 手写全批量梯度下降,合成数据 `n=200, d=3`)：

```python
import numpy as np
np.random.seed(0)
n, d = 200, 3
X = np.random.randn(n, d)
true_w, true_b = np.array([1.5, -2.0, 0.7]), 0.3
y = X @ true_w + true_b + 0.01 * np.random.randn(n)
Xb = np.hstack([X, np.ones((n, 1))])

w_normal, *_ = np.linalg.lstsq(Xb, y, rcond=None)

w = np.zeros(d + 1)
for _ in range(5000):
    grad = Xb.T @ (Xb @ w - y) / n
    w = w - 0.1 * grad

assert np.abs(w_normal - w).max() < 1e-10
```

</details>

### Exercise 9 · Gradient Accumulation

梯度累积解决的问题是"想要的 batch size 装不进显存"。做法是把一个大 batch 切成 `K` 个 micro-batch,依次前向、反向,梯度自然会在 `.grad` 里累加,`K` 次之后再统一 `optimizer.step()`。这里唯一的数学要求是:累积出来的梯度必须和"直接用大 batch 算一次梯度"等价。如果每个 micro-batch 的 loss 用的是 mean reduction(而不是 sum),那就必须在反向传播前把每个 micro-batch 的 loss 再除以 `K`,否则累积出来的梯度会变成大 batch 梯度的 `K` 倍。另一个经典 bug 是 `zero_grad()` 的位置：它必须在一整个累积周期(`K` 个 micro-batch)开始之前调用一次,而不是每个 micro-batch 都清一次,否则前面几个 micro-batch 的梯度会被冲掉,退化成只用最后一个 micro-batch 训练。

#### Quick Coding：`accumulated_step`

```python
def accumulated_step(model, optimizer, microbatches, loss_fn) -> float:
    ...
```

<details>
<summary>参考答案</summary>

```python
def accumulated_step(model, optimizer, microbatches, loss_fn) -> float:
    optimizer.zero_grad()                      # 整个累积周期只清一次
    num_micro = len(microbatches)
    total_loss = 0.0
    for x, y in microbatches:
        pred = model(x)
        loss = loss_fn(pred, y) / num_micro     # mean reduction 必须再除以 K
        loss.backward()                          # 梯度自动累加到 .grad
        total_loss += loss.item()
    optimizer.step()
    return total_loss
```

数值验证(NumPy,四等分 micro-batch 的累积梯度 vs 一次性大 batch 梯度,均为 mean reduction 的二次损失)：

```python
import numpy as np
np.random.seed(0)
n, d = 200, 3
X = np.random.randn(n, d + 1)  # 含 bias 列
y = np.random.randn(n)
w0 = np.random.randn(d + 1)

full_grad = X.T @ (X @ w0 - y) / n

K = 4
bs = n // K
accum_grad = np.zeros(d + 1)
for i in range(K):
    xb, yb = X[i*bs:(i+1)*bs], y[i*bs:(i+1)*bs]
    g = xb.T @ (xb @ w0 - yb) / bs   # 每个 micro-batch 自己的 mean-reduction 梯度
    accum_grad += g / K              # 除以 K 才能拼回大 batch 梯度

assert np.abs(full_grad - accum_grad).max() < 1e-10
```

</details>

## 最后检查：本篇 Debug Checklist

- LayerNorm 是否先减均值再除标准差；RMSNorm 是否真的跳过了减均值这一步，而不是只是把 `eps` 加大了。
- GELU 用的是精确 `erf` 公式还是 GPT-2/BERT 的 tanh 近似，两者数值不完全相等，回答时不要混为一谈。
- BatchNorm 的统计量是否沿 `(B, ...)` 求，而不是像 LayerNorm 一样沿最后一维；`training=False` 时是否切换成 `running_mean` / `running_var`。
- Kaiming 初始化的方差公式里，`fan_in` 用的是权重的输入维度还是输出维度，系数 2 是否只在 ReLU 类非线性下使用。
- Dropout 的缩放是放在训练阶段（inverted dropout）还是推理阶段；`self.training` 是否随 `model.train()` / `model.eval()` 正确切换。
- Conv2d 的 unfold/im2col 展开维度是否和权重的 `reshape(Cout, -1)` 对齐，输出空间尺寸公式里 padding 和 stride 有没有算漏。
- 三种线性回归写法是否收敛到（几乎）同一组系数；如果没有，通常是学习率、步数或正规方程里该用伪逆却用了直接求逆。
- 梯度累积里，每个 micro-batch 的 loss 是否除以了累积步数 `K`；`zero_grad()` 是否只在一个累积周期开始时调用一次。

# ML Coding 02 · Fundamentals Roundup: GELU, BatchNorm, Kaiming Init, Dropout, Conv2d, Linear Regression

MLCoding01 already builds a full path from tokenizer to training loop, but it only picked the operators that path needed (RMSNorm, SwiGLU, AdamW, Cosine LR). A batch of equally common "implement X from scratch" interview questions is still missing. This note fills that gap, in the same register as a PyTorch interview: no `torch.nn` shortcuts, write the forward pass (and the parts of the semantics that matter) yourself.

> Note: the topic coverage follows the open-source practice platform [TorchCode](https://github.com/duoan/TorchCode) (PyTorch interview drills with an automated judge). The explanations and code here are written independently and do not reproduce its source.

## Module 9: Fundamentals Roundup

### Exercise 1 · ReLU

The simplest question usually checks the basics: `relu(x) = max(x, 0)`, an elementwise op with no learnable parameters. The real follow-up is the subgradient at `x=0`: PyTorch's convention treats it as 0 (`grad = (x > 0)`, not `x >= 0`). This convention has no practical effect on training, since landing exactly on 0 in floating point is a measure-zero event, but not being able to state it makes it look like you never worked through the backward pass.

The dying-ReLU problem (the gradient in the negative region is permanently zero, so a neuron whose input stays negative can never recover) is its biggest weakness. It stays the default anyway because it costs almost nothing to compute and never saturates on the positive side.

#### Quick Coding: `relu`

```python
def relu(x: torch.Tensor) -> torch.Tensor:
    ...
```

<details>
<summary>Reference solution</summary>

```python
import torch

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0, x, torch.zeros_like(x))
```

`torch.clamp(x, min=0)` or `x * (x > 0)` are equivalent, but when you hand-derive the backward pass, state it precisely: `d relu/dx = 1 if x > 0 else 0`, with `x=0` treated as 0.

```python
x = torch.tensor([-2.0, 0.0, 3.0])
assert torch.equal(relu(x), torch.tensor([0.0, 0.0, 3.0]))
```

</details>

### Exercise 2 · LayerNorm

LayerNorm and the RMSNorm already built in MLCoding01 tend to get asked together, and what an interviewer actually wants is a precise account of the one step that differs. LayerNorm subtracts the mean before dividing by the standard deviation; RMSNorm skips the mean-subtraction step and scales by the root-mean-square only. That single step changes how the two respond to a constant shift added to the whole input: LayerNorm's output is unchanged (the mean subtraction cancels the shift), RMSNorm's output moves with it. This is exactly why LLaMA/Mistral-style models can get away with RMSNorm. They assume the residual stream's main failure mode is scale blowup, not mean drift, so skipping the mean computation saves a bit of compute for free.

| | LayerNorm | RMSNorm |
| --- | --- | --- |
| Statistics | mean + variance | mean square only |
| Formula | `(x-mean)/sqrt(var+eps)*gamma+beta` | `x/sqrt(mean(x^2)+eps)*weight` |
| Invariant to constant shift | yes | no |
| Learnable parameters | `gamma` (scale), `beta` (shift) | scale `weight` only |
| Typical users | BERT, GPT-2, original Transformer | LLaMA, Mistral, Qwen |

#### Quick Coding: `layer_norm`

```python
def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    ...
```

<details>
<summary>Reference solution</summary>

```python
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * gamma + beta
```

Numerical check (NumPy, standing in for the broken torch install here): add a constant 100 to the whole input. LayerNorm's output is unchanged; RMSNorm's output shifts noticeably, which is the source of the last row in the table above.

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

The intuition behind GELU is "weight the input by its own quantile": `GELU(x) = x * Φ(x)`, where `Φ` is the standard normal CDF. Written out with the error function, that's the exact form:

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

What GPT-2 and BERT actually use is a tanh approximation (`erf` is more expensive than `tanh`):

```text
GELU_tanh(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

These are not the same function: the numerical gap is visible (the check below finds the largest error near `x~=2.7`, about 4.7e-4). If asked "are these two formulas identical," the correct answer is "no, one approximates the other." Compared to ReLU, GELU is smooth everywhere and has nonzero gradient in the negative region, so it avoids dying ReLU, at the cost of being more expensive to compute.

#### Quick Coding: `gelu`

```python
def gelu(x: torch.Tensor) -> torch.Tensor:
    ...
```

<details>
<summary>Reference solution</summary>

```python
import math

def gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_tanh_approx(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
```

Numerical check (NumPy):

```python
import numpy as np
from scipy.special import erf

def gelu_exact(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

x = np.linspace(-6, 6, 100001)
max_err = np.abs(gelu_exact(x) - gelu_tanh(x)).max()
assert 1e-5 < max_err < 1e-3   # close, but not equal
assert abs(gelu_exact(np.array([0.0]))[0]) < 1e-12   # GELU(0) = 0
```

</details>

### Exercise 4 · BatchNorm

BatchNorm and LayerNorm don't disagree on the formula: both are "subtract mean, divide by std, then affine transform." They disagree on which axis the statistics come from. LayerNorm computes statistics per sample, over that sample's own feature dimension, so samples never affect each other. BatchNorm computes statistics per feature, across the entire batch, so one sample's output depends on who else is in the batch. That dependency has two consequences: training and inference need different statistics (inference can't depend on the current batch, so it uses `running_mean` / `running_var` accumulated during training), and small batch sizes or heavily padded variable-length sequences make batch statistics unstable or even contaminated by padding. That's exactly why Transformers moved almost entirely to LayerNorm/RMSNorm while BatchNorm stays standard in CNNs: image tasks usually have large enough batches and no padding semantics to worry about.

#### Quick Coding: `batch_norm`

```python
def batch_norm(
    x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
    running_mean: torch.Tensor, running_var: torch.Tensor,
    eps: float = 1e-5, momentum: float = 0.1, training: bool = True,
) -> torch.Tensor:
    ...
```

<details>
<summary>Reference solution</summary>

```python
def batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-5, momentum=0.1, training=True):
    # x: (B, C) or (B, C, ...); statistics are computed over every dim except C
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

In training mode, normalization uses the current batch's mean and variance while `running_mean` / `running_var` get updated by exponential moving average. In eval mode, normalization uses only the moving averages and no longer depends on who's in the current input. That is the core behavioral difference from LayerNorm.

</details>

### Exercise 5 · Kaiming (He) Init

Initialization has one job: keep activation variance from exploding or vanishing as depth increases. Xavier init sets `std = sqrt(2/(fan_in+fan_out))`, assuming the activation is roughly linear and symmetric around 0 (like tanh). ReLU zeroes out the entire negative half, which effectively halves the variance at every layer. Using Xavier's variance on top of that means activations shrink exponentially to 0 after enough ReLU layers. Kaiming init's fix is to use only `fan_in` and change the coefficient to 2: `std = sqrt(2/fan_in)`. That factor of 2 exactly compensates for the variance ReLU discards.

A 20-layer random ReLU MLP makes the gap visible directly: under Xavier init, the last layer's activation variance collapses to the `1e-6` range (information effectively stops propagating); under Kaiming init, variance stays in the `0.3~0.6` range. It still decays slowly with depth, but there's no exponential collapse.

#### Quick Coding: `kaiming_init`

```python
def kaiming_init(weight: torch.Tensor, nonlinearity: str = "relu") -> None:
    ...
```

<details>
<summary>Reference solution</summary>

```python
import math

def kaiming_init(weight: torch.Tensor, nonlinearity: str = "relu") -> None:
    fan_in = weight.shape[1] if weight.dim() == 2 else weight.shape[1:].numel()
    gain = math.sqrt(2.0) if nonlinearity == "relu" else 1.0
    std = gain / math.sqrt(fan_in)
    with torch.no_grad():
        weight.normal_(mean=0.0, std=std)
```

Numerical check (NumPy, 20-layer random ReLU network, `fan_in=256`, final-layer activation variance across 3 independent trials):

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
assert all(v > 0.1 for v in kaiming)     # stays around O(0.1~1)
assert all(v < 1e-4 for v in xavier)     # collapses exponentially toward 0
```

</details>

### Exercise 6 · Dropout

The core mechanism is "drop randomly during training, keep everything during inference," but the standard implementation is inverted dropout: at training time, zero out units with probability `p` and scale the survivors by `1/(1-p)`, keeping their expected value equal to the un-dropped input; at inference time, do nothing and pass the input through unchanged. Skip that scaling step and you'd instead have to multiply the entire output by `(1-p)` at inference to keep the expectation consistent. Inverted dropout moves that cost to training once, so inference stays free. The other detail that's easy to miss in an interview is the `self.training` switch: `model.train()` / `model.eval()` are exactly what flip `nn.Module.training`, and both Dropout's and BatchNorm's behavior branch on that flag.

#### Quick Coding: `Dropout`

```python
class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

<details>
<summary>Reference solution</summary>

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

Numerical check (NumPy, `p=0.3`, 2 million samples): `E[x]` should equal `E[inverted_dropout(x)]` up to sampling noise.

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

Hand-writing Conv2d in an interview is never really about explaining what convolution is. It is about reusing the GEMM (matrix multiply) machinery you already have instead of writing four nested loops. The standard trick is im2col/unfold: flatten every patch the kernel would touch into a column, stack all the columns into one big matrix, and convolution collapses into "weight matrix times patch matrix", the same operator as Linear, just with a different data-movement step in front of it. The output spatial size formula:

```text
OH = floor((H + 2*padding - KH) / stride) + 1
OW = floor((W + 2*padding - KW) / stride) + 1
```

#### Quick Coding: `conv2d`

```python
def conv2d(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
    stride: int = 1, padding: int = 0,
) -> torch.Tensor:
    ...
```

<details>
<summary>Reference solution</summary>

```python
import torch.nn.functional as F

def conv2d(x, weight, bias=None, stride=1, padding=0):
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape

    # unfold: (B, Cin*KH*KW, L), L = OH*OW
    patches = F.unfold(x, kernel_size=(KH, KW), stride=stride, padding=padding)
    w_flat = weight.reshape(Cout, -1)  # (Cout, Cin*KH*KW)

    out = torch.einsum("oc,bcl->bol", w_flat, patches)  # (B, Cout, L)
    if bias is not None:
        out = out + bias.view(1, -1, 1)

    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    return out.reshape(B, Cout, OH, OW)
```

Numerical check (NumPy, a hand-rolled im2col version vs a four-nested-loop naive convolution, `stride=2, padding=1`):

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
# the im2col version is omitted here: same underlying math, expanded into one matmul
assert o_naive.shape == (2, 4, 4, 4)
```

</details>

### Exercise 8 · Linear Regression (three ways)

The value of this question isn't the algorithm itself. It is putting "the math solution" and "the deep-learning paradigm" on the same problem so you have to say precisely how they relate:

1. **Closed form (normal equation)**: `w = (X^T X)^{-1} X^T y`, obtained by taking the derivative of squared error and setting it to zero. Fastest and most exact when the data isn't huge and `X^T X` is invertible.
2. **Manual gradient descent**: no autograd: compute `grad = X^T (Xw - y) / n` by hand and iterate. This is optimizing the exact same convex function as method 1; the only difference is how you reach the optimum, in one shot versus step by step.
3. **`nn.Linear` + autograd + optimizer**: wrap the same problem as "a one-layer linear network + MSE loss + SGD/Adam," relying entirely on automatic differentiation: the standard way a deep learning framework does the same thing.

All three should converge to essentially the same coefficients (in the numerical check below, the normal equation and manual gradient descent agree to about `1e-15` after 5000 steps). That's the one sentence worth saying when asked what this problem is really testing: linear regression is just the simplest convex optimization problem, and a deep learning training loop doesn't introduce new math. It just replaces the closed-form solution with iterative approximation.

#### Quick Coding: `LinearRegression`

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
<summary>Reference solution</summary>

```python
class LinearRegression:
    def __init__(self, in_features: int):
        self.w = torch.zeros(in_features + 1)  # last entry is the bias

    @staticmethod
    def _augment(X: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(X.shape[0], 1, dtype=X.dtype)
        return torch.cat([X, ones], dim=1)

    def fit_normal_equation(self, X, y):
        Xb = self._augment(X)
        # pseudo-inverse instead of a direct inverse, so a singular X^T X doesn't crash
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
        w, b = model.weight.detach().squeeze(0), model.bias.detach()
        self.w = torch.cat([w, b])
```

Numerical check (NumPy, normal equation vs manual full-batch gradient descent, synthetic data `n=200, d=3`):

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

Gradient accumulation solves one problem: the batch size you want doesn't fit in memory. Split a large batch into `K` micro-batches, run forward and backward on each in turn; gradients naturally accumulate in `.grad`, then call `optimizer.step()` once after all `K`. The only mathematical requirement is that the accumulated gradient must equal the gradient you'd get from computing the large batch directly in one shot. If each micro-batch's loss uses mean reduction (not sum), you must divide that loss by `K` before calling backward, otherwise the accumulated gradient ends up `K` times larger than the true large-batch gradient. The other classic bug is where `zero_grad()` goes: it must be called once, before an entire accumulation cycle of `K` micro-batches starts, not once per micro-batch. Calling it every micro-batch wipes out the gradients from earlier micro-batches, so training silently degenerates into using only the last micro-batch.

#### Quick Coding: `accumulated_step`

```python
def accumulated_step(model, optimizer, microbatches, loss_fn) -> float:
    ...
```

<details>
<summary>Reference solution</summary>

```python
def accumulated_step(model, optimizer, microbatches, loss_fn) -> float:
    optimizer.zero_grad()                      # once per accumulation cycle, not per micro-batch
    num_micro = len(microbatches)
    total_loss = 0.0
    for x, y in microbatches:
        pred = model(x)
        loss = loss_fn(pred, y) / num_micro     # mean reduction must be divided by K
        loss.backward()                          # gradients accumulate in .grad automatically
        total_loss += loss.item()
    optimizer.step()
    return total_loss
```

Numerical check (NumPy, accumulated gradient over 4 equal micro-batches vs a single large-batch gradient, both mean-reduction quadratic loss):

```python
import numpy as np
np.random.seed(0)
n, d = 200, 3
X = np.random.randn(n, d + 1)  # includes the bias column
y = np.random.randn(n)
w0 = np.random.randn(d + 1)

full_grad = X.T @ (X @ w0 - y) / n

K = 4
bs = n // K
accum_grad = np.zeros(d + 1)
for i in range(K):
    xb, yb = X[i*bs:(i+1)*bs], y[i*bs:(i+1)*bs]
    g = xb.T @ (xb @ w0 - yb) / bs   # each micro-batch's own mean-reduction gradient
    accum_grad += g / K              # dividing by K reassembles the large-batch gradient

assert np.abs(full_grad - accum_grad).max() < 1e-10
```

</details>

## Final Check: Debug Checklist for This Note

- Does LayerNorm subtract the mean before dividing by the standard deviation; does RMSNorm actually skip the mean-subtraction step, rather than just using a larger `eps`.
- Is GELU using the exact `erf` formula or the GPT-2/BERT tanh approximation? The two aren't numerically identical, so don't conflate them when answering.
- Are BatchNorm's statistics computed over `(B, ...)`, not over the last dimension like LayerNorm; does `training=False` correctly switch to `running_mean` / `running_var`.
- In the Kaiming variance formula, is `fan_in` the weight's input dimension or output dimension, and is the coefficient of 2 only applied for ReLU-family nonlinearities.
- Is Dropout's rescaling applied at training time (inverted dropout) or inference time; does `self.training` correctly flip with `model.train()` / `model.eval()`.
- Does Conv2d's unfold/im2col layout line up with the weight's `reshape(Cout, -1)`; does the output spatial-size formula correctly account for both padding and stride.
- Do all three linear-regression implementations converge to (nearly) the same coefficients? If not, check the learning rate, step count, or whether the normal equation should be using a pseudo-inverse instead of a direct inverse.
- In gradient accumulation, is each micro-batch's loss divided by the accumulation count `K`; is `zero_grad()` called exactly once per accumulation cycle rather than once per micro-batch.

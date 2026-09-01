# ML Coding 04 · Architecture Extensions: LoRA, ViT Patch Embedding, and Mixture of Experts

MLCoding01 already assembled a working Transformer block. This section doesn't rebuild the backbone; it bolts three real-world architecture extensions onto it: how to fine-tune cheaply, how to feed images into the same attention backbone, and how to swap the FFN for a pool of experts so parameter count and per-token compute decouple.

## Module 9: Architecture Extensions

### Exercise 1 · LoRA Linear

Full fine-tuning of a large model is expensive mostly because of optimizer state (Adam's first and second moments sit at the same size as the weights themselves), not the forward pass. LoRA freezes the original weight `W` and adds a low-rank side path instead:

```text
h = x @ W^T + (alpha / r) * x @ (B @ A)^T
```

where `A` has shape `(r, in_features)` and `B` has shape `(out_features, r)`, with `r` much smaller than both `in_features` and `out_features`. Trainable parameters drop from `in_features * out_features` to `r * (in_features + out_features)`.

One initialization detail gets asked about more than the formula itself: `B` must be initialized to all zeros, while `A` gets a normal or Kaiming initialization. The reason is that at the start of training you don't want this new path to disturb an already-pretrained model. If both `A` and `B` were random, `B @ A` would start as a random noise matrix and adding it to `W` would scramble the model's initial behavior. Setting `B` to zero makes `B @ A` identically zero at step 0, so `h` exactly equals the frozen base output at initialization. Fine-tuning starts from "no change at all," and gradients gradually push `A` and `B` toward a useful update direction.

The second detail that comes up is whether this can be zero-overhead at inference time. Since `W' = W + (alpha / r) * B @ A` has exactly the same shape as `W`, you can add `B @ A` back into `W` once training finishes, and inference afterward is a plain single matrix multiply, with no extra compute or memory over the original model. But that merge is a one-time operation: if you need to switch between different task-specific LoRA adapters on the same base model at request time (a serving system handling dozens of tenants, one adapter each), merging means recomputing `W + delta` on every switch, which doesn't pencil out for multi-tenant serving. Such systems typically keep `A` and `B` unmerged and compute the low-rank branch separately at forward time, adding it to the base output.

#### Quick Coding: `LoRALinear`

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, device=None, dtype=None):
        ...

    def forward(self, x):
        ...

    def merge(self):
        ...
```

<details>
<summary>Reference solution</summary>

```python
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, alpha, device=None, dtype=None):
        super().__init__()
        self.r = r
        self.scaling = alpha / r

        # Frozen base weight, excluded from gradient updates
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype),
            requires_grad=False,
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Low-rank path: A gets a normal init, B starts at all zeros
        self.lora_A = nn.Parameter(torch.empty(r, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = torch.einsum("...i,oi->...o", x, self.we\right)
        delta = torch.einsum("...i,ri->...r", x, self.lora_A)
        delta = torch.einsum("...r,or->...o", delta, self.lora_B)
        return base + self.scaling * delta

    @torch.no_grad()
    def merge(self):
        # One-time merge after training; inference degenerates to a plain Linear, zero extra cost
        self.weight += self.scaling * (self.lora_B @ self.lora_A)
        self.lora_A.zero_()
        self.lora_B.zero_()
```

Verified two things with NumPy: with `lora_B` all zeros, `forward` matches plain `x @ W^T` element-wise (the module is an identity map at init), and the output on the same `x` is unchanged before and after `merge()` (error at the `1e-15` scale, pure floating-point rounding).

</details>

### Exercise 2 · ViT Patch Embedding

Vision Transformer wants to reuse the exact NLP Transformer block, so the first step is turning an image `(B, C, H, W)` into a sequence of tokens. It cuts the image into non-overlapping `P × P` patches, flattens each patch into a vector, and projects it linearly to `d_model`:

```text
image (B, C, H, W)
  -> patches (B, num_patches, C*P*P)   # num_patches = (H/P) * (W/P)
  -> linear projection
  -> tokens (B, num_patches, d_model)
```

Interviews often ask for two equivalent implementations, plus why they're equivalent:

1. **Explicit unfold + reshape + linear**: cut the image into patches, flatten each, run through a `Linear(C*P*P, d_model)`.
2. **A single `Conv2d(C, d_model, kernel_size=P, stride=P)`**: kernel size and stride both equal the patch size.

The two are equivalent precisely because `kernel_size == stride == P`: every output position of the convolution corresponds to one completely non-overlapping `P × P` window of the input, and what happens inside that window is "flatten the window, flatten the kernel, take their dot product," exactly what the linear projection does. If `stride < kernel_size` (overlapping windows) or `stride > kernel_size` (skipped pixels), this equivalence breaks. ViT patch embedding can be implemented as a convolution only because it never actually uses the "sliding window with shared receptive field" property of convolution: it's borrowing the conv operator to perform one block-diagonal matrix multiply.

After the patch tokens are assembled, two more things usually happen: a learnable `[CLS]` token gets prepended to the sequence to serve as the global representation for classification, and a learnable (or sinusoidal) position embedding gets added to every position, since flattening the patches already threw away their 2D spatial location in the original image.

#### Quick Coding: `PatchEmbedding`

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model, device=None, dtype=None):
        ...

    def forward(self, img):
        ...
```

<details>
<summary>Reference solution</summary>

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model, device=None, dtype=None):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        # A conv with stride == kernel_size implements "split into patches + linear projection"
        self.proj = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size,
            device=device, dtype=dtype,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model, device=device, dtype=dtype))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, d_model, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, img):
        B = img.shape[0]
        x = self.proj(img)                      # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)         # (B, num_patches, d_model)
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)           # (B, num_patches + 1, d_model)
        return x + self.pos_embed
```

Replacing `proj` with "manual unfold, then a `Linear`" is a fully equivalent rewrite: the matrix multiply is just being handled by the conv operator instead. Both paths were implemented in NumPy and run forward on the same random image and weights; the outputs matched element-wise (error at the `1e-15` scale).

</details>

### Exercise 3 · Mixture of Experts

MoE is trying to decouple parameter count from per-token compute. In a dense FFN, more parameters means more compute per token, linearly. MoE instead replaces one FFN with `num_experts` independent FFNs (each shaped like MLCoding01's `SwiGLU`), but routes each token to only `k` of them (Mixtral uses `k=2`); experts that aren't selected do zero work for that token.

```text
router_logits = Linear(d_model, num_experts)(x)      # (B, T, num_experts)
router_probs  = softmax(router_logits, dim=-1)
top_k_probs, top_k_idx = router_probs.topk(k, dim=-1)
top_k_probs  = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # renormalize

output = sum_{i in top_k_idx} top_k_probs[i] * expert_i(x)
```

Total parameter count is `num_experts` times a single expert's, but per-token compute is only about `k` times. With `num_experts=8, k=2`, that's `2/8 = 25%` of the compute buying `8x` the parameter capacity, which is the core pitch behind "huge parameter count, cheap inference" MoE models.

One implementation detail that gets probed hard is the renormalization step: after selecting the top-k experts, their router probabilities sum to less than 1 (the denominator still includes the unselected experts). Using the raw probabilities as weights would systematically shrink the output scale, by an amount that varies per token, not something a learning rate or initialization can compensate for. The standard fix is renormalizing the selected `k` probabilities to sum to 1 before the weighted sum. Verified with NumPy that the pre-renormalization and post-renormalization weighted sums over the same expert outputs are not equal, confirming renormalization genuinely changes the output scale.

The other common follow-up is: what if a handful of experts keep getting selected and the rest barely receive gradient, a failure mode known as expert collapse. The standard fix is a load-balancing auxiliary loss that encourages the router to spread tokens roughly evenly across experts (a common form minimizes the dot product between "how often each expert gets selected" and "each expert's average router probability"). This auxiliary loss is added to the main loss with a small weight, acting only as a mild corrective nudge.

#### Quick Coding: `MixtureOfExperts`

```python
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, device=None, dtype=None):
        ...

    def forward(self, x):
        ...
```

<details>
<summary>Reference solution</summary>

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k, device=None, dtype=None):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False, device=device, dtype=dtype)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff, device=device, dtype=dtype) for _ in range(num_experts)]
        )

    def forward(self, x):
        B, T, D = x.shape
        flat_x = x.reshape(-1, D)                              # (N, D), N = B*T

        router_logits = self.router(flat_x)                    # (N, num_experts)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_probs, top_k_idx = router_probs.topk(self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # renormalize

        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            # find every (token, slot) position where this expert was selected
            token_idx, slot_idx = (top_k_idx == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            weight = top_k_probs[token_idx, slot_idx].unsqueeze(-1)
            out[token_idx] += weight * expert(flat_x[token_idx])

        return out.reshape(B, T, D)
```

Reproduced the router's numerical logic in NumPy: softmax over random logits, take the top 2, renormalize so their sum is 1 (versus about 0.68 before renormalizing), and confirmed the pre- and post-renormalization weighted sums actually differ.

</details>

#### Common mistakes in this module

- LoRA's `B` must be zero-initialized, since `A` is the side that carries the "exploration direction." Swap the two and training starts by perturbing the frozen base we\right.
- The patch-embedding convolution equivalence only holds when `stride == kernel_size` (no overlap, no skipped pixels). Don't conflate it with ordinary convolution's shared-receptive-field behavior.
- MoE's top-k probabilities must be renormalized, or the output scale drifts systematically with whichever experts happen to get routed to.

# ML Coding · Experiments & Ablations

Corresponds to CS336 Assignment 1: Section 7.

**Usage:** This page consists of experimental tasks rather than traditional coding problems. First, use the "Experiment Template" to fill in configurations, metrics, and output tables; finally, expand the reference solutions to compare your experimental design and conclusion writing.

## Experiment 1 · Experiment Logger

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `experiment_log`

Logging fields:

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

Output:

```text
CSV / JSONL / wandb run
learning curve plot
experiment log markdown
```

Experiment template:

```python
class ExperimentLogger:
    def __init__(self, path: str, config: dict):
        self.path = path
        self.config = config
        self.run_id = ...
        self.git_commit = ...

    def log(self, step: int, **metrics):
        row = {
            "step": step,
            "run_id": self.run_id,
            "config": self.config,
            "git_commit": self.git_commit,
            **metrics,
        }
        ...

def make_experiment_report(log_path: str) -> dict:
    return {
        "best_val_loss": ...,
        "final_train_loss": ...,
        "tokens_processed": ...,
        "wall_clock": ...,
    }
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The goal of an experiment logger is reproducibility: when looking at a loss curve later, you should be able to identify which code, configuration, and data it originated from.

A simple JSONL logger:

```python
import json, time, subprocess

def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

class JSONLLogger:
    def __init__(self, path, config):
        self.f = open(path, "a", encoding="utf-8")
        self.config = config
        self.commit = git_commit()

    def log(self, step, **metrics):
        row = {
            "time": time.time(),
            "step": step,
            "git_commit": self.commit,
            "config": self.config,
            **metrics,
        }
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()
```

Minimum requirements:

- Each run has a unique name.
- Configuration is saved in full, not just labeled as "small model".
- Train/val loss, tokens processed, and wall time are all recorded.
- Checkpoint paths and generated samples can be traced back to the same run.

</details>

## Experiment 2 · TinyStories LR Sweep

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `learning_rate`

Experimental design:

```text
fixed model config
fixed total tokens processed
vary learning rate logarithmically
include at least one divergent run
```

Output:

```text
learning curves
best validation loss
divergence threshold
edge-of-stability discussion
```

Experiment template:

```python
def run_lr_sweep(base_config, learning_rates: list[float]):
    """
    Output:
        one row per LR with final loss, best val loss, divergence flag
    """
    rows = []
    for lr in learning_rates:
        config = {**base_config, "learning_rate": lr}
        result = ...  # train/eval run
        rows.append({
            "lr": lr,
            "final_train_loss": ...,
            "best_val_loss": ...,
            "diverged": ...,
            "notes": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The correct way to perform an LR sweep is to vary only the learning rate while keeping other conditions fixed:

```text
same model
same tokenizer
same data order or same seed
same total tokens
same batch size
same warmup/cosine schedule shape
```

Log-scale search is recommended:

```text
1e-4, 3e-4, 1e-3, 3e-3, 1e-2
```

In your report, don't just state "a certain LR was best"; explain the curves:

| Curve Shape | Explanation |
|---|---|
| Loss barely decreases | LR too small or training time too short |
| Loss decreases then explodes | LR near/exceeding the stability boundary |
| Train loss decreases, val loss does not | Potential overfitting or dataset too small |
| Best LR changes with batch size | Gradient noise scale has changed |

The final output should include a learning curve plot and a table:

```text
lr | final train loss | best val loss | diverged? | tokens/sec
```

</details>

## Experiment 3 · Batch Size Sweep

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `batch_size_experiment`

Experimental matrix:

```text
batch_size = 1
batch_size = 64
batch_size = 128
batch_size near memory limit
```

Records:

```text
tokens/sec
step time
validation loss
GPU memory
best LR per batch size
```

Experiment template:

```python
def run_batch_size_sweep(base_config, batch_sizes: list[int]):
    rows = []
    for bs in batch_sizes:
        config = {**base_config, "batch_size": bs}
        result = ...
        rows.append({
            "batch_size": bs,
            "effective_batch_size": ...,
            "tokens_per_sec": ...,
            "step_time_ms": ...,
            "max_memory_gb": ...,
            "best_val_loss": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

In a batch size sweep, distinguish between two variables: micro-batch size affects VRAM and step time, while effective batch size may also be determined by gradient accumulation.

Experiment table:

```text
batch_size | grad_accum | effective_batch | lr | tokens/sec | max_mem | best_val_loss
```

Analysis approach:

- Batch too small: Poor GPU utilization, low tokens/sec, but high gradient noise, which may improve generalization.
- Batch size increases: Throughput usually rises, but the LR needs to be tuned.
- Near memory limit: Potential OOM, or activation memory causing jitter in step time.
- If total tokens are fixed, a larger batch means fewer optimizer steps.

Conclusions should not be based solely on step time. Look at validation loss under the same wall-clock time or the same number of tokens processed.

</details>

## Experiment 4 · Generate TinyStories Samples

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `generate`

Variables:

```text
temperature
top_p
prompt
checkpoint step
```

Output:

```text
sample text, at least 256 tokens or until EOS
sampling config
fluency comment
two factors affecting quality
```

Experiment template:

```python
def sample_checkpoint(model, tokenizer, prompts, temperatures, top_ps):
    rows = []
    for prompt in prompts:
        for temperature in temperatures:
            for top_p in top_ps:
                text = generate(...)
                rows.append({
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "sample": text,
                    "comment": ...,
                })
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

For generation experiments, fix the prompt and vary the sampling parameters to observe output quality and diversity.

Recommended records:

```text
checkpoint_step
prompt
temperature
top_p
max_new_tokens
generated_text
stop_reason
```

Explanation framework:

| Phenomenon | Potential Cause |
|---|---|
| Repeated short sentences | Model undertrained, temperature too low, simple data patterns |
| Grammatical errors | Checkpoint too early, temperature too high |
| Fluent but monotonous | top_p too small or prompt too strong |
| Early EOS | Tokenizer/EOS learned too strongly, or training data contains many short texts |

For TinyStories samples, focus on story coherence, character consistency, and sentence completeness; do not use factual accuracy as a primary metric.

</details>

## Experiment 5 · Remove RMSNorm

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `layer_norm_ablation`

Experiment:

```text
baseline pre-norm
no RMSNorm at previous best LR
no RMSNorm at lower LR
```

Output:

```text
learning curves
best stable LR
activation / gradient norm observations
normalization commentary
```

Experiment template:

```python
def run_remove_rmsnorm_ablation(base_config):
    variants = [
        {"name": "baseline_prenorm", "use_rmsnorm": True, "lr": base_config["lr"]},
        {"name": "no_rmsnorm_same_lr", "use_rmsnorm": False, "lr": base_config["lr"]},
        {"name": "no_rmsnorm_lower_lr", "use_rmsnorm": False, "lr": base_config["lr"] / 3},
    ]
    rows = []
    for variant in variants:
        result = ...
        rows.append({
            **variant,
            "best_val_loss": ...,
            "diverged": ...,
            "grad_norm_p95": ...,
            "activation_norm_p95": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

Removing RMSNorm is a stability ablation. The experiment must be aligned with the baseline:

```text
baseline: pre-norm transformer + best LR
no_norm_a: remove RMSNorm + same LR
no_norm_b: remove RMSNorm + lower LR
```

Records needed:

```text
train loss
val loss
grad norm
activation norm
divergence step if any
```

Expected explanation: RMSNorm controls the scale of the residual stream, stabilizing the input distribution for attention/MLP. Without it, loss spikes or increased grad norms are more likely at the same LR; lowering the LR might allow training, but convergence speed and final loss may degrade.

Write conclusions as experimental observations rather than absolute rules; sometimes no-norm models can run for a while on small models/datasets.

</details>

## Experiment 6 · Post-Norm Transformer

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `pre_norm_ablation`

Comparison:

```text
pre-norm:
  z = x + MHA(RMSNorm(x))
  y = z + FFN(RMSNorm(z))

post-norm:
  z = RMSNorm(x + MHA(x))
  y = RMSNorm(z + FFN(z))
```

Output:

```text
learning curve comparison
stability notes
best LR if retuned
```

Experiment template:

```python
def run_norm_position_ablation(base_config):
    variants = ["pre_norm", "post_norm", "post_norm_retuned_lr"]
    rows = []
    for variant in variants:
        config = {**base_config, "block_type": variant}
        result = ...
        rows.append({
            "variant": variant,
            "lr": ...,
            "best_val_loss": ...,
            "stability": ...,
            "notes": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The difference between pre-norm and post-norm is whether the norm is placed before or after the residual branch.

Implementing post-norm:

```python
class PostNormBlock(nn.Module):
    def forward(self, x, token_positions=None):
        x = self.ln1(x + self.attn(x, token_positions=token_positions))
        x = self.ln2(x + self.ffn(x))
        return x
```

Keep parameter counts similar when comparing, changing only the block structure. Observations:

- Post-norm may be more sensitive to LR.
- In deep networks, pre-norm has a more direct gradient path and is generally easier to train.
- If post-norm performs poorly, try a lower LR rather than drawing conclusions from a single data point.

It is recommended to plot two curves in the report: one comparing the same LR, and one comparing the retuned best LR.

</details>

## Experiment 7 · NoPE vs RoPE

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `no_pos_emb`

Experiment:

```text
baseline RoPE
NoPE model
same training budget
same tokenizer / data
```

Output:

```text
validation loss curve
sample generation
position information discussion
```

Experiment template:

```python
def run_position_ablation(base_config):
    variants = [
        {"name": "rope", "use_rope": True},
        {"name": "nope", "use_rope": False},
    ]
    rows = []
    for variant in variants:
        result = ...
        rows.append({
            "variant": variant["name"],
            "best_val_loss": ...,
            "sample_text": ...,
            "long_context_behavior": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The NoPE vs RoPE experiment answers: can a model learn sequential information without explicit positional encoding?

Experimental setup:

```text
baseline: RoPE
ablation: remove RoPE from Q/K
same parameter count
same training tokens
same optimizer config
```

Analysis angles:

- The causal mask itself provides directionality, but not precise relative position representation.
- With short context and small data, NoPE might still show loss reduction.
- With long context or tasks requiring positional relationships, RoPE is usually more stable.
- During generation, you might see repetitions, disordered sequences, or degraded long-range consistency.

Do not look only at final loss; include generated samples and context length sensitivity.

</details>

## Experiment 8 · SwiGLU vs SiLU FFN

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `swiglu_ablation`

Constraints:

```text
SwiGLU: three matrices, d_ff around 8/3 d_model
SiLU baseline: two matrices, d_ff around 4 d_model
parameter counts approximately matched
```

Output:

```text
learning curves
parameter count comparison
validation loss comparison
gating discussion
```

Experiment template:

```python
def run_ffn_ablation(base_config):
    variants = [
        {"name": "swiglu", "ffn_type": "swiglu", "d_ff": ...},
        {"name": "silu", "ffn_type": "silu", "d_ff": ...},
    ]
    rows = []
    for variant in variants:
        params = ...
        result = ...
        rows.append({
            "variant": variant["name"],
            "params": params,
            "tokens_per_sec": ...,
            "best_val_loss": ...,
            "notes": ...,
        })
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

When comparing SwiGLU and SiLU FFN, you must match parameter counts; otherwise, the conclusion is unfair.

Approximate parameter counts:

```text
SwiGLU: 3 * d_model * d_ff_swiglu
SiLU FFN: 2 * d_model * d_ff_silu
```

Common settings:

```text
d_ff_silu ≈ 4 * d_model
d_ff_swiglu ≈ 8/3 * d_model
```

Experiment records:

```text
variant | params | tokens/sec | best_val_loss | final_train_loss
```

Explanation: The SwiGLU gate can dynamically modulate channels, providing higher expressive power; however, the extra projection path makes kernel fusion and memory access more complex. Differences in small models may be unstable, so report multiple seeds or at least provide run variance.

</details>

## Experiment 9 · OpenWebText Run

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `main_experiment`

Output:

```text
OpenWebText learning curve
generated sample
TinyStories vs OWT loss comparison
fluency analysis
```

Analysis focus:

OpenWebText is more diverse, has a longer tail, and is harder to compress; under the same model and compute budget, loss and generation quality cannot be directly compared to TinyStories.

Experiment template:

```python
def run_openwebtext_experiment(config):
    result = ...  # train on OWT
    sample = ...  # generate text from selected prompts
    return {
        "dataset": "OpenWebText",
        "tokenizer": ...,
        "training_tokens": ...,
        "best_val_loss": ...,
        "tokens_per_sec": ...,
        "generated_sample": sample,
        "tinystories_comparison": ...,
    }
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The focus of the OpenWebText run is domain shift. Compared to TinyStories, OWT is more diverse, containing web pages, code snippets, forum language, long-tail entities, and noise.

Report structure:

```text
dataset/tokenizer
model config
training tokens
best validation loss
tokens/sec
generated sample
failure examples
```

When analyzing, avoid simply stating "OWT loss is higher, so the model is worse." Different data distributions have different entropies, making loss incomparable across datasets. More reasonable questions are:

- Does the same model show stable loss reduction on OWT?
- Does generation exhibit web-like fragments, repetitions, or garbled text?
- Does the tokenizer's bytes/token metric degrade?
- Under the same compute, is the language capability learned on OWT more general?

</details>

## Experiment 10 · Leaderboard-Style Modification

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `leaderboard`

Optional directions:

```text
weight tying
better initialization
LR / batch size schedule
architecture small changes
tokenizer changes within allowed data
throughput optimization
```

Output:

```text
final validation loss
wall-clock-bounded learning curve
description of modification
evidence the modification helped
```

Experiment template:

```python
def run_leaderboard_modification(base_config, modification_name: str):
    baseline = ...
    modified_config = apply_modification(base_config, modification_name)
    modified = ...
    return {
        "modification": modification_name,
        "baseline_val_loss": ...,
        "modified_val_loss": ...,
        "delta": ...,
        "tokens_per_sec_delta": ...,
        "evidence": ...,
        "failure_modes": ...,
    }
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

A leaderboard-style modification should be like a small research experiment: state a hypothesis, change one major variable, and provide evidence.

Possible directions:

| Modification | Hypothesis | Risk |
|---|---|---|
| Weight tying | Sharing embedding and LM head reduces parameters and may regularize | May limit output layer expressivity |
| Retune LR schedule | Original schedule is not optimal | Requires more experimental budget |
| Better init | Improves early stability | May only affect the first few hundred steps |
| Fused kernels / compile | Increases tokens/sec | Does not necessarily improve final loss |
| Tokenizer change | Compresses data better | Changes training distribution and loss comparability |

Example using weight tying:

```python
model.lm_head.weight = model.token_embeddings.weight
```

The report must include at least:

```text
baseline run
modified run
same token budget
same eval protocol
validation loss delta
throughput delta
one paragraph explaining why it helped or failed
```

If the modification does not yield improvements, it can still be a good answer; the key is clean experimental design and conclusions consistent with the data.

</details>

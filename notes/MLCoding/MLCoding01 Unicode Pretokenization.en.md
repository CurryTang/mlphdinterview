# ML Coding · From-Scratch LLM Implementation

This chapter merges the original eight ML Coding notes into one build path: first turn raw text into tokens, then turn tokens into a Transformer, then connect training, sampling, and experiment loops. Each exercise corresponds to a real subsystem, so the best reading order is the assembly order of a working LLM rather than a bag of isolated templates.

The full pipeline is:

```text
raw text
-> Unicode / UTF-8
-> pre-tokenization
-> BPE training
-> tokenizer runtime
-> token id dataset
-> tensor modules
-> attention + Transformer LM
-> loss / optimizer / scheduler
-> training loop / checkpoint / generation
-> experiments / ablations / iteration
```

## Learning Order

| Order | Module | What you build | Core question |
| --- | --- | --- | --- |
| 1 | Unicode, UTF-8, and pretokenization | the input boundary of a byte-level tokenizer | what is the difference between Python `str`, code points, and UTF-8 bytes |
| 2 | BPE training | `vocab` and `merges` | which adjacent pairs should be merged, and how do you preserve hard boundaries |
| 3 | Tokenizer runtime and dataset export | `encode` / `decode` / token arrays | how do trained merges become a stable runtime interface |
| 4 | Tensor modules | Embedding, RMSNorm, SwiGLU, RoPE | how do Transformer building blocks fit together by shape |
| 5 | Attention and Transformer LM | a trainable language-model forward pass | how do causal attention, residual streams, and logits connect |
| 6 | Training components | loss, AdamW, LR schedule, grad clip | why is one training recipe stable and another unstable |
| 7 | Training loop and generation | dataloader, checkpointing, decoding | what is the minimum loop required to train and sample |
| 8 | Experiments and ablations | a reproducible experimentation workflow | which design choices actually help, and which only sound plausible |

## Module 1: Unicode, UTF-8, and Pretokenization

Corresponds to CS336 Assignment 1: Sections 2.1-2.4.

The first fact of tokenization is that the model never sees characters directly. It sees integers, and a byte-level tokenizer reaches those integers through bytes. `str`, code points, and UTF-8 bytes are different abstraction layers. If you blur them together, special-token boundaries, BPE merges, and decode behavior all become unreliable.

### Lab · Unicode Probe

A Unicode code point is an abstract character identifier. UTF-8 is the variable-length byte representation of that identifier. `ord` and `chr` operate at the code-point layer; `encode("utf-8")` enters the byte layer. `repr` and `print` also serve different purposes, because control characters may exist without producing visible output.

The most useful first contrast is between an ASCII code point such as `U+0041`, which occupies one byte, and an emoji such as `U+1F600`, which occupies four bytes. That single comparison already explains why character count and byte count diverge.

#### Quick Coding: `inspect_unicode_codepoint`

```python
def inspect_unicode_codepoint(cp: int) -> dict:
    ...
```

<details>
<summary>Reference answer</summary>

```python
def inspect_unicode_codepoint(cp: int) -> dict:
    ch = chr(cp)
    return {
        "codepoint": f"U+{cp:04X}",
        "character": ch,
        "repr": repr(ch),
        "utf8_bytes": list(ch.encode("utf-8")),
        "utf8_hex": ch.encode("utf-8").hex(" "),
        "is_printable": ch.isprintable(),
    }
```

```python
assert ord(chr(65)) == 65
assert inspect_unicode_codepoint(0x41)["utf8_bytes"] == [65]
assert inspect_unicode_codepoint(0x1F600)["utf8_bytes"] == [240, 159, 152, 128]
```

`print(ch)` is about display behavior. `repr(ch)` is about debugging. Zero-width characters, newlines, and control characters usually only become obvious in the latter.

</details>

### Lab · UTF-8 Encoding

The difference between UTF-8, UTF-16, and UTF-32 is not “which one is more modern.” The real question is which encoding is cheaper, safer, and easier to connect to a byte stream for the data you actually have. LLM tokenizers almost always operate on UTF-8 because the training corpus already lives as bytes, and UTF-8 avoids the fixed-width overhead that UTF-16 or UTF-32 impose on short English text.

`"hello"` has the same character count and UTF-8 byte count. `"こんにちは"` and emoji do not. One step further, invalid byte sequences force you to choose `strict`, `replace`, or `ignore`, which is exactly why later decode behavior must be explicit.

#### Quick Coding: `compare_encodings`

```python
def compare_encodings(text: str) -> list[dict]:
    ...

def decode_invalid(raw: bytes, encoding="utf-8") -> dict:
    ...
```

<details>
<summary>Reference answer</summary>

```python
def compare_encodings(text: str):
    rows = []
    for enc in ["utf-8", "utf-16", "utf-32"]:
        raw = text.encode(enc)
        rows.append({
            "encoding": enc,
            "num_chars": len(text),
            "num_bytes": len(raw),
            "bytes": raw,
            "roundtrip": raw.decode(enc),
        })
    return rows

def decode_invalid(raw: bytes, encoding="utf-8"):
    return {
        "strict": _try_decode(raw, encoding, "strict"),
        "replace": raw.decode(encoding, errors="replace"),
        "ignore": raw.decode(encoding, errors="ignore"),
    }

def _try_decode(raw, encoding, errors):
    try:
        return raw.decode(encoding, errors=errors)
    except UnicodeDecodeError as exc:
        return type(exc).__name__
```

The conclusions should be explicit:

- ASCII in UTF-8 is usually 1 character = 1 byte.
- CJK and emoji occupy multiple bytes in UTF-8.
- UTF-16 and UTF-32 often pay BOM or fixed-width overhead.
- Invalid byte sequences should never be silently swallowed.

</details>

### Exercise 1 · GPT-2 Style Pretokenizer

Pretokenization does more than “split first, then apply BPE.” Its real job is to define the scope in which BPE is allowed to learn frequent byte patterns. Without this boundary layer, merges can cross spaces, punctuation, special tokens, or document separators, and the longest learned tokens will often expose the bug immediately.

Four constraints matter most:

| Constraint | Why it matters |
| --- | --- |
| special tokens are hard boundaries | control tokens such as `<|endoftext|>` must remain intact |
| special tokens do not enter statistics | otherwise BPE will learn the internals of control tokens |
| pair counts never cross a pre-token boundary | BPE is local compression, not global free-form merging |
| use iterator-style processing when possible | full materialization wastes memory on large corpora |

For example, `Doc1<|endoftext|>Doc2` must never contribute cross-document pair statistics. Likewise, `"some text that i'll pre-tokenize"` is useful because the GPT-2 regex preserves leading spaces and follows the assignment’s contraction rules, which determines what local statistics BPE is even allowed to see.

#### Quick Coding: `pretoken_counts`

```python
def split_by_special(text: str, special_tokens: list[str]):
    ...

def pretoken_counts(text: str, special_tokens: list[str] | None = None) -> dict:
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import Counter
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_by_special(text, special_tokens):
    if not special_tokens:
        yield False, text
        return

    pattern = "(" + "|".join(re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)) + ")"
    for part in re.split(pattern, text):
        if not part:
            continue
        yield part in special_tokens, part

def pretoken_counts(text, special_tokens=None):
    counts = Counter()
    for is_special, part in split_by_special(text, special_tokens or []):
        if is_special:
            continue
        for match in re.finditer(PAT, part):
            token_bytes = match.group(0).encode("utf-8")
            counts[tuple(bytes([b]) for b in token_bytes)] += 1
    return counts
```

Representing each pre-token as `tuple[bytes, ...]` is what makes later BPE merges easy: adjacent `bytes` objects can be concatenated into longer `bytes` objects directly.

</details>

#### Common mistakes in this module

- The base unit of a byte-level tokenizer is `bytes`, not Python `str`.
- A single byte must still be stored as a `bytes` object, such as `b"a"`.
- Special tokens are training boundaries and whole tokens at runtime.
- `len(text)` and `len(text.encode("utf-8"))` are different quantities.

## Module 2: BPE Training

Corresponds to CS336 Assignment 1: Sections 2.4-2.5.

BPE training does one concrete thing: repeatedly find the most valuable adjacent pair and merge it into a longer token. It does not understand semantics. It only compresses frequent local byte patterns inside the legal boundary defined by pretokenization.

Two details are easy to miss but absolutely central: weighted pair frequency and deterministic tie-breaking. If a pre-token appears 1000 times, every pair inside it contributes 1000 times. If two pairs tie, the assignment requires the lexicographically greater pair to win.

### Exercise 1 · Toy BPE Merge Simulator

Running the merge loop on a tiny corpus is the fastest way to separate “count adjacent pairs” from “apply one non-overlapping replacement.” For example:

| Round | Pretoken counts | Main intuition |
| --- | --- | --- |
| 0 | `(l, o, w) x2`, `(l, o, w, e, r) x1` | `(o, w)` and `(l, o)` are both frequent, so tie-breaking matters |
| 1 | merge the winner non-overlappingly | only adjacent, non-overlapping occurrences are replaced |
| 2 | recount pairs on the new representation | old pair statistics change because token boundaries changed |

The worked example matters more than the final winner. BPE does not rewrite all seen pairs at once. It chooses one winner, rewrites the corpus representation, then recomputes what “frequent” means in the new representation.

#### Quick Coding: `run_bpe_merges`

```python
from collections import Counter

def count_pairs(pretoken_counts: dict[tuple[bytes, ...], int]) -> Counter:
    ...

def merge_one_pretoken(pieces: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    ...

def run_bpe_merges(pretoken_counts, num_merges):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import Counter

def count_pairs(pretoken_counts):
    pair_counts = Counter()
    for pieces, freq in pretoken_counts.items():
        for a, b in zip(pieces, pieces[1:]):
            pair_counts[(a, b)] += freq
    return pair_counts

def merge_one_pretoken(pieces, pair):
    out = []
    i = 0
    while i < len(pieces):
        if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
            out.append(pieces[i] + pieces[i + 1])
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return tuple(out)

def run_bpe_merges(pretoken_counts, num_merges):
    counts = dict(pretoken_counts)
    merges = []
    for _ in range(num_merges):
        pair_counts = count_pairs(counts)
        if not pair_counts:
            break
        winner = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(winner)
        next_counts = Counter()
        for pieces, freq in counts.items():
            next_counts[merge_one_pretoken(pieces, winner)] += freq
        counts = dict(next_counts)
    return merges, counts
```

Do not use `Counter.most_common(1)` for winner selection. Its tie-breaking depends on insertion order and therefore does not satisfy the assignment’s deterministic rule.

</details>

### Exercise 2 · Full BPE Trainer

The full trainer takes the local logic from the previous problem and places it in an end-to-end workflow: start from byte vocabulary `0..255`, append special tokens, repeatedly select a winner pair, and stop only when the total vocabulary size reaches `vocab_size`.

The phrase “total vocabulary size” matters. `vocab_size` is not “number of merges.” It must include the 256 byte tokens, the special tokens, and every merge-produced token.

#### Quick Coding: `train_bpe`

```python
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    ...
```

<details>
<summary>Reference answer</summary>

```python
from collections import Counter

def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for tok in special_tokens:
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1

    pretoken_counts = build_pretoken_counts(input_path, special_tokens)
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(pretoken_counts)
        if not pair_counts:
            break

        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged = pair[0] + pair[1]
        merges.append(pair)
        vocab[len(vocab)] = merged

        updated = Counter()
        for pieces, freq in pretoken_counts.items():
            updated[merge_one_pretoken(pieces, pair)] += freq
        pretoken_counts = updated

    return vocab, merges
```

`build_pretoken_counts` must reuse the exact boundary semantics from Module 1. Quietly changing pretokenization rules here means the trained merges no longer match the earlier assumptions.

</details>

### Exercise 3 · BPE Performance Pass

The naive trainer has an obvious bottleneck: every round rescans the full corpus and recounts all pairs. TinyStories is already large enough to make that approach painful. The optimization path is therefore straightforward: parallel pretokenization, cached inverted indices, and pair updates only for pre-tokens affected by the most recent merge.

The main question is not “what data structure is clever.” It is “what state can be updated incrementally, and what state must truly be recomputed.” In practice, the most useful structure is an inverted index from `pair -> set[pretoken]`.

#### Quick Coding: `build_pair_index`

```python
def build_pair_index(pretoken_counts):
    ...

def update_after_merge(pretoken_counts, pair_counts, pair_to_pretokens, winner):
    ...
```

<details>
<summary>Reference answer</summary>

The optimization path should be explicit:

```text
1. Split the raw corpus into document chunks, with special tokens used only as boundaries.
2. Parallelize pretokenization to obtain Counter[tuple[bytes, ...]].
3. Build an inverted index pair -> set[pretoken].
4. Pick the winner pair each round.
5. Update only the pre-tokens that contain the winner.
6. Decrement their old pair contributions and increment their new ones.
```

```python
pair_counts = build_pair_counts(pretoken_counts)
pair_to_pretokens = build_inverted_index(pretoken_counts)

for _ in range(num_merges):
    pair = argmax_pair(pair_counts)
    affected = list(pair_to_pretokens[pair])

    for old_pieces in affected:
        freq = pretoken_counts.pop(old_pieces, 0)
        if freq == 0:
            continue
        decrement_pairs(old_pieces, freq, pair_counts, pair_to_pretokens)
        new_pieces = merge_one_pretoken(old_pieces, pair)
        pretoken_counts[new_pieces] += freq
        increment_pairs(new_pieces, freq, pair_counts, pair_to_pretokens)
```

Validation should report at least wall-clock time, peak memory, and per-stage speedup, not just a single total runtime.

</details>

### Experiment · Train TinyStories Tokenizer

This experiment is not just “can a 10K tokenizer be trained.” The real question is whether the learned long tokens match the domain. TinyStories has a narrow distribution, so a reasonable vocabulary should merge frequent English words, space-prefixed words, common suffixes, and story-specific names into longer tokens.

Use the fixed setup:

| Item | Value |
| --- | --- |
| dataset | TinyStories |
| vocab_size | 10,000 |
| special token | `<|endoftext|>` |
| artifacts | `vocab.json`, `merges.txt`, profile report |

The most important checks are not a single scalar:

- `<|endoftext|>` must exist as its own token.
- `num_merges = 10000 - 256 - 1 = 9743` must be accounted for correctly.
- The longest tokens should look like domain words and names, not gibberish.
- If the longest token appears to cross document boundaries, the special-token handling is probably wrong.

### Experiment · Train OpenWebText Tokenizer

Moving the same algorithm to OpenWebText turns the problem into domain shift. OWT has more vocabulary diversity, a heavier tail, and more URLs, HTML, code fragments, symbol strings, and multilingual content. A 32K tokenizer trained there should therefore look very different from a TinyStories tokenizer.

A good comparison matrix is:

| data | tokenizer | main metric |
| --- | --- | --- |
| TinyStories sample | TinyStories 10K | bytes/token |
| TinyStories sample | OWT 32K | bytes/token |
| OWT sample | TinyStories 10K | bytes/token |
| OWT sample | OWT 32K | bytes/token |

A solid report should explain:

- why the OWT tokenizer compresses OWT better,
- why the TinyStories tokenizer looks cleaner but transfers poorly to OWT,
- how longest tokens, most frequent tokens, and encode throughput reflect domain differences.

#### Common mistakes in this module

- Special tokens get split and participate in merges.
- Merges cross pre-token boundaries.
- Pair tie-breaking is not deterministic.
- Individual bytes are stored as `int` instead of `bytes`.
- Every round rescans the full corpus from scratch.

## Module 3: Tokenizer Runtime and Dataset Export

Corresponds to CS336 Assignment 1: Sections 2.6-2.7.

Training `vocab` and `merges` only solves half of the problem. A usable tokenizer runtime must carry the training-time merge order and boundary rules into `encode`, `decode`, streaming interfaces, and dataset export without changing semantics.

### Exercise 1 · Tokenizer Class

The runtime needs to fix three behaviors:

| Behavior | Correct rule |
| --- | --- |
| how special tokens are handled during encode | protect boundaries first, then pretokenize ordinary text |
| how merges are applied | in training creation order, i.e. lowest merge rank first |
| how decode reconstructs text | concatenate token bytes first, then do one UTF-8 decode |

The last rule matters more than it first appears. Token-by-token decoding can corrupt multi-byte characters that span token boundaries. The correct approach is always `b"".join(...)` followed by a single `errors="replace"` decode.

#### Quick Coding: `Tokenizer`

```python
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        ...

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        ...

    def encode(self, text: str) -> list[int]:
        ...

    def encode_iterable(self, iterable):
        ...

    def decode(self, ids: list[int]) -> str:
        ...
```

<details>
<summary>Reference answer</summary>

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)

        next_id = max(self.vocab) + 1 if self.vocab else 0
        for tok in self.special_tokens:
            b = tok.encode("utf-8")
            if b not in self.inverse_vocab:
                self.vocab[next_id] = b
                self.inverse_vocab[b] = next_id
                next_id += 1

        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import json
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(i): bytes(v) for i, v in raw_vocab.items()}

        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                a, b = line.rstrip("\n").split(" ")
                merges.append((a.encode("latin1"), b.encode("latin1")))
        return cls(vocab, merges, special_tokens)

    def apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        pieces = [bytes([b]) for b in token_bytes]
        while len(pieces) >= 2:
            pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
            best = min(pairs, key=lambda p: self.merge_rank.get(p, float("inf")))
            if best not in self.merge_rank:
                break

            out = []
            i = 0
            while i < len(pieces):
                if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == best:
                    out.append(pieces[i] + pieces[i + 1])
                    i += 2
                else:
                    out.append(pieces[i])
                    i += 1
            pieces = out
        return pieces

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")
```

At minimum, verify:

- `decode(encode(text)) == text` for valid UTF-8 text,
- special tokens are not fragmented,
- merges stay inside one pre-token,
- malformed bytes decode via the replacement character rather than raising.

</details>

### Trace Lab · BPE Encoding Trace

The most effective way to debug BPE is not to inspect the final token IDs. It is to print the pieces after each merge round. For `"the cat ate"`, you must first see `["the", " cat", " ate"]` as three pre-tokens. If some merge rule combines `"e"` with the following space, the boundary logic is already broken.

The point of a trace helper is to expose intermediate state. A one-rank merge-order bug may be invisible in the final IDs but obvious in a `start -> step 1 -> step 2` trace.

#### Quick Coding: `trace_bpe_token`

```python
def trace_bpe_token(token: str, tokenizer: Tokenizer) -> list[int]:
    ...
```

<details>
<summary>Reference answer</summary>

```python
def trace_bpe_token(token: str, tokenizer):
    pieces = [bytes([b]) for b in token.encode("utf-8")]
    print("start:", pieces)

    step = 0
    while len(pieces) >= 2:
        pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
        ranked = [
            (tokenizer.merge_rank[p], p)
            for p in pairs
            if p in tokenizer.merge_rank
        ]
        if not ranked:
            break

        _, pair = min(ranked)
        new_pieces = []
        i = 0
        while i < len(pieces):
            if i + 1 < len(pieces) and (pieces[i], pieces[i + 1]) == pair:
                new_pieces.append(pieces[i] + pieces[i + 1])
                i += 2
            else:
                new_pieces.append(pieces[i])
                i += 1

        step += 1
        print(f"step {step}: merge {pair} -> {new_pieces}")
        pieces = new_pieces

    ids = [tokenizer.inverse_vocab[p] for p in pieces]
    print("ids:", ids)
    return ids
```

The most common bug is to concatenate the full string into one byte sequence and let merges cross spaces or special-token boundaries.

</details>

### Exercise 2 · Streaming Encode

The difficulty in streaming encode is not `yield`. It is safe boundaries. If chunk boundaries alter pretokenization or merge scope, the streamed result will differ from encoding the full text at once. Arbitrary splits such as `"intern" + "ational"` are the standard counterexample.

So the safest interface is not “any iterable of strings.” It is “an iterable whose boundaries are already document boundaries or special-token boundaries.” If that is not available, you need an overlap buffer and delayed emission.

#### Quick Coding: `encode_iterable`

```python
def encode_iterable(self, iterable):
    ...
```

<details>
<summary>Reference answer</summary>

```python
def encode_iterable(self, iterable):
    for chunk in iterable:
        yield from self.encode(chunk)
```

This is correct only if each `chunk` is already a tokenizer-safe boundary, such as a full document or a segment already split by `<|endoftext|>`.

The wrong version is:

```python
def encode_iterable(iterable):
    return self.encode("".join(iterable))
```

That defeats streaming and materializes the full input in memory.

</details>

### Experiment · Compression Ratio

A tokenizer is not judged only by whether it can encode. It is also judged by how well it compresses and how fast it runs on a specific distribution. The three most useful metrics are `bytes/token`, `tokens/s`, and `bytes/s`. The first approximates compression quality. The latter two capture runtime throughput.

Interpret results with the data distribution in mind:

| Observation | Better explanation |
| --- | --- |
| TinyStories tokenizer has high `bytes/token` on TinyStories | domain match lets common story words become long tokens |
| OWT tokenizer is steadier on OWT | the larger vocab covers more web noise, symbols, and multilingual text |
| Cross-domain `bytes/token` gets worse | merges learned local statistics of the training corpus |
| `tokens/s` varies | regex pretokenization, merge data structures, and Python overhead all matter |

To estimate tokenization time for an 825GB corpus:

```python
seconds = 825 * 1024**3 / report["bytes_per_second"]
hours = seconds / 3600
```

### Exercise 3 · Token ID Serialization

The reason to pre-encode text into token ID arrays is practical: you do not want to rerun the tokenizer at every training step. Under the assignment-scale assumption `vocab_size <= 65536`, `uint16` is a good fit because token IDs are non-negative and the storage cost is half of `int32`.

The important validation is dtype-vocab consistency. If the vocabulary has already grown beyond 65,536 and you still use `uint16`, truncation or overflow will show up later in confusing ways.

#### Quick Coding: `encode_to_array`

```python
def encode_to_array(tokenizer: Tokenizer, texts, out_path: str, dtype="uint16") -> dict:
    ...
```

<details>
<summary>Reference answer</summary>

```python
import numpy as np

def encode_to_uint16(tokenizer, texts, out_path):
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text))

    arr = np.asarray(ids, dtype=np.uint16)
    np.save(out_path, arr)
    return {
        "num_tokens": int(arr.shape[0]),
        "dtype": str(arr.dtype),
        "path": str(out_path),
    }
```

For large data, sharded saving is better, and the dataloader can use `np.load(path, mmap_mode="r")`.

</details>

#### Common mistakes in this module

- Decoding token-by-token instead of concatenating bytes first.
- Failing to append special tokens to the vocabulary.
- `encode_iterable` silently materializing the whole file.
- Arbitrary chunk boundaries changing tokenization.

## Module 4: Tensor Modules

Corresponds to CS336 Assignment 1: Sections 3.2-3.4.3.

The jump from tokenization to model implementation is mostly about shape discipline. Most Transformer bugs are not formula bugs. They are batch, sequence, head, and feature dimensions being mixed up. Every exercise in this module trains the same habit: treat the last dimension as feature and everything before it as batch-like.

### Warmup · Tensor Shape Gym

Before writing any real module, it helps to internalize the most common shape transforms:

| Operation | Input | Output |
| --- | --- | --- |
| Linear | `(..., d_in)` | `(..., d_out)` |
| split heads | `(B, T, D)` | `(B, H, T, Dh)` |
| merge heads | `(B, H, T, Dh)` | `(B, T, D)` |
| RMSNorm | `(..., D)` | `(..., D)` |
| RoPE | `(..., T, Dh)` | `(..., T, Dh)` |

If these toy transforms are not clear, later attention and RoPE implementations are unlikely to work the first time.

#### Quick Coding: `shape_gym`

```python
def shape_gym():
    ...
```

<details>
<summary>Reference answer</summary>

```python
import torch
from einops import rearrange

x = torch.randn(2, 3, 12)
W = torch.randn(16, 12)

y = torch.einsum("...i,oi->...o", x, W)
assert y.shape == (2, 3, 16)

h = rearrange(x, "b s (nh dh) -> b nh s dh", nh=3)
assert h.shape == (2, 3, 3, 4)

x2 = rearrange(h, "b nh s dh -> b s (nh dh)")
assert x2.shape == x.shape
```

</details>

### Exercise 1 · Linear Module

`Linear` looks basic, but it fixes the shape convention that almost every later module inherits. The weight must be `(out_features, in_features)` so that `y[..., o] = sum_i x[..., i] * weight[o, i]` matches the semantics of standard linear layers.

Because arbitrary leading dimensions must be preserved, the cleanest implementation uses einsum and treats the last dimension as feature.

#### Quick Coding: `Linear`

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        ...

    def forward(self, x):
        ...
```

<details>
<summary>Reference answer</summary>

```python
import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype
        ))
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return torch.einsum("...i,oi->...o", x, self.we\right)
```

</details>

### Exercise 2 · Embedding Module

An embedding is a lookup table, not a projection. Token IDs are integer indices, and the output is the corresponding row vector. If the weight shape is reversed, the entire `(B, T, D)` convention collapses downstream.

#### Quick Coding: `Embedding`

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        ...

    def forward(self, token_ids):
        ...
```

<details>
<summary>Reference answer</summary>

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        ))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]
```

</details>

### Exercise 3 · RMSNorm

RMSNorm controls the scale of the residual stream. Unlike LayerNorm, it does not subtract the mean. It normalizes only by root-mean-square magnitude along the last dimension.

The real implementation detail is the upcast. BF16 and FP16 can be unstable for sum-of-squares and mean computations, especially at large hidden dimensions, so the statistics are usually computed in FP32 and only cast back at the end.

#### Quick Coding: `RMSNorm`

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        ...

    def forward(self, x):
        ...
```

<details>
<summary>Reference answer</summary>

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        y = x_float / rms
        return (y * self.we\right).to(in_dtype)
```

</details>

### Exercise 4 · SwiGLU Feed-Forward

The intuition of SwiGLU is that one projection provides content and another projection provides a gate. `SiLU(gate) * up` then performs channel-wise modulation. Because SwiGLU adds a third matrix compared with a standard two-matrix FFN, its hidden dimension is usually reduced to roughly `8/3 * d_model` so the parameter count stays comparable to a `4 * d_model` baseline.

#### Quick Coding: `SwiGLU`

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        ...

    def forward(self, x):
        ...
```

<details>
<summary>Reference answer</summary>

```python
import math

def round_up_to_multiple(x, multiple):
    return multiple * math.ceil(x / multiple)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = round_up_to_multiple(int(8 * d_model / 3), 64)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
```

</details>

### Exercise 5 · RoPE

RoPE does not add a positional vector. Instead, it treats each pair of Q/K dimensions as a 2D plane and rotates that plane according to position. This makes the attention score depend more naturally on relative offset rather than only on absolute position.

The core formula is short:

```text
[x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
```

But two implementation details matter:

- `cos` and `sin` can be precomputed for all positions and dimension pairs.
- RoPE applies to Q/K only, not to V.

#### Quick Coding: `RotaryPositionalEmbedding`

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        ...

    def forward(self, x, token_positions):
        ...
```

<details>
<summary>Reference answer</summary>

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        assert d_k % 2 == 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x, token_positions):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.stack((y1, y2), dim=-1).flatten(-2)
```

One very good sanity check is that the L2 norm of each rotated pair should be preserved.

</details>

#### Common mistakes in this module

- All modules should support `device` and `dtype`.
- Print small intermediate tensors before running a full model.
- Mark sequence and head dimensions explicitly instead of inferring them by hope.

## Module 5: Attention and Transformer LM

Corresponds to CS336 Assignment 1: Sections 3.4.4-3.5.

This is where the earlier pieces become a language model. The main thread is not “implement attention.” It is to understand the residual stream: token embeddings enter the model, and every block reads from and writes back to the same `(B, T, D)` stream. Attention and the MLP are just two kinds of updates to that stream.

### Exercise 1 · Stable Softmax

Everyone remembers the softmax formula. What fails in practice is numerical stability: large logits overflow if you exponentiate them directly. So the first step of stable softmax is always subtracting the maximum value along the normalization dimension.

#### Quick Coding: `softmax`

```python
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    ...
```

<details>
<summary>Reference answer</summary>

```python
import torch

def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    shifted = x - x_max
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)
```

</details>

### Exercise 2 · Scaled Dot-Product Attention

Attention is three steps:

1. compute matching scores with `QK^T / sqrt(d_k)`,
2. mask out illegal positions,
3. use softmax weights to aggregate `V`.

The `sqrt(d_k)` scaling is not cosmetic. Without it, the variance of logits grows with the head dimension and softmax becomes too sharp too quickly.

#### Quick Coding: `scaled_dot_product_attention`

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kd->...qd", attn, V)
```

Here `True` means visible and `False` means masked. That semantic choice must stay consistent with every later causal mask.

</details>

### Exercise 3 · Causal MHA

The hard part of causal multi-head self-attention is not the formula but the shapes. Input `(B, T, D)` must be projected into Q/K/V, reshaped to `(B, H, T, Dh)`, treated with the head dimension as batch-like, and then merged back to `(B, T, D)`.

If RoPE is enabled, it rotates Q/K at this stage. The causal mask allows only `j <= i`, so the token at position `i` never sees future tokens.

#### Quick Coding: `CausalMultiHeadSelfAttention`

```python
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        ...

    def forward(self, x, token_positions=None):
        ...
```

<details>
<summary>Reference answer</summary>

```python
from einops import rearrange

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x, token_positions=None):
        B, T, D = x.shape
        q = rearrange(self.q_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "b t (h d) -> b h t d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "b t (h d) -> b h t d", h=self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions[:, None, :])
            k = self.rope(k, token_positions[:, None, :])

        mask = causal_mask(T, T, device=x.device)[None, None, :, :]
        out = scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.o_proj(out)
```

</details>

### Exercise 4 · Transformer Block

The course uses the pre-norm block:

```text
y = x + MHA(RMSNorm(x))
out = y + FFN(RMSNorm(y))
```

Its value is not aesthetics. It makes the gradient path through the residual stream more direct and is usually easier to train. Throughout the block, the residual stream must remain `(B, T, D)`. If that invariant breaks, every later block and the LM head break too.

#### Quick Coding: `TransformerBlock`

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        ...

    def forward(self, x, token_positions=None):
        ...
```

<details>
<summary>Reference answer</summary>

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
```

</details>

### Exercise 5 · Transformer LM

The language-model forward pass is simple: token IDs go through embeddings, then through `N` blocks, then a final norm, then an LM head that maps to vocabulary logits. The model does not need an internal softmax because training consumes logits directly and generation only needs the last position.

#### Quick Coding: `TransformerLM`

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, ...):
        ...

    def forward(self, token_ids):
        ...
```

<details>
<summary>Reference answer</summary>

```python
class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, context_length, d_model, num_layers,
        num_heads, d_ff, rope_theta=10000, device=None, dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(
            rope_theta, d_model // num_heads, context_length, device=device
        )
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, rope=self.rope, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids):
        B, T = token_ids.shape
        assert T <= self.context_length
        positions = torch.arange(T, device=token_ids.device).expand(B, T)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions=positions)
        x = self.ln_final(x)
        return self.lm_head(x)
```

</details>

### Exercise 6 · Resource Accounting

Parameter and FLOP accounting is useful not because it replaces a profiler, but because it tells you where the bottlenecks will come from before you run anything. For assignment-scale Transformers, a useful approximation is:

| Component | Approximate parameter count |
| --- | --- |
| Q/K/V/O | `4 * d_model^2` |
| SwiGLU FFN | `3 * d_model * d_ff` |
| two norms | `2 * d_model` |
| token embedding / LM head | `vocab_size * d_model` |

On the FLOP side, the quantity to watch is the `T^2` term in attention. When context length grows from 1024 to 16384, the MLP still scales linearly in `T`, but QK and PV both scale quadratically.

#### Quick Coding: `transformer_accounting`

```python
def transformer_accounting(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    ...
```

<details>
<summary>Reference answer</summary>

```python
def transformer_params(vocab_size, num_layers, d_model, num_heads, d_ff):
    token_emb = vocab_size * d_model
    final_norm = d_model

    attn = 4 * d_model * d_model
    ffn = 3 * d_model * d_ff
    norms = 2 * d_model
    per_layer = attn + ffn + norms

    lm_head = vocab_size * d_model
    return {
        "token_embedding": token_emb,
        "layers": num_layers * per_layer,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": token_emb + num_layers * per_layer + final_norm + lm_head,
    }
```

Rough FLOPs:

```text
Linear forward FLOPs ≈ 2 * tokens * in_dim * out_dim
Attention QK FLOPs ≈ 2 * B * H * T * T * Dh
Attention PV FLOPs ≈ 2 * B * H * T * T * Dh
MLP FLOPs ≈ 2 * B * T * D * Dff * 3
```

</details>

#### Common mistakes in this module

- The causal-mask `True`/`False` meaning must match the SDPA implementation.
- RoPE applies to Q/K only, not to V.
- The LM head outputs logits, not probabilities.

## Module 6: Training Components

Corresponds to CS336 Assignment 1: Section 4.

Once the forward graph is correct, training stability mostly comes from four classes of components: the loss, the optimizer, the LR schedule, and gradient control. Every problem in this module answers the same question: why can the same model architecture either train smoothly or explode depending on the recipe around it.

### Exercise 1 · Cross-Entropy

Language-model training uses cross entropy on logits, not on already-normalized probabilities. The stable form is log-sum-exp:

```text
CE = logsumexp(logits) - logits[target]
```

This is the same numerical idea as stable softmax: subtract the max first and avoid overflow.

#### Quick Coding: `cross_entropy`

```python
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    ...
```

<details>
<summary>Reference answer</summary>

```python
def cross_entropy(inputs, targets):
    logits = inputs.reshape(-1, inputs.shape[-1])
    y = targets.reshape(-1)

    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    shifted = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + max_logits.squeeze(-1)
    correct = logits[torch.arange(logits.shape[0], device=logits.device), y]
    loss = log_sum_exp - correct
    return loss.mean()
```

</details>

### Lab · SGD LR Toy Sweep

Learning-rate intuition is easiest to build on a 1D quadratic. For `f(x)=x^2`, gradient descent obeys:

```text
x_{t+1} = (1 - 2lr) x_t
```

That one equation already explains the behavior: very small `lr` converges slowly, `lr` near `0.5` is fastest, and large `lr` oscillates or diverges. The assignment’s `1e1/1e2/1e3` settings are intentionally unstable so that you see that regime clearly.

#### Quick Coding: `run_sgd_lr`

```python
def run_sgd_lr(lr: float, steps: int = 10) -> list[float]:
    ...
```

<details>
<summary>Reference answer</summary>

```python
def run_sgd_lr(lr, steps=10):
    x = torch.tensor([10.0])
    values = []
    for _ in range(steps):
        loss = x.pow(2).sum()
        grad = 2 * x
        x = x - lr * grad
        values.append(float(loss))
    return values
```

</details>

### Exercise 2 · AdamW

AdamW is not the same as “Adam plus L2 regularization.” The defining idea is decoupled weight decay: moment statistics see only gradients, while weight decay is applied separately. `weight_decay * p` must not be mixed into `m` and `v`.

Another common error is starting bias correction from timestep 0 instead of 1. That first-step correction is exactly where small models are often most sensitive.

#### Quick Coding: `AdamW.step`

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        ...

    @torch.no_grad()
    def step(self, closure=None):
        ...
```

<details>
<summary>Reference answer</summary>

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                update = m_hat / (torch.sqrt(v_hat) + eps)

                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss
```

</details>

### Exercise 3 · AdamW Accounting

In training memory budgets, optimizer state is often larger than the parameters themselves. If a model has `P` parameters, BF16 parameters plus FP32 moments already means `m` and `v` alone consume an extra `8P` bytes.

When answering this kind of accounting question, separate four categories:

- parameters,
- gradients,
- optimizer state,
- activations.

If activations are not separated out, the effect of batch size and context length is invisible.

#### Quick Coding: `adamw_memory_accounting`

```python
def adamw_memory_accounting(num_params, param_bytes=2, grad_bytes=2, state_bytes=4):
    ...
```

<details>
<summary>Reference answer</summary>

```text
parameters:       P * param_bytes
gradients:        P * grad_bytes
AdamW m:          P * state_bytes
AdamW v:          P * state_bytes
master weights:   optional
activations:      depends on batch_size * context_length * d_model * layers
```

Rough training FLOPs:

```text
forward FLOPs = F
backward FLOPs ≈ 2F
optimizer step FLOPs ≈ O(P)
one train step ≈ 3F + optimizer
```

</details>

### Exercise 4 · Cosine LR with Warmup

Warmup and cosine decay solve different problems. Warmup makes the start of training safe. Cosine decay gradually lowers the step size toward convergence. Once implemented as a piecewise function, the main checks are simple: `it=warmup_iters` should reach `max_lr`, and `it=cosine_cycle_iters` should reach `min_lr`.

#### Quick Coding: `get_lr`

```python
def get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import math

def get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_lr

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)
```

Handle `warmup_iters == 0` separately if needed to avoid division by zero.

</details>

### Exercise 5 · Gradient Clipping

Global gradient-norm clipping treats all parameter gradients as one long vector and limits its L2 norm. The easiest mistake is clipping each tensor independently, which is not the same algorithm.

#### Quick Coding: `clip_grad_norm`

```python
def clip_grad_norm(parameters, max_l2_norm, eps=1e-6):
    ...
```

<details>
<summary>Reference answer</summary>

```python
def clip_grad_norm(parameters, max_l2_norm, eps=1e-6):
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    total = torch.zeros((), device=params[0].grad.device)
    for p in params:
        total += torch.sum(p.grad.detach() ** 2)
    norm = torch.sqrt(total)

    scale = torch.clamp(max_l2_norm / (norm + eps), max=1.0)
    for p in params:
        p.grad.mul_(scale)
    return norm
```

</details>

## Module 7: Training Loop and Generation

Corresponds to CS336 Assignment 1: Sections 5-6.

This is the first place where the earlier components form a real closed loop. A training loop is not a long blob of boilerplate. It is a reproducible sequence that ties together data sampling, forward pass, loss, backward pass, gradient control, LR updates, evaluation, and checkpointing.

### Exercise 1 · Next-Token Batch Sampler

Language-model data loading is simple in principle: sample a window of length `context_length + 1` from a 1D token array, take the first `context_length` tokens as `x`, and the last `context_length` tokens as `y`. In practice, an off-by-one error in the start index immediately corrupts the target shift.

#### Quick Coding: `get_batch`

```python
def get_batch(dataset, batch_size: int, context_length: int, device):
    ...
```

<details>
<summary>Reference answer</summary>

```python
import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    n = len(dataset)
    starts = torch.randint(0, n - context_length, (batch_size,))

    xs, ys = [], []
    for s in starts.tolist():
        chunk = dataset[s : s + context_length + 1]
        if isinstance(chunk, np.ndarray):
            chunk = torch.from_numpy(chunk.astype(np.int64))
        else:
            chunk = torch.as_tensor(chunk, dtype=torch.long)
        xs.append(chunk[:-1])
        ys.append(chunk[1:])

    x = torch.stack(xs).to(device=device, dtype=torch.long)
    y = torch.stack(ys).to(device=device, dtype=torch.long)
    return x, y
```

</details>

### Exercise 2 · Checkpoint Save / Load

The minimum checkpoint contains only three things: `model.state_dict()`, `optimizer.state_dict()`, and the iteration number. Dropping any of them changes resume behavior. In particular, if optimizer state is missing, AdamW loses its moments and the loss curve often jumps at resume time.

#### Quick Coding: `save_checkpoint`

```python
def save_checkpoint(model, optimizer, iteration: int, out):
    ...

def load_checkpoint(src, model, optimizer) -> int:
    ...
```

<details>
<summary>Reference answer</summary>

```python
def save_checkpoint(model, optimizer, iteration, out):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, out)

def load_checkpoint(src, model, optimizer):
    payload = torch.load(src, map_location="cpu")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload["iteration"]
```

</details>

### Exercise 3 · Full Training Script

The safest debugging order for a full training loop is:

1. overfit one fixed minibatch,
2. connect the real dataloader,
3. enable validation evaluation,
4. test checkpoint resume.

This sequence works because each step adds exactly one new source of complexity. If you launch full training immediately, a non-decreasing loss gives you no clue whether the issue is data, model, loss, or optimizer.

#### Quick Coding: `train`

```python
def train(config):
    ...
```

<details>
<summary>Reference answer</summary>

```python
for it in range(start_iter, max_iters):
    lr = get_lr(it, max_lr, min_lr, warmup_iters, cosine_iters)
    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = get_batch(train_tokens, batch_size, context_length, device)
    logits = model(x)
    loss = cross_entropy(logits, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm(model.parameters(), max_grad_norm)
    optimizer.step()

    if it % log_interval == 0:
        print({"iter": it, "loss": float(loss), "lr": lr})

    if it % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            for _ in range(eval_iters):
                vx, vy = get_batch(val_tokens, batch_size, context_length, device)
                val_losses.append(cross_entropy(model(vx), vy).item())
        model.train()
        print({"iter": it, "val_loss": sum(val_losses) / len(val_losses)})

    if it % ckpt_interval == 0:
        save_checkpoint(model, optimizer, it, ckpt_path)
```

The most common implementation errors are:

- forgetting `optimizer.zero_grad()`,
- forgetting `torch.no_grad()` during eval,
- calling `model.eval()` and never switching back to `model.train()`,
- logging the loss tensor itself instead of `loss.item()`.

</details>

### Exercise 4 · Autoregressive Decoder

Each generation step does one thing: read the logits at the final position, sample one next token, and append it to the context. Temperature and top-p change the sampling distribution, not the model itself. If the context grows too long, only the most recent window matters because a causal LM cannot attend outside its configured context anyway.

#### Quick Coding: `generate`

```python
@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature=1.0, top_p=1.0, eos_token_id=None):
    ...
```

<details>
<summary>Reference answer</summary>

```python
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature=1.0, top_p=1.0, eos_token_id=None):
    model.eval()
    ids = tokenizer.encode(prompt)
    tokens = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)

    for _ in range(max_new_tokens):
        idx_cond = tokens[:, -model.context_length:]
        logits = model(idx_cond)[:, -1, :]

        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            probs = top_p_filter(probs, top_p)
            next_id = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_id], dim=1)
        if eos_token_id is not None and int(next_id.item()) == eos_token_id:
            break

    return tokenizer.decode(tokens[0].tolist())

def top_p_filter(probs, top_p):
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    keep = cdf <= top_p
    keep[..., 0] = True
    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_idx, src=sorted_probs * keep)
    return filtered / filtered.sum(dim=-1, keepdim=True)
```

Top-p must be followed by re-normalization, and `temperature=0` should explicitly choose greedy decoding instead of dividing by zero.

</details>

## Module 8: Experiments and Ablations

Corresponds to CS336 Assignment 1: Section 7.

This module is not about writing more functions. It is about building a reproducible experimentation habit. Training a model is not the end. You also need to know which code, config, and token budget produced a run, and whether a modification actually helped.

### Experiment 1 · Experiment Logger

If you later look at a validation-loss curve and cannot tell which code, config, and checkpoint produced it, that run is barely useful. At minimum, the logger should record a run name, git commit or config hash, train/val loss, tokens processed, wall-clock time, checkpoint path, and generated samples.

#### Quick Coding: `ExperimentLogger`

```python
class ExperimentLogger:
    def __init__(self, path: str, config: dict):
        ...

    def log(self, step: int, **metrics):
        ...
```

<details>
<summary>Reference answer</summary>

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

</details>

### Experiment 2 · TinyStories LR Sweep

An LR sweep is about finding the stability boundary, not just a best-looking point. So only the learning rate should change. Model, tokenizer, data order or seed, total token budget, batch size, and schedule shape must stay fixed.

A log-scale sweep is a good default:

```text
1e-4, 3e-4, 1e-3, 3e-3, 1e-2
```

The report should interpret curve shapes rather than only present a final table:

- loss barely decreases: LR too small or budget too short,
- loss decreases then explodes: LR near or beyond the stability boundary,
- train loss falls while val loss does not: likely overfitting or tiny data,
- best LR shifts with batch size: gradient noise scale changed.

### Experiment 3 · Batch Size Sweep

A batch-size study must separate micro-batch size from effective batch size. The former controls VRAM and step time. The latter may include gradient accumulation and directly changes optimizer-step frequency.

A useful table is:

```text
batch_size | grad_accum | effective_batch | lr | tokens/sec | max_mem | best_val_loss
```

Step time alone is not a sufficient conclusion. Better comparisons are validation loss at equal wall-clock time or equal tokens processed.

### Experiment 4 · Generate TinyStories Samples

Sample generation is really about the interaction between checkpoint quality and sampling parameters. TinyStories is best judged by story coherence, character consistency, and sentence completeness, not by factual accuracy.

After fixing a prompt, compare at least:

| Variable | Role |
| --- | --- |
| checkpoint step | how much the model has learned |
| temperature | how sharp the distribution is |
| top_p | how much probability mass is retained |
| prompt | where the conditional distribution starts |

Typical interpretations:

- repeated short sentences: undertraining or too-low temperature,
- broken grammar: early checkpoint or too-high temperature,
- fluent but monotonous: top-p too small or prompt too constraining,
- very early EOS: EOS learned too strongly or training data is dominated by short texts.

### Experiment 5 · Remove RMSNorm

This ablation is about stability, not just one best loss value. The clean baseline is pre-norm with the previously best LR, then compare “remove RMSNorm at the same LR” and “remove RMSNorm at a lower LR.”

Track more than train/val loss:

- grad norm,
- activation norm,
- divergence step.

The reasonable expectation is that removing RMSNorm makes loss spikes or large grad norms more likely at the same LR. Lowering the LR may make training possible but often slows convergence or worsens final loss.

### Experiment 6 · Post-Norm Transformer

The pre-norm versus post-norm comparison should change only the block structure, not the parameter scale. A standard post-norm block is:

```python
class PostNormBlock(nn.Module):
    def forward(self, x, token_positions=None):
        x = self.ln1(x + self.attn(x, token_positions=token_positions))
        x = self.ln2(x + self.ffn(x))
        return x
```

The best report usually includes two plots:

- same-LR comparison,
- retuned-best-LR comparison.

That separates “this structure is intrinsically less stable” from “this structure merely needs a lower LR.”

### Experiment 7 · NoPE vs RoPE

The real question is whether a model can learn sequence information without explicit positional encoding. The answer is usually not a clean yes/no. On short context and small data, NoPE may still reduce loss. On long context or tasks that require positional structure, it usually degrades more clearly.

Do not look only at final loss. Add at least:

- generated samples,
- context-length sensitivity.

That is where NoPE often reveals repetitions, ordering failures, and weaker long-range consistency.

### Experiment 8 · SwiGLU vs SiLU FFN

This comparison only makes sense if parameter counts are approximately matched:

```text
SwiGLU: 3 * d_model * d_ff_swiglu
SiLU FFN: 2 * d_model * d_ff_silu
```

That is why the common settings are:

```text
d_ff_silu ≈ 4 * d_model
d_ff_swiglu ≈ 8/3 * d_model
```

Without that normalization, saying “SwiGLU is better” is not a fair comparison. The report should include parameter count, tokens/sec, best val loss, and run variance.

### Experiment 9 · OpenWebText Run

The main theme of this run is domain shift. OWT is more diverse, noisier, and harder to compress than TinyStories. The easy mistake is to compare their losses directly. Different data distributions have different entropy, so a higher OWT loss does not automatically mean the model is worse.

Better questions are:

- does the OWT curve still decrease stably,
- do samples look more general or just noisier,
- does tokenizer `bytes/token` degrade substantially,
- does the same compute budget produce more transferable language behavior.

### Experiment 10 · Leaderboard-Style Modification

This problem is closest to a small research experiment. The correct structure is:

1. state a hypothesis,
2. change one major variable,
3. compare baseline and modified runs under the same token budget and eval protocol,
4. provide evidence instead of slogans.

Common directions:

| Modification | Hypothesis | Risk |
| --- | --- | --- |
| weight tying | fewer parameters and possible regularization | reduced output-layer flexibility |
| retuned LR schedule | the original schedule is suboptimal | more experiment budget required |
| better init | improved early stability | may only affect the first few hundred steps |
| fused kernels / compile | higher tokens/sec | may not improve final loss |
| tokenizer change | better compression | changes the training distribution and hurts comparability |

For example, weight tying is as simple as:

```python
model.lm_head.weight = model.token_embeddings.weight
```

A negative result can still be a good answer. The important part is a clean design and a conclusion that matches the evidence.

## Final End-to-End Debug Checklist

### 1. Tokenizer path

- Are `str`, code points, and UTF-8 bytes clearly separated?
- Are special tokens hard boundaries during training and whole tokens at runtime?
- Do BPE merges stay inside one pre-token?
- Does decode concatenate bytes before UTF-8 decoding?

### 2. Model forward path

- Do shapes consistently follow `(B, T, D)` or `(B, H, T, Dh)`?
- Is RoPE applied only to Q/K?
- Does causal-mask semantics match the SDPA implementation?
- Does the LM head output logits rather than probabilities?

### 3. Training path

- Is `y` really the one-token-right-shifted target of `x`?
- Does cross entropy consume logits directly?
- Does AdamW use decoupled weight decay and correct bias correction?
- Does gradient clipping apply one global scale?
- Does evaluation run under `torch.no_grad()`?

### 4. Experiment path

- Does every run record config, commit, loss, tokens, wall-clock time, and checkpoint?
- Does each ablation change one major variable at a time?
- Are comparisons made at matched token budget or matched wall-clock?
- Do conclusions come from curves, samples, and logs rather than preference?

If these four paths can each be debugged independently, you already have a workable minimal workflow for going from raw text to reproducible LLM experiments.

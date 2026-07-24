# ML Coding · BPE Training

Corresponds to CS336 Assignment 1: Section 2.4-2.5.

Usage: For each problem, first review the objectives and acceptance criteria, then complete the `TODO`s using the "Problem Template." Finally, expand the reference solution to check against boundary conditions, sanity checks, and implementation details.

## Exercise 1 · Toy BPE Merge Simulator

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `bpe_example`

Objective: Implement a naive BPE merge loop on a small corpus.

Input:

```text
pretoken_counts: dict[tuple[bytes, ...], int]
num_merges: int
```

Output:

```text
merges: list[tuple[bytes, bytes]]
updated pretoken representation
```

Key Constraints:

- Pair frequency must be multiplied by pre-token frequency.
- Only count adjacent pairs.
- In case of frequency ties, select the lexicographically greater pair.
- Merges replace non-overlapping adjacent occurrences.

PDF toy corpus:

```text
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

Objective: The first few merge rounds should reproduce the handout results.

Problem Template:

```python
from collections import Counter

def count_pairs(pretoken_counts: dict[tuple[bytes, ...], int]) -> Counter:
    """Output: Counter mapping adjacent byte-pair to weighted frequency."""
    pair_counts = Counter()
    for pieces, freq in pretoken_counts.items():
        ...  # add freq for every adjacent pair
    return pair_counts

def merge_one_pretoken(pieces: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Output: pieces after non-overlapping replacement of pair."""
    out = []
    i = 0
    while i < len(pieces):
        ...  # if pieces[i:i+2] == pair, append merged bytes and skip 2
    return tuple(out)

def run_bpe_merges(pretoken_counts, num_merges):
    """Output: (merges, updated_counts)."""
    merges = []
    counts = dict(pretoken_counts)
    for _ in range(num_merges):
        pair_counts = ...
        winner = ...  # max by (frequency, pair)
        merges.append(winner)
        counts = ...  # rebuild counts after merge
    return merges, counts
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The core of the naive version consists of two steps: counting pair frequencies, and then performing non-overlapping replacement of the winner pair within each pre-token.

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

Use `(frequency, pair)` for tie-breaking, as the assignment requires selecting the lexicographically greater pair when frequencies are equal. Do not use `Counter.most_common(1)`, as its tie-breaking depends on insertion order.

</details>

## Exercise 2 · Full BPE Trainer

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `train_bpe`

Interface:

```text
train_bpe(input_path, vocab_size, special_tokens)
```

Returns:

```text
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
```

Key Constraints:

- Initial byte vocabulary covers 0..255.
- Special tokens are added to the vocabulary.
- Special tokens act as hard boundaries for merges.
- Special tokens do not participate in pair statistics.
- `vocab_size` includes the byte vocab, merge vocab, and special tokens.
- Merges are returned in the order they were created.

Test:

```bash
uv run pytest tests/test_train_bpe.py
```

Problem Template:

```python
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    """
    Input:
        raw corpus path, target vocab size, special tokens
    Output:
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
    """
    vocab = {i: bytes([i]) for i in range(256)}
    ...  # append special tokens to vocab

    pretoken_counts = ...  # build counts with special-token boundaries
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = ...
        if not pair_counts:
            break
        pair = ...          # deterministic winner
        merges.append(pair)
        vocab[len(vocab)] = ...
        pretoken_counts = ...

    return vocab, merges
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

Recommended structure:

```python
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

`build_pretoken_counts` should reuse the rules from the previous chapter: special tokens are hard boundaries and are not included in statistics; only within normal pre-tokens are sequences split into byte pieces. `vocab_size` includes the 256 byte tokens, special tokens, and new tokens generated by merges, so the loop condition is `len(vocab) < vocab_size`.

</details>

## Exercise 3 · BPE Performance Pass

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Objective: Optimize the naive trainer to handle TinyStories.

Optimization sequence:

```text
1. profile pre-tokenization
2. split corpus by special-token boundary
3. parallelize pre-tokenization
4. cache pair counts
5. update only pairs affected by last merge
```

Record:

```text
wall-clock time
peak memory
top bottleneck
speedup after each optimization
```

Problem Template:

```python
def profile_stage(name: str, fn, *args, **kwargs):
    """Run one stage and return result + timing/memory metadata."""
    ...

def build_pair_index(pretoken_counts):
    """
    Output:
        pair_counts: pair -> frequency
        pair_to_pretokens: pair -> affected pre-token set
    """
    ...

def update_after_merge(pretoken_counts, pair_counts, pair_to_pretokens, winner):
    """
    Only update pre-tokens containing winner.
    """
    affected = ...
    for old_pieces in affected:
        ...  # remove old pair contributions
        new_pieces = ...
        ...  # add new pair contributions
    return pretoken_counts, pair_counts, pair_to_pretokens

def benchmark_bpe_trainer(input_path):
    rows = []
    for version in ["naive", "parallel_pretok", "incremental_pairs"]:
        ...  # record time, memory, speedup
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The optimized version should not perform a full scan of the corpus in every round. A robust engineering approach:

```text
1. First, split the raw corpus into document chunks, using special tokens only as boundaries.
2. Parallelize pretokenization to obtain Counter[tuple[bytes, ...]].
3. Build an inverted index of pair -> set[pretoken].
4. Select the winner pair each round.
5. Update only the pre-tokens containing the winner pair.
6. Decrement old pairs and increment new pairs for these pre-tokens.
```

Pseudocode:

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

When validating, report not just the total time, but the wall-clock time and peak memory after each optimization step. The most common bottlenecks for a BPE trainer are typically regex pretokenization and full-corpus pair recounting in every round.

</details>

## Experiment · Train TinyStories Tokenizer

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `train_bpe_tinystories`

Configuration:

```text
dataset: TinyStories
vocab_size: 10_000
special token: <|endoftext|>
```

Output:

```text
serialized vocab
serialized merges
training time
memory usage
longest token
profile bottleneck
```

Experiment Template:

```python
def train_tinystories_tokenizer(data_path: str, out_dir: str):
    """
    Output artifacts:
        vocab.json
        merges.txt
        report dict
    """
    vocab, merges = train_bpe(
        input_path=data_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )
    ...  # serialize vocab and merges
    report = {
        "dataset": "TinyStories",
        "vocab_size": 10_000,
        "num_merges": ...,
        "train_time_sec": ...,
        "peak_rss_mb": ...,
        "longest_tokens": ...,
    }
    return report
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

A reproducible output template:

```text
dataset: TinyStories
vocab_size: 10000
special_tokens: ["<|endoftext|>"]
num_merges: 10000 - 256 - 1 = 9743
artifacts:
  vocab.json
  merges.txt
metrics:
  train_time_sec
  peak_rss_mb
  pretoken_count
  longest_token_bytes
  top_20_longest_tokens
```

Evaluating if the tokenizer is reasonable:

- High-frequency English words, space-prefixed words, and common suffixes should become longer tokens.
- `<|endoftext|>` must exist as a standalone token.
- The TinyStories domain is simple; long tokens should mostly be common words and names found in children's stories, rather than gibberish.
- If the longest token is formed by merging across documents, the special-token boundary handling is likely incorrect.

</details>

## Experiment · Train OpenWebText Tokenizer

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `train_bpe_expts_owt`

Configuration:

```text
dataset: OpenWebText sample
vocab_size: 32_000
```

Comparison:

| Dimension | TinyStories | OpenWebText |
|---|---|---|
| domain | children stories | web text |
| vocabulary diversity | lower | higher |
| long tokens | simple words / names | URLs, markup, rare strings |
| compression | easier | more variable |

Experiment Template:

```python
def train_owt_tokenizer(data_path: str, out_dir: str):
    """
    Train a 32K OpenWebText tokenizer and return artifacts + diagnostics.
    """
    vocab, merges = train_bpe(
        input_path=data_path,
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
    )
    return {
        "dataset": "OpenWebText",
        "vocab_size": 32_000,
        "num_merges": ...,
        "bytes_per_token": ...,
        "tokens_per_second": ...,
        "longest_tokens": ...,
        "most_frequent_tokens": ...,
    }

def compare_tokenizers(tinystories_tok, owt_tok, tinystories_sample, owt_sample):
    """
    Output:
        table comparing bytes/token across tokenizer/data pairs.
    """
    ...
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The focus of the OpenWebText answer is not just to "generate a fixed table," but to explain why the results differ from TinyStories:

```text
dataset: OpenWebText sample
vocab_size: 32000
expected differences:
  more URL / markup / code-like fragments
  more rare names and non-English text
  heavier long tail in pre-token distribution
  slower pretokenization and pair update
```

Include at least the following in the report:

```text
bytes/token on OWT validation sample
tokens/sec encode throughput
top longest tokens
top most frequent tokens
TinyStories tokenizer on OWT vs OWT tokenizer on OWT
```

Typical conclusion: The OWT tokenizer achieves better compression on web text, but the vocabulary will be partially occupied by URLs, HTML, symbol strings, and multilingual content; the TinyStories tokenizer is cleaner, but tokens/byte will degrade when transferred to OWT.

</details>

## Common Failure Modes

<details class="exercise">
<summary><span class="q-label">Pitfalls</span> <span class="q-text">Expand common errors</span></summary>

- Special tokens are split and participate in merges.
- Merges cross pre-token boundaries.
- Pair tie-breaking is not deterministic.
- Individual bytes are represented as `int`, causing type mismatches in vocab/merges.
- Scanning the entire corpus from scratch every round, causing training speed to degrade.

</details>

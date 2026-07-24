# ML Coding · Tokenizer Runtime

Corresponds to CS336 Assignment 1: Sections 2.6-2.7.

Usage: For each exercise, first review the objectives and acceptance criteria, then complete the TODOs using the "Solution Template." Finally, expand the reference solution to compare against boundary conditions, sanity checks, and implementation details.

## Exercise 1 · Tokenizer Class

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `tokenizer`

Interface:

```text
Tokenizer(vocab, merges, special_tokens=None)
Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=None)
encode(text: str) -> list[int]
encode_iterable(iterable: Iterable[str]) -> Iterator[int]
decode(ids: list[int]) -> str
```

Key constraints:

- Special tokens must remain intact during encoding.
- Ordinary text is pre-tokenized first, then BPE merges are applied within each pre-token.
- Merges must be applied in the order they were created during training.
- Decoding involves concatenating token bytes followed by UTF-8 decoding.
- Malformed bytes must use the Unicode replacement character.
- `encode_iterable` must be memory-efficient.

Testing:

```bash
uv run pytest tests/test_tokenizer.py
```

Solution Template:

```python
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = ...
        self.inverse_vocab = ...
        self.merge_rank = ...
        self.special_tokens = ...
        ...  # add special tokens to vocab if missing

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = ...     # load int -> bytes
        merges = ...    # load pair order
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        ids = []
        ...  # split special tokens, pretokenize ordinary text, apply BPE
        return ids

    def encode_iterable(self, iterable):
        for chunk in iterable:
            yield from ...

    def decode(self, ids: list[int]) -> str:
        raw = ...       # concatenate token bytes
        return ...      # utf-8 decode with replacement
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

This exercise requires implementing a functional BPE tokenizer runtime: the training phase has already provided the `vocab` and `merges`, and the runtime is responsible for stably mapping strings to token IDs and reconstructing text from IDs.

A clean implementation can be broken down into three layers:

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)                  # int -> bytes
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
```

The core of encoding is: first protect special token boundaries, then pre-tokenize ordinary segments, and finally perform BPE merges within each pre-token.

```python
def apply_bpe(self, token_bytes: bytes) -> list[bytes]:
    pieces = [bytes([b]) for b in token_bytes]
    while len(pieces) >= 2:
        pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
        best = min(
            pairs,
            key=lambda p: self.merge_rank.get(p, float("inf")),
        )
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
```

Do not decode token-by-token in `decode`. The correct approach is to concatenate the bytes first, then perform a single UTF-8 decode:

```python
def decode(self, ids: list[int]) -> str:
    raw = b"".join(self.vocab[i] for i in ids)
    return raw.decode("utf-8", errors="replace")
```

Acceptance criteria:

- `decode(encode(text)) == text` holds for valid UTF-8 text.
- Special tokens are not fragmented by regex.
- Merges occur only within the same pre-token and cannot cross pre-token boundaries.
- Malformed byte sequences use the replacement character during decoding without throwing exceptions.

</details>

## Trace Lab · BPE Encoding Trace

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Manually trace using a small vocabulary and set of merges:

```text
input: "the cat ate"
pre-tokenized: ["the", " cat", " ate"]
```

Exercise objectives:

- Print the byte pieces for each pre-token.
- Print the pieces after each merge step.
- Finally, map to token IDs.

Checkpoints:

```text
merge order changes output
merge only applies inside one pre-token
unknown text still encodable because byte vocab is complete
```

Trace Template:

```python
def trace_bpe_token(token: str, tokenizer: Tokenizer) -> list[int]:
    """
    Input:
        one pre-token string, not a full document
    Output:
        final token ids, while printing each merge step
    """
    pieces = ...        # byte pieces
    print("start:", pieces)
    while True:
        candidate_pairs = ...
        best_pair = ... # lowest merge rank among current adjacent pairs
        if best_pair is None:
            break
        pieces = ...    # merge best_pair
        print(...)
    ids = ...
    return ids
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

This exercise is designed to help you understand the intermediate states of the tokenizer. When debugging BPE, the final token IDs often don't reveal the source of an issue; you must print the pieces after each round of merging.

A trace helper can be written as follows:

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

For `"the cat ate"`, you must first observe the pre-token boundaries:

```text
"the"
" cat"
" ate"
```

Even if a merge rule could combine `"e"` and `" "`, it cannot merge across the boundary between `"the"` and `" cat"`. The BPE merge scope is limited to the interior of a pre-token, not the entire string.

Common bugs:

- Concatenating all bytes into a single sequence before merging, causing merges across spaces/special tokens.
- Incorrect merge tie-breaking or merge rank, resulting in output IDs that differ slightly from the reference.
- Missing byte fallback in the vocabulary, causing unknown characters to be unencodable.

</details>

## Exercise 2 · Streaming Encode

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Objective: Perform lazy tokenization on large files that cannot fit into memory.

Input:

```text
Iterable[str]
```

Output:

```text
Iterator[int]
```

Key issues:

- Chunk boundaries must not alter tokenization.
- Special tokens / document boundaries are naturally safe split points.
- If chunks are split arbitrarily by character count, it may change pre-tokenization and merge results.

Solution Template:

```python
def encode_iterable(self, iterable):
    """
    Input:
        iterable of safe-boundary text chunks
    Output:
        lazy iterator of token ids
    """
    for chunk in iterable:
        for token_id in self.encode(chunk):
            yield token_id

def encode_documents(self, docs):
    """
    Same idea, but docs are explicit tokenizer-safe boundaries.
    """
    for doc in docs:
        yield from self.encode(doc)
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The goal of streaming encoding is to process massive text files without loading the entire content into memory. The safest version is to have the caller provide an iterable of documents or lines, where these boundaries are semantically valid split points.

```python
def encode_iterable(self, iterable):
    for chunk in iterable:
        yield from self.encode(chunk)
```

This version is simple but assumes that `chunk` boundaries do not alter tokenization. For example, each chunk is a document, or the upstream has already split by special tokens:

```text
doc1 <|endoftext|> doc2 <|endoftext|> doc3
```

If you split arbitrarily by a fixed character count:

```text
"intern" + "ational"
```

A word that could have been merged into a long token might be cut, changing the final token IDs. Therefore, there are two more rigorous streaming strategies:

1. Split by document/special token boundaries and use `yield from encode(doc)`.
2. If you must read by byte chunks, maintain an overlap buffer and only yield tokens before a confirmed safe boundary.

In the CS336 assignment implementation, processing by iterable text chunks is usually sufficient; the key is to avoid writing:

```python
def encode_iterable(iterable):
    return self.encode("".join(iterable))  # Wrong: materializes the entire file
```

Verification method:

```python
ids_stream = list(tok.encode_iterable(docs))
ids_joined = tok.encode("".join(docs))
```

The two should be identical only when the boundaries of `docs` are safe boundaries. This condition should be clearly stated in the experiment documentation.

</details>

## Experiment · Compression Ratio

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Corresponds to PDF: `tokenizer_experiments`

Metrics:

```text
bytes / token
tokens / second
bytes / second
estimated time for 825GB corpus
```

Experiment Matrix:

| data | tokenizer | report |
|---|---|---|
| TinyStories sample | TinyStories 10K | bytes/token |
| TinyStories sample | OWT 32K | bytes/token |
| OWT sample | TinyStories 10K | bytes/token |
| OWT sample | OWT 32K | bytes/token |

Experiment Template:

```python
def tokenizer_report(tokenizer: Tokenizer, text: str) -> dict:
    """
    Output:
        compression and throughput metrics
    """
    raw = text.encode("utf-8")
    start = ...
    ids = tokenizer.encode(text)
    elapsed = ...
    return {
        "num_bytes": ...,
        "num_tokens": ...,
        "bytes_per_token": ...,
        "tokens_per_second": ...,
        "bytes_per_second": ...,
    }

def compare_tokenizer_matrix(samples: dict, tokenizers: dict) -> list[dict]:
    rows = []
    for sample_name, text in samples.items():
        for tok_name, tok in tokenizers.items():
            rows.append({...})
    return rows
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

This section is not about implementing new algorithms, but about evaluating whether a tokenizer is suitable for a specific data distribution. It is recommended to write a unified benchmark function:

```python
import time

def tokenizer_report(tokenizer, text: str) -> dict:
    raw = text.encode("utf-8")
    start = time.perf_counter()
    ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start

    return {
        "num_bytes": len(raw),
        "num_tokens": len(ids),
        "bytes_per_token": len(raw) / max(len(ids), 1),
        "tokens_per_second": len(ids) / max(elapsed, 1e-12),
        "bytes_per_second": len(raw) / max(elapsed, 1e-12),
    }
```

When interpreting results, consider the corpus:

| Phenomenon | Explanation |
|---|---|
| TinyStories tokenizer has high bytes/token on TinyStories | Domain match; common children's story words are merged into longer tokens |
| OWT tokenizer is more stable on OWT | OWT has more diverse vocabulary, symbols, and web noise; 32K vocab provides better coverage |
| Cross-domain bytes/token worsens | Merges learn local statistics of the training corpus, not universal compression truths |
| tokens/s is not determined solely by vocab | Regex pre-tokenization, merge data structures, and Python overhead all affect speed |

Estimating time for an 825GB corpus:

```python
seconds = 825 * 1024**3 / report["bytes_per_second"]
hours = seconds / 3600
```

The report should cover at least three things:

1. Compression ratio: `bytes/token`.
2. Throughput: `tokens/s` and `bytes/s`.
3. Transferability: Why the same tokenizer improves or worsens across different data domains.

</details>

## Exercise 3 · Token ID Serialization

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interface, and acceptance criteria</span></summary>

Objective: Encode train/dev text into a token ID array.

Output:

```text
np.ndarray dtype uint16
```

Why `uint16` is appropriate:

```text
vocab_size <= 65,536
uint16 halves storage compared with int32
token ids are non-negative integers
```

Solution Template:

```python
def encode_to_array(tokenizer: Tokenizer, texts, out_path: str, dtype="uint16") -> dict:
    """
    Input:
        iterable of text documents
    Output:
        saved numpy token-id array and metadata
    """
    ids = []
    for text in texts:
        ids.extend(...)
    arr = ...
    ...  # validate max id fits dtype
    ...  # save arr
    return {
        "path": out_path,
        "num_tokens": ...,
        "dtype": ...,
        "max_token_id": ...,
    }
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

Before training, it is common to pre-encode text into a token ID array to avoid running the tokenizer at every training step. Basic implementation:

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

If the data is very large, do not accumulate a Python list; write in shards instead:

```python
def write_token_shards(tokenizer, docs, shard_size, prefix):
    buf = []
    shard_id = 0
    for doc in docs:
        buf.extend(tokenizer.encode(doc))
        if len(buf) >= shard_size:
            np.save(f"{prefix}-{shard_id:05d}.npy", np.asarray(buf, dtype=np.uint16))
            buf.clear()
            shard_id += 1
    if buf:
        np.save(f"{prefix}-{shard_id:05d}.npy", np.asarray(buf, dtype=np.uint16))
```

The prerequisite for `uint16` is that the vocabulary size does not exceed 65,536. If the vocabulary size is 100K, you must use `uint32` or a larger dtype. When reading with a training dataloader, you can use:

```python
tokens = np.load(path, mmap_mode="r")
```

This prevents loading the entire array into memory at once.

Checkpoints:

- `arr.min() >= 0`
- `arr.max() < vocab_size`
- dtype matches vocabulary size
- Randomly sampling a segment of IDs and decoding them yields reasonable text

</details>

## Common Failure Modes

<details class="exercise">
<summary><span class="q-label">Pitfalls</span> <span class="q-text">Expand common errors</span></summary>

- Decoding token-by-token instead of concatenating bytes first.
- Failing to append special tokens to the vocabulary.
- `encode_iterable` silently materializing the whole file.
- Streaming chunks altering tokenization.

</details>

# ML Coding · Unicode, UTF-8 & Pretokenization

Corresponds to CS336 Assignment 1: Sections 2.1-2.4.

Usage: `Lab` is intended for building intuition and does not count as a formal coding assignment; `Exercise` tasks require you to complete the implementation yourself. Follow the template first, then expand the reference solution to check against edge cases.

## Lab · Unicode Probe

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `unicode1`

Implement a small utility to inspect code points:

```text
inspect_unicode_codepoint(cp: int)
```

Returns:

```text
character
repr(character)
UTF-8 bytes
visible printing behavior
```

Checkpoints:

- `ord` / `chr` are inverses at the code point level.
- `repr(x)` and `print(x)` serve different display purposes.
- Control characters may exist but remain invisible.

Observation template:

```python
def inspect_unicode_codepoint(cp: int) -> dict:
    """
    Input:
        cp: Unicode code point, e.g. 65 or 0x1F600
    Output:
        metadata useful for debugging display vs encoding
    """
    ch = ...       # chr(cp)
    utf8 = ...     # ch.encode("utf-8")
    return {
        "codepoint": ...,
        "character": ...,
        "repr": ...,
        "utf8_bytes": ...,
        "utf8_hex": ...,
        "is_printable": ...,
    }
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

Minimal implementation:

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

Sanity check:

```python
assert ord(chr(65)) == 65
assert inspect_unicode_codepoint(0x41)["utf8_bytes"] == [65]
assert inspect_unicode_codepoint(0x1F600)["utf8_bytes"] == [240, 159, 152, 128]
```

`print(ch)` is the human-readable display effect, where control characters may be invisible; `repr(ch)` is the debugging representation, which exposes content like newlines and zero-width characters.

</details>

## Lab · UTF-8 Encoding

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: `unicode2`

Compare UTF-8, UTF-16, and UTF-32 encoding for different types of text:

```text
English
CJK
emoji
mixed text
invalid byte sequence
```

Output:

```text
encoded bytes
byte length
round-trip result
decode error behavior
```

Sanity cases:

```text
"hello" round-trip succeeds
"こんにちは" byte length > character count
single-byte decoding a multi-byte character is wrong
invalid two-byte sequence raises or replaces
```

Observation template:

```python
def compare_encodings(text: str) -> list[dict]:
    """
    Input:
        text: Python unicode string
    Output:
        rows comparing utf-8 / utf-16 / utf-32 byte length and round-trip
    """
    rows = []
    for enc in ["utf-8", "utf-16", "utf-32"]:
        raw = ...              # text.encode(enc)
        rows.append({
            "encoding": enc,
            "num_chars": ...,
            "num_bytes": ...,
            "bytes": ...,
            "roundtrip": ...,
        })
    return rows

def decode_invalid(raw: bytes, encoding="utf-8") -> dict:
    """
    Input:
        raw byte sequence, possibly malformed
    Output:
        strict / replace / ignore behavior
    """
    return {
        "strict": ...,
        "replace": ...,
        "ignore": ...,
    }
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

Experimental functions:

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

Conclusions:

- ASCII is typically 1 char = 1 byte in UTF-8.
- CJK and emojis are multi-byte in UTF-8; `len(text)` and `len(text.encode("utf-8"))` are not the same.
- UTF-16 / UTF-32 usually have BOM or fixed-width overhead, so short English text is not necessarily more efficient.
- For invalid byte sequences, you should explicitly decide whether to use `strict`, `replace`, or `ignore`; do not silently swallow errors in tokenizers or debugging tools.

</details>

## Exercise 1 · GPT-2 Style Pretokenizer

<details class="exercise">
<summary><span class="q-label">Reference</span> <span class="q-text">Expand objectives, interfaces, and acceptance criteria</span></summary>

Corresponds to PDF: pre-tokenization in Section 2.4.

Objective: Use the assignment regex to split raw text into pre-tokens and convert them into byte tuples.

Input:

```text
raw text
special_tokens: list[str]
```

Output:

```text
dict[tuple[bytes, ...], int]
```

Key constraints:

- Special tokens are hard boundaries.
- Special tokens do not contribute to merge statistics.
- Do not count byte pairs across pre-token boundaries.
- Use an iterator-style approach to avoid materializing a massive token list.

Sanity cases:

```text
"some text that i'll pre-tokenize" matches PDF regex example
"Doc1<|endoftext|>Doc2" never merges across document boundary
repeated pre-token count accumulates frequency
```

Solution template:

```python
def split_by_special(text: str, special_tokens: list[str]):
    """
    Input:
        raw text and special tokens
    Output:
        iterator of (is_special, segment)
    """
    ...

def pretoken_counts(text: str, special_tokens: list[str] | None = None) -> dict:
    """
    Input:
        raw text
    Output:
        Counter mapping tuple[bytes, ...] to frequency
    """
    counts = {}
    for is_special, part in split_by_special(text, special_tokens or []):
        if is_special:
            ...                 # skip statistics
        else:
            for token in ...:    # regex matches
                pieces = ...     # tuple(bytes([b]) for b in token.encode("utf-8"))
                counts[pieces] = counts.get(pieces, 0) + 1
    return counts
```

</details>

<details class="solution">
<summary>Reference Solution</summary>

The core implementation can be divided into three layers: first, split boundaries by special tokens; second, run the GPT-2 regex on normal segments; finally, convert each pre-token into a byte tuple and accumulate frequencies.

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

A common pitfall is that special tokens cannot contribute to pair statistics, nor should they allow normal text on either side to merge across the boundary. Here, representing tokens as `tuple[bytes, ...]` allows subsequent BPE merges to directly combine adjacent `bytes` into longer `bytes`.

</details>

## Debug Checklist

<details class="exercise">
<summary><span class="q-label">Debug</span> <span class="q-text">Expand checklist</span></summary>

- The fundamental unit of a byte-level tokenizer is `bytes`, not Python `str`.
- A single byte is also represented as a `bytes` object, e.g., `b'a'`.
- Special tokens act as boundaries during training and as whole tokens during encoding.

</details>

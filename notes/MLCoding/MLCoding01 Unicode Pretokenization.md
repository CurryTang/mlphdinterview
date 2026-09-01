# ML Coding · 从零实现 LLM

这一章把原来的八个 ML Coding 小节合并成一条完整路线：先把原始文本变成 token，再把 token 喂进 Transformer，再把训练、采样和实验闭环接起来。每个练习都对应一块真实系统部件，所以阅读顺序也应该跟系统装配顺序一致，而不是把它们当成互不相关的模板题。

核心心智模型只有一条：

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

## 学习顺序

| 顺序 | 模块 | 你会搭出的部件 | 关键问题 |
| --- | --- | --- | --- |
| 1 | Unicode、UTF-8 与 pretokenization | byte-level tokenizer 的输入边界 | Python `str`、code point、UTF-8 bytes 到底差在哪 |
| 2 | BPE 训练 | `vocab` 与 `merges` | 什么 pair 值得合并，怎么保证统计和边界都正确 |
| 3 | Tokenizer runtime 与数据导出 | `encode` / `decode` / token array | 训练好的 merges 怎么稳定落地到推理与训练数据 |
| 4 | Tensor modules | Embedding、RMSNorm、SwiGLU、RoPE | Transformer 的基础张量模块怎么按 shape 拼起来 |
| 5 | Attention 与 Transformer LM | 可训练的语言模型前向图 | 因果注意力、残差流和 logits 怎么连起来 |
| 6 | 训练组件 | loss、AdamW、LR schedule、grad clip | 为什么训练会稳定或不稳定 |
| 7 | 训练循环与生成 | dataloader、checkpoint、decoding | 一个能跑起来的训练脚本最少需要什么 |
| 8 | 实验与 ablation | 可复现实验框架 | 哪些设计是真的在帮模型，哪些只是直觉 |

## 模块一：Unicode、UTF-8 与 Pretokenization

对应 CS336 Assignment 1：Section 2.1-2.4。

Tokenizer 的最小事实是：模型不看“字符”，模型只看整数；而 byte-level tokenizer 在映射成整数之前，先看的是 bytes。`str`、code point 和 UTF-8 bytes 不是同一层抽象，如果这三层没分清，后面 special token 边界、BPE merge、decode replacement character 都会一起出错。

### Lab · Unicode Probe

Unicode code point 是抽象字符编号，UTF-8 是它的可变长字节表示。`ord` / `chr` 工作在 code point 层，`encode("utf-8")` 才进入 byte 层；`repr` 和 `print` 也不是一回事，因为控制字符可能存在但不可见。

最值得先看的不是复杂字符，而是对比两个极端：`U+0041` 这样的 ASCII 字符只占一个 byte，`U+1F600` 这样的 emoji 需要四个 byte。这个差别就是后面“字符数”和“token 前 bytes 数”不相等的根源。

#### Quick Coding：`inspect_unicode_codepoint`

```python
def inspect_unicode_codepoint(cp: int) -> dict:
    ...
```

<details>
<summary>参考答案</summary>

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

`print(ch)` 面向显示效果，`repr(ch)` 面向调试。零宽字符、换行、控制字符这类问题，通常只有 `repr` 看得清。

</details>

### Lab · UTF-8 Encoding

UTF-8、UTF-16、UTF-32 的差异不是“谁更先进”，而是谁在当前文本分布上更省、更稳、更方便和字节流对接。LLM tokenizer 几乎总是在 UTF-8 上工作，因为训练语料本来就是字节流，UTF-8 也不会像 UTF-16/32 那样在英文上额外付固定宽度成本。

一个简单观察足够说明问题：`"hello"` 的 char count 和 UTF-8 byte count 一样，但 `"こんにちは"` 和 emoji 的 byte count 明显更长。再往前一步，invalid byte sequence 还逼你显式选择 `strict`、`replace` 或 `ignore`，这正是后面 decode 语义必须固定的原因。

#### Quick Coding：`compare_encodings`

```python
def compare_encodings(text: str) -> list[dict]:
    ...

def decode_invalid(raw: bytes, encoding="utf-8") -> dict:
    ...
```

<details>
<summary>参考答案</summary>

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

结论应该写清楚：

- ASCII 在 UTF-8 下通常是 1 char = 1 byte。
- CJK 和 emoji 在 UTF-8 下是多 byte。
- UTF-16 / UTF-32 常有 BOM 或固定宽度开销。
- invalid byte sequence 不能被默默吞掉，必须明确选择解码策略。

</details>

### Exercise 1 · GPT-2 Style Pretokenizer

Pretokenization 的作用不是“先分词再做 BPE”这么简单。它真正做的是限制 merge 的作用域，让 BPE 只在一个局部片段里学习高频 byte pattern，而不是跨句号、空格、special token 或文档边界随意拼接。没有这层边界，训练出来的 longest token 很容易直接暴露 bug。

这一题里最重要的约束有四个：

| 约束 | 为什么重要 |
| --- | --- |
| special token 是 hard boundary | `<|endoftext|>` 之类的控制符必须保持整体 |
| special token 不进入统计 | 否则 merge 会把控制 token 的内部 bytes 学坏 |
| 不跨 pre-token boundary 统计 pair | BPE 的作用域是局部片段，不是整段字符串 |
| 尽量 iterator 风格处理 | 大语料下先 materialize 全量 token list 会浪费内存 |

比如文本 `Doc1<|endoftext|>Doc2`，正确行为不是把 `<|endoftext|>` 左右的 bytes 放进同一个 merge 池，而是把它当成训练边界。另一个常见例子是 `"some text that i'll pre-tokenize"`，GPT-2 regex 会保留前导空格，并把缩写拆成 assignment 规定的 pattern，这决定了后面 BPE 能看到什么局部统计。

#### Quick Coding：`pretoken_counts`

```python
def split_by_special(text: str, special_tokens: list[str]):
    ...

def pretoken_counts(text: str, special_tokens: list[str] | None = None) -> dict:
    ...
```

<details>
<summary>参考答案</summary>

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

这里把每个 pre-token 表示成 `tuple[bytes, ...]`，后面的 BPE merge 才能直接把相邻 `bytes` 拼成更长的 `bytes`。

</details>

#### 本模块易错点

- byte-level tokenizer 的基础单位是 `bytes`，不是 Python `str`。
- 单个 byte 也必须是 `bytes` object，例如 `b"a"`。
- special token 在训练中是边界，在 encode 中是整体 token。
- `len(text)` 和 `len(text.encode("utf-8"))` 不是同一个量。

## 模块二：BPE 训练

对应 CS336 Assignment 1：Section 2.4-2.5。

BPE 训练在做一件很具体的事：从 byte 序列里反复找最值得合并的相邻 pair，把最常见的局部模式变成更长的 token。它不理解语义，也不“知道单词”，它只是在一个受限边界内压缩高频局部统计。

最容易被忽略的两点是 weighted frequency 和 deterministic tie-breaking。一个 pre-token 如果出现 1000 次，它里面的每个 pair 都要被算 1000 次；两个 pair 频率相同时，assignment 要求选 lexicographically greater pair，不能让结果依赖 `Counter` 插入顺序。

### Exercise 1 · Toy BPE Merge Simulator

先在小语料上手推 merge loop，能帮你看清“统计 pair”与“非重叠替换”是两步不同的逻辑。比如：

| round | pre-token counts | top pair 直觉 |
| --- | --- | --- |
| 0 | `(l, o, w) x2`, `(l, o, w, e, r) x1` | `(o, w)` 和 `(l, o)` 频率都高，tie-breaking 会决定先合谁 |
| 1 | 把 winner 做 non-overlapping merge | 只替换相邻且不重叠的出现位置 |
| 2 | 重新统计新的 pair | 旧 pair 频率会因为 token 边界变化而消失 |

这里的 worked example 比结论更重要：BPE 不是把所有出现过的 pair 一起改写，而是每轮只选一个 winner，再重建新的局部表示。

#### Quick Coding：`run_bpe_merges`

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
<summary>参考答案</summary>

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

不要用 `Counter.most_common(1)` 做 winner 选择。它的 tie-breaking 依赖插入顺序，不满足 assignment 的确定性要求。

</details>

### Exercise 2 · Full BPE Trainer

完整 trainer 把上一题的局部逻辑放进一个全流程里：先准备 0..255 的 byte vocabulary，再加 special tokens，再根据语料反复找 winner pair，直到 `vocab_size` 满。这里的 `vocab_size` 不是“merge 多少次”，而是总词表大小，必须把 byte vocab、special tokens 和 merge 新 token 全部算进去。

最关键的边界条件还是上一模块的那几个：special token 是 hard boundary，不参与 pair statistics，也不允许跨边界 merge。如果这一层写错，后面训练出的 longest token 通常会直接跨文档。

#### Quick Coding：`train_bpe`

```python
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    ...
```

<details>
<summary>参考答案</summary>

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

`build_pretoken_counts` 需要复用前一模块的规则，不能悄悄变成另一套 pretokenization 语义。

</details>

### Exercise 3 · BPE Performance Pass

朴素 trainer 的复杂度瓶颈非常直接：每轮都重新扫描全语料、重跑 pair recount。TinyStories 这种数据规模已经能把这种写法拖垮。优化方向也因此很朴素：并行 pretokenization、缓存倒排索引、每轮只更新受上次 merge 影响的 pre-token。

这类优化题的重点不是“写出某个神奇数据结构”，而是能明确说出什么状态可以增量更新，什么必须重算。这里最常用的是 `pair -> set[pretoken]` 的倒排索引。

#### Quick Coding：`build_pair_index`

```python
def build_pair_index(pretoken_counts):
    ...

def update_after_merge(pretoken_counts, pair_counts, pair_to_pretokens, winner):
    ...
```

<details>
<summary>参考答案</summary>

优化路线应该写清楚：

```text
1. 先把 raw corpus 切成 document chunks，special token 只作为边界。
2. 并行 pretokenization，得到 Counter[tuple[bytes, ...]]。
3. 建 pair -> set[pretoken] 的倒排索引。
4. 每轮选 winner pair。
5. 只更新包含 winner pair 的 pre-token。
6. 对这些 pre-token 的旧 pairs 做 decrement，新 pairs 做 increment。
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

验收时至少报告 wall-clock、peak memory 和每一步优化后的 speedup，不要只给一个总时间。

</details>

### Experiment · Train TinyStories Tokenizer

这个实验回答的问题不是“10K tokenizer 能不能训出来”，而是“它学到的长 token 是否和数据域一致”。TinyStories 的语言分布很窄，所以一个合理的词表会把高频英文词、带前导空格的词、儿童故事里常见名字和后缀合并成较长 token。

配置固定为：

| 项目 | 值 |
| --- | --- |
| dataset | TinyStories |
| vocab_size | 10,000 |
| special token | `<|endoftext|>` |
| 产物 | `vocab.json`、`merges.txt`、profile report |

要重点检查的不是单一 loss，而是 tokenizer 产物是否“像这个域”：

- `<|endoftext|>` 必须是单独 token。
- `num_merges = 10000 - 256 - 1 = 9743` 这个账要对。
- 最长 token 应该更像常见词和名字，而不是跨文档乱码。
- 如果 longest token 看起来像把多篇文档拼到一起，通常是 special-token boundary 写错了。

### Experiment · Train OpenWebText Tokenizer

同样的算法搬到 OpenWebText，问题就变成了分布迁移。OWT 的 vocabulary diversity 更高，长尾更重，URL、HTML、代码片段、符号串和多语种内容都会占掉一部分 merge 预算，所以 32K tokenizer 的“样子”应该和 TinyStories 很不一样。

建议固定对比矩阵：

| data | tokenizer | 主要指标 |
| --- | --- | --- |
| TinyStories sample | TinyStories 10K | bytes/token |
| TinyStories sample | OWT 32K | bytes/token |
| OWT sample | TinyStories 10K | bytes/token |
| OWT sample | OWT 32K | bytes/token |

一份合格报告至少要解释三件事：

- OWT tokenizer 为什么在 OWT 上 compression 更好。
- TinyStories tokenizer 为什么更“干净”，但迁移到 OWT 会退化。
- 最长 token、最常见 token 和 encode throughput 如何反映数据域差异。

#### 本模块易错点

- special token 被拆开后参与 merge。
- merge 跨 pre-token boundary。
- pair tie-breaking 不 deterministic。
- 单个 byte 用 `int` 而不是 `bytes` 表示，导致 vocab 类型错位。
- 每轮完整重扫全语料，复杂度退化到无法处理真实数据。

## 模块三：Tokenizer Runtime 与数据导出

对应 CS336 Assignment 1：Section 2.6-2.7。

训练出 `vocab` 和 `merges` 只是完成了一半工作。真正可用的 tokenizer runtime 需要把“训练时的 merge 顺序和边界语义”原样搬到 encode/decode 流程里，并保证它能落地到大语料导出、流式处理和 benchmark。

### Exercise 1 · Tokenizer Class

Runtime 的核心不是实现一个类，而是固定三件语义：

| 语义 | 正确做法 |
| --- | --- |
| encode 时 special tokens 怎么处理 | 先保护边界，再对普通片段 pre-tokenize |
| merges 怎么应用 | 按训练时创建顺序，也就是 merge rank 最低者优先 |
| decode 怎么恢复文本 | 先拼 token bytes，再整体做 UTF-8 decode |

最后一点尤其重要。逐 token decode 会把跨 token 的多 byte 字符拆坏；正确做法是先 `b"".join(...)`，再统一 `errors="replace"`。

#### Quick Coding：`Tokenizer`

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
<summary>参考答案</summary>

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

检查时至少确认：

- `decode(encode(text)) == text` 对合法 UTF-8 文本成立。
- special token 不会被 regex 拆碎。
- merge 只在单个 pre-token 内发生。
- malformed bytes 用 replacement character，而不是抛异常。

</details>

### Trace Lab · BPE Encoding Trace

BPE debug 最有效的方法不是看最终 ids，而是打印每一轮 merge 后的 pieces。比如 `"the cat ate"` 必须先被看成 `["the", " cat", " ate"]` 三个 pre-token；如果某条 merge 规则把 `"e"` 和后面的空格拼起来，边界已经错了。

这类 trace helper 的价值在于暴露中间状态。最终 ids 即使只差一个 merge rank，也可能完全看不出问题；打印 `start -> step 1 -> step 2` 的 pieces 则能立刻看出 winner pair 是否选错。

#### Quick Coding：`trace_bpe_token`

```python
def trace_bpe_token(token: str, tokenizer: Tokenizer) -> list[int]:
    ...
```

<details>
<summary>参考答案</summary>

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

最常见 bug 是把整段文本拼成一个 byte 序列后统一 merge，结果跨空格或跨 special token 合并。

</details>

### Exercise 2 · Streaming Encode

流式 encode 的难点不是 `yield` 语法，而是 safe boundary。只要 chunk 边界改变 pre-tokenization 或 merge 作用域，流式结果就会和整段 encode 不一致。`"intern" + "ational"` 这种随意切块就是经典反例。

因此，最安全的接口不是“任意字符串 chunk”，而是“文档边界或 special-token 边界已经安全切好的 iterable”。如果做不到这一点，就必须保留 overlap buffer，只在确认边界之前产出 token。

#### Quick Coding：`encode_iterable`

```python
def encode_iterable(self, iterable):
    ...
```

<details>
<summary>参考答案</summary>

```python
def encode_iterable(self, iterable):
    for chunk in iterable:
        yield from self.encode(chunk)
```

这个版本正确的前提是 `chunk` 本身就是 tokenizer-safe boundary，例如每个 chunk 是一篇 document，或者上游已经按 `<|endoftext|>` 分开。

错误示例是：

```python
def encode_iterable(iterable):
    return self.encode("".join(iterable))
```

这会直接 materialize 全文件，也失去了 streaming 的意义。

</details>

### Experiment · Compression Ratio

Tokenizer 不是只看“能不能 encode”，还要看它在某个分布上压得好不好、跑得快不快。这里最常用的三个指标是 `bytes/token`、`tokens/s` 和 `bytes/s`。前者近似反映压缩率，后两者反映运行时吞吐。

解释这类结果时不要脱离数据分布：

| 现象 | 更合理的解释 |
| --- | --- |
| TinyStories tokenizer 在 TinyStories 上 `bytes/token` 高 | 训练域匹配，常见词被 merge 成更长 token |
| OWT tokenizer 在 OWT 上更稳 | 网页噪声、多语种和符号串被更大的词表覆盖 |
| 跨域 `bytes/token` 变差 | merges 学到的是训练语料的局部统计 |
| `tokens/s` 不稳定 | regex pretokenization、merge 数据结构和 Python overhead 都在起作用 |

如果要估 825GB 语料的大致 tokenization 耗时，可以直接用：

```python
seconds = 825 * 1024**3 / report["bytes_per_second"]
hours = seconds / 3600
```

### Exercise 3 · Token ID Serialization

训练前把文本预编码成 token id array 的原因很现实：不想每个 training step 都重复跑 tokenizer。对于 `vocab_size <= 65536` 的课程设定，`uint16` 是合适的，因为 token id 非负，而且比 `int32` 省一半存储。

真正该检查的是 dtype 是否和词表规模匹配。如果词表已经 100K，还继续写 `uint16`，数组会静默截断或溢出，问题通常直到训练时才暴露。

#### Quick Coding：`encode_to_array`

```python
def encode_to_array(tokenizer: Tokenizer, texts, out_path: str, dtype="uint16") -> dict:
    ...
```

<details>
<summary>参考答案</summary>

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

数据很大时更合理的写法是分 shard 保存，并在 dataloader 里用 `np.load(path, mmap_mode="r")`。

</details>

#### 本模块易错点

- decode 时逐 token decode，而不是先拼 bytes 再 decode。
- special token 没有补进 vocab。
- `encode_iterable` 悄悄把全文件 materialize 到内存里。
- 随意切 chunk，结果改变 tokenization。

## 模块四：Tensor Modules

对应 CS336 Assignment 1：Section 3.2-3.4.3。

从 tokenizer 走到模型实现，中间最大的转折是 shape discipline。Transformer 大多数 bug 不是“公式错”，而是 batch 维、sequence 维、head 维和 feature 维摆错了。这个模块的练习都在训练一件事：把最后一维当 feature，前面各维都当 batch-like dims。

### Warmup · Tensor Shape Gym

写任何模块前，先把几个最常用的 shape 变换练熟，会节省很多无效调试时间：

| 操作 | 输入 | 输出 |
| --- | --- | --- |
| Linear | `(..., d_in)` | `(..., d_out)` |
| split heads | `(B, T, D)` | `(B, H, T, Dh)` |
| merge heads | `(B, H, T, Dh)` | `(B, T, D)` |
| RMSNorm | `(..., D)` | `(..., D)` |
| RoPE | `(..., T, Dh)` | `(..., T, Dh)` |

如果这些 toy shape 都讲不清，后面的 attention 和 RoPE 基本不可能一次写对。

#### Quick Coding：`shape_gym`

```python
def shape_gym():
    ...
```

<details>
<summary>参考答案</summary>

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

`Linear` 看起来最基础，但它固定了后面几乎所有模块的 shape 约定。这里 weight 必须是 `(out_features, in_features)`，这样 `y[..., o] = sum_i x[..., i] * weight[o, i]` 才和 PyTorch 线性层保持一致。

因为要支持 arbitrary leading dims，所以最自然的写法不是手工 `matmul` 展平，而是直接用 einsum 把最后一维当 feature，前面维度全透传。

#### Quick Coding：`Linear`

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        ...

    def forward(self, x):
        ...
```

<details>
<summary>参考答案</summary>

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

Embedding 本质上是查表，而不是线性投影。token ids 是整数索引，输出是这些索引对应的行向量。只要你把 weight shape 写反，后面所有 `(B, T, D)` 的假设都会崩掉。

#### Quick Coding：`Embedding`

```python
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        ...

    def forward(self, token_ids):
        ...
```

<details>
<summary>参考答案</summary>

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

RMSNorm 的作用是控制 residual stream 的尺度，而不是像 LayerNorm 那样减均值。它只对最后一维做 root-mean-square normalization，所以对 Transformer 来说，它更像一个“稳定输入分布”的装置。

这里真正的工程细节是 upcast。BF16/FP16 在做平方和均值时精度容易不稳，hidden dim 一大更明显，所以统计量通常先用 FP32 算，再 cast 回原 dtype。

#### Quick Coding：`RMSNorm`

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        ...

    def forward(self, x):
        ...
```

<details>
<summary>参考答案</summary>

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

SwiGLU FFN 的直觉是：一条投影给内容，一条投影给 gate，再用 `SiLU(gate) * up` 做逐通道调制。它和普通两层 FFN 的差别不是“多一个激活函数”，而是多了一条 gating path，所以 hidden dim 要相应缩小到大约 `8/3 * d_model`，参数量才和传统 `4 * d_model` FFN 接近。

#### Quick Coding：`SwiGLU`

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        ...

    def forward(self, x):
        ...
```

<details>
<summary>参考答案</summary>

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

RoPE 的作用不是“加一个位置向量”，而是把 Q/K 的每两个维度当成二维平面，按位置做旋转。这样 attention score 不直接依赖绝对位置 embedding，而是更自然地编码相对位移关系。

核心公式很短：

```text
[x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
```

但两个实现细节必须记住：

- `cos` / `sin` 可以按 `max_seq_len` 和 dim pair 预先缓存。
- RoPE 只作用在 Q/K 上，不作用在 V 上。

#### Quick Coding：`RotaryPositionalEmbedding`

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        ...

    def forward(self, x, token_positions):
        ...
```

<details>
<summary>参考答案</summary>

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

RoPE 的一个好 sanity check 是旋转前后每对维度的 L2 norm 应该保持不变。

</details>

#### 本模块易错点

- 所有 module 都要支持 `device` / `dtype`。
- 先用小 shape 打印中间张量，再跑完整模型。
- sequence 维和 head 维要显式标注，别靠猜。

## 模块五：Attention 与 Transformer LM

对应 CS336 Assignment 1：Section 3.4.4-3.5。

这一模块把前面的零件装配成一个语言模型。真正的主线不是“实现 attention”，而是理解 residual stream：token embedding 进入模型后，所有 block 都在同一个 `(B, T, D)` 通道里做读写，attention 和 MLP 只是给这个 residual stream 提供两类更新。

### Exercise 1 · Stable Softmax

Softmax 公式人人都知道，训练里容易出问题的只有一件事：大 logit 先 `exp` 会直接 overflow。所以 stable softmax 的第一步永远是沿归一化维减最大值，再做指数和归一化。

#### Quick Coding：`softmax`

```python
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

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

Attention 本质上是三步：

1. 用 `QK^T / sqrt(d_k)` 算匹配分数。
2. 用 mask 删掉不允许看的位置。
3. 用 softmax 权重对 `V` 做加权和。

这里 `sqrt(d_k)` 的缩放不是装饰。没有它，`d_k` 一大，logits 方差也会跟着涨，softmax 很快变得过于尖锐。

#### Quick Coding：`scaled_dot_product_attention`

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    ...
```

<details>
<summary>参考答案</summary>

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

这里固定 `True` 表示可见、`False` 表示禁止看。mask 语义必须和后续 causal mask 保持一致。

</details>

### Exercise 3 · Causal MHA

Causal multi-head self-attention 的难点不是公式，而是 shape。输入 `(B, T, D)` 必须先投影成 Q/K/V，再 reshape 成 `(B, H, T, Dh)`，把 head 维当作 batch-like 维度处理。最后再 merge 回 `(B, T, D)`。

RoPE 如果启用，也是在这个阶段给 Q/K 加位置旋转。因果 mask 则只允许 `j <= i`，保证第 `i` 个位置永远看不到未来 token。

#### Quick Coding：`CausalMultiHeadSelfAttention`

```python
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None, device=None, dtype=None):
        ...

    def forward(self, x, token_positions=None):
        ...
```

<details>
<summary>参考答案</summary>

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

课程里的 block 采用 pre-norm：

```text
y = x + MHA(RMSNorm(x))
out = y + FFN(RMSNorm(y))
```

它的意义不是公式更漂亮，而是梯度路径更直接，训练通常更稳。这里 residual stream 的 shape 必须始终保持 `(B, T, D)`；只要这个不变量破了，后续 block 和 LM head 都会一起坏。

#### Quick Coding：`TransformerBlock`

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope=None, device=None, dtype=None):
        ...

    def forward(self, x, token_positions=None):
        ...
```

<details>
<summary>参考答案</summary>

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

语言模型前向图的逻辑很简单：token ids 进 embedding，过 N 个 block，做 final norm，再映射到 vocab logits。模型内部不需要 softmax，因为训练时 cross entropy 直接吃 logits，生成时只用最后一个位置的 logits。

#### Quick Coding：`TransformerLM`

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, num_heads, d_ff, ...):
        ...

    def forward(self, token_ids):
        ...
```

<details>
<summary>参考答案</summary>

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

参数量和 FLOPs 估算的价值不在于替代 profiler，而在于让你先知道瓶颈会出在哪。对这个课程规模的 Transformer，一个非常有用的近似是：

| 组件 | 参数量近似 |
| --- | --- |
| Q/K/V/O | `4 * d_model^2` |
| SwiGLU FFN | `3 * d_model * d_ff` |
| 两个 norm | `2 * d_model` |
| token embedding / LM head | `vocab_size * d_model` |

FLOPs 里最值得盯的是 attention 的 `T^2` 项。context length 从 1024 拉到 16384，MLP 还是线性涨，attention 的 QK/PV 却会被平方项放大。

#### Quick Coding：`transformer_accounting`

```python
def transformer_accounting(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    ...
```

<details>
<summary>参考答案</summary>

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

粗略 FLOPs：

```text
Linear forward FLOPs ≈ 2 * tokens * in_dim * out_dim
Attention QK FLOPs ≈ 2 * B * H * T * T * Dh
Attention PV FLOPs ≈ 2 * B * H * T * T * Dh
MLP FLOPs ≈ 2 * B * T * D * Dff * 3
```

</details>

#### 本模块易错点

- causal mask 的 True/False 语义必须和 SDPA 一致。
- RoPE 只作用于 Q/K，不作用于 V。
- LM head 输出是 logits，不是 softmax 概率。

## 模块六：训练组件

对应 CS336 Assignment 1：Section 4。

模型前向图写对以后，训练是否稳定主要取决于四类部件：loss、optimizer、LR schedule 和 gradient control。这个模块的题都在回答同一个问题：为什么同一份前向图，换一个训练配方就会从稳定下降变成 loss spike。

### Exercise 1 · Cross-Entropy

语言模型训练用的是 logits 上的交叉熵，不是先 softmax 再喂一个概率分布进去。稳定写法一定走 log-sum-exp：

```text
CE = logsumexp(logits) - logits[target]
```

这个形式和 stable softmax 一样，本质上都在先减最大值避免数值溢出。

#### Quick Coding：`cross_entropy`

```python
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    ...
```

<details>
<summary>参考答案</summary>

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

学习率实验最适合先在一维二次函数上建立直觉。对 `f(x)=x^2`，梯度下降更新是：

```text
x_{t+1} = (1 - 2lr) x_t
```

这行式子已经说明了一切：`lr` 很小，慢慢收敛；`lr` 接近 0.5，最快；`lr` 大到一定程度后会振荡甚至发散。assignment 里给的 `1e1/1e2/1e3` 正是故意让你看到 instability。

#### Quick Coding：`run_sgd_lr`

```python
def run_sgd_lr(lr: float, steps: int = 10) -> list[float]:
    ...
```

<details>
<summary>参考答案</summary>

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

AdamW 和 “Adam + L2 regularization” 不是同一回事。它的关键在 decoupled weight decay：moment 统计只看梯度，权重衰减作为独立步骤处理，不能把 `weight_decay * p` 混进 `m` 和 `v` 的更新里。

另一个常见错误是 bias correction 的 timestep 从 0 开始。这会让第一步修正系数错掉，而这正是小模型前期最敏感的阶段。

#### Quick Coding：`AdamW.step`

```python
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        ...

    @torch.no_grad()
    def step(self, closure=None):
        ...
```

<details>
<summary>参考答案</summary>

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

训练显存账本里，optimizer state 经常比参数本体还大。一个参数量为 `P` 的模型，如果参数是 BF16、moments 是 FP32，那么仅 `m` 和 `v` 就要额外吃掉 `8P` bytes。

回答这类题时一定要把四种内存拆开讲：

- parameters
- gradients
- optimizer states
- activations

不把 activation 单独拿出来，batch size 和 context length 对显存的影响就完全看不见。

#### Quick Coding：`adamw_memory_accounting`

```python
def adamw_memory_accounting(num_params, param_bytes=2, grad_bytes=2, state_bytes=4):
    ...
```

<details>
<summary>参考答案</summary>

```text
parameters:       P * param_bytes
gradients:        P * grad_bytes
AdamW m:          P * state_bytes
AdamW v:          P * state_bytes
master weights:   可选
activations:      depends on batch_size * context_length * d_model * layers
```

训练 FLOPs 粗略可以记成：

```text
forward FLOPs = F
backward FLOPs ≈ 2F
optimizer step FLOPs ≈ O(P)
one train step ≈ 3F + optimizer
```

</details>

### Exercise 4 · Cosine LR with Warmup

warmup 和 cosine decay 解决的是两个不同问题。warmup 负责让训练从安全的小步长起跑，cosine decay 负责在后期逐步收敛到较小学习率。把二者写成分段函数以后，检查点其实很简单：`it=warmup_iters` 时必须到达 `max_lr`，`it=cosine_cycle_iters` 时必须落到 `min_lr`。

#### Quick Coding：`get_lr`

```python
def get_lr(it, max_lr, min_lr, warmup_iters, cosine_cycle_iters):
    ...
```

<details>
<summary>参考答案</summary>

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

`warmup_iters == 0` 时要单独处理，避免除零。

</details>

### Exercise 5 · Gradient Clipping

global grad norm clipping 的意义是把所有参数梯度视为一个长向量，再整体限制它的 L2 norm。这里最容易错的是把每个 tensor 单独 clip；那样得到的不是 global norm clipping，而是另一种局部启发式。

#### Quick Coding：`clip_grad_norm`

```python
def clip_grad_norm(parameters, max_l2_norm, eps=1e-6):
    ...
```

<details>
<summary>参考答案</summary>

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

## 模块七：训练循环与生成

对应 CS336 Assignment 1：Section 5-6。

前面所有组件在这里第一次真正闭环。训练 loop 的本质不是一长串样板代码，而是把数据采样、前向、loss、反向、梯度控制、LR 更新、eval 和 checkpoint 放进一个可复现的顺序里。

### Exercise 1 · Next-Token Batch Sampler

语言模型数据加载的核心非常简单：从一维 token array 里随机截取长度为 `context_length + 1` 的窗口，前 `context_length` 个 token 作为 `x`，后 `context_length` 个 token 作为右移一位的 `y`。看起来朴素，但 start index 只要多一位少一位，target 就会越界或错位。

#### Quick Coding：`get_batch`

```python
def get_batch(dataset, batch_size: int, context_length: int, device):
    ...
```

<details>
<summary>参考答案</summary>

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

Checkpoint 的最低要求只有三项：`model.state_dict()`、`optimizer.state_dict()` 和 iteration。少任何一项都会让 resume 行为变形；尤其是少 optimizer state 时，AdamW moments 会丢失，曲线往往会在恢复点附近突然跳一下。

#### Quick Coding：`save_checkpoint`

```python
def save_checkpoint(model, optimizer, iteration: int, out):
    ...

def load_checkpoint(src, model, optimizer) -> int:
    ...
```

<details>
<summary>参考答案</summary>

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

训练脚本调试最好按最短闭环推进：

1. 先 overfit 一个固定 minibatch。
2. 再接真实 dataloader。
3. 再打开 validation eval。
4. 最后测 checkpoint resume。

这样做的原因很简单：每一步只新增一种系统复杂度，出错时更容易定位。直接一上来跑全量训练，loss 不降时通常不知道是数据、模型、loss 还是 optimizer 的问题。

#### Quick Coding：`train`

```python
def train(config):
    ...
```

<details>
<summary>参考答案</summary>

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

最常见的实现错误：

- 忘记 `optimizer.zero_grad()`。
- eval 时忘记 `torch.no_grad()`。
- `model.eval()` 后忘记切回 `model.train()`。
- logging 直接持有 loss tensor 而不 `.item()`。

</details>

### Exercise 4 · Autoregressive Decoder

生成循环每轮只做一件事：读取最后一个位置的 logits，采一个 next token，再把它接回上下文。temperature 和 top-p 改的是采样分布，不改模型本身；context 超长时保留最近窗口，是因为因果 LM 根本看不到更早历史。

#### Quick Coding：`generate`

```python
@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature=1.0, top_p=1.0, eos_token_id=None):
    ...
```

<details>
<summary>参考答案</summary>

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

top-p 之后必须重新 normalize；`temperature=0` 也应该显式走 greedy，而不是去做除零。

</details>

## 模块八：实验与 Ablation

对应 CS336 Assignment 1：Section 7。

这一模块的目标不是再写更多函数，而是建立“可复现实验”的工作方式。能训练一个模型不算结束；你还需要知道哪次 run 用了哪份代码、哪组超参、在哪个 token budget 下收敛，以及某个改动是否真的带来了收益。

### Experiment 1 · Experiment Logger

如果以后看到一条 val loss 曲线，却说不清它对应哪份代码、哪组配置和哪份 checkpoint，这次训练基本等于白跑。最小 logger 至少要保存 run name、git commit 或 config hash、train/val loss、tokens processed、wall-clock time、checkpoint path 和生成样例。

#### Quick Coding：`ExperimentLogger`

```python
class ExperimentLogger:
    def __init__(self, path: str, config: dict):
        ...

    def log(self, step: int, **metrics):
        ...
```

<details>
<summary>参考答案</summary>

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

LR sweep 要回答的是稳定边界在哪里，而不是“哪个点看起来最好”。因此实验里只能改 learning rate，其他变量都必须固定：模型、tokenizer、数据顺序或 seed、总 token budget、batch size 和 schedule 形状都不能乱动。

推荐用 log-scale：

```text
1e-4, 3e-4, 1e-3, 3e-3, 1e-2
```

报告时要把曲线形态解释出来，而不是只给最终表格：

- loss 几乎不降，说明 LR 太小或训练太短。
- loss 先降后爆，说明接近或越过稳定边界。
- train loss 降、val loss 不降，可能是过拟合或数据太小。
- 最佳 LR 随 batch size 漂移，说明 gradient noise scale 变了。

### Experiment 3 · Batch Size Sweep

batch size 实验要把 micro batch 和 effective batch 分开写。前者影响显存和 step time，后者还可能包含 gradient accumulation，对 optimizer step 频率有直接影响。

推荐实验表：

```text
batch_size | grad_accum | effective_batch | lr | tokens/sec | max_mem | best_val_loss
```

结论不能只看 step time。更好的比较标准是同样 wall-clock 下的 val loss，或者同样 tokens processed 下的 val loss。

### Experiment 4 · Generate TinyStories Samples

生成样例实验真正要观察的是采样参数和 checkpoint 质量的交互。TinyStories 适合看故事连贯性、角色一致性和句子完整性，不适合拿“事实正确性”当主指标。

固定 prompt 后，至少比较：

| 变量 | 作用 |
| --- | --- |
| checkpoint step | 模型学到了多少模式 |
| temperature | 控制分布尖锐程度 |
| top_p | 控制保留多少概率质量 |
| prompt | 控制条件分布的起点 |

常见解读：

- 重复短句：模型欠训练，或 temperature 太低。
- 语法乱：checkpoint 太早，或 temperature 太高。
- 流畅但单调：top-p 太小，或 prompt 太强。
- 很早 EOS：EOS 学得太强，或数据里短文本过多。

### Experiment 5 · Remove RMSNorm

这个 ablation 关心的是稳定性，不是单点最好 loss。baseline 应该是 pre-norm + previous best LR，然后对比“去掉 RMSNorm 但 LR 不变”和“去掉 RMSNorm 并把 LR 调低”。

要记录的不只是 train/val loss，还包括：

- grad norm
- activation norm
- divergence step

更合理的预期是：去掉 RMSNorm 后，同样 LR 下更容易出现 loss spike 或 grad norm 放大；降低 LR 可能能训，但收敛速度和最终 loss 常会变差。

### Experiment 6 · Post-Norm Transformer

pre-norm 和 post-norm 的比较要只改 block 结构，不改参数量规模。post-norm 常见实现是：

```python
class PostNormBlock(nn.Module):
    def forward(self, x, token_positions=None):
        x = self.ln1(x + self.attn(x, token_positions=token_positions))
        x = self.ln2(x + self.ffn(x))
        return x
```

更好的实验写法是画两张图：

- same LR 对比
- retuned best LR 对比

这样才能区分“结构本身不稳”和“只是需要更低 LR”。

### Experiment 7 · NoPE vs RoPE

这个问题在问：没有显式位置编码时，模型能不能仅靠 causal mask 学到顺序信息。答案通常不是简单的“能/不能”，而是“短 context、小数据上可能还能降，但长 context 和位置关系复杂的任务上会明显变差”。

报告里不要只看 final loss，至少再加两样：

- 生成样例
- context length sensitivity

因为 NoPE 的很多退化恰好体现在重复、顺序混乱和长距离一致性下降上。

### Experiment 8 · SwiGLU vs SiLU FFN

这个对比最重要的前提是参数量近似匹配：

```text
SwiGLU: 3 * d_model * d_ff_swiglu
SiLU FFN: 2 * d_model * d_ff_silu
```

所以常见设定才会取：

```text
d_ff_silu ≈ 4 * d_model
d_ff_swiglu ≈ 8/3 * d_model
```

如果不先配平参数量，再说 “SwiGLU 更好” 没有可比性。输出里至少要同时给参数量、tokens/sec、best val loss 和 run variance。

### Experiment 9 · OpenWebText Run

把同一个小模型从 TinyStories 搬到 OWT，最容易犯的错是直接拿 loss 横比。不同数据分布的 entropy 不同，所以更合理的问题是：

- 同样训练 budget 下，OWT 曲线是否稳定下降。
- 生成结果是否更通用，还是只是多了网页碎片和噪声。
- tokenizer 的 `bytes/token` 是否显著变差。
- 同样 compute 下，模型是否学到了更泛化的语言模式。

OWT 更难压缩、更长尾、更嘈杂，所以 loss 更高本身并不等于模型更差。

### Experiment 10 · Leaderboard-Style Modification

这一题最像小型研究实验。正确写法是：

1. 先写假设。
2. 只改一个主要变量。
3. 在相同 token budget 和 eval protocol 下比较 baseline 与 modified。
4. 给证据，不给口号。

几个常见方向：

| 修改 | 假设 | 风险 |
| --- | --- | --- |
| weight tying | 减参数并带一点正则化 | 输出层表达受限 |
| retune LR schedule | 原 schedule 不是最优 | 需要更多实验预算 |
| better init | 改善前期稳定性 | 可能只影响前几百步 |
| fused kernels / compile | 提高 tokens/sec | 不一定提升最终 loss |
| tokenizer change | 更好压缩数据 | 会改变训练分布，可比性下降 |

以 weight tying 为例：

```python
model.lm_head.weight = model.token_embeddings.weight
```

即使改动没有提升，也可以是好答案。关键是实验设计干净，结论和数据一致。

## 最后检查：端到端 Debug Checklist

### 1. Tokenizer 链路

- `str`、code point、UTF-8 bytes 三层是否分清。
- special token 是否在训练时当边界、在 encode 时当整体 token。
- BPE merge 是否只发生在单个 pre-token 内。
- decode 是否先拼 bytes 再整体 UTF-8 decode。

### 2. 模型前向链路

- 所有 shape 是否沿着 `(B, T, D)` 或 `(B, H, T, Dh)` 保持一致。
- RoPE 是否只作用于 Q/K。
- causal mask 的语义是否和 SDPA 一致。
- LM head 是否输出 logits，而不是概率。

### 3. 训练链路

- `y` 是否真的是 `x` 的右移一位 target。
- cross entropy 是否直接吃 logits。
- AdamW 是否用了 decoupled weight decay 和正确的 bias correction。
- gradient clipping 是否按 global norm 做统一缩放。
- eval 是否放在 `torch.no_grad()` 里。

### 4. 实验链路

- 每个 run 是否记录了 config、commit、loss、tokens、wall-clock 和 checkpoint。
- ablation 是否只改一个主要变量。
- 比较是否在相同 token budget 或 wall-clock 下进行。
- 结论是否来自曲线、样例和日志，而不是先入为主的偏好。

如果这四条链都能独立检查，你就已经有了一套能从 raw text 走到可复现实验的最小 LLM 实现工作流。

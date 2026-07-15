# Sliding Window

## 这个专题解决什么问题

滑动窗口专门处理**连续子数组或连续子串**。它维护一个区间：

$$
[left,right]
$$

并且只对窗口两端做增量更新：

```text
right 右移：加入一个新元素
left 右移：删除一个旧元素
```

它和普通双指针的区别在于，重点不只是两个下标，而是窗口内部有一份持续维护的状态，例如：

```text
window sum
字符频次 frequency map
distinct count
满足要求的字符种类数 formed
窗口内最大值 / 最小值的单调队列
```

只要能在加入、删除一个边界元素时快速更新状态，就不需要为每个区间重新扫描内部元素。

---

## 一张图记住万能循环

下面用“窗口内不能有重复字符”演示通用过程。具体题目可以替换状态和合法条件，但四拍循环不变：

```sliding-window-demo
```

核心记忆：

```text
1. right 右扩，加入新元素
2. 更新窗口状态
3. while 条件触发：删除 nums[left]，left += 1
4. 在题目要求的正确时机更新答案
```

第 4 步最容易写错：

```text
求最长合法窗口：
  先缩到合法，再更新 max。

求最短满足窗口：
  一旦合法，在 while valid 内先更新 min，再继续缩。
```

---

## 万能模板

### 模板 A：最长合法窗口

目标通常是：满足某个约束的最长连续区间。

```python
def longest_window(items):
    left = 0
    state = initialize_state()
    answer = 0

    for right, item in enumerate(items):
        add(state, item)

        while window_is_invalid(state):
            remove(state, items[left])
            left += 1

        answer = max(answer, right - left + 1)

    return answer
```

不变量是：执行 `answer = max(...)` 时，窗口一定合法。

适合：

```text
最长无重复子串
最多包含 K 种字符的最长子串
最多翻转 K 个 0 后的最长连续 1
乘积小于 K 的连续子数组计数（状态和更新方式略有变化）
```

### 模板 B：最短满足窗口

目标通常是：包含足够信息的最短连续区间。

```python
def shortest_window(items):
    left = 0
    state = initialize_state()
    answer = float('inf')

    for right, item in enumerate(items):
        add(state, item)

        while window_is_valid(state):
            answer = min(answer, right - left + 1)
            remove(state, items[left])
            left += 1

    return 0 if answer == float('inf') else answer
```

不变量是：只要窗口仍然满足要求，就继续从左边压缩；压缩前的每个窗口都是候选答案。

适合：

```text
Minimum Window Substring
和至少为 target 的最短正数子数组
覆盖所有要求字符的最短区间
```

### 模板 C：固定长度窗口

窗口长度始终是 $k$：

```python
def fixed_window(items, k):
    left = 0
    state = initialize_state()
    answer = initialize_answer()

    for right, item in enumerate(items):
        add(state, item)

        if right - left + 1 > k:
            remove(state, items[left])
            left += 1

        if right - left + 1 == k:
            answer = update(answer, state)

    return answer
```

固定长度时通常只需要收缩一次，因为每轮 `right` 也只增加 1。

适合：

```text
长度为 k 的最大子数组和
长度为 k 的窗口平均值
检查所有长度为 k 的异位词窗口
```

---

## 状态到底应该维护什么

先把题目的条件翻译成可以增量更新的状态。

| 题目条件 | 常用状态 | add / remove 做什么 |
|---|---|---|
| 窗口和不超过 limit | `window_sum` | 加上或减去边界值 |
| 没有重复字符 | `count[char]` | 字符频次 `+1 / -1` |
| 最多 K 种字符 | `count` + `distinct` | 频次从 0 变 1 或从 1 变 0 时更新 distinct |
| 覆盖目标字符串 | `need/window` + `formed` | 某字符达到或失去要求频次时更新 formed |
| 窗口最大值 | 单调递减 deque | 加入时弹掉更小值，移出时删除过期下标 |

万能问题不是“left 和 right 怎么移动”，而是：

```text
窗口加入一个元素时，状态如何 O(1) 更新？
窗口删除一个元素时，状态如何 O(1) 恢复？
什么条件表示窗口合法或满足要求？
答案应该在收缩之前还是之后更新？
```

---

## 示例一：最长无重复子串

目标：求不含重复字符的最长连续子串长度。

状态：每个字符在窗口中的出现次数。

非法条件：刚加入的字符频次大于 1。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        count = {}
        left = 0
        answer = 0

        for right, char in enumerate(s):
            count[char] = count.get(char, 0) + 1

            while count[char] > 1:
                left_char = s[left]
                count[left_char] -= 1
                left += 1

            answer = max(answer, right - left + 1)

        return answer
```

为什么 `while` 只检查 `count[char] > 1`？

在加入新字符之前，旧窗口已经合法。唯一可能制造重复的就是刚加入的 `char`，所以只需要持续删除左端，直到它的频次回到 1。

---

## 示例二：和至少为 target 的最短子数组

假设数组元素都是正数。

状态：`window_sum`。

满足条件：`window_sum >= target`。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        left = 0
        window_sum = 0
        answer = float('inf')

        for right, value in enumerate(nums):
            window_sum += value

            while window_sum >= target:
                answer = min(answer, right - left + 1)
                window_sum -= nums[left]
                left += 1

        return 0 if answer == float('inf') else answer
```

这里必须在 `while window_sum >= target` 内更新答案，因为当前窗口满足要求后，还要继续缩，寻找以同一个 `right` 结尾的更短答案。

正数条件非常重要：

```text
right 右移 -> sum 只会增大
left 右移  -> sum 只会减小
```

如果允许负数，扩大窗口不一定让和变大，缩小窗口也不一定让和变小，普通滑窗的单调性会失效。这时通常需要前缀和、哈希表或单调队列。

---

## 示例三：Minimum Window Substring 的状态压缩

目标：找到 `s` 中覆盖 `t` 所有字符及频次的最短子串。

如果每次都比较整张频次表，检查窗口是否合法会很慢。可以维护：

```text
need[c]   = t 要求字符 c 出现多少次
window[c] = 当前窗口中 c 出现多少次
required  = need 中不同字符的数量
formed    = 已经满足所需频次的字符种类数
```

窗口合法当且仅当：

$$
formed=required.
$$

关键更新：

```python
# right 加入 char
window[char] += 1
if char in need and window[char] == need[char]:
    formed += 1

# left 删除 char
if char in need and window[char] == need[char]:
    formed -= 1
window[char] -= 1
```

注意删除时要在频次减 1 **之前**判断是否会从“刚好满足”变成“不满足”。

完整外层结构仍然没有变化：

```python
for right in range(len(s)):
    add(s[right])

    while formed == required:
        update_min_answer()
        remove(s[left])
        left += 1
```

这就是万能模板的价值：复杂题通常只是 `state`、`add`、`remove` 和合法条件更复杂，窗口骨架本身不变。

---

## 为什么嵌套 while 仍然是 O(n)

代码看起来有两层循环：

```python
for right in range(n):
    while need_shrink():
        left += 1
```

但 `right` 从 0 到 $n-1$ 只走 $n$ 次，`left` 在整个算法中也只从 0 向右走，最多走 $n$ 次。

总移动次数至多是：

$$
n+n=2n.
$$

因此，如果每次 `add`、`remove` 和合法性检查都是 $O(1)$，总时间就是：

$$
O(n).
$$

如果状态操作本身不是 $O(1)$，需要把它单独乘进去。例如每轮重新扫描整个频次表，就可能退化。

---

## 什么时候不能用普通滑动窗口

### 1. 问题不是连续区间

滑窗只能表示连续的 `[left, right]`。如果题目允许任意挑选元素或子序列，窗口通常不是正确模型。

### 2. 条件没有单调性

必须能判断：

```text
扩大窗口以后，条件朝哪个方向变化？
收缩窗口以后，能否逐步恢复合法或失去满足状态？
```

带负数的区间和是常见反例。

### 3. 状态无法增量维护

如果删除左端元素后无法快速恢复状态，可能需要 multiset、heap、单调队列或其他结构；这时仍可能是窗口，但不再是最简单的哈希表模板。

---

## 常见错误

### 1. 把 while 写成 if

加入一个新元素后，窗口可能需要连续删除多个左端元素才能恢复合法。只缩一次不够：

```python
while window_is_invalid():
    remove(items[left])
    left += 1
```

### 2. 先移动 left，再删除元素

错误顺序会删除新的左端，而不是离开窗口的旧元素。正确顺序是：

```python
remove(items[left])
left += 1
```

### 3. 窗口长度少写 1

闭区间 `[left, right]` 的长度是：

$$
right-left+1.
$$

### 4. 最长和最短模板更新时机混淆

```text
最长合法：缩完以后 update max。
最短满足：while valid 内先 update min，再缩。
```

### 5. 删除后没有清理状态

维护 distinct count 时，频次降到 0 必须同步减少 distinct。维护字典时是否删除零频 key 取决于合法性判断，但状态含义必须始终一致。

### 6. 无答案时返回值错误

最短窗口常用 `float('inf')` 初始化。循环结束后要转换成题目要求的 `0`、空串或 `-1`。

---

## 最终记忆版

```text
固定长度：
  右扩 -> 超长时左缩一次 -> 长度等于 k 时记录

最长合法：
  右扩 -> while 不合法左缩 -> 缩完记录 max

最短满足：
  右扩 -> while 合法：先记录 min，再左缩

统一不变量：
  window 永远是 items[left:right + 1]
  add / remove 必须和指针移动同步
  left 与 right 都只向右，所以总扫描是 O(n)
```

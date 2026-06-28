# Design Dynamic Array (Resizable Array)

## 面试目标

实现一个可扩容数组，重点是理解连续内存、容量 capacity、长度 size、扩容复制和摊还复杂度。

## 核心设计

- `size` 表示当前元素个数，`capacity` 表示底层数组容量。
- `get(i)` 和 `set(i, val)` 需要先检查 `0 <= i < size`。
- `pushback(val)` 如果 `size == capacity`，先扩容到 `capacity * 2`，再写入。
- `popback()` 只需要减少 `size`，通常不强制缩容。

## 复杂度

- 随机访问：`O(1)`
- 单次扩容：`O(n)`
- 连续 `pushback` 的摊还复杂度：`O(1)`

## 常见坑

- 扩容后忘记复制旧元素。
- 把 `capacity` 当成 `size` 使用。
- 空数组初始容量不能是 0，否则翻倍仍然是 0。

## 参考解法

<details class="solution">
<summary>展开解法</summary>

核心是维护 `data / size / capacity` 三个字段。插入时如果满了，就申请两倍容量的新数组，把旧元素复制过去，再写入新元素。

```text
pushback(x):
  if size == capacity:
    resize(max(1, capacity * 2))
  data[size] = x
  size += 1

resize(new_cap):
  new_data = array(new_cap)
  for i in [0, size):
    new_data[i] = data[i]
  data = new_data
  capacity = new_cap
```

`get`、`set` 只访问 `0 <= i < size` 的位置；`popback` 返回 `data[size - 1]` 后把 `size -= 1`。

</details>

<details class="solution">
<summary>Python 写法与记忆点</summary>

这题最容易混的是两个变量：

```text
n:
  当前有效元素个数，也就是逻辑长度

capacity:
  底层数组真实分配的空间
```

`capacity` 可以大于 `n`。数组里 `n` 之后的位置只是预留空间，不属于当前有效数组。

```python
class DynamicArray:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.array = [0 for _ in range(self.capacity)]
        self.n = 0

    def get(self, i: int) -> int:
        return self.array[i]

    def set(self, i: int, n: int) -> None:
        self.array[i] = n

    def pushback(self, n: int) -> None:
        if self.n == self.capacity:
            self.resize()
        self.array[self.n] = n
        self.n += 1

    def popback(self) -> int:
        self.n -= 1
        return self.array[self.n]

    def resize(self) -> None:
        self.capacity *= 2
        new_array = [0 for _ in range(self.capacity)]
        for i in range(self.n):
            new_array[i] = self.array[i]
        self.array = new_array

    def getSize(self) -> int:
        return self.n

    def getCapacity(self) -> int:
        return self.capacity
```

`popback` 是 lazy 的：不需要真的把底层数组里的值清成 `0`。只要 `self.n -= 1`，那个位置就已经不属于有效数组；下一次 `pushback` 会直接覆盖它。

一句话记：

```text
capacity 管物理空间，n 管逻辑长度；popback 只移动 n，不清数组。
```

</details>

```quiz
title: 练习 1
question: Dynamic Array 的 pushback 为什么通常说是摊还 O(1)？
answer: B
A. 每次 pushback 都不会复制元素
B. 扩容复制不频繁，成本被很多次普通插入摊平
C. 数组访问天然是 O(log n)
explanation: 扩容时会复制 O(n)，但扩容次数按倍增增长，平均到每次插入是 O(1)。
```

```quiz
title: 练习 2
question: size 和 capacity 的区别是什么？
answer: A
A. size 是已存元素数，capacity 是底层数组能容纳的数量
B. size 是数组最大值，capacity 是数组最小值
C. 二者总是完全相同
explanation: 动态数组经常保留未使用容量，避免每次插入都重新分配。
```

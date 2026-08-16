# Trees

## 前置：Design Binary Search Tree

### 面试目标

实现二叉搜索树，掌握插入、查找、删除和中序遍历的有序性质。

### 核心设计

- 对任意节点，左子树值更小，右子树值更大。
- 查找时按大小关系决定向左或向右。
- 删除节点分三类：叶子、单子树、双子树。
- 双子树删除常用右子树最小节点或左子树最大节点替换。

### 复杂度

- 平衡时查找/插入/删除：`O(log n)`
- 极端退化成链表时：`O(n)`
- 中序遍历：`O(n)`

### 常见坑

- 删除双子树节点后忘记删除替代节点原位置。
- 没有返回更新后的子树根。
- 忽略重复值策略。

### 参考解法

<details class="solution">
<summary>展开解法</summary>

插入和查找都按大小关系向左或向右走。删除时递归返回新的子树根，便于父节点接上更新后的子树。

```text
delete(root, key):
  if root is null: return null
  if key < root.val: root.left = delete(root.left, key)
  else if key > root.val: root.right = delete(root.right, key)
  else:
    if root.left is null: return root.right
    if root.right is null: return root.left
    succ = minNode(root.right)
    root.val = succ.val
    root.right = delete(root.right, succ.val)
  return root
```

双子树删除用中序后继替换，替换后还要从右子树里删除后继节点。

</details>

上面的 BST ADT 是本章的前置。后续内容统一处理四类信息：遍历顺序、递归返回值、路径状态和 BST 有序约束。15 道 Trees 题目都由这些固定结构组合而成。

## 学习顺序

题目来自 [NeetCode 150](https://neetcode.io/practice/practice/neetcode150) 的 Trees 模块。顺序先建立遍历和高度递归，再进入结构比较、BST 约束、树重建和路径聚合。

| 顺序 | 原题 | 要掌握的内容 |
|---:|---|---|
| 1 | [226. Invert Binary Tree](https://neetcode.io/problems/invert-a-binary-tree/question?list=neetcode150) | 前序或后序递归修改左右指针 |
| 2 | [104. Maximum Depth of Binary Tree](https://neetcode.io/problems/depth-of-binary-tree/question?list=neetcode150) | 高度递归的基本返回值 |
| 3 | [543. Diameter of Binary Tree](https://neetcode.io/problems/binary-tree-diameter/question?list=neetcode150) | 返回高度，同时更新全局直径 |
| 4 | [110. Balanced Binary Tree](https://neetcode.io/problems/balanced-binary-tree/question?list=neetcode150) | 高度哨兵与提前结束 |
| 5 | [100. Same Tree](https://neetcode.io/problems/same-binary-tree/question?list=neetcode150) | 两棵树逐节点比较 |
| 6 | [572. Subtree of Another Tree](https://neetcode.io/problems/subtree-of-a-binary-tree/question?list=neetcode150) | 在每个节点复用 Same Tree |
| 7 | [235. Lowest Common Ancestor of a BST](https://neetcode.io/problems/lowest-common-ancestor-in-binary-search-tree/question?list=neetcode150) | 利用 BST 值域单向下降 |
| 8 | [102. Binary Tree Level Order Traversal](https://neetcode.io/problems/level-order-traversal-of-binary-tree/question?list=neetcode150) | 队列与逐层快照 |
| 9 | [199. Binary Tree Right Side View](https://neetcode.io/problems/binary-tree-right-side-view/question?list=neetcode150) | 每层保留最后一个节点 |
| 10 | [1448. Count Good Nodes in Binary Tree](https://neetcode.io/problems/count-good-nodes-in-binary-tree/question?list=neetcode150) | 沿路径传递最大值 |
| 11 | [98. Validate Binary Search Tree](https://neetcode.io/problems/valid-binary-search-tree/question?list=neetcode150) | 向下传递合法开区间 |
| 12 | [230. Kth Smallest Element in a BST](https://neetcode.io/problems/kth-smallest-integer-in-bst/question?list=neetcode150) | 中序遍历与第 `k` 次访问 |
| 13 | [105. Construct Binary Tree from Preorder and Inorder Traversal](https://neetcode.io/problems/binary-tree-from-preorder-and-inorder-traversal/question?list=neetcode150) | 前序定根，中序分割 |
| 14 | [124. Binary Tree Maximum Path Sum](https://neetcode.io/problems/binary-tree-maximum-path-sum/question?list=neetcode150) | 向下路径收益与全局路径和 |
| 15 | [297. Serialize and Deserialize Binary Tree](https://neetcode.io/problems/serialize-and-deserialize-binary-tree/question?list=neetcode150) | 带空标记的前序编码 |

## 模块一：四种遍历模板

固定示例树如下。四种输出分别是前序 `1, 2, 4, 5, 3, 6`，中序 `4, 2, 5, 1, 3, 6`，后序 `4, 5, 2, 6, 3, 1`，层序 `1, 2, 3, 4, 5, 6`。

```text
        1
      /   \
     2     3
    / \     \
   4   5     6
```

```tree-traversal-demo
```

### 前序：root → left → right

递归版直接按访问顺序拼接结果。

```python
def preorder(root):
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []
```

迭代版先压右子节点，再压左子节点。栈的后进先出性质保证左子树先处理。

```python
def preorder_iterative(root):
    if not root:
        return []

    stack = [root]
    order = []
    while stack:
        node = stack.pop()
        order.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return order
```

### 中序：left → root → right

递归版先完整处理左子树。BST 的中序结果严格递增，前提是题目采用互异键值。

```python
def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []
```

迭代版反复压入左链。当前指针为空时弹栈、访问，再转向右子树。

```python
def inorder_iterative(root):
    stack, order = [], []
    current = root

    while stack or current:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        order.append(current.val)
        current = current.right

    return order
```

### 后序：left → right → root

递归版把根节点放在两个子树之后。

```python
def postorder(root):
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []
```

迭代版先生成修改前序 `root → right → left`，再整体反转。为了让右子节点先出栈，代码先压左子节点，再压右子节点。单栈加 `last_visited` 也可实现后序，但状态分支更多。

```python
def postorder_iterative(root):
    if not root:
        return []

    stack = [root]
    reverse_order = []
    while stack:
        node = stack.pop()
        reverse_order.append(node.val)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    return reverse_order[::-1]
```

### 层序：BFS 队列

队列按先进先出顺序处理节点。左、右子节点依次进入队尾，因此输出按层从左到右排列。

```python
from collections import deque


def level_order(root):
    if not root:
        return []

    queue = deque([root])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return order
```

四种遍历都访问每个节点一次，时间复杂度为 `O(n)`。递归 DFS 的调用栈和迭代 DFS 的显式栈都是 `O(h)`；BFS 队列最多保存一层节点，空间为 `O(w)`。

## 模块二：五个核心递归模式

### 1. 自底向上返回值 + 全局最优值

子节点先算出一个值，父节点用这个值算出当前节点的贡献，同时用一个独立变量记录扫描过程中出现的最优答案。整棵树的答案（`best`）和父节点需要的量（`value_for_parent`）往往不是同一个东西；混用这两个量会产生错误。这个结构既可以写成递归，也可以写成迭代后序遍历，选哪种取决于哪种写法在具体这道题上更容易说清楚，不需要每次都用同一种形式。

迭代版本用显式栈模拟调用栈，把每个节点的返回值缓存在字典里，等它的两个子节点都处理完再计算当前节点：

```python
def solve(root):
    value = {None: 0}
    best = 0
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if node is None:
            continue
        if processed:
            left = value[node.left]
            right = value[node.right]
            best = max(best, combine_for_answer(left, right, node))
            value[node] = value_for_parent(left, right, node)
        else:
            stack.append((node, True))
            stack.append((node.left, False))
            stack.append((node.right, False))
    return best
```

`processed=False` 表示先展开子节点；`processed=True` 表示两个子节点的缓存值已经确定，可以计算当前节点。Diameter 用这个迭代版本缓存高度，答案记录 `left_height + right_height`。

递归版本更直接，不需要额外的缓存字典，因为子节点的返回值就是对应递归调用的结果。Maximum Path Sum 用的是这个形式：递归返回单边向下收益，`self.max_sum` 记录任意两个分支拼接后的最大路径和。

Count Good Nodes 使用同一项“状态与答案聚合分开”的原则，用显式栈把路径最大值从父节点传给子节点，独立变量记录计数，属于该模式的自顶向下变体。

使用题目：Diameter of Binary Tree（迭代）、Count Good Nodes in Binary Tree（迭代）、Binary Tree Maximum Path Sum（递归）。

### 2. 结构比较递归

显式栈同时保存两个树中位置对应的节点。空节点组合和值相等条件共同决定结果。

```python
def same(a, b):
    stack = [(a, b)]
    while stack:
        x, y = stack.pop()
        if not x and not y:
            continue
        if not x or not y or x.val != y.val:
            return False
        stack.append((x.left, y.left))
        stack.append((x.right, y.right))
    return True
```

Same Tree 直接使用该模板。Subtree of Another Tree 在主树的每个节点调用 `same(node, subRoot)`，匹配失败后继续检查左右子树。

使用题目：Same Tree、Subtree of Another Tree。

### 3. BST 有序约束

BST 最重要的专用性质是中序遍历有序。它支持三种常见写法。

| 题目 | 使用方式 |
|---|---|
| Kth Smallest | 中序第 `k` 次访问就是第 `k` 小值，可以提前结束 |
| Validate BST | 每个节点携带合法开区间 `(low, high)`；左子树上界收紧为当前值，右子树下界收紧为当前值 |
| LCA of a BST | 两个目标值都小于当前值时向左，都大于当前值时向右，否则当前节点就是分叉点 |

只检查 `node.left.val < node.val < node.right.val` 会漏掉跨越多层的违规节点。合法区间必须从所有祖先传递下来。LCA 可以沿一条路径运行，时间复杂度为 `O(h)`，无需同时搜索两棵子树。

Validate BST 使用显式栈携带每个节点的祖先边界：

```python
import math


def valid(root):
    stack = [(root, -math.inf, math.inf)]
    while stack:
        node, low, high = stack.pop()
        if not node:
            continue
        if not low < node.val < high:
            return False
        stack.append((node.left, low, node.val))
        stack.append((node.right, node.val, high))
    return True
```

也可以用迭代中序遍历检查节点值是否严格递增，但区间栈更直接地展示了所有祖先共同施加的约束。

### 4. 遍历序列重建

前序序列依次给出新节点。栈保存当前右链中仍在等待右子节点的节点，`j` 指向中序序列中下一个待完成的节点。

```python
def build_tree(preorder, inorder):
    root = TreeNode(preorder[0])
    stack = [root]
    j = 0
    for i in range(1, len(preorder)):
        node = TreeNode(preorder[i])
        parent = None
        while stack and stack[-1].val == inorder[j]:
            parent = stack.pop()
            j += 1
        if parent:
            parent.right = node
        else:
            stack[-1].left = node
        stack.append(node)
    return root
```

下一个前序值与 `inorder[j]` 不同时，它是栈顶节点的左子节点。相同时，当前左子树已经完成；连续弹出与中序值匹配的节点后，新节点连接为最后一个弹出节点的右子节点。每个节点入栈、出栈各一次，时间和额外空间均为 `O(n)`。

Serialize and Deserialize 用的是带显式空标记的前序序列：前序先给出当前根，后续标记按“左子树、右子树”的固定顺序消费。这道题递归写法更直接，不需要额外的栈；空标记本身就告诉递归调用子树在哪结束。中序序列先出现左侧内容，缺少额外遍历时无法确定当前根的位置，因此不能单独完成无歧义反序列化，这也是为什么 Construct Binary Tree from Preorder and Inorder Traversal 必须用中序分割，而不能像这道题一样只靠前序加空标记。

使用题目：Construct Binary Tree from Preorder and Inorder Traversal（迭代）、Serialize and Deserialize Binary Tree（递归）。

### 5. 高度重复计算与哨兵修复

Balanced Binary Tree 的直接写法会在每个节点重新调用 `height()`。倾斜树上的同一批节点被重复访问，总时间达到 `O(n^2)`。

一次迭代后序遍历可以同时计算高度和判断平衡。缓存保存已处理子树的高度；发现失衡时直接返回 `False`。

```python
def is_balanced(root):
    height = {None: 0}
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if node is None:
            continue
        if processed:
            left_h = height[node.left]
            right_h = height[node.right]
            if abs(left_h - right_h) > 1:
                return False
            height[node] = 1 + max(left_h, right_h)
        else:
            stack.append((node, True))
            stack.append((node.left, False))
            stack.append((node.right, False))
    return True
```

递归版本需要用 `-1` 跨调用帧传递“子树已经失衡”的信号。迭代版本在发现失衡时可以直接从函数返回 `False`，因此不需要哨兵值。两种机制都实现了同一原则：答案确定后立即停止。递归版本的 `-1` 与二分查找的边界哨兵、Largest Rectangle 栈中的尾部哨兵仍属于同一类接口设计：特殊值携带控制信息并缩短后续处理。

## 模块三：自平衡树的基本概念

按升序插入普通 BST 时，每个节点可能只有右子节点，树高变为 `n`，查找、插入和删除都退化为 `O(n)`。自平衡树通过局部重排或多路节点约束树高。

### AVL Tree

AVL 对每个节点维护平衡因子：

$$
\text{balance}(node) = \text{height}(node.left) - \text{height}(node.right)
$$

任意节点都要求 `|balance| <= 1`。插入或删除导致失衡后，根据较重路径的两次方向选择旋转。

| 情况 | 较重路径 | 修复操作 |
|---|---|---|
| LL | 左 → 左 | 对失衡节点右旋 |
| RR | 右 → 右 | 对失衡节点左旋 |
| LR | 左 → 右 | 先对左子节点左旋，再对失衡节点右旋 |
| RL | 右 → 左 | 先对右子节点右旋，再对失衡节点左旋 |

LL 示例使用节点 `30, 20, 10, 25`。右旋后 `20` 成为新根，`30` 下移到右侧，原来的 `20.right = 25` 重新接到 `30.left`。中序顺序保持 `10, 20, 25, 30`。

```text
        30                 20
       /                  /  \
     20        ->        10   30
    /  \                    /
   10  25                  25
       T2                  T2
```

LR 示例使用 `30, 10, 20`。先围绕 `10` 左旋，把结构转换成 LL；再围绕 `30` 右旋。

```text
      30              30              20
     /               /               /  \
   10       ->      20      ->      10   30
     \             /
     20           10
```

```avl-rotation-demo
```

### Red-Black Tree

红黑树维护以下颜色约束：

1. 每个节点是红色或黑色。
2. 根节点和所有空叶节点为黑色。
3. 红色节点的子节点都为黑色。
4. 从任意节点到其后代空叶节点的每条路径包含相同数量的黑色节点。

这些约束禁止连续红节点，并固定每条路径的黑高。最长路径最多约为最短路径的两倍，因此树高保持 `O(log n)`。红黑树的平衡要求比 AVL 宽松，插入和删除通常需要较少旋转。C++ `std::map` 和 Java `TreeMap` 常由红黑树实现。

### B-Tree

B-Tree 的一个节点保存多个有序键和多个子指针。高分支因子显著降低树高，一个磁盘页可以容纳一个完整节点，因此一次页读取会带回多个分割键。数据库索引和文件系统使用这类结构减少磁盘或存储页访问次数。

### 对比

| 结构 | 平衡条件 | 插入 / 删除的再平衡成本 | 常见用途 |
|---|---|---|---|
| AVL | 每个节点左右高度差最多 1 | 旋转次数少且查询高度严格；删除可能沿祖先链继续修复 | 读操作密集的内存有序结构 |
| Red-Black | 颜色、红节点和黑高约束 | 常数次旋转配合重新着色；更新成本通常较低 | `std::map`、`TreeMap` 等通用有序映射 |
| B-Tree | 每个多路节点的键数保持在容量范围内，所有叶子同层 | 节点分裂、合并或向兄弟节点借键 | 数据库索引、文件系统、块存储 |

## 模块四：15 道题目的映射

### 1. Invert Binary Tree

这道题组合迭代 DFS 与指针交换。每个节点交换左右子树，再把非空子节点压栈。

| 项目 | 内容 |
|---|---|
| 组合模式 | 树遍历 + 局部指针修改 |
| 关键状态 | 当前节点的左右子指针 |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Invert Binary Tree

```python
def invertTree(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        stack = [root]
        while stack:
            node = stack.pop()
            node.left, node.right = node.right, node.left
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)
        return root
```

</details>

### 2. Maximum Depth of Binary Tree

层序 BFS 每处理完一层就把深度加一。队列为空时，累计的层数就是最大深度。

| 项目 | 内容 |
|---|---|
| 组合模式 | 层序遍历 + 层数累计 |
| 关键状态 | 当前队列长度表示本层节点数 |
| 时间 / 空间 | `O(n) / O(w)` |

#### Quick Coding：Maximum Depth of Binary Tree

```python
def maxDepth(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from collections import deque
from typing import Optional


class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        depth = 0
        queue = deque([root])
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return depth
```

</details>

### 3. Diameter of Binary Tree

迭代后序遍历先确定左右子树高度，再计算当前节点高度。独立变量记录每个节点处的 `left_height + right_height`，直径按边数计算。

| 项目 | 内容 |
|---|---|
| 组合模式 | 后序栈 + 高度缓存 + 全局最优值 |
| 缓存量 / 答案量 | 子树高度 / 最大直径 |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Diameter of Binary Tree

```python
def diameterOfBinaryTree(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        height = {None: 0}
        diameter = 0
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if node is None:
                continue
            if processed:
                left_h, right_h = height[node.left], height[node.right]
                diameter = max(diameter, left_h + right_h)
                height[node] = 1 + max(left_h, right_h)
            else:
                stack.append((node, True))
                stack.append((node.left, False))
                stack.append((node.right, False))
        return diameter
```

栈中的 `(node, processed)` 记录处理阶段。`processed=False` 表示先展开当前节点的子节点；`processed=True` 表示两个子节点的高度已经写入 `height`，此时可以计算当前节点高度。这是标准的迭代后序模拟，Maximum Path Sum 会复用同一结构。

</details>

### 4. Balanced Binary Tree

迭代后序遍历缓存每个子树的高度。任意节点的左右高度差超过 `1` 时直接返回 `False`。

| 项目 | 内容 |
|---|---|
| 组合模式 | 后序栈 + 高度缓存 + 提前结束 |
| 关键条件 | `abs(left - right) <= 1` |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Balanced Binary Tree

```python
def isBalanced(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        height = {None: 0}
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if node is None:
                continue
            if processed:
                left_h, right_h = height[node.left], height[node.right]
                if abs(left_h - right_h) > 1:
                    return False
                height[node] = 1 + max(left_h, right_h)
            else:
                stack.append((node, True))
                stack.append((node.left, False))
                stack.append((node.right, False))
        return True
```

</details>

### 5. Same Tree

这是结构比较的基本题。栈中的每个元素是一对结构位置相同的节点。两个空节点匹配；只有一个空节点或节点值不同都直接失败。

| 项目 | 内容 |
|---|---|
| 组合模式 | 节点对栈 + 结构比较 |
| 关键状态 | 同一位置的两个节点 |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Same Tree

```python
def isSameTree(p, q):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        stack = [(p, q)]
        while stack:
            a, b = stack.pop()
            if not a and not b:
                continue
            if not a or not b or a.val != b.val:
                return False
            stack.append((a.left, b.left))
            stack.append((a.right, b.right))
        return True
```

</details>

### 6. Subtree of Another Tree

这道题把迭代 Same Tree 作为子程序。当前节点匹配失败后，主栈继续在左右子树中寻找候选根。

| 项目 | 内容 |
|---|---|
| 组合模式 | 结构比较栈 + 候选根栈 |
| 关键操作 | `same(node, subRoot)` |
| 时间 / 空间 | 最坏 `O(mn) / O(h)` |

#### Quick Coding：Subtree of Another Tree

```python
def isSubtree(root, subRoot):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def same(a: Optional[TreeNode], b: Optional[TreeNode]) -> bool:
            stack = [(a, b)]
            while stack:
                x, y = stack.pop()
                if not x and not y:
                    continue
                if not x or not y or x.val != y.val:
                    return False
                stack.append((x.left, y.left))
                stack.append((x.right, y.right))
            return True

        if not subRoot:
            return True
        stack = [root]
        while stack:
            node = stack.pop()
            if not node:
                continue
            if same(node, subRoot):
                return True
            stack.append(node.left)
            stack.append(node.right)
        return False
```

</details>

### 7. Lowest Common Ancestor of a BST

BST 有序约束把搜索限制在一条根到叶路径。目标值位于当前值两侧时，当前节点就是分叉点。

| 项目 | 内容 |
|---|---|
| 组合模式 | BST 有序约束 |
| 关键分支 | 同左、同右、分叉 |
| 时间 / 空间 | `O(h) / O(1)` |

#### Quick Coding：Lowest Common Ancestor of a BST

```python
def lowestCommonAncestor(root, p, q):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def lowestCommonAncestor(
        self,
        root: TreeNode,
        p: TreeNode,
        q: TreeNode,
    ) -> Optional[TreeNode]:
        current = root
        low, high = sorted((p.val, q.val))

        while current:
            if high < current.val:
                current = current.left
            elif low > current.val:
                current = current.right
            else:
                return current
        return None
```

</details>

### 8. Binary Tree Level Order Traversal

基础 BFS 再增加一层循环。每轮先读取当前队列长度，该长度就是本层节点数。

| 项目 | 内容 |
|---|---|
| 组合模式 | 层序遍历 + 层大小快照 |
| 关键状态 | `level_size = len(queue)` |
| 时间 / 空间 | `O(n) / O(w)` |

#### Quick Coding：Binary Tree Level Order Traversal

```python
def levelOrder(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from collections import deque
from typing import List, Optional


class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        result = []
        queue = deque([root])
        while queue:
            level = []
            for _ in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result
```

</details>

### 9. Binary Tree Right Side View

这道题复用逐层 BFS。每层出队的最后一个节点就是从右侧可见的节点。

| 项目 | 内容 |
|---|---|
| 组合模式 | 层序遍历 + 每层最后一次访问 |
| 关键条件 | `i == level_size - 1` |
| 时间 / 空间 | `O(n) / O(w)` |

#### Quick Coding：Binary Tree Right Side View

```python
def rightSideView(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from collections import deque
from typing import List, Optional


class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        result = []
        queue = deque([root])
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                if i == level_size - 1:
                    result.append(node.val)
        return result
```

</details>

### 10. Count Good Nodes in Binary Tree

栈中的 `(node, max_seen)` 把路径最大值从根向下传递。当前值不小于祖先路径最大值时计数加一，然后更新子节点收到的最大值。

| 项目 | 内容 |
|---|---|
| 组合模式 | 路径状态 + 外层聚合计数 |
| 关键状态 | `max_seen` |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Count Good Nodes in Binary Tree

```python
def goodNodes(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        good = 0
        stack = [(root, root.val)]
        while stack:
            node, max_seen = stack.pop()
            if node.val >= max_seen:
                good += 1
            next_max = max(max_seen, node.val)
            if node.left:
                stack.append((node.left, next_max))
            if node.right:
                stack.append((node.right, next_max))
        return good
```

</details>

### 11. Validate Binary Search Tree

显式栈让每个节点携带所有祖先共同确定的开区间 `(low, high)`。使用开区间也明确排除了重复键值。

| 项目 | 内容 |
|---|---|
| 组合模式 | BST 有序约束 + 区间传递 |
| 关键条件 | `low < node.val < high` |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Validate Binary Search Tree

```python
def isValidBST(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
import math
from typing import Optional


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        stack = [(root, -math.inf, math.inf)]
        while stack:
            node, low, high = stack.pop()
            if not node:
                continue
            if not low < node.val < high:
                return False
            stack.append((node.left, low, node.val))
            stack.append((node.right, node.val, high))
        return True
```

也可以用迭代中序遍历检查节点值是否严格递增。这里保留区间栈，因为它直接展示了合法范围必须包含所有祖先的约束。

</details>

### 12. Kth Smallest Element in a BST

迭代中序遍历按升序弹出节点。第 `k` 次弹栈访问时直接返回当前值。

| 项目 | 内容 |
|---|---|
| 组合模式 | BST 有序约束 + 迭代中序 |
| 关键状态 | 剩余访问次数 `k` |
| 时间 / 空间 | `O(h + k) / O(h)` |

#### Quick Coding：Kth Smallest Element in a BST

```python
def kthSmallest(root, k):
    ...
```

<details>
<summary>参考答案</summary>

```python
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = []
        current = root

        while stack or current:
            while current:
                stack.append(current)
                current = current.left

            current = stack.pop()
            k -= 1
            if k == 0:
                return current.val
            current = current.right

        raise ValueError("k exceeds the number of nodes")
```

</details>

### 13. Construct Binary Tree from Preorder and Inorder Traversal

前序序列依次创建节点。栈保存当前右链中仍在等待右子节点的节点，`j` 与中序序列同步前进。

| 项目 | 内容 |
|---|---|
| 组合模式 | 遍历序列重建 |
| 关键状态 | 待完成节点栈 + 中序指针 |
| 时间 / 空间 | `O(n) / O(n)` |

#### Quick Coding：Construct Binary Tree from Preorder and Inorder Traversal

```python
def buildTree(preorder, inorder):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import List, Optional


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        root = TreeNode(preorder[0])
        stack = [root]
        j = 0
        for i in range(1, len(preorder)):
            node = TreeNode(preorder[i])
            parent = None
            while stack and stack[-1].val == inorder[j]:
                parent = stack.pop()
                j += 1
            if parent:
                parent.right = node
            else:
                stack[-1].left = node
            stack.append(node)
        return root
```

不变量是：`stack` 保存当前右链中还在等待右子节点的节点，`j` 指向中序序列中下一个待完成的位置。处理下一个前序值前，如果栈顶值与 `inorder[j]` 不同，新节点必须是栈顶的左子节点，表示仍在沿左链下降。如果相同，说明对应节点的左子树已经完成；连续弹出与 `inorder[j]` 匹配的节点并同步增加 `j`，新节点成为最后一个弹出节点的右子节点。没有节点弹出时，新节点仍连接为当前栈顶的左子节点。

示例 `preorder=[3,9,20,15,7]`、`inorder=[9,3,15,20,7]` 的执行过程如下：

1. 创建根节点 `3`：`stack=[3]`，`j=0`，下一个中序值为 `9`。
2. 处理 `9`：栈顶 `3 != 9`，连接 `3.left=9`，得到 `stack=[3,9]`。
3. 处理 `20`：栈顶 `9` 与中序值 `9` 匹配，弹出 `9`；随后 `3` 与中序值 `3` 匹配，再弹出 `3`。此时 `j=2`，连接 `3.right=20`，得到 `stack=[20]`。
4. 处理 `15`：栈顶 `20 != 15`，连接 `20.left=15`，得到 `stack=[20,15]`。
5. 处理 `7`：依次弹出与中序值匹配的 `15` 和 `20`，此时 `j=4`，连接 `20.right=7`。

最终得到 `3(9, 20(15,7))`。

</details>

下面的演示逐步执行这个例子，可以直接看到栈、`j` 指针和构建中的树如何一起变化。

```build-tree-demo
```

### 14. Binary Tree Maximum Path Sum

父节点只能继续一条向下分支，因此递归返回 `node.val + max(left_gain, right_gain)`。当前节点处的完整候选路径可以同时连接左右分支，写入 `self.max_sum`。负收益按 `0` 丢弃。

| 项目 | 内容 |
|---|---|
| 组合模式 | 自底向上返回值 + 全局最优值 |
| 返回量 / 答案量 | 单边向下收益 / 任意端点最大路径和 |
| 时间 / 空间 | `O(n) / O(h)` |

这道题用递归比迭代版本更直接：Python 里 `self.max_sum` 直接是可变状态，不需要 `nonlocal`，也不需要额外的高度缓存字典去跨越栈帧传递子节点的返回值。是否用迭代还是递归，取决于哪种写法在这道题上更容易说清楚，不是所有题目都天然适合同一种写法。Diameter 和 Balanced Binary Tree 用迭代后序遍历，是因为它们本身就在演示"用显式栈模拟调用栈"这个技巧；这里没有这个额外目的，直接写递归更清楚。

#### Quick Coding：Binary Tree Maximum Path Sum

```python
def maxPathSum(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = float('-inf')

        def dfs(node):
            if not node:
                return 0

            max_left = max(0, dfs(node.left))
            max_right = max(0, dfs(node.right))

            path_sum = node.val + max_left + max_right
            self.max_sum = max(self.max_sum, path_sum)

            return node.val + max(max_left, max_right)

        dfs(root)
        return self.max_sum
```

`dfs` 每次递归调用天然对应一层调用帧，`max_left`/`max_right` 就是子节点已经算好的返回值，不需要手动缓存。`self.max_sum` 在每次进入新节点时更新，函数返回后就是最终答案。

</details>

### 15. Serialize and Deserialize Binary Tree

前序序列记录节点值，并为每个空子节点写入一个空标记。这道题递归比迭代更清楚：`serialize` 只是一次前序遍历，`deserialize` 只是按同样的顺序消费 token，不需要额外的栈和 `fill_count` 记账；递归调用本身就在追踪"当前该填哪个子节点"。

| 项目 | 内容 |
|---|---|
| 组合模式 | 带空标记的前序重建 |
| 关键不变量 | 序列化和反序列化使用同一 token 顺序 |
| 时间 / 空间 | `O(n) / O(n)` |

#### Quick Coding：Serialize and Deserialize Binary Tree

```python
class Codec:
    def serialize(self, root):
        ...

    def deserialize(self, data):
        ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        res = []

        def dfs(node):
            if not node:
                res.append("N")  # 用 "N" 代表空节点
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(res)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        vals = iter(data.split(","))

        def dfs():
            val = next(vals)
            if val == "N":
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()
```

`serialize` 里的 `dfs` 就是普通前序遍历，只是空节点也要写一个标记，否则 `deserialize` 无法判断某个子节点是否存在。

`deserialize` 用 `iter()` 把 token 列表包成迭代器，每次 `next(vals)` 按序列化时同样的顺序取出下一个 token。先建当前节点，再递归建左子树、右子树，顺序和 `serialize` 完全对应，所以每次 `next()` 取到的 token 总是当前正确的那个。

这道题和 Preorder+Inorder 重建是同一个大类：都是"前序定下一个节点，某种机制告诉你子树在哪结束"。那道题里这个机制是中序序列的分割位置，所以需要额外的栈来处理；这里机制就是显式的空标记，递归调用本身天然知道子树的边界，不需要再维护额外状态。

</details>

## 模块五：面试前最后检查

1. 当前题目需要前序、中序、后序还是层序？访问时机是否与目标一致？
2. 递归向父节点返回什么量？整棵树的答案是否需要独立聚合？
3. 是否有路径状态需要向下传递，例如合法区间或路径最大值？
4. BST 题是否已经利用中序有序或单向下降性质？
5. 高度是否被重复计算？是否可以用一个 DFS 和哨兵合并状态？
6. 重建题是否显式记录空节点或使用第二种遍历确定边界？
7. 递归深度、显式栈或 BFS 队列的最坏空间是否已经说明？

最后只记一句：

> 树题的核心是明确每个节点的访问时机、向下传递的状态，以及向父节点返回的量。

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

递归向父节点返回一个可继续组合的量，同时在外层变量中记录整棵树的答案。

```python
best = 0

def dfs(node):
    nonlocal best
    if not node:
        return 0

    left = dfs(node.left)
    right = dfs(node.right)
    best = max(best, combine_for_answer(left, right, node))
    return value_for_parent(left, right, node)
```

Diameter 返回高度，外层记录 `left_height + right_height`。Maximum Path Sum 返回单边向下路径和，外层记录可以同时使用左右分支的完整路径和。整棵树的答案与父节点需要的递归量不同；混用这两个量会产生错误。

Count Good Nodes 使用同一项“递归状态与答案聚合分开”的原则。它的路径最大值从父节点向子节点传递，计数保存在外层变量中，因此属于该模式的自顶向下变体。

使用题目：Diameter of Binary Tree、Binary Tree Maximum Path Sum、Count Good Nodes in Binary Tree。

### 2. 结构比较递归

两个节点同时进入递归。空节点组合和值相等条件共同决定结果。

```python
def same(a, b):
    if not a or not b:
        return a is b
    return (
        a.val == b.val
        and same(a.left, b.left)
        and same(a.right, b.right)
    )
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

### 4. 遍历序列重建

前序序列的第一个值确定当前子树根节点。该值在中序序列中的位置把节点分成左、右子树。用哈希表记录中序下标后，每个节点只处理一次。

```python
preorder_index = 0
inorder_index = {value: i for i, value in enumerate(inorder)}

def build(left, right):
    nonlocal preorder_index
    if left > right:
        return None
    root_value = preorder[preorder_index]
    preorder_index += 1
    root = TreeNode(root_value)
    split = inorder_index[root_value]
    root.left = build(left, split - 1)
    root.right = build(split + 1, right)
    return root
```

Serialize and Deserialize 使用带显式空标记的前序序列。前序先给出当前根，后续标记可以按“左子树、右子树”的固定顺序消费。中序序列先出现左侧内容，缺少额外遍历时无法确定当前根的位置，因此不能单独完成无歧义反序列化。

使用题目：Construct Binary Tree from Preorder and Inorder Traversal、Serialize and Deserialize Binary Tree。

### 5. 高度重复计算与哨兵修复

Balanced Binary Tree 的直接写法会在每个节点重新调用 `height()`。倾斜树上的同一批节点被重复访问，总时间达到 `O(n^2)`。

一次自底向上的 DFS 可以同时计算高度和判断平衡。子树已经失衡时返回 `-1`，父节点立即继续返回 `-1`。

```python
def height_or_unbalanced(node):
    if not node:
        return 0

    left = height_or_unbalanced(node.left)
    if left == -1:
        return -1

    right = height_or_unbalanced(node.right)
    if right == -1 or abs(left - right) > 1:
        return -1

    return 1 + max(left, right)
```

这个 `-1` 与二分查找的边界哨兵、Largest Rectangle 栈中的尾部哨兵属于同一类接口设计：一个特殊值同时携带控制信息并缩短后续处理。

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

这道题组合基础 DFS 与指针交换。每个节点交换左右子树，再递归处理两个子树。前序和后序都可行。

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

        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

</details>

### 2. Maximum Depth of Binary Tree

这是高度递归的基本形式。空树高度为 `0`，非空节点高度为左右子树最大高度加一。

| 项目 | 内容 |
|---|---|
| 组合模式 | 自底向上返回高度 |
| 关键递推 | `1 + max(left_depth, right_depth)` |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Maximum Depth of Binary Tree

```python
def maxDepth(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
```

</details>

### 3. Diameter of Binary Tree

递归向父节点返回高度，外层变量记录任意节点处的 `left_height + right_height`。直径按边数计算。

| 项目 | 内容 |
|---|---|
| 组合模式 | 返回值 + 全局最优值 |
| 返回量 / 答案量 | 子树高度 / 最大直径 |
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
        diameter = 0

        def height(node: Optional[TreeNode]) -> int:
            nonlocal diameter
            if not node:
                return 0

            left = height(node.left)
            right = height(node.right)
            diameter = max(diameter, left + right)
            return 1 + max(left, right)

        height(root)
        return diameter
```

</details>

### 4. Balanced Binary Tree

高度 DFS 使用 `-1` 表示子树已经失衡。该哨兵沿调用栈向上传播，整棵树只遍历一次。

| 项目 | 内容 |
|---|---|
| 组合模式 | 高度递归 + 失衡哨兵 |
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
        def height(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            left = height(node.left)
            if left == -1:
                return -1

            right = height(node.right)
            if right == -1 or abs(left - right) > 1:
                return -1

            return 1 + max(left, right)

        return height(root) != -1
```

</details>

### 5. Same Tree

这是结构比较递归的基本题。两个空节点匹配；只有一个空节点或节点值不同都直接失败。

| 项目 | 内容 |
|---|---|
| 组合模式 | 结构比较递归 |
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
    def isSameTree(
        self,
        p: Optional[TreeNode],
        q: Optional[TreeNode],
    ) -> bool:
        if not p or not q:
            return p is q
        return (
            p.val == q.val
            and self.isSameTree(p.left, q.left)
            and self.isSameTree(p.right, q.right)
        )
```

</details>

### 6. Subtree of Another Tree

这道题把 Same Tree 作为子程序。当前节点匹配失败后，继续在主树左右子树中寻找候选根。

| 项目 | 内容 |
|---|---|
| 组合模式 | 结构比较递归 + 全树枚举候选根 |
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
    def isSubtree(
        self,
        root: Optional[TreeNode],
        subRoot: Optional[TreeNode],
    ) -> bool:
        def same(a: Optional[TreeNode], b: Optional[TreeNode]) -> bool:
            if not a or not b:
                return a is b
            return (
                a.val == b.val
                and same(a.left, b.left)
                and same(a.right, b.right)
            )

        if not subRoot:
            return True
        if not root:
            return False
        return (
            same(root, subRoot)
            or self.isSubtree(root.left, subRoot)
            or self.isSubtree(root.right, subRoot)
        )
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

路径最大值从根向下传递。当前值不小于祖先路径最大值时计数加一，然后更新子节点收到的最大值。

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

        def dfs(node: TreeNode, max_seen: int) -> None:
            nonlocal good
            if not node:
                return

            if node.val >= max_seen:
                good += 1
            next_max = max(max_seen, node.val)
            dfs(node.left, next_max)
            dfs(node.right, next_max)

        dfs(root, root.val)
        return good
```

</details>

### 11. Validate Binary Search Tree

每个节点必须位于所有祖先共同确定的开区间 `(low, high)` 内。使用开区间也明确排除了重复键值。

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
        def valid(node: Optional[TreeNode], low: float, high: float) -> bool:
            if not node:
                return True
            if not low < node.val < high:
                return False
            return (
                valid(node.left, low, node.val)
                and valid(node.right, node.val, high)
            )

        return valid(root, -math.inf, math.inf)
```

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

前序指针依次选择子树根。中序哈希表给出分割位置，递归边界决定左右子树范围。

| 项目 | 内容 |
|---|---|
| 组合模式 | 遍历序列重建 |
| 关键状态 | 前序指针 + 中序区间 |
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
    def buildTree(
        self,
        preorder: List[int],
        inorder: List[int],
    ) -> Optional[TreeNode]:
        inorder_index = {value: i for i, value in enumerate(inorder)}
        preorder_index = 0

        def build(left: int, right: int) -> Optional[TreeNode]:
            nonlocal preorder_index
            if left > right:
                return None

            root_value = preorder[preorder_index]
            preorder_index += 1
            root = TreeNode(root_value)
            split = inorder_index[root_value]
            root.left = build(left, split - 1)
            root.right = build(split + 1, right)
            return root

        return build(0, len(inorder) - 1)
```

</details>

### 14. Binary Tree Maximum Path Sum

父节点只能继续一条向下分支，因此递归返回 `node.val + max(left_gain, right_gain)`。当前节点处的完整候选路径可以同时连接左右分支，并用于更新全局答案。负收益按 `0` 丢弃。

| 项目 | 内容 |
|---|---|
| 组合模式 | 返回值 + 全局最优值 |
| 返回量 / 答案量 | 单边向下收益 / 任意端点最大路径和 |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Binary Tree Maximum Path Sum

```python
def maxPathSum(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
import math


class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        best = -math.inf

        def gain(node: TreeNode) -> int:
            nonlocal best
            if not node:
                return 0

            left = max(0, gain(node.left))
            right = max(0, gain(node.right))
            best = max(best, node.val + left + right)
            return node.val + max(left, right)

        gain(root)
        return best
```

</details>

### 15. Serialize and Deserialize Binary Tree

前序序列记录节点值，并为每个空子节点写入 `#`。反序列化按相同顺序消费 token，每次递归都能确定当前子树是否为空。

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
        tokens = []

        def encode(node: Optional[TreeNode]) -> None:
            if not node:
                tokens.append("#")
                return
            tokens.append(str(node.val))
            encode(node.left)
            encode(node.right)

        encode(root)
        return ",".join(tokens)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        tokens = iter(data.split(","))

        def decode() -> Optional[TreeNode]:
            token = next(tokens)
            if token == "#":
                return None
            node = TreeNode(int(token))
            node.left = decode()
            node.right = decode()
            return node

        return decode()
```

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

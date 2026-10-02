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

面试中树类题目的核心首选永远是**递归**。递归写法不仅代码极其精炼（通常 5~15 行），而且逻辑与树的递归定义天然契合。在白板上手写时，迭代显式栈（尤其是带回溯或状态机的后序模拟）不仅冗长，而且指针极易出错。

掌握以下 5 个核心递归模板，即可覆盖绝大多数二叉树面试题。

---

### 1. 自底向上返回值 + 全局最优值（树形 DP）

子节点向父节点汇报“单边最大收益/单边高度”，父节点拿到左右两边的汇报后，结合自身更新跨越当前节点的“全局最优答案”（左右拼接），最后向上一层父节点返回单边贡献。

**核心直觉**：
- **向上返回**：只能选一条边向上走（单边链，不能分叉）。
- **全局更新**：左右两条分支可以在当前节点拼接成“人字形”路径（全局最优解）。

```python
def solve(root):
    best = 0  # 记录整棵树全局最优值

    def dfs(node):
        nonlocal best
        if not node:
            return 0  # 空节点贡献为 0

        # 1. 递归获得左右子树自底向上的汇报
        left = dfs(node.left)
        right = dfs(node.right)

        # 2. 结合左右汇报更新全局最优解（例如：直径 left + right，或路径和 left + right + val）
        best = max(best, combine_for_answer(left, right, node))

        # 3. 向父节点返回当前节点能提供的单边最大贡献（不能分叉）
        return value_for_parent(left, right, node)

    dfs(root)
    return best
```

> **常见坑**：混淆“向父节点返回的单边量”与“整棵树更新的全局量”。前者不能分叉，后者可以跨越左右子树。

使用题目：Diameter of Binary Tree (543)、Binary Tree Maximum Path Sum (124)。

---

### 2. 结构比较递归（双树同步遍历）

同时递归遍历两棵树的对应节点。判断两棵树结构与值是否相同，只需关注 Base Case 与左右子树的同步递归。

```python
def is_same(a, b):
    # 1. 两者皆空，结构匹配成功
    if not a and not b:
        return True
    # 2. 仅一个为空，或值不相等，匹配失败
    if not a or not b or a.val != b.val:
        return False
    # 3. 递归比较左左与右右
    return is_same(a.left, b.left) and is_same(a.right, b.right)
```

- **核心直觉**：只有当前节点值相等，且左子树和右子树均完全匹配时，两棵树才相同。
- **衍生题**：Subtree of Another Tree (572) 在主树的每个节点调用 `is_same(node, subRoot)`，匹配失败则继续递归左右子树 `isSubtree(node.left, subRoot) or isSubtree(node.right, subRoot)`。

使用题目：Same Tree (100)、Subtree of Another Tree (572)。

---

### 3. BST 有序约束（自顶向下传递有效开区间）

二叉搜索树（BST）的核心性质是**中序遍历严格递增**（每个节点大于左子树的所有节点，且小于右子树的所有节点）。

仅仅检查局部 `node.left.val < node.val < node.right.val` 是**典型错误**，因为这无法发现跨越多层的祖先约束违规（例如右子树的左孙子比根节点还小）。正确解法是**自顶向下传递合法的开区间 `(low, high)`**。

```python
def is_valid_bst(root):
    def validate(node, low, high):
        if not node:
            return True
        # 当前节点值必须落在严格开区间 (low, high) 内
        if not (low < node.val < high):
            return False
        # 往左走上界收紧为 node.val；往右走下界收紧为 node.val
        return validate(node.left, low, node.val) and validate(node.right, node.val, high)

    return validate(root, float('-inf'), float('inf'))
```

BST 的三大经典衍生用法：

| 题目 | 使用方式 | 复杂度 |
|---|---|---|
| Validate BST (98) | 自顶向下传递 `(low, high)` 开区间递归校验 | `O(n) / O(h)` |
| Kth Smallest (230) | 中序遍历天然严格递增，第 $k$ 次访问即为答案 | `O(h + k) / O(h)` |
| LCA of a BST (235) | 二分向下单向搜索：两值均小于当前值往左，均大于往右，分叉处即 LCA | `O(h) / O(1)` |

使用题目：Validate Binary Search Tree (98)、Kth Smallest Element in a BST (230)、Lowest Common Ancestor of a BST (235)。

---

### 4. 遍历序列重建（前序定根，中序切分）

迭代法的单调栈与 `parent` 回溯指针在面试紧张时极易写错指针；而这道题（LeetCode 105）最容易理解和记忆的做法是**带哈希表的递归分治法**。

递归法的本质只有一句话：**“前序找根，中序切成左右两半”**。

#### 核心直觉（3 秒记住）

* **前序遍历（Preorder）：** `[根节点, ...左子树全部节点..., ...右子树全部节点...]`
  * 作用：**第一个元素永远是当前子树的根节点。**
* **中序遍历（Inorder）：** `[...左子树全部节点..., 根节点, ...右子树全部节点...]`
  * 作用：**找到根节点的位置后，其左侧就是左子树全部节点，右侧就是右子树全部节点。**

知道左子树有几个节点后，就能在前序数组中把左子树区间和右子树区间切分开，随后递归构建即可。

```text
preorder: [ 根 |  --- 左子树 (k个) ---  |  --- 右子树 ---  ]
             ↓
inorder:  [ --- 左子树 (k个) --- | 根 | --- 右子树 ---  ]
```

#### 标准解法（指针 + 哈希表）

用哈希表预存 `inorder` 中每个值的下标，可以做到 $O(1)$ 查找根节点；用下标范围代替数组切片，空间和时间都是最优的 $O(n)$。

```python
def build_tree(preorder, inorder):
    # 1. 哈希表快速定位根节点在 inorder 中的位置
    val_to_idx = {val: i for i, val in enumerate(inorder)}

    def helper(pre_left, pre_right, in_left, in_right):
        if pre_left > pre_right:
            return None

        # 前序区间的第一个元素是根
        root_val = preorder[pre_left]
        root = TreeNode(root_val)

        # 找到根在中序中的位置，算出左子树的大小
        in_root_idx = val_to_idx[root_val]
        left_size = in_root_idx - in_left

        # 递归构造左右子树
        root.left = helper(
            pre_left + 1, pre_left + left_size,
            in_left, in_root_idx - 1
        )
        root.right = helper(
            pre_left + left_size + 1, pre_right,
            in_root_idx + 1, in_right
        )

        return root

    return helper(0, len(preorder) - 1, 0, len(inorder) - 1)
```

#### 白板速写版（5 行切片极简代码）

如果不卡切片的复制开销，直接用 Python 切片只有 5 行，直观且无需推导下标：

```python
def build_tree_slice(preorder, inorder):
    if not preorder:
        return None

    root_val = preorder[0]
    root = TreeNode(root_val)

    # 根节点在中序里的索引，也就是左子树的节点数量
    k = inorder.index(root_val)

    root.left = build_tree_slice(preorder[1 : 1 + k], inorder[:k])
    root.right = build_tree_slice(preorder[1 + k :], inorder[k + 1 :])

    return root
```

> **3 秒记忆口诀**：
> 1. 前序拿首位建根；
> 2. 中序找根算长度 $k$；
> 3. 前序跳过首位切 $k$ 个给左边，剩余给右边。

使用题目：Construct Binary Tree from Preorder and Inorder Traversal (105)、Serialize and Deserialize Binary Tree (297)（前序序列带空标记，天然确定子树边界）。

---

### 5. 高度重复计算与哨兵修复（自底向上剪枝）

Balanced Binary Tree 如果用自顶向下的直接写法，会在每个节点重复调用 `height()`，倾斜树上总时间退化为 $O(n^2)$。

最优写法是**自底向上的后序递归**：在返回树高的同时合并平衡校验。约定用 `-1` 作为**失衡哨兵信号**，一旦某棵子树失衡，立即逐层向上透传 `-1` 提前终止，将时间复杂度优化到 $O(n)$。

```python
def is_balanced(root):
    def check(node):
        if not node:
            return 0  # 空节点高度为 0

        left_h = check(node.left)
        if left_h == -1:  # 左子树已失衡，快速剪枝
            return -1

        right_h = check(node.right)
        if right_h == -1:  # 右子树已失衡，快速剪枝
            return -1

        if abs(left_h - right_h) > 1:  # 当前节点左右失衡
            return -1

        return 1 + max(left_h, right_h)  # 平衡时返回真实高度

    return check(root) != -1
```

- **核心直觉**：自底向上后序遍历天然掌握子树所有信息，使用 `-1` 哨兵实现发现违规立即熔断，避免多余递归。

使用题目：Balanced Binary Tree (110)。

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

翻转二叉树最自然的做法是递归：当前节点的左子树等于翻转后的右子树，右子树等于翻转后的左子树。

| 项目 | 内容 |
|---|---|
| 组合模式 | 递归遍历 + 局部指针交换 |
| 关键状态 | `root.left, root.right = invert(root.right), invert(root.left)` |
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
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

> **注**：面试中 4 行递归最为精炼直观。若要求非递归，可用 BFS 队列或栈逐层/逐个交换左右子指针。

</details>

### 2. Maximum Depth of Binary Tree

树的最大深度等于根节点（1）加上左右子树深度的较大值。递归基准为当前节点为空时返回 0。

| 项目 | 内容 |
|---|---|
| 组合模式 | 自底向上高度递归 |
| 关键转移 | `1 + max(maxDepth(root.left), maxDepth(root.right))` |
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

> **注**：也可用 BFS 层序遍历（逐层出队，队列每轮清空一次 `depth += 1`），空间复杂度为 `O(w)`。

</details>

### 3. Diameter of Binary Tree

递归后序遍历计算子树深度。对任意节点，经过它的最长路径所包含的边数为 `left_depth + right_depth`，用全局变量维护最大直径；函数向父节点返回单边最大深度 `1 + max(left_depth, right_depth)`。

| 项目 | 内容 |
|---|---|
| 组合模式 | 自底向上返回值 + 全局最优值 |
| 返回量 / 答案量 | 单边最大深度 / 树的最大直径 |
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
        self.diameter = 0

        def depth(node: Optional[TreeNode]) -> int:
            if not node:
                return 0
            left = depth(node.left)
            right = depth(node.right)
            # 经过当前节点的最长路径更新全局直径
            self.diameter = max(self.diameter, left + right)
            # 向父节点返回单边最大深度
            return 1 + max(left, right)

        depth(root)
        return self.diameter
```

</details>

### 4. Balanced Binary Tree

自底向上递归计算树高。若左右子树任一失衡（返回 `-1`）或左右高度差大于 1，则直接返回 `-1` 提前剪枝；否则返回真实高度 `1 + max(left, right)`。

| 项目 | 内容 |
|---|---|
| 组合模式 | 后序递归 + 哨兵剪枝 |
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
        def check(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            left = check(node.left)
            if left == -1:
                return -1

            right = check(node.right)
            if right == -1:
                return -1

            if abs(left - right) > 1:
                return -1

            return 1 + max(left, right)

        return check(root) != -1
```

</details>

### 5. Same Tree

递归比较两棵树：当前节点值相等，且左子树与左子树相同、右子树与右子树相同。

| 项目 | 内容 |
|---|---|
| 组合模式 | 双树同步递归 |
| 关键状态 | `p.val == q.val and isSame(p.left, q.left) and isSame(p.right, q.right)` |
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
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```

</details>

### 6. Subtree of Another Tree

递归检查：当前树是否与 `subRoot` 完全相同；若不相同，则递归在左子树或右子树中继续寻找。

| 项目 | 内容 |
|---|---|
| 组合模式 | 双重递归（树遍历 + 结构比较） |
| 关键操作 | `self.isSame(root, subRoot)` |
| 时间 / 空间 | 最坏 `O(m * n) / O(h)` |

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
        if not root:
            return False
        if self.isSame(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def isSame(self, s: Optional[TreeNode], t: Optional[TreeNode]) -> bool:
        if not s and not t:
            return True
        if not s or not t or s.val != t.val:
            return False
        return self.isSame(s.left, t.left) and self.isSame(s.right, t.right)
```

</details>

### 7. Lowest Common Ancestor of a BST

利用 BST 有序性单向下降：若 `p` 和 `q` 的值都小于当前节点，LCA 必在左子树；都大于当前节点，必在右子树；否则（分居两侧或其中一个为当前节点），当前节点即为分叉点。

| 项目 | 内容 |
|---|---|
| 组合模式 | BST 有序约束 + 单向分支 |
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
class Solution:
    def lowestCommonAncestor(
        self, root: TreeNode, p: TreeNode, q: TreeNode
    ) -> TreeNode:
        curr = root
        while curr:
            if p.val < curr.val and q.val < curr.val:
                curr = curr.left
            elif p.val > curr.val and q.val > curr.val:
                curr = curr.right
            else:
                return curr
        return root
```

> **注**：递归写法同样极简（仅 4 行）：

```python
if p.val < root.val and q.val < root.val:
    return self.lowestCommonAncestor(root.left, p, q)
if p.val > root.val and q.val > root.val:
    return self.lowestCommonAncestor(root.right, p, q)
return root
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

从根节点递归向下遍历，沿途维护从根到当前路径上的最大值 `max_val`。若当前节点值 `>= max_val`，则计入好节点，并向下传递更新后的最大值。

| 项目 | 内容 |
|---|---|
| 组合模式 | 递归向下传递路径状态 |
| 关键状态 | `cur_max = max(max_val, node.val)` |
| 时间 / 空间 | `O(n) / O(h)` |

#### Quick Coding：Count Good Nodes in Binary Tree

```python
def goodNodes(root):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node: Optional[TreeNode], max_val: int) -> int:
            if not node:
                return 0
            count = 1 if node.val >= max_val else 0
            cur_max = max(max_val, node.val)
            return count + dfs(node.left, cur_max) + dfs(node.right, cur_max)

        return dfs(root, root.val)
```

</details>

### 11. Validate Binary Search Tree

递归自顶向下传递每个节点必须满足的开区间 `(low, high)`。往左走上界收紧为当前值，往右走下界收紧为当前值。

| 项目 | 内容 |
|---|---|
| 组合模式 | BST 有序约束 + 开区间传递 |
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
from typing import Optional


class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def validate(node: Optional[TreeNode], low: float, high: float) -> bool:
            if not node:
                return True
            if not (low < node.val < high):
                return False
            return validate(node.left, low, node.val) and validate(node.right, node.val, high)

        return validate(root, float('-inf'), float('inf'))
```

</details>

### 12. Kth Smallest Element in a BST

BST 的中序遍历天然严格单调递增。通过中序遍历，第 $k$ 个被访问的节点值即为第 $k$ 小元素。

| 项目 | 内容 |
|---|---|
| 组合模式 | BST 中序遍历有序性 |
| 关键状态 | 计数器 `k` 减至 0 提前结束 |
| 时间 / 空间 | `O(h + k) / O(h)` |

#### Quick Coding：Kth Smallest Element in a BST

```python
def kthSmallest(root, k):
    ...
```

<details>
<summary>参考答案</summary>

```python
from typing import Optional


class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.k = k
        self.res = None

        def inorder(node: Optional[TreeNode]):
            if not node or self.res is not None:
                return
            inorder(node.left)
            self.k -= 1
            if self.k == 0:
                self.res = node.val
                return
            inorder(node.right)

        inorder(root)
        return self.res
```

> **注**：也可以用显式栈迭代中序遍历，当第 $k$ 次弹栈时直接返回：

```python
stack = []
curr = root
while stack or curr:
    while curr:
        stack.append(curr)
        curr = curr.left
    curr = stack.pop()
    k -= 1
    if k == 0:
        return curr.val
    curr = curr.right
```

</details>

### 13. Construct Binary Tree from Preorder and Inorder Traversal

这道题最容易理解和记忆的做法是**带哈希表的递归分治法**。
迭代法的栈回溯逻辑（如维护 `parent`）在面试紧张时极易写错指针；而递归法的本质只有一句话：**“前序找根，中序切成左右两半”**。

| 项目 | 内容 |
|---|---|
| 组合模式 | 递归分治 + 哈希表快速定位 |
| 关键状态 | 前序区间 `[pre_left, pre_right]` 与 中序区间 `[in_left, in_right]` |
| 时间 / 空间 | `O(n) / O(n)` |

#### 核心直觉（3 秒记住）

* **前序遍历（Preorder）：** `[根节点, ...左子树全部节点..., ...右子树全部节点...]`
  * 作用：**第一个元素永远是当前子树的根节点。**
* **中序遍历（Inorder）：** `[...左子树全部节点..., 根节点, ...右子树全部节点...]`
  * 作用：**找到根节点的位置后，其左侧就是左子树全部节点，右侧就是右子树全部节点。**

知道左子树有几个节点后，就能在前序数组中把左子树区间和右子树区间切分开，随后递归构建即可。

```text
preorder: [ 根 |  --- 左子树 (k个) ---  |  --- 右子树 ---  ]
             ↓
inorder:  [ --- 左子树 (k个) --- | 根 | --- 右子树 ---  ]
```

#### Quick Coding：Construct Binary Tree from Preorder and Inorder Traversal

```python
def buildTree(preorder, inorder):
    ...
```

<details>
<summary>参考答案</summary>

##### 1. 标准解法（指针 + 哈希表）

用哈希表预存 `inorder` 中每个值的下标，做到 $O(1)$ 查找根节点；用下标范围代替数组切片，空间和时间都是最优的 $O(n)$。

```python
from typing import List, Optional


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # 1. 哈希表快速定位根节点在 inorder 中的位置
        val_to_idx = {val: i for i, val in enumerate(inorder)}

        def helper(pre_left: int, pre_right: int, in_left: int, in_right: int) -> Optional[TreeNode]:
            if pre_left > pre_right:
                return None

            # 前序区间的第一个元素是根
            root_val = preorder[pre_left]
            root = TreeNode(root_val)

            # 找到根在中序中的位置，算出左子树的大小
            in_root_idx = val_to_idx[root_val]
            left_size = in_root_idx - in_left

            # 递归构造左右子树
            root.left = helper(
                pre_left + 1, pre_left + left_size,
                in_left, in_root_idx - 1
            )
            root.right = helper(
                pre_left + left_size + 1, pre_right,
                in_root_idx + 1, in_right
            )

            return root

        return helper(0, len(preorder) - 1, 0, len(inorder) - 1)
```

##### 2. 白板速写极简版（5 行切片代码）

如果不卡切片的开销，直接用 Python 切片代码只有 5 行，直观且无需推导下标：

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None

        root_val = preorder[0]
        root = TreeNode(root_val)

        # 根节点在中序里的索引，也就是左子树的节点数量
        k = inorder.index(root_val)

        root.left = self.buildTree(preorder[1 : 1 + k], inorder[:k])
        root.right = self.buildTree(preorder[1 + k :], inorder[k + 1 :])

        return root
```

> **记忆口诀**：
> 1. 前序拿首位建根；
> 2. 中序找根算长度 $k$；
> 3. 前序跳过首位切 $k$ 个给左边，剩余给右边。

</details>

下面的交互式演示展示了带哈希表的递归分治法逐步构建二叉树的过程：前序定位当前子树根节点，中序切分左右子树区间并递归构造。

```build-tree-demo
```

### 14. Binary Tree Maximum Path Sum

父节点只能继续一条向下分支，因此递归返回 `node.val + max(left_gain, right_gain)`。当前节点处的完整候选路径可以同时连接左右分支，写入 `self.max_sum`。负收益按 `0` 丢弃。

| 项目 | 内容 |
|---|---|
| 组合模式 | 自底向上返回值 + 全局最优值 |
| 返回量 / 答案量 | 单边向下收益 / 任意端点最大路径和 |
| 时间 / 空间 | `O(n) / O(h)` |

这道题与 Diameter of Binary Tree 属于同一种“自底向上返回值 + 全局最优值”的树形 DP 模式。递归向父节点返回单边向下收益（不能分叉），同时用全局变量 `self.max_sum` 记录以当前节点为折返点的最大路径和（左右拼接）。负收益节点直接舍弃（与 0 取较大值）。

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

        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            # 递归计算左右子树单边最大增益，负收益直接舍弃为 0
            max_left = max(0, dfs(node.left))
            max_right = max(0, dfs(node.right))

            # 更新以当前节点为折返点的全局最大路径和
            path_sum = node.val + max_left + max_right
            self.max_sum = max(self.max_sum, path_sum)

            # 向父节点返回当前节点能提供的单边最大增益
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

        def dfs(node: Optional[TreeNode]):
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

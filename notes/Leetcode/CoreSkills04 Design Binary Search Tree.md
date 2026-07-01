# Design Binary Search Tree

## 面试目标

实现二叉搜索树，掌握插入、查找、删除和中序遍历的有序性质。

## 核心设计

- 对任意节点，左子树值更小，右子树值更大。
- 查找时按大小关系决定向左或向右。
- 删除节点分三类：叶子、单子树、双子树。
- 双子树删除常用右子树最小节点或左子树最大节点替换。

## 复杂度

- 平衡时查找/插入/删除：`O(log n)`
- 极端退化成链表时：`O(n)`
- 中序遍历：`O(n)`

## 常见坑

- 删除双子树节点后忘记删除替代节点原位置。
- 没有返回更新后的子树根。
- 忽略重复值策略。

## 参考解法

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

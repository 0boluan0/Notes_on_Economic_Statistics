---
aliases:
- Row Space
- row space
- 行空间
- C(A^T)
tags:
- 线性代数
- concept
---
# Row Space

## 先记一句话

行空间就是：**矩阵的行向量能组合出来的所有输入端方向**。

如果 [[Column Space]] 描述 $A$ 能输出什么，那么 [[Row Space]] 描述的是：

> $A$ 的方程到底在输入 $x$ 身上检查哪些方向？

## 为什么 row space 活在输入空间

对 $m\times n$ 矩阵 $A$，每一行都有 $n$ 个数。它们会和输入向量 $x\in\mathbb{R}^n$ 做点积。

所以 row space 不是 $\mathbb{R}^m$ 里的东西，而是 $\mathbb{R}^n$ 里的子空间。

记号：
$$
C(A^T)=\operatorname{Row}(A).
$$

也就是说，行空间就是 $A^T$ 的列空间。

## 一个最小例子

令
$$
A=\begin{bmatrix}
1&2&3\\
2&4&6
\end{bmatrix}.
$$

第二行是第一行的 2 倍，所以两行其实只提供一个方向：
$$
\operatorname{Row}(A)=
\operatorname{span}\left\{
\begin{bmatrix}1&2&3\end{bmatrix}
\right\}.
$$

因此 row space 是 $\mathbb{R}^3$ 中的一条直线。

这和 [[Column Space]] 不一样：同一个矩阵的列空间在 $\mathbb{R}^2$ 里，而行空间在 $\mathbb{R}^3$ 里。

## 和 [[Null Space]] 的关系

row space 和 nullspace 是输入空间里的正交搭档：
$$
C(A^T)\perp N(A).
$$

直觉是：

- row space 是 $A$ 能看见的输入方向；
- nullspace 是 $A$ 看不见、会压成 0 的输入方向。

如果 $x\in N(A)$，那么 $Ax=0$，这表示 $x$ 和 $A$ 的每一行做点积都等于 0。也就是说，$x$ 和整个 row space 正交。

## 它在题里负责什么

- 判断输入空间 $\mathbb{R}^n$ 被分成了哪些“看得见”和“看不见”的方向。
- 和 [[Null Space]] 一起解释自由变量为什么出现。
- 在四个基本子空间里，row space 的维数也是 [[Matrix Rank]]。

## 必要公式

$$
\operatorname{Row}(A)=C(A^T)\subseteq\mathbb{R}^n.
$$

如果 $\operatorname{rank}(A)=r$，那么
$$
\dim C(A^T)=r,
\qquad
\dim N(A)=n-r.
$$

## 常见误区

- row space 不在 $\mathbb{R}^m$，而在 $\mathbb{R}^n$。
- row reduction 会保持 row space，所以 rref 的非零行可以作为 row space 的基。
- row space 和 [[Column Space]] 维数相同，都是 rank，但它们通常活在不同空间。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.6 Transposes, permutations, vector spaces|Session 1.6]]：transpose 把行和列互换。
- [[01_Ax = b and the Four Subspaces#Session 1.10 Independence, basis, and dimension|Session 1.10]]：rref 的非零行给出 row space 的基。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：row space 与 nullspace 在 $\mathbb{R}^n$ 中正交互补。

## 关联卡片

- [[Null Space]]
- [[Column Space]]
- [[Left Nullspace]]
- [[Matrix Rank]]
- [[Reading the Four Fundamental Subspaces from RREF]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```

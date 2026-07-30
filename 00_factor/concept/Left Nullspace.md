---
aliases:
- Left Nullspace
- left null space
- Left Null Space
- 左零空间
- N(A^T)
tags:
  - 线性代数
  - concept
---
# Left Nullspace

## 先记一句话

左零空间就是：**输出空间里所有和列空间垂直的方向**。

如果 [[Column Space]] 是 $A$ 能打出来的输出，那么 [[Left Nullspace]] 是：

> 输出空间里那些完全不可能由 $A$ 打出来的正交方向。

它最常用来解释：为什么有些 $b$ 不可能让 $Ax=b$ 有解。

## 为什么叫 left nullspace

普通零空间看的是
$$
Ax=0.
$$

左零空间看的是
$$
A^Ty=0.
$$

这等价于
$$
y^TA=0.
$$

因为 $y^T$ 在 $A$ 左边，所以叫 left nullspace。

## 一个最小例子

令
$$
A=\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}.
$$

列空间是
$$
C(A)=\operatorname{span}\left\{
\begin{bmatrix}1\\2\end{bmatrix}
\right\}.
$$

一个和这条线垂直的向量是
$$
y=\begin{bmatrix}2\\-1\end{bmatrix},
$$
因为
$$
\begin{bmatrix}2&-1\end{bmatrix}
\begin{bmatrix}1\\2\end{bmatrix}=0.
$$

所以
$$
N(A^T)=\operatorname{span}\left\{
\begin{bmatrix}2\\-1\end{bmatrix}
\right\}.
$$

## 它怎样给出相容条件

如果 $Ax=b$ 有解，并且 $y\in N(A^T)$，那么
$$
y^Tb=y^TAx=0.
$$

所以所有 left nullspace 里的 $y$ 都会给出一个必要条件：

> $b$ 必须和 left nullspace 正交。

在上面的例子里，
$$
y^Tb=
\begin{bmatrix}2&-1\end{bmatrix}
\begin{bmatrix}b_1\\b_2\end{bmatrix}
=2b_1-b_2.
$$

因此 $Ax=b$ 有解必须满足
$$
2b_1-b_2=0,
$$
也就是 $b_2=2b_1$。这正好是在说：$b$ 必须落在 [[Column Space]] 那条直线上。

## 它在题里负责什么

- 检查 $b$ 是否满足相容条件。
- 解释为什么 $b\notin C(A)$ 时会无解。
- 在 Unit II 里，它会变成最小二乘中的“残差方向”：残差与 column space 正交。

## 必要公式

对 $m\times n$ 矩阵 $A$，
$$
N(A^T)=\{y\in\mathbb{R}^m:A^Ty=0\}.
$$

left nullspace 活在 $\mathbb{R}^m$，因为它描述的是输出端的正交方向。

如果 $\operatorname{rank}(A)=r$，那么
$$
\dim N(A^T)=m-r.
$$

## 常见误区

- left nullspace 不是 [[Null Space]]，它们一个看 $A^T y=0$，一个看 $Ax=0$。
- left nullspace 活在输出空间 $\mathbb{R}^m$，不是输入空间 $\mathbb{R}^n$。
- 它不是考试里可跳过的第四个空间；它负责解释“无解的相容条件”。

## 来自课程位置

- [[01_Ax = b and the Four Subspaces#Session 1.9 Solving Ax = b: row reduced form R|Session 1.9]]：相容条件可以用 left nullspace 理解。
- [[01_Ax = b and the Four Subspaces#Session 1.11 The four fundamental subspaces|Session 1.11]]：left nullspace 与 column space 在 $\mathbb{R}^m$ 中正交互补。
- [[01_Ax = b and the Four Subspaces#Session 1.13 Graphs, networks, and incidence matrices|Session 1.13]]：网络矩阵中 left nullspace 对应连通分量等约束结构。

## 关联卡片

- [[Column Space]]
- [[Row Space]]
- [[Null Space]]
- [[Matrix Rank]]
- [[Orthogonality]]
- [[Reading the Four Fundamental Subspaces from RREF]]


## 最小例子

把 **Left Nullspace** 放在最简单的可计算情形中：先给定定义所需的最小输入，再按定义计算输出；若关键关系不成立，就不能把该对象归入本概念。这个检查也能帮助区分相近概念。
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

---
aliases:
- Orthogonal Complement
- orthogonal complement
- 正交补
tags:
- concept
- 线性代数
---
# Orthogonal Complement

## 先记一句话

正交补就是：**给定一个子空间，把所有与它正交的方向收集起来**。

如果 $S\subseteq\mathbb{R}^n$，那么
$$
S^\perp=\{x\in\mathbb{R}^n:x^Ts=0,\ \forall s\in S\}.
$$

## 它解决什么判断

正交补回答：

> 哪些方向完全不被这个子空间解释？

这句话在线代里很常见：

- 投影里，误差 $e=b-p$ 落在 $S^\perp$。
- 四个基本子空间里，$N(A)$ 是 row space 的正交补。
- left nullspace 是 column space 的正交补。

## 一个最小例子

令
$$
S=\operatorname{span}\left\{
\begin{bmatrix}1\\1\\0\end{bmatrix}
\right\}\subseteq\mathbb{R}^3.
$$

那么 $x\in S^\perp$ 等价于
$$
\begin{bmatrix}x_1&x_2&x_3\end{bmatrix}
\begin{bmatrix}1\\1\\0\end{bmatrix}=0,
$$
即
$$
x_1+x_2=0.
$$

所以 $S^\perp$ 是一个二维平面，而不是“一条垂直线”。

## 四个基本子空间里的位置

对 $m\times n$ 矩阵 $A$：
$$
C(A^T)^\perp=N(A)\subseteq\mathbb{R}^n,
$$
$$
C(A)^\perp=N(A^T)\subseteq\mathbb{R}^m.
$$

这说明：

- row space 和 nullspace 在输入空间里互补；
- column space 和 left nullspace 在输出空间里互补。

## 常见误区

- 正交补的维数不一定是 1；它取决于 ambient space 的维数。
- $S^\perp$ 本身也是子空间。
- 必须先说清楚 $S$ 活在哪个 $\mathbb{R}^n$ 里，否则维数会说错。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.1 Orthogonal vectors and subspaces|Session 2.1]]：正交补与四个基本子空间。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.2 Projections onto subspaces|Session 2.2]]：投影误差落在正交补中。

## 关联卡片

- [[Orthogonality]]
- [[Orthogonal Projection]]
- [[Row Space]]
- [[Null Space]]
- [[Column Space]]
- [[Left Nullspace]]

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

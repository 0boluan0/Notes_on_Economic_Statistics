---
aliases:
- Projection Matrix
- projection operator
- 投影矩阵
tags:
- concept
- 线性代数
---
# Projection Matrix

## 先记一句话

投影矩阵就是：**把任何向量直接送到某个子空间投影点上的矩阵**。

如果 $P$ 是投影到子空间 $S$ 的矩阵，那么对任意 $b$：
$$
Pb=p,
$$
其中 $p$ 是 $b$ 在 $S$ 上的投影。

## 它是什么

若 $A$ 的列张成目标子空间，并且列线性无关，那么投影到 $C(A)$ 的矩阵是
$$
P=A(A^TA)^{-1}A^T.
$$

这条公式来自 [[Orthogonal Projection]] 的误差正交条件：
$$
A^T(b-A\hat{x})=0.
$$

解出
$$
\hat{x}=(A^TA)^{-1}A^Tb,
$$
再代回 $p=A\hat{x}$，得到
$$
p=A(A^TA)^{-1}A^Tb.
$$

所以中间那一串矩阵就是 $P$。

## 一个最小例子

投到一条线 $S=\operatorname{span}(a)$ 上时，
$$
P=\frac{aa^T}{a^Ta}.
$$

若
$$
a=\begin{bmatrix}1\\2\end{bmatrix},
$$
则
$$
P=\frac{1}{5}
\begin{bmatrix}
1&2\\
2&4
\end{bmatrix}.
$$

给定任意 $b$，$Pb$ 就是 $b$ 在这条线上的投影。

## 它在题里负责什么

- 需要快速从 $b$ 得到投影点 $p$ 时，用 $p=Pb$。
- 需要描述 least squares fitted values 时，用 $\hat{b}=Pb$。
- 需要描述 residual maker 时，用 $I-P$，因为
  $$
  e=(I-P)b.
  $$

## 必要性质

投影矩阵一定满足
$$
P^2=P.
$$

意思是：投影一次以后已经落在子空间里，再投影不会改变。

正交投影矩阵还满足
$$
P^T=P.
$$

所以正交投影矩阵既是 idempotent，又是 symmetric。

## 常见误区

- $P$ 投出的是点 $p$，不是坐标 $\hat{x}$。
- $A(A^TA)^{-1}A^T$ 要求 $A$ 的列独立；列不独立时不要硬求 inverse，要改用 [[Pseudoinverse]]。
- $P$ 的列空间就是目标子空间；$I-P$ 的列空间是对应的正交误差方向。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.3 Projection matrices and least squares|Session 2.3]]：从正规方程推导投影矩阵。

## 关联卡片

- [[Orthogonal Projection]]
- [[Least Squares]]
- [[Least Squares via Normal Equations]]
- [[Orthogonal Matrix]]
- [[Pseudoinverse]]

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

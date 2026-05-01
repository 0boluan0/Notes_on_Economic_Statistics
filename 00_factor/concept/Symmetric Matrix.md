---
aliases:
- Symmetric Matrix
- symmetric matrix
- 对称矩阵
tags:
- concept
- 线性代数
---
# Symmetric Matrix

## 先记一句话

对称矩阵就是：**满足 $A^T=A$ 的方阵，也是最容易被正交特征向量拆开的矩阵**。

它的特殊性不只是左右对称，而是：

> 它有实特征值，并且可以选出一组正交特征向量基。

## 它是什么

若
$$
A^T=A,
$$
则 $A$ 是 symmetric matrix。

等价地说：
$$
a_{ij}=a_{ji}.
$$

例如
$$
\begin{bmatrix}
2&1\\
1&3
\end{bmatrix}
$$
是对称矩阵。

## 它为什么重要

对称矩阵有三个关键结构：

- 所有特征值都是实数；
- 不同特征值对应的特征向量互相正交；
- 可以正交对角化：
  $$
  A=Q\Lambda Q^T.
  $$

这就是 [[Spectral Decomposition]] 的基础。

## 一个最小例子

若
$$
A=
\begin{bmatrix}
5&2\\
2&3
\end{bmatrix},
$$
它对称，所以可以用正交特征向量把二次型
$$
x^TAx
$$
拆成互不混杂的平方项。

这就是为什么对称矩阵在 [[Positive Definite Matrix]]、优化和 PCA 里特别重要。

## 它在题里负责什么

- 判断能否使用正交对角化。
- 把二次型 $x^TAx$ 转到特征向量基底。
- 支撑 positive definite 的特征值判据。
- 解释为什么协方差矩阵、Hessian、投影矩阵常常有好性质。

## 常见误区

- 对称矩阵必须是方阵。
- 正对角元全为正不代表矩阵正定；还要看整体结构。
- 一般可对角化不等于正交对角化；正交对角化依赖对称性。

## 来自课程位置

- [[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Session 3.1]]：对称矩阵、实特征值、正交对角化。

## 关联卡片

- [[Spectral Decomposition]]
- [[Positive Definite Matrix]]
- [[Orthogonal Matrix]]
- [[Eigenvalues]]
- [[Eigenvectors]]

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

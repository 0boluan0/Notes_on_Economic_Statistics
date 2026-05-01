---
aliases:
- Orthogonal Matrix
- orthonormal matrix
- 正交矩阵
tags:
- concept
- 线性代数
---
# Orthogonal Matrix

## 先记一句话

正交矩阵就是：**列向量是一组标准正交基的方阵**。

它最重要的性质是
$$
Q^TQ=I,
$$
所以
$$
Q^{-1}=Q^T.
$$

这意味着用 $Q$ 做变换时，长度、角度和内积都不会被破坏。

## 它是什么

一个方阵 $Q$ 是 orthogonal matrix，当且仅当它的列向量满足：

- 每列长度为 1；
- 不同列互相正交。

于是
$$
q_i^Tq_j=
\begin{cases}
1,&i=j,\\
0,&i\neq j.
\end{cases}
$$

把这些内积收集起来就是 $Q^TQ=I$。

## 一个最小例子

二维旋转矩阵
$$
Q=
\begin{bmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{bmatrix}
$$
是正交矩阵。

它会旋转向量，但不会改变向量长度。

## 它在题里负责什么

- 在 [[QR Decomposition]] 中，$Q$ 提供标准正交列。
- 在 [[Spectral Decomposition]] 中，对称矩阵可写成 $A=Q\Lambda Q^T$。
- 在投影和最小二乘中，正交列让计算从 $A^TA$ 简化到 $I$。
- 在换基中，正交基最稳定，因为逆矩阵只是转置。

## 常见误区

- “orthogonal matrix” 说的是矩阵列向量标准正交，不是矩阵和另一个矩阵正交。
- 只有方阵时才说 $Q^{-1}=Q^T$；非方阵的 thin $Q$ 通常只有 $Q^TQ=I$。
- 列正交还不够，必须每列长度为 1，才是 orthogonal matrix。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.4 Orthogonal matrices and Gram-Schmidt|Session 2.4]]：正交矩阵、Gram-Schmidt 与 QR。
- [[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Session 3.1]]：对称矩阵的正交对角化。

## 关联卡片

- [[Orthogonality]]
- [[Gram-Schmidt Orthogonalization]]
- [[QR Decomposition]]
- [[Spectral Decomposition]]
- [[Change of Basis]]

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

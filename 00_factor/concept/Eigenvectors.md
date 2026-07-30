---
aliases:
- Eigenvectors
- eigenvector
- characteristic vector
- 特征向量
tags:
- concept
- 线性代数
---

# Eigenvectors

## 先记一句话

特征向量就是：**被矩阵作用后方向不变的非零向量**。

它满足
$$
Av=\lambda v,\qquad v\neq0.
$$

方向不变，长度和朝向可以变；变化倍数由 [[Eigenvalues]] 记录。

## 它是什么

矩阵 $A$ 通常会把向量旋转到别的方向。

如果某个方向 $v$ 被 $A$ 作用后仍在同一条线上，那么这个方向就是 $A$ 的特征方向。

求特征向量时，对每个特征值 $\lambda$ 解
$$
(A-\lambda I)v=0.
$$

所以特征向量本质上是 $A-\lambda I$ 的 [[Null Space]] 里的非零向量。

## 一个最小例子

令
$$
A=
\begin{bmatrix}
2&0\\
0&3
\end{bmatrix}.
$$

则
$$
A\begin{bmatrix}1\\0\end{bmatrix}
=
2\begin{bmatrix}1\\0\end{bmatrix},
\qquad
A\begin{bmatrix}0\\1\end{bmatrix}
=
3\begin{bmatrix}0\\1\end{bmatrix}.
$$

这两个坐标轴方向都是特征向量方向。

## 它在题里负责什么

- 提供能让矩阵作用解耦的方向。
- 若有足够多线性无关特征向量，可构造 [[Diagonalization]]。
- 对称矩阵的特征向量可以选成正交基，形成 [[Spectral Decomposition]]。
- 在 PCA、Markov 稳态、微分方程中，特征向量给出核心方向。

## 常见误区

- 零向量不能作为特征向量。
- 一个特征值通常对应一整个 eigenspace，不是一根固定向量。
- 特征值有重根时，要检查特征向量数量；不能默认可对角化。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.8 Eigenvalues and eigenvectors|Session 2.8]]：特征向量定义与求法。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.9 Diagonalization and powers of A|Session 2.9]]：用特征向量组成好基。

## 关联卡片

- [[Eigenvalues]]
- [[Diagonalization]]
- [[Null Space]]
- [[Spectral Decomposition]]
- [[Markov Matrix]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Eigenvalues]]、[[Null Space]]、[[Diagonalization]]、[[Spectral Decomposition]]、[[02_Least Squares, Determinants and Eigenvalues]]、[[Markov Matrix]]。

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

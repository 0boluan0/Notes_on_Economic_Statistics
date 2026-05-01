---
aliases:
- Eigenvalues
- eigenvalue
- characteristic value
- 特征值
tags:
- concept
- 线性代数
---
# Eigenvalues

## 先记一句话

特征值就是：**矩阵沿某个不变方向伸缩了多少倍**。

如果存在非零向量 $v$，使得
$$
Av=\lambda v,
$$
那么 $\lambda$ 是特征值，$v$ 是对应的 [[Eigenvectors]]。

## 它是什么

一般向量被矩阵作用后会改变方向。

特征向量特殊在于：方向不变，只被拉伸、压缩、翻转或保持。

特征值 $\lambda$ 记录这个变化倍数：

- $|\lambda|>1$：沿该方向放大；
- $|\lambda|<1$：沿该方向缩小；
- $\lambda<0$：方向翻转；
- $\lambda=0$：该方向被压成 0。

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
Ae_1=2e_1,\qquad Ae_2=3e_2.
$$

所以 $2$ 和 $3$ 是特征值，标准坐标轴方向是特征向量方向。

## 怎么求

从
$$
Av=\lambda v
$$
移项得到
$$
(A-\lambda I)v=0.
$$

要有非零解，矩阵 $A-\lambda I$ 必须不可逆：
$$
\det(A-\lambda I)=0.
$$

这就是 characteristic equation。

## 它在题里负责什么

- 分析 $A^k$ 的长期行为。
- 判断稳定、增长、衰减或振荡。
- 支撑 [[Diagonalization]] 和 [[Matrix Exponential]]。
- 对 [[Symmetric Matrix]]，连接 [[Spectral Decomposition]] 与 [[Positive Definite Matrix]]。

## 常见误区

- 特征值不是随便的缩放倍数；它必须配一个非零特征向量。
- $\lambda=0$ 是合法特征值，并且说明矩阵有非零 nullspace。
- 重根不保证有足够多特征向量；这会影响能否 diagonalize。

## 来自课程位置

- [[02_Least Squares, Determinants and Eigenvalues#Session 2.8 Eigenvalues and eigenvectors|Session 2.8]]：特征值和特征向量定义。
- [[02_Least Squares, Determinants and Eigenvalues#Session 2.9 Diagonalization and powers of A|Session 2.9]]：特征值控制矩阵幂。
- [[03_Positive Definite Matrices and Applications#Session 3.1 Symmetric matrices and positive definiteness|Session 3.1]]：正定矩阵等价于特征值全正。

## 关联卡片

- [[Eigenvectors]]
- [[Diagonalization]]
- [[Matrix Exponential]]
- [[Spectral Decomposition]]
- [[Positive Definite Matrix]]

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

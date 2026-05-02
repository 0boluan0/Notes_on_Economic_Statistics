---
aliases:
- Quadratic Form
- 二次型
tags:
- concept
- multivariate statistics
- 线性代数
---
# Quadratic Form

>[!note] 一句话记忆
> 二次型是形如 $x'Ax$ 的标量表达式，用矩阵描述多维方向上的弯曲、距离或变异。

## 它是什么

给定向量 $x$ 和对称矩阵 $A$，
$$
q(x)=x'Ax.
$$

展开后包含平方项和交叉项：
$$
x'Ax=\sum_i a_{ii}x_i^2+2\sum_{i<j}a_{ij}x_ix_j.
$$

## 解决什么判断

- 矩阵是否正定。
- 马哈拉诺比斯距离如何度量加权距离。
- 多元正态密度中的指数项如何形成椭球。

## 最小例子

$$
\begin{bmatrix}x_1&x_2\end{bmatrix}
\begin{bmatrix}2&1\\1&3\end{bmatrix}
\begin{bmatrix}x_1\\x_2\end{bmatrix}
=2x_1^2+2x_1x_2+3x_2^2.
$$

## 易混点

- $x'Ax$ 是标量，不是矩阵。
- 二次型通常只看 $A$ 的对称部分，因为反对称部分会抵消。
- 正定二次型总是正，半正定二次型允许为 0。

## 来自课程位置

- [[02_矩阵代数和随机向量Matrix Algebra and Random Vectors#1.6. 矩阵平方根与二次型|第2章 6 二次型与 Rayleigh 商]]

## 关联卡片

- [[Positive Definite Matrix]]
- [[Mahalanobis Distance]]
- [[Rayleigh Quotient]]

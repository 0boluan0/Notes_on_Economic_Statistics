---
aliases:
- Matrix Square Root
- Square Root of a Matrix
- 矩阵平方根
tags:
  - concept
  - multivariate statistics
  - 线性代数
---
# Matrix Square Root

>[!note] 一句话记忆
> 矩阵平方根是满足 $A^{1/2}A^{1/2}=A$ 的矩阵，在协方差变换和标准化中很常用。

## 它是什么

若 $A$ 是对称正定矩阵，且
$$
A=Q\Lambda Q',
$$
则
$$
A^{1/2}=Q\Lambda^{1/2}Q',
$$
其中 $\Lambda^{1/2}$ 的对角元是 $\sqrt{\lambda_i}$。

## 解决什么判断

- 如何把标准正态向量变成有协方差 $\Sigma$ 的向量。
- 如何构造 $\Sigma^{-1/2}$ 做标准化。
- 多元正态二次型如何化成标准欧氏距离。

## 最小例子

如果
$$
A=\begin{bmatrix}4&0\\0&9\end{bmatrix},
$$
则
$$
A^{1/2}=\begin{bmatrix}2&0\\0&3\end{bmatrix}.
$$

## 易混点

- 标量平方根的直觉不能直接套到所有矩阵；课程中主要使用对称正定矩阵。
- $A^{-1/2}$ 是 $A^{1/2}$ 的逆，不是对每个原矩阵元素开倒数平方根。

## 来自课程位置

- [[02_矩阵代数和随机向量Matrix Algebra and Random Vectors#1.6.1. 矩阵平方根（Matrix Square Root）|第2章 5 矩阵平方根]]

## 关联卡片

- [[Spectral Decomposition]]
- [[Positive Definite Matrix]]
- [[Multivariate Normal Distribution]]

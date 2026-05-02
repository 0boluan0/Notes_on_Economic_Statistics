---
aliases:
- Covariance Matrix
- Variance-Covariance Matrix
- 协方差矩阵
- 方差协方差矩阵
tags:
- concept
- multivariate statistics
---
# Covariance Matrix

>[!note] 一句话记忆
> 协方差矩阵把每个变量的方差和变量之间的协方差放进同一个对称矩阵。

## 它是什么

对随机向量 $X$，
$$
\Sigma=\operatorname{Cov}(X)=E[(X-\mu)(X-\mu)'].
$$

第 $i,j$ 个元素为
$$
\sigma_{ij}=\operatorname{Cov}(X_i,X_j).
$$

## 解决什么判断

- 哪些变量一起上升或一起下降。
- 多元正态密度、马哈拉诺比斯距离、Hotelling $T^2$ 的尺度如何调整。
- PCA 中哪些方向方差最大。

## 最小例子

二维情形：
$$
\Sigma=
\begin{bmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2\\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{bmatrix}.
$$

## 易混点

- 协方差矩阵受变量量纲影响；量纲差异大时常改用 [[Correlation Matrix]]。
- 理论协方差矩阵一般半正定；课程推断中常需要正定和可逆。
- 样本协方差矩阵在 $n\leq p$ 时通常不可逆。

## 来自课程位置

- [[02_矩阵代数和随机向量Matrix Algebra and Random Vectors#1.7. 随机向量与随机矩阵|第2章 7 随机向量与随机矩阵]]

## 关联卡片

- [[Sample Covariance Matrix]]
- [[Correlation Matrix]]
- [[Positive Definite Matrix]]
- [[Generalized Variance]]

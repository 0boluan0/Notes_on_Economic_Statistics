---
aliases:
- Rayleigh Quotient
- Rayleigh 商
tags:
- concept
- multivariate statistics
- 线性代数
---
# Rayleigh Quotient

>[!note] 一句话记忆
> Rayleigh 商把某个方向上的二次型强度标准化，用来找矩阵最大和最小特征值。

## 它是什么

对对称矩阵 $A$ 和非零向量 $x$，
$$
R(x)=\frac{x'Ax}{x'x}.
$$

若 $A$ 的特征值排序为 $\lambda_1\geq\cdots\geq\lambda_p$，则
$$
\lambda_p\leq R(x)\leq \lambda_1.
$$

最大值在最大特征值对应的特征向量方向取得。

## 解决什么判断

- 为什么 PCA 第一个方向是最大特征值方向。
- 二次型在不同方向上的最大和最小伸缩是多少。
- 正定矩阵为什么能通过特征值判断。

## 最小例子

若 $x=e_1$ 是 $A$ 的单位特征向量且 $Ae_1=\lambda_1e_1$，则
$$
R(e_1)=\lambda_1.
$$

## 易混点

- Rayleigh 商要求 $x\neq0$。
- 它返回标量，不是向量。
- 在课程里常服务于 PCA 和正定矩阵，而不是单独作为计算目标。

## 来自课程位置

- [[02_矩阵代数和随机向量Matrix Algebra and Random Vectors#1.6. 矩阵平方根与二次型|第2章 6 二次型与 Rayleigh 商]]

## 关联卡片

- [[Quadratic Form]]
- [[Eigenvalues]]
- [[PCA]]

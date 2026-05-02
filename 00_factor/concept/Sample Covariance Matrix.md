---
aliases:
- Sample Covariance Matrix
- Sample Variance-Covariance Matrix
- 样本协方差矩阵
tags:
- concept
- multivariate statistics
---
# Sample Covariance Matrix

>[!note] 一句话记忆
> 样本协方差矩阵是总体协方差矩阵的样本估计，也是多元统计推断的尺度矩阵。

## 它是什么

对样本 $X_1,\ldots,X_n$，
$$
S=\frac{1}{n-1}\sum_{j=1}^n (X_j-\bar X)(X_j-\bar X)'.
$$

如果 $D$ 是中心化后的偏差矩阵，则
$$
S=\frac{1}{n-1}D'D.
$$

## 解决什么判断

- 样本中变量之间的线性关系如何。
- $S$ 是否可逆，能否直接做 Hotelling $T^2$。
- 数据的总变异、广义方差和 PCA 特征值如何计算。

## 最小例子

二维样本的 $S$ 为
$$
S=
\begin{bmatrix}
s_{11}&s_{12}\\
s_{12}&s_{22}
\end{bmatrix}.
$$

## 易混点

- 公式必须使用中心化后的数据；直接用原始 $X'X$ 会把均值也混进去。
- 当 $n\leq p$ 或变量完全线性相关时，$S$ 可能奇异。
- $S$ 的对角线是样本方差，非对角线是样本协方差。

## 来自课程位置

- [[03_样本几何与随机抽样Sample Geometry and Random Sampling#1.2. 随机样本与样本矩|第3章 2 随机样本与样本矩]]

## 关联卡片

- [[Covariance Matrix]]
- [[Generalized Variance]]
- [[Wishart Distribution]]
- [[Matrix Rank]]

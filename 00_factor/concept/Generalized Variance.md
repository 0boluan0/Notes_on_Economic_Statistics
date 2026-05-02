---
aliases:
- Generalized Variance
- Generalized Sample Variance
- 广义方差
- 推广方差
tags:
- concept
- multivariate statistics
---
# Generalized Variance

>[!note] 一句话记忆
> 广义方差用协方差矩阵的行列式，把多维数据的联合离散程度压成一个数。

## 它是什么

总体广义方差为
$$
|\Sigma|.
$$

样本广义方差为
$$
|S|.
$$

如果 $\lambda_1,\ldots,\lambda_p$ 是协方差矩阵的特征值，则
$$
|S|=\prod_{i=1}^p \lambda_i.
$$

## 解决什么判断

- 多变量数据整体占据的面积、体积或超体积有多大。
- 变量是否存在完全线性相关。
- 协方差矩阵是否退化。

## 最小例子

二维情形：
$$
|S|=s_{11}s_{22}-s_{12}^2=s_{11}s_{22}(1-r_{12}^2).
$$
若 $|r_{12}|=1$，则 $|S|=0$，数据退化成一条线。

## 易混点

- Trace 只加总单变量方差；广义方差还会惩罚变量之间的高度相关。
- 广义方差为 0 通常意味着协方差矩阵不可逆。
- 变量量纲会影响 $|S|$，比较不同量纲数据时要谨慎。

## 来自课程位置

- [[03_样本几何与随机抽样Sample Geometry and Random Sampling#1.4. 广义方差的几何解释|第3章 3 广义方差几何解释]]

## 关联卡片

- [[Sample Covariance Matrix]]
- [[Determinant]]
- [[Eigenvalues]]
- [[PCA]]

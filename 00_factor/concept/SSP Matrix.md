---
aliases:
- SSP Matrix
- Sum of Squares and Cross Products Matrix
- Sum of Squares and Products Matrix
- 平方和交叉乘积矩阵
tags:
- concept
- multivariate statistics
---
# SSP Matrix

>[!note] 一句话记忆
> SSP 矩阵是多元版平方和，把每个变量的平方和和变量之间的交叉乘积一起记录。

## 它是什么

若 $x_j$ 是观测向量，$\bar x$ 是样本均值向量，则总 SSP 可写作
$$
T=\sum_j (x_j-\bar x)(x_j-\bar x)'.
$$

在 MANOVA 中常分解为
$$
T=H+E,
$$
其中 $H$ 是组间 SSP，$E$ 是组内误差 SSP。

## 解决什么判断

- 多元方差分析中变异如何分解。
- 组间差异和组内误差如何分别进入 Wilks Lambda。
- 为什么 MANOVA 不是逐变量 ANOVA。

## 最小例子

二维变量时，SSP 的对角线是两个变量的平方和，非对角线是两个变量偏差的交叉乘积。

## 易混点

- SSP 还没有除以自由度；除以自由度后才得到均方或协方差形式。
- 非对角线元素保留变量之间的共同变化。

## 来自课程位置

- [[06_比较多个均值向量comparisons of multivariate mean vectors#1.5.3. SSP 矩阵分解|第6章 4.3 SSP 分解]]

## 关联卡片

- [[MANOVA]]
- [[Wilks Lambda]]
- [[Sample Covariance Matrix]]

---
aliases:
- Wilks Lambda
- Wilks' Lambda
- Wilks Λ
- Wilks Lambda 检验
tags:
- concept
- multivariate statistics
---
# Wilks Lambda

>[!note] 一句话记忆
> Wilks Lambda 用“误差变异占总变异的比例”衡量组间均值向量差异。

## 它是什么

在 MANOVA 中，若 $E$ 是误差 SSP 矩阵，$H$ 是组间 SSP 矩阵，则
$$
\Lambda^*=\frac{|E|}{|E+H|}.
$$

值越小，说明组间差异越大。

## 解决什么判断

- 多组均值向量是否有显著差异。
- 组间解释的多元变异是否足够大。
- MANOVA 表里的核心统计量如何读。

## 最小例子

若 $|E|$ 相对 $|E+H|$ 很小，说明总变异中有较大部分来自组别差异，倾向拒绝 $H_0$。

## 易混点

- Wilks Lambda 越小越显著，不是越大越显著。
- 它依赖行列式，因此当矩阵奇异时会出问题。
- 课程中通常先掌握定义和判别方向，再查近似分布或临界值。

## 来自课程位置

- [[06_比较多个均值向量comparisons of multivariate mean vectors#1.5.4. Wilks Lambda 检验|第6章 4.4 Wilks Lambda]]

## 关联卡片

- [[MANOVA]]
- [[SSP Matrix]]
- [[Generalized Variance]]

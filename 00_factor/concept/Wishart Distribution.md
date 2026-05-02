---
aliases:
- Wishart Distribution
- Wishart分布
- Wishart 分布
tags:
- concept
- multivariate statistics
---
# Wishart Distribution

>[!note] 一句话记忆
> Wishart 分布是多元版卡方分布，用来描述正态样本协方差矩阵的随机性。

## 它是什么

若
$$
Z_1,\ldots,Z_m\overset{iid}{\sim}N_p(0,\Sigma),
$$
则
$$
\sum_{j=1}^m Z_jZ_j'\sim W_p(m,\Sigma).
$$

在多元正态样本中：
$$
(n-1)S\sim W_p(n-1,\Sigma).
$$

## 解决什么判断

- 样本协方差矩阵在正态总体下如何分布。
- Hotelling $T^2$ 的 F 分布关系从哪里来。
- MANOVA 中 SSP 矩阵为什么有分布理论。

## 最小例子

当 $p=1$ 时，Wishart 分布退化为卡方分布：
$$
(n-1)\frac{s^2}{\sigma^2}\sim\chi_{n-1}^2.
$$

## 易混点

- Wishart 描述的是矩阵随机变量，不是普通标量。
- $(n-1)S$ 才是 Wishart 形式；不要漏掉缩放。
- 样本均值和样本协方差矩阵独立是正态总体的重要性质。

## 来自课程位置

- [[04_多元正态分布The Multivariate Normal Distribution#1.6. Wishart 分布|第4章 1.6 Wishart 分布]]

## 关联卡片

- [[Multivariate Normal Distribution]]
- [[Sample Covariance Matrix]]
- [[Hotelling T2 Test]]
- [[MANOVA]]

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

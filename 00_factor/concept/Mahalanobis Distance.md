---
aliases:
- Mahalanobis Distance
- 马哈拉诺比斯距离
- Mahalanobis 距离
tags:
- concept
- multivariate statistics
---
# Mahalanobis Distance

>[!note] 一句话记忆
> Mahalanobis 距离是在协方差矩阵调整后的尺度上，衡量点到均值向量有多远。

## 它是什么

对随机向量 $X$、中心 $\mu$ 和协方差矩阵 $\Sigma$，
$$
D^2=(X-\mu)'\Sigma^{-1}(X-\mu).
$$

如果
$$
X\sim N_p(\mu,\Sigma),
$$
则
$$
D^2\sim\chi_p^2.
$$

## 解决什么判断

- 一个观测是否在多元意义上离中心太远。
- 多元正态等密度椭球如何定义。
- Hotelling $T^2$ 为什么是协方差调整后的均值距离。

## 最小例子

两个变量高度相关时，沿着相关方向偏离不一定异常；垂直于相关方向的小偏离可能更异常。Mahalanobis 距离会用 $\Sigma^{-1}$ 调整这种方向差异。

## 易混点

- 它不是普通欧氏距离，变量尺度和相关性都会影响结果。
- 要计算样本版本，需要 $S^{-1}$ 存在。
- 距离大通常提示异常，但是否异常要结合 $\chi_p^2$ 分位数或具体背景。

## 来自课程位置

- [[04_多元正态分布The Multivariate Normal Distribution#1.4. 二次型与相关分布|第4章 1.4 二次型与相关分布]]

## 关联卡片

- [[Multivariate Normal Distribution]]
- [[Covariance Matrix]]
- [[Chi-square Distribution]]
- [[Hotelling T2 Test]]
- [[Outlier Detection]]

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

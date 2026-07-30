---
aliases:
- Hotelling's T-squared Test
- Hotelling T2 Test
- Hotelling T² Test
- Hotelling T² 检验
- Hotelling T2 检验
tags:
  - concept
  - multivariate statistics
---
# Hotelling T2 Test

>[!note] 一句话记忆
> Hotelling $T^2$ 是单样本多元均值检验的多变量版 $t$ 检验。

## 它是什么

若 $X_1,\ldots,X_n\sim N_p(\mu,\Sigma)$，检验
$$
H_0:\mu=\mu_0
$$
时使用
$$
T^2=n(\bar X-\mu_0)'S^{-1}(\bar X-\mu_0).
$$

在 $H_0$ 下，
$$
\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.
$$

## 解决什么判断

- 多个指标的均值向量是否整体等于目标值。
- 一个联合检验是否显著，即使单个分量置信区间看起来不显著。
- 配对样本、两样本均值向量比较能否转化为 $T^2$ 问题。

## 最小例子

同时检验产品的重量、硬度、厚度三个均值是否等于设计标准，不能只做三个独立 $t$ 检验；Hotelling $T^2$ 会把变量间协方差纳入统计量。

## 易混点

- $T^2$ 的核心是 [[Mahalanobis Distance]]：离目标越远、协方差调整后越异常，统计量越大。
- $S$ 必须可逆；当 $n\leq p$ 或变量完全线性相关时不能直接使用。
- 大样本时可用 $\chi_p^2$ 近似，但小样本正态条件下用 $F$ 转换。

## 来自课程位置

- [[05_ 总体平均向量的推论#1. 第5章：总体平均向量的推论（Inferences about Population Mean Vector）|第5章 总体平均向量的推论]]
- [[06_比较多个均值向量comparisons of multivariate mean vectors#1. 第6章：比较多个均值向量（Comparisons of Multivariate Mean Vectors）|第6章 比较多个均值向量]]

## 关联卡片

- [[Hotelling T2 Test Procedure]]
- [[Paired Mean Vector Comparison]]
- [[Two-Sample Hotelling T2 Test]]
- [[Multivariate Mean Inference Map]]
- [[Mahalanobis Distance]]
- [[Wishart Distribution]]

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

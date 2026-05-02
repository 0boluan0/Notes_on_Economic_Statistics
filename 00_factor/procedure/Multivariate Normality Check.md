---
aliases:
- Multivariate Normality Check
- Multivariate Normality Assessment
- 多元正态性检查
tags:
- procedure
- multivariate statistics
---
# Multivariate Normality Check

## 输入

- 多变量样本 $X_1,\ldots,X_n$。
- 样本均值向量 $\bar X$。
- 样本协方差矩阵 $S$。

## 输出

- 对多元正态假设是否合理的诊断判断。

## Step 1. 先看单变量图形

对每个变量看直方图、箱线图、单变量 QQ 图。

## Step 2. 计算平方 Mahalanobis 距离

$$
d_j^2=(X_j-\bar X)'S^{-1}(X_j-\bar X).
$$

## Step 3. 和卡方分位数比较

若样本近似来自 $N_p(\mu,\Sigma)$，则 $d_j^2$ 应近似服从 $\chi_p^2$。

把排序后的 $d_j^2$ 与 $\chi_p^2$ 分位数作 QQ 比较。

## Step 4. 检查异常点

距离特别大的样本可能是多元异常值。

## 检查点

- $S$ 必须可逆。
- 单变量近似正态不保证多元正态。
- 异常值会强烈影响 $\bar X$ 和 $S$。

## 来自课程位置

- [[04_多元正态分布The Multivariate Normal Distribution#1.8. 正态性假设检验|第4章 8 正态性假设检验]]

## 关联卡片

- [[Multivariate Normal Distribution]]
- [[Mahalanobis Distance]]
- [[Chi-square Distribution]]
- [[Outlier Detection]]

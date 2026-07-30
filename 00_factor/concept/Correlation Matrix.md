---
aliases:
- Correlation Matrix
- 相关矩阵
- 相关系数矩阵
tags:
- concept
- multivariate statistics
---

# Correlation Matrix

>[!note] 一句话记忆
> 相关矩阵是把协方差矩阵标准化后得到的无量纲关系矩阵。

## 它是什么

若 $D=\operatorname{diag}(\sigma_{11},\ldots,\sigma_{pp})$，则总体相关矩阵为
$$
\rho=D^{-1/2}\Sigma D^{-1/2}.
$$

第 $i,j$ 个元素为
$$
\rho_{ij}=\frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}}.
$$

## 解决什么判断

- 变量量纲不同但想比较线性关系时用什么矩阵。
- PCA 是否应该基于相关矩阵而不是协方差矩阵。
- 多变量之间的线性关联是否强。

## 最小例子

身高和收入量纲不同，协方差数值不好解释；相关系数把关系压到 $[-1,1]$。

## 易混点

- 相关矩阵只消除量纲，不消除异常值和非线性关系。
- 相关为 0 表示线性无关，不一定独立；但在多元正态下，零协方差等价于独立。

## 来自课程位置

- [[02_矩阵代数和随机向量Matrix Algebra and Random Vectors#1.8. 相关矩阵（Correlation Matrix）|第2章 8 相关矩阵]]
- [[03_样本几何与随机抽样Sample Geometry and Random Sampling#1.6. 标准化协方差矩阵与相关矩阵|第3章 5 标准化与相关矩阵]]

## 关联卡片

- [[Covariance Matrix]]
- [[Choosing Covariance vs Correlation Matrix]]
- [[Correlation Coefficient]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[02_矩阵代数和随机向量Matrix Algebra and Random Vectors]]、[[03_样本几何与随机抽样Sample Geometry and Random Sampling]]、[[Covariance Matrix]]、[[Choosing Covariance vs Correlation Matrix]]、[[Correlation Coefficient]]。

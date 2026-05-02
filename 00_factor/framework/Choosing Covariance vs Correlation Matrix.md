---
aliases:
- Choosing Covariance vs Correlation Matrix
- 协方差矩阵和相关矩阵选择
tags:
- framework
- multivariate statistics
---
# Choosing Covariance vs Correlation Matrix

## 什么时候用

当 PCA、因子分析、聚类或描述统计需要选择矩阵尺度时，用这张卡判断用协方差矩阵还是相关矩阵。

## 如何识别

| 情况 | 更适合 |
|---|---|
| 变量量纲相同，方差大小本身有意义 | [[Covariance Matrix]] |
| 变量量纲差异大 | [[Correlation Matrix]] |
| 希望保留原始尺度下的大方差变量优势 | 协方差矩阵 |
| 希望每个变量先等权进入分析 | 相关矩阵 |

## 为什么这样看

协方差矩阵保留原始尺度，所以方差大的变量会主导结果；相关矩阵相当于先标准化变量，使每个变量方差为 1。

## 边界条件

- 标准化不能修复异常值和非线性关系。
- 如果变量单位相同且方差差异有业务含义，不要机械用相关矩阵。

## 失败模式

- 身高、收入、年龄混在一起直接用协方差矩阵做 PCA。
- 在变量本来同单位且方差大小重要时过度标准化。

## 来自课程位置

- [[03_样本几何与随机抽样Sample Geometry and Random Sampling#1.6. 标准化协方差矩阵与相关矩阵|第3章 5 标准化与相关矩阵]]
- [[08_主成分分析principal component#1.4. 标准化变量的主成分|第8章 3 标准化变量的主成分]]

## 关联卡片

- [[Covariance Matrix]]
- [[Correlation Matrix]]
- [[PCA Procedure]]

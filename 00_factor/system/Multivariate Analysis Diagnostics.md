---
aliases:
- Multivariate Analysis Diagnostics
- 多元统计诊断
tags:
- system
- multivariate statistics
---
# Multivariate Analysis Diagnostics

## 诊断目标

多元统计的主要风险不是公式不会套，而是矩阵条件、分布假设和变量尺度不满足公式要求。

## 关键检查

| 检查项 | 为什么重要 |
|---|---|
| $n>p$ | 避免样本协方差矩阵奇异 |
| $S$ 是否可逆 | Hotelling $T^2$、Mahalanobis 距离都需要 |
| 变量是否同量纲 | 影响 PCA、聚类和距离 |
| 异常值 | 强烈影响 $\bar X$ 与 $S$ |
| 多元正态性 | 小样本精确推断的基础 |
| 协方差是否相等 | 两样本 pooled 检验和 Fisher 判别的边界 |

## 稳健性做法

- 先画图：散点图、箱线图、距离图。
- 对尺度差异大的变量先标准化。
- 当 $S$ 奇异时，考虑删冗余变量、PCA 降维或正则化。
- 分类和聚类结果要做交叉验证或敏感性检查。

## 风险点

- 多个单变量显著不等于多元联合显著。
- 多元联合显著也不说明每个分量都单独显著。
- 自动链接如果把普通词错连到其他课程卡，会破坏复习路径。

## 来自课程位置

- [[03_样本几何与随机抽样Sample Geometry and Random Sampling]]
- [[04_多元正态分布The Multivariate Normal Distribution]]
- [[05_ 总体平均向量的推论]]
- [[11_分类与判别Discrimination and Classifications]]
- [[12_层次聚类和K-means聚类]]

## 关联卡片

- [[Sample Covariance Matrix]]
- [[Multivariate Normality Check]]
- [[Choosing Covariance vs Correlation Matrix]]

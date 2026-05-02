---
aliases:
- Variance Explained
- Proportion of Variance Explained
- Cumulative Variance Explained
- 方差解释率
- 累计方差解释率
tags:
- concept
- multivariate statistics
---
# Variance Explained

>[!note] 一句话记忆
> 方差解释率衡量某个主成分或前几个主成分保留了多少原始总变异。

## 它是什么

第 $k$ 个主成分的方差解释率为
$$
\frac{\lambda_k}{\sum_{i=1}^p\lambda_i}.
$$

前 $m$ 个主成分的累计解释率为
$$
\frac{\sum_{i=1}^m\lambda_i}{\sum_{i=1}^p\lambda_i}.
$$

若 PCA 基于相关矩阵，总方差为 $p$。

## 解决什么判断

- 降维后损失了多少信息。
- 需要保留几个主成分。
- PCA 结果是否足够概括数据。

## 最小例子

若特征值为 $4,1,1$，前两个主成分解释 $(4+1)/6=83.3\%$ 的总变异。

## 易混点

- 解释方差高不等于解释因果机制。
- 对相关矩阵做 PCA 时总方差是变量数 $p$，不是原始量纲下的方差和。

## 来自课程位置

- [[08_主成分分析principal component#1.6. 主成分数量选择|第8章 4 主成分数量选择]]

## 关联卡片

- [[PCA]]
- [[Scree Plot]]
- [[Eigenvalues]]

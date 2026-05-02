---
aliases:
- Principal Component Analysis
- PCA
- 主成分分析
tags:
- concept
- multivariate statistics
---
# Principal Component Analysis (PCA)

>[!note] 一句话记忆
> PCA 是把数据旋转到一组互相正交的新坐标，使第一个方向解释最大方差，第二个方向解释剩余最大方差，以此类推。

## 它是什么

给定随机向量 $X$ 的协方差矩阵 $\Sigma$，令
$$
\Sigma e_i=\lambda_i e_i,\qquad \lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p.
$$

第 $i$ 个总体主成分是
$$
Y_i=e_i'X.
$$

它的方差是
$$
\operatorname{Var}(Y_i)=\lambda_i,
$$
不同主成分之间互不相关。

## 解决什么判断

- 数据能不能用少数几个方向解释大部分变异。
- 协方差矩阵还是相关矩阵更适合做降维。
- 某个变量是否和其他变量共享了主要变异方向。

## 最小例子

如果前三个特征值为 $5,2,1$，总方差为 $8$，则前两个主成分解释
$$
\frac{5+2}{8}=87.5\%
$$
的总变异。

## 易混点

- PCA 不是因果模型，只是方差分解和坐标旋转。
- 用协方差矩阵做 PCA 会受量纲影响；量纲差异大时通常改用相关矩阵。
- PCA 解释的是总方差，[[Factor Analysis]] 试图解释变量间的共同协方差结构。

## 来自课程位置

- [[08_主成分分析principal component#1. 第8章：主成分分析（Principal Component Analysis）|第8章 主成分分析]]

## 关联卡片

- [[PCA Procedure]]
- [[PCA vs Factor Analysis]]
- [[Scree Plot]]
- [[Variance Explained]]
- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Choosing Covariance vs Correlation Matrix]]

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

---
aliases:
- K-means Clustering
- K-means
- K 均值聚类
- K-means 聚类
tags:
  - concept
  - multivariate statistics
---
# K-means Clustering

>[!note] 一句话记忆
> K-means 把样本分到 $K$ 个中心附近，通过反复更新类别和中心最小化组内平方距离。

## 它是什么

目标函数可写作
$$
\min_{C_1,\ldots,C_K}\sum_{k=1}^K\sum_{x_i\in C_k}\|x_i-\bar x_k\|^2.
$$

其中 $\bar x_k$ 是第 $k$ 类的中心。

## 解决什么判断

- 给定类别数 $K$ 时如何把样本分成紧凑的组。
- 哪些样本离同一中心更近。
- 聚类结果是否受初始中心影响。

## 最小例子

把客户按消费频率和消费金额聚成 $K=3$ 类：低频低额、中等客户、高价值客户。

## 易混点

- $K$ 要事先给定或另行选择。
- 对初始中心敏感，通常要多次随机初始化。
- 对尺度敏感，变量常需要标准化。

## 来自课程位置

- [[12_层次聚类和K-means聚类#1.3. K-means 聚类（K-means Clustering）|第12章 3 K-means 聚类]]

## 关联卡片

- [[K-means Algorithm]]
- [[Clustering Method Selection]]
- [[Hierarchical Clustering]]

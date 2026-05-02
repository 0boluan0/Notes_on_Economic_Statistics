---
aliases:
- Hierarchical Clustering
- 层次聚类
tags:
- concept
- multivariate statistics
---
# Hierarchical Clustering

>[!note] 一句话记忆
> 层次聚类通过逐步合并或拆分样本形成树状结构，不需要一开始固定每个样本的最终类别。

## 它是什么

课程常见的是凝聚型层次聚类：

1. 每个样本先自成一类。
2. 按 linkage 定义的簇间距离，合并最近的两个簇。
3. 重复直到所有样本合并成一棵树。

## 解决什么判断

- 数据是否自然形成层级结构。
- 不同距离定义下聚类结果如何变化。
- dendrogram 在哪里切可以得到合适类别数。

## 最小例子

根据城市之间的人口、收入、消费结构距离，先合并最相似城市，再逐层形成区域分组。

## 易混点

- linkage 选择会显著改变结果。
- 层次聚类一旦合并通常不回退。
- 距离尺度会影响聚类，变量常需要标准化。

## 来自课程位置

- [[12_层次聚类和K-means聚类#1.2. 层次聚类（Hierarchical Clustering）|第12章 2 层次聚类]]

## 关联卡片

- [[Hierarchical Clustering Procedure]]
- [[Linkage Criterion]]
- [[Clustering Method Selection]]

---
aliases:
- Clustering Method Selection
- 聚类方法选择
tags:
- framework
- multivariate statistics
type: framework
---
# Clustering Method Selection

## 什么时候用

当题目要求把未标记样本分组时，用这张卡判断层次聚类还是 K-means。

## 如何识别

| 需求 | 更适合 |
|---|---|
| 想看层级关系或树状图 | [[Hierarchical Clustering]] |
| 题目给定 $K$，希望迭代求中心 | [[K-means Clustering]] |
| 样本量小，重视解释结构 | 层次聚类 |
| 样本量大，重视快速分组 | K-means |

## 边界条件

- 两者都依赖距离定义。
- 变量尺度不一致时通常先标准化。
- K-means 对初始中心敏感。

## 失败模式

- 不标准化就把不同量纲变量放进距离。
- 把层次聚类的 linkage 当成样本距离本身。
- 忘记 K-means 的 $K$ 不是自动真理。

## 来自课程位置

- [[12_层次聚类和K-means聚类]]

## 关联卡片

- [[Hierarchical Clustering Procedure]]
- [[K-means Algorithm]]
- [[Linkage Criterion]]

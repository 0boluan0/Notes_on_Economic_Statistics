---
aliases:
- Linkage Criterion
- Linkage
- 簇间距离准则
- 层次聚类连接准则
tags:
  - concept
  - multivariate statistics
---
# Linkage Criterion

>[!note] 一句话记忆
> linkage 是层次聚类中定义两个簇之间距离的规则。

## 它是什么

常见规则：

- single linkage：两簇最近样本点之间的距离；
- complete linkage：两簇最远样本点之间的距离；
- average linkage：两簇样本点两两距离的平均值。

## 解决什么判断

- 层次聚类每一步应该合并哪两个簇。
- 聚类结果更偏链状、紧凑还是折中。
- dendrogram 的形状为什么随规则改变。

## 最小例子

两个簇之间最近点很近但其余点相距很远时，single linkage 可能先合并，complete linkage 往往不会。

## 易混点

- linkage 是簇间距离，不是样本点之间的原始距离。
- single linkage 容易产生链式聚类。
- complete linkage 更偏好紧凑簇。

## 来自课程位置

- [[12_层次聚类和K-means聚类#1.2. 层次聚类（Hierarchical Clustering）|第12章 2 层次聚类]]

## 关联卡片

- [[Hierarchical Clustering]]
- [[Hierarchical Clustering Procedure]]
- [[Clustering Method Selection]]
## 符号表达

将本概念记为 $C_{LinkageCrite}$；使用时先明确对象、条件与输出，再判断 $C$ 是否满足定义。

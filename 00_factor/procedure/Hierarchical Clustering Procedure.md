---
aliases:
- Hierarchical Clustering Procedure
- 层次聚类步骤
tags:
- procedure
- multivariate statistics
---
# Hierarchical Clustering Procedure

## 输入

- 样本点。
- 样本距离定义。
- linkage criterion。

## 输出

- 层次聚类树。
- 选定切割高度下的类别。

## Step 1. 计算样本距离矩阵

根据欧氏距离或其他距离度量，计算所有样本两两距离。

## Step 2. 初始化簇

每个样本单独作为一个簇。

## Step 3. 计算簇间距离

使用 [[Linkage Criterion]]：

- single linkage；
- complete linkage；
- average linkage。

## Step 4. 合并最近簇

找到距离最小的两个簇并合并。

## Step 5. 重复直到完成

不断更新簇间距离并合并，直到所有样本在同一棵树中。

## 检查点

- 聚类前变量通常要标准化。
- linkage 改变会改变树形结构。
- 树状图切割高度要结合业务解释。

## 来自课程位置

- [[12_层次聚类和K-means聚类#1.2. 层次聚类（Hierarchical Clustering）|第12章 2 层次聚类]]

## 关联卡片

- [[Hierarchical Clustering]]
- [[Linkage Criterion]]
- [[Clustering Method Selection]]

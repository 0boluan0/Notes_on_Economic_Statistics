---
aliases:
- K-means Algorithm
- K-means Steps
- K-means 算法
tags:
- procedure
- multivariate statistics
---
# K-means Algorithm

## 输入

- 数据点 $x_1,\ldots,x_n$。
- 类别数 $K$。
- 初始中心或初始分类。

## 输出

- 每个样本所属类别。
- 每个类别的中心。

## Step 1. 初始化

随机选取 $K$ 个中心，或随机给样本分配类别。

## Step 2. 更新中心

对每一类 $C_k$ 计算
$$
\bar x_k=\frac{1}{|C_k|}\sum_{x_i\in C_k}x_i.
$$

## Step 3. 重新分配类别

把每个样本分配给最近中心：
$$
\arg\min_k \|x_i-\bar x_k\|^2.
$$

## Step 4. 迭代到收敛

重复 Step 2 和 Step 3，直到分类不再变化或目标函数变化很小。

## 检查点

- 变量尺度会影响距离，通常先标准化。
- 不同初始值可能得到不同结果。
- 空簇需要重新初始化。

## 常见错误

- 把 $K$ 当成算法自动给出的结果。
- 用非数值或尺度不一致变量直接算欧氏距离。

## 来自课程位置

- [[12_层次聚类和K-means聚类#1.3. K-means 聚类（K-means Clustering）|第12章 3 K-means 聚类]]

## 关联卡片

- [[K-means Clustering]]
- [[Clustering Method Selection]]

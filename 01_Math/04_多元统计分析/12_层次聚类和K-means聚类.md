# 层次聚类与 K-means

## Linkage（层次聚类的簇间距离）

本节常考三种 linkage 定义。

- [[Single-Factor Model|single]] linkage：两簇最近样本点之间的距离（取最小）。
- complete linkage：两簇最远样本点之间的距离（取最大）。
- average linkage：两簇样本点两两距离的平均值。

## K-means（迭代聚类）

聚类个数 $K$ 一般由题目给定。算法流程可记为：

1. 随机初始化每个样本的类别（或随机初始化 $K$ 个中心）。
2. 计算每一类的中心（centroid）。
3. 按“到各中心距离最小”原则，重新分配每个样本的类别。
4. 重复步骤 2-3，直到类别不再变化（或目标函数收敛）。

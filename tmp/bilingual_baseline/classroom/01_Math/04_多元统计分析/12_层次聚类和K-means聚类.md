# 1. 第12章：层次聚类与 K-means

>[!note] 本章主线
> 聚类是无监督分组：没有预先给定类别标签，只根据样本之间的距离或相似性形成类别。

## 1.1. 聚类前的共同问题

聚类前先确认三件事：

1. 距离如何定义。
2. 变量是否需要标准化。
3. 希望得到层级结构，还是固定数量的类别。

>[!attention] 尺度问题
> 如果一个变量以“万元”为单位，另一个变量以“百分比”为单位，直接用欧氏距离会让量纲大的变量主导聚类。

## 1.2. 层次聚类（Hierarchical Clustering）

层次聚类常见为凝聚式流程：

1. 每个样本先自成一类。
2. 计算所有簇之间的距离。
3. 合并距离最近的两个簇。
4. 重复直到所有样本合并成一棵树。

### 1.2.1. Linkage：簇间距离

常考三种 linkage：

| linkage | 定义 | 直觉 |
|---|---|---|
| single linkage | 两簇最近样本点之间的距离 | 容易形成链状结构 |
| complete linkage | 两簇最远样本点之间的距离 | 偏好紧凑簇 |
| average linkage | 两簇样本点两两距离的平均值 | 折中 |

>[!note] 读树状图
> dendrogram 的纵轴通常表示合并距离。切割高度越低，类别越多；切割高度越高，类别越少。

## 1.3. K-means 聚类（K-means Clustering）

K-means 的类别数 $K$ 通常由题目给定。

算法流程：

1. 随机初始化 $K$ 个中心，或随机初始化样本类别。
2. 计算每一类的中心（centroid）。
3. 按到中心距离最小原则，重新分配每个样本。
4. 重复步骤 2-3，直到类别不再变化或目标函数收敛。

目标函数为
$$
\min_{C_1,\ldots,C_K}
\sum_{k=1}^K\sum_{x_i\in C_k}\|x_i-\bar x_k\|^2.
$$

## 1.4. 层次聚类 vs K-means

| 问题 | 层次聚类 | K-means |
|---|---|---|
| 是否预设类别数 | 不一定 | 需要 $K$ |
| 输出 | 树状结构 | 固定 $K$ 类 |
| 是否会回退 | 合并后通常不回退 | 每轮可重新分配 |
| 对初始值敏感 | 较少 | 明显 |
| 适合 | 看层级关系 | 快速分成给定类数 |

## 1.5. 关联卡片

- [[Clustering Method Selection]]
- [[Hierarchical Clustering]]
- [[Hierarchical Clustering Procedure]]
- [[Linkage Criterion]]
- [[K-means Clustering]]
- [[K-means Algorithm]]

# 1. 第12章：层次聚类与 K-means
<!-- bilingual-en:start -->
*1. Chapter 12: Hierarchical Clustering and K-means*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 聚类是无监督分组：没有预先给定类别标签，只根据样本之间的距离或相似性形成类别。
> <!-- bilingual-en:start -->
> Clustering is unsupervised grouping: no class labels are supplied in advance, and groups are formed only from distances or similarities among observations.
> <!-- bilingual-en:end -->

## 1.1. 聚类前的共同问题
<!-- bilingual-en:start -->
*1.1. Questions Shared by All Clustering Methods*
<!-- bilingual-en:end -->

聚类前先确认三件事：
<!-- bilingual-en:start -->
Before clustering, settle three questions:
<!-- bilingual-en:end -->

1. 距离如何定义。
2. 变量是否需要标准化。
3. 希望得到层级结构，还是固定数量的类别。
<!-- bilingual-en:start -->
1. How is distance defined?
2. Do the variables need to be standardised?
3. Is the goal a hierarchy or a fixed number of groups?
<!-- bilingual-en:end -->

>[!attention] 尺度问题
> 如果一个变量以“万元”为单位，另一个变量以“百分比”为单位，直接用欧氏距离会让量纲大的变量主导聚类。
> <!-- bilingual-en:start -->
> If one variable is measured in tens of thousands of currency units and another in percentages, using Euclidean distance directly will allow the larger-scale variable to dominate the clustering.
> <!-- bilingual-en:end -->

## 1.2. 层次聚类（Hierarchical Clustering）
<!-- bilingual-en:start -->
*1.2. Hierarchical Clustering*
<!-- bilingual-en:end -->

层次聚类常见为凝聚式流程：
<!-- bilingual-en:start -->
Hierarchical clustering commonly follows an agglomerative procedure:
<!-- bilingual-en:end -->

1. 每个样本先自成一类。
2. 计算所有簇之间的距离。
3. 合并距离最近的两个簇。
4. 重复直到所有样本合并成一棵树。
<!-- bilingual-en:start -->
1. Begin with each observation in its own cluster.
2. Compute the distances between all clusters.
3. Merge the two closest clusters.
4. Repeat until all observations have been merged into one tree.
<!-- bilingual-en:end -->

### 1.2.1. Linkage：簇间距离
<!-- bilingual-en:start -->
*1.2.1. Linkage: Distance Between Clusters*
<!-- bilingual-en:end -->

常考三种 linkage：
<!-- bilingual-en:start -->
Three linkage rules are commonly examined:
<!-- bilingual-en:end -->

| linkage | 定义 | 直觉 |
|---|---|---|
| single linkage | 两簇最近样本点之间的距离 | 容易形成链状结构 |
| complete linkage | 两簇最远样本点之间的距离 | 偏好紧凑簇 |
| average linkage | 两簇样本点两两距离的平均值 | 折中 |
<!-- bilingual-en:start -->
| Linkage | Definition | Intuition |
|---|---|---|
| single linkage | Distance between the closest pair of points in the two clusters | Tends to produce chains |
| complete linkage | Distance between the farthest pair of points in the two clusters | Favours compact clusters |
| average linkage | Mean of all pairwise distances between points in the two clusters | A compromise |
<!-- bilingual-en:end -->

>[!note] 读树状图
> dendrogram 的纵轴通常表示合并距离。切割高度越低，类别越多；切割高度越高，类别越少。
> <!-- bilingual-en:start -->
> The vertical axis of a dendrogram usually represents the merge distance. Cutting at a lower height produces more clusters; cutting at a higher height produces fewer clusters.
> <!-- bilingual-en:end -->

## 1.3. K-means 聚类（K-means Clustering）
<!-- bilingual-en:start -->
*1.3. K-means Clustering*
<!-- bilingual-en:end -->

K-means 的类别数 $K$ 通常由题目给定。
<!-- bilingual-en:start -->
The number of K-means clusters, $K$, is usually specified by the question.
<!-- bilingual-en:end -->

算法流程：
<!-- bilingual-en:start -->
The algorithm proceeds as follows:
<!-- bilingual-en:end -->

1. 随机初始化 $K$ 个中心，或随机初始化样本类别。
2. 计算每一类的中心（centroid）。
3. 按到中心距离最小原则，重新分配每个样本。
4. 重复步骤 2-3，直到类别不再变化或目标函数收敛。
<!-- bilingual-en:start -->
1. Randomly initialise $K$ centres or randomly assign observations to clusters.
2. Compute each cluster's centroid.
3. Reassign each observation to its nearest centroid.
4. Repeat steps 2–3 until assignments stop changing or the objective converges.
<!-- bilingual-en:end -->

目标函数为
<!-- bilingual-en:start -->
The objective function is
<!-- bilingual-en:end -->
$$
\min_{C_1,\ldots,C_K}
\sum_{k=1}^K\sum_{x_i\in C_k}\|x_i-\bar x_k\|^2.
$$

## 1.4. 层次聚类 vs K-means
<!-- bilingual-en:start -->
*1.4. Hierarchical Clustering versus K-means*
<!-- bilingual-en:end -->

| 问题 | 层次聚类 | K-means |
|---|---|---|
| 是否预设类别数 | 不一定 | 需要 $K$ |
| 输出 | 树状结构 | 固定 $K$ 类 |
| 是否会回退 | 合并后通常不回退 | 每轮可重新分配 |
| 对初始值敏感 | 较少 | 明显 |
| 适合 | 看层级关系 | 快速分成给定类数 |
<!-- bilingual-en:start -->
| Question | Hierarchical clustering | K-means |
|---|---|---|
| Must the number of clusters be set in advance? | Not necessarily | Yes; $K$ is required |
| Output | A tree structure | A fixed set of $K$ clusters |
| Can assignments be revised? | Merges are normally irreversible | Observations may be reassigned each iteration |
| Sensitivity to initial values | Lower | Substantial |
| Best suited to | Examining hierarchical relationships | Quickly producing a specified number of clusters |
<!-- bilingual-en:end -->

## 1.5. 关联卡片
<!-- bilingual-en:start -->
*1.5. Related Cards*
<!-- bilingual-en:end -->

- [[聚类：层次聚类与 K-means#表示、尺度与距离|Clustering Method Selection]]
- [[聚类：层次聚类与 K-means#层次聚类|Hierarchical Clustering]]
- [[聚类：层次聚类与 K-means#层次聚类|Hierarchical Clustering Procedure]]
- [[聚类：层次聚类与 K-means#层次聚类|Linkage Criterion]]
- [[聚类：层次聚类与 K-means#K-means|K-means Clustering]]
- [[聚类：层次聚类与 K-means#K-means|K-means Algorithm]]

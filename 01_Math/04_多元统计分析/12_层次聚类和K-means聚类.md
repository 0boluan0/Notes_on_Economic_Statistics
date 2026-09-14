# 1. 第12章：层次聚类与 K-means
<!-- bilingual-en:start -->
*1. Chapter 12: Hierarchical Clustering and K-means*
<!-- bilingual-en:end -->

>[!note] 本章主线
> [[聚类]]是无监督分组：没有预先给定类别标签，而是在选定的表示、相异度与算法下探索分组。
> <!-- bilingual-en:start -->
> [[聚类|Clustering]] is unsupervised grouping: no class labels are supplied in advance, and groups are explored under a chosen representation, dissimilarity, and algorithm.
> <!-- bilingual-en:end -->

![[聚类.canvas]]

## 1.1. 聚类前的共同问题
<!-- bilingual-en:start -->
*1.1. Questions Shared by All Clustering Methods*
<!-- bilingual-en:end -->

聚类前先确认四件事：
<!-- bilingual-en:start -->
Before clustering, settle four questions:
<!-- bilingual-en:end -->

1. 一个观察对象是什么，用哪些特征和编码来表示。
2. 各变量如何缩放，是否需要标准化。
3. 距离或相异度如何定义。
4. 希望得到层级结构，还是固定数量的簇。
<!-- bilingual-en:start -->

&nbsp;
**1.** What counts as one observation, and which features and encodings represent it?<br>
**2.** How should variables be scaled, and should they be standardised?<br>
**3.** How is distance or dissimilarity defined?<br>
**4.** Is the goal a hierarchy or a fixed number of clusters?<br>
<!-- bilingual-en:end -->

前三个问题依次对应 [[聚类表示|谁被表示、保留什么信息]]、[[聚类尺度|各变量怎样加权]]和[[聚类距离|什么差异算接近]]，共同定义算法看到的输入几何；还要再检查[[距离须匹配聚类算法|数值上可计算的距离是否与算法目标和更新规则相容]]。第四个问题决定进入层次法还是固定 $K$ 的分区法。四项都先于具体计算；任一项改变，结果都可能成为另一套结构。
<!-- bilingual-en:start -->
The first three questions correspond to [[聚类表示|the observational unit and retained information]], [[聚类尺度|feature weighting]], and [[聚类距离|the meaning of proximity]]; together they define the input geometry seen by the algorithm. One must also check whether a numerically available distance is [[距离须匹配聚类算法|compatible with the algorithm's objective and updates]]. The fourth question chooses between a hierarchical route and a fixed-$K$ partitioning route. All four precede the calculation, and changing any one can produce a different structure.
<!-- bilingual-en:end -->

>[!attention] 尺度问题
> 如果一个变量以“万元”为单位，另一个变量以“百分比”为单位，直接用欧氏距离会让量纲大的变量主导聚类。
> 标准化能消除任意计量单位的支配，却同时把“一列一个标准差”设为同等重要；若绝对量级本来有实际含义，它也不自动更正确。
> <!-- bilingual-en:start -->
> If one variable is measured in tens of thousands of currency units and another in percentages, using Euclidean distance directly will allow the larger-scale variable to dominate the clustering.
> Standardisation removes domination by arbitrary measurement units, but it also treats one standard deviation in each feature as equally important. It is not automatically preferable when absolute magnitude is substantively meaningful.
> <!-- bilingual-en:end -->

## 1.2. 层次聚类（Hierarchical Clustering）
<!-- bilingual-en:start -->
*1.2. Hierarchical Clustering*
<!-- bilingual-en:end -->

[[凝聚层次聚类]]给出下面四步的共同算法边界：合并是贪心且不可逆的，输出是完整层级；具体分区还要另做树切。
<!-- bilingual-en:start -->
[[凝聚层次聚类|Agglomerative hierarchical clustering]] supplies the common boundary for the four steps below: merges are greedy and irreversible, and the output is a hierarchy whose flat partition still requires a tree cut.
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

&nbsp;
**1.** Begin with each observation in its own cluster.<br>
**2.** Compute the distances between all clusters.<br>
**3.** Merge the two closest clusters.<br>
**4.** Repeat until all observations have been merged into one tree.<br>
<!-- bilingual-en:end -->

### 1.2.1. Linkage：簇间距离
<!-- bilingual-en:start -->
*1.2.1. Linkage: Distance Between Clusters*
<!-- bilingual-en:end -->

本章需要区分四种 linkage：
<!-- bilingual-en:start -->
This chapter distinguishes four linkage rules:
<!-- bilingual-en:end -->

| linkage | 定义 | 直觉 |
|---|---|---|
| single linkage | 两簇最近样本点之间的距离 | 容易形成链状结构 |
| complete linkage | 两簇最远样本点之间的距离 | 偏好紧凑簇 |
| average linkage | 两簇样本点两两距离的平均值 | 折中 |
| Ward linkage | 合并后总组内平方和的增量 | 偏好 Euclidean 空间中的紧凑簇 |
<!-- bilingual-en:start -->
| Linkage | Definition | Intuition |
|---|---|---|
| single linkage | Distance between the closest pair of points in the two clusters | Tends to produce chains |
| complete linkage | Distance between the farthest pair of points in the two clusters | Favours compact clusters |
| average linkage | Mean of all pairwise distances between points in the two clusters | A compromise |
| Ward linkage | Increase in total within-cluster sum of squares after the merge | Favours compact groups in Euclidean geometry |
<!-- bilingual-en:end -->

四种准则的精确定义分别见 [[单连接]]、[[完全连接]]、[[平均连接]]与[[Ward连接]]。Average linkage 不是质心距离；Ward 的平方 Euclidean 几何见 [[Ward平方欧氏边界]]，软件纵轴和 method 口径见 [[Ward高度口径]]。
<!-- bilingual-en:start -->
See [[单连接|single linkage]], [[完全连接|complete linkage]], [[平均连接|average linkage]], and [[Ward连接|Ward linkage]] for their exact definitions. Average linkage is not centroid distance. Ward's squared-Euclidean geometry is treated in [[Ward平方欧氏边界]], and its software method and height conventions in [[Ward高度口径]].
<!-- bilingual-en:end -->

>[!note] 读树状图
> dendrogram 的纵轴表示连接规则和软件口径特定的合并高度，不一定是原始点距离。对单调树，切割高度越低通常簇越多，切割高度越高通常簇越少；具体分区见[[树状图切割]]。横向叶序可以在不改变同一棵树的情况下翻转，不能当作额外距离证据，见[[树状图叶序]]。
> <!-- bilingual-en:start -->
> A dendrogram's vertical axis is a linkage- and implementation-specific merge height, not necessarily an original point-to-point distance. For a monotone tree, a lower cut usually gives more clusters and a higher cut fewer; see [[树状图切割|dendrogram cutting]] for the resulting partition. Horizontal leaf order can be flipped without changing the tree and is not additional distance evidence; see [[树状图叶序]].
> <!-- bilingual-en:end -->

## 1.3. K-means 聚类（K-means Clustering）
<!-- bilingual-en:start -->
*1.3. K-means Clustering*
<!-- bilingual-en:end -->

先把七个对象分开：[[K-means目标]]定义平方 Euclidean 损失，[[Lloyd算法]]定义交替求解步骤，[[Lloyd局部收敛]]区分目标不增、固定点与全局最优；[[K-means初始化]]定义一次运行的起点，[[K-means++]]是一种具体播种方法，[[K-means多启动]]重复整次求解并按 inertia 选择，[[K-means空簇]]则是均值更新无定义时的实现边界。
<!-- bilingual-en:start -->
Keep seven objects separate: the [[K-means目标|K-means objective]] defines the squared-Euclidean loss, [[Lloyd算法|Lloyd's algorithm]] the alternating steps, and [[Lloyd局部收敛]] the difference between a non-increasing objective, a fixed point, and a global optimum. [[K-means初始化|Initialisation]] defines one run's starting centres, [[K-means++]] one seeding method, [[K-means多启动|multiple starts]] repeated complete solves selected by inertia, and [[K-means空簇]] the implementation boundary when a mean cannot be updated.
<!-- bilingual-en:end -->

K-means 的类别数 $K$ 通常由题目给定。
<!-- bilingual-en:start -->
The number of K-means clusters, $K$, is usually specified by the question.
<!-- bilingual-en:end -->

算法流程：
<!-- bilingual-en:start -->
The algorithm proceeds as follows:
<!-- bilingual-en:end -->

1. 初始化 $K$ 个中心。
2. 把每个样本分配给最近的中心。
3. 对每个非空簇，用簇内样本均值更新中心（centroid）。
4. 重复步骤 2–3，直到分配不再变化，或实现采用的容差、迭代上限等停止条件触发。
<!-- bilingual-en:start -->

&nbsp;
**1.** Initialise $K$ centres.<br>
**2.** Assign each observation to its nearest centre.<br>
**3.** For every non-empty cluster, update its centre to the mean of its assigned observations.<br>
**4.** Repeat steps 2–3 until assignments stop changing or an implementation-specific tolerance or iteration limit is reached.<br>
<!-- bilingual-en:end -->

目标函数为
<!-- bilingual-en:start -->
The objective function is
<!-- bilingual-en:end -->
$$
\min_{C_1,\ldots,C_K}
\sum_{k=1}^K\sum_{x_i\in C_k}\|x_i-\bar x_k\|^2.
$$

每次最近中心分配和均值更新都不会增加这个目标，但只在明确的非空簇与并列处理条件下得到教科书式 fixed point；软件还可能因 tolerance 或迭代上限提前停止。[[Lloyd局部收敛|目标不增不等于全局最优]]，因此应报告初始化、[[K-means多启动|多启动]]、空簇规则与停止条件。
<!-- bilingual-en:start -->
Nearest-centre assignment and mean updates do not increase this objective, but the textbook fixed-point conclusion needs explicit non-empty-cluster and tie-handling conditions; software may also stop at a tolerance or iteration limit. [[Lloyd局部收敛|A non-increasing objective is not a global-optimum guarantee]], so initialisation, restarts, empty-cluster rules, and stopping conditions should be reported.
<!-- bilingual-en:end -->

## 1.4. 层次聚类 vs K-means
<!-- bilingual-en:start -->
*1.4. Hierarchical Clustering versus K-means*
<!-- bilingual-en:end -->

| 问题 | 层次聚类 | K-means |
|---|---|---|
| 是否预设类别数 | 不一定 | 需要 $K$ |
| 输出 | 树状结构 | 固定 $K$ 类 |
| 是否会回退 | 合并后通常不回退 | 每轮可重新分配 |
| 初始与路径依赖 | 给定 dissimilarity、linkage 与并列规则后无随机初始化；仍依赖预处理、样本与 ties | 明显依赖初始中心与局部解 |
| 适合 | 看层级关系 | 快速分成给定类数 |
<!-- bilingual-en:start -->
| Question | Hierarchical clustering | K-means |
|---|---|---|
| Must the number of clusters be set in advance? | Not necessarily | Yes; $K$ is required |
| Output | A tree structure | A fixed set of $K$ clusters |
| Can assignments be revised? | Merges are normally irreversible | Observations may be reassigned each iteration |
| Initialisation and path dependence | No random initialisation once dissimilarity, linkage, and tie rules are fixed; still sensitive to preprocessing, sample, and ties | Strong dependence on initial centres and local solutions |
| Best suited to | Examining hierarchical relationships | Quickly producing a specified number of clusters |
<!-- bilingual-en:end -->

## 1.5. 簇数、验证与解释
<!-- bilingual-en:start -->
*1.5. Cluster Count, Validation, and Interpretation*
<!-- bilingual-en:end -->

层次树切割、elbow、[[轮廓系数]]和 gap statistic 都只在各自准则下给出候选分辨率。[[簇数选择]]是把这些候选与稳定性、外部证据和领域用途并列的过程；它为什么不是一个自动吐出真实 $K$ 的按钮，见 [[簇数不是自动真值]]。
<!-- bilingual-en:start -->
Tree cuts, elbow, the [[轮廓系数|silhouette coefficient]], and the gap statistic offer candidate resolutions under different criteria. [[簇数选择|Choosing the number of clusters]] combines them with stability, external evidence, and substantive purpose; [[簇数不是自动真值|the cluster count is not an automatic truth]].
<!-- bilingual-en:end -->

选出一个分区后，[[聚类验证]]还要区分内部几何、随机初值、重采样、外部标签和实际效用。即使结果稳定紧密，[[聚类解释边界]]也禁止直接把它写成自然种类或因果机制；若同一变量既用于造簇又用于普通组间检验，还要遵守 [[聚类后推断]] 的选择性推断边界。
<!-- bilingual-en:start -->
After selecting a partition, [[聚类验证|cluster validation]] separates internal geometry, random starts, resampling, external labels, and practical utility. Even stable compact groups remain subject to the [[聚类解释边界|interpretation boundary]], and using the same variables both to create clusters and in ordinary between-group tests invokes the [[聚类后推断|post-clustering inference]] problem.
<!-- bilingual-en:end -->

## 1.6. 知识地图
<!-- bilingual-en:start -->
*1.6. Knowledge Map*
<!-- bilingual-en:end -->

- **共同入口：** [[聚类]]
- **输入定义：** [[聚类表示]]、[[聚类尺度]]、[[聚类距离]]、[[距离须匹配聚类算法]]
- **层次分支：** [[凝聚层次聚类]]、[[单连接]]、[[完全连接]]、[[平均连接]]、[[Ward连接]]、[[Ward平方欧氏边界]]、[[Ward高度口径]]、[[树状图切割]]、[[树状图叶序]]
- **K-means 分支：** [[K-means目标]]、[[Lloyd算法]]、[[Lloyd局部收敛]]、[[K-means初始化]]、[[K-means++]]、[[K-means多启动]]、[[K-means空簇]]
- **选择与证据：** [[轮廓系数]]、[[簇数选择]]、[[簇数不是自动真值]]、[[聚类验证]]、[[聚类解释边界]]、[[聚类后推断]]

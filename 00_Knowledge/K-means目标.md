---
aliases:
  - "给定 K 个非空簇，K-means 以簇均值为中心最小化组内平方 Euclidean 距离和"
  - "K-means minimises squared Euclidean distance around cluster means"
student_os: knowledge-atom
atom_id: STAT-CL-010
atom_set: cluster-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[聚类距离]]"
  - "[[聚类尺度]]"
related:
  - "[[距离须匹配聚类算法]]"
leads_to:
  - "[[Lloyd算法]]"
  - "[[K-means++]]"
  - "[[簇数选择]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# 给定 K 个非空簇，K-means 以簇均值为中心最小化组内平方 Euclidean 距离和
<!-- bilingual-en:start -->
*K-means minimises within-cluster squared Euclidean distance around cluster means*
<!-- bilingual-en:end -->

> [!summary] 原子目标
> 给定簇数 $K$，K-means 在非空分区和中心之间寻找组内平方和（WCSS，也称 inertia）尽可能小的组合。平方 Euclidean 损失决定了每个固定簇的最优中心是算术均值，也决定了该方法偏好 Euclidean 空间中紧凑、近似凸的分组。
> <!-- bilingual-en:start -->
> For a fixed $K$, K-means seeks a non-empty partition and centres with low within-cluster sum of squared Euclidean distances. Squared loss makes the arithmetic mean the optimal centre and favours compact, approximately convex groups.
> <!-- bilingual-en:end -->

## 目标函数拥有非空分区与均值中心

设 $x_1,\ldots,x_n\in\mathbb R^p$，$C_1,\ldots,C_K$ 是 ${1,\ldots,n}$ 的非空、不相交分区。K-means 目标为

$$
W(C,\mu)
=\sum_{k=1}^{K}\sum_{i\in C_k}\lVert x_i-\mu_k\rVert_2^2.
$$

对固定的非空簇 $C_k$，令

$$
\bar x_k=\frac{1}{|C_k|}\sum_{i\in C_k}x_i.
$$

则对任意中心 $m$，

$$
\sum_{i\in C_k}\lVert x_i-m\rVert_2^2
=\sum_{i\in C_k}\lVert x_i-\bar x_k\rVert_2^2
+|C_k|\lVert m-\bar x_k\rVert_2^2.
$$

第二项非负，所以 $m=\bar x_k$ 是 minimizer。把每个中心代回簇均值后，目标就是 WCSS；scikit-learn 的 `inertia_` 是样本到其最近中心的平方距离和（有样本权重时为加权和）。

## 目标函数也规定分区几何

固定中心时，最近中心分配把特征空间划成 Euclidean Voronoi cells。因此这个目标倾向表达紧凑的 Euclidean 分区。“算术均值”和“平方 Euclidean 损失”是同一定义的两面；为什么不能只换 Manhattan、Jaccard 或 correlation dissimilarity 却保留原解释，单独见 [[距离须匹配聚类算法]]。[[聚类尺度]]则决定每一维进入这个平方和的权重。

> [!question]- 自检
> 为什么在固定簇成员后，把中心换成簇内任意一个实际样本通常不会优于算术均值？这是否意味着均值必须是实际样本？
>
> **答案：** 平方和分解显示，任何偏离簇均值的中心都会额外增加 $|C_k|\lVert m-\bar x_k\rVert^2$。均值无需是实际样本；若中心必须取自样本，问题会变成 medoid 类目标，而不是标准 K-means。

## 来源与核验

- [scikit-learn, K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)：核对 inertia/WCSS、簇均值中心、最近中心分配、Voronoi 解释及凸、各向同性偏好。
- [scikit-learn, `KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)：核对 `inertia_` 为样本到最近中心的加权平方距离和。
- [Penn State STAT 505, Lesson 14](https://online.stat.psu.edu/stat505/Lesson14)：核对 Euclidean 距离、簇中心与 K-means 的课程定义。

> [!success] 审核状态
> 本卡已通过独立内容审核，状态为 `source-checked`；掌握状态仍为 `unassessed`。

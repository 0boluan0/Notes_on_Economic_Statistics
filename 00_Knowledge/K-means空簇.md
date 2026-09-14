---
aliases:
  - "K-means 分配产生空簇时簇均值未定义，任何重置、迁移或保留规则都是目标函数之外的实现选择"
  - "An empty K-means cluster has no mean and requires an implementation-specific repair"
student_os: knowledge-atom
atom_id: STAT-CL-023
atom_set: cluster-analysis
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Lloyd算法]]"
related:
  - "[[K-means初始化]]"
  - "[[K-means++]]"
  - "[[Lloyd局部收敛]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# K-means 分配产生空簇时簇均值未定义，任何重置、迁移或保留规则都是目标函数之外的实现选择
<!-- bilingual-en:start -->
*When K-means assignment creates an empty cluster, its mean is undefined; every reset, relocation, or retention rule is an implementation choice beyond the objective itself*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Lloyd 更新需要把簇 $C_k$ 的中心设为 $|C_k|^{-1}\sum_{i\in C_k}x_i$。若分配步骤没有把任何样本分给某中心，$|C_k|=0$，这个均值就没有定义。K-means 目标只要求最终分区非空，不自动告诉迭代中如何修复空簇。
> <!-- bilingual-en:start -->
> Lloyd updates require the centre of cluster $C_k$ to be $|C_k|^{-1}\sum_{i\in C_k}x_i$. If no observation is assigned to a centre, then $|C_k|=0$ and the mean is undefined. The K-means objective requires a non-empty final partition but does not itself specify how an iteration should repair an empty cluster.
> <!-- bilingual-en:end -->

## 常见修复规则解决的是实现问题

可能的处理包括保留旧中心、重新随机播种、拆分一个大簇，或把离现有中心最远的样本迁入空簇。这些规则可以给出不同后续路径；一个库“没有报错”不表示空簇在数学上自动消失了。

若 $K$ 大于非重复样本位置的数量，要求 $K$ 个拥有不同几何支撑的非空簇本身就可能不合理。空簇因此不只是软件细节，也是检查 $K$、重复点、尺度与初始化的诊断信号。

## 修复后要重新检查收敛声明

教科书式的 Lloyd 单调与有限终止结论对非空簇和 tie handling 有条件。加入一个迁移规则后，应依该规则重新核对目标是否不增和停止条件，不能只引用无空簇版本的 [[Lloyd局部收敛]] 结论。

> [!question]- 自检
> 分配后某个中心没有任何样本。能否继续用“该簇样本的均值”更新它？
>
> **答案：** 不能。空集合没有这个算术均值。必须明确选择一个额外修复规则，记录实现口径，并重新检查它对优化路径的影响。

## 来源与核验

- [scikit-learn, K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)：核对均值更新、局部优化与重启动语境。
- [scikit-learn, `KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)：核对 K-means 迭代参数与初始化口径。
- [scikit-learn, empty-cluster relocation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_k_means_common.pyx)：核对该库把远点迁入空簇的具体实现选择。

> [!success] 审核状态
> 本卡已通过内容核验，状态为 `source-checked`；掌握状态仍为 `unassessed`。

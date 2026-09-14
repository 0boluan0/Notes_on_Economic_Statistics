---
aliases:
  - "Lloyd 迭代使 K-means 目标单调不增，但固定点、有限终止和全局最优是三个不同结论"
  - "Lloyd's non-increasing objective does not imply global optimality"
student_os: knowledge-atom
atom_id: STAT-CL-021
atom_set: cluster-analysis
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Lloyd算法]]"
related:
  - "[[K-means空簇]]"
leads_to:
  - "[[K-means初始化]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# Lloyd 迭代使 K-means 目标单调不增，但固定点、有限终止和全局最优是三个不同结论
<!-- bilingual-en:start -->
*Lloyd iterations make the K-means objective non-increasing, but reaching a fixed point, terminating finitely, and finding a global optimum are three different conclusions*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 在簇不为空时，最近中心分配对固定中心不会增加 WCSS，簇均值更新对固定分配也不会增加 WCSS。所以每轮有 $W^{(t+1)}\le W^{(t)}$。但不同初始中心可通向不同 fixed points，单调不增不保证达到全局最小 WCSS。
> <!-- bilingual-en:start -->
> With non-empty clusters, nearest-centre assignment does not increase WCSS for fixed centres, and cluster-mean updates do not increase WCSS for a fixed assignment. Thus $W^{(t+1)}\le W^{(t)}$. Different starting centres can nevertheless lead to different fixed points, so a non-increasing objective is not a global-optimum guarantee.
> <!-- bilingual-en:end -->

## 有限终止需要把并列和空簇说清楚

如果每簇非空，等距样本使用不会来回改派的固定规则，而且分配一旦不变就停止，有限个可能分区给出教科书式的有限终止论证。“目标不增”不等于每轮严格下降：ties、重复点和 fixed point 都可以使目标保持不变。

若分配产生空簇，均值更新未定义，必须加入 [[K-means空簇|空簇修复规则]]；原来的论证不能不加检查地继续使用。

## 软件停止不一定等到 fixed point

实际实现常在中心移动小于 `tol` 或迭代达到 `max_iter` 时停止。这只表示触发了软件的停止条件，不自动表示分配已完全不变。scikit-learn 还说明，若在完全收敛前停止，`cluster_centers_` 与最终重分配的 `labels_` 可能不满足“每个中心就是对应标签样本的均值”。

> [!question]- 自检
> 某次运行的 WCSS 在十轮中从未增加。能否据此断言已找到全局最优分区？
>
> **答案：** 不能。这只证明交替更新没有把该目标推高。运行可能停在依赖初值的 fixed point，也可能因容差或迭代上限提前停止。

## 来源与核验

- [scikit-learn, K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)：核对交替更新、局部最小与初值依赖。
- [scikit-learn, `KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)：核对 `tol`、`max_iter` 及提前停止时 centres/labels 的限定。
- Schubert (2023), [*Stop using the elbow criterion for k-means*](https://arxiv.org/abs/2212.12189)：交叉核对固定 tie handling、有限分区和有限终止的条件。

> [!success] 审核状态
> 本卡已通过内容核验，状态为 `source-checked`；掌握状态仍为 `unassessed`。

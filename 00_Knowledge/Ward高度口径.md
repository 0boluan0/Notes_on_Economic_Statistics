---
aliases:
  - "Ward 树的合并高度取决于准则和实现口径，不同软件的纵轴数值不能直接比较"
  - "Ward dendrogram heights depend on the criterion and implementation convention"
student_os: knowledge-atom
atom_id: STAT-CL-027
atom_set: cluster-analysis
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Ward连接]]"
related:
  - "[[Ward平方欧氏边界]]"
  - "[[树状图切割]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# Ward 树的合并高度取决于准则和实现口径，不同软件的纵轴数值不能直接比较
<!-- bilingual-en:start -->
*Ward dendrogram merge heights depend on the criterion and implementation convention, so vertical-axis values from different software cannot be compared directly*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Ward 树上的 height 是该实现为本次合并记录的准则值。它可能是 WSS 增量、其平方根或与之成比例的数值；输入距离是否已平方也会改变递推。因此同名“Ward”不足以确定纵轴单位。
> <!-- bilingual-en:start -->
> A Ward tree's height is the criterion value recorded by that implementation for a merge. It may be a WSS increment, its square root, or a proportional quantity, and the recursion also depends on whether input distances are squared. The name “Ward” alone therefore does not determine the vertical-axis unit.
> <!-- bilingual-en:end -->

## 先核对算法，再核对纵轴

R 的 `hclust(method = "ward.D2")` 实现 Ward (1963) 的准则，并在更新前对 dissimilarities 平方；`ward.D` 不实现同一准则。这不只是画图单位不同：方法选项可以改变合并顺序和整棵树。

即使两个实现都与 Ward 准则相容，树上显示的数值也可能只保留了同一合并顺序的单调变换。切树时可以在各自树内使用 height，但不能在没有口径换算时把“高度 5”当作跨软件的同一物理量。

## 最小可复现报告

至少记录：软件与版本、method 名、输入是原始观测还是 distance matrix、distance 是否已平方，以及 height 的定义。几何本身能否承担 WSS 解释，另见 [[Ward平方欧氏边界]]。

> [!question]- 自检
> 两个软件都输出名为 Ward 的 dendrogram，某次合并的 height 分别是 4 和 2。能否立即断言第一个合并的 WSS 代价是第二个的两倍？
>
> **答案：** 不能。先要确认两者是否实现同一 Ward 准则，再确认输入是否平方以及 height 是 $\Delta$、$\sqrt{\Delta}$ 还是其他比例量。

## 来源与核验

- [R, `hclust`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/hclust.html)：核对 `ward.D` 与 `ward.D2` 的官方区分，以及 height 是 method-specific criterion value。
- Murtagh and Legendre (2014), [*Ward's Hierarchical Agglomerative Clustering Method*](https://doi.org/10.1007/s00357-014-9161-z)：核对 Ward1/Ward2、距离平方和 dendrogram height 口径。

> [!success] 审核状态
> 本卡已通过内容核验，状态为 `source-checked`；掌握状态仍为 `unassessed`。

---
aliases:
  - "Ward 连接的组内平方和解释要求平方 Euclidean 几何，任意相异度不能直接继承这一解释"
  - "Ward's within-cluster sum-of-squares interpretation requires squared-Euclidean geometry"
student_os: knowledge-atom
atom_id: STAT-CL-020
atom_set: cluster-analysis
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Ward连接]]"
  - "[[距离须匹配聚类算法]]"
related:
  - "[[聚类尺度]]"
  - "[[树状图切割]]"
  - "[[Ward高度口径]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# Ward 连接的组内平方和解释要求平方 Euclidean 几何，任意相异度不能直接继承这一解释
<!-- bilingual-en:start -->
*Ward's within-cluster sum-of-squares interpretation requires squared-Euclidean geometry; an arbitrary dissimilarity cannot inherit that interpretation directly*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> Ward 的合并增量使用簇均值和平方 Euclidean 距离的分解。把 Jaccard、correlation dissimilarity 或任意距离矩阵交给名为“Ward”的实现，即使软件返回一棵树，也不自动使这棵树具有原数据 WSS 增量解释。
> <!-- bilingual-en:start -->
> Ward's merge increment relies on the decomposition of cluster means and squared Euclidean distances. Passing Jaccard, correlation dissimilarity, or an arbitrary distance matrix to an implementation labelled “Ward” may return a tree, but it does not automatically give that tree a within-cluster sum-of-squares interpretation for the original data.
> <!-- bilingual-en:end -->

## 相容 embedding 也要核对输入口径

某些 dissimilarity 可以在特定处理后写成 Euclidean 或 squared-Euclidean 距离。这时也必须说明对象已映射到哪个新表示，并核对实现把输入当作距离还是平方距离。否则，“先给距离再由库平方”与“先给平方距离再被库平方”会产生不同准则。

> [!question]- 自检
> 分析者用 $1-r$ 构造 correlation dissimilarity，随后选择 Ward linkage。能否不经核对就说树在每步最小化原始数据的 WSS 增量？
>
> **答案：** 不能。对经中心化和单位化的向量，$1-r$ 可与 squared Euclidean distance 成比例；但还要核对 embedding、比例以及实现是否再对输入平方。可能成立的是新表示下的 WSS 解释，不是无条件的原数据解释。

## 来源与核验

- Ward (1963), [*Hierarchical Grouping to Optimize an Objective Function*](https://doi.org/10.1080/01621459.1963.10500845)：核对逐步优化组内变差准则的原始对象。
- Murtagh and Legendre (2014), [*Ward's Hierarchical Agglomerative Clustering Method*](https://doi.org/10.1007/s00357-014-9161-z)：核对 squared-Euclidean 条件和 Ward 准则的输入口径。
- [[Ward高度口径]]：单独承载软件 method、输入口径和 dendrogram 纵轴数值的比较边界。

> [!success] 审核状态
> 本卡已通过内容核验，状态为 `source-checked`；掌握状态仍为 `unassessed`。

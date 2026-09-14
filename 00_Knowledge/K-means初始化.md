---
aliases:
  - "K-means 初始化是在 Lloyd 迭代前选定 K 个起始中心，不同起点可以通向不同局部结果"
  - "K-means initialisation chooses the K starting centres for Lloyd iterations"
student_os: knowledge-atom
atom_id: STAT-CL-012
atom_set: cluster-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Lloyd算法]]"
related:
  - "[[K-means空簇]]"
  - "[[Lloyd局部收敛]]"
leads_to:
  - "[[K-means++]]"
  - "[[K-means多启动]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# K-means 初始化是在 Lloyd 迭代前选定 K 个起始中心，不同起点可以通向不同局部结果
<!-- bilingual-en:start -->
*K-means initialisation chooses the $K$ starting centres for Lloyd iterations, and different starts can lead to different local results*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> K-means 初始化是在一次 Lloyd 迭代开始前选定 $K$ 个起始中心。因为同一目标可以有不同局部结果，这组中心会决定本次求解从哪里出发。
> <!-- bilingual-en:start -->
> K-means initialisation selects the $K$ starting centres before one Lloyd run begins. Because the same objective can have different local results, those centres determine where this run starts its search.
> <!-- bilingual-en:end -->

## 起点是一次运行的输入

随机初始化可以从样本中选 $K$ 个起始中心；[[K-means++]] 则按到已选中心的最近平方距离进行播种。两者都只在规定“一次运行从哪组中心开始”。

固定随机种子可以让这一次初始化可复现，但不会使其局部结果更接近全局最优。

## 一次初始化不等于多启动

`init` 规定一次运行怎样选起点；`n_init` 或等价设置则规定独立运行多少次。后者属于独立的 [[K-means多启动]] 策略，不再由本卡同时承担。

若一轮分配后某簇没有任何样本，问题已从“怎样选起点”转为“怎样定义未定义的均值更新”，应单独进入 [[K-means空簇]]。

> [!question]- 自检
> 两次运行使用相同数据、相同随机初始化方法，但随机种子不同。为什么最终分区也可能不同？
>
> **答案：** 随机种子可以产生不同起始中心；Lloyd 迭代便从不同位置出发，并可能进入不同局部结果。

## 来源与核验

- [scikit-learn, `KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)：核对初始化方法、`n_init` 使用多次运行中最低 inertia 的结果，以及随机种子的作用。
- [[K-means++]]：单独承载 $D^2$ 播种定义。
- [[K-means多启动]]：单独承载独立运行多次并选择最低 inertia 结果的策略。
- [[K-means空簇]]：单独承载空簇均值未定义与实现修复口径。

> [!success] 审核状态
> 本卡已通过独立内容审核，状态为 `source-checked`；掌握状态仍为 `unassessed`。

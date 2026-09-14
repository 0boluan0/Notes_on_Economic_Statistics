---
aliases:
  - "Lloyd 算法是交替执行最近中心分配和簇均值更新来求解 K-means 目标的迭代算法"
  - "Lloyd's algorithm alternates nearest-centre assignment and cluster-mean updates"
student_os: knowledge-atom
atom_id: STAT-CL-011
atom_set: cluster-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[K-means目标]]"
leads_to:
  - "[[Lloyd局部收敛]]"
  - "[[K-means初始化]]"
  - "[[K-means空簇]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# Lloyd 算法是交替执行最近中心分配和簇均值更新来求解 K-means 目标的迭代算法
<!-- bilingual-en:start -->
*Lloyd's algorithm solves the K-means objective by alternating nearest-centre assignment and cluster-mean updates*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> Lloyd 算法从 $K$ 个初始中心出发，先把每个样本分给最近中心，再把每个非空簇的中心更新为簇内均值，反复两步直到达停止条件。它是 K-means 目标的一种交替优化算法，不是目标函数本身。
> <!-- bilingual-en:start -->
> Lloyd's algorithm starts from $K$ centres, alternates nearest-centre assignment with cluster-mean updates, and stops when its specified stopping condition is met. It is an alternating solver for the K-means objective rather than the objective itself.
> <!-- bilingual-en:end -->

## 两步各自精确优化一个部分

给定第 $t$ 轮中心 $\mu_1^{(t)},\ldots,\mu_K^{(t)}$：

1. **Assignment step**

$$
c_i^{(t+1)}\in\arg\min_{1\le k\le K}
\lVert x_i-\mu_k^{(t)}\rVert_2^2.
$$

每个样本选择最近中心，因此在中心固定时不会增加 [[K-means目标|WCSS]]。

2. **Update step**

$$
\mu_k^{(t+1)}
=\frac{1}{|C_k^{(t+1)}|}
\sum_{i\in C_k^{(t+1)}}x_i,
\qquad |C_k^{(t+1)}|>0.
$$

算术均值最小化固定簇内的平方距离和，因此在分配固定时也不会增加 WCSS。一次完整迭代遂满足

$$
W^{(t+1)}\le W^{(t)}.
$$

## 收敛与空簇是算法之外的独立边界

上述两步定义了求解器；它们能支持多强的收敛结论，单独见 [[Lloyd局部收敛]]。若分配步骤产生空簇，均值更新没有定义，则进入独立的 [[K-means空簇]] 处理问题。

> [!question]- 自检
> Lloyd 算法为什么要在“分给最近中心”之后，再把每个中心改成当前簇均值？
>
> **答案：** 第一步在中心固定时，为每个样本选择平方距离最小的中心；第二步在分配固定时，用簇均值最小化该簇的平方距离和。两步分别精确优化一部分。

## 来源与核验

- [scikit-learn, K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)：核对最近中心分配和簇均值更新的交替流程。
- [Penn State STAT 505, Lesson 14](https://online.stat.psu.edu/stat505/Lesson14)：核对 K-means 的分配、中心更新与迭代步骤。
- [[Lloyd局部收敛]]：单独承载 tie handling、提前停止与全局最优边界。

> [!success] 审核状态
> 本卡已通过独立内容审核，状态为 `source-checked`；掌握状态仍为 `unassessed`。

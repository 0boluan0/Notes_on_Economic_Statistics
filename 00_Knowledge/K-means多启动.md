---
aliases:
  - "K-means 多启动是从多组初始中心独立运行求解器并保留目标值最低的结果"
  - "K-means multiple starts run the solver from several initial centre sets and retain the lowest-objective result"
student_os: knowledge-atom
atom_id: STAT-CL-025
atom_set: cluster-analysis
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[K-means初始化]]"
  - "[[Lloyd局部收敛]]"
related:
  - "[[K-means++]]"
leads_to:
  - "[[簇数选择]]"
  - "[[聚类验证]]"
part_of:
  - "[[聚类.canvas|聚类]]"
---

# K-means 多启动是从多组初始中心独立运行求解器并保留目标值最低的结果
<!-- bilingual-en:start -->
*K-means multiple starts run the solver independently from several initial centre sets and retain the result with the lowest objective value*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 多启动先生成多组起始中心，对每一组完整运行一次 K-means 求解器，再在这些已尝试的结果中保留 inertia 最小的一次。它重复的是整次求解，不是在一次 Lloyd 迭代里多选几个中心。
> <!-- bilingual-en:start -->
> Multiple starts generate several initial centre sets, run a complete K-means solver from each set, and retain the tried result with the smallest inertia. They repeat the whole solve rather than choosing several centres inside one Lloyd iteration.
> <!-- bilingual-en:end -->

## 比较的必须是同一目标

设第 $r$ 次运行的最终组内平方和为 $W_r$。若所有运行都使用相同的数据表示、尺度、$K$ 和停止规则，则选择

$$
r^*\in\arg\min_r W_r
$$

是在已尝试路径中选择对 [[K-means目标]] 最好的一次。若每次运行同时改了尺度、变量或 $K$，inertia 就不再是只比较不同起点。

## 多启动改善搜索覆盖，不提供全局证明

一组差起点可能通向较高 inertia 的局部结果；多启动通过尝试更多起点降低这种风险。但它只在有限次已尝试的运行中选最好者，不能排除所有运行都错过更好分区。

报告应保留初始化方法、启动次数、随机种子、每次停止规则和被选运行的 inertia。某些库的 `n_init="auto"` 会根据 `init` 选项自动决定运行次数，因此不应把 `auto` 自行解读为“已做充分多启动”。

> [!question]- 自检
> 从 20 组起点运行 K-means，保留 inertia 最小的结果。能否据此声称它是全局最优分区？
>
> **答案：** 不能。这只说明它是二十次已尝试运行中 inertia 最小的一次；未尝试的起点仍可能通向更低目标值。

## 来源与核验

- [scikit-learn, `KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)：核对 `n_init` 以不同初始中心独立运行并按 inertia 选择结果，以及 `auto` 的次数依赖 `init` 设置。
- [[Lloyd局部收敛]]：复用不同起点可进入不同局部结果的求解边界。

> [!success] 审核状态
> 本卡已通过内容核验，状态为 `source-checked`；掌握状态仍为 `unassessed`。

---
aliases:
  - "部分固定效应 MANOVA 违背情形下 Pillai–Bartlett trace 往往较稳健但这种优势不是无条件定理"
  - Pillai-Bartlett trace is often comparatively robust in some fixed-effects MANOVA settings but has no universal guarantee
  - Pillai trace robustness boundary
student_os: knowledge-atom
atom_id: STAT-MAN-006
atom_set: manova
atom_type: robustness-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[MANOVA四种统计量]]"
  - "[[MANOVA经典假设]]"
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
related:
  - "[[MANOVA检验校准]]"
  - "[[多响应联合显著边界]]"
---

# 部分固定效应 MANOVA 违背情形下 Pillai–Bartlett trace 往往较稳健但这种优势不是无条件定理
<!-- bilingual-en:start -->
*Pillai–Bartlett trace is often comparatively robust in some fixed-effects MANOVA settings, but this advantage is not a universal theorem*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 在 Olson 比较的若干固定效应 MANOVA 仿真中，面对非正态和组间协方差异质，Pillai–Bartlett trace 通常比其他经典统计量更能维持合理的第一类错误表现，Roy's greatest root 往往较差。R 的 summary.manova 也据既有比较研究把 Pillai 设为默认并推荐优先考虑。这里的结论是**有条件的经验比较**，不是“假设失效时 Pillai 仍自动有效”。
> <!-- bilingual-en:start -->
> Comparative studies and software guidance often favour Pillai's trace in some fixed-effects settings with nonnormality or covariance heterogeneity. They do not turn it into an assumption-free test.
> <!-- bilingual-en:end -->

“较稳健”首先是相对于某个违背程度、样本量、组间平衡和候选统计量而言，通常关注第一类错误控制与功效的折中。换一组数据生成机制，排序可能改变；小样本、严重失衡、厚尾、离群点或协方差差异的组合也可能超出已有比较证据。

Pillai 的选择不能修复下列结构问题：

- 观测行相关，却仍按独立样本计算参考分布；
- 目标对比不可估，或误差 SSCP 因残差自由度不足而秩亏；
- 看完四个 $p$ 值后才挑最小者；
- 把总体联合拒绝扩张为每个响应都显著，或扩张为因果结论。

因此，若研究计划依据相符的证据预先选择 Pillai，可以把它作为较谨慎的经典选项；但仍要核对 [[MANOVA经典假设]]，并在必要时明确采用重抽样、稳健或高维校准。统计量的原始定义见 [[MANOVA四种统计量]]，从统计量到尾概率的区别见 [[MANOVA检验校准]]。

> [!question]- 自检
> 数据明显偏离多元正态时，改报 Pillai 是否足以继续声称经典有限样本 $p$ 值精确？
>
> **答案：** 不足。Pillai 的比较稳健性来自有范围的经验研究；当前设计、样本量、依赖、协方差异质与校准方法仍需单独核对。

## 来源与核验

- [Olson (1974), *Comparative Robustness of Six Tests in Multivariate Analysis of Variance*](https://doi.org/10.1080/01621459.1974.10480224)：核对 Pillai–Bartlett 与 Roy 的比较结论来自特定固定效应仿真，而非无条件定理。
- [R summary.manova documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html)：核对 R 默认使用并推荐 Pillai–Bartlett，同时允许选择其他经典统计量。
- [[MANOVA经典假设]] 与 [[MANOVA检验校准]]：核对统计量选择不会消除误差模型、可估性、秩与参考分布条件。

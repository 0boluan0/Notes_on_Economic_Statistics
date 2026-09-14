---
aliases:
  - "联合 Gaussian 诊断应合并检查边际、投影、Mahalanobis 结构与下游敏感性"
  - "Multivariate Gaussian diagnostic workflow"
  - "多元正态性诊断"
student_os: knowledge-atom
atom_id: STAT-MVN-007
atom_set: multivariate-normal
atom_type: diagnostic-workflow
status: source-checked
mastery_state: unassessed
requires:
  - "[[联合Gaussian]]"
  - "[[Gaussian马氏平方律]]"
related:
  - "[[边际与马氏图不证联合Gaussian]]"
  - "[[多元数据诊断]]"
  - "[[马氏距离]]"
  - "[[异常观测处理原则]]"
part_of:
  - "[[多元正态分布.canvas]]"
---

# 联合 Gaussian 诊断应合并检查边际、投影、Mahalanobis 结构与下游敏感性
<!-- bilingual-en:start -->
*Diagnosing joint Gaussianity should combine marginal, projection, Mahalanobis, and downstream-sensitivity checks*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 联合 Gaussian 诊断是一套互补检查：先核实数据生成与测量，再看边际和多方向投影；在协方差可稳定估计时加入 Mahalanobis 结构；最后检查下游结论对偏离是否敏感。没有一张图或一个检验能够独自完成这项工作。
> <!-- bilingual-en:start -->
> A joint-Gaussian diagnostic combines data provenance, marginal and multivariate projections, Mahalanobis structure when covariance can be estimated, and sensitivity of the intended downstream analysis.
> <!-- bilingual-en:end -->

## 一套最小但完整的顺序

1. **先核实数据来源。** 检查观察单位、时间或分组结构、缺失、单位、录入错误与测量上限。分布图无法修复对象定义错位。
2. **看每个边际。** 用直方图、箱线图和单变量 Q–Q 图找偏斜、厚尾、多峰与边界堆积。
3. **看联合投影。** 检查成对散点、按组着色的投影和若干有实质意义的线性组合，寻找弯曲、分群、异方差与方向性尾部。
4. **再看 Mahalanobis 结构。** 仅在 $S$ 可逆且估计不过度不稳定时，计算
   $$d_i^2=(x_i-\bar x)^TS^{-1}(x_i-\bar x)$$
   并与 $\chi_p^2$ 分位作图。这里的参考线是近似形状参照，不是每个点的精确独立抽样律。
5. **回到用途做敏感性分析。** 比较稳健协方差、合理变换、替代分布或删留经核实异常记录后的下游结论。偏离是否重要取决于要做的推断、分类或风险计算。

## 怎样解释结果

某个点很远，可能是录入错误、总体外对象、真实稀有事件、混合群体或模型漏项。先追查来源，再按 [[异常观测处理原则]] 处理；不能为了让图更直而自动删除。

形式检验也只是证据的一部分。大样本会对很小偏离敏感，小样本则可能缺乏功效。最终报告应写清发现了什么、哪些检查受维度或样本量限制，以及下游结论是否稳健。认识论边界见 [[边际与马氏图不证联合Gaussian]]。

> [!question]- 自检
> 若 $p\ge n$ 导致 $S$ 奇异，联合 Gaussian 诊断是否只能停止？
>
> **答案：** 不是。普通 Mahalanobis 图不能直接做，但仍可检查数据来源、边际、低维投影和下游敏感性；若采用正则化或子空间距离，必须明确那是额外方法。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.8. 正态性假设检验|多元统计课程 §1.8]]：核对边际图与 Mahalanobis Q–Q 的课程流程。
- [Penn State STAT 505, Lesson 3](https://online.stat.psu.edu/stat505/Lesson03)：核对多元图形探索、变换、异常点与敏感性检查。
- [NIST/SEMATECH e-Handbook, Q–Q plots](https://www.itl.nist.gov/div898/handbook/eda/section3/eda33o.htm)：核对分位图的解释边界。
- [[边际与马氏图不证联合Gaussian]]：核对有限诊断不能升级为分布证明。

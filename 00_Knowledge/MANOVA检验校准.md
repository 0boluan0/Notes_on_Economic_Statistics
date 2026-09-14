---
aliases:
  - "MANOVA 统计量必须结合假设秩、响应维数和残差自由度校准"
  - "MANOVA reference-distribution calibration"
  - "MANOVA approximate F calibration"
student_os: knowledge-atom
atom_id: STAT-MAN-004
atom_set: manova
atom_type: inference-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
requires:
  - "[[MANOVA四种统计量]]"
  - "[[MANOVA经典假设]]"
implies:
  - "[[多响应联合显著边界]]"
related:
  - "[[Wishart抽样假设]]"
  - "[[Pillai稳健性边界]]"
---

# MANOVA 统计量必须结合假设秩、响应维数和残差自由度校准
<!-- bilingual-en:start -->
*A MANOVA statistic must be calibrated using hypothesis rank, response dimension, and residual degrees of freedom*
<!-- bilingual-en:end -->

> [!summary] 原子校准边界
> $\Lambda,V,U,\Theta$ 的观测值本身不是 $p$ 值。把它们换成拒绝规则还需要被检验假设的自由度或秩、有效响应维数、误差自由度，以及经典参考分布所依赖的误差模型。常用软件默认报告的是各统计量变换后的**近似 $F$**；不能只看表头里的 “F” 就称为精确有限样本 $F$ 检验。
> <!-- bilingual-en:start -->
> The raw multivariate criterion is not a p-value. Its reference distribution depends on hypothesis dimension, response dimension, residual degrees of freedom, and the assumed error model; common software reports transformed approximate $F$ statistics by default.
> <!-- bilingual-en:end -->

R 的 `summary.manova` 明确把四种统计量转换为近似 $F$。当被检验项只有一个自由度时，四种近似 $F$ 相同；更一般时它们可以不同，而且 R 对 Roy 统计量报告的 $F$ 是上界。SAS 默认同样使用 $F$ 近似；其 `MSTAT=EXACT` 能在所支持的经典情形下对 Wilks、Hotelling–Lawley 与 Roy 计算精确尾概率，但 Pillai 仍采用更精细的近似。因此“软件能给 $p$ 值”不等于“所用换算在当前问题里是精确的”。

校准还不能替代模型检查。即使数值算法顺利完成，相关观测行、组间协方差异质、非 Gaussian 小样本、不可估对比或残差 SSCP 秩亏都可能使经典参考分布失去依据。相反，在非正态大样本中使用重抽样、稳健方法或渐近结果也可以有合理依据，但必须明确那是新的校准方法，不能借经典 MANOVA 的名字省略条件。

报告时至少写出：被检验效应或对比、所选统计量、统计量值、采用的精确或近似换算、自由度与 $p$ 值。四种统计量若给出不同 $p$ 值，应按事先规定或有理论理由的选择解释；结果出来后择最小 $p$ 值会增加选择性错误。

> [!question]- 自检
> 两次分析得到相同的 Wilks $\Lambda$，能否不看维数和自由度就断定它们有相同 $p$ 值？
>
> **答案：** 不能。参考分布和近似 $F$ 换算还依赖假设维数、响应维数、误差自由度与所用模型。

## 来源与核验

- [R `summary.manova` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html)：核对默认输出为近似 $F$，单自由度效应时四种近似 $F$ 相同，以及一般情形 Roy 的 $F$ 是上界。
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm)：核对默认 $F$ 近似和 `MSTAT=EXACT` 对四种统计量不同的精确/近似处理。
- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08)：核对 Wilks 在课程 one-way MANOVA 表中需要按样本量、组数和响应维数作参考分布换算。
<!-- bilingual-en:start -->
- The official R and SAS documentation distinguishes approximate from exact calibration; Penn State fixes the one-way MANOVA degrees-of-freedom context.
<!-- bilingual-en:end -->

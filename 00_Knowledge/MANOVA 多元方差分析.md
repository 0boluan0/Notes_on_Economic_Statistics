---
aliases:
  - "MANOVA"
  - "Multivariate Analysis of Variance"
  - "多元方差分析"
status: source-checked
---

# MANOVA 多元方差分析
<!-- bilingual-en:start -->
*Multivariate analysis of variance (MANOVA)*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 当组别或实验因素可能同时改变多个相关响应时，MANOVA 检验整个均值向量是否随因素改变。
> **具体锚点：** 比较三种教学方案对数学、阅读和写作成绩的联合影响，而不是做三次互不联系的 ANOVA。
> **核心难点：** 总体检验比较假设 SSP 矩阵 $H$ 与误差 SSP 矩阵 $E$；Wilks、Pillai 等统计量摘要同一组特征根却有不同稳健性。
> **为什么重要：** 它先控制全局问题，再通过预定义对比和同时区间定位差异，避免将一串单变量检验误当整体结论。
> **继续：** 两组或单一均值向量问题见 [[Hotelling T² 与多元均值推断]]；一般设计矩阵表达见 [[多元线性回归]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** When groups or experimental factors may change several correlated outcomes at once, MANOVA tests whether the whole mean vector varies with those factors.
> **Concrete anchor:** Compare three teaching programmes on mathematics, reading, and writing jointly instead of running three unrelated ANOVAs.
> **Central difficulty:** The omnibus test compares a hypothesis SSP matrix $H$ with an error SSP matrix $E$. Wilks, Pillai, and related statistics summarise the same characteristic roots differently and have different robustness properties.
> **Why it matters:** It controls the global question first and then localises differences with prespecified contrasts and simultaneous intervals, avoiding the mistake of treating a collection of univariate tests as one joint conclusion.
> **Continue with:** See [[Hotelling T² 与多元均值推断|Hotelling's T-squared]] for one-vector or two-group problems and [[多元线性回归|multivariate linear regression]] for the general design-matrix formulation.
<!-- bilingual-en:end -->

> [!source] 本节依据
> - [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08)：核验 MANOVA 的 SSP 分解、统计量、对比与解释步骤。
> - Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验一般线性假设、特征根表示与经典假设。
<!-- bilingual-en:start -->
> [!source] Sources for this section
> - [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08) was checked for SSP decomposition, omnibus statistics, contrasts, and interpretation.
> - Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was checked for general linear hypotheses, characteristic-root representations, and classical assumptions.
<!-- bilingual-en:end -->

## MANOVA 的模型
<!-- bilingual-en:start -->
*The MANOVA model*
<!-- bilingual-en:end -->

单因素 MANOVA 比较多个组的均值向量，分解组间和组内平方和—交叉乘积矩阵。Wilks' lambda、Pillai trace 等统计量从不同角度比较解释与残差变异；它们可能对假设偏离有不同稳健性。
<!-- bilingual-en:start -->
A one-factor MANOVA compares several group mean vectors by decomposing between-group and within-group sums-of-squares-and-cross-products matrices. Wilks' lambda, Pillai's trace, and related statistics compare explained and residual variation in different ways and can differ in robustness to assumption violations.
<!-- bilingual-en:end -->

一般形式是 $Y=XB+E$，并检验 $H_0:CBA=D$。$C$ 选择或对比设计矩阵的效应，$A$ 选择或组合响应；因此 MANOVA 不只是“多个 ANOVA 一起跑”，而是对预先指定的效应和响应方向做联合检验。
<!-- bilingual-en:start -->
The general form is $Y=XB+E$ with hypothesis $H_0:CBA=D$. Matrix $C$ selects or contrasts effects in the design matrix, while $A$ selects or combines responses. MANOVA is therefore not merely several ANOVAs run together; it jointly tests prespecified effects and response directions.
<!-- bilingual-en:end -->

## $H$ 与 $E$ 矩阵
<!-- bilingual-en:start -->
*The $H$ and $E$ matrices*
<!-- bilingual-en:end -->

$E$ 是完整模型下的误差 SSP 矩阵，$H$ 是施加零假设后多出的未解释 SSP。若 $H$ 相对 $E$ 在某些响应组合上很大，则该因素对均值向量有证据。比较通常通过 $E^{-1}H$ 的非零特征根 $\lambda_j$ 完成。
<!-- bilingual-en:start -->
$E$ is the error SSP matrix under the full model, whereas $H$ is the additional unexplained SSP created by imposing the null hypothesis. If $H$ is large relative to $E$ along some response combination, the factor has evidence of an effect on the mean vector. The comparison is usually expressed through the nonzero characteristic roots $\lambda_j$ of $E^{-1}H$.
<!-- bilingual-en:end -->

- Wilks' lambda: $\Lambda=\prod_j(1+\lambda_j)^{-1}$，越小越反对零假设。
- Pillai trace: $V=\sum_j\lambda_j/(1+\lambda_j)$，通常在假设偏离时较稳健。
- Hotelling–Lawley trace: $U=\sum_j\lambda_j$，强调特征根总量。
- Roy's largest root: $\Theta=\max_j\lambda_j$，只聚焦最强的单一响应方向。
<!-- bilingual-en:start -->
- Wilks' lambda is $\Lambda=\prod_j(1+\lambda_j)^{-1}$; smaller values oppose the null more strongly.
- Pillai's trace is $V=\sum_j\lambda_j/(1+\lambda_j)$ and is often comparatively robust to assumption violations.
- The Hotelling–Lawley trace is $U=\sum_j\lambda_j$ and emphasises the total characteristic-root magnitude.
- Roy's largest root is $\Theta=\max_j\lambda_j$ and focuses only on the strongest response direction.
<!-- bilingual-en:end -->

## 显著后的解释
<!-- bilingual-en:start -->
*Interpreting a significant result*
<!-- bilingual-en:end -->

总体拒绝只说明至少某个均值向量不同。接下来应做预先定义的对比、同时区间或有控制的后续分析，并结合效应量和可视化。逐变量 ANOVA 是解释工具之一，不是自动的因果定位。
<!-- bilingual-en:start -->
An omnibus rejection says only that at least one mean vector differs. Follow it with prespecified contrasts, simultaneous intervals, or controlled post-hoc analyses, together with effect sizes and visualisation. Variable-by-variable ANOVA is one localisation tool, not an automatic causal diagnosis.
<!-- bilingual-en:end -->

MANOVA 显著后能否直接说每个响应都不同？
<!-- bilingual-en:start -->
*Can a significant MANOVA justify saying that every response differs?*
<!-- bilingual-en:end -->

不能。总体结论只说明某个线性组合或某些均值差异，需要后续对比和区间定位。
<!-- bilingual-en:start -->
No. The global conclusion establishes only that some linear combination or some mean contrast differs; follow-up contrasts and intervals are required to locate it.
<!-- bilingual-en:end -->

## 假设与设计
<!-- bilingual-en:start -->
*Assumptions and design*
<!-- bilingual-en:end -->

独立观察、适当的多元正态、协方差结构和设计矩阵秩是经典推断基础。MANOVA 不能修复混杂、选择偏差或测量不一致。
<!-- bilingual-en:start -->
Independent observations, an appropriate multivariate-normal model, suitable covariance structure, and a full-rank design matrix support classical inference. MANOVA cannot repair confounding, selection bias, or inconsistent measurement.
<!-- bilingual-en:end -->

对等协方差的检验本身也可以对非正态和样本量不平衡敏感，不应将“未拒绝”读成“完全相同”。设计严重不平衡且协方差差异明显时，应根据研究设计选择稳健方法、重抽样或明确限制结论。
<!-- bilingual-en:start -->
Tests of equal covariance matrices can themselves be sensitive to nonnormality and unequal sample sizes, so failure to reject is not proof of identical covariances. When a severely unbalanced design also has clear covariance heterogeneity, choose a design-appropriate robust or resampling method, or explicitly narrow the conclusion.
<!-- bilingual-en:end -->

## Worked example：三种教学方案
<!-- bilingual-en:start -->
*Worked example: three teaching programmes*
<!-- bilingual-en:end -->

设三组学生的响应向量为 $(\text{数学},\text{阅读},\text{写作})^T$。零假设是三个组的均值向量完全相同。若 Pillai trace 拒绝零假设，先画三组均值及同时区间，再检验预先规定的“新方案对传统方案”对比。若仅数学与阅读的组合差异明显，结论应如实说“联合学业结果不同”，而不是“三科都提高”。
<!-- bilingual-en:start -->
Suppose three student groups have response vector $(\text{mathematics},\text{reading},\text{writing})^T$. The null says that all three group mean vectors are identical. If Pillai's trace rejects it, first plot group means with simultaneous intervals and then test the prespecified contrast of a new programme against the traditional programme. If the difference is driven only by a mathematics–reading combination, report a difference in the joint academic outcome rather than claiming that all three subjects improved.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### MANOVA 显著后能否直接说每个响应都不同？
<!-- bilingual-en:start -->
*After a significant MANOVA, can one immediately say that every response differs?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不能。总体结论只说明某个线性组合或某些均值差异，需要后续对比和区间定位。
<!-- bilingual-en:start -->
> [!answer]- Answer
> No. The omnibus result establishes only that some response combination or mean contrast differs; follow-up contrasts and intervals are needed to locate it.
<!-- bilingual-en:end -->

### Wilks' lambda 越小表示什么？
<!-- bilingual-en:start -->
*What does a smaller Wilks' lambda indicate?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> $H$ 相对 $E$ 越大，完整模型未被零假设解释的响应组合越明显，因而反对零假设的证据越强。
<!-- bilingual-en:start -->
> [!answer]- Answer
> $H$ is larger relative to $E$, so response variation attributable to the tested effect is more pronounced and evidence against the null is stronger.
<!-- bilingual-en:end -->

### 为什么不能把 MANOVA 当成混杂的补救方法？
<!-- bilingual-en:start -->
*Why cannot MANOVA be used as a remedy for confounding?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> MANOVA 只联合建模多个响应，不会创造未观察混杂因素的信息，也不会修复非随机分组或选择偏差。
<!-- bilingual-en:start -->
> [!answer]- Answer
> MANOVA models several responses jointly; it does not create information about unobserved confounders or repair non-random assignment and selection bias.
<!-- bilingual-en:end -->

### 用自己的话解释 $H$ 与 $E$ 分别表示什么。
<!-- bilingual-en:start -->
*Explain in your own words what the $H$ and $E$ matrices represent.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> $H$ 表示被检验因素所对应的多响应变异，$E$ 表示完整模型仍未解释的多响应变异；检验问前者是否相对后者过大。
<!-- bilingual-en:start -->
> [!answer]- Answer
> $H$ represents multivariate variation associated with the tested effect, whereas $E$ represents multivariate variation left unexplained by the full model. The test asks whether the former is large relative to the latter.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08)：核验 MANOVA 的 SSP 分解、统计量、对比与解释步骤。
<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08) was checked for SSP decomposition, omnibus statistics, contrasts, and interpretation.
<!-- bilingual-en:end -->

- Johnson & Wichern, *Applied Multivariate Statistical Analysis*, 6th ed.：核验一般线性假设、特征根表示与经典假设。
<!-- bilingual-en:start -->
- Johnson and Wichern, *Applied Multivariate Statistical Analysis*, 6th ed., was checked for general linear hypotheses, characteristic-root representations, and classical assumptions.
<!-- bilingual-en:end -->

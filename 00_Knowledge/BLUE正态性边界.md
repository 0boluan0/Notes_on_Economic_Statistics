---
aliases:
  - "正态性不是 OLS 成为 BLUE 的条件"
  - OLS BLUE 无需正态性
  - Normality is not required for OLS to be BLUE
  - Normality and OLS inference
  - OLS 正态性边界
student_os: knowledge-atom
atom_id: ECON-OLS-013
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
  - "[[回归推断.canvas|回归推断]]"
requires:
  - "[[经典 Gauss–Markov 定理]]"
---

# 正态性不是 OLS 成为 BLUE 的条件
<!-- bilingual-en:start -->
*Normality is not required for OLS to be BLUE*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> 经典 Gauss–Markov 的 BLUE 结论不需要误差正态。若在经典线性模型条件之外再加 $u\mid X\sim N(0,\sigma^2I)$，才得到 OLS 系数的精确条件正态分布，以及使用通常方差估计量时精确的小样本 $t$ 与 $F$ 分布。
> <!-- bilingual-en:start -->
> The classical Gauss–Markov BLUE result does not require normally distributed errors. Adding $u\mid X\sim N(0,\sigma^2I)$ to the classical linear-model conditions yields an exact conditional normal distribution for OLS coefficients and exact finite-sample $t$ and $F$ distributions when the usual variance estimator is used.
> <!-- bilingual-en:end -->

## 三件事不要混在一起
<!-- bilingual-en:start -->
*Three distinct roles*
<!-- bilingual-en:end -->

1. 计算 OLS 系数只需要解最小二乘问题。
2. BLUE 来自线性、外生、满秩和球形误差方差等 Gauss–Markov 条件。
3. 正态性为经典小样本检验提供精确参考分布。

<!-- bilingual-en:start -->
1. Computing OLS coefficients only requires solving the least-squares problem.
2. BLUE follows from the linear model, exogeneity, full rank, and spherical error variance in the Gauss–Markov conditions.
3. Normality supplies exact reference distributions for classical finite-sample tests.
<!-- bilingual-en:end -->

大样本下，即使误差不正态，也常可在适当矩条件、抽样/依赖条件和正确标准误下用渐近正态近似做推断。反过来，仅有“误差看起来正态”不能修复内生性、异方差、序列相关或模型设定错误；在这些问题存在时，常规 $t/F$ 仍可能失效。
<!-- bilingual-en:start -->
In large samples, inference can often use asymptotic normal approximations even with non-normal errors, provided suitable moment and sampling or dependence conditions hold and the standard errors are appropriate. Conversely, errors that look normal do not repair endogeneity, heteroskedasticity, serial correlation, or misspecification; conventional $t/F$ procedures can still fail in their presence.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 误差不正态时，哪一个结论不会因此自动消失：OLS 可计算、BLUE，还是精确小样本 $t/F$？
> <!-- bilingual-en:start -->
> If errors are non-normal, which result does not automatically disappear: computability of OLS, BLUE, or exact finite-sample $t/F$ inference?
> <!-- bilingual-en:end -->
>
> **答案：** OLS 仍可计算；若其余 Gauss–Markov 条件成立，BLUE 也仍成立。正态性直接提供的是经典检验的精确小样本分布。
> <!-- bilingual-en:start -->
> **Answer:** OLS remains computable, and BLUE still holds if the other Gauss–Markov conditions hold. Normality directly supplies the exact finite-sample distributions for classical tests.
> <!-- bilingual-en:end -->

## 继续

- [[回归t检验]]：区分精确小样本参考分布与渐近校准。
- [[标准误口径匹配]]：按异方差、时间或组内依赖选择协方差估计。
- [[残差图边界|残差诊断]]：把单项诊断放回其能发现、不能证明的正确范围。
<!-- bilingual-en:start -->
- [[回归t检验|Regression t tests]] distinguish exact finite-sample reference distributions from asymptotic calibration.
- [[标准误口径匹配|Covariance-estimator matching]] selects a standard-error structure for heteroskedastic, temporal, or within-group dependence.
- [[残差图边界|Directed residual diagnostics]] keeps any single diagnostic within the scope of what it can reveal and what it cannot prove.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 4 §4.1：核验正态性是 MLR.6、它附加于 Gauss–Markov 假设之上，以及精确 $t/F$ 分布的用途。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#4.1. 极大似然估计|本地课程：正态误差下 MLE 与 OLS]] 与 [[02_Economy/01_Econometrics/03_多元线性回归.md#6.1. JB 正态性检验|本地课程：JB 正态性检验]]：核对正态性在本课程中的位置。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 4 §4.1, supports normality as assumption MLR.6 added to the Gauss–Markov conditions and its role in exact $t/F$ distributions.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#4.1. 极大似然估计|The local course section on MLE and OLS under normal errors]] and [[02_Economy/01_Econometrics/03_多元线性回归.md#6.1. JB 正态性检验|the local JB normality-test section]] locate normality within the course sequence.
<!-- bilingual-en:end -->

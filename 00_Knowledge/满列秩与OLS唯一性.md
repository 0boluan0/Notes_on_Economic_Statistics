---
aliases:
  - "满列秩使 OLS 系数解唯一"
  - Full column rank gives a unique OLS coefficient vector
  - OLS rank condition
  - OLS 唯一解条件
student_os: knowledge-atom
atom_id: ECON-OLS-004
atom_type: proposition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
  - "[[多重共线性.canvas|多重共线性与设计矩阵诊断]]"
  - "[[虚拟变量与交互项.canvas|虚拟变量与交互项]]"
requires:
  - "[[普通最小二乘]]"
related:
  - "[[最小二乘解唯一性]]"
implies:
  - "[[OLS正规方程]]"
---

# 满列秩使 OLS 系数解唯一
<!-- bilingual-en:start -->
*Full column rank makes the OLS coefficient solution unique*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> 若设计矩阵 $X$ 满列秩，则 $X'X$ 可逆，OLS 系数向量唯一，并可写成
> $$
> \hat\beta=(X'X)^{-1}X'y.
> $$
> <!-- bilingual-en:start -->
> If the design matrix $X$ has full column rank, then $X'X$ is invertible, the OLS coefficient vector is unique, and $\hat\beta=(X'X)^{-1}X'y$.
> <!-- bilingual-en:end -->

## 使用条件与含义边界
<!-- bilingual-en:start -->
*Use condition and meaning boundary*
<!-- bilingual-en:end -->

秩条件要求没有一列能由其他列精确线性表示。若秩亏，投影到 $C(X)$ 的拟合值仍然唯一，但产生同一拟合值的系数可能有无穷多个；此时逆矩阵公式失效。高度相关但不精确相关时，系数仍可计算，只是数值和统计精度可能很差，这属于近共线性而非“没有唯一解”。
<!-- bilingual-en:start -->
The rank condition requires that no column be an exact linear combination of the others. Under rank deficiency, the fitted projection onto $C(X)$ remains unique, but infinitely many coefficient vectors may produce it, so the inverse formula fails. With high but inexact correlation, coefficients remain computable, although numerical and statistical precision may be poor; that is near-collinearity, not non-uniqueness.
<!-- bilingual-en:end -->

## 最小反例
<!-- bilingual-en:start -->
*Minimal counterexample*
<!-- bilingual-en:end -->

模型同时包含截距、`male` 和 `female`，且每人恰属于两组之一，则 `male + female = 1`。三列精确线性相关，软件必须删除一列或施加约束，才能确定唯一系数。
<!-- bilingual-en:start -->
If a model contains an intercept, `male`, and `female`, and every observation belongs to exactly one group, then `male + female = 1`. The three columns are exactly collinear, so one column must be removed or a restriction imposed before the coefficients are unique.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> $X$ 秩亏时，是拟合值不唯一，还是系数不唯一？
> <!-- bilingual-en:start -->
> When $X$ is rank deficient, is the fitted value non-unique or the coefficient vector non-unique?
> <!-- bilingual-en:end -->
>
> **答案：** 到列空间的拟合值仍唯一；系数向量通常不唯一。
> <!-- bilingual-en:start -->
> **Answer:** The fitted projection onto the column space remains unique; the coefficient vector is generally non-unique.
> <!-- bilingual-en:end -->

## 继续

- [[OLS正规方程]]：秩条件决定正规方程是否给出唯一系数。
- [[多重共线性]]：区分完全共线与近共线；[[近似共线性精度]]继续诊断后者带来的不稳定。
<!-- bilingual-en:start -->
- [[OLS正规方程|The OLS normal equations make sample residuals orthogonal to the design matrix]] shows where the rank condition enters the coefficient solution.
- [[多重共线性|Multicollinearity]] distinguishes exact from near dependence; [[近似共线性精度|precision under near multicollinearity]] develops the resulting instability.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 3，假定 MLR.3 与 §§3.2a、3.3：核验无完全共线、唯一 OLS 解及其与无偏假设的分离。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S03_Lecture_Lecture_16_Projection_Matrices_and_Least_Squares.pdf|MIT 18.06SC Lecture 16]] 与 [[AA+列空间投影]]：核验满列秩、投影唯一和系数可能不唯一的边界。
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#2.1. OLS|本地课程：OLS 矩阵表达]]：核对 $(X'X)^{-1}X'y$ 公式。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 3, assumption MLR.3 and §§3.2a and 3.3, supports no perfect collinearity, uniqueness, and its separation from unbiasedness assumptions.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S03_Lecture_Lecture_16_Projection_Matrices_and_Least_Squares.pdf|MIT 18.06SC Lecture 16]] and [[AA+列空间投影|the pseudoinverse projector]] support full column rank, uniqueness of the projection, and possible non-uniqueness of coefficients.
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#2.1. OLS|The local matrix treatment of OLS]] fixes the formula $(X'X)^{-1}X'y$.
<!-- bilingual-en:end -->

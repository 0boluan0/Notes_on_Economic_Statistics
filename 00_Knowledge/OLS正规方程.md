---
aliases:
  - "OLS 正规方程使样本残差与设计矩阵正交"
  - OLS normal equations
  - Sample residual orthogonality
  - OLS 正规方程
student_os: knowledge-atom
atom_id: ECON-OLS-005
atom_type: proposition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
requires:
  - "[[普通最小二乘]]"
related:
  - "[[最小二乘正规方程]]"
contrasts_with:
  - "[[样本正交与总体外生性]]"
implies:
  - "[[一元 OLS 斜率]]"
---

# OLS 正规方程使样本残差与设计矩阵正交
<!-- bilingual-en:start -->
*The OLS normal equations make sample residuals orthogonal to the design matrix*
<!-- bilingual-en:end -->

> [!summary] 核心结论
> OLS 目标函数的一阶条件是
> $$
> X'(y-X\hat\beta)=X'\hat u=0.
> $$
> 因而样本残差向量与设计矩阵的每一列正交。
> <!-- bilingual-en:start -->
> The first-order condition for the OLS objective is $X'(y-X\hat\beta)=X'\hat u=0$. The sample residual vector is therefore orthogonal to every column of the design matrix.
> <!-- bilingual-en:end -->

## 使用条件与直接推论
<!-- bilingual-en:start -->
*Use condition and immediate implications*
<!-- bilingual-en:end -->

这条关系适用于实际放入该次回归的列和同一估计样本。若 $X$ 含截距列，则 $\sum_i\hat u_i=0$，所以残差均值为零，且 $\bar{\hat y}=\bar y$。若没有截距，这两个均值结论不必成立；加权最小二乘得到的则是加权正交关系。
<!-- bilingual-en:start -->
The relation applies to the columns actually included in that regression and to the same estimation sample. If $X$ contains an intercept, then $\sum_i\hat u_i=0$, so the residual mean is zero and $\bar{\hat y}=\bar y$. Without an intercept, those mean identities need not hold. Weighted least squares instead produces weighted orthogonality.
<!-- bilingual-en:end -->

## 自然解释
<!-- bilingual-en:start -->
*Natural interpretation*
<!-- bilingual-en:end -->

若残差还沿某个已含回归变量方向系统偏正或偏负，就能沿该方向微调系数并继续降低平方和。最优点处，这种可下降方向已经被消除；残差可以不为零，但不能再被任何已含列线性解释。
<!-- bilingual-en:start -->
If residuals remained systematically positive or negative along an included regressor direction, the corresponding coefficient could be adjusted to lower the squared error further. At the optimum, no such descent direction remains. Residuals need not be zero, but no included column can linearly explain them within the sample.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 含截距的 OLS 为什么必有残差和为零？
> <!-- bilingual-en:start -->
> Why must OLS residuals sum to zero when an intercept is included?
> <!-- bilingual-en:end -->
>
> **答案：** 截距列是全 1 向量，正规方程对这一列给出 $1'\hat u=\sum_i\hat u_i=0$。
> <!-- bilingual-en:start -->
> **Answer:** The intercept column is the vector of ones, so its normal equation is $1'\hat u=\sum_i\hat u_i=0$.
> <!-- bilingual-en:end -->

## 继续

- [[一元 OLS 斜率]]：一元设计下，正规方程可化成显式斜率。
- [[样本正交与总体外生性]]：防止把样本恒等式误当成总体假设的证据。
<!-- bilingual-en:start -->
- [[一元 OLS 斜率|The simple OLS slope equals sample covariance divided by sample variance]] derives the explicit one-regressor slope from these equations.
- [[样本正交与总体外生性|Sample residual orthogonality does not prove population exogeneity]] prevents a sample identity from being mistaken for evidence about the data-generating process.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 2 §§2.2–2.3 与 Chapter 3 §§3.2a、3.2e：核验一阶条件、残差和为零、残差与各回归元样本协方差为零及均值推论。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S03_Lecture_Lecture_16_Projection_Matrices_and_Least_Squares.pdf|MIT 18.06SC Lecture 16]]：核验残差属于 $C(X)^\perp$ 的几何解释。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1. 回归系数的估计|本地课程：多元 OLS 一阶条件]]：核对矩阵和逐列写法。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 2 §§2.2–2.3 and Chapter 3 §§3.2a and 3.2e, supports the first-order conditions, zero residual sum, zero sample covariance with each regressor, and the mean identities.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S03_Lecture_Lecture_16_Projection_Matrices_and_Least_Squares.pdf|MIT 18.06SC Lecture 16]] supports the geometry $\hat u\in C(X)^\perp$.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1. 回归系数的估计|The local course section on multiple-regression OLS]] fixes the matrix and columnwise notation.
<!-- bilingual-en:end -->

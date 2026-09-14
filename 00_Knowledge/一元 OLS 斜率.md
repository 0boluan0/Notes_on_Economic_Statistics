---
aliases:
  - "一元 OLS 斜率等于样本协方差除以样本方差"
  - Simple OLS slope as covariance over variance
  - Sample covariance divided by sample variance
  - 一元回归斜率公式
student_os: knowledge-atom
atom_id: ECON-OLS-007
atom_type: proposition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
requires:
  - "[[OLS正规方程]]"
contrasts_with:
  - "[[多元回归系数]]"
---

# 一元 OLS 斜率等于样本协方差除以样本方差
<!-- bilingual-en:start -->
*The simple OLS slope equals sample covariance divided by sample variance*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 在含截距的一元回归中，只要 $X$ 有样本变动，
> $$
> \hat\beta_1
> =\frac{\sum_i(X_i-\bar X)(Y_i-\bar Y)}{\sum_i(X_i-\bar X)^2}
> =\frac{\widehat{\operatorname{Cov}}(X,Y)}{\widehat{\operatorname{Var}}(X)}.
> $$
> <!-- bilingual-en:start -->
> In a simple regression with an intercept, provided $X$ varies in the sample, $\hat\beta_1=\sum_i(X_i-\bar X)(Y_i-\bar Y)/\sum_i(X_i-\bar X)^2=\widehat{\operatorname{Cov}}(X,Y)/\widehat{\operatorname{Var}}(X)$.
> <!-- bilingual-en:end -->

## 自然解释与边界
<!-- bilingual-en:start -->
*Natural interpretation and boundary*
<!-- bilingual-en:end -->

分子衡量 $X$ 与 $Y$ 一起变化的程度，分母把它换算成“每一单位 $X$ 变化对应多少 $Y$ 变化”。若 $X$ 在样本中不变，分母为零，斜率无法识别。这个公式描述样本双变量线性关联，不会自动控制遗漏因素，也不会自动获得因果解释。
<!-- bilingual-en:start -->
The numerator measures how $X$ and $Y$ move together; the denominator converts that co-movement into change in $Y$ per unit change in $X$. If $X$ does not vary in the sample, the denominator is zero and the slope is unidentified. The formula describes a sample bivariate linear association; it neither controls omitted factors nor supplies a causal interpretation.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

在消费对收入的一元回归中，$\hat\beta_1=0.8$ 表示样本回归线上收入每增加 1 单位，消费平均增加 0.8 单位；它可作为样本中的边际消费倾向估计，但是否对应结构性消费反应仍取决于模型和外生性。
<!-- bilingual-en:start -->
In a simple regression of consumption on income, $\hat\beta_1=0.8$ means that a one-unit increase in income corresponds to an average 0.8-unit increase in consumption along the sample regression line. It estimates the sample marginal propensity to consume, but a structural interpretation still depends on the model and exogeneity.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 若 $X$ 与 $Y$ 同向变化更强，而 $X$ 自身波动不变，OLS 斜率怎样变？
> <!-- bilingual-en:start -->
> If $X$ and $Y$ move together more strongly while the variation in $X$ is unchanged, what happens to the OLS slope?
> <!-- bilingual-en:end -->
>
> **答案：** 分子协方差增大、分母不变，所以斜率增大。
> <!-- bilingual-en:start -->
> **Answer:** The covariance numerator rises while the variance denominator stays fixed, so the slope rises.
> <!-- bilingual-en:end -->

## 继续

- [[多元回归系数]]：加入控制变量后，斜率不再是原始两变量协方差之比。
- [[FWL残差化定理]]：用残差化后的变量恢复同样的“一元斜率”形式。
<!-- bilingual-en:start -->
- [[多元回归系数|A multiple-regression coefficient describes a linear association conditional on included controls]] explains why the raw bivariate covariance ratio no longer applies after controls are added.
- [[FWL残差化定理|The FWL theorem turns a multiple-regression coefficient into a slope between residuals]] recovers the same simple-slope form after residualisation.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 2 §§2.2–2.3：核验斜率公式、样本波动条件和样本回归线解释。
- [[02_Economy/01_Econometrics/02_一元线性回归.md#3.3. $\hat\beta_0$ 和 $\hat\beta_1$ 的解法|本地课程：一元 OLS 解法]]：核对课程中的协方差—方差写法与消费例子。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 2 §§2.2–2.3, supports the slope formula, the requirement that $X$ vary, and the sample-regression-line interpretation.
- [[02_Economy/01_Econometrics/02_一元线性回归.md#3.3. $\hat\beta_0$ 和 $\hat\beta_1$ 的解法|The local course solution for the simple OLS coefficients]] fixes the covariance-over-variance notation and consumption example.
<!-- bilingual-en:end -->

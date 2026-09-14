---
aliases:
  - "level 与 log 的位置决定回归系数单位"
  - Level and log positions determine coefficient units
  - Level-log regression interpretation
  - 对数回归系数解释
student_os: knowledge-atom
atom_id: ECON-OLS-014
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
  - "[[回归模型设定与函数形式.canvas|回归模型设定与函数形式]]"
requires:
  - "[[参数线性]]"
---

# level 与 log 的位置决定回归系数单位
<!-- bilingual-en:start -->
*The positions of levels and logs determine regression-coefficient units*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> 在其他已含变量不变时，因变量和解释变量是否取对数，决定 $\beta_1$ 表示单位变化、百分比变化、半弹性还是弹性。
> <!-- bilingual-en:start -->
> Holding the other included variables fixed, whether the outcome and regressor are logged determines whether $\beta_1$ describes a unit change, a percentage change, a semielasticity, or an elasticity.
> <!-- bilingual-en:end -->

表中“条件均值”的读法假定相应方程确实是该尺度上的线性条件均值；若只把方程当作最佳线性投影，就应把它读成该投影拟合值的变化，而不是完整条件均值的变化。
<!-- bilingual-en:start -->
References to a “conditional mean” in the table assume that the equation is in fact a linear conditional-mean model on the stated scale. Under a best-linear-projection-only interpretation, read the entries as changes in the fitted projection rather than in the full conditional mean.
<!-- bilingual-en:end -->

| 形式 | $\beta_1$ 的常用读法 | 精确边界 |
|---|---|---|
| $Y=\beta_0+\beta_1X+u$ | $X$ 增 1 单位，$Y$ 的条件均值增 $\beta_1$ 单位 | 两边都是原单位 |
| $\log Y=\beta_0+\beta_1X+u$ | $X$ 增 1 单位，对数方程的指数化拟合值约变 $100\beta_1\%$ | 精确比例变化为 $100(e^{\beta_1}-1)\%$；这不是一般意义下 $E(Y\mid X)$ 的变化 |
| $Y=\beta_0+\beta_1\log X+u$ | $X$ 增 1%，$Y$ 约变 $0.01\beta_1$ 单位 | 精确变化为 $\beta_1\log(1.01)$ |
| $\log Y=\beta_0+\beta_1\log X+u$ | $\beta_1$ 是对数方程指数化拟合值相对于 $X$ 的弹性；$X$ 增 1%，前者约变 $\beta_1\%$ | 精确比例变化为 $100(1.01^{\beta_1}-1)\%$；回到算术条件均值仍需重变换假设 |

<!-- bilingual-en:start -->
| Form | Usual reading of $\beta_1$ | Exact boundary |
|---|---|---|
| $Y=\beta_0+\beta_1X+u$ | A one-unit increase in $X$ changes conditional mean $Y$ by $\beta_1$ units | Both variables remain on their original scales |
| $\log Y=\beta_0+\beta_1X+u$ | A one-unit increase in $X$ changes the exponentiated fitted value from the log equation by approximately $100\beta_1\%$ | Its exact proportional change is $100(e^{\beta_1}-1)\%$; this is not generally the change in $E(Y\mid X)$ |
| $Y=\beta_0+\beta_1\log X+u$ | A one-percent increase in $X$ changes $Y$ by approximately $0.01\beta_1$ units | The exact change is $\beta_1\log(1.01)$ |
| $\log Y=\beta_0+\beta_1\log X+u$ | $\beta_1$ is the elasticity of the exponentiated fitted value from the log equation with respect to $X$; a one-percent increase in $X$ changes that fitted value by approximately $\beta_1\%$ | The exact proportional change is $100(1.01^{\beta_1}-1)\%$; returning to the arithmetic conditional mean still requires a retransformation assumption |
<!-- bilingual-en:end -->

## 使用条件与含义边界
<!-- bilingual-en:start -->
*Use condition and meaning boundary*
<!-- bilingual-en:end -->

取对数的变量必须为正。百分比近似只在变化较小时可靠；系数较大或变化不是 1% 时应使用精确换算。对 $Y$ 取对数还会改变建模对象：$E(\log Y\mid X)$ 指数化后通常不等于 $E(Y\mid X)$，所以回到 level 预测时需要明确重变换目标和方法。
<!-- bilingual-en:start -->
A logged variable must be positive. Percentage interpretations are approximations that work best for small changes; large coefficients or changes other than one percent call for exact conversion. Logging $Y$ also changes the modelled object: exponentiating $E(\log Y\mid X)$ generally does not recover $E(Y\mid X)$, so prediction on the level scale requires an explicit retransformation target and method.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 在 $Y=\beta_0+\beta_1\log X+u$ 中，为什么 $\beta_1$ 不是“$X$ 增 1 单位时 $Y$ 的变化”？
> <!-- bilingual-en:start -->
> In $Y=\beta_0+\beta_1\log X+u$, why is $\beta_1$ not the change in $Y$ for a one-unit increase in $X$?
> <!-- bilingual-en:end -->
>
> **答案：** 模型中的输入是 $\log X$；$\beta_1$ 按对数变化计量，1% 的 $X$ 变化约对应 $0.01\beta_1$ 单位的 $Y$ 变化。
> <!-- bilingual-en:start -->
> **Answer:** The regressor is $\log X$, so $\beta_1$ is measured per log change; a one-percent increase in $X$ corresponds to approximately $0.01\beta_1$ units of $Y$.
> <!-- bilingual-en:end -->

## 继续

- [[取值依赖边际效应]]：处理系数不能单独解释的另一类常见设定。
- [[回归模型设定与函数形式.canvas|回归模型设定与函数形式]]：把变换、重变换和预测目标放回完整设定路径。
<!-- bilingual-en:start -->
- [[取值依赖边际效应|Quadratic and interaction terms make marginal effects depend on variable values]] handles another common case in which a coefficient cannot be interpreted alone.
- [[回归模型设定与函数形式.canvas|Regression model specification and functional form]] places transformations, retransformation, and prediction targets in the complete specification path.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 2 §2.4：核验四种 level/log 形式、半弹性和弹性的近似读法。
- UCLA Statistical Consulting, [*How do I interpret a regression model when some variables are log transformed?*](https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faqhow-do-i-interpret-a-regression-model-when-some-variables-are-log-transformed/)：核验精确换算、几何均值与算术均值的区分。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#5.1. 含有对数变化的模型|本地课程：对数模型]]：核对课程单位、精确百分比与重变换边界。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 2 §2.4, supports the four level/log forms and the approximate semielasticity and elasticity readings.
- [UCLA Statistical Consulting's guide to log-transformed regression](https://stats.oarc.ucla.edu/other/mult-pkg/faq/general/faqhow-do-i-interpret-a-regression-model-when-some-variables-are-log-transformed/) supports the exact conversions and the distinction between geometric and arithmetic means.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#5.1. 含有对数变化的模型|The local course section on logarithmic models]] supports the course units, exact percentage conversion, and retransformation boundary.
<!-- bilingual-en:end -->

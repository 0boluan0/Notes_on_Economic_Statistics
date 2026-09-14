---
aliases:
  - "无变换和交互的加性 LPM 把连续回归量的概率斜率设为常数，因而可能漏掉弯曲或饱和关系"
  - Constant slopes in a basic LPM
student_os: knowledge-atom
atom_id: ECON-BIN-006
atom_set: binary-outcome-models
atom_type: functional-form-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[线性概率模型]]"
  - "[[LPM系数]]"
related:
  - "[[类别连续交互]]"
leads_to:
  - "[[非线性边际效应]]"
  - "[[二元模型选择]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# 无变换和交互的加性 LPM 把连续回归量的概率斜率设为常数，因而可能漏掉弯曲或饱和关系

<!-- bilingual-en:start -->
*An additive LPM without transformations or interactions imposes a constant probability slope on each continuous regressor and may therefore miss curved or saturating relationships.*
<!-- bilingual-en:end -->

> [!summary] 函数形式边界
> 在基本 LPM
> $$
> p(X)=\beta_0+\sum_j\beta_jX_j
> $$
> 中，$\partial p/\partial X_j=\beta_j$ 与协变量位置无关。现实概率却常在中间区间变化较快、接近 0 或 1 时变平；基本 LPM 无法表达这种形状。
> <!-- bilingual-en:start -->
> In the basic LPM above, $\partial p/\partial X_j=\beta_j$ does not depend on the covariate values. Real-world probabilities, however, often change more rapidly in the middle of their range and flatten as they approach 0 or 1; the basic LPM cannot represent this shape.
> <!-- bilingual-en:end -->

“LPM 的斜率恒定”只针对变量以单一线性主效应进入的规格。加入 $X^2$、样条或 $X\times D$ 后，完整斜率可以随 $X$ 或组别变化；此时 LPM 仍对参数线性，但不能再把单个 $X$ 系数称为完整概率效应。

<!-- bilingual-en:start -->
The statement that “an LPM has a constant slope” applies only to a specification in which the variable enters as a single linear main effect. After adding $X^2$, a spline, or $X\times D$, the full slope may vary with $X$ or across groups. The LPM remains linear in its parameters, but the coefficient on $X$ alone can no longer be called the complete probability effect.
<!-- bilingual-en:end -->

例如 $p=0.10+0.05\,experience$ 强迫第 1 年和第 20 年经验都增加 5 个百分点。若真实就业概率在职业早期快速上升、随后饱和，常数斜率会同时造成中段欠拟合和尾部越界。应先画出支持范围内的预测与残差，再决定加入有理论依据的非线性项，或改用 [[Logit模型]]、[[Probit模型]]；换链接也不替代对线性指数本身的正确设定。

<!-- bilingual-en:start -->
For example, $p=0.10+0.05\,experience$ forces both the first and the twentieth year of experience to raise the probability by 5 percentage points. If the true employment probability rises quickly early in a career and then saturates, the constant slope will cause both underfitting in the middle and out-of-range predictions in the tail. First plot predictions and residuals over the observed support; then decide whether to add theoretically justified nonlinear terms or use a [[Logit模型|logit model]] or [[Probit模型|probit model]]. Changing the link function does not replace correct specification of the linear index itself.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §§7.5、17.1：核对基本 LPM 的常数部分效应，以及 logit/probit 中取值依赖的概率效应。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §§7.5 and 17.1: verifies the constant partial effect in the basic LPM and the covariate-dependent probability effects in logit and probit models.
<!-- bilingual-en:end -->
- [[类别连续交互]]：复用交互使条件斜率按组别改变的规则。
<!-- bilingual-en:start -->
- [[类别连续交互|Indicator-by-continuous interactions]]: supplies the rule that interactions allow conditional slopes to vary by group.
<!-- bilingual-en:end -->

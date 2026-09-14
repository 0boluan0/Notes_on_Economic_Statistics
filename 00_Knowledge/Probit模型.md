---
aliases:
  - "Probit 用标准正态 CDF 将线性指数映射到 (0,1)，原始系数刻画规范化潜变量指数"
  - Probit model
student_os: knowledge-atom
atom_id: ECON-BIN-009
atom_set: binary-outcome-models
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[潜变量二元选择]]"
leads_to:
  - "[[二元响应MLE]]"
  - "[[二元预测概率]]"
  - "[[非线性边际效应]]"
related:
  - "[[完全分离]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# Probit 用标准正态 CDF 将线性指数映射到 (0,1)，原始系数刻画规范化潜变量指数

<!-- bilingual-en:start -->
*Probit maps a linear index into $(0,1)$ through the standard normal CDF, and its raw coefficients describe a normalized latent-variable index.*
<!-- bilingual-en:end -->

> [!summary] 模型定义
> Probit 设定
> $$
> p(X)=P(Y=1\mid X)=\Phi(X'\beta),
> $$
> 其中 $\Phi$ 是标准正态 CDF。潜变量表示把误差方差规范化为 1，所以原始系数位于标准化指数尺度，不是概率变化。
> <!-- bilingual-en:start -->
> Probit specifies the probability through the equation above, where $\Phi$ is the standard normal CDF. The latent-variable representation normalizes the error variance to 1, so a raw coefficient is measured on the standardized index scale rather than as a change in probability.
> <!-- bilingual-en:end -->

$\Phi$ 严格递增，且密度 $\phi(z)>0$。若 $X_j$ 只以单一未变换主效应进入，则概率边际效应为

<!-- bilingual-en:start -->
$\Phi$ is strictly increasing, and its density satisfies $\phi(z)>0$. If $X_j$ enters only as a single untransformed main effect, its marginal effect on the probability is
<!-- bilingual-en:end -->

$$
\frac{\partial p(X)}{\partial X_j}=\phi(X'\beta)\beta_j.
$$

因此简单主效应的系数符号给出方向，但效应大小随 $X'\beta$ 改变。若有平方、交互或其他重复进入，必须把 $\beta_j$ 换成完整的 $\partial(X'\beta)/\partial X_j$；此时单个系数的符号未必是变量总效应的符号。

<!-- bilingual-en:start -->
Thus, the coefficient sign gives the direction of a simple main effect, but the effect's magnitude varies with $X'\beta$. If the variable also appears through a square, an interaction, or another repeated term, $\beta_j$ must be replaced by the complete derivative $\partial(X'\beta)/\partial X_j$; in that case, the sign of a single coefficient need not equal the sign of the variable's total effect.
<!-- bilingual-en:end -->

Probit 与 logit 的中段形状常接近，尾部和尺度不同。不能以“哪个原始系数更大”选择模型，也不能把 probit 系数转成 odds ratio；应比较同一协变量取值下的预测概率、目标效应和经验证的概率表现。

<!-- bilingual-en:start -->
Probit and logit often have similar shapes in the middle of the probability range, but they differ in their tails and scales. A model should not be chosen by asking which has the larger raw coefficient, and a probit coefficient cannot be converted into an odds ratio. Instead, compare predicted probabilities at the same covariate values, the effects relevant to the target question, and validated probability performance.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- StataCorp, [*probit — Probit regression*](https://www.stata.com/manuals/rprobit.pdf), Methods and formulas：核对标准正态 CDF 规格、Bernoulli 似然与估计定义。
<!-- bilingual-en:start -->
- StataCorp, [*probit — Probit regression*](https://www.stata.com/manuals/rprobit.pdf), Methods and formulas: verifies the standard normal CDF specification, Bernoulli likelihood, and estimator definition.
<!-- bilingual-en:end -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §17.1：核对潜变量表示、误差方差规范化、probit 概率式、系数方向与取值依赖的边际效应。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §17.1: verifies the latent-variable representation, normalization of the error variance, the probit probability equation, coefficient direction, and covariate-dependent marginal effects.
<!-- bilingual-en:end -->

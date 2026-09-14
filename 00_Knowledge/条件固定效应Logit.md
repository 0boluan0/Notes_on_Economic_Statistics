---
aliases:
  - "在静态面板 Logit 的联合条件似然成立时，对单位内成功次数条件化可以消去单位截距"
  - Conditional fixed-effects logit
  - Chamberlain conditional logit
student_os: knowledge-atom
atom_id: ECON-BIN-019
atom_set: binary-outcome-models
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Logit模型]]"
  - "[[个体固定效应]]"
  - "[[固定效应严格外生]]"
related:
  - "[[非线性固定效应偏误]]"
  - "[[组内变异边界]]"
leads_to:
  - "[[条件Logit只用变化组]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# 在静态面板 Logit 的联合条件似然成立时，对单位内成功次数条件化可以消去单位截距

<!-- bilingual-en:start -->
*When the joint conditional likelihood for a static panel logit is valid, conditioning on each unit's number of successes removes the unit intercept*
<!-- bilingual-en:end -->

> [!summary] 消去单位截距靠的是联合似然，不是单期公式
> 在静态设定下，令 $X_i=(X_{i1},\ldots,X_{iT})$。假设给定完整协变量路径 $X_i$ 与单位效应 $\alpha_i$ 后，各期二元结果的联合条件分布可写成
> $$
> P(Y_i=y_i\mid X_i,\alpha_i)
> =\prod_{t=1}^T
> \Lambda(\alpha_i+X_{it}'\beta)^{y_{it}}
> [1-\Lambda(\alpha_i+X_{it}'\beta)]^{1-y_{it}}.
> $$
> 这个联合式包含给定 $X_i,\alpha_i$ 的跨期条件独立结构，并要求解释变量满足静态固定效应模型所需的时序条件。此时单位内成功次数 $S_i=\sum_tY_{it}$ 是关于 $\alpha_i$ 的充分统计量；对 $S_i$ 条件化后，可以不逐个估计单位截距而估计共同斜率 $\beta$。
> <!-- bilingual-en:start -->
> In the static setting, let $X_i=(X_{i1},\ldots,X_{iT})$. Conditional on the complete covariate path $X_i$ and the unit effect $\alpha_i$, assume that the joint conditional distribution of the binary outcomes factorizes into the Bernoulli logit probabilities shown above. This joint expression imposes conditional independence over time given $X_i,\alpha_i$ and requires the regressors to satisfy the timing conditions of the static fixed-effects model. Then the within-unit success count $S_i=\sum_tY_{it}$ is sufficient for $\alpha_i$. Conditioning on $S_i$ makes it possible to estimate the common slope $\beta$ without estimating every unit intercept separately.
> <!-- bilingual-en:end -->

具体地，在所有满足 $\sum_td_t=S_i$ 的 0/1 序列集合 $\mathcal D(S_i)$ 上，条件概率为

<!-- bilingual-en:start -->
More specifically, over the set $\mathcal D(S_i)$ of all binary sequences satisfying $\sum_td_t=S_i$, the conditional probability is
<!-- bilingual-en:end -->

$$
P(Y_i=y_i\mid S_i,X_i)
=\frac{\exp\!\left(\sum_ty_{it}X_{it}'\beta\right)}
{\sum_{d\in\mathcal D(S_i)}\exp\!\left(\sum_td_tX_{it}'\beta\right)}.
$$

分子、分母比较的是成功次数相同的结果排列，公共的 $\alpha_iS_i$ 因子因而消失。这一步解释了为什么“对成功次数条件化”有效；仅写一个单期 logit 概率式，还不足以推出充分统计量或条件似然。

<!-- bilingual-en:start -->
The numerator and denominator compare outcome sequences with the same number of successes, so the common factor $\alpha_iS_i$ cancels. This is why conditioning on the success count works. A one-period logit probability by itself is not enough to derive either the sufficient statistic or the conditional likelihood.
<!-- bilingual-en:end -->

条件 FE logit 是 logistic 链接的特殊便利；一般 probit 没有同样的低维充分统计量可直接消去每个 $\alpha_i$。条件估计也没有给出各单位截距，所以不能仅凭其输出恢复每个单位的绝对概率或以 $\alpha_i$ 为输入的平均部分效应。

<!-- bilingual-en:start -->
Conditional FE logit is a special convenience of the logistic link. A general probit model has no analogous low-dimensional sufficient statistic that directly removes every $\alpha_i$. Conditional estimation also does not recover the unit intercepts, so its output alone cannot reconstruct each unit's absolute probability or an average partial effect that requires $\alpha_i$ as an input.
<!-- bilingual-en:end -->

本卡只覆盖静态条件 logit。若模型含滞后结果、真实状态依赖或初始条件问题，条件化 $S_i$ 不能自动解决这些额外结构；它们不属于本批原子。

<!-- bilingual-en:start -->
This card covers only static conditional logit. If the model contains a lagged outcome, genuine state dependence, or an initial-conditions problem, conditioning on $S_i$ does not automatically resolve those additional structures; they are outside the scope of this atom set.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Chamberlain (1980), [“Analysis of Covariance with Qualitative Data”](https://doi.org/10.2307/2297110), *Review of Economic Studies* 47(1): 225–238：直接来源，核对给定完整协变量路径的联合模型、对成功次数条件化以及定性面板数据的固定效应条件似然。
  <!-- bilingual-en:start -->
  Chamberlain (1980), [“Analysis of Covariance with Qualitative Data”](https://doi.org/10.2307/2297110), *Review of Economic Studies* 47(1): 225–238, is the primary source for the joint model conditional on the complete covariate path, conditioning on the number of successes, and the fixed-effects conditional likelihood for qualitative panel data.
  <!-- bilingual-en:end -->
- StataCorp, [*clogit — Conditional (fixed-effects) logistic regression*](https://www.stata.com/manuals/rclogit.pdf), “Fixed-effects logit” 与 Methods and formulas：核对联合 likelihood 以及对 $\sum_tY_{it}$ 条件化消去 $\alpha_i$。
  <!-- bilingual-en:start -->
  StataCorp, [*clogit — Conditional (fixed-effects) logistic regression*](https://www.stata.com/manuals/rclogit.pdf), “Fixed-effects logit” and Methods and formulas, supports the joint likelihood and removal of $\alpha_i$ by conditioning on $\sum_tY_{it}$.
  <!-- bilingual-en:end -->
- [[固定效应严格外生]]：复用静态固定效应模型对完整协变量路径的时序要求；哪些组实际进入条件似然，另见 [[条件Logit只用变化组]]。
  <!-- bilingual-en:start -->
  [[固定效应严格外生|Strict exogeneity for fixed effects]] supplies the timing requirements imposed on the complete covariate path by the static fixed-effects model. Which groups actually enter the conditional likelihood is developed separately in [[条件Logit只用变化组|the outcome-variation boundary]].
  <!-- bilingual-en:end -->

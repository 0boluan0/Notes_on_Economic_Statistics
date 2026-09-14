---
aliases:
  - "在 logit 中，若 X_j 只以未变换主效应进入，则其他变量固定时增加一单位使条件胜算乘以 exp(β_j)"
  - Odds ratios in logit
student_os: knowledge-atom
atom_id: ECON-BIN-011
atom_set: binary-outcome-models
atom_type: interpretation
status: source-checked
mastery_state: unassessed
requires:
  - "[[Logit模型]]"
related:
  - "[[二元预测概率]]"
  - "[[非线性边际效应]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# 在 logit 中，若 X_j 只以未变换主效应进入，则其他变量固定时增加一单位使条件胜算乘以 exp(β_j)

<!-- bilingual-en:start -->
*In a logit model, if $X_j$ enters only as an untransformed main effect, a one-unit increase multiplies the conditional odds by $\exp(\beta_j)$, holding the other variables fixed*
<!-- bilingual-en:end -->

> [!summary] 严格解释
> Logit 满足
> $$
> \log\frac{p(X)}{1-p(X)}=X'\beta.
> $$
> 若 $X_j$ 只以 $\beta_jX_j$ 进入，则在其他变量固定、$X_j$ 增加一单位时，log-odds 增加 $\beta_j$，条件 odds 乘以 $e^{\beta_j}$。
> <!-- bilingual-en:start -->
> In a logit model, the log-odds equal $X'\beta$. If $X_j$ enters only through $\beta_jX_j$, then, holding all other variables fixed, a one-unit increase in $X_j$ raises the log-odds by $\beta_j$ and multiplies the conditional odds by $e^{\beta_j}$.
> <!-- bilingual-en:end -->

odds 是 $p/(1-p)$，不是概率本身。若 $\beta_j=\log 2$，结论是条件 odds 翻倍；概率从 0.1 变到约 0.182，从 0.5 则变到约 0.667，绝不是在所有起点都“概率翻倍”。odds ratio 也不是 risk ratio。

<!-- bilingual-en:start -->
Odds are $p/(1-p)$, not probability itself. If $\beta_j=\log 2$, the conditional odds double; a probability of 0.1 rises to about 0.182, whereas a probability of 0.5 rises to about 0.667. It is therefore never correct to say that the probability “doubles” from every baseline. An odds ratio is not a risk ratio either.
<!-- bilingual-en:end -->

二元变量 $D$ 的 $e^{\beta_D}$ 对比 $D=1$ 与 $D=0$ 的条件 odds，仍要求其他变量固定且规格没有涉及 $D$ 的交互或变换。若模型含 $D\times Z$，在给定 $Z=z$ 时的 odds ratio 是 $\exp(\beta_D+z\beta_{DZ})$；只报 $e^{\beta_D}$ 等于只报基准 $Z=0$ 的对比。

<!-- bilingual-en:start -->
For a binary variable $D$, $e^{\beta_D}$ compares the conditional odds at $D=1$ with those at $D=0$. This interpretation still requires the other variables to be held fixed and the specification to contain no interaction or transformation involving $D$. If the model includes $D\times Z$, the odds ratio at $Z=z$ is $\exp(\beta_D+z\beta_{DZ})$; reporting only $e^{\beta_D}$ reports only the comparison at the reference value $Z=0$.
<!-- bilingual-en:end -->

这个解释专属于 logit 的 log-odds 线性结构，不能搬到 probit。面向实质问题时最好同时报告有明确协变量基准的两组预测概率。

<!-- bilingual-en:start -->
This interpretation is specific to the log-linear odds structure of logit and cannot be transferred to probit. For substantive interpretation, it is usually better to report both predicted probabilities for a clearly stated covariate configuration as well.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- StataCorp, [*logistic — Logistic regression, reporting odds ratios*](https://www.stata.com/manuals/rlogistic.pdf), Description and Methods and formulas：核对 $e^{\beta_j}$ 的条件 odds-ratio 解释。
  <!-- bilingual-en:start -->
  StataCorp, [*logistic — Logistic regression, reporting odds ratios*](https://www.stata.com/manuals/rlogistic.pdf), Description and Methods and formulas, supports the conditional odds-ratio interpretation of $e^{\beta_j}$.
  <!-- bilingual-en:end -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §17.1：核对 logit 的 log-odds 表达及从系数回到概率尺度的边界。
  <!-- bilingual-en:start -->
  Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §17.1, supports the log-odds representation of logit and the limits of translating coefficients back to the probability scale.
  <!-- bilingual-en:end -->
- [[交互主效应基准]]：复用交互存在时主效应只属于基准条件的规则。
  <!-- bilingual-en:start -->
  [[交互主效应基准|The main-effect reference boundary]] supplies the rule that a main effect applies only at the reference condition when an interaction is present.
  <!-- bilingual-en:end -->

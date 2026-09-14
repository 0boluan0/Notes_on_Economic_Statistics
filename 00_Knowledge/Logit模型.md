---
aliases:
  - "Logit 用 logistic CDF 将线性指数映射到 (0,1)，原始系数位于 log-odds 尺度"
  - Logit model
  - Logistic regression
student_os: knowledge-atom
atom_id: ECON-BIN-008
atom_set: binary-outcome-models
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[潜变量二元选择]]"
leads_to:
  - "[[二元响应MLE]]"
  - "[[Logit胜算比]]"
  - "[[二元预测概率]]"
  - "[[非线性边际效应]]"
related:
  - "[[完全分离]]"
  - "[[准完全分离]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# Logit 用 logistic CDF 将线性指数映射到 (0,1)，原始系数位于 log-odds 尺度

<!-- bilingual-en:start -->
*Logit maps a linear index into $(0,1)$ through the logistic CDF, and its raw coefficients are measured on the log-odds scale.*
<!-- bilingual-en:end -->

> [!summary] 模型定义
> Logit 设定
> $$
> p(X)=P(Y=1\mid X)=\Lambda(\eta),
> \qquad
> \eta=X'\beta,
> \qquad
> \Lambda(z)=\frac{e^z}{1+e^z}.
> $$
> 等价地，$\log[p(X)/(1-p(X))]=X'\beta$。因此原始系数改变的是 log-odds，不是概率百分点。
> <!-- bilingual-en:start -->
> Logit specifies the probability through the logistic CDF as shown above. Equivalently, $\log[p(X)/(1-p(X))]=X'\beta$. Thus, a raw coefficient changes the log-odds, not the probability in percentage points.
> <!-- bilingual-en:end -->

$\Lambda$ 严格递增，所以当 $X_j$ 只以单一未变换主效应进入时，$\beta_j$ 的符号与其概率效应方向相同；大小仍要乘以 logistic 密度，因而随其他协变量位置变化。若 $X_j$ 还进入平方项或交互项，方向由完整指数导数决定，不能只看一个系数。

<!-- bilingual-en:start -->
Because $\Lambda$ is strictly increasing, when $X_j$ enters only as a single untransformed main effect, the sign of $\beta_j$ matches the direction of its probability effect. The magnitude must still be multiplied by the logistic density and therefore varies with the values of the other covariates. If $X_j$ also enters through a squared term or interaction, the direction is determined by the derivative of the complete index, not by one coefficient alone.
<!-- bilingual-en:end -->

例如 $\eta=-2+0.8D$ 时，$D=0$ 的概率约为 0.119，$D=1$ 约为 0.231，概率差约 11.2 个百分点；原始系数 0.8 不是“概率增加 80%”。同一 0.8 在别的基准指数上会给出不同概率差。

<!-- bilingual-en:start -->
For example, if $\eta=-2+0.8D$, the probability is approximately 0.119 when $D=0$ and 0.231 when $D=1$, a difference of about 11.2 percentage points. The raw coefficient 0.8 does not mean that “the probability increases by 80%.” The same coefficient of 0.8 would produce a different probability difference at another baseline index.
<!-- bilingual-en:end -->

Logit 保证模型内预测位于 $(0,1)$，但不保证函数形式正确、概率已经校准或关系具有因果意义。仍需检查支持、规格、分离和样本外表现。

<!-- bilingual-en:start -->
Logit guarantees that predictions from the specified model lie in $(0,1)$, but it does not guarantee that the functional form is correct, the probabilities are calibrated, or the relationship is causal. The covariate support, specification, separation, and out-of-sample performance still require examination.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- StataCorp, [*logit — Logistic regression, reporting coefficients*](https://www.stata.com/manuals/rlogit.pdf), Methods and formulas：核对 logistic CDF、log-odds 线性化与系数尺度。
<!-- bilingual-en:start -->
- StataCorp, [*logit — Logistic regression, reporting coefficients*](https://www.stata.com/manuals/rlogit.pdf), Methods and formulas: verifies the logistic CDF, the linear log-odds representation, and the coefficient scale.
<!-- bilingual-en:end -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §17.1：核对二元响应 logit 的概率式及部分效应解释。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §17.1: verifies the probability specification and interpretation of partial effects in a binary-response logit model.
<!-- bilingual-en:end -->

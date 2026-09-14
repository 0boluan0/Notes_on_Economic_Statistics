---
aliases:
  - "二元响应模型的 MLE 通过最大化 Bernoulli 对数似然选择参数，并依赖所设概率函数与抽样结构"
  - Maximum likelihood for binary response models
student_os: knowledge-atom
atom_id: ECON-BIN-010
atom_set: binary-outcome-models
atom_type: estimator
status: source-checked
mastery_state: unassessed
requires:
  - "[[Logit模型]]"
  - "[[Probit模型]]"
related:
  - "[[标准误口径匹配]]"
leads_to:
  - "[[二元预测概率]]"
  - "[[完全分离]]"
  - "[[准完全分离]]"
  - "[[非线性固定效应偏误]]"
part_of:
  - "[[二元结果模型.canvas|二元结果模型]]"
---

# 二元响应模型的 MLE 通过最大化 Bernoulli 对数似然选择参数，并依赖所设概率函数与抽样结构

<!-- bilingual-en:start -->
*MLE for a binary-response model selects parameters by maximizing the Bernoulli log-likelihood and depends on the specified probability function and sampling structure.*
<!-- bilingual-en:end -->

> [!summary] 估计原理
> 令 $p_i(\beta)=G(X_i'\beta)$。对条件独立的 Bernoulli 观测，样本对数似然是
> $$
> \ell(\beta)=\sum_{i=1}^n\{Y_i\log p_i(\beta)+(1-Y_i)\log[1-p_i(\beta)]\}.
> $$
> MLE 选择使已观察 0/1 结果整体最可能的 $\hat\beta$；logit 取 $G=\Lambda$，probit 取 $G=\Phi$。
> <!-- bilingual-en:start -->
> Let $p_i(\beta)=G(X_i'\beta)$. For conditionally independent Bernoulli observations, the sample log-likelihood is given by the expression above. MLE selects the $\hat\beta$ under which the observed collection of 0/1 outcomes is most likely; logit uses $G=\Lambda$, whereas probit uses $G=\Phi$.
> <!-- bilingual-en:end -->

似然使用每个观测的完整概率，而不是最小化 0/1 分类错误数。因此估计后应把系数转回预测概率、离散变化或边际效应；在某个阈值下分类正确并不是 MLE 的定义。

<!-- bilingual-en:start -->
The likelihood uses the full probability assigned to every observation rather than minimizing the number of 0/1 classification errors. After estimation, coefficients should therefore be translated into predicted probabilities, discrete changes, or marginal effects. Correct classification at a particular threshold is not the definition of MLE.
<!-- bilingual-en:end -->

经典 MLE 标准误依赖概率规格和独立结构。若观测在个人、学校或地区内相关，要使用与抽样和分配层级匹配的聚类或其他推断方法。稳健标准误可以放宽部分方差或相关结构要求，却不会把错误的条件均值函数改成正确模型。

<!-- bilingual-en:start -->
Classical MLE standard errors rely on the probability specification and independence structure. If observations are correlated within individuals, schools, or regions, use clustered or other inference methods that match the sampling and assignment levels. Robust standard errors can relax some requirements on variances or dependence, but they cannot turn an incorrect conditional-mean function into the correct model.
<!-- bilingual-en:end -->

优化器显示“收敛”只说明找到了数值驻点附近，不能证明识别、支持或规格正确。[[完全分离]] 时 logit 和 probit 的似然都可沿某个方向持续改善而没有有限最大值；换起始值或增加迭代次数不是根治。

<!-- bilingual-en:start -->
An optimizer reporting “convergence” means only that it has reached the neighborhood of a numerical stationary point; it does not establish identification, adequate support, or correct specification. Under [[完全分离|complete separation]], both the logit and probit likelihoods can keep improving along a direction without attaining a finite maximum. Changing starting values or allowing more iterations does not solve the underlying problem.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §17.1：核对 Bernoulli 对数似然及 logit/probit 的极大似然估计。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., §17.1: verifies the Bernoulli log-likelihood and maximum-likelihood estimation of logit and probit models.
<!-- bilingual-en:end -->
- StataCorp, [*logit*](https://www.stata.com/manuals/rlogit.pdf) 与 [*probit*](https://www.stata.com/manuals/rprobit.pdf) 手册，Methods and formulas：交叉核对两种链接下的似然、优化与方差估计。
<!-- bilingual-en:start -->
- StataCorp's [*logit*](https://www.stata.com/manuals/rlogit.pdf) and [*probit*](https://www.stata.com/manuals/rprobit.pdf) manuals, Methods and formulas: cross-check the likelihood, optimization, and variance estimation under the two link functions.
<!-- bilingual-en:end -->
- [[标准误口径匹配]]：复用独立、异方差与聚类依赖下的推断边界。
<!-- bilingual-en:start -->
- [[标准误口径匹配|Matching the standard-error method to the dependence structure]]: supplies the inference boundaries under independence, heteroskedasticity, and clustered dependence.
<!-- bilingual-en:end -->

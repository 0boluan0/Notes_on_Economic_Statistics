---
aliases:
  - "可积损失的 Expected Shortfall 是最坏一减 alpha 概率质量的平均损失"
  - "可积损失的 Expected Shortfall 可写成上尾分位积分，严格超越条件均值只在分位点概率恰好对齐时成立"
  - Expected Shortfall as an integrated upper-tail quantile
  - 离散分布中的 ES 定义
student_os: knowledge-atom
atom_id: RM-VAR-003
atom_set: var-es-backtesting
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[风险度量口径]]"
  - "[[VaR定义]]"
  - "[[矩存在性的使用边界]]"
related:
  - "[[POT尾部ES]]"
  - "[[FRTB 市场风险]]"
leads_to:
  - "[[ES严格超越均值边界]]"
  - "[[ES一致性]]"
  - "[[ES是光谱特例]]"
  - "[[VaR-ES联合识别]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 可积损失的 Expected Shortfall 是最坏一减 alpha 概率质量的平均损失
<!-- bilingual-en:start -->
*For an integrable loss, Expected Shortfall is the average loss in the worst one-minus-alpha probability mass*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对可积损失 $L$，置信水平 $\alpha$ 下的 ES 把最坏 $1-\alpha$ 概率质量完整取出并求平均。它不是随意挑选“超过某个数”的观测，而是固定要平均多少尾部概率质量。
>
> <!-- bilingual-en:start -->
> For an integrable loss $L$, ES at confidence level $\alpha$ takes the complete worst $1-\alpha$ probability mass and averages it. It does not select an arbitrary exceedance set; the amount of tail probability to be averaged is fixed.
> <!-- bilingual-en:end -->

令 $q_u(L)$ 为损失的左分位点。一般定义为

$$
\operatorname{ES}_{\alpha}(L)
=\frac{1}{1-\alpha}
\int_\alpha^1q_u(L)\,du.
$$

令 $q=q_\alpha(L)$，同一定义也可写成

$$
\operatorname{ES}_{\alpha}(L)
=\frac{
E\!\left[L\mathbf 1_{\{L>q\}}\right]
+q\left(F_L(q)-\alpha\right)
}{1-\alpha}.
$$

第一项纳入全部严格大于 $q$ 的损失；第二项从分位点 $q$ 上的概率质量中补入恰好需要的部分，使被平均的尾部概率总量严格等于 $1-\alpha$。

<!-- bilingual-en:start -->
The first term includes every loss strictly above $q$. The second takes exactly the required fraction of probability mass at the quantile so that the averaged tail probability is precisely $1-\alpha$.
<!-- bilingual-en:end -->

当阈值处概率恰好对齐时，这个量才可简写为严格超越 VaR 的条件均值；具体条件与离散反例见 [[ES严格超越均值边界]]。有限 ES 还要求相应尾部可积，见 [[矩存在性的使用边界]]。

<!-- bilingual-en:start -->
Only when probability aligns at the threshold can ES be shortened to a conditional mean above VaR; see [[ES严格超越均值边界|the strict-exceedance boundary for ES]] for the exact condition and a discrete counterexample. Finite ES also requires the relevant tail integrability; see [[矩存在性的使用边界|the boundary for using moments]].
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 ES 的定义要固定最坏 $1-\alpha$ 的概率质量，而不能只说“平均所有严格超过 VaR 的损失”？
>
> **答案：** 分位点上可能有概率质量；严格超越事件可能不足 $1-\alpha$，因此必须按定义从分位点质量中补足。
>
> <!-- bilingual-en:start -->
> **Self-check:** Why does ES fix the worst $1-\alpha$ probability mass instead of merely averaging all losses strictly above VaR?
>
> **Answer:** Probability mass may sit at the quantile, making the strict-exceedance event smaller than $1-\alpha$. The definition must then fill the missing tail mass from the quantile itself.
> <!-- bilingual-en:end -->

## 不能越界

- ES 是所指定损失分布的尾部函数，不是信用风险会计语境中的无条件 expected loss。
- CVaR、tail conditional expectation 等名称在不同文献中可能采用不同约定，必须先核对公式。
- ES 描述分位点以外的损失程度，不自动保证更易估计、校准或验证。

<!-- bilingual-en:start -->
- ES is a tail functional of the specified loss distribution, not unconditional expected loss in credit-accounting terminology.
- Names such as CVaR and tail conditional expectation can follow different conventions across sources; inspect the formula first.
- ES describes loss severity beyond a quantile but is not automatically easier to estimate, calibrate, or validate.
<!-- bilingual-en:end -->

## 来源与核验

- [Acerbi & Tasche, *On the Coherence of Expected Shortfall*](https://arxiv.org/abs/cond-mat/0104295)：核对连续与非连续分布中的 ES 定义。
- [Rockafellar & Uryasev, *Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)：核对一般损失分布和分位点概率质量的处理。
- [[矩存在性的使用边界]]：承载尾部矩不存在时不能默认 ES 有限的边界。

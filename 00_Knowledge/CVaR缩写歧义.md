---
aliases:
  - "CVaR 既可能指 Component VaR，也可能指 Conditional VaR，必须用公式消歧"
  - CVaR abbreviation ambiguity
  - CVaR 术语歧义
student_os: knowledge-atom
atom_id: RM-VAR-042
atom_set: var-es-backtesting
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[ES定义]]"
  - "[[成分VaR]]"
related:
  - "[[VaR贡献口径]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# CVaR 既可能指 Component VaR，也可能指 Conditional VaR，必须用公式消歧
<!-- bilingual-en:start -->
*CVaR can mean either Component VaR or Conditional VaR, so the formula must disambiguate the term*
<!-- bilingual-en:end -->

> [!summary] 同一缩写对应两个不同对象
> 在风险分摊文献中，CVaR 可能表示 [[成分VaR|Component VaR]]；在尾部风险与优化文献中，它常表示 Conditional Value-at-Risk，并通常与适当定义的 [[ES定义|Expected Shortfall]] 对应。只看缩写无法判断作者在说哪一个。
>
> <!-- bilingual-en:start -->
> In risk-allocation sources, CVaR can denote [[成分VaR|Component VaR]]. In tail-risk and optimization sources, it often denotes Conditional Value-at-Risk, usually corresponding to properly defined [[ES定义|Expected Shortfall]]. The abbreviation alone does not identify the object.
> <!-- bilingual-en:end -->

两类公式回答完全不同的问题：

$$
\text{Component VaR}_i
=x_i\frac{\partial\rho(x)}{\partial x_i}
$$

是在当前组合中分摊第 $i$ 项的局部风险贡献；而

$$
\operatorname{ES}_{\alpha}(L)
=\frac{1}{1-\alpha}\int_\alpha^1q_u(L)\,du
$$

是对损失分布的固定上尾概率质量求平均。前者带头寸下标并依赖组合导数，后者带置信水平并依赖损失分布尾部。

<!-- bilingual-en:start -->
The first expression allocates a local contribution to position $i$ in the current portfolio. The second averages a fixed upper-tail probability mass of a loss distribution. The former carries a position index and a portfolio derivative; the latter carries a confidence level and a loss tail.
<!-- bilingual-en:end -->

阅读或写作时应在首次出现处展开全称并给出公式。即使作者把 Conditional VaR 当作 ES 的别名，离散分布中也仍要检查其采用的是一般分布定义，不能机械写成严格超越 VaR 的条件均值。

<!-- bilingual-en:start -->
Expand the term and show its formula at first use. Even when Conditional VaR is used as a synonym for ES, discrete distributions still require the general-distribution definition rather than a mechanical strict-exceedance conditional mean.
<!-- bilingual-en:end -->

> [!question]- 自检
> 某文献写“CVaR$_i=x_i\partial\rho/\partial x_i$”，这里的 CVaR 是什么？
>
> **答案：** Component VaR，因为它按头寸 $i$ 分摊局部风险贡献，不是尾部条件风险度量。
>
> <!-- bilingual-en:start -->
> **Self-check:** A source writes “CVaR$_i=x_i\partial\rho/\partial x_i$.” What does CVaR mean here?
>
> **Answer:** Component VaR, because the formula allocates a local contribution to position $i$ rather than measuring a loss tail.
> <!-- bilingual-en:end -->

## 来源与核验

- *Journal of Risk* 编辑信, [Vol. 5 No. 2](https://www.risk.net/journal-of-risk/volume-5-number-2-winter-2002)：明确把 component VaR 缩写为 CVAR，核对该缩写在风险分摊文献中的实际用法。
- Tasche, [*Capital Allocation to Business Units and Sub-Portfolios: the Euler Principle*](https://arxiv.org/abs/0708.2542)：核对 component risk contribution 的持仓乘导数口径。
- [Rockafellar & Uryasev, *Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)：核对 Conditional Value-at-Risk 的一般损失分布口径。

---
aliases:
  - "在本组采用的口径下，边际 VaR 是组合 VaR 对某项头寸的局部偏导数"
  - Marginal VaR
student_os: knowledge-atom
atom_id: RM-VAR-026
atom_set: var-es-backtesting
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
  - "[[全微分]]"
related:
  - "[[VaR贡献口径]]"
  - "[[递增VaR]]"
leads_to:
  - "[[成分VaR]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 在本组采用的口径下，边际 VaR 是组合 VaR 对某项头寸的局部偏导数
<!-- bilingual-en:start -->
*In the convention used here, marginal VaR is the local derivative of portfolio VaR with respect to a position*
<!-- bilingual-en:end -->

> [!summary] 它回答“再加一点，风险怎样变”
> 若组合风险 $\rho(x)$ 在当前头寸 $x$ 处可微，第 $i$ 项的边际 VaR 是 $\partial\rho(x)/\partial x_i$。它是每单位头寸的局部风险斜率，不是该资产单独的 VaR。
>
> <!-- bilingual-en:start -->
> If portfolio risk $\rho(x)$ is differentiable at current positions $x$, marginal VaR for position $i$ is $\partial\rho(x)/\partial x_i$. It is a local risk slope per unit of position, not the stand-alone VaR of that asset.
> <!-- bilingual-en:end -->

$$
\operatorname{MVaR}_i(x)
=\frac{\partial\rho(x)}{\partial x_i},
\qquad
\rho(x+\Delta x_i e_i)-\rho(x)
\approx
\operatorname{MVaR}_i(x)\Delta x_i.
$$

有效对冲的边际 VaR 可以为负：在当前组合附近增加一点对冲头寸会降低总 VaR。这不表示该头寸自身无风险。

<!-- bilingual-en:start -->
An effective hedge can have negative marginal VaR: a small increase reduces total VaR near the current portfolio. That does not make the hedge riskless on its own.
<!-- bilingual-en:end -->

> [!question]- 自检
> 边际 VaR 为 $-2$ 表示什么？
>
> **答案：** 在当前点附近增加一单位该头寸，组合 VaR 约下降 2；结论只是一阶局部近似。
>
> <!-- bilingual-en:start -->
> **Self-check:** What does marginal VaR of $-2$ mean?
>
> **Answer:** Near the current portfolio, adding one unit of the position reduces portfolio VaR by approximately 2. This is only a first-order local approximation.
> <!-- bilingual-en:end -->

## 边界

- 不可微的分位点跳变不能强行解释为稳定梯度。
- 大额交易需要 [[递增VaR]] 的有限差分。
- 边际导数本身不要求各项加总到组合 VaR。

<!-- bilingual-en:start -->
- Quantile jumps may make a stable derivative unavailable.
- A large trade requires the finite difference in [[递增VaR|incremental VaR]].
- Marginal derivatives alone need not sum to portfolio VaR.
<!-- bilingual-en:end -->

## 来源与核验

- Hallerbach, [*Decomposing Portfolio Value-at-Risk: A General Analysis*](https://repub.eur.nl/pub/7723/1999-0342.pdf)，§2 式 (4)：直接核对本组采用的 marginal VaR 局部导数口径。
- Tasche, [*Capital Allocation to Business Units and Sub-Portfolios: the Euler Principle*](https://arxiv.org/abs/0708.2542)：核对风险度量的局部 Gateaux/偏导贡献。
- RiskMetrics Group, [*Risk Management: A Practical Guide*](https://www.msci.com/documents/10199/3c2dcea9-97be-4fb4-befe-a03b75c885aa)，§1.2：提供会把整项移除差称为 marginal VaR 的另一套实务命名，说明阅读时必须先核对公式。

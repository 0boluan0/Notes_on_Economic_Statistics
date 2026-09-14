---
aliases:
  - "本组采用的 VaR 贡献口径把边际、成分与递增 VaR 分别定义为局部斜率、Euler 分摊与有限交易前后差"
  - "VaR 风险贡献是把组合 VaR 的局部敏感度、Euler 分摊与有限交易影响区分开的归因框架"
  - VaR风险贡献
  - VaR 风险贡献分解
  - Marginal, component, and incremental VaR conventions
student_os: knowledge-atom
atom_id: RM-VAR-012
atom_set: var-es-backtesting
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
requires:
  - "[[VaR定义]]"
related:
  - "[[CVaR缩写歧义]]"
leads_to:
  - "[[边际VaR]]"
  - "[[成分VaR]]"
  - "[[递增VaR]]"
---

# 本组采用的 VaR 贡献口径把边际、成分与递增 VaR 分别定义为局部斜率、Euler 分摊与有限交易前后差
<!-- bilingual-en:start -->
*In the convention used here, marginal, component, and incremental VaR mean a local slope, an Euler allocation, and a finite before-and-after change, respectively*
<!-- bilingual-en:end -->

> [!summary] 本组先固定一套可计算口径
> [[边际VaR]] 问“再加一点头寸，组合 VaR 的局部斜率是多少”；[[成分VaR]] 问“当前头寸按 Euler 口径分到多少风险”；[[递增VaR]] 问“完成一笔有限交易后，组合 VaR 实际改变多少”。三者相关，但不能互换。
>
> <!-- bilingual-en:start -->
> [[边际VaR|Marginal VaR]] asks for the local slope of portfolio VaR after a tiny position change. [[成分VaR|Component VaR]] asks for the current position's Euler contribution. [[递增VaR|Incremental VaR]] asks for the finite change after a trade is completed. The three quantities are related but not interchangeable.
> <!-- bilingual-en:end -->

令组合头寸为 $x$，风险函数为

$$
\rho(x)=\operatorname{VaR}_{\alpha}(L(x)).
$$

- **局部斜率：** $\partial\rho(x)/\partial x_i$ 是第 $i$ 项的 [[边际VaR]]。
- **当前分摊：** $x_i\,\partial\rho(x)/\partial x_i$ 是第 $i$ 项的 [[成分VaR]]；只有在相应条件下，各成分才按 Euler 定理加总为组合 VaR。
- **有限变化：** $\rho(x+\Delta x)-\rho(x)$ 是交易 $\Delta x$ 的 [[递增VaR]]。

<!-- bilingual-en:start -->
- **Local slope:** $\partial\rho(x)/\partial x_i$ is marginal VaR for position $i$.
- **Current allocation:** $x_i\,\partial\rho(x)/\partial x_i$ is component VaR; the components sum to portfolio VaR by Euler's theorem only under the required conditions.
- **Finite change:** $\rho(x+\Delta x)-\rho(x)$ is incremental VaR for trade $\Delta x$.
<!-- bilingual-en:end -->

文献有时交换 marginal、incremental 与 component 的标签，因此读取公式比只看名称可靠。缩写 `CVaR` 还可能指另一类对象，见 [[CVaR缩写歧义]]。

<!-- bilingual-en:start -->
Some sources swap the labels marginal, incremental, and component, so the formula is more reliable than the name alone. The abbreviation `CVaR` can also refer to a different object; see [[CVaR缩写歧义|the ambiguity of CVaR]].
<!-- bilingual-en:end -->

> [!question]- 自检
> “某头寸的成分 VaR 为 20”是否足以推出删除整笔头寸后组合 VaR 精确下降 20？
>
> **答案：** 不足。成分 VaR 是当前点的分摊；删除整笔头寸是有限变化，应计算递增 VaR。
>
> <!-- bilingual-en:start -->
> **Self-check:** If a position's component VaR is 20, does deleting the entire position reduce portfolio VaR by exactly 20?
>
> **Answer:** Not necessarily. Component VaR is an allocation at the current portfolio, whereas deleting a whole position is a finite change measured by incremental VaR.
> <!-- bilingual-en:end -->

## 来源与核验

- Hallerbach, [*Decomposing Portfolio Value-at-Risk: A General Analysis*](https://repub.eur.nl/pub/7723/1999-0342.pdf)，§2 式 (4)–(6) 与 §3 式 (13)–(14)：核对本组采用的 marginal 导数、component 分摊和 incremental 新交易影响口径。
- Tasche, [*Capital Allocation to Business Units and Sub-Portfolios: the Euler Principle*](https://arxiv.org/abs/0708.2542)：核对局部导数与 Euler 风险分摊的条件。
- RiskMetrics Group, [*Risk Management: A Practical Guide*](https://www.msci.com/documents/10199/3c2dcea9-97be-4fb4-befe-a03b75c885aa)，§1.2：核对另一套会交换 marginal 与 incremental 标签的实务口径，因此本卡把公式约定放在名称之前。

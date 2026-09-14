---
aliases:
  - "成分 VaR 是当前头寸乘以该头寸的边际 VaR"
  - "成分 VaR 是正齐次可微条件下的持仓乘边际 VaR，并按 Euler 定理加总为组合 VaR"
  - Component VaR
student_os: knowledge-atom
atom_id: RM-VAR-027
atom_set: var-es-backtesting
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[边际VaR]]"
related:
  - "[[VaR贡献口径]]"
  - "[[CVaR缩写歧义]]"
leads_to:
  - "[[成分VaR加总]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 成分 VaR 是当前头寸乘以该头寸的边际 VaR
<!-- bilingual-en:start -->
*Component VaR is the current position multiplied by that position's marginal VaR*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 若组合风险为 $\rho(x)$ 且在当前头寸 $x$ 处可微，第 $i$ 项的成分 VaR 是当前持仓 $x_i$ 乘以局部风险斜率 $\partial\rho(x)/\partial x_i$。它把“每单位再增加一点会怎样”换算成“当前这整项对应多少局部风险贡献”。
>
> <!-- bilingual-en:start -->
> If portfolio risk $\rho(x)$ is differentiable at current positions $x$, component VaR for position $i$ is the current holding $x_i$ times the local risk slope $\partial\rho(x)/\partial x_i$. It converts a per-unit local sensitivity into a contribution associated with the current position.
> <!-- bilingual-en:end -->

$$
\operatorname{CVaR}_i(x)
=x_i\operatorname{MVaR}_i(x)
=x_i\frac{\partial\rho(x)}{\partial x_i}.
$$

例如头寸为 5、边际 VaR 为 2 时，该头寸的成分 VaR 为 10。这个 10 是当前组合点上的局部贡献量，不是删除整笔头寸后的有限风险差；后者由 [[递增VaR]] 回答。

<!-- bilingual-en:start -->
If the position is 5 and marginal VaR is 2, component VaR is 10. This is a local contribution at the current portfolio, not the finite risk change caused by deleting the whole position; the latter is measured by [[递增VaR|incremental VaR]].
<!-- bilingual-en:end -->

各项成分能否精确加总为组合 VaR 还需要风险函数一次正齐次且可微，见 [[成分VaR加总]]。缩写 `CVaR` 在此表示 component VaR；其术语冲突见 [[CVaR缩写歧义]]。

<!-- bilingual-en:start -->
Exact add-up to portfolio VaR additionally requires a differentiable risk functional that is positively homogeneous of degree one; see [[成分VaR加总|the component-VaR add-up theorem]]. Here `CVaR` denotes component VaR; see [[CVaR缩写歧义|the ambiguity of CVaR]] for the terminology conflict.
<!-- bilingual-en:end -->

> [!question]- 自检
> 当前头寸为 4、边际 VaR 为 $-1.5$ 时，成分 VaR 是多少？
>
> **答案：** $-6$。负号表示该头寸在当前组合附近具有对冲贡献，不表示它自身没有风险。
>
> <!-- bilingual-en:start -->
> **Self-check:** If the current position is 4 and marginal VaR is $-1.5$, what is component VaR?
>
> **Answer:** $-6$. The negative sign indicates a hedging contribution near the current portfolio; it does not make the position risk-free in isolation.
> <!-- bilingual-en:end -->

## 边界

- 分位点跳变可能使边际导数不存在，进而使这一定义不能直接使用。
- 成分 VaR 是当前点的局部贡献，不是大额交易的精确前后差。
- 成分为负可以表示组合内的对冲作用。

<!-- bilingual-en:start -->
- Quantile jumps can make the marginal derivative unavailable, so this definition cannot be applied directly.
- Component VaR is a local contribution at the current portfolio, not the exact before-and-after effect of a large trade.
- A negative component can represent hedging within the portfolio.
<!-- bilingual-en:end -->

## 来源与核验

- Hallerbach, [*Decomposing Portfolio Value-at-Risk: A General Analysis*](https://repub.eur.nl/pub/7723/1999-0342.pdf)，§2 式 (4)–(5) 与 §3 式 (13)–(14)：核对 component VaR 的命名、持仓乘 marginal VaR 的公式及 Euler 加总关系。
- Tasche, [*Capital Allocation to Business Units and Sub-Portfolios: the Euler Principle*](https://arxiv.org/abs/0708.2542)：核对持仓乘局部导数的风险贡献定义，以及它与有限差分的区别。

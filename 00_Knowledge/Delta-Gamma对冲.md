---
aliases:
  - "Delta-Gamma对冲先用非零Gamma工具抵销曲率再抵销新增Delta"
student_os: knowledge-atom
atom_id: FI-HEDGE-006
atom_type: method
status: source-checked
requires:
  - "[[Delta对冲]]"
  - "[[Gamma]]"
  - "[[Gamma中性]]"
  - "[[持仓Greeks聚合]]"
leads_to:
  - "[[多Greek对冲]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Delta-Gamma对冲先用非零Gamma工具抵销曲率再抵销新增Delta
<!-- bilingual-en:start -->
*Delta–gamma hedging first offsets curvature with a nonzero-gamma instrument, then offsets the resulting delta*
<!-- bilingual-en:end -->

对于同一标的 $S$、同一时点与其他输入约定，设原组合敏感度为 $(\delta_P,\Gamma_P)$，每多头单位期权工具为 $(\delta_H,\Gamma_H)$，且 $\Gamma_H\ne0$。一股同一现货的 Delta 为 $1$、Gamma 为 $0$，故期权数量 $a$ 与股票数量 $b$ 可依次求得：
<!-- bilingual-en:start -->
For one underlying $S$ and common time and input conventions, let the original portfolio have sensitivities $(\delta_P,\Gamma_P)$ and each long unit of option hedge have $(\delta_H,\Gamma_H)$, with $\Gamma_H\ne0$. Since one underlying share has delta 1 and gamma 0, option quantity $a$ and stock quantity $b$ follow sequentially:
<!-- bilingual-en:end -->

$$
\Gamma_P+a\Gamma_H=0\quad\Rightarrow\quad a=-\frac{\Gamma_P}{\Gamma_H},
\qquad b=-(\delta_P+a\delta_H).
$$

例如原组合 $\delta_P=-30,\Gamma_P=-6$，工具 $\delta_H=0.6,\Gamma_H=1.5$，则买入 $a=4$ 单位工具后，Gamma 为零，Delta 仍为 $-30+4\times0.6=-27.6$；再买入 $27.6$ 股才同时消去 Delta。这里的单位已经包含合约乘数，未假设实际市场允许任意碎份合约。
<!-- bilingual-en:start -->
With $\delta_P=-30,\Gamma_P=-6$ and $\delta_H=0.6,\Gamma_H=1.5$, buying four hedge units removes gamma but leaves delta $-27.6$. Buying 27.6 shares then removes delta. Units already include contract multipliers; the calculation does not assume that real contracts are infinitely divisible.
<!-- bilingual-en:end -->

这只是当前点针对 $S$ 的两个导数约束。两者同时为零不消去 Vega、交叉敏感度、高阶项或跳跃损失，也不保证未来继续中性。若交易工具几乎没有 Gamma，比例可能很大；还须检查成本、流动性与其他新增敞口。多工具同时匹配更多敏感度时，使用 [[多Greek对冲]]。
<!-- bilingual-en:start -->
These are two derivative constraints on $S$ at the current point. They do not remove vega, mixed sensitivities, higher-order terms, or jump losses, nor preserve neutrality into the future. A nearly zero hedge gamma can require a very large position. Costs, liquidity, and newly introduced exposures still matter; use [[多Greek对冲|multi-Greek hedging]] for additional simultaneous constraints.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Kohn／Allen，Section 5，PDF 第 5–6、8 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开导数聚合、添加独立工具后可施加的敏感度约束及 Gamma 的对冲含义。本卡二元解法将这些导数约束用于 Gamma 与 Delta，公式和数值为独立代数核验，不是原文交易报价。
<!-- bilingual-en:start -->
- The reopened NYU notes, pp. 5–6 and 8, support derivative aggregation, sensitivity constraints with additional independent instruments, and gamma's hedging role. The two-equation solution and numerical example are independent algebraic applications, not quoted market prices.
<!-- bilingual-en:end -->

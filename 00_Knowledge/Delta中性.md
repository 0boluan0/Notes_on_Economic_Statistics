---
aliases:
  - "Delta中性是组合在当前点对指定标的价格的一阶敏感度为零"
student_os: knowledge-atom
atom_id: FI-HEDGE-004
atom_type: definition
status: source-checked
requires:
  - "[[Delta]]"
  - "[[持仓Greeks聚合]]"
related:
  - "[[Gamma]]"
leads_to:
  - "[[Delta对冲]]"
  - "[[动态Delta对冲]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Delta中性是组合在当前点对指定标的价格的一阶敏感度为零
<!-- bilingual-en:start -->
*Delta neutrality means zero first-order portfolio sensitivity to a specified underlying price at the current point*
<!-- bilingual-en:end -->

Delta 中性指在当前估值时点、标的价格 $S_0$ 和保持其他输入不变的约定下，冻结持仓组合 $V$ 满足 $\partial V/\partial S=0$。它消去该方向的一阶价格项，不要求组合价值为零，也不消去所有损益。
<!-- bilingual-en:start -->
A delta-neutral portfolio has $\partial V/\partial S=0$ at the current valuation time and underlying price $S_0$, under a stated convention for holding other inputs fixed. This removes the first-order price term in that direction; it does not require zero portfolio value or eliminate all P&L.
<!-- bilingual-en:end -->

例如 100 份、每份对应一股的期权，当前每份 Delta 为 $0.6$；配上做空 60 股，组合 Delta 为 $100\times0.6-60=0$。但若每份 Gamma 为 $0.04$，组合 Gamma 仍为 $4$。在同一时点、其他输入固定的小价格变化下，二阶价格项仍为 $\frac12\times4(\Delta S)^2$；其符号与大小不是全部期间损益。
<!-- bilingual-en:start -->
One hundred one-share options with delta 0.6 each, hedged by shorting 60 shares, have zero aggregate delta. If each option has gamma 0.04, portfolio gamma remains 4. A small same-time price move still produces the quadratic term $\frac12\times4(\Delta S)^2$ with other inputs fixed. That term alone is not total holding-period P&L.
<!-- bilingual-en:end -->

Delta 会随价格、时间与波动率等输入变化。多标的组合只令一个分量为零，也不等于整个 Delta 向量为零。Gamma、Vega、跳跃、基差、融资、流动性及模型风险是否仍重要，要分别判断；“中性”始终要连同因子与当前状态说明。
<!-- bilingual-en:start -->
Delta changes with price, time, volatility, and other inputs. Setting one component to zero in a multi-underlying portfolio does not make its entire delta vector zero. Gamma, vega, jumps, basis, funding, liquidity, and model risk must be assessed separately. Neutrality always refers to specified factors and a current state.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Kohn／Allen，Section 5，PDF 第 5–6 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开 $n_1\Delta_1+n_2\Delta_2+n_s=0$ 及其一阶不敏感解释；原文的零价值约束是复制组合的额外条件，不纳入 Delta 中性定义。算例的 Delta 与 Gamma 分别核算。
<!-- bilingual-en:start -->
- The reopened NYU notes, pp. 5–6, support the zero-delta constraint and its first-order interpretation. Their zero-value condition is an additional replication constraint, not part of delta neutrality itself. Delta and gamma in the example were checked separately.
<!-- bilingual-en:end -->

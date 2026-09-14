---
aliases:
  - "共同因子与报价口径下固定持仓的Greeks按数量和合约乘数带符号相加"
student_os: knowledge-atom
atom_id: FI-HEDGE-003
atom_type: theorem
status: source-checked
requires:
  - "[[Greeks]]"
  - "[[导数线性法则]]"
  - "[[Greek报价换算]]"
related:
  - "[[组合DV01聚合]]"
leads_to:
  - "[[Delta中性]]"
  - "[[Gamma中性]]"
  - "[[多Greek对冲]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# 共同因子与报价口径下固定持仓的Greeks按数量和合约乘数带符号相加
<!-- bilingual-en:start -->
*For common factors and quotation conventions, fixed-position Greeks add with signed quantities and contract multipliers*
<!-- bilingual-en:end -->

在同一估值时点、币种和模型约定下，设每单位报价价值为 $v_j(x)$，带符号合约数量为 $n_j$，合约乘数为 $m_j$。保持数量与乘数不随本次风险扰动改变，若所需导数存在，则
<!-- bilingual-en:start -->
At one valuation time, currency, and model convention, let $v_j(x)$ be value per quoted unit, $n_j$ signed contract quantity, and $m_j$ the contract multiplier. Holding quantities and multipliers fixed through the risk perturbation, the following identities hold wherever the derivatives exist:
<!-- bilingual-en:end -->

$$
V(x)=\sum_j n_jm_jv_j(x),\qquad
\partial_a V=\sum_jn_jm_j\partial_a v_j,\qquad
\partial_b\partial_a V=\sum_jn_jm_j\partial_b\partial_a v_j.
$$

因此 Delta、Gamma、Vega 等可按这条规则聚合，但导数必须针对**同一个输入及同一种尺度**。两个不同标的的 Delta 不是同一个方向；两个不同波动率节点的 Vega 也不能仅因名称相同就相抵。若报告已经是整个头寸的 Greek，不要再乘一次数量。
<!-- bilingual-en:start -->
Delta, gamma, vega, and other Greeks aggregate this way only when they refer to the **same input and scale**. Deltas on different underlyings are different directions, as are vegas on different volatility nodes. A Greek already reported for the whole position must not be multiplied by quantity again.
<!-- bilingual-en:end -->

例如持有 10 份期权，每份对应 100 股，每股报价 Delta 为 $0.6$；另做空 300 股同一股票。组合 Delta 为 $10\times100\times0.6-300=300$ 股等价敞口。若每股 Gamma 为 $0.02$，期权部分 Gamma 为 $20$；股票自身 Gamma 为零，不能抵销这项曲率。
<!-- bilingual-en:start -->
Ten option contracts, each covering 100 shares with delta 0.6 per quoted share, plus a short position of 300 shares have delta $10\times100\times0.6-300=300$ share-equivalents. With gamma 0.02 per quoted share, the options contribute gamma 20. The stock's zero gamma does not offset that curvature.
<!-- bilingual-en:end -->

这不是市值加权平均，也不能先取绝对值。若持仓数量本身按 $x$ 改变，对整个交易策略求导会出现持仓变化项；那不是本卡的“冻结持仓 Greeks”。策略的交易与现金账须另按 [[自融资策略]] 处理。跨币种时，若汇率也是风险因子，还要对换汇关系求导，而不只是用固定汇率乘一个数字。
<!-- bilingual-en:start -->
This is neither a market-value-weighted average nor a sum of absolute values. If holdings change with $x$, differentiating the trading strategy introduces holding-change terms; those are not frozen-position Greeks. Trades and cash must instead follow [[自融资策略|self-financing accounting]]. When FX is itself a risk factor, differentiate the currency-conversion relation rather than merely applying a fixed conversion rate.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Kohn／Allen，Derivative Securities，Section 5，PDF 第 5–6、8 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开组合价值、Delta／Vega 约束及买卖双方符号说明；支持固定数量的敏感度求和。合约乘数与 10 份期权算例按导数线性法则独立核验。
<!-- bilingual-en:start -->
- The reopened NYU notes, pp. 5–6 and 8, support portfolio-value differentiation, delta/vega constraints, and opposite signs for long and short positions. Contract scaling and the ten-contract example were independently checked using derivative linearity.
<!-- bilingual-en:end -->

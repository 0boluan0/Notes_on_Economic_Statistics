---
aliases:
  - "Delta对冲用共同价格因子的带符号敏感度求出抵销头寸"
student_os: knowledge-atom
atom_id: FI-HEDGE-005
atom_type: method
status: source-checked
requires:
  - "[[Delta中性]]"
  - "[[持仓Greeks聚合]]"
related:
  - "[[DV01对冲]]"
leads_to:
  - "[[Delta-Gamma对冲]]"
  - "[[动态Delta对冲]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Delta对冲用共同价格因子的带符号敏感度求出抵销头寸
<!-- bilingual-en:start -->
*Delta hedging determines an offsetting position from signed sensitivities to a common price factor*
<!-- bilingual-en:end -->

先确定要对冲的价格 $S$、共同报价单位和当前估值约定。若原组合 Delta 为 $\delta_P$，每一多头单位对冲工具的 Delta 为 $\delta_H\ne0$，保持本次计算的持仓固定，则零 Delta 的工具数量为
<!-- bilingual-en:start -->
First identify the price $S$, common units, and current valuation convention. With portfolio delta $\delta_P$ and delta $\delta_H\ne0$ per long unit of hedge instrument, the quantity that makes frozen-position delta zero is:
<!-- bilingual-en:end -->

$$\delta_P+h\delta_H=0,\qquad h=-\frac{\delta_P}{\delta_H}.$$

若工具就是同一现货股票，一股的 Delta 为 $1$，所以 $h=-\delta_P$。例如原组合 Delta 为 $+600$ 股等价敞口，须做空 600 股；若每份对冲合约的 Delta 为 $+50$ 股等价敞口，则须做空 12 份。买入正 Delta 的工具会增加而不是抵销正 Delta。
<!-- bilingual-en:start -->
One share of the same stock has delta 1, giving $h=-\delta_P$. A portfolio delta of +600 share-equivalents therefore requires shorting 600 shares, or 12 contracts if each long contract has delta +50 share-equivalents. Buying a positive-delta instrument adds to a positive exposure rather than offsetting it.
<!-- bilingual-en:end -->

还须分清目标总头寸和新增订单。若 $\delta_P$ 只包含原组合、不含旧对冲，$h$ 是对冲工具的目标总头寸，已有 $h_{old}$ 时新增交易为 $h-h_{old}$。例如原组合当前 Delta 为 $27600$，目标为 $-27600$ 股，原已做空 $30000$ 股，就应买回 $2400$ 股。若输入的 $\delta_P$ 已包含旧对冲，方程直接求出新增交易，不可再减一次旧头寸；本例全组合 Delta 为 $27600-30000=-2400$，直接得到买入 $2400$ 股。用 $\delta_{new}\approx\delta_{old}+\Gamma\Delta S$ 推算的新 Delta 只是局部近似，还须考虑时间和其他输入变化。
<!-- bilingual-en:start -->
Distinguish a target total position from an incremental order. If $\delta_P$ excludes the existing hedge, $h$ is the target total hedge and the order is $h-h_{old}$. For a current unhedged delta of 27,600, the target is −27,600 shares; an existing short of 30,000 shares requires buying back 2,400. If $\delta_P$ already includes that hedge, the equation gives the incremental order directly: net delta $27600-30000=-2400$ requires buying 2,400 shares, without subtracting the old hedge again. Estimating updated delta by $\delta_{new}\approx\delta_{old}+\Gamma\Delta S$ is only local and must account separately for time and other input changes.
<!-- bilingual-en:end -->

使用其他标的、远期或期货时，必须确认它对同一个 $S$ 的敏感度；不能把“远期价格变化 1”与“现货价格变化 1”当成同一扰动。若 $\delta_H=0$，该工具不能抵销非零 $\delta_P$。交易数量受整数或限制约束时，应记录实际剩余 Delta，而不宣称完全中性。
<!-- bilingual-en:start -->
For another underlying, a forward, or a future, determine its sensitivity to the same $S$. A unit move in a forward price is not automatically the same shock as a unit spot move. A zero-delta tool cannot offset nonzero portfolio delta. Quantity restrictions or integer contracts can leave residual delta that must be reported.
<!-- bilingual-en:end -->

上述方程只决定当前的一阶风险头寸，不说明购买资金从哪里来，也不保证对冲后的价值或收益。现金账户由 [[自融资策略]] 管理；市场和时间改变后，按 [[动态Delta对冲]] 更新头寸。费用、买卖价差与借券约束会改变实施结果。
<!-- bilingual-en:start -->
The equation determines current first-order exposure, not the source of financing or the value and return of the hedged book. [[自融资策略|Self-financing accounting]] governs cash, while [[动态Delta对冲|dynamic delta hedging]] updates holdings as markets and time change. Fees, bid–ask spreads, and stock-borrow constraints affect implementation.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Kohn／Allen，Section 5，PDF 第 5–6、8 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开股票与期权 Delta 约束、持仓符号及现货／远期敏感度的区别。一般比例式、目标与增量的区别由该线性约束独立推导；600 股、12 份及买回 2400 股的例子按共同尺度复算。
<!-- bilingual-en:start -->
- The reopened NYU notes, pp. 5–6 and 8, support stock/option delta constraints, position signs, and the distinction between spot and forward sensitivities. The ratio and total/incremental distinction follow independently from that constraint; the 600-share, 12-contract, and 2,400-share buyback examples were checked on a common scale.
<!-- bilingual-en:end -->

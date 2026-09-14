---
aliases:
  - "动态Delta对冲按当期信息再平衡股票并通过现金账户保存自融资资金约束"
student_os: knowledge-atom
atom_id: FI-HEDGE-009
atom_type: method
status: source-checked
requires:
  - "[[Delta对冲]]"
  - "[[自融资策略]]"
related:
  - "[[风险模拟路径依赖]]"
  - "[[BSM期权定价]]"
leads_to:
  - "[[离散对冲误差]]"
  - "[[Gamma与Theta的对冲损益]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# 动态Delta对冲按当期信息再平衡股票并通过现金账户保存自融资资金约束
<!-- bilingual-en:start -->
*Dynamic delta hedging rebalances stock using current information while preserving the self-financing cash constraint*
<!-- bilingual-en:end -->

对于卖出的一份期权，建立股票与现金构成的对冲资产组合 $X$。在时点 $t_i$ 用当前信息算出期权多头 Delta $\delta_i$，持股 $h_i=\delta_i$ 抵销期权空头的 Delta；现金余额由既有资金决定，不重新设为“让对冲资产恰等于当前期权价格”的任意数。
<!-- bilingual-en:start -->
For one short option, form a hedge asset portfolio $X$ of stock and cash. At $t_i$, compute the long option's delta $\delta_i$ from current information and hold $h_i=\delta_i$ shares to offset the short delta. Existing funds determine cash; it must not be arbitrarily reset to make hedge assets equal the option's current price.
<!-- bilingual-en:end -->

在无分红、无费用、可按同一固定连续复利率 $r$ 借贷的模型中，令 $b_i$ 为再平衡后现金金额，$C_0$ 为收取的期权价格。先初始化，再让旧持仓经历下一段市场变化，最后在新价格交易：
<!-- bilingual-en:start -->
Assume no dividends or costs and borrowing/lending at one constant continuously compounded rate $r$. Let $b_i$ be post-rebalancing cash and $C_0$ the option premium received. Initialise the hedge, carry old holdings through the next interval, then trade at the new price:
<!-- bilingual-en:end -->

$$
X_0=C_0,\quad h_0=\delta_0,\quad b_0=X_0-h_0S_0,
$$
$$
X_{i+1}^{pre}=h_iS_{i+1}+b_i e^{r(t_{i+1}-t_i)},\qquad
h_{i+1}=\delta_{i+1},\qquad
b_{i+1}=X_{i+1}^{pre}-h_{i+1}S_{i+1}.
$$

因此再平衡后的 $X_{i+1}$ 等于 $X_{i+1}^{pre}$，但未必等于 $C_{i+1}$。若在该时点按模型价格平仓，期权空头与对冲资产合计损益为 $X_{i+1}-C_{i+1}$；实际成交价格和费用应替换或补充模型金额。
<!-- bilingual-en:start -->
Post-trade $X_{i+1}$ equals $X_{i+1}^{pre}$, but need not equal $C_{i+1}$. Closing at model prices gives combined short-option and hedge P&L $X_{i+1}-C_{i+1}$. Realised execution prices and costs must replace or supplement those model amounts.
<!-- bilingual-en:end -->

例如一份对应一股的欧式看涨期权，$S_0=K=100$、剩余期限 $0.25$ 年、无分红、$r=0$，始终用波动率 $0.2$ 的 [[BSM期权定价|BSM 模型]]估值及计算 Delta。每段为 $1/252$ 年，观察到价格路径 $100\to102\to101\to103$。允许分数股、忽略费用，前两次再平衡的完整账如下；表中显示舍入值，计算使用未舍入数。
<!-- bilingual-en:start -->
Consider a one-share European call with $S_0=K=100$, initial maturity 0.25 years, no dividends, and $r=0$. Use [[BSM期权定价|BSM pricing]] with volatility 0.2 throughout for marks and deltas. Each interval is $1/252$ year and the observed path is $100\to102\to101\to103$. Allow fractional shares and ignore costs. The ledger through two rebalances is shown rounded; calculations use unrounded values.
<!-- bilingual-en:end -->

| 时点 | 股价 | 期权价值 $C_i$ | 再平衡后持股 $h_i$ | 再平衡后现金 $b_i$ | 对冲资产 $X_i$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 3.987761 | 0.519939 | −48.006119 | 3.987761 |
| 1 | 102 | 5.074846 | 0.598404 | −56.009579 | 5.027639 |
| 2 | 101 | 4.463754 | 0.559744 | −52.104953 | 4.429235 |
<!-- bilingual-en:start -->
| Time | Stock | Option $C_i$ | Post-trade shares $h_i$ | Post-trade cash $b_i$ | Hedge assets $X_i$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 100 | 3.987761 | 0.519939 | −48.006119 | 3.987761 |
| 1 | 102 | 5.074846 | 0.598404 | −56.009579 | 5.027639 |
| 2 | 101 | 4.463754 | 0.559744 | −52.104953 | 4.429235 |
<!-- bilingual-en:end -->

时点 1 买股支出 $8.0034598941$；时点 2 卖股收入 $3.9046258841$，都进入现金账户。到时点 3 不再调仓，而是卖出已有股票并按 $C_3=5.6297712636$ 买回期权空头：
<!-- bilingual-en:start -->
The first rebalance buys stock for 8.0034598941; the second sells stock for 3.9046258841. Both trades change cash. At time 3, stop rebalancing, sell the remaining stock, and buy back the short option at $C_3=5.6297712636$:
<!-- bilingual-en:end -->

$$
X_3=0.5597444366993758\times103-52.10495342620366
=5.548723553832048,
$$
$$\Pi_3=X_3-C_3=-0.08104770975103293.$$

初始期权收款恰好建立 $X_0$，所以合计初始净投入为零；最终负数是这个无费用、零利率算例的平仓损益，而不是外部注资。时点 1 若强行令 $b_1=C_1-h_1S_1$，会额外加入 $C_1-X_1=0.0472071631$，从而掩盖离散误差。
<!-- bilingual-en:start -->
The initial premium exactly funds $X_0$, so the combined book begins at zero net investment. The final negative number is closeout P&L in this zero-rate, costless example, not an external contribution. Resetting time-1 cash to $C_1-h_1S_1$ would silently inject $C_1-X_1=0.0472071631$ and hide the discrete hedge error.
<!-- bilingual-en:end -->

这条方法不保证复制成功。融资不对称、分红、借券费、保证金、成交价差和模型变化需要相应现金规则；不能因为更频繁更新 Delta 就忽略它们。对冲频率和模拟网格也不是同一选择。
<!-- bilingual-en:start -->
The procedure does not guarantee replication. Asymmetric funding, dividends, stock-borrow fees, margins, execution spreads, and model changes require corresponding cash rules. More frequent delta updates do not remove those requirements, and hedge frequency is distinct from simulation-grid resolution.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Haugh，Derivatives Pricing & Hedging，PDF 第 18–19 页，式 (25)–(26)](https://www.columbia.edu/~mh2078/QRM/DerivativesReview.pdf#page=18)：已重开并目视股票／现金再平衡、初始价格与复制损益。原式使用期间简单计息及分红项；本卡明确无分红并改用连续复利现金增长，三步 BSM 数值为独立复算。
- [同一讲义 PDF 第 9 页式 (13) 与第 14 页](https://www.columbia.edu/~mh2078/QRM/DerivativesReview.pdf#page=9)：已重开无分红 BSM 看涨公式和现货 Delta，并目视第 9 页；支持完整账的期权价格及持股数量计算。第 14 页把 Delta 等同实值概率的说法不用于本卡。
- [NYU Kohn／Allen，Section 5，PDF 第 4 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=4)：已重开“离散交易时，不能同时强制每次价值精确相等又保持自融资”的边界；用于检查额外注资反例。
<!-- bilingual-en:start -->
- Haugh pp. 18–19 and equations (25)–(26), reopened and visually checked, support stock/cash rebalancing, initial funding, and replication P&L. The source uses period interest and dividends; this card explicitly uses no dividends and continuously compounded cash growth. The three-step BSM ledger was independently recalculated.
- The same notes' equation (13) on p. 9 and spot delta on p. 14 were reopened, with p. 9 visually checked, for the ledger's prices and share quantities. The p. 14 claim identifying delta with the in-the-money probability is not used here.
- The reopened NYU p. 4 supports the incompatibility of forcing exact value matching at every discrete rebalance while retaining self-financing; it was checked against the hidden-cash-injection example.
<!-- bilingual-en:end -->

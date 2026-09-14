---
aliases:
  - "Delta对冲后的Gamma曲率项必须连同Theta与融资条件判断损益"
student_os: knowledge-atom
atom_id: FI-HEDGE-011
atom_type: model-boundary
status: source-checked
requires:
  - "[[Gamma]]"
  - "[[Theta]]"
  - "[[动态Delta对冲]]"
related:
  - "[[隐含波动率]]"
  - "[[Greeks损益归因]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Delta对冲后的Gamma曲率项必须连同Theta与融资条件判断损益
<!-- bilingual-en:start -->
*The gamma curvature term after delta hedging must be assessed together with theta and financing conditions*
<!-- bilingual-en:end -->

一个同刻、固定其他输入的价格冲击给出 $\frac12\Gamma(\Delta S)^2$，不等于持有期的对冲净收益。时间流逝会产生 Theta，股票与期权的建仓及借款有资金成本，市场波动率报价变化还可能产生 Vega 等项。正 Gamma 本身不是盈利保证。
<!-- bilingual-en:start -->
The term $\frac12\Gamma(\Delta S)^2$ for a same-time price shock with other inputs fixed is not net holding-period hedge profit. Time produces theta, option/stock positions and borrowing involve financing, and volatility-quote changes can add vega and other terms. Positive gamma alone does not guarantee profit.
<!-- bilingual-en:end -->

先限定 BSM 定价关系：常数波动率输入 $\sigma_m$、常数利率 $r$、连续股息率 $d$，$\Theta=C_t$ 按日历时间定义。该模型的价格方程给出
<!-- bilingual-en:start -->
First restrict attention to the BSM pricing relation with constant volatility input $\sigma_m$, rate $r$, dividend yield $d$, and calendar-time theta $\Theta=C_t$. Its pricing equation implies:
<!-- bilingual-en:end -->

$$\Theta+(r-d)S\delta+\frac12\sigma_m^2S^2\Gamma=rC.$$

在特别清楚的 $r=d=0$ 情形，$\Theta=-\frac12\sigma_m^2S^2\Gamma$。持有一份期权多头，做空当前 $\delta$ 股，并用现金账户使初始合计净投入为零；若短期不调仓、保持波动率输入不变，仅保留 Theta 与价格二阶项，则
<!-- bilingual-en:start -->
In the transparent case $r=d=0$, $\Theta=-\frac12\sigma_m^2S^2\Gamma$. Hold one long option, short its current delta in shares, and use cash to make initial combined investment zero. Over a short interval without rebalancing, keeping the volatility input fixed and retaining theta and quadratic price terms gives:
<!-- bilingual-en:end -->

$$
\Delta\Pi\approx\Theta\Delta t+\frac12\Gamma(\Delta S)^2
=\frac12\Gamma\left[(\Delta S)^2-\sigma_m^2S^2\Delta t\right].
$$

这是短期局部近似，不是任意路径的精确离散收益公式；时间交叉项、更高阶项与波动率变化没有被自动消除。非零利率或分红时，先把现金利息和空头股票的分红支付入账，不能只从期权价格变化扣除 $\delta\Delta S$ 就宣布净损益。
<!-- bilingual-en:start -->
This is a short-horizon local approximation, not an exact discrete profit formula for arbitrary paths. Time cross-terms, higher-order terms, and volatility moves do not vanish automatically. With rates or dividends, record cash interest and dividends owed on short stock before interpreting P&L; subtracting only $\delta\Delta S$ from the option change is insufficient.
<!-- bilingual-en:end -->

沿用 [[动态Delta对冲]] 的初始期权，$S=100,\sigma_m=0.2,\Gamma=0.0398443914$，故 $\Theta=-7.9688782819$ 元／年。一天 $\Delta t=1/252$ 内若 $\Delta S=2$，上述多头对冲近似为 $+0.0480662500$，完整重估为 $+0.0472071631$；若股价不动，局部估计则为 $-0.0316225329$。卖出期权并持有复制资产的符号相反。
<!-- bilingual-en:start -->
Using the initial option in [[动态Delta对冲|the dynamic hedge example]], $S=100,\sigma_m=0.2,\Gamma=0.0398443914$ and theta is −7.9688782819 currency units per year. For one day and a price move of 2, the long hedged option's approximation is +0.0480662500 versus full repricing of +0.0472071631. With no stock move, its local estimate is −0.0316225329. A short option with hedge assets has the opposite sign.
<!-- bilingual-en:end -->

多期中，每段损益还按当时的 $\Gamma S^2$ 加权。即使某种全期平均实现波动率低于卖出时的隐含波动率，也不能仅凭两个平均数判定盈利：高波动是否恰好出现在高 Gamma 区间、离散交易和成本都重要。这里的模型关系不是对真实波动率或可获收益的预测。
<!-- bilingual-en:start -->
Across periods, contributions are weighted by the prevailing $\Gamma S^2$. Comparing average realised volatility with the initial implied quote alone does not establish profit: high-volatility episodes may coincide with high gamma, and discrete trading and costs matter. The model identity is not a forecast of actual volatility or attainable returns.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Haugh，Derivatives Pricing & Hedging，PDF 第 17–19 页，式 (22)–(27)](https://www.columbia.edu/~mh2078/QRM/DerivativesReview.pdf#page=17)：已重开并目视 Gamma、Theta、carry 关系与对冲损益讨论。本卡局部公式从式 (23) 在 $r=d=0$ 下独立推导；不把第 19 页未显式累计非零利息的积分式当作一般到期金额公式。
- [NYU Kohn／Allen，Section 5，PDF 第 9–10 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=9)：已重开“平均实现波动较低仍可能亏损”的 Gamma 权重路径反例，支持不能只比较全期平均波动率。BSM 算例逐项复算。
<!-- bilingual-en:start -->
- Haugh pp. 17–19 and equations (22)–(27) were reopened and visually checked. The local formula here is independently derived from (23) at zero rates and dividends; the p. 19 integral, which does not explicitly accumulate nonzero interest, is not used as a general terminal-currency formula.
- The reopened NYU pp. 9–10 support the gamma-weighted path counterexample to judging hedge profit from average volatility alone. The BSM example was independently recalculated.
<!-- bilingual-en:end -->

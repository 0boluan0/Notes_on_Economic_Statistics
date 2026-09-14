---
aliases:
  - "Delta是价值对指定标的价格的一阶偏导数"
student_os: knowledge-atom
atom_id: FI-GREEK-002
atom_type: definition
status: source-checked
requires:
  - "[[Greeks]]"
related:
  - "[[Greek报价换算]]"
  - "[[Delta与实值概率]]"
leads_to:
  - "[[Gamma]]"
  - "[[Delta中性]]"
  - "[[Delta-Gamma价格近似]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Delta是价值对指定标的价格的一阶偏导数
<!-- bilingual-en:start -->
*Delta is the first partial derivative of value with respect to a specified underlying price*
<!-- bilingual-en:end -->

Delta 是在当前估值点、其他输入保持不变时，价值 $V$ 对指定标的价格 $S$ 的局部变化率；只要该偏导存在，定义为
<!-- bilingual-en:start -->
Delta is the local rate of change of value $V$ with respect to a specified underlying price $S$ at the current valuation point, holding the other inputs fixed. When the partial derivative exists, it is defined by:
<!-- bilingual-en:end -->

$$\Delta_S=\frac{\partial V}{\partial S},\qquad V(S+h)-V(S)=\Delta_S h+o(|h|).$$

它的单位是“价值单位／价格单位”。若 $V$ 是以元计的整个股票衍生品头寸价值，而 $S$ 的单位是元／股，Delta 的单位就是股；若 $V$ 是每股标的对应的期权报价，常见 Delta 数字则是两种价格的比率。每份报价不能直接当作整个合约的金额敞口，见 [[Greek报价换算]]。
<!-- bilingual-en:start -->
Its unit is value units per price unit. If $V$ is a whole equity-derivative position in currency and $S$ is currency per share, delta is measured in shares. If $V$ is the option premium quoted per underlying share, the familiar delta is a ratio of the two prices. A quoted per-unit sensitivity is not the whole contract's exposure; see [[Greek报价换算|Greek quote conversion]].
<!-- bilingual-en:end -->

例如某每单位期权报价的 Delta 为 $0.5$，标的价格小幅上升 $0.20$ 元，则一阶估计期权报价增加 $0.10$ 元。这里的“约”来自局部线性近似，不保证标的任意上涨 $1$ 元都恰好增加 $0.5$ 元；[[Gamma]] 描述 Delta 随同一价格变化的局部速度。
<!-- bilingual-en:start -->
If a per-unit option quote has delta $0.5$ and the underlying rises slightly by $0.20$, its first-order estimated increase is $0.10$. This is a local linear approximation, not a guarantee of an exact $0.5$ gain for every $1$ rise. [[Gamma|Gamma]] describes how delta changes locally with the same price.
<!-- bilingual-en:end -->

Delta 的符号、范围与解释取决于产品、持仓方向和输入坐标。对到期前且标的价格、执行价和波动率均为正的普通无股息 [[BSM期权定价|BSM 欧式看涨期权]]，现货 Delta 在 $0$ 与 $1$ 之间；这不是任意衍生品或组合的通用范围。Delta 也不能直接当作到期实值概率，见 [[Delta与实值概率]]；把现货 Delta 换成远期 Delta 必须改变求导坐标。
<!-- bilingual-en:start -->
Delta's sign, range, and interpretation depend on the instrument, position direction, and input coordinate. Before expiry and with positive underlying price, strike, and volatility, the spot delta of a standard non-dividend-paying [[BSM期权定价|BSM European call]] lies between zero and one. This is not a universal bound for derivatives or portfolios. Nor is delta automatically an expiry probability; see [[Delta与实值概率|delta versus the probability of expiring in the money]]. Forward delta differentiates with respect to a different coordinate from spot delta.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Courant，Kohn 与 Allen，Section 5，第 4–5、8 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=4)：已重开第 4–9 页并目视第 5、8 页，核对 Delta 偏导、无股息欧式期权的范围及现货／远期坐标区别。
- [OIC，Delta](https://www.optionseducation.org/advancedconcepts/delta)：已重开全文，仅采用其每单位报价、小价格变化、负持仓反号及估计不精确的说明；不采用把 Delta 等同到期实值概率的直觉说法。
<!-- bilingual-en:start -->
- [NYU Courant, Kohn and Allen, Section 5, pp. 4–5 and 8](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=4), reopened with the tables on pp. 5 and 8 visually checked, supports the derivative, the non-dividend-paying European-option bounds, and the spot/forward distinction.
- [OIC, Delta](https://www.optionseducation.org/advancedconcepts/delta), reopened in full, supports per-unit quotation, small-move estimates, and the sign of a short position. Its probability heuristic is not used as an identity.
<!-- bilingual-en:end -->

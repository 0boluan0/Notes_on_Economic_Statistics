---
aliases:
  - "Gamma是价值对指定标的价格的二阶偏导数"
student_os: knowledge-atom
atom_id: FI-GREEK-003
atom_type: definition
status: source-checked
requires:
  - "[[Delta]]"
related:
  - "[[交叉Gamma]]"
  - "[[Greek报价换算]]"
leads_to:
  - "[[Gamma中性]]"
  - "[[Delta-Gamma价格近似]]"
  - "[[Delta-Gamma对冲]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Gamma是价值对指定标的价格的二阶偏导数
<!-- bilingual-en:start -->
*Gamma is the second partial derivative of value with respect to a specified underlying price*
<!-- bilingual-en:end -->

Gamma 是其他输入固定时，价值 $V$ 对同一个标的价格 $S$ 的二阶偏导，也就是 [[Delta]] 对 $S$ 的变化率；所需导数存在时，
<!-- bilingual-en:start -->
Gamma is the second partial derivative of value $V$ with respect to the same underlying price $S$, holding the other inputs fixed. Equivalently, it is the rate of change of [[Delta|delta]] with respect to $S$. When the required derivatives exist:
<!-- bilingual-en:end -->

$$\Gamma_{SS}=\frac{\partial^2V}{\partial S^2}=\frac{\partial\Delta_S}{\partial S}.$$

Gamma 的单位是“价值单位／价格单位的平方”。正 Gamma 表示在当前点附近，价格上升使 Delta 增加；负 Gamma 表示相反方向。它不是价格变化的百分比，也不包含 [[Delta-Gamma价格近似|二阶价格近似]] 中另外出现的 $\tfrac12$。
<!-- bilingual-en:start -->
Gamma has units of value per squared price unit. Positive gamma means that, locally, a price rise increases delta; negative gamma means the opposite. Gamma is not a percentage price change and does not include the separate factor of $\tfrac12$ in the [[Delta-Gamma价格近似|second-order price approximation]].
<!-- bilingual-en:end -->

例如当前 Delta 为 $0.5$、Gamma 为 $0.04$，只让同一标的价格增加 $2$，则新的 Delta 可一阶估计为 $0.5+0.04\times2=0.58$。这是对 Delta 的近似，不是对价值增加量的计算，也不是 Gamma 在整个区间都保持不变的断言。
<!-- bilingual-en:start -->
With current delta $0.5$ and gamma $0.04$, a move of $2$ in the same underlying gives the first-order estimate $0.5+0.04\times2=0.58$ for the new delta. This estimates delta, not the value increment, and does not assert that gamma remains constant over the interval.
<!-- bilingual-en:end -->

对到期前且标的价格、执行价和波动率均为正的普通 [[BSM期权定价|BSM 欧式看涨与看跌期权]]，多头均有正 Gamma；空头反号。这不构成所有奇异期权或组合的符号规则。两个不同价格坐标之间的二阶作用属于 [[交叉Gamma]]，不能用单个 $\Gamma_{SS}$ 代替；行权或障碍等非光滑位置还须检查普通二阶导数是否存在。
<!-- bilingual-en:start -->
Before expiry and with positive underlying price, strike, and volatility, long standard [[BSM期权定价|BSM European calls and puts]] both have positive gamma; short positions reverse the sign. This is not a sign rule for every exotic option or portfolio. Second-order interactions between distinct price coordinates belong to [[交叉Gamma|cross-gamma]], not to a single $\Gamma_{SS}$. At nonsmooth exercise or barrier locations, the existence of an ordinary second derivative must also be checked.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Courant，Kohn 与 Allen，Section 5，第 5、8 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开并目视两页，核对 Gamma 二阶定义、所列欧式模型的正号及多空反号。单位与局部算例由定义独立核算。
- [OIC，Gamma](https://www.optionseducation.org/advancedconcepts/gamma)：已重开全文，支持用 Gamma 估计 Delta 变化及 Gamma 本身也会改变；其普通期权直觉不扩展为所有产品的定理。
<!-- bilingual-en:start -->
- [NYU Courant, Kohn and Allen, Section 5, pp. 5 and 8](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5), reopened and visually checked, supports the second derivative, the signs for the displayed European models, and long/short reversal. Units and the example were checked independently from the definition.
- [OIC, Gamma](https://www.optionseducation.org/advancedconcepts/gamma), reopened in full, supports estimating delta changes and recognizing that gamma itself changes. Its ordinary-option intuition is not extended to all products.
<!-- bilingual-en:end -->

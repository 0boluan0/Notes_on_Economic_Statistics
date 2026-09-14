---
aliases:
  - "Vega是价值对指定波动率输入的一阶偏导数"
student_os: knowledge-atom
atom_id: FI-GREEK-004
atom_type: definition
status: source-checked
requires:
  - "[[Greeks]]"
related:
  - "[[隐含波动率]]"
  - "[[Greek报价换算]]"
leads_to:
  - "[[多Greek对冲]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Vega是价值对指定波动率输入的一阶偏导数
<!-- bilingual-en:start -->
*Vega is the first partial derivative of value with respect to a specified volatility input*
<!-- bilingual-en:end -->

Vega 是在估值模型、合约及其他输入固定时，价值 $V$ 对指定波动率参数 $\sigma$ 的局部变化率。以年化波动率的小数形式作为坐标，例如 $20\%=0.20$，数学定义为
<!-- bilingual-en:start -->
Vega is the local rate of change of value $V$ with respect to a specified volatility parameter $\sigma$, keeping the valuation model, contract, and other inputs fixed. Using annualized volatility in decimal form, such as $20\%=0.20$, the mathematical definition is:
<!-- bilingual-en:end -->

$$\nu=\frac{\partial V}{\partial\sigma},\qquad \Delta V\approx\nu\,\Delta\sigma.$$

数学 Vega 按波动率增加 $1.00$ 计；市场常报每增加一个波动率百分点的价值响应，两者相差 $100$ 倍：$\nu_{\mathrm{1pp}}=0.01\nu$。例如 $\nu=20$ 时，波动率从 $20\%$ 到 $21\%$ 的一阶价值变化为 $0.20$，不是 $20$；若报价明确为“每百分点 Vega 为 $20$”，才应估为 $20$。
<!-- bilingual-en:start -->
Mathematical vega is scaled to a volatility change of $1.00$. Market quotations often use one volatility percentage point, giving $\nu_{\mathrm{1pp}}=0.01\nu$. If mathematical vega is $20$, a move from $20\%$ to $21\%$ gives a first-order value change of $0.20$, not $20$. The latter estimate applies only when the quote explicitly states vega of $20$ per percentage point.
<!-- bilingual-en:end -->

用于期权市场风险时，输入常是 [[隐含波动率]]，而不是刚观测到的历史波动率。若模型使用整张波动率曲面，必须声明改变哪个节点或采用怎样的共同移动；一个总 Vega 不能描述任意曲面变化。参数的年化方式、波动率模型和保持不变的条件也属于定义口径。
<!-- bilingual-en:start -->
In option market-risk applications the input is often [[隐含波动率|implied volatility]], not a newly observed historical-volatility estimate. With a volatility surface, specify the node being changed or the common shift being applied; one aggregate vega cannot describe every surface deformation. Annualization, the volatility model, and fixed-input conditions are part of the convention.
<!-- bilingual-en:end -->

Vega 的符号须由具体模型和持仓判断。对到期前且标的价格、执行价和波动率均为正的普通 [[BSM期权定价|BSM 欧式期权]]，多头的 Vega 为正；但“对价格线性”并不能单独推出 Vega 为零：例如局部价值函数 $V(S,\sigma)=\sigma S$ 对 $S$ 是直线，却有 $V_\sigma=S$。若定价表达式在指定条件下不依赖 $\sigma$，则该 Vega 为零。
<!-- bilingual-en:start -->
Vega's sign depends on the model and position. Before expiry and with positive underlying price, strike, and volatility, long standard [[BSM期权定价|BSM European options]] have positive vega. But linearity in price alone does not imply zero vega: the illustrative local value function $V(S,\sigma)=\sigma S$ is linear in $S$ yet has $V_\sigma=S$. If the valuation expression is independent of $\sigma$ under the specified conditions, its vega is zero.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Courant，Kohn 与 Allen，Section 5，第 5、8 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开并目视，核对对 $\sigma$ 求导及普通欧式模型的 Vega；曲面方向与线性反例按偏导定义独立说明。
- [OIC，Vega](https://www.optionseducation.org/advancedconcepts/vega)：已重开全文，核对隐含波动率输入以及一个百分点的报价尺度；不采用把隐含波动率无条件当作未来真实波动预测的说法。
<!-- bilingual-en:start -->
- [NYU Courant, Kohn and Allen, Section 5, pp. 5 and 8](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5), reopened and visually checked, supports differentiation with respect to $\sigma$ and vega in the displayed European models. Surface directions and the linearity counterexample are explained directly from partial derivatives.
- [OIC, Vega](https://www.optionseducation.org/advancedconcepts/vega), reopened in full, supports the implied-volatility input and percentage-point quotation. Its forecasting language is not adopted as an unconditional claim about future realized volatility.
<!-- bilingual-en:end -->

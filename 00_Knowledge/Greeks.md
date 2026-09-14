---
aliases:
  - "Greeks是在指定估值模型和输入口径下描述价值局部敏感度的导数集合"
  - Greek sensitivities
student_os: knowledge-atom
atom_id: FI-GREEK-001
atom_type: definition
status: source-checked
requires:
  - "[[偏导数]]"
related:
  - "[[风险因子映射]]"
  - "[[Greek报价换算]]"
leads_to:
  - "[[Delta]]"
  - "[[Gamma]]"
  - "[[Vega]]"
  - "[[Theta]]"
  - "[[Rho]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Greeks是在指定估值模型和输入口径下描述价值局部敏感度的导数集合
<!-- bilingual-en:start -->
*Greeks are derivatives describing local value sensitivities under a specified valuation model and input convention*
<!-- bilingual-en:end -->

Greeks（希腊值）是价值函数对选定输入的一阶或更高阶导数，用来描述当前状态附近的响应。必须先说明价值按什么币种和合约单位计、改变哪个输入，以及哪些输入保持不变。它们不是风险因子本身，也不是未来损益的概率预测。
<!-- bilingual-en:start -->
Greeks are first- or higher-order derivatives of a value function with respect to selected inputs. They describe responses near the current state. Specify the valuation currency and contract unit, the input being changed, and the inputs held fixed. Greeks are neither risk factors themselves nor probability forecasts of future P&L.
<!-- bilingual-en:end -->

例如写成 $V(S,\sigma,t,r)$，合约条款及其他模型输入固定：$S$ 是指定的标的价格，$\sigma$ 是指定波动率参数，$t$ 是日历时间，$r$ 是指定利率参数。[[Delta]]、[[Vega]]、[[Theta]]、[[Rho]] 分别对应这些坐标的一阶偏导；[[Gamma]] 对 $S$ 再求一次偏导。时间虽可进入价值函数，却不是因此成为随机市场冲击。
<!-- bilingual-en:start -->
For example, write $V(S,\sigma,t,r)$ with contract terms and other model inputs fixed: $S$ is the specified underlying price, $\sigma$ a volatility parameter, $t$ calendar time, and $r$ a rate parameter. [[Delta|Delta]], [[Vega|vega]], [[Theta|theta]], and [[Rho|rho]] are the corresponding first partial derivatives; [[Gamma|gamma]] differentiates twice with respect to $S$. Including time in the value function does not make it a random market shock.
<!-- bilingual-en:end -->

同一个产品，改用现货价或远期价、改变波动率曲面的移动方式，或改用不同利率曲线，得到的敏感度可以不同。只有输入坐标、保持不变的条件与 [[Greek报价换算|报价尺度]] 一致，数字才可直接比较。解析公式、自动微分或数值差分是计算方法；有限 bump 的差商通常只是导数的估计。
<!-- bilingual-en:start -->
The same instrument can have different sensitivities when the price coordinate changes from spot to forward, the volatility surface is moved differently, or another rate curve is used. Direct comparison requires matching coordinates, fixed-input conditions, and [[Greek报价换算|reporting scales]]. Analytic formulas, automatic differentiation, and finite differences are computational methods; a finite-bump quotient generally only estimates a derivative.
<!-- bilingual-en:end -->

例如，当前 Delta 为 $0.5$ 只给出其他输入固定时的小幅价格响应。它不能单独解释一天的全部收益；波动率、时间、利率、其他二阶项以及交易现金流都可能同时变化。
<!-- bilingual-en:start -->
A current delta of $0.5$ describes only a small price response with the other inputs fixed. It cannot explain an entire day's return on its own: volatility, time, rates, other second-order effects, and trading cash flows may also matter.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Courant，Kohn 与 Allen，Derivative Securities，2007，Section 5，第 5、8 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开正文并目视两页导数表，核对五种 Greek 及现货／远期输入的区别；数值差商的边界复用 [[偏导数]] 的极限定义。
- [OIC，Understanding Options Greeks](https://www.optionseducation.org/advancedconcepts/understanding-options-greeks)：已重开全文，支持多输入估值及 Greek 只提供局部理论估计、不保证精确价格变化的说明。
<!-- bilingual-en:start -->
- [NYU Courant, Kohn and Allen, Derivative Securities, 2007, Section 5, pp. 5 and 8](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5), reopened and visually checked, supplies the derivative tables and the spot/forward distinction. The finite-difference boundary follows the existing [[偏导数|limit definition of a partial derivative]].
- [OIC, Understanding Options Greeks](https://www.optionseducation.org/advancedconcepts/understanding-options-greeks), reopened in full, supports the multiple-input framework and the qualification that Greek estimates do not guarantee exact price changes.
<!-- bilingual-en:end -->

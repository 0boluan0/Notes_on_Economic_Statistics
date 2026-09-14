---
aliases:
  - "Rho是价值对指定利率输入的一阶偏导数"
student_os: knowledge-atom
atom_id: FI-GREEK-006
atom_type: definition
status: source-checked
requires:
  - "[[Greeks]]"
related:
  - "[[无风险利率口径]]"
  - "[[Greek报价换算]]"
  - "[[DV01]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Rho是价值对指定利率输入的一阶偏导数
<!-- bilingual-en:start -->
*Rho is the first partial derivative of value with respect to a specified interest-rate input*
<!-- bilingual-en:end -->

Rho 是当前估值点上，其他输入保持不变时，价值 $V$ 对指定利率参数 $r$ 的局部变化率。必须声明该利率的币种、期限或曲线位置、复利方式，以及“其他输入固定”是否包括现货或远期价格。以年利率的小数形式作坐标时，
<!-- bilingual-en:start -->
Rho is the local rate of change of value $V$ with respect to a specified rate parameter $r$, holding the other inputs fixed at the current valuation point. State its currency, maturity or curve location, compounding convention, and whether spot or forward prices are among the fixed inputs. With an annual rate expressed as a decimal:
<!-- bilingual-en:end -->

$$\rho_r=\frac{\partial V}{\partial r},\qquad \Delta V\approx\rho_r\,\Delta r.$$

数学 Rho 按利率增加 $1.00$ 计。每增加一个百分点的报价为 $0.01\rho_r$，每增加一个基点的一阶价值变化为 $10^{-4}\rho_r$。例如 $\rho_r=40$，利率从 $3\%$ 到 $3.25\%$，一阶变化约为 $40\times0.0025=0.10$。若与 [[DV01]] 使用完全相同的价值和冲击参数，其负导数约定给出 $\mathrm{DV01}=-10^{-4}\rho_r$。
<!-- bilingual-en:start -->
Mathematical rho is scaled to a rate increase of $1.00$. Its per-percentage-point quote is $0.01\rho_r$, while the first-order value change for a one-basis-point increase is $10^{-4}\rho_r$. With $\rho_r=40$, a move from $3\%$ to $3.25\%$ gives approximately $40\times0.0025=0.10$. For exactly the same value and shock parameter, the negative-derivative convention in [[DV01|DV01]] gives $\mathrm{DV01}=-10^{-4}\rho_r$.
<!-- bilingual-en:end -->

“看涨期权 Rho 为正”需要模型和固定输入条件。例如对到期前且标的价格、执行价和波动率均为正的普通无股息 [[BSM期权定价|BSM 看涨期权]]，保持现货价 $S$ 固定时 Rho 为正；若把同一类期权写成 $V=e^{-r\tau}B(F,\sigma,\tau)$ 并保持远期价 $F$ 固定，则 $\rho_r=-\tau V<0$（$\tau,V>0$）。两者改变的是不同的市场输入组合，并不矛盾。
<!-- bilingual-en:start -->
“Call rho is positive” requires a model and fixed-input convention. Before expiry and with positive underlying price, strike, and volatility, a standard non-dividend-paying [[BSM期权定价|BSM call]] has positive rho when spot price $S$ stays fixed. Expressing the same type of option as $V=e^{-r\tau}B(F,\sigma,\tau)$ and instead keeping forward price $F$ fixed gives $\rho_r=-\tau V<0$ for $\tau,V>0$. These are different combinations of input changes, so there is no contradiction.
<!-- bilingual-en:end -->

有多条曲线或多个期限节点时，应分别定义相应敏感度或声明共同移动方向。外汇期权的本币利率 Rho 与外币利率敏感度也不是同一个数；不能把单个 Rho 当作全部利率风险。
<!-- bilingual-en:start -->
With multiple curves or maturity nodes, define the individual sensitivities or specify a common shift direction. An FX option's domestic-rate rho and foreign-rate sensitivity are also different quantities. A single rho is not a complete description of interest-rate risk.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [NYU Courant，Kohn 与 Allen，Section 5，第 5、7–9 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5)：已重开正文并目视第 5、8 页，核对现货模型与固定远期价模型的 Rho 公式和不同符号。
- [OIC，Rho](https://www.optionseducation.org/advancedconcepts/rho)：已重开全文，支持每一个利率百分点的报价及固定其他定价输入的要求；符号按 NYU 的具体模型限定。基点与 DV01 的关系由相同坐标下的定义直接核算。
<!-- bilingual-en:start -->
- [NYU Courant, Kohn and Allen, Section 5, pp. 5 and 7–9](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5), reopened with pp. 5 and 8 visually checked, supports the spot-model and fixed-forward-price rho formulas and their different signs.
- [OIC, Rho](https://www.optionseducation.org/advancedconcepts/rho), reopened in full, supports percentage-point reporting and fixed-input conditions. Sign claims are restricted to the explicit NYU models. Basis-point scaling and the DV01 relation follow directly from definitions in the same coordinate.
<!-- bilingual-en:end -->

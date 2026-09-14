---
aliases:
  - "Greek报价换算要同时匹配输入变动单位与合约价值尺度"
student_os: knowledge-atom
atom_id: FI-GREEK-007
atom_type: method
status: source-checked
requires:
  - "[[Greeks]]"
related:
  - "[[Vega]]"
  - "[[Theta]]"
  - "[[Rho]]"
  - "[[持仓Greeks聚合]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Greek报价换算要同时匹配输入变动单位与合约价值尺度
<!-- bilingual-en:start -->
*Greek quote conversion must match both the input-change unit and the contract-value scale*
<!-- bilingual-en:end -->

使用一个 Greek 估算金额前，先确认它针对什么输入单位、什么价值单位及什么持仓方向。设每报价单位价值为 $v(x)$，数学一阶敏感度为 $g=\partial v/\partial x$；若报告把一个“报价步长”定义为输入增加 $a>0$，则报告敏感度为 $g_a=ag$，实际变动对应的步数为 $\Delta x/a$。
<!-- bilingual-en:start -->
Before converting a Greek into a monetary estimate, identify its input unit, value unit, and position direction. Let $v(x)$ be value per quoted unit and $g=\partial v/\partial x$ the mathematical first-order sensitivity. If one reporting step means an input increase of $a>0$, the reported sensitivity is $g_a=ag$, and an actual move contains $\Delta x/a$ such steps.
<!-- bilingual-en:end -->

$$\Delta v\approx g\,\Delta x=g_a\frac{\Delta x}{a}.$$

若每份合约明确对应 $M$ 个报价单位、持有带符号数量 $q$，并且 $q,M$ 在本次冲击中固定，则这一头寸的价值变化是 $qM\Delta v$，一阶估计为 $qMg\Delta x$。只有拿到每报价单位 Greek 时才乘 $qM$；已经报整个头寸的 Greek 时不可再乘。$M$ 应由合约规格确认，不能默认所有期权都是 $100$。
<!-- bilingual-en:start -->
If one contract explicitly contains $M$ quoted units and the signed holding is $q$, with both fixed through the shock, its value change is $qM\Delta v$, estimated to first order by $qMg\Delta x$. Multiply by $qM$ only when the Greek is quoted per unit; do not multiply again if it already describes the whole position. Obtain $M$ from the contract specification rather than assuming every option uses 100.
<!-- bilingual-en:end -->

例如，报价单位的数学 Vega 为 $20$，每份合约的乘数明确为 $100$，持有 $3$ 份多头。波动率由 $20\%$ 升至 $21.5\%$ 时，$\Delta\sigma=0.015$，金额变化一阶估计为
<!-- bilingual-en:start -->
Suppose mathematical vega per quoted unit is $20$, the specified contract multiplier is $100$, and the position is long three contracts. A volatility move from $20\%$ to $21.5\%$ gives $\Delta\sigma=0.015$ and the first-order monetary estimate:
<!-- bilingual-en:end -->

$$3\times100\times20\times0.015=90.$$

等价地，每百分点 Vega 为 $0.20$，变动为 $1.5$ 个百分点，计算 $3\times100\times0.20\times1.5$ 仍得 $90$。若头寸为空头 $q=-3$，同一变化的估计反号。这里 $20\%\to21.5\%$ 是增加 $1.5$ 个百分点，而不是相对增加 $1.5\%$。
<!-- bilingual-en:start -->
Equivalently, vega per percentage point is $0.20$ and the move is 1.5 percentage points, so $3\times100\times0.20\times1.5$ again gives $90$. A short holding $q=-3$ reverses the estimate's sign. The move from $20\%$ to $21.5\%$ is a rise of 1.5 percentage points, not a relative rise of $1.5\%$.
<!-- bilingual-en:end -->

Rho 的百分点与基点按同样规则换算；Theta 还要核对时间方向及日计数。Gamma 是二阶敏感度：若价格坐标每一步为 $a$，对“步数”求二阶导数的尺度是 $a^2\Gamma$，不能沿用一阶的 $a$ 倍。币种不同、输入方向不同或已做标准化的数字，应先恢复相同口径再比较与 [[持仓Greeks聚合|加总]]。
<!-- bilingual-en:start -->
Apply the same first-order rule to rho's percentage points and basis points; theta additionally requires the time direction and day-count convention. Gamma is second order: if one price step is $a$, its second derivative with respect to the step count is $a^2\Gamma$, not $a\Gamma$. Different currencies, shock directions, or normalized reports must be converted to compatible conventions before comparison or [[持仓Greeks聚合|aggregation]].
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [OIC，Vega](https://www.optionseducation.org/advancedconcepts/vega) 与 [Rho](https://www.optionseducation.org/advancedconcepts/rho)：已重开两页全文，核对一个百分点报价。例子与换算式为按导数单位独立构造，乘数 $100$ 是例子给定条件。
- [NYU Courant，Kohn 与 Allen，Section 5，第 8–9 页](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=8)：已重开正文、目视第 8 页，核对持仓反号和第 9 页的 Theta 日尺度；一阶与二阶坐标缩放分别由链式法则得到。
<!-- bilingual-en:start -->
- [OIC, Vega](https://www.optionseducation.org/advancedconcepts/vega) and [Rho](https://www.optionseducation.org/advancedconcepts/rho), both reopened in full, support percentage-point quotation. The conversion equations and example are independent dimensional derivations; the multiplier of 100 is an explicit example assumption.
- [NYU Courant, Kohn and Allen, Section 5, pp. 8–9](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=8), reopened with p. 8 visually checked, supports position-sign reversal and daily theta scaling. First- and second-order coordinate scaling follow from the chain rule.
<!-- bilingual-en:end -->

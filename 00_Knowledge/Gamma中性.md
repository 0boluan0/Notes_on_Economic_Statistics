---
aliases:
  - Gamma中性是组合在当前点对指定标的价格的二阶敏感度为零
  - Gamma neutrality
student_os: knowledge-atom
atom_id: FI-HEDGE-013
atom_type: definition
status: source-checked
requires:
  - "[[Gamma]]"
  - "[[持仓Greeks聚合]]"
related:
  - "[[Delta中性]]"
leads_to:
  - "[[Delta-Gamma对冲]]"
part_of:
  - "[[市场风险、Greeks 与动态对冲.canvas]]"
---

# Gamma中性是组合在当前点对指定标的价格的二阶敏感度为零
<!-- bilingual-en:start -->
*Gamma neutrality means zero second-order portfolio sensitivity to a specified underlying price at the current point*
<!-- bilingual-en:end -->

Gamma 中性是指：在当前估值点、持仓与其他输入固定时，组合价值 $V$ 对同一个标的价格 $S$ 的二阶偏导满足 $V_{SS}=0$。它抵销该方向的局部曲率，不要求 [[Delta中性|一阶敏感度也为零]]。
<!-- bilingual-en:start -->
Gamma neutrality means $V_{SS}=0$ at the current valuation point for a specified underlying price $S$, with holdings and other inputs fixed. It offsets local curvature in that direction without requiring [[Delta中性|zero first-order sensitivity]].
<!-- bilingual-en:end -->

例如两种同标的期权，每单位的 $(\Delta,\Gamma)$ 分别是 $(0.6,0.03)$ 与 $(0.2,0.02)$，数量已包含合约乘数。持有第一种 2 单位、做空第二种 3 单位，则

$$\Gamma_P=2(0.03)-3(0.02)=0,\qquad\Delta_P=2(0.6)-3(0.2)=0.6.$$

该组合已经 Gamma 中性，却仍有正 Delta。再卖空 0.6 股同一标的，才满足这两个目标；通用求解步骤见 [[Delta-Gamma对冲]]。
<!-- bilingual-en:start -->
Two long units of an option with $(\Delta,\Gamma)=(0.6,0.03)$ and three short units with $(0.2,0.02)$ give zero gamma but delta 0.6. Units already include contract scaling. Shorting another 0.6 underlying share satisfies both targets; see [[Delta-Gamma对冲|delta–gamma hedging]] for the general method.
<!-- bilingual-en:end -->

当前一点的 Gamma 为零，不保证整个价格区间线性，也不消去 Vega 或 [[交叉Gamma]]。它与“通过某些交易达到 Gamma 中性”的方法是两个对象：前者是状态条件，后者还须考虑工具、数量与实施约束。
<!-- bilingual-en:start -->
Zero gamma at one point does not make an entire price range linear or remove vega and [[交叉Gamma|cross-gamma]]. Neutrality is a state condition; achieving it through trades is a separate method involving instruments, quantities and implementation constraints.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Kohn–Allen, [*Derivative Securities*, §5](https://math.nyu.edu/~kohn/derivative.securities/2007/section5.pdf#page=5), pp. 5–6、8：组合敏感度由带符号持仓导数相加；此处以二阶导数为零定义状态，数量例独立核算。
- 课程 [[02_Economy/07_金融机构与风险管理/08_操作员如何管理风险暴露]] §1.2.2–1.2.3：区分 Gamma 中性条件与共同实现 Delta、Gamma 中性的交易。
<!-- bilingual-en:start -->
- The NYU notes, pp. 5–6 and 8, support signed aggregation of portfolio derivatives. This card defines the zero-second-derivative state and independently checks the quantities.
- Course Sections 1.2.2–1.2.3 distinguish gamma neutrality from trades jointly neutralising delta and gamma.
<!-- bilingual-en:end -->

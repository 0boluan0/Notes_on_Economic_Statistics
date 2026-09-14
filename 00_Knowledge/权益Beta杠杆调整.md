---
aliases:
  - 权益 Beta 同时包含经营风险与融资杠杆
  - Levered and unlevered beta
  - Equity beta leverage adjustment
student_os: knowledge-atom
atom_id: INV-CAPM-008
atom_set: capm-systematic-risk
atom_type: decomposition-method
status: source-checked
mastery_state: unassessed
requires:
  - "[[Beta历史估计]]"
related:
  - "[[银行资产负债恒等式]]"
part_of:
  - "[[CAPM、系统风险与资本成本.canvas]]"
leads_to:
  - "[[项目Beta匹配]]"
---

# 权益 Beta 同时包含经营风险与融资杠杆
<!-- bilingual-en:start -->
*Equity beta contains both operating risk and financial leverage*
<!-- bilingual-en:end -->

> [!summary] 原子分解
> 债务与权益共同索取企业总价值中的现金流。把总价值所含的资产和融资副作用都纳入口径后，使用市场价值权重有
> $$\beta_A=\frac{E}{D+E}\beta_E+\frac{D}{D+E}\beta_D.$$
> 这是对**总企业价值 beta** 的索取权分解；若目标是经营资产 beta，还要把非经营现金、税盾等单独处理。在相同经营资产风险下，提高债务占比会把更大残余波动集中到权益，因此权益 beta 通常上升。比较公司时必须先分离经营资产风险与融资杠杆，不能把观测权益 beta 直接当作项目风险。
> <!-- bilingual-en:start -->
> Debt and equity are joint claims on the cash flows contained in total firm value. Once all assets and financing side effects included in that value are kept in the same scope, market-value weights give $\beta_A=E\beta_E/(D+E)+D\beta_D/(D+E)$. This decomposes the beta of total firm value; recovering an operating-asset beta also requires separate treatment of non-operating cash, tax shields, and similar items. With the same operating risk, more debt concentrates residual variation in equity, so equity beta generally rises.
> <!-- bilingual-en:end -->

常见 Hamada 近似
$$
\beta_U=\frac{\beta_E}{1+(1-T_c)D/E}
$$
还假定债务 beta 近似为零，并对税盾风险、债务政策和税率作了简化。债务风险高、杠杆动态变化、利息扣除受限或税盾与经营资产风险不同的时候，应保留 $\beta_D$ 或把税盾用 APV 单独处理，不能把近似公式当恒等式。
<!-- bilingual-en:start -->
The common Hamada approximation, $\beta_U=\beta_E/[1+(1-T_c)D/E]$, additionally treats debt beta as approximately zero and simplifies tax-shield risk, debt policy, and taxation. With risky debt, changing leverage, constrained interest deductions, or tax shields of different risk, retain $\beta_D$ or value financing effects separately with APV; the approximation is not an identity.
<!-- bilingual-en:end -->

去杠杆和加杠杆都应使用债务与权益的市场价值，并保持债务定义、税率和现金处理一致。大量低 beta 现金会稀释整个公司的资产 beta；若目标是经营项目，应在可比公司中把非经营现金与经营资产分开。
<!-- bilingual-en:start -->
Both unlevering and relevering should use market values and consistent definitions of debt, tax rate, and cash. A large low-beta cash balance dilutes firm-wide asset beta; a project analysis should separate non-operating cash from comparable operating assets.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两家公司经营业务相同，但 A 几乎无债、B 高杠杆。为什么不能直接平均两者权益 beta 作为行业风险？
>
> **答案：** B 的权益 beta 还包含更强融资杠杆。应先在一致假设下去杠杆得到经营资产 beta，再汇总并按目标资本结构重新加杠杆。

## 来源与核验

- [Aswath Damodaran, Valuation Online, Session 5](https://www.stern.nyu.edu/~adamodar/pdfiles/valonlineslides/session5.pdf)：核对可比公司 beta、去杠杆/加杠杆公式、市场价值债务权益比与现金调整。
- [Hamada (1972), “The Effect of the Firm's Capital Structure on the Systematic Risk of Common Stocks”](https://doi.org/10.1111/j.1540-6261.1972.tb00971.x)：核对资本结构与权益系统风险之间的理论关系。
- [[WACC 与无杠杆 FCF]]：复用融资流与经营资产现金流必须分开的估值边界。

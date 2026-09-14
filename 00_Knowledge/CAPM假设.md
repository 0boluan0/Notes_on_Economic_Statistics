---
aliases:
  - CAPM 的定价结论依赖单期均值方差均衡与共同机会集等假设
  - CAPM assumptions
  - Sharpe-Lintner CAPM assumptions
student_os: knowledge-atom
atom_id: INV-CAPM-001
atom_set: capm-systematic-risk
atom_type: model-assumptions
status: source-checked
mastery_state: unassessed
requires:
  - "[[均值方差准则]]"
  - "[[风险资产与无风险分离]]"
part_of:
  - "[[CAPM、系统风险与资本成本.canvas]]"
leads_to:
  - "[[特有风险定价边界]]"
  - "[[证券市场线]]"
---

# CAPM 的定价结论依赖单期均值方差均衡与共同机会集等假设
<!-- bilingual-en:start -->
*CAPM pricing conclusions depend on a one-period mean–variance equilibrium and a common opportunity set, among other assumptions*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 标准 Sharpe–Lintner CAPM 不是从一条回归线直接“读出来”的经验规律。它从一个单期均衡模型出发：投资者按期望收益与方差选择组合，对资产收益有共同预期，面对无税费、可分割且可交易的资产，并能按同一无风险利率借贷。在这些条件下，所有人持有同一个切点风险组合；市场出清使它成为市场组合，随后才得到 beta—期望收益关系。
> <!-- bilingual-en:start -->
> The standard Sharpe–Lintner CAPM is not an empirical regularity read directly from a regression line. It starts from a one-period equilibrium in which investors choose by mean and variance, share beliefs about returns, trade divisible assets without taxes or transaction costs, and borrow or lend at one risk-free rate. These conditions produce a common tangency portfolio; market clearing makes it the market portfolio, and only then follows the beta–expected-return relation.
> <!-- bilingual-en:end -->

这些是假设与结论之间的一条充分推导路径，不是“少一条就完全不能研究资产定价”的清单。允许借贷利率不同、卖空受限、投资者预期异质或资产不能交易时，均衡组合和定价关系可能改变；应换用相应扩展模型，而不是继续沿用原公式后假装前提没有变化。
<!-- bilingual-en:start -->
These assumptions provide a sufficient route to the result, not a claim that asset pricing becomes impossible if one condition fails. Different borrowing and lending rates, short-sale constraints, heterogeneous beliefs, or nontradable assets can change both equilibrium holdings and the pricing relation. Such cases require an appropriate extension rather than an unchanged formula with hidden premise violations.
<!-- bilingual-en:end -->

> [!warning] 市场组合来自出清，不是先指定一个股票指数
> “大家都选同一个切点组合”与“所有风险资产的总供给由市场价值权重构成”合在一起，才使切点组合等于理论市场组合。任意选定的股票指数只是代理，不能由模型假设直接宣布为真实市场组合。

> [!question]- 自检
> 为什么“投资者都风险厌恶”本身还推不出 CAPM？
>
> **答案：** 还需要说明他们怎样评价风险、是否面对同一机会集、能否按同一无风险利率借贷，以及市场怎样出清。风险厌恶只描述偏好方向，不能单独确定均衡组合和证券市场线。

## 来源与核验

- [Sharpe (1964), “Capital Asset Prices”](https://doi.org/10.1111/j.1540-6261.1964.tb02865.x)：核对无风险资产、切点组合、市场均衡与风险价格的原始推导。
- [Jensen (1968), “The Performance of Mutual Funds in the Period 1945–1964”](https://doi.org/10.1111/j.1540-6261.1968.tb00815.x)：核对单期、同质预期、均值—方差选择、无税费和资产可分割等标准假设的集中表述。
- [[02_Economy/06_证券投资学/11_风险资产的定价.md#一、CAPM 的基本假设|课程 CAPM 假设部分]]：核对课程使用的假设口径与后续 CML/SML 顺序。

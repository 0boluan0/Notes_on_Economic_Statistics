---
aliases:
  - "风险模拟用真实世界测度 P 生成外层状态，并可在每个状态下用风险中性测度 Q 估值剩余现金流"
  - Outer-P and inner-Q roles in risk simulation
student_os: knowledge-atom
atom_id: RM-VAR-032
atom_set: var-es-backtesting
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[风险蒙特卡洛]]"
related:
  - "[[风险中性定价]]"
  - "[[预测时点与信息集]]"
  - "[[风险因子联合生成]]"
leads_to:
  - "[[风险模拟估值层]]"
part_of:
  - "[[风险模拟.canvas|风险模拟]]"
---

# 风险模拟用真实世界测度 P 生成外层状态，并可在每个状态下用风险中性测度 Q 估值剩余现金流
<!-- bilingual-en:start -->
*Risk simulation uses the real-world measure P to generate outer states and may use the risk-neutral measure Q to value remaining cash flows within each state*
<!-- bilingual-en:end -->

> [!summary] $P$ 回答会发生什么，$Q$ 回答在该状态怎样定价
> VaR 或 ES 的外层损失分布需要未来市场状态的真实世界概率，因此由 $P$ 生成。若持有期末仍有未到期衍生品，其条件公允价值可在该状态下用 $Q$ 计算。
>
> <!-- bilingual-en:start -->
> The outer VaR or ES distribution needs real-world probabilities for future market states, so it is generated under $P$. If derivatives remain alive at the horizon, their conditional fair values may be computed under $Q$ in each state.
> <!-- bilingual-en:end -->

$$
x_{t_0:t_0+H}^{(i)}\sim P_{\widehat\theta},
$$

$$
V_{t_0+H}^{(i)}
=E^{Q^{(i)}}\!\left[
\text{discounted remaining cash flows}
\mid\mathcal F_{t_0+H}^{(i)}
\right].
$$

直接用风险中性路径的频率当作真实世界 VaR 概率，会把定价测度与风险预测测度混在一起。若产品有闭式或可靠数值定价器，条件估值不必另做内层 Monte Carlo。

<!-- bilingual-en:start -->
Treating risk-neutral path frequencies as real-world VaR probabilities confuses pricing with forecasting. If a closed-form or reliable numerical pricer is available, conditional valuation need not use another inner Monte Carlo.
<!-- bilingual-en:end -->

> [!question]- 自检
> 期权按 $Q$ 定价，是否意味着风险 VaR 的外层情景也应全部从 $Q$ 生成？
>
> **答案：** 不意味着。外层频率要描述真实世界状态，条件估值才使用与该状态一致的 $Q$。
>
> <!-- bilingual-en:start -->
> **Self-check:** Because an option is priced under $Q$, must all outer scenarios for risk VaR also be generated under $Q$?
>
> **Answer:** No. The outer distribution represents real-world state frequencies; conditional valuation uses a $Q$ measure consistent with each state.
> <!-- bilingual-en:end -->

## 边界

- $P$ 与 $Q$ 的模型都可能错设。
- 风险期限、现金流边界和折现口径必须一致。
- 流动性或不可交易风险可能需要超出标准无套利定价的调整。

<!-- bilingual-en:start -->
- Both $P$ and $Q$ models can be misspecified.
- Horizon, cash-flow boundary, and discounting convention must align.
- Liquidity or non-tradable risks may require adjustments beyond standard arbitrage pricing.
<!-- bilingual-en:end -->

## 来源与核验

- Glasserman, [*Monte Carlo Methods in Financial Engineering*](https://doi.org/10.1007/978-0-387-21617-1)：核对真实世界动态与风险中性定价的不同角色。
- [Federal Reserve, FEDS 2008-21](https://www.federalreserve.gov/pubs/feds/2008/200821/index.html)：核对外层风险情景与内层条件估值的嵌套结构。

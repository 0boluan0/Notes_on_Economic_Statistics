---
aliases:
  - "FRTB 回测分别用 APL 与 HPL 对照一日 VaR，RTPL 只用于损益归因"
  - FRTB backtesting P&L
student_os: knowledge-atom
atom_id: RM-VAR-022
atom_set: var-es-backtesting
atom_type: regulatory-convention
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR回测损益口径]]"
  - "[[FRTB 市场风险]]"
related:
  - "[[FRTB 交易台资格]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# FRTB 回测分别用 APL 与 HPL 对照一日 VaR，RTPL 只用于损益归因
<!-- bilingual-en:start -->
*FRTB backtesting separately compares APL and HPL with one-day VaR, while RTPL is used only for P&L attribution*
<!-- bilingual-en:end -->

> [!summary] 三种损益有不同监管角色
> FRTB 的 actual P&L（APL）与 hypothetical P&L（HPL）各自形成 VaR 例外记录；risk-theoretical P&L（RTPL）与 HPL 比较，用于损益归因检验。RTPL 不是第三条 VaR 回测序列。
>
> <!-- bilingual-en:start -->
> Under FRTB, actual P&L (APL) and hypothetical P&L (HPL) each produce a VaR exception record. Risk-theoretical P&L (RTPL) is compared with HPL in the P&L attribution test; it is not a third VaR backtesting series.
> <!-- bilingual-en:end -->

HPL 固定上一交易日收盘头寸，用本日收盘市场数据重估；它排除日内交易、新增或修改交易、费用与佣金。APL 来自实际日常损益流程，包含日内交易和时间效应，但按规则同样处理费用、佣金与特定估值调整。两者不能合并后再数一次例外。

<!-- bilingual-en:start -->
HPL freezes prior-close positions and revalues them with current-close market data, excluding intraday and new or modified trades, fees, and commissions. APL comes from the daily actual-P&L process and includes intraday trading and time effects, subject to the framework's treatment of fees, commissions, and valuation adjustments. The two series are not merged before counting hits.
<!-- bilingual-en:end -->

当前 Basel 回测把一日 VaR 分别与 APL、HPL 比较；具体资格阈值与两组例外怎样进入监管决定，见 [[FRTB 交易台资格]]。

<!-- bilingual-en:start -->
The current Basel backtest compares one-day VaR separately with APL and HPL. See [[FRTB 交易台资格|FRTB trading-desk eligibility]] for qualification thresholds and supervisory use.
<!-- bilingual-en:end -->

> [!question]- 自检
> RTPL 超过 VaR 是否自动增加第三条 VaR 例外记录？
>
> **答案：** 不会。RTPL 用于与 HPL 做损益归因；VaR 回测记录来自 APL 与 HPL。
>
> <!-- bilingual-en:start -->
> **Self-check:** If RTPL exceeds VaR, does that automatically create a third VaR exception record?
>
> **Answer:** No. RTPL is compared with HPL for P&L attribution; VaR backtesting records come from APL and HPL.
> <!-- bilingual-en:end -->

## 边界

- 损益对象的定义与监管分区阈值是不同问题；后者须按适用规则另行核对。
- APL 与 HPL 单日不同不能唯一归因于尾部分布模型。
- 监管规则具有版本和法域实施日期，使用时应核对现行文本。

<!-- bilingual-en:start -->
- P&L-object definitions and regulatory zone thresholds are separate questions; the latter must be checked against the applicable rules.
- A daily APL–HPL difference does not uniquely diagnose the tail model.
- Regulatory rules require version- and jurisdiction-specific checking.
<!-- bilingual-en:end -->

## 来源与核验

- [Basel Framework, MAR10](https://www.bis.org/basel_framework/chapter/MAR/10.htm)：核对 APL、HPL 与 RTPL 的定义。
- [Basel Framework, MAR32](https://www.bis.org/basel_framework/chapter/MAR/32.htm)：核对一日 VaR 分别与 APL、HPL 回测，以及 RTPL–HPL 的损益归因角色。
- [Basel Framework, MAR99](https://www.bis.org/committees/bcbs/basel-framework/standard/mar/99/inforce/2023-01-01/published/2020-03-27)：核对两组回测例外的监管使用。

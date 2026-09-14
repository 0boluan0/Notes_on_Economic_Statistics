---
aliases:
  - "VaR 回测只有让预测与实现损益的头寸、估值、期限和信息集一致才可解释"
  - VaR backtesting P&L alignment
  - VaR 例外与损益口径
student_os: knowledge-atom
atom_id: RM-VAR-007
atom_set: var-es-backtesting
atom_type: measurement-convention
status: source-checked
mastery_state: unassessed
requires:
  - "[[风险度量口径]]"
  - "[[VaR定义]]"
  - "[[预测时点与信息集]]"
related:
  - "[[VaR时间缩放]]"
  - "[[风险窗口选择冻结]]"
  - "[[风险模拟.canvas|风险模拟]]"
  - "[[FRTB回测损益]]"
leads_to:
  - "[[Kupiec无条件覆盖]]"
  - "[[Christoffersen独立性]]"
  - "[[Christoffersen条件覆盖]]"
  - "[[VaR-ES联合识别]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# VaR 回测只有让预测与实现损益的头寸、估值、期限和信息集一致才可解释
<!-- bilingual-en:start -->
*A VaR backtest is interpretable only when forecast and realised P&L align in positions, valuation, horizon, and information*
<!-- bilingual-en:end -->

> [!summary] 先定义比较对象，再数例外
> 回测把事前 VaR 与随后实现的同一对象损失比较。预测和实现必须使用一致的头寸范围、估值规则、持有期、币种与信息截点；否则例外可能只说明两边测量了不同对象。
>
> <!-- bilingual-en:start -->
> A backtest compares ex-ante VaR with the subsequently realised loss on the same object. Position scope, valuation rules, horizon, currency, and information cut-off must align; otherwise an exception may only reveal a measurement mismatch.
> <!-- bilingual-en:end -->

采用损失为正，令 $\mathcal F_{t-1}$ 是预测时可得信息：

<!-- bilingual-en:start -->
Using positive loss and information $\mathcal F_{t-1}$ available at forecast time:
<!-- bilingual-en:end -->

$$
q_t=\operatorname{VaR}_{\alpha}(L_t\mid\mathcal F_{t-1}),
\qquad
I_t=\mathbf 1\{L_t>q_t\}.
$$

$q_t$ 必须在 $L_t$ 实现前冻结。不能用当天收盘后才知道的数据重估阈值，再把结果称为真正的样本外回测。还应逐项对齐头寸、市场数据截点、估值模型、现金流、费用、时间效应和估值调整。

<!-- bilingual-en:start -->
$q_t$ must be frozen before $L_t$ is realised. Re-estimating the threshold with end-of-day information is not a genuine out-of-sample backtest. Positions, market-data cut-offs, valuation models, cash flows, fees, time effects, and valuation adjustments must also be aligned.
<!-- bilingual-en:end -->

严格不等号也是口径的一部分。若分位点有概率质量，$L_t>q_t$ 与 $L_t\ge q_t$ 给出不同例外率；规则必须预先固定并与 [[VaR定义]] 一致。多日滚动损益若重叠，还会共享收益并机械制造相邻依赖。

<!-- bilingual-en:start -->
The inequality is also part of the convention. With probability mass at the quantile, $L_t>q_t$ and $L_t\ge q_t$ produce different exception rates, so the rule must be fixed in advance and match the [[VaR定义|VaR definition]]. Overlapping multi-day losses mechanically share returns and induce serial dependence.
<!-- bilingual-en:end -->

FRTB 对 APL、HPL 与 RTPL 的特定定义不属于一般对齐原则，见 [[FRTB回测损益]]。

<!-- bilingual-en:start -->
The FRTB-specific definitions of APL, HPL, and RTPL are separate from this general alignment rule; see [[FRTB回测损益|FRTB backtesting P&L]].
<!-- bilingual-en:end -->

> [!question]- 自检
> 能否用今天收盘后重估的 VaR 与今天损益比较，并称为今天的事前回测？
>
> **答案：** 不能。预测必须在损益实现前冻结，否则使用了结果期信息。
>
> <!-- bilingual-en:start -->
> **Self-check:** Can VaR re-estimated after today's close be compared with today's P&L and called today's ex-ante backtest?
>
> **Answer:** No. A forecast re-estimated after today's close uses information from the outcome period and is not today's ex-ante backtest.
> <!-- bilingual-en:end -->

## 边界

- 对齐口径只是可解释性的前提，不保证覆盖率或独立性正确。
- 例外指标不记录超越幅度，也不能单独定位数据、动态或估值中的根因。
- 不能为了减少例外而在看见结果后改变损益清洗规则。

<!-- bilingual-en:start -->
- Alignment is necessary for interpretation; it does not guarantee correct coverage or independence.
- The hit indicator omits exceedance severity and cannot by itself locate the source of failure.
- P&L-cleaning rules must not be changed after observing outcomes merely to reduce exceptions.
<!-- bilingual-en:end -->

## 来源与核验

- Christoffersen, [*Evaluating Interval Forecasts*](https://doi.org/10.2307/2527341)：核对区间预测必须是事前条件预测，以及覆盖与时间独立性是不同要求。
- [Basel Framework, MAR32](https://www.bis.org/basel_framework/chapter/MAR/32.htm)：核对一日 VaR 必须与对应日损益比较的回测结构。

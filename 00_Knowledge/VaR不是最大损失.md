---
aliases:
  - "VaR 是损失分位点而不是最大可能损失"
  - VaR is not a maximum loss
  - VaR 不给出损失上限
student_os: knowledge-atom
atom_id: RM-VAR-038
atom_set: var-es-backtesting
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
related:
  - "[[ES定义]]"
  - "[[相同EL不同尾部]]"
  - "[[压力测试方法]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# VaR 是损失分位点而不是最大可能损失
<!-- bilingual-en:start -->
*VaR is a loss quantile, not the maximum possible loss*
<!-- bilingual-en:end -->

> [!summary] 分位点之外仍有尾部
> VaR 只定位损失分布中的一个概率位置。落在该位置以上的损失仍然可能发生，而且 VaR 本身既不报告其中最大的损失，也不报告它们的平均严重程度。
>
> <!-- bilingual-en:start -->
> VaR locates one probability position in a loss distribution. Losses beyond that position can still occur, and VaR by itself reports neither their maximum nor their average severity.
> <!-- bilingual-en:end -->

考虑

$$
P(L=0)=0.98,
\qquad
P(L=100)=0.01,
\qquad
P(L=1000)=0.01.
$$

在 $\alpha=0.99$ 时，

$$
\operatorname{VaR}_{0.99}(L)=100,
$$

但损失 1000 仍以 1% 概率发生。VaR=100 因而不能被解释为“最多损失 100”。若问题是最坏尾部的平均严重程度，应转向 [[ES定义]]；若问题是模型之外的严重但合理情景，则还需要压力测试。

<!-- bilingual-en:start -->
Yet a loss of 1000 still occurs with probability 1%. VaR equal to 100 therefore cannot mean that loss is capped at 100. Use [[ES定义|Expected Shortfall]] for average severity in a fixed worst tail, and stress testing for severe plausible scenarios outside the fitted quantile summary.
<!-- bilingual-en:end -->

> [!question]- 自检
> 99% VaR 为 100，是否排除了损失 1000？
>
> **答案：** 没有。VaR 只给出 99% 分位点；更大的损失仍可位于剩余尾部中。
>
> <!-- bilingual-en:start -->
> **Self-check:** Does 99% VaR equal to 100 rule out a loss of 1000?
>
> **Answer:** No. VaR gives the 99th percentile; larger losses can remain in the residual tail.
> <!-- bilingual-en:end -->

## 来源与核验

- [Rockafellar & Uryasev, *Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)：核对分位型 VaR 与尾部损失严重程度的区别。

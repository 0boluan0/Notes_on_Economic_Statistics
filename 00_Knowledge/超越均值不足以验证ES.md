---
aliases:
  - "VaR 例外日的平均损失不足以构成一般动态 ES 验证"
  - Mean loss on VaR exception days is insufficient to validate ES
student_os: knowledge-atom
atom_id: RM-VAR-025
atom_set: var-es-backtesting
atom_type: validation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[ES定义]]"
  - "[[VaR-ES联合识别]]"
related:
  - "[[VaR回测损益口径]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# VaR 例外日的平均损失不足以构成一般动态 ES 验证
<!-- bilingual-en:start -->
*The mean loss on VaR exception days is not a general validation method for dynamic ES forecasts*
<!-- bilingual-en:end -->

> [!summary] 入选日期由 VaR 决定，样本又极少
> 只平均满足 $L_t>q_t$ 的日期，既没有检查 $q_t$ 是否正确，也没有给出识别每期动态 $e_t$ 的一般条件矩；零例外时还无法计算。
>
> <!-- bilingual-en:start -->
> Averaging only dates with $L_t>q_t$ neither checks whether $q_t$ is correct nor supplies a general moment identifying each dynamic ES forecast $e_t$. With zero exceptions, it is undefined.
> <!-- bilingual-en:end -->

朴素统计量为

$$
\overline L_{\mathrm{exc}}
=
\frac{\sum_{t=1}^n\mathbf 1\{L_t>q_t\}L_t}
{\sum_{t=1}^n\mathbf 1\{L_t>q_t\}}.
$$

在固定阈值、连续 IID 分布等特殊条件下，它可估一个尾部条件均值；动态回测中，阈值 $q_t$ 随信息变化，入选日期本身依赖 VaR 预测。例外很少时统计量不稳定，且它不能替代 [[VaR-ES联合识别]] 的成对校准矩。

<!-- bilingual-en:start -->
With a fixed threshold under special continuous iid conditions, this statistic can estimate a tail conditional mean. In dynamic backtesting, $q_t$ changes with information and determines which dates enter. Sparse exceptions make the statistic unstable, and it cannot replace the paired calibration moments in [[VaR-ES联合识别|joint VaR–ES identification]].
<!-- bilingual-en:end -->

> [!question]- 自检
> 样本内没有 VaR 例外，能否据此说 ES 已通过“例外日均值检验”？
>
> **答案：** 不能。分母为零，统计量没有定义；零例外也可能反映样本短、VaR 过高或阈值质量。
>
> <!-- bilingual-en:start -->
> **Self-check:** If a sample contains no VaR exceptions, can ES be said to have passed an “exception-day mean test”?
>
> **Answer:** No. The statistic is undefined because its denominator is zero. Zero exceptions can also reflect a short sample, an excessive VaR forecast, or a poor threshold.
> <!-- bilingual-en:end -->

## 边界

- 固定阈值 IID 尾均估计不是一般动态 ES 回测。
- 外部只提交 ES 的回归检验仍可在内部使用辅助分位数结构。
- ES 可以回测，但一个朴素的例外日均值不足以完成验证。

<!-- bilingual-en:start -->
- A fixed-threshold iid tail mean is not a general dynamic ES backtest.
- Regression tests that accept only ES inputs can still use an auxiliary quantile structure internally.
- ES can be backtested, but a naive exception-day mean is not sufficient validation.
<!-- bilingual-en:end -->

## 来源与核验

- Nolde & Ziegel, [*Elicitability and Backtesting*](https://arxiv.org/abs/1608.05498)：核对 ES 单变量识别边界与传统回测。
- Bayer & Dimitriadis, [*Regression-Based Expected Shortfall Backtesting*](https://doi.org/10.1093/jjfinec/nbaa013)：核对只外部输入 ES 的检验仍依赖联合回归结构。

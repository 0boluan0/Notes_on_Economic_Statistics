---
aliases:
  - "ETS 创新残差诊断不能排除未来结构突变"
  - ETS innovation residual diagnostics
  - ETS structural break boundary
  - ETS 残差与断点
student_os: knowledge-atom
atom_id: TS-ETS-008
atom_set: exponential-smoothing-ets
atom_type: diagnostic-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[ETS创新残差]]"
  - "[[ARMA残差诊断]]"
related:
  - "[[Ljung-Box检验]]"
  - "[[滚动起点评估]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# ETS 创新残差诊断不能排除未来结构突变
<!-- bilingual-en:start -->
*ETS innovation diagnostics cannot rule out future structural breaks*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> ETS 拟合后应检查一步 innovation 是否近似零均值、无剩余自相关且误差尺度/分布假设合理；即使这些诊断通过，也只说明历史拟合期未发现相应失配，不能证明未来 level、trend 或 seasonal pattern 不会突变。
> <!-- bilingual-en:start -->
> ETS diagnostics assess one-step innovations for bias, remaining dependence, and error-scale assumptions within the fitted history. Passing them does not establish that future level, trend, or seasonality will remain structurally stable.
> <!-- bilingual-en:end -->

诊断前先按 [[ETS创新残差|ETS 的误差形式]]确定检查对象：additive error 检查原尺度一步误差，multiplicative error 检查相对创新。把两者混用，会把随水平扩张的正常尺度变化误判为异常，或漏掉相对误差中的结构。

诊断至少包括 innovation time plot、均值与持续偏差、ACF 与有预先说明滞后的 [[Ljung-Box检验]]，并检查绝对值/平方 innovation、异常值和分布尾部。未拒绝白噪声只说明在该样本与检验力下未发现所检线性相关；它不证明独立、正态、同方差或模型唯一正确，这些通用边界直接复用 [[ARMA残差诊断]]。

level shift、季节相位变化、移动节假日、新产品或制度变化发生后，固定参数 ETS 只能在观测到新数据后逐步更新状态，不能提前知道断点。较大 $\alpha$ 可能追得更快，却也更敏感于一次噪声；自动扩大参数不能替代断点解释。持续同号 innovation、CUSUM/监控信号与 rolling-origin 在不同历史制度下的劣化，应触发重新设定、干预变量或缩短训练窗口，而不是只继续递推旧模型。
<!-- bilingual-en:start -->
Inspect the innovation defined by the fitted error form. Residual checks are retrospective: state updating reacts after a break but cannot anticipate it, so monitoring, contextual intervention, and chronological evaluation remain necessary.
<!-- bilingual-en:end -->

> [!question]- 自检
> ETS 的训练期 Ljung–Box 未拒绝，但促销制度下月永久改变。能否据此继续宣称未来误差无偏？
>
> **答案：** 不能。检验只覆盖旧制度中的所检相关；制度变化会改变条件位置或季节模式，必须重新评估规格与信息集。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §5.4](https://otexts.com/fpp3/diagnostics.html)：核对创新残差的零均值、无剩余相关要求，以及通过这些检查仍不能证明模型不可改进。
- [Gardner (1985), “Exponential Smoothing: The State of the Art”](https://doi.org/10.1002/for.3980040103)：核对 forecast-error monitoring、bias/CUSUM 与结构变化下模型维护的边界。
- [[ETS创新残差]]、[[ARMA残差诊断]] 与 [[滚动起点评估]]：分别复用残差定义、检验解释与未来评估的通用结论。

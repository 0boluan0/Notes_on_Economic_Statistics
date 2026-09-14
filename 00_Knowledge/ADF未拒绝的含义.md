---
aliases:
  - "ADF 未拒绝单位根只表示样本证据不足"
  - Failure to reject unit root
  - 单位根未拒绝
student_os: knowledge-atom
atom_id: TS-UR-012
atom_set: trends-unit-roots-differencing
atom_type: interpretation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[ADF检验]]"
related:
  - "[[近单位根]]"
  - "[[单位根证据组合]]"
part_of:
  - "[[趋势、单位根与差分.canvas]]"
---

# ADF 未拒绝单位根只表示样本证据不足
<!-- bilingual-en:start -->
*Failure to reject an ADF unit-root null means insufficient sample evidence, not proof of a unit root*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 当 ADF 统计量没有越过与规格相匹配的临界值时，规范表述是“未拒绝单位根原假设”。这不等于接受或证明序列有单位根。

短样本和近单位根会使 ADF 功效偏低；错误的确定性项、滞后阶选择不当、结构突变和异常值则可能降低功效，也可能造成检验大小失真。反过来，拒绝 $H_0$ 也只是在给定规格和显著性水平下反对该单位根原假设；它不证明残差白噪声、不排除季节单位根，也不证明所选模型适合预测。

因此结论应连同样本区间、确定性项、滞后选择、统计量与临界值或 $p$ 值、残差诊断和替代规格一起报告，而不是输出二元标签。

> [!question]- 自检
> ADF 的 $p=0.18$ 能否写成“序列一定是 $I(1)$”？
>
> **答案：** 不能。它只表示在当前样本和规格下没有足够证据拒绝单位根，还要检查功效、结构突变、季节根和差分后的性质。

## 来源与核验

- [Dickey & Fuller (1979)](https://doi.org/10.1080/01621459.1979.10482531)：核对检验原假设与拒绝域。
- [Phillips (1987), *Towards a Unified Asymptotic Theory for Autoregression*](https://doi.org/10.1093/biomet/74.3.535)：核对近单位根下有限样本区分困难的理论背景。

---
aliases:
  - "Copula 拟合必须分开验证边际、依赖与尾部，样本内似然不能证明联合模型正确"
  - "Copula validation must audit marginals, dependence, and tails separately"
  - "Copula模型验证"
student_os: knowledge-atom
atom_id: RM-DEP-004
atom_set: dependence-and-copulas
atom_type: validation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Copula拟合]]"
  - "[[尾部依赖]]"
related:
  - "[[Gaussian与t Copula尾部]]"
  - "[[相关度量比较]]"
  - "[[Rosenblatt变换]]"
  - "[[风险模型验证边界]]"
  - "[[风险窗口权衡]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# Copula 拟合必须分开验证边际、依赖与尾部，样本内似然不能证明联合模型正确
<!-- bilingual-en:start -->
*Copula validation must audit margins, dependence, and tails separately; in-sample likelihood does not prove that the joint model is correct*
<!-- bilingual-en:end -->

> [!summary] 验证要沿模型结构逐层追问
> 一个已拟合的 copula 模型同时包含边际、依赖族、参数和时间条件。样本内似然只说明某个候选模型在给定候选集与估计流程中的相对拟合；它不能单独证明边际正确、尾部正确、时间稳定或样本外风险预测有效。
> <!-- bilingual-en:start -->
> A fitted copula model combines margins, a dependence family, parameters, and conditioning assumptions. In-sample likelihood is only relative evidence within the chosen candidate set and estimation procedure.
> <!-- bilingual-en:end -->

## 四层验证

1. **边际层。** 检查概率积分变换后的残差是否接近 Uniform$(0,1)$，并诊断剩余的均值、波动或时间结构。边际错设会污染后续依赖判断。
2. **整体依赖层。** 比较经验 copula 与拟合 copula，核对 Spearman、Kendall 等摘要，并对含已估参数的拟合优度统计量做适当的 bootstrap 校准。
3. **目标尾部层。** 单独比较上下尾共同超越频率、目标分位数和压力区间。整体拟合好不保证目标尾部好。
4. **稳定性与用途层。** 检查窗口变化、参数不确定性和样本外风险预测；验证标准必须对准模型将承担的决策。

增加 Monte Carlo 路径只能减少“给定模型下”的抽样误差，不能修复边际或 copula 族选错，见 [[风险模型验证边界]]。
<!-- bilingual-en:start -->
Validation should separately diagnose margins, overall dependence, target tails, and stability for the intended use. More simulation paths reduce conditional Monte Carlo error but do not repair model misspecification.
<!-- bilingual-en:end -->

## 完整联合模型的诊断

[[Rosenblatt变换]]可以把一个给定顺序下的连续联合模型转成独立均匀变量，再检查是否仍有边际或依赖残差。它的顺序、连续性条件与检验力边界留在方法卡；在这里必须保留的结论只有一个：任何单项检验“未拒绝”，都不能证明联合模型正确。

即使 Gaussian copula 复现了样本 Kendall $\tau$，若真实数据的下尾共同超越系统性高于模型模拟，它仍在目标风险区域失配。

> [!question]- 自检
> 候选 t copula 的样本内对数似然最高，是否足以批准它用于尾部风险？
>
> **答案：** 不足。至少还要核对边际、bootstrap 校准的整体拟合优度、目标尾部、时间稳定性和样本外表现。

## 来源与核验

- Murray Rosenblatt (1952), [“Remarks on a Multivariate Transformation”](https://doi.org/10.1214/aoms/1177729394)：核对顺序条件分布变换。
- Christian Genest, Bruno Rémillard and David Beaudoin (2009), [“Goodness-of-fit tests for copulas: A review and a power study”](https://doi.org/10.1016/j.insmatheco.2007.10.005)：核对 bootstrap 校准、检验力与 Rosenblatt 顺序边界。
- 作者逐项复核日：2026-09-01；验证层次与适用边界已按上述来源核对。

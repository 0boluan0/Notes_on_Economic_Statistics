---
aliases:
  - "VIF 的固定阈值只是筛查信号，不能脱离估计目标、样本规模和所需精度成为删变量规则"
  - VIF threshold boundary
  - VIF cutoffs are screening conventions
student_os: knowledge-atom
atom_id: ECON-COL-005
atom_set: multicollinearity-design-diagnostics
atom_type: diagnostic-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[方差膨胀因子]]"
  - "[[模型选择目标]]"
related:
  - "[[显著性解释边界]]"
  - "[[共线性补救原则]]"
part_of:
  - "[[多重共线性.canvas|多重共线性与设计矩阵诊断]]"
---

# VIF 的固定阈值只是筛查信号，不能脱离估计目标、样本规模和所需精度成为删变量规则
<!-- bilingual-en:start -->
*A fixed VIF cutoff is only a screening signal, not a variable-deletion rule independent of the estimand, sample size, and required precision*
<!-- bilingual-en:end -->

> [!summary] 判断边界
> 常见 VIF 阈值把连续的方差膨胀压成“通过／不通过”标签，适合提醒你继续调查，却不能决定模型是否可用。真正的问题是：被放大的是否为目标系数，剩余区间是否仍足以回答研究问题，以及相关变量在机制和识别中承担什么角色。
> <!-- bilingual-en:start -->
> Common VIF cutoffs turn continuous variance inflation into a pass/fail label. They are useful prompts for further investigation, not verdicts on model usability. The relevant questions are whether the affected coefficient is the target, whether its remaining interval is adequate for the decision, and what role the correlated variables play in the mechanism and identification strategy.
> <!-- bilingual-en:end -->

同一个 VIF 在不同任务中意义不同。一个处理效应的 VIF=3，若样本很小、最小可接受精度很高，可能已经让研究没有辨别力；一个纯预测模型的 VIF=20，若样本外预测稳定，未必妨碍预测本身，但逐个系数仍不宜作结构解释。
<!-- bilingual-en:start -->
The same VIF can matter differently across tasks. A treatment-effect VIF of three may be fatal in a small sample with a demanding precision target. A prediction-only model with a VIF of twenty may still predict well out of sample, although its individual coefficients remain poor candidates for structural interpretation.
<!-- bilingual-en:end -->

## 高 VIF 之后应问什么
<!-- bilingual-en:start -->
*What to ask after a high VIF*
<!-- bilingual-en:end -->

1. 辅助回归是否使用了与主模型完全相同的样本和设计列？
2. 这是编码恒等式、构造项相关，还是样本支持本来就狭窄？
3. 目标是单个系数、联合效应、预测，还是某个可识别线性组合？
4. 删除变量会不会改变 estimand 或破坏外生性论证？
5. 区间、联合检验和合理规格下的稳定性是否已经足够？
<!-- bilingual-en:start -->
1. Was the auxiliary regression fitted on exactly the same sample and design columns as the main model?
2. Is the signal caused by a coding identity, constructed terms, or genuinely narrow sample support?
3. Is the target an individual coefficient, a joint effect, prediction, or an identifiable linear combination?
4. Would deleting a variable change the estimand or damage the exogeneity argument?
5. Are intervals, joint tests, and stability across defensible specifications already adequate?
<!-- bilingual-en:end -->

为了把 VIF 降到阈值以下而反复删变量，再把最终模型的普通 $p$ 值当作预先给定的推断，还会忽略选择过程带来的不确定性。诊断应触发[[共线性补救原则|有理由的修复]]，而不是触发自动逐步筛选。
<!-- bilingual-en:start -->
Repeatedly deleting variables until every VIF falls below a cutoff and then treating the final ordinary p-values as prespecified inference also ignores model-selection uncertainty. A diagnostic should trigger a reasoned repair, not automatic stepwise screening.
<!-- bilingual-en:end -->

> [!question]- 自检
> 必要混杂控制变量的 VIF=12，是否足以授权删除它？
>
> **答案：** 不足。应先评估目标、区间、重叠与设计；删除它可能用较小方差换来更严重的遗漏变量偏差。

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Penn State STAT 501, Lesson 12.4](https://online.stat.psu.edu/stat501/Lesson12)：核对 VIF 阈值被明确称为经验法则，以及 VIF 用于进一步调查的角色。
- [[模型选择目标]]、[[遗漏变量偏差]]与[[选择后推断]]：核对预测／解释／因果目标差异、删控制项的偏差风险与选择后普通推断的边界。
- [[02_Economy/01_Econometrics/06_多重共线性.md|本地课程：多重共线性补救]]：核对“保留理论必要变量并报告不精确”是合法选项。
<!-- bilingual-en:start -->
- Penn State labels common cutoffs as rules of thumb and uses them as prompts for investigation. The linked source-checked atoms establish goal dependence, omitted-variable risk, and post-selection inference boundaries. The local course explicitly allows retaining theoretically required variables while reporting imprecision.
<!-- bilingual-en:end -->

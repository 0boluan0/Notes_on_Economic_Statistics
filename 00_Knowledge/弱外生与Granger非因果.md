---
aliases:
  - "弱外生性与 Granger 非因果检验的是不同限制"
  - Weak exogeneity versus Granger noncausality
  - 弱外生与 Granger 非因果
student_os: knowledge-atom
atom_id: TS-CI-010
atom_set: cointegration-error-correction
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[弱外生性与调整载荷]]"
related:
  - "[[协整解释边界]]"
  - "[[Granger因果]]"
  - "[[Granger因果边界]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# 弱外生性与 Granger 非因果检验的是不同限制
<!-- bilingual-en:start -->
*Weak exogeneity and Granger noncausality impose different restrictions*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 弱外生性关心能否为某组参数做条件推断；Granger 非因果关心加入一个变量的过去后，是否改善另一个变量的条件预测。二者目标不同，限制也不同。

在协整 VECM 中，预测通道既可能来自滞后差分，也可能来自误差修正项。把变量分成 $y$ 与 $x$ 后，要说 $x$ 不 Granger 导致 $y$，通常需要联合限制：$y$ 方程中所有 $\Delta x$ 滞后的系数为零，且长期水平通道的总系数块 $\alpha_y\beta_x'$ 也为零。多条协整关系时，不能只逐个看某个 $\alpha$ 或 $\beta$ 元素，因为各关系的作用可在矩阵乘积中组合或抵消。$\alpha_y=0$ 会删除 $y$ 方程的全部误差修正通道，却仍不排除 $\Delta x_{t-j}$ 进入 $\Delta y_t$。

即使拒绝 Granger 非因果，结论也只是给定信息集、滞后阶数和模型稳定性下的增量预测关系。同期共同冲击、遗漏变量和政策预期仍可产生这种关系，不能自动升级为结构因果。

> [!question]- 自检
> $y$ 方程的整行误差修正载荷为零，是否足以证明 $x$ 不 Granger 导致 $y$？
>
> **答案：** 不足。还要联合检验 $x$ 的相关滞后差分系数；并且结论只针对给定信息集的预测关系。

## 来源与核验

- [Engle, Hendry & Richard (1983), *Exogeneity*](https://doi.org/10.2307/1911990)：核对弱外生性是相对于推断目标定义的概念。
- [Granger (1988), *Some Recent Development in a Concept of Causality*](https://doi.org/10.1016/0304-4076(88)90045-0)：核对协整系统中的预测因果边界。

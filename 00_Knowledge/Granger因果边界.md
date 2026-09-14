---
aliases:
  - "Granger 关系依赖信息集与规格且不等于结构因果"
  - Granger causality interpretation boundary
  - 预测因果解释边界
student_os: knowledge-atom
atom_id: TS-VAR-020
atom_set: vector-autoregression
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Granger因果]]"
related:
  - "[[VAR Granger检验]]"
  - "[[弱外生与Granger非因果]]"
  - "[[简约型VAR创新]]"
  - "[[VAR规格选择]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# Granger 关系依赖信息集与规格且不等于结构因果
<!-- bilingual-en:start -->
*Granger relations depend on the information set and specification and are not structural causality*
<!-- bilingual-en:end -->

> [!summary] 解释边界
> Granger 因果比较的是两个信息集对未来条件分布或预测的贡献；结论随控制变量、滞后、频率、变换和模型类而变，并不等于对 $x$ 作外生干预会改变 $y$ 的结构因果效应。

<!-- bilingual-en:start -->
> [!summary] Interpretation boundary
> Granger causality compares the contribution of two information sets to a future conditional distribution or forecast. Its conclusion changes with controls, lags, frequency, transformations, and model class, and it is not the structural effect of an external intervention on $x$.
<!-- bilingual-en:end -->

[[Granger因果]] 的一般定义是明确的，但一个实际检验只在某个有限模型里近似这个定义。例如，线性 VAR 的系数检验只检查所选滞后对条件均值的线性增量预测力；它不自动排除非线性、波动或分布尾部中的关系。

<!-- bilingual-en:start -->
The general definition is clear, but an empirical test approximates it within a finite model. A linear VAR coefficient test examines only linear incremental predictive content for the conditional mean at the selected lags; it does not automatically rule out nonlinear, volatility, or tail-distribution relations.
<!-- bilingual-en:end -->

遗漏共同原因可能让 $x$ 的过去代理未纳入的信息；时间聚合可把同期作用变成表观滞后；非线性、结构突变或错误差分也会改变结果。加入新的控制变量后，原有 Granger 关系可以出现、消失或反向，这不是逻辑矛盾，而是信息集已经改变。

<!-- bilingual-en:start -->
An omitted common driver can make the history of $x$ proxy for missing information; temporal aggregation can turn contemporaneous effects into apparent lags; nonlinearities, structural breaks, and incorrect differencing can all change the result. A relation may appear, disappear, or reverse after controls are added because the comparison information set has changed.
<!-- bilingual-en:end -->

结构因果问的是一个明确干预或已识别冲击对反事实路径的影响，需要外生变异、结构限制或其他识别设计。Granger 关系只报告预测先后与增量信息。同期相关也不由单向滞后检验完整描述。报告时应写“在给定变量、滞后和样本下拒绝／未拒绝 Granger 非因果”，而不是写“证明 $x$ 导致／不导致 $y$”。

<!-- bilingual-en:start -->
Structural causality concerns the counterfactual path under an explicit intervention or identified shock and requires exogenous variation or structural restrictions. Granger results report predictive precedence and incremental information. Write that the specified noncausality null was rejected or not rejected in a stated sample and information set; do not rewrite this as proof that $x$ structurally causes or does not cause $y$.
<!-- bilingual-en:end -->

> [!question]- 自检
> 加入第三个变量后，原先从 $x$ 到 $y$ 的 Granger 关系消失，最直接说明什么？
>
> **答案：** 说明预测关系依赖信息集；第三个变量吸收了原先由 $x$ 历史提供的增量信息。它本身仍不能确定结构因果方向。

## 来源与核验

- [Granger (1969)](https://doi.org/10.2307/1912791)：核对以信息集和可预测性定义的原始概念。
- [Granger (1988), *Some Recent Development in a Concept of Causality*](https://doi.org/10.1016/0304-4076(88)90045-0)：核对协整、反馈与预测因果的边界。
- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)：核对多变量信息集中的因果分析。

---
aliases:
  - "线性 VAR 中 Granger 非因果对应目标方程滞后系数的联合零限制"
  - Granger Causality Test
  - Linear VAR Granger noncausality
student_os: knowledge-atom
atom_id: TS-VAR-019
atom_set: vector-autoregression
atom_type: condition
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR(p)模型]]"
  - "[[Granger因果]]"
related:
  - "[[Granger因果边界]]"
  - "[[弱外生与Granger非因果]]"
  - "[[ARMA残差诊断]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# 线性 VAR 中 Granger 非因果对应目标方程滞后系数的联合零限制
<!-- bilingual-en:start -->
*In a linear VAR, Granger noncausality is a joint zero restriction on lag coefficients in the target equation*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 在正确设定的线性 VAR 中，检验 $x$ 不 Granger 导致 $y$，就是联合检验 $y$ 的目标方程内所有所列 $x$ 滞后系数为零；这是一个联合原假设，未拒绝不等于证明这些系数确实为零。

以二变量 VAR($p$) 的 $y$ 方程为例：
$$
y_t=c+\sum_{i=1}^p a_i y_{t-i}
+\sum_{i=1}^p b_i x_{t-i}+u_{yt}.
$$
在线性条件均值口径下，
$$
H_0:\ b_1=b_2=\cdots=b_p=0
$$
表示给定模型中 $y$ 自身滞后及其他控制信息后，$x$ 的这些滞后不增加对 $y_t$ 的线性预测。应使用 Wald、F、LR 等联合检验，而不是因为某一个 $b_i$ 单独不显著就宣布非因果。若检验变量块对目标变量块的关系，相应限制是多个系数矩阵块的联合零。

拒绝 $H_0$ 表示至少一个所列滞后在给定规格下含有增量预测信息。未拒绝只能说样本证据不足以排除联合零，不能改写成“$x$ 没有预测力”或“证明不存在 Granger 关系”；功效、样本长度、滞后选择和参数不稳定都可能造成未拒绝。

在协整 VECM 中，预测通道还可能经过误差修正项，不能只检验差分滞后。此时应复用 [[弱外生与Granger非因果]] 中对短期与长期通道的联合限制。

> [!question]- 自检
> $b_1$ 显著而 $b_2,b_3$ 不显著，能否只看后二者就认定 $x$ 不 Granger 导致 $y$？
>
> **答案：** 不能。非因果原假设要求全部相关滞后系数联合为零，必须作联合检验。

## 来源与核验

- [Granger (1969), *Investigating Causal Relations by Econometric Models and Cross-spectral Methods*](https://doi.org/10.2307/1912791)：核对预测信息意义上的因果概念。
- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，因果分析章节：核对线性 VAR 中的系数限制与检验。
- [[弱外生与Granger非因果]]：复用协整系统的额外长期通道边界。

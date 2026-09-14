---
aliases:
  - "Granger 因果是一个变量的过去信息对另一个变量未来分布具有增量预测内容"
  - Granger causality
  - Predictive causality
student_os: knowledge-atom
atom_id: TS-VAR-026
atom_set: vector-autoregression
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[VAR Granger检验]]"
  - "[[Granger因果边界]]"
  - "[[弱外生与Granger非因果]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# Granger 因果是一个变量的过去信息对另一个变量未来分布具有增量预测内容
<!-- bilingual-en:start -->
*Granger causality means that one variable's past contains incremental predictive content for another variable's future distribution*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> Granger 因果是一个基于信息集的预测关系：在已经知道其他所列信息时，若加入 $x$ 的过去会改变对未来 $y$ 的条件分布，就说 $x$ Granger 导致 $y$。“导致”在这里首先指增量预测内容，不是尚未识别的政策干预效应。

<!-- bilingual-en:start -->
> [!summary] What it is
> Granger causality is an information-set-based predictive relation. Conditional on the other stated information, $x$ Granger-causes $y$ when adding the history of $x$ changes the conditional distribution of future $y$. Here "causes" first means incremental predictive content, not an already identified policy-intervention effect.
<!-- bilingual-en:end -->

令 $\mathcal F_t$ 是包含 $x$ 历史的信息集，$\mathcal F_t^{-x}$ 是保留其他信息但删去 $x$ 历史的比较集。$x$ **不** Granger 导致 $y$ 的分布定义要求，对所有相关期限 $h$ 和事件集 $A$，
$$
P\{y_{t+h}\in A\mid\mathcal F_t\}
=P\{y_{t+h}\in A\mid\mathcal F_t^{-x}\}.
$$
若等式在某个期限或事件上失败，$x$ 的历史就包含原信息集未提供的预测信息。

<!-- bilingual-en:start -->
Let $\mathcal F_t$ include the history of $x$ and let $\mathcal F_t^{-x}$ retain the comparison information while excluding that history. Distributional noncausality requires the two conditional probabilities above to agree for every relevant horizon and event. Failure for at least one comparison means that the history of $x$ contributes predictive information.
<!-- bilingual-en:end -->

实证中常用的线性 VAR 版本更窄：它只检查在所选变量、滞后和线性条件均值模型中，$x$ 的过去是否改善 $y$ 的条件均值预测。这时一般定义落地为 [[VAR Granger检验|目标方程中一组滞后系数的联合零限制]]。线性均值非因果不排除非线性、方差或更高阶分布中仍存在预测内容。

<!-- bilingual-en:start -->
The familiar linear-VAR implementation is narrower: it asks whether lags of $x$ improve the conditional-mean forecast of $y$ in the chosen linear specification. It becomes a joint zero restriction on a block of lag coefficients. Linear mean noncausality does not rule out predictive content in nonlinear features, variance, or higher-order aspects of the distribution.
<!-- bilingual-en:end -->

> [!example] 增量信息，不是单纯先后
> 若昨日期货价格在已经知道现货价格自身历史后，仍能改善对今日现货价格的预测，就可能出现从期货到现货的 Granger 关系。这不只要求期货变化得早，还要求它在比较信息集之外提供额外预测内容。

> [!question]- 自检
> 若 $x$ 在时间上总是早于 $y$ 变化，是否已足以说 $x$ Granger 导致 $y$？
>
> **答案：** 不足。还要在明确的比较信息集下证明 $x$ 的历史提供了增量预测内容；共同先行变量可以让 $x$ 只是更早的代理信号。

## 来源与核验

- [Granger (1969), *Investigating Causal Relations by Econometric Models and Cross-spectral Methods*](https://doi.org/10.2307/1912791)：核对以时间顺序、信息集和可预测性定义的原始概念。
- [Granger (1988), *Some Recent Development in a Concept of Causality*](https://doi.org/10.1016/0304-4076(88)90045-0)：核对预测因果、协整及政策解释间的区别。
- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)：核对多变量信息集下的因果定义与线性 VAR 落地。
- [[01_Math/06_时间序列分析/lecture.pdf#page=238|课程讲义 p. 238]]：核对本课以比较信息集与联合零限制介绍 Granger 因果的口径。

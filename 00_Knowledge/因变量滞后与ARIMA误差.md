---
aliases:
  - "因变量滞后传播条件均值而 ARIMA 误差传播未解释偏离"
  - "Lagged outcomes propagate the conditional mean whereas ARIMA errors propagate unexplained deviations"
  - "Lagged dependent variable versus ARIMA errors"
student_os: knowledge-atom
atom_id: TS-DYN-009
atom_set: dynamic-regression-intervention
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归预测与动态回归.canvas]]"
requires:
  - "[[ARIMA误差回归]]"
  - "[[自回归分布滞后]]"
related:
  - "[[误差序列相关]]"
  - "[[内生性]]"
leads_to:
  - "[[动态回归规格流程]]"
---

# 因变量滞后传播条件均值而 ARIMA 误差传播未解释偏离
<!-- bilingual-en:start -->
*Lagged outcomes propagate the conditional mean whereas ARIMA errors propagate unexplained deviations*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> 在 ADL 中，$y_{t-1}$ 直接进入 $E(y_t\mid\mathcal F_{t-1},x)$，所以过去结果会传播解释变量效应和初始条件。ARIMA 误差回归则让 $\eta_t=y_t-\beta'x_t$ 自相关，表示回归没有解释的偏离会持续。两种模型都能产生平滑预测路径，但系数含义、初始条件和外生性要求并不相同。
> <!-- bilingual-en:start -->
> A lagged outcome enters the conditional mean directly and propagates covariate effects and initial conditions. ARIMA errors instead make the unexplained regression deviation persistent. The two specifications can generate similar forecasts, but their coefficients, conditioning information, and exogeneity requirements differ.
> <!-- bilingual-en:end -->

一个简单代数关系说明它们为什么容易被混淆。若忽略截距，
$$
y_t=\beta x_t+\eta_t,
\qquad
\eta_t=\rho\eta_{t-1}+\varepsilon_t,
$$
则代入 $\eta_{t-1}=y_{t-1}-\beta x_{t-1}$ 得
$$
y_t=\rho y_{t-1}+\beta x_t-\rho\beta x_{t-1}+\varepsilon_t.
$$
它看起来像 ADL(1,1)，却带有特定限制：滞后 $x$ 的系数必须等于 $-\rho\beta$。一个自由估计的 ADL 不必满足这条限制，因此不能把两种规格的参数逐项当作同一对象。
<!-- bilingual-en:start -->
An AR(1) regression error can be rewritten as a restricted ADL: $y_t=\rho y_{t-1}+\beta x_t-\rho\beta x_{t-1}+\varepsilon_t$. The lagged-predictor coefficient is constrained by the error process. An unrestricted ADL need not satisfy this restriction, so its coefficients cannot be read as though they came from the same decomposition.
<!-- bilingual-en:end -->

选择哪种表达取决于问题。若理论说结果向目标水平部分调整，滞后 $y$ 是条件均值的一部分；若协变量关系是主要结构，而遗漏冲击按 ARIMA 持续，ARIMA 误差更自然。把两者和大量 $x$ 滞后同时加入，会让多个组件争相解释同一相关结构，造成弱识别、系数不稳定和外样本退化。
<!-- bilingual-en:start -->
Use lagged $y$ when the mechanism concerns partial adjustment or state persistence in the conditional mean. Use ARIMA errors when the covariate relation is primary and unexplained shocks have their own serial law. Combining many lags of $y$ and $x$ with a high-order error model can make several components compete for the same dependence.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两种模型的一步预测几乎相同，能否据此说它们的 $x$ 系数有相同经济含义？
>
> **答案：** 不能。预测等价或接近不等于参数分解相同；要检查条件均值、误差过程和参数限制。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 Chapter 10](https://otexts.com/fpp3/dynamic.html)：核对回归偏离 $\eta_t$ 与白噪声创新 $\varepsilon_t$ 的 ARIMA 误差表示。
- [[01_Math/06_时间序列分析/lecture.pdf#page=203|课程讲义 pp. 203–204]] 与 [[自回归分布滞后]]：核对滞后因变量进入条件均值的 ADL 表示；本卡的受限 ADL 关系由两式直接代入得到。

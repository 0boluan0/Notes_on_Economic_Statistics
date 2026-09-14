---
aliases:
  - "条件平方和条件高斯似然与精确高斯似然处理初值方式不同"
  - CSS versus exact Gaussian likelihood
  - Conditional versus exact ARMA likelihood
  - ARMA likelihood initialisation
  - Conditional and exact ARMA likelihood
  - ARMA initial-condition treatment
student_os: knowledge-atom
atom_id: TS-ARMA-014
atom_set: arma-modeling
atom_type: estimation-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARMA(p,q)模型]]"
  - "[[ARMA无限AR表示]]"
related:
  - "[[Yule-Walker方程]]"
  - "[[ARMA信息准则]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# 条件平方和条件高斯似然与精确高斯似然处理初值方式不同
<!-- bilingual-en:start -->
*Conditional sum of squares, conditional Gaussian likelihood, and exact Gaussian likelihood treat initial conditions differently*
<!-- bilingual-en:end -->

> [!summary] 原子区分
> ARMA 的创新不可直接观察，必须按候选参数递归重建。三种常见做法不能统称为一个算法：
> - **条件平方和（CSS/conditional least squares）**：给定或置零样本前误差与初值，最小化可计算期残差平方和；
> - **条件 Gaussian 似然**：在同样条件化的初值下，为这些创新加上独立同方差 Gaussian 密度；
> - **精确 Gaussian 似然**：在 Gaussian 状态空间中计入/积分初始状态不确定性，通常由 Kalman filter 评价联合似然。
> <!-- bilingual-en:start -->
> ARMA innovations are latent and reconstructed recursively. CSS conditions on chosen pre-sample values and minimizes residual squares. Conditional Gaussian likelihood adds a Gaussian density under that conditioning. Exact Gaussian likelihood evaluates the joint Gaussian model while accounting for initial-state uncertainty, often through a state-space Kalman filter.
> <!-- bilingual-en:end -->

在同方差 Gaussian 条件模型中，把 $\sigma^2$ profile 掉后，条件 Gaussian 对 ARMA 系数的最优点常与 CSS 相同；这只是特定条件下的目标函数对应，不代表两者的分布假设与报告的 likelihood 完全相同。精确似然使用最早观测所含的信息，有限样本估计可与 CSS 不同。

纯 AR 在给定最初 $p$ 个观测后，条件最小二乘可由 OLS 完成；含 MA 项时残差递归依赖未知参数，通常需要数值优化。软件的 `CSS`、`CSS-ML`、`ML` 或 “exact” 选项因此必须连同初值、有效样本和似然定义一起报告。以当前 R `stats::arima` 为例，`CSS-ML` 不是把两个 likelihood 混成一个目标：它先用 CSS 找起始值，再以 state-space/Kalman 计算的 exact Gaussian ML 完成估计；`CSS` 才把早期创新置零并使用条件平方和。
<!-- bilingual-en:start -->
After profiling the variance, conditional Gaussian likelihood and CSS can have the same coefficient minimizer under homoscedastic Gaussian conditioning, but they remain different statistical statements. Exact likelihood uses information in the initial observations. Pure AR conditional least squares can be OLS; latent MA innovations make ARMA estimation nonlinear and numerical. In current R `stats::arima`, `CSS-ML` uses CSS only to obtain starting values and then performs exact Gaussian ML; it is not a hybrid likelihood objective.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两个软件都写 “MLE”，但一个把样本前误差置零，另一个用 Kalman filter。AIC 能直接比较吗？
>
> **答案：** 不能先假定可比。必须确认它们计算的是同一观测样本上的同一似然定义；置零递归可能只是 CSS/条件似然。

## 来源与核验

- [R `stats::arima` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html)：直接区分 `CSS`、`CSS-ML` 与 exact likelihood，说明 CSS 的早期创新处理以及 ML 的 state-space/Kalman 计算。
- [[01_Math/06_时间序列分析/lecture.pdf#page=111|课程讲义 pp. 111–119]]：核对纯 AR 的 OLS、MA(1) 递归残差与数值似然例子。
- [Hyndman & Athanasopoulos, FPP3 §9.6](https://otexts.com/fpp3/arima-estimation.html)：核对 ARIMA MLE 与平方和估计的关系及软件差异。

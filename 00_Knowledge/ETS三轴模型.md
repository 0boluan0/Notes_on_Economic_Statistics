---
aliases:
  - "ETS 用误差趋势季节三轴定义创新状态空间模型"
  - ETS error trend seasonal notation
  - ETS innovations state-space model
  - ETS 状态空间记号
student_os: knowledge-atom
atom_id: TS-ETS-005
atom_set: exponential-smoothing-ets
atom_type: model-definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[SES权重机制]]"
related:
  - "[[Holt阻尼趋势]]"
  - "[[Holt-Winters季节形式]]"
  - "[[加法平滑与加法误差]]"
  - "[[ETS创新残差]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# ETS 用误差趋势季节三轴定义创新状态空间模型
<!-- bilingual-en:start -->
*ETS defines innovations state-space models along error, trend, and seasonal axes*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> ETS$(E,T,S)$ 的三个位置依次是 Error、Trend、Seasonal，而不是三种可随意交换的“平滑成分”。观测方程、状态更新和创新分布共同把点预测算法变成可写 likelihood 和 forecast distribution 的随机模型。
> <!-- bilingual-en:start -->
> In ETS$(E,T,S)$, the positions mean Error, Trend, and Seasonal. Measurement and state equations, together with an innovation distribution, turn a point-forecasting recursion into an innovations state-space model with a likelihood and forecast distribution.
> <!-- bilingual-en:end -->

常用符号是：$E\in\{A,M\}$，$T\in\{N,A,A_d\}$，$S\in\{N,A,M\}$。例如 ETS$(A,A_d,A)$ 表示 additive error、additive damped trend、additive seasonality；下标 $d$ 修饰趋势，不是误差。

这个三轴记号只说明规格，不保证任意字母组合都适合数据或数值稳定。特别是含 multiplicative error 的模型通常要求严格正值数据；某些 additive-error、multiplicative-seasonal 组合也会因状态方程除以接近零的量而被自动选模实现排除。

最简单的 ETS$(A,N,N)$ 写成
$$
y_t=\ell_{t-1}+\varepsilon_t,\qquad
\ell_t=\ell_{t-1}+\alpha\varepsilon_t.
$$
再指定例如 $\varepsilon_t\overset{iid}{\sim}N(0,\sigma^2)$，才得到完整的 Gaussian additive-error 随机模型。第一式是 measurement equation，说明观测等于一步条件位置加创新；第二式是 state equation，说明同一创新怎样修正不可观测水平。消去 $\varepsilon_t=y_t-\ell_{t-1}$，就恢复 SES 的平滑递推。更复杂 ETS 以同一原则更新 level、trend 和 seasonal states。

“innovations state space” 的关键是同一个一步创新驱动观测偏差与状态修正。它不等同于任意多源噪声的结构状态空间模型，也不表示 level/trend/seasonal 是可以直接观测或具有唯一经济解释的真实成分。不同初始状态、误差尺度与参数限制会给出不同 likelihood 与区间，即使点预测递推看起来相近。
<!-- bilingual-en:start -->
For ETS(A,N,N), $y_t=\ell_{t-1}+\varepsilon_t$ and $\ell_t=\ell_{t-1}+\alpha\varepsilon_t$. The same one-step innovation both explains the observation error and updates the latent state. More complex ETS models extend this mechanism to trend and seasonal states; these states are statistical constructs, not uniquely identified economic mechanisms.
<!-- bilingual-en:end -->

> [!question]- 自检
> ETS$(M,A_d,N)$ 的三个字母分别说明什么？能否把第一个 M 读成“乘法季节”？
>
> **答案：** 依次是 multiplicative error、additive damped trend、no seasonality。第一个位置只描述误差，不能读成季节形式。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §8.5](https://otexts.com/fpp3/ets.html)：核对 ETS 三位置记号、观测/状态方程与 ETS(A,N,N) 的创新形式。
- [Hyndman, Koehler, Snyder & Grose (2002)](https://doi.org/10.1016/S0169-2070(01)00110-8)：原始 innovations state-space ETS 框架；核对单一误差源、likelihood 与区间的连接。

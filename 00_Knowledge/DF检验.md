---
aliases:
  - "DF 检验把 AR 单位根改写为滞后水平系数为零"
  - Dickey-Fuller test
  - DF test
  - DF 检验
student_os: knowledge-atom
atom_id: TS-UR-009
atom_set: trends-unit-roots-differencing
atom_type: test-principle
status: source-checked
mastery_state: unassessed
requires:
  - "[[无漂移随机游走]]"
  - "[[单位根最小表示]]"
related:
  - "[[单位根确定项规格]]"
  - "[[ADF检验]]"
part_of:
  - "[[趋势、单位根与差分.canvas]]"
---

# DF 检验把 AR 单位根改写为滞后水平系数为零
<!-- bilingual-en:start -->
*The Dickey–Fuller test rewrites an AR unit root as a zero coefficient on the lagged level*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 对 $y_t=\rho y_{t-1}+\varepsilon_t$ 两边减去 $y_{t-1}$，得到
> $$
> \Delta y_t=\gamma y_{t-1}+\varepsilon_t,\qquad \gamma=\rho-1.
> $$
> 因而单位根原假设 $\rho=1$ 等价于 $H_0:\gamma=0$。Dickey–Fuller 检验的核心就是检验滞后水平项系数是否为零。

在最基本的 DF 模型中，创新被假定为独立同分布且方差有限；加入截距或趋势会改变检验设定。由于 $H_0$ 下 $y_{t-1}$ 是累积创新而非通常的平稳回归量，$\widehat\gamma$ 的 $t$ 统计量不服从普通 Student-$t$ 极限分布，必须使用与确定性项相匹配的 DF 临界值。

左尾拒绝 $H_0$ 是沿 $\rho<1$ 的方向反对根 $\rho=1$。只有在检验预先把备择维持在 $-1<\rho<1$ 时，才能把这个方向解释为平稳 AR(1)；单凭 $\gamma<0$ 并不能排除 $\rho\le -1$。即使拒绝，也仍需检查其余根、残差和确定性项，不能直接写成“模型已正确且平稳”。

> [!question]- 自检
> 为什么不能拿普通回归的 $-1.96$ 来判断 DF 统计量？
>
> **答案：** 因为单位根原假设下滞后水平不是通常的平稳回归量，统计量具有非标准极限分布；还必须匹配回归中是否含截距和趋势。

## 来源与核验

- [Dickey & Fuller (1979), *Distribution of the Estimators for Autoregressive Time Series With a Unit Root*](https://doi.org/10.1080/01621459.1979.10482531)：核对差分参数化、原假设和非标准分布。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程中的 AR(1) 单位根推导与 DF 统计量。

---
aliases:
  - "ARIMA 误差回归把协变量解释与未解释偏离的序列动态分开"
  - "Regression with ARIMA errors separates covariate effects from serial dynamics in unexplained deviations"
  - "Regression with ARIMA errors"
student_os: knowledge-atom
atom_id: TS-DYN-002
atom_set: dynamic-regression-intervention
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归预测与动态回归.canvas]]"
requires:
  - "[[动态回归模型]]"
  - "[[ARIMA模型]]"
leads_to:
  - "[[事前与事后回归预测]]"
  - "[[预测变量路径与条件区间]]"
related:
  - "[[误差序列相关]]"
  - "[[ARMA残差诊断]]"
contrasts_with:
  - "[[HAC协方差]]"
  - "[[因变量滞后与ARIMA误差]]"
---

# ARIMA 误差回归把协变量解释与未解释偏离的序列动态分开
<!-- bilingual-en:start -->
*Regression with ARIMA errors separates covariate effects from serial dynamics in unexplained deviations*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 回归的系统部分解释 $x_t$ 与 $y_t$ 的条件关系，回归误差 $\eta_t$ 再服从 ARIMA。模型有两个不能混称的误差对象：
> $$
> y_t=\beta_0+\beta'x_t+\eta_t,
> \qquad
> \phi(B)(1-B)^d\eta_t=\theta(B)\varepsilon_t,
> $$
> 其中 $\eta_t$ 是回归未解释的偏离，只有 ARIMA 创新 $\varepsilon_t$ 才应近似白噪声。
> <!-- bilingual-en:start -->
> The regression component explains the conditional relation between $x_t$ and $y_t$, while the regression error $\eta_t$ follows an ARIMA process. The regression deviation $\eta_t$ and the ARIMA innovation $\varepsilon_t$ are different objects; only the latter should resemble white noise.
> <!-- bilingual-en:end -->

这一规格不是“先做 OLS，再给标准误加一个自相关修正”。ARIMA 误差进入联合 likelihood、系数的有效估计、未来误差递推和预测区间。[[HAC协方差]]只修正某些估计量的协方差口径，并不会自动给出 ARIMA 的多步预测分布；两者回答的问题不同。
<!-- bilingual-en:start -->
This is not OLS followed by a standard-error patch. The ARIMA error law affects the joint likelihood, efficient coefficient estimation, forecast recursion, and predictive distribution. HAC covariance adjusts an inference formula but does not supply an ARIMA multi-step forecast distribution.
<!-- bilingual-en:end -->

诊断时要查看 innovation residual，而不是只看 regression residual。回归残差 $\hat\eta_t$ 可以保留模型明确描述的 ARIMA 相关；若创新残差 $\hat\varepsilon_t$ 仍有系统性自相关，才说明当前回归项或误差阶数没有吸收完可预测的线性结构。通过白噪声检查仍不证明正态、同方差或模型真实，解释边界沿用 [[ARMA残差诊断]]。
<!-- bilingual-en:start -->
Diagnostics target innovation residuals. Regression residuals may retain the ARIMA dependence that the model explicitly represents; remaining correlation in estimated innovations instead signals an incomplete mean or error specification. A white-noise diagnostic still does not establish Gaussianity, homoscedasticity, or model truth.
<!-- bilingual-en:end -->

若模型需要差分，必须说明差分作用在哪些变量和截距上。某些软件会把同一个差分算子同时施加到 $y$ 与全部回归量；这是实现约定，不是可以忽略的细节。含 $I(1)$ 水平变量时还要先检查 [[协整与差分边界]]，不能把 ARIMA 误差当成伪回归的自动修复。
<!-- bilingual-en:start -->
When differencing is used, document which variables and intercept terms are transformed. Some implementations apply the same differencing operator to the response and every regressor; that is a consequential software convention. ARIMA errors do not automatically repair a spurious levels regression among nonstationary variables.
<!-- bilingual-en:end -->

> [!question]- 自检
> 拟合输出同时给出 regression residual 和 innovation residual。哪一个应接受“近似白噪声”的核心检查？
>
> **答案：** innovation residual；regression residual 本来就可以按拟合的 ARIMA 过程相关。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 Chapter 10](https://otexts.com/fpp3/dynamic.html)：核对 $\eta_t$ 与 $\varepsilon_t$ 两层误差及只有后者为白噪声。
- [FPP3 §10.2](https://otexts.com/fpp3/regarima.html)：核对联合估计、差分实现和 regression/innovation residual 的区别。
- [[ARMA残差诊断]] 与 [[协整与差分边界]]：复用残差解释和非平稳变量的已核验边界。

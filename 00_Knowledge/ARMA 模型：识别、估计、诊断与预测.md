---
aliases:
  - "ARMA"
  - "ARMA Models"
  - "Stationary Time Series and ARMA"
  - "平稳时间序列"
status: source-checked
---

# ARMA 模型：识别、估计、诊断与预测
<!-- bilingual-en:start -->
*ARMA models: identification, estimation, diagnostics, and forecasting*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用有限参数描述均值回归序列的动态依赖，并从 ACF/PACF、残差和预测表现判断模型是否够用。
> **具体锚点：** AR(1) $y_t=\phi y_{t-1}+\varepsilon_t$ 在 $|\phi|<1$ 时冲击按 $\phi^h$ 衰减，相关性也呈几何下降。
> **核心难点：** 平稳性、因果性和可逆性是关于表示能否稳定展开的不同条件；样本 ACF 只是有误差的线索。
> **为什么重要：** 它是时序预测、动态回归和更复杂波动/多变量模型的基线。
> **继续：** 均值结构合格但波动聚集时进入 [[条件异方差：ARCH 与 GARCH]]；趋势明显时先看 [[趋势、单位根与差分]]。
> <!-- bilingual-en:start -->
> **What it solves:** ARMA uses a finite number of parameters to describe dynamic dependence in a mean-reverting series, then tests adequacy through ACF/PACF, residuals, and forecasting performance.
> **Concrete anchor:** In the AR(1) model $y_t=\phi y_{t-1}+\varepsilon_t$, $|\phi|<1$ makes a shock and the autocorrelation decay geometrically as $\phi^h$.
> **Central difficulty:** Stationarity, causal representation, and invertibility are distinct conditions about stable expansions. A sample ACF is noisy evidence, not a model identifier.
> **Why it matters:** ARMA is a baseline for time-series forecasting, dynamic regression, conditional-volatility models, and multivariate dynamics.
> **Continue with:** Move to [[条件异方差：ARCH 与 GARCH|ARCH and GARCH]] when mean residuals are uncorrelated but volatility clusters, or first examine [[趋势、单位根与差分|trends, unit roots, and differencing]] when the level is nonstationary.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
> - Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
> <!-- bilingual-en:start -->
> - [[01_Math/06_时间序列分析/lecture.pdf|time-series lecture notes]] and [[01_Math/06_时间序列分析/lecture-dual.pdf|bilingual time-series lecture notes]] were checked for definitions, operator conventions, ACF/PACF patterns, estimation, diagnostics, and forecast formulas.
> - Hyndman and Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/), was used to cross-check ARIMA modelling, residual diagnostics, and time-series cross-validation.
> <!-- bilingual-en:end -->

## 弱平稳与白噪声
<!-- bilingual-en:start -->
*Weak stationarity and white noise*
<!-- bilingual-en:end -->

> [!note] 理论与实操的分工
> 下面一段保留课程中使用 ARMA 所需的最小前提；平稳性、遍历性和谱的完整理论见 [[平稳性、遍历性与谱]]。
> <!-- bilingual-en:start -->
> The next paragraph retains the minimum prerequisite needed to use ARMA in this course. See [[平稳性、遍历性与谱|stationarity, ergodicity, and spectral analysis]] for the full theory.
> <!-- bilingual-en:end -->

弱平稳要求均值不随时间变、协方差只依赖滞后。白噪声均值为零、方差恒定、不同期不相关；它不必独立或正态。模型残差“近似白噪声”表示线性均值结构未留下可预测自相关，不等于所有结构都正确。
<!-- bilingual-en:start -->
Weak stationarity requires a time-invariant mean and covariance that depends only on lag. White noise has zero mean, constant variance, and zero correlation across different dates; it need not be independent or Gaussian. Residuals that are approximately white noise indicate that no predictable linear mean dependence remains, not that every aspect of the model is correct.
<!-- bilingual-en:end -->

## AR、MA 与 ARMA
<!-- bilingual-en:start -->
*AR, MA, and ARMA representations*
<!-- bilingual-en:end -->

AR(p) 用过去观测解释当前，MA(q) 用当前和过去冲击表示当前。ARMA 结合两者。平稳因果 AR 表示可展开为收敛的无限 MA；可逆 MA 表示可由过去观测恢复冲击。公共因子应约去，否则参数不识别。
<!-- bilingual-en:start -->
An AR($p$) uses past observations to explain the present, an MA($q$) represents the present with current and past innovations, and ARMA combines both. A stationary causal AR representation expands as a convergent infinite MA; an invertible MA permits innovations to be recovered from past observations. Common AR and MA factors must be cancelled or the parameters are not identified.
<!-- bilingual-en:end -->

用滞后多项式写成
$$
\phi(L)(y_t-\mu)=\theta(L)\varepsilon_t,
$$
其中 $\phi(z)=1-\phi_1z-\cdots-\phi_pz^p$，$\theta(z)=1+\theta_1z+\cdots+\theta_qz^q$。因果性要求 $\phi(z)$ 的零点在单位圆外，可逆性要求 $\theta(z)$ 的零点在单位圆外。两者一个约束“观测对冲击”的稳定表示，一个约束“冲击对观测”的稳定恢复。
<!-- bilingual-en:start -->
In lag-polynomial notation,
$$
\phi(L)(y_t-\mu)=\theta(L)\varepsilon_t,
$$
where $\phi(z)=1-\phi_1z-\cdots-\phi_pz^p$ and $\theta(z)=1+\theta_1z+\cdots+\theta_qz^q$. Causality requires the zeros of $\phi(z)$ outside the unit circle, and invertibility requires the zeros of $\theta(z)$ outside it. One guarantees a stable representation of observations in innovations; the other guarantees stable recovery of innovations from observations.
<!-- bilingual-en:end -->

## ACF 与 PACF
<!-- bilingual-en:start -->
*ACF and PACF*
<!-- bilingual-en:end -->

ACF 衡量 $y_t$ 与 $y_{t-h}$ 线性相关；PACF 控制中间滞后后看净相关。理想 AR(p) 的 PACF 在 p 后截尾、MA(q) 的 ACF 在 q 后截尾，但有限样本、近单位根和混合模型会让模式模糊。
<!-- bilingual-en:start -->
The ACF measures linear correlation between $y_t$ and $y_{t-h}$; the PACF measures their remaining correlation after controlling for intervening lags. In ideal population patterns, an AR($p$) PACF cuts off after $p$ and an MA($q$) ACF cuts off after $q$, but finite samples, near-unit roots, and mixed ARMA models blur these signatures.
<!-- bilingual-en:end -->

样本 ACF/PACF 应用来生成少量候选而不是反向“读图解方程”。例如一个强持续 AR(1) 的 ACF 可在很多滞后上显著，而 PACF 在第 1 阶之后的小幅波动只可能是抽样噪声。选阶必须与信息准则、残差和外样本用途合并。
<!-- bilingual-en:start -->
Use sample ACF and PACF to generate a small candidate set, not to solve a model backwards from a plot. For instance, a persistent AR(1) can have significant ACF values at many lags, while small PACF fluctuations after lag 1 may be sampling noise. Order selection must also use information criteria, residuals, and out-of-sample purpose.
<!-- bilingual-en:end -->

## 识别、估计与诊断
<!-- bilingual-en:start -->
*Identification, estimation, and diagnostics*
<!-- bilingual-en:end -->

先画序列并处理明显趋势/季节性，再用 ACF/PACF 和理论提出小规模候选；以 MLE 或其他方法估计，用 AIC/BIC、残差 ACF 和 Ljung–Box 检查。对多个模型的选择应面向用途：解释、一步预测和多步预测可能偏好不同结构。
<!-- bilingual-en:start -->
Plot the series and address obvious trend or seasonality before using ACF/PACF and theory to propose a small candidate set. Estimate by maximum likelihood or another stated method, then inspect AIC/BIC, residual ACF, and Ljung–Box tests. Model choice must match purpose: interpretation, one-step forecasting, and multi-step forecasting may favour different structures.
<!-- bilingual-en:end -->

建议的验收顺序是：

1. 参数是否满足因果/可逆条件，且没有近似抵消的 AR–MA 公共因子；
2. 标准化残差的序列图和 ACF 没有明显结构；
3. Ljung–Box 在事先选定的多个滞后上不显示整体自相关，但不把“未拒绝”读成模型为真；
4. 残差平方、异常点和结构断点也被检查；
5. 在 rolling-origin 中与 naïve 及其他小模型比较。
<!-- bilingual-en:start -->
A practical acceptance sequence is:

**1.** Verify causal and invertible parameter regions and the absence of nearly cancelling AR–MA factors.<br>
**2.** Inspect the standardised residual path and residual ACF for remaining structure.<br>
**3.** Use Ljung–Box tests at several prespecified lags, without interpreting failure to reject as proof that the model is true.<br>
**4.** Inspect squared residuals, unusual observations, and structural breaks as well.<br>
**5.** Compare against naïve and other small models under rolling-origin evaluation.<br>
<!-- bilingual-en:end -->

## 预测
<!-- bilingual-en:start -->
*Forecasting*
<!-- bilingual-en:end -->

### ARIMA 作为差分后的 ARMA
<!-- bilingual-en:start -->
*ARIMA as ARMA after differencing*
<!-- bilingual-en:end -->

ARIMA 先通过常规/季节差分处理非平稳，再用 AR/MA 捕捉差分序列自相关。阶数由图形、信息准则和残差共同选择。差分次数比增加 AR/MA 阶数更需谨慎。
<!-- bilingual-en:start -->
ARIMA first uses ordinary or seasonal differencing to address nonstationarity, then uses AR and MA terms to capture autocorrelation in the differenced series. Plots, information criteria, and residuals jointly inform order selection. The number of differences requires even more caution than additional AR or MA orders.
<!-- bilingual-en:end -->

ARIMA($p,d,q$) 可写为 $\phi(L)(1-L)^dy_t=\theta(L)\varepsilon_t$；季节版本再乘上 $(1-L^s)^D$ 及季节 AR/MA 多项式。$d$ 和 $D$ 改变被建模的对象，而 $p,q$ 只在给定差分序列上改变短期依赖表示，这就是为什么应先审核差分。
<!-- bilingual-en:start -->
ARIMA($p,d,q$) can be written $\phi(L)(1-L)^dy_t=\theta(L)\varepsilon_t$; a seasonal model adds $(1-L^s)^D$ and seasonal AR and MA polynomials. The choices $d$ and $D$ change the object being modelled, while $p$ and $q$ alter short-run dependence for a given differenced series. Differencing therefore deserves review before AR and MA orders.
<!-- bilingual-en:end -->

点预测是条件期望或相应最优预测，预测误差方差随期限累积并趋向无条件方差（稳定过程）。未来冲击未知，递归预测将其期望设为零；区间必须反映参数和创新不确定性。
<!-- bilingual-en:start -->
A point forecast is a conditional expectation or the corresponding optimal predictor. For a stable process, forecast-error variance accumulates with horizon and approaches the unconditional variance. Recursive forecasts set unknown future innovations to their conditional mean of zero; intervals must reflect innovation and, where relevant, parameter uncertainty.
<!-- bilingual-en:end -->

对 AR(1) $y_t-\mu=\phi(y_{t-1}-\mu)+\varepsilon_t$，
$$
\hat y_{t+h|t}=\mu+\phi^h(y_t-\mu),\qquad
\operatorname{Var}(e_{t+h|t})=\sigma^2\sum_{j=0}^{h-1}\phi^{2j}.
$$
因此 $|\phi|<1$ 时点预测向 $\mu$ 回归，误差方差向 $\sigma^2/(1-\phi^2)$ 回归。这不是说远期值更稳定，而是说远期的最佳点预测只剩无条件均值，同时不确定性更大。
<!-- bilingual-en:start -->
For the AR(1) model $y_t-\mu=\phi(y_{t-1}-\mu)+\varepsilon_t$,
$$
\hat y_{t+h|t}=\mu+\phi^h(y_t-\mu),\qquad
\operatorname{Var}(e_{t+h|t})=\sigma^2\sum_{j=0}^{h-1}\phi^{2j}.
$$
When $|\phi|<1$, the point forecast returns toward $\mu$ and forecast-error variance approaches $\sigma^2/(1-\phi^2)$. This does not mean distant outcomes are more stable; it means the best distant point forecast contains only the unconditional mean while uncertainty is larger.
<!-- bilingual-en:end -->

## 边界
<!-- bilingual-en:start -->
*Boundaries*
<!-- bilingual-en:end -->

结构突变、非线性、厚尾和条件异方差可使 ARMA 均值模型看似合格但区间失真。不要用增加滞后阶数代替对数据生成机制的思考。
<!-- bilingual-en:start -->
Structural breaks, nonlinearity, heavy tails, and conditional heteroskedasticity can leave an ARMA mean model apparently adequate while making intervals unreliable. Adding lags is not a substitute for reasoning about the data-generating mechanism.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 白噪声是否一定独立同分布？
<!-- bilingual-en:start -->
*Must white noise be independent and identically distributed?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不是。定义通常只要求零均值、恒方差和零自协方差；独立或 Gaussian 是更强条件。
> <!-- bilingual-en:start -->
> No. The usual definition requires only zero mean, constant variance, and zero autocovariance; independence or Gaussianity is stronger.
> <!-- bilingual-en:end -->

### AR(1) 的 $|\phi|<1$ 同时带来什么直觉？
<!-- bilingual-en:start -->
*What connected intuitions follow from $|\phi|<1$ in an AR(1)?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 冲击几何衰减、存在稳定无限 MA 表示、无条件方差有限且相关随滞后衰减。
> <!-- bilingual-en:start -->
> Shocks decay geometrically, a stable infinite-MA representation exists, unconditional variance is finite, and autocorrelation decays with lag.
> <!-- bilingual-en:end -->

### 为什么不能只凭样本 ACF 截尾就确定模型阶数？
<!-- bilingual-en:start -->
*Why can model order not be determined from a cutoff in the sample ACF alone?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 样本 ACF 有抽样误差，近单位根、混合 ARMA、季节性和结构突变都会模糊理论图形。
> <!-- bilingual-en:start -->
> The sample ACF has sampling error, and near-unit roots, mixed ARMA terms, seasonality, and structural breaks all blur theoretical patterns.
> <!-- bilingual-en:end -->

### 用自己的话区分因果表示与可逆表示。
<!-- bilingual-en:start -->
*Distinguish a causal representation from an invertible representation in your own words.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 因果表示用当前和过去冲击稳定地生成观测；可逆表示用当前和过去观测稳定地恢复冲击。
> <!-- bilingual-en:start -->
> A causal representation stably generates observations from current and past innovations; an invertible representation stably recovers innovations from current and past observations.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
- Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
<!-- bilingual-en:start -->
- [[01_Math/06_时间序列分析/lecture.pdf|time-series lecture notes]] and [[01_Math/06_时间序列分析/lecture-dual.pdf|bilingual time-series lecture notes]] were checked for ARMA definitions, operator conventions, ACF/PACF patterns, estimation, diagnostics, and forecast formulas.
- Hyndman and Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/), was checked against the ARIMA modelling workflow, residual diagnostics, and forecast evaluation.
<!-- bilingual-en:end -->

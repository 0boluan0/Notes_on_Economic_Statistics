---
aliases:
  - "Ljung–Box 联合检验多个残差自相关并需按拟合调整自由度"
  - Ljung-Box test
  - Ljung–Box test
  - Ljung–Box portmanteau test
  - Ljung–Box 白噪声联合检验
student_os: knowledge-atom
atom_id: TS-ARMA-017
atom_set: arma-modeling
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[白噪声二阶定义]]"
  - "[[自协方差与ACF]]"
related:
  - "[[ARMA残差诊断]]"
  - "[[Box-Jenkins流程]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# Ljung–Box 联合检验多个残差自相关并需按拟合调整自由度
<!-- bilingual-en:start -->
*Ljung–Box jointly tests several residual autocorrelations and adjusts degrees of freedom for the fitted model*
<!-- bilingual-en:end -->

> [!summary] 原子检验
> 对选定最大滞后 $\ell$，Ljung–Box 的原假设是
> $$H_0:\rho(1)=\cdots=\rho(\ell)=0,$$
> 并用
> $$Q^*=n_{\mathrm{eff}}(n_{\mathrm{eff}}+2)\sum_{k=1}^{\ell}\frac{r_k^2}{n_{\mathrm{eff}}-k}$$
> 联合衡量前 $\ell$ 个样本自相关。它是 portmanteau 联合检验，不是逐个尖峰检验。
> <!-- bilingual-en:start -->
> At a chosen maximum lag, Ljung–Box tests the joint null that all population autocorrelations through that lag are zero. Its statistic aggregates the corresponding sample autocorrelations rather than testing one spike at a time.
> <!-- bilingual-en:end -->

$n_{\mathrm{eff}}$ 是实际进入检验的残差数；初始化丢失、缺失值或其他样本裁剪会使它不同于原始样本长度。若直接检验未拟合序列，常用近似参考自由度 $\ell$。若检验模型残差，估计参数已经消耗拟合自由度，需按所用程序的 `fitdf`/`model_df` 约定调整；对常规非季节 ARMA，教材与 FPP 常取动态参数数 $p+q$，参考自由度为 $\ell-(p+q)$。但截距、回归项、季节参数和约束怎样计数依实现与理论近似，不能机械地永远“再减一”。必须记录实际传入的 model degrees of freedom。当前 statsmodels 还会先对传入序列去均值，因此调用口径也应记录。

若 $\ell-\text{model\_df}\le0$，参考检验没有正自由度；statsmodels 会返回 NaN，而不是一个可解释 p 值。$\ell$ 也应在看结果前按样本量与季节周期选择。未拒绝只表示该联合检验没有发现这些滞后上的线性相关，不是接受模型为真。
<!-- bilingual-en:start -->
The statistic uses the effective number of residuals, which can differ from the original sample after initialization or missing-value handling. For fitted residuals, the reference degrees of freedom must reflect the model and implementation. A common non-seasonal ARMA adjustment uses $p+q$, but intercepts, regressors, seasonal terms, and constraints are not governed by a universal “always subtract one” rule. Current statsmodels also demeans the supplied series; lags with $\ell-\text{model_df}\le0$ return NaN. Failure to reject is limited evidence about linear autocorrelation, not proof of model truth.
<!-- bilingual-en:end -->

> [!question]- 自检
> 拟合 ARMA(3,2) 后在 $\ell=4$、`model_df=5` 做 Ljung–Box，p 值应如何解释？
>
> **答案：** 不应产生可解释 p 值，因为调整后自由度不为正；应选择合理更大的 $\ell$，而不是忽略 `model_df`。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=121|课程讲义 pp. 121–123]]：核对 Box–Pierce/Ljung–Box 统计量、联合原假设与课程自由度近似。
- [Hyndman & Athanasopoulos, FPP3 §5.4](https://otexts.com/fpp3/diagnostics.html)：核对 portmanteau 检验的联合原假设与滞后选择。
- [Hyndman & Athanasopoulos, FPP3 §9.7](https://otexts.com/fpp3/arima-r.html)：核对拟合 ARIMA 残差时的参数自由度调整。
- [statsmodels `acorr_ljungbox`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html)：核对 `model_df`、$\text{lag}-\text{model_df}$ 与非正自由度返回 NaN 的实现边界。

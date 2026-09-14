---
aliases:
  - "Breusch–Godfrey 检验在保留原回归量的辅助回归中联合检验多个残差滞后"
  - Breusch-Godfrey test
  - BG serial-correlation test
student_os: knowledge-atom
atom_id: ECON-ERR-008
atom_set: regression-error-covariance
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[误差序列相关]]"
  - "[[残差协方差诊断]]"
related:
  - "[[Durbin-Watson检验]]"
  - "[[Ljung-Box检验]]"
leads_to:
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# Breusch–Godfrey 检验在保留原回归量的辅助回归中联合检验多个残差滞后
<!-- bilingual-en:start -->
*The Breusch–Godfrey test jointly tests several residual lags in an auxiliary regression that retains the original regressors*
<!-- bilingual-en:end -->

> [!summary] 检验怎样做
> 先拟合原回归取得残差 $\hat u_t$，选定最高滞后 $p$，再估计
> $$
> \hat u_t=X_t'\delta+\rho_1\hat u_{t-1}+\cdots+\rho_p\hat u_{t-p}+v_t.
> $$
> 原假设是 $H_0:\rho_1=\cdots=\rho_p=0$。常见 LM 版本用有效辅助样本的 $nR^2$ 与 $\chi_p^2$ 比较，也可报告相应的 F 版本。
>
> <!-- bilingual-en:start -->
> Fit the original regression, then regress its residuals on the original regressors and residual lags through a pre-chosen order $p$. The null jointly sets all lag coefficients to zero; common software reports LM and F versions.
> <!-- bilingual-en:end -->

保留 $X_t$ 很重要：我们是在问“控制原模型解释部分后，过去残差还有没有增量解释力”，而不是只计算一个残差相关系数。与[[Durbin-Watson检验]]相比，BG 可以检验高于一阶的误差相关，并能用于包含滞后因变量的许多动态回归设定；但它仍需要相应的大样本、外生性与误差矩条件，不能把“更一般”理解为无条件有效。
<!-- bilingual-en:start -->
Retaining $X_t$ asks whether lagged residuals add explanatory power after controlling for the original model. Compared with DW, BG can test higher orders and is usable in many dynamic regressions with lagged outcomes, subject to its asymptotic, exogeneity, and moment conditions.
<!-- bilingual-en:end -->

$p$ 是检验问题的一部分，不能看完所有结果再挑最显著的阶数。它应依据数据频率、理论动态和预先关心的时间窗口选择；若做多个 $p$ 的敏感性分析，应完整报告。
<!-- bilingual-en:start -->
The order $p$ is part of the question, not a result to optimise after inspecting many tests. Choose it from frequency, theory, and the dependence horizon of interest, and report any sensitivity analysis transparently.
<!-- bilingual-en:end -->

前 $p$ 期的残差滞后没有自然观测值，但这不推出所有实现都必须删掉前 $p$ 行。常见口径有两种：

- **截短口径：** 从 $t=p+1$ 开始估计辅助回归，若没有其他缺失，辅助样本量为 $T-p$；
- **补零口径：** 在残差序列前补 $p$ 个零，再构造滞后，从而保留原回归的 $T$ 个观测。statsmodels 的 `acorr_breusch_godfrey` 采用这一口径。

两种处理在常见正则条件下具有相同的渐近目标，但有限样本统计量不必相同。LM 公式中的 $n$ 必须是**实际进入辅助回归的观测数**；F 版本的分母自由度也必须根据该实际样本和辅助设计矩阵的实际秩计算，不能一律把 $n$ 写成 $T-p$。

<!-- bilingual-en:start -->
The first $p$ residual lags are not naturally observed, but implementations need not all delete the first $p$ rows. A truncated convention starts the auxiliary regression at $t=p+1$, giving $T-p$ observations absent other missing values. A zero-padding convention prepends $p$ zeros before forming the lag matrix and can retain all $T$ original-regression observations; statsmodels uses this convention. The LM multiplier $n$ and the F denominator degrees of freedom must follow the observations and rank actually used in the auxiliary regression, not a universal $T-p$ rule.
<!-- bilingual-en:end -->

BG 拒绝并没有告诉你相关一定来自 AR($p$) 误差。遗漏季节性、趋势、结构突变或滞后解释变量都可能留下可预测残差。若目的只是对某个稳定均值参数做推断，可考虑合适的[[HAC协方差]]；若相关揭示均值遗漏，则应先重设模型。
<!-- bilingual-en:start -->
Rejection does not prove that the true error follows AR($p$). Omitted seasonality, trend, breaks, or lagged covariates can leave predictable residuals. HAC may suit inference about a stable mean parameter, whereas omitted mean dynamics require respecification.
<!-- bilingual-en:end -->

> [!question]- 自检
> 季度数据只做 $p=1$ 的 BG 未拒绝，能否排除一年内的残差季节相关？
>
> **答案：** 不能。$p=1$ 只联合检验到一阶；若一年窗口有理论意义，应预先检验包含相应季度滞后的设定。

## 来源与核验

- [[02_Economy/01_Econometrics/08_自相关.md#4.3. 布罗施–戈弗雷 BG 检验|本地课程：BG]]：核对辅助回归和 $nR^2$ 的课程顺序。
- [statsmodels `acorr_breusch_godfrey` 源码](https://www.statsmodels.org/stable/_modules/statsmodels/stats/diagnostic.html)：核对其先补 $p$ 个零、保留原回归全部观测，并以实际辅助样本的 `nobs * R²` 计算 LM；F 检验沿用实际辅助回归自由度。
- [Stata, *Time-Series Reference Manual*, pp. 9–10 and 562](https://www.stata.com/manuals18/ts.pdf)：核对 BG 的高阶能力、动态回归适用边界和 AR(1) 示例。

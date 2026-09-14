---
aliases:
  - "Durbin–Watson 统计量主要诊断特定静态回归中的一阶误差序列相关"
  - Durbin-Watson test
  - DW test
student_os: knowledge-atom
atom_id: ECON-ERR-007
atom_set: regression-error-covariance
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[误差序列相关]]"
  - "[[残差协方差诊断]]"
related:
  - "[[Breusch-Godfrey检验]]"
  - "[[自协方差与ACF]]"
leads_to:
  - "[[AR1准差分]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# Durbin–Watson 统计量主要诊断特定静态回归中的一阶误差序列相关
<!-- bilingual-en:start -->
*The Durbin–Watson statistic mainly diagnoses first-order error serial correlation in a restricted static regression setting*
<!-- bilingual-en:end -->

> [!summary] 数值先怎么读
> 对按时间排序的 OLS 残差 $e_t$，
> $$d=\frac{\sum_{t=2}^{T}(e_t-e_{t-1})^2}{\sum_{t=1}^{T}e_t^2}.$$
> $d$ 位于 0 与 4 之间，并近似满足 $d\approx2(1-r_1)$：接近 2 表示样本一阶残差相关接近 0，靠近 0/4 分别提示正/负一阶相关。
>
> <!-- bilingual-en:start -->
> For time-ordered OLS residuals, the statistic is the ratio of successive-difference variation to residual variation. It lies between zero and four and is approximately $2(1-r_1)$, so values near two correspond to little sample lag-one residual correlation.
> <!-- bilingual-en:end -->

这个近似只给直觉，不给完整判决。经典 DW 检验的有限样本分布依赖原回归的解释变量，因此要用下界 $d_L$、上界 $d_U$ 构造拒绝、不拒绝和**无法判定**区间；“没落入拒绝域”不能一律写成“证明无自相关”。双侧检验还要对靠近 4 的区域做对称处理。
<!-- bilingual-en:start -->
The approximation is intuition, not a complete decision rule. The classical finite-sample distribution depends on the regressors, giving lower and upper bounds with an inconclusive region. Merely avoiding a rejection region does not prove no serial correlation, and a two-sided test must also handle values near four.
<!-- bilingual-en:end -->

DW 的关键适用边界是：它针对一阶误差相关，且经典校准要求回归量满足严格外生等条件。模型含滞后因变量时，通常的 DW 表不再适用；要改用在该设定下有校准的 Durbin $h$/alternative 或[[Breusch-Godfrey检验]]。高阶、季节或一般动态相关也不能由一个 $d$ 值包办。
<!-- bilingual-en:start -->
DW targets lag-one error correlation and its classical calibration requires conditions such as strict exogeneity. The usual tables are invalid with a lagged dependent variable; use an appropriate Durbin alternative or Breusch–Godfrey test. A single $d$ statistic also cannot cover higher-order or seasonal dependence.
<!-- bilingual-en:end -->

最后，DW 检验的是**当前均值方程的残差**。显著结果可能意味着误差协方差确有 AR(1) 结构，也可能是趋势、季节或滞后项被漏掉。先解释产生相关的机制，再决定用 HAC、重设均值，还是估计 AR 误差模型。
<!-- bilingual-en:start -->
DW tests residuals from the current mean equation. Rejection may reflect an AR(1) covariance structure or omitted trend, seasonality, or dynamics. Diagnose the mechanism before choosing HAC, mean respecification, or an autoregressive-error model.
<!-- bilingual-en:end -->

> [!question]- 自检
> 含 $Y_{t-1}$ 的回归得到 $d=1.1$。能否直接查普通 DW 表并拒绝无自相关？
>
> **答案：** 不能。滞后因变量破坏普通 DW 的校准条件；应使用适合动态回归的检验，例如在条件满足时用 BG 或 Durbin alternative/h。

## 来源与核验

- [[02_Economy/01_Econometrics/08_自相关.md#4.2. 德宾–沃森 DW 检验|本地课程：DW]]：核对课程公式、范围和一阶相关直觉。
- [statsmodels `durbin_watson`](https://www.statsmodels.org/stable/generated/statsmodels.stats.stattools.durbin_watson.html)：核对统计量定义、$0$–$4$ 范围和 $2(1-r)$ 近似。
- [Stata, *Time-Series Reference Manual*, pp. 9–10](https://www.stata.com/manuals18/ts.pdf)：核对 DW 表、严格外生与更一般替代检验的边界。

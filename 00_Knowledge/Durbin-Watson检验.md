---
aliases:
  - Durbin-Watson test
  - DW test
student_os: knowledge-atom
atom_id: ECON-ERR-007
atom_set: regression-error-covariance
atom_type: diagnostic-test
status: source-checked
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

<!-- bilingual-en:start -->
# Durbin–Watson test
<!-- bilingual-en:end -->

*Durbin–Watson 检验*

<!-- bilingual-en:start -->
Suppose a regression predicts a monthly outcome. If it underpredicts this month, does it tend to underpredict next month too? **The Durbin–Watson (DW) statistic compares changes between neighbouring residuals with the overall size of the residuals.** Small changes suggest persistence; repeated reversals suggest negative [[误差序列相关|lag-one association]]. Turning that pattern into a test requires assumptions about the fitted model.
<!-- bilingual-en:end -->

假设回归预测每月结果：这个月预测偏低，下个月是否也容易偏低？**Durbin–Watson（DW）统计量把相邻残差的变化，与残差本身的大小作比较。** 变化较小提示持续性，反复翻转提示[[误差序列相关|一阶负关联]]；但要把图案变成正式检验，还需要拟合模型的条件。

> [!example] The same residual sizes, in a different order · 大小相同，顺序不同
> <!-- bilingual-en:start -->
> [[普通最小二乘|Ordinary least squares (OLS)]] chooses coefficients to minimise the sum of squared residuals. Fit an intercept alone to $(11,11,9,9)$. The fitted mean is $10$, so the [[回归残差|residuals]]—observed minus fitted values—are $(1,1,-1,-1)$. Their squared values sum to $4$; the successive changes $(0,-2,0)$ also have squared sum $4$. DW is therefore $4/4=1$.
>
> Reorder the observations to $(11,9,11,9)$. The residuals are now $(1,-1,1,-1)$: their squared sum stays $4$, but the changes have squared sum $12$, giving DW $3$. This illustrates what the statistic measures. Four observations are not a basis for a reliable general diagnostic conclusion, and observations must retain their actual time order.
> <!-- bilingual-en:end -->
>
> [[普通最小二乘|普通最小二乘法（OLS）]]选择让残差平方和最小的系数。对 $(11,11,9,9)$ 只拟合截距，拟合均值为 $10$，所以[[回归残差|残差]]（观测减拟合）是 $(1,1,-1,-1)$。残差平方和为 $4$，相邻变化 $(0,-2,0)$ 的平方和也是 $4$，DW 因而为 $1$。
>
> 若顺序变为 $(11,9,11,9)$，残差成为 $(1,-1,1,-1)$，平方和仍为 $4$，变化的平方和却为 $12$，DW 为 $3$。这说明统计量在测什么；四个观测不足以支持可靠的一般诊断结论，实际分析必须保持真实时间顺序。

<!-- bilingual-en:start -->
## The statistic and the endpoint correction
<!-- bilingual-en:end -->

*统计量与首尾修正*

<!-- bilingual-en:start -->
Take $T\ge2$ consecutive, equally spaced observations, with OLS residual $e_t$ at time $t$. Define $S=\sum_{t=1}^T e_t^2$ and require $S>0$. If every residual is zero, the ratio below is undefined. The definition is
<!-- bilingual-en:end -->

取 $T\ge2$ 个连续、等间隔观测，$e_t$ 是第 $t$ 期的 OLS 残差。记 $S=\sum_{t=1}^T e_t^2$，要求 $S>0$；所有残差都为零时，下式没有定义。统计量为

$$d=\frac{\sum_{t=2}^{T}(e_t-e_{t-1})^2}{S}.$$

<!-- bilingual-en:start -->
The numerator measures successive changes; the denominator sets the residual scale. Multiplying all residuals by the same nonzero number leaves $d$ unchanged. Squared-difference inequalities give $0\le d\le4$; these are bounds, not a claim that both endpoints occur in every fitted model.

To make the usual approximation precise, define $r_1=\sum_{t=2}^T e_te_{t-1}/S$. With an intercept, the residual mean is zero, so this is a common sample version of [[自协方差与ACF|autocorrelation]]. It is not the Pearson correlation of the two separately centred, shortened vectors. Expanding the numerator gives the exact identity
<!-- bilingual-en:end -->

分子衡量相邻变化，分母设定残差尺度。所有残差乘同一个非零数，$d$ 不变。由平方差不等式得到 $0\le d\le4$；这是界限，并不表示每个拟合模型都能取到两个端点。

为精确理解常见近似，定义 $r_1=\sum_{t=2}^T e_te_{t-1}/S$。含截距时残差均值为零，这是[[自协方差与ACF|样本自相关]]的一种常见口径，并不是把两个截短向量分别中心化后计算的 Pearson 相关系数。展开分子得到精确恒等式：

$$d=2(1-r_1)-\frac{e_1^2+e_T^2}{S}.$$

<!-- bilingual-en:start -->
Thus $d\approx2(1-r_1)$ only when the two endpoint squares are small relative to $S$. In the first example, $r_1=1/4$ and the correction is $1/2$, so $d=1$, not $1.5$. Even $d=2$ need not mean $r_1=0$: residuals $(1,-1,-1,1)$ give $d=2$ but $r_1=-1/4$. None of these sample quantities is the population error-correlation parameter itself.
<!-- bilingual-en:end -->

所以只有首尾两项平方相对 $S$ 足够小时，才有 $d\approx2(1-r_1)$。第一个例子中 $r_1=1/4$，修正项为 $1/2$，所以 $d=1$ 而非 $1.5$。即使 $d=2$，也不一定有 $r_1=0$：残差 $(1,-1,-1,1)$ 给出 $d=2$、$r_1=-1/4$。这些样本量都不是总体误差相关参数本身。

<!-- bilingual-en:start -->
## From a number to a test
<!-- bilingual-en:end -->

*从数值到检验*

<!-- bilingual-en:start -->
The classical question uses a correct linear mean $Y_t=X_t^T\beta+u_t$ and an AR(1) error alternative $u_t=\rho u_{t-1}+\varepsilon_t$: the previous error can carry a fraction $\rho$ into the next period, alongside a new shock $\varepsilon_t$. Here $X_t$ lists the regressors, $\beta$ their coefficients, and $T$ in the superscript denotes [[矩阵转置|transpose]]. The [[零假设与备择假设|null hypothesis]] is $H_0:\rho=0$; choose a positive, negative or two-sided alternative before inspecting the statistic.

One clear setting for the classical finite-sample null calibration is a fixed $T\times k$ design matrix $X$, with an intercept, [[满列秩与OLS唯一性|full column rank]] (no column is a combination of the others), $k<T$, and $u\sim N(0,\sigma^2I_T)$ with $\sigma^2>0$, where $I_T$ is the $T\times T$ identity matrix (ones on the diagonal, zeros elsewhere). That means independent, mean-zero [[正态分布|Gaussian]] errors of a common variance under the null. With random regressors, an appropriate sufficient formulation is the same distribution **conditional on the entire design** $X$. Merely having zero correlation between $X_t$ and $u_t$ at the same date does not give that conditional model.
<!-- bilingual-en:end -->

经典问题从正确线性均值 $Y_t=X_t^T\beta+u_t$ 出发，以 AR(1) 误差 $u_t=\rho u_{t-1}+\varepsilon_t$ 为备择：上期误差把其中比例 $\rho$ 延续到本期，再加新冲击 $\varepsilon_t$。$X_t$ 列出回归量，$\beta$ 是系数，上标 $T$ 表示[[矩阵转置]]。[[零假设与备择假设|零假设]]是 $H_0:\rho=0$；应事先选择正相关、负相关或双侧备择。

经典有限样本零假设校准的一种清楚设定是：固定的 $T\times k$ 设计矩阵 $X$ 含截距、[[满列秩与OLS唯一性|满列秩]]（没有列能由其余列组合而成）、$k<T$，并且 $u\sim N(0,\sigma^2I_T)$、$\sigma^2>0$，其中 $I_T$ 是 $T\times T$ 单位矩阵（对角元为一，其余为零）；即零假设下误差独立、均值零、[[正态分布|Gaussian]] 且共同方差。若回归量随机，一个相应充分表述是：**给定整个设计矩阵** $X$ 后，误差仍服从该分布。仅仅同一期 $X_t$ 与 $u_t$ 不相关，不足以得到这个条件模型。

<!-- bilingual-en:start -->
Fitting projects the error vector into a residual subspace. The null hypothesis does not in general force $d=2$; its null sampling law depends on $X$. An implementation can calculate design-specific tail probabilities; traditional tables instead give bounding critical values $d_L,d_U$. For a positive-correlation alternative at level $\alpha$, $d<d_L$ rejects, $d>d_U$ does not reject, and the interval between them is inconclusive by those bounds. Check the table's intercept and regressor-count convention.
<!-- bilingual-en:end -->

拟合把误差向量投到残差子空间。零假设一般不使 $d$ 恒等于 $2$，其抽样分布依赖 $X$。实现可以计算针对该设计的尾概率；传统表则提供临界值下界 $d_L$、上界 $d_U$。在水平 $\alpha$ 的正相关单侧检验中，$d<d_L$ 拒绝，$d>d_U$ 不拒绝，夹在二者之间则无法用这套界限判定。查表时要核对截距与回归量数量的计数口径。

> [!note] Inconclusive differs from nonrejection · 无法判定不等于不拒绝
> <!-- bilingual-en:start -->
> Davidson and MacKinnon give $d_L=1.335,d_U=1.771$ for $T=50$, six regressors including the intercept, and a 5% positive-correlation test. Values $1.2$, $1.5$ and $2.0$ therefore illustrate rejection, an inconclusive bounds result and nonrejection, respectively. Nonrejection still does not establish zero error correlation. For negative correlation, large $d$ is the relevant direction; two-sided testing must allocate the significance level across both tails and use the appropriate calibration, not reuse two separate 5% tests as a 5% test.
> <!-- bilingual-en:end -->
>
> Davidson 与 MacKinnon 给出的例子是：$T=50$、包括截距在内六个回归量、5% 正相关单侧检验，$d_L=1.335,d_U=1.771$。因此 $1.2$、$1.5$、$2.0$ 分别表示拒绝、界限法无法判定、不拒绝。不拒绝仍不证明误差相关为零。负相关应看较大的 $d$；双侧检验须向两尾分配显著性水平并采用相应校准，不能把两个各 5% 的单侧检验当作一个 5% 检验。

<!-- bilingual-en:start -->
## What DW cannot settle
<!-- bilingual-en:end -->

*DW 不能单独解决什么*

<!-- bilingual-en:start -->
A regressor such as $Y_{t-1}$ contains past errors. The usual DW tables are then generally invalid, even though the ratio can still be computed. A reported $d=1.1$ from such a model is not permission to use those tables. [[Breusch-Godfrey检验|Breusch–Godfrey]], Durbin's alternative or Durbin's $h$ require their own valid dynamic-model conditions; their names alone do not supply them.

DW focuses on lag one. A value near two cannot exclude higher-order, seasonal or nonlinear dependence, and missing dates should not silently be compressed into adjacent periods. Finally, [[残差协方差诊断|residuals are fitted quantities]]: rejection may reflect omitted trend, seasonality or mean dynamics rather than a correctly specified AR(1) error mechanism. Return to [[设定问题分层|the regression target and specification layer]] before choosing HAC inference, a revised mean, or an AR error model.
<!-- bilingual-en:end -->

$Y_{t-1}$ 这样的回归量包含过去误差。此时普通 DW 表一般失效，尽管比值仍能算；这类模型报告 $d=1.1$，并不授权直接查普通表。[[Breusch-Godfrey检验|Breusch–Godfrey]]、Durbin alternative 或 Durbin $h$ 都需要各自有效的动态模型条件，名称本身不提供这些条件。

DW 聚焦一阶；接近二不能排除高阶、季节或非线性依赖，也不能把缺失日期悄悄压缩成相邻期。最后，[[残差协方差诊断|残差是拟合生成量]]：拒绝可能反映遗漏趋势、季节或均值动态，并不证明真实误差机制恰为 AR(1)。选择 HAC 推断、修改均值或拟合 AR 误差模型前，应回到[[设定问题分层|回归目标与设定层次]]。

## 来源与核验

<!-- bilingual-en:start -->
- [Davidson and MacKinnon, *Estimation and Inference in Econometrics*, corrected p.362](https://econ.queensu.ca/pub/dm-book/seventh/pdf/pg362.pdf): the fixed-design Gaussian ratio-of-quadratic-forms distribution, design dependence, critical bounds and the $T=50$ example. The endpoint identity and four-observation examples above are direct calculations.
- [Stata, *regress postestimation time series*, pp.4–6,10](https://www.stata.com/manuals/rregresspostestimationtimeseries.pdf): classical DW assumptions, lagged-outcome limitations, alternatives and definition. The null hypothesis does not in general force the statistic to equal two.
- [statsmodels, `durbin_watson` implementation](https://www.statsmodels.org/stable/_modules/statsmodels/stats/stattools.html#durbin_watson): the successive-difference numerator and squared-residual denominator.
- [[02_Economy/01_Econometrics/08_自相关.md#4.2. 德宾–沃森 DW 检验|Permanent course section 4.2]]: local notation and position in the serial-correlation diagnostic sequence.
<!-- bilingual-en:end -->

- Davidson 与 MacKinnon 更正后的第 362 页核对固定设计 Gaussian 二次型比值、设计依赖、临界界限与 $T=50$ 例子；首尾恒等式及四观测例子直接计算。
- Stata 第 4–6、10 页核对经典条件、滞后因变量限制、替代检验及定义；零假设一般不使统计量恒等于二。
- statsmodels 原实现核对相邻差平方和分子、残差平方和分母。
- 永久课程 4.2 节保留本地符号与序列相关诊断顺序。

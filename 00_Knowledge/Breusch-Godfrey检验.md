---
aliases:
  - Breusch-Godfrey test
  - BG serial-correlation test
student_os: knowledge-atom
atom_id: ECON-ERR-008
atom_set: regression-error-covariance
atom_type: diagnostic-test
status: source-checked
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

<!-- bilingual-en:start -->
# Breusch–Godfrey test
<!-- bilingual-en:end -->

*Breusch–Godfrey 检验*

<!-- bilingual-en:start -->
After a regression has used today's predictors, can earlier prediction errors still help explain today's error? **The Breusch–Godfrey (BG) test asks whether a chosen group of lagged residuals adds information after retaining the original regressors.** It can examine several lags together, and its valid settings include many models containing lagged outcomes. This flexibility comes with a large-sample calibration, not an unconditional guarantee.
<!-- bilingual-en:end -->

回归已经用了今天的解释变量，过去的预测误差还能帮助解释今天的误差吗？**Breusch–Godfrey（BG）检验保留原回归量，再问预先选定的一组滞后残差是否有额外解释力。** 它可以同时检查多个滞后，在适当条件下也适用于含滞后因变量的许多模型；这种灵活性依赖大样本校准，不是无条件保证。

<!-- bilingual-en:start -->
## Keep the original regression in the question
<!-- bilingual-en:end -->

*检验问题要保留原回归*

<!-- bilingual-en:start -->
Suppose consecutive, equally spaced observations follow the proposed mean model $Y_t=X_t^T\beta+u_t$, $t=1,\ldots,T$. The vector $X_t$ contains $k$ regressors, including an intercept; $\beta$ contains their coefficients, and $u_t$ is the unobserved error. The superscript $T$ denotes [[矩阵转置|transpose]], so $X_t^T\beta$ sums the regressor–coefficient products. Assume the original design has [[满列秩与OLS唯一性|full column rank]], so no column is a combination of the others. [[普通最小二乘|Ordinary least squares (OLS)]] minimises the sum of squared fitting errors and produces [[回归残差|residuals]] $e_t=Y_t-X_t^T\hat\beta$, observed minus fitted values.

Choose a fixed positive integer lag order $p$ before examining results. A lag-one residual is yesterday's residual $e_{t-1}$; lag two is $e_{t-2}$. Fit an **auxiliary regression**, a second regression used to calculate the diagnostic:
<!-- bilingual-en:end -->

设连续、等间隔观测采用候选均值模型 $Y_t=X_t^T\beta+u_t$，$t=1,\ldots,T$。向量 $X_t$ 包含截距在内的 $k$ 个回归量，$\beta$ 是系数，$u_t$ 是未观测误差。上标 $T$ 表示[[矩阵转置]]，所以 $X_t^T\beta$ 把每个回归量乘其系数再加总。假定原设计[[满列秩与OLS唯一性|满列秩]]，即没有列能由其余列组合而成。[[普通最小二乘|普通最小二乘法（OLS）]]最小化拟合误差平方和，得到[[回归残差|残差]] $e_t=Y_t-X_t^T\hat\beta$，即观测减拟合。

看结果前，先选固定正整数滞后阶数 $p$。一阶残差就是上一期的 $e_{t-1}$，二阶是 $e_{t-2}$。再做**辅助回归**，即用于计算诊断的第二个回归：

$$e_t=X_t^T\delta+\rho_1e_{t-1}+\cdots+\rho_pe_{t-p}+v_t.$$

<!-- bilingual-en:start -->
Here $\delta$ allows the original regressors to remain in the auxiliary fit, $\rho_j$ measures the extra contribution of lag $j$, and $v_t$ is its remaining residual. The [[零假设与备择假设|joint null]] sets all lag coefficients to zero: $H_0:\rho_1=\cdots=\rho_p=0$. In the usual AR($p$) alternative for the underlying errors, this removes their lagged linear contributions. The test also has power against other serial-correlation patterns; rejection does not identify an AR($p$) law.

Retaining $X_t$ matters because a lagged residual can share variation with an original predictor. Dropping $X_t$ would ask a different question. In the original OLS sample, $X^Te=0$ is a [[样本正交与总体外生性|fitting identity]]; it does not establish error exogeneity or remain true automatically after dropping observations.
<!-- bilingual-en:end -->

这里 $\delta$ 让原回归量仍参与辅助拟合，$\rho_j$ 衡量第 $j$ 阶滞后的额外贡献，$v_t$ 是剩余残差。[[零假设与备择假设|联合零假设]]令全部滞后系数为零：$H_0:\rho_1=\cdots=\rho_p=0$。在原误差的常见 AR($p$) 备择中，这就去掉了误差的滞后线性贡献。检验也可能对其他序列相关模式有功效，因此拒绝不识别出一个 AR($p$) 定律。

保留 $X_t$ 很重要：滞后残差可能与原解释变量共享变化；删掉 $X_t$ 就换了问题。原 OLS 样本中的 $X^Te=0$ 是[[样本正交与总体外生性|拟合恒等式]]，不建立误差外生性，删去观测后也不会自动继续成立。

> [!example] A four-row auxiliary regression · 四行数据看清辅助回归
> <!-- bilingual-en:start -->
> An intercept-only fit to $(11,11,9,9)$ has residuals $e=(1,1,-1,-1)$. For $p=1$, one convention places a zero before the first residual and retains all four rows. The lag column is then $(0,1,1,-1)$; the auxiliary regression uses an intercept and that column.
>
> Direct least squares gives fitted residual $-1/11+(4/11)e_{t-1}$ and auxiliary $R^2=1/11$. With four auxiliary observations, the LM statistic is $4R^2=4/11$. This tiny calculation explains the recipe; its large-sample reference distribution is not a reliable four-observation conclusion.
> <!-- bilingual-en:end -->
>
> 对 $(11,11,9,9)$ 只拟合截距，残差为 $e=(1,1,-1,-1)$。令 $p=1$，一种口径在最前面补零、保留四行，于是滞后列为 $(0,1,1,-1)$，辅助回归使用截距和这列。
>
> 直接最小二乘得到拟合残差 $-1/11+(4/11)e_{t-1}$，辅助回归的 $R^2=1/11$。辅助观测有四个，故 LM 统计量为 $4R^2=4/11$。这只是说明计算步骤；大样本参考分布不足以支持可靠的四观测结论。

<!-- bilingual-en:start -->
## Sample size and the reference distribution
<!-- bilingual-en:end -->

*样本量与参考分布*

<!-- bilingual-en:start -->
The first $p$ lagged residuals are not all observed. **Zero padding** fills those initial missing lags with zeros and retains the original $T$ observations. **Truncation** uses only $t=p+1,\ldots,T$, giving $n=T-p$ when there are no other missing values. Neither convention says the unobserved true errors were zero. Record the convention, actual auxiliary size $n$, and included columns; do not join observations across missing dates as if they were consecutive.

The usual LM calculation is
<!-- bilingual-en:end -->

最前面 $p$ 期所需的滞后残差未被完整观察。**补零口径**把这些初始缺失滞后填零，保留原 $T$ 行；**截短口径**只用 $t=p+1,\ldots,T$，没有其他缺失时 $n=T-p$。两种做法都不声称未观测的真实误差等于零。要记录口径、实际辅助样本量 $n$ 与列，也不能把缺失日期前后观测直接拼成连续期。

常见 LM 计算是

$$LM=nR^2_{\mathrm{aux}}\ \overset{H_0,\ \mathrm{asymptotic}}{\sim}\ \chi_p^2.$$

<!-- bilingual-en:start -->
Here $R^2_{\mathrm{aux}}$ is the fraction of centred residual variation explained by the auxiliary fit, with its intercept. The [[卡方分布|chi-square distribution]] supplies a large-sample upper-tail reference, not an exact finite-sample law. This expression assumes the $p$ lag directions are identifiable in addition to the original regressors. Check the auxiliary rank; redundant lag columns do not justify blindly reporting $p$ independent restrictions.

In a common valid setting, the mean model is correct; the errors are mean-zero, have a positive constant conditional variance and adequate finite moments, and are innovations relative to the included predictors and past information under the null. The process and design must support [[估计量一致性|consistent OLS estimation]] (estimates approaching their target with more observations), the required laws of large numbers and central limit theorems, and nonsingular limiting regressor matrices, with $k$ and $p$ fixed as sample size grows. Suitable [[严平稳定义|stationary]] processes have joint distributions that do not change under a common shift in time; stable dynamic regressions with independent innovations provide familiar examples. Gaussian errors are not required for this asymptotic argument. Allowing lagged outcomes does not allow arbitrary endogeneity, unit roots or an incorrectly specified mean.
<!-- bilingual-en:end -->

这里 $R^2_{\mathrm{aux}}$ 是含截距辅助回归解释的、中心化残差变动比例。[[卡方分布]]提供大样本上尾参考，不是精确有限样本定律。该式还要求除了原回归量之外，$p$ 个滞后方向可识别。应检查辅助设计的秩；滞后列冗余时，不能照报 $p$ 个独立限制。

一种常见有效框架要求均值模型正确；零假设下，误差相对于所含解释变量和过去信息是均值零的新冲击，条件方差为正常数，并有适当有限矩。过程与设计还须支持 [[估计量一致性|OLS 一致估计]]（观测增多时估计趋近目标）、所需大数定律和中心极限定理、非奇异的极限回归量矩阵，而且样本增大时 $k,p$ 固定。[[严平稳定义|平稳]]过程的联合分布不随共同时间平移而改变；满足上述条件的稳定动态回归和独立新冲击是常见例子。渐近论证不要求误差 Gaussian；允许滞后因变量，不等于允许任意内生性、单位根或错误均值。

<!-- bilingual-en:start -->
With fixed $p$ and the required regularity, changing finitely many initial rows does not change the asymptotic target, but the finite-sample statistics can differ. In the four-row example, truncation leaves response $(1,-1,-1)$ and lag $(1,1,-1)$, giving $R^2=1/4$ and $LM=3/4$, rather than $4/11$. On a truncated sample, the original regressors may explain some residual variation themselves; $nR^2$ measures total auxiliary fit, so it is not exactly a finite-sample measure of only the lags' incremental fit. Zero padding preserves the original-sample orthogonality and is the default in the cited Stata and statsmodels implementations.
<!-- bilingual-en:end -->

$p$ 固定且正则条件成立时，改变有限个初始行不改变渐近目标，但有限样本统计量可能不同。四行例子截短后，响应为 $(1,-1,-1)$、滞后为 $(1,1,-1)$，得到 $R^2=1/4$、$LM=3/4$，而非 $4/11$。截短样本中，原回归量自身可能解释部分残差变化；$nR^2$ 衡量全部辅助拟合，所以有限样本下不精确等于“只有滞后的增量解释力”。补零保留原样本正交性，所引 Stata 与 statsmodels 实现默认采用该做法。

<!-- bilingual-en:start -->
Software may also report an F form. With full auxiliary rank $k+p$, compare the restricted and unrestricted fits **on the same auxiliary rows**:
<!-- bilingual-en:end -->

软件还可能报告 F 形式。当辅助设计满秩 $k+p$ 时，要在**同一组辅助行**上比较受限与不受限拟合：

$$F_{\mathrm{aux}}=\frac{(SSR_R-SSR_U)/p}{SSR_U/(n-k-p)}.$$

<!-- bilingual-en:start -->
$SSR_R$ is the residual sum of squares using only $X_t$; $SSR_U$ also includes the lag columns. The denominator requires $n>k+p$ and $SSR_U>0$. Because the response and lag columns are generated residuals, the label “F” does not make the usual [[F分布|$F_{p,n-k-p}$ reference distribution]] an exact small-sample theorem, even with Gaussian original errors. Rank deficiencies require a revised estimable test, not a mechanical degrees-of-freedom substitution.
<!-- bilingual-en:end -->

$SSR_R$ 是只用 $X_t$ 的残差平方和，$SSR_U$ 还包含滞后列；分母要求 $n>k+p$ 且 $SSR_U>0$。响应与滞后列都是生成残差，因此写成“F”不使常见 [[F分布|$F_{p,n-k-p}$ 参考分布]]校准变成精确小样本定理，即使原误差 Gaussian。秩亏时应重新建立可估的检验，而非机械换一个自由度。

> [!warning] A robust covariance option does not change $nR^2$ · 稳健协方差选项不会改变 $nR^2$
> <!-- bilingual-en:start -->
> Ordinary BG's chi-square calibration is not generally robust to [[异方差|conditional heteroskedasticity]]. Attaching robust standard errors to the auxiliary fit leaves its $R^2$, and hence $nR^2$, unchanged. A suitably justified heteroskedasticity-robust Wald/Durbin-alternative test uses a different covariance calculation for the lag coefficients; it must be specified and validated as such.
> <!-- bilingual-en:end -->
>
> 普通 BG 的卡方校准一般不能抵抗[[异方差|条件异方差]]。给辅助回归换稳健标准误，不改变其 $R^2$，所以也不改变 $nR^2$。经适当论证的异方差稳健 Wald／Durbin alternative 检验，要用另一套协方差计算检验滞后系数，必须按其自身方法说明和核验。

<!-- bilingual-en:start -->
## Interpret the chosen lag window
<!-- bilingual-en:end -->

*解释预先选定的滞后窗口*

<!-- bilingual-en:start -->
Quarterly data tested only at $p=1$ have not been checked for every seasonal pattern within a year. Choose $p$ from the scientific question, sampling frequency and plausible dynamics; if several orders are explored, report that search rather than selecting the smallest p-value and claiming a single prespecified test. Nonrejection does not prove independence or the absence of higher-order or nonlinear dependence.

Compared with [[Durbin-Watson检验|DW]], BG offers a useful dynamic-regression and multiple-lag framework, under its own conditions. A rejection still may reflect omitted seasonality, trend, breaks or lagged predictors. Use [[残差协方差诊断|residual diagnostics]] to investigate the mechanism and [[误差协方差决策|the covariance decision]] to distinguish valid HAC inference for a stated target from a change to the mean model. The test does not choose the remedy by itself.
<!-- bilingual-en:end -->

季度数据只检验 $p=1$，不等于检查了一年窗口内所有季节模式。应根据研究问题、采样频率与合理动态选择 $p$；若探索多个阶数，应报告这段搜索，不能只挑最小 p 值却声称做的是一个事先指定检验。不拒绝也不证明独立，或不存在高阶、非线性依赖。

相比 [[Durbin-Watson检验|DW]]，BG 在自身条件下提供动态回归与多滞后检验框架。拒绝仍可能来自遗漏季节、趋势、断点或滞后解释变量。用[[残差协方差诊断]]调查机制，再通过[[误差协方差决策]]分清针对指定目标的有效 HAC 推断与均值模型修改；检验不会替你选择补救方法。

## 来源与核验

<!-- bilingual-en:start -->
- [Stata, *regress postestimation time series*, pp.2–6,10–11](https://www.stata.com/manuals/rregresspostestimationtimeseries.pdf): auxiliary regression, BG versus Durbin's alternative, ordinary BG's heteroskedasticity boundary, initial zero padding/truncation, $nR^2$ and reported F calibration.
- [Davidson and MacKinnon, *Estimation and Inference in Econometrics*, corrected p.360](https://econ.queensu.ca/pub/dm-book/seventh/pdf/pg360.pdf): normality is unnecessary for the asymptotic procedure, lagged-outcome regressors require regularity, and generated residuals prevent a general exact finite-sample interpretation.
- [statsmodels, `acorr_breusch_godfrey` implementation](https://www.statsmodels.org/stable/_modules/statsmodels/stats/diagnostic.html#acorr_breusch_godfrey): the initial zeros, original regressors, actual auxiliary observation count and joint F restriction. The small examples and rank/SSR calculations above are direct least-squares calculations.
- [[02_Economy/01_Econometrics/08_自相关.md#4.3. 布罗施–戈弗雷 BG 检验|Permanent course section 4.3]]: the local sequence from original fit to residual lags and a joint restriction.
<!-- bilingual-en:end -->

- Stata 第 2–6、10–11 页核对辅助回归、BG 与 Durbin alternative 的区别、普通 BG 的异方差边界、补零／截短、$nR^2$ 和软件报告的 F 校准。
- Davidson 与 MacKinnon 更正后的第 360 页核对渐近过程不必正态、滞后因变量仍需正则条件，以及生成残差不能一般地作精确有限样本解释。
- statsmodels 原实现核对初始补零、保留原回归量、实际辅助观测数和 F 联合限制；小例及秩／SSR 关系直接最小二乘计算。
- 永久课程 4.3 节保留从原拟合到残差滞后与联合限制的本地顺序。

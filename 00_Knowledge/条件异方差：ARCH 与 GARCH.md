---
aliases:
  - "ARCH and GARCH"
  - "Volatility Models"
  - "条件异方差模型"
status: source-checked
---

# 条件异方差：ARCH 与 GARCH
<!-- bilingual-en:start -->
*Conditional heteroskedasticity: ARCH and GARCH*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 当收益均值难预测但大波动成群出现时，直接建模条件方差如何随过去信息变化。
> **具体锚点：** 金融收益可能几乎无自相关，但平方收益显著相关；这说明方向不可预测不等于风险恒定。
> **核心难点：** 条件方差必须非负且参数要满足稳定条件；标准化残差而非原残差用于检验剩余波动结构。
> **为什么重要：** 风险预测、VaR、期权与资产配置依赖随时间变化的波动率。
> **继续：** 先建好均值方程，再做 ARCH-LM；风险度量见 [[波动率、相关性与 Copula]] 和 [[VaR、ES、回测与压力测试]]。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** When returns have little predictable mean but large movements arrive in clusters, ARCH/GARCH models the conditional variance using past information.
> **Concrete anchor:** Financial returns may have almost no autocorrelation while squared returns are strongly autocorrelated. Unpredictable direction does not imply constant risk.
> **Central difficulty:** Conditional variance must remain nonnegative and parameters must satisfy stability conditions. Remaining volatility structure is diagnosed with standardised, not raw, residuals.
> **Why it matters:** Risk forecasts, VaR, options, and asset allocation depend on time-varying volatility.
> **Continue with:** Specify the conditional mean first, test for ARCH effects, then connect volatility forecasts to [[波动率、相关性与 Copula|volatility, dependence, and copulas]] and [[VaR、ES、回测与压力测试|VaR, ES, backtesting, and stress testing]].
<!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
> - Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
> - 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
> - [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
> - Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
<!-- bilingual-en:start -->
> [!source] Sources for this section
> - [[01_Math/06_时间序列分析/lecture.pdf|time-series lecture notes]] and [[01_Math/06_时间序列分析/lecture-dual.pdf|bilingual time-series lecture notes]] were checked for ARCH/GARCH definitions, diagnostics, estimation, forecasts, and extensions.
> - Engle (1982) and Bollerslev (1986) were checked for the ARCH and GARCH formulations and stationarity conditions.
> - Hull, *Risk Management and Financial Institutions*, and the [Basel Framework](https://www.bis.org/basel_framework/) were used only for downstream risk-measurement context.
<!-- bilingual-en:end -->

## 条件方差与波动聚集
<!-- bilingual-en:start -->
*Conditional variance and volatility clustering*
<!-- bilingual-en:end -->

写 $y_t=\mu_t+\varepsilon_t$，$\varepsilon_t=\sigma_t z_t$，其中 $E(z_t\mid\mathcal F_{t-1})=0$、$Var(z_t\mid\mathcal F_{t-1})=1$。$\sigma_t^2$ 是已知过去后的风险，不是无条件样本方差。波动聚集表现为 $|\varepsilon_t|$ 或 $\varepsilon_t^2$ 的持续性。
<!-- bilingual-en:start -->
Write $y_t=\mu_t+\varepsilon_t$ and $\varepsilon_t=\sigma_t z_t$, where $E(z_t\mid\mathcal F_{t-1})=0$ and $\operatorname{Var}(z_t\mid\mathcal F_{t-1})=1$. The quantity $\sigma_t^2$ is risk conditional on known history, not the unconditional sample variance. Volatility clustering appears as persistence in $|\varepsilon_t|$ or $\varepsilon_t^2$.
<!-- bilingual-en:end -->

因此要分清三个量：$\varepsilon_t$ 是实现的创新，$\sigma_t$ 是在 $t-1$ 时可预测的条件尺度，$z_t$ 是去除尺度后的标准化创新。GARCH 预测的是 $\sigma_{t+h}^2$ 的条件期望，不是未来收益绝对值的确定结果。
<!-- bilingual-en:start -->
Distinguish three quantities: $\varepsilon_t$ is the realised innovation, $\sigma_t$ is its predictable conditional scale at time $t-1$, and $z_t$ is the innovation after removing that scale. GARCH forecasts a conditional expectation of future $\sigma_{t+h}^2$, not a deterministic value of future absolute return.
<!-- bilingual-en:end -->

## ARCH 与 GARCH
<!-- bilingual-en:start -->
*ARCH and GARCH specifications*
<!-- bilingual-en:end -->

ARCH(q)：$\sigma_t^2=\omega+\sum_{i=1}^q\alpha_i\varepsilon_{t-i}^2$。GARCH(1,1) 再加 $\beta\sigma_{t-1}^2$，用较少参数捕捉持久性。常用充分约束为 $\omega>0,\alpha_i,\beta_j\ge0$；GARCH(1,1) 有有限无条件方差通常需 $\alpha+\beta<1$。
<!-- bilingual-en:start -->
ARCH($q$) specifies $\sigma_t^2=\omega+\sum_{i=1}^q\alpha_i\varepsilon_{t-i}^2$. GARCH(1,1) adds $\beta\sigma_{t-1}^2$ and captures persistence parsimoniously. Common sufficient constraints are $\omega>0$ and nonnegative $\alpha_i,\beta_j$; a GARCH(1,1) ordinarily requires $\alpha+\beta<1$ for a finite unconditional variance.
<!-- bilingual-en:end -->

对 GARCH(1,1)，取无条件期望得
$$
E(\sigma_t^2)=\omega+(\alpha+\beta)E(\sigma_t^2),
$$
所以长期方差 $\bar\sigma^2=\omega/(1-\alpha-\beta)$。$\alpha$ 衡量新的平方冲击反应，$\beta$ 衡量旧方差预测的延续，而 $\alpha+\beta$ 是波动冲击的总持久度。
<!-- bilingual-en:start -->
For GARCH(1,1), unconditional expectations give
$$
E(\sigma_t^2)=\omega+(\alpha+\beta)E(\sigma_t^2),
$$
so long-run variance is $\bar\sigma^2=\omega/(1-\alpha-\beta)$. Parameter $\alpha$ measures response to a new squared shock, $\beta$ carries forward the previous variance forecast, and $\alpha+\beta$ is overall volatility persistence.
<!-- bilingual-en:end -->

## 估计与创新分布
<!-- bilingual-en:start -->
*Estimation and innovation distributions*
<!-- bilingual-en:end -->

常用极大似然或准 MLE。金融收益厚尾时，Gaussian likelihood 的方差动态仍可作 QMLE，但区间和尾部风险常需 Student-t 等分布并做稳健推断。分布选择不应替代残差诊断。
<!-- bilingual-en:start -->
Estimation commonly uses maximum likelihood or quasi-maximum likelihood. With heavy-tailed financial returns, a Gaussian likelihood can still estimate variance dynamics as QMLE, but intervals and tail risk often require Student-t or another distribution together with robust inference. Distribution choice does not replace residual diagnostics.
<!-- bilingual-en:end -->

高斯条件对数似然的每期部分为
$$
\ell_t=-\frac12\left[\log(2\pi)+\log\sigma_t^2+\frac{\varepsilon_t^2}{\sigma_t^2}\right].
$$
该式同时惩罚预测方差过大的 $\log\sigma_t^2$ 和方差过小时的标准化平方误差。对 Student-t 分布，自由度必须与“方差已标准化为 1”的参数化口径一起说清。
<!-- bilingual-en:start -->
Under Gaussian conditional likelihood, each contribution is
$$
\ell_t=-\frac12\left[\log(2\pi)+\log\sigma_t^2+\frac{\varepsilon_t^2}{\sigma_t^2}\right].
$$
It penalises both an excessively large variance forecast through $\log\sigma_t^2$ and an excessively small one through the standardised squared error. With Student-t innovations, degrees of freedom must be stated together with the parameterisation that standardises variance to one.
<!-- bilingual-en:end -->

## ARCH-LM 与标准化残差
<!-- bilingual-en:start -->
*ARCH-LM and standardised residuals*
<!-- bilingual-en:end -->

先估计均值方程，检验残差平方对其滞后是否有解释力。拟合后检查 $z_t=\epsilon_t/\hat\sigma_t$ 及 $z_t^2$ 的相关、尾部和偏态；只有原残差去相关而平方仍相关，说明波动模型不足。
<!-- bilingual-en:start -->
Estimate the mean equation first and test whether lagged squared residuals explain current squared residuals. After fitting, inspect correlation, tails, and skewness of $z_t=\epsilon_t/\hat\sigma_t$ and $z_t^2$. If raw residuals are uncorrelated but their squares remain correlated, the volatility model is inadequate.
<!-- bilingual-en:end -->

ARCH-LM 通常回归 $\hat\varepsilon_t^2$ 于常数与 $q$ 个滞后平方，以 $TR^2$ 的渐近 $\chi_q^2$ 分布检验所有滞后系数为零。它检验“是否有某种 ARCH 信号”，并不直接决定 GARCH 阶数、创新分布或是否需不对称项。
<!-- bilingual-en:start -->
ARCH-LM typically regresses $\hat\varepsilon_t^2$ on a constant and $q$ lagged squares and tests all lag coefficients using the asymptotic $\chi_q^2$ distribution of $TR^2$. It detects some ARCH signal; it does not by itself select GARCH order, innovation distribution, or asymmetric terms.
<!-- bilingual-en:end -->

## 多步波动预测
<!-- bilingual-en:start -->
*Multi-step volatility forecasting*
<!-- bilingual-en:end -->

GARCH 预测递归地向长期方差回归；持久度越高，衰减越慢。接近 IGARCH 时长期方差估计极敏感，结构突变也可能伪装成高持久性。
<!-- bilingual-en:start -->
GARCH variance forecasts recursively revert toward long-run variance; greater persistence means slower decay. Near IGARCH, long-run variance estimates are extremely sensitive, and structural breaks can masquerade as high persistence.
<!-- bilingual-en:end -->

对 GARCH(1,1)，当 $h\ge2$ 时，
$$
E_t(\sigma_{t+h}^2)=\bar\sigma^2+(\alpha+\beta)^{h-1}
\left(E_t(\sigma_{t+1}^2)-\bar\sigma^2\right).
$$
这个式子直接显示 half-life 与 $\alpha+\beta$ 的关系。但若参数跨过样本断点不稳，“回到同一长期方差”的数学递归可以是错误外推。
<!-- bilingual-en:start -->
For GARCH(1,1), when $h\ge2$,
$$
E_t(\sigma_{t+h}^2)=\bar\sigma^2+(\alpha+\beta)^{h-1}
\left(E_t(\sigma_{t+1}^2)-\bar\sigma^2\right).
$$
This directly links shock half-life to $\alpha+\beta$. If parameters are unstable across a sample break, however, mathematical reversion toward one long-run variance may be a false extrapolation.
<!-- bilingual-en:end -->

## 扩展与边界
<!-- bilingual-en:start -->
*Extensions and boundaries*
<!-- bilingual-en:end -->

EGARCH、GJR-GARCH 可表示正负冲击不对称。模型描述条件二阶矩，不等于解释波动的经济原因。
<!-- bilingual-en:start -->
EGARCH and GJR-GARCH can represent asymmetric responses to positive and negative shocks. These models describe conditional second moments; they do not identify the economic causes of volatility.
<!-- bilingual-en:end -->

GJR-GARCH 通过 $\gamma\mathbf 1(\varepsilon_{t-1}<0)\varepsilon_{t-1}^2$ 让负冲击有额外方差影响；EGARCH 建模 $\log\sigma_t^2$，因而无需用全部非负系数保证方差为正。它们的非对称参数解释取决于精确符号约定，不能只记“负面新闻增大波动”而不写模型。
<!-- bilingual-en:start -->
GJR-GARCH adds $\gamma\mathbf 1(\varepsilon_{t-1}<0)\varepsilon_{t-1}^2$ so negative shocks can have an extra variance effect. EGARCH models $\log\sigma_t^2$, avoiding nonnegative coefficient restrictions solely to keep variance positive. Interpretation of asymmetry parameters depends on the exact sign convention, so the model equation must accompany any verbal claim.
<!-- bilingual-en:end -->

## Worked example：高持久 GARCH(1,1)
<!-- bilingual-en:start -->
*Worked example: a persistent GARCH(1,1)*
<!-- bilingual-en:end -->

设 $\omega=0.02$、$\alpha=0.08$、$\beta=0.90$，则 $\alpha+\beta=0.98$，长期方差为 $0.02/(1-0.98)=1$。若一步条件方差为 4，则 $h$ 步方差预测为 $1+0.98^{h-1}(4-1)$；10 步后仍约为 3.50，显示大冲击衰减很慢。这个例子也提醒：小幅修改 $0.98$ 会大幅改变长期方差与半衰期。
<!-- bilingual-en:start -->
Let $\omega=0.02$, $\alpha=0.08$, and $\beta=0.90$. Then $\alpha+\beta=0.98$ and long-run variance is $0.02/(1-0.98)=1$. If the one-step conditional variance is four, the $h$-step forecast is $1+0.98^{h-1}(4-1)$; after ten steps it is still about 3.50, showing slow shock decay. A small change in 0.98 also changes long-run variance and half-life substantially.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 收益无自相关，为什么仍可能需要 GARCH？
<!-- bilingual-en:start -->
*Why might GARCH still be needed when returns have no autocorrelation?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 均值可近似不可预测，但平方或绝对收益可能相关，说明条件方差随时间变化。
<!-- bilingual-en:start -->
> [!answer]- Answer
> The conditional mean may be nearly unpredictable while squared or absolute returns remain dependent, showing that conditional variance changes over time.
<!-- bilingual-en:end -->

### GARCH(1,1) 中 $\alpha+\beta$ 接近 1 表示什么？
<!-- bilingual-en:start -->
*What does $\alpha+\beta$ near one mean in a GARCH(1,1)?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 波动冲击衰减很慢、条件方差高度持久，长期方差估计也更敏感。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Volatility shocks decay slowly, conditional variance is highly persistent, and the long-run variance estimate becomes sensitive.
<!-- bilingual-en:end -->

### 拟合后为什么检查标准化残差平方？
<!-- bilingual-en:start -->
*Why inspect squared standardised residuals after fitting?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它剔除了模型预测的时变尺度；若仍有相关，说明条件方差动态没有被充分吸收。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Standardisation removes the model's predicted time-varying scale. Remaining dependence in the squares means conditional-variance dynamics were not fully absorbed.
<!-- bilingual-en:end -->

### 用自己的话解释为什么高估 $\beta$ 可能是结构突变的假象。
<!-- bilingual-en:start -->
*Explain in your own words why a large estimate of $\beta$ may be an artefact of a structural break.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 若样本中方差水平永久换档，固定参数 GARCH 只能用缓慢衰减追赶新水平，于是把制度变化误记为单次冲击的长持续。
<!-- bilingual-en:start -->
> [!answer]- Answer
> If variance shifts permanently within the sample, a fixed-parameter GARCH can only approach the new level through slow decay, misreading a regime change as persistence of one shock.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
- Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
- 本库金融机构与风险管理课程笔记：支持课程范围、课堂例题和记号。
- [Basel Framework](https://www.bis.org/basel_framework/)：核验资本、市场风险、信用风险、CVA 与监管口径。
- Hull, *Risk Management and Financial Institutions*：交叉核验 VaR、ES、Greek、利率风险、信用风险与模拟方法。
<!-- bilingual-en:start -->
- [[01_Math/06_时间序列分析/lecture.pdf|time-series lecture notes]] and [[01_Math/06_时间序列分析/lecture-dual.pdf|bilingual time-series lecture notes]] were checked section by section for ARCH/GARCH specifications, ARCH-LM, maximum likelihood, variance forecasts, IGARCH, and asymmetric extensions.
- Engle (1982), "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation," was checked for ARCH and its LM test.
- Bollerslev (1986), "Generalized Autoregressive Conditional Heteroskedasticity," was checked for GARCH and covariance-stationarity conditions.
- Hull, *Risk Management and Financial Institutions*, and the [Basel Framework](https://www.bis.org/basel_framework/) were used only for downstream risk-measurement context.
<!-- bilingual-en:end -->

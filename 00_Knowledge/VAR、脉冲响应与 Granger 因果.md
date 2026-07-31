---
aliases:
  - "VAR Model"
  - "Vector Autoregression"
  - "Impulse Response"
  - "Impulse Response Function"
  - "Granger Causality"
  - "Granger Causality Test"
  - "Dynamic Regression and VAR"
status: source-checked
---

# VAR、脉冲响应与 Granger 因果
<!-- bilingual-en:start -->
*VAR, impulse responses, and Granger causality*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 描述多个时间序列彼此滞后影响，并把一次冲击如何在系统中传播分解出来。
> **具体锚点：** 利率、通胀和产出相互影响；单方程很难预先指定谁完全外生，VAR 让每个变量都由系统过去共同解释。
> **核心难点：** reduced-form 残差通常同期相关，未经识别的“冲击”没有结构经济含义；Granger 因果只是增量预测关系。
> **为什么重要：** 它连接政策动态、预测、干预分析和结构识别。
> **继续：** 先确保平稳/协整处理，再解释 IRF；长期关系见 [[协整与误差修正模型]]。
> <!-- bilingual-en:start -->
> **What it solves:** VAR describes lagged interactions among several time series and traces how an innovation propagates through the system.
> **Concrete anchor:** Interest rates, inflation, and output influence one another. Rather than declaring one variable fully exogenous, a VAR lets every variable depend on the system's past.
> **Central difficulty:** Reduced-form residuals are usually contemporaneously correlated, so an unidentified "shock" has no structural economic meaning. Granger causality is only incremental predictive content.
> **Why it matters:** VAR connects policy dynamics, forecasting, intervention analysis, and structural identification.
> **Continue with:** Establish stationarity or cointegration before interpreting impulse responses; see [[协整与误差修正模型|cointegration and error correction]] for long-run restrictions.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
> - Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
> <!-- bilingual-en:start -->
> - [[01_Math/06_时间序列分析/lecture.pdf|time-series lecture notes]] and [[01_Math/06_时间序列分析/lecture-dual.pdf|bilingual time-series lecture notes]] were checked for reduced and structural VARs, stability, estimation, identification, IRFs, FEVD, and Granger tests.
> - Lütkepohl, *New Introduction to Multiple Time Series Analysis*, was checked for VAR representation, stability, forecasting, and structural analysis.
> <!-- bilingual-en:end -->

## VAR
<!-- bilingual-en:start -->
*Vector autoregression*
<!-- bilingual-en:end -->

VAR(p) 写作 $y_t=c+A_1y_{t-1}+\cdots+A_py_{t-p}+u_t$。每个方程可用 OLS 估计，但系统稳定性由 companion matrix 特征值决定。阶数选择结合信息准则、残差和经济用途，参数随维度和阶数快速增长。
<!-- bilingual-en:start -->
A VAR($p$) is $y_t=c+A_1y_{t-1}+\cdots+A_py_{t-p}+u_t$. Each equation can be estimated by OLS, but system stability is determined by eigenvalues of the companion matrix. Lag order combines information criteria, residuals, and economic purpose, while parameter count grows rapidly with dimension and order.
<!-- bilingual-en:end -->

对 $K$ 个变量、$p$ 阶、含截距的系统，每个方程有 $1+Kp$ 个系数，全系统有 $K+K^2p$。所以将变量数翻倍会让动态系数约增长四倍；大 VAR 需正则化、Bayesian shrinkage 或有理由的变量精简，不能只依赖自动阶数选择。
<!-- bilingual-en:start -->
With $K$ variables, order $p$, and an intercept, each equation has $1+Kp$ coefficients and the system has $K+K^2p$. Doubling the number of variables therefore roughly quadruples dynamic coefficients. Large VARs require regularisation, Bayesian shrinkage, or justified variable reduction rather than lag-order automation alone.
<!-- bilingual-en:end -->

## 稳定性、VMA 表示与预测
<!-- bilingual-en:start -->
*Stability, VMA representation, and forecasting*
<!-- bilingual-en:end -->

若 companion matrix 的所有特征值模小于 1，则 VAR 稳定并可写成
$$
y_t=\mu+\sum_{j=0}^{\infty}\Phi_j u_{t-j},\qquad \Phi_0=I.
$$
$\Phi_j$ 描述 reduced-form innovation 在 $j$ 期后的传播。预测可用 VAR 方程递归，也可用 VMA 将预测误差写成未来 innovations 的线性组合。
<!-- bilingual-en:start -->
If every companion-matrix eigenvalue has modulus below one, the VAR is stable and has representation
$$
y_t=\mu+\sum_{j=0}^{\infty}\Phi_j u_{t-j},\qquad \Phi_0=I.
$$
The matrix $\Phi_j$ traces propagation of a reduced-form innovation after $j$ periods. Forecasts follow recursively from the VAR, while the VMA expresses forecast errors as linear combinations of future innovations.
<!-- bilingual-en:end -->

稳定性是解释长期衰减的前提，不是估计 OLS 系数前必须强行的“数据清洗”。如果变量有单位根且协整，正确表示是 VECM；机械差分可以让根看似稳定，却丢掉长期约束。
<!-- bilingual-en:start -->
Stability is a prerequisite for interpreting long-run decay, not a data-cleaning operation to force before OLS. If variables have unit roots and are cointegrated, the correct representation is a VECM; mechanical differencing can produce stable-looking roots while discarding long-run restrictions.
<!-- bilingual-en:end -->

## reduced-form 与结构识别
<!-- bilingual-en:start -->
*Reduced form and structural identification*
<!-- bilingual-en:end -->

结构系统可写 $B_0y_t=d+B_1y_{t-1}+\varepsilon_t$，乘 $B_0^{-1}$ 后得 reduced form $y_t=c+A_1y_{t-1}+u_t$，其中 $u_t=B_0^{-1}\varepsilon_t$。即使结构冲击 $\varepsilon_t$ 彼此正交，reduced residuals $u_t$ 也通常同期相关。从 $\Sigma_u$ 分解出 $B_0$ 需要数据之外的限制。
<!-- bilingual-en:start -->
A structural system can be written $B_0y_t=d+B_1y_{t-1}+\varepsilon_t$. Multiplying by $B_0^{-1}$ yields the reduced form $y_t=c+A_1y_{t-1}+u_t$, where $u_t=B_0^{-1}\varepsilon_t$. Even if structural shocks $\varepsilon_t$ are orthogonal, reduced-form residuals $u_t$ are usually contemporaneously correlated. Recovering $B_0$ from $\Sigma_u$ needs restrictions beyond the data.
<!-- bilingual-en:end -->

常见识别包括 Cholesky 递归排序、同期零限制、长期限制、符号限制与外部工具。它们不是不同计算技巧而是不同结构假设。应报告限制的经济理由，并在可能时检查合理替代识别下的稳健性。
<!-- bilingual-en:start -->
Common identification schemes include recursive Cholesky ordering, contemporaneous zero restrictions, long-run restrictions, sign restrictions, and external instruments. They are different structural assumptions, not merely different computational techniques. Report their economic rationale and, where possible, robustness to defensible alternatives.
<!-- bilingual-en:end -->

## 脉冲响应
<!-- bilingual-en:start -->
*Impulse responses*
<!-- bilingual-en:end -->

MA 表示把当前和过去创新映射到未来 y。若 reduced-form 创新同期相关，要通过 Cholesky 排序、短期/长期限制、符号限制或外部工具识别结构冲击。不同识别产生不同 IRF，必须报告假设。
<!-- bilingual-en:start -->
The moving-average representation maps current and past innovations into future $y$. If reduced-form innovations are contemporaneously correlated, structural shocks require identification through Cholesky ordering, short- or long-run restrictions, sign restrictions, or external instruments. Different identification choices produce different IRFs and must be reported.
<!-- bilingual-en:end -->

若 $u_t=P\varepsilon_t$ 且 $E(\varepsilon_t\varepsilon_t^T)=I$，则结构 VMA 为 $y_t=\mu+\sum_j\Phi_jP\varepsilon_{t-j}$，第 $k$ 个结构冲击对所有变量在 horizon $j$ 的响应是 $\Phi_jP$ 的第 $k$ 列。IRF 图应同时标出冲击尺度、变量单位、累积与否、区间构造方法和识别。
<!-- bilingual-en:start -->
If $u_t=P\varepsilon_t$ and $E(\varepsilon_t\varepsilon_t^T)=I$, the structural VMA is $y_t=\mu+\sum_j\Phi_jP\varepsilon_{t-j}$. The response of all variables at horizon $j$ to structural shock $k$ is column $k$ of $\Phi_jP$. An IRF figure should state shock scale, variable units, whether responses are cumulative, the interval method, and identification scheme.
<!-- bilingual-en:end -->

## 预测误差方差分解
<!-- bilingual-en:start -->
*Forecast-error variance decomposition*
<!-- bilingual-en:end -->

FEVD 把 h 步预测误差方差归因于各结构冲击，依赖同一识别和尺度。它不是现实世界因果贡献的无条件真值。
<!-- bilingual-en:start -->
FEVD attributes an $h$-step forecast-error variance to structural shocks under the same identification and scaling. It is not an unconditional truth about causal contributions in the real world.
<!-- bilingual-en:end -->

对第 $i$ 个变量，冲击 $k$ 在 $h$ 步 FEVD 中的分子是 $\sum_{j=0}^{h-1}(e_i^T\Phi_jPe_k)^2$，分母是对所有冲击求和。因此它回答“在这个模型与识别下，未来不确定性由哪些冲击解释”，而不是“历史水平中多少是某政策造成”。
<!-- bilingual-en:start -->
For variable $i$, the numerator of shock $k$'s $h$-step FEVD is $\sum_{j=0}^{h-1}(e_i^T\Phi_jPe_k)^2$, with the denominator summed over all shocks. It asks which identified shocks explain future uncertainty under this model, not how much of the historical level was caused by a policy.
<!-- bilingual-en:end -->

## Granger 因果
<!-- bilingual-en:start -->
*Granger causality*
<!-- bilingual-en:end -->

若在控制系统过去后，x 的滞后提高 y 的预测，称 x Granger-causes y。检验是对 x 滞后系数的联合限制。它不排除遗漏共同驱动、同期因果或纯信息领先，因此不能简写为结构因果。
<!-- bilingual-en:start -->
If lags of $x$ improve forecasts of $y$ after controlling for the system's past, $x$ is said to Granger-cause $y$. The test is a joint restriction on lagged $x$ coefficients. It does not rule out omitted common drivers, contemporaneous causality, or mere informational leadership, so it cannot be abbreviated to structural causality.
<!-- bilingual-en:end -->

检验结果取决于信息集、滞后阶数、变量变换和样本窗。“x 不 Granger-cause y”只表示在当前信息集与模型中，x 的所选滞后没有额外线性预测力，不是说 x 没有同期、非线性或结构效应。
<!-- bilingual-en:start -->
The result depends on information set, lag order, transformations, and sample window. "No Granger causality from $x$ to $y$" means only that the chosen lags of $x$ add no linear predictive content in this model; it does not rule out contemporaneous, nonlinear, or structural effects.
<!-- bilingual-en:end -->

## 非平稳与协整
<!-- bilingual-en:start -->
*Nonstationarity and cointegration*
<!-- bilingual-en:end -->

对有单位根且协整的变量直接差分 VAR 会丢失长期关系，水平 VAR 的常规推断又可能失效；VECM 显式结合误差修正与短期动态。
<!-- bilingual-en:start -->
For unit-root variables that are cointegrated, a VAR in first differences discards the long-run relation, while conventional inference in a level VAR may fail. A VECM explicitly combines error correction with short-run dynamics.
<!-- bilingual-en:end -->

## Worked example：递归识别的两变量 VAR
<!-- bilingual-en:start -->
*Worked example: a recursively identified two-variable VAR*
<!-- bilingual-en:end -->

设 $y_t=(\text{产出增长},\text{政策利率})^T$，Cholesky 排序先产出、后利率意味着在当期内利率可响应产出创新，而产出不响应利率结构冲击。若把排序反转，当期零限制也反转，IRF 会改变。所以“一个标准差利率冲击后产出的路径”不是 reduced-form VAR 自动给出，而是数据与递归假设的联合结果。
<!-- bilingual-en:start -->
Let $y_t=(\text{output growth},\text{policy rate})^T$. A Cholesky ordering with output first and the rate second says that the rate may respond contemporaneously to an output innovation, while output does not respond within the period to a structural rate shock. Reversing the ordering reverses that zero restriction and changes the IRF. The path of output after a one-standard-deviation rate shock therefore comes from data plus recursive assumptions, not from the reduced-form VAR alone.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 为什么 reduced-form VAR 残差不能直接叫经济结构冲击？
<!-- bilingual-en:start -->
*Why cannot reduced-form VAR residuals be called economic structural shocks directly?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不同方程残差可同期相关，只表示无法由过去预测的组合；需要额外识别限制分解为有含义的正交冲击。
> <!-- bilingual-en:start -->
> Residuals across equations can be contemporaneously correlated and represent only combinations unpredictable from the past. Additional restrictions are needed to decompose them into meaningful orthogonal shocks.
> <!-- bilingual-en:end -->

### Granger 因果能否证明政策变量造成结果变化？
<!-- bilingual-en:start -->
*Can Granger causality prove that a policy variable caused an outcome?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不能。它只说明滞后信息提高预测，仍可能由遗漏变量、预期或共同冲击产生。
> <!-- bilingual-en:start -->
> No. It says only that lagged information improves prediction; omitted variables, expectations, or common shocks may still generate the relation.
> <!-- bilingual-en:end -->

### IRF 为什么要报告变量排序或识别方法？
<!-- bilingual-en:start -->
*Why must an IRF report variable ordering or another identification method?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 冲击正交化不是数据唯一决定的；不同限制会改变冲击定义和传播路径。
> <!-- bilingual-en:start -->
> Orthogonalisation is not uniquely determined by the data; different restrictions change both shock definitions and propagation paths.
> <!-- bilingual-en:end -->

### 用自己的话区分 IRF 与 FEVD 回答的问题。
<!-- bilingual-en:start -->
*Distinguish in your own words the questions answered by an IRF and an FEVD.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> IRF 追踪给定结构冲击对变量的时间路径；FEVD 衡量在给定 horizon 的预测误差不确定性中，各结构冲击占多少。
> <!-- bilingual-en:start -->
> An IRF traces the time path after one identified shock; an FEVD measures how much each identified shock contributes to forecast-error uncertainty at a given horizon.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
- Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
<!-- bilingual-en:start -->
- [[01_Math/06_时间序列分析/lecture.pdf|time-series lecture notes]] and [[01_Math/06_时间序列分析/lecture-dual.pdf|bilingual time-series lecture notes]] were checked section by section for reduced and structural VARs, stability, separate-equation OLS, identification, IRFs, FEVD, Granger tests, and the cointegration boundary.
- Lütkepohl, *New Introduction to Multiple Time Series Analysis*, was checked for VAR representation, stability, forecasting, and structural analysis.
<!-- bilingual-en:end -->

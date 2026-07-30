---
aliases:
  - "Exponential Smoothing"
  - "ETS"
  - "Holt-Winters"
  - "指数平滑"
status: source-checked
---

# 指数平滑与 ETS
<!-- bilingual-en:start -->
*Exponential smoothing and ETS*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 根据序列的水平、趋势、季节、相关和外生驱动选择可解释预测方法，并组合不确定性。
> **具体锚点：** 零售月销量有趋势和年度季节性，ETS 直接更新这些成分；ARIMA 则用差分与自相关建模，两者不是谁完全包含谁。
> **核心难点：** 预测解释变量本身未知时，回归预测必须给其未来值/情景；自动选模仍要诊断残差和结构变化。
> **为什么重要：** 不同方法利用不同可预测结构，基准和时序验证决定是否值得使用。
> **继续：** 先分解结构，再并行比较 ETS、ARIMA 和 dynamic regression；选择依据是外样本与用途。
<!-- bilingual-en:start -->
> [!summary] Quick recovery
> **What it solves:** ETS forecasts a series by recursively updating level, trend, and seasonal states, with a statistical error model that supplies likelihoods and intervals.
> **Concrete anchor:** Monthly retail sales can have a changing level, a damped trend, and annual seasonality; ETS updates those components directly as each observation arrives.
> **Central difficulty:** Additive versus multiplicative components, damping, and initial states change both forecast shape and uncertainty. Automatic selection still requires residual and out-of-sample checks.
> **Why it matters:** ETS offers an interpretable baseline for many operational series and provides structures that ARIMA does not fully contain.
> **Continue with:** Compare it against [[ARMA 模型：识别、估计、诊断与预测|ARIMA-family models]] and [[回归预测与动态回归|dynamic regression]] under the evaluation design in [[预测问题、基准方法与评估|forecasting workflow and evaluation]].
<!-- bilingual-en:end -->

> [!source] 本节依据
> - Hyndman & Athanasopoulos, [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)：支持预测流程、基准、评估、ETS、回归与 ARIMA。
> - [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
> - Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
<!-- bilingual-en:start -->
> [!source] Sources for this section
> - Hyndman and Athanasopoulos, [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/), was checked for decomposition, simple exponential smoothing, Holt and damped trend, Holt–Winters, innovations state-space ETS, and forecast intervals.
> - [[01_Math/06_时间序列分析/lecture.pdf|time-series lecture notes]] and [[01_Math/06_时间序列分析/lecture-dual.pdf|bilingual time-series lecture notes]] were checked for course notation and comparisons with ARIMA.
<!-- bilingual-en:end -->

## 分解与季节性
<!-- bilingual-en:start -->
*Decomposition and seasonality*
<!-- bilingual-en:end -->

加法分解适合季节振幅近恒定，乘法/对数适合振幅随水平增长。经典分解用于理解，STL 更灵活稳健。分解后的 seasonal adjustment 仍可能含日历、异常和修订影响。
<!-- bilingual-en:start -->
Additive decomposition suits seasonal amplitude that is roughly constant, while multiplicative decomposition or a logarithm suits amplitude that grows with the level. Classical decomposition aids understanding, whereas STL is more flexible and robust. Seasonal adjustment can still contain calendar effects, outliers, and revision effects.
<!-- bilingual-en:end -->

分解是诊断与表示，不自动等于预测模型。STL 产生趋势、季节和 remainder，但要外推还必须为去季节序列选择预测方法，并将季节成分恢复到未来。
<!-- bilingual-en:start -->
Decomposition is a diagnostic and representation, not automatically a forecasting model. STL yields trend, seasonal, and remainder components, but extrapolation still requires a forecasting method for the seasonally adjusted series and restoration of future seasonal components.
<!-- bilingual-en:end -->

## 指数平滑与 ETS
<!-- bilingual-en:start -->
*Exponential smoothing and ETS*
<!-- bilingual-en:end -->

### 简单指数平滑
<!-- bilingual-en:start -->
*Simple exponential smoothing*
<!-- bilingual-en:end -->

simple exponential smoothing 更新水平；Holt 加趋势，damped trend 让长期趋势逐渐变平；Holt–Winters 加季节。ETS 把 error、trend、seasonal 组合成统计模型并生成区间。平滑参数控制新旧信息权重。
<!-- bilingual-en:start -->
Simple exponential smoothing updates a level; Holt adds a trend, a damped trend flattens long-run extrapolation, and Holt–Winters adds seasonality. ETS combines error, trend, and seasonal choices into a statistical model that produces intervals. Smoothing parameters control the weights on new and old information.
<!-- bilingual-en:end -->

无趋势、无季节时，
$$
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1},\qquad
\hat y_{t+h|t}=\ell_t.
$$
展开递推得到对历史观测的几何权重 $\alpha(1-\alpha)^j$。$\alpha$ 大时水平追踪新数据快，但也更易跟随噪声；$\alpha$ 小时更平滑但对转折反应慢。
<!-- bilingual-en:start -->
With neither trend nor seasonality,
$$
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1},\qquad
\hat y_{t+h|t}=\ell_t.
$$
Expanding the recursion gives geometric weights $\alpha(1-\alpha)^j$ on past observations. A large $\alpha$ follows new data quickly but is more sensitive to noise; a small $\alpha$ is smoother but reacts slowly to changes.
<!-- bilingual-en:end -->

## 趋势、阻尼与季节
<!-- bilingual-en:start -->
*Trend, damping, and seasonality*
<!-- bilingual-en:end -->

Holt 线性趋势用水平 $\ell_t$ 与斜率 $b_t$ 给出 $\hat y_{t+h|t}=\ell_t+hb_t$。长期将最近斜率无限延伸往往太强，阻尼参数 $0<\phi<1$ 把趋势项改为 $(\phi+\cdots+\phi^h)b_t$，使长期预测逐渐趋平。
<!-- bilingual-en:start -->
Holt's linear trend uses level $\ell_t$ and slope $b_t$ to produce $\hat y_{t+h|t}=\ell_t+hb_t$. Extrapolating the latest slope indefinitely is often too strong. A damping parameter $0<\phi<1$ replaces the trend term by $(\phi+\cdots+\phi^h)b_t$, causing long-run forecasts to flatten gradually.
<!-- bilingual-en:end -->

Holt–Winters 的加法季节适合季节差值近似恒定，乘法季节适合季节比率近似恒定且数据为正。乘法误差/季节在零值或负值时可能无定义，不能只因为季节振幅大就机械选它。
<!-- bilingual-en:start -->
Additive Holt–Winters seasonality suits approximately constant seasonal differences, whereas multiplicative seasonality suits approximately constant seasonal ratios with positive data. Multiplicative error or seasonality may be undefined with zeros or negative values and should not be chosen mechanically merely because seasonal amplitude is large.
<!-- bilingual-en:end -->

## ETS 状态空间模型
<!-- bilingual-en:start -->
*ETS state-space models*
<!-- bilingual-en:end -->

ETS 名称中 E/T/S 分别指 error、trend、seasonal，例如 ETS(A,Ad,A) 是加法误差、加法阻尼趋势和加法季节。把平滑递推放入 innovations state-space 模型后，可用似然估计初始状态与参数，用 AICc 等比较候选，并从模型生成预测分布。
<!-- bilingual-en:start -->
In ETS notation, E, T, and S denote error, trend, and seasonal forms. For example, ETS(A,Ad,A) has additive errors, an additive damped trend, and additive seasonality. Embedding smoothing recursions in an innovations state-space model allows likelihood estimation of initial states and parameters, comparison via criteria such as AICc, and model-based predictive distributions.
<!-- bilingual-en:end -->

“加法方法”与“加法误差”不是同一选择：前者描述状态成分如何组合，后者描述随机创新的尺度如何进入。所以预测点路径相同的两个平滑方法，也可因误差模型不同而有不同区间和似然。
<!-- bilingual-en:start -->
An additive method and additive errors are distinct choices: the former describes how state components combine, while the latter describes how random innovations enter the scale. Two smoothing methods with the same point-forecast path can therefore have different likelihoods and intervals because their error models differ.
<!-- bilingual-en:end -->

## ETS 与 ARIMA
<!-- bilingual-en:start -->
*ETS versus ARIMA*
<!-- bilingual-en:end -->

ETS 以成分演化描述趋势/季节，ARIMA 以差分后自相关描述；部分线性模型等价但各自也有对方没有的结构。用 rolling-origin 多 horizon 表现而非神话式优劣选择。
<!-- bilingual-en:start -->
ETS describes evolving trend and seasonal components, whereas ARIMA describes autocorrelation after differencing. Some linear models are equivalent, but each family also contains structures absent from the other. Choose by rolling-origin performance across relevant horizons, not by a universal ranking.
<!-- bilingual-en:end -->

例如 simple exponential smoothing 对应特定 ARIMA(0,1,1) 表示，Holt 线性法与某些 ARIMA(0,2,2) 有对应；但带乘法季节或非线性状态更新的 ETS 不是普通线性 ARIMA。这些等价是理解模型的桥，不是说两个家族可以合并成一张卡。
<!-- bilingual-en:start -->
For example, simple exponential smoothing corresponds to a particular ARIMA(0,1,1) representation, and Holt's linear method has links to some ARIMA(0,2,2) models. ETS models with multiplicative seasonality or nonlinear state updates are not ordinary linear ARIMA models. These equivalences are conceptual bridges, not evidence that the two families should be collapsed into one card.
<!-- bilingual-en:end -->

## Worked example：月度零售销量
<!-- bilingual-en:start -->
*Worked example: monthly retail sales*
<!-- bilingual-en:end -->

若月度销量随水平增长而季节振幅也扩大，先检查对数变换后季节振幅是否稳定。候选可包括对数尺度的加法季节 ETS 与原尺度的乘法季节 ETS，再与 seasonal naïve 及季节 ARIMA 做 1–12 步 rolling-origin 比较。若阻尼趋势在长 horizon 更稳，选它的理由应是外样本与合理远期形状，而不是训练期拟合更紧。
<!-- bilingual-en:start -->
If monthly sales have seasonal amplitude that grows with the level, first examine whether a log transform stabilises it. Candidates can include additive seasonal ETS on the log scale and multiplicative seasonal ETS on the original scale, compared with seasonal naïve and seasonal ARIMA under rolling origins at horizons 1–12. If a damped trend is more stable at long horizons, justify it by out-of-sample performance and plausible distant shape rather than tighter training fit.
<!-- bilingual-en:end -->

## 诊断与边界
<!-- bilingual-en:start -->
*Diagnostics and boundaries*
<!-- bilingual-en:end -->

检查 innovations 是否近似零均值、无自相关，方差与分布假设是否合理。季节周期设错、节假日漂移、结构突变和新产品会让状态更新追赶过去却无法外推未来。
<!-- bilingual-en:start -->
Check whether innovations are approximately zero-mean and uncorrelated, and whether variance and distributional assumptions are plausible. A wrong seasonal period, moving holidays, structural breaks, and new products can make state updates follow the past while failing to extrapolate the future.
<!-- bilingual-en:end -->

自动 ETS 选模通常是在允许的 E/T/S 组合中比较信息准则，它不会替你确定数据是否被修订、是否需要节假日回归项或未来是否会换制度。统计选模的候选空间本身就是人的建模假设。
<!-- bilingual-en:start -->
Automatic ETS selection ordinarily compares information criteria across permitted E/T/S combinations. It cannot determine whether data revisions matter, whether moving-holiday regressors are needed, or whether the future will change regime. The candidate space itself is a modelling judgement.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### ETS 与 ARIMA 的核心视角差别是什么？
<!-- bilingual-en:start -->
*What is the central difference in perspective between ETS and ARIMA?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> ETS 直接更新水平/趋势/季节成分，ARIMA 通过差分和自相关结构描述序列；二者部分重叠但不互相完全包含。
<!-- bilingual-en:start -->
> [!answer]- Answer
> ETS directly updates level, trend, and seasonal states, whereas ARIMA describes a series through differencing and autocorrelation. They overlap partly but neither fully contains the other.
<!-- bilingual-en:end -->

### $\alpha$ 较大在简单指数平滑中意味着什么？
<!-- bilingual-en:start -->
*What does a larger $\alpha$ mean in simple exponential smoothing?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 最新观测获得更大权重，水平对变化反应更快，但也更容易追随短期噪声。
<!-- bilingual-en:start -->
> [!answer]- Answer
> The newest observation receives more weight, so the level reacts faster to change but follows short-run noise more readily.
<!-- bilingual-en:end -->

### 为什么长期预测常考虑阻尼趋势？
<!-- bilingual-en:start -->
*Why is a damped trend often considered for long-horizon forecasting?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> Holt 线性趋势会把最近斜率无限延伸，阻尼让趋势贡献随 horizon 逐渐封顶，常能避免不合理的远期爆炸。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Holt's linear method extrapolates the latest slope indefinitely; damping caps the cumulative trend contribution as horizon grows and often avoids implausible distant explosion.
<!-- bilingual-en:end -->

### 用自己的话区分“乘法季节”和“乘法误差”。
<!-- bilingual-en:start -->
*Distinguish multiplicative seasonality from multiplicative errors in your own words.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 乘法季节说季节效应以比率与水平结合；乘法误差说随机创新的尺度随预测水平缩放。一个是系统成分，一个是随机误差机制。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Multiplicative seasonality combines seasonal effects with the level as ratios; multiplicative errors scale random innovations with the forecast level. One is a systematic component and the other an error mechanism.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Hyndman & Athanasopoulos, [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/)：支持预测流程、基准、评估、ETS、回归与 ARIMA。
- [[01_Math/06_时间序列分析/lecture.pdf]] 与 [[01_Math/06_时间序列分析/lecture-dual.pdf]]：支持课程范围、记号、模型、检验与例题。
- Hyndman & Athanasopoulos, [Forecasting: Principles and Practice](https://otexts.com/fpp3/)：交叉核验预测、ARIMA、诊断与时序交叉验证。
<!-- bilingual-en:start -->
- Hyndman and Athanasopoulos, [Forecasting: Principles and Practice, 3rd ed.](https://otexts.com/fpp3/), was checked section by section for STL, simple exponential smoothing, Holt and damped trend, Holt–Winters, innovations state-space ETS, model selection, and forecast intervals.
- [[01_Math/06_时间序列分析/lecture.pdf|time-series lecture notes]] and [[01_Math/06_时间序列分析/lecture-dual.pdf|bilingual time-series lecture notes]] were checked for course notation and comparisons with ARIMA.
<!-- bilingual-en:end -->

# 指数平滑与 ETS：从递推更新到可检验预测分布
<!-- bilingual-en:start -->
*Exponential smoothing and ETS: from recursive updating to a testable forecast distribution*
<!-- bilingual-en:end -->

> [!tip] 这条路径解决什么
> 指数平滑常被误解成“把曲线画平滑”。真正的学习主线是：先判断哪些结构值得外推，再用递推状态形成点预测；随后加入误差模型，得到 likelihood、预测分布与可检查的 innovation；最后用真实时间顺序比较它是否胜过基准和 ARIMA。
>
> 全局关系见[[指数平滑与 ETS.canvas|指数平滑与 ETS 主题图]]；全课程顺序见[[02_Economy/08_经济预测与决策/00_课程总览|经济预测与决策课程总览]]。本页负责连续讲解，链接到的原子负责独立复习与跨课程复用。
> <!-- bilingual-en:start -->
> Exponential smoothing is not merely a device for drawing a smooth curve. The learning path is to decide which structure can be extrapolated, update states recursively to form point forecasts, add an error model to obtain a likelihood and forecast distribution, inspect the resulting innovations, and finally compare the model against benchmarks and ARIMA in genuine time order. Use the topic Canvas for the global structure and the course overview for the full course sequence; this page supplies the continuous explanation, while linked atoms remain independently reviewable and reusable.
> <!-- bilingual-en:end -->

## 1. 先固定预测问题，再讨论模型
<!-- bilingual-en:start -->
*1. Fix the forecasting problem before discussing the model*
<!-- bilingual-en:end -->

假设我们每月底预测一家门店未来 1、3 和 12 个月的销量。图上也许能看到缓慢增长、十二个月一轮的季节变化和偶发促销尖峰，但这还不是一个完整预测问题。必须先按 [[预测任务定义]] 与 [[预测时点与信息集]] 固定：预测变量和单位、每个 origin 真正可见的数据、各 horizon 的决策用途，以及促销计划在当时究竟是否已知。
<!-- bilingual-en:start -->
Suppose that at the end of each month we forecast a store's sales one, three, and twelve months ahead. A graph may show gradual growth, an annual seasonal pattern, and occasional promotion spikes, but this is not yet a complete forecasting problem. We must first fix the target and units, the information genuinely available at each origin, the decision attached to each horizon, and whether future promotion plans were actually known at that date.
<!-- bilingual-en:end -->

这一步决定后面所有比较是否公平。季节 naïve 可能是十二个月 horizon 的强 [[预测基准]]；若 ETS 在回测时偷看了后来修订的数据或未来促销，而基准没有，两者的误差不能解释为模型能力差异。模型从来不是脱离任务单独“最好”的对象。
<!-- bilingual-en:start -->
This design determines whether every later comparison is fair. Seasonal naive may be a strong benchmark at a twelve-month horizon. If ETS is backtested using later data revisions or future promotion information that the benchmark did not receive, the error difference cannot be attributed to model quality. A model is never simply “best” outside a specified forecasting task.
<!-- bilingual-en:end -->

## 2. 分解描述过去，预测必须另加外推规则
<!-- bilingual-en:start -->
*2. Decomposition describes the past; forecasting requires extrapolation rules*
<!-- bilingual-en:end -->

加法分解
$$
y_t=T_t+S_t+R_t
$$
把历史观测拆成趋势—周期、季节与 remainder；乘法分解 $y_t=T_tS_tR_t$ 则让季节振幅随水平按比例变化。它们帮助我们看清样本内结构，却没有告诉我们 $T_{T+h}$ 和 $S_{T+h}$ 在未来取什么值。因此 [[时间序列分解与预测|分解不等于预测]]。
<!-- bilingual-en:start -->
An additive decomposition separates the observed history into trend-cycle, seasonal, and remainder components; a multiplicative decomposition lets seasonal amplitude vary proportionally with the level. Both illuminate in-sample structure, but neither specifies the future values of the trend or seasonal components. Decomposition is therefore not itself a forecast model.
<!-- bilingual-en:end -->

若先用 STL 得到 seasonally adjusted series，仍要为这条序列选择 Holt、ARIMA 或其他模型，并为未来季节项选择 seasonal naïve 等规则，最后再把季节加回或乘回。即使历史分解完全相同，换一条外推规则也会改变预测。平滑的历史趋势线只是候选结构，不是向未来延伸的许可证。
<!-- bilingual-en:start -->
After using STL to obtain a seasonally adjusted series, we still need a model such as Holt or ARIMA for that series and a rule such as seasonal naive for future seasonal indices, followed by reseasonalisation. Identical historical decompositions can therefore produce different forecasts under different extrapolation rules. A smooth historical trend is a candidate structure, not a licence to extend it indefinitely.
<!-- bilingual-en:end -->

## 3. SES：把最新误差写回当前水平
<!-- bilingual-en:start -->
*3. SES: feed the latest error back into the current level*
<!-- bilingual-en:end -->

没有要外推的趋势和季节时，simple exponential smoothing（SES）只维护一个水平状态：
$$
\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1}
=\ell_{t-1}+\alpha(y_t-\ell_{t-1}),
$$
$$
\hat y_{t+h|t}=\ell_t,qquad h\ge1.
$$
第二种写法尤其重要：模型先用旧水平预测 $y_t$，再把一步误差的 $\alpha$ 比例写回水平。所有未来 horizon 都等于当前水平，所以 SES 是“局部水平”预测，不会凭空生成趋势。
<!-- bilingual-en:start -->
When neither trend nor seasonality needs to be extrapolated, simple exponential smoothing maintains one level state. The error-correction form is especially revealing: the old level predicts the new observation, and a fraction alpha of the one-step error is fed back into the level. Every future horizon equals the current level, so SES is a local-level forecast and cannot invent a trend.
<!-- bilingual-en:end -->

反复展开递推，最近观测的权重依次为 $\alpha,\alpha(1-\alpha),\alpha(1-\alpha)^2,\ldots$，另有不能在短样本中随手删掉的初值余项 $(1-\alpha)^T\ell_0$。这就是 [[SES权重机制|指数权重]]：$\alpha$ 大，旧信息更快衰减，模型转向更快也更容易追随噪声；$\alpha$ 小，路径更平滑，但真实水平改变后会滞后。
<!-- bilingual-en:start -->
Expanding the recursion gives weights alpha, alpha times one-minus-alpha, alpha times one-minus-alpha squared, and so on, plus an initial-state remainder that cannot be casually dropped in a short sample. A larger alpha forgets old information more quickly, allowing faster turns but greater noise sensitivity; a smaller alpha produces a smoother path but lags after a genuine level shift.
<!-- bilingual-en:end -->

### 一个最小计算例子
<!-- bilingual-en:start -->
*A minimal calculation*
<!-- bilingual-en:end -->

令 $\ell_0=100$、$\alpha=0.3$，接下来三个观测为 $110,90,120$：
$$
\ell_1=0.3(110)+0.7(100)=103,
$$
$$
\ell_2=0.3(90)+0.7(103)=99.1,
$$
$$
\ell_3=0.3(120)+0.7(99.1)=105.37.
$$
于是从第三期出发，1、3 或 12 期点预测全是 $105.37$。这个结果不是“模型没算完”，而是 SES 对未来结构作出的明确假设：当前水平会延续，但没有独立趋势或季节状态。
<!-- bilingual-en:start -->
Starting from a level of 100 with alpha equal to 0.3, the observations 110, 90, and 120 update the level to 103, 99.1, and 105.37. Every forecast made after the third observation is then 105.37, whether the horizon is one, three, or twelve periods. This is not an unfinished computation; it is the SES assumption that the current level persists without a separate trend or seasonal state.
<!-- bilingual-en:end -->

## 4. Holt：加入趋势，也要约束远期形状
<!-- bilingual-en:start -->
*4. Holt: add a trend, then discipline its long-run shape*
<!-- bilingual-en:end -->

若序列有局部斜率，Holt 法同时更新水平 $\ell_t$ 与趋势 $b_t$，并给出
$$
\hat y_{t+h|t}=\ell_t+hb_t.
$$
这里的 $b_t$ 是由近期数据更新的统计状态，不是永远不变的经济增长率。把它乘以任意大的 $h$，等于把最后一段局部趋势无限外推；短期合理，不代表长期也合理。
<!-- bilingual-en:start -->
When a series has a local slope, Holt's method updates both a level and a trend and extrapolates the trend linearly. The slope is a data-updated statistical state, not a permanent economic growth rate. Multiplying it by an arbitrarily large horizon extends the latest local trend indefinitely, an assumption that may be plausible nearby but not far away.
<!-- bilingual-en:end -->

[[Holt阻尼趋势|阻尼 Holt]] 用几何和替代 $h$：
$$
\hat y_{t+h|t}=\ell_t+\left(\phi+\phi^2+\cdots+\phi^h\right)b_t,
\qquad 0<\phi<1.
$$
当 $h\to\infty$，趋势贡献趋于 $\phi b_t/(1-\phi)$。例如 $\phi=0.9$ 时上限是 $9b_t$。这并不是声称真实趋势最终消失，而是在承认“越远期，最后一段斜率越不值得原样相信”。同一个 $\phi$ 也必须进入状态更新，不能先拟合无阻尼 Holt，再只在输出端压低预测。
<!-- bilingual-en:start -->
Damped Holt replaces the horizon with a geometric sum. As the horizon grows, the trend contribution approaches a finite limit; with phi equal to 0.9, that limit is nine times the current slope. This does not claim that the real trend vanishes. It formalises declining confidence in extending the latest slope unchanged. The same damping parameter must enter the state updates, not merely be applied to forecasts after fitting an undamped model.
<!-- bilingual-en:end -->

## 5. Holt–Winters：季节是差值还是比率
<!-- bilingual-en:start -->
*5. Holt-Winters: is seasonality a difference or a ratio?*
<!-- bilingual-en:end -->

加入周期为 $m$ 的季节状态后，[[Holt-Winters季节形式|加法与乘法季节]]回答的是两个不同问题：

- 加法季节：旺季比基础路径高多少个单位？季节指标与 $y_t$ 同单位，一个周期通常和为零。
- 乘法季节：旺季是基础路径的多少倍？季节指标是比率，一个周期通常平均为一。

<!-- bilingual-en:start -->
With a seasonal state of period m, additive and multiplicative seasonality answer different questions. Additive seasonality asks how many observation units a season lies above or below the baseline, with indices summing roughly to zero over a cycle. Multiplicative seasonality asks by what ratio the season scales the baseline, with indices averaging roughly one.
<!-- bilingual-en:end -->

若一家店不论年销量高低，十二月都比常态多卖约 200 件，加法季节较自然；若十二月稳定约为常态的 1.4 倍，乘法季节较自然。选择依据不是“图上尖峰很大”，而是振幅与水平的关系。乘法更新还需要用 level 或 seasonal factor 作分母，因此零、负值与接近零的状态会破坏定义、解释或数值稳定性。
<!-- bilingual-en:start -->
If December sales are about 200 units above normal regardless of the annual level, additive seasonality is natural. If December is consistently about 1.4 times the baseline, multiplicative seasonality is natural. The choice concerns how amplitude relates to level, not whether the peaks look large. Multiplicative updates also divide by level or seasonal factors, so zero, negative, or near-zero states threaten their definition, interpretation, or numerical stability.
<!-- bilingual-en:end -->

## 6. 从平滑方法到 ETS 随机模型
<!-- bilingual-en:start -->
*6. From smoothing methods to stochastic ETS models*
<!-- bilingual-en:end -->

SES、Holt 和 Holt–Winters 首先是生成点预测的递推方法。[[ETS三轴模型|ETS$(E,T,S)$]] 再为这些递推补上观测方程、状态方程与创新分布；三个位置固定表示 Error、Trend、Seasonal。例如 ETS$(A,A_d,M)$ 是 additive error、additive damped trend、multiplicative seasonality，不能把第一个 A 读成季节形式。
<!-- bilingual-en:start -->
SES, Holt, and Holt-Winters begin as recursive point-forecasting methods. ETS adds measurement equations, state equations, and an innovation distribution. Its three positions are fixed as Error, Trend, and Seasonal. Thus ETS(A,Ad,M) means additive error, additive damped trend, and multiplicative seasonality; the first letter cannot be read as the seasonal form.
<!-- bilingual-en:end -->

最简单的 ETS$(A,N,N)$ 是
$$
y_t=\ell_{t-1}+\varepsilon_t,
\qquad
\ell_t=\ell_{t-1}+\alpha\varepsilon_t.
$$
同一个一步创新既解释观测为何偏离条件位置，又决定水平怎样更新。指定 $\varepsilon_t$ 的概率分布后，模型才不只给一条点预测，还能写 likelihood 并生成预测分布。level、trend 与 seasonal states 是服务预测的潜在统计状态，不应未经额外识别就解释成唯一真实的经济机制。
<!-- bilingual-en:start -->
In ETS(A,N,N), the same one-step innovation explains why the observation differs from its conditional location and determines how the level is updated. Once a probability law for that innovation is specified, the model supplies not only point forecasts but also a likelihood and forecast distribution. Level, trend, and seasonal states are latent statistical devices for prediction, not uniquely identified economic mechanisms.
<!-- bilingual-en:end -->

## 7. “成分怎样组合”与“误差怎样进入”是两件事
<!-- bilingual-en:start -->
*7. Component combination and error entry are separate choices*
<!-- bilingual-en:end -->

设状态给出的条件位置为
$$
\mu_t=\ell_{t-1}+b_{t-1}+s_{t-m}.
$$
additive error 写成 $y_t=\mu_t+\varepsilon_t$；multiplicative error 写成 $y_t=\mu_t(1+\varepsilon_t)$。若状态和参数相同，两者可以有相同点预测位置，却有不同的条件方差、likelihood 和预测区间。这正是 [[加法平滑与加法误差|平滑形式与误差形式不能混读]] 的原因。
<!-- bilingual-en:start -->
Given the same state-generated conditional location, additive and multiplicative error models may share the same point forecast. They nevertheless imply different conditional variances, likelihoods, and forecast intervals. How components combine inside the location and how random error enters around that location are separate modelling choices.
<!-- bilingual-en:end -->

这个区别也决定残差的定义。[[ETS创新残差|ETS innovation residual]] 以 $\mu_t=\hat y_{t|t-1}$ 为基准：
$$
\hat\varepsilon_t=
\begin{cases}
y_t-\mu_t, & \text{additive error},\\[4pt]
(y_t-\mu_t)/\mu_t, & \text{multiplicative error}.
\end{cases}
$$
multiplicative innovation 是相对误差；原尺度 regular residual 仍是 $e_t=y_t-\mu_t=\mu_t\hat\varepsilon_t$。预测位置从 50 增到 200 时，同样的 10% surprise 会产生 5 与 20 两个 raw residual，却产生相同的 innovation。诊断时检查错对象，会把尺度变化误判成模型失配。
<!-- bilingual-en:start -->
The error form also determines the innovation residual. Under additive errors it is the raw one-step error; under multiplicative errors it is that error divided by the conditional location. A ten-percent surprise therefore produces the same multiplicative innovation at forecast levels 50 and 200, even though the raw residuals are 5 and 20. Diagnosing the wrong quantity confuses ordinary scale variation with model failure.
<!-- bilingual-en:end -->

## 8. Likelihood 同时估计平滑参数和初始状态
<!-- bilingual-en:start -->
*8. Likelihood estimates smoothing parameters and initial states together*
<!-- bilingual-en:end -->

季节 ETS 的未知量不只有 $\alpha,\beta,\gamma,\phi$，还包括 $\ell_0,b_0$ 与一组季节初值。[[ETS似然与初始状态|现代 ETS 估计]]通常在给定误差模型后联合选择这些量，而不是随手指定初值再只优化平滑参数。加法季节指标有和为零的归一化，乘法季节指标有平均为一的归一化，所以“显示了 $m$ 个季节初值”不等于增加了 $m$ 个自由参数。
<!-- bilingual-en:start -->
A seasonal ETS model has unknown initial level, trend, and seasonal states in addition to its smoothing parameters. Modern estimation usually chooses these quantities jointly under the specified error model rather than fixing arbitrary starting states. Seasonal normalisations mean that displaying m initial indices does not automatically add m independent parameters.
<!-- bilingual-en:end -->

在对应的 Gaussian additive-error 模型中，最大化 likelihood 可与最小化一步平方误差一致；multiplicative-error likelihood 还包含随 $\mu_t$ 变化的尺度，因此通常不能用原尺度 SSE 代替。AICc 中的 $k$ 要计入自由初始状态与创新方差，而且只有当响应、有效样本、变换/Jacobian、常数与初始化口径可比时，信息准则差值才有共同含义。这些边界与 [[ARMA信息准则]] 共用，不应为 ETS 再造一套较松的标准。
<!-- bilingual-en:start -->
For the corresponding Gaussian additive-error setup, likelihood maximisation can coincide with minimising one-step squared errors. A multiplicative-error likelihood contains a state-dependent scale term and generally cannot be replaced by raw-scale SSE. AICc must count free initial states and innovation variance, and its differences are meaningful only under comparable responses, samples, transformations, likelihood constants, and initialisation conventions. ETS reuses the same information-criterion discipline as ARMA.
<!-- bilingual-en:end -->

信息准则适合在可比的 ETS 候选内部筛选；它不替代样本外证据。尤其在常用实现对 ETS 与 ARIMA 使用不同 likelihood 口径时，直接比较软件打印的两族 AICc 没有同一个尺子，跨族选择必须回到 [[滚动起点评估]]。
<!-- bilingual-en:start -->
Information criteria are useful for screening comparable ETS candidates, but they do not replace out-of-sample evidence. When software computes ETS and ARIMA likelihoods under different conventions, their printed AICc values do not share a common ruler. Cross-family choice must return to rolling-origin evaluation.
<!-- bilingual-en:end -->

## 9. 残差能发现历史失配，不能预言未来制度
<!-- bilingual-en:start -->
*9. Residuals can reveal historical misspecification, not future regimes*
<!-- bilingual-en:end -->

拟合后应先画 innovation 序列，再检查均值、ACF、异常值、尾部以及绝对值或平方后的剩余结构；[[Ljung-Box检验]]用于联合检查一组预先说明的滞后。若 innovation 持续同号，条件位置存在系统偏差；若仍有自相关，历史信息尚未被模型用完。若预测区间依赖正态或稳定尺度，还必须检查相应分布与方差假设。
<!-- bilingual-en:start -->
After fitting, plot the innovation sequence and inspect its mean, ACF, outliers, tails, and any remaining structure in absolute or squared innovations. A Ljung-Box test jointly checks a predeclared set of lags. Persistent same-sign innovations indicate location bias, while residual autocorrelation indicates unused historical information. Distributional and variance assumptions also matter when they underpin forecast intervals.
<!-- bilingual-en:end -->

但“Ljung–Box 未拒绝”只是在该样本、该滞后与该检验力下没有发现所检线性相关；它不证明 innovations 独立、正态、同方差，也不证明模型唯一正确。这里直接复用 [[ARMA残差诊断]] 的检验解释，不因模型叫 ETS 就放宽证据标准。
<!-- bilingual-en:start -->
A non-rejection by Ljung-Box means only that the tested linear dependence was not detected at those lags in that sample and with that power. It does not prove independence, normality, homoskedasticity, or uniqueness of the model. ETS therefore reuses the same diagnostic interpretation as ARMA.
<!-- bilingual-en:end -->

更重要的是，[[ETS残差诊断边界|历史残差诊断不能排除未来结构突变]]。促销制度、产品范围或季节相位下月改变时，旧样本完全可能诊断良好；固定参数 ETS 只能在新观测到来后逐步更新状态，不能提前知道断点。持续偏移应触发信息集核查、干预变量、重新设定或训练窗口调整，而不是只把 $\alpha$ 调大继续追。训练期诊断与部署后监控回答的是不同时间问题。
<!-- bilingual-en:start -->
More importantly, historical residual diagnostics cannot rule out a future structural break. A promotion regime, product range, or seasonal timing may change next month even when the old sample diagnoses well. Fixed-parameter ETS can update after new observations arrive but cannot anticipate the break. Persistent displacement should trigger investigation of the information set, interventions, specification, or training window rather than an automatic increase in alpha. Training diagnostics and deployment monitoring answer different temporal questions.
<!-- bilingual-en:end -->

## 10. ETS 与 ARIMA 有交集，但谁也不包含谁
<!-- bilingual-en:start -->
*10. ETS and ARIMA overlap, but neither contains the other*
<!-- bilingual-en:end -->

[[ETS与ARIMA边界|两族只在受限的线性加法子类中重叠]]。部分 additive-error ETS 可化为带参数限制的 ARIMA；multiplicative error 或 multiplicative seasonality 产生的非线性 ETS 没有普通线性 ARIMA 对应。反过来，许多平稳 [[ARIMA模型|ARIMA]] 也没有 ETS 对应。因此从一个等价特例推出“ETS 只是 ARIMA”或“ARIMA 总比 ETS 一般”都越过了成立范围。
<!-- bilingual-en:start -->
The two families overlap only in restricted linear additive subclasses. Some additive-error ETS models reduce to ARIMA models with parameter restrictions; nonlinear ETS models with multiplicative errors or seasonality have no ordinary linear ARIMA counterpart. Conversely, many stationary ARIMA models have no ETS counterpart. One special equivalence therefore cannot establish that either whole family contains the other.
<!-- bilingual-en:end -->

最有用的代数锚点是 [[SES-ARIMA受限等价|ETS$(A,N,N)$ 与 ARIMA$(0,1,1)$]]。由
$$
y_t=\ell_{t-1}+\varepsilon_t,
\qquad
\ell_t=\ell_{t-1}+\alpha\varepsilon_t
$$
以及上一期的两式相减，可得
$$
\Delta y_t=\varepsilon_t+(\alpha-1)\varepsilon_{t-1}.
$$
若 MA 采用 $(1+\theta B)\varepsilon_t$ 约定，则 $\theta=\alpha-1$；若采用 $(1-\theta B)\varepsilon_t$，报告符号变成 $\theta=1-\alpha$。有限样本预测还要求 ETS 的 $\ell_0$ 与 ARIMA 的 pre-sample innovations、条件化和 likelihood 初始化一致。默认软件结果略有差异，并不自动推翻方程层面的等价。
<!-- bilingual-en:start -->
The most useful algebraic anchor is the equivalence between ETS(A,N,N) and a restricted ARIMA(0,1,1). Subtracting adjacent measurement and state equations gives a first difference equal to the current innovation plus alpha-minus-one times the previous innovation. The reported MA sign changes with the software convention, and finite-sample equality also requires aligned initial levels, pre-sample innovations, conditioning, and likelihood initialisation. Small differences between default software fits do not by themselves refute the equation-level equivalence.
<!-- bilingual-en:end -->

## 11. 一条可执行的 ETS 工作流
<!-- bilingual-en:start -->
*11. An executable ETS workflow*
<!-- bilingual-en:end -->

面对一个新序列，可以按以下顺序工作：

1. 固定 target、origin、horizon、信息集与评价损失；选同条件下的 naïve 或 seasonal-naïve 基准。
2. 先画原序列和 seasonally adjusted view，判断分解是描述工具还是确实配有外推规则。
3. 根据结构提出少量候选：只有水平用 SES；有局部趋势考虑 Holt 与阻尼；有季节再判断差值还是比率。
4. 把点预测方法写成明确的 ETS$(E,T,S)$，分别判断 component form 与 error form；检查零值、负值和接近零的边界。
5. 用可复现的 likelihood 与约束联合估计平滑参数和初始状态；只在可比候选内解释 AICc。
6. 检查正确的 innovation，而不是含义不明的“残差”；处理偏差、相关、尺度、异常值和结构变化线索。
7. 按实际 horizon 做 rolling-origin 比较，并与基准、ARIMA 或动态回归使用完全相同的信息集和损失。
8. 输出与决策匹配的点、区间或完整分布；区间解释继续复用 [[ARMA预测区间|预测区间的分布依据与遗漏边界]]。
9. 部署后等待 outcome 成熟并对齐预测版本；监控信号先触发诊断，不自动授权重训。

<!-- bilingual-en:start -->
For a new series, first fix the target, origin, horizon, information set, loss, and matched benchmark. Use plots and decomposition to propose a small set of structural candidates. Translate the chosen recursion into an explicit ETS error-trend-seasonal specification, checking scale and positivity boundaries. Estimate smoothing parameters and initial states under a reproducible likelihood, interpret information criteria only among comparable candidates, and diagnose the correctly defined innovation. Compare candidates by rolling origins at the decision-relevant horizons under the same information set and loss. Produce the forecast object the decision actually requires, then monitor only matured outcomes aligned to their forecast versions; a signal starts diagnosis rather than automatically authorising retraining.
<!-- bilingual-en:end -->

这条流程的核心不是“自动 ETS 替你选三个字母”，而是把每个字母背后的预测承诺、误差尺度、估计口径和检验边界都暴露出来。只有这样，递推公式才会变成一套能解释、能反驳、也能在新数据到来后维护的预测系统。
<!-- bilingual-en:start -->
The point is not that an automatic ETS routine chooses three letters for us. It is to expose the forecasting commitment, error scale, estimation convention, and diagnostic boundary behind each letter. Only then does a recursion become a forecasting system that can be explained, challenged, and maintained as new data arrive.
<!-- bilingual-en:end -->

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §5.7](https://otexts.com/fpp3/forecasting-decomposition.html)：支持分解后仍需分别外推 seasonally adjusted 与 seasonal components，再重新季节化。
- [Hyndman & Athanasopoulos, FPP3 §§8.1–8.3](https://otexts.com/fpp3/expsmooth.html)：支持 SES、Holt、阻尼趋势与 Holt–Winters 的递推、权重和远期预测形状。
- [Hyndman & Athanasopoulos, FPP3 §§8.5–8.6](https://otexts.com/fpp3/ets.html)：支持 ETS 三轴、创新状态空间、误差形式、likelihood、初始状态和模型选择边界。
- [Hyndman & Athanasopoulos, FPP3 §5.4](https://otexts.com/fpp3/diagnostics.html)：支持 innovation 的零均值与无剩余相关要求，以及通过诊断不等于模型已被证明正确。
- [Hyndman & Athanasopoulos, FPP3 §9.10](https://otexts.com/fpp3/arima-ets.html)：支持 ETS–ARIMA 的部分重叠、SES 参数映射与跨族时序评估。
- [Hyndman, Koehler, Snyder & Grose (2002)](https://doi.org/10.1016/S0169-2070(01)00110-8)：支持自动预测中的 innovations state-space ETS 框架与模型族构造。

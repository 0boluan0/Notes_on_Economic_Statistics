# ARCH/GARCH：从条件方差到可检验的波动预测
<!-- bilingual-en:start -->
*ARCH/GARCH: from conditional variance to testable volatility forecasts*
<!-- bilingual-en:end -->

ARCH/GARCH 的核心不是“收益大了，下一期方差也大”这一句口号，而是一条完整的建模链：先把条件均值与创新分开，再定义预测时点上的条件方差；用过去信息递推这条方差；区分正性、平稳和矩存在；用平方残差发现遗漏；最后在真实历史起点上评价预测。[[条件异方差：ARCH 与 GARCH.canvas|主题 Canvas]] 展示全局关系；本章把这些关系排成一条可以连续阅读和实际执行的路径。原始课堂展开、图表和例题仍保留在[[01_Math/06_时间序列分析/04_波动建模 Modeling Volatility|课堂记录]]中。
<!-- bilingual-en:start -->
ARCH/GARCH is a complete modelling chain rather than a slogan about large returns and high future variance. Separate the conditional mean from its innovation, define variance at the forecast origin, model that variance from past information, distinguish positivity from stationarity and moment existence, diagnose squared residuals, and evaluate forecasts at honest historical origins. The Canvas shows the global structure; this chapter supplies the continuous argument.
<!-- bilingual-en:end -->

## 1. 波动是一个条件对象
<!-- bilingual-en:start -->
*1. Volatility is a conditional object*
<!-- bilingual-en:end -->

给定预测前已经知道的信息集 $\mathcal F_{t-1}$，并假定所需二阶矩有限，先写
$$
\mu_t=E(y_t\mid\mathcal F_{t-1}),\qquad
\varepsilon_t=y_t-\mu_t.
$$
[[条件方差]]是
$$
h_t=\operatorname{Var}(y_t\mid\mathcal F_{t-1})
=E(\varepsilon_t^2\mid\mathcal F_{t-1}).
$$
它是在 $y_t$ 到来之前，对本期剩余不确定性的判断。$h_t$ 是方差，$\sqrt{h_t}$ 才是波动率或条件标准差；随后观察到的 $\varepsilon_t^2$ 是这次冲击的实现，不是事先知道的方差。
<!-- bilingual-en:start -->
With the required second moments finite, conditional variance is the remaining uncertainty assessed before $y_t$ arrives. The variance is $h_t$, its square root is conditional volatility, and the subsequently realised squared innovation is an observation generated around that forecast—not the forecast itself.
<!-- bilingual-en:end -->

[[条件异方差]]指 $h_t$ 会随已知信息状态改变，而不是几乎处处等于同一个常数。这与“无条件方差随日历时间改变”不同：严格平稳的 GARCH 可以有不变的无条件分布，但路径上的 $h_t$ 仍每天更新；真正的长期方差断点也可能被固定参数模型误吸收为高持久性。
<!-- bilingual-en:start -->
Conditional heteroskedasticity means that $h_t$ varies with the information state. It is distinct from a calendar-time change in unconditional variance: a stationary GARCH process updates its conditional variance along the path, while an omitted structural break can imitate persistent GARCH dynamics.
<!-- bilingual-en:end -->

经验上，[[波动率聚集]]常表现为原始收益或均值残差的线性相关很弱，而其绝对值或平方持续相关。方向可以难以预测，幅度却有记忆。于是“残差 ACF 看起来像白噪声”只说明均值动态可能处理得不错，不能宣布风险恒定。
<!-- bilingual-en:start -->
Volatility clustering often combines weak linear dependence in returns with persistent dependence in their magnitudes or squares. Direction can be unpredictable while scale remains predictable, so a quiet residual ACF is not evidence of constant risk.
<!-- bilingual-en:end -->

为了固定单位，使用[[条件尺度与标准化冲击|条件尺度分解]]
$$
\varepsilon_t=\sqrt{h_t}\,z_t,\qquad
E(z_t\mid\mathcal F_{t-1})=0,\qquad
E(z_t^2\mid\mathcal F_{t-1})=1.
$$
$\varepsilon_t$ 是均值创新，$h_t$ 是预测方差，$z_t$ 是去掉时变尺度后的标准化冲击。后续似然、残差诊断和预测区间都依赖这三个对象不被混称为“残差”。
<!-- bilingual-en:start -->
The scale decomposition separates the mean innovation, its predictable variance, and the standardized shock. Likelihoods, diagnostics, and prediction intervals all fail dimensionally if these three objects are collapsed into the single word “residual”.
<!-- bilingual-en:end -->

## 2. ARCH 与 GARCH 在递推什么
<!-- bilingual-en:start -->
*2. What ARCH and GARCH actually recurse on*
<!-- bilingual-en:end -->

[[ARCH(q)模型|ARCH($q$)]]让最近 $q$ 个平方创新直接进入当前方差：
$$
h_t=\omega+\sum_{i=1}^{q}\alpha_i\varepsilon_{t-i}^2.
$$
平方意味着模型先对冲击大小反应，而不区分正负；$q$ 是直接记忆长度。若需要许多滞后才能表示缓慢衰减，ARCH 会迅速变得笨重。
<!-- bilingual-en:start -->
ARCH($q$) feeds the latest $q$ squared innovations directly into current variance. Squaring makes the baseline model respond to magnitude rather than sign, and a long decay may require many coefficients.
<!-- bilingual-en:end -->

[[GARCH(p,q)模型|GARCH($p,q$)]]再让方差依赖自身的过去：
$$
h_t=\omega+\sum_{i=1}^{q}\alpha_i\varepsilon_{t-i}^2
+\sum_{j=1}^{p}\beta_jh_{t-j}.
$$
过去的 $h$ 已经汇总了更早的平方冲击，所以低阶 GARCH 能产生无限衰减的记忆。这里采用“$p$ 个方差滞后、$q$ 个平方创新滞后”的 Bollerslev 记号；软件可能交换 $p,q$，因此阶数标签必须与方程一起读。
<!-- bilingual-en:start -->
GARCH adds lags of conditional variance. Because yesterday's variance already summarizes older squared shocks, a low-order recursion can generate long, decaying memory. Lag-order conventions differ across sources, so the equation—not the label alone—fixes the model.
<!-- bilingual-en:end -->

以 GARCH(1,1) 为例，若
$$
\omega=0.002,\quad \alpha=0.10,\quad \beta=0.85,\quad
h_t=0.04,\quad \varepsilon_t^2=0.09,
$$
则
$$
h_{t+1}=0.002+0.10(0.09)+0.85(0.04)=0.045.
$$
最新冲击把明日方差从当前状态往上推；之后若不再出现大冲击，递推会逐步消化这次更新。这个例子同时说明：$\varepsilon_t^2$ 是当日观察，$h_{t+1}$ 才是据此形成的次日预测。
<!-- bilingual-en:start -->
In the numerical update, today's realized squared innovation moves tomorrow's variance from its filtered state. The indexing matters: the observed square belongs to date $t$, while the resulting variance forecast belongs to date $t+1$.
<!-- bilingual-en:end -->

## 3. 写得出方程，不等于模型在概率上合法
<!-- bilingual-en:start -->
*3. A writable recursion is not yet a valid stochastic model*
<!-- bilingual-en:end -->

第一层是[[ARCH-GARCH正性条件|正性]]。对线性 ARCH/GARCH，$\omega>0$、$\alpha_i\ge0$、$\beta_j\ge0$ 是保证所有允许历史下 $h_t>0$ 的常用充分条件。它只回答“方差会不会跑到负数”，不回答分布是否平稳，也不保证无条件方差有限。
<!-- bilingual-en:start -->
Positivity restrictions keep the variance recursion in its legal range. They do not establish stationarity or the existence of unconditional moments.
<!-- bilingual-en:end -->

第二层是[[GARCH严格平稳条件|严格平稳]]。对 GARCH(1,1)
$$
h_t=\omega+(\alpha z_{t-1}^2+\beta)h_{t-1},
$$
随机系数乘积是否长期收缩由
$$
E\!\left[\log(\alpha z_t^2+\beta)\right]<0
$$
刻画，并伴随相应可积条件。它看的是平均对数增长率，而不是简单的 $\alpha+\beta$。
<!-- bilingual-en:start -->
Strict stationarity of GARCH(1,1) is governed by contraction in the random coefficient product, summarized by a negative log moment. This is a distributional condition, not the familiar coefficient-sum rule.
<!-- bilingual-en:end -->

第三层才是[[GARCH有限方差条件|有限二阶矩]]。在标准化冲击且参数非负时，GARCH(1,1) 有有限、时间不变的无条件方差当且仅当
$$
\rho:=\alpha+\beta<1,\qquad
\bar h=E(h_t)=\frac{\omega}{1-\rho}.
$$
有限二阶矩通常足以推出严格平稳，但严格平稳不保证二阶矩存在。把 $\rho<1$ 简写成“平稳条件”会掩盖这两种不同结论。
<!-- bilingual-en:start -->
Finite unconditional variance requires $\alpha+\beta<1$ under the standardized GARCH(1,1) setup. This stronger moment condition usually implies strict stationarity, but the converse can fail; calling it simply “the stationarity condition” loses that distinction.
<!-- bilingual-en:end -->

在有限二阶矩世界中，[[GARCH参数持久性|$\alpha$、$\beta$ 与 $\rho$]]回答不同问题：$\alpha$ 控制对新平方冲击的即时反应，$\beta$ 控制旧方差状态的延续，$\rho$ 控制条件期望中的方差缺口衰减。相同的 $\rho$ 可以来自完全不同的“快反应/慢延续”组合；它也会受频率、异常点、分布和结构断点影响，不是脱离模型的自然常数。
<!-- bilingual-en:start -->
Alpha measures immediate news response, beta carries forward the existing variance state, and their sum controls decay in the expected variance gap when second moments exist. Equal sums need not imply equal paths, and the estimated sum is model- and sample-dependent.
<!-- bilingual-en:end -->

定义 $v_t=\varepsilon_t^2-h_t$，GARCH(1,1) 可改写成[[平方创新ARMA表示|平方创新的 ARMA(1,1) 型表示]]：
$$
\varepsilon_t^2
=\omega+(\alpha+\beta)\varepsilon_{t-1}^2+v_t-\beta v_{t-1}.
$$
这解释了为什么原创新可不相关而平方创新仍有持续结构。若要进一步讨论平方过程的 ACF，需要 $E(\varepsilon_t^4)<\infty$；代数表示本身不能自动赋予它普通 Gaussian ARMA 的分布性质。
<!-- bilingual-en:start -->
The squared innovations have an ARMA-type algebraic representation, which explains persistent magnitude dependence despite uncorrelated innovations. Treating the squared series as a covariance-stationary process requires a finite fourth moment, and its errors are not ordinary homoskedastic Gaussian ARMA errors.
<!-- bilingual-en:end -->

## 4. 从均值残差中发现遗漏的方差动态
<!-- bilingual-en:start -->
*4. Detect omitted variance dynamics in mean-model residuals*
<!-- bilingual-en:end -->

先拟合足够的条件均值，再观察 $\hat\varepsilon_t$ 的时间图、绝对值和平方。原残差的 Ljung–Box 检查剩余线性均值相关；平方残差检查的是幅度依赖。若均值还漏了季节性、断点或自相关，方差检验可能把这些错误吸收成 ARCH，因此顺序不能颠倒。
<!-- bilingual-en:start -->
Fit and diagnose the conditional mean first, then inspect residual magnitudes and squares. Mean misspecification can spill into variance diagnostics, so the order is substantive rather than cosmetic.
<!-- bilingual-en:end -->

[[ARCH-LM检验]]估计辅助回归
$$
\hat\varepsilon_t^2=c+\sum_{j=1}^{q}a_j\hat\varepsilon_{t-j}^2+u_t
$$
并检验 $a_1=\cdots=a_q=0$。在相应正则条件下，有效样本量乘 $R^2$ 渐近服从 $\chi_q^2$。拒绝只表示所选 $q$ 个滞后上存在 ARCH 型平方依赖；它不自动选择 GARCH 阶数，也不证明经济机制。
<!-- bilingual-en:start -->
ARCH-LM tests a predeclared set of squared-residual lags through an auxiliary regression. Rejection detects ARCH-type dependence at those lags; it does not identify a unique volatility model or its economic cause.
<!-- bilingual-en:end -->

[[McLeod-Li检验]]则对平方残差的若干自相关做 portmanteau 联合检验。它与 ARCH-LM 是相关但不同的诊断视角；软件对拟合后自由度和有限样本校正的处理并不完全相同。平方自相关至少依赖有限四阶矩，经典渐近校准还可能需要更高矩，所以重尾数据里不能只报一个名义 p 值。
<!-- bilingual-en:start -->
McLeod–Li is a portmanteau test of joint squared-residual autocorrelations. Implementations differ in fitted-model adjustments, and heavy tails can undermine the moment assumptions behind nominal chi-square calibration.
<!-- bilingual-en:end -->

## 5. 估计时，密度与初值都属于模型
<!-- bilingual-en:start -->
*5. The density and initialization are both part of estimation*
<!-- bilingual-en:end -->

若假定 $z_t\mid\mathcal F_{t-1}\sim N(0,1)$，[[GARCH条件似然|单期条件 Gaussian 对数似然]]为
$$
\ell_t=-\frac12\left[\log(2\pi)+\log h_t+\frac{\varepsilon_t^2}{h_t}\right].
$$
样本开始处缺少过去的 $h$ 与创新，因此还要声明用无条件方差、样本方差、backcast 还是 presample 值初始化，以及是否丢弃 burn-in。短样本或接近边界时，初值影响尤其不能忽略。
<!-- bilingual-en:start -->
A conditional Gaussian likelihood also needs a presample convention. Long-run variance, sample variance, backcasting, and burn-in do not produce identical finite-sample objectives, especially in short or highly persistent samples.
<!-- bilingual-en:end -->

即使真实 $z_t$ 不是正态，也可以最大化 Gaussian 目标得到[[Gaussian QMLE|Gaussian QMLE]]。但一致性依赖条件均值、条件方差递推、识别、平稳遍历和所需矩都正确；非正态下推断应使用 sandwich/robust covariance。稳健标准误修正的是估计量的协方差，不会把遗漏断点或错误方差方程变正确。
<!-- bilingual-en:start -->
Gaussian QMLE can remain consistent under non-Gaussian shocks when the conditional moments and regularity conditions are correct. Robust covariance is then required, but it cannot repair a misspecified conditional variance.
<!-- bilingual-en:end -->

若用 Student-$t$ 描述厚尾，[[Student-t GARCH标准化|冲击必须标准化]]。普通 $u_t\sim t_\nu$ 的方差为 $\nu/(\nu-2)$，所以要令
$$
z_t=u_t\sqrt{\frac{\nu-2}{\nu}},\qquad \nu>2,
$$
才能继续把 $h_t$ 称为条件方差。不同软件对 scale、variance 和自由度的定义可能不同，复算 VaR 或迁移参数前必须查清口径。
<!-- bilingual-en:start -->
Student-$t$ innovations must be standardized to unit variance if $h_t$ is to retain its conditional-variance meaning. Package-specific scale conventions must be checked before transferring coefficients or computing tail risk.
<!-- bilingual-en:end -->

## 6. 诊断、比较与预测评价是一条闭环
<!-- bilingual-en:start -->
*6. Diagnosis, comparison, and forecast evaluation form one loop*
<!-- bilingual-en:end -->

拟合后先构造
$$
\hat z_t=\frac{\hat\varepsilon_t}{\sqrt{\hat h_t}}.
$$
[[GARCH残差双层诊断]]要求分别检查 $\hat z_t$ 与 $\hat z_t^2$：前者看均值动态是否仍有线性遗漏，后者看方差动态是否仍可预测。若声明了完整 Gaussian 或 Student-$t$ 分布，还要看 QQ/PIT、偏态、尾部、异常点和时间稳定性。两组相关检验未拒绝，只表示在所检查的滞后、统计量与检验力下没有发现相应剩余相关证据；它不能证明残差独立、创新分布正确或不存在其他遗漏。
<!-- bilingual-en:start -->
Standardized residuals and their squares answer different diagnostic questions. Failure to reject the corresponding correlation tests means only that no such residual dependence was detected at the inspected lags and statistics, given the tests' power. It does not establish independence, a correct innovation distribution, or the absence of other misspecification; distributional claims additionally require tail, skewness, PIT or QQ, outlier, and stability checks.
<!-- bilingual-en:end -->

[[GARCH选择与预测评估]]区分样本内比较和真正的预测选择。AIC/BIC 要求候选针对同一观测响应与尺度、同一有效 likelihood 样本，并使用可比的完整概率密度与计分约定；常数、变换 Jacobian、条件/初值处理必须相容，每个候选都要分别最大化 likelihood 并完整计入均值、方差和分布参数。均值动态与创新分布可以是合法候选差异，未明确归一化的准似然分数却不能直接冒充可比的完整 likelihood。未来方差预测应按[[滚动起点评估|rolling/expanding origin]]在每个历史时点只使用当时可得信息，并保留最终 untouched test period。
<!-- bilingual-en:start -->
Information criteria may compare different mean dynamics and innovation distributions when candidates model the same observed response and effective sample with mutually comparable, maximised full likelihoods. Likelihood constants, transformation Jacobians, conditional or presample treatment, and parameter counts must be compatible; an unnormalised quasi-likelihood score is not automatically comparable. Forecast selection still requires ordered historical origins, complete refitting with then-available information, and a final period untouched during selection.
<!-- bilingual-en:end -->

真实 $h_t$ 不可直接观察，平方收益只是噪声很大的代理。若 $\tilde v_t$ 是实现方差代理、$\hat h_t$ 是预测，常见 QLIKE 写成
$$
L_{\mathrm{QLIKE}}(\tilde v_t,\hat h_t)
=\frac{\tilde v_t}{\hat h_t}-\log\!\left(\frac{\tilde v_t}{\hat h_t}\right)-1.
$$
在代理满足相应条件无偏性时，它具有对代理噪声稳健的比较性质；这不是让任何含噪代理都自动可靠。MSE、QLIKE、VaR exceptions 与区间覆盖服务不同决策目标，评价指标应在看结果前确定。
<!-- bilingual-en:start -->
Latent variance is evaluated through imperfect proxies. QLIKE has useful robustness properties under an appropriate conditionally unbiased proxy, not under arbitrary measurement error. Loss functions and tail-coverage criteria answer different decisions and should be fixed before inspecting rankings.
<!-- bilingual-en:end -->

## 7. 预测必须从正确的时间索引开始
<!-- bilingual-en:start -->
*7. Forecasting starts with the correct time index*
<!-- bilingual-en:end -->

在时点 $t$ 已观察 $\varepsilon_t$ 并滤得 $h_t$，[[GARCH一步方差预测|一步预测]]是
$$
\hat h_{t+1\mid t}
=\omega+\alpha\hat\varepsilon_t^2+\beta\hat h_t.
$$
必须使用刚刚实现的 $t$ 期平方创新。课堂幻灯片在这一处把平方项错印成 $\varepsilon_{t-1}^2$；按模型递推应如上纠正，否则预测无故落后一格。
<!-- bilingual-en:start -->
The one-step update uses the squared innovation observed at the current date. The lecture slide lags this term once too far; the recursion itself determines the corrected index.
<!-- bilingual-en:end -->

若标准化创新的条件分位数为 $q_p$，观测的 plug-in 条件分位数为
$$
Q_p(y_{t+1}\mid\mathcal F_t)
=\hat\mu_{t+1\mid t}+q_p\sqrt{\hat h_{t+1\mid t}}.
$$
只有 Gaussian 对称分布才把 95% 区间简化为 $\hat\mu\pm1.96\sqrt{\hat h}$。厚尾、偏态、参数估计误差和断点都会影响覆盖。
<!-- bilingual-en:start -->
Conditional variance supplies the scale, while the innovation distribution supplies the quantile. A symmetric Gaussian interval is only one special case and typically omits parameter and model uncertainty.
<!-- bilingual-en:end -->

对有限方差的 GARCH(1,1)，令 $\rho=\alpha+\beta<1$、$\bar h=\omega/(1-\rho)$，[[GARCH多步方差预测|多步预测]]为
$$
h_{t+j\mid t}
=\bar h+\rho^{j-1}\bigl(h_{t+1\mid t}-\bar h\bigr),\qquad j\ge1.
$$
半衰期 $\log(1/2)/\log\rho$ 描述的是**期望方差缺口**按当前数据频率减半，不是危机影响或实现波动保证在该日期消失。期限累计收益的风险还要汇总各期方差并处理均值动态和跨期协方差，不能把终点方差直接当整段方差。
<!-- bilingual-en:start -->
Multi-step forecasts revert geometrically to long-run variance when a finite second moment exists. The half-life describes decay in the expected variance gap, not a guaranteed date when realized volatility or an economic shock disappears. Horizon risk requires aggregation across the whole path.
<!-- bilingual-en:end -->

## 8. 接近边界时，先怀疑解释而不是只加模型
<!-- bilingual-en:start -->
*8. Near a boundary, question the interpretation before adding complexity*
<!-- bilingual-en:end -->

[[IGARCH平稳与矩边界|IGARCH]]通常指 $\alpha+\beta=1$。此时不存在有限长期方差，不能使用 $\omega/(1-\alpha-\beta)$；但若 log-moment 为负，过程仍可能严格平稳。这正是“严平稳”“有限方差”和“二阶持久性”不能互换的典型例子。
<!-- bilingual-en:start -->
IGARCH sits on the finite-second-moment boundary. It may nevertheless possess a strictly stationary solution under a negative log moment, so integrated second-moment behaviour is not identical to nonstationarity in distribution.
<!-- bilingual-en:end -->

接近一的 $\hat\alpha+\hat\beta$ 也可能来自[[方差断点伪GARCH持久性|未建模的方差断点]]。若模型被迫用一组参数覆盖两个长期尺度，它会用缓慢衰减来吸收制度切换。应比较分样本、滚动参数、断点或 regime 规格，并检查加入稳定性处理后持久性是否明显下降。
<!-- bilingual-en:start -->
An omitted variance break can masquerade as near-integrated persistence. Subsample, rolling, and break-aware analyses help distinguish slow within-regime decay from a shift between regimes.
<!-- bilingual-en:end -->

标准 GARCH 对正负同幅冲击反应相同。若数据支持符号差异，要先区分[[波动不对称与杠杆|统计不对称与杠杆机制]]：前者是条件预测关系，后者是需要资本结构等额外证据的经济因果解释。
<!-- bilingual-en:start -->
Statistical sign asymmetry is a predictive property. A leverage mechanism is one possible causal explanation and requires evidence beyond an asymmetric coefficient in a return series.
<!-- bilingual-en:end -->

[[GJR-GARCH模型]]在方差方程中加入负冲击指标，让正、负同幅冲击分别以 $\alpha$ 与 $\alpha+\gamma$ 更新 $h_t$。但“TARCH/TGARCH”并不唯一：[[TARCH命名边界]]提醒我们，有些来源用它指 GJR 型方差递推，另一些指 Zakoïan 的条件标准差递推。名称不能替代方程，两个尺度上的参数限制也不能互相搬运。
<!-- bilingual-en:start -->
GJR-GARCH changes the squared-shock coefficient by sign. The TARCH label is ambiguous across variance and conditional-standard-deviation recursions, so equations and scale-specific restrictions must be read explicitly.
<!-- bilingual-en:end -->

[[EGARCH模型]]递推 $\log h_t$，因此指数化后自然为正；中心化幅度项表示冲击大小，有符号项表示方向。它避免标准线性 GARCH 的非负系数框，但仍有自己的稳定与矩条件。换一套创新符号约定时，不对称系数的正负解释也会反转。
<!-- bilingual-en:start -->
EGARCH models log variance, separating centred shock magnitude from signed news and guaranteeing positivity after exponentiation. Its coefficients remain convention-dependent, and positivity does not eliminate separate stability and moment questions.
<!-- bilingual-en:end -->

最后还有两个常见解释越界。[[ARCH-M风险溢价边界|ARCH-M/GARCH-M]]把 $h_t$、$\sqrt{h_t}$ 或 $\log h_t$ 放进均值方程；模型定义不规定风险价格系数必须为正。[[GARCH-X因果边界|GARCH-X]]把变量放进方差方程，只说明当前信息集和规格下的条件二阶矩关联；若变量在预测时点尚未知，甚至不能作为真实一步预测量，更不能仅凭显著系数宣布事件因果。
<!-- bilingual-en:start -->
ARCH-in-mean does not hard-code a positive risk premium, while GARCH-X does not identify causality merely by adding a significant variance covariate. Timing, information availability, and a defensible counterfactual remain necessary.
<!-- bilingual-en:end -->

## 9. 一条可以直接执行的波动建模路径
<!-- bilingual-en:start -->
*9. An executable volatility-modelling workflow*
<!-- bilingual-en:end -->

1. 写清观测频率、预测期限、信息集和业务损失；先画水平、收益与可能的结构断点。
2. 建立并诊断条件均值，保存其创新估计；不要让遗漏均值动态冒充 ARCH。
3. 检查原残差与平方/绝对残差；预先规定滞后，用 ARCH-LM 或 McLeod–Li 判断是否有可建模的幅度依赖。
4. 从简约 ARCH/GARCH 候选开始，明确阶数约定、创新分布、初始化和有效样本。
5. 分别检查正性、严格平稳、有限二阶矩和必要的高阶矩；不要用一个 $\alpha+\beta$ 结论替代全部存在性问题。
6. 估计后同时诊断标准化残差及其平方，再检查尾部、异常点和稳定性；诊断失败就回到均值、方差或断点设定。
7. 只有在完全可比的 likelihood 上使用 AIC/BIC；按真实时间顺序，用预先选择的方差或尾部损失做 rolling-origin 评价。
8. 先核对一步索引，再递推多步；把预测方差、创新分位数、参数不确定性和累计期限风险分开。
9. 只有在对称性诊断或经济问题确有需要时才加入 GJR、EGARCH、ARCH-M 或 GARCH-X，并逐项收紧它们的解释边界。

<!-- bilingual-en:start -->
In practice: define the forecast information and loss, fit an adequate mean, test residual magnitudes, start with parsimonious variance candidates, check existence conditions separately, diagnose standardized residuals at two levels, compare only like-for-like likelihoods, validate at ordered origins, and add asymmetric or covariate extensions only when the data and decision require them.
<!-- bilingual-en:end -->

## 10. 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=151|课程讲义 Lecture 4, pp. 151–177]]：核对课程范围、ARCH/GARCH 递推、LM 检验、似然、残差诊断、预测及扩展模型；本章在一步预测索引、IGARCH 严平稳边界和 9·11 因果措辞处作了明确纠正。
- [Engle (1982)](https://doi.org/10.2307/1912773)：核对 ARCH 定义、条件方差与 LM 检验。
- [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)：核对 GARCH 定义、二阶平稳条件与平方过程结构。
- [Nelson (1990)](https://doi.org/10.1017/S0266466600005296)：核对 GARCH/IGARCH 的严格平稳与矩边界。
- [Bollerslev & Wooldridge (1992)](https://doi.org/10.1080/07474939208800229)：核对非正态 QMLE 与稳健协方差。
- [Nelson (1991)](https://doi.org/10.2307/2938260)、[Glosten, Jagannathan & Runkle (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)、[Zakoïan (1994)](https://doi.org/10.1016/0165-1889(94)90039-6)：核对不对称模型及 TARCH 命名边界。
- [Lamoureux & Lastrapes (1990)](https://doi.org/10.1080/07350015.1990.10509794)、[Patton (2011)](https://doi.org/10.1016/j.jeconom.2010.03.034)：核对结构变化造成的伪持久性，以及含噪波动代理下的预测比较边界。
- [R `stats::AIC` 官方文档](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/AIC.html)、[`arch` 单变量波动模型官方文档](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html)：核对可比最大化 likelihood 的条件，并确认均值动态、波动递推和创新分布均可作为候选差异；固定参数结果不作为最大似然比较证据。

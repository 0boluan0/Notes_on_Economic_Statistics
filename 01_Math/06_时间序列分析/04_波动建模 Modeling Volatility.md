> [!tip] 连续学习入口
> 完整学习路径见 [[04_ARCH-GARCH|ARCH/GARCH：从条件方差到可检验的波动预测]]；全局关系见 [[条件异方差：ARCH 与 GARCH.canvas|主题 Canvas]]。本页继续保留原始课堂展开和例题。
> <!-- bilingual-en:start -->
> Follow [[04_ARCH-GARCH|the continuous ARCH/GARCH learning path]] for the full argument and use the [[条件异方差：ARCH 与 GARCH.canvas|topic Canvas]] for global relationships. This page remains the original classroom record with its examples and figures.
> <!-- bilingual-en:end -->

# 0. 回忆用
<!-- bilingual-en:start -->
*0. Quick recall*
<!-- bilingual-en:end -->

1.

# 1. 引言
<!-- bilingual-en:start -->
*1. Introduction*
<!-- bilingual-en:end -->

本节概览波动建模动机与事实特征，重点在第 2–3 节的 [[ARCH(q)模型|ARCH]]、[[GARCH(p,q)模型|GARCH]] 与拓展模型。主题关系见 [[条件异方差：ARCH 与 GARCH.canvas|条件异方差主题图]]。~~没什么意义.~~
<!-- bilingual-en:start -->
This section surveys the motivation and stylized facts behind volatility modeling. The main material is in Sections 2–3, covering [[ARCH(q)模型|ARCH]], [[GARCH(p,q)模型|GARCH]], and their extensions. The [[条件异方差：ARCH 与 GARCH.canvas|topic map]] shows their relationships. ~~Not especially useful.~~
<!-- bilingual-en:end -->

## 1.1. 为什么要进行波动建模
<!-- bilingual-en:start -->
*1.1. Why model volatility?*
<!-- bilingual-en:end -->

[[条件方差]]是条件二阶中心矩；波动率通常指它的平方根。只有条件均值为零时，条件二阶原点矩才等于条件方差。
<!-- bilingual-en:start -->
[[条件方差|Conditional variance]] is the conditional second central moment, while volatility usually means its square root. The conditional raw second moment equals conditional variance only when the conditional mean is zero.
<!-- bilingual-en:end -->

1. 金融和经济时间序列往往表现出**[[条件异方差]]**（conditional heteroskedasticity）：条件方差随信息集变化，而不是固定常数。
2. **[[波动率聚集|波动性聚集]]**（volatility clustering）指高波动期往往邻接高波动期、低波动期邻接低波动期；它是常见经验模式，不是每个样本都必然出现的定义事实。
<!-- bilingual-en:start -->

&nbsp;
**1.** Financial and economic time series often exhibit **[[条件异方差|conditional heteroskedasticity]]**: conditional variance changes with the information set rather than remaining fixed.<br>
**2.** **[[波动率聚集|Volatility clustering]]** means that high-volatility periods tend to be followed by high-volatility periods, while low-volatility periods tend to follow low-volatility periods. It is a common empirical pattern, not a definitional fact in every sample.<br>
<!-- bilingual-en:end -->

对于波动建模,有三种方法  [[ARCH(q)模型|ARCH]]、[[GARCH(p,q)模型|GARCH]] 和SV
除此之外,还有RV,是对于高频数据的建模.
<!-- bilingual-en:start -->
Three broad approaches to volatility modeling are [[ARCH(q)模型|ARCH]], [[GARCH(p,q)模型|GARCH]], and stochastic volatility (SV). Realized volatility (RV) provides another approach designed for high-frequency data.
<!-- bilingual-en:end -->

## 1.2. 经济学领域的特征事实
<!-- bilingual-en:start -->
*1.2. Stylized facts in economics*
<!-- bilingual-en:end -->

 1.  许多宏观经济序列具有明显的趋势（例如美国实际GDP呈上升趋势）；
 2. **许多时间序列的波动性并不恒定**，会随着时间发生变化。例如，美国实际GDP增速的波动在1984年左右明显下降，在2007年出现了一个大的负向波动尖峰，随后波动性有所稳定 。这表明方差存在结构性变化;
 3. 序列受到冲击后的影响可能具有高度**持久性**（persistence），即冲击的影响在序列中持续很长时间；
 4. 有些金融序列表现出类似随机游走的行为，没有均值回归趋势，例如汇率呈长时间升值或贬值的漫步状（见随机游走模型）；
 5. 一些序列与其他序列存在共移动现象，例如短期和长期利率可能受共同随机趋势驱动；只有在它们各自为 $I(1)$ 且某个非零线性组合为 $I(0)$ 时，才能进一步称为**[[协整秩与共同趋势|协整]]**；
 6. 某些序列存在**结构性突变**，例如金融危机后油价出现跳变。
<!-- bilingual-en:start -->

&nbsp;
**1.** Many macroeconomic series have pronounced trends; for example, real US GDP trends upward.<br>
**2.** **The volatility of many time series is not constant** but changes over time. The volatility of US real-GDP growth, for instance, fell markedly around 1984, showed a large negative spike in 2007, and later stabilized. This pattern suggests structural changes in variance.<br>
**3.** The effects of shocks can be highly **persistent**, remaining in the series for a long time.<br>
**4.** Some financial series, such as exchange rates, behave like random walks and show no tendency to return to a fixed mean.<br>
**5.** Some series move together. Short- and long-term interest rates, for example, may be driven by common stochastic trends; they are [[协整秩与共同趋势|cointegrated]] only if each is $I(1)$ and a nonzero linear combination is $I(0)$.<br>
**6.** Some series undergo **structural breaks**; oil prices, for example, may jump after a financial crisis.<br>
<!-- bilingual-en:end -->

# 2. [[ARCH(q)模型|ARCH]] 与 [[GARCH(p,q)模型|GARCH]]
<!-- bilingual-en:start -->
*2. [[ARCH(q)模型|ARCH]] and [[GARCH(p,q)模型|GARCH]]*
<!-- bilingual-en:end -->

~~加了条件异方差不影响是白噪声~~（说明：白噪声通常指无条件零均值、方差常数且相互不相关；存在条件异方差不影响“不相关”，但对“方差常数”的理解需区分“条件/无条件”层面，常用鞅差序列刻画条件零均值）
<!-- bilingual-en:start -->
~~Conditional heteroskedasticity does not stop the process from being white noise.~~ More precisely, white noise usually means unconditional zero mean, constant unconditional variance, and zero serial correlation. Conditional heteroskedasticity is compatible with zero serial correlation, but the distinction between conditional and unconditional variance must be kept clear. A martingale-difference sequence is often used to express conditional mean zero.
<!-- bilingual-en:end -->

## 2.1. 初步分析
<!-- bilingual-en:start -->
*2.1. Preliminary analysis*
<!-- bilingual-en:end -->

为了刻画波动群聚现象,可以引入一个状态变量:
<!-- bilingual-en:start -->
A state variable can be introduced to represent volatility clustering:
<!-- bilingual-en:end -->

>[!note] 状态变量 State Variable
>状态变量使得方差可以随状态变化。例如，假设模型：$$y_{t+1} = \epsilon_{t+1} x_t,$$其中 $x_t$ 对 $\mathcal F_t$ 可测，并进一步假设 $E(\epsilon_{t+1}\mid\mathcal F_t)=0$、$\operatorname{Var}(\epsilon_{t+1}\mid\mathcal F_t)=\sigma^2$。这比只说二阶**[[白噪声二阶定义|白噪声]]**更强；在这些条件下，$$\operatorname{Var}(y_{t+1} \mid \mathcal{F}_t) = \sigma^2 x_t^2.$$
> <!-- bilingual-en:start -->
> A state variable allows the variance to change with the state. Suppose
> $$y_{t+1}=\epsilon_{t+1}x_t,$$
> where $x_t$ is $\mathcal F_t$-measurable and the innovation satisfies $E(\epsilon_{t+1}\mid\mathcal F_t)=0$ and $\operatorname{Var}(\epsilon_{t+1}\mid\mathcal F_t)=\sigma^2$. These are stronger conditions than second-order [[白噪声二阶定义|white noise]] alone. The conditional variance is then
> $$\operatorname{Var}(y_{t+1}\mid\mathcal{F}_t)=\sigma^2x_t^2.$$
> <!-- bilingual-en:end -->

通过这种方式，如果$x_t$随时间变化，那么$y$的条件方差也会随之变化。当$x_t$较大时，$\sigma^2 x_t^2$也较大，表示波动性提高；当$x_t$较小时，波动性降低。这为捕捉非恒定方差提供了一个思路。
<!-- bilingual-en:start -->
If $x_t$ changes over time, the conditional variance of $y$ changes with it. A large $x_t$ produces a large $\sigma^2x_t^2$ and hence high volatility; a small $x_t$ produces low volatility. This gives a simple mechanism for modeling time-varying variance.
<!-- bilingual-en:end -->

## 2.2. [[ARCH(q)模型|ARCH]]
<!-- bilingual-en:start -->
*2.2. [[ARCH(q)模型|ARCH]]*
<!-- bilingual-en:end -->

>[!note] **ARCH(1)模型定义：**
>$$\epsilon_t = \nu_t \sqrt{\alpha_0 + \alpha_1 \epsilon_{t-1}^2}$$或者写作:$$\begin{cases}
\varepsilon_t = \nu_t \sqrt{h_t}  \\
h_t = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2
\end{cases}$$
>其中${\nu_t}$是一列独立同分布（i.i.d.）的随机变量，满足
>$\mathbb{E}(\nu_t)=0, \operatorname{Var}(\nu_t)=1$。
>$\alpha_0$和$\alpha_1$为常数参数。$\alpha_0>0$、$\alpha_1\ge0$是保证线性方差递推非负的常用条件；$\alpha_1<1$保证有限无条件二阶矩与协方差平稳。严格平稳另由相应 log-moment 条件判断，不能与二阶矩条件混写。
>$h_t$表示$\varepsilon_t$在$t$期的条件方差（即$h_t = \mathrm{Var}(\varepsilon_t \mid \mathcal{F}_{t-1})$）。
>这里$\epsilon_t$是均值已滤除后的**创新**；模型拟合后得到的 residual 只是它的样本估计。ARCH 表示创新的条件方差并不恒定，而是由上一期创新平方$\epsilon_{t-1}^2$更新。
> <!-- bilingual-en:start -->
> $$\epsilon_t = \nu_t \sqrt{\alpha_0 + \alpha_1 \epsilon_{t-1}^2},$$
> or, equivalently,
> $$\begin{cases}
> \varepsilon_t = \nu_t \sqrt{h_t}  \\
> h_t = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2.
> \end{cases}$$
> The standardized shocks $\{\nu_t\}$ are i.i.d. random variables satisfying $\mathbb{E}(\nu_t)=0$ and $\operatorname{Var}(\nu_t)=1$. The usual linear positivity conditions are $\alpha_0>0$ and $\alpha_1\ge0$; $\alpha_1<1$ gives a finite unconditional second moment and covariance stationarity. Strict stationarity instead uses the relevant log-moment condition. The quantity $h_t$ is the conditional variance of the mean innovation $\varepsilon_t$: $h_t=\mathrm{Var}(\varepsilon_t\mid\mathcal{F}_{t-1})$. A fitted residual estimates this latent innovation; it is not the same object by definition. ARCH lets the innovation variance vary with the previous squared innovation.
> <!-- bilingual-en:end -->

所以==它是一个鞅差分==.因为t-1期的所有项在算期望的时候都能提出来.
<!-- bilingual-en:start -->
It is therefore ==a martingale-difference sequence==: conditional on information through $t-1$, all lagged terms are known and can be taken outside the conditional expectation.
<!-- bilingual-en:end -->

<span style="color: yellow;">关键</span>: 在ARCH(1)中，$\epsilon_t$的t-1期条件方差为$\alpha_0 + \alpha_1 \epsilon_{t-1}^2$。当$\alpha_1<1$且二阶矩有限时，再取期望得到$\bar{h} = \frac{\alpha_0}{1-\alpha_1}$。
<!-- bilingual-en:start -->
<span style="color: yellow;">Key point</span>: in ARCH(1), the time-$(t-1)$ conditional variance of $\epsilon_t$ is $\alpha_0+\alpha_1\epsilon_{t-1}^2$. When $\alpha_1<1$ and the second moment is finite, taking one more expectation gives $\bar h=\frac{\alpha_0}{1-\alpha_1}$.
<!-- bilingual-en:end -->

==常数项 $\alpha_0$ 不能在保留正的有限无条件方差时随意删除。若方差递推改成 $h_t=\alpha_1\epsilon_{t-1}^2$，对两侧取无条件期望会迫使 $\alpha_1=1$；若二阶矩不存在，则这一步本身不能使用。==
<!-- bilingual-en:start -->
==The constant $\alpha_0$ cannot simply be removed. If the variance recursion were $h_t=\alpha_1\epsilon_{t-1}^2$, taking unconditional expectations on both sides would force $\alpha_1=1$ whenever the variance is positive and finite.==
<!-- bilingual-en:end -->

## 2.3. [[GARCH(p,q)模型|GARCH]]
<!-- bilingual-en:start -->
*2.3. [[GARCH(p,q)模型|GARCH]]*
<!-- bilingual-en:end -->

ARCH的N要取得比较大.所以发明了GARCH模型,在保持对条件异方差性建模能力的同时，用更少的参数捕捉长期的波动影响。
<!-- bilingual-en:start -->
An ARCH model may require a large lag order. GARCH was developed to capture persistent volatility with fewer parameters while retaining a model of conditional heteroskedasticity.
<!-- bilingual-en:end -->

>[!note] GARCH(p,q)模型
>$$\begin{cases}
> \varepsilon_t = \nu_t \sqrt{h_t}  \\
> h_t = \alpha_0 + \sum_{i=1}^{q} \alpha_i\varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j h_{t-j}
> \end{cases}$$
> **其中$h_t$依赖于$q$阶误差平方**和$p$阶**自身滞后**。$\alpha_0>0$、$\alpha_i\ge0$、$\beta_j\ge0$是保证线性递推非负的常用充分条件；$\sum_i\alpha_i+\sum_j\beta_j<1$是这里采用的有限二阶矩/协方差平稳条件，不是一般严格平稳条件。
>
> <!-- bilingual-en:start -->
> $$\begin{cases}
> \varepsilon_t = \nu_t \sqrt{h_t}  \\
> h_t = \alpha_0 + \sum_{i=1}^{q} \alpha_i\varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j h_{t-j}.
> \end{cases}$$
> The conditional variance $h_t$ depends on $q$ lags of squared innovations and $p$ lags of itself. Standard sufficient positivity restrictions are $\alpha_0>0$, $\alpha_i\geq0$, and $\beta_j\geq0$. The condition $\sum_i\alpha_i+\sum_j\beta_j<1$ is the finite-second-moment/covariance-stationarity condition used here, not the general strict-stationarity condition.
> <!-- bilingual-en:end -->

GARCH模型通常能够用更少的滞后项近似高阶ARCH的持续波动，因此更**参数节省**（parsimonious）。标准 GARCH 的二阶冲击通常几何衰减；这类高持久性不应自动称为真正的长记忆。
<!-- bilingual-en:start -->
A GARCH model can often approximate high-order ARCH persistence with far fewer lags and is therefore more **parsimonious**. Standard GARCH persistence usually decays geometrically and should not automatically be labelled genuine long memory.
<!-- bilingual-en:end -->

条件方差:$\mathbb{E}(\varepsilon_t^2 \mid \mathcal{F}_{t-1}) = h_t = \alpha_0 + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^p \beta_j h_{t-j}$.
若 $\sum_{i=1}^q \alpha_i + \sum_{j=1}^p \beta_j < 1$，则无条件二阶矩（长期平均方差）有限，记作：${ \mathbb{E}(\varepsilon_t^2) = \frac{\alpha_0}{1 - \sum_{i=1}^q \alpha_i - \sum_{j=1}^p \beta_j} }$。GARCH(1,1) 的严格平稳与有限二阶矩区别见 [[GARCH严格平稳条件|GARCH(1,1) 严平稳条件]]；一般高阶 GARCH 需用随机矩阵乘积的条件，不能直接套用该标量公式。
<!-- bilingual-en:start -->
The conditional variance is
$\mathbb{E}(\varepsilon_t^2\mid\mathcal{F}_{t-1})=h_t=\alpha_0+\sum_{i=1}^q\alpha_i\varepsilon_{t-i}^2+\sum_{j=1}^p\beta_jh_{t-j}$.
If $\sum_{i=1}^q\alpha_i+\sum_{j=1}^p\beta_j<1$, the process is covariance-stationary and has the finite unconditional second moment
${\mathbb{E}(\varepsilon_t^2)=\frac{\alpha_0}{1-\sum_{i=1}^q\alpha_i-\sum_{j=1}^p\beta_j}}$.
For GARCH(1,1), [[GARCH严格平稳条件|strict stationarity uses a separate log-moment condition]]. Higher-order GARCH models require the corresponding random-matrix product condition rather than an unqualified reuse of the scalar GARCH(1,1) formula.
<!-- bilingual-en:end -->

实证分析中最常用的就是GARCH(1,1)
<!-- bilingual-en:start -->
GARCH(1,1) is the specification used most often in empirical work.
<!-- bilingual-en:end -->

## 2.4. 侦测 ARCH/GARCH 效应
<!-- bilingual-en:start -->
*2.4. Detecting ARCH/GARCH effects*
<!-- bilingual-en:end -->

参见：[[ARCH-LM检验|ARCH-LM]] 与 [[McLeod-Li检验|McLeod–Li]]。

在对时间序列进行建模时，我们首先常用ARMA模型拟合均值部分，然后需要判断残差序列中是否存在ARCH/GARCH效应（即条件异方差）。
<!-- bilingual-en:start -->
When modeling a time series, an ARMA model is often fitted first to describe the conditional mean. The next question is whether the resulting residuals contain ARCH/GARCH effects—that is, conditional heteroskedasticity.
<!-- bilingual-en:end -->

使用两种方法检验残差,两种方法都是在对原始序列先拟合一个最好的ARMA模型,并得到一个残差序列${\hat{\varepsilon}_t}$.而后对残差序列进行操作
<!-- bilingual-en:start -->
Two common residual tests begin in the same way: fit an adequate ARMA model to the original series, obtain the residual sequence $\hat{\varepsilon}_t$, and then examine functions of those residuals.
<!-- bilingual-en:end -->

>[!note] [[McLeod-Li检验|McLeod–Li 检验]]
> 拟合一个充分诊断过的[[ARMA(p,q)模型|ARMA 均值模型]]，得到残差序列 $\hat{\varepsilon}_t$
>
>  对残差序列平方$\hat{\varepsilon}_t^2$，计算其样本自相关.定义第 i 阶自相关：
> $$r_i = \frac{\sum_{t=i+1}^{T} (\hat{\varepsilon}_t^2 - \bar{\sigma}^2)(\hat{\varepsilon}_{t-i}^2 - \bar{\sigma}^2)}{\sum_{t=1}^{T} (\hat{\varepsilon}_t^2 - \bar{\sigma}^2)^2}$$
> 其中 $\bar{\sigma}^2 = \frac{1}{T} \sum \hat{\varepsilon}_t^2$ 是残差平方均值。
>
> 而后使用样本自相关构建检验统计量
>
> $$Q = T(T+2) \sum_{i=1}^m \frac{r_i^2}{T - i},$$
> 在相应矩与正则条件下，用合适的渐近 $\chi^2$ 参考分布校准。拟合 ARMA 后的自由度、有限样本修正和高阶矩要求依实现而异，详见 [[McLeod-Li检验|检验边界]]。
>
> - 如果显著 ⇒ 拒绝平方残差前 $m$ 阶自相关全为零，支持存在 ARCH 型幅度依赖；
> - 若不显著 ⇒ 只表示在所选滞后与当前样本下没有检测到这类平方相关，不能排除所有条件异方差。
> <!-- bilingual-en:start -->
> Fit an adequate [[ARMA(p,q)模型|ARMA mean model]] and obtain residuals $\hat{\varepsilon}_t$.
>
> Square the residuals and calculate their sample autocorrelations. The lag-$i$ autocorrelation is
> $$r_i = \frac{\sum_{t=i+1}^{T} (\hat{\varepsilon}_t^2 - \bar{\sigma}^2)(\hat{\varepsilon}_{t-i}^2 - \bar{\sigma}^2)}{\sum_{t=1}^{T} (\hat{\varepsilon}_t^2 - \bar{\sigma}^2)^2},$$
> where $\bar{\sigma}^2=\frac{1}{T}\sum\hat{\varepsilon}_t^2$ is the mean squared residual.
>
> Use these autocorrelations to construct
>
> $$Q = T(T+2) \sum_{i=1}^m \frac{r_i^2}{T - i},$$
>
> Under the relevant moment and regularity conditions, it is calibrated against an asymptotic chi-square reference distribution. Degrees-of-freedom adjustments after fitting ARMA, finite-sample corrections, and higher-moment requirements depend on the implementation; see [[McLeod-Li检验|the diagnostic boundary]].
>
> - A significant result rejects the joint zero-autocorrelation null and supports [[ARCH(q)模型|ARCH]]-type magnitude dependence.
> - A nonsignificant result says only that this squared-residual dependence was not detected at the tested lags; it does not exclude every form of conditional heteroskedasticity.
> <!-- bilingual-en:end -->

>[!note] [[ARCH-LM检验|ARCH-LM 检验]]
>同样先拟合一个 ARMA 模型 ⇒ 得到残差 $\hat{\varepsilon}_t$
> 使用残差做回归：
> $$\hat{\varepsilon}_t^2 = \alpha_0 + \sum_{j=1}^{q} \alpha_j \hat{\varepsilon}_{t-j}^2 + \eta_t$$
> 这个是检验是否存在 ARCH(q) 的标准形式。
> 检验思想：
> - 原假设 $H_0$: $\alpha_1 = \alpha_2 = \cdots = \alpha_q = 0$（无 ARCH）
> - 计算 R^2：这个回归的决定系数
> - 构造统计量：$L = T R^2 \sim \chi^2_q$
> 	- 如果 L 显著 ⇒ 存在 ARCH 效应
> 	- 若不显著 ⇒ 在所选阶数与样本下没有检测到该 ARCH 效应，不能当作所有异方差均不存在
> <!-- bilingual-en:start -->
> Again, first fit an ARMA model and obtain residuals $\hat{\varepsilon}_t$. Then run the auxiliary regression
> $$\hat{\varepsilon}_t^2 = \alpha_0 + \sum_{j=1}^{q} \alpha_j \hat{\varepsilon}_{t-j}^2 + \eta_t.$$
> This is the standard test for an ARCH($q$) effect.
> - The null is $H_0:\alpha_1=\alpha_2=\cdots=\alpha_q=0$, meaning no ARCH effect.
> - Calculate the auxiliary regression's $R^2$.
> - Form the statistic $L=TR^2\sim\chi_q^2$ under the null.
> - A significant $L$ indicates an ARCH effect.
> - A nonsignificant $L$ provides no evidence of an ARCH effect at the chosen order.
> <!-- bilingual-en:end -->

## 2.5. 极大似然估计MLE
<!-- bilingual-en:start -->
*2.5. Maximum-likelihood estimation (MLE)*
<!-- bilingual-en:end -->

参见：[[GARCH条件似然|条件似然与初值]]；把 Gaussian likelihood 当作准似然时还要看 [[Gaussian QMLE|Gaussian QMLE 与稳健标准误]]。

根据正态密度函数写出条件似然：
<!-- bilingual-en:start -->
Under conditionally Gaussian innovations, write the conditional likelihood as
<!-- bilingual-en:end -->

$L = \prod_{t=1}^T \left( \frac{1}{\sqrt{2\pi h_t}} \exp\left( -\frac{\varepsilon_t^2}{2h_t} \right) \right)$
对数似然为：
<!-- bilingual-en:start -->
The log-likelihood is
<!-- bilingual-en:end -->
$\log L = -\frac{T}{2} \log(2\pi) - \frac{1}{2} \sum_{t=1}^T \log h_t - \frac{1}{2} \sum_{t=1}^T \frac{\varepsilon_t^2}{h_t}$

总之原理是这么个原理.不能手动算的.别管了
<!-- bilingual-en:start -->
That is the principle. In practice, the likelihood is optimized numerically rather than by hand.
<!-- bilingual-en:end -->

## 2.6. 评估拟合
<!-- bilingual-en:start -->
*2.6. Evaluating the fit*
<!-- bilingual-en:end -->

### (1) 模型拟合优度的评估:AIC和SBC.
<!-- bilingual-en:start -->
*(1) Comparing model fit with AIC and SBC/BIC*
<!-- bilingual-en:end -->

• [[AIC]] 定义为：$\displaystyle \text{AIC} = -2\ln L_{\text{max}} + 2k$，其中$L_{\text{max}}$是模型最大化后的似然值，$k$是模型中估计参数的个数 。$-2\ln L$衡量 likelihood fit，而$2k$是对模型复杂度的惩罚（参数越多惩罚越大）。
<!-- bilingual-en:start -->
• [[AIC]] is $\displaystyle \text{AIC}=-2\ln L_{\text{max}}+2k$, where $L_{\text{max}}$ is the maximized likelihood and $k$ is the number of estimated parameters. The first term rewards fit, while $2k$ penalizes complexity; smaller values are preferred when comparing models fitted to the same data.
<!-- bilingual-en:end -->

• [[BIC]] 定义为：$\displaystyle \text{BIC} = -2\ln L_{\text{max}} + k \ln T$，其中$T$为样本容量 。相比AIC，BIC对参数个数的惩罚更严厉（乘以$\ln T$因子），在大样本下倾向于选择更简洁的模型。
<!-- bilingual-en:start -->
• [[BIC]], also called SBC, is $\displaystyle \text{BIC}=-2\ln L_{\text{max}}+k\ln T$, where $T$ is the sample size. It penalizes additional parameters more strongly than AIC when $T$ is large and therefore tends to select more parsimonious models.
<!-- bilingual-en:end -->

### (2) 模型诊断
<!-- bilingual-en:start -->
*(2) Model diagnostics*
<!-- bilingual-en:end -->

 使用经过ARMA-GARCH模型预测后的残差计算标准化残差 $s_t = \frac{\hat{\varepsilon}_t}{\sqrt{\hat{h}_t}}$，其中$\hat{\varepsilon}_t$是模型估计后的残差，$\hat{h}_t$是对应的拟合条件方差。理论上，如果均值模型和波动模型都正确，那么$s_t$应当是一个i.i.d.标准正态序列（在假定正态创新的情形下）。
<!-- bilingual-en:start -->
Compute standardized residuals $s_t=\frac{\hat{\varepsilon}_t}{\sqrt{\hat h_t}}$ from the fitted ARMA–GARCH model, where $\hat{\varepsilon}_t$ is the estimated residual and $\hat h_t$ is its fitted conditional variance. If both the mean and volatility specifications are correct, $s_t$ should be i.i.d. standard normal under the assumed Gaussian innovation distribution.
<!-- bilingual-en:end -->

对标准化残差进行[[Ljung-Box检验|白噪声联合诊断]]，并分别检查其平方是否仍有 ARCH 结构；完整分工见 [[GARCH残差双层诊断|GARCH 残差诊断]]。
<!-- bilingual-en:start -->
Apply white-noise diagnostics to the standardized residuals.
<!-- bilingual-en:end -->

## 2.7. 预测方差
<!-- bilingual-en:start -->
*2.7. Forecasting variance*
<!-- bilingual-en:end -->

可以进行均值预测和方差预测,均值部分和前面ARMA一样.
<!-- bilingual-en:start -->
Both the conditional mean and conditional variance can be forecast. The mean forecast follows the same ARMA procedure introduced earlier.
<!-- bilingual-en:end -->

有了GARCH模型，我们可以预测下一期的波动水平，即计算$h_{t+1|t} = E_t[h_{t+1}]$（下标$t+1|t$表示在$t$期基于信息$\mathcal{F}_t$对$t+1$期的预测）。以GARCH(1,1)为例，根据模型：
<!-- bilingual-en:start -->
A GARCH model can forecast next period's volatility by calculating $h_{t+1|t}=E_t[h_{t+1}]$, where $t+1|t$ means a forecast for $t+1$ based on information $\mathcal F_t$ available at time $t$. For GARCH(1,1),
<!-- bilingual-en:end -->
$$h_{t+1} = \alpha_0 + \alpha_1 \varepsilon_t^2 + \beta_1 h_t$$
在$t$时刻已知$\varepsilon_t$和$h_t$，则**一步前方差预测**为：
<!-- bilingual-en:start -->
Because $\varepsilon_t$ and $h_t$ are known at time $t$, the **one-step-ahead variance forecast** is
<!-- bilingual-en:end -->
$$\hat{h}_{t+1|t} = \alpha_0 + \alpha_1 \varepsilon_t^2 + \beta_1 h_t$$

这实际上就是把当期已发生的冲击 $\varepsilon_t^2$ 代入，对下一期进行更新。若进一步假设标准化创新条件 Gaussian，并暂把参数和滤波状态当作已知，下一观测的 95% **预测区间**可写为：
<!-- bilingual-en:start -->
This forecast updates next period's variance using the shock $\varepsilon_t^2$ observed in the current period. A Gaussian 95% prediction interval for the next observation can be written as
<!-- bilingual-en:end -->

$$\hat{y}_{t+1|t} \pm 1.96 \sqrt{\hat{h}_{t+1|t}} $$

与固定方差情形不同的是，这个区间的宽度 $\sqrt{\hat{h}_{t+1|t}}$ 动态变化：高波动期更宽，低波动期更窄。实际 plug-in 区间还使用估计参数、fitted innovation 与滤波方差；厚尾、偏态和参数不确定性会改变覆盖率，不能只由原残差不相关推出 Gaussian 覆盖。
<!-- bilingual-en:start -->
Unlike a constant-variance interval, its width $\sqrt{\hat h_{t+1|t}}$ changes over time. This Gaussian plug-in interval treats parameters and filtered states as known; heavy tails, asymmetry, and parameter uncertainty can alter coverage and must be checked separately.
<!-- bilingual-en:end -->

如果是更多步的预测，就从一步预测递推；有限长期方差下回归长期均值，索引与半衰期边界见 [[GARCH多步方差预测|GARCH 多步方差预测]]。课程讲义原图把一步更新写成了 $\varepsilon_{t-1}^2$；按 $\mathcal F_t$ 预测 $t+1$ 时应使用已观测的 $\varepsilon_t^2$。
<!-- bilingual-en:start -->
Multi-step forecasts are obtained recursively; see [[GARCH多步方差预测|the indexing and half-life boundary]]. The original slide's one-step expression used $\varepsilon_{t-1}^2$; a forecast of $t+1$ from $\mathcal F_t$ must use the observed $\varepsilon_t^2$.
<!-- bilingual-en:end -->

# 3. 扩展模型
<!-- bilingual-en:start -->
*3. Extended models*
<!-- bilingual-en:end -->

==极大概率不考,如果没时间了就别学了.==
<!-- bilingual-en:start -->
==This is very unlikely to be examined; skip it if time is short.==
<!-- bilingual-en:end -->

## 3.1. [[IGARCH平稳与矩边界|IGARCH]]
<!-- bilingual-en:start -->
*3.1. [[IGARCH平稳与矩边界|IGARCH]]*
<!-- bilingual-en:end -->

参见：[[IGARCH平稳与矩边界|IGARCH 的矩与平稳边界]]

金融时间序列的一个典型特征是波动性的**高度持久**（persistent）。实证中，对许多资产回报率拟合GARCH(1,1)模型时，常常发现估计得到的$\hat{\alpha}_1 + \hat{\beta}_1$非常接近1。
<!-- bilingual-en:start -->
A typical feature of financial time series is highly **persistent** volatility. Empirical GARCH(1,1) estimates for many asset returns yield $\hat\alpha_1+\hat\beta_1$ very close to one.
<!-- bilingual-en:end -->

>[!note] **积整GARCH模型**（Integrated GARCH，简称IGARCH）。
>$\alpha_1 + \beta_1 = 1$的GARCH模型.
>IGARCH(1,1)实际上就是$\alpha_1 + \beta_1 = 1$的GARCH(1,1)模型。这个等式让原先的GARCH模型少了一个参数
>
>其特性为:
>1. $\omega>0$ 时没有有限无条件方差；
>2. 多步预测不回归有限长期方差，GARCH(1,1) 下为 $h_{t+j|t}=h_{t+1|t}+(j-1)\omega$；
>3. 这并不自动否定严格平稳：非退化 IGARCH 仍可能满足 log-moment 严平稳条件，只是二阶矩发散。
> <!-- bilingual-en:start -->
> An IGARCH(1,1) model is a GARCH(1,1) model satisfying $\alpha_1+\beta_1=1$. The equality removes one free parameter from the original model.
>
> Its main implications are:
> **1.** With $\omega>0$, no finite unconditional variance exists.<br>
> **2.** Multi-step forecasts do not return to a finite long-run variance; in GARCH(1,1), $h_{t+j|t}=h_{t+1|t}+(j-1)\omega$.<br>
> **3.** This does not automatically rule out strict stationarity: a non-degenerate IGARCH process may satisfy the log-moment stationarity condition while lacking a finite second moment.<br>
> <!-- bilingual-en:end -->

## 3.2. [[ARCH-M风险溢价边界|ARCH-M]]
<!-- bilingual-en:start -->
*3.2. ARCH-in-mean*
<!-- bilingual-en:end -->

参见：[[ARCH-M风险溢价边界|ARCH-M 的符号边界]]

ARCH-M 把条件风险量放进均值方程；“风险越大，预期回报越高”是待检验的经济假说，不是模型定义。
<!-- bilingual-en:start -->
ARCH-M places a conditional risk measure in the mean equation. A positive risk–return relation is an economic hypothesis to estimate, not part of the model's definition.
<!-- bilingual-en:end -->

>[!note] ARCH-M
>$$\begin{cases} y_t &= \mu_t + \varepsilon_t \\ \mu_t &= \beta + \delta h_t \\ h_t &= \alpha_0 + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 \end{cases}$$
> <!-- bilingual-en:start -->
> $$\begin{cases} y_t &= \mu_t + \varepsilon_t \\ \mu_t &= \beta + \delta h_t \\ h_t &= \alpha_0 + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2. \end{cases}$$
> <!-- bilingual-en:end -->

| 方程                                                   | **含义**                     |
| ---------------------------------------------------- | -------------------------- |
| $y_t = \mu_t + \varepsilon_t$                        | 观测值等于期望值 + 噪声              |
| $\mu_t = \beta + \delta h_t$                         | 条件风险量进入均值；影响方向由 $\delta$ 的估计与推断决定 |
| $h_t = \alpha_0 + \sum \alpha_i \varepsilon_{t-i}^2$ | 标准 [[ARCH(q)模型|ARCH]](q) 波动结构            |
<!-- bilingual-en:start -->
| Equation | **Meaning** |
| --- | --- |
| $y_t=\mu_t+\varepsilon_t$ | The observation equals its conditional mean plus noise. |
| $\mu_t=\beta+\delta h_t$ | Conditional risk enters the mean; the sign must be estimated and interpreted. |
| $h_t=\alpha_0+\sum\alpha_i\varepsilon_{t-i}^2$ | A standard [[ARCH(q)模型|ARCH]]($q$) volatility equation. |
<!-- bilingual-en:end -->

## 3.3. 带有解释变量的波动模型
<!-- bilingual-en:start -->
*3.3. Volatility models with explanatory variables*
<!-- bilingual-en:end -->

可把事件哑变量或其他已知协变量加入方差方程，描述它们与条件二阶矩的关系；系数显著本身不识别经济因果。
<!-- bilingual-en:start -->
Event dummies or other known covariates can be added to the variance equation to describe their association with conditional second moments. A significant coefficient alone does not identify an economic cause.
<!-- bilingual-en:end -->

>[!example] 示例:衡量911事件前后的波动
> 我们想检验“9·11事件”前后模型中的条件方差水平是否存在与事件窗口相关的变化。可在GARCH方差方程中加入事件哑变量$D_t$：
>
> $$h_t = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 h_{t-1} + \gamma D_t,$$
>
> 其中$D_t$在2001年9月11日前为0、当日及以后为1。这里把 $D_t$ 作为**事后编码**的日期变量来描述断点关联；若 $h_t$ 定义为给定 $\mathcal F_{t-1}$ 的真实一步预测，右侧回归量必须在 $t-1$ 已知，突发事件当日 dummy 不能假装为事前信息。$\gamma$描述控制该 GARCH 规格后与该断点共同出现的条件方差水平差异；它仍可能混入同期冲击、均值错设或结构变化，不能仅凭这一回归断言“9·11导致了多少波动”。详见 [[GARCH-X因果边界|方差协变量的因果边界]]。
> <!-- bilingual-en:start -->
> Suppose we want to test whether the fitted conditional variance has a level shift associated with the 11 September 2001 event window. Add an event dummy $D_t$ to the GARCH variance equation:
>
> $$h_t = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 h_{t-1} + \gamma D_t.$$
>
> Set $D_t=0$ before 11 September 2001 and $D_t=1$ on and after that date. Here it is an **ex-post coded** date regressor: if $h_t$ is a genuine forecast conditional on $\mathcal F_{t-1}$, every right-hand-side regressor must be known at $t-1$, so an unexpected event-day dummy cannot be treated as advance information. The coefficient $\gamma$ describes the conditional-variance level difference associated with that break after controlling for this GARCH specification. Concurrent shocks, mean misspecification, and other breaks prevent the coefficient alone from identifying the event's causal effect; see [[GARCH-X因果边界|the causal boundary]].
> <!-- bilingual-en:end -->

## 3.4. 非对称模型：[[GJR-GARCH模型|GJR-GARCH]]、[[EGARCH模型|EGARCH]]
<!-- bilingual-en:start -->
*3.4. Asymmetric models: [[GJR-GARCH模型|GJR-GARCH]] and [[EGARCH模型|EGARCH]]*
<!-- bilingual-en:end -->

>[!note] [[波动不对称与杠杆|波动不对称与杠杆机制]]
>大小相同、符号相反的冲击对未来方差影响不同，是统计不对称；公司负面信息通过资本结构提高权益风险只是可能的杠杆解释，需要额外证据。
> <!-- bilingual-en:start -->
> An asymmetric volatility response is a statistical pattern. The balance-sheet leverage mechanism is one possible economic explanation, and requires separate evidence.
> <!-- bilingual-en:end -->

>[!note] [[GJR-GARCH模型|GJR-GARCH 模型]]与[[TARCH命名边界|TARCH 命名边界]]
> 这里课程所谓 TARCH 使用的是 GJR-GARCH 的**方差递推**：
>
> $$h_t = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \lambda_1d_{t-1}\epsilon_{t-1}^2 + \beta_1 h_{t-1},$$
>
> ==其中$d_{t-1}$是一个哑变量==，当$\epsilon_{t-1}<0$时为1，否则为0。正冲击的平方系数是$\alpha_1$，负冲击的是$\alpha_1+\lambda_1$。常用正性条件要求$\alpha_1\ge0$且$\alpha_1+\lambda_1\ge0$；在标准化创新对称时，有限二阶矩常用条件为$\alpha_1+\beta_1+\lambda_1/2<1$。其他文献也把递推条件标准差的 Zakoïan 模型称为 TARCH，因此必须先读公式再搬参数解释。
> <!-- bilingual-en:start -->
> The course's TARCH label here refers to the GJR-GARCH variance recursion
>
> $$h_t = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \lambda_1d_{t-1}\epsilon_{t-1}^2 + \beta_1 h_{t-1}.$$
>
> ==Here $d_{t-1}$ is a dummy variable==: it equals one for a negative shock and zero otherwise. Positivity commonly requires $\alpha_1\ge0$ and $\alpha_1+\lambda_1\ge0$; under symmetric standardized innovations, the usual finite-second-moment condition is $\alpha_1+\beta_1+\lambda_1/2<1$. Some sources use TARCH for Zakoïan's conditional-standard-deviation recursion, so the equation—not the label—determines the restrictions and interpretation.
> <!-- bilingual-en:end -->

>[!note] [[EGARCH模型|EGARCH 指数 GARCH 模型]]
>令$z_{t-1}=\epsilon_{t-1}/\sqrt{h_{t-1}}$。一种常见 EGARCH(1,1) 记号是：
>
> $$\ln h_t = \omega + \beta\ln h_{t-1}+\alpha\bigl(|z_{t-1}|-E|z_{t-1}|\bigr)+\gamma z_{t-1}.$$
>
> 对数方差经指数化后自动为正，无需把所有系数限制为非负。中心化幅度项$|z|-E|z|$把“冲击有多大”与截距分开；有符号项$\gamma z$表示正负冲击差异。在本记号下$\gamma<0$意味着负冲击提高 log variance 更多；换用$-\gamma z$时符号解释会反转，所以必须连同方程解释系数。
>
> <!-- bilingual-en:start -->
> EGARCH uses a log-variance equation. One common form is
>
> $$\ln h_t = \omega + \beta\ln h_{t-1}+\alpha\bigl(|z_{t-1}|-E|z_{t-1}|\bigr)+\gamma z_{t-1},\qquad z_{t-1}=\frac{\epsilon_{t-1}}{\sqrt{h_{t-1}}}.$$
>
> Modeling $\ln h_t$ guarantees a positive variance after exponentiation without standard GARCH's coefficient nonnegativity restrictions. The centred magnitude term separates shock size from the intercept, while $\gamma z_{t-1}$ captures sign asymmetry. Under this convention, $\gamma<0$ gives a larger log-variance response to a negative shock; the interpretation reverses if the equation uses $-\gamma z$, so signs must always be read with the equation.
> <!-- bilingual-en:end -->

# 4. 关联卡片

- Volatility Modeling-hub
- [[条件异方差：ARCH 与 GARCH.canvas|Conditional Heteroskedasticity topic map]]
- [[条件方差|Conditional Variance]]
- [[条件异方差|Conditional Heteroskedasticity]]
- [[条件尺度与标准化冲击|Conditional Scale and Standardised Shocks]]
- [[波动率聚集|Volatility Clustering]]
- [[ARCH(q)模型|ARCH]]
- [[GARCH(p,q)模型|GARCH]]
- [[ARCH-LM检验|ARCH-LM Test]]
- [[McLeod-Li检验|McLeod–Li Test]]
- [[GARCH条件似然|GARCH Estimation and Initialisation]]
- [[GARCH残差双层诊断|Standardised-residual Diagnostics]]
- [[IGARCH平稳与矩边界|IGARCH]]
- [[ARCH-M风险溢价边界|ARCH-M]]
- [[GJR-GARCH模型|GJR-GARCH]]
- [[TARCH命名边界|TARCH Naming Boundary]]
- [[EGARCH模型|EGARCH]]

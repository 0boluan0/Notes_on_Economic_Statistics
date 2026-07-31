
# 0. 回忆用
<!-- bilingual-en:start -->
*0. Recall*
<!-- bilingual-en:end -->

# 1. 引言
<!-- bilingual-en:start -->
*1. Introduction*
<!-- bilingual-en:end -->

==废话==
<!-- bilingual-en:start -->
==Background remarks==
<!-- bilingual-en:end -->

## 1.1. 干预分析
<!-- bilingual-en:start -->
*1.1. Intervention analysis*
<!-- bilingual-en:end -->

>[!note] 干预分析
>
为正式检验金属探测安检政策的效果，我们建立一个包含干预变量的时间序列模型。例如，用 $y_t$ 表示$t$期发生的劫机事件数量，引入干预哑变量$z_t$来表示政策实施：1973年第一季度（含）之后 $z_t=1$，之前 $z_t=0$。一个简单的干预模型可以写作：
>
> $$
>
> y_t = a_0 + a_1 y_{t-1} + c_0 z_t + \varepsilon_t,\quad |a_1|<1。
>
> $$
>
> 这里，$a_0$ 是常数项，$a_1$ 是劫机次数的**[[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|AR]](1)系数**（表示过去劫机次数对当前的影响），$c_0$ 是干预效应系数，$\varepsilon_t$ 是白噪声误差项。模型中包含一个滞后项$y_{t-1}$，假定序列在没有干预时服从AR(1)动态，并以$c_0 z_t$项刻画政策干预的即时影响。
> <!-- bilingual-en:start -->
> ** intervention analysis **
> To formally test the effect of metal detection security policy, we establish a time series model including intervention variables.  For example, $y_t$ is used to denote the number of hijackings that occurred during the $t$ period, and the intervention dummy variable $z_t$ is introduced to denote policy implementation: $z_t=1$ after Q1 1973, $z_t=0$ before.  A simple intervention model can be written:
> Here, $a_0$ is the constant, $a_1$ is the **[[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|AR(1)]] coefficient** governing how the previous period's hijacking count affects the current count, $c_0$ measures the intervention effect, and $\varepsilon_t$ is a white-noise error. The lagged term $y_{t-1}$ captures the series' ordinary AR(1) dynamics, while $c_0z_t$ captures the policy's immediate effect.
> <!-- bilingual-en:end -->

• 如果没有干预（$z_t=0$），模型退化为 $y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$，是一个均值为 $a_0/(1-a_1)$ 的稳定AR(1)过程。
<!-- bilingual-en:start -->
- Without intervention ($z_t=0$), the model reduces to $y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$, a stationary AR(1) process with mean $a_0/(1-a_1)$.
<!-- bilingual-en:end -->

• 干预后（$z_t=1$），$c_0$ 捕捉了政策对劫机数量的直接影响：$c_0$为负意味着政策使劫机次数立即降低，为正则表示提高。
<!-- bilingual-en:start -->
- After the intervention ($z_t=1$), $c_0$ captures the policy's direct effect on the number of hijackings: $c_0<0$ means an immediate reduction, while $c_0>0$ means an immediate increase.
<!-- bilingual-en:end -->

**模型求解与长期水平变化**
<!-- bilingual-en:start -->
** Model Solving and Long-term Horizontal Variation **
<!-- bilingual-en:end -->

为了量化干预效应，我们可以**求解**上述方程，表示出 $y_t$ 关于冲击和干预项的无限滞后表示形式。因为这是一个AR(1)过程（$|a_1|<1$保证平稳），我们可以向前迭代展开：
<!-- bilingual-en:start -->
To quantify the intervention effect, solve the equation above for the infinite distributed-lag representation of $y_t$ in terms of shocks and the intervention. Because this is an AR(1) process and $|a_1|<1$ ensures stationarity, we can expand it recursively:
<!-- bilingual-en:end -->

通过递归替换 $y_{t-1}$，得到：
<!-- bilingual-en:start -->
By replacing $y_{t-1}$ recursively, we get:
<!-- bilingual-en:end -->

$$
y_t = \frac{a_0}{1 - a_1} + c_0 \sum_{i=0}^{\infty} a_1^i  z_{t-i} + \sum_{i=0}^{\infty} a_1^i  \varepsilon_{t-i} 。
$$

上述展开由三部分组成：
<!-- bilingual-en:start -->
The expansion consists of three parts:
<!-- bilingual-en:end -->

• 第一项 $\frac{a_0}{1 - a_1}$ 是没有干预和无扰动时序列的**稳态均值**（漂移项贡献的长期水平）。
• 第二项表示干预哑变量对 $y_t$ 的累积影响：当前以及之前各期$z$的效应经过系数$c_0$和AR系数$a_1$的滞后传播累积到本期。由于$z_t$在1973Q1之后为1，在此之前为0，这一项会体现政策使序列水平改变的贡献。
• 第三项是误差项$\varepsilon$的影响经由AR(1)滞后传递的累积，对应序列自身的随机波动。
<!-- bilingual-en:start -->
- The first term $\frac{a_0}{1 - a_1}$ is **steady-state mean**(long-term level of drift contribution) of the time series without intervention and disturbance.
- The second term represents the cumulative effect of the intervention dummy variable on $y_t$: current and prior periods of $z$ effects are cumulative to the current period through a lagged propagation of the coefficient $c_0$ and AR coefficient $a_1$.  Since $z_t$ is 1 after 1973Q1 and 0 before that, this term reflects the contribution of the policy to the change in the sequence level.
- The third term is the accumulation of the effect of the error term $\varepsilon$ transmitted through the AR(1) lag, corresponding to the random fluctuation of the sequence itself.
<!-- bilingual-en:end -->

**长期均值的变化：** 没有干预时，系列的长期均值为 $\frac{a_0}{1 - a_1}$。当干预施行后，由于 $z_t$ 从0变为1 且持续为1，序列将收敛到新的均值。可以将上式取 $t \to \infty$ 且假定$\varepsilon$均值为0，则长期均值 $y_{\infty}$ 满足：
<!-- bilingual-en:start -->
** Change in long-term mean: ** The long-term mean of the series was $\frac{a_0}{1 - a_1}$ without intervention.  When the intervention is implemented, the sequence converges to a new mean because $z_t$ changes from 0 to 1 and remains 1.  You can take the above formula as $t \to \infty$ and assume that the $\varepsilon$ mean is 0, then the long-term mean $y_{\infty}$ satisfies:
<!-- bilingual-en:end -->

$$
y_{\infty} = \frac{a_0}{1 - a_1} + c_0 \sum_{i=0}^{\infty} a_1^i \cdot 1 = \frac{a_0}{1-a_1} + \frac{c_0}{1 - a_1} = \frac{a_0 + c_0}{1 - a_1}。
$$


可以看出，干预使序列的漂移项从 $a_0$ 变为 $a_0 + c_0$，从而将**长期均值**提高（或降低）了 $\frac{c_0}{1 - a_1}$。这个长期效应就是干预的永久影响。当 $c_0<0$ 时，新的长期均值低于原先，说明政策永久性地减少了劫机事件发生的平均水平；$c_0>0$ 则反之。这个结果直观：$c_0$ 是干预导致的瞬时变动，而 $\frac{c_0}{1 - a_1}$ 表示经过AR(1)动态放大后，干预对平稳均值的总改变量。
<!-- bilingual-en:start -->
It can be seen that the intervention changes the drift term of the sequence from $a_0$ to $a_0 + c_0$, thereby increasing (or decreasing) ** the long-term mean ** by $\frac{c_0}{1 - a_1}$.  This long-term effect is the permanent effect of the intervention.  When $c_0<0$, the new long-term average is lower than the previous one, indicating that the policy permanently reduces the average level of hijackings; $c_0>0$ is the opposite.  The result is intuitive: $c_0$ is the instantaneous change caused by the intervention, and $\frac{c_0}{1 - a_1}$ represents the total change of the intervention to the stationary mean after the AR(1) dynamic amplification.
<!-- bilingual-en:end -->

## 1.2. [[Impulse Response Function|脉冲响应函数]] IRF Impulse Response Analysis
<!-- bilingual-en:start -->
*1.2. [[Impulse Response Function|impulse response function]] IRF Impulse Response Analysis*
<!-- bilingual-en:end -->

>[!note] 脉冲响应函数
>即干预发生后各期对 $y$ 的影响路径。这里的“冲击”指的是干预变量$z_t$从0跳变为1的变化，相当于一次永久性的干预。
>
> 对于任意$j \ge 0$，我们可以计算**脉冲响应**：干预在$t$期发生对未来第$j$期 ($t+j$) 的影响大小。由于模型是线性的，冲击响应等于对 $z_t$ 求偏导：
>
> $$
> \frac{\partial y_{t+j}}{\partial z_t} = c_0  a_1^j, \qquad j = 0,1,2,\dots
> $$
>
> 该公式表明：在干预发生的当期（$j=0$），$y_t$对$z_t$的立即反应为$c_0$（这正是回归中的$c_0$系数）。随后每一时期的影响按因子$a_1$衰减：$j=1$ 时影响为 $c_0 a_1$，$j=2$ 时为 $c_0 a_1^2$，以此类推。因为 $|a_1|<1$，干预对 $y$ 的边际影响会几何级数地减弱，但**不会消失**——这是由于$z_t$的变化是永久性的：干预实施后$z_{t+j}$在之后各期都保持1。因此，$y$ 水平永久地移至新的轨道上（相对于未干预的情形）。
> <!-- bilingual-en:start -->
> **impulse response function**
> This traces how the intervention affects $y$ over subsequent periods. Here the “shock” is the permanent switch of the intervention variable $z_t$ from zero to one.
> For any $j \ge 0$, the impulse response is the effect at time $t+j$ of an intervention introduced at time $t$. Because the model is linear, it is given by the partial derivative with respect to $z_t$:
> The contemporaneous response ($j=0$) is $c_0$. Each subsequent marginal response decays by a factor of $a_1$: it is $c_0a_1$ at $j=1$, $c_0a_1^2$ at $j=2$, and so on. Since $|a_1|<1$, these marginal responses decay geometrically. The intervention itself is permanent, however, because $z_{t+j}$ remains one after implementation; consequently, the level of $y$ moves permanently onto a new trajectory relative to the no-intervention counterfactual.
> <!-- bilingual-en:end -->

那么按照这个定义,我们也可以定义**累计冲击响应函数**，表示干预发生后直到第$j$期末总的影响积累。由于干预在$t$期后一直“在位”，累计影响等于将各期的边际影响相加：
<!-- bilingual-en:start -->
Then according to this definition, we can also define**cumulative impulse response function**, which represents the total impact accumulation after the intervention occurs until the end of the $j$ period.  Since the intervention has been "in place" since $t$, the cumulative effect is equal to adding the marginal effects of each period:
<!-- bilingual-en:end -->

$$
I_t(j) = \sum_{i=0}^{j} \frac{\partial y_{t+i}}{\partial z_t} = c_0 [1 + a_1 + a_1^2 + \cdots + a_1^j]。
$$

这是一个**部分和**，随着$j$增大逐渐逼近$c_0/(1 - a_1)$，与前述长期效应相符。当$j \to \infty$时，$I_t(j)$的极限即为干预的长期总影响 $\frac{c_0}{1 - a_1}$。图形上看，$I_t(j)$曲线会逐渐趋于平稳，水平线对应长期效应值。
<!-- bilingual-en:start -->
This is a **partial sum of **, which gradually approaches the $c_0/(1 - a_1)$ as the $j$ increases, in line with the long-term effect mentioned above.  When $j \to \infty$, the limit of $I_t(j)$ is $\frac{c_0}{1 - a_1}$.  In the graph, the $I_t(j)$ curve will gradually tend to be stable, and the horizontal line corresponds to the long-term effect value.
<!-- bilingual-en:end -->

## 1.3. 干预效应的识别与估计步骤
<!-- bilingual-en:start -->
*1.3. Steps for identification and estimation of intervention effects*
<!-- bilingual-en:end -->

1. **样本分段检验：** 先对干预发生前和发生后的数据分别拟合合适的ARIMA模型，并比较估计的模型参数是否存在显著差异。例如，在劫机案例中，可以分别用1973年前的数据和1973年后的数据估计AR(1)模型，检查$a_1$等系数有无明显变化。这可以帮助确认干预可能影响了数据生成过程。
<!-- bilingual-en:start -->
1.**Sample segmentation test:**First, fit the appropriate ARIMA model to the data before and after the intervention, and compare the estimated model parameters whether there are significant differences.  For example, in the case of hijacking, the AR(1) model can be estimated using pre-1973 data and post-1973 data respectively to check for significant changes in the $a_1$ and other coefficients.  This can help to confirm that the intervention may have affected the data generation process.
<!-- bilingual-en:end -->

2. **构建全样本初始模型：** 使用包含整个样本期的所有数据，先不考虑干预项，尝试识别一个适合的ARIMA模型作为**基准模型**。比如，用1968–1978年的劫机数据拟合一个ARIMA模型（也可能需要差分使其平稳），以确保对干预效应以外的动态有正确建模。
<!-- bilingual-en:start -->
2. **Build an initial full-sample model:** Using the complete sample but omitting the intervention term, identify a suitable ARIMA **benchmark model**. For example, fit an ARIMA model to the 1968–1978 hijacking data, differencing if necessary to achieve stationarity, so that dynamics unrelated to the intervention are modeled properly.
<!-- bilingual-en:end -->

3. **加入干预项并估计：** 在基准模型中加入干预哑变量（或相应的干预函数）$z_t$，估计**干预模型**的参数，包括$c_0$。此时检验：
<!-- bilingual-en:start -->
3. **Add intervention terms and estimate: **Add the intervention dummy variable (or corresponding intervention function) $z_t$ to the baseline model and estimate the parameters of **the intervention model**, including $c_0$.  At this point, verify:
<!-- bilingual-en:end -->

• 干预系数$c_0$是否显著，显著则意味着干预有统计学上的显著影响。
<!-- bilingual-en:start -->
- $c_0$ was significant, which meant that the intervention had a statistically significant effect.
<!-- bilingual-en:end -->

• 其它模型系数在加入干预后是否依然合理。
<!-- bilingual-en:start -->
- Whether other model factors remain appropriate after intervention.
<!-- bilingual-en:end -->

• 干预模型的残差是否近似白噪声（无自相关）。
<!-- bilingual-en:start -->
- Whether the residual error of the intervention model approximates white noise (no autocorrelation).
<!-- bilingual-en:end -->

4. **模型诊断与比较：** 对比包含干预项的模型与不含干预项的基准模型，或者与其他可能的备选模型，使用信息准则（[[回归模型比较与选择#AIC 与 BIC：likelihood fit 加 complexity penalty|AIC]]、SBC/[[回归模型比较与选择#AIC 与 BIC：likelihood fit 加 complexity penalty|BIC]]）来评估优劣。理想情况下，含干预模型应当有更低的信息准则值、残差更随机，以及**优于**不包含干预的模型。
<!-- bilingual-en:start -->
4. **Model diagnosis and comparison:** Compare the intervention model with the no-intervention benchmark and other plausible alternatives using information criteria ([[回归模型比较与选择#AIC 与 BIC：likelihood fit 加 complexity penalty|AIC]], SBC/[[回归模型比较与选择#AIC 与 BIC：likelihood fit 加 complexity penalty|BIC]]). A well-specified intervention model should have lower information-criterion values and residuals that are closer to white noise than the corresponding benchmark.
<!-- bilingual-en:end -->

# 2. 政策分析
<!-- bilingual-en:start -->
*2. Policy analysis*
<!-- bilingual-en:end -->

废话
<!-- bilingual-en:start -->
Background remarks.
<!-- bilingual-en:end -->

# 3. 自回归分布滞后ADL  Autoregressive [[回归预测与动态回归#分布滞后与动态乘数|Distributed]] Lag
<!-- bilingual-en:start -->
*3. Autoregressive Distributed Lag (ADL) model*
<!-- bilingual-en:end -->

## 3.1. ADL的定义
<!-- bilingual-en:start -->
*3.1. Definition of ADL*
<!-- bilingual-en:end -->

我们考虑如下的一般模型形式：
$$
 y_t = a_0 + A(L)y_{t-1} + C(L)z_t + B(L)\epsilon_t
$$
<!-- bilingual-en:start -->
We consider the following general model form:
<!-- bilingual-en:end -->

 其中 $A(L)$、$C(L)$ 和 $B(L)$ 分别是在滞后算子 $L$ 上的多项式。更具体地：
<!-- bilingual-en:start -->
$A(L)$, $C(L)$ and $B(L)$ are polynomials on the lag operator $L$.  More specifically:
<!-- bilingual-en:end -->

 • $A(L)$ 表示 $y$ 的自回归部分，如 $A(L)y_{t-1} = a_1 y_{t-1} + a_2 y_{t-2} + \cdots + a_p y_{t-p}$（这里为了方便，$A(L)$ 不写常数1项，只表示滞后项的和）。
<!-- bilingual-en:start -->
- $A(L)$ denotes the autoregressive part of $y$, such as $A(L)y_{t-1} = a_1 y_{t-1} + a_2 y_{t-2} + \cdots + a_p y_{t-p}$ (here, for convenience, $A(L)$ does not write a constant 1 term, only the sum of the lag terms).
<!-- bilingual-en:end -->

 • $C(L)$ 表示 $z$ 对 $y$ 的**传递函数**（transfer function），可写作 $C(L) = c_0 + c_1 L + c_2 L^2 + \cdots + c_r L^r$，意味着 $z_t$ 的当期值和多个滞后值共同作用于 $y_t$。
<!-- bilingual-en:start -->
- $C(L)$ is the **transfer function** from $z$ to $y$. Writing $C(L) = c_0 + c_1 L + c_2 L^2 + \cdots + c_r L^r$ means that the current and several lagged values of $z_t$ jointly affect $y_t$.
<!-- bilingual-en:end -->

 • $B(L)$ 表示 $\epsilon_t$ 的移动平均(MA)部分，如 $B(L)\epsilon_t = \epsilon_t + b_1 \epsilon_{t-1} + \cdots + b_q \epsilon_{t-q}$。很多情况下我们可以假定 $B(L)=1$（即噪声项为白噪声，无MA结构），以简化分析。
<!-- bilingual-en:start -->
- $B(L)$ represents the moving average (MA) portion of the $\epsilon_t$, such as $B(L)\epsilon_t = \epsilon_t + b_1 \epsilon_{t-1} + \cdots + b_q \epsilon_{t-q}$.  In many cases, we can assume $B(L)=1$ (i.e. the noise term is white noise, no MA structure) to simplify the analysis.
<!-- bilingual-en:end -->

>[!note] [[回归预测与动态回归#分布滞后与动态乘数|自回归分布滞后模型]]
>
> **自回归分布滞后模型**（Autoregressive Distributed Lag Model, 简称 **ADL 模型**）就是干预模型的自然延伸。在 ADL 模型中，$z_t$ 可以是随机的外生变量序列.在上面的模型的基础上当我们忽略$B(L)$（设$B(L)=1$）时，就得到**自回归分布滞后（ADL）模型**：
$$y_t = a_0 + A(L)y_{t-1} + C(L) z_t + \varepsilon_t$$
<!-- bilingual-en:start -->
>The **[[回归预测与动态回归#分布滞后与动态乘数|autoregressive distributed lag model]]** (Autoregressive Distributed Lag Model, abbreviated **ADL**) is a natural extension of the intervention model. In an ADL model, $z_t$ may be a stochastic exogenous variable. Setting $B(L)=1$ in the model above gives:
$$y_t = a_0 + A(L)y_{t-1} + C(L)z_t + \varepsilon_t$$
<!-- bilingual-en:end -->

• $C(L)$ 被称为**传递函数**（transfer function），因为它描述了外生变量$z_t$的变动如何通过不同滞后传递到内生变量$y_t$上。例如，如果 $C(L) = c_0 + c_1L + c_2L^2$，说明$z_t$对$y_t$有当期效应$c_0$，对下一期$y$有滞后一阶效应$c_1$，对再下一期有滞后两期效应$c_2$，以此类推。
<!-- bilingual-en:start -->
- $C(L)$ is referred to as the ** transfer function ** (transfer function) because it describes how variations of the exogenous variable $z_t$ are passed to the endogenous variable $y_t$ by different lag.  For example, if $C(L) = c_0 + c_1L + c_2L^2$, $z_t$ has a current effect on $y_t$ $c_0$, a lagging first order effect on the next $y$ $c_1$, a lagging two periods effect on the next period $c_2$, and so on.
<!-- bilingual-en:end -->

• $C(L)$ 多项式的系数${c_i}$称为**传递函数权重**。它们刻画了$z$的一个单位变化对$y$在不同滞后期的影响强度。
<!-- bilingual-en:start -->
- The coefficient ${c_i}$ of the $C(L)$ polynomial is called **transfer function weight **.  They characterize the influence of a unit change of $z$ on the $y$ at different lag.
<!-- bilingual-en:end -->

• 由于$z_t$影响被分布在多个时滞上，因此此类模型也常被称为“**[[回归预测与动态回归#分布滞后与动态乘数|分布滞后模型]]**”。
<!-- bilingual-en:start -->
- Since the $z_t$ effect is distributed over multiple time delays, such models are often referred to as "**[[回归预测与动态回归#分布滞后与动态乘数|distributed lag model]]**".
<!-- bilingual-en:end -->

>[!note] leading indicator
>在传递函数$C(L)$中，**$c_0$系数的重要性**值得注意：
>
> • 如果 $c_0 \neq 0$，则$z_t$对$y_t$有**当期直接影响**。$z_t$的变动立即反映在同时期的$y_t$上。
>
> • 如果 $c_0 = 0$，意味着**无当期效应**，$z_t$的变化不会直接影响当期$y_t$，最早的影响要经过至少1期滞后才能体现。在这种情况下，我们称$z_t$为**领先指标**（leading indicator），因为$z_t$的变化领先于$y_t$的变化。例如，如果经济指标$X$在时间上领先于$Y$，那么当前$Y$主要受之前的$X$影响，而当前$X$对当前$Y$没有直接影响。
> <!-- bilingual-en:start -->
> In transfer function $C(L)$, the importance of **$c_0$ coefficient**is noteworthy:
> · If $c_0 \neq 0$, then $z_t$ has a **immediate impact** on $y_t$.  Changes in $z_t$ are immediately reflected in $y_t$ for the same period.
> · If $c_0 = 0$ means **no current effect**, the change in $z_t$ does not directly affect current $y_t$, and the earliest effect must be at least 1 period behind.  In this case, we call $z_t$ the ** leading indicator**, because the change in $z_t$ is ahead of the change in $y_t$.  For example, if the economic metric $X$ is ahead of the $Y$ in time, then the current $Y$ is mainly affected by the previous $X$, while the current $X$ has no direct impact on the current $Y$.
> <!-- bilingual-en:end -->

## 3.2. ADL的模型性质
<!-- bilingual-en:start -->
*3.2. Properties of ADL*
<!-- bilingual-en:end -->

考虑一个简单情形来探究ADL模型的统计性质和如何识别滞后效应：**[[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|AR]](1)过程 + 延迟$d$期的单一滞后效应**。具体模型：
<!-- bilingual-en:start -->
A simple case is considered to investigate the statistical properties of the ADL model and how to identify the lag effect: **[[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|AR]](1) process + single lag effect ** in the delayed $d$ period.  Specific model:
<!-- bilingual-en:end -->

$$
y_t = a_1 y_{t-1} + c_d z_{t-d} + \varepsilon_t, \tag{[[回归预测与动态回归#分布滞后与动态乘数|ADL]](1, d)}
$$

其中 $z_t$ 是外生的白噪声过程（均值0，方差$\sigma_z^2$），$\varepsilon_t$是白噪声误差（与$z_t$独立），$d \ge 0$为整数，表示$z$对$y$影响的延迟长度。
<!-- bilingual-en:start -->
$z_t$ is the exogenous white noise process (mean 0, variance $\sigma_z^2$), $\varepsilon_t$ is the white noise error (independent of $z_t$), and $d \ge 0$ is an integer, which indicates the delay length of the influence of $z$ on $y$.
<!-- bilingual-en:end -->

这个模型意味着：$z$对$y$的**唯一影响**发生在滞后$d$期处，而且影响强度为$c_d$。例如若$d=2$，则$z_{t-2}$影响$y_t$，$z_{t-1}$和$z_t$对$y_t$无直接作用。
<!-- bilingual-en:start -->
This model implies that the **only effect**of $z$ on $y$ occurs at the $d$ lag period and the effect strength is $c_d$.  For example, if $d=2$, $z_{t-2}$ affects $y_t$, and $z_{t-1}$ and $z_t$ have no direct effect on $y_t$.
<!-- bilingual-en:end -->

为了刻画$y$与$z$之间的动态相关关系，我们引入:
>[!note] 交叉相关函数(Cross-correlation function, CCF)：
>
> $$
> \rho_{yz}(i) = \frac{\mathrm{cov}(y_t; z_{t-i})}{\sigma_y\sigma_z}
> $$
>
> 其中$\sigma_y$和$\sigma_z$分别是$y$和$z$的标准差。$\rho_{yz}(i)$描述了$y_t$与$z$在领先/滞后$i$期时的线性相关强度。当我们绘制$\rho_{yz}(i)$随$i$变化的图（$i$可以取负值表示$y$落后于$z$的情形），就得到**交叉相关图**或**交叉相关函数图 (CCF图)**。
> <!-- bilingual-en:start -->
> In order to characterize the dynamic correlation between $y$ and $z$, we introduce:
> $\sigma_y$ and $\sigma_z$ are the standard deviations of $y$ and $z$, respectively.  $\rho_{yz}(i)$ describes the linear correlation strength between $y_t$ and $z$ in the leading/lagging $i$ period.  When we draw the graph of $\rho_{yz}(i)$ changing with $i$ ($i$ can take negative value to represent $y$ is behind $z$), we can get **cross-correlation graph**or **cross-correlation function graph (CCF graph)**.
> <!-- bilingual-en:end -->

对于模型ADL(1,d)，我们可以根据模型结构推导出理论上的协方差$\mathrm{cov}(y_t, z_{t-i})$。因为$z$是白噪声且独立于$y$的冲击，利用滞后运算展开$y_t$：
(使用的是:$\frac{1}{1 - a_1 L} = 1 + a_1 L + a_1^2 L^2 + a_1^3 L^3 + \cdots \quad \text{当 } |a_1| < 1$)
<!-- bilingual-en:start -->
For the model ADL(1,d), we can derive the theoretical covariance $\mathrm{cov}(y_t, z_{t-i})$ according to the structure of the model.  Because $z$ is white noise and independent of the impact of $y$, the lag operation is used to expand $y_t$:
(Using: $\frac{1}{1 - a_1 L} = 1 + a_1 L + a_1^2 L^2 + a_1^3 L^3 + \cdots \quad \text{when } |a_1| < 1$)
<!-- bilingual-en:end -->

$$y_t = \frac{c_d z_{t-d}}{1 - a_1 L} + \frac{\varepsilon_t}{1 - a_1 L} \Rightarrow y_t = c_d[z_{t-d} + a_1 z_{t-d-1} + a_1^2 z_{t-d-2} + \cdots] + \text{其他噪声项}$$
<!-- bilingual-en:start -->
$$y_t = \frac{c_d z_{t-d}}{1 - a_1 L} + \frac{\varepsilon_t}{1 - a_1 L} \Rightarrow y_t = c_d[z_{t-d} + a_1 z_{t-d-1} + a_1^2 z_{t-d-2} + \cdots] + \text{Other Noise Items}$$
<!-- bilingual-en:end -->

由此可见，$y_t$涉及到$z$的滞后$d, d+1, d+2,\dots$期。进一步考虑协方差：
<!-- bilingual-en:start -->
Thus, $y_t$ involves the delayed $d, d+1, d+2,\dots$ phase of $z$.  Further consider covariance:
<!-- bilingual-en:end -->

• 对于$i < d$：$z_{t-i}$是发生在$y_t$影响之前的$z$，$y_t$不包含如此“晚近”的$z$项（因为$y_t$最早只包括到$z_{t-d}$）。由于$z$和$\varepsilon$独立，且$z$是白噪声不自相关，可知 $\mathrm{cov}(y_t, z_{t-i})=0$，因此$\rho_{yz}(i)=0$。换言之，在干预生效延迟$d$期之前，$y$与$z$毫无线性关联。
<!-- bilingual-en:start -->
- For $z$ where $i < d$:$z_{t-i}$ occurred before the impact of $y_t$, $y_t$ does not include such "late" $z$ entries (since $y_t$ included only $z_{t-d}$ at the earliest).  Because $z$ and $\varepsilon$ are independent, and $z$ is white noise and non-autocorrelation, $\mathrm{cov}(y_t, z_{t-i})=0$ is known, so $\rho_{yz}(i)=0$.  In other words, the $y$ is associated with the $z$ milliwirelessness prior to the intervention taking effect delaying the $d$ period.
<!-- bilingual-en:end -->

• 对于$i = d$：$\mathrm{cov}(y_t, z_{t-d})$ 由 $y_t$中的 $c_d z_{t-d}$ 项贡献，其协方差为 $c_d \mathrm{cov}(z_{t-d}, z_{t-d}) = c_d \sigma_z^2$。同时$y_t$中的其它项（如误差项和不同滞后的$z$）要么与$z_{t-d}$独立，要么不包含$z_{t-d}$。所以 $\mathrm{cov}(y_t, z_{t-d}) = c_d \sigma_z^2$，因而 $\rho_{yz}(d) = \frac{c_d \sigma_z^2}{\sigma_y \sigma_z}$。在数值上，如果$c_d\neq0$，我们会在滞后$d$处观察到一个显著**峰值**相关。
<!-- bilingual-en:start -->
- The $i = d$:$\mathrm{cov}(y_t, z_{t-d})$ is contributed by the $c_d z_{t-d}$ term in $y_t$ with a covariance of $c_d \mathrm{cov}(z_{t-d}, z_{t-d}) = c_d \sigma_z^2$.  At the same time, the other terms in $y_t$, such as error term and different lag $z$, are either independent of $z_{t-d}$ or do not contain $z_{t-d}$.  So $\mathrm{cov}(y_t, z_{t-d}) = c_d \sigma_z^2$, $\rho_{yz}(d) = \frac{c_d \sigma_z^2}{\sigma_y \sigma_z}$.  Numerically, if $c_d\neq0$, we observe a significant ** peak ** correlation at a lag of $d$.
<!-- bilingual-en:end -->

• 对于$i = d+1$：$y_t$包含 $c_d a_1 z_{t-d-1}$ 项，与 $z_{t-(d+1)} = z_{t-d-1}$ 完全同步。该协方差为 $c_d a_1 \sigma_z^2$。但同时$y_t$中也有 $c_d z_{t-d}$项，它与$z_{t-d-1}$不相关（因不同期的$z$不相关）。所以 $\mathrm{cov}(y_t, z_{t-d-1}) = c_d a_1 \sigma_z^2$。因此$\rho_{yz}(d+1) = \frac{c_d a_1 \sigma_z^2}{\sigma_y \sigma_z}$。
<!-- bilingual-en:start -->
- For $i = d+1$:$y_t$, include a $c_d a_1 z_{t-d-1}$ entry, fully synchronized with $z_{t-(d+1)} = z_{t-d-1}$.  The covariance is $c_d a_1 \sigma_z^2$.  However, there is also a $c_d z_{t-d}$ term in $y_t$, which is not related to $z_{t-d-1}$ (because $z$ is not related at different stages).  So, $\mathrm{cov}(y_t, z_{t-d-1}) = c_d a_1 \sigma_z^2$.  So $\rho_{yz}(d+1) = \frac{c_d a_1 \sigma_z^2}{\sigma_y \sigma_z}$.
<!-- bilingual-en:end -->

• 推广$i \ge d$：对一般 $i = d + k$（$k \ge 0$），$y_t$中与$z_{t-i}$同步的那一项是 $c_d a_1^k z_{t-d-k}$（因为$y_t$包含$z_{t-d-k}$乘以$a_1^k c_d$）。于是：
$$
\mathrm{cov}(y_t, z_{t-i}) = c_d a_1^k \sigma_z^2 \qquad \text{当 } i \ge d,
$$
即 $i=d+k$ 时协方差为 $c_d a_1^k \sigma_z^2$。转化为相关系数，
$$

\rho_{yz}(i) = \frac{c_d a_1^{i-d} \sigma_z^2}{\sigma_y \sigma_z}, \qquad i \ge d。

$$
<!-- bilingual-en:start -->
- Promote $i \ge d$: For generic $i = d + k$ ($k \ge 0$), the one in $y_t$ that synchronizes with $z_{t-i}$ is $c_d a_1^k z_{t-d-k}$ (because $y_t$ contains $z_{t-d-k}$ times $a_1^k c_d$).  Therefore:
That is, the covariance is $c_d a_1^k \sigma_z^2$ when $i=d+k$.  convert to correlation coefficient,
<!-- bilingual-en:end -->

综合以上分析，可以总结ADL(1,d)模型下 $y$与$z$的交叉协方差（和相关）特征：
<!-- bilingual-en:start -->
Based on the above analysis, we can summarize the cross-covariance (and correlation) characteristics of $y$ and $z$ in ADL (1,d) model:
<!-- bilingual-en:end -->

• **对于 $i < d$：** $E[y_t z_{t-i}] = 0$，因此 $\rho_{yz}(i) = 0$（在交叉相关图上，干预滞后之前所有点相关为零）。
<!-- bilingual-en:start -->
- **For $i < d$:** $E[y_t z_{t-i}] = 0$, so $\rho_{yz}(i) = 0$ (on the cross-correlation graph, all points had zero correlation before the intervention lagged).
<!-- bilingual-en:end -->

• **对于 $i \ge d$：** $E[y_t z_{t-i}] = c_d a_1^{i-d}  \sigma_z^2$。特别地，$i=d$ 时 $E[y_t z_{t-d}] = c_d \sigma_z^2$ 是第一个非零协方差；$i=d+1$ 时 $E[y_t z_{t-d-1}] = c_d a_1 \sigma_z^2$；随着滞后$i$增加，协方差按比率$a_1$几何衰减。
<!-- bilingual-en:start -->
- **For $i \ge d$:** $E[y_t z_{t-i}] = c_d a_1^{i-d}  \sigma_z^2$.  In particular, $E[y_t z_{t-d}] = c_d \sigma_z^2$ is the first non-zero covariance for $i=d$, $E[y_t z_{t-d-1}] = c_d a_1 \sigma_z^2$ for $i=d+1$, and the covariance decays geometrically in proportion to $a_1$ as the lag $i$ increases.
<!-- bilingual-en:end -->

对应的CCF图像特征为：在滞后$d$之前相关为零，滞后$d$处出现**尖刺**（正或负，取决于$c_d$符号），随后相关系数按指数规律衰减。如果观测数据的样本交叉相关图呈现这种形状，我们可以推断$z$对$y$的主要影响开始于某个延迟$d$，并随后按AR(1)过程衰减，从而帮助我们识别$C(L)$和$A(L)$的大致形式。
<!-- bilingual-en:start -->
The corresponding CCF is zero at lags before $d$, has a **spike** at lag $d$—positive or negative according to the sign of $c_d$—and then decays geometrically. If the sample cross-correlation plot has this shape, it suggests that the main effect of $z$ on $y$ begins after a delay of $d$ periods and then propagates through the AR(1) dynamics. This pattern helps identify the approximate forms of $C(L)$ and $A(L)$.
<!-- bilingual-en:end -->

## 3.3. 外生输入序列存在自相关的情况
<!-- bilingual-en:start -->
*3.3. Autocorrelation in the exogenous input series*
<!-- bilingual-en:end -->

前述分析假定$z_t$是白噪声（无序列相关）的外生过程，这简化了$C(L)$的识别。然而，在现实中，外生变量$z_t$本身往往具有动态结构，而非独立同分布。例如，$z_t$可能是另一经济变量（如利率、收入等），它本身可以用AR或其它模型描述。
<!-- bilingual-en:start -->
The previous analysis assumes that $z_t$ is an exogenous process of white noise (no sequence correlation), which simplifies the identification of $C(L)$.  However, in reality, the exogenous variable $z_t$ is often dynamically structured rather than independently and identically distributed.  For example, $z_t$ may be another economic variable (such as interest rates, income, etc.), which itself can be described in AR or other models.
<!-- bilingual-en:end -->

因此，一般的**传递函数模型**会包含对$z_t$动态的建模：
<!-- bilingual-en:start -->
Therefore, the general**transfer function model**includes the modeling of $z_t$ dynamics:
<!-- bilingual-en:end -->

$$
\begin{aligned}
y_t &= a_0 + A(L)y_{t-1} + C(L)z_t + \varepsilon_t\\
z_t &= D(L)z_{t-1} + \varepsilon_{zt}
\end{aligned}
$$

这里 $D(L)$ 是$z_t$自身的滞后多项式，$\varepsilon_{zt}$是$z_t$过程的白噪声冲击。也就是说，我们将 $z_t$ 建模为一个ARMA过程（或近似如此）。
<!-- bilingual-en:start -->
Here $D(L)$ is the lag polynomial governing $z_t$, and $\varepsilon_{zt}$ is the white-noise innovation in the $z_t$ process. In other words, $z_t$ is modeled as an ARMA process, or approximately so.
<!-- bilingual-en:end -->

在这个框架下，我们可以考虑**三类脉冲响应函数**来全面理解系统动态：
<!-- bilingual-en:start -->
In this framework, we can consider**three types of impulse response functions**to understand the system dynamics:
<!-- bilingual-en:end -->

1. **$z$序列自身的冲击响应：** $z_t$受到它自己的冲击$\varepsilon^z_t$时，如何通过(6)式在未来演化。这由$D(L)$决定，通常就是$z_t$的ACF/PACF性质。
2. **$y$序列自身的冲击响应：** $y_t$受到它自身的误差冲击$\varepsilon_t$时，通过(5)式$A(L)$的传递对未来$y$的影响（这类似于我们之前ARIMA模型中的IRF）。
3. **$z$序列冲击对$y$的响应：** 这是关键，我们关心外生变量的变动如何传递到内生变量。由于$z_t$本身有动态，我们需要综合(5)和(6)来分析**$z$的冲击传递到$y$**的过程。
<!-- bilingual-en:start -->
1. **Response of $z$ to its own shock:** How $z_t$ evolves after its own innovation $\varepsilon^z_t$ through equation (6). This is governed by $D(L)$ and reflected in the ACF/PACF of $z_t$.
2. **Response of $y$ to its own shock:** How an innovation $\varepsilon_t$ propagates through $A(L)$ in equation (5) and affects future values of $y$, analogous to an ARIMA impulse response.
3. **Response of $y$ to a shock in $z$:** This is the central object: how a change in the exogenous variable is transmitted to the endogenous variable. Because $z_t$ has its own dynamics, equations (5) and (6) must be combined to trace transmission from **$z$ to $y$**.
<!-- bilingual-en:end -->

具体来说，考虑第3种冲击：令$t$期发生$z$序列的一个冲击$\varepsilon_{zt}$（大小为1的单位冲击），并观察$y_t$随后各期的变化。因为$z_t$满足(6)，我们可以将(6)代入(5)，消除$z_t$：
<!-- bilingual-en:start -->
Specifically, consider the third response: introduce a unit innovation $\varepsilon_{zt}$ to the $z$ process at time $t$ and trace the subsequent path of $y$. Because $z_t$ satisfies equation (6), substitute (6) into (5) to eliminate $z_t$:
<!-- bilingual-en:end -->

$$
y_t = a_0 + A(L)y_{t-1} + \frac{C(L)}{1 - D(L)L}\varepsilon_{zt} + \varepsilon_t。
$$

## 3.4. 识别与估计
<!-- bilingual-en:start -->
*3.4. Identification and estimation*
<!-- bilingual-en:end -->

上面的那个模型有识别问题,具体一点来讲,就是不能唯一确定值.
<!-- bilingual-en:start -->
The model above has an identification problem: its structural parameters cannot be uniquely recovered from the observed data without further restrictions.
<!-- bilingual-en:end -->

>[!quote] 不能识别的具体分析
> 回到前面简单的例子，如果我们不知道结构，只凭数据去拟合模型：
>
> $$
>
> y_t = a_1 y_{t-1} + c_1 z_t + \epsilon_t, \qquad
>
> z_t = d_1 z_{t-1} + \epsilon^z_t,
>
> $$
>
> 我们发现观测到的 $y_t$ 实际满足：
>
> $$
>
> y_t = a_1 y_{t-1} + c_1(d_1 z_{t-1} + \epsilon^z_t) + \epsilon_t = a_1 y_{t-1} + c_1 d_1 z_{t-1} + c_1 \epsilon^z_t + \epsilon_t.
>
> $$
>
> 那么关于 $z$ 的影响项，我们既可以将其解释为“$z_t$ 对 $y_t$ 有当期影响 $c_1$，同时 $z_{t-1}$ 对 $y_t$ 有影响 $c_1 d_1$”（对应于原结构参数），也可以等价地解释为“$z_t$ 对 $y_t$ 没有直接影响（$c_1=0$），而是 $z_{t-1}$ 对 $y_t$ 有影响 $c_1 d_1$”（把 $c_1 d_1$ 看作新的滞后影响系数）。这就导致了**传递函数 $C(L)$ 无法唯一确定**：数据上看，你很难区分是 $c_1$ 作用于当期，还是 $c_1 d_1$ 作用于滞后一期，因为两种解释都符合观测。这就是所谓的识别问题。
> <!-- bilingual-en:start -->
> Return to the simple example. If the structural form is unknown and we fit only what the data reveal,
> $$
> y_t=a_1y_{t-1}+c_1z_t+\epsilon_t,\qquad
> z_t=d_1z_{t-1}+\epsilon_t^z,
> $$
> then substitution gives
> $$
> y_t=a_1y_{t-1}+c_1d_1z_{t-1}+c_1\epsilon_t^z+\epsilon_t.
> $$
> The observed lag term can be interpreted through the original contemporaneous effect $c_1$ and the dynamics $d_1$, or as a new direct lag effect with no contemporaneous effect. The data alone cannot distinguish these observationally equivalent explanations. Consequently, **the transfer function $C(L)$ is not uniquely identified** without additional restrictions.
> <!-- bilingual-en:end -->

有两种方法解决:
<!-- bilingual-en:start -->
There are two solutions:
<!-- bilingual-en:end -->

方法1：不预设传递函数结构，直接在多元模型中以信息准则/检验确定滞后阶与影响项。~~内容略.~~
$$
y_t = a_0 + \sum_{i=1}^{p} a_iy_{t-i} + \sum_{i=0}^{n} c_i z_{t-i} + \epsilon_t
$$
<!-- bilingual-en:start -->
Method 1: Do not impose a transfer-function structure in advance. Instead, use information criteria and statistical tests to choose the relevant lags directly in the multivariate model. ~~Details omitted.~~
<!-- bilingual-en:end -->

方法2：预滤波法（预白化法）识别传递函数。 这是前面推导中启发的方法：我们先估计并得到 $z_t$ 的模型 $D(L)$，然后用 $(1 - \hat{D}(L)L)$ 滤波（预白化）$y_t$ 和 $z_t$，以去除 $z$ 的自相关结构，使之接近白噪声。具体步骤如下：
<!-- bilingual-en:start -->
Method 2: Identify the transfer function by **prefiltering**, or **prewhitening**. First estimate the model $D(L)$ for $z_t$. Then apply the filter $(1-\hat D(L)L)$ to both $y_t$ and $z_t$, removing the autocorrelation in $z_t$ so that the filtered input is approximately white noise. The steps are:
<!-- bilingual-en:end -->

# 4. 向量自回归VAR
<!-- bilingual-en:start -->
*4. Vector autoregressive (VAR)*
<!-- bilingual-en:end -->

## 4.1. VAR定义
<!-- bilingual-en:start -->
*4.1. VAR Definition*
<!-- bilingual-en:end -->

==是用来处理内生性问题的工具==
>[!note] [[VAR Model|VAR]]
> 考虑一个二元系统（bivariate VAR）：
> $$
> \begin{aligned}
> y_t &= b_{10} - b_{12} z_t + \gamma_{11} y_{t-1} + \gamma_{12} z_{t-1} + \varepsilon_{yt}  \\
> z_t &= b_{20} - b_{21} y_t + \gamma_{21} y_{t-1} + \gamma_{22} z_{t-1} + \varepsilon_{zt}
> \end{aligned}
> $$
> 其中：
> - $y_t$、$z_t$：两个**内生变量**
> - $\varepsilon_{yt}$,$\varepsilon_{zt}$：白噪声扰动项
> - $b_{12}$, $b_{21}$：**即时交叉依赖**（contemporaneous interaction）
>
> 我们将系统写成如下矩阵形式==结构VAR==：
> $$
> \begin{bmatrix} 1 & b_{12} \\ b_{21} & 1 \end{bmatrix} \begin{bmatrix} y_t \\ z_t \end{bmatrix} =\begin{bmatrix} b_{10} \\ b_{20} \end{bmatrix} + \begin{bmatrix} \gamma_{11} & \gamma_{12} \\ \gamma_{21} & \gamma_{22} \end{bmatrix} \begin{bmatrix} y_{t-1} \\ z_{t-1} \end{bmatrix} + \begin{bmatrix} \varepsilon_{yt} \\ \varepsilon_{zt} \end{bmatrix}
> $$
> 记作：
> $$B x_t = \Gamma_0 + \Gamma_1 x_{t-1} + \varepsilon_t$$
> 我们乘以 $B^{-1}$,==这个叫做同期变换,该是白噪声还是白噪声,不影响的==，得到==简约VAR==：
> $$x_t = A_0 + A_1 x_{t-1} + e_t $$
> 其中：
> - $A_0 = B^{-1} \Gamma_0$
> - $A_1 = B^{-1} \Gamma_1$
> - $e_t = B^{-1} \varepsilon_t$
> <!-- bilingual-en:start -->
> ==VAR is a tool for modeling a system of jointly determined variables.==
> Consider a bivariate system:
> $$
> \begin{aligned}
> y_t &= b_{10} - b_{12} z_t + \gamma_{11} y_{t-1} + \gamma_{12} z_{t-1} + \varepsilon_{yt}  \\
> z_t &= b_{20} - b_{21} y_t + \gamma_{21} y_{t-1} + \gamma_{22} z_{t-1} + \varepsilon_{zt}.
> \end{aligned}
> $$
> Here:
> - $y_t$ and $z_t$ are two **endogenous variables**;
> - $\varepsilon_{yt}$ and $\varepsilon_{zt}$ are white-noise shocks;
> - $b_{12}$ and $b_{21}$ capture **contemporaneous cross-dependence**.
>
> In matrix notation, this is the **structural VAR**
> $$B x_t = \Gamma_0 + \Gamma_1 x_{t-1} + \varepsilon_t.$$
> Premultiplying by $B^{-1}$ is a contemporaneous transformation. It preserves the white-noise property and yields the **reduced-form VAR**
> $$x_t = A_0 + A_1 x_{t-1} + e_t,$$
> where:
> - $A_0 = B^{-1} \Gamma_0$;
> - $A_1 = B^{-1} \Gamma_1$;
> - $e_t = B^{-1} \varepsilon_t$.
> <!-- bilingual-en:end -->

==简约VAR中的$e_{t}$依旧是白噪声,并且消除了内生性.那就意味着可以直接用OLS估计.但是扰动项出现了同期相关==
<!-- bilingual-en:start -->
==The reduced-form innovations $e_t$ are still white noise, and no current endogenous variable appears on the right-hand side. Each reduced-form equation can therefore be estimated by OLS. However, the innovations in different equations may be contemporaneously correlated.==
<!-- bilingual-en:end -->

由于$\text{Cov}(e_{1t}, e_{2t}) = \mathbb{E}[e_{1t} e_{2t}] = \mathbb{E} \left[ \frac{(\varepsilon_{yt} - b_{12} \varepsilon_{zt})(\varepsilon_{zt} - b_{21} \varepsilon_{yt})} {(1 - b_{12}b_{21})^2} \right]$
展开乘法期望后，注意 $\varepsilon_{yt}, \varepsilon_{zt}$ 是**独立白噪声** ⇒ 所有交叉项消掉，只剩下：$\text{Cov}(e_{1t}, e_{2t}) = \frac{ - b_{21} \sigma_y^2 + b_{12} \sigma_z^2 } {(1 - b_{12}b_{21})^2}$.
<!-- bilingual-en:start -->
Because $\text{Cov}(e_{1t}, e_{2t}) = \mathbb{E}[e_{1t} e_{2t}] = \mathbb{E} \left[ \frac{(\varepsilon_{yt} - b_{12} \varepsilon_{zt})(\varepsilon_{zt} - b_{21} \varepsilon_{yt})} {(1 - b_{12}b_{21})^2} \right]$
When the expectation of multiplication is expanded, note that $\varepsilon_{yt}, \varepsilon_{zt}$ is **independent white noise**⇒all the cross terms are eliminated, and only $\text{Cov}(e_{1t}, e_{2t}) = \frac{ - b_{21} \sigma_y^2 + b_{12} \sigma_z^2 } {(1 - b_{12}b_{21})^2}$ is left.
<!-- bilingual-en:end -->

>[!note] 方差-协方差矩阵 $\Sigma = \mathbb{E}[e_t e_t']$
>
> $\Sigma = \begin{bmatrix} \text{Var}(e_{1t}) & \text{Cov}(e_{1t}, e_{2t}) \\ \text{Cov}(e_{1t}, e_{2t}) & \text{Var}(e_{2t}) \end{bmatrix}$
>
> <!-- bilingual-en:start -->
> ** variance-covariance matrix $\Sigma = \mathbb{E}[e_t e_t']$**
> $\Sigma = \begin{bmatrix} \text{Var}(e_{1t}) & \text{Cov}(e_{1t}, e_{2t}) \\ \text{Cov}(e_{1t}, e_{2t}) & \text{Var}(e_{2t}) \end{bmatrix}$
> <!-- bilingual-en:end -->

## 4.2. VAR稳定性和平稳性
<!-- bilingual-en:start -->
*4.2. VAR stability and stationarity*
<!-- bilingual-en:end -->

使用迭代法得到:$x_t = \left( I + A_1 + A_1^2 + \dots + A_1^n \right) A_0 + \sum_{i=0}^n A_1^i e_{t-i} + A_1^{n+1} x_{t-n-1}$.
<!-- bilingual-en:start -->
Using the iteration method, we get the following result: $x_t = \left( I + A_1 + A_1^2 + \dots + A_1^n \right) A_0 + \sum_{i=0}^n A_1^i e_{t-i} + A_1^{n+1} x_{t-n-1}$.
<!-- bilingual-en:end -->

>[!note] VAR的稳定性
>要让 [[VAR Model|VAR]] 模型在长期不爆炸（即收敛），我们要求：
>
> $\boxed{ \lim_{n \to \infty} A_1^n = 0 } \Rightarrow \text{所有特征值（eigenvalues）都在单位圆内}$
>
> 对二维 VAR，稳定性等价于这个多项式的根都在单位圆外：
>
> $I - A_1 L = \begin{bmatrix} 1 - a_{11}L & -a_{12}L \\ -a_{21}L & 1 - a_{22}L \end{bmatrix}$
>
> 求其行列式：
>
> $\det(I - A_1 L) = (1 - a_{11}L)(1 - a_{22}L) - a_{12}a_{21}L^2$
>
> 如果这个方程的根（特征根）都小于 1 ⇒ 模型稳定。
> <!-- bilingual-en:start -->
> **Stability of VAR**
> For a [[VAR Model|VAR]] process to be stable rather than explosive, we require:
> $\boxed{\lim_{n\to\infty}A_1^n=0}\quad\Longleftrightarrow\quad\text{every eigenvalue }\lambda\text{ of }A_1\text{ satisfies }|\lambda|<1.$
> Equivalently, the roots $z$ of the lag polynomial $\det(I-A_1z)=0$ must lie **outside** the unit circle, because those roots are reciprocals of the eigenvalues of $A_1$. For a two-variable VAR:
> $I - A_1 L = \begin{bmatrix} 1 - a_{11}L & -a_{12}L \\ -a_{21}L & 1 - a_{22}L \end{bmatrix}$
> Taking its determinant gives:
> $\det(I - A_1 L) = (1 - a_{11}L)(1 - a_{22}L) - a_{12}a_{21}L^2$
> Thus the two statements use different objects but are consistent: the eigenvalues of $A_1$ have modulus below one, while the roots of $\det(I-A_1L)=0$ have modulus above one.
> <!-- bilingual-en:end -->

==在VAR中,如果稳定,就一定弱平稳==
<!-- bilingual-en:start -->
== In VAR, if it is stable, it must be weakly stationary ==
<!-- bilingual-en:end -->

当$n \to \infty$，如果$A_1^{n+1} \to 0$（这正是$A_1$特征值小于1的要求），则无穷求和收敛，我们得到：
<!-- bilingual-en:start -->
When $n \to \infty$, if $A_1^{n+1} \to 0$ (this is exactly the requirement of $A_1$ eigenvalue less than 1), the infinite sum converges, and we get:
<!-- bilingual-en:end -->

$$

\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} A_1^i, e_{t-i}，

$$

其中 $\displaystyle \mu = (I - A_1)^{-1} A_0$ 是VAR过程的**无条件均值**（平稳均值）。换句话说，在稳定条件下，VAR模型是**协方差平稳过程**，其均值为$\mu$，方差有限且时间不变，并且可以表示为白噪声$e_t$的无限响应之和。
<!-- bilingual-en:start -->
where $\displaystyle \mu = (I - A_1)^{-1} A_0$ is the **unconditional mean** (stationary mean) of the VAR process.  In other words, under stable conditions, the VAR model is **covariance stationary process**, its mean is $\mu$, variance is finite and time-invariant, and can be expressed as the sum of the infinite responses of white noise $e_t$.
<!-- bilingual-en:end -->

>[!note] 协方差矩阵（Wold 表达式）：
>
> $x_t = \mu + \sum_{i=0}^\infty A_1^i e_{t-i}$
> 那么：
> $\begin{aligned} \text{Cov}(x_t) &= \mathbb{E}[(x_t - \mu)(x_t - \mu)’] = \sum_{i=0}^{\infty} A_1^i \Sigma (A_1^i)’ \\ &= \boxed{ (I - A_1)^{-1} \Sigma [(I - A_1)^{-1}]’ } \end{aligned}$
> 这里 $\Sigma = \text{Cov}(e_t)$，即 [[VAR、脉冲响应与 Granger 因果#reduced-form 与结构识别|reduced form]] 误差的协方差矩阵。
> <!-- bilingual-en:start -->
> **Covariance matrix (Wold expression): **
> $x_t = \mu + \sum_{i=0}^\infty A_1^i e_{t-i}$
> So:
> $\begin{aligned} \text{Cov}(x_t) &= \mathbb{E}[(x_t - \mu)(x_t - \mu)’] = \sum_{i=0}^{\infty} A_1^i \Sigma (A_1^i)’ \\ &= \boxed{ (I - A_1)^{-1} \Sigma [(I - A_1)^{-1}]’ } \end{aligned}$
> Here $\Sigma = \text{Cov}(e_t)$, the covariance matrix of [[VAR、脉冲响应与 Granger 因果#reduced-form 与结构识别|reduced form]] error.
> <!-- bilingual-en:end -->

## 4.3. [[VAR Model|VAR]] 的估计
<!-- bilingual-en:start -->
*4.3. Estimate of [[VAR Model|VAR]]*
<!-- bilingual-en:end -->

<span style="color: yellow;">简约的VAR可以使用OLS估计,但是简约的VAR不能反推结构化VAR.</span>
<!-- bilingual-en:start -->
<span style="color: yellow;">A simple VAR can be estimated using OLS, but a simple VAR cannot backtrack structured VAR.</span>
<!-- bilingual-en:end -->

>[!note] Cholesky 识别法
>我们前面提到过：[[VAR、脉冲响应与 Granger 因果#reduced-form 与结构识别|Reduced]]-form VAR $x_t = A_0 + A_1 x_{t-1} + e_t$ 只有 9 个估计量（如果二维系统），而结构系统（SVAR）有 10 个参数 ⇒ **未识别（under-identified）**
> 假设第一个变量对第二个变量有即时影响，但反过来没有：
>$B = \begin{bmatrix} 1 & b_{12} \\ 0 & 1 \end{bmatrix}$
> $$
> \begin{cases}
> y_t = b_{10} + b_{12} z_t + \gamma_{11} y_{t-1} + \gamma_{12} z_{t-1} + \varepsilon^y_t  \\
> z_t = b_{20}  + \gamma_{21} y_{t-1} + \gamma_{22} z_{t-1} + \varepsilon^z_t.
> \end{cases}
> $$
> 这样**原始结构参数数降为9个**（因为$b_{21}$被设为0）。而简约形式提供9个信息，因此可以完全识别。
> <!-- bilingual-en:start -->
> As noted above, a two-variable [[VAR、脉冲响应与 Granger 因果#reduced-form 与结构识别|reduced-form]] VAR $x_t=A_0+A_1x_{t-1}+e_t$ supplies only nine estimated quantities, whereas the unrestricted structural VAR has ten parameters. The structural system is therefore underidentified.
> Suppose the first variable has a contemporaneous effect on the second, but not vice versa:
> $B = \begin{bmatrix} 1 & b_{12} \\ 0 & 1 \end{bmatrix}$
> This restriction sets $b_{21}=0$ and reduces the number of **structural parameters to nine**. The nine reduced-form quantities are then sufficient to identify the structural system.
> <!-- bilingual-en:end -->

## 4.4. 脉冲响应函数在VAR中的应用
<!-- bilingual-en:start -->
*4.4. Application of impulse response function in VAR*
<!-- bilingual-en:end -->

在平稳VAR模型中，我们有简约形式的VMA表示：
<!-- bilingual-en:start -->
For a stationary VAR model, the reduced form has the following VMA representation:
<!-- bilingual-en:end -->

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Phi(i)e_{t-i}
$$

其中$\Phi(0) = I$（冲击对自身的即时影响矩阵即单位阵），$\Phi(1)=A_1, \Phi(2)=A_1^2$, … 一般$\Phi(i) = A_1^i$。**但是**，$e_{t-i}$并非结构冲击。如果想用结构冲击表示，可以利用 $e_t = B^{-1}\varepsilon_t$，也即 $\varepsilon_t = B e_t$。代入上式：
<!-- bilingual-en:start -->
$\Phi(0) = I$, $\Phi(1)=A_1, \Phi(2)=A_1^2$, ...   Generic $\Phi(i) = A_1^i$.  **But**$e_{t-i}$ is not a structural shock.  If you want to use structural shocks, you can use $e_t = B^{-1}\varepsilon_t$, or $\varepsilon_t = B e_t$.  Substitute in:
<!-- bilingual-en:end -->

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Phi(i) B^{-1} \varepsilon_{t-i}。
$$

定义 $\Psi(i) = \Phi(i)B^{-1} = A_1^i B^{-1}$，则：
<!-- bilingual-en:start -->
Define $\Psi(i) = \Phi(i)B^{-1} = A_1^i B^{-1}$, then:
<!-- bilingual-en:end -->

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Psi(i)\varepsilon_{t-i}，
$$

这就是使用**结构冲击**的VMA表示。矩阵$\Psi(i)$的元素 $\psi_{jk}(i)$ 就表示**第$k$个结构冲击在滞后$i$期对第$j$个变量的影响**。这组${\psi_{jk}(i)}$就是VAR的**冲击响应函数**([[Impulse Response Function|Impulse Response Functions]])。对于二维例子：
<!-- bilingual-en:start -->
This is the VMA representation in terms of **structural shocks**. The element $\psi_{jk}(i)$ of $\Psi(i)$ is **the effect of the $k$-th structural shock on the $j$-th variable after $i$ periods**. The collection ${\psi_{jk}(i)}$ forms the VAR's **[[Impulse Response Function|impulse response functions]]**. For a two-variable example:
<!-- bilingual-en:end -->

• $\psi_{11}(i)$：$y$对自身冲击$\varepsilon^y$在$i$期后的响应，
• $\psi_{12}(i)$：$y$对$z$的冲击$\varepsilon^z$在$i$期后的响应，
• $\psi_{21}(i)$：$z$对$y$冲击在$i$期后的响应，
• $\psi_{22}(i)$：$z$对自身冲击在$i$期后的响应。
<!-- bilingual-en:start -->
- $\psi_{11}(i)$: the response of $y$ after $i$ periods to its own shock $\varepsilon^y$;
- $\psi_{12}(i)$: the response of $y$ after $i$ periods to a shock $\varepsilon^z$ in $z$;
- $\psi_{21}(i)$: the response of $z$ after $i$ periods to a shock in $y$;
- $\psi_{22}(i)$: the response of $z$ after $i$ periods to its own shock.
<!-- bilingual-en:end -->

特别地，$\Psi(0) = B^{-1}$，其元素$\psi_{jk}(0)$被称为**冲击乘数**或**即刻影响系数**：$\psi_{jk}(0)$表示第$k$个冲击对第$j$个变量的当期影响（$i=0$即时刻）。
<!-- bilingual-en:start -->
In particular, $\Psi(0)=B^{-1}$. Its element $\psi_{jk}(0)$ is called an **impact multiplier** or **impact coefficient**: it gives the contemporaneous effect of structural shock $k$ on variable $j$ ($i=0$).
<!-- bilingual-en:end -->

==必须知道结构VAR才能进行脉冲响应分析==
<!-- bilingual-en:start -->
==Impulse-response analysis requires an identified structural VAR.==
<!-- bilingual-en:end -->

## 4.5. [[Granger Causality Test|格兰杰因果检验]]
<!-- bilingual-en:start -->
*4.5. [[Granger Causality Test|Granger causality test]]*
<!-- bilingual-en:end -->

>[!note] 格兰杰因果
>形式定义为：若包含变量 $X$ 的过去信息能够提高对变量 $Y$ 未来的预测，那么称 $X$ Granger成因于（Granger-cause） $Y$。数学表述为，对于任意事件集 $A$，
>
> $$
> P\{Y_{t+1}\in A \mid \mathcal{F}_t\} \neq P\{Y_{t+1}\in A \mid \mathcal{F}_{-X,t}\}
> $$
> <!-- bilingual-en:start -->
> **Granger causality**
> Formally, $X$ Granger-causes $Y$ if past information about $X$ improves the prediction of future $Y$ beyond the information already available without $X$. For any event set $A$, this can be written as:
> <!-- bilingual-en:end -->

说是没有放之四海而皆准的检验
<!-- bilingual-en:start -->
It says there's no universal test
<!-- bilingual-en:end -->

在VAR模型中，这个概念可简化为对滞后系数的检验。例如，对于二元VAR(p)，如果我们想检验“$y$ 是否格兰杰导致 $z$”，只需检验 $z_t$ 方程中 $y$ 的所有滞后系数是否同时为0。具体来说，$z_t$ 方程可表示为 $z_t = a_{20} + \sum_{i=1}^p a_{21,i} y_{t-i} + \sum_{i=1}^p a_{22,i} z_{t-i} + e_{2,t}$。$y$ 不格兰杰成因 $z$ 当且仅当 $a_{21,1} = a_{21,2} = \cdots = a_{21,p} = 0$。因此，可以通过对这些系数的联合零假设进行F检验或似然比检验来判断。如果拒绝假设，则认为 $y$ 的滞后总体上显著影响 $z$，即 $y$ 格兰杰致因 $z$；若不拒绝，则 $y$ 在有 $z$ 自身滞后作为控制后对 $z$ 没有预测力。
<!-- bilingual-en:start -->
In a VAR model, this becomes a joint test of lag coefficients. For a bivariate VAR($p$), testing whether $y$ Granger-causes $z$ means testing whether all coefficients on lagged $y$ in the $z_t$ equation are jointly zero. With $z_t = a_{20} + \sum_{i=1}^p a_{21,i} y_{t-i} + \sum_{i=1}^p a_{22,i} z_{t-i} + e_{2,t}$, $y$ does not Granger-cause $z$ if and only if $a_{21,1} = a_{21,2} = \cdots = a_{21,p} = 0$. An F-test or likelihood-ratio test can assess this joint null. Rejecting it means lagged $y$ adds predictive information for $z$; failing to reject means it does not, conditional on $z$'s own lags.
<!-- bilingual-en:end -->



# 5. 关联卡片
<!-- bilingual-en:start -->
*5. Associated Cards*
<!-- bilingual-en:end -->

- Time Series Analysis-hub
- [[回归预测与动态回归#动态回归与干预|Intervention Analysis]]
- [[回归预测与动态回归#分布滞后与动态乘数|ADL]]
- [[回归预测与动态回归#分布滞后与动态乘数|Distributed Lag Model]]
- [[回归预测与动态回归#干预变量|Transfer Function Model]]
- Cross-Correlation Function
- Leading Indicator
- [[VAR Model]]
- [[VAR、脉冲响应与 Granger 因果#reduced-form 与结构识别|Structural VAR]]
- [[VAR、脉冲响应与 Granger 因果#reduced-form 与结构识别|Reduced Form VAR]]
- [[Impulse Response Function]]
- [[VAR、脉冲响应与 Granger 因果#预测误差方差分解|Variance Decomposition]]
- [[Granger Causality Test]]

# 6. 作业
<!-- bilingual-en:start -->
*6. Operations*
<!-- bilingual-en:end -->

## 4. 第 4 题
<!-- bilingual-en:start -->
*4. Question 4*
<!-- bilingual-en:end -->
4. 考察传递函数模型
$$
y_t=0.5y_{t-1}+z_t+\varepsilon_t,\qquad
z_t=0.5z_{t-1}+\varepsilon_{zt}.
$$
<!-- bilingual-en:start -->
4. Examine the transfer function model
<!-- bilingual-en:end -->

a. 求滤波后的序列 $\{y_t\}$ 与序列 $\{\varepsilon_{zt}\}$ 间的互相关系数。
b. 现假设
$$
y_t=0.5y_{t-1}+z_t+0.5z_{t-1}+\varepsilon_t,\qquad
z_t=0.5z_{t-1}+\varepsilon_{zt},
$$
求滤波后的序列 $\{y_t\}$ 与 $\varepsilon_{zt}$ 的标准化互协方差；证明第 1 个和第 2 个互协方差成比例，并证明互协方差以 0.5 的比例衰减。
<!-- bilingual-en:start -->
a. Find the cross-correlation coefficient between the filtered series $\{y_t\}$ and $\{\varepsilon_{zt}\}$.
b. Now suppose that
$$
y_t=0.5y_{t-1}+z_t+0.5z_{t-1}+\varepsilon_t,\qquad
z_t=0.5z_{t-1}+\varepsilon_{zt}.
$$
Find the normalized cross-covariance between the filtered series $\{y_t\}$ and $\varepsilon_{zt}$. Show that the first two cross-covariances are proportional and that the cross-covariance decays by a factor of $0.5$ at each subsequent lag.
<!-- bilingual-en:end -->

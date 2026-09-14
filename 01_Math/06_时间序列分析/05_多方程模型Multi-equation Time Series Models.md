
# 0. 回忆用
<!-- bilingual-en:start -->
*0. Recall*
<!-- bilingual-en:end -->

> [!tip] 连续学习入口
> 动态回归、干预函数与 ADL 的连续学习见 [[01_Math/06_时间序列分析/08_动态回归与干预|动态回归与干预完整路径]]，全局关系见 [[回归预测与动态回归.canvas|主题 Canvas]]；VAR 从 [[01_Math/06_时间序列分析/05_VAR-脉冲响应与Granger因果|VAR 完整学习路径]] 进入。本文仍作为课堂板书、原题与作业的永久源记录。

<!-- bilingual-en:start -->
> [!tip] Continuous learning entries
> Use the [[01_Math/06_时间序列分析/08_动态回归与干预|complete dynamic-regression and intervention path]] with its [[回归预测与动态回归.canvas|topic Canvas]], then enter the [[01_Math/06_时间序列分析/05_VAR-脉冲响应与Granger因果|complete VAR path]]. This file remains the permanent source record for board work, original questions, and homework.
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
> 这里，$a_0$ 是常数项，$a_1$ 是劫机次数的**[[AR(p)模型|AR]](1)系数**（表示过去劫机次数对当前的影响），$c_0$ 是干预效应系数，$\varepsilon_t$ 是白噪声误差项。模型中包含一个滞后项$y_{t-1}$，假定序列在没有干预时服从AR(1)动态，并以$c_0 z_t$项刻画政策干预的即时影响。
> <!-- bilingual-en:start -->
> To test the effect of airport metal-detector screening, include an intervention variable in a time-series model. Let $y_t$ be the number of hijackings in period $t$, and let $z_t$ equal zero before 1973 Q1 and one from 1973 Q1 onward. A simple intervention model is
> $$y_t=a_0+a_1y_{t-1}+c_0z_t+\varepsilon_t,\qquad |a_1|<1.$$
> Here, $a_0$ is the constant, $a_1$ is the **[[AR(p)模型|AR(1)]] coefficient** governing how the previous period's hijacking count affects the current count, $c_0$ measures the intervention effect, and $\varepsilon_t$ is a white-noise error. The lagged term $y_{t-1}$ captures the series' ordinary AR(1) dynamics, while $c_0z_t$ captures the policy's immediate effect.
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
**Solving the model and the change in its long-run level**
<!-- bilingual-en:end -->

为了量化干预效应，我们可以**求解**上述方程，表示出 $y_t$ 关于冲击和干预项的无限滞后表示形式。因为这是一个AR(1)过程（$|a_1|<1$保证平稳），我们可以向前迭代展开：
<!-- bilingual-en:start -->
To quantify the intervention effect, solve the equation above for the infinite distributed-lag representation of $y_t$ in terms of shocks and the intervention. Because this is an AR(1) process and $|a_1|<1$ ensures stationarity, we can expand it recursively:
<!-- bilingual-en:end -->

通过递归替换 $y_{t-1}$，得到：
<!-- bilingual-en:start -->
Repeatedly substituting for $y_{t-1}$ gives
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
- The first term, $a_0/(1-a_1)$, is the **steady-state mean** without the intervention or random shocks.
- The second term accumulates the current and past values of the intervention through $c_0$ and the AR propagation factor $a_1$. Because $z_t$ remains one after 1973 Q1, this term captures the policy-induced level shift.
- The third term accumulates random innovations through the AR(1) dynamics.
<!-- bilingual-en:end -->

**长期均值的变化：** 没有干预时，系列的长期均值为 $\frac{a_0}{1 - a_1}$。当干预施行后，由于 $z_t$ 从0变为1 且持续为1，序列将收敛到新的均值。可以将上式取 $t \to \infty$ 且假定$\varepsilon$均值为0，则长期均值 $y_{\infty}$ 满足：
<!-- bilingual-en:start -->
**Change in the long-run mean:** before the intervention, the long-run mean is $a_0/(1-a_1)$. Once $z_t$ switches to one permanently, the conditional mean converges to a new level. Taking expectations and letting the time since intervention grow gives
<!-- bilingual-en:end -->

$$
y_{\infty} = \frac{a_0}{1 - a_1} + c_0 \sum_{i=0}^{\infty} a_1^i \cdot 1 = \frac{a_0}{1-a_1} + \frac{c_0}{1 - a_1} = \frac{a_0 + c_0}{1 - a_1}。
$$


可以看出，干预使序列的漂移项从 $a_0$ 变为 $a_0 + c_0$，从而将**长期均值**提高（或降低）了 $\frac{c_0}{1 - a_1}$。这个长期效应就是干预的永久影响。当 $c_0<0$ 时，新的长期均值低于原先，说明政策永久性地减少了劫机事件发生的平均水平；$c_0>0$ 则反之。这个结果直观：$c_0$ 是干预导致的瞬时变动，而 $\frac{c_0}{1 - a_1}$ 表示经过AR(1)动态放大后，干预对平稳均值的总改变量。
<!-- bilingual-en:start -->
The intervention changes the effective intercept from $a_0$ to $a_0+c_0$, shifting the **long-run mean** by $c_0/(1-a_1)$. If $c_0<0$, the policy permanently lowers the mean number of hijackings; if $c_0>0$, it raises it. The immediate coefficient $c_0$ is propagated through the AR(1) dynamics to produce the larger total level change.
<!-- bilingual-en:end -->

## 1.2. 脉冲响应与阶跃响应
<!-- bilingual-en:start -->
*1.2. Pulse and step responses*
<!-- bilingual-en:end -->

>[!note] 先区分 pulse 与 step
>在这个[[干预函数类型|干预模型]]中，**pulse** 是 $z_t$ 只在一期从 0 变为 1，之后立即回到 0；**step** 是从 $t$ 起永久地从 0 变为 1。两者的期限响应不同，不能把“一次脉冲”与“永久干预”混为同一输入。
>
> 对孤立单位脉冲，任意 $j\ge0$ 的响应为
>
> $$
> \frac{\partial y_{t+j}}{\partial z_t} = c_0  a_1^j, \qquad j = 0,1,2,\dots
> $$
>
> 当 $|a_1|<1$ 时，这条 pulse response 按几何速度衰减并趋于 0。对永久 step，期限 $j$ 的水平响应则是截至该期所有 pulse response 的和：
> $$
> c_0\sum_{i=0}^{j}a_1^i.
> $$
> 它随 $j\to\infty$ 趋于 $c_0/(1-a_1)$，所以相对于无干预反事实形成永久水平位移。
> <!-- bilingual-en:start -->
> An impulse response traces how a one-period change in the intervention input affects future $y$. For an isolated unit pulse in $z_t$,
> $$\frac{\partial y_{t+j}}{\partial z_t}=c_0a_1^j,\qquad j=0,1,2,\ldots.$$
> The contemporaneous response is $c_0$, and each later response is multiplied by $a_1$, so it decays geometrically to zero when $|a_1|<1$.
>
> A permanent intervention dummy is a **step**, not a one-period pulse: $z_t,z_{t+1},\ldots$ all change from zero to one. Its effect at horizon $j$ is therefore the sum of the pulse responses through that horizon, and the level of $y$ moves permanently relative to the no-intervention counterfactual.
> <!-- bilingual-en:end -->

因此，**累计 pulse response** 与**永久 step 的期限响应**在这个线性模型中数值相同：
<!-- bilingual-en:start -->
The **cumulative impulse response** through horizon $j$—equivalently, the horizon-$j$ response to a permanent step intervention—is the sum of the individual pulse responses:
<!-- bilingual-en:end -->

$$
I_t(j) = \sum_{i=0}^{j} \frac{\partial y_{t+i}}{\partial z_t} = c_0 [1 + a_1 + a_1^2 + \cdots + a_1^j]。
$$

这是一个**部分和**，随着$j$增大逐渐逼近$c_0/(1 - a_1)$，与前述长期效应相符。当$j \to \infty$时，$I_t(j)$的极限即为干预的长期总影响 $\frac{c_0}{1 - a_1}$。图形上看，$I_t(j)$曲线会逐渐趋于平稳，水平线对应长期效应值。
<!-- bilingual-en:start -->
This geometric **partial sum** approaches $c_0/(1-a_1)$ as $j$ increases, matching the long-run level shift derived above. A plot of $I_t(j)$ therefore converges to a horizontal asymptote at the long-run effect.
<!-- bilingual-en:end -->

## 1.3. 干预效应的识别与估计步骤
<!-- bilingual-en:start -->
*1.3. Steps for identification and estimation of intervention effects*
<!-- bilingual-en:end -->

1. **样本分段检验：** 先对干预发生前和发生后的数据分别拟合合适的ARIMA模型，并比较估计的模型参数是否存在显著差异。例如，在劫机案例中，可以分别用1973年前的数据和1973年后的数据估计AR(1)模型，检查$a_1$等系数有无明显变化。这可以帮助确认干预可能影响了数据生成过程。
<!-- bilingual-en:start -->

&nbsp;
**1.** **Compare subsamples.** Fit suitable ARIMA models before and after the intervention and test whether their parameters differ materially. In the hijacking example, estimate AR(1) models separately before and after 1973 and compare $a_1$ and the other coefficients. A difference suggests that the intervention may have changed the data-generating process.<br>
<!-- bilingual-en:end -->

2. **构建全样本初始模型：** 使用包含整个样本期的所有数据，先不考虑干预项，尝试识别一个适合的ARIMA模型作为**基准模型**。比如，用1968–1978年的劫机数据拟合一个ARIMA模型（也可能需要差分使其平稳），以确保对干预效应以外的动态有正确建模。
<!-- bilingual-en:start -->

&nbsp;
**2.** **Build an initial full-sample model:** Using the complete sample but omitting the intervention term, identify a suitable ARIMA **benchmark model**. For example, fit an ARIMA model to the 1968–1978 hijacking data, differencing if necessary to achieve stationarity, so that dynamics unrelated to the intervention are modeled properly.<br>
<!-- bilingual-en:end -->

3. **加入干预项并估计：** 在基准模型中加入干预哑变量（或相应的干预函数）$z_t$，估计**干预模型**的参数，包括$c_0$。此时检验：
<!-- bilingual-en:start -->

&nbsp;
**3.** **Add the intervention and estimate the model.** Include the dummy or intervention function $z_t$ in the benchmark specification and estimate all parameters, including $c_0$. Then check:<br>
<!-- bilingual-en:end -->

• 干预系数$c_0$是否显著；显著只表示给定模型中的条件差异可从零区分，不能单独证明干预造成了该变化。
<!-- bilingual-en:start -->
- whether $c_0$ is statistically distinguishable from zero in the stated model, without treating significance alone as causal identification;
<!-- bilingual-en:end -->

• 其它模型系数在加入干预后是否依然合理。
<!-- bilingual-en:start -->
- whether the other coefficient estimates remain plausible after adding the intervention; and
<!-- bilingual-en:end -->

• 干预模型的残差是否近似白噪声（无自相关）。
<!-- bilingual-en:start -->
- whether the intervention-model residuals are approximately white noise.
<!-- bilingual-en:end -->

4. **模型诊断与比较：** 对比包含干预项的模型与不含干预项的基准模型，或者与其他可能的备选模型，使用信息准则（[[AIC]]、SBC/[[BIC]]）来评估优劣。理想情况下，含干预模型应当有更低的信息准则值、残差更随机，以及**优于**不包含干预的模型。
<!-- bilingual-en:start -->

&nbsp;
**4.** **Model diagnosis and comparison:** Compare the intervention model with the no-intervention benchmark and other plausible alternatives using information criteria ([[AIC]], SBC/[[BIC]]). A well-specified intervention model should have lower information-criterion values and residuals that are closer to white noise than the corresponding benchmark.<br>
<!-- bilingual-en:end -->

> [!attention] 统计干预不自动等于因果干预
> 上述流程检验的是给定时间序列规格中的条件差异。若政策因预期结果恶化而启动、同日还有其他变化，或主体提前反应，即使 $c_0$ 显著且残差近似白噪声，也不能单独识别政策反事实效应；见 [[干预系数因果边界]]。
> <!-- bilingual-en:start -->
> The workflow estimates a conditional difference in the stated time-series model. Anticipatory adoption, coincident changes, or pre-event behavioural responses prevent a causal interpretation even when $c_0$ is significant and residuals look white; see the [[干预系数因果边界|causal boundary of intervention analysis]].
> <!-- bilingual-en:end -->

# 2. 政策分析
<!-- bilingual-en:start -->
*2. Policy analysis*
<!-- bilingual-en:end -->

废话
<!-- bilingual-en:start -->
Background remarks.
<!-- bilingual-en:end -->

# 3. [[自回归分布滞后|自回归分布滞后 ADL]]  Autoregressive Distributed Lag
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
Consider the general model
<!-- bilingual-en:end -->

 其中 $A(L)$、$C(L)$ 和 $B(L)$ 分别是在滞后算子 $L$ 上的多项式。更具体地：
<!-- bilingual-en:start -->
where $A(L)$, $C(L)$, and $B(L)$ are polynomials in the lag operator $L$:
<!-- bilingual-en:end -->

 • $A(L)$ 表示 $y$ 的自回归部分，如 $A(L)y_{t-1} = a_1 y_{t-1} + a_2 y_{t-2} + \cdots + a_p y_{t-p}$（这里为了方便，$A(L)$ 不写常数1项，只表示滞后项的和）。
<!-- bilingual-en:start -->
- $A(L)$ represents the autoregressive component, for example $A(L)y_{t-1}=a_1y_{t-1}+a_2y_{t-2}+\cdots+a_py_{t-p}$. Under this notation, $A(L)$ contains only the lag coefficients, not a leading 1.
<!-- bilingual-en:end -->

 • $C(L)$ 表示 $z$ 对 $y$ 的**传递函数**（transfer function），可写作 $C(L) = c_0 + c_1 L + c_2 L^2 + \cdots + c_r L^r$，意味着 $z_t$ 的当期值和多个滞后值共同作用于 $y_t$。
<!-- bilingual-en:start -->
- $C(L)$ is the **transfer function** from $z$ to $y$. Writing $C(L) = c_0 + c_1 L + c_2 L^2 + \cdots + c_r L^r$ means that the current and several lagged values of $z_t$ jointly affect $y_t$.
<!-- bilingual-en:end -->

 • $B(L)$ 表示 $\epsilon_t$ 的移动平均(MA)部分，如 $B(L)\epsilon_t = \epsilon_t + b_1 \epsilon_{t-1} + \cdots + b_q \epsilon_{t-q}$。很多情况下我们可以假定 $B(L)=1$（即噪声项为白噪声，无MA结构），以简化分析。
<!-- bilingual-en:start -->
- $B(L)$ represents the moving-average component of the disturbance, for example $B(L)\epsilon_t=\epsilon_t+b_1\epsilon_{t-1}+\cdots+b_q\epsilon_{t-q}$. Setting $B(L)=1$ assumes white-noise errors with no MA dynamics and simplifies the analysis.
<!-- bilingual-en:end -->

>[!note] [[自回归分布滞后|自回归分布滞后模型]]
>
> **自回归分布滞后模型**（Autoregressive Distributed Lag Model, 简称 **ADL 模型**）就是干预模型的自然延伸。在 ADL 模型中，$z_t$ 可以是随机的外生变量序列.在上面的模型的基础上当我们忽略$B(L)$（设$B(L)=1$）时，就得到**自回归分布滞后（ADL）模型**：
$$y_t = a_0 + A(L)y_{t-1} + C(L) z_t + \varepsilon_t$$
<!-- bilingual-en:start -->
>The **[[自回归分布滞后|autoregressive distributed lag model]]** (Autoregressive Distributed Lag Model, abbreviated **ADL**) is a natural extension of the intervention model. In an ADL model, $z_t$ may be a stochastic exogenous variable. Setting $B(L)=1$ in the model above gives:
$$y_t = a_0 + A(L)y_{t-1} + C(L)z_t + \varepsilon_t$$
<!-- bilingual-en:end -->

• $C(L)$ 被称为**传递函数**（transfer function），因为它描述了外生变量$z_t$的变动如何通过不同滞后传递到内生变量$y_t$上。例如，如果 $C(L) = c_0 + c_1L + c_2L^2$，说明$z_t$对$y_t$有当期效应$c_0$，对下一期$y$有滞后一阶效应$c_1$，对再下一期有滞后两期效应$c_2$，以此类推。
<!-- bilingual-en:start -->
- $C(L)$ is the **transfer function** because it describes how changes in the exogenous variable $z$ reach the endogenous variable $y$ across lags. If $C(L)=c_0+c_1L+c_2L^2$, a unit change in $z_t$ affects $y_t$ by $c_0$, $y_{t+1}$ by $c_1$, and $y_{t+2}$ by $c_2$, before accounting for any further propagation through the AR dynamics of $y$.
<!-- bilingual-en:end -->

• $C(L)$ 多项式的系数${c_i}$称为**传递函数权重**。它们刻画了$z$的一个单位变化对$y$在不同滞后期的影响强度。
<!-- bilingual-en:start -->
- The coefficients $c_i$ are **transfer-function weights**. They measure the direct effect of a unit change in $z$ at each lag.
<!-- bilingual-en:end -->

• 由于$z_t$影响被分布在多个时滞上，因此这部分 $C(L)z_t$ 构成**[[分布滞后模型|分布滞后结构]]**。
<!-- bilingual-en:start -->
- Because the effect of $z_t$ is distributed across several lags, $C(L)z_t$ is the model's **[[分布滞后模型|distributed-lag component]]**.
<!-- bilingual-en:end -->

>[!note] leading indicator
>在传递函数$C(L)$中，**$c_0$系数的重要性**值得注意：
>
> • 如果 $c_0 \neq 0$，则$z_t$对$y_t$有**当期直接影响**。$z_t$的变动立即反映在同时期的$y_t$上。
>
> • 如果 $c_0 = 0$，意味着**无当期效应**，$z_t$的变化不会直接影响当期$y_t$，最早的影响要经过至少1期滞后才能体现。在这种情况下，我们称$z_t$为**领先指标**（leading indicator），因为$z_t$的变化领先于$y_t$的变化。例如，如果经济指标$X$在时间上领先于$Y$，那么当前$Y$主要受之前的$X$影响，而当前$X$对当前$Y$没有直接影响。
> <!-- bilingual-en:start -->
> The coefficient $c_0$ determines whether the input has a contemporaneous effect:
> - If $c_0\neq0$, $z_t$ has an **immediate effect** on $y_t$.
> - If $c_0=0$, the earliest effect occurs after at least one lag. In that sense, $z$ is a **leading indicator** for $y$: changes in $z$ precede the changes they help predict in $y$.
> <!-- bilingual-en:end -->

## 3.2. ADL的模型性质
<!-- bilingual-en:start -->
*3.2. Properties of ADL*
<!-- bilingual-en:end -->

考虑一个简单情形来探究ADL模型的统计性质和如何识别滞后效应：**[[AR(p)模型|AR]](1)过程 + 延迟$d$期的单一滞后效应**。具体模型：
<!-- bilingual-en:start -->
To study the statistical properties of an ADL model and identify its delay, consider an [[AR(p)模型|AR]](1) process with a single effect from $z$ after $d$ periods:
<!-- bilingual-en:end -->

$$
y_t = a_1 y_{t-1} + c_d z_{t-d} + \varepsilon_t. \tag{ADL(1,d)}
$$

其中 $z_t$ 是外生的白噪声过程（均值0，方差$\sigma_z^2$），$\varepsilon_t$是白噪声误差（与$z_t$独立），$d \ge 0$为整数，表示$z$对$y$影响的延迟长度。
<!-- bilingual-en:start -->
Here $z_t$ is exogenous white noise with mean zero and variance $\sigma_z^2$, $\varepsilon_t$ is white noise independent of $z_t$, and the integer $d\ge0$ is the delay before $z$ affects $y$.
<!-- bilingual-en:end -->

这个模型意味着：$z$对$y$的**唯一影响**发生在滞后$d$期处，而且影响强度为$c_d$。例如若$d=2$，则$z_{t-2}$影响$y_t$，$z_{t-1}$和$z_t$对$y_t$无直接作用。
<!-- bilingual-en:start -->
The only direct input term is $c_dz_{t-d}$. If $d=2$, for example, $z_{t-2}$ directly affects $y_t$, while $z_{t-1}$ and $z_t$ do not.
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
> The cross-correlation function is
> $$\rho_{yz}(i)=\frac{\operatorname{Cov}(y_t,z_{t-i})}{\sigma_y\sigma_z},$$
> where $\sigma_y$ and $\sigma_z$ are the standard deviations of the two series. It measures their linear association at lead or lag $i$. Plotting it against $i$, including negative lags, produces the **cross-correlation function (CCF) plot**.
> <!-- bilingual-en:end -->

对于模型ADL(1,d)，我们可以根据模型结构推导出理论上的协方差$\mathrm{cov}(y_t, z_{t-i})$。因为$z$是白噪声且独立于$y$的冲击，利用滞后运算展开$y_t$：
(使用的是:$\frac{1}{1 - a_1 L} = 1 + a_1 L + a_1^2 L^2 + a_1^3 L^3 + \cdots \quad \text{当 } |a_1| < 1$)
<!-- bilingual-en:start -->
For ADL(1,$d$), derive $\operatorname{Cov}(y_t,z_{t-i})$ by expanding the stable AR operator. Since $z$ is white noise and independent of $\varepsilon$,
$$\frac1{1-a_1L}=1+a_1L+a_1^2L^2+\cdots\qquad\text{when }|a_1|<1.$$
<!-- bilingual-en:end -->

$$y_t = \frac{c_d z_{t-d}}{1 - a_1 L} + \frac{\varepsilon_t}{1 - a_1 L} \Rightarrow y_t = c_d[z_{t-d} + a_1 z_{t-d-1} + a_1^2 z_{t-d-2} + \cdots] + \text{其他噪声项}$$
<!-- bilingual-en:start -->
$$y_t = \frac{c_d z_{t-d}}{1 - a_1 L} + \frac{\varepsilon_t}{1 - a_1 L} \Rightarrow y_t = c_d[z_{t-d} + a_1 z_{t-d-1} + a_1^2 z_{t-d-2} + \cdots] + \text{other noise terms}.$$
<!-- bilingual-en:end -->

由此可见，$y_t$涉及到$z$的滞后$d, d+1, d+2,\dots$期。进一步考虑协方差：
<!-- bilingual-en:start -->
Thus $y_t$ contains values of $z$ at lags $d,d+1,d+2,\ldots$. Now consider each possible lag in the cross-covariance.
<!-- bilingual-en:end -->

• 对于$i < d$：$z_{t-i}$是发生在$y_t$影响之前的$z$，$y_t$不包含如此“晚近”的$z$项（因为$y_t$最早只包括到$z_{t-d}$）。由于$z$和$\varepsilon$独立，且$z$是白噪声不自相关，可知 $\mathrm{cov}(y_t, z_{t-i})=0$，因此$\rho_{yz}(i)=0$。换言之，在干预生效延迟$d$期之前，$y$与$z$毫无线性关联。
<!-- bilingual-en:start -->
- If $i<d$, $y_t$ contains no term involving the more recent input $z_{t-i}$. Because $z$ is serially uncorrelated and independent of $\varepsilon$, $\operatorname{Cov}(y_t,z_{t-i})=0$ and hence $\rho_{yz}(i)=0$. There is no linear response before the delay $d$ has elapsed.
<!-- bilingual-en:end -->

• 对于$i = d$：$\mathrm{cov}(y_t, z_{t-d})$ 由 $y_t$中的 $c_d z_{t-d}$ 项贡献，其协方差为 $c_d \mathrm{cov}(z_{t-d}, z_{t-d}) = c_d \sigma_z^2$。同时$y_t$中的其它项（如误差项和不同滞后的$z$）要么与$z_{t-d}$独立，要么不包含$z_{t-d}$。所以 $\mathrm{cov}(y_t, z_{t-d}) = c_d \sigma_z^2$，因而 $\rho_{yz}(d) = \frac{c_d \sigma_z^2}{\sigma_y \sigma_z}$。在数值上，如果$c_d\neq0$，我们会在滞后$d$处观察到一个显著**峰值**相关。
<!-- bilingual-en:start -->
- If $i=d$, the common term is $c_dz_{t-d}$, so $\operatorname{Cov}(y_t,z_{t-d})=c_d\sigma_z^2$ and $\rho_{yz}(d)=c_d\sigma_z^2/(\sigma_y\sigma_z)$. If $c_d\neq0$, the population CCF first becomes nonzero at lag $d$; a sample CCF should show a corresponding spike, subject to sampling noise.
<!-- bilingual-en:end -->

• 对于$i = d+1$：$y_t$包含 $c_d a_1 z_{t-d-1}$ 项，与 $z_{t-(d+1)} = z_{t-d-1}$ 完全同步。该协方差为 $c_d a_1 \sigma_z^2$。但同时$y_t$中也有 $c_d z_{t-d}$项，它与$z_{t-d-1}$不相关（因不同期的$z$不相关）。所以 $\mathrm{cov}(y_t, z_{t-d-1}) = c_d a_1 \sigma_z^2$。因此$\rho_{yz}(d+1) = \frac{c_d a_1 \sigma_z^2}{\sigma_y \sigma_z}$。
<!-- bilingual-en:start -->
- If $i=d+1$, the matching term is $c_da_1z_{t-d-1}$. All other $z$ terms are uncorrelated with $z_{t-d-1}$, so $\operatorname{Cov}(y_t,z_{t-d-1})=c_da_1\sigma_z^2$ and $\rho_{yz}(d+1)=c_da_1\sigma_z^2/(\sigma_y\sigma_z)$.
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
- More generally, for $i=d+k$ with $k\ge0$, the matching term is $c_da_1^kz_{t-d-k}$. Therefore $\operatorname{Cov}(y_t,z_{t-i})=c_da_1^k\sigma_z^2$, and standardizing gives the correlation shown above.
<!-- bilingual-en:end -->

综合以上分析，可以总结ADL(1,d)模型下 $y$与$z$的交叉协方差（和相关）特征：
<!-- bilingual-en:start -->
The ADL(1,$d$) cross-covariance pattern is therefore:
<!-- bilingual-en:end -->

• **对于 $i < d$：** $E[y_t z_{t-i}] = 0$，因此 $\rho_{yz}(i) = 0$（在交叉相关图上，干预滞后之前所有点相关为零）。
<!-- bilingual-en:start -->
- **For $i<d$:** $E[y_tz_{t-i}]=0$, so $\rho_{yz}(i)=0$; the population CCF is zero before the input delay.
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
The preceding analysis assumes that the exogenous input $z_t$ is white noise, which makes $C(L)$ easy to identify. In practice, an input such as an interest rate or income usually has its own serial dynamics and may itself follow an AR or ARMA model.
<!-- bilingual-en:end -->

因此，一般的**传递函数模型**会包含对$z_t$动态的建模：
<!-- bilingual-en:start -->
A general **transfer-function model** therefore includes a separate dynamic equation for $z_t$:
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
This framework has **three distinct impulse responses**:
<!-- bilingual-en:end -->

1. **$z$序列自身的冲击响应：** $z_t$受到它自己的冲击$\varepsilon^z_t$时，如何通过(6)式在未来演化。这由$D(L)$决定，通常就是$z_t$的ACF/PACF性质。
2. **$y$序列自身的冲击响应：** $y_t$受到它自身的误差冲击$\varepsilon_t$时，通过(5)式$A(L)$的传递对未来$y$的影响（这类似于我们之前ARIMA模型中的IRF）。
3. **$z$序列冲击对$y$的响应：** 这是关键，我们关心外生变量的变动如何传递到内生变量。由于$z_t$本身有动态，我们需要综合(5)和(6)来分析**$z$的冲击传递到$y$**的过程。
<!-- bilingual-en:start -->

&nbsp;
**1.** **Response of $z$ to its own shock:** How $z_t$ evolves after its own innovation $\varepsilon^z_t$ through equation (6). This is governed by $D(L)$ and reflected in the ACF/PACF of $z_t$.<br>
**2.** **Response of $y$ to its own shock:** How an innovation $\varepsilon_t$ propagates through $A(L)$ in equation (5) and affects future values of $y$, analogous to an ARIMA impulse response.<br>
**3.** **Response of $y$ to a shock in $z$:** This is the central object: how a change in the exogenous variable is transmitted to the endogenous variable. Because $z_t$ has its own dynamics, equations (5) and (6) must be combined to trace transmission from **$z$ to $y$**.<br>
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

# 4. [[VAR、脉冲响应与 Granger 因果.canvas|向量自回归 VAR]]
<!-- bilingual-en:start -->
*4. Vector autoregression (VAR)*
<!-- bilingual-en:end -->

## 4.1. VAR定义
<!-- bilingual-en:start -->
*4.1. Definition of a VAR*
<!-- bilingual-en:end -->

[[VAR(p)模型|简约型 VAR]] 用系统变量的共同滞后描述联合线性动态；它不会因变量被共同建模就自动解决内生性或成为结构模型。
>[!note] 从候选结构式到简约式
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
> 把这组带同期反馈的方程写成一个**候选结构 VAR**：
> $$
> \begin{bmatrix} 1 & b_{12} \\ b_{21} & 1 \end{bmatrix} \begin{bmatrix} y_t \\ z_t \end{bmatrix} =\begin{bmatrix} b_{10} \\ b_{20} \end{bmatrix} + \begin{bmatrix} \gamma_{11} & \gamma_{12} \\ \gamma_{21} & \gamma_{22} \end{bmatrix} \begin{bmatrix} y_{t-1} \\ z_{t-1} \end{bmatrix} + \begin{bmatrix} \varepsilon_{yt} \\ \varepsilon_{zt} \end{bmatrix}
> $$
> 记作：
> $$B x_t = \Gamma_0 + \Gamma_1 x_{t-1} + \varepsilon_t$$
> 在 $B$ 非奇异时左乘 $B^{-1}$，得到**简约型 VAR**：
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
> In matrix notation, this is a candidate **structural VAR**
> $$B x_t = \Gamma_0 + \Gamma_1 x_{t-1} + \varepsilon_t.$$
> Premultiplying by $B^{-1}$ is a contemporaneous transformation. It preserves the white-noise property and yields the **reduced-form VAR**
> $$x_t = A_0 + A_1 x_{t-1} + e_t,$$
> where:
> - $A_0 = B^{-1} \Gamma_0$;
> - $A_1 = B^{-1} \Gamma_1$;
> - $e_t = B^{-1} \varepsilon_t$.
> <!-- bilingual-en:end -->

若结构扰动跨期为白噪声，固定非奇异变换后的 $e_t$ 仍跨期为白噪声。简约式右侧不含当期系统变量；在创新对过去回归量正交、设计矩阵满秩及相应动态正则条件下，可按 [[VAR逐方程OLS|逐方程 OLS]] 估计。这个变换没有“消除所有内生性”，而且不同方程的简约型创新仍可同期相关。
<!-- bilingual-en:start -->
If the structural disturbances are white noise over time, their fixed nonsingular transformation $e_t$ remains white noise over time. No current system variable appears on the right-hand side of the reduced form. Equation-by-equation OLS is therefore available when innovations are orthogonal to lagged regressors, the design has full rank, and the required dynamic regularity conditions hold. This does not eliminate every endogeneity problem, and innovations from different equations may remain contemporaneously correlated.
<!-- bilingual-en:end -->

由于$\text{Cov}(e_{1t}, e_{2t}) = \mathbb{E}[e_{1t} e_{2t}] = \mathbb{E} \left[ \frac{(\varepsilon_{yt} - b_{12} \varepsilon_{zt})(\varepsilon_{zt} - b_{21} \varepsilon_{yt})} {(1 - b_{12}b_{21})^2} \right]$
展开后，若 $\varepsilon_{yt}$ 与 $\varepsilon_{zt}$ 同期正交，则交叉乘积期望为零。按上面结构方程的负号约定，
$$
\operatorname{Cov}(e_{1t},e_{2t})
=\frac{-b_{21}\sigma_y^2-b_{12}\sigma_z^2}
{(1-b_{12}b_{21})^2}.
$$
<!-- bilingual-en:start -->
Since
$$\operatorname{Cov}(e_{1t},e_{2t})=E\left[\frac{(\varepsilon_{yt}-b_{12}\varepsilon_{zt})(\varepsilon_{zt}-b_{21}\varepsilon_{yt})}{(1-b_{12}b_{21})^2}\right],$$
independence of the structural shocks eliminates the cross-products. With the signs used in the displayed structural equations, the result is
$$\operatorname{Cov}(e_{1t},e_{2t})=\frac{-b_{21}\sigma_y^2-b_{12}\sigma_z^2}{(1-b_{12}b_{21})^2}.$$
<!-- bilingual-en:end -->

>[!note] 方差-协方差矩阵 $\Sigma = \mathbb{E}[e_t e_t']$
>
> $\Sigma = \begin{bmatrix} \text{Var}(e_{1t}) & \text{Cov}(e_{1t}, e_{2t}) \\ \text{Cov}(e_{1t}, e_{2t}) & \text{Var}(e_{2t}) \end{bmatrix}$
>
> <!-- bilingual-en:start -->
> The **variance–covariance matrix** of the reduced-form innovations is $\Sigma=E(e_te_t')$:
> $\Sigma = \begin{bmatrix} \text{Var}(e_{1t}) & \text{Cov}(e_{1t}, e_{2t}) \\ \text{Cov}(e_{1t}, e_{2t}) & \text{Var}(e_{2t}) \end{bmatrix}$
> <!-- bilingual-en:end -->

## 4.2. VAR稳定性和平稳性
<!-- bilingual-en:start -->
*4.2. VAR stability and stationarity*
<!-- bilingual-en:end -->

使用迭代法得到:$x_t = \left( I + A_1 + A_1^2 + \dots + A_1^n \right) A_0 + \sum_{i=0}^n A_1^i e_{t-i} + A_1^{n+1} x_{t-n-1}$.
<!-- bilingual-en:start -->
Iterating the VAR gives
$$x_t=\left(I+A_1+A_1^2+\cdots+A_1^n\right)A_0+\sum_{i=0}^nA_1^ie_{t-i}+A_1^{n+1}x_{t-n-1}.$$
<!-- bilingual-en:end -->

>[!note] [[VAR稳定根条件|VAR 的稳定性]]
>对这里的 VAR(1)，长期不爆炸要求：
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
> 因而要区分两个对象：$A_1$ 的特征值模必须小于 1；$\det(I-A_1L)=0$ 的滞后多项式根模必须大于 1。
> <!-- bilingual-en:start -->
> **Stability of VAR**
> For this VAR(1) process to be stable rather than explosive, we require:
> $\boxed{\lim_{n\to\infty}A_1^n=0}\quad\Longleftrightarrow\quad\text{every eigenvalue }\lambda\text{ of }A_1\text{ satisfies }|\lambda|<1.$
> Equivalently, the roots $z$ of the lag polynomial $\det(I-A_1z)=0$ must lie **outside** the unit circle, because those roots are reciprocals of the eigenvalues of $A_1$. For a two-variable VAR:
> $I - A_1 L = \begin{bmatrix} 1 - a_{11}L & -a_{12}L \\ -a_{21}L & 1 - a_{22}L \end{bmatrix}$
> Taking its determinant gives:
> $\det(I - A_1 L) = (1 - a_{11}L)(1 - a_{22}L) - a_{12}a_{21}L^2$
> Thus the two statements use different objects but are consistent: the eigenvalues of $A_1$ have modulus below one, while the roots of $\det(I-A_1L)=0$ have modulus above one.
> <!-- bilingual-en:end -->

稳定根条件与有限创新二阶矩保证存在唯一的 [[VAR因果VMA表示|因果协方差平稳解]]。只有从该平稳分布初始化时，过程才从第一期起弱平稳；任意有限初值产生的过渡项会消失，矩只是在以后逐渐收敛到平稳解。
<!-- bilingual-en:start -->
==A stable VAR initialized in its stationary distribution has a weakly stationary solution. With an arbitrary finite initial condition, its moments converge to those of that solution as the initial effect vanishes.==
<!-- bilingual-en:end -->

当$n \to \infty$，如果$A_1^{n+1} \to 0$（这正是$A_1$特征值小于1的要求），则无穷求和收敛，我们得到：
<!-- bilingual-en:start -->
If every eigenvalue of $A_1$ has modulus below one, then $A_1^{n+1}\to0$ and the infinite sum converges, giving
<!-- bilingual-en:end -->

$$

\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} A_1^i e_{t-i},

$$

其中 $\displaystyle \mu=(I-A_1)^{-1}A_0$ 是平稳解的无条件均值。在稳定条件与有限创新方差下，该平稳解可表示为简约型创新 $e_t$ 的无限响应之和；这不把从任意固定初值启动的早期过渡过程自动变成平稳过程。
<!-- bilingual-en:start -->
Here $\mu=(I-A_1)^{-1}A_0$ is the VAR's **unconditional mean**. Under stability and finite innovation variance, the stationary solution is covariance-stationary and has an infinite moving-average representation in the reduced-form innovations $e_t$.
<!-- bilingual-en:end -->

>[!note] 协方差矩阵（Wold 表达式）：
>
> $x_t = \mu + \sum_{i=0}^\infty A_1^i e_{t-i}$
> 那么：
> $$
> \Omega\equiv\operatorname{Cov}(x_t)
> =\sum_{i=0}^{\infty}A_1^i\Sigma(A_1^i)'
> $$
> 等价地，$\Omega$ 是 [[VAR无条件协方差|离散 Lyapunov 方程]]
> $$
> \Omega=A_1\Omega A_1'+\Sigma
> $$
> 的唯一有限解。这里 $\Sigma=\operatorname{Cov}(e_t)$ 是 [[简约型VAR创新|简约型创新]] 的协方差矩阵。一般不能把同期无条件协方差写成 $(I-A_1)^{-1}\Sigma[(I-A_1)^{-1}]'$；后者使用长期累计乘数并带有跨滞后交叉项。
> <!-- bilingual-en:start -->
> **Covariance matrix from the VMA representation**
> $x_t = \mu + \sum_{i=0}^\infty A_1^i e_{t-i}$
> Therefore
> $$\Omega\equiv\operatorname{Cov}(x_t)=\sum_{i=0}^{\infty}A_1^i\Sigma(A_1^i)'$$
> where $\Sigma=\operatorname{Cov}(e_t)$ is the covariance matrix of the reduced-form innovations. Equivalently, $\Omega$ solves the discrete Lyapunov equation $\Omega=A_1\Omega A_1'+\Sigma$. The long-run-multiplier expression $(I-A_1)^{-1}\Sigma[(I-A_1)^{-1}]'$ is not generally the contemporaneous covariance.
> <!-- bilingual-en:end -->

## 4.3. 简约型估计与结构识别
<!-- bilingual-en:start -->
*4.3. Reduced-form estimation and structural identification*
<!-- bilingual-en:end -->

简约型 VAR 可在条件满足时 [[VAR逐方程OLS|逐方程 OLS]] 估计；但 $\Sigma_e$ 只识别创新混合后的协方差，不能单独反推出结构冲击。
<!-- bilingual-en:start -->
A reduced-form VAR can be estimated equation by equation by OLS under the stated conditions, but its covariance matrix alone does not identify the structural shocks.
<!-- bilingual-en:end -->

[[结构VAR|结构 VAR]] 是在简约型联合动态之上，再用当期关系与可解释冲击说明经济传导的模型。简约型创新可同期相关，而且 [[简约型创新不是结构冲击|不能直接按方程名称命名]]。
<!-- bilingual-en:start -->
A [[结构VAR|structural VAR]] adds contemporaneous relations and interpretable shocks to the reduced-form dynamics. Reduced-form innovations may be contemporaneously correlated and cannot be named structural shocks directly from their equation labels.
<!-- bilingual-en:end -->

>[!note] [[SVAR识别条件|识别计数]]与 [[Cholesky递归识别|Cholesky]]
>若写 $e_t=P\nu_t$ 且 $E(\nu_t\nu_t')=I_K$，$P$ 有 $K^2$ 个未知元素，而对称 $\Sigma_e=PP'$ 只有 $K(K+1)/2$ 个独立矩。这个明确参数化下，常见精确识别至少还需 $K(K-1)/2$ 个独立、有效的限制。
>
>对给定变量顺序取 $P$ 为正对角下三角 Cholesky 因子，恰好加入这些上三角零。方向是：$P_{ij}=0$（$j>i$）表示第 $i$ 个变量当期不响应排在其后的第 $j$ 个冲击。改变排序就改变同期零限制与结果；代数分解本身不证明这种递归经济结构成立。
> <!-- bilingual-en:start -->
> With $e_t=P\nu_t$ and $E(\nu_t\nu_t')=I_K$, $P$ has $K^2$ unknown entries whereas the symmetric covariance $\Sigma_e=PP'$ supplies $K(K+1)/2$ independent moments. In this parameterization, exact identification commonly needs at least $K(K-1)/2$ independent valid restrictions. A lower-triangular Cholesky factor supplies those zeros for a chosen ordering; changing the ordering changes the restrictions and the results.
> <!-- bilingual-en:end -->

## 4.4. 脉冲响应函数在VAR中的应用
<!-- bilingual-en:start -->
*4.4. Impulse responses in a VAR*
<!-- bilingual-en:end -->

稳定 VAR 有 [[VAR因果VMA表示|简约型 VMA 表示]]：
<!-- bilingual-en:start -->
For a stationary VAR model, the reduced form has the following VMA representation:
<!-- bilingual-en:end -->

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Phi(i)e_{t-i}
$$

其中 $\Phi(0)=I$。对本节 VAR(1)，$\Phi(i)=A_1^i$；对 VAR($p>1$)，$\Phi(i)$ 必须按各滞后矩阵递推，或用 [[VAR伴随形式|伴随矩阵]] 计算。$e_t$ 是 [[简约型VAR创新|简约型创新]]；它可同期相关，且 [[简约型创新不是结构冲击|不是已经命名的结构冲击]]。若已通过有效方案识别候选结构式 $e_t=B^{-1}\varepsilon_t$，则：
<!-- bilingual-en:start -->
Here $\Phi(0)=I$. For the VAR(1) in this section, $\Phi(i)=A_1^i$; for VAR($p>1$), the matrices follow a recursion involving every lag matrix or are computed through companion form. The $e_t$ are reduced-form innovations. If an identifying scheme has justified $e_t=B^{-1}\varepsilon_t$, substitution gives:
<!-- bilingual-en:end -->

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Phi(i) B^{-1} \varepsilon_{t-i}。
$$

定义 $\Psi(i)=\Phi(i)B^{-1}$；仅在本节 VAR(1) 中才可进一步写为 $A_1^iB^{-1}$。于是：
<!-- bilingual-en:start -->
Define $\Psi(i)=\Phi(i)B^{-1}$, which reduces to $A_1^iB^{-1}$ only for this VAR(1). Then:
<!-- bilingual-en:end -->

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Psi(i)\varepsilon_{t-i}，
$$

在识别方案有效且冲击尺度已经说明时，这就是 [[结构脉冲响应|结构 VMA 与结构 IRF]]。$\psi_{jk}(i)$ 表示第 $k$ 个结构冲击在期限 $i$ 对第 $j$ 个变量的响应。对于二维例子：
<!-- bilingual-en:start -->
Provided the identifying scheme is valid and the shock scale is stated, this is the structural VMA. The element $\psi_{jk}(i)$ is the response of variable $j$ at horizon $i$ to structural shock $k$. For a two-variable example:
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

简约型单位创新响应和 [[广义脉冲响应|GIRF]] 不要求先识别完整 SVAR；但 [[广义脉冲响应边界|排序不变并不完成结构识别]]。要把曲线解释为某个经济结构冲击的因果路径，必须给出识别方案，并报告冲击尺度、变量单位、是否累计、期限和区间。
<!-- bilingual-en:start -->
Reduced-form unit-innovation responses and GIRFs can be computed without a fully identified SVAR. Economically interpretable structural responses, however, require an identifying scheme and explicit reporting of shock scale, variable units, cumulative versus point responses, horizons, and uncertainty.
<!-- bilingual-en:end -->

## 4.5. 预测误差方差分解
<!-- bilingual-en:start -->
*4.5. Forecast error variance decomposition*
<!-- bilingual-en:end -->

[[预测误差方差分解|FEVD]] 在指定期限 $h$ 下，把某个变量的预测误差方差分摊给一组已识别、按所用口径正交化的冲击。它不是无条件的“变量方差来源表”：份额随期限变化，并随 Cholesky 排序或其他结构识别方案改变。若只使用相关的简约型创新，不能直接把每个份额命名为经济冲击贡献。
<!-- bilingual-en:start -->
At a specified horizon $h$, FEVD allocates a variable's forecast error variance to a set of identified shocks under the chosen orthogonalization. It is not an unconditional table of the sources of a variable's variance: shares change with the horizon and with the Cholesky ordering or other identifying scheme.
<!-- bilingual-en:end -->

## 4.6. 格兰杰因果检验
<!-- bilingual-en:start -->
*4.6. Granger causality test*
<!-- bilingual-en:end -->

>[!note] [[Granger因果|Granger 因果]]
>令 $\mathcal F_t$ 包含 $X$ 的过去，$\mathcal F^{-X}_t$ 删除这部分历史。$X$ **不** Granger 导致 $Y$ 的分布定义要求，对所有相关期限 $h$ 和事件集 $A$，
>
> $$
> P\{Y_{t+h}\in A\mid\mathcal F_t\}
> =P\{Y_{t+h}\in A\mid\mathcal F^{-X}_t\}.
> $$
> 若该等式对某个期限或事件集失败，$X$ 的历史含有增量预测信息。结论依赖信息集、滞后阶数和模型类，不等于结构干预因果。
> <!-- bilingual-en:start -->
> **Granger causality**
> Granger noncausality requires the conditional distribution of future $Y$ to be unchanged when the history of $X$ is removed from the information set, for every relevant horizon and event. Granger causality is present if this equality fails for at least one such comparison. This is an information-set-dependent predictive relation, not structural intervention causality.
> <!-- bilingual-en:end -->

遗漏共同驱动、时间聚合、错误变换或新增控制都可能改变结论，具体见 [[Granger因果边界|解释边界]]。
<!-- bilingual-en:start -->
Omitted common drivers, temporal aggregation, incorrect transformations, and added controls can all change the conclusion; see the [[Granger因果边界|interpretation boundary]].
<!-- bilingual-en:end -->

没有脱离规格、放之四海而皆准的 Granger 检验。
<!-- bilingual-en:start -->
There is no single specification-free Granger-causality test; the implemented test depends on the chosen information set, lag order, and model class.
<!-- bilingual-en:end -->

在线性 VAR($p$) 中，[[VAR Granger检验|检验 $y$ 不 Granger 导致 $z$]]，就是联合检验 $z_t$ 方程中 $y$ 的全部滞后系数：
$$
H_0:a_{21,1}=\cdots=a_{21,p}=0.
$$
拒绝表示在给定变量集、滞后和样本下，至少一个 $y$ 的滞后为预测 $z$ 提供增量信息。**未拒绝只表示证据不足以排除联合零，不能改写成已经证明 $y$ 没有预测力。**
<!-- bilingual-en:start -->
In a linear VAR($p$), testing whether $y$ does not Granger-cause $z$ is a joint test that every coefficient on lagged $y$ in the $z_t$ equation equals zero. Rejection indicates incremental predictive information in the stated specification. Failure to reject is insufficient evidence against the joint null; it does not prove that predictive content is absent.
<!-- bilingual-en:end -->



# 5. 关联卡片

- Time Series Analysis-hub
- [[干预函数类型|Intervention functions]] · [[干预系数因果边界|causal boundary]]
- [[自回归分布滞后|ADL / ARDL]]
- [[分布滞后模型|Distributed lag model]] · [[分布滞后乘数|dynamic multipliers]]
- [[动态回归模型|Dynamic regression and transfer-function family]]
- Cross-Correlation Function
- Leading Indicator
- [[VAR、脉冲响应与 Granger 因果.canvas|VAR topic map]]
- [[结构VAR|Structural VAR]]
- [[SVAR识别条件|Structural VAR identification]]
- [[简约型VAR创新|Reduced-form VAR innovations]]
- [[简约型创新不是结构冲击|Reduced-form/structural-shock boundary]]
- [[Proxy SVAR|External-instrument SVAR]]
- [[外部工具识别条件|External-instrument validity]]
- [[结构脉冲响应|Impulse Response Function]]
- [[广义脉冲响应|Generalized Impulse Response]]
- [[广义脉冲响应边界|GIRF interpretation boundary]]
- [[预测误差方差分解|Forecast Error Variance Decomposition]]
- [[Granger因果|Granger Causality]]
- [[VAR Granger检验|Granger Causality Test]]
- [[Granger因果边界|Granger interpretation boundary]]

# 6. 作业
<!-- bilingual-en:start -->
*6. Exercises*
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

&nbsp;
**4.** Examine the transfer function model<br>
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

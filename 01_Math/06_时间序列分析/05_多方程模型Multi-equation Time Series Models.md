
# 0.回忆用

# 1. 引言

==废话==

## 1.1 干预分析

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
> 这里，$a_0$ 是常数项，$a_1$ 是劫机次数的**AR(1)系数**（表示过去劫机次数对当前的影响），$c_0$ 是干预效应系数，$\varepsilon_t$ 是白噪声误差项。模型中包含一个滞后项$y_{t-1}$，假定序列在没有干预时服从AR(1)动态，并以$c_0 z_t$项刻画政策干预的即时影响。

• 如果没有干预（$z_t=0$），模型退化为 $y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$，是一个均值为 $a_0/(1-a_1)$ 的稳定AR(1)过程。

• 干预后（$z_t=1$），$c_0$ 捕捉了政策对劫机数量的直接影响：$c_0$为负意味着政策使劫机次数立即降低，为正则表示提高。

**模型求解与长期水平变化**

为了量化干预效应，我们可以**求解**上述方程，表示出 $y_t$ 关于冲击和干预项的无限滞后表示形式。因为这是一个AR(1)过程（$|a_1|<1$保证平稳），我们可以向前迭代展开：

通过递归替换 $y_{t-1}$，得到：

$$
y_t = \frac{a_0}{1 - a_1} + c_0 \sum_{i=0}^{\infty} a_1^i  z_{t-i} + \sum_{i=0}^{\infty} a_1^i  \varepsilon_{t-i} 。
$$

上述展开由三部分组成：

• 第一项 $\frac{a_0}{1 - a_1}$ 是没有干预和无扰动时序列的**稳态均值**（漂移项贡献的长期水平）。
• 第二项表示干预哑变量对 $y_t$ 的累积影响：当前以及之前各期$z$的效应经过系数$c_0$和AR系数$a_1$的滞后传播累积到本期。由于$z_t$在1973Q1之后为1，在此之前为0，这一项会体现政策使序列水平改变的贡献。
• 第三项是误差项$\varepsilon$的影响经由AR(1)滞后传递的累积，对应序列自身的随机波动。

**长期均值的变化：** 没有干预时，系列的长期均值为 $\frac{a_0}{1 - a_1}$。当干预施行后，由于 $z_t$ 从0变为1 且持续为1，序列将收敛到新的均值。可以将上式取 $t \to \infty$ 且假定$\varepsilon$均值为0，则长期均值 $y_{\infty}$ 满足：

$$
y_{\infty} = \frac{a_0}{1 - a_1} + c_0 \sum_{i=0}^{\infty} a_1^i \cdot 1 = \frac{a_0}{1-a_1} + \frac{c_0}{1 - a_1} = \frac{a_0 + c_0}{1 - a_1}。
$$

  
可以看出，干预使序列的漂移项从 $a_0$ 变为 $a_0 + c_0$，从而将**长期均值**提高（或降低）了 $\frac{c_0}{1 - a_1}$。这个长期效应就是干预的永久影响。当 $c_0<0$ 时，新的长期均值低于原先，说明政策永久性地减少了劫机事件发生的平均水平；$c_0>0$ 则反之。这个结果直观：$c_0$ 是干预导致的瞬时变动，而 $\frac{c_0}{1 - a_1}$ 表示经过AR(1)动态放大后，干预对平稳均值的总改变量。

## 1.2 脉冲响应函数 IRF Impulse Response Analysis

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

那么按照这个定义,我们也可以定义**累计冲击响应函数**，表示干预发生后直到第$j$期末总的影响积累。由于干预在$t$期后一直“在位”，累计影响等于将各期的边际影响相加：

$$
I_t(j) = \sum_{i=0}^{j} \frac{\partial y_{t+i}}{\partial z_t} = c_0 [1 + a_1 + a_1^2 + \cdots + a_1^j]。
$$

这是一个**部分和**，随着$j$增大逐渐逼近$c_0/(1 - a_1)$，与前述长期效应相符。当$j \to \infty$时，$I_t(j)$的极限即为干预的长期总影响 $\frac{c_0}{1 - a_1}$。图形上看，$I_t(j)$曲线会逐渐趋于平稳，水平线对应长期效应值。

## 1.3 干预效应的识别与估计步骤

1. **样本分段检验：** 先对干预发生前和发生后的数据分别拟合合适的ARIMA模型，并比较估计的模型参数是否存在显著差异。例如，在劫机案例中，可以分别用1973年前的数据和1973年后的数据估计AR(1)模型，检查$a_1$等系数有无明显变化。这可以帮助确认干预可能影响了数据生成过程。

2. **构建全样本初始模型：** 使用包含整个样本期的所有数据，先不考虑干预项，尝试识别一个适合的ARIMA模型作为**基准模型**。比如，用1968–1978年的劫机数据拟合一个ARIMA模型（也可能需要差分使其平稳），以确保对干预效应以外的动态有正确建模。

3. **加入干预项并估计：** 在基准模型中加入干预哑变量（或相应的干预函数）$z_t$，估计**干预模型**的参数，包括$c_0$。此时检验：

• 干预系数$c_0$是否显著，显著则意味着干预有统计学上的显著影响。

• 其它模型系数在加入干预后是否依然合理。

• 干预模型的残差是否近似白噪声（无自相关）。

4. **模型诊断与比较：** 对比包含干预项的模型与不含干预项的基准模型，或者与其他可能的备选模型，使用信息准则（AIC、SBC/BIC）来评估优劣。理想情况下，含干预模型应当有更低的信息准则值、残差更随机，以及**优于**不包含干预的模型。

# 2.政策分析

废话

# 3. 自回归分布滞后ADL  Autoregressive Distributed Lag

## 3.1 ADL的定义

我们考虑如下的一般模型形式：
$$
 y_t = a_0 + A(L)y_{t-1} + C(L)z_t + B(L)\epsilon_t 
$$
 
 其中 $A(L)$、$C(L)$ 和 $B(L)$ 分别是在滞后算子 $L$ 上的多项式。更具体地：
 
 • $A(L)$ 表示 $y$ 的自回归部分，如 $A(L)y_{t-1} = a_1 y_{t-1} + a_2 y_{t-2} + \cdots + a_p y_{t-p}$（这里为了方便，$A(L)$ 不写常数1项，只表示滞后项的和）。
 
 • $C(L)$ 表示 $z$ 对 $y$ 的**传递函数**（transfer function），可写作 $C(L) = c_0 + c_1 L + c_2 L^2 + \cdots + c_r L^r$，意味着 $z_t$ 的当期值和多个滞后值共同作用于 $y_t$。
 
 • $B(L)$ 表示 $\epsilon_t$ 的移动平均(MA)部分，如 $B(L)\epsilon_t = \epsilon_t + b_1 \epsilon_{t-1} + \cdots + b_q \epsilon_{t-q}$。很多情况下我们可以假定 $B(L)=1$（即噪声项为白噪声，无MA结构），以简化分析。

>[!note] 自回归分布滞后模型
>
> **自回归分布滞后模型**（Autoregressive Distributed Lag Model, 简称 **ADL 模型**）就是干预模型的自然延伸。在 ADL 模型中，$z_t$ 可以是随机的外生变量序列.在上面的模型的基础上当我们忽略$B(L)$（设$B(L)=1$）时，就得到**自回归分布滞后（ADL）模型**：
$$y_t = a_0 + A(L)y_{t-1} + C(L) z_t + \varepsilon_t$$

• $C(L)$ 被称为**传递函数**（transfer function），因为它描述了外生变量$z_t$的变动如何通过不同滞后传递到内生变量$y_t$上。例如，如果 $C(L) = c_0 + c_1L + c_2L^2$，说明$z_t$对$y_t$有当期效应$c_0$，对下一期$y$有滞后一阶效应$c_1$，对再下一期有滞后两期效应$c_2$，以此类推。

• $C(L)$ 多项式的系数${c_i}$称为**传递函数权重**。它们刻画了$z$的一个单位变化对$y$在不同滞后期的影响强度。

• 由于$z_t$影响被分布在多个时滞上，因此此类模型也常被称为“**分布滞后模型**”。

>[!note] leading indicator
>在传递函数$C(L)$中，**$c_0$系数的重要性**值得注意：
> 
> • 如果 $c_0 \neq 0$，则$z_t$对$y_t$有**当期直接影响**。$z_t$的变动立即反映在同时期的$y_t$上。
> 
> • 如果 $c_0 = 0$，意味着**无当期效应**，$z_t$的变化不会直接影响当期$y_t$，最早的影响要经过至少1期滞后才能体现。在这种情况下，我们称$z_t$为**领先指标**（leading indicator），因为$z_t$的变化领先于$y_t$的变化。例如，如果经济指标$X$在时间上领先于$Y$，那么当前$Y$主要受之前的$X$影响，而当前$X$对当前$Y$没有直接影响。

## 3.2 ADL的模型性质

考虑一个简单情形来探究ADL模型的统计性质和如何识别滞后效应：**AR(1)过程 + 延迟$d$期的单一滞后效应**。具体模型：

$$
y_t = a_1 y_{t-1} + c_d z_{t-d} + \varepsilon_t, \tag{ADL(1, d)}
$$

其中 $z_t$ 是外生的白噪声过程（均值0，方差$\sigma_z^2$），$\varepsilon_t$是白噪声误差（与$z_t$独立），$d \ge 0$为整数，表示$z$对$y$影响的延迟长度。

这个模型意味着：$z$对$y$的**唯一影响**发生在滞后$d$期处，而且影响强度为$c_d$。例如若$d=2$，则$z_{t-2}$影响$y_t$，$z_{t-1}$和$z_t$对$y_t$无直接作用。

为了刻画$y$与$z$之间的动态相关关系，我们引入:
>[!note] 交叉相关函数(Cross-correlation function, CCF)：
> 
> $$
> \rho_{yz}(i) = \frac{\mathrm{cov}(y_t; z_{t-i})}{\sigma_y\sigma_z}
> $$
> 
> 其中$\sigma_y$和$\sigma_z$分别是$y$和$z$的标准差。$\rho_{yz}(i)$描述了$y_t$与$z$在领先/滞后$i$期时的线性相关强度。当我们绘制$\rho_{yz}(i)$随$i$变化的图（$i$可以取负值表示$y$落后于$z$的情形），就得到**交叉相关图**或**交叉相关函数图 (CCF图)**。 

对于模型ADL(1,d)，我们可以根据模型结构推导出理论上的协方差$\mathrm{cov}(y_t, z_{t-i})$。因为$z$是白噪声且独立于$y$的冲击，利用滞后运算展开$y_t$：
(使用的是:$\frac{1}{1 - a_1 L} = 1 + a_1 L + a_1^2 L^2 + a_1^3 L^3 + \cdots \quad \text{当 } |a_1| < 1$)

$$y_t = \frac{c_d z_{t-d}}{1 - a_1 L} + \frac{\varepsilon_t}{1 - a_1 L} \Rightarrow y_t = c_d[z_{t-d} + a_1 z_{t-d-1} + a_1^2 z_{t-d-2} + \cdots] + \text{其他噪声项}$$

由此可见，$y_t$涉及到$z$的滞后$d, d+1, d+2,\dots$期。进一步考虑协方差：

• 对于$i < d$：$z_{t-i}$是发生在$y_t$影响之前的$z$，$y_t$不包含如此“晚近”的$z$项（因为$y_t$最早只包括到$z_{t-d}$）。由于$z$和$\varepsilon$独立，且$z$是白噪声不自相关，可知 $\mathrm{cov}(y_t,; z_{t-i})=0$，因此$\rho_{yz}(i)=0$。换言之，在干预生效延迟$d$期之前，$y$与$z$毫无线性关联。

• 对于$i = d$：$\mathrm{cov}(y_t, z_{t-d})$ 由 $y_t$中的 $c_d z_{t-d}$ 项贡献，其协方差为 $c_d \mathrm{cov}(z_{t-d}, z_{t-d}) = c_d \sigma_z^2$。同时$y_t$中的其它项（如误差项和不同滞后的$z$）要么与$z_{t-d}$独立，要么不包含$z_{t-d}$。所以 $\mathrm{cov}(y_t, z_{t-d}) = c_d \sigma_z^2$，因而 $\rho_{yz}(d) = \frac{c_d \sigma_z^2}{\sigma_y \sigma_z}$。在数值上，如果$c_d\neq0$，我们会在滞后$d$处观察到一个显著**峰值**相关。

• 对于$i = d+1$：$y_t$包含 $c_d a_1 z_{t-d-1}$ 项，与 $z_{t-(d+1)} = z_{t-d-1}$ 完全同步。该协方差为 $c_d a_1 \sigma_z^2$。但同时$y_t$中也有 $c_d z_{t-d}$项，它与$z_{t-d-1}$不相关（因不同期的$z$不相关）。所以 $\mathrm{cov}(y_t, z_{t-d-1}) = c_d a_1 \sigma_z^2$。因此$\rho_{yz}(d+1) = \frac{c_d a_1 \sigma_z^2}{\sigma_y \sigma_z}$。

• 推广$i \ge d$：对一般 $i = d + k$（$k \ge 0$），$y_t$中与$z_{t-i}$同步的那一项是 $c_d a_1^{,k} z_{t-d-k}$（因为$y_t$包含$z_{t-d-k}$乘以$a_1^k c_d$）。于是：
$$
\mathrm{cov}(y_t, z_{t-i}) = c_d a_1^{,k} \sigma_z^2 \qquad \text{当 } i \ge d,
$$
即 $i=d+k$ 时协方差为 $c_d a_1^k \sigma_z^2$。转化为相关系数，
$$

\rho_{yz}(i) = \frac{c_d a_1^{,i-d} \sigma_z^2}{\sigma_y \sigma_z}, \qquad i \ge d。

$$

综合以上分析，可以总结ADL(1,d)模型下 $y$与$z$的交叉协方差（和相关）特征：

• **对于 $i < d$：** $E[y_t z_{t-i}] = 0$，因此 $\rho_{yz}(i) = 0$（在交叉相关图上，干预滞后之前所有点相关为零）。

• **对于 $i \ge d$：** $E[y_t z_{t-i}] = c_d a_1^{,i-d}  \sigma_z^2$。特别地，$i=d$ 时 $E[y_t z_{t-d}] = c_d \sigma_z^2$ 是第一个非零协方差；$i=d+1$ 时 $E[y_t z_{t-d-1}] = c_d a_1 \sigma_z^2$；随着滞后$i$增加，协方差按比率$a_1$几何衰减。

对应的CCF图像特征为：在滞后$d$之前相关为零，滞后$d$处出现**尖刺**（正或负，取决于$c_d$符号），随后相关系数按指数规律衰减。如果观测数据的样本交叉相关图呈现这种形状，我们可以推断$z$对$y$的主要影响开始于某个延迟$d$，并随后按AR(1)过程衰减，从而帮助我们识别$C(L)$和$A(L)$的大致形式。

## 3.3 外生输入序列存在自相关的情况

前述分析假定$z_t$是白噪声（无序列相关）的外生过程，这简化了$C(L)$的识别。然而，在现实中，外生变量$z_t$本身往往具有动态结构，而非独立同分布。例如，$z_t$可能是另一经济变量（如利率、收入等），它本身可以用AR或其它模型描述。

因此，一般的**传递函数模型**会包含对$z_t$动态的建模：

$$
\begin{aligned}
y_t &= a_0 + A(L)y_{t-1} + C(L)z_t + \varepsilon_t\\
z_t &= D(L)z_{t-1} + \varepsilon_{zt}
\end{aligned}
$$

这里 $D(L)$ 是$z_t$自身的滞后多项式，$\varepsilon_{zt}$是$z_t$过程的白噪声冲击。也就是说，我们将 $z_t$ 建模为一个ARMA过程（或近似如此）。

在这个框架下，我们可以考虑**三类脉冲响应函数**来全面理解系统动态：

1. **$z$序列自身的冲击响应：** $z_t$受到它自己的冲击$\varepsilon^z_t$时，如何通过(6)式在未来演化。这由$D(L)$决定，通常就是$z_t$的ACF/PACF性质。
2. **$y$序列自身的冲击响应：** $y_t$受到它自身的误差冲击$\varepsilon_t$时，通过(5)式$A(L)$的传递对未来$y$的影响（这类似于我们之前ARIMA模型中的IRF）。
3. **$z$序列冲击对$y$的响应：** 这是关键，我们关心外生变量的变动如何传递到内生变量。由于$z_t$本身有动态，我们需要综合(5)和(6)来分析**$z$的冲击传递到$y$**的过程。

具体来说，考虑第3种冲击：令$t$期发生$z$序列的一个冲击$\varepsilon_{zt}$（大小为1的单位冲击），并观察$y_t$随后各期的变化。因为$z_t$满足(6)，我们可以将(6)代入(5)，消除$z_t$：

$$
y_t = a_0 + A(L)y_{t-1} + \frac{C(L)}{1 - D(L)L}\varepsilon_{zt} + \varepsilon_t。
$$

## 3.4 识别与估计

上面的那个模型有识别问题,具体一点来讲,就是不能唯一确定值.

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
> y_t = a_1 y_{t-1} + c_1(d_1 z_{t-1} + \epsilon^z_t) + \epsilon_t = a_1 y_{t-1} + c_1 d_1 z_{t-1} + c_1 \epsilon^z_t + \epsilon_t,.
> 
> $$
> 
> 那么关于 $z$ 的影响项，我们既可以将其解释为“$z_t$ 对 $y_t$ 有当期影响 $c_1$，同时 $z_{t-1}$ 对 $y_t$ 有影响 $c_1 d_1$”（对应于原结构参数），也可以等价地解释为“$z_t$ 对 $y_t$ 没有直接影响（$c_1=0$），而是 $z_{t-1}$ 对 $y_t$ 有影响 $c_1 d_1$”（把 $c_1 d_1$ 看作新的滞后影响系数）。这就导致了**传递函数 $C(L)$ 无法唯一确定**：数据上看，你很难区分是 $c_1$ 作用于当期，还是 $c_1 d_1$ 作用于滞后一期，因为两种解释都符合观测。这就是所谓的识别问题。

有两种方法解决:

方法1：不预设传递函数结构，直接在多元模型中以信息准则/检验确定滞后阶与影响项。~~内容略.~~
$$
y_t = a_0 + \sum_{i=1}^{p} a_iy_{t-i} + \sum_{i=0}^{n} c_i z_{t-i} + \epsilon_t
$$

方法2：预滤波法（预白化法）识别传递函数。 这是前面推导中启发的方法：我们先估计并得到 $z_t$ 的模型 $D(L)$，然后用 $(1 - \hat{D}(L)L)$ 滤波（预白化）$y_t$ 和 $z_t$，以去除 $z$ 的自相关结构，使之接近白噪声。具体步骤如下：

# 4. 向量自回归VAR

## 4.1 VAR定义

==是用来处理内生性问题的工具==
>[!note] VAR
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

==简约VAR中的$e_{t}$依旧是白噪声,并且消除了内生性.那就意味着可以直接用OLS估计.但是扰动项出现了同期相关==

由于$\text{Cov}(e_{1t}, e_{2t}) = \mathbb{E}[e_{1t} e_{2t}] = \mathbb{E} \left[ \frac{(\varepsilon_{yt} - b_{12} \varepsilon_{zt})(\varepsilon_{zt} - b_{21} \varepsilon_{yt})} {(1 - b_{12}b_{21})^2} \right]$
展开乘法期望后，注意 $\varepsilon_{yt}, \varepsilon_{zt}$ 是**独立白噪声** ⇒ 所有交叉项消掉，只剩下：$\text{Cov}(e_{1t}, e_{2t}) = \frac{ - b_{21} \sigma_y^2 + b_{12} \sigma_z^2 } {(1 - b_{12}b_{21})^2}$.

>[!note] ##  **方差-协方差矩阵** $\Sigma = \mathbb{E}[e_t e_t’]$
> 
> $\Sigma = \begin{bmatrix} \text{Var}(e_{1t}) & \text{Cov}(e_{1t}, e_{2t}) \\ \text{Cov}(e_{1t}, e_{2t}) & \text{Var}(e_{2t}) \end{bmatrix}$
> 

## 4.2 VAR稳定性和平稳性

使用迭代法得到:$x_t = \left( I + A_1 + A_1^2 + \dots + A_1^n \right) A_0 + \sum_{i=0}^n A_1^i e_{t-i} + A_1^{n+1} x_{t-n-1}$.

>[!note] VAR的稳定性
>要让 VAR 模型在长期不爆炸（即收敛），我们要求：
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

==在VAR中,如果稳定,就一定弱平稳==

当$n \to \infty$，如果$A_1^{n+1} \to 0$（这正是$A_1$特征值小于1的要求），则无穷求和收敛，我们得到：

$$

\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} A_1^i, e_{t-i}，

$$

其中 $\displaystyle \mu = (I - A_1)^{-1} A_0$ 是VAR过程的**无条件均值**（平稳均值）。换句话说，在稳定条件下，VAR模型是**协方差平稳过程**，其均值为$\mu$，方差有限且时间不变，并且可以表示为白噪声$e_t$的无限响应之和。

>[!note] 协方差矩阵（Wold 表达式）：
> 
> $x_t = \mu + \sum_{i=0}^\infty A_1^i e_{t-i}$
> 那么：
> $\begin{aligned} \text{Cov}(x_t) &= \mathbb{E}[(x_t - \mu)(x_t - \mu)’] = \sum_{i=0}^{\infty} A_1^i \Sigma (A_1^i)’ \\ &= \boxed{ (I - A_1)^{-1} \Sigma [(I - A_1)^{-1}]’ } \end{aligned}$
> 这里 $\Sigma = \text{Cov}(e_t)$，即 reduced form 误差的协方差矩阵。

## 4.3 VAR 的估计

<span style="color: yellow;">简约的VAR可以使用OLS估计,但是简约的VAR不能反推结构化VAR.</span>

>[!note] Cholesky 识别法
>我们前面提到过：Reduced-form VAR $x_t = A_0 + A_1 x_{t-1} + e_t$ 只有 9 个估计量（如果二维系统），而结构系统（SVAR）有 10 个参数 ⇒ **未识别（under-identified）**
> 假设第一个变量对第二个变量有即时影响，但反过来没有：
>$B = \begin{bmatrix} 1 & b_{12} \\ 0 & 1 \end{bmatrix}$
> $$
> \begin{cases}
> y_t = b_{10} + b_{12} z_t + \gamma_{11} y_{t-1} + \gamma_{12} z_{t-1} + \varepsilon^y_t  \\
> z_t = b_{20}  + \gamma_{21} y_{t-1} + \gamma_{22} z_{t-1} + \varepsilon^z_t.
> \end{cases}
> $$
> 这样**原始结构参数数降为9个**（因为$b_{21}$被设为0）。而简约形式提供9个信息，因此可以完全识别。

## 4.4 脉冲响应函数在VAR中的应用

在平稳VAR模型中，我们有简约形式的VMA表示：

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Phi(i)e_{t-i}
$$

其中$\Phi(0) = I$（冲击对自身的即时影响矩阵即单位阵），$\Phi(1)=A_1, \Phi(2)=A_1^2$, … 一般$\Phi(i) = A_1^i$。**但是**，$e_{t-i}$并非结构冲击。如果想用结构冲击表示，可以利用 $e_t = B^{-1}\varepsilon_t$，也即 $\varepsilon_t = B e_t$。代入上式：

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Phi(i) B^{-1} \varepsilon_{t-i}。
$$

定义 $\Psi(i) = \Phi(i)B^{-1} = A_1^i B^{-1}$，则：

$$
\mathbf{x}_t = \mu + \sum_{i=0}^{\infty} \Psi(i)\varepsilon_{t-i}，
$$

这就是使用**结构冲击**的VMA表示。矩阵$\Psi(i)$的元素 $\psi_{jk}(i)$ 就表示**第$k$个结构冲击在滞后$i$期对第$j$个变量的影响**。这组${\psi_{jk}(i)}$就是VAR的**冲击响应函数**(Impulse Response Functions)。对于二维例子：

• $\psi_{11}(i)$：$y$对自身冲击$\varepsilon^y$在$i$期后的响应，
• $\psi_{12}(i)$：$y$对$z$的冲击$\varepsilon^z$在$i$期后的响应，
• $\psi_{21}(i)$：$z$对$y$冲击在$i$期后的响应，
• $\psi_{22}(i)$：$z$对自身冲击在$i$期后的响应。

特别地，$\Psi(0) = B^{-1}$，其元素$\psi_{jk}(0)$被称为**冲击乘数**或**即刻影响系数**：$\psi_{jk}(0)$表示第$k$个冲击对第$j$个变量的当期影响（$i=0$即时刻）。

==必须知道结构VAR才能进行脉冲响应分析==

## 4.5 格兰杰因果检验

>[!note] 格兰杰因果
>形式定义为：若包含变量 $X$ 的过去信息能够提高对变量 $Y$ 未来的预测，那么称 $X$ Granger成因于（Granger-cause） $Y$。数学表述为，对于任意事件集 $A$，
> 
> $$
> P\{Y_{t+1}\in A \mid \mathcal{F}_t\} \neq P\{Y_{t+1}\in A \mid \mathcal{F}_{-X,t}\}
> $$

说是没有放之四海而皆准的检验

在VAR模型中，这个概念可简化为对滞后系数的检验。例如，对于二元VAR(p)，如果我们想检验“$y$ 是否格兰杰导致 $z$”，只需检验 $z_t$ 方程中 $y$ 的所有滞后系数是否同时为0。具体来说，$z_t$ 方程可表示为 $z_t = a_{20} + \sum_{i=1}^p a_{21,i} y_{t-i} + \sum_{i=1}^p a_{22,i} z_{t-i} + e_{2,t}$。$y$ 不格兰杰成因 $z$ 当且仅当 $a_{21,1} = a_{21,2} = \cdots = a_{21,p} = 0$。因此，可以通过对这些系数的联合零假设进行F检验或似然比检验来判断。如果拒绝假设，则认为 $y$ 的滞后总体上显著影响 $z$，即 $y$ 格兰杰致因 $z$；若不拒绝，则 $y$ 在有 $z$ 自身滞后作为控制后对 $z$ 没有预测力。


# 作业

## 4.
4. REE PRR y, =0.5y,, +z, +6,, # 中 ，z 为 自 回归 过 程 x =0. 5z,_, +e., a. 求 滤波 后 的 序列 |y,| 与 序列 { se。 | 间 的 互 相关 系数 。 b. 现 假设 y =0.5y, ,+z +0.5z ,+s,， 并 Az, =0.5z ,+es， 求 滤波 后 的 序列 |y,| 与 ss 标准 化 互 协 方差 。 证 明 第 1 个 和 第 2 个 互 协 方差 成 比例 。 证 明 互 协 方差 以 0.5 的 比例 衰减 。
![[Pasted image 20250525194711.png]]

## 6.
Use (5.28) to find the appropriate second-order stochastic difference equation for y, ; . cg: oe wee y, _ 08 0.2 Vy . el Hh ER ety 和 0.2 0.8j|z- e>, a 7 oe A. Determine whether the {y,} sequence is stationary. - oe B. Discuss the shape of the impulse response function of y, to a one-trit shock in el and a one-unit shock in e,,. mae C. Suppose evy = er + 0.5e,, and that ez = €,,. Discuss the shape of the impulse response function of y, to a one-unit shock in ¢€,,. Repeat for a one-unit shock in €,,. D. Suppose e,, = €,, and that ex = 0.5¢,, + er Discuss the shape of the impulse response function of y, to a one unit shock in ev Repeat for a one-unit shock in €,,. E. Use your answers to C and D to explain why the ordering in a Choleski de- composition is important. F. Using the notation in (5.21), find A? and A}. Does A7 appear to approach zero (i.e., the null matrix)?
Use (5.28) to find the appropriate second-order stochastic difference equation for $y_t$:\n\n$$\n\begin{bmatrix} y_t \\ z_t \end{bmatrix} = \begin{bmatrix} 0.8 & 0.2 \\ 0.2 & 0.8 \end{bmatrix}\begin{bmatrix} y_{t-1} \\ z_{t-1} \end{bmatrix}+\begin{bmatrix} e_{1t} \\ e_{2t} \end{bmatrix}.\n$$\n\nA. Determine whether the $\{y_t\}$ sequence is stationary.\nB. Discuss the shape of the impulse response function of $y_t$ to a one-unit shock in $e_{1t}$ and a one-unit shock in $e_{2t}$.\nC. Suppose $e_{1t}=e_{yt}+0.5e_{zt}$ and that $e_{2t}=e_{yt}$. Discuss the shape of the impulse response function of $y_t$ to a one-unit shock in $e_{yt}$. Repeat for a one-unit shock in $e_{zt}$.\nD. Suppose $e_{1t}=e_{yt}$ and that $e_{2t}=0.5e_{yt}+e_{zt}$. Discuss the shape of the impulse response function of $y_t$ to a one-unit shock in $e_{yt}$. Repeat for a one-unit shock in $e_{zt}$.\nE. Use your answers to C and D to explain why the ordering in a Choleski decomposition is important.\nF. Using the notation in (5.21), find $A_1^2$ and $A_1^3$. Does $A_1^n$ appear to approach zero (i.e., the null matrix)?

## 8.
6. Suppose the residuals of a VAR are such that var(e,) = 0.75, var(e,) = 0.5, and Cov(el ez) = 0.25. A. Using (5.53) through (5.56) as guides, show that it is not possible to iden- tify the structural VAR. ‘-¢ B, Using Choleski decomposition such that b,, = 0, find the identified values of b,,, var(e,), and var(e,). C. Using Choleski decomposition such that b,, = 0, find the identified values of b,, var(e,), and var(e,). D. Using a Sims-Bernanke decomposition such that 6, = 0.5, find the identi- fied values of bj, var(e,), and Var(e,).
6. Suppose the residuals of a VAR are such that $\operatorname{var}(e_1)=0.75$, $\operatorname{var}(e_2)=0.5$, and $\operatorname{cov}(e_1,e_2)=0.25$.\n\nA. Using (5.53) through (5.56) as guides, show that it is not possible to identify the structural VAR.\nB. Using Choleski decomposition such that $b_{12}=0$, find the identified values of $b_{21}$, $\operatorname{var}(\varepsilon_1)$, and $\operatorname{var}(\varepsilon_2)$.\nC. Using Choleski decomposition such that $b_{21}=0$, find the identified values of $b_{12}$, $\operatorname{var}(\varepsilon_1)$, and $\operatorname{var}(\varepsilon_2)$.\nD. Using a Sims-Bernanke decomposition such that $b_{12}=0.5$, find the identified values of $b_{21}$, $\operatorname{var}(\varepsilon_1)$, and $\operatorname{var}(\varepsilon_2)$.
E. Using a Sims-Bernanke decomposition such that bz, = 0.5, find the identi- fied values of bz var(e,), and var(€). F. Suppose that the first three values of e,, are estimated to be 1, 0, and -1 and “ the first three values of ex are estimated to be -1, 0, and 1. Find the first . three values of eu and ex using each of the decompositions in parts B through E.
E. Using a Sims-Bernanke decomposition such that $b_{21}=0.5$, find the identified values of $b_{12}$, $\operatorname{var}(\varepsilon_1)$, and $\operatorname{var}(\varepsilon_2)$.\n\nF. Suppose that the first three values of $e_{1t}$ are estimated to be $1,0,-1$ and the first three values of $e_{2t}$ are estimated to be $-1,0,1$. Find the first three values of $\varepsilon_1$ and $\varepsilon_2$ using each of the decompositions in parts B through E.

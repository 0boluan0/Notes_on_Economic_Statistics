
# 0. 回忆用

1. 

# 1. 引言

本节概览波动建模动机与事实特征，重点在第 2–3 节的 ARCH/GARCH 与拓展模型。~~没什么意义.~~

## 1.1 为什么要进行波动建模

Volatility 就是条件二阶矩.

1. 金融和经济时间序列往往表现出**条件异方差**（conditional heteroskedasticity）现象，即随着时间推移序列的**波动性**(volatility)并非恒定不变，而是呈现出集中的波动期和平稳期交替出现的特征。
2. **波动性聚集现象**（volatility clustering） :波动性聚集是指金融时间序列中高波动时期往往紧随高波动时期，而低波动时期往往紧随低波动时期。换言之，剧烈波动的冲击往往簇集出现。

对于波动建模,有三种方法  ARCH GARCH 和SV
除此之外,还有RV,是对于高频数据的建模.

## 1.2 经济学领域的特征事实

 1.  许多宏观经济序列具有明显的趋势（例如美国实际GDP呈上升趋势）； 
 2. **许多时间序列的波动性并不恒定**，会随着时间发生变化。例如，美国实际GDP增速的波动在1984年左右明显下降，在2007年出现了一个大的负向波动尖峰，随后波动性有所稳定 。这表明方差存在结构性变化; 
 3. 序列受到冲击后的影响可能具有高度**持久性**（persistence），即冲击的影响在序列中持续很长时间； 
 4. 有些金融序列表现出类似随机游走的行为，没有均值回归趋势，例如汇率呈长时间升值或贬值的漫步状（见随机游走模型）； 
 5. 一些序列与其他序列存在共移动现象，例如短期和长期利率常常一起变动，体现**共同趋势**和**协整**关系； 
 6. 某些序列存在**结构性突变**，例如金融危机后油价出现跳变。

# 2. ARCH,GARCH

~~加了条件异方差不影响是白噪声~~（说明：白噪声通常指无条件零均值、方差常数且相互不相关；存在条件异方差不影响“不相关”，但对“方差常数”的理解需区分“条件/无条件”层面，常用[[鞅差序列]]刻画条件零均值）

## 2.1 初步分析

为了刻画波动群聚现象,可以引入一个状态变量:

>[!note] 状态变量 State Variable
>状态变量使得方差可以随状态变化。例如，假设模型：$$y_{t+1} = \epsilon_{t+1} x_t$$其中$y_{t+1}$是我们关心的变量（已经去除均值，使其均值为0，以专注于波动部分），$\epsilon_{t+1}$是均值为0、方差为$\sigma^2$的**白噪声**（white noise），而$x_t$是第$t$期的某个状态变量。这样，条件方差就是：$$\operatorname{Var}(y_{t+1} \mid \mathcal{F}_t) = \sigma^2 x_t^2.$$

通过这种方式，如果$x_t$随时间变化，那么$y$的条件方差也会随之变化。当$x_t$较大时，$\sigma^2 x_t^2$也较大，表示波动性提高；当$x_t$较小时，波动性降低。这为捕捉非恒定方差提供了一个思路。

## 2.2 [[ARCH]]

>[!note] **ARCH(1)模型定义：**
>$$\epsilon_t = \nu_t \sqrt{\alpha_0 + \alpha_1 \epsilon_{t-1}^2}$$或者写作:$$\begin{cases}
\varepsilon_t = \nu_t \sqrt{h_t}  \\
h_t = \alpha_0 + \alpha_1 \varepsilon_{t-1}^2
\end{cases}$$
>其中${\nu_t}$是一列独立同分布（i.i.d.）的随机变量，满足
>$\mathbb{E}(\nu_t)=0, \operatorname{Var}(\nu_t)=1$。
>$\alpha_0$和$\alpha_1$为常数参数，且$\alpha_0 > 0$确保是正的，$0 \leq \alpha_1 < 1$保证平稳性。
>$h_t$表示$\varepsilon_t$在$t$期的条件方差（即$h_t = \mathrm{Var}(\varepsilon_t \mid \mathcal{F}_{t-1})$)
>这里$\epsilon_t$可以看作我们感兴趣序列（例如资产回报率）的均值已滤除后的**随机扰动**（即残差）。模型表示此残差的方差并不恒定，而是由上一期残差的平方$\epsilon_{t-1}^2$决定。

所以==它是一个鞅差分==.因为t-1期的所有项在算期望的时候都能提出来.

<span style="color: yellow;">关键</span>: 在ARCH(1)中，$\epsilon_t$的t-1期条件方差为$\alpha_0 + \alpha_1 \epsilon_{t-1}^2$。(无条件方差就是对条件方差再取一次期望,得到$\bar{h} = \frac{\alpha_0}{1-\alpha_1}$ )

==其中,常数项$\alpha_{0}$不能被删除.因为,如果给$\epsilon_{1}=\alpha_1 \epsilon_{t-1}^2$两侧同时取期望,最后算出来$\alpha_{1}$的值一定为1==

## 2.3 [[GARCH]]

ARCH的N要取得比较大.所以发明了GARCH模型,在保持对条件异方差性建模能力的同时，用更少的参数捕捉长期的波动影响。

>[!note] GARCH(p,q)模型
>$$\begin{cases}
> \varepsilon_t = \nu_t \sqrt{h_t}  \\
> h_t = \alpha_0 + \sum_{i=1}^{q} \alpha_i\varepsilon_{t-i}^2 + \sum_{j=1}^{p} \beta_j h_{t-j}
> \end{cases}$$
> **其中$h_t$依赖于$q$阶误差平方**和$p$阶**自身滞后**。要求参数满足$\alpha_0>0$，$\alpha_i \ge 0$ ( $i=1,\ldots,q$)，$\beta_j \ge 0$ ( $j=1,\ldots,p$ )，且保证平稳性的约束$\sum_{i=1}^q \alpha_i + \sum_{j=1}^p \beta_j < 1$ 。
> 

GARCH模型通常能够用更少的滞后项达到与高阶ARCH模型同样的效果 。因此，GARCH模型在刻画长记忆的波动性时更**参数节省**（parsimonious）。

条件方差:$\mathbb{E}(\varepsilon_t^2 \mid \mathcal{F}{t-1}) = h_t = \alpha_0 + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^p \beta_j h_{t-j}$.
如果过程是平稳的，即满足： $\sum_{i=1}^q \alpha_i + \sum_{j=1}^p \beta_j < 1$那么整个序列的无条件二阶矩（长期平均方差）存在，记作：${ \mathbb{E}(\varepsilon_t^2) = \frac{\alpha_0}{1 - \sum_{i=1}^q \alpha_i - \sum_{j=1}^p \beta_j} }$

实证分析中最常用的就是GARCH(1,1)

## 2.4 侦测ARCH/GARCH效应

参见：[[ARCH Effects Test|ARCH效应检验]]

在对时间序列进行建模时，我们首先常用ARMA模型拟合均值部分，然后需要判断残差序列中是否存在ARCH/GARCH效应（即条件异方差）。

使用两种方法检验残差,两种方法都是在对原始序列先拟合一个最好的ARMA模型,并得到一个残差序列${\hat{\varepsilon}_t}$.而后对残差序列进行操作

>[!note] McLeod-Li检验
> 拟合一个“最优”ARMA模型,得到残差序列 $\hat{\varepsilon}_t$
>   
>  对残差序列平方$\hat{\varepsilon}_t^2$，计算其样本自相关.定义第 i 阶自相关：
> $$r_i = \frac{\sum_{t=i+1}^{T} (\hat{\varepsilon}_t^2 - \bar{\sigma}^2)(\hat{\varepsilon}_{t-i}^2 - \bar{\sigma}^2)}{\sum_{t=1}^{T} (\hat{\varepsilon}_t^2 - \bar{\sigma}^2)^2}$$
> 其中 $\bar{\sigma}^2 = \frac{1}{T} \sum \hat{\varepsilon}_t^2$ 是残差平方均值。
> 
> 而后使用样本自相关构建检验统计量
> 
> $$Q = T(T+2) \sum_{i=1}^m \frac{r_i^2}{T - i} \quad \text{服从 } \chi^2_m \text{ 分布}$$
> 
> - 如果显著 ⇒ 拒绝 $H_0$,即意味着残差序列中存在 ARCH 效应
> - 若不显著 ⇒ 没有显著条件异方差

>[!note] ARCH-LM检验
>同样先拟合一个 ARMA 模型 ⇒ 得到残差 $\hat{\varepsilon}_t$
> 使用残差做回归：
> $$\hat{\varepsilon}_t^2 = \alpha_0 + \sum_{j=1}^{q} \alpha_j \hat{\varepsilon}_{t-j}^2 + \eta_t$$
> 这个是检验是否存在 ARCH(q) 的标准形式。
> 检验思想：
> - 原假设 $H_0$: $\alpha_1 = \alpha_2 = \cdots = \alpha_q = 0$（无 ARCH）
> - 计算 R^2：这个回归的决定系数
> - 构造统计量：$L = T R^2 \sim \chi^2_q$
> 	- 如果 L 显著 ⇒ 存在 ARCH 效应
> 	- 若不显著 ⇒ 没有 ARCH 效应

## 2.5 极大似然估计MLE

参见：[[极大似然估计]]

根据正态密度函数写出条件似然：

$L = \prod_{t=1}^T \left( \frac{1}{\sqrt{2\pi h_t}} \exp\left( -\frac{\varepsilon_t^2}{2h_t} \right) \right)$
对数似然为：
$\log L = -\frac{T}{2} \log(2\pi) - \frac{1}{2} \sum_{t=1}^T \log h_t - \frac{1}{2} \sum_{t=1}^T \frac{\varepsilon_t^2}{h_t}$

总之原理是这么个原理.不能手动算的.别管了

## 2.6 评估拟合

### (1)模型拟合优度的评估:AIC和SBC.

• AIC 定义为：$\displaystyle \text{AIC} = -2\ln L_{\text{max}} + 2k$，其中$L_{\text{max}}$是模型最大化后的似然值，$k$是模型中估计参数的个数 。$-2\ln L$衡量了模型残差的不可解释程度（越小表示模型拟合越好），而$2k$是对模型复杂度的惩罚（参数越多惩罚越大）。

• BIC 定义为：$\displaystyle \text{BIC} = -2\ln L_{\text{max}} + k \ln T$，其中$T$为样本容量 。相比AIC，BIC对参数个数的惩罚更严厉（乘以$\ln T$因子），在大样本下倾向于选择更简洁的模型。

### (2)模型诊断

 使用经过ARMA-GARCH模型预测后的残差计算标准化残差 $s_t = \frac{\hat{\varepsilon}_t}{\sqrt{\hat{h}_t}}$，其中$\hat{\varepsilon}_t$是模型估计后的残差，$\hat{h}_t$是对应的拟合条件方差。理论上，如果均值模型和波动模型都正确，那么$s_t$应当是一个i.i.d.标准正态序列（在假定正态创新的情形下）。

对标准化方差进行白噪声检验[[03_平稳时间序列模型#4.5 白噪声检验]]

## 2.7 预测方差

可以进行均值预测和方差预测,均值部分和前面ARMA一样.

有了GARCH模型，我们可以预测下一期的波动水平，即计算$h_{t+1|t} = E_t[h_{t+1}]$（下标$t+1|t$表示在$t$期基于信息$\mathcal{F}_t$对$t+1$期的预测）。以GARCH(1,1)为例，根据模型：
$$h_{t+1} = \alpha_0 + \alpha_1 \varepsilon_t^2 + \beta_1 h_t$$
在$t$时刻已知$\varepsilon_t$和$h_t$，则**一步前方差预测**为：
$$\hat{h}_{t+1|t} = \alpha_0 + \alpha_1 \varepsilon_t^2 + \beta_1 h_t$$

这实际上就是把当期已发生的冲击$\varepsilon_t^2$代入，对下一期进行更新。如果我们关心下一期预测的95%置信区间，可以写为：

$$\hat{y}_{t+1|t} \pm 1.96 \sqrt{\hat{h}_{t+1|t}} $$

与固定方差情形不同的是，这个区间的宽度$\sqrt{\hat{h}_{t+1|t}}$是动态变化的：在高波动时期，$\hat{h}_{t+1|t}$较大，置信区间更宽；在低波动时期，置信区间更窄 。因此，GARCH模型能够提供随市场状况变化的风险度量。

如果是更多步的预测,就是递推.

# 3. 扩展模型

==极大概率不考,如果没时间了就别学了.==

## 3.1 IGARCH

参见：[[IGARCH]]

金融时间序列的一个典型特征是波动性的**高度持久**（persistent）。实证中，对许多资产回报率拟合GARCH(1,1)模型时，常常发现估计得到的$\hat{\alpha}_1 + \hat{\beta}_1$非常接近1。

>[!note] **积整GARCH模型**（Integrated GARCH，简称IGARCH）。
>$\alpha_1 + \beta_1 = 1$的GARCH模型.
>IGARCH(1,1)实际上就是$\alpha_1 + \beta_1 = 1$的GARCH(1,1)模型。这个等式让原先的GARCH模型少了一个参数
>
>其特性为:
>1. 无条件方差不存在
>2. 预测方差不断积累
>
>总之就是会永远记住之前发生的事情

## 3.2 ARCH-M

参见：[[ARCH-M]]

风险越大,要求的收益越高

>[!note] ARCH-M 
>$$\begin{cases} y_t &= \mu_t + \varepsilon_t \\ \mu_t &= \beta + \delta h_t \quad (\delta > 0) \\ h_t &= \alpha_0 + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 \end{cases}$$ 

| 方程                                                   | **含义**                     |
| ---------------------------------------------------- | -------------------------- |
| $y_t = \mu_t + \varepsilon_t$                        | 观测值等于期望值 + 噪声              |
| $\mu_t = \beta + \delta h_t$                         | **均值受到波动** $h_t$ **的正向影响** |
| $h_t = \alpha_0 + \sum \alpha_i \varepsilon_{t-i}^2$ | 标准 ARCH(q) 波动结构            |

## 3.3 带有解释变量的波动模型

加了一些会影响波动性的哑变量.通过引入新的信息来更好拟合.

>[!example] 示例:衡量911事件前后的波动
> 我们想检验“9·11事件”（2001年9月11日的恐怖袭击）是否显著提高了美国股市的波动性。为此，可在GARCH方差方程中加入一个对应该事件的哑变量$D_t$。模型可以设定为：
> 
> $$h_t = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \beta_1 h_{t-1} + \gamma D_t,$$
> 
> 其中$D_t$是在2001年9月11日之前$D_t=0$，在2001年9月11日及之后$D_t=1$的哑变量。这里我们使用了一个GARCH(1,1)结构并叠加事件哑变量。

## 3.4 非对称模型:[[TARCH]],[[EGARCH]]

>[!note] 杠杆效应leverage effect
>这表现为**波动性的非对称性**.例如公司的负面信息对公司的影响大于公司的正面信息.

>[!note] TARCH ,门限GARCH效应 Threshold GARCH
> TARCH通过在方差方程中引入一个针对负残差的指示变量来实现非对称效应。以TARCH(1,1)为例，其形式可写为：
> 
> $$h_t = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \lambda_1d_{t-1}\epsilon_{t-1}^2 + \beta_1 h_{t-1},$$
> 
> ==其中$d_{t-1}$是一个哑变量==，当$\epsilon_{t-1} < 0$时$d_{t-1}=1$，当$\epsilon_{t-1} \ge 0$时$d_{t-1}=0$。也就是说，$\epsilon_{t-1}^2$项会根据$\epsilon_{t-1}$的符号被赋予不同的系数：如果前一期是负冲击，则方差方程中实际影响是$(\alpha_1+\lambda_1)\epsilon_{t-1}^2$；如果前一期是正冲击，则影响是$\alpha_1 \epsilon_{t-1}^2$（因为这时$d_{t-1}=0$，额外项不起作用）。

>[!note] EGARCH 指数GARCH模型 Exponential GARCH
>指数GARCH模型采用对数形式的方差方程，形式例如EGARCH(1,1)：
> 
> $$\ln h_t = \alpha_0 + \alpha_1 \left(\frac{\epsilon_{t-1}}{\sqrt{h_{t-1}}}\right) + \lambda_1 \left|\frac{\epsilon_{t-1}}{\sqrt{h_{t-1}}}\right| + \beta_1 \ln h_{t-1}.$$
> 
> 这里，引入$\frac{\epsilon_{t-1}}{\sqrt{h_{t-1}}}$作为标准化的残差（通常称为_z-score_，表示上一期残差相对于其标准差的大小和方向），这样做有两个好处：其一，使用$\ln h_t$确保了预测的$h_t$永远为正（因为指数的输出总是正的），不需要像标准GARCH那样对参数非负作约束；其二，通过$\alpha_1$乘以标准化残差和$\lambda_1$乘以残差的绝对值，相当于把残差的符号和幅度分离来影响$\ln h_t$，从而实现非对称效果。
> 

# 作业

## 3.
3. Bollerslev (1986) proves that the ACF of the squared residuals resulting from the GARCH(p, q) process represented by (3.9) acts as an ARMA(m, p) process, where m = max(p, qg). You arc to illustrate this result using the exam- ples below. : A. Consider the GARCH(I, 2) process h, = Oy + oue2 + ooe 2 十 有 NT Add the expression (€?— h,) to each side, so that 名 = Oy + OER + pepe + Bilt, + (EF hy) = Oy + (0, + Bide? y + Ozer 2 — Biles — Ma) + (eF— hy) Define n, = (€?— h,), so that bey. ; = Op + (ol+B)ei+ooeza 一 Bt+tnm Show that: ， ae : i. 1, is serially uncorrelated. : , 四 . : ii, The {e3 sequence acts as an ARMA(2, 1) process. : . ， ; 了 B、Consider the GARCH(2, 1) process h, = Oy + Qie_ + By, + Byf,_2, Show _ that it is possible to add 7, to each side so as to obtain . ; . . = Oy + Oy€F + Bit + T+ Bolt,.2 : : Show that adding. and subtracting the terms B,n,_, and B,7,. to the right- :* + hand side of this equation yield an ARCH(2, 2) process. 网 | C, Provide an intuitive explanation of the statement: “The Lagrange multiplicr B test for ARCH errors cannot be used to test the null of white-noise squared 3 residuals against an alternative of a specific GARCH(p, q) proccss.” - : -D. Sketch the proof of the general statement that the ACF of the squared resid- a uals resulting from the GARCH(p, q) process represented by (3.9) acts as a an ARMA(im, p) process, where m= max(p,q). . an
~~[[Pasted image 20250523201006.png]]~~ (已转文字)

## 7.
9. Consider the ARCH(2) process Ee?= dg + 04€2., + ooe2 A. Suppose that y, = ao + ayy... + € Find the conditional and unconditional 3 variance of {y,} in terms of of the parameters a), oo, 0, and O. oe B. Suppose that {y,} is an ARCH-M process such that the level of y, is posi- 4 tively related to its own conditional variance. For simplicity, let y, =O) + ° % oue2 ,+ ooge2 + €, Trace out the impulse response function of {y,} toan 污 - {e,} shock. You may assume that the system has been in long-run equilib- ... plum (€,_2 =, = 0) but now €, = 1. ‘Thus, the issue is to find the values of 六 9 yy You 3, and yy given that €, =€,=- =0. E: C. Use your answer to part B to explain the following result. A student esti- oS mated {y,} as an MA(2) process and found the residuals to be white-noise. ss A sccond student estimated the same series as the ARCH-M process y,= Up -总 + Q,€_, + one +e Why might both estimates appear reasonable? How * would you decide which is the better model? ‘ D. In general, explain why an ARCH-M model might appear to be a moving a average process. 3
~~[[Pasted image 20250523201044.png]]~~ (已转文字)

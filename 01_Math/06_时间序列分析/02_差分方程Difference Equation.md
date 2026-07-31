
==主要讲低阶差分方程,考试最多考到二阶==
<!-- bilingual-en:start -->
==This course focuses on low-order difference equations; the exam covers at most second-order equations.==
<!-- bilingual-en:end -->

# 0. 回忆用问题
<!-- bilingual-en:start -->
*0. Recall questions*
<!-- bilingual-en:end -->

1. 线性差分方程是什么
2. 如何求解一个线性差分方程
	1. 迭代求解
	2. 特征方程求解
3. 稳定的特解,存在的条件
4. 平稳过程,单位根过程,爆炸过程的判别(使用算子多项式和特征方程都要会)
<!-- bilingual-en:start -->
1. What is a linear difference equation?
2. How do we solve one?
	1. By iteration
	2. By the characteristic equation
3. What is a stable particular solution, and when does it exist?
4. How do we distinguish stationary, unit-root, and explosive processes using both operator polynomials and characteristic equations?
<!-- bilingual-en:end -->

# 1. 介绍
<!-- bilingual-en:start -->
*1. Introduction*
<!-- bilingual-en:end -->

## 1.1. 术语：算子（operator）
<!-- bilingual-en:start -->
*1.1. Terminology: operator*
<!-- bilingual-en:end -->

从一个拓扑空间到另一个拓扑空间的映射.
<!-- bilingual-en:start -->
A mapping from one topological space to another.
<!-- bilingual-en:end -->

## 1.2. 术语：差分算子（[[Difference Equation|Difference]] operator）
<!-- bilingual-en:start -->
*1.2. Terminology: difference operator ([[Difference Equation|difference]] operator)*
<!-- bilingual-en:end -->

>[!note] 差分算子 $\Delta$的定义
> 定义一阶差分为 $\Delta y_t \equiv y_t - y_{t-1}$，表示变量在相邻两个时期的变化量。例如，如果 $y_t$ 是某期的值，那么 $\Delta y_t$ 就是 $y_t$ 与前一期 $y_{t-1}$ 之差。
>
> 同理，**二阶差分**定义为 $\Delta^2 y_t \equiv \Delta(\Delta y_t) = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) = y_t - 2y_{t-1} + y_{t-2}$ 。差分算子在时间序列分析中类似于微积分中的导数运算，用于考察序列的变化。
> <!-- bilingual-en:start -->
> The first difference is $\Delta y_t\equiv y_t-y_{t-1}$, the change between two adjacent periods.
>
> The **second difference** is $\Delta^2y_t=y_t-2y_{t-1}+y_{t-2}$. Differencing is the discrete analogue of differentiation.
> <!-- bilingual-en:end -->

**直观理解**：$\Delta f(x)$ 计算的是 $f(x)$ 在 $x$ 处的增量，相当于离散版本的**一阶导数**。
<!-- bilingual-en:start -->
**Intuition:** $\Delta f(x)$ measures an increment in $f$ and is the discrete analogue of a **first derivative**.
<!-- bilingual-en:end -->

## 1.3. 术语：滞后算子（Lag operator）
<!-- bilingual-en:start -->
*1.3. Terminology: lag operator*
<!-- bilingual-en:end -->

>[!note] 滞后算子 $L$的定义
> $L^i y_t = y_{t-i}$，即将序列“往后推”$i$期（滞后 $i$ 期） 。例如，$L y_t = y_{t-1}$，$L^2 y_t = y_{t-2}$ 等。
>
> 滞后算子有一些类似指数的运算性质，如 $L^i \cdot L^j = L^{i+j}$，$L^0$ 为恒等算子等~~/~~。利用滞后算子可以简洁地表示差分，例如 $\Delta y_t = y_t - y_{t-1} = y_t - L y_t = (1 - L) y_t$。
> <!-- bilingual-en:start -->
> $L^iy_t=y_{t-i}$ shifts a series back by $i$ periods. Thus $Ly_t=y_{t-1}$ and $L^2y_t=y_{t-2}$.
>
> It obeys $L^iL^j=L^{i+j}$, while $L^0$ is the identity. Hence $\Delta y_t=(1-L)y_t$.
> <!-- bilingual-en:end -->

## 1.4. 为什么要研究差分方程
<!-- bilingual-en:start -->
*1.4. Why study difference equations?*
<!-- bilingual-en:end -->

考虑经典的分解方法：
$$y_t = T_t + S_t + I_t$$
并设：
$$T_t = 1 + 0.1 t, \quad S_t = 1.6 \sin\left(\frac{t\pi}{6}\right), \quad I_t = 0.7 I_{t-1} + \varepsilon_t$$
这个 $y_t$ 可以拆分为三个部分：
1. **趋势成分（Trend component）$T_t$**
   这里的趋势项 $T_t = 1 + 0.1 t$ 表示 $y_t$ 随着时间 $t$ 线性增长，每个时间点 $t$ 增加 $0.1$。
2. **季节成分（Seasonal component）$S_t$**
   季节项 $S_t = 1.6 \sin(t\pi/6)$ 采用正弦函数，表示 $y_t$ 随着时间 $t$ 具有周期性变化。周期为 $12$（因为 $\sin$ 函数的周期是 $2\pi$，即 $t\pi/6 = 2\pi$ 对应 $t=12$），说明该数据可能有**年度季节性**（比如按月变化的经济指标）。
3. **不规则成分（Irregular component）$I_t$**
   不规则项 $I_t$ 由一阶自回归模型（[[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|AR]](1) 过程）给出：$$I_t = 0.7 I_{t-1} + \varepsilon_t$$
   其中 $\varepsilon_t$ 是随机扰动，表示无法预测的随机变化。这个方程表示 $I_t$ 依赖于其前一期值 $I_{t-1}$，并受到随机冲击的影响。这是一种**平稳时间序列模型**，如果 $|\phi| < 1$（这里 $\phi = 0.7$），则该过程是均值回归的，不会发散。
<!-- bilingual-en:start -->
Consider
$$y_t=T_t+S_t+I_t,$$
where
$$T_t=1+0.1t,\quad S_t=1.6\sin\left(\frac{t\pi}{6}\right),\quad I_t=0.7I_{t-1}+\varepsilon_t.$$

1. **Trend $T_t$:** a linear increase of $0.1$ per period.
2. **Seasonality $S_t$:** a sinusoid with period 12, which gives annual seasonality to monthly data.
3. **Irregular component $I_t$:** an [[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|AR(1) process]] driven by unpredictable innovations. Since $|0.7|<1$, it is stationary and mean-reverting.
<!-- bilingual-en:end -->

==差分方程最一般的形~~势~~式就是某个变量当前的取值可由自身的滞后项、时间以及其他变量的函数共同决定。==
<!-- bilingual-en:start -->
==A difference equation makes a current value a function of its own lags, time, and other variables.==
<!-- bilingual-en:end -->

# 2. 线性差分方程和其解Linear [[Difference Equation|Difference Equation]] and Its Solution
<!-- bilingual-en:start -->
*2. Linear difference equations and their solutions*
<!-- bilingual-en:end -->

## 2.1. 线性差分方程
<!-- bilingual-en:start -->
*2.1. Linear difference equations*
<!-- bilingual-en:end -->

差分方程过于~~general~~一般，因此仅讨论线性差分方程。
<!-- bilingual-en:start -->
Difference equations are broad, so this course considers only linear ones.
<!-- bilingual-en:end -->

<span style="color: yellow;">线性差分方程的“线性”指的是在方程中，变量及其滞后项只以一次幂出现，并且不会相互乘积或出现在非线性函数中。</span>
<!-- bilingual-en:start -->
<span style="color: yellow;">“Linear” means that variables and their lags appear only to the first power, are not multiplied together, and do not enter nonlinear functions.</span>
<!-- bilingual-en:end -->

>[!note] 线性差分方程
>具有常系数的$n$ 阶线性差分方程的一般形式如下所示：
>$$y_t = a_0 + \sum_{i=1}^n a_i \, y_{t-i} + x_t$$
>   表示当前时刻 $t$ 的变量 $y_t$ 可以由其过去 $n$ 期的值 $y_{t-1}, y_{t-2}, \dots, y_{t-n}$ 以及一个额外的外生项 $x_t$ 决定。系数 $a_0, a_1, \dots, a_n$ 都是常数。
>
>    **外生过程（Forcing Process）$x_t$**
>    - $x_t$ 可以是时间 $t$ 的确定性函数，比如趋势或季节性项。
>    - 也可以是其他经济变量（或随机过程）的滞后值或当前值。
>    - 或者是随机扰动，如白噪声的加权和，$x_t = \sum_{i=0}^\infty \beta_i \, \varepsilon_{t-i}$
>
>   **差分算子表示**     $$\Delta y_t = a_0 + \gamma \, y_{t-1} + \sum_{i=2}^n a_i \, y_{t-i} + x_t$$ 其中 $\gamma = a_1 - 1$。这是把一阶差分后的 $y_t$ 与其滞后值联系起来的一种写法，有时在研究单位根、平稳性等问题时更方便。
> <!-- bilingual-en:start -->
> An $n$th-order equation with constant coefficients has the form
> $$y_t=a_0+\sum_{i=1}^na_i y_{t-i}+x_t.$$
> The current value depends on its previous $n$ values and a forcing term $x_t$; all $a_i$ are constants.
>
> **Forcing process $x_t$**
> - a deterministic trend or seasonal term;
> - a current or lagged economic variable or stochastic process; or
> - a random disturbance such as $\sum_{i=0}^{\infty}\beta_i\varepsilon_{t-i}$.
>
> With $\gamma=a_1-1$, the difference form is
> $$\Delta y_t=a_0+\gamma y_{t-1}+\sum_{i=2}^na_i y_{t-i}+x_t,$$
> which is useful for studying unit roots and stationarity.
> <!-- bilingual-en:end -->

## 2.2. 差分方程的解
<!-- bilingual-en:start -->
*2.2. Solutions of difference equations*
<!-- bilingual-en:end -->

>[!note] 差分方程的解
>一个差分方程的解，给出了 $y_t$ 如何作为 $\{x_t\}$ 的元素与时间 $t$（以及可能给定的 $y_t$ 序列初始条件）来确定其数值的函数形式.
> <!-- bilingual-en:start -->
> A solution specifies $y_t$ as a function of time, the forcing sequence $\{x_t\}$, and any initial conditions.
> <!-- bilingual-en:end -->

>[!example] **示例 1**：
>  $$\Delta y_t = 2$$
>  其解为
>  $$y_t = 2t + c$$
>  其中 $c$ 是任意常数。我们可以通过代入来验证。
> <!-- bilingual-en:start -->
> $\Delta y_t=2$ has solution $y_t=2t+c$, where $c$ is arbitrary. Substitution verifies it.
> <!-- bilingual-en:end -->

>[!example] **示例 2**：
>  $$I_t = 0.7 I_{t-1} + \varepsilon_t$$  （这是不规则成分 $I_t$ 的模型），其解为
>  $$I_t = \sum_{i=0}^\infty 0.7^i \,\varepsilon_{t-i}$$
>  同样可以通过代入来验证。
> <!-- bilingual-en:start -->
> $I_t=0.7I_{t-1}+\varepsilon_t$ has solution
> $$I_t=\sum_{i=0}^{\infty}0.7^i\varepsilon_{t-i},$$
> again verified by substitution.
> <!-- bilingual-en:end -->

## 2.3. 求解差分方程
<!-- bilingual-en:start -->
*2.3. Solving difference equations*
<!-- bilingual-en:end -->

### 2.3.1. 迭代求解
<!-- bilingual-en:start -->
*2.3.1. Solution by iteration*
<!-- bilingual-en:end -->

#### (1) **正向迭代（Forward Iteration）**
<!-- bilingual-en:start -->
*(1) Forward iteration*
<!-- bilingual-en:end -->

从已知初值开始，逐步推导后续$y_t$：
• 当$t=1$时：
$$y_1 = a_0 + a_1 y_0 + \varepsilon_1.$$
• 当$t=2$时：
$$y_2 = a_0 + a_1 y_1 + \varepsilon_2 = a_0 + a_1(a_0 + a_1 y_0 + \varepsilon_1) + \varepsilon_2.$$
展开并整理，可以得到：
$$y_2 = a_0(1 + a_1) + a_1^2 y_0 + a_1 \varepsilon_1 + \varepsilon_2.$$
• 可以看出，随着$t$增加，表达式会变得愈加复杂，但我们可以归纳出一般规律。当$t > 0$时，不断迭代会得到一个通式。**经过重复迭代推导，可得一般解：**
<!-- bilingual-en:start -->
Starting from $y_0$:

- $y_1=a_0+a_1y_0+\varepsilon_1$.
- Substitution at $t=2$ gives
  $$y_2=a_0(1+a_1)+a_1^2y_0+a_1\varepsilon_1+\varepsilon_2.$$

Repeated substitution reveals the general solution:
<!-- bilingual-en:end -->

$$
y_t = a_0 \sum_{i=0}^{t-1} a_1^i \;+\; a_1^t y_0 \;+\; \sum_{i=0}^{t-1} a_1^i \, \varepsilon_{t-i}
$$

#### (2) 反向迭代(backward iteration)
<!-- bilingual-en:start -->
*(2) Backward iteration*
<!-- bilingual-en:end -->

同理,（向前推回初值）：**假设初始条件$y_0$未知，我们依然可以将方程向过去迭代：
<!-- bilingual-en:start -->
Even if $y_0$ is unknown, the recurrence can be substituted backward into the past.
<!-- bilingual-en:end -->

• 对于$t=0$时刻，我们有
$$y_0 = a_0 + a_1 y_{-1} + \varepsilon_0,$$
此时,将$y_{1}$时刻的$y_0$用这一长串公式替代,
$$y_{1}=a_{0}+(a_0 + a_1 y_{-1} + \varepsilon_0)+\varepsilon_{1}$$
而后这么一步一步带入并展开得到:$$y_t = a_0 \sum_{i=0}^{t} a_1^i \;+\; \sum_{i=0}^{t} a_1^i \, \varepsilon_{t-i} \;+\; a_1^{\,t+1} y_{-1}$$
• 如果我们对未知初始值向前迭代$m$期（即假定我们追溯到$y_{-m-1}$仍未知），则可以得到：
$$y_t = a_0 \sum_{i=0}^{\,t+m} a_1^i \;+\; \sum_{i=0}^{\,t+m} a_1^i \, \varepsilon_{t-i} \;+\; a_1^{\,t+m+1} \, y_{-m-1}$$
<!-- bilingual-en:start -->
At $t=0$,
$$y_0=a_0+a_1y_{-1}+\varepsilon_0.$$
Substituting backward gives
$$y_t=a_0\sum_{i=0}^{t}a_1^i+\sum_{i=0}^{t}a_1^i\varepsilon_{t-i}+a_1^{t+1}y_{-1}.$$
Continuing $m$ further periods gives
$$y_t=a_0\sum_{i=0}^{t+m}a_1^i+\sum_{i=0}^{t+m}a_1^i\varepsilon_{t-i}+a_1^{t+m+1}y_{-m-1}.
<!-- bilingual-en:end -->

>[!example] 反向迭代的例子
> - 对于形如
>  $$I_t = 0.7 I_{t-1} + \varepsilon_t,$$
> 	可以反复向前展开：
> 	$$ I_t = 0.7 I_{t-1} + \varepsilon_t = 0.7 \bigl(0.7 I_{t-2} + \varepsilon_{t-1}\bigr) + \varepsilon_t = 0.7^2 I_{t-2} + 0.7 \varepsilon_{t-1} + \varepsilon_t$$
>      继续迭代，就能得到
>      $$
>      I_t = 0.7^k I_{t-k} + \sum_{i=0}^{k-1} 0.7^i \,\varepsilon_{t-i}.
>      $$
> 	此时由于 $|0.7|<1$ 且 $k \to \infty$，$0.7^k I_{t-k}$ 收敛为 0，最终得到
>      $$ I_t = \sum_{i=0}^{\infty} 0.7^i \,\varepsilon_{t-i}$$
> <!-- bilingual-en:start -->
> For $I_t=0.7I_{t-1}+\varepsilon_t$, $k$ substitutions give
> $$I_t=0.7^kI_{t-k}+\sum_{i=0}^{k-1}0.7^i\varepsilon_{t-i}.$$
> Since $|0.7|<1$, the initial-condition term vanishes as $k\to\infty$, leaving
> $$I_t=\sum_{i=0}^{\infty}0.7^i\varepsilon_{t-i}.
> <!-- bilingual-en:end -->

>[!attention] 注意:稳定的特解
>和正向迭代不同,当 $m$ 趋于无穷大（即将初始时刻推至远过去）时，结果取决于 $a_1$ 的大小：如果 $|a_1|<1$，则 $a_1^{t+m} y_{t-m}$ 项随着 $m \to \infty$ 衰减为0 。此时我们得到**稳定的特解**（particular solution）：$$y_t = \frac{a_0}{1 - a_1} + \sum_{i=0}^{\infty} a_1^i\varepsilon_{t-i}$$
> <!-- bilingual-en:start -->
> If $|a_1|<1$, pushing the initial date into the remote past makes the initial-condition term vanish, leaving
> $$y_t=\frac{a_0}{1-a_1}+\sum_{i=0}^{\infty}a_1^i\varepsilon_{t-i}.
> <!-- bilingual-en:end -->

其中第一项 $\frac{a_0}{1-a_1}$ 是系统在没有随机波动时的稳态水平（因为设置 $\varepsilon_t = 0$ 时，该值使 $y_t = y_{t-1}$），第二项是所有过去随机冲击经过衰减后的累计影响 。由于此解不包含任意常数，它是特定于假设“远过去无影响”的特解。
<!-- bilingual-en:start -->
The first term is the deterministic steady state; the second accumulates past shocks with decaying weights. The solution assumes that the remote past has no remaining effect.
<!-- bilingual-en:end -->

那在知道特解的情况下,可以给特解加上一个常数项$Aa_1^t$构成通解:在$|a_1|<1$条件下**任意**一个解都可以表示为：$$y_t = Aa_1^t + \frac{a_0}{1 - a_1} + \sum_{i=0}^{\infty} a_1^i  \varepsilon_{t-i}$$
如果初始值$y_0$没有给定,这个A是什么都行.如果初始值$y_0$给定了,就可以反推计算出A.此时方程只有一个解.
如果初始值$y_0$给定了,那么这个差分方程就只有一个解.如果初始值没有给定,那这个方程就有一个特解和无数个通解
<!-- bilingual-en:start -->
Adding the homogeneous term gives
$$y_t=Aa_1^t+\frac{a_0}{1-a_1}+\sum_{i=0}^{\infty}a_1^i\varepsilon_{t-i}.$$
Without $y_0$, $A$ is arbitrary. A specified $y_0$ uniquely determines $A$ and the full solution.
<!-- bilingual-en:end -->


==$|a_1|<1$叫做平稳过程.$|a_1|=1$叫做单位根过程,$|a_1|>1$时叫做爆炸过程.==
<!-- bilingual-en:start -->
==$|a_1|<1$ gives a stationary process, $|a_1|=1$ a unit-root process, and $|a_1|>1$ an explosive process.==
<!-- bilingual-en:end -->

### 2.3.2. 特征方程法（Characteristic Equation)
<!-- bilingual-en:start -->
*2.3.2. Characteristic-equation method*
<!-- bilingual-en:end -->

迭代法在一阶和二阶的时候比较好使,高阶不好用.所以引出另一种方法,特征工程法.
<!-- bilingual-en:start -->
Iteration becomes unwieldy at higher orders, which motivates the characteristic-equation method.
<!-- bilingual-en:end -->

>[!note] 方程的齐次部分
> 方程的右手边只保留和$y_t$的滞后项有关的部分,常数和扰动都扣掉.
> <!-- bilingual-en:start -->
> Retain only terms involving $y_t$ and its lags; remove constants and forcing terms.
> <!-- bilingual-en:end -->

步骤:
<!-- bilingual-en:start -->
Procedure:
<!-- bilingual-en:end -->

1. **写出齐次方程并求其全部齐次解。**
将原差分方程中的常数项和扰动项扣掉,得到对应的**齐次方程**。求解该齐次方程，找出$n$个线性独立的齐次解。
2. **求一个特解（特定解）。**
为原始非齐次方程（即带有 $a_0$ 和 $x_t$）找到任意一个解$y_t^p$。这个解不需要包含所有任意常数，因为任意常数组合会在下一步加入。
3. **写出方程的一般解。**
一般解 = 特解 $+$ 齐次解的线性组合。也就是说，将求得的一个特解与所有齐次解的线性组合相加，构成原方程的一般形式。由于齐次解包含任意常数，这一组合会带有$n$个待定常数。
4. **利用初始条件确定常数。**
根据给定的初始条件（一般需要给定 $n$ 个初始值，例如 $y_0, y_1, \dots, y_{n-1}$），构造方程并解出上述线性组合中的$n$个常数，从而得到满足初值条件的唯一特定解。
<!-- bilingual-en:start -->
1. **Find all homogeneous solutions.** Remove constants and forcing terms, then find $n$ linearly independent solutions.
2. **Find one particular solution** $y_t^p$ to the nonhomogeneous equation.
3. **Form the general solution:** particular solution plus a linear combination of homogeneous solutions.
4. **Use $n$ initial conditions** to determine the $n$ constants.
<!-- bilingual-en:end -->

>[!example] **例子：二阶线性差分方程的求解**
> 考虑如下二阶差分方程：
>
> $$y_t = 0.9y_{t-1} - 0.2y_{t-2} + 3$$
>
> 这是形如 $y_t = a_0 + a_1 y_{t-1} + a_2 y_{t-2}$ 的情况，其中 $a_0=3, a_1=0.9, a_2=-0.2$（注意这里负号属于系数的一部分）。假设初始条件 $y_0=13;y_1=11.3$。我们按四步求解：
>
> **第1步（齐次方程）：** 将常数和外生项设为0，得到齐次方程
> $$y_t - 0.9y_{t-1} + 0.2y_{t-2} = 0$$
> 求该齐次方程的通解。我们**猜测**齐次解形式为 $y_t^h = r^t$（这是线性齐次方程常用的假设形式），代入齐次方程得到**特征方程** ：$$r^2 - 0.9r + 0.2 = 0$$
> 求解此二次特征方程：$r^2 - 0.9r + 0.2 = 0$。使用求根公式，得到两个实特征根：$r_1 = \frac{0.9+0.1}{2} = 0.5$，$r_2 = \frac{0.9-0.1}{2} = 0.4$，互不相等。对应的两个线性独立齐次解为 $y_{t}^{h(1)} = (0.5)^t$ 和 $y_{t}^{h(2)} = (0.4)^t$ 。因此齐次方程的一般解为这两个解的线性组合：
> $$y_t^h = A_1 \cdot (0.5)^t + A_2 \cdot (0.4)^t$$
>
> 其中 $A_1, A_2$ 为任意常数。
>
> **第2步（特解）：** 寻找原非齐次方程 $y_t = 0.9y_{t-1} - 0.2y_{t-2} + 3$ 的一个特解。观察驱动项为常数3，我们可以尝试令解为常数形式 $y_t^p = C$（即假设解不随时间变化）。将其代入方程左侧：$y_t^p = Cy_{t-1}^p = Cy_{t-2}^p = C$，代入原方程得到
> $$C = 0.9C - 0.2C + 3.$$
> 即 $C = 0.7C + 3$。解得 $0.3C = 3$，于是 $C = 10$。因此 $y_t^p = 10$ 是原方程的一个特解 。_(验证：$0.9\cdot 10 - 0.2\cdot 10 + 3 = 9 - 2 + 3 = 10$，左等于右，成立。)_
>
> **第3步（一般解）：** 将特解和齐次解相加，得到原方程的一般解形式：
>
> $$y_t = A_1 \cdot (0.5)^t + A_2 \cdot (0.4)^t + 10.$$
>
> 此时 $A_1, A_2$ 仍是待定常数。
>
> **第4步（套用初始条件）：** 利用给定的初始值求 $A_1, A_2$。已知 $y_0 = 13,; y_1 = 11.3$。代入一般解：
> • 当 $t=0$ 时：$y_0 = A_1(0.5^0) + A_2(0.4^0) + 10 = A_1 + A_2 + 10 = 13$，得到方程 (i)：$A_1 + A_2 = 3$。
> • 当 $t=1$ 时：$y_1 = A_1(0.5^1) + A_2(0.4^1) + 10 = 0.5A_1 + 0.4A_2 + 10 = 11.3$，得到方程 (ii)：$0.5A_1 + 0.4A_2 = 1.3$（将10移项：$11.3-10=1.3$）。
>
> 求解联立方程 (i) 和 (ii),确定常数，得到满足初值的**特定解**：
> $$\boxed{y_t = 1 \cdot (0.5)^t + 2 \cdot (0.4)^t + 10}$$
> 可以检查 $t=0$ 和 $t=1$ 时上述解确实给出 $y_0=13, y_1=11.3$。并且当 $t \to \infty$ 时，由于 $0<0.5,0.4<1$，齐次项$(0.5)^t,(0.4)^t$将衰减为0，$y_t$ 趋于稳定在10，这正是特解对应的稳态水平。
> <!-- bilingual-en:start -->
> For
> $$y_t=0.9y_{t-1}-0.2y_{t-2}+3,\qquad y_0=13,\ y_1=11.3,$$
> the homogeneous characteristic equation is
> $$r^2-0.9r+0.2=0,$$
> with roots $0.5$ and $0.4$. Thus
> $$y_t^h=A_1(0.5)^t+A_2(0.4)^t.$$
> A constant particular solution gives $C=10$, so
> $$y_t=A_1(0.5)^t+A_2(0.4)^t+10.$$
> The initial conditions imply $A_1=1$ and $A_2=2$:
> $$\boxed{y_t=(0.5)^t+2(0.4)^t+10}.$$
> Both homogeneous terms vanish in the limit, so $y_t\to10$.
> <!-- bilingual-en:end -->

# 3. 蛛网模型 Cobweb Model
<!-- bilingual-en:start -->
*3. The cobweb model*
<!-- bilingual-en:end -->

## 3.1. 蛛网模型描述
<!-- bilingual-en:start -->
*3.1. Description*
<!-- bilingual-en:end -->

它描述了某些市场中**价格与供给-需求**的动态调整过程。当供给对过去价格的反应存在滞后时，市场价格会呈现振荡收敛或发散的行为，形成类似蜘蛛网的轨迹。
整个市场由三个方程构成
• **需求方程：**
$$d_t = a - \gamma p_t$$
表示第$t$期商品的需求量$d_t$随当期市场价格$p_t$的降低而增加（$\gamma>0$，价格上升导致需求下降）。
• **供给方程：**
$$s_t = b + \beta p_{t}^* + \varepsilon_t$$
表示第$t$期的供给量$s_t$取决于农民对价格的预期$p_{t}^*$以及一个随机供给冲击$\varepsilon_t$。<span style="color: yellow;">在此处,我们设定</span>$p_{t}^*=p_{t-1}$.<span style="color: yellow;">即本期的供给依据的是上一期的价格.</span>其中$b$为价格为零时的基准供给，$\beta>0$表示预期价格对供给的影响程度，$\varepsilon_t$是均值为零的随机扰动项（如天气导致的产量波动）。
• **市场出清条件：**
<!-- bilingual-en:start -->
The cobweb model describes adjustment among **price, supply, and demand** when supply responds to past prices.

- **Demand:** $d_t=a-\gamma p_t$, so demand falls as price rises.
- **Supply:** $s_t=b+\beta p_t^*+\varepsilon_t$, with <span style="color: yellow;">$p_t^*=p_{t-1}$</span>; current supply therefore uses last period's price.
- **Market clearing:**
<!-- bilingual-en:end -->

$$s_t = d_t$$


• 如果 $\big|\frac{\beta}{\gamma}\big| < 1$（供给曲线相对需求曲线**较平缓**），则特征根 $r = -\frac{\beta}{\gamma}$ 满足 $|r|<1$，价格齐次项 $A r^t$ 会指数衰减至0。此时价格的波动将**收敛**于长期均衡 $p^*$ 。具体表现为价格围绕 $p^*$ 震荡且幅度越来越小，最终趋于稳定。此情形被称为**收敛的蜘蛛网**。
• 如果 $\big|\frac{\beta}{\gamma}\big| > 1$（供给曲线**更陡峭**），则 $|r|>1$，齐次项的影响会随时间放大，导致价格波动**发散** 。也就是说，价格将越来越偏离均衡，呈现爆炸性的振荡。此为**发散的蜘蛛网**情况，现实中意味着市场不稳定，价格可能越来越极端。
• 如果 $\big|\frac{\beta}{\gamma}\big| = 1$，则 $|r|=1$，价格会**持续震荡**且幅度保持不变（$r=-1$ 会导致价格在两个值之间来回跳动，称为边际稳定或周期2振荡） 。这种情况下系统处于临界状态，轻微扰动将导致持续波动。
<!-- bilingual-en:start -->
- If $|\beta/\gamma|<1$, oscillations decay toward $p^*$: a **convergent cobweb**.
- If $|\beta/\gamma|>1$, oscillations grow: a **divergent cobweb**.
- If $|\beta/\gamma|=1$, oscillations persist; $r=-1$ gives a period-two cycle.
<!-- bilingual-en:end -->


![[Pasted image 20250316150336.png]]

## 3.2. 蛛网模型的数学解释
<!-- bilingual-en:start -->
*3.2. Mathematical explanation*
<!-- bilingual-en:end -->

联立上面的四个等式,得到$$a - \gamma p_t = b + \beta p_{t-1} + \varepsilon_t.$$这是一个一阶线性差分方程.
$$p_t = -\frac{\beta}{\gamma} p_{t-1} + \frac{a - b}{\gamma} - \frac{1}{\gamma} \varepsilon_t$$
这就对应了上面说的两条线绝对值之间的关系.只有$\beta<\gamma$时才会趋于稳定.
<!-- bilingual-en:start -->
Combining the equations gives
$$p_t=-\frac{\beta}{\gamma}p_{t-1}+\frac{a-b}{\gamma}-\frac{1}{\gamma}\varepsilon_t.$$
Stability requires $|\beta/\gamma|<1$, or $\beta<\gamma$ when both are positive.
<!-- bilingual-en:end -->

使用四步法求解:
<!-- bilingual-en:start -->
Apply four steps:
<!-- bilingual-en:end -->

 1. ==找通解==忽略常数和随机冲击，得到齐次方程 $p_t = -\frac{\beta}{\gamma} p_{t-1}$ 或等价写为
	$$p_t + \frac{\beta}{\gamma} p_{t-1} = 0$$
	其特征方程为 $r -(-\frac{\beta}{\gamma}) = r + \frac{\beta}{\gamma} = 0$，解得特征根 $r = -\frac{\beta}{\gamma}$。因此齐次通解为
	$$p_t^h = A \Big(-\frac{\beta}{\gamma}\Big)^t$$
2. ==找特解==
	$$p^* = -\frac{\beta}{\gamma} p^* + \frac{a - b}{\gamma}$$
	移项整理得到**长期均衡价格**
	$$p^* = \frac{a - b}{\gamma + \beta}$$
3. ==合并成为一般解== 将齐次解和特解相加，得到 (14) 的一般解：
$$p_t = \frac{a - b}{\gamma + \beta}- \frac{1}{\gamma} \sum_{i=0}^{\infty} \Big(-\frac{\beta}{\gamma}\Big)^i \varepsilon_{t-i} + A \Big(-\frac{\beta}{\gamma}\Big)^t$$
 4. ==使用初始条件求系数==
$$A = p_0 - \frac{a - b}{\gamma + \beta} + \frac{1}{\gamma} \sum_{i=0}^{\infty} \Big(-\frac{\beta}{\gamma}\Big)^i  \varepsilon_{-i}.$$
	最终得到的方程为:$$p_t = \frac{a - b}{\,\gamma + \beta\,} \;-\; \frac{1}{\gamma} \sum_{i=0}^{\,t-1} \Big(-\frac{\beta}{\gamma}\Big)^i \, \varepsilon_{t-i} \;+\; \Big(-\frac{\beta}{\gamma}\Big)^t \Big(p_0 - \frac{a - b}{\,\gamma + \beta\,}\Big)$$
	等式右侧的三部分都有其经济含义
	• 第一部分 $\frac{a - b}{\gamma + \beta}$ 是**长期均衡价格**。它由供需基本面决定，具有常数值。如果满足稳定性条件$\beta/\gamma<1$，那么价格序列${p_t}$会趋向于该均衡值。
	• 第二部分 $-\frac{1}{\gamma}\sum_{i=0}^{t-1}(-\frac{\beta}{\gamma})^i \varepsilon_{t-i}$ 是**冲击的短期影响**累积项。它捕捉了供给冲击对价格的即时和滞后影响。在稳定情形下，冲击的影响会逐步衰减（因为$|-\beta/\gamma|<1$时高次项很小），表示市场对供给扰动的短期调整过程。
	• 第三部分 $\Big(-\frac{\beta}{\gamma}\Big)^t\Big(p_0 - \frac{a - b}{\gamma + \beta}\Big)$ 是**初始偏离的影响**。它代表了初始价格偏离均衡时，价格如何随着时间动态调整回归均衡。如果$\beta/\gamma<1$，这一项会随着$t$增大而衰减（因为$|-\beta/\gamma|<1$），表明初始条件的影响逐渐消失；如果$\beta/\gamma>1$，这一项将随时间放大，体现出市场不稳定性。
<!-- bilingual-en:start -->
1. ==Homogeneous solution:==
   $$p_t^h=A\left(-\frac{\beta}{\gamma}\right)^t.$$
2. ==Particular solution:==
   $$p^*=\frac{a-b}{\gamma+\beta}.$$
3. ==General solution:== add the equilibrium, shock filter, and homogeneous term.
4. ==Initial condition:== determine $A$ from $p_0$.

The resulting expression separates the long-run equilibrium, accumulated short-run supply shocks, and the initial deviation. Under stability, shock weights and the initial deviation decay.
<!-- bilingual-en:end -->

# 4. 解二阶齐次微分方程Solving Second order Homogeneous [[Difference Equation|Difference]] Equations
<!-- bilingual-en:start -->
*4. Solving second-order homogeneous difference equations*
<!-- bilingual-en:end -->

#必考 <span style="color: yellow;">考试的时候只会出实数解,算出来虚数只可能是你算错了.</span>
<!-- bilingual-en:start -->
*Exam focus:* <span style="color: yellow;">Only real-root cases are examined; an unexpected complex root signals a calculation error.</span>
<!-- bilingual-en:end -->

是对前面的[[02_差分方程Difference Equation#2.3.2. 特征方程法（Characteristic Equation)|02_差分方程Difference Equation#2.3.2特征方程法（Characteristic Equation)]]的拓展.
<!-- bilingual-en:start -->
This extends the earlier [[02_差分方程Difference Equation#2.3.2. 特征方程法（Characteristic Equation)|characteristic-equation method]].
<!-- bilingual-en:end -->

## 4.1. 不同情况的判别式的处理
<!-- bilingual-en:start -->
*4.1. Cases determined by the discriminant*
<!-- bilingual-en:end -->

1. 当构筑的方程的根为两个不相等实根时,依照[[02_差分方程Difference Equation#2.3.2. 特征方程法（Characteristic Equation)|02_差分方程Difference Equation#2.3.2特征方程法（Characteristic Equation)]]来正常解.
2. 当构筑的方程的根中含有相等的实根时,此时仅有一个独立的解$A_1 \alpha_1^t$是不够的——我们需要找到另一个线性独立的齐次解。经验上，当重根出现时，可以尝试第二个解为$y_t = t \alpha^t$形式。因此，当$\alpha_1 = \alpha_2$时，齐次通解可以写为：
$$y_t^h = A_1 {\alpha_1}^t + A_2t{\alpha_1}^t$$
3. 当构筑的方程有虚数解的时候(这个不考).如果存在**共轭复根**：设一对复根为 $\lambda_{1,2} = \mu \pm i \nu$（其中 $\mu,\nu \in \mathbb{R}$，且 $\nu \neq 0$）。可以将它们表示为极坐标形式 $\lambda_{1,2} = r e^{\pm i \theta}$，其中 $r = \sqrt{\mu^2 + \nu^2}$，$\cos \theta = \mu/r$，$\sin \theta = \nu/r$。对应的齐次解可以用实数函数表示为：$$y_t^h = B_1 r^t \cos(\theta t) + B_2 r^t \sin(\theta t)$$或等价地写成 $y_t^h = C_1r^t \cos(\theta t + \phi)$ 的形式（$C_1,\phi$为常数），这与前述 $\beta_1 r^t \cos(\theta t + \beta_2)$ 表达一致 。这种形式保证了解为实数。从物理意义看，$r^t$部分决定振幅的膨胀或收缩，$\cos/\sin$部分决定震荡。比如若$r<1$，则振荡幅度随时间递减。
<!-- bilingual-en:start -->
1. With distinct real roots, apply the [[02_差分方程Difference Equation#2.3.2. 特征方程法（Characteristic Equation)|standard method]].
2. With a repeated root,
   $$y_t^h=A_1\alpha_1^t+A_2t\alpha_1^t.$$
3. With roots $re^{\pm i\theta}$,
   $$y_t^h=B_1r^t\cos(\theta t)+B_2r^t\sin(\theta t).$$
   The factor $r^t$ controls the amplitude; it decays when $r<1$.
<!-- bilingual-en:end -->

## 4.2. 稳定性条件
<!-- bilingual-en:start -->
*4.2. Stability conditions*
<!-- bilingual-en:end -->

对于二阶差分方程，稳定性可以通过特征根的大小来判断。一般地，**稳定要求所有特征根的绝对值都小于1**（即根落在复平面的单位圆内）。复数根就看模是否大于1.$$|\lambda| = r = \sqrt{(\text{实部})^2 + (\text{虚部})^2}$$
<!-- bilingual-en:start -->
Stability requires every characteristic root to have modulus below one. For a complex root,
$$|\lambda|=\sqrt{(\text{real part})^2+(\text{imaginary part})^2}.
<!-- bilingual-en:end -->

更一般的: 一个$n$阶差分方程稳定（解不发散）当且仅当**所有特征根都位于单位圆内部**，即$|\alpha_i|<1$对于$i=1,\ldots,n$均成立。通常有一些快速判别稳定性的充分必要条件：
• **必要条件：** 系数满足 $a_1 + a_2 + \cdots + a_n < 1$（所有滞后系数和小于1）。直观理解：如果系数和>=1，意味着单位根或爆炸根存在的可能性很大。
• **充分条件：** $|a_1| + |a_2| + \cdots + |a_n| < 1$（所有系数绝对值之和小于1）则一定稳定。这是绝对收敛条件，保证特征根绝对值<1。
• 特殊地，如果 $a_1 + \cdots + a_n = 1$，则特征多项式在$\alpha=1$处有根（即存在单位根），系统处于边界稳定（单位根过程）。例如一阶情况$a_1=1$就是单位根。
<!-- bilingual-en:start -->
For an $n$th-order recurrence, the roots remain the definitive stability test. The notes also give:

- $a_1+\cdots+a_n<1$ as a coefficient-sum check;
- $|a_1|+\cdots+|a_n|<1$ as a sufficient condition;
- $a_1+\cdots+a_n=1$ implies a unit root at 1.
<!-- bilingual-en:end -->

# 5. 找特解Finding Particular Solution
<!-- bilingual-en:start -->
*5. Finding a particular solution*
<!-- bilingual-en:end -->

可以理解为对[[02_差分方程Difference Equation#2.3.2. 特征方程法（Characteristic Equation)|02_差分方程Difference Equation#2.3.2特征方程法（Characteristic Equation)]]中讲到的第二步的深入探讨
<!-- bilingual-en:start -->
This develops the particular-solution step of the earlier [[02_差分方程Difference Equation#2.3.2. 特征方程法（Characteristic Equation)|procedure]].
<!-- bilingual-en:end -->

## 5.1. 确定性过程的特解 (Particular Solution for Deterministic Process)
<!-- bilingual-en:start -->
*5.1. Deterministic forcing*
<!-- bilingual-en:end -->

确定性过程指的是差分方程：$y_t = a_0 + a_1 y_{t-1} + \cdots + a_n y_{t-n} + x_t$中$x_t$是不含随机成分的确定性序列。
<!-- bilingual-en:start -->
The forcing process $x_t$ is deterministic when it contains no random component.
<!-- bilingual-en:end -->

此部分老师说“我不是很喜欢这个方法,感觉还不如迭代求解”.所以上课的时候草草带过.
<!-- bilingual-en:start -->
The lecturer preferred iteration, so this method was covered briefly.
<!-- bilingual-en:end -->

过程简单总结为根据驱动$x_t$ 选择对应的特解猜测形势,而后进行待定系数求解.对应关系如下:
<!-- bilingual-en:start -->
Match the trial solution to $x_t$, substitute it, and solve for its coefficients:
<!-- bilingual-en:end -->

| **驱动 $x_t$ 类型**  | **特解猜测形式 $y_t^p$**                     | **注意事项**                             |
| ---------------- | -------------------------------------- | ------------------------------------ |
| 常数 $C$ 或 0       | 常数 $y_t^p = C$                         | 若 $\sum a_i = 1$（[[趋势、单位根与差分#确定性与随机趋势|单位根]]），改试 $C t$ 等     |
| 指数 $B \lambda^t$ | 同形 $y_t^p = K \lambda^t$               | 若 $\lambda$ 是特征根，则改试 $K t \lambda^t$ |
| $d$次多项式 $B t^d$  | 同次多项式 $y_t^p = C_0 + \cdots + C_d t^d$ | 若 $r=1$ 为特征根（单位根），需升高次数（乘$t$）        |
<!-- bilingual-en:start -->
| **Forcing** | **Trial** | **Overlap adjustment** |
| --- | --- | --- |
| Constant | $C$ | With a [[趋势、单位根与差分#确定性与随机趋势|unit root]], try $Ct$. |
| $B\lambda^t$ | $K\lambda^t$ | If $\lambda$ is a root, try $Kt\lambda^t$. |
| $Bt^d$ | $C_0+\cdots+C_dt^d$ | With a root at 1, multiply by $t$. |
<!-- bilingual-en:end -->

>[!quote] 上述情况的具体分析(可略过,意义不大)
> **情形1：** $x_t$为常数或零（最简单情形）。
>
> • 如果$x_t = 0$（纯齐次方程有常数项$a_0$），那么可以猜测一个**常数特解**。令$y_t^p = c$为常数，代入方程得到：
>
> $$c = a_0 + a_1 c + a_2 c + \cdots + a_n c = a_0 + (a_1 + a_2 + \cdots + a_n) c.$$
>
> 解出$c = \frac{a_0}{,1 - (a_1 + \cdots + a_n),}$，前提是$1 - (a_1 + \cdots + a_n) \neq 0$。这个分母为零的情况意味着**单位根**存在（特征根$\alpha=1$导致齐次解中包含常数项），也就是$a_1 + \cdots + a_n = 1$，导致简单常数猜测失败。
>
> • 如果$1 - (a_1 + \cdots + a_n) = 0$（存在单位根），说明方程的齐次解本身有一个常数解分量，这时驱动为常数会产生一个**线性趋势特解**。换句话说，需要尝试$y_t^p = c , t$这样的形式。将$y_t^p = c t$代入方程：
>
> $$c t = a_0 + a_1 c (t-1) + a_2 c(t-2) + \cdots + a_n c(t-n).$$
>
> 展开整理，可以求出$c = \frac{a_0}{a_1 + 2a_2 + \cdots + n a_n}$（假设分母不为0）。若$ct$形式仍然失败，再尝试更高次幂，如$y_t^p = c t^2$，甚至$c t^m$直到成功为止。这对应于更高阶的单位根情形，例如二阶单位根可能需要二次多项式特解。
>
> _考场技巧：_ 碰到求特解时，如果$y_t^p = C$常数不奏效（通常是因为分母=0即齐次部分有相同形式的解），记住要**乘以$t$再试**。这是未定系数法的重要原则：当你猜的特解形式与齐次解的某部分重复时，乘以$t$能够产生一个新的独立形式。
>
> **情形2：** $x_t = B \cdot r^t$（指数形式驱动）。
>
> 这种情况下，驱动项随时间按指数规律变化。例如$x_t = B r^t$，$r$为常数基数（可能大于1或小于1）。我们猜测特解也可以包含一个与$r^t$相关的项。
>
> • 尝试令 $y_t^p = C_0 + C_1 r^t$（常数项加一个指数项）。将其代入方程：
>
> $$C_0 + C_1 r^t = a_0 + a_1 (C_0 + C_1 r^{t-1}) + \cdots + a_n (C_0 + C_1 r^{t-n}) + B r^t.$$
>
> 将左、右两边按照常数项和$r^t$项分别比较，可以得到关于$C_0, C_1$的方程。特别地，常数项比较会给出：
>
> $$C_0 = a_0 + a_1 C_0 + \cdots + a_n C_0,$$
>
> 这与情形1的常数解相同，解得 $C_0 = \frac{a_0}{1 - (a_1 + \cdots + a_n)}$（若满足条件）。$r^t$项比较则大致会给出：
>
> $$C_1 r^t = a_1 C_1 r^{t-1} + \cdots + a_n C_1 r^{t-n} + B r^t.$$
>
> 进一步化简可得到 $C_1 (1 - a_1 r^{-1} - \cdots - a_n r^{-n}) = B$，或者写成：
>
> $$C_1 = \frac{B}{r^n - a_1 r^{n-1} - \cdots - a_n}.$$
>
> 这个分母其实就是把特征多项式$\alpha^n - a_1 \alpha^{n-1} - \cdots - a_n$替换$\alpha = r$得到的值。如果$r$刚好是特征根（即分母=0，表示$r$与齐次解冲突），则$y_t^p = C_0 + C_1 r^t$失败，需要改用 $y_t^p = C_0 + C_1 t , r^t$ 再试，类似情形1乘以$t$的处理。这对应于驱动项频率与系统本身固有频率相同时的情况。
>
> 对于具体的一阶方程$y_t = a_0 + a_1 y_{t-1} + B r^t$，上述公式化简后结果为：
>
> $$C_0 = \frac{a_0}{1 - a_1}, \qquad C_1 = \frac{B}{,r - a_1,},$$
>
> 前提是$a_1 \neq 1$且$a_1 \neq r$。若$a_1 = 1$（单位根），我们需要在$C_0$部分乘$t$（即尝试$C_0 t$形式）；若$a_1 = r$（齐次解与驱动频率冲突），需要在$C_1 r^t$部分乘$t$（即尝试$C_1 t r^t$）。这与前述一般原则一致。
>
> **情形3：** $x_t = B , t^d$（多项式时间趋势）。
>
> $x_t$是关于时间$t$的一个$d$次多项式，例如线性趋势($d=1$)或二次趋势($d=2$)等。这时可以猜测特解也是一个$d$次多项式：
>
> $$y_t^p = C_0 + C_1 t + C_2 t^2 + \cdots + C_d t^d.$$
>
> 将其代入方程，将两边同次幂的$t$项进行比较，可以解出$C_0, C_1, \ldots, C_d$。解这些系数通常需要联立$d+1$个线性方程，过程比较机械。需要注意的是，如果在求解过程中发现分母为0或者解不出，通常意味着猜测的多项式次数需要提高（这往往是因为差分方程的齐次部分本身也有类似多项式解，例如包含单位根导致线性漂移）。例如，如果$x_t$是线性而齐次部分有单位根，则可能需要尝试二次多项式作为特解。
>
> <!-- bilingual-en:start -->
> **Constant forcing:** try $c$; if it overlaps a unit-root component, try $ct$ and then higher powers if needed.
>
> **Exponential forcing $Br^t$:** try $C_0+C_1r^t$; if $r$ is a characteristic root, replace $C_1r^t$ with $C_1tr^t$.
>
> **Polynomial forcing $Bt^d$:** try a degree-$d$ polynomial and equate coefficients; multiply by $t$ if the trial overlaps the homogeneous solution.
> <!-- bilingual-en:end -->

## 5.2. 随机过程的特解(Particular Solution for [[随机过程基础#随机过程的对象|Stochastic]] Process)
<!-- bilingual-en:start -->
*5.2. [[随机过程基础#随机过程的对象|Stochastic]] forcing*
<!-- bilingual-en:end -->

### (1) 待定系数法
<!-- bilingual-en:start -->
*(1) Method of undetermined coefficients*
<!-- bilingual-en:end -->

思路与上方确定性过程类似.**猜测解的结构只包含：时间函数、常数和${\varepsilon_t}$的项**
<!-- bilingual-en:start -->
Use the same idea as above, building the trial from time functions, constants, and innovations $\varepsilon_t$.
<!-- bilingual-en:end -->

>[!example] 一阶待定系数法的例子
>例如，在一阶随机方程$y_t = a_0 + a_1 y_{t-1} + \varepsilon_t$中，我们可以假设：$$y_t^p = b_0 + b_1 t + \sum_{i=0}^{\infty} \alpha_i \varepsilon_{t-i}$$
> <!-- bilingual-en:start -->
> $$y_t^p=b_0+b_1t+\sum_{i=0}^{\infty}\alpha_i\varepsilon_{t-i}.
> <!-- bilingual-en:end -->

### (2) 滞后算子法
<!-- bilingual-en:start -->
*(2) Lag-operator method*
<!-- bilingual-en:end -->

使用滞后算子法是有要求的.即需要满足上面的那个稳定性条件:特征方程的根都在单位圆内.<span style="color: yellow;">或者inverse characteristic equation 的根在单位圆外.算子多项式对应的根和特征方程的根互为倒数.</span>
<!-- bilingual-en:start -->
This method requires stability. Dynamic roots lie inside the unit circle, while reciprocal operator-polynomial roots lie outside it.
<!-- bilingual-en:end -->

算子多项式记为$A(L)$：$$A(L) = 1 - a_1L - a_2 L^2 - \cdots - a_p L^p$$
<!-- bilingual-en:start -->
Write
$$A(L)=1-a_1L-\cdots-a_pL^p.
<!-- bilingual-en:end -->

核心是将$y_{t-1}$写为$Ly$.使用滞后算子来替代滞后项.==滞后算子L不能进行简单的等式由左移到右乘变除.而是要进行求逆.过程类似反向的等比数列求和.==
在$|a_1|<1$的条件下，$(1 - a_1 L)$是可逆的，我们可以将其形式上展开为几何级数的和：
$$\frac{1}{1 - a_1 L} = 1 + a_1 L + a_1^2 L^2 + a_1^3 L^3 + \cdots.$$（这类似于$\frac{1}{1-x} = 1 + x + x^2 + \cdots$在$|x|<1$时成立）。
<!-- bilingual-en:start -->
Write $y_{t-1}=Ly_t$. Moving a lag polynomial across an equation means applying its inverse. When $|a_1|<1$,
$$\frac{1}{1-a_1L}=1+a_1L+a_1^2L^2+\cdots.
<!-- bilingual-en:end -->

>[!example] 二阶线性差分方程的求解
>
> 假定方程是稳定的以便做因式分解。设已知：
> $$1 - a_1 L - a_2 L^2 = (1 - b_1 L)(1 - b_2 L),$$
> 也就是说我们已经找到$A(L)$的因式分解形式（$b_1,b_2$就是特征根$\alpha_1,\alpha_2$，在稳定条件下$|b_1|<1, |b_2|<1$）。那么原方程可以改写为：
> $$(1 - b_1 L)(1 - b_2 L) y_t = a_0 + \varepsilon_t.$$
> 接下来分步求$y_t$：
> 1. 首先对$(1 - b_1 L)$取逆，等式两边同除$(1 - b_2 L)$，得到：$$(1 - b_1 L) y_t = \frac{a_0 + \varepsilon_t}{1 - b_2 L}$$
> 	由于$|b_2|<1$，可以使用几何级数展开$$\frac{1}{1 - b_2 L} x_t = x_t + b_2 x_{t-1} + b_2^2 x_{t-2} + \cdots.$$
> 	将右侧$a_0 + \varepsilon_t$乘以这个展开，我们得到：$$\frac{a_0 + \varepsilon_t}{1 - b_2 L} = a_0 + b_2 a_0 + b_2^2 a_0 + \cdots + \varepsilon_t + b_2 \varepsilon_{t-1} + b_2^2 \varepsilon_{t-2} + \cdots$$
> 	其中，$\frac{a_0}{1-b_2} = a_0 (1 + b_2 + b_2^2 + \cdots)$是一个常数项的和（如果收敛，这和$\frac{a_0}{1-b_2}$等值）。<span style="color: yellow;">此处由于a0是一个确定的常数,所以再怎么lag都不变,</span>因此上式可以写为：$$\frac{a_0 + \varepsilon_t}{1 - b_2 L} = \frac{a_0}{1 - b_2} + \sum_{i=0}^{\infty} b_2^i \varepsilon_{t-i}.$$
> 	这样，我们将右侧简化成了一个常数加一个冲击的累积项形式。现在方程变成：$$(1 - b_1 L) y_t = \frac{a_0}{1 - b_2} + \sum_{i=0}^{\infty} b_2^i \varepsilon_{t-i}.$$
>
> 2. 接着，对$(1 - b_1 L)$也取逆（同样由于$|b_1|<1$可以展开）。逆算子$(1 - b_1 L)^{-1}$的效果是：$$(1 - b_1 L)^{-1} x_t = x_t + b_1 x_{t-1} + b_1^2 x_{t-2} + \cdots.$$将上一步的结果乘以$(1 - b_1 L)^{-1}$，得到：$$y_t = \frac{a_0}{1 - b_2} \cdot \frac{1}{1 - b_1 L} + \frac{1}{1 - b_1 L} \sum_{i=0}^{\infty} b_2^i \varepsilon_{t-i}$$
> 	再次展开各部分：$$\frac{a_0}{1 - b_2} \cdot \frac{1}{1 - b_1 L} = \frac{a_0}{1 - b_2} (1 + b_1 L + b_1^2 L^2 + \cdots) = \frac{a_0}{1 - b_2} \frac{1}{1 - b_1}$$（这里我们用到了$\frac{1}{1 - b_1 L} a_0 = a_0 (1 + b_1 + b_1^2 + \cdots) = \frac{a_0}{1 - b_1}$，适用$|b_1|<1$）。
> 	第二部分：$$\frac{1}{1 - b_1 L} \sum_{i=0}^{\infty} b_2^i \varepsilon_{t-i} = \sum_{i=0}^{\infty} b_2^i \frac{1}{1 - b_1 L} \varepsilon_{t-i} = \sum_{i=0}^{\infty} b_2^i \sum_{j=0}^{\infty} b_1^j \varepsilon_{t-i-j}.$$调整求和顺序并让$k = i+j$，上式实际上会重排为类似$\sum_{k=0}^{\infty} (\text{某系数}) \varepsilon_{t-k}$形式，不用深入展开，我们知道它代表了$\varepsilon$的某种线性滤波。**重要的是**，这个结果依然是$\varepsilon$的冲击序列的加权和形式，因此整个$y_t$解可表示为：$$y_t = \frac{a_0}{(1 - b_1)(1 - b_2)} + \sum_{i=0}^{\infty} \tilde{\alpha}_i \varepsilon_{t-i},$$
> 	其中$\tilde{\alpha}_i$是由$b_1, b_2$共同作用得到的冲击权重。注意$(1 - b_1)(1 - b_2)$其实等于$1 - (b_1+b_2) + b_1 b_2$，而$b_1+b_2 = a_1, b_1 b_2 = -a_2$，所以$\frac{a_0}{(1 - b_1)(1 - b_2)} = \frac{a_0}{1 - a_1 - a_2}$，这正是我们未定系数法情形1中得到的常数解（假设$1 - a_1 - a_2 \neq 0$）。
> <!-- bilingual-en:start -->
>
> Factor
> $$1-a_1L-a_2L^2=(1-b_1L)(1-b_2L),\qquad |b_1|,|b_2|<1.$$
> Successive inversion produces a constant component
> $$\frac{a_0}{(1-b_1)(1-b_2)}=\frac{a_0}{1-a_1-a_2}$$
> and a linear filter of innovations
> $$\sum_{k=0}^{\infty}\tilde{\alpha}_k\varepsilon_{t-k}.$$
> This matches the constant particular solution from undetermined coefficients.
> <!-- bilingual-en:end -->


# 6. 关联卡片
<!-- bilingual-en:start -->
*6. Related notes*
<!-- bilingual-en:end -->

- Time Series Analysis-hub
- [[Difference Equation]]
- [[差分方程与滞后算子#滞后与差分算子|Difference Operator]]
- [[Lag Operator]]
- [[趋势、单位根与差分#差分与整合阶数|First Difference]]
- [[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|Autoregressive Model]]
- [[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|AR(1) stationarity]]
- [[趋势、单位根与差分#确定性与随机趋势|Random Walk]]
<!-- bilingual-en:start -->
- Time Series Analysis-hub
- [[Difference Equation]]
- [[差分方程与滞后算子#滞后与差分算子|Difference operator]]
- [[Lag Operator]]
- [[趋势、单位根与差分#差分与整合阶数|First difference]]
- [[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|Autoregressive model]]
- [[ARMA 模型：识别、估计、诊断与预测#AR、MA 与 ARMA|AR(1) stationarity]]
- [[趋势、单位根与差分#确定性与随机趋势|Random walk]]
<!-- bilingual-en:end -->

# 7. 作业
<!-- bilingual-en:start -->
*7. Exercises*
<!-- bilingual-en:end -->

## 1. 第 1 题
<!-- bilingual-en:start -->
*1. Question 1*
<!-- bilingual-en:end -->

1. Consider the [[Difference Equation|difference equation]] $y_t=a_0+a_1y_{t-1}$ with the initial condition $y_0$. Jill solved the difference equation by iterating backward:$$\begin{aligned}y_t &= a_0+a_1y_{t-1}\\   &= a_0+a_1(a_0+a_1y_{t-2})\\    &= a_0+a_0a_1+a_0a_1^2+\cdots+a_0a_1^{t-1}+a_1^ty_0\end{aligned}$$Bill added the homogeneous and particular solutions to obtain$$y_t=\frac{a_0}{1-a_1}+a_1^t\left[y_0-\frac{a_0}{1-a_1}\right].$$
	1. A. Show that the two solutions are identical for $|a_1|<1$.
	2. B. Show that for $a_1=1$, Jill's solution is equivalent to $y_t=a_0t+y_0$. How would you use Bill's method to arrive at this same conclusion in the case $a_1=1$?

 a)很简单,就是一个等比数列求和公式的应用
 b)Jill那个就硬推就行了.bill是按照特解-齐次解-通解的过程来搞的.实际上也可以使用洛必达法则来处理.总之就是不太重要
<!-- bilingual-en:start -->
a) Apply the finite geometric-series formula.
b) Jill uses direct iteration; Bill combines a particular and a homogeneous solution. The limiting case can also be handled with L’Hôpital's rule.
<!-- bilingual-en:end -->

## 3. 第 3 题
<!-- bilingual-en:start -->
*3. Question 3*
<!-- bilingual-en:end -->
3. Suppose that the money supply process has the form $m_t=m+\rho m_{t-1}+\varepsilon_t$, where $m$ is a constant and $0<\rho<1$.
	1. A. Show that it is possible to express $m_{t+n}$ in terms of the known value $m_t$ and the sequence $\{\varepsilon_{t+1},\varepsilon_{t+2},\ldots,\varepsilon_{t+n}\}$.
	2. B. Suppose that all values of $\varepsilon_{t+i}$ for $i>0$ have a mean value of zero. Explain how you could use your result in part A to forecast the money supply $n$ periods into the future.

a)使用迭代法
b)没看懂他要干嘛.总之就是不难
<!-- bilingual-en:start -->
a) Use iteration.
b) I did not initially understand part B, but the calculation is straightforward.
<!-- bilingual-en:end -->

## 7. 第 7 题
<!-- bilingual-en:start -->
*7. Question 7*
<!-- bilingual-en:end -->
7. 考虑随机过程
$$
y_t=a_0+a_1y_{t-1}+\varepsilon_t.
$$
a) 求齐次解并给出稳定性条件；
b) 用待定系数法求特解；
c) 证明与滞后算子法得到相同结果。
<!-- bilingual-en:start -->
7. Consider the stochastic process
$$
y_t=a_0+a_1y_{t-1}+\varepsilon_t.
$$
a) Find the homogeneous solution and state the stability condition.
b) Find a particular solution by undetermined coefficients.
c) Show that the lag-operator method gives the same result.
<!-- bilingual-en:end -->

a)正常的操作而已
b)设为常数然后解方程
c)[[02_差分方程Difference Equation#(2) 滞后算子法|02_差分方程Difference Equation#(2)滞后算子法]]
<!-- bilingual-en:start -->
a) Apply the standard homogeneous-solution procedure.
b) Try a constant particular solution.
c) See the [[02_差分方程Difference Equation#(2) 滞后算子法|lag-operator method]].
<!-- bilingual-en:end -->

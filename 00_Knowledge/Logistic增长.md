---
aliases:
  - Logistic growth
student_os: knowledge-atom
atom_id: CALC-APP-030
atom_set: derivative-applications
atom_type: model
status: source-checked
part_of:
  - "[[导数的应用.canvas]]"
requires:
  - "[[指数增长与衰减]]"
  - "[[分离变量]]"
related:
  - "[[稳态与稳定性]]"
  - "[[微分方程最大区间]]"
---

<!-- bilingual-en:start -->
# Logistic growth
<!-- bilingual-en:end -->

*Logistic 增长*

<!-- bilingual-en:start -->
**Logistic growth** models a population whose increase slows as its size approaches a characteristic level. It starts from [[指数增长与衰减|exponential growth]], where every unit of the current population has the same constant proportional rate, and makes that per-capita rate decline linearly with population size. This specifies a particular feedback mechanism, rather than just naming an S-shaped curve.
<!-- bilingual-en:end -->

**Logistic 增长**描述数量接近某个特征水平时，增长逐渐受到限制的人口。它从[[指数增长与衰减]]出发：指数模型中，每单位现有人口对应相同的恒定比例率；Logistic 模型则让这个人均比例率随人口规模线性下降。它规定了一种具体反馈机制，而不只是给 S 形曲线起名字。

<!-- bilingual-en:start -->
Let $y(t)\ge0$ be the modelled amount at time $t$. Fix $r>0$ with units of inverse time and $K>0$ with the same units as $y$. The logistic equation and, for positive $y$, its per-capita form are
<!-- bilingual-en:end -->

令 $y(t)\ge0$ 为时刻 $t$ 的模型数量。固定 $r>0$，单位为时间的倒数；固定 $K>0$，单位与 $y$ 相同。Logistic 方程及其在 $y>0$ 时的人均形式为

$$y'(t)=r y(t)\left(1-\frac{y(t)}K\right),\qquad
\frac{y'(t)}{y(t)}=r\left(1-\frac{y(t)}K\right).$$

<!-- bilingual-en:start -->
The ratio $y/K$ is dimensionless. At small positive $y/K$, the proportional rate is close to $r$; as $y$ increases, that rate falls. The parameter $K$ is called the carrying capacity: it is the equilibrium level towards which positive solutions move in forward time. It is a model parameter, not a rule forbidding an initial population above it.
<!-- bilingual-en:end -->

比值 $y/K$ 没有单位。当正的 $y/K$ 很小时，比例率接近 $r$；随着 $y$ 增大，这个比例率下降。参数 $K$ 称为承载量：它是正解沿向前时间趋近的平衡水平。它是模型参数，并不是禁止初始人口超过它的一条规则。

<!-- bilingual-en:start -->
## Read the direction before solving
<!-- bilingual-en:end -->

*求解前先看变化方向*

<!-- bilingual-en:start -->
An [[分离变量漏平衡解#Check the zero factor in the original equation|equilibrium]] here means a constant solution: the right-hand side is zero for all time. Substitution gives $y\equiv0$ and $y\equiv K$. Between them, $0<y<K$ makes $y'>0$; above $K$, it makes $y'<0$. These signs say which way the population moves, but the explicit solution below will also establish that a nonconstant positive trajectory approaches $K$ without crossing it in finite time.
<!-- bilingual-en:end -->

这里的[[分离变量漏平衡解#Check the zero factor in the original equation|平衡]]是一个常数解，即右端在所有时刻都为零。代入可得 $y\equiv0$ 与 $y\equiv K$。两者之间，$0<y<K$ 使 $y'>0$；高于 $K$ 时则有 $y'<0$。这些符号说明运动方向，下面的显式解还会证明：非恒定的正轨迹趋近 $K$，不会在有限时间越过它。

<!-- bilingual-en:start -->
For comparison, a [[稳态与稳定性|steady state of a discrete recurrence]] is defined by the next value equalling the present value. Here the continuous-time condition is zero derivative. In either setting, being an equilibrium and being approached from nearby states are different claims.
<!-- bilingual-en:end -->

作个比较，[[稳态与稳定性|离散递推的稳态]]要求下一步取值等于当前取值；这里连续时间的条件则是导数为零。在两种情形下，“是平衡状态”与“附近状态会趋近它”都是不同的判断。

<!-- bilingual-en:start -->
The course sometimes writes $y'=\rho y(s-y)$ with $\rho,s>0$. This is the same equation with $K=s$ and $r=\rho s$. The parameter $\rho$ alone is not the low-density per-capita rate: it has units of inverse amount per time.
<!-- bilingual-en:end -->

课程有时写成 $y'=\rho y(s-y)$，其中 $\rho,s>0$。令 $K=s$、$r=\rho s$，就得到相同方程。参数 $\rho$ 本身不是低密度时的人均增长率，它的单位是数量与时间乘积的倒数。

<!-- bilingual-en:start -->
## An exponential hidden inside the equation
<!-- bilingual-en:end -->

*方程里藏着一个指数解*

<!-- bilingual-en:start -->
Suppose $y(t_0)=y_0>0$. Introduce the ratio $q=K/y-1$, which measures how the capacity compares with the current population. Differentiating where $y>0$ gives
<!-- bilingual-en:end -->

设 $y(t_0)=y_0>0$。引入比值 $q=K/y-1$，描述承载量相对于当前人口的大小。在 $y>0$ 的地方求导，得到

$$q'=-\frac{K}{y^2}y'
=-r\left(\frac Ky-1\right)=-rq.$$

<!-- bilingual-en:start -->
This is the exponential equation just studied, with rate $-r$ and initial value $q_0=K/y_0-1$. Its solution is $q(t)=q_0e^{-r(t-t_0)}$. Solving $q=K/y-1$ back for $y$ yields
<!-- bilingual-en:end -->

这正是刚学过的指数方程，比例率为 $-r$，初值为 $q_0=K/y_0-1$。因此 $q(t)=q_0e^{-r(t-t_0)}$。再由 $q=K/y-1$ 解回 $y$，得到

$$y(t)=\frac{K}{1+\left(\frac{K-y_0}{y_0}\right)e^{-r(t-t_0)}}.$$

<!-- bilingual-en:start -->
At $t=t_0$, the denominator is $K/y_0$, so the initial value is correct. Differentiating the formula verifies the original equation wherever its denominator is nonzero. It includes the equilibrium $y_0=K$. The zero equilibrium is added separately because the substitution divided by $y$. Likewise, [[分离变量|separating variables]] by dividing by $y(K-y)$ requires retaining both equilibria first.
<!-- bilingual-en:end -->

在 $t=t_0$，分母为 $K/y_0$，所以初值正确。对公式求导，可验证它在分母非零处满足原方程。公式包含 $y_0=K$ 的平衡解；零平衡解则要另加，因为代换除以了 $y$。同样，若通过除以 $y(K-y)$ 来[[分离变量]]，也必须先保留两个平衡解。

<!-- bilingual-en:start -->
## Which times belong to this solution
<!-- bilingual-en:end -->

*哪些时间属于这个解*

<!-- bilingual-en:start -->
For $0<y_0<K$, the coefficient $(K-y_0)/y_0$ is positive. The denominator exceeds one at every finite time: $0<y(t)<K$ for all real $t$, and $y(t)\to K$ as $t\to+\infty$. At $y_0=K$, the solution stays at $K$ for all real times.
<!-- bilingual-en:end -->

若 $0<y_0<K$，系数 $(K-y_0)/y_0$ 为正。每个有限时刻的分母都大于一，所以对全部实数 $t$ 有 $0<y(t)<K$，且 $t\to+\infty$ 时 $y(t)\to K$。若 $y_0=K$，解在全部实数时间上恒等于 $K$。

<!-- bilingual-en:start -->
For $y_0>K$, the coefficient lies between $-1$ and zero. The denominator is positive and below one for every $t\ge t_0$, so $y(t)>K$ and decreases towards $K$. Going backwards, however, the denominator reaches zero at
<!-- bilingual-en:end -->

若 $y_0>K$，系数位于 $-1$ 与零之间。对所有 $t\ge t_0$，分母为正且小于一，所以 $y(t)>K$，并递减趋向 $K$。但是向后走时，分母会在下面的时刻变成零：

$$t_*=t_0+\frac1r\ln\left(1-\frac K{y_0}\right)<t_0.$$

<!-- bilingual-en:start -->
The [[微分方程最大区间|maximal solution interval]] containing the initial time is $(t_*,\infty)$. As $t\to t_*^+$, the population tends to $+\infty$; the same algebraic expression beyond that pole is not a continuation through it. Thus forward convergence does not imply existence for every real time. The zero equilibrium separately exists for all real times.
<!-- bilingual-en:end -->

包含初始时刻的[[微分方程最大区间|最大解区间]]是 $(t_*,\infty)$。当 $t\to t_*^+$ 时，数量趋于 $+\infty$；同一代数式在极点另一侧的值，不是穿过极点的延拓。因此向前收敛，并不意味着在全部实数时间上都有解。另行保留的零平衡解则在全部实数时间上存在。

<!-- bilingual-en:start -->
## Total growth and growth per individual are different
<!-- bilingual-en:end -->

*总增长与人均增长不同*

<!-- bilingual-en:start -->
The total growth rate is a quadratic function of the current amount. Completing the square shows where it is largest:
<!-- bilingual-en:end -->

总增长率是当前规模的二次函数。配方可直接找出它在哪里最大：

$$ry\left(1-\frac yK\right)=\frac{rK}{4}-\frac rK\left(y-\frac K2\right)^2.$$

<!-- bilingual-en:start -->
Its maximum over nonnegative $y$ is $rK/4$, attained at $y=K/2$. The per-capita rate $r(1-y/K)$ instead strictly decreases with $y$; for positive $y$ it approaches its [[上确界|supremum]] $r$, the smallest upper bound, as $y\to0^+$ without attaining it. A decreasing per-capita rate therefore need not mean a decreasing total growth rate.
<!-- bilingual-en:end -->

它在非负 $y$ 上的最大值为 $rK/4$，在 $y=K/2$ 取得。人均增长率 $r(1-y/K)$ 却随 $y$ 严格下降：在正数量范围中，当 $y\to0^+$ 时趋近[[上确界]] $r$，即最小的上界，但不取得它。因此人均比例率下降，不代表总增长速度也已下降。

<!-- bilingual-en:start -->
Differentiating the equation gives $y''=r y'(1-2y/K)$. On a growing trajectory with $0<y<K$, the [[导数判凹凸|second derivative]] is positive below $K/2$ and negative above it. The [[拐点|inflection point]] is therefore where that trajectory reaches $K/2$. If the initial value is already at or above $K/2$, the future trajectory need not display the full S shape. For $y_0>K$, it decreases and bends upwards throughout its interval.
<!-- bilingual-en:end -->

对方程求导得到 $y''=r y'(1-2y/K)$。在 $0<y<K$ 的增长轨迹上，$K/2$ 以下的[[导数判凹凸|二阶导数]]为正，以上为负。因此[[拐点]]出现在轨迹达到 $K/2$ 的位置。若初值已经位于或高于 $K/2$，未来轨迹未必展示完整的 S 形。若 $y_0>K$，它则在整个解区间上递减且向上弯。

> [!example] From 25 to 50 to 75 · 从 25 到 50 再到 75
> <!-- bilingual-en:start -->
> Take $K=100$ units, $r=\ln3$ per day and $y(0)=25$. Then $y(t)=100/(1+3e^{-(\ln3)t})$. At days $0,1,2,3$ the amounts are $25,50,75,90$. The total growth rate reaches its maximum $25\ln3$ units per day at day one. The population keeps increasing after that, while the daily instantaneous increase becomes smaller.
> <!-- bilingual-en:end -->
>
> 取 $K=100$ 单位、$r=\ln3$ 每天、$y(0)=25$。于是 $y(t)=100/(1+3e^{-(\ln3)t})$。第 $0,1,2,3$ 天的数量分别为 $25,50,75,90$。总增长率在第一天达到最大值 $25\ln3$ 单位每天。此后人口继续增加，但每天的瞬时增加速度变小。

<!-- bilingual-en:start -->
The model assumes constant parameters and an instantaneous, linear relation between per-capita rate and population. It includes no time delay, age structure or random shocks. An observed S shape alone cannot establish those assumptions, and a fitted carrying capacity need not remain fixed if the environment changes.
<!-- bilingual-en:end -->

这个模型假设参数恒定，人均比例率与当前人口之间具有即时的线性关系，不包含时滞、年龄结构或随机冲击。仅凭观测到的 S 形，不能确定这些假设成立；环境改变时，拟合出的承载量也未必保持不变。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[Exercise040_Solutions.pdf#page=1|MIT 18.01SC Exercise 40 solutions, pp.1–2]]: the course parameterisation, separation and equilibrium directions. Its phrase about decreasing until reaching or falling below capacity is replaced here by the finite-time noncrossing result from the explicit solution.
- [Hunter, Introduction to Dynamical Systems, §2.2, PDF pp.22–23](https://www.math.ucdavis.edu/~hunter/m207/m207.pdf#page=22): the logistic feedback model, normalised explicit solution and forward convergence.
- [OpenStax Calculus 2, §4.4](https://openstax.org/books/calculus-volume-2/pages/4-4-the-logistic-equation): carrying capacity, the solution in Theorem 4.2 and the inflection calculation. The positive-parameter assumptions and total-versus-per-capita distinction are explicit above; the pole and numerical example are independently calculated.
- [[01_Math/01_calculus/02_Applications_of_Differentiation#配套练习：指数与受限增长|Permanent course record, exponential and limited growth exercises]]: notation and the original exercise context.
<!-- bilingual-en:end -->

- [[Exercise040_Solutions.pdf#page=1|MIT 18.01SC 第40次练习题解，第1–2页]]：课程参数写法、分离变量与平衡方向；原文关于下降到或低于承载量的措辞，按显式解改为有限时间不穿越。
- Hunter《Introduction to Dynamical Systems》第2.2节，PDF 第22–23页：Logistic 反馈、标准化显式解与向前收敛。
- OpenStax《Calculus 2》第4.4节：承载量、定理4.2解公式及拐点计算；正文明确正参数条件和总增长与人均增长的区别，并独立计算极点与数值例子。
- [[01_Math/01_calculus/02_Applications_of_Differentiation#配套练习：指数与受限增长|永久课程记录：指数与受限增长练习]]：符号与原练习背景。

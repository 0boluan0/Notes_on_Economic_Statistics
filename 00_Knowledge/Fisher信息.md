---
student_os: knowledge-atom
atom_id: STAT-INF-FISHER-001
atom_type: definition
status: source-checked
aliases:
  - Fisher information
  - Expected Fisher information
  - 费舍尔信息
  - 费雪尔信息
---

<!-- bilingual-en:start -->
# Fisher information
<!-- bilingual-en:end -->

*Fisher 信息*

<!-- bilingual-en:start -->
**Fisher information** measures the expected squared sensitivity of a model’s log likelihood to a parameter. It is an average over all possible data under a specified model, not the slope from one observed sample. The parameter, its units and the amount of data must be specified before information values can be compared.
<!-- bilingual-en:end -->

**Fisher 信息**衡量：模型对数似然对某个参数的敏感程度，平方后在可能数据上平均有多大。它是按指定模型对所有可能数据求平均，并非某一个样本的斜率。比较信息量之前，需要明确参数、参数单位和观测数据量。

<!-- bilingual-en:start -->
## Square the score before averaging
<!-- bilingual-en:end -->

*先把得分平方，再求平均*

<!-- bilingual-en:start -->
For a [[Bernoulli分布|Bernoulli observation]] $X$, let $p=P(X=1)$ with $0<p<1$. The [[得分函数|score]] differentiates the log probability with respect to $p$, holding the observation fixed. It is $1/p$ for a success and $-1/(1-p)$ for a failure. Its mean is zero, so the ordinary average would hide its size. Squaring first avoids cancellation:
<!-- bilingual-en:end -->

对一个 [[Bernoulli分布|Bernoulli 观测]] $X$，设 $p=P(X=1)$、$0<p<1$。[[得分函数]]保持观测不动，对概率的对数关于 $p$ 求导。成功时得分为 $1/p$，失败时为 $-1/(1-p)$。它的均值为零，直接平均会把大小信息抵消；先平方则得到

$$
I_1(p)=p\left(\frac1p\right)^2+(1-p)\left(-\frac1{1-p}\right)^2=\frac1{p(1-p)}.
$$

> [!example] One trial at p = 1/4 · 成功概率为 1/4 的一次试验
> <!-- bilingual-en:start -->
> The score is $4$ with probability $1/4$ and $-4/3$ with probability $3/4$. Its mean is $1-1=0$, but its expected square is $(1/4)16+(3/4)(16/9)=16/3$. This $16/3$ is information about the success probability from one trial at this parameter value; it is neither the variance of $X$ nor a realised estimate of $p$.
> <!-- bilingual-en:end -->
>
> 得分以概率 $1/4$ 取 $4$，以概率 $3/4$ 取 $-4/3$。其均值为 $1-1=0$，但平方的期望为 $(1/4)16+(3/4)(16/9)=16/3$。这个 $16/3$ 表示在该参数值处，一次试验关于成功概率的信息量；它既不是 $X$ 的方差，也不是某次对 $p$ 的估计值。

<!-- bilingual-en:start -->
## State the model and sample size
<!-- bilingual-en:end -->

*明确模型与样本口径*

<!-- bilingual-en:start -->
Let $X$ denote the whole observation and $f_\theta(x)$ its [[概率密度函数|density]] or [[概率质量函数|mass function]], with scalar parameter $\theta$ in an open interval. Write $S_\theta(X)=\partial_\theta\log f_\theta(X)$ for the score. The expected Fisher information is
<!-- bilingual-en:end -->

用 $X$ 表示整个观测，$f_\theta(x)$ 表示其[[概率密度函数|密度]]或[[概率质量函数]]，标量参数 $\theta$ 位于开区间。记得分为 $S_\theta(X)=\partial_\theta\log f_\theta(X)$。期望 Fisher 信息为

$$
I(\theta)=E_\theta[S_\theta(X)^2].
$$

<!-- bilingual-en:start -->
The subscript means both the score and the averaging distribution use the same parameter value. The nonnegative expectation can in principle be infinite; finite positive information is an additional condition when using reciprocal-information bounds. In a regular model with common support and a justified first differentiation exchange, the score has mean zero, so $I(\theta)=\operatorname{Var}_\theta(S_\theta)$. The definition as an expected square does not justify silently invoking that identity in a nonregular model.
<!-- bilingual-en:end -->

下标表示得分与求平均所用的分布采用同一个参数值。非负变量的期望原则上可以为无穷大；使用信息倒数作为界时，有限且严格为正是额外条件。在共同支撑、第一次求导交换合法的正则模型中，得分均值为零，所以 $I(\theta)=\operatorname{Var}_\theta(S_\theta)$。仅写成平方期望，并不能允许在非正则模型中默默套用这个等式。

<!-- bilingual-en:start -->
With $n$ [[独立同分布|independent and identically distributed]] regular observations, their scores add and are independent with mean zero. Consequently $I_n(\theta)=nI_1(\theta)$. Three Bernoulli trials at $p=1/4$ have information $3(16/3)=16$. Here $I_1$ refers to one observation and $I_n$ to the full sample; multiplying a full-sample value by $n$ again double-counts the data. Dependent observations require the joint model and do not automatically give this addition rule.
<!-- bilingual-en:end -->

若有 $n$ 个[[独立同分布]]的正则观测，各观测得分相加，且相互独立、均值为零。因此 $I_n(\theta)=nI_1(\theta)$。在 $p=1/4$ 处，三个 Bernoulli 试验的信息为 $3(16/3)=16$。这里 $I_1$ 对应一个观测，$I_n$ 对应整个样本；把全样本信息再乘一次 $n$，就重复计算了数据。依赖观测需要使用联合模型，不能自动沿用这一相加规则。

<!-- bilingual-en:start -->
## Expected information and observed curvature
<!-- bilingual-en:end -->

*期望信息与观测到的曲率*

<!-- bilingual-en:start -->
If the log likelihood $\ell(\theta;x)=\log f_\theta(x)$ is twice differentiable, its negative second derivative $J(\theta;x)=-\partial_\theta^2\ell(\theta;x)$ is called **observed information**. It uses the actual data $x$. To identify its expectation with Fisher information, more is needed: the second derivative must also pass through the probability integral or sum, and the displayed expectations must be finite. The algebra is
<!-- bilingual-en:end -->

如果对数似然 $\ell(\theta;x)=\log f_\theta(x)$ 二次可微，其负二阶导数 $J(\theta;x)=-\partial_\theta^2\ell(\theta;x)$ 称为**观测信息（observed information）**，它使用实际数据 $x$。要把它的期望认作 Fisher 信息，还需额外条件：二阶导数也能穿过总概率的积分或求和，并且所列期望有限。对应代数为

$$
\partial_\theta^2\log f_\theta=\frac{\partial_\theta^2f_\theta}{f_\theta}-\left(\frac{\partial_\theta f_\theta}{f_\theta}\right)^2,
\qquad E_\theta[J(\theta;X)]= -\int\partial_\theta^2f_\theta(x)\,dx+I(\theta)=I(\theta).
$$

<!-- bilingual-en:start -->
The last integral vanishes only after the second differentiation exchange is justified: it equals the second derivative of total probability one. Thus the familiar formula $I(\theta)=-E_\theta[\partial_\theta^2\log f_\theta(X)]$ is a regularity-based identity, not a licence to replace every expected square by a second derivative.
<!-- bilingual-en:end -->

最后一个积分为零，依赖第二次求导交换合法：它才等于总概率一的二阶导数。因此常见公式 $I(\theta)=-E_\theta[\partial_\theta^2\log f_\theta(X)]$ 是依靠正则条件成立的恒等式，不是允许把任意平方期望都换成二阶导。

> [!warning] The expectation is not the observed value · 期望不等于某次观测值
> <!-- bilingual-en:start -->
> For one Bernoulli trial, $J(p;x)=x/p^2+(1-x)/(1-p)^2$. At $p=1/4$, it is $16$ if $x=1$ and $16/9$ if $x=0$. Neither equals $I_1(p)=16/3$, although their probability-weighted average does. Observed information, the score itself, and the observed squared score are conceptually different quantities. A model may make some of their numerical values coincide, but that is not a general identity.
> <!-- bilingual-en:end -->
>
> 对一个 Bernoulli 试验，$J(p;x)=x/p^2+(1-x)/(1-p)^2$。当 $p=1/4$ 时，若 $x=1$，它是 $16$；若 $x=0$，它是 $16/9$。两者都不等于 $I_1(p)=16/3$，但按概率加权平均后等于。观测信息、得分本身与观测得分的平方是不同的概念；某些模型可能使它们中的某些数值相同，却不是一般恒等式。

<!-- bilingual-en:start -->
## What information says about precision
<!-- bilingual-en:end -->

*它怎样联系估计精度*

<!-- bilingual-en:start -->
For a [[正态分布|normal observation]] $X\sim N(\mu,\sigma^2)$ with known $\sigma^2>0$ and unknown mean $\mu$, the score is $(X-\mu)/\sigma^2$ and $I_1(\mu)=1/\sigma^2$. Less measurement noise gives more information about the mean. With $n$ iid observations, $I_n(\mu)=n/\sigma^2$. Under the full conditions of the [[Cramér–Rao下界|Cramér–Rao bound]], an unbiased estimator of $\mu$ has variance at least $1/I_n=\sigma^2/n$; the sample mean attains it in this model. Information alone does not promise that any estimator in any model achieves its reciprocal.
<!-- bilingual-en:end -->

对于[[正态分布|正态观测]] $X\sim N(\mu,\sigma^2)$，若 $\sigma^2>0$ 已知、均值 $\mu$ 未知，得分为 $(X-\mu)/\sigma^2$，$I_1(\mu)=1/\sigma^2$。测量噪声越小，关于均值的信息越多。$n$ 个 iid 观测给出 $I_n(\mu)=n/\sigma^2$。满足[[Cramér–Rao下界]]的完整条件时，$\mu$ 的无偏估计量方差至少为 $1/I_n=\sigma^2/n$；在此模型中样本均值达到它。但仅有信息量，不保证任意模型都有估计量达到其倒数。

<!-- bilingual-en:start -->
Information is tied to the chosen parameter scale. If the same model is rewritten using $\theta=h(\eta)$, the chain rule gives $S_\eta=h^{\prime}(\eta)S_\theta$ and $I_\eta=[h^{\prime}(\eta)]^2I_\theta$, evaluated at $\theta=h(\eta)$. A change of units therefore changes the number without adding data. Likewise, the Bernoulli formula above applies only to $0<p<1$; its divergence as $p$ approaches an endpoint does not justify substituting $p=0$ or $1$ into the regular formulas. If information is zero, a reciprocal bound cannot be formed; zero local first-order sensitivity alone does not settle global identification.
<!-- bilingual-en:end -->

信息量与所选参数尺度相连。若用 $\theta=h(\eta)$ 重写同一个模型，链式法则给出 $S_\eta=h^{\prime}(\eta)S_\theta$、$I_\eta=[h^{\prime}(\eta)]^2I_\theta$，右侧在 $\theta=h(\eta)$ 处计算。因此只换计量单位就能改变数值，并没有增加数据。同样，前面的 Bernoulli 公式只适用于 $0<p<1$；靠近端点时公式发散，并不允许把 $p=0$ 或 $1$ 直接代入正则公式。若信息为零，就不能取倒数来形成下界；局部一阶敏感度为零，本身也不能判断全局识别。

## 来源与核验

<!-- bilingual-en:start -->
- [Berkeley Stat 210A, Score Function and Fisher Information, §§2, 4–5](https://stat210a.berkeley.edu/fall-2025/reader/score-fisher.html): supports the regular score identities, independent-sample addition and change of parameter scale.
- [Geyer, Stat 5102 Notes, §2.2 and Example 2.1, PDF pp. 4–7](https://www.stat.umn.edu/geyer/s06/5102/notes/fish.pdf#page=4): supports expected versus observed information and the Bernoulli/binomial calculation. The negative-second-derivative convention follows equation (2.12) and the worked example; equation (2.10) omits a minus sign in its expanded sum.
- [Shao, Stat 610 Lecture 3, PDF pp. 9–12 and 15](https://pages.stat.wisc.edu/~shao/stat610/stat610-03.pdf#page=9): supports the variance-bound conditions, information additivity/parameter transformation, and normal-mean calculation. The one-trial numbers are evaluated explicitly above.
<!-- bilingual-en:end -->

- Berkeley Stat 210A 第 2、4–5 节核对正则得分恒等式、独立样本相加与参数尺度变换。
- Geyer Stat 5102 第 2.2 节及例 2.1（PDF pp. 4–7）核对期望信息与观测信息、Bernoulli/二项计算。负二阶导约定按式 (2.12) 与算例核验；式 (2.10) 展开后的求和漏了负号。
- Shao Stat 610 Lecture 3（PDF pp. 9–12、15）核对方差界条件、信息可加性、参数变换及正态均值计算。单次试验数值在正文直接展开。


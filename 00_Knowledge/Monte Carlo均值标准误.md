---
aliases:
  - "Monte Carlo standard error"
  - "Standard error of a Monte Carlo mean"
  - "蒙特卡洛均值标准误"
student_os: knowledge-atom
atom_id: PROB-MC-002
atom_set: monte-carlo-methods
atom_type: theorem
status: source-checked
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[Monte Carlo估计]]"
  - "[[期望线性性]]"
  - "[[方差的仿射变换]]"
leads_to:
  - "[[Monte Carlo近似区间]]"
  - "[[Monte Carlo平方根律]]"
related:
  - "[[标准误含义]]"
  - "[[矩存在性的使用边界]]"
  - "[[路径数不修模型]]"
---

<!-- bilingual-en:start -->
# Standard error of a Monte Carlo mean
<!-- bilingual-en:end -->

*Monte Carlo 均值标准误*

<!-- bilingual-en:start -->
A Monte Carlo run produces an average, but a fresh run with new random inputs would usually produce a different average. The [[标准误含义|standard error]] measures the spread of those repeated averages. It is different from the spread of the individual simulation outputs, and it is not the actual error of the one answer currently on screen.
<!-- bilingual-en:end -->

一次 Monte Carlo 运行会给出平均数，重新生成随机输入再运行，通常得到不同的平均数。[[标准误含义|标准误]]衡量这些重复平均数的分散程度。它不同于单个模拟输出的分散程度，也不是屏幕上这一次答案的实际误差。

> [!example] Individual outputs and their average · 单次输出与均值
> <!-- bilingual-en:start -->
> Suppose one output has variance $9$, so its [[标准差|standard deviation]] is $3$. Average $N=100$ independent outputs from this same distribution. The standard error of that average is $3/\sqrt{100}=0.3$. Thus individual outputs fluctuate on scale $3$, whereas repeated 100-output averages fluctuate on scale $0.3$. This does not assert that every average is within $0.3$ of the target.
> <!-- bilingual-en:end -->
>
> 假设单次输出方差为 $9$，即[[标准差]]为 $3$。平均来自同一分布的 $N=100$ 个独立输出，均值的标准误为 $3/\sqrt{100}=0.3$。单次输出的波动尺度是 $3$，不同的百次平均的波动尺度是 $0.3$；这不表示每次平均都离目标不超过 $0.3$。

<!-- bilingual-en:start -->
## The exact variance calculation
<!-- bilingual-en:end -->

*先明确什么是固定的、什么是随机的*

<!-- bilingual-en:start -->
In [[Monte Carlo估计|simple Monte Carlo]], the real output on trial $i$ is $Y_i=g(X_i)$: the random input is $X_i$ and the fixed output rule is $g$. Choose a fixed positive integer $N$ before seeing the outputs. Assume $Y_1,\ldots,Y_N$ are [[独立同分布|iid]] and have [[矩存在性的使用边界|finite second moment]] $E[Y_i^2]<\infty$: their squared output has a finite expectation. Write $\mu=E[Y_i]$ for their common mean and $\sigma^2=\operatorname{Var}(Y_i)$ for their common variance, allowing $\sigma^2=0$. These are properties of the **output** distribution, not necessarily of the input $X_i$. Define
$$\widehat\mu_N=\frac1N\sum_{i=1}^N Y_i.$$
The true standard error is the square root of this average's variance under repeated simulation with the same model and the same $N$.
<!-- bilingual-en:end -->

在[[Monte Carlo估计|简单 Monte Carlo]]中，第 $i$ 次的实数输出是 $Y_i=g(X_i)$，其中 $X_i$ 是随机输入，$g$ 是固定输出规则。观察结果前先选定正整数 $N$。假设输出 $Y_1,\ldots,Y_N$ [[独立同分布|iid]]，且[[矩存在性的使用边界|二阶矩有限]]，即平方输出的期望 $E[Y_i^2]<\infty$。用 $\mu=E[Y_i]$ 表示共同均值、$\sigma^2=\operatorname{Var}(Y_i)$ 表示共同方差，允许 $\sigma^2=0$。这些量描述的是**输出**分布，不一定是输入 $X_i$ 的分布。均值估计量如上；真实标准误就是保持模型和 $N$ 不变、反复模拟时，这个平均数的方差平方根。

<!-- bilingual-en:start -->
[[期望线性性|Linearity of expectation]] averages the common means. For the variance, the factor $1/N$ becomes $1/N^2$ by the [[方差的仿射变换|variance scaling rule]], and independence makes all the cross-covariances in the [[和的方差协方差项|variance of the sum]] vanish:
<!-- bilingual-en:end -->

[[期望线性性]]把相同的均值取平均。计算方差时，[[方差的仿射变换|方差缩放规则]]把 $1/N$ 变成 $1/N^2$，独立又使[[和的方差协方差项|求和方差]]中的交叉协方差全部消失：

> [!important] Exact moments, without a normality assumption · 不需要正态的精确矩结果
> $$E[\widehat\mu_N]=\mu,\qquad
> \operatorname{Var}(\widehat\mu_N)
> =\frac1{N^2}\sum_{i=1}^N\sigma^2
> =\frac{\sigma^2}{N},\qquad
> SE(\widehat\mu_N)=\frac{\sigma}{\sqrt N}.$$
> <!-- bilingual-en:start -->
> Here $\sigma=\sqrt{\sigma^2}\ge0$. These identities hold for every fixed $N\ge1$ under the stated assumptions, even if the outputs are highly nonnormal. The expectation identity needs only a common finite mean; independence is a sufficient condition for the variance calculation, rather than a requirement for unbiasedness alone.
> <!-- bilingual-en:end -->
>
> 这里 $\sigma=\sqrt{\sigma^2}\ge0$。给定条件下，每个固定 $N\ge1$ 都满足这些恒等式，即使输出很不正态也一样。期望恒等式只需共同的有限均值；独立是这条方差计算的充分条件，不是单独保证无偏所必需的条件。

<!-- bilingual-en:start -->
Since the mean is [[估计量无偏性|unbiased]] for $\mu$, its [[均方误差|root mean squared error]] is also $\sigma/\sqrt N$: there is no squared-bias term to add. Both standard error and root mean squared error have the same units as $Y_i$. A confidence interval still requires a separate distributional or approximation argument, developed in [[Monte Carlo近似区间|Monte Carlo interval estimation]]; the exact variance formula alone does not supply a coverage percentage.
<!-- bilingual-en:end -->

因为这个均值对 $\mu$ [[估计量无偏性|无偏]]，其[[均方误差|均方根误差]]也等于 $\sigma/\sqrt N$，不需要另加偏差平方项。标准误与均方根误差都和 $Y_i$ 同单位。置信区间仍需另外的分布或近似论证，见[[Monte Carlo近似区间]]；精确方差公式本身并不给出覆盖率。

<!-- bilingual-en:start -->
## Estimate the unknown output variance from the run
<!-- bilingual-en:end -->

*用这一轮输出估计未知尺度*

<!-- bilingual-en:start -->
Usually $\sigma^2$ is unknown. For $N\ge2$, let $s_N^2$ be the usual [[样本协方差无偏分母|sample variance with denominator $N-1$]], and let $s_N$ be its nonnegative square root. The deviations $Y_i-\widehat\mu_N$ measure each output's distance from this run's average. Define
$$s_N^2=\frac1{N-1}\sum_{i=1}^N(Y_i-\widehat\mu_N)^2,
\qquad
\widehat{SE}(\widehat\mu_N)=\frac{s_N}{\sqrt N}.$$
Under the iid finite-variance assumptions, $E[s_N^2]=\sigma^2$, so $s_N^2/N$ is an unbiased estimate of the mean's variance. Taking a square root is nonlinear: $s_N$ and $s_N/\sqrt N$ are **not generally unbiased** for the corresponding standard deviations. The hat reminds us that the reported standard error is itself an estimate that varies between runs.
<!-- bilingual-en:end -->

通常 $\sigma^2$ 未知。对 $N\ge2$，用[[样本协方差无偏分母|分母为 $N-1$ 的样本方差]] $s_N^2$ 估计它，再取非负平方根 $s_N$。差值 $Y_i-\widehat\mu_N$ 衡量每个输出偏离本轮均值多少。在 iid 有限方差条件下，$E[s_N^2]=\sigma^2$，所以 $s_N^2/N$ 无偏估计均值的方差。但开平方是非线性操作，$s_N$ 与 $s_N/\sqrt N$ **一般不无偏**估计对应的标准差。帽子提醒我们：报告的标准误本身也是估计值，会随运行改变。

> [!example] A standard error calculated from four outputs · 四个输出算出的估计标准误
> <!-- bilingual-en:start -->
> For the observed outputs $2,4,4,6$, the mean is $4$ and the squared deviations sum to $4+0+0+4=8$. Hence
> $$s_4^2=\frac83,\qquad \widehat{SE}(\widehat\mu_4)
> =\sqrt{\frac{8/3}{4}}=\sqrt{\frac23}\approx0.816.$$
> This is an estimated uncertainty scale for the mean, assuming the four outputs came from the stated iid model. These four numbers alone do not reveal the true $\sigma$ or the true target $\mu$.
> <!-- bilingual-en:end -->
>
> 对实际输出 $2,4,4,6$，平均为 $4$，平方偏差之和为 $8$，所以 $s_4^2=8/3$，估计标准误约为 $0.816$。这是在四个输出来自上述 iid 模型的前提下，估计出来的均值不确定性尺度；四个数本身并不告诉我们真实的 $\sigma$ 或目标 $\mu$。

<!-- bilingual-en:start -->
When $N=1$, the theoretical formula still applies if $\sigma$ is known, but there is no usual within-run sample variance with denominator $N-1$. If the true $\sigma^2=0$, the output is constant with probability $1$ and the theoretical standard error is zero. Observing $s_N=0$ in one finite run does not establish that fact: for example, a nonconstant binary output can happen to take the same value on every trial.
<!-- bilingual-en:end -->

$N=1$ 时，若已知 $\sigma$，理论公式仍成立，但无法用分母 $N-1$ 定义通常的本轮样本方差。若真实 $\sigma^2=0$，输出以概率 $1$ 为常数，理论标准误为零。一次有限运行里看到 $s_N=0$，却不能断定真实方差为零；非常数的二元输出也可能恰好次次相同。

<!-- bilingual-en:start -->
## Check the sampling scheme before dividing by the square root
<!-- bilingual-en:end -->

*先检查重复是否提供了新的信息*

<!-- bilingual-en:start -->
For arbitrary finite-second-moment outputs, the actual variance is
$$\operatorname{Var}(\widehat\mu_N)
=\frac1{N^2}\left[\sum_{i=1}^N\operatorname{Var}(Y_i)
+2\sum_{i<j}\operatorname{Cov}(Y_i,Y_j)\right].$$
The second sum includes each pair of distinct trials once. Pairwise zero covariance and a common variance are already sufficient for the simpler $\sigma^2/N$ result; full independence is stronger. Conversely, nonzero pairwise covariances can cancel, so independence is not a necessary condition for that numerical variance identity.
<!-- bilingual-en:end -->

对任意二阶矩有限的输出，均值方差由上式给出，第二个求和把每对不同实验各计一次。两两零协方差且方差相同，已经足以推出 $\sigma^2/N$，完全独立是更强的条件。反过来，非零的协方差也可能相互抵消，所以不能把独立说成这个数值方差恒等式的必要条件。

> [!warning] Copying one random output · 复制一次随机结果
> <!-- bilingual-en:start -->
> For $N\ge2$, if a program generates one nonconstant output $Y$ and copies it $N$ times, then every entry has the correct marginal distribution, but the average is still exactly $Y$. Its standard error is $\sigma$, while the sample variance of the copied entries is zero. Counting entries as independent trials would therefore both invent precision and conceal uncertainty.
> <!-- bilingual-en:end -->
>
> 对 $N\ge2$，若程序只生成一次非常数输出 $Y$，再复制 $N$ 份，每个位置的边际分布都正确，可平均数仍恰好是 $Y$，标准误仍为 $\sigma$；复制条目的样本方差却等于零。把这些条目当成独立实验，既会虚构精度，又会掩盖不确定性。

<!-- bilingual-en:start -->
Infinite output variance gives no finite standard error of this form. Correlated sampling, MCMC, quasi-Monte Carlo, stratification, and nested estimators need an error formula matched to their design. The comparison between independent trial counts is developed in [[Monte Carlo平方根律|the square-root law]]. All these precision statements concern the stated model and output rule; [[路径数不修模型|additional paths do not validate the model]].
<!-- bilingual-en:end -->

输出方差无限时，这种意义下没有有限标准误。相关抽样、MCMC、quasi-Monte Carlo、分层或嵌套估计需要与具体设计匹配的误差公式。独立实验次数之间如何换算精度，见[[Monte Carlo平方根律]]。这些精度结论始终针对明示模型与输出规则，[[路径数不修模型|增加路径并不验证模型]]。

## 来源与核验

<!-- bilingual-en:start -->
- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2, §2.1, pp. 16–17, and §2.2, p. 18](https://artowen.su.domains/mc/Ch-intro.pdf#page=16): unbiased iid means, exact variance and root mean squared error, plus estimating the variance from a single run with denominator $N-1$. Pages 19–20 make normal-approximation intervals an additional step.
- [MIT 18.05 Reading 6b, §5.2, p. 7](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class06-prep-b.pdf#page=7): independently states the mean, variance, and standard deviation of an iid average before the central limit approximation. The covariance expansion follows from [[和的方差协方差项|variance of a sum]]; both numerical examples and the copied-output boundary are calculated directly.
<!-- bilingual-en:end -->

- Owen 第 16–18 页核对 iid 均值的无偏性、精确方差、均方根误差及分母 $N-1$ 的方差估计；第 19–20 页区分这些矩结果与另外的区间近似步骤。
- MIT Reading 6b 第 7 页 §5.2 独立核对 iid 均值的均值、方差与标准差。协方差展开来自求和方差；两个数值例子与复制结果的边界直接计算。

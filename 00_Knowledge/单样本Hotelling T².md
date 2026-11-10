---
aliases:
  - "One-sample Hotelling T-squared test"
  - "单样本 Hotelling T² 检验"
  - "单总体 Hotelling 均值检验"
student_os: knowledge-atom
atom_id: STAT-HOT-001
atom_set: hotelling-mean-inference
atom_type: theorem-procedure
status: source-checked
requires:
  - "[[样本协方差Wishart律]]"
  - "[[正态均值协方差独立]]"
  - "[[Wishart抽样假设]]"
part_of:
  - "[[Hotelling T² 与多元均值推断.canvas]]"
implies:
  - "[[Hotelling投影极值]]"
  - "[[Hotelling置信椭球]]"
related:
  - "[[线性约束Hotelling T²]]"
  - "[[配对Hotelling T²]]"
  - "[[pooled Hotelling T²]]"
---

<!-- bilingual-en:start -->
# One-sample Hotelling T-squared
<!-- bilingual-en:end -->

*单样本 Hotelling T² 检验*

<!-- bilingual-en:start -->
Suppose each manufactured item is measured for two properties, and we want to check whether their **population means jointly equal a specified target**. A sample mean will rarely equal that target exactly. Hotelling's $T^2$ asks whether the observed departure is large relative to the uncertainty in estimating the mean, accounting for how the properties vary together. The population covariance is unknown and is estimated from the same sample.
<!-- bilingual-en:end -->

例如，每个产品都测量两项指标，我们想检验这两个指标的**总体均值是否共同等于指定目标**。样本均值几乎不会恰好等于目标。Hotelling $T^2$ 衡量的是：考虑均值估计的随机误差以及指标之间的共同变化后，当前偏离是否足够大。总体协方差未知，需要用同一批样本估计。

<!-- bilingual-en:start -->
Let $X_i$ be the $p$-coordinate column vector for observation $i$, with $n\ge2$ observations in total and $p\ge1$. The [[样本均值向量|sample mean]] and the [[样本协方差矩阵|sample covariance]] are
$$\bar X=\frac1n\sum_{i=1}^nX_i,\qquad S=\frac1{n-1}\sum_{i=1}^n(X_i-\bar X)(X_i-\bar X)^T.$$
The mean vector $\mu$ describes the population; $\mu_0$ is the fixed target chosen for the test. The null hypothesis $H_0:\mu=\mu_0$ says that **all** its coordinates equal their targets. The alternative $\mu\ne\mu_0$ means at least one differs, without specifying a sign or a particular coordinate.
<!-- bilingual-en:end -->

$X_i$ 表示第 $i$ 次观测的 $p$ 维列向量，共有 $n\ge2$ 次观测，且 $p\ge1$。上式分别定义[[样本均值向量]]与[[样本协方差矩阵]]，注意 $S$ 使用分母 $n-1$。$\mu$ 是总体均值向量，$\mu_0$ 是检验所指定的固定目标。原假设 $H_0:\mu=\mu_0$ 要求**全部**分量等于各自目标；备择 $\mu\ne\mu_0$ 表示至少一个不同，不预先限定是哪个分量或偏离方向。

<!-- bilingual-en:start -->
Write $d=\bar X-\mu_0$. The estimated covariance of the sample mean is $S/n$, rather than $S$, because averaging reduces sampling variability; see [[样本均值协方差|sample-mean covariance]]. With $S$ positive definite,
$$T^2=d^T(S/n)^{-1}d=n\,d^TS^{-1}d.$$
This is a [[马氏距离|squared Mahalanobis distance]] measured on the mean's uncertainty scale. It adjusts for both units and correlations. A departure along a direction with little sample variation contributes more than the same departure along a noisy direction.
<!-- bilingual-en:end -->

记 $d=\bar X-\mu_0$。均值估计的协方差用 $S/n$ 估计，而不是 $S$，因为取平均会减少抽样变动，见[[样本均值协方差]]。$S$ 正定时，$T^2$ 由上式定义：它是在均值不确定性尺度上的[[马氏距离|平方 Mahalanobis 距离]]，同时调整单位和相关性。沿样本变动很小的方向偏离，比沿噪声很大的方向偏离贡献更多。

<!-- bilingual-en:start -->
## The exact reference distribution
<!-- bilingual-en:end -->

*精确参考分布*

<!-- bilingual-en:start -->
For the finite-sample theorem, assume [[独立同分布|iid]] observations from one [[联合Gaussian|joint Gaussian]] population,
$$X_1,\ldots,X_n\overset{\mathrm{iid}}{\sim}N_p(\mu,\Sigma),\qquad \Sigma\succ0,\qquad n>p\ge1.$$
Here $\Sigma\succ0$ means a [[正定矩阵|positive-definite covariance]]: every nonzero direction has positive population variance. Independence is between observations; coordinates within an observation may be correlated. These conditions make $S$ invertible almost surely.
<!-- bilingual-en:end -->

有限样本定理要求来自同一个[[联合Gaussian|联合 Gaussian]]总体的[[独立同分布|iid]]观测，且 $\Sigma\succ0$、$n>p\ge1$。[[正定矩阵|协方差正定]]意味着每个非零方向都有正的总体方差。这里要求不同观测之间独立，一次观测内部的坐标可以相关。这些条件保证 $S$ 几乎必然可逆。

> [!important] Convert the statistic before comparing it · 先转换，再比较
> <!-- bilingual-en:start -->
> Under $H_0$,
> $$F_{\mathrm{obs}}=\frac{n-p}{p(n-1)}T^2\ \sim\ F_{p,n-p}.$$
> The [[F分布|F distribution]] $F_{r,s}$ is the law of $(U/r)/(V/s)$ for independent $U\sim\chi_r^2$ and $V\sim\chi_s^2$. Its two degrees of freedom have different roles. Here they are **$p$ and $n-p$**, not $p$ and $n-1$.
> <!-- bilingual-en:end -->
>
> 在原假设下，将 $T^2$ 乘以 $(n-p)/(p(n-1))$，得到 $F_{p,n-p}$ 分布。[[F分布]] $F_{r,s}$ 来自两个独立卡方量各除以自由度后的比值 $(U/r)/(V/s)$。分子和分母自由度各有职责；本检验是 **$p$ 与 $n-p$**，不能把第二个写成 $n-1$。

<!-- bilingual-en:start -->
Choose a significance level $0<\alpha<1$ before inspecting the test results. It is the probability of rejecting a true null under this model. If $F_{p,n-p}(1-\alpha)$ denotes the $(1-\alpha)$ [[总体分位数|quantile]], reject when
$$F_{\mathrm{obs}}>F_{p,n-p}(1-\alpha).$$
Equivalently compare $T^2$ with $c_\alpha=p(n-1)F_{p,n-p}(1-\alpha)/(n-p)$. A large squared departure is evidence against the target vector; not rejecting does not establish that the target is true.
<!-- bilingual-en:end -->

在查看检验结果前选定显著性水平 $0<\alpha<1$；在模型成立且原假设为真时，它控制错误拒绝的概率。用 $F_{p,n-p}(1-\alpha)$ 表示相应的 $(1-\alpha)$ [[总体分位数|分位数]]，超过它就拒绝；也可以直接把 $T^2$ 与 $c_\alpha=p(n-1)F_{p,n-p}(1-\alpha)/(n-p)$ 比较。较大的平方偏离提供反对目标向量的证据；没有拒绝并不证明目标正确。

> [!example] Two responses, ten observations · 两项指标，十次观测
> <!-- bilingual-en:start -->
> Suppose $n=10$, $p=2$, $\bar x=(1,1)^T$, $\mu_0=(0,0)^T$, and
> $$S=\begin{bmatrix}1&0\\0&4\end{bmatrix}.$$
> Then $T^2=10(1^2/1+1^2/4)=12.5$ and $F_{\mathrm{obs}}=\frac8{18}(12.5)=50/9\approx5.556$. Since $F_{2,8}(0.95)\approx4.459$, the 5% test rejects the target vector. This is one joint conclusion; it does not say that both coordinates separately pass a significance test.
> <!-- bilingual-en:end -->
>
> 给定上述样本均值和协方差，$T^2=12.5$，转换后 $F_{\mathrm{obs}}\approx5.556$。$F_{2,8}$ 的 95% 分位数约为 $4.459$，所以在 5% 水平拒绝目标向量。这是一个联合结论，不等于两个分量分别做检验都显著。

<!-- bilingual-en:start -->
## Why estimating covariance changes the calibration
<!-- bilingual-en:end -->

*为什么估计协方差会改变校准*

<!-- bilingual-en:start -->
If the true positive-definite $\Sigma$ were known, then $n d^T\Sigma^{-1}d\sim\chi_p^2$ under $H_0$, by the [[Gaussian马氏平方律|Gaussian squared-distance law]] applied to the sample mean. Replacing $\Sigma$ with the random $S$ adds uncertainty. The exact result uses both $(n-1)S$'s [[样本协方差Wishart律|Wishart law]] and [[正态均值协方差独立|independence of the mean and covariance]]. Equal moments or an unbiased covariance estimate alone cannot supply that distributional argument.
<!-- bilingual-en:end -->

如果真实正定 $\Sigma$ 已知，原假设下 $n d^T\Sigma^{-1}d\sim\chi_p^2$，这是把[[Gaussian马氏平方律]]用于样本均值。用随机的 $S$ 替换 $\Sigma$ 会增加估计不确定性；精确 F 结果同时利用 $(n-1)S$ 的[[样本协方差Wishart律|Wishart 律]]与[[正态均值协方差独立|均值—协方差独立性]]。相同的矩或协方差估计无偏，均不足以替代这段分布论证。

<!-- bilingual-en:start -->
When $p=1$, write the scalar $S=s^2$. Then $T^2=[(\bar X-\mu_0)/(s/\sqrt n)]^2$, the square of the ordinary one-sample $t$ statistic. Its reference law is $F_{1,n-1}=t_{n-1}^2$. The vector statistic also has a [[Hotelling投影极值|maximum-projection interpretation]], which explains why a correlated combination can be unusual even when the coordinates look unremarkable separately.
<!-- bilingual-en:end -->

当 $p=1$ 时，把标量 $S$ 写成 $s^2$，$T^2$ 就是普通单样本 $t$ 统计量的平方，参考分布为 $F_{1,n-1}=t_{n-1}^2$。多元统计量还具有[[Hotelling投影极值|所有投影中最大偏离的解释]]，因此某个相关组合可能很异常，尽管单看每个坐标都不突出。

<!-- bilingual-en:start -->
The likelihood-ratio route gives the same rejection ordering. Let $\Lambda$ be the maximized Gaussian likelihood with $\mu=\mu_0$ imposed, divided by the maximized likelihood with $\mu$ unrestricted; both optimize over $\Sigma\succ0$. For full-rank sample scatter,
$$\Lambda^{2/n}=\left(1+\frac{T^2}{n-1}\right)^{-1}.$$
Thus larger $T^2$ means a smaller relative fit for the constrained model. This explains another origin of the statistic; the reference distribution still needs its sampling assumptions.
<!-- bilingual-en:end -->

似然比也给出相同的拒绝顺序。$\Lambda$ 是“固定 $\mu=\mu_0$ 时的最大 Gaussian 似然”除以“允许 $\mu$ 自由变化时的最大似然”，两者都在 $\Sigma\succ0$ 中优化。样本散布满秩时满足上式。因此 $T^2$ 越大，约束模型相对最优无约束模型的拟合越差。这说明统计量的另一种来源，但参考分布仍需抽样条件。

> [!warning] An inverse and an exact law are separate requirements · 可求逆与精确分布是两件事
> <!-- bilingual-en:start -->
> If $n\le p$, the centered sample covariance is singular; substituting a pseudoinverse does not preserve this F theorem. See [[Wishart秩与可逆性|the rank boundary]]. Non-Gaussian or dependent observations also need a new justification even when $S$ is invertible. Under the null, fixed $p$, iid observations, finite second moments and positive-definite population covariance instead give the asymptotic result $T^2\Rightarrow\chi_p^2$. This uses the [[经典中心极限定理|central limit theorem]] for each fixed linear combination and [[Slutsky定理|consistent covariance substitution]], not a finite-sample identity. Consistency makes $S$ invertible with probability tending to one; assigning any fixed value to $T^2$ on the remaining singular-sample events does not change this limit. It gives no universal sample-size threshold and does not cover growing dimension or arbitrary dependence.
> <!-- bilingual-en:end -->
>
> 若 $n\le p$，中心化样本协方差必奇异，换成伪逆不会保留本卡的 F 定理，见[[Wishart秩与可逆性|秩边界]]。即使 $S$ 可逆，非 Gaussian 或依赖观测也需要另建依据。另一条结论是：原假设下，固定 $p$、iid、有限二阶矩且总体协方差正定时，$T^2\Rightarrow\chi_p^2$。它把[[经典中心极限定理]]用于每个固定线性组合，再通过[[Slutsky定理|一致协方差代入]]得到渐近结论，不是有限样本等式。一致性使 $S$ 可逆的概率趋于 1；在其余奇异样本事件上把 $T^2$ 定义为任意固定值，不改变此极限。该近似既无通用的“大样本”门槛，也不覆盖维数增长或任意依赖。

## 来源与核验

<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 7, §§7.1.1–7.1.4](https://online.stat.psu.edu/stat505/Lesson07): iid multivariate-normal sampling, the one-sample statistic, exact F conversion, known-covariance comparison and interpretation of a joint rejection.
- [Peter Hoff, Multivariate Analysis notes, PDF pp.164–166](https://www2.stat.duke.edu/~pdh10/Teaching/832/Materials/notes.pdf): known-covariance chi-square statistic, independent Gaussian/Wishart pivot and its F scaling. The source uses $S$ for unnormalized scatter.
- [NIST/SEMATECH, §6.5.4.3](https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc543.htm): scalar $t^2$ connection and the factor $p(n-1)/(n-p)$.
- [Stanford STATS305C, One sample problem, “Back to test” and “Hotelling's $T^2$”](https://web.stanford.edu/class/stats305c/lectures/Onesample.html): likelihood-ratio ordering and independent Gaussian/Wishart representation.
- [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.2. 均值向量的假设检验|Course §1.2]], [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.4. T² 与似然比检验|§1.4]] and [[01_Math/04_多元统计分析/05_ 总体平均向量的推论.md#1.7. 大样本近似|§1.7]]: notation, likelihood ratio and the fixed-dimensional approximation. The numerical example is calculated directly.
<!-- bilingual-en:end -->

- Penn State 核对独立多元正态抽样、统计量与 F 转换、已知协方差的区别及联合拒绝的含义。
- Hoff PDF 第164–166页核对已知协方差的卡方量、独立 Gaussian/Wishart 枢轴及其 F 缩放；原源 $S$ 是未归一化散布。
- NIST 核对一元 $t^2$ 特例与缩放系数；Stanford 核对似然比顺序及 Gaussian/Wishart 表示。
- 课程 §§1.2、1.4、1.7 核对记号、似然比与固定维数近似；数值例子由正文公式直接计算。
